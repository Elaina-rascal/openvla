"""
run_libero_eval.py  —  纯推理版，无 draccus，无 openvla.experiments.* 依赖

用法:
    python libero/run_libero_eval.py \
        --model_path ./openvla-7b \
        --task_suite_name libero_spatial \
        --num_trials_per_task 5
"""

import argparse
import os
import sys
import time
import json

# ★ 必须在 import 之前设置所有环境变量
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 防止 LIBERO fork 后 tokenizer 损坏

sys.path.insert(0, "/openvla/LIBERO")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# ── LIBERO benchmark ──────────────────────────────────────────
from libero.libero import benchmark

# ── 本地 libero utils ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)

# ═══════════════════════════════════════════════════════════════
#  模型加载（与 inference.py 同款 4-bit）
# ═══════════════════════════════════════════════════════════════

def load_model(model_path: str):
    print(f"[*] Loading VLA from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # 加载 norm_stats
    stats_path = os.path.join(model_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            model.norm_stats = json.load(f)
    else:
        print("[!] No dataset_statistics.json found; will fallback to bridge_orig")

    return model, processor


# ═══════════════════════════════════════════════════════════════
#  动作推理
# ═══════════════════════════════════════════════════════════════

@torch.inference_mode()
def get_action(model, processor, obs, task_label, unnorm_key):
    """与 inference.py 完全一致的推理方式"""
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    return action.flatten()


def normalize_gripper_action(action, binarize=True):
    """Gripper [0,1] → [-1,+1]"""
    action = action.copy()
    action[-1] = 2 * (action[-1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[-1] = np.sign(action[-1])
    return action


def invert_gripper_action(action):
    """Flip sign: open=-1, close=+1"""
    action = action.copy()
    action[-1] *= -1.0
    return action


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/openvla/openvla-7b")
    parser.add_argument("--task_suite_name", default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num_trials_per_task", type=int, default=10)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_dir", default="./experiments/logs")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Load model
    model, processor = load_model(args.model_path)

    # Unnorm key: try task_suite, fallback bridge_orig
    unnorm_key = args.task_suite_name
    if unnorm_key not in model.norm_stats:
        print(f"[!] '{unnorm_key}' not in norm_stats, using 'bridge_orig'")
        unnorm_key = "bridge_orig"

    # Logging
    run_id = f"EVAL-{args.task_suite_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, run_id + ".txt")
    log_file = open(log_path, "w")
    print(f"Logging to: {log_path}")

    # LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks = task_suite.n_tasks
    print(f"Task suite: {args.task_suite_name} ({n_tasks} tasks)")

    # Image resize for OpenVLA
    resize_size = (224, 224)

    # Max steps per suite
    max_steps_map = {
        "libero_spatial": 220, "libero_object": 280,
        "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }
    max_steps = max_steps_map[args.task_suite_name]

    total_eps, total_ok = 0, 0

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_desc = get_libero_env(task, "openvla", resolution=256)

        task_eps, task_ok = 0, 0
        for ep in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[ep])

            t = 0
            replay = []
            print(f"\nTask: {task_desc}  |  Ep {ep+1}/{args.num_trials_per_task}")

            while t < max_steps + args.num_steps_wait:
                done = False
                try:
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate([
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                    }

                    action = get_action(model, processor, observation, task_desc, unnorm_key)
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_ok += 1
                        total_ok += 1
                        break
                    t += 1
                except Exception as e:
                    print(f"  [!] Exception: {e}")
                    break

            task_eps += 1
            total_eps += 1

            save_rollout_video(replay, total_eps, success=done, task_description=task_desc, log_file=log_file)

            print(f"  Success: {done}  |  Task SR: {task_ok}/{task_eps}  |  Total SR: {total_ok}/{total_eps}")
            log_file.write(f"Success: {done}  |  #: {total_eps}  |  SR: {total_ok}/{total_eps}\n")
            log_file.flush()

        env.close()

    log_file.write(f"\nFINAL: {total_ok}/{total_eps} = {total_ok/total_eps*100:.1f}%\n")
    log_file.close()
    print(f"\n===== Done: {total_ok}/{total_eps} ({total_ok/total_eps*100:.1f}%) =====")


if __name__ == "__main__":
    main()
