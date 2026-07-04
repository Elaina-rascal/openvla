"""
demo_robot.py — MuJoCo + IK + OpenVLA 闭环控制
===============================================
VLA 推理在独立线程运行，不阻塞仿真。
"""

import os, sys, time, json, threading, queue

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mujoco
import mujoco.viewer
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ik_controller import solve_ik

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openvla-7b"))
SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panda_scene.xml")
UNNORM_KEY = "bridge_orig"
EE_BODY_NAME = "hand"
N_ARM_JOINTS = 7
USE_VLA = True  # 关掉就只跑场景+IK，不加载 VLA


# ═══════════════════════════════════════════════════════════════
#  VLA
# ═══════════════════════════════════════════════════════════════

def load_vla():
    print("[VLA] Loading 4-bit model ...")
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    stats_path = os.path.join(MODEL_PATH, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            model.norm_stats = json.load(f)
    print("[VLA] OK\n")
    return model, proc


@torch.inference_mode()
def predict_vla(model, proc, img: np.ndarray, instruction: str):
    image = Image.fromarray(img[::-1, ::-1]).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = proc(prompt, image).to("cuda", dtype=torch.bfloat16)
    a = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    return a.flatten()


def vla_worker(model, proc, img_queue, action_queue, instruction):
    """独立线程：从 img_queue 取最新图像，推理后将动作推入 action_queue"""
    while True:
        img = img_queue.get()
        if img is None:  # 结束信号
            break
        raw = predict_vla(model, proc, img, instruction)
        raw[3:6] = 0.0  # 只用平移 + 抓取，不用旋转
        # 清空旧动作，只保留最新
        while not action_queue.empty():
            action_queue.get_nowait()
        action_queue.put(raw)


# ═══════════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════════

def main():
    instruction = "pick up the red cube"

    # ── 加载场景 ──
    print(f"[MuJoCo] Loading: {SCENE_PATH}")
    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    mj_data = mujoco.MjData(mj_model)
    ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    mujoco.mj_forward(mj_model, mj_data)

    # 初始位姿 —— 让机械臂伸展到视野中央，看得清动作
    init_q = np.array([0.5, -0.4, 0.3, -1.6, 0.0, 1.2, 0.5])

    mj_data.qpos[:N_ARM_JOINTS] = init_q
    mj_data.ctrl[:N_ARM_JOINTS] = init_q
    mujoco.mj_forward(mj_model, mj_data)
    print(f"[MuJoCo] EE: ({mj_data.body(ee_body_id).xpos[0]:.3f}, "
          f"{mj_data.body(ee_body_id).xpos[1]:.3f}, {mj_data.body(ee_body_id).xpos[2]:.3f})")

    # ── 场景 OK 后再加载 VLA（可选）──
    if USE_VLA:
        model_vla, processor = load_vla()
        img_queue = queue.Queue(maxsize=2)
        action_queue = queue.Queue(maxsize=1)
        worker = threading.Thread(target=vla_worker, args=(
            model_vla, processor, img_queue, action_queue, instruction), daemon=True)
        worker.start()
        latest_action = np.zeros(7)
        print(f"[VLA] Instruction: {instruction}\n")
    else:
        print("[VLA] 已禁用，只跑场景\n")

    # ── Viewer + Renderer ──
    renderer = mujoco.Renderer(mj_model, 224, 224)
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    prev_time = mj_data.time
    target_pos = mj_data.body(ee_body_id).xpos.copy()
    try:
        while viewer.is_running():
            # 监听 viewer reset：当仿真时间回跳时，重置机械臂到自定义初始位姿。
            if mj_data.time < prev_time:
                mj_data.qpos[:N_ARM_JOINTS] = init_q
                mj_data.ctrl[:N_ARM_JOINTS] = init_q
                mujoco.mj_forward(mj_model, mj_data)
                target_pos = mj_data.body(ee_body_id).xpos.copy()
            mujoco.mj_forward(mj_model, mj_data)
            ee_pos = mj_data.body(ee_body_id).xpos.copy()
            
            # ── 渲染（主线程）──
            renderer.update_scene(mj_data, camera=cam_id)
            img = renderer.render()

            # ── 送图给 VLA 线程 / 或用零动作作为占位 ──
            if USE_VLA:
                if not img_queue.full():
                    img_queue.put_nowait(img)
                if not action_queue.empty():
                    latest_action = action_queue.get_nowait()
                elif action_queue.empty():
                    latest_action = np.zeros(7)
                delta = latest_action[:3]
                grip = latest_action[6]
            else:
                delta = np.zeros(3)

            # ── 目标 = 当前 + VLA delta ──
            target_pos = target_pos + delta

            # IK → 执行
            q_target = solve_ik(mj_model, mj_data, ee_body_id, target_pos, n_arm_joints=N_ARM_JOINTS)
            mj_data.ctrl[:N_ARM_JOINTS] = q_target

            mujoco.mj_step(mj_model, mj_data)
            prev_time = mj_data.time
            viewer.sync()

    except KeyboardInterrupt:
        print("\n⏹ Interrupted")
    finally:
        if USE_VLA:
            img_queue.put(None)  # 结束 VLA 线程
        viewer.close()
        renderer.close()
        print("👋 Done")


if __name__ == "__main__":
    main()

           


