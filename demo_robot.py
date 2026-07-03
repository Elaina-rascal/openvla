"""
OpenVLA + robosuite (Panda) 闭环控制 Demo
=========================================
流程: 仿真图像 → VLA 推理 → 执行动作 → 仿真图像 → ...

用法:
    python demo_robot.py                                          # 默认指令
    python demo_robot.py --instruction "pick up the red cube"     # 自定义指令
    python demo_robot.py --interactive                            # 运行时输入指令
"""

import argparse
import os
import time
from dataclasses import dataclass

# ── 强制禁用 TensorFlow（否则 transformers 4.40 会因 numpy API 版本不匹配崩溃）──
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
from PIL import Image

# ── VLA 模型 ──────────────────────────────────────────────────
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# ── robosuite ──────────────────────────────────────────────────
import robosuite as suite
from robosuite.controllers import load_controller_config

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

LOCAL_MODEL_PATH = os.path.abspath("./openvla-7b")        # 本地模型目录
UNNORM_KEY = "bridge_orig"                                 # 数据集反归一化 key
CONTROL_FREQ = 20                                          # 控制频率 (Hz)
MAX_STEPS = 200                                            # 每轮最大步数

# ═══════════════════════════════════════════════════════════════
#  VLA 模型加载
# ═══════════════════════════════════════════════════════════════

def build_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def load_vla():
    """加载 4bit 量化 OpenVLA 模型到 GPU"""
    print("[VLA] Loading processor...")
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

    print("[VLA] Loading 4-bit model...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        LOCAL_MODEL_PATH,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    print("[VLA] ✅ Model loaded on GPU\n")
    return model, processor


@torch.inference_mode()
def predict_action(model, processor, image: Image.Image, instruction: str) -> np.ndarray:
    """VLA 推理 → 返回 7-DOF 动作 [x, y, z, roll, pitch, yaw, gripper]"""
    prompt = build_prompt(instruction)
    inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    return action.flatten()


# ═══════════════════════════════════════════════════════════════
#  robosuite 环境
# ═══════════════════════════════════════════════════════════════

def create_env(render: bool = True):
    """
    创建 Panda Lift 环境

    控制器: OSC_POSE (增量末端位姿控制)
    动作空间: [dx, dy, dz, droll, dpitch, dyaw, gripper] ∈ [-1, 1]
    """
    controller = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        "Lift",
        robots="Panda",
        controller_configs=controller,
        has_renderer=render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=256,
        camera_widths=256,
        control_freq=CONTROL_FREQ,
    )
    return env


# ═══════════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════════

def run_episode(model, processor, env, instruction: str, max_steps: int, render: bool = True):
    """运行一轮闭环控制"""
    obs = env.reset()
    episode_reward = 0.0

    for step in range(max_steps):
        # ── 1. 获取仿真图像 ──
        img = obs["agentview_image"]          # (256, 256, 3), uint8, RGB
        pil_img = Image.fromarray(img)

        # ── 2. VLA 推理 ──
        t0 = time.time()
        action_7dof = predict_action(model, processor, pil_img, instruction)
        inference_time = time.time() - t0

        # ── 3. 映射到环境动作 ──
        #   OSC_POSE: [dx,dy,dz, droll,dpitch,dyaw, gripper] ∈ [-1, 1]
        #   OpenVLA 输出也是 7-DOF，和 OSC_POSE 动作空间对齐
        env_action = np.clip(action_7dof, -1.0, 1.0).astype(np.float64)

        # ── 4. 执行 ──
        obs, reward, done, info = env.step(env_action)
        episode_reward += reward

        # ── 5. 渲染 ──
        if render:
            env.render()

        # 日志（每 10 步）
        if step % 10 == 0:
            print(
                f"  step {step:3d} | "
                f"action=[{action_7dof[0]:+.3f} {action_7dof[1]:+.3f} {action_7dof[2]:+.3f} "
                f"{action_7dof[3]:+.3f} {action_7dof[4]:+.3f} {action_7dof[5]:+.3f} "
                f"{action_7dof[6]:+.3f}] | "
                f"infer={inference_time:.2f}s"
            )

        if done:
            print(f"  ✅ Episode done! reward={episode_reward:.2f}")
            break

    return episode_reward


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OpenVLA + robosuite 闭环控制")
    parser.add_argument("--instruction", default=None, help="初始指令")
    parser.add_argument("--interactive", action="store_true", help="交互模式，每轮可输入新指令")
    parser.add_argument("--no-render", action="store_true", help="不显示仿真窗口")
    args = parser.parse_args()

    # 加载 VLA
    model, processor = load_vla()

    # 创建环境
    env = create_env(render=not args.no_render)
    print(f"[Env] robosuite Lift + Panda (OSC_POSE, {CONTROL_FREQ}Hz)")
    print(f"[Env] Action dim: {env.action_spec[0].shape[0]}")
    print()

    # 初始指令
    instruction = args.instruction or "pick up the red cube"
    print(f"📝 初始指令: \"{instruction}\"")
    if args.interactive:
        print("💬 交互模式已开启 — 每轮结束后可输入新指令\n")

    episode = 0
    try:
        while True:
            episode += 1
            print(f"\n{'='*50}")
            print(f"  Episode {episode}")
            print(f"  Instruction: \"{instruction}\"")
            print(f"{'='*50}")

            reward = run_episode(model, processor, env, instruction, MAX_STEPS, render=not args.no_render)

            print(f"\n📊 Episode {episode} 完成, 累计奖励: {reward:.2f}")

            # 交互模式：输入新指令
            if args.interactive:
                try:
                    new_inst = input("\n输入新指令 (留空退出, 输入 'quit' 退出): ").strip()
                    if not new_inst or new_inst.lower() == "quit":
                        break
                    instruction = new_inst
                    print(f"📝 新指令: \"{instruction}\"\n")
                except (EOFError, KeyboardInterrupt):
                    break
            else:
                # 非交互模式：只跑一轮
                break

    except KeyboardInterrupt:
        print("\n\n⏹ 用户中断")
    finally:
        env.close()
        print("👋 已退出")


if __name__ == "__main__":
    main()
