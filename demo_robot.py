"""
OpenVLA + robosuite (Panda) 闭环控制 Demo
=========================================
对标 OpenVLA 官方 run_libero_eval.py 的关键做法：
  - 图像 224×224
  - Gripper: 二值化 → [-1,+1] → 取反 (open=-1, close=+1)
  - 动作映射: VLA 物理值 → robosuite OSC_POSE [-1,1] 范围

用法:
    python demo_robot.py
    python demo_robot.py --instruction "pick up the red cube"
    python demo_robot.py --interactive
"""

import argparse, os, time

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np, torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import robosuite as suite
from robosuite.controllers import load_controller_config

MODEL_PATH = os.path.abspath("./openvla-7b")
UNNORM_KEY = "bridge_orig"
CONTROL_FREQ = 20
MAX_STEPS = 500
IMG_SIZE = 224       # ★ 官方 OpenVLA 用的是 224

# ═══════════════════════════════════════════════════════════════
#  VLA
# ═══════════════════════════════════════════════════════════════

def build_prompt(ins: str) -> str:
    return f"In: What action should the robot take to {ins.lower()}?\nOut:"

def load_vla():
    print("[VLA] Loading 4-bit model from local...")
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True)
    print("[VLA] OK\n")
    return model, proc

@torch.inference_mode()
def predict_action(model, proc, img: Image.Image, ins: str):
    inputs = proc(build_prompt(ins), img).to("cuda", dtype=torch.bfloat16)
    a = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    if isinstance(a, torch.Tensor): a = a.cpu().numpy()
    return a.flatten()

# ═══════════════════════════════════════════════════════════════
#  env
# ═══════════════════════════════════════════════════════════════

def create_env(render=True):
    """OSC_POSE: 输入 [-1,1] 映射到物理 [±0.05m, ±0.5rad]"""
    return suite.make(
        "Lift", robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=render, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview"],
        camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
        control_freq=CONTROL_FREQ)

# ═══════════════════════════════════════════════════════════════
#  动作映射 + Gripper 处理
# ═══════════════════════════════════════════════════════════════

def vla_to_env(action_7dof: np.ndarray) -> np.ndarray:
    """
    VLA 输出: [dx,dy,dz, droll,dpitch,dyaw, grip {0=close,1=open}]  (物理 m, rad)
    robosuite OSC_POSE 输入: [-1, 1] 归一化
    默认映射: 输入[-1,1] → 物理[0.05m, 0.5rad]
    ∴ 物理值 ÷ 输出范围 = 归一化值
    """
    a = action_7dof.copy()
    a[0:3] /= 0.05   # m → [-1,1]  (×20)
    a[3:6] /= 0.5    # rad → [-1,1] (×2)

    # Gripper: 对标官方做法
    #   normalize_gripper_action: [0,1] → [-1,+1], binarize
    #   invert_gripper_action: × -1 → open=-1, close=+1
    a[6] = 2 * a[6] - 1.0      # [0,1] → [-1,1]
    a[6] = np.sign(a[6])        # 二值化
    a[6] *= -1.0                # 取反

    return np.clip(a, -1.0, 1.0).astype(np.float64)

# ═══════════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════════

def run_episode(model, proc, env, ins: str, max_steps: int, render: bool):
    obs = env.reset()
    ep_r = 0.0
    for step in range(max_steps):
        img = obs["agentview_image"]
        pil=img[::-1,::-1]  # 224×224, uint8, 旋转180度对标训练
        pil=Image.fromarray(pil)
        # pil = Image.fromarray(obs["agentview_image"])  # 224×224, uint8
        t0 = time.time()
        raw = predict_action(model, proc, pil, ins)
        dt = time.time() - t0
        act = vla_to_env(raw)
        obs, reward, done, info = env.step(act)
        ep_r += reward
        if render: env.render()
        if step % 10 == 0:
            print(f"  step {step:3d} | raw=[{raw[0]:+.3f} {raw[1]:+.3f} {raw[2]:+.3f} "
                  f"{raw[3]:+.3f} {raw[4]:+.3f} {raw[5]:+.3f} {raw[6]:+.3f}] | "
                  f"cmd=[{act[0]:+.3f} {act[1]:+.3f} {act[2]:+.3f} "
                  f"{act[3]:+.3f} {act[4]:+.3f} {act[5]:+.3f} {act[6]:+.1f}] | {dt:.2f}s")
        if done: print(f"  ✅ done, reward={ep_r:.2f}"); break
    return ep_r

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    model, proc = load_vla()
    env = create_env(render=not args.no_render)
    print(f"[Env] Lift+Panda OSC_POSE | {IMG_SIZE}×{IMG_SIZE} | {CONTROL_FREQ}Hz\n")

    ins = args.instruction or "pick up the red cube"
    print(f'📝 "{ins}"')
    if args.interactive: print("💬 交互模式\n")

    ep = 0
    try:
        while True:
            ep += 1
            print(f"\n{'='*50}\n  Ep {ep}  \"{ins}\"\n{'='*50}")
            r = run_episode(model, proc, env, ins, MAX_STEPS, not args.no_render)
            print(f"\n📊 Ep {ep} reward={r:.2f}")
            if args.interactive:
                ni = input("\n新指令 (回车退出): ").strip()
                if not ni or ni == "quit": break
                ins = ni
            else: break
    except KeyboardInterrupt: print("\n⏹ 中断")
    finally: env.close(); print("👋")

if __name__ == "__main__": main()
