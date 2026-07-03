"""
demo_robot.py — 最简测试：MuJoCo 场景 + IK 移动到目标位置
=========================================================
先跑通机械臂运动，再加 VLA。
"""

import os, sys, time

import mujoco
import mujoco.viewer
import numpy as np

# 从同目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ik_controller import solve_ik

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panda_scene.xml")
EE_BODY_NAME = "hand"
N_ARM_JOINTS = 7  # joint1-7, 不含 gripper


def main():
    # ── 加载场景 ──
    print(f"[MuJoCo] Loading: {SCENE_PATH}")
    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    mj_data = mujoco.MjData(mj_model)

    ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    mujoco.mj_forward(mj_model, mj_data)
    ee_start = mj_data.body(ee_body_id).xpos.copy()
    print(f"[MuJoCo] EE start: ({ee_start[0]:.3f}, {ee_start[1]:.3f}, {ee_start[2]:.3f})")

    # ── 目标点会来回移动 ──
    target_base = ee_start + np.array([0.3, 0.0, -0.1])
    print(f"[Target base]   ({target_base[0]:.3f}, {target_base[1]:.3f}, {target_base[2]:.3f})")
    print(f"目标沿 Y 方向来回摆动，验证闭环跟踪\n")

    # ── Viewer ──
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # ── 持续闭环控制 ──
    step = 0
    try:
        while viewer.is_running():
            mujoco.mj_forward(mj_model, mj_data)

            # 目标沿 Y 方向正弦摆动
            # target_pos = target_base + np.array([0.0, 0.15 * np.sin(step * 0.008), 0.0])
            target_pos = target_base +np.array([step*1e-5,step*1e-5,step*1e-5] )
            ee_pos = mj_data.body(ee_body_id).xpos.copy()
            err = np.linalg.norm(target_pos - ee_pos)

            q_target = solve_ik(mj_model, mj_data, ee_body_id, target_pos, n_arm_joints=N_ARM_JOINTS)
            mj_data.ctrl[:N_ARM_JOINTS] = q_target

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            if step % 100 == 0:
                print(f"  step {step:4d} | target Y={target_pos[1]:+.3f} "
                      f"EE=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) | err={err:.4f}")
            step += 1
    except KeyboardInterrupt:
        pass
    viewer.close()


if __name__ == "__main__":
    main()


