"""
ik_controller.py — MuJoCo Jacobian Damped Least Squares IK
===========================================================
零额外依赖，只用 numpy + mujoco 自带的 Jacobian API。
"""

import numpy as np
import mujoco


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray = None,
    n_arm_joints: int = 7,
    max_iter: int = 30,
    tol: float = 1e-3,
    damping: float = 0.05,
    step_size: float = 0.5,
) -> np.ndarray:
    """
    Damped Least Squares IK。

    Args:
        model, data: MuJoCo 模型和数据
        body_id:   末端执行器 body ID
        target_pos: 目标位置 (3,)
        target_quat: 目标四元数 (4,) 或 None
        n_arm_joints: 机械臂关节数（不含夹爪）
        max_iter, tol, damping, step_size: IK 参数

    Returns:
        qpos: n_arm_joints 维关节角（arm 部分）
    """
    q = data.qpos[:n_arm_joints].copy()
    for _ in range(max_iter):
        data.qpos[:n_arm_joints] = q
        mujoco.mj_forward(model, data)

        ee_pos = data.body(body_id).xpos.copy()
        pos_err = target_pos - ee_pos

        jac_pos = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac_pos, None, ee_pos, body_id)
        J = jac_pos[:, :n_arm_joints]

        A = J @ J.T + damping**2 * np.eye(3)
        delta = np.linalg.solve(A, pos_err)
        dq = J.T @ delta

        if target_quat is not None:
            ee_mat = data.body(body_id).xmat.copy().reshape(3, 3)
            R_target = np.empty(9)
            mujoco.mju_quat2Mat(R_target, target_quat)
            R_target = R_target.reshape(3, 3)
            R_err = R_target @ ee_mat.T

            ori_err = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])

            jac_rot = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, None, jac_rot, ee_pos, body_id)
            Jr = jac_rot[:, :n_arm_joints]

            A_full = J @ J.T + Jr @ Jr.T + damping**2 * np.eye(3)
            delta_full = np.linalg.solve(A_full, pos_err + ori_err)
            dq = J.T @ delta_full[:3] + Jr.T @ delta_full

        q += step_size * dq
        if np.linalg.norm(pos_err) < tol:
            break

    return q
