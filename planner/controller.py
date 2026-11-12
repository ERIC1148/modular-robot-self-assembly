"""
对接三段式控制律（论文 Sec 5.3）—— 引擎无关，只吃位姿、吐 (v, omega)。

    导航 navigate  → 位姿调整 adjust → 逼近 approach

阶段一只实现论文 5.3.3 的「情况一」: c_i 为 TOP/BOTTOM 面，靠往前开直接推入对接。
「情况二」(c_i 为 LEFT/RIGHT 面，需 helping module 抬举) 留待阶段二。
"""

from __future__ import annotations

import numpy as np

from .faces import Face, wrap_angle

# ---- 控制参数 ----
V_MAX = 0.18  # 线速度上限 (m/s)
W_MAX = 2.5  # 角速度上限 (rad/s)

K_DIST = 1.2  # 导航: 距离 → 线速度
K_HEAD = 3.0  # 导航: 航向误差 → 角速度
HEAD_GATE = 0.35  # 导航: 航向误差大于此值时先原地转，不前进 (rad)

# 论文式(6) 的增益，实验取 K = diag(2, 1)
K_ADJUST = np.array([2.0, 1.0])

# ⚠️ 式(6) 的矩阵 [[sin θ', 0], [0, 1]] 在 θ'→0 时奇异，求逆会爆掉。
# 论文没有讨论这一点（硬件上靠位置控制轮天然限幅）。仿真里必须显式保护。
SIN_FLOOR = 0.15  # |sin θ'| 的下限（≈8.6°）
THETA_DEADZONE = 0.02  # θ' 小于此值且横向误差仍大时，主动打一点角度破除奇异

ADJUST_POS_TOL = 0.004  # 位姿调整完成判据: 横向误差 (m)
ADJUST_ANG_TOL = np.deg2rad(2.5)  # 位姿调整完成判据: 角度误差 (rad)


def _clip(v, lo, hi):
    return float(np.clip(v, lo, hi))


def to_local(pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """把世界位姿 pose 表达到 frame 的体坐标系下，返回 (x', y', θ')。"""
    d = pose[:2] - frame[:2]
    c, s = np.cos(-frame[2]), np.sin(-frame[2])
    return np.array(
        [c * d[0] - s * d[1], s * d[0] + c * d[1], wrap_angle(pose[2] - frame[2])]
    )


# ----------------------------------------------------------------------
# 阶段 1: 导航（论文 5.3.1，式(4) 的差速车模型）
# ----------------------------------------------------------------------
def navigate(pose: np.ndarray, target_xy: np.ndarray, arrive_tol: float = 0.02):
    """开到 target_xy 附近。返回 (v, omega, done)。"""
    d = target_xy - pose[:2]
    dist = float(np.linalg.norm(d))
    if dist < arrive_tol:
        return 0.0, 0.0, True

    head_err = wrap_angle(np.arctan2(d[1], d[0]) - pose[2])
    omega = _clip(K_HEAD * head_err, -W_MAX, W_MAX)
    # 航向偏差大时先原地转正，避免画大圈
    v = 0.0 if abs(head_err) > HEAD_GATE else _clip(K_DIST * dist, 0.0, V_MAX)
    return v, omega, False


# ----------------------------------------------------------------------
# 阶段 2: 位姿调整（论文 5.3.2，式(5)(6)）
# ----------------------------------------------------------------------
def adjust(pose: np.ndarray, goal_pose: np.ndarray, face: Face):
    """
    把模块调整到与目标位姿 goal_pose 对齐，为逼近做准备。

    论文按 c_i 的类型分两种（Fig 6）:
      - c_i 是 TOP/BOTTOM → 调 y' 和 θ' 到零，放开 x'（逼近时沿 x' 推进消掉）
      - c_i 是 LEFT/RIGHT → 调 x' 和 θ' 到零，放开 y'
    阶段一只走前者。

    返回 (v, omega, done)。
    """
    xl, yl, th = to_local(pose, goal_pose)

    if face in (Face.LEFT, Face.RIGHT):
        # 论文的另一种情况，阶段一不会走到；留出接口
        raise NotImplementedError(
            "c_i 为 LEFT/RIGHT 时需 helping module（论文 5.3.3 情况二），属阶段二"
        )

    lateral = yl  # 被约束的那一维
    if abs(lateral) < ADJUST_POS_TOL and abs(th) < ADJUST_ANG_TOL:
        return 0.0, 0.0, True

    # 式(6): [v; ω] = [[sinθ',0],[0,1]]^-1 · K · [−y'; −θ']
    s = np.sin(th)
    if abs(lateral) > ADJUST_POS_TOL and abs(th) < THETA_DEADZONE:
        # 奇异区且横向误差还没消: 先主动打一点角度，让 sinθ' 离开零点。
        # 打的方向要能减小 y'（往横向误差的反方向偏头）。
        return 0.0, _clip(-np.sign(lateral) * 0.6, -W_MAX, W_MAX), False

    s_safe = np.sign(s) * max(abs(s), SIN_FLOOR) if s != 0.0 else SIN_FLOOR
    v = _clip(K_ADJUST[0] * (-lateral) / s_safe, -V_MAX, V_MAX)
    omega = _clip(K_ADJUST[1] * (-th), -W_MAX, W_MAX)
    return v, omega, False


# ----------------------------------------------------------------------
# 阶段 3: 逼近（论文 5.3.3 情况一）
# ----------------------------------------------------------------------
def approach(pose: np.ndarray, goal_pose: np.ndarray, v_push: float = 0.05):
    """
    沿直线推进完成对接，同时用小增益维持朝向不跑偏
    （论文强调平移过程中必须保持朝向固定，以便磁面正确贴合）。
    """
    _, yl, th = to_local(pose, goal_pose)
    omega = _clip(-1.5 * th - 6.0 * yl, -0.8, 0.8)
    return v_push, omega, False
