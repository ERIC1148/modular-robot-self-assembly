"""
MuJoCo 仿真接口层。

这一层是算法层与仿真引擎之间唯一的边界。算法层（planner/）只允许调用
本文件暴露的接口，不得直接碰 mujoco API —— 这样将来换 Isaac Sim 时，
只需重写本文件，planner/ 一行不动。

三个核心接口（架构约定，阶段一定的）:
    get_pose(module_id)                       -> (x, y, yaw)
    set_wheel_velocity(module_id, v, omega)
    activate_weld(mi, face_i, mj, face_j)

外加 docking 条件判定 try_dock()，对应论文 EP-Face 的 area of acceptance。

阶段二新增（TILT/PAN 手腕关节，见 model_gen.py 顶部的设计说明）:
    get_tilt(i) / get_pan(i)                  -> 当前角度 (rad)
    set_tilt_target(i, angle) / set_pan_target(i, angle)

⚠️ 已知限制: activate_weld() 焊接的是模块的**机体** body("m{i}")，不是手腕
子链末端。这在阶段一（对接时 TILT/PAN 恒为零）没有问题，但如果对接完成后
再驱动**已经用 TOP 面对接的那个模块**自己的 TILT/PAN，会与既有 weld 冲突
（连接口应该保持不动、机体转，但现在 weld 钉死的是机体）。这个限制会在
后续把 weld 挂接点改到手腕末端时解决，目前尚未解决。
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 mujoco：pip install mujoco") from e

from planner.faces import FACE_CHAR, W, Face, mate_relpose, wrap_angle

from .model_gen import STATE_COLORS, WHEEL_R, WHEEL_Y, build_xml

# 轮距（两轮中面间距）
WHEEL_BASE = 2 * WHEEL_Y

# ---- docking 接受域 ----
# 论文 EP-Face 硬件: 法向 4mm / 切向 7mm 的 area of acceptance（模块 80mm）。
# 阶段一放宽到 ~12mm / 12°，提高 demo 鲁棒性；调通后可收紧对齐论文数值。
DOCK_POS_TOL = 0.012  # 两连接面中心距阈值 (m)
DOCK_ANG_TOL = np.deg2rad(12.0)  # 两面外法线反向平行的角度阈值


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_to_yaw(q: np.ndarray) -> float:
    # 平面运动，只取绕 z 的分量
    return wrap_angle(2.0 * np.arctan2(q[3], q[0]))


class MuJoCoBridge:
    """封装一份 MjModel/MjData，并提供算法层需要的全部操作。"""

    def __init__(
        self,
        init_poses: list[tuple[float, float, float]],
        ghost_poses: list[tuple[float, float, float]] | None = None,
    ):
        self.n = len(init_poses)
        self.xml = build_xml(init_poses, ghost_poses)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)

        # ---- 名字 → id 的查表，避免每帧做字符串查找 ----
        M, O = self.model, mujoco.mjtObj
        self._bid = [mujoco.mj_name2id(M, O.mjOBJ_BODY, f"m{i}") for i in range(self.n)]
        self._qadr = [
            M.jnt_qposadr[mujoco.mj_name2id(M, O.mjOBJ_JOINT, f"m{i}_free")]
            for i in range(self.n)
        ]
        self._vadr = [
            M.jnt_dofadr[mujoco.mj_name2id(M, O.mjOBJ_JOINT, f"m{i}_free")]
            for i in range(self.n)
        ]
        self._act = [
            (
                mujoco.mj_name2id(M, O.mjOBJ_ACTUATOR, f"a_m{i}_l"),
                mujoco.mj_name2id(M, O.mjOBJ_ACTUATOR, f"a_m{i}_r"),
            )
            for i in range(self.n)
        ]
        self._tilt_qadr = [
            M.jnt_qposadr[mujoco.mj_name2id(M, O.mjOBJ_JOINT, f"m{i}_tilt")]
            for i in range(self.n)
        ]
        self._pan_qadr = [
            M.jnt_qposadr[mujoco.mj_name2id(M, O.mjOBJ_JOINT, f"m{i}_pan")]
            for i in range(self.n)
        ]
        self._tilt_act = [
            mujoco.mj_name2id(M, O.mjOBJ_ACTUATOR, f"a_m{i}_tilt")
            for i in range(self.n)
        ]
        self._pan_act = [
            mujoco.mj_name2id(M, O.mjOBJ_ACTUATOR, f"a_m{i}_pan")
            for i in range(self.n)
        ]
        self._site = {
            (i, f): mujoco.mj_name2id(M, O.mjOBJ_SITE, f"m{i}_{FACE_CHAR[f]}")
            for i in range(self.n)
            for f in Face
        }
        self._chassis_gid = [
            mujoco.mj_name2id(M, O.mjOBJ_GEOM, f"m{i}_chassis") for i in range(self.n)
        ]
        # 每个模块的全部碰撞 geom（对接后要成组清碰撞位）
        self._geoms = [
            [
                mujoco.mj_name2id(M, O.mjOBJ_GEOM, f"m{i}_{s}")
                for s in ("chassis", "caster_f", "caster_b", "gwl", "gwr")
            ]
            for i in range(self.n)
        ]
        self._eq = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self._eq[(i, j)] = mujoco.mj_name2id(M, O.mjOBJ_EQUALITY, f"w_{i}_{j}")

        # 已建立的连接，元素为 (mi, face_i, mj, face_j)
        self.connections: list[tuple[int, Face, int, Face]] = []

        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # 接口 1: 读位姿
    # ------------------------------------------------------------------
    def get_pose(self, i: int) -> np.ndarray:
        """返回模块 i 的平面位姿 (x, y, yaw)。仿真里是 ground truth，
        对应论文实验中 VICON 动捕提供的 100Hz 位姿反馈。"""
        q = self.data.qpos[self._qadr[i] : self._qadr[i] + 7]
        return np.array([q[0], q[1], quat_to_yaw(q[3:7])])

    def get_all_poses(self) -> np.ndarray:
        return np.array([self.get_pose(i) for i in range(self.n)])

    # ------------------------------------------------------------------
    # 接口 2: 写轮速（差速驱动，论文式(4)）
    # ------------------------------------------------------------------
    def set_wheel_velocity(self, i: int, v: float, omega: float) -> None:
        """
        v     : 体坐标系 x 轴线速度 (m/s)
        omega : 绕 z 轴角速度 (rad/s)

        差速分解: 左轮在 +y，其接地点线速度 = v - omega*b/2
                  右轮在 -y，其接地点线速度 = v + omega*b/2
        再除以轮半径得到轮关节角速度。
        """
        half_b = WHEEL_BASE / 2
        wl = (v - omega * half_b) / WHEEL_R
        wr = (v + omega * half_b) / WHEEL_R
        al, ar = self._act[i]
        self.data.ctrl[al] = wl
        self.data.ctrl[ar] = wr

    def stop(self, i: int) -> None:
        self.set_wheel_velocity(i, 0.0, 0.0)

    def stop_all(self) -> None:
        self.data.ctrl[:] = 0.0

    # ------------------------------------------------------------------
    # 阶段二: TILT / PAN 手腕关节（见 model_gen.py 顶部设计说明）
    # ------------------------------------------------------------------
    def get_tilt(self, i: int) -> float:
        """当前弯折角 (rad)，0 = 水平（与阶段一 TOP 面朝向一致）。"""
        return float(self.data.qpos[self._tilt_qadr[i]])

    def get_pan(self, i: int) -> float:
        """当前扭转角 (rad)，连续量，可能超出 [-π, π]（多圈旋转不会被截断）。"""
        return float(self.data.qpos[self._pan_qadr[i]])

    def set_tilt_target(self, i: int, angle: float) -> None:
        """
        下达 TILT 目标角。论文: TILT DOF 限位 ±90°，这里在写入前主动 clip，
        与 MJCF 里的 joint range 形成双重保险（actuator 的 ctrlrange 已经
        限定了这个范围，clip 是再上一层防御，便于上层代码不必自己记边界值）。
        """
        self.data.ctrl[self._tilt_act[i]] = float(np.clip(angle, -np.pi / 2, np.pi / 2))

    def set_pan_target(self, i: int, angle: float) -> None:
        """下达 PAN 目标角。论文: PAN DOF 连续旋转，不设限位，传入弧度即可。"""
        self.data.ctrl[self._pan_act[i]] = float(angle)

    # ------------------------------------------------------------------
    # 接口 3: 建立连接
    # ------------------------------------------------------------------
    def activate_weld(
        self, mi: int, fi: Face, mj: int, fj: Face, snap: bool = True
    ) -> None:
        """
        把模块 mi 的 fi 面与模块 mj 的 fj 面刚性连接。

        snap=True 时先把 mi 瞬移到理想配合位姿再焊接（阶段一的简化 docking）。
        这样装配体几何始终精确，误差不会逐级累积导致后面的模块对不上 ——
        代价是接受域较宽时会看到一次轻微跳变。
        """
        a, b = (mi, mj) if mi < mj else (mj, mi)
        eq = self._eq[(a, b)]

        # relpose 定义为 body2 在 body1 坐标系下的位姿，body1=m_a, body2=m_b
        if a == mi:
            rel = mate_relpose(fi, fj)  # m_j 相对 m_i
        else:
            rel = mate_relpose(fj, fi)  # m_i 相对 m_j
        rx, ry, ryaw = rel
        rq = yaw_to_quat(ryaw)

        if snap:
            # ⚠️ 必须移动**移动模块 mi**，不能移动父模块 mj。
            # weld 的 body1/body2 是按编号排序声明的，与谁是移动模块无关；
            # 若照着 (a,b) 顺序 snap，当 mi<mj 时会去瞬移已经焊在装配体上的父模块，
            # 与它既有的 weld 打架，装配体被拽变形（实测残差从 0.2mm 恶化到 11mm）。
            self._snap(mj, mi, mate_relpose(fj, fi))

        # eq_data 布局 (mjNEQDATA=11): anchor(3) | relpose_pos(3) | relpose_quat(4) | torquescale(1)
        d = np.zeros(11)
        d[0:3] = 0.0  # anchor 取 body2 原点
        d[3:6] = [rx, ry, 0.0]
        d[6:10] = rq
        d[10] = 1.0
        self.model.eq_data[eq] = d
        self.data.eq_active[eq] = 1

        self._disable_collision_between(mi, mj)
        self.connections.append((mi, fi, mj, fj))
        self.set_color(mi, "docked")

    def _disable_collision_between(self, a: int, b: int) -> None:
        """
        关闭已对接两模块之间的碰撞。

        ⚠️ 这一步不能省。L/R 面就是轮子，对接后邻居机体与轮子恰好贴面，
        微小穿透会让 weld 约束和接触约束往复振荡，能量经地面接触传播出去，
        把场上其他模块一起弹飞（实测远处模块被弹到 1.5 m/s）。
        互相清掉对方的碰撞位即可精确解除这一对，不影响与地面及其他模块的碰撞。
        """
        for ga in self._geoms[a]:
            self.model.geom_conaffinity[ga] &= ~self.model.geom_contype[
                self._geoms[b][0]
            ]
        for gb in self._geoms[b]:
            self.model.geom_conaffinity[gb] &= ~self.model.geom_contype[
                self._geoms[a][0]
            ]

    def _snap(self, a: int, b: int, rel: np.ndarray) -> None:
        """把 b 瞬移到相对 a 的理想配合位姿，并清零两者速度。"""
        xa, ya, yawa = self.get_pose(a)
        rx, ry, ryaw = rel
        c, s = np.cos(yawa), np.sin(yawa)
        xb = xa + c * rx - s * ry
        yb = ya + s * rx + c * ry
        yawb = wrap_angle(yawa + ryaw)

        q = self.data.qpos
        adr = self._qadr[b]
        q[adr + 0] = xb
        q[adr + 1] = yb
        q[adr + 2] = WHEEL_R
        q[adr + 3 : adr + 7] = yaw_to_quat(yawb)
        # 清零速度，避免瞬移产生的冲量把装配体弹飞
        self.data.qvel[self._vadr[b] : self._vadr[b] + 6] = 0.0
        self.data.qvel[self._vadr[a] : self._vadr[a] + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # docking 条件判定 —— 对应论文 EP-Face 的 area of acceptance
    # ------------------------------------------------------------------
    def face_pose(self, i: int, f: Face) -> tuple[np.ndarray, np.ndarray]:
        """返回连接面的世界坐标 (中心位置, 外法线)。site 的 z 轴即外法线。"""
        sid = self._site[(i, f)]
        pos = self.data.site_xpos[sid].copy()
        normal = self.data.site_xmat[sid].reshape(3, 3)[:, 2].copy()
        return pos, normal

    def dock_error(
        self, mi: int, fi: Face, mj: int, fj: Face
    ) -> tuple[float, float]:
        """返回 (面心距离, 法线反向偏差角)。两者都趋于 0 表示对准。"""
        pi, ni = self.face_pose(mi, fi)
        pj, nj = self.face_pose(mj, fj)
        d = float(np.linalg.norm(pi - pj))
        # 理想情况 ni 与 nj 反向 ⇒ dot = -1
        cosang = float(np.clip(-np.dot(ni, nj), -1.0, 1.0))
        return d, float(np.arccos(cosang))

    def can_dock(self, mi: int, fi: Face, mj: int, fj: Face) -> bool:
        d, ang = self.dock_error(mi, fi, mj, fj)
        return d < DOCK_POS_TOL and ang < DOCK_ANG_TOL

    def try_dock(self, mi: int, fi: Face, mj: int, fj: Face) -> bool:
        """满足接受域条件就建立连接，返回是否成功。"""
        if self.can_dock(mi, fi, mj, fj):
            self.activate_weld(mi, fi, mj, fj)
            return True
        return False

    # ------------------------------------------------------------------
    # 可视化辅助
    # ------------------------------------------------------------------
    def set_color(self, i: int, state: str) -> None:
        """按算法状态给模块上色，让并行分层过程一眼可见。"""
        self.model.geom_rgba[self._chassis_gid[i]] = STATE_COLORS[state]

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)
