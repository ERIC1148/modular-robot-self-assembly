"""
单个模块的对接状态机 —— 把论文 Sec 5.3 的三段式串成可执行流程。

    navigate（导航到停车位后方）
        ↓ 到位
    adjust（位姿调整，式(5)(6)）
        ↓ 对准
    approach（直线逼近，推入对接）
        ↓ 满足接受域条件
    docked

外加一个极简的 Wait 避让（论文 5.3.1 引 Binder et al. 2019 的三策略之一）。
论文的完整方案是多车规划器 + Wait/Avoid/Push + 重调度；本项目阶段一
只实现最保守的 Wait —— 前方有其他模块挡路就原地等，等它走开再继续。
"""

from __future__ import annotations

import numpy as np

from planner import controller as ctl
from planner.faces import W, Face, compose, mate_relpose, wrap_angle
from planner.routing import GridRouter

# 逼近起点相对目标位姿沿 -x' 退让的距离
STANDOFF = 0.075

# Wait 避让参数
AVOID_RADIUS = 0.13  # 前方多近算挡路 (m)
AVOID_CONE = np.deg2rad(50)  # 只看正前方这个张角内的模块

# 栅格路由参数
REPLAN_DT = 1.5  # 重规划间隔 (s)。装配体在长大，路径要跟着更新
WP_TOL = 0.05  # 到达航点的判据 (m)


class ModuleTask:
    """执行单个装配动作 (mi, ci, mj, cj) 的状态机。"""

    def __init__(
        self,
        bridge,
        mi: int,
        ci: Face,
        mj: int,
        cj: Face,
        router: GridRouter | None = None,
    ):
        self.br = bridge
        self.mi, self.ci, self.mj, self.cj = mi, ci, mj, cj
        self.state = "navigate"
        self.log: list[tuple[str, float]] = []
        self.waited = 0.0
        self.router = router
        self.path: list[np.ndarray] = []
        self._t_replan = -1e9
        self.replans = 0

    # ------------------------------------------------------------------
    def _replan(self, pose: np.ndarray) -> None:
        """
        在栅格路由图上重规划到逼近起点的路径（论文 5.3.1）。
        障碍 = 除自己以外的所有模块（含已装好的装配体）。
        逼近起点必然贴着装配体，所以用 free_around 豁免它周围的膨胀区。
        """
        target = self.standoff_xy()
        obstacles = [
            self.br.get_pose(k)[:2]
            for k in range(self.br.n)
            if k != self.mi
        ]
        blocked = self.router.blocked_cells(obstacles, free_around=target)
        self.path = self.router.plan(pose[:2], target, blocked)
        self.replans += 1

    def _next_waypoint(self, pose: np.ndarray, t: float) -> np.ndarray:
        if self.router is None:
            return self.standoff_xy()
        if not self.path or t - self._t_replan > REPLAN_DT:
            self._replan(pose)
            self._t_replan = t
        while self.path and np.linalg.norm(self.path[0] - pose[:2]) < WP_TOL:
            self.path.pop(0)
        return self.path[0] if self.path else self.standoff_xy()

    # ------------------------------------------------------------------
    def goal_pose(self) -> np.ndarray:
        """
        目标停车位 —— 由父模块的**当前**位姿实时推出（论文式(2)）。

        用当前位姿而不是预先算好的静态目标，是为了容忍论文提到的那个现象:
        根模块没固定在地上，第一批对接时可能被推动一点点。
        实时重算父模块位姿，后续批次就能自动跟上。
        """
        return compose(self.br.get_pose(self.mj), mate_relpose(self.cj, self.ci))

    def standoff_xy(self) -> np.ndarray:
        g = self.goal_pose()
        return g[:2] - STANDOFF * np.array([np.cos(g[2]), np.sin(g[2])])

    # ------------------------------------------------------------------
    def _blocked(self, pose: np.ndarray, others: list[int]) -> bool:
        """Wait 策略：正前方锥形区域内有其他模块就判定挡路。"""
        for k in others:
            if k in (self.mi, self.mj):
                continue
            d = self.br.get_pose(k)[:2] - pose[:2]
            dist = float(np.linalg.norm(d))
            if dist > AVOID_RADIUS:
                continue
            bearing = wrap_angle(np.arctan2(d[1], d[0]) - pose[2])
            if abs(bearing) < AVOID_CONE:
                return True
        return False

    def _to(self, s: str, t: float) -> None:
        self.log.append((s, round(t, 2)))
        self.state = s

    # ------------------------------------------------------------------
    def update(self, t: float, others: list[int] | None = None) -> None:
        if self.state == "docked":
            return
        pose = self.br.get_pose(self.mi)
        goal = self.goal_pose()

        if self.state == "navigate":
            # 只在导航阶段避让；位姿调整和逼近阶段已经贴近目标，不再让路
            if others and self._blocked(pose, others):
                self.br.set_color(self.mi, "idle")
                self.br.stop(self.mi)
                self.waited += 1
                return
            self.br.set_color(self.mi, "navigate")
            wp = self._next_waypoint(pose, t)
            is_last = not self.path or len(self.path) <= 1
            v, w, reached = ctl.navigate(pose, wp)
            # 只有走到最后一个航点（= 逼近起点）才算导航完成
            if reached and is_last:
                self._to("adjust", t)

        elif self.state == "adjust":
            self.br.set_color(self.mi, "adjust")
            v, w, done = ctl.adjust(pose, goal, self.ci)
            if done:
                self._to("approach", t)

        else:  # approach
            self.br.set_color(self.mi, "approach")
            v, w, _ = ctl.approach(pose, goal)
            if self.br.try_dock(self.mi, self.ci, self.mj, self.cj):
                self.br.stop(self.mi)
                self._to("docked", t)
                return

        self.br.set_wheel_velocity(self.mi, v, w)

    @property
    def done(self) -> bool:
        return self.state == "docked"

    def residual(self) -> tuple[float, float]:
        return self.br.dock_error(self.mi, self.ci, self.mj, self.cj)
