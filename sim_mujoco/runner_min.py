"""
最小可视化验证 —— 三模块自组装（论文 Fig 4 的规模）。

跑起来能看到:
  1. 差速驱动的模块自主开动
  2. 论文三段式对接: 导航(蓝) → 位姿调整(黄) → 逼近(橙) → 已对接(绿)
  3. 根模块固定不动（红），其余模块向它聚拢
  4. 半透明幽灵框标出展开图算出的「停车位」
  5. 连接面 site 坐标系可视化 —— 青色 = T/B 可直接驱动对接，
     品红 = L/R 轮子面（需 helping module，阶段一回避）

用法:
    mjpython -m sim_mujoco.runner_min           # 开窗口看实时画面
    python   -m sim_mujoco.runner_min --headless # 无头自检，打印时序

⚠️ macOS 上开窗口必须用 mjpython，不能用 python（launch_passive 要求主线程渲染）。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from planner import controller as ctl
from planner.faces import Face, compose, mate_relpose

from .bridge import MuJoCoBridge

# ---- 目标构型 ----
# 根模块 m1 不动；m0 与 m2 分别接到它的 TOP 面和 BOTTOM 面。
# 全部用 T/B 面对接 ⇒ 完全不需要 helping module（论文 5.3.3 情况一）。
ROOT = 1
PLAN = [
    # (移动模块, 它用的面 c_i, 父模块, 父模块的面 c_j)
    (0, Face.TOP, 1, Face.TOP),
    (2, Face.TOP, 1, Face.BOTTOM),
]

INIT_POSES = [
    (0.34, 0.26, -2.4),  # m0
    (0.00, 0.00, 0.30),  # m1 根模块
    (-0.30, -0.28, 0.9),  # m2
]

STANDOFF = 0.07  # 逼近起点相对目标位姿沿 -x' 退让的距离 (m)


class ModuleTask:
    """单个模块的三段式状态机。"""

    STATES = ("navigate", "adjust", "approach", "docked")

    def __init__(self, bridge, mi, fi, mj, fj):
        self.br, self.mi, self.fi, self.mj, self.fj = bridge, mi, fi, mj, fj
        self.state = "navigate"
        self.t_enter = 0.0
        self.log = []

    def goal_pose(self):
        """目标停车位: 由父模块当前位姿 + 面配对关系推出（论文式(2)）。"""
        return compose(self.br.get_pose(self.mj), mate_relpose(self.fj, self.fi))

    def standoff_xy(self):
        g = self.goal_pose()
        return g[:2] - STANDOFF * np.array([np.cos(g[2]), np.sin(g[2])])

    def _to(self, s, t):
        self.log.append((s, round(t, 2)))
        self.state, self.t_enter = s, t

    def update(self, t):
        if self.state == "docked":
            return
        pose = self.br.get_pose(self.mi)
        goal = self.goal_pose()

        if self.state == "navigate":
            self.br.set_color(self.mi, "navigate")
            v, w, done = ctl.navigate(pose, self.standoff_xy())
            if done:
                self._to("adjust", t)
        elif self.state == "adjust":
            self.br.set_color(self.mi, "adjust")
            v, w, done = ctl.adjust(pose, goal, self.fi)
            if done:
                self._to("approach", t)
        else:  # approach
            self.br.set_color(self.mi, "approach")
            v, w, _ = ctl.approach(pose, goal)
            if self.br.try_dock(self.mi, self.fi, self.mj, self.fj):
                self.br.stop(self.mi)
                self._to("docked", t)
                return

        self.br.set_wheel_velocity(self.mi, v, w)


def build():
    br = MuJoCoBridge(INIT_POSES)
    br.set_color(ROOT, "root")
    tasks = [ModuleTask(br, *a) for a in PLAN]
    return br, tasks


def run(headless: bool, max_time: float = 40.0):
    br, tasks = build()
    dt = br.model.opt.timestep

    def tick():
        for tk in tasks:
            tk.update(br.time)
        br.step()

    if headless:
        while br.time < max_time and any(t.state != "docked" for t in tasks):
            tick()
    else:
        import mujoco
        import mujoco.viewer

        with mujoco.viewer.launch_passive(br.model, br.data) as viewer:
            # 显示 site 坐标系 —— 位姿调整对得准不准，一眼能看出来
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.cam.azimuth, viewer.cam.elevation = 140, -40
            viewer.cam.distance, viewer.cam.lookat[:] = 1.3, [0, 0, 0.05]
            while viewer.is_running() and br.time < max_time:
                t0 = time.time()
                with viewer.lock():
                    tick()
                viewer.sync()
                lag = dt - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)

    # ---- 时序报告 ----
    print(f"\n仿真时间 {br.time:.2f}s，已建立连接 {len(br.connections)}/{len(PLAN)}")
    for tk in tasks:
        seq = " → ".join(f"{s}@{t}s" for s, t in tk.log)
        print(f"  m{tk.mi} [{tk.state:8s}] {seq}")
        if tk.state == "docked":
            d, a = br.dock_error(tk.mi, tk.fi, tk.mj, tk.fj)
            print(f"           对接残差: 面心距 {d*1000:.2f}mm, "
                  f"法线偏差 {np.rad2deg(a):.2f}°")
    return br, tasks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="不开窗口，只跑数值自检")
    args = ap.parse_args()
    br, tasks = run(args.headless)
    ok = all(t.state == "docked" for t in tasks)
    print("\n" + ("✓ 三模块自组装完成" if ok else "✗ 未全部完成"))
