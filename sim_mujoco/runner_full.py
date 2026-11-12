"""
完整自组装 —— 把论文 Sec 4–5 的整条规划链接到 MuJoCo 上执行。

        目标构型
           ↓ Definition 1 校验（unfold）
           ↓ Algorithm 1 找根 τ（topology）
           ↓ 式(2) 摊平出停车位（unfold）
           ↓ 式(3) 选实际根 m_τ + 匈牙利分配 f（assignment）
           ↓ Algorithm 2 分层批次（scheduler）
           ↓ 式(4)(5)(6) 三段式对接（controller + statemachine）
        MuJoCo 执行

用法:
    mjpython -m sim_mujoco.runner_full                    # 论文 Task 2 九模块，开窗口
    mjpython -m sim_mujoco.runner_full --task ten         # 十模块（本项目目标）
    python   -m sim_mujoco.runner_full --headless         # 无头自检
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from planner.assignment import TABLE2_POSES, assign
from planner.faces import Face
from planner.scheduler import build_schedule, describe, summary
from planner.topology import PAPER_TASK2, Topology
from planner.routing import GridRouter
from planner.unfold import ascii_layout, validate_definition1

from .bridge import MuJoCoBridge
from .statemachine import ModuleTask

T, L, B, R = Face.TOP, Face.LEFT, Face.BOTTOM, Face.RIGHT

# 十模块构型：在论文 Task 2 的十字上，把 +y 那条臂再加长一节（模块 9）。
# 仍然全部用 T 面对接 ⇒ 不需要 helping module。
TEN_MODULE = Topology(
    10,
    [
        (0, T, 1, L), (5, T, 1, R), (4, T, 1, B), (8, T, 1, T),
        (3, T, 8, B), (2, T, 5, B), (7, T, 4, B), (6, T, 0, B),
        (9, T, 6, B),
    ],
)

# 十模块的初始散落位置：在论文 Table 2 基础上多加一个模块
TEN_POSES = np.vstack([TABLE2_POSES, [[0.480, 0.150, 2.90]]])

TASKS = {
    "task2": ("论文 Task 2 · 九模块全向车", PAPER_TASK2, TABLE2_POSES),
    "ten": ("十模块（Task 2 加长一臂）", TEN_MODULE, TEN_POSES),
}


class AssemblyRunner:
    """按批次执行装配序列。组间串行，组内并行。"""

    def __init__(self, topo: Topology, init_poses: np.ndarray, verbose: bool = True,
                 split_root: bool = True):
        self.topo = topo

        # ---- 规划（全部离线，纯几何+组合优化）----
        rep = validate_definition1(topo)
        if not rep["ok"]:
            raise SystemExit("目标构型不可自组装: " + "; ".join(rep["problems"]))
        self.plan_report = rep

        self.asg = assign(init_poses, topo, tau=rep["root"])
        self.schedule = build_schedule(topo, rep["root"], self.asg["f"], split_root)

        if verbose:
            self._print_plan(init_poses)

        # ---- 建仿真 ----
        # 幽灵框摆在算出来的停车位上，让人直观看到每个模块要去哪
        ghosts = [tuple(g) for g in self.asg["goals"]]
        self.br = MuJoCoBridge([tuple(p) for p in init_poses], ghost_poses=ghosts)
        self.br.set_color(self.asg["m_tau"], "root")

        # 【论文 Sec 5.2 末】根模块一确定，就以它为中心生成方形栅格路由图
        self.router = GridRouter(init_poses[self.asg["m_tau"]][:2])

        self.batch = 0
        self.tasks: list[ModuleTask] = []
        self.batch_log: list[tuple[int, float, float]] = []  # (批次, 起, 止)
        self._t_batch_start = 0.0
        self._load_batch()

    # ------------------------------------------------------------------
    def _print_plan(self, init_poses):
        r, a = self.plan_report, self.asg
        print(f"\n{'='*66}")
        print(f"规划结果  n={self.topo.n}")
        print(f"{'='*66}")
        print(f"  Definition 1 校验     : 通过")
        print(f"  Algorithm 1 目标图根 τ : {r['root']}")
        print(f"  式(3) 实际根模块 m_τ   : {a['m_tau']}"
              f"（质心 {init_poses[:, :2].mean(axis=0).round(4).tolist()}）")
        print(f"  匈牙利分配 f          : {a['f']}")
        print(f"  总移动距离            : {a['total_distance']:.3f} m")
        print(f"  摊平后的停车位布局:")
        print(ascii_layout(self.topo, r["root"], a["target_rel"]))
        s = summary(self.schedule)
        print(f"  Algorithm 2 装配批次:")
        print(describe(self.schedule))
        print(f"    → {s['n_actions']} 动作 / {s['n_groups']} 批，"
              f"最大并行度 {s['max_parallel']}，加速 {s['speedup_vs_serial']:.2f}×")
        if s["n_helping"]:
            print(f"    ⚠ 有 {s['n_helping']} 个动作需要 helping module（阶段一未实现）")
        print(f"{'='*66}\n")

    def _load_batch(self) -> None:
        if self.batch >= len(self.schedule):
            self.tasks = []
            return
        group = self.schedule[self.batch]
        self.tasks = [
            ModuleTask(self.br, a.mi, a.ci, a.mj, a.cj, router=self.router)
            for a in group
        ]
        self._t_batch_start = self.br.time

    # ------------------------------------------------------------------
    def tick(self) -> None:
        if self.tasks:
            moving = [t.mi for t in self.tasks if not t.done]
            for tk in self.tasks:
                tk.update(self.br.time, others=moving)
            if all(t.done for t in self.tasks):
                self.batch_log.append(
                    (self.batch + 1, self._t_batch_start, self.br.time)
                )
                self.batch += 1
                self._load_batch()
        self.br.step()

    @property
    def finished(self) -> bool:
        return self.batch >= len(self.schedule)

    # ------------------------------------------------------------------
    def report(self) -> None:
        print(f"\n{'='*66}")
        print(f"执行结果   仿真时间 {self.br.time:.1f}s   "
              f"连接 {len(self.br.connections)}/{self.topo.n - 1}")
        print(f"{'='*66}")
        for k, t0, t1 in self.batch_log:
            acts = "  ".join(
                f"({a.mi},{a.ci.name[0]},{a.mj},{a.cj.name[0]})"
                for a in self.schedule[k - 1]
            )
            print(f"  第 {k} 批  {t0:6.1f}s → {t1:6.1f}s  "
                  f"(耗时 {t1-t0:5.1f}s)   {acts}")
        if self.finished:
            errs = [
                self.br.dock_error(a.mi, a.ci, a.mj, a.cj)
                for g in self.schedule for a in g
            ]
            d = max(e[0] for e in errs)
            ang = max(e[1] for e in errs)
            print(f"\n  ✓ 装配完成。最大对接残差: "
                  f"面心距 {d*1000:.2f}mm, 法线偏差 {np.rad2deg(ang):.2f}°")
        else:
            cur = self.schedule[self.batch] if self.batch < len(self.schedule) else []
            stuck = [f"m{t.mi}[{t.state}]" for t in self.tasks if not t.done]
            print(f"\n  ✗ 超时，卡在第 {self.batch+1} 批: {' '.join(stuck)}")


def run(task: str, headless: bool, max_time: float):
    name, topo, poses = TASKS[task]
    print(f"\n### {name} ###")
    runner = AssemblyRunner(topo, poses)
    dt = runner.br.model.opt.timestep

    if headless:
        while not runner.finished and runner.br.time < max_time:
            runner.tick()
    else:
        import mujoco
        import mujoco.viewer

        with mujoco.viewer.launch_passive(runner.br.model, runner.br.data) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.cam.azimuth, viewer.cam.elevation = 90, -55
            viewer.cam.distance = 1.6
            viewer.cam.lookat[:] = [0, 0, 0.04]
            while viewer.is_running() and not runner.finished \
                    and runner.br.time < max_time:
                t0 = time.time()
                with viewer.lock():
                    runner.tick()
                viewer.sync()
                lag = dt - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
            runner.report()
            # 装完后停住让人看成品
            while viewer.is_running():
                with viewer.lock():
                    runner.br.stop_all()
                    runner.br.step()
                viewer.sync()
                time.sleep(dt)
            return runner

    runner.report()
    return runner


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="task2")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-time", type=float, default=180.0)
    args = ap.parse_args()
    r = run(args.task, args.headless, args.max_time)
    raise SystemExit(0 if r.finished else 1)
