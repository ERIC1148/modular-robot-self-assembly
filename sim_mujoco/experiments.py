"""
两个小创新，每个对应一块 MuJoCo 核心知识。

创新 A —— 轨迹尾迹 + 复刻论文路径图（论文 Fig 9 / 14 / 17 那种彩色路径）
    MuJoCo 知识点:
      1. viewer.user_scn —— 往渲染场景里塞「只画不算」的装饰几何体
      2. mujoco.mjv_connector —— 用两点画一根胶囊/线段
      3. 每隔 N 步从 data.qpos 采样 —— 仿真数据采集的基本套路

创新 B —— 消融实验：论文 5.2「根模块那层分两批」到底有没有用
    MuJoCo 知识点:
      1. data.qpos 读根模块位移 —— 定量测「根被推跑多远」
      2. mujoco.mj_contactForce —— 从接触点里取出法向力，测「推力多大」
      3. 同一模型、只改一个开关、对比两组数据 —— 仿真做对照实验的标准做法

用法:
    mjpython -m sim_mujoco.experiments trails          # 开窗口看实时尾迹（macOS 必须 mjpython）
    python   -m sim_mujoco.experiments plot            # 无头跑完，画论文风格路径图 png
    python   -m sim_mujoco.experiments ablation        # 消融实验，打印对照表
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from .runner_full import TASKS, AssemblyRunner

# 每个模块一种颜色（按模块身份上色 —— 这正是论文路径图的做法，区别于状态配色）
PALETTE = [
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
]
SAMPLE_EVERY = 25  # 每 25 步（0.05s）采一个轨迹点


# ======================================================================
# 创新 A：轨迹尾迹
# ======================================================================
class TrailRecorder:
    """记录每个模块的平面轨迹。"""

    def __init__(self, n: int):
        self.pts: list[list[np.ndarray]] = [[] for _ in range(n)]

    def sample(self, br) -> None:
        for i in range(br.n):
            p = br.get_pose(i)[:2].copy()
            # 没动就不重复记，免得尾迹几何体数量白白爆掉
            if not self.pts[i] or np.linalg.norm(p - self.pts[i][-1]) > 0.004:
                self.pts[i].append(p)


def draw_trails(scn: mujoco.MjvScene, trails: TrailRecorder, z: float = 0.003) -> None:
    """
    把轨迹画进 MjvScene。

    要点（MuJoCo 知识）:
      - scn.geoms 是一个固定容量的数组（容量 = scn.maxgeom），
        scn.ngeom 是当前已用数量。我们在它后面追加，**每帧先重置**，
        否则线段会无限累积直到塞满报错。
      - mjv_initGeom 初始化一个几何体槽位，mjv_connector 把它变成连接两点的胶囊。
      - 这些几何体只参与渲染，不参与物理 —— 和 MJCF 里的 geom 是两回事。
    """
    scn.ngeom = 0
    for i, pts in enumerate(trails.pts):
        rgba = np.array([*PALETTE[i % len(PALETTE)], 0.9], dtype=np.float32)
        for a, b in zip(pts[:-1], pts[1:]):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.zeros(9), rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025,
                                 np.array([a[0], a[1], z]), np.array([b[0], b[1], z]))
            scn.ngeom += 1


def run_trails(task: str):
    import mujoco.viewer

    _, topo, poses = TASKS[task]
    r = AssemblyRunner(topo, poses, verbose=False)
    tr = TrailRecorder(r.br.n)
    dt = r.br.model.opt.timestep
    k = 0
    with mujoco.viewer.launch_passive(r.br.model, r.br.data) as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 90, -70, 1.5
        v.cam.lookat[:] = [0, 0, 0.04]
        while v.is_running():
            t0 = time.time()
            with v.lock():
                if not r.finished:
                    r.tick()
                else:
                    r.br.stop_all()
                    r.br.step()
                if k % SAMPLE_EVERY == 0:
                    tr.sample(r.br)
                    draw_trails(v.user_scn, tr)
            v.sync()
            k += 1
            lag = dt - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


def run_plot(task: str, out: str):
    """无头跑完，画一张和论文 Fig 9/14 同风格的路径图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, topo, poses = TASKS[task]
    r = AssemblyRunner(topo, poses, verbose=False)
    tr = TrailRecorder(r.br.n)
    k = 0
    while not r.finished and r.br.time < 250:
        r.tick()
        if k % SAMPLE_EVERY == 0:
            tr.sample(r.br)
        k += 1
    tr.sample(r.br)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, pts in enumerate(tr.pts):
        p = np.array(pts)
        c = PALETTE[i % len(PALETTE)]
        ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.4)
        ax.plot(*p[0], "o", color=c, ms=9)
        ax.annotate(str(i), p[0], color="white", ha="center", va="center", fontsize=7)
        ax.plot(*p[-1], "s", mfc="none", mec=c, ms=10, mew=1.2)  # 终点 = 停车位
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Actual path of each module ({task}, sim t={r.br.time:.1f}s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"✓ 路径图已保存: {out}（实心圆 = 起点，空心方块 = 停车位）")


# ======================================================================
# 创新 B：消融实验
# ======================================================================
def root_push_force(br, m_root: int) -> float:
    """
    当前时刻，根模块受到的「来自其他模块」的接触法向力之和 (N)。

    MuJoCo 知识点:
      - data.contact[k] 只有几何信息（哪两个 geom、穿透深度、接触坐标系），
        **没有力**。力要用 mj_contactForce(model, data, k, out6) 单独算，
        out6 的前 3 个分量是接触坐标系下的力，[0] 就是法向力。
      - geom → body 用 model.geom_bodyid；但手腕/轮子是根模块 body 的子 body，
        所以要用 model.body_rootid 归到「顶层 body」再判断属于哪个模块。
    """
    M, D = br.model, br.data
    root_body = br._bid[m_root]
    module_bodies = set(br._bid)
    f6 = np.zeros(6)
    total = 0.0
    for k in range(D.ncon):
        c = D.contact[k]
        b1 = M.body_rootid[M.geom_bodyid[c.geom1]]
        b2 = M.body_rootid[M.geom_bodyid[c.geom2]]
        if root_body in (b1, b2):
            other = b2 if b1 == root_body else b1
            if other in module_bodies:  # 排除地面
                mujoco.mj_contactForce(M, D, k, f6)
                total += abs(f6[0])
    return total


SETTLE_T = 1.0  # 开局沉降时间：模块刚生成时会落稳几毫米，不能算作"被推跑"


def run_one(task: str, split_root: bool, dock_tol: float) -> dict:
    from . import bridge

    # ⚠️ 接受域是关键变量：默认 12mm 时，模块在还没碰到根模块之前就触发焊接，
    # 根本不会产生推挤 —— 论文要解决的那个物理问题在仿真里被简化掉了。
    # 收紧到接近论文 EP-Face 硬件（法向 4mm）时，模块才会真的顶上去。
    bridge.DOCK_POS_TOL = dock_tol
    _, topo, poses = TASKS[task]
    r = AssemblyRunner(topo, poses, verbose=False, split_root=split_root)
    m_root = r.asg["m_tau"]
    while r.br.time < SETTLE_T:  # 先只让物理落稳，不发任何指令
        r.br.step()
    p0 = r.br.get_pose(m_root).copy()
    max_disp, max_force = 0.0, 0.0
    while not r.finished and r.br.time < 250:
        r.tick()
        p = r.br.get_pose(m_root)
        max_disp = max(max_disp, float(np.linalg.norm(p[:2] - p0[:2])))
        max_force = max(max_force, root_push_force(r.br, m_root))
    errs = [r.br.dock_error(a.mi, a.ci, a.mj, a.cj) for g in r.schedule for a in g]
    return {
        "split": split_root,
        "tol": dock_tol,
        "batches": len(r.schedule),
        "finished": r.finished,
        "time": r.br.time,
        "root_disp_mm": max_disp * 1000,
        "root_force_N": max_force,
        "max_residual_mm": max(e[0] for e in errs) * 1000 if r.finished else float("nan"),
    }


def run_ablation(task: str):
    print(f"### 消融实验：{TASKS[task][0]} —— 根模块那层分两批 vs 不分 ###\n")
    rows = [run_one(task, s, tol) for tol in (0.012, 0.002) for s in (True, False)]
    print(f"{'接受域':<8}{'条件':<16}{'批数':>4}{'完成':>6}{'耗时s':>8}"
          f"{'根最大位移mm':>14}{'根受推力峰值N':>14}{'最大残差mm':>12}")
    for d in rows:
        name = "分两批(论文)" if d["split"] else "不分批(对照)"
        print(f"{d['tol']*1000:>4.0f}mm   {name:<14}{d['batches']:>4}"
              f"{str(d['finished']):>6}{d['time']:>8.1f}"
              f"{d['root_disp_mm']:>14.2f}{d['root_force_N']:>14.2f}"
              f"{d['max_residual_mm']:>12.2f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["trails", "plot", "ablation"])
    ap.add_argument("--task", choices=list(TASKS), default="task2")
    ap.add_argument("--out", default="media/paths-task2.png")
    a = ap.parse_args()
    if a.mode == "trails":
        run_trails(a.task)
    elif a.mode == "plot":
        run_plot(a.task, a.out)
    else:
        run_ablation(a.task)
