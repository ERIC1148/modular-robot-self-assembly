"""
任务分配（论文 Sec 5.1 后半段）：把地上散落的 n 个模块，最优地分配到目标构型的 n 个坑位。

因为所有模块长得一模一样（同质），派谁去哪个坑位都行，
所以这是一个纯粹的**最优匹配问题**（assignment problem）。

三步:
  1. 式(3)  选实际根模块 m_τ —— 离所有模块质心最近的那个
  2. 锚定    强制把 m_τ 映射到目标图的根 τ
  3. 匈牙利  剩下的模块最小化总移动距离，Kuhn-Munkres 求全局最优

输出映射 f: V → M（目标顶点 → 实际模块），供 Algorithm 2 生成装配序列。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .controller import to_local
from .topology import Topology
from .unfold import unfold

# 用于"钉死"某个配对的巨大代价（比任何真实距离都大得多）
_BIG = 1e6


def select_root_module(poses: np.ndarray) -> int:
    """
    论文式(3)：选离所有模块质心最近的模块当地基。

        o_c = [Σx_i/n, Σy_i/n]ᵀ
        m_τ = argmin_i ‖o_i − o_c‖

    理由很朴素：地基不动、其他模块开过来找它，
    它待在正中间时大家跑的总路程最短。
    """
    centroid = poses[:, :2].mean(axis=0)
    return int(np.argmin(np.linalg.norm(poses[:, :2] - centroid, axis=1)))


def build_cost_matrix(
    actual_rel: np.ndarray,
    target_rel: np.ndarray,
    m_tau: int,
    tau: int,
    w_orient: float = 0.0,
) -> np.ndarray:
    """
    构造代价矩阵 C[j][i] = 模块 m_i 去填坑位 v_j 的代价。

    actual_rel : (n,3) 每个实际模块相对 m_τ 的位姿  ô_i
    target_rel : (n,3) 每个目标顶点相对 τ 的位姿    õ_j
    w_orient   : 朝向差异的权重。论文说"代价不限于距离，也可以把朝向等因素
                 加进去"，但实验用的是纯距离，故默认 0。

    锚定处理：论文把 m_τ 映射到 τ（‖ô_τ − õ_τ‖ = 0）。
    实现上给该行/列的其他配对赋巨大代价，强制匈牙利算法选中这一对。
    """
    n = len(actual_rel)
    diff = target_rel[:, None, :2] - actual_rel[None, :, :2]  # (n_target, n_actual, 2)
    C = np.linalg.norm(diff, axis=2)

    if w_orient > 0:
        dtheta = target_rel[:, None, 2] - actual_rel[None, :, 2]
        dtheta = np.abs(np.arctan2(np.sin(dtheta), np.cos(dtheta)))
        C = C + w_orient * dtheta

    # 钉死 τ ↔ m_τ
    C[tau, :] += _BIG
    C[:, m_tau] += _BIG
    C[tau, m_tau] -= 2 * _BIG
    return C


def assign(
    poses: np.ndarray,
    topo: Topology,
    tau: int | None = None,
    m_tau: int | None = None,
    w_orient: float = 0.0,
) -> dict:
    """
    完整的任务分配流程。

    poses : (n,3) 各实际模块在世界系下的位姿 (x, y, yaw)
    topo  : 目标构型

    返回诊断字典，其中
        f      : list，f[j] = 目标顶点 j 应由哪个实际模块充当
        f_inv  : list，f_inv[i] = 实际模块 i 对应哪个目标顶点（它的停车位）
        goals  : (n,3) 每个**实际模块**的目标位姿（世界系），可直接喂给控制器
    """
    n = len(poses)
    assert topo.n == n, f"模块数 {n} 与目标构型顶点数 {topo.n} 不符"

    if tau is None:
        tau = topo.find_root()
    if m_tau is None:
        m_tau = select_root_module(poses)

    # ô_i：实际模块相对 m_τ 的位姿（m_τ 自己是 [0,0,0]）
    actual_rel = np.array([to_local(poses[i], poses[m_tau]) for i in range(n)])
    # õ_j：目标顶点相对 τ 的位姿（τ 自己是 [0,0,0]）
    target_rel = unfold(topo, tau)

    C = build_cost_matrix(actual_rel, target_rel, m_tau, tau, w_orient)
    rows, cols = linear_sum_assignment(C)  # rows = 目标顶点, cols = 实际模块

    f = [-1] * n
    for j, i in zip(rows, cols):
        f[j] = int(i)
    f_inv = [-1] * n
    for j, i in enumerate(f):
        f_inv[i] = j

    # 每个实际模块的目标位姿（世界系）:
    # 目标布局整体锚定在 m_τ 上，所以把 õ 从 τ 系换算到 m_τ 的世界位姿下
    from .faces import compose

    goals = np.array([compose(poses[m_tau], target_rel[f_inv[i]]) for i in range(n)])

    # 总移动距离（不含被钉死的根模块）
    total = sum(
        float(np.linalg.norm(actual_rel[f[j]][:2] - target_rel[j][:2]))
        for j in range(n)
        if j != tau
    )

    return {
        "tau": tau,
        "m_tau": m_tau,
        "f": f,
        "f_inv": f_inv,
        "actual_rel": actual_rel,
        "target_rel": target_rel,
        "goals": goals,
        "cost_matrix": C,
        "total_distance": total,
    }


def greedy_assign_for_comparison(C: np.ndarray) -> tuple[list[int], float]:
    """
    贪心分配（每次挑当前最小），**仅用于对照说明匈牙利算法的必要性**。
    贪心会掉进局部最优，见 __main__ 里的对比输出。
    """
    C = C.copy()
    n = C.shape[0]
    f = [-1] * n
    total = 0.0
    for _ in range(n):
        j, i = np.unravel_index(np.argmin(C), C.shape)
        f[j] = int(i)
        total += float(C[j, i])
        C[j, :] = np.inf
        C[:, i] = np.inf
    return f, total


# ----------------------------------------------------------------------
# 论文 Task 2 的原始实验数据（用于严格验证）
# ----------------------------------------------------------------------
# 论文 Table 2：九个模块的初始位姿 (x, y, θ)。
# 注意 Module 1 恰在 (0,0,0)，即该表就是以 Module 1 的体坐标系表达的。
TABLE2_POSES = np.array([
    [-0.421,  0.388, -0.760],   # 0
    [ 0.000,  0.000,  0.000],   # 1
    [-0.110, -0.469,  1.445],   # 2
    [ 0.386, -0.143,  0.509],   # 3
    [-0.343,  0.082, -2.961],   # 4
    [ 0.118, -0.472,  1.700],   # 5
    [ 0.215,  0.428, -2.058],   # 6
    [-0.342, -0.044, -1.249],   # 7
    [ 0.270, -0.317,  2.416],   # 8
])

# 论文 Sec 6.2 给出的最优映射 f: 目标顶点 → 实际模块
PAPER_TASK2_MAPPING = [1, 0, 8, 5, 4, 6, 3, 2, 7]


def _paper_task2_target() -> Topology:
    """
    论文 Task 2 的目标构型，**用论文自己的顶点编号**（根为顶点 0）。

    论文 Sec 6.2 列出的装配动作是用**实际模块编号**写的:
        (0,T,1,L) (5,T,1,R) (4,T,1,B) (8,T,1,T)
        (3,T,8,B) (2,T,5,B) (7,T,4,B) (6,T,0,B)
    用论文给的映射 f 反查回目标顶点编号，即得下面的边表。
    （topology.PAPER_TASK2 是同一个构型，但顶点直接沿用实际模块编号 ——
      两者是同构的，只是标号不同，各自用途不同。）
    """
    from .faces import Face

    T, L, B, R = Face.TOP, Face.LEFT, Face.BOTTOM, Face.RIGHT
    inv = {m: v for v, m in enumerate(PAPER_TASK2_MAPPING)}  # 实际模块 → 目标顶点
    actions = [
        (0, T, 1, L), (5, T, 1, R), (4, T, 1, B), (8, T, 1, T),
        (3, T, 8, B), (2, T, 5, B), (7, T, 4, B), (6, T, 0, B),
    ]
    return Topology(9, [(inv[a], fa, inv[b], fb) for a, fa, b, fb in actions])


PAPER_TASK2_TARGET = _paper_task2_target()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    topo = PAPER_TASK2_TARGET
    poses = TABLE2_POSES

    print("[论文 Task 2 · 用论文 Table 2 的真实初始位姿做验证]\n")

    # --- 式(3) ---
    centroid = poses[:, :2].mean(axis=0)
    m_tau = select_root_module(poses)
    print(f"  所有模块质心 o_c = ({centroid[0]:+.4f}, {centroid[1]:+.4f})")
    dists = np.linalg.norm(poses[:, :2] - centroid, axis=1)
    for i, d in enumerate(dists):
        mark = "  ← 最近，选为 m_τ" if i == m_tau else ""
        print(f"    模块 {i}: ‖o_i − o_c‖ = {d:.4f} m{mark}")
    print(f"  ⇒ 式(3) 选出 m_τ = {m_tau}（论文实验: Module 1）")
    assert m_tau == 1

    # --- 目标图的根 ---
    tau = topo.find_root()
    print(f"\n  Algorithm 1 求得目标图的根 τ = {tau}（论文: 目标构型根为 Module 0）")
    assert tau == 0

    # --- 匈牙利 ---
    res = assign(poses, topo)
    print(f"\n  匈牙利算法求得映射 f（目标顶点 → 实际模块）:")
    print(f"    本项目算出 : {res['f']}")
    print(f"    论文给出   : {PAPER_TASK2_MAPPING}")
    print(f"    总移动距离 : {res['total_distance']:.4f} m")
    assert res["f"] == PAPER_TASK2_MAPPING, "映射与论文不符"
    print(f"  ✓ 与论文 Sec 6.2 给出的最优映射**完全一致**")

    # --- 贪心对照 ---
    C_clean = build_cost_matrix(
        res["actual_rel"], res["target_rel"], m_tau, tau
    ).copy()
    C_clean[tau, :] -= _BIG
    C_clean[:, m_tau] -= _BIG
    C_clean[tau, m_tau] += 2 * _BIG
    g_f, g_total = greedy_assign_for_comparison(C_clean)
    h_total = sum(C_clean[j, res["f"][j]] for j in range(topo.n))
    print(f"\n  【为什么必须用匈牙利算法而不能贪心】")
    print(f"    贪心分配   : {g_f}   总代价 {g_total:.4f} m")
    print(f"    匈牙利分配 : {res['f']}   总代价 {h_total:.4f} m")
    if g_total > h_total:
        print(f"    ⇒ 贪心比最优多走了 {(g_total - h_total)*100:.1f} cm "
              f"({(g_total/h_total - 1)*100:.1f}%)")
    else:
        print(f"    ⇒ 本例中贪心恰好也取到最优（同质模块+分散初值时常见），"
              f"但贪心无法保证最优")

    print("\n✓ 式(3) + 匈牙利分配复现论文结果")
