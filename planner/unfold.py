"""
把目标构型「摊平」到平面 —— 论文 Sec 5.1 的展开计算 + Sec 4.2 的 Definition 1 校验。

为什么要摊平:
    模块是差速轮地面车，只能在平面上开，不会飞、也不能自己爬到别人头顶。
    所以装配实际发生在**二维平面**上：先把目标构型展开成平面图案，各模块开到
    自己的「停车位」完成对接；装完之后再靠各自的转动自由度折成三维形态
    —— 就像折纸的展开图 → 成品。

论文的关键前提（Sec 5.1）:
    所有模块初始平放在地面、电路板朝上。在此前提下，若要求所有模块位移不变，
    则把 G 展开成平面图案的**解是唯一的** ⇒ 不需要搜索，直接递推算出来。

递推核心（式(2)）:
    一个模块的位姿 = 父模块位姿 ⊕ 二者的连接关系
    （与机器人学 D-H 参数递推同构：知道上一节在哪、这一节怎么接的，就能算出这一节）
"""

from __future__ import annotations

import numpy as np

from .faces import W, Face, compose, mate_relpose
from .topology import Topology, TopologyError


def unfold(topo: Topology, root: int) -> np.ndarray:
    """
    从根模块出发做 BFS，用式(2) 逐层算出每个顶点摊平后的位姿。

    返回 shape (n, 3) 的数组，每行 (x, y, yaw)，均以根模块为原点。
    根模块自身为 [0, 0, 0]。
    """
    poses = np.full((topo.n, 3), np.nan)
    poses[root] = [0.0, 0.0, 0.0]

    parent = topo.parent_of(root)
    # 按深度升序访问，保证访问子节点时父节点已经算好（BFS 序）
    order = sorted(range(topo.n), key=lambda v: topo.depths(root)[v])
    for v in order:
        if parent[v] is None:
            continue
        p, f_child, f_parent = parent[v]
        # 父模块的 f_parent 面 与 本模块的 f_child 面 对接
        # ⇒ 本模块相对父模块的位姿 = mate_relpose(f_parent, f_child)
        poses[v] = compose(poses[p], mate_relpose(f_parent, f_child))
    if np.isnan(poses).any():
        raise TopologyError("有模块未被访问到 —— 图不连通")
    return poses


def to_grid(poses: np.ndarray) -> np.ndarray:
    """把展开位姿的 xy 量化到以 W 为单位的整数格点（Definition 1 条件 2/3 的载体）。"""
    return np.rint(poses[:, :2] / W).astype(int)


def validate_definition1(topo: Topology, root: int | None = None) -> dict:
    """
    Definition 1（论文 Sec 4.2）: 目标构型可自组装 ⟺ 能完全摊平到一个平面，满足

        1. G 是连通图
        2. 相邻两模块的中心距 = w
        3. 每个模块的中心占据唯一位置

    条件 2+3 合起来的效果是：展开图必须是**正方形网格上的无重叠贴片图案**
    （像俄罗斯方块，格子对格子，不许叠）。

    返回诊断字典；不抛异常，便于上层打印完整报告。
    """
    report: dict = {"ok": True, "problems": []}

    # ---- 条件 1: 连通（论文还要求是树，有回路要先砍成生成树）----
    report["connected"] = topo.is_connected()
    report["is_tree"] = topo.is_tree()
    if not report["connected"]:
        report["ok"] = False
        report["problems"].append("条件1 违反: 图不连通")
        return report
    if not report["is_tree"]:
        report["ok"] = False
        report["problems"].append(
            f"图有回路（{len(topo.edges)} 边 / {topo.n} 顶点），"
            f"论文只处理树，需先跑生成树算法"
        )
        return report

    if root is None:
        root = topo.find_root()
    report["root"] = root
    poses = unfold(topo, root)
    report["poses"] = poses
    report["grid"] = to_grid(poses)

    # ---- 条件 2: 相邻模块中心距 = w ----
    # 由 mate_relpose 的构造天然满足；这里做数值复核，防止实现漂移。
    worst = 0.0
    for e in topo.edges:
        d = float(np.linalg.norm(poses[e.i][:2] - poses[e.j][:2]))
        worst = max(worst, abs(d - W))
    report["max_edge_len_error"] = worst
    if worst > 1e-9:
        report["ok"] = False
        report["problems"].append(f"条件2 违反: 相邻中心距偏离 w 达 {worst:.2e} m")

    # ---- 条件 3: 位置唯一（真正的判别条件）----
    grid = report["grid"]
    seen: dict[tuple[int, int], int] = {}
    overlaps = []
    for v, cell in enumerate(map(tuple, grid)):
        if cell in seen:
            overlaps.append((seen[cell], v, cell))
        else:
            seen[cell] = v
    report["overlaps"] = overlaps
    if overlaps:
        report["ok"] = False
        for a, b, cell in overlaps:
            report["problems"].append(
                f"条件3 违反: 模块 {a} 与 {b} 摊平后都落在格点 {cell}"
            )

    # ---- 附加提示: 哪些模块需要 helping module（论文 5.3.3 情况二）----
    # 判据是**移动模块自己**用来对接的面 c_i 是否为 L/R（轮子面）。
    parent = topo.parent_of(root)
    need_help = [
        (v, parent[v][1]) for v in range(topo.n)
        if parent[v] is not None and parent[v][1] in (Face.LEFT, Face.RIGHT)
    ]
    report["need_helping_module"] = need_help
    return report


def ascii_layout(topo: Topology, root: int, poses: np.ndarray) -> str:
    """把展开图画成 ASCII 方格，让 Definition 1 变得可见。"""
    grid = to_grid(poses)
    gx, gy = grid[:, 0], grid[:, 1]
    x0, x1, y0, y1 = gx.min(), gx.max(), gy.min(), gy.max()
    occ: dict[tuple[int, int], list[int]] = {}
    for v, cell in enumerate(map(tuple, grid)):
        occ.setdefault(cell, []).append(v)

    lines = []
    for y in range(y1, y0 - 1, -1):  # +y 朝上
        cells = []
        for x in range(x0, x1 + 1):
            vs = occ.get((x, y), [])
            if not vs:
                cells.append(" . ")
            elif len(vs) > 1:
                cells.append(" ✗ ")  # 重叠 —— 违反条件 3
            else:
                v = vs[0]
                cells.append(f"[{v}]" if v == root else f" {v} ")
        lines.append("".join(cells))
    return "\n".join("    " + ln for ln in lines)


def print_report(topo: Topology, name: str = "") -> dict:
    print(f"\n[{name}]  {topo.summary()}")
    rep = validate_definition1(topo)
    if not rep["connected"] or not rep["is_tree"]:
        for p in rep["problems"]:
            print("  ✗ " + p)
        return rep

    root = rep["root"]
    print(f"  Algorithm 1 求得根模块 τ = {root}")
    print(f"  Definition 1 校验:")
    print(f"    条件1 连通           : {'✓' if rep['connected'] else '✗'}"
          f"（且为树: {'✓' if rep['is_tree'] else '✗'}）")
    print(f"    条件2 相邻中心距 = w : "
          f"{'✓' if rep['max_edge_len_error'] < 1e-9 else '✗'}"
          f"（最大偏差 {rep['max_edge_len_error']:.1e} m）")
    print(f"    条件3 位置唯一       : "
          f"{'✓' if not rep['overlaps'] else '✗ ' + str(rep['overlaps'])}")
    print(f"  ⇒ {'可自组装 ✓' if rep['ok'] else '不可自组装 ✗'}")

    nh = rep["need_helping_module"]
    if nh:
        print(f"  ⚠ 需 helping module 的模块（c_i 为轮子面）: "
              + ", ".join(f"m{v}用{f.name}" for v, f in nh))
    else:
        print(f"  ✓ 所有移动模块都用 T/B 面对接，全程不需要 helping module")

    print(f"  摊平后的平面图案（[n] 为根，+y 朝上，+x 朝右）:")
    print(ascii_layout(topo, root, rep["poses"]))
    return rep


if __name__ == "__main__":
    from .topology import PAPER_TASK1, PAPER_TASK2, Topology

    r2 = print_report(PAPER_TASK2, "论文 Task 2 · 九模块全向车 (Fig 13f)")
    assert r2["ok"]
    r1 = print_report(PAPER_TASK1, "论文 Task 1 · 七模块移动机械臂 (Fig 8g)")
    assert r1["ok"]

    T, L, B, R = Face.TOP, Face.LEFT, Face.BOTTOM, Face.RIGHT

    # 反例 1: 五模块链，每节左转 90°，绕一圈后第 5 个模块正好折回第 1 个的格点。
    # 图本身是合法的树，但摊平后违反条件 3 —— 这正是 Definition 1 存在的意义:
    # 拓扑上讲得通，物理上摆不下。
    folded = Topology(5, [(1, B, 0, T), (2, B, 1, L), (3, B, 2, L), (4, B, 3, L)])
    rb = print_report(folded, "反例1 · 链条折回自我重叠（违反条件3）")
    assert not rb["ok"], "该构型摊平后重叠，应判为不可自组装"

    # 反例 2: 含回路。论文只处理树，回路要先跑生成树砍掉。
    loop = Topology(
        4, [(1, B, 0, T), (2, B, 1, L), (3, B, 2, L), (3, L, 0, L)]
    )
    rl = print_report(loop, "反例2 · 含回路（论文只处理树）")
    assert not rl["ok"], "含回路的构型应被拒绝"

    print("\n✓ 式(2) 展开 + Definition 1 校验通过（含两类反例检出）")
