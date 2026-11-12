"""
并行装配序列生成（论文 Sec 5.2, Algorithm 2）。

核心思想:
    **目标图中深度相同的模块可以同时装配。**
    因为深度 d 的模块，其父模块深度是 d−1，上一轮已经装好了 ——
    同一层彼此之间没有先后依赖，天然可以并发。

装配方向: 从根往叶子长，d 从 1 递增到 d(G)，像树一样一圈一圈往外扩。

唯一的例外（论文 5.2）:
    多个模块同时怼**根模块**的不同连接面时，必须拆成两批 ——
    先 LEFT/RIGHT，后 TOP/BOTTOM。
    理由: 根模块没有固定在地面上，难以保证所有模块同时对称贴上；
    受力不平衡会把根模块推跑，导致后续对接全部失准。
    而对接非根模块时，它早已连成一整串（质量大、摩擦大），不易被推动。

输出: 动作组队列 —— **组间串行，组内并行**。
"""

from __future__ import annotations

from dataclasses import dataclass

from .faces import FACE_CHAR, Face
from .topology import Topology


@dataclass(frozen=True)
class AssemblyAction:
    """
    论文的装配动作四元组 a = (m_i, c_i, m_j, c_j):
    把模块 m_i 的连接面 c_i，接到模块 m_j 的连接面 c_j 上。
    """

    mi: int  # 移动模块
    ci: Face  # 移动模块用来对接的面
    mj: int  # 父模块（已经在装配体上，不动）
    cj: Face  # 父模块被接的面

    @property
    def needs_helping_module(self) -> bool:
        """
        论文 5.3.3 情况二: 判据是**移动模块自己**用的面 c_i 是否为轮子面，
        与父模块用什么面 c_j 无关。
        因为 L/R 面就是两个驱动轮所在的面，模块无法一边用轮子面当连接器
        贴合对方、一边还用轮子驱动自己前进。
        """
        return self.ci in (Face.LEFT, Face.RIGHT)

    def __str__(self) -> str:
        s = f"({self.mi},{FACE_CHAR[self.ci]},{self.mj},{FACE_CHAR[self.cj]})"
        return s + " ⚠需helping" if self.needs_helping_module else s


def _split_root_group(
    actions: list[AssemblyAction], m_root: int
) -> list[list[AssemblyAction]]:
    """
    论文 5.2 的例外规则：把涉及根模块的动作组拆成两批。
      第一批 —— 接根模块 LEFT / RIGHT 面的动作
      第二批 —— 接根模块 TOP / BOTTOM 面的动作
    注意判据是**父模块（根）被接的那个面 c_j**，不是移动模块的 c_i。
    """
    first = [a for a in actions if a.mj == m_root and a.cj in (Face.LEFT, Face.RIGHT)]
    later = [a for a in actions if a.mj == m_root and a.cj in (Face.TOP, Face.BOTTOM)]
    others = [a for a in actions if a.mj != m_root]  # 同层里不接根的，可跟第一批并行
    groups = []
    if first or others:
        groups.append(first + others)
    if later:
        groups.append(later)
    return groups


def build_schedule(
    topo: Topology,
    tau: int | None = None,
    f: list[int] | None = None,
    split_root: bool = True,
) -> list[list[AssemblyAction]]:
    """
    Algorithm 2。返回动作组列表，组间串行、组内并行。

    topo : 目标构型
    tau  : 目标图的根顶点（默认用 Algorithm 1 求）
    f    : 分配映射 f[v] = 实际模块编号（默认恒等映射，即顶点标号就是模块编号）
    """
    if tau is None:
        tau = topo.find_root()
    if f is None:
        f = list(range(topo.n))  # 恒等映射

    depths = topo.depths(tau)
    parent = topo.parent_of(tau)
    max_d = max(depths)

    schedule: list[list[AssemblyAction]] = []
    for d in range(1, max_d + 1):
        layer = [v for v in range(topo.n) if depths[v] == d]
        actions = []
        for v in sorted(layer):
            p, c_child, c_parent = parent[v]
            # 论文 Algorithm 2: 由映射 f 查出实际的移动模块和它的父模块
            actions.append(AssemblyAction(f[v], c_child, f[p], c_parent))

        # split_root=False 仅用于消融实验：关掉论文 5.2 的分批规则做对照
        if d == 1 and split_root:
            # 这一层全部接在根模块上 → 套用例外规则拆两批
            schedule.extend(_split_root_group(actions, f[tau]))
        else:
            schedule.append(actions)
    return schedule


def describe(schedule: list[list[AssemblyAction]]) -> str:
    lines = []
    for k, group in enumerate(schedule, 1):
        tag = "  ".join(str(a) for a in group)
        lines.append(f"    第 {k} 批（组内 {len(group)} 个动作并行）: {tag}")
    return "\n".join(lines)


def summary(schedule: list[list[AssemblyAction]]) -> dict:
    n_actions = sum(len(g) for g in schedule)
    n_help = sum(a.needs_helping_module for g in schedule for a in g)
    return {
        "n_groups": len(schedule),
        "n_actions": n_actions,
        "n_helping": n_help,
        "max_parallel": max(len(g) for g in schedule) if schedule else 0,
        "speedup_vs_serial": n_actions / len(schedule) if schedule else 0.0,
    }


if __name__ == "__main__":
    from .topology import PAPER_TASK1, PAPER_TASK2

    # 论文 Sec 6.2 / Fig 13 描述的 Task 2 实际执行时序
    EXPECT_TASK2 = [
        ["(0,T,1,L)", "(5,T,1,R)"],
        ["(4,T,1,B)", "(8,T,1,T)"],
        ["(2,T,5,B)", "(3,T,8,B)", "(6,T,0,B)", "(7,T,4,B)"],
    ]
    # 论文 Sec 6.1 / Fig 8 描述的 Task 1 实际执行时序
    EXPECT_TASK1 = [
        ["(0,B,1,L)", "(5,B,1,R)"],
        ["(2,B,1,T)", "(6,T,1,B)"],
        ["(4,T,6,B)"],
        ["(3,T,4,B)"],
    ]

    for name, topo, expect in (
        ("论文 Task 2 · 九模块全向车", PAPER_TASK2, EXPECT_TASK2),
        ("论文 Task 1 · 七模块移动机械臂", PAPER_TASK1, EXPECT_TASK1),
    ):
        # 这两个构型的顶点标号直接沿用实际模块编号，故 f 取恒等映射
        sched = build_schedule(topo)
        print(f"\n[{name}]")
        print(describe(sched))
        s = summary(sched)
        print(f"    → {s['n_actions']} 个动作压缩成 {s['n_groups']} 批"
              f"（最大并行度 {s['max_parallel']}，"
              f"相对全串行加速 {s['speedup_vs_serial']:.2f}×，"
              f"需 helping module 的动作 {s['n_helping']} 个）")

        got = [sorted(str(a).split(" ")[0] for a in g) for g in sched]
        exp = [sorted(g) for g in expect]
        assert got == exp, f"\n  算出: {got}\n  论文: {exp}"
        print(f"    ✓ 与论文描述的实际执行时序逐批一致")

    print("\n✓ Algorithm 2 在论文两个实验上都复现出论文的装配批次划分")
