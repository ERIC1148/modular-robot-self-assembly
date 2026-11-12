"""
模块机器人拓扑构型的图模型（论文 Sec 4.1）。

论文把「一堆模块拼成什么形状」抽象成图 G = (V, E):
    顶点 V = 模块，边 E = 一次物理连接
每条边记 (Face, Face2Con, Orientation) 三个属性。

本文件实现:
  - Topology      图模型 + 结构性校验（连通、树）
  - subtree_counts()  论文的 CN^v(c)
  - find_root()   Algorithm 1「根模块搜索」
  - depths()      Algorithm 2 分层并行所需的深度

关于 Orientation: 论文说该属性只在两个 BOTTOM 面对接时才有意义（0 或 π），
因为 L/R/T 面装在可转动自由度上、接上后还能自转。论文并主动排除了
Orientation=1（180°）的 BOTTOM-BOTTOM 连接（Fig 3b），理由是它会强加多余的
相对朝向约束。本项目是纯平面模型，不建模该自由度，故 Orientation 恒为 Null。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .faces import FACE_CHAR, Face


@dataclass(frozen=True)
class Connection:
    """一条边。含义: 模块 i 的 face_i 面 连接 模块 j 的 face_j 面。"""

    i: int
    face_i: Face
    j: int
    face_j: Face

    def __str__(self) -> str:
        return (
            f"({self.i},{FACE_CHAR[self.face_i]}) — "
            f"({self.j},{FACE_CHAR[self.face_j]})"
        )


class TopologyError(ValueError):
    pass


class Topology:
    """
    目标构型的图模型。

    adj[v] 是 dict: {v 的连接面 -> (邻居模块, 邻居用的面)}
    每个面最多只能有一条连接（物理上显然）。
    """

    def __init__(self, n: int, edges: list[tuple[int, Face, int, Face]]):
        self.n = n
        self.edges = [Connection(*e) for e in edges]
        self.adj: list[dict[Face, tuple[int, Face]]] = [{} for _ in range(n)]

        for e in self.edges:
            for a, fa, b, fb in ((e.i, e.face_i, e.j, e.face_j),
                                 (e.j, e.face_j, e.i, e.face_i)):
                if not (0 <= a < n):
                    raise TopologyError(f"模块编号越界: {a}")
                if fa in self.adj[a]:
                    raise TopologyError(
                        f"模块 {a} 的 {FACE_CHAR[fa]} 面被重复占用: "
                        f"已连 {self.adj[a][fa]}，又要连 ({b},{FACE_CHAR[fb]})"
                    )
                self.adj[a][fa] = (b, fb)
            if e.i == e.j:
                raise TopologyError(f"自环: 模块 {e.i} 连自己")

    # ------------------------------------------------------------------
    # 结构性校验
    # ------------------------------------------------------------------
    def neighbors(self, v: int) -> list[tuple[int, Face, Face]]:
        """返回 [(邻居, v 用的面, 邻居用的面), ...]。"""
        return [(u, f, fu) for f, (u, fu) in self.adj[v].items()]

    def is_connected(self) -> bool:
        """Definition 1 条件 1: G 必须连通。"""
        if self.n == 0:
            return False
        seen = {0}
        dq = deque([0])
        while dq:
            v = dq.popleft()
            for u, _, _ in self.neighbors(v):
                if u not in seen:
                    seen.add(u)
                    dq.append(u)
        return len(seen) == self.n

    def is_tree(self) -> bool:
        """
        论文只处理树形拓扑（无回路）。
        有回路时论文的做法是先跑生成树算法砍成树，装完再补边。
        n 个顶点的连通图是树 ⟺ 恰有 n-1 条边。
        """
        return self.is_connected() and len(self.edges) == self.n - 1

    def check_tree(self) -> None:
        if not self.is_connected():
            raise TopologyError("图不连通 —— 违反 Definition 1 条件 1")
        if len(self.edges) != self.n - 1:
            raise TopologyError(
                f"图有回路（{len(self.edges)} 条边，树应为 {self.n - 1} 条）。"
                f"论文的处理是先跑生成树算法砍成树。"
            )

    # ------------------------------------------------------------------
    # Algorithm 1: 根模块搜索
    # ------------------------------------------------------------------
    def subtree_counts(self) -> list[dict[Face, int]]:
        """
        计算论文的 CN^v(c)：从模块 v 的 c 面看出去，挂着多少个模块。

        论文 Algorithm 1 的写法是「以任意 v0 为根，按 height 由小到大
        自底向上做 DP」。这里用等价的子树规模法（同为 O(|V|)，结果相同）:
            以 v0 为根做后序遍历得到每个节点的子树规模 sz[v]，则对每条
            父子边 (p, c):
                CN^p(p 朝向 c 的面) = sz[c]
                CN^c(c 朝向 p 的面) = n - sz[c]
        未占用的连接面 CN 为 0。
        """
        self.check_tree()
        n = self.n
        parent: list[int | None] = [None] * n
        parent_face: list[Face | None] = [None] * n  # 子节点朝向父节点的那个面
        order: list[int] = []
        seen = [False] * n
        seen[0] = True
        dq = deque([0])
        while dq:  # BFS 得到一个拓扑序，逆序即后序
            v = dq.popleft()
            order.append(v)
            for u, f_v, f_u in self.neighbors(v):
                if not seen[u]:
                    seen[u] = True
                    parent[u] = v
                    parent_face[u] = f_u
                    dq.append(u)

        sz = [1] * n
        for v in reversed(order):  # 自底向上累加子树规模
            p = parent[v]
            if p is not None:
                sz[p] += sz[v]

        cn: list[dict[Face, int]] = [{f: 0 for f in Face} for _ in range(n)]
        for v in order:
            p = parent[v]
            if p is None:
                continue
            f_child = parent_face[v]  # v 朝向 p 的面
            f_parent = next(f for f, (u, _) in self.adj[p].items() if u == v)
            cn[p][f_parent] = sz[v]  # p 从这个面看出去挂着 v 的整棵子树
            cn[v][f_child] = n - sz[v]  # v 从这个面看出去是「其余全部」
        return cn

    def find_root(self, verbose: bool = False) -> int:
        """
        Algorithm 1。返回满足论文式(1) 的根模块 τ:

            CN^τ(c) ≤ ½|V|,  ∀c ∈ C

        直观含义: 从 τ 的任何一个面看出去，挂着的模块都不超过总数的一半 ——
        即图论中的**树的重心**。树的重心一定存在（1 个或 2 个）。
        选它当根，装配时左右两边可以同时开工，且它作为地基受力最平衡。
        """
        cn = self.subtree_counts()
        half = self.n / 2.0
        cands = [v for v in range(self.n) if max(cn[v].values()) <= half]
        if verbose:
            for v in range(self.n):
                detail = " ".join(
                    f"{FACE_CHAR[f]}={cn[v][f]}" for f in Face if cn[v][f]
                )
                mark = "  ← 满足式(1)" if v in cands else ""
                print(f"    模块 {v}: max CN = {max(cn[v].values())}  [{detail}]{mark}")
        if not cands:
            raise TopologyError("找不到满足式(1) 的根 —— 树重心必然存在，图结构有误")
        return cands[0]  # 有两个重心时任取其一

    # ------------------------------------------------------------------
    # 深度 / 高度（Algorithm 2 分层并行要用）
    # ------------------------------------------------------------------
    def depths(self, root: int) -> list[int]:
        """d(v)：从根 τ 沿树往下走到 v 的边数。根的深度为 0。"""
        self.check_tree()
        d = [-1] * self.n
        d[root] = 0
        dq = deque([root])
        while dq:
            v = dq.popleft()
            for u, _, _ in self.neighbors(v):
                if d[u] < 0:
                    d[u] = d[v] + 1
                    dq.append(u)
        return d

    def parent_of(self, root: int) -> list[tuple[int, Face, Face] | None]:
        """
        以 root 为根时，每个顶点的父节点信息。
        返回 [(父模块, 本模块朝向父的面 c, 父模块的配对面 c̃'), ...]，根为 None。
        这正是 Algorithm 2 每轮迭代要取的 (v, c, ṽ^c, c̃')。
        """
        self.check_tree()
        res: list[tuple[int, Face, Face] | None] = [None] * self.n
        seen = [False] * self.n
        seen[root] = True
        dq = deque([root])
        while dq:
            v = dq.popleft()
            for u, f_v, f_u in self.neighbors(v):
                if not seen[u]:
                    seen[u] = True
                    res[u] = (v, f_u, f_v)  # u 用 f_u 面 接 v 的 f_v 面
                    dq.append(u)
        return res

    def summary(self) -> str:
        return (
            f"Topology(n={self.n}, edges={len(self.edges)}, "
            f"tree={self.is_tree()})"
        )


# ----------------------------------------------------------------------
# 论文实验的目标构型（用于验证）
# ----------------------------------------------------------------------
T, L, B, R = Face.TOP, Face.LEFT, Face.BOTTOM, Face.RIGHT

# 论文 Task 2「九模块全向车」(Fig 13f)。
# 边直接取自论文 Sec 6.2 列出的装配动作:
#   (0,T,1,L) (5,T,1,R) (4,T,1,B) (8,T,1,T)
#   (3,T,8,B) (2,T,5,B) (7,T,4,B) (6,T,0,B)
# 注意所有移动模块用的都是 TOP 面 ⇒ 全程不需要 helping module。
PAPER_TASK2 = Topology(
    9,
    [
        (0, T, 1, L),
        (5, T, 1, R),
        (4, T, 1, B),
        (8, T, 1, T),
        (3, T, 8, B),
        (2, T, 5, B),
        (7, T, 4, B),
        (6, T, 0, B),
    ],
)

# 论文 Task 1「七模块移动机械臂」(Fig 8g)，边取自 Sec 6.1 的装配动作:
#   (0,B,1,L) (5,B,1,R) (2,B,1,T) (6,T,1,B) (4,T,6,B) (3,T,4,B)
PAPER_TASK1 = Topology(
    7,
    [
        (0, B, 1, L),
        (5, B, 1, R),
        (2, B, 1, T),
        (6, T, 1, B),
        (4, T, 6, B),
        (3, T, 4, B),
    ],
)


if __name__ == "__main__":
    for name, topo, expect_root in (
        ("论文 Task 2 · 九模块全向车", PAPER_TASK2, 1),
        ("论文 Task 1 · 七模块移动机械臂", PAPER_TASK1, 1),
    ):
        print(f"\n[{name}]  {topo.summary()}")
        print("  Algorithm 1 逐模块 CN^v(c):")
        root = topo.find_root(verbose=True)
        print(f"  ⇒ 根模块 τ = {root}（论文实验中的根模块 = {expect_root}）")
        assert root == expect_root, f"根模块不符: {root} != {expect_root}"

        d = topo.depths(root)
        by_depth: dict[int, list[int]] = {}
        for v, dv in enumerate(d):
            by_depth.setdefault(dv, []).append(v)
        print("  按深度分层（Algorithm 2 的并行单位）:")
        for dv in sorted(by_depth):
            print(f"    depth {dv}: {by_depth[dv]}")
    print("\n✓ Algorithm 1 在论文两个实验构型上都复现出论文的根模块")
