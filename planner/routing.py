"""
栅格路由（论文 Sec 5.2 末 + 5.3.1）。

论文原文的做法:
    「根模块一确定，就以它为中心生成一张方形栅格地图，这张栅格图作为所有模块
      导航的路由图（routing graph）。」实验中每格 10cm × 10cm（模块本体 8cm）。
    路径由多车规划器生成（引 Binder et al. 2019），冲突用 Wait / Avoid / Push 处理。

本文件实现栅格 + A*（单车，逐个规划，把其他模块当静态障碍）。
论文用的是真正的多车规划器；这里是简化版，但保留了「在栅格上绕障」这一核心 ——
**没有它，模块会直接走直线撞进已装好的装配体**（实测会被硬 weld 弹飞到 1 m/s）。
"""

from __future__ import annotations

import heapq

import numpy as np

CELL = 0.10  # 栅格边长 (m)，论文实验值
HALF_EXTENT = 1.30  # 栅格半径 (m)，覆盖整个作业区
INFLATE = 1  # 障碍膨胀半径（格）。模块 8cm、格 10cm，膨胀 1 格留出转身余量

_NEIGHBORS = [
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
]


class GridRouter:
    """以根模块为中心的方形栅格路由图。"""

    def __init__(self, origin: np.ndarray, cell: float = CELL,
                 half_extent: float = HALF_EXTENT):
        self.origin = np.asarray(origin[:2], dtype=float)
        self.cell = cell
        self.r = int(round(half_extent / cell))

    # ------------------------------------------------------------------
    def to_cell(self, xy) -> tuple[int, int]:
        d = (np.asarray(xy[:2]) - self.origin) / self.cell
        return (int(round(d[0])), int(round(d[1])))

    def to_world(self, cell) -> np.ndarray:
        return self.origin + np.array(cell, dtype=float) * self.cell

    def in_bounds(self, c) -> bool:
        return abs(c[0]) <= self.r and abs(c[1]) <= self.r

    # ------------------------------------------------------------------
    def blocked_cells(
        self,
        obstacles: list[np.ndarray],
        free_around: np.ndarray | None = None,
        free_radius: int = 2,
    ) -> set[tuple[int, int]]:
        """
        把障碍模块的位置转成被占据的格子（含膨胀）。

        free_around: 目标点附近强制留空的中心。因为逼近起点就贴在装配体旁边，
                     必然落在装配体的膨胀区里；不豁免的话 A* 永远找不到路。
        """
        blocked: set[tuple[int, int]] = set()
        for o in obstacles:
            cx, cy = self.to_cell(o)
            for dx in range(-INFLATE, INFLATE + 1):
                for dy in range(-INFLATE, INFLATE + 1):
                    blocked.add((cx + dx, cy + dy))
        if free_around is not None:
            gx, gy = self.to_cell(free_around)
            for dx in range(-free_radius, free_radius + 1):
                for dy in range(-free_radius, free_radius + 1):
                    blocked.discard((gx + dx, gy + dy))
        return blocked

    # ------------------------------------------------------------------
    def plan(
        self, start_xy, goal_xy, blocked: set[tuple[int, int]]
    ) -> list[np.ndarray]:
        """
        8 邻域 A*。返回世界坐标航点列表（不含起点，含终点的精确坐标）。
        找不到路时退化为直连（让下游控制器自己撞运气，总比停死强）。
        """
        start = self.to_cell(start_xy)
        goal = self.to_cell(goal_xy)
        blocked = blocked - {start, goal}

        if start == goal:
            return [np.asarray(goal_xy[:2], dtype=float)]

        def h(c):
            return np.hypot(c[0] - goal[0], c[1] - goal[1])

        open_heap = [(h(start), 0.0, start)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        gscore = {start: 0.0}
        found = False
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur == goal:
                found = True
                break
            if g > gscore.get(cur, np.inf):
                continue
            for dx, dy in _NEIGHBORS:
                nxt = (cur[0] + dx, cur[1] + dy)
                if not self.in_bounds(nxt) or nxt in blocked:
                    continue
                # 禁止斜穿两个障碍夹成的对角缝
                if dx and dy:
                    if (cur[0] + dx, cur[1]) in blocked and \
                       (cur[0], cur[1] + dy) in blocked:
                        continue
                ng = g + np.hypot(dx, dy)
                if ng < gscore.get(nxt, np.inf) - 1e-9:
                    gscore[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))

        if not found:
            return [np.asarray(goal_xy[:2], dtype=float)]

        path = [goal]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        pts = [self.to_world(c) for c in path[1:]]
        pts = _shortcut(pts, blocked, self)
        if pts:
            pts[-1] = np.asarray(goal_xy[:2], dtype=float)  # 末点用精确坐标
        else:
            pts = [np.asarray(goal_xy[:2], dtype=float)]
        return pts


def _shortcut(pts, blocked, grid) -> list[np.ndarray]:
    """
    把 A* 的锯齿路径拉直：能直连就跳过中间点。
    栅格路径转折多，差速车走起来会不停原地转向，很慢也很难看。
    """
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _line_free(pts[i], pts[j], blocked, grid):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _line_free(a, b, blocked, grid) -> bool:
    n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / (grid.cell * 0.4)))
    for t in np.linspace(0.0, 1.0, n + 1):
        p = np.asarray(a) * (1 - t) + np.asarray(b) * t
        if grid.to_cell(p) in blocked:
            return False
    return True


if __name__ == "__main__":
    # 自检：中间横着一排障碍，A* 必须绕过去而不是穿过去
    g = GridRouter(np.zeros(2))
    obs = [np.array([x, 0.0]) for x in np.arange(-0.3, 0.31, 0.1)]
    blocked = g.blocked_cells(obs)
    path = g.plan(np.array([0.0, -0.5]), np.array([0.0, 0.5]), blocked)
    print(f"  障碍: y=0 处一排 7 个模块（膨胀后占 y∈[-1,1] 格）")
    print(f"  路径航点数 {len(path)}: "
          + " → ".join(f"({p[0]:+.2f},{p[1]:+.2f})" for p in path))
    assert all(g.to_cell(p) not in blocked for p in path), "路径穿过了障碍"
    assert max(abs(p[0]) for p in path) > 0.3, "没有绕开，直接穿过去了"
    print("  ✓ A* 正确绕开障碍")

    # 自检：目标点在障碍膨胀区内时，free_around 豁免必须生效
    blocked2 = g.blocked_cells(obs, free_around=np.array([0.0, 0.15]))
    p2 = g.plan(np.array([0.0, -0.5]), np.array([0.0, 0.15]), blocked2)
    print(f"  贴着装配体的目标: {len(p2)} 个航点，终点 "
          f"({p2[-1][0]:+.2f},{p2[-1][1]:+.2f})")
    assert np.allclose(p2[-1], [0.0, 0.15]), "终点不对"
    print("  ✓ free_around 豁免生效，能规划到贴着装配体的逼近起点")
