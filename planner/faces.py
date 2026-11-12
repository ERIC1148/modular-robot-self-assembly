"""
连接面约定 —— 引擎无关，整个项目的几何地基。

模块体坐标系（俯视，模块直立放在地面上）:

                  +y  LEFT (轮)
                   ↑
                   │
    BOTTOM ────────┼────────→ +x   TOP (前进方向)
      (-x)         │
                   ↓
                  -y  RIGHT (轮)

为什么 TOP/BOTTOM 指向前后、LEFT/RIGHT 是轮子？依据论文两处:

1. Sec 5.3.3「逼近」: c_i 是 TOP 或 BOTTOM 面时，模块直接往前开、推着对方
   完成对接; c_i 是 LEFT 或 RIGHT 面时必须借助 helping module。
   → LEFT/RIGHT 就是两个驱动轮所在的面，无法一边当连接器一边驱动自己。
   → TOP/BOTTOM 必须在前进方向上，才能靠"往前开"完成对接。

2. Sec 5.1 式(2) 举例: m_j 的 TOP 面接 m_i 的 TOP 面时 p̃_ji = [w, 0, π]。
   本模块的 mate_relpose(TOP, TOP) 必须复现这个结果 —— 见文件末尾自检。
"""

from enum import IntEnum

import numpy as np

# 模块边长 (m)。论文: 80-mm-wide cube
W = 0.08


class Face(IntEnum):
    """四个连接面。顺序按逆时针排列，便于取对侧面。"""

    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3


FACE_CHAR = {Face.TOP: "T", Face.LEFT: "L", Face.BOTTOM: "B", Face.RIGHT: "R"}
CHAR_FACE = {v: k for k, v in FACE_CHAR.items()}

# 每个面的外法线在模块体坐标系 xy 平面内的方位角
FACE_ANGLE = {
    Face.TOP: 0.0,
    Face.LEFT: np.pi / 2,
    Face.BOTTOM: np.pi,
    Face.RIGHT: -np.pi / 2,
}

# 面的外法线单位向量（3D，z 分量恒为 0 —— 平面自组装）
FACE_NORMAL = {
    f: np.array([np.cos(a), np.sin(a), 0.0]) for f, a in FACE_ANGLE.items()
}

# 面中心在模块体坐标系中的位置
FACE_POS = {f: (W / 2) * n for f, n in FACE_NORMAL.items()}

# 可以靠"往前开"直接完成对接的面（论文 5.3.3 情况一）
DRIVE_FACES = (Face.TOP, Face.BOTTOM)

# 轮子所在的面，用它对接需要 helping module（论文 5.3.3 情况二）
# 阶段一设计目标构型时应完全回避这两个面作为 c_i
WHEEL_FACES = (Face.LEFT, Face.RIGHT)


def opposite(face: Face) -> Face:
    """对侧面。helping module 动作 (m_H, T, m_i, c̃_i) 里的 c̃_i 就是它。"""
    return Face((int(face) + 2) % 4)


def wrap_angle(a: float) -> float:
    """
    把角度归一化到半开区间 (-π, π]。

    注意是 (-π, π] 而非 [-π, π): 论文式(2) 的 TOP-TOP 例子给的是 +π，
    用常见的 [-π, π) 写法会归一化成 -π，数值等价但对不上论文，容易看糊涂。
    """
    return -np.mod(-a + np.pi, 2 * np.pi) + np.pi


def mate_relpose(face_i: Face, face_j: Face) -> np.ndarray:
    """
    模块 i 的 face_i 与模块 j 的 face_j 完成对接时，
    j 相对 i 的平面位姿 (x, y, yaw)。即论文式(2) 里的 p̃_ji。

    推导（全部在 i 的体坐标系内）:
      - 两模块面贴面 ⇒ 中心距 = W，方向沿 face_i 的外法线
            pos = W * [cos a_i, sin a_i]
      - 两个面的外法线必须反向平行 ⇒ a_j + yaw = a_i + π
            yaw = a_i + π - a_j
    """
    a_i = FACE_ANGLE[face_i]
    a_j = FACE_ANGLE[face_j]
    return np.array(
        [W * np.cos(a_i), W * np.sin(a_i), wrap_angle(a_i + np.pi - a_j)]
    )


def compose(base: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """
    base ⊕ rel：把「base 体坐标系下的相对位姿 rel」换算到世界坐标系。
    这就是论文式(2) 做的事，只是写成了通用形式。
    """
    c, s = np.cos(base[2]), np.sin(base[2])
    return np.array(
        [
            base[0] + c * rel[0] - s * rel[1],
            base[1] + s * rel[0] + c * rel[1],
            wrap_angle(base[2] + rel[2]),
        ]
    )


def needs_helping_module(face_i: Face) -> bool:
    """移动模块 m_i 用 face_i 去对接时是否需要 helping module。"""
    return face_i in WHEEL_FACES


if __name__ == "__main__":
    # 对着论文 Sec 5.1 的例子自检: TOP 面接 TOP 面 ⇒ p̃_ji = [w, 0, π]
    got = mate_relpose(Face.TOP, Face.TOP)
    assert np.allclose(got, [W, 0.0, np.pi]), got
    print(f"✓ mate_relpose(TOP, TOP) = {got}  == 论文式(2) 的 [w, 0, π]")

    # 另外三种也顺手验一下几何自洽性
    for fi in Face:
        for fj in Face:
            x, y, yaw = mate_relpose(fi, fj)
            # j 的 face_j 外法线转到 i 的坐标系后，应当与 face_i 的外法线反向
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            n_j_in_i = R @ FACE_NORMAL[fj][:2]
            assert np.allclose(n_j_in_i, -FACE_NORMAL[fi][:2], atol=1e-12)
            # 两个面的中心应当重合
            assert np.allclose(
                FACE_POS[fi][:2], np.array([x, y]) + R @ FACE_POS[fj][:2], atol=1e-12
            )
    print("✓ 全部 16 种面配对的法线反向 & 面心重合校验通过")
