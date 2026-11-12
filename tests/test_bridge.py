"""
无头自检：不开窗口，验证仿真接口层的正确性。
运行: python -m tests.test_bridge
"""

import numpy as np

from planner.faces import W, Face, mate_relpose, wrap_angle
from sim_mujoco.bridge import MuJoCoBridge


def t1_drive_straight():
    """差速驱动: 给定 v 直行，位移应 ≈ v*t，横向漂移和转角应很小。"""
    br = MuJoCoBridge([(0.0, 0.0, 0.0)])
    v, T = 0.10, 2.0
    br.set_wheel_velocity(0, v, 0.0)
    br.step(int(T / br.model.opt.timestep))
    x, y, yaw = br.get_pose(0)
    print(f"  直行 {T}s @ {v}m/s → x={x:.4f} y={y:+.4f} yaw={np.rad2deg(yaw):+.2f}°")
    assert abs(x - v * T) < 0.03, f"前进距离偏差过大: {x} vs {v*T}"
    assert abs(y) < 0.02 and abs(yaw) < 0.15, "直行不应有明显横漂或偏航"
    print("  ✓ 差速直行正常")


def t2_turn_in_place():
    """原地转向: v=0, omega>0 应逆时针转，位置基本不动。"""
    br = MuJoCoBridge([(0.0, 0.0, 0.0)])
    om, T = 1.0, 1.5
    br.set_wheel_velocity(0, 0.0, om)
    br.step(int(T / br.model.opt.timestep))
    x, y, yaw = br.get_pose(0)
    print(f"  原地转 {T}s @ {om}rad/s → yaw={np.rad2deg(yaw):+.2f}° "
          f"(期望 {np.rad2deg(wrap_angle(om*T)):+.2f}°) 位移={np.hypot(x,y)*1000:.1f}mm")
    assert abs(wrap_angle(yaw - om * T)) < 0.25, "转角偏差过大"
    assert np.hypot(x, y) < 0.02, "原地转向不应有明显平移"
    print("  ✓ 原地转向正常（omega>0 为逆时针）")


def t3_weld_relpose():
    """
    关键验证: eq_data 的 relpose 布局是否正确。
    两个模块 TOP-TOP 对接后，相对位姿必须稳定保持在论文式(2) 的 [w, 0, π]。
    """
    br = MuJoCoBridge([(0.0, 0.0, 0.0), (0.35, 0.20, 1.1)])
    br.activate_weld(0, Face.TOP, 1, Face.TOP, snap=True)

    expect = mate_relpose(Face.TOP, Face.TOP)
    print(f"  期望相对位姿 p̃_ji = [{expect[0]:.3f}, {expect[1]:.3f}, "
          f"{np.rad2deg(expect[2]):.1f}°]")

    def rel_pose():
        p0, p1 = br.get_pose(0), br.get_pose(1)
        d = p1[:2] - p0[:2]
        c, s = np.cos(-p0[2]), np.sin(-p0[2])
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1],
                         wrap_angle(p1[2] - p0[2])])

    r0 = rel_pose()
    assert np.allclose(r0[:2], expect[:2], atol=2e-3), f"snap 后位置不对: {r0}"
    assert abs(wrap_angle(r0[2] - expect[2])) < 0.02, f"snap 后转角不对: {r0}"
    print(f"  snap 后           = [{r0[0]:.3f}, {r0[1]:.3f}, {np.rad2deg(r0[2]):.1f}°]")

    # 驱动整个装配体运动 3 秒，相对位姿必须保持住
    br.set_wheel_velocity(0, 0.12, 0.6)
    br.set_wheel_velocity(1, 0.05, -0.3)
    br.step(int(3.0 / br.model.opt.timestep))
    r1 = rel_pose()
    print(f"  驱动 3s 后        = [{r1[0]:.3f}, {r1[1]:.3f}, {np.rad2deg(r1[2]):.1f}°]")
    drift_pos = np.linalg.norm(r1[:2] - expect[:2])
    drift_ang = abs(wrap_angle(r1[2] - expect[2]))
    print(f"  漂移: 位置 {drift_pos*1000:.2f}mm, 角度 {np.rad2deg(drift_ang):.2f}°")
    assert drift_pos < 0.004, "weld 太软，装配体被拉开"
    assert drift_ang < 0.06, "weld 太软，装配体相对转动"
    print("  ✓ eq_data relpose 布局正确，weld 刚性足够")


def t4_dock_detection():
    """docking 条件判定: 逐渐靠近时误差应单调下降并最终进入接受域。"""
    # m1 放在 m0 的 TOP 面外侧一点点，姿态已对好（yaw=π 即 TOP 对 TOP）
    gap = 0.03
    br = MuJoCoBridge([(0.0, 0.0, 0.0), (W + gap, 0.0, np.pi)])
    d, a = br.dock_error(0, Face.TOP, 1, Face.TOP)
    print(f"  初始间隙 {gap*1000:.0f}mm → 面心距 {d*1000:.1f}mm, "
          f"法线偏差 {np.rad2deg(a):.2f}°")
    assert not br.can_dock(0, Face.TOP, 1, Face.TOP), "这个距离不该判定为可对接"

    # m1 的 yaw=π，其 TOP 面（体坐标 +x）已朝向 m0，
    # 因此「往前开」即可逼近 —— 正是论文 5.3.3 情况一：用 T/B 面直接推。
    docked = False
    for _ in range(3000):
        br.set_wheel_velocity(1, 0.03, 0.0)
        br.step()
        if br.try_dock(0, Face.TOP, 1, Face.TOP):
            docked = True
            break
    d2, a2 = br.dock_error(0, Face.TOP, 1, Face.TOP)
    print(f"  逼近 {br.time:.2f}s 后对接{'成功' if docked else '失败'} → "
          f"面心距 {d2*1000:.2f}mm, 法线偏差 {np.rad2deg(a2):.2f}°")
    assert docked, "逼近后应当成功对接"
    assert len(br.connections) == 1
    print("  ✓ docking 接受域判定 + 自动建立连接正常")


def t5_chain_of_three():
    """三模块链: 验证装配体逐级增长后仍然刚性（论文 Fig 4 的三模块例子规模）。"""
    br = MuJoCoBridge([(0, 0, 0), (0.3, 0.15, 2.0), (-0.3, -0.15, -1.0)])
    br.set_color(0, "root")
    br.activate_weld(1, Face.TOP, 0, Face.TOP)
    br.activate_weld(2, Face.TOP, 0, Face.BOTTOM)

    poses0 = br.get_all_poses()
    br.set_wheel_velocity(0, 0.10, 0.5)
    br.step(int(3.0 / br.model.opt.timestep))
    poses1 = br.get_all_poses()

    def rel(ps, i, j):
        d = ps[j][:2] - ps[i][:2]
        c, s = np.cos(-ps[i][2]), np.sin(-ps[i][2])
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1],
                         wrap_angle(ps[j][2] - ps[i][2])])

    moved = np.linalg.norm(poses1[0][:2] - poses0[0][:2])
    print(f"  装配体整体移动 {moved*1000:.0f}mm")
    # 注意: 只有 m0 在驱动，m1/m2 的轮子 ctrl=0（速度伺服锁零 ≈ 刹车），
    # 再加上轮子力矩被 forcerange 限幅，一个模块拖三个模块本来就慢。
    # 本用例考察的是**连接刚性**，不是速度，所以位移阈值取得很松。
    for (i, j) in [(0, 1), (0, 2)]:
        r0, r1 = rel(poses0, i, j), rel(poses1, i, j)
        # 角度差必须归一化：+π 与 −π 是同一朝向，直接相减会得到 360°
        dpos = np.linalg.norm(r1[:2] - r0[:2])
        dang = abs(wrap_angle(r1[2] - r0[2]))
        print(f"  m{i}-m{j} 相对漂移: {dpos*1000:.2f}mm, {np.rad2deg(dang):.2f}°")
        assert dpos < 0.004
        assert dang < 0.06
    assert moved > 0.005, "装配体应当被驱动起来（阈值放松，见上方注释）"
    print("  ✓ 三模块装配体作为整体运动，连接保持刚性")


if __name__ == "__main__":
    tests = [
        ("T1 差速直行", t1_drive_straight),
        ("T2 原地转向", t2_turn_in_place),
        ("T3 weld relpose 布局", t3_weld_relpose),
        ("T4 docking 判定", t4_dock_detection),
        ("T5 三模块装配体刚性", t5_chain_of_three),
    ]
    for name, fn in tests:
        print(f"\n[{name}]")
        fn()
    print("\n全部通过 ✓")
