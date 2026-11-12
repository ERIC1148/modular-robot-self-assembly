"""
TILT/PAN 手腕关节的回归与功能验证。
运行: python -m tests.test_wrist
"""

import numpy as np

from planner.faces import FACE_POS, Face
from sim_mujoco.bridge import MuJoCoBridge
from sim_mujoco.model_gen import WHEEL_R


def t1_zero_pose_matches_rigid_face():
    """
    最关键的回归测试: TILT=PAN=0 时，TOP 面的世界坐标必须和
    「机体直接挂 site，没有手腕子链」时逐位一致。

    这是不能破的底线 —— 阶段一已跑通的展开/分配/调度/docking 全部依赖
    TOP 面在零位时的精确几何，如果这里对不上，后面全部作废。
    """
    br = MuJoCoBridge([(0.13, -0.27, 0.9)])  # 随便一个非零位姿，更能测出问题
    pos, normal = br.face_pose(0, Face.TOP)

    x, y, yaw = 0.13, -0.27, 0.9
    c, s = np.cos(yaw), np.sin(yaw)
    fx, fy, fz = FACE_POS[Face.TOP]
    expect_pos = np.array([x + c * fx - s * fy, y + s * fx + c * fy, WHEEL_R])
    expect_normal = np.array([c, s, 0.0])  # TOP 法线在体系下是 (1,0,0)

    print(f"  TOP 面世界坐标: {pos.round(6).tolist()}")
    print(f"  期望坐标        : {expect_pos.round(6).tolist()}")
    assert np.allclose(pos, expect_pos, atol=1e-6), "TILT=PAN=0 时 TOP 面位置对不上！"
    assert np.allclose(normal, expect_normal, atol=1e-6), "TILT=PAN=0 时 TOP 面法线对不上！"
    print("  ✓ TILT=PAN=0 时 TOP 面几何与阶段一（无手腕关节）完全一致")


def t2_tilt_limit_enforced():
    """TILT 目标角超出 ±90° 时应被限幅，且实际转动能到达边界附近。"""
    br = MuJoCoBridge([(0.0, 0.0, 0.0)])
    br.set_tilt_target(0, np.deg2rad(150))  # 故意给一个超限的目标
    for _ in range(4000):
        br.step()
    tilt = br.get_tilt(0)
    print(f"  下达 150° 目标，clip 后实际收敛到 {np.rad2deg(tilt):.2f}°"
          f"（期望接近 90°）")
    assert abs(tilt - np.pi / 2) < 0.05, "TILT 限位没有生效"
    print("  ✓ TILT ±90° 限位生效")


def t3_pan_continuous():
    """PAN 应能连续转动超过一整圈，不受限位约束。"""
    br = MuJoCoBridge([(0.0, 0.0, 0.0)])
    target = np.deg2rad(540)  # 一圈半，超过 ±π 也应该能到
    br.set_pan_target(0, target)
    for _ in range(4000):
        br.step()
    pan = br.get_pan(0)
    print(f"  下达 540°（一圈半）目标，实际转到 {np.rad2deg(pan):.2f}°")
    assert abs(pan - target) < 0.05, "PAN 应能连续转动超过一圈"
    print("  ✓ PAN 连续旋转、无限位")


def t4_tilt_lifts_top_out_of_plane():
    """
    功能验证: TILT 转到 90° 后，TOP 面的法线应当从水平变为竖直 ——
    这是让平面装配体最终能折成三维形态的关键能力。

    ⚠️ 第一次跑这个测试时，我原本断言"TOP 面的位置应该大幅升高"，结果失败了
    （实测只挪了 5mm，而不是预想的几厘米）。排查后发现这不是 bug，是我对
    自己设计的儿何想当然了: TOP site 就定义在 pan_link 的原点，而 pan_link
    的原点恰好就是 TILT 关节的转动轴心 —— 转动轴心自己转动，位置当然不变，
    只有朝向在变。这是"零长度手腕子链"这个设计选择的直接推论: TILT/PAN
    在这个实现里只改变连接口的朝向，不产生大幅度的位置摆动。如果以后要做
    "手臂真的甩出一大截"那种视觉效果，需要在 TILT 和连接口之间插入一段有
    长度的连杆，这个暂时没做，留作后续。这里改成验证朝向变化，这是当前
    设计下真正应该发生、也确实发生的事。
    """
    br = MuJoCoBridge([(0.0, 0.0, 0.0)])
    pos0, normal0 = br.face_pose(0, Face.TOP)

    br.set_tilt_target(0, np.pi / 2)
    for _ in range(4000):
        br.step()
    pos1, normal1 = br.face_pose(0, Face.TOP)

    print(f"  TILT 0°→90°: TOP 面法线从 {normal0.round(3).tolist()} "
          f"变为 {normal1.round(3).tolist()}")
    print(f"  位置从 {pos0.round(4).tolist()} 变为 {pos1.round(4).tolist()}"
          f"（位移 {np.linalg.norm(pos1-pos0)*1000:.1f}mm，"
          f"预期很小 —— 连接口就在转轴上）")
    assert abs(normal0[2]) < 0.05, "零位时法线应接近水平"
    assert abs(normal1[2]) > 0.9, "TILT 到 90° 后法线应接近竖直，TOP 面被折出水平面"
    print("  ✓ TILT 能把 TOP 面的朝向从水平折到竖直，具备三维折叠所需的自由度")


def t5_pan_axis_follows_tilt():
    """
    验证设计意图: PAN 的转轴应「跟随」TILT ——即先把 TILT 转到 90°，
    此时再转 PAN，转的应该是已经竖起来的那根轴，而不是原来水平的轴。
    用这个来确认 model_gen.py 里"PAN 定义在 TILT 之后的子body"这个
    嵌套顺序，在 MuJoCo 里的实际语义符合设计预期。
    """
    br = MuJoCoBridge([(0.0, 0.0, 0.0)])
    br.set_tilt_target(0, np.pi / 2)
    for _ in range(4000):
        br.step()
    _, normal_before_pan = br.face_pose(0, Face.TOP)

    br.set_pan_target(0, np.pi / 2)  # 转 90° 的"扭转"
    for _ in range(4000):
        br.step()
    _, normal_after_pan = br.face_pose(0, Face.TOP)

    print(f"  TILT=90° 时 PAN 转动前法线: {normal_before_pan.round(3).tolist()}")
    print(f"  TILT=90° 时 PAN 转动后法线: {normal_after_pan.round(3).tolist()}")
    # PAN 是绕法线自身转，所以转动前后法线方向应该几乎不变
    # （如果 PAN 轴没有跟随 TILT，转 90° 后法线会被带偏，明显偏离竖直）
    assert np.allclose(normal_before_pan, normal_after_pan, atol=0.05), (
        "PAN 应该绕着连接口自己的法线转，转动前后法线方向不应改变 —— "
        "如果这里对不上，说明 PAN 的轴没有正确跟随 TILT"
    )
    print("  ✓ PAN 转轴正确跟随 TILT：扭转不改变连接口的指向，只改变绕轴的自转角")


def t6_zero_state_does_not_break_docking():
    """
    集成回归: 两模块 TOP-TOP 对接，全程不碰 TILT/PAN（阶段一场景），
    结果应与加关节之前完全一样 —— 复用报告06里 t3 的验证逻辑。
    """
    from planner.faces import mate_relpose, wrap_angle

    br = MuJoCoBridge([(0.0, 0.0, 0.0), (0.35, 0.20, 1.1)])
    br.activate_weld(0, Face.TOP, 1, Face.TOP, snap=True)
    expect = mate_relpose(Face.TOP, Face.TOP)

    def rel_pose():
        p0, p1 = br.get_pose(0), br.get_pose(1)
        d = p1[:2] - p0[:2]
        c, s = np.cos(-p0[2]), np.sin(-p0[2])
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1],
                         wrap_angle(p1[2] - p0[2])])

    r0 = rel_pose()
    assert np.allclose(r0[:2], expect[:2], atol=2e-3)
    assert abs(wrap_angle(r0[2] - expect[2])) < 0.02
    br.set_wheel_velocity(0, 0.12, 0.6)
    br.set_wheel_velocity(1, 0.05, -0.3)
    br.step(1500)
    r1 = rel_pose()
    drift = np.linalg.norm(r1[:2] - expect[:2])
    print(f"  加了 TILT/PAN 关节后，TOP-TOP 对接漂移: {drift*1000:.3f}mm"
          f"（阶段一原始结果是 0.00mm）")
    assert drift < 0.004, "加关节后不应影响阶段一的对接精度"
    print("  ✓ 阶段一 docking 精度未受影响，新关节是纯增量、无回归")


if __name__ == "__main__":
    tests = [
        ("T1 零位几何回归", t1_zero_pose_matches_rigid_face),
        ("T2 TILT 限位", t2_tilt_limit_enforced),
        ("T3 PAN 连续旋转", t3_pan_continuous),
        ("T4 TILT 折出平面", t4_tilt_lifts_top_out_of_plane),
        ("T5 PAN 轴跟随 TILT", t5_pan_axis_follows_tilt),
        ("T6 对接精度无回归", t6_zero_state_does_not_break_docking),
    ]
    for name, fn in tests:
        print(f"\n[{name}]")
        fn()
    print("\n全部通过 ✓")
