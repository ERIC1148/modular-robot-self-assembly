"""
生成 SMORES-EP 简化模块的 MJCF 模型。

论文 Sec 3 硬件描述（4 DOF: LEFT/RIGHT/PAN/TILT）:
  - LEFT DOF、RIGHT DOF、PAN DOF 均可连续旋转，用于扭转连接口相对模块本体的朝向；
    其中 LEFT/RIGHT 兼作驱动轮，差速驱动模块移动
  - TILT DOF 限位 ±90°，产生一个弯折关节

阶段一（已实现）: 只建了 LEFT/RIGHT 两个驱动轮，机体是完全刚性的单一 box。
阶段二（本次新增）: 在 TOP 面接出 TILT→PAN 两关节子链，机体本身保持刚性不变。

  设计依据（这不是论文给出的精确机械图纸，是根据上面这段功能描述做的工程实现，
  论文引用的 Knaian 博士论文和 Tosun et al. (2016) 里应该有更精确的机械细节，
  本项目没有那两份原始资料，所以下面的关节顺序、转轴方向、子链长度都是自行设计的，
  仅保证功能上符合"PAN 连续扭转连接口、TILT ±90° 弯折"这个描述）:
    机体(刚性，含左右轮) → TILT关节(弯折 ±90°) → PAN关节(连续旋转) → TOP 连接面

  TILT 在前、PAN 在后: TILT 负责把 TOP 连接口从水平面"掰"出去（这是让平面装配体
  最终能折成三维形态的关键自由度）；PAN 的转轴设定为"沿着连接口当前指向的方向"，
  即 PAN 关节定义在 TILT 之后的子body里，这样无论 TILT 转到多少度，PAN 拧的
  始终是连接口自己的轴线，符合论文"扭转连接口"的描述。

  ⚠️ 关键约束: TILT=PAN=0 时，TOP 连接面的世界坐标必须与「阶段一没有这两个关节
  时」逐位一致 —— 现有的式(2)展开、匈牙利分配、docking 判定全部依赖 TOP 面的
  精确几何，这条底线由 tests/test_wrist.py 做回归验证。

  ⚠️ 已知限制（暂未解决，留给后续）: 现在的 weld 焊接的是两个模块的**机体**
  (body "m{i}")，不是 TOP 面所在的手腕子链末端。这意味着：如果一个模块的 TOP
  面已经通过 weld 焊死，此后再驱动它自己的 TILT/PAN，物理上并不会像真实磁吸
  连接那样「连接口不动、机体转」，而是会与既有的 weld 约束打架（类似报告06记录
  的 snap 方向 bug）。所以「装配完成后主动折叠」这个功能，还需要把 TOP 面对接
  时的 weld 挂接点从机体改成手腕子链的末端 body，这一步本次没有做，仅新增了
  关节本身、并确保它们在阶段一的装配流程中保持零位、不引入任何回归。

EP 电永磁体 → 不建模磁场，用「距离+姿态阈值 → 激活 weld 约束」代替（不变）。
4 个连接面 → 用 site 表示（无碰撞体积，只提供位置+朝向），
site 的 z 轴 = 该面的外法线（不变）。

尺寸（论文: 80mm 立方体）:
    W = 0.08          模块边长
    轮半径 = W/2      ⇒ 轴心离地 0.04，模块正好占据 z ∈ [0, 0.08]
"""

from __future__ import annotations

import numpy as np

from planner.faces import FACE_CHAR, FACE_POS, W, Face

# ---- 结构尺寸 (m) ----
WHEEL_R = W / 2  # 轮半径 = 0.040，轴心离地正好 W/2
WHEEL_TH = 0.007  # 轮半厚
WHEEL_Y = W / 2 - WHEEL_TH  # 轮中面 y = 0.033，外缘正好 0.040
CHASSIS = (W / 2, 0.026, 0.034)  # 机体半尺寸；x 方向占满 W，使 T/B 面齐平
CASTER_R = 0.006  # 万向支撑球半径
CASTER_X = 0.033  # 支撑球前后位置
# ⚠️ 支撑球必须离地留间隙，只在模块前后俯仰时才触地。
# 若让它和轮子同高（z=-0.034），支撑球会分担载荷，与轮子形成过约束接触，
# 差速驱动被严重拖慢，且提高 kv 会让模块直接倒着跑。实测详见 报告/03。
CASTER_Z = -0.030  # ⇒ 最低点离地 4mm

# ---- TILT/PAN 手腕子链 ----
# 论文: TILT DOF 限位 ±90°，产生弯折关节。
TILT_LIMIT = np.pi / 2
# 手腕关节的小球几何体半径。这两个 body（tilt_link、pan_link）本身必须有非零
# 质量/惯量 —— MuJoCo 要求每个带关节的运动体都有正定惯量，否则编译报错
# "mass and inertia of moving bodies must be positive"（site 没有质量，
# 光挂 TOP site 在 pan_link 上不够）。半径取得很小，contype=0/conaffinity=0
# 保证它永不参与碰撞，纯粹是为了满足编译器的物理量要求，同时兼作可视化的
# "关节标记点"。
WRIST_R = 0.008

# ---- 碰撞位掩码 ----
# ⚠️ 已对接的两个模块之间**必须关闭碰撞**，否则 weld 约束与接触约束会互相打架:
# L/R 面就是轮子，对接后邻居机体与轮子恰好贴面，微小穿透就让两组约束往复振荡，
# 能量经地面接触传播，会把场上其他模块一起弹飞（实测远处模块被弹到 1.5 m/s）。
# 方案: 模块 i 的 contype = 1<<i，conaffinity 初值 MASK_ALL|FLOOR_BIT；
# 对接时互相清掉对方的位，即可精确关闭这一对的碰撞，不影响它与地面和其他模块。
#
# ⚠️ 地面必须单独占一个位（FLOOR_BIT），不能混用模块位。
# 否则模块与 0 号模块对接、清掉 bit0 之后，会连地面碰撞一起丢掉，模块直接穿地。
MASK_ALL = 0x000FFFFF  # 模块位 0..19，最多支持 20 个模块
FLOOR_BIT = 1 << 20

# 模块状态配色 —— 让并行分层的过程一眼可见
STATE_COLORS = {
    "idle": (0.55, 0.55, 0.58, 1.0),  # 灰   待命
    "navigate": (0.20, 0.45, 0.85, 1.0),  # 蓝   导航中
    "adjust": (0.95, 0.75, 0.15, 1.0),  # 黄   位姿调整中
    "approach": (0.95, 0.45, 0.10, 1.0),  # 橙   逼近中
    "docked": (0.20, 0.70, 0.35, 1.0),  # 绿   已对接
    "root": (0.80, 0.20, 0.25, 1.0),  # 红   根模块
}

# 连接面配色: 可驱动对接的 T/B 用青色，轮子面 L/R 用品红（提醒需 helping module）
FACE_SITE_COLORS = {
    Face.TOP: (0.10, 0.85, 0.85, 1.0),
    Face.BOTTOM: (0.10, 0.85, 0.85, 1.0),
    Face.LEFT: (0.90, 0.20, 0.80, 1.0),
    Face.RIGHT: (0.90, 0.20, 0.80, 1.0),
}


def _module_xml(idx: int, pose: tuple[float, float, float]) -> str:
    """单个模块的 body 段。pose = (x, y, yaw)。"""
    x, y, yaw = pose
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
    r, g, b, a = STATE_COLORS["idle"]

    # LEFT/BOTTOM/RIGHT 三个面仍固定在刚性机体上，写法和之前完全一样
    rigid_sites = []
    for f in (Face.LEFT, Face.BOTTOM, Face.RIGHT):
        px, py, pz = FACE_POS[f]
        n = FACE_POS[f] / np.linalg.norm(FACE_POS[f])
        fr, fg, fb, fa = FACE_SITE_COLORS[f]
        rigid_sites.append(
            f'      <site name="m{idx}_{FACE_CHAR[f]}" pos="{px:.4f} {py:.4f} {pz:.4f}" '
            f'zaxis="{n[0]:.0f} {n[1]:.0f} {n[2]:.0f}" '
            f'size="0.010 0.010 0.003" type="box" '
            f'rgba="{fr} {fg} {fb} {fa}"/>'
        )
    rigid_sites_xml = "\n".join(rigid_sites)

    # TOP 面 site 挪到手腕子链末端（pan_link 内），坐标改成相对 pan_link 原点。
    # ⚠️ pan_link 的原点就是手腕子链在机体上的挂载点 FACE_POS[TOP]，所以这里
    # 写 pos="0 0 0" —— 当 tilt=pan=0 时，这个 site 的世界坐标与「挂在刚性机体
    # 上、坐标为 FACE_POS[TOP]」完全等价，靠子body的位移把差值补回来了。
    tfr, tfg, tfb, tfa = FACE_SITE_COLORS[Face.TOP]
    top_site = (
        f'          <site name="m{idx}_{FACE_CHAR[Face.TOP]}" pos="0 0 0" zaxis="1 0 0" '
        f'size="0.010 0.010 0.003" type="box" rgba="{tfr} {tfg} {tfb} {tfa}"/>'
    )

    # 每个模块占一个碰撞位，运行时对接后互相清位以关闭该对碰撞
    cm = f'contype="{1 << idx}" conaffinity="{MASK_ALL | FLOOR_BIT}"'

    # 手腕子链挂载点 = 刚性机体上原来 TOP 面的位置
    wx, wy, wz = FACE_POS[Face.TOP]

    return f"""    <body name="m{idx}" pos="{x:.4f} {y:.4f} {WHEEL_R:.4f}" quat="{qw:.6f} 0 0 {qz:.6f}">
      <freejoint name="m{idx}_free"/>
      <geom name="m{idx}_chassis" class="chassis" size="{CHASSIS[0]} {CHASSIS[1]} {CHASSIS[2]}"
            rgba="{r} {g} {b} {a}" {cm}/>
      <geom name="m{idx}_caster_f" class="caster" pos="{CASTER_X} 0 {CASTER_Z}" {cm}/>
      <geom name="m{idx}_caster_b" class="caster" pos="{-CASTER_X} 0 {CASTER_Z}" {cm}/>
{rigid_sites_xml}
      <body name="m{idx}_wheel_L" pos="0 {WHEEL_Y} 0">
        <joint name="m{idx}_wl" class="wheel"/>
        <geom name="m{idx}_gwl" class="wheel" {cm}/>
      </body>
      <body name="m{idx}_wheel_R" pos="0 {-WHEEL_Y} 0">
        <joint name="m{idx}_wr" class="wheel"/>
        <geom name="m{idx}_gwr" class="wheel" {cm}/>
      </body>
      <body name="m{idx}_tilt_link" pos="{wx:.4f} {wy:.4f} {wz:.4f}">
        <joint name="m{idx}_tilt" class="tilt"/>
        <geom name="m{idx}_wrist_tilt" class="wrist"/>
        <body name="m{idx}_pan_link" pos="0 0 0">
          <joint name="m{idx}_pan" class="pan"/>
          <geom name="m{idx}_wrist_pan" class="wrist"/>
{top_site}
        </body>
      </body>
    </body>"""


def build_xml(
    poses: list[tuple[float, float, float]],
    ghost_poses: list[tuple[float, float, float]] | None = None,
) -> str:
    """
    poses:       每个模块的初始 (x, y, yaw)
    ghost_poses: 可选，目标位姿「幽灵框」—— 展开图算出的停车位，
                 半透明、不参与碰撞，纯粹给人看
    """
    n = len(poses)
    modules = "\n".join(_module_xml(i, p) for i, p in enumerate(poses))

    # 幽灵框：目标停车位
    ghosts = ""
    if ghost_poses:
        gs = []
        for i, (x, y, yaw) in enumerate(ghost_poses):
            qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
            gs.append(
                f'    <body name="ghost{i}" pos="{x:.4f} {y:.4f} {WHEEL_R:.4f}" '
                f'quat="{qw:.6f} 0 0 {qz:.6f}">\n'
                f'      <geom name="ghost{i}_g" class="ghost" '
                f'size="{W/2} {W/2} {W/2}"/>\n'
                f'      <site name="ghost{i}_T" pos="{W/2} 0 0" zaxis="1 0 0" '
                f'size="0.008 0.008 0.002" type="box" rgba="0.1 0.9 0.9 0.35"/>\n'
                f"    </body>"
            )
        ghosts = "\n" + "\n".join(gs)

    # ---- 预声明 weld 约束: 每个无序模块对一条，初始全部关闭 ----
    # 论文的 docking 在算法层就是「条件满足 → 建立刚性连接」，
    # 这里用 equality/weld 实现。relpose 在运行时按面配对写入 eq_data。
    welds = "\n".join(
        # ⚠️ solref 不能太硬。取 0.004（仅为时间步的 2 倍）时，未对接模块撞上
        # 装配体会被弹到 1 m/s。0.012 仍然足够刚（实测漂移 <0.05mm），且稳定得多。
        f'    <weld name="w_{i}_{j}" body1="m{i}" body2="m{j}" active="false"'
        f' solref="0.012 1" solimp="0.9 0.98 0.001"/>'
        for i in range(n)
        for j in range(i + 1, n)
    )

    return f"""<mujoco model="smores_selfassembly">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
  <size njmax="4000" nconmax="1500"/>

  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.20 0.23 1"/>
    <global azimuth="140" elevation="-35" offwidth="1920" offheight="1080"/>
    <map znear="0.01"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.25 0.30 0.38" rgb2="0.06 0.07 0.10" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.24 0.26 0.29" rgb2="0.30 0.32 0.35" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="20 20" texuniform="true"
              reflectance="0.05"/>
    <material name="rubber" rgba="0.13 0.13 0.15 1" specular="0.2" shininess="0.1"/>
  </asset>

  <default>
    <default class="chassis">
      <geom type="box" friction="0.6 0.005 0.0001" density="900"/>
    </default>
    <default class="wheel">
      <joint type="hinge" axis="0 1 0" damping="0.004" armature="1e-5"/>
      <geom type="cylinder" size="{WHEEL_R} {WHEEL_TH}" zaxis="0 1 0"
            material="rubber" friction="1.2 0.02 0.001" density="1100"/>
    </default>
    <default class="caster">
      <!-- 低摩擦支撑球，防止两轮车前后翻倒；摩擦调到很小以免干扰差速运动 -->
      <geom type="sphere" size="{CASTER_R}" friction="0.002 0.001 0.0001"
            density="200" rgba="0.2 0.2 0.22 1"/>
    </default>
    <default class="ghost">
      <!-- 目标停车位幽灵框：不参与碰撞 -->
      <geom type="box" contype="0" conaffinity="0" group="2"
            rgba="0.35 0.85 0.45 0.13"/>
    </default>
    <default class="tilt">
      <!-- 弯折关节：轴沿本地 y，转动发生在 x-z 平面内，把 TOP 连接口
           从水平面"掰"出去。论文: TILT DOF 限位 ±90°。 -->
      <joint type="hinge" axis="0 1 0" range="{-TILT_LIMIT:.7f} {TILT_LIMIT:.7f}"
             damping="0.02" armature="1e-5"/>
    </default>
    <default class="pan">
      <!-- 扭转关节：轴沿本地 x，即挂载它的 body（tilt_link）当前朝向的方向 ——
           这个方向会随 TILT 转动而变化，所以 PAN 拧的始终是"连接口自己此刻
           指向哪"，符合论文"扭转连接口"的描述，不受 TILT 角度影响。
           不设 range ⇒ 连续旋转，无限位，对应论文 PAN DOF 的连续旋转特性。 -->
      <joint type="hinge" axis="1 0 0" damping="0.02" armature="1e-5"/>
    </default>
    <default class="wrist">
      <!-- 手腕关节的可视化标记 + 惯量占位，contype=0/conaffinity=0 永不参与碰撞。
           半径极小，颜色和机体、轮子、连接面都不同，纯粹是为了在画面里能
           看出"这里有个关节"。 -->
      <geom type="sphere" size="{WRIST_R}" contype="0" conaffinity="0"
            density="300" rgba="0.85 0.85 0.90 1"/>
    </default>
  </default>

  <worldbody>
    <light name="key" pos="0 0 2.5" dir="0 0 -1" directional="true"
           diffuse="0.55 0.55 0.55"/>
    <geom name="floor" type="plane" size="4 4 0.05" material="grid"
          friction="0.9 0.005 0.0001"
          contype="{FLOOR_BIT}" conaffinity="{MASK_ALL}"/>
{ghosts}
{modules}
  </worldbody>

  <actuator>
{chr(10).join(
    # ⚠️ forcerange 不能省。速度伺服若不限力矩，轮子一被卡住误差就 ≈2.5 rad/s，
    # 力矩 = kv×误差 = 5 N·m，除以 4cm 轮半径 = 125 N 推力作用在 0.41kg 的模块上
    # （≈300 m/s²），打滑释放瞬间会把模块弹射出去（实测飞出作业区 2.5 m）。
    # 模块自重 0.41kg、摩擦系数 1.2 ⇒ 可用牵引力上限 ≈4.8 N ⇒ 每轮约 0.1 N·m。
    # 取 ±0.12 N·m：够到摩擦极限，又不会凭空造出物理上不存在的推力。
    f'    <velocity name="a_m{i}_{s}" joint="m{i}_w{s}" kv="2.0" '
    f'ctrlrange="-40 40" forcerange="-0.12 0.12"/>'
    for i in range(n) for s in ("l", "r")
)}
{chr(10).join(
    # TILT/PAN 用 position（位置伺服）而不是 velocity：折叠动作天然是
    # "转到某个角度"，不是"以某个速度转"，这点和轮子的驱动方式不同。
    #
    # ⚠️ forcerange 同样不能省 —— 报告06 记录过轮子伺服不限力矩时，
    # 关节被卡住的瞬间会输出远超物理真实值的力矩，把模块弹飞。
    # 这里的数值是按物理量级估算的（论文没有给出 PAN/TILT 的力矩规格）:
    #   模块自重 ≈0.41kg ⇒ 重力 ≈4.0N；TILT 万一要吊起最多两三个模块的
    #   悬臂重量，力臂按约 1.5 倍模块边长(0.12m)估，需要力矩量级
    #   3×4.0N×0.12m ≈ 1.4 N·m，取 forcerange=±1.5 N·m 留一点余量。
    #   PAN 只负责扭转、不承重，力矩需求小得多，forcerange 取 ±0.5 N·m。
    # 这两个数字都是估算值，不是论文给出的规格，真正接入折叠控制器、
    # 有实际负载工况后应当重新标定。
    f'    <position name="a_m{i}_tilt" joint="m{i}_tilt" kp="3.0" '
    f'ctrlrange="{-TILT_LIMIT:.7f} {TILT_LIMIT:.7f}" forcerange="-1.5 1.5"/>\n'
    f'    <position name="a_m{i}_pan" joint="m{i}_pan" kp="1.5" '
    f'forcerange="-0.5 0.5"/>'
    for i in range(n)
)}
  </actuator>

  <equality>
{welds}
  </equality>
</mujoco>
"""


if __name__ == "__main__":
    import mujoco

    poses = [(0.0, 0.0, 0.0), (0.30, 0.10, 2.0), (-0.25, -0.20, -1.0)]
    xml = build_xml(poses, ghost_poses=[(0.0, 0.0, 0.0), (0.08, 0.0, np.pi)])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"✓ 模型编译通过: {model.nbody} bodies, {model.nq} qpos, "
          f"{model.nu} actuators, {model.neq} weld 约束")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "m0")
    total = model.body_subtreemass[bid]
    print(f"  单模块总质量 = {total*1000:.0f} g（含两轮）")
