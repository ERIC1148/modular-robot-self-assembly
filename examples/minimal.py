"""MuJoCo 最小例子：一个带轮小车 + 一个方块，演示建模、读写状态、执行器、接触、约束。"""
import mujoco
import numpy as np

XML = """
<mujoco model="minimal">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="cart" pos="0 0 0.05">
      <freejoint name="cart_free"/>
      <geom name="chassis" type="box" size="0.05 0.03 0.02" mass="0.3"/>
      <site name="front" pos="0.05 0 0" zaxis="1 0 0"/>
      <body name="wheel_L" pos="0 0.04 -0.02">
        <joint name="wl" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.03 0.005" zaxis="0 1 0" friction="1.2"/>
      </body>
      <body name="wheel_R" pos="0 -0.04 -0.02">
        <joint name="wr" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.03 0.005" zaxis="0 1 0" friction="1.2"/>
      </body>
      <geom type="sphere" size="0.008" pos="0.04 0 -0.04" friction="0.01"/>
    </body>
    <body name="box" pos="0.4 0 0.03">
      <freejoint/>
      <geom name="box_g" type="box" size="0.03 0.03 0.03" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vl" joint="wl" kv="0.5" forcerange="-0.2 0.2"/>
    <velocity name="vr" joint="wr" kv="0.5" forcerange="-0.2 0.2"/>
  </actuator>
  <equality>
    <weld name="grab" body1="cart" body2="box" active="false"/>
  </equality>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)   # 静态：编译后的模型
data = mujoco.MjData(model)                   # 动态：状态 + 中间计算结果

print("nq =", model.nq, " nv =", model.nv, " nu =", model.nu)  # 7+1+1+7=16, 6+1+1+6=14, 2

# 名字 -> id -> 在大数组里的地址
jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_free")
qadr = model.jnt_qposadr[jid]
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "front")
eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grab")

data.ctrl[:] = [5.0, 5.0]                      # 两轮同速 -> 直行
for step in range(1500):                       # 3 秒
    mujoco.mj_step(model, data)
    if data.eq_active[eq] == 0 and data.ncon:
        for k in range(data.ncon):             # 小车撞到方块就“焊上”
            c = data.contact[k]
            names = {model.geom(c.geom1).name, model.geom(c.geom2).name}
            if "box_g" in names and "floor" not in names:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, k, f)
                print(f"t={data.time:.2f}s 接触! 法向力 {f[0]:.2f} N -> 激活 weld")
                data.eq_active[eq] = 1
                break

print("小车位置 x =", round(data.qpos[qadr], 3))
print("前端 site 世界坐标 =", data.site_xpos[sid].round(3))
print("方块位置 x =", round(data.body('box').xpos[0], 3), "(随小车一起被带走)")
