"""
离屏渲染录制演示视频。

用的是 mujoco.Renderer（离屏），**不需要 mjpython**，也不开窗口，
所以能在无图形界面的环境里跑。

用法:
    python -m sim_mujoco.record --task ten
    python -m sim_mujoco.record --task task2 --speed 2
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np

from .runner_full import TASKS, AssemblyRunner

FPS = 30


def record(task: str, out: str, speed: float, width: int, height: int,
           max_time: float, tail: float):
    name, topo, poses = TASKS[task]
    print(f"### 录制 {name} ###")
    runner = AssemblyRunner(topo, poses)
    br = runner.br
    dt = br.model.opt.timestep

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 90.0, -68.0
    cam.distance = 1.45
    cam.lookat[:] = [0.0, 0.0, 0.04]

    scn = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scn)
    # 显示幽灵框（目标停车位）所在的 geom group 2
    scn.geomgroup[2] = 1

    steps_per_frame = max(1, int(round(speed / (FPS * dt))))
    renderer = mujoco.Renderer(br.model, height=height, width=width)

    n_frames = 0
    with imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=1) as w:
        def grab():
            nonlocal n_frames
            renderer.update_scene(br.data, camera=cam, scene_option=scn)
            w.append_data(renderer.render())
            n_frames += 1

        grab()  # 首帧：散落的初始状态
        while not runner.finished and br.time < max_time:
            for _ in range(steps_per_frame):
                runner.tick()
                if runner.finished:
                    break
            grab()
            if n_frames % 60 == 0:
                print(f"  t={br.time:6.1f}s  批次 {runner.batch + 1}/"
                      f"{len(runner.schedule)}  已录 {n_frames} 帧")

        # 结尾多停几秒展示成品
        for _ in range(int(tail * FPS)):
            for _ in range(steps_per_frame):
                br.stop_all()
                br.step()
            grab()

    renderer.close()
    runner.report()
    print(f"\n  视频已保存: {out}")
    print(f"  {n_frames} 帧 @ {FPS}fps ≈ {n_frames/FPS:.1f}s"
          f"（{speed:g}× 速）  {width}×{height}")
    return runner


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="ten")
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="倍速")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-time", type=float, default=300.0)
    ap.add_argument("--tail", type=float, default=3.0, help="结尾展示成品的秒数")
    a = ap.parse_args()
    out = a.out or f"media/demo-{a.task}.mp4"
    record(a.task, out, a.speed, a.width, a.height, a.max_time, a.tail)
