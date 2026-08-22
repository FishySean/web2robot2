"""任务B 第二块：按片段那台相机把机器人渲出来（`src/web2robot/synth/render.py`）。

这个测试守的东西按重要性排序：

1. **改完 `model.cam_pos` 必须重算 `data.cam_xpos`**。定焦相机的全局位姿在 `data` 里，
   `mjv_updateScene` 读的是 `data`；写完 model 直接渲会拿**上一次 `mj_forward` 的相机**
   去渲 —— 第 0 帧渲成相机在世界原点（实测整屏 66% 落在 2.6 cm，从骨盆内部往外看），
   之后每帧都差一帧。静止底座看不出来，neural 求解器逐帧动底座就整段错。
   钉法：底座逐帧移动的合成轨迹上，`render(t)` 的结果**只由 t 决定**，与调用历史无关。
2. **渲出来的画面和解析投影必须是同一台相机**。手腕的解析投影点落在渲出来的机器人
   掩码里 —— 这一条同时验相机内参（fovy）、外参（位置/朝向）、OpenCV→MuJoCo 的轴翻转。
   任何一处错了，投影点就飘出机器人。
3. **主点偏心当场报错，不许默默近似**。MuJoCo 定焦相机的主点在图像正中；官方 10 段实测
   精确居中，将来遇到偏心的片段必须停下来（`CameraNotSupported`），不能硬渲。
4. **左右手看 `hand_meta.json`，不看槽位下标**。官方数据里确实有帧把槽 0 标成右手
   （`-0RheyDV3a0_474.8_487.3` 有 27 帧，见 BACKLOG B14）—— 按槽位当左右手会错手。
5. **背景是 `inf`，不是某个魔法阈值**。深度背景（远平面）转成 `inf`，`mask` 就是
   `isfinite(depth)`，下游不用记 `< 10.0` 这种数。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_synth_render -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")     # 无头机器：GLFW 起不来，egl 抛 EGLError

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.synth import render as R  # noqa: E402

CLIPS_DIR = REPO / "data" / "clips_official"

#: 合成相机：853×480、focal 966.56 —— 抄官方片段的实际内参，主点精确居中。
CAM = {"focal": 966.55872797966, "cx": 426.5, "cy": 240.0, "width": 853, "height": 480}


def write_camera(clip: Path, **over) -> dict:
    cam = dict(CAM)
    cam.update(over)
    clip.mkdir(parents=True, exist_ok=True)
    with open(clip / "camera.json", "w") as fh:
        json.dump(cam, fh)
    return cam


def synth_run(run: Path, n: int, moving: bool = True) -> None:
    """造一份最小的重定向产物：机器人正对相机、站在 1.7 m 处，可选逐帧横移。

    根位姿是"根系 → 相机系"（和 `root_frames.npz` 同一个口径）。基座朝向取
    `根 x（前）→ 相机 −z`、`根 z（上）→ 相机 −y`、`根 y（左）→ 相机 +x` —— 也就是
    "人正对镜头，他的左手出现在画面右侧"，再叠一个小角度免得矩阵太特殊。
    这样两只手腕都落在画面里（零位手臂下垂 14 cm、左右 ±24.5 cm），投影判据才有意义。
    """
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    ang = 0.15
    c, s = np.cos(ang), np.sin(ang)
    R_one = A @ np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    Rs = np.repeat(R_one[None], n, axis=0)
    ts = np.zeros((n, 3))
    ts[:] = (0.0, 0.0, 1.7)
    if moving:
        ts[:, 0] += np.linspace(0.0, 0.2, n)         # 逐帧横向平移 → 相机逐帧动
    run.mkdir(parents=True, exist_ok=True)
    np.savez(run / "root_frames.npz", R_per_frame=Rs, t_per_frame=ts)
    q = np.zeros((n, 7))
    q[:, 3] = np.linspace(-0.5, -1.2, n)             # 弯肘，别让两帧一模一样
    np.savez(run / "trajectory.npz", q_left=q, q_right=q.copy(), fps=20.0,
             clip_id=run.name, robot="m7")


def write_hand_meta(clip: Path, rows) -> None:
    with open(clip / "hand_meta.json", "w") as fh:
        json.dump({"is_right_per_frame": rows}, fh)


class TestCameraModel(unittest.TestCase):
    """内参换算 + "能不能用定焦相机表达"的门槛。"""

    def test_fovy_matches_pinhole(self):
        # focal = H/2 ⇒ 垂直视场角正好 90°
        self.assertAlmostEqual(
            R.fovy_degrees({"focal": 240.0, "height": 480}), 90.0, places=9)
        self.assertAlmostEqual(R.fovy_degrees(CAM), 27.88945448851599, places=9)

    def test_offcenter_principal_point_raises(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "c"
            write_camera(clip, cx=380.0)
            with self.assertRaises(R.CameraNotSupported):
                R.clip_camera(clip)

    def test_centered_principal_point_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "c"
            write_camera(clip)
            self.assertEqual(R.clip_camera(clip)["width"], 853)

    def test_missing_field_and_bad_focal_raise(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "c"
            clip.mkdir()
            with open(clip / "camera.json", "w") as fh:
                json.dump({"focal": 100.0, "cx": 50.0, "cy": 40.0, "width": 100}, fh)
            with self.assertRaises(R.CameraNotSupported):
                R.clip_camera(clip)
            write_camera(clip, focal=0.0)
            with self.assertRaises(R.CameraNotSupported):
                R.clip_camera(clip)


class TestRetargetLoaders(unittest.TestCase):
    """产物读进来就得验：形状、帧数、旋转矩阵是不是真的旋转。"""

    def test_loads_and_checks_frame_count(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "run"
            synth_run(run, 5)
            Rp, tp = R.load_root_frames(run, 5)
            self.assertEqual(Rp.shape, (5, 3, 3))
            self.assertEqual(tp.shape, (5, 3))
            with self.assertRaises(R.RetargetRunMissing):
                R.load_root_frames(run, 6)
            traj = R.load_joint_trajectory(run, 5)
            self.assertEqual(traj["q_left"].shape, (5, 7))
            self.assertEqual(traj["fps"], 20.0)
            self.assertIsNone(traj["q_left_fingers"])
            with self.assertRaises(R.RetargetRunMissing):
                R.load_joint_trajectory(run, 4)

    def test_missing_files_say_which_one(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "run"
            run.mkdir()
            with self.assertRaises(R.RetargetRunMissing):
                R.load_root_frames(run)
            with self.assertRaises(R.RetargetRunMissing):
                R.load_joint_trajectory(run)

    def test_non_rotation_matrix_is_rejected(self):
        """R 不正交（比如被谁按尺度缩过）会让相机整体错位，必须当场炸。"""
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "run"
            synth_run(run, 3)
            with np.load(run / "root_frames.npz") as z:
                Rs, ts = z["R_per_frame"].copy(), z["t_per_frame"].copy()
            Rs[1] *= 1.3
            np.savez(run / "root_frames.npz", R_per_frame=Rs, t_per_frame=ts)
            with self.assertRaises(R.RetargetRunMissing):
                R.load_root_frames(run, 3)


class TestHandSlots(unittest.TestCase):
    """左右手归属只认 `hand_meta.json`（BACKLOG B14）。"""

    def _clip(self, td, rows, joints3d, joints2d):
        clip = Path(td) / "c"
        write_camera(clip)
        write_hand_meta(clip, rows)
        np.asarray(joints3d, dtype=np.float32).tofile(clip / "hand_joints.bin")
        np.asarray(joints2d, dtype=np.float32).tofile(clip / "hand_joints_2d.bin")
        return clip

    def test_slot_zero_labelled_right_swaps_the_hands(self):
        n = 2
        j3 = np.zeros((n, 2, R.N_JOINTS, 3), dtype=np.float32)
        j2 = np.zeros((n, 2, R.N_JOINTS, 2), dtype=np.float32)
        j3[:, 0, R.WRIST_JOINT] = (0.1, 0.0, 2.0)     # 槽 0
        j3[:, 1, R.WRIST_JOINT] = (-0.1, 0.0, 2.0)    # 槽 1
        j2[:, 0, R.WRIST_JOINT] = (10.0, 20.0)
        j2[:, 1, R.WRIST_JOINT] = (30.0, 40.0)
        with tempfile.TemporaryDirectory() as td:
            # 第 0 帧是常规的"槽0=左、槽1=右"，第 1 帧官方把槽 0 标成了右手
            clip = self._clip(td, [[False, True], [True, False]], j3, j2)
            uv = R.official_wrist_uv(clip, 2)
            xyz = R.official_wrist_xyz(clip, 2)
            self.assertTrue(np.allclose(uv[0, 0], (10.0, 20.0)))   # 左 = 槽 0
            self.assertTrue(np.allclose(uv[1, 0], (30.0, 40.0)))   # 左 = 槽 1（换了）
            self.assertTrue(np.allclose(xyz[1, 1, 0], 0.1))        # 右手拿到槽 0 的 3D
            sides = R.hand_slot_sides(clip, 2)
            self.assertEqual(sides.tolist(), [[0, 1], [1, 0]])

    def test_absent_hand_stays_nan(self):
        n = 1
        j3 = np.zeros((n, 2, R.N_JOINTS, 3), dtype=np.float32)
        j2 = np.zeros((n, 2, R.N_JOINTS, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            clip = self._clip(td, [[False, None]], j3, j2)          # 右手不在场
            uv = R.official_wrist_uv(clip, 1)
            self.assertTrue(np.isfinite(uv[0, 0]).all())
            self.assertTrue(np.isnan(uv[0, 1]).all())

    def test_frame_count_mismatch_raises(self):
        n = 2
        j3 = np.zeros((n, 2, R.N_JOINTS, 3), dtype=np.float32)
        j2 = np.zeros((n, 2, R.N_JOINTS, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            clip = self._clip(td, [[False, True]] * n, j3, j2)
            with self.assertRaises(ValueError):
                R.official_wrist_uv(clip, 3)


class TestPoserGeometry(unittest.TestCase):
    """相机位姿换算（不渲染）。"""

    @classmethod
    def setUpClass(cls):
        cls.poser = R.RobotPoser("m7")

    def test_unknown_robot_is_refused_with_a_hint(self):
        with self.assertRaises(ValueError):
            R.RobotPoser("g1")

    def test_camera_pose_round_trips(self):
        """把世界系里的手腕点用返回的相机位姿变回相机系，必须等于直接算的相机系坐标。"""
        ang = 0.4
        c, s = np.cos(ang), np.sin(ang)
        R_cr = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        t_cr = np.array([0.05, 0.2, 1.9])
        self.poser.pose(np.zeros(7), np.zeros(7))
        R_w_c, p_w = self.poser.camera_pose_in_world(R_cr, t_cr)
        self.assertAlmostEqual(float(np.linalg.det(R_w_c)), 1.0, places=9)
        for side, bid in self.poser._wrist_ids.items():
            x_world = self.poser.data.xpos[bid]
            direct = self.poser.body_in_camera(bid, R_cr, t_cr)
            via_cam = R_w_c.T @ (x_world - p_w)
            self.assertTrue(np.allclose(direct, via_cam, atol=1e-9), side)

    def test_root_is_the_ik_chain_root(self):
        from web2robot.robots.m7 import CONFIG
        self.assertEqual(CONFIG["torso_body"], "waist_pitch_link")
        R_w_r, _ = self.poser.root_pose()
        self.assertAlmostEqual(float(np.linalg.det(R_w_r)), 1.0, places=9)


@unittest.skipUnless(
    (REPO / "assets" / "robots" / "m7" / "m7.xml").exists(), "M7 资产没同步")
class TestClipRobotRenderer(unittest.TestCase):
    """真渲。合成轨迹，不依赖 outputs/ 里的任何产物。"""

    N = 4

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        td = Path(cls._td.name)
        # 属性别叫 run —— TestCase.run 是跑测试用的方法，覆盖掉整个 unittest 就崩
        cls.run_dir = td / "run"
        synth_run(cls.run_dir, cls.N, moving=True)
        cls.Rp, cls.tp = R.load_root_frames(cls.run_dir, cls.N)
        cls.traj = R.load_joint_trajectory(cls.run_dir, cls.N)
        cls.rd = R.ClipRobotRenderer(CAM)

    @classmethod
    def tearDownClass(cls):
        cls.rd.close()
        cls._td.cleanup()

    def shoot(self, t: int):
        return self.rd.render(self.Rp[t], self.tp[t],
                              self.traj["q_left"][t], self.traj["q_right"][t])

    def test_render_shape_and_depth_units(self):
        f = self.shoot(0)
        self.assertEqual(f.rgb.shape, (CAM["height"], CAM["width"], 3))
        self.assertEqual(f.rgb.dtype, np.uint8)
        self.assertEqual(f.depth.shape, (CAM["height"], CAM["width"]))
        self.assertEqual(f.depth.dtype, np.float32)
        self.assertTrue(np.array_equal(f.mask, np.isfinite(f.depth)))
        self.assertTrue(f.mask.any(), "整帧没渲到机器人")
        d = f.depth[f.mask]
        # 底座在相机前 1.7 m；渲出来的深度必须在这个量级，不能是"相机在原点"的几厘米
        self.assertGreater(float(d.min()), 0.5)
        self.assertLess(float(d.max()), 5.0)
        self.assertTrue(np.isinf(f.depth[~f.mask]).all())

    def test_result_depends_only_on_t_not_on_call_history(self):
        """底座逐帧移动时，render(t) 与调用顺序无关 —— 这是那个"相机差一帧"的回归钉。"""
        first = self.shoot(0)
        _ = self.shoot(self.N - 1)
        again = self.shoot(0)
        self.assertTrue(np.array_equal(first.rgb, again.rgb), "第 0 帧和重渲的不一样")
        self.assertTrue(np.array_equal(np.nan_to_num(first.depth, posinf=-1.0),
                                       np.nan_to_num(again.depth, posinf=-1.0)))

    def test_moving_base_actually_moves_the_picture(self):
        """底座真在动，所以帧与帧必须不同 —— 否则上一条会因为"啥都没变"而空转。"""
        a, b = self.shoot(0), self.shoot(self.N - 1)
        self.assertFalse(np.array_equal(a.rgb, b.rgb))
        self.assertGreater(float(np.linalg.norm(a.wrist_uv["left"] - b.wrist_uv["left"])), 5.0)

    def test_analytic_wrist_projection_lands_on_the_rendered_robot(self):
        """解析投影和渲出来的画面是同一台相机 —— 内参/外参/轴翻转任何一处错都会飘出去。"""
        yy, xx = np.mgrid[0:CAM["height"], 0:CAM["width"]]
        for t in range(self.N):
            f = self.shoot(t)
            for side in ("left", "right"):
                u, v = f.wrist_uv[side]
                self.assertTrue(np.isfinite([u, v]).all(), f"t={t} {side} 投影没算出来")
                near = ((xx - u) ** 2 + (yy - v) ** 2) <= 20.0 ** 2
                frac = float((f.mask & near).sum()) / float(near.sum())
                self.assertGreater(frac, 0.5, f"t={t} {side} 手腕投影点周围没有机器人")

    def test_wrist_depth_matches_the_rendered_depth_there(self):
        """手腕的解析深度和那个像素渲出来的深度对得上（差的只是手腕在壳体内的几厘米）。"""
        f = self.shoot(1)
        for side in ("left", "right"):
            u, v = f.wrist_uv[side]
            z_rendered = f.depth[int(round(v)), int(round(u))]
            self.assertTrue(np.isfinite(z_rendered))
            self.assertLess(abs(float(z_rendered) - float(f.wrist_xyz[side][2])), 0.10, side)

    def test_camera_is_added_in_memory_only(self):
        """MJCF 资产一个字节都不能被改（磁盘上本来就没有 <camera>）。"""
        xml = (REPO / "assets" / "robots" / "m7" / "m7.xml").read_text()
        self.assertNotIn("<camera", xml)
        self.assertNotIn(R.CAMERA_NAME, xml)


@unittest.skipUnless((CLIPS_DIR / "-1r9yl-P-Ao_86.3_90.8" / "camera.json").exists(),
                     "官方片段没同步")
class TestRealClipInputs(unittest.TestCase):
    """真片段的成像模型确实能用定焦相机表达 —— 这条是整块渲染成立的前提。"""

    def test_every_official_clip_has_a_centered_principal_point(self):
        clips = sorted(p for p in CLIPS_DIR.iterdir() if (p / "camera.json").exists())
        self.assertGreater(len(clips), 5)
        for clip in clips:
            cam = R.clip_camera(clip)              # 偏心就会抛 CameraNotSupported
            self.assertEqual(float(cam["cx"]), float(cam["width"]) / 2, clip.name)
            self.assertEqual(float(cam["cy"]), float(cam["height"]) / 2, clip.name)
            self.assertGreater(R.fovy_degrees(cam), 10.0)
            self.assertLess(R.fovy_degrees(cam), 90.0)

    def test_some_clip_really_labels_slot_zero_as_right(self):
        """B14 那个矛盾是真的存在（不是我们读错）—— 哪天官方修了，这条会红，正好提醒。"""
        odd = {}
        for clip in sorted(p for p in CLIPS_DIR.iterdir() if (p / "hand_meta.json").exists()):
            with open(clip / "scene.json") as fh:
                n = int(json.load(fh)["stats"]["n_frames"])
            sides = R.hand_slot_sides(clip, n)
            k = int((sides[:, 0] == 1).sum())
            if k:
                odd[clip.name] = k
        self.assertTrue(odd, "没有任何片段把槽 0 标成右手 —— 去核对 BACKLOG B14 是否已过期")


if __name__ == "__main__":
    unittest.main()
