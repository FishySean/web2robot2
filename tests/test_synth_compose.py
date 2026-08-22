"""任务B 第三块：抠人 → 补背景 → 按深度贴机器人（`src/web2robot/synth/compose.py`）。

这个测试守的东西按重要性排序：

1. **深度排序里那两条"看着像放水"的例外必须成立**。被抠掉的人形区域内无条件画机器人
   （那块地方 `depth.npz` 存的是**人手自己的深度**，人已经擦掉了，拿它挡机器人等于让
   一个不存在的东西遮住机器人）；场景深度无效（0）的像素也不参与判遮挡（深度未知就不
   能判）。这两条各有一个用例 —— 少了任何一条，机器人会被自己要接替的那只手挡住。
2. **顺序是先抠人再贴机器人**。反过来的话人形掩码会把刚贴上的机器人手擦掉一块 ——
   两者本来就大面积重叠。钉法：机器人掩码 ∩ 人形掩码那块像素，合成结果必须是机器人。
3. **两路深度的单位换算**。官方 `depth.npz` 是 uint16 **毫米**、0 = 没测出来；机器人渲
   出来的是**米**。差一千倍的话，机器人会整体跑到所有物体前面（或后面），而单帧看着
   "只是有点怪"，很容易被误判成渲染问题。
4. **人形掩码取左右手的并集，不按名字取单只手**。官方有 34 帧把右手存在槽 0
   （BACKLOG **B14**）—— 并集对这个错误免疫，按名字取会取错手。钉法：把左右两份掩码
   互换，并集逐字节不变。
5. **背景板两条路各自的前提**。相机不动 → 时间中值能把扫过的手完全消掉（中值不是均值：
   均值会留淡影）；整段都被挡住的像素中值给不出值，得 `cv2.inpaint` 补，不能留黑洞。
   相机在动 → 逐帧 inpaint，且**掩码外的像素逐字节不动**。
6. **没有真 RGB 时要明确报错，不许悄悄拿深度替身顶上**（`RgbMissing`），
   而用了替身时清单里的 `rgb_source` 必须写明白 —— 否则替身底图会被当成验收依据。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_synth_compose -v
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

from web2robot.synth import compose as C  # noqa: E402
from web2robot.synth.render import RobotFrame  # noqa: E402

#: 合成相机：小一号（64×48）好算，主点精确居中 —— `clip_camera` 的前提。
CAM = {"focal": 100.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48}
H, W = CAM["height"], CAM["width"]


def write_clip(clip: Path, n: int, depth_mm: np.ndarray = None) -> None:
    """造一个最小片段：`camera.json` + `scene.json`（+ 可选 `depth.npz`）。"""
    clip.mkdir(parents=True, exist_ok=True)
    with open(clip / "camera.json", "w") as fh:
        json.dump(CAM, fh)
    with open(clip / "scene.json", "w") as fh:
        json.dump({"stats": {"n_frames": n}}, fh)
    if depth_mm is not None:
        np.savez(clip / "depth.npz", depth=depth_mm.astype(np.uint16))


def write_hand_masks(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    """按官方 `masks.npz` 的口径按位打包存 —— 和 `cli.masks_npz` 写的是同一种文件。"""
    n = len(left)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, left=np.stack([np.packbits(m.ravel()) for m in left]),
             right=np.stack([np.packbits(m.ravel()) for m in right]),
             shape=np.array([n, H, W]))


def robot_frame(depth_m: np.ndarray, color=(10, 200, 30)) -> RobotFrame:
    """一帧假的渲染结果：`depth_m` 里 `inf` 就是背景（`mask` 由它派生）。"""
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[np.isfinite(depth_m)] = color
    nan2 = np.full(2, np.nan)
    return RobotFrame(rgb=rgb, depth=depth_m.astype(np.float32),
                      wrist_uv={"left": nan2, "right": nan2.copy()},
                      wrist_xyz={"left": np.zeros(3), "right": np.zeros(3)})


class TestDepthOrdering(unittest.TestCase):
    """机器人和场景谁挡谁 —— 连同那两条例外。"""

    def setUp(self):
        # 机器人占左半幅，距相机 2 m
        self.rdepth = np.full((H, W), np.inf, dtype=np.float32)
        self.rdepth[:, :W // 2] = 2.0
        self.frame = robot_frame(self.rdepth)
        self.none_erased = np.zeros((H, W), dtype=bool)

    def test_scene_in_front_hides_robot(self):
        scene = np.full((H, W), 1.0, dtype=np.float32)          # 场景全在 1 m，更近
        vis = C.robot_visible(self.frame, scene, self.none_erased)
        self.assertFalse(vis.any())

    def test_scene_behind_shows_robot(self):
        scene = np.full((H, W), 5.0, dtype=np.float32)
        vis = C.robot_visible(self.frame, scene, self.none_erased)
        np.testing.assert_array_equal(vis, self.frame.mask)

    def test_erased_region_always_draws_robot(self):
        """人被抠掉的地方，`depth.npz` 存的是那只手的深度 —— 不能拿它挡机器人。"""
        scene = np.full((H, W), 1.0, dtype=np.float32)          # 手在 1 m，比机器人近
        erased = np.zeros((H, W), dtype=bool)
        erased[:, :10] = True                                    # 手在最左边一条
        vis = C.robot_visible(self.frame, scene, erased)
        self.assertTrue(vis[:, :10].all())                       # 抠掉的那条：画机器人
        self.assertFalse(vis[:, 10:].any())                      # 其余仍被场景挡住

    def test_invalid_scene_depth_shows_robot(self):
        """`depth.npz` 里的 0（没测出来）→ nan → 不参与判遮挡，否则黑洞会啃掉机器人。"""
        scene = np.full((H, W), 1.0, dtype=np.float32)
        scene[:, :10] = np.nan
        vis = C.robot_visible(self.frame, scene, self.none_erased)
        self.assertTrue(vis[:, :10].all())
        self.assertFalse(vis[:, 10:].any())

    def test_tolerance_favours_robot_when_depths_tie(self):
        scene = np.full((H, W), 2.0, dtype=np.float32)           # 和机器人一样远
        self.assertTrue(C.robot_visible(self.frame, scene, self.none_erased)[:, :W // 2].all())
        scene[:] = 2.0 - 2 * C.DEPTH_TOL_M                        # 明确更近，超出容差
        self.assertFalse(C.robot_visible(self.frame, scene, self.none_erased).any())

    def test_no_scene_depth_means_robot_on_top(self):
        vis = C.robot_visible(self.frame, None, self.none_erased)
        np.testing.assert_array_equal(vis, self.frame.mask)

    def test_override_fraction_counts_only_flipped_pixels(self):
        """那条例外有代价（人形掩码溢到桌沿上，机器人就盖住桌沿）—— 所以先把面积量出来。"""
        scene = np.full((H, W), 5.0, dtype=np.float32)            # 场景比机器人远
        erased = np.zeros((H, W), dtype=bool)
        erased[:, :10] = True
        # 场景更远 ⇒ 机器人本来就该画 ⇒ 例外没起作用
        self.assertEqual(C.override_fraction(self.frame, scene, erased), 0.0)
        scene[:, :10] = 1.0                                       # 抠掉那条里场景更近了
        n_robot = int(self.frame.mask.sum())
        self.assertAlmostEqual(C.override_fraction(self.frame, scene, erased),
                               H * 10 / n_robot, places=9)
        self.assertEqual(C.override_fraction(self.frame, None, erased), 0.0)


class TestComposeFrame(unittest.TestCase):
    """一帧合成：先抠人、后贴机器人，两步的先后不能反。"""

    def setUp(self):
        self.rgb = np.full((H, W, 3), 200, dtype=np.uint8)
        self.plate = np.full((H, W, 3), 50, dtype=np.uint8)
        self.rdepth = np.full((H, W), np.inf, dtype=np.float32)
        self.rdepth[10:20, 10:20] = 2.0
        self.frame = robot_frame(self.rdepth, color=(10, 200, 30))

    def test_person_replaced_by_plate(self):
        erased = np.zeros((H, W), dtype=bool)
        erased[30:40, 30:40] = True                              # 和机器人不重叠
        out, _ = C.compose_frame(self.rgb, self.plate, erased, self.frame, None)
        np.testing.assert_array_equal(out[30:40, 30:40], self.plate[30:40, 30:40])
        np.testing.assert_array_equal(out[0:5, 0:5], self.rgb[0:5, 0:5])

    def test_robot_wins_inside_erased_region(self):
        """人形掩码和机器人重叠的地方必须是机器人 —— 顺序反了这里就成背景板。"""
        erased = np.zeros((H, W), dtype=bool)
        erased[5:25, 5:25] = True                                # 完全盖住机器人那块
        out, vis = C.compose_frame(self.rgb, self.plate, erased, self.frame, None)
        np.testing.assert_array_equal(out[10:20, 10:20],
                                      np.broadcast_to(np.array([10, 200, 30], np.uint8),
                                                      (10, 10, 3)))
        self.assertTrue(vis[10:20, 10:20].all())
        np.testing.assert_array_equal(out[5:10, 5:10], self.plate[5:10, 5:10])

    def test_input_frame_not_mutated(self):
        rgb0 = self.rgb.copy()
        erased = np.ones((H, W), dtype=bool)
        C.compose_frame(self.rgb, self.plate, erased, self.frame, None)
        np.testing.assert_array_equal(self.rgb, rgb0)


class TestSceneDepth(unittest.TestCase):
    """毫米 → 米，0 → nan。差一千倍的错误在单帧上看不出来，所以在这里钉住。"""

    def test_millimetres_to_metres_and_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "c"
            mm = np.zeros((3, H, W), dtype=np.uint16)
            mm[:] = 1500                                          # 1.5 m
            mm[:, 0, 0] = 0                                       # 没测出来
            write_clip(clip, 3, depth_mm=mm)
            d = C.scene_depth_m(clip, 3)
            self.assertEqual(d.dtype, np.float32)
            self.assertAlmostEqual(float(d[0, 5, 5]), 1.5, places=6)
            self.assertTrue(np.isnan(d[:, 0, 0]).all())

    def test_missing_file_returns_none_and_count_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "c"
            write_clip(clip, 3)
            self.assertIsNone(C.scene_depth_m(clip, 3))
            np.savez(clip / "depth.npz", depth=np.zeros((2, H, W), np.uint16))
            with self.assertRaises(ValueError):
                C.scene_depth_m(clip, 3)


class TestPersonMask(unittest.TestCase):
    """人形掩码：并集、膨胀、尺寸核对，以及"只有手"这件事要说出来。"""

    def _clip(self, td, n=3):
        clip = Path(td) / "c"
        write_clip(clip, n)
        return clip

    def test_union_of_both_hands(self):
        with tempfile.TemporaryDirectory() as td:
            clip = self._clip(td)
            left = np.zeros((3, H, W), dtype=bool)
            right = np.zeros((3, H, W), dtype=bool)
            left[:, 5:10, 5:10] = True
            right[:, 20:25, 20:25] = True
            p = Path(td) / "hand_masks.npz"
            write_hand_masks(p, left, right)
            m, src = C.load_person_mask(clip, 3, CAM, hand_masks=p, dilate_px=0)
            self.assertTrue(m[:, 5:10, 5:10].all())
            self.assertTrue(m[:, 20:25, 20:25].all())
            self.assertAlmostEqual(float(m.mean()), 50 / (H * W), places=9)
            self.assertIn("只有手", src)                          # 别让调用方误会是整个人

    def test_union_is_immune_to_swapped_slots(self):
        """B14：官方有帧把右手存在槽 0。并集对左右互换逐字节不变，按名字取就会取错手。"""
        with tempfile.TemporaryDirectory() as td:
            clip = self._clip(td)
            left = np.zeros((3, H, W), dtype=bool)
            right = np.zeros((3, H, W), dtype=bool)
            left[:, 5:10, 5:10] = True
            right[:, 20:25, 20:25] = True
            a, b = Path(td) / "a.npz", Path(td) / "b.npz"
            write_hand_masks(a, left, right)
            write_hand_masks(b, right, left)                      # 两只手换个槽
            ma, _ = C.load_person_mask(clip, 3, CAM, hand_masks=a, dilate_px=0)
            mb, _ = C.load_person_mask(clip, 3, CAM, hand_masks=b, dilate_px=0)
            np.testing.assert_array_equal(ma, mb)

    def test_dilate_grows_mask(self):
        with tempfile.TemporaryDirectory() as td:
            clip = self._clip(td)
            left = np.zeros((3, H, W), dtype=bool)
            left[:, 20:25, 20:25] = True
            p = Path(td) / "hand_masks.npz"
            write_hand_masks(p, left, np.zeros_like(left))
            tight, _ = C.load_person_mask(clip, 3, CAM, hand_masks=p, dilate_px=0)
            wide, _ = C.load_person_mask(clip, 3, CAM, hand_masks=p, dilate_px=3)
            self.assertGreater(wide.sum(), tight.sum())
            self.assertTrue(wide[tight].all())                    # 只长不缩

    def test_shape_and_count_mismatch_raise(self):
        with tempfile.TemporaryDirectory() as td:
            clip = self._clip(td)
            left = np.zeros((3, H, W), dtype=bool)
            p = Path(td) / "hand_masks.npz"
            write_hand_masks(p, left, left.copy())
            with self.assertRaises(ValueError):
                C.load_person_mask(clip, 4, CAM, hand_masks=p)     # 帧数不对
            with self.assertRaises(ValueError):
                C.load_person_mask(clip, 3, dict(CAM, height=99), hand_masks=p)
            with self.assertRaises(FileNotFoundError):
                C.load_person_mask(clip, 3, CAM, hand_masks=Path(td) / "nope.npz")

    def test_external_mask_dir(self):
        """外部分割器（SAM3 / 内部服务）产的掩码 —— 接口不锁死在某个模型上。"""
        import cv2
        with tempfile.TemporaryDirectory() as td:
            clip = self._clip(td)
            md = Path(td) / "masks"
            md.mkdir()
            for t in range(3):
                img = np.zeros((H, W), dtype=np.uint8)
                img[10:20, 10:20] = 255
                cv2.imwrite(str(md / f"{t:05d}.png"), img)
            m, src = C.load_person_mask(clip, 3, CAM, person_dir=md, dilate_px=0)
            self.assertTrue(m[:, 10:20, 10:20].all())
            self.assertAlmostEqual(float(m.mean()), 100 / (H * W), places=9)
            self.assertIn("外部人形掩码", src)
            with self.assertRaises(ValueError):
                C.load_person_mask(clip, 5, CAM, person_dir=md)


class TestBackgroundPlate(unittest.TestCase):
    """背景板两条路：相机不动用时间中值，相机在动逐帧 inpaint。"""

    def _moving_hand(self, n=9, static_bg=True):
        rng = np.random.default_rng(0)
        bg = rng.integers(0, 200, (H, W, 3), dtype=np.uint8)
        rgb = np.empty((n, H, W, 3), dtype=np.uint8)
        mask = np.zeros((n, H, W), dtype=bool)
        for t in range(n):
            frame = bg.copy() if static_bg else np.roll(bg, t * 7, axis=1)
            x = 4 + t * 5
            frame[10:20, x:x + 8] = 255                            # 一只白"手"横着扫
            mask[t, 10:20, x:x + 8] = True
            rgb[t] = frame
        return rgb, mask, bg

    def test_median_plate_removes_the_hand(self):
        rgb, mask, bg = self._moving_hand()
        plate, seen = C.median_plate(rgb, mask)
        self.assertTrue(seen.all())                                # 每个像素都露过脸
        np.testing.assert_array_equal(plate, bg)                   # 手被完全消掉

    def test_median_beats_mean_on_swept_pixels(self):
        """中值和均值的差别不是风格问题：均值会在扫过的地方留一道淡影。"""
        rgb, mask, bg = self._moving_hand()
        plate, _ = C.median_plate(rgb, mask)
        mean = rgb.mean(axis=0).round().astype(np.uint8)
        swept = mask.any(axis=0)
        self.assertEqual(int(np.abs(plate[swept].astype(int) - bg[swept].astype(int)).max()), 0)
        self.assertGreater(int(np.abs(mean[swept].astype(int) - bg[swept].astype(int)).max()), 0)

    def test_block_size_does_not_change_result(self):
        rgb, mask, _ = self._moving_hand()
        a, sa = C.median_plate(rgb, mask, block=64)
        b, sb = C.median_plate(rgb, mask, block=7)                 # 分块只为省内存
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(sa, sb)

    def test_always_covered_pixels_are_inpainted_not_black(self):
        rgb, mask, _ = self._moving_hand()
        mask[:, 30:34, 30:34] = True                               # 整段都被挡住
        rgb[:, 30:34, 30:34] = 255
        plate, seen = C.median_plate(rgb, mask)
        self.assertFalse(seen[30:34, 30:34].any())
        self.assertEqual(int(plate[30:34, 30:34].max()), 0)         # 中值给不出值 → 0
        filled = C.fill_holes(plate, seen)
        self.assertGreater(int(filled[30:34, 30:34].max()), 0)      # inpaint 补上了
        np.testing.assert_array_equal(filled[seen], plate[seen])    # 有观测的地方不动

    def test_inpaint_plate_leaves_unmasked_pixels_byte_identical(self):
        rgb, mask, _ = self._moving_hand(static_bg=False)
        plate = C.inpaint_plate(rgb, mask)
        self.assertEqual(plate.shape, rgb.shape)
        np.testing.assert_array_equal(plate[~mask], rgb[~mask])
        self.assertTrue((plate[mask] != rgb[mask]).any())

    def test_auto_picks_median_for_static_camera(self):
        rgb, mask, bg = self._moving_hand(static_bg=True)
        plate, info = C.background_plate(rgb, mask, mode="auto")
        self.assertEqual(info["mode"], "median")
        self.assertEqual(len(plate), 1)                             # 整段共用一块板
        self.assertLess(info["motion_score"], C.STATIC_MOTION_THRESH)
        np.testing.assert_array_equal(plate[0], bg)

    def test_auto_picks_inpaint_for_moving_camera(self):
        rgb, mask, _ = self._moving_hand(static_bg=False)
        plate, info = C.background_plate(rgb, mask, mode="auto")
        self.assertEqual(info["mode"], "inpaint")
        self.assertEqual(len(plate), len(rgb))                      # 逐帧一块
        self.assertGreater(info["motion_score"], C.STATIC_MOTION_THRESH)

    def test_explicit_mode_overrides_the_guess(self):
        """相机动不动这件事第②步路由已经在判 —— 给了标签就不该再靠猜。"""
        rgb, mask, _ = self._moving_hand(static_bg=False)
        _, info = C.background_plate(rgb, mask, mode="median")
        self.assertEqual(info["mode"], "median")
        self.assertNotIn("motion_score", info)
        with self.assertRaises(ValueError):
            C.background_plate(rgb, mask, mode="nonesuch")


class TestRgbSource(unittest.TestCase):
    """底图从哪来：没有真画面就明确报错，用了替身就写清楚是替身。"""

    def test_auto_without_rgb_raises(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "no-such-clip-id"
            write_clip(clip, 3)
            with self.assertRaises(C.RgbMissing) as cm:
                C.load_rgb("auto", clip, 3, CAM)
            self.assertIn("B12", str(cm.exception))                 # 指到待拍板项

    def test_depth_standin_is_labelled(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "c"
            mm = np.tile(np.linspace(500, 3000, W).astype(np.uint16), (3, H, 1))
            write_clip(clip, 3, depth_mm=mm)
            rgb, src = C.load_rgb("depth", clip, 3, CAM)
            self.assertEqual(rgb.shape, (3, H, W, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])  # 灰度铺三通道
            self.assertIn("替身", src)                               # 别当成真画面
            self.assertIn("不是真画面", src)

    def test_depth_standin_without_depth_npz_raises(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "c"
            write_clip(clip, 3)
            with self.assertRaises(C.RgbMissing):
                C.load_rgb("depth", clip, 3, CAM)

    def test_image_dir_and_size_check(self):
        import cv2
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "c"
            write_clip(clip, 3)
            d = Path(td) / "frames"
            d.mkdir()
            for t in range(3):
                cv2.imwrite(str(d / f"{t:05d}.png"), np.full((H, W, 3), 30 * t, np.uint8))
            rgb, src = C.load_rgb(str(d), clip, 3, CAM)
            self.assertEqual(rgb.shape, (3, H, W, 3))
            self.assertEqual(int(rgb[1].min()), 30)
            self.assertEqual(src, str(d))
            with self.assertRaises(ValueError):
                C.load_rgb(str(d), clip, 3, dict(CAM, width=99))     # 尺寸和内参对不上
            with self.assertRaises(ValueError):
                C.load_rgb(str(d), clip, 5, CAM)                     # 帧数对不上
            with self.assertRaises(C.RgbMissing):
                C.load_rgb(str(Path(td) / "nope"), clip, 3, CAM)


class TestMontage(unittest.TestCase):
    """核对图三列并排 —— 只看合成结果分不清"机器人贴歪"和"人没抠干净"。"""

    def test_three_columns_side_by_side(self):
        import cv2
        from web2robot.synth.cli import compose_montage
        with tempfile.TemporaryDirectory() as td:
            row = [np.full((H, W, 3), v, np.uint8) for v in (10, 20, 30)]
            png = Path(td) / "check.png"
            compose_montage([row, row], png, ["t=0", "t=5"])
            img = cv2.imread(str(png))
            self.assertEqual(img.shape, (2 * H, 3 * W, 3))


class TestCliWiring(unittest.TestCase):
    """子命令挂上去了、默认值来自 compose 模块（不许两处各写一个数）。"""

    def test_compose_subcommand_defaults(self):
        from web2robot.synth.cli import build_parser
        args = build_parser().parse_args(["compose", "data/clips_official",
                                          "--runs_dir", "outputs/retarget/collcmp"])
        self.assertEqual(args.rgb, "auto")
        self.assertEqual(args.plate, "auto")
        self.assertEqual(args.dilate, C.DEFAULT_DILATE_PX)
        self.assertEqual(args.depth_tol, C.DEPTH_TOL_M)
        self.assertFalse(args.no_depth_order)
        self.assertEqual(args.out, "outputs/synth/compose")

    def test_resolve_runs_shared_with_render(self):
        """`render` 和 `compose` 必须用同一份"哪段配哪份产物"的口径。"""
        from web2robot.synth.cli import build_parser, resolve_runs
        ap = build_parser()
        for cmd in ("render", "compose"):
            args = ap.parse_args([cmd, "clips", "--run", "cid=/tmp/run"])
            self.assertEqual(resolve_runs(args), {"cid": Path("/tmp/run")})
            with self.assertRaises(ValueError):
                resolve_runs(ap.parse_args([cmd, "clips", "--run", "no-equals-sign"]))
            with self.assertRaises(ValueError):
                resolve_runs(ap.parse_args([cmd, "clips"]))


if __name__ == "__main__":
    unittest.main()
