"""任务B 第一块：MANO 手部网格 → 手部掩码（`src/web2robot/synth/`）。

这个测试守的东西按重要性排序：

1. **"投影出来就能用"是错的，必须逐帧对齐。** 造一支合成片段，让 2D 关节 = 投影
   结果做一个已知的 (s,tx,ty) 变换，`frame_alignments()` 必须解回那三个数；
   对齐后的掩码盖住 2D 关节，不对齐的盖不住。真实数据实测裸投影和官方
   `hand_joints_2d.bin` 差 9.3 px 中位、逐帧缩放在 0.32–0.95 漂 —— 少了这一步，
   合成时贴上去的机器人手会整体错位。
2. **形状靠算不靠信。** `.bin` 里没有形状信息，面片数/顶点索引/帧数任何一处对不上
   都得当场报错。读错一段的数据不会自己现形，只会静默产出"另一段的手"。
3. **不在场的手是 NaN，不是 0。** NaN 的整只手必须整片丢掉；要是被当成
   (0,0,0) 投影，掩码里会凭空多一只贴在画面某处的手。
4. **判据自己得分得清好坏。** `joints_inside_fraction()` 分开报指尖和非指尖 ——
   指尖骨节点本来就在网格表面上或略微在外（实测落在外面的离边界中位 2 px），
   混在一起算会把"几何本来如此"和"掩码错位"搅成一个数。真实数据这条钉住：
   对齐后非指尖 ≥0.9，且明显高于不对齐。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_synth_handmask -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.synth import handmask as hm  # noqa: E402

CLIPS_DIR = REPO / "data" / "clips_official"

CAM = {"focal": 100.0, "cx": 50.0, "cy": 40.0, "width": 100, "height": 80}
#: 合成片段里 2D 关节相对投影结果的真值变换
TRUE_S, TRUE_TX, TRUE_TY = 0.5, 20.0, -5.0


def square_mesh(z: float = 2.0, half: float = 0.2):
    """一块正对相机的正方形补片，顶点/面片数凑成 MANO 的规格。

    只有前 4 个顶点有用（正方形四角），面片头两片铺满它，剩下 1536 片是同一片的
    副本 —— 数目要对得上 `N_MANO_FACES`，形状不重要。用 `CAM` 投影出来正好落在
    u∈[40,60]、v∈[30,50]。
    """
    verts = np.zeros((hm.N_MANO_VERTS, 3), dtype=np.float32)
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    for i, (x, y) in enumerate(corners):
        verts[i] = (x, y, z)
    verts[4:] = verts[0]
    faces = np.zeros((hm.N_MANO_FACES, 3), dtype=np.int32)
    faces[0] = (0, 1, 2)
    faces[1] = (0, 2, 3)
    faces[2:] = (0, 1, 2)
    return verts, faces


def joint_grid(z: float = 2.0, half: float = 0.12) -> np.ndarray:
    """21 个落在正方形内部的三维关节点（网格排布，避免共线导致拟合退化）。"""
    xs = np.linspace(-half, half, 7)
    ys = np.linspace(-half, half, 3)
    pts = [(x, y, z) for y in ys for x in xs]
    return np.asarray(pts[:hm.N_JOINTS], dtype=np.float32)


def make_clip(root: Path, n_frames: int = 3, absent_hand: int | None = None,
              s: float = TRUE_S, tx: float = TRUE_TX, ty: float = TRUE_TY) -> Path:
    """造一段合成片段目录：2D 关节 = 投影结果 × s + t，是拟合要解回的真值。

    `absent_hand` 给了就把那只手整段写成 NaN（顶点和 3D/2D 关节都是）—— 官方口径
    是"不在场用 NaN"，`hand_meta.json` 的 `nan_means_absent` 写着。
    """
    clip = root / "synthclip_0.0_1.0"
    clip.mkdir(parents=True, exist_ok=True)
    with open(clip / "camera.json", "w") as fh:
        json.dump(CAM, fh)
    with open(clip / "scene.json", "w") as fh:
        json.dump({"n_frames": n_frames}, fh)

    verts1, faces = square_mesh()
    j3_1 = joint_grid()
    verts = np.tile(verts1, (n_frames, 2, 1, 1)).astype(np.float32)
    j3 = np.tile(j3_1, (n_frames, 2, 1, 1)).astype(np.float32)
    uv = hm.project_points(j3, CAM)
    j2 = np.stack([s * uv[..., 0] + tx, s * uv[..., 1] + ty], axis=-1).astype(np.float32)
    if absent_hand is not None:
        verts[:, absent_hand] = np.nan
        j3[:, absent_hand] = np.nan
        j2[:, absent_hand] = np.nan

    verts.tofile(clip / "hand_verts.bin")
    faces.tofile(clip / "hand_faces.bin")
    j3.tofile(clip / "hand_joints.bin")
    j2.tofile(clip / "hand_joints_2d.bin")
    return clip


class TestProjection(unittest.TestCase):
    def test_on_axis_point_lands_on_principal_point(self):
        uv = hm.project_points(np.array([0.0, 0.0, 1.5]), CAM)
        self.assertAlmostEqual(uv[0], CAM["cx"], places=6)
        self.assertAlmostEqual(uv[1], CAM["cy"], places=6)

    def test_known_offset(self):
        uv = hm.project_points(np.array([0.2, -0.4, 2.0]), CAM)
        self.assertAlmostEqual(uv[0], 100 * 0.2 / 2.0 + 50, places=6)
        self.assertAlmostEqual(uv[1], 100 * -0.4 / 2.0 + 40, places=6)

    def test_points_behind_camera_become_nan(self):
        """z ≤ 0 的点没有像点。要是让它算出个有限值，掩码里会多出一块镜像的手。"""
        uv = hm.project_points(np.array([[0.1, 0.1, -1.0], [0.1, 0.1, 0.0]]), CAM)
        self.assertTrue(np.isnan(uv).all())

    def test_nan_input_stays_nan(self):
        uv = hm.project_points(np.array([np.nan, 0.1, 1.0]), CAM)
        self.assertTrue(np.isnan(uv).all())


class TestSolveSimilarity(unittest.TestCase):
    def test_recovers_known_transform_exactly(self):
        src = joint_grid()[:, :2].astype(np.float64) * 100
        dst = 0.37 * src + np.array([12.0, -3.5])
        s, tx, ty = hm.solve_similarity(src, dst)
        self.assertAlmostEqual(s, 0.37, places=6)
        self.assertAlmostEqual(tx, 12.0, places=5)
        self.assertAlmostEqual(ty, -3.5, places=5)

    def test_nan_points_are_skipped_not_poisoning(self):
        src = joint_grid()[:, :2].astype(np.float64) * 100
        dst = 0.5 * src + np.array([1.0, 2.0])
        src = src.copy()
        src[3] = np.nan
        dst[7] = np.nan
        s, tx, ty = hm.solve_similarity(src, dst)
        self.assertAlmostEqual(s, 0.5, places=6)

    def test_too_few_points_gives_none(self):
        self.assertIsNone(hm.solve_similarity(np.zeros((2, 2)), np.zeros((2, 2))))

    def test_all_nan_gives_none(self):
        a = np.full((21, 2), np.nan)
        self.assertIsNone(hm.solve_similarity(a, a))

    def test_mirrored_points_rejected(self):
        """解出负缩放说明数据是镜像的，不是"缩小一点" —— 宁可返回 None 退回裸投影。"""
        src = joint_grid()[:, :2].astype(np.float64) * 100
        self.assertIsNone(hm.solve_similarity(src, -src))


class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clip = make_clip(Path(self.tmp.name), n_frames=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_shapes(self):
        verts, faces = hm.load_hand_mesh(self.clip, 3)
        self.assertEqual(verts.shape, (3, 2, hm.N_MANO_VERTS, 3))
        self.assertEqual(faces.shape, (hm.N_MANO_FACES, 3))
        self.assertEqual(hm.load_joints_2d(self.clip, 3).shape, (3, 2, hm.N_JOINTS, 2))

    def test_missing_mesh_raises_handmeshmissing(self):
        (self.clip / "hand_verts.bin").unlink()
        with self.assertRaises(hm.HandMeshMissing):
            hm.load_hand_mesh(self.clip)

    def test_missing_joints_2d_raises_handmeshmissing(self):
        (self.clip / "hand_joints_2d.bin").unlink()
        with self.assertRaises(hm.HandMeshMissing):
            hm.load_joints_2d(self.clip)

    def test_wrong_face_count_raises(self):
        np.zeros((10, 3), dtype=np.int32).tofile(self.clip / "hand_faces.bin")
        with self.assertRaises(ValueError):
            hm.load_hand_mesh(self.clip)

    def test_out_of_range_vertex_index_raises(self):
        faces = np.zeros((hm.N_MANO_FACES, 3), dtype=np.int32)
        faces[0] = (0, 1, hm.N_MANO_VERTS)          # 差一个
        faces.tofile(self.clip / "hand_faces.bin")
        with self.assertRaises(ValueError):
            hm.load_hand_mesh(self.clip)

    def test_frame_count_mismatch_raises(self):
        """顶点是 3 帧却说 5 帧 —— 这种情况下静默继续等于读了另一段的数据。"""
        with self.assertRaises(ValueError):
            hm.load_hand_mesh(self.clip, 5)
        with self.assertRaises(ValueError):
            hm.load_joints_2d(self.clip, 5)

    def test_ragged_file_raises(self):
        with open(self.clip / "hand_joints_2d.bin", "ab") as fh:
            fh.write(np.float32(1.0).tobytes())
        with self.assertRaises(ValueError):
            hm.load_joints_2d(self.clip)


class TestMaskGeometry(unittest.TestCase):
    def setUp(self):
        self.verts, self.faces = square_mesh()
        self.frame = np.stack([self.verts, self.verts])

    def test_mask_covers_the_projected_square(self):
        m = hm.hand_mask(self.frame, self.faces, CAM)
        ys, xs = np.nonzero(m)
        self.assertAlmostEqual(xs.min(), 40, delta=1)
        self.assertAlmostEqual(xs.max(), 60, delta=1)
        self.assertAlmostEqual(ys.min(), 30, delta=1)
        self.assertAlmostEqual(ys.max(), 50, delta=1)
        self.assertEqual(m.shape, (CAM["height"], CAM["width"]))
        self.assertEqual(m.dtype, np.dtype(bool))

    def test_absent_hand_contributes_nothing(self):
        """NaN 的手要整片丢掉。要是被当成 0 投影，画面上会凭空多一只手。"""
        frame = self.frame.copy()
        frame[1] = np.nan
        both = hm.hand_mask(frame, self.faces, CAM, hands="both")
        left = hm.hand_mask(frame, self.faces, CAM, hands="left")
        right = hm.hand_mask(frame, self.faces, CAM, hands="right")
        self.assertFalse(right.any())
        self.assertTrue(left.any())
        np.testing.assert_array_equal(both, left)

    def test_hand_selection_picks_the_right_index(self):
        frame = self.frame.copy()
        frame[1, :4, 0] += 0.4                       # 右手整体右移
        left = hm.hand_mask(frame, self.faces, CAM, hands="left")
        right = hm.hand_mask(frame, self.faces, CAM, hands="right")
        self.assertLess(np.nonzero(left)[1].mean(), np.nonzero(right)[1].mean())

    def test_dilate_grows_the_mask(self):
        base = hm.hand_mask(self.frame, self.faces, CAM)
        grown = hm.hand_mask(self.frame, self.faces, CAM, dilate=3)
        self.assertGreater(grown.sum(), base.sum())
        self.assertTrue((base & ~grown).sum() == 0)   # 只长不缩

    def test_mesh_entirely_behind_camera_gives_empty_mask(self):
        frame = self.frame.copy()
        frame[..., 2] = -2.0
        self.assertFalse(hm.hand_mask(frame, self.faces, CAM).any())

    def test_mesh_off_canvas_gives_empty_mask(self):
        frame = self.frame.copy()
        frame[..., 0] += 50.0                        # 推到画面外几十屏
        self.assertFalse(hm.hand_mask(frame, self.faces, CAM).any())

    def test_align_shifts_the_mask_as_asked(self):
        align = np.array([[TRUE_S, TRUE_TX, TRUE_TY]] * 2)
        m = hm.hand_mask(self.frame, self.faces, CAM, align=align)
        xs = np.nonzero(m)[1]
        self.assertAlmostEqual(xs.min(), TRUE_S * 40 + TRUE_TX, delta=1)
        self.assertAlmostEqual(xs.max(), TRUE_S * 60 + TRUE_TX, delta=1)

    def test_nan_align_falls_back_to_bare_projection(self):
        """解不出变换的那只手不能整只丢掉 —— 退回裸投影，至少位置大致对。"""
        align = np.full((2, 3), np.nan)
        np.testing.assert_array_equal(
            hm.hand_mask(self.frame, self.faces, CAM, align=align),
            hm.hand_mask(self.frame, self.faces, CAM))


class TestAlignmentOnSyntheticClip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clip = make_clip(Path(self.tmp.name), n_frames=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovers_the_planted_transform(self):
        fits = hm.frame_alignments(self.clip, 4)
        self.assertEqual(fits.shape, (4, 2, 3))
        np.testing.assert_allclose(fits[..., 0], TRUE_S, atol=1e-4)
        np.testing.assert_allclose(fits[..., 1], TRUE_TX, atol=1e-3)
        np.testing.assert_allclose(fits[..., 2], TRUE_TY, atol=1e-3)

    def test_absent_hand_gets_nan_row(self):
        clip = make_clip(Path(self.tmp.name) / "b", n_frames=2, absent_hand=1)
        fits = hm.frame_alignments(clip, 2)
        self.assertTrue(np.isfinite(fits[:, 0]).all())
        self.assertTrue(np.isnan(fits[:, 1]).all())

    def test_report_says_alignment_helps(self):
        rep = hm.alignment_report(self.clip, 4)
        self.assertLess(rep["residual_px"]["mean"], 0.01)
        self.assertGreater(rep["unaligned_px"]["mean"], 5.0)
        self.assertAlmostEqual(rep["scale"]["median"], TRUE_S, places=4)
        self.assertEqual(rep["n_fits"], 8)

    def test_aligned_mask_contains_the_2d_joints_and_unaligned_does_not(self):
        """这条是整个模块的立论：光投影是对不上画面的，对齐之后才对得上。"""
        good = hm.joints_inside_fraction(self.clip, 4, align=True)
        bad = hm.joints_inside_fraction(self.clip, 4, align=False)
        self.assertEqual(good["non_tip"]["fraction"], 1.0)
        self.assertEqual(bad["overall_fraction"], 0.0)
        self.assertTrue(good["aligned"])
        self.assertFalse(bad["aligned"])

    def test_series_yields_one_mask_per_frame(self):
        masks = list(hm.hand_mask_series(self.clip, 4))
        self.assertEqual(len(masks), 4)
        self.assertTrue(all(m.shape == (CAM["height"], CAM["width"]) for m in masks))
        self.assertTrue(all(m.any() for m in masks))

    def test_empty_mask_counts_as_miss_not_as_skip(self):
        """有关节却没掩码，比例必须掉下去。悄悄跳过会让判据虚高。"""
        verts = np.fromfile(self.clip / "hand_verts.bin", dtype=np.float32)
        verts = verts.reshape(4, 2, hm.N_MANO_VERTS, 3)
        verts[..., 2] = -2.0                         # 整只手挪到相机后面 → 没掩码
        verts.tofile(self.clip / "hand_verts.bin")
        out = hm.joints_inside_fraction(self.clip, 4, align=False)
        self.assertEqual(out["overall_fraction"], 0.0)
        self.assertGreater(out["left"]["n_joints"], 0)


@unittest.skipUnless(
    (CLIPS_DIR / "-1r9yl-P-Ao_60.4_68.4" / "hand_verts.bin").exists(),
    "官方 MANO 网格没同步（scripts/dev/fetch_official_extras.py 补）")
class TestRealClip(unittest.TestCase):
    """拿真数据钉住两件事：对齐确实必要，且对齐后非指尖关节基本都落在掩码里。"""

    CLIP = CLIPS_DIR / "-1r9yl-P-Ao_60.4_68.4"

    @classmethod
    def setUpClass(cls):
        # 帧数取 `stats.n_frames`（和 fetch 那边同一个口径）—— 顺带让所有加载器的
        # 帧数校验在真数据上真跑一遍，不是只在合成数据上跑。
        with open(cls.CLIP / "scene.json") as fh:
            cls.n = int(json.load(fh)["stats"]["n_frames"])
        cls.rep = hm.alignment_report(cls.CLIP, cls.n)
        cls.good = hm.joints_inside_fraction(cls.CLIP, cls.n, align=True)
        cls.bad = hm.joints_inside_fraction(cls.CLIP, cls.n, align=False)

    def test_bare_projection_really_is_off(self):
        """官方 3D 手投出来和官方 2D 关节差一大截 —— 这是"必须对齐"的证据本身。"""
        self.assertGreater(self.rep["unaligned_px"]["median"], 5.0)
        self.assertLess(self.rep["residual_px"]["median"], 5.0)

    def test_scale_really_drifts_frame_to_frame(self):
        """缩放逐帧在漂（不是恒 1），所以只能逐帧拟合，不能全段一个变换。"""
        self.assertLess(self.rep["scale"]["min"], 0.9)
        self.assertGreater(self.rep["scale"]["max"], 1.1)

    def test_aligned_non_tip_joints_are_inside(self):
        self.assertGreater(self.good["non_tip"]["fraction"], 0.9)
        self.assertGreater(self.good["non_tip"]["fraction"],
                           self.bad["non_tip"]["fraction"] + 0.05)

    def test_tips_are_worse_by_construction(self):
        """指尖比其余关节差是几何本身如此，不是掩码错位 —— 判据分开报就是为了这个。"""
        self.assertLess(self.good["tip"]["fraction"], self.good["non_tip"]["fraction"])
        self.assertGreater(self.good["tip"]["fraction"], 0.5)

    def test_masks_are_nonempty_and_plausibly_sized(self):
        areas = [m.mean() for m in hm.hand_mask_series(self.CLIP, self.n)]
        self.assertEqual(len(areas), self.n)
        self.assertTrue(all(a > 0 for a in areas), "有整帧没掩码")
        self.assertLess(max(areas), 0.25, "掩码占了四分之一以上画面，不像手")


if __name__ == "__main__":
    unittest.main()
