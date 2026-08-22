"""流水线第 0 步：官方片段 → 原始 RGB（`src/web2robot/fetch/`）。

这个测试守的东西按重要性排序：

1. **取的是"对的那一帧"，不只是"对的帧数"。** 造一支每帧亮度按帧号线性递增的
   源视频，截完逐帧核对亮度 —— 帧数对但整体错位一帧，这条会红。这是本模块唯一
   真正要命的东西：错一帧，后面视觉合成贴上去的机器人手就和画面里的手不在同一时刻。
2. **时间轴只认 `scene.json` 的 `video_source`，不认目录名。** 实测 10 段官方片段
   里有 1 段目录名和元数据差 3.39 s（按源视频 30 fps 算是 102 帧）。这条拿真实
   数据钉住那个差值 —— 哪天有人"顺手"改成读目录名，它会红。
3. **截断的下载必须被判死。** 不带 PO token 从 YouTube 下会拿到"容器完整、
   媒体数据截断"的文件：`ffprobe` 照样报 278 s / 8331 帧。所以验收必须真解码。
   这条造一个同样形状的文件（合法 mp4 截掉后半段字节）来验。
4. **对齐判据能分辨对齐和错位**：同内容 → lag 0 / aligned；平移过的 → misaligned。
   判据自己分不清好坏，用它去验收就是自欺。
5. yt-dlp 命令形状：**不用 `--download-sections`**（那条路走 ffmpeg 直连 CDN，
   实测被 4XX 拒），`--remote-components` 只在显式打开时出现（运行时下载第三方脚本
   得有人点头）。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_fetch_rgb -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.fetch import align, download, frames, sources, video  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
CLIPS_DIR = REPO / "data" / "clips_official"

W, H = 64, 48
SRC_FPS = 30.0


def make_ramp_video(path: Path, n_frames: int, fps: float = SRC_FPS,
                    w: int = W, h: int = H, step: int = 4,
                    faststart: bool = False) -> None:
    """第 i 帧是一张亮度恒为 ``i*step`` 的灰图（无损 h264，便于逐帧核对）。

    `faststart` 把 moov 挪到文件头 —— 这样掐掉后半段字节，就能造出和
    YouTube 截断下载**同一种形状**的文件：元数据完整、媒体数据缺失。
    """
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
           "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
           "-pix_fmt", "yuv444p"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(path))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    for i in range(n_frames):
        val = (i * step) % 256
        proc.stdin.write(np.full((h, w, 3), val, dtype=np.uint8).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode()
    proc.stderr.close()
    if proc.wait() != 0:
        raise RuntimeError(f"造测试视频失败：{err[:200]}")


def fake_scene(youtube_id: str, start: float, end: float, fps: float, n_frames: int,
               width: int = W, height: int = H) -> dict:
    return {
        "id": f"{youtube_id}_{start}_{end}",
        "fps": fps,
        "duration": n_frames / fps,
        "video_source": {"type": "youtube", "youtube_id": youtube_id,
                         "start_seconds": start, "end_seconds": end},
        "camera": {"focal": 100.0, "cx": width / 2 - 0.5, "cy": height / 2 - 0.5,
                   "width": width, "height": height},
        "stats": {"n_frames": n_frames},
        "action100m_metadata": {"video_duration": 60.0},
    }


class TestClipSource(unittest.TestCase):
    def test_timeline_is_exact(self):
        c = sources.ClipSource.from_scene(fake_scene("vid", 10.0, 14.0, 20.0, 80), "c")
        tl = c.timeline
        self.assertEqual(len(tl), 80)
        self.assertAlmostEqual(tl[0], 10.0, places=9)
        self.assertAlmostEqual(tl[1] - tl[0], 1 / 20.0, places=9)
        self.assertAlmostEqual(c.span_end, 10.0 + 79 / 20.0, places=9)
        self.assertTrue(c.self_consistent)

    def test_name_gap_is_reported_not_used(self):
        scene = fake_scene("vid", 195.79, 205.523, 15.0, 146)
        c = sources.ClipSource.from_scene(scene, "c", dir_name="vid_192.4_209.2")
        gap = c.name_gap_seconds
        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap[0], 192.4 - 195.79, places=3)
        # 计算一律用 video_source：第 0 帧的时刻必须是 195.79，不是 192.4
        self.assertAlmostEqual(c.timeline[0], 195.79, places=6)

    def test_youtube_id_with_dashes_parses(self):
        # 真实 ID 里有 `-` 和 `_`，目录名只能从右往左切两刀
        c = sources.ClipSource.from_scene(
            fake_scene("--oo8_XIuOM", 799.53, 809.81, 15.0778, 155), "c",
            dir_name="--oo8_XIuOM_799.5_809.8")
        self.assertEqual(c.youtube_id, "--oo8_XIuOM")
        self.assertAlmostEqual(c.name_start, 799.5, places=3)

    def test_non_youtube_source_is_refused(self):
        scene = fake_scene("vid", 0.0, 1.0, 10.0, 10)
        scene["video_source"]["type"] = "internal_batch"
        with self.assertRaises(sources.ClipMetadataError):
            sources.ClipSource.from_scene(scene, "c")

    def test_missing_n_frames_is_refused(self):
        scene = fake_scene("vid", 0.0, 1.0, 10.0, 10)
        scene["stats"] = {}
        with self.assertRaises(sources.ClipMetadataError):
            sources.ClipSource.from_scene(scene, "c")


@unittest.skipUnless(CLIPS_DIR.is_dir(), "没有 data/clips_official")
class TestRealClipMetadata(unittest.TestCase):
    """拿真实的 10 段官方片段钉住"目录名不可信"这个实测事实。"""

    def test_all_clips_parse_and_are_self_consistent(self):
        clips = sources.load_clip_sources(CLIPS_DIR)
        self.assertGreaterEqual(len(clips), 10)
        for c in clips:
            with self.subTest(clip=c.clip_id):
                self.assertTrue(c.self_consistent,
                                f"{c.clip_id}: n_frames/fps 与 duration/(end-start) 不闭合")

    def test_directory_name_disagrees_on_a_known_clip(self):
        clips = {c.clip_id: c for c in sources.load_clip_sources(CLIPS_DIR)}
        c = clips.get("-2cNMO9Mm3Q_192.4_209.2")
        if c is None:
            self.skipTest("这段片段不在库里")
        gap = c.name_gap_seconds
        # 实测 −3.39 s；按源视频 30 fps 就是 102 帧。用目录名截会整段错开。
        self.assertLess(gap[0], -3.0)
        self.assertAlmostEqual(c.start_seconds, 195.79, places=3)

    def test_one_video_serves_several_clips(self):
        groups = sources.group_by_video(sources.load_clip_sources(CLIPS_DIR))
        self.assertLess(len(groups), 10)  # 10 段只对应 6 支视频，别重复下载


class TestNearestFrames(unittest.TestCase):
    def test_picks_nearest_and_reports_error(self):
        src = np.arange(0, 1.0, 1 / 30.0)          # 30 fps 的时间戳
        targets = [0.0, 0.1, 0.5, 0.9]
        idx, dt = video.nearest_frames(src, targets)
        self.assertEqual(list(idx), [0, 3, 15, 27])
        self.assertTrue(np.all(np.abs(dt) <= 0.5 / 30 + 1e-9))

    def test_ties_and_bounds(self):
        src = np.array([0.0, 1.0, 2.0])
        idx, _ = video.nearest_frames(src, [-5.0, 0.5, 9.0])
        self.assertEqual(list(idx), [0, 0, 2])     # 越界夹到端点，正中取左


@unittest.skipUnless(HAVE_FFMPEG, "机器上没有 ffmpeg/ffprobe")
class TestSourceVerification(unittest.TestCase):
    def test_good_video_passes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "src.mp4"
            make_ramp_video(p, 60)
            meta = video.verify_playable(p, need_until=1.5)
            self.assertEqual((meta.width, meta.height), (W, H))
            self.assertAlmostEqual(meta.fps, SRC_FPS, places=3)

    def test_truncated_video_is_rejected(self):
        """容器完整、媒体数据截断 —— 这正是 YouTube 不给 PO token 时返回的样子。"""
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / "src.mp4"
            make_ramp_video(good, 300, faststart=True)   # moov 在头，掐尾巴还能 probe
            bad = Path(td) / "trunc.mp4"
            data = good.read_bytes()
            bad.write_bytes(data[: len(data) // 3])
            # 先确认造出来的东西确实是"能 probe 的"：不然这条测的就不是截断
            meta = video.probe(bad)
            self.assertGreater(meta.duration, 8.0)
            with self.assertRaises(video.VideoError) as cm:
                video.verify_playable(bad, need_until=9.0)
            self.assertIn("截断", str(cm.exception))

    def test_too_short_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "src.mp4"
            make_ramp_video(p, 30)                 # 只有 1 s
            with self.assertRaises(video.VideoError):
                video.verify_playable(p, need_until=20.0)


@unittest.skipUnless(HAVE_FFMPEG, "机器上没有 ffmpeg/ffprobe")
class TestExtraction(unittest.TestCase):
    def _source(self, td: Path, n: int = 240):
        p = td / "src.mp4"
        make_ramp_video(p, n)
        meta = video.verify_playable(p, need_until=n / SRC_FPS - 0.1)
        return download.SourceVideo(youtube_id="vid", path=p, meta=meta,
                                    backend="local", sha256="deadbeef")

    def test_frame_count_codec_and_size(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._source(td)
            clip = sources.ClipSource.from_scene(fake_scene("vid", 1.0, 3.0, 15.0, 30), "c")
            rec = frames.extract_clip_rgb(src, clip, td / "out")
            self.assertEqual(rec["rgb"]["n_frames"], 30)
            out = video.probe(td / "out" / "rgb.mp4")
            self.assertEqual(out.codec, "h264")
            self.assertEqual((out.width, out.height), (W, H))
            self.assertTrue(rec["sampling"]["within_half_frame"],
                            rec["sampling"]["max_abs_dt"])
            self.assertEqual(len(rec["sampling"]["frames"]), 30)

    def test_extracted_frames_are_the_right_ones(self):
        """核对内容，不只核对帧数：亮度 = 源帧号 × 4。"""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._source(td)
            clip = sources.ClipSource.from_scene(fake_scene("vid", 2.0, 4.0, 15.0, 30), "c")
            rec = frames.extract_clip_rgb(src, clip, td / "out")
            got = [float(g.mean()) for g in video.stream_gray(td / "out" / "rgb.mp4")]
            self.assertEqual(len(got), 30)
            for entry, mean in zip(rec["sampling"]["frames"], got):
                expect = (entry["source_frame"] * 4) % 256
                # 期望源帧号也必须是 t_target*30 附近 —— 双保险，防止索引和内容一起偏
                self.assertAlmostEqual(entry["source_frame"], entry["t_target"] * SRC_FPS,
                                       delta=0.6)
                self.assertAlmostEqual(mean, expect, delta=4.0,
                                       msg=f"第 {entry['i']} 帧内容不是源帧 {entry['source_frame']}")

    def test_source_too_short_for_span_fails_loudly(self):
        """源视频没覆盖片段时段 → 必须报错，不能靠复制最后一帧把帧数凑够。"""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._source(td, n=60)           # 只有 2 s
            clip = sources.ClipSource.from_scene(fake_scene("vid", 1.0, 5.0, 15.0, 60), "c")
            with self.assertRaises(RuntimeError) as cm:
                frames.extract_clip_rgb(src, clip, td / "out")
            self.assertIn("找不到对应帧", str(cm.exception))

    def test_keep_png_writes_every_frame(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._source(td)
            clip = sources.ClipSource.from_scene(fake_scene("vid", 1.0, 2.0, 15.0, 15), "c")
            frames.extract_clip_rgb(src, clip, td / "out", keep_png=True)
            pngs = sorted((td / "out" / "rgb_frames").glob("*.png"))
            self.assertEqual(len(pngs), 15)


class TestCrossCorrelation(unittest.TestCase):
    def test_zero_lag_on_identical(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=120)
        lag, corr, _ = align.xcorr_best_lag(a, a.copy(), max_lag=20)
        self.assertEqual(lag, 0)
        self.assertGreater(corr, 0.99)

    def test_recovers_known_shift(self):
        # 约定是 b[i] ≈ a[i+L] → 用 roll(-k) 造出 b[i]=base[i+k]，应认出 L=+k
        rng = np.random.default_rng(1)
        base = rng.normal(size=200)
        for k in (3, 7, -5):
            b = np.roll(base, -k)
            lag, corr, _ = align.xcorr_best_lag(base, b, max_lag=20)
            self.assertEqual(lag, k, f"平移 {k} 帧没被认出来（认成 {lag}）")
            self.assertGreater(corr, 0.9)

    def test_short_input_does_not_crash(self):
        lag, corr, curve = align.xcorr_best_lag(np.zeros(2), np.zeros(2))
        self.assertEqual(lag, 0)
        self.assertEqual(curve, {})


@unittest.skipUnless(HAVE_FFMPEG, "机器上没有 ffmpeg/ffprobe")
class TestAlignReport(unittest.TestCase):
    def _clip_dir(self, td: Path, n: int, fps: float, shift: int = 0) -> Path:
        """造一个假片段目录：depth.mp4 与 rgb 同内容（shift≠0 时故意错开）。"""
        d = td / "clip"
        d.mkdir(parents=True, exist_ok=True)
        make_ramp_video(d / "depth.mp4", n + abs(shift), fps=fps)
        if shift:
            # 掐掉前 shift 帧 → 与 rgb 错开 shift 帧
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "depth.mp4"),
                            "-vf", f"select=gte(n\\,{shift})", "-vsync", "0",
                            "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
                            str(d / "depth_shift.mp4")], check=True)
            (d / "depth.mp4").unlink()
            (d / "depth_shift.mp4").rename(d / "depth.mp4")
        with open(d / "scene.json", "w") as fh:
            json.dump(fake_scene("vid", 0.0, n / fps, fps, n), fh)
        return d

    def test_aligned_case(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n, fps = 40, 15.0
            d = self._clip_dir(td, n, fps)
            clip = sources.ClipSource.from_clip_dir(d)
            rgb = td / "out" / "rgb.mp4"
            rgb.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(d / "depth.mp4", rgb)
            rep = align.align_report(clip, d, rgb, td / "out")
            self.assertTrue(rep["counts_ok"], rep["counts"])
            self.assertEqual(rep["motion_lag"]["depth.mp4"]["best_lag"], 0)
            self.assertEqual(rep["verdict"], "aligned")

    def test_shifted_case_is_caught(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n, fps, shift = 60, 15.0, 5
            d = self._clip_dir(td, n, fps, shift=shift)
            clip = sources.ClipSource.from_clip_dir(d)
            # rgb 用"正确"的那段（从第 0 帧起），depth 已被掐掉前 5 帧
            rgb = td / "out" / "rgb.mp4"
            rgb.parent.mkdir(parents=True, exist_ok=True)
            make_ramp_video(rgb, n, fps=fps)
            rep = align.align_report(clip, d, rgb, td / "out", montage=False)
            self.assertNotEqual(rep["verdict"], "aligned")


class TestJointProjection(unittest.TestCase):
    def test_projects_3d_joints_when_2d_missing(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n = 5
            clip = sources.ClipSource.from_scene(fake_scene("vid", 0.0, 1.0, 5.0, n), "c")
            j3 = np.zeros((n, 2, 21, 3), dtype=np.float32)
            j3[..., 2] = 1.0                      # z = 1 m
            j3[..., 0] = 0.0
            j3[..., 1] = 0.0
            j3[0, 1, :, :] = np.nan               # 第 0 帧右手缺失
            (td / "hand_joints.bin").write_bytes(j3.tobytes())
            with open(td / "camera.json", "w") as fh:
                json.dump({"focal": 100.0, "cx": 31.5, "cy": 23.5,
                           "width": W, "height": H}, fh)
            uv = align.load_joints_2d(td, clip)
            self.assertEqual(uv.shape, (n, 2, 21, 2))
            self.assertAlmostEqual(uv[1, 0, 0, 0], 31.5, places=4)   # x=0 → cx
            self.assertTrue(np.isnan(uv[0, 1, 0, 0]))                # NaN 传播
            self.assertAlmostEqual(align.in_frame_fraction(uv, W, H), 1.0, places=6)


class TestDownloadCommand(unittest.TestCase):
    def test_no_download_sections_and_opt_in_remote(self):
        cmd = download.ytdlp_command("abc", "/tmp/%(id)s.%(ext)s", python="/usr/bin/python3")
        self.assertNotIn("--download-sections", cmd)   # 实测那条路被 CDN 4XX 拒
        self.assertNotIn("--remote-components", cmd)   # 运行时拉第三方脚本要显式打开
        self.assertIn("-f", cmd)
        self.assertIn("--no-part", cmd)
        self.assertTrue(cmd[-1].endswith("watch?v=abc"))
        opt = download.ytdlp_command("abc", "/tmp/x", ejs_remote=True)
        self.assertIn("--remote-components", opt)

    def test_local_backend_says_where_it_looked(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(download.SourceUnavailable) as cm:
                download.ensure_source("nosuchid", Path(td) / "cache", 1.0, backend="local")
            self.assertIn("local", str(cm.exception))

    def test_unknown_backend_refused(self):
        with self.assertRaises(ValueError):
            download.ensure_source("x", "/tmp", 1.0, backend="magic")

    @unittest.skipUnless(HAVE_FFMPEG, "机器上没有 ffmpeg/ffprobe")
    def test_local_backend_finds_and_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "cache").mkdir()
            make_ramp_video(td / "cache" / "vid.mp4", 90)
            got = download.ensure_source("vid", td / "cache", 2.0, backend="local")
            self.assertEqual(got.backend, "local")
            self.assertEqual(len(got.sha256), 64)


if __name__ == "__main__":
    unittest.main()
