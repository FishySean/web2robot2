"""ffmpeg / ffprobe 薄封装 —— 只做本模块要的四件事。

为什么不直接用 cv2：
1. **`cv2.VideoCapture` 的按帧号定位在长视频上不可靠**（内部还是关键帧 seek + 计数，
   源视频只要有一处 B 帧重排就偏），而我们要的是"片段第 i 帧对应源视频哪一帧"这种
   逐帧级的对应，偏一帧就是错一帧。
2. 上游 `retarget/utils/viz.py` 用 `cv2.VideoWriter_fourcc(*"mp4v")` 出的是 mpeg4，
   我方约定 §3 要 h264/yuv420p（`docs/BACKLOG.md` B11）。新产物直接用 libx264。

**不 seek，是故意的。** `-ss` 之前/之后放、ffprobe 的 `-read_intervals`
三种写法各有一套关键帧对齐行为，两次 pass（一次取时间戳、一次解码取像素）
只要seek行为不同就会错位。所以两趟都从头解码整支视频，第 i 帧 ↔ 时间戳数组第 i 项，
一一对应不靠假设。278 s / 30 fps 的源视频实测整趟 ffprobe ≈ 3 s，不值得为它冒错位的风险。
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class VideoError(RuntimeError):
    """视频不可用（不存在 / 解不开 / 截断）。"""


@dataclass(frozen=True)
class VideoMeta:
    path: str
    codec: str
    width: int
    height: int
    fps: float             # r_frame_rate（容器声明的，可能和实际不符）
    duration: float        # format.duration
    nb_frames: Optional[int]   # 容器声明的帧数，mp4 有、有些格式没有

    def to_dict(self) -> dict:
        return dict(path=self.path, codec=self.codec, width=self.width,
                    height=self.height, fps=self.fps, duration=self.duration,
                    nb_frames=self.nb_frames)


def _run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, **kw)


def _rate(text: str) -> float:
    """`30000/1001` → 29.97；空/0 → 0.0。"""
    if not text or text in ("0/0", "N/A"):
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(text)


def probe(path) -> VideoMeta:
    path = Path(path)
    if not path.exists():
        raise VideoError(f"视频不存在：{path}")
    cp = _run([FFPROBE, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames",
               "-show_entries", "format=duration", "-of", "json", str(path)])
    if cp.returncode != 0:
        raise VideoError(f"ffprobe 读不了 {path}：{cp.stderr.strip()[:300]}")
    info = json.loads(cp.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise VideoError(f"{path} 里没有视频流")
    st = streams[0]
    nb = st.get("nb_frames")
    return VideoMeta(
        path=str(path),
        codec=st.get("codec_name", "?"),
        width=int(st.get("width") or 0),
        height=int(st.get("height") or 0),
        fps=_rate(st.get("r_frame_rate", "")),
        duration=float((info.get("format") or {}).get("duration") or 0.0),
        nb_frames=int(nb) if nb not in (None, "N/A") else None,
    )


def decodes_at(path, t_seconds: float) -> bool:
    """`t_seconds` 处能不能真解出一帧。

    这条检查是有来历的：不带 PO token 从 YouTube 下载会拿到一个
    **容器完整、媒体数据被截断**的文件 —— `ffprobe` 照样报 278 s / 8331 帧，
    只有真去解码才会露出 `partial file`。光看 duration 会以为下载成功了。
    """
    cp = _run([FFMPEG, "-v", "error", "-ss", f"{max(t_seconds, 0):.3f}", "-i", str(path),
               "-frames:v", "1", "-f", "null", "-"])
    return cp.returncode == 0 and "partial file" not in cp.stderr


def verify_playable(path, need_until: float) -> VideoMeta:
    """确认视频至少能解到 `need_until` 秒 —— 不行就带诊断报错。"""
    meta = probe(path)
    if meta.duration <= 0:
        raise VideoError(f"{path}: ffprobe 报 duration=0")
    if meta.duration + 0.5 < need_until:
        raise VideoError(
            f"{path}: 只有 {meta.duration:.1f} s，够不到要截取的 {need_until:.1f} s")
    probe_points = [0.0, need_until - 0.5]
    mid = need_until / 2
    if mid > 1.0:
        probe_points.insert(1, mid)
    bad = [t for t in probe_points if not decodes_at(path, t)]
    if bad:
        size = Path(path).stat().st_size
        raise VideoError(
            f"{path}: 容器说有 {meta.duration:.1f} s，但 t={bad} 处解不出帧"
            f"（文件 {size} 字节）。这就是**截断下载**的样子：moov 完整、媒体数据没下全。"
            " 不要把它当成可用素材。")
    return meta


def frame_times(path, cache: Optional[Path] = None) -> np.ndarray:
    """整支视频每一帧的时间戳（秒），顺序就是解码顺序。

    `cache` 给定时缓存成 json（同一支源视频会被多段片段共用，实测 10 段只 6 支视频）。
    """
    path = Path(path)
    if cache is not None and Path(cache).exists():
        with open(cache) as fh:
            payload = json.load(fh)
        if payload.get("size") == path.stat().st_size:
            return np.asarray(payload["times"], dtype=np.float64)
    cp = _run([FFPROBE, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "frame=best_effort_timestamp_time",
               "-of", "csv=p=0", str(path)])
    if cp.returncode != 0:
        raise VideoError(f"ffprobe 取不到帧时间戳：{cp.stderr.strip()[:300]}")
    times = []
    for line in cp.stdout.splitlines():
        tok = line.strip().rstrip(",")
        if tok and tok != "N/A":
            times.append(float(tok))
    if not times:
        raise VideoError(f"{path}: 一个帧时间戳都没读到")
    arr = np.asarray(times, dtype=np.float64)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as fh:
            json.dump({"size": path.stat().st_size, "times": times}, fh)
    return arr


def nearest_frames(src_times: np.ndarray, targets: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """每个目标时刻取最近的源帧。返回 (帧号数组, 时间误差数组=源−目标)。"""
    tgt = np.asarray(targets, dtype=np.float64)
    idx = np.searchsorted(src_times, tgt)
    idx = np.clip(idx, 1, len(src_times) - 1)
    left, right = src_times[idx - 1], src_times[idx]
    take_left = (tgt - left) <= (right - tgt)
    picked = np.where(take_left, idx - 1, idx)
    return picked.astype(np.int64), src_times[picked] - tgt


def stream_rgb(path, want: Sequence[int], width: int, height: int) -> Iterator[Tuple[int, np.ndarray]]:
    """整支视频解一遍，只把 `want` 里的帧号交出来（按帧号升序）。

    `width`/`height` 是**目标**尺寸：官方片段的 2D 手部关节坐标是在
    `camera.json` 的 853×480 上标的，源视频是 1280×720，不缩放就对不上像素。
    """
    want_sorted = sorted(set(int(i) for i in want))
    if not want_sorted:
        return
    frame_bytes = width * height * 3
    cmd = [FFMPEG, "-v", "error", "-i", str(path),
           "-vf", f"scale={width}:{height}", "-pix_fmt", "rgb24",
           "-f", "rawvideo", "-vsync", "0", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wanted = set(want_sorted)
    last = want_sorted[-1]
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if idx in wanted:
                yield idx, np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            if idx >= last:
                break
            idx += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.stderr.close()
        proc.wait()
        if "partial file" in err:
            raise VideoError(f"{path}: 解码中途遇到 partial file —— 素材是截断的")


def stream_gray(path, width: Optional[int] = None, height: Optional[int] = None) -> Iterator[np.ndarray]:
    """按顺序吐出灰度帧（给对齐用的运动能量曲线）。"""
    meta = probe(path)
    w = width or meta.width
    h = height or meta.height
    frame_bytes = w * h
    vf = f"scale={w}:{h},format=gray" if (width or height) else "format=gray"
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-vf", vf,
           "-pix_fmt", "gray", "-f", "rawvideo", "-vsync", "0", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait()


class H264Writer:
    """把 RGB 帧写成 h264 / yuv420p（我方约定 §3）。

    `crf=12` 是"近无损但别爆盘"的取法：这是**流水线输入**，后面视觉合成还要抠它，
    压太狠会把手边缘的细节吃掉。真要逐字节复现原帧就用 `--keep_png`。
    """

    def __init__(self, path, fps: float, width: int, height: int, crf: int = 12):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [FFMPEG, "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
               "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf),
               "-pix_fmt", "yuv420p", str(self.path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.n_written = 0

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.n_written += 1

    def close(self) -> None:
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode("utf-8", "replace")
        self._proc.stderr.close()
        rc = self._proc.wait()
        if rc != 0:
            raise VideoError(f"libx264 编码失败（rc={rc}）：{err.strip()[:300]}")

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()
        else:
            self._proc.kill()
            self._proc.wait()
