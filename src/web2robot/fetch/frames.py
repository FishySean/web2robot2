"""从源视频里截出片段的 RGB —— 逐帧对应，不靠整数跨步。

做法（三步，都是可复查的）：

1. 目标时间轴：``t_i = start_seconds + i / fps``，i = 0 … n_frames−1（`sources.py`）。
   这就是官方片段每一帧在源视频里的时刻。
2. 源帧时间戳：整支视频过一遍 ffprobe，拿到每一帧的 `best_effort_timestamp_time`。
3. 最近邻取帧，并把**每一帧的时间误差记进 `frames_index.json`**。

第 3 步那个误差是这条链路唯一的自证据：源视频 30 fps 时，半帧 = 16.7 ms，
所以 ``max|dt| ≤ 1/(2·fps_src)`` 才叫"取到了该取的帧"；超了说明时间轴假设错了
（比如目录名和 `video_source` 用混了，那会错出秒级的偏差）。

**不做重采样、不做插值**：片段第 i 帧就是源视频某一个真实帧的缩放，
缩放到 `camera.json` 的 853×480（官方的 2D 关节坐标标在这个尺寸上）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .download import SourceVideo
from .sources import ClipSource
from .video import H264Writer, frame_times, nearest_frames, stream_rgb


def extract_clip_rgb(source: SourceVideo, clip: ClipSource, out_dir,
                     pts_cache: Optional[Path] = None, keep_png: bool = False,
                     crf: int = 12, max_dt_frames: float = 1.0) -> Dict:
    """截出 `clip` 的 RGB，写 `rgb.mp4` + `frames_index.json`，返回索引记录。

    `max_dt_frames`：允许的最大取帧时间误差，单位是**源视频帧周期**。正常最近邻
    取帧的误差不超过半帧；超过一整帧只有一种解释 —— 目标时刻在源视频里没有对应
    的帧（素材没覆盖这一段，或者时间轴口径用错了）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    width = clip.width or source.meta.width
    height = clip.height or source.meta.height

    src_times = frame_times(source.path, cache=pts_cache)
    targets = clip.timeline
    picked, dt = nearest_frames(src_times, targets)

    # 越界的目标时刻会被夹到端点 —— 那些帧是"把最后一帧复制一遍"，帧数照样凑够，
    # 内容是错的。所以要在写盘之前按 dt 判死；只把它记进 json 里等人去看，
    # 等于静默交付错数据。
    if len(dt) and source.meta.fps:
        worst = int(np.argmax(np.abs(dt)))
        tol = max_dt_frames / source.meta.fps
        if abs(float(dt[worst])) > tol:
            raise RuntimeError(
                f"{clip.clip_id}: 第 {worst} 帧要的时刻 t={targets[worst]:.3f}s 在源视频里"
                f"找不到对应帧（最近的一帧差 {float(dt[worst])*1000:.0f} ms，超过 "
                f"{max_dt_frames:g} 个源帧 = {tol*1000:.0f} ms）。"
                f"源视频只覆盖 {src_times[0]:.3f}–{src_times[-1]:.3f}s，"
                f"片段要 {targets[0]:.3f}–{targets[-1]:.3f}s —— 素材和元数据不是一回事")

    png_dir = out_dir / "rgb_frames"
    if keep_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = out_dir / "rgb.mp4"
    ptr = 0
    n = len(targets)
    with H264Writer(rgb_path, clip.fps, width, height, crf=crf) as writer:
        for sidx, frame in stream_rgb(source.path, picked.tolist(), width, height):
            # picked 单调不减：同一源帧可能被连续多个片段帧用到（片段 fps 高于源时）
            while ptr < n and int(picked[ptr]) == sidx:
                writer.write(frame)
                if keep_png:
                    from PIL import Image  # 只有 --keep_png 才需要 pillow
                    Image.fromarray(frame).save(png_dir / f"{ptr:06d}.png")
                ptr += 1
        written = writer.n_written

    if ptr != n:
        raise RuntimeError(
            f"{clip.clip_id}: 只取到 {ptr}/{n} 帧 —— 源视频在 "
            f"t={targets[ptr] if ptr < n else '?'} 之前就结束了，素材和元数据不是一回事")

    record = {
        "clip": clip.to_dict(),
        "source": source.to_dict(),
        "rgb": {
            "path": str(rgb_path),
            "n_frames": int(written),
            "fps": clip.fps,
            "width": width,
            "height": height,
            "codec": "h264",
            "pix_fmt": "yuv420p",
            "crf": crf,
        },
        "sampling": {
            "source_fps": source.meta.fps,
            "max_abs_dt": float(np.max(np.abs(dt))) if len(dt) else 0.0,
            "mean_abs_dt": float(np.mean(np.abs(dt))) if len(dt) else 0.0,
            "half_source_frame": (0.5 / source.meta.fps) if source.meta.fps else None,
            "n_unique_source_frames": int(len(set(picked.tolist()))),
            "frames": [
                {"i": i, "t_target": round(float(targets[i]), 6),
                 "source_frame": int(picked[i]),
                 "t_source": round(float(src_times[picked[i]]), 6),
                 "dt": round(float(dt[i]), 6)}
                for i in range(n)
            ],
        },
    }
    sampling = record["sampling"]
    half = sampling["half_source_frame"]
    sampling["within_half_frame"] = bool(half is not None and sampling["max_abs_dt"] <= half + 1e-6)
    with open(out_dir / "frames_index.json", "w") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    return record
