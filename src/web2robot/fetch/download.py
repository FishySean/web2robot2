"""把源视频弄到本地 —— 两个后端，都要过同一道"能不能真解码"的验收。

后端：

* ``local``：从一个目录里按 ``<youtube_id>.<ext>`` 找现成的文件。
  给"视频从别处来"（公司镜像、别人下好的、手工下的）留的口子 —— 后面的截取/对齐
  代码完全不关心字节是怎么来的。
* ``ytdlp``：调 ``envs/perception_env`` 里的 yt_dlp 下整支视频（**不用
  ``--download-sections``**：那条路走 ffmpeg 直接拉 CDN 的字节区间，实测被
  4XX 拒；而且同一支视频常被多段片段共用，整支下一次更省）。

**2026-08-22 实测：`ytdlp` 后端在这台机器上拿不到可用视频。** 11 个
``player_client`` 全试过，结果只有两种：``android_vr`` / ``tv_embedded`` 是
``HTTP 403``，其余能出文件的（``mweb`` / ``android`` / ``web_embedded`` /
``tv_simply``）全都返回**同一个 145471 字节的截断文件** —— 容器说 278 s / 8331 帧，
真解码 0 帧（`ffmpeg` 报 `partial file`）。机器上有 node v22 可以当 JS runtime，
yt-dlp 也认，但缺的是 GVS PO Token，不是 JS runtime。所以本模块里那道
`verify_playable` 不是防御性编程，是**当天真被它挡下来的**那道检查。
详见 `docs/BACKLOG.md` B12。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .video import VideoError, VideoMeta, probe, verify_playable

#: 只要 720p 及以下的视频轨。官方片段本身是 853×480，下 1080p 纯浪费。
DEFAULT_FORMAT = "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]"

#: 认得的容器后缀（`local` 后端按这个顺序找）
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")

BACKENDS = ("local", "ytdlp")


class SourceUnavailable(RuntimeError):
    """源视频拿不到 —— 消息里必须带"为什么拿不到"，不许只说失败。"""


@dataclass
class SourceVideo:
    youtube_id: str
    path: Path
    meta: VideoMeta
    backend: str
    sha256: str

    def to_dict(self) -> Dict:
        return {"youtube_id": self.youtube_id, "path": str(self.path),
                "backend": self.backend, "sha256": self.sha256, **self.meta.to_dict()}


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def find_local(youtube_id: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    for d in search_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for ext in VIDEO_EXTS:
            for name in (f"{youtube_id}{ext}", f"src_{youtube_id}{ext}"):
                cand = d / name
                if cand.exists():
                    return cand
        # 退一步：文件名里包含 ID 的（手工下的常带标题）
        hits = sorted(p for p in d.iterdir()
                      if p.is_file() and youtube_id in p.name and p.suffix.lower() in VIDEO_EXTS)
        if hits:
            return hits[0]
    return None


def ytdlp_command(youtube_id: str, out_template: str, python: Optional[str] = None,
                  fmt: str = DEFAULT_FORMAT, js_runtime: Optional[str] = "node",
                  ejs_remote: bool = False) -> List[str]:
    """拼 yt_dlp 命令。单独抽出来是为了能在测试里断言参数，不必真联网。"""
    cmd = [python or sys.executable, "-m", "yt_dlp"]
    if js_runtime:
        cmd += ["--js-runtimes", js_runtime]
    if ejs_remote:
        # 会从 GitHub 拉 yt-dlp 的 challenge solver 脚本再交给 node 跑。
        # 默认关：那是运行时下载第三方脚本，得有人明确同意。
        cmd += ["--remote-components", "ejs:github"]
    cmd += ["-f", fmt, "--no-part", "--retries", "3",
            "-o", out_template, f"https://www.youtube.com/watch?v={youtube_id}"]
    return cmd


def _ytdlp_download(youtube_id: str, cache_dir: Path, python: Optional[str],
                    fmt: str, js_runtime: Optional[str], ejs_remote: bool,
                    timeout: float) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmpl = str(cache_dir / "src_%(id)s.%(ext)s")
    cmd = ytdlp_command(youtube_id, tmpl, python=python, fmt=fmt,
                        js_runtime=js_runtime, ejs_remote=ejs_remote)
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SourceUnavailable(f"{youtube_id}: yt_dlp 超过 {timeout:.0f}s 还没下完")
    hits = sorted(p for p in cache_dir.glob(f"src_{youtube_id}.*")
                  if p.suffix.lower() in VIDEO_EXTS)
    if not hits:
        tail = "\n".join((cp.stderr or cp.stdout or "").strip().splitlines()[-6:])
        raise SourceUnavailable(f"{youtube_id}: yt_dlp 没产出文件。yt_dlp 最后几行：\n{tail}")
    return hits[0]


def ensure_source(youtube_id: str, cache_dir, need_until: float,
                  backend: str = "local", search_dirs: Sequence = (),
                  python: Optional[str] = None, fmt: str = DEFAULT_FORMAT,
                  js_runtime: Optional[str] = "node", ejs_remote: bool = False,
                  timeout: float = 900.0) -> SourceVideo:
    """拿到一支能解码到 `need_until` 秒的源视频，否则抛 `SourceUnavailable`。

    两个后端都先看缓存目录：已经下好的不重下（同一支视频常被多段片段共用）。
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend 只能是 {BACKENDS}，拿到 {backend!r}")
    cache_dir = Path(cache_dir)
    dirs: List[Path] = [cache_dir, *[Path(d) for d in search_dirs]]

    path = find_local(youtube_id, dirs)
    if path is None:
        if backend == "local":
            raise SourceUnavailable(
                f"{youtube_id}: 在 {[str(d) for d in dirs]} 里找不到源视频。"
                f" local 后端不下载 —— 把文件命名成 {youtube_id}.mp4 放进去，或换 --backend ytdlp")
        path = _ytdlp_download(youtube_id, cache_dir, python, fmt, js_runtime,
                               ejs_remote, timeout)
    try:
        meta = verify_playable(path, need_until)
    except VideoError as exc:
        raise SourceUnavailable(str(exc)) from exc
    return SourceVideo(youtube_id=youtube_id, path=Path(path), meta=meta,
                       backend=backend, sha256=sha256_file(path))


def default_python() -> Optional[str]:
    """哪个解释器里有 yt_dlp。

    实测：`envs/perception_env` 有（yt_dlp 2026.07.04），`rt_env` / `hawor_env` 没有。
    另外 `envs/perception_env/bin/yt-dlp` 那个可执行壳是坏的（shebang 指向搬走前的
    旧路径），必须走 `python -m yt_dlp`。
    """
    try:
        from ..paths import P  # 延迟 import：本模块要能脱离仓库单测
        cand = P.env("perception")
    except Exception:
        return None
    return str(cand) if cand and os.path.exists(str(cand)) else None
