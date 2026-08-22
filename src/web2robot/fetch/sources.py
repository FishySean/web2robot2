"""官方片段 → 原始视频坐标（YouTube ID + 起止秒数 + 采样时间轴）。

**为什么不直接拿目录名里的两个秒数**：那两个数是四舍五入过的，而且有的**根本对不上**。
`scene.json` 里的 `video_source` 才是权威来源。实测 10 段官方片段（2026-08-22）：

| 片段 | 目录名 | `video_source` | 差 |
|---|---|---|---|
| `-2cNMO9Mm3Q_192.4_209.2` | 192.4 – 209.2 | **195.790 – 205.523** | 起点 +3.39 s |
| `-1r9yl-P-Ao_231.8_241.5` | 231.8 – 241.5 | 231.760 – **239.693** | 终点 −1.81 s |
| 其余 8 段 | — | — | ≤ 0.04 s（就是四舍五入） |

`-2cNMO9Mm3Q` 那 3.39 s 按源视频 30 fps 算是 **102 帧**：照目录名截，那一段的画面
和 `hand_joints.bin` 会整体错开 102 帧，等于完全对不上。所以本模块只认 `video_source`，
目录名只用来做一致性检查（`name_gap_seconds`），不参与任何计算。

**时间轴**：片段是从源视频按等间隔采样出来的 `n_frames` 帧，
第 i 帧对应源视频时刻 ``t_i = start_seconds + i / fps``（i = 0 … n_frames−1）。
`fps` 不是整数也不是 30，逐段都不一样（15.0000 … 18.4041，见 `docs/BACKLOG.md` B9），
所以**不能**用"每隔 2 帧取 1 帧"这种整数跨步去还原。

**`self_consistent` 那条检查是干什么的**：`duration`、`fps`、`n_frames`、
`end - start` 四个数里只有三个自由度，官方元数据自己有 1 段不闭合
（`-1r9yl-P-Ao_231.8_241.5`：146 帧 / 18.4041 fps = 7.933 s 与 `video_source`
闭合，但和目录名的 9.7 s 差 1.8 s）。这条不闭合的段留着，只标记，不猜哪个数是对的
—— 真值等有了 RGB 画面之后用 `align.py` 量出来。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# 目录名形如 `<youtube_id>_<start>_<end>`，而 YouTube ID 本身可以带 `-` 和 `_`
# （实测 `--oo8_XIuOM`、`-1r9yl-P-Ao` 都是真实 ID），所以只能从右往左切两刀。
_NAME_RE = re.compile(r"^(?P<vid>.+)_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)$")

#: 元数据自洽的容差（秒）。官方的秒数写到 3 位小数，1 ms 足够。
CONSISTENCY_TOL = 1e-3


class ClipMetadataError(ValueError):
    """`scene.json` 缺字段或字段类型不对 —— 早失败，不猜。"""


@dataclass(frozen=True)
class ClipSource:
    """一段官方片段在源视频里的坐标，外加还原时间轴需要的参数。"""

    clip_id: str
    youtube_id: str
    start_seconds: float
    end_seconds: float
    fps: float
    n_frames: int
    duration: float
    width: int
    height: int
    #: 目录名里那两个秒数（只用于一致性检查；目录名不合规范时是 None）
    name_start: Optional[float] = None
    name_end: Optional[float] = None
    #: `action100m_metadata.video_duration`，用来判断下载到的源视频是不是同一支
    source_video_duration: Optional[float] = None

    # ---- 时间轴 --------------------------------------------------------
    @property
    def timeline(self) -> List[float]:
        """片段第 i 帧在**源视频**里的时刻，长度恒等于 `n_frames`。"""
        return [self.start_seconds + i / self.fps for i in range(self.n_frames)]

    @property
    def span_end(self) -> float:
        """最后一帧的时刻（不是 `end_seconds`：末帧之后还有 1/fps 的一帧时长）。"""
        return self.start_seconds + (self.n_frames - 1) / self.fps

    # ---- 一致性 --------------------------------------------------------
    @property
    def name_gap_seconds(self) -> Optional[tuple]:
        """(起点差, 终点差) = 目录名 − `video_source`；目录名不可解析时 None。"""
        if self.name_start is None or self.name_end is None:
            return None
        return (self.name_start - self.start_seconds, self.name_end - self.end_seconds)

    @property
    def self_consistent(self) -> bool:
        """`n_frames / fps` 是否同时等于 `duration` 和 `end − start`。"""
        by_count = self.n_frames / self.fps
        return (abs(by_count - self.duration) < CONSISTENCY_TOL
                and abs((self.end_seconds - self.start_seconds) - self.duration) < CONSISTENCY_TOL)

    def to_dict(self) -> Dict:
        gap = self.name_gap_seconds
        return {
            "clip_id": self.clip_id,
            "youtube_id": self.youtube_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "span_end": self.span_end,
            "self_consistent": self.self_consistent,
            "name_gap_seconds": list(gap) if gap else None,
            "source_video_duration": self.source_video_duration,
        }

    # ---- 构造 ----------------------------------------------------------
    @classmethod
    def from_clip_dir(cls, clip_dir) -> "ClipSource":
        clip_dir = Path(clip_dir)
        scene_path = clip_dir / "scene.json"
        if not scene_path.exists():
            raise ClipMetadataError(f"{clip_dir} 里没有 scene.json，不是官方片段目录")
        with open(scene_path) as fh:
            scene = json.load(fh)
        return cls.from_scene(scene, clip_id=scene.get("id") or clip_dir.name,
                             dir_name=clip_dir.name)

    @classmethod
    def from_scene(cls, scene: Dict, clip_id: str, dir_name: Optional[str] = None) -> "ClipSource":
        vs = scene.get("video_source") or {}
        if vs.get("type") != "youtube" or not vs.get("youtube_id"):
            raise ClipMetadataError(
                f"{clip_id}: video_source 不是 youtube 来源（拿到 {vs.get('type')!r}）—— "
                "本模块只会从 YouTube 还原 RGB，别的来源要另外接")
        for key in ("start_seconds", "end_seconds"):
            if not isinstance(vs.get(key), (int, float)):
                raise ClipMetadataError(f"{clip_id}: video_source.{key} 缺失或不是数字")

        stats = scene.get("stats") or {}
        cam = scene.get("camera") or {}
        n_frames = stats.get("n_frames")
        if not isinstance(n_frames, int) or n_frames <= 0:
            raise ClipMetadataError(f"{clip_id}: stats.n_frames 缺失或不是正整数")
        fps = scene.get("fps")
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ClipMetadataError(f"{clip_id}: fps 缺失或不是正数")

        name_start = name_end = None
        if dir_name:
            m = _NAME_RE.match(dir_name)
            if m:
                name_start, name_end = float(m.group("start")), float(m.group("end"))

        meta = scene.get("action100m_metadata") or {}
        return cls(
            clip_id=clip_id,
            youtube_id=str(vs["youtube_id"]),
            start_seconds=float(vs["start_seconds"]),
            end_seconds=float(vs["end_seconds"]),
            fps=float(fps),
            n_frames=int(n_frames),
            duration=float(scene.get("duration", n_frames / float(fps))),
            width=int(cam.get("width", 0)) or 0,
            height=int(cam.get("height", 0)) or 0,
            name_start=name_start,
            name_end=name_end,
            source_video_duration=(float(meta["video_duration"])
                                   if isinstance(meta.get("video_duration"), (int, float))
                                   else None),
        )


def load_clip_sources(clips_dir, only: Optional[List[str]] = None) -> List[ClipSource]:
    """扫一个片段库目录，按 `clip_id` 排序返回。

    `only` 给定时按 `clip_id` 过滤（缺哪个就报哪个，不静默少跑）。
    """
    clips_dir = Path(clips_dir)
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"片段库目录不存在：{clips_dir}")
    found = []
    for child in sorted(clips_dir.iterdir()):
        if child.is_dir() and (child / "scene.json").exists():
            found.append(ClipSource.from_clip_dir(child))
    if only:
        by_id = {c.clip_id: c for c in found}
        missing = [c for c in only if c not in by_id]
        if missing:
            raise KeyError(f"{clips_dir} 里没有这些片段：{missing}")
        found = [by_id[c] for c in only]
    return found


def group_by_video(clips: List[ClipSource]) -> Dict[str, List[ClipSource]]:
    """按 YouTube ID 归组 —— 一支源视频只下一次（实测 10 段只对应 6 支视频）。"""
    out: Dict[str, List[ClipSource]] = {}
    for c in clips:
        out.setdefault(c.youtube_id, []).append(c)
    return out
