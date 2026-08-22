"""官方片段的原始 RGB 画面还原（流水线第 0 步）。

官方发布的片段里**一帧真 RGB 都没有**（`depth.mp4` 是深度、`thumb.jpg` 是深度
的伪彩缩略图、`bg_template.png` 是 16 位深度背景板），而"抠掉人换成机器人"这件事
的输入必须是 RGB。片段元数据里的 `video_source` 给了 YouTube ID 和起止秒数 ——
这个模块就是把那两个数还原成画面，并且量出还原得对不对。

    from web2robot.fetch import load_clip_sources, ensure_source, extract_clip_rgb

细节见 README.md。
"""
from .sources import ClipSource, ClipMetadataError, group_by_video, load_clip_sources
from .download import SourceUnavailable, SourceVideo, ensure_source
from .frames import extract_clip_rgb
from .align import align_report, motion_energy, xcorr_best_lag
from .video import VideoError, VideoMeta, probe, verify_playable

__all__ = ["ClipSource", "ClipMetadataError", "group_by_video", "load_clip_sources",
           "SourceUnavailable", "SourceVideo", "ensure_source",
           "extract_clip_rgb", "align_report", "motion_energy", "xcorr_best_lag",
           "VideoError", "VideoMeta", "probe", "verify_playable"]
