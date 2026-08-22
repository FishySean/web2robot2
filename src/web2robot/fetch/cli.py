"""``python -m web2robot.fetch`` —— 官方片段 → 原始 RGB 画面。

    scripts/s0_fetch_rgb.sh data/clips_official --out outputs/fetch
    scripts/s0_fetch_rgb.sh data/clips_official --out outputs/fetch --backend local \\
                            --source_dir /path/to/已经下好的视频
    scripts/s0_fetch_rgb.sh data/clips_official --clip -1r9yl-P-Ao_60.4_68.4 --dry_run

`--dry_run` 只解析元数据、不碰网络：出一张"每段要下哪支视频、截哪一段、
目录名和 `video_source` 差多少"的表。链路卡在下载时先用它把上游的账算清楚。

**一段失败不拖累其他段**：每段单独 try，失败写进 `fetch.jsonl` 的 `error` 字段，
退出码非 0（有失败就不能算成功），但已经成的段照样留在盘上。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from .align import align_report
from .download import BACKENDS, DEFAULT_FORMAT, SourceUnavailable, default_python, ensure_source
from .frames import extract_clip_rgb
from .sources import ClipSource, group_by_video, load_clip_sources


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m web2robot.fetch",
        description="按片段元数据里的 YouTube ID + 起止秒数，还原官方片段的 RGB 画面")
    ap.add_argument("clips_dir", help="片段库目录（每个子目录里有 scene.json）")
    ap.add_argument("--out", default="outputs/fetch", help="产物根目录（默认 outputs/fetch）")
    ap.add_argument("--clip", action="append", dest="clips", default=None,
                    help="只处理这些片段 id，可重复；默认全跑")
    ap.add_argument("--backend", choices=BACKENDS, default="ytdlp",
                    help="源视频从哪来：ytdlp 下载 / local 从 --source_dir 找现成的")
    ap.add_argument("--source_dir", action="append", dest="source_dirs", default=None,
                    help="local 后端的搜索目录，可重复")
    ap.add_argument("--cache_dir", default=None,
                    help="源视频缓存目录（默认 <out>/_sources）；同一支视频只下一次")
    ap.add_argument("--format", default=DEFAULT_FORMAT, help="yt-dlp 的 -f 表达式")
    ap.add_argument("--python", default=None, help="有 yt_dlp 的解释器（默认 perception_env）")
    ap.add_argument("--js_runtime", default="node",
                    help="yt-dlp 的 JS runtime（这台机器有 node v22）；传空字符串关掉")
    ap.add_argument("--ejs_remote", action="store_true",
                    help="允许 yt-dlp 运行时从 GitHub 拉 challenge solver 脚本（默认关）")
    ap.add_argument("--timeout", type=float, default=900.0, help="单支视频下载超时（秒）")
    ap.add_argument("--keep_png", action="store_true", help="额外存逐帧 PNG（体积大）")
    ap.add_argument("--crf", type=int, default=12, help="libx264 的 crf（默认 12，近无损）")
    ap.add_argument("--no_align", action="store_true", help="只截图不做对齐验收")
    ap.add_argument("--max_lag", type=int, default=30, help="互相关搜索的最大帧偏移")
    ap.add_argument("--dry_run", action="store_true", help="只解析元数据，不下载不截取")
    return ap


def plan_row(clip: ClipSource) -> Dict:
    d = clip.to_dict()
    return {k: d[k] for k in ("clip_id", "youtube_id", "start_seconds", "end_seconds",
                              "fps", "n_frames", "duration", "span_end",
                              "self_consistent", "name_gap_seconds")}


def _print_plan(clips: List[ClipSource]) -> None:
    groups = group_by_video(clips)
    print(f"{len(clips)} 段片段 → {len(groups)} 支源视频")
    hdr = f"{'clip_id':32s} {'youtube_id':13s} {'start':>9s} {'end':>9s} {'fps':>8s} {'n':>4s} {'name差(起,止)':>16s} 自洽"
    print(hdr)
    for c in clips:
        gap = c.name_gap_seconds
        gap_s = f"({gap[0]:+.2f},{gap[1]:+.2f})" if gap else "-"
        print(f"{c.clip_id:32s} {c.youtube_id:13s} {c.start_seconds:9.3f} {c.end_seconds:9.3f} "
              f"{c.fps:8.4f} {c.n_frames:4d} {gap_s:>16s} {'是' if c.self_consistent else '否'}")


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    clips = load_clip_sources(args.clips_dir, only=args.clips)
    out_root = Path(args.out)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_root / "_sources"

    _print_plan(clips)
    if args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / "fetch_plan.json", "w") as fh:
            json.dump([plan_row(c) for c in clips], fh, indent=2, ensure_ascii=False)
        print(f"\n--dry_run：只写了 {out_root / 'fetch_plan.json'}，没碰网络")
        return 0

    python = args.python or default_python()
    records: List[Dict] = []
    failures = 0
    for youtube_id, group in group_by_video(clips).items():
        need_until = max(c.span_end for c in group) + 1.0 / min(c.fps for c in group)
        try:
            source = ensure_source(
                youtube_id, cache_dir, need_until,
                backend=args.backend, search_dirs=args.source_dirs or (),
                python=python, fmt=args.format,
                js_runtime=(args.js_runtime or None), ejs_remote=args.ejs_remote,
                timeout=args.timeout)
        except SourceUnavailable as exc:
            failures += len(group)
            for c in group:
                records.append({"clip_id": c.clip_id, "youtube_id": youtube_id,
                                "ok": False, "error": str(exc)})
                print(f"[跳过] {c.clip_id}: {exc}", file=sys.stderr)
            continue
        print(f"[源视频] {youtube_id}: {source.meta.width}x{source.meta.height} "
              f"{source.meta.fps:.3f} fps {source.meta.duration:.1f}s sha256={source.sha256[:12]}")
        for clip in group:
            out_dir = out_root / clip.clip_id
            try:
                rec = extract_clip_rgb(source, clip, out_dir,
                                       pts_cache=cache_dir / f"{youtube_id}.pts.json",
                                       keep_png=args.keep_png, crf=args.crf)
                entry = {"clip_id": clip.clip_id, "youtube_id": youtube_id, "ok": True,
                         "out_dir": str(out_dir),
                         "max_abs_dt": rec["sampling"]["max_abs_dt"],
                         "within_half_frame": rec["sampling"]["within_half_frame"]}
                if not args.no_align:
                    rep = align_report(clip, Path(args.clips_dir) / clip.clip_id,
                                       out_dir / "rgb.mp4", out_dir, max_lag=args.max_lag)
                    entry["align_verdict"] = rep["verdict"]
                    entry["lag"] = {k: v["best_lag"] for k, v in rep["motion_lag"].items()}
                    if rep["verdict"] != "aligned":
                        failures += 1
                records.append(entry)
                print(f"[完成] {clip.clip_id}: {rec['rgb']['n_frames']} 帧, "
                      f"max|dt|={rec['sampling']['max_abs_dt']*1000:.1f} ms, "
                      f"对齐={entry.get('align_verdict', '未验')}")
            except Exception as exc:  # noqa: BLE001 —— 一段坏不拖累其他段
                failures += 1
                records.append({"clip_id": clip.clip_id, "youtube_id": youtube_id,
                                "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[失败] {clip.clip_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                traceback.print_exc(limit=3, file=sys.stderr)

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "fetch.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in records if r.get("ok"))
    print(f"\n成功 {ok}/{len(records)}；清单 {out_root / 'fetch.jsonl'}")
    return 0 if failures == 0 else 1


def main() -> None:
    sys.exit(run())
