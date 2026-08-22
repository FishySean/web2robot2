"""``python -m web2robot.synth`` —— 出手部掩码 + 一张能用眼睛看的核对图。

    scripts/s5_hand_mask.sh data/clips_official --out outputs/synth
    scripts/s5_hand_mask.sh data/clips_official --clip -1r9yl-P-Ao_60.4_68.4
    scripts/s5_hand_mask.sh data/clips_official --no_align      # 做对照，看不对齐差多少

**背景为什么是深度图**：真 RGB 还没到位（BACKLOG B12），而 `depth.npz` 是这条链路上
目前唯一一份真实成像 —— 手在深度图里的轮廓清清楚楚，掩码贴不贴合边缘一眼能看出来。
RGB 到位之后把背景换成 RGB，其余不用改。

**一段失败不拖累其他段**：每段单独 try，失败写进 `handmask.jsonl` 的 `error`，
退出码非 0，已成的段照样留在盘上。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .handmask import (alignment_report, frame_alignments, hand_mask, joints_inside_fraction,
                       load_hand_mesh, load_joints_2d)


def clip_frames(clip_dir: Path) -> int:
    """帧数只认 `scene.json` 的 `stats.n_frames`（和 `fetch/` 同一个口径）。"""
    with open(clip_dir / "scene.json") as fh:
        return int(json.load(fh)["stats"]["n_frames"])


def depth_background(clip_dir: Path, n_frames: int) -> Optional[np.ndarray]:
    """`depth.npz` → (T,H,W) uint8 灰度。没这个文件就返回 None（背景画黑底）。

    逐帧各自拉伸对比度：整段统一拉伸的话，近处的手在远景帧里会糊成一片白。
    """
    p = clip_dir / "depth.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        arr = z[z.files[0]]
    if len(arr) != n_frames:
        raise ValueError(f"{p}: {len(arr)} 帧 ≠ scene.json 的 {n_frames} 帧")
    out = np.zeros(arr.shape, dtype=np.uint8)
    for t, d in enumerate(arr):
        good = d > 0
        if not good.any():
            continue
        lo, hi = np.percentile(d[good], [2, 98])
        if hi <= lo:
            continue
        v = np.clip((d.astype(np.float32) - lo) / (hi - lo), 0, 1)
        out[t] = (v * 255).astype(np.uint8)
        out[t][~good] = 0
    return out


def montage(clip_dir: Path, n_frames: int, out_png: Path, align: bool = True,
            n_cols: int = 3, n_rows: int = 3) -> Dict:
    """挑 n_cols×n_rows 帧铺一张核对图：深度底 + 掩码染色 + 掩码轮廓 + 官方 2D 关节点。

    关节点画在掩码上面 —— 点落在染色区里就是对的，飘在外面一眼看得见。
    """
    import cv2

    verts, faces = load_hand_mesh(clip_dir, n_frames)
    with open(clip_dir / "camera.json") as fh:
        camera = json.load(fh)
    W, H = int(camera["width"]), int(camera["height"])
    fits = frame_alignments(clip_dir, n_frames) if align else None
    joints = load_joints_2d(clip_dir, n_frames)
    bg = depth_background(clip_dir, n_frames)

    take = np.linspace(0, n_frames - 1, n_cols * n_rows).round().astype(int)
    tiles = []
    for t in take:
        base = (np.dstack([bg[t]] * 3) if bg is not None
                else np.zeros((H, W, 3), dtype=np.uint8))
        if base.shape[:2] != (H, W):                  # 深度和内参对不上就别硬贴
            base = cv2.resize(base, (W, H), interpolation=cv2.INTER_NEAREST)
        canvas = base.copy()
        for h, color in ((0, (60, 200, 255)), (1, (255, 120, 60))):
            m = hand_mask(verts[t], faces, camera, hands=("left" if h == 0 else "right"),
                          align=None if fits is None else fits[t])
            if not m.any():
                continue
            canvas[m] = (0.45 * canvas[m] + 0.55 * np.array(color)).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, color, 1)
        for h in (0, 1):
            for uv in joints[t, h]:
                if not np.isfinite(uv).all():
                    continue
                cv2.circle(canvas, (int(round(uv[0])), int(round(uv[1]))), 2, (0, 255, 0), -1)
        cv2.putText(canvas, f"t={int(t)}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    grid = np.vstack([np.hstack(tiles[r * n_cols:(r + 1) * n_cols]) for r in range(n_rows)])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), grid[:, :, ::-1])       # cv2 存 BGR
    return {"png": str(out_png), "frames": [int(t) for t in take],
            "background": "depth.npz" if bg is not None else "black"}


def masks_npz(clip_dir: Path, n_frames: int, out_npz: Path, align: bool = True) -> Dict:
    """整段左右手掩码按位打包存 npz（口径照官方 `masks.npz`：`unpackbits` 解回来）。

    155×480×853 的 bool 存 63 MB，打包后 8 MB。下游合成读这份，不用重算。
    """
    verts, faces = load_hand_mesh(clip_dir, n_frames)
    with open(clip_dir / "camera.json") as fh:
        camera = json.load(fh)
    fits = frame_alignments(clip_dir, n_frames) if align else None
    packed = {}
    area = []
    for h, name in ((0, "left"), (1, "right")):
        rows = []
        for t in range(n_frames):
            m = hand_mask(verts[t], faces, camera, hands=name,
                          align=None if fits is None else fits[t])
            rows.append(np.packbits(m.ravel()))
            area.append(float(m.mean()))
        packed[name] = np.stack(rows)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **packed,
                        shape=np.array([n_frames, int(camera["height"]), int(camera["width"])]))
    return {"npz": str(out_npz), "mean_area": float(np.mean(area)),
            "max_area": float(np.max(area)),
            "empty_frames": int(sum(1 for a in area if a == 0.0))}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m web2robot.synth",
        description="MANO 手部网格 → 手部掩码，附一张深度底的核对图")
    ap.add_argument("clips_dir", help="片段库目录（每个子目录里有 scene.json）")
    ap.add_argument("--out", default="outputs/synth", help="产物根目录（默认 outputs/synth）")
    ap.add_argument("--clip", action="append", dest="clips", default=None,
                    help="只处理这些片段 id，可重复；默认全跑")
    ap.add_argument("--no_align", action="store_true",
                    help="跳过逐帧对齐（只做对照用 —— 不对齐的掩码和画面差 9 px 量级）")
    ap.add_argument("--no_masks", action="store_true", help="只出核对图，不存整段掩码")
    ap.add_argument("--dilate", type=int, default=0, help="掩码膨胀半径（像素）")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    clips_dir = Path(args.clips_dir)
    out = Path(args.out)
    align = not args.no_align

    dirs = sorted(p for p in clips_dir.iterdir() if p.is_dir() and (p / "scene.json").exists())
    if args.clips:
        keep = set(args.clips)
        dirs = [p for p in dirs if p.name in keep]
    if not dirs:
        print(f"{clips_dir} 下没有片段目录", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    records, failed = [], 0
    for i, clip in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] {clip.name}")
        rec: Dict = {"clip_id": clip.name, "aligned": align}
        try:
            n = clip_frames(clip)
            rec["n_frames"] = n
            rec["alignment"] = alignment_report(clip, n)
            rec["montage"] = montage(clip, n, out / clip.name / "handmask_check.png", align=align)
            if not args.no_masks:
                rec["masks"] = masks_npz(clip, n, out / clip.name / "hand_masks.npz", align=align)
            rec["joints_inside"] = joints_inside_fraction(clip, n, dilate=args.dilate, align=align)
            nt = rec["joints_inside"]["non_tip"]["fraction"]
            print(f"    非指尖落点 {nt:.3f}  指尖 {rec['joints_inside']['tip']['fraction']:.3f}"
                  f"  拟合残差 {rec['alignment']['residual_px']['median']:.2f} px")
        except Exception as exc:                       # 一段坏不拖累其他段
            failed += 1
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    失败：{rec['error']}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        records.append(rec)

    with open(out / "handmask.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ok = [r for r in records if "error" not in r]
    if ok:
        print(f"\n{len(ok)}/{len(records)} 段成功；非指尖落点均值 "
              f"{np.mean([r['joints_inside']['non_tip']['fraction'] for r in ok]):.3f}")
    print(f"清单 {out / 'handmask.jsonl'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
