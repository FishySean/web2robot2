"""``python -m web2robot.synth`` —— 视觉合成的命令行入口，两个子命令。

    # mask：官方 MANO 网格 → 手部掩码 + 一张能用眼睛看的核对图
    scripts/s5_hand_mask.sh data/clips_official --out outputs/synth
    scripts/s5_hand_mask.sh data/clips_official --clip -1r9yl-P-Ao_60.4_68.4
    scripts/s5_hand_mask.sh data/clips_official --no_align      # 做对照，看不对齐差多少

    # render：按片段那台相机把机器人渲出来（彩色 + 深度 + 掩码）
    scripts/s6_robot_render.sh data/clips_official \\
        --runs_dir outputs/retarget/collcmp --pattern '*_grid' --out outputs/synth/render

子命令是照 ``web2robot.perception`` 的先例加的（一个包一个 ``-m`` 入口，里面分子命令）。

**背景为什么是深度图**：真 RGB 还没到位（BACKLOG B12），而 `depth.npz` 是这条链路上
目前唯一一份真实成像 —— 手在深度图里的轮廓清清楚楚，掩码贴不贴合边缘一眼能看出来。
RGB 到位之后把背景换成 RGB，其余不用改。

**一段失败不拖累其他段**：每段单独 try，失败写进清单的 `error` 字段，退出码非 0，
已成的段照样留在盘上。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .handmask import (alignment_report, frame_alignments, hand_mask, joints_inside_fraction,
                       load_hand_mesh, load_joints_2d)
from .render import (ClipRobotRenderer, clip_camera, fovy_degrees, load_joint_trajectory,
                     load_root_frames, official_wrist_uv, wrist_alignment_report)


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


def _clip_dirs(clips_dir: Path, only: Optional[List[str]]) -> List[Path]:
    dirs = sorted(p for p in clips_dir.iterdir() if p.is_dir() and (p / "scene.json").exists())
    if only:
        keep = set(only)
        dirs = [p for p in dirs if p.name in keep]
    return dirs


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_h264(path: Path, frames_bgr: List[np.ndarray], fps: float) -> None:
    """cv2 写 mp4v 再用 ffmpeg 转 h264/yuv420p —— mpeg4 在 VSCode 里放不出来（照 twin/cli.py）。"""
    import cv2
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        tmp = str(Path(td) / "raw.mp4")
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
        for f in frames_bgr:
            vw.write(np.ascontiguousarray(f))
        vw.release()
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        str(path)], check=True)


# ── render：按片段相机渲机器人 ────────────────────────────────────────────────

def discover_runs(runs_dir: Path, pattern: str = "*") -> Dict[str, Path]:
    """扫一层子目录 → {clip_id: run_dir}。

    归属**读 `trajectory.npz` 里的 `clip_id`**，不靠目录名 —— 目录名是人取的
    （`fill_jar_grid`、`qalmp_neural`、`m7_b2s`…），猜不出来也不该猜。
    一个片段撞上多份产物就报错并列出来：该用哪份是**调用方的决定**，不是这里默认挑一个。
    """
    found: Dict[str, List[Path]] = {}
    for sub in sorted(runs_dir.glob(pattern)):
        if not sub.is_dir() or not (sub / "trajectory.npz").exists():
            continue
        if not (sub / "root_frames.npz").exists():
            continue
        with np.load(sub / "trajectory.npz", allow_pickle=True) as z:
            if "clip_id" not in z.files:
                continue
            cid = str(z["clip_id"])
        found.setdefault(cid, []).append(sub)
    dupes = {k: v for k, v in found.items() if len(v) > 1}
    if dupes:
        lines = "\n".join(f"  {k}: " + ", ".join(str(p) for p in v) for k, v in dupes.items())
        raise ValueError(
            f"同一片段有多份重定向产物，挑哪份得你说：\n{lines}\n"
            f"用 --pattern '*_grid' 之类筛，或者 --run CLIP_ID=RUN_DIR 逐个指定")
    return {k: v[0] for k, v in found.items()}


def robot_overlay(frame, bg_gray: Optional[np.ndarray], official_uv: Optional[np.ndarray],
                  label: str = "") -> np.ndarray:
    """一帧核对图（BGR）：深度底 + 机器人彩色贴在掩码里 + 手腕落点两色标记。

    洋红 = 机器人手腕的投影，绿 = 官方 2D 手腕。两点的距离就是画面上肉眼能看到的错位；
    分不清是我方链路还是官方数据自相矛盾时看 `wrist_alignment_report()` 的拆分。
    这里**不做深度排序合成** —— 那是下一块（`compose.py`）的事，这张图只验对齐。
    """
    import cv2
    H, W = frame.depth.shape
    base = (np.dstack([bg_gray] * 3) if bg_gray is not None
            else np.zeros((H, W, 3), dtype=np.uint8))
    if base.shape[:2] != (H, W):
        base = cv2.resize(base, (W, H), interpolation=cv2.INTER_NEAREST)
    canvas = base.copy()
    m = frame.mask
    canvas[m] = frame.rgb[m][:, ::-1]                       # 渲的是 RGB，画布是 BGR
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
    for side in ("left", "right"):
        uv = frame.wrist_uv[side]
        if np.isfinite(uv).all():
            cv2.drawMarker(canvas, (int(round(uv[0])), int(round(uv[1]))), (255, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
    if official_uv is not None:
        for uv in official_uv:
            if np.isfinite(uv).all():
                cv2.circle(canvas, (int(round(uv[0])), int(round(uv[1]))), 4, (0, 255, 0), -1)
    if label:
        cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_clip(clip_dir: Path, run_dir: Path, out_dir: Path, robot: str = "m7",
                video: bool = True, dump_npz: bool = False,
                n_cols: int = 3, n_rows: int = 3) -> Dict:
    """渲一整段：核对图 + （可选）视频 + （可选）深度/掩码 npz，并返回统计。"""
    import cv2

    n = clip_frames(clip_dir)
    camera = clip_camera(clip_dir)
    R_pf, t_pf = load_root_frames(run_dir, n)
    traj = load_joint_trajectory(run_dir, n)
    bg = depth_background(clip_dir, n)
    uv_2d = official_wrist_uv(clip_dir, n)

    take = set(np.linspace(0, n - 1, n_cols * n_rows).round().astype(int).tolist())
    tiles: Dict[int, np.ndarray] = {}
    frames: List[np.ndarray] = []
    areas, dmins = [], []
    depth_mm = np.zeros((n, int(camera["height"]), int(camera["width"])), dtype=np.uint16) \
        if dump_npz else None
    packed = [] if dump_npz else None

    with ClipRobotRenderer(camera, robot=robot) as rd:
        for t in range(n):
            f = rd.render(
                R_pf[t], t_pf[t], traj["q_left"][t], traj["q_right"][t],
                None if traj["q_left_fingers"] is None else traj["q_left_fingers"][t],
                None if traj["q_right_fingers"] is None else traj["q_right_fingers"][t],
                traj["left_finger_joint_names"], traj["right_finger_joint_names"])
            areas.append(f.area)
            d = f.depth[f.mask]
            dmins.append(float(d.min()) if d.size else np.nan)
            if video or t in take:
                vis = robot_overlay(f, None if bg is None else bg[t], uv_2d[t], label=f"t={t}")
                if t in take:
                    tiles[t] = vis
                if video:
                    frames.append(vis)
            if dump_npz:
                # 米 → 毫米，照官方 depth.npz 的口径；背景（inf）存 0 = 无效。
                mm = np.where(f.mask, np.minimum(f.depth * 1000.0, 65535.0), 0.0)
                depth_mm[t] = mm.astype(np.uint16)
                packed.append(np.packbits(f.mask.ravel()))

    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(tiles)
    grid = np.vstack([np.hstack([tiles[k] for k in order[r * n_cols:(r + 1) * n_cols]])
                      for r in range(n_rows)])
    png = out_dir / "robot_render_check.png"
    cv2.imwrite(str(png), grid)
    rec: Dict = {
        "png": str(png),
        "frames": order,
        "background": "depth.npz" if bg is not None else "black",
        "fovy_deg": round(fovy_degrees(camera), 4),
        "mean_area": float(np.mean(areas)),
        "min_area": float(np.min(areas)),
        "empty_frames": int(sum(1 for a in areas if a == 0.0)),
        "median_depth_min_m": float(np.nanmedian(dmins)),
    }
    if video:
        mp4 = out_dir / "robot_render.mp4"
        _write_h264(mp4, frames, traj["fps"])
        rec["mp4"] = str(mp4)
    if dump_npz:
        npz = out_dir / "robot_render.npz"
        np.savez_compressed(npz, depth_mm=depth_mm, mask=np.stack(packed),
                            shape=np.array([n, int(camera["height"]), int(camera["width"])]))
        rec["npz"] = str(npz)
    return rec


def run_render(args: argparse.Namespace) -> int:
    clips_dir = Path(args.clips_dir)
    out = Path(args.out)

    runs: Dict[str, Path] = {}
    if args.runs:
        for item in args.runs:
            if "=" not in item:
                print(f"--run 要写成 CLIP_ID=RUN_DIR，收到 {item!r}", file=sys.stderr)
                return 2
            cid, path = item.split("=", 1)
            runs[cid] = Path(path)
    elif args.runs_dir:
        try:
            runs = discover_runs(Path(args.runs_dir), args.pattern)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    else:
        print("要么给 --runs_dir，要么给 --run CLIP_ID=RUN_DIR", file=sys.stderr)
        return 2

    dirs = [p for p in _clip_dirs(clips_dir, args.clips) if p.name in runs]
    if not dirs:
        print(f"{clips_dir} 里没有和重定向产物对得上的片段（产物覆盖 {len(runs)} 段）",
              file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    records, failed = [], 0
    for i, clip in enumerate(dirs, 1):
        run_dir = runs[clip.name]
        print(f"[{i}/{len(dirs)}] {clip.name}  ← {run_dir}")
        rec: Dict = {"clip_id": clip.name, "run_dir": str(run_dir), "robot": args.robot}
        try:
            rec["n_frames"] = clip_frames(clip)
            rec["wrist_alignment"] = wrist_alignment_report(clip, run_dir, rec["n_frames"],
                                                            robot=args.robot)
            rec["render"] = render_clip(clip, run_dir, out / clip.name, robot=args.robot,
                                        video=not args.no_video, dump_npz=args.npz)
            for side in ("left", "right"):
                a = rec["wrist_alignment"][side]
                print(f"    {side:5s} 机器人↔MANO {a['robot_mano_px']:6.2f} px"
                      f"  MANO↔官方2D {a['mano_2d_px']:6.2f} px"
                      f"  合计 {a['robot_2d_px']:6.2f} px")
            print(f"    机器人占画面 {rec['render']['mean_area']:.3f}（均值）")
        except Exception as exc:                       # 一段坏不拖累其他段
            failed += 1
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    失败：{rec['error']}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        records.append(rec)

    _write_jsonl(out / "render.jsonl", records)
    ok = [r for r in records if "error" not in r]
    if ok:
        vals = [r["wrist_alignment"][s]["robot_mano_px"] for r in ok for s in ("left", "right")
                if r["wrist_alignment"][s]["robot_mano_px"] is not None]
        print(f"\n{len(ok)}/{len(records)} 段成功；机器人↔MANO 中位 {np.median(vals):.2f} px"
              f"（这一项量的是我方链路；官方 3D/2D 自身不一致另算）")
    print(f"清单 {out / 'render.jsonl'}")
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m web2robot.synth",
        description="视觉合成：手部掩码（mask）/ 按片段相机渲机器人（render）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mask", help="MANO 手部网格 → 手部掩码，附一张深度底的核对图")
    m.add_argument("clips_dir", help="片段库目录（每个子目录里有 scene.json）")
    m.add_argument("--out", default="outputs/synth", help="产物根目录（默认 outputs/synth）")
    m.add_argument("--clip", action="append", dest="clips", default=None,
                   help="只处理这些片段 id，可重复；默认全跑")
    m.add_argument("--no_align", action="store_true",
                   help="跳过逐帧对齐（只做对照用 —— 不对齐的掩码和画面差 9 px 量级）")
    m.add_argument("--no_masks", action="store_true", help="只出核对图，不存整段掩码")
    m.add_argument("--dilate", type=int, default=0, help="掩码膨胀半径（像素）")
    m.set_defaults(fn=run_mask)

    r = sub.add_parser("render", help="按片段那台相机渲机器人（彩色 + 深度 + 掩码）")
    r.add_argument("clips_dir", help="片段库目录")
    r.add_argument("--runs_dir", default=None,
                   help="重定向产物根目录：扫一层子目录，片段归属读 trajectory.npz 的 clip_id")
    r.add_argument("--pattern", default="*",
                   help="只认名字匹配这个 glob 的子目录（同一片段有 _grid/_neural 两份时用它挑）")
    r.add_argument("--run", action="append", dest="runs", default=None,
                   help="显式指定：--run CLIP_ID=RUN_DIR，可重复；比 --runs_dir 优先")
    r.add_argument("--out", default="outputs/synth/render", help="产物根目录")
    r.add_argument("--clip", action="append", dest="clips", default=None,
                   help="只处理这些片段 id，可重复")
    r.add_argument("--robot", default="m7", help="机器人（目前只验过 m7）")
    r.add_argument("--no_video", action="store_true", help="只出核对图，不出整段视频")
    r.add_argument("--npz", action="store_true",
                   help="把逐帧深度（uint16 毫米）和掩码存下来给下一步合成用")
    r.set_defaults(fn=run_render)
    return ap


def run_mask(args: argparse.Namespace) -> int:
    clips_dir = Path(args.clips_dir)
    out = Path(args.out)
    align = not args.no_align

    dirs = _clip_dirs(clips_dir, args.clips)
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

    _write_jsonl(out / "handmask.jsonl", records)
    ok = [r for r in records if "error" not in r]
    if ok:
        print(f"\n{len(ok)}/{len(records)} 段成功；非指尖落点均值 "
              f"{np.mean([r['joints_inside']['non_tip']['fraction'] for r in ok]):.3f}")
    print(f"清单 {out / 'handmask.jsonl'}")
    return 0 if failed == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
