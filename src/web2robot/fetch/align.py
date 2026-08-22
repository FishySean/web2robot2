"""下载来的 RGB 和片段原有数据在时间轴上对不对得上 —— 三条判据，都出数字。

为什么要三条：截出来的画面"看着像"没有意义，必须能量出偏了几帧。
现成的对齐真值有三样，各自能钉住一件事：

1. **帧数**（`depth.mp4` / `mask.mp4` 的解码帧数恰好等于 `stats.n_frames`）
   → 钉住"取的帧数对"。这条最弱但最先炸：错一帧这里就露。
2. **运动能量曲线的互相关**（RGB 相邻帧差 vs 深度图相邻帧差）
   → 钉住"时间上没整体平移"。峰值该落在 lag=0；落在别处就是整段错位，
   `-2cNMO9Mm3Q` 那段用目录名截会错 102 帧，这条会直接把它抓出来。
3. **2D 手部关节落点**（`hand_joints_2d.bin`，没有就用 `hand_joints.bin` 3D
   加 `camera.json` 内参投影）→ 钉住"每一帧的内容真的是那一帧"。
   出四宫格叠图给人眼确认（指标 ≠ 画面可信，见 memory `feedback-metric-vs-visual`），
   同时给一个数：关节点落在画面内的比例。

**判据的方向性**：lag 是"深度相对 RGB 要移多少帧才最像"。lag>0 表示 RGB 取早了。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sources import ClipSource
from .video import probe, stream_gray


def motion_energy(frames) -> np.ndarray:
    """逐帧的"画面变了多少"：与上一帧的平均绝对差。第 0 帧定义为 0。"""
    prev = None
    out: List[float] = []
    for f in frames:
        cur = f.astype(np.float32)
        out.append(0.0 if prev is None else float(np.mean(np.abs(cur - prev))))
        prev = cur
    return np.asarray(out, dtype=np.float64)


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    s = np.linalg.norm(x)
    return x / s if s > 0 else x


def xcorr_best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 30) -> Tuple[int, float, Dict[int, float]]:
    """`b` 相对 `a` 平移多少帧最像。返回 (最佳 lag, 该处相关系数, 整条曲线)。

    方向约定：lag = L 意味着 ``b[i] ≈ a[i+L]`` —— b 的内容在 a 里出现得更晚 L 帧。
    只在重叠区间上算零均值归一化相关，所以不同 lag 的分数可比。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 4:
        return 0, float("nan"), {}
    curve: Dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa, bb = a[lag:n], b[: n - lag]
        else:
            aa, bb = a[: n + lag], b[-lag:n]
        if len(aa) < 4:
            continue
        curve[lag] = float(np.dot(_norm(aa), _norm(bb)))
    if not curve:
        return 0, float("nan"), {}
    best = max(curve, key=lambda k: curve[k])
    return best, curve[best], curve


# ---------------------------------------------------------------------------
# 2D 手部关节
# ---------------------------------------------------------------------------
def load_joints_2d(clip_dir, clip: ClipSource) -> Optional[np.ndarray]:
    """(T, 2, 21, 2) 像素坐标；拿不到返回 None。

    优先 `hand_joints_2d.bin`（官方 HF 上有，我们本地那 10 段没同步下来）；
    否则拿 `hand_joints.bin` 的 3D 关节用 `camera.json` 内参投影 —— 官方的 3D
    关节就在相机系里（`utils/pose_utils.py` 整条链都按这个口径），所以
    ``u = f·x/z + cx``、``v = f·y/z + cy`` 就是它自己的成像模型，不是我们外加的假设。
    NaN 表示这一帧这只手不在（`hand_meta.json` 的 `nan_means_absent`）。
    """
    clip_dir = Path(clip_dir)
    T = clip.n_frames
    p2 = clip_dir / "hand_joints_2d.bin"
    if p2.exists():
        arr = np.fromfile(p2, dtype=np.float32)
        if arr.size == T * 2 * 21 * 2:
            return arr.reshape(T, 2, 21, 2).astype(np.float64)
    p3 = clip_dir / "hand_joints.bin"
    cam_path = clip_dir / "camera.json"
    if not (p3.exists() and cam_path.exists()):
        return None
    arr = np.fromfile(p3, dtype=np.float32)
    if arr.size != T * 2 * 21 * 3:
        return None
    j3 = arr.reshape(T, 2, 21, 3).astype(np.float64)
    with open(cam_path) as fh:
        cam = json.load(fh)
    f, cx, cy = float(cam["focal"]), float(cam["cx"]), float(cam["cy"])
    z = j3[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = f * j3[..., 0] / z + cx
        v = f * j3[..., 1] / z + cy
    uv = np.stack([u, v], axis=-1)
    uv[~np.isfinite(uv)] = np.nan
    return uv


def in_frame_fraction(joints_2d: np.ndarray, width: int, height: int) -> float:
    """有效关节点里落在画面内的比例（NaN 不算分母）。"""
    uv = joints_2d.reshape(-1, 2)
    valid = np.isfinite(uv).all(axis=1)
    if not valid.any():
        return float("nan")
    uv = uv[valid]
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    return float(inside.mean())


def joint_overlay_montage(rgb_path, joints_2d: np.ndarray, out_png, n_panels: int = 4,
                          radius: int = 3) -> Optional[Path]:
    """在等间隔取的几帧 RGB 上画 2D 关节，拼成一张图给人眼看。

    左手画一种颜色、右手另一种。**这张图是验收的一部分**：帧数和互相关都
    只说"没整体错位"，"这一帧的手真的在这个位置"只有眼睛能确认。
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    meta = probe(rgb_path)
    T = min(len(joints_2d), meta.nb_frames or len(joints_2d))
    if T <= 0:
        return None
    picks = sorted({int(round(i)) for i in np.linspace(0, T - 1, n_panels)})
    frames: Dict[int, np.ndarray] = {}
    for idx, g in enumerate(stream_gray(rgb_path)):
        if idx in picks:
            frames[idx] = g.copy()
        if idx >= picks[-1]:
            break
    if not frames:
        return None
    colors = [(255, 80, 80), (80, 160, 255)]
    panels = []
    for idx in picks:
        g = frames.get(idx)
        if g is None:
            continue
        im = Image.fromarray(np.repeat(g[:, :, None], 3, axis=2))
        dr = ImageDraw.Draw(im)
        for hand in range(joints_2d.shape[1]):
            for u, v in joints_2d[idx, hand]:
                if np.isfinite(u) and np.isfinite(v):
                    dr.ellipse([u - radius, v - radius, u + radius, v + radius],
                               fill=colors[hand % 2])
        dr.text((6, 6), f"frame {idx}", fill=(255, 255, 0))
        panels.append(im)
    if not panels:
        return None
    w, h = panels[0].size
    cols = 2 if len(panels) > 1 else 1
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for k, im in enumerate(panels):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def align_report(clip: ClipSource, clip_dir, rgb_path, out_dir,
                 max_lag: int = 30, montage: bool = True) -> Dict:
    """把三条判据跑一遍，写 `align_report.json`（+ 叠图），返回报告 dict。"""
    clip_dir, out_dir = Path(clip_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_meta = probe(rgb_path)

    rgb_gray = list(stream_gray(rgb_path))
    counts = {"scene_n_frames": clip.n_frames, "rgb_decoded": len(rgb_gray)}
    energies = {"rgb": motion_energy(iter(rgb_gray))}

    lags: Dict[str, Dict] = {}
    for name in ("depth.mp4", "mask.mp4"):
        p = clip_dir / name
        if not p.exists():
            continue
        gray = list(stream_gray(p, width=rgb_meta.width, height=rgb_meta.height))
        counts[f"{name}_decoded"] = len(gray)
        e = motion_energy(iter(gray))
        energies[name] = e
        lag, corr, curve = xcorr_best_lag(energies["rgb"], e, max_lag=max_lag)
        lags[name] = {
            "best_lag": lag,
            "corr_at_best": corr,
            "corr_at_zero": curve.get(0, float("nan")),
            "curve": {str(k): round(v, 4) for k, v in sorted(curve.items())},
        }

    joints = load_joints_2d(clip_dir, clip)
    joints_block: Dict = {"available": joints is not None}
    if joints is not None:
        joints_block.update({
            "source": "hand_joints_2d.bin" if (clip_dir / "hand_joints_2d.bin").exists()
                      else "projected from hand_joints.bin",
            "in_frame_fraction": in_frame_fraction(joints, rgb_meta.width, rgb_meta.height),
        })
        if montage:
            png = joint_overlay_montage(rgb_path, joints, out_dir / "align_montage.png")
            joints_block["montage"] = str(png) if png else None

    counts_ok = all(v == clip.n_frames for k, v in counts.items() if k.endswith("decoded"))
    lag_ok = all(v["best_lag"] == 0 for v in lags.values()) if lags else None
    report = {
        "clip_id": clip.clip_id,
        "counts": counts,
        "counts_ok": counts_ok,
        "motion_lag": lags,
        "lag_ok": lag_ok,
        "joints_2d": joints_block,
        # 只有三条都有结论且都过才算 aligned；缺判据写 unknown，不写 pass
        "verdict": ("aligned" if (counts_ok and lag_ok) else
                    "misaligned" if (lags and not lag_ok) or not counts_ok else "unknown"),
    }
    with open(out_dir / "align_report.json", "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report
