"""MANO 手部网格 → 手部掩码（不用分割模型）。

官方每段片段给了：

* `hand_verts.bin` —— `(T, 2, 778, 3)` float32，**相机系、米制**（实测 z 在 1.5–2.2 m）。
  第 1 维是左右手；这只手这一帧不在场时整片 NaN（口径同 `hand_joints.bin`，
  见 `hand_meta.json` 的 `nan_means_absent`）。
* `hand_faces.bin` —— `(1538, 3)` int32，0 基顶点索引，两只手共用同一套拓扑。

投影用 `camera.json` 自己的成像模型 ``u = f·x/z + cx``、``v = f·y/z + cy``，
和 `fetch/align.py` 里那份一致 —— 官方 3D 全在相机系，这不是我们外加的假设。

**但只投影是对不上画面的。** 实测（`-0RheyDV3a0_474.8_487.3`，见 `frame_alignments`
的说明和 `docs/VISUAL_SYNTH_INPUTS.md`）：裸投影与官方 `hand_joints_2d.bin` 差 9.3 px
中位；逐帧逐手拟合一个"缩放+平移"能降到 3.7 px，而解出来的缩放中位只有 0.67 且逐帧
在 0.32–0.95 漂。换句话说官方 3D 手的**尺度/深度逐帧不定**，像素空间的真值是那份
2D 关节。所以流程是「投影 → 逐帧逐手相似变换 → 光栅化」，中间那步默认开。

**没有 z-buffer**：手掩码是二值的并集，不需要谁遮谁（自遮挡不影响"这个像素是手"）。
真正要比深度的是"机器人 vs 场景"，那一步用 `depth.npz` 的毫米深度和 MuJoCo 渲出来的
深度，不走这里。

怎么知道掩码对：`joints_inside_fraction()` 拿 `hand_joints_2d.bin` 的 21 个关节点核对
落点 —— 但开了对齐之后这条判据是**半自证**的（对齐用的就是这份关节），它证的是
"网格形状加这个变换能包住关节"。真正独立的复验要等真 RGB 到位（BACKLOG B12）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

N_MANO_VERTS = 778
N_MANO_FACES = 1538
N_JOINTS = 21
#: 五个指尖关节的下标。指尖**骨节点在网格表面上甚至略微在外**（骨头的端点，不是
#: 肉的端点），所以它落在掩码外是几何本身如此，不是掩码错位 —— 实测落在外面的
#: 指尖离掩码边界中位只有 2 px，其余关节 1 px。判掩码好坏时这两组必须分开看。
TIP_JOINTS = (4, 8, 12, 16, 20)


class HandMeshMissing(FileNotFoundError):
    """片段目录里没有 MANO 网格（本地那 10 段原来就没同步，用
    `scripts/dev/fetch_official_extras.py` 补）。"""


def _camera(clip_dir: Path) -> Dict:
    with open(clip_dir / "camera.json") as fh:
        return json.load(fh)


def load_hand_mesh(clip_dir, n_frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (verts (T,2,778,3) float32 相机系米制, faces (1538,3) int32)。

    `n_frames` 给了就核对元素个数对不上直接报错 —— 形状靠算而不是靠信，
    因为 `.bin` 里没有形状信息，错了会静默变成"另一段的数据"。
    """
    clip_dir = Path(clip_dir)
    vp, fp = clip_dir / "hand_verts.bin", clip_dir / "hand_faces.bin"
    for p in (vp, fp):
        if not p.exists():
            raise HandMeshMissing(
                f"{p} 不存在。本地片段目录原来只有重定向那四件套，MANO 网格在 HF 上，"
                f"用 scripts/dev/fetch_official_extras.py 补")
    faces = np.fromfile(fp, dtype=np.int32)
    if faces.size % 3 or faces.size // 3 != N_MANO_FACES:
        raise ValueError(f"{fp}: 面片数不是 {N_MANO_FACES}（读到 {faces.size / 3}）")
    faces = faces.reshape(-1, 3)
    if faces.max() >= N_MANO_VERTS or faces.min() < 0:
        raise ValueError(f"{fp}: 顶点索引越界（{faces.min()}..{faces.max()}）")

    verts = np.fromfile(vp, dtype=np.float32)
    per_frame = 2 * N_MANO_VERTS * 3
    if verts.size % per_frame:
        raise ValueError(f"{vp}: {verts.size} 个 float 不是 {per_frame} 的整数倍")
    T = verts.size // per_frame
    if n_frames is not None and T != n_frames:
        raise ValueError(f"{vp}: 顶点是 {T} 帧，scene.json 说 {n_frames} 帧 —— 素材对不上")
    return verts.reshape(T, 2, N_MANO_VERTS, 3), faces


def project_points(xyz: np.ndarray, camera: Dict) -> np.ndarray:
    """相机系 (…,3) → 像素 (…,2)。

    出 NaN 的两种情况：z ≤ 0（在相机后面，没有像点）；输入本身有非有限值（官方口径
    "不在场的手整片 NaN"）。**两个通道一起判死** —— 只有 x 是 NaN 时 v 还算得出来，
    留一个"半个 NaN"的点等着下游忘记检查，不如当场整点作废。
    """
    f = float(camera["focal"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    z = xyz[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = f * xyz[..., 0] / z + cx
        v = f * xyz[..., 1] / z + cy
    bad = ~np.isfinite(xyz).all(axis=-1) | (z <= 0)
    u = np.where(bad, np.nan, u)
    v = np.where(bad, np.nan, v)
    return np.stack([u, v], axis=-1)


def solve_similarity(src: np.ndarray, dst: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """最小二乘解 (s, tx, ty)，使 ``s·src + t ≈ dst``（各向同性缩放 + 平移，不含旋转）。

    点数不够或退化返回 None。只用有限值的那些点。
    """
    m = np.isfinite(src).all(axis=-1) & np.isfinite(dst).all(axis=-1)
    a, b = src[m], dst[m]
    if len(a) < 3:
        return None
    A = np.zeros((2 * len(a), 3))
    rhs = np.empty(2 * len(a))
    A[0::2, 0], A[0::2, 1], rhs[0::2] = a[:, 0], 1.0, b[:, 0]
    A[1::2, 0], A[1::2, 2], rhs[1::2] = a[:, 1], 1.0, b[:, 1]
    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    s, tx, ty = (float(x) for x in sol)
    if not np.isfinite([s, tx, ty]).all() or s <= 0:
        return None
    return s, tx, ty


def frame_alignments(clip_dir, n_frames: Optional[int] = None) -> np.ndarray:
    """(T, 2, 3) 的逐帧逐手 (s, tx, ty)；解不出来的填 NaN。

    **为什么需要这一步**（实测，不是保险起见）：把 `hand_joints.bin` 的 3D 关节用
    `camera.json` 投影，和官方 `hand_joints_2d.bin` 对不上 —— 全段统一的
    缩放+平移拟合完还剩 4–19 px 残差，而**逐帧逐手**拟合只剩 3.7 px，且解出来的
    缩放在 0.32–0.95 之间逐帧变化（`-0RheyDV3a0_474.8_487.3` 实测，中位 0.67）。

    也就是说官方 3D 手的**尺度/深度是逐帧漂的**（单目手部深度的老问题，见
    memory `wilor-depth-two-strategies`），`hand_joints_2d.bin` 才是像素空间里的证据。
    所以要做"和画面对得上的掩码"，必须逐帧把投影后的网格对齐到 2D 关节上。
    """
    clip_dir = Path(clip_dir)
    camera = _camera(clip_dir)
    T = n_frames
    j2 = load_joints_2d(clip_dir, T)
    T = len(j2)
    j3 = np.fromfile(clip_dir / "hand_joints.bin", dtype=np.float32)
    if j3.size != T * 2 * N_JOINTS * 3:
        raise ValueError(f"{clip_dir / 'hand_joints.bin'}: 元素个数与 {T} 帧对不上")
    uv3 = project_points(j3.reshape(T, 2, N_JOINTS, 3), camera)

    out = np.full((T, 2, 3), np.nan)
    for t in range(T):
        for h in (0, 1):
            sol = solve_similarity(uv3[t, h], j2[t, h])
            if sol is not None:
                out[t, h] = sol
    return out


def _apply_similarity(uv: np.ndarray, align: Optional[np.ndarray]) -> np.ndarray:
    if align is None or not np.isfinite(align).all():
        return uv
    s, tx, ty = float(align[0]), float(align[1]), float(align[2])
    return np.stack([s * uv[..., 0] + tx, s * uv[..., 1] + ty], axis=-1)


def _fill(uv: np.ndarray, faces: np.ndarray, width: int, height: int) -> np.ndarray:
    """把三角面片填成二值掩码。uv 里有 NaN 的面片整片丢掉。"""
    import cv2  # rt_env 里现成（质检模块也在用）

    mask = np.zeros((height, width), dtype=np.uint8)
    tri = uv[faces]                                   # (F, 3, 2)
    ok = np.isfinite(tri).all(axis=(1, 2))
    tri = tri[ok]
    if not len(tri):
        return mask.astype(bool)
    # 全画到画布外的面片没必要送进去（画面 853×480，手偶尔跑出边界）
    inside = ((tri[..., 0] > -1) & (tri[..., 0] < width) &
              (tri[..., 1] > -1) & (tri[..., 1] < height)).any(axis=1)
    tri = tri[inside]
    if not len(tri):
        return mask.astype(bool)
    polys = [np.round(t).astype(np.int32) for t in tri]
    cv2.fillPoly(mask, polys, 255)                    # 一次调用画完全部面片
    return mask.astype(bool)


def alignment_report(clip_dir, n_frames: Optional[int] = None) -> Dict:
    """量一下 `frame_alignments()` 拟合得有多准，以及解出来的缩放漂了多少。

    残差是拟合后 21 个关节的像素距离。缩放的离散程度就是"3D 手的深度逐帧在漂"这件事
    的量化 —— 如果它恒等于 1，说明 3D→2D 本来就对得上，这一步就是多余的。
    """
    clip_dir = Path(clip_dir)
    camera = _camera(clip_dir)
    fits = frame_alignments(clip_dir, n_frames)
    T = len(fits)
    j2 = load_joints_2d(clip_dir, T)
    j3 = np.fromfile(clip_dir / "hand_joints.bin", dtype=np.float32).reshape(T, 2, N_JOINTS, 3)
    uv3 = project_points(j3, camera)

    res, scales, raw = [], [], []
    for t in range(T):
        for h in (0, 1):
            if not np.isfinite(fits[t, h]).all():
                continue
            m = np.isfinite(uv3[t, h]).all(axis=-1) & np.isfinite(j2[t, h]).all(axis=-1)
            if not m.any():
                continue
            fit = _apply_similarity(uv3[t, h], fits[t, h])
            res.append(np.linalg.norm(fit[m] - j2[t, h][m], axis=-1))
            raw.append(np.linalg.norm(uv3[t, h][m] - j2[t, h][m], axis=-1))
            scales.append(float(fits[t, h, 0]))
    if not res:
        return {"n_fits": 0}
    res = np.concatenate(res)
    raw = np.concatenate(raw)
    s = np.asarray(scales)
    return {
        "n_frames": int(T), "n_fits": int(len(s)), "n_points": int(res.size),
        "residual_px": {"mean": float(res.mean()), "median": float(np.median(res)),
                        "p90": float(np.percentile(res, 90)), "max": float(res.max())},
        "unaligned_px": {"mean": float(raw.mean()), "median": float(np.median(raw))},
        "scale": {"median": float(np.median(s)), "min": float(s.min()), "max": float(s.max())},
    }


def hand_mask(verts_frame: np.ndarray, faces: np.ndarray, camera: Dict,
              hands: str = "both", dilate: int = 0,
              align: Optional[np.ndarray] = None) -> np.ndarray:
    """一帧的手部掩码 (H, W) bool。

    `verts_frame`：(2, 778, 3)。`hands`：`both` / `left` / `right`
    （第 0 维是左手 —— 与 `hand_joints.bin` 同序）。
    `dilate`：膨胀半径（像素）。合成时手边缘要盖住一点点，留这个口。
    `align`：(2, 3) 的逐手 (s, tx, ty)，来自 `frame_alignments()`；给了就在
    投影之后做这个像素域相似变换。**要和画面对齐必须给** —— 原因见
    `frame_alignments` 的说明。NaN 的那只手当没给，退回裸投影。
    """
    W, H = int(camera["width"]), int(camera["height"])
    idx = {"both": (0, 1), "left": (0,), "right": (1,)}[hands]
    mask = np.zeros((H, W), dtype=bool)
    for h in idx:
        uv = project_points(verts_frame[h], camera)
        uv = _apply_similarity(uv, None if align is None else align[h])
        mask |= _fill(uv, faces, W, H)
    if dilate > 0 and mask.any():
        import cv2
        k = 2 * int(dilate) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def hand_mask_series(clip_dir, n_frames: Optional[int] = None, hands: str = "both",
                     dilate: int = 0, align: bool = True) -> Iterator[np.ndarray]:
    """逐帧产出手部掩码。逐帧 yield（整段 155×480×853 bool 也才 63 MB，但没必要都留着）。

    `align=True`（默认）会先算一遍逐帧逐手相似变换 —— 不对齐的掩码和画面差得远
    （关节落点只有 56%），所以默认就是对齐的；`align=False` 只为了做对照。
    """
    clip_dir = Path(clip_dir)
    camera = _camera(clip_dir)
    verts, faces = load_hand_mesh(clip_dir, n_frames)
    fits = frame_alignments(clip_dir, len(verts)) if align else None
    for t in range(len(verts)):
        yield hand_mask(verts[t], faces, camera, hands=hands, dilate=dilate,
                        align=None if fits is None else fits[t])


def load_joints_2d(clip_dir, n_frames: Optional[int] = None) -> np.ndarray:
    """官方 `hand_joints_2d.bin` → (T, 2, 21, 2) 像素坐标。缺文件抛 `HandMeshMissing`。"""
    clip_dir = Path(clip_dir)
    p = clip_dir / "hand_joints_2d.bin"
    if not p.exists():
        raise HandMeshMissing(f"{p} 不存在（同样可以用 fetch_official_extras.py 补）")
    arr = np.fromfile(p, dtype=np.float32)
    per_frame = 2 * N_JOINTS * 2
    if arr.size % per_frame:
        raise ValueError(f"{p}: {arr.size} 个 float 不是 {per_frame} 的整数倍")
    T = arr.size // per_frame
    if n_frames is not None and T != n_frames:
        raise ValueError(f"{p}: {T} 帧 ≠ scene.json 的 {n_frames} 帧")
    return arr.reshape(T, 2, N_JOINTS, 2)


def joints_inside_fraction(clip_dir, n_frames: Optional[int] = None, dilate: int = 0,
                           tol: int = 2, align: bool = True) -> Dict:
    """**掩码对不对的判据**：官方 2D 关节点有多大比例落在掩码里。

    `hand_joints_2d.bin` 和 `hand_verts.bin` 是两个文件、两条产出路径。`tol` 是允许的
    容差（像素），用一次膨胀吸收。

    **看结果要分开看指尖和其余关节**（`TIP_JOINTS`）：指尖骨节点在网格表面上或略微在
    外，落不进掩码是几何本身如此。10 段实测非指尖 0.965、指尖 0.789 —— 判"掩码有没有
    错位"应该看 `non_tip`。

    注意 `align=True` 时这条判据只剩一半独立性：对齐用的就是这份 2D 关节，所以它证的是
    "网格形状 + 这个相似变换能把关节包住"，证不了 3D→2D 那一步。真正独立的复验要等
    RGB 到位（BACKLOG B12）拿眼睛看边缘。`align=False` 留着做对照。

    返回逐手 + 指尖/非指尖分组的比例和点数（NaN 的手不计入）。
    """
    import cv2

    clip_dir = Path(clip_dir)
    camera = _camera(clip_dir)
    verts, faces = load_hand_mesh(clip_dir, n_frames)
    joints = load_joints_2d(clip_dir, len(verts))
    fits = frame_alignments(clip_dir, len(verts)) if align else None
    W, H = int(camera["width"]), int(camera["height"])
    is_tip = np.zeros(N_JOINTS, dtype=bool)
    is_tip[list(TIP_JOINTS)] = True

    hit = np.zeros(2, dtype=np.int64)
    total = np.zeros(2, dtype=np.int64)
    grp_hit = {"tip": 0, "non_tip": 0}
    grp_total = {"tip": 0, "non_tip": 0}
    for t in range(len(verts)):
        for h in (0, 1):
            uv = joints[t, h]
            good = np.isfinite(uv).all(axis=1)
            good &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if not good.any():
                continue
            m = hand_mask(verts[t], faces, camera, hands=("left" if h == 0 else "right"),
                          dilate=dilate, align=None if fits is None else fits[t])
            if not m.any():
                total[h] += int(good.sum())     # 有关节却没掩码 → 全算不中，别偷偷跳过
                grp_total["tip"] += int((good & is_tip).sum())
                grp_total["non_tip"] += int((good & ~is_tip).sum())
                continue
            if tol > 0:
                k = 2 * int(tol) + 1
                m = cv2.dilate(m.astype(np.uint8),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
            uvi = np.round(uv[good]).astype(np.int64)
            inside = m[uvi[:, 1], uvi[:, 0]]
            tips = is_tip[good]
            grp_hit["tip"] += int(inside[tips].sum())
            grp_hit["non_tip"] += int(inside[~tips].sum())
            grp_total["tip"] += int(tips.sum())
            grp_total["non_tip"] += int((~tips).sum())
            hit[h] += int(inside.sum())
            total[h] += int(good.sum())

    out = {"n_frames": int(len(verts)), "tol_px": int(tol), "dilate_px": int(dilate),
           "aligned": bool(align)}
    for h, name in ((0, "left"), (1, "right")):
        out[name] = {"n_joints": int(total[h]),
                     "inside": int(hit[h]),
                     "fraction": (float(hit[h] / total[h]) if total[h] else None)}
    both_t, both_h = int(total.sum()), int(hit.sum())
    out["overall_fraction"] = float(both_h / both_t) if both_t else None
    for g in ("tip", "non_tip"):
        out[g] = {"n_joints": grp_total[g], "inside": grp_hit[g],
                  "fraction": (float(grp_hit[g] / grp_total[g]) if grp_total[g] else None)}
    return out
