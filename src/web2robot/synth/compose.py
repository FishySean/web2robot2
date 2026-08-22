"""任务B 第三块：把渲好的机器人**按深度合成回原画面**（抠人 → 补背景 → 按谁挡谁贴）。

前两块给的是零件：`handmask.py` 给"画面里的人手在哪"，`render.py` 给"和画面同一台相机
拍出来的机器人（彩色 + 米制深度 + 掩码）"。这一块把它们拼成一帧最终画面，三步：

    1. **抠人**：人的掩码内改成背景板的像素（原画面里的人从此不见）
    2. **补背景**：背景板怎么来 —— 相机不动用整段时间中值，相机动用 `cv2.inpaint`
    3. **按深度贴机器人**：机器人比场景近的地方才画它，远的地方让场景挡住

**为什么机器人是现渲，不读 `robot_render.npz`**：那份 npz 只有深度和掩码，**没有彩色**
（存下来 190 MB 量级，而且会和轨迹脱钩 —— 换了一次重定向就成了过期数据）。所以这里
直接用 `render.ClipRobotRenderer` 现渲，和 `render` 子命令**同一条代码路径**，彩色/深度/
掩码天然自洽。npz 留着给别的消费方（比如只要深度的下游）。

**深度排序那条规则里有一处不显然的地方，写清楚**：

    机器人可见 = 机器人掩码 ∧ ( 在被抠掉的人形区域内  ∨  场景深度无效  ∨
                                机器人深度 ≤ 场景深度 + 容差 )

中间那两项不是放水：

* **被抠掉的区域**里，`depth.npz` 存的是**人手自己的深度**（人还在画面里时测的）。人已经
  被擦掉了，那块地方真正的背景深度我们不知道 —— 拿人手的深度去挡机器人，等于让一个不
  存在的东西遮住机器人。机器人本来就是来接替那只手的，所以这块地方无条件画。
* **场景深度无效（0）**的像素官方本来就没测出来（反光、太远、重建失败）。深度未知就不能
  拿它判遮挡，否则整片黑洞会把机器人啃掉。这两条都在 `tests/test_synth_compose.py`
  里各有一个用例钉住。

**两路深度共用一个尺度**这件事是前提，不是假设：官方 `depth.npz` 是 uint16 毫米、
独立于 `root_frames.npz` 的平移，实测机器人最近点和场景 1% 分位在 6/8 段差 ≤ 0.17 m
（见 `docs/VERIFICATION.md` ⑦）。差一个量级的话，合成出来的机器人会整体跑到物体前面
或后面 —— 那种错误在单帧里看着"只是有点怪"，很容易被当成渲染问题。

**现在还缺什么（诚实说明）**：真 RGB 还卡在 BACKLOG B12。所以

* 默认的"人形掩码"其实只有**手部**（`hand_masks.npz` 两只手的并集 + 膨胀）——
  整个人的分割要跑 Mask R-CNN，而它的输入就是 RGB，没 RGB 连跑都跑不起来。
  接口留了 `--person_mask_dir`：谁产的人形掩码都能塞进来（SAM3 / 公司内部服务都行）。
* `--rgb depth` 是一个**替身底图**（深度灰度转三通道），用来在没有真 RGB 的时候把整条链
  跑通、出一段能用眼睛看的片子。它**不是**真画面，产物里 `rgb_source` 会写明白，
  别拿它当验收依据。
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .render import RobotFrame

#: 人形掩码默认膨胀半径（像素）—— 边缘要盖过去一点，不然抠完剩一圈人皮。
DEFAULT_DILATE_PX = 3
#: 深度排序的容差（米）：两路深度各有噪声，贴着相等时倾向于画机器人。
DEPTH_TOL_M = 0.02
#: `cv2.inpaint` 的半径（像素）。
INPAINT_RADIUS = 3
#: "相机算不算不动"的判据阈值（相邻帧灰度平均绝对差，0–255）。
#: **这是个惯例值，没在我们的素材上标定过**（记在 BACKLOG C25）—— 所以
#: `plate_mode`/`motion_score` 都会写进产物清单，将来能回头扫。
#: 第②步路由本来就在判"相机动不动"，有那个标签时应当用 `mode=` 直接指定，别靠这里猜。
STATIC_MOTION_THRESH = 3.0


class RgbMissing(FileNotFoundError):
    """没有真 RGB 可用（BACKLOG B12）—— 明确报错，不悄悄拿深度替身顶上。"""


# ── 输入 ────────────────────────────────────────────────────────────────────

def _decode_video(path: Path, n_frames: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr[:, :, ::-1].copy())        # BGR → RGB
    cap.release()
    if len(frames) != n_frames:
        raise ValueError(f"{path}: 解出 {len(frames)} 帧 ≠ 片段的 {n_frames} 帧")
    return np.stack(frames)


def _decode_dir(path: Path, n_frames: int) -> np.ndarray:
    import cv2
    files = sorted(p for p in path.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(files) != n_frames:
        raise ValueError(f"{path}: {len(files)} 张图 ≠ 片段的 {n_frames} 帧")
    return np.stack([cv2.imread(str(p))[:, :, ::-1] for p in files])


def depth_gray_rgb(clip_dir: Path, n_frames: int) -> np.ndarray:
    """深度灰度当**替身底图**（逐帧各自拉伸，和核对图同一口径）。不是真画面。"""
    from .cli import depth_background
    gray = depth_background(clip_dir, n_frames)
    if gray is None:
        raise RgbMissing(f"{clip_dir} 既没有真 RGB，也没有 depth.npz 可当替身")
    return np.repeat(gray[..., None], 3, axis=3)


def load_rgb(spec: str, clip_dir: Path, n_frames: int,
             camera: Dict) -> Tuple[np.ndarray, str]:
    """→ ((T,H,W,3) uint8 RGB, 这份画面是哪来的)。

    `spec` 四种写法：`auto`（找 `outputs/fetch/<片段>/rgb.mp4`，没有就
    `RgbMissing`）、`depth`（深度替身，明确标注）、一个 mp4 路径、一个图片目录。
    尺寸必须和 `camera.json` 对得上 —— 差一个像素后面所有掩码都错位。
    """
    H, W = int(camera["height"]), int(camera["width"])
    if spec == "depth":
        rgb, src = depth_gray_rgb(clip_dir, n_frames), "depth.npz（替身，不是真画面）"
    elif spec == "auto":
        cand = Path("outputs/fetch") / clip_dir.name / "rgb.mp4"
        if not cand.exists():
            raise RgbMissing(
                f"没找到 {cand} —— 官方片段不带 RGB，下载还卡在 BACKLOG B12。"
                f"想先把链路跑通就用 --rgb depth（深度替身），别拿它当验收依据")
        rgb, src = _decode_video(cand, n_frames), str(cand)
    else:
        p = Path(spec)
        if not p.exists():
            raise RgbMissing(f"--rgb {spec} 不存在")
        rgb = _decode_dir(p, n_frames) if p.is_dir() else _decode_video(p, n_frames)
        src = str(p)
    if rgb.shape[1:3] != (H, W):
        raise ValueError(f"{src}: 画面 {rgb.shape[2]}×{rgb.shape[1]} ≠ "
                         f"camera.json 的 {W}×{H}")
    return np.ascontiguousarray(rgb, dtype=np.uint8), src


def load_person_mask(clip_dir: Path, n_frames: int, camera: Dict,
                     hand_masks: Optional[Path] = None,
                     person_dir: Optional[Path] = None,
                     dilate_px: int = DEFAULT_DILATE_PX) -> Tuple[np.ndarray, str]:
    """→ ((T,H,W) bool 要抠掉的区域, 掩码是哪来的)。

    `person_dir` 给了就读那里的逐帧掩码（任何分割器产的都行，>0 即前景）；否则退回
    `hand_masks.npz` 的**左右手并集**。并集这一点很重要：B14 说官方有 34 帧把右手存在
    槽 0，**按名字取单只手会取错，取并集不会**。
    """
    import cv2
    H, W = int(camera["height"]), int(camera["width"])
    if person_dir is not None:
        files = sorted(p for p in Path(person_dir).iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if len(files) != n_frames:
            raise ValueError(f"{person_dir}: {len(files)} 张掩码 ≠ {n_frames} 帧")
        mask = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in files])
        src = f"{person_dir}（外部人形掩码）"
    else:
        p = Path(hand_masks) if hand_masks is not None else \
            Path("outputs/synth") / clip_dir.name / "hand_masks.npz"
        if not p.exists():
            raise FileNotFoundError(f"没有 {p} —— 先跑 scripts/s5_hand_mask.sh")
        with np.load(p) as z:
            n, h, w = (int(v) for v in z["shape"])
            if (n, h, w) != (n_frames, H, W):
                raise ValueError(f"{p}: 掩码 {n}×{h}×{w} ≠ 片段 {n_frames}×{H}×{W}")
            both = np.zeros((n, h, w), dtype=bool)
            for side in ("left", "right"):
                bits = np.unpackbits(z[side], axis=1)[:, :h * w]
                both |= bits.reshape(n, h, w).astype(bool)
        mask, src = both, f"{p}（只有手，不是整个人 —— 见模块说明）"
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
        mask = np.stack([cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in mask])
    return mask, src


def scene_depth_m(clip_dir: Path, n_frames: int) -> Optional[np.ndarray]:
    """官方 `depth.npz`（uint16 毫米，0 = 没测出来）→ (T,H,W) float32 米，无效处 `nan`。"""
    p = clip_dir / "depth.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        arr = z[z.files[0]]
    if len(arr) != n_frames:
        raise ValueError(f"{p}: {len(arr)} 帧 ≠ 片段的 {n_frames} 帧")
    out = arr.astype(np.float32) / 1000.0
    out[arr == 0] = np.nan
    return out


# ── 背景板 ──────────────────────────────────────────────────────────────────

def motion_score(rgb: np.ndarray, mask: np.ndarray) -> float:
    """相邻帧灰度平均绝对差（只统计两帧都没被掩掉的像素）。0–255，越小越"相机不动"。"""
    gray = rgb.astype(np.float32).mean(axis=3)
    diffs = []
    for t in range(1, len(gray)):
        good = ~(mask[t] | mask[t - 1])
        if good.any():
            diffs.append(float(np.abs(gray[t][good] - gray[t - 1][good]).mean()))
    return float(np.median(diffs)) if diffs else 0.0


def median_plate(rgb: np.ndarray, mask: np.ndarray,
                 block: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """相机不动时的背景板：逐像素在**没被掩掉的那些帧**上取中值。

    → (板 (H,W,3) uint8, 哪些像素真的有观测 (H,W) bool)。
    分块算是为了内存：整段一次转 float32 是 GB 量级（257×480×853×3）。
    中值而不是均值 —— 人手扫过某个像素的帧数只要少于一半，中值就完全不受它影响，
    均值会留一道淡影。
    """
    T, H, W, _ = rgb.shape
    plate = np.zeros((H, W, 3), dtype=np.uint8)
    seen = np.zeros((H, W), dtype=bool)
    for y0 in range(0, H, block):
        y1 = min(H, y0 + block)
        m = mask[:, y0:y1]                                   # (T,h,W)
        seen[y0:y1] = (~m).any(axis=0)
        blk = rgb[:, y0:y1].astype(np.float32)
        blk[np.broadcast_to(m[..., None], blk.shape)] = np.nan
        with warnings.catch_warnings():
            # 整段都被掩掉的像素 → 全 nan 切片 → nan（下面 fill_holes 去补），
            # 这是预期路径，不是异常，所以把 numpy 的 All-NaN 警告压掉。
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(blk, axis=0)
        plate[y0:y1] = np.nan_to_num(med, nan=0.0).round().astype(np.uint8)
    return plate, seen


def fill_holes(plate: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """从没露过脸的像素（整段都被手挡着）用 `cv2.inpaint` 补 —— 中值那步给不出值。"""
    import cv2
    holes = (~seen).astype(np.uint8)
    if not holes.any():
        return plate
    bgr = cv2.inpaint(plate[:, :, ::-1].copy(), holes, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def inpaint_plate(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """相机在动时的背景板：逐帧 `cv2.inpaint`。→ (T,H,W,3) uint8。

    相机一动，"整段同一块背景"这个前提就没了，时间中值会把不同位置的东西糊在一起。
    """
    import cv2
    out = np.empty_like(rgb)
    for t in range(len(rgb)):
        m = mask[t].astype(np.uint8)
        if not m.any():
            out[t] = rgb[t]
            continue
        bgr = cv2.inpaint(rgb[t][:, :, ::-1].copy(), m, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        out[t] = bgr[:, :, ::-1]
    return out


def background_plate(rgb: np.ndarray, mask: np.ndarray, mode: str = "auto",
                     static_thresh: float = STATIC_MOTION_THRESH) -> Tuple[np.ndarray, Dict]:
    """→ ((1 或 T, H, W, 3) uint8 背景板, 说明)。

    `mode`：`median`（相机不动）/ `inpaint`（相机在动）/ `auto`（按 `motion_score` 猜，
    **能拿到第②步路由的相机标签时请直接指定，别靠猜**）。
    返回第一维是 1 表示"整段共用一块板"，调用方按 `plate[min(t, len(plate)-1)]` 取。
    """
    info: Dict = {"mode_requested": mode}
    if mode == "auto":
        score = motion_score(rgb, mask)
        mode = "median" if score < static_thresh else "inpaint"
        info["motion_score"] = round(score, 4)
        info["static_thresh"] = static_thresh
    if mode == "median":
        plate, seen = median_plate(rgb, mask)
        info["hole_fraction"] = float((~seen).mean())
        plate = fill_holes(plate, seen)[None]
    elif mode == "inpaint":
        plate = inpaint_plate(rgb, mask)
    else:
        raise ValueError(f"背景板模式只有 median / inpaint / auto，收到 {mode!r}")
    info["mode"] = mode
    return plate, info


# ── 合成 ────────────────────────────────────────────────────────────────────

def robot_visible(robot: RobotFrame, scene_m: Optional[np.ndarray],
                  erased: np.ndarray, tol: float = DEPTH_TOL_M) -> np.ndarray:
    """一帧的"机器人该露出来的地方"(H,W) bool —— 规则和取舍见模块 docstring。"""
    m = robot.mask
    if scene_m is None:
        return m
    nearer = np.zeros_like(m)
    good = np.isfinite(scene_m)
    nearer[good] = robot.depth[good] <= scene_m[good] + tol
    return m & (erased | ~good | nearer)


def override_fraction(robot: RobotFrame, scene_m: Optional[np.ndarray],
                      erased: np.ndarray, tol: float = DEPTH_TOL_M) -> float:
    """机器人像素里，**只因为"落在被抠掉的人形区域内"这条例外才画出来**的比例。

    量这个是因为那条例外有代价：人形掩码（还带膨胀）一旦溢到**真的比机器人近**的东西上
    （手按在桌上时，膨胀那几像素就落到桌沿），机器人就会盖住那个东西。这个数就是
    "例外一共影响了多大面积" —— 小就说明代价可忽略，大就得把例外收紧成"只在人手自己的
    深度附近生效"。先量，别先改。
    """
    if scene_m is None:
        return 0.0
    good = np.isfinite(scene_m)
    farther = np.zeros_like(robot.mask)
    farther[good] = robot.depth[good] > scene_m[good] + tol
    n = int(robot.mask.sum())
    return float((robot.mask & erased & good & farther).sum() / n) if n else 0.0


def compose_frame(rgb: np.ndarray, plate: np.ndarray, erased: np.ndarray,
                  robot: RobotFrame, scene_m: Optional[np.ndarray],
                  tol: float = DEPTH_TOL_M) -> Tuple[np.ndarray, np.ndarray]:
    """一帧合成 → ((H,W,3) uint8 RGB, 机器人实际露出来的地方)。

    顺序是**先抠人再贴机器人**：反过来的话，人形掩码会把刚贴上去的机器人手擦掉一块
    （两者本来就大面积重叠 —— 机器人的手就该出现在人的手那个位置）。
    """
    out = rgb.copy()
    out[erased] = plate[erased]
    vis = robot_visible(robot, scene_m, erased, tol)
    out[vis] = robot.rgb[vis]
    return out, vis
