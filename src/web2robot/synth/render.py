"""按片段那台相机把机器人渲出来 —— 彩色 + 深度 + 掩码，和画面逐像素对齐。

## 为什么不能直接用上游那条渲染链

`external/EgoInfinity/retarget/utils/viz.py::render_robot_sim` 用的是**自由相机**
（azimuth / elevation / distance / lookat 从 `robot_cfg` 读），只返回彩色帧。
也就是说：既不是片段那台相机（渲出来的机器人和画面对不上），也没有深度缓冲
（没法和场景深度比大小、决定谁遮谁）。所以这一层是新写的，不是包一层。

## 三件必须对齐的事

1. **内参**。MuJoCo 的定焦相机只有一个 `fovy`，主点隐含在图像正中。官方 10 段
   `camera.json` 实测 `cx == W/2` 且 `cy == H/2`（**精确相等**，10 段全查过），所以
   针孔模型完全等价：`fovy = 2·atan(H/(2f))`。主点偏心的片段直接抛
   `CameraNotSupported` —— 不做偏心近似，宁可停下来。
2. **外参**。`root_frames.npz` 里的 `(R_per_frame, t_per_frame)` 是**串链根
   （M7 是 `waist_pitch_link`）在相机系里的位姿**。这个口径不是推测：上游
   `scripts/test.py:518` 直接把它 `draw_frame(..., K, ...)` 画在画面上，而
   `cam_to_root_targets()` 用它把相机系的手腕位姿转成根系 IK 目标。M7 既没有
   `workspace_center` 也没有 `bilateral_target_sep`（两个配置都查过），这两条会破坏
   刚性对应的重定标都不生效，所以相机↔机器人是严格刚性的。
3. **MJCF 里没有相机**。官方资产不带 `<camera>`，用 `mujoco.MjSpec` 在内存里加一个，
   **不改磁盘上任何资产**；顺手把 `offwidth/offheight` 顶到画面尺寸（`m7.xml` 没写
   `<global>`，默认 640×480 装不下 853×480）。

## 两个踩到的坑

* **渲机器人本体，不渲 `scene_vis.xml`**。后者带一块无限大地板，相机稍低就整屏被地板
  糊住 —— 实测某一帧 65% 像素是 4 cm 处的地面。要合成的只有机器人。
* **改完 `model.cam_pos` 必须重算 `data.cam_xpos`**。定焦相机挂在 body 上，它的全局
  位姿在 `data.cam_xpos/cam_xmat` 里，由 `mj_forward`（`mj_camlight`）从
  `model.cam_pos/cam_quat` 算出来；而 `mjv_updateScene` 读的是 `data`。所以"写完 model
  就直接渲"会拿**上一次 `mj_forward` 时的相机**去渲：第 0 帧渲成相机在世界原点（整屏
  66% 像素落在 2.6–4.4 cm，就是从骨盆内部往外看），第 1 帧起渲的是**上一帧的相机**。
  静止底座的片段看不出来（前后帧相机一样），neural 求解器逐帧动底座的片段就会整段
  差一帧 —— 这种错法很难从画面上发现，所以这里写死顺序：**摆姿势 → 写相机 →
  `mj_camlight` → 渲**。（早先一版把这个现象误判成"Renderer 第一帧是脏缓冲"，加了个
  warm-up 渲染；那是错的 —— 同一状态连渲 10 次结果完全一致，不是缓冲脏，是 `data`
  没更新。）

## 深度的口径

MuJoCo 开 `enable_depth_rendering()` 后返回的是**沿光轴的米制深度**（实测偏轴点的值
小于欧氏距离，见上面那组数），背景是远平面 `zfar·extent`。这里把背景改成 `inf`，
`RobotFrame.mask` 就是 `isfinite(depth)`，下游不用记魔法阈值。官方 `depth.npz` 是
uint16 毫米，两边都按"沿光轴"理解 —— 边缘处两种口径差 ~2%，比深度本身的误差小得多。

## 怎么知道对齐是对的（`wrist_alignment_report`）

把误差拆成两段，各自归因：

* **robot↔mano**：机器人手腕投影 vs 官方 MANO 手腕 3D 投影。这一段量的是**我方链路**
  （根位姿 + IK + FK + 相机摆放）。实测 8 段 16 只手里 14 只在 **0.6–4.1 px**。
  两只例外是重定向那一步造成的，不是相机：`--oo8_XIuOM_900.3_917.4` 右手
  `ik_rate_r=0.276`、`pos_err=113 mm`；另外 collcmp 那批跑的时候开了
  `--arm_torso_collision --dual_hand_collision`，过滤器会**故意**把穿躯的手臂推出来
  （实测右手中位 3D 偏移 5.2 cm ≈ 30 px）。
* **mano↔官方 2D**：官方数据自己 3D 和 2D 不一致，3–89 px，最差那段就是 BACKLOG B13
  记的 `-20k07PjLTA_48.0_52.4`（89 px）。这一段**不是我们能修的**，也是合成画面对齐
  精度的下限。见 `handmask.frame_alignments` 和 `docs/VISUAL_SYNTH_INPUTS.md` §2 第 5 条。

左右手的槽位取自 `hand_meta.json` 的 `is_right_per_frame`，不是"槽 0 就是左手"——
官方数据里确实有几帧把槽 0 标成右手（见 BACKLOG B14）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .handmask import N_JOINTS, project_points

#: 相机名，加在内存里那份模型上，磁盘资产不动。
CAMERA_NAME = "clip_cam"

#: 主点允许离图像正中多远（像素）。官方 10 段是精确居中的，这个容差只是给将来别的
#: 数据源留一点浮点余量 —— 不是用来放行偏心相机的。
PRINCIPAL_POINT_TOL_PX = 1.0

#: 深度缓冲里"什么都没有"的判据：MuJoCo 把背景填成远平面 `zfar·extent`。
BACKGROUND_DEPTH_FRAC = 0.99

#: OpenCV 相机系（x 右、y 下、z 前）→ MuJoCo 相机系（x 右、y 上、z 后）。
CV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])

#: MANO 的手腕关节下标（和 `hand_joints.bin` / `hand_joints_2d.bin` 同一套 21 点）。
WRIST_JOINT = 0

SIDES = ("left", "right")


class CameraNotSupported(ValueError):
    """片段的成像模型没法用 MuJoCo 的定焦相机表达（目前只有主点偏心这一种）。"""


class RetargetRunMissing(FileNotFoundError):
    """重定向产物不全（缺 `trajectory.npz` 或 `root_frames.npz`）。"""


# ── 输入 ──────────────────────────────────────────────────────────────────────

def clip_camera(clip_dir) -> Dict:
    """读 `camera.json`，顺手把"能不能用定焦相机表达"这件事查掉。"""
    clip_dir = Path(clip_dir)
    with open(clip_dir / "camera.json") as fh:
        camera = json.load(fh)
    for key in ("focal", "cx", "cy", "width", "height"):
        if key not in camera:
            raise CameraNotSupported(f"{clip_dir/'camera.json'} 缺 {key}")
    W, H = float(camera["width"]), float(camera["height"])
    dx, dy = abs(float(camera["cx"]) - W / 2), abs(float(camera["cy"]) - H / 2)
    if max(dx, dy) > PRINCIPAL_POINT_TOL_PX:
        raise CameraNotSupported(
            f"{clip_dir.name}: 主点偏心 ({dx:.2f}, {dy:.2f}) px，超过 "
            f"{PRINCIPAL_POINT_TOL_PX} px。MuJoCo 的定焦相机主点在正中，硬渲会整体错位；"
            f"官方 10 段是精确居中的，遇到偏心的片段要先决定怎么处理，别默默近似")
    if float(camera["focal"]) <= 0:
        raise CameraNotSupported(f"{clip_dir.name}: focal={camera['focal']}")
    return camera


def fovy_degrees(camera: Dict) -> float:
    """针孔焦距 → MuJoCo 的**垂直** fovy（度）。"""
    f, H = float(camera["focal"]), float(camera["height"])
    return float(np.degrees(2.0 * np.arctan(H / (2.0 * f))))


def load_root_frames(run_dir, n_frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """`root_frames.npz` → (R (T,3,3), t (T,3))，串链根在相机系里的逐帧位姿。

    顺手验旋转矩阵是不是真的正交（实测残差 1e-9~1e-7）—— 这份文件是别的脚本产的，
    形状对不代表内容对。
    """
    p = Path(run_dir) / "root_frames.npz"
    if not p.exists():
        raise RetargetRunMissing(f"{p} 不存在 —— 这个重定向产物是 scripts/s4_retarget.sh 出的")
    with np.load(p) as z:
        for key in ("R_per_frame", "t_per_frame"):
            if key not in z.files:
                raise RetargetRunMissing(f"{p} 里没有 {key}（有 {z.files}）")
        R = np.asarray(z["R_per_frame"], dtype=np.float64)
        t = np.asarray(z["t_per_frame"], dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3) or t.shape != (len(R), 3):
        raise RetargetRunMissing(f"{p}: 形状不对（R {R.shape}, t {t.shape}）")
    if n_frames is not None and len(R) != n_frames:
        raise RetargetRunMissing(f"{p}: {len(R)} 帧 ≠ 片段的 {n_frames} 帧 —— 素材对不上")
    err = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
    det = np.linalg.det(R)
    if err > 1e-4 or np.abs(det - 1).max() > 1e-4:
        raise RetargetRunMissing(f"{p}: R_per_frame 不是旋转矩阵（正交残差 {err:.2e}）")
    return R, t


def load_joint_trajectory(run_dir, n_frames: Optional[int] = None) -> Dict:
    """`trajectory.npz` → 手臂/手指关节角 + 关节名 + fps。

    `allow_pickle=True` 是必须的：里面的关节名是 object 数组（上游 `save_trajectory`
    就这么存的）。
    """
    p = Path(run_dir) / "trajectory.npz"
    if not p.exists():
        raise RetargetRunMissing(f"{p} 不存在")
    with np.load(p, allow_pickle=True) as z:
        out = {
            "q_left": np.asarray(z["q_left"], dtype=np.float64),
            "q_right": np.asarray(z["q_right"], dtype=np.float64),
            "fps": float(z["fps"]) if "fps" in z.files else 20.0,
            "clip_id": str(z["clip_id"]) if "clip_id" in z.files else Path(run_dir).name,
            "robot": str(z["robot"]) if "robot" in z.files else "",
        }
        for side in SIDES:
            fq, fn = f"q_{side}_fingers", f"{side}_finger_joint_names"
            out[fq] = np.asarray(z[fq], dtype=np.float64) if fq in z.files else None
            out[fn] = [str(x) for x in z[fn]] if fn in z.files else None
    if len(out["q_left"]) != len(out["q_right"]):
        raise RetargetRunMissing(f"{p}: 左右手臂帧数不等")
    if n_frames is not None and len(out["q_left"]) != n_frames:
        raise RetargetRunMissing(
            f"{p}: {len(out['q_left'])} 帧 ≠ 片段的 {n_frames} 帧 —— 素材对不上")
    return out


def hand_slot_sides(clip_dir, n_frames: int) -> np.ndarray:
    """`hand_meta.json` → (T, 2) int8，每个槽位这一帧是哪只手：0 左、1 右、-1 不在场。

    **槽位不等于左右手**。官方多数片段是"槽 0 左、槽 1 右"，但实测
    `-0RheyDV3a0_474.8_487.3` 有 27 帧把槽 0 标成右手，`-1r9yl-P-Ao_60.4_68.4` 有 5 帧。
    上游 `utils/clip_io.py::_build_trajectories` 就是照这份 meta 分左右的，所以任何
    "按槽位当左右手"的代码都会在那些帧上错手（BACKLOG B14）。
    """
    p = Path(clip_dir) / "hand_meta.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在 —— 左右手归属只能靠它，不能按槽位猜")
    with open(p) as fh:
        meta = json.load(fh)
    rows = meta["is_right_per_frame"]
    if len(rows) != n_frames:
        raise ValueError(f"{p}: is_right_per_frame 是 {len(rows)} 帧，片段说 {n_frames} 帧")
    out = np.full((n_frames, len(rows[0])), -1, dtype=np.int8)
    for t, slots in enumerate(rows):
        for s, val in enumerate(slots):
            if val is True:
                out[t, s] = 1
            elif val is False:
                out[t, s] = 0
    return out


def _by_side(per_slot: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """(T, n_slot, D) 按槽位的数组 → (T, 2, D) 按左右手的数组，缺的填 NaN。

    同一帧两个槽位标成同一只手时**后者赢** —— 和上游 `_build_trajectories` 的循环
    顺序一致，不另立口径。
    """
    T, _, D = per_slot.shape
    out = np.full((T, 2, D), np.nan, dtype=np.float64)
    for t in range(T):
        for s in range(sides.shape[1]):
            side = int(sides[t, s])
            if side >= 0:
                out[t, side] = per_slot[t, s]
    return out


def official_wrist_uv(clip_dir, n_frames: int) -> np.ndarray:
    """官方 `hand_joints_2d.bin` 的手腕点 → (T, 2, 2)，第 1 维是左右手。"""
    clip_dir = Path(clip_dir)
    a = np.fromfile(clip_dir / "hand_joints_2d.bin", dtype=np.float32)
    per_frame = 2 * N_JOINTS * 2
    if a.size % per_frame or a.size // per_frame != n_frames:
        raise ValueError(f"{clip_dir/'hand_joints_2d.bin'}: {a.size} 个 float 对不上 {n_frames} 帧")
    a = a.reshape(n_frames, 2, N_JOINTS, 2)[:, :, WRIST_JOINT, :]
    return _by_side(a, hand_slot_sides(clip_dir, n_frames))


def official_wrist_xyz(clip_dir, n_frames: int) -> np.ndarray:
    """官方 `hand_joints.bin` 的手腕点（相机系米制）→ (T, 2, 3)。"""
    clip_dir = Path(clip_dir)
    a = np.fromfile(clip_dir / "hand_joints.bin", dtype=np.float32)
    per_frame = 2 * N_JOINTS * 3
    if a.size % per_frame or a.size // per_frame != n_frames:
        raise ValueError(f"{clip_dir/'hand_joints.bin'}: {a.size} 个 float 对不上 {n_frames} 帧")
    a = a.reshape(n_frames, 2, N_JOINTS, 3)[:, :, WRIST_JOINT, :]
    return _by_side(a, hand_slot_sides(clip_dir, n_frames))


# ── 机器人 ────────────────────────────────────────────────────────────────────

def _robot_assets(robot: str) -> Tuple[str, Dict]:
    """机器人名 → (MJCF 资产键, CONFIG)。只认已经验过的那台。"""
    if robot == "m7":
        from web2robot.robots.m7 import CONFIG
        return "m7_mjcf", CONFIG
    raise ValueError(
        f"渲染目前只验过 m7，给的是 {robot!r}。要加第二台：给它的 env 类补上 "
        f"`model=` 入口（见 robots/m7/env.py），再在这里注册 MJCF 键")


class RobotPoser:
    """把关节角摆进 MuJoCo，并把 body 位姿换算到片段相机系。**不含渲染。**

    渲染要的那份模型是内存里加过相机的（`ClipRobotRenderer` 传 `model`）；只做几何
    核对时不必加相机，直接从磁盘那份 MJCF 建。
    """

    def __init__(self, robot: str = "m7", model=None):
        import mujoco

        from web2robot.paths import P
        asset_key, config = _robot_assets(robot)
        self.robot = robot
        self.config = config
        env_cls = config["env_cls"]
        self.env = (env_cls(model=model) if model is not None
                    else env_cls(mjcf_path=P.asset(asset_key)))
        self.model, self.data = self.env.model, self.env.data
        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, config["torso_body"])
        self._wrist_ids = {
            side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for side, name in config["wrist_body"].items()}
        for name, bid in [(config["torso_body"], self._torso_id),
                          *((n, self._wrist_ids[s]) for s, n in config["wrist_body"].items())]:
            if bid < 0:
                raise ValueError(f"MJCF 里没有 body {name!r}")

    def pose(self, q_left, q_right, q_left_fingers=None, q_right_fingers=None,
             left_finger_names: Optional[Sequence[str]] = None,
             right_finger_names: Optional[Sequence[str]] = None) -> None:
        """摆一帧。手指给了就摆手指（名字用 `trajectory.npz` 里存的那份）。"""
        self.env.set_arm_joints("left", np.asarray(q_left, dtype=np.float64))
        self.env.set_arm_joints("right", np.asarray(q_right, dtype=np.float64))
        for q, names in ((q_left_fingers, left_finger_names),
                         (q_right_fingers, right_finger_names)):
            if q is not None and names is not None:
                self.env.set_finger_joints(np.asarray(q, dtype=np.float64), list(names))

    def root_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """串链根 body 在 MuJoCo 世界系里的位姿 (R, t)。"""
        return (self.data.xmat[self._torso_id].reshape(3, 3).copy(),
                self.data.xpos[self._torso_id].copy())

    def body_in_camera(self, body_id: int, R_root_cam, t_root_cam) -> np.ndarray:
        """MuJoCo 世界系里的 body 原点 → 片段相机系。"""
        R_w_r, t_w_r = self.root_pose()
        x_root = R_w_r.T @ (self.data.xpos[body_id] - t_w_r)
        return np.asarray(R_root_cam) @ x_root + np.asarray(t_root_cam)

    def wrists_in_camera(self, R_root_cam, t_root_cam) -> Dict[str, np.ndarray]:
        return {side: self.body_in_camera(bid, R_root_cam, t_root_cam)
                for side, bid in self._wrist_ids.items()}

    def camera_pose_in_world(self, R_root_cam, t_root_cam) -> Tuple[np.ndarray, np.ndarray]:
        """(R_root_cam, t_root_cam) → 相机在 MuJoCo 世界系里的 (旋转, 位置)。

        推导：`x_cam = R_cr·x_root + t_cr`，而 `x_root = R_wrᵀ(x_world - t_wr)`，
        于是 `R_cw = R_cr·R_wrᵀ`、`t_cw = t_cr - R_cw·t_wr`，相机位置就是 `-R_cwᵀ·t_cw`。
        返回的旋转是**相机→世界**（列是相机轴在世界系里的方向）。
        """
        R_w_r, t_w_r = self.root_pose()
        R_c_w = np.asarray(R_root_cam) @ R_w_r.T
        t_c_w = np.asarray(t_root_cam) - R_c_w @ t_w_r
        R_w_c = R_c_w.T
        return R_w_c, -R_w_c @ t_c_w


@dataclass(frozen=True)
class RobotFrame:
    """一帧渲染结果。`depth` 里背景是 `inf`，所以 `mask` 就是"哪儿有机器人"。"""

    rgb: np.ndarray                  # (H, W, 3) uint8
    depth: np.ndarray                # (H, W) float32，米，沿光轴；背景 inf
    wrist_uv: Dict[str, np.ndarray]  # 'left'/'right' → (2,) 像素，z<=0 时 NaN
    wrist_xyz: Dict[str, np.ndarray]  # 'left'/'right' → (3,) 相机系米制

    @property
    def mask(self) -> np.ndarray:
        return np.isfinite(self.depth)

    @property
    def area(self) -> float:
        return float(self.mask.mean())


class ClipRobotRenderer:
    """按一段片段的相机渲机器人。一段片段建一个，逐帧调 `render()`。"""

    def __init__(self, camera: Dict, robot: str = "m7"):
        import mujoco

        from web2robot.paths import P
        self.camera = camera
        self.width, self.height = int(camera["width"]), int(camera["height"])
        self.fovy = fovy_degrees(camera)
        asset_key, _ = _robot_assets(robot)

        # 内存里加一台相机 + 把离屏缓冲顶到画面尺寸。磁盘资产一个字节都不动。
        spec = mujoco.MjSpec.from_file(str(P.asset(asset_key)))
        spec.worldbody.add_camera(name=CAMERA_NAME, fovy=self.fovy)
        model = spec.compile()
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)

        self.poser = RobotPoser(robot=robot, model=model)
        self.model = self.poser.model
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        self._far = float(self.model.stat.extent) * float(self.model.vis.map.zfar)
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

    # ── 生命周期 ─────────────────────────────────────────────────────────────
    def close(self) -> None:
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "ClipRobotRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── 渲染 ────────────────────────────────────────────────────────────────
    def _place_camera(self, R_root_cam, t_root_cam) -> None:
        """写相机外参，**并立刻把 `data.cam_xpos` 重算出来**（见模块 docstring 第二坑）。"""
        import mujoco
        R_w_c, p_w = self.poser.camera_pose_in_world(R_root_cam, t_root_cam)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R_w_c @ CV_TO_MUJOCO).ravel())
        self.model.cam_pos[self._cam_id] = p_w
        self.model.cam_quat[self._cam_id] = quat
        # 只重算相机/光源的全局位姿（body 运动学由 poser.pose() 里的 mj_forward 算过了）。
        mujoco.mj_camlight(self.model, self.poser.data)

    def _shoot(self, depth: bool) -> np.ndarray:
        if depth:
            self._renderer.enable_depth_rendering()
        else:
            self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self.poser.data, camera=CAMERA_NAME)
        return self._renderer.render().copy()

    def render(self, R_root_cam, t_root_cam, q_left, q_right,
               q_left_fingers=None, q_right_fingers=None,
               left_finger_names=None, right_finger_names=None) -> RobotFrame:
        """摆一帧、摆相机、出彩色 + 深度。**顺序不能换**，见模块 docstring 第二坑。"""
        self.poser.pose(q_left, q_right, q_left_fingers, q_right_fingers,
                        left_finger_names, right_finger_names)
        self._place_camera(R_root_cam, t_root_cam)
        rgb = self._shoot(depth=False)
        raw = self._shoot(depth=True)
        depth = np.where(raw >= BACKGROUND_DEPTH_FRAC * self._far, np.inf, raw).astype(np.float32)

        xyz = self.poser.wrists_in_camera(R_root_cam, t_root_cam)
        uv = {side: project_points(x, self.camera) for side, x in xyz.items()}
        return RobotFrame(rgb=rgb, depth=depth, wrist_uv=uv, wrist_xyz=xyz)


# ── 判据 ──────────────────────────────────────────────────────────────────────

def _median(a: np.ndarray) -> Optional[float]:
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def wrist_alignment_report(clip_dir, run_dir, n_frames: int, robot: str = "m7") -> Dict:
    """机器人手腕落点对不对：拆成"我方链路"和"官方数据自相矛盾"两段，各自报中位。

    不渲染（只 FK + 投影），所以很快 —— 判据和出片是两件事，出片挂了也该能量。
    每只手报：`robot_mano_px`（我方链路）、`mano_2d_px`（官方 3D/2D 不一致）、
    `robot_2d_px`（两者叠加，也就是画面上实际看到的错位）、`robot_mano_m`（3D 偏移）。
    """
    clip_dir, run_dir = Path(clip_dir), Path(run_dir)
    camera = clip_camera(clip_dir)
    R_pf, t_pf = load_root_frames(run_dir, n_frames)
    traj = load_joint_trajectory(run_dir, n_frames)
    uv_2d = official_wrist_uv(clip_dir, n_frames)
    xyz_mano = official_wrist_xyz(clip_dir, n_frames)

    poser = RobotPoser(robot=robot)
    rows = {side: {"rm_px": [], "m2_px": [], "r2_px": [], "rm_m": []} for side in SIDES}
    for t in range(n_frames):
        poser.pose(traj["q_left"][t], traj["q_right"][t],
                   None if traj["q_left_fingers"] is None else traj["q_left_fingers"][t],
                   None if traj["q_right_fingers"] is None else traj["q_right_fingers"][t],
                   traj["left_finger_joint_names"], traj["right_finger_joint_names"])
        xyz_robot = poser.wrists_in_camera(R_pf[t], t_pf[t])
        for si, side in enumerate(SIDES):
            u_r = project_points(xyz_robot[side], camera)
            u_m = project_points(xyz_mano[t, si], camera)
            u_2 = uv_2d[t, si]
            rows[side]["rm_px"].append(np.linalg.norm(u_r - u_m))
            rows[side]["m2_px"].append(np.linalg.norm(u_m - u_2))
            rows[side]["r2_px"].append(np.linalg.norm(u_r - u_2))
            rows[side]["rm_m"].append(np.linalg.norm(xyz_robot[side] - xyz_mano[t, si]))

    out: Dict[str, Dict] = {}
    for side in SIDES:
        a = {k: np.asarray(v, dtype=np.float64) for k, v in rows[side].items()}
        out[side] = {
            "n_frames": int(np.isfinite(a["r2_px"]).sum()),
            "robot_mano_px": _median(a["rm_px"]),
            "mano_2d_px": _median(a["m2_px"]),
            "robot_2d_px": _median(a["r2_px"]),
            "robot_mano_m": _median(a["rm_m"]),
        }
    return out
