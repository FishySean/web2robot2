"""
Single M7 humanoid MuJoCo environment.

M7 = dual 7-DoF arms + two 5-finger dexterous hands on a 3-DoF waist torso.

Action:  dict with optional keys "left" and "right", each (7,) joint angles [rad]
         for [shoulder_pitch, shoulder_roll, arm_yaw, elbow_pitch, elbow_yaw,
              wrist_pitch, wrist_roll].

Observation: dict with optional keys "left" and "right", each containing:
    - "pos":  (3,) wrist (hand_frame) position in world frame
    - "quat": (4,) hand_frame orientation quaternion (w, x, y, z) in world frame

NOTE (2026-07-23, Step A): the M7 MJCF was converted from URDF and has **no
<actuator> block** (only joints).  set_arm_joints / set_finger_joints therefore
guard `aid >= 0` and fall back to writing qpos directly.  This is sufficient for
the kinematic render / IK verification path (set qpos → mj_forward → read xpos).
Position actuators will be added before Step B training.

迁移说明（2026-08-07）：这个类**不再继承上游的 ``sim.base_env.BaseEnv``**。

原因不是嫌它，而是模块边界：M7 的机器人定义要能被别的重定向框架直接拿去用，
继承上游 ABC 就把它焊死在 EgoInfinity 的目录结构上了（``tests/test_module_boundaries.py``
钉住这条）。这样做是安全的，因为 ``BaseEnv`` 是纯抽象类（4 个 abstractmethod、
零实现），且**全仓库没有一处 ``isinstance(env, BaseEnv)``**（查过），运行期完全靠
鸭子类型。

代价是丢了"忘实现某个方法 → 实例化时 TypeError"这个保护。补偿是
``tests/test_m7_robot.py::TestBaseEnvConformance`` —— 它 import 上游 BaseEnv
（测试允许，``src/`` 不允许），逐个断言我们实现了每个 abstractmethod。
比继承更早报错：测试时就红，而不是跑到一半 AttributeError。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from web2robot.paths import P

# ── robot description ─────────────────────────────────────────────────────────
# 迁移前是 Path(__file__).parents[3] / "robots" / "m7"（相对上游目录结构推）。
# 现在资产在 web2robot/assets/robots/m7/，唯一来源是 configs/paths.yaml。
_MJCF_PATH     = P.asset("m7_mjcf")
_MJCF_MJX_PATH = P.asset("m7_mjx")      # arms-only MJX model (scripts/dev/generate_m7_mjx.py)
_SCENE_PATH    = P.asset("m7_scene")
_ROBOT_DIR     = _MJCF_PATH.parent

_ARM_JOINTS = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_arm_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_arm_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    ],
}

_EE_BODY    = {"left": "left_hand_frame", "right": "right_hand_frame"}
# waist_pitch_link is the real tree ancestor shared by both arms (torso_frame is a
# coincident leaf, unusable as an IK-chain root).  Use it as the torso reference.
_TORSO_BODY = "waist_pitch_link"

# Maps retargeter short names → M7 MJCF finger-joint names (WITHOUT side prefix).
# The env re-attaches the "left_"/"right_" side prefix per call.
_FINGER_JOINT_NAMES = {
    "thumb_bend":  "hand_thumb_bend_joint",
    "thumb_rota1": "hand_thumb_rota_joint1",
    "thumb_rota2": "hand_thumb_rota_joint2",
    "index_abd":   "hand_index_bend_joint",
    "index_mcp":   "hand_index_joint1",
    "index_pip":   "hand_index_joint2",
    "middle_mcp":  "hand_mid_joint1",
    "middle_pip":  "hand_mid_joint2",
    "ring_mcp":    "hand_ring_joint1",
    "ring_pip":    "hand_ring_joint2",
    "pinky_mcp":   "hand_pinky_joint1",
    "pinky_pip":   "hand_pinky_joint2",
}


class M7Env:
    """Single-instance dual-arm M7 simulation environment.

    实现的是上游 ``sim.base_env.BaseEnv`` 的接口（reset / set_arm_joints /
    step_joints / get_wrist_pose），但不继承它 —— 见模块头部说明。
    """

    def __init__(
        self,
        mjcf_path: Optional[Path] = None,
        start_config: Optional[dict] = None,
        model: Optional["mujoco.MjModel"] = None,
    ):
        # ``model=`` 是给"模型需要在内存里改过再用"的调用方留的入口：
        # ``synth/render.py`` 要往模型里加一台按片段内参配好的 <camera>、并把离屏缓冲
        # 顶到画面尺寸，这两件事只能在 compile 期做（MjSpec），**不能改磁盘上的资产**。
        # 给了 model 就直接用它，mjcf_path 被忽略。
        if model is not None:
            if mjcf_path is not None:
                raise ValueError("mjcf_path 和 model 只能给一个")
            self.model = model
        else:
            mjcf_path = Path(mjcf_path) if mjcf_path is not None else _SCENE_PATH
            self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data  = mujoco.MjData(self.model)

        self._start_config = start_config

        self._joint_ids:    dict[str, list[int]] = {}
        self._actuator_ids: dict[str, list[int]] = {}
        self._body_ids:     dict[str, int]       = {}

        for side, joints in _ARM_JOINTS.items():
            self._joint_ids[side] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in joints
            ]
            self._actuator_ids[side] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
                for j in joints
            ]
            self._body_ids[side] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, _EE_BODY[side]
            )

        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, _TORSO_BODY
        )

        self._joint_limits: dict[str, np.ndarray] = {}
        for side, jids in self._joint_ids.items():
            self._joint_limits[side] = np.array(
                [self.model.jnt_range[jid] for jid in jids]
            )

        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)
        if self._start_config is not None:
            for side, q in self._start_config.items():
                self.set_arm_joints(side, np.asarray(q, dtype=np.float64))
        return self._get_obs()

    # ── action ────────────────────────────────────────────────────────────────

    def set_arm_joints(self, side: str, q: np.ndarray):
        assert q.shape == (7,), f"Expected (7,) joint angles, got {q.shape}"
        for i, aid in enumerate(self._actuator_ids[side]):
            if aid >= 0:                        # no actuators in URDF-converted MJCF
                self.data.ctrl[aid] = q[i]
        for i, jid in enumerate(self._joint_ids[side]):
            self.data.qpos[self.model.jnt_qposadr[jid]] = q[i]
        mujoco.mj_forward(self.model, self.data)

    def set_finger_joints(self, q: np.ndarray, joint_names: list[str]):
        """Set M7 finger joints from retargeter output (left_/right_ prefixed short names)."""
        for i, name in enumerate(joint_names):
            if name.startswith("left_"):
                side, short = "left", name[len("left_"):]
            elif name.startswith("right_"):
                side, short = "right", name[len("right_"):]
            else:
                continue
            if short.endswith("_joint"):
                short = short[:-len("_joint")]
            suffix = _FINGER_JOINT_NAMES.get(short)
            if suffix is None:
                continue
            mjcf_name = f"{side}_{suffix}"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
            if jid < 0:
                continue
            adr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[adr] = float(np.clip(q[i], lo, hi))
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mjcf_name)
            if aid >= 0:
                self.data.ctrl[aid] = float(np.clip(q[i], lo, hi))
        mujoco.mj_forward(self.model, self.data)

    def step_joints(self, action: dict) -> dict:
        for side, q in action.items():
            self.set_arm_joints(side, np.asarray(q, dtype=np.float64))
        return self._get_obs()

    # ── observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        return {
            side: {"pos": self.data.xpos[bid].copy(), "quat": self.data.xquat[bid].copy()}
            for side, bid in self._body_ids.items()
        }

    def get_wrist_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self._body_ids[side]
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
