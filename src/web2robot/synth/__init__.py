"""视觉合成（任务B）—— 把画面里的人换成机器人。

现在只有第一块：**手部掩码**。它是这条链路上唯一一件**不依赖真 RGB** 就能做完的东西 ——
官方给了 MANO 手部网格（`hand_verts.bin` + `hand_faces.bin`），配上 `camera.json` 内参
光栅化就出掩码。

但**不是拿投影就能用**：实测官方 3D 手投出来和官方 `hand_joints_2d.bin` 差 9.3 px
（中位），而且逐帧解出来的缩放在 0.32–0.95 之间漂 —— 官方 3D 手的深度/尺度是逐帧
不定的，像素空间里的证据是那份 2D 关节。所以先逐帧逐手拟合一个相似变换（残差降到
3.7 px）再光栅化，这一步是 `frame_alignments()`，默认开。

验对错：`joints_inside_fraction()` 用 2D 关节点核对落点。要注意开了对齐之后这条判据
只能证"形状 + 变换包得住关节"，证不了 3D→2D —— 完整复验要等真 RGB（BACKLOG B12）。

素材清单和还缺什么见 `docs/VISUAL_SYNTH_INPUTS.md`。
"""
from .handmask import (HandMeshMissing, alignment_report, frame_alignments, hand_mask,
                       hand_mask_series, joints_inside_fraction, load_hand_mesh,
                       load_joints_2d, project_points, solve_similarity)

__all__ = ["HandMeshMissing", "alignment_report", "frame_alignments", "hand_mask",
           "hand_mask_series", "joints_inside_fraction", "load_hand_mesh",
           "load_joints_2d", "project_points", "solve_similarity"]
