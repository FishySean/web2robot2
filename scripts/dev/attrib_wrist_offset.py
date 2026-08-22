#!/usr/bin/env python3
"""归因：机器人手腕在画面上的错位，是 IK 没解出来，还是碰撞过滤把手推走了？

`outputs/synth/render/render.jsonl` 里 8 段 16 只手的 `robot_mano_px` 中位数
大多在 2 px 以内，但有 4 只手在 7–32 px。两种可能的成因，判据不同：

  * **IK 没解出来** —— `metrics.npz` 自己就报了：`ik_rate_*` < 1 且 `pos_err_*`
    有大值。这种不用重跑就能定。
  * **碰撞过滤把手推走了** —— `ik_rate == 1`、`pos_err ≤ 1 cm`，也就是 IK 明明
    解到了，但 `--arm_torso_collision` / `--dual_hand_collision` 这些**解完之后**
    的后处理为了不穿躯/不撞手把关节角改了。这种从产物里看不出来，
    **只能拿"同 seed、同 solver、不开碰撞过滤"重跑一份来对**。

第二种就是这个脚本干的事。grid 求解器不含随机先验（随机的是 neural），所以
同 seed 重跑出来的 `root_frames.npz` 应当逐字节一致 —— 脚本会先核对这一点，
核对不过就说明还有别的变量在动，归因不成立，直接报错而不是给个好看的数。

    envs/rt_env/bin/python scripts/dev/attrib_wrist_offset.py \
        data/clips_official/-1r9yl-P-Ao_86.3_90.8 \
        outputs/retarget/collcmp/-1r9yl-P-Ao_86.3_90.8_grid \
        outputs/dev/nocoll/-1r9yl-P-Ao_86.3_90.8_grid

产物是 stdout 的一张表，不写文件 —— 数字要留档就抄进 docs/VERIFICATION.md。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from web2robot.synth.render import (  # noqa: E402
    SIDES, RobotPoser, load_joint_trajectory, load_root_frames, wrist_alignment_report,
)


def main(clip: str, coll_run: str, nocoll_run: str) -> int:
    clip_dir, a_dir, b_dir = Path(clip), Path(coll_run), Path(nocoll_run)
    n = int(json.loads((clip_dir / "scene.json").read_text())["stats"]["n_frames"])

    # 前置：两份产物的根位姿必须一致，否则"只有碰撞过滤这一个变量"不成立
    Ra, ta = load_root_frames(a_dir, n)
    Rb, tb = load_root_frames(b_dir, n)
    if not (np.allclose(Ra, Rb, atol=1e-12) and np.allclose(ta, tb, atol=1e-12)):
        raise SystemExit("根位姿两份不一样，归因不成立（grid + 同 seed 本该一致）")
    print(f"{clip_dir.name}  {n} 帧  根位姿两份一致 ✓")

    rep = {"开碰撞过滤": wrist_alignment_report(clip_dir, a_dir, n),
           "关碰撞过滤": wrist_alignment_report(clip_dir, b_dir, n)}
    print(f"\n{'':12s} " + "  ".join(f"{s:>22s}" for s in SIDES))
    for tag, r in rep.items():
        cells = [f"{r[s]['robot_mano_px']:8.2f} px {r[s]['robot_mano_m']*100:6.2f} cm"
                 for s in SIDES]
        print(f"{tag:12s} " + "  ".join(f"{c:>22s}" for c in cells))

    # 关节角本身差多少（不经过相机，纯 FK）——过滤器真的动了手才有上面的差
    qa, qb = load_joint_trajectory(a_dir, n), load_joint_trajectory(b_dir, n)
    poser = RobotPoser()
    d = {s: [] for s in SIDES}
    for t in range(n):
        p = {}
        for tag, q in (("a", qa), ("b", qb)):
            poser.pose(q["q_left"][t], q["q_right"][t],
                       None if q["q_left_fingers"] is None else q["q_left_fingers"][t],
                       None if q["q_right_fingers"] is None else q["q_right_fingers"][t],
                       q["left_finger_joint_names"], q["right_finger_joint_names"])
            p[tag] = poser.wrists_in_camera(Ra[t], ta[t])
        for s in SIDES:
            d[s].append(float(np.linalg.norm(p["a"][s] - p["b"][s])))
    print("\n两份轨迹的手腕 3D 位移（开 − 关）:")
    for s in SIDES:
        v = np.asarray(d[s])
        print(f"  {s:5s} 中位 {np.median(v)*100:6.2f} cm  最大 {v.max()*100:6.2f} cm  "
              f"改动帧数 {(v > 5e-4).sum()}/{n}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
