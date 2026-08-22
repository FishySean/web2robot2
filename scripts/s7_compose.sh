#!/usr/bin/env bash
# 视觉合成第三块：抠掉画面里的人 → 补背景 → 按深度把机器人贴回去，出一帧最终画面。
#
# 薄壳，只做三件事：用对的解释器（这台机器 conda activate 不生效，必须绝对路径）、
# 把 src/ 放进 PYTHONPATH、把 MUJOCO_GL 设成 osmesa（机器人要现渲，无显示器离屏）。
# 逻辑全在 src/web2robot/synth/compose.py，参数直接透传。
#
#   # 真 RGB 还卡在 BACKLOG B12，所以先用深度替身底图把整条链跑通
#   scripts/s7_compose.sh data/clips_official \
#       --runs_dir outputs/retarget/collcmp --pattern '*_grid' --rgb depth
#   # 有真画面之后（--rgb auto 会去找 outputs/fetch/<片段>/rgb.mp4）
#   scripts/s7_compose.sh data/clips_official --runs_dir ... --rgb auto
#   # 对照：不做深度排序，机器人整个盖在最上层
#   scripts/s7_compose.sh data/clips_official --runs_dir ... --rgb depth --no_depth_order
#
# 前置：先跑 scripts/s5_hand_mask.sh 出 hand_masks.npz（人形掩码的默认来源）。
# 有别的分割器产的人形掩码就 --mask_dir 指过去 —— 默认那份**只有手，不是整个人**。
#
# 注意片段目录名以 `-` 开头（`--oo8_XIuOM_799.5_809.8`），直接当参数会被当成选项，
# 所以 --clip / --run 传值时用 `--clip=--oo8_XIuOM_799.5_809.8` 这种写法。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
     MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
     "$ROOT/envs/rt_env/bin/python" -m web2robot.synth compose "$@"
