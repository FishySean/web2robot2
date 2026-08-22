#!/usr/bin/env bash
# 视觉合成第二块：按片段那台相机把机器人渲出来（彩色 + 深度 + 掩码 + 对齐核对图）。
#
# 薄壳，只做三件事：用对的解释器（这台机器 conda activate 不生效，必须绝对路径）、
# 把 src/ 放进 PYTHONPATH、把 MUJOCO_GL 设成 osmesa（无显示器离屏渲染）。
# 逻辑全在 src/web2robot/synth/render.py，参数直接透传。
#
#   scripts/s6_robot_render.sh data/clips_official \
#       --runs_dir outputs/retarget/collcmp --pattern '*_grid' --out outputs/synth/render
#   scripts/s6_robot_render.sh data/clips_official \
#       --run -1r9yl-P-Ao_86.3_90.8=outputs/retarget/collcmp/-1r9yl-P-Ao_86.3_90.8_grid
#   scripts/s6_robot_render.sh data/clips_official --runs_dir ... --npz   # 存深度/掩码给合成用
#
# 一段片段同时有 _grid 和 _neural 两份产物时会报错让你挑（--pattern 或 --run），
# 不会默认替你选一份 —— 两个根位姿求解器出来的画面不一样。
#
# 注意片段目录名以 `-` 开头（`--oo8_XIuOM_799.5_809.8`），直接当参数会被当成选项，
# 所以 --clip / --run 传值时用 `--clip=--oo8_XIuOM_799.5_809.8` 这种写法。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
     MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
     "$ROOT/envs/rt_env/bin/python" -m web2robot.synth render "$@"
