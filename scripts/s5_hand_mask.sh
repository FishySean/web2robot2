#!/usr/bin/env bash
# 视觉合成第一块：官方 MANO 网格 → 手部掩码 + 一张能用眼睛看的核对图。
#
# 薄壳，只做两件事：用对的解释器（这台机器 conda activate 不生效，必须绝对路径）、
# 把 src/ 放进 PYTHONPATH。逻辑全在 src/web2robot/synth/，参数直接透传。
#
#   scripts/s5_hand_mask.sh data/clips_official --out outputs/synth
#   scripts/s5_hand_mask.sh data/clips_official --clip=-1r9yl-P-Ao_60.4_68.4
#   scripts/s5_hand_mask.sh data/clips_official --no_align   # 对照：不逐帧对齐差多少
#
# 注意片段目录名以 `-` 开头（`--oo8_XIuOM_799.5_809.8`），直接当参数会被当成选项，
# 所以 --clip 传值时用 `--clip=--oo8_XIuOM_799.5_809.8` 这种写法。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
     "$ROOT/envs/rt_env/bin/python" -m web2robot.synth "$@"
