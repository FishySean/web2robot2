#!/usr/bin/env bash
# 流水线第 0 步：把官方片段的原始 RGB 画面还原出来。
#
# 薄壳，只做两件事：用对的解释器（这台机器 conda activate 不生效，必须绝对路径）、
# 把 src/ 放进 PYTHONPATH。逻辑全在 src/web2robot/fetch/，参数直接透传。
#
#   scripts/s0_fetch_rgb.sh data/clips_official --out outputs/fetch --dry_run
#   scripts/s0_fetch_rgb.sh data/clips_official --out outputs/fetch
#   scripts/s0_fetch_rgb.sh data/clips_official --backend local --source_dir /存视频的目录
#
# 注意片段目录名以 `-` 开头（`--oo8_XIuOM_799.5_809.8`），直接当参数会被当成选项，
# 所以 --clip 传值时用 `--clip=--oo8_XIuOM_799.5_809.8` 这种写法。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
     "$ROOT/envs/rt_env/bin/python" -m web2robot.fetch "$@"
