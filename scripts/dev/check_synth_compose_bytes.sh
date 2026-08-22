#!/usr/bin/env bash
# 「synth 加了 compose 子命令 ⇒ mask / render 两条路一字节没变」验证。
#
# 这一块（任务B 第三块）动的共享点只有一个文件：src/web2robot/synth/cli.py
#   1. 顶上多 `from . import compose`（新模块，谁都还没引用过 —— 但导入本身会跑模块级代码）
#   2. `run_render()` 里那段"--run / --runs_dir 怎么解析"抽成了 `resolve_runs()`，
#      因为 compose 要用同一份口径。抽函数看着无害，**但它就在 render 的入口路径上**
#   3. 模块 docstring 从"两个子命令"改成"三个"
# 掩码和渲染产物都是下游合成的输入，不能只靠单元测试通过就断言没坏。
#
# 判据：**逐字节**。这两条路都不碰随机数；渲染走 osmesa 软件光栅，同机同版本可复现
# （对比 check_quality_switch_bytes.sh：那边默认档跑 KeypointRCNN，GPU 上不确定，
# 所以只能比判决字段；这里没有那个问题，所以要求最严的那种一致）。
#
#   before = HEAD 的 src/（git archive 抽到 /tmp）
#   after  = 工作区的 src/
#
# 两条路都跑全量：mask 全 10 段（PNG + NPZ + JSONL），render 全 8 段（PNG + MP4 + NPZ + JSONL，
# 8 段是"有重定向产物"的那些）。清单 JSONL 里存了 --out 路径前缀，先归一化再比 —— 其余一字不许差。
#
#   bash scripts/dev/check_synth_compose_bytes.sh > outputs/dev/synth_compose_bytecheck.log 2>&1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY=envs/rt_env/bin/python
OUT="outputs/dev/synth_compose_bytecheck"
OLDSRC="$OUT/head_src"
RUNS="outputs/retarget/collcmp"

rm -rf "$OUT"
mkdir -p "$OLDSRC"
git archive HEAD src | tar -x -C "$OLDSRC"

run_old() {   # 用 HEAD 的 src 跑（子命令和参数照现在的写法，HEAD 已经有 mask/render）
  # WEB2ROBOT_ROOT 必须显式指回真仓库：`paths.P` 是按**模块所在位置**往上找 configs/ 的，
  # 而 git archive 抽出来的树只有 src/ —— 不指的话 render 一上来就 FileNotFoundError。
  # 这不削弱隔离性：变量只影响 configs/ 和 assets/ 的定位，那些本次一个字没动，
  # 唯一的自变量还是 src/。
  PYTHONPATH="$OLDSRC/src" WEB2ROBOT_ROOT="$ROOT" MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
      $PY -m web2robot.synth "$@"
}
run_new() {   # 用工作区的 src 跑
  PYTHONPATH="$ROOT/src" WEB2ROBOT_ROOT="$ROOT" MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
      $PY -m web2robot.synth "$@"
}

cmp_tree() {  # $1 before 目录  $2 after 目录  $3 清单文件名
  local b="$1" a="$2" jsonl="$3"
  ( cd "$b" && find . -type f ! -name "$jsonl" | sort | xargs md5sum ) > "$OUT/$jsonl.before.md5"
  ( cd "$a" && find . -type f ! -name "$jsonl" | sort | xargs md5sum ) > "$OUT/$jsonl.after.md5"
  diff "$OUT/$jsonl.before.md5" "$OUT/$jsonl.after.md5"
  echo "  产物逐字节一致：$(wc -l < "$OUT/$jsonl.before.md5") 个文件"
  sed "s#$b/#OUT/#g" "$b/$jsonl" > "$OUT/$jsonl.before"
  sed "s#$a/#OUT/#g" "$a/$jsonl" > "$OUT/$jsonl.after"
  diff "$OUT/$jsonl.before" "$OUT/$jsonl.after"
  echo "  $jsonl（归一化 --out 前缀后）逐字节一致：$(md5sum < "$OUT/$jsonl.before")"
}

echo "=== mask：before（HEAD）==="
run_old mask data/clips_official --out "$OUT/mask_before"
echo "=== mask：after（工作区）==="
run_new mask data/clips_official --out "$OUT/mask_after"
echo "=== mask 比对 ==="
cmp_tree "$OUT/mask_before" "$OUT/mask_after" handmask.jsonl

echo "=== render：before（HEAD）==="
run_old render data/clips_official --runs_dir "$RUNS" --pattern '*_grid' --npz \
        --out "$OUT/render_before"
echo "=== render：after（工作区，run_render 里那段解析抽成了 resolve_runs）==="
run_new render data/clips_official --runs_dir "$RUNS" --pattern '*_grid' --npz \
        --out "$OUT/render_after"
echo "=== render 比对 ==="
cmp_tree "$OUT/render_before" "$OUT/render_after" render.jsonl

echo "=== 顺带：resolve_runs 在两个子命令下解出同一份归属 ==="
PYTHONPATH="$ROOT/src" $PY - <<'EOF'
from web2robot.synth.cli import build_parser, resolve_runs
ap = build_parser()
a = resolve_runs(ap.parse_args(["render", "data/clips_official",
                                "--runs_dir", "outputs/retarget/collcmp",
                                "--pattern", "*_grid"]))
b = resolve_runs(ap.parse_args(["compose", "data/clips_official",
                                "--runs_dir", "outputs/retarget/collcmp",
                                "--pattern", "*_grid"]))
assert a == b, (sorted(a), sorted(b))
print(f"两个子命令解出同样的 {len(a)} 段归属")
EOF
echo "=== 全部通过 ==="
