#!/usr/bin/env bash
# 「synth 加了 render 子命令 ⇒ 手部掩码（mask）那条路一字节没变」验证。
#
# 这一块（任务B 第二块）动了三个文件里的两个共享点：
#   1. src/web2robot/synth/cli.py —— 从"一个平铺的命令行"改成"mask / render 两个子命令"，
#      原来的 main() 逻辑整块搬进 run_mask()
#   2. src/web2robot/robots/m7/env.py —— M7Env.__init__ 多了一个 model= 入口
#      （渲染要往内存里那份模型加相机，磁盘资产不能动）
# 两处都"看起来只是搬家"，但掩码产物是下游合成的输入，不能只靠单元测试通过就断言没坏。
#
# 判据：**逐字节**。掩码这条路不碰 GPU、不碰随机数，本来就该逐位确定
# （对比 check_quality_switch_bytes.sh：那边默认档要跑 KeypointRCNN，GPU 上不确定，
# 所以只能比判决字段；这里没有那个问题，所以要求最严的那种一致）。
#
#   before = HEAD 的 src/（git archive 抽到 /tmp，旧用法：无子命令）
#   after  = 工作区的 src/（新用法：scripts/s5_hand_mask.sh → `mask` 子命令）
#
# 两次都对 data/clips_official 全 10 段跑完整流程（核对图 PNG + 掩码 NPZ + 清单 JSONL）。
# handmask.jsonl 里存了 --out 的路径前缀，所以先把前缀归一化再比 —— 其余一字不许差。
#
#   bash scripts/dev/check_synth_render_bytes.sh > outputs/dev/synth_render_bytecheck.log 2>&1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY=envs/rt_env/bin/python
OUT="outputs/dev/synth_render_bytecheck"
OLDSRC="$OUT/head_src"
BEFORE="$OUT/before"
AFTER="$OUT/after"

rm -rf "$OUT"
mkdir -p "$OLDSRC"
git archive HEAD src | tar -x -C "$OLDSRC"

echo "=== before：HEAD 的 cli.py（旧用法，没有子命令）==="
PYTHONPATH="$OLDSRC/src" $PY -m web2robot.synth data/clips_official --out "$BEFORE"

echo "=== after：工作区的 cli.py（mask 子命令）==="
scripts/s5_hand_mask.sh data/clips_official --out "$AFTER"

echo "=== 比对 ==="
( cd "$BEFORE" && find . -type f ! -name handmask.jsonl | sort | xargs md5sum ) > "$OUT/before.md5"
( cd "$AFTER"  && find . -type f ! -name handmask.jsonl | sort | xargs md5sum ) > "$OUT/after.md5"
diff "$OUT/before.md5" "$OUT/after.md5"
echo "PNG/NPZ 逐字节一致：$(wc -l < "$OUT/before.md5") 个文件"

sed "s#$BEFORE/#OUT/#g" "$BEFORE/handmask.jsonl" > "$OUT/before.jsonl"
sed "s#$AFTER/#OUT/#g"  "$AFTER/handmask.jsonl"  > "$OUT/after.jsonl"
diff "$OUT/before.jsonl" "$OUT/after.jsonl"
echo "handmask.jsonl（归一化 --out 前缀后）逐字节一致：$(md5sum < "$OUT/before.jsonl")"

echo "=== 顺带：M7Env 多出来的 model= 入口不影响原来的构造路径 ==="
PYTHONPATH=src MUJOCO_GL=osmesa $PY - <<'EOF'
import numpy as np
from web2robot.paths import P
from web2robot.robots.m7 import M7Env

a = M7Env()                                  # 默认：scene_vis.xml
b = M7Env(mjcf_path=P.asset("m7_mjcf"))      # 显式：robot-only m7.xml
q = np.array([0.3, 0.1, -0.2, -1.0, 0.2, 0.1, 0.0], dtype=np.float64)
for env in (a, b):
    env.set_arm_joints("left", q)
print("默认构造 qpos 指纹 ", float(a.data.qpos.sum()), a.model.nq, a.model.ncam)
print("m7.xml 构造 qpos 指纹", float(b.data.qpos.sum()), b.model.nq, b.model.ncam)
assert a.model.ncam == b.model.ncam == 0, "磁盘资产里不该有相机（渲染那台是内存里加的）"
try:
    M7Env(mjcf_path=P.asset("m7_mjcf"), model=object())
except ValueError as exc:
    print("两个都给会报错，符合预期：", exc)
else:
    raise AssertionError("mjcf_path 和 model 同时给了却没报错")
EOF
echo "=== 全部通过 ==="
