# 改完之后怎么验收

这个工程的验收判据**一个模块一套**，因为各模块的确定性不同：碰撞过滤是纯 CPU
无随机源，可以要求逐位相同；质检里跑着 GPU 神经网络，逐位相同是不可能的目标。
拿错判据的后果是双向的 —— 要么放过真错误，要么把浮点噪声当成 bug 追半天。

约定见 [`CONVENTIONS.md`](CONVENTIONS.md)，坑见 [`PITFALLS.md`](PITFALLS.md)。

**所有模块共通的最后一步：出片，用眼睛看。** 指标 ≠ 画面。

```bash
envs/rt_env/bin/python -m unittest discover -s tests -v     # 秒级，全套 427 个用例
```

---

## 判据一览

| 改了什么 | 判据 | 为什么是这个判据 |
|---|---|---|
| ⓿取原始画面 | **逐帧核对内容**（不只核对帧数）+ 三条对齐判据出数字 | 帧数对而整段错位一帧，是这一档唯一致命的失效方式 |
| ①质检 / ②路由 | **判决字段逐字一致** + 每个信号没越阈 | GPU 上的 KeypointRCNN 不是逐位确定的 |
| 质检/路由整档跳过 | builtin 档同上，**skip 档要求逐字节相同** | skip 不加载任何模型，那一档是纯确定的 |
| ③感知前端 | 单测（注入假 callable）+ 冻结值比对 | 前端要 GPU，但算术部分可以纯 numpy 测 |
| ④重定向 | 隔离对比逐位一致 + 固定 seed 的端到端 | 根锚点有随机源，不固定 seed 无法比 |
| ⑤碰撞 / 轨迹 | **逐位相同** | 纯 CPU 有限差分，没有随机源 |
| ⑥手部掩码 | 官方 2D 关节落在掩码里的比例，**指尖/非指尖分开报** + 核对图用眼睛看 | 指尖骨节点本来就在网格外，混着算会把"几何本来如此"当成"掩码错位" |
| M7 机器人定义 | 两个验收脚本输出逐字节一致 | 资产是静态的，任何变化都该是有意的 |
| 新加一台机器人 | 生成脚本七步自检 + **表 vs 自己的 MJCF** + 老机器人逐字节不变 | 两台同构机器人之间不许互当真源，见下 |
| `evidence/` 里的数 | 断言**具体数值**，不是大于小于 | 防的是论文数字和证据脱钩 |

---

## ⓿取原始画面（`src/web2robot/fetch/`，2026-08-23）

这一档的失效方式很特殊：**帧数永远是对的**（目标时间轴长度就是 `n_frames`），错的是
"第 i 帧到底是源视频的哪一帧"。错开一帧，后面视觉合成贴上去的机器人手就和画面里的手
不在同一时刻，而任何"数一数帧数"的检查都发现不了。所以判据是**内容级**的。

```bash
envs/rt_env/bin/python -m unittest tests.test_fetch_rgb -v      # 27 个用例，约 6 秒
```

三件事被钉住：

1. **取的是对的那一帧。** 造一支每帧亮度 = 帧号 × 4 的源视频，截完逐帧核对亮度，
   同时核对 `frames_index.json` 里记的 `source_frame` 是否等于 `t_target × fps_src`
   （±0.6 帧）。两个都对才算取对 —— 只核对其一，索引和内容一起偏的情况会漏过去。
2. **时间轴只认 `video_source`。** 拿真实的 10 段官方片段断言
   `-2cNMO9Mm3Q_192.4_209.2` 的起点是 **195.790**（目录名差 −3.39 s ≈ 102 源帧），
   哪天有人"顺手"改成读目录名，这条会红。
3. **截断的素材必须被判死。** 造一个 `+movflags faststart` 之后掐掉后 2/3 字节的文件
   —— 这和 YouTube 不给 PO Token 时返回的残件**是同一种形状**：`ffprobe` 报得出完整
   时长，解码 0 帧。测试先断言"它确实还能 probe"，再断言 `verify_playable` 报"截断"。
   **验收必须真解码**，只信 ffprobe 会被骗。

对齐判据自己也要被验：同内容 → `lag == 0` / `verdict == aligned`；把 `depth.mp4`
掐掉前 5 帧 → `verdict != aligned`。判据分不清对齐和错位，拿它验收就是自欺。

跑真实数据时的验收线（`align_report.json` / `frames_index.json`）：

| 字段 | 该是什么 |
|---|---|
| `sampling.within_half_frame` | `true`（`max|dt| ≤ 0.5/fps_src`，30 fps 源 = 16.7 ms） |
| `counts` 里所有 `*_decoded` | 全部等于 `scene.json` 的 `stats.n_frames` |
| `motion_lag["depth.mp4"].best_lag` | `0` |
| `verdict` | `aligned`（缺判据只会写 `unknown`，**不会写 pass**） |
| `align_montage.png` | **用眼睛看**：手部关节点要落在画面里手的位置上 |

**这一档不适用逐字节基线**：`rgb.mp4` 的字节取决于源视频文件本身（不同 client / 不同
清晰度下载到的不是同一份），基线会绑死在一次下载上。确定性的部分（时间轴、取帧索引）
已经由上面的合成素材测试钉住，那才是我们自己的逻辑。

「没破坏现有行为」这条另外给凭据：这一档**全是新文件**，所以直接用 md5 证明 —— 把
`git ls-files` 里所有 `.py` / `.sh` / `.yaml` / `.xml`（139 个）逐个与 `HEAD` 比，
不同的 0 个（改动只有 4 份 `.md` 加新增文件）。这比"单测全绿所以没破坏"实在。

```bash
while IFS= read -r f; do case "$f" in *.py|*.sh|*.yaml|*.xml)
  a=$(git show "HEAD:$f" | md5sum | cut -d' ' -f1); b=$(md5sum "$f" | cut -d' ' -f1)
  [ "$a" = "$b" ] || echo "CHANGED: $f";; esac; done < <(git ls-files)
```

---

## ①质检 / ②路由（`src/web2robot/quality/`、`routing/`）

判决必须与基准逐字一致：

```bash
PYTHONPATH=src envs/rt_env/bin/python -m web2robot.quality \
    data/videos/ tests/regression/*.mp4 --out /tmp/re/qc.jsonl --viz /tmp/re/ev
envs/rt_env/bin/python scripts/dev/diff_quality_run.py /tmp/re/qc.jsonl
```

**不要求数值逐位相同** —— KeypointRCNN 在 GPU 上不是逐位确定的，实测重跑一次
`cup_cpvH8gzUTko` 的 `torso_rate` 就从 0.4828 变 0.4655（n=58，差值正好 1/58，
一帧翻转）。判的是两件更贴近实质的事：**判决字段逐字一致**，以及**每个参与判决的
信号都没有越过它的阈值**（后者能抓到"判决碰巧没变但信号已经贴着阈值了"）。

再看一眼 contact sheet（`--viz`）确认画面。

### 整档跳过的开关（`--quality_gate` / `--routing`，2026-08-21）

公司内部已有质检评估体系，这两步从"必须"降级为"可选"。两个开关**各自独立**
（`builtin|skip`，默认 `builtin`），因为将来可能只换掉其中一个。

```bash
# 1. 单测：21 个用例，0.14s
envs/rt_env/bin/python -m unittest tests.test_quality_switch -v

# 2. 端到端五遍对比（base / 显式默认值 / --routing skip / skip ×2，约 80 秒）
bash scripts/dev/check_quality_switch_bytes.sh > outputs/dev/quality_switch_bytecheck.log 2>&1
```

**判据为什么在这里是分裂的**：builtin 档要跑 KeypointRCNN，逐字节比 md5 本身就不成立
（上一节那个 1/58 一帧翻转；`qc.md` 里还写了 wall time）。所以 builtin 档判
`diff_quality_run.py`，**skip 档反过来要求逐字节相同** —— 它一个模型都不加载，
是纯确定的，那一档如果不逐位相同就说明有隐藏状态。

2026-08-21 实测四条（10 段：`data/videos/` 7 段 + 3 个回归对照）：

1. **`base`（一个新参数都不传）对 2026-08-05 回归基准**：判决字段 11 项 × 10 段
   **全部一致**，12 条参与判决的信号**没有一条换边**，20 处数值漂移（最大 6.32%，
   是 `hand_lapvar_med`）。
2. **`bi`（显式写 `--quality_gate builtin --routing builtin`）对 `base`**：数值漂移
   **0 处**，md5 都相同（同为 `e05741199a48c1172425650fe02b42ff`）—— 加开关没改默认档。
3. **`--routing skip` 对 `base`**：全 10 段**只有 `suggested_route` / `route_rationale`
   两项变**，`verdict` / `reasons` / `signals` / `stages_run` 一字不变，路线全成 `None`，
   每段的 `route_rationale` 里都写明是"关了路由"而不是"判不出来"。
4. **`--quality_gate skip`**：两遍 jsonl 逐字节相同
   （`f4228bc484f0b195bd29d0cd5d988ed5`），判决集合 `['skipped']`、跑过的 stage `[]`、
   日志里 `hand detector on` 出现 **0** 次（base 那遍 1 次），耗时 **0.0s**（base 39.8s）。

**两处设计选择，都是被实测逼出来的，不是想出来的：**

- **skip 写 `skipped` 而不是 `accept`。** `accept` 是在断言一次没发生过的测量；
  谁拿 `qc.jsonl` 算通过率，都会把没量过的片段算进分母。
- **`routing=skip` 不写 reason 码。** 第一版给 `reasons` 前面插了个
  `routing_skipped`，跑出来是 `['routing_skipped', 'no_person']` —— 9/10 段的真实
  理由被挤到第二位，直接违反 `reasons` 那句"没通过的检查、最决定性的在前"的契约。
  改成信息只落在 `suggested_route` / `route_rationale`，并加了两个单测钉住
  （`test_routing_skip_is_not_a_reason_code` 连源码里出现 `add_reason("routing_skipped")`
  都算不过）。这次是判据抓住了代码，不是代码逼弯了判据。

暂时**没有** `external` 这一档：现在没有对接对象，不知道公司系统的判决是什么格式，
先占个名字会有人去实现它。`test_no_external_mode_yet` 将来会因此变红 —— 那是提醒不是
故障，记在 [`BACKLOG.md`](BACKLOG.md) C22。

## ③感知前端（`src/web2robot/perception/`）

分两层，因为变更理由不同：`to_clip.py` 是**下游的输入契约**，跟用哪个前端无关；
`hawor.py` / `wilor.py` + `moge.py` 一个前端一个。前端的函数（`run_mano` /
`load_slam_cam` / WiLoR 的 `predict` / MoGe 的 `infer`）是**参数注入**进来的，
所以单测不需要 GPU、不需要 checkpoint、不需要第三方仓库。

```bash
envs/rt_env/bin/python -m unittest tests.test_perception_modules -v   # HaWoR，20 个
envs/rt_env/bin/python -m unittest tests.test_wilor_modules -v        # WiLoR+MoGe，40 个
```

改了算术部分，还要和冻结基线比：

```bash
envs/rt_env/bin/python scripts/dev/check_wilor_vs_baseline.py
#   期望：与 2026-07-14 冻结的 wilor_wrist 最大差 < 6e-8（float32 的量化步）
```

三个"错了不报错"的地方由测试钉住：einsum 下标顺序、`hand_joints.bin` 与
`joints_shape` 一致、两条取样路径故意不同的取整方式。细节见
[`PITFALLS.md`](PITFALLS.md) 第 6~9 条。

想看两条深度策略的差别：

```bash
envs/perception_env/bin/python scripts/dev/viz_wilor_depth_modes.py \
  --clips outputs/clips/<pointmap> outputs/clips/<globalscale> \
  --rgb <图片目录> --out outputs/viz/wilor_depth_modes.mp4
```

## ④重定向（`src/web2robot/retarget/`）

三条线，另外这里多一个"参照物"的讲究：`tests/test_retarget_modules.py` 里
`_old_*` 那几个函数是**迁移前 `test.py` 内联版的逐字复制，故意没整理**。整理它就等于
把参照物改成了被测物，比对就不作数了 —— 文件头的注释写着"勿整理"，请当真。

```bash
# 1. 隔离对比（合成输入，秒级，不需要 external/）：40 个用例
envs/rt_env/bin/python -m unittest tests.test_retarget_modules -v

# 2. 隔离对比（真实片段，需要 external/）：12 个数组 + 叠字逐位比
envs/rt_env/bin/python scripts/dev/check_fallback_vs_baseline.py
#   期望 11 个片段全部"逐位一致 ✓"，ours_webapple 那段整段单手→两边都拒掉

# 3. 端到端 seed-0，再出四宫格看画面
envs/rt_env/bin/python scripts/dev/render_compare_grid.py --runs ... \
    --out outputs/dev/compare_grid_retarget/
```

合成输入那份有一个用例专门断言"三档补洞 + 判坏"四种状态**都被打到了**
（`test_the_synthetic_input_actually_exercises_all_three_branches`）——
参照物再准，输入没打到分支也证明不了什么。

### ④里的第二条根位姿路线（`retarget/root_grid.py`，静态网格搜索）

这条**没有参照物**可比 —— 它不是迁移，是新方法（Qwen-RobotManip 公式 3），
上游没有对应实现。所以判据换成三层：

```bash
# 1. 隔离单测（纯 numpy 假 IK，秒级，不要 GPU/checkpoint）：44 个用例
envs/rt_env/bin/python -m unittest tests.test_root_grid -v

# 2. 端到端切换（同一条流水线，只换 --root_solver）
scripts/s4_retarget.sh examples/fill_jar --robot m7 --seed 0 \
    --ckpt runs/m7/taskspace_v2/checkpoints/final.pt --root_solver grid \
    --out outputs/retarget/rootcmp/fill_jar_grid

# 3. 两条路线在 11 个片段上的对比表（约 15 分钟/片段，要后台跑）
scripts/dev/m7_tool.sh compare_root_pose_solvers.py \
    --spacing 0.05 --rotation both --yaws 12 --device cuda:3 --seed 0
```

单测里必须钉住的三条（都各有一个用例，删了就等于把这个模块的安全网拆了）：

1. **剪枝可采纳** —— 剪枝版和 `exhaustive=True` 版**逐位相同**。这是模块里唯一
   一处"为了快而改变搜索顺序"的地方，破掉之后跑出来的数就不再是 argmax，而单看
   结果发现不了（照样返回一个位姿、照样有个可行率）。
2. **空操作会炸** —— 候选平移不起作用时抛 `RuntimeError`。这条是照
   `postik-smoother-noop`（`uniform_filter1d(size=1)` 恒等）那次教训加的：这里
   对应的坑是给 `cam_to_root_targets` 传了 `workspace_center`，位置被重新居中，
   整个搜索退化成空操作。判据是"分数不变**而可达上界在变**"，所以可行率饱和
   （常态）不会误报。
3. **平局是常态，不是异常** —— 公式 (3) 的 argmax 是个**集合**（K 只覆盖位置极值，
   完全不看腕部朝向），实测 `-QALmP1nHtM_678.2_682.2` 上同分候选 292 个。所以
   `n_tied` 要报出来，且换 `tie_break` 不许改变目标函数值。**这不是学术洁癖**：
   同一个片段上，任意取一个同分成员给出全轨迹 66.7%，取同分集合的最内部点给出
   100.0%——差别全在没进 K 的帧上。

第 3 条也是这条路线上"指标 ≠ 画面"的具体形态：`keyframe_ik_rate` 100% 完全可能
配一个不能用的解，所以对比表里 `kf / 全部帧 / 同分` 三个数要一起看，最后还是要出片。

## ⑥视觉合成 · 手部掩码（`src/web2robot/synth/`，2026-08-23）

这一档没有"逐位相同"可比（全是新代码），也**暂时没有真 RGB 可看**（BACKLOG B12）。
所以判据分两层：合成素材上的确定性测试，加真实素材上的量化落点。

```bash
envs/rt_env/bin/python -m unittest tests.test_synth_handmask -v   # 35 个用例，约 5 秒
scripts/s5_hand_mask.sh data/clips_official --out outputs/synth   # 10 段，约 2 分钟
```

**合成素材测的是"对齐这一步真的在起作用"**：造一段片段，让 2D 关节 = 3D 投影结果做一个
已知的 (s, tx, ty) 变换，`frame_alignments()` 必须把那三个数解回来（残差 < 0.01 px），
且对齐后的掩码把 2D 关节全包住（1.000）、不对齐的一个都包不住（0.000）。这一条如果
只在真实数据上看比例，"从 0.765 涨到 0.964"说不清是对齐对了还是碰巧。

**真实素材的验收线**（`outputs/synth/handmask.jsonl`，10 段官方片段）：

| 字段 | 该是什么 | 实测 |
|---|---|---|
| `joints_inside.non_tip.fraction` | > 0.9（判"掩码有没有错位"看这个） | 合并 **0.964**，最差一段 0.847 |
| `joints_inside.tip.fraction` | 明显低于 non_tip（几何本来如此，不是缺陷） | 合并 0.789 |
| 同上，`--no_align` 对照 | 必须明显更差，否则说明对齐是多余的 | non_tip 0.765 / tip 0.457 |
| `alignment.residual_px.median` | < 5 px | 1.08–7.04（最差那段见 BACKLOG B13） |
| `alignment.scale.min/max` | **不该恒等于 1** —— 恒 1 说明这一步是多余的 | 0.088–1.806 |
| `masks.empty_frames` | `0`（有手的帧不该没掩码） | 全部 0 |
| `handmask_check.png` | **用眼睛看**：掩码贴在手上、绿色关节点落在染色区里 | 10 段逐段看过 |

**为什么核对图的底是深度**：`depth.npz` 是这条链路上目前唯一一份真实成像，手在深度图里
轮廓清楚，掩码贴不贴边一眼能看出来。RGB 到位之后把底图换掉，判据不用改。

**这条判据只有一半独立性，要写明白**：对齐用的就是 `hand_joints_2d.bin`，所以落点比例
证的是"网格形状 + 那个变换能包住关节"，证不了 3D→2D 那一步。完整的复验要等真 RGB。

「没破坏现有行为」同样用 md5 证：`git ls-files` 里 149 个 `.py`/`.sh`/`.yaml`/`.xml`
逐个与 `HEAD` 比，不同的 **0 个**（改动只有 3 份 `.md` 加新增文件）。命令见上面 ⓿ 那节。

---

## ⑤碰撞检测 / 轨迹清洗（`src/web2robot/collision/`、`trajectory/`）

碰撞过滤是有限差分梯度下降、纯 CPU、没有随机源，**要求逐位相同**（比①那条严格）：

```bash
# 1. 隔离对比：把过滤器从整条链里拽出来，喂同一份输入跑旧/新两份实现
envs/rt_env/bin/python scripts/dev/diff_collision_migration.py

# 2. 端到端：同 seed 跑两遍，trajectory.npz 每个 key 逐位相同、robot_sim.mp4 md5 相同
# 3. 出四宫格（源视频 / 不开碰撞 / 新代码 / 旧代码）
scripts/dev/render_quad.sh ...
```

端到端**不能单独作为判据**：上游锚点的随机性会把差异掩盖或伪造成几十度的关节角差 ——
第一次跑就被这个骗过，以为迁移改坏了 108°。

### 独立复核：官方 MuJoCo mesh contacts

上面那三步验的是"我方实现有没有被改坏"，验不了"我方代理判得对不对"。后者用第二个
**独立判据** —— `m7.xml` 里本来就开着的 98 个 mesh 碰撞 geom（只报告，不改轨迹）：

```bash
# 注意要绝对路径：m7_tool.sh 会 cd 到上游 retarget/ 目录
scripts/dev/m7_tool.sh audit_mujoco_contacts.py "$PWD/outputs/retarget/fill_jar_e2e_retarget"
```

它打印三段：两个 MJCF 的碰撞 geom 数（`m7.xml` 98 / `m7_mjx.xml` 0）、上游 geom 集合圈到了
什么（跨臂 contact 应为 0，印证它在 M7 上是瞎的）、以及逐帧分歧。fill_jar 的基线
（2026-08-11，216 帧，已开碰撞纠正）：

| | MuJoCo | 我方代理 | 只有 MuJoCo 判 |
|---|---|---|---|
| 左臂 | 50 帧 / 最深 **8.07 cm** | 96 帧 / 6.20 cm | 3 帧（169/172/173，≤1.86 cm） |
| 右臂 | 34 帧 / 1.15 cm | 46 帧 / 2.36 cm | 0 |

**这两组数不该被当成"通过"**：不开纠正时左臂是 99 帧 / 14.19 cm，所以纠正确实起了作用，
但残留 8.07 cm 是真实网格穿透。已知原因（不用再查）：`enter_thresh=0.04` 只管深过 4 cm 的
（右臂最深 2.36 cm 全在阈值下，整段没被动过），且深帧不收敛 —— 过滤器自己的日志是
`left: fixed 53/71 (remaining 18)`，`w_ee=60` 压着 `w_pen=20` 在 60 步内解不开。
换过滤器参数后拿这张表对比即可。

### 代理盒的标定：怎么量、怎么验（grid 路线，2026-08-20）

上面那张表暴露的问题不是"检测漏了"，而是**代理盒的零点不对**：盒子偏大，很多帧代理
说穿、真实网格说干净。这件事必须拿真实网格 contacts 当真值去标，两阶段：

```bash
# 素材：必须是**没开碰撞过滤**的跑（拿过滤后的产物标定 = 循环论证），落 collcal/prefilter/
# phase1 纯几何穷举盒半长（秒级，不跑过滤器）
scripts/dev/m7_tool.sh sweep_arm_torso_params.py phase1 --lo 0.40
# phase2 真跑过滤器扫门槛（分钟级），默认那一行就是"调参前"基线
scripts/dev/m7_tool.sh sweep_arm_torso_params.py phase2 --half 0.0695 0.119 0.239 \
    --enter_thresh 0.02 0.03 --margin 0.01 0.02
```

3 段 542 帧（`fill_jar` / `sip_coffee` / `-2cNMO9Mm3Q_192.4_209.2`）的结论：

| | 盒半长 [m] | 漏 / 误 | AUC | 穿模帧（真实网格） | 最深 | 手腕挪动 均/最 |
|---|---|---|---|---|---|---|
| 调参前 | `[0.105, 0.135, 0.215]` | 0 / 183 | 0.9997 | 53/542 (9.8%) | 6.07 cm | 2.36 / 15.73 cm |
| 标定后 | `[0.0695, 0.119, 0.239]` | **0 / 0** | **1.0000** | **24/542 (4.4%)** | **4.49 cm** | 2.49 / 16.06 cm |

两件事值得记住，别下次重新推一遍：

- **形状本来就是对的**（调参前 AUC 已经 0.9997），错的是零点。所以标定的目标是"距离 0
  ⇔ 真实网格接触"，不是把检测做得更灵。AUC 在这里的用处是**破平局** —— 只按帧数排会
  出现大片同分配置，`grid_tie_break` 那笔账（66.7% vs 100%）就是这么来的。
- **零点一挪，`enter_thresh` 就不能再兼职了**。旧的大盒子隐含地提供了推出余量，标定后
  的盒子不提供，于是余量必须显式化成 `margin`：*深过 `(enter_thresh − margin)` 才修，
  推到离面 `margin` 才停*。只缩盒不给余量的那一版实测把最坏穿透从 6.07 推到 8.99 cm。

验收（两条路线各一条，都是后台跑）：

```bash
# ① 13 段 grid 重跑，和旧表逐段对比（约 7 小时；底座求解确定性，差别只有过滤器那一步）
nohup bash scripts/dev/run_collcal_ab.sh > outputs/dev/collcal_ab.log 2>&1 &
scripts/dev/m7_tool.sh collcmp_table.py --root outputs/retarget/collcmp_cal \
    --out outputs/dev/collcal_ab_table       # 漏/误两列默认按各路线自己的盒子算
# ② neural 一个字节都没动（预设为空 ⇒ 照旧构造）
bash scripts/dev/check_neural_bytes.sh > outputs/dev/neural_bytecheck.log 2>&1
#   期望 trajectory.npz / metrics.npz / robot_sim.mp4 三个 SAME（2026-08-20 实测全同）
```

#### 验收结果（2026-08-21）：一条判据过了，一条没过

13 段跑完了（`outputs/dev/collcal_ab_table/`），**grid** 的前后对比（`neural` 那三列
逐位不变，已核对，因为预设为空）：

| | 穿躯帧（网格判据） | 有残留的段数 | 代理 vs 网格 帧数差 | 其中 漏 / 误 | 最深 | ik（段均） |
|---|---|---|---|---|---|---|
| 全 13 段 调参前 | 507/1755 (28.9%) | 12/13 | 406 | 17 / 423 | 12.64 cm | 96.7% |
| 全 13 段 标定后 | **234/1755 (13.3%)** | **9/13** | 222 | 222 / **0** | 13.16 cm | 96.7% |
| 留出的 10 段 调参前 | 454/1213 (37.4%) | — | 180 | 6 / 186 | 12.64 cm | 93.8% |
| 留出的 10 段 标定后 | **210/1213 (17.3%)** | — | **198** | 198 / **0** | 13.16 cm | 93.8% |

- **判据二（穿模帧占比不许变差）过了，而且是大幅改善**：28.9% → 13.3%，没有一段变差，
  可行率一位没变（碰撞过滤在 IK 之后，本来就不该动它 —— 这一列没变本身是个正确性检查）。
- **判据一（代理/网格帧数差明显收窄）只在标定用的那 3 段上成立**（226 → 24），留出的
  10 段上 180 → 198，**没收窄，而且方向翻了面**：误报 423 → 0，漏报 17 → 222。

判据一为什么不泛化 —— 是**代理形状的天花板，不是参数没调好**。躯干真身是圆的，用一个
轴对齐盒去拟合：把角上的误报压到 0，就必须把 x 半长压到真身的 0.50 倍（0.0695 vs
0.139），于是**平面方向欠覆盖**，~1.7 cm 以内的真穿透对代理是隐形的。实测
`-0RheyDV3a0_48.6_55.3` 的 90 个残留帧，代理读数是 +0.08 ~ +0.48 cm（"还差半毫米才报警"），
网格读数却是 1.26 cm 已经穿了。所以下一步该做的是**把"检测"和"推出目标"解耦**
（大盒判、标定盒推），或者换个更贴身的代理形状，而不是继续调这三个数字。

顺带确认了一件事，**残留深的帧不是漏检**：`--oo8_XIuOM_900.3_917.4` 最坏那几帧代理读数
是 −1.92 ~ −4.70 cm，代理**报了**，是过滤器没修得动（那段 ik 只有 84.8%，属于源头坏帧）。
所以"漏 222"和"最深 13.16 cm"是两个不同的病，别混成一个。上面这些逐帧的数怎么重出：

```bash
scripts/dev/m7_tool.sh peek_penetration_frames.py \
    "$PWD/outputs/retarget/collcmp_cal/-0RheyDV3a0_48.6_55.3_grid" \
    "$PWD/outputs/retarget/collcmp_cal/--oo8_XIuOM_900.3_917.4_grid" --route grid
#   代理读数的符号就是判据：正数 = 漏检（病在检测），负数 = 报了没修动（病在过滤器/源头）
#   注意传绝对路径 —— m7_tool.sh 会 cd 到上游 retarget/，相对路径会落到别处
```

按"指标≠画面"抽帧看过（`outputs/dev/collcal_ab_table/frames/`，帧号就是上面那条命令
报的最坏帧，`ffmpeg -i <run>/robot_sim.mp4 -vf select=eq(n\,45) -vsync 0 -frames:v 1`）：
1.63 cm 那档（`-0RheyDV3a0` f45）画面上是双手抱在胸前、前臂贴着胸甲，**看不出穿**；
13.16 cm 那档（`--oo8_XIuOM_900.3` f17）**一眼就是坏的** —— 整条左小臂埋进躯干，
指尖从胸口另一侧戳出来。所以这一列必须和深度一起看。

### 默认 `--root_solver` 定为 `grid`（2026-08-21，人拍板）

碰撞过滤参数按路线分开标定之后，13 段官方片段上两条路线的账（数字出自上面
`outputs/dev/collcal_ab_table/` 那张表和 `scripts/dev/collcmp_table.py` 的产物）：

| | IK 可行率（段均） | 残留穿躯帧占比 | 有残留的段数 | ρ̄（参照 Ego2Robot 的 0.65） |
|---|---|---|---|---|
| `neural` | 87.1% | 23.8% | 11/13 | 0.441 |
| `grid` | **96.7%** | **13.3%** | **9/13** | 0.393 |

标定之前 grid 是"可行率赢、穿模输"（28.9% vs 23.8%），标定之后两项都赢，
唯一还输的是 ρ̄ 更偏离 0.65。**取默认的理由是这两件事不等价**：ρ̄ 偏低是"姿态不够
拟人"（论文里要讨论的量，但数据还能用），穿模是"这一帧直接不能用"。

**为什么改这个默认值不会让前面几节的字节比对失效** —— 四个守卫脚本
（`check_neural_bytes.sh:20` / `check_m7_unchanged_by_l3_4.sh:33` /
`check_object_tracking_bytes.sh:14` / `check_action_refine_bytes.sh:17`）**都显式传
`--root_solver neural`**，比的一直是同一条路线；显式选 `neural` 时行为逐字节不变。
反过来这是一条约束：以后新写的字节比对脚本也必须显式传这个参数，别靠默认值。

**别把 grid 说成"不用 checkpoint"** —— 它不用模型出根位姿（training-free），但
`test.py` 是无条件 `_load_model` 的，IK 求解器（`opt.ik_left/right`）挂在那个对象上，
所以裸跑 grid 仍然要 `--ckpt`。README 和 patch 帮助文本都按这个口径写。

**改完实跑过一遍，验的是"默认值真的接通了"**（2026-08-21，`-QALmP1nHtM_678.2_682.2`，
M7 / seed 0 / 两条碰撞过滤都开，命令里**一个 `--root_solver` 都没传**，
日志 `outputs/dev/b2_default_smoke.log`）：

```
K=31/63 帧（convex_hull）  质心=[-0.005, 0.084, 1.198]  r_max=1.007
最优: ik_rate=100.0%  t=[0.145, 0.184, 1.448]  （实打 210912/423612，同分 292）
IK L: 63/63 (100.0%)  R: 63/63 (100.0%)  overall: 100.0%
[ArmTorsoFilter] enter_thresh=0.020m margin=0.020m ... torso_half=[0.0695, 0.119, 0.239]
```

两件事同时被这一跑证实：① 走的是网格搜索（有 K / 候选 / 同分那三行，neural 那条不打这些）；
② `--atf_preset auto` 跟着新默认值走到了 **grid 那组标定参数**（`0.0695/0.119/0.239` +
两个 0.020，正是 `presets.GRID`），不是过滤器自己的未标定默认值 —— 这两个默认值是联动的，
只改一个会得到"grid 路线配 neural 参数"的组合，所以必须一起验。

受影响的已有素材：README 里 `demo_fill_jar.gif` 是默认值还是 `neural` 时生成的
（可行率 64% 那段），照 README 快速上手那条命令重跑现在走 grid，画面会不一样；
要复现那张图得显式加 `--root_solver neural`。

### README 里那两张图：怎么重出、图里的数从哪来

图进 git（`docs/assets/`），所以它比别的产物更容易过期 —— 代码改了、图没重出，
读者看到的就是一张**再也复现不出来**的宣传物料。防这件事的办法是把生成命令和来源 run
写死在这里，任何时候能一条命令重出：

```bash
# ①碰撞修复前后对照图（自动挑"修得最多"的那一帧，当前是 f144）
MUJOCO_GL=osmesa envs/rt_env/bin/python scripts/dev/make_readme_assets.py collision \
    outputs/migration_check/new_nocoll outputs/migration_check/new_coll --lookat 0.15 0 0.30
#   → docs/assets/collision_fix_fill_jar.png
#   预期打印：frame 144: before -10.48 cm -> after +0.04 cm（穿透帧 178 → 141 / 216）

# ②输入-输出并排 GIF
MUJOCO_GL=osmesa envs/rt_env/bin/python scripts/dev/make_readme_assets.py demo \
    outputs/retarget/collcmp/fill_jar_neural --start 20 --count 50 --step 3 --height 250 \
    --out docs/assets/demo_fill_jar.gif
#   → 50 帧 / 约 2.9 MB
```

两条要求：

1. **对照图的两个 run 必须是"同一次 IK、只差碰撞开关"**，脚本会核对 `ik_rate` 一致，
   不一致直接 `SystemExit`。否则图上的差别里混进了 IK 的随机性，就不再是碰撞过滤的功劳。
2. **挑帧规则是"修好得最多"，不是"修复前最深"。** 最深的那些帧过滤器未必修得动
   （`w_ee=60` 的保真项压着推出项），拿修不动的帧当示意图就是自欺。所以脚本挑的是
   `argmax(修复后深度 − 修复前深度)`，并且**同时把整段的穿透帧数打出来**
   （178 → 141，降了没清零）—— 一张挑出来的好帧配一个全段的真实数字，才不算选择性展示。

### 轨迹清洗：空洞填补的位置感知策略

`trajectory/traj_cleanup.py` 的空洞策略是**按位置**分的（2026-08-11 定），`FILL_REST`
是最后兜底。改这块必须跑两样：

```bash
# 1. 单测钉住策略（TestGapPolicyByPosition，6 个用例）
envs/rt_env/bin/python -m unittest tests.test_retarget_modules -v

# 2. 看画面 —— 结尾"沿袭"vs"渐入静息位"左右对比
MUJOCO_GL=osmesa envs/rt_env/bin/python scripts/dev/…（一次性脚本，见下）
```

判据不是数字而是画面：serve_cake 结尾（右手 44 帧 / 2.9 s、左手 17 帧 / 1.1 s）旧策略
渐入静息位，等于**凭空编出最多 74.5° 的关节运动**（右臂逐关节最大
`[10.9, 22.3, 10.2, 45.1, 16.2, 5.7, 74.5]`，均值 18.2°；左臂 24.0°/均值 9.4°）。
f175 / f188 两张图能直接看出来：新策略两手停在最后一次测到的持盘姿态，旧策略两条手臂
垂到体侧默认位。存档 `outputs/dev/tail_policy_serve_cake/tail_policy_hold_vs_rest.mp4`。

结尾保持出来的帧**仍然标 `FILL_HOLD` 而不是 `OK`**，长度记在 `report["tail_hold"]`
并打一行 ⚠ —— 单测 `test_long_tail_hold_is_reported_not_silent` 钉的就是这一点。
全片段普查（11 个官方片段）确认这次改动只翻了三处片尾（serve_cake 左 17f / 右 44f、
ours_webapple 右 58f），别的空洞一帧没动。

### 坏帧过滤的另外两个粒度（episode / segment，2026-08-21）

`trajectory/tiers.py`，判据来源是 EgoSmith（EgoSteer，arXiv 2607.09701 §3）那套按
**三个粒度分别设判据**的清洗管线。frame 级早就有了（上一节那个 `traj_cleanup.py`），
这次补的是 episode 和 segment。开关 `--bad_frame_tiers episode,segment,frame`，
**默认 `frame`，等于现状不变**。

```bash
# 1. 单测：26 个用例，秒级（判据本身 + "什么都不改"这件事）
envs/rt_env/bin/python -m unittest tests.test_badframe_tiers -v

# 2. 端到端：三层全开 vs 默认跑法，产物必须逐字节相同
bash scripts/dev/check_tiers_yaml_bytes.sh > outputs/dev/tiers_yaml_bytecheck.log 2>&1
```

**这里"验过了"具体指什么**（三条，都是实测不是推理）：

1. **默认跑法的产物一个字节没变。** 参照物是 A1 标定那次留下的
   `outputs/dev/neural_bytecheck/base/`，时间戳早于本次全部改动，`cmp` + md5 双查。
   2026-08-21 实测三个 SAME：`trajectory.npz` = `9ef35b4eed590c543ae4af9c9b89e5c9`、
   `metrics.npz` = `33c049ac5b26fd848cdbcfa93321fae8`、
   `robot_sim.mp4` = `205d96dba4a701e4be19a88ff1ec0483`（这个数和
   `external/patches/README.md` 里那个对得上，是同一条参照线）。
   为什么非要真跑：这次动了上游 `scripts/test.py` 的 **argparse**，而 argparse 的
   参数顺序都可能挪动随机数流 —— 这是这份文档里已经栽过一次的坑（见文末方法论第 4 步）。
2. **三层全开时"只看不动"。** `tiers` 那一遍和 `base` 比 `trajectory.npz` /
   `root_frames.npz` / `metrics.npz` / `robot_sim.mp4` / `input_viz.mp4` 五个全同
   （2026-08-21 实测 5/5 SAME），只许多出 `bad_frame_tiers.json` 一个文件；
   反过来 `base` 里**不许**有这个文件（多写一个文件也算产物变了，实测确认没有），
   `base.log` 里 `[tier:` 出现 **0** 次。单测那边还有一条更狠的：
   `test_nothing_is_modified` 用 `np.array_equal(..., equal_nan=True)` 逐位比
   输入数组，连 NaN 的位置都得一样。
3. **检出之后的处理方式和论文不同，是我们定的。** episode 级**只警告不阻断**、
   segment 级**只标记不处理**（不插值/不丢弃/不填补）。实测那段官方片段
   （`-1r9yl-P-Ao_86.3_90.8`）打的就是这个：
   ```
   [tier:episode] 相机运动分布正常（2/68 对帧离群，占 2.9%）
   [tier:segment] 7 处空间离群标记（手腕 1 / 手指 6）—— **不做任何自动处理**，
                  留给 refine 决定要不要升级精修
   ```
   episode 这次判的是"正常"（`warn: false`，2.9% < `frac_thresh` 5%，但 `max_robust_z`
   到了 5.3，所以 f23/24 那两帧还是记进了 `outlier_frames` —— **判决和证据分开写**，
   人要复核有据可查）。segment 那 7 处 **全部原样留在 json 里**，`trajectory.npz`
   逐字节没变就是它"什么都没做"的证据。报告里还带一行 `source`，把出处
   （arXiv 2607.09701）、判据依据（§V2/§V3）、和"处置策略是我方决定、与原文 discard
   不同"三件事写在产物里，免得下游读到这个 json 的人以为该照论文那样扔掉。

**一处判据本身的修正（单测逼出来的，值得记）：** 稳健 z 原来只有 MAD 一个尺度，
`MAD == 0` 就直接返回全零"没有离群"。但**零阶保持填补过的段是逐位常量**，MAD 恰好
为 0 —— 于是一段"常量 + 一个 25 cm 尖峰"会被判成干净的。这是真盲区不是测试写错，
所以改的是统计量：MAD 为 0 时退到平均绝对偏差尺度（`1.253314`，Iglewicz–Hoaglin
原文自己给 MAD=0 开的那条路），两个分支各有一个用例钉住
（`test_hold_filled_segment_with_one_spike_is_still_caught` /
`test_truly_constant_segment_is_not_flagged`）。

阈值那三个数（`z_thresh=3.5` / `frac_thresh=0.05` / `seg_sec=2.0`）**是惯例不是实测**，
以及我们做不到原文哪一步（跨语料离群、硬旋转阈值），记在
[`BACKLOG.md`](BACKLOG.md) C19/C20，模块 docstring 里也逐条写了。

### 机器人参数搬进 yaml（2026-08-21）

格式借 HandUMI（robonet-ai.github.io/handumi-sw）的机器人配置设计：**一台机器人一个
yaml**，装关节限位 / 静息姿态 / 碰撞参数，每组带 `verified: true|false`。只借格式，
数值全是我们自己的。入口 `robots/params.py`，文件 `configs/robots/{m7,l3_4}.yaml`。

```bash
# 1. 单测：17 个用例，秒级
envs/rt_env/bin/python -m unittest tests.test_robot_params_yaml -v

# 2. 看哪些数字是实测的（✅ / ⬜ 一览）
envs/rt_env/bin/python -m web2robot.robots.params m7

# 3. 端到端：同上那个脚本（搬参数和加 tiers 是同一次改动，一起验的）
bash scripts/dev/check_tiers_yaml_bytes.sh > outputs/dev/tiers_yaml_bytecheck.log 2>&1
```

**为什么这件事非要端到端实测**，光看代码不够：搬参数踩的坑不在数值上，在**类型**和
**求值时机**上。实际遇到的两个 ——

* PyYAML 把 `1e-3` 读成**字符串**（它的 float 解析器要求有小数点和带符号的指数），
  所以 yaml 里必须写 `0.001`；
* `CONFIG["start_config"]` 存的是 **float32**，`0.20` 取出来是 `0.20000000298023224`，
  拿 yaml 的 float64 去比会红 —— 比较必须在 float32 上做。

还有 list vs tuple（`GRID["torso_half"]` 要 tuple，调用方直接往 `M7CapsuleModel` 传，
不该能被就地改掉）。这些都是"逐位相同"级别的差别，只有比 md5 才发现。

**"不留第二份"是怎么钉住的**（`tests/test_robot_params_yaml.py` 三件事）：

1. **AST 扫 `src/**/*.py` 的浮点字面量**去撞已搬走的那 15 个数。用 AST 而不是 grep 是
   故意的 —— 注释和 docstring 里引用 MJCF 原值**应该**留着，只有代码里的字面量才算
   第二份来源。搬迁时这条真的抓到一处：`scripts/dev/sweep_arm_torso_params.py` 自己
   抄了一份 `MESH_HALF`（标定脚本的分母和被标定的代码不同源，扫出来的比例会悄悄错位），
   已改成从 `presets.MESH_HALF` 读。
2. **构造签名的默认值 == yaml 里那个数。** yaml 在 **import 时**加载，加载到的值**就是**
   `__init__` 的默认值，所以 `inspect.signature(ArmTorsoFilter.__init__)` 拿到的
   `enter_thresh` 仍然是 `0.04`，`tests/test_module_boundaries.py` 那边钉的"这个数必须是
   0.04"照旧过。两个测试合起来才完整：一个守数值，一个守来源。
3. **`verified: true` 按名单钉死。** 只有两组：`collision.arm_torso.routes.grid`
   （A1 那次 sweep 标定的 `torso_half=[0.0695, 0.119, 0.239]` / `enter_thresh=0.02` /
   `margin=0.02`）和 `collision.mesh_aabb_half`（量自躯干网格 AABB，是 sweep 的分母）。
   谁想给一组没标定过的参数标 true，必须先来改这个名单 —— 这个标志位的全部价值就是
   让人分清"实测"和"暂时用着"，一旦扩散就废了。

**yaml 里的数真的送进了 MuJoCo，不只是被 import 了。** 上面那个脚本第 4 遍跑
`--root_solver grid --atf_preset auto`，过滤器自己打出来的就是 yaml 里那一组
（2026-08-21 实测）：

```
[ArmTorsoFilter] enter_thresh=0.020m margin=0.020m w_pen=20.0 w_ee=60.0 w_prox=1.0
                 torso_half=[0.0695, 0.119, 0.239]
[ArmTorsoFilter] left: fixed 25/25 (remaining 0); right: fixed 46/46 (remaining 0)
```

对照 `neural` 那两遍打的是过滤器默认值（`enter_thresh=0.040m margin=0.000m
torso_half=[0.105, 0.135, 0.215]`）—— 两条路线各取各的一组，这才叫"预设按路线生效"。
只验单测不跑这一遍是不够的：单测能证明 `presets.GRID` 等于 yaml，证明不了它被传到了
构造函数里。

**两个刻意的"不"**：

* **`neural` 路线标 `verified: false`。** 它的覆盖集是**空的** —— 空覆盖集谈不上标定过。
  它的凭据是字节级 md5（`check_neural_bytes.sh`），不是量出来的数字，yaml 的 `source`
  字段就这么写的。
* **L3.4 没有 `collision:` 一节。** 这是主张不是漏写，有测试
  （`TestL34HasNoCollisionSection`）钉住。写一节 `verified: false` 的复制品进来，只会
  让人以为"L3.4 也支持，参数就在这儿"。见 [`BACKLOG.md`](BACKLOG.md) C21。
* **两台机器人的 yaml 不用 anchor 共享。** 数值目前相同（量过的），但各自的 MJCF 才是
  各自的真源；共享会在哪天拿到不同批次机器人的时候，把 M7 的数悄悄按到 L3.4 上。

**哪些参数刻意**不**进 yaml**：body 名、骨骼拓扑、指尖清单、`TORSO_CENTER`、骨半径 ——
这些是**结构事实**，唯一真源是 MJCF，留在模型旁边。判据是"标定扫描能不能设它"
（= 是不是构造参数）：能设的进 yaml，结构事实不进。写错一个 body 名不会报错，
`pytorch_kinematics` 会安静地建出一条空链。

搬迁过程中看出来的数值疑点**一个都没改**，记在 [`BACKLOG.md`](BACKLOG.md) C18
（代理盒比网格 AABB 收了 3 cm 没依据、`start_config` 的 ±0.20 从没比过、
`arm_torso.defaults` 11 个里只有 3 个标定过）—— 参数改动是单独一件要决策的事。

## M7 机器人定义（`src/web2robot/robots/m7/`、`assets/robots/m7/`）

两个验收脚本，输出要和上一次逐字节一致：

```bash
scripts/dev/m7_tool.sh verify_m7_mjx_fk.py            # 期望 0.0000 mm / 0.0000 deg  MATCH ✓
scripts/dev/m7_tool.sh check_handframe_convention.py  # 期望 m7 左手 finger=+y thumb=-x palm=+z，右手镜像

# 再加一档：拿一段真实重定向轨迹逐帧验，而不是只验资产里写死的 home 姿态
scripts/dev/m7_tool.sh check_handframe_convention.py \
    --traj outputs/legacy_runs/runs/m7/validation/fill_jar
#   期望 "违反约定的帧: 0/216 ✓"，且两只手整段各只出现过一种轴向组合
#   （--traj 要给绝对路径或相对**上游 retarget/** 的路径，m7_tool.sh 会 cd 过去）
```

**永远不要只验一侧**，理由见 [`PITFALLS.md`](PITFALLS.md) 第 16 条。
动完资产位置、删掉旧目录之后**必须再跑一次端到端**，理由见第 11 条。

## L3.4 机器人定义（`src/web2robot/robots/l3_4/`、`assets/robots/l3_4/`）

第二台机器人，和 M7 **并列可切换**（`--robot m7|l3_4`）。它的上肢和 M7 逐位同构，
所以验收的重点和 M7 那节不同：不是"数值对不对"，而是**"同构有没有被偷偷写成依赖"**。

```bash
# 1. 资产：从厂家原包重建，七步自检，任何一步不过就 SystemExit
envs/rt_env/bin/python scripts/dev/build_l3_4_assets.py --force
#   期望七行全过，其中三行是这台机器人的立身之本：
#   [4/7] hand_frame quat 现算 + 逐轴断言（左 finger+y/thumb-x/palm+z，右镜像）
#         —— 算出来的两个 quat 和 M7 已提交的那两个逐位相同，是对整条链的独立交叉验证
#   [6/7] 双臂链与 m7_mjx.xml 逐 body / 逐关节比对（期望最大偏差 ~3.8e-07）
#         —— **这一行就是"借 M7 根模型 ckpt"的全部依据；它红了就必须重训，不许照跑**
#   [7/7] 对厂家自带的 l3.4.xml 校验（55 个关节 axis/range + 55 个 body pos 全同）
#         —— 回答"厂家给的 .urdf 和 .xml 是不是一致可用"，不靠肉眼看

# 2. 表 vs MJCF：12 个用例，秒级
envs/rt_env/bin/python -m unittest tests.test_l3_4_robot -v

# 3. hand_frame 逐帧验（同 M7，永远两只手都验）
envs/rt_env/bin/python -m unittest tests.test_l3_4_robot.TestHandFrameConvention -v
```

三条一定要知道的：

**① 真源是各自的 MJCF，不是另一台机器人的表。** L3.4 的限位表、start_config、采样参数
和 `robots/m7/` 数值相同（量出来的：43 个同名关节 axis/range 全同、43 个同名 body 的
pos/quat 全同），但两个包**一行代码都不共享**，也没有 alias。测试断言的是
"表 == `l3_4.xml` 里那个关节的 `range`"，**不是** "表 == M7 的表" —— 后者会在哪天真拿到
不同批次的机器人时红在"和 M7 不一样"上，而那时候不一样才是对的。

**② 借 M7 的 ckpt 有明确的失效条件。** 根位姿模型的输入输出由
`waist_pitch_link → hand_frame` 这条链决定，而这条链两台逐位相同，所以
`--robot l3_4 --ckpt runs/m7/taskspace_v2/checkpoints/final.pt` 是有依据的，不是凑合。
**腿一旦从 `LOCKED_JOINTS` 里解锁，base_link 相对地面的高度就变了，这个结论立刻失效。**
（`--root_solver grid` 那条路线压根不用模型，只用 IK；但上游 `test.py` 无条件 `_load_model`，
所以还是得传 `--ckpt`。）

**③ 锁死不是"没人去写"。** 腰/颈/腿 17 个自由度由 `env._apply_locked()` 在**每次
`mj_forward` 之前**按住，因为上层（碰撞过滤、渲染）会直接改 `data.qpos` 再 forward。
测试里专门有一步把这 17 个 qpos 篡改成 0.37 再走一次，验它们回到锁定值。

**碰撞参数不用重标。** 躯干代理盒挂在 `waist_pitch_link` 上，两台机器人这个 mesh 的
AABB 逐位相同（中心 `[0.0057, 0, 0.2166]`、半长 `[0.1313, 0.17, 0.2326]`），
`M7CapsuleModel` 用到的 `BONES` / `FINGERTIPS` 名字在 L3.4 里全部存在 —— 所以
`presets.py` 里那份**已标定**的 grid 预设是精确适用的，不是"按比例套一版粗略的"。

**加了 L3.4，M7 的产物一个字节都没变** —— 这是硬要求，所以有脚本：

```bash
bash scripts/dev/check_m7_unchanged_by_l3_4.sh > outputs/dev/l34_m7_unchanged.log 2>&1
#   同片段（-1r9yl-P-Ao_86.3_90.8）/ 同 seed / 同路线（neural）/ 两条碰撞过滤都开，
#   和 L3.4 改动**之前**留下的 outputs/dev/neural_bytecheck/base/ 比 md5
#   2026-08-20 实测 trajectory.npz / metrics.npz / robot_sim.mp4 三个 SAME
#   （robot_sim.mp4 = 205d96dba4a701e4be19a88ff1ec0483，和 patches/README.md 里那个数一致）
```

只跑一遍就够：参照物是 A1 那次标定验证留下的，时间戳早于 L3.4 的所有改动。
为什么不能只看代码：`sim/robots/__init__.py` 里那两行 `from web2robot.robots.l3_4 import ...`
是**模块顶层**的，跑 M7 也会执行 —— "顶层 import 应该没有副作用"和"产物没变"是两件事。

**腰以下在画面里是空的。** 厂家包里一个 mesh 都没有；94 个零件和 M7 相同（质量/惯量/COM
逐位相同）直接 symlink，14 个腿部 + 盆骨 `base_link` 没有正确的 mesh（M7 那个同名
`base_link.STL` 是升降柱底座，是另一个零件），几何被删掉。上肢重定向的每个数字都不受
影响，但**出片之前得把腿的 mesh 要到**，见 [`BACKLOG.md`](BACKLOG.md)。

## `evidence/` 里的数（`src/web2robot/eval/`）

```bash
envs/rt_env/bin/python -m unittest tests.test_depth_benchmark -v   # 19 个用例，0.3 秒
envs/rt_env/bin/python scripts/dev/render_depth_benchmark_fig.py   # 重画汇总图
```

这份测试的作用和别的不一样：它不防"代码改坏"，防的是**论文里的数字和仓库里的证据
悄悄脱钩**。所以断言写的是具体数值不是"大于小于" —— 把中位偷偷改成均值，ABF12 的
11.0 会变 11.26、SMu41 的 3.5 会变 6.7，测试当场变红（验证过）。

引用那张深度误差表时必须一起写上的两句 caveat，见
[`PROJECT_LAYOUT.md` §3.1](PROJECT_LAYOUT.md)。

---

## 迁移/重构的验收方法论（五步，后续模块照抄）

这套是 2026-08 那轮重构定下来的，五步缺一步都出过事：

1. **逐行 diff 证明是纯移动** —— 先证明"没改逻辑"，再谈别的。
2. **隔离对比**：把模块从整条链里拽出来，喂同一份输入跑旧/新两份实现。纯 CPU 的
   要求逐位相同；有 GPU/随机源的降级到"判决一致 + 信号未越阈"。
3. **端到端必须固定 `--seed`**，否则上游的随机锚点会把差异伪造成几十度关节角差。
4. **确认没有留下重复副本**。留两份比留一份危险 —— 下次改代码会改到错的那一份，
   而这正是重构要消灭的失效模式。删除后再跑一次端到端（拼接路径躲得过 grep）。
5. **出片，用眼睛看。**

另外有一个整体性的指标：**迁移做对了，上游 patch 的行数就该往下走**
（313 → 233 insertions）。逻辑进 `src/`、上游只剩接线，patch 就该变小。

它变大不一定是错，但**必须当场解释清楚多的是什么**：2026-08-18 加
`--root_solver grid` 让它涨到 342，多出来的是一个 if/else 开关＋把 FK/IK 包成
callable 的接线；2026-08-19 加 `--object_tracking` 和 `--action_refine` 再涨到 428，
多出来的是三组 argparse 选项、一处参数矛盾检查、两个调用点；2026-08-20 加
`--atf_preset` 和第二台机器人 L3.4 涨到 520，多出来的是"去 `presets.py` 查表"的接线
和一台新机器人在注册表 / IK / 手部重定向器三处的注册（明细都在
[`external/patches/README.md`](../external/patches/README.md)）。
判据不是行数本身，是"上游文件里有没有出现只有那里才有的方法逻辑"——
没解释的增长才是警报。

**新开关还要额外过一道"默认关闭 ⇒ 产物逐字节不变"**：同一段片段、同一个 seed
跑 base / 显式关 / 显式开三遍以上，原有产物的 md5 必须全同，新开关只许**多**
产物、不许改产物。目前有两道，都是脚本化的：
[`check_object_tracking_bytes.sh`](../scripts/dev/check_object_tracking_bytes.sh)（3 遍）
和 [`check_action_refine_bytes.sh`](../scripts/dev/check_action_refine_bytes.sh)（4 遍，
含 `--object_tracking on --action_refine none` 这种组合）。2026-08-19 实测两道都是
原有 5 个产物全同、新文件只增不改。
这比"读代码看默认值"强 —— 默认值对但 import 有副作用、或者 argparse 顺序变了影响
随机数流，都只有比 md5 才发现。

**参数矛盾要当场退出，不许静默降级。** `--action_refine mpc|rl` 缺
`--object_tracking on` 时直接 `SystemExit`（在 `run()` 第一行，不是跑完才报），
`mpc` / `rl` 求解器本身也 `NotImplementedError` 而不是退回 Replay。理由是同一条：
一份"以为精修过"的数据会直接进训练集，比一次失败贵得多。
