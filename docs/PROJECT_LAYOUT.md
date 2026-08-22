# 工程结构总览 —— 一切有迹可循

这份文档的用途：**不翻文件夹就能定位到想找的东西。** 新接手这个工程的人（包括半年后的
自己）只看这一份，应该就知道每样东西该在哪、为什么在那。

[`../README.md`](../README.md) 是给新人看的"这个项目是什么、怎么启动"；这份是"什么东西
放在哪"。规矩看 [`CONVENTIONS.md`](CONVENTIONS.md)，坑看 [`PITFALLS.md`](PITFALLS.md)，
验收看 [`VERIFICATION.md`](VERIFICATION.md)。四份不重复。

---

## 0. 一句话导航

| 我要找… | 去哪 |
|---|---|
| 某个流水线环节的代码 | `src/web2robot/<环节名>/` —— 包名就是环节名 |
| 怎么跑一条命令 | `scripts/s*.sh`（薄壳，一个环节一个） |
| 论文要引的实验数字 | `evidence/` —— **只有这里的产物进 git** |
| 跑出来的视频 / npz / clip | `outputs/`，按"谁写的"分子目录（见 §4） |
| 原始视频素材 | `data/` |
| 官方片段对应的原片画面 | `scripts/s0_fetch_rgb.sh`（`src/web2robot/fetch/`）—— 起止时刻**只认 `scene.json` 的 `video_source`**，目录名里的秒数实测不可信 |
| 画面里的人手在哪（掩码） | `scripts/s5_hand_mask.sh`（`src/web2robot/synth/`）—— 官方 MANO 网格投影后**必须逐帧对齐**才和画面对得上，素材清单见 [`VISUAL_SYNTH_INPUTS.md`](VISUAL_SYNTH_INPUTS.md) |
| 机器人 MJCF / mesh | `assets/robots/<机器人名>/`（`m7`、`l3_4`） |
| 机器人参数（关节限位 / 静息姿态 / 碰撞盒与门槛） | `configs/robots/<机器人名>.yaml` —— **一台机器人一个文件，代码里不留第二份**；每组带 `verified` 说明是不是实测标定的 |
| 想跳过质检/路由（公司自己有一套） | `--quality_gate skip` / `--routing skip`，取值集合只写在 `src/web2robot/quality/config.py` 的 `GATE_MODES` / `ROUTING_MODES`；默认 `builtin` = 行为不变 |
| 绝对路径、checkpoint 位置 | `configs/paths.yaml` —— **全工程唯一允许写绝对路径的文件** |
| 我们对第三方仓库改了什么 | `external/patches/README.md` |
| 某个决定当时为什么那么定 | `docs/` + 各模块 `__init__.py` 的文档字符串 |
| 重要参考资料的确切路径 | **本文 §3** |

---

## 1. 目录树 + 流水线环节对应

```
web2robot/
│
├── src/web2robot/            ← 全部逻辑。一个流水线环节一个包
│   ├── paths.py                 路径解析总入口（P.weights() / P.check_output_dir()）
│   ├── common/                  跨环节共用（video_io：解码抽帧）
│   ├── fetch/       ══════ ⓿取原始画面   官方片段元数据 → 源视频 → 逐帧对应的 RGB
│   │                            官方发布里一帧真 RGB 都没有（depth.mp4 是深度），
│   │                            而"抠人换机器人"必须要 RGB，所以补了这一档；
│   │                            时间轴只认 scene.json 的 video_source（目录名不可信，
│   │                            实测有一段差 3.39 s = 102 源帧）。`--dry_run` 不碰网络
│   ├── quality/     ══════ ①取景质检     拍全了没 / 稳不稳 / 背景合不合适
│   │                            `python -m web2robot.quality --quality_gate skip` 整档跳过
│   │                            （2026-08-21：公司已有质检体系，这一档降级为可选）
│   ├── routing/     ══════ ②视角与运动分类  第一或第三人称、相机动不动 → 选技术路线
│   │                            `--routing skip` 只关路由，质检信号照样全跑
│   ├── perception/  ══════ ③感知前端
│   │   ├── hawor.py             相机运动的片段走这条（SLAM，深度准，条件不满足整段失败）
│   │   ├── wilor.py + moge.py   相机固定的片段走这条（逐帧，从不崩溃，深度差 → 见 §3）
│   │   └── to_clip.py           下游输入契约（EgoInfinity clip 目录），与用哪个前端无关
│   ├── retarget/    ══════ ④重定向        坏帧兜底 + 两条并列的根位姿路线
│   │   ├── fallback.py          坏帧/丢帧兜底在流水线里的编排
│   │   ├── root_anchor.py       逐帧生成模型的 best-of-N 锚点采样（上游那条）
│   │   └── root_grid.py         静态网格搜索根位姿（Qwen-RobotManip 公式 3），
│   │                            `test.py --root_solver grid` 切换，不训练
│   ├── robots/                  机器人定义（IK 链、hand_frame 约定、采样配置）
│   │                            ——**不 import 任何重定向框架**，换框架时不用改；
│   │                            **两台机器人之间也不互相 import**，各自以自己的 MJCF 为真源
│   │   ├── params.py            读 `configs/robots/<机器人>.yaml`（IK 关节限位/静息姿态/
│   │   │                        碰撞参数的**唯一来源** + `verified` 标志位）
│   │   ├── m7/                  M7（RoboEra），双 7-DoF 臂 + 两只 12-DoF 手 + 升降柱
│   │   └── l3_4/                L3.4（rel3_4），同一双臂同一只手，挂在腰+腿上；
│   │                            本阶段只做上肢，腰/颈/腿 17 个自由度锁死在 `LOCKED_JOINTS`
│   ├── twin/                    物体 6D 位姿（EgoEngine §3.1 数字孪生），
│   │                            `test.py --object_tracking on` 切换，默认 off
│   ├── refine/                  动作分级精修的判决（EgoEngine §3.2.2），
│   │                            `test.py --action_refine mpc|rl` 切换，默认 none。
│   │                            只判不解 —— mpc/rl 求解器未实现，明确报错
│   ├── collision/   ══════ ⑤碰撞检测      臂-躯 / 双手 / 手指胶囊过滤
│   ├── trajectory/  ══════ ⑤轨迹处理      坏帧三级检测 + 长度感知填补
│   │   ├── traj_cleanup.py      **逐帧**那一层（跳变/鼓包/四元数翻转）+ 填补
│   │   └── tiers.py             EgoSmith 的另外两个粒度（整段/轨迹段），
│   │                            `test.py --bad_frame_tiers` 切换，默认只有 frame；
│   │                            **只警告/只标记，一个数都不改**
│   ├── synth/       ══════ ⑥视觉合成      抠掉画面里的人、贴上渲染的机器人（任务B）
│   │                            现在只有第一块：MANO 网格 → 手部掩码。**只投影是对不上
│   │                            画面的** —— 官方 3D 手的尺度逐帧在漂（实测 0.32–0.95），
│   │                            所以先用 hand_joints_2d.bin 逐帧逐手拟合 s+t 再光栅化
│   │                            （裸投影差 9.3 px 中位 → 对齐后 3.7 px）。
│   │                            核对图的底是深度 —— 真 RGB 还卡在 BACKLOG B12
│   └── eval/                    评测代码（给 evidence/ 算表用，纯 numpy、秒级）
│
├── scripts/                 ← 薄壳：只负责"用对的解释器 + 设好 PYTHONPATH"，不含逻辑
│   ├── s0_fetch_rgb.sh          ⓿（片段名以 `-` 开头，--clip 要写成 `--clip=-xxx_1.0_2.0`）
│   ├── s1_quality_gate.sh       ①
│   ├── s3_to_clip.sh            ③（子命令 hawor / wilor，各自的 venv）
│   ├── s4_retarget.sh           ④＋⑤（调上游主流程，碰撞/清洗走我方包）
│   ├── s5_hand_mask.sh          ⑥（`--no_align` 是对照开关，不是省事开关）
│   └── dev/                     开发期工具：check_* 回归比对、render_*/viz_* 出片、
│                                 build_l3_4_assets.py（从厂家原包生成 L3.4 资产）
│
├── tests/                   ← stdlib unittest，秒级，427/427
│   └── regression/              回归基准片段 + 期望判决（qc.jsonl / contact_sheet.png）
│
├── configs/
│   ├── paths.yaml           ← 唯一允许写绝对路径的地方。换机器只改这一个文件
│   └── robots/<机器人>.yaml ← **一台机器人一个文件**（格式借 HandUMI）：IK 关节限位 /
│                              静息姿态 / 碰撞参数，每组带 `verified: true|false` 说明
│                              这些数字是实测标定的还是"暂时用着"的默认值。
│                              代码侧唯一入口 `robots/params.py`，**不许留第二份**
│
├── assets/                  ← 我们产出的资产，进 git
│   ├── robots/m7/               MJCF / URDF / mesh / MJX（103 个文件）
│   ├── robots/l3_4/             同上，由 `scripts/dev/build_l3_4_assets.py` 从厂家原包
│   │                            生成（七步自检，**别手改**）；94 个 mesh 是指向 m7/ 的
│   │                            symlink（同一批零件），腿部 14 个 + 盆骨没 mesh
│   ├── robots/urdf.tar.gz       厂家原包（L3.4），生成脚本的唯一输入 —— **现在缺这个文件**，
│   │                            所以生成脚本暂时跑不了（已生成的资产不受影响），见 BACKLOG §D
│   └── weights/                 第三方权重的落地点（gitignore，`.gitkeep` 占位）
│
├── evidence/                ← 论文要引的证据。**进 git**（详见 §2 的三方边界）
│   └── depth_benchmark_ho3d/    深度误差 11cm → 0.6cm 那份，见 §3.1
│
├── data/                    ← 原始素材，不进 git（只有 README/MANIFEST 进）
│   ├── videos/                  质检用的候选视频（symlink 到旧目录）
│   └── webvid/raw/              手工挑的 7 段原片 + MANIFEST.md5（重抓不回来）
│
├── outputs/                 ← 全部产物，不进 git。**产物只许落这里**（见 §4）
│   ├── fetch/                   ⓿的产物：<片段>/rgb.mp4 + frames_index.json +
│   │                            align_report.json；`_sources/` 是源视频缓存（同一支只下一次）
│   ├── synth/                   ⑥的产物：<片段>/handmask_check.png（深度底的核对图）+
│   │                            hand_masks.npz（左右手按位打包）+ handmask.jsonl（逐段判据）
│   ├── clips/                   ③的产物：EgoInfinity clip 目录
│   ├── retarget/                ④⑤的产物：trajectory.npz / robot_sim.mp4 / input_viz.mp4
│   ├── twin/                    物体位姿单跑的产物：object_poses.npz / .json / object_viz.mp4
│   ├── viz/                     给人看的结论片（四宫格、对比图）← §3.2
│   ├── dev/                     scripts/dev/ 出的片
│   ├── migration_check/         迁移期的新旧对比 run
│   ├── legacy_runs/             2026-08-10 从 external/ 搬回来的 316 MB 存量
│   └── archive/                 阶段性封存（`<主题>_<年-月>/`）
│
├── external/                ← 第三方仓库的 **symlink**，里面不改代码
│   ├── EgoInfinity -> ../../EgoInfinity
│   ├── HaWoR       -> ../../HaWoR
│   └── patches/                 我们对上游的改动全部记在这里
│
├── envs/                    ← 三个 venv 的 symlink + requirements-*.txt
├── docs/                    ← 决策记录、优先级、待办（本文也在这）
│   └── assets/                  README 引用的图 / GIF，**进 git**（唯一一处产物不在 outputs/）
└── archive/                 ← 空占位；重构前的旧目录在 configs/paths.yaml 里注册为只读
```

流水线图和目录的对应关系是**一对一的**，这是故意的：看到图上某个框，包名就是框上的字。
唯一的例外是⑤，它是两个包 —— `collision/`（空间上的对不对）和 `trajectory/`（时间上的
连不连），因为它们的失效方式不同、验证方式也不同。

---

## 2. 三个最容易混的目录：`data/` vs `outputs/` vs `evidence/`

分界线只有一条问题：**丢了以后能不能拿回来。**

| | 丢了怎么办 | 进 git 吗 | 放什么 |
|---|---|---|---|
| `data/` | **拿不回来**（手工挑的素材，没有下载脚本） | 素材不进，**说明和 md5 清单进** | 原始视频 |
| `outputs/` | 重跑一遍流水线 | 不进 | clip、轨迹、视频、日志 |
| `evidence/` | **可能再也算不出来** | **进** | 论文要引的原始测量值 |

`evidence/` 之所以单列，是因为它的复现路径很脆：外部数据集会被清、第三方 checkout 会被
`git clean`、3 GB checkpoint 不在库里、机器会换。所以它的规矩比 `outputs/` 严
（见 [`evidence/README.md`](../evidence/README.md)）：**小 / 存原始测量值不存结论数字 /
秒级可复核零重依赖 / 每个数都有测试钉着**。

`external/` 的性质要单说：它是**别人家的目录**，我们只有读权限的心态。产物落进去的危害
不是乱，是一次 `git clean -xdf` 就全没 —— 实测攒过 408 MB 我们的产物在里面，而上游 git
只跟踪其中 1 个。这条判据写成了代码（`P.check_output_dir()`，违反就 `SystemExit`）而不是
写在文档里，理由见 [`CONVENTIONS.md` 第 1 条](CONVENTIONS.md)。

---

## 3. 重要参考资料 —— 逐条记住位置

### 3.1 深度误差对比实验（11 cm → 0.6 cm）★ 论文核心材料

整条链路"单目深度是硬瓶颈、HaWoR 是解法"的证据。**这是目前最不可替代的一份材料。**

| 路径 | 是什么 |
|---|---|
| [`evidence/depth_benchmark_ho3d/README.md`](../evidence/depth_benchmark_ho3d/README.md) | 结论、口径、怎么复核。**先看这个** |
| `evidence/depth_benchmark_ho3d/data/bench_{ABF12,SMu41,MC4}.npz` | 冻结的原始 3D 手腕点（24 KB）。存的是**测量值不是"11.0 cm"** |
| `evidence/depth_benchmark_ho3d/figures/FIG_SUMMARY_3seq.png` | 汇总图（可由脚本重画） |
| `evidence/depth_benchmark_ho3d/figures/original_2026-07-14/` | 2026-07-14 首次跑出来的原始四张图，**不重画，留档** |
| `evidence/depth_benchmark_ho3d/provenance/` | 那三次运行的 stdout（gz，22 KB）—— 溯源，见下 |
| `src/web2robot/eval/depth_benchmark.py` | 算表的代码，纯 numpy |
| `tests/test_depth_benchmark.py` | 19 个用例 0.3 秒，把论文里的每个数钉住 |

**引用这张表时有两句话必须一起写上**（钉在 `tests/test_depth_benchmark.py::TestProvenance`）：

1. **HaWoR 的度量尺度是它每段现估的** —— 三条序列量到 0.19 / 2.34 / 3.92，**差 20 倍**。
   重跑拿到别的尺度，整张表都会变，所以出现异常先查尺度再怀疑别的。
2. **HaWoR 跑的是默认 focal 600，WiLoR 那条用了 HO-3D 的真 `camMat`** —— 也就是这份对比
   **对 WiLoR 有利**，而 WiLoR 仍差一个量级。方向因此更稳，但不能不提。

### 3.2 "两种深度估计策略各错一半"——新发现的对比视频

| 路径 | 是什么 |
|---|---|
| **`outputs/viz/wilor_depth_modes.mp4`** | 四宫格对比片（h264，1430×770）。左右两个 3D 面板**共用同一个视野半径** —— 各自 autoscale 会把 6.5 倍的尺度差藏起来 |
| `scripts/dev/viz_wilor_depth_modes.py` | 出这个片的脚本（可重跑，命令在 [`VERIFICATION.md`](VERIFICATION.md) 的③感知小节） |
| `src/web2robot/perception/wilor.py` 文件头 | 那张骨长表 + 为什么两条策略的"开合"数字不可比 |
| `outputs/clips/cli_smoke_abf12_{pointmap,globalscale,K}/` | 三条深度路径各自的 clip 产物 |

结论（ABF12 前 30 帧实测，判据是骨长 —— 真手 MANO 骨长 2~4 cm 且逐帧近似常数）：

| | 骨长均值 | 骨长逐帧变异 | 病在哪 |
|---|---|---|---|
| `pointmap` | 2.94 cm ✓ | **5.7%** ✗ | 尺度对，手形被深度噪声撕开 |
| `global-scale` | **0.45 cm** ✗ | 0.5% ✓ | 手形对，整只手缩小约 6.5 倍 |

**这是一个待评估的新方向，不是已完成的工作**：取长补短（WiLoR 手形 + MoGe 逐帧手腕深度锚）
是新设计，要单独立项、单独量。6.5 倍不是常数（= 场景深度中位 / WiLoR 手腕深度中位），
换视频就变，所以 `global-scale` 出来的**绝对尺寸整段不可信**，只有形状和相对变化可信。

### 3.3 四宫格验证视频 —— 都在哪

四宫格是这个工程的标准验收形式（"指标 ≠ 画面"那条规矩的落地）。**规律是按"谁出的片"分**：

| 位置 | 哪个环节 / 哪次验收 |
|---|---|
| **`outputs/viz/<主题>.mp4`** | **给人看的结论片**（不是调试）。目前：`wilor_depth_modes.mp4` |
| `outputs/dev/compare_grid/fill_jar_grid_h264.mp4` | ⑤碰撞迁移：源 / 不开碰撞 / 新代码 / 旧代码 |
| `outputs/dev/compare_grid_retarget/fill_jar_grid_h264.mp4` | ④重定向迁移的同一组对比 |
| `outputs/dev/fill_jar/robot_sim_axes_h264.mp4` | `hand_frame` 轴向验收（带坐标轴叠加） |
| `outputs/migration_check/fill_jar_migration_quad.mp4` | 碰撞迁移期那次四宫格 |
| `outputs/legacy_runs/examples/_compare/{fill_jar,serve_cake}_badframe_quad.mp4` | 坏帧兜底机制的前后对比 |
| `outputs/legacy_runs/runs/m7/validation/fill_jar/robot_sim_axes_h264.mp4` | M7 资产迁移后的逐帧 hand_frame 验收 |
| `outputs/retarget/<片段名>/robot_sim.mp4` + `input_viz.mp4` | ④⑤每次正式跑的产物（不是四宫格，是单画面） |

**约定（往后请照这个放）**：

- 想让别人看结论的片 → `outputs/viz/<主题>.mp4`，一个主题一个文件，别套目录。
- 只为自己排查的片 → `outputs/dev/<run 名>/`，由 `scripts/dev/_devcli.py` 自动落位。
- 视频**一律 h264 / yuv420p**，否则 VSCode 里放不出来（mpeg4 踩过）。
- `outputs/migration_check/` 是迁移期的历史目录，**已封存不再往里写**。

### 3.4 其它需要记住位置的

| 路径 | 是什么 |
|---|---|
| [`../README.md`](../README.md) | 项目介绍：做什么、有哪些环节、每个环节用什么技术、怎么启动。**给新人的第一份** |
| [`CONVENTIONS.md`](CONVENTIONS.md) | 9 条必须遵守的工程规矩 + 每条由哪个测试钉着。**动手写代码之前看** |
| [`VERIFICATION.md`](VERIFICATION.md) | 一个模块一套验收判据 + 迁移的五步方法论。**改完之后看** |
| [`PITFALLS.md`](PITFALLS.md) | 18 个踩过的坑，现象 → 真因 → 怎么防。**报错方向不对时看** |
| [`external/patches/README.md`](../external/patches/README.md) | 我们对上游改了什么、为什么，以及每次迁移的处置记录。**动上游之前必读** |
| `external/patches/egoinfinity-modified.patch` | 唯一一份上游 diff（428 insertions，逐次增长的明细在 `patches/README.md`）。**它变小是迁移做对了，变大就是有人往上游写逻辑** |
| `outputs/legacy_runs/MANIFEST.tsv` | 从 `external/` 搬回来的 316 MB 存量的逐文件清单（保持原相对路径，没重命名） |
| `data/webvid/README.md` + `raw/MANIFEST.md5` | 7 段手工挑的原片是什么、`md5sum -c` 怎么复核。**注意：这批是挑过的，不能当质检评测集**（选择偏差正好抵消掉质检要测的东西） |
| `tests/regression/` | 质检的回归基准：3 段片 + 期望判决 + contact sheet |
| `scripts/dev/audit_mujoco_contacts.py` | 用官方 MuJoCo mesh contacts 独立复核我方碰撞代理（只报告不改轨迹）。基线数字和命令在 [`VERIFICATION.md` 的⑤小节](VERIFICATION.md) |
| `scripts/dev/collcmp_table.py` | 根位姿两条路线的**画面级**对比表（穿躯帧数 / 最深穿透 / 臂展利用率 ρ̄），吃 `run_collcmp.sh` 的产物，落 `outputs/dev/collcmp_table/`。`ik_rate` 单独看会把"穿躯换来的高可行率"记成进步，这张表就是钉这一点的。`--proxy` 决定漏/误两列用哪把尺子（每条路线自己标定的盒子 / 类默认盒），口径连同 `torso_half` 一起写进 `results.json` |
| `scripts/dev/sweep_arm_torso_params.py` | 臂-躯代理盒的**标定**：拿 MuJoCo 真实网格 contacts 当真值，phase1 纯几何穷举盒半长（秒级）、phase2 真跑过滤器扫门槛（分钟级）。素材必须是**没开碰撞过滤**的跑（用过滤后的产物标定是循环论证），落 `outputs/dev/collcal/`。结论进 [`configs/robots/m7.yaml`](../configs/robots/m7.yaml) 的 `collision.arm_torso.routes.*`（标 `verified: true`），[`collision/presets.py`](../src/web2robot/collision/presets.py) 只是把它读出来 |
| `scripts/dev/run_collcal_ab.sh` + `check_neural_bytes.sh` | 标定的验收：13 段 grid 路线重跑一遍出前后对照；neural 路线跑两遍比 md5，钉"另一条路线一个字节都没动" |
| `scripts/dev/make_readme_assets.py` | 生成 README 里那两张图（碰撞修复前后对照 / 输入-输出并排 GIF），落 [`docs/assets/`](assets/)。**全工程唯一一个产物不落 `outputs/` 的脚本** —— README 的图必须进 git，不然别人 clone 下来是一片红叉；命令和当前那两张图的来源 run 记在 [`VERIFICATION.md`](VERIFICATION.md) 里 |
| [`docs/BACKLOG.md`](BACKLOG.md) | **被打断的活 + 欠账清单**。新消息是打断不是排队，做到一半被切走的事当场记这里，每条带"怎么续"（状态在哪个目录、下一步是哪个文件）。做完就删 |
| [`docs/PRIORITY_2026-08-07.md`](PRIORITY_2026-08-07.md) | 当前优先级：质检/路由暂停自研，重定向第一 |
| [`docs/TODO22_FRONTEND_CONSOLE.md`](TODO22_FRONTEND_CONSOLE.md) | 前端控制台的设计要求 |
| [`docs/SYNC_2026-08-07.md`](SYNC_2026-08-07.md) | 阶段性同步记录 |
| [`docs/VIDEO_SELECTION_GUIDE.md`](VIDEO_SELECTION_GUIDE.md) | **"什么样的视频能用"的唯一真源**：准入判据 §V1–§V4（连续性 / 人体位置 / 人体朝向 / 切镜拆段）+ 模型分配 §0.1 + HaWoR 硬性前提 §0.0。**编号是接口** —— 质检/路由的代码注释直接引用 `§V1`、`§0.1`、`B3` 这些号，改判据能 grep 出所有该跟着改的地方，所以别重排编号 |
| [`docs/RELATED_WORK_2026-08-11.md`](RELATED_WORK_2026-08-11.md) | 四篇相关工作逐条对照（Ego2Robot / Phantom / Do as I Do / HandUMI）+ ICLR 定位结论 + 由此产生的待办。**写论文和定优先级时看** |
| [`docs/LEROBOT_ALIGNMENT_GAP.md`](LEROBOT_ALIGNMENT_GAP.md) | 现有 `trajectory.npz` 对齐到 LeRobot v3.0 的**差距分析**（2026-08-22，任务A）。目标格式逐字段实测、我们缺什么/多什么、要动的代码范围。**只是分析，没有转换代码** —— 参考数据集在 `/mnt/vlm/common/datasets/ABC-130k_lerobot_v30_repair_filter_qf094`（不在仓库里），复现命令附在文末 |
| `envs/requirements-{rt,hawor,perception}.txt` | 三个环境的精确版本。**共享机器，不要 pip install** |

---

## 4. 产物落点的规律：按"谁写的"分

`outputs/` 不是随手建目录 —— 每个写入口都有固定落点，看到路径就知道是谁跑的：

| 写入口 | 落点 |
|---|---|
| `scripts/s0_fetch_rgb.sh` | `outputs/fetch/<片段名>/`（`rgb.mp4` + `frames_index.json` + `align_report.json`）+ `outputs/fetch/_sources/` 源视频缓存（同一支视频只下一次，别手删） |
| `scripts/s3_to_clip.sh` | `outputs/clips/<片段名>/`（3~4 个 clip 契约文件） |
| `scripts/s4_retarget.sh` | `outputs/retarget/<片段名>/`（顶掉上游"写在素材旁边"的默认值） |
| `scripts/dev/_devcli.py`（7 个开发期脚本共用：出片 6 个 + 碰撞审计 1 个） | `outputs/dev/<run 名>/` |
| `scripts/dev/render_compare_grid.py` | 自带 `--out`，习惯落 `outputs/dev/compare_grid*/` |
| `scripts/dev/sweep_arm_torso_params.py` | `outputs/dev/collcal/`：`prefilter/<短名>/` 是**没开碰撞过滤**的标定素材，`phase1.json` / `phase2.json` 是两阶段的结果 |
| `scripts/dev/run_collcal_ab.sh` | `outputs/retarget/collcmp_cal/<短名>_grid/`（`_neural` 是软链到 `collcmp/` 的旧跑，因为那条路线按构造没变） |
| `python -m web2robot.twin`（物体位姿单跑） | `outputs/twin/<片段名>/`（`object_poses.npz` + `.json` + `--viz` 时的 `object_viz.mp4`）。走 `test.py --object_tracking on` 时不落这里，`object_poses.npz` 直接落那次重定向的 `--out` 目录，和 `root_frames.npz` 同级同命名 |
| `test.py --action_refine`（动作精修判决） | 落那次重定向自己的 `--out` 目录：`action_refine.json` / `action_refine.npz` / `hand_poses.npz`。**不新建顶层目录** —— 判决只对那一次 run 有意义，和它的轨迹放一起才对得上。`python -m web2robot.refine --run <目录>` 事后重判默认写回同一个目录，`--out` 可另指 |
| `scripts/s5_hand_mask.sh` | `outputs/synth/<片段名>/`（`handmask_check.png` 核对图 + `hand_masks.npz` 按位打包的左右手掩码）+ `outputs/synth/handmask.jsonl`（逐段判据，一行一段） |
| 人工封存 | `outputs/archive/<主题>_<年-月>/` |

六个写入口都过 `P.check_output_dir()` 这道闸，落点在 `external/` 里就直接 `SystemExit`。
（⓿ 和 ⑥ 这两档不经过那道闸 —— 它们只往 `--out` 写，默认值就在 `outputs/` 下。）

---

## 5. 怎么防这份文档过期

文档会烂，所以它有测试钉着：`tests/test_docs_layout.py`

- **新建一个顶层目录但没在本文说明 → 测试变红**（这是最容易烂的地方）。
- 本文提到的、**应该进 git 的**路径（`src/` `evidence/` `configs/` `docs/` 之类）
  必须真实存在。
- `outputs/` `data/` 下的路径不做存在性断言 —— 它们不进 git，新克隆本来就没有；
  但它们的**父目录约定**要在本文 §4 的表里出现。

```bash
envs/rt_env/bin/python -m unittest tests.test_docs_layout -v
```
