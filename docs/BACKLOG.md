# 待办 / 被打断的活

**这份文档存在的理由**：新消息是**打断**，不是排队。Claude 没有跨会话自动维护的待办
列表，一件事做到一半被切走，除非当场写下来，否则就丢了 —— 而且几周后连"丢了什么"
都问不出来。所以规矩很简单：

> **被打断时，先往这里补一行，再去做新的那件。**

每条必须带**怎么续**（状态存在哪个目录 / 下一步是哪个文件哪个函数），不能只写标题。
只写标题的待办等于没写：过两周看到"碰撞校准 未完成"，还是得从头把上下文读回来。

做完的条目**删掉**，不要留一片 ~~划掉~~ ——真正做完的东西该在 README / 代码 /
`docs/VERIFICATION.md` 里有落点，这里只留"还欠着的"。

---

## A. 被打断，随时能续

### A1. grid 路线的碰撞过滤参数重校准（**验收过了一半**，2026-08-21）

- **要什么**：`--root_solver grid` 走的是网格搜索根位姿，目标函数不看身体，手臂贴身
  穿模比 neural 多（13 段实测 28.9% vs 23.8%，都是碰撞过滤**跑完之后**的残留）。
  只重新校准代理几何的膨胀/安全余量，**不改过滤器的核心逻辑**。
- **硬约束**：`neural` 那条路线的参数**一个都不许动**，两条路线的参数要能分开配置，
  不能绑成一套。做法是 `src/web2robot/collision/presets.py` 里 `neural` 留空 →
  今天的行为逐位不变。（已验：13 段的 neural 三列和旧表逐位相同 + 字节比对三个 SAME）
- **已做完**：`scripts/dev/sweep_arm_torso_params.py`（两阶段标定）→ 结论落
  `src/web2robot/collision/presets.py`（`neural` 空、`grid` = 盒 `[0.0695,0.119,0.239]`
  + `enter_thresh 0.02` + `margin 0.02`）→ `test.py` 接上 `--atf_preset/--atf_*` 覆盖 →
  `tests/test_module_boundaries.py::TestArmTorsoPresets` → 13 段 A/B 跑完出表
  （`outputs/dev/collcal_ab_table/`，逐段数字和抽帧都在 `docs/VERIFICATION.md`）。
- **验收结论（两条判据，一过一不过）**：
  - ✅ 穿模帧占比（真实网格判据）：13 段 **28.9% → 13.3%**，留出的 10 段 37.4% → 17.3%，
    12/13 段有残留 → 9/13，**没有一段变差**，ik 可行率一位没变。这条是**要什么**里
    真正在意的那条，泛化了。
  - ❌ 代理判据 vs 网格判据的帧数差：只在标定用的 3 段上收窄（226 → 24），留出的 10 段
    180 → **198**，而且方向翻面（误报 423 → 0，漏报 17 → **222**）。
- **为什么不过，以及下一步该动哪**（这是本条还留着的唯一原因）：不是参数没调好，是
  **代理形状到顶了** —— 躯干真身是圆的，轴对齐盒要把角上的误报压到 0 就得把 x 半长压到
  真身的 0.50 倍，于是平面方向欠覆盖，~1.7 cm 以内的真穿透对代理隐形。
  **下一步：把"检测"和"推出目标"解耦** —— 判是否触发用接近真身尺寸的盒
  （`presets.MESH_HALF`），推出目标仍用标定盒。落点是
  `src/web2robot/collision/arm_torso_filter.py`（`_sdf` 现在一个盒兼两职，
  给 `M7CapsuleModel` 加第二个 `torso_half` 或给过滤器加 `detect_half`），
  改完拿 `sweep_arm_torso_params.py phase2` 在同一批 prefilter 素材上重扫一遍，
  再跑一遍 `run_collcal_ab.sh` 看留出 10 段的漏报有没有下来。
  素材还在：`outputs/dev/collcal/prefilter/<短名>`（不带过滤的原始 q，换参数不必重跑 IK）。
- **另一件已经查清、别再当成同一个病的事**：残留里**深**的那些不是漏检 ——
  `--oo8_XIuOM_900.3_917.4` 最坏几帧代理读数 −1.92 ~ −4.70 cm，代理报了，是过滤器
  没修得动（那段 ik 只有 84.8%，源头坏帧）。抽帧确认它**肉眼可见地坏**（左小臂埋进躯干、
  指尖从胸口另一侧戳出来）；而 1.6 cm 那档看不出来。要治它得从坏帧兜底/源头感知那边走。


### A2. L3.4（rel3_4）接入（**第一阶段已完成**，2026-08-20；只剩第二阶段的欠账）

- **要什么**：第二台机器人和 M7 并列可切换（`--robot m7|l3_4`），**只做上肢**（双臂 7×2
  + 双手 12×2），腰/颈/腿锁死在 URDF 默认值（是锁死不是删除，以后要开随时开）。
  不动任何 M7 的现有文件，零上游 import，不反向依赖 `robots/m7/`。
- **已量到的关键事实**（决定了工作量，别再重新推一遍）：`assets/robots/urdf.tar.gz` 里的
  `l3.4.xml` 与我方 `m7.xml` **上肢完全同构** —— 43 个同名关节的 axis/range 全同
  （只有 `neck_pitch` 上限 0.54 vs 我方 0.48）、43 个同名 body 的 pos/quat 全同。
  手是**同一只 12 自由度手**（`thumb_bend + thumb_rota1/2`、其余四指 `bend/joint1/joint2`），
  不是 11 自由度的另一款；"xhand" 只是 URDF mesh 路径里的目录名。所以 MANO→手的映射
  表可以原样复用，不需要重新设计。L3.4 = M7 上身 + 12 个腿关节 + `base_link`。
- **mesh 那件事已经绕过（不是解决）**：包里一个 mesh 都没有。94 个零件和 M7 逐位相同
  （mass/inertia/COM 全同，是同一批零件）→ 建成**相对 symlink**（不是拷 19 MB，而且
  一个文件一个链接，以后拿到真的 L3.4 STL 换掉单个链接就行）；14 个腿部 + 盆骨
  `base_link` 没有正确的 mesh（M7 那个同名 `base_link.STL` 是升降柱底座，另一个零件），
  它们的 `visual`/`collision` 被删掉 → **渲出来腰以下是空的**。上肢的每个数字都不受影响
  （IK 链根和碰撞代理盒都挂 `waist_pitch_link`），但**出片之前得把腿的 mesh 要到**，见 D 节。
- **已做完**：`scripts/dev/build_l3_4_assets.py`（从原包生成整个 `assets/robots/l3_4/`，
  七步自检，含"对厂家 `l3.4.xml` 交叉校验"和"双臂链 vs `m7_mjx.xml` 逐位比对"两道）→
  `src/web2robot/robots/l3_4/` 五个模块（腰/颈/腿 17 个自由度锁在 `LOCKED_JOINTS`，
  `env._apply_locked()` 在每次 `mj_forward` 前按住）→ 上游三处注册（`sim/robots/__init__.py`、
  `RobotIKConfig.l3_4`、`_l3_4_12dof_from_keypoints`）+ `--robot` choices →
  `tests/test_l3_4_robot.py` 12 例、全量 301 全绿 → patch 重导 520 insertions（replay 6/6）
  → `PROJECT_LAYOUT.md` / `VERIFICATION.md` 已写。
- **状态存在哪**：资产 `assets/robots/l3_4/`（可 `--force` 重建）；端到端跑
  `outputs/retarget/l3_4_<片段>`，日志 `outputs/dev/l34_<片段>.log`。
- **第一阶段验收已过（2026-08-20 21:50）**：3 段官方片段端到端跑通（`fill_jar` /
  `serve_cake` / `sip_coffee`），IK 可行率 97.0% / 100% / 100%，`ArmTorsoFilter` 照旧开火
  （`fill_jar` 右臂 164/164 修净、`serve_cake` 左臂 187/188 剩 1、`sip_coffee` 右臂 2/2），
  三段的 `robot_sim.mp4` 逐帧看过，姿态靠谱（h264 版和抽帧在
  `outputs/dev/l3_4_stage1/`）。M7 逐字节不变已验：
  `scripts/dev/check_m7_unchanged_by_l3_4.sh` 三个产物全 `SAME`。
- **还欠着的（第二阶段，不阻塞主线）**：① 腿部 mesh（见 D 节）—— 补上之前 demo 不能用这台；
  ② `--root_solver neural` 那条路线在 L3.4 上没跑过（借的是 M7 的 ckpt，依据和失效条件见
  `VERIFICATION.md` 的 L3.4 一节；真要给 L3.4 单独训根模型时
  `robots/l3_4/sample_config.py` 已经备好但**没被跑过**）；
  ③ 解锁腰/腿要重训根模型（`LOCKED_JOINTS` 删一行就解锁一个自由度）。
  命令形式：`--robot l3_4 --root_solver grid --ckpt runs/m7/taskspace_v2/checkpoints/final.pt`
  （grid 压根不用模型，但上游 `test.py` 无条件 `_load_model`）。

## B. 等人拍板，我不该自己决定

（B2「默认 `--root_solver` 选哪条」2026-08-21 拍了 `grid`，已落地 —— 上游 argparse
默认值 + patch 重导 + README ④ + `docs/VERIFICATION.md` 一节。编号不复用。
注意这里的 B 编号和碰撞过滤那套 B0–B4 是两套东西，别串。）

下面九条（B3–B11）是 **2026-08-21 方向调整**（从"打磨 demo"转向"批量产出 LeRobot v3.0
数据集"）交下来的四个任务在实现过程中撞出来的矛盾。按用户自己的规矩「发现新依赖或矛盾先
记录同步，不要自己假设一个答案接着往下做」记在这里，**当时没有一条是我自己拍的**。

> **2026-08-22：九条全部拍板，这一节的待决队列现在是空的。** 下表只留"当时撞到什么 +
> 用户定了什么 + 落到哪"。B4/B5/B9/B10/B11 的实测明细在
> [`LEROBOT_ALIGNMENT_GAP.md`](LEROBOT_ALIGNMENT_GAP.md) §6，搁置的三条（B6/B7/B8）
> 转成 C 节的欠账。编号退役，不复用。
>
> **拍板时用户重申的唯一目标**（比任何单条决定都重要）：现在不是要产出大规模数据集，
> 也不是要把格式做到跟公司标准完全吻合；唯一要做的是把**"原始视频 → 视觉合成 →
> 产出带画面的数据"这条链路，在 M7 一台机器人、少量官方示例视频上完整跑通**。
> 格式细节允许临时占位，**不要因为格式不确定就卡住不推进**。
>
> **2026-08-23 追加 B12（新的待决项，队列又不空了）**：按 B3 定的路线去下载原始视频，
> 撞上 YouTube 的 PO Token 门槛 —— 四个可选方案里有两个涉及新依赖/账号，不是我该自己
> 决定的，明细见下表和 §B12。
>
> **2026-08-23 追加 B13（记录项，不用你拍板）**：做手部掩码时量出官方 3D 手和官方
> 2D 关节**逐帧对不上**，已经按"逐帧拟合"绕过去了（掩码本身没问题）；但有 1 段官方
> 片段的手势本身 3D/2D 不一致，绕不过去。按规矩记下来不自行处理，明细见 §B13。

| # | 当时撞到什么 | 用户 2026-08-22 定的 | 落到哪 / 还欠什么 |
|---|---|---|---|
| B3 | 15 段官方片段里**一帧 RGB 都没有**（只有 `depth.mp4` / `mask.mp4` / `hand_joints.bin` / `object_pose.bin`），视觉合成没有输入 | **用片段文件名里的 YouTube 视频 ID + 起止秒数，直接下载原始视频截取对应片段。** 这不是我们额外发明的流程 —— EgoInfinity 官方 pipeline 本身的标准输入就是"从视频抽出的 RGB 帧"。先走这条路，不用绕道问张勃要内部 exo 批次的位置 | **当前主线**：实现"ID+起止时间→下载→截取→抽帧"，并在 15 段官方片段上验证下载的 RGB 与 `hand_joints.bin` 等在**时间轴上对得上**（`depth.mp4`/`mask.mp4` 帧数正好等于 `n_frames`，是现成的对齐真值） |
| B4 | 视觉合成其实在格式对齐的关键路径上，不是可并行支线 | **确认：最终发布的数据集必须包含画面。** 所以视觉合成是必须的一环、优先级高，不是可选项 | 排期串成 B3 → 视觉合成 → 带画面的导出，不再当两条并行线 |
| B5 | 38 维 vs 参考的 `float32[14]`、fps 15.4 vs 30 —— 改哪边都是改数据本身 | **临时占位方案，先跟通链路**：`robot_type` 用真实维度的临时名（如 `m7_bimanual_dex`）、`action`/`observation.state` 直接写 38 维、字段名用现有 `*_joint_names`（已是全称，不改）、fps 写名义值。**整套明确标注为临时占位**，等张勃正式格式规范文档到了再调 | [`LEROBOT_ALIGNMENT_GAP.md`](LEROBOT_ALIGNMENT_GAP.md) §6；导出模块本身还没写 |
| B6 | 任务C（MPC）缺前向模型，"误差降到阈值以内"会自指 | **继续暂停，任务C 整体搁置**；这期间**不要做"运动学层面的伪 MPC"这类自证方案** | 转 C1（那条加了 2026-08-22 的注） |
| B7 | "G1 已接入完成"和仓库现状不符，两种理解的工作量差一个数量级 | **G1 完全搁置，不投入任何精力** —— 包括我提的"先用官方 G1 + 官方 ckpt 批量出数据、不做我们的碰撞过滤"这个折中方案**也不要做**。现阶段只做 M7 一条线 | 转 C23 |
| B8 | 批量的"现有 exo 视频"在哪；规模差两个数量级就是两套写法 | **现在不做批量转换**，不需要去确认公司内部 exo 语料库的位置 | 转 C24 |
| B9 | fps 不只是"不是 30"，是逐段都不一样（15.0000 / 15.0468 / … / **18.4041**） | **方案③**：`info.json` 写一个名义 fps，每段真实 fps 进 episodes parquet 的自定义列（先例是参考自己的 `dense_subtask_*`）。**不做重采样、不按 fps 分 shard** | [`LEROBOT_ALIGNMENT_GAP.md`](LEROBOT_ALIGNMENT_GAP.md) §6 |
| B10 | 三个 env 都没装 `pyarrow`，而规矩是"共享机器不要 pip install" | **批准安装**，加进 `envs/requirements-rt.txt`，并**破例授权我自己执行这次安装** | **已做完**：`pyarrow==25.0.1` 装入 `rt_env`（`pip install --no-deps`，freeze 前后 diff 只多这一行，numpy 仍 2.2.6），requirements 已加，365 个测试全绿；`external/patches/_pre_migration_snapshot/README.md` 那条 8/06 的 md5 加了脚注（`e8f9e2f7…` → `841a67ed…`，历史存档值不改） |
| B11 | 我们产的 mp4 是 mpeg4，参考要 h264，还违反我方约定 §3；换编码器会让 `robot_sim.mp4 = 205d96db…` 这条基线失效 | **现在不改**：只有真正要打包发布的数据才用 `libx264` 转 h264；现有调试产物（`robot_sim.mp4` 等）保持 mpeg4 原样，`docs/VERIFICATION.md` 里已建立的验收基准一条都不动 | 转码放在导出模块自己做；上游 `retarget/utils/viz.py::write_video` 不碰 |
| B12 | **按 B3 走下载这条路，字节拿不到**：不带 GVS PO Token，11 个 yt-dlp client 全失败（403 / "format not available" / 同一个 145471 字节的残件：容器声称 278 s、8331 帧，实际解码 0 帧） | **待拍板**（2026-08-23 提出） | 见下方 §B12：四个方案，两个需要你点头。截取+对齐验收那一半已经写完并用合成素材测通，`--backend local` 一开、字节一到位就能跑 |

| B13 | 官方 3D 手投影和官方 `hand_joints_2d.bin` 差 9.3 px 中位、逐帧缩放在 0.32–0.95 漂；其中 `-20k07PjLTA_48.0_52.4` 一段是**手势本身 3D/2D 不一致**（残差 7.04 px，全仿射也只降到 8.52 px），逐帧拟合绕不过去 | — （记录项，2026-08-23） | 见下方 §B13。掩码链路已按"逐帧拟合"实现并跑通 10 段；那一段的**要不要丢掉**是数据取舍，我不自己定 |

### B12. 原始视频下载被 PO Token 卡住（2026-08-23，待拍板）

B3 定的路线（YouTube ID + 起止秒数 → 下载原片 → 截取）里，**截取和验收这一半已经做完**
（`src/web2robot/fetch/`，27 个测试，逐帧核对内容），卡住的只有"拿到源视频字节"这一步。

实测（11 个 client 全试过）：

| client | 结果 |
|---|---|
| `android_vr`、`tv_embedded` | `HTTP Error 403: Forbidden` |
| `web`、`web_safari`、`ios`、`mweb` | "Requested format is not available"，或返回**同一个 145471 字节的残件** |
| `web_creator` | "Please sign in" |
| `tv` | "The page needs to be reloaded" |

那个残件的形状值得记住：`ffprobe` 报 278 s / 8331 帧（**元数据完整**），实际
`ffmpeg -ss 1 -frames:v 1` 报 `partial file`、解出 0 帧。所以**验收必须真解码**，
只看 ffprobe 会被骗过去（这条已经写成测试钉住了）。

试过但不解决的：装 `node v22` 当 JS runtime（`--js-runtimes node`）、
`--remote-components ejs:github`。根因是缺 GVS PO Token，不是格式表达式、不是 JS runtime。

四个方案：

| 方案 | 代价 / 需要你点头的地方 |
|---|---|
| ① 装 node 版 PO-token provider（bgutil） | npm registry 通（200），技术上可行。但这是**新的依赖类别** —— 运行时执行 Google 的 BotGuard JS，而且要往共享机器装 node 包。**要你批准** |
| ② 提供登录账号的 cookies | 要你给一份 cookies；且账号可能被风控 |
| ③ 从别处拿源视频（公司镜像/别的机器已下好的） | 本模块**已经支持**：`--backend local --source_dir …`，零改动 |
| ④ 不要真 RGB，继续往下做 | 视觉合成没有输入，等于放弃 B4 定的"发布数据必须带画面" |

**我的建议是 ③ → ①**：③ 零新依赖零风险，只要那 6 支视频在公司内网哪台机器上存在；
③ 不通再考虑 ①。**在你拍板前我不会装任何东西、不会引入 cookies。**

（顺带一条实测事实，与下载无关但影响正确性：片段**目录名里的秒数不可信** ——
`-2cNMO9Mm3Q_192.4_209.2` 的真实起点是 195.790，按目录名截会整段错开约 102 个源帧。
代码里只认 `scene.json` 的 `video_source`，目录名仅用于交叉核对。）

## C. 有意推后的欠账

按"值不值得现在做"排的，不是按重要性。

| # | 事 | 怎么续 / 卡在哪 |
|---|---|---|
| C1 | `refine/` 真正的修复算法 | 现在只做到**诊断判断**；Replay 实现了，MPC / RL 是占位，调用直接 `NotImplementedError`（不静默降级是故意的）。**2026-08-22：用户明确任务C 整体搁置（原 B6），而且这期间不要做"运动学层面的伪 MPC"这类自证方案** —— 那样的验收数字出自和目标同一个刚连假设，必然达标而画面里什么都没变。要续的前置还是 C2（物体网格 + 米制深度） |
| C2 | `twin/` 的 SAM2 + FoundationPose 后端 | 只有 `official` 那条能跑；卡在缺物体网格 + 缺米制深度 |
| C3 | `hand_conf.bin (T,2)` 加进 clip 契约 | 是 Phantom 遮挡关节合并的前置条件 |
| C4 | Ego2Robot 0.65 臂展项的 ablation | 目标函数改动，之前明确划在校准任务范围外 |
| C5 | `make_keyframe_scorer` 的候选循环向量化 | 纯性能 |
| C6 | `scripts/dev/audit_retarget_feasibility.py` | 还没写 |
| C7 | episode 级判决聚合器 | 现在判据都是逐帧的 |
| C8 | 重投影–分割 IoU 自检 | Ego2Robot 质检里值得抄的一条 |
| ~~C9~~ | ~~per-embodiment robot YAML~~ | **2026-08-21 做完**（编号退役不复用）：`configs/robots/{m7,l3_4}.yaml` + `robots/params.py`，照 HandUMI 的格式（一机一 yaml + `verified` 标志位），代码侧唯一来源由 `tests/test_robot_params_yaml.py` 守。搬迁中发现的数值疑点见 C18–C20 |
| C10 | VLM 语义一致性检查 | ①② 暂停期间一起搁着 |
| C11 | 视觉合成（新视角/渲染）那一摊 | 已归档，明确不占精力 |
| C12 | 前端控制台 | 见 [TODO22_FRONTEND_CONSOLE.md](TODO22_FRONTEND_CONSOLE.md) |
| C13 | github.io 页面 + demo 素材 | 目标已经改成 "repo + demo"，页面还没开工；`docs/assets/` 里的两张图是第一笔 |
| C14 | 质检/路由接 [`VIDEO_SELECTION_GUIDE.md`](VIDEO_SELECTION_GUIDE.md) 的 §V1–§V4 | 判据文档 2026-08-21 已重写并搬进本仓库，**代码还是旧认知**。接的时候三件具体事：① `quality/` 现在没有"画面变化是否连续"这个准入判据（`camera_motion` 是路由标签，不是准入，别拿它顶替 §V1）；② `pipeline.py` 的 `trim` 只裁到 `usable_span` 最长一段，§V4 要的是**按切点拆成多段全部保留**、每段各自判；③ 每个判据函数的注释要写 `依据 VIDEO_SELECTION_GUIDE.md §Vx`（编号是接口）—— 这是这次文档任务定的验收标准，本次**只改了文档、没动代码**。卡在①②暂停自研等对接 wangjufei |
| C15 | §V5"机器人抽搐 ⇔ 切镜"的定量复现 | 现在是**有机理支撑的观察，本仓库没有数字** —— 我们端到端跑的官方片段本身不含切镜。做法：找一段有切镜的原始视频跑完整条链，看 `root_frames.npz` 的位姿和 `trajectory.npz` 的关节角在切镜帧上的**一阶差分尖峰**位置和 ffmpeg 报的切点对不对得上。做完把数字写进 §V5，把"未定量复现"那句删掉 |
| C16 | 手部目标 lift 到世界系 + 在世界系里搜根位姿（解开 §V3 的朝向禁令） | 现在整条链的手部 IK 目标是**相机系**的（`utils/pose_utils.py::cam_to_root_targets` 算 `p_root = R_rootᵀ(p_hand_cam − t_root)`），而 grid 路线的躯干位姿是 `np.broadcast_to(_sol.R, (T,3,3))` —— **相机系里的一个常量**。后果：相机一转/一走，假的手部位移 1:1 注入，所以 §V3 只能写成无条件禁止转身转头。要真正支持"人转头/走位"的素材，得 ① 把手腕轨迹用相机位姿 lift 到世界系（HaWoR 那条路线本来就有世界系输出，只是 clip 契约没往下传，喂的是 `left_cam_np`）；② 网格搜索改在世界系里做，或者让根位姿逐帧跟随相机而不是被 `--torso_alpha` 往锚点压。**大改，动的是 upstream 接口，不在当前范围。** 做之前先做 C17 确认收益值不值 |
| C17 | 坐实 §V2 那四个数（1°→0.9 cm / 5°→4.4 cm / 30°→26 cm / 一步→60 cm） | 现在是拿 `cam_to_root_targets` 的公式和默认值（`--tol_pos 0.01`、`margin 0.02`、M7 实测 `r_max 1.007`）推出来的**几何推论**，不是端到端实测。做法：拿一段官方片段，人工往相机位姿上注入已知的旋转 θ / 平移 d，量重定向输出的手部末端位置偏移是不是跟着 `d·θ` 走，顺便看 ik_rate 和残余穿透从哪个角度开始塌。做完把 §V2 那句"几何推论，不是实测"换成实测数 |
| C18 | `verified: false` 那些数字里，真正"从没量过"的三处 | YAML 搬迁（C9）时逐个看过来的，**只记录、一个数都没改** —— 参数改动是单独一件要决策的事，改完还得重跑 `check_neural_bytes.sh`。① `collision.proxy.torso_half=[0.105, 0.135, 0.215]` 和 `tip_radius=0.012`：代理盒比躯干网格 AABB `[0.139, 0.170, 0.239]`（这个是量的，`verified: true`）三轴各收了 3.4/3.5/2.4 cm，**为什么收这么多没有依据**，是当初手挑的；② `ik.start_config` 的肩外展 ±0.20 rad 从没和别的静息姿态比过 ik_rate，就是个看着顺眼的种子；③ `collision.arm_torso.defaults` 那 11 个值里只有 grid 路线覆盖的 3 个（`torso_half`/`enter_thresh`/`margin`）被 sweep 标定过，剩下 8 个（`w_pen`/`w_ee`/`w_prox`/`fd_eps`/…）是默认值。要动的话：先扫一遍，再改 yaml，再重跑字节验证 |
| C19 | 新增两层坏帧粒度的三个阈值是**惯例，不是实测** | `trajectory/tiers.py` 里 `z_thresh=3.5`（Iglewicz–Hoaglin 论文的建议值）、`frac_thresh=0.05`（"5% 帧离群才算整段有问题"）、`seg_sec=2.0`（轨迹段长度）—— 都是拿约定值起的头，没在我们的素材上扫过。做法：拿 HF 那 106 段官方片段跑一遍，人工标"这段镜头是不是真的乱"，看这三个数在什么组合下和人工判断吻合。注意判据是**只警告/只标记**，所以误报的代价比漏报低，别照抄论文的剔除口径来定阈值 |
| C20 | episode 级只能做 clip **内部**的离群，跨语料的做不了 | EgoSmith 原文（arXiv 2607.09701 §3）是在整个语料上算相机平移分布再丢离群 episode；我们的 pipeline 一次只见一个 clip，所以 `episode_camera_check` 判的是"这段片子内部有没有几对帧的机位运动格外大"。真正的跨语料离群该在质检阶段做（C14 那一摊，`quality/` 已经有 `_camera_motion_score_flow` 的分数，缺的是把整批分数存下来再回头比）。同一条：原文那个"硬旋转阈值丢掉头部大幅转动的 episode"我们**没有对应物** —— clip 契约里没有逐帧相机位姿（`camera.json` 只有内参 + 重力），光流也分不开平移和旋转，所以警告文案只能把 §V2/§V3 一起引 |
| C21 | L3.4 一个碰撞参数都没标定过 | `configs/robots/l3_4.yaml` **刻意没有 `collision:` 一节**（`tests/test_robot_params_yaml.py::TestL34HasNoCollisionSection` 把这条"故意不写"钉住了，免得有人把一份 `verified: false` 的复制品读成"L3.4 也支持"）。现在那套过滤器是 M7 专用的：代理盒挂 `waist_pitch_link`、body 名写死 `left/right_hand_frame`。等真要支持 L3.4，加 yaml 那一节的同时必须连标定一起加 |
| C22 | `--quality_gate external` / `--routing external` 第三档 | 2026-08-21 明确**先不加**：现在没有对接对象，不知道公司那套质检输出什么格式的判决，先留一个名字会有人去实现它。开关的取值集合只写在 `src/web2robot/quality/config.py` 的 `GATE_MODES` / `ROUTING_MODES` 两个常量里，加档就改那一处（argparse 的 choices 和单测都引用它，`tests/test_quality_switch.py::test_no_external_mode_yet` 把"现在只有两档"钉住了，加档时会红，那是提醒不是故障）。接的时候要想清楚的是：`external` 读进来的判决要映射到 `Verdict` 的哪一档，以及它给不给 `suggested_route` |
| C23 | G1 接入（原 B7） | **2026-08-22 用户明确：G1 完全搁置，不投入任何精力**，连"先用 upstream 官方 G1 + 官方 ckpt 出数据、不做我们的碰撞过滤"这个折中也不做。现阶段只做 M7 一条线。要续的话素材是现成的：upstream `external/EgoInfinity/retarget/sim/robots/g1/`（config/env/sample_config）+ 官方权重 `/mnt/vlm/fanshaoheng/EgoInfinity/retarget/ckpts/g1.pt`；我们缺的是 MJCF、`hand_frame` 约定（M7 那次就是这里转错手掌，见 memory `m7-handframe-convention`）、碰撞覆盖和 `configs/robots/g1.yaml`。按 M7 的经验，真正花时间的是 hand_frame + 自碰撞标定，不是跑通 |
| C24 | 批量转换公司 exo 语料（原 B8） | **2026-08-22 用户明确：现在不做批量转换**，也不需要去确认内部语料库在哪。所以并行调度 / 断点续跑 / 按 shard 切分这些架构决定一并推后 —— 规模没定之前写哪套都是猜。现在的范围就是"少量官方示例片段"（`data/clips_official/` 15 段 + HF 那 106 段可扩） |

## D. 不是技术活，但会忘

- 把穿透 / ρ̄ 那个发现同步给白琦呈；找曹源江确认 Qwen-RobotManip 公式 (3) 的读法。
- 问魏庆功要内部 UMI 数据。
- **要 L3.4 的腿部 mesh**（14 个：`{left,right}_{hip_roll,hip_yaw,hip_pitch,knee,ankle_pitch,
  ankle_roll,foot_ee}_link.STL`）和盆骨 `base_link` 的真 STL —— 厂家那个 50 KB 的包里
  一个 mesh 都没有。现在渲出来腰以下是空的，**上肢数字不受影响，但 demo 出片之前必须补**。
  拿到之后：丢进 `assets/robots/l3_4/meshes/`（覆盖同名 symlink 即可），把
  `scripts/dev/build_l3_4_assets.py` 里 `NO_MESH_LINKS` 对应的行删掉，`--force` 重建。
- **再要一份 L3.4 的厂家原包 `urdf.tar.gz`** —— 生成资产那次用的原包现在**不在磁盘上了**
  （`assets/robots/urdf.tar.gz` 不存在，全盘也没有），所以
  `build_l3_4_assets.py` 现在跑不了（`P.asset("l3_4_src_tar")` 就会炸）。
  **已生成的资产和所有跑出来的数字都不受影响**，缺的只是"能重跑生成脚本"这件事。
  包里三个文件，两个已经留档在版本库里：`l3.4.xml` → `assets/robots/l3_4/l3_4_vendor.xml`、
  `l3_4.urdf.xacro` → 同目录同名，都是 `shutil.copy2` 的逐字节副本；
  **只缺 `l3_4.urdf` 的原件**（版本库里那份 `l3_4_from_urdf.urdf` 是加了两处改动之后的，
  header 里写了改了什么，理论上能反推回去，但不如直接再要一份）。
  拿到之后：丢到 `assets/robots/urdf.tar.gz`，`--force` 重建，产物应当和现在的逐字节相同
  （脚本七步自检会自己核对），顺手把它提交进去，下次就不会再丢。
- 请人 `chown fanshaoheng` memory 目录里那 6 个 root 所有的文件（现在改不动）。
- **GitLab 还欠一次 push**（2026-08-22 起，8/23 又试两次仍然不通）：`890d857` 及之前的
  17 个提交已经推到 `github`/`github2` 两个 remote 的 `main`，但 `origin`
  （`gitlab.robotera.com`）服务端在报 `Internal API unreachable`（GitLab Shell 连不上
  自己的内部 API，https 也是 SSL EOF）—— **不是权限问题也不是本地问题**，两天里重试五次
  都一样。等它恢复后补一条：`git push origin main:web2robot`
  （**落点是 `web2robot` 分支，不是 `main`**，本地 `main` 跟踪的就是 `origin/web2robot`）。

---

## 更早被打断的活，怎么捞

这份清单是 2026-08-20 从 memory 和当时的上下文里重建的，**只覆盖当时还记得的**。
更早的打断（比如 7 月那些）没有记录，但会话记录还在：

    ls /mnt/vlm/fanshaoheng/.claude/projects/-mnt-vlm-fanshaoheng/*.jsonl

要挖的话，找用户消息紧跟在我一串工具调用之后、且话题突然换掉的位置 —— 那就是打断点。
成本不低，除非真的怀疑漏了要紧的东西，否则不值得挖。

### B13. 官方 3D 手和官方 2D 关节对不上（2026-08-23，记录项）

做手部掩码（任务B 第一块）时量出来的。**不用拍板**，只是"发现了但没敢自行处理"的那类
事，按规矩记在这。

**现象**：拿 `camera.json` 把 `hand_joints.bin` 投到像素平面，和官方
`hand_joints_2d.bin` 比，差 **9.3 px 中位**（最差一段 71 px）。全段统一拟合一个
缩放+平移只降到 4.4–19.5 px；**逐帧逐手**拟合降到 3.7 px，解出来的缩放逐帧在
**0.32–0.95** 之间漂（10 段合起来 0.088–1.806）。

**判断**：官方 3D 手的尺度/深度逐帧不定（单目手部深度的老问题），`hand_joints_2d.bin`
才是像素空间里的真值。所以掩码流程做成「投影 → 逐帧逐手相似变换 → 光栅化」，
默认开对齐。这一段**已经自行处理了** —— 因为它自带判据（对齐后非指尖关节落点
0.765 → 0.964），不需要谁点头。

**绕不过去的那一段**：`-20k07PjLTA_48.0_52.4`。拟合残差 7.04 px（其余段 1.1–4.3），
非指尖落点只有 0.847。把模型从"缩放+平移"放宽到**全仿射**，残差只从 8.96 降到
8.52 px —— 说明剩下的误差不是我的 2D 模型自由度不够，而是**官方数据里这一段的 3D 手势
和 2D 手势本身不一致**。看核对图也印证：那段的 `depth.npz` 本身重建得很差（大片平坦的
黑块）。

**我没做的事，以及为什么**：

- 没给这段设"掩码质量门槛"把它自动踢掉。少量示例视频跑链路的阶段，10 段砍掉 1 段是
  可感知的取舍；而且门槛值定在哪（0.9？0.95？）会直接决定以后批量时丢多少数据，
  这是数据取舍不是实现细节。
- 没加旋转/仿射自由度去硬压残差。上面那个 8.96 → 8.52 的实测说明加了也没用，
  只会多一份复杂度和一份"看起来在改进"的假象。
- 没去查上游 EgoInfinity 是怎么生成这两份文件的（那要读它的重建代码，属于另一件事）。

**要动的时候动什么**：`joints_inside_fraction()` 已经把 `non_tip` 单独报出来，
`outputs/synth/handmask.jsonl` 逐段都有；真要设门槛，加一个 `--min_non_tip` 就够，
判据本身不用改。
