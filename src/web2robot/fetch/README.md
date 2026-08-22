# fetch — 流水线第 0 步：把官方片段的**原始 RGB 画面**还原出来

```
官方片段(scene.json 里的 YouTube ID + 起止秒数) → 【本模块】下载源视频 + 逐帧对应截取
    → rgb.mp4 + 对齐验收报告 → 第2步 视觉合成(抠人换机器人) → 带画面的数据集
```

## 为什么需要这一档

官方发布的片段里**一帧真 RGB 都没有**。逐个查过：

| 文件 | 实际是什么 |
|---|---|
| `depth.mp4` | 深度（灰度），不是画面 |
| `mask.mp4` | 掩码 |
| `thumb.jpg` | 320×180，**深度图的伪彩缩略图**（肉眼可确认，不是 RGB 帧） |
| `bg_template.png`（只在 HF 上） | 853×480 **uint16 深度**背景板（值域 0–3344） |
| HF 上另有 | `hand_joints_2d.bin`、`depth.npz`、`flow.mp4`、`recording.viser`、`retarget/{franka,g1,robonaut2,xlerobot}/` |

HF 仓库 4374 个文件里 **RGB 视频 0 个**。而"抠掉人、贴上渲染的机器人"这件事的输入必须是
RGB，发布的数据集也必须带画面。所以画面只能从原片还原 —— 这不是我们额外发明的流程：
EgoInfinity 官方 pipeline 本身的标准输入就是"从视频抽出的 RGB 帧"，`video_source` 字段
就是为此留的。

## 时间轴口径：只认 `video_source`，**不认目录名**

片段第 i 帧在源视频里的时刻是

```
t_i = video_source.start_seconds + i / fps        i = 0 … stats.n_frames−1
```

`fps` 是每段各自的、**非整数**的值（实测 15.0000–18.4041），源视频是 30 fps，
所以**不存在整数跨步**的取帧方式，只能逐帧算时刻再最近邻取。

目录名里的两个秒数是四舍五入/有时干脆是错的。实测 10 段：

| 片段 | 目录名起止 | `video_source` 起止 | 差 |
|---|---|---|---|
| `-2cNMO9Mm3Q_192.4_209.2` | 192.4 / 209.2 | **195.790 / 205.523** | **起 −3.39 s ≈ 102 源帧** |
| `-1r9yl-P-Ao_231.8_241.5` | 231.8 / 241.5 | 231.760 / **239.693** | 止 +1.81 s |
| 其余 8 段 | — | — | ≤ 0.04 s |

按目录名截 `-2cNMO9Mm3Q`，整段画面会错开三秒多 —— 帧数一个不差，内容全错。
所以代码里目录名**只用来交叉核对**（`ClipSource.name_gap_seconds`），一个计算都不参与。

## 取帧不 seek —— 故意的

`-ss` 放在 `-i` 前后行为不同、`-read_intervals` 又有自己的关键帧规则。为了让
"片段第 i 帧 = 源视频第 k 帧"是**事实而不是假设**，两趟都整支过：

1. `ffprobe` 整支扫一遍，拿每一帧的 `best_effort_timestamp_time` → `frame_times()`（带缓存）
2. `ffmpeg` 整支解一遍，边解边挑 → `stream_rgb()`（单调指针，O(1) 内存）

然后把**每一帧的时间误差**写进 `frames_index.json`。源视频 30 fps 时半帧 = 16.7 ms，
`max|dt| ≤ 16.7 ms`（`within_half_frame: true`）才叫取对了。

误差超过**一整个源帧**直接报错，不写盘（`max_dt_frames=1.0`）：越界的目标时刻会被夹到
端点，那等于"复制最后一帧把帧数凑够"—— 帧数看着对，内容是错的。宁可失败。

## 对齐验收：三条判据，都出数字

画面"看着像"不算验收，必须能量出偏了几帧。现成的真值有三样：

| 判据 | 钉住什么 | 通过条件 |
|---|---|---|
| 帧数 | 取的帧数对 | `depth.mp4` / `mask.mp4` 解码帧数 == `stats.n_frames` == RGB 帧数 |
| 运动能量互相关 | 时间上没整体平移 | 峰值落在 `lag == 0`（lag 约定：`b[i] ≈ a[i+L]`） |
| 2D 手部关节叠图 | 每一帧内容真是那一帧 | 关节落在画面内的比例 + **四宫格给人眼看** |

三条**不全有结论就写 `unknown`，绝不写 pass**（`align_report.json` 的 `verdict` 只有
`aligned` / `misaligned` / `unknown`）。第三条一定出图，因为指标 ≠ 画面可信。

## 怎么跑

```bash
# 只算账不碰网络：每段要下哪支视频、截哪一段、目录名差多少
scripts/s0_fetch_rgb.sh data/clips_official --out outputs/dev/fetch_dry --dry_run

# 真跑（10 段片段自动归并成 6 支源视频，同一支只下一次）
scripts/s0_fetch_rgb.sh data/clips_official --out outputs/fetch

# 视频已经在手上（推荐：本模块对"字节从哪来"完全不敏感）
scripts/s0_fetch_rgb.sh data/clips_official --backend local --source_dir /存视频的目录
```

片段目录名以 `-` 开头，`--clip` 必须写成 `--clip=--oo8_XIuOM_799.5_809.8`。
一段失败不拖累其他段：错误进 `fetch.jsonl` 的 `error` 字段，退出码非 0。

## 产物

```
outputs/fetch/
├── _sources/<youtube_id>.mp4        源视频缓存 + <id>.pts.json 时间戳缓存
├── fetch.jsonl                      每段一行：ok / error / max_abs_dt / align_verdict
└── <片段 id>/
    ├── rgb.mp4                      h264 / yuv420p / crf 12，缩放到 camera.json 尺寸
    ├── frames_index.json            逐帧 {i, t_target, source_frame, t_source, dt}
    ├── align_report.json            三条判据 + verdict
    └── align_montage.png            2D 关节叠图四宫格
```

## 目前卡在哪（2026-08-22 实测）

**下载这一半拿不到字节**：不带 GVS PO Token，11 个 yt-dlp client 全军覆没 ——
`android_vr` / `tv_embedded` 是 `HTTP 403`，`web` / `web_safari` / `ios` / `mweb` 要么
"Requested format is not available"、要么返回**同一个 145471 字节的残件**（容器声称
278 s / 8331 帧，实际解码 0 帧、报 `partial file`），`web_creator` 要求登录，
`tv` 要求重载页面。装了 `node v22` 当 JS runtime、试过 `--remote-components ejs:github`，
都不解决。

所以本模块**故意做成与来源无关**：`--backend local --source_dir …` 一开，字节一到位，
下游（截取 + 对齐验收）立刻能跑。截取和验收的正确性已由 `tests/test_fetch_rgb.py`
用合成素材钉住（逐帧核对内容，不只核对帧数）。

见 `docs/BACKLOG.md` B12。
