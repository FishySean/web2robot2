#!/usr/bin/env python3
"""把官方 HF 片段里**本地没同步下来的那几件**补齐（一次性，可重跑）。

`data/clips_official/<片段>/` 里现在只有重定向需要的四件套 + 掩码/物体那几样。
`scene.json` 自己声明了另外几个文件（`reconstruction.raw.*`），HF 上都有，本地没有：

| 文件 | 是什么 | 视觉合成为什么需要它 |
|---|---|---|
| `depth.npz` | uint16 **毫米**，(T,480,853) | 本地 `depth.mp4` 是 8 位归一化、**没存标定尺度**，没法和机器人渲出来的米制深度比大小。深度排序必须要这份 |
| `hand_verts.bin` + `hand_faces.bin` | MANO 手部网格 | 光栅化就是**精确的手部掩码**，不用分割模型 |
| `hand_joints_2d.bin` | (T,2,21,2) 像素坐标 | 对齐验收的第三条判据（本地靠 3D 投影代替，有了这个就是官方口径） |
| `bg_template.png` | 853×480 **uint16 深度**背景板 | 背景深度参考（注意它不是 RGB，别当画面用） |

**不下 `flow.mp4` / `recording.viser` / `retarget/`**：前两个这条链路用不上，
`retarget/` 是官方四台机器人的结果（要用时单独拉，别混进输入目录）。

放在片段目录里（和 `scene.json` 里写的相对文件名一致）。已确认上游
`external/EgoInfinity/` 全仓库**没有一处**引用这几个文件名，所以补进去不会改变
现有重定向行为 —— 这不是推测，是 grep 过的。

    envs/rt_env/bin/python scripts/dev/fetch_official_extras.py            # 全部 10 段
    envs/rt_env/bin/python scripts/dev/fetch_official_extras.py --dry_run  # 只看要下什么

HF 这条链路会零星 SSL 断流，所以每个文件 curl 自带重试 + 外面再套一层 python 重试
（`--retry-all-errors`，见 memory `official-egoinfinity-hf-clips`）。下完写
`data/clips_official/EXTRAS.md5`，重跑会跳过 md5 已对上的文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLIPS = REPO / "data" / "clips_official"
MANIFEST = CLIPS / "EXTRAS.md5"
REPO_ID = "Rice-RobotPI-Lab/egoinfinity"
BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/samples"
WANT = ["depth.npz", "hand_verts.bin", "hand_faces.bin", "hand_joints_2d.bin", "bg_template.png"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl(url: str, dest: Path, tries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = ""
    for attempt in range(1, tries + 1):
        proc = subprocess.run(
            ["curl", "-sL", "--fail", "--retry", "8", "--retry-delay", "2",
             "--retry-all-errors", "--max-time", "600", "-o", str(tmp), url],
            capture_output=True, text=True)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.rename(dest)
            return
        last = f"rc={proc.returncode} {proc.stderr.strip()[:120]}"
        print(f"    第 {attempt}/{tries} 次失败：{last}", file=sys.stderr)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"下不下来：{url}（{last}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="补齐官方 HF 片段里本地缺的几个文件")
    ap.add_argument("--clip", action="append", dest="clips", default=None, help="只补这些片段")
    ap.add_argument("--only", action="append", dest="files", default=None,
                    help=f"只补这些文件名（默认 {WANT}）")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    want = args.files or WANT
    clip_dirs = sorted(p for p in CLIPS.iterdir() if p.is_dir() and (p / "scene.json").exists())
    if args.clips:
        keep = set(args.clips)
        clip_dirs = [p for p in clip_dirs if p.name in keep]
    if not clip_dirs:
        print("没找到片段目录", file=sys.stderr)
        return 1

    have = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                digest, rel = line.split(maxsplit=1)
                have[rel.strip()] = digest

    todo, skipped = [], 0
    for d in clip_dirs:
        for name in want:
            rel = f"{d.name}/{name}"
            dest = d / name
            if dest.exists() and have.get(rel) == md5(dest):
                skipped += 1
                continue
            todo.append((d, name, dest))

    print(f"{len(clip_dirs)} 段片段 × {len(want)} 个文件：要下 {len(todo)} 个，已对上 md5 跳过 {skipped} 个")
    if args.dry_run:
        for d, name, _ in todo[:20]:
            print(f"  {d.name}/{name}")
        return 0

    failed = 0
    lines = dict(have)
    for i, (d, name, dest) in enumerate(todo, 1):
        url = f"{BASE}/{d.name}/{name}"
        print(f"[{i}/{len(todo)}] {d.name}/{name}")
        try:
            curl(url, dest)
        except RuntimeError as exc:
            failed += 1
            print(f"  跳过：{exc}", file=sys.stderr)
            continue
        lines[f"{d.name}/{name}"] = md5(dest)
        print(f"  {dest.stat().st_size/1024:.1f} KB  md5={lines[f'{d.name}/{name}'][:12]}")

    with open(MANIFEST, "w") as fh:
        fh.write("# 官方 HF 片段补下来的文件（scripts/dev/fetch_official_extras.py 写的）\n")
        fh.write(f"# 来源 https://huggingface.co/datasets/{REPO_ID} samples/<片段>/\n")
        for rel in sorted(lines):
            fh.write(f"{lines[rel]}  {rel}\n")
    print(f"\n清单 {MANIFEST}（{len(lines)} 条）；失败 {failed} 个")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
