"""
download_generated_videos.py  (本地运行)

读取服务器生成的 generation_results.csv，
把每个视频按 prompt_level 下载到本地对应子文件夹。

本地输出结构：
  LOCAL_OUT_ROOT/
    level_1/  id_0000_score4_xxx_level1.mp4
    level_2/  ...
    level_3/  ...
    level_4/  ...
"""

import csv
import time
import urllib.request
from pathlib import Path

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

# CSV 文件路径（从服务器拷贝到本地后放这里，或直接挂载读取）
CSV_PATH = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                "/shu_inverse_label/generated_videos/generation_results.csv")

# 本地保存根目录
LOCAL_OUT_ROOT = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                      "/shu_inverse_label/generated_videos")

# 只下载指定 level，None 表示全部
FILTER_LEVELS: list[int] | None = None   # e.g. [1, 3] 或 None

# 已存在的文件是否跳过
SKIP_EXISTING = True


# ═══════════════════════ DOWNLOAD ════════════════════════════════════════════

def download_all():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"CSV 共 {len(rows)} 条记录")

    if FILTER_LEVELS is not None:
        rows = [r for r in rows if int(r["prompt_level"]) in FILTER_LEVELS]
        print(f"过滤后 {len(rows)} 条 (levels={FILTER_LEVELS})")

    ok = skip = fail = 0

    for row in rows:
        vid_label    = row["vid_label"]
        level        = int(row["prompt_level"])
        video_url    = row["video_url"]
        score        = row.get("score", "?")
        duration     = row.get("duration", "?")
        prompt_words = row.get("prompt_words", "?")

        out_dir  = LOCAL_OUT_ROOT / f"level_{level}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{vid_label}_plevel{level}.mp4"

        if SKIP_EXISTING and out_path.exists():
            print(f"[SKIP] {out_path.name}")
            skip += 1
            continue

        print(f"[↓] level={level}  score={score}  dur={duration}s  "
              f"words={prompt_words}  {vid_label}")
        print(f"    {video_url}")

        try:
            urllib.request.urlretrieve(video_url, str(out_path))
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"    ✓ saved ({size_mb:.1f} MB) -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"    ✗ failed: {e}")
            fail += 1
            time.sleep(1)

    print(f"\n完成：下载 {ok} 个，跳过 {skip} 个，失败 {fail} 个")
    print(f"保存路径：{LOCAL_OUT_ROOT}")


if __name__ == "__main__":
    download_all()
