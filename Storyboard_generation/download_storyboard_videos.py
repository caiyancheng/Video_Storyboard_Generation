"""
download_storyboard_videos.py  (本地运行)

读取服务器生成的 generation_results.csv（从服务器拷贝到本地后放这里），
将生成视频下载到本地目录。

参考 Video_Generation/download_generated_videos.py 的方式。

本地输出：
  LOCAL_OUT_DIR/
    {vid_label}.mp4

用法：
  python download_storyboard_videos.py
"""

import csv
import time
import urllib.request
from pathlib import Path

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

# 将服务器上的 generation_results.csv 拷贝到本地后，填写路径
CSV_PATH = Path(
    "/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/generated_storyboard/generation_results.csv"
)

# 本地视频保存目录
LOCAL_OUT_DIR = Path(
    "/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/generated_storyboard/videos"
)

SKIP_EXISTING = True


# ═══════════════════════ DOWNLOAD ════════════════════════════════════════════

def download_all():
    if not CSV_PATH.exists():
        print(f"CSV 不存在: {CSV_PATH}")
        print("请先将服务器上的 generation_results.csv 拷贝到本地对应路径。")
        return

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"CSV 共 {len(rows)} 条记录")
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0

    for row in rows:
        vid_label = row["vid_label"]
        video_url = row.get("video_url", "").strip()
        score     = row.get("score", "?")
        duration  = row.get("duration", "?")

        if not video_url:
            print(f"[SKIP] {vid_label} — 无 video_url")
            fail += 1
            continue

        out_path = LOCAL_OUT_DIR / f"{vid_label}.mp4"

        if SKIP_EXISTING and out_path.exists():
            print(f"[SKIP] {out_path.name}")
            skip += 1
            continue

        print(f"[↓] score={score}  dur={duration}s  {vid_label}")
        print(f"    {video_url}")

        try:
            urllib.request.urlretrieve(video_url, str(out_path))
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"    ✓ saved ({size_mb:.1f} MB) → {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"    ✗ failed: {e}")
            fail += 1
            time.sleep(1)

    print(f"\n完成：下载 {ok} 个，跳过 {skip} 个，失败 {fail} 个")
    print(f"保存路径：{LOCAL_OUT_DIR}")


if __name__ == "__main__":
    download_all()
