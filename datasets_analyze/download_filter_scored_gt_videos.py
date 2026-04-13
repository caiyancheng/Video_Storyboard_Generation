"""
download_filter_scored_gt_videos.py  (本地运行)

读取 scored JSONL（评分 4/5 分），下载原始 GT 视频到本地。

输出结构：
  LOCAL_OUT_ROOT/
    score4/  id_0000_score4_xxx.mp4
    score5/  id_0003_score5_xxx.mp4
"""

import json
import time
import urllib.request
from pathlib import Path

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

SCORED_FILE = Path("/Users/bytedance/Datasets/tt_template_hq_publish_data_1400k_USAU"
                   ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl")

LOCAL_OUT_ROOT = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                      "/filter_scored_gt_videos")

TOS_BASE_URL = "https://tosv-va.tiktok-row.org/obj/nebudata-us/"

TARGET_SCORES: set[int] = {4, 5}

SKIP_EXISTING = True


# ═══════════════════════ DATA ════════════════════════════════════════════════

def load_scored() -> list[dict]:
    records = []
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_score") in TARGET_SCORES:
                records.append(rec)
    return records


# ═══════════════════════ DOWNLOAD ════════════════════════════════════════════

def download_all():
    records = load_scored()
    print(f"找到 {len(records)} 条评分 {sorted(TARGET_SCORES)} 分的记录\n")

    ok = skip = fail = 0

    for i, rec in enumerate(records):
        video_id = rec.get("video_id", "unknown")
        tos_key  = rec.get("tos_key", "")
        url      = rec.get("video_url") or (TOS_BASE_URL + tos_key if tos_key else None)
        score    = rec.get("_score", "?")
        dur      = rec.get("_duration") or rec.get("duration") or \
                   rec.get("result", {}).get("duration")
        dur_s    = f"{float(dur):.2f}s" if dur is not None else "N/A"

        if not url:
            print(f"[SKIP] {video_id} — 无有效 URL")
            fail += 1
            continue

        out_dir = LOCAL_OUT_ROOT / f"score{score}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"id_{i:04d}_score{score}_{video_id}.mp4"

        if SKIP_EXISTING and out_path.exists():
            print(f"[SKIP] {out_path.name}")
            skip += 1
            continue

        print(f"[↓] id={i:04d}  score={score}  dur={dur_s}  {video_id}")
        print(f"    {url}")

        try:
            urllib.request.urlretrieve(url, str(out_path))
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"    ✓ saved ({size_mb:.1f} MB) -> {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"    ✗ failed: {e}")
            fail += 1
            time.sleep(1)

    print(f"\n完成：下载 {ok} 个，跳过 {skip} 个，失败 {fail} 个")
    print(f"保存路径：{LOCAL_OUT_ROOT}")


if __name__ == "__main__":
    download_all()
