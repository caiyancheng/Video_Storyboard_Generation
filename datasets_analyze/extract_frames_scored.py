"""
extract_frames_scored.py

读取 scored JSONL，找出评分 4/5 分的视频，
用 ffmpeg 提取第一帧和最后一帧，保存为 PNG。

输出目录：
  ~/Datasets/15s_video_sample/first_frame/{video_id}.png
  ~/Datasets/15s_video_sample/last_frame/{video_id}.png
"""

import json
import subprocess
from pathlib import Path

from tqdm import tqdm

SCORED_FILE  = Path("/Users/bytedance/Datasets/tt_template_hq_publish_data_1400k_USAU.dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl")
TOS_BASE_URL = "https://tosv-va.tiktok-row.org/obj/nebudata-us/"

OUT_ROOT    = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample")
FIRST_DIR   = OUT_ROOT / "first_frame"
LAST_DIR    = OUT_ROOT / "last_frame"

TARGET_SCORES = {4, 5}


# ========== IO ==========

def load_target_records() -> list[dict]:
    records = []
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_score") in TARGET_SCORES:
                records.append(rec)
    print(f"找到 {len(records)} 条评分 {sorted(TARGET_SCORES)} 分的记录")
    return records


# ========== ffmpeg 工具 ==========

def get_duration(rec: dict) -> float | None:
    """优先用记录自带的 _duration，否则用 ffprobe 探测"""
    dur = rec.get("_duration") or rec.get("duration") or rec.get("result", {}).get("duration")
    if dur is not None:
        try:
            return float(dur)
        except (TypeError, ValueError):
            pass

    url = rec.get("video_url") or (TOS_BASE_URL + rec.get("tos_key", ""))
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        d = data.get("format", {}).get("duration")
        return float(d) if d is not None else None
    except Exception:
        return None


def extract_frame(url: str, timestamp: float, out_path: Path) -> bool:
    """用 ffmpeg 从 url 提取指定时间戳的帧，保存为 PNG，成功返回 True"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", url,
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


# ========== 主流程 ==========

def extract_all():
    # 创建输出目录
    FIRST_DIR.mkdir(parents=True, exist_ok=True)
    LAST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：\n  {FIRST_DIR}\n  {LAST_DIR}\n")

    records = load_target_records()

    ok = 0
    skip = 0
    fail = 0

    for idx, rec in enumerate(tqdm(records, desc="提取帧")):
        vid   = rec.get("video_id", "unknown")
        score = rec.get("_score", "x")
        url   = rec.get("video_url") or (TOS_BASE_URL + rec.get("tos_key", ""))

        stem = f"id_{idx:04d}_score{score}_{vid}"
        first_out = FIRST_DIR / f"{stem}_first.png"
        last_out  = LAST_DIR  / f"{stem}_last.png"

        # 已存在则跳过
        if first_out.exists() and last_out.exists():
            skip += 1
            continue

        # 获取时长（用于定位最后一帧）
        duration = get_duration(rec)
        last_ts  = max(0.0, duration - 0.1) if duration else None

        score = rec.get("_score", "?")
        tqdm.write(f"[{score}分] {vid}  dur={duration}s  {url}")

        # 提取第一帧（ts=0）
        if not first_out.exists():
            success = extract_frame(url, 0.0, first_out)
            if not success:
                tqdm.write(f"  ✗ 第一帧提取失败")
                fail += 1
                continue

        # 提取最后一帧
        if not last_out.exists():
            if last_ts is None:
                tqdm.write(f"  ✗ 无法获取时长，跳过最后一帧")
                fail += 1
                continue
            success = extract_frame(url, last_ts, last_out)
            if not success:
                tqdm.write(f"  ✗ 最后一帧提取失败")
                fail += 1
                continue

        tqdm.write(f"  ✓ 保存成功")
        ok += 1

    print(f"\n完成：成功 {ok} 条，跳过(已存在) {skip} 条，失败 {fail} 条")
    print(f"first_frame → {FIRST_DIR}")
    print(f"last_frame  → {LAST_DIR}")


if __name__ == "__main__":
    extract_all()
