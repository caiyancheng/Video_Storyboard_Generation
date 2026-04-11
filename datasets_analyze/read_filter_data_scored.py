"""
read_filter_data_scored.py

读取 new_scored_filter_datasets 输出的 scored JSONL 文件，
筛选出指定评分的记录并打印关键信息和视频链接。
"""

import json
from pathlib import Path

SCORED_FILE  = Path("/Users/bytedance/Datasets/tt_template_hq_publish_data_1400k_USAU.dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl")
TOS_BASE_URL = "https://tosv-va.tiktok-row.org/obj/nebudata-us/"

TARGET_SCORES = {4, 5}


def load_scored(target_scores: set[int] = TARGET_SCORES) -> list[dict]:
    records = []
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_score") in target_scores:
                records.append(rec)
    return records


def print_records(records: list[dict]) -> None:
    print(f"\n共找到 {len(records)} 条评分为 {sorted(TARGET_SCORES)} 分的记录\n")
    print(f"{'No.':<5} {'分数':>4}  {'时长':>7}  {'confidence':>10}  {'category':<25}  {'video_url'}")
    print("-" * 130)
    for i, rec in enumerate(records, 1):
        result = rec.get("result", {})
        tos    = rec.get("tos_key", "")
        url    = rec.get("video_url") or (TOS_BASE_URL + tos if tos else "N/A")
        dur    = rec.get("_duration") or rec.get("duration") or result.get("duration")
        dur_s  = f"{float(dur):.2f}s" if dur is not None else "   N/A"
        score  = rec.get("_score", "?")
        conf   = result.get("confidence", "")
        cat    = result.get("category", "")
        print(f"{i:<5} {score:>4}  {dur_s:>7}  {str(conf):>10}  {cat:<25}  {url}")


if __name__ == "__main__":
    records = load_scored(target_scores=TARGET_SCORES)
    print_records(records)
