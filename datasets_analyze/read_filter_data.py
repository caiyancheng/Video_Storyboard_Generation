"""
read_filter_data.py

读取 filter_datasets.py 输出的 filtered JSONL 文件，展示每条记录的关键信息和视频链接。
"""

import json
from pathlib import Path

from tqdm import tqdm

FILTERED_FILE = Path("/Users/bytedance/Datasets/tt_template_hq_publish_data_1400k_USAU.dedup_item_id_aesthetic_quality_v1_filtered.jsonl")
TOS_BASE_URL = "https://tosv-va.tiktok-row.org/obj/nebudata-us/"


def load_records() -> list[dict]:
    records = []
    with FILTERED_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="读取"):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def print_records(records: list[dict], limit: int | None = None) -> None:
    to_show = records if limit is None else records[:limit]
    print(f"\n共 {len(records)} 条记录，显示前 {len(to_show)} 条\n")
    print(f"{'No.':<6} {'duration':>8}  {'source_q':<10} {'quality_l':<10} {'conf':>5}  {'video_url'}")
    print("-" * 120)
    for i, rec in enumerate(to_show, 1):
        result = rec.get("result", {})
        tos_key = rec.get("tos_key", "")
        url = rec.get("video_url") or (TOS_BASE_URL + tos_key if tos_key else "N/A")
        dur = rec.get("_duration") or rec.get("duration") or result.get("duration")
        dur_str = f"{dur:.2f}s" if dur is not None else "  N/A"
        print(
            f"{i:<6} {dur_str:>8}  "
            f"{result.get('source_quality', ''):<10} "
            f"{result.get('quality_level', ''):<10} "
            f"{result.get('confidence', 0):>5.2f}  "
            f"{url}"
        )


if __name__ == "__main__":
    print(f"读取文件: {FILTERED_FILE}")
    records = load_records()
    print_records(records)
