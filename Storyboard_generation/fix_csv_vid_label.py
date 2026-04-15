"""
fix_csv_vid_label.py  (本地运行)

generation_results.csv 中所有 vid_label 都错误地记录为 id_0000_...（_idx 默认值）。
本脚本通过 video_id 字段在 first_frame/ 目录中查找真实文件名，
将 vid_label 和 idx 修正为正确值，并原地覆写 CSV。

用法：
  python fix_csv_vid_label.py              # 修正 + 原地覆写
  python fix_csv_vid_label.py --dry-run    # 仅打印对比，不写入
"""

import argparse
import csv
import re
from pathlib import Path

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

CSV_PATH = Path(
    "/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/generated_storyboard/generation_results.csv"
)

FIRST_FRAME_DIR = Path(
    "/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample/first_frame"
)

# ═══════════════════════ BUILD LOOKUP ════════════════════════════════════════

def build_video_id_to_stem(first_frame_dir: Path) -> dict[str, str]:
    """
    扫描 first_frame_dir，建立 video_id → stem 的映射。
    文件名格式：id_XXXX_scoreY_<video_id>_first.png
    返回：{"<video_id>": "id_XXXX_scoreY_<video_id>", ...}
    """
    lookup: dict[str, str] = {}
    for p in first_frame_dir.glob("*_first.png"):
        # stem 去掉 _first 后缀
        stem = p.name.replace("_first.png", "")
        # 格式：id_DDDD_scoreD_<video_id>
        m = re.match(r"(id_\d{4}_score\d+_)(.+)", stem)
        if m:
            video_id = m.group(2)
            lookup[video_id] = stem
    return lookup


# ═══════════════════════ FIX ═════════════════════════════════════════════════

def fix_csv(dry_run: bool = False):
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV 不存在: {CSV_PATH}")
        return
    if not FIRST_FRAME_DIR.exists():
        print(f"[ERROR] first_frame 目录不存在: {FIRST_FRAME_DIR}")
        return

    lookup = build_video_id_to_stem(FIRST_FRAME_DIR)
    print(f"first_frame 目录共 {len(lookup)} 条映射")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    fixed = 0
    not_found = 0
    updated_rows = []

    for row in rows:
        video_id   = row.get("video_id", "").strip()
        old_label  = row.get("vid_label", "").strip()

        if video_id not in lookup:
            print(f"  [WARN] video_id 未找到对应首帧: {video_id}")
            not_found += 1
            updated_rows.append(row)
            continue

        new_stem  = lookup[video_id]
        # 从 new_stem 中提取 idx（id_XXXX 中的 XXXX）
        idx_match = re.match(r"id_(\d{4})", new_stem)
        new_idx   = str(int(idx_match.group(1))) if idx_match else row.get("idx", "0")

        if old_label != new_stem:
            print(f"  FIX: {old_label!r:55s} → {new_stem!r}")
            fixed += 1

        new_row = dict(row)
        new_row["vid_label"] = new_stem
        new_row["idx"]       = new_idx
        updated_rows.append(new_row)

    print(f"\n共修正 {fixed} 条，未匹配 {not_found} 条，共 {len(rows)} 行")

    if dry_run:
        print("[dry-run] 未写入，加 --dry-run=false 或去掉该参数以实际写入")
        return

    # 原地覆写（先写临时文件再 rename 保证原子性）
    tmp_path = CSV_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    tmp_path.replace(CSV_PATH)
    print(f"已覆写: {CSV_PATH}")


# ═══════════════════════ ENTRY ═══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修正 generation_results.csv 中错误的 vid_label / idx")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印修改内容，不写入文件")
    args = parser.parse_args()
    fix_csv(dry_run=args.dry_run)
