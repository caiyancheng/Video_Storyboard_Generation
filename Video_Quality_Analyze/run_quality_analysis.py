"""
run_quality_analysis.py

批量对所有 GT + 生成视频对进行 Gemini 质量评分，结果写入 CSV。

输出：
  /Users/bytedance/Datasets/tt_template_1400k_15s_video_sample/
    shu_inverse_label/generated_videos/quality_scores/quality_scores.csv

用法：
  python run_quality_analysis.py              # 全量（所有 stem × 所有 level）
  python run_quality_analysis.py --stem id_0003_score5_xxx   # 单 stem
  python run_quality_analysis.py --level 1 2  # 仅评估指定 level
  python run_quality_analysis.py --overwrite  # 重新评分（忽略已有记录）
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# ── 将项目根目录加入 sys.path，确保包导入正常 ──────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Video_Quality_Analyze.score_video_pair import CSV_SCORE_COLUMNS, score_video_pair

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

GT_ROOT   = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                 "/filter_scored_gt_videos")
GEN_ROOT  = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                 "/shu_inverse_label/generated_videos")
OUT_DIR   = GEN_ROOT / "quality_scores"
CSV_PATH  = OUT_DIR / "quality_scores.csv"

NUM_LEVELS = 4

# CSV 列顺序
META_COLUMNS = [
    "vid_label",       # GT stem（与 compare_video_levels.py 一致）
    "prompt_level",    # 1-4
    "gt_path",
    "gen_path",
    "scored_at",
]
ALL_COLUMNS = META_COLUMNS + CSV_SCORE_COLUMNS


# ═══════════════════════ FILE LOOKUP ═════════════════════════════════════════

def find_gt_video(stem: str) -> Path | None:
    """在 score4/ score5/ 下查找含 stem 的 GT 视频。"""
    for score_dir in GT_ROOT.iterdir():
        if not score_dir.is_dir():
            continue
        for p in score_dir.glob("*.mp4"):
            if stem in p.stem:
                return p
    return None


def find_gen_video(stem: str, level: int) -> Path | None:
    """在 generated_videos/level_{N}/ 下查找 {stem}_plevel{N}.mp4。"""
    level_dir = GEN_ROOT / f"level_{level}"
    if not level_dir.exists():
        return None
    for p in level_dir.glob("*.mp4"):
        if stem in p.stem:
            return p
    return None


def collect_all_gt_stems() -> list[str]:
    """收集 GT 目录下所有视频的 stem。"""
    stems = []
    for score_dir in sorted(GT_ROOT.iterdir()):
        if not score_dir.is_dir():
            continue
        for p in sorted(score_dir.glob("*.mp4")):
            stems.append(p.stem)
    return stems


# ═══════════════════════ CSV UPSERT ══════════════════════════════════════════

def load_existing_keys(csv_path: Path) -> set[tuple[str, str]]:
    """已评分记录的 (vid_label, prompt_level) 集合。"""
    if not csv_path.exists():
        return set()
    keys = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["vid_label"], str(row["prompt_level"])))
    return keys


def append_row(csv_path: Path, row: dict):
    """追加一行到 CSV。
    - 新建文件：用 utf-8-sig（写入 BOM），Excel 打开中文不乱码。
    - 追加行：用 utf-8，避免 utf-8-sig 在 append 模式下每次都往末尾重复写入 BOM。
    """
    if not csv_path.exists():
        # 新建：写 BOM + 表头 + 第一行
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})
    else:
        # 追加：不写 BOM，只追加数据行
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def run(stems: list[str], levels: list[int], overwrite: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = set() if overwrite else load_existing_keys(CSV_PATH)
    print(f"已有评分记录：{len(existing)} 条")

    pairs: list[tuple[str, int]] = []
    for stem in stems:
        for lv in levels:
            if (stem, str(lv)) not in existing:
                pairs.append((stem, lv))

    print(f"待评分对数：{len(pairs)}")

    ok = skip_no_file = fail = 0

    for stem, lv in tqdm(pairs, desc="质量评分"):
        gt_path  = find_gt_video(stem)
        gen_path = find_gen_video(stem, lv)

        if gt_path is None:
            tqdm.write(f"  [SKIP] GT 不存在：{stem}")
            skip_no_file += 1
            continue
        if gen_path is None:
            tqdm.write(f"  [SKIP] 生成视频不存在：{stem} level={lv}")
            skip_no_file += 1
            continue

        tqdm.write(f"\n[{stem}]  level={lv}")

        scores = score_video_pair(gt_path, gen_path)

        if scores.get("error"):
            tqdm.write(f"  ✗ 评分失败（不写入 CSV）：{scores['error']}")
            fail += 1
            continue

        row = {
            "vid_label":    stem,
            "prompt_level": lv,
            "gt_path":      str(gt_path),
            "gen_path":     str(gen_path),
            "scored_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        row.update(scores)

        append_row(CSV_PATH, row)
        tqdm.write(
            f"  ✓  total={scores['total_score']:.2f}  "
            f"sim={scores['sim_score']:.2f}  "
            f"aes={scores['aes_score']:.2f}  "
            f"aud={scores['aud_score']:.2f}  "
            f"nar={scores['nar_score']:.2f}"
        )
        ok += 1

    print(f"\n完成：成功 {ok} 对，文件缺失跳过 {skip_no_file} 对，失败 {fail} 对")
    print(f"结果 CSV：{CSV_PATH}")


# ═══════════════════════ ENTRY ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gemini 视频质量批量评分")
    parser.add_argument("--stem",      nargs="+", default=None,
                        help="仅评估指定 stem（空=全量）")
    parser.add_argument("--level",     nargs="+", type=int, default=None,
                        help="仅评估指定 level（如 1 2 3 4，空=全量）")
    parser.add_argument("--overwrite", action="store_true",
                        help="重新评估已有记录（默认跳过）")
    args = parser.parse_args()

    stems  = args.stem  if args.stem  else collect_all_gt_stems()
    levels = args.level if args.level else list(range(1, NUM_LEVELS + 1))

    print(f"GT 根目录：{GT_ROOT}")
    print(f"生成视频根目录：{GEN_ROOT}")
    print(f"评估 stem 数：{len(stems)}")
    print(f"评估 level：{levels}")

    run(stems, levels, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
