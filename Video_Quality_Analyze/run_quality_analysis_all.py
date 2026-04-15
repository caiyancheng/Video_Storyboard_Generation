"""
run_quality_analysis_all.py

批量对三个来源的生成视频与 GT 进行 Gemini 质量评分。

三个来源（结果分别写入各自根目录下的 quality_scores.csv）：
  1. generated_videos/level_{1-4}/          → generated_videos/quality_scores.csv
  2. generated_videos_first_last/level_{1-4}/ → generated_videos_first_last/quality_scores.csv
  3. generated_storyboard/videos/            → generated_storyboard/videos/quality_scores.csv

遍历顺序：按 id 索引排序，对每个 stem：
  先跑来源1 level 1→4，再跑来源2 level 1→4，最后跑来源3（单个视频）。

用法：
  python run_quality_analysis_all.py                        # 全量
  python run_quality_analysis_all.py --stem id_0003_...    # 单 stem
  python run_quality_analysis_all.py --overwrite           # 重新评分
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# ── sys.path ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Video_Quality_Analyze.score_video_pair import CSV_SCORE_COLUMNS, score_video_pair

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

_BASE = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample")

GT_ROOT = _BASE / "filter_scored_gt_videos"

# 三个来源的根目录
SRC1_ROOT = _BASE / "shu_inverse_label" / "generated_videos"
SRC2_ROOT = _BASE / "shu_inverse_label" / "generated_videos_first_last"
SRC3_ROOT = _BASE / "shu_inverse_label" / "generated_storyboard" / "videos"

# 各来源输出 CSV 路径
SRC1_CSV = SRC1_ROOT / "quality_scores.csv"
SRC2_CSV = SRC2_ROOT / "quality_scores.csv"
SRC3_CSV = SRC3_ROOT / "quality_scores.csv"

NUM_LEVELS = 4   # 来源1、2 各有 4 个 level；来源3 只有 1 个视频（prompt_level=0）

# CSV 列定义
META_COLUMNS = ["vid_label", "source", "prompt_level", "gt_path", "gen_path", "scored_at"]
ALL_COLUMNS  = META_COLUMNS + CSV_SCORE_COLUMNS


# ═══════════════════════ FILE LOOKUP ═════════════════════════════════════════

def find_gt_video(stem: str) -> Path | None:
    for score_dir in GT_ROOT.iterdir():
        if not score_dir.is_dir():
            continue
        for p in score_dir.glob("*.mp4"):
            if stem in p.stem:
                return p
    return None


def find_leveled_video(src_root: Path, stem: str, level: int) -> Path | None:
    """在 src_root/level_{N}/ 下找 {stem}_plevel{N}.mp4。"""
    level_dir = src_root / f"level_{level}"
    if not level_dir.exists():
        return None
    for p in level_dir.glob("*.mp4"):
        if stem in p.stem:
            return p
    return None


def find_storyboard_video(stem: str) -> Path | None:
    """在 generated_storyboard/videos/ 下找 {stem}.mp4。"""
    p = SRC3_ROOT / f"{stem}.mp4"
    if p.exists():
        return p
    # fallback: glob by video_id
    parts = stem.split("_", 3)
    if len(parts) >= 4:
        for p2 in SRC3_ROOT.glob(f"*{parts[3]}*.mp4"):
            return p2
    return None


def collect_all_stems() -> list[str]:
    """收集三个来源所有出现过的 stem，按 id 数字排序后去重。"""
    stems: set[str] = set()
    # 来源1、2：从 level_1 目录扫描
    for src_root in (SRC1_ROOT, SRC2_ROOT):
        level1 = src_root / "level_1"
        if level1.exists():
            for p in level1.glob("*.mp4"):
                # stem = 去掉 _plevel1 后缀
                stems.add(p.stem.replace("_plevel1", ""))
    # 来源3
    if SRC3_ROOT.exists():
        for p in SRC3_ROOT.glob("*.mp4"):
            stems.add(p.stem)
    # 按 id 前缀数字排序
    def sort_key(s: str):
        import re
        m = re.match(r"id_(\d+)", s)
        return int(m.group(1)) if m else 0
    return sorted(stems, key=sort_key)


# ═══════════════════════ CSV WRITE ═══════════════════════════════════════════

def load_existing_keys(csv_path: Path) -> set[tuple[str, str]]:
    """已评分记录的 (vid_label, prompt_level) 集合。"""
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["vid_label"], str(row["prompt_level"])))
    return keys


def write_row(csv_path: Path, row: dict):
    """首次建文件写 BOM + 表头；后续追加不写 BOM。"""
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})
    else:
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})


# ═══════════════════════ SCORE ONE PAIR ══════════════════════════════════════

def score_and_write(
    csv_path: Path,
    existing: set[tuple[str, str]],
    stem: str,
    source: str,
    prompt_level: int,
    gt_path: Path,
    gen_path: Path,
) -> str:
    """
    评分并写入 CSV。返回 "ok" / "skip" / "fail"。
    existing 集合会在成功写入后原地更新，避免同一 run 内重复评分。
    写入前再次从磁盘加载已有记录，防止多进程并发时重复写入。
    """
    key = (stem, str(prompt_level))
    if key in existing:
        return "skip"

    # 写入前再次落盘校验（防并发重复）
    on_disk = load_existing_keys(csv_path)
    if key in on_disk:
        existing.add(key)
        return "skip"

    if not gen_path.exists():
        tqdm.write(f"    [SKIP] 文件不存在: {gen_path.name}")
        return "skip"

    tqdm.write(f"    → Gemini scoring: {gen_path.name}")
    scores = score_video_pair(gt_path, gen_path)

    if scores.get("error"):
        tqdm.write(f"    ✗ 失败（不写入）: {scores['error']}")
        return "fail"

    row = {
        "vid_label":    stem,
        "source":       source,
        "prompt_level": prompt_level,
        "gt_path":      str(gt_path),
        "gen_path":     str(gen_path),
        "scored_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    row.update(scores)
    write_row(csv_path, row)
    existing.add(key)

    tqdm.write(
        f"    ✓ total={scores['total_score']:.2f}  "
        f"sim={scores['sim_score']:.2f}  "
        f"aes={scores['aes_score']:.2f}  "
        f"aud={scores['aud_score']:.2f}  "
        f"nar={scores['nar_score']:.2f}"
    )
    return "ok"


# ═══════════════════════ MAIN LOOP ═══════════════════════════════════════════

def run(stems: list[str], overwrite: bool):
    # 加载三个 CSV 的已有记录
    ex1 = set() if overwrite else load_existing_keys(SRC1_CSV)
    ex2 = set() if overwrite else load_existing_keys(SRC2_CSV)
    ex3 = set() if overwrite else load_existing_keys(SRC3_CSV)
    print(f"已有记录 — src1:{len(ex1)}  src2:{len(ex2)}  src3:{len(ex3)}")

    ok = fail = skip_no_gt = skip_no_gen = 0

    for stem in tqdm(stems, desc="遍历 stem"):
        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"[{stem}]")

        gt_path = find_gt_video(stem)
        if gt_path is None:
            tqdm.write(f"  [SKIP] GT 视频不存在: {stem}")
            skip_no_gt += 1
            continue

        # ── 来源1: generated_videos level 1-4 ────────────────────────────
        tqdm.write(f"  [src1] generated_videos")
        for lv in range(1, NUM_LEVELS + 1):
            gen = find_leveled_video(SRC1_ROOT, stem, lv)
            if gen is None:
                tqdm.write(f"    [SKIP] level={lv} 不存在")
                skip_no_gen += 1
                continue
            result = score_and_write(SRC1_CSV, ex1, stem, "generated_videos", lv, gt_path, gen)
            if result == "ok":    ok   += 1
            elif result == "fail": fail += 1
            else:                  skip_no_gen += 1

        # ── 来源2: generated_videos_first_last level 1-4 ─────────────────
        tqdm.write(f"  [src2] generated_videos_first_last")
        for lv in range(1, NUM_LEVELS + 1):
            gen = find_leveled_video(SRC2_ROOT, stem, lv)
            if gen is None:
                tqdm.write(f"    [SKIP] level={lv} 不存在")
                skip_no_gen += 1
                continue
            result = score_and_write(SRC2_CSV, ex2, stem, "generated_videos_first_last", lv, gt_path, gen)
            if result == "ok":    ok   += 1
            elif result == "fail": fail += 1
            else:                  skip_no_gen += 1

        # ── 来源3: generated_storyboard/videos（单个，level=0）───────────
        tqdm.write(f"  [src3] generated_storyboard/videos")
        gen = find_storyboard_video(stem)
        if gen is None:
            tqdm.write(f"    [SKIP] storyboard 视频不存在")
            skip_no_gen += 1
        else:
            result = score_and_write(SRC3_CSV, ex3, stem, "generated_storyboard", 0, gt_path, gen)
            if result == "ok":    ok   += 1
            elif result == "fail": fail += 1
            else:                  skip_no_gen += 1

    print(f"\n{'='*60}")
    print(f"完成：成功 {ok}，失败 {fail}，跳过(GT缺失) {skip_no_gt}，跳过(生成缺失/已有) {skip_no_gen}")
    print(f"CSV 路径:")
    print(f"  {SRC1_CSV}")
    print(f"  {SRC2_CSV}")
    print(f"  {SRC3_CSV}")


# ═══════════════════════ ENTRY ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="三来源视频 Gemini 质量批量评分")
    parser.add_argument("--stem",      nargs="+", default=None,
                        help="仅评估指定 stem（空=全量）")
    parser.add_argument("--overwrite", action="store_true",
                        help="重新评估已有记录（默认跳过）")
    args = parser.parse_args()

    all_stems = collect_all_stems()
    stems = args.stem if args.stem else all_stems

    print(f"GT 根目录       : {GT_ROOT}")
    print(f"来源1           : {SRC1_ROOT}")
    print(f"来源2           : {SRC2_ROOT}")
    print(f"来源3           : {SRC3_ROOT}")
    print(f"共 {len(stems)} 个 stem，每 stem 最多 9 对（4+4+1）")

    run(stems, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
