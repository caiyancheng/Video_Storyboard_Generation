"""
run_quality_analysis_all_seed.py

使用 Seed 模型（Seed1.8 / Seed2.0-Lite）对三个来源的生成视频与 GT 进行质量评分。
每对视频重复打分 5 次（用于方差分析）。

输出 CSV（各模型独立）：
  quality_scores_seed18.csv      ← Seed1.8
  quality_scores_seed20lite.csv  ← Seed2.0-Lite
每个 CSV 包含三个来源的所有评分，通过 source 列区分。

用法：
  python run_quality_analysis_all_seed.py                         # 全量，两个模型
  python run_quality_analysis_all_seed.py --model Seed1.8        # 仅 Seed1.8
  python run_quality_analysis_all_seed.py --stem id_0003_...     # 单 stem
  python run_quality_analysis_all_seed.py --overwrite            # 重新评分（清空后写）
  python run_quality_analysis_all_seed.py --runs 5               # 每对打分次数（默认5）
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# ── sys.path ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Video_Quality_Analyze.score_video_pair import (
    CSV_SCORE_COLUMNS,
    EVAL_PROMPT,
    _compute_scores,
    _extract_json,
)

# ═══════════════════════ SEED API CONFIG ═════════════════════════════════════

SEED_URL = "https://api2.musical.ly/media/api/pic/afr"
SEED_ALGORITHMS = "tt_gpt_vlm"

MODEL2KEY = {
    "Seed1.8":      "ep-20260221081845-5cxpl",
    "Seed2.0-Lite": "ep-20260319113202-l7lrq",
}

# CSV 文件名后缀（按模型名称映射）
MODEL2CSV_SUFFIX = {
    "Seed1.8":      "seed18",
    "Seed2.0-Lite": "seed20lite",
}

# ═══════════════════════ PATH CONFIG ═════════════════════════════════════════

_BASE    = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample")
GT_ROOT  = _BASE / "filter_scored_gt_videos"

SRC1_ROOT = _BASE / "shu_inverse_label" / "generated_videos"
SRC2_ROOT = _BASE / "shu_inverse_label" / "generated_videos_first_last"
SRC3_ROOT = _BASE / "shu_inverse_label" / "generated_storyboard" / "videos"

NUM_LEVELS = 4

# ── 输出目录（放在 Video_Quality_Analyze/ 旁边，或自定义）────────────────────
_OUTPUT_DIR = Path(__file__).resolve().parent

META_COLUMNS = [
    "vid_label", "model_name", "source", "prompt_level",
    "run_index", "gt_path", "gen_path", "scored_at",
]
ALL_COLUMNS = META_COLUMNS + CSV_SCORE_COLUMNS


# ═══════════════════════ SEED CALLER ═════════════════════════════════════════

def call_seed(
    model_key: str,
    gt_path: Path,
    gen_path: Path,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> str | None:
    """
    调用 Seed 多模态 API，传入 GT + 生成视频，返回模型文本输出。
    失败返回 None。
    """
    conf = {
        "biz_id": "ai_theater_video_infer",
        "caller": "veark_caller",
        "model":  model_key,
        "system_prompt": (
            "You are a professional short-video quality evaluator. "
            "Respond ONLY with the requested JSON object."
        ),
        "prompt": EVAL_PROMPT,
        "extra_body": {
            "thinking": {"type": "disabled"}
        },
    }

    for attempt in range(retries):
        try:
            with open(gt_path, "rb") as gt_f, open(gen_path, "rb") as gen_f:
                files = [
                    ("algorithms",    (None, SEED_ALGORITHMS)),
                    ("conf",          (None, json.dumps(conf, ensure_ascii=False))),
                    ("input_img_type", (None, "multiple_files")),
                    ("files[]",       (gt_path.name,  gt_f,  "video/mp4")),
                    ("files[]",       (gen_path.name, gen_f, "video/mp4")),
                ]
                resp = requests.post(SEED_URL, files=files, timeout=120)

            resp.raise_for_status()
            data = resp.json()

            # 解析响应结构：{"code":0,"data":{"text":"..."}}
            if data.get("code") != 0:
                raise ValueError(f"API error code={data.get('code')}: {data.get('message')}")

            text = (
                data.get("data", {}).get("text")
                or data.get("data", {}).get("content")
                or data.get("data", {}).get("answer")
            )
            if text:
                return text

            # fallback：直接返回 data 字段字符串
            tqdm.write(f"    [WARN] Unexpected response structure: {str(data)[:200]}")
            return str(data)

        except Exception as e:
            tqdm.write(f"    [Seed] attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    return None


def score_with_seed(
    model_name: str,
    gt_path: Path,
    gen_path: Path,
) -> dict:
    """
    用指定 Seed 模型对一对视频评分，返回含所有子维度/总分的 dict。
    """
    model_key = MODEL2KEY[model_name]

    tqdm.write(f"    → Seed({model_name}) scoring: {gen_path.name}")
    response = call_seed(model_key, gt_path, gen_path)

    if response is None:
        return {"error": "Seed returned None"}

    parsed = _extract_json(response)
    if parsed is None:
        tqdm.write(f"    [DEBUG] Full response ({len(response)} chars):\n{response[:500]}")
        return {"error": f"Failed to parse JSON (len={len(response)})"}

    scores = _compute_scores(parsed)
    scores["error"] = ""
    return scores


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
    level_dir = src_root / f"level_{level}"
    if not level_dir.exists():
        return None
    for p in level_dir.glob("*.mp4"):
        if stem in p.stem:
            return p
    return None


def find_storyboard_video(stem: str) -> Path | None:
    p = SRC3_ROOT / f"{stem}.mp4"
    if p.exists():
        return p
    parts = stem.split("_", 3)
    if len(parts) >= 4:
        for p2 in SRC3_ROOT.glob(f"*{parts[3]}*.mp4"):
            return p2
    return None


def collect_all_stems() -> list[str]:
    stems: set[str] = set()
    for src_root in (SRC1_ROOT, SRC2_ROOT):
        level1 = src_root / "level_1"
        if level1.exists():
            for p in level1.glob("*.mp4"):
                stems.add(p.stem.replace("_plevel1", ""))
    if SRC3_ROOT.exists():
        for p in SRC3_ROOT.glob("*.mp4"):
            stems.add(p.stem)

    def sort_key(s: str):
        m = re.match(r"id_(\d+)", s)
        return int(m.group(1)) if m else 0

    return sorted(stems, key=sort_key)


# ═══════════════════════ CSV HELPERS ═════════════════════════════════════════

def csv_path_for_model(model_name: str) -> Path:
    suffix = MODEL2CSV_SUFFIX[model_name]
    return _OUTPUT_DIR / f"quality_scores_{suffix}.csv"


def load_existing_keys(csv_path: Path) -> set[tuple[str, str, str, str]]:
    """已评分记录的 (vid_label, source, prompt_level, run_index) 集合。"""
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            keys.add((
                row.get("vid_label", ""),
                row.get("source", ""),
                str(row.get("prompt_level", "")),
                str(row.get("run_index", "")),
            ))
    return keys


def write_row(csv_path: Path, row: dict):
    """首次建文件写 BOM + 表头；后续追加行。"""
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})
    else:
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})


# ═══════════════════════ SCORE ONE PAIR (N RUNS) ══════════════════════════════

def score_and_write_n_runs(
    csv_path: Path,
    existing: set[tuple[str, str, str, str]],
    model_name: str,
    stem: str,
    source: str,
    prompt_level: int,
    gt_path: Path,
    gen_path: Path,
    num_runs: int,
) -> tuple[int, int, int]:
    """
    对一对视频打分 num_runs 次（跳过已有的 run_index）。
    返回 (ok, fail, skip) 计数。
    """
    ok = fail = skip = 0

    if not gen_path.exists():
        tqdm.write(f"    [SKIP] 文件不存在: {gen_path.name}")
        return 0, 0, num_runs

    for run_idx in range(1, num_runs + 1):
        key = (stem, source, str(prompt_level), str(run_idx))

        if key in existing:
            skip += 1
            continue

        # 再次落盘校验（防并发重复写）
        on_disk = load_existing_keys(csv_path)
        if key in on_disk:
            existing.add(key)
            skip += 1
            continue

        tqdm.write(f"      run {run_idx}/{num_runs}")
        scores = score_with_seed(model_name, gt_path, gen_path)

        if scores.get("error"):
            tqdm.write(f"      ✗ 失败（不写入）: {scores['error']}")
            fail += 1
            continue

        row = {
            "vid_label":    stem,
            "model_name":   model_name,
            "source":       source,
            "prompt_level": prompt_level,
            "run_index":    run_idx,
            "gt_path":      str(gt_path),
            "gen_path":     str(gen_path),
            "scored_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        row.update(scores)
        write_row(csv_path, row)
        existing.add(key)

        tqdm.write(
            f"      ✓ total={scores['total_score']:.2f}  "
            f"sim={scores['sim_score']:.2f}  "
            f"aes={scores['aes_score']:.2f}  "
            f"aud={scores['aud_score']:.2f}  "
            f"nar={scores['nar_score']:.2f}"
        )
        ok += 1

    return ok, fail, skip


# ═══════════════════════ MAIN LOOP ════════════════════════════════════════════

def run(model_name: str, stems: list[str], overwrite: bool, num_runs: int):
    out_csv = csv_path_for_model(model_name)
    existing = set() if overwrite else load_existing_keys(out_csv)

    print(f"\n{'═'*60}")
    print(f"模型: {model_name}  →  {out_csv.name}")
    print(f"已有记录: {len(existing)}  |  stem 数: {len(stems)}  |  每对打分: {num_runs} 次")
    print(f"{'═'*60}")

    total_ok = total_fail = total_skip_gt = total_skip_gen = 0

    for stem in tqdm(stems, desc=f"{model_name} 遍历 stem"):
        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"[{stem}]")

        gt_path = find_gt_video(stem)
        if gt_path is None:
            tqdm.write(f"  [SKIP] GT 视频不存在: {stem}")
            total_skip_gt += 1
            continue

        # ── 来源1: generated_videos level 1-4 ─────────────────────────────
        tqdm.write(f"  [src1] generated_videos")
        for lv in range(1, NUM_LEVELS + 1):
            gen = find_leveled_video(SRC1_ROOT, stem, lv)
            if gen is None:
                tqdm.write(f"    [SKIP] level={lv} 不存在")
                total_skip_gen += num_runs
                continue
            ok, fail, skip = score_and_write_n_runs(
                out_csv, existing, model_name, stem,
                "generated_videos", lv, gt_path, gen, num_runs,
            )
            total_ok += ok; total_fail += fail; total_skip_gen += skip

        # ── 来源2: generated_videos_first_last level 1-4 ──────────────────
        tqdm.write(f"  [src2] generated_videos_first_last")
        for lv in range(1, NUM_LEVELS + 1):
            gen = find_leveled_video(SRC2_ROOT, stem, lv)
            if gen is None:
                tqdm.write(f"    [SKIP] level={lv} 不存在")
                total_skip_gen += num_runs
                continue
            ok, fail, skip = score_and_write_n_runs(
                out_csv, existing, model_name, stem,
                "generated_videos_first_last", lv, gt_path, gen, num_runs,
            )
            total_ok += ok; total_fail += fail; total_skip_gen += skip

        # ── 来源3: generated_storyboard/videos（level=0）──────────────────
        tqdm.write(f"  [src3] generated_storyboard/videos")
        gen = find_storyboard_video(stem)
        if gen is None:
            tqdm.write(f"    [SKIP] storyboard 视频不存在")
            total_skip_gen += num_runs
        else:
            ok, fail, skip = score_and_write_n_runs(
                out_csv, existing, model_name, stem,
                "generated_storyboard", 0, gt_path, gen, num_runs,
            )
            total_ok += ok; total_fail += fail; total_skip_gen += skip

    print(f"\n{'='*60}")
    print(f"[{model_name}] 完成：成功 {total_ok}，失败 {total_fail}，"
          f"跳过(GT缺失) {total_skip_gt*num_runs}，跳过(生成缺失/已有) {total_skip_gen}")
    print(f"CSV → {out_csv}")


# ═══════════════════════ ENTRY ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Seed 模型视频质量批量评分（含方差重复）")
    parser.add_argument(
        "--model", nargs="+",
        default=list(MODEL2KEY.keys()),
        choices=list(MODEL2KEY.keys()),
        help=f"要评分的模型（默认全部）: {list(MODEL2KEY.keys())}",
    )
    parser.add_argument(
        "--stem", nargs="+", default=None,
        help="仅评估指定 stem（空=全量）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="重新评估（忽略已有记录）",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="每对视频重复评分次数（默认 5）",
    )
    args = parser.parse_args()

    all_stems = collect_all_stems()
    stems = args.stem if args.stem else all_stems

    print(f"GT 根目录  : {GT_ROOT}")
    print(f"来源1      : {SRC1_ROOT}")
    print(f"来源2      : {SRC2_ROOT}")
    print(f"来源3      : {SRC3_ROOT}")
    print(f"共 {len(stems)} 个 stem，每 stem 最多 9 对（4+4+1），每对打 {args.runs} 次")
    print(f"模型       : {args.model}")

    for model_name in args.model:
        run(model_name, stems, args.overwrite, args.runs)

    print("\n全部完成。")
    print("CSV 输出：")
    for m in args.model:
        print(f"  {csv_path_for_model(m)}")


if __name__ == "__main__":
    main()