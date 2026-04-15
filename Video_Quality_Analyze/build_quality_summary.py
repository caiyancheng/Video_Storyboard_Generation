"""
build_quality_summary.py

读取三个来源的 quality_scores.csv，合并为一份可读性强的汇总表，
保存到 shu_inverse_label/quality_summary.csv。

表结构：
  - 每行 = 一个视频索引 × 一个来源 × 一个提示级别
  - 每个视频索引最多 9 行：
      首帧+剧本 (SRC1)         × level 1~4
      首帧+尾帧+剧本 (SRC2)    × level 1~4
      首帧+首帧生成剧本 (SRC3) × level 1（单个）
  - 列标题全部使用中文描述

用法：
  python Video_Quality_Analyze/build_quality_summary.py
"""

import csv
import re
import sys
from pathlib import Path

# ── sys.path ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

_BASE = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample")

SRC1_CSV = _BASE / "shu_inverse_label/generated_videos/quality_scores.csv"
SRC2_CSV = _BASE / "shu_inverse_label/generated_videos_first_last/quality_scores.csv"
SRC3_CSV = _BASE / "shu_inverse_label/generated_storyboard/videos/quality_scores.csv"

OUT_CSV  = _BASE / "shu_inverse_label/quality_summary.csv"

# ═══════════════════════ 来源标签映射 ════════════════════════════════════════

SOURCE_LABEL = {
    "generated_videos":         "首帧+剧本 (SRC1)",
    "generated_videos_first_last": "首帧+尾帧+剧本 (SRC2)",
    "generated_storyboard":     "首帧+首帧生成剧本 (SRC3)",
}

# ═══════════════════════ 列名中英文映射 ══════════════════════════════════════

# (原始 CSV 字段, 输出中文列名)
COLUMN_MAP = [
    # 基础信息
    ("vid_label",                   "视频索引"),
    ("来源_display",                 "来源"),         # 处理后的来源标签
    ("prompt_level_display",        "提示级别"),      # 处理后的级别描述
    # 综合与主维度
    ("total_score",                 "综合评分"),
    ("sim_score",                   "相似度评分"),
    ("aes_score",                   "美学质量评分"),
    ("aud_score",                   "音频质量评分"),
    ("nar_score",                   "叙事吸引力评分"),
    # 相似度子维度
    ("sim_subject_consistency",     "相似度-主体一致性"),
    ("sim_style_consistency",       "相似度-风格一致性"),
    ("sim_motion_consistency",      "相似度-动作一致性"),
    ("sim_scene_consistency",       "相似度-场景一致性"),
    ("sim_audio_consistency",       "相似度-音频一致性"),
    # 美学子维度
    ("aes_image_quality",           "美学-画面质量"),
    ("aes_composition",             "美学-构图美感"),
    ("aes_color",                   "美学-色彩表现"),
    ("aes_motion_smoothness",       "美学-运动流畅度"),
    ("aes_temporal_consistency",    "美学-时序一致性"),
    # 音频子维度
    ("aud_audio_clarity",           "音频-清晰度"),
    ("aud_timbre_consistency",      "音频-音色一致性"),
    ("aud_av_sync",                 "音频-音画同步"),
    ("aud_rhythm_matching",         "音频-节奏匹配"),
    # 叙事子维度
    ("nar_hook_strength",           "叙事-钩子强度"),
    ("nar_narrative_arc",           "叙事-故事弧线"),
    ("nar_rhythm_engagement",       "叙事-节奏吸引力"),
    ("nar_emotional_resonance",     "叙事-情感共鸣"),
    ("nar_replay_value",            "叙事-复看价值"),
]

OUTPUT_FIELDNAMES = [cn for _, cn in COLUMN_MAP]
FIELD_KEY_TO_CN  = {k: cn for k, cn in COLUMN_MAP}


# ═══════════════════════ LOAD ════════════════════════════════════════════════

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [SKIP] CSV 不存在: {path}")
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    print(f"  载入 {len(rows)} 条: {path.name}")
    return rows


def stem_sort_key(stem: str) -> int:
    m = re.match(r"id_(\d+)", stem)
    return int(m.group(1)) if m else 0


# ═══════════════════════ TRANSFORM ══════════════════════════════════════════

def transform_row(raw: dict) -> dict:
    """将原始 CSV 行转换为输出行（中文字段名）。"""
    source_key  = raw.get("source", "")
    level_raw   = str(raw.get("prompt_level", "0"))

    # 来源中文标签（fallback：原始值）
    source_cn = SOURCE_LABEL.get(source_key, source_key)

    # 提示级别描述
    try:
        lv = int(level_raw)
    except ValueError:
        lv = 0
    level_cn = "单级" if lv == 0 else f"Level {lv}"

    enriched = dict(raw)
    enriched["来源_display"]          = source_cn
    enriched["prompt_level_display"]  = level_cn

    out: dict = {}
    for src_key, cn_key in COLUMN_MAP:
        out[cn_key] = enriched.get(src_key, "")
    return out


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def main():
    print("读取各来源 CSV …")
    rows1 = load_csv(SRC1_CSV)
    rows2 = load_csv(SRC2_CSV)
    rows3 = load_csv(SRC3_CSV)

    all_rows = rows1 + rows2 + rows3
    if not all_rows:
        print("[ERROR] 三个 CSV 均为空或不存在，无数据可合并。")
        return

    # 确定所有 stem 并按 id 数字排序
    all_stems = sorted(
        {r.get("vid_label", "") for r in all_rows if r.get("vid_label", "")},
        key=stem_sort_key,
    )
    print(f"\n共 {len(all_stems)} 个视频索引，{len(all_rows)} 条评分记录")

    # 按 stem → 来源顺序（src1 < src2 < src3）→ level 排序
    SOURCE_ORDER = {
        "generated_videos":              0,
        "generated_videos_first_last":   1,
        "generated_storyboard":          2,
    }

    def row_sort_key(r: dict):
        stem  = r.get("vid_label", "")
        src   = r.get("source", "")
        try:
            lv = int(r.get("prompt_level", 0))
        except ValueError:
            lv = 0
        return (stem_sort_key(stem), SOURCE_ORDER.get(src, 9), lv)

    all_rows_sorted = sorted(all_rows, key=row_sort_key)

    # 转换 & 写出
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        for raw in all_rows_sorted:
            writer.writerow(transform_row(raw))

    print(f"\n汇总表已保存到:\n  {OUT_CSV}")
    print(f"共写入 {len(all_rows_sorted)} 行（列数：{len(OUTPUT_FIELDNAMES)}）")


if __name__ == "__main__":
    main()
