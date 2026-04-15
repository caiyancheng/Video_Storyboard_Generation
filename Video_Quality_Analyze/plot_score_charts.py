"""
plot_score_charts.py

为每个视频索引绘制一张分组柱状图：
  - X 轴：该索引下所有存在的视频变体（SRC1 L1-4 / SRC2 L1-4 / SRC3）
  - Y 轴：质量分数（0–10）
  - 每个 X 位置绘制 4 根柱子（4 个维度评分）

保存目录：
  .../shu_inverse_label/score_charts/
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体（macOS 系统字体）
_CN_FONT = next(
    (f.name for f in fm.fontManager.ttflist
     if any(kw in f.name for kw in ("PingFang", "STHeiti", "Hiragino Sans GB", "Arial Unicode"))),
    None,
)
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches
import numpy as np

# ── sys.path ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

_BASE    = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample")
SRC1_CSV = _BASE / "shu_inverse_label/generated_videos/quality_scores.csv"
SRC2_CSV = _BASE / "shu_inverse_label/generated_videos_first_last/quality_scores.csv"
SRC3_CSV = _BASE / "shu_inverse_label/generated_storyboard/videos/quality_scores.csv"
OUT_DIR  = _BASE / "shu_inverse_label/score_charts"

# 4 个主维度
DIMS = [
    ("sim_score",  "相似度"),
    ("aes_score",  "美学质量"),
    ("aud_score",  "音频质量"),
    ("nar_score",  "叙事吸引力"),
]
DIM_KEYS   = [k for k, _ in DIMS]
DIM_LABELS = [l for _, l in DIMS]

# 柱子颜色（对应 4 个维度）
DIM_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

# X 轴位置定义（source, level） → 显示标签
X_SLOTS = [
    ("generated_videos",          1, "首帧+剧本\nL1"),
    ("generated_videos",          2, "首帧+剧本\nL2"),
    ("generated_videos",          3, "首帧+剧本\nL3"),
    ("generated_videos",          4, "首帧+剧本\nL4"),
    ("generated_videos_first_last", 1, "首+尾+剧本\nL1"),
    ("generated_videos_first_last", 2, "首+尾+剧本\nL2"),
    ("generated_videos_first_last", 3, "首+尾+剧本\nL3"),
    ("generated_videos_first_last", 4, "首+尾+剧本\nL4"),
    ("generated_storyboard",      0, "首帧生成\n剧本"),
]

# 来源分组颜色（用于 x 轴标签背景 / 分隔线）
GROUP_COLORS = {
    "generated_videos":           "#EAF2FB",
    "generated_videos_first_last":"#EAF9EA",
    "generated_storyboard":       "#FDF2E9",
}
GROUP_EDGE = {
    "generated_videos":           "#4C72B0",
    "generated_videos_first_last":"#55A868",
    "generated_storyboard":       "#DD8452",
}


# ═══════════════════════ LOAD ════════════════════════════════════════════════

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stem_sort_key(stem: str) -> int:
    m = re.match(r"id_(\d+)", stem)
    return int(m.group(1)) if m else 0


# ═══════════════════════ PLOT ════════════════════════════════════════════════

def plot_stem(stem: str, records: dict[tuple, dict]):
    """
    records: {(source, level): row_dict}
    """
    # 确定本 stem 存在哪些 X 槽
    present_slots = [(src, lv, lbl)
                     for src, lv, lbl in X_SLOTS
                     if (src, lv) in records]
    if not present_slots:
        return

    n_slots = len(present_slots)
    n_dims  = len(DIMS)

    # 分组柱子布局参数
    group_w  = 0.7          # 每个 X 槽的总柱宽
    bar_w    = group_w / n_dims
    offsets  = np.linspace(-(group_w - bar_w) / 2,
                            (group_w - bar_w) / 2, n_dims)

    fig_w = max(10, n_slots * 1.4 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    fig.patch.set_facecolor("white")

    x_pos = np.arange(n_slots)

    # ── 背景色块（按来源分组） ────────────────────────────────────────────────
    prev_src = None
    block_start = 0
    for i, (src, lv, _) in enumerate(present_slots):
        if src != prev_src:
            if prev_src is not None:
                ax.axvspan(block_start - 0.5, i - 0.5,
                           color=GROUP_COLORS[prev_src], alpha=0.35, zorder=0)
            block_start = i
            prev_src = src
    # 最后一个块
    if prev_src is not None:
        ax.axvspan(block_start - 0.5, n_slots - 0.5,
                   color=GROUP_COLORS[prev_src], alpha=0.35, zorder=0)

    # ── 绘制柱子 ──────────────────────────────────────────────────────────────
    for di, (dim_key, dim_label, color) in enumerate(zip(DIM_KEYS, DIM_LABELS, DIM_COLORS)):
        vals = []
        for src, lv, _ in present_slots:
            row = records[(src, lv)]
            try:
                v = float(row.get(dim_key, 0) or 0)
            except ValueError:
                v = 0.0
            vals.append(v)

        bars = ax.bar(
            x_pos + offsets[di], vals,
            width=bar_w,
            color=color, alpha=0.85,
            label=dim_label,
            zorder=2,
        )
        # 在柱顶标注分数
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.12,
                    f"{val:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#333333",
                )

    # ── 分隔线（来源之间） ────────────────────────────────────────────────────
    prev_src = None
    for i, (src, lv, _) in enumerate(present_slots):
        if src != prev_src and prev_src is not None:
            ax.axvline(i - 0.5, color="#aaaaaa", linewidth=1.0,
                       linestyle="--", zorder=1)
        prev_src = src

    # ── 坐标轴 ────────────────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels([lbl for _, _, lbl in present_slots],
                       fontsize=9.5, linespacing=1.4)
    ax.set_ylim(0, 11.5)
    ax.set_yticks(range(0, 11))
    ax.set_ylabel("分数（0–10）", fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ── 图例 ─────────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(DIM_COLORS, DIM_LABELS)]
    ax.legend(handles=handles, loc="upper right",
              fontsize=10, framealpha=0.85)

    # ── 标题 ─────────────────────────────────────────────────────────────────
    ax.set_title(f"视频质量评分对比  |  {stem}",
                 fontsize=13, pad=12, color="#222222")

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / f"{stem}_scores.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows1 = load_csv(SRC1_CSV)
    rows2 = load_csv(SRC2_CSV)
    rows3 = load_csv(SRC3_CSV)
    print(f"载入: SRC1={len(rows1)} SRC2={len(rows2)} SRC3={len(rows3)}")

    # 按 stem 汇总
    stems_data: dict[str, dict[tuple, dict]] = {}
    for rows in (rows1, rows2, rows3):
        for row in rows:
            stem = row.get("vid_label", "")
            if not stem:
                continue
            try:
                lv = int(row.get("prompt_level", 0))
            except ValueError:
                lv = 0
            src = row.get("source", "")
            stems_data.setdefault(stem, {})[(src, lv)] = row

    all_stems = sorted(stems_data.keys(), key=stem_sort_key)
    print(f"共 {len(all_stems)} 个视频索引\n")

    for stem in all_stems:
        print(f"[{stem}]")
        plot_stem(stem, stems_data[stem])

    print(f"\n全部完成。输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
