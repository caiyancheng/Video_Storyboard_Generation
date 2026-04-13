"""
compare_video_levels.py

给定一个视频文件名索引（如 id_0003_score5_v090443f0000bkgjtldbdtei4u0auogg），
生成一张对比子图：
  行 0  : Ground Truth
  行 1-4: Prompt Level 1-4（Seedance 生成）
  列    : 按 1s/帧采样（GT 视频时长决定列数）
  顶部  : 时间轴
  左侧  : 行标签

保存到：
  .../generated_videos/comparison_plevel_seedance_GenAI/{stem}.png
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

GT_ROOT   = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                 "/filter_scored_gt_videos")
GEN_ROOT  = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                 "/shu_inverse_label/generated_videos")
OUT_DIR   = GEN_ROOT / "comparison_plevel_seedance_GenAI"

ROW_LABELS = [
    "Ground Truth",
    "Prompt Level 1\n(minimal)",
    "Prompt Level 2\n(standard)",
    "Prompt Level 3\n(detailed)",
    "Prompt Level 4\n(full)",
]
NUM_LEVELS = 4


# ═══════════════════════ VIDEO UTILS ═════════════════════════════════════════

def get_video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps


def extract_frame_at(path: Path, t_sec: float) -> np.ndarray | None:
    """Extract one frame at time t_sec (seconds). Returns RGB array or None."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_frames_1fps(path: Path, num_cols: int) -> list[np.ndarray | None]:
    """Extract frames at t=0,1,2,...,num_cols-1 seconds."""
    return [extract_frame_at(path, t) for t in range(num_cols)]


# ═══════════════════════ FILE LOOKUP ═════════════════════════════════════════

def find_gt_video(stem: str) -> Path | None:
    """Search score4/ and score5/ for a file containing the stem."""
    for score_dir in GT_ROOT.iterdir():
        if not score_dir.is_dir():
            continue
        for p in score_dir.glob("*.mp4"):
            if stem in p.stem:
                return p
    return None


def find_gen_video(stem: str, level: int) -> Path | None:
    """Look for stem + _plevel{level}.mp4 in generated_videos/level_{level}/."""
    level_dir = GEN_ROOT / f"level_{level}"
    if not level_dir.exists():
        return None
    for p in level_dir.glob("*.mp4"):
        if stem in p.stem:
            return p
    return None


# ═══════════════════════ PLOT ════════════════════════════════════════════════

PLACEHOLDER_COLOR = (220, 220, 220)   # light grey for missing frames


def make_placeholder(h: int, w: int) -> np.ndarray:
    img = np.full((h, w, 3), PLACEHOLDER_COLOR, dtype=np.uint8)
    return img


def build_comparison(stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Find GT video & determine columns ────────────────────────────────────
    gt_path = find_gt_video(stem)
    if gt_path is None:
        print(f"[ERROR] GT video not found for stem: {stem}")
        return

    duration  = get_video_duration(gt_path)
    num_cols  = max(1, int(duration))          # 1 col per second
    print(f"GT: {gt_path.name}  duration={duration:.2f}s  cols={num_cols}")

    # ── Collect all video paths ───────────────────────────────────────────────
    video_paths: list[Path | None] = [gt_path]
    for lv in range(1, NUM_LEVELS + 1):
        p = find_gen_video(stem, lv)
        if p:
            print(f"  Level {lv}: {p.name}")
        else:
            print(f"  Level {lv}: NOT FOUND")
        video_paths.append(p)

    num_rows = len(video_paths)   # 5

    # ── Extract all frames ────────────────────────────────────────────────────
    all_frames: list[list[np.ndarray | None]] = []
    for vpath in video_paths:
        if vpath is None:
            all_frames.append([None] * num_cols)
        else:
            all_frames.append(extract_frames_1fps(vpath, num_cols))

    # Determine canonical frame size from first valid frame
    ref_frame = next(
        (f for row in all_frames for f in row if f is not None), None
    )
    if ref_frame is None:
        print("[ERROR] No valid frames found at all.")
        return
    fh, fw = ref_frame.shape[:2]

    # ── Build figure ──────────────────────────────────────────────────────────
    label_col_w  = 1.6   # inches for row-label column
    frame_w_inch = 1.4   # inches per video frame cell
    frame_h_inch = frame_w_inch * fh / fw
    time_row_h   = 0.35  # inches for time-axis row
    top_pad      = 0.5
    bot_pad      = 0.3

    fig_w = label_col_w + num_cols * frame_w_inch
    fig_h = top_pad + time_row_h + num_rows * frame_h_inch + bot_pad

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor("white")

    # GridSpec: row 0 = time axis, rows 1..num_rows = video rows
    # cols: col 0 = labels, cols 1..num_cols = frames
    total_rows = num_rows + 1
    total_cols = num_cols + 1

    gs = fig.add_gridspec(
        total_rows, total_cols,
        left   = label_col_w / fig_w,
        right  = 1.0 - 0.02,
        top    = 1.0 - top_pad / fig_h,
        bottom = bot_pad / fig_h,
        wspace = 0.03,
        hspace = 0.03,
        width_ratios  = [0] + [1] * num_cols,   # col 0 placeholder (labels drawn via fig.text)
        height_ratios = [0.25] + [1] * num_rows,
    )

    # ── Time axis (row 0) ────────────────────────────────────────────────────
    for c in range(num_cols):
        ax = fig.add_subplot(gs[0, c + 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.5, f"{c}s", ha="center", va="center",
                fontsize=7.5, color="#333333", fontweight="bold")
        # tick line
        ax.axvline(0.5, ymin=0, ymax=0.3, color="#888", linewidth=0.8)

    # Continuous time bar
    ax_bar = fig.add_subplot(gs[0, 1:])
    ax_bar.set_xlim(0, num_cols); ax_bar.set_ylim(0, 1)
    ax_bar.axhline(0.15, color="#aaaaaa", linewidth=1.0, zorder=0)
    ax_bar.axis("off")

    # ── Video frame cells ────────────────────────────────────────────────────
    row_label_x = label_col_w / fig_w - 0.01   # normalized fig x for labels

    for r in range(num_rows):
        frames = all_frames[r]

        # Row label (drawn in figure coordinates)
        y_top    = 1.0 - top_pad / fig_h - time_row_h / fig_h
        cell_h   = (1.0 - top_pad / fig_h - time_row_h / fig_h - bot_pad / fig_h) / num_rows
        y_center = y_top - (r + 0.5) * cell_h

        color = "#c0392b" if r == 0 else f"#2471a3"
        fig.text(row_label_x, y_center, ROW_LABELS[r],
                 ha="right", va="center",
                 fontsize=13, color=color,
                 fontweight="bold" if r == 0 else "normal",
                 wrap=True)

        # Horizontal divider between GT and generated
        if r == 1:
            line_y = y_top - r * cell_h
            fig.add_artist(
                plt.Line2D([label_col_w / fig_w, 0.98], [line_y, line_y],
                           transform=fig.transFigure,
                           color="#888888", linewidth=0.8, linestyle="--")
            )

        row_missing = video_paths[r] is None   # entire row: video file not found

        for c in range(num_cols):
            ax = fig.add_subplot(gs[r + 1, c + 1])

            if row_missing:
                # Whole row grey + "Video Not Found" text in centre cell only
                ax.set_facecolor("#d5d8dc")
                ax.axis("off")
                if c == num_cols // 2:
                    ax.text(0.5, 0.5, "Video Not Found",
                            transform=ax.transAxes,
                            ha="center", va="center",
                            fontsize=7, color="#555555", style="italic")
            else:
                frame = frames[c] if c < len(frames) else None
                if frame is not None:
                    ax.imshow(frame, aspect="auto")
                else:
                    ph = make_placeholder(fh, fw)
                    ax.imshow(ph, aspect="auto")
                    ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                            ha="center", va="center", fontsize=6, color="#999")
                ax.axis("off")

            # Thin border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_edgecolor("#cccccc" if not row_missing else "#aaaaaa")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(f"Seedance Generation Comparison  |  {stem}",
                 fontsize=9, y=1.0 - top_pad / fig_h / 2,
                 ha="center", color="#222222")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#c0392b", label="Ground Truth"),
        mpatches.Patch(color="#2471a3", label="Seedance Generated (Level 1-4)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / f"{stem}_comparison.png"
    fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ═══════════════════════ BATCH ═══════════════════════════════════════════════

def collect_all_stems() -> list[str]:
    """Collect stems from all GT videos across score sub-dirs."""
    stems = []
    for score_dir in sorted(GT_ROOT.iterdir()):
        if not score_dir.is_dir():
            continue
        for p in sorted(score_dir.glob("*.mp4")):
            # stem = filename without extension
            stems.append(p.stem)
    return stems


# ═══════════════════════ ENTRY ═══════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].strip().lower() != "all":
        # Single stem mode
        stems = [sys.argv[1].strip()]
        print(f"Single mode: {stems[0]}")
    else:
        # Batch mode: process all GT videos
        stems = collect_all_stems()
        print(f"Batch mode: {len(stems)} GT videos found in {GT_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(stems)
    for i, stem in enumerate(stems, 1):
        out_path = OUT_DIR / f"{stem}_comparison.png"
        if out_path.exists():
            print(f"[{i}/{total}] SKIP (exists): {stem}")
            continue
        print(f"\n[{i}/{total}] Processing: {stem}")
        build_comparison(stem)

    print(f"\nAll done. Output dir: {OUT_DIR}")
