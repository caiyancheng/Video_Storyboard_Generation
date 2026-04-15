"""
compare_storyboard_videos.py  (本地运行)

遍历 generated_storyboard/videos/ 中的每个生成视频，
找到对应的首帧图片，绘制单行对比图：
  最左列 : 首帧图片（绿色边框，标注 "first frame"）
  中间列 : 生成视频按 1s/帧采样（0s, 1s, 2s, ...）
  最右列 : 生成视频的真实尾帧（紫色边框，标注真实时长）
  顶部   : 时间轴

保存到：
  .../generated_storyboard/comparison/
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

VIDEO_DIR       = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                       "/shu_inverse_label/generated_storyboard/videos")
FIRST_FRAME_DIR = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                       "/first_frame")
OUT_DIR         = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                       "/shu_inverse_label/generated_storyboard/comparison")

# ═══════════════════════ VIDEO UTILS ═════════════════════════════════════════

def get_video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps


def extract_frame_at(path: Path, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_last_frame(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_image_rgb(path: Path) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def make_placeholder(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 220, dtype=np.uint8)


# ═══════════════════════ LOOKUP ══════════════════════════════════════════════

def find_first_frame(stem: str) -> Path | None:
    """根据视频 stem 在 FIRST_FRAME_DIR 中找首帧图片。"""
    p = FIRST_FRAME_DIR / f"{stem}_first.png"
    if p.exists():
        return p
    # fallback：提取 video_id 部分做 glob
    parts = stem.split("_", 3)
    if len(parts) >= 4:
        video_id = parts[3]
        matches = list(FIRST_FRAME_DIR.glob(f"*{video_id}_first.png"))
        if matches:
            return matches[0]
    return None


# ═══════════════════════ PLOT ════════════════════════════════════════════════

def build_comparison(video_path: Path):
    stem = video_path.stem          # e.g. id_0017_score4_v09044010000...
    out_path = OUT_DIR / f"{stem}_comparison.png"
    if out_path.exists():
        print(f"  [SKIP] already exists: {out_path.name}")
        return

    # ── 首帧图片 ──────────────────────────────────────────────────────────────
    ff_path  = find_first_frame(stem)
    ff_image = load_image_rgb(ff_path) if ff_path else None
    if ff_image is None:
        print(f"  [WARN] first frame not found for {stem}")

    # ── 生成视频采样 ──────────────────────────────────────────────────────────
    duration  = get_video_duration(video_path)
    num_cols  = max(1, int(duration))        # 1fps 列数（0s … (num_cols-1)s）
    frames_1fps = [extract_frame_at(video_path, t) for t in range(num_cols)]
    last_frame  = extract_last_frame(video_path)
    print(f"  video: {video_path.name}  dur={duration:.2f}s  cols={num_cols}")

    # ── 确定帧尺寸 ────────────────────────────────────────────────────────────
    ref = next((f for f in [ff_image] + frames_1fps + [last_frame] if f is not None), None)
    if ref is None:
        print(f"  [ERROR] no valid frames at all, skip.")
        return
    fh, fw = ref.shape[:2]

    # ── 布局计算 ──────────────────────────────────────────────────────────────
    # 列结构：[首帧] [0s][1s]...[N-1s] [last]
    # GridSpec 列：col0=首帧占位符(width_ratio=0用于左侧留白),
    #              col1=首帧图, col2..col1+num_cols=1fps帧, col_last=尾帧
    frame_w_inch = 1.4
    frame_h_inch = frame_w_inch * fh / fw
    time_row_h   = 0.60
    top_pad      = 0.80
    bot_pad      = 0.30
    left_pad_w   = 0.15   # 左侧少量留白（无行标签）

    # 总列数：1(首帧) + num_cols(1fps) + 1(尾帧)
    total_display_cols = 1 + num_cols + 1

    fig_w = left_pad_w + total_display_cols * frame_w_inch
    fig_h = top_pad + time_row_h + frame_h_inch + bot_pad

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor("white")

    # GridSpec: row0=时间轴, row1=视频帧行
    # cols: col0=首帧, col1..num_cols=1fps帧, col_{num_cols+1}=尾帧
    gs = fig.add_gridspec(
        2, total_display_cols,
        left   = left_pad_w / fig_w,
        right  = 1.0 - 0.01,
        top    = 1.0 - top_pad / fig_h,
        bottom = bot_pad / fig_h,
        wspace = 0.03,
        hspace = 0.03,
        width_ratios  = [1] + [1] * num_cols + [1],
        height_ratios = [0.22, 1],
    )

    # ── 时间轴 row ────────────────────────────────────────────────────────────
    # 首帧列 header
    ax_ff_hdr = fig.add_subplot(gs[0, 0])
    ax_ff_hdr.set_xlim(0, 1); ax_ff_hdr.set_ylim(0, 1)
    ax_ff_hdr.axis("off")
    ax_ff_hdr.text(0.5, 0.5, "first\nframe",
                   ha="center", va="center",
                   fontsize=13, color="#27ae60", fontweight="bold")

    # 1fps 列 headers
    for c in range(num_cols):
        ax = fig.add_subplot(gs[0, c + 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.5, f"{c}s", ha="center", va="center",
                fontsize=15, color="#333333", fontweight="bold")
        ax.axvline(0.5, ymin=0, ymax=0.3, color="#888", linewidth=0.8)

    # 尾帧列 header
    ax_last_hdr = fig.add_subplot(gs[0, num_cols + 1])
    ax_last_hdr.set_xlim(0, 1); ax_last_hdr.set_ylim(0, 1)
    ax_last_hdr.axis("off")
    ax_last_hdr.text(0.5, 0.5, "last",
                     ha="center", va="center",
                     fontsize=15, color="#8e44ad", fontweight="bold")

    # 连续时间条（仅覆盖 1fps 列）
    ax_bar = fig.add_subplot(gs[0, 1 : num_cols + 1])
    ax_bar.set_xlim(0, num_cols); ax_bar.set_ylim(0, 1)
    ax_bar.axhline(0.15, color="#aaaaaa", linewidth=1.0, zorder=0)
    ax_bar.axis("off")

    # ── 首帧图格 ──────────────────────────────────────────────────────────────
    ax_ff = fig.add_subplot(gs[1, 0])
    if ff_image is not None:
        ax_ff.imshow(ff_image, aspect="auto")
    else:
        ax_ff.imshow(make_placeholder(fh, fw), aspect="auto")
        ax_ff.text(0.5, 0.5, "N/A", transform=ax_ff.transAxes,
                   ha="center", va="center", fontsize=12, color="#999")
    ax_ff.axis("off")
    for spine in ax_ff.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#27ae60")   # 绿色边框区分首帧

    # ── 1fps 帧格 ─────────────────────────────────────────────────────────────
    for c in range(num_cols):
        ax = fig.add_subplot(gs[1, c + 1])
        frame = frames_1fps[c] if c < len(frames_1fps) else None
        if frame is not None:
            ax.imshow(frame, aspect="auto")
        else:
            ax.imshow(make_placeholder(fh, fw), aspect="auto")
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#999")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_edgecolor("#cccccc")

    # ── 尾帧格 ────────────────────────────────────────────────────────────────
    ax_last = fig.add_subplot(gs[1, num_cols + 1])
    if last_frame is not None:
        ax_last.imshow(last_frame, aspect="auto")
        ax_last.text(0.98, 0.02, f"{duration:.1f}s",
                     transform=ax_last.transAxes,
                     ha="right", va="bottom",
                     fontsize=12, color="white",
                     bbox=dict(boxstyle="round,pad=0.15",
                               fc="#8e44ad", alpha=0.75, lw=0))
    else:
        ax_last.imshow(make_placeholder(fh, fw), aspect="auto")
        ax_last.text(0.5, 0.5, "N/A", transform=ax_last.transAxes,
                     ha="center", va="center", fontsize=12, color="#999")
    ax_last.axis("off")
    for spine in ax_last.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("#8e44ad")   # 紫色边框区分尾帧

    # ── 标题 ──────────────────────────────────────────────────────────────────
    fig.suptitle(f"Storyboard Generation  |  {stem}",
                 fontsize=17, y=1.0 - top_pad / fig_h / 2,
                 ha="center", color="#222222")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════ ENTRY ═══════════════════════════════════════════════

if __name__ == "__main__":
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print(f"[ERROR] No mp4 files found in {VIDEO_DIR}")
        sys.exit(1)

    print(f"Found {len(videos)} videos in {VIDEO_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {vp.name}")
        build_comparison(vp)

    print(f"\nAll done. Output dir: {OUT_DIR}")
