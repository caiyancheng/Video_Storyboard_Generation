"""
compress_to_gt_profile.py

分析两个视频的差异，然后将 src 视频按照 ref (GT) 的码率/分辨率/fps 重新压缩。

用法：
  # 分析两个视频的差异
  python compress_to_gt_profile.py --analyze SRC REF

  # 压缩单个视频，对齐 GT 配置
  python compress_to_gt_profile.py SRC REF [--output OUT]

  # 批量压缩整个目录下所有 mp4（按 GT 目录找同名 GT 文件）
  python compress_to_gt_profile.py --batch SRC_DIR GT_ROOT [--output-dir OUT_DIR]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ═══════════════════════ PROBE ════════════════════════════════════════════════

def probe(path: Path) -> dict:
    """用 ffprobe 获取视频元信息。"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
    data = json.loads(result.stdout)

    fmt = data["format"]
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio_streams = [s for s in data["streams"] if s["codec_type"] == "audio"]

    # 解析 fps（格式为 "24/1" 或 "62500000/2485621"）
    fps_str = video_stream.get("r_frame_rate", "0/1")
    num, den = map(int, fps_str.split("/"))
    fps = num / den if den else 0

    return {
        "path":      path,
        "size_mb":   round(int(fmt["size"]) / 1024 / 1024, 2),
        "duration":  float(fmt.get("duration", 0)),
        "bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000),
        "codec":     video_stream["codec_name"],
        "width":     video_stream["width"],
        "height":    video_stream["height"],
        "fps":       round(fps, 3),
        "fps_str":   fps_str,
        "pix_fmt":   video_stream.get("pix_fmt", "yuv420p"),
        "has_audio": len(audio_streams) > 0,
        "audio_codec": audio_streams[0]["codec_name"] if audio_streams else None,
        "audio_bitrate_kbps": round(
            int(audio_streams[0].get("bit_rate", 0)) / 1000
        ) if audio_streams else 0,
    }


# ═══════════════════════ ANALYZE ═════════════════════════════════════════════

def analyze(src_info: dict, ref_info: dict):
    """打印两个视频的对比分析。"""
    labels = ["属性", "生成视频 (src)", "GT 参考 (ref)", "差异"]
    rows = [
        ("大小",      f"{src_info['size_mb']} MB",        f"{ref_info['size_mb']} MB",
         f"{src_info['size_mb']/ref_info['size_mb']:.1f}x 更大" if ref_info['size_mb'] else "—"),
        ("码率",      f"{src_info['bitrate_kbps']} kbps", f"{ref_info['bitrate_kbps']} kbps",
         f"{src_info['bitrate_kbps']/max(ref_info['bitrate_kbps'],1):.1f}x 更高"),
        ("分辨率",    f"{src_info['width']}×{src_info['height']}",
                      f"{ref_info['width']}×{ref_info['height']}",
         "不同" if src_info['width'] != ref_info['width'] else "相同"),
        ("时长",      f"{src_info['duration']:.2f}s",     f"{ref_info['duration']:.2f}s",
         f"差 {abs(src_info['duration']-ref_info['duration']):.2f}s"),
        ("FPS",       f"{src_info['fps']}",               f"{ref_info['fps']}",
         "不同" if abs(src_info['fps']-ref_info['fps']) > 0.5 else "相近"),
        ("编码",      src_info['codec'],                  ref_info['codec'],
         "不同" if src_info['codec'] != ref_info['codec'] else "相同"),
        ("像素格式",  src_info['pix_fmt'],                ref_info['pix_fmt'],
         "不同" if src_info['pix_fmt'] != ref_info['pix_fmt'] else "相同"),
        ("音频",      src_info['audio_codec'] or "无",    ref_info['audio_codec'] or "无",
         ""),
    ]

    col_w = [10, 22, 22, 20]
    sep = "─" * sum(col_w + [len(labels)*3])
    print(f"\n{'═'*70}")
    print(f"  src : {src_info['path']}")
    print(f"  ref : {ref_info['path']}")
    print(f"{'═'*70}")
    header = "  ".join(f"{h:<{w}}" for h, w in zip(labels, col_w))
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(f"{v:<{w}}" for v, w in zip(row, col_w)))
    print(sep)
    print()


# ═══════════════════════ COMPRESS ════════════════════════════════════════════

def compress(
    src_path: Path,
    ref_info: dict,
    output_path: Path,
    crf: int = 23,
    match_resolution: bool = True,
    match_fps: bool = False,         # fps 差别不大时不强制转，避免丢帧
    audio_bitrate: str = "128k",
):
    """
    将 src_path 按照 ref_info 的配置重新编码。

    策略：
      - 视频码率：用 CRF 模式（质量恒定），目标文件大小约等于 ref
      - 分辨率：缩放到 ref 的宽高（保持比例，必要时加黑边 pad）
      - FPS：仅在差别 > 1 时才转换
      - 音频：AAC 128k（够用且小）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vf_filters = []

    # 分辨率对齐
    if match_resolution:
        tw, th = ref_info["width"], ref_info["height"]
        # scale + pad 保持比例
        vf_filters.append(
            f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2"
        )

    cmd = ["ffmpeg", "-y", "-i", str(src_path)]

    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    # FPS
    if match_fps and abs(ref_info["fps"] - 0) > 0.1:
        # 用 ref 的 fps 字符串（精确分数），fallback 到近似值
        fps_target = ref_info["fps_str"] if "/" in ref_info["fps_str"] else str(round(ref_info["fps"]))
        cmd += ["-r", fps_target]

    # 视频编码：H.264 + CRF
    cmd += [
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
    ]

    # 音频
    if ref_info["has_audio"]:
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate]
    else:
        cmd += ["-an"]

    cmd += ["-movflags", "+faststart", str(output_path)]

    print(f"  → ffmpeg: {' '.join(cmd[-10:])}")  # 只打印最后几段避免太长
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed:\n{result.stderr[-800:]}")
        return False

    out_info = probe(output_path)
    print(f"  ✓ 压缩完成: {src_path.name}")
    print(f"    原始: {probe(src_path)['size_mb']} MB  →  压缩后: {out_info['size_mb']} MB  "
          f"({out_info['bitrate_kbps']} kbps, {out_info['width']}×{out_info['height']})")
    return True


# ═══════════════════════ BATCH ════════════════════════════════════════════════

def find_gt(gt_root: Path, stem: str) -> Path | None:
    """在 gt_root 子目录中找匹配 stem 的 GT 文件。

    支持两种匹配方向：
      - GT stem 包含在 src stem 中（src 有 _plevelN 后缀等额外后缀）
      - src stem 包含在 GT stem 中（原来的逻辑，通常不需要）
    """
    def match(gt_stem: str) -> bool:
        return gt_stem in stem or stem in gt_stem

    for sub in gt_root.iterdir():
        if not sub.is_dir():
            continue
        for p in sub.glob("*.mp4"):
            if match(p.stem):
                return p
    # 也直接在 gt_root 下找
    for p in gt_root.glob("*.mp4"):
        if match(p.stem):
            return p
    return None


def batch_compress(src_dir: Path, gt_root: Path, output_dir: Path, crf: int = 23):
    """批量压缩 src_dir 下所有 mp4，按同名找 GT 参考。"""
    src_files = sorted(src_dir.glob("*.mp4"))
    print(f"找到 {len(src_files)} 个源视频")

    ok = fail = skip = 0
    for src in src_files:
        gt = find_gt(gt_root, src.stem)
        if gt is None:
            print(f"  [SKIP] 找不到 GT: {src.name}")
            skip += 1
            continue

        out = output_dir / src.name
        if out.exists():
            print(f"  [SKIP] 已存在: {out.name}")
            skip += 1
            continue

        ref_info = probe(gt)
        print(f"\n[{src.name}]  GT={gt.name}")
        success = compress(src, ref_info, out, crf=crf)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\n完成：成功 {ok}，失败 {fail}，跳过 {skip}")


# ═══════════════════════ PRESET PATHS ════════════════════════════════════════

_BASE    = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample")
_GT_ROOT = _BASE / "filter_scored_gt_videos"
_SHU     = _BASE / "shu_inverse_label"

# 三个来源的预设批量任务：(src_dir, output_dir)
PRESET_BATCH_JOBS = (
    # ── 来源1: generated_storyboard/videos ───────────────────────────────────
    [
        (
            _SHU / "generated_storyboard" / "videos",
            _SHU / "generated_storyboard" / "videos_compressed",
        )
    ]
    # ── 来源2: generated_videos / level_{1-4} ────────────────────────────────
    + [
        (
            _SHU / "generated_videos" / f"level_{lv}",
            _SHU / "generated_videos" / f"level_{lv}_compressed",
        )
        for lv in range(1, 5)
    ]
    # ── 来源3: generated_videos_first_last / level_{1-4} ─────────────────────
    + [
        (
            _SHU / "generated_videos_first_last" / f"level_{lv}",
            _SHU / "generated_videos_first_last" / f"level_{lv}_compressed",
        )
        for lv in range(1, 5)
    ]
)


def run_all_presets(crf: int = 23):
    """一键压缩全部三个来源（共 9 个目录）。"""
    print(f"GT 根目录: {_GT_ROOT}")
    print(f"共 {len(PRESET_BATCH_JOBS)} 个目录任务，CRF={crf}\n")

    total_ok = total_fail = total_skip = 0
    for src_dir, out_dir in PRESET_BATCH_JOBS:
        if not src_dir.exists():
            print(f"[SKIP] 目录不存在: {src_dir}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'═'*60}")
        print(f"  src : {src_dir}")
        print(f"  out : {out_dir}")
        print(f"{'═'*60}")
        ok, fail, skip = _batch_compress_counted(src_dir, _GT_ROOT, out_dir, crf)
        total_ok += ok; total_fail += fail; total_skip += skip

    print(f"\n{'═'*60}")
    print(f"全部完成：成功 {total_ok}，失败 {total_fail}，跳过 {total_skip}")


def _batch_compress_counted(
    src_dir: Path, gt_root: Path, output_dir: Path, crf: int
) -> tuple[int, int, int]:
    """batch_compress 的返回计数版本。"""
    src_files = sorted(src_dir.glob("*.mp4"))
    print(f"找到 {len(src_files)} 个源视频")
    ok = fail = skip = 0
    for src in src_files:
        gt = find_gt(gt_root, src.stem)
        if gt is None:
            print(f"  [SKIP] 找不到 GT: {src.name}")
            skip += 1
            continue
        out = output_dir / src.name
        if out.exists():
            print(f"  [SKIP] 已存在: {out.name}")
            skip += 1
            continue
        ref_info = probe(gt)
        print(f"\n[{src.name}]  GT={gt.name}")
        if compress(src, ref_info, out, crf=crf):
            ok += 1
        else:
            fail += 1
    print(f"\n完成：成功 {ok}，失败 {fail}，跳过 {skip}")
    return ok, fail, skip


# ═══════════════════════ ENTRY ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="视频压缩：对齐 GT 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例：
  # 一键压缩全部三个来源（预设路径，无需额外参数）
  python compress_to_gt_profile.py --all

  # 仅分析两个视频差异
  python compress_to_gt_profile.py --analyze SRC.mp4 REF.mp4

  # 压缩单个视频
  python compress_to_gt_profile.py SRC.mp4 REF.mp4 [--output OUT.mp4]

  # 批量压缩自定义目录
  python compress_to_gt_profile.py --batch SRC_DIR GT_ROOT [--output-dir OUT_DIR]

预设路径（--all 模式）：
  GT  : .../filter_scored_gt_videos
  src1: .../generated_storyboard/videos          → videos_compressed
  src2: .../generated_videos/level_{1-4}         → level_{1-4}_compressed
  src3: .../generated_videos_first_last/level_{1-4} → level_{1-4}_compressed
        """,
    )
    parser.add_argument("src", nargs="?", help="源视频路径（单文件模式）")
    parser.add_argument("ref", nargs="?", help="参考 GT 视频路径（单文件模式）")
    parser.add_argument("--output", "-o",
                        help="输出路径（单文件模式，默认加 _compressed 后缀）")
    parser.add_argument("--all", action="store_true",
                        help="一键压缩全部三个来源（使用预设路径）")
    parser.add_argument("--analyze", "-a", nargs=2, metavar=("SRC", "REF"),
                        help="仅分析两个视频差异，不压缩")
    parser.add_argument("--batch", nargs=2, metavar=("SRC_DIR", "GT_ROOT"),
                        help="批量模式：SRC_DIR 下所有 mp4 对齐 GT_ROOT")
    parser.add_argument("--output-dir",
                        help="批量模式输出目录（默认: src_dir 同级加 _compressed）")
    parser.add_argument("--crf", type=int, default=23,
                        help="H.264 CRF（18=高质量, 23=默认, 28=更小）")
    parser.add_argument("--no-resize", action="store_true",
                        help="不改变分辨率，只降码率")
    args = parser.parse_args()

    # ── 一键全量模式 ──────────────────────────────────────────────────────────
    if args.all:
        run_all_presets(crf=args.crf)
        return

    # ── 仅分析模式 ────────────────────────────────────────────────────────────
    if args.analyze:
        src_info = probe(Path(args.analyze[0]))
        ref_info = probe(Path(args.analyze[1]))
        analyze(src_info, ref_info)
        return

    # ── 批量模式 ──────────────────────────────────────────────────────────────
    if args.batch:
        src_dir = Path(args.batch[0])
        gt_root = Path(args.batch[1])
        out_dir = (Path(args.output_dir) if args.output_dir
                   else src_dir.parent / (src_dir.name + "_compressed"))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"批量压缩: {src_dir} → {out_dir}")
        batch_compress(src_dir, gt_root, out_dir, crf=args.crf)
        return

    # ── 单文件模式 ────────────────────────────────────────────────────────────
    if not args.src or not args.ref:
        parser.print_help()
        sys.exit(1)

    src_path = Path(args.src)
    ref_path = Path(args.ref)
    src_info = probe(src_path)
    ref_info = probe(ref_path)
    analyze(src_info, ref_info)

    out_path = Path(args.output) if args.output else src_path.parent / (src_path.stem + "_compressed.mp4")
    print(f"开始压缩 → {out_path}")
    compress(src_path, ref_info, out_path, crf=args.crf, match_resolution=not args.no_resize)


if __name__ == "__main__":
    main()