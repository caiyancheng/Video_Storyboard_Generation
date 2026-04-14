"""
generate_storyboard_from_image.py  (本地运行)

给定一张输入图片，使用 Gemini 2.5 生成 Level-4 精度的视频故事板 prompt，
并保存为 txt 文件。

输出目录：
  /Users/bytedance/Datasets/.../generated_storyboard/
    {image_stem}.txt

用法：
  python generate_storyboard_from_image.py --image /path/to/image.jpg
  python generate_storyboard_from_image.py --image /path/to/image.jpg --out-dir /custom/path
  python generate_storyboard_from_image.py --image /path/to/image.jpg --overwrite
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# ── 加入项目根目录 ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 加入 editing_magic_prompt ────────────────────────────────────────────────
_EMP_ROOT = Path("/Users/bytedance/Py_codes/editing_magic_prompt")
if str(_EMP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EMP_ROOT))

from editing_magic_prompt.modules.gpt_caller import GEMINI_MODEL, get_gpt_resp
from Storyboard_generation.storyboard_prompt_template import STORYBOARD_SYSTEM_PROMPT

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

DEFAULT_OUT_DIR = Path(
    "/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/generated_videos/generated_storyboard"
)


# ═══════════════════════ GEMINI CALL ═════════════════════════════════════════

def generate_storyboard_prompt(image_path: Path) -> str | None:
    """
    调用 Gemini 2.5，分析图片并生成 Level 4 故事板 prompt 文本。
    """
    print(f"  → 调用 Gemini 2.5 分析图片: {image_path.name}")

    result = get_gpt_resp(
        model=GEMINI_MODEL,
        prompt=STORYBOARD_SYSTEM_PROMPT,
        image_path=str(image_path),
        enable_thinking=False,
        max_tokens=8192,
    )

    if result is None:
        print("  ✗ Gemini 返回 None")
        return None

    text, cot, usage = result
    print(f"  ✓ Gemini 完成  tokens: prompt={usage['prompt_tokens']}  "
          f"output={usage['completion_tokens']}")
    return text


# ═══════════════════════ POST-PROCESS ════════════════════════════════════════

def clean_response(text: str) -> str:
    """
    清理 Gemini 输出：去除多余的 markdown 代码块标记。
    """
    text = text.strip()
    # 去掉开头的 ```...``` 包装（若有）
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉第一行（```或```text等）
        lines = lines[1:]
        # 去掉最后一行如果是 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def run(image_path: Path, out_dir: Path, overwrite: bool) -> Path | None:
    """
    主流程：图片 → Gemini → 保存 txt。
    返回保存的 txt 路径（或 None 失败时）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}.txt"

    if out_path.exists() and not overwrite:
        print(f"[SKIP] 已存在（使用 --overwrite 强制重新生成）: {out_path}")
        return out_path

    if not image_path.exists():
        print(f"[ERROR] 图片文件不存在: {image_path}")
        return None

    print(f"\n{'='*60}")
    print(f"图片  : {image_path}")
    print(f"输出  : {out_path}")
    print(f"时间  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    raw_text = generate_storyboard_prompt(image_path)
    if raw_text is None:
        return None

    prompt_text = clean_response(raw_text)

    out_path.write_text(prompt_text, encoding="utf-8")
    print(f"\n  ✓ 故事板已保存: {out_path}")
    print(f"\n{'─'*60}")
    print("Prompt 预览（前 500 字符）：")
    print(prompt_text[:500])
    print("...")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="使用 Gemini 2.5 从单张图片生成 Level-4 视频故事板 prompt"
    )
    parser.add_argument(
        "--image", required=True,
        help="输入图片路径（.jpg / .png / .webp 等）"
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help=f"输出目录（默认：{DEFAULT_OUT_DIR}）"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="若 txt 已存在则强制重新生成"
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_dir    = Path(args.out_dir).expanduser().resolve()

    result = run(image_path, out_dir, overwrite=args.overwrite)

    if result:
        print(f"\n完成。txt 路径：{result}")
    else:
        print("\n失败。")
        sys.exit(1)


if __name__ == "__main__":
    main()
