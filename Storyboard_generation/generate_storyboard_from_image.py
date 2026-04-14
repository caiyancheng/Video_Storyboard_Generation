"""
generate_storyboard_from_image.py  (服务器运行)

遍历评分 4/5 分的视频，读取其对应的首帧图片，
使用 Gemini 2.5 分析首帧，生成 Level-4 精度的视频故事板 prompt，
保存为 txt 文件。

首帧来源（本地服务器路径，不走 TOS）：
  FIRST_FRAME_DIR/{stem}_first.png

输出：
  STORYBOARD_OUT_DIR/{stem}.txt

用法：
  python generate_storyboard_from_image.py              # 全量
  python generate_storyboard_from_image.py --overwrite  # 重新生成已有 txt
"""

import json
import sys
import argparse
from pathlib import Path

from tqdm import tqdm

# ── 项目根目录 ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── editing_magic_prompt（服务器路径）────────────────────────────────────────
_EMP_ROOT = Path("/mnt/bn/yilin4/yancheng/Py_codes/editing_magic_prompt")
if str(_EMP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EMP_ROOT))

from editing_magic_prompt.modules.gpt_caller import GEMINI_MODEL, get_gpt_resp
from Storyboard_generation.storyboard_prompt_template import STORYBOARD_SYSTEM_PROMPT

GEMINI_API_KEY = "HqvkgdAdyXd5TMfhXjFLp4JncRWeEMvW"

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

SCORED_FILE = Path(
    "/mnt/bn/yilin4/yancheng/Datasets/"
    "tt_template_hq_publish_data_1400k_USAU"
    ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl"
)

FIRST_FRAME_DIR = Path(
    "/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
    "/first_frame"
)

STORYBOARD_OUT_DIR = Path(
    "/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/generated_storyboard"
)

TARGET_SCORES = {4, 5}


# ═══════════════════════ DATA ════════════════════════════════════════════════

def load_scored_records() -> list[dict]:
    records = []
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("_score") in TARGET_SCORES:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def find_first_frame(stem: str) -> Path | None:
    """在 FIRST_FRAME_DIR 中查找 {stem}_first.png。"""
    p = FIRST_FRAME_DIR / f"{stem}_first.png"
    if p.exists():
        return p
    # fallback: glob by video_id part
    video_id = stem.split("_", 3)[-1] if stem.count("_") >= 3 else stem
    matches = list(FIRST_FRAME_DIR.glob(f"*{video_id}_first.png"))
    return matches[0] if matches else None


# ═══════════════════════ GEMINI ══════════════════════════════════════════════

def generate_storyboard_prompt(image_path: Path) -> str | None:
    """调用 Gemini 2.5，分析首帧图片，输出 Level-4 故事板 prompt。"""
    result = get_gpt_resp(
        model=GEMINI_MODEL,
        prompt=STORYBOARD_SYSTEM_PROMPT,
        image_path=str(image_path),
        api_key=GEMINI_API_KEY,
        enable_thinking=False,
        max_tokens=8192,
    )
    if result is None:
        return None
    text, cot, usage = result
    tqdm.write(f"    tokens: prompt={usage['prompt_tokens']}  "
               f"output={usage['completion_tokens']}")
    return text


def clean_response(text: str) -> str:
    """去除 Gemini 输出中多余的 markdown 代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="批量从首帧图片生成 Level-4 视频故事板 prompt（Gemini 2.5）"
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="重新生成已存在的 txt 文件")
    args = parser.parse_args()

    STORYBOARD_OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_scored_records()
    print(f"找到 {len(records)} 条评分 {sorted(TARGET_SCORES)} 分记录")
    print(f"首帧目录  : {FIRST_FRAME_DIR}")
    print(f"输出目录  : {STORYBOARD_OUT_DIR}\n")

    ok = skip = fail_no_frame = fail_gemini = 0

    for rec in tqdm(records, desc="生成 storyboard"):
        video_id  = rec.get("video_id", "unknown")
        rec_idx   = rec.get("_idx",   0)
        rec_score = rec.get("_score", 0)
        stem      = f"id_{rec_idx:04d}_score{rec_score}_{video_id}"

        out_path = STORYBOARD_OUT_DIR / f"{stem}.txt"

        if out_path.exists() and not args.overwrite:
            skip += 1
            continue

        # 找首帧
        frame_path = find_first_frame(stem)
        if frame_path is None:
            tqdm.write(f"  [SKIP] 首帧不存在: {stem}_first.png")
            fail_no_frame += 1
            continue

        tqdm.write(f"\n[{stem}]  首帧: {frame_path.name}")

        # Gemini 生成
        raw = generate_storyboard_prompt(frame_path)
        if raw is None:
            tqdm.write(f"  [FAIL] Gemini 返回 None")
            fail_gemini += 1
            continue

        prompt_text = clean_response(raw)
        out_path.write_text(prompt_text, encoding="utf-8")
        tqdm.write(f"  ✓ 已保存: {out_path.name}")
        ok += 1

    print(f"\n完成：生成 {ok} 个，跳过(已存在) {skip} 个，"
          f"首帧缺失 {fail_no_frame} 个，Gemini 失败 {fail_gemini} 个")
    print(f"输出目录: {STORYBOARD_OUT_DIR}")


if __name__ == "__main__":
    main()
