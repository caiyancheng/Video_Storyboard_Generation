"""
test_storyboard_seedance_2_0_it2v_filter_15s_data_pe_analyze.py

对每个视频的每个 prompt level，调用 Seedance PE（Prompt Enhancement）API，
只保存原始 prompt 和 PE 增强后的结果，不生成视频。

Pipeline：
  1. 加载 scored JSONL → 找 label 文件 → 构建 prompt（同 compress 版）
  2. 找到对应的本地 first-frame PNG
  3. 调用 tt_seedance2_pe_test API（参考 submit_sep_pe_dit.py）
  4. 解析 pe_prompt，写入 CSV（一行 = 一个 vid × level）

输出：
  OUT_ROOT/pe_results.csv  —— 每行含 original_prompt + pe_prompt_json
  OUT_ROOT/logs/           —— per-video 日志
"""

import csv
import json
import logging
import time
from pathlib import Path

import requests

# ═══════════════════════════ CONFIG ══════════════════════════════════════════

PROMPT_LEVELS = [1, 2, 3, 4]

SCORED_FILE = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_hq_publish_data_1400k_USAU"
                   ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl")

LABEL_ROOT   = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
                    "/shu_inverse_label")
PHASE0_DIR   = LABEL_ROOT / "phase0_chinese_labels"
PHASE1_DIR   = LABEL_ROOT / "phase1_chinese_labels"
PHASE1_5_DIR = LABEL_ROOT / "phase1_5_chinese_labels"
FIRST_FRAME_DIR = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
                       "/first_frame")

OUT_ROOT = LABEL_ROOT / "pe_analyze"
LOG_DIR  = OUT_ROOT / "logs"
CSV_PATH = OUT_ROOT / "pe_results.csv"

CSV_FIELDS = [
    "vid_label", "video_id", "idx", "score", "prompt_level",
    "duration", "prompt_words",
    "original_prompt", "pe_prompt_json",
    "vivid_instruction", "vivid_instruction_chars",
    "pe_at",
]

TARGET_SCORES = {4, 5}

# PE API（参考 submit_sep_pe_dit.py）
PE_URL        = "https://api2.musical.ly/media/api/pic/afr"
PE_ALGORITHMS = "tt_seedance2_pe_test"
PE_ASPECT     = "9:16"
PE_LANGUAGE   = "zh"

# 版权安全声明（与 compress 版保持一致）
COPYRIGHT_NOTICE = (
    "\n\n[Copyright & Safety Notice] "
    "Do NOT use any celebrity likeness, public figure appearance, or real person identity. "
    "Do NOT use any famous, recognizable, or copyrighted music, song lyrics, or melodies. "
    "Use only original, royalty-free audio and avoid depicting any real famous individuals."
)


# ═══════════════════════════ DATA LOADING ════════════════════════════════════

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


def find_label_files(video_id: str, idx: int, score: int):
    stem = f"id_{idx:04d}_score{score}_{video_id}"
    p0   = PHASE0_DIR   / f"{stem}_phase0.json"
    p1   = PHASE1_DIR   / f"{stem}_phase1.json"
    p15  = PHASE1_5_DIR / f"{stem}_phase1_5.json"
    if p0.exists() and p1.exists() and p15.exists():
        return p0, p1, p15
    matches0  = list(PHASE0_DIR.glob(f"*{video_id}_phase0.json"))
    matches1  = list(PHASE1_DIR.glob(f"*{video_id}_phase1.json"))
    matches15 = list(PHASE1_5_DIR.glob(f"*{video_id}_phase1_5.json"))
    if matches0 and matches1 and matches15:
        return matches0[0], matches1[0], matches15[0]
    return None, None, None


def find_first_frame(video_id: str, idx: int, score: int) -> Path | None:
    stem = f"id_{idx:04d}_score{score}_{video_id}_first.png"
    p = FIRST_FRAME_DIR / stem
    if p.exists():
        return p
    matches = list(FIRST_FRAME_DIR.glob(f"*{video_id}_first.png"))
    return matches[0] if matches else None


# ═══════════════════════════ PROMPT BUILDING ═════════════════════════════════

def parse_time_range(time_range: str) -> tuple[float, float]:
    start_str, end_str = time_range.split('-')
    def to_sec(ts):
        mm, ss = ts.split(':')
        return int(mm) * 60 + float(ss)
    return to_sec(start_str), to_sec(end_str)


def build_whole_video_prompt(phase0: dict, phase1: dict, phase1_5: dict,
                             level: int) -> tuple[str, float]:
    """返回 (prompt_text, duration_seconds)，prompt 末尾含版权声明。"""
    audio_map    = {a['id']: a for a in phase0.get('audio_registry',    [])}
    subject_map  = {s['id']: s for s in phase0.get('subject_registry',  [])}
    prop_map     = {p['id']: p for p in phase0.get('prop_registry',     [])}
    location_map = {l['id']: l for l in phase0.get('location_registry', [])}
    shots_p0     = phase0.get('shot_registry', [])
    scene        = (phase0.get('scene_registry') or [{}])[0]

    phase1_shot_map  = {s['id']: s for s in phase1.get('shot_registry', [])}
    phase15_shot_map = {s['shot_id']: s for s in phase1_5.get('shot_in_scene_registry', [])}

    last_shot = shots_p0[-1] if shots_p0 else {}
    _, end_sec = parse_time_range(last_shot.get('time_range', '00:00.000-00:00.000'))
    duration = round(max(4.0, min(15.0, end_sec)), 3)

    parts = []

    scene_caption = scene.get('caption', '')
    parts.append(
        f"[Whole-video generation | {duration}s | Level {level}]\n"
        f"Scene overview: {scene_caption}"
    )

    if subject_map:
        lines = ["Characters:"]
        for sid, s in subject_map.items():
            line = f"  {sid} ({s.get('name', '')})"
            if level >= 2:
                line += f": {s.get('visual_features', '')}"
            if level >= 3 and s.get('rationale'):
                line += f" — Role: {s['rationale']}"
            lines.append(line)
        parts.append('\n'.join(lines))

    if location_map:
        lines = ["Locations:"]
        for lid, loc in location_map.items():
            line = f"  {lid} ({loc.get('name', '')})"
            if level >= 3:
                line += f": {loc.get('visual_features', '')}"
            lines.append(line)
        parts.append('\n'.join(lines))

    if level >= 2 and audio_map:
        lines = ["Audio:"]
        for aid, a in audio_map.items():
            line = f"  {a.get('name', '')} [{a.get('type', '')}]"
            if a.get('style'):
                line += f" — {a['style']}"
            lines.append(line)
        parts.append('\n'.join(lines))

    if level >= 2:
        lines = ["Shot flow (chronological):"]
        for shot in shots_p0:
            sid  = shot['id']
            _, end   = parse_time_range(shot['time_range'])
            _, start = parse_time_range(shot['time_range'])
            dur_s = round(end - start, 2)
            p1s   = phase1_shot_map.get(sid, {})
            cam   = p1s.get('visual', {}).get('camera', {})
            p15s  = phase15_shot_map.get(sid, {})
            line  = f"  {shot['time_range']} ({dur_s}s): {shot.get('caption', '')}"
            if level >= 3:
                mood   = cam.get('mood_atmosphere', '')
                beat   = (p15s.get('role_in_scene') or {}).get('beat_description', '')
                extras = []
                if mood: extras.append(f"mood={mood}")
                if beat: extras.append(f"beat={beat}")
                if extras:
                    line += f"\n    [{'; '.join(extras)}]"
            if level >= 4:
                dense = p1s.get('dense_caption', '')
                if dense:
                    line += f"\n    Dense: {dense}"
                cont    = p15s.get('continuity_logic', {})
                to_next = cont.get('to_next_shot', {}).get('relation', '')
                if to_next:
                    line += f"\n    → next: {to_next}"
                for it in p1s.get('visual', {}).get('interaction_tracking', []):
                    line += f"\n    interact: {it.get('interaction', '')}"
            lines.append(line)
        parts.append('\n'.join(lines))

    if level >= 3:
        arc_parts = []
        for shot in shots_p0:
            sid  = shot['id']
            p15s = phase15_shot_map.get(sid, {})
            ep   = p15s.get('scene_contribution', {}).get('emotion_pacing', '')
            if ep:
                arc_parts.append(f"{shot['time_range']}: {ep}")
        if arc_parts:
            parts.append("Emotional arc:\n" + '\n'.join(f"  {a}" for a in arc_parts))

    if level >= 3 and prop_map:
        lines = ["Key props:"]
        for pid, p in prop_map.items():
            lines.append(f"  {pid} ({p.get('name', '')}): {p.get('visual_features', '')}")
        parts.append('\n'.join(lines))

    parts.append(
        "Reference image: The provided image is the FIRST FRAME of the entire video. "
        "Use it as a precise visual anchor for scene composition, character appearance, "
        "lighting, and setting throughout the whole video."
    )

    prompt = '\n\n'.join(parts) + COPYRIGHT_NOTICE
    return prompt, duration


# ═══════════════════════════ PE API ══════════════════════════════════════════

def call_pe(
    prompt: str,
    duration: float,
    first_frame_path: Path,
    logger: logging.Logger,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> dict | None:
    """
    调用 tt_seedance2_pe_test PE API，返回解析后的 pe_prompt dict，失败返回 None。
    参考 submit_sep_pe_dit.py 的请求结构。
    """
    conf = {
        "prompt":          prompt,
        "duration":        duration,
        "aspect_ratio":    PE_ASPECT,
        "language":        PE_LANGUAGE,
        "with_audio":      True,
        "version":         "v2.0",
        "binary_var_name": ["image"],
        "task_type":       "R2V",
    }

    for attempt in range(retries):
        try:
            with first_frame_path.open("rb") as img_f:
                response = requests.post(
                    PE_URL,
                    data={
                        "algorithms":     PE_ALGORITHMS,
                        "conf":           json.dumps(conf, ensure_ascii=False),
                        "input_img_type": "multiple_files",
                    },
                    files=[
                        ("files[]", ("image.png", img_f, "application/octet-stream"))
                    ],
                    timeout=180,   # PE 处理视频较慢，允许等 3 分钟
                )
            response.raise_for_status()
            raw = response.json()

            # 解析 PE 输出（结构：data.afr_data[0].pic_conf → JSON string）
            pe_prompt = json.loads(raw["data"]["afr_data"][0]["pic_conf"])
            logger.info(f"PE OK: {str(pe_prompt)[:200]}")
            return pe_prompt

        except Exception as e:
            logger.warning(f"PE attempt {attempt + 1}/{retries} failed: {e}")
            print(f"    [PE] attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    return None


def extract_vivid_instruction(pe_prompt: dict) -> str:
    """
    从 PE 返回的 dict 中提取 vivid_instruction 文本。
    路径：pe_prompt["mm_caption"] → JSON → r2v_caption_zh[0]["vivid_instruction"]
    """
    try:
        mm = json.loads(pe_prompt.get("mm_caption", "{}"))
        return mm.get("r2v_caption_zh", [{}])[0].get("vivid_instruction", "")
    except Exception:
        return ""


# ═══════════════════════════ CSV HELPERS ═════════════════════════════════════

def load_csv_index() -> dict[tuple, dict]:
    index = {}
    if not CSV_PATH.exists():
        return index
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_label    = row.get("vid_label") or row.get("\ufeffvid_label", "")
            prompt_level = row.get("prompt_level", "")
            if not vid_label:
                print(f"  [WARN] Skipping CSV row with missing vid_label, keys={list(row.keys())[:5]}")
                continue
            key = (vid_label, prompt_level)
            index[key] = row
    return index


def save_csv_index(index: dict) -> None:
    rows = sorted(index.values(), key=lambda r: (r["idx"], r["prompt_level"]))
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:  # BOM → Excel 中文不乱码
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def record_result(index: dict, vid_label: str, video_id: str,
                  idx: int, score: int, level: int, duration: float,
                  prompt_words: int, original_prompt: str,
                  pe_prompt_json: str, vivid_instruction: str,
                  vivid_instruction_chars: int) -> None:
    key = (vid_label, str(level))
    index[key] = {
        "vid_label":               vid_label,
        "video_id":                video_id,
        "idx":                     str(idx),
        "score":                   str(score),
        "prompt_level":            str(level),
        "duration":                str(duration),
        "prompt_words":            str(prompt_words),
        "original_prompt":         original_prompt,
        "pe_prompt_json":          pe_prompt_json,
        "vivid_instruction":       vivid_instruction,
        "vivid_instruction_chars": str(vivid_instruction_chars),
        "pe_at":                   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_csv_index(index)
    print(f"  CSV updated: {CSV_PATH.name}")


# ═══════════════════════════ MAIN ════════════════════════════════════════════

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_index   = load_csv_index()
    all_records = load_scored_records()

    print(f"Loaded {len(csv_index)} existing CSV entries from {CSV_PATH}")
    print(f"Found {len(all_records)} records with score {sorted(TARGET_SCORES)}")
    print(f"Prompt levels: {PROMPT_LEVELS}\n")

    for rec in all_records:
        video_id  = rec.get("video_id", "unknown")
        rec_idx   = rec.get("_idx", 0)
        rec_score = rec.get("_score", 0)

        p0_path, p1_path, p15_path = find_label_files(video_id, rec_idx, rec_score)
        if not p0_path:
            print(f"[SKIP] No label files for {video_id}")
            continue

        actual_stem = p0_path.stem
        try:
            rec_idx   = int(actual_stem.split('_')[1])
            rec_score = int(actual_stem.split('_')[2].replace('score', ''))
        except (IndexError, ValueError):
            pass

        vid_label = f"id_{rec_idx:04d}_score{rec_score}_{video_id}"

        # 找本地 first-frame（PE API 需要上传文件，不能用 TOS URL）
        first_frame_path = find_first_frame(video_id, rec_idx, rec_score)
        if first_frame_path is None:
            print(f"[SKIP] No first frame for {vid_label}")
            continue

        with p0_path.open(encoding='utf-8')  as fp: phase0   = json.load(fp)
        with p1_path.open(encoding='utf-8')  as fp: phase1   = json.load(fp)
        with p15_path.open(encoding='utf-8') as fp: phase1_5 = json.load(fp)

        for lv in PROMPT_LEVELS:
            csv_key = (vid_label, str(lv))

            logger_name = f"{vid_label}_lv{lv}"
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                fh = logging.FileHandler(LOG_DIR / f"{logger_name}.log", encoding='utf-8')
                fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
                logger.addHandler(fh)

            print(f"\n{'='*70}")
            print(f"Processing {vid_label}  score={rec_score}  level={lv}")
            logger.info(f"=== Start {vid_label} score={rec_score} level={lv} ===")

            if csv_key in csv_index:
                print(f"  [SKIP] already in CSV")
                continue

            # ── 构建 prompt ───────────────────────────────────────────────────
            prompt, duration = build_whole_video_prompt(phase0, phase1, phase1_5, lv)
            prompt_words = len(prompt.split())
            logger.info(f"duration={duration}s  prompt_words={prompt_words}")
            print(f"  Duration: {duration}s  |  Prompt words: {prompt_words}")
            print(f"  First frame: {first_frame_path.name}")

            # ── 调用 PE API ───────────────────────────────────────────────────
            pe_prompt = call_pe(prompt, duration, first_frame_path, logger)
            if pe_prompt is None:
                print(f"  [FAIL] PE API returned None, skipping")
                logger.error("PE API returned None")
                continue

            pe_prompt_json          = json.dumps(pe_prompt, ensure_ascii=False)
            vivid_instruction       = extract_vivid_instruction(pe_prompt)
            vivid_instruction_chars = len(vivid_instruction)

            print(f"  PE OK: vivid_instruction_chars={vivid_instruction_chars}")
            logger.info(f"pe_prompt_json: {pe_prompt_json}")
            logger.info(
                f"vivid_instruction_chars={vivid_instruction_chars}  "
                f"vivid_instruction(preview): {vivid_instruction[:100]}"
            )

            # ── 写入 CSV ──────────────────────────────────────────────────────
            record_result(
                csv_index, vid_label, video_id,
                rec_idx, rec_score, lv, duration,
                prompt_words, prompt, pe_prompt_json,
                vivid_instruction, vivid_instruction_chars,
            )

    print(f"\n{'='*70}")
    print(f"All done. CSV: {CSV_PATH}")