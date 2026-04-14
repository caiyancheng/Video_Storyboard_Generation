"""
save_prompts_local.py  (本地运行)

读取评分 4/5 分的视频记录，为每条记录生成 Level 1-4 的完整 prompt，
保存到本地文件夹。不涉及任何 Cairo / Seedance 调用。

输出结构：
  PROMPT_OUT_ROOT/
    level1/  {vid_label}.txt
    level2/  {vid_label}.txt
    level3/  {vid_label}.txt
    level4/  {vid_label}.txt
"""

import json
from pathlib import Path

from tqdm import tqdm

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

SCORED_FILE = Path(
    "/Users/bytedance/Datasets/"
    "tt_template_hq_publish_data_1400k_USAU"
    ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl"
)

LABEL_ROOT   = Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                    "/shu_inverse_label")
PHASE0_DIR   = LABEL_ROOT / "phase0_chinese_labels"
PHASE1_DIR   = LABEL_ROOT / "phase1_chinese_labels"
PHASE1_5_DIR = LABEL_ROOT / "phase1_5_chinese_labels"

PROMPT_OUT_ROOT = (Path("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
                        "/shu_inverse_label/generated_videos/prompts"))

PROMPT_LEVELS  = [1, 2, 3, 4]
TARGET_SCORES  = {4, 5}
SKIP_EXISTING  = True   # 已存在的 txt 跳过


# ═══════════════════════ PROMPT BUILDING (与生成脚本保持一致) ═════════════════

def parse_time_range(time_range: str) -> tuple[float, float]:
    start_str, end_str = time_range.split('-')

    def to_sec(ts):
        mm, ss = ts.split(':')
        return int(mm) * 60 + float(ss)

    return to_sec(start_str), to_sec(end_str)


def build_whole_video_prompt(phase0: dict, phase1: dict, phase1_5: dict,
                             level: int) -> tuple[str, float]:
    """
    Build a single whole-video generation prompt.
    Returns (prompt_text, duration_seconds).
    与 test_storyboard_seedance_2_0_it2v_filter_15s_data.py 保持完全一致。
    """
    # ── Registries ───────────────────────────────────────────────────────────
    audio_map    = {a['id']: a for a in phase0.get('audio_registry',    [])}
    subject_map  = {s['id']: s for s in phase0.get('subject_registry',  [])}
    prop_map     = {p['id']: p for p in phase0.get('prop_registry',     [])}
    location_map = {l['id']: l for l in phase0.get('location_registry', [])}
    shots_p0     = phase0.get('shot_registry', [])
    scene        = (phase0.get('scene_registry') or [{}])[0]

    phase1_shot_map  = {s['id']: s for s in phase1.get('shot_registry', [])}
    phase15_shot_map = {s['shot_id']: s for s in phase1_5.get('shot_in_scene_registry', [])}

    # ── Total duration from last shot's end time ──────────────────────────────
    last_shot = shots_p0[-1] if shots_p0 else {}
    _, end_sec = parse_time_range(last_shot.get('time_range', '00:00.000-00:00.000'))
    duration = round(max(4.0, min(15.0, end_sec)), 3)

    parts = []

    # ── 1. Scene overview (all levels) ───────────────────────────────────────
    scene_caption = scene.get('caption', '')
    parts.append(
        f"[Whole-video generation | {duration}s | Level {level}]\n"
        f"Scene overview: {scene_caption}"
    )

    # ── 2. Characters (all levels) ───────────────────────────────────────────
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

    # ── 3. Locations (all levels) ────────────────────────────────────────────
    if location_map:
        lines = ["Locations:"]
        for lid, loc in location_map.items():
            line = f"  {lid} ({loc.get('name', '')})"
            if level >= 3:
                line += f": {loc.get('visual_features', '')}"
            lines.append(line)
        parts.append('\n'.join(lines))

    # ── 4. Audio (level ≥ 2) ─────────────────────────────────────────────────
    if level >= 2 and audio_map:
        lines = ["Audio:"]
        for aid, a in audio_map.items():
            line = f"  {a.get('name', '')} [{a.get('type', '')}]"
            if a.get('style'):
                line += f" — {a['style']}"
            lines.append(line)
        parts.append('\n'.join(lines))

    # ── 5. Shot flow ─────────────────────────────────────────────────────────
    if level >= 2:
        lines = ["Shot flow (chronological):"]
        for shot in shots_p0:
            sid = shot['id']
            start, end = parse_time_range(shot['time_range'])
            dur_s = round(end - start, 2)

            p1s  = phase1_shot_map.get(sid, {})
            cam  = p1s.get('visual', {}).get('camera', {})
            p15s = phase15_shot_map.get(sid, {})

            line = f"  {shot['time_range']} ({dur_s}s): {shot.get('caption', '')}"

            if level >= 3:
                mood = cam.get('mood_atmosphere', '')
                beat = (p15s.get('role_in_scene') or {}).get('beat_description', '')
                extras = []
                if mood:
                    extras.append(f"mood={mood}")
                if beat:
                    extras.append(f"beat={beat}")
                if extras:
                    line += f"\n    [{'; '.join(extras)}]"

            if level >= 4:
                dense = p1s.get('dense_caption', '')
                if dense:
                    line += f"\n    Dense: {dense}"
                cont   = p15s.get('continuity_logic', {})
                to_next = cont.get('to_next_shot', {}).get('relation', '')
                if to_next:
                    line += f"\n    → next: {to_next}"
                interactions = p1s.get('visual', {}).get('interaction_tracking', [])
                for it in interactions:
                    line += f"\n    interact: {it.get('interaction', '')}"

            lines.append(line)
        parts.append('\n'.join(lines))

    # ── 6. Emotional arc (level ≥ 3) ─────────────────────────────────────────
    if level >= 3:
        arc_parts = []
        for shot in shots_p0:
            sid  = shot['id']
            p15s = phase15_shot_map.get(sid, {})
            contrib = p15s.get('scene_contribution', {})
            ep = contrib.get('emotion_pacing', '')
            if ep:
                arc_parts.append(f"{shot['time_range']}: {ep}")
        if arc_parts:
            parts.append("Emotional arc:\n" + '\n'.join(f"  {a}" for a in arc_parts))

    # ── 7. Props summary (level ≥ 3) ─────────────────────────────────────────
    if level >= 3 and prop_map:
        lines = ["Key props:"]
        for pid, p in prop_map.items():
            lines.append(f"  {pid} ({p.get('name', '')}): {p.get('visual_features', '')}")
        parts.append('\n'.join(lines))

    # ── 8. First-frame reference (all levels) ────────────────────────────────
    parts.append(
        "Reference image: The provided image is the FIRST FRAME of the entire video. "
        "Use it as a precise visual anchor for scene composition, character appearance, "
        "lighting, and setting throughout the whole video."
    )

    return '\n\n'.join(parts), duration


# ═══════════════════════ FILE LOOKUP ═════════════════════════════════════════

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
    """返回 (p0_path, p1_path, p15_path) 或 (None, None, None)。"""
    stem = f"id_{idx:04d}_score{score}_{video_id}"
    p0   = PHASE0_DIR   / f"{stem}_phase0.json"
    p1   = PHASE1_DIR   / f"{stem}_phase1.json"
    p15  = PHASE1_5_DIR / f"{stem}_phase1_5.json"
    if p0.exists() and p1.exists() and p15.exists():
        return p0, p1, p15
    # Fallback: 只按 video_id glob
    m0  = list(PHASE0_DIR.glob(f"*{video_id}_phase0.json"))
    m1  = list(PHASE1_DIR.glob(f"*{video_id}_phase1.json"))
    m15 = list(PHASE1_5_DIR.glob(f"*{video_id}_phase1_5.json"))
    if m0 and m1 and m15:
        return m0[0], m1[0], m15[0]
    return None, None, None


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def main():
    # 创建输出子文件夹
    level_dirs = {}
    for lv in PROMPT_LEVELS:
        d = PROMPT_OUT_ROOT / f"level{lv}"
        d.mkdir(parents=True, exist_ok=True)
        level_dirs[lv] = d

    records = load_scored_records()
    print(f"找到 {len(records)} 条评分 {sorted(TARGET_SCORES)} 分记录\n")

    ok = skip = fail = 0

    for rec in tqdm(records, desc="生成 prompt"):
        video_id  = rec.get("video_id", "unknown")
        rec_idx   = rec.get("_idx",   0)
        rec_score = rec.get("_score", 0)

        p0_path, p1_path, p15_path = find_label_files(video_id, rec_idx, rec_score)
        if not p0_path:
            tqdm.write(f"[SKIP] 无标注文件: {video_id}")
            fail += 1
            continue

        # 从实际文件名解析 idx / score
        actual_stem = p0_path.stem   # e.g. id_0003_score5_xxx_phase0
        try:
            parts_stem  = actual_stem.split('_')
            rec_idx     = int(parts_stem[1])
            rec_score   = int(parts_stem[2].replace('score', ''))
        except (IndexError, ValueError):
            pass

        vid_label = f"id_{rec_idx:04d}_score{rec_score}_{video_id}"

        # 加载标注 JSON（只读一次）
        try:
            with p0_path.open(encoding='utf-8')  as fp: phase0   = json.load(fp)
            with p1_path.open(encoding='utf-8')  as fp: phase1   = json.load(fp)
            with p15_path.open(encoding='utf-8') as fp: phase1_5 = json.load(fp)
        except Exception as e:
            tqdm.write(f"[FAIL] JSON 读取失败 {vid_label}: {e}")
            fail += 1
            continue

        all_skipped = True
        for lv in PROMPT_LEVELS:
            out_path = level_dirs[lv] / f"{vid_label}.txt"

            if SKIP_EXISTING and out_path.exists():
                skip += 1
                continue

            try:
                prompt, duration = build_whole_video_prompt(phase0, phase1, phase1_5, lv)
                out_path.write_text(prompt, encoding='utf-8')
                all_skipped = False
                ok += 1
            except Exception as e:
                tqdm.write(f"[FAIL] {vid_label} level={lv}: {e}")
                fail += 1

        if not all_skipped:
            tqdm.write(f"  ✓ {vid_label}")

    print(f"\n完成：写入 {ok} 个文件，跳过(已存在) {skip} 个，失败 {fail} 个")
    print(f"输出目录：{PROMPT_OUT_ROOT}")


if __name__ == "__main__":
    main()
