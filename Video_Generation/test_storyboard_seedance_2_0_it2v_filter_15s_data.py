"""
test_storyboard_seedance_2_0_it2v_filter_15s_data.py

Whole-video Seedance generation from filtered 15s TikTok template clips.

Pipeline:
  1. Read scored records (score 4/5) from scored JSONL
  2. Match phase0 / phase1 / phase1_5 label files per video
  3. Build a single whole-video text prompt at a chosen verbosity level (1-4)
  4. Upload first-frame PNG to TOS (bucket: dm-stickers-rec-sg)
  5. Submit to Seedance it2v API via Cairo
  6. Poll → get video_url → download .mp4
  7. Save under shu_inverse_label/generated_videos/level_{N}/

Prompt verbosity levels (whole-video, not per-shot):
  1  minimal  (~100 w): scene caption + characters + locations + duration
  2  standard (~250 w): + ordered shot captions + audio style + mood
  3  detailed (~500 w): + appearances/actions + location detail + emotional arc
  4  full     (no lim): + per-shot dense captions + transitions + interactions
"""

import csv
import json
import logging
import os
import time
from pathlib import Path

import euler
from euler import base_compat_middleware
from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task
from cairo_v2.idls.thrift import GetTaskReportRequestThrift

# ═══════════════════════════ CONFIG ══════════════════════════════════════════

PROMPT_LEVELS = [1, 2, 3, 4]   # list of levels to run; e.g. [1, 2] or [4]

SCORED_FILE = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_hq_publish_data_1400k_USAU"
                   ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl")

LABEL_ROOT  = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
                   "/shu_inverse_label")
PHASE0_DIR  = LABEL_ROOT / "phase0_chinese_labels"
PHASE1_DIR  = LABEL_ROOT / "phase1_chinese_labels"
PHASE1_5_DIR = LABEL_ROOT / "phase1_5_chinese_labels"
FIRST_FRAME_DIR = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
                       "/first_frame")

OUT_ROOT    = LABEL_ROOT / "generated_videos_2"
LOG_DIR     = OUT_ROOT / "logs"
CSV_PATH    = OUT_ROOT / "generation_results.csv"

CSV_FIELDS  = ["vid_label", "video_id", "idx", "score", "prompt_level",
               "duration", "prompt_words", "video_url", "generated_at"]

TARGET_SCORES = {4, 5}

# First frames already uploaded to TOS; URL pattern:
# https://tosv.byted.org/obj/dm-stickers-rec-sg/tt_template_1400k_15s_video_sample/first_frame/{filename}
FIRST_FRAME_TOS_BASE = ("https://tosv.byted.org/obj/dm-stickers-rec-sg"
                        "/tt_template_1400k_15s_video_sample/first_frame")

WORKFLOW_ID  = "seedance_2_0_ti2v_e2e_with_pe_test_inference_only_v2"
ASPECT_RATIO = "9:16"
RESOLUTION   = "480p"
SEED         = 42


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
    """
    Returns (phase0_path, phase1_path, phase1_5_path) or (None, None, None).
    Filename pattern: id_{idx:04d}_score{score}_{video_id}_phase*.json
    """
    stem = f"id_{idx:04d}_score{score}_{video_id}"
    p0  = PHASE0_DIR   / f"{stem}_phase0.json"
    p1  = PHASE1_DIR   / f"{stem}_phase1.json"
    p15 = PHASE1_5_DIR / f"{stem}_phase1_5.json"
    if p0.exists() and p1.exists() and p15.exists():
        return p0, p1, p15
    # Fallback: glob by video_id only (idx/score may differ)
    matches0 = list(PHASE0_DIR.glob(f"*{video_id}_phase0.json"))
    matches1 = list(PHASE1_DIR.glob(f"*{video_id}_phase1.json"))
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
    """
    Build a single whole-video generation prompt.
    Returns (prompt_text, duration_seconds).

    Level 1  minimal  (~100 w): scene caption + characters + locations + duration
    Level 2  standard (~250 w): + ordered shot captions + audio style + mood
    Level 3  detailed (~500 w): + appearances/actions + location detail + emotional arc
    Level 4  full     (no lim): + per-shot dense captions + continuity + interactions
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
            _, end = parse_time_range(shot['time_range'])
            _, start = parse_time_range(shot['time_range'])
            dur_s = round(end - start, 2)

            p1s = phase1_shot_map.get(sid, {})
            cam = p1s.get('visual', {}).get('camera', {})
            p15s = phase15_shot_map.get(sid, {})

            # Level 2: caption only
            line = f"  {shot['time_range']} ({dur_s}s): {shot.get('caption', '')}"

            if level >= 3:
                # Add mood + beat
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
                # Add dense caption
                dense = p1s.get('dense_caption', '')
                if dense:
                    line += f"\n    Dense: {dense}"
                # Add continuity
                cont = p15s.get('continuity_logic', {})
                to_next = cont.get('to_next_shot', {}).get('relation', '')
                if to_next:
                    line += f"\n    → next: {to_next}"
                # Add interactions
                interactions = p1s.get('visual', {}).get('interaction_tracking', [])
                for it in interactions:
                    line += f"\n    interact: {it.get('interaction', '')}"

            lines.append(line)
        parts.append('\n'.join(lines))

    # ── 6. Emotional arc (level ≥ 3) ─────────────────────────────────────────
    if level >= 3:
        arc_parts = []
        for shot in shots_p0:
            sid = shot['id']
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


def get_first_frame_url(vid_label: str) -> str:
    """Construct TOS URL for the pre-uploaded first frame PNG."""
    return f"{FIRST_FRAME_TOS_BASE}/{vid_label}_first.png"


# ═══════════════════════════ CAIRO / API ════════════════════════════════════

def setup_cairo_client():
    client = euler.Client(
        CairoService,
        target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva",
        transport="ttheader"
    )
    client.use(base_compat_middleware.client_middleware)
    return client


def get_task_report(cairo_client, task_id: str):
    req = GetTaskReportRequestThrift(task_id=task_id)
    resp = cairo_client.GetTaskReport(req)
    return json.loads(resp.task), json.loads(resp.report)


def submit_and_poll(cairo_client, prompt: str, duration: float,
                    first_frame_url: str, logger: logging.Logger) -> str | None:
    """Submit to Seedance, poll until done, return video_url or None."""
    task_input = json.dumps({
        "binary_data": [
            {"data": first_frame_url, "type": "image"}
        ],
        "req_json": {
            "prompt": prompt,
            "language": "zh",
            "duration": duration,
            "seed": SEED,
            "aspect_ratio": ASPECT_RATIO,
            "resolution": RESOLUTION,
            "binary_var_name": ["image"],
            "workflow": "seedance_2_0_pe_integration.json"
        }
    })

    submit_req = SubmitAsyncTaskRequest()
    submit_req.task = Task(
        input=task_input,
        priority=7,
        tags={"second_biz_name": "test"}
    )
    submit_req.workflow_id = WORKFLOW_ID

    try:
        resp = cairo_client.SubmitAsyncTask(submit_req)
        task_id = resp.task_id
        logger.info(f"Submitted task_id={task_id}")
        print(f"  Submitted task_id: {task_id}")
    except Exception as e:
        logger.error(f"Submit failed: {e}")
        print(f"  ERROR submit: {e}")
        return None

    gen_start = time.time()
    while True:
        try:
            task_report, report = get_task_report(cairo_client, task_id)
            status = task_report["status"]
            logger.info(f"Poll {task_id} -> {status}")
            print(f"  Polling {task_id} -> {status}")
            if status == "succeeded":
                results = json.loads(task_report["output"])["results"]
                key = list(results.keys())[0]
                video_url = f"https://tosv-sg.tiktok-row.org/obj/iccv-vpipe-sg/{key}"
                logger.info(f"Succeeded! video_url={video_url}")
                print(f"  Succeeded! URL: {video_url}")
                break
            elif status in ("failed", "cancelled"):
                logger.error(f"Task ended: {status}  output={task_report.get('output')}")
                print(f"  Task ended: {status}")
                print(task_report.get("output"))
                return None
        except Exception as e:
            logger.warning(f"Poll error: {e}")
            print(f"  Poll error: {e}")
        time.sleep(5)

    elapsed = time.time() - gen_start
    logger.info(f"Generation time: {elapsed:.1f}s")
    print(f"  Generation time: {elapsed:.1f}s")
    return video_url


def load_csv_index() -> dict[str, dict]:
    """Load existing CSV into a dict keyed by (vid_label, prompt_level)."""
    index = {}
    if not CSV_PATH.exists():
        return index
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["vid_label"], row["prompt_level"])
            index[key] = row
    return index


def save_csv_index(index: dict[str, dict]) -> None:
    """Write all rows back to CSV (sorted for readability)."""
    rows = sorted(index.values(), key=lambda r: (r["idx"], r["prompt_level"]))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def record_result(index: dict[str, dict], vid_label: str, video_id: str,
                  idx: int, score: int, level: int, duration: float,
                  prompt_words: int, video_url: str) -> None:
    """Upsert one row into the in-memory index, then flush to disk."""
    key = (vid_label, str(level))
    index[key] = {
        "vid_label":    vid_label,
        "video_id":     video_id,
        "idx":          str(idx),
        "score":        str(score),
        "prompt_level": str(level),
        "duration":     str(duration),
        "prompt_words": str(prompt_words),
        "video_url":    video_url,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_csv_index(index)
    print(f"  CSV updated: {CSV_PATH.name}")


# ═══════════════════════════ MAIN ═══════════════════════════════════════════

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Load existing CSV index (for upsert logic) ───────────────────────────
    csv_index = load_csv_index()
    print(f"Loaded {len(csv_index)} existing CSV entries from {CSV_PATH}")

    # ── Load scored records ──────────────────────────────────────────────────
    all_records = load_scored_records()
    print(f"Found {len(all_records)} records with score {sorted(TARGET_SCORES)}")
    print(f"Prompt levels: {PROMPT_LEVELS}\n")

    # ── Cairo client ─────────────────────────────────────────────────────────
    cairo_client = setup_cairo_client()

    # ── Per-video loop ───────────────────────────────────────────────────────
    for rec in all_records:
        video_id = rec.get("video_id", "unknown")
        rec_idx  = rec.get("_idx", 0)
        rec_score = rec.get("_score", 0)

        p0_path, p1_path, p15_path = find_label_files(video_id, rec_idx, rec_score)
        if not p0_path:
            print(f"[SKIP] No label files for {video_id}")
            continue

        # Parse idx/score from actual filename
        actual_stem = p0_path.stem
        try:
            rec_idx   = int(actual_stem.split('_')[1])
            rec_score = int(actual_stem.split('_')[2].replace('score', ''))
        except (IndexError, ValueError):
            pass

        vid_label = f"id_{rec_idx:04d}_score{rec_score}_{video_id}"

        # ── Load label JSONs once per video ──────────────────────────────────
        with p0_path.open(encoding='utf-8')  as fp: phase0   = json.load(fp)
        with p1_path.open(encoding='utf-8')  as fp: phase1   = json.load(fp)
        with p15_path.open(encoding='utf-8') as fp: phase1_5 = json.load(fp)

        first_frame_url = get_first_frame_url(vid_label)

        # ── Inner loop: one submission per level ──────────────────────────────
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

            # Skip if already recorded in CSV
            if csv_key in csv_index:
                print(f"  [SKIP] already in CSV: {csv_index[csv_key]['video_url']}")
                continue

            # ── Build whole-video prompt ─────────────────────────────────────
            prompt, duration = build_whole_video_prompt(phase0, phase1, phase1_5, lv)
            prompt_words = len(prompt.split())
            logger.info(f"duration={duration}s  prompt_words={prompt_words}")
            logger.info(f"Full prompt:\n{prompt}")
            print(f"  Duration: {duration}s")
            print(f"  Prompt (~{prompt_words} words):\n    {prompt[:200].replace(chr(10), ' ')}...")
            print(f"  First frame URL: {first_frame_url}")

            # ── Submit & poll ────────────────────────────────────────────────
            video_url = submit_and_poll(cairo_client, prompt, duration,
                                        first_frame_url, logger)
            if not video_url:
                continue

            # ── Record to CSV (no local download) ───────────────────────────
            record_result(csv_index, vid_label, video_id,
                          rec_idx, rec_score, lv, duration,
                          prompt_words, video_url)
            logger.info(f"Recorded video_url={video_url}")

    print(f"\n{'='*70}")
    print(f"All done. CSV: {CSV_PATH}")
