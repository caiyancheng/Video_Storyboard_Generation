import time
import json
import os
import logging
import euler
from euler import base_compat_middleware

# bytedeuler==2.4.3
# byted_cairo==2.6.6
from cairo_v2.idls.thrift import GetTaskReportRequestThrift
from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task

# ── Prompt verbosity level ────────────────────────────────────────────────────
# 1  minimal  (~80  words): caption + character names/actions + location name
# 2  standard (~200 words): + dense_caption + camera mood + appearances + props + audio transcript
# 3  detailed (~400 words): + full camera + rationales + location env + audio content + scene beat
# 4  full     (no limit)  : + interaction tracking + continuity transitions + information_gain
PROMPT_LEVEL = 3

STORYBOARD_PHASE0_PATH = "Video_Generation/Sample_Prompts/sample_storyboard_phase0_v15_result.json"
STORYBOARD_PHASE1_PATH = "Video_Generation/Sample_Prompts/sample_storyboard_phase1_v15_result.json"
STORYBOARD_PHASE1_5_PATH = "Video_Generation/Sample_Prompts/sample_storyboard_phase1_5_v15_result.json"
OUTPUT_DIR = "Video_Generation/Sample_results"
LOG_DIR = "Video_Generation/Sample_results/logs"


def parse_time_range(time_range):
    """Parse 'MM:SS.mmm-MM:SS.mmm' and return (start_sec, end_sec) as floats."""
    start_str, end_str = time_range.split('-')

    def to_seconds(ts):
        mm, ss = ts.split(':')
        return int(mm) * 60 + float(ss)

    return to_seconds(start_str), to_seconds(end_str)


def build_shot_prompt(shot, phase1_shot, phase1_5_info,
                      audio_map, subject_map, prop_map, location_map,
                      level=PROMPT_LEVEL):
    """
    Build a generation prompt for a single shot at the requested verbosity level.

    Level 1  minimal  (~80  words): caption + character names/actions + location name
    Level 2  standard (~200 words): + dense_caption + camera mood + appearances
                                      + prop descriptions + audio transcript
    Level 3  detailed (~400 words): + full camera + rationales + location env
                                      + audio content/rationale + scene beat + emotion
    Level 4  full     (no limit)  : + interaction_tracking + continuity transitions
                                      + information_gain
    """
    lv = level
    parts = []

    # ── Pre-compute shared data ───────────────────────────────────────────────
    visual = phase1_shot.get('visual', {})
    p1_subject_map  = {t['id']: t for t in visual.get('subject_tracking',  [])}
    p1_prop_list    = visual.get('prop_tracking', [])
    p1_interact     = visual.get('interaction_tracking', [])
    p1_loc_map      = {t['id']: t for t in visual.get('location_tracking', [])}
    camera          = visual.get('camera', {})

    subject_ids  = shot.get('subject_id',  [])
    location_ids = shot.get('location_id', [])
    audio_ids    = shot.get('audio_id',    [])

    # Speech transcripts from phase1
    all_transcripts = [
        f'"{ev.get("transcript", "").strip()}"'
        for sp in phase1_shot.get('audio', {}).get('speech', [])
        for ev in sp.get('speech_events', [])
        if ev.get('transcript', '').strip()
    ]

    # ── 1. Shot overview (all levels) ────────────────────────────────────────
    parts.append(
        f"Shot: {shot['id']}  |  Time: {shot['time_range']}\n"
        f"Summary: {shot.get('caption', '')}"
    )

    # ── 2. Dense caption (level ≥ 2) ─────────────────────────────────────────
    if lv >= 2:
        dense = phase1_shot.get('dense_caption', '')
        if dense:
            parts.append(f"Detailed visual description:\n{dense}")

    # ── 3. Camera & atmosphere ────────────────────────────────────────────────
    if lv >= 2 and camera:
        cam_lines = ["Camera & atmosphere:"]
        # level 2: mood + scale only; level 3+: all fields
        if camera.get('shot_scale'):
            cam_lines.append(f"  Shot scale: {camera['shot_scale']}")
        if camera.get('mood_atmosphere'):
            cam_lines.append(f"  Mood/atmosphere: {camera['mood_atmosphere']}")
        if lv >= 3:
            if camera.get('camera_pose'):
                cam_lines.append(f"  Camera pose: {camera['camera_pose']}")
            if camera.get('lighting_style'):
                cam_lines.append(f"  Lighting: {camera['lighting_style']}")
            if camera.get('color_tone'):
                cam_lines.append(f"  Color tone: {camera['color_tone']}")
        parts.append('\n'.join(cam_lines))

    # ── 4. Characters ─────────────────────────────────────────────────────────
    if subject_ids:
        lines = ["Characters:"]
        for sid in subject_ids:
            s0 = subject_map.get(sid, {})
            s1 = p1_subject_map.get(sid, {})
            lines.append(f"  {sid} — {s0.get('name', 'Unknown')}")
            if lv >= 2 and s1.get('appearance_description'):
                lines.append(f"    Appearance: {s1['appearance_description']}")
            if s1.get('action'):                                  # all levels
                lines.append(f"    Action: {s1['action']}")
            if lv >= 3 and s0.get('visual_features'):
                lines.append(f"    Overall appearance: {s0['visual_features']}")
            if lv >= 3 and s0.get('rationale'):
                lines.append(f"    Role: {s0['rationale']}")
        parts.append('\n'.join(lines))

    # ── 5. Props ──────────────────────────────────────────────────────────────
    if lv >= 2 and p1_prop_list:
        lines = ["Props:"]
        for pt in p1_prop_list:
            pid = pt['id']
            p0  = prop_map.get(pid, {})
            label = p0.get('name') or pt.get('prop_description', '')[:50]
            lines.append(f"  {pid} — {label}")
            if pt.get('prop_description'):
                lines.append(f"    Description: {pt['prop_description']}")
            if pt.get('interaction'):
                lines.append(f"    Interaction: {pt['interaction']}")
            if lv >= 3 and p0.get('rationale'):
                lines.append(f"    Role: {p0['rationale']}")
        parts.append('\n'.join(lines))

    # ── 6. Subject–prop interactions (level 4 only) ───────────────────────────
    if lv >= 4 and p1_interact:
        lines = ["Subject–prop interactions:"]
        for it in p1_interact:
            lines.append(f"  {it.get('interaction', '')}")
        parts.append('\n'.join(lines))

    # ── 7. Location ───────────────────────────────────────────────────────────
    if location_ids:
        lines = ["Location:"]
        for lid in location_ids:
            loc = location_map.get(lid, {})
            l1  = p1_loc_map.get(lid, {})
            loc_type = ' / '.join(filter(None, [loc.get('type', ''), loc.get('secondary_type', '')]))
            lines.append(f"  {lid} — {loc.get('name', 'Unknown')}" +
                         (f" ({loc_type})" if lv >= 3 and loc_type else ""))
            if lv >= 2 and loc.get('visual_features'):
                lines.append(f"    Visual features: {loc['visual_features']}")
            if lv >= 3 and l1.get('visual_environment'):
                lines.append(f"    Environment (this shot): {l1['visual_environment']}")
            if lv >= 3:
                static = loc.get('static_elements', [])
                if static:
                    lines.append(f"    Key elements: {', '.join(static)}")
        parts.append('\n'.join(lines))

    # ── 8. Audio ──────────────────────────────────────────────────────────────
    if audio_ids:
        lines = ["Audio:"]
        for aid in audio_ids:
            a = audio_map.get(aid, {})
            atype = ' / '.join(filter(None, [a.get('type', ''), a.get('secondary_type', '')]))
            lines.append(f"  {aid} — {a.get('name', 'Unknown')}" +
                         (f" [{atype}]" if lv >= 3 and atype else ""))
            if lv >= 2 and a.get('style'):
                lines.append(f"    Style: {a['style']}")
            if lv >= 3 and a.get('content') and a['content'] != 'na':
                lines.append(f"    Content: {a['content']}")
            if lv >= 3 and a.get('rationale'):
                lines.append(f"    Rationale: {a['rationale']}")
        if lv >= 2 and all_transcripts:
            lines.append(f"  Spoken VO/dialogue: {' / '.join(all_transcripts)}")
        parts.append('\n'.join(lines))

    # ── 9. Scene narrative (phase1_5) ─────────────────────────────────────────
    if lv >= 3 and phase1_5_info:
        role       = phase1_5_info.get('role_in_scene', {})
        contrib    = phase1_5_info.get('scene_contribution', {})
        continuity = phase1_5_info.get('continuity_logic', {})
        scene_lines = [f"Scene narrative ({phase1_5_info.get('scene_id', '')}):"]
        if role.get('beat_description'):
            scene_lines.append(f"  Beat: {role['beat_description']}")
        if contrib.get('emotion_pacing'):
            scene_lines.append(f"  Emotional pacing: {contrib['emotion_pacing']}")
        if lv >= 4 and contrib.get('information_gain'):
            scene_lines.append(f"  Information conveyed: {contrib['information_gain']}")
        if lv >= 4:
            from_rel = continuity.get('from_previous_shot', {}).get('relation', '')
            to_rel   = continuity.get('to_next_shot',       {}).get('relation', '')
            if from_rel:
                scene_lines.append(f"  Transition in:  {from_rel}")
            if to_rel:
                scene_lines.append(f"  Transition out: {to_rel}")
        parts.append('\n'.join(scene_lines))

    # ── 10. First-frame reference (all levels) ────────────────────────────────
    parts.append(
        "Reference image: The provided image is the first frame of this shot. "
        "Use it as a precise visual reference for scene composition, "
        "character appearance, props, and setting."
    )

    return '\n\n'.join(parts)


def setup_shot_logger(shot_id_safe, log_dir):
    logger = logging.getLogger(shot_id_safe)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{shot_id_safe}.log"), encoding='utf-8'
        )
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    return logger


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load all three storyboard JSON files
    with open(STORYBOARD_PHASE0_PATH, 'r', encoding='utf-8') as f:
        phase0 = json.load(f)
    with open(STORYBOARD_PHASE1_PATH, 'r', encoding='utf-8') as f:
        phase1 = json.load(f)
    with open(STORYBOARD_PHASE1_5_PATH, 'r', encoding='utf-8') as f:
        phase1_5 = json.load(f)

    # Build lookups
    # phase0: entity registries (audio, subjects, props, locations)
    audio_map    = {a['id']: a for a in phase0.get('audio_registry',    [])}
    subject_map  = {s['id']: s for s in phase0.get('subject_registry',  [])}
    prop_map     = {p['id']: p for p in phase0.get('prop_registry',     [])}
    location_map = {l['id']: l for l in phase0.get('location_registry', [])}
    # phase1: shot_id -> shot dict (for dense_caption)
    phase1_shot_map = {s['id']: s for s in phase1['shot_registry']}
    # phase1_5: shot_id -> entry (for scene_id and role_in_scene)
    phase1_5_shot_map = {s['shot_id']: s for s in phase1_5['shot_in_scene_registry']}

    # Cairo client setup
    cairo_client = euler.Client(
        CairoService,
        target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva",
        transport="ttheader"
    )
    cairo_client.use(base_compat_middleware.client_middleware)
    workflow_id = "seedance_2_0_ti2v_e2e_with_pe_test_inference_only_v2"

    def get_task_report(task_id):
        get_req = GetTaskReportRequestThrift(task_id=task_id)
        get_resp = cairo_client.GetTaskReport(get_req)
        task = json.loads(get_resp.task)
        report = json.loads(get_resp.report)
        return task, report

    shots = phase0['shot_registry']
    print(f"Total shots to generate: {len(shots)}")

    for shot in shots:
        shot_id = shot['id']                          # e.g. "<Shot_1>"
        shot_id_safe = shot_id.strip('<>')            # e.g. "Shot_1"
        logger = setup_shot_logger(shot_id_safe, LOG_DIR)

        print(f"\n{'='*60}")
        print(f"Processing {shot_id}  time_range={shot['time_range']}")
        logger.info(f"=== Start processing {shot_id} time_range={shot['time_range']} ===")

        # --- Duration ---
        start_sec, end_sec = parse_time_range(shot['time_range'])
        duration_sec = end_sec - start_sec
        duration_int = max(1, round(duration_sec))
        logger.info(f"duration: {duration_sec:.3f}s -> API duration={duration_int}")
        print(f"  Duration: {duration_sec:.3f}s -> API duration={duration_int}s")

        # --- Build prompt ---
        phase1_shot   = phase1_shot_map.get(shot_id, {})
        phase1_5_info = phase1_5_shot_map.get(shot_id, {})
        scene_id = phase1_5_info.get('scene_id', '') if phase1_5_info else ''

        prompt = build_shot_prompt(
            shot, phase1_shot, phase1_5_info,
            audio_map, subject_map, prop_map, location_map
        )

        logger.info(f"scene_id={scene_id}")
        logger.info(f"full prompt:\n{prompt}")
        print(f"  Scene: {scene_id}")
        print(f"  Prompt preview: {prompt[:120]}...")

        # --- Submit task ---
        first_frame_url = (
            f"https://tosv.byted.org/obj/dm-stickers-rec-sg/dm-stickers-rec-sg/yancheng"
            f"/{shot_id_safe}_first_frame.jpg"
        )
        logger.info(f"first_frame_url={first_frame_url}")
        print(f"  First frame: {first_frame_url}")

        task_input = json.dumps({
            "binary_data": [
                {
                    "data": first_frame_url,
                    "type": "image"
                }
            ],
            "req_json": {
                "prompt": prompt,
                "language": "en",
                "duration": duration_int,
                "seed": 42,
                "aspect_ratio": "9:16",
                "resolution": "576x1024",
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
        submit_req.workflow_id = workflow_id
        print(f"  WorkflowID: {workflow_id}")

        try:
            submit_resp = cairo_client.SubmitAsyncTask(submit_req)
            task_id = submit_resp.task_id
            logger.info(f"Submitted task_id={task_id}")
            print(f"  Submitted task_id: {task_id}")
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            print(f"  ERROR: Failed to submit task: {e}")
            continue

        # --- Poll until done ---
        gen_start = time.time()
        video_url = None
        while True:
            try:
                task_report, report = get_task_report(task_id)
                status = task_report["status"]
                logger.info(f"Poll {task_id} -> {status}")
                print(f"  Polling {task_id} -> {status}")
                if status == "succeeded":
                    results = json.loads(task_report["output"])["results"]
                    key = list(results.keys())[0]
                    storage = results[key]['Extra']["storage"]
                    video_url = f"https://tosv-sg.tiktok-row.org/obj/iccv-vpipe-sg/{key}"
                    logger.info(f"Succeeded! video_key={key} storage={storage}")
                    logger.info(f"Video URL: {video_url}")
                    print(f"  Succeeded! Video URL: {video_url}")
                    break
                elif status in ("failed", "cancelled"):
                    logger.error(f"Task {task_id} ended with status={status}, report={report}")
                    print(f"  Task {task_id} ended: {status}")
                    break
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                print(f"  Poll error: {e}")
            time.sleep(5)

        elapsed = time.time() - gen_start
        logger.info(f"Generation time: {elapsed:.1f}s")
        print(f"  Generation time: {elapsed:.1f}s")

        # --- Download video (errors here won't stop the loop) ---
        if video_url:
            try:
                import urllib.request
                output_filename = f"Sample_results/{shot_id_safe}.mp4"
                output_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), output_filename
                )
                print(f"  Downloading video -> {output_path}")
                logger.info(f"Downloading from {video_url}")
                urllib.request.urlretrieve(video_url, output_path)
                logger.info(f"Saved: {output_path}")
                print(f"  Saved: {output_path}")
            except Exception as download_err:
                logger.error(f"Download failed: {download_err}")
                print(f"  Download failed: {download_err}")

    print(f"\n{'='*60}")
    print("All shots processed.")