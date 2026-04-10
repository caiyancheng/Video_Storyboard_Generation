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

STORYBOARD_PHASE0_PATH = "Sample_Prompts/sample_storyboard_phase0_v15_result.json"
STORYBOARD_PHASE1_PATH = "Sample_Prompts/sample_storyboard_phase1_v15_result.json"
STORYBOARD_PHASE1_5_PATH = "Sample_Prompts/sample_storyboard_phase1_5_v15_result.json"
OUTPUT_DIR = "Sample_results"
LOG_DIR = "Sample_results/logs"


def parse_time_range(time_range):
    """Parse 'MM:SS.mmm-MM:SS.mmm' and return (start_sec, end_sec) as floats."""
    start_str, end_str = time_range.split('-')

    def to_seconds(ts):
        mm, ss = ts.split(':')
        return int(mm) * 60 + float(ss)

    return to_seconds(start_str), to_seconds(end_str)


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
        # Use dense_caption from phase1 as the main description
        phase1_shot = phase1_shot_map.get(shot_id, {})
        dense_caption = phase1_shot.get('dense_caption', shot['caption'])

        # Add beat_description from phase1_5 as scene context
        phase1_5_info = phase1_5_shot_map.get(shot_id, {})
        beat_description = ''
        scene_id = ''
        if phase1_5_info:
            scene_id = phase1_5_info.get('scene_id', '')
            role = phase1_5_info.get('role_in_scene', {})
            beat_description = role.get('beat_description', '')

        prompt_parts = [dense_caption]
        if beat_description:
            prompt_parts.append(f"Scene context: {beat_description}")
        prompt = '\n'.join(prompt_parts)

        logger.info(f"scene_id={scene_id}")
        logger.info(f"prompt (first 300 chars): {prompt[:300]}")
        print(f"  Scene: {scene_id}")
        print(f"  Prompt preview: {prompt[:100]}...")

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
                "aspect_ratio": "16:9",
                "resolution": "576p",
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