"""
test_storyboard_seedance_2_0_it2v_filter_15s_data_compress.py

与原版的差异：
  1. 每次生成视频都重新创建 Cairo client，避免长连接复用问题
  2. Prompt 末尾追加版权安全声明，规避 music_illegal 风险
  3. 生成完后立即下载视频，同时保存原版 + 压缩版（目标 ≤4MB）
     - 原版  → level_{N}/
     - 压缩版 → level_{N}_compressed/

其余逻辑（数据加载、prompt 构建、Cairo 提交轮询）与原版保持一致。
"""

import csv
import json
import logging
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import euler
from euler import base_compat_middleware
from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task
from cairo_v2.idls.thrift import GetTaskReportRequestThrift

# ═══════════════════════════ CONFIG ══════════════════════════════════════════

PROMPT_LEVELS = [1, 2, 3, 4]   # list of levels to run; e.g. [1, 2] or [4]

SCORED_FILE = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_hq_publish_data_1400k_USAU"
                   ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl")

LABEL_ROOT   = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
                    "/shu_inverse_label")
PHASE0_DIR   = LABEL_ROOT / "phase0_chinese_labels"
PHASE1_DIR   = LABEL_ROOT / "phase1_chinese_labels"
PHASE1_5_DIR = LABEL_ROOT / "phase1_5_chinese_labels"
FIRST_FRAME_DIR = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
                       "/first_frame")

OUT_ROOT = LABEL_ROOT / "generated_videos_2"
LOG_DIR  = OUT_ROOT / "logs"
CSV_PATH = OUT_ROOT / "generation_results.csv"

CSV_FIELDS = [
    "vid_label", "video_id", "idx", "score", "prompt_level",
    "duration", "prompt_words", "video_url",
    "local_path", "compressed_path", "generated_at",
]

TARGET_SCORES = {4, 5}

FIRST_FRAME_TOS_BASE = ("https://tosv.byted.org/obj/dm-stickers-rec-sg"
                        "/tt_template_1400k_15s_video_sample/first_frame")

WORKFLOW_ID  = "seedance_2_0_ti2v_e2e_with_pe_test_inference_only_v2"
ASPECT_RATIO = "9:16"
RESOLUTION   = "480p"
SEED         = 42

# 压缩目标：15s 视频控制在 ≤4MB，对应总码率约 2000kbps
# 用 CRF+maxrate 双保险：CRF=28 控质量下限，maxrate 控体积上限
COMPRESS_CRF      = 28
COMPRESS_MAXRATE  = "2200k"   # 峰值码率
COMPRESS_BUFSIZE  = "4400k"   # 缓冲区 = 2× maxrate
COMPRESS_AUDIO    = "128k"

# 版权安全声明（追加到每条 prompt 末尾）
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
    """
    Build a single whole-video generation prompt.
    Returns (prompt_text, duration_seconds).
    COPYRIGHT_NOTICE is appended at the end of every prompt.
    """
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

    # 1. Scene overview
    scene_caption = scene.get('caption', '')
    parts.append(
        f"[Whole-video generation | {duration}s | Level {level}]\n"
        f"Scene overview: {scene_caption}"
    )

    # 2. Characters
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

    # 3. Locations
    if location_map:
        lines = ["Locations:"]
        for lid, loc in location_map.items():
            line = f"  {lid} ({loc.get('name', '')})"
            if level >= 3:
                line += f": {loc.get('visual_features', '')}"
            lines.append(line)
        parts.append('\n'.join(lines))

    # 4. Audio (level ≥ 2)
    if level >= 2 and audio_map:
        lines = ["Audio:"]
        for aid, a in audio_map.items():
            line = f"  {a.get('name', '')} [{a.get('type', '')}]"
            if a.get('style'):
                line += f" — {a['style']}"
            lines.append(line)
        parts.append('\n'.join(lines))

    # 5. Shot flow (level ≥ 2)
    if level >= 2:
        lines = ["Shot flow (chronological):"]
        for shot in shots_p0:
            sid = shot['id']
            _, end   = parse_time_range(shot['time_range'])
            _, start = parse_time_range(shot['time_range'])
            dur_s = round(end - start, 2)

            p1s  = phase1_shot_map.get(sid, {})
            cam  = p1s.get('visual', {}).get('camera', {})
            p15s = phase15_shot_map.get(sid, {})

            line = f"  {shot['time_range']} ({dur_s}s): {shot.get('caption', '')}"

            if level >= 3:
                mood = cam.get('mood_atmosphere', '')
                beat = (p15s.get('role_in_scene') or {}).get('beat_description', '')
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

    # 6. Emotional arc (level ≥ 3)
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

    # 7. Props (level ≥ 3)
    if level >= 3 and prop_map:
        lines = ["Key props:"]
        for pid, p in prop_map.items():
            lines.append(f"  {pid} ({p.get('name', '')}): {p.get('visual_features', '')}")
        parts.append('\n'.join(lines))

    # 8. First-frame reference
    parts.append(
        "Reference image: The provided image is the FIRST FRAME of the entire video. "
        "Use it as a precise visual anchor for scene composition, character appearance, "
        "lighting, and setting throughout the whole video."
    )

    prompt = '\n\n'.join(parts) + COPYRIGHT_NOTICE
    return prompt, duration


def get_first_frame_url(vid_label: str) -> str:
    return f"{FIRST_FRAME_TOS_BASE}/{vid_label}_first.png"


# ═══════════════════════════ CAIRO / API ═════════════════════════════════════

def make_cairo_client():
    """每次调用都新建一个 Cairo client，避免长连接复用导致的异常。"""
    client = euler.Client(
        CairoService,
        target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva",
        transport="ttheader"
    )
    client.use(base_compat_middleware.client_middleware)
    return client


def get_task_report(cairo_client, task_id: str):
    req  = GetTaskReportRequestThrift(task_id=task_id)
    resp = cairo_client.GetTaskReport(req)
    return json.loads(resp.task), json.loads(resp.report)


def submit_and_poll(prompt: str, duration: float,
                    first_frame_url: str, logger: logging.Logger) -> str | None:
    """
    新建 client → 提交任务 → 轮询直到完成，返回 video_url 或 None。
    每次调用都使用全新的 Cairo client。
    """
    cairo_client = make_cairo_client()

    task_input = json.dumps({
        "binary_data": [
            {"data": first_frame_url, "type": "image"}
        ],
        "req_json": {
            "prompt":          prompt,
            "language":        "zh",
            "duration":        duration,
            "seed":            SEED,
            "aspect_ratio":    ASPECT_RATIO,
            "resolution":      RESOLUTION,
            "binary_var_name": ["image"],
            "workflow":        "seedance_2_0_pe_integration.json",
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
        resp    = cairo_client.SubmitAsyncTask(submit_req)
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
            task_report, _ = get_task_report(cairo_client, task_id)
            status = task_report["status"]
            logger.info(f"Poll {task_id} -> {status}")
            print(f"  Polling {task_id} -> {status}")
            if status == "succeeded":
                results   = json.loads(task_report["output"])["results"]
                key       = list(results.keys())[0]
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


# ═══════════════════════════ VIDEO DOWNLOAD & COMPRESS ═══════════════════════

def download_video(url: str, dest: Path, logger: logging.Logger) -> bool:
    """下载视频到本地，返回是否成功。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading {url} → {dest}")
        print(f"  Downloading → {dest.name}")
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  Downloaded: {size_mb:.2f} MB")
        logger.info(f"Download OK: {size_mb:.2f} MB")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"  ERROR download: {e}")
        return False


def compress_video(src: Path, dest: Path, logger: logging.Logger) -> bool:
    """
    将视频压缩到 ≤4MB（不依赖 GT 参考，直接用固定参数）。

    策略：
      - H.264 CRF=28（质量下限）+ maxrate/bufsize（码率上限）
      - 分辨率保持原始（Seedance 输出通常 480p-720p，无需缩小）
      - 音频 AAC 128kbps
    目标：15s 视频 ≈ 2000kbps 总码率 ≈ 3.7MB
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v",    "libx264",
        "-crf",    str(COMPRESS_CRF),
        "-maxrate", COMPRESS_MAXRATE,
        "-bufsize", COMPRESS_BUFSIZE,
        "-preset",  "medium",
        "-pix_fmt", "yuv420p",
        "-c:a",    "aac",
        "-b:a",    COMPRESS_AUDIO,
        "-movflags", "+faststart",
        str(dest),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg failed:\n{result.stderr[-500:]}")
            print(f"  ERROR compress: ffmpeg returned {result.returncode}")
            return False
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  Compressed: {size_mb:.2f} MB → {dest.name}")
        logger.info(f"Compress OK: {size_mb:.2f} MB")
        return True
    except Exception as e:
        logger.error(f"Compress exception: {e}")
        print(f"  ERROR compress: {e}")
        return False


# ═══════════════════════════ CSV HELPERS ═════════════════════════════════════

def load_csv_index() -> dict[tuple, dict]:
    index = {}
    if not CSV_PATH.exists():
        return index
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["vid_label"], row["prompt_level"])
            index[key] = row
    return index


def save_csv_index(index: dict[tuple, dict]) -> None:
    rows = sorted(index.values(), key=lambda r: (r["idx"], r["prompt_level"]))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def record_result(index: dict, vid_label: str, video_id: str,
                  idx: int, score: int, level: int, duration: float,
                  prompt_words: int, video_url: str,
                  local_path: str, compressed_path: str) -> None:
    key = (vid_label, str(level))
    index[key] = {
        "vid_label":       vid_label,
        "video_id":        video_id,
        "idx":             str(idx),
        "score":           str(score),
        "prompt_level":    str(level),
        "duration":        str(duration),
        "prompt_words":    str(prompt_words),
        "video_url":       video_url,
        "local_path":      local_path,
        "compressed_path": compressed_path,
        "generated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
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

        with p0_path.open(encoding='utf-8')  as fp: phase0   = json.load(fp)
        with p1_path.open(encoding='utf-8')  as fp: phase1   = json.load(fp)
        with p15_path.open(encoding='utf-8') as fp: phase1_5 = json.load(fp)

        first_frame_url = get_first_frame_url(vid_label)

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
                print(f"  [SKIP] already in CSV: {csv_index[csv_key]['video_url']}")
                continue

            # ── Build prompt ─────────────────────────────────────────────────
            prompt, duration = build_whole_video_prompt(phase0, phase1, phase1_5, lv)
            prompt_words = len(prompt.split())
            logger.info(f"duration={duration}s  prompt_words={prompt_words}")
            logger.info(f"Full prompt:\n{prompt}")
            print(f"  Duration: {duration}s")
            print(f"  Prompt (~{prompt_words} words):\n    {prompt[:200].replace(chr(10), ' ')}...")
            print(f"  First frame URL: {first_frame_url}")

            # ── Submit & poll（每次新建 client）──────────────────────────────
            video_url = submit_and_poll(prompt, duration, first_frame_url, logger)
            if not video_url:
                continue

            # ── 下载原版视频 ──────────────────────────────────────────────────
            out_dir   = OUT_ROOT / f"level_{lv}"
            comp_dir  = OUT_ROOT / f"level_{lv}_compressed"
            filename  = f"{vid_label}_plevel{lv}.mp4"
            local_path      = out_dir  / filename
            compressed_path = comp_dir / filename

            dl_ok = download_video(video_url, local_path, logger)
            if not dl_ok:
                # 仍记录 URL，本地路径留空
                record_result(csv_index, vid_label, video_id,
                              rec_idx, rec_score, lv, duration,
                              prompt_words, video_url, "", "")
                continue

            # ── 压缩视频（目标 ≤4MB）─────────────────────────────────────────
            comp_ok = compress_video(local_path, compressed_path, logger)

            record_result(
                csv_index, vid_label, video_id,
                rec_idx, rec_score, lv, duration,
                prompt_words, video_url,
                str(local_path),
                str(compressed_path) if comp_ok else "",
            )
            logger.info(f"Done: local={local_path}  compressed={compressed_path}")

    print(f"\n{'='*70}")
    print(f"All done. CSV: {CSV_PATH}")