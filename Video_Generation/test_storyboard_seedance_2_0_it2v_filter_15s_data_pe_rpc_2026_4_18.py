"""
test_storyboard_seedance_2_0_it2v_filter_15s_data_pe_rpc_2026_4_18.py

每条记录生成 6 种视频，覆盖全部对比组合：
  Lv1_pe  / Lv2_pe  / Lv3_pe   ← 故事线 prompt 先经 PE RPC 改写，再送 Cairo
  Lv1_nope / Lv2_nope / Lv3_nope ← 故事线 prompt 直接送 Cairo，不经 PE

跳过策略：
  - v16 phase1 文件不存在 → 打印 [SKIP-NO-V16]，跳过整条记录
  - 首帧 PNG 不存在       → 打印 [SKIP-NO-FRAME]，跳过整条记录
  - PE RPC 失败           → 打印 [SKIP-PE-FAIL]，跳过该 variant
  - Cairo 生成失败        → 打印 [SKIP-GEN-FAIL]，跳过该 variant
"""

import csv
import json
import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import euler
from euler import base_compat_middleware
from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task
from cairo_v2.idls.thrift import GetTaskReportRequestThrift

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vproxy import VproxyClient

# ═══════════════════════════ CONFIG ══════════════════════════════════════════

# 6 种对比变体：(level, use_pe)
# variant_key = "Lv1_pe" / "Lv1_nope" / ...
VARIANTS: list[tuple[str, bool]] = [
    ("Lv1", True),
    ("Lv1", False),
    ("Lv2", True),
    ("Lv2", False),
    ("Lv3", True),
    ("Lv3", False),
]

SCORED_FILE = Path(
    "/mnt/bn/yilin4/yancheng/Datasets/tt_template_hq_publish_data_1400k_USAU"
    ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl"
)

LABEL_ROOT      = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
                       "/shu_inverse_label")
PHASE1_V16_DIR  = LABEL_ROOT / "phase1_chinese_labels_v16"
FIRST_FRAME_DIR = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
                       "/first_frame")

OUT_ROOT    = LABEL_ROOT / "generated_videos_v2_pe_rpc"
LOG_DIR     = OUT_ROOT / "logs"
PE_CSV_PATH  = OUT_ROOT / "pe_results.csv"
GEN_CSV_PATH = OUT_ROOT / "generation_results.csv"

PE_CSV_FIELDS = [
    "vid_label", "video_id", "idx", "score", "prompt_level",
    "duration", "prompt_words",
    "original_prompt", "pe_prompt_json",
    "dynamic_caption", "dynamic_caption_chars",
    "pe_at",
]

GEN_CSV_FIELDS = [
    "vid_label", "video_id", "idx", "score", "prompt_level",
    "duration", "final_prompt_words",
    "video_url", "local_path", "compressed_path", "generated_at",
]

TARGET_SCORES = {4, 5}

# PE RPC
PE_REQ_KEY  = "tt_seedance2_pe_test"
PE_ASPECT   = "9:16"
PE_LANGUAGE = "zh"

# Cairo
FIRST_FRAME_TOS_BASE = ("https://tosv.byted.org/obj/dm-stickers-rec-sg"
                        "/tt_template_1400k_15s_video_sample/first_frame")
WORKFLOW_ID  = "seedance_2_0_ti2v_e2e_with_pe_test_inference_only_v2"
ASPECT_RATIO = "9:16"
RESOLUTION   = "480p"
SEED         = 42

# 压缩参数
COMPRESS_CRF     = 28
COMPRESS_MAXRATE = "2200k"
COMPRESS_BUFSIZE = "4400k"
COMPRESS_AUDIO   = "128k"

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


def find_v16_file(video_id: str, idx: int, score: int) -> Path | None:
    exact = PHASE1_V16_DIR / f"id_{idx:04d}_score{score}_{video_id}_phase1.json"
    if exact.exists():
        return exact
    matches = list(PHASE1_V16_DIR.glob(f"*{video_id}_phase1.json"))
    return matches[0] if matches else None


def find_first_frame(video_id: str, idx: int, score: int) -> Path | None:
    exact = FIRST_FRAME_DIR / f"id_{idx:04d}_score{score}_{video_id}_first.png"
    if exact.exists():
        return exact
    matches = list(FIRST_FRAME_DIR.glob(f"*{video_id}_first.png"))
    return matches[0] if matches else None


def get_first_frame_url(vid_label: str) -> str:
    return f"{FIRST_FRAME_TOS_BASE}/{vid_label}_first.png"


# ═══════════════════════════ PROMPT BUILDING (v16 → Lv1/Lv2/Lv3) ════════════

def _parse_time_range(time_range: str) -> tuple[float, float]:
    start_str, end_str = time_range.split("-")
    def to_sec(ts):
        mm, ss = ts.split(":")
        return int(mm) * 60 + float(ss)
    return to_sec(start_str), to_sec(end_str)


def _speech_lines(shot: dict) -> list[str]:
    lines = []
    for sp in shot.get("audio", {}).get("speech", []):
        for ev in sp.get("speech_events", []):
            ts = ev.get("timestamp", "")
            transcript = ev.get("transcript", "")
            style = ev.get("style", "")
            quality = ev.get("voice_fingerprint", {}).get("voice_quality", "")
            if transcript:
                line = f"  [{ts}] 台词: {transcript}"
                details = []
                if style:
                    details.append(f"语气: {style}")
                if quality:
                    details.append(f"声音: {quality}")
                if details:
                    line += "  |  " + "  ".join(details)
                lines.append(line)
    return lines


def _sfx_lines(shot: dict) -> list[str]:
    lines = []
    for sfx in shot.get("audio", {}).get("sound_effects", []):
        for ev in sfx.get("sound_events", []):
            ts = ev.get("timestamp", "")
            desc = ev.get("description", "")
            if desc:
                lines.append(f"  [{ts}] 音效: {desc}")
    return lines


def _music_lines(shot: dict, lv: int) -> list[str]:
    lines = []
    for mus in shot.get("audio", {}).get("music", []):
        ts = mus.get("timestamp", "")
        style = mus.get("style", "")
        mf = mus.get("music_features", {})
        instru = mf.get("instrumentation", "")
        impact = mf.get("emotional_impact", "")
        if style:
            line = f"  [{ts}] 音乐: {style}"
            if impact:
                line += f"  情绪: {impact}"
            if lv >= 3 and instru:
                line += f"  乐器: {instru}"
            lines.append(line)
    return lines


def _camera_lines(shot: dict, lv: int) -> list[str]:
    cam = shot.get("visual", {}).get("camera", {})
    if not cam:
        return []
    lines = []
    angle = cam.get("camera_angle", "")
    scale = cam.get("shot_scale", "")
    mv = cam.get("camera_movement", {})
    traj = mv.get("movement_trajectory", "")
    speed = mv.get("speed", "")
    mood = cam.get("mood_atmosphere", "")
    color_tone = cam.get("color_tone", "")
    if angle or scale:
        lines.append(f"  角度/景别: {angle}  |  {scale}")
    if traj:
        lines.append(f"  运动轨迹: {traj}" + (f"  速度: {speed}" if speed else ""))
    if lv >= 2 and color_tone:
        lines.append(f"  色调: {color_tone}")
    if lv >= 2 and mood:
        lines.append(f"  氛围: {mood}")
    return lines


def _subject_lines(shot: dict, lv: int) -> list[str]:
    lines = []
    for subj in shot.get("visual", {}).get("subject_tracking", []):
        sid = subj.get("id", "")
        ts_str = ", ".join(subj.get("timestamp_presence", []))
        lines.append(f"  {sid}  [{ts_str}]")
        if subj.get("facial_expression"):
            lines.append(f"    表情: {subj['facial_expression']}")
        if subj.get("body_movement"):
            lines.append(f"    肢体动作: {subj['body_movement']}")
        if subj.get("action"):
            lines.append(f"    行为: {subj['action']}")
        if lv >= 3 and subj.get("appearance_description"):
            lines.append(f"    外观: {subj['appearance_description']}")
    return lines


def _location_lines(shot: dict) -> list[str]:
    lines = []
    for loc in shot.get("visual", {}).get("location_tracking", []):
        lid = loc.get("id", "")
        env = loc.get("visual_environment", "")
        ts_str = ", ".join(loc.get("timestamp_presence", []))
        if env:
            lines.append(f"  {lid}  [{ts_str}]")
            lines.append(f"    {env}")
    return lines


def _visual_style_lines(shot: dict) -> list[str]:
    vs = shot.get("visual", {}).get("visual_style", {})
    if not vs:
        return []
    lines = []
    if vs.get("art_style"):
        lines.append(f"  美术风格: {vs['art_style']}")
    if vs.get("color_palette"):
        lines.append(f"  色彩: {vs['color_palette']}")
    la = vs.get("lighting_analysis", {})
    parts = [f"{k}={la[k]}" for k in ("brightness", "saturation", "contrast") if la.get(k)]
    if parts:
        lines.append(f"  光线: {',  '.join(parts)}")
    return lines


def _vfx_lines(shot: dict) -> list[str]:
    lines = []
    for vfx in shot.get("visual", {}).get("visual_effects", []):
        ts = vfx.get("timestamp", "")
        desc = vfx.get("description", "")
        if desc:
            line = f"  [{ts}] 特效({vfx.get('effect_type', '')}): {desc}"
            if vfx.get("color"):
                line += f"  颜色: {vfx['color']}"
            if vfx.get("intensity"):
                line += f"  强度: {vfx['intensity']}"
            lines.append(line)
    return lines


def build_prompt(phase1_v16: dict, level: str) -> tuple[str, float]:
    """返回 (prompt_text, duration_seconds)。"""
    lv = int(level[2])
    shots = phase1_v16.get("shot_registry", [])
    video_id = phase1_v16.get("video_id", "")

    # 从最后一个 shot 的 time_range 计算总时长
    if shots:
        _, end_sec = _parse_time_range(shots[-1].get("time_range", "00:00.000-00:00.000"))
        duration = round(max(4.0, min(15.0, end_sec)), 3)
    else:
        duration = 4.0

    sections = [f"[视频生成 Prompt | {level}]\nVideo ID: {video_id}"]

    for shot in shots:
        block = [f"── {shot.get('id', '')}  {shot.get('time_range', '')} ──"]
        block.append(f"概述: {shot.get('caption', '')}")

        cam = _camera_lines(shot, lv)
        if cam:
            block.append("摄像机:")
            block.extend(cam)

        speech = _speech_lines(shot)
        if speech:
            block.append("台词/语音:")
            block.extend(speech)

        sfx = _sfx_lines(shot)
        if sfx:
            block.append("音效:")
            block.extend(sfx)

        if lv >= 2:
            music = _music_lines(shot, lv)
            if music:
                block.append("音乐:")
                block.extend(music)

        subj = _subject_lines(shot, lv)
        if subj:
            block.append("角色:")
            block.extend(subj)

        if lv >= 2:
            loc = _location_lines(shot)
            if loc:
                block.append("场景/背景:")
                block.extend(loc)
            style = _visual_style_lines(shot)
            if style:
                block.append("视觉风格:")
                block.extend(style)

        if lv >= 3:
            vfx = _vfx_lines(shot)
            if vfx:
                block.append("视觉特效:")
                block.extend(vfx)

            ns = shot.get("narrative_structure", {})
            ns_parts = []
            if ns.get("style_features"):
                ns_parts.append(f"  [风格特点] {ns['style_features']}")
            if ns.get("content_summary"):
                ns_parts.append(f"  [内容总结] {ns['content_summary']}")
            if ns.get("dynamic_description"):
                ns_parts.append(f"  [动态描述] {ns['dynamic_description']}")
            if ns.get("static_description"):
                ns_parts.append(f"  [静态描述] {ns['static_description']}")
            ai = ns.get("additional_info", {})
            if ai.get("duration") or ai.get("aspect_ratio"):
                ns_parts.append(f"  [附加信息] 时长={ai.get('duration', '')}  宽高比={ai.get('aspect_ratio', '')}")
            if ns_parts:
                block.append("叙事结构:")
                block.extend(ns_parts)

            if shot.get("dense_caption"):
                block.append(f"详细描述:\n  {shot['dense_caption']}")

        sections.append("\n".join(block))

    prompt = "\n\n".join(sections) + COPYRIGHT_NOTICE
    return prompt, duration


# ═══════════════════════════ PE RPC ══════════════════════════════════════════

def call_pe_rpc(
    prompt: str,
    duration: float,
    first_frame_path: Path,
    logger: logging.Logger,
    retries: int = 3,
    retry_delay: float = 10.0,
) -> dict | None:
    req_json = json.dumps({
        "prompt":          prompt,
        "duration":        duration,
        "aspect_ratio":    PE_ASPECT,
        "language":        PE_LANGUAGE,
        "with_audio":      True,
        "version":         "v2.0",
        "binary_var_name": ["image"],
        "task_type":       "IT2V",
    }, ensure_ascii=False)

    img_bytes = first_frame_path.read_bytes()

    for attempt in range(retries):
        try:
            resp = VproxyClient().process(PE_REQ_KEY, req_json, [img_bytes])
            pe_result = json.loads(resp.resp_json)
            logger.info(f"PE RPC OK: {str(pe_result)[:200]}")
            return pe_result
        except Exception as e:
            logger.warning(f"PE RPC attempt {attempt + 1}/{retries} failed: {e}")
            print(f"    [PE] attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
    return None


def extract_dynamic_caption(pe_result: dict) -> str:
    """IT2V 路径：rephrase_result_zh.dynamic_caption"""
    rephrase = pe_result.get("rephrase_result_zh", {})
    if isinstance(rephrase, str):
        try:
            rephrase = json.loads(rephrase)
        except Exception:
            return ""
    return rephrase.get("dynamic_caption", "") if isinstance(rephrase, dict) else ""


# ═══════════════════════════ CAIRO ═══════════════════════════════════════════

def make_cairo_client():
    client = euler.Client(
        CairoService,
        target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva",
        transport="ttheader"
    )
    client.use(base_compat_middleware.client_middleware)
    return client


POLL_TIMEOUT_SEC = 1800   # 单个任务最长等待 30 分钟，超时视为失败


def submit_and_poll(
    prompt: str,
    duration: float,
    first_frame_url: str,
    logger: logging.Logger,
) -> str | None:
    cairo_client = make_cairo_client()

    task_input = json.dumps({
        "binary_data": [{"data": first_frame_url, "type": "image"}],
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
    submit_req.task = Task(input=task_input, priority=7, tags={"second_biz_name": "test"})
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

    video_url = None
    gen_start = time.time()
    while True:
        if time.time() - gen_start > POLL_TIMEOUT_SEC:
            logger.error(f"Poll timeout after {POLL_TIMEOUT_SEC}s for task_id={task_id}")
            print(f"  ERROR: poll timeout ({POLL_TIMEOUT_SEC}s) for task_id={task_id}")
            return None
        try:
            req = GetTaskReportRequestThrift(task_id=task_id)
            resp = cairo_client.GetTaskReport(req)
            task_report = json.loads(resp.task)
            status = task_report.get("status", "unknown")
            logger.info(f"Poll {task_id} -> {status}")
            print(f"  Polling {task_id} -> {status}")
            if status == "succeeded":
                output = json.loads(task_report.get("output", "{}"))
                results = output.get("results", {})
                if not results:
                    logger.error(f"succeeded but results empty: {output}")
                    print(f"  ERROR: succeeded but results empty")
                    return None
                key = list(results.keys())[0]
                video_url = f"https://tosv-sg.tiktok-row.org/obj/iccv-vpipe-sg/{key}"
                logger.info(f"Succeeded! video_url={video_url}")
                print(f"  Succeeded! URL: {video_url}")
                break
            elif status in ("failed", "cancelled"):
                logger.error(f"Task ended: {status}  output={task_report.get('output')}")
                print(f"  Task ended: {status}")
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264",
        "-crf", str(COMPRESS_CRF),
        "-maxrate", COMPRESS_MAXRATE,
        "-bufsize", COMPRESS_BUFSIZE,
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", COMPRESS_AUDIO,
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

def _load_csv(path: Path, fields: list[str]) -> dict[tuple, dict]:
    index = {}
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vid_label = row.get("vid_label") or row.get("\ufeffvid_label", "")
            level = row.get("prompt_level", "")
            if vid_label:
                index[(vid_label, level)] = row
    return index


def _save_csv(path: Path, fields: list[str], index: dict) -> None:
    rows = sorted(index.values(), key=lambda r: (r.get("idx", ""), r.get("prompt_level", "")))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def record_pe(index: dict, vid_label: str, video_id: str, idx: int, score: int,
              level: str, duration: float, prompt_words: int, original_prompt: str,
              pe_result: dict, dynamic_caption: str) -> None:
    key = (vid_label, level)
    index[key] = {
        "vid_label":            vid_label,
        "video_id":             video_id,
        "idx":                  str(idx),
        "score":                str(score),
        "prompt_level":         level,
        "duration":             str(duration),
        "prompt_words":         str(prompt_words),
        "original_prompt":      original_prompt,
        "pe_prompt_json":       json.dumps(pe_result, ensure_ascii=False),
        "dynamic_caption":      dynamic_caption,
        "dynamic_caption_chars": str(len(dynamic_caption)),
        "pe_at":                time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_csv(PE_CSV_PATH, PE_CSV_FIELDS, index)


def record_gen(index: dict, vid_label: str, video_id: str, idx: int, score: int,
               level: str, duration: float, final_prompt_words: int,
               video_url: str, local_path: str, compressed_path: str) -> None:
    key = (vid_label, level)
    index[key] = {
        "vid_label":         vid_label,
        "video_id":          video_id,
        "idx":               str(idx),
        "score":             str(score),
        "prompt_level":      level,
        "duration":          str(duration),
        "final_prompt_words": str(final_prompt_words),
        "video_url":         video_url,
        "local_path":        local_path,
        "compressed_path":   compressed_path,
        "generated_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_csv(GEN_CSV_PATH, GEN_CSV_FIELDS, index)


# ═══════════════════════════ MAIN ════════════════════════════════════════════

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pe_index  = _load_csv(PE_CSV_PATH,  PE_CSV_FIELDS)
    gen_index = _load_csv(GEN_CSV_PATH, GEN_CSV_FIELDS)
    all_records = load_scored_records()

    print(f"PE  CSV: {len(pe_index)} existing entries")
    print(f"Gen CSV: {len(gen_index)} existing entries")
    print(f"Records: {len(all_records)} with score {sorted(TARGET_SCORES)}")
    variant_names = [f"{l}_{'pe' if p else 'nope'}" for l, p in VARIANTS]
    print(f"Variants: {variant_names}\n")

    for rec in all_records:
        video_id  = rec.get("video_id", "unknown")
        rec_idx   = rec.get("_idx", 0)
        rec_score = rec.get("_score", 0)

        # ── 查找 v16 phase1 文件 ─────────────────────────────────────────────
        v16_path = find_v16_file(video_id, rec_idx, rec_score)
        if v16_path is None:
            print(f"[SKIP-NO-V16] {video_id}")
            continue

        # 从文件名解析准确的 idx / score
        stem_parts = v16_path.stem.split("_")
        try:
            rec_idx   = int(stem_parts[1])
            rec_score = int(stem_parts[2].replace("score", ""))
        except (IndexError, ValueError):
            pass

        vid_label = f"id_{rec_idx:04d}_score{rec_score}_{video_id}"

        # ── 查找首帧 ────────────────────────────────────────────────────────
        first_frame_path = find_first_frame(video_id, rec_idx, rec_score)
        if first_frame_path is None:
            print(f"[SKIP-NO-FRAME] {vid_label}")
            continue

        # ── 加载 v16 JSON ───────────────────────────────────────────────────
        try:
            with v16_path.open(encoding="utf-8") as fp:
                phase1_v16 = json.load(fp)
        except Exception as e:
            print(f"[SKIP-JSON-ERR] {vid_label}: {e}")
            continue

        first_frame_url = get_first_frame_url(vid_label)

        for level, use_pe in VARIANTS:
            variant = f"{level}_{'pe' if use_pe else 'nope'}"
            gen_key = (vid_label, variant)

            logger_name = f"{vid_label}_{variant}"
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                fh = logging.FileHandler(LOG_DIR / f"{logger_name}.log", encoding="utf-8")
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                logger.addHandler(fh)

            print(f"\n{'='*70}")
            print(f"Processing {vid_label}  score={rec_score}  variant={variant}")
            logger.info(f"=== Start {vid_label} score={rec_score} variant={variant} ===")

            if gen_key in gen_index:
                print(f"  [SKIP] already generated: {gen_index[gen_key].get('video_url', '')[:80]}")
                continue

            # ── 构建故事线 prompt ────────────────────────────────────────────
            try:
                original_prompt, duration = build_prompt(phase1_v16, level)
            except Exception as e:
                print(f"  [SKIP-BUILD-ERR] {e}")
                logger.error(f"build_prompt failed: {e}")
                continue

            prompt_words = len(original_prompt.split())
            logger.info(f"duration={duration}s  prompt_words={prompt_words}")
            print(f"  Duration: {duration}s  |  Prompt words: {prompt_words}")

            # ── PE RPC（仅 use_pe=True 分支）────────────────────────────────
            if use_pe:
                pe_key = (vid_label, level)  # PE 结果按 level 共享（nope/pe 复用同一次 PE 调用）
                if pe_key in pe_index and pe_index[pe_key].get("dynamic_caption"):
                    print(f"  [PE-CACHED] using existing PE result")
                    logger.info("Using cached PE result")
                    dynamic_caption = pe_index[pe_key]["dynamic_caption"]
                else:
                    pe_result = call_pe_rpc(original_prompt, duration, first_frame_path, logger)
                    if pe_result is None:
                        print(f"  [SKIP-PE-FAIL] {vid_label} {variant}")
                        logger.error("PE RPC failed, skipping")
                        continue
                    dynamic_caption = extract_dynamic_caption(pe_result)
                    record_pe(pe_index, vid_label, video_id, rec_idx, rec_score,
                              level, duration, prompt_words, original_prompt,
                              pe_result, dynamic_caption)
                    print(f"  PE OK: dynamic_caption_chars={len(dynamic_caption)}")
                    logger.info(f"dynamic_caption (preview): {dynamic_caption[:200]}")

                if not dynamic_caption:
                    print(f"  [SKIP-PE-EMPTY] dynamic_caption is empty for {vid_label} {variant}")
                    logger.warning("dynamic_caption is empty, skipping generation")
                    continue

                final_prompt = dynamic_caption
            else:
                # nope：直接用故事线 prompt
                final_prompt = original_prompt

            final_prompt_words = len(final_prompt.split())
            print(f"  Final prompt words: {final_prompt_words}  (use_pe={use_pe})")

            # ── Cairo 视频生成 ────────────────────────────────────────────────
            video_url = submit_and_poll(final_prompt, duration, first_frame_url, logger)
            if not video_url:
                print(f"  [SKIP-GEN-FAIL] {vid_label} {variant}")
                continue

            # ── 下载 + 压缩 ──────────────────────────────────────────────────
            out_dir  = OUT_ROOT / variant
            comp_dir = OUT_ROOT / f"{variant}_compressed"
            filename = f"{vid_label}_{variant}.mp4"
            local_path      = out_dir  / filename
            compressed_path = comp_dir / filename

            dl_ok = download_video(video_url, local_path, logger)
            if not dl_ok:
                record_gen(gen_index, vid_label, video_id, rec_idx, rec_score,
                           variant, duration, final_prompt_words,
                           video_url, "", "")
                continue

            comp_ok = compress_video(local_path, compressed_path, logger)
            record_gen(gen_index, vid_label, video_id, rec_idx, rec_score,
                       variant, duration, final_prompt_words,
                       video_url, str(local_path),
                       str(compressed_path) if comp_ok else "")

            logger.info(f"Done: local={local_path}  compressed={compressed_path}")

    print(f"\n{'='*70}")
    print(f"All done.")
    print(f"PE  CSV: {PE_CSV_PATH}")
    print(f"Gen CSV: {GEN_CSV_PATH}")