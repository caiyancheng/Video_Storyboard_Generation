"""
submit_storyboard_to_seedance.py  (服务器运行，需要 euler + cairo_v2)

遍历评分 4/5 分的视频，读取对应的 storyboard txt，
使用 TOS 首帧 URL 提交 Seedance it2v 生成任务，
结果（video_url）写入 CSV，不下载视频文件。

首帧 TOS URL 规则（与 test_storyboard_seedance_2_0_it2v_filter_15s_data.py 一致）：
  https://tosv.byted.org/obj/dm-stickers-rec-sg/
    tt_template_1400k_15s_video_sample/first_frame/{stem}_first.png

Storyboard txt 来源（generate_storyboard_from_image.py 的输出）：
  STORYBOARD_OUT_DIR/{stem}.txt

用法：
  python submit_storyboard_to_seedance.py              # 全量
  python submit_storyboard_to_seedance.py --overwrite  # 重新提交已有 CSV 记录
"""

import csv
import json
import re
import sys
import time
import logging
import argparse
from pathlib import Path

# ── 项目根目录 ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── editing_magic_prompt（服务器路径）────────────────────────────────────────
_EMP_ROOT = Path("/mnt/bn/yilin4/yancheng/Py_codes/editing_magic_prompt")
if str(_EMP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EMP_ROOT))

import euler
from euler import base_compat_middleware
from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task
from cairo_v2.idls.thrift import GetTaskReportRequestThrift

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

SCORED_FILE = Path(
    "/mnt/bn/yilin4/yancheng/Datasets/"
    "tt_template_hq_publish_data_1400k_USAU"
    ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl"
)

STORYBOARD_OUT_DIR = Path(
    "/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/generated_storyboard"
)

CSV_PATH = STORYBOARD_OUT_DIR / "generation_results.csv"
LOG_DIR  = STORYBOARD_OUT_DIR / "logs"

FIRST_FRAME_TOS_BASE = (
    "https://tosv.byted.org/obj/dm-stickers-rec-sg"
    "/tt_template_1400k_15s_video_sample/first_frame"
)

WORKFLOW_ID  = "seedance_2_0_ti2v_e2e_with_pe_test_inference_only_v2"
ASPECT_RATIO = "9:16"
RESOLUTION   = "480p"
SEED         = 42
TARGET_SCORES = {4, 5}

CSV_FIELDS = ["vid_label", "video_id", "idx", "score",
              "prompt_words", "duration", "video_url", "generated_at"]


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


def load_csv_index() -> dict[str, dict]:
    index = {}
    if not CSV_PATH.exists():
        return index
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            index[row["vid_label"]] = row
    return index


def save_csv_index(index: dict[str, dict]):
    rows = sorted(index.values(), key=lambda r: (r["idx"], r["vid_label"]))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def record_result(index, vid_label, video_id, idx, score,
                  prompt_words, duration, video_url):
    index[vid_label] = {
        "vid_label":    vid_label,
        "video_id":     video_id,
        "idx":          str(idx),
        "score":        str(score),
        "prompt_words": str(prompt_words),
        "duration":     str(duration),
        "video_url":    video_url,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_csv_index(index)


# ═══════════════════════ PROMPT UTILS ════════════════════════════════════════

def parse_duration_from_prompt(prompt_text: str) -> float:
    """从 prompt 头部解析时长，兼容两种格式：
       旧：[Whole-video generation | 15.0s | Level 4]
       新：[Whole-video generation | 15.0s]
    """
    m = re.search(r"\|\s*([\d.]+)s\s*[\|\]]", prompt_text)
    if m:
        return round(max(4.0, min(15.0, float(m.group(1)))), 3)
    return 15.0


# ═══════════════════════ CAIRO ═══════════════════════════════════════════════

def setup_cairo_client():
    client = euler.Client(
        CairoService,
        target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva",
        transport="ttheader"
    )
    client.use(base_compat_middleware.client_middleware)
    return client


def submit_and_poll(cairo_client, prompt: str, duration: float,
                    first_frame_url: str, logger: logging.Logger) -> str | None:
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
            "workflow":        "seedance_2_0_pe_integration.json"
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
        print(f"  [ERROR] Submit: {e}")
        return None

    gen_start = time.time()
    while True:
        try:
            req  = GetTaskReportRequestThrift(task_id=task_id)
            resp = cairo_client.GetTaskReport(req)
            task_report = json.loads(resp.task)
            status = task_report["status"]
            logger.info(f"Poll {task_id} → {status}")
            print(f"  Polling {task_id} → {status}")

            if status == "succeeded":
                results   = json.loads(task_report["output"])["results"]
                key       = list(results.keys())[0]
                video_url = f"https://tosv-sg.tiktok-row.org/obj/iccv-vpipe-sg/{key}"
                logger.info(f"Succeeded! url={video_url}")
                print(f"  ✓ 生成成功！URL: {video_url}")
                print(f"  耗时: {time.time() - gen_start:.1f}s")
                return video_url
            elif status in ("failed", "cancelled"):
                logger.error(f"Task ended: {status}  output={task_report.get('output')}")
                print(f"  ✗ 任务结束: {status}")
                print(task_report.get("output"))
                return None
        except Exception as e:
            logger.warning(f"Poll error: {e}")
            print(f"  Poll error: {e}")
        time.sleep(5)


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="批量提交 storyboard prompt 到 Seedance（服务器运行）"
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="重新提交已有 CSV 记录")
    args = parser.parse_args()

    STORYBOARD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    records   = load_scored_records()
    csv_index = load_csv_index()
    print(f"评分记录数      : {len(records)}")
    print(f"已有 CSV 记录   : {len(csv_index)}")

    cairo_client = setup_cairo_client()

    ok = skip_no_txt = skip_csv = fail = 0

    for rec in records:
        video_id  = rec.get("video_id", "unknown")
        rec_idx   = rec.get("_idx",   0)
        rec_score = rec.get("_score", 0)
        stem      = f"id_{rec_idx:04d}_score{rec_score}_{video_id}"
        vid_label = stem

        print(f"\n{'='*70}")
        print(f"Processing: {vid_label}  score={rec_score}")

        # 检查 storyboard txt 是否存在
        txt_path = STORYBOARD_OUT_DIR / f"{stem}.txt"
        if not txt_path.exists():
            print(f"  [SKIP] storyboard txt 不存在，请先运行 generate_storyboard_from_image.py")
            skip_no_txt += 1
            continue

        # 检查 CSV（skip already done）
        if vid_label in csv_index and not args.overwrite:
            print(f"  [SKIP] 已在 CSV: {csv_index[vid_label]['video_url']}")
            skip_csv += 1
            continue

        # 读取 prompt
        prompt_text  = txt_path.read_text(encoding="utf-8")
        duration     = parse_duration_from_prompt(prompt_text)
        prompt_words = len(prompt_text.split())
        print(f"  Duration: {duration}s  Words: {prompt_words}")

        # TOS 首帧 URL（与 test_storyboard... 脚本相同）
        first_frame_url = f"{FIRST_FRAME_TOS_BASE}/{stem}_first.png"
        print(f"  首帧 URL: {first_frame_url}")

        # Logger
        logger_name = f"{vid_label}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            fh = logging.FileHandler(LOG_DIR / f"{logger_name}.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)

        # 提交 & 轮询
        video_url = submit_and_poll(cairo_client, prompt_text, duration,
                                    first_frame_url, logger)
        if not video_url:
            fail += 1
            continue

        record_result(csv_index, vid_label, video_id,
                      rec_idx, rec_score, prompt_words, duration, video_url)
        print(f"  CSV 已更新: {CSV_PATH.name}")
        ok += 1

    print(f"\n{'='*70}")
    print(f"完成：成功 {ok}，storyboard 缺失跳过 {skip_no_txt}，"
          f"CSV 已有跳过 {skip_csv}，失败 {fail}")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
