"""
submit_storyboard_to_seedance.py  (服务器运行，需要 euler + cairo_v2)

读取由 generate_storyboard_from_image.py 生成的故事板 txt，
将图片上传到 TOS，然后提交 Seedance it2v 生成任务。

用法：
  python submit_storyboard_to_seedance.py \\
      --prompt /path/to/storyboard.txt \\
      --image  /path/to/first_frame.jpg

  python submit_storyboard_to_seedance.py \\
      --prompt /path/to/storyboard.txt \\
      --image  /path/to/first_frame.jpg \\
      --out-csv /path/to/results.csv
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

# ── 项目根目录（用于 import TOS_BUCKET_VA 和 Storyboard_generation 模块）──────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── editing_magic_prompt（服务器路径）────────────────────────────────────────
_EMP_ROOT = Path("/mnt/bn/yilin4/Py_codes/editing_magic_prompt")
if str(_EMP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EMP_ROOT))

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

TOS_BUCKET   = "dm-stickers-rec-sg"
TOS_KEY_PREFIX = "tt_template_1400k_15s_video_sample/generated_storyboard_frames"

DEFAULT_CSV = Path(
    "/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/generated_storyboard"
    "/seedance_results.csv"
)

WORKFLOW_ID  = "seedance_2_0_ti2v_e2e_with_pe_test_inference_only_v2"
ASPECT_RATIO = "9:16"
RESOLUTION   = "480p"
SEED         = 42

CSV_FIELDS = ["stem", "image_path", "prompt_path", "duration",
              "prompt_words", "video_url", "generated_at", "error"]


# ═══════════════════════ IMPORT CAIRO (服务器专用) ════════════════════════════

def _import_cairo():
    try:
        import euler
        from euler import base_compat_middleware
        from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task
        from cairo_v2.idls.thrift import GetTaskReportRequestThrift
        return euler, base_compat_middleware, CairoService, SubmitAsyncTaskRequest, Task, GetTaskReportRequestThrift
    except ImportError as e:
        print(f"[ERROR] Cairo/Euler 不可用（需在服务器运行）: {e}")
        sys.exit(1)


# ═══════════════════════ TOS UPLOAD ══════════════════════════════════════════

def upload_image_to_tos(image_path: Path, stem: str) -> str:
    """上传图片到 TOS，返回可访问的 HTTPS URL。"""
    try:
        import tos as tos_sdk
    except ImportError:
        raise RuntimeError("tos SDK 不可用，请安装：pip install tos")

    from TOS_BUCKET_VA import ak  # noqa: E402
    # secret key 从环境变量读取
    import os
    sk = os.environ.get("TOS_SK", "")
    if not sk:
        raise RuntimeError("请设置环境变量 TOS_SK（TOS Secret Key）")

    client = tos_sdk.TosClientV2(
        ak=ak, sk=sk,
        endpoint="https://tosv.byted.org",
        region="cn-beijing",
    )

    tos_key  = f"{TOS_KEY_PREFIX}/{stem}{image_path.suffix}"
    tos_url  = f"https://tosv.byted.org/obj/{TOS_BUCKET}/{tos_key}"

    with image_path.open("rb") as f:
        client.put_object(bucket=TOS_BUCKET, key=tos_key, content=f)

    print(f"  ✓ 已上传到 TOS: {tos_url}")
    return tos_url


# ═══════════════════════ PARSE DURATION ══════════════════════════════════════

def parse_duration_from_prompt(prompt_text: str) -> float:
    """从 prompt 头部 [Whole-video generation | {duration}s | Level 4] 解析时长。"""
    m = re.search(r"\|\s*([\d.]+)s\s*\|", prompt_text)
    if m:
        return round(max(4.0, min(15.0, float(m.group(1)))), 3)
    return 10.0   # 默认 10s


# ═══════════════════════ CAIRO SUBMIT & POLL ═════════════════════════════════

def setup_cairo_client(euler, base_compat_middleware, CairoService):
    client = euler.Client(
        CairoService,
        target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva",
        transport="ttheader"
    )
    client.use(base_compat_middleware.client_middleware)
    return client


def submit_and_poll(cairo_client, prompt: str, duration: float,
                    first_frame_url: str,
                    SubmitAsyncTaskRequest, Task, GetTaskReportRequestThrift) -> str | None:
    """提交 Seedance 任务，轮询直到完成，返回 video_url 或 None。"""
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
        print(f"  Submitted task_id: {task_id}")
    except Exception as e:
        print(f"  [ERROR] Submit failed: {e}")
        return None

    gen_start = time.time()
    while True:
        try:
            req = GetTaskReportRequestThrift(task_id=task_id)
            resp = cairo_client.GetTaskReport(req)
            task_report = json.loads(resp.task)
            status = task_report["status"]
            print(f"  Polling {task_id} → {status}")

            if status == "succeeded":
                results = json.loads(task_report["output"])["results"]
                key = list(results.keys())[0]
                video_url = f"https://tosv-sg.tiktok-row.org/obj/iccv-vpipe-sg/{key}"
                print(f"  ✓ 生成成功！URL: {video_url}")
                print(f"  耗时: {time.time() - gen_start:.1f}s")
                return video_url
            elif status in ("failed", "cancelled"):
                print(f"  ✗ 任务结束: {status}")
                print(f"  output: {task_report.get('output')}")
                return None
        except Exception as e:
            print(f"  Poll error: {e}")

        time.sleep(5)


# ═══════════════════════ CSV ════════════════════════════════════════════════

def append_csv(csv_path: Path, row: dict):
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in CSV_FIELDS})


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def run(prompt_path: Path, image_path: Path, out_csv: Path):
    euler, base_compat_middleware, CairoService, \
    SubmitAsyncTaskRequest, Task, GetTaskReportRequestThrift = _import_cairo()

    stem = prompt_path.stem

    print(f"\n{'='*60}")
    print(f"Stem    : {stem}")
    print(f"Prompt  : {prompt_path}")
    print(f"Image   : {image_path}")
    print(f"CSV     : {out_csv}")
    print(f"{'='*60}")

    # 读取 prompt
    prompt_text = prompt_path.read_text(encoding="utf-8")
    duration    = parse_duration_from_prompt(prompt_text)
    prompt_words = len(prompt_text.split())
    print(f"  Duration: {duration}s  Words: {prompt_words}")

    # 上传图片到 TOS
    print("  上传首帧图片到 TOS...")
    try:
        first_frame_url = upload_image_to_tos(image_path, stem)
    except Exception as e:
        print(f"  [ERROR] TOS 上传失败: {e}")
        append_csv(out_csv, {
            "stem": stem, "image_path": str(image_path),
            "prompt_path": str(prompt_path), "duration": duration,
            "prompt_words": prompt_words, "video_url": "",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e),
        })
        return None

    # 提交 Seedance
    print("  提交 Seedance 生成任务...")
    cairo_client = setup_cairo_client(euler, base_compat_middleware, CairoService)
    video_url = submit_and_poll(
        cairo_client, prompt_text, duration, first_frame_url,
        SubmitAsyncTaskRequest, Task, GetTaskReportRequestThrift
    )

    row = {
        "stem": stem,
        "image_path":   str(image_path),
        "prompt_path":  str(prompt_path),
        "duration":     duration,
        "prompt_words": prompt_words,
        "video_url":    video_url or "",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error":        "" if video_url else "generation failed",
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    append_csv(out_csv, row)
    print(f"  CSV 已更新: {out_csv}")
    return video_url


def main():
    parser = argparse.ArgumentParser(
        description="将故事板 prompt + 图片提交到 Seedance（服务器运行）"
    )
    parser.add_argument("--prompt", required=True,
                        help="generate_storyboard_from_image.py 生成的 .txt 文件路径")
    parser.add_argument("--image",  required=True,
                        help="首帧图片路径（.jpg/.png 等），将上传到 TOS")
    parser.add_argument("--out-csv", default=str(DEFAULT_CSV),
                        help=f"结果 CSV 路径（默认：{DEFAULT_CSV}）")
    args = parser.parse_args()

    run(
        prompt_path=Path(args.prompt).expanduser().resolve(),
        image_path=Path(args.image).expanduser().resolve(),
        out_csv=Path(args.out_csv).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
