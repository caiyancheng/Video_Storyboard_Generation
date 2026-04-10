import time
import json
import subprocess
import euler
from euler import base_compat_middleware

# bytedeuler==2.4.3
# byted_cairo==2.6.6
from cairo_v2.idls.thrift import GetTaskReportRequestThrift
from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task

# Keys to exclude from the storyboard JSON before using it as prompt
STORYBOARD_EXCLUDE_KEYS = [
    "video_id",
    "item_id",
    "tos_key",
    "video_url",
]

STORYBOARD_JSON_PATH = "Sample_Prompts/sample_storyboard_phase0_v15_result.json"

if __name__ == "__main__":

    cairo_client = euler.Client(CairoService, target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva",
                                transport="ttheader")  # 上线请切换到aip.tce.cairo_v2
    cairo_client.use(base_compat_middleware.client_middleware)

    workflow_id = "seedance_2_0_ti2v_e2e_with_pe_test_inference_only_v2"  # 新业务场景接入请联系 @george.zhang01 提供新的workflow_id

    # Load storyboard JSON and filter out unwanted keys to build prompt
    with open(STORYBOARD_JSON_PATH, 'r', encoding='utf-8') as f:
        storyboard_data = json.load(f)
    filtered_storyboard = {k: v for k, v in storyboard_data.items() if k not in STORYBOARD_EXCLUDE_KEYS}
    storyboard_prompt = json.dumps(filtered_storyboard, ensure_ascii=False, indent=2)

    # input
    # task_input = json.dumps({
    #     "binary_data": [
    #         {
    #             "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/dd.png",
    #             "type": "image"
    #         }
    #     ],
    #     "req_json": {
    #         "prompt": storyboard_prompt,
    #         "language": "zh",
    #         "duration": 10,
    #         "seed": 42,
    #         "aspect_ratio": "9:16",
    #         "resolution": "480p",
    #         "binary_var_name": ["image"],
    #         "workflow": "seedance_2_0_pe_integration.json"
    #     }
    # })
    task_input = json.dumps({
        "binary_data": [],
        "req_json": {
            "prompt": storyboard_prompt,
            "language": "zh",
            "duration": 10,
            "seed": 42,
            "aspect_ratio": "16:9",
            "resolution": "576p",
            "binary_var_name": [],
            "workflow": "seedance_2_0_pe_integration.json"
        }
    })

    # submit a task
    submit_req = SubmitAsyncTaskRequest()
    submit_req.task = Task(
        input=task_input,
        priority=7,
        tags={
            "second_biz_name": "test"  # <BA空间名>" #业务场景唯一标识字段，和BA空间名一致，用来标识流量来源
        }
    )
    submit_req.workflow_id = workflow_id
    print(f"WorkflowID: {submit_req.workflow_id}")

    submit_resp = cairo_client.SubmitAsyncTask(submit_req)
    task_id = submit_resp.task_id
    print(f"Submitted Task ID: {task_id}")
    print(f"submit_resp: {submit_resp}")


    # poll to get the result
    def get_task_report(task_id):
        get_req = GetTaskReportRequestThrift(task_id=task_id)
        get_resp = cairo_client.GetTaskReport(get_req)
        task = json.loads(get_resp.task)
        report = json.loads(get_resp.report)
        return task, report


    start_time = time.time()
    while True:
        try:
            task, report = get_task_report(task_id)
            status = task["status"]

            print(f"Polling Task {task_id} -> {status}")
            if status == "succeeded":
                # download video
                results = json.loads(task["output"])["results"]
                print(results)
                key = list(results.keys())[0]
                storage = results[key]['Extra']["storage"]
                print(f"result vid: {key}, with storage info: {storage}")

                # if you did not pass-in Vod config, get video from tos; please access SG tos url in SG instances
                # url = f"https://tosv.byted.org/obj/iccv-vpipe-sg/{key}" #传到默认tos空间
                # print(f"Downloading from {url}")
                # subprocess.run(["wget", url, "-O", "./output.mp4"], check=True)

                # or download in office net
                url = f"https://tosv-sg.tiktok-row.org/obj/iccv-vpipe-sg/{key}"
                print(f"Video link is {url}")
                break
            elif status in ("failed", "cancelled"):
                raise RuntimeError(f"Task {task_id} failed with status {status}")
        except Exception as e:
            print(f"Error polling task {task_id}: {e}")
        time.sleep(5)
    end_time = time.time()
    print(f"The time used for generation is: ")
    print(end_time - start_time)

    # Download the generated video (errors here won't affect the main flow)
    try:
        import urllib.request
        import os
        output_filename = f"Sample_results/output_{task_id}.mp4"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
        print(f"Downloading video from {url} ...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Video saved to {output_path}")
    except Exception as download_err:
        print(f"Failed to download video: {download_err}")