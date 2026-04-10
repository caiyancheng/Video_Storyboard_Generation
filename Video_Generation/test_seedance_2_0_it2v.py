import time
import json
import subprocess
import euler
from euler import base_compat_middleware

# bytedeuler==2.4.3
# byted_cairo==2.6.6
from cairo_v2.idls.thrift import GetTaskReportRequestThrift
from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task

if __name__ == "__main__":

    cairo_client = euler.Client(CairoService, target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva", transport="ttheader") #上线请切换到aip.tce.cairo_v2
    cairo_client.use(base_compat_middleware.client_middleware)
    
    workflow_id = "seedance_2_0_ti2v_e2e_with_pe_test_inference_only_v2" #新业务场景接入请联系 @george.zhang01 提供新的workflow_id

    # input
    task_input=json.dumps({
        "binary_data": [
            {
                "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/yc.jpeg",
                "type": "image"
            }
        ],
        "req_json": {
            # "prompt": """
            # 两张图连续叙事：男主神情郑重地提出近乎苛刻的要求，语气认真又略带夸张地说道：“要十担白雪，团作雪人，不要见半点泥沙在上面；再要十担坚冰，凿作冰人，不要见些雪屑在上面，也要打磨光亮；再要十担细沙，塑作沙人，也要细细地拍实，不要见些雪冰在上面。” 画面富有戏剧张力和节奏感，整体风格略带幽默夸张，人物表情生动，电影感构图。
            #         """,
            "prompt": """
            男主在海边自拍然后开始尝试滑翔伞，转场后潜水，人物表情生动，电影感构图。
                    """,
            "language": "zh", 
            "duration": 3,
            "seed": 42,
            "aspect_ratio": "9:16",
            "resolution": "480p", 
            "binary_var_name": ["image"], 
            "workflow": "seedance_2_0_pe_integration.json"
        }
    })

    # submit a task
    submit_req = SubmitAsyncTaskRequest()
    submit_req.task = Task(
        input=task_input,
        priority=7,
        tags={
                "second_biz_name": "test" #<BA空间名>" #业务场景唯一标识字段，和BA空间名一致，用来标识流量来源
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