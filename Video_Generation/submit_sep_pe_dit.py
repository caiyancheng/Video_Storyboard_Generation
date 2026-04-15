import time
import json
import subprocess
import euler
from euler import base_compat_middleware

from cairo_v2.idls.thrift import GetTaskReportRequestThrift
from cairo_v2.idls import CairoService, SubmitAsyncTaskRequest, Task

import os
from io import BytesIO
from PIL import Image
# from utils import upload_to_imagex, download_video

input_dir = "/mnt/bn/ic-aip/zgjj/workspace/i2v_batch/0211_seedance/downloaded_images"

if __name__ == "__main__":

    id = 2
    local_path = "/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample/first_frame/id_0000_score4_v09044be0000bib8k1rdjls2dbu62kug_first.png"
    media_type = "image"
    prompt = '''Scene overview: 在倒计时声中，<Subject_1 (D.Va)>驾驶她的<Prop_1 (D.Va的粉色机甲)>从<Location_1 (机甲发射舱)>发射，高速穿过<Location_2 (未来主义隧道)>，最终进入开阔的<Location_3 (云层密布的天空)>，标志着任务的开始。'''
    import requests

    url = "https://api2.musical.ly/media/api/pic/afr"


    # ── 方案A：不传图片，去掉 binary_var_name（T2V 模式）─────────────────────
    print("\n=== 尝试不传图片（方案A: 去掉 binary_var_name）===")
    payload_no_img_a = {
        'algorithms': 'tt_seedance2_pe_test',
        'conf': json.dumps({
            "prompt": prompt,
            "duration": 10,
            "aspect_ratio": "9:16",
            "language": "zh",
            "with_audio": True,
            "version": "v2.0",
            "task_type": "T2V",   # 改为 T2V（纯文本转视频）
        }),
    }
    resp_a = requests.post(url, data=payload_no_img_a, timeout=180)
    print(f"方案A status={resp_a.status_code}  body={resp_a.text[:300]}")

    # ── 方案B：不传图片，但保留 binary_var_name 为空列表 ─────────────────────
    print("\n=== 尝试不传图片（方案B: binary_var_name=[]）===")
    payload_no_img_b = {
        'algorithms': 'tt_seedance2_pe_test',
        'conf': json.dumps({
            "prompt": prompt,
            "duration": 10,
            "aspect_ratio": "9:16",
            "language": "zh",
            "with_audio": True,
            "version": "v2.0",
            "binary_var_name": [],
            "task_type": "R2V",
        }),
    }
    resp_b = requests.post(url, data=payload_no_img_b, timeout=180)
    print(f"方案B status={resp_b.status_code}  body={resp_b.text[:300]}")

    payload = {
        'algorithms': 'tt_seedance2_pe_test',
        'conf': json.dumps({
            "prompt": prompt,
            "duration": 10,
            "aspect_ratio": "9:16",
            "language": "zh",
            "with_audio": True,
            "version": "v2.0",
            "binary_var_name": [f"{media_type}"],
            "task_type": "R2V"
        }),
        'input_img_type': 'multiple_files'
    }
    files = [
        ('files[]', ('file', open(local_path, 'rb'), 'application/octet-stream'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
    pe_result = json.loads(response.text)
    pe_prompt = json.loads(pe_result["data"]["afr_data"][0]["pic_conf"])
    print(f"pe_prompt: {pe_prompt}")  # pe输出的中间结果

    image = Image.open(local_path)
    # 将图像保存为二进制数据
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    assert len(img_byte_arr) > 0, "image is empty"
    imagex_uri = upload_to_imagex(img_byte_arr)
    print(f"[image-saver] upload image to imagex, uri is {imagex_uri}")
    url = f"http://p-tt-aigc-avatar-va.byteintl.net/{imagex_uri}~tplv-51pdn7bo8b-image.image"

    cairo_client = euler.Client(CairoService, target="sd://aip.tce.cairo_v2?cluster=default&idc=maliva",
                                transport="ttheader")  # 上线请切换到aip.tce.cairo_v2
    cairo_client.use(base_compat_middleware.client_middleware)

    workflow_id = "wf_1pESDxHZRPmaHzWaRhs5MM"  # 新业务场景接入请联系 @george.zhang01 提供新的workflow_id

    task_input = json.dumps(
        {
            "pre_processing": {
                "binary_data": [
                    {
                        "data": url,
                        "type": media_type,
                    }
                ],
                "meta": {},
                "req_json": {
                    "task": "r2v",
                    "prompt": pe_prompt,
                    "file_types": [media_type],
                },
                "output_data": [{"storage": {"platform": "Tos", "idc": "sg1"}}],
            },
            "end_node": "main_dit",
        }
    )

    json.loads(task_input)

    # submit a task
    submit_req = SubmitAsyncTaskRequest()
    submit_req.task = Task(
        input=task_input,
        priority=7,
        tags={
            "second_biz_name": "<BA空间名>"  # 业务场景唯一标识字段，和BA空间名一致，用来标识流量来源
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


    while True:
        try:
            task, report = get_task_report(task_id)
            status = task["status"]

            print(f"Polling Task {task_id} -> {status}")
            if status == "succeeded":
                # download video
                results = json.loads(task["output"])["results"]
                key = list(results.keys())[0]
                storage = results[key]['Extra']["storage"]
                print(f"result vid: {key}, with storage info: {storage}")

                # if did not pass-in Vod config, get video from tos; please access SG tos url in SG instances
                url = f"https://tosv.byted.org/obj/iccv-vpipe-sg/{key}"  # 传到默认tos空间
                print(f"Downloading from {url}")
                subprocess.run(["wget", url, "-O", "./output.mp4"], check=True)

                # or download in office net
                # url = f"https://tosv-sg.tiktok-row.org/obj/iccv-vpipe-sg/{key}"
                break
            elif status in ("failed", "cancelled"):
                print(f"Task {task_id} failed with status {status}")
                print(task)
                break
        except Exception as e:
            print(f"Error polling task {task_id}: {e}")
        time.sleep(5)
