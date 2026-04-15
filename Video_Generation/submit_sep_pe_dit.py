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
from utils import upload_to_imagex, download_video

input_dir = "/mnt/bn/ic-aip/zgjj/workspace/i2v_batch/0211_seedance/downloaded_images"

if __name__ == "__main__":

    id = 2
    local_path = "/mnt/bn/ic-aip/zgjj/workspace/i2v_batch/0211_seedance/images/20260216-230953.png"
    media_type = "image"
    prompt = '''生成一段视频，视频的主角是画面中的年轻亚裔女性，她的形象以图片中那位佩戴黑框眼镜、留着金色齐刘海发型的女性为基础，但进行了以下调整：将她的服装修改为一件浅灰色 Stussy 品牌帽衫，黑色耳机挂在脖子上，头发保持蓬松。视频的场景设定在一个装饰有雪花和彩球的机场大厅。视频开始时，她面带笑容，以近景视角直视镜头。随后，镜头跟随她的脚步，转换为一个中景后视镜头，展示她行走的背影。机场大厅人来人往，可以清晰看到指示牌和办理登机的人群。地面铺设着蓝灰色地毯，天花板上悬挂着节日装饰，整体光线充足。场景快速切换，来到一个阳光明媚的海边。两位穿着连衣裙的年轻女性（其中一位是参考图中的人物）手牵着手，站在海边的岩石上，远处是湛蓝的海水和雄伟的山脉，天空晴朗。海浪拍打着岸边，可以感受到轻微的海风。场景再次切换，两位女性（其中一位是参考图中的人物）站在一处岩石滩上，周围有企鹅。她们面带笑容，手牵着手，站在水中。水面清澈，可以反射出天空的颜色。场景继续切换，两位女性（其中一位是参考图中的人物）站在雾气缭绕的山顶公路上，手牵着手，彼此相望。路面湿滑，可以感受到空气中的湿度。场景又一次切换，夕阳西下，两位女性（其中一位是参考图中的人物）站在一个可以俯瞰城市和海湾的观景台上，手牵着手，欣赏着美丽的景色。最终，场景切换到一条街道上，两位女性（其中一位是参考图中的人物）与一群身着传统服饰的表演者互动。表演者们化着黑白相间的油彩，穿着民族服装，正在进行表演。女性们面带笑容，与表演者们握手。在最后一个场景中，两位女性（其中一位是参考图中的人物）坐在一个开放式的车辆里，对着镜头开怀大笑。她们看起来非常开心，背景是模糊的户外景色。整个视频以快节奏的剪辑和场景切换为特点，展现了两位女性在不同地点旅行和互动的片段。背景音乐节奏轻快，与画面的切换节奏相呼应，营造出一种轻松愉快的氛围。'''

    import requests

    url = "https://api2.musical.ly/media/api/pic/afr"
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
