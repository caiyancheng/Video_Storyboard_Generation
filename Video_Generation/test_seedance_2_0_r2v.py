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
    
    workflow_id = "seedance_2_0_r2v_e2e_with_pe_test_inference_only_v2_mirror" #新业务场景接入请联系 @george.zhang01 提供新的workflow_id

    # input
    task_input=json.dumps({
        "binary_data": [
            # {
            #     "data": "https://tosv.byted.org/obj/iccv-vpipe-sg/645fe4f04ce50415da6acc297a479caa.mp4",
            #     "type": "video"
            # },
            # {
            #     "data": "https://tosv.byted.org/obj/iccv-vpipe-sg/3fe2668bf0225d18c7ddca30d1e7c01f.png",
            #     "type": "image"
            # },
            # {
            #     "data": "https://tosv.byted.org/obj/iccv-vpipe-sg/cdd9105e1d6e4defe38c06d1fd553fe1.png",
            #     "type": "image"
            # }
            # {
            #     "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/dd.png",
            #     "type": "image"
            # },
            # {
            #     "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/yy.png",
            #     "type": "image"
            # },
            # {
            #     "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/lx.png",
            #     "type": "image"
            # }
            # {
            #     "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/yc.png",
            #     "type": "image"
            # }
            # {
            #     "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/kf.mp4",
            #     "type": "video"
            # },
            {
                "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/tj.png",
                "type": "image"
            },
            {
                "data": "https://tosv.byted.org/obj/dm-stickers-rec-sg/yj.png",
                "type": "image"
            }
        ],
        "req_json": {
            # "prompt": "给定三张图片中的人物，生成一个约30秒的写实风格短视频剧情：图1、图2、图3中的三个人物在会议室围坐开会讨论一个重要方案，一开始三人意见明显不同，图1态度谨慎、提出风险与问题，图3持反对意见且情绪略显激动，而图2保持冷静自信、认真倾听；镜头先以三人围坐的全景开场，再切换人物特写，通过皱眉、摇头、摊手等肢体语言强化分歧氛围；随后图2起身，有条理地分析问题并配合手势进行解释，随着陈述推进，图1逐渐从质疑转为思考并点头认可，图3也从激动转为沉默思索，最终被说服表示赞同；结尾三人达成一致，气氛转为积极，镜头以中景呈现三人点头或握手，体现合作与共识的达成。",
            # "prompt": "复刻视频的创意广告，手机用图1、女主用图2，开头的场景变成现代化的城市街头，女主在马路中央",
            # "prompt": "复刻视频的创意，后空翻的男主用图1的男生，另一个男主用图2的女生，开头的场景变成现代化的城市街头，所有人在马路中央，要保留原始的声音和对话",
            "prompt": 
            # """
            # 两张图连续叙事：男主，图1，神情郑重地对图2女主提出近乎苛刻的要求，语气认真又略带夸张地说道：“要十担白雪，团作雪人，不要见半点泥沙在上面；再要十担坚冰，凿作冰人，不要见些雪屑在上面，也要打磨光亮；再要十担细沙，塑作沙人，也要细细地拍实，不要见些雪冰在上面。” 女主先是愣住，露出为难又不服气的神情，周围堆满雪、冰块和细沙的材料；呈现她开始认真动手制作，画面富有戏剧张力和节奏感，整体风格略带幽默夸张，人物表情生动，电影感构图。
            # """,
            """
            图1中的人物自然走向图2中的人物，围成半圆。大家微笑鼓掌，有人轻拍肩膀表示祝贺，有人递上花束。新人略带害羞地微笑回应。整体氛围温暖、真实、有生活感，像公司同事为新人庆祝婚礼并表达感谢。
            """,
            # "prompt": """
            # 以输入图片为首帧，生成一个具有戏剧张力和轻微幽默感的短视频。画面中，一个穿军装的人（代表“YIMING”）张开双臂站立，正在为床上熟睡的人（代表“AIGT”）抵挡从空中落下的刀、手雷和各种攻击物。

            # 视频中加入动态效果：

            # 天空中不断有刀具、手雷等物体从上方高速坠落

            # “YIMING”保持保护姿态，身体略微晃动，被击中后有轻微火花或烟雾效果，但依然坚持站立

            # 床上的“AIGT”始终安静熟睡，偶尔轻微翻身或呼吸起伏

            # 画面节奏紧张但带有漫画风格的夸张表现

            # 风格要求：

            # 漫画 / meme 风格动画

            # 轻微夸张的物理效果（火花、冲击波、烟雾）

            # 3–5 秒循环视频

            # 保持原图人物构图和文字“YIMING”“AIGT”不变

            # 镜头轻微抖动和慢慢推近（cinematic zoom in）
            # """,
            # "prompt": "以图1为基础，卡通动画风格的场景，一个可爱的卡通头像人物在舞台中央自信地进行钢管舞表演。舞台灯光绚丽多彩，聚光灯打在人物图2身上，背景充满动感的灯效和节奏感。角色动作流畅有力，带有夸张的卡通表现力和节奏感。舞台下围满了观众，大家神情兴奋、专注观看，有人图3拿手机拍摄，有人随着节奏摆动身体，整体气氛热烈欢快。人群中有一个名叫 Yiming 的观众， 参考图2，正面带笑容用力鼓掌，为表演喝彩。镜头在舞台表演和观众反应之间切换，突出现场热闹、欢快、娱乐性的氛围。整体画面为2D卡通插画风格，色彩鲜艳，线条清晰，具有动画片质感和轻松愉快的节奏感。",
            "binary_var_name": ["image", "image"],
            "duration": 15,
            "workflow": "seedance_2_0_pe_integration_r2v.json"
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