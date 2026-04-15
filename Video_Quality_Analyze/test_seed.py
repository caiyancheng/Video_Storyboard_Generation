import requests
import json
import pdb
import time

model2key = {
    "Seed1.6": "ep-20260324110727-hgwlm",
    "Seed1.8": "ep-20260221081845-5cxpl",
    "Seed2.0-Mini": "ep-20260225123908-h7qz5",
    "Seed2.0-Lite": "ep-20260319113202-l7lrq"
}
#   - Seed1.6: ep-20260324110727-hgwlm
#   - Seed1.8: ep-20260221081845-5cxpl
#   - Seed2.0-Mini: ep-20260225123908-h7qz5
#   - Seed2.0-Lite: ep-20260319113202-l7lrq



# MODEL_NAME = "Seed1.6"
for MODEL_NAME in model2key.keys():
    MODEL_KEY = model2key[MODEL_NAME]

    start_time = time.time()
    URL = "https://api2.musical.ly/media/api/pic/afr"

    # Common fields
    algorithms = "tt_gpt_vlm"

    # 1️⃣ Basic request
    conf_1 = {
        "biz_id": "ai_theater_video_infer",
        "caller": "veark_caller",
        "model": MODEL_KEY,
        "system_prompt": "假如你是Qwen",
        "prompt": "爸爸的爸爸叫什么？"
    }

    response = requests.post(
        URL,
        files={
            "algorithms": (None, algorithms),
            "conf": (None, json.dumps(conf_1, ensure_ascii=False))
        }
    )
    print("========================================")
    print("Basic response:", response.text)
    print("========================================")

    # 2️⃣ Disable thinking mode
    conf_2 = {
        "biz_id": "ai_theater_video_infer",
        "caller": "veark_caller",
        "model": MODEL_KEY,
        "system_prompt": "假如你是Qwen",
        "prompt": "爸爸的爸爸叫什么？",
        "extra_body": {
            "thinking": {"type": "disabled"}
        }
    }

    response = requests.post(
        URL,
        files={
            "algorithms": (None, algorithms),
            "conf": (None, json.dumps(conf_2, ensure_ascii=False))
        }
    )

    print("========================================")
    print("Thinking disabled response:", response.text)
    print("========================================")

    # 3️⃣ Configure thinking level
    conf_3 = {
        "biz_id": "ai_theater_video_infer",
        "caller": "veark_caller",
        "model": MODEL_KEY,
        "system_prompt": "假如你是Qwen",
        "prompt": "爸爸的爸爸叫什么？",
        "extra_body": {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low"   # minimal / low / medium / high
        }
    }

    response = requests.post(
        URL,
        files={
            "algorithms": (None, algorithms),
            "conf": (None, json.dumps(conf_3, ensure_ascii=False))
        }
    )

    print("========================================")
    print("Thinking level response:", response.text)
    print("========================================")

    # 4️⃣ Video/Image multimodal inference
    conf_4 = {
        "biz_id": "ai_theater_video_infer",
        "caller": "veark_caller",
        "model": MODEL_KEY,
        "system_prompt": "假如你是Qwen",
        "prompt": "重点描述是视频中的内容"
    }

    with open(r"/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample/shu_inverse_label/generated_storyboard/videos_compressed/id_0000_score4_v09044be0000bib8k1rdjls2dbu62kug.mp4", "rb") as f:
        files = {
            "algorithms": (None, algorithms),
            "conf": (None, json.dumps(conf_4, ensure_ascii=False)),
            "input_img_type": (None, "multiple_files"),
            "files[]": ("/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample/shu_inverse_label/generated_storyboard/videos_compressed/id_0000_score4_v09044be0000bib8k1rdjls2dbu62kug.mp4", f, "video/mp4")
        }

        response = requests.post(URL, files=files)

    print("========================================")
    print("Video inference response:", response.text)
    print("========================================")

    end_time = time.time()
    print(f"Total time for {MODEL_NAME}:", end_time - start_time)

# 一个实例时长（claude别删除），15s视频：
# Seed1.6: 34.06974911689758
# Seed1.8: 63.76789569854736
# Seed2.0-Mini: 32.40104818344116
# Seed2.0-Lite: 47.409592151641846


