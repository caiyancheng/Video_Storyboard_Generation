import argparse
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp/hf_cache"
from modelscope import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import warnings
import json
import cv2
from vllm import LLM
from vllm import SamplingParams
import torch
import gc
import sys

def suppress_ffmpeg():
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

suppress_ffmpeg()
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

# 设置命令行参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="Generate screenplay from video and audio.")
    parser.add_argument('--video_path', type=str, default=r"/root/autodl-tmp/Datasets/China/Xi_an_360p.mp4",
                        help="Path to the input video file")
    return parser.parse_args()

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    VIDEO_PATH = args.video_path
    VIDEO_DURATION = get_video_duration(VIDEO_PATH)

    MODEL_PATH = "/root/autodl-tmp/hf_cache/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"

    # 经过优化的 Prompt：保留严格数量限制，采用纯时间戳定位
    # 经过深度优化的 Prompt：强化时间轴连续性与逻辑闭环
    prompt_text = f"""
    You are a senior animation screenwriter and visual analyst. Analyze the video and audio (Total Duration: {VIDEO_DURATION:.1f}s). 
    Generate a professional, production-ready screenplay in strict JSON format.

    ### Output Schema:
    All timestamps must be normalized to the range [0.0, 1.0], where 0.0 corresponds to the start of the video and 1.0 to the end.
    Normalization is computed as: normalized_time = actual_time_in_seconds / {VIDEO_DURATION:.1f}

    {{
      "Characters": {{
        "Character_1": {{
          "title": "Short character title",
          "character_description": "Concise 1-2 sentences on visual identity.",
          "character_image_time": 0.25
        }}
      }},
      "Environments": {{
        "Environment_1": {{
          "title": "Short environment title",
          "environment_description": "Concise 1-2 sentences on cinematic setting.",
          "environment_image_time": 0.1
        }}
      }},
      "Scripts": [
        {{
          "title": "Short scene title",
          "duration": [0.0, 0.3],
          "key_frame_time": 0.15,
          "key_frame_description": "Present-tense narrative using defined Character_* and Environment_*.",
          "video_description": "Present-tense narrative using defined Character_* and Environment_*.",
          "camera_motion": "e.g., Static Shot / Tracking Shot / Dolly In / Dolly Out / Pan / Tilt / Crane Shot / Steadicam Shot / Orbit Shot / Handheld",
          "light_condition": "e.g., High-key lighting / Low-key lighting / Chiaroscuro / Backlighting / Rim lighting / Rembrandt lighting / Split lighting / Butterfly lighting / Soft diffused lighting / Hard lighting / Golden hour lighting / Blue hour lighting / Neon lighting / Practical lighting / Volumetric lighting"
        }}
      ]
    }}

    ### Strict Constraints:

    1. Chronological Continuity (Critical):
       - No gaps and no overlaps: The scripts must form a continuous timeline from 0.0 to 1.0.
       - Seamless Connection: For any script segment i, its duration[1] MUST be exactly equal to the duration[0] of segment i+1.
       - Full Coverage: The first script must start at 0.0, and the last script must end at 1.0.
       - Segment boundaries must correspond to meaningful visual transitions.

    2. Quantity Limits:
       - Characters: Maximum 3. Use an empty object {{}} if none exist.
       - Environments: Maximum 8.
       - Scripts: Maximum 10 segments. Don't only focus on the start or end, equally distributed. Every Environment should exist in at least one segment.

    3. Timestamp Validity:
       - All timestamps (character_image, environment_image, key_frame, and all values in duration) MUST be floats in [0.0, 1.0].
       - Do not guess timestamps. Compute them based on actual frame positions:
            FPS = total_frame_count / {VIDEO_DURATION:.1f}
            timestamp_seconds = frame_number / FPS
            normalized_time = timestamp_seconds / {VIDEO_DURATION:.1f}
       - Only use frames where the character or environment is clearly and fully visible.

    4. JSON Format Requirements:
       - "Scripts" must be an array of objects. Do not include keys like "Script_1" inside the array.
       - Output only valid JSON. Do not include explanations, markdown, or extra text before or after the JSON object.
    """

    print("Loading model and processor...")
    # 开始计时
    start_time = time.time()

    # 使用vllm加载模型
    model = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.75,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={'image': 1, 'video': 3, 'audio': 3},
        max_num_seqs=1,
        max_model_len=65536, # 32768
        seed=1234,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": VIDEO_PATH},
                {"type": "text", "text": prompt_text}
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Inference: Generation of the output text and audio
    sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = {'prompt': text, 'multi_modal_data': {}, "mm_processor_kwargs": {"use_audio_in_video": USE_AUDIO_IN_VIDEO}}
    if images is not None: inputs['multi_modal_data']['image'] = images
    if videos is not None: inputs['multi_modal_data']['video'] = videos
    if audios is not None: inputs['multi_modal_data']['audio'] = audios
    outputs = model.generate(inputs, sampling_params=sampling_params)
    response = outputs[0].outputs[0].text

    # 尝试解析 JSON（可能需要清理）
    video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    os.makedirs(f"generate_script", exist_ok=True)
    try:
        # 清理可能的非 JSON 前缀/后缀
        start = response.find('{')
        end = response.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        json_str = response[start:end]
        script_json = json.loads(json_str)
        print("✅ Successfully parsed JSON screenplay.")

        # 保存
        output_path = f"generate_script/structured_screenplay_{video_basename}_vllm.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(script_json, f, indent=2, ensure_ascii=False)
        print(f"📄 Saved to {output_path}")

    except Exception as e:
        print("❌ Failed to parse JSON. Raw output saved for debugging.")
        with open(f"generate_script/raw_output_{video_basename}_vllm.txt", 'w', encoding='utf-8') as f:
            f.write(response)
        raise e

    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Video Length: {VIDEO_DURATION:.2f} seconds")
    print(f"Total execution time: {elapsed_time:.2f} seconds")


    with open(f"generate_script/execution_time_{video_basename}_vllm.txt", 'w', encoding='utf-8') as f:
        f.write(f"Total Video Length: {VIDEO_DURATION:.2f} seconds\n")
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
    print("📄 Execution time saved to execution_time.txt")

if __name__ == '__main__':
    main()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
