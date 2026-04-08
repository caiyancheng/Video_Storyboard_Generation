import argparse
import os
import time
import json
import cv2
from pathlib import Path
import torch

os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp/hf_cache"

from modelscope import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from vllm import LLM, SamplingParams
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ===============================
# 获取真实视频长度
# ===============================
def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration


# ===============================
# 参数
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate a stitched storyline from real clips + AIGC segments.")
    parser.add_argument(
        '--video_path_dir',
        type=str,
        default="/root/autodl-tmp/Datasets/China/Xi_an_360p_clips/",
        help="Directory containing input MP4 clips"
    )
    parser.add_argument(
        '--max_aigc_clip_time',
        type=int,
        default=15,
        help="Max duration for each AIGC segment in seconds"
    )
    return parser.parse_args()


# ===============================
# 主函数
# ===============================
def main():
    args = parse_args()
    VIDEO_DIR = Path(args.video_path_dir)
    video_dir_base_name = os.path.basename(VIDEO_DIR)
    MAX_AIGC_CLIP_TIME = args.max_aigc_clip_time

    # ===============================
    # Step 1: Collect real clips
    # ===============================
    clip_paths = sorted([str(p) for p in VIDEO_DIR.glob("*.mp4")])
    if not clip_paths:
        raise ValueError(f"No MP4 files found in {VIDEO_DIR}")

    real_clips = []
    for path in clip_paths:
        dur = get_video_duration(path)
        if dur > 0:
            real_clips.append({"path": path, "duration": dur})

    print(f"Found {len(real_clips)} real clips.")

    # ===============================
    # Step 2: 构建 Prompt（完整保留你的结构）
    # ===============================
    MODEL_PATH = "/root/autodl-tmp/hf_cache/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"

    prompt_text = f"""
You are a senior animation screenwriter. Your task is to create a coherent, cinematic storyline.

You are given a set of REAL video clips (with known durations). You must INTERLEAVE these real clips with AI-generated (AIGC) narrative segments to form a continuous story. 

Each REAL clip must appear EXACTLY ONCE, in an order you choose (you may reorder them for better storytelling). NOT ordered by numerical index.

### Input Clips Summary:
{json.dumps([
    {"clip_index": i, "duration_sec": round(clip['duration'], 2), "filename": os.path.basename(clip['path'])}
    for i, clip in enumerate(real_clips)
], indent=2)}

### Rules:
1. Between/around real clips, insert AIGC segments. Each AIGC segment can be up to {MAX_AIGC_CLIP_TIME} seconds long, the length should depends on the c0ontent, don't fix.
2. Each REAL clip must appear EXACTLY ONCE, in an order you choose (you may reorder them for better storytelling). NOT ordered by numerical index.
3. Output a JSON with:
   - "Characters": up to 3
   - "Environments": up to 12
   - "Scripts": list of segments in chronological order, up to 20, each with:
        - "type": "real" or "aigc"
        - "clip_index": (only for real)
        - "length": (only for aigc) length in seconds, <= {MAX_AIGC_CLIP_TIME}
        - "title", "key_frame_description", "video_description", "camera_motion", "light_condition"

### Output Schema (strict JSON):
{{
  "Characters": {{
        "Character_1": {{
          "title": "Short character title",
          "character_description": "Concise 1-2 sentences on visual identity.",
          "character_image_time": {{'clip_index': <clip_index>, 'time': <float_seconds>}}, # 'time' must < real clip length
        }}
      }},
  "Environments": {{
        "Environment_1": {{
          "title": "Short environment title",
          "environment_description": "Concise 1-2 sentences on cinematic setting.",
          "environment_image_time": {{'clip_index': <clip_index>, 'time': <float_seconds>}}, # 'time' must < real clip length
        }}
      }},
  "Scripts": [
    {{
      "type": "real",
      "clip_index": <clip_index>,
      "title": "Short scene title",
      "key_frame_description": "Present-tense narrative using defined Character_* and Environment_*.",
      "video_description": "Present-tense narrative using defined Character_* and Environment_*.",
      "camera_motion": "e.g., Static Shot / Tracking Shot / Dolly In / Dolly Out / Pan / Tilt / Crane Shot / Steadicam Shot / Orbit Shot / Handheld",
      "light_condition": "e.g., High-key lighting / Low-key lighting / Chiaroscuro / Backlighting / Rim lighting",
    }},
    {{
      "type": "aigc",
      "length": <float_seconds>,
      "title": "...",
      "key_frame_description": "...",
      "video_description": "...",
      "camera_motion": "...",
      "light_condition": "...",
    }},
    {{
      "type": "real",
      "clip_index": <clip_index>,
      "title": "...",
      "key_frame_description": "...",
      "video_description": "...",
      "camera_motion": "...",
      "light_condition": "...",
    }}
  ]
}}

### Structural Enforcement:
- You MUST interleave REAL and AIGC segments.
- Maximum of TWO consecutive AIGC segments allowed.
- Total number of AIGC segments within ±2 of real clips.
- NEVER repeat consecutive AIGC concepts.
"""

    print("Loading Qwen3-Omni model...")
    start_time = time.time()

    model = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.75,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={'image': 1, 'video': len(real_clips), 'audio': len(real_clips)},
        max_num_seqs=1,
        max_model_len=65536,
        seed=1234,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    conversation = [{
        "role": "user",
        "content": [
            *[{"type": "video", "video": clip["path"]} for clip in real_clips],
            {"type": "text", "text": prompt_text}
        ],
    }]
    USE_AUDIO_IN_VIDEO = True
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.8,
        top_k=40,
        max_tokens=12288
    )

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    inputs = {'prompt': text, 'multi_modal_data': {}, "mm_processor_kwargs": {"use_audio_in_video": USE_AUDIO_IN_VIDEO}}
    if images is not None: inputs['multi_modal_data']['image'] = images
    if videos is not None: inputs['multi_modal_data']['video'] = videos
    if audios is not None: inputs['multi_modal_data']['audio'] = audios

    outputs = model.generate(inputs, sampling_params=sampling_params)
    response = outputs[0].outputs[0].text

    os.makedirs("generate_script", exist_ok=True)
    with open("generate_script/raw_output.json", "w", encoding="utf-8") as f:
        f.write(response)

    # ===============================
    # 解析 JSON（不再使用 fix_duration_array）
    # ===============================
    start_idx = response.find('{')
    end_idx = response.rfind('}') + 1
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No JSON object found")

    json_str = response[start_idx:end_idx]
    result = json.loads(json_str)

    scripts = result.get("Scripts", [])
    if not scripts:
        raise ValueError("Scripts list is empty")

    # ===============================
    # 构建连续 duration
    # ===============================
    current_time = 0.0
    used_real_indices = set()

    for i, seg in enumerate(scripts):

        seg_type = seg.get("type")

        if seg_type == "real":
            clip_idx = seg.get("clip_index")

            if clip_idx is None:
                raise ValueError(f"Segment {i}: real missing clip_index")

            if clip_idx in used_real_indices:
                raise ValueError(f"Real clip {clip_idx} used twice")

            if clip_idx >= len(real_clips):
                raise ValueError(f"Invalid clip_index {clip_idx}")

            used_real_indices.add(clip_idx)
            duration_sec = real_clips[clip_idx]["duration"]

        elif seg_type == "aigc":
            length = float(seg.get("length", 3.0))
            duration_sec = max(0.5, min(MAX_AIGC_CLIP_TIME, length))

        else:
            raise ValueError(f"Invalid segment type {seg_type}")

        start = round(current_time, 2)
        end = round(current_time + duration_sec, 2)

        seg["duration"] = [start, end]
        current_time = end

    # if len(used_real_indices) != len(real_clips):
    #     missing = set(range(len(real_clips))) - used_real_indices
    #     raise ValueError(f"Not all real clips used. Missing: {missing}")

    result["RealClips"] = {
        str(i): real_clips[i]["path"]
        for i in range(len(real_clips))
    }

    output_path = f"generate_script/stitched_storyline_{video_dir_base_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully generated stitched storyline: {output_path}")
    print(f"Total duration: {round(current_time, 2)} seconds")
    print(f"Total execution time: {round(time.time() - start_time, 2)} seconds")


if __name__ == '__main__':
    main()
