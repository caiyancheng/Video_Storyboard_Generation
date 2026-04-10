"""
extract_shot_first_frames.py

For each shot in sample_storyboard_phase0_v15_result.json, extract the first
frame from the original video (video_url field) at the shot's start timestamp
and save it as a JPEG in Sample_results/shot_first_frames/.

Requirements: ffmpeg must be available on PATH.
"""

import json
import os
import subprocess
import sys

STORYBOARD_PHASE0_PATH = "Sample_Prompts/sample_storyboard_phase0_v15_result.json"
OUTPUT_DIR = "Sample_results/shot_first_frames"


def parse_start_seconds(time_range: str) -> float:
    """Parse 'MM:SS.mmm-MM:SS.mmm' and return start time in seconds."""
    start_str = time_range.split('-')[0]
    mm, ss = start_str.split(':')
    return int(mm) * 60 + float(ss)


def seconds_to_ffmpeg_time(secs: float) -> str:
    """Convert float seconds to ffmpeg-compatible time string 'HH:MM:SS.mmm'."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_frame(video_url: str, timestamp: str, output_path: str) -> bool:
    """
    Use ffmpeg to extract a single frame at the given timestamp from video_url.
    Returns True on success, False on failure.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", timestamp,       # seek before opening input for speed
        "-i", video_url,
        "-vframes", "1",        # extract exactly 1 frame
        "-q:v", "2",            # JPEG quality (2 = high quality)
        output_path
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return True
        else:
            print(f"    ffmpeg stderr: {result.stderr[-600:]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    ffmpeg timed out for {output_path}")
        return False
    except FileNotFoundError:
        print("    ERROR: ffmpeg not found. Please install ffmpeg and add it to PATH.")
        sys.exit(1)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(STORYBOARD_PHASE0_PATH, 'r', encoding='utf-8') as f:
        phase0 = json.load(f)

    video_url = phase0['video_url']
    shots = phase0['shot_registry']

    print(f"Source video : {video_url}")
    print(f"Total shots  : {len(shots)}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print()

    success_count = 0
    fail_count = 0

    for shot in shots:
        shot_id = shot['id']                   # e.g. "<Shot_1>"
        shot_id_safe = shot_id.strip('<>')     # e.g. "Shot_1"
        time_range = shot['time_range']
        start_sec = parse_start_seconds(time_range)
        ffmpeg_time = seconds_to_ffmpeg_time(start_sec)

        output_path = os.path.join(OUTPUT_DIR, f"{shot_id_safe}_first_frame.jpg")
        print(f"[{shot_id_safe}]  t={time_range.split('-')[0]}  ({ffmpeg_time})  ->  {output_path}")

        ok = extract_frame(video_url, ffmpeg_time, output_path)
        if ok:
            print(f"    OK: saved {output_path}")
            success_count += 1
        else:
            print(f"    FAILED")
            fail_count += 1

    print()
    print(f"Done. {success_count} succeeded, {fail_count} failed.")