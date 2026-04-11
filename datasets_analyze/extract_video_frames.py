"""
extract_video_frames.py

读取 scored JSONL 文件，筛选出评分4分和5分的视频，
提取每个视频的第一帧和最后一帧，保存为PNG格式。
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# 配置文件路径
SCORED_FILE = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_hq_publish_data_1400k_USAU.dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl")
TOS_BASE_URL = "https://tosv-va.tiktok-row.org/obj/nebudata-us/"

# 输出目录
OUTPUT_BASE = Path("/mnt/bn/yilin4/yancheng/Datasets/tt_template_1400k_15s_video_sample")
FIRST_FRAME_DIR = OUTPUT_BASE / "first_frame"
LAST_FRAME_DIR = OUTPUT_BASE / "last_frame"

# 目标评分
TARGET_SCORES = {4, 5}


def create_directories():
    """创建输出目录"""
    FIRST_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    LAST_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    print(f"创建目录: {FIRST_FRAME_DIR}")
    print(f"创建目录: {LAST_FRAME_DIR}")


def load_scored_records(target_scores: set[int] = TARGET_SCORES) -> list[dict]:
    """加载评分符合要求的记录"""
    records = []
    if not SCORED_FILE.exists():
        print(f"错误: 文件 {SCORED_FILE} 不存在")
        return records
    
    print(f"正在读取文件: {SCORED_FILE}")
    
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="读取JSONL文件"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if rec.get("_score") in target_scores:
                records.append(rec)
    
    print(f"找到 {len(records)} 条评分为 {sorted(target_scores)} 分的记录")
    return records


def get_video_path(record: dict) -> str:
    """从记录中获取视频路径"""
    # 优先使用 video_url
    video_url = record.get("video_url")
    if video_url:
        return video_url
    
    # 如果没有 video_url，使用 tos_key 构建路径
    tos_key = record.get("tos_key")
    if tos_key:
        return TOS_BASE_URL + tos_key
    
    return None


def extract_frame_from_video(video_path: str, frame_type: str = "first") -> np.ndarray:
    """从视频中提取指定帧"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"视频没有帧: {video_path}")
        cap.release()
        return None
    
    # 根据帧类型设置目标帧
    if frame_type == "first":
        target_frame = 0
    elif frame_type == "last":
        target_frame = total_frames - 1
    else:
        print(f"未知帧类型: {frame_type}")
        cap.release()
        return None
    
    # 设置到目标帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    # 读取帧
    ret, frame = cap.read()
    
    cap.release()
    
    if ret:
        return frame
    else:
        print(f"无法读取{frame_type}帧: {video_path}")
        return None


def save_frame(frame: np.ndarray, output_path: Path):
    """保存帧为PNG文件"""
    try:
        cv2.imwrite(str(output_path), frame)
        return True
    except Exception as e:
        print(f"保存帧失败 {output_path}: {e}")
        return False


def process_video_frames(record: dict, index: int):
    """处理单个视频的帧提取"""
    video_path = get_video_path(record)
    if not video_path:
        print(f"记录 {index} 没有有效的视频路径")
        return
    
    # 生成文件名
    score = record.get("_score", "unknown")
    filename = f"video_{index:06d}_score_{score}_vid_{record['video_id']}.png"
    
    # 提取第一帧
    first_frame = extract_frame_from_video(video_path, "first")
    if first_frame is not None:
        first_output = FIRST_FRAME_DIR / filename
        if save_frame(first_frame, first_output):
            print(f"保存第一帧: {first_output}")
    
    # 提取最后一帧
    last_frame = extract_frame_from_video(video_path, "last")
    if last_frame is not None:
        last_output = LAST_FRAME_DIR / filename
        if save_frame(last_frame, last_output):
            print(f"保存最后一帧: {last_output}")


def main():
    """主函数"""
    # 创建输出目录
    create_directories()
    
    # 加载评分记录
    records = load_scored_records()
    
    if not records:
        print("没有找到符合条件的记录")
        return
    
    print(f"开始处理 {len(records)} 个视频...")
    
    # 处理每个视频
    for i, record in enumerate(tqdm(records, desc="处理视频")):
        process_video_frames(record, i)
    
    print("处理完成!")
    print(f"第一帧保存在: {FIRST_FRAME_DIR}")
    print(f"最后一帧保存在: {LAST_FRAME_DIR}")


if __name__ == "__main__":
    main()