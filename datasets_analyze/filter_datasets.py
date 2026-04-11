"""
filter_datasets.py

从 aesthetic quality JSONL 中筛选满足以下全部条件的记录：
  1. result.source_quality == "high"
  2. result.quality_level  == "high"
  3. result.confidence     >= 0.95
  4. 视频时长在 [DURATION_MIN, DURATION_MAX] 秒之间

时长获取优先级：
  - 记录本身的 "duration" 键（若存在且非 null）
  - 否则调用 ffprobe 通过 TOS URL 探测

输出文件与源文件同目录，文件名自动加 _filtered 后缀。
"""

import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# ========== 配置 ==========
SRC_FILE = Path("/Users/bytedance/Datasets/tt_template_hq_publish_data_1400k_USAU.dedup_item_id_aesthetic_quality_v1.jsonl")
OUT_FILE = SRC_FILE.with_name(SRC_FILE.stem + "_filtered.jsonl")

TOS_BASE_URL = "https://tosv-va.tiktok-row.org/obj/nebudata-us/"

DURATION_MIN = 14.0   # 秒（含）
DURATION_MAX = 15.0   # 秒（含）

# ffprobe 并发线程数（网络 IO 为主，可适当调大）
FFPROBE_WORKERS = 8


# ========== 工具函数 ==========

def is_high_quality(result: dict) -> bool:
    return (
        (result.get("source_quality") or "") == "high"
        and (result.get("quality_level") or "") == "high"
        and (result.get("confidence") or 0) >= 0.95
        and result.get("has_strong_hook") is True
    )


def ffprobe_duration(tos_key: str) -> float | None:
    """通过 ffprobe 获取视频时长（秒），失败返回 None"""
    url = TOS_BASE_URL + tos_key
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(proc.stdout)
    except Exception:
        return None

    # 优先从 video stream 取，回退到 format
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            dur = stream.get("duration") or data.get("format", {}).get("duration")
            if dur is not None:
                return float(dur)
    dur = data.get("format", {}).get("duration")
    return float(dur) if dur is not None else None


def get_duration(rec: dict) -> float | None:
    """先查记录自带的 duration 键，没有则调用 ffprobe"""
    dur = rec.get("duration")
    if dur is not None:
        try:
            return float(dur)
        except (TypeError, ValueError):
            pass
    # 也检查 result 内部
    dur = rec.get("result", {}).get("duration")
    if dur is not None:
        try:
            return float(dur)
        except (TypeError, ValueError):
            pass
    return ffprobe_duration(rec.get("tos_key", ""))


def in_duration_range(duration: float | None) -> bool:
    if duration is None:
        return False
    return DURATION_MIN <= duration <= DURATION_MAX


# ========== 主筛选流程 ==========

def filter_datasets(
    duration_min: float = DURATION_MIN,
    duration_max: float = DURATION_MAX,
    ffprobe_workers: int = FFPROBE_WORKERS,
    max_results: int | None = None,
):
    """
    max_results: 找到指定数量的符合条件的记录后立即停止，None 表示扫全量
    """
    global DURATION_MIN, DURATION_MAX
    DURATION_MIN, DURATION_MAX = duration_min, duration_max

    # 统计总行数（给 tqdm 用）
    print(f"统计总行数...")
    total_lines = sum(1 for _ in SRC_FILE.open("r", encoding="utf-8"))
    print(f"总行数: {total_lines:,}")
    if max_results is not None:
        print(f"目标: 找到 {max_results} 条符合条件的记录后停止")

    # 第一步：质量过滤（纯内存，极快）
    print("\n[Step 1] 质量过滤（source_quality=high, quality_level=high, confidence>=0.95）...")
    quality_passed = []
    bad_lines = 0
    with SRC_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="质量过滤"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if is_high_quality(rec.get("result", {})):
                quality_passed.append(rec)

    if bad_lines:
        print(f"  [警告] 跳过 {bad_lines:,} 条 JSON 解析失败的行")
    print(f"  质量过滤后剩余: {len(quality_passed):,} 条")

    # 第二步：时长过滤（先用记录自带字段，缺失才 ffprobe）
    print(f"\n[Step 2] 时长过滤（{duration_min}s ~ {duration_max}s）...")

    # 分成两类：有内置 duration 的直接判断，没有的走 ffprobe
    direct_pass = []
    need_probe = []

    for r in quality_passed:
        if max_results is not None and len(direct_pass) >= max_results:
            break
        d = r.get("duration")
        if d is None:
            d = r.get("result", {}).get("duration")
        if d is not None:
            try:
                if DURATION_MIN <= float(d) <= DURATION_MAX:
                    r["_duration"] = float(d)
                    direct_pass.append(r)
            except (TypeError, ValueError):
                need_probe.append(r)
        else:
            need_probe.append(r)

    print(f"  内置 duration 命中: {len(direct_pass):,} 条")

    # 若已达到目标数量，跳过 ffprobe
    remaining = None
    if max_results is not None:
        remaining = max_results - len(direct_pass)
        if remaining <= 0:
            need_probe = []
        else:
            need_probe = need_probe  # 保持原列表，ffprobe 阶段再截断

    print(f"  需要 ffprobe 探测: {len(need_probe):,} 条（并发={ffprobe_workers}）")

    probe_pass = []
    probe_fail = 0

    if need_probe:
        def probe_one(r):
            d = ffprobe_duration(r.get("tos_key", ""))
            return r, d

        with ThreadPoolExecutor(max_workers=ffprobe_workers) as executor:
            futures = {executor.submit(probe_one, r): r for r in need_probe}
            with tqdm(total=len(need_probe), desc="ffprobe 探测") as pbar:
                for future in as_completed(futures):
                    r, d = future.result()
                    pbar.update(1)
                    if d is None:
                        probe_fail += 1
                    elif DURATION_MIN <= d <= DURATION_MAX:
                        r["_duration"] = d
                        probe_pass.append(r)
                        if remaining is not None and len(probe_pass) >= remaining:
                            # 取消剩余任务
                            for f in futures:
                                f.cancel()
                            break

        print(f"  ffprobe 命中: {len(probe_pass):,} 条，无法获取时长: {probe_fail:,} 条")

    final = direct_pass + probe_pass
    if max_results is not None:
        final = final[:max_results]
    print(f"\n最终命中: {len(final):,} 条")

    # 写输出文件
    print(f"\n写入 {OUT_FILE} ...")
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for rec in tqdm(final, desc="写入"):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"完成！共写入 {len(final):,} 条记录 -> {OUT_FILE}")
    return final


# ========== 入口 ==========
if __name__ == "__main__":
    filter_datasets(
        duration_min=14.0,
        duration_max=15.0,
        ffprobe_workers=8,
        max_results=200,
    )
