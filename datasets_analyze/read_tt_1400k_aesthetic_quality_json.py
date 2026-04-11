import json
from collections import Counter
import pandas as pd
from pathlib import Path
from tqdm import tqdm

FILE_PATH = "/Users/bytedance/Datasets/tt_template_hq_publish_data_1400k_USAU.dedup_item_id_aesthetic_quality_v1.jsonl"

# ========== 方法1：查看前几行结构 ==========
def peek(n=5):
    """快速查看前 n 行的结构"""
    print(f"=== 前 {n} 条记录 ===")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            record = json.loads(line)
            print(f"\n[{i}] keys: {list(record.keys())}")
            print(json.dumps(record, ensure_ascii=False, indent=2)[:1000])  # 只打印前500字符

# ========== 方法2：统计总行数 ==========
def count_lines():
    """统计总行数（不加载到内存）"""
    count = 0
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    print(f"总行数: {count:,}")
    return count

# ========== 方法3：逐行读取 + 过滤（省内存）==========
def read_by_line(filter_fn=None, max_rows=None):
    """
    逐行读取，可选过滤条件
    filter_fn: 传入 record dict，返回 True/False
    max_rows:  最多读取多少行
    """
    results = []
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            record = json.loads(line)
            if filter_fn is None or filter_fn(record):
                results.append(record)
    print(f"读取到 {len(results):,} 条记录")
    return results

# ========== 方法4：pandas 分块读取（适合数据分析）==========
def read_with_pandas(chunk_size=50000, filter_col=None, filter_val=None):
    """
    用 pandas 分块读取，返回完整 DataFrame
    chunk_size:  每块行数，根据内存调整
    filter_col:  过滤的列名（可选）
    filter_val:  过滤的阈值（保留大于该值的行）
    """
    chunks = []
    for i, chunk in enumerate(pd.read_json(FILE_PATH, lines=True, chunksize=chunk_size)):
        if filter_col and filter_col in chunk.columns:
            chunk = chunk[chunk[filter_col] > filter_val]
        chunks.append(chunk)
        print(f"  已处理 {(i+1) * chunk_size:,} 行...", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n最终 DataFrame: {df.shape[0]:,} 行 x {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}")
    return df


# ========== 方法5：统计 source_quality / quality_level 分布 ==========
def analyze_quality_distribution():
    """
    扫描全量，统计 result.source_quality 和 result.quality_level 的值分布，
    用 tqdm 显示进度，输出各值数量及占比。
    """
    total_lines = 0
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    source_quality_counter: Counter = Counter()
    quality_level_counter: Counter = Counter()

    bad_lines = 0
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="扫描质量分布"):
            try:
                result = json.loads(line).get("result", {})
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            source_quality_counter[result.get("source_quality") or "<missing>"] += 1
            quality_level_counter[result.get("quality_level") or "<missing>"] += 1

    if bad_lines:
        print(f"\n  [警告] 跳过 {bad_lines:,} 条 JSON 解析失败的行")

    def _print_dist(name: str, counter: Counter) -> None:
        total = sum(counter.values())
        print(f"\n  {name} 分布（共 {total:,} 条）：")
        for val, cnt in sorted(counter.items(), key=lambda x: -x[1]):
            bar = "#" * int(cnt / total * 40)
            print(f"    {val:<20s}  {cnt:>8,}  ({cnt/total*100:5.2f}%)  {bar}")

    print("\n===== quality 字段分布统计 =====")
    _print_dist("source_quality", source_quality_counter)
    _print_dist("quality_level",  quality_level_counter)

    return source_quality_counter, quality_level_counter


# ========== 方法7：筛选 high quality 记录 ==========
def filter_high_quality(max_rows=None, print_n=20):
    """
    筛选满足以下条件的记录：
      - result.confidence >= 0.95
      - result.source_quality == "high"
      - result.quality_level  == "high"

    max_rows:  扫描的最大行数，None 表示全量
    print_n:   最多打印多少条结果（None 表示全部）
    返回符合条件的记录列表
    """
    def is_high_quality(record):
        result = record.get("result", {})
        return (
            result.get("confidence", 0) >= 0.95
            and result.get("source_quality") == "high"
            and result.get("quality_level") == "high"
            and result.get("has_strong_hook") is True
        )

    matched = []
    scanned = 0
    bad_lines = 0
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="筛选 high quality"):
            if max_rows is not None and scanned >= max_rows:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            scanned += 1
            if is_high_quality(rec):
                matched.append(rec)

    if bad_lines:
        print(f"\n  [警告] 跳过 {bad_lines:,} 条 JSON 解析失败的行")
    print(f"扫描 {scanned:,} 行，命中 {len(matched):,} 条 high quality 记录")

    to_print = matched if print_n is None else matched[:print_n]
    for i, rec in enumerate(to_print):
        res = rec.get("result", {})
        print(
            f"\n[{i+1}] video_id={rec.get('video_id')}  item_id={rec.get('item_id')}\n"
            f"     tos_key={rec.get('tos_key')}\n"
            f"     source_quality={res.get('source_quality')}  quality_level={res.get('quality_level')}  "
            f"confidence={res.get('confidence')}  category={res.get('category')}\n"
            f"     quality_notes={res.get('quality_notes')}"
        )

    return matched


# ========== 主流程 ==========
if __name__ == "__main__":
    print(">>> 1. 查看前5条记录结构")
    peek(5)

    print("\n>>> 2. 统计总行数")
    count_lines()

    print("\n>>> 3. 逐行读取前1000行（示例）")
    records = read_by_line(max_rows=1000)
    print(f"   示例字段: {list(records[0].keys()) if records else '无数据'}")

    print("\n>>> 4. 统计 source_quality / quality_level 全量分布")
    analyze_quality_distribution()

    print("\n>>> 5. 筛选 high quality 记录（confidence>=0.95, source_quality=high, quality_level=high）")
    high_quality_records = filter_high_quality(max_rows=None, print_n=20)

    # 示例：pandas 分块读全量（数据量大时慢，但可做统计分析）
    # 取消注释下面这行来读全量：
    # df = read_with_pandas(chunk_size=50000)