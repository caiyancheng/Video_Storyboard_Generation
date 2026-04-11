import subprocess
import json
import pandas as pd
from pathlib import Path

TOS_BASE_URL = "https://tosv-va.tiktok-row.org/obj/nebudata-us/"


def get_video_info_ffprobe(tos_key: str) -> dict:
    """用 ffprobe 获取视频分辨率和时长，失败返回空字典"""
    url = TOS_BASE_URL + tos_key
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}

    width, height, duration = None, None, None

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width")
            height = stream.get("height")
            duration = stream.get("duration") or data.get("format", {}).get("duration")
            break

    if duration is None:
        duration = data.get("format", {}).get("duration")

    return {
        "width": width,
        "height": height,
        "duration": float(duration) if duration is not None else None,
    }

FILE_PATH = "/Users/bytedance/Datasets/tt_template_hq_publish_data_1400k_USAU.dedup_item_id.csv"

# ========== 方法1：快速查看文件信息 ==========
def peek(n=5):
    """读取前 n 行，查看结构"""
    df = pd.read_csv(FILE_PATH, nrows=n)
    print(f"=== 前 {n} 行预览 ===")
    print(df.to_string())
    print(f"\n列名: {list(df.columns)}")
    print(f"数据类型:\n{df.dtypes}")
    return df

# ========== 方法2：获取基本信息（不加载全量）==========
def get_info():
    """只读元信息：行数、列名、文件大小"""
    file_size = Path(FILE_PATH).stat().st_size / (1024 ** 3)
    print(f"文件大小: {file_size:.2f} GB")

    # 只读列名（极快）
    header = pd.read_csv(FILE_PATH, nrows=0)
    print(f"列名 ({len(header.columns)} 列): {list(header.columns)}")

    # 统计行数
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1  # 减去 header
    print(f"总行数: {row_count:,}")

# ========== 方法3：一次性读取全量（内存够用时）==========
def read_all(usecols=None):
    """
    读取全量 CSV
    usecols: 只读指定列，如 ['item_id', 'score']，可大幅节省内存
    """
    print("正在读取全量 CSV...")
    df = pd.read_csv(
        FILE_PATH,
        usecols=usecols,        # 只读需要的列
        low_memory=False,       # 避免混合类型警告
        encoding="utf-8",
    )
    print(f"读取完成: {df.shape[0]:,} 行 x {df.shape[1]} 列")
    print(df.head())
    print(f"\n内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df

# ========== 方法4：分块读取（大文件省内存）==========
def read_by_chunk(chunk_size=100000, filter_col=None, filter_val=None):
    """
    分块读取 CSV
    chunk_size:  每块行数，根据内存调整（100k 约占 50-200MB）
    filter_col:  过滤列名（可选）
    filter_val:  保留等于该值的行（可选）
    """
    chunks = []
    reader = pd.read_csv(FILE_PATH, chunksize=chunk_size, low_memory=False)

    for i, chunk in enumerate(reader):
        if filter_col and filter_col in chunk.columns:
            chunk = chunk[chunk[filter_col] == filter_val]
        chunks.append(chunk)
        print(f"  已处理 {(i+1) * chunk_size:,} 行...", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n最终 DataFrame: {df.shape[0]:,} 行 x {df.shape[1]} 列")
    return df

# ========== 方法5：逐行读取并用 ffprobe 探测视频信息 ==========
def read_and_probe(max_rows=10):
    """
    逐行读取 CSV，提取 tos_key，用 ffprobe 获取视频分辨率和时长并打印。
    max_rows: 最多处理的行数（避免意外跑全量），传 None 表示全量。
    """
    reader = pd.read_csv(FILE_PATH, chunksize=1, low_memory=False)
    count = 0
    for chunk in reader:
        if max_rows is not None and count >= max_rows:
            break
        row = chunk.iloc[0]
        tos_key = row.get("tos_key")
        if pd.isna(tos_key) or not tos_key:
            print(f"[行 {count+1}] tos_key 为空，跳过")
            count += 1
            continue

        info = get_video_info_ffprobe(str(tos_key))

        if "error" in info:
            print(f"[行 {count+1}] tos_key={tos_key}  ffprobe 错误: {info['error']}")
        else:
            res = f"{info['width']}x{info['height']}" if info['width'] else "未知"
            dur = f"{info['duration']:.2f}s" if info['duration'] is not None else "未知"
            print(f"[行 {count+1}] tos_key={tos_key}  分辨率={res}  时长={dur}")

        count += 1


# ========== 主流程 ==========
if __name__ == "__main__":
    print(">>> 1. 文件基本信息")
    get_info()

    print("\n>>> 2. 查看前5行")
    peek(5)

    print("\n>>> 3. 逐行读取并探测视频信息（前10行）")
    read_and_probe(max_rows=10)

    # 示例：只读 item_id 列（速度快、内存小）
    # df = read_all(usecols=["item_id"])

    # 示例：读全量
    # df = read_all()

    # 示例：分块读取
    # df = read_by_chunk(chunk_size=100000)