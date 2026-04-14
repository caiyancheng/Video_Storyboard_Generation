"""
gemini_caller.py

直接复用 editing_magic_prompt 里的 get_gpt_resp 调用 Gemini 2.5。
"""

import sys
import time
from pathlib import Path

# 把 editing_magic_prompt 加入 path
_EMP_ROOT = Path("/Users/bytedance/Py_codes/editing_magic_prompt")
if str(_EMP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EMP_ROOT))

from editing_magic_prompt.modules.gpt_caller import GEMINI_MODEL, get_gpt_resp


def call_gemini(
    prompt: str,
    video_paths: list[Path] | None = None,
    enable_thinking: bool = False,
    max_tokens: int | None = 4096,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> str | None:
    """
    调用 Gemini 2.5，支持传入多个本地视频文件（base64 编码发送）。

    Returns
    -------
    str : 模型文本输出，失败返回 None。
    """
    data_list = []
    if video_paths:
        for vp in video_paths:
            data_list.append({"video_path": str(vp)})

    for attempt in range(retries):
        try:
            result = get_gpt_resp(
                model=GEMINI_MODEL,
                prompt=prompt,
                data_list=data_list if data_list else None,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
            )
            if result is not None:
                text, cot, usage = result
                return text
        except Exception as e:
            print(f"  [Gemini] attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    return None
