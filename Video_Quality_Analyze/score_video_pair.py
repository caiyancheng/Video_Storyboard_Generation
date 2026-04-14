"""
score_video_pair.py

用 Gemini 2.5 对一对视频（GT + 生成视频）进行多维度质量评分。

评分维度：
  1. 相似性（40%）：5 个子维度
  2. 视频美学（30%）：5 个子维度
  3. 音频质量（30%）：4 个子维度

子维度取均值 → 母维度；母维度加权平均 → 总分。

返回 dict，包含所有子维度/母维度/总分，可直接写入 CSV。
"""

import json
import re
from pathlib import Path

from Video_Quality_Analyze.gemini_caller import call_gemini

# ═══════════════════════ 评分维度定义 ════════════════════════════════════════

SIMILARITY_KEYS = [
    "subject_consistency",   # 主体一致性
    "style_consistency",     # 风格一致性
    "motion_consistency",    # 动作一致性
    "scene_consistency",     # 场景一致性
    "audio_consistency",     # 音频一致性
]

AESTHETICS_KEYS = [
    "image_quality",         # 画面质量
    "composition",           # 构图美感
    "color",                 # 色彩表现
    "motion_smoothness",     # 运动流畅度
    "temporal_consistency",  # 时序一致性
]

AUDIO_KEYS = [
    "audio_clarity",         # 音频清晰度
    "timbre_consistency",    # 音色一致性
    "av_sync",               # 音画同步
    "rhythm_matching",       # 节奏匹配
]

WEIGHTS = {
    "similarity": 0.40,
    "aesthetics": 0.30,
    "audio":      0.30,
}

# ═══════════════════════ PROMPT ══════════════════════════════════════════════

EVAL_PROMPT = """\
你是一位专业的视频质量评估专家。我将提供两个视频：
  - 视频1：参考视频（Ground Truth）
  - 视频2：AI 生成视频

请对 AI 生成视频从以下三个维度进行评分，每个子维度给出 0-10 的整数分（0=极差，10=完美）。

────────────────────────────────────────────
【维度一：相似性（生成视频 vs 参考视频）】
  subject_consistency   : 主体一致性——人物/角色外观是否与参考视频吻合
  style_consistency     : 风格一致性——色调、光影、画风是否接近参考视频
  motion_consistency    : 动作一致性——运动幅度、节奏是否与参考视频相似
  scene_consistency     : 场景一致性——背景、环境是否与参考视频相似
  audio_consistency     : 音频一致性——音频风格、韵律是否与参考视频相似

【维度二：视频美学（仅针对生成视频本身）】
  image_quality         : 画面质量——清晰度、噪点
  composition           : 构图美感——主体位置、视觉平衡
  color                 : 色彩表现——饱和度、对比度、色彩和谐
  motion_smoothness     : 运动流畅度——帧间过渡是否平滑、有无明显抖动
  temporal_consistency  : 时序一致性——有无闪烁、角色/物体变形

【维度三：音频质量（仅针对生成视频本身）】
  audio_clarity         : 音频清晰度——噪点、失真程度
  timbre_consistency    : 音色一致性——整体音频风格是否前后一致
  av_sync               : 音画同步——声音与画面是否对齐
  rhythm_matching       : 节奏匹配——音乐节拍与画面动作是否协调
────────────────────────────────────────────

请严格按照以下 JSON 格式输出，不要添加其他内容：
```json
{
  "similarity": {
    "subject_consistency": <整数0-10>,
    "style_consistency": <整数0-10>,
    "motion_consistency": <整数0-10>,
    "scene_consistency": <整数0-10>,
    "audio_consistency": <整数0-10>
  },
  "aesthetics": {
    "image_quality": <整数0-10>,
    "composition": <整数0-10>,
    "color": <整数0-10>,
    "motion_smoothness": <整数0-10>,
    "temporal_consistency": <整数0-10>
  },
  "audio": {
    "audio_clarity": <整数0-10>,
    "timbre_consistency": <整数0-10>,
    "av_sync": <整数0-10>,
    "rhythm_matching": <整数0-10>
  },
  "reasoning": "<100字以内简述评分依据>"
}
```
"""


# ═══════════════════════ PARSE RESPONSE ══════════════════════════════════════

def _extract_json(text: str) -> dict | None:
    """从模型输出中提取 JSON 块。"""
    # 尝试 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 退而求其次：找第一个 { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _safe_score(d: dict, key: str) -> float:
    """安全读取分数，容错 None / 越界。"""
    val = d.get(key)
    if val is None:
        return 5.0   # 缺省中位分
    return max(0.0, min(10.0, float(val)))


def _compute_scores(parsed: dict) -> dict:
    """从解析后的 JSON 计算所有分数。"""
    scores: dict = {}

    # ── 相似性子维度 ──
    sim_raw = parsed.get("similarity", {})
    for k in SIMILARITY_KEYS:
        scores[f"sim_{k}"] = _safe_score(sim_raw, k)
    scores["sim_score"] = round(
        sum(scores[f"sim_{k}"] for k in SIMILARITY_KEYS) / len(SIMILARITY_KEYS), 4
    )

    # ── 美学子维度 ──
    aes_raw = parsed.get("aesthetics", {})
    for k in AESTHETICS_KEYS:
        scores[f"aes_{k}"] = _safe_score(aes_raw, k)
    scores["aes_score"] = round(
        sum(scores[f"aes_{k}"] for k in AESTHETICS_KEYS) / len(AESTHETICS_KEYS), 4
    )

    # ── 音频子维度 ──
    aud_raw = parsed.get("audio", {})
    for k in AUDIO_KEYS:
        scores[f"aud_{k}"] = _safe_score(aud_raw, k)
    scores["aud_score"] = round(
        sum(scores[f"aud_{k}"] for k in AUDIO_KEYS) / len(AUDIO_KEYS), 4
    )

    # ── 总分 ──
    scores["total_score"] = round(
        WEIGHTS["similarity"] * scores["sim_score"]
        + WEIGHTS["aesthetics"] * scores["aes_score"]
        + WEIGHTS["audio"] * scores["aud_score"],
        4,
    )

    scores["reasoning"] = parsed.get("reasoning", "")
    return scores


# ═══════════════════════ MAIN API ════════════════════════════════════════════

def score_video_pair(
    gt_path: Path,
    gen_path: Path,
) -> dict:
    """
    对一对视频（GT + 生成）进行质量评分。

    Returns
    -------
    dict with keys:
      sim_{sub}, aes_{sub}, aud_{sub},
      sim_score, aes_score, aud_score,
      total_score, reasoning, error
    """
    gt_path  = Path(gt_path)
    gen_path = Path(gen_path)

    if not gt_path.exists():
        return {"error": f"GT not found: {gt_path}"}
    if not gen_path.exists():
        return {"error": f"Gen not found: {gen_path}"}

    # ── 调用 Gemini（GT 在前，生成视频在后）────────────────────────────────
    print(f"  → calling Gemini for: {gen_path.name}")
    response = call_gemini(
        prompt=EVAL_PROMPT,
        video_paths=[gt_path, gen_path],
        enable_thinking=False,
        max_tokens=2048,
    )

    if response is None:
        return {"error": "Gemini returned None"}

    # ── 解析 JSON ────────────────────────────────────────────────────────────
    parsed = _extract_json(response)
    if parsed is None:
        return {"error": f"Failed to parse JSON from response: {response[:200]}"}

    scores = _compute_scores(parsed)
    scores["error"] = ""
    return scores


# ═══════════════════════ CSV COLUMN ORDER ════════════════════════════════════

CSV_SCORE_COLUMNS = (
    [f"sim_{k}" for k in SIMILARITY_KEYS]
    + ["sim_score"]
    + [f"aes_{k}" for k in AESTHETICS_KEYS]
    + ["aes_score"]
    + [f"aud_{k}" for k in AUDIO_KEYS]
    + ["aud_score", "total_score", "reasoning", "error"]
)
