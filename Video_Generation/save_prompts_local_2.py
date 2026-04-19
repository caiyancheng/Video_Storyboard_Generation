"""
save_prompts_local_2.py  (本地运行)

读取评分 4/5 分的视频记录，基于 phase1_v16 标注文件为每条记录生成
Lv1 / Lv2 / Lv3 三个复杂度级别的完整 prompt，保存到本地文件夹。

复杂度说明：
  Lv1 - 基础级：镜头时间轴 + 简要描述 + 台词({}) + 音效(<>) + 摄像机基础信息
  Lv2 - 中级：Lv1 + 角色动作/表情 + 场景环境 + 视觉风格 + 音乐
  Lv3 - 完整级：Lv2 + 视觉特效 + 角色外观 + 完整叙事结构
         (风格特点→内容总结→动态描述→静态描述→附加信息)

输入：phase1_chinese_labels_v16/*.json（v16版本，含所有所需字段）

输出结构：
  PROMPT_OUT_ROOT/
    Lv1/  {vid_label}.txt
    Lv2/  {vid_label}.txt
    Lv3/  {vid_label}.txt
"""

import json
from pathlib import Path

from tqdm import tqdm

# ═══════════════════════ CONFIG ══════════════════════════════════════════════

SCORED_FILE = Path(
    "/Users/bytedance/Datasets/"
    "tt_template_hq_publish_data_1400k_USAU"
    ".dedup_item_id_aesthetic_quality_v1_filtered_scored.jsonl"
)

PHASE1_V16_DIR = Path(
    "/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/phase1_chinese_labels_v16"
)

PROMPT_OUT_ROOT = Path(
    "/Users/bytedance/Datasets/tt_template_1400k_15s_video_sample"
    "/shu_inverse_label/generated_videos/prompts_v2"
)

PROMPT_LEVELS = ["Lv1", "Lv2", "Lv3"]
TARGET_SCORES = {4, 5}
SKIP_EXISTING = True


# ═══════════════════════ AUDIO EXTRACTORS ════════════════════════════════════

def _extract_speech_lines(shot: dict) -> list[str]:
    """台词段落，用 {} 标注说话内容。"""
    lines = []
    for sp in shot.get("audio", {}).get("speech", []):
        for ev in sp.get("speech_events", []):
            ts = ev.get("timestamp", "")
            transcript = ev.get("transcript", "")
            style = ev.get("style", "")
            vf = ev.get("voice_fingerprint", {})
            quality = vf.get("voice_quality", "")
            if transcript:
                line = f"  [{ts}] 台词: {transcript}"
                details = []
                if style:
                    details.append(f"语气: {style}")
                if quality:
                    details.append(f"声音: {quality}")
                if details:
                    line += "  |  " + "  ".join(details)
                lines.append(line)
    return lines


def _extract_sfx_lines(shot: dict) -> list[str]:
    """音效段落，用 <> 标注音效内容。"""
    lines = []
    for sfx in shot.get("audio", {}).get("sound_effects", []):
        for ev in sfx.get("sound_events", []):
            ts = ev.get("timestamp", "")
            desc = ev.get("description", "")
            if desc:
                lines.append(f"  [{ts}] 音效: {desc}")
    return lines


def _extract_music_lines(shot: dict, lv: int) -> list[str]:
    lines = []
    for mus in shot.get("audio", {}).get("music", []):
        ts = mus.get("timestamp", "")
        style = mus.get("style", "")
        mf = mus.get("music_features", {})
        instru = mf.get("instrumentation", "")
        impact = mf.get("emotional_impact", "")
        if style:
            line = f"  [{ts}] 音乐: {style}"
            if impact:
                line += f"  情绪: {impact}"
            if lv >= 3 and instru:
                line += f"  乐器: {instru}"
            lines.append(line)
    return lines


# ═══════════════════════ VISUAL EXTRACTORS ═══════════════════════════════════

def _extract_camera_lines(shot: dict, lv: int) -> list[str]:
    cam = shot.get("visual", {}).get("camera", {})
    if not cam:
        return []
    angle = cam.get("camera_angle", "")
    scale = cam.get("shot_scale", "")
    mv = cam.get("camera_movement", {})
    traj = mv.get("movement_trajectory", "")
    speed = mv.get("speed", "")
    mood = cam.get("mood_atmosphere", "")
    color_tone = cam.get("color_tone", "")

    lines = []
    if angle or scale:
        lines.append(f"  角度/景别: {angle}  |  {scale}")
    if traj:
        spd_str = f"  速度: {speed}" if speed else ""
        lines.append(f"  运动轨迹: {traj}{spd_str}")
    if lv >= 2 and color_tone:
        lines.append(f"  色调: {color_tone}")
    if lv >= 2 and mood:
        lines.append(f"  氛围: {mood}")
    return lines


def _extract_subject_lines(shot: dict, lv: int) -> list[str]:
    lines = []
    for subj in shot.get("visual", {}).get("subject_tracking", []):
        sid = subj.get("id", "")
        ts_list = subj.get("timestamp_presence", [])
        ts_str = ", ".join(ts_list)
        action = subj.get("action", "")
        body = subj.get("body_movement", "")
        face = subj.get("facial_expression", "")
        appear = subj.get("appearance_description", "")

        lines.append(f"  {sid}  [{ts_str}]")
        if face:
            lines.append(f"    表情: {face}")
        if body:
            lines.append(f"    肢体动作: {body}")
        if action:
            lines.append(f"    行为: {action}")
        if lv >= 3 and appear:
            lines.append(f"    外观: {appear}")
    return lines


def _extract_location_lines(shot: dict) -> list[str]:
    lines = []
    for loc in shot.get("visual", {}).get("location_tracking", []):
        lid = loc.get("id", "")
        env = loc.get("visual_environment", "")
        ts_list = loc.get("timestamp_presence", [])
        ts_str = ", ".join(ts_list)
        if env:
            lines.append(f"  {lid}  [{ts_str}]")
            lines.append(f"    {env}")
    return lines


def _extract_visual_style_lines(shot: dict) -> list[str]:
    vs = shot.get("visual", {}).get("visual_style", {})
    if not vs:
        return []
    art = vs.get("art_style", "")
    palette = vs.get("color_palette", "")
    la = vs.get("lighting_analysis", {})
    brightness = la.get("brightness", "")
    saturation = la.get("saturation", "")
    contrast = la.get("contrast", "")

    lines = []
    if art:
        lines.append(f"  美术风格: {art}")
    if palette:
        lines.append(f"  色彩: {palette}")
    parts = []
    if brightness:
        parts.append(f"亮度={brightness}")
    if saturation:
        parts.append(f"饱和度={saturation}")
    if contrast:
        parts.append(f"对比度={contrast}")
    if parts:
        lines.append(f"  光线: {',  '.join(parts)}")
    return lines


def _extract_vfx_lines(shot: dict) -> list[str]:
    lines = []
    for vfx in shot.get("visual", {}).get("visual_effects", []):
        ts = vfx.get("timestamp", "")
        desc = vfx.get("description", "")
        color = vfx.get("color", "")
        intensity = vfx.get("intensity", "")
        etype = vfx.get("effect_type", "")
        if desc:
            line = f"  [{ts}] 特效({etype}): {desc}"
            if color:
                line += f"  颜色: {color}"
            if intensity:
                line += f"  强度: {intensity}"
            lines.append(line)
    return lines


# ═══════════════════════ PROMPT BUILDER ══════════════════════════════════════

def build_prompt(phase1_v16: dict, level: str) -> str:
    """
    Build prompt from v16 phase1 data.
    level: "Lv1" | "Lv2" | "Lv3"
    """
    lv = int(level[2])  # 1, 2, or 3
    shots = phase1_v16.get("shot_registry", [])
    video_id = phase1_v16.get("video_id", "")

    sections = [f"[视频生成 Prompt | {level}]\nVideo ID: {video_id}"]

    for shot in shots:
        shot_id = shot.get("id", "")
        time_range = shot.get("time_range", "")
        caption = shot.get("caption", "")

        block = [f"── {shot_id}  {time_range} ──"]
        block.append(f"概述: {caption}")

        # 1. 摄像机运动（所有级别，细节递增）
        cam_lines = _extract_camera_lines(shot, lv)
        if cam_lines:
            block.append("摄像机:")
            block.extend(cam_lines)

        # 2. 台词（所有级别，{} 标注）
        speech = _extract_speech_lines(shot)
        if speech:
            block.append("台词/语音:")
            block.extend(speech)

        # 3. 音效（所有级别，<> 标注）
        sfx = _extract_sfx_lines(shot)
        if sfx:
            block.append("音效:")
            block.extend(sfx)

        # 4. 音乐（Lv2+）
        if lv >= 2:
            music = _extract_music_lines(shot, lv)
            if music:
                block.append("音乐:")
                block.extend(music)

        # 5. 角色动作/表情（所有级别，Lv3 额外加外观）
        subj_lines = _extract_subject_lines(shot, lv)
        if subj_lines:
            block.append("角色:")
            block.extend(subj_lines)

        # 6. 场景/背景环境（Lv2+）
        if lv >= 2:
            loc_lines = _extract_location_lines(shot)
            if loc_lines:
                block.append("场景/背景:")
                block.extend(loc_lines)

        # 7. 视觉风格（Lv2+）
        if lv >= 2:
            style_lines = _extract_visual_style_lines(shot)
            if style_lines:
                block.append("视觉风格:")
                block.extend(style_lines)

        # 8. 视觉特效（Lv3）
        if lv >= 3:
            vfx_lines = _extract_vfx_lines(shot)
            if vfx_lines:
                block.append("视觉特效:")
                block.extend(vfx_lines)

        # 9. 完整叙事结构（Lv3）
        # 风格特点 → 内容总结 → 动态描述 → 静态描述 → 附加信息
        if lv >= 3:
            ns = shot.get("narrative_structure", {})
            style_feat = ns.get("style_features", "")
            content_sum = ns.get("content_summary", "")
            dynamic = ns.get("dynamic_description", "")
            static = ns.get("static_description", "")
            add_info = ns.get("additional_info", {})
            duration = add_info.get("duration", "")
            ar = add_info.get("aspect_ratio", "")

            if any([style_feat, content_sum, dynamic, static]):
                block.append("叙事结构:")
                if style_feat:
                    block.append(f"  [风格特点] {style_feat}")
                if content_sum:
                    block.append(f"  [内容总结] {content_sum}")
                if dynamic:
                    block.append(f"  [动态描述] {dynamic}")
                if static:
                    block.append(f"  [静态描述] {static}")
                if duration or ar:
                    block.append(f"  [附加信息] 时长={duration}  宽高比={ar}")

            dense = shot.get("dense_caption", "")
            if dense:
                block.append(f"详细描述:\n  {dense}")

        sections.append("\n".join(block))

    return "\n\n".join(sections)


# ═══════════════════════ FILE LOOKUP ═════════════════════════════════════════

def load_scored_records() -> list[dict]:
    records = []
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("_score") in TARGET_SCORES:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def find_v16_file(video_id: str, idx: int, score: int) -> Path | None:
    """查找 phase1_v16 文件，先按精确文件名查，再按 video_id glob。"""
    stem = f"id_{idx:04d}_score{score}_{video_id}"
    exact = PHASE1_V16_DIR / f"{stem}_phase1.json"
    if exact.exists():
        return exact
    matches = list(PHASE1_V16_DIR.glob(f"*{video_id}_phase1.json"))
    return matches[0] if matches else None


# ═══════════════════════ MAIN ════════════════════════════════════════════════

def main():
    level_dirs = {}
    for lv in PROMPT_LEVELS:
        d = PROMPT_OUT_ROOT / lv
        d.mkdir(parents=True, exist_ok=True)
        level_dirs[lv] = d

    records = load_scored_records()
    print(f"找到 {len(records)} 条评分 {sorted(TARGET_SCORES)} 分记录\n")

    ok = skip = fail = 0

    for rec in tqdm(records, desc="生成 prompt"):
        video_id = rec.get("video_id", "unknown")
        rec_idx = rec.get("_idx", 0)
        rec_score = rec.get("_score", 0)

        v16_path = find_v16_file(video_id, rec_idx, rec_score)
        if not v16_path:
            tqdm.write(f"[SKIP] 无 v16 标注文件: {video_id}")
            fail += 1
            continue

        # 从实际文件名解析 idx / score
        stem_parts = v16_path.stem.split("_")  # id_XXXX_scoreY_...
        try:
            rec_idx = int(stem_parts[1])
            rec_score = int(stem_parts[2].replace("score", ""))
        except (IndexError, ValueError):
            pass

        vid_label = f"id_{rec_idx:04d}_score{rec_score}_{video_id}"

        try:
            with v16_path.open(encoding="utf-8") as fp:
                phase1_v16 = json.load(fp)
        except Exception as e:
            tqdm.write(f"[FAIL] JSON 读取失败 {vid_label}: {e}")
            fail += 1
            continue

        all_skipped = True
        for lv in PROMPT_LEVELS:
            out_path = level_dirs[lv] / f"{vid_label}.txt"
            if SKIP_EXISTING and out_path.exists():
                skip += 1
                continue
            try:
                prompt = build_prompt(phase1_v16, lv)
                out_path.write_text(prompt, encoding="utf-8")
                all_skipped = False
                ok += 1
            except Exception as e:
                tqdm.write(f"[FAIL] {vid_label} {lv}: {e}")
                fail += 1

        if not all_skipped:
            tqdm.write(f"  ✓ {vid_label}")

    print(f"\n完成：写入 {ok} 个文件，跳过(已存在) {skip} 个，失败 {fail} 个")
    print(f"输出目录：{PROMPT_OUT_ROOT}")


if __name__ == "__main__":
    main()