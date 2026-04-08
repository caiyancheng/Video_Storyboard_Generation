import os
import json
import cv2
from pathlib import Path
import argparse
import numpy as np


def render_script_generated_storyline(json_path: str, output_dir: str = "output_html"):
    """
    Render a storyline JSON into an HTML file with visualized real clips (green) and AIGC segments (purple).
    Skips frames that are too dark or too bright based on pixel mean.
    Uses a modern, fashion-forward HTML/CSS design.
    """
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    real_clips = data.get("RealClips", {})
    scripts = data.get("Scripts", [])
    characters = data.get("Characters", {})
    environments = data.get("Environments", {})

    # Setup output dirs
    output_path = Path(output_dir)
    img_dir = output_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def is_valid_brightness(frame, dark_thresh=10, bright_thresh=245):
        """Check if frame is not too dark or too bright."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        return dark_thresh <= mean_val <= bright_thresh

    def extract_frame(video_path: str, time_sec: float, output_img_path: str):
        """Extract frame at time_sec from video, skip if too dark/bright."""
        if not os.path.exists(video_path):
            print(f"⚠️ Video not found: {video_path}")
            return False
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return False

        duration = total_frames / fps
        if time_sec >= duration:
            frame_id = total_frames - 1
        else:
            frame_id = int(time_sec * fps)
        frame_id = max(0, frame_id)

        current_frame = frame_id
        found_valid = False

        while current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            if is_valid_brightness(frame):
                found_valid = True
                break

            current_frame += 1
        if not found_valid:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False

        cv2.imwrite(output_img_path, frame)
        cap.release()
        return True

    # Pre-extract character and environment images
    char_env_imgs = {}

    for char_id, char_info in characters.items():
        ref = char_info.get("character_image_time", {})
        clip_idx = str(ref.get("clip_index"))
        time_sec = ref.get("time", 0.0)
        video_path = real_clips.get(clip_idx)
        if video_path:
            img_name = f"{char_id}.jpg"
            img_path = str(img_dir / img_name)
            if extract_frame(video_path, time_sec, img_path):
                char_env_imgs[char_id] = f"images/{img_name}"

    for env_id, env_info in environments.items():
        ref = env_info.get("environment_image_time", {})
        clip_idx = str(ref.get("clip_index"))
        time_sec = ref.get("time", 0.0)
        video_path = real_clips.get(clip_idx)
        if video_path:
            img_name = f"{env_id}.jpg"
            img_path = str(img_dir / img_name)
            if extract_frame(video_path, time_sec, img_path):
                char_env_imgs[env_id] = f"images/{img_name}"

    # Modern, fashion-forward HTML/CSS
    html_parts = []
    html_parts.append("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI-Generated Storyline</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --real-border: linear-gradient(135deg, #4CAF50, #81C784);
                --aigc-border: linear-gradient(135deg, #9C27B0, #BA68C8);
                --bg-light: #fafafa;
                --text-primary: #2d2d2d;
                --text-secondary: #666;
                --card-radius: 16px;
                --shadow: 0 6px 20px rgba(0,0,0,0.08);
            }
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Inter', system-ui, sans-serif;
                background-color: var(--bg-light);
                color: var(--text-primary);
                padding: 40px 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 960px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                font-weight: 700;
                font-size: 2.2em;
                margin-bottom: 40px;
                background: linear-gradient(90deg, #4CAF50, #9C27B0);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }
            .segment {
                background: white;
                border-radius: var(--card-radius);
                padding: 24px;
                margin-bottom: 32px;
                position: relative;
                box-shadow: var(--shadow);
                transition: transform 0.2s ease;
            }
            .segment:hover {
                transform: translateY(-2px);
            }
            .segment.real::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 5px;
                background: var(--real-border);
                border-radius: var(--card-radius) var(--card-radius) 0 0;
            }
            .segment.aigc::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 5px;
                background: var(--aigc-border);
                border-radius: var(--card-radius) var(--card-radius) 0 0;
            }
            .title {
                font-size: 1.4em;
                font-weight: 700;
                margin-bottom: 12px;
                color: var(--text-primary);
            }
            .duration {
                font-size: 0.95em;
                color: var(--text-secondary);
                margin-bottom: 16px;
                display: flex;
                gap: 16px;
            }
            .tag {
                background: #f0f0f0;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
            }
            .real .tag { color: #2E7D32; background: #E8F5E9; }
            .aigc .tag { color: #6A1B9A; background: #F3E5F5; }

            .description {
                margin: 14px 0;
                color: var(--text-primary);
                font-weight: 400;
            }
            .description strong {
                font-weight: 600;
                color: #444;
                margin-right: 6px;
            }
            .meta {
                font-size: 0.9em;
                color: var(--text-secondary);
                margin-top: 12px;
                display: flex;
                gap: 20px;
            }
            img.thumbnail {
                width: 100%;
                max-height: 300px;
                object-fit: cover;
                border-radius: 12px;
                margin-top: 16px;
                border: 1px solid #eee;
            }
            .ref-section {
                margin-top: 50px;
            }
            .ref-section h2 {
                text-align: center;
                margin-bottom: 24px;
                font-weight: 600;
                color: #333;
            }
            .ref-item {
                display: flex;
                align-items: flex-start;
                margin-bottom: 20px;
                gap: 16px;
            }
            img.ref-image {
                width: 90px;
                height: 90px;
                object-fit: cover;
                border-radius: 12px;
                border: 1px solid #eee;
                flex-shrink: 0;
            }
            .ref-content h3 {
                font-size: 1.1em;
                margin-bottom: 6px;
                font-weight: 600;
            }
            .ref-content p {
                color: var(--text-secondary);
                font-size: 0.95em;
            }
            hr {
                border: 0;
                height: 1px;
                background: #eee;
                margin: 50px 0;
            }
            @media (max-width: 600px) {
                .segment { padding: 20px; }
                .title { font-size: 1.2em; }
                .meta { flex-direction: column; gap: 8px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI-Generated Storyline</h1>
    """)

    # Process each script segment
    for idx, seg in enumerate(scripts):
        seg_type = seg.get("type", "unknown")
        duration = seg.get("duration", [0, 0])
        start, end = duration[0], duration[1]
        title = seg.get("title", "Untitled")
        key_desc = seg.get("key_frame_description", "")
        video_desc = seg.get("video_description", "")
        cam_motion = seg.get("camera_motion", "N/A")
        light_cond = seg.get("light_condition", "N/A")

        cls = "real" if seg_type == "real" else "aigc"
        html_parts.append(f'<div class="segment {cls}">')
        html_parts.append(f'<div class="title">{idx + 1}. {title}</div>')
        html_parts.append(
            f'<div class="duration">'
            f'<span>⏱️ [{start:.2f}s – {end:.2f}s]</span>'
            f'<span class="tag">{seg_type.upper()}</span>'
            f'</div>'
        )

        if seg_type == "real":
            clip_idx = str(seg.get("clip_index", -1))
            video_path = real_clips.get(clip_idx)
            if video_path:
                img_name = f"script_real_{idx}.jpg"
                img_path = str(img_dir / img_name)
                success = extract_frame(video_path, 0, img_path)  # use start time instead of 0
                if success:
                    html_parts.append(f'<img class="thumbnail" src="images/{img_name}" alt="Real clip frame">')
                else:
                    html_parts.append('<p style="color:#888; font-style:italic;">⏭️ No valid frame extracted (too dark/bright or missing)</p>')

        # Descriptions
        if key_desc.strip():
            html_parts.append(f'<div class="description"><strong>Key Frame:</strong> {key_desc}</div>')
        if video_desc.strip():
            html_parts.append(f'<div class="description"><strong>Video:</strong> {video_desc}</div>')

        # Camera & Lighting
        html_parts.append(f'<div class="meta">🎥 Camera: {cam_motion} | 💡 Lighting: {light_cond}</div>')

        html_parts.append('</div>')

    # Add Characters & Environments reference section
    html_parts.append('<div class="ref-section">')
    html_parts.append('<hr><h2>Reference Assets</h2>')

    if characters:
        html_parts.append('<h3>🎭 Characters</h3>')
        for char_id, info in characters.items():
            img_src = char_env_imgs.get(char_id, "")
            img_tag = f'<img class="ref-image" src="{img_src}">' if img_src else '<div class="ref-image" style="background:#f5f5f5;display:flex;align-items:center;justify-content:center;color:#aaa;">?</div>'
            html_parts.append(
                f'<div class="ref-item">'
                f'{img_tag}'
                f'<div class="ref-content">'
                f'<h3>{info.get("title", char_id)}</h3>'
                f'<p>{info.get("character_description", "")}</p>'
                f'</div>'
                f'</div>'
            )

    if environments:
        html_parts.append('<h3>🌆 Environments</h3>')
        for env_id, info in environments.items():
            img_src = char_env_imgs.get(env_id, "")
            img_tag = f'<img class="ref-image" src="{img_src}">' if img_src else '<div class="ref-image" style="background:#f5f5f5;display:flex;align-items:center;justify-content:center;color:#aaa;">?</div>'
            html_parts.append(
                f'<div class="ref-item">'
                f'{img_tag}'
                f'<div class="ref-content">'
                f'<h3>{info.get("title", env_id)}</h3>'
                f'<p>{info.get("environment_description", "")}</p>'
                f'</div>'
                f'</div>'
            )

    html_parts.append('</div>')  # ref-section
    html_parts.append("</div></body></html>")  # container

    # Write HTML
    output_file = output_path / "storyline.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    print(f"✅ HTML rendered successfully: {output_file.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a storyline JSON into an HTML visualization.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="generate_script/stitched_storyline_Xi_an_360p_clips.json",
        help="Path to the input storyline JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rendered_script_ai_generated",
        help="Directory to save the output HTML and images."
    )
    args = parser.parse_args()
    base_name = '_'.join(os.path.basename(args.json_path).split(".")[0].split('_')[2:4])
    output_dir = os.path.join(args.output_dir, base_name)
    render_script_generated_storyline(
        json_path=args.json_path,
        output_dir=output_dir,
    )