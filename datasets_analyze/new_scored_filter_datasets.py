"""
new_scored_filter_datasets.py

用 tkinter GUI 对 filter_datasets 输出的结果逐条打分（1-5），
支持中途关闭后续接着打，每次打分立即保存。
"""

import json
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

FILTERED_FILE = Path("/Users/bytedance/Datasets/tt_template_hq_publish_data_1400k_USAU.dedup_item_id_aesthetic_quality_v1_filtered.jsonl")
SCORED_FILE   = FILTERED_FILE.with_name(FILTERED_FILE.stem + "_scored.jsonl")
TOS_BASE_URL  = "https://tosv-va.tiktok-row.org/obj/nebudata-us/"


# ========== 数据 IO ==========

def load_filtered() -> list[dict]:
    records = []
    with FILTERED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_scored() -> dict[str, dict]:
    scored = {}
    if not SCORED_FILE.exists():
        return scored
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                scored[rec["video_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return scored


def save_all(scored: dict[str, dict]) -> None:
    with SCORED_FILE.open("w", encoding="utf-8") as f:
        for rec in scored.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ========== GUI ==========

class ScoringApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("视频评分工具")
        self.root.resizable(True, True)

        self.records = load_filtered()
        self.scored  = load_scored()
        self.total   = len(self.records)

        # 找第一条未打分的
        self.idx = 0
        self._advance_to_unscored()

        self._build_ui()
        self._render()

    # ------ 导航 ------

    def _advance_to_unscored(self):
        while self.idx < self.total and self._vid() in self.scored:
            self.idx += 1

    def _vid(self) -> str:
        return self.records[self.idx]["video_id"] if self.idx < self.total else ""

    def _rec(self) -> dict:
        return self.records[self.idx]

    # ------ UI 构建 ------

    def _build_ui(self):
        mono = tkfont.Font(family="Menlo", size=13)
        bold = tkfont.Font(family="Menlo", size=13, weight="bold")

        # 进度
        self.progress_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.progress_var, font=bold, fg="#333").pack(pady=(12, 0))

        # 信息卡片
        frame = tk.Frame(self.root, bd=1, relief=tk.SOLID, padx=16, pady=12, bg="#f9f9f9")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.fields: dict[str, tk.StringVar] = {}
        labels = [
            ("video_id",   "video_id"),
            ("duration",   "时长"),
            ("source_q",   "source_quality"),
            ("quality_l",  "quality_level"),
            ("confidence", "confidence"),
            ("has_hook",   "has_strong_hook"),
            ("category",   "category"),
            ("notes",      "quality_notes"),
        ]
        for row, (key, display) in enumerate(labels):
            tk.Label(frame, text=f"{display}:", font=bold, bg="#f9f9f9",
                     anchor="w", width=18).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            self.fields[key] = var
            tk.Label(frame, textvariable=var, font=mono, bg="#f9f9f9",
                     anchor="w", wraplength=620, justify="left").grid(row=row, column=1, sticky="w", pady=2)

        # URL（可点击）
        tk.Label(frame, text="URL:", font=bold, bg="#f9f9f9",
                 anchor="w", width=18).grid(row=len(labels), column=0, sticky="w", pady=2)
        self.url_var = tk.StringVar()
        url_label = tk.Label(frame, textvariable=self.url_var, font=mono, bg="#f9f9f9",
                             fg="#1a6ecc", cursor="hand2", anchor="w",
                             wraplength=620, justify="left")
        url_label.grid(row=len(labels), column=1, sticky="w", pady=2)
        url_label.bind("<Button-1>", lambda _: webbrowser.open(self.url_var.get()))

        # 已打分标签
        self.score_label_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.score_label_var,
                 font=bold, fg="#888").pack(pady=(0, 4))

        # 打分按钮
        score_frame = tk.Frame(self.root)
        score_frame.pack(pady=6)
        self.score_btns = []
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
        for s in range(1, 6):
            btn = tk.Button(
                score_frame, text=f"  {s}  ",
                font=bold, bg=colors[s - 1], fg="white",
                activebackground=colors[s - 1],
                relief=tk.FLAT, padx=14, pady=8,
                command=lambda score=s: self._submit(score),
            )
            btn.pack(side=tk.LEFT, padx=6)
            self.score_btns.append(btn)

        # 操作按钮
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=(4, 14))
        tk.Button(nav_frame, text="← 回退", font=mono, width=10,
                  command=self._back).pack(side=tk.LEFT, padx=8)
        tk.Button(nav_frame, text="跳过 →", font=mono, width=10,
                  command=self._skip).pack(side=tk.LEFT, padx=8)
        tk.Button(nav_frame, text="统计", font=mono, width=10,
                  command=self._show_stats).pack(side=tk.LEFT, padx=8)

        self.root.bind("<Key>", self._on_key)

    # ------ 渲染当前记录 ------

    def _render(self):
        done  = sum(1 for r in self.records if r["video_id"] in self.scored)
        remaining = self.total - done
        self.progress_var.set(f"进度：{done} / {self.total}（剩余 {remaining} 条未打分）")

        if self.idx >= self.total:
            for var in self.fields.values():
                var.set("")
            self.url_var.set("")
            self.score_label_var.set("全部完成！")
            for btn in self.score_btns:
                btn.config(state=tk.DISABLED)
            messagebox.showinfo("完成", f"全部 {self.total} 条已打分！\n结果保存至：\n{SCORED_FILE}")
            return

        rec    = self._rec()
        result = rec.get("result", {})
        tos    = rec.get("tos_key", "")
        url    = rec.get("video_url") or (TOS_BASE_URL + tos if tos else "N/A")
        dur    = rec.get("_duration") or rec.get("duration") or result.get("duration")
        dur_s  = f"{dur:.2f}s" if dur is not None else "N/A"

        self.fields["video_id"].set(rec.get("video_id", ""))
        self.fields["duration"].set(dur_s)
        self.fields["source_q"].set(result.get("source_quality", ""))
        self.fields["quality_l"].set(result.get("quality_level", ""))
        self.fields["confidence"].set(str(result.get("confidence", "")))
        self.fields["has_hook"].set(str(result.get("has_strong_hook", "")))
        self.fields["category"].set(result.get("category", ""))
        self.fields["notes"].set(result.get("quality_notes", ""))
        self.url_var.set(url)

        vid = self._vid()
        if vid in self.scored:
            self.score_label_var.set(f"已打分：{self.scored[vid]['_score']} 分（可重新打）")
        else:
            self.score_label_var.set(f"第 {self.idx + 1} 条，请打分：")

    # ------ 操作 ------

    def _submit(self, score: int):
        if self.idx >= self.total:
            return
        rec = dict(self._rec())
        rec["_score"] = score
        self.scored[rec["video_id"]] = rec
        save_all(self.scored)
        self.idx += 1
        self._advance_to_unscored()
        self._render()

    def _skip(self):
        if self.idx >= self.total:
            return
        self.idx += 1
        self._advance_to_unscored()
        self._render()

    def _back(self):
        if self.idx <= 0:
            return
        self.idx -= 1
        self._render()

    def _on_key(self, event):
        if event.char in {"1", "2", "3", "4", "5"}:
            self._submit(int(event.char))
        elif event.char in {"s", "S"}:
            self._skip()
        elif event.char in {"b", "B"}:
            self._back()

    def _show_stats(self):
        dist: dict[int, int] = {}
        for r in self.scored.values():
            s = r.get("_score")
            if isinstance(s, int):
                dist[s] = dist.get(s, 0) + 1
        if not dist:
            messagebox.showinfo("统计", "还没有任何评分记录")
            return
        lines = [f"已打分 {len(self.scored)} / {self.total} 条\n"]
        for score in sorted(dist):
            bar = "█" * dist[score]
            lines.append(f"{score} 分：{dist[score]:>4} 条  {bar}")
        messagebox.showinfo("评分分布", "\n".join(lines))


# ========== 入口 ==========

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x620")
    ScoringApp(root)
    root.mainloop()
