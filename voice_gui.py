"""Window for vocal timbre, noise, jitter, smoothness, and random micro-pauses."""

from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import tkinter as tk

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import voice_pipeline as pipeline

FONT = "Microsoft YaHei UI"
BG = "#1c1b22"
CARD = "#2a2833"
TEXT = "#f6f2ea"
MUTED = "#b7b1c2"
ACCENT = "#e7a3bc"
ACCENT_DARK = "#c97d99"
TROUGH = "#3c3846"

GROUPS = (
    (
        "声音",
        (
            ("formant", "音色", 0.70, 1.45, 0.01, 0.86, "更暖", "更亮"),
            ("pitch", "音高", -12, 12, 1, 0, "更低", "更高"),
            ("smooth", "平滑度", 0, 100, 1, 45, "保留细节", "更顺"),
        ),
    ),
    (
        "噪声",
        (
            ("denoise", "降噪", 0, 100, 1, 40, "关闭", "更干净"),
            ("noise", "模拟噪声", 0, 100, 1, 0, "无", "更明显"),
        ),
    ),
    (
        "随机抖动",
        (
            ("jitter", "抖动范围", 0, 80, 1, 0, "稳定", "更晃"),
            ("chance", "抖动概率", 0, 100, 1, 0, "很少", "经常"),
        ),
    ),
    (
        "随机停顿",
        (
            ("count", "停顿次数", 0, 40, 1, 9, "无", "更密"),
            ("length", "停顿时长", 20, 160, 5, 50, "更短", "更长"),
            ("seed", "随机种子", 1, 500, 1, 1, "一组", "另一组"),
        ),
    ),
)
SLIDERS = tuple(item for _title, items in GROUPS for item in items)


def _enable_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class VoiceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("人声调整")
        self.root.configure(bg=BG)
        self.root.minsize(760, 680)
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.media = tk.StringVar(value=self._default_media())
        self.formant = tk.DoubleVar(value=0.86)
        self.pitch = tk.DoubleVar(value=0)
        self.smooth = tk.DoubleVar(value=45)
        self.denoise = tk.DoubleVar(value=40)
        self.noise = tk.DoubleVar(value=0)
        self.jitter = tk.DoubleVar(value=0)
        self.chance = tk.DoubleVar(value=0)
        self.count = tk.DoubleVar(value=9)
        self.length = tk.DoubleVar(value=50)
        self.seed = tk.DoubleVar(value=1)
        self.vars = {
            "formant": self.formant,
            "pitch": self.pitch,
            "smooth": self.smooth,
            "denoise": self.denoise,
            "noise": self.noise,
            "jitter": self.jitter,
            "chance": self.chance,
            "count": self.count,
            "length": self.length,
            "seed": self.seed,
        }
        self.value_labels: dict[str, tk.Label] = {}
        self._resolution = {item[0]: item[4] for _title, items in GROUPS for item in items}
        self._build()
        self.refresh_labels()
        self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _default_media(self) -> str:
        folders = (pipeline.APP_DIR, pipeline.APP_DIR.parent, Path.cwd())
        for folder in folders:
            candidate = folder / "music.mp4"
            if candidate.exists():
                return str(candidate.resolve())
        return ""

    def _build(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, font=(FONT, 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#241820",
            font=(FONT, 11, "bold"),
            padding=(16, 8),
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#f3c4d6"), ("disabled", "#5c5460")],
            foreground=[("disabled", "#d9d3df")],
        )
        style.configure(
            "Ghost.TButton",
            background="#3a3542",
            foreground=TEXT,
            font=(FONT, 10),
            padding=(14, 8),
            borderwidth=0,
        )
        style.map("Ghost.TButton", background=[("active", "#4a4454"), ("disabled", "#2e2b34")])
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=TROUGH,
            background=ACCENT,
            bordercolor=CARD,
            lightcolor=ACCENT,
            darkcolor=ACCENT_DARK,
        )
        style.configure(
            "Pink.Horizontal.TScale",
            background=ACCENT,
            troughcolor=TROUGH,
            bordercolor=CARD,
            lightcolor=ACCENT,
            darkcolor=ACCENT_DARK,
        )
        style.map("Pink.Horizontal.TScale", background=[("active", "#f3c4d6")])

        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=22, pady=18)

        tk.Label(outer, text="人声调整", bg=BG, fg=TEXT, font=(FONT, 22, "bold")).pack(anchor="w")
        tk.Label(
            outer,
            text="分离只做一次。默认会压低电流噪声；模拟噪声和抖动默认关闭。滑块可滚动。",
            bg=BG,
            fg=MUTED,
            font=(FONT, 10),
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(4, 14))

        file_card = tk.Frame(outer, bg=CARD, highlightthickness=0)
        file_card.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(file_card, bg=CARD)
        inner.pack(fill="x", padx=16, pady=14)
        tk.Label(inner, text="输入文件", bg=CARD, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
        row = tk.Frame(inner, bg=CARD)
        row.pack(fill="x", pady=(6, 8))
        entry = tk.Entry(
            row,
            textvariable=self.media,
            bg="#211f28",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=(FONT, 10),
        )
        entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        ttk.Button(row, text="选择", style="Ghost.TButton", command=self._browse).pack(side="left")
        self.separate_btn = ttk.Button(inner, text="分离人声", style="Accent.TButton", command=self._separate)
        self.separate_btn.pack(anchor="w")

        self.summary = tk.Label(
            outer,
            text="",
            bg=BG,
            fg=ACCENT,
            font=(FONT, 10),
            anchor="w",
            justify="left",
            wraplength=760,
        )

        holder = tk.Frame(outer, bg=CARD)
        self.slider_holder = holder
        canvas = tk.Canvas(holder, bg=CARD, highlightthickness=0, bd=0, width=820, height=380)
        self.slider_canvas = canvas
        scroll = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = tk.Frame(canvas, bg=CARD)
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))

        inner = tk.Frame(body, bg=CARD)
        inner.pack(fill="x", padx=16, pady=(8, 12))
        tk.Label(inner, text="调整", bg=CARD, fg=MUTED, font=(FONT, 9)).pack(anchor="w", pady=(4, 2))
        for index, (title, items) in enumerate(GROUPS):
            tk.Label(
                inner,
                text=title,
                bg=CARD,
                fg=MUTED,
                font=(FONT, 10, "bold"),
            ).pack(anchor="w", pady=(10 if index else 4, 0))
            for key, label, start, end, resolution, default, left, right in items:
                self._slider(inner, key, label, start, end, resolution, default, left, right)

        self._bind_wheel(holder)

        # Pack the footer first so a short window cannot squeeze the buttons away.
        self.status = tk.Label(outer, text=self._idle_status(), bg=BG, fg=MUTED, font=(FONT, 10), anchor="w")
        self.status.pack(side="bottom", fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(outer, style="Accent.Horizontal.TProgressbar", mode="determinate", maximum=1)
        self.progress.pack(side="bottom", fill="x", pady=(8, 0))
        actions = tk.Frame(outer, bg=BG)
        actions.pack(side="bottom", fill="x", pady=(10, 0))
        self.preview_btn = ttk.Button(actions, text="预览 20 秒", style="Ghost.TButton", command=self._preview)
        self.preview_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="停止播放", style="Ghost.TButton", command=self._stop)
        self.stop_btn.pack(side="left", padx=8)
        self.export_btn = ttk.Button(actions, text="导出 MP3", style="Accent.TButton", command=self._export)
        self.export_btn.pack(side="right")
        self.summary.pack(side="bottom", fill="x", pady=(8, 0))
        holder.pack(fill="both", expand=True, pady=(0, 4))
        self._place_window()
        self.root.after(50, self._place_window)

    def _place_window(self) -> None:
        self.root.update_idletasks()
        # A frozen build can report a tiny screen and then clamp the window to it.
        self.root.maxsize(1800, 1400)
        self.root.minsize(860, 720)
        self.root.geometry("900x920+80+40")

    def _slider(self, parent, key, label, start, end, resolution, default, left, right) -> None:
        block = tk.Frame(parent, bg=CARD)
        block.pack(fill="x", pady=2)
        header = tk.Frame(block, bg=CARD)
        header.pack(fill="x")
        tk.Label(header, text=label, bg=CARD, fg=TEXT, font=(FONT, 11)).pack(side="left")
        value = tk.Label(header, text="", bg=CARD, fg=ACCENT, font=(FONT, 11, "bold"))
        value.pack(side="right")
        self.value_labels[key] = value
        scale = ttk.Scale(
            block,
            from_=start,
            to=end,
            orient="horizontal",
            variable=self.vars[key],
            command=lambda _value, name=key: self._on_scale(name),
            style="Pink.Horizontal.TScale",
        )
        self.vars[key].set(default)
        scale.pack(fill="x", pady=(2, 0))
        ends = tk.Frame(block, bg=CARD)
        ends.pack(fill="x")
        tk.Label(ends, text=left, bg=CARD, fg=MUTED, font=(FONT, 8)).pack(side="left")
        tk.Label(ends, text=right, bg=CARD, fg=MUTED, font=(FONT, 8)).pack(side="right")

    def _bind_wheel(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", self._on_wheel)
        for child in widget.winfo_children():
            self._bind_wheel(child)

    def _on_wheel(self, event) -> None:
        self.slider_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_scale(self, key: str) -> None:
        resolution = self._resolution[key]
        var = self.vars[key]
        value = float(var.get())
        snapped = round(value / resolution) * resolution
        if abs(snapped - value) > resolution * 0.2:
            var.set(snapped)
            return
        self.refresh_labels()

    def settings(self) -> pipeline.VoiceSettings:
        return pipeline.VoiceSettings(
            formant_ratio=float(self.formant.get()),
            semitones=float(self.pitch.get()),
            pause_count=int(round(self.count.get())),
            pause_ms=int(round(self.length.get())),
            seed=int(round(self.seed.get())),
            denoise=float(self.denoise.get()),
            noise=float(self.noise.get()),
            jitter_cents=float(self.jitter.get()),
            jitter_probability=float(self.chance.get()),
            smoothness=float(self.smooth.get()),
        )

    def refresh_labels(self) -> None:
        if len(self.value_labels) < len(SLIDERS):
            return
        settings = self.settings()
        self.value_labels["formant"].configure(text=f"{settings.formant_ratio:.2f}")
        sign = "+" if settings.semitones > 0 else ""
        self.value_labels["pitch"].configure(text=f"{sign}{settings.semitones:.0f} 半音")
        self.value_labels["smooth"].configure(text=f"{settings.smoothness:.0f}")
        self.value_labels["denoise"].configure(text=f"{settings.denoise:.0f}")
        self.value_labels["noise"].configure(text=f"{settings.noise:.0f}")
        self.value_labels["jitter"].configure(text=f"±{settings.jitter_cents:.0f} 音分")
        self.value_labels["chance"].configure(text=f"{settings.jitter_probability:.0f}%")
        self.value_labels["count"].configure(text=f"{settings.pause_count} 次")
        self.value_labels["length"].configure(text=f"{settings.pause_ms} 毫秒")
        self.value_labels["seed"].configure(text=str(settings.seed))
        self.summary.configure(text=pipeline.describe(settings))

    def _idle_status(self) -> str:
        path = Path(self.media.get())
        if path.exists() and pipeline.stems_ready(path):
            return "人声已缓存。拖动滑块后可以预览或导出，不必重新分离。"
        return "先选择文件并分离人声。之后拖动滑块，预览约 20 秒，或导出整首 MP3。"

    def _browse(self) -> None:
        if self.busy:
            return
        chosen = filedialog.askopenfilename(
            title="选择视频或音频",
            initialdir=str(pipeline.APP_DIR),
            filetypes=[("媒体", "*.mp4 *.mkv *.mov *.wav *.mp3 *.flac *.m4a"), ("全部", "*.*")],
        )
        if chosen:
            self.media.set(chosen)
            self.status.configure(text=self._idle_status())

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.separate_btn, self.preview_btn, self.export_btn):
            button.configure(state=state)

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    frac, text = payload
                    self.progress["value"] = frac
                    self.status.configure(text=text)
                elif kind == "done":
                    self._set_busy(False)
                    self.progress["value"] = 1
                    self.status.configure(text=payload)
                elif kind == "play":
                    self._play(payload)
                elif kind == "error":
                    self._set_busy(False)
                    self.status.configure(text="处理没有完成")
                    messagebox.showerror("人声调整", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _start(self, target) -> None:
        if self.busy:
            return
        path = Path(self.media.get())
        if not path.exists():
            messagebox.showwarning("人声调整", "请先选择一个存在的视频或音频文件。")
            return
        self._stop()
        self._set_busy(True)
        self.progress["value"] = 0
        threading.Thread(target=target, args=(path, self.settings()), daemon=True).start()

    def _separate(self) -> None:
        def work(path: Path, _settings: pipeline.VoiceSettings) -> None:
            try:
                pipeline.prepare_stems(path, lambda frac, text: self.events.put(("progress", (frac, text))))
                self.events.put(("done", "人声分离完成。可以拖动滑块，然后预览或导出。"))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(work)

    def _preview(self) -> None:
        def work(path: Path, settings: pipeline.VoiceSettings) -> None:
            try:
                stems = pipeline.prepare_stems(path, lambda frac, text: self.events.put(("progress", (frac, text))))
                start, end = pipeline.preview_range(stems, settings, 20)
                self.events.put(("progress", (0.05, "正在按当前滑块生成 20 秒预览…")))
                result = pipeline.render(
                    stems,
                    settings,
                    start,
                    end,
                    lambda frac, text: self.events.put(("progress", (0.05 + 0.9 * frac, text))),
                )
                dest = pipeline.cache_folder(path) / "preview.wav"
                pipeline.write_wav(dest, result.audio)
                begin = start / pipeline.SR
                self.events.put(("play", dest))
                self.events.put(
                    (
                        "done",
                        f"正在播放 {begin:.1f} 秒附近的 20 秒。这段里有 {result.pause_count} 处停顿。",
                    )
                )
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(work)

    def _export(self) -> None:
        path = Path(self.media.get())
        if not path.exists():
            messagebox.showwarning("人声调整", "请先选择一个存在的视频或音频文件。")
            return
        dest = filedialog.asksaveasfilename(
            title="导出 MP3",
            initialdir=str(pipeline.APP_DIR),
            initialfile="music_调整.mp3",
            defaultextension=".mp3",
            filetypes=[("MP3", "*.mp3")],
        )
        if not dest:
            return
        output = Path(dest)

        def work(media: Path, settings: pipeline.VoiceSettings) -> None:
            try:
                stems = pipeline.prepare_stems(media, lambda frac, text: self.events.put(("progress", (frac * 0.55, text))))
                self.events.put(("progress", (0.55, "正在按当前滑块处理整首…")))
                result = pipeline.render(
                    stems,
                    settings,
                    progress=lambda frac, text: self.events.put(("progress", (0.55 + 0.4 * frac, text))),
                )
                self.events.put(("progress", (0.96, "正在导出 MP3…")))
                pipeline.export_mp3(result.audio, output)
                self.events.put(("done", f"已导出 {output.name}，其中放入 {result.pause_count} 处随机停顿。"))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(work)

    def _play(self, wav_path: Path) -> None:
        self._stop()
        try:
            pipeline.play_wav(wav_path)
        except Exception as exc:
            messagebox.showerror("人声调整", str(exc))

    def _stop(self) -> None:
        pipeline.stop_playback()

    def _close(self) -> None:
        self._stop()
        self.root.destroy()


def main() -> None:
    _enable_dpi()
    root = tk.Tk()
    VoiceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
