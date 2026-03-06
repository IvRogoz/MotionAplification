import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ampEul import eulerian_video_magnification
from ampLive import live_eulerian_magnification

try:
    from cv2_enumerate_cameras import enumerate_cameras
except Exception:
    enumerate_cameras = None


class MotionAmplificationUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Motion Amplification Studio")
        self.geometry("1220x760")
        self.minsize(1100, 700)
        self.configure(bg="#e7efe8")

        self._build_theme()
        self._build_layout()
        self._refresh_cameras()

    def _build_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background="#e7efe8", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff"), ("!selected", "#d6e1dc")])
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Header.TLabel", background="#ffffff", foreground="#11261f", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", background="#ffffff", foreground="#39524a", font=("Segoe UI", 10))
        style.configure("Field.TLabel", background="#ffffff", foreground="#11261f", font=("Segoe UI", 9, "bold"))
        style.configure("Value.TLabel", background="#ffffff", foreground="#243a33", font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))

    def _build_layout(self):
        wrapper = ttk.Frame(self, style="Card.TFrame", padding=16)
        wrapper.pack(fill="both", expand=True, padx=16, pady=16)

        title = ttk.Label(wrapper, text="Motion Amplification Studio", style="Header.TLabel")
        title.pack(anchor="w")
        subtitle = ttk.Label(
            wrapper,
            text="Desktop UI for Eulerian video magnification and live camera mode.",
            style="Sub.TLabel",
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        tabs = ttk.Notebook(wrapper)
        tabs.pack(fill="both", expand=True)

        self.video_tab = ttk.Frame(tabs, style="Card.TFrame", padding=12)
        self.live_tab = ttk.Frame(tabs, style="Card.TFrame", padding=12)
        tabs.add(self.video_tab, text="Video Processing")
        tabs.add(self.live_tab, text="Live Camera")

        self._build_video_tab()
        self._build_live_tab()

    def _build_video_tab(self):
        left = ttk.Frame(self.video_tab, style="Card.TFrame")
        right = ttk.Frame(self.video_tab, style="Card.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="right", fill="y")
        right_col1 = ttk.Frame(right, style="Card.TFrame")
        right_col2 = ttk.Frame(right, style="Card.TFrame")
        right_col1.pack(side="left", anchor="n", padx=(0, 14))
        right_col2.pack(side="left", anchor="n")

        self.input_video_var = tk.StringVar()
        self.output_video_var = tk.StringVar()
        self.video_status_var = tk.StringVar(value="Ready.")

        ttk.Label(left, text="Input Video", style="Field.TLabel").pack(anchor="w")
        in_row = ttk.Frame(left, style="Card.TFrame")
        in_row.pack(fill="x", pady=(4, 10))
        ttk.Entry(in_row, textvariable=self.input_video_var).pack(side="left", fill="x", expand=True)
        ttk.Button(in_row, text="Browse", command=self._pick_input_video).pack(side="left", padx=(8, 0))

        ttk.Label(left, text="Output Video", style="Field.TLabel").pack(anchor="w")
        out_row = ttk.Frame(left, style="Card.TFrame")
        out_row.pack(fill="x", pady=(4, 10))
        ttk.Entry(out_row, textvariable=self.output_video_var).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="Save As", command=self._pick_output_video).pack(side="left", padx=(8, 0))

        ttk.Button(left, text="Process Video", style="Primary.TButton", command=self._start_video_processing).pack(
            anchor="w", pady=(8, 6)
        )
        ttk.Label(left, textvariable=self.video_status_var, style="Value.TLabel", wraplength=560).pack(anchor="w")

        self.amp_v = tk.DoubleVar(value=20)
        self.low_v = tk.DoubleVar(value=0.4)
        self.high_v = tk.DoubleVar(value=3.0)
        self.fps_v = tk.DoubleVar(value=0.0)
        self.levels_v = tk.IntVar(value=4)
        self.chrom_v = tk.DoubleVar(value=0.1)
        self.chunk_v = tk.IntVar(value=60)
        self.overlap_v = tk.IntVar(value=30)

        self._add_labeled_spin(right_col1, "Amplification", self.amp_v, 1, 80, 1)
        self._add_labeled_spin(right_col1, "Low Cutoff (Hz)", self.low_v, 0.05, 5.0, 0.05)
        self._add_labeled_spin(right_col1, "High Cutoff (Hz)", self.high_v, 0.1, 10.0, 0.05)
        self._add_labeled_spin(right_col1, "FPS Override (0 = auto)", self.fps_v, 0.0, 240.0, 1.0)
        self._add_labeled_spin(right_col2, "Pyramid Levels", self.levels_v, 1, 8, 1)
        self._add_labeled_spin(right_col2, "Chrom Attenuation", self.chrom_v, 0.0, 1.0, 0.05)
        self._add_labeled_spin(right_col2, "Chunk Size", self.chunk_v, 20, 240, 5)
        self._add_labeled_spin(right_col2, "Overlap", self.overlap_v, 0, 180, 5)

    def _build_live_tab(self):
        left = ttk.Frame(self.live_tab, style="Card.TFrame")
        right = ttk.Frame(self.live_tab, style="Card.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="right", fill="y")
        right_col1 = ttk.Frame(right, style="Card.TFrame")
        right_col2 = ttk.Frame(right, style="Card.TFrame")
        right_col1.pack(side="left", anchor="n", padx=(0, 14))
        right_col2.pack(side="left", anchor="n")

        self.live_status_var = tk.StringVar(value="Idle.")
        self.record_live_var = tk.BooleanVar(value=False)
        self.live_output_path_var = tk.StringVar()
        self.mode_options = []
        self.mode_choice_var = tk.StringVar(value="No probed modes yet")
        ttk.Label(
            left,
            text="Press Start to open the OpenCV camera window. Press 'q' there to stop live mode.",
            style="Value.TLabel",
            wraplength=560,
        ).pack(anchor="w", pady=(0, 10))
        ttk.Checkbutton(
            left,
            text="Record processed live output to file",
            variable=self.record_live_var,
        ).pack(anchor="w", pady=(0, 8))
        live_out_row = ttk.Frame(left, style="Card.TFrame")
        live_out_row.pack(fill="x", pady=(0, 8))
        ttk.Entry(live_out_row, textvariable=self.live_output_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(live_out_row, text="Save As", command=self._pick_live_output_video).pack(side="left", padx=(8, 0))
        probe_row = ttk.Frame(left, style="Card.TFrame")
        probe_row.pack(fill="x", pady=(0, 8))
        self.mode_combo = ttk.Combobox(
            probe_row,
            textvariable=self.mode_choice_var,
            state="readonly",
            width=48,
            values=["No probed modes yet"],
        )
        self.mode_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(probe_row, text="Probe Modes", command=self._start_probe_modes).pack(side="left", padx=(8, 0))
        ttk.Button(left, text="Use Selected Mode", command=self._apply_selected_mode).pack(anchor="w", pady=(0, 8))
        ttk.Button(left, text="Start Live Camera", style="Primary.TButton", command=self._start_live_mode).pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Label(left, textvariable=self.live_status_var, style="Value.TLabel", wraplength=560).pack(anchor="w")

        self.cam_idx = tk.IntVar(value=0)
        self.camera_options = []
        self.camera_choice_var = tk.StringVar(value="No cameras detected")
        self.amp_l = tk.DoubleVar(value=20)
        self.low_l = tk.DoubleVar(value=0.4)
        self.high_l = tk.DoubleVar(value=3.0)
        self.levels_l = tk.IntVar(value=4)
        self.chrom_l = tk.DoubleVar(value=0.2)
        self.color_l = tk.DoubleVar(value=1.5)
        self.chunk_l = tk.IntVar(value=60)
        self.overlap_l = tk.IntVar(value=30)
        self.fps_l = tk.DoubleVar(value=30.0)
        self.capture_w_l = tk.IntVar(value=0)
        self.capture_h_l = tk.IntVar(value=0)
        self.save_w_l = tk.IntVar(value=0)
        self.save_h_l = tk.IntVar(value=0)

        ttk.Label(right_col1, text="Camera Device", style="Field.TLabel").pack(anchor="w", pady=(0, 2))
        cam_row = ttk.Frame(right_col1, style="Card.TFrame")
        cam_row.pack(anchor="w", fill="x", pady=(0, 10))
        self.camera_combo = ttk.Combobox(
            cam_row,
            textvariable=self.camera_choice_var,
            state="readonly",
            width=34,
            values=["Loading cameras..."],
        )
        self.camera_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(cam_row, text="Refresh", command=self._refresh_cameras).pack(side="left", padx=(8, 0))

        self._add_labeled_spin(right_col1, "Capture Width (0 = camera default)", self.capture_w_l, 0, 3840, 16)
        self._add_labeled_spin(right_col1, "Capture Height (0 = camera default)", self.capture_h_l, 0, 2160, 16)
        self._add_labeled_spin(right_col1, "Amplification", self.amp_l, 1, 80, 1)
        self._add_labeled_spin(right_col1, "Low Cutoff (Hz)", self.low_l, 0.05, 5.0, 0.05)
        self._add_labeled_spin(right_col1, "High Cutoff (Hz)", self.high_l, 0.1, 10.0, 0.05)
        self._add_labeled_spin(right_col1, "Pyramid Levels", self.levels_l, 1, 8, 1)
        self._add_labeled_spin(right_col2, "Chrom Attenuation", self.chrom_l, 0.0, 1.0, 0.05)
        self._add_labeled_spin(right_col2, "Color Amplification", self.color_l, 0.5, 3.0, 0.1)
        self._add_labeled_spin(right_col2, "Chunk Size", self.chunk_l, 20, 240, 5)
        self._add_labeled_spin(right_col2, "Overlap", self.overlap_l, 0, 180, 5)
        self._add_labeled_spin(right_col2, "Default FPS", self.fps_l, 1.0, 240.0, 1.0)
        self._add_labeled_spin(right_col2, "Save Width (0 = match capture)", self.save_w_l, 0, 3840, 16)
        self._add_labeled_spin(right_col2, "Save Height (0 = match capture)", self.save_h_l, 0, 2160, 16)

    def _add_labeled_spin(self, parent, label, variable, from_, to, increment):
        ttk.Label(parent, text=label, style="Field.TLabel").pack(anchor="w", pady=(0, 2))
        spin = ttk.Spinbox(parent, textvariable=variable, from_=from_, to=to, increment=increment, width=16)
        spin.pack(anchor="w", pady=(0, 7))

    def _pick_input_video(self):
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")],
        )
        if path:
            self.input_video_var.set(path)

    def _pick_output_video(self):
        path = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All files", "*.*")],
        )
        if path:
            self.output_video_var.set(path)

    def _pick_live_output_video(self):
        path = filedialog.asksaveasfilename(
            title="Save live output video as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All files", "*.*")],
        )
        if path:
            self.live_output_path_var.set(path)

    def _refresh_cameras(self):
        options = []

        if enumerate_cameras is not None:
            try:
                for cam in enumerate_cameras():
                    cam_index = int(getattr(cam, "index", 0))
                    cam_name = str(getattr(cam, "name", f"Camera {cam_index}"))
                    options.append((cam_index, f"{cam_name} (index {cam_index})"))
            except Exception:
                options = []

        if not options:
            for cam_index in range(10):
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    options.append((cam_index, f"Camera {cam_index}"))
                    cap.release()
                else:
                    cap.release()

        if not options:
            self.camera_options = [(0, "Camera 0")]
            self.camera_combo["values"] = ["Camera 0"]
            self.camera_choice_var.set("Camera 0")
            self.cam_idx.set(0)
            return

        self.camera_options = options
        labels = [label for _, label in options]
        self.camera_combo["values"] = labels
        self.camera_choice_var.set(labels[0])
        self.cam_idx.set(options[0][0])

    def _selected_camera_index(self):
        selected = self.camera_choice_var.get().strip()
        for cam_index, label in self.camera_options:
            if label == selected:
                return cam_index
        return 0

    def _start_probe_modes(self):
        self.live_status_var.set("Probing camera modes... this can take a few seconds.")
        threading.Thread(
            target=self._probe_camera_modes,
            daemon=True,
        ).start()

    def _probe_camera_modes(self):
        cam_index = self._selected_camera_index()
        candidate_resolutions = [
            (640, 480),
            (800, 600),
            (960, 540),
            (1024, 576),
            (1024, 768),
            (1280, 720),
            (1280, 800),
            (1280, 960),
            (1600, 900),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
        ]
        candidate_fps = [24, 30, 60]

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            self.after(0, lambda: self.live_status_var.set("Failed to open selected camera for probing."))
            return

        found = []
        seen = set()
        try:
            for width, height in candidate_resolutions:
                for fps in candidate_fps:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    cap.read()
                    cap.read()

                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if actual_w <= 0 or actual_h <= 0:
                        continue

                    if abs(actual_w - width) > 24 or abs(actual_h - height) > 24:
                        continue

                    rounded_fps = int(round(actual_fps)) if actual_fps and actual_fps > 1 else fps
                    key = (actual_w, actual_h, rounded_fps)
                    if key in seen:
                        continue
                    seen.add(key)
                    label = f"{actual_w}x{actual_h} @ {rounded_fps} fps"
                    found.append((actual_w, actual_h, rounded_fps, label))
        finally:
            cap.release()

        if not found:
            found = [(0, 0, 0, "No standard modes detected")]

        def update_ui():
            self.mode_options = found
            labels = [item[3] for item in found]
            self.mode_combo["values"] = labels
            self.mode_choice_var.set(labels[0])
            if found[0][0] > 0:
                self.live_status_var.set(f"Found {len(found)} mode(s). Choose one and click 'Use Selected Mode'.")
            else:
                self.live_status_var.set("No standard modes were detected. You can still set capture values manually.")

        self.after(0, update_ui)

    def _apply_selected_mode(self):
        selected = self.mode_choice_var.get().strip()
        for width, height, fps, label in self.mode_options:
            if label == selected and width > 0 and height > 0:
                self.capture_w_l.set(width)
                self.capture_h_l.set(height)
                self.save_w_l.set(width)
                self.save_h_l.set(height)
                if fps > 0:
                    self.fps_l.set(float(fps))
                self.live_status_var.set(f"Applied {label} to capture/save/default FPS.")
                return
        self.live_status_var.set("No valid probed mode selected.")

    def _start_video_processing(self):
        input_path = self.input_video_var.get().strip()
        output_path = self.output_video_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Select an input video.")
            return
        if not output_path:
            messagebox.showerror("Missing output", "Select an output path.")
            return
        if self.high_v.get() <= self.low_v.get():
            messagebox.showerror("Invalid cutoff", "High cutoff must be greater than low cutoff.")
            return
        if self.chunk_v.get() <= self.overlap_v.get():
            messagebox.showerror("Invalid chunking", "Chunk size must be greater than overlap.")
            return

        self.video_status_var.set("Processing video...")
        threading.Thread(
            target=self._run_video_processing,
            daemon=True,
        ).start()

    def _run_video_processing(self):
        started = time.time()
        try:
            fps_override = self.fps_v.get()
            eulerian_video_magnification(
                input_video_path=self.input_video_var.get().strip(),
                output_video_path=self.output_video_var.get().strip(),
                amplification_factor=float(self.amp_v.get()),
                low_cutoff=float(self.low_v.get()),
                high_cutoff=float(self.high_v.get()),
                fps=None if fps_override <= 0 else float(fps_override),
                levels=int(self.levels_v.get()),
                chrom_attenuation=float(self.chrom_v.get()),
                chunk_size=int(self.chunk_v.get()),
                overlap=int(self.overlap_v.get()),
            )
            elapsed = time.time() - started
            self.after(0, lambda: self.video_status_var.set(f"Done in {elapsed:.1f}s."))
        except Exception as exc:
            self.after(0, lambda: self.video_status_var.set(f"Failed: {exc}"))

    def _start_live_mode(self):
        if self.high_l.get() <= self.low_l.get():
            messagebox.showerror("Invalid cutoff", "High cutoff must be greater than low cutoff.")
            return
        if self.chunk_l.get() <= self.overlap_l.get():
            messagebox.showerror("Invalid chunking", "Chunk size must be greater than overlap.")
            return
        if self.capture_w_l.get() < 0 or self.capture_h_l.get() < 0:
            messagebox.showerror("Invalid capture size", "Capture width/height must be 0 or positive.")
            return
        if self.save_w_l.get() < 0 or self.save_h_l.get() < 0:
            messagebox.showerror("Invalid save size", "Save width/height must be 0 or positive.")
            return
        if self.record_live_var.get() and not self.live_output_path_var.get().strip():
            messagebox.showerror("Missing output", "Choose a path for live recording.")
            return

        self.cam_idx.set(self._selected_camera_index())
        self.live_status_var.set("Live mode running. Press 'q' in camera window to stop.")
        threading.Thread(
            target=self._run_live_mode,
            daemon=True,
        ).start()

    def _run_live_mode(self):
        try:
            output_path = self.live_output_path_var.get().strip() if self.record_live_var.get() else None
            recording_result = live_eulerian_magnification(
                camera_index=int(self.cam_idx.get()),
                amplification_factor=float(self.amp_l.get()),
                low_cutoff=float(self.low_l.get()),
                high_cutoff=float(self.high_l.get()),
                levels=int(self.levels_l.get()),
                chrom_attenuation=float(self.chrom_l.get()),
                color_amplification=float(self.color_l.get()),
                chunk_size=int(self.chunk_l.get()),
                overlap=int(self.overlap_l.get()),
                default_fps=float(self.fps_l.get()),
                capture_width=int(self.capture_w_l.get()),
                capture_height=int(self.capture_h_l.get()),
                save_width=int(self.save_w_l.get()),
                save_height=int(self.save_h_l.get()),
                record_output_path=output_path,
            )
            if recording_result:
                self.after(0, lambda: self.live_status_var.set(f"Live mode closed. Recording saved: {recording_result}"))
            else:
                self.after(0, lambda: self.live_status_var.set("Live mode closed."))
        except Exception as exc:
            self.after(0, lambda: self.live_status_var.set(f"Failed: {exc}"))


if __name__ == "__main__":
    app = MotionAmplificationUI()
    app.mainloop()
