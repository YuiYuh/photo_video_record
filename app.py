from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import exifread
except Exception:
    exifread = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
mp_face = mp.solutions.face_mesh


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


def bundled_ffmpeg() -> str | None:
    candidates = [
        resource_path("ffmpeg.exe"),
        resource_path("bin/ffmpeg.exe"),
        Path("F:/tools/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe"),
        Path("D:/anaconda/envs/face_align/Library/bin/ffmpeg.exe"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return shutil.which("ffmpeg")


def parse_exif_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def get_taken_time(path: Path) -> datetime | None:
    if exifread and path.suffix.lower() in {".jpg", ".jpeg"}:
        try:
            with path.open("rb") as file:
                tags = exifread.process_file(file, details=False)
            for key in ("EXIF DateTimeOriginal", "Image DateTime"):
                if key in tags:
                    parsed = parse_exif_datetime(str(tags[key]))
                    if parsed:
                        return parsed
        except Exception:
            pass

    try:
        image = Image.open(path)
        exif = getattr(image, "_getexif", lambda: None)()
        if exif:
            for tag in (36867, 306):
                if tag in exif:
                    parsed = parse_exif_datetime(str(exif[tag]))
                    if parsed:
                        return parsed
    except Exception:
        pass

    return None


def write_jpeg(source: Path, destination: Path) -> None:
    image = Image.open(source)
    image = image.convert("RGB")
    image.save(destination, quality=95)


def eye_info(image: np.ndarray, face_mesh: object) -> tuple[np.ndarray, float, float] | None:
    height, width = image.shape[:2]
    result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    left = np.array([landmarks[33].x * width, landmarks[33].y * height], dtype=np.float32)
    right = np.array([landmarks[263].x * width, landmarks[263].y * height], dtype=np.float32)
    center = (left + right) / 2.0
    distance = float(np.linalg.norm(right - left))
    angle = math.degrees(math.atan2(float(right[1] - left[1]), float(right[0] - left[0])))
    return center, distance, angle


def compute_base(input_dir: Path, start: int, end: int, face_mesh: object) -> tuple[float, np.ndarray]:
    distances: list[float] = []
    centers: list[np.ndarray] = []

    for index in range(start, end + 1):
        image = cv2.imread(str(input_dir / f"{index:04d}.jpg"))
        if image is None:
            continue
        info = eye_info(image, face_mesh)
        if info:
            center, distance, _ = info
            centers.append(center)
            distances.append(distance)

    if not distances:
        raise RuntimeError("The selected base frame range has no detectable faces.")

    return float(np.median(np.array(distances))), np.mean(np.stack(centers), axis=0)


def align_image(image: np.ndarray, face_mesh: object, base_distance: float, base_center: np.ndarray) -> np.ndarray:
    info = eye_info(image, face_mesh)
    if not info:
        return image

    center, distance, angle = info
    height, width = image.shape[:2]
    scale = base_distance / max(distance, 1e-6)
    scale = max(0.85, min(1.15, scale))
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, scale)
    shift = base_center - center
    matrix[0, 2] += float(shift[0])
    matrix[1, 2] += float(shift[1])
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def black_ratio(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((gray <= 8).sum()) / float(gray.size)


def robust_mask(values: list[float], k: float) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    median = np.median(array)
    mad = np.median(np.abs(array - median)) + 1e-9
    return (array >= median - k * mad) & (array <= median + k * mad)


def make_cover(aligned_dir: Path, output: Path, mode: str, top_n: int, log: callable) -> None:
    files = sorted(aligned_dir.glob("*.jpg"))
    records: list[dict[str, object]] = []

    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                continue
            metrics = eye_info(image, face_mesh)
            if not metrics:
                continue
            center, distance, angle_degrees = metrics
            records.append(
                {
                    "path": path,
                    "center": center,
                    "distance": distance,
                    "angle": abs(math.radians(angle_degrees)),
                    "black": black_ratio(image),
                }
            )

    if len(records) < 5:
        raise RuntimeError("Too few detectable faces to build a cover image.")

    centers = np.stack([record["center"] for record in records], axis=0)
    base_center = centers.mean(axis=0)
    distances = [float(record["distance"]) for record in records]
    angles = [float(record["angle"]) for record in records]
    positions = [float(np.linalg.norm(record["center"] - base_center)) for record in records]
    blacks = [float(record["black"]) for record in records]
    mask = (
        (np.array(blacks) <= 0.18)
        & robust_mask(angles, 3.0)
        & robust_mask(distances, 3.0)
        & robust_mask(positions, 3.0)
    )
    inliers = [records[index] for index in range(len(records)) if mask[index]]
    if len(inliers) < 5:
        raise RuntimeError("Too few stable frames for cover image.")

    selected = inliers
    if mode == "sharp":
        base_distance = float(np.median(np.array([record["distance"] for record in inliers], dtype=np.float64)))
        scored: list[tuple[float, dict[str, object]]] = []
        for record in inliers:
            center = record["center"]
            distance = float(record["distance"])
            angle = float(record["angle"])
            black = float(record["black"])
            position = float(np.linalg.norm(center - base_center))
            distance_error = abs(distance - base_distance) / (base_distance + 1e-9)
            score = (angle * 2.0) + (position / 120.0) + (distance_error * 2.0) + (black * 1.5)
            scored.append((score, record))
        scored.sort(key=lambda item: item[0])
        selected = [record for _, record in scored[: min(top_n, len(scored))]]

    accumulator = None
    for record in selected:
        image = cv2.imread(str(record["path"])).astype(np.float32)
        accumulator = image if accumulator is None else accumulator + image

    cover = np.clip(accumulator / float(len(selected)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(output), cover, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    log(f"Cover built from {len(selected)} stable frames")


class TimelapseApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Photo Video Record")
        self.root.geometry("820x620")
        self.root.minsize(780, 560)

        self.files: list[Path] = []
        self.output_path = StringVar(value=str(Path.home() / "Desktop" / "timelapse.mp4"))
        self.ffmpeg_path = StringVar(value=bundled_ffmpeg() or "")
        self.fps = IntVar(value=20)
        self.base_start = IntVar(value=0)
        self.base_end = IntVar(value=20)
        self.cover_mode = StringVar(value="sharp")
        self.top_n = IntVar(value=12)
        self.keep_work = BooleanVar(value=False)
        self.progress = IntVar(value=0)
        self.status = StringVar(value="Ready")
        self.running = False

        self.build_ui()

    def build_ui(self) -> None:
        pad = {"padx": 14, "pady": 8}
        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        top = ttk.Frame(frame)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="Add photos", command=self.select_photos).pack(side="left")
        ttk.Button(top, text="Clear", command=self.clear_photos).pack(side="left", padx=(8, 0))
        self.count_label = ttk.Label(top, text="0 photos selected")
        self.count_label.pack(side="left", padx=14)

        output = ttk.LabelFrame(frame, text="Output")
        output.pack(fill="x", **pad)
        ttk.Entry(output, textvariable=self.output_path).pack(side="left", fill="x", expand=True, padx=10, pady=10)
        ttk.Button(output, text="Choose", command=self.select_output).pack(side="left", padx=(0, 10))

        options = ttk.LabelFrame(frame, text="Options")
        options.pack(fill="x", **pad)
        for column in range(8):
            options.columnconfigure(column, weight=1)

        ttk.Label(options, text="FPS").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        ttk.Combobox(options, textvariable=self.fps, values=[6, 12, 16, 20, 24, 25, 30, 60], width=8).grid(
            row=0, column=1, sticky="w", padx=10, pady=8
        )
        ttk.Label(options, text="Base start").grid(row=0, column=2, sticky="w", padx=10, pady=8)
        ttk.Spinbox(options, from_=0, to=9999, textvariable=self.base_start, width=8).grid(
            row=0, column=3, sticky="w", padx=10, pady=8
        )
        ttk.Label(options, text="Base end").grid(row=0, column=4, sticky="w", padx=10, pady=8)
        ttk.Spinbox(options, from_=0, to=9999, textvariable=self.base_end, width=8).grid(
            row=0, column=5, sticky="w", padx=10, pady=8
        )
        ttk.Label(options, text="Cover").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        ttk.Combobox(options, textvariable=self.cover_mode, values=["sharp", "all"], width=8).grid(
            row=1, column=1, sticky="w", padx=10, pady=8
        )
        ttk.Label(options, text="Top frames").grid(row=1, column=2, sticky="w", padx=10, pady=8)
        ttk.Spinbox(options, from_=5, to=100, textvariable=self.top_n, width=8).grid(
            row=1, column=3, sticky="w", padx=10, pady=8
        )
        ttk.Checkbutton(options, text="Keep intermediate frames", variable=self.keep_work).grid(
            row=1, column=4, columnspan=3, sticky="w", padx=10, pady=8
        )

        ffmpeg = ttk.LabelFrame(frame, text="ffmpeg")
        ffmpeg.pack(fill="x", **pad)
        ttk.Entry(ffmpeg, textvariable=self.ffmpeg_path).pack(side="left", fill="x", expand=True, padx=10, pady=10)
        ttk.Button(ffmpeg, text="Choose", command=self.select_ffmpeg).pack(side="left", padx=(0, 10))

        actions = ttk.Frame(frame)
        actions.pack(fill="x", **pad)
        self.start_button = ttk.Button(actions, text="Generate video", command=self.start)
        self.start_button.pack(side="left")
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=14)

        progress = ttk.Progressbar(frame, maximum=100, variable=self.progress)
        progress.pack(fill="x", **pad)

        log_frame = ttk.LabelFrame(frame, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_box = ttk.Treeview(log_frame, columns=("message",), show="headings", height=12)
        self.log_box.heading("message", text="Message")
        self.log_box.column("message", width=760, anchor="w")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)

    def log(self, message: str) -> None:
        def write() -> None:
            self.log_box.insert("", "end", values=(message,))
            self.log_box.yview_moveto(1)
            self.status.set(message)

        self.root.after(0, write)

    def set_progress(self, value: int) -> None:
        self.root.after(0, lambda: self.progress.set(value))

    def select_photos(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Select photos",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.heic *.webp"), ("All files", "*.*")],
        )
        if selected:
            paths = [Path(item) for item in selected if Path(item).suffix.lower() in IMAGE_EXTENSIONS]
            self.files = sorted(set(self.files + paths), key=lambda item: str(item).lower())
            self.count_label.configure(text=f"{len(self.files)} photos selected")

    def clear_photos(self) -> None:
        self.files = []
        self.count_label.configure(text="0 photos selected")

    def select_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if selected:
            self.output_path.set(selected)

    def select_ffmpeg(self) -> None:
        selected = filedialog.askopenfilename(title="Select ffmpeg.exe", filetypes=[("ffmpeg", "ffmpeg.exe"), ("All", "*.*")])
        if selected:
            self.ffmpeg_path.set(selected)

    def validate(self) -> tuple[Path, str]:
        if self.running:
            raise ValueError("A job is already running.")
        if not self.files:
            raise ValueError("Select photos first.")
        output = Path(self.output_path.get())
        if output.suffix.lower() != ".mp4":
            raise ValueError("Output must be an MP4 file.")
        ffmpeg = self.ffmpeg_path.get().strip()
        if not ffmpeg or not Path(ffmpeg).exists():
            raise ValueError("Choose a valid ffmpeg.exe.")
        if self.base_end.get() < self.base_start.get():
            raise ValueError("Base end must be greater than or equal to base start.")
        if self.base_end.get() >= len(self.files):
            raise ValueError("Base frame range is outside the selected photos.")
        return output, ffmpeg

    def start(self) -> None:
        try:
            output, ffmpeg = self.validate()
        except ValueError as error:
            messagebox.showerror("Cannot start", str(error))
            return

        self.running = True
        self.start_button.configure(state="disabled")
        self.progress.set(0)
        thread = threading.Thread(target=self.run_pipeline, args=(output, ffmpeg), daemon=True)
        thread.start()

    def finish(self) -> None:
        self.running = False
        self.root.after(0, lambda: self.start_button.configure(state="normal"))

    def run_pipeline(self, output: Path, ffmpeg: str) -> None:
        work_root = output.parent / f"{output.stem}_work" if self.keep_work.get() else Path(tempfile.mkdtemp())
        sorted_dir = work_root / "sorted"
        aligned_dir = work_root / "aligned"
        cover_path = work_root / "cover.jpg"
        sorted_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.log(f"Sorting {len(self.files)} photos")
            ordered = sorted(self.files, key=lambda path: (get_taken_time(path) or datetime.max, path.name.lower()))
            for index, source in enumerate(ordered):
                write_jpeg(source, sorted_dir / f"{index:04d}.jpg")
                self.set_progress(int((index + 1) / len(ordered) * 20))

            self.log("Aligning faces")
            files = sorted(sorted_dir.glob("*.jpg"))
            with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
                base_distance, base_center = compute_base(sorted_dir, self.base_start.get(), self.base_end.get(), face_mesh)
                self.log(f"Base eye distance: {base_distance:.1f}px")
                for index, path in enumerate(files):
                    image = cv2.imread(str(path))
                    if image is None:
                        continue
                    aligned = align_image(image, face_mesh, base_distance, base_center)
                    cv2.imwrite(str(aligned_dir / f"{index:04d}.jpg"), aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    self.set_progress(20 + int((index + 1) / len(files) * 50))

            self.log("Building cover image")
            make_cover(aligned_dir, cover_path, self.cover_mode.get(), self.top_n.get(), self.log)
            self.set_progress(80)

            self.log("Rendering MP4")
            command = [
                ffmpeg,
                "-y",
                "-loop",
                "1",
                "-t",
                "1",
                "-i",
                str(cover_path),
                "-framerate",
                str(self.fps.get()),
                "-i",
                str(aligned_dir / "%04d.jpg"),
                "-filter_complex",
                (
                    f"[0:v]fps={self.fps.get()},setsar=1,format=yuv420p[v0];"
                    f"[1:v]fps={self.fps.get()},setsar=1,format=yuv420p[v1];"
                    "[v0][v1]concat=n=2:v=1:a=0[v]"
                ),
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                str(output),
            ]
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.set_progress(100)
            self.log(f"Done: {output}")
            messagebox.showinfo("Finished", f"Video saved:\n{output}")
        except Exception as error:
            self.log(f"Failed: {error}")
            messagebox.showerror("Generation failed", str(error))
        finally:
            if not self.keep_work.get():
                shutil.rmtree(work_root, ignore_errors=True)
            self.finish()


def main() -> None:
    root = Tk()
    ttk.Style().theme_use("clam")
    TimelapseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
