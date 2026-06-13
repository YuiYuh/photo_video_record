from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def ffmpeg_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if not found:
        raise SystemExit("ffmpeg was not found on PATH. Install ffmpeg or pass --ffmpeg.")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Render aligned JPG frames to an MP4 timelapse.")
    parser.add_argument("input", type=Path, help="Folder containing aligned JPG frames named 0000.jpg, 0001.jpg, ...")
    parser.add_argument("output", type=Path, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=20, help="Playback frames per second.")
    parser.add_argument("--cover", type=Path, help="Optional cover image shown for one second before the timelapse.")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality. Lower is higher quality.")
    parser.add_argument("--preset", default="slow", help="ffmpeg x264 preset.")
    parser.add_argument("--ffmpeg", help="Path to ffmpeg executable.")
    args = parser.parse_args()

    if not (args.input / "0000.jpg").exists():
        raise SystemExit(f"Missing first frame: {args.input / '0000.jpg'}")

    encoder = ffmpeg_path(args.ffmpeg)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.cover:
        command = [
            encoder,
            "-y",
            "-loop",
            "1",
            "-t",
            "1",
            "-i",
            str(args.cover),
            "-framerate",
            str(args.fps),
            "-i",
            str(args.input / "%04d.jpg"),
            "-filter_complex",
            (
                f"[0:v]fps={args.fps},setsar=1,format=yuv420p[v0];"
                f"[1:v]fps={args.fps},setsar=1,format=yuv420p[v1];"
                "[v0][v1]concat=n=2:v=1:a=0[v]"
            ),
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-movflags",
            "+faststart",
            str(args.output),
        ]
    else:
        command = [
            encoder,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(args.input / "%04d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ]

    subprocess.run(command, check=True)
    print(f"Wrote video: {args.output}")


if __name__ == "__main__":
    main()
