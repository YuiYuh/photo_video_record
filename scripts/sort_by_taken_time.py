from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import exifread
from PIL import Image
from tqdm import tqdm

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def parse_exif_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def taken_time_from_name(path: Path) -> datetime | None:
    match = re.search(r"(20\d{2})[-_:]?(0\d|1[0-2])[-_:]?([0-2]\d|3[01])", path.name)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def get_taken_time(path: Path) -> datetime | None:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
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

    return taken_time_from_name(path)


def convert_to_jpeg(source: Path, destination: Path) -> None:
    image = Image.open(source)
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    image.save(destination, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort photos by capture time into a numbered JPEG sequence.")
    parser.add_argument("input", type=Path, help="Folder containing source photos.")
    parser.add_argument("--output", type=Path, default=Path("data/sorted"), help="Folder for sorted frames.")
    parser.add_argument("--copy-originals", action="store_true", help="Copy files without converting to JPEG.")
    args = parser.parse_args()

    files = sorted(path for path in args.input.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No supported images found in {args.input}")

    with_time: list[tuple[Path, datetime]] = []
    without_time: list[Path] = []
    for path in tqdm(files, desc="Reading capture times"):
        captured = get_taken_time(path)
        if captured:
            with_time.append((path, captured))
        else:
            without_time.append(path)

    ordered = [path for path, _ in sorted(with_time, key=lambda item: item[1])] + without_time
    args.output.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(tqdm(ordered, desc="Writing sorted frames")):
        if args.copy_originals:
            destination = args.output / f"{index:04d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
        else:
            destination = args.output / f"{index:04d}.jpg"
            convert_to_jpeg(source, destination)

    print(f"Wrote {len(ordered)} frames to {args.output}")


if __name__ == "__main__":
    main()
