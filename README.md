# Face Timelapse Toolkit

Small scripts for turning a sequence of face photos into a stabilized timelapse video.

The workflow is:

1. Sort source photos by EXIF capture time.
2. Align each face using MediaPipe Face Mesh eye landmarks.
3. Generate a clean cover image from stable frames.
4. Render an MP4 timelapse with ffmpeg.

No personal photos or generated videos are included in this repository.

## Requirements

- Python 3.10+
- ffmpeg available on `PATH`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Directory Layout

The scripts use these folders by default:

```text
data/raw/       input photos
data/aligned/   aligned frames
output/         cover images and videos
```

These folders are ignored by Git so private photos and videos are not committed by accident.

## Usage

Sort images by capture time into a numbered sequence:

```bash
python scripts/sort_by_taken_time.py data/raw --output data/sorted
```

Align faces:

```bash
python scripts/align_faces.py data/sorted data/aligned --base-start 251 --base-end 273
```

Create a cover image:

```bash
python scripts/make_cover.py data/aligned output/cover.jpg --mode sharp
```

Render a video:

```bash
python scripts/render_video.py data/aligned output/timelapse.mp4 --cover output/cover.jpg --fps 20
```

## Notes

- The base frame range should point to a stable, well-framed section of the sequence.
- If face detection fails often, remove very blurry or profile-view photos before alignment.
- ffmpeg is used only for final video encoding; image processing happens in Python.

## License

MIT
