from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


mp_face = mp.solutions.face_mesh


def eye_metrics(image: np.ndarray, face_mesh: object) -> tuple[np.ndarray, float, float] | None:
    height, width = image.shape[:2]
    result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    left = np.array([landmarks[33].x * width, landmarks[33].y * height], dtype=np.float32)
    right = np.array([landmarks[263].x * width, landmarks[263].y * height], dtype=np.float32)
    center = (left + right) / 2.0
    distance = float(np.linalg.norm(right - left))
    angle = abs(math.atan2(float(right[1] - left[1]), float(right[0] - left[0])))
    return center, distance, angle


def black_ratio(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((gray <= 8).sum()) / float(gray.size)


def robust_mask(values: list[float], k: float) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    median = np.median(array)
    mad = np.median(np.abs(array - median)) + 1e-9
    return (array >= median - k * mad) & (array <= median + k * mad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stable cover image from aligned frames.")
    parser.add_argument("input", type=Path, help="Folder containing aligned JPG frames.")
    parser.add_argument("output", type=Path, help="Output cover image path.")
    parser.add_argument("--mode", choices=["all", "sharp"], default="sharp", help="Blend all stable frames or only top frames.")
    parser.add_argument("--top-n", type=int, default=12, help="Number of top frames for sharp mode.")
    parser.add_argument("--max-black-ratio", type=float, default=0.18, help="Reject frames with too much black border.")
    parser.add_argument("--robust-k", type=float, default=3.0, help="MAD outlier threshold.")
    args = parser.parse_args()

    files = sorted(args.input.glob("*.jpg"))
    if not files:
        raise SystemExit(f"No JPG frames found in {args.input}")

    records: list[dict[str, object]] = []
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        for path in tqdm(files, desc="Scanning frames"):
            image = cv2.imread(str(path))
            if image is None:
                continue
            metrics = eye_metrics(image, face_mesh)
            if not metrics:
                continue
            center, distance, angle = metrics
            records.append({"path": path, "center": center, "distance": distance, "angle": angle, "black": black_ratio(image)})

    if len(records) < 5:
        raise SystemExit("Too few detectable faces to build a cover image.")

    centers = np.stack([record["center"] for record in records], axis=0)
    base_center = centers.mean(axis=0)
    distances = [float(record["distance"]) for record in records]
    angles = [float(record["angle"]) for record in records]
    positions = [float(np.linalg.norm(record["center"] - base_center)) for record in records]
    blacks = [float(record["black"]) for record in records]

    mask = (
        (np.array(blacks) <= args.max_black_ratio)
        & robust_mask(angles, args.robust_k)
        & robust_mask(distances, args.robust_k)
        & robust_mask(positions, args.robust_k)
    )
    inliers = [records[index] for index in range(len(records)) if mask[index]]
    if len(inliers) < 5:
        raise SystemExit("Too few stable frames. Try increasing --robust-k or --max-black-ratio.")

    selected = inliers
    if args.mode == "sharp":
        base_distance = float(np.median(np.array([record["distance"] for record in inliers], dtype=np.float64)))
        scored: list[tuple[float, Path]] = []
        for record in inliers:
            center = record["center"]
            distance = float(record["distance"])
            angle = float(record["angle"])
            black = float(record["black"])
            position = float(np.linalg.norm(center - base_center))
            distance_error = abs(distance - base_distance) / (base_distance + 1e-9)
            score = (angle * 2.0) + (position / 120.0) + (distance_error * 2.0) + (black * 1.5)
            scored.append((score, record["path"]))
        scored.sort(key=lambda item: item[0])
        selected_paths = {path for _, path in scored[: min(args.top_n, len(scored))]}
        selected = [record for record in inliers if record["path"] in selected_paths]

    accumulator = None
    for record in tqdm(selected, desc="Blending cover"):
        image = cv2.imread(str(record["path"])).astype(np.float32)
        accumulator = image if accumulator is None else accumulator + image

    output = np.clip(accumulator / float(len(selected)), 0, 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), output, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    print(f"Wrote cover from {len(selected)} frames: {args.output}")


if __name__ == "__main__":
    main()
