from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


mp_face = mp.solutions.face_mesh


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
        path = input_dir / f"{index:04d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            continue
        info = eye_info(image, face_mesh)
        if info:
            center, distance, _ = info
            centers.append(center)
            distances.append(distance)

    if not distances:
        raise RuntimeError("No faces detected in base frame range.")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Align face photos using eye landmarks.")
    parser.add_argument("input", type=Path, help="Folder containing numbered JPEG frames.")
    parser.add_argument("output", type=Path, help="Folder for aligned frames.")
    parser.add_argument("--base-start", type=int, default=0, help="First stable frame index for baseline.")
    parser.add_argument("--base-end", type=int, default=20, help="Last stable frame index for baseline.")
    args = parser.parse_args()

    files = sorted(args.input.glob("*.jpg"))
    if not files:
        raise SystemExit(f"No JPG frames found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        base_distance, base_center = compute_base(args.input, args.base_start, args.base_end, face_mesh)
        print(f"Base eye distance: {base_distance:.2f}px")
        print(f"Base eye center: ({base_center[0]:.1f}, {base_center[1]:.1f})")

        for index, path in enumerate(tqdm(files, desc="Aligning frames")):
            image = cv2.imread(str(path))
            if image is None:
                continue
            aligned = align_image(image, face_mesh, base_distance, base_center)
            cv2.imwrite(str(args.output / f"{index:04d}.jpg"), aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == "__main__":
    main()
