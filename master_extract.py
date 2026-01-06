import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ================= CONFIG =================
OUTPUT_DIR = "processed_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256
MARGIN = 0.30
MAX_FRAMES_PER_VIDEO = 12

TARGETS = {
    "celeb": {"real": 2000, "fake": 2000},
    "ff": {"real": 1000, "fake": 1000},
    "dfdc": {"real": 2000, "fake": 2000},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

mtcnn = MTCNN(keep_all=False, device=DEVICE)

# ================= UTILS =================
def expand_bbox(bbox, width, height, margin):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - w * margin))
    y1 = max(0, int(y1 - h * margin))
    x2 = min(width, int(x2 + w * margin))
    y2 = min(height, int(y2 + h * margin))
    return x1, y1, x2, y2


def extract_faces(video_path, label, prefix, target, current):
    if current >= target:
        return 0

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return 0

    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

    frames = np.linspace(0, total - 1, MAX_FRAMES_PER_VIDEO, dtype=int)
    count = 0

    for idx in frames:
        if current + count >= target:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        boxes, _ = mtcnn.detect(frame)
        if boxes is None:
            continue

        x1, y1, x2, y2 = expand_bbox(
            boxes[0], frame.shape[1], frame.shape[0], MARGIN
        )

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, label, f"{prefix}_{idx}.jpg"), face
        )
        count += 1

    cap.release()
    return count


# ================= DATASETS =================
def run_celeb():
    print("\n🔥 Processing CELEB-DF")
    counts = {"real": 0, "fake": 0}

    celeb_map = {
        "Celeb-real": "real",
        "celeb-synthesis": "fake",
    }

    for folder, label in celeb_map.items():
        path = os.path.join("DATASET", folder)
        videos = [v for v in os.listdir(path) if v.endswith(".mp4")]

        for v in tqdm(videos):
            counts[label] += extract_faces(
                os.path.join(path, v),
                label,
                f"celeb_{v}",
                TARGETS["celeb"][label],
                counts[label],
            )
            if counts[label] >= TARGETS["celeb"][label]:
                break


def run_ff():
    print("\n🔥 Processing FaceForensics++")
    counts = {"real": 0, "fake": 0}
    base = "DATASET/FaceForensics++_C23"

    real_folders = ["original", "DeepFakeDetection"]
    fake_folders = [
        "DeepFakes",
        "Face2Face",
        "FaceSwap",
        "FaceShifter",
        "NeuralTextures",
    ]

    for folder in real_folders:
        path = os.path.join(base, folder)
        if not os.path.exists(path):
            continue

        for v in os.listdir(path):
            if not v.endswith(".mp4"):
                continue

            counts["real"] += extract_faces(
                os.path.join(path, v),
                "real",
                f"ff_real_{folder}_{v}",
                TARGETS["ff"]["real"],
                counts["real"],
            )
            if counts["real"] >= TARGETS["ff"]["real"]:
                break

    for folder in fake_folders:
        path = os.path.join(base, folder)
        if not os.path.exists(path):
            continue

        for v in os.listdir(path):
            if not v.endswith(".mp4"):
                continue

            counts["fake"] += extract_faces(
                os.path.join(path, v),
                "fake",
                f"ff_fake_{folder}_{v}",
                TARGETS["ff"]["fake"],
                counts["fake"],
            )
            if counts["fake"] >= TARGETS["ff"]["fake"]:
                break


def run_dfdc():
    print("\n🔥 Processing DFDC")
    base = "DATASET/train_sample_videos"

    with open(os.path.join(base, "metadata.json")) as f:
        meta = json.load(f)

    counts = {"real": 0, "fake": 0}

    for v, info in tqdm(meta.items()):
        label = info["label"].lower()

        counts[label] += extract_faces(
            os.path.join(base, v),
            label,
            f"dfdc_{v}",
            TARGETS["dfdc"][label],
            counts[label],
        )

        if (
            counts["real"] >= TARGETS["dfdc"]["real"]
            and counts["fake"] >= TARGETS["dfdc"]["fake"]
        ):
            break


# ================= RUN =================
if __name__ == "__main__":
    run_celeb()
    run_ff()
    run_dfdc()

    print(f"\n✅ DONE! Check '{OUTPUT_DIR}/real' and '{OUTPUT_DIR}/fake'")
