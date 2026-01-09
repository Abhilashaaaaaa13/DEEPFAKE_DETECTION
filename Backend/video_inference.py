# ================= video_inference.py =================
import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from torchvision import models
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_FAKE = 0.65     # Fake threshold
THRESHOLD_REAL = 0.35     # Real threshold
SMOOTHING_WINDOW = 5       # Temporal smoothing window

# -1m model loader
def get_model(path):
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

# 2. transforms
test_tfms = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

# -3 . temporal/out;ier functions
def smooth_scores(scores, window=SMOOTHING_WINDOW):
    smoothed = []
    for i in range(len(scores)):
        start = max(0, i - window//2)
        end = min(len(scores), i + window//2 + 1)
        smoothed.append(sum(scores[start:end]) / (end - start))
    return smoothed

def remove_outliers(scores, low=0.05, high=0.95):
    return [s for s in scores if low <= s <= high]

def weighted_average(scores):
    if not scores:
        return 0.0
    weights = [s if s>0.5 else 1-s for s in scores]
    return sum(s*w for s,w in zip(scores, weights)) / sum(weights)

# -4 video inference
mtcnn = MTCNN(keep_all=False, device=DEVICE)

def process_video(video_path, model, max_frames=40, margin=0.3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_scores = []

    if total_frames <= 0:
        return "No Frames Found", 0.0

    frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            boxes, _ = mtcnn.detect(rgb_frame)
            if boxes is None:
                continue

            box = boxes[0].astype(int)
            x1, y1, x2, y2 = box
            w, h = x2-x1, y2-y1
            x1, y1 = max(0, int(x1 - margin*w)), max(0, int(y1 - margin*h))
            x2, y2 = min(frame.shape[1], int(x2 + margin*w)), min(frame.shape[0], int(y2 + margin*h))

            face = rgb_frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            input_img = test_tfms(image=face)["image"].unsqueeze(0).to(DEVICE)
            output = model(input_img)
            probs = torch.softmax(output, dim=1)
            fake_prob = probs[0][1].item()
            frame_scores.append(fake_prob)

    cap.release()

    if not frame_scores:
        return "No Face Detected", 0.0

    #  Temporal smoothing + outlier removal + weighted average 
    smoothed = smooth_scores(frame_scores)
    filtered = remove_outliers(smoothed)
    avg_score = weighted_average(filtered)

    # uNCERTAIN logic 
    if avg_score >= THRESHOLD_FAKE:
        verdict = "🔴 FAKE"
    elif avg_score <= THRESHOLD_REAL:
        verdict = "🟢 REAL"
    else:
        verdict = "🟡 UNCERTAIN"

    return verdict, avg_score

#5. Example Usage 
if __name__ == "__main__":
    MODEL_PATH = "best_deepfake_pro.pth"
    VIDEO_FOLDER = "./DATASET/test_videos"

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
    else:
        model = get_model(MODEL_PATH)
        videos = [v for v in os.listdir(VIDEO_FOLDER) if v.endswith(('.mp4','.mkv','.mov'))]
        print(f"{'Video Name':<30} | {'Fake Prob':<12} | {'Verdict'}")
        print("-"*60)
        for v_name in videos:
            v_path = os.path.join(VIDEO_FOLDER, v_name)
            verdict, score = process_video(v_path, model)
            print(f"{v_name[:30]:<30} | {score*100:>8.2f}% | {verdict}")
