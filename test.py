import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from tqdm import tqdm

# config
VIDEO_FOLDER = "./DATASET./test_videos"  # Apne testing videos ka folder
MODEL_PATH = "best_deepfake_pro.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. MODEL LOADER 
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

# 2. inference transforms
test_tfms = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Initialize MTCNN for Face Detection
mtcnn = MTCNN(keep_all=False, device=DEVICE)

# 3. Video prediction logic
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_scores = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 40, dtype=int)  # Fast check

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            boxes, _ = mtcnn.detect(rgb_frame)
            if boxes is not None:
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                # 30% Margin
                x1, y1 = max(0, int(x1 - 0.3*w)), max(0, int(y1 - 0.3*h))
                x2, y2 = min(frame.shape[1], int(x2 + 0.3*w)), min(frame.shape[0], int(y2 + 0.3*h))

                face = rgb_frame[y1:y2, x1:x2]
                if face.size > 0:
                    input_img = test_tfms(image=face)["image"].unsqueeze(0).to(DEVICE)
                    output = model(input_img)
                    probs = torch.softmax(output, dim=1)
                    fake_prob = probs[0][1].item()
                    frame_scores.append(fake_prob)

    cap.release()

    if not frame_scores:
        return "No Face Detected", 0.0

    # 3-Level Verdict (REAL / FAKE / UNCERTAIN)
    avg_fake_score = sum(frame_scores) / len(frame_scores)

    if avg_fake_score >= 0.80:
        verdict = "🔴 FAKE"
    elif avg_fake_score <= 0.20:
        verdict = "🟢 REAL"
    else:
        verdict = "🟡 UNCERTAIN"

    return verdict, avg_fake_score

# 4. run test
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f" Error: {MODEL_PATH} nahi mila! Pehle training poori karo.")
    else:
        print(f" Loading Model & Testing on {DEVICE}...")
        pro_model = get_model(MODEL_PATH)

        videos = [v for v in os.listdir(VIDEO_FOLDER) if v.endswith(('.mp4', '.mkv', '.mov'))]

        print(f"\n{'Video Name':<30} | {'Fake Prob':<12} | {'Verdict'}")
        print("-" * 60)

        for v_name in videos:
            v_path = os.path.join(VIDEO_FOLDER, v_name)
            verdict, score = process_video(v_path, pro_model)
            print(f"{v_name[:30]:<30} | {score*100:>8.2f}% | {verdict}")
