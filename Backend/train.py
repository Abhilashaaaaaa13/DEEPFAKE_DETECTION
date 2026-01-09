import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# config
DATA_DIR = "processed_data"
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.7          #  Fake declare threshold
TEMPERATURE = 1.5        #  Confidence calibration

# augmentations
def get_train_transforms():
    return A.Compose([
        A.OneOf([
            A.ImageCompression(quality_range=(60, 100), p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0)
        ], p=0.5),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
        A.ToGray(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-15,15), p=0.3),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

val_tfms = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

# dataset
class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.file_paths[idx])
        if img is None:
            return torch.zeros((3,256,256)), self.labels[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx]

# video temporal smoothing
def video_level_prediction(frame_fake_probs, threshold=0.6):
    avg_prob = sum(frame_fake_probs) / len(frame_fake_probs)
    return "Fake" if avg_prob >= threshold else "Real"

# training
if __name__ == "__main__":

    all_paths, all_labels = [], []
    for label_name, label_idx in [('real',0), ('fake',1)]:
        dir_path = os.path.join(DATA_DIR, label_name)
        files = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith(('.jpg','.png'))]
        all_paths.extend(files)
        all_labels.extend([label_idx]*len(files))

    train_p, val_p, train_l, val_l = train_test_split(
        all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=42
    )

    train_ds = DeepfakeDataset(train_p, train_l, get_train_transforms())
    val_ds = DeepfakeDataset(val_p, val_l, val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs,256),
        nn.ReLU(),
        nn.Linear(256,2)
    )
    model.to(DEVICE)

    # Calibrated loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    print(f"🚀 Training on {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                    outputs = model(imgs)
                    outputs = outputs / TEMPERATURE   #  Temperature scaling

                    probs = F.softmax(outputs, dim=1)
                    fake_prob = probs[:,1]

                    preds = (fake_prob >= THRESHOLD).long()

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        total = len(all_labels)
        val_acc = 100 * correct / total

        print(f"\n Epoch {epoch+1}")
        print(f"Loss: {running_loss/len(train_loader):.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")
        print(classification_report(all_labels, all_preds, target_names=["Real","Fake"]))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
            }, "best_deepfake_pro.pth")

    print(f"\n Training Done | Best Accuracy: {best_acc:.2f}%")
