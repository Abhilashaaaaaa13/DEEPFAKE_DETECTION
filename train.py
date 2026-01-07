import torch
import torch.nn as nn
import torch.optim as optim #optimizer
from torch.utils.data import DataLoader, Dataset  #data ko batch m lane k liyee
from torchvision import models  #efficient net k liyee
import albumentations as A  #for augmentations
from albumentations.pytorch import ToTensorV2
import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# config
DATA_DIR = "processed_data"
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# augmentations
def get_train_transforms():
    return A.Compose([
        A.OneOf([
            A.ImageCompression(quality_range=(60, 100), p=1.0),   # fixed warning
            A.GaussianBlur(blur_limit=3, p=1.0)
        ], p=0.5),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
        A.ToGray(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-15,15), p=0.3), #thoda ghumana, zoom
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), #no ko control range m rkhna
        ToTensorV2()
    ])

val_tfms = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()  #image ->tensor
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

# main training
if __name__ == "__main__":  #  Windows-safe guard

    #  Prepare file paths and labels 
    all_paths, all_labels = [], []
    for label_name, label_idx in [('real',0), ('fake',1)]:
        dir_path = os.path.join(DATA_DIR, label_name)
        files = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith(('.jpg','.png'))]
        all_paths.extend(files)
        all_labels.extend([label_idx]*len(files))

    #  Stratified split 
    train_p, val_p, train_l, val_l = train_test_split(
        all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=42
    )

    train_ds = DeepfakeDataset(train_p, train_l, get_train_transforms())
    val_ds = DeepfakeDataset(val_p, val_l, val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4,inplace=True),
        nn.Linear(num_ftrs,256),
        nn.ReLU(),
        nn.Linear(256,2)
    )
    model = model.to(DEVICE)

    #Loss & Optimizer 
    weights = torch.tensor([1.22,0.85]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")  # updated for PyTorch >=2.0

    # Training Loop 
    best_acc = 0.0
    print(f" Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        # Validation 
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        avg_train_loss = running_loss/len(train_loader)
        print(f"📊 Epoch {epoch+1}: Loss={avg_train_loss:.4f} | Val Acc={val_acc:.2f}%")

        #  Save Best Model 
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }, "best_deepfake_pro.pth")

    print(f"\n Training Complete! Best Accuracy: {best_acc:.2f}%")
