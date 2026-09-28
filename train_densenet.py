import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= تنظیمات اصلی =================
CSV_PATH = r"D:\Programming Projects\Python Projects\BreastCancer\final_dataset.csv"
PROCESSED_IMG_DIR = r"D:\Programming Projects\Python Projects\BreastCancer\CBIS-DDSM_processed"
SAVED_MODEL_NAME = "breast_cancer_densenet121.pth"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"در حال اجرای آموزش روی سخت‌افزار: {DEVICE}")

# --- ۱. تطبیق تصاویر با CSV ---
df = pd.read_csv(CSV_PATH)
all_jpg_files = glob.glob(os.path.join(PROCESSED_IMG_DIR, "**", "*.jpg"), recursive=True)
if not all_jpg_files:
    all_jpg_files = glob.glob(os.path.join(PROCESSED_IMG_DIR, "**", "*.jpeg"), recursive=True)

jpg_lookup = {}
for filepath in all_jpg_files:
    parent_folder = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    fname_no_ext = os.path.splitext(filename)[0]
    jpg_lookup[(parent_folder, fname_no_ext)] = filepath
    jpg_lookup[parent_folder] = filepath
    jpg_lookup[fname_no_ext] = filepath


def find_actual_image_path(dicom_path_str):
    if pd.isna(dicom_path_str): return None
    clean_path = str(dicom_path_str).replace('\\', '/').strip()
    parts = [p for p in clean_path.split('/') if p]
    if len(parts) >= 2:
        folder, fname = parts[-2], os.path.splitext(parts[-1])[0]
        if (folder, fname) in jpg_lookup: return jpg_lookup[(folder, fname)]
    for part in reversed(parts):
        fname_no_ext = os.path.splitext(part)[0]
        if part in jpg_lookup: return jpg_lookup[part]
        if fname_no_ext in jpg_lookup: return jpg_lookup[fname_no_ext]
    return None


df['valid_image_path'] = df['image file path'].apply(find_actual_image_path)
df_cleaned = df.dropna(subset=['valid_image_path']).reset_index(drop=True)

train_df, val_df = train_test_split(df_cleaned, test_size=0.2, stratify=df_cleaned['target_label'], random_state=42)


# --- ۲. ساخت Dataset و DataLoader ---
class BreastDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['valid_image_path']).convert('RGB')
        return self.transform(img), torch.tensor(int(row['target_label']), dtype=torch.float32)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(BreastDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BreastDataset(val_df, val_transform), batch_size=BATCH_SIZE, shuffle=False)

# --- ۳. تعریف مدل DenseNet121 ---
print("در حال دانلود و ساخت مدل DenseNet121...")
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.classifier = nn.Linear(model.classifier.in_features, 1)  # لایه آخر برای تشخیص دوتایی
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- ۴. حلقه آموزش ---
print("شروع فرآیند آموزش DenseNet121...")
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        train_correct += (preds == labels).sum().item()

    # ارزیابی
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            val_correct += (preds == labels).sum().item()

    train_acc = (train_correct / len(train_df)) * 100
    val_acc = (val_correct / len(val_df)) * 100
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] -> Train Loss: {train_loss / len(train_df):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss / len(val_df):.4f} | Val Acc: {val_acc:.2f}%")

# ذخیره مدل جدید
torch.save(model.state_dict(), SAVED_MODEL_NAME)
print(f"\nآموزش کامل شد! وزن‌های مدل جدید در {SAVED_MODEL_NAME} ذخیره شد.")