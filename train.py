import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= تنظیمات مسیرها و هایپرپارامترها =================
CSV_PATH = r"D:\Programming Projects\Python Projects\BreastCancer\final_dataset.csv"
PROCESSED_IMG_DIR = r"D:\Programming Projects\Python Projects\BreastCancer\CBIS-DDSM_processed"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================================================

print("در حال همگام‌سازی هوشمند فایل CSV با عکس‌های JPG...")
df = pd.read_csv(CSV_PATH)

# ۱. یافتن تمام فایل‌های JPG روی هارد و ساخت دیکشنری جستجوی سریع
all_jpg_files = glob.glob(os.path.join(PROCESSED_IMG_DIR, "**", "*.jpg"), recursive=True)
if not all_jpg_files:
    # اگر پسوند فایل‌ها jpeg باشد
    all_jpg_files = glob.glob(os.path.join(PROCESSED_IMG_DIR, "**", "*.jpeg"), recursive=True)

print(f"تعداد کل تصاویر یافت‌شده در پوشه پیش‌پردازش: {len(all_jpg_files)}")

# نگاشت بر اساس (نام پوشه پدر، نام فایل بدون پسوند) و همچنین (نام پوشه پدر)
jpg_lookup = {}
for filepath in all_jpg_files:
    parent_folder = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    fname_no_ext = os.path.splitext(filename)[0]

    # کلید ۱: ترکیبی از فولدر و نام فایل
    jpg_lookup[(parent_folder, fname_no_ext)] = filepath
    # کلید ۲: فقط نام فولدر (جهت پشتیبانی از تک‌فایل در فولدر)
    jpg_lookup[parent_folder] = filepath
    # کلید ۳: فقط نام فایل بدون پسوند
    jpg_lookup[fname_no_ext] = filepath


def find_actual_image_path(dicom_path_str):
    if pd.isna(dicom_path_str):
        return None

    # استانداردسازی اسلش‌ها
    clean_path = str(dicom_path_str).replace('\\', '/').strip()
    parts = [p for p in clean_path.split('/') if p]

    if not parts:
        return None

    # بررسی تطابق پوشه + نام فایل
    if len(parts) >= 2:
        folder = parts[-2]
        fname = os.path.splitext(parts[-1])[0]
        if (folder, fname) in jpg_lookup:
            return jpg_lookup[(folder, fname)]

    # بررسی اجزای مسیر به صورت معکوس
    for part in reversed(parts):
        fname_no_ext = os.path.splitext(part)[0]
        if part in jpg_lookup:
            return jpg_lookup[part]
        if fname_no_ext in jpg_lookup:
            return jpg_lookup[fname_no_ext]

    return None


df['valid_image_path'] = df['image file path'].apply(find_actual_image_path)
df_cleaned = df.dropna(subset=['valid_image_path']).reset_index(drop=True)

print(f"تعداد {len(df_cleaned)} تصویر تطبیق داده شده و آماده برای آموزش پیدا شد.")

if len(df_cleaned) == 0:
    print("\n[اشکال‌یابی] نمونه‌ای از آدرس در CSV:")
    print(df['image file path'].dropna().iloc[0])
    print("[اشکال‌یابی] نمونه‌ای از آدرس عکس روی هارد:")
    if all_jpg_files:
        print(all_jpg_files[0])
    raise ValueError("هیچ تطابقی پیدا نشد. لطفا دو نمونه آدرس بالا را بررسی کنید.")

# تفکیک داده‌ها به بخش آموزش (Train) و اعتبارسنجی (Validation)
train_df, val_df = train_test_split(
    df_cleaned,
    test_size=0.2,
    stratify=df_cleaned['target_label'],
    random_state=42
)


# --- ۲. تعریف Custom Dataset برای PyTorch ---
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['valid_image_path']
        label = int(row['target_label'])

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = BreastCancerDataset(train_df, transform=train_transforms)
val_dataset = BreastCancerDataset(val_df, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- ۳. تعریف و آموزش مدل ---
print(f"\nدر حال انتقال مدل به سخت‌افزار: {DEVICE}")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("شروع فرآیند آموزش...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    epoch_loss = running_loss / total_train
    epoch_acc = correct_train / total_train

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_loss / total_val
    val_acc = correct_val / total_val

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] -> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

torch.save(model.state_dict(), "breast_cancer_resnet50.pth")
print("\nآموزش با موفقیت پایان یافت و مدل ذخیره شد.")