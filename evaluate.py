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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ================= تنظیمات =================
CSV_PATH = r"D:\Programming Projects\Python Projects\BreastCancer\final_dataset.csv"
PROCESSED_IMG_DIR = r"D:\Programming Projects\Python Projects\BreastCancer\CBIS-DDSM_processed"

MODELS_TO_COMPARE = [
    {"name": "ResNet50", "arch": "resnet50", "path": "breast_cancer_resnet50.pth"},
    {"name": "DenseNet121", "arch": "densenet121", "path": "breast_cancer_densenet121.pth"},
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================

print(f"در حال آماده‌سازی داده‌ها جهت مقایسه مدل‌ها روی سخت‌افزار: {DEVICE}...")

# ۱. همگام‌سازی فایل CSV با عکس‌ها
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

_, val_df = train_test_split(df_cleaned, test_size=0.2, stratify=df_cleaned['target_label'], random_state=42)


# ۲. تعریف DataLoader
class BreastDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['valid_image_path']).convert('RGB')
        return self.transform(img), torch.tensor(int(row['target_label']), dtype=torch.float32)


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_loader = DataLoader(BreastDataset(val_df, val_transform), batch_size=16, shuffle=False)


# ۳. توابع بارگذاری و ارزیابی
def load_model_weights(arch, weights_path):
    if arch == "resnet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "densenet121":
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        raise ValueError(f"معماری {arch} ناشناخته است.")

    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_single_model(model):
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_probs)


# ۴. اجرای ارزیابی روی تمام مدل‌ها
results = {}

for m in MODELS_TO_COMPARE:
    name, arch, path = m["name"], m["arch"], m["path"]
    if os.path.exists(path):
        print(f"\n================ ارزیابی مدل: {name} ================")
        model = load_model_weights(arch, path)
        y_true, y_pred, y_probs = evaluate_single_model(model)
        auc_score = roc_auc_score(y_true, y_probs)

        results[name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs,
            "auc": auc_score
        }

        print(classification_report(y_true, y_pred, target_names=['Benign (خوش‌خیم)', 'Malignant (بدخیم)']))
        print(f"نمره ROC-AUC برای {name}: {auc_score:.4f}")
    else:
        print(f"\n[!] فایل وزنی مدل {name} در مسیر '{path}' پیدا نشد. این مدل از مقایسه حذف می‌شود.")

# ۵. رسم نمودارهای مقایسه‌ای
if len(results) > 0:
    # الف) رسم Confusion Matrix مدل‌ها کنار هم
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        axes[idx].set_title(f"Confusion Matrix ({name})")
        axes[idx].set_xlabel('پیش‌بینی مدل')
        axes[idx].set_ylabel('واقعیت')

    plt.tight_layout()
    plt.savefig("models_confusion_matrices.png")
    print("\n[+] تصویر مقایسه‌ای ماتریس‌های درهم‌ریختگی در 'models_confusion_matrices.png' ذخیره شد.")

    # ب) رسم منحنی مقایسه‌ای ROC Curve
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_probs"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.4f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label="حدس تصادفی (AUC = 0.50)")
    plt.xlabel("False Positive Rate (تولید هشدار اشتباه)")
    plt.ylabel("True Positive Rate / Recall (حساسیت تشخیص بدخیمی)")
    plt.title("نمودار مقایسه‌ای منحنی ROC مدل‌ها")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("models_roc_comparison.png")
    print("[+] نمودار مقایسه‌ای منحنی ROC در 'models_roc_comparison.png' ذخیره شد.")
    plt.show()