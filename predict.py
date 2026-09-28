import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "breast_cancer_resnet50.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"خطا: فایلی در مسیر {image_path} یافت نشد.")
        return

    # ۱. بارگذاری مدل
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)

    # استفاده از weights_only=True جهت رفع Warning
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model = model.to(DEVICE)
    model.eval()

    # ۲. پیش‌پردازش تصویر
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ۳. پیش‌بینی
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    is_malignant = probability > 0.5
    label = "Malignant (بدخیم)" if is_malignant else "Benign (خوش‌خیم)"
    confidence = probability if is_malignant else (1 - probability)

    print(f"\n================ نتایج تحلیل تصویر ================")
    print(f"مسیر تصویر: {image_path}")
    print(f"تشخیص مدل: {label}")
    print(f"احتمال بدخیم بودن: {probability * 100:.2f}%")
    print(f"میزان اطمینان مدل: {confidence * 100:.2f}%")
    print(f"====================================================\n")


if __name__ == "__main__":
    # آدرس یک تصویر نمونه جهت تست (می‌توانید مسیر هر تصویر JPG دیگری را جایگزین کنید)
    sample_image = r"D:\Programming Projects\Python Projects\BreastCancer\CBIS-DDSM_processed\1.3.6.1.4.1.9590.100.1.2.101282442711963955211635160453427358442\1-072.jpg"
    predict_image(sample_image)