import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= تنظیمات مسیرها =================
INPUT_DIR = r"D:\Programming Projects\Python Projects\BreastCancer\CBIS-DDSM\jpeg"
OUTPUT_DIR = r"D:\Programming Projects\Python Projects\BreastCancer\CBIS-DDSM_processed"
TARGET_SIZE = (512, 512)


# ===================================================

def crop_breast_region(image):
    """شناسایی و برش بافت اصلی سینه و حذف پس‌زمینه سیاه"""
    # آستانه‌گذاری برای جداسازی بافت از پس‌زمینه
    _, thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)

    # پیدا کردن کانتورها (محدوده‌های متصل)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # یافتن بزرگ‌ترین کانتور که همان بافت سینه است
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # برش تصویر بر اساس کادر بافت سینه
    return image[y:y + h, x:x + w]


def process_single_image(img_path):
    """اعمال مراحل پیش‌پردازش روی یک تصویر"""
    # ۱. خواندن تصویر به صورت خاکستری (Grayscale)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # ۲. حذف پس‌زمینه سیاه و برش بافت سینه
    cropped_img = crop_breast_region(img)

    # ۳. بهبود کنتراست با روش CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(cropped_img)

    # ۴. تغییر ابعاد به سایز استاندارد
    resized_img = cv2.resize(enhanced_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    return resized_img


def main():
    print("شروع جستجوی تصاویر...")
    image_paths = []

    # پیدا کردن تمام فایل‌های JPG در تمام زیرپوشه‌ها
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    print(f"تعداد {len(image_paths)} تصویر یافت شد. شروع پیش‌پردازش...")

    processed_count = 0
    for src_path in tqdm(image_paths, desc="در حال پردازش"):
        # حفظ ساختار پوشه‌بندی در پوشه خروجی
        relative_path = os.path.relpath(src_path, INPUT_DIR)
        dest_path = os.path.join(OUTPUT_DIR, relative_path)

        # ساخت پوشه‌های مقصد در صورت عدم وجود
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # پردازش و ذخیره
        processed_img = process_single_image(src_path)
        if processed_img is not None:
            cv2.imwrite(dest_path, processed_img)
            processed_count += 1

    print(f"\nعملیات با موفقیت تمام شد! {processed_count} تصویر در مسیر زیر ذخیره شدند:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()