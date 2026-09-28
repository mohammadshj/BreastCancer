import os
import pandas as pd

# ================= مسیرها =================
CSV_DIR = r"D:\Programming Projects\Python Projects\BreastCancer\CBIS-DDSM\csv"
OUTPUT_CSV = r"D:\Programming Projects\Python Projects\BreastCancer\final_dataset.csv"


# =========================================

def load_and_clean_csv(file_name, case_type, split_type):
    file_path = os.path.join(CSV_DIR, file_name)
    df = pd.read_csv(file_path)

    # انتخاب ستون‌های کلیدی
    selected_cols = ['patient_id', 'left or right breast', 'image view', 'abnormality type', 'pathology',
                     'image file path']
    available_cols = [col for col in selected_cols if col in df.columns]
    df = df[available_cols].copy()

    df['case_type'] = case_type
    df['split'] = split_type

    # استانداردسازی برچسب‌ها (0 برای خوش‌خیم، 1 برای بدخیم)
    df['target_label'] = df['pathology'].apply(lambda x: 1 if str(x).startswith('MALIGNANT') else 0)
    df['pathology_class'] = df['pathology'].replace({
        'BENIGN_WITHOUT_CALLBACK': 'BENIGN',
        'BENIGN': 'BENIGN',
        'MALIGNANT': 'MALIGNANT'
    })

    return df


print("در حال استخراج و ترکیب برچسب‌ها...")

# بارگذاری تمامی فایل‌های اصلی
mass_train = load_and_clean_csv('mass_case_description_train_set.csv', 'mass', 'train')
mass_test = load_and_clean_csv('mass_case_description_test_set.csv', 'mass', 'test')
calc_train = load_and_clean_csv('calc_case_description_train_set.csv', 'calc', 'train')
calc_test = load_and_clean_csv('calc_case_description_test_set.csv', 'calc', 'test')

# ترکیب تمام جدول‌ها در یک جدول واحد
final_df = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True)

# ذخیره خروجی نهایی
final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"فایل نهایی برچسب‌ها با {len(final_df)} ردیف ساخته شد:")
print(OUTPUT_CSV)
print("\nتوزیع برچسب‌ها:")
print(final_df['pathology_class'].value_counts())