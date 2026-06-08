import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# =========================================
# CONFIG
# =========================================

EXCEL_PATH = "hasil_pengujian_svm.xlsx"
HYBRID_JSON_PATH = "hybrid.json"

OUTPUT_EXCEL = "hasil_evaluasi_hybrid.xlsx"
OUTPUT_CM_IMAGE = "confusion_matrix_hybrid.png"

# =========================================
# LOAD DATA TEST
# =========================================

print("📥 Load data testing asli...")

df_test = pd.read_excel(EXCEL_PATH)

print("\n📌 Kolom ditemukan:")
print(df_test.columns.tolist())

# =========================================
# KOLOM LABEL ASLI
# =========================================

label_col = "Label_Asli"

if label_col not in df_test.columns:
    print(f"\n❌ ERROR: Kolom '{label_col}' tidak ditemukan!")
    exit()

# =========================================
# LABEL MAPPING
# =========================================

LABEL_MAP = {
    'hoax': 0,
    'hoaks': 0,
    'fake': 0,
    'false': 0,
    'tidak benar': 0,

    'valid': 1,
    'fakta': 1,
    'benar': 1,
    'true': 1
}

# =========================================
# CLEAN LABEL
# =========================================

df_test[label_col] = (
    df_test[label_col]
    .astype(str)
    .str.lower()
    .str.strip()
)

# ubah ke numerik
df_test['label_num'] = df_test[label_col].map(LABEL_MAP)

# hapus data gagal mapping
df_test = df_test.dropna(subset=['label_num'])

# label asli
y_true = df_test['label_num'].astype(int).tolist()

# =========================================
# LOAD HYBRID.JSON
# =========================================

print("\n📦 Load hybrid.json...")

with open(HYBRID_JSON_PATH, "r", encoding="utf-8") as f:
    hybrid_results = json.load(f)

# =========================================
# AMBIL HASIL PREDIKSI HYBRID
# =========================================

y_pred = []
prediksi_text = []

for item in hybrid_results:

    final_label = str(
        item.get("final_label", "")
    ).upper().strip()

    if final_label == "VALID":
        y_pred.append(1)
        prediksi_text.append("VALID")

    elif final_label == "HOAKS":
        y_pred.append(0)
        prediksi_text.append("HOAKS")

# =========================================
# VALIDASI
# =========================================

print(f"\nJumlah label asli : {len(y_true)}")
print(f"Jumlah prediksi   : {len(y_pred)}")

if len(y_true) != len(y_pred):

    print("\n❌ ERROR:")
    print("Jumlah data tidak sama!")
    exit()

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

# =========================================
# METRIK EVALUASI
# =========================================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# =========================================
# TAMPILKAN HASIL
# =========================================

print("\n📊 HASIL EVALUASI HYBRID SVM + API\n")

print("Confusion Matrix:")
print(cm)

print("\n📌 DETAIL CONFUSION MATRIX")
print(f"True Positive  (TP): {tp}")
print(f"True Negative  (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")

print("\n📈 METRIK PERFORMA")

print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")

print("\n📄 Classification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=["Hoaks", "Valid"]
))

# =========================================
# STATUS BENAR / SALAH
# =========================================

status = [
    "BENAR" if a == b else "SALAH"
    for a, b in zip(y_true, y_pred)
]

# =========================================
# TAMBAHKAN KE DATAFRAME
# =========================================

df_test['Prediksi_Hybrid'] = prediksi_text
df_test['Status_Hybrid'] = status

# =========================================
# SIMPAN EXCEL HASIL
# =========================================

df_test.to_excel(OUTPUT_EXCEL, index=False)

print(f"\n💾 File excel hasil hybrid:")
print(OUTPUT_EXCEL)

# =========================================
# BUAT GAMBAR CONFUSION MATRIX
# =========================================

plt.figure(figsize=(6, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["HOAKS", "VALID"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix Hybrid SVM + API")

plt.savefig(OUTPUT_CM_IMAGE)

plt.close()

print(f"\n🖼️ Gambar confusion matrix disimpan:")
print(OUTPUT_CM_IMAGE)

# =========================================
# RINGKASAN
# =========================================

benar = sum(
    1 for a, b in zip(y_true, y_pred)
    if a == b
)

salah = sum(
    1 for a, b in zip(y_true, y_pred)
    if a != b
)

print("\n📌 RINGKASAN")

print(f"Total Data : {len(y_true)}")
print(f"Benar      : {benar}")
print(f"Salah      : {salah}")

print("\n✅ Evaluasi Hybrid selesai.")