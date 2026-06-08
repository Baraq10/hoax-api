import matplotlib
matplotlib.use('Agg')
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================
# CONFIG
# =========================================

EXCEL_PATH = "hasil_pengujian_svm.xlsx"
HYBRID_JSON_PATH = "hybrid.json"

# nama file output otomatis agar tidak bentrok
timestamp = int(time.time())

OUTPUT_EXCEL = f"hasil_pengujian_hybrid_{timestamp}.xlsx"
OUTPUT_METRICS = f"metrics_hybrid_{timestamp}.png"
OUTPUT_CM = f"confusion_matrix_hybrid_{timestamp}.png"

# =========================================
# LOAD FILE EXCEL
# =========================================

print("📥 Load hasil pengujian SVM...")

df_test = pd.read_excel(EXCEL_PATH)

required_cols = ["Text", "Label_Asli"]

for col in required_cols:

    if col not in df_test.columns:

        raise ValueError(
            f"Kolom '{col}' tidak ditemukan!"
        )

# =========================================
# LOAD HYBRID.JSON
# =========================================

print("📦 Load hybrid.json...")

with open(
    HYBRID_JSON_PATH,
    "r",
    encoding="utf-8"
) as f:

    hybrid_results = json.load(f)

# =========================================
# VALIDASI JUMLAH DATA
# =========================================

if len(df_test) != len(hybrid_results):

    print("\n❌ ERROR:")
    print("Jumlah data excel dan hybrid.json berbeda!")

    print(f"Excel  : {len(df_test)}")
    print(f"Hybrid : {len(hybrid_results)}")

    exit()

# =========================================
# LABEL MAP
# =========================================

LABEL_MAP = {
    "HOAKS": 0,
    "VALID": 1
}

# =========================================
# LABEL ASLI
# =========================================

y_true = []

for label in df_test["Label_Asli"]:

    label_clean = str(label).upper().strip()

    y_true.append(
        LABEL_MAP[label_clean]
    )

# =========================================
# HASIL HYBRID
# =========================================

y_pred = []

prediksi_text = []

confidence_list = []

for item in hybrid_results:

    final_label = str(
        item.get("final_label", "")
    ).upper().strip()

    confidence = item.get(
        "evidence_confidence_percent",
        0
    )

    confidence_list.append(confidence)

    # VALID
    if final_label == "VALID":

        y_pred.append(1)

        prediksi_text.append("VALID")

    # HOAKS
    else:

        y_pred.append(0)

        prediksi_text.append("HOAKS")

# =========================================
# STATUS BENAR / SALAH
# =========================================

status_list = []

for asli, pred in zip(y_true, y_pred):

    if asli == pred:

        status_list.append("BENAR")

    else:

        status_list.append("SALAH")

# =========================================
# EVALUASI
# =========================================

print("\n===== HASIL HYBRID SVM + API =====\n")

cm = confusion_matrix(
    y_true,
    y_pred
)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=["Hoaks", "Valid"]
))

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# =========================================
# VISUALISASI METRICS
# =========================================

# =========================================
# VISUALISASI METRICS
# =========================================

print("\n📊 Membuat metrics_hybrid.png ...")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

values = [
    accuracy,
    precision,
    recall,
    f1
]

# reset figure
plt.clf()

fig = plt.figure(figsize=(7,5))

ax = fig.add_subplot(111)

bars = ax.bar(metrics, values)

# tampilkan angka
for i, v in enumerate(values):

    ax.text(
        i,
        v + 0.02,
        f"{v:.2f}",
        ha='center',
        fontsize=10
    )

ax.set_ylim(0, 1)

ax.set_ylabel("Score")

ax.set_title("Performa Hybrid SVM + API")

# save image
metrics_path = "metrics_hybrid.png"

fig.savefig(
    metrics_path,
    dpi=300,
    bbox_inches='tight'
)

print(f"✅ Metrics berhasil disimpan: {metrics_path}")

plt.close(fig)

# =========================================
# CONFUSION MATRIX
# =========================================

# =========================================
# CONFUSION MATRIX
# =========================================

print("\n🖼️ Membuat confusion matrix...")

plt.clf()

fig = plt.figure(figsize=(6,5))

ax = fig.add_subplot(111)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['HOAKS', 'VALID'],
    yticklabels=['HOAKS', 'VALID'],
    ax=ax
)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

ax.set_title("Confusion Matrix Hybrid SVM + API")

cm_path = "confusion_matrix_hybrid.png"

fig.savefig(
    cm_path,
    dpi=300,
    bbox_inches='tight'
)

print(f"✅ Confusion matrix berhasil disimpan: {cm_path}")

plt.close(fig)

print("✅ Confusion matrix berhasil dibuat:")
print(OUTPUT_CM)

# =========================================
# SIMPAN HASIL EXCEL
# =========================================

print("\n📄 Membuat file excel hasil hybrid...")

hasil_hybrid = pd.DataFrame({

    "Text": df_test["Text"],

    "Label_Asli": df_test["Label_Asli"],

    "Prediksi_Hybrid": prediksi_text,

    "Confidence_Hybrid": confidence_list,

    "Status": status_list
})

hasil_hybrid.to_excel(
    OUTPUT_EXCEL,
    index=False
)

print("✅ File excel berhasil dibuat:")
print(OUTPUT_EXCEL)

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

print("\n===== RINGKASAN =====")

print(f"Total Data : {len(y_true)}")
print(f"Benar      : {benar}")
print(f"Salah      : {salah}")

print("\n✅ Evaluasi hybrid selesai.")