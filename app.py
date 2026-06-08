from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import pickle
import re

from openai import OpenAI
from news_fetcher import fetch_evidence_snippets

from scipy.sparse import hstack
from scipy.special import expit

from datetime import datetime

app = Flask(__name__)
CORS(app)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# LOAD MODEL
model = pickle.load(open("model/model.pkl", "rb"))
tfidf_word = pickle.load(open("model/tfidf_word.pkl", "rb"))
tfidf_char = pickle.load(open("model/tfidf_char.pkl", "rb"))


# EKSTRAK TANGGAL
def extract_date(snippet):
    pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}"
    match = re.search(pattern, snippet)

    if match:
        try:
            return datetime.strptime(match.group(), "%b %d, %Y")
        except:
            return None

    return None


# RECENCY
def recency_weight(date):

    if not date:
        return 0.5

    days = (datetime.now() - date).days

    if days <= 7:
        return 1.0

    elif days <= 30:
        return 0.9

    elif days <= 365:
        return 0.7

    elif days <= 365 * 3:
        return 0.5

    else:
        return 0.3


# API PREDICT
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    claim = data.get("text", "").strip()

    if not claim:
        return jsonify({
            "error": "Text kosong"
        }), 400


    # =========================
    # TF-IDF + SVM
    # =========================

    X = hstack([
        tfidf_word.transform([claim]),
        tfidf_char.transform([claim])
    ])


    # =========================
    # DEBUG TF-IDF WORD
    # =========================

    word_matrix = tfidf_word.transform([claim])

    word_features = tfidf_word.get_feature_names_out()

    word_scores = word_matrix.toarray()[0]

    print("\n========================================")
    print("HASIL EKSTRAKSI FITUR TF-IDF")
    print("========================================")

    for word, score in zip(word_features, word_scores):

        if score > 0:
            print(f"{word} : {round(score, 4)}")

    print("========================================")


    # =========================
    # SVM PREDICTION
    # =========================

    decision_val = model.decision_function(X)[0]

    prob = float(expit(decision_val))

    ml_confidence = round(prob * 100, 2)

    ml_label = "VALID" if prob >= 0.5 else "HOAKS"


    # =========================
    # DEBUG SVM
    # =========================

    print("\n========================================")
    print("HASIL KLASIFIKASI SVM")
    print("========================================")

    print("LABEL :", ml_label)

    print("CONFIDENCE :", ml_confidence, "%")

    print("========================================")


    # =========================
    # EVIDENCE
    # =========================

    evidences = fetch_evidence_snippets(claim)

    for e in evidences:

        tanggal = extract_date(e.get("snippet", ""))

        recency = recency_weight(tanggal)

        similarity = e.get("similarity", 0)

        credibility = e.get("credibility", 0)

        final_score = (
            similarity * 0.5 +
            credibility * 0.3 +
            recency * 0.2
        )

        e["date"] = (
            tanggal.strftime("%Y-%m-%d")
            if tanggal else None
        )

        e["recency_weight"] = recency

        e["score"] = final_score


    evidences = sorted(
        evidences,
        key=lambda x: x["score"],
        reverse=True
    )

    evidences = evidences[:5]


    # =========================
    # DEBUG API EVIDENCE
    # =========================

    print("\n========================================")
    print("HASIL INTEGRASI API BERITA")
    print("========================================")

    for i, e in enumerate(evidences, start=1):

        print(f"\nBUKTI #{i}")

        print("JUDUL :", e.get("title"))

        print("SIMILARITY :", round(e.get("similarity", 0), 4))

        print("CREDIBILITY :", round(e.get("credibility", 0), 4))

        print("FINAL SCORE :", round(e.get("score", 0), 4))

    print("========================================")


    # FINAL EVIDENCE SCORE
    best_score = max(
        [e["score"] for e in evidences],
        default=0
    )

    evidence_conf_percent = round(best_score * 100, 2)


    # DEFAULT ANALYSIS
    evidence_analysis = {
        "claim_type": "UNKNOWN",
        "evidence_support": "TIDAK_CUKUP",
        "logic_check": "TIDAK_DAPAT_DIPASTIKAN",
        "confidence": 0,
        "reason": "Tidak ada bukti",
        "explanation": "Tidak ditemukan bukti berita yang relevan."
    }


    # OPENAI (ARAH EVIDENCE)
    if evidences:

        evidence_text = ""

        for i, e in enumerate(evidences, start=1):

            evidence_text += f"""
BUKTI #{i}
Judul: {e['title']}
Tanggal: {e['date']}
Skor: {round(e['score'],3)}
Snippet: {e['snippet']}
Sumber: {e['source_type']}
"""

        prompt = f"""
KLAIM:
{claim}

BUKTI:
{evidence_text}

TUGAS:

Analisis apakah BUKTI mendukung MAKNA KESELURUHAN dari klaim pengguna.

ATURAN PENTING:

- Fokus pada inti makna klaim pengguna.
- Jangan hanya mencocokkan keyword literal.
- Jangan membuat asumsi tambahan di luar klaim.
- Jangan menambahkan opini pribadi atau interpretasi hukum.

- Jika bukti memperkuat atau membenarkan isi klaim pengguna,
  gunakan: "MENDUKUNG"

- Jika bukti secara jelas menyangkal inti klaim pengguna,
  gunakan: "BERTENTANGAN"

- Jika bukti tidak cukup relevan atau tidak cukup kuat,
  gunakan: "TIDAK_CUKUP"

- Jika pengguna mengatakan suatu informasi adalah HOAKS
  dan bukti juga menyatakan informasi itu HOAKS,
  maka hasilnya adalah "MENDUKUNG"

- Jika pengguna mengatakan suatu informasi BENAR
  tetapi bukti menyatakan HOAKS,
  maka hasilnya adalah "BERTENTANGAN"

- Jangan menggunakan label lain selain:
  "MENDUKUNG"
  "BERTENTANGAN"
  "TIDAK_CUKUP"

OUTPUT HARUS JSON VALID TANPA PENJELASAN TAMBAHAN:

{{
  "claim_type":"FACT",
  "evidence_support":"MENDUKUNG",
  "logic_check":"LOGIS",
  "confidence":0.9,
  "reason":"",
  "explanation":""
}}
"""
        try:

            res = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            evidence_analysis = json.loads(
                res.choices[0].message.content.strip()
            )

        except Exception as e:

            evidence_analysis["reason"] = "OpenAI error"

            evidence_analysis["explanation"] = str(e)


    # =========================
    # FINAL DECISION
    # =========================

    claim_type = evidence_analysis.get(
        "claim_type", ""
    ).upper()

    support = evidence_analysis.get(
        "evidence_support", ""
    ).upper()

    logic = evidence_analysis.get(
        "logic_check", ""
    ).upper()

    confidence = evidence_analysis.get("confidence")

    warning_message = None


    # ABSURD / TIDAK LOGIS
    if "ABSURD" in claim_type or "TIDAK LOGIS" in logic:

        final_label = "HOAKS"


    # ADA EVIDENCE
    elif evidences and support in [
        "MENDUKUNG",
        "BERTENTANGAN"
    ]:

        # KONSISTEN
        if (
            (ml_label == "VALID" and support == "MENDUKUNG")
            or
            (ml_label == "HOAKS" and support == "BERTENTANGAN")
        ):

            final_label = ml_label


        # KONFLIK
        else:

            if (
                support == "MENDUKUNG"
                and
                "LOGIS" in logic
            ):

                final_label = "VALID"

            elif (
                support == "BERTENTANGAN"
                and
                best_score >= 0.3
            ):

                final_label = "HOAKS"

            else:

                final_label = ml_label


    # TIDAK ADA EVIDENCE
    else:

        final_label = ml_label

        warning_message = (
            "Evidence tidak cukup kuat, "
            "menggunakan hasil model."
        )


    # =========================
    # DEBUG FINAL RESULT
    # =========================

    print("\n========================================")
    print("HASIL AKHIR DETEKSI HOAKS")
    print("========================================")

    print("LABEL SVM :", ml_label)

    print("CONFIDENCE :", ml_confidence, "%")

    print("BEST EVIDENCE SCORE :", round(best_score, 4))

    print("LABEL EVIDENCE:", support )

    print("FINAL LABEL :", final_label)

    print("========================================")


    # RESPONSE
    return jsonify({

        "input_text": claim,

        "final_label": final_label,

        "ml_result": {
            "label": ml_label,
            "confidence": ml_confidence
        },

        "evidence_score": best_score,

        "evidence_confidence_percent": (
            evidence_conf_percent
        ),

        "evidence_analysis": {

            "claim_type": claim_type,

            "support": support,

            "logic_check": logic,

            "confidence": confidence,

            "reason": evidence_analysis.get("reason")
        },

        "openai_explanation": (
            evidence_analysis.get("explanation")
        ),

        "warning_message": warning_message,

        "evidence_used": evidences
    })


if __name__ == "__main__":
    app.run(debug=True)