import requests
import re
from urllib.parse import quote
from urllib.parse import urlparse
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BRAVE_API_KEY = "BSA8iL5ijeMe0wI6sf8ye5KNIEXlyBQ"

TIMEOUT = 5
MAX_CANDIDATES = 40
TOP_ARTICLES = 3


# ==============================
# SUMBER TERPERCAYA
# ==============================

TRUSTED_MEDIA = (
    "kompas.com","detik.com","tempo.co","cnnindonesia.com",
    "antaranews.com","republika.co.id","kumparan.com",
    "tribunnews.com","liputan6.com","bbc.com","reuters.com",
    "aljazeera.com","cnbcindonesia.com","inews.id","tempo.co"
)

FACT_CHECK_SITES = (
    "turnbackhoax.id",
    "cekfakta.com",
    "snopes.com",
    "factcheck.org",
    "politifact.com"
)

GOV_DOMAINS = (
    ".go.id",
    "kominfo.go.id",
    "polri.go.id",
    "kpu.go.id"
)


# ==============================
# TEXT CLEANING
# ==============================

def clean(text):

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==============================
# NORMALISASI KLAIM
# ==============================

def normalize_claim(text):

    text = text.lower()

    text = re.sub(r"\bberedar kabar\b","",text)
    text = re.sub(r"\bdikabarkan\b","",text)
    text = re.sub(r"\bviral\b","",text)

    text = re.sub(r"\s+"," ",text)

    return text.strip()


# ==============================
# EKSTRAK KEYWORD
# ==============================

def extract_keywords(text):

    stopwords = {
        "yang","dan","di","ke","dari","untuk","akan",
        "telah","adalah","pada","ini","itu","oleh",
        "dengan","dalam","para","tersebut","sebuah"
    }

    words = clean(text).split()

    keywords = [
        w for w in words
        if w not in stopwords and len(w) > 3
    ]

    return keywords[:10]


# ==============================
# DETEKSI ENTITAS SEDERHANA
# ==============================

def detect_entities(text):

    entities = []

    # deteksi tahun
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    entities.extend(years)

    # deteksi kata kapital
    words = text.split()

    for w in words:
        if w.istitle():
            entities.append(w.lower())

    return list(set(entities))


# ==============================
# BUILD QUERY CERDAS
# ==============================

def build_queries(claim):

    claim_norm = normalize_claim(claim)

    keywords = extract_keywords(claim_norm)

    entities = detect_entities(claim)

    queries = []

    # klaim asli
    queries.append(claim_norm)

    # keyword search
    if keywords:
        queries.append(" ".join(keywords))

    # phrase search
    queries.append(f'"{claim_norm}"')

    # potong klaim
    words = claim_norm.split()

    if len(words) > 4:
        queries.append(" ".join(words[:4]))
        queries.append(" ".join(words[-4:]))

    # keyword + entity
    if entities:
        queries.append(" ".join(keywords[:4] + entities))

    # query berita
    queries.append(claim_norm + " berita")
    queries.append(claim_norm + " news")

    return list(set(queries))


# ==============================
# TIPE DOMAIN
# ==============================

def domain_type(url):

    domain = urlparse(url).netloc.lower()

    if any(g in domain for g in GOV_DOMAINS):
        return "GOV"

    if any(f in domain for f in FACT_CHECK_SITES):
        return "FACT_CHECK"

    if any(m in domain for m in TRUSTED_MEDIA):
        return "MEDIA"

    return "OTHER"


# ==============================
# GOOGLE SEARCH
# ==============================

def google_search(query):

    url = f"https://api.search.brave.com/res/v1/web/search?q={quote(query)}&count=10"

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }

    try:

        r = requests.get(
            url,
            headers=headers,
            timeout=TIMEOUT
        )

        data = r.json()

        items = data.get("web", {}).get("results", [])

        results = []

        for item in items:

            link = item.get("url", "")

            results.append({
                "url": link,
                "title": item.get("title", ""),
                "snippet": item.get("description", ""),
                "source_type": domain_type(link),
                "published_date": item.get("page_age")
            })

        return results

    except Exception as e:
        print("Brave API Error:", e)
        return []


# ==============================
# MULTI QUERY SEARCH
# ==============================

def search_candidates(claim):

    queries = build_queries(claim)

    results = []
    seen = set()

    for q in queries:

        items = google_search(q)

        for r in items:

            if r["url"] not in seen:

                results.append(r)
                seen.add(r["url"])

    return results[:MAX_CANDIDATES]


# ==============================
# EKSTRAK TANGGAL
# ==============================

def extract_date(snippet):

    if not snippet:
        return None

    patterns = [

        # Feb 22, 2023
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}",

        # 22 Feb 2023
        r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",

        # 2023-02-22
        r"\d{4}-\d{2}-\d{2}",

        # 22/02/2023
        r"\d{1,2}/\d{1,2}/\d{4}"
    ]

    for pattern in patterns:

        match = re.search(pattern, snippet)

        if match:

            date_str = match.group()

            formats = [
                "%b %d, %Y",
                "%d %b %Y",
                "%Y-%m-%d",
                "%d/%m/%Y"
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    pass

    return None


# ==============================
# BOBOT BERITA TERBARU
# ==============================

def recency_weight(date):

    if not date:
        return 0.5

    days = (datetime.now() - date).days

    if days <= 7:
        return 1.0
    if days <= 30:
        return 0.9
    if days <= 365:
        return 0.7
    if days <= 365*3:
        return 0.5

    return 0.3


# ==============================
# RANKING EVIDENCE
# ==============================

def rank_evidences(claim, evidences):

    if not evidences:
        return []

    texts = [clean(claim)]

    for e in evidences:

        texts.append(
            clean(
                e["title"] + " " + e["snippet"]
            )
        )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3)
    )

    tfidf = vectorizer.fit_transform(texts)

    claim_vec = tfidf[0]

    article_vecs = tfidf[1:]

    sims = cosine_similarity(
        claim_vec,
        article_vecs
    )[0]

    for i, e in enumerate(evidences):

        similarity = float(sims[i])

        # =========================
        # AMBIL TANGGAL
        # =========================

        date = None

        if e.get("published_date"):

            try:

                published = str(
                    e["published_date"]
                ).strip()

                # contoh:
                # 2025-10-01T20:15:06
                published = published[:19]

                date = datetime.strptime(
                    published,
                    "%Y-%m-%dT%H:%M:%S"
                )

            except Exception as ex:

                print(
                    "DATE PARSE ERROR:",
                    ex
                )

                date = extract_date(
                    e["snippet"]
                )

        else:

            date = extract_date(
                e["snippet"]
            )

        # =========================
        # RECENCY
        # =========================

        recency = recency_weight(date)

        # =========================
        # CREDIBILITY
        # =========================

        if e["source_type"] == "GOV":

            credibility = 1.0

        elif e["source_type"] == "FACT_CHECK":

            credibility = 0.95

        elif e["source_type"] == "MEDIA":

            credibility = 0.8

        else:

            credibility = 0.5

        # =========================
        # TITLE SIMILARITY
        # =========================

        title_sim = cosine_similarity(
            claim_vec,
            vectorizer.transform([
                clean(e["title"])
            ])
        )[0][0]

        title_boost = title_sim * 0.2

        # =========================
        # FINAL SCORE
        # =========================

        score = (
            similarity * 0.5
            + credibility * 0.25
            + recency * 0.15
            + title_boost
        )

        # =========================
        # SAVE RESULT
        # =========================

        e["similarity"] = similarity
        e["credibility"] = credibility
        e["score"] = score
        e["recency_weight"] = recency
        e["published_date"] = e.get("published_date")

        e["date"] = (
            date.strftime("%Y-%m-%d")
            if date else None
        )

    evidences.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return evidences[:TOP_ARTICLES]


# ==============================
# MAIN FUNCTION
# ==============================

def fetch_evidence_snippets(claim):

    candidates=search_candidates(claim)

    ranked=rank_evidences(claim,candidates)

    return ranked