"""
Sentiment Analysis pipeline — DistilBERT edition.

Stack:
  - DistilBERT-SST2  (distilbert-base-uncased-finetuned-sst-2-english, ~67 MB)
      → sentence-level sentiment classification
  - Keyword matching  (replaces MiniLM cosine similarity)
      → fast aspect attribution with no extra model
  - TF-IDF  (scikit-learn, replaces KeyBERT)
      → dynamic aspect phrase extraction

Performance strategy:
  - All sentence scores are pre-computed during analyze_reviews() and stored
    in a '_sentences' column, so get_phone_sentiment_summary() does zero
    further inference (just aggregates from stored data).
  - analyze_reviews() saves results to CACHE_FILE on disk; subsequent app
    restarts load the cache instantly (no model needed at all after first run).
  - get_phone_sentiment_summary() results are cached in _summary_cache so
    switching phones in the same session is an O(1) dict lookup.
"""

import os
import re
import pickle
import string
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

CACHE_FILE = "sentiment_cache.pkl"
DISTILBERT_BATCH_SIZE = 64

# ── Module-level singletons ───────────────────────────────────────────────────

_sentiment_pipe = None          # HuggingFace pipeline
_summary_cache: dict = {}       # {phone_id: summary_dict}

# ── Aspect keyword sets ───────────────────────────────────────────────────────
# Sets allow O(1) lookup per word.  Using stem-friendly short roots so that
# variants (e.g. "photos", "photography") still match.

ASPECT_KEYWORDS = {
    "camera": {
        "camera", "photo", "picture", "image", "lens", "selfie",
        "video", "photography", "portrait", "night", "shot", "zoom",
        "megapixel", "mp", "optical",
    },
    "battery": {
        "battery", "charge", "charging", "power", "drain", "standby",
        "mah", "runtime", "backup", "endurance",
    },
    "performance": {
        "fast", "slow", "speed", "performance", "processor", "smooth",
        "responsive", "lag", "freeze", "heating", "gaming", "snapdragon",
        "dimensity", "multitask", "benchmark", "fluent", "stutter",
    },
    "display": {
        "display", "screen", "resolution", "brightness", "amoled", "lcd",
        "oled", "color", "refresh", "sunlight", "contrast", "vibrant",
        "sharp", "pixel",
    },
    "build_quality": {
        "build", "quality", "design", "durability", "sturdy", "premium",
        "feel", "scratch", "material", "grip", "finish", "plastic",
        "glass", "metal", "weight",
    },
    "value": {
        "price", "value", "worth", "cost", "affordable", "expensive",
        "budget", "cheap", "money", "overpriced", "deal",
    },
}


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model():
    """Lazy-load DistilBERT-SST2 once; return the pipeline singleton."""
    global _sentiment_pipe
    if _sentiment_pipe is None:
        print("Loading DistilBERT sentiment model …")
        from transformers import pipeline as hf_pipeline
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1
        _sentiment_pipe = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            truncation=True,
            max_length=512,
        )
        print(f"  DistilBERT loaded (device={'GPU' if device == 0 else 'CPU'})")
    return _sentiment_pipe


# ── Helpers ───────────────────────────────────────────────────────────────────

def _distilbert_to_compound(result: dict) -> float:
    """
    Map a DistilBERT-SST2 result dict to a compound score in [-1, 1].
    POSITIVE score p  →  2p - 1
    NEGATIVE score p  →  1 - 2p
    """
    p = result["score"]
    if result["label"] == "POSITIVE":
        return float(2 * p - 1)
    return float(1 - 2 * p)


def _batch_score(texts: list) -> list:
    """
    Run DistilBERT over a list of raw texts in mini-batches.
    Returns a parallel list of compound floats.
    """
    pipe = _load_model()
    compounds = []
    for i in range(0, len(texts), DISTILBERT_BATCH_SIZE):
        batch = texts[i: i + DISTILBERT_BATCH_SIZE]
        try:
            results = pipe(batch)
            compounds.extend([_distilbert_to_compound(r) for r in results])
        except Exception as exc:
            print(f"  DistilBERT batch error: {exc}")
            compounds.extend([0.0] * len(batch))
    return compounds


def clean_text(text: str) -> str:
    """Light cleaning for display / legacy column (NOT used as model input)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> list:
    """Split a review into sentences; discard very short fragments."""
    if not isinstance(text, str) or not text.strip():
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents:
        sents = [text.strip()]
    return [s for s in sents if len(s) >= 8]


def _sentence_matches_aspect(sentence: str, keywords: set) -> bool:
    """Return True if any keyword appears as a word-token in the sentence."""
    lower = sentence.lower()
    return any(kw in lower for kw in keywords)


# ── Public API (backward-compatible with previous versions) ───────────────────

def analyze_sentiment(text: str) -> dict:
    """
    Classify sentiment of a single text with DistilBERT.
    Returns {compound, pos, neu, neg} — same keys as the old VADER version.
    """
    if not isinstance(text, str) or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
    pipe = _load_model()
    try:
        result = pipe(text[:512])[0]
        c = _distilbert_to_compound(result)
        pos = max(c, 0.0)
        neg = max(-c, 0.0)
        neu = 1.0 - abs(c)
        return {"compound": c, "pos": pos, "neu": neu, "neg": neg}
    except Exception as exc:
        print(f"analyze_sentiment error: {exc}")
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}


def extract_dynamic_aspects(reviews: list, top_n: int = 10) -> list:
    """
    TF-IDF based keyphrase extraction (replaces KeyBERT).
    Returns [(phrase, relevance_score), …] sorted by relevance.
    """
    corpus = [str(r) for r in reviews if isinstance(r, str) and r.strip()]
    if not corpus:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            max_features=300,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        scores = tfidf_matrix.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[::-1][:top_n]
        return [(terms[i], round(float(scores[i]), 3)) for i in top_indices]
    except Exception as exc:
        print(f"TF-IDF dynamic aspects error: {exc}")
        return []


def analyze_reviews(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full DistilBERT pipeline over the combined phone DataFrame.

    Steps
    -----
    1. Check disk cache — return immediately if valid cache exists.
    2. Pass 1: batch DistilBERT over all review texts → document-level scores.
    3. Pass 2: collect every sentence from every review, batch DistilBERT.
    4. Keyword matching → per-row aspect sentiment columns.
    5. Store per-row sentence lists in '_sentences' for zero-inference summaries.
    6. Save result to disk cache.
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    if os.path.exists(CACHE_FILE):
        print(f"Loading sentiment data from disk cache ({CACHE_FILE}) …")
        try:
            with open(CACHE_FILE, "rb") as fh:
                cached = pickle.load(fh)
            print("  Cache loaded — skipping all inference.")
            return cached
        except Exception as exc:
            print(f"  Cache load failed ({exc}), recomputing …")

    df = combined_df.copy()
    print(f"Running DistilBERT pipeline on {len(df)} records …")

    df["Clean_Review"] = df["Reviews"].apply(clean_text)

    reviews_raw = df["Reviews"].tolist()

    # ── Pass 1: document-level sentiment ─────────────────────────────────────
    print("  Pass 1/2: document-level DistilBERT inference …")
    valid_idxs = [
        i for i, r in enumerate(reviews_raw)
        if isinstance(r, str) and r.strip()
    ]
    valid_texts = [reviews_raw[i][:512] for i in valid_idxs]

    id_to_compound: dict = {}
    for i in range(0, len(valid_texts), DISTILBERT_BATCH_SIZE):
        batch = valid_texts[i: i + DISTILBERT_BATCH_SIZE]
        try:
            results = _load_model()(batch)
            for orig_idx, res in zip(
                valid_idxs[i: i + DISTILBERT_BATCH_SIZE], results
            ):
                id_to_compound[orig_idx] = _distilbert_to_compound(res)
        except Exception as exc:
            print(f"    Batch error: {exc}")

    df["Sentiment_Score"] = [id_to_compound.get(i, 0.0) for i in range(len(df))]
    df["Sentiment"] = df["Sentiment_Score"].apply(lambda x: {"compound": x})
    df["Sentiment_Category"] = df["Sentiment_Score"].apply(
        lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
    )

    # ── Pass 2: sentence-level inference ─────────────────────────────────────
    print("  Pass 2/2: sentence-level DistilBERT inference …")
    all_sents: list = []
    sent_row_idxs: list = []

    for row_idx, review in enumerate(reviews_raw):
        for sent in _split_sentences(review):
            all_sents.append(sent)
            sent_row_idxs.append(row_idx)

    sent_compounds: list = []
    if all_sents:
        sent_compounds = _batch_score(all_sents)

    # ── Aggregate aspect scores + store sentences per row ────────────────────
    row_aspect: dict = defaultdict(lambda: defaultdict(list))
    row_sentences: dict = defaultdict(list)   # {row_idx: [(sent, compound), …]}

    for sent, row_idx, compound in zip(all_sents, sent_row_idxs, sent_compounds):
        row_sentences[row_idx].append((sent, compound))
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if _sentence_matches_aspect(sent, keywords):
                row_aspect[row_idx][aspect].append(compound)

    # Write aspect columns
    for feature in ASPECT_KEYWORDS:
        col = []
        for row_idx in range(len(df)):
            entries = row_aspect[row_idx].get(feature, [])
            col.append(
                float(sum(entries) / len(entries)) if entries else None
            )
        df[f"{feature}_sentiment"] = col

    # Store pre-computed sentences (avoids re-inference in get_phone_sentiment_summary)
    df["_sentences"] = [
        row_sentences.get(i, []) for i in range(len(df))
    ]

    # ── Persist to disk ───────────────────────────────────────────────────────
    print(f"  Saving results to {CACHE_FILE} …")
    try:
        with open(CACHE_FILE, "wb") as fh:
            pickle.dump(df, fh)
        print("  Cache saved.")
    except Exception as exc:
        print(f"  Cache save failed: {exc}")

    print("DistilBERT pipeline complete.")
    return df


def get_phone_sentiment_summary(phone_id: str, analyzed_df: pd.DataFrame) -> dict | None:
    """
    Return a sentiment summary dict for a single phone.

    Uses pre-computed '_sentences' column — zero additional inference.
    Results are cached in _summary_cache for instant repeat calls.

    Return schema (identical to old transformer version):
      {
        'overall':  {positive, negative, neutral, total_reviews},
        'features': {aspect: score, …},
        'positive_reviews': [str, …],
        'negative_reviews': [str, …],
        'dynamic_aspects':  [(phrase, score), …],
      }
    """
    # In-memory cache: same phone = instant lookup
    if phone_id in _summary_cache:
        return _summary_cache[phone_id]

    phone_data = analyzed_df[analyzed_df["Product_ID"] == phone_id]
    if phone_data.empty:
        return None

    # Aggregate pre-computed sentences
    all_scored: list = []
    for sent_list in phone_data["_sentences"]:
        if isinstance(sent_list, list):
            all_scored.extend(sent_list)

    if not all_scored:
        return None

    positive = sorted(
        [(s, c) for s, c in all_scored if c > 0.2], key=lambda x: -x[1]
    )
    negative = sorted(
        [(s, c) for s, c in all_scored if c < -0.2], key=lambda x: x[1]
    )
    neutral_count = sum(1 for _, c in all_scored if abs(c) <= 0.2)

    total = len(all_scored)
    pos_pct = len(positive) / total * 100 if total else 0.0
    neg_pct = len(negative) / total * 100 if total else 0.0
    neu_pct = neutral_count / total * 100 if total else 0.0

    # Feature scores from pre-computed columns
    feature_scores: dict = {}
    for feature in ASPECT_KEYWORDS:
        col = f"{feature}_sentiment"
        if col in phone_data.columns:
            vals = phone_data[col].dropna()
            feature_scores[feature] = float(vals.mean()) if len(vals) > 0 else 0.0
        else:
            feature_scores[feature] = 0.0

    # Dynamic aspects via TF-IDF (fast — no ML model)
    all_reviews = [
        r for r in phone_data["Reviews"].tolist() if isinstance(r, str)
    ]
    dynamic_aspects = extract_dynamic_aspects(all_reviews)

    result = {
        "overall": {
            "positive": pos_pct,
            "negative": neg_pct,
            "neutral": neu_pct,
            "total_reviews": total,
        },
        "features": feature_scores,
        "positive_reviews": [s for s, _ in positive[:10]],
        "negative_reviews": [s for s, _ in negative[:10]],
        "dynamic_aspects": dynamic_aspects,
    }

    _summary_cache[phone_id] = result
    return result
