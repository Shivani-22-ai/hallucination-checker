import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load and return the sentence embedding model."""
    return SentenceTransformer(MODEL_NAME)


def clean_text(text):
    """Clean markdown artifacts, zero-width characters, excessive whitespace, and formatting."""
    if not text:
        return ""
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)  # strip zero-width spaces
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)               # remove markdown links
    text = re.sub(r'#+\s*', '', text)                        # remove markdown headers
    text = re.sub(r'\|.*?\|', '', text)                      # remove table markup
    text = re.sub(r'\[\s*\.\.\.\s*\]', '', text)             # remove ellipsis brackets
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_passages(text, title=""):
    """
    Decompose web content into clean declarative candidate sentences
    and compact multi-sentence passages.
    """
    text = clean_text(text)
    if not text:
        return []

    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw_sentences:
        s = s.strip().strip("[]\"'")
        # Filter out questions, URL fragments, and navigation boilerplate
        if len(s) < 20 or s.endswith('?') or s.startswith('http') or ('-' in s and ' ' not in s):
            continue
        if re.match(r'^(Edit|View|Share|Menu|Cookie|Subscribe|Sign in|Read more|Search|Home)', s, re.I):
            continue
        sentences.append(s)

    passages = []

    for s in sentences:
        passages.append(s)

    # 2-sentence context windows for richer context
    for i in range(len(sentences) - 1):
        combo = f"{sentences[i]} {sentences[i+1]}"
        if len(combo) <= 350:
            passages.append(combo)

    # Whole text if concise
    if 50 <= len(text) <= 350:
        passages.append(text)

    return list(dict.fromkeys(passages))


def rank_evidence_candidates(
    claim,
    search_results,
    embedding_model,
    min_similarity=0.30,
    top_k=8
):
    """
    Extract and rank candidate evidence passages across all search results
    using semantic embedding similarity against the claim.
    """
    if not search_results:
        return []

    candidates = []

    for result in search_results:
        content = result.get("content", "")
        title = result.get("title", "")
        url = result.get("url", "")
        score = float(result.get("score", 0.5) or 0.5)

        passages = split_into_passages(content, title)
        for p in passages:
            candidates.append({
                "passage": p,
                "title": title,
                "url": url,
                "tavily_score": score,
                "full_content": content
            })

    if not candidates:
        return []

    claim_embedding = embedding_model.encode([claim])
    passage_texts = [c["passage"] for c in candidates]
    content_embeddings = embedding_model.encode(passage_texts)

    similarities = cosine_similarity(
        claim_embedding,
        content_embeddings
    )[0]

    for i, c in enumerate(candidates):
        c["similarity"] = float(similarities[i])

    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    filtered = [c for c in candidates[:top_k] if c["similarity"] >= min_similarity]
    return filtered if filtered else candidates[:2]


def retrieve_best_evidence(
    claim,
    search_results,
    embedding_model
):
    """
    Select the single most semantically relevant web result for a claim.
    Maintained for direct backward compatibility.
    """
    ranked = rank_evidence_candidates(
        claim,
        search_results,
        embedding_model,
        top_k=1
    )

    if not ranked:
        return None

    best = ranked[0]
    return {
        "title": best.get("title", ""),
        "url": best.get("url", ""),
        "content": best.get("passage", ""),
        "similarity": best.get("similarity", 0.0),
        "score": best.get("tavily_score", 0.0)
    }