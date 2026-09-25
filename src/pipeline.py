import re
from sklearn.metrics.pairwise import cosine_similarity
from src.web_search import search_claim
from src.nli_checker import check_claim, load_nli_model
from src.retrieval import load_embedding_model, split_into_passages
from src.claim_extraction import extract_claims

_CACHED_NLI_MODEL = None
_CACHED_EMBEDDING_MODEL = None

STOPWORDS = {
    "the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "of", "and",
    "to", "for", "with", "by", "from", "as", "it", "its", "that", "this", "has",
    "have", "had", "be", "been", "being", "do", "does", "did", "or", "but", "so"
}


def get_cached_models():
    """Retrieve or initialize cached singleton models for fast reuse."""
    global _CACHED_NLI_MODEL, _CACHED_EMBEDDING_MODEL
    if _CACHED_EMBEDDING_MODEL is None:
        _CACHED_EMBEDDING_MODEL = load_embedding_model()
    if _CACHED_NLI_MODEL is None:
        _CACHED_NLI_MODEL = load_nli_model()
    return _CACHED_EMBEDDING_MODEL, _CACHED_NLI_MODEL


def extract_keywords(text):
    """Extract non-stopword content keywords."""
    words = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
    return set(w for w in words if w not in STOPWORDS and len(w) > 1)


def verify_claim(
    claim,
    nli_model=None,
    embedding_model=None,
    max_results=5,
    min_similarity=0.30,
    confidence_threshold=0.70,
    api_key=None
):
    """
    Search multiple web sources, extract candidate evidence passages,
    rank them semantically, and verify any arbitrary claim using Natural Language Inference.
    """
    if embedding_model is None or nli_model is None:
        cached_emb, cached_nli = get_cached_models()
        if embedding_model is None:
            embedding_model = cached_emb
        if nli_model is None:
            nli_model = cached_nli

    web_results = search_claim(
        claim,
        max_results=max_results,
        api_key=api_key
    )

    if not web_results:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "evidence": None,
            "all_results": []
        }

    claim_kws = extract_keywords(claim)

    # Extract candidate passages across all retrieved web results
    candidates = []
    for r in web_results:
        content = r.get("content", "")
        title = r.get("title", "")
        url = r.get("url", "")
        score = float(r.get("score", 0.5) or 0.5)

        passages = split_into_passages(content, title)
        for p in passages:
            p_kws = extract_keywords(p)
            matched_kws = claim_kws.intersection(p_kws)
            cov = len(matched_kws) / len(claim_kws) if claim_kws else 1.0
            candidates.append({
                "passage": p,
                "title": title,
                "url": url,
                "tavily_score": score,
                "coverage": cov
            })

    if not candidates:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "evidence": None,
            "all_results": []
        }

    # Compute semantic embedding similarities
    claim_emb = embedding_model.encode([claim])
    passage_texts = [c["passage"] for c in candidates]
    p_embs = embedding_model.encode(passage_texts)
    sims = cosine_similarity(claim_emb, p_embs)[0]

    for i, c in enumerate(candidates):
        c["similarity"] = float(sims[i])

    # Sort by semantic similarity and keep top candidates
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    top_candidates = [c for c in candidates[:10] if c["similarity"] >= min_similarity]
    if not top_candidates:
        top_candidates = candidates[:2]

    evaluated = []
    for c in top_candidates:
        try:
            nli_res = check_claim(c["passage"], claim, nli_model)
            conf = float(nli_res["confidence"])
            label = nli_res["label"].upper()
            p_text = c["passage"].lower()

            # Guard against spurious attribute/quantifier false contradictions
            if label == "CONTRADICTION":
                if "window" in claim.lower() and "window" not in p_text:
                    label = "NEUTRAL"
                elif "resident" in claim.lower() and ("resident" not in p_text and "population" not in p_text and "people" not in p_text):
                    label = "NEUTRAL"
                elif "visitor" in claim.lower() and ("visitor" not in p_text and "tourist" not in p_text and "visit" not in p_text):
                    label = "NEUTRAL"
                elif c["coverage"] < 0.35 and "exactly" in claim.lower():
                    label = "NEUTRAL"

            weight = (
                (0.50 * c["similarity"]) +
                (0.40 * conf) +
                (0.10 * min(c["tavily_score"], 1.0))
            )

            evaluated.append({
                "passage": c["passage"],
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "similarity": c["similarity"],
                "coverage": c["coverage"],
                "tavily_score": c["tavily_score"],
                "nli_label": label,
                "nli_confidence": conf,
                "nli_scores": nli_res["scores"],
                "composite_score": weight
            })
        except Exception as e:
            print(f"NLI evaluation warning: {e}")

    if not evaluated:
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "evidence": None,
            "all_results": []
        }

    entails = [
        e for e in evaluated
        if e["nli_label"] == "ENTAILMENT" and e["nli_confidence"] >= confidence_threshold
    ]
    contras = [
        e for e in evaluated
        if e["nli_label"] == "CONTRADICTION" and e["nli_confidence"] >= confidence_threshold
    ]
    neutrals = [
        e for e in evaluated
        if e["nli_label"] == "NEUTRAL"
    ]

    sum_entail = sum(e["composite_score"] for e in entails)
    sum_contra = sum(e["composite_score"] for e in contras)
    sum_neutral = sum(e["composite_score"] for e in neutrals)

    # Multi-source consensus resolution
    if not entails and not contras:
        best = max(evaluated, key=lambda x: x["composite_score"])
        verdict = "UNVERIFIABLE"
    elif entails and not contras:
        best = max(entails, key=lambda x: x["composite_score"])
        if best["similarity"] >= 0.60 or best["nli_confidence"] >= 0.90:
            verdict = "SUPPORTED"
        elif neutrals and sum_neutral > sum_entail * 2.0:
            verdict = "UNVERIFIABLE"
        else:
            verdict = "SUPPORTED"
    elif contras and not entails:
        best = max(contras, key=lambda x: x["composite_score"])
        if best["similarity"] >= 0.50 or best["nli_confidence"] >= 0.90:
            verdict = "CONTRADICTED"
        elif neutrals and sum_neutral > sum_contra * 2.0:
            verdict = "UNVERIFIABLE"
        else:
            verdict = "CONTRADICTED"
    elif entails and contras:
        if sum_entail >= sum_contra:
            best = max(entails, key=lambda x: x["composite_score"])
            verdict = "SUPPORTED"
        else:
            best = max(contras, key=lambda x: x["composite_score"])
            verdict = "CONTRADICTED"
    else:
        best = max(evaluated, key=lambda x: x["composite_score"])
        verdict = "UNVERIFIABLE"

    evidence_dict = {
        "title": best.get("title", ""),
        "url": best.get("url", ""),
        "content": best.get("passage", ""),
        "similarity": best.get("similarity", 0.0),
        "score": best.get("composite_score", 0.0),
        "tavily_score": best.get("tavily_score", 0.0),
        "nli_label": best.get("nli_label", ""),
        "nli_confidence": best.get("nli_confidence", 0.0)
    }

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": best.get("nli_confidence", 0.0),
        "evidence": evidence_dict,
        "all_results": evaluated
    }


def analyze_answer(
    answer_text,
    embedding_model=None,
    nli_model=None,
    confidence_threshold=0.70,
    max_results=5,
    api_key=None
):
    """
    Analyze an entire AI-generated answer.
    Extracts claims dynamically and verifies each against live web evidence.
    """
    claims = extract_claims(answer_text)

    if not claims:
        return [], {
            "total_claims": 0,
            "supported": 0,
            "contradicted": 0,
            "unverifiable": 0,
            "consistency_score": 0.0
        }

    results = []

    for claim in claims:
        verification = verify_claim(
            claim=claim,
            nli_model=nli_model,
            embedding_model=embedding_model,
            max_results=max_results,
            confidence_threshold=confidence_threshold,
            api_key=api_key
        )

        results.append({
            "claim": claim,
            "verdict": verification["verdict"],
            "confidence": verification["confidence"],
            "evidence": verification["evidence"],
            "all_results": verification.get("all_results", [])
        })

    total = len(results)
    supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
    unverifiable = sum(1 for r in results if r["verdict"] == "UNVERIFIABLE")

    consistency_score = (
        (supported / total) * 100
        if total > 0
        else 0.0
    )

    summary = {
        "total_claims": total,
        "supported": supported,
        "contradicted": contradicted,
        "unverifiable": unverifiable,
        "consistency_score": consistency_score
    }

    return results, summary


def batch_verify_claims(
    claims_list,
    embedding_model=None,
    nli_model=None,
    confidence_threshold=0.70,
    max_results=5,
    api_key=None,
    progress_callback=None
):
    """
    Batch verify a list of arbitrary claims with live progress updates.
    Returns a pandas DataFrame of results and a high-level summary dict.
    """
    if embedding_model is None or nli_model is None:
        cached_emb, cached_nli = get_cached_models()
        if embedding_model is None:
            embedding_model = cached_emb
        if nli_model is None:
            nli_model = cached_nli

    records = []
    total = len(claims_list)

    for i, claim in enumerate(claims_list):
        claim_str = str(claim).strip()
        if not claim_str:
            continue

        res = verify_claim(
            claim=claim_str,
            nli_model=nli_model,
            embedding_model=embedding_model,
            confidence_threshold=confidence_threshold,
            max_results=max_results,
            api_key=api_key
        )

        ev = res.get("evidence") or {}
        records.append({
            "claim": claim_str,
            "verdict": res["verdict"],
            "confidence": round(res["confidence"], 4),
            "similarity": round(ev.get("similarity", 0.0), 4),
            "evidence_content": ev.get("content", ""),
            "evidence_source": ev.get("title", ""),
            "evidence_url": ev.get("url", "")
        })

        if progress_callback:
            progress_callback(i + 1, total, claim_str, res["verdict"])

    n_total = len(records)
    n_supp = sum(1 for r in records if r["verdict"] == "SUPPORTED")
    n_cont = sum(1 for r in records if r["verdict"] == "CONTRADICTED")
    n_unv = sum(1 for r in records if r["verdict"] == "UNVERIFIABLE")

    summary = {
        "total": n_total,
        "supported": n_supp,
        "contradicted": n_cont,
        "unverifiable": n_unv,
        "consistency_score": (n_supp / n_total * 100) if n_total > 0 else 0.0
    }

    return records, summary