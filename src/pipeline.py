from src.claim_extraction import (
    extract_claims,
    create_chunks
)

from src.retrieval import retrieve_evidence

from src.nli_checker import check_claim


def analyze_answer(
    source_text,
    answer_text,
    embedding_model,
    nli_model
):
    """
    Run the complete hallucination-checking pipeline.

    Steps:
    1. Extract claims from generated answer.
    2. Split source document into chunks.
    3. Retrieve relevant evidence for each claim.
    4. Run NLI classification.
    5. Calculate consistency score.
    """

    claims = extract_claims(answer_text)

    source_chunks = create_chunks(source_text)

    results = []

    for claim in claims:

        evidence, similarity = retrieve_evidence(
            claim,
            source_chunks,
            embedding_model
        )

        if evidence is None:
            continue

        nli_result = check_claim(
            evidence,
            claim,
            nli_model
        )

        label = nli_result["label"].upper()
        confidence = float(nli_result["score"])

        results.append({
            "claim": claim,
            "evidence": evidence,
            "similarity": similarity,
            "label": label,
            "confidence": confidence
        })

    total = len(results)

    supported = sum(
        1
        for result in results
        if result["label"] == "ENTAILMENT"
    )

    contradicted = sum(
        1
        for result in results
        if result["label"] == "CONTRADICTION"
    )

    unsupported = sum(
        1
        for result in results
        if result["label"] == "NEUTRAL"
    )

    consistency_score = (
        supported / total * 100
        if total > 0
        else 0
    )

    summary = {
        "total": total,
        "supported": supported,
        "contradicted": contradicted,
        "unsupported": unsupported,
        "consistency_score": consistency_score
    }

    return results, summary