from transformers import pipeline

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


def load_nli_model():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None
    )


def check_claim(evidence, claim, nli_model):
    """
    Evaluate the Natural Language Inference relation between evidence and claim.
    Returns predicted label (ENTAILMENT, CONTRADICTION, NEUTRAL), confidence, and full score distribution.
    """
    text = f"{evidence} [SEP] {claim}"
    results = nli_model(
        text,
        truncation=True,
        max_length=512
    )

    # Some transformers versions return:
    # [[{label, score}, ...]]
    if isinstance(results, list) and len(results) > 0:
        results = results[0]

    # Convert labels to a dictionary
    scores = {
        item["label"].upper(): float(item["score"])
        for item in results
    }

    # Handle models that use LABEL_0 / LABEL_1 / LABEL_2
    if "LABEL_0" in scores:

        # DeBERTa NLI convention:
        # LABEL_0 = CONTRADICTION
        # LABEL_1 = ENTAILMENT
        # LABEL_2 = NEUTRAL

        scores = {
            "CONTRADICTION": scores.get("LABEL_0", 0),
            "ENTAILMENT": scores.get("LABEL_1", 0),
            "NEUTRAL": scores.get("LABEL_2", 0)
        }

    label = max(
        scores,
        key=scores.get
    )

    return {
        "label": label,
        "confidence": scores[label],
        "scores": scores
    }