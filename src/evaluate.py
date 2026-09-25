import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.nli_checker import load_nli_model
from src.retrieval import load_embedding_model
from src.pipeline import verify_claim

LABELS = [
    "supported",
    "contradicted",
    "unverifiable"
]


def evaluate_dataset(file_path="data/labeled_test_set.csv"):
    """
    Run full 3-class evaluation across the test dataset.
    Logs metrics, classification report, confusion matrix, and exports results.
    """
    df = pd.read_csv(file_path)

    print("=" * 60)
    print("FACTCHECK AI — FACTUAL CONSISTENCY EVALUATION")
    print("=" * 60)

    print(f"\nTotal test claims: {len(df)}")

    print("\nLoading NLI & Embedding models...")
    nli_model = load_nli_model()
    embedding_model = load_embedding_model()

    predictions = []
    confidences = []
    source_titles = []
    source_urls = []
    similarities = []
    evidence_texts = []

    for i, row in df.iterrows():
        claim = row["claim"]
        expected = str(row["label"]).lower().strip()

        print(
            f"\n[{i + 1}/{len(df)}] "
            f"Checking: {claim}"
        )

        try:
            result = verify_claim(
                claim=claim,
                nli_model=nli_model,
                embedding_model=embedding_model,
                max_results=5
            )

            verdict = result["verdict"].lower()
            conf = float(result["confidence"])
            evidence = result.get("evidence") or {}

            predictions.append(verdict)
            confidences.append(conf)
            source_titles.append(evidence.get("title", ""))
            source_urls.append(evidence.get("url", ""))
            similarities.append(float(evidence.get("similarity", 0.0)))
            evidence_texts.append(evidence.get("content", ""))

            print(
                f"Expected: {expected:<12} | "
                f"Predicted: {verdict:<12} | "
                f"Confidence: {conf:.2f} | "
                f"Similarity: {evidence.get('similarity', 0.0):.2f}"
            )

        except Exception as e:
            print(f"Error during verification: {e}")
            predictions.append("unverifiable")
            confidences.append(0.0)
            source_titles.append("")
            source_urls.append("")
            similarities.append(0.0)
            evidence_texts.append("")

    df["prediction"] = predictions
    df["confidence"] = confidences
    df["source_title"] = source_titles
    df["source_url"] = source_urls
    df["semantic_similarity"] = similarities
    df["evidence"] = evidence_texts

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    y_true = df["label"].str.lower().str.strip()
    y_pred = df["prediction"].str.lower().str.strip()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0
    )

    print("\n")
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"F1 Score : {f1:.2%}")

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=[
                "Supported",
                "Contradicted",
                "Unverifiable"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    print("Confusion Matrix:\n")
    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Supported",
            "Actual Contradicted",
            "Actual Unverifiable"
        ],
        columns=[
            "Pred Supported",
            "Pred Contradicted",
            "Pred Unverifiable"
        ]
    )
    print(cm_df)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    output_file = "data/evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed evaluation results saved to: {output_file}")

    return df


if __name__ == "__main__":
    evaluate_dataset("data/labeled_test_set.csv")