from transformers import pipeline


MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"


def load_nli_model():
    """
    Load the Natural Language Inference model.
    """

    return pipeline(
        "text-classification",
        model=MODEL_NAME
    )


def check_claim(
    evidence,
    claim,
    nli_model
):
    """
    Compare retrieved evidence with a claim
    and return the NLI prediction.
    """

    result = nli_model(
        f"{evidence} </s></s> {claim}"
    )

    return result[0]