from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """

    return SentenceTransformer(MODEL_NAME)


def retrieve_evidence(
    claim,
    source_chunks,
    embedding_model
):
    """
    Retrieve the source chunk that is most
    semantically similar to the claim.
    """

    if not source_chunks:
        return None, 0.0

    claim_embedding = embedding_model.encode([claim])

    chunk_embeddings = embedding_model.encode(
        source_chunks
    )

    similarities = cosine_similarity(
        claim_embedding,
        chunk_embeddings
    )[0]

    best_index = similarities.argmax()

    return (
        source_chunks[best_index],
        float(similarities[best_index])
    )