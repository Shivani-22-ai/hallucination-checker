import re
def extract_claims(text):
    """
    Split generated text into individual claims.

    Current implementation uses sentence-level splitting
    as a simple baseline.
    """

    if not text or not text.strip():
        return []

    claims = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        claim.strip()
        for claim in claims
        if claim.strip()
    ]


def create_chunks(text):
    """
    Split the source document into sentence-level chunks.
    """

    if not text or not text.strip():
        return []

    chunks = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]