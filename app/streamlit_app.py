import streamlit as st
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


st.set_page_config(
    page_title="Hallucination Checker",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 LLM Hallucination Checker")
st.write("Check whether an LLM-generated answer is supported by a source document.")


# -----------------------------
# Load models
# -----------------------------

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    nli_model = pipeline(
        "text-classification",
        model="cross-encoder/nli-MiniLM2-L6-H768"
    )

    return embedding_model, nli_model


embedding_model, nli_model = load_models()


# -----------------------------
# Claim extraction
# -----------------------------

def extract_claims(text):

    claims = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        claim.strip()
        for claim in claims
        if claim.strip()
    ]


# -----------------------------
# Source chunking
# -----------------------------

def create_chunks(text):

    chunks = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# -----------------------------
# Retrieve evidence
# -----------------------------

def retrieve_evidence(claim, source_chunks):

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
        similarities[best_index]
    )


# -----------------------------
# NLI classification
# -----------------------------

def check_claim(evidence, claim):

    result = nli_model(
        f"{evidence} </s></s> {claim}"
    )

    return result[0]


# -----------------------------
# User input
# -----------------------------

st.subheader("Source Document")

source_text = st.text_area(
    "Paste your source document here",
    height=250
)

st.subheader("LLM Generated Answer")

answer_text = st.text_area(
    "Paste the LLM-generated answer here",
    height=200
)


check_button = st.button(
    "Check Answer",
    type="primary"
)


# -----------------------------
# Complete pipeline
# -----------------------------

if check_button:

    if not source_text or not answer_text:

        st.warning(
            "Please enter both the source document "
            "and generated answer."
        )

    else:

        claims = extract_claims(answer_text)

        source_chunks = create_chunks(source_text)

        results = []

        for claim in claims:

            evidence, similarity = retrieve_evidence(
                claim,
                source_chunks
            )

            nli_result = check_claim(
                evidence,
                claim
            )

            label = nli_result["label"].upper()
            confidence = nli_result["score"]

            results.append({
                "claim": claim,
                "evidence": evidence,
                "similarity": similarity,
                "label": label,
                "confidence": confidence
            })

        # -----------------------------
        # Results Summary
        # -----------------------------

        supported = sum(
            1 for result in results
            if result["label"] == "ENTAILMENT"
        )           

        contradicted = sum(
        1 for result in results
        if result["label"] == "CONTRADICTION"
        )

        unsupported = sum(
            1 for result in results
            if result["label"] == "NEUTRAL"
        )

        total = len(results)

        consistency_score = (
            supported / total * 100
        )

        st.subheader("📊 Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Claims", total)

        with col2:
            st.metric("Supported", supported)

        with col3:
            st.metric("Contradicted", contradicted)

        with col4:
            st.metric("Unsupported", unsupported)

        st.progress(
            consistency_score / 100,
            text=f"Factual Consistency: {consistency_score:.1f}%"
        )

        st.divider()

        st.subheader("Results")
        # -----------------------------
        # Results
        # -----------------------------

        st.subheader("Results")

        for i, result in enumerate(results, 1):

            st.markdown(f"### Claim {i}")

            st.write(result["claim"])

            st.markdown("**Retrieved Evidence:**")

            st.info(result["evidence"])

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Similarity",
                    f"{result['similarity']:.2f}"
                )

            with col2:
                st.metric(
                    "NLI Confidence",
                    f"{result['confidence']:.2f}"
                )

            label = result["label"]

            if label == "ENTAILMENT":

                st.success(
                    "🟢 Supported by evidence"
                )

            elif label == "CONTRADICTION":

                st.error(
                    "🔴 Contradicted by evidence"
                )

            else:

                st.warning(
                    "🟡 Not supported by evidence"
                )


        # -----------------------------
        # Consistency score
        # -----------------------------

        st.subheader("Overall Consistency")

        st.metric(
            "Factual Consistency Score",
            f"{consistency_score:.1f}%"
        )