import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# --------------------------------------------------
# Import project modules
# --------------------------------------------------

from src.retrieval import load_embedding_model
from src.nli_checker import load_nli_model
from src.pipeline import analyze_answer


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="LLM Hallucination Checker",
    page_icon="🔎",
    layout="wide"
)


# --------------------------------------------------
# Load models
# --------------------------------------------------

@st.cache_resource
def load_models():

    embedding_model = load_embedding_model()

    nli_model = load_nli_model()

    return embedding_model, nli_model


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🔎 LLM Hallucination Checker")

st.write(
    "Check whether an LLM-generated answer is "
    "supported by a source document."
)


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.subheader("📄 Source Document")

source_text = st.text_area(
    "Paste your source document here",
    height=250,
    placeholder="Enter the source information..."
)


st.subheader("🤖 LLM Generated Answer")

answer_text = st.text_area(
    "Paste the generated answer here",
    height=200,
    placeholder="Enter the LLM-generated answer..."
)


check_button = st.button(
    "🔍 Check Answer",
    type="primary"
)


# --------------------------------------------------
# Analysis
# --------------------------------------------------

if check_button:

    if not source_text.strip() or not answer_text.strip():

        st.warning(
            "Please enter both the source document "
            "and generated answer."
        )

    else:

        with st.spinner(
            "Analyzing the answer..."
        ):

            embedding_model, nli_model = load_models()

            results, summary = analyze_answer(
                source_text,
                answer_text,
                embedding_model,
                nli_model
            )


        if not results:

            st.warning(
                "No claims could be analyzed."
            )

        else:

            # ------------------------------------------
            # Summary
            # ------------------------------------------

            st.subheader("📊 Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Claims",
                    summary["total"]
                )

            with col2:
                st.metric(
                    "Supported",
                    summary["supported"]
                )

            with col3:
                st.metric(
                    "Contradicted",
                    summary["contradicted"]
                )

            with col4:
                st.metric(
                    "Unsupported",
                    summary["unsupported"]
                )


            # ------------------------------------------
            # Consistency score
            # ------------------------------------------

            score = summary["consistency_score"]

            st.progress(
                score / 100,
                text=f"Factual Consistency: {score:.1f}%"
            )

            st.divider()


            # ------------------------------------------
            # Claim-level results
            # ------------------------------------------

            st.subheader("🔎 Claim-by-Claim Analysis")

            for i, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"### Claim {i}"
                )

                st.write(
                    result["claim"]
                )

                st.markdown(
                    "**Retrieved Evidence:**"
                )

                st.info(
                    result["evidence"]
                )

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

                st.divider()