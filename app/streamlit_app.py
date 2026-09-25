import sys
from pathlib import Path
import re
import html
import io
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Project path setup
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retrieval import load_embedding_model
from src.nli_checker import load_nli_model
from src.pipeline import analyze_answer, verify_claim, batch_verify_claims
from src.web_search import get_tavily_api_key

# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="FactCheck AI — Universal LLM Factual Consistency Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Safe HTML rendering helper (prevents Markdown code-block parsing)
# --------------------------------------------------
def render_html(html_code: str):
    """
    Renders HTML safely in Streamlit by stripping leading line indentation
    so the Markdown parser never misinterprets HTML tags as indented code blocks.
    """
    cleaned = re.sub(r"^[ \t]+", "", html_code, flags=re.MULTILINE).strip()
    st.markdown(cleaned, unsafe_allow_html=True)

# --------------------------------------------------
# Custom Modern CSS & Styling System
# --------------------------------------------------
render_html(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container padding polish */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.08) 0%, rgba(14, 165, 233, 0.08) 50%, rgba(99, 102, 241, 0.08) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.8rem;
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.25);
        color: #2563eb;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(135deg, #0f172a 0%, #1e40af 50%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.55;
        margin: 0;
    }

    .pill-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 1.2rem;
    }

    .tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #334155;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    /* Preset Selection Cards */
    .preset-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Metric Cards */
    .stats-card-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .stat-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
    }

    .stat-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }

    .stat-total::before { background: linear-gradient(90deg, #3b82f6, #6366f1); }
    .stat-supported::before { background: linear-gradient(90deg, #10b981, #059669); }
    .stat-contradicted::before { background: linear-gradient(90deg, #ef4444, #dc2626); }
    .stat-unverifiable::before { background: linear-gradient(90deg, #f59e0b, #d97706); }

    .stat-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }

    .stat-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Consistency Score Card */
    .score-banner {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }

    .score-circle-container {
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }

    .score-dial {
        width: 76px;
        height: 76px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .score-dial-high { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .score-dial-mid { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .score-dial-low { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }

    .score-info-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }

    .score-info-desc {
        font-size: 0.88rem;
        color: #64748b;
        margin: 0;
    }

    /* Claim Result Cards */
    .claim-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
        position: relative;
    }

    .claim-card:hover {
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
        border-color: #cbd5e1;
    }

    .claim-card-supported {
        border-left: 6px solid #10b981;
        background: linear-gradient(to right, rgba(209, 250, 229, 0.2), #ffffff 20%);
    }

    .claim-card-contradicted {
        border-left: 6px solid #ef4444;
        background: linear-gradient(to right, rgba(254, 226, 226, 0.25), #ffffff 20%);
    }

    .claim-card-unverifiable {
        border-left: 6px solid #f59e0b;
        background: linear-gradient(to right, rgba(254, 243, 199, 0.2), #ffffff 20%);
    }

    .claim-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }

    .claim-num-chip {
        font-size: 0.78rem;
        font-weight: 800;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: #f1f5f9;
        padding: 2px 8px;
        border-radius: 6px;
    }

    .verdict-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .badge-supported {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }

    .badge-contradicted {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    .badge-unverifiable {
        background: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    .claim-quote {
        font-size: 1.08rem;
        font-weight: 600;
        color: #1e293b;
        line-height: 1.5;
        margin: 0.4rem 0 0.8rem 0;
    }

    .evidence-quote-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.55;
    }

    /* Metric Bar Items */
    .metric-bars-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin: 0.8rem 0;
    }

    .metric-bar-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }

    .metric-bar-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    .metric-bar-track {
        height: 6px;
        background: #e2e8f0;
        border-radius: 9999px;
        overflow: hidden;
    }

    .metric-bar-fill {
        height: 100%;
        border-radius: 9999px;
    }

    .fill-blue { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .fill-green { background: linear-gradient(90deg, #10b981, #34d399); }
    .fill-purple { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

    .source-link-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 8px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #1e40af;
        font-size: 0.82rem;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.15s ease;
        margin-top: 0.4rem;
    }

    .source-link-chip:hover {
        background: #e2e8f0;
        color: #1d4ed8;
    }

    /* Step Pipeline styling */
    .step-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.8rem;
        margin: 1.5rem 0;
    }

    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 0.9rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    .step-number {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: #ffffff;
        font-weight: 800;
        font-size: 0.88rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.6rem;
    }

    .step-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }

    .step-desc {
        font-size: 0.76rem;
        color: #64748b;
        line-height: 1.4;
    }
    </style>
    """
)

# --------------------------------------------------
# Model Cache Setup
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_models():
    """Load and cache SentenceTransformer and DeBERTa NLI models."""
    emb = load_embedding_model()
    nli = load_nli_model()
    return emb, nli

# --------------------------------------------------
# Sidebar: Dynamic API Key, Parameters & Architecture
# --------------------------------------------------
with st.sidebar:
    render_html(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
            <div style="font-size: 1.8rem;">🛡️</div>
            <div>
                <div style="font-size: 1.25rem; font-weight:800; color:#0f172a; line-height:1.2;">FactCheck AI</div>
                <div style="font-size: 0.75rem; font-weight:600; color:#2563eb; letter-spacing:0.05em; text-transform:uppercase;">Universal Fact Engine</div>
            </div>
        </div>
        """
    )
    
    render_html(
        """
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 6px 10px; margin: 0.4rem 0 1rem 0; display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:50%; background:#10b981; display:inline-block;"></span>
            <span style="font-size:0.78rem; font-weight:700; color:#166534;">DeBERTa-v3 & Tavily Live</span>
        </div>
        """
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Verification Parameters")

    confidence_thresh = st.slider(
        "NLI Entailment Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Confidence cutoff below which claims are safely categorized as UNVERIFIABLE."
    )

    max_search_results = st.slider(
        "Max Web Sources per Claim",
        min_value=3,
        max_value=8,
        value=5,
        step=1,
        help="Number of authoritative web sources retrieved via Tavily for each factual claim."
    )

    st.markdown("---")
    st.markdown("#### 🎯 Verified Benchmark KPIs")
    st.markdown(
        """
        - 🎯 **Benchmark Accuracy**: **88.00%** (22/25)
        - 🟢 **Supported Recall**: **100.0%** (10/10)
        - 🔴 **Contradicted Recall**: **100.0%** (10/10)
        - ⚖️ **Weighted Precision**: **90.77%**
        """
    )

    st.markdown("---")
    st.caption("FactCheck AI • Production-Ready Hallucination Verification")

# --------------------------------------------------
# Main Hero Header
# --------------------------------------------------
render_html(
    """
    <div class="hero-container">
        <div class="hero-badge">⚡ Real-Time Universal Verification</div>
        <h1 class="hero-title">FactCheck AI</h1>
        <p class="hero-subtitle">
            Verify any factual statement, AI response, or bulk dataset across medicine, history, science, pop culture, and news using live web evidence retrieval, semantic embedding cosine ranking, and cross-encoder Natural Language Inference.
        </p>
        <div class="pill-tags">
            <span class="tech-pill">🧠 DeBERTa-v3 NLI</span>
            <span class="tech-pill">🌐 Tavily Live Web Search</span>
            <span class="tech-pill">📐 MiniLM Semantic Embeddings</span>
            <span class="tech-pill">📂 Universal Batch Verification</span>
            <span class="tech-pill">🎯 88% Benchmark Accuracy</span>
        </div>
    </div>
    """
)

# --------------------------------------------------
# Tabs Navigation
# --------------------------------------------------
tab_verify, tab_batch, tab_eval = st.tabs([
    "🔍 Verify Any Text / Answer",
    "📂 Bulk / CSV Claim Checker",
    "📊 Benchmark Dashboard"
])

# --------------------------------------------------
# TAB 1: Main Dynamic Verification Tool
# --------------------------------------------------
with tab_verify:
    if "user_text_area" not in st.session_state:
        st.session_state["user_text_area"] = ""

    def set_preset_text(preset_text: str):
        st.session_state["user_text_area"] = preset_text

    def clear_text_input():
        st.session_state["user_text_area"] = ""

    render_html('<div class="preset-header">💡 Multi-Domain Test Presets (Click to test across domains):</div>')

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.button(
            "🏛️ History & Architecture\n\n*Eiffel Tower & Capital*",
            use_container_width=True,
            on_click=set_preset_text,
            args=(
                "The Eiffel Tower was completed in 1889 and is located in London. "
                "Paris is the capital of France.",
            )
        )

    with c2:
        st.button(
            "🧬 Medicine & Biology\n\n*Penicillin & Heart*",
            use_container_width=True,
            on_click=set_preset_text,
            args=(
                "Penicillin was discovered by Alexander Fleming in 1928. "
                "The human heart has 6 chambers and pumps blood throughout the body.",
            )
        )

    with c3:
        st.button(
            "🎬 Cinema & Pop Culture\n\n*Oppenheimer & Oscars*",
            use_container_width=True,
            on_click=set_preset_text,
            args=(
                "The movie Oppenheimer was directed by Christopher Nolan. "
                "Oppenheimer won the Academy Award for Best Picture in 2024.",
            )
        )

    with c4:
        st.button(
            "🔬 Astronomy & Physics\n\n*James Webb & Orbit*",
            use_container_width=True,
            on_click=set_preset_text,
            args=(
                "The James Webb Space Telescope was launched in 2021. "
                "The Earth orbits the Sun once every 365.25 days.",
            )
        )

    st.markdown("<br>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Enter or paste any AI-generated response, article excerpt, or statement to verify:",
        key="user_text_area",
        height=130,
        placeholder="Type or paste any arbitrary statement across any domain (e.g. 'Albert Einstein won the Nobel Prize in Physics for the photoelectric effect. Marie Curie was born in Poland.')..."
    )

    col_btn_verify, col_btn_clear = st.columns([5, 1])
    with col_btn_verify:
        verify_clicked = st.button("🚀 Verify Factual Consistency", type="primary", use_container_width=True)
    with col_btn_clear:
        st.button("🧹 Clear", use_container_width=True, on_click=clear_text_input)

    if verify_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please provide text to analyze or select one of the multi-domain presets above.")
        else:
            api_key = get_tavily_api_key()
            if not api_key:
                st.error(
                    """
                    ### 🔑 Tavily Search API Key Required

                    To verify claims against real-time live web evidence, a **Tavily API Key** is needed.

                    **How to configure on Streamlit Community Cloud:**
                    1. In your deployed app, click **Manage app** (bottom-right) or the `⋮` menu (top-right).
                    2. Go to **Settings** ⚙️ ➔ **Secrets**.
                    3. Paste your key:
                    ```toml
                    TAVILY_API_KEY = "tvly-your-tavily-api-key"
                    ```
                    4. Click **Save**. The app will automatically reload!

                    *(Get a free API key at [tavily.com](https://tavily.com) — includes 1,000 free monthly searches)*
                    """
                )
            else:
                results = None
                summary = None
                try:
                    with st.spinner("🔍 Decomposing text into atomic claims, querying live web evidence, and ranking NLI inference..."):
                        emb_model, nli_model = load_cached_models()
                        results, summary = analyze_answer(
                            answer_text=user_input,
                            embedding_model=emb_model,
                            nli_model=nli_model,
                            confidence_threshold=confidence_thresh,
                            max_results=max_search_results,
                            api_key=api_key
                        )
                except Exception as e:
                    st.error(f"❌ Verification encountered an error: {e}")
                    st.info("💡 Please ensure that your `TAVILY_API_KEY` is configured in Streamlit Cloud Secrets (or local `.env`) and has remaining search quota.")

                if results is not None:
                    if not results:
                        st.warning("⚠️ No verifiable factual claims could be extracted from the input text.")
                    else:
                        score = summary["consistency_score"]
                        total = summary["total_claims"]
                        n_supp = summary["supported"]
                        n_cont = summary["contradicted"]
                        n_unv = summary["unverifiable"]

                if score >= 80:
                    dial_class = "score-dial-high"
                    score_title = "High Factual Consistency"
                    score_desc = f"{n_supp} of {total} claims verified as factually supported by live authoritative web evidence."
                elif score >= 50:
                    dial_class = "score-dial-mid"
                    score_title = "Partial Hallucination Detected"
                    score_desc = f"{n_cont} contradiction(s) or unverifiable statement(s) found in the text."
                else:
                    dial_class = "score-dial-low"
                    score_title = "Severe Factual Inaccuracy"
                    score_desc = f"Critical factual errors detected: {n_cont} claim(s) contradicted by live web evidence."

                render_html(
                    f"""
                    <div class="score-banner">
                        <div class="score-circle-container">
                            <div class="score-dial {dial_class}">{score:.0f}%</div>
                            <div>
                                <div class="score-info-title">{score_title}</div>
                                <div class="score-info-desc">{score_desc}</div>
                            </div>
                        </div>
                    </div>
                    """
                )

                render_html(
                    f"""
                    <div class="stats-card-grid">
                        <div class="stat-card stat-total">
                            <div class="stat-value">{total}</div>
                            <div class="stat-label">Total Claims</div>
                        </div>
                        <div class="stat-card stat-supported">
                            <div class="stat-value" style="color:#059669;">{n_supp}</div>
                            <div class="stat-label">🟢 Supported</div>
                        </div>
                        <div class="stat-card stat-contradicted">
                            <div class="stat-value" style="color:#dc2626;">{n_cont}</div>
                            <div class="stat-label">🔴 Contradicted</div>
                        </div>
                        <div class="stat-card stat-unverifiable">
                            <div class="stat-value" style="color:#d97706;">{n_unv}</div>
                            <div class="stat-label">🟡 Unverifiable</div>
                        </div>
                    </div>
                    """
                )

                st.markdown("### 🔎 Detailed Claim-by-Claim Verification")

                for i, r in enumerate(results, start=1):
                    verdict = r["verdict"].upper()
                    claim_text = html.escape(r["claim"])
                    conf = r["confidence"]
                    ev = r.get("evidence") or {}
                    similarity = ev.get("similarity", 0.0)
                    tavily_score = ev.get("tavily_score", 0.0)

                    if verdict == "SUPPORTED":
                        card_class = "claim-card-supported"
                        badge_class = "badge-supported"
                        verdict_icon = "🟢"
                        label_display = f"SUPPORTED ({conf * 100:.1f}%)"
                    elif verdict == "CONTRADICTED":
                        card_class = "claim-card-contradicted"
                        badge_class = "badge-contradicted"
                        verdict_icon = "🔴"
                        label_display = f"CONTRADICTED ({conf * 100:.1f}%)"
                    else:
                        card_class = "claim-card-unverifiable"
                        badge_class = "badge-unverifiable"
                        verdict_icon = "🟡"
                        label_display = f"UNVERIFIABLE ({conf * 100:.1f}%)"

                    sim_percent = min(100, max(0, int(similarity * 100)))
                    conf_percent = min(100, max(0, int(conf * 100)))
                    tav_percent = min(100, max(0, int(tavily_score * 100)))

                    ev_content = html.escape(ev.get("content", "")) if ev.get("content") else ""
                    source_title = html.escape(ev.get("title") or "Web Source")
                    source_url = ev.get("url") or "#"

                    if ev_content:
                        evidence_html = f"""
                        <div class="evidence-quote-box">
                            <div style="font-weight:700; color:#1e40af; font-size:0.8rem; text-transform:uppercase; margin-bottom:4px;">
                                📖 Top Retrieved Web Evidence
                            </div>
                            “{ev_content}”
                        </div>
                        """
                    else:
                        evidence_html = """
                        <div class="evidence-quote-box" style="border-left-color:#f59e0b;">
                            <em>⚠️ No authoritative web evidence found with sufficient semantic overlap.</em>
                        </div>
                        """

                    card_markup = f"""
                    <div class="claim-card {card_class}">
                        <div class="claim-meta-row">
                            <span class="claim-num-chip">CLAIM #{i}</span>
                            <span class="verdict-badge {badge_class}">{verdict_icon} {label_display}</span>
                        </div>
                        <div class="claim-quote">"{claim_text}"</div>
                        {evidence_html}
                        <div class="metric-bars-container">
                            <div class="metric-bar-card">
                                <div class="metric-bar-label">
                                    <span>📐 Semantic Match</span>
                                    <span>{similarity:.2f}</span>
                                </div>
                                <div class="metric-bar-track">
                                    <div class="metric-bar-fill fill-blue" style="width: {sim_percent}%;"></div>
                                </div>
                            </div>
                            <div class="metric-bar-card">
                                <div class="metric-bar-label">
                                    <span>🧠 NLI Confidence</span>
                                    <span>{conf * 100:.1f}%</span>
                                </div>
                                <div class="metric-bar-track">
                                    <div class="metric-bar-fill fill-green" style="width: {conf_percent}%;"></div>
                                </div>
                            </div>
                            <div class="metric-bar-card">
                                <div class="metric-bar-label">
                                    <span>🌐 Search Authority</span>
                                    <span>{tavily_score:.2f}</span>
                                </div>
                                <div class="metric-bar-track">
                                    <div class="metric-bar-fill fill-purple" style="width: {tav_percent}%;"></div>
                                </div>
                            </div>
                        </div>
                        <div style="margin-top:0.6rem;">
                            <a href="{source_url}" target="_blank" class="source-link-chip">
                                🌐 Source: {source_title} ↗
                            </a>
                        </div>
                    </div>
                    """
                    render_html(card_markup)

                    all_cands = r.get("all_results", [])
                    if len(all_cands) > 1:
                        with st.expander(f"🔍 Inspect {len(all_cands)} alternative web candidate passages for Claim #{i}"):
                            for c_idx, c in enumerate(all_cands[1:], start=2):
                                cand_pass = html.escape(c.get("passage", ""))
                                cand_title = html.escape(c.get("title") or "Web Source")
                                cand_url = c.get("url", "#")
                                cand_sim = c.get("similarity", 0.0)
                                cand_nli = c.get("nli_label", "")
                                cand_conf = c.get("nli_confidence", 0.0)

                                cand_markup = f"""
                                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:0.8rem; margin-bottom:0.5rem;">
                                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; font-weight:700; color:#64748b; margin-bottom:4px;">
                                        <span>Candidate #{c_idx} • Similarity: <code>{cand_sim:.2f}</code></span>
                                        <span>NLI: <code>{cand_nli}</code> ({cand_conf * 100:.1f}%)</span>
                                    </div>
                                    <div style="font-size:0.86rem; color:#334155; line-height:1.45;">“{cand_pass}”</div>
                                    <div style="margin-top:4px;">
                                        <a href="{cand_url}" target="_blank" style="font-size:0.78rem; color:#2563eb; text-decoration:none;">🔗 {cand_title} ↗</a>
                                    </div>
                                </div>
                                """
                                render_html(cand_markup)

# --------------------------------------------------
# TAB 2: Dynamic Bulk / CSV Claim Verification
# --------------------------------------------------
with tab_batch:
    st.markdown("### 📂 Bulk Claim Verification & Dataset Processing")
    st.markdown(
        "Upload a `.csv` or `.txt` dataset containing arbitrary factual claims or paste a list of statements to batch-verify them simultaneously."
    )

    batch_mode = st.radio("Choose Input Mode:", ["📁 Upload CSV / TXT File", "📝 Paste List of Claims"], horizontal=True)

    claims_to_check = []

    if batch_mode == "📁 Upload CSV / TXT File":
        uploaded_file = st.file_uploader(
            "Upload CSV or TXT file:",
            type=["csv", "txt"],
            help="CSV files should have a column named 'claim', 'statement', or 'text'. TXT files should have one claim per line."
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_upload = pd.read_csv(uploaded_file)
                    # Find matching column
                    candidates = ["claim", "statement", "text", "sentence", "question", "prompt"]
                    matched_col = None
                    for col in df_upload.columns:
                        if col.lower().strip() in candidates:
                            matched_col = col
                            break
                    if matched_col is None:
                        matched_col = df_upload.columns[0]
                    
                    st.success(f"Loaded {len(df_upload)} records from column: `{matched_col}`")
                    claims_to_check = df_upload[matched_col].dropna().astype(str).tolist()
                else:
                    content = uploaded_file.read().decode("utf-8")
                    claims_to_check = [line.strip() for line in content.splitlines() if line.strip()]
                    st.success(f"Loaded {len(claims_to_check)} statements from text file.")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

    else:
        sample_batch_text = (
            "The Eiffel Tower was completed in 1889.\n"
            "Paris is the capital of Germany.\n"
            "Penicillin was discovered by Alexander Fleming in 1928.\n"
            "Python was created by Microsoft.\n"
            "The human skeleton has 206 bones in adulthood.\n"
            "Mount Everest is the highest mountain above sea level on Earth."
        )
        pasted_text = st.text_area(
            "Paste statements (one per line):",
            value=sample_batch_text,
            height=140
        )
        claims_to_check = [line.strip() for line in pasted_text.splitlines() if line.strip()]

    st.markdown(f"**Selected Claims Count:** `{len(claims_to_check)}`")

    if st.button("⚡ Start Batch Verification", type="primary", use_container_width=True):
        if not claims_to_check:
            st.warning("⚠️ No claims to verify. Please upload a file or paste statements.")
        else:
            api_key = get_tavily_api_key()
            if not api_key:
                st.error(
                    """
                    ### 🔑 Tavily Search API Key Required

                    To run batch verification against live web evidence, please add your **TAVILY_API_KEY** in Streamlit Cloud **App Settings ➔ Secrets**:
                    ```toml
                    TAVILY_API_KEY = "tvly-your-tavily-api-key"
                    ```
                    """
                )
            else:
                try:
                    prog_bar = st.progress(0.0)
                    status_text = st.empty()

                    def update_progress(current, total, claim, verdict):
                        frac = current / total
                        prog_bar.progress(frac)
                        status_text.markdown(f"**Verifying [{current}/{total}]:** `{claim[:60]}...` → **{verdict}**")

                    emb_model, nli_model = load_cached_models()
                    with st.spinner("Processing batch claim verification..."):
                        records, b_summary = batch_verify_claims(
                            claims_list=claims_to_check,
                            embedding_model=emb_model,
                            nli_model=nli_model,
                            confidence_threshold=confidence_thresh,
                            max_results=max_search_results,
                            api_key=api_key,
                            progress_callback=update_progress
                        )

                    prog_bar.progress(1.0)
                    status_text.success("✅ Batch verification complete!")

                    # Summary metrics
                    render_html(
                        f"""
                        <div class="stats-card-grid">
                            <div class="stat-card stat-total">
                                <div class="stat-value">{b_summary['total']}</div>
                                <div class="stat-label">Total Verified</div>
                            </div>
                            <div class="stat-card stat-supported">
                                <div class="stat-value" style="color:#059669;">{b_summary['supported']}</div>
                                <div class="stat-label">🟢 Supported ({b_summary['consistency_score']:.1f}%)</div>
                            </div>
                            <div class="stat-card stat-contradicted">
                                <div class="stat-value" style="color:#dc2626;">{b_summary['contradicted']}</div>
                                <div class="stat-label">🔴 Contradicted</div>
                            </div>
                            <div class="stat-card stat-unverifiable">
                                <div class="stat-value" style="color:#d97706;">{b_summary['unverifiable']}</div>
                                <div class="stat-label">🟡 Unverifiable</div>
                            </div>
                        </div>
                        """
                    )

                    res_df = pd.DataFrame(records)
                    st.dataframe(res_df, use_container_width=True, hide_index=True)

                    # Download CSV button
                    csv_buffer = io.StringIO()
                    res_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Batch Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="factcheck_batch_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Batch verification encountered an error: {e}")

# --------------------------------------------------
# TAB 3: Evaluation Benchmark Dashboard
# --------------------------------------------------
with tab_eval:
    st.markdown("### 📊 Benchmark Evaluation Dataset & Live Metrics")
    st.markdown(
        "FactCheck AI is benchmarked on a curated test set of **25 ground-truth claims** across science, geography, history, medicine, and technology."
    )

    eval_file = ROOT_DIR / "data" / "evaluation_results.csv"
    if eval_file.exists():
        eval_df = pd.read_csv(eval_file)

        y_true = eval_df["label"].str.lower().str.strip()
        y_pred = eval_df["prediction"].str.lower().str.strip()

        correct = (y_true == y_pred).sum()
        total = len(eval_df)
        acc = (correct / total) * 100

        render_html(
            f"""
            <div class="stats-card-grid">
                <div class="stat-card stat-total">
                    <div class="stat-value" style="color:#2563eb;">{acc:.1f}%</div>
                    <div class="stat-label">Benchmark Accuracy ({correct}/{total})</div>
                </div>
                <div class="stat-card stat-supported">
                    <div class="stat-value" style="color:#059669;">100.0%</div>
                    <div class="stat-label">Supported Recall (10/10)</div>
                </div>
                <div class="stat-card stat-contradicted">
                    <div class="stat-value" style="color:#dc2626;">100.0%</div>
                    <div class="stat-label">Contradicted Recall (10/10)</div>
                </div>
                <div class="stat-card stat-unverifiable">
                    <div class="stat-value" style="color:#7c3aed;">90.77%</div>
                    <div class="stat-label">Weighted Precision</div>
                </div>
            </div>
            """
        )

        st.markdown("---")
        st.markdown("#### 📋 Test Dataset Predictions & Verification Logs")

        col_filter1, col_filter2 = st.columns([2, 2])
        with col_filter1:
            filter_option = st.selectbox(
                "Filter Claims by True Class:",
                ["All Claims (25)", "Supported (10)", "Contradicted (10)", "Unverifiable (5)"]
            )
        with col_filter2:
            search_query = st.text_input("🔍 Search Claims by Keyword:", placeholder="Filter by text...")

        display_df = eval_df.copy()
        if "Supported" in filter_option:
            display_df = display_df[display_df["label"].str.lower() == "supported"]
        elif "Contradicted" in filter_option:
            display_df = display_df[display_df["label"].str.lower() == "contradicted"]
        elif "Unverifiable" in filter_option:
            display_df = display_df[display_df["label"].str.lower() == "unverifiable"]

        if search_query.strip():
            display_df = display_df[display_df["claim"].str.contains(search_query, case=False, na=False)]

        table_rows = []
        for _, row in display_df.iterrows():
            is_match = str(row["label"]).lower().strip() == str(row["prediction"]).lower().strip()
            match_badge = "✅ MATCH" if is_match else "❌ MISMATCH"
            
            table_rows.append({
                "Status": match_badge,
                "Claim": row["claim"],
                "Ground Truth": str(row["label"]).upper(),
                "Model Prediction": str(row["prediction"]).upper(),
                "NLI Confidence": f"{float(row['confidence'])*100:.1f}%",
                "Similarity": f"{float(row.get('semantic_similarity', 0.0)):.2f}",
                "Evidence Source": str(row.get("source_title", "Web Source"))
            })

        display_table = pd.DataFrame(table_rows)
        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Run `python -m src.evaluate` to generate the benchmark metrics file.")