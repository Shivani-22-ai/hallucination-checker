# 🛡️ FactCheck AI

### Universal LLM Factual Consistency & Real-Time Web Evidence Verification Engine

**FactCheck AI** is a production-grade, multi-stage NLP verification system that evaluates the factual consistency of arbitrary AI-generated texts, essays, statements, and bulk datasets against authoritative live web evidence.

Instead of requiring pre-indexed local documents, FactCheck AI dynamically decomposes complex multi-sentence passages into atomic propositions, retrieves live authoritative web sources using Tavily API, semantically ranks candidate evidence passages with dense vector cosine similarity (`all-MiniLM-L6-v2`), and evaluates entailment/contradiction with a cross-encoder Natural Language Inference model (`cross-encoder/nli-deberta-v3-small`).

---

## 🏗️ End-to-End System Pipeline

```text
               Arbitrary AI Text / User Claim / CSV Dataset
                                    ↓
                   1. Atomic Claim Decomposition
             (Multi-Subject & Compound Clause Parser)
                                    ↓
                        2. Live Web Retrieval
                 (Tavily API with Exponential Backoff)
                                    ↓
                    3. Semantic Evidence Ranking
               (Dense Embeddings: all-MiniLM-L6-v2)
                                    ↓
                    4. Cross-Encoder NLI Inference
                 (Model: nli-deberta-v3-small)
                                    ↓
               5. Multi-Source Consensus & Calibration
              (Weight = 0.50*Sim + 0.40*Conf + 0.10*Search)
                                    ↓
                     6. Factual Verdict & Diagnosis
                (SUPPORTED / CONTRADICTED / UNVERIFIABLE)
                                    ↓
                      7. Consistency Score Meter
                                    ↓
            Interactive Streamlit UI / CSV Bulk Export
```

---

## 🌟 Key Capabilities

- 🌐 **Universal Multi-Domain Verification**: Evaluates statements across any field (Medicine, History, Cinema, Physics, Pop Culture, Geography, Astronomy, Technology).
- 📂 **Bulk & Batch Verification**: Upload `.csv` or `.txt` datasets with dozens to hundreds of statements, process with a live progress bar, and download verification reports.
- ✂️ **Atomic Claim Decomposition**: Splits compound sentences with shared subjects (e.g. *"The Eiffel Tower was completed in 1889 and is located in London"*) and multi-subject independent clauses.
- 🧠 **Cross-Encoder NLI + Dense Retrieval**: MiniLM-L6-v2 semantic search isolates the exact premise, followed by DeBERTa-v3 cross-encoder classification.
- ⚖️ **Multi-Source Evidence Consensus**: Prevents single-source anomalies and protects against false contradictions.
- 🔒 **Zero-Leakage Production UI**: Streamlit web dashboard with custom typography, radial consistency meters, and evidence inspection drawers.

---

## 📊 Benchmark Evaluation Metrics

Tested on a curated multi-domain benchmark test set:

| Metric | Baseline | FactCheck AI (Production) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **68.00%** | **88.00%** | **+20.00%** |
| **Weighted Precision** | **69.01%** | **90.77%** | **+21.76%** |
| **Weighted Recall** | **68.00%** | **88.00%** | **+20.00%** |
| **Supported Recall** | **80.00%** | **100.00% (10/10)** | **+20.00%** |
| **Contradicted Recall**| **70.00%** | **100.00% (10/10)** | **+30.00%** |
| **Unverifiable Precision** | **40.00%** | **100.00%** | **+60.00%** |

---

## 🚀 Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended & Free)
1. Fork or push this repository to GitHub: `https://github.com/Shivani-22-ai/hallucination-checker.git`
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New App"** and select:
   - **Repository**: `Shivani-22-ai/hallucination-checker`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
4. In **Advanced Settings → Secrets**, add:
   ```toml
   TAVILY_API_KEY = "your-tavily-api-key"
   ```
5. Click **Deploy**. Your app will be live on a public URL in 2 minutes!

### Option 2: Hugging Face Spaces (Free)
1. Create a Space at **[huggingface.co/spaces](https://huggingface.co/spaces)**.
2. Select **Streamlit** as the SDK.
3. Push the codebase to the Space git remote.
4. In **Settings → Secrets**, set `TAVILY_API_KEY`.

### Option 3: Docker Deployment
```bash
# Build the Docker image
docker build -t factcheck-ai .

# Run container with API key
docker run -p 8501:8501 -e TAVILY_API_KEY="your-tavily-api-key" factcheck-ai
```

---

## 💻 Local Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/Shivani-22-ai/hallucination-checker.git
cd hallucination-checker

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Configure environment (.env)
echo "TAVILY_API_KEY=your_key_here" > .env

# 5. Run application
streamlit run app/streamlit_app.py
```