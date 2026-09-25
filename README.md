# 🔎 FactCheck AI

### LLM Factual Consistency & Automated Web Evidence Verification System

**FactCheck AI** is an end-to-end AI/ML pipeline that evaluates the factual accuracy of AI-generated answers against live web evidence. Instead of requiring users to manually upload reference documents, FactCheck AI automatically decomposes responses into discrete factual propositions, queries the web via Tavily, ranks retrieved evidence using dense semantic vector embeddings (`all-MiniLM-L6-v2`), and evaluates entailment/contradiction with a Natural Language Inference cross-encoder (`cross-encoder/nli-deberta-v3-small`).

---

## 🏗️ Architecture & Pipeline

```text
               AI-Generated Answer
                       ↓
         1. Claim Extraction & Decomposition
       (Multi-Subject & Compound Clause Parser)
                       ↓
            2. Live Web Evidence Search
               (Tavily API with Auto-Retry)
                       ↓
        3. Semantic Passage Extraction & Ranking
         (Dense Embeddings: all-MiniLM-L6-v2)
                       ↓
         4. Natural Language Inference (NLI)
       (Cross-Encoder: nli-deberta-v3-small)
                       ↓
     5. Multi-Source Consensus & Composite Scoring
   (Score = 0.50*Sim + 0.40*Confidence + 0.10*Search)
                       ↓
       6. Claim-Level Verdict & Explanation
       (SUPPORTED / CONTRADICTED / UNVERIFIABLE)
                       ↓
        7. Overall Factual Consistency Score
                       ↓
          Interactive Streamlit Web Dashboard
```

---

## 🚀 Key Features & Robustness Enhancements

- **Multi-Clause & Compound Claim Decomposition**: Decomposes complex compound sentences with shared subjects (e.g. *"The Eiffel Tower was completed in 1889 and is located in London"*) as well as independent multi-subject clauses (e.g. *"Python was designed by Guido van Rossum and C++ was developed by Bjarne Stroustrup"*).
- **Markdown & Bullet Filtering**: Automatically parses markdown lists, bullet points (`*`, `-`, `•`), numbered enumerations (`1.`, `2)`), and strips standalone section titles.
- **Dense Vector Semantic Ranking**: Embeds and ranks clean candidate passages with `all-MiniLM-L6-v2` cosine similarity.
- **Multi-Source Evidence Consensus**: Prevents noisy or sensationalized web page titles from causing false entailments by requiring consensus across candidate evidence snippets.
- **Attribute & Predicate Mismatch Guard**: Prevents false contradictions on unverified numerical claims when the evidence discusses an unrelated attribute (e.g. *"120 antennas"* vs *"1,247 windows"*).
- **Zero-Width & Unicode Sanitization**: Automatically strips non-printable control characters, zero-width spaces (`\u200b`), and formatting artifacts.
- **Network Resilience**: Automatic retry with exponential backoff on network disconnections or rate limits.

---

## 📊 Evaluation Benchmark Results

Evaluated on the standardized **25-claim benchmark** across 3 classes (Supported, Contradicted, Unverifiable):

| Metric | Baseline | FactCheck AI (Enhanced) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **68.00%** | **88.00%** | **+20.00%** |
| **Weighted Precision** | **69.01%** | **90.77%** | **+21.76%** |
| **Weighted Recall** | **68.00%** | **88.00%** | **+20.00%** |
| **Weighted F1 Score** | **68.35%** | **86.21%** | **+17.86%** |
| **Supported Recall** | **80.00%** | **100.00% (10/10)** | **+20.00%** |
| **Supported Precision**| **89.00%** | **100.00%** | **+11.00%** |
| **Contradicted Recall**| **70.00%** | **100.00% (10/10)** | **+30.00%** |
| **Unverifiable Precision** | **40.00%** | **100.00%** | **+60.00%** |

### Confusion Matrix

| Actual \ Predicted | Pred Supported | Pred Contradicted | Pred Unverifiable |
| :--- | :---: | :---: | :---: |
| **Actual Supported** | **10** | 0 | 0 |
| **Actual Contradicted** | 0 | **10** | 0 |
| **Actual Unverifiable** | 0 | 3 | **2** |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Tested with Python 3.14 on Windows)
- Tavily Search API Key

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 💻 Usage

### Run the Streamlit Web Application
```powershell
streamlit run app/streamlit_app.py
```

### Run the Benchmark Evaluation
```powershell
python -m src.evaluate
```

Results and detailed metrics will be displayed in the terminal and saved to [`data/evaluation_results.csv`](file:///c:/Users/chand/Desktop/AIML%20Learning/hallucination-checker/data/evaluation_results.csv).

---

## 📂 Repository Structure

```text
hallucination-checker/
├── app/
│   └── streamlit_app.py        # Streamlit interactive UI
├── data/
│   ├── labeled_test_set.csv    # 25-claim benchmark dataset
│   └── evaluation_results.csv  # Logged evaluation metrics and predictions
├── src/
│   ├── __init__.py
│   ├── claim_extraction.py     # Independent & compound clause decomposition
│   ├── web_search.py           # Tavily web search with auto-retry
│   ├── retrieval.py            # Passage splitting, sanitization & ranking
│   ├── nli_checker.py          # DeBERTa-v3 Natural Language Inference
│   ├── pipeline.py             # Consensus evidence verification engine
│   └── evaluate.py             # Automated benchmarking suite
├── requirements.txt            # Dependencies
├── .env                        # API credentials (gitignored)
└── README.md                   # Documentation
```