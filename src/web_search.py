import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

TAVILY_URL = "https://api.tavily.com/search"


def get_tavily_api_key(override_key: str = None) -> str:
    """
    Retrieve the Tavily API key from override parameter,
    Streamlit secrets, or environment variables.
    """
    if override_key and override_key.strip():
        return override_key.strip()
    
    # 1. Check environment variables
    for env_var in ["TAVILY_API_KEY", "tavily_api_key", "TAVILY_KEY", "tavily_key"]:
        key = os.getenv(env_var)
        if key and key.strip():
            return key.strip()
    
    # 2. Check Streamlit secrets (Streamlit Cloud & local .streamlit/secrets.toml)
    try:
        import streamlit as st
        # Check standard root-level keys
        for secret_name in ["TAVILY_API_KEY", "tavily_api_key", "TAVILY_KEY", "tavily_key", "API_KEY"]:
            if secret_name in st.secrets:
                val = st.secrets[secret_name]
                if val and str(val).strip():
                    return str(val).strip()
        
        # Check nested dictionary secrets (e.g., [tavily] api_key = "...")
        if "tavily" in st.secrets:
            sec_tavily = st.secrets["tavily"]
            if isinstance(sec_tavily, dict):
                for sub_name in ["api_key", "API_KEY", "key", "KEY", "tavily_api_key"]:
                    if sub_name in sec_tavily:
                        val = sec_tavily[sub_name]
                        if val and str(val).strip():
                            return str(val).strip()
    except Exception:
        pass
    
    return ""


def clean_search_query(query: str) -> str:
    """
    Sanitize and optimize arbitrary natural language claims into effective web search queries.
    Removes conversational prefixes, punctuation clutter, and reduces query bloat.
    """
    if not query:
        return ""
    
    cleaned = query.strip()
    
    # Remove conversational / boilerplate intros
    prefixes_to_strip = [
        r"^(?:in my opinion|according to(?: research| studies| reports)?|it is believed that|it is known that|i think that|fact check(?::| if)?|is it true that|claim(?::|\s*\d+:)?)\s*",
        r"^(?:actually|in fact|specifically|notably|furthermore|moreover),\s*"
    ]
    for pattern in prefixes_to_strip:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    # Clean quotes and bracket noise
    cleaned = re.sub(r'["\'`“”‘’]', "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # If the query is excessively long (> 18 words), extract the most significant content words
    words = cleaned.split()
    if len(words) > 18:
        # Keep the first 16 words for search precision
        cleaned = " ".join(words[:16])
        
    return cleaned.strip()


def search_web(query, max_results=5, max_retries=3, backoff_seconds=1.5, api_key=None):
    """
    Search the web using Tavily with automatic retry, exponential backoff,
    and smart query reformulation fallback.
    """
    resolved_key = get_tavily_api_key(api_key)

    if not resolved_key:
        raise ValueError(
            "TAVILY_API_KEY was not found. "
            "Please provide an API key in the sidebar, .env file, or Streamlit secrets."
        )

    search_query = clean_search_query(query)
    if not search_query:
        search_query = query.strip()

    payload = {
        "api_key": resolved_key,
        "query": search_query,
        "search_depth": "basic",
        "max_results": max_results
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                TAVILY_URL,
                json=payload,
                timeout=25
            )

            if not response.ok:
                raise RuntimeError(
                    f"Tavily API error "
                    f"(HTTP {response.status_code}): "
                    f"{response.text}"
                )

            data = response.json()
            results = data.get("results", [])
            
            # If 0 results were found and query was modified, fallback to original query
            if not results and search_query != query.strip():
                fallback_payload = {
                    "api_key": resolved_key,
                    "query": query.strip(),
                    "search_depth": "basic",
                    "max_results": max_results
                }
                fallback_resp = requests.post(TAVILY_URL, json=fallback_payload, timeout=25)
                if fallback_resp.ok:
                    results = fallback_resp.json().get("results", [])
                    
            return results

        except (requests.exceptions.RequestException, RuntimeError) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (attempt + 1))
            else:
                raise RuntimeError(
                    f"Could not connect to Tavily after {max_retries} attempts: {last_error}"
                )


def search_claim(claim, max_results=5, api_key=None):
    """
    Search the web for evidence related to any arbitrary claim.
    Returns structured list of evidence items with title, url, content, and score.
    """
    results = search_web(
        claim,
        max_results=max_results,
        api_key=api_key
    )

    evidence = []

    for result in results:
        evidence.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "score": float(result.get("score", 0.5) or 0.5)
        })

    return evidence