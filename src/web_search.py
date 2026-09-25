import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

TAVILY_URL = "https://api.tavily.com/search"


def search_web(query, max_results=5, max_retries=3, backoff_seconds=1.5):
    """
    Search the web using Tavily with automatic retry and exponential backoff.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY was not found. "
            "Check your .env file."
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                TAVILY_URL,
                json=payload,
                timeout=30
            )

            if not response.ok:
                raise RuntimeError(
                    f"Tavily API error "
                    f"(HTTP {response.status_code}): "
                    f"{response.text}"
                )

            data = response.json()
            return data.get("results", [])

        except (requests.exceptions.RequestException, RuntimeError) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (attempt + 1))
            else:
                raise RuntimeError(
                    f"Could not connect to Tavily after {max_retries} attempts: {last_error}"
                )


def search_claim(claim, max_results=5):
    """
    Search the web for evidence related to a claim.
    Returns structured list of evidence items with title, url, content, and score.
    """
    results = search_web(
        claim,
        max_results=max_results
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