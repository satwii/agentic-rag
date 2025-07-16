import os
import requests
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def search_web(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "q": query
    }

    response = requests.post(url, json=data, headers=headers)
    results = response.json()

    links = []
    for r in results.get("organic", [])[:5]:  # Get top 5 results
        links.append({
            "title": r.get("title"),
            "link": r.get("link"),
            "snippet": r.get("snippet")
        })
    return links


# this code is working 