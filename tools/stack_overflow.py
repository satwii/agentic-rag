import requests

def search_stackoverflow(query):
    url = "https://api.stackexchange.com/2.3/search"
    params = {
        "order": "desc",
        "sort": "relevance",
        "intitle": query,
        "site": "stackoverflow"
    }
    response = requests.get(url, params=params)
    data = response.json()
    print(response.status_code)
    print(response.text)

    results = []
    for item in data.get("items", [])[:5]:  # limit to top 5
        results.append({
            "title": item["title"],
            "url": item["link"]
        })
    return results

