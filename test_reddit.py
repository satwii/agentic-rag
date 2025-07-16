from tools.serper_api import search_web

query = "how to fix laptop not charging"
results = search_web(query)

for r in results:
    print("🔧", r["title"])
    print("📄", r["snippet"])
    print("🔗", r["link"])
    print()
