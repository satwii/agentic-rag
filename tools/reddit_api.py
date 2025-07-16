import requests 
import os
from dotenv import load_dotenv

load_dotenv()

def search_reddit(query):
    print("search reddit() called")

    auth = requests.auth.HTTPBasicAuth(os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET"))  # Fixed: was REDDIT_SECRET

    data = {
        "grant_type": "password",  # Fixed: was "grat_type" in get_token()
        "username": os.getenv("REDDIT_USERNAME"),
        "password": os.getenv("REDDIT_PASSWORD")
    }

    headers = {"User-Agent": "fix-agent/0.1"}

    res = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
    token = res.json().get("access_token")  # Fixed: was "access-token" in get_token()

    print("Got token", token)

    if not token:
        print("Failed to get token")
        print("Response:", res.text)  # Added for debugging
        return []
    
    headers["Authorization"] = f"bearer {token}"
    params = {"q": query, "limit": 5, "sort": "relevance"}

    response = requests.get("https://oauth.reddit.com/r/techsupport/search", headers=headers, params=params)

    print("Reddit status:", response.status_code)
    print("Reddit response:", response.text)

    results = []
    if response.status_code == 200:
        for post in response.json()["data"]["children"]:
            results.append({
                "title": post["data"]["title"],
                "url": f"https://www.reddit.com{post['data']['permalink']}"
            })

    return results

# Fixed version of get_token function
def get_token():
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    SECRET = os.getenv("REDDIT_CLIENT_SECRET")  # Fixed: was REDDIT_SECRET
    USERNAME = os.getenv("REDDIT_USERNAME")
    PASSWORD = os.getenv("REDDIT_PASSWORD")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT")
    
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET)
    data = {
        'grant_type': 'password',  # Fixed: was 'grat_type'
        'username': USERNAME,
        'password': PASSWORD
    }

    headers = {'User-Agent': USER_AGENT}

    res = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
    
    if res.status_code != 200:
        print(f"Error getting token: {res.status_code}")
        print(f"Response: {res.text}")
        return None
    
    token = res.json().get("access_token")  # Fixed: was "access-token"
    return token