import os, requests, json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env")
    raise SystemExit(1)

url = "https://api.groq.com/openai/v1/models"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
r = requests.get(url, headers=headers)
print("HTTP Status:", r.status_code)
print(json.dumps(r.json(), indent=2))
