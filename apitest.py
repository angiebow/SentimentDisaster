import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[0]
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENROUTER_API_KEY")


def test_api():
    if not API_KEY:
        print("OPENROUTER_API_KEY tidak ditemukan. Set di .env atau environment variable.")
        return

    url = "https://openrouter.ai/api/v1/chat/completions"
    prompt = "Classify this text as Positive, Negative, or Neutral: The government responded quickly."

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 20,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "ThesisResearch",
    }

    print("Sending request...")
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print("Status Code:", response.status_code)

    if response.status_code != 200:
        print("\nAPI returned an error:\n", response.text)
        return

    try:
        data = response.json()
        print("\nAPI Response JSON:\n", json.dumps(data, indent=2))
        result = data["choices"][0]["message"]["content"].strip()
        print("\nExtracted Model Output:", result)
    except Exception as e:
        print("\nFailed to parse JSON:", e)
        print(response.text)


if __name__ == "__main__":
    test_api()
