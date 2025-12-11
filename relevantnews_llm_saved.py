import os
import pandas as pd
import requests
import json
import re
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[0]
load_dotenv(ROOT / ".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def is_news(cleaned_content):
    """Check if the cleaned_content is a news article using LLM."""
    if not OPENROUTER_API_KEY:
        return "tidak"
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "user", "content": f"Apakah teks ini adalah berita yang relevan tentang bencana? Jawab 'iya' atau 'tidak': {cleaned_content}"}
            ],
            "temperature": 0.1,
            "max_tokens": 10
        })
    )
    try:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip().lower()
    except Exception as e:
        print("Error parsing response:", e)
        return "tidak"

def extract_disaster_and_location(file_name):
    match = re.search(r"^(\w+)_(\w+)_", file_name)
    if match:
        location, disaster = match.groups()
        return location, disaster
    return None, None

# Main loop
input_folder = "./clean"  # adjust as needed

for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        print(f"Processing file: {file_name}")
        location, disaster = extract_disaster_and_location(file_name)
        print(f"Extracted location: {location}, disaster: {disaster}")

        df = pd.read_csv(os.path.join(input_folder, file_name))
        print(f"Total rows before filtering: {len(df)}")

        valid_rows = []
        for _, row in df.iterrows():
            cleaned = row.get("cleaned_content", "")
            print(f"Checking cleaned_content: {cleaned[:60]}...")
            result = is_news(cleaned)
            print(f"LLM Response: {result}")
            if "iya" in result:
                valid_rows.append(row)

        if valid_rows:
            filtered_df = pd.DataFrame(valid_rows)
            output_file = f"filtered_{file_name}"
            filtered_df.to_csv(output_file, index=False)
            print(f"Saved {len(filtered_df)} valid news rows to {output_file}")
        else:
            print(f"No valid news found in file: {file_name}")
