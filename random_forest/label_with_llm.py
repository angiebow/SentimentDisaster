"""
Job 2: Label sentimen dengan LLM (guru) untuk Random Forest.
Input default: output Job 1 (merged_clean_rf.csv).
Output: labeled_sentiment_data_rf.csv → dipakai Job 3 (training RF).

Contoh run:
poetry run python random_forest/label_with_llm.py ^
  --input random_forest/output_rf/merged_clean_rf.csv ^
  --output random_forest/output_rf/labeled_sentiment_data_rf.csv ^
  --api-key <OPENROUTER_API_KEY>
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def parse_args():
    parser = argparse.ArgumentParser(description="Label sentimen via LLM (guru) untuk RF.")
    parser.add_argument(
        "--input",
        default="random_forest/output_rf/merged_clean_rf.csv",
        help="CSV hasil preprocess (Job 1).",
    )
    parser.add_argument(
        "--output",
        default="random_forest/output_rf/labeled_sentiment_data.csv",
        help="Lokasi keluaran CSV berlabel.",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="API key OpenRouter.")
    parser.add_argument("--model", default="deepseek/deepseek-chat", help="Model LLM.")
    parser.add_argument("--max-rows", type=int, default=None, help="Batasi jumlah baris untuk labeling (opsional).")
    parser.add_argument("--force", action="store_true", help="Abaikan file output lama dan relabel semua data.")
    return parser.parse_args()


def get_llm_label(text: str, api_key: str, model: str) -> Optional[str]:
    prompt = f"""
    Analisis sentimen tanggapan pemerintah dalam berita bencana berikut.
    Klasifikasikan: Positive, Negative, atau Neutral.

    - Negative: respon lambat, keluhan, kurang persiapan/alat, penanganan buruk.
    - Positive: respon cepat, bantuan efektif, apresiasi, peringatan tepat waktu.
    - Neutral: laporan faktual tanpa pujian/keluhan.

    Berikan hanya satu kata: Positive, Negative, atau Neutral.
    Berita:
    {text[:2000]}
    """

    try:
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 30,
                }
            ),
            timeout=60,
        )
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    try:
        result = resp.json()
        raw = result["choices"][0]["message"]["content"].strip().lower()
    except Exception:
        return None

    if "negative" in raw:
        return "Negative"
    if "positive" in raw:
        return "Positive"
    if "neutral" in raw:
        return "Neutral"
    return None


def main():
    args = parse_args()

    if not args.api_key:
        print("API key OpenRouter tidak ditemukan. Set --api-key atau env OPENROUTER_API_KEY.")
        sys.exit(1)

    if os.path.exists(args.output) and not args.force:
        print(f"File output sudah ada. Gunakan --force untuk relabel. Memuat: {args.output}")
        return

    if not os.path.exists(args.input):
        print(f"Input tidak ditemukan: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Memuat data dari {args.input} ...")
    df = pd.read_csv(args.input)
    df = df.dropna(subset=["cleaned_content"])
    if args.max_rows:
        df = df.head(args.max_rows)
    print(f"Total data yang akan dilabel: {len(df)}")

    tqdm.pandas(desc="Labeling LLM")
    df["sentiment_label"] = df["cleaned_content"].progress_apply(
        lambda x: get_llm_label(str(x), args.api_key, args.model)
    )
    df = df.dropna(subset=["sentiment_label"])
    print(f"Sukses diberi label: {len(df)} baris")

    df.to_csv(args.output, index=False)
    print(f"Dataset berlabel disimpan di {args.output}")


if __name__ == "__main__":
    main()
