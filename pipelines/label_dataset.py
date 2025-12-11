"""Shared LLM labeling pipeline for both SVM and Random Forest students."""
import argparse
import glob
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
DEFAULT_PATTERN = str(ROOT / "unified" / "processed_*.csv")
DEFAULT_OUTPUT = ROOT / "shared_data" / "labeled_sentiment_data_unified.csv"
DEFAULT_TEXT_COLUMN = "cleaned_content"

load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sentiment labels via LLM once and reuse for both SVM and RF students."
    )
    parser.add_argument(
        "--input-pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for raw processed CSV files (used when --input-csv is not provided).",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional pre-merged CSV file. Overrides --input-pattern when supplied.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Column name that contains cleaned text content.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the labeled dataset (shared by SVM & RF).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (falls back to environment variable).",
    )
    parser.add_argument("--model", default="deepseek/deepseek-chat", help="LLM model ID to call via OpenRouter.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optionally limit number of rows to label.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing labeled CSV instead of skipping when output already exists.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for each OpenRouter request.",
    )
    return parser.parse_args()


def load_raw_dataframe(pattern: str, text_column: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"Tidak menemukan file yang cocok dengan pola {pattern}. Pastikan sudah menjalankan preprocess_merge."
        )

    dfs = []
    for filename in files:
        try:
            df = pd.read_csv(filename)
            if text_column not in df.columns:
                print(f"Skip {filename} (kolom {text_column} tidak ada)")
                continue
            base = os.path.basename(filename).replace(".csv", "")
            parts = base.split("_")
            if len(parts) >= 4:
                df["Location"] = parts[1]
                df["Disaster"] = parts[2]
                df["Source"] = parts[3]
            dfs.append(df)
        except Exception as exc:
            print(f"Gagal baca {filename}: {exc}")

    if not dfs:
        raise ValueError("Tidak ada dataset yang valid dari pola yang diberikan.")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=[text_column])
    return merged


def read_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        path = Path(args.input_csv)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV tidak ditemukan: {path}")
        df = pd.read_csv(path)
        if args.text_column not in df.columns:
            raise ValueError(f"Kolom {args.text_column} tidak ada di {path}")
        return df.dropna(subset=[args.text_column])

    return load_raw_dataframe(args.input_pattern, args.text_column)


def request_sentiment(text: str, api_key: str, model: str, timeout: int) -> Optional[str]:
    prompt = f"""
    Analisis sentimen respon pemerintah pada berita bencana berikut.
    Jawab hanya dengan satu kata: Positive, Negative, atau Neutral.

    - Negative: respon lambat, keluhan publik, kurang persiapan.
    - Positive: respon cepat, bantuan efektif, apresiasi warga, peringatan tepat waktu.
    - Neutral: laporan faktual tanpa pujian atau keluhan.

    Berita:
    {text[:2000]}
    """

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
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
            timeout=timeout,
        )
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    try:
        raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
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
        print("OPENROUTER_API_KEY tidak ditemukan. Isi di .env atau gunakan --api-key.")
        sys.exit(1)

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"File keluaran sudah ada ({output_path}). Gunakan --force untuk menimpa.")
        return

    print("Memuat data sumber...")
    df = read_input_dataframe(args)
    if args.max_rows:
        df = df.head(args.max_rows)
    print(f"Total baris untuk dilabel: {len(df)}")

    tqdm.pandas(desc="Labeling LLM")
    df["sentiment_label"] = df[args.text_column].progress_apply(
        lambda text: request_sentiment(str(text), args.api_key, args.model, args.request_timeout)
    )
    df = df.dropna(subset=["sentiment_label"])
    print(f"Berhasil memberi label pada {len(df)} baris")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset berlabel tersedia di {output_path}")


if __name__ == "__main__":
    main()
