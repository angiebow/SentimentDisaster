"""
Job 1: gabung & preprocess data untuk Random Forest (tanpa LLM).
Output ini dipakai oleh tahap labeling LLM (Job 2).

Contoh run:
poetry run python random_forest/preprocess_merge.py ^
  --data-dir ./unified ^
  --pattern "processed_*.csv" ^
  --output random_forest/output_rf/merged_clean_rf.csv
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess & merge data untuk RF (tanpa LLM).")
    parser.add_argument("--data-dir", default="./unified", help="Folder sumber processed_*.csv.")
    parser.add_argument("--pattern", default="processed_*.csv", help="Pola file yang akan dibaca.")
    parser.add_argument(
        "--output",
        default="random_forest/output_rf/merged_clean_rf.csv",
        help="Lokasi keluaran CSV hasil gabung/bersih.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Batasi jumlah baris (opsional).")
    return parser.parse_args()


def list_data_files(data_dir: str, pattern: str) -> List[str]:
    return glob.glob(os.path.join(data_dir, pattern))


def extract_metadata_from_filename(filename: str):
    """
    Contoh nama: processed_Klungkung_Banjir_CNN.csv
    Hasil:
        Location=Klungkung, Disaster=Banjir, Source=CNN
    """
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return {
            "Location": parts[1],
            "Disaster": parts[2],
            "Source": parts[3],
        }
    return {}


def load_and_prepare_data(data_dir: str, pattern: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    files = list_data_files(data_dir, pattern)
    if not files:
        print("Tidak menemukan file dengan pola tersebut.")
        return pd.DataFrame()

    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path)
            if "cleaned_content" not in df.columns:
                print(f"Skip (tidak ada cleaned_content): {path}")
                continue
            meta = extract_metadata_from_filename(path)
            for k, v in meta.items():
                df[k] = v
            dfs.append(df)
        except Exception as e:
            print(f"Gagal baca {path}: {e}")

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["cleaned_content"])
    if max_rows:
        merged = merged.head(max_rows)
    return merged


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Memuat data...")
    df = load_and_prepare_data(args.data_dir, args.pattern, args.max_rows)
    print(f"Total data setelah bersih: {len(df)}")

    if df.empty:
        print("Tidak ada data untuk disimpan.")
        return

    df.to_csv(args.output, index=False)
    print(f"Sukses disimpan ke {args.output}")


if __name__ == "__main__":
    main()
