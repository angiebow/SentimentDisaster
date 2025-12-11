"""Shared preprocess+merge job for Random Forest and SVM students."""
import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "unified"
DEFAULT_PATTERN = "processed_*.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "random_forest" / "merged_clean_rf.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess & merge processed_*.csv files into one dataset.")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Folder sumber processed_*.csv (umumnya ./unified).",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Pola nama file yang akan dibaca.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Lokasi keluaran CSV hasil gabung/bersih.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Batasi jumlah baris (opsional).")
    return parser.parse_args()


def list_data_files(data_dir: str, pattern: str) -> List[str]:
    return glob.glob(os.path.join(data_dir, pattern))


def extract_metadata_from_filename(filename: str):
    """Tambahkan kolom metadata dari nama file (Location, Disaster, Source)."""
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
        except Exception as exc:
            print(f"Gagal baca {path}: {exc}")

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["cleaned_content"])
    if max_rows:
        merged = merged.head(max_rows)
    return merged


def main() -> None:
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
