"""Unified training entry-point for SVM and Random Forest students."""
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SHARED_DATA_PATH = ROOT / "shared_data" / "labeled_sentiment_data_unified.csv"
ARTIFACTS_ROOT = ROOT / "artifacts"
SVM_MODEL_PATH = ARTIFACTS_ROOT / "svm" / "svm_sentiment_model_unified.pkl"
SVM_VECTORIZER_PATH = ARTIFACTS_ROOT / "svm" / "tfidf_vectorizer.pkl"
RF_OUTPUT_DIR = ARTIFACTS_ROOT / "random_forest"


def log_dataset_stats(stage: str, labels: pd.Series) -> None:
    """Print class distribution for easier comparison between runs."""
    print(f"\n[{stage}] total samples: {len(labels)}")
    distribution = labels.value_counts().sort_index()
    print(distribution.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SVM or Random Forest student on the unified labeled dataset.")
    parser.add_argument("--model", choices=["svm", "rf"], required=True, help="Model to train.")
    parser.add_argument(
        "--data",
        default=str(SHARED_DATA_PATH),
        help="Path to the labeled dataset (shared between SVM & RF).",
    )
    parser.add_argument("--text-col", default="cleaned_content", help="Name of the text column.")
    parser.add_argument("--label-col", default="sentiment_label", help="Name of the label column.")
    parser.add_argument("--max-features", type=int, default=5000, help="Max TF-IDF features.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")

    # SVM specific outputs
    parser.add_argument(
        "--svm-model-path",
        default=str(SVM_MODEL_PATH),
        help="Output path for SVM model (only used when --model=svm).",
    )
    parser.add_argument(
        "--svm-vectorizer-path",
        default=str(SVM_VECTORIZER_PATH),
        help="Output path for SVM TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--svm-output-dir",
        default=str(ARTIFACTS_ROOT / "svm"),
        help="Directory for SVM artifacts (report/confusion matrix).",
    )
    parser.add_argument("--svm-prefix", default="svm", help="Filename prefix for SVM artifacts.")
    parser.add_argument(
        "--svm-class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Use sklearn's class_weight handling for SVM.",
    )
    parser.add_argument(
        "--svm-oversample-minority",
        action="store_true",
        help="Naive oversampling of minority classes before vectorization (SVM only).",
    )
    parser.add_argument(
        "--svm-legacy-mode",
        action="store_true",
        help="Revert to legacy flow (vectorize seluruh data sebelum split).",
    )

    # Random Forest specific options
    parser.add_argument("--bigrams", action="store_true", help="Include bigrams (RF only).")
    parser.add_argument("--n-estimators", type=int, default=400, help="Number of trees (RF only).")
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "balanced_subsample", "none"],
        default="balanced_subsample",
        help="Class weight strategy for RF.",
    )
    parser.add_argument(
        "--neutral-weight",
        type=float,
        default=1.0,
        help="Manual weight for Neutral class (RF only).",
    )
    parser.add_argument(
        "--oversample-minority",
        action="store_true",
        help="Naive oversampling of minority classes in RF training split.",
    )
    parser.add_argument(
        "--rf-output-dir",
        default=str(RF_OUTPUT_DIR),
        help="Directory for RF artifacts (report/model/vectorizer/confusion matrix).",
    )
    parser.add_argument("--rf-prefix", default="rf", help="Filename prefix for RF artifacts.")
    return parser.parse_args()


def load_dataset(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col, label_col])
    if df.empty:
        raise ValueError("Dataset kosong setelah menghapus NA pada kolom teks/label.")
    return df


def train_svm(args: argparse.Namespace, df: pd.DataFrame):
    texts = df[args.text_col]
    labels = df[args.label_col]
    log_dataset_stats("Dataset", labels)

    if args.svm_legacy_mode:
        if args.svm_oversample_minority:
            print("Peringatan: opsi --svm-oversample-minority diabaikan pada legacy mode.")
        vectorizer = TfidfVectorizer(max_features=args.max_features)
        X = vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=labels,
        )
        log_dataset_stats("Train", y_train)
        log_dataset_stats("Test", y_test)
    else:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=labels,
        )
        log_dataset_stats("Train", y_train)
        log_dataset_stats("Test", y_test)

        if args.svm_oversample_minority:
            train_df = pd.DataFrame({args.text_col: X_train_text, args.label_col: y_train})
            counts = train_df[args.label_col].value_counts()
            target = counts.max()
            balanced_parts = []
            for cls, cnt in counts.items():
                cls_df = train_df[train_df[args.label_col] == cls]
                reps = max(1, target // cnt)
                extra = target - cnt * reps
                duplicated = pd.concat([cls_df] * reps, ignore_index=True)
                if extra > 0:
                    duplicated = pd.concat(
                        [duplicated, cls_df.sample(n=extra, replace=True, random_state=args.random_state)],
                        ignore_index=True,
                    )
                balanced_parts.append(duplicated)
            balanced_df = pd.concat(balanced_parts, ignore_index=True)
            X_train_text = balanced_df[args.text_col]
            y_train = balanced_df[args.label_col]
            log_dataset_stats("Train (oversampled)", y_train)

        vectorizer = TfidfVectorizer(max_features=args.max_features)
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

    class_weight = None if args.svm_class_weight == "none" else args.svm_class_weight
    clf = SVC(kernel="linear", random_state=args.random_state, class_weight=class_weight)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=clf.classes_)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    output_dir = Path(args.svm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.svm_prefix}_classification_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Accuracy: {accuracy:.4f}\n\n")
        fh.write("Classification Report\n")
        fh.write(report)
        fh.write("\nConfusion Matrix\n")
        fh.write(str(cm))

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=clf.classes_,
        yticklabels=clf.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (SVM)")
    plt.tight_layout()
    cm_path = output_dir / f"{args.svm_prefix}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    model_path = Path(args.svm_model_path)
    vectorizer_path = Path(args.svm_vectorizer_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"SVM model saved to {model_path}")
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")
    print(f"SVM report & confusion matrix saved to {output_dir}")


def train_random_forest(args: argparse.Namespace, df: pd.DataFrame):
    X = df[args.text_col]
    y = df[args.label_col]
    log_dataset_stats("Dataset", y)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    log_dataset_stats("Train (raw)", y_train)
    log_dataset_stats("Test", y_test)

    if args.oversample_minority:
        train_df = pd.DataFrame({args.text_col: X_train_raw, args.label_col: y_train})
        counts = train_df[args.label_col].value_counts()
        target = counts.max()
        balanced_parts = []
        for cls, cnt in counts.items():
            df_cls = train_df[train_df[args.label_col] == cls]
            reps = max(1, target // cnt)
            extra = target - cnt * reps
            duplicated = pd.concat([df_cls] * reps, ignore_index=True)
            if extra > 0:
                duplicated = pd.concat(
                    [duplicated, df_cls.sample(n=extra, replace=True, random_state=args.random_state)],
                    ignore_index=True,
                )
            balanced_parts.append(duplicated)
        train_balanced = pd.concat(balanced_parts, ignore_index=True)
        X_train_raw = train_balanced[args.text_col]
        y_train = train_balanced[args.label_col]
        log_dataset_stats("Train (oversampled)", y_train)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2) if args.bigrams else (1, 1),
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight=None if args.class_weight == "none" else args.class_weight,
    )

    sample_weight = None
    if args.neutral_weight != 1.0:
        weight_map = {cls: 1.0 for cls in pd.unique(y_train)}
        if "Neutral" in weight_map:
            weight_map["Neutral"] = args.neutral_weight
        sample_weight = y_train.map(weight_map)

    clf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    output_dir = Path(args.rf_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{args.rf_prefix}_classification_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Accuracy: {accuracy:.4f}\n\n")
        fh.write("Classification Report\n")
        fh.write(report)
        fh.write("\nConfusion Matrix\n")
        fh.write(str(cm))

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clf.classes_,
        yticklabels=clf.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Random Forest)")
    plt.tight_layout()
    cm_path = output_dir / f"{args.rf_prefix}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    model_path = output_dir / f"{args.rf_prefix}_sentiment_model.pkl"
    vect_path = output_dir / f"{args.rf_prefix}_tfidf_vectorizer.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vect_path)

    print(f"RF artifacts saved to {output_dir}")


def main():
    args = parse_args()
    dataset = load_dataset(Path(args.data), args.text_col, args.label_col)

    if args.model == "svm":
        train_svm(args, dataset)
    else:
        train_random_forest(args, dataset)


if __name__ == "__main__":
    main()
