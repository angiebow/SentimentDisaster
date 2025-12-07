import argparse
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Gunakan backend non-interaktif agar aman dijalankan headless
matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Train & eval Random Forest (alurnya sama dengan SVM).")
    parser.add_argument(
        "--data",
        default="./output_rf/labeled_sentiment_data_rf.csv",
        # default="../labeled_sentiment_data_rf.csv",
        help="CSV berlabel (hasil preprocess_and_label).",
    )
    parser.add_argument(
        "--output-dir",
        default=".random_forest/output_rf",
        help="Folder keluaran report/model.",
    )
    parser.add_argument("--text-col", default="cleaned_content", help="Kolom teks.")
    parser.add_argument("--label-col", default="sentiment_label", help="Kolom label.")
    parser.add_argument("--max-features", type=int, default=8000, help="Fitur TF-IDF.")
    parser.add_argument("--bigrams", action="store_true", help="Gunakan unigram+bigrams pada TF-IDF.")
    parser.add_argument("--n-estimators", type=int, default=400, help="Jumlah tree RF.")
    parser.add_argument(
        "--class-weight",
        default="balanced_subsample",
        choices=["balanced", "balanced_subsample", None],
        help="Penyesuaian bobot kelas untuk menangani ketidakseimbangan.",
    )
    parser.add_argument(
        "--neutral-weight",
        type=float,
        default=1.0,
        help="Bobot khusus untuk kelas Neutral saat training (mis. 0.7 untuk menurunkan dominasi).",
    )
    parser.add_argument(
        "--oversample-minority",
        action="store_true",
        help="Duplikasi kelas minoritas di data train agar seimbang (tanpa dependensi tambahan).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Porsi test split.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed reproducible.")
    parser.add_argument("--prefix", default="rf", help="Prefix nama file keluaran.")
    return parser.parse_args()


def run_random_forest(
    df,
    text_column,
    label_column,
    output_dir,
    output_prefix,
    max_features,
    use_bigrams,
    n_estimators,
    class_weight,
    neutral_weight,
    oversample_minority,
    test_size,
    random_state,
):
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Kolom {text_column} atau {label_column} tidak ditemukan!")

    before_len = len(df)
    df = df.dropna(subset=[text_column, label_column])
    after_len = len(df)
    if df.empty:
        raise ValueError("Data kosong setelah drop NA pada teks/label.")
    if after_len < before_len:
        print(f"Warning: {before_len - after_len} baris dibuang karena NaN di teks/label (sisa {after_len}).")

    X = df[text_column]
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if oversample_minority:
        # Oversample sederhana: duplikasi kelas hingga jumlah = kelas mayoritas
        import pandas as pd

        train_df = pd.DataFrame({text_column: X_train, label_column: y_train})
        counts = train_df[label_column].value_counts()
        target = counts.max()
        balanced_parts = []
        for cls, cnt in counts.items():
            df_cls = train_df[train_df[label_column] == cls]
            reps = max(1, target // cnt)
            extra = target - cnt * reps
            duplicated = pd.concat([df_cls] * reps, ignore_index=True)
            if extra > 0:
                duplicated = pd.concat([duplicated, df_cls.sample(n=extra, replace=True, random_state=random_state)])
            balanced_parts.append(duplicated)
        train_df_bal = pd.concat(balanced_parts, ignore_index=True)
        X_train = train_df_bal[text_column]
        y_train = train_df_bal[label_column]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2) if use_bigrams else (1, 1),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )
    sample_weight = None
    if neutral_weight != 1.0:
        weight_map = {cls: 1.0 for cls in pd.unique(y_train)}
        if "Neutral" in weight_map:
            weight_map["Neutral"] = neutral_weight
        sample_weight = y_train.map(weight_map)

    clf.fit(X_train_vec, y_train, sample_weight=sample_weight)
    y_pred = clf.predict(X_test_vec)

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{output_prefix}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write(report)
        f.write("\nConfusion Matrix\n")
        f.write(str(cm))

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
    cm_path = os.path.join(output_dir, f"{output_prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    model_path = os.path.join(output_dir, f"{output_prefix}_sentiment_model.pkl")
    vect_path = os.path.join(output_dir, f"{output_prefix}_tfidf_vectorizer.pkl")
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vect_path)

    return {
        "report_path": report_path,
        "cm_path": cm_path,
        "model_path": model_path,
        "vect_path": vect_path,
    }


def main():
    args = parse_args()
    print(f"Membaca data dari {args.data} ...")
    df = pd.read_csv(args.data)
    print(f"Total data: {len(df)}")
    if len(df) == 0:
        print("Tidak ada data yang bisa diproses.")
        return

    print("Training dan evaluasi Random Forest (alurnya mengikuti SVM: TF-IDF -> RF)...")
    outputs = run_random_forest(
        df,
        text_column=args.text_col,
        label_column=args.label_col,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        max_features=args.max_features,
        use_bigrams=args.bigrams,
        n_estimators=args.n_estimators,
        class_weight=args.class_weight,
        neutral_weight=args.neutral_weight,
        oversample_minority=args.oversample_minority,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Selesai. File tersimpan:")
    for k, v in outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
