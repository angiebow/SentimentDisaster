from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
LABELED_DATA = ROOT / "shared_data" / "labeled_sentiment_data_unified.csv"

# -----------------------------
# 1. LOAD LABELED DATA
# -----------------------------
df = pd.read_csv(LABELED_DATA)

text_col = "cleaned_content"
label_col = "sentiment_label"

df = df.dropna(subset=[text_col, label_col])

X = df[text_col]
y = df[label_col]

# -----------------------------
# 2. TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. TF-IDF + SVM
# -----------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LinearSVC()
model.fit(X_train_vec, y_train)

# -----------------------------
# 4. PREDICT + CONFUSION MATRIX
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# 5. VISUAL HEATMAP (FIXED)
# -----------------------------
plt.figure(figsize=(8, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"],
    annot_kws={"color": "black", "fontsize": 18}
)

plt.xlabel("Predicted", fontsize=16)
plt.ylabel("True", fontsize=16)
plt.title("Confusion Matrix", fontsize=18)
plt.tight_layout()
plt.savefig("confusion_matrix_fixed.png", dpi=300)
plt.show()
