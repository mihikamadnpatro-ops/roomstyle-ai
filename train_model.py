"""
RoomStyle AI - training and comparison script (FOML project)

Compares four models on the extracted features:
  1. Logistic Regression baseline
  2. MLP on raw scaled features
  3. PCA (95% variance) + MLP
  4. LDA (max 3 components for 4 classes) + MLP

* Split happens FIRST; scaler/PCA/LDA live inside a Pipeline, so they are fitted
  on training data only (no leakage), also inside each CV fold.
* 5-fold stratified CV; best model chosen by CV accuracy, not test accuracy.
* Saves ONE pipeline (style_classifier.pkl) + label_encoder.pkl, and plots in reports/.

Run:  python train_model.py
"""
import os
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42
os.makedirs("reports", exist_ok=True)

# ------------------------------------------------------------------ data
X = np.load("X_features.npy")
y_raw = np.load("y_labels.npy", allow_pickle=True)
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Features:", X.shape, "| Classes:", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")


def make_mlp():
    # alpha = L2 regularisation for a small dataset; no early_stopping
    # (it would waste 10% of an already tiny training set)
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                         alpha=1.0, max_iter=2000, random_state=RANDOM_STATE)


models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
    "MLP (raw features)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", make_mlp())]),
    "PCA (95% var) + MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("clf", make_mlp())]),
    "LDA (3 comp) + MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(n_components=min(3, len(le.classes_) - 1))),
        ("clf", make_mlp())]),
}

# ------------------------------------------------------------------ evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}
for name, pipe in models.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = dict(cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
                         test_acc=acc, time=train_time, y_pred=y_pred)
    print(f"\n=== {name} ===")
    print(f"CV accuracy : {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"Test accuracy: {acc:.3f}   (fit time {train_time:.2f}s)")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print("\n" + "=" * 72)
print(f"{'Model':<24}{'CV acc':>14}{'Test acc':>12}{'Fit time (s)':>16}")
print("-" * 72)
for name, r in results.items():
    print(f"{name:<24}{r['cv_mean']:.3f}+/-{r['cv_std']:.3f}{r['test_acc']:>12.3f}{r['time']:>16.2f}")
print("=" * 72)

n_comp = models["PCA (95% var) + MLP"].named_steps["pca"].n_components_
print(f"\nPCA kept {n_comp} of {X.shape[1]} components (95% variance).")

# ------------------------------------------------------------------ plots
scaler_tmp = StandardScaler().fit(X_train)
Xs_train = scaler_tmp.transform(X_train)

evr = PCA().fit(Xs_train).explained_variance_ratio_
cum = np.cumsum(evr)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(range(1, len(evr) + 1), evr)
ax[0].set(title="Scree plot", xlabel="Principal component", ylabel="Explained variance ratio")
ax[1].plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
ax[1].axhline(0.95, color="r", ls="--", label="95%")
ax[1].axvline(n_comp, color="g", ls="--", label=f"{n_comp} components")
ax[1].set(title="Cumulative explained variance", xlabel="Number of components",
          ylabel="Cumulative variance")
ax[1].legend()
plt.tight_layout(); plt.savefig("reports/pca_scree.png", dpi=150); plt.close()


def scatter2d(Z, labels, title, fname, xl, yl):
    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(le.classes_):
        m = labels == i
        plt.scatter(Z[m, 0], Z[m, 1], label=cls, alpha=0.7, s=25)
    plt.title(title); plt.xlabel(xl); plt.ylabel(yl); plt.legend()
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


scatter2d(PCA(n_components=2).fit_transform(Xs_train), y_train,
          "PCA projection (training data)", "reports/pca_2d.png", "PC1", "PC2")
scatter2d(LinearDiscriminantAnalysis(n_components=2).fit_transform(Xs_train, y_train), y_train,
          "LDA projection (training data)", "reports/lda_2d.png", "LD1", "LD2")

fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4))
for ax_, (name, r) in zip(axes, results.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, r["y_pred"]),
                           display_labels=le.classes_).plot(
        ax=ax_, colorbar=False, xticks_rotation=45)
    ax_.set_title(f"{name}\nacc={r['test_acc']:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig("reports/confusion_matrices.png", dpi=150); plt.close()

names = list(results)
x = np.arange(len(names))
plt.figure(figsize=(8, 4))
plt.bar(x - 0.2, [results[n]["cv_mean"] for n in names], 0.4,
        yerr=[results[n]["cv_std"] for n in names], label="CV accuracy", capsize=3)
plt.bar(x + 0.2, [results[n]["test_acc"] for n in names], 0.4, label="Test accuracy")
plt.xticks(x, names, rotation=15, ha="right", fontsize=8)
plt.ylabel("Accuracy"); plt.title("Model comparison"); plt.legend()
plt.tight_layout(); plt.savefig("reports/model_comparison.png", dpi=150); plt.close()

# ------------------------------------------------------------------ save best
best_name = max(results, key=lambda n: results[n]["cv_mean"])
print(f"\nBest model by cross-validation: {best_name}")
joblib.dump(models[best_name], "style_classifier.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Saved style_classifier.pkl (full pipeline) and label_encoder.pkl; plots in reports/")
