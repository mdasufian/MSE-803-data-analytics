"""
MSE-803 Week 5 Activity 1 — SVM Classifier for Three Classes (Iris dataset)
---------------------------------------------------------------------------
Builds a Support Vector Machine that separates the three iris species:
  0. Iris-setosa
  1. Iris-versicolor
  2. Iris-virginica

Steps:
  1. Load and explore the data
  2. Split into train / test (stratified, so all 3 classes stay balanced)
  3. Scale the features (SVMs are distance based, so scaling matters)
  4. Compare kernels (linear, RBF, polynomial) with cross-validation
  5. Tune C and gamma with GridSearchCV
  6. Evaluate on the held-out test set (accuracy, per-class report, confusion matrix)
  7. Plot the decision boundaries using the two petal features

Kept simple for a master's data analytics class.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to file instead of opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "iris" / "iris.data"  # UCI raw file (no header row)
OUTPUT_DIR = BASE_DIR / "outputs"

COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
FEATURES = COLUMNS[:-1]
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Two features used only for the 2-D decision-boundary picture
PLOT_FEATURES = ["petal_length", "petal_width"]

TEST_SIZE = 0.2      # 30 flowers held back for testing (10 per species)
RANDOM_STATE = 42    # fixed seed so the results repeat exactly
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# 1. Load and explore
# ---------------------------------------------------------------------------
def load_data():
    """Read the raw UCI iris file into a DataFrame."""
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    df = df.dropna()                      # the raw file ends with a blank line
    df["species"] = df["species"].str.strip()
    return df


def explore(df):
    """Print a short summary of the dataset."""
    print("=" * 70)
    print("1. DATA EXPLORATION")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print("\nClass distribution (3 classes):")
    print(df["species"].value_counts().sort_index().to_string())
    print("\nFeature summary:")
    print(df[FEATURES].describe().round(2).to_string())
    print("\nMean of each feature per species:")
    print(df.groupby("species")[FEATURES].mean().round(2).to_string())
    print()


# ---------------------------------------------------------------------------
# 2-3. Split and build the model pipeline
# ---------------------------------------------------------------------------
def make_pipeline(**svc_params):
    """Scaler + SVM in one object, so scaling is learned from the training fold only."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(random_state=RANDOM_STATE, **svc_params)),
        ]
    )


def split_data(df):
    """Stratified 80/20 split so each species keeps the same proportion."""
    X = df[FEATURES]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("=" * 70)
    print("2. TRAIN / TEST SPLIT")
    print("=" * 70)
    print(f"Training set: {len(X_train)} flowers")
    print(f"Test set:     {len(X_test)} flowers")
    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index().to_string())
    print()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. Compare kernels
# ---------------------------------------------------------------------------
def compare_kernels(X_train, y_train):
    """Cross-validate the three usual SVM kernels and print a comparison table."""
    print("=" * 70)
    print("3. KERNEL COMPARISON (5-fold cross-validation on training data)")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    kernels = {
        "linear": {"kernel": "linear", "C": 1.0},
        "rbf": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "poly (deg 3)": {"kernel": "poly", "degree": 3, "C": 1.0, "gamma": "scale"},
    }

    rows = []
    for label, params in kernels.items():
        scores = cross_val_score(make_pipeline(**params), X_train, y_train, cv=cv, scoring="accuracy")
        rows.append({"kernel": label, "mean_accuracy": scores.mean(), "std": scores.std()})

    table = pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)
    print(table.round(4).to_string(index=False))
    print(f"\nBest kernel by cross-validation: {table.iloc[0]['kernel']}")
    print()
    return table


# ---------------------------------------------------------------------------
# 5. Tune the hyper-parameters
# ---------------------------------------------------------------------------
def tune_model(X_train, y_train):
    """Grid search over kernel, C and gamma. Returns the fitted best model."""
    print("=" * 70)
    print("4. HYPER-PARAMETER TUNING (GridSearchCV)")
    print("=" * 70)

    param_grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10, 100]},
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.01, 0.1, 1],
        },
        {
            "svm__kernel": ["poly"],
            "svm__C": [0.1, 1, 10],
            "svm__degree": [2, 3],
            "svm__gamma": ["scale"],
        },
    ]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(make_pipeline(), param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"Combinations tested: {len(search.cv_results_['params'])}")
    print(f"Best parameters:     {search.best_params_}")
    print(f"Best CV accuracy:    {search.best_score_:.4f}")
    print()
    return search.best_estimator_


# ---------------------------------------------------------------------------
# 6. Evaluate on the test set
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    """Accuracy, per-class precision/recall/F1, and a confusion-matrix figure."""
    print("=" * 70)
    print("5. TEST-SET EVALUATION")
    print("=" * 70)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}  ({int(round(accuracy * len(y_test)))}/{len(y_test)} correct)\n")

    print("Classification report (all three classes):")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    # Save the report as CSV so it can be pasted into the write-up
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=3
    )
    pd.DataFrame(report).transpose().round(3).to_csv(OUTPUT_DIR / "classification_report.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("SVM confusion matrix — Iris test set")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    misses = [
        (i, actual, pred)
        for i, (actual, pred) in enumerate(zip(y_test, y_pred))
        if actual != pred
    ]
    print(f"\nMisclassified flowers: {len(misses)}")
    for i, actual, pred in misses:
        print(f"  test row {i}: actual = {actual}, predicted = {pred}")
    print()
    return accuracy


# ---------------------------------------------------------------------------
# 7. Decision boundary picture (2 features so it can be drawn)
# ---------------------------------------------------------------------------
def plot_decision_boundaries(df):
    """Train a separate 2-feature SVM per kernel and draw the class regions."""
    X = df[PLOT_FEATURES].to_numpy()
    y = df["species"].to_numpy()
    y_code = np.searchsorted(CLASS_NAMES, y)  # 0 / 1 / 2 for colouring

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    kernels = [
        ("Linear kernel", {"kernel": "linear", "C": 1.0}),
        ("RBF kernel", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
        ("Polynomial kernel (deg 3)", {"kernel": "poly", "degree": 3, "C": 1.0, "gamma": "scale"}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (title, params) in zip(axes, kernels):
        model = make_pipeline(**params).fit(X, y)
        zz = np.searchsorted(CLASS_NAMES, model.predict(grid)).reshape(xx.shape)

        ax.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5, 2.5], cmap="viridis")
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_code, cmap="viridis", edgecolor="k", s=45)
        ax.set_title(f"{title}\ntraining accuracy = {model.score(X, y):.3f}")
        ax.set_xlabel("petal length (cm)")
    axes[0].set_ylabel("petal width (cm)")
    axes[-1].legend(scatter.legend_elements()[0], CLASS_NAMES, loc="lower right", fontsize=8)

    fig.suptitle("SVM decision boundaries for the three iris classes (petal features)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "decision_boundaries.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'decision_boundaries.png'}")


def plot_feature_pairs(df):
    """Scatter of the two feature pairs, coloured by species — shows why petals separate best."""
    y_code = np.searchsorted(CLASS_NAMES, df["species"].to_numpy())
    pairs = [("sepal_length", "sepal_width"), ("petal_length", "petal_width")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (fx, fy) in zip(axes, pairs):
        scatter = ax.scatter(df[fx], df[fy], c=y_code, cmap="viridis", edgecolor="k", s=45)
        ax.set_xlabel(f"{fx.replace('_', ' ')} (cm)")
        ax.set_ylabel(f"{fy.replace('_', ' ')} (cm)")
        ax.set_title(f"{fx} vs {fy}")
    axes[1].legend(scatter.legend_elements()[0], CLASS_NAMES, loc="lower right", fontsize=8)

    fig.suptitle("Iris features by species")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_scatter.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'feature_scatter.png'}")


# ---------------------------------------------------------------------------
# 8. Predict a few new flowers
# ---------------------------------------------------------------------------
def predict_new_flowers(model):
    """Show the trained model classifying three unseen measurements."""
    print("=" * 70)
    print("7. PREDICTING NEW FLOWERS")
    print("=" * 70)

    new_flowers = pd.DataFrame(
        [
            [5.0, 3.4, 1.5, 0.2],   # looks like setosa
            [6.0, 2.7, 4.2, 1.3],   # looks like versicolor
            [6.7, 3.0, 5.8, 2.1],   # looks like virginica
        ],
        columns=FEATURES,
    )
    predictions = model.predict(new_flowers)
    for (_, row), pred in zip(new_flowers.iterrows(), predictions):
        measurements = ", ".join(f"{c}={row[c]}" for c in FEATURES)
        print(f"  {measurements}  ->  {pred}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    explore(df)

    X_train, X_test, y_train, y_test = split_data(df)
    compare_kernels(X_train, y_train)

    best_model = tune_model(X_train, y_train)
    evaluate(best_model, X_test, y_test)

    svm = best_model.named_steps["svm"]
    print(f"Support vectors per class: {dict(zip(CLASS_NAMES, svm.n_support_.tolist()))}")
    print(f"Total support vectors: {svm.n_support_.sum()} out of {len(X_train)} training flowers")
    print("(3 classes are handled with one-vs-one: 3 binary SVMs are combined by voting.)\n")

    print("=" * 70)
    print("6. FIGURES")
    print("=" * 70)
    plot_feature_pairs(df)
    plot_decision_boundaries(df)
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"Saved: {OUTPUT_DIR / 'classification_report.csv'}\n")

    predict_new_flowers(best_model)
    print("Done.")


if __name__ == "__main__":
    main()
