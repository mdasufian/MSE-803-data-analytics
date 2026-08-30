# MSE-803 Week 5 — Activity 1: SVM Classifier (Three Classes)

Trains a **Support Vector Machine** to classify the three iris species in Fisher's classic Iris dataset, compares kernels, tunes the hyper-parameters, and evaluates the model on a held-out test set.

## Files

| File | Description |
|------|-------------|
| `svm_iris_classifier.ipynb` | **Notebook version** — same pipeline with all charts and tables rendered inline |
| `svm_iris_classifier.py` | Script version: explore → split → kernel comparison → tuning → evaluation → plots |
| `../iris/iris.data` | Raw UCI Iris dataset (150 rows, no header) |
| `outputs/feature_scatter.png` | Sepal vs petal scatter plots coloured by species |
| `outputs/decision_boundaries.png` | Decision regions for linear / RBF / polynomial kernels |
| `outputs/confusion_matrix.png` | Confusion matrix on the test set (best overall model) |
| `outputs/confusion_matrix_all_kernels.png` | Confusion matrix per kernel (linear, RBF, poly), side by side |
| `outputs/classification_report.csv` | Per-class precision, recall, F1 |

The notebook adds four views the script does not save: box plots of each feature by species, a bar chart of the kernel comparison, an RBF `C` × `gamma` accuracy heatmap, and a plot with the support vectors circled.

## How to run

**Notebook (recommended — charts appear inline):**

```bash
cd week-5/activity-1
jupyter notebook svm_iris_classifier.ipynb
```
Or open `svm_iris_classifier.ipynb` in VS Code and click **Run All**.

**Script (saves the figures to `outputs/`):**

```bash
cd week-5/activity-1
python3 svm_iris_classifier.py
```

**Requirements:** Python 3 with `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.
The notebook's first cell (`%pip install ...`) installs them into the active kernel if any are missing.

---

## 1. The data

150 flowers, 4 numeric features, **3 balanced classes** (50 each):

| Species | sepal_length | sepal_width | petal_length | petal_width |
|---------|--------------|-------------|--------------|-------------|
| Iris-setosa | 5.01 | 3.42 | 1.46 | 0.24 |
| Iris-versicolor | 5.94 | 2.77 | 4.26 | 1.33 |
| Iris-virginica | 6.59 | 2.97 | 5.55 | 2.03 |

No missing values. The petal measurements separate the species far better than the sepal ones — see `outputs/feature_scatter.png`.

---

## 2. Method

| Step | Choice | Why |
|------|--------|-----|
| Split | 80 / 20, **stratified** | 120 train / 30 test, 10 of each species in the test set |
| Scaling | `StandardScaler` inside a `Pipeline` | SVMs use distances, so features must share a scale; the pipeline keeps the scaler from seeing the test fold |
| Multi-class | `SVC` default **one-vs-one** | 3 binary SVMs (setosa–versicolor, setosa–virginica, versicolor–virginica) combined by voting |
| Validation | 5-fold stratified CV | Small dataset, so a single split is unreliable |
| Seed | `random_state=42` | Reproducible results |

---

## 3. Kernel comparison (5-fold CV on training data)

| Kernel | Mean accuracy | Std |
|--------|---------------|-----|
| **linear** | **0.9750** | 0.0333 |
| rbf | 0.9667 | 0.0167 |
| poly (deg 3) | 0.9000 | 0.0425 |

The linear kernel wins: after scaling, the three classes are almost linearly separable, so the extra flexibility of RBF/polynomial only adds variance.

---

## 4. Hyper-parameter tuning

`GridSearchCV` over 26 combinations of kernel, `C`, `gamma`, and `degree`.

- **Best parameters:** `kernel='linear'`, `C=0.1`
- **Best CV accuracy:** 0.9750

A small `C` means a **wide, soft margin** — the model tolerates a few training errors instead of contorting the boundary around them, which generalises better here.

---

## 5. Test-set results

**Accuracy: 0.9333 (28 / 30 correct)**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Iris-setosa | 1.000 | 1.000 | 1.000 | 10 |
| Iris-versicolor | 0.900 | 0.900 | 0.900 | 10 |
| Iris-virginica | 0.900 | 0.900 | 0.900 | 10 |
| **Accuracy** | | | **0.933** | 30 |

Confusion matrix (rows = actual, columns = predicted):

|  | setosa | versicolor | virginica |
|--|--------|------------|-----------|
| **setosa** | 10 | 0 | 0 |
| **versicolor** | 0 | 9 | 1 |
| **virginica** | 0 | 1 | 9 |

Only **2 errors**, both between *versicolor* and *virginica*.

**Support vectors:** 56 of 120 training flowers — setosa 6, versicolor 28, virginica 22. Setosa needs almost no support vectors because it sits far from the others; the versicolor/virginica border is where the model does its real work.

---

## 6. Confusion matrix per kernel

The test-set evaluation above uses only the **overall best model** (linear, picked by cross-validation). To see how each kernel actually behaves, every kernel is refit with **its own best hyper-parameters** from the grid search and evaluated on the same test set:

| Kernel | Best params | Test accuracy |
|--------|-------------|----------------|
| linear | `C=0.1` | 0.933 (28/30) |
| **rbf** | `C=1, gamma=0.1` | **0.967 (29/30)** |
| poly | `C=10, degree=3` | 0.933 (28/30) |

| | setosa | versicolor | virginica |  | | setosa | versicolor | virginica |  | | setosa | versicolor | virginica |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| **linear** — setosa | 10 | 0 | 0 | | **rbf** — setosa | 10 | 0 | 0 | | **poly** — setosa | 10 | 0 | 0 |
| versicolor | 0 | 9 | 1 | | versicolor | 0 | 9 | 1 | | versicolor | 0 | 10 | 0 |
| virginica | 0 | 1 | 9 | | virginica | 0 | 0 | 10 | | virginica | 0 | 2 | 8 |

See `outputs/confusion_matrix_all_kernels.png` for the visual side-by-side.

On *this particular* 30-flower test split, RBF happens to make only 1 mistake instead of 2. It doesn't overturn the earlier choice, though: cross-validation (Section 3) averages over many splits — 5 folds × several repeats worth of held-out folds — and there RBF (0.967) still trails linear (0.975). A single test split is a noisy estimate; the CV comparison is the more reliable one, which is why linear stays the recommended model. It's still worth noting *where* each kernel's errors land — poly's 2 mistakes are both virginica flowers mislabelled as versicolor, while linear splits its 2 errors one each way.

---

## 7. Interpretation

- **Setosa is perfectly classified.** It is linearly separable from the other two on petal size alone.
- **Versicolor and virginica overlap.** Their petal measurements touch around petal length ≈ 5 cm, and every error the model makes is on that border — a known property of this dataset, not a modelling fault.
- **Simpler is better.** The linear kernel with a soft margin beat the more flexible kernels; the degree-3 polynomial curves its boundary to fit noise and lost ~7 points of CV accuracy.
- **Scaling matters.** Without `StandardScaler`, sepal length (range 4.3–7.9) would dominate petal width (0.1–2.5) in the distance calculations.

---

## 8. Example predictions

| Measurements (sepal L/W, petal L/W) | Predicted species |
|-------------------------------------|-------------------|
| 5.0, 3.4, 1.5, 0.2 | Iris-setosa |
| 6.0, 2.7, 4.2, 1.3 | Iris-versicolor |
| 6.7, 3.0, 5.8, 2.1 | Iris-virginica |
