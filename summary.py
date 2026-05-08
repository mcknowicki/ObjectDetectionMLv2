import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from config import DATASET
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# słownik modeli ML
model_paths = {
    "KNN": f'./data/models/{DATASET}/model_knn.p',
    "Random Forest": f'./data/models/{DATASET}/model_random_forest.p',
"Logistic Regression": f'./data/models/{DATASET}/model_logistic_regression.p',
    "SVM": f'./data/models/{DATASET}/model_svm.p',
    "MLP": f'./data/models/{DATASET}/model_mlp.p'
}

# wczytanie danych testowych
input_file = f'./data/dataset_{DATASET}.h5'

with h5py.File(input_file, 'r') as f:
    x_test = np.array(f['test_data'])
    y_test = np.array(f['test_labels'])

    if 'test_paths' in f:
        file_paths = f['test_paths'][:].astype(str)
    else:
        file_paths = None
print(f"Wczytano dane testowe: {x_test.shape}")

# struktury
results = []
roc_data = []
cm_data = []

# pętla analizująca po kolei modele
for model_name, model_path in model_paths.items():

    if not os.path.exists(model_path):
        print(f"Brak modelu: {model_path}")
        continue

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    model = data["model"]
    metrics = data["metrics"]
    categories = data["categories"]

    # predykcje
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int) # standardowy próg rozpoznania 0.5

    # ROC/AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # wskaźnik Youdena - dobranie progu rozpoznawania
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]

    # F1 vs threshold
    f1_scores = []

    for thr in thresholds:
        y_pred_thr = (y_prob >= thr).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_thr))

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, f1_scores)
    plt.axvline(best_threshold, linestyle='--', label='Best threshold')
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title(f"F1 vs threshold - {model_name}")
    plt.legend()
    plt.grid()
    plt.show()

    # predykcja z najlepszym progiem
    y_pred_best = (y_prob >= best_threshold).astype(int)

    roc_data.append((model_name, fpr, tpr, roc_auc, best_idx))

    # confusion matrix (dla best treshold)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_data.append((model_name, cm, categories))

    # metryki
    precision_default = precision_score(y_test, y_pred_default)
    recall_default = recall_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default)
    acc_default = accuracy_score(y_test, y_pred_default)

    precision_best = precision_score(y_test, y_pred_best)
    recall_best = recall_score(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best)
    acc_best = accuracy_score(y_test, y_pred_best)

    # overfitting
    overfit_gap = metrics["cv_score"] - metrics["test_accuracy"]

    # wyniki
    results.append({
        "Model": model_name,

        "AUC": roc_auc,
        "Best Threshold": best_threshold,

        "Accuracy (0.5)": acc_default,
        "F1 (0.5)": f1_default,

        "Accuracy (best)": acc_best,
        "F1 (best)": f1_best,

        "Precision (best)": precision_best,
        "Recall (best)": recall_best,

        "CV Score": metrics["cv_score"],
        "Overfitting Gap": overfit_gap,

        "Training Time [s]": metrics["training_time"],
        "Inference Time [s]": metrics["inference_time"],
        "Prediction/sample [s]": metrics["prediction_per_sample"]
    })

    # wizualizacja błędów
    from skimage.io import imread

    if file_paths is None:
        print("Brak test_paths – pomijam wizualizację błędów")
    else:
        # błędne klasyfikacje
        fp_idx = np.where((y_test == 0) & (y_pred_best == 1))[0]
        fn_idx = np.where((y_test == 1) & (y_pred_best == 0))[0]


        def show_errors(indices, title, max_images=6):
            plt.figure(figsize=(12, 6))

            for i, idx in enumerate(indices[:max_images]):
                img = imread(file_paths[idx], as_gray=True)

                plt.subplot(2, 3, i + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f"T:{y_test[idx]} P:{y_pred_best[idx]} ({y_prob[idx]:.2f})")
                plt.axis('off')

            plt.suptitle(title)
            plt.tight_layout()
            plt.show()


        show_errors(fp_idx, f"{model_name} - False Positives")
        show_errors(fn_idx, f"{model_name} - False Negatives")


# wykres ROC
plt.figure(figsize=(8, 6))

for model_name, fpr, tpr, roc_auc, best_idx in roc_data:
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    # punkt Youdena
    plt.scatter(
        fpr[best_idx],
        tpr[best_idx],
        s=50
    )

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Porównanie ROC modeli")
plt.legend()
plt.grid()
plt.show()

# porównanie confusion matrix
num_models = len(cm_data)
cols = min(3, num_models)
rows = int(np.ceil(num_models / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

# gdy jest tylko 1 model
if num_models == 1:
    axes = [axes]

axes = np.array(axes).reshape(-1)

for ax, (model_name, cm, categories) in zip(axes, cm_data):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories,
                yticklabels=categories,
                ax=ax)
    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# usunięcie pustych subplotów
for i in range(len(cm_data), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# tabela wyników sortowana po AUC
df = pd.DataFrame(results)
df_sorted = df.sort_values(by="AUC", ascending=False)
print("\nTABELA WYNIKÓW:")
print(df_sorted)

# zaokrąglenie wartości i zapis do csv
df_rounded = df_sorted.round(4)
output_csv = f'./data/results_{DATASET}.csv'
df_rounded.to_csv(output_csv, index=False)
print(f"\nWyniki zapisane do: {output_csv}")