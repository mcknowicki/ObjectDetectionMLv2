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
    x_test_clean = np.array(f['test_data_clean'])
    y_test_clean = np.array(f['test_labels_clean'])

    x_test_corrupted = np.array(f['test_data_corrupted'])
    y_test_corrupted = np.array(f['test_labels_corrupted'])

    test_images_clean = np.array(f['test_images_clean'])
    test_images_corrupted = np.array(f['test_images_corrupted'])

    if 'test_paths' in f:
        file_paths = f['test_paths'][:].astype(str)
    else:
        file_paths = None

print(f"Wczytano dane testowe (clean): {x_test_clean.shape}")
print(f"Wczytano dane testowe (corrupted): {x_test_corrupted.shape}")

# struktury
results = []
roc_data = []
cm_data = []

test_variants = {
    "Clean": (
        x_test_clean,
        y_test_clean,
        test_images_clean
    ),

    "Corrupted": (
        x_test_corrupted,
        y_test_corrupted,
        test_images_corrupted
    )
}

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

    for test_name, (x_test, y_test, test_images) in test_variants.items():
        print(f"\n===== {model_name} | TEST: {test_name} =====")

        # predykcja
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # ROC/AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # obliczenie najlepszego progu decyzyjnego
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

        roc_data.append((
            model_name,
            test_name,
            fpr,
            tpr,
            roc_auc,
            best_idx
        ))

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_data.append((
            model_name,
            test_name,
            cm,
            categories
        ))

        # metryki sklearn
        precision_default = precision_score(y_test, y_pred)
        recall_default = recall_score(y_test, y_pred)
        f1_default = f1_score(y_test, y_pred)
        acc_default = accuracy_score(y_test, y_pred)

        # overfitting gap
        overfit_gap = metrics["cv_score"] - metrics["test_accuracy"]

        # wyniki
        results.append({
            "Model": model_name,
            "Test Variant": test_name,

            "AUC": roc_auc,

            "Accuracy": acc_default,
            "Precision": precision_default,
            "Recall": recall_default,
            "F1": f1_default,

            "CV Score": metrics["cv_score"],
            "Overfitting Gap": overfit_gap,

            "Training Time [s]": metrics["training_time"],
            "Inference Time [s]": metrics["inference_time"]
        })

        # wizualizacja błędów
        from skimage.io import imread

        if file_paths is None:
            print("Brak test_paths – pomijam wizualizację błędów")
        else:
            # błędne klasyfikacje
            fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
            fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]


            def show_errors(indices, title, max_images=6):
                plt.figure(figsize=(12, 6))

                for i, idx in enumerate(indices[:max_images]):
                    img = test_images[idx]

                    plt.subplot(2, 3, i + 1)
                    plt.imshow(img, cmap='gray')
                    plt.title(f"T:{y_test[idx]} P:{y_pred[idx]} ({y_prob[idx]:.2f})")
                    plt.axis('off')

                plt.suptitle(title)
                plt.tight_layout()
                plt.show()


            show_errors(
                fp_idx,
                f"{model_name} [{test_name}] - False Positives"
            )

            show_errors(
                fn_idx,
                f"{model_name} [{test_name}] - False Negatives"
            )


# wspólny wykres ROC
plt.figure(figsize=(8, 6))

for model_name, test_name, fpr, tpr, roc_auc, best_idx in roc_data:
    plt.plot(fpr, tpr, label=f'{model_name} [{test_name}] (AUC={roc_auc:.2f})')
    # najlepszy próg decyzyjny
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

for ax, (model_name, test_name, cm, categories) in zip(axes, cm_data):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories,
                yticklabels=categories,
                ax=ax)
    ax.set_title(f"{model_name} [{test_name}]")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# usunięcie pustych subplotów
for i in range(len(cm_data), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# tabela wyników
df = pd.DataFrame(results)

robustness_results = []

for model_name in df["Model"].unique():

    clean_row = df[
        (df["Model"] == model_name) &
        (df["Test Variant"] == "Clean")
    ].iloc[0]

    corrupted_row = df[
        (df["Model"] == model_name) &
        (df["Test Variant"] == "Corrupted")
    ].iloc[0]

    accuracy_clean = clean_row["Accuracy"]
    accuracy_corrupted = corrupted_row["Accuracy"]

    auc_clean = clean_row["AUC"]
    auc_corrupted = corrupted_row["AUC"]

    f1_clean = clean_row["F1"]
    f1_corrupted = corrupted_row["F1"]

    robustness_results.append({

        "Model": model_name,

        "Accuracy Clean": accuracy_clean,
        "Accuracy Corrupted": accuracy_corrupted,
        "Accuracy Drop": accuracy_clean - accuracy_corrupted,

        "AUC Clean": auc_clean,
        "AUC Corrupted": auc_corrupted,
        "AUC Drop": auc_clean - auc_corrupted,

        "F1 Clean": f1_clean,
        "F1 Corrupted": f1_corrupted,
        "F1 Drop": f1_clean - f1_corrupted,

        "Robustness Score": (
            accuracy_corrupted / accuracy_clean
        )
    })

df_sorted = df.sort_values(by="AUC", ascending=False)
print("\nTABELA WYNIKÓW:")
print(df_sorted)

df_robustness = pd.DataFrame(robustness_results)

print("\nODPORNOŚĆ MODELI NA ZAKŁÓCENIA:")
print(df_robustness.round(4))

plt.figure(figsize=(8, 5))

plt.bar(
    df_robustness["Model"],
    df_robustness["Robustness Score"]
)

plt.ylabel("Robustness Score")
plt.ylim(0, 1.05)
plt.title("Odporność modeli na zakłócenia")
plt.grid(axis='y')

plt.show()

# zaokrąglenie wartości i zapis do csv
df_rounded = df_sorted.round(4)
output_csv = f'./data/results_{DATASET}.csv'
df_rounded.to_csv(output_csv, index=False)
print(f"\nWyniki zapisane do: {output_csv}")

df_robustness_rounded = df_robustness.round(4)
output_csv_robustness = f'./data/results_{DATASET}_robustness.csv'
df_robustness_rounded.to_csv(output_csv_robustness, index=False)
print(f"\nTabela odporności zapisana do: {output_csv_robustness}")