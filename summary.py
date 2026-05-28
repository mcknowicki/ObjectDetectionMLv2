import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time

from config import DATASET, SUFFIX, SHOW_FALSE_PREDICTIONS
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
    "Logistic Regression": f'./data/models/{DATASET}{SUFFIX}/model_logistic_regression.p',
    "KNN": f'./data/models/{DATASET}{SUFFIX}/model_knn.p',
    "Random Forest": f'./data/models/{DATASET}{SUFFIX}/model_random_forest.p',
    "SVM": f'./data/models/{DATASET}{SUFFIX}/model_svm.p'
}

# wczytanie danych testowych
input_file = f'./data/dataset_{DATASET}{SUFFIX}.h5'

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


# organizacja struktur danych
results = []
roc_data = []
cm_data = []
f1_data = []

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


# pętla ewaluacji modeli
for model_name, model_path in model_paths.items():

    if not os.path.exists(model_path):
        print(f"Brak modelu: {model_path}")
        continue

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    model = data["model"]
    metrics = data["metrics"]
    categories = data["categories"]
    data_config = data["data_config"]

    img_size = data_config["img_size"]
    pixels_per_cell = data_config["pixels_per_cell"]
    cells_per_block = data_config["cells_per_block"]

    for test_name, (x_test, y_test, test_images) in test_variants.items():
        print(f"\n{model_name}: {test_name}")

        # predykcja wraz z pomiarem czasu
        start_inference = time.perf_counter()
        y_prob = model.predict_proba(x_test)[:, 1]
        end_inference = time.perf_counter()
        inference_time = end_inference - start_inference
        y_pred = (y_prob >= 0.5).astype(int)

        # ROC i AUC
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

        f1_data.append((
            model_name,
            test_name,
            thresholds,
            f1_scores,
            best_threshold
        ))

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

        # metryki z sklearn
        precision_default = precision_score(y_test, y_pred)
        recall_default = recall_score(y_test, y_pred)
        f1_default = f1_score(y_test, y_pred)
        acc_default = accuracy_score(y_test, y_pred)

        # overfitting gap
        overfit_gap = metrics["cv_score"] - metrics["test_accuracy"]

        # zapis wyników
        results.append({
            "Model": model_name,
            "Test Variant": test_name,

            "IMG Size": img_size,
            "Pixels per cell": pixels_per_cell,
            "Cells per block": cells_per_block,

            "AUC": roc_auc,
            "Accuracy": acc_default,
            "Precision": precision_default,
            "Recall": recall_default,
            "F1": f1_default,

            "CV Score": metrics["cv_score"],
            "Overfitting Gap": overfit_gap,

            "Training Time [s]": metrics["training_time"],
            "Inference Time [s]": inference_time
        })

        # wyświetlanie błędnych predykcji
        if SHOW_FALSE_PREDICTIONS:
            if file_paths is None:
                print("Brak test_paths – wizualizacja błędów niemożliwa")
            else:
                fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
                fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

                # funkcja wyświetlająca obrazy w oknie
                def show_false(indices, title, images_per_window=6):
                    if len(indices) == 0:
                        print(f"Brak błędnych klasyfikacji dla modelu {title}")
                        return

                    for start in range(0, len(indices), images_per_window):
                        batch = indices[start:start + images_per_window]

                        plt.figure(figsize=(12, 6))

                        for i, idx in enumerate(batch):
                            img = test_images[idx]

                            plt.subplot(2, 3, i + 1)
                            plt.imshow(img, cmap='gray')
                            plt.title(f"T:{y_test[idx]} P:{y_pred[idx]} ({y_prob[idx]:.2f})")
                            plt.axis('off')

                        window_no = start // images_per_window + 1
                        total_windows = int(np.ceil(len(indices) / images_per_window))

                        plt.suptitle(f"{title} ({window_no}/{total_windows})")
                        plt.tight_layout()
                        plt.show()

                # użycie funkcji dla FP i FN
                show_false(
                    fp_idx,
                    f"{model_name} [{test_name}] - False Positives"
                )

                show_false(
                    fn_idx,
                    f"{model_name} [{test_name}] - False Negatives"
                )


# pętla rysująca wykresy i macierz pomyłek
for model_name in model_paths.keys():

    # ROC
    model_roc_data = [
        item for item in roc_data
        if item[0] == model_name
    ]

    if len(model_roc_data) == 0:
        continue

    plt.figure(figsize=(7, 6))

    for _, test_name, fpr, tpr, roc_auc, best_idx in model_roc_data:

        plt.plot(
            fpr,
            tpr,
            label=f'{test_name} (AUC={roc_auc:.2f})'
        )

        plt.scatter(
            fpr[best_idx],
            tpr[best_idx],
            s=50
        )

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {model_name}")
    plt.legend()
    plt.grid()

    plt.show()

    # F1 vs threshold
    model_f1_data = [
        item for item in f1_data
        if item[0] == model_name
    ]

    if len(model_f1_data) == 0:
        continue

    plt.figure(figsize=(7, 5))

    for _, test_name, thresholds, f1_scores, best_threshold in model_f1_data:
        line, = plt.plot(
            thresholds,
            f1_scores,
            label=test_name
        )

        color = line.get_color()

        plt.axvline(
            best_threshold,
            linestyle='--',
            color=color,
            alpha=0.8
        )

    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title(f"F1 vs threshold - {model_name}")

    plt.legend()
    plt.grid()

    plt.show()

    # confusion matrix
    model_cm_data = [
        item for item in cm_data
        if item[0] == model_name
    ]

    if len(model_cm_data) == 0:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (_, test_name, cm, categories) in zip(axes, model_cm_data):

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=categories,
            yticklabels=categories,
            ax=ax
        )

        ax.set_title(test_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle(f"Confusion Matrix - {model_name}")

    plt.tight_layout()
    plt.show()


# przygotowanie tabeli wyników
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

    # przygotowanie tabeli odporności na zakłócenia
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

# wydruk tabel oraz ich eksport
print("\nTABELA WYNIKÓW:")
print(df)

df_robustness = pd.DataFrame(robustness_results)

print("\nODPORNOŚĆ MODELI NA ZAKŁÓCENIA:")
print(df_robustness.round(4))

# wykres odporności poszczególnych modeli
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
df_rounded = df.round(4)
output_csv = f'./data/results_{DATASET}{SUFFIX}.csv'
df_rounded.to_csv(output_csv, index=False)
print(f"\nWyniki zapisane do: {output_csv}")

df_robustness_rounded = df_robustness.round(4)
output_csv_robustness = f'./data/results_{DATASET}{SUFFIX}_robustness.csv'
df_robustness_rounded.to_csv(output_csv_robustness, index=False)
print(f"\nTabela odporności zapisana do: {output_csv_robustness}")