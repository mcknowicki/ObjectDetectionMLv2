import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import os

from config import DATASET
from sklearn.metrics import roc_curve, auc

# import modeli
model_paths = {
    "KNN": f'./data/models/{DATASET}/model_knn.p',
    "Random Forest": f'./data/models/{DATASET}/model_random_forest.p'
}

# wczytanie danych testowych
input_file = f'./data/dataset_{DATASET}.h5'

with h5py.File(input_file, 'r') as f:
    x_test = np.array(f['test_data'])
    y_test = np.array(f['test_labels'])
print(f"Wczytano dane testowe: {x_test.shape}")

# struktury
results = []
plt.figure()

# pętla po modelach
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
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # najlepszy threshold (Youden's J)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]

    # sprawdzenie overfittingu (różnica CV vs test)
    overfit_gap = metrics["cv_score"] - metrics["test_accuracy"]

    # zebranie wyników
    results.append({
        "model": model_name,
        "cv_score": metrics["cv_score"],
        "test_accuracy": metrics["test_accuracy"],
        "auc": roc_auc,
        "overfit_gap": overfit_gap,
        "best_threshold": best_threshold
    })

    # ROC na wspólnym wykresie
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# wyniki
print("\n=== WYNIKI ===")

for r in results:
    print(f"\nModel: {r['model']}")
    print(f"CV score: {r['cv_score']:.4f}")
    print(f"Test accuracy: {r['test_accuracy']:.4f}")
    print(f"AUC: {r['auc']:.4f}")
    print(f"Overfitting gap: {r['overfit_gap']:.4f}")
    print(f"Best threshold: {r['best_threshold']:.4f}")

# wykres ROC
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Porównanie ROC modeli")
plt.legend()
plt.show()