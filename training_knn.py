import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt

from config import DATASET
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# wczytanie danych z pliku HDF5
input_file = f'./data/dataset_{DATASET}.h5'
with h5py.File(input_file, 'r') as f:
    x_train = np.array(f['train_data'])
    y_train = np.array(f['train_labels'])
    x_test = np.array(f['test_data'])
    y_test = np.array(f['test_labels'])

    img_size = f.attrs['img_size']
    pixels_per_cell = f.attrs['pixels_per_cell']
    cells_per_block = f.attrs['cells_per_block']
    categories = list(f.attrs['categories'])

print(f"Wczytano dane z {input_file}")
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
print(f"Kategorie: {categories}")

# pipeline - skalowanie  + knn
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

parameters = {
    'knn__n_neighbors': [3, 5, 7]
}

# model wraz z walidacją krzyżowa *5
grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")

# ewaluacja
best_score = grid_search.best_score_
y_pred = best_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)

# ROC / AUC
y_prob = best_model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"CV score: {best_score:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")

""" TO WRZUCIMY DO ANALIZ
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC - KNN")
plt.legend()
plt.show()
"""

# zapis modelu razem z metadanymi
model_file = f'./data/models/{DATASET}/model_random_forest.p'
output = {
    "model": best_model,
    "categories": categories,
    "metrics": {
        "cv_score": best_score,
        "test_accuracy": test_accuracy,
        "auc": roc_auc
    },
    "data_config": {
        "img_size": img_size,
        "pixels_per_cell": pixels_per_cell,
        "cells_per_block": cells_per_block
    }
}

pickle.dump(output, open(model_file, 'wb'))
print(f"Model zapisano jako {model_file}")

"""
# obliczanie prawdopodobieństw dla krzywej ROC
y_probabilities = best_estimator.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# rysowanie krzywej ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Krzywa ROC klasyfikatora KNN')
plt.legend(loc='lower right')
plt.show()

# zapis wytrenowanego modelu
model_file = './data/model_knn.p'
pickle.dump(best_estimator, open(model_file, 'wb'))
print(f"Model zapisano jako {model_file}")"""