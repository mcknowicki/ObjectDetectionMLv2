import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import os

from config import DATASET, INPUT_DIR, IMG_SIZE, PIXELS_PER_CELL, CELLS_PER_BLOCK
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

print(f"Wczytano dane z {input_file}")
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# mapping kategorii
categories = sorted([
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
])

print(f"Kategorie: {categories}")

# pipeline - skalowanie  + knn
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

parameters = {
    'knn__n_neighbors': [3, 5, 7]
}

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)
best_estimator = grid_search.best_estimator_

print(f"Najlepsze parametry: {grid_search.best_params_}")

# ewaluacja
y_pred = best_estimator.predict(x_test)
score = accuracy_score(y_test, y_pred)

print(f"Wynik testu skuteczności: {score * 100:.2f}%")

# ROC / AUC
if len(np.unique(y_test)) == 2:
    y_probabilities = best_estimator.predict_proba(x_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Krzywa ROC - KNN')
    plt.legend(loc='lower right')
    plt.show()
else:
    print("ROC dostępny tylko dla klasyfikacji binarnej")

# zapis modelu razem z mappingiem klas i configiem
model_file = './data/model_knn.p'

pickle.dump({
    "model": best_estimator,
    "categories": categories,
    "config": {
        "img_size": IMG_SIZE,
        "pixels_per_cell": PIXELS_PER_CELL,
        "cells_per_block": CELLS_PER_BLOCK
    }
}, open(model_file, 'wb'))

print(f"Model zapisano jako {model_file}")

"""# trening klasyfikatora k-Nearest Neighbor z optymalizacją parametrów
classifier = KNeighborsClassifier()
parameters = [{'n_neighbors': [3, 5, 7]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)
best_estimator = grid_search.best_estimator_

# ocena modelu

score = accuracy_score(best_estimator.predict(x_test), y_test)

print(f"Wynik testu skuteczności: {score * 100:.2f}%")

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