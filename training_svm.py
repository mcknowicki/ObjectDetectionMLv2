import numpy as np
import h5py
import pickle

from config import DATASET
from sklearn.svm import SVC
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
    ('svm', SVC(probability=True))
])

parameters = {
    'svm__C': [1, 10],
    'svm__gamma': ['scale']
}

# model wraz z walidacją krzyżowa *5
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")

# ewaluacja
best_score = grid_search.best_score_

y_pred = best_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)

# ROC / AUC
y_prob = best_model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"CV score: {best_score:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")

# zapis modelu razem z metadanymi
model_file = f'./data/models/{DATASET}/model_svm.p'

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