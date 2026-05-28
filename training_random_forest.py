import numpy as np
import h5py
import pickle
import time

from config import DATASET, SUFFIX
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, roc_curve, auc


# wczytanie danych z pliku HDF5
input_file = f'./data/dataset_{DATASET}{SUFFIX}.h5'
with h5py.File(input_file, 'r') as f:
    x_train = np.array(f['train_data'])
    y_train = np.array(f['train_labels'])
    x_val = np.array(f['val_data'])
    y_val = np.array(f['val_labels'])
    x_test = np.array(f['test_data_clean'])
    y_test = np.array(f['test_labels_clean'])

    img_size = f.attrs['img_size']
    pixels_per_cell = f.attrs['pixels_per_cell']
    cells_per_block = f.attrs['cells_per_block']
    categories = list(f.attrs['categories'])

print(f"Wczytano dane z {input_file}")
print(f"Train shape: {x_train.shape}")
print(f"Validation shape: {x_val.shape}")
print(f"Test shape: {x_test.shape}")
print(f"Kategorie: {categories}")


classifier = RandomForestClassifier(random_state=42)

# parametry modelu
parameters = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}


# sklejenie danych walidacyjnych i treningowych w jedną tablicę
x_train_val = np.concatenate([x_train, x_val])
y_train_val = np.concatenate([y_train, y_val])

# instrukcja podziału danych dla funkcji GridSearchCV
test_fold = np.concatenate([
    np.full(len(x_train), -1),
    np.zeros(len(x_val))
])

predefined_split = PredefinedSplit(test_fold)

# walidacja krzyżowa
grid_search = GridSearchCV(
    classifier,
    parameters,
    cv=predefined_split,
    n_jobs=-1
)

# pomiar czasu treningu
start_train = time.perf_counter()
grid_search.fit(x_train_val, y_train_val)
end_train = time.perf_counter()
training_time = end_train - start_train

# wybór najlepszego modelu
best_model = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")

# ewaluacja
best_score = grid_search.best_score_
y_pred = best_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)

# AUC
y_prob = best_model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"CV score: {best_score:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Training time: {training_time:.4f} s")

# zapis modelu z metadanymi
model_file = f'./data/models/{DATASET}{SUFFIX}/model_random_forest.p'
output = {
    "model": best_model,
    "categories": categories,
    "metrics": {
        "cv_score": best_score,
        "test_accuracy": test_accuracy,
        "auc": roc_auc,
        "training_time": training_time
    },
    "data_config": {
        "img_size": img_size,
        "pixels_per_cell": pixels_per_cell,
        "cells_per_block": cells_per_block
    }
}

pickle.dump(output, open(model_file, 'wb'))
print(f"Model zapisano jako {model_file}")