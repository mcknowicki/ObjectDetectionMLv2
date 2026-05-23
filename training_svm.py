import numpy as np
import h5py
import pickle
import time

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
    x_test = np.array(f['test_clean_data'])
    y_test = np.array(f['test_clean_labels'])

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

# trening modelu z walidacją krzyżową *5
parameters = {
    'svm__C': [1, 10],
    'svm__gamma': ['scale']
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)

# pomiar czasu treningu
start_train = time.perf_counter()
grid_search.fit(x_train, y_train)
end_train = time.perf_counter()
training_time = end_train - start_train

# wybór najlepszego modelu
best_model = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")

# ewaluacja + pomiar czasu predykcji
best_score = grid_search.best_score_
start_inference = time.perf_counter()
y_pred = best_model.predict(x_test)
end_inference = time.perf_counter()
inference_time = end_inference - start_inference
test_accuracy = accuracy_score(y_test, y_pred)

# ROC / AUC
y_prob = best_model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"CV score: {best_score:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Training time: {training_time:.4f} s")
print(f"Inference time: {inference_time:.6f} s")

# zapis modelu z metadanymi
model_file = f'./data/models/{DATASET}/model_svm.p'

output = {
    "model": best_model,
    "categories": categories,
    "metrics": {
        "cv_score": best_score,
        "test_accuracy": test_accuracy,
        "auc": roc_auc,
        "training_time": training_time,
        "inference_time": inference_time
    },
    "data_config": {
        "img_size": img_size,
        "pixels_per_cell": pixels_per_cell,
        "cells_per_block": cells_per_block
    }
}

pickle.dump(output, open(model_file, 'wb'))
print(f"Model zapisano jako {model_file}")