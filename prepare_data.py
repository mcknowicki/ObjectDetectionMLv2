import os
import numpy as np
import h5py
import random

from config import IMG_SIZE, PIXELS_PER_CELL, CELLS_PER_BLOCK, DATASET, INPUT_DIR
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.feature import hog
from sklearn.model_selection import train_test_split

# seed dla powtarzalności eksperymentu na różnych zbiorach danych
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# parametry preparowanego zbioru
#MAX_SAMPLES = 5000
BG_VALUE = 1 # kolor tła
NUM_ROTATIONS = 5 # liczba losowych rotacji

# wczytywanie kategorii
categories = sorted([
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
])
print(f"Wykryto następujące kategorie obiektów: {categories}")

all_files = []
all_labels = []

# przetwarzanie danych
for category_index, category in enumerate(categories):
    cat_path = os.path.join(INPUT_DIR, category)
    files = os.listdir(cat_path)

    for file in files:
        full_path = os.path.join(cat_path, file)
        all_files.append(full_path)
        all_labels.append(category_index)

all_files = np.array(all_files)
all_labels = np.array(all_labels)

print(f"Łączna liczba próbek: {len(all_files)}")

# globalny podział danych (stratify) na zbiór treningowy i testowy
train_files, test_files, train_labels, test_labels = train_test_split(
    all_files,
    all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=SEED
)

print(f"Train: {len(train_files)}, Test: {len(test_files)}")

# listy wyjściowe
train_data = []
train_labels_processed = []

test_data = []
test_labels_processed = []

# funkcja ekstrakcji cech
def extract_features(img):
    img_resized = resize(img, IMG_SIZE)
    features = hog(
        img_resized,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        feature_vector=True
    )
    return features

# zbiór treningowy z augmentacją
print("Przetwarzanie zbioru treningowego...")

for img_path, label in zip(train_files, train_labels):
    try:
        img = imread(img_path, as_gray=True)
        # losowe rotacje w realistycznym zakresie
        angles = np.random.uniform(-20, 20, NUM_ROTATIONS)

        for angle in angles:
            rotated_img = rotate(
                img,
                angle,
                resize=False,
                mode='constant',
                cval=BG_VALUE
            )

            features = extract_features(rotated_img)

            train_data.append(features)
            train_labels_processed.append(label)

    except Exception as e:
        print(f"Błąd (train): {img_path} -> {e}")

# zbiór testowy bez augmentacji
print("Przetwarzanie zbioru testowego...")

for img_path, label in zip(test_files, test_labels):
    try:
        img = imread(img_path, as_gray=True)
        features = extract_features(img)

        test_data.append(features)
        test_labels_processed.append(label)

    except Exception as e:
        print(f"Błąd (test): {img_path} -> {e}")

# konwersja
train_data = np.array(train_data)
train_labels_processed = np.array(train_labels_processed)

test_data = np.array(test_data)
test_labels_processed = np.array(test_labels_processed)

#mieszanie w zbiorze train
indices = np.arange(len(train_data))
np.random.shuffle(indices)

train_data = train_data[indices]
train_labels_processed = train_labels_processed[indices]

# ograniczenie ilości próbek według parametru MAX_SAMPLES
"""train_data = train_data[:MAX_SAMPLES]
train_labels = train_labels[:MAX_SAMPLES]"""

# zapis danych do pliku HDF5
output_file = f'./data/dataset_{DATASET}.h5'
with h5py.File(output_file, 'w') as f:
    f.create_dataset('train_data', data=train_data)
    f.create_dataset('train_labels', data=train_labels_processed)
    f.create_dataset('test_data', data=test_data)
    f.create_dataset('test_labels', data=test_labels_processed)

print(f"Dane zapisane do {output_file}")
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")