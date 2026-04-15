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
MAX_SAMPLES = 5000
BG_VALUE = 1 # kolor tła
NUM_ROTATIONS = 5 # liczba losowych rotacji

# wczytywanie kategorii
categories = [
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
]
print(f"Wykryto następujące kategorie obiektów: {categories}")

train_data, train_labels = [], []
test_data, test_labels = [], []

# przetwarzanie danych
for category_index, category in enumerate(categories):
    cat_path = os.path.join(INPUT_DIR, category)
    files = os.listdir(cat_path)

    # podział danych na zbiór treningowy i testowy
    train_files, test_files = train_test_split(
        files,
        test_size=0.2,
        random_state=SEED
    ) # nie wiem czy nie dodać parametru stratify = labels

    print(f"{category}: train={len(train_files)}, test={len(test_files)}")

    # zbiór treningowy z augmentacją
    for file in train_files:
        img_path = os.path.join(cat_path, file)
        try:
            img = imread(img_path, as_gray=True)

            # obrót o losowe kąty
            angles = np.random.randint(0, 360, NUM_ROTATIONS)

            for angle in angles:
                rotated_img = rotate(
                    img,
                    angle,
                    resize=False,
                    mode='constant',
                    cval=BG_VALUE
                )
                resized_img = resize(rotated_img, IMG_SIZE)
                features = hog(
                    resized_img,
                    pixels_per_cell=PIXELS_PER_CELL,
                    cells_per_block=CELLS_PER_BLOCK,
                    feature_vector=True
                )
                train_data.append(features)
                train_labels.append(category_index)

        except Exception as e:
            print(f"Błąd przetwarzania w zbiorze treningowym {img_path}: {e}")

    # zbiór testowy bez augmentacji
    for file in test_files:
        img_path = os.path.join(cat_path, file)
        try:
            img = imread(img_path, as_gray=True)
            resized_img = resize(img, IMG_SIZE)
            features = hog(
                resized_img,
                pixels_per_cell=PIXELS_PER_CELL,
                cells_per_block=CELLS_PER_BLOCK,
                feature_vector=True
            )
            test_data.append(features)
            test_labels.append(category_index)

        except Exception as e:
            print(f"Błąd przetwarzania w zbiorze testowym: {img_path}: {e}")

# konwersja
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

#mieszanie w zbiorze train
train_indices = np.arange(len(train_data))
np.random.shuffle(train_indices)
train_data = train_data[train_indices]
train_labels = train_labels[train_indices]

# ograniczenie ilości próbek według parametru MAX_SAMPLES
"""train_data = train_data[:MAX_SAMPLES]
train_labels = train_labels[:MAX_SAMPLES]"""

# zapis danych do pliku HDF5
output_file = f'./data/dataset{DATASET}.h5'
with h5py.File(output_file, 'w') as f:
    f.create_dataset('train_data', data=train_data)
    f.create_dataset('train_labels', data=train_labels)
    f.create_dataset('test_data', data=test_data)
    f.create_dataset('test_labels', data=test_labels)

print(f"Dane zapisane do {output_file}")
