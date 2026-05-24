import os
import numpy as np
import h5py
import random

from config import (
    IMG_SIZE,
    PIXELS_PER_CELL,
    CELLS_PER_BLOCK,
    DATASET,
    INPUT_DIR,
    NUM_ROTATIONS,
    NOISE_STD,
    BLUR_SIGMA,
    ENABLE_OCCLUSION,
    NOISE_PROBABILITY,
    BLUR_PROBABILITY,
    OCCLUSION_PROBABILITY
)
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
from skimage.filters import gaussian

# seed dla powtarzalności eksperymentu na różnych zbiorach danych
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# parametry preparowanego zbioru
#MAX_SAMPLES = 5000
BG_VALUE = 1 # kolor tła dla obróconego obrazu

# wczytywanie kategorii
categories = sorted([
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
])
print(f"Wykryto następujące kategorie obiektów dla zbioru danych {DATASET}: {categories}")

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

print(f"Liczba próbek treningowych: {len(train_files)}, testowych: {len(test_files)}")

# listy wyjściowe
train_data = []
train_labels_processed = []

test_data_clean = []
test_labels_clean = []

test_data_corrupted = []
test_labels_corrupted = []

test_paths = []
test_images_clean = []
test_images_corrupted = []

# funkcja generująca zakłócenia
def corrupt_image(img):

    corrupted = img.copy()

    # gaussian noise
    if np.random.rand() < NOISE_PROBABILITY:

        corrupted = random_noise(
            corrupted,
            mode='gaussian',
            var=NOISE_STD ** 2
        )

    # blur
    if np.random.rand() < BLUR_PROBABILITY:

        corrupted = gaussian(
            corrupted,
            sigma=BLUR_SIGMA
        )

    # losowe zasłonięcie fragmentu
    if ENABLE_OCCLUSION and np.random.rand() < OCCLUSION_PROBABILITY:

        h, w = corrupted.shape

        occ_size = np.random.randint(20, 40)

        x = np.random.randint(0, w - occ_size)
        y = np.random.randint(0, h - occ_size)

        corrupted[y:y + occ_size, x:x + occ_size] = BG_VALUE

    return corrupted

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
        # losowe rotacje w zadanym zakresie
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


# zbiór testowy
print("Przetwarzanie zbioru testowego...")

for img_path, label in zip(test_files, test_labels):

    try:
        img = imread(img_path, as_gray=True)

        # zbiór bez zakłóceń
        clean_features = extract_features(img)

        test_data_clean.append(clean_features)
        test_labels_clean.append(label)

        # zbiór z zakłóceniami
        corrupted_img = corrupt_image(img)
        corrupted_features = extract_features(corrupted_img)

        test_data_corrupted.append(corrupted_features)
        test_labels_corrupted.append(label)

        # zapis obrazów do wizualizacji błędnych predykcji
        test_images_clean.append(resize(img, IMG_SIZE))
        test_images_corrupted.append(resize(corrupted_img, IMG_SIZE))

        # ścieżki do obrazów
        test_paths.append(img_path)

    except Exception as e:
        print(f"Błąd (test): {img_path} -> {e}")

# konwersja
train_data = np.array(train_data)
train_labels_processed = np.array(train_labels_processed)

test_data_clean = np.array(test_data_clean)
test_labels_clean = np.array(test_labels_clean)

test_data_corrupted = np.array(test_data_corrupted)
test_labels_corrupted = np.array(test_labels_corrupted)

test_images_clean = np.array(test_images_clean)
test_images_corrupted = np.array(test_images_corrupted)

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
    f.attrs['img_size'] = IMG_SIZE
    f.attrs['pixels_per_cell'] = PIXELS_PER_CELL
    f.attrs['cells_per_block'] = CELLS_PER_BLOCK
    f.attrs['categories'] = categories

    f.create_dataset('train_data', data=train_data)
    f.create_dataset('train_labels', data=train_labels_processed)
    f.create_dataset('test_data_clean', data=test_data_clean)
    f.create_dataset('test_labels_clean', data=test_labels_clean)
    f.create_dataset('test_data_corrupted', data=test_data_corrupted)
    f.create_dataset('test_labels_corrupted', data=test_labels_corrupted)

    f.create_dataset('test_images_clean', data=test_images_clean)
    f.create_dataset('test_images_corrupted', data=test_images_corrupted)
    f.create_dataset('test_paths', data=np.array(test_paths, dtype='S'))

print(f"Train shape: {train_data.shape}")
print(f"Test clean shape: {test_data_clean.shape}")
print(f"Test corrupted shape: {test_data_corrupted.shape}")
print(f"Dane zapisane do {output_file}")