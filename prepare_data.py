import os
import numpy as np
import h5py
import random

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

# katalog z danymi wejściowymi
input_dir = './data/input'
categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
print(f"Wykryto następujące foldery dla poszczególnych obiektów: {categories}")

data = []
labels = []

# przetwarzanie obrazów
for category_index, category in enumerate(categories):
    cat_path = os.path.join(input_dir, category)

    for file in os.listdir(cat_path):
        img_path = os.path.join(cat_path, file)
        try:
            img = imread(img_path, as_gray=True)

            for angle in range(1, 361):
                rotated_img = rotate(img, angle, resize=False, mode='constant', cval=1)
                img = resize(img, (256, 256))
                features = hog(rotated_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
                data.append(features)
                labels.append(category_index)

        except Exception as e:
            print(f"Błąd przetwarzania {img_path}: {e}")

# mieszanie próbek
data = np.array(data)
labels = np.array(labels)

indices = np.arange(len(data))
np.random.shuffle(indices)
data, labels = data[indices], labels[indices]

# ograniczenie ilości próbek według parametru MAX_SAMPLES
data = data[:MAX_SAMPLES]
labels = labels[:MAX_SAMPLES]

# podział na zbiór treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=False) # proporcje 4 do 1, brak shuffle, aby zachować rozmieszczenie etykiet z ziarna seed
# nie wiem czy nie dodać parametru stratify = labels

# zapis danych do pliku HDF5
output_file = './data/dataset.h5'
with h5py.File(output_file, 'w') as f:
    f.create_dataset('x_train', data=x_train)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('x_test', data=x_test)
    f.create_dataset('y_test', data=y_test)

print(f"Dane zapisane do {output_file}")
