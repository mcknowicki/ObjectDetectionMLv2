import os
import numpy as np
import pickle
import cv2

from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC


# przetworzenie danych wejściowych na takie, które będą kompatybilne z algorytmami ML
input_dir  = './data/input'
categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
print(f"Wykryto następujące foldery dla poszczególnych obiektów {categories}")

data = []
labels = []

for category_index, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path  = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (200, 200)) # można zrobić w Gimpie
        data.append(img.flatten()) # można zrobić w Gimpie
        labels.append(category_index)

data = np.asarray(data)
labels = np.asarray(labels)

# podział danych na zbiór treningowy i testowy
# (dane rozmieszczone losowo, proporcje zbiorów 4 do 1, zbiory rozwarstwione proporcjonalnie wg etykiet)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)


# tutaj można rozbić program na obiekty

# trening klasyfikatora z optymalizacją parametrów
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# test wydajności
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print(f"Wynik testu skuteczności: {score * 100:.2f}%")

pickle.dump(best_estimator, open('./model_svm.p', 'wb'))