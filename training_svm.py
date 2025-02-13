import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc


# przetworzenie danych wejściowych na takie, które będą kompatybilne z algorytmami ML
input_dir  = './data/input'
categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
print(f"Wykryto następujące foldery dla poszczególnych obiektów {categories}")

data = []
labels = []

for category_index, category in enumerate(categories):
    cat_path = os.path.join(input_dir, category)
    for file in os.listdir(cat_path):
        img_path  = os.path.join(cat_path, file)
        try:
            img = imread(img_path, as_gray=True) # wczytanie obrazu i konwersja do skali szarości
            img = resize(img, (256, 256)) # można zrobić w Gimpie, HOG dobrze działa na mniejszych obrazach
            features, hog_image = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, visualize=True)

            # poniższy fragment wyświetla po kolei odczytane obrazy, a także wyekstrahowane przez HOG gradienty
            """
            plt.figure(figsize=(8, 4))

            # Oryginalny obraz
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title("Oryginalny obraz")

            # Wizualizacja cech HOG
            plt.subplot(1, 2, 2)
            plt.imshow(hog_image, cmap='gray')
            plt.title("Wizualizacja cech HOG")

            plt.show()
            """

            data.append(features)
            labels.append(category_index)
        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku:{img_path}: {e}")

# konwersja list na tablice
data = np.asarray(data)
labels = np.asarray(labels)

# podział danych na zbiór treningowy i testowy
# (dane rozmieszczone losowo, proporcje zbiorów 4 do 1, zbiory rozwarstwione proporcjonalnie wg etykiet)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)


# tutaj można rozbić program na 2 osobne klasy obiektów

# trening klasyfikatora z optymalizacją parametrów
classifier = SVC(probability=True)
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# test wydajności
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print(f"Wynik testu skuteczności: {score * 100:.2f}%")

# Zamieniamy model na taki, który zwraca prawdopodobieństwa
best_estimator.probability = True

# Obliczamy prawdopodobieństwa dla klasy pozytywnej
y_probabilities = best_estimator.predict_proba(x_test)[:, 1]

# Obliczenie krzywej ROC
fpr, tpr, _ = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# Wykres krzywej ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # linia referencyjna
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Krzywa ROC klasyfikatora SVM')
plt.legend(loc='lower right')
plt.show()

pickle.dump(best_estimator, open('./model_svm.p', 'wb'))