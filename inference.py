import pickle
import os

from config import IMG_SIZE, PIXELS_PER_CELL, CELLS_PER_BLOCK, INPUT_DIR
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt

#config
model_path = './data/model_random_forest.p'
img_path = './data/val/stop_test2.jpg'

# wczytanie modelu z pliku
# finalnie program powinien wczytywać kolejno modele w pętli i podać predykcję dla każdego z nich
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# wczytanie kategorii obiektów
categories = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
print(f"Wykryto następujące kategorie obiektów {categories}")

# wczytanie obrazu do sklasyfikowania oraz konwersja jak w przypadku treningu klasyfikatora
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Nie znaleziono obrazu: {img_path}")

img = imread(img_path, as_gray=True)
img = resize(img, IMG_SIZE)
features, hog_image = hog(img, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, feature_vector=True, visualize=True)


probabilities = model.predict_proba([features])[0]

best_index = probabilities.argmax()
best_prob = probabilities[best_index]

THRESHOLD = 0.5  # do przetestowania i dobrania

if best_prob < THRESHOLD:
    print(f"Obiekt nierozpoznany (pewność {best_prob:.2f})")
else:
    pred_label = categories[best_index]
    print(f"Predykcja: {pred_label} (pewność {best_prob:.2f})")

# przedstawienie obrazu oraz
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Oryginalny obraz")

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title("Cechy HOG")

plt.show()