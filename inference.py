import pickle
import os

from config import IMG_SIZE, PIXELS_PER_CELL, CELLS_PER_BLOCK
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt

#config
model_path = './data/models/roadsigns1/model_knn.p'
img_path = './data/additional_images/Disruption_4.png'

# wczytanie modelu i kategorii z pliku
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

with open(model_path, 'rb') as f:
    data = pickle.load(f)

model = data["model"]
categories = data["categories"]

print(f"Wczytane kategorie: {categories}")


# wczytanie obrazu do sklasyfikowania oraz konwersja jak w przypadku treningu klasyfikatora
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Nie znaleziono obrazu: {img_path}")

img = imread(img_path, as_gray=True)
img = resize(img, IMG_SIZE)
features, hog_image = hog(img, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, feature_vector=True, visualize=True)


probabilities = model.predict_proba([features])[0]

best_index = probabilities.argmax()
best_prob = probabilities[best_index]

THRESHOLD = 0.75  # próg rozpoznania do
pred_label = categories[best_index]

if best_prob < THRESHOLD:
    print(f"Obiekt nierozpoznany (probability {best_prob:.2f} {pred_label})")
else:
    print(f"Predykcja: {pred_label} (probability {best_prob:.2f})")

# przedstawienie obrazu oraz HOG
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Oryginalny obraz")

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title("Cechy HOG")

plt.show()