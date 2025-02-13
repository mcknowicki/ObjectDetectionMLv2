import pickle
import os

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt


# wczytanie modelu z pliku
# finalnie program powinien wczytywać kolejno modele w pętli i podać predykcję dla każdego z nich
model_path = './model_svm.p'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# wczytanie kategorii obiektów
input_dir = './data/input'
categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
print(f"Wykryto następujące kategorie obiektów {categories}")

# wczytanie obrazu do sklasyfikowania oraz konwersja jak w przypadku treningu klasyfikatora
img_path = './data/val/eiffel_test2.jpg'
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Nie znaleziono obrazu: {img_path}")

img = imread(img_path, as_gray=True)
img = resize(img, (256, 256))
features, hog_image = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, visualize=True)

pred_index = model.predict([features])[0]
pred_label = categories[pred_index]

print(f"Obiekt widoczny na obrazie należy do kategorii {pred_label}")

# przedstawienie obrazu oraz
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Oryginalny obraz")

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title("Cechy HOG")

plt.show()