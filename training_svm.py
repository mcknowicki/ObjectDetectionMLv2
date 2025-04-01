import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,  roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# wczytanie danych z pliku HDF5
input_file = './data/dataset.h5'
with h5py.File(input_file, 'r') as f:
    x_train = np.array(f['x_train'])
    y_train = np.array(f['y_train'])
    x_test = np.array(f['x_test'])
    y_test = np.array(f['y_test'])

print(f"Wczytano dane z pliku {input_file}")

# trening klasyfikatora SVM z optymalizacją parametrów
classifier = SVC(probability=True)
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}] # trzeba lepiej dobrać nastawy, bo program zajmuje ponad godzinę
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)
best_estimator = grid_search.best_estimator_

# ocena skuteczności
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print(f"Ocena skuteczności modelu: {score * 100:.2f}%")

# obliczamy prawdopodobieństw dla krzywej ROC
y_probabilities = best_estimator.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# wykres krzywej ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # linia referencyjna
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Krzywa ROC klasyfikatora SVM')
plt.legend(loc='lower right')
plt.show()

# zapis modelu do pliku
model_file = './data/model_svm.p'
pickle.dump(best_estimator, open(model_file, 'wb'))
print(f"Model zapisany do {model_file}")