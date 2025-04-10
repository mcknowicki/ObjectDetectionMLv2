import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc


# wczytanie danych z pliku HDF5
input_file = './data/dataset.h5'
with h5py.File(input_file, 'r') as f:
    x_train = np.array(f['x_train'])
    y_train = np.array(f['y_train'])
    x_test = np.array(f['x_test'])
    y_test = np.array(f['y_test'])

print(f"Wczytano dane z {input_file}")

# trening klasyfikatora Random Forest z optymalizacją parametrów
classifier = RandomForestClassifier(random_state=42)
parameters = [{'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)
best_estimator = grid_search.best_estimator_

# ocena modelu
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print(f"Wynik testu skuteczności: {score * 100:.2f}%")

# obliczanie prawdopodobieństw dla krzywej ROC
y_probabilities = best_estimator.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# rysowanie krzywej ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Krzywa ROC klasyfikatora Random Forest')
plt.legend(loc='lower right')
plt.show()

# zapis wytrenowanego modelu
model_file = './data/model_random_forest.p'
pickle.dump(best_estimator, open(model_file, 'wb'))
print(f"Model zapisano jako {model_file}")