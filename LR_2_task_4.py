# --- Необхідні імпорти ---
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
# Імпорт моделей
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Додамо імпорт для ігнорування попереджень
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Параметри ---
input_file = 'income_data.txt'
max_datapoints = 25000
test_set_size = 0.2
random_seed = 1

class_labels = ['<=50K', '>50K']

# --- 1. Завантаження та попередня обробка даних ---
X_list = []
y_list = []
count_class0 = 0; count_class1 = 0
try:
    with open(input_file, 'r') as f: lines = f.readlines()
    for line in lines:
        if '?' in line or line.strip() == '': continue
        if count_class0 >= max_datapoints and count_class1 >= max_datapoints: break
        data = [item.strip() for item in line.strip().split(',')]
        if len(data) < 2: continue
        label = data[-1]; features = data[:-1]
        if label == class_labels[0] and count_class0 < max_datapoints:
            X_list.append(features); y_list.append(0); count_class0 += 1
        elif label == class_labels[1] and count_class1 < max_datapoints:
            X_list.append(features); y_list.append(1); count_class1 += 1
except FileNotFoundError: print(f"Помилка: Файл '{input_file}' не знайдено."); exit()
if not X_list: print("Помилка: Не вдалося завантажити дані."); exit()
X = np.array(X_list); y = np.array(y_list)

# --- 2. Кодування категоріальних ознак ---
label_encoders = []
num_cols = X.shape[1]
X_encoded = np.empty(X.shape, dtype=int)
for i in range(num_cols):
    try: X_encoded[:, i] = X[:, i].astype(float).astype(int)
    except ValueError:
        le = preprocessing.LabelEncoder(); X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append({'column_index': i, 'encoder': le})
X_final = X_encoded

# --- 3. Розділення даних ---
# Потрібне лише для отримання X_train, на якому будемо масштабувати та проводити CV
X_train, X_test_dummy, y_train, y_test_dummy = train_test_split(
    X_final, y, test_size=test_set_size, random_state=random_seed, stratify=y
)

# --- 4. Масштабування навчальних даних ---
# Масштабуємо ТІЛЬКИ навчальні дані, бо CV працює лише з ними
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 5. Порівняння алгоритмів за допомогою крос-валідації ---
print("\n--- Порівняння алгоритмів на даних 'income_data.txt' (10-fold cross-validation) ---")
models = []
models.append(('LR', LogisticRegression(solver='liblinear'))) # Як у завд. 2.3
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
print("Точність (Accuracy) для кожного алгоритму:")
for name, model in models:
	# Використовуємо StratifiedKFold для збереження пропорцій класів у кожному фолді
	kfold = StratifiedKFold(n_splits=10, random_state=random_seed, shuffle=True)
	# Оцінюємо модель на масштабованих навчальних даних
	cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# --- 6. Візуалізація результатів порівняння ---
plt.figure(figsize=(10, 6))
plt.boxplot(results, tick_labels=names)
plt.title('Порівняння точності алгоритмів (Income Data)')
plt.ylabel('Accuracy (Cross-Validation)')
plt.show()

print("\n--- Завдання 2.4 Завершено ---")