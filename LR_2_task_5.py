# --- Необхідні імпорти ---
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# Імпорти для візуалізації
import matplotlib.pyplot as plt
import seaborn as sns
# Імпорт BytesIO (хоча збереження в SVG може бути необов'язковим)
from io import BytesIO

# --- 1. Завантаження даних ---
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names # Отримуємо назви класів для графіків

# --- 2. Розділення даних ---
# Розділяємо на навчальний (70%) та тестовий (30%) набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# --- 3. Створення та навчання моделі RidgeClassifier ---
clf = RidgeClassifier(tol=1e-2, solver="sag")
# Навчання моделі
clf.fit(X_train, y_train)

# --- 4. Прогнозування на тестовому наборі ---
y_pred = clf.predict(X_test)

# --- 5. Розрахунок та виведення метрик якості ---
print("--- Метрики якості класифікатора Ridge ---")
accuracy = metrics.accuracy_score(y_test, y_pred)
# Використовуємо f-string для форматування
print(f"Accuracy: {accuracy:.4f}")

# Для багатокласової класифікації вказуємо average='weighted' для precision, recall, f1
precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1 Score (Weighted): {f1:.4f}")

# Коефіцієнт Коена Каппа
kappa = metrics.cohen_kappa_score(y_test, y_pred)
print(f"Cohen Kappa Score: {kappa:.4f}")

# Коефіцієнт кореляції Метьюза (MCC)
mcc = metrics.matthews_corrcoef(y_test, y_pred)
print(f"Matthews Corrcoef: {mcc:.4f}")

# Classification Report (виправлено порядок аргументів)
print('\n--- Classification Report ---:\n', metrics.classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# --- 6. Побудова та збереження матриці плутанини ---
print("\n--- Генерація матриці плутанини ---")
mat = confusion_matrix(y_test, y_pred)

# Використовуємо seaborn для кращого вигляду
plt.figure(figsize=(7, 6))
sns.set() # Застосовуємо стиль seaborn
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True, cmap='Blues', # square=True, annot=True, fmt='d', cbar=True
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Передбачений клас')
plt.ylabel('Істинний клас')
plt.title('Матриця плутанини (Ridge Classifier)')

# Збереження зображення
try:
    plt.savefig("Confusion.jpg")
    print("Матрицю плутанини збережено у файл 'Confusion.jpg'")
except Exception as e:
    print(f"Не вдалося зберегти матрицю плутанини: {e}")

# Показати графік
plt.show()

print("\n--- Завдання 2.5 Завершено ---")