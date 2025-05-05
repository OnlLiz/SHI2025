import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
# Імпортуємо необхідні метрики
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Імпортуємо для графіка
import matplotlib.pyplot as plt
import seaborn as sns

# --- Параметри ---
input_file = 'income_data.txt'
max_datapoints = 25000
test_set_size = 0.2
random_seed = 5
svm_max_iter = 5000
class_labels = ['<=50K', '>50K'] # Назви класів

# --- 1. Читання та попередня обробка даних ---
X_list = []
y_list = []
count_class0 = 0
count_class1 = 0

try:
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if '?' in line or line.strip() == '': continue
        if count_class0 >= max_datapoints and count_class1 >= max_datapoints: break
        data = [item.strip() for item in line.strip().split(',')]
        if len(data) < 2: continue
        label = data[-1]
        features = data[:-1]
        if label == class_labels[0] and count_class0 < max_datapoints:
            X_list.append(features); y_list.append(0); count_class0 += 1
        elif label == class_labels[1] and count_class1 < max_datapoints:
            X_list.append(features); y_list.append(1); count_class1 += 1
except FileNotFoundError:
    print(f"Помилка: Файл '{input_file}' не знайдено.")
    exit()

if not X_list:
    print("Помилка: Не вдалося завантажити дані.")
    exit()

X = np.array(X_list)
y = np.array(y_list)

# --- 2. Кодування категоріальних ознак ---
label_encoders = []
num_cols = X.shape[1]
X_encoded = np.empty(X.shape, dtype=int)
original_cols_are_numeric = []

for i in range(num_cols):
    try:
        X_encoded[:, i] = X[:, i].astype(float).astype(int)
        original_cols_are_numeric.append(True)
    except ValueError:
        original_cols_are_numeric.append(False)
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append({'column_index': i, 'encoder': le})

X_final = X_encoded

# --- 3. Розділення даних ---
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y,
    test_size=test_set_size,
    random_state=random_seed,
    stratify=y
)

# --- 4. Створення та навчання моделі ---
classifier = OneVsOneClassifier(LinearSVC(random_state=random_seed, max_iter=svm_max_iter, dual='auto'))
classifier.fit(X_train, y_train)

# --- 5. Прогнозування та оцінка ---
y_test_pred = classifier.predict(X_test)

# Виведення основних метрик через Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_test_pred, target_names=class_labels, zero_division=0))

# Окремо виведемо точність (Accuracy)
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy:.3f}")

# --- 6. Матриця плутанини ---
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Передбачений клас')
plt.ylabel('Істинний клас')
plt.title('Матриця плутанини (Linear SVM)')
# plt.savefig("confusion_matrix_task1.png") # Розкоментуйте для збереження
plt.show()

# --- 7. Прогноз для прикладу ---
input_data_example = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

print("\n--- Прогноз для тестової точки даних ---")
print(f"Вхідні дані: {input_data_example}")

if len(input_data_example) == num_cols:
    input_data_encoded = [0] * num_cols
    for i, item in enumerate(input_data_example):
        if original_cols_are_numeric[i]:
            try: input_data_encoded[i] = int(item)
            except ValueError: input_data_encoded[i] = 0 # Або інша обробка помилки
        else:
            le_info = next((info for info in label_encoders if info['column_index'] == i), None)
            if le_info:
                try: input_data_encoded[i] = int(le_info['encoder'].transform([item])[0])
                except ValueError: input_data_encoded[i] = 0 # Обробка нового значення
            else: input_data_encoded[i] = 0 # Обробка помилки відсутності енкодера

    input_data_encoded_array = np.array(input_data_encoded).reshape(1, -1)
    predicted_class_encoded = classifier.predict(input_data_encoded_array)
    predicted_class_label = class_labels[predicted_class_encoded[0]]
    print(f"Передбачений клас: {predicted_class_label}")
else:
     print(f"Помилка: кількість ознак у прикладі ({len(input_data_example)}) не відповідає ({num_cols}).")