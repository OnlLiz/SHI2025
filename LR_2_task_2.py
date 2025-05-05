import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# Додаємо StandardScaler
from sklearn.preprocessing import StandardScaler
# Імпортуємо необхідні метрики
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Імпортуємо для графіка
import matplotlib.pyplot as plt
import seaborn as sns
import time # Для вимірювання часу навчання

# --- Параметри ---
input_file = 'income_data.txt'
max_datapoints = 25000
test_set_size = 0.2
random_seed = 5
# Параметри для SVC ядер
poly_degree = 8 # Степінь для поліноміального ядра

class_labels = ['<=50K', '>50K'] # Назви класів

# --- 1. Читання та попередня обробка даних ---
X_list = []
y_list = []
count_class0 = 0
count_class1 = 0
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
except FileNotFoundError:
    print(f"Помилка: Файл '{input_file}' не знайдено."); exit()
if not X_list: print("Помилка: Не вдалося завантажити дані."); exit()

X = np.array(X_list); y = np.array(y_list)

# --- 2. Кодування категоріальних ознак ---
label_encoders = []
num_cols = X.shape[1]
X_encoded = np.empty(X.shape, dtype=int)
for i in range(num_cols):
    try:
        X_encoded[:, i] = X[:, i].astype(float).astype(int)
    except ValueError:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append({'column_index': i, 'encoder': le})
X_final = X_encoded

# --- 3. Розділення даних ---
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=test_set_size, random_state=random_seed, stratify=y
)

# --- 4. Масштабування даних ---
scaler = StandardScaler()
# Навчаємо scaler ТІЛЬКИ на тренувальних даних
X_train_scaled = scaler.fit_transform(X_train)
# Застосовуємо scaler до тестових даних
X_test_scaled = scaler.transform(X_test)


# --- Функція для навчання та оцінки моделі ---
def train_evaluate_svm(kernel_type, kernel_params, X_train_data, y_train_data, X_test_data, y_test_data):
    """Навчає та оцінює SVC модель з заданим ядром на наданих даних."""
    print(f"\n--- Тестування SVM з ядром: {kernel_type.upper()} ---")

    classifier = SVC(kernel=kernel_type, random_state=random_seed, **kernel_params)

    start_time = time.time()
    classifier.fit(X_train_data, y_train_data)
    end_time = time.time()
    print(f"Час навчання: {end_time - start_time:.2f} сек.")

    y_test_pred = classifier.predict(X_test_data)

    print("\n--- Classification Report ---")
    print(classification_report(y_test_data, y_test_pred, target_names=class_labels, zero_division=0))
    accuracy = accuracy_score(y_test_data, y_test_pred)
    print(f"Accuracy: {accuracy:.3f}")

    cm = confusion_matrix(y_test_data, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', # Інший колір
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Передбачений клас')
    plt.ylabel('Істинний клас')
    plt.title(f'Матриця плутанини (SVM, ядро={kernel_type.upper()})')
    # plt.savefig(f"confusion_matrix_svm_{kernel_type}.png")
    plt.show()

# --- 5. Тестування різних ядер (на масштабованих даних) ---

# 5.1 Поліноміальне ядро (degree=8)
poly_params = {'degree': poly_degree, 'gamma': 'auto'}
train_evaluate_svm('poly', poly_params, X_train_scaled, y_train, X_test_scaled, y_test)

# 5.2 Гаусове (RBF) ядро
rbf_params = {'gamma': 'auto'} # Використовуємо типове значення 'auto'
train_evaluate_svm('rbf', rbf_params, X_train_scaled, y_train, X_test_scaled, y_test)

# 5.3 Сигмоїдальне ядро
sigmoid_params = {'gamma': 'auto'} # Використовуємо типове значення 'auto'
train_evaluate_svm('sigmoid', sigmoid_params, X_train_scaled, y_train, X_test_scaled, y_test)

print("\n--- Завдання 2.2 Завершено ---")