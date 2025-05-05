# --- Необхідні імпорти ---
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns # Для кращої візуалізації матриці плутанини
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Імпорт моделей
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
# Додамо імпорт для ігнорування попереджень (опціонально, але прибирає вивід)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # Іноді pandas дає UserWarning

# --- 1. Завантаження та вивчення даних ---

# Завантаження датасету з URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# Назви стовпців, як зазначено в PDF
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
try:
    dataset = pd.read_csv(url, names=names)
except Exception as e:
    print(f"Помилка завантаження даних з URL: {e}")
    print("Спроба завантаження даних з sklearn.datasets...")
    from sklearn.datasets import load_iris
    iris_sk = load_iris()
    dataset = pd.DataFrame(data=np.c_[iris_sk['data'], iris_sk['target']],
                           columns=names[:-1] + ['class_id'])
    target_map = dict(enumerate(iris_sk.target_names))
    dataset['class'] = dataset['class_id'].map(target_map)
    dataset = dataset.drop('class_id', axis=1)
    print("Дані успішно завантажено з sklearn.")


print("--- 1. Аналіз даних ---")
print("Розмір датасету (рядки, стовпці):", dataset.shape)
print("\nСтатистичне зведення:\n", dataset.describe())
print("\nРозподіл екземплярів за класами:\n", dataset.groupby('class').size())

# --- 2. Візуалізація даних ---
print("\n--- 2. Візуалізація даних ---")

# Діаграма розмаху ("скриня з вусами")
plt.figure(figsize=(10, 6))
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(10,6))
plt.suptitle("Діаграма розмаху для кожної ознаки")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Гістограми розподілу
plt.figure(figsize=(10, 6))
dataset.hist(figsize=(10,6))
plt.suptitle("Гістограми розподілу для кожної ознаки")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Матриця діаграм розсіювання
plt.figure(figsize=(10, 8))
scatter_matrix(dataset, figsize=(10,8))
plt.suptitle("Матриця діаграм розсіювання")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- 3. Розділення датасету на навчальну та контрольну вибірки ---
print("\n--- 3. Розділення даних ---")
array = dataset.values
X = array[:,0:4].astype(float) # Ознаки
y = array[:,4] # Класи

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
print(f"Розмір навчального набору X: {X_train.shape}")
print(f"Розмір валідаційного набору X: {X_validation.shape}")


# --- 4. Побудова та оцінка моделей за допомогою крос-валідації ---
print("\n--- 4. Оцінка алгоритмів за допомогою крос-валідації ---")
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
print("Точність (Accuracy) для кожного алгоритму (10-fold cross-validation):")
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Діаграма розмаху для порівняння результатів крос-валідації
plt.figure(figsize=(10, 6))
plt.boxplot(results, tick_labels=names) # Використовуємо tick_labels
plt.title('Порівняння точності алгоритмів (Cross-Validation)')
plt.ylabel('Accuracy')
plt.show()


# --- 5. Вибір моделі, фінальне навчання та оцінка на валідаційному наборі ---
print("\n--- 5. Фінальна оцінка на валідаційному наборі ---")
print("Вибрана модель для фінальної оцінки: SVM")
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\nТочність на валідаційному наборі:", accuracy_score(Y_validation, predictions))

print("\nМатриця плутанини (Validation Set):")
cm_val = confusion_matrix(Y_validation, predictions)
unique_classes = sorted(dataset['class'].unique())
plt.figure(figsize=(6, 5))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_classes, yticklabels=unique_classes)
plt.xlabel('Передбачений клас')
plt.ylabel('Істинний клас')
plt.title('Матриця плутанини (Validation Set - SVM)')
plt.show()

print("\nClassification Report (Validation Set):\n", classification_report(Y_validation, predictions))


# --- 6. Прогноз для нового екземпляра ---
print("\n--- 6. Прогноз для нового екземпляра ---")
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
print(f"Новий екземпляр: {X_new[0]}")

prediction_new = model.predict(X_new)
print(f"Передбачений клас для нового екземпляра: {prediction_new[0]}")


print("\n--- Завдання 2.3 Завершено ---")