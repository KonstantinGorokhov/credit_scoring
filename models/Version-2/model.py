"""Скрипт для обучения и оценки моделей кредитного скоринга (Версия 2).

Этот скрипт представляет собой улучшенную и более структурированную
версию `Version-1/model.py`. Он выполняет полный цикл построения
модели машинного обучения:
1.  Загрузка и объединение данных.
2.  Предварительная обработка: удаление неинформативных признаков и
    пропущенных значений.
3.  Разведочный анализ данных (EDA), включая корреляционный анализ.
4.  Создание конвейера (pipeline) для предобработки числовых и
    категориальных признаков.
5.  Разделение данных на обучающую и тестовую выборки.
6.  Балансировка классов с использованием техники SMOTE.
7.  Визуализация данных до и после балансировки с помощью PCA.
8.  Обучение и подбор гиперпараметров для различных моделей:
    - Логистическая регрессия
    - XGBoost
    - CatBoost
    - Случайный лес
9.  Оценка качества моделей по метрике F1-score.
10. Создание, обучение и сохранение финального продакшн-пайплайна,
    включающего предобработку, SMOTE и логистическую регрессию с
    оптимальными параметрами.
"""

# -- ОБЩИЕ ИМПОРТЫ: РАБОТА С ДАННЫМИ И ВИЗУАЛИЗАЦИЯ --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -- ИМПОРТЫ SKLEARN: ПРЕДОБРАБОТКА, МОДЕЛИ, МЕТРИКИ --
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# -- РАБОТА С ДИСБАЛАНСОМ КЛАССОВ --
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -- МОДЕЛИ ГРАДИЕНТНОГО БУСТИНГА --
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# -- СОХРАНЕНИЕ МОДЕЛЕЙ --
import joblib


# ===================================
#      КОНФИГУРАЦИЯ И ЗАГРУЗКА ДАННЫХ
# ===================================

# Определение путей к файлам и директориям
DATA_DIR = '/Users/masha/src/creditscoring-personal/Submodules/credit_scoring/fintech-credit-scoring'
APPLICATION_INFO_FILE = os.path.join(DATA_DIR, 'application_info.csv')
DEFAULT_FLG_FILE = os.path.join(DATA_DIR, 'default_flg.csv')
MODEL_OUTPUT_PATH = "../credit_scoring_pipeline.pkl"

# Загрузка анкетных данных и информации о дефолтах
try:
    application_df = pd.read_csv(APPLICATION_INFO_FILE)
    default_flg_df = pd.read_csv(DEFAULT_FLG_FILE)
except FileNotFoundError as e:
    print(f"Ошибка: Не удалось найти файлы данных. Проверьте пути. {e}")
    exit()

# Объединение двух наборов данных по идентификатору 'id'
df = pd.merge(application_df, default_flg_df, on='id')


# ===================================
#      ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА
# ===================================

# Удаление идентификаторов и неинформативных признаков
df = df.drop(['id', 'application_dt', 'sample_cd', 'gender_cd'], axis=1)

# Анализ пропущенных значений и их удаление
# print(df.isnull().sum())
df = df.dropna()
# print(df.isnull().sum())


# ===================================
#      РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)
# ===================================

# Построение корреляционной матрицы для числовых признаков
corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Корреляционная матрица")
plt.show()


# ===================================
#      ПОДГОТОВКА ДАННЫХ
# ===================================

# Разделение на признаки (X) и целевую переменную (y)
X = df.drop(columns=['default_flg'], axis=1)
y = df['default_flg']

# Определение числовых и категориальных признаков
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Создание конвейера для предобработки признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])


# ===================================
#      РАЗДЕЛЕНИЕ И ПРЕОБРАЗОВАНИЕ ДАННЫХ
# ===================================

# Разделение данных с сохранением пропорций классов
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучение препроцессора и преобразование обучающей выборки
preprocessed_X_train = preprocessor.fit_transform(X_train)
# Преобразование тестовой выборки
preprocessed_X_test = preprocessor.transform(X_test)


# ===================================
#      БАЛАНСИРОВКА КЛАССОВ (SMOTE)
# ===================================

# Визуализация распределения до балансировки
sns.countplot(x=y_train)
plt.title("Распределение классов до SMOTE")
plt.show()

# Применение SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(preprocessed_X_train, y_train)

# Визуализация распределения после балансировки
sns.countplot(x=y_res)
plt.title("Распределение классов после SMOTE")
plt.show()


# ===================================
#      ВИЗУАЛИЗАЦИЯ С ПОМОЩЬЮ PCA
# ===================================

# PCA до SMOTE
pca_before = PCA(n_components=2)
X_pca_before = pca_before.fit_transform(preprocessed_X_train)
sns.scatterplot(x=X_pca_before[:, 0], y=X_pca_before[:, 1], hue=y_train, palette='viridis', alpha=0.7)
plt.title("PCA до SMOTE")
plt.show()

# PCA после SMOTE
pca_after = PCA(n_components=2)
X_pca_after = pca_after.fit_transform(X_res)
sns.scatterplot(x=X_pca_after[:, 0], y=X_pca_after[:, 1], hue=y_res, palette='viridis', alpha=0.7)
plt.title("PCA после SMOTE")
plt.show()


# ===================================
#      ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ
# ===================================

def evaluate_model(model, X_test_data, y_test_data, model_name=""):
    """Оценивает модель и выводит отчет по классификации.

    Args:
        model: Обученная модель.
        X_test_data: Тестовые данные (признаки).
        y_test_data: Тестовые данные (целевая переменная).
        model_name (str): Название модели для вывода.
    """
    y_pred = model.predict(X_test_data)
    print(f"\n--- {model_name} Отчет по классификации ---")
    print(classification_report(y_test_data, y_pred))

# --- 1. Логистическая регрессия ---
lr_model = LogisticRegression(random_state=42)
lr_params = {'C': np.logspace(-4, 2, 20)}
lr_search = RandomizedSearchCV(lr_model, lr_params, n_iter=10, scoring='f1', random_state=42, cv=5, n_jobs=-1)
lr_search.fit(X_res, y_res)
evaluate_model(lr_search.best_estimator_, preprocessed_X_test, y_test, "Логистическая регрессия")
print(f"Лучшие параметры: {lr_search.best_params_}")

# --- 2. XGBoost ---
xgb_model = XGBClassifier(random_state=42)
xgb_params = {
    'max_depth': [7, 9, 10], 'gamma': [0.1, 0.15, 0.3], 'alpha': [0.1, 0.15, 0.3],
    'reg_lambda': [1.5, 2, 2.5], 'learning_rate': [0.02, 0.05, 0.1], 'n_estimators': [200, 300, 400]
}
xgb_search = RandomizedSearchCV(xgb_model, xgb_params, n_iter=10, scoring='f1', cv=4, n_jobs=-1, random_state=42)
xgb_search.fit(X_res, y_res)
evaluate_model(xgb_search.best_estimator_, preprocessed_X_test, y_test, "XGBoost")
print(f"Лучшие параметры: {xgb_search.best_params_}")

# --- 3. CatBoost ---
cat_model = CatBoostClassifier(silent=True, random_state=42)
cat_params = {
    "iterations": [300, 500, 800], "learning_rate": [0.01, 0.03, 0.05],
    "depth": [4, 6, 8], "l2_leaf_reg": [1, 3, 5, 9],
}
cat_search = RandomizedSearchCV(cat_model, cat_params, n_iter=5, scoring='f1', cv=4, random_state=42, n_jobs=-1)
cat_search.fit(X_res, y_res)
evaluate_model(cat_search.best_estimator_, preprocessed_X_test, y_test, "CatBoost")
print(f"Лучшие параметры: {cat_search.best_params_}")

# --- 4. Случайный лес ---
rf_model = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [100, 200, 500], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4], "class_weight": ["balanced", "balanced_subsample"]
}
rf_search = RandomizedSearchCV(rf_model, rf_params, n_iter=5, scoring="f1", cv=4, random_state=42, n_jobs=-1)
rf_search.fit(X_res, y_res)
evaluate_model(rf_search.best_estimator_, preprocessed_X_test, y_test, "Случайный лес")
print(f"Лучшие параметры: {rf_search.best_params_}")


# ===================================
#      СОЗДАНИЕ И СОХРАНЕНИЕ ФИНАЛЬНОГО PIPELINE
# ===================================

# Инициализация логистической регрессии с оптимальными параметрами,
# найденными с помощью RandomizedSearchCV.
final_model = LogisticRegression(
    C=lr_search.best_params_['C'],
    max_iter=1000,
    random_state=42
)

# Создание финального pipeline, который включает предобработку,
# балансировку SMOTE и модель.
final_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', final_model)
])

# Обучение финального pipeline на всем обучающем наборе данных.
final_pipeline.fit(X_train, y_train)

# Оценка финального pipeline на тестовых данных.
evaluate_model(final_pipeline, X_test, y_test, "Финальный Pipeline (Logistic Regression + SMOTE)")

# Сохранение обученного pipeline на диск.
joblib.dump(final_pipeline, MODEL_OUTPUT_PATH)
print(f"\nФинальный pipeline сохранен по пути: {MODEL_OUTPUT_PATH}")

# Вывод используемых признаков для информации
print("\nИспользуемые числовые признаки:", list(numeric_features))
print("Используемые категориальные признаки:", list(categorical_features))
