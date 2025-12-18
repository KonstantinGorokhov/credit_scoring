"""Скрипт для обучения и оценки моделей кредитного скоринга.

Этот скрипт выполняет следующие шаги:
1.  Загрузка и предварительная обработка данных.
2.  Разведочный анализ данных (EDA), включая построение корреляционной матрицы.
3.  Разделение данных на обучающую и тестовую выборки.
4.  Применение техник для балансировки классов (SMOTE).
5.  Обучение и подбор гиперпараметров для различных моделей.
6.  Оценка качества моделей.
7.  Создание и сохранение финального конвейера (pipeline) для использования в продакшене.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных о заявках
data = pd.read_csv('../fintech-credit-scoring/application_info.csv')
# Загрузка данных о дефолтах
flags = pd.read_csv('../fintech-credit-scoring/default_flg.csv')

# Объединение данных по 'id'
df = pd.merge(data, flags, on='id')

# Удаление неинформативных признаков
df = df.drop(['id', 'application_dt', 'sample_cd', 'gender_cd'], axis=1)

# Подсчет пропущенных значений
df.isnull().sum()

# Удаление строк с пропущенными значениями
df = df.dropna()

# Повторная проверка пропущенных значений
df.isnull().sum()

# Построение корреляционной матрицы для числовых признаков
corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Разделение на признаки (X) и целевую переменную (y)
X = df.drop(columns=['default_flg'], axis=1)
y = df['default_flg']

# Выделение числовых признаков
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

# Выделение категориальных признаков
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Подсчет уникальных значений категориальных признаков
X[categorical_features].nunique()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Создание препроцессора для числовых и категориальных признаков
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Предобработка обучающей и тестовой выборок
preprocessed_X_train = preprocessor.fit_transform(X_train)
preprocessed_X_test = preprocessor.transform(X_test)

# Получение названий признаков после предобработки (закомментировано, так как не используется далее)
features_names = preprocessor.get_feature_names_out()

# Визуализация дисбаланса классов до SMOTE
sns.countplot(x=y)
plt.show()

from imblearn import over_sampling

# Инициализация SMOTE для балансировки классов
smote = over_sampling.SMOTE(
    random_state=42
)

# Применение SMOTE к обучающей выборке
X_res, y_res = smote.fit_resample(preprocessed_X_train, y_train)

from sklearn.decomposition import PCA

# PCA для визуализации данных до SMOTE
pca = PCA(n_components=2)
X_pca = pca.fit_transform(preprocessed_X_train)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette='viridis')

# PCA для визуализации данных после SMOTE
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_res)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_res, palette='viridis')

# Визуализация сбалансированных классов после SMOTE
sns.countplot(x=y_res)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# Определение модели логистической регрессии
lr_model = LogisticRegression()

# Пространство гиперпараметров для LogisticRegression
lr_params ={
    'C': np.logspace(-4, 2, 20)
}

# Настройка RandomizedSearchCV для поиска лучших гиперпараметров
lr_search = RandomizedSearchCV(
    lr_model, lr_params, n_iter=10, scoring='f1', random_state=42, cv=5, n_jobs=-1
)

# Обучение модели
lr_search.fit(X_res, y_res)

from sklearn.metrics import classification_report

def evaluate_model(model):
    """Оценивает модель и выводит отчет по классификации.

    Args:
        model: Обученная модель.
    """
    y_pred = model.predict(preprocessed_X_test)
    print(classification_report(y_test,y_pred))

# Оценка лучшей модели логистической регрессии
evaluate_model(lr_search.best_estimator_)

# Вывод лучших параметров
lr_search.best_params_

from xgboost import XGBClassifier

# Определение модели XGBoost
xgb_model = XGBClassifier()

# Пространство гиперпараметров для XGBoost
xgb_params = {
    'max_depth': [7, 9, 10],
    'gamma': [0.1, 0.15, 0.3],
    'alpha': [0.1, 0.15, 0.3],
    'reg_lambda': [1.5, 2, 2.5],
    'learning_rate': [0.02, 0.05, 0.1],
    'n_estimators': [200, 300, 400]
}

# Настройка RandomizedSearchCV для поиска лучших гиперпараметров XGBoost
xgb_search = RandomizedSearchCV(
    xgb_model, xgb_params, n_iter=10, scoring='f1', cv=4, n_jobs=-1, random_state=42
)

# Обучение модели
xgb_search.fit(X_res, y_res)

# Оценка лучшей модели XGBoost
evaluate_model(xgb_search.best_estimator_)

from catboost import CatBoostClassifier

# Определение модели CatBoost
cat_model = CatBoostClassifier(silent=True)

# Пространство гиперпараметров для CatBoost
cat_params = {
    "iterations": [300, 500, 800, 1200, 1500],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "depth": [4, 5, 6, 7, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 7, 9, 15, 20],
    "bagging_temperature": [0, 1, 5, 10],
    "random_strength": [0.5, 1, 2, 3, 5],
    "border_count": [32, 64, 128, 254],
}

# Настройка RandomizedSearchCV для поиска лучших гиперпараметров CatBoost
cat_search = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=cat_params,
    n_iter=5,
    scoring='f1',
    cv=4,
    random_state=42,
    n_jobs=-1,
)

# Обучение модели
cat_search.fit(X_res, y_res)

# Оценка лучшей модели CatBoost
evaluate_model(cat_search.best_estimator_)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# базовая модель RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# пространство поиска гиперпараметров RandomForest
rf_params = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [None, 5, 8, 10, 12, 15, 20],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": ["balanced", "balanced_subsample"]  # ВАЖНО для дисбаланса
}

# RandomizedSearchCV для RandomForest
rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_params,
    n_iter=5,
    scoring="f1",
    cv=4,
    random_state=42,
    n_jobs=-1,
)

# обучение модели
rf_search.fit(X_res, y_res)

# Вывод лучших параметров RandomForest
rf_search.best_params_

# Оценка лучшей модели RandomForest
evaluate_model(rf_search.best_estimator_)

import joblib

from sklearn.linear_model import LogisticRegression

# Инициализация модели логистической регрессии для финального пайплайна
model = LogisticRegression(max_iter=1000)

# Создание финального пайплайна, включающего препроцессор и модель
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

# Обучение финального пайплайна на обучающих данных
pipeline.fit(X_train, y_train)

# Сохранение обученного пайплайна в файл "pipeline.pkl"
joblib.dump(pipeline, "pipeline.pkl")


# === IMPORTS ===
# Импорт необходимых библиотек для второго пайплайна
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import joblib


# === LOAD DATA ===
# Загрузка данных (закомментировано, так как 'df' уже загружен)
# df = pd.read_csv("application_info.csv")  # если нужно — раскомментируй
# предполагаем, что df уже в памяти


# === FEATURES ===
# Определение числовых признаков для второго пайплайна
numeric_features = [
    "age",
    "appl_rej_cnt",
    "Score_bki",
    "out_request_cnt",
    "region_rating",
    "income",
    "SNA",
    "first_time_cd",
]

# Определение категориальных признаков для второго пайплайна
categorical_features = [
    "education_cd",
    "car_own_flg",
    "car_type_flg",
    "good_work_flg",
    "home_address_cd",
    "work_address_cd",
    "Air_flg",
]


# === PREPROCESSOR ===
# Создание трансформера для числовых признаков
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Создание трансформера для категориальных признаков
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# Объединение трансформеров в ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# === MODEL ===
# Инициализация модели логистической регрессии с заданными параметрами
log_reg = LogisticRegression(
    C=0.03359818286283781,
    tol=0.0001,
    max_iter=100,
    solver="lbfgs"
)


# === PIPELINE ===
# Создание второго пайплайна с SMOTE и моделью логистической регрессии
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", log_reg)
])


# === TRAIN/TEST SPLIT ===
# Разделение данных на обучающую и тестовую выборки
X = df.drop(columns=["default_flg"])
y = df["default_flg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# === TRAIN MODEL ===
# Обучение второго пайплайна
pipeline.fit(X_train, y_train)


# === EVALUATE ===
# Оценка второго пайплайна и вывод отчета по классификации
y_pred = pipeline.predict(X_test)
print("\n=== Classification Report ===\n")
print(classification_report(y_test, y_pred))


# === SAVE PIPELINE ===
# Сохранение второго пайплайна
joblib.dump(pipeline, "credit_scoring_pipeline.pkl")
print("\nPipeline saved as credit_scoring_pipeline.pkl")

# Дополнительное сохранение модели
import joblib
joblib.dump(pipeline, "models/credit_scoring_model.pkl")

print("NUMERIC:", numeric_features)
print("CATEGORICAL:", categorical_features)