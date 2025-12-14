# -- ОБЩИЕ ИМПОРТЫ: РАБОТА С ДАННЫМИ И ВИЗУАЛИЗАЦИЯ --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -- ИМПОРТЫ SKLEARN: ПРЕДОБРАБОТКА, МОДЕЛИ, МЕТРИКИ --

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


# -- РАБОТА С ДИСБАЛАНСОМ КЛАССОВ --

from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# -- МОДЕЛИ ГРАДИЕНТНОГО БУСТИНГА --

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# -- СОХРАНЕНИЕ МОДЕЛЕЙ --

import joblib


# -- КОНФИГУРАЦИЯ И ЗАГРУЗКА ДАННЫХ --

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

df

# -- УДАЛЕНИЕ НЕИНФОРМАТИВНЫХ ПРИЗНАКОВ --

# Удаление идентификаторов и неинформативных признаков
df = df.drop(['id', 'application_dt', 'sample_cd', 'gender_cd'], axis=1)

df


# -- АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ --

# Подсчёт количества пропусков по каждому признаку
df.isnull().sum()

# Удаление строк с пропущенными значениями
df = df.dropna()

# Повторная проверка на пропуски
df.isnull().sum()


# -- КОРРЕЛЯЦИОННЫЙ АНАЛИЗ --

corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()


# -- РАЗДЕЛЕНИЕ НА ПРИЗНАКИ И ЦЕЛЕВУЮ ПЕРЕМЕННУЮ --

# Матрица признаков
X = df.drop(columns=['default_flg'], axis=1)

# Целевая переменная
y = df['default_flg']


# -- ОПРЕДЕЛЕНИЕ ТИПОВ ПРИЗНАКОВ --

# Числовые признаки
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

numeric_features

# Категориальные признаки
categorical_features = X.select_dtypes(include=['object', 'category']).columns

categorical_features

# Количество уникальных значений в категориальных признаках
X[categorical_features].nunique()


# -- ПРЕДОБРАБОТКА ПРИЗНАКОВ --

# Предобработка: числовые + категориальные признаки
preprocessor = ColumnTransformer([
    # Числовые признаки: медиана + стандартизация
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),

    # Категориальные признаки: мода + One-Hot Encoding
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])


# -- РАЗБИЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ --

# Разделение данных с сохранением пропорций классов
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучение предобработчика на train
preprocessed_X_train = preprocessor.fit_transform(X_train)

# Применение предобработки к test
preprocessed_X_test = preprocessor.transform(X_test)

# Получение имён сгенерированных признаков
features_names = preprocessor.get_feature_names_out()


# -- АНАЛИЗ РАСПРЕДЕЛЕНИЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ --

# Визуализация распределения целевой переменной
sns.countplot(x=y)
plt.show()


# -- БАЛАНСИРОВКА КЛАССОВ (SMOTE) --

# Инициализация SMOTE
smote = over_sampling.SMOTE(random_state=42)

# Oversampling обучающей выборки
X_res, y_res = smote.fit_resample(preprocessed_X_train, y_train)


# -- ВИЗУАЛИЗАЦИЯ ДАННЫХ С ПОМОЩЬЮ PCA --

# PCA до SMOTE
pca = PCA(n_components=2)
X_pca = pca.fit_transform(preprocessed_X_train)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette='viridis')

# PCA после SMOTE
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_res)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_res, palette='viridis')

# Проверка баланса классов после SMOTE
sns.countplot(x=y_res)
plt.show()


# -- ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ С ПОДБОРОМ ПАРАМЕТРОВ --

# Базовая модель логистической регрессии
lr_model = LogisticRegression()

# Пространство гиперпараметров
lr_params = {
    'C': np.logspace(-4, 2, 20)
}

# Поиск оптимальных параметров
lr_search = RandomizedSearchCV(
    lr_model,          # базовая модель
    lr_params,         # пространство гиперпараметров
    n_iter=10,         # количество случайных комбинаций
    scoring='f1',      # целевая метрика (F1-score)
    random_state=42,   # фиксируем генератор случайных чисел
    cv=5,              # 5-кратная кросс-валидация
    n_jobs=-1          # использование всех доступных ядер CPU
)

# Обучение модели
lr_search.fit(X_res, y_res)


# Функция оценки модели
def evaluate_model(model):
    y_pred = model.predict(preprocessed_X_test)
    print(classification_report(y_test, y_pred))


# Оценка лучшей модели
evaluate_model(lr_search.best_estimator_)

# Лучшие параметры
lr_search.best_params_

# -- ОБУЧЕНИЕ МОДЕЛИ XGBOOST --

# Инициализация классификатора XGBoost с параметрами по умолчанию
xgb_model = XGBClassifier()

# Пространство поиска гиперпараметров для RandomizedSearchCV
# Подбираются параметры сложности модели, регуляризации и скорости обучения
xgb_params = {
    'max_depth': [7, 9, 10],                 # максимальная глубина деревьев
    'gamma': [0.1, 0.15, 0.3],                # минимальное снижение функции потерь для разбиения
    'alpha': [0.1, 0.15, 0.3],                # L1-регуляризация (разреживание)
    'reg_lambda': [1.5, 2, 2.5],              # L2-регуляризация
    'learning_rate': [0.02, 0.05, 0.1],       # шаг обучения (shrinkage)
    'n_estimators': [200, 300, 400]           # количество деревьев
}

# Настройка случайного поиска гиперпараметров
xgb_search = RandomizedSearchCV(
    xgb_model,                                # базовая модель
    xgb_params,                               # пространство параметров
    n_iter=10,                                # количество случайных конфигураций
    scoring='f1',                             # оптимизация по F1-мере
    cv=4,                                     # кросс-валидация
    n_jobs=-1,                                # использование всех ядер CPU
    random_state=42                           # воспроизводимость результатов
)

# Обучение моделей XGBoost на сбалансированных данных (SMOTE)
xgb_search.fit(X_res, y_res)

# Оценка качества лучшей найденной модели на тестовой выборке
evaluate_model(xgb_search.best_estimator_)


# -- ОБУЧЕНИЕ МОДЕЛИ CATBOOST --

# Инициализация CatBoost-классификатора
# Параметр silent=True отключает логирование в stdout
cat_model = CatBoostClassifier(silent=True)

# Пространство поиска гиперпараметров CatBoost
# Включает параметры сложности модели, регуляризации и стохастичности
cat_params = {
    "iterations": [300, 500, 800, 1200, 1500],     # число деревьев
    "learning_rate": [0.01, 0.02, 0.03, 0.05],     # скорость обучения
    "depth": [4, 5, 6, 7, 8, 10],                  # глубина деревьев
    "l2_leaf_reg": [1, 3, 5, 7, 9, 15, 20],        # L2-регуляризация листьев
    "bagging_temperature": [0, 1, 5, 10],          # степень стохастичности бутстрэппинга
    "random_strength": [0.5, 1, 2, 3, 5],          # уровень случайности при выборе разбиений
    "border_count": [32, 64, 128, 254],             # количество бинов для числовых признаков
}

# Настройка RandomizedSearchCV для CatBoost
cat_search = RandomizedSearchCV(
    estimator=cat_model,                           # базовая модель
    param_distributions=cat_params,                # пространство параметров
    n_iter=5,                                      # число случайных конфигураций
    scoring='f1',                                  # метрика оптимизации
    cv=4,                                          # кросс-валидация
    random_state=42,                               # воспроизводимость
    n_jobs=-1,                                     # параллельное выполнение
)

# Обучение CatBoost на сбалансированных данных
cat_search.fit(X_res, y_res)

# Оценка лучшей модели CatBoost на тестовой выборке
evaluate_model(cat_search.best_estimator_)


# -- ОБУЧЕНИЕ МОДЕЛИ RANDOM FOREST --

# Инициализация случайного леса
rf_model = RandomForestClassifier(random_state=42)

# Пространство поиска гиперпараметров Random Forest
# Учитывает контроль сложности модели и работу с дисбалансом классов
rf_params = {
    "n_estimators": [100, 200, 300, 500, 800],     # количество деревьев
    "max_depth": [None, 5, 8, 10, 12, 15, 20],     # максимальная глубина деревьев
    "min_samples_split": [2, 5, 10, 20, 50],       # минимальное число объектов для разбиения
    "min_samples_leaf": [1, 2, 5, 10, 20],         # минимальное число объектов в листе
    "max_features": ["sqrt", "log2", None],        # число признаков при разбиении
    "bootstrap": [True, False],                    # использование бутстрэппинга
    "class_weight": ["balanced", "balanced_subsample"]  # компенсация дисбаланса классов
}

# Настройка случайного поиска гиперпараметров
rf_search = RandomizedSearchCV(
    estimator=rf_model,                            # базовая модель
    param_distributions=rf_params,                 # пространство параметров
    n_iter=5,                                      # число итераций поиска
    scoring="f1",                                  # целевая метрика
    cv=4,                                          # кросс-валидация
    random_state=42,                               # воспроизводимость
    n_jobs=-1,                                     # параллельные вычисления
)

# Обучение Random Forest на сбалансированных данных
rf_search.fit(X_res, y_res)

# Просмотр лучших гиперпараметров
rf_search.best_params_

# Оценка качества лучшей модели
evaluate_model(rf_search.best_estimator_)


# -- СОХРАНЕНИЕ PIPELINE --

# Инициализация базовой логистической регрессии
model = LogisticRegression(max_iter=1000)

# Формирование pipeline: предобработка + модель
pipeline = Pipeline([
    ("preprocessor", preprocessor),                # этап предобработки признаков
    ("model", model),                              # классификатор
])

# Обучение pipeline на исходных обучающих данных
pipeline.fit(X_train, y_train)

# Сохранение обученного pipeline на диск
joblib.dump(pipeline, "pipeline.pkl")


# -- ФИНАЛЬНЫЙ ПРОДАКШН PIPELINE С SMOTE --

# Явное задание числовых признаков
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

# Явное задание категориальных признаков
categorical_features = [
    "education_cd", 
    "car_own_flg", 
    "car_type_flg",
    "good_work_flg", 
    "home_address_cd", 
    "work_address_cd", 
    "Air_flg",
]

# Пайплайн предобработки числовых признаков
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()), 
])

# Пайплайн предобработки категориальных признаков
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("encoder", OneHotEncoder(handle_unknown="ignore")), 
])

# Объединение пайплайнов предобработки
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features), 
    ]
)

# Инициализация логистической регрессии с оптимальными параметрами
log_reg = LogisticRegression(
    C=0.03359818286283781,          # коэффициент регуляризации
    tol=0.0001,                     # критерий сходимости
    max_iter=100,                   # максимальное число итераций
    solver="lbfgs"                  # оптимизатор
)

# Финальный pipeline: предобработка + SMOTE + модель
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor), # предобработка признаков
    ("smote", SMOTE(random_state=42)),  # балансировка классов
    ("model", log_reg)               # классификатор
])

# Формирование матрицы признаков и целевой переменной
X = df.drop(columns=["default_flg"])
y = df["default_flg"]

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучение финального pipeline
pipeline.fit(X_train, y_train)

# Получение предсказаний на тестовой выборке
y_pred = pipeline.predict(X_test)

# Вывод отчёта по качеству классификации
print("\n=== Classification Report ===\n")
print(classification_report(y_test, y_pred))

# Сохранение финального pipeline
joblib.dump(pipeline, "credit_scoring_pipeline.pkl")
joblib.dump(pipeline, MODEL_OUTPUT_PATH)

# Вывод списка используемых признаков
print("NUMERIC:", numeric_features)
print("CATEGORICAL:", categorical_features)
