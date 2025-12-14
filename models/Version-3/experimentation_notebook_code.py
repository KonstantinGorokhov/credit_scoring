# -*- coding: utf-8 -*-
"""
Исследовательский ноутбук для задачи кредитного скоринга.

Цель:
- Провести разведочный анализ данных (EDA).
- Выполнить предобработку признаков и калибровку моделей.
- Сравнить различные алгоритмы классификации.
- Выбрать лучшую модель на основе целевых метрик (F1-score, ROC-AUC).
"""

# -- БЛОК 1: ИМПОРТ БИБЛИОТЕК --

# Работа с данными
import pandas as pd
import numpy as np
import os

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка визуализации
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Компоненты для предобработки и моделирования
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.decomposition import PCA

# Модели машинного обучения
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Техники для работы с несбалансированными выборками
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# -- БЛОК 2: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ --

# Определение констант и путей к данным
DATA_DIR = '/Users/masha/src/creditscoring-personal/Submodules/credit_scoring/fintech-credit-scoring'
APPLICATION_INFO_FILE = os.path.join(DATA_DIR, 'application_info.csv')
DEFAULT_FLG_FILE = os.path.join(DATA_DIR, 'default_flg.csv')

# Загрузка и объединение данных
try:
    application_df = pd.read_csv(APPLICATION_INFO_FILE)
    default_flg_df = pd.read_csv(DEFAULT_FLG_FILE)
    print("Данные успешно загружены.")
except FileNotFoundError as e:
    print(f"Ошибка: Не удалось найти файлы данных. Проверьте пути. {e}")
    exit()

# Объединение данных по 'id' для формирования единого датасета
df = pd.merge(application_df, default_flg_df, on='id')
print(f"Размер объединенного датафрейма: {df.shape}")
display(df.head())


# -- БЛОК 3: РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA) --

print("\n--- Общая информация о типах данных и пропусках ---")
df.info()

print("\n--- Описательные статистики для числовых признаков ---")
display(df.describe())

print("\n--- Проверка на наличие дублирующихся записей ---")
print(f"Количество полных дубликатов строк: {df.duplicated().sum()}")


# -- БЛОК 4: ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ --

# Удаление неинформативных признаков, таких как идентификаторы
# и поля, нерелевантные для предсказания дефолта.
initial_features = df.columns.tolist()
df = df.drop(['id', 'application_dt', 'sample_cd', 'gender_cd'], axis=1)
print(f"\nУдалены неинформативные признаки. Количество признаков до: {len(initial_features)}, после: {len(df.columns)}")

# Анализ и обработка пропущенных значений
print("\n--- Анализ пропущенных значений ---")
missing_values = df.isnull().sum()
display(missing_values[missing_values > 0].sort_values(ascending=False))

# Стратегия обработки пропусков: полное удаление строк (listwise deletion).
# Примечание: Это упрощенный подход. В продуктовых решениях предпочтительны
# методы импутации (например, медиана, мода) или использование моделей,
# устойчивых к пропускам.
df = df.dropna()
print(f"\nРазмер датафрейма после удаления строк с пропусками: {df.shape}")
print(f"Общее количество пропусков после обработки: {df.isnull().sum().sum()}")


# -- БЛОК 5: ВИЗУАЛЬНЫЙ АНАЛИЗ И КОРРЕЛЯЦИИ --

print("\n--- Корреляционная матрица числовых признаков ---")
# Анализ линейных взаимосвязей между числовыми признаками и целевой переменной.
corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title("Корреляционная матрица числовых признаков", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# Выводы: [Заполнить на основе анализа. Например, отметить признаки с высокой
# корреляцией с `default_flg` или сильную мультиколлинеарность.]


# -- БЛОК 6: ФОРМИРОВАНИЕ ВЫБОРОК И ПРИЗНАКОВ --

# Разделение датасета на матрицу признаков (X) и вектор целевой переменной (y).
X = df.drop(columns=['default_flg'], axis=1)
y = df['default_flg']

print(f"\nРазмер матрицы признаков (X): {X.shape}")
print(f"Размер вектора целевой переменной (y): {y.shape}")

# Идентификация числовых и категориальных признаков
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nЧисловые признаки ({len(numeric_features)}): {numeric_features}")
print(f"Категориальные признаки ({len(categorical_features)}): {categorical_features}")

# Анализ мощности (количества уникальных значений) категориальных признаков.
# Это важно для оценки применимости One-Hot Encoding и потенциального
# увеличения размерности.
print("\n--- Анализ мощности категориальных признаков ---")
display(X[categorical_features].nunique().sort_values(ascending=False))
# Выводы: [Заполнить на основе анализа. Например, указать признаки
# с высокой мощностью, которые могут потребовать альтернативных техник кодирования.]


# -- ПРЕДОБРАБОТКА ПРИЗНАКОВ --

# Создаем пайплайны для предобработки числовых и категориальных признаков.
# Числовые признаки: пропуски заполняются медианой, затем стандартизация.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Категориальные признаки: пропуски заполняются модой, затем One-Hot Encoding.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Объединяем пайплайны предобработки с помощью ColumnTransformer.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Оставляем остальные признаки как есть (если есть)
)
print("\nКонфигурация препроцессора:")
display(preprocessor)


# -- РАЗБИЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ --

# Разделение данных с сохранением пропорций классов (stratify=y) для обеспечения
# репрезентативности выборок по целевой переменной.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nРазмер обучающей выборки X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Размер тестовой выборки X_test: {X_test.shape}, y_test: {y_test.shape}")

# Обучение предобработчика только на обучающей выборке во избежание утечки данных.
preprocessed_X_train = preprocessor.fit_transform(X_train)
# Применение предобработки к тестовой выборке.
preprocessed_X_test = preprocessor.transform(X_test)

# Получение имен сгенерированных признаков для анализа (может быть полезно для отладки)
features_names = preprocessor.get_feature_names_out()
print(f"Количество признаков после предобработки: {len(features_names)}")
# print(features_names[:10]) # Для примера, можно вывести первые 10 имен

# -- АНАЛИЗ РАСПРЕДЕЛЕНИЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ --

print("\n--- Распределение целевой переменной до балансировки ---")
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train)
plt.title("Распределение целевой переменной в обучающей выборке")
plt.show()
print(f"Соотношение классов в y_train:\n{y_train.value_counts(normalize=True)}")
print("Наблюдается существенный дисбаланс классов, что требует балансировки.")


# -- БАЛАНСИРОВКА КЛАССОВ (SMOTE) --

# SMOTE (Synthetic Minority Over-sampling Technique) используется для увеличения
# количества объектов миноритарного класса путем генерации синтетических примеров.
print("\n--- Балансировка классов с помощью SMOTE ---")
smote = over_sampling.SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(preprocessed_X_train, y_train)

print(f"Размер обучающей выборки после SMOTE: {X_res.shape}, y_res: {y_res.shape}")
print(f"Соотношение классов в y_res после SMOTE:\n{y_res.value_counts(normalize=True)}")

print("\n--- Распределение целевой переменной после SMOTE ---")
plt.figure(figsize=(6, 4))
sns.countplot(x=y_res)
plt.title("Распределение целевой переменной после SMOTE")
plt.show()
print("Теперь классы сбалансированы, что должно улучшить обучение моделей на миноритарном классе.")


# -- ВИЗУАЛИЗАЦИЯ ДАННЫХ С ПОМОЩЬЮ PCA --

print("\n--- Визуализация данных до и после SMOTE с помощью PCA ---")
# Применяем PCA для снижения размерности до 2 компонентов для визуализации.

# PCA до SMOTE
pca_before = PCA(n_components=2)
X_pca_before = pca_before.fit_transform(preprocessed_X_train)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca_before[:, 0], y=X_pca_before[:, 1], hue=y_train, palette='viridis', alpha=0.7)
plt.title("Данные до SMOTE (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
print("На графике до SMOTE видно, что классы сильно несбалансированы и пересекаются.")

# PCA после SMOTE
pca_after = PCA(n_components=2)
X_pca_after = pca_after.fit_transform(X_res)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca_after[:, 0], y=X_pca_after[:, 1], hue=y_res, palette='viridis', alpha=0.7)
plt.title("Данные после SMOTE (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
print("На графике после SMOTE видно, что синтетические примеры миноритарного класса сгенерированы, улучшая баланс, но возможно увеличивая перекрытие классов.")


# -- БЛОК 7: ПОДГОТОВКА К МОДЕЛИРОВАНИЮ --

def evaluate_model(model, X_test_data, y_test_data, model_name="Модель"):
    """
    Унифицированная функция для оценки качества модели.
    
    Рассчитывает и выводит classification_report, F1-score и ROC-AUC.
    
    :param model: Обученная модель Sklearn-совместимого интерфейса.
    :param X_test_data: Тестовая матрица признаков.
    :param y_test_data: Тестовый вектор целевой переменной.
    :param model_name: Название модели для вывода в отчете.
    :return: Кортеж (f1_score, y_pred, y_pred_proba).
    """
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n--- Отчет о классификации для: {model_name} ---\n")
    print(classification_report(y_test_data, y_pred))

    f1 = f1_score(y_test_data, y_pred)
    print(f"F1-score: {f1:.4f}")

    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test_data, y_pred_proba)
        print(f"ROC-AUC: {roc_auc:.4f}")
    
    return f1, y_pred, y_pred_proba


# -- БЛОК 8: ЭКСПЕРИМЕНТЫ С МОДЕЛЯМИ И ПОДБОР ГИПЕРПАРАМЕТРОВ --

print("\n--- Начало этапа калибровки и сравнения моделей ---")

# Инициализация словаря для сохранения результатов
best_models = {}

# --- 8.1: Логистическая регрессия ---
print("\n[1/4] Модель: Логистическая регрессия (Baseline)")
# Baseline-модель для оценки начального качества
lr_model = LogisticRegression(random_state=42, solver='liblinear')
lr_params = {'C': np.logspace(-4, 2, 20)}

lr_search = RandomizedSearchCV(
    lr_model, lr_params, n_iter=10, scoring='f1', random_state=42, cv=5, n_jobs=-1
)
lr_search.fit(X_res, y_res)
print(f"Лучшие параметры: {lr_search.best_params_}")
lr_f1, _, _ = evaluate_model(lr_search.best_estimator_, preprocessed_X_test, y_test, "Логистическая регрессия")
best_models["Logistic Regression"] = {"model": lr_search.best_estimator_, "f1": lr_f1, "params": lr_search.best_params_}
# Выводы: [Заполнить. Оценить, является ли модель адекватной, какие параметры выбраны.]


# --- 8.2: XGBoost ---
print("\n[2/4] Модель: XGBoost")
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_params = {
    'max_depth': [7, 9, 10],
    'gamma': [0.1, 0.15, 0.3],
    'alpha': [0.1, 0.15, 0.3],
    'reg_lambda': [1.5, 2, 2.5],
    'learning_rate': [0.02, 0.05, 0.1],
    'n_estimators': [200, 300, 400]
}
xgb_search = RandomizedSearchCV(
    xgb_model, xgb_params, n_iter=10, scoring='f1', cv=4, n_jobs=-1, random_state=42
)
xgb_search.fit(X_res, y_res)
print(f"Лучшие параметры: {xgb_search.best_params_}")
xgb_f1, _, _ = evaluate_model(xgb_search.best_estimator_, preprocessed_X_test, y_test, "XGBoost")
best_models["XGBoost"] = {"model": xgb_search.best_estimator_, "f1": xgb_f1, "params": xgb_search.best_params_}
# Выводы: [Заполнить. Сравнить производительность с Logistic Regression.]


# --- 8.3: CatBoost ---
print("\n[3/4] Модель: CatBoost")
cat_model = CatBoostClassifier(random_state=42, silent=True)
cat_params = {
    "iterations": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [1, 3, 5],
    "bagging_temperature": [0, 1],
    "random_strength": [1, 2],
    "border_count": [64, 128],
}
cat_search = RandomizedSearchCV(
    estimator=cat_model, param_distributions=cat_params, n_iter=5, scoring='f1', cv=4, random_state=42, n_jobs=-1
)
cat_search.fit(X_res, y_res)
print(f"Лучшие параметры: {cat_search.best_params_}")
cat_f1, _, _ = evaluate_model(cat_search.best_estimator_, preprocessed_X_test, y_test, "CatBoost")
best_models["CatBoost"] = {"model": cat_search.best_estimator_, "f1": cat_f1, "params": cat_search.best_params_}
# Выводы: [Заполнить. Оценить производительность и сравнить с другими моделями.]


# --- 8.4: Random Forest ---
print("\n[4/4] Модель: Random Forest")
rf_model = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 8, 12, 15],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": ["balanced", "balanced_subsample"]
}
rf_search = RandomizedSearchCV(
    estimator=rf_model, param_distributions=rf_params, n_iter=5, scoring="f1", cv=4, random_state=42, n_jobs=-1
)
rf_search.fit(X_res, y_res)
print(f"Лучшие параметры: {rf_search.best_params_}")
rf_f1, _, _ = evaluate_model(rf_search.best_estimator_, preprocessed_X_test, y_test, "Random Forest")
best_models["Random Forest"] = {"model": rf_search.best_estimator_, "f1": rf_f1, "params": rf_search.best_params_}
# Выводы: [Заполнить. Оценить производительность, особенно с учетом параметра class_weight.]


# -- БЛОК 9: СРАВНЕНИЕ РЕЗУЛЬТАТОВ И ВЫБОР ЛУЧШЕЙ МОДЕЛИ --

print("\n--- Итоговое сравнение моделей по метрике F1-Score ---")
model_f1_scores = {name: data["f1"] for name, data in best_models.items()}
sorted_models = sorted(model_f1_scores.items(), key=lambda item: item[1], reverse=True)

for model_name, f1_score_val in sorted_models:
    print(f"- {model_name}: {f1_score_val:.4f}")

# Определение лучшей модели для дальнейшего анализа
best_model_name = sorted_models[0][0]
best_model_data = best_models[best_model_name]
print(f"\nЛучшая модель по F1-score: {best_model_name}")
print("Оптимальные гиперпараметры:")
display(best_model_data["params"])

# Анализ важности признаков для лучшей модели, если она поддерживает этот атрибут
if hasattr(best_model_data["model"], 'feature_importances_'):
    print(f"\n--- Анализ важности признаков для модели {best_model_name} ---")
    
    final_features_names = preprocessor.get_feature_names_out()
    feature_importances = pd.Series(
        best_model_data["model"].feature_importances_,
        index=final_features_names
    ).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.head(20).values, y=feature_importances.head(20).index)
    plt.title(f"ТОП-20 признаков для модели '{best_model_name}'")
    plt.xlabel("Важность признака (Feature Importance)")
    plt.ylabel("Признак")
    plt.show()
    # Выводы: [Заполнить. Какие признаки наиболее влиятельны? Согласуется ли это
    # с бизнес-логикой и результатами EDA?]

print("\n--- Итоговые выводы по исследованию ---")
# Финальное резюме, объясняющее выбор модели для продакшена.
print("""
[Пример выводов]:
В ходе исследования были протестированы и сравнены четыре модели: Logistic Regression, XGBoost, CatBoost и Random Forest.
Целевой метрикой для сравнения служила F1-мера, так как она является гармоническим средним точности и полноты и хорошо подходит для несбалансированных выборок.

Лучшие результаты продемонстрировала модель [название лучшей модели] с F1-score = [значение].
Анализ важности признаков показал, что ключевую роль играют [...].

Рекомендации:
Для внедрения в продуктовый pipeline рекомендуется использовать модель [название лучшей модели] с гиперпараметрами,
найденными в ходе RandomizedSearchCV.
Дальнейшие шаги могут включать более глубокий feature engineering, использование альтернативных техник семплирования
и более исчерпывающий подбор гиперпараметров с помощью GridSearch.
""")
