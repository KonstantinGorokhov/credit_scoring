import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# -- КОНФИГУРАЦИЯ И ЗАГРУЗКА ДАННЫХ --

DATA_DIR = '/Users/masha/src/creditscoring-personal/Submodules/credit_scoring/fintech-credit-scoring'
APPLICATION_INFO_FILE = os.path.join(DATA_DIR, 'application_info.csv')
DEFAULT_FLG_FILE = os.path.join(DATA_DIR, 'default_flg.csv')
MODEL_OUTPUT_PATH = "credit_scoring_pipeline.pkl" # Save in the current directory

try:
    application_df = pd.read_csv(APPLICATION_INFO_FILE)
    default_flg_df = pd.read_csv(DEFAULT_FLG_FILE)
except FileNotFoundError as e:
    print(f"Ошибка: Не удалось найти файлы данных. Проверьте пути. {e}")
    exit()

df = pd.merge(application_df, default_flg_df, on='id')

# -- УДАЛЕНИЕ НЕИНФОРМАТИВНЫХ ПРИЗНАКОВ И ОБРАБОТКА ПРОПУСКОВ --

df = df.drop(['id', 'application_dt', 'sample_cd', 'gender_cd'], axis=1)
df = df.dropna()


# -- ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ И ЦЕЛЕВОЙ ПЕРЕМЕННОЙ --

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

categorical_features = [
    "education_cd",
    "car_own_flg",
    "car_type_flg",
    "good_work_flg",
    "home_address_cd",
    "work_address_cd",
    "Air_flg",
]

X = df.drop(columns=["default_flg"])
y = df["default_flg"]


# -- СОЗДАНИЕ И ОБУЧЕНИЕ ФИНАЛЬНОГО PIPELINE --

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

log_reg = LogisticRegression(
    C=0.03359818286283781,
    tol=0.0001,
    max_iter=100,
    solver="lbfgs"
)

pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", log_reg)
])

# Разделение данных для обучения (хотя в продакшене пайплайн будет обучаться на всех данных)
# Здесь оставляем train_test_split для соответствия исходному коду и воспроизводимости,
# но для чистого продакшена можно обучить на всем X, y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)


# -- СОХРАНЕНИЕ ОБУЧЕННОГО PIPELINE --

joblib.dump(pipeline, MODEL_OUTPUT_PATH)

print(f"Production pipeline saved to {MODEL_OUTPUT_PATH}")
