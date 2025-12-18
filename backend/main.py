"""API для кредитного скоринга.

Предоставляет HTTP-эндпоинты для:
- Оценки кредитоспособности клиента на основе предоставленных данных.
- Проверки работоспособности сервиса.

Использует предварительно обученную модель машинного обучения (pipeline)
для предсказания вероятности дефолта клиента.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0"
)

MODEL_PATH = Path("models/credit_scoring_pipeline.pkl")
if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found")

pipeline = joblib.load(MODEL_PATH)

executor = ThreadPoolExecutor(max_workers=4)


class ClientFeatures(BaseModel):
    """Модель данных для входящих признаков клиента.

    Определяет структуру JSON-запроса, который должен быть отправлен
    для получения кредитного скоринга.
    """
    education_cd: str
    age: int
    car_own_flg: str
    car_type_flg: str
    appl_rej_cnt: int
    good_work_flg: str
    Score_bki: float
    out_request_cnt: int
    region_rating: int
    home_address_cd: int
    work_address_cd: int
    income: int
    SNA: int
    first_time_cd: int
    Air_flg: str


def predict_sync(df: pd.DataFrame) -> int:
    """Выполняет синхронное предсказание с использованием обученного пайплайна.

    Эта функция обертывает вызов модели, чтобы ее можно было запускать
    в отдельном потоке (через ThreadPoolExecutor) для предотвращения
    блокировки цикла событий FastAPI.

    Args:
        df: DataFrame с признаками одного клиента, готовыми для модели.

    Returns:
        Результат предсказания модели (0 - одобрен, 1 - не одобрен).
    """
    return int(pipeline.predict(df)[0])


@app.post("/score")
async def score(data: ClientFeatures):
    """Предсказывает кредитоспособность клиента.

    Принимает данные клиента, преобразует их в DataFrame,
    использует обученную модель для предсказания и возвращает
    статус одобрения кредита.

    Args:
        data: Объект ClientFeatures, содержащий признаки клиента.

    Raises:
        HTTPException: Если происходит ошибка при выполнении скоринга.

    Returns:
        Словарь с результатом одобрения: {"approved": True/False}.
    """
    try:
        df = pd.DataFrame([data.dict()])
        loop = asyncio.get_running_loop()

        prediction = await loop.run_in_executor(
            executor,
            predict_sync,
            df
        )

        # Модель предсказывает 0 для одобрения, 1 для отказа.
        # Преобразуем это в булево значение для удобства API.
        return {"approved": not bool(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Проверяет работоспособность API.

    Возвращает статус "ok", если сервис запущен.

    Returns:
        Словарь со статусом: {"status": "ok"}.
    """
    return {"status": "ok"}
