import httpx
import os

API_URL = os.getenv("API_URL", "http://backend:8000/score")

async def score_client(payload: dict) -> dict:
    """Асинхронно отправляет запрос на API скоринга.

    Args:
        payload: Словарь с данными для запроса.

    Returns:
        Словарь, представляющий JSON-ответ от API.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
