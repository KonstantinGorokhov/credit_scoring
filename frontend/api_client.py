import httpx
import os

API_URL = os.getenv("API_URL", "http://backend:8000/score")

async def score_client(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
