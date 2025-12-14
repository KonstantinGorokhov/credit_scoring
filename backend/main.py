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
    return int(pipeline.predict(df)[0])


@app.post("/score")
async def score(data: ClientFeatures):
    try:
        df = pd.DataFrame([data.dict()])
        loop = asyncio.get_running_loop()

        prediction = await loop.run_in_executor(
            executor,
            predict_sync,
            df
        )

        return {"approved": not bool(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
