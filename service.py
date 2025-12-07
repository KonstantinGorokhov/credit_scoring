from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

pipeline = joblib.load("models/credit_scoring_pipeline.pkl")

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
    home_address_cd: int      # <-- исправить
    work_address_cd: int      # <-- исправить
    income: int
    SNA: int
    first_time_cd: int
    Air_flg: str


@app.post("/score")
def score(data: ClientFeatures):
    df = pd.DataFrame([data.dict()])
    pred = pipeline.predict(df)[0]
    approved = not bool(pred)
    return {"approved": approved}
