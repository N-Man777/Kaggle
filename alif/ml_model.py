import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
from pydantic import BaseModel
from prep_function import test_preprocessing # функция подготовки данных

# Загружаем веса обученной модели
model_weights = np.load("custom_svm_weights.npy")

# Загружаем коэффициенты для препроцессинга
prep_coefs = np.load("prep_coefs.npy", allow_pickle=True)

class Object(BaseModel):
    client_id: int
    gender: str
    age: int
    marital_status: str
    job_position: str
    credit_sum: float
    credit_month: int
    tariff_id: float
    score_shk: float
    education: str
    living_region: str
    monthly_income: float
    credit_count: int
    overdue_credit_count: int
    open_account_flg: int

app = FastAPI()

@app.get('/')
def root():
    return {"message": "Welcome\nPlease put your data"}

@app.post("/predict")
def predict(data: Object):
    req_data = pd.DataFrame(dict(data), index=[0])
    clean_data = test_preprocessing(req_data, prep_coefs)
    predict = np.sign(clean_data.dot(model_weights.reshape(-1, 1))).flatten()
    return{
        "PREDICT: ": predict[0]
    }