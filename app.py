from fastapi import FastAPI, Request, Depends, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

with open("salary_predict.pkl", "rb") as model_file:
    model = pickle.load(model_file)

class InputData(BaseModel):
    remote_ratio:       int
    work_year:          int
    experience_level:   int
    job_title:          int
    company_size:       int
    

app = FastAPI()


@app.get("/")

async def read_root():
    return {"message": "Welcome to the ML model API!"}

@app.post("/predict/")

async def predict(data: InputData):
    try:
        input_data = [data.remote_ratio, data.work_year, data.experience_level, data.job_title, data.company_size]
        prediction = model.predict([input_data])
        return {"predicted Salary is": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
