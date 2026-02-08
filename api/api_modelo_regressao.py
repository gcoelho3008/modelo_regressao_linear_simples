from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar intania do FastAPI
app = FastAPI()

# Criar uma classe  que terá os dados do request body para a API
class request_body(BaseModel):
    horas_estudo: float

# Carregar o modelo de regressão linear treinado para predição
modelo_pontuacao = joblib.load("modelo_pontuacao.pkl")

@app.post("/predict")
def predict(data: request_body):
    input_features = [[data.horas_estudo]]
    y_pred = modelo_pontuacao.predict(input_features)[0].astype(int)

    return {'pontuacao_teste': y_pred.tolist()}
