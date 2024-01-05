from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

allowed_ip = os.getenv("ALLOWED_ORIGIN")


@app.get("/")
def default():
    return {"model_server_active": True}


@app.get("/predict")
def predict(request: Request):
    # if request.client.host != allowed_ip:
    #     raise HTTPException(status_code=403, detail="Forbidden")
    return {"prediction": "some prediction", "request_host": request.client.host}
