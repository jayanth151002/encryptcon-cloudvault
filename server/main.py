import boto3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from auth import Auth
import requests

from models.create_user import CreateUser
from models.login_user import LoginUser
from models.verify_2fa import Verify2Fa

load_dotenv()

dynamodb = boto3.client(
    "dynamodb",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_table = os.getenv("AWS_USER_TABLE")
ml_model_api = os.getenv("ML_MODEL_API")


@app.get("/")
def default():
    return {"server_active": True}


@app.post("/user/create")
def create_user(create_user: CreateUser):
    try:
        auth = Auth()
        response = auth.create_user(create_user.username, create_user.user_type)
        return response
    except Exception as e:
        return {"error": e}


@app.post("/user/verify-password")
def verify_password(login_user: LoginUser):
    try:
        auth = Auth()
        response = auth.verify_password(login_user.username, login_user.password)
        return response
    except Exception as e:
        return {"error": e}


@app.get("/user/qr")
def get_qr(id: str):
    try:
        auth = Auth()
        response = auth.generate_qrcode(id)
        return response
    except Exception as e:
        return {"error": e}


@app.post("/user/verify-2fa")
def verify_2fa(verify_2fa: Verify2Fa):
    try:
        auth = Auth()
        response = auth.verify_2fa(verify_2fa.id, verify_2fa.code, verify_2fa.password)
        return response
    except Exception as e:
        return {"error": e}


@app.post("/predict")
def create_prediction(id: str, token: str):
    try:
        # auth = Auth()
        # response = auth.verify_token(id, token)
        # if response["status"] == "success":
        response = requests.post(
            ml_model_api,
            json={
                "id": id,
            },
        )
        return response.json()
    # else:
    #     return response
    except Exception as e:
        return {"error": e}
