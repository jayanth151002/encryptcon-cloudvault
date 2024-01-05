import boto3
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from auth import Auth
from transaction_service import TransactionService
from typing import Optional
import pandas as pd

from models.create_user import CreateUser
from models.login_user import LoginUser
from models.verify_2fa import Verify2Fa
from models.transaction import Transaction
from models.credit_approver import CreditApprover
from models.loan_approver import LoanApprover

from predictor.test import Test

from approver.credit_main import CreditDataset
from approver.loan_main import LoanDataset

load_dotenv()

predictor = Test()

credit_approver = CreditDataset(
    pd.read_csv("approver\\data\\credit_data.csv")[:100], "Card_Issued"
)

loan_approver = LoanDataset(
    pd.read_csv("approver\\data\\loan_data.csv")[:100],
    "Loan_Status",
    "Loan_Amount",
    "Loan_Amount_Term",
    "Loan_Interest",
    embed_size=32,
)

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


@app.post("/transaction/predict")
def predict_transaction(transaction: Transaction):
    try:
        prediction = predictor.query(transaction.__dict__)
        transaction_service = TransactionService()
        response = transaction_service.add_entry(transaction, prediction)
        return response
    except Exception as e:
        return {"error": str(e)}


@app.get("/transactions")
def predict_transaction(authorization: Optional[str] = Header(None)):
    try:
        if not authorization:
            raise Exception("No authorization header found")
        token = authorization.split("Bearer ")[1]
        verification_response = Auth().verify_token(token)
        if verification_response["success"]:
            transaction_service = TransactionService()
            response = transaction_service.get_transactions(100)
            return response

    except Exception as e:
        return {"error": str(e)}


@app.post("/credit-approver")
def credit_approver_fn(credit_data: CreditApprover):
    try:
        credit_approver_dict = credit_data.__dict__
        res = credit_approver.predict(credit_approver_dict)
        return {"success": True, "result": res}
    except Exception as e:
        return {"error": str(e)}


@app.post("/loan-approver")
def loan_approver_fn(loan_data: LoanApprover):
    try:
        loan_approver_dict = loan_data.__dict__
        res = loan_approver.predict(loan_approver_dict)
        return {
            "success": True,
            "reject_loan": res[0],
            "loan_amount": res[1] * 5 + 5,
            "duration": res[2],
            "interest": res[3],
        }
    except Exception as e:
        return {"error": str(e)}
