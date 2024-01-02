import boto3
from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

dynamodb = boto3.client(
    "dynamodb",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
app = FastAPI()

user_table = os.getenv("AWS_USER_TABLE")


@app.get("/")
def default():
    return {"server_active": True}


@app.get("/user/")
def get_user(name: str):
    try:
        key = {"username": {"S": name}}
        response = dynamodb.get_item(
            TableName=user_table,
            Key=key,
        )
        item = response.get("Item", None)
        if item:
            return {"user": item, "method": "GET"}
        else:
            return {"error": "User not found"}
    except Exception as e:
        return {"error": e}


@app.post("/user/")
def post_user(name: str):
    print("user table: ", user_table)
    try:
        item = {"username": {"S": name}}
        response = dynamodb.put_item(
            TableName=user_table,
            Item=item,
        )
        print(response)
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            return {"user": item, "method": "POST"}
        else:
            return {"error": "Failed to add user"}
    except Exception as e:
        return {"error": e}
