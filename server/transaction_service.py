from dotenv import load_dotenv
import os
import boto3
import uuid
from models.transaction import Transaction
from datetime import datetime, timezone


load_dotenv()

dynamodb = boto3.client(
    "dynamodb",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

transaction_table = os.getenv("AWS_TRANSACTION_TABLE")


class TransactionService:
    def __init__(self):
        pass

    def add_entry(self, transaction: Transaction, prediction: int):
        try:
            current_timestamp = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            response = dynamodb.put_item(
                TableName=transaction_table,
                Item={
                    "id": {"S": str(uuid.uuid4())},
                    "Time": {"N": str(transaction.Time)},
                    "Source": {"S": transaction.Source},
                    "Target": {"S": transaction.Target},
                    "Location": {"S": transaction.Location},
                    "Type": {"S": transaction.Type},
                    "Amount": {"N": str(transaction.Amount)},
                    "Labels": {"N": str(transaction.Labels)},
                    "Fraudulent": {"BOOL": bool(prediction)},
                    "timestamp": {"S": current_timestamp},
                },
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
