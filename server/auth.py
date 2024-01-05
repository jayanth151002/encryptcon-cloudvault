from dotenv import load_dotenv
import os
import pyotp
from io import BytesIO
import qrcode
from base64 import b64encode
import jwt
import boto3
import uuid

from datetime import datetime, timedelta
from password import PasswordManager

load_dotenv()

dynamodb = boto3.client(
    "dynamodb",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

user_table = os.getenv("AWS_USER_TABLE")
jwt_secret = os.getenv("JWT_SECRET")


class Auth:
    def __init__(self):
        self.key = os.getenv("TOTP_KEY")
        self.password_manager = PasswordManager()

    def create_user(self, username, user_type):
        try:
            response = dynamodb.scan(
                TableName=user_table,
                FilterExpression="username = :val",
                ExpressionAttributeValues={":val": {"S": username}},
            )
            item = response.get("Items", [])[0]
            if item:
                return {"success": False, "error": "User already exists"}
            else:
                new_password = self.password_manager.generate_password()
                encoded_password, salt = self.password_manager.encode_password(
                    new_password
                )
                item = {
                    "id": {"S": str(uuid.uuid4())},
                    "username": {"S": username},
                    "password": {"S": encoded_password},
                    "salt": {"B": salt},
                    "user_type": {"S": user_type},
                    "qr_scanned": {"BOOL": False},
                }
                dynamodb.put_item(TableName=user_table, Item=item)
                return {
                    "success": True,
                    "credentials": {"username": username, "password": new_password},
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def verify_password(self, username, password):
        try:
            response = dynamodb.scan(
                TableName=user_table,
                FilterExpression="username = :val",
                ExpressionAttributeValues={":val": {"S": username}},
            )
            item = response.get("Items", [])[0]
            if item:
                salt = item.get("salt", None)
                stored_password = item.get("password", None)
                if self.password_manager.verify_password(
                    password, stored_password["S"], salt["B"]
                ):
                    return {
                        "success": True,
                        "user_id": item["id"]["S"],
                        "qr_scanned": item["qr_scanned"]["BOOL"],
                    }
                else:
                    return {"success": False, "error": "Invalid password"}
            else:
                return {"success": False, "error": "User not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_qrcode(self, user_id):
        try:
            try:
                key = {"id": {"S": user_id}}
                response = dynamodb.get_item(
                    TableName=user_table,
                    Key=key,
                )
                user = response.get("Item", None)
                if user:
                    if user["qr_scanned"]["BOOL"]:
                        return {"success": False, "error": "QR code already scanned"}
                else:
                    return {"success": False, "error": "User not found"}
            except Exception as e:
                return {"success": False, "error": "User not found"}
            uri = pyotp.totp.TOTP(self.key).provisioning_uri(
                name=user["username"]["S"], issuer_name="Encryptcon"
            )

            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(uri)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buffered = BytesIO()
            img.save(buffered)
            qr_byte = b64encode(buffered.getvalue()).decode("utf-8")

            dynamodb.update_item(
                TableName=user_table,
                Key={"id": {"S": user_id}},
                UpdateExpression="SET qr_scanned = :q",
                ExpressionAttributeValues={
                    ":q": {"BOOL": True},
                },
            )
            return {"success": True, "qr": qr_byte}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def verify_2fa(self, user_id, code, password):
        try:
            try:
                key = {"id": {"S": user_id}}
                response = dynamodb.get_item(
                    TableName=user_table,
                    Key=key,
                )
                user = response.get("Item", None)
                if user:
                    salt = user.get("salt", None)
                    stored_password = user.get("password", None)
                    if not self.password_manager.verify_password(
                        password, stored_password["S"], salt["B"]
                    ):
                        return {"success": False, "error": "Invalid password"}
                else:
                    return {"success": False, "error": "User not found"}
            except Exception as e:
                return {"success": False, "error": "User not found"}
            totp = pyotp.TOTP(self.key)
            is_valid = totp.verify(code)
            if is_valid:
                payload = {
                    "id": user_id,
                    "exp": datetime.utcnow() + timedelta(hours=24),
                }
                jwt_token = jwt.encode(payload, jwt_secret, algorithm="HS256")
                return {
                    "success": True,
                    "user_id": user["id"]["S"],
                    "user_type": user["user_type"]["S"],
                    "token": jwt_token,
                }
            else:
                return {"success": False, "error": "Invalid code"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def verify_token(self, token):
        try:
            payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
            user_id = payload["id"]
            response = dynamodb.scan(
                TableName=user_table,
                FilterExpression="id = :val",
                ExpressionAttributeValues={":val": {"S": user_id}},
            )
            item = response.get("Items", [])[0]
            if item:
                return {"success": True, "user_id": payload["id"]}
            else:
                raise Exception("User not found")
        except Exception as e:
            return {"success": False, "error": str(e)}
