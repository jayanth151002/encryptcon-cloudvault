from fastapi import HTTPException
from starlette.requests import Request
from dotenv import load_dotenv
import os

load_dotenv()


allowed_ip = os.getenv("ALLOWED_ORIGIN")


async def check_request_origin(request: Request):
    if request.client.host != allowed_ip:
        raise HTTPException(status_code=403, detail="Forbidden")
    return request
