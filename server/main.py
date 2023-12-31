from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

test_string: int = os.getenv("TEST_STRING")

app = FastAPI()


@app.get("/")
def default():
    return {"server_active": True, "test_string": test_string}
