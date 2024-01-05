from fastapi import FastAPI, HTTPException
import csv
import random
import requests
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, os.getenv("CSV_FILE_PATH"))
database_api_endpoint = os.getenv("TRANSACTION_DATABASE_API")
is_sending_enabled = True


def get_random_entry():
    with open(csv_file_path, "r") as file:
        csv_reader = csv.DictReader(file)
        rows = list(csv_reader)
        return random.choice(rows)


def simulate_database_entry(data):
    response = requests.post(database_api_endpoint, json=data)
    if response.status_code == 200:
        return {"message": f"Data sent to database: {data}"}
    else:
        return {"error": "Failed to send data to database"}


async def continuous_send():
    global is_sending_enabled
    while is_sending_enabled:
        await asyncio.sleep(120)
        random_entry = get_random_entry()
        simulate_database_entry(random_entry)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(continuous_send())


@app.post("/send-to-database")
async def trigger_send_to_database():
    random_entry = get_random_entry()
    return simulate_database_entry(random_entry)


@app.post("/pause")
async def pause_continuous_send():
    global is_sending_enabled
    if is_sending_enabled:
        is_sending_enabled = False
        return {"message": "Continuous sending paused"}
    else:
        raise HTTPException(
            status_code=400, detail="Continuous sending is already paused"
        )


@app.post("/resume")
async def resume_continuous_send():
    global is_sending_enabled
    if not is_sending_enabled:
        is_sending_enabled = True
        asyncio.create_task(continuous_send())
        return {"message": "Continuous sending resumed"}
    else:
        raise HTTPException(
            status_code=400, detail="Continuous sending is already running"
        )


@app.post("/stop")
async def stop_continuous_send():
    global is_sending_enabled
    is_sending_enabled = False
    return {"message": "Continuous sending stopped"}
