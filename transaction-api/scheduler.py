import csv
import random
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, os.getenv("CSV_FILE_PATH"))
database_api_endpoint = os.getenv("TRANSACTION_DATABASE_API")


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


def send_to_database():
    random_entry = get_random_entry()
    simulate_database_entry(random_entry)


if __name__ == "__main__":
    i = 0
    while i < 1e5:
        send_to_database()
        time.sleep(10)
        i += 1
