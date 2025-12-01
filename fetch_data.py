import requests
import time
import csv
import os
from datetime import datetime

API_URL = "https://api.gold-api.com/price/XAU"
CSV_FILE = "data/gold_price.csv"

def get_gold_price():
    try:
        response = requests.get(API_URL, timeout=5).json()
        price = response.get("price") 
        return float(price)
    except:
        return None

def init_csv():
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "price"])

def append_price():
    price = get_gold_price()
    if price is None:
        print("Failed to fetch price")
        return

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, price])

    print(f"[OK] {ts} → {price}")

if __name__ == "__main__":
    init_csv()
    print("Fetching realtime gold price every 60 seconds...")

    while True:
        append_price()
        time.sleep(60)  
