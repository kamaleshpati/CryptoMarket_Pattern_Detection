import os
import requests
from datetime import datetime, timedelta
from zipfile import ZipFile
import pandas as pd

def download_binance_1m_klines(symbol, start_date, end_date, save_path):
    base_url = "https://data.binance.vision/data/futures/um/daily/klines"
    os.makedirs(save_path, exist_ok=True)
    date = datetime.strptime(start_date, "%Y-%m-%d")

    while date <= datetime.strptime(end_date, "%Y-%m-%d"):
        file_name = f"{symbol}-1m-{date.strftime('%Y-%m-%d')}"
        url = f"{base_url}/{symbol}/1m/{file_name}.zip"
        local_zip = os.path.join(save_path, file_name + ".zip")

        print(f"Downloading {url}")
        r = requests.get(url)
        if r.status_code == 200:
            with open(local_zip, "wb") as f:
                f.write(r.content)
            with ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(save_path)
            os.remove(local_zip)
        else:
            print(f"File not found for {date.strftime('%Y-%m-%d')}")

        date += timedelta(days=1)


