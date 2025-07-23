import os
import glob
import pandas as pd

def merge_binance_csv(input_folder, output_file):
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ]

    all_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    data_frames = []

    for file in all_files:
        try:
            # Read first line to check for header
            with open(file, 'r') as f:
                first_line = f.readline()
            if 'open_time' in first_line.lower():
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(file, header=None)
                df.columns = columns

            # Standardize column names just in case
            df.columns = columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            data_frames.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not data_frames:
        print("No data found.")
        return

    merged_df = pd.concat(data_frames, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(data_frames)} files into {output_file}")

merge_binance_csv(
    input_folder="./data/BTCUSDT_1m",
    output_file="./data/market-data/raw/binance_1m.csv"
)
