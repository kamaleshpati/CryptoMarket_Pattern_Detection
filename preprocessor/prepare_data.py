from market_data_downloader import download_binance_1m_klines
from data_merger import merge_binance_csv

if __name__ == "__main__":
    # Step 1: Download Binance data
    download_binance_1m_klines(
        symbol="BTCUSDT",
        start_date="2024-01-01",
        end_date="2025-01-01",  
        save_path="./data/market-data/raw/downloads/BTCUSDT_1m/"
    )
    
    # Step 2: Merge downloaded CSV files into a single file
    merge_binance_csv(
    input_folder="./data/BTCUSDT_1m",
    output_file="./data/market-data/raw/binance_1m.csv"
    )

