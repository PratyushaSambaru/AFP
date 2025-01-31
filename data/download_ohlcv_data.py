import os
import pandas as pd
import databento as db

API_KEY = "db-9x94mGuCwFGamACcV7j5JhRGVy9eH"

client = db.Historical(API_KEY)

stocks = [
    # Consumer Discretionary - Large-Cap
    "AMZN", "TSLA", "HD", "MCD",
    # Consumer Discretionary - Mid-Cap
    "ROKU", "YUM", "DLTR", "BURL",
    # Consumer Discretionary - Small-Cap
    "CROX", "PLNT", "AEO", "PLAY",
    # Technology - Large-Cap
    "AAPL", "MSFT", "NVDA", "GOOGL",
    # Technology - Mid-Cap
    "PANW", "ZM", "DDOG", "NET",
    # Technology - Small-Cap
    "SMAR", "RPD", "ZS", "PATH",
    # Index
    "SPY"
]

dataset = "XNAS.ITCH"
schema = "ohlcv-1m"   # 1m ohlcv
start_date = "2019-01-01"
end_date = "2024-12-31"
output_folder = "data\ohlcv"

os.makedirs(output_folder, exist_ok=True)

# Download data
for symbol in stocks:
    output_file = os.path.join(output_folder, f"{symbol}_ohlcv.parquet")
    if not os.path.exists(output_file):
        print(f"File {output_file} does not exist. Downloading data for {symbol}...")
        try:
            # Query the data
            dbn_data = client.timeseries.get_range(
                dataset=dataset,
                schema=schema,
                symbols=[symbol],
                start=start_date,
                end=end_date,
            )

            data = dbn_data.to_df()
            data['timestamp'] = data.index

            data.to_parquet(output_file, index=False)
            print(f"Data for {symbol} successfully saved to {output_file}")
        except Exception as e:
            print(f"Failed to download or save data for {symbol}: {e}")
    else:
        print(f"File {output_file} already exists. Skipping download for {symbol}.")
