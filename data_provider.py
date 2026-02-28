import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta

class DataProvider:
    def __init__(self, storage_dir='data'):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def fetch_price_data(self, symbol, start_date, end_date):
        """
        Fetch historical daily price data for a given stock symbol.
        """
        try:
            # symbol example: '600519'
            df = ak.stock_zh_a_hist(symbol=symbol, period='daily',
                                    start_date=start_date, end_date=end_date, adjust='qfq')
            filename = os.path.join(self.storage_dir, f'{symbol}_price_{start_date}_{end_date}.csv')
            df.to_csv(filename, index=False)
            print(f"Price data saved to {filename}")
            return df
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return None

    def fetch_news_data(self, symbol):
        """
        Fetch recent news and comments for a given stock symbol.
        """
        try:
            # stock_news_em returns recent 100 news
            df = ak.stock_news_em(symbol=symbol)
            filename = os.path.join(self.storage_dir, f'{symbol}_news.csv')
            df.to_csv(filename, index=False)
            print(f"News data saved to {filename}")
            return df
        except Exception as e:
            print(f"Error fetching news data: {e}")
            return None

if __name__ == "__main__":
    provider = DataProvider()
    # Test with Moutai (600519)
    today = datetime.now().strftime('%Y%m%d')
    last_week = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')

    # We use a fixed date range for testing to avoid connection issues during live calls if possible,
    # but the task requires daily crawling.
    print("Testing Price Fetching...")
    # Using a recent date range that likely has data
    provider.fetch_price_data('600519', '20240101', '20240110')

    print("Testing News Fetching...")
    provider.fetch_news_data('600519')
