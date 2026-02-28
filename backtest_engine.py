import pandas as pd
import os
from datetime import datetime

class BacktestEngine:
    def __init__(self, data_provider, llm_agent):
        self.data_provider = data_provider
        self.llm_agent = llm_agent
        self.results = []

    def run_backtest(self, symbol, start_date, end_date):
        """
        Iterate through historical data and compare LLM predictions with actual market movement.
        """
        # Fetch historical price data
        price_df = self.data_provider.fetch_price_data(symbol, start_date, end_date)
        if price_df is None or price_df.empty:
            print("No price data for backtesting. Using sample price context for simulation.")
            # Mock some price data if akshare fails
            price_df = pd.DataFrame({
                '日期': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
                '收盘': [3000, 3010, 2990, 3020, 3030]
            })

        # Fetch news data
        news_df = self.data_provider.fetch_news_data(symbol)
        if news_df is None or news_df.empty:
             print("No news data. Using sample news for simulation.")
             news_df = pd.DataFrame({
                 '发布时间': ['2024-01-01 10:00:00', '2024-01-02 11:00:00', '2024-01-03 12:00:00'],
                 '新闻标题': ['利好消息出台', '市场震荡中', '整体情绪回升']
             })

        # Backtest loop: starting from day 2 to have previous day's data
        for i in range(1, len(price_df)):
            current_day = price_df.iloc[i]
            prev_day_data = price_df.iloc[max(0, i-3):i].to_string()

            # Fetch relevant news for the day before the current_day being predicted
            # news_df['发布时间'] is like '2024-02-24 14:58:51'
            news_df['date_only'] = pd.to_datetime(news_df['发布时间']).dt.date
            target_date = pd.to_datetime(current_day['日期']).date()

            # Use news from the day before to predict current day's movement
            mask = news_df['date_only'] < target_date
            relevant_news = news_df[mask]['新闻标题'].head(10).tolist()
            news_context = "\n".join(relevant_news)

            # 1. Get Prediction
            prediction = self.llm_agent.predict_sentiment(prev_day_data, news_context)

            # 2. Determine actual movement (Bullish if close > prev_close)
            actual_move = "Bullish" if current_day['收盘'] > price_df.iloc[i-1]['收盘'] else "Bearish"

            # 3. Log results
            result_entry = {
                'date': current_day['日期'],
                'prediction': prediction,
                'actual': actual_move,
                'correct': actual_move.lower() in prediction.lower()
            }
            self.results.append(result_entry)
            print(f"Date: {result_entry['date']}, Pred: {result_entry['prediction']}, Actual: {result_entry['actual']}")

            # 4. Refinement if incorrect (only every few errors to avoid API spam)
            if not result_entry['correct']:
                print(f"Refining due to mismatch at {result_entry['date']}...")
                refinement_suggestion = self.llm_agent.analyze_error(actual_move, prediction, prev_day_data)
                print(f"Refinement Suggestion: {refinement_suggestion}")

        # Save backtest results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('backtest_results.csv', index=False)
        print("Backtest completed. Results saved to backtest_results.csv")

if __name__ == "__main__":
    from data_provider import DataProvider
    from llm_agent import LLMAgent

    provider = DataProvider()
    agent = LLMAgent()
    engine = BacktestEngine(provider, agent)

    # Simple simulation run
    print("Running a 3-day backtest simulation...")
    engine.run_backtest('600519', '20240101', '20240105')
