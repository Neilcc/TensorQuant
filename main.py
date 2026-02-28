from data_provider import DataProvider
from llm_agent import LLMAgent
from backtest_engine import BacktestEngine
import argparse

def main():
    parser = argparse.ArgumentParser(description='A-share Market Sentiment Prediction and Backtesting Engine')
    parser.add_argument('--symbol', type=str, default='600519', help='Stock symbol (e.g., 600519 for Moutai)')
    parser.add_argument('--start', type=str, default='20240101', help='Start date (YYYYMMDD)')
    parser.add_argument('--end', type=str, default='20240110', help='End date (YYYYMMDD)')
    parser.add_argument('--api-key', type=str, help='Volcengine Ark API Key')
    parser.add_argument('--endpoint', type=str, help='Volcengine Ark Endpoint ID')
    args = parser.parse_args()

    # 1. Initialize modules
    provider = DataProvider()
    agent = LLMAgent(api_key=args.api_key)
    if args.endpoint:
        agent.endpoint_id = args.endpoint

    engine = BacktestEngine(provider, agent)

    # 2. Run Backtest & Refinement loop
    print(f"Starting end-to-end sentiment prediction for {args.symbol} from {args.start} to {args.end}")
    engine.run_backtest(args.symbol, args.start, args.end)

    # 3. Print Summary
    if engine.results:
        correct_count = sum(1 for r in engine.results if r['correct'])
        accuracy = (correct_count / len(engine.results)) * 100
        print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct_count}/{len(engine.results)})")
    else:
        print("\nNo backtest results available.")

if __name__ == "__main__":
    main()
