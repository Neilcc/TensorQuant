from volcenginesdkarkruntime import Ark
import os
import json

class LLMAgent:
    def __init__(self, api_key=None, base_url="https://ark.cn-beijing.volces.com/api/v3"):
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        if not self.api_key:
            print("Warning: ARK_API_KEY not found in environment. Using a mock prediction.")
        self.client = Ark(api_key=self.api_key, base_url=base_url) if self.api_key else None
        # Example Endpoint ID from user instruction, assuming it needs to be set.
        self.endpoint_id = os.environ.get("ARK_ENDPOINT_ID", "your-endpoint-id")

    def predict_sentiment(self, market_data, news_titles):
        """
        Predict market sentiment for the next day.
        """
        prompt = self._build_prediction_prompt(market_data, news_titles)

        if not self.client:
            # Mock prediction for verification purposes if API key is missing
            return "Bullish (Mocked: No API Key Provided)"

        try:
            completion = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=[
                    {"role": "system", "content": "你是一个资深的量化交易策略师，擅长结合市场数据和舆情进行情绪预测。"},
                    {"role": "user", "content": prompt}
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error during LLM prediction: {e}"

    def analyze_error(self, actual_move, prediction, data_context):
        """
        Refine logic: analyze why a prediction was wrong.
        """
        prompt = f"""
        实际市场走势: {actual_move}
        之前的预测: {prediction}
        当时的背景数据: {data_context}
        请分析为什么之前的预测不准确，并给出改进未来预测提示词（Prompt）的建议。
        """
        if not self.client:
            return "Refinement suggestion (Mocked)"

        try:
            completion = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=[
                    {"role": "system", "content": "你是一个专注于自我优化和改进的AI分析师。"},
                    {"role": "user", "content": prompt}
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error during LLM refinement: {e}"

    def _build_prediction_prompt(self, market_data, news_titles):
        return f"""
        基于以下最近的A股市场数据和新闻“小作文”，请预测下一交易日市场的情绪走向（看涨/看跌/震荡），并给出理由。

        市场数据（最近几日价格）:
        {market_data}

        新闻/小作文标题:
        {news_titles}

        预测结果格式：
        情绪：[看涨/看跌/震荡]
        理由：...
        """

if __name__ == "__main__":
    agent = LLMAgent()
    # Sample data for test
    sample_market = "Day 1: 3000, Day 2: 3010, Day 3: 3005"
    sample_news = "- 某大厂发布重磅利好\n- 市场担忧通胀数据"

    print("Testing LLM Prediction (might be mocked)...")
    result = agent.predict_sentiment(sample_market, sample_news)
    print(f"Prediction result: {result}")
