import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SMABacktester():
    def __init__(self ,symbol: str , sma_s: int ,sma_l: int , start: str, end: str):
        self.symbol = symbol
        self.sma_s = sma_s
        self.sma_l = sma_l
        self.start = start
        self.end = end
        self.results = None
        self.get_data()

    def get_data(self):
        df = yf.download(self.symbol, start= self.start, end = self.end, auto_adjust= False)
        data = df[["Close"]].copy()
        data["returns"] = np.log(data["Close"]/data["Close"].shift(1))
        data["SMA_S"] = data["Close"].rolling(window = self.sma_s).mean()
        data["SMA_L"] = data["Close"].rolling(window = self.sma_l).mean()
        data = data.dropna()
        self.data = data

        return data

    def test_results(self):
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["returns"] * data["position"].shift(1)
        data = data.dropna()
        data["returns_bh"] = np.exp(data["returns"].cumsum())
        data["returns_strategy"] = np.exp(data["strategy"].cumsum())

        performance = data["returns_strategy"].iloc[-1]
        outperformance = performance - data["returns_bh"].iloc[-1]

        self.results = data
        return round(performance, 6), round(outperformance, 6)

    def plot_results(self):
        if self.results is None:
            print("No results to plot. Please run test_results() first.")
            return
        
        title = f"{self.symbol} | SMA_S={self.sma_s} | SMA_L={self.sma_l}"
        
        self.results[["returns_bh", "returns_strategy"]].plot(title=title, figsize=(12, 8))