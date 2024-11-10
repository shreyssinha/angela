import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class PairsMonitor:
    def __init__(self, pairs_df, zscore_threshold=2.0):
        """
        Initialize with DataFrame of correlated pairs from your analysis
        pairs_df should have columns: ['asset1', 'asset2', 'correlation']
        """
        self.pairs = pairs_df
        self.zscore_threshold = zscore_threshold
        self.historical_ratios = {}
        self.initialize_historical_ratios()
    
    def initialize_historical_ratios(self):
        """Get historical data to establish baseline ratios"""
        for _, pair in self.pairs.iterrows():
            asset1, asset2 = pair['asset1'], pair['asset2']
            # Get last 30 days of data
            end = datetime.now()
            start = end - timedelta(days=30)
            
            stock1 = yf.download(asset1, start=start, end=end)
            stock2 = yf.download(asset2, start=start, end=end)
            
            ratio = stock1['Close'] / stock2['Close']
            self.historical_ratios[f"{asset1}/{asset2}"] = {
                'mean': ratio.mean(),
                'std': ratio.std()
            }
    
    def check_divergence(self):
        """Check current prices against historical ratios"""
        alerts = []
        
        for _, pair in self.pairs.iterrows():
            asset1, asset2 = pair['asset1'], pair['asset2']
            
            # Get current prices
            current_price1 = yf.Ticker(asset1).info['regularMarketPrice']
            current_price2 = yf.Ticker(asset2).info['regularMarketPrice']
            
            current_ratio = current_price1 / current_price2
            hist_stats = self.historical_ratios[f"{asset1}/{asset2}"]
            
            # Calculate z-score
            zscore = (current_ratio - hist_stats['mean']) / hist_stats['std']
            
            if abs(zscore) > self.zscore_threshold:
                alerts.append({
                    'pair': f"{asset1}/{asset2}",
                    'zscore': zscore,
                    'action': 'SHORT' if zscore > 0 else 'LONG',
                    'timestamp': datetime.now()
                })
        
        return alerts

    def monitor_continuously(self, interval_seconds=60):
        """Run continuous monitoring"""
        print(f"Starting pairs monitoring at {datetime.now()}")
        
        while True:
            try:
                alerts = self.check_divergence()
                
                if alerts:
                    print("\nDivergence Alerts:")
                    for alert in alerts:
                        print(f"Pair: {alert['pair']}")
                        print(f"Z-Score: {alert['zscore']:.2f}")
                        print(f"Suggested Action: {alert['action']}")
                        print(f"Time: {alert['timestamp']}\n")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Error occurred: {e}")
                time.sleep(10)  # Wait before retrying

# Usage example:
if __name__ == "__main__":
    # Load your correlated pairs from your analysis
    pairs_df = pd.read_csv('./research/correlated_pairs.csv')  # You'll need to save this first
    
    # Initialize and run monitor
    monitor = PairsMonitor(pairs_df, zscore_threshold=2.0)
    monitor.monitor_continuously(interval_seconds=60) 