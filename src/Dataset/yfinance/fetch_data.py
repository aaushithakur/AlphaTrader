import yfinance as yf
import pandas as pd
from copy import deepcopy
import sys
import os
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
import Shared.configs.config_main as main_config

symbols = main_config.TICKERS
start_date = main_config.STOCK_DATA_START
end_date = main_config.STOCK_DATA_END

def fetch_stock_data(ticker, start=start_date, end=end_date):

  tick = yf.Ticker(ticker)
  df = tick.history(start=start, end=end)
  df['Adj Close'] = deepcopy(df['Close'])
  df.drop(columns=['Adj Close'])
  df = df.rename(columns = {
    'Open': 'AdjOpen',
    'High': 'AdjHigh',
    'Low': 'AdjLow',
    'Close': 'AdjClose',
    'Volume': 'AdjVolume',
    })
  df = df.reset_index()
  df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
  df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
  
  df = df[['Date', 'AdjOpen', 'AdjHigh', 'AdjLow', 'AdjClose', 'AdjVolume']]
  #Date	      AdjLow	           AdjOpen	    AdjVolume	 AdjHigh	            AdjClose	         Adjusted Close
  #19-11-2019	66.34750366210940	66.9749984741211	76167200	67.0	            66.57250213623050	   65.41852569580080
  #20-11-2019	65.0999984741211	66.38500213623050	106234400	66.5199966430664	65.79750061035160	   64.65695190429690
  return df

for ticker in symbols:
    data = fetch_stock_data(ticker, start=start_date, end=end_date)
    data.to_csv("src/Dataset/stocks/"+ ticker +".csv", index=False)