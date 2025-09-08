import pandas as pd
import numpy as np
import sys, os, pickle
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
from Dataset.environment.env import get_ohlcv, get_transformer_train, get_transformer_test
from Shared.configs.config_main import TICKERS, STOCK_DATA_START, STOCK_DATA_END, ROW_DATA
from Shared.configs.Transformer.train_config import MAX_LEN, OUTPUT_SIZE, TRAIN_PROFIT_LOOKAHEAD_DAYS, TRAIN_PROFIT_PERCENT
def get_raw_scaled_data(ticker):
    # Sample DataFrame with OHLCV data and additional features
    # Replace this with your actual DataFrame
    # set training environment
    environment = get_transformer_train(ticker)
    ohlcv = get_ohlcv(ticker, STOCK_DATA_START, STOCK_DATA_END)
    # ohlcv['AdjClose_pct_change'] = ohlcv['AdjClose'].shift(-1).pct_change() * 100
    ohlcv['AdjOpen_pct_change'] = ((ohlcv['AdjLow'].shift(-TRAIN_PROFIT_LOOKAHEAD_DAYS) - ohlcv['AdjClose']) / ohlcv['AdjClose']) * 100
    ohlcv['AdjOpen_pct_change'] = ohlcv['AdjOpen_pct_change'].fillna(0)
    ohlcv['Positive_pct_change'] = ohlcv['AdjOpen_pct_change'] < TRAIN_PROFIT_PERCENT
    return environment, ohlcv.iloc[ROW_DATA+50:-200] # due to 50 dyas moving average

def get_raw_scaled_data_test(ticker, testing_days=200):
    # Sample DataFrame with OHLCV data and additional features
    # Replace this with your actual DataFrame
    # set training environment
    environment = get_transformer_test(ticker)
    scaler = pickle.load(open(f'{base_path}/Shared/models/Transformers/scaler_transformer_data_train_' + ticker + '.pkl','rb'))
    environment = pd.DataFrame(scaler.transform(environment),columns=environment.columns) 
    ohlcv = get_ohlcv(ticker, STOCK_DATA_START, STOCK_DATA_END)
    # ohlcv['AdjClose_pct_change'] = ohlcv['AdjClose'].shift(-1).pct_change() * 100
    ohlcv['AdjOpen_pct_change'] = ((ohlcv['AdjLow'].shift(-TRAIN_PROFIT_LOOKAHEAD_DAYS) - ohlcv['AdjClose']) / ohlcv['AdjClose']) * 100
    ohlcv['AdjOpen_pct_change'] = ohlcv['AdjOpen_pct_change'].fillna(0)
    ohlcv['Positive_pct_change'] = ohlcv['AdjOpen_pct_change'] < TRAIN_PROFIT_PERCENT
    return environment[-testing_days:], ohlcv.iloc[-testing_days:] 

def create_sequences(data, ohlcv_data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - OUTPUT_SIZE):  # OUTPUT_SIZE for predicting next OUTPUT_SIZE days
        X.append(data.iloc[i:i+seq_length].values)
        y.append(ohlcv_data.iloc[i+seq_length:i+seq_length+OUTPUT_SIZE]['Positive_pct_change'].values)  # Closing price for next OUTPUT_SIZE days

    return np.array(X), np.array(y)

def create_X_sequence(data, ohlcv_data, seq_length):
    X = []
    for i in range(len(data) - seq_length):  # OUTPUT_SIZE for predicting next OUTPUT_SIZE days
        X.append(data.iloc[i:i+seq_length].values)
    return np.array(X)

def create_dataset_train(ticker):
    # Define sequence length
    raw_scaled_data, ohlcv_data = get_raw_scaled_data(ticker)
    seq_length = MAX_LEN - 1 # learnable embedding will make it MAX_LEN
    # Create sequences
    X, y = create_sequences(raw_scaled_data, ohlcv_data, seq_length)

    return X, y

def create_dataset_test(ticker):
    # Define sequence length
    raw_scaled_data, ohlcv_data = get_raw_scaled_data_test(ticker)
    seq_length = MAX_LEN - 1 # learnable embedding will make it MAX_LEN
    # Create sequences
    X, y = create_sequences(raw_scaled_data, ohlcv_data, seq_length)

    return X, y

def create_dataset_prod(ticker, sim_days):
    # Define sequence length
    seq_length = MAX_LEN - 1 # learnable embedding will make it MAX_LEN
    raw_scaled_data, ohlcv_data = get_raw_scaled_data_test(ticker, testing_days=sim_days+seq_length)
    # Create sequences
    X = create_X_sequence(raw_scaled_data, ohlcv_data, seq_length)
    return X
