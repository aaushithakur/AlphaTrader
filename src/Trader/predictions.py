import pandas as pd
import os, sys
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
sys.path.append(f'{base_path}/ML')
sys.path.append(f'{base_path}/ML/RL/PPO')
from ML.NN.predict import should_buy_NN
from Dataset.yfinance.data_process import get_feature_df
import pickle
import time
import warnings
import threading
import torch
# import Shared.configs.config_main as main_config
from Shared.configs.config_main import *
# import Shared.configs.RL.PPO.PPO_config as ppo_config
from Shared.configs.RL.PPO.PPO_config import *
from Dataset.environment.env import get_environment
from Dataset.yfinance.fetch_data import fetch_stock_data
from ML.Transformers.make_dataset import create_dataset_prod

warnings.filterwarnings("ignore")

# Paths and configurations
base_path = os.path.join(os.getcwd(), 'src')
sys.path.extend([base_path, f'{base_path}/ML'])

write_lock = threading.Lock()
read_lock_ = threading.Lock()

buying_power_no_RL = 0

features_df = {}
model_prediction = {}
model_prediction_trans = {}


def check_days_true(arr):
    return bool(arr[0][-1])


def get_env_RL():
    rl_env_dict  = {}
    for TICKER in TICKERS:
        data = fetch_stock_data(TICKER)
        data.to_csv(f"src/Dataset/stocks/{TICKER}.csv", index=False)
        environment = get_environment(TICKER)
        date_column = environment.pop('Date')
        
        scaler = pickle.load(open(f'{base_path}/Shared/models/RL/PPO/scaler_{TICKER}.pkl', 'rb'))
        scaled_env = pd.DataFrame(scaler.transform(environment), columns=environment.columns)
        
        pca = pickle.load(open(f'{base_path}/Shared/models/RL/PPO/pca_{TICKER}.pkl', 'rb'))
        scaled_env = pd.DataFrame(pca.transform(scaled_env))
        scaled_env.insert(0, 'Date', date_column)
        rl_env_dict[TICKER] = scaled_env
    
    return rl_env_dict


def get_RL_prediction(environments):
    global buying_power_no_RL
    rl_bp_dict = {}
    for TICKER in TICKERS:
        input_RL = torch.FloatTensor(environments[TICKER].tail(1).drop(columns=["Date"]).to_numpy())
        RL_model = torch.load(f"{base_path}/Shared/models/RL/PPO/target_model_{TICKER}.pth")
        RL_model.eval()

        with torch.no_grad():
            action_probs = RL_model(input_RL)
            top_prob_indices = torch.topk(action_probs, k=3)[1].tolist()[0]
            action_1, action_2 = ACTIONS[top_prob_indices[0]], ACTIONS[top_prob_indices[1]]

            buying_power = ((action_1 + action_2) / 3) * 5000 if action_1 == 0 \
                else ((action_1 + action_2) / 2) * 5000
            buying_power = buying_power if buying_power != 0 else buying_power_no_RL
            print(f"Buying power for {TICKER} is {buying_power}")
            rl_bp_dict[TICKER] = buying_power
    
    return rl_bp_dict


def get_transformer_prediction():
    transformer_pred = {}
    for TICKER in TICKERS:
        input_data = torch.Tensor(create_dataset_prod(TICKER, 5))
        model = torch.load(f'{base_path}/Shared/models/Transformers/model_{TICKER}.pth')
        model.eval()
        with torch.no_grad():
            outputs = model(input_data)
            transformer_pred[TICKER] = [pred >= 0.5 for pred in outputs]
            
    return transformer_pred


def predictions_in_loop():
    while True:
        try:
            environments = get_env_RL()
            portfolio_percent = get_RL_prediction(environments)
            transformer_prediction = get_transformer_prediction()
            prediction_dict = {}
            for TICKER in TICKERS:
                features_df[TICKER] = get_feature_df(TICKER, LOOKBACK_INTERVAL)
                model_prediction[TICKER] = [should_buy_NN(TICKER, features_df[TICKER].tail(25))]
                print(f"[{TICKER}] Script buy prediction NN: {model_prediction[TICKER]}")
                stock = TICKER
                open_close = [val for pair in zip(fetch_stock_data(stock)['AdjOpen'].tail(7), \
                                                  fetch_stock_data(stock)['AdjClose'].tail(7)) for val in pair]
                open_close = [0 if pd.isna(val) else val for val in open_close]
                prediction_dict[stock] = open_close

            to_buy_stocks = [ticker for ticker in model_prediction if check_days_true(model_prediction[ticker])]
            
            for ticker in TICKERS:
                if (ticker not in to_buy_stocks) and not transformer_prediction[ticker][-1]:
                    prediction_dict[f'asset_allocation_{ticker}'] = 0 if portfolio_percent[ticker] < 100 \
                        else float(portfolio_percent[ticker])
                else:
                    if ticker in to_buy_stocks and portfolio_percent[ticker] <= 100:
                        print(f"asset_allocation_{ticker}: less than 100 but multiplied by 2")
                        prediction_dict[f'asset_allocation_{ticker}'] = float(portfolio_percent[ticker]) * 2
                    elif int(portfolio_percent[ticker]) >=  500 and (transformer_prediction[ticker][-1]):
                        print(f"Open Price can be low tomorrow. Limiting buy price to 500 for {ticker}")
                        prediction_dict[f'asset_allocation_{ticker}'] = 500
                    else:
                        print(f"asset_allocation_{ticker}: remains same")
                        prediction_dict[f'asset_allocation_{ticker}'] = float(portfolio_percent[ticker])

                # if prediction_dict[f'asset_allocation_{ticker}'] > 500:
                #     prediction_dict[f'asset_allocation_{ticker}'] = 500


            prediction_dict['time'] = time.time()
            
            if 'asset_allocation_SMH' in prediction_dict and prediction_dict[f'asset_allocation_SMH'] > 500:
                prediction_dict[f'asset_allocation_SMH'] = 500

            with write_lock:
                with open('predictions.pickle', 'wb') as file:
                    pickle.dump(prediction_dict, file)

            print("sleeping as data is written in file\n")
            time.sleep(60)

        except Exception as e:
            print("NOT ABLE TO FETCH DATA FROM YAHOO API\nRETRYING\n", e)
            time.sleep(200)


def get_predicted_stocks():
    while True:
        if os.path.isfile("predictions.pickle"):
            with read_lock_:
                with open('predictions.pickle', 'rb') as file:
                    data = pickle.load(file)
            
            print("Dictionary parsed from file:", data)
            print(data)
            return data
        else:
            print("File doesn't exist yet")
            time.sleep(5)


def set_buying_power_no_RL(value):
    global buying_power_no_RL
    buying_power_no_RL = float(value)
    print(f'The buying Power for no RL changed to {buying_power_no_RL}')
