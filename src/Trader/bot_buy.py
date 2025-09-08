import datetime
import pytz
import time
import os
import pickle
import sys
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
import threading
import torch
from datetime import date
from server.mongo import get_users, update_buy_dates, update_bid_price_dict
from trade import Trade
# from bot_sell import set_loss_sell_threshold
import Shared.configs.config_main as config_main
from trade_crypto import api_trading_client
import uuid

read_lock = threading.Lock()
device = torch.device("cpu")

def get_new_york_time():
    new_york_timezone = pytz.timezone('America/New_York')
    return datetime.datetime.now(new_york_timezone)

def is_within_time_range(start_hour, start_minute, end_hour, end_minute):
    current_time = get_new_york_time()
    start_time = current_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    end_time = current_time.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
    return start_time <= current_time <= end_time

def read_predictions():
    if os.path.isfile("predictions.pickle"):
        try:
            with read_lock:
                with open('predictions.pickle', 'rb') as file:
                    return pickle.load(file)
        except Exception as e:
            print(f"There were issues reading the predictions pickle file: {e}")
            return None
    return None


def sell_all_positions(api_trading_client, user):
    try:
        holdings = api_trading_client.get_holdings()
        for holding in holdings['results']:
            symbol = holding['asset_code']+ "-USD"
            if symbol not in config_main.TICKERS:
                continue
            asset_qty = round(float(holding['quantity_available_for_trading']), 8)
            order = api_trading_client.place_order(
                  str(uuid.uuid4()),
                  "sell",
                  "market",
                  symbol,
                  {"asset_quantity": str(asset_qty)}
            )
            print(order)
            if symbol in user['BID_PRICE_DICT']:
                del user["BID_PRICE_DICT"][symbol]
                update_bid_price_dict(user["username"], user["BID_PRICE_DICT"])
                
            if symbol in user["BUY_DATES"]:
                del user["BUY_DATES"][symbol]
                update_buy_dates(user["username"], user["BUY_DATES"])
    except Exception as e:
        print(f"Error selling positions for user {user['username']}: {e}")

def buy_positions(api_trading_client, user, data):
    try:
        position_list = []
        holdings = api_trading_client.get_holdings()
        for holding in holdings['results']:
            symbol = holding['asset_code']+ "-USD"
            position_list.append(symbol)
        print("       ------       ")
        print(f'{user["username"]} has following open positions: {position_list}. It should be [] as everything is sold out')
        print("       ------       ")
        for ticker in data.keys():
            if ticker == "time" or ticker.startswith("asset_allocation"):
                continue

            if ticker not in position_list:
                money = float(data[f"asset_allocation_{ticker}"])
                # threshold loss edit: new symbol
                if ticker == 'BTC-USD':
                    from bot_sell import set_loss_sell_threshold
                    set_loss_sell_threshold(-0.75 if money >= 500 else -1, ticker)

                amt_to_buy = int(user['constants']['multiplier']) * (int(money / len(config_main.TICKERS)))
                account_info = api_trading_client.get_account()
                buying_power = float(account_info['buying_power'])
                print('buying power: ', buying_power)
                print('amt to buy: ', amt_to_buy)
                if buying_power > amt_to_buy > 10:
                    bid_ask = api_trading_client.get_best_bid_ask(ticker)
                    bid_ask_price = float(bid_ask['results'][0]['price'])
                    qty_to_buy = round(float ( (amt_to_buy - 20) / bid_ask_price ), 8) # -20 to give some room
                    order = api_trading_client.place_order(
                        str(uuid.uuid4()),
                        "buy",
                        "market",
                        ticker,
                        {"asset_quantity": str(qty_to_buy)}
                    )
                    print(order)
                    user["BUY_DATES"][ticker] = str(date.today())
                    user['BID_PRICE_DICT'][ticker] = bid_ask_price
                    update_bid_price_dict(user["username"], user["BID_PRICE_DICT"])
                    update_buy_dates(user["username"], user["BUY_DATES"])
                else:
                    print(f"Insufficient funds or amount to buy too low for user {user['username']}. Buying power: {buying_power}, Amt to buy: {amt_to_buy} for ticker {ticker}")
    except Exception as e:
        print(f"Error buying positions for user {user['username']}: {e}")

def fetch_users():
    try:
        return get_users()
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []

def check_and_buy():
    while True:
        if not is_within_time_range(6, 56, 7, 0):
            print("The current time in New York is not between 6:56 AM and 7:00 AM.")
            time.sleep(60)
            continue

        print("The current time in New York is between 6:56 AM and 7:00 AM.")
        data = read_predictions()
        if data is None or (time.time() - data.get('time', 0) >= 500):
            print("The data written on prediction file is old or unavailable. SOME PROBLEM! SLEEPING without buying")
            time.sleep(300)
            continue

        users = fetch_users()
        for user in users:
            if api_trading_client:
                sell_all_positions(api_trading_client, user)
                time.sleep(1)
                buy_positions(api_trading_client, user, data)

        time.sleep(2000)

if __name__ == "__main__":
    check_and_buy()
