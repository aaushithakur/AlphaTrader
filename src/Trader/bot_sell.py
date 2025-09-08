import datetime
import pytz
import time
from datetime import date
from server.mongo import find_user, update_user, get_user, get_users, update_buy_dates, update_bid_price_dict
from trade import Trade
from trade_crypto import api_trading_client
import uuid

# Constants
SELL_FLAG = {
    'BTC-USD': False,
}
THRESHOLD_PROFIT = {
    'BTC-USD': 3,
}
THRESHOLD_LOSS = {
    'BTC-USD': -0.75,
}

def get_new_york_time_sell():
    new_york_timezone = pytz.timezone('America/New_York')
    return datetime.datetime.now(new_york_timezone)

def is_within_time_range_sell(start_hour, start_minute, end_hour, end_minute):
    current_time = get_new_york_time_sell()
    start_time = current_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    end_time = current_time.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
    return start_time <= current_time <= end_time

def reset_sell_flag():
    global SELL_FLAG  # Access the global variable
    
    if is_within_time_range_sell(7, 0, 7, 5):
        for ticker in SELL_FLAG:
            if not SELL_FLAG[ticker]:
                SELL_FLAG[ticker] = True
                print(f'SELL_FLAG set to TRUE for {ticker} for the day')
            else:
                print(f'SELL_FLAG is already TRUE for {ticker} for the day')
    print('Waiting till end of the day to set SELL_FLAG')
    

def get_date_difference(start_date, end_date):
    start_time = time.strptime(start_date, "%Y-%m-%d")
    end_time = time.strptime(end_date, "%Y-%m-%d")
    difference = time.mktime(end_time) - time.mktime(start_time)
    return int(difference / (24 * 60 * 60))


def sell_position(api_trading_client, user, holding, current_date):
    global THRESHOLD_LOSS, SELL_FLAG
    symbol = holding['asset_code']+ "-USD"
    try:
        if symbol not in user["BUY_DATES"]:
            print(f'BUY_DATES does not contain the symbol {symbol}')
            return
        
        if symbol not in user['BID_PRICE_DICT']:
            print(f'BID_PRICE_DICT does not contain the symbol {symbol}')
            return
        
        curr_bid_ask = api_trading_client.get_best_bid_ask(symbol)
        curr_bid_ask_price = float(curr_bid_ask['results'][0]['price'])
        avg_entry_price = user['BID_PRICE_DICT'][symbol]
        
        
        pl_percentage = ((curr_bid_ask_price - avg_entry_price) / avg_entry_price ) * 100
        print('P/L Percentage: ', pl_percentage)
        days_after_buy = get_date_difference(user["BUY_DATES"][symbol], current_date)
        print(f"Stock {symbol} has been held for {days_after_buy} days")

        if (pl_percentage >= THRESHOLD_PROFIT[symbol] and SELL_FLAG[symbol]):
            SELL_FLAG[symbol] = False
            sell_qty = round(float(holding['quantity_available_for_trading']) / 2, 8)
            order = api_trading_client.place_order(
                  str(uuid.uuid4()),
                  "sell",
                  "market",
                  symbol,
                  {"asset_quantity": str(sell_qty)}
            )
            print(order)
            
        elif ((pl_percentage <= THRESHOLD_LOSS[symbol])):
            asset_qty = round(float(holding['quantity_available_for_trading']), 8)
            order = api_trading_client.place_order(
                  str(uuid.uuid4()),
                  "sell",
                  "market",
                  symbol,
                  {"asset_quantity": str(asset_qty)}
            )
            print(order)
            del user["BID_PRICE_DICT"][symbol]
            del user["BUY_DATES"][symbol]
            update_bid_price_dict(user["username"], user["BID_PRICE_DICT"])
            update_buy_dates(user["username"], user["BUY_DATES"])
            print(f"Sold position in {symbol}. Updated BUY_DATES: {user['BUY_DATES']}")
    except Exception as e:
        print(f"Error during position sell: {e}")

def check_and_sell():
    global THRESHOLD_LOSS
    while True:
        reset_sell_flag()
        try:
            current_date = str(date.today())
            users = get_users()
        except Exception as e:
            print(f"Error fetching users from database in selling thread: {e}")
            time.sleep(100)
            continue

        for user in users:
            try:
                holdings = api_trading_client.get_holdings()
                for holding in holdings['results']:
                    sell_position(api_trading_client, user, holding, current_date)
                    
            except Exception as e:
                print(f"Error setting up Alpaca instance or getting positions: {e}")
                time.sleep(100)
                continue

        print(f"Current thresholds: profit={THRESHOLD_PROFIT}, loss={THRESHOLD_LOSS}, SELL_FLAG={SELL_FLAG}")
        print("Current Time: ",  str(time.time()))
        time.sleep(60)

def set_profit_sell_threshold(thresh_profit):
    global THRESHOLD_PROFIT
    THRESHOLD_PROFIT = thresh_profit
    print(f'Threshold profit changed to {thresh_profit}')

def set_loss_sell_threshold(thresh_loss, ticker):
    global THRESHOLD_LOSS
    THRESHOLD_LOSS[ticker] = thresh_loss
    print(f'Threshold for {ticker} loss changed to {thresh_loss}') 
