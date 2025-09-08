import os, sys
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
from flask import Flask, send_from_directory, jsonify
from flask import request, session
from flask_cors import CORS
from bson import json_util
from server.mongo import find_user, update_user, get_user, get_users, update_buy_dates
import json
import flask_login
# from env.config import get_env
from trade import Trade
from Trader.predictions import get_predicted_stocks, predictions_in_loop, set_buying_power_no_RL
import threading
from copy import deepcopy
from bot_sell import check_and_sell, set_profit_sell_threshold, set_loss_sell_threshold
from bot_buy import check_and_buy
from datetime import date

login_manager = flask_login.LoginManager()
app = Flask(__name__, static_folder='../../stock-frontend/build')
CORS(app)
login_manager.init_app(app)
# app.secret_key = get_env("SERVER_SECRET_KEY")
app.secret_key = "akshat"

class ThreadManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ThreadManager, cls).__new__(cls)
            cls._instance.init_threads()
        return cls._instance

    def init_threads(self):
        self.predictions_in_loop_started = False
        self.check_and_sell_started = False
        self.check_and_buy_started = False

    def start_threads(self):
        if not self.predictions_in_loop_started:
            thread = threading.Thread(target=predictions_in_loop)
            thread.start()
            self.predictions_in_loop_started = True

        if not self.check_and_sell_started:
            sell_thread = threading.Thread(target=check_and_sell)
            sell_thread.start()
            self.check_and_sell_started = True
        
        if not self.check_and_buy_started:
            buy_thread = threading.Thread(target=check_and_buy)
            buy_thread.start()
            self.check_and_buy_started = True

# Create an instance of ThreadManager
thread_manager = ThreadManager()
thread_manager.start_threads()



def parse_json(data):
  return json.loads(json_util.dumps(data))

@app.before_request
def make_session_permanent():
    session.permanent = True

############ USER ROUTES ################

@app.get("/api/v1/user")
@flask_login.login_required
def fetch_user():
  return flask_login.current_user.to_json()

@app.patch("/api/v1/user")
@flask_login.login_required
def update():
  print(request.json)
  input_user = request.json['user']
  user = update_user(flask_login.current_user, input_user)
  return user.to_json()

############ SCRIPT ROUTES ################


#-------------------MANUAL OPERATION---------------------------------------
@app.post("/api/v1/buy_stocks")
@flask_login.login_required
def buy_stocks():
  print(request.json)
  ticker = request.json['ticker']
  money = request.json['value']
  users = get_users()
  for user in users:
    if ticker=='INTU' and user['username']=='akarshit':
      print("NOT buying INTU for Akarshit")
      continue
    # if user['username']=='akarshit':
    #   print("NOT buying for Akarshit")
    #   continue
    alpaca_instance = Trade(user['constants']['API_KEY'], user['constants']['API_SECRET'], user['constants']['IS_PAPER'])
    alpaca_instance.setup()
    amt_to_buy = int(user['constants']['multiplier']) * int(money)
    resp = alpaca_instance.place_buy_order(ticker, str(amt_to_buy))
    print("Response----->  ",resp["filled_qty"])
    # if resp["filled_qty"]>0:
    user["BUY_DATES"][ticker] = str(date.today())
    print("BUY DATES:  ->   ",user['BUY_DATES'])
    update_buy_dates(user["username"], user["BUY_DATES"])
  
  return "{ 'ok': 1}"

@app.post("/api/v1/sell_stocks")
@flask_login.login_required
def sell_stocks():
  print(request.json)
  ticker = request.json['ticker']
  users = get_users()
  for user in users:
    if ticker=='INTU' and user['username']=='akarshit':
      print("NOT selling INTU for Akarshit")
      continue
    # if user['username']=='akarshit':
    #   print("NOT selling for Akarshit")
    #   continue
    alpaca_instance = Trade(user['constants']['API_KEY'], user['constants']['API_SECRET'], user['constants']['IS_PAPER'])
    alpaca_instance.setup()
    resp = alpaca_instance.close_position(ticker)
    # print("Sell_RESP---> ", resp["filled_qty"])
    print(user["BUY_DATES"])
    if ticker in user["BUY_DATES"]:
      del user["BUY_DATES"][ticker]
      print("BUY DATES:  ->   ",user['BUY_DATES'])
      update_buy_dates(user["username"], user["BUY_DATES"])
    
  return "{ 'ok': 1}"

@app.post("/api/v1/get_predictions")
@flask_login.login_required
def get_predictions():
  print(request.json)
  predictions = get_predicted_stocks()
  print(predictions)
  return jsonify(predictions)

@app.post("/api/v1/get_predictions_without_login")
@flask_login.login_required
def get_predictions_without_login():
  print(request.json)
  predictions = get_predicted_stocks()
  return jsonify(predictions)

@app.post("/api/v1/set_threshold_profit")
def set_profit_threshold():
  print(request.json)
  thresh = float(request.json['threshold'])
  # set_profit_sell_threshold(thresh)
  return "{ 'ok': 1}"

@app.post("/api/v1/set_threshold_loss")
def set_loss_threshold():
  print(request.json)
  thresh = float(request.json['threshold'])
  # threshold loss edit: new symbol
  # set_loss_sell_threshold(thresh, 'IGM')
  return "{ 'ok': 1}"

@app.post("/api/v1/change_buying_power_no_RL")
def change_buying_power_no_RL():
  print(request.json)
  bp = float(request.json['bp'])
  set_buying_power_no_RL(bp)
  return "{ 'ok': 1}"

@app.post("/api/v1/get_portfolio")
@flask_login.login_required
def get_portfolio():
  print(request.json)
  users = get_users()
  portfolio = []
  for user in users:
    # if user['username']=='akarshit':
    #   print("NO portfolio for akarshit")
    #   continue
    temp = {}
    user_portfolio = deepcopy(temp)
    user_portfolio["user"] = user["username"]
    alpaca_instance = Trade(user['constants']['API_KEY'], user['constants']['API_SECRET'], user['constants']['IS_PAPER'])
    alpaca_instance.setup()

    # Get the account information
    account = alpaca_instance.trading_client.get_account()
    # Retrieve the portfolio value
    portfolio_value = float(account.equity)
    user_portfolio["DayTradeCount"] = int(account.daytrade_count)
    print("Portfolio Value: $", portfolio_value)
    user_portfolio["Portfolio"] = portfolio_value
    positions = alpaca_instance.trading_client.get_all_positions()
    
    for position in positions:
        qty = float(position.qty)
        avg_entry_price = float(position.avg_entry_price)
        user_portfolio[str(position.symbol)+"_Value"] = float(position.market_value)
        user_portfolio[str(position.symbol)+"_P/L"] = float(position.unrealized_pl)
        unrealized_pl = float(position.unrealized_pl)
        
        if avg_entry_price != 0:
            pl_percentage = (unrealized_pl / (avg_entry_price * qty)) * 100
        else:
            pl_percentage = 0
        user_portfolio[str(position.symbol) + "_PNL"] = pl_percentage
    
    portfolio.append(user_portfolio)

  
  # portfolio = jsonify(portfolio)
  portfolio_str = json.dumps(portfolio)
  print("Portfolio ready to be sent")
  return portfolio_str


############ AUTH ROUTES ################

@app.post("/api/v1/auth/login")
def login():
  print(request.json)
  username = request.json['username']
  password = request.json['password']
  user = find_user(username, password)
  flask_login.login_user(user)
  return user.to_json()

@app.route("/api/v1/logout")
@flask_login.login_required
def logout():
    flask_login.logout_user()
    return ""
@login_manager.user_loader
def load_user(user_id):
    return get_user(user_id)

############ STATIC ASSET ################

# Serve React App
@app.route('/', defaults={'path': ''})

@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')
