import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrderByIdRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

class Trade():

  def __init__(self, API_PUB = None, API_SEC=None, is_paper=True):
    self.trading_client = None
    self.account = None
    self.api_pub = API_PUB
    self.api_sec = API_SEC
    self.is_paper = is_paper

  def setup(self):
    self.trading_client = TradingClient(self.api_pub, self.api_sec, paper=self.is_paper)
    self.account = self.trading_client.get_account()
    # print(self.account)

  def place_buy_order(self, ticker, money):
    # preparing orders
    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        notional=money,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                        )

    # Market order
    market_order = self.trading_client.submit_order(
                    order_data=market_order_data
                  )

    # order_id = market_order.id

    # Wait for the order to execute
    # while (True):
    # order_request = GetOrderByIdRequest(nested=False)
    # orders that satisfy params
    # market_order = self.trading_client.get_order_by_id(order_id=order_id, filter=order_request)
    # if market_order.status == OrderStatus.FILLED:
    #     break

    print(market_order)

    return {
      "amount": money,
      "filled_qty": float(market_order.filled_qty)
    }

  def place_sell_order(self, ticker, money):
    # preparing orders
    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        notional=money,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                        )

    # Market order
    market_order = self.trading_client.submit_order(
                    order_data=market_order_data
                  )
    
    # order_id = market_order.id

    # Wait for the order to execute
    # while (True):
    #     order_request = GetOrderByIdRequest(nested=False)
    #     # orders that satisfy params
    #     market_order = self.trading_client.get_order_by_id(order_id=order_id, filter=order_request)
    #     if market_order.status == OrderStatus.FILLED:
    #         break

    print(market_order)
    
    # return {
    #   "amount": float(market_order.filled_avg_price) * float(market_order.filled_qty)
    # }
    return {
      "amount": money,
      "filled_qty": float(market_order.filled_qty)
    }
  
  def close_position(self, ticker):
    positions = self.trading_client.get_all_positions()
    position = next((p for p in positions if p.symbol == ticker), None)
    print(position)

    if position:
        # Close the position
        resp = self.trading_client.close_position(position.symbol)
        return resp
        # print(f"Successfully sold all shares of {position.symbol}")
    else:
        print(f"No open position found for {ticker}")
        return {"filled_qty": 0}
        
