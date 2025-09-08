import unittest
from unittest.mock import MagicMock, patch
from datetime import date
import uuid

# Mock for the update functions
def update_bid_price_dict(username, bid_price_dict):
    pass

def update_buy_dates(username, buy_dates):
    pass

class TestBuyPositions(unittest.TestCase):
    @patch('bot_sell.set_loss_sell_threshold')
    @patch('config_main.TICKERS', ['BTC-USD', 'ETH-USD'])
    def test_buy_positions(self, mock_set_loss_sell_threshold):
        # Mocking dependencies
        api_trading_client = MagicMock()
        user = {
            "username": "test_user",
            "constants": {"multiplier": 2},
            "BUY_DATES": {},
            "BID_PRICE_DICT": {}
        }
        data = {
            "asset_allocation_BTC-USD": 600,
            "asset_allocation_ETH-USD": 400,
            "time": "2024-11-18"
        }

        # Mocking client behavior
        api_trading_client.get_holdings.return_value = {"results": []}
        api_trading_client.get_account.return_value = {"buying_power": 1000}
        api_trading_client.get_best_bid_ask.return_value = {"results": [{"price": "200"}]}
        api_trading_client.place_order.return_value = {"status": "success", "id": "order_12345"}

        # Call the function
        with patch('builtins.print') as mock_print:  # Mock print for cleaner output
            buy_positions(api_trading_client, user, data)

        # Assertions
        api_trading_client.get_holdings.assert_called_once()
        api_trading_client.get_account.assert_called_once()
        api_trading_client.get_best_bid_ask.assert_called()
        api_trading_client.place_order.assert_called()
        self.assertIn("BTC-USD", user["BUY_DATES"])
        self.assertIn("ETH-USD", user["BUY_DATES"])
        self.assertIn("BTC-USD", user["BID_PRICE_DICT"])
        self.assertIn("ETH-USD", user["BID_PRICE_DICT"])

        # Verify threshold setting was called
        mock_set_loss_sell_threshold.assert_called_with(-0.75, "BTC-USD")
        self.assertEqual(mock_set_loss_sell_threshold.call_count, 1)

        # Check print statements
        mock_print.assert_any_call("       ------       ")
        mock_print.assert_any_call("test_user has following open positions: []. It should be [] as everything is sold out")
        mock_print.assert_any_call("       ------       ")

if __name__ == "__main__":
    unittest.main()
