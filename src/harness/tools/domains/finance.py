# Ported from ShishirPatil/gorilla BFCL trading_bot, Apache-2.0. Reimplemented as pure-Python ToolEnvironment (no upstream code copied).
"""Finance/brokerage stateful tool domain.

Provides tools that read/write a shared state dict simulating a simple stock
brokerage: account info and balance, orders, a watchlist, a stock/quote
table, market status, and a login-gated trading session. Loosely modeled on
the shape of the BFCL multi-turn ``trading_bot`` simulator (account_info,
orders, stocks, watch_list, transaction_history, market_status), but
reimplemented from scratch as a ``ToolEnvironment`` subclass.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry

_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


@register_domain("finance")
class FinanceEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "account_info": {
                "account_id": "ACC-7788",
                "owner": "Alex Rivera",
                "balance": 10000.0,
                "currency": "USD",
                "binding_card": "1974-2021-4096-5533",
            },
            "credentials": {
                "username": "arivera",
                "password": "Trade#2024",
            },
            "session": {"authenticated": False, "username": None},
            "market_status": "Closed",
            "order_counter": 12447,
            "orders": {
                "ORD-12345": {
                    "order_id": "ORD-12345",
                    "order_type": "Buy",
                    "symbol": "AAPL",
                    "price": 210.65,
                    "amount": 10,
                    "status": "Completed",
                },
                "ORD-12446": {
                    "order_id": "ORD-12446",
                    "order_type": "Sell",
                    "symbol": "GOOG",
                    "price": 2840.56,
                    "amount": 5,
                    "status": "Pending",
                },
                "ORD-12447": {
                    "order_id": "ORD-12447",
                    "order_type": "Buy",
                    "symbol": "TSLA",
                    "price": 650.00,
                    "amount": 3,
                    "status": "Open",
                },
            },
            "stocks": {
                "AAPL": {
                    "symbol": "AAPL",
                    "price": 227.16,
                    "percent_change": 0.17,
                    "volume": 2.552,
                    "ma_5": 227.11,
                    "ma_20": 227.09,
                },
                "GOOG": {
                    "symbol": "GOOG",
                    "price": 2840.34,
                    "percent_change": 0.24,
                    "volume": 1.123,
                    "ma_5": 2835.67,
                    "ma_20": 2842.15,
                },
                "TSLA": {
                    "symbol": "TSLA",
                    "price": 667.92,
                    "percent_change": -0.12,
                    "volume": 1.654,
                    "ma_5": 671.15,
                    "ma_20": 668.20,
                },
                "MSFT": {
                    "symbol": "MSFT",
                    "price": 310.23,
                    "percent_change": 0.09,
                    "volume": 3.234,
                    "ma_5": 309.88,
                    "ma_20": 310.11,
                },
                "NVDA": {
                    "symbol": "NVDA",
                    "price": 220.34,
                    "percent_change": 0.34,
                    "volume": 1.234,
                    "ma_5": 220.45,
                    "ma_20": 220.67,
                },
                "ALPH": {
                    "symbol": "ALPH",
                    "price": 1320.45,
                    "percent_change": -0.08,
                    "volume": 1.567,
                    "ma_5": 1321.12,
                    "ma_20": 1325.78,
                },
                "OMEG": {
                    "symbol": "OMEG",
                    "price": 457.23,
                    "percent_change": 0.12,
                    "volume": 2.345,
                    "ma_5": 456.78,
                    "ma_20": 458.12,
                },
                "QUAS": {
                    "symbol": "QUAS",
                    "price": 725.89,
                    "percent_change": -0.03,
                    "volume": 1.789,
                    "ma_5": 726.45,
                    "ma_20": 728.00,
                },
                "NEPT": {
                    "symbol": "NEPT",
                    "price": 88.34,
                    "percent_change": 0.19,
                    "volume": 0.654,
                    "ma_5": 88.21,
                    "ma_20": 88.67,
                },
                "SYNX": {
                    "symbol": "SYNX",
                    "price": 345.67,
                    "percent_change": 0.11,
                    "volume": 2.112,
                    "ma_5": 345.34,
                    "ma_20": 346.12,
                },
            },
            "watchlist": ["NVDA", "TSLA"],
            "transaction_history": [
                {
                    "type": "deposit",
                    "amount": 5000.0,
                    "timestamp": "2026-09-01 09:15:00",
                },
                {
                    "type": "withdrawal",
                    "amount": 1200.0,
                    "timestamp": "2026-09-03 14:30:00",
                },
            ],
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        def _next_order_id() -> str:
            data["order_counter"] += 1
            return f"ORD-{data['order_counter']}"

        def _log_transaction(kind: str, amount: float) -> None:
            data["transaction_history"].append(
                {
                    "type": kind,
                    "amount": amount,
                    "timestamp": datetime.now(UTC).strftime(_TIMESTAMP_FMT),
                }
            )

        @registry.tool()
        def get_stock_quote(symbol: str) -> str:
            """Get the current price and trading stats for a stock symbol."""
            symbol = symbol.upper()
            if symbol not in data["stocks"]:
                return json.dumps({"error": f"Stock symbol {symbol} not found"})
            return json.dumps(data["stocks"][symbol])

        @registry.tool()
        def get_market_status() -> str:
            """Get whether the market is currently Open or Closed."""
            return json.dumps({"market_status": data["market_status"]})

        @registry.tool()
        def login(username: str, password: str) -> str:
            """Log in to the trading account using a username and password."""
            if data["session"]["authenticated"]:
                return json.dumps({"status": "success", "message": "Already logged in"})
            creds = data["credentials"]
            if username != creds["username"] or password != creds["password"]:
                return json.dumps(
                    {"error": "Invalid username or password", "status_code": 403}
                )
            data["session"]["authenticated"] = True
            data["session"]["username"] = username
            return json.dumps(
                {"status": "success", "message": "Logged in successfully"}
            )

        @registry.tool()
        def logout() -> str:
            """Log out of the current trading session."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {"status": "success", "message": "No user is currently logged in"}
                )
            data["session"]["authenticated"] = False
            data["session"]["username"] = None
            return json.dumps(
                {"status": "success", "message": "Logged out successfully"}
            )

        @registry.tool()
        def get_login_status() -> str:
            """Check whether a user is currently authenticated."""
            return json.dumps({"authenticated": data["session"]["authenticated"]})

        @registry.tool()
        def get_account_info() -> str:
            """Get the authenticated user's account balance and details."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {
                        "error": "Not authenticated. Please log in first.",
                        "status_code": 403,
                    }
                )
            return json.dumps(data["account_info"])

        @registry.tool()
        def fund_account(amount: float) -> str:
            """Deposit funds into the trading account balance."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {
                        "error": "Not authenticated. Please log in first.",
                        "status_code": 403,
                    }
                )
            if amount <= 0:
                return json.dumps({"error": "Funding amount must be positive"})
            data["account_info"]["balance"] = round(
                data["account_info"]["balance"] + amount, 2
            )
            _log_transaction("deposit", amount)
            return json.dumps(
                {
                    "status": "success",
                    "message": "Account funded successfully",
                    "new_balance": data["account_info"]["balance"],
                }
            )

        @registry.tool()
        def withdraw_funds(amount: float) -> str:
            """Withdraw funds from the trading account balance."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {
                        "error": "Not authenticated. Please log in first.",
                        "status_code": 403,
                    }
                )
            if data["market_status"] != "Open":
                return json.dumps(
                    {
                        "error": "Market is closed. Withdrawals are not allowed.",
                        "status_code": 409,
                    }
                )
            if amount <= 0:
                return json.dumps({"error": "Withdrawal amount must be positive"})
            if amount > data["account_info"]["balance"]:
                return json.dumps(
                    {"error": "Insufficient funds for withdrawal", "status_code": 422}
                )
            data["account_info"]["balance"] = round(
                data["account_info"]["balance"] - amount, 2
            )
            _log_transaction("withdrawal", amount)
            return json.dumps(
                {
                    "status": "success",
                    "message": "Withdrawal successful",
                    "new_balance": data["account_info"]["balance"],
                }
            )

        @registry.tool()
        def get_transaction_history(start_date: str = "", end_date: str = "") -> str:
            """Get deposit/withdrawal transaction history, optionally filtered by date range (YYYY-MM-DD)."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {
                        "error": "Not authenticated. Please log in first.",
                        "status_code": 403,
                    }
                )

            def _parse(value: str) -> datetime:
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)

            def _parse_ts(value: str) -> datetime:
                return datetime.strptime(value, _TIMESTAMP_FMT).replace(tzinfo=UTC)

            start = (
                _parse(start_date) if start_date else datetime.min.replace(tzinfo=UTC)
            )
            end = (
                _parse(end_date) + timedelta(days=1)
                if end_date
                else datetime.max.replace(tzinfo=UTC)
            )
            history = [
                t
                for t in data["transaction_history"]
                if start <= _parse_ts(t["timestamp"]) <= end
            ]
            return json.dumps({"transaction_history": history})

        @registry.tool()
        def get_order(order_id: str) -> str:
            """Get the details of a specific order by its order ID."""
            if order_id not in data["orders"]:
                return json.dumps({"error": f"Order {order_id} not found"})
            return json.dumps(data["orders"][order_id])

        @registry.tool()
        def list_orders() -> str:
            """List all orders in the authenticated user's order history."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {
                        "error": "Not authenticated. Please log in first.",
                        "status_code": 403,
                    }
                )
            return json.dumps({"orders": list(data["orders"].values())})

        @registry.tool()
        def place_order(order_type: str, symbol: str, price: float, amount: int) -> str:
            """Place a Buy or Sell order for a stock symbol."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {
                        "error": "Not authenticated. Please log in first.",
                        "status_code": 403,
                    }
                )
            if data["market_status"] != "Open":
                return json.dumps(
                    {
                        "error": "Market is closed. Trading is not allowed.",
                        "status_code": 409,
                    }
                )
            symbol = symbol.upper()
            if symbol not in data["stocks"]:
                return json.dumps({"error": f"Invalid stock symbol: {symbol}"})
            if order_type.capitalize() not in ("Buy", "Sell"):
                return json.dumps({"error": "order_type must be 'Buy' or 'Sell'"})
            if price <= 0 or amount <= 0:
                return json.dumps({"error": "Price and amount must be positive values"})
            if order_type.capitalize() == "Buy":
                total_cost = float(price) * int(amount)
                if total_cost > data["account_info"]["balance"]:
                    return json.dumps(
                        {
                            "error": (
                                f"Insufficient funds: required ${total_cost:.2f} but "
                                f"only ${data['account_info']['balance']:.2f} available"
                            ),
                            "status_code": 422,
                        }
                    )
            order_id = _next_order_id()
            order = {
                "order_id": order_id,
                "order_type": order_type.capitalize(),
                "symbol": symbol,
                "price": float(price),
                "amount": amount,
                "status": "Open",
            }
            data["orders"][order_id] = order
            return json.dumps({**order, "status": "Pending"})

        @registry.tool()
        def cancel_order(order_id: str) -> str:
            """Cancel an existing order by its order ID."""
            if order_id not in data["orders"]:
                return json.dumps({"error": f"Order {order_id} not found"})
            order = data["orders"][order_id]
            if order["status"] == "Completed":
                return json.dumps(
                    {
                        "error": f"Can't cancel order {order_id}. Order is already completed.",
                        "status_code": 409,
                    }
                )
            order["status"] = "Cancelled"
            return json.dumps({"order_id": order_id, "status": "Cancelled"})

        @registry.tool()
        def list_watchlist() -> str:
            """List the stock symbols on the authenticated user's watchlist."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {
                        "error": "Not authenticated. Please log in first.",
                        "status_code": 403,
                    }
                )
            return json.dumps({"watchlist": data["watchlist"]})

        @registry.tool()
        def add_to_watchlist(symbol: str) -> str:
            """Add a stock symbol to the watchlist."""
            symbol = symbol.upper()
            if symbol not in data["stocks"]:
                return json.dumps({"error": f"Invalid stock symbol: {symbol}"})
            if symbol not in data["watchlist"]:
                data["watchlist"].append(symbol)
            return json.dumps({"watchlist": data["watchlist"]})

        @registry.tool()
        def remove_from_watchlist(symbol: str) -> str:
            """Remove a stock symbol from the watchlist."""
            if not data["session"]["authenticated"]:
                return json.dumps(
                    {
                        "error": "Not authenticated. Please log in first.",
                        "status_code": 403,
                    }
                )
            symbol = symbol.upper()
            if symbol not in data["watchlist"]:
                return json.dumps({"error": f"Stock {symbol} not found in watchlist"})
            data["watchlist"].remove(symbol)
            return json.dumps({"watchlist": data["watchlist"]})
