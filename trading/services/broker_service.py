"""
Broker Execution Service — SmartAPI wrapper with paper mode.

TRADING_MODE=paper → simulate fills, log everything, never touch broker.
TRADING_MODE=live  → real orders via SmartAPI.
"""
import os
import uuid
from datetime import datetime
from typing import Optional

from logzero import logger
from dotenv import load_dotenv

load_dotenv()

TRADING_MODE = os.getenv("TRADING_MODE", "paper")


class BrokerService:
    """
    Execution service. Isolated from planner and risk engine.

    In paper mode: simulates order fills instantly at entry price.
    In live mode: calls SmartAPI placeOrder/modifyOrder/cancelOrder.
    """

    def __init__(self, smart_api=None):
        """
        Args:
            smart_api: SmartConnect instance (required for live mode).
                       If None, auto-resolves from the singleton BrokerClient.
        """
        self._api = smart_api
        self.mode = TRADING_MODE
        logger.info(f"BrokerService initialized in {self.mode.upper()} mode")

        # Auto-wire the singleton broker for live mode
        if self._api is None and self.mode == "live":
            from trading.services.data_service import BrokerClient
            b = BrokerClient.get_instance()
            b.ensure_login()
            self._api = b.smart_api

    # ──────────────────────────────────────────
    # Place order
    # ──────────────────────────────────────────
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
        product_type: str = "INTRADAY",
        exchange: str = "NSE",
        symbol_token: str = "",
    ) -> dict:
        """
        Place a buy/sell order.

        Returns:
            dict with keys: success, order_id, fill_price, fill_quantity, message, mode
        """
        if self.mode == "paper":
            return self._paper_fill(symbol, side, quantity, price)

        # ── Live mode ──
        if self._api is None:
            return {
                "success": False,
                "order_id": "",
                "fill_price": 0.0,
                "fill_quantity": 0,
                "message": "SmartAPI instance not provided for live trading",
                "mode": "live",
            }

        try:
            # NSE equity symbols need -EQ suffix; NFO options use the symbol as-is
            trading_symbol = f"{symbol}-EQ" if exchange == "NSE" else symbol
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": trading_symbol,
                "symboltoken": symbol_token,
                "transactiontype": side,
                "exchange": exchange,
                "ordertype": order_type,
                "producttype": product_type,
                "duration": "DAY",
                "price": str(price),
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(quantity),
            }

            response = self._api.placeOrder(order_params)
            order_id = response if isinstance(response, str) else str(response)

            logger.info(f"LIVE ORDER placed: {side} {quantity}x {symbol} @ {price} | ID: {order_id}")

            return {
                "success": True,
                "order_id": order_id,
                "fill_price": price,  # actual fill comes from order status
                "fill_quantity": quantity,
                "message": f"Live order placed: {order_id}",
                "mode": "live",
            }

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return {
                "success": False,
                "order_id": "",
                "fill_price": 0.0,
                "fill_quantity": 0,
                "message": f"Order failed: {str(e)}",
                "mode": "live",
            }

    # ──────────────────────────────────────────
    # Modify order
    # ──────────────────────────────────────────
    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_quantity: Optional[int] = None,
    ) -> dict:
        if self.mode == "paper":
            logger.info(f"PAPER: Modify order {order_id} | price={new_price}, qty={new_quantity}")
            return {"success": True, "message": f"Paper order {order_id} modified", "mode": "paper"}

        if self._api is None:
            return {"success": False, "message": "No SmartAPI instance", "mode": "live"}

        try:
            params = {"orderid": order_id, "variety": "NORMAL"}
            if new_price is not None:
                params["price"] = str(new_price)
            if new_quantity is not None:
                params["quantity"] = str(new_quantity)

            self._api.modifyOrder(params)
            logger.info(f"LIVE: Order {order_id} modified")
            return {"success": True, "message": f"Order {order_id} modified", "mode": "live"}
        except Exception as e:
            return {"success": False, "message": f"Modify failed: {e}", "mode": "live"}

    # ──────────────────────────────────────────
    # Cancel order
    # ──────────────────────────────────────────
    def cancel_order(self, order_id: str) -> dict:
        if self.mode == "paper":
            logger.info(f"PAPER: Cancel order {order_id}")
            return {"success": True, "message": f"Paper order {order_id} cancelled", "mode": "paper"}

        if self._api is None:
            return {"success": False, "message": "No SmartAPI instance", "mode": "live"}

        try:
            self._api.cancelOrder(order_id, "NORMAL")
            logger.info(f"LIVE: Order {order_id} cancelled")
            return {"success": True, "message": f"Order {order_id} cancelled", "mode": "live"}
        except Exception as e:
            return {"success": False, "message": f"Cancel failed: {e}", "mode": "live"}

    # ──────────────────────────────────────────
    # Order status
    # ──────────────────────────────────────────
    def get_order_status(self, order_id: str) -> dict:
        if self.mode == "paper":
            return {
                "order_id": order_id,
                "status": "FILLED",
                "message": "Paper order — instant fill",
                "mode": "paper",
            }

        if self._api is None:
            return {"order_id": order_id, "status": "UNKNOWN", "message": "No API", "mode": "live"}

        try:
            book = self._api.orderBook()
            if book and "data" in book:
                for order in book["data"]:
                    if order.get("orderid") == order_id:
                        return {
                            "order_id": order_id,
                            "status": order.get("orderstatus", "UNKNOWN"),
                            "message": order.get("text", ""),
                            "mode": "live",
                        }
            return {"order_id": order_id, "status": "NOT_FOUND", "message": "", "mode": "live"}
        except Exception as e:
            return {"order_id": order_id, "status": "ERROR", "message": str(e), "mode": "live"}

    # ──────────────────────────────────────────
    # Paper mode helpers
    # ──────────────────────────────────────────
    def _paper_fill(self, symbol: str, side: str, quantity: int, price: float) -> dict:
        """Simulate an instant fill at the requested price."""
        paper_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        logger.info(
            f"PAPER TRADE: {side} {quantity}x {symbol} @ {price:.2f} | ID: {paper_id}"
        )
        return {
            "success": True,
            "order_id": paper_id,
            "fill_price": price,
            "fill_quantity": quantity,
            "message": f"Paper fill: {side} {quantity}x {symbol} @ {price:.2f}",
            "mode": "paper",
        }
