"""Basket executor — scale-in tranches + pyramiding.

Execution model:
  T1 (50%): MARKET immediately
  T2 (30%): LIMIT at entry - 0.1%  (cancel if unfilled after 5 bars)
  T3 (20%): MARKET after 3 bars confirm (price holds above entry)

Pyramiding: add 50% at +1R, max 3 pyramids per leg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from logzero import logger

from trading.basket.config import BasketConfig
from trading.basket.signals import BasketSignal


@dataclass
class TrancheResult:
    """Outcome of a single tranche order."""
    tranche: int               # 1, 2, or 3
    order_type: str            # MARKET or LIMIT
    quantity: int
    status: str = "PENDING"    # PENDING, FILLED, CANCELLED
    order_id: str = ""
    fill_price: float = 0.0
    fill_qty: int = 0


@dataclass
class LegExecution:
    """Execution state of a basket leg."""
    signal: BasketSignal
    total_qty: int
    avg_entry: float = 0.0
    filled_qty: int = 0
    tranches: List[TrancheResult] = field(default_factory=list)
    pyramids: int = 0


class BasketExecutor:
    """Execute basket signals with scale-in tranches."""

    def __init__(self, cfg: BasketConfig = None, broker_svc=None):
        self.cfg = cfg or BasketConfig()
        self._broker = broker_svc

    def _ensure_broker(self):
        if self._broker is None:
            from trading.services.broker_service import BrokerService
            self._broker = BrokerService()

    def execute_signal(self, signal: BasketSignal, total_qty: int) -> LegExecution:
        """Execute a signal with 3-tranche scale-in.

        T1: MARKET immediately (50% of qty)
        T2: LIMIT at entry - 0.1% (30%)
        T3: deferred — placed later by manager after 3-bar confirmation (20%)

        Returns LegExecution with T1 filled, T2 placed, T3 pending.
        """
        self._ensure_broker()

        leg = LegExecution(signal=signal, total_qty=total_qty)
        exchange = "NFO" if signal.leg_type == "option" else "NSE"
        product = "CARRYFORWARD" if signal.leg_type == "option" else "INTRADAY"

        # Token resolution
        if signal.leg_type == "option":
            token = signal.option_token
            sym = signal.option_symbol
        else:
            from trading.services.ticker_service import ticker_service
            token = ticker_service.get_token(signal.symbol) or ""
            sym = signal.symbol

        # ── T1: MARKET (50%) ──
        t1_qty = int(total_qty * self.cfg.t1_pct / 100)
        if t1_qty > 0:
            t1 = self._place_order(
                sym, signal.side, t1_qty, signal.entry_price,
                order_type="MARKET", exchange=exchange, product=product, token=token,
            )
            t1.tranche = 1
            leg.tranches.append(t1)

            if t1.status == "FILLED":
                leg.filled_qty += t1.fill_qty
                leg.avg_entry = t1.fill_price

        # ── T2: LIMIT at entry ± 0.1% (30%) ──
        t2_qty = int(total_qty * self.cfg.t2_pct / 100)
        if t2_qty > 0:
            offset = signal.entry_price * 0.001
            if signal.side == "BUY":
                limit_price = round(signal.entry_price - offset, 2)
            else:
                limit_price = round(signal.entry_price + offset, 2)

            t2 = self._place_order(
                sym, signal.side, t2_qty, limit_price,
                order_type="LIMIT", exchange=exchange, product=product, token=token,
            )
            t2.tranche = 2
            leg.tranches.append(t2)

            if t2.status == "FILLED":
                leg.filled_qty += t2.fill_qty
                # Update avg entry
                total_cost = leg.avg_entry * (leg.filled_qty - t2.fill_qty) + t2.fill_price * t2.fill_qty
                leg.avg_entry = round(total_cost / leg.filled_qty, 2) if leg.filled_qty > 0 else 0

        # ── T3: PENDING — deferred for confirmation ──
        t3_qty = total_qty - t1_qty - t2_qty
        if t3_qty > 0:
            t3 = TrancheResult(tranche=3, order_type="MARKET", quantity=t3_qty, status="PENDING")
            leg.tranches.append(t3)

        logger.info(
            f"Basket exec {signal.symbol} {signal.side}: "
            f"filled={leg.filled_qty}/{total_qty} avg={leg.avg_entry:.2f}"
        )
        return leg

    def execute_t3(self, leg: LegExecution) -> bool:
        """Execute the deferred T3 tranche after 3-bar confirmation."""
        self._ensure_broker()

        t3 = next((t for t in leg.tranches if t.tranche == 3 and t.status == "PENDING"), None)
        if not t3:
            return False

        signal = leg.signal
        exchange = "NFO" if signal.leg_type == "option" else "NSE"
        product = "CARRYFORWARD" if signal.leg_type == "option" else "INTRADAY"
        token = signal.option_token if signal.leg_type == "option" else ""
        if not token:
            from trading.services.ticker_service import ticker_service
            token = ticker_service.get_token(signal.symbol) or ""

        result = self._place_order(
            signal.option_symbol if signal.leg_type == "option" else signal.symbol,
            signal.side, t3.quantity, signal.entry_price,
            order_type="MARKET", exchange=exchange, product=product, token=token,
        )

        t3.status = result.status
        t3.order_id = result.order_id
        t3.fill_price = result.fill_price
        t3.fill_qty = result.fill_qty

        if result.status == "FILLED":
            old_cost = leg.avg_entry * leg.filled_qty
            leg.filled_qty += result.fill_qty
            leg.avg_entry = round((old_cost + result.fill_price * result.fill_qty) / leg.filled_qty, 2)

        return result.status == "FILLED"

    def execute_pyramid(self, leg: LegExecution, current_price: float) -> Optional[TrancheResult]:
        """Add a pyramid tranche at current price if conditions met."""
        if leg.pyramids >= self.cfg.max_pyramids:
            return None

        self._ensure_broker()

        add_qty = int(leg.total_qty * self.cfg.pyramid_add_pct / 100)
        if add_qty <= 0:
            return None

        signal = leg.signal
        exchange = "NFO" if signal.leg_type == "option" else "NSE"
        product = "CARRYFORWARD" if signal.leg_type == "option" else "INTRADAY"
        token = signal.option_token if signal.leg_type == "option" else ""
        if not token:
            from trading.services.ticker_service import ticker_service
            token = ticker_service.get_token(signal.symbol) or ""

        result = self._place_order(
            signal.option_symbol if signal.leg_type == "option" else signal.symbol,
            signal.side, add_qty, current_price,
            order_type="MARKET", exchange=exchange, product=product, token=token,
        )

        if result.status == "FILLED":
            old_cost = leg.avg_entry * leg.filled_qty
            leg.filled_qty += result.fill_qty
            leg.avg_entry = round((old_cost + result.fill_price * result.fill_qty) / leg.filled_qty, 2)
            leg.pyramids += 1
            logger.info(f"Basket pyramid #{leg.pyramids} {signal.symbol}: +{result.fill_qty} @ {result.fill_price:.2f}")

        return result

    def close_leg(self, leg: LegExecution, current_price: float, reason: str) -> TrancheResult:
        """Close an entire leg at current price."""
        self._ensure_broker()

        if leg.filled_qty <= 0:
            return TrancheResult(tranche=0, order_type="MARKET", quantity=0, status="CANCELLED")

        signal = leg.signal
        close_side = "SELL" if signal.side == "BUY" else "BUY"
        exchange = "NFO" if signal.leg_type == "option" else "NSE"
        product = "CARRYFORWARD" if signal.leg_type == "option" else "INTRADAY"
        token = signal.option_token if signal.leg_type == "option" else ""
        if not token:
            from trading.services.ticker_service import ticker_service
            token = ticker_service.get_token(signal.symbol) or ""

        result = self._place_order(
            signal.option_symbol if signal.leg_type == "option" else signal.symbol,
            close_side, leg.filled_qty, current_price,
            order_type="MARKET", exchange=exchange, product=product, token=token,
        )

        logger.info(f"Basket close {signal.symbol} ({reason}): {result.status} @ {result.fill_price:.2f}")
        return result

    def _place_order(
        self, symbol: str, side: str, qty: int, price: float,
        order_type: str = "MARKET", exchange: str = "NSE",
        product: str = "INTRADAY", token: str = "",
    ) -> TrancheResult:
        """Place a single order via BrokerService."""
        try:
            result = self._broker.place_order(
                symbol=symbol, side=side, quantity=qty, price=price,
                order_type=order_type, product_type=product,
                exchange=exchange, symbol_token=token,
            )
            return TrancheResult(
                tranche=0,
                order_type=order_type,
                quantity=qty,
                status="FILLED" if result.get("success") else "REJECTED",
                order_id=result.get("order_id", ""),
                fill_price=result.get("fill_price", price),
                fill_qty=result.get("fill_quantity", qty) if result.get("success") else 0,
            )
        except Exception as e:
            logger.error(f"Order failed {symbol} {side} {qty}: {e}")
            return TrancheResult(
                tranche=0, order_type=order_type, quantity=qty,
                status="ERROR",
            )
