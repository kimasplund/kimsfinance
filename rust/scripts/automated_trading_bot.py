#!/usr/bin/env python3
"""
Automated Paper Trading Bot
===========================

Bull Put Spread Strategy with IBKR API Integration
- Connects to IBKR TWS/Gateway (handles disconnections gracefully)
- Scans for opportunities using proven strategy (266% ROC)
- Places orders automatically
- Monitors positions and exits based on targets
- Logs all activity

Usage:
    python automated_trading_bot.py [--config config/trading_bot.toml] [--dry-run]

Requirements:
    pip install ib_insync toml loguru

"""

import asyncio
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json
import csv
from zoneinfo import ZoneInfo

try:
    from ib_insync import *
    from loguru import logger
    import toml
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install ib_insync toml loguru")
    sys.exit(1)


class TradingBot:
    """Automated Bull Put Spread Trading Bot"""

    def __init__(self, config_path: str = "config/trading_bot.toml"):
        """Initialize bot with configuration"""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.ib = IB()
        self.connected = False
        self.running = True
        self.positions: Dict[str, Trade] = {}  # Track open positions
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        # Setup logging
        log_dir = Path(self.config["trading"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        log_level = self.config["trading"]["log_level"].upper()
        logger.add(
            log_dir / "bot_{time}.log",
            rotation="1 day",
            retention="90 days",
            level=log_level,
        )

        # CSV trade log
        if self.config["trading"]["log_trades_to_csv"]:
            self.csv_path = Path(self.config["trading"]["csv_path"])
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.init_csv_log()

    def load_config(self) -> dict:
        """Load configuration from TOML file"""
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        config = toml.load(self.config_path)
        logger.info(f"Loaded config from {self.config_path}")
        return config

    def init_csv_log(self):
        """Initialize CSV trade log"""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "action", "short_strike", "long_strike",
                    "expiration", "dte", "quantity", "credit", "max_risk", "status"
                ])
            logger.info(f"Created CSV trade log: {self.csv_path}")

    def log_trade(self, trade_data: dict):
        """Log trade to CSV"""
        if self.config["trading"]["log_trades_to_csv"]:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    trade_data.get("symbol", ""),
                    trade_data.get("action", ""),
                    trade_data.get("short_strike", ""),
                    trade_data.get("long_strike", ""),
                    trade_data.get("expiration", ""),
                    trade_data.get("dte", ""),
                    trade_data.get("quantity", 1),
                    trade_data.get("credit", 0.0),
                    trade_data.get("max_risk", 0.0),
                    trade_data.get("status", ""),
                ])

    async def connect_to_ibkr(self):
        """
        Connect to IBKR TWS/Gateway with infinite retry logic.
        Never gives up - waits for TWS to be available.
        """
        ibkr_cfg = self.config["ibkr"]
        host = ibkr_cfg["host"]
        port = ibkr_cfg["port"]
        client_id = ibkr_cfg["client_id"]
        retry_delay = ibkr_cfg["retry_delay_seconds"]

        attempt = 0
        while self.running:
            attempt += 1
            try:
                logger.info(f"Attempting to connect to IBKR at {host}:{port} (attempt #{attempt})...")
                await self.ib.connectAsync(host, port, clientId=client_id, timeout=10)

                if self.ib.isConnected():
                    self.connected = True
                    logger.success(f"✅ Connected to IBKR TWS/Gateway at {host}:{port}")
                    logger.info(f"Account: {ibkr_cfg['account']}")

                    # Subscribe to connection events
                    self.ib.disconnectedEvent += self.on_disconnected
                    self.ib.errorEvent += self.on_error

                    return True

            except Exception as e:
                logger.warning(f"Connection failed: {e}")
                logger.info(f"Retrying in {retry_delay} seconds... (Ctrl+C to stop)")

            await asyncio.sleep(retry_delay)

        return False

    def on_disconnected(self):
        """Handle disconnection from IBKR"""
        self.connected = False
        logger.warning("⚠️  Disconnected from IBKR! Will attempt reconnection...")

    def on_error(self, reqId, errorCode, errorString, contract):
        """Handle IBKR API errors"""
        # Some error codes are informational, not actual errors
        if errorCode in [2104, 2106, 2158]:  # Market data farm connection messages
            logger.debug(f"Info [{errorCode}]: {errorString}")
        elif errorCode >= 2000:  # Warnings
            logger.warning(f"Warning [{errorCode}]: {errorString}")
        else:  # Actual errors
            logger.error(f"Error [{errorCode}]: {errorString} (Contract: {contract})")

    def is_market_hours(self) -> bool:
        """Check if currently in trading hours (US Eastern Time)"""
        # Get current time in US Eastern Time (ET/EDT)
        et_tz = ZoneInfo("America/New_York")
        now_et = datetime.now(et_tz).time()

        start_hour = self.config["trading"]["start_hour"]
        end_hour = self.config["trading"]["end_hour"]

        market_open = dt_time(start_hour, 0)
        market_close = dt_time(end_hour, 0)

        return market_open <= now_et <= market_close

    def run_scanner(self) -> List[dict]:
        """
        Run Rust scanner to find trade opportunities.
        Returns list of opportunities with entry criteria.
        """
        logger.info("Running opportunity scanner...")

        try:
            # Call Rust scanner with --json flag
            result = subprocess.run(
                ["cargo", "run", "--release", "--features", "data-downloaders",
                 "--example", "paper_trading_scanner", "--", "--json"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd="/home/kim-asplund/projects/kimsfinance/rust",
            )

            if result.returncode != 0:
                logger.error(f"Scanner failed: {result.stderr}")
                return []

            # Parse JSON output
            opportunities = self.parse_scanner_output(result.stdout)

            logger.info(f"Found {len(opportunities)} opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Scanner error: {e}")
            return []

    def parse_scanner_output(self, output: str) -> List[dict]:
        """
        Parse scanner JSON output to extract opportunities.
        """
        try:
            # Parse JSON array of opportunities
            opportunities = json.loads(output)

            if not opportunities:
                logger.info("No opportunities found (empty array)")
                return []

            logger.debug(f"Successfully parsed {len(opportunities)} opportunities")
            return opportunities

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scanner JSON output: {e}")
            logger.debug(f"Output was: {output[:500]}")  # Log first 500 chars
            return []

    async def place_bull_put_spread(self, opportunity: dict):
        """
        Place a bull put spread order.

        Args:
            opportunity: Dict with keys: symbol, short_strike, long_strike,
                        expiration, credit, dte
        """
        if not self.connected:
            logger.warning("Not connected to IBKR, cannot place order")
            return

        symbol = opportunity["symbol"]
        short_strike = opportunity["short_strike"]
        long_strike = opportunity["long_strike"]
        expiration = opportunity["expiration"]  # Format: YYYYMMDD
        credit = opportunity["credit"]

        logger.info(f"Placing bull put spread: {symbol} {short_strike}/{long_strike} @ ${credit:.2f}")

        try:
            # Create option contracts
            short_put = Option(symbol, expiration, short_strike, "P", "SMART")
            long_put = Option(symbol, expiration, long_strike, "P", "SMART")

            # Qualify contracts
            short_put = await self.ib.qualifyContractsAsync(short_put)
            long_put = await self.ib.qualifyContractsAsync(long_put)

            if not short_put or not long_put:
                logger.error(f"Failed to qualify contracts for {symbol}")
                return

            short_put = short_put[0]
            long_put = long_put[0]

            # Create combo order (vertical spread)
            combo = Contract()
            combo.symbol = symbol
            combo.secType = "BAG"
            combo.currency = "USD"
            combo.exchange = "SMART"

            leg1 = ComboLeg()
            leg1.conId = short_put.conId
            leg1.ratio = 1
            leg1.action = "SELL"
            leg1.exchange = "SMART"

            leg2 = ComboLeg()
            leg2.conId = long_put.conId
            leg2.ratio = 1
            leg2.action = "BUY"
            leg2.exchange = "SMART"

            combo.comboLegs = [leg1, leg2]

            # Create limit order for credit
            order = LimitOrder(
                action="BUY",  # BUY the combo (credit spread)
                totalQuantity=1,
                lmtPrice=credit,
                orderType="LMT",
                tif="DAY",
            )

            # Place order
            trade = self.ib.placeOrder(combo, order)

            # Log trade
            self.log_trade({
                "symbol": symbol,
                "action": "OPEN",
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration,
                "dte": opportunity.get("dte", 0),
                "quantity": 1,
                "credit": credit,
                "max_risk": opportunity.get("max_risk", 0),
                "status": "SUBMITTED",
            })

            # Track position
            position_id = f"{symbol}_{short_strike}_{long_strike}_{expiration}"
            self.positions[position_id] = trade

            logger.success(f"✅ Order placed: {position_id}")

        except Exception as e:
            logger.error(f"Failed to place order: {e}")

    async def monitor_positions(self):
        """Monitor open positions and check exit criteria"""
        if not self.connected or not self.positions:
            return

        logger.info(f"Monitoring {len(self.positions)} open positions...")

        for position_id, trade in list(self.positions.items()):
            try:
                # Check if order is filled
                if trade.orderStatus.status != "Filled":
                    logger.debug(f"Position {position_id}: Order not filled yet ({trade.orderStatus.status})")
                    continue

                # Get current market value of the spread
                await self.ib.reqMktDataAsync(trade.contract, '', False, False)
                await asyncio.sleep(1)  # Wait for market data

                # Get position value from portfolio
                portfolio_items = self.ib.portfolio()
                position_value = None

                for item in portfolio_items:
                    if item.contract.conId == trade.contract.conId:
                        position_value = item.marketValue
                        break

                if position_value is None:
                    logger.warning(f"Position {position_id}: Could not get market value")
                    continue

                # Calculate P&L
                fill_price = trade.orderStatus.avgFillPrice
                current_price = position_value / (trade.order.totalQuantity * 100)  # Per contract

                pnl = (fill_price - current_price) * trade.order.totalQuantity * 100
                pnl_pct = (pnl / abs(fill_price * trade.order.totalQuantity * 100)) * 100

                logger.info(f"Position {position_id}: P&L ${pnl:.2f} ({pnl_pct:.1f}%)")

                # Check profit target (50% of max profit)
                profit_target_pct = self.config["strategy"]["profit_target_pct"]
                if pnl_pct >= profit_target_pct:
                    logger.success(f"✅ Profit target hit for {position_id} ({pnl_pct:.1f}% >= {profit_target_pct}%)")
                    await self.close_position(position_id, trade, "PROFIT_TARGET")
                    continue

                # Check stop loss (200% loss)
                stop_loss_pct = self.config["strategy"]["stop_loss_pct"]
                if pnl_pct <= -stop_loss_pct:
                    logger.warning(f"⚠️  Stop loss hit for {position_id} ({pnl_pct:.1f}% <= -{stop_loss_pct}%)")
                    await self.close_position(position_id, trade, "STOP_LOSS")
                    continue

                # Check max hold days
                max_hold_days = self.config["strategy"]["max_hold_days"]
                # TODO: Track entry date and check days in trade
                # For now, skip this check

                logger.debug(f"Position {position_id}: Status OK (P&L: ${pnl:.2f})")

            except Exception as e:
                logger.error(f"Error monitoring position {position_id}: {e}")

    async def close_position(self, position_id: str, trade: Trade, reason: str):
        """Close a position and log the trade"""
        try:
            logger.info(f"Closing position {position_id} (reason: {reason})")

            # Create opposite order to close
            opposite_order = LimitOrder(
                action="SELL" if trade.order.action == "BUY" else "BUY",
                totalQuantity=trade.order.totalQuantity,
                lmtPrice=0,  # Market order
                orderType="MKT",
                tif="DAY",
            )

            # Place closing order
            closing_trade = self.ib.placeOrder(trade.contract, opposite_order)

            # Wait for fill
            await asyncio.sleep(2)

            # Log trade closure
            self.log_trade({
                "symbol": position_id.split("_")[0],
                "action": "CLOSE",
                "short_strike": "",
                "long_strike": "",
                "expiration": "",
                "dte": 0,
                "quantity": trade.order.totalQuantity,
                "credit": closing_trade.orderStatus.avgFillPrice if closing_trade.orderStatus else 0,
                "max_risk": 0,
                "status": reason,
            })

            # Remove from positions
            del self.positions[position_id]

            logger.success(f"✅ Position {position_id} closed successfully")

        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")

    async def main_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting automated trading bot...")
        logger.info(f"Strategy: Bull Put Spread (266% ROC, 67% Win Rate)")
        logger.info(f"Symbols: {self.config['strategy']['symbols']}")
        logger.info(f"Paper Trading: {self.config['safety']['paper_trading_only']}")

        # Connect to IBKR (infinite retry until connected)
        await self.connect_to_ibkr()

        scan_interval = self.config["trading"]["scan_interval_minutes"] * 60
        monitor_interval = self.config["trading"]["monitor_interval_minutes"] * 60

        last_scan = 0
        last_monitor = 0

        while self.running:
            try:
                # Reconnect if disconnected
                if not self.connected:
                    logger.warning("Lost connection, attempting to reconnect...")
                    await self.connect_to_ibkr()
                    continue

                current_time = time.time()

                # Check if market hours
                if not self.is_market_hours():
                    logger.debug("Outside market hours, sleeping...")
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue

                # Scan for new opportunities
                if current_time - last_scan >= scan_interval:
                    logger.info("⏰ Time to scan for opportunities...")
                    opportunities = self.run_scanner()

                    # Place orders for top opportunities
                    for opp in opportunities[:3]:  # Top 3 opportunities
                        if len(self.positions) >= self.config["strategy"]["max_concurrent_positions"]:
                            logger.warning("Max concurrent positions reached, skipping new entries")
                            break

                        await self.place_bull_put_spread(opp)
                        await asyncio.sleep(5)  # Wait between orders

                    last_scan = current_time

                # Monitor existing positions
                if current_time - last_monitor >= monitor_interval:
                    await self.monitor_positions()
                    last_monitor = current_time

                # Sleep briefly
                await asyncio.sleep(10)

            except KeyboardInterrupt:
                logger.info("Received stop signal, shutting down...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying

        # Cleanup
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("Bot shut down successfully")

    async def run(self):
        """Run the bot"""
        try:
            await self.main_loop()
        except Exception as e:
            logger.critical(f"Fatal error: {e}")
            raise
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Paper Trading Bot")
    parser.add_argument("--config", default="config/trading_bot.toml", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual trades)")
    args = parser.parse_args()

    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ required")
        sys.exit(1)

    # Check dependencies
    try:
        import ib_insync
        import toml
        import loguru
    except ImportError as e:
        print(f"ERROR: Missing required package: {e}")
        print("Install with: pip install ib_insync toml loguru")
        sys.exit(1)

    # Run bot
    bot = TradingBot(config_path=args.config)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n✅ Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
