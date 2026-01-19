#!/usr/bin/env python3
"""
Trade Notification System

Sends notifications for:
- Trade executions
- Risk alerts (circuit breaker)
- Daily summaries
- Performance milestones

Supports:
- Slack webhooks
- Email (SMTP)
- Console logging

Usage:
    # Set SLACK_WEBHOOK_URL in .env file
    notifier = Notifier()
    await notifier.send_trade_notification(trade)
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class Notifier:
    """Multi-channel notification system."""

    def __init__(self):
        """Initialize notifier with environment config."""
        # Slack
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        # Email
        self.email_enabled = os.getenv("EMAIL_NOTIFICATIONS", "false").lower() == "true"
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.email_to = os.getenv("EMAIL_TO")

        # Check what's enabled
        self.slack_enabled = bool(self.slack_webhook_url)
        self.email_enabled = all([self.smtp_user, self.smtp_password, self.email_to])

        if self.slack_enabled:
            logger.info("✅ Slack notifications enabled")
        if self.email_enabled:
            logger.info("✅ Email notifications enabled")
        if not self.slack_enabled and not self.email_enabled:
            logger.info("📝 Notifications disabled (console only)")

    async def send_trade_notification(
        self, symbol: str, side: str, quantity: float, price: float, pnl: Optional[float] = None
    ):
        """
        Send notification for trade execution.

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            price: Execution price
            pnl: P/L if closing position
        """
        emoji = "📈" if side == "buy" else "📉"
        action = "BUY" if side == "buy" else "SELL"

        # Build message
        message = f"{emoji} **{action} {symbol}**\n"
        message += f"Quantity: {quantity:.2f} shares @ ${price:.2f}\n"
        message += f"Total Value: ${quantity * price:,.2f}\n"

        if pnl is not None:
            pnl_emoji = "💰" if pnl > 0 else "💸"
            message += f"{pnl_emoji} P/L: ${pnl:+,.2f}\n"

        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Send via all enabled channels
        await self._send(f"Trade Executed: {action} {symbol}", message)

    async def send_circuit_breaker_alert(self, daily_loss: float, max_loss: float):
        """
        Send critical alert for circuit breaker trigger.

        Args:
            daily_loss: Current daily loss percentage
            max_loss: Maximum allowed loss percentage
        """
        message = "🚨 **CIRCUIT BREAKER TRIGGERED** 🚨\n\n"
        message += f"Daily Loss: {daily_loss:.2%}\n"
        message += f"Max Allowed: {max_loss:.2%}\n"
        message += "**TRADING HALTED FOR THE DAY**\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        await self._send("⚠️ CIRCUIT BREAKER ALERT", message, urgent=True)

    async def send_daily_summary(self, metrics: dict):
        """
        Send daily performance summary.

        Args:
            metrics: Dictionary of performance metrics
        """
        message = "📊 **Daily Trading Summary**\n\n"
        message += f"Total Trades: {metrics.get('total_trades', 0)}\n"
        message += f"Win Rate: {metrics.get('win_rate', 0):.1%}\n"
        message += f"Total P/L: ${metrics.get('total_pnl', 0):+,.2f}\n"
        message += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
        message += f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}\n"
        message += f"\nDate: {datetime.now().strftime('%Y-%m-%d')}"

        await self._send("Daily Summary", message)

    async def send_position_alert(self, symbol: str, pnl_pct: float):
        """
        Send alert for significant position move.

        Args:
            symbol: Stock symbol
            pnl_pct: P/L percentage
        """
        emoji = "🚀" if pnl_pct > 0 else "⚠️"
        direction = "UP" if pnl_pct > 0 else "DOWN"

        message = f"{emoji} **Position Alert: {symbol}**\n"
        message += f"Current P/L: {pnl_pct:+.2%}\n"
        message += f"Position is {direction} significantly\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        await self._send(f"Position Alert: {symbol}", message)

    async def _send(self, title: str, message: str, urgent: bool = False):
        """
        Send notification via all enabled channels.

        Args:
            title: Notification title
            message: Notification message
            urgent: Whether this is an urgent notification
        """
        # Always log to console
        logger.info(f"\n{'='*60}\n{title}\n{'-'*60}\n{message}\n{'='*60}")

        # Send via Slack
        if self.slack_enabled:
            await self._send_slack(title, message, urgent)

        # Send via Email
        if self.email_enabled:
            self._send_email(title, message, urgent)

    async def _send_slack(self, title: str, message: str, urgent: bool = False):
        """Send notification to Slack webhook."""
        try:
            # Format for Slack
            slack_message = {
                "text": f"*{title}*",
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": title}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": message}},
                ],
            }

            # Add urgency indicator
            if urgent:
                slack_message["blocks"].insert(
                    0,
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":rotating_light: *URGENT* :rotating_light:",
                        },
                    },
                )

            # Send via webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook_url, json=slack_message) as response:
                    if response.status == 200:
                        logger.debug("Slack notification sent successfully")
                    else:
                        logger.error(f"Slack notification failed: {response.status}")

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

    def _send_email(self, title: str, message: str, urgent: bool = False):
        """Send notification via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = self.email_to
            msg["Subject"] = f"{'[URGENT] ' if urgent else ''}{title}"

            # Format message
            body = f"""
Trading Bot Notification
{'='*60}

{message}

{'='*60}
Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            msg.attach(MIMEText(body, "plain"))

            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.debug("Email notification sent successfully")

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")


# Example usage and testing
async def test_notifier():
    """Test notification system."""
    notifier = Notifier()

    print("\n🔔 Testing notification system...")

    # Test trade notification
    await notifier.send_trade_notification(symbol="AAPL", side="buy", quantity=10, price=150.50)

    # Test trade with P/L
    await notifier.send_trade_notification(
        symbol="MSFT", side="sell", quantity=5, price=300.00, pnl=150.50
    )

    # Test circuit breaker alert
    await notifier.send_circuit_breaker_alert(daily_loss=0.035, max_loss=0.03)

    # Test daily summary
    await notifier.send_daily_summary(
        {
            "total_trades": 10,
            "win_rate": 0.60,
            "total_pnl": 1250.50,
            "sharpe_ratio": 1.85,
            "max_drawdown_pct": 0.08,
        }
    )

    print("✅ Notification tests complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_notifier())
