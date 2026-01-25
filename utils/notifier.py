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
- Discord webhooks
- Telegram bot
- Email (SMTP)
- Console logging

Usage:
    # Set environment variables in .env file:
    # SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL,
    # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    notifier = Notifier()
    await notifier.send_trade_notification(trade)
"""

import asyncio
import logging
import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: float):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period
            period_seconds: Time period in seconds
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls: list[float] = []

    def can_call(self) -> bool:
        """Check if a call is allowed under rate limit."""
        now = time.time()
        # Remove expired calls
        self.calls = [t for t in self.calls if now - t < self.period_seconds]
        return len(self.calls) < self.max_calls

    def record_call(self):
        """Record a call timestamp."""
        self.calls.append(time.time())


class Notifier:
    """Multi-channel notification system."""

    def __init__(self):
        """Initialize notifier with environment config."""
        # Slack
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        # Discord
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

        # Telegram
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # Email
        self.email_enabled = os.getenv("EMAIL_NOTIFICATIONS", "false").lower() == "true"
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.email_to = os.getenv("EMAIL_TO")

        # Check what's enabled
        self.slack_enabled = bool(self.slack_webhook_url)
        self.discord_enabled = bool(self.discord_webhook_url)
        self.telegram_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)
        self.email_enabled = all([self.smtp_user, self.smtp_password, self.email_to])

        # Rate limiters (Discord: 30/min per webhook, Telegram: 30/sec but we use 20/min for safety)
        self.discord_rate_limiter = RateLimiter(max_calls=30, period_seconds=60)
        self.telegram_rate_limiter = RateLimiter(max_calls=20, period_seconds=60)

        # Log enabled channels
        if self.slack_enabled:
            logger.info("Slack notifications enabled")
        if self.discord_enabled:
            logger.info("Discord notifications enabled")
        if self.telegram_enabled:
            logger.info("Telegram notifications enabled")
        if self.email_enabled:
            logger.info("Email notifications enabled")
        if not any([self.slack_enabled, self.discord_enabled, self.telegram_enabled, self.email_enabled]):
            logger.info("Notifications disabled (console only)")

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

        # Build list of async tasks for parallel sending
        tasks = []

        # Send via Slack
        if self.slack_enabled:
            tasks.append(self._send_slack(title, message, urgent))

        # Send via Discord
        if self.discord_enabled:
            tasks.append(self._send_discord(title, message, urgent))

        # Send via Telegram
        if self.telegram_enabled:
            tasks.append(self._send_telegram(title, message, urgent))

        # Execute all async notifications in parallel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Send via Email (synchronous, run last)
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

    async def _send_discord(self, title: str, message: str, urgent: bool = False):
        """
        Send notification to Discord via webhook.

        Args:
            title: Notification title
            message: Notification message
            urgent: Whether this is an urgent notification
        """
        if not self.discord_rate_limiter.can_call():
            logger.warning("Discord rate limit reached, skipping notification")
            return

        try:
            # Determine embed color based on urgency and content
            if urgent:
                color = 0xE74C3C  # Red
            elif "buy" in message.lower():
                color = 0x2ECC71  # Green
            elif "sell" in message.lower():
                color = 0xE67E22  # Orange
            else:
                color = 0x3498DB  # Blue

            # Build Discord embed for rich formatting
            embed = {
                "title": title,
                "description": message.replace("**", ""),  # Discord handles bold differently
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Trading Bot Notification"}
            }

            # Add urgency indicator
            if urgent:
                embed["title"] = f"URGENT: {title}"

            payload = {
                "embeds": [embed]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.discord_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 204:
                        logger.debug("Discord notification sent successfully")
                        self.discord_rate_limiter.record_call()
                    elif response.status == 429:
                        # Rate limited by Discord
                        retry_after = response.headers.get("Retry-After", "60")
                        logger.warning(f"Discord rate limited, retry after {retry_after}s")
                    else:
                        text = await response.text()
                        logger.error(f"Discord notification failed: {response.status} - {text}")

        except asyncio.TimeoutError:
            logger.error("Discord notification timed out")
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")

    async def _send_telegram(self, title: str, message: str, urgent: bool = False):
        """
        Send notification to Telegram via Bot API.

        Args:
            title: Notification title
            message: Notification message
            urgent: Whether this is an urgent notification
        """
        if not self.telegram_rate_limiter.can_call():
            logger.warning("Telegram rate limit reached, skipping notification")
            return

        try:
            # Format message for Telegram with HTML
            formatted_message = f"<b>{title}</b>\n\n"
            if urgent:
                formatted_message = f"<b>URGENT</b>\n\n{formatted_message}"

            # Convert markdown-style bold to HTML
            formatted_message += message.replace("**", "").replace("*", "")
            # Escape HTML special characters in content (except our tags)
            # but preserve newlines

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": formatted_message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()
                    if result.get("ok"):
                        logger.debug("Telegram notification sent successfully")
                        self.telegram_rate_limiter.record_call()
                    elif result.get("error_code") == 429:
                        # Rate limited by Telegram
                        retry_after = result.get("parameters", {}).get("retry_after", 60)
                        logger.warning(f"Telegram rate limited, retry after {retry_after}s")
                    else:
                        logger.error(f"Telegram notification failed: {result}")

        except asyncio.TimeoutError:
            logger.error("Telegram notification timed out")
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")

    async def send_discord(self, message: str, embed: Optional[dict] = None) -> bool:
        """
        Send a custom message to Discord via webhook.

        Args:
            message: Text message to send
            embed: Optional Discord embed dict for rich formatting

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.discord_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        if not self.discord_rate_limiter.can_call():
            logger.warning("Discord rate limit reached")
            return False

        try:
            payload = {"content": message}
            if embed:
                payload["embeds"] = [embed]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.discord_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 204:
                        self.discord_rate_limiter.record_call()
                        return True
                    else:
                        logger.error(f"Discord error: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False

    async def send_telegram(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a custom message to Telegram.

        Args:
            message: Text message to send
            parse_mode: "HTML" or "Markdown"

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return False

        if not self.telegram_rate_limiter.can_call():
            logger.warning("Telegram rate limit reached")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()
                    if result.get("ok"):
                        self.telegram_rate_limiter.record_call()
                        return True
                    else:
                        logger.error(f"Telegram error: {result}")
                        return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def notify_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        strategy: str
    ):
        """
        Send trade notification to all configured channels with rich formatting.

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            qty: Number of shares
            price: Execution price
            strategy: Strategy name that generated the trade
        """
        action = side.upper()
        total_value = qty * price

        # Build base message
        message = f"**{action}** {qty} {symbol} @ ${price:.2f}\n"
        message += f"Total: ${total_value:,.2f}\n"
        message += f"Strategy: {strategy}\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Build list of async tasks for parallel sending
        tasks = []

        # Discord with embed
        if self.discord_enabled:
            color = 0x2ECC71 if side == "buy" else 0xE74C3C
            embed = {
                "title": f"{action} {symbol}",
                "color": color,
                "fields": [
                    {"name": "Quantity", "value": str(qty), "inline": True},
                    {"name": "Price", "value": f"${price:.2f}", "inline": True},
                    {"name": "Total", "value": f"${total_value:,.2f}", "inline": True},
                    {"name": "Strategy", "value": strategy, "inline": True}
                ],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Trading Bot"}
            }
            tasks.append(self.send_discord(f"Trade: {action} {symbol}", embed))

        # Telegram with HTML
        if self.telegram_enabled:
            emoji = "BUY" if side == "buy" else "SELL"
            tg_message = f"<b>{emoji} {symbol}</b>\n\n"
            tg_message += f"Quantity: {qty}\n"
            tg_message += f"Price: ${price:.2f}\n"
            tg_message += f"Total: ${total_value:,.2f}\n"
            tg_message += f"Strategy: {strategy}\n"
            tg_message += f"Time: {datetime.now().strftime('%H:%M:%S')}"
            tasks.append(self.send_telegram(tg_message))

        # Slack
        if self.slack_enabled:
            tasks.append(self._send_slack(f"Trade: {action} {symbol}", message, urgent=False))

        # Execute all in parallel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Email (synchronous)
        if self.email_enabled:
            self._send_email(f"Trade: {action} {symbol}", message, urgent=False)

    async def notify_alert(self, title: str, message: str, level: str = "info"):
        """
        Send alert notification to all configured channels.

        Args:
            title: Alert title
            message: Alert message
            level: Severity level ('info', 'warning', 'error')
        """
        level_colors = {
            "info": 0x3498DB,     # Blue
            "warning": 0xF39C12,   # Orange
            "error": 0xE74C3C      # Red
        }
        color = level_colors.get(level, 0x95A5A6)
        urgent = level == "error"

        tasks = []

        # Discord
        if self.discord_enabled:
            embed = {
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }
            tasks.append(self.send_discord(f"Alert: {title}", embed))

        # Telegram
        if self.telegram_enabled:
            level_emoji = {"info": "INFO", "warning": "WARNING", "error": "ERROR"}.get(level, "ALERT")
            tg_message = f"<b>{level_emoji}: {title}</b>\n\n{message}"
            tasks.append(self.send_telegram(tg_message))

        # Slack
        if self.slack_enabled:
            tasks.append(self._send_slack(title, message, urgent=urgent))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self.email_enabled:
            self._send_email(title, message, urgent=urgent)

    async def notify_daily_summary_formatted(
        self,
        pnl: float,
        pnl_pct: float,
        trades: int,
        win_rate: float
    ):
        """
        Send formatted daily performance summary to all channels.

        Args:
            pnl: Daily P&L in dollars
            pnl_pct: Daily P&L percentage
            trades: Number of trades
            win_rate: Win rate percentage (0-100)
        """
        direction = "UP" if pnl >= 0 else "DOWN"

        tasks = []

        # Discord embed
        if self.discord_enabled:
            color = 0x2ECC71 if pnl >= 0 else 0xE74C3C
            embed = {
                "title": f"Daily Summary - {direction}",
                "color": color,
                "fields": [
                    {"name": "P&L", "value": f"${pnl:,.2f}", "inline": True},
                    {"name": "Return", "value": f"{pnl_pct:+.2f}%", "inline": True},
                    {"name": "Trades", "value": str(trades), "inline": True},
                    {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True}
                ],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": f"Date: {datetime.now().strftime('%Y-%m-%d')}"}
            }
            tasks.append(self.send_discord("Daily Summary", embed))

        # Telegram
        if self.telegram_enabled:
            tg_message = f"<b>Daily Summary</b>\n\n"
            tg_message += f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)\n"
            tg_message += f"Trades: {trades}\n"
            tg_message += f"Win Rate: {win_rate:.1f}%\n"
            tg_message += f"Date: {datetime.now().strftime('%Y-%m-%d')}"
            tasks.append(self.send_telegram(tg_message))

        # Slack
        if self.slack_enabled:
            message = f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)\n"
            message += f"Trades: {trades}\n"
            message += f"Win Rate: {win_rate:.1f}%"
            tasks.append(self._send_slack(f"Daily Summary - {direction}", message, urgent=False))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self.email_enabled:
            message = f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)\n"
            message += f"Trades: {trades}\n"
            message += f"Win Rate: {win_rate:.1f}%"
            self._send_email(f"Daily Summary - {direction}", message, urgent=False)

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

    print("\nTesting notification system...")
    print(f"  Slack enabled: {notifier.slack_enabled}")
    print(f"  Discord enabled: {notifier.discord_enabled}")
    print(f"  Telegram enabled: {notifier.telegram_enabled}")
    print(f"  Email enabled: {notifier.email_enabled}")

    # Test trade notification (legacy)
    await notifier.send_trade_notification(symbol="AAPL", side="buy", quantity=10, price=150.50)

    # Test trade with P/L (legacy)
    await notifier.send_trade_notification(
        symbol="MSFT", side="sell", quantity=5, price=300.00, pnl=150.50
    )

    # Test circuit breaker alert
    await notifier.send_circuit_breaker_alert(daily_loss=0.035, max_loss=0.03)

    # Test daily summary (legacy)
    await notifier.send_daily_summary(
        {
            "total_trades": 10,
            "win_rate": 0.60,
            "total_pnl": 1250.50,
            "sharpe_ratio": 1.85,
            "max_drawdown_pct": 0.08,
        }
    )

    # Test new formatted trade notification
    await notifier.notify_trade(
        symbol="GOOGL",
        side="buy",
        qty=5,
        price=175.25,
        strategy="MomentumStrategy"
    )

    # Test new alert notification
    await notifier.notify_alert(
        title="High Volatility Detected",
        message="VIX is above 25, reducing position sizes",
        level="warning"
    )

    # Test new formatted daily summary
    await notifier.notify_daily_summary_formatted(
        pnl=1250.50,
        pnl_pct=2.5,
        trades=10,
        win_rate=60.0
    )

    # Test direct Discord/Telegram methods
    if notifier.discord_enabled:
        success = await notifier.send_discord(
            "Direct Discord test message",
            {"title": "Test Embed", "description": "Testing direct Discord API", "color": 0x00FF00}
        )
        print(f"  Discord direct send: {'success' if success else 'failed'}")

    if notifier.telegram_enabled:
        success = await notifier.send_telegram("<b>Direct Telegram test</b>\nTesting direct Telegram API")
        print(f"  Telegram direct send: {'success' if success else 'failed'}")

    print("Notification tests complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_notifier())
