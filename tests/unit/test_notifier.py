"""
Unit tests for Notifier.

Tests the notification system including:
- Initialization with environment config
- Trade notifications
- Circuit breaker alerts
- Daily summaries
- Position alerts
- Slack and email delivery
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNotifierInit:
    """Test Notifier initialization."""

    @patch.dict("os.environ", {}, clear=True)
    def test_init_no_config(self):
        """Test initialization with no config (console only)."""
        from utils.notifier import Notifier

        notifier = Notifier()

        assert notifier.slack_enabled is False
        assert notifier.email_enabled is False

    @patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}, clear=True)
    def test_init_with_slack(self):
        """Test initialization with Slack webhook."""
        from utils.notifier import Notifier

        notifier = Notifier()

        assert notifier.slack_enabled is True
        assert notifier.slack_webhook_url == "https://hooks.slack.com/test"
        assert notifier.email_enabled is False

    @patch.dict("os.environ", {
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "password",
        "EMAIL_TO": "recipient@example.com",
    }, clear=True)
    def test_init_with_email(self):
        """Test initialization with email config."""
        from utils.notifier import Notifier

        notifier = Notifier()

        assert notifier.email_enabled is True
        assert notifier.smtp_user == "user@example.com"
        assert notifier.smtp_password == "password"
        assert notifier.email_to == "recipient@example.com"
        assert notifier.slack_enabled is False

    @patch.dict("os.environ", {
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "password",
        "EMAIL_TO": "recipient@example.com",
    }, clear=True)
    def test_init_with_both(self):
        """Test initialization with both Slack and email."""
        from utils.notifier import Notifier

        notifier = Notifier()

        assert notifier.slack_enabled is True
        assert notifier.email_enabled is True

    @patch.dict("os.environ", {
        "SMTP_SERVER": "smtp.custom.com",
        "SMTP_PORT": "465",
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "password",
        "EMAIL_TO": "recipient@example.com",
    }, clear=True)
    def test_init_custom_smtp(self):
        """Test initialization with custom SMTP settings."""
        from utils.notifier import Notifier

        notifier = Notifier()

        assert notifier.smtp_server == "smtp.custom.com"
        assert notifier.smtp_port == 465


class TestSendTradeNotification:
    """Test trade notification sending."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_trade_notification_buy(self):
        """Test sending buy trade notification."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_trade_notification(
                symbol="AAPL",
                side="buy",
                quantity=10,
                price=150.00
            )

            mock_send.assert_called_once()
            title, message = mock_send.call_args[0]
            assert "BUY" in title
            assert "AAPL" in title
            assert "📈" in message
            assert "10.00 shares" in message
            assert "$150.00" in message

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_trade_notification_sell(self):
        """Test sending sell trade notification."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_trade_notification(
                symbol="MSFT",
                side="sell",
                quantity=5,
                price=300.00
            )

            mock_send.assert_called_once()
            title, message = mock_send.call_args[0]
            assert "SELL" in title
            assert "MSFT" in title
            assert "📉" in message

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_trade_notification_with_profit(self):
        """Test trade notification with positive P/L."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_trade_notification(
                symbol="AAPL",
                side="sell",
                quantity=10,
                price=160.00,
                pnl=100.00
            )

            title, message = mock_send.call_args[0]
            assert "💰" in message
            assert "+100.00" in message

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_trade_notification_with_loss(self):
        """Test trade notification with negative P/L."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_trade_notification(
                symbol="AAPL",
                side="sell",
                quantity=10,
                price=140.00,
                pnl=-100.00
            )

            title, message = mock_send.call_args[0]
            assert "💸" in message
            assert "-100.00" in message


class TestSendCircuitBreakerAlert:
    """Test circuit breaker alert sending."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_circuit_breaker_alert(self):
        """Test sending circuit breaker alert."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_circuit_breaker_alert(
                daily_loss=0.035,
                max_loss=0.03
            )

            mock_send.assert_called_once()
            title, message, kwargs = mock_send.call_args[0][0], mock_send.call_args[0][1], mock_send.call_args[1]
            assert "CIRCUIT BREAKER" in title
            assert "🚨" in message
            assert "3.50%" in message
            assert "3.00%" in message
            assert kwargs.get("urgent") is True


class TestSendDailySummary:
    """Test daily summary sending."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_daily_summary(self):
        """Test sending daily summary."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_daily_summary({
                "total_trades": 10,
                "win_rate": 0.60,
                "total_pnl": 1250.50,
                "sharpe_ratio": 1.85,
                "max_drawdown_pct": 0.08
            })

            mock_send.assert_called_once()
            title, message = mock_send.call_args[0]
            assert "Daily Summary" in title
            assert "Total Trades: 10" in message
            assert "60.0%" in message
            assert "1,250.50" in message
            assert "1.85" in message
            assert "8.00%" in message

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_daily_summary_empty_metrics(self):
        """Test sending daily summary with empty metrics."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_daily_summary({})

            mock_send.assert_called_once()
            title, message = mock_send.call_args[0]
            assert "Total Trades: 0" in message


class TestSendPositionAlert:
    """Test position alert sending."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_position_alert_up(self):
        """Test position alert for significant gain."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_position_alert(symbol="AAPL", pnl_pct=0.10)

            mock_send.assert_called_once()
            title, message = mock_send.call_args[0]
            assert "AAPL" in title
            assert "🚀" in message
            assert "UP" in message
            assert "+10.00%" in message

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_position_alert_down(self):
        """Test position alert for significant loss."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch.object(notifier, '_send', new_callable=AsyncMock) as mock_send:
            await notifier.send_position_alert(symbol="AAPL", pnl_pct=-0.05)

            mock_send.assert_called_once()
            title, message = mock_send.call_args[0]
            assert "⚠️" in message
            assert "DOWN" in message
            assert "-5.00%" in message


class TestSend:
    """Test internal _send method."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_send_console_only(self):
        """Test sending with console only (no Slack/email)."""
        from utils.notifier import Notifier

        notifier = Notifier()
        notifier.slack_enabled = False
        notifier.email_enabled = False

        # Should not raise
        await notifier._send("Test Title", "Test Message")

    @pytest.mark.asyncio
    async def test_send_with_slack(self):
        """Test sending with Slack enabled."""
        from utils.notifier import Notifier

        with patch.dict("os.environ", {}, clear=True):
            notifier = Notifier()
            notifier.slack_enabled = True
            notifier.slack_webhook_url = "https://hooks.slack.com/test"

            with patch.object(notifier, '_send_slack', new_callable=AsyncMock) as mock_slack:
                await notifier._send("Test Title", "Test Message")

                mock_slack.assert_called_once_with("Test Title", "Test Message", False)

    @pytest.mark.asyncio
    async def test_send_with_email(self):
        """Test sending with email enabled."""
        from utils.notifier import Notifier

        with patch.dict("os.environ", {}, clear=True):
            notifier = Notifier()
            notifier.email_enabled = True

            with patch.object(notifier, '_send_email') as mock_email:
                await notifier._send("Test Title", "Test Message")

                mock_email.assert_called_once_with("Test Title", "Test Message", False)

    @pytest.mark.asyncio
    async def test_send_urgent(self):
        """Test sending urgent notification."""
        from utils.notifier import Notifier

        with patch.dict("os.environ", {}, clear=True):
            notifier = Notifier()
            notifier.slack_enabled = True
            notifier.slack_webhook_url = "https://hooks.slack.com/test"

            with patch.object(notifier, '_send_slack', new_callable=AsyncMock) as mock_slack:
                await notifier._send("Test Title", "Test Message", urgent=True)

                mock_slack.assert_called_once_with("Test Title", "Test Message", True)


class TestSendSlack:
    """Test Slack notification sending."""

    @pytest.mark.asyncio
    async def test_send_slack_success(self):
        """Test successful Slack notification."""
        from utils.notifier import Notifier

        with patch.dict("os.environ", {}, clear=True):
            notifier = Notifier()
            notifier.slack_webhook_url = "https://hooks.slack.com/test"

            mock_response = MagicMock()
            mock_response.status = 200

            with patch("utils.notifier.aiohttp.ClientSession") as mock_session_class:
                mock_post_context = AsyncMock()
                mock_post_context.__aenter__.return_value = mock_response
                mock_post_context.__aexit__.return_value = None

                mock_session = MagicMock()
                mock_session.post.return_value = mock_post_context

                mock_session_context = MagicMock()
                mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_context.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value = mock_session_context

                await notifier._send_slack("Test Title", "Test Message")

                mock_session.post.assert_called_once()
                call_args = mock_session.post.call_args
                assert call_args[0][0] == "https://hooks.slack.com/test"
                assert "json" in call_args[1]

    @pytest.mark.asyncio
    async def test_send_slack_urgent(self):
        """Test urgent Slack notification has urgency indicator."""
        from utils.notifier import Notifier

        with patch.dict("os.environ", {}, clear=True):
            notifier = Notifier()
            notifier.slack_webhook_url = "https://hooks.slack.com/test"

            mock_response = MagicMock()
            mock_response.status = 200

            with patch("utils.notifier.aiohttp.ClientSession") as mock_session_class:
                mock_post_context = AsyncMock()
                mock_post_context.__aenter__.return_value = mock_response
                mock_post_context.__aexit__.return_value = None

                mock_session = MagicMock()
                mock_session.post.return_value = mock_post_context

                mock_session_context = MagicMock()
                mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_context.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value = mock_session_context

                await notifier._send_slack("Test Title", "Test Message", urgent=True)

                call_args = mock_session.post.call_args
                json_payload = call_args[1]["json"]
                # Urgent message should have extra block
                assert len(json_payload["blocks"]) == 3

    @pytest.mark.asyncio
    async def test_send_slack_failure(self):
        """Test Slack notification failure handling."""
        from utils.notifier import Notifier

        with patch.dict("os.environ", {}, clear=True):
            notifier = Notifier()
            notifier.slack_webhook_url = "https://hooks.slack.com/test"

            mock_response = MagicMock()
            mock_response.status = 500

            with patch("utils.notifier.aiohttp.ClientSession") as mock_session_class:
                mock_post_context = AsyncMock()
                mock_post_context.__aenter__.return_value = mock_response
                mock_post_context.__aexit__.return_value = None

                mock_session = MagicMock()
                mock_session.post.return_value = mock_post_context

                mock_session_context = MagicMock()
                mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_context.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value = mock_session_context

                # Should not raise
                await notifier._send_slack("Test Title", "Test Message")

    @pytest.mark.asyncio
    async def test_send_slack_exception(self):
        """Test Slack notification exception handling."""
        from utils.notifier import Notifier

        with patch.dict("os.environ", {}, clear=True):
            notifier = Notifier()
            notifier.slack_webhook_url = "https://hooks.slack.com/test"

            with patch("utils.notifier.aiohttp.ClientSession") as mock_session_class:
                mock_session_class.side_effect = Exception("Connection error")

                # Should not raise
                await notifier._send_slack("Test Title", "Test Message")


class TestSendEmail:
    """Test email notification sending."""

    @patch.dict("os.environ", {
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "password",
        "EMAIL_TO": "recipient@example.com",
    }, clear=True)
    def test_send_email_success(self):
        """Test successful email sending."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            notifier._send_email("Test Title", "Test Message")

            mock_smtp.assert_called_once_with("smtp.gmail.com", 587)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("user@example.com", "password")
            mock_server.send_message.assert_called_once()

    @patch.dict("os.environ", {
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "password",
        "EMAIL_TO": "recipient@example.com",
    }, clear=True)
    def test_send_email_urgent(self):
        """Test urgent email has [URGENT] prefix."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            notifier._send_email("Test Title", "Test Message", urgent=True)

            call_args = mock_server.send_message.call_args[0][0]
            assert "[URGENT]" in call_args["Subject"]

    @patch.dict("os.environ", {
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "password",
        "EMAIL_TO": "recipient@example.com",
    }, clear=True)
    def test_send_email_exception(self):
        """Test email exception handling."""
        from utils.notifier import Notifier

        notifier = Notifier()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.__enter__.side_effect = Exception("SMTP error")

            # Should not raise
            notifier._send_email("Test Title", "Test Message")
