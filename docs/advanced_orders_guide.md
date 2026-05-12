# Advanced Order Types - Complete Guide

This guide covers all advanced order types now available in the trading bot, powered by the Alpaca Trading API.

## Table of Contents
- [Quick Start](#quick-start)
- [Order Types](#order-types)
- [Time-In-Force Options](#time-in-force-options)
- [OrderBuilder API](#orderbuilder-api)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation
All advanced order functionality is already included. No additional dependencies needed.

### Basic Usage
```python
from brokers.alpaca_broker import AlpacaBroker
from brokers.order_builder import OrderBuilder

# Initialize broker (paper trading)
broker = AlpacaBroker(paper=True)

# Create and submit a bracket order
order = (OrderBuilder('AAPL', 'buy', 10)
         .market()
         .bracket(take_profit=180.00, stop_loss=160.00)
         .gtc()
         .build())

result = await broker.submit_order_advanced(order)
print(f"Order submitted: {result.id}")
```

---

## Order Types

### 1. Market Orders
Execute immediately at the best available price.

```python
# Buy at market
order = OrderBuilder('AAPL', 'buy', 100).market().day().build()

# Sell at market with GTC
order = OrderBuilder('TSLA', 'sell', 50).market().gtc().build()
```

**Pros:**
- Guaranteed execution (in liquid markets)
- Simple and fast

**Cons:**
- No price control
- Potential slippage

---

### 2. Limit Orders
Execute only at specified price or better.

```python
# Buy limit order
order = OrderBuilder('NVDA', 'buy', 20).limit(450.00).gtc().build()

# Sell limit order
order = OrderBuilder('MSFT', 'sell', 100).limit(350.00).day().build()

# Extended hours limit order
order = (OrderBuilder('AAPL', 'buy', 10)
         .limit(150.00)
         .extended_hours()
         .day()
         .build())
```

**Pros:**
- Price control
- No slippage beyond limit

**Cons:**
- May not execute
- Requires price monitoring

**Price Rules:**
- Prices ≥ $1.00: Max 2 decimal places (e.g., $150.25)
- Prices < $1.00: Max 4 decimal places (e.g., $0.9875)

---

### 3. Stop Orders
Trigger at stop price, then become market orders.

```python
# Buy stop (breakout trading)
order = OrderBuilder('AAPL', 'buy', 50).stop(175.00).gtc().build()

# Sell stop (stop-loss)
order = OrderBuilder('TSLA', 'sell', 100).stop(200.00).gtc().build()
```

**Pros:**
- Automated trigger
- Good for stop-losses

**Cons:**
- Becomes market order (slippage possible)
- No price guarantee after trigger

**Note:** Alpaca converts buy stop orders to stop-limit orders automatically:
- Stop price < $50: Limit = stop price + 4%
- Stop price ≥ $50: Limit = stop price + 2.5%

---

### 4. Stop-Limit Orders
Trigger at stop price, then execute as limit order.

```python
# Stop-limit order
order = (OrderBuilder('AAPL', 'sell', 100)
         .stop_limit(stop_price=165.00, limit_price=164.00)
         .gtc()
         .build())
```

**Pros:**
- Price protection after trigger
- No slippage beyond limit

**Cons:**
- May not execute if price gaps
- More complex

**Use Case:** Stop-loss with price control (prevent "flash crash" executions)

---

### 5. Trailing Stop Orders
Automatically adjust stop price as market moves favorably.

```python
# Trailing stop by percentage
order = (OrderBuilder('TSLA', 'sell', 50)
         .trailing_stop(trail_percent=5.0)  # Trail by 5%
         .gtc()
         .build())

# Trailing stop by dollar amount
order = (OrderBuilder('AAPL', 'sell', 100)
         .trailing_stop(trail_price=10.00)  # Trail by $10
         .gtc()
         .build())
```

**Pros:**
- Locks in profits automatically
- Adjusts with market
- No manual monitoring needed

**Cons:**
- Only supports DAY or GTC
- Can be triggered by volatility

**How it works:**
- Tracks the "high water mark" (hwm)
- For sells: Stop = hwm - trail amount
- For buys: Stop = hwm + trail amount
- Only trails in favorable direction

---

### 6. Bracket Orders
Entry order + take-profit + stop-loss in one.

```python
# Market entry with bracket
order = (OrderBuilder('NVDA', 'buy', 10)
         .market()
         .bracket(
             take_profit=500.00,
             stop_loss=420.00,
             stop_limit=418.00  # Optional
         )
         .gtc()
         .build())

# Limit entry with bracket
order = (OrderBuilder('AAPL', 'buy', 50)
         .limit(165.00)
         .bracket(
             take_profit=180.00,
             stop_loss=155.00
         )
         .gtc()
         .build())
```

**Pros:**
- Complete risk management in one order
- One exit executes, other cancels automatically
- Perfect for swing trading

**Cons:**
- No extended hours support
- Only DAY or GTC allowed
- All three orders must use same side

**Requirements:**
- Time-in-force: DAY or GTC only
- Take-profit: Always limit order
- Stop-loss: Can be stop or stop-limit
- All orders include DNR (Do Not Reduce)

**Partial Fills:**
If take-profit partially fills, stop-loss adjusts to remaining quantity automatically.

---

### 7. OCO Orders (One-Cancels-Other)
Two exit orders: one executes, other cancels.

```python
# OCO exit for existing position
order = (OrderBuilder('AAPL', 'sell', 100)
         .limit(170.00)  # Required for OCO
         .oco(
             take_profit=180.00,
             stop_loss=160.00,
             stop_limit=159.00  # Optional
         )
         .gtc()
         .build())
```

**Pros:**
- Exit automation for existing positions
- Risk management without monitoring

**Cons:**
- Requires existing position
- Must be limit order type

**Use Case:** You already own stock and want automated exit with profit target and stop-loss.

---

### 8. OTO Orders (One-Triggers-Other)
Entry order with single exit order.

```python
# Entry with take-profit only
order = (OrderBuilder('TSLA', 'buy', 50)
         .market()
         .oto(take_profit=250.00)
         .gtc()
         .build())

# Entry with stop-loss only
order = (OrderBuilder('NVDA', 'buy', 10)
         .limit(450.00)
         .oto(stop_loss=420.00, stop_limit=418.00)
         .gtc()
         .build())
```

**Pros:**
- Simpler than bracket
- Good when only one exit needed

**Cons:**
- Only one exit order
- Limited risk management

**Use Case:** When you want profit target OR stop-loss, but not both.

---

## Time-In-Force Options

### DAY
Valid only during trading day, cancels at market close.

```python
order = OrderBuilder('AAPL', 'buy', 100).market().day().build()
```

**Best for:**
- Day trading
- Extended hours trading
- Short-term positions

---

### GTC (Good-Till-Canceled)
Remains active until filled or canceled (max 90 days).

```python
order = OrderBuilder('TSLA', 'sell', 50).limit(300.00).gtc().build()
```

**Best for:**
- Swing trading
- Long-term orders
- Bracket orders

---

### IOC (Immediate-Or-Cancel)
Fill immediately (full or partial), cancel remainder.

```python
order = OrderBuilder('NVDA', 'buy', 100).limit(450.00).ioc().build()
```

**Best for:**
- Large orders
- Testing liquidity
- Avoiding market impact

---

### FOK (Fill-Or-Kill)
Fill entire order immediately or cancel completely.

```python
order = OrderBuilder('AAPL', 'buy', 1000).limit(170.00).fok().build()
```

**Best for:**
- All-or-nothing execution
- Block trades
- Algorithmic trading

---

### OPG (Market/Limit on Open)
Executes only in opening auction.

```python
order = OrderBuilder('SPY', 'buy', 100).limit(450.00).opg().build()
```

**Best for:**
- Opening trades
- Avoiding intraday volatility

---

### CLS (Market/Limit on Close)
Executes only in closing auction.

```python
order = OrderBuilder('SPY', 'sell', 100).limit(455.00).cls().build()
```

**Best for:**
- Closing positions
- End-of-day rebalancing

---

## OrderBuilder API

### Fluent Interface Pattern

OrderBuilder uses a fluent/chainable interface:

```python
order = (OrderBuilder('SYMBOL', 'side', qty)
         .order_type()      # market(), limit(), stop(), etc.
         .time_in_force()   # day(), gtc(), ioc(), etc.
         .advanced()        # bracket(), oco(), oto()
         .options()         # extended_hours(), client_order_id()
         .build())          # Returns Alpaca order request object
```

### Complete Example

```python
order = (OrderBuilder('AAPL', 'buy', 100)
         .limit(165.00)
         .bracket(
             take_profit=180.00,
             stop_loss=155.00,
             stop_limit=154.00
         )
         .gtc()
         .client_order_id('my-bracket-001')
         .build())
```

---

## Practical Examples

### Example 1: Breakout Strategy with Bracket

```python
async def trade_breakout(symbol, breakout_price, current_price):
    """Enter on breakout with automatic exits."""

    # Calculate levels
    take_profit = breakout_price * 1.08  # 8% profit
    stop_loss = current_price * 0.97     # 3% loss

    order = (OrderBuilder(symbol, 'buy', 100)
             .stop(breakout_price)  # Entry on breakout
             .bracket(
                 take_profit=take_profit,
                 stop_loss=stop_loss
             )
             .gtc()
             .build())

    result = await broker.submit_order_advanced(order)
    return result
```

### Example 2: Trailing Stop for Profit Protection

```python
async def protect_profits(symbol, qty):
    """Lock in profits with trailing stop."""

    order = (OrderBuilder(symbol, 'sell', qty)
             .trailing_stop(trail_percent=3.0)  # Trail by 3%
             .gtc()
             .build())

    result = await broker.submit_order_advanced(order)
    print(f"Trailing stop active: {result.id}")
```

### Example 3: Extended Hours Trading

```python
async def after_hours_limit(symbol, qty, price):
    """Trade in extended hours with limit order."""

    order = (OrderBuilder(symbol, 'buy', qty)
             .limit(price)
             .extended_hours(True)
             .day()  # Required for extended hours
             .build())

    result = await broker.submit_order_advanced(order)
    return result
```

### Example 4: Replace Order (Adjust Price)

```python
async def adjust_limit_order(order_id, new_price):
    """Modify existing limit order price."""

    result = await broker.replace_order(
        order_id=order_id,
        limit_price=new_price
    )

    print(f"Order {order_id} updated to ${new_price:.2f}")
    return result
```

---

## Best Practices

### 1. Always Use Bracket Orders for Entries
```python
# ❌ BAD: Manual stop-loss tracking
order = OrderBuilder('AAPL', 'buy', 100).market().day().build()
# Then manually track stop-loss...

# ✅ GOOD: Automatic risk management
order = (OrderBuilder('AAPL', 'buy', 100)
         .market()
         .bracket(take_profit=180.00, stop_loss=160.00)
         .gtc()
         .build())
```

### 2. Use GTC for Swing Trades
```python
# DAY orders cancel at market close
# GTC orders persist (up to 90 days)

# ✅ For multi-day positions
order = OrderBuilder('TSLA', 'buy', 50).limit(200.00).gtc().build()
```

### 3. Validate Prices Before Submitting
```python
async def validate_and_submit(symbol, side, qty, limit_price):
    """Validate price before order submission."""

    current_price = await broker.get_last_price(symbol)

    # Check if limit price is reasonable
    if side == 'buy' and limit_price > current_price * 1.05:
        print(f"Warning: Buy limit 5% above market")

    order = OrderBuilder(symbol, side, qty).limit(limit_price).gtc().build()
    return await broker.submit_order_advanced(order)
```

### 4. Use Client Order IDs for Tracking
```python
from datetime import datetime

# Generate unique ID
client_id = f"bracket-{symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

order = (OrderBuilder(symbol, 'buy', 100)
         .market()
         .bracket(take_profit=180.00, stop_loss=160.00)
         .client_order_id(client_id)
         .gtc()
         .build())

# Later retrieve by client ID
order = await broker.get_order_by_client_id(client_id)
```

### 5. Cancel Orders Before Market Close (if needed)
```python
async def cancel_day_orders():
    """Cancel all DAY orders before market close."""
    from alpaca.trading.enums import QueryOrderStatus

    open_orders = await broker.get_orders(status=QueryOrderStatus.OPEN)

    for order in open_orders:
        if order.time_in_force == TimeInForce.DAY:
            await broker.cancel_order(order.id)
            print(f"Canceled DAY order: {order.id}")
```

---

## Troubleshooting

### Error: "Extended hours requires LIMIT order type"
```python
# ❌ Wrong: Market order with extended hours
order = OrderBuilder('AAPL', 'buy', 100).market().extended_hours().build()

# ✅ Correct: Limit order with extended hours
order = (OrderBuilder('AAPL', 'buy', 100)
         .limit(170.00)
         .extended_hours()
         .day()
         .build())
```

### Error: "Bracket orders only support DAY or GTC"
```python
# ❌ Wrong: IOC with bracket
order = OrderBuilder('AAPL', 'buy', 100).market().bracket(...).ioc().build()

# ✅ Correct: GTC with bracket
order = OrderBuilder('AAPL', 'buy', 100).market().bracket(...).gtc().build()
```

### Error: "OCO orders must be LIMIT type"
```python
# ❌ Wrong: Market order with OCO
order = OrderBuilder('AAPL', 'sell', 100).market().oco(...).build()

# ✅ Correct: Limit order with OCO
order = (OrderBuilder('AAPL', 'sell', 100)
         .limit(170.00)
         .oco(take_profit=180.00, stop_loss=160.00)
         .gtc()
         .build())
```

### Order Not Executing
1. **Check market hours** - Use `broker.get_account()` to check market status
2. **Check buying power** - Ensure sufficient funds
3. **Check price levels** - Limit orders may be too aggressive
4. **Check order status** - `await broker.get_order_by_id(order_id)`

### Testing Orders Safely
```bash
# Always test with paper trading first
python examples/test_advanced_orders.py
```

---

## Additional Resources

- **Alpaca Documentation**: https://docs.alpaca.markets/docs/trading-api
- **OrderBuilder Source**: `brokers/order_builder.py`
- **Example Strategy**: `strategies/bracket_momentum_strategy.py`
- **Test Script**: `examples/test_advanced_orders.py`

---

## Support

For issues or questions:
1. Check this guide first
2. Review `CLAUDE.md` for architecture details
3. Test with `examples/test_advanced_orders.py`
4. Check broker logs for detailed error messages
