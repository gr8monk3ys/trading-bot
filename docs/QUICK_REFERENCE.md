# Quick Reference Card - Advanced Orders

## Order Type Cheat Sheet

```python
from brokers.order_builder import OrderBuilder
from brokers.alpaca_broker import AlpacaBroker

broker = AlpacaBroker(paper=True)
```

| Order Type | Code Example |
|------------|--------------|
| **Market** | `OrderBuilder('AAPL', 'buy', 100).market().day().build()` |
| **Limit** | `OrderBuilder('AAPL', 'buy', 100).limit(170.00).gtc().build()` |
| **Stop** | `OrderBuilder('AAPL', 'sell', 100).stop(165.00).gtc().build()` |
| **Stop-Limit** | `OrderBuilder('AAPL', 'sell', 100).stop_limit(165.00, 164.00).gtc().build()` |
| **Trailing Stop %** | `OrderBuilder('AAPL', 'sell', 100).trailing_stop(trail_percent=5.0).gtc().build()` |
| **Trailing Stop $** | `OrderBuilder('AAPL', 'sell', 100).trailing_stop(trail_price=10.00).gtc().build()` |

## Advanced Order Classes

| Order Class | Code Example |
|-------------|--------------|
| **Bracket** | `OrderBuilder('AAPL', 'buy', 100).market().bracket(take_profit=180, stop_loss=160).gtc().build()` |
| **OCO** | `OrderBuilder('AAPL', 'sell', 100).limit(170).oco(take_profit=180, stop_loss=160).gtc().build()` |
| **OTO** | `OrderBuilder('AAPL', 'buy', 100).market().oto(take_profit=180).gtc().build()` |

## Time-In-Force Options

| TIF | Method | Description |
|-----|--------|-------------|
| **DAY** | `.day()` | Valid only during trading day |
| **GTC** | `.gtc()` | Good-till-canceled (90 days) |
| **IOC** | `.ioc()` | Immediate-or-cancel |
| **FOK** | `.fok()` | Fill-or-kill (all or nothing) |
| **OPG** | `.opg()` | Market/Limit on Open |
| **CLS** | `.cls()` | Market/Limit on Close |

## Broker Methods

```python
# Submit orders
result = await broker.submit_order_advanced(order)

# Order management
await broker.cancel_order(order_id)
await broker.cancel_all_orders()
await broker.replace_order(order_id, qty=150, limit_price=175.00)

# Retrieve orders
orders = await broker.get_orders(status=QueryOrderStatus.OPEN)
order = await broker.get_order_by_id(order_id)
order = await broker.get_order_by_client_id(client_order_id)
```

## Common Patterns

### Pattern: Enter with Auto Risk Management
```python
order = (OrderBuilder('AAPL', 'buy', 100)
         .market()
         .bracket(take_profit=180.00, stop_loss=160.00, stop_limit=159.00)
         .gtc()
         .build())
```

### Pattern: Protect Profits with Trailing Stop
```python
order = (OrderBuilder('AAPL', 'sell', 100)
         .trailing_stop(trail_percent=5.0)
         .gtc()
         .build())
```

### Pattern: Breakout Entry
```python
current_price = await broker.get_last_price('AAPL')
breakout_level = 175.00

order = (OrderBuilder('AAPL', 'buy', 100)
         .stop(breakout_level)
         .bracket(
             take_profit=breakout_level * 1.08,  # 8% profit
             stop_loss=current_price * 0.97       # 3% loss
         )
         .gtc()
         .build())
```

### Pattern: Extended Hours Limit
```python
order = (OrderBuilder('AAPL', 'buy', 100)
         .limit(170.00)
         .extended_hours(True)
         .day()
         .build())
```

## Price Precision Rules

- **≥ $1.00**: Max 2 decimals (`$150.25` ✅, `$150.255` ❌)
- **< $1.00**: Max 4 decimals (`$0.9875` ✅, `$0.98755` ❌)

## Order Class Restrictions

| Order Class | Entry Type | TIF Allowed | Extended Hours |
|-------------|------------|-------------|----------------|
| **Bracket** | Any | DAY, GTC | ❌ |
| **OCO** | Limit only | Any | ✅ |
| **OTO** | Any | Any | ✅ |

## Convenience Functions

```python
from brokers.order_builder import market_order, limit_order, bracket_order

# Quick functions
order = market_order('AAPL', 'buy', 100, gtc=True)
order = limit_order('AAPL', 'sell', 100, 180.00, gtc=True)
order = bracket_order('AAPL', 'buy', 100,
                     take_profit=180.00,
                     stop_loss=160.00,
                     stop_limit=159.00)
```

## Testing

```bash
# Test all order types (paper trading)
python examples/test_advanced_orders.py

# Run bracket momentum strategy
python strategies/bracket_momentum_strategy.py

# Connection test
python tests/test_connection.py
```

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| Extended hours requires LIMIT | Use `.limit(price).extended_hours().day()` |
| Bracket only supports DAY/GTC | Use `.gtc()` or `.day()`, not IOC/FOK |
| OCO must be LIMIT | Use `.limit(price).oco(...)` |
| Trailing stop only DAY/GTC | Don't use IOC, FOK, OPG, or CLS |

## Safety Checklist

- ✅ Test with paper trading first (`paper=True`)
- ✅ Use bracket orders for all entries
- ✅ Set reasonable stop-losses (2-5%)
- ✅ Validate prices before submission
- ✅ Use client_order_id for tracking
- ✅ Monitor open orders regularly
- ✅ Cancel unnecessary orders before market close

---

**Full Documentation:** `docs/advanced_orders_guide.md`
