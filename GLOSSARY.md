# Trading bot

A paper-only Alpaca trading bot and its backtester. The same vocabulary describes a live session and a historical one; where the two differ today, the glossary names what they should share.

## Market data

**Bar**:
One OHLCV observation for one symbol at one timestamp. The unit both a live session and a backtest advance by.
_Avoid_: candle, tick, row

**Bar history**:
The ordered bars a strategy may look back over for one symbol. Always a sequence of bars, never bare closes.
_Avoid_: price history, price list

**Data outcome**:
What loading bars for one symbol over a window produced: loaded, empty, or failed with a cause. A failed outcome is never reported as empty.

## Deciding

**Strategy**:
A decider. Given the current bar, its bar history, and the portfolio, it returns order intents. It never talks to a broker.
_Avoid_: bot, algo, engine

**Arm**:
One of the strategies an adaptive strategy chooses between. Only one arm is active at a time.
_Avoid_: sub-strategy, child

**Regime**:
The session's one-word reading of market conditions (trend and volatility) for the current bar, with a single position multiplier. There is one regime per bar, never two composed.
_Avoid_: volatility regime, market regime (as separate things)

**Signal**:
A strategy's directional view on a symbol at a bar. Internal to the strategy; what leaves the strategy is an order intent.

**Order intent**:
What a strategy asks for: symbol, direction, target notional or quantity, and protective levels. It is not an order until submitted.
_Avoid_: order request, signal (for the outgoing thing)

**Protective levels**:
The stop-loss and take-profit prices attached to an order intent. They travel with the intent into every broker, including the backtest one.
_Avoid_: bracket legs, managed SL/TP

## Executing

**Session**:
One strategy run against a stream of bars from a clock: the websocket for live, the historical window for a backtest. The session updates bar histories, applies exits, collects intents, and submits them.
_Avoid_: run loop, engine, runner

**Order submission**:
The single act of turning an order intent into an order outcome: size, gate, build, dispatch, record. There is exactly one of these; live and backtest differ only in the broker adapter behind it.
_Avoid_: gateway, order path

**Order outcome**:
What submission returned: filled, accepted, partially filled, rejected, or halted, with the reason. Rejected and halted are outcomes, not fills of quantity zero.
_Avoid_: order result, success flag

**Broker**:
The thing that holds positions and accepts orders. Two exist: Alpaca paper and the backtest simulator. Both present the same positions shape and the same async surface.
_Avoid_: broker interface (the deleted abstract class), data broker

**Position**:
A signed quantity of one symbol held at the broker: positive long, negative short. One shape everywhere.
_Avoid_: holding, position dict

**Position sizing**:
The one decision of how large an intent should be, from base fraction through Kelly, regime multiplier, correlation haircut, and the hard equity cap.
_Avoid_: risk adjustment, position value calculation

**Halt**:
The circuit breaker's refusal to let new entries through. Exits are never halted.
_Avoid_: trading not allowed, block

## Recording

**Fill**:
A broker's confirmation that some quantity of an order executed at a price.

**Trade**:
One completed round trip in one symbol: entry fill(s) to exit fill(s), with realised P&L.
_Avoid_: order (a trade is made of orders), transaction

**Trade history**:
The store of trades and fills written by order submission and read by dashboards and the Kelly estimator. One writer, many readers.
_Avoid_: database, performance tracker, trading history

**Audit trail**:
The hash-chained record of system and order events for tamper evidence. It is not the trade history and is never queried for performance.
_Avoid_: audit log (as a data source)

## Evaluating

**Baseline**:
The canonical 2020–2024 ETF backtest whose numbers the repo cites. Any refactor must reproduce it before its results can be trusted.
_Avoid_: the backtest, the results

**Verdict**:
Whether a backtest's numbers are quotable: at least fifty trades, every symbol's data outcome loaded, no suspicious statistics. Computed in one place, rendered by every script.
_Avoid_: significance gate, honesty check

**Exposure**:
Gross position value as a fraction of equity. The baseline is reported at matched exposure against buy-and-hold.
