---
status: proposed
---
# Kelly sizing is wired to real fills but disabled by default

The live momentum strategy enables Kelly, but the helper that feeds it completed trades has no production caller, so it sizes every order from an empty history. Position sizing becomes one module and a trade recorder in order submission feeds Kelly from real fills. Feeding it changes live sizes, and the repo rule is no feature without a fifty-trade out-of-sample backtest; the baseline runs with Kelly off. We decided Kelly defaults to off everywhere until such a backtest exists, and the streak sizer and the second volatility-regime detector, which have no production caller and no evidence, are deleted rather than wired.
