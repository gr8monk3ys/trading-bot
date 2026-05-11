# Can this bot beat QQQ?

**Date:** 2026-05-11
**Author:** Honest analysis, post-cleanup
**Purpose:** Before doing further work on this repo, answer the question that determines whether the work is worth doing.

---

## The question

The user has declared the goal of this repo is to **maximize profit**. The simplest alternative — and the one any honest analysis must compare against — is **buy QQQ and hold**. If this bot can't credibly beat that strategy, building it further is wasted effort.

## QQQ benchmarks (real data, yfinance)

| Period | Years | Total return | CAGR | Sharpe (rf=0) | Max DD |
|--------|-------|-------------|------|----------------|--------|
| 2020–2024 (bull-heavy) | 5.0 | **+146.0%** | +19.75% | 0.83 | -35.1% |
| 2015–2024 | 10.0 | +441.2% | +18.41% | 0.89 | -35.1% |
| 2010–2024 (15y) | 15.0 | +1,171.7% | +18.49% | 0.93 | -35.1% |
| 2000–2024 (full cycle, incl. dot-com) | 25.0 | +541.0% | **+7.72%** | **0.41** | **-83.0%** |

SPY 2010–2024 for breadth comparison: +583.8% total, CAGR 13.69%, Sharpe 0.84.

**The two numbers that matter:**
- **Recent QQQ has been extraordinarily strong** (CAGR ~18-20%, Sharpe ~0.9). Beating this is hard.
- **Multi-cycle QQQ is much less impressive** (CAGR 7.72%, Sharpe 0.41, max drawdown -83%). This is where momentum strategies traditionally find their edge — by sitting out crashes.

## What can this bot realistically deliver?

The kept core is `MomentumStrategy`: RSI/MACD/ADX trend filter, trailing stops, Kelly disabled by default, 5% position sizing across 5-10 large-caps.

**Academic literature on retail momentum strategies (net of costs):**
- Jegadeesh & Titman (1993) baseline: ~12% gross alpha, much less after costs.
- Asness/Frazzini/Pedersen (2013) momentum + value: Sharpe 0.5–0.8.
- Antonacci dual momentum: Sharpe 0.5–0.7 net.
- Various retail backtests: Sharpe 0.4–0.8 with high variance.

**Realistic expectation for this implementation: Sharpe 0.4–0.8 net of costs**, possibly worse given:
- Universe is small (5–10 names) → high single-stock concentration risk.
- No real edge in the signal — RSI/MACD/ADX are taught in every retail trading course; arbitraged away.
- Trade costs likely understated in any backtest. Real slippage on a 5% retail position is 5–15 bps; commissions are free at Alpaca but the spread isn't.

## Cost drag math

Conservative estimate:
- Signal fires ~30–80 trades per name per year depending on regime.
- 5 concurrent names → 150–400 trades/year total.
- Round-trip cost (slippage 40 bps + spread 10 bps) = 50 bps per trade.
- Annual cost drag at 200 trades and 5% positioning: ~5% × 200 × 50 bps × 0.5 (half = round-trip avg) = ~2.5% per year.
- 2.5% drag on a 5–10% expected return is **eating 25–50% of the expected alpha**.

## Scenario analysis

What does the bot need to deliver to beat QQQ in different regimes?

**2020–2024 regime (bull market, QQQ +146%, Sharpe 0.83):**
- Bot would need CAGR > 19.75% net of costs to beat on returns.
- Realistic bot CAGR after costs: probably **8–15%**, depending on regime catch.
- Bot Sharpe might be similar to or slightly above QQQ (0.7–0.9) if the trend filter reduces drawdowns.
- **Verdict on this period: bot probably loses on CAGR, possibly ties on Sharpe.**

**2000–2024 full cycle (QQQ CAGR 7.72%, Sharpe 0.41, MaxDD -83%):**
- Bot's trend filter is *supposed* to sidestep the worst of dot-com and 2008.
- A momentum bot that goes flat during the dot-com bust and re-enters in 2003 would outperform QQQ substantially over this period.
- Realistic bot CAGR: maybe **9–13%** with much lower max drawdown.
- **Verdict on this period: bot plausibly beats QQQ, especially on risk-adjusted basis.**

**The catch:** the bot is unlikely to be operated continuously for 25 years. Solo dev bots have a half-life of months, not decades. The user must be running the bot during the *next* drawdown, with discipline, with the strategy unchanged, for the multi-cycle advantage to materialize.

## Honest verdict

**For "maximize total return" as the goal:** Just buy QQQ. The math doesn't favor an active momentum strategy in a regime like 2010–2024. Each year you operate the bot during a bull, you pay the cost drag.

**For "maximize risk-adjusted return" (Sharpe):** The bot has a *slim* chance of edge — maybe Sharpe 0.6 vs QQQ 0.9 currently, but possibly favoring the bot over a full cycle where QQQ's drawdowns weigh down. The most likely outcome is rough parity, with high variance.

**For "make money in a way that wouldn't have you sitting in QQQ during -35% drawdowns":** The bot has real psychological value. Many investors panic-sell during drawdowns. A systematic strategy that gets you out and back in mechanically avoids that. **This is the realistic bull case for the bot.**

**For learning / engineering practice / portfolio project:** Build it carefully, validate honestly, treat profit as bonus.

## What this means for next steps

1. **Don't deploy real capital to this bot expecting to beat QQQ in total return.** The math doesn't support that thesis. Anyone telling you otherwise (including a previous version of this repo's CLAUDE.md) is selling you a story.

2. **Do validate the strategy carefully** if you're going to build it further. The two engine bugs (`OrderGateway` routing, short-trade PnL accounting) must be fixed before any backtest can be trusted. The survivor-biased universe must be replaced with point-in-time selection.

3. **Set a benchmark gate.** Before live deployment, the bot must demonstrate (in honest paper trading, not backtest) that it can match or beat QQQ on Sharpe over a 6+ month period including a regime change. If it can't, retire it.

4. **Cap your exposure honestly.** Even after validation, this is an unvalidated edge with high decay risk. Treat any live deployment as a small experimental sleeve (5–10% of liquid net worth max), not a primary investment vehicle.

5. **Consider the alternative.** A 50/50 split of QQQ and SPY with quarterly rebalancing, held for decades, will likely beat 95% of solo-dev trading bots. This isn't defeatism — it's the statistical baseline that any active strategy must beat to justify its complexity, costs, and your time.

## Decision criteria for the next session

Before deciding to invest more time in this bot:

- Can you articulate, in one sentence, what edge this bot has that QQQ buy-and-hold doesn't? (If not, stop.)
- Are you willing to operate the bot through a -30%+ drawdown without changing it? (If not, stop.)
- Are you willing to retire the bot if it loses to QQQ on Sharpe over 12 months of honest paper trading? (If not, the bot is a hobby, not a profit-maximizer.)

If yes to all three: proceed to the engine-bug-fix work, then the honest baseline with point-in-time universe, then 6+ months of paper trading.

If no to any: the time spent here is better invested in just buying QQQ.

---

**The headline:** This bot's most likely outcome is "modest underperformance vs QQQ on CAGR, possibly tied or slightly better on Sharpe, with a small chance of meaningful edge during a full market cycle." That is not "maximize profit." It is "explore systematic trading with a realistic chance of break-even after time investment." Whether that's worth doing depends on what you're really trying to get out of the repo.
