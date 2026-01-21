# Reality Check: Why Mean Reversion Models Struggle

## Summary of Attempts

### Attempt 1: Factor-Based Model (6-7 factors)
- Training: 2016-2019
- Test IC: **-0.21** (Excellent!)
- Real-life (2026): **Failed** - all stocks -100% undervalued

### Attempt 2: All-Fundamentals Model (169 features)
- Training: 2016-2019
- Test IC: **-0.14** (Good!)
- Real-life (2025): **Failed** - all stocks -100% undervalued

### Attempt 3: All-Fundamentals + Production Mode (all data)
- Training: 2016-2025
- Test IC: **-0.07** (Weak)
- R²: **0.368** (Low - only explains 37% of variance)
- Real-life (2025): **Still Failed** - predictions 100-150x too high

## The Fundamental Problem

Mean reversion models for absolute valuation (predicting market cap) are **inherently difficult** because:

### 1. Low R² = High Prediction Error

```
R² = 0.368 means:
  - Model explains 37% of variance
  - 63% is unexplained (residual variance)

In practice:
  - Actual Log MCap: 25.66
  - Predicted LogMCap: 30.84
  - Error: 5.18 log points = 150x in actual terms!
```

###2. Market Cap is a **Product** Not a **Sum**

```
Market Cap = Price × Shares Outstanding

Fundamentals drive Price, but:
  - Price = f(sentiment, liquidity, macro, technicals, ...)
  - Only partially explained by fundamentals

Even with perfect fundamental data, you can't perfectly predict market cap.
```

### 3. Regime Shifts

```
2016-2019: Low interest rates, high growth expectations
2023-2025: High interest rates, recession fears
2026: ??? (transition period?)

The relationship between fundamentals and valuation changes!
```

### 4. Missing Variables (Omitted Variable Bias)

Even with 169 fundamentals, we're missing:
- Management quality
- Brand value
- Network effects
- Competitive moat
- Market sentiment
- Liquidity
- Foreign ownership
- Political connections (especially in Turkey!)

## Why IC Was Good But Predictions Are Bad

**IC (Information Coefficient)** measures **relative ranking**:
```
IC = correlation(residuals, future returns)

Stock A: Actual=25, Predicted=30, Residual=-5
Stock B: Actual=26, Predicted=32, Residual=-6

A is less undervalued than B (correct ranking!)
→ IC looks good

But both predictions are 100x too high!
→ Absolute values are useless
```

**This is why quant funds focus on relative value (long/short) not absolute value!**

## What Works vs What Doesn't

### ❌ Doesn't Work: Absolute Valuation

```python
# Predict: "Stock A should be worth $500M"
predicted_mcap = model.predict(fundamentals)

# Problem: Even 100% error means factor-of-2 mistake
# In log space, 5-point error = 150x mistake!
```

### ✅ Works Better: Relative Valuation

```python
# Predict: "Stock A is cheaper than Stock B"
residual_A = actual_A - predicted_A
residual_B = actual_B - predicted_B

if residual_A < residual_B:
    # A is undervalued relative to B
    # Long A, Short B
```

### ✅ Works Best: Factor Portfolios

```python
# Rank stocks by factor (e.g., P/E ratio, P/B ratio)
# Long bottom quintile, short top quintile
# Don't predict absolute values at all!
```

## Why Your Backtest Looked Good

1. **IC was real** - the model found a signal
2. **Ranking was correct** - undervalued stocks did outperform (relatively)
3. **But absolute predictions were always bad** - you just didn't see it in IC

## The Uncomfortable Truth

**You cannot reliably predict absolute market caps from fundamentals alone.**

Even the best quant models have R² around 0.4-0.6, which means:
- Predictions can be off by 2-5x easily
- In log space, that's -100% "undervaluation"

## What to Do Instead

### Option 1: Accept Ranking, Ignore Absolute Values ✅

```python
# Don't use: "THYAO is 99% undervalued"
# Instead use: "THYAO is in bottom 10% of valuations"

# Build portfolio:
scores = df.groupby('QuarterDate')['Residual_LogMarketCap'].rank(pct=True)
long_portfolio = stocks[scores < 0.2]  # Bottom 20%
short_portfolio = stocks[scores > 0.8]  # Top 20%
```

### Option 2: Use Simple Ratios (P/E, P/B) ✅

```python
# Much simpler, often works better:
df['PE'] = df['MarketCap'] / df['NetIncome']
df['PB'] = df['MarketCap'] / df['TotalEquity']

# Long low P/E, short high P/E
# No fancy models needed!
```

### Option 3: Factor Investing (Fama-French style) ✅

```python
# Build factor portfolios:
# - Value: P/B, P/E, P/S
# - Quality: ROE, margins, stability
# - Momentum: Past 12-month returns
# - Size: Market cap

# Long stocks strong on multiple factors
```

### Option 4: Stop Trying to Time the Market ✅

```python
# Just buy and hold BIST100 index
# Hard to beat after costs!
```

## Realistic Expectations

### What Mean Reversion Models CAN Do:
- ✅ Identify relatively cheap vs expensive stocks
- ✅ Generate positive IC (-0.05 to -0.20)
- ✅ Long/short portfolios with modest alpha (2-5% annualized)
- ✅ Work in some regimes, fail in others

### What They CANNOT Do:
- ❌ Predict absolute valuations accurately
- ❌ Tell you "This stock is 99% undervalued"
- ❌ Work consistently across all market conditions
- ❌ Replace fundamental analysis and due diligence
- ❌ Generate 20-40% annual returns with low risk

## My Recommendation

**Stop trying to predict absolute values. Focus on relative rankings.**

### Practical Implementation:

```python
# 1. Compute residuals (actual - predicted)
df['Residual'] = df['Actual_LogMCap'] - df['Predicted_LogMCap']

# 2. Rank stocks within each quarter (percentile)
df['Undervaluation_Percentile'] = df.groupby('QuarterDate')['Residual'].rank(pct=True)

# 3. Build portfolio from bottom 20% (most undervalued)
undervalued = df[df['Undervaluation_Percentile'] < 0.2]

# 4. Add quality filters
undervalued = undervalued[
    (undervalued['NetIncome'] > 0) &  # Profitable
    (undervalued['NetDebt'] / undervalued['TotalEquity'] < 2) &  # Not overleveraged
    (undervalued['MarketCap'] > 1e9)  # Liquid
]

# 5. Equal-weight the portfolio
portfolio = undervalued.groupby('QuarterDate').apply(
    lambda x: x.sample(min(20, len(x)))  # Max 20 stocks
)
```

This approach:
- ✅ Uses the model's **ranking** ability (works!)
- ✅ Ignores **absolute predictions** (broken!)
- ✅ Adds quality filters (reduces risk)
- ✅ Diversifies (20 stocks, not 1-2)
- ✅ Rebalances quarterly (adapts to regime changes)

## Final Thoughts

Your models aren't "bad" - they're just being asked to do something nearly impossible (predict absolute valuations).

The IC of -0.07 to -0.21 is actually decent! It means the model **can** identify relatively cheap stocks.

But "relatively cheap" ≠ "99% undervalued"

Think of it like weather forecasting:
- ✅ "Tomorrow will be warmer than today" (relative, often right)
- ❌ "Tomorrow will be exactly 23.7°C" (absolute, often wrong)

Your model is good at the first, bad at the second. Use it accordingly!

---

**Bottom line**: Use the model for **ranking stocks**, not for **predicting their true values**. Build a diversified long-only or long/short portfolio based on rankings, and you might see the alpha that the IC suggests exists.
