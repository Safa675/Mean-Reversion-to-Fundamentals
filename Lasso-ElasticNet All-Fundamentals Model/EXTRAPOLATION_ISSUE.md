#

 Extrapolation Issue - Same Problem as Before

## The Problem

Both your models (Factor-Based and All-Fundamentals) suffer from **the exact same extrapolation issue**:

### Factor-Based Model (Lasso-Factor 0.1)
- Backtest IC (2023-2025): **-0.21** ⭐⭐⭐ (Excellent!)
- Real-life (2026): **+0.10%** vs BIST30 +0.12% ❌ (Underperformed)
- Predictions: 9 out of 20 stocks had **-100% "undervaluation"** (meaningless)

### All-Fundamentals Model (This Model)
- Test IC (2023-2025): **-0.14** ⭐⭐ (Good!)
- Latest quarter (2025-09-30): **ALL stocks have -100% undervaluation** (meaningless)
- Predictions: Model predicts market caps **trillions of times higher** than actual

## Root Cause

Both models train on **2016-2019** data and try to predict **2025-2026** values.

The problem: **Fundamentals in 2025 are FAR outside the 2016-2019 range**.

### Example: FROTO (Ford Turkey)
```
Training Range (2016-2019):
  LogMarketCap: 14.43 to 25.56
  Mean: 18.28
  Std: 2.32

2025 Prediction:
  Actual LogMarketCap: 25.66 (at edge of training range)
  Predicted LogMarketCap: 48.93 (20+ standard deviations away!)

This is EXTREME extrapolation - the model has never seen
fundamentals like 2025's during training.
```

## Why This Happens

### 1. Market Cap Growth
Turkish companies grew significantly 2016 → 2025:
- Inflation
- Market expansion
- Currency effects

Training on 2016-2019 means the model learned relationships from **much smaller companies**.

### 2. Fundamental Shift
Key fundamentals changed:
- Interest rate environment (low 2016-2019 → high 2023-2025)
- Profitability patterns
- Leverage ratios

The model learned relationships that **no longer hold**.

### 3. Linear Extrapolation Fails
OLS (and Lasso/ElasticNet) are **linear models**:
```
Predicted = β₀ + β₁×Revenue + β₂×Debt + ...
```

When Revenue/Debt values in 2025 are 10x larger than in 2016-2019,
the prediction explodes because the model **extrapolates linearly**
beyond the training data.

## Why IC Was Still Good

Your backtest IC of -0.14 to -0.21 was computed on **2023-2025 data**.

But wait - if 2025 causes extrapolation issues, why was test IC good?

**Answer**: The test period (2023-2025) **overlaps with the extrapolation problem zone**.

The IC metric measures **rank correlation**:
- Even if predicted MCap is wrong by 1000x, the **ranking** may still be correct
- "Stock A is more undervalued than Stock B" can be true even if both predictions are wildly wrong in absolute terms

### Analogy
```
Truth:
  Stock A MCap = $100M
  Stock B MCap = $200M
  B is 2x larger than A

Model Predictions:
  Stock A MCap = $1 Trillion
  Stock B MCap = $2 Trillion
  B is still 2x larger than A ← Ranking is correct!

IC = corr(residuals, returns) = 0.9 ← Excellent!

But actual predictions are useless for portfolio construction.
```

## The Uncomfortable Truth

Your "fantastic backtest" results were an **illusion**:

1. ✅ **IC was real** - the model did find a signal
2. ❌ **IC measures relative ranking, not absolute accuracy**
3. ❌ **Absolute predictions are meaningless** (trillions of $ market caps)
4. ❌ **Real-life portfolio fails** because you can't trade "relative undervaluation" - you need actual stocks

This is a **classic quant finance pitfall**: Great IC, useless model.

## Solutions

### Solution 1: Retrain on Recent Data (Recommended)

**Instead of training on 2016-2019, train on 2020-2023**:

```python
# Current (BROKEN)
train_end = "2020-01-01"  # Train on 2016-2019
val_end = "2023-01-01"    # Validate on 2020-2022
# Test on 2023-2025

# Fixed (BETTER)
train_end = "2024-01-01"  # Train on 2020-2023 ← More recent!
val_end = "2025-01-01"    # Validate on 2024
# Test on 2025-2026
```

**Pros**:
- Fundamentals in 2025 are closer to 2020-2023 range
- Less extrapolation
- More relevant regime (high interest rates, etc.)

**Cons**:
- Smaller training set
- IC may be lower (less data to learn from)
- Still doesn't solve regime shift problem

### Solution 2: Add Prediction Constraints

**Clip predictions to reasonable bounds**:

```python
# After prediction
mean_log = train_log_mcap.mean()
std_log = train_log_mcap.std()

# Don't allow predictions beyond 3 standard deviations
lower = mean_log - 3 * std_log
upper = mean_log + 3 * std_log

predicted_log_mcap = predicted_log_mcap.clip(lower, upper)
```

**Pros**:
- Prevents extreme extrapolations
- Simple to implement (already done in `fix_extrapolation.py`)

**Cons**:
- Arbitrary cutoff (why 3 std?)
- Doesn't fix the underlying problem
- Still may produce poor real-world results

### Solution 3: Use Percentile-Based Valuation

**Instead of absolute MCap prediction, predict percentile**:

```python
# Instead of: "Stock A should have MCap = $500M"
# Predict: "Stock A is in the 75th percentile of valuations"

# Then rank stocks by how far they are from expected percentile
undervalued = actual_percentile < predicted_percentile
```

**Pros**:
- No extrapolation issues (percentiles bounded [0, 100])
- More robust across regimes
- Works even with distribution shifts

**Cons**:
- Requires rebuilding the model
- Less intuitive than absolute valuation

### Solution 4: Rolling Retraining (Best Long-Term)

**Retrain the model every quarter with latest data**:

```
Q1 2024: Train on 2020-2023 → Predict Q2 2024
Q2 2024: Train on 2020-Q1 2024 → Predict Q3 2024
Q3 2024: Train on 2020-Q2 2024 → Predict Q4 2024
...
```

**Pros**:
- Always uses recent data
- Adapts to regime changes
- Minimizes extrapolation

**Cons**:
- Computationally expensive
- Need to track model versions
- IC becomes harder to evaluate (walk-forward only)

## Recommendation

### Short-Term Fix
Run `fix_extrapolation.py` to clip predictions:
```bash
python fix_extrapolation.py
python find_undervalued.py --input outputs/all_fundamentals_lasso_results_fixed.csv
```

But recognize this is a **band-aid** - the underlying predictions are still unreliable.

### Medium-Term Fix
Retrain the model using 2020-2023 instead of 2016-2019.

I can help you modify `all_fundamentals_model.py` to add `--train-start` and `--train-end` parameters.

### Long-Term Fix
Implement one of:
1. **Percentile-based valuation** (most robust)
2. **Rolling retraining** (most adaptive)
3. **Ensemble of multiple time windows** (most stable)

## The Bigger Picture

This problem isn't specific to your model - it's **fundamental to cross-sectional equity models**:

1. Markets evolve
2. Fundamentals shift
3. Linear models extrapolate poorly
4. IC ≠ profitability

**Famous examples**:
- Long-Term Capital Management (1998) - Great backtests, blew up in real-world
- Renaissance Technologies - Reruns models daily with latest data
- AQR - Uses ensemble of 100+ signals, constantly rebalanced

The lesson: **A model is only as good as its most recent training data**.

## What to Do Now

1. ✅ **Run the fix script** (temporary solution)
2. ❓ **Decide**: Do you want to:
   - A) Retrain on recent data (2020-2023)?
   - B) Switch to percentile-based approach?
   - C) Accept that mean reversion may not work in 2025-2026 regime?

3. 🔍 **Investigate**: Why did the signal work in 2023-2025 backtest but not in 2026 real-life?
   - Regime shift (high rates → lower rates)?
   - Model overfitting to specific period?
   - Just bad luck (11 days is too short to judge)?

Let me know which direction you'd like to go, and I can help implement it!
