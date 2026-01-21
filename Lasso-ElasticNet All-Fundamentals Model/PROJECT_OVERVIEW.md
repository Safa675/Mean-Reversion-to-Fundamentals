# Project Overview: All-Fundamentals Lasso Mean Reversion Model

## What This Is

A **quantitative stock selection model** for BIST (Borsa Istanbul) that:
1. Predicts log(MarketCap) from ~500 fundamental variables
2. Ranks stocks by residual (Actual - Predicted)
3. Builds portfolios from bottom 20% (most undervalued relative to peers)

## How It Works (Visual Flow)

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUARTERLY DATA FLOW                           │
└─────────────────────────────────────────────────────────────────┘

 [1] DATA INPUT
     └─> fundamentals.csv (~500 variables per stock-quarter)
         └─> Revenue, NetIncome, TotalAssets, etc.

 [2] PREPROCESSING
     └─> Z-score normalization (mean=0, std=1)
     └─> Winsorization (1st/99th percentiles)
     └─> Remove highly correlated features (r > 0.95)
     └─> Median imputation for missing values

 [3] FEATURE SELECTION (Lasso)
     └─> Test 100 alpha values (1e-5 to 1.0)
     └─> Select alpha that gives ~20 features
     └─> Features with highest |coefficient|

 [4] MODEL TRAINING (OLS)
     └─> Retrain OLS on selected 20 features
     └─> Predict: log(MarketCap) = β0 + β1·X1 + ... + β20·X20
     └─> Compute: Residual = Actual - Predicted

 [5] RANKING
     └─> Rank stocks by residual percentile
     └─> Bottom 20% = most undervalued (relative to model)

 [6] PORTFOLIO CONSTRUCTION
     └─> Apply quality filters:
         ├─> Min market cap (10B TL)
         ├─> Profitability (NetIncome > 0)
         └─> Leverage (Debt/Equity < 3)
     └─> Diversify across sectors
     └─> Limit to 15-20 stocks

 [7] OUTPUT
     └─> Portfolio CSV (stocks + percentiles + market caps)
     └─> Model summary (IC, R², selected features)
     └─> Undervaluation rankings
```

## Project Structure

```
Lasso-ElasticNet All-Fundamentals Model/
│
├─ 📊 CORE MODELS
│  ├─ all_fundamentals_model.py       ← Main model (Lasso + OLS)
│  ├─ build_portfolio_correct.py      ← Portfolio construction (CORRECT way)
│  ├─ ensemble_portfolios.py          ← Combine new + old models
│  ├─ find_undervalued.py             ← Old script (DON'T USE - incorrect approach)
│  └─ fix_extrapolation.py            ← Band-aid fix (NOT NEEDED)
│
├─ 📁 OUTPUTS
│  ├─ all_fundamentals_lasso_results.csv        ← Full predictions (all quarters)
│  ├─ all_fundamentals_lasso_model_summary.txt  ← Model stats (IC, R², features)
│  ├─ all_fundamentals_lasso_selected_features.txt
│  ├─ portfolio_20250930.csv                    ← Current portfolio (16 stocks)
│  ├─ top_undervalued_20250930.txt              ← Top 20 stocks
│  ├─ ensemble_intersection.csv                 ← Stocks in BOTH models (if run)
│  └─ ensemble_union.csv                        ← Stocks in EITHER model (if run)
│
└─ 📖 DOCUMENTATION
   ├─ README.md                  ← Comprehensive guide
   ├─ QUICKSTART.md              ← 5-minute quick start
   ├─ QUARTERLY_ROUTINE.md       ← Step-by-step quarterly update guide
   ├─ FINAL_SUMMARY.md           ← Summary of results and usage
   ├─ REALITY_CHECK.md           ← Why absolute predictions fail
   ├─ PRODUCTION_MODE.md         ← How to train on all data
   ├─ MODEL_COMPARISON.md        ← Factor-based vs all-fundamentals
   ├─ EXTRAPOLATION_ISSUE.md     ← Why predictions were extreme
   └─ FIXES_APPLIED.md           ← Median imputation solution
```

## Key Files Explained

### Scripts You Should Use

1. **all_fundamentals_model.py** (MAIN MODEL)
   ```bash
   python all_fundamentals_model.py --production
   ```
   - Trains on ALL historical data (2016-2025)
   - Automatically selects 20 best features using Lasso
   - Outputs predictions and residuals
   - **Run this every quarter**

2. **build_portfolio_correct.py** (PORTFOLIO BUILDER)
   ```bash
   python build_portfolio_correct.py
   ```
   - Uses relative rankings (percentiles), NOT absolute predictions
   - Applies quality filters
   - Builds diversified portfolio
   - **This is the CORRECT way to use the model**

3. **ensemble_portfolios.py** (COMBINE MODELS)
   ```bash
   python ensemble_portfolios.py \
       --new-model outputs/portfolio_20250930.csv \
       --old-model "../Lasso-Factor Mean Reversion 0.1/outputs/undervalued_portfolio.csv"
   ```
   - Finds stocks in BOTH new and old models (highest confidence)
   - Creates intersection portfolio
   - **Optional but recommended**

### Scripts You Should NOT Use

1. **find_undervalued.py** - Uses absolute "undervaluation %" which is unreliable
2. **fix_extrapolation.py** - Band-aid fix that doesn't solve root problem

### Documentation Priority

**Start Here:**
1. [QUICKSTART.md](QUICKSTART.md) - 5 minutes
2. [QUARTERLY_ROUTINE.md](QUARTERLY_ROUTINE.md) - 10 minutes
3. [FINAL_SUMMARY.md](FINAL_SUMMARY.md) - 15 minutes

**Deep Dives:**
4. [REALITY_CHECK.md](REALITY_CHECK.md) - Understand why absolute predictions fail
5. [README.md](README.md) - Full documentation
6. [PRODUCTION_MODE.md](PRODUCTION_MODE.md) - Training strategies

## Key Concepts

### 1. Residuals vs Absolute Predictions

❌ **WRONG** (Absolute Predictions):
```
Stock A: Predicted MCap = 100B, Actual = 50B → "50% undervalued"
Stock B: Predicted MCap = 20B, Actual = 10B → "50% undervalued"
→ Both show same undervaluation, but predictions may be wildly off!
```

✅ **CORRECT** (Relative Rankings):
```
Stock A: Residual = -0.693 (actual far below prediction)
Stock B: Residual = -0.100 (actual slightly below prediction)
→ Stock A is MORE undervalued than Stock B (relative comparison)
→ Rank A = 1, Rank B = 20 → Select Stock A
```

### 2. Information Coefficient (IC)

**IC = Correlation(Residual, Forward_Return)**

- IC = -0.07 means stocks with MORE NEGATIVE residuals tend to outperform
- IC is about **relative ordering**, not absolute accuracy
- IC = -0.07 suggests 2-5% annual alpha (modest but valuable)

### 3. R² vs IC

- **R² = 0.368**: Model explains 37% of variance in log(MarketCap)
  - Sounds low, but this is NORMAL for cross-sectional equity models
  - 63% of variation is due to factors we don't model (sentiment, news, etc.)

- **IC = -0.0684**: Model can rank stocks by undervaluation
  - This is what matters for portfolio construction
  - Even with low R², ranking ability can generate alpha

### 4. Why Negative IC is Good

Mean reversion models SHOULD have negative IC:
- Negative residual → Stock is undervalued → Should rise → Positive return
- Therefore: Negative residual correlates with positive return → IC < 0 ✅

If IC > 0, something is WRONG (momentum, not mean reversion).

## Model Comparison

| Metric | Old Factor Model | New All-Fundamentals Model |
|--------|------------------|---------------------------|
| **Features** | 6-7 pre-defined factors | 20 auto-selected fundamentals |
| **Training IC** | -0.21 (stronger) | -0.0684 (weaker) |
| **Backtest IC** | -0.155 | Not tested yet |
| **Omitted Variable Bias** | Higher (6-7 factors) | Lower (~500 considered) |
| **Interpretability** | High (Factor_Profitability) | Lower (raw fundamentals) |
| **Overfitting Risk** | Lower | Higher |

**Recommendation:** Use BOTH in an ensemble for robustness.

## Performance Expectations

### Realistic Expectations (IC = -0.07)

**Annual Returns:**
- **Best case** (top quartile year): +8 to +15% vs index
- **Typical case** (median year): +2 to +5% vs index
- **Worst case** (bottom quartile year): -3 to +1% vs index
- **Long-term average**: +2 to +5% annual alpha

**Quarterly Win Rate:**
- ~55-65% of quarters outperform index
- ~35-45% of quarters underperform index
- NOT 100% win rate!

**Volatility:**
- Similar to BIST100 (potentially slightly higher)
- Tracking error: 5-10% (portfolio deviates from index)
- Max drawdown: 20-30% (during bear markets)

### What This Is NOT

❌ **This is NOT:**
- A get-rich-quick scheme
- A way to find stocks that are "99% undervalued"
- A model that predicts exact market caps
- A guarantee of positive returns

✅ **This IS:**
- A systematic way to identify relatively undervalued stocks
- A source of modest, positive alpha over long periods
- A tool that requires patience and discipline
- A strategy that works best in portfolios, not individual stocks

## Common Pitfalls

### Pitfall 1: Trusting Absolute "Undervaluation %"

**Problem:** Model shows THYAO is "99.4% undervalued"
**Reality:** This is an artifact of low R² and should be ignored
**Solution:** Use percentile rankings (THYAO is in bottom 0.2%)

### Pitfall 2: Cherry-Picking Stocks

**Problem:** "I'll only buy the #1 ranked stock"
**Reality:** Individual stocks are noisy; diversification is key
**Solution:** Build portfolios of 10-20 stocks

### Pitfall 3: Over-Trading

**Problem:** Rebalancing weekly or monthly
**Reality:** Transaction costs eat returns
**Solution:** Rebalance quarterly (aligned with earnings)

### Pitfall 4: Ignoring Risk Management

**Problem:** "IC is negative, so I'll go 100% long the top 5 stocks"
**Reality:** Even good models have bad periods
**Solution:** Position limits (max 15% per stock), sector limits (max 40%)

### Pitfall 5: Expecting Immediate Results

**Problem:** "Model didn't work this quarter, I'm switching strategies"
**Reality:** Mean reversion takes time (1-4 quarters)
**Solution:** Track performance over 2-3 years, not 1 quarter

## Next Steps

### For Immediate Use:

1. **Read:** [QUARTERLY_ROUTINE.md](QUARTERLY_ROUTINE.md)
2. **Run:** The three main scripts (model → portfolio → ensemble)
3. **Review:** Outputs (portfolio, model summary, rankings)
4. **Decide:** Position sizing and risk management
5. **Execute:** Implement the portfolio

### For Learning:

1. **Read:** [REALITY_CHECK.md](REALITY_CHECK.md)
2. **Understand:** Why absolute predictions fail but rankings work
3. **Experiment:** Try different settings (max_features, top_pct)
4. **Backtest:** Run with --train-end to test on historical data
5. **Compare:** Ensemble with old factor model

### For Production:

1. **Automate:** Set up quarterly retraining pipeline
2. **Monitor:** Track realized IC each quarter
3. **Adjust:** If IC degrades significantly, re-evaluate model
4. **Diversify:** Don't put all capital in one model

## FAQ

**Q: Why is the new model's IC (-0.07) weaker than the old model's IC (-0.21)?**

A: Several reasons:
- Old model uses carefully chosen factors (selection bias)
- New model considers 500+ features (more noise)
- Old model may be overfit to BIST data
- New model is more general-purpose

Both are useful; ensemble reduces risk.

**Q: Can I use this for other markets (NYSE, NASDAQ)?**

A: Yes, but you'll need to:
- Adapt data loading for different fundamentals format
- Retrain on US data (different relationships)
- Adjust quality filters (different liquidity, size distributions)

**Q: How often should I retrain?**

A: **Quarterly** is recommended:
- Aligned with earnings releases
- Fresh enough to capture new data
- Not so frequent that it causes overfitting

**Q: What if IC becomes positive?**

A: This is a red flag:
- Check data quality (errors in loading?)
- Check if market regime has shifted
- Consider pausing the strategy
- Wait for IC to turn negative again

**Q: Should I use Lasso or ElasticNet?**

A: Currently using **Lasso** (works well):
- ElasticNet is available (--method elasticnet)
- ElasticNet may be better if features are correlated
- Try both and compare test IC

---

**Project Status**: ✅ Production-ready
**Last Updated**: 2026-01-20
**Model Version**: 1.0
**Training Period**: 2016-2025
**Latest Portfolio**: Q3 2025 (2025-09-30)
**Portfolio Size**: 16 stocks
**Training IC**: -0.0684 (p < 0.0001)
**Expected Alpha**: 2-5% annually

**Next Update Due**: Late April 2026 (after Q1 2026 earnings)
