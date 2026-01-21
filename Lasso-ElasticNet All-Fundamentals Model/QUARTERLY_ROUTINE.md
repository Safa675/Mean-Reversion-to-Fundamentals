# Quarterly Portfolio Update Routine

This guide shows you EXACTLY what to run each quarter to update your mean reversion portfolios.

## Timeline

Run this routine **after each quarter-end** when new financial statements are published:
- Q1 (Jan-Mar): Run in late April
- Q2 (Apr-Jun): Run in late July
- Q3 (Jul-Sep): Run in late October
- Q4 (Oct-Dec): Run in late January

## Step-by-Step Process

### Step 1: Update Your Data

First, ensure your fundamentals data is updated with the latest quarter.

```bash
# This depends on your data source
# You might need to:
# - Download new financial statements from KAP/Finnet
# - Update your fundamentals.csv or database
# - Verify the new quarter is included
```

### Step 2: Run New All-Fundamentals Model

```bash
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"

# Retrain model on all historical data (2016-present)
/home/safa/anaconda3/bin/python all_fundamentals_model.py --production

# Build portfolio from latest quarter
/home/safa/anaconda3/bin/python build_portfolio_correct.py

# View portfolio
cat outputs/portfolio_*.csv
```

**Expected output:**
```
FINAL PORTFOLIO: 15-20 STOCKS
Rank   Ticker     Percentile   MCap (B TL)     Sector
1      XXXXX            0.2%        139.62 CAP
2      XXXXX            0.4%        139.62 INT
...

[OK] Portfolio saved to: outputs/portfolio_20250930.csv
```

### Step 3: Run Old Factor-Based Model (Optional)

```bash
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-Factor Mean Reversion 0.1"

# Run old model
/home/safa/anaconda3/bin/python pooled_ols_residuals_bist.py

# View results
cat outputs/undervalued_portfolio.csv
```

### Step 4: Create Ensemble Portfolio (Recommended)

```bash
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"

# Combine both models
/home/safa/anaconda3/bin/python ensemble_portfolios.py \
    --new-model outputs/portfolio_20250930.csv \
    --old-model "../Lasso-Factor Mean Reversion 0.1/outputs/undervalued_portfolio.csv"

# View intersection (stocks in BOTH models)
cat outputs/ensemble_intersection.csv
```

**Expected output:**
```
INTERSECTION (5-10 stocks)
→ Stocks in BOTH portfolios (highest confidence)
→ Recommended for core positions

  THYAO      Rank=1   Percentile=  0.2%  MCap=139.62B  CAP
  ASELS      Rank=2   Percentile=  0.4%  MCap=139.62B  INT
  ...

✅ GOOD CONSENSUS: 8 stocks in both portfolios
   → Allocate 60-70% to intersection (core)
   → Allocate 15-20% to new-only (satellite)
   → Allocate 15-20% to old-only (satellite)
```

### Step 5: Review Results

Check the outputs for any issues:

```bash
# Check model summary
cat outputs/all_fundamentals_lasso_model_summary.txt

# Look for:
# - Training IC (should be negative, e.g., -0.05 to -0.15)
# - R² (typically 0.30 to 0.40)
# - Number of observations (should be > 15000)
# - Selected features (should be ~20)
```

### Step 6: Quality Checks

Before implementing the portfolio, verify:

1. **Portfolio Size**: 10-20 stocks (not too few, not too many)
2. **Sector Diversification**: No single sector > 40%
3. **Market Cap**: Average > 10B TL (avoid micro-caps)
4. **Overlap**: 5-10 stocks in ensemble intersection (good consensus)
5. **Sanity Check**: Do the stocks make fundamental sense?

```bash
# Quick sanity check script
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"

cat outputs/portfolio_*.csv | awk -F',' 'NR>1 {print $1}' | while read ticker; do
    echo "Checking $ticker..."
    # Add your own checks here
    # - Is it liquid enough?
    # - Any recent bad news?
    # - Is the sector in good shape?
done
```

### Step 7: Position Sizing

Decide how to allocate capital:

#### Option A: Equal Weight
```
If 10 stocks in intersection:
  Each stock = 100% / 10 = 10% of portfolio
```

#### Option B: Weighted by Percentile
```
Lower percentile = higher weight
Stock at 0.2% percentile → 2x weight
Stock at 2.0% percentile → 1x weight
```

#### Option C: Weighted by Market Cap (Liquidity)
```
Larger market cap = higher weight
(Up to a max of 15% per stock)
```

#### Recommended: Hybrid Approach
```
Core (60%):
  - Ensemble intersection stocks
  - Equal weight or percentile-weighted
  - 5-10 stocks

Satellite (40%):
  - New-only and old-only stocks
  - Smaller positions (2-5% each)
  - 5-10 stocks

Total: 10-20 stocks
```

### Step 8: Risk Management

Before executing trades:

1. **Position Size Limits**:
   - Max 15% per stock (concentration risk)
   - Max 40% per sector (sector risk)
   - Max 30% in micro-caps (< 10B TL)

2. **Stop Losses** (optional):
   - -20% from entry price
   - -30% from peak price (trailing stop)
   - Review after major drawdowns

3. **Rebalancing Rules**:
   - Quarterly: Replace stocks no longer in portfolio
   - Hold existing positions if still in new portfolio
   - Don't chase momentum (avoid buying after large gains)

4. **Transaction Costs**:
   - Brokerage fees: ~0.1-0.2% per trade
   - Market impact: ~0.5-1% for illiquid stocks
   - Minimize turnover to reduce costs

### Step 9: Track Performance

Create a performance log:

```bash
# Create performance tracking file
cat > performance_log.csv << EOF
Date,Action,Ticker,Price,Shares,Value,Notes
2026-01-20,BUY,THYAO,45.50,1000,45500,Q3 2025 portfolio
2026-01-20,BUY,ASELS,135.20,300,40560,Q3 2025 portfolio
...
EOF
```

Each quarter, update the log:
- Record new positions (BUY)
- Record closed positions (SELL)
- Calculate returns vs BIST100 benchmark
- Update realized IC (correlation between predicted rank and actual return)

### Step 10: Monitor and Adjust

Between quarters, monitor:

1. **Weekly**: Check for major news on portfolio stocks
2. **Monthly**: Review portfolio value and compare to benchmark
3. **Quarterly**: Full rebalancing based on new model run

If realized IC degrades significantly (e.g., becomes positive or close to 0):
- Review model assumptions
- Check for structural breaks in market
- Consider reducing position sizes
- Re-evaluate strategy

## Quick Command Summary

```bash
# Complete quarterly routine (copy-paste this)
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"

# 1. Run new model
/home/safa/anaconda3/bin/python all_fundamentals_model.py --production
/home/safa/anaconda3/bin/python build_portfolio_correct.py

# 2. Run old model (optional)
cd "../Lasso-Factor Mean Reversion 0.1"
/home/safa/anaconda3/bin/python pooled_ols_residuals_bist.py
cd "../Lasso-ElasticNet All-Fundamentals Model"

# 3. Create ensemble
/home/safa/anaconda3/bin/python ensemble_portfolios.py

# 4. Review results
cat outputs/ensemble_intersection.csv
cat outputs/all_fundamentals_lasso_model_summary.txt

# 5. Done! Now implement the portfolio.
```

## Troubleshooting

### Issue: "Training sample too small"
**Solution**: Check if your fundamentals data is loaded correctly. Ensure you have data for 2016-present.

### Issue: "No stocks in intersection"
**Solution**: This means the two models disagree completely. Use them separately or trust the one with stronger historical IC.

### Issue: "Portfolio has only 3 stocks"
**Solution**: Relax quality filters in build_portfolio_correct.py:
- Lower min_mcap to 5B TL (from 10B)
- Remove profitability filter temporarily
- Increase top_pct to 30% (from 20%)

### Issue: "All stocks showing -99% undervaluation"
**Solution**: This is expected! IGNORE absolute undervaluation%. Use percentile rankings instead.

### Issue: "Model IC is positive (wrong sign)"
**Solution**: IC should be negative for mean reversion. If positive, something is wrong:
- Check if residuals are computed correctly (Actual - Predicted)
- Verify forward returns are computed correctly
- Consider abandoning the model for this quarter

## Expected Results

Based on IC = -0.07 to -0.21:

**Best Case** (1 in 4 quarters):
- Portfolio outperforms BIST100 by 5-10%
- Most stocks rise
- IC remains negative and strong

**Typical Case** (2 in 4 quarters):
- Portfolio outperforms BIST100 by 1-3%
- Some winners, some losers
- IC slightly negative

**Bad Case** (1 in 4 quarters):
- Portfolio underperforms BIST100 by 2-5%
- More losers than winners
- IC close to zero or slightly positive

**Long-term** (over multiple years):
- Expected annual alpha: 2-5% above BIST100
- Sharpe ratio: ~0.3-0.5 higher than index
- Win rate: 55-65% (not 100%!)

## Final Checklist

Before executing trades:

- [ ] Model retrained on latest data
- [ ] Portfolio has 10-20 stocks
- [ ] Sector diversification checked (no sector > 40%)
- [ ] Market caps are reasonable (avg > 10B TL)
- [ ] Ensemble intersection has 5-10 stocks
- [ ] Position sizes calculated (max 15% per stock)
- [ ] Stop losses set (optional but recommended)
- [ ] Performance tracking file updated
- [ ] Broker has sufficient liquidity for all stocks
- [ ] Total portfolio size matches your risk tolerance

---

**Last Updated**: 2026-01-20
**For Quarter**: Q4 2025 (ending 2025-09-30)
**Next Update**: Late April 2026 (Q1 2026 results)
