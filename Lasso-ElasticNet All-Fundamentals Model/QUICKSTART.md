# Quick Start Guide

## 1. Run the Model (2 minutes)

```bash
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"

# Make sure you're using the venv with all dependencies
python all_fundamentals_model.py
```

**Expected output:**
```
Loading panel data...
Loaded 16,776 observations
...
Test IC = -0.XX (p=0.XXX, N=X,XXX)
[SUCCESS] Model completed successfully!
```

---

## 2. Check Results (1 minute)

```bash
# View model summary
cat outputs/all_fundamentals_lasso_model_summary.txt | head -100

# View selected features
cat outputs/all_fundamentals_lasso_selected_features.txt | head -50
```

**Look for:**
- ✅ Test IC < -0.10 with p < 0.05 (good signal)
- ✅ Selected features make economic sense
- ✅ Most features have >20% coverage

---

## 3. Find Undervalued Stocks (30 seconds)

```bash
python find_undervalued.py
```

**Output:**
```
TOP 20 MOST UNDERVALUED STOCKS - 2025-09-30
================================================
Rank   Ticker     Residual     Underval %   Sector
------------------------------------------------------
1      AGHOL      -2.5123      -91.83%      IND
2      TUPRS      -2.3456      -90.42%      COM
...
```

---

## 4. Compare with Previous Model (5 minutes)

```bash
# Check previous model IC
cat "../Lasso-Factor Mean Reversion 0.1/recommended_model_outputs/ALL_lasso_model_summary.txt" | grep "test_ic"

# Compare:
# Factor-Based Test IC: -0.2125
# All-Fundamentals Test IC: ??? (from step 1)
```

**Decision:**
- If All-Fundamentals IC is MORE negative → Use this model
- If similar (within ±0.02) → Use factor-based (simpler)
- If LESS negative → Factor-based is better

---

## 5. Experiment (optional)

### Try ElasticNet
```bash
python all_fundamentals_model.py --method elasticnet --max-features 25
```

### Try More Features
```bash
python all_fundamentals_model.py --max-features 30
```

### Lower Coverage Threshold
```bash
python all_fundamentals_model.py --min-coverage 5.0 --max-features 30
```

---

## Common Issues

### Error: "No module named pandas"
```bash
# Use the venv
/home/safa/Documents/Fundamental Mean Reversion Models/BIST/.venv/bin/python all_fundamentals_model.py
```

### Warning: "Test IC close to zero"
- Try: `--method elasticnet`
- Try: `--max-features 30`
- May just mean the signal is weak in current regime

### Model takes too long
- Increase: `--min-coverage 15.0` (fewer features)
- Increase: `--corr-threshold 0.98` (remove more correlated features)

---

## Files Created

After running, check:
```
outputs/
├── all_fundamentals_lasso_results.csv          ← Full predictions
├── all_fundamentals_lasso_model_summary.txt    ← Model stats & IC
├── all_fundamentals_lasso_selected_features.txt ← Which fundamentals were chosen
├── top_undervalued_20250930.txt                 ← Portfolio (from find_undervalued.py)
└── top_undervalued_20250930.csv                 ← Portfolio CSV
```

---

## Next Steps

1. ✅ Compare IC with previous model
2. ✅ Analyze which fundamentals were selected
3. ✅ Build portfolio from undervalued stocks
4. ✅ Apply quality filters (positive earnings, liquidity)
5. ✅ Monitor IC over time to detect signal decay

---

**That's it! You now have a model that uses ALL available fundamentals to avoid omitted variable bias.**

For more details, see [README.md](README.md) and [MODEL_COMPARISON.md](MODEL_COMPARISON.md).
