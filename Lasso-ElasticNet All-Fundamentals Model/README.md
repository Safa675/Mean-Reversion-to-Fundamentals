# All-Fundamentals Lasso/ElasticNet Model

## Overview

This model addresses the **omitted variable bias** problem in the previous 6-7 factor model by using ALL ~500+ available fundamental variables and letting Lasso/ElasticNet automatically select the most relevant ones.

### Key Improvements

1. **No Manual Factor Construction**: Uses all raw fundamentals instead of pre-defined factors
2. **Automatic Feature Selection**: Lasso/ElasticNet selects the most predictive variables
3. **No Omitted Variable Bias**: Considers ALL available information
4. **Transparent**: Shows exactly which fundamentals matter most
5. **Robust**: Pre-filters by coverage and removes highly correlated features

---

## Differences from Previous Model

| Aspect | Previous Model (Lasso-Factor 0.1) | This Model |
|--------|----------------------------------|------------|
| **Features** | 6-7 pre-defined factors | ALL ~500+ z-scored fundamentals |
| **Selection** | Manual factor construction | Automatic Lasso/ElasticNet selection |
| **Coverage** | Factor_Intangibles only 1.1% | Filters features <10% coverage |
| **Bias** | Possible omitted variable bias | Minimal - uses all available data |
| **Interpretation** | Factor-level (Scale, Profitability, etc.) | Specific fundamentals (Revenue_z, NetDebt_z, etc.) |

---

## Quick Start

### 1. Run with Default Settings (Lasso, 20 features)

```bash
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"

# Using the venv
/home/safa/Documents/Fundamental Mean Reversion Models/BIST/.venv/bin/python all_fundamentals_model.py
```

### 2. Run with ElasticNet (30 features)

```bash
python all_fundamentals_model.py --method elasticnet --max-features 30
```

### 3. Adjust Coverage Requirements

```bash
# More lenient coverage (5% instead of 10%)
python all_fundamentals_model.py --min-coverage 5.0 --max-features 30

# Stricter coverage (20%)
python all_fundamentals_model.py --min-coverage 20.0 --max-features 15
```

### 4. Adjust Correlation Threshold

```bash
# More aggressive correlation removal (0.90 instead of 0.95)
python all_fundamentals_model.py --corr-threshold 0.90

# Less aggressive (0.98)
python all_fundamentals_model.py --corr-threshold 0.98
```

---

## How It Works

### Step 1: Load ALL Fundamentals

```
Total available: ~500+ z-scored fundamentals
Including:
  - Balance Sheet items (BS_*_z)
  - Income Statement items (IS_*_z)
  - Cash Flow items (CF_*_z)
  - Calculated ratios (*_Margin_z, *_Growth_z, etc.)
```

### Step 2: Filter by Coverage

```
Remove features with:
  - <10% overlap with target (LogMarketCap)
  - <500 observations

Example:
  Factor_Intangibles_z: 1.1% coverage → REMOVED
  Revenue_z: 99.9% coverage → KEPT
```

### Step 3: Remove Highly Correlated Features

```
For each pair with correlation >0.95:
  - Keep the one more correlated with target
  - Drop the other

Example:
  Revenue_z and IS_satis_gelirleri_z have corr=0.99
  → Keep the one with higher correlation to LogMarketCap
```

### Step 4: Lasso/ElasticNet Selection

```
Fit LassoCV or ElasticNetCV with cross-validation
  - Automatically selects optimal alpha (regularization strength)
  - Shrinks less important coefficients to zero
  - Keeps only the most predictive features

Example output:
  Starting features: 250
  After Lasso: 35 non-zero coefficients
  Top 20 selected for final model
```

### Step 5: Refit with OLS

```
Refit using OLS on selected features
  - Lasso/ElasticNet coefficients are biased (shrunk)
  - OLS gives unbiased estimates
  - Use robust standard errors (HC3)
```

### Step 6: Evaluate with IC

```
Information Coefficient (Spearman correlation):
  IC = corr(residuals, 1Q forward returns)

Negative IC is GOOD:
  - Undervalued stocks (negative residuals) outperform
  - Overvalued stocks (positive residuals) underperform

Target: IC < -0.10 with p < 0.05
```

---

## Output Files

All outputs are saved to `outputs/` subdirectory:

### 1. `all_fundamentals_{method}_results.csv`

Full dataset with predictions:
- All original columns
- `Predicted_LogMarketCap`: Model prediction
- `Residual_LogMarketCap`: Actual - Predicted
- `Overvaluation_pct`: % overvaluation

### 2. `all_fundamentals_{method}_model_summary.txt`

Contains:
- Selected features with coefficients and coverage
- Full OLS regression output (R², p-values, etc.)
- Information Coefficients for train/val/test periods

### 3. `all_fundamentals_{method}_selected_features.txt`

Detailed feature list showing:
- Rank
- Feature name
- Lasso/ElasticNet coefficient
- OLS coefficient
- Coverage %

---

## Interpreting Results

### Good Model Indicators

✅ **Test IC < -0.10 and p < 0.05**
  - Strong predictive signal on unseen data

✅ **IC strengthens from train → val → test**
  - Signal is robust, not overfit

✅ **R² 0.70-0.90**
  - Model explains significant variance in market cap

✅ **Most selected features have >50% coverage**
  - Predictions based on real data, not imputation

✅ **Feature coefficients make economic sense**
  - Positive: Revenue, Equity, Income (larger = higher valuation)
  - Negative: Debt, Losses (more debt/losses = lower valuation)

### Warning Signs

⚠️ **Test IC close to zero or positive**
  - Model doesn't predict forward returns well
  - May need more features or different method

⚠️ **IC degrades from train → test**
  - Possible overfitting
  - Try fewer features or stronger regularization

⚠️ **Many selected features with <20% coverage**
  - Predictions rely heavily on imputed/missing data
  - Increase min-coverage threshold

⚠️ **Illogical coefficient signs**
  - Revenue has negative coefficient (larger revenue = lower valuation?)
  - May indicate multicollinearity or data issues

---

## Comparison with Previous Model

Run both models and compare:

### Previous Model (Factor-Based):
```bash
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-Factor Mean Reversion 0.1"
python recommended_model_pipeline.py --sector ALL
```

### This Model (All-Fundamentals):
```bash
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"
python all_fundamentals_model.py --method lasso
```

### Expected Differences:

| Metric | Factor Model | All-Fundamentals Model |
|--------|-------------|------------------------|
| **Features Used** | 6 factors | 20-30 specific fundamentals |
| **IC (Test)** | -0.21 | ? (to be determined) |
| **R²** | ~0.85 | ? (possibly higher) |
| **Interpretability** | Factor-level | Variable-level (more granular) |
| **Omitted Variable Risk** | Higher | Lower |

---

## Advanced Usage

### Experiment with Different Methods

```bash
# Lasso (L1 regularization)
python all_fundamentals_model.py --method lasso --max-features 20

# ElasticNet (L1 + L2 regularization)
python all_fundamentals_model.py --method elasticnet --max-features 25
```

**When to use which:**
- **Lasso**: When you want sparse selection (many zeros)
- **ElasticNet**: When features are highly correlated (groups of related variables)

### Experiment with Feature Count

```bash
# Conservative (fewer features, less overfitting risk)
python all_fundamentals_model.py --max-features 10

# Moderate
python all_fundamentals_model.py --max-features 20

# Aggressive (more features, captures more signal but higher overfitting risk)
python all_fundamentals_model.py --max-features 40
```

### Create a Portfolio

After running the model, use the results to find undervalued stocks:

```python
import pandas as pd

# Load results
df = pd.read_csv("outputs/all_fundamentals_lasso_results.csv")

# Get latest quarter
latest_quarter = df['QuarterDate'].max()
df_latest = df[df['QuarterDate'] == latest_quarter].copy()

# Sort by residual (most undervalued first)
df_latest = df_latest.sort_values('Residual_LogMarketCap')

# Top 20 most undervalued
top_20 = df_latest.head(20)[['Ticker', 'Residual_LogMarketCap', 'Overvaluation_pct']]
print(top_20)
```

---

## Troubleshooting

### Error: "Only X features with sufficient coverage"

**Cause**: Too few features meet the coverage threshold

**Solution**: Lower min-coverage
```bash
python all_fundamentals_model.py --min-coverage 5.0
```

### Error: "Lasso selected ZERO features"

**Cause**: Regularization too strong (alpha too high)

**Solution**: This shouldn't happen with LassoCV (it auto-selects alpha), but if it does:
- Try ElasticNet instead
- Check if your data has enough variance

### Warning: "Test IC close to zero"

**Cause**: Model doesn't predict forward returns well

**Solutions**:
1. Try more features: `--max-features 30`
2. Try ElasticNet: `--method elasticnet`
3. Lower correlation threshold: `--corr-threshold 0.90`
4. This might just mean mean reversion isn't working in current regime

### Performance Issues

**Model takes too long to run:**
- Reduce feature pool by increasing `--min-coverage`
- Increase `--corr-threshold` to remove more correlated features
- Reduce `--max-features`

---

## Next Steps

### 1. Compare with Previous Model

Run both models and check if this model has:
- Higher IC on test set
- Better out-of-sample performance
- More stable predictions

### 2. Analyze Selected Features

Look at which fundamentals were selected:
- Are they mostly from one statement (BS/IS/CF)?
- Do they make economic sense?
- Are they stable across different runs?

### 3. Build Portfolio

Use the latest quarter predictions to:
- Identify top 20 undervalued stocks
- Apply quality filters (positive earnings, adequate liquidity)
- Backtest the strategy

### 4. Monitor IC Over Time

Track IC each quarter to detect signal decay:
```
2024-Q1: IC = -0.15
2024-Q2: IC = -0.12
2024-Q3: IC = -0.08  ← Signal decaying, retrain soon
```

---

## Technical Notes

### Why Z-Score Fundamentals?

All fundamentals are z-scored (mean=0, std=1) before model training:
- Makes variables comparable (different units/scales)
- Reduces impact of outliers (already winsorized at 1%-99%)
- Standard practice in factor models

### Why Refit with OLS?

Lasso/ElasticNet coefficients are **biased** (shrunk toward zero):
- Good for feature selection
- Bad for interpretation and prediction

OLS coefficients are **unbiased**:
- Better for final predictions
- More interpretable (true marginal effects)

### Why Walk-Forward Validation?

Three-way split prevents data leakage:
- **Train (2016-2019)**: Fit Lasso, select features
- **Validation (2020-2022)**: Check if signal holds
- **Test (2023-2025)**: Final out-of-sample test

This mimics real-world usage where you:
1. Train on historical data
2. Validate before deploying
3. Test on truly unseen future data

---

## Contact & Support

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Review the output logs (saved in `.txt` files)
3. Compare with the previous factor model to isolate the issue

**Key Insight**: This model should have **lower omitted variable bias** than the previous 6-7 factor model because it considers ALL available fundamentals rather than a small pre-selected set.

---

**Good luck with your model! The automatic feature selection should identify the truly important fundamentals for predicting market cap.**
