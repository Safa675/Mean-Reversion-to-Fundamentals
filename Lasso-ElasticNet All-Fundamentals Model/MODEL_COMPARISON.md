# Model Comparison: Factor-Based vs All-Fundamentals

## Executive Summary

This document compares the two approaches for fundamental mean reversion modeling:

1. **Factor-Based Model** (Lasso-Factor Mean Reversion 0.1)
2. **All-Fundamentals Model** (this model)

---

## Key Differences

### 1. Feature Construction

| Aspect | Factor-Based | All-Fundamentals |
|--------|-------------|------------------|
| **Input** | 6-7 manually constructed factors | ~500 z-scored fundamentals |
| **Construction** | Factor_Scale = mean(Revenue_z, OpIncome_z, ...) | Uses raw fundamentals directly |
| **Coverage** | Factor_Intangibles: 1.1% | All features >10% coverage |
| **Selection** | Manual/pre-defined | Automatic via Lasso/ElasticNet |

**Example:**

Factor-Based:
```python
Factor_Scale = mean([Revenue_z, OperatingIncome_z, FreeCashFlow_z, ...])
# Averages 7 fundamentals into 1 factor
```

All-Fundamentals:
```python
# Lasso selects individually:
Selected: [Revenue_z, OperatingIncome_z, NetDebt_z, IS_faaliyet_kari_zarari_z, ...]
# Each fundamental enters separately with its own coefficient
```

---

### 2. Omitted Variable Bias

**Factor-Based Model:**
- Uses only 6-7 factors
- Factor_Intangibles (Goodwill) has 1.1% coverage → effectively unused
- May miss important fundamentals that don't fit into pre-defined factors
- **Risk**: High omitted variable bias

**All-Fundamentals Model:**
- Starts with ALL ~500 fundamentals
- Lasso/ElasticNet automatically selects the most predictive ones
- No assumptions about which fundamentals matter
- **Risk**: Lower omitted variable bias

**Why This Matters:**

If an important fundamental (e.g., `IS_net_faiz_geliri_gideri_z` - net interest income for banks) doesn't fit neatly into your 6-7 factors, the factor-based model will miss it. The all-fundamentals model won't.

---

### 3. Interpretability

**Factor-Based Model:**
```
Selected Features:
  - Factor_Scale (coef = +0.85)
  - Factor_Profitability (coef = +0.12)
  - Factor_Growth (coef = +0.05)
```

**Interpretation**: "Larger, more profitable, faster-growing companies have higher valuations"
- High-level, intuitive
- But you don't know which specific fundamentals drive Scale/Profitability/Growth

**All-Fundamentals Model:**
```
Selected Features:
  - Revenue_z (coef = +0.42)
  - NetDebt_z (coef = -0.18)
  - IS_faaliyet_kari_zarari_z (coef = +0.25)
  - BS_maddi_duran_varliklar_z (coef = +0.15)
```

**Interpretation**: "Companies with higher revenue, lower debt, higher operating income, and more fixed assets have higher valuations"
- Granular, specific
- Tells you exactly which line items from financial statements matter

**Trade-off:**
- Factor-based: Easier to explain to non-technical audience
- All-fundamentals: More actionable for analysts (know exactly what to look for)

---

### 4. Model Comparison Table

| Metric | Factor-Based | All-Fundamentals | Winner |
|--------|-------------|------------------|--------|
| **# Input Features** | 7 factors | ~250-300 (after filtering) | - |
| **# Selected Features** | 6 (Factor_Intangibles dropped) | 20-30 (Lasso/ElasticNet selected) | - |
| **Coverage** | Factor_Intangibles 1.1% | All >10% | ✅ All-Fundamentals |
| **Omitted Variable Bias** | Higher risk | Lower risk | ✅ All-Fundamentals |
| **Interpretability** | Factor-level (easier) | Variable-level (more specific) | Tie |
| **IC (Test Set)** | -0.21 | ? (to be measured) | TBD |
| **R²** | ~0.85 | ? (possibly higher) | TBD |
| **Training Time** | Fast (6 features) | Slower (250+ features) | ✅ Factor-Based |

---

## When to Use Each Model

### Use Factor-Based Model When:

1. **Quick iteration needed**
   - Factor-based model trains faster
   - Good for rapid prototyping

2. **Presentation to non-technical audience**
   - Easier to explain "Scale, Profitability, Growth"
   - Less overwhelming than 20-30 specific fundamentals

3. **Limited computational resources**
   - Smaller feature space = faster training

4. **You have strong prior beliefs about factor structure**
   - E.g., "I believe Scale, Profitability, and Growth are the only things that matter"

### Use All-Fundamentals Model When:

1. **Maximizing predictive accuracy**
   - Lower omitted variable bias
   - Let the data speak

2. **Research/discovery mode**
   - Find out which fundamentals actually matter
   - May discover unexpected relationships

3. **Avoiding assumptions**
   - Don't want to impose factor structure
   - Let Lasso/ElasticNet decide what's important

4. **Granular actionable insights**
   - Need to know specific line items to focus on
   - E.g., "We should focus on companies with low NetDebt_z and high Operating_Margin_z"

---

## Expected Performance Differences

### Scenario 1: Factor-Based Model Wins

**Condition**: The 6-7 pre-defined factors capture most of the signal

**Example**: If valuation really is just about Scale, Profitability, and Growth:
```
Factor-Based IC: -0.21
All-Fundamentals IC: -0.19
```

**Why**: All-fundamentals model adds noise by including irrelevant fundamentals

**Indicator**: Most selected fundamentals in all-fundamentals model belong to Scale/Profitability/Growth

### Scenario 2: All-Fundamentals Model Wins

**Condition**: Important fundamentals were omitted from the 6-7 factors

**Example**: Bank-specific fundamentals (net interest income) or sector-specific items matter:
```
Factor-Based IC: -0.21
All-Fundamentals IC: -0.26
```

**Why**: All-fundamentals model captures signal that factor-based model missed

**Indicator**: Selected fundamentals include items NOT in the 6-7 factors

### Scenario 3: Similar Performance

**Condition**: The 6-7 factors are well-designed and capture most signal

**Example**:
```
Factor-Based IC: -0.21
All-Fundamentals IC: -0.22
```

**Why**: Diminishing returns from additional fundamentals

**Decision**: Use factor-based model (simpler, faster)

---

## How to Compare

### Step 1: Run Both Models

```bash
# Factor-based model
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-Factor Mean Reversion 0.1"
python recommended_model_pipeline.py --sector ALL

# All-fundamentals model
cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"
python all_fundamentals_model.py --method lasso --max-features 20
```

### Step 2: Compare Test IC

```
Factor-Based Test IC: -0.2125 (from PORTFOLIO_RESULTS.md)
All-Fundamentals Test IC: ??? (check outputs/all_fundamentals_lasso_model_summary.txt)

IF All-Fundamentals IC is MORE negative by >0.02:
  → All-fundamentals model is better
  → Use it for portfolio construction

IF similar (within ±0.02):
  → Use factor-based model (simpler)

IF All-Fundamentals IC is LESS negative:
  → Factor-based model is better
  → All-fundamentals model is overfitting or adding noise
```

### Step 3: Compare Feature Composition

Look at selected features in all-fundamentals model:

**Check 1: Are they mostly from the 6-7 factors?**
```
Selected features:
  - Revenue_z (part of Factor_Scale) ✓
  - OperatingIncome_z (part of Factor_Scale) ✓
  - FreeCashFlow_z (part of Factor_Scale) ✓
  - Gross_Margin_z (part of Factor_Profitability) ✓
  - Revenue_Growth_z (part of Factor_Growth) ✓

Conclusion: Factor-based model was well-designed.
             No major omissions.
```

**Check 2: Do they include new fundamentals?**
```
Selected features:
  - Revenue_z (part of Factor_Scale) ✓
  - BS_finansal_borclar_z (NOT in any factor) ← NEW
  - IS_net_faiz_geliri_z (NOT in any factor) ← NEW
  - CF_yatirim_faaliyetlerinden_nakit_z (NOT in any factor) ← NEW

Conclusion: Factor-based model missed important fundamentals.
             All-fundamentals model captures additional signal.
```

### Step 4: Compare R²

```
Factor-Based R²: ~0.85
All-Fundamentals R²: ???

Higher R² = better fit to market cap
(But watch for overfitting - check if test IC also improved)
```

### Step 5: Compare Overvaluation Predictions

For the same stock on the same date:

```python
import pandas as pd

# Load both results
df_factor = pd.read_csv("../Lasso-Factor Mean Reversion 0.1/recommended_model_outputs/ALL_lasso_results.csv")
df_all = pd.read_csv("outputs/all_fundamentals_lasso_results.csv")

# Compare predictions for a specific stock
ticker = "TUPRS"
quarter = "2025-09-30"

factor_pred = df_factor[(df_factor['Ticker']==ticker) & (df_factor['QuarterDate']==quarter)]
all_pred = df_all[(df_all['Ticker']==ticker) & (df_all['QuarterDate']==quarter)]

print(f"Factor-Based: Overvaluation = {factor_pred['Overvaluation_pct'].values[0]:.2f}%")
print(f"All-Fundamentals: Overvaluation = {all_pred['Overvaluation_pct'].values[0]:.2f}%")
```

**Large differences suggest**:
- Models are capturing different signals
- One may be more accurate than the other
- Check which one has better IC

---

## Recommendation

### For Maximum Accuracy:
→ **Use All-Fundamentals Model**

Reasons:
1. Lower omitted variable bias
2. Captures all available information
3. Let the data decide what matters

### For Simplicity/Speed:
→ **Use Factor-Based Model**

Reasons:
1. Faster to train
2. Easier to explain
3. Good enough if IC is similar

### Hybrid Approach:
→ **Use Both**

1. Run all-fundamentals model to discover important fundamentals
2. Group discovered fundamentals into intuitive factors
3. Create improved factor-based model with new factors

Example:
```
Discovery (All-Fundamentals):
  Important: Revenue_z, NetDebt_z, IS_net_faiz_geliri_z, Cash_z

New Factor (Liquidity):
  Factor_Liquidity = mean([Cash_z, FreeCashFlow_z, ...])

New Factor (Financial Health):
  Factor_FinHealth = mean([NetDebt_z, IS_net_faiz_geliri_z, ...])
```

---

## Conclusion

Both models have merits:

**Factor-Based Model:**
- Simpler, faster, more interpretable
- Test IC = -0.21 (excellent)
- May have omitted variable bias

**All-Fundamentals Model:**
- More comprehensive, lower bias
- Test IC = ??? (to be measured)
- More complex, slower

**Next Step**: Run both models and compare test IC. If all-fundamentals IC is significantly better (>-0.23), use it. If similar, stick with factor-based for simplicity.

**Key Question**: Did the 6-7 manual factors miss important fundamentals? The all-fundamentals model will answer this.
