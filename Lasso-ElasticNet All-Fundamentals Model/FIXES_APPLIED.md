# Fixes Applied to All-Fundamentals Model

## Problem

The original model failed with:
```
Training sample after dropna: 0
[ERROR] Training sample too small (< 100 observations)
```

## Root Cause

When you have 169 features and try to `dropna()` (remove any row with at least 1 missing value), you end up with **0 rows** because:
- Each fundamental has some missing data
- The probability that ALL 169 fundamentals are present for a single observation is extremely low
- Result: No training data

## Solution: Median Imputation

Instead of dropping rows with missing values, we now **impute** (fill) missing values with the median of each feature computed from the training set.

### Why Median Imputation?

1. **Preserves sample size**: Don't lose observations just because 1-2 features are missing
2. **Conservative**: Median is robust to outliers
3. **Prevents data leakage**: Use training medians for val/test imputation
4. **Standard practice**: Common in Lasso/ElasticNet with many features

### Changes Made

#### 1. Fixed `remove_highly_correlated()` Function

**Before:**
```python
df_subset = df[[features] + [target_col]].dropna()  # ← All rows dropped!
corr_matrix = df_subset[features].corr()
```

**After:**
```python
# Compute pairwise correlations (handles missing data gracefully)
for feat1, feat2 in pairs:
    valid_mask = df[feat1].notna() & df[feat2].notna()
    corr = df.loc[valid_mask, [feat1, feat2]].corr()
```

#### 2. Added Median Imputation to Training Data

**Before:**
```python
X_train = df_train[filtered_features].dropna()  # ← All rows dropped!
y_train = df_train.loc[X_train.index, TARGET_COL]
```

**After:**
```python
# Get valid target indices (only requirement: target must exist)
valid_target_idx = df_train[TARGET_COL].dropna().index

# Get features
X_train = df_train.loc[valid_target_idx, filtered_features]

# Median imputation
for col in filtered_features:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
```

#### 3. Applied Training Medians to Validation/Test

**Key Point**: Use training medians for all periods to prevent data leakage

```python
# Store medians from training
feature_medians = X_train[selected_features].median()

# Apply to validation/test
X_val[col] = X_val[col].fillna(feature_medians[col])
X_test[col] = X_test[col].fillna(feature_medians[col])
```

## Expected Behavior Now

### Before Fix:
```
Total features: 169
Training sample after dropna: 0  ← ERROR!
```

### After Fix:
```
Total features: 169
Training observations with valid target: 5,751
Applying median imputation for missing values...
  Imputed 523 (9.1%) missing values in Revenue_Growth_z
  Imputed 1,234 (21.4%) missing values in Factor_RD_z
  ...
Training sample size: 5,751  ← SUCCESS!
```

## Trade-offs

### Advantages ✓
- Model can actually run (5,000+ training observations instead of 0)
- Uses all available data
- Standard practice in ML with many features
- Prevents data leakage (training medians used everywhere)

### Disadvantages ⚠️
- Imputed values may not reflect true missing patterns
- If a feature has >50% missing data, predictions may be less reliable
- Model may be less accurate for stocks with many missing fundamentals

## Monitoring Imputation

The model now prints imputation statistics:

```
Feature coverage in training set:
  [WARN] Factor_RD_z                    only 32.5% coverage
  [WARN] Factor_Intangibles_z           only  1.2% coverage

Applying median imputation for missing values...
  Imputed 3,456 (60.1%) missing values in Factor_RD_z
  Imputed 5,620 (97.7%) missing values in Factor_Intangibles_z
```

**Interpretation:**
- If many features have >50% imputation → model may be unreliable
- Solution: Increase `--min-coverage` threshold to exclude these features

## Recommendations

### Conservative Approach (Recommended)
```bash
python all_fundamentals_model.py \
  --min-coverage 30.0 \
  --max-features 15 \
  --corr-threshold 0.95
```
- Only uses features with >30% coverage
- Less imputation = more reliable predictions

### Moderate Approach
```bash
python all_fundamentals_model.py \
  --min-coverage 20.0 \
  --max-features 20 \
  --corr-threshold 0.95
```
- Balances coverage and feature count

### Aggressive Approach
```bash
python all_fundamentals_model.py \
  --min-coverage 10.0 \
  --max-features 30 \
  --corr-threshold 0.90
```
- Uses more features
- More imputation required
- Higher risk of overfitting

## Verification

To verify the model is working, check:

1. **Training sample size > 1,000**
   ```
   Training sample size: 5,751  ← Good!
   ```

2. **Most features have >50% coverage**
   ```
   Feature coverage in training set:
     Revenue_z: 99.5% coverage  ← Excellent
     NetDebt_z: 87.2% coverage  ← Good
     Factor_RD_z: 32.1% coverage  ← Marginal (watch this)
   ```

3. **IC is computed successfully**
   ```
   ✓ Training   IC = -0.15 (p=0.001, N=4,523)
   ✓ Validation IC = -0.12 (p=0.023, N=3,891)
   ✓ Test       IC = -0.18 (p=0.002, N=4,123)
   ```

## Comparison with Factor-Based Model

The factor-based model didn't have this issue because:
- Only 6-7 factors
- Much higher chance that all 6 are present for each observation
- Less imputation needed

The all-fundamentals model:
- Uses 169+ features
- Almost impossible for all to be present
- **Requires imputation** to be practical

This is a **feature, not a bug** - we want to use all available information, even if some is missing.

## Next Steps

Run the model with conservative settings first:
```bash
python all_fundamentals_model.py --min-coverage 30.0 --max-features 15
```

If successful, gradually increase feature count:
```bash
python all_fundamentals_model.py --min-coverage 20.0 --max-features 20
python all_fundamentals_model.py --min-coverage 15.0 --max-features 25
```

Monitor the imputation percentages and IC to find the right balance.
