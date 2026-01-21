#!/bin/bash

# Test script for the all-fundamentals model
# This will run a quick test with reduced feature count

cd "/home/safa/Documents/Fundamental Mean Reversion Models/BIST/Lasso-ElasticNet All-Fundamentals Model"

echo "Testing All-Fundamentals Model..."
echo "================================="
echo ""

# Run with conservative settings for testing
python all_fundamentals_model.py \
  --method lasso \
  --max-features 15 \
  --min-coverage 20.0 \
  --corr-threshold 0.95

echo ""
echo "================================="
echo "Test Complete!"
echo ""

if [ $? -eq 0 ]; then
    echo "✓ Model ran successfully!"
    echo ""
    echo "Check outputs:"
    echo "  - outputs/all_fundamentals_lasso_model_summary.txt"
    echo "  - outputs/all_fundamentals_lasso_results.csv"
    echo "  - outputs/all_fundamentals_lasso_selected_features.txt"
else
    echo "✗ Model failed. Check errors above."
    exit 1
fi
