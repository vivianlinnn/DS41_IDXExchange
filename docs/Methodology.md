# Model Methodology

## 1. Baseline Model (Linear Regression model _ Liz Choi + Evaluation_Steven)

We first built a Linear Regression model using core numerical housing features:

**Features:**
- LivingArea
- BedroomsTotal
- BathroomsTotalInteger
- LotSizeAcres
- YearBuilt

These variables capture the fundamental structural characteristics of a property.

---

## 2. Model Improvement

To improve predictive performance, we incorporated additional categorical features:

- ViewYN
- PoolPrivateYN
- NewConstructionYN
- FireplaceYN

Since Linear Regression requires numerical inputs, categorical variables were converted using one-hot encoding.

## 3. Evaluation

We evalated the model using matrics R^2, MAPE and MdAPE:

LOG SCALE PERFORMANCE
-R² (Log): 0.4520
-MAPE (Log): 2.58%
-MdAPE (Log): 2.09%

ORIGINAL SCALE PERFORMANCE
-R² (Original): -0.0100
-MAPE (Original): 39.10%
-MdAPE (Original): 28.41%

Log Scale Performance
-R² (Log scale): 0.4520 Indicates that the model explains 45.2% of the variance in the log-transformed target, meaning the selected features capture a moderate amount of variability in log house prices.

- MAPE (Mean Absolute Percentage Error): 2.58% On average, predictions deviate by only 2.58% in log space, showing the model fits the log-transformed values very closely.

-MdAPE (Median Absolute Percentage Error): 2.09% The typical (median) prediction error is just 2.09%, confirming that most predictions are highly accurate in log space and robust to extreme values.
