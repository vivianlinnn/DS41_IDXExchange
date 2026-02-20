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
