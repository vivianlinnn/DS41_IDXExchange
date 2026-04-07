# DS41_IDXExchange
# House Price Prediction Project

## 1. Project Overview

This project aims to predict residential property closing prices in California using supervised machine learning models. The prediction target is the final transaction price, `ClosePrice`, and the modeling task is framed as a regression problem. Because housing prices are highly skewed and contain extreme values, the models are trained on a log-transformed target, `LogClosePrice`, rather than on raw prices directly.

The project compares models of increasing complexity, beginning with a simple linear baseline and extending to tree-based ensemble methods and gradient boosting models. The main goal is not only to improve predictive accuracy, but also to understand how preprocessing, feature engineering, and model selection affect performance and generalization.

The models evaluated in this project include:

- Linear Regression
- Decision Tree
- Decision Tree with engineered features
- Random Forest
- Random Forest with leakage-prone features
- XGBoost
- LightGBM

Model performance is compared using both statistical and practical error measures, including \(R^2\), MAPE, and MdAPE.

---

## 2. Data Source

The dataset consists of California residential property transaction records. Each observation represents a sold property, and the available variables describe the property itself, its location, and its transaction timing.

The raw data includes several categories of information:

### 2.1 Property Attributes
These variables describe the structure and size of the house. Examples include:

- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeSquareFeet`
- `YearBuilt`
- `GarageSpaces`
- `ParkingTotal`
- `Levels`
- `Stories`
- `PoolPrivateYN`
- `FireplaceYN`

### 2.2 Location Information
These variables capture neighborhood or geographic context. Examples include:

- `PostalCode`
- `City`
- `MLSAreaMajor`
- `HighSchoolDistrict`
- `Latitude`
- `Longitude`

### 2.3 Transaction and Time Information
These variables describe when the transaction occurred and help model market timing effects. Examples include:

- `CloseDate`
- Derived month index relative to a reference date
- Property age based on year built

### 2.4 Target Variable
The original response variable is:

- `ClosePrice`

Because raw housing prices are strongly right-skewed, the target used for training is:

\[
\text{LogClosePrice} = \log(\text{ClosePrice})
\]

Using the log-transformed target improves numerical stability, reduces the influence of extreme values, and helps the models learn relative price patterns more effectively.

---

## 3. Preprocessing

The preprocessing pipeline was designed to create a clean, consistent, and leakage-aware dataset for modeling. The overall logic follows a clear order:

1. clean the raw data,
2. define the time-based split,
3. transform the target,
4. engineer informative features,
5. encode location information,
6. remove problematic variables,
7. construct the final modeling tables.

This section is detailed because preprocessing is one of the most important parts of the project. Much of the final model performance depends not only on the choice of algorithm, but on how the raw housing data is prepared before training.

### 3.1 Data Cleaning and Type Standardization

The raw dataset contains mixed formats and potentially invalid entries across numeric columns. Before any model is trained, the features must be converted into a consistent numerical form.

The following cleaning logic is applied:

- Numeric columns are explicitly converted using `pd.to_numeric(..., errors='coerce')`
- Non-numeric or malformed values are coerced to `NaN`
- Infinite values such as `inf` and `-inf` are replaced with `NaN`
- Rows missing required predictors are removed before model fitting

This step is necessary for three reasons:

- Most machine learning models require numeric input
- Invalid values can silently distort model training
- A common cleaning process ensures fair comparison across models

In practice, this means that the dataset used by the models is not simply the raw export, but a filtered and standardized version suitable for regression.

---

### 3.2 Time-Based Train/Test Split

The data is split chronologically rather than randomly. This is important because housing price prediction is naturally a forward-looking task: the model should be trained on past transactions and tested on later ones.

The project uses a time-based split where:

- the training set contains earlier transactions,
- the test set contains later transactions,
- forward evaluation better reflects real deployment conditions.

This design avoids an overly optimistic evaluation that can happen when future examples leak into the training sample through random splitting.

The logic is:

- train on past market behavior,
- test on future market behavior,
- measure whether the model generalizes to unseen time periods.

This is especially important in real estate because market conditions shift over time.

---

### 3.3 Target Transformation

The raw target, `ClosePrice`, is highly skewed. Expensive properties create a long right tail, which can make regression unstable and can cause large houses to dominate the loss function.

To address this, the target is transformed using the natural logarithm:

\[
\text{LogClosePrice} = \log(\text{ClosePrice})
\]

This transformation has several benefits:

- reduces skewness,
- stabilizes variance,
- improves model fit,
- makes relative error structure easier to learn.

Training is performed on `LogClosePrice`, but evaluation is conducted both on the log scale and on the original dollar scale.

When predictions are interpreted in actual price units, the inverse transformation is applied:

\[
\widehat{\text{ClosePrice}} = \exp(\widehat{\text{LogClosePrice}})
\]

This allows the project to report practical measures such as percentage prediction error in real price terms.

---

### 3.4 Feature Engineering

Raw variables alone do not fully capture the relationships that drive house prices. As a result, several engineered features are created to represent structural efficiency, density, and time-related effects more directly.

Feature engineering is one of the core improvements in this project.

#### 3.4.1 Ratio-Based Features

Several ratio features are created to express the internal structure of a house more meaningfully than raw counts alone.

##### BedBathRatio

\[
\text{BedBathRatio} = \frac{\text{BedroomsTotal}}{\text{BathroomsTotalInteger}}
\]

This feature captures the balance between bedrooms and bathrooms. Two homes with the same bedroom count may differ substantially in value depending on whether the bathroom count is low or adequate.

##### SqftPerBedroom

\[
\text{SqftPerBedroom} = \frac{\text{LivingArea}}{\text{BedroomsTotal}}
\]

This feature measures how much living space is available per bedroom. It helps distinguish between homes that are spacious and homes that are crowded relative to room count.

##### LotUtilization

\[
\text{LotUtilization} = \frac{\text{LivingArea}}{\text{LotSizeSquareFeet}}
\]

This feature measures how intensively the lot is used. It can help separate compact high-utility homes from properties with large lots but relatively small living space.

#### 3.4.2 Time and Age Features

##### PropertyAge

\[
\text{PropertyAge} = \text{Reference Year} - \text{YearBuilt}
\]

Property age helps capture depreciation, modernization, and renovation effects. Older homes may sell for less unless offset by location or upgrades.

##### MonthsFromDec2025

A month-offset feature is constructed relative to a fixed reference point, December 2025. This allows the model to represent market timing and transaction recency in a simple numerical form.

This feature is useful because house prices are influenced not only by structure and location, but also by the timing of the sale.

#### 3.4.3 Edge Case Handling in Engineered Features

Ratio features can create division-by-zero problems when denominators are zero or missing. To avoid invalid values:

- zero denominators are treated carefully,
- invalid ratios are converted to missing values,
- the resulting data is cleaned before model fitting.

This prevents the engineered features from introducing instability into the model pipeline.

---

### 3.5 Location Encoding

Location is one of the strongest drivers of housing price. However, raw categorical location variables such as ZIP code can create very high-dimensional representations if one-hot encoded directly.

To solve this, the project uses a target-based location encoding:

- `ZipMedianPrice`

This feature is computed by taking the median `LogClosePrice` for each ZIP code in the training set. Each property is then assigned the corresponding ZIP-level median value.

Formally, for ZIP code \(z\),

\[
\text{ZipMedianPrice}_z = \text{median}(\text{LogClosePrice} \mid \text{PostalCode} = z)
\]

This design has several advantages:

- captures neighborhood-level price patterns,
- reduces dimensionality,
- works well with tree-based models,
- avoids sparse one-hot vectors.

A very important rule is followed here:

- ZIP statistics are computed using training data only.

This prevents the test set from influencing the representation learned during training.

For ZIP codes that appear in the test set but not in the training set, a fallback value such as the global median log price is used. This ensures that unseen locations can still be handled consistently.

---

### 3.6 Feature Selection Logic

Not every available variable is equally useful. Some features carry strong predictive signal, while others add noise or increase overfitting risk.

Feature selection is based on a combination of:

- univariate predictive strength,
- domain knowledge,
- model suitability,
- generalization concerns.

For example, earlier exploratory analysis showed that some location-related variables, especially `PostalCode`, had strong standalone explanatory power, while many weak variables contributed very little.

In simpler models, feature selection is stricter to preserve interpretability and reduce noise. In more flexible models, broader feature sets can be used as long as they do not create leakage.

This creates a progression across the project:

- simple baseline with a minimal feature set,
- stronger tree models with selected structural and location features,
- boosted models with richer engineered features.

---

### 3.7 Data Leakage Prevention

Leakage prevention is a critical part of preprocessing. A model that indirectly receives the answer through a predictor may appear extremely accurate, but that performance will not hold in real use.

The clearest leakage issue in this project is the feature:

- `PPSF` (Price Per Square Foot)

This variable is problematic because:

\[
\text{PPSF} \times \text{LivingArea} \approx \text{ClosePrice}
\]

That means a model using both `PPSF` and `LivingArea` can nearly reconstruct the target directly. As a result:

- error metrics become unrealistically low,
- \(R^2\) becomes suspiciously high,
- the model no longer represents a valid predictive system.

For this reason, models that include such features are treated as leakage cases and are not considered valid final models, even if their numerical performance appears best.

Leakage prevention also affects location encoding:

- ZIP-level target statistics are computed from training data only,
- future or test information is never used to define training features.

This ensures that reported performance reflects genuine predictive power rather than information shortcuts.

---

### 3.8 Final Modeling Dataset

After cleaning, transformation, feature engineering, encoding, and leakage control, the final modeling dataset contains:

- standardized numeric predictors,
- engineered structural and timing features,
- encoded location features,
- a log-transformed target.

At that point, the final design matrices are created:

- \(X\): feature matrix,
- \(y\): target vector (`LogClosePrice`).

Different models use slightly different feature subsets, but all are derived from the same preprocessing logic.

---

### 3.9 Summary of the Preprocessing Pipeline

The full preprocessing pipeline can be summarized as follows:

1. Convert raw columns into consistent numeric formats  
2. Remove invalid and unusable values  
3. Split the data chronologically into training and testing periods  
4. Transform `ClosePrice` into `LogClosePrice`  
5. Create engineered ratio, age, and time features  
6. Encode ZIP-level neighborhood price effects using training data only  
7. Remove weak, redundant, or leakage-prone features  
8. Build the final feature matrix for modeling  

This preprocessing design is one of the main reasons the stronger models perform well. It provides cleaner inputs, more meaningful features, and a more realistic evaluation setting.

---

## 4. Models Tested

The project compares several regression models with increasing flexibility and predictive power.

### 4.1 Linear Regression (Baseline)

The baseline model is a linear regression model trained on a very simple feature setup. In the baseline analysis, `PostalCode` is used as the main predictor because it showed the strongest single-feature explanatory power.

This model serves as an interpretable reference point.

#### Strengths
- simple and transparent,
- fast to train,
- easy to explain,
- stable across training and test sets.

#### Weaknesses
- cannot model nonlinear effects,
- cannot automatically capture interactions,
- limited predictive capacity for a complex housing market.

The baseline is useful because it shows what can be achieved with a very simple specification before adding more advanced modeling.

---

### 4.2 Decision Tree

The decision tree model is introduced to capture nonlinear structure in the data. Unlike linear regression, a tree can represent threshold effects and hierarchical feature interactions automatically.

An initial shallow decision tree is evaluated first. This version underfits because it is too simple to represent the complexity of housing price patterns.

Even so, it helps demonstrate an important lesson:

- not all nonlinear models are automatically strong,
- model capacity and feature design still matter.

---

### 4.3 Decision Tree with Engineered Features

A stronger decision tree model is then built using more carefully selected and engineered features. Instead of relying on raw location categories, it uses `ZipMedianPrice` together with key structural variables such as:

- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- engineered ratios
- age/time variables

This improved tree performs much better than the shallow version and substantially improves over the linear baseline.

It shows that tree models benefit greatly from:

- better feature design,
- stronger location representation,
- moderate regularization.

This model also remains relatively interpretable compared with larger ensemble methods.

---

### 4.4 Random Forest

The random forest model extends the tree approach by averaging many decision trees. This improves stability and usually increases predictive accuracy.

The random forest captures:

- nonlinear relationships,
- feature interactions,
- more flexible decision boundaries than a single tree.

Compared with the baseline and individual tree models, random forest provides a major performance gain. However, the gap between training and testing results indicates mild overfitting, which is expected in a flexible ensemble model.

Still, it is a strong and practical model, especially before moving to boosting methods.

---

### 4.5 Random Forest with Leakage-Prone Features

A version of random forest that includes additional features such as `PPSF` produces near-perfect results. However, this model is not considered valid because the added features leak direct information about the target.

This model is useful only as a diagnostic example showing why preprocessing and feature screening matter.

Its performance should not be interpreted as true predictive ability.

---

### 4.6 XGBoost

XGBoost is a gradient boosting tree model that builds trees sequentially, with each new tree attempting to correct the errors of the previous ones.

This model is well suited for tabular data and performs especially well when:

- relationships are nonlinear,
- interactions matter,
- feature effects vary across regions and price levels.

The project uses a tuned XGBoost model with features such as:

- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `BedBathRatio`
- `SqftPerBedroom`
- `LotUtilization`
- `PropertyAge`
- `MonthsFromDec2025`
- `ZipMedianPrice`

Hyperparameter tuning is applied using randomized search and cross-validation. This allows the model to balance flexibility and generalization more effectively than a default configuration.

Among the valid models, XGBoost achieves the best overall balance of fit and prediction accuracy.

---

### 4.7 LightGBM

LightGBM is another gradient boosting framework designed for efficiency and speed, especially on structured/tabular datasets.

Like XGBoost, it uses engineered property and location features and is tuned through cross-validation. Its predictive performance is very close to XGBoost, with only slightly lower accuracy in the final comparison.

LightGBM is valuable because it offers:

- fast training,
- strong predictive power,
- good handling of nonlinear patterns.

In this project, it is the closest competitor to XGBoost.

---

## 5. Evaluation Metrics

Model performance is evaluated using four complementary metrics.

### 5.1 \(R^2\) on Log Scale

This metric measures how well the model explains variation in `LogClosePrice`, which is the actual training target.

A higher value indicates better fit in the transformed space.

### 5.2 \(R^2\) on Original Price Scale

This measures how well the model explains variation in actual housing prices after back-transforming predictions.

This is useful because it reflects performance in the scale people care about most: real prices.

### 5.3 MAPE

Mean Absolute Percentage Error is defined as:

\[
\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|
\]

MAPE measures the average relative prediction error in percentage terms. It is easy to interpret and useful in applied settings.

### 5.4 MdAPE

Median Absolute Percentage Error reports the median percentage error rather than the mean.

This makes it more robust to outliers and gives a clearer picture of the typical prediction error for most homes.

Using all four metrics together allows the project to compare models from both statistical and practical perspectives.

---

## 6. Results Summary

The model comparison shows a clear pattern.

### 6.1 Linear Regression
The baseline model is stable and interpretable, but it underfits. Its predictive error is higher than that of the nonlinear models, and it cannot capture the complexity of the housing market.

### 6.2 Decision Tree
The shallow tree underfits and performs poorly. The improved decision tree, however, performs much better once stronger features and location encoding are introduced.

### 6.3 Random Forest
Random forest substantially improves performance and captures nonlinear relationships well. It is one of the strongest non-boosting models in the project, though mild overfitting is visible.

### 6.4 Leakage-Affected Random Forest
This model appears almost perfect, but that performance is not trustworthy because it relies on leakage-prone variables.

### 6.5 XGBoost
XGBoost delivers the strongest overall results among the valid models. It combines high explanatory power with low percentage error and strong generalization to unseen data.

### 6.6 LightGBM
LightGBM performs very similarly to XGBoost and is an excellent alternative. Its accuracy is slightly lower, but the difference is small.

---

## 7. Best Model

The best practical model in this project is **XGBoost**.

It is selected as the final model because it provides the best combination of:

- high \(R^2\),
- low MAPE,
- low MdAPE,
- strong generalization,
- effective handling of nonlinear feature relationships.

The project findings show that XGBoost outperforms the baseline linear regression, the decision tree models, and slightly outperforms LightGBM in the final comparison.

Although a leakage-affected random forest variant produces even stronger numbers, it is not considered a valid model because its predictors reveal too much of the target.

Therefore, among realistic and deployable models, **XGBoost is the best final choice**.

---

## 8. Conclusion

This project shows that accurate house price prediction depends on more than simply choosing a sophisticated algorithm. The strongest results come from the combination of:

- careful preprocessing,
- meaningful feature engineering,
- correct location encoding,
- strong leakage prevention,
- and appropriate model selection.

Several important conclusions emerge from the project:

1. **Preprocessing matters greatly.**  
   Cleaning raw data, transforming the target, and constructing meaningful engineered features are essential for good model performance.

2. **Location is a dominant signal.**  
   Encoding neighborhood information through `ZipMedianPrice` is much more effective than relying only on raw categorical ZIP codes.

3. **Nonlinear models outperform linear models.**  
   Housing prices are driven by complex interactions and threshold effects that linear regression cannot fully capture.

4. **Leakage must be actively controlled.**  
   Features that are too directly tied to the target can make a model look excellent while making it useless in practice.

5. **Boosting models perform best.**  
   Among all valid models, XGBoost provides the strongest balance of accuracy and generalization, with LightGBM as a very close second.

Overall, the final project demonstrates that a well-designed machine learning pipeline can predict California residential prices effectively, provided that preprocessing and evaluation are handled carefully and realistically.