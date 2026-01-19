# 🏠 Ames Housing Price Prediction

### 🏆 Kaggle Competition Project

**Score:** 0.130 RMSLE (Top 35% — 1636 / 4769)

**Tech Stack:** Python, Scikit-Learn, Pandas, Gradient Boosting

## 📖 Overview

This project is my solution for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition. The goal was to predict final house prices in Ames, Iowa, based on 79 explanatory variables describing nearly every aspect of residential homes.

This was my first end-to-end machine learning project.

## 🔧 The Challenge

The dataset presents several challenges typical of real-world data:

* **High Dimensionality:** 79 initial features, expanding to 200+ after one-hot encoding
* **Missing Data:** Significant gaps in features like `LotFrontage` (17%), `GarageYrBlt` (5%), and `MasVnrArea`
* **Mixed Feature Types:** A combination of numerical, ordinal, and nominal categorical variables requiring different encoding strategies

## 🛠️ My Approach

### 1. Data Cleaning & Preprocessing

* **Domain-Specific Imputation:** Handled missing values logically rather than blindly using the mean:
  - `LotFrontage` → Filled with neighborhood median (similar houses have similar lot sizes)
  - `GarageYrBlt` → Set to 0 when no garage exists
  - `PoolQC`, `Fence`, `Alley` → NaN means "None", not missing data
* **Production Safety:** Implemented a "Safety Net" pipeline with `.fillna(0)` after all mappings to handle edge cases in test data

### 2. Feature Engineering

* **TotalSF:** Combined `1stFlrSF + 2ndFlrSF + TotalBsmtSF` to capture true living space (correlation: 0.78)
* **TotalBath:** Aggregated all bathroom types with half-baths weighted at 0.5 (correlation: 0.63)
* **HouseAge:** Calculated `YrSold - YearBuilt` to capture depreciation effects (correlation: -0.52)
* **Ordinal Encoding:** Preserved rank order for quality variables (Ex→5, Gd→4, TA→3, Fa→2, Po→1) across 10+ columns

### 3. Model Selection

Compared multiple models on the same train/test split:

| Model | RMSE |
|-------|------|
| Linear Regression | $43,031 |
| Ridge (α=100) | $30,219 |
| Random Forest | $26,423 |
| **Gradient Boosting** | **$23,790** |

**Final Model:** Gradient Boosting Regressor with default hyperparameters

## 📈 Results & Iteration

My score progression on Kaggle:

| Submission | Score (RMSLE) | What Changed |
|------------|---------------|--------------|
| #1 | 0.156 | Initial submission |
| #2 | 11.95 | Broke preprocessing (lesson learned!) |
| #3 | **0.130** | Fixed encoding + removed Id from features |

**Key Debugging Insights:**
- Accidentally included `Id` as a feature — the model tried to learn patterns from row numbers
- Forgot to one-hot encode test data — model received zeros for all categorical features
- Missing `.fillna(0)` after `.map()` caused silent NaN propagation

## 🚫 What Didn't Work

* **Log-Transform on Target:** Tried `np.log1p(SalePrice)` — no improvement (0.136)
* **Outlier Removal:** Removed houses >4000 sqft with low prices — made score worse (0.138)

Not every best practice works for every dataset. The model was already handling these cases well.

## 💡 Lessons Learned

1. **Train and test data must undergo identical preprocessing** — this caused my biggest bugs
2. **Sanity check predictions** before submitting (should be in $50k-$600k range, not negative or millions)
3. **Run notebooks top-to-bottom** after changes — out-of-order execution causes subtle bugs
4. **Domain knowledge matters** — knowing that NaN in `PoolQC` means "no pool" beats blind imputation

## 📁 Project Structure

```
├── House_Prices_Regression.ipynb   # Main notebook with full analysis
├── train.csv                        # Training data (from Kaggle)
├── test.csv                         # Test data (from Kaggle)
├── submission.csv                   # Final predictions
└── README.md
```

## 🔮 Future Improvements

* Use cross-validation instead of a single train/test split
* Try XGBoost or LightGBM for potentially better performance
* Build a sklearn Pipeline to enforce consistent preprocessing
* Tune hyperparameters with GridSearchCV

## 💻 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/omarovski-27/ames-housing-prediction.git
   ```
2. Download data from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook House_Prices_Regression.ipynb
   ```

## 🔗 Links

* [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
* [Dataset Documentation](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
