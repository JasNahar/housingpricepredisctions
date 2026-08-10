# House Price Prediction — Linear Regression

Predicts house price from area-level statistics (income, house age, rooms, bedrooms, population) using Multiple Linear Regression.

## Dataset

- **File:** `USA_Housing.csv`
- **Size:** 5,000 rows, 6 features + target (`Price`)
- **Features:** Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population (Address column dropped — text, not usable)

## Approach

1. EDA — correlation heatmap + scatterplots to check features actually relate linearly to price
2. Train/test split (80/20) + feature scaling
3. Trained `LinearRegression` from scikit-learn
4. Evaluated with MAE, RMSE, R²
5. Interpreted coefficients — which features drive price, and by how much
6. Residual analysis — verified the linearity assumption held (not just accepting the fit blindly)

## Results

- **R² Score:** 0.918 (model explains ~92% of price variation)
- **MAE:** ~$80,879
- **RMSE:** ~$100,444
- **Strongest driver of price:** Avg. Area Income, followed by House Age, Population, and Number of Rooms. Number of Bedrooms had almost no effect.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
jupyter notebook House_Price_Linear_Regression.ipynb
```

## Next Steps

- Try Ridge/Lasso regression and compare
- Add polynomial features if relationships look non-linear
- Move to a dataset with categorical features (e.g. Kaggle's Ames Housing) to practice one-hot encoding
