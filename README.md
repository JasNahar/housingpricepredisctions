House Price Prediction — Linear Regression

Predicts house price from area-level statistics (income, house age, rooms, bedrooms, population) using Multiple Linear Regression — trained, served, and deployed as a live web app.

Live demo: https://marvelous-seahorse-c632dc.netlify.app API: https://housingpricepredisctions.onrender.com

Dataset
File: USA_Housing.csv
Size: 5,000 rows, 6 features + target (Price)
Features: Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population (Address column dropped — text, not usable)
Approach
EDA — correlation heatmap + scatterplots to check features actually relate linearly to price
Train/test split (80/20) + feature scaling
Trained LinearRegression from scikit-learn
Evaluated with MAE, RMSE, R²
Interpreted coefficients — which features drive price, and by how much
Residual analysis — verified the linearity assumption held (not just accepting the fit blindly)
Results
R² Score: 0.918 (model explains ~92% of price variation)
MAE: ~$80,879
RMSE: ~$100,444
Strongest driver of price: Avg. Area Income, followed by House Age, Population, and Number of Rooms. Number of Bedrooms had almost no effect.
From notebook to production

The notebook (House_Price_Linear_Regression.ipynb) covers the modelling work above. To turn it into something more than a static analysis, the same pipeline was rebuilt as a small full-stack app:

USA_Housing.csv → Flask API (trains on startup) → static frontend (fetches live coefficients + metrics)
app.py — a Flask API that loads USA_Housing.csv, retrains the model on startup (same steps as the notebook), and exposes:
GET /model-info — coefficients, feature ranges, and R²/MAE/RMSE, computed fresh
POST /predict — real-time prediction from the trained model
POST /retrain — reload the CSV and retrain without restarting the server
index.html — a static frontend with no hardcoded numbers. On load, it fetches /model-info and builds its sliders, impact chart, and metrics from whatever the API currently reports — so updating the dataset and redeploying the backend is the only step needed to update the live site.

Deployed on Render (API) and Netlify (frontend).

Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Flask, Flask-CORS, gunicorn, HTML/CSS/JavaScript

