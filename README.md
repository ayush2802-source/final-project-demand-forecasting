# final-project-demand-forecasting
📦 Demand Forecasting for Retail Chain (SKU-Level) A machine learning project that predicts weekly sales demand for thousands of SKUs across 76 retail stores using historical sales, pricing, and promotional data.

🔍 Problem Statement One of the largest retail chains in the world aims to improve its demand planning by forecasting the number of units sold for each SKU-store combination on a weekly basis. With only historical sales, pricing, and promotional data (and no product/store meta info), the goal is to forecast the next 12 weeks accurately.

🧠 Solution Overview This project uses a Random Forest Regressor model to predict future demand at a granular SKU-store-week level. The model was trained on 3 years of historical weekly sales data, including base price, total price, and promotional flags.

✅ Features Used store_id, sku_id

total_price, base_price

is_featured_sku, is_display_sku

Extracted features from week (day, month, year)

One-hot encoding for store_id and sku_id

⚙️ Techniques Applied Data cleaning & preprocessing (outlier removal, date parsing)

Feature engineering (categorical encoding, time components)

Random Forest Regressor with hyperparameter tuning via GridSearchCV

Model evaluation using Root Mean Squared Error (RMSE)

📈 Model Performance Test RMSE: 17 units

Standard Deviation of target (units_sold): ~60 units

Interpretation: RMSE is ~28% of the SD, which indicates solid predictive accuracy in a volatile retail dataset.
