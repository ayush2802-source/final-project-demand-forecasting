import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

# Load the trained model
MODEL_PATH = 'artifacts/demand_model_1.joblib'  # Update with your model path
try:
    model = joblib.load(MODEL_PATH)  # Directly loading the trained model
except Exception as e:
    st.error(f"Error loading model: {e}")

# Set the page configuration and title
st.set_page_config(page_title="Demand Forecasting App", page_icon="📊")
st.title("Demand Forecasting with Random Forest")

# Create rows of three columns each
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)

# User inputs for prediction
with row1[0]:
    store_id = st.number_input("Store ID", min_value=0, step=1, value=1)
with row1[1]:
    sku_id = st.number_input("SKU ID", min_value=0, step=1, value=1001)
with row1[2]:
    total_price = st.number_input("Total Price", value=500.0)

with row2[0]:
    base_price = st.number_input("Base Price", value=450.0)
with row2[1]:
    is_featured_sku = st.selectbox("Is Featured SKU", [0, 1])
with row2[2]:
    is_display_sku = st.selectbox("Is Display SKU", [0, 1])

with row3[0]:
    day = st.number_input("Day", min_value=1, max_value=31, value=15)
with row3[1]:
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
with row3[2]:
    year = st.number_input("Year", value=2025)

# Predict button
if st.button("Predict Units Sold"):
    if 'model' not in globals():
        st.error("Model not loaded. Please check your model path.")
    else:
        try:
            # Preparing user input for model
            user_input = pd.DataFrame({
                'store_id': [store_id],
                'sku_id': [sku_id],
                'total_price': [total_price],
                'base_price': [base_price],
                'is_featured_sku': [is_featured_sku],
                'is_display_sku': [is_display_sku],
                'day': [day],
                'month': [month],
                'year': [year]
            })

            # Align user_input to model feature names
            missing_features = set(model.feature_names_in_) - set(user_input.columns)
            for feature in missing_features:
                user_input[feature] = 0  # Add missing columns with default value

            # Ensure columns are in the same order as the model
            user_input = user_input[model.feature_names_in_]

            # Predict using the model
            prediction = model.predict(user_input)

            # Display result
            st.success(f"Predicted Units Sold: {prediction[0]:.2f}")
        except NotFittedError:
            st.error("The model is not trained. Please ensure the model is fitted before saving.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")