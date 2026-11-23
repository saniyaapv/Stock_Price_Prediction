import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Stock Price Prediction")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_excel(os.path.join(BASE_DIR, "stock_data.xlsx"))

# Display dataset
st.subheader("Historical Stock Market Data")
st.dataframe(data.head(10))

st.success("Dataset loaded successfully!")


# -----------------------------------
# HISTORICAL STOCK PRICE GRAPH
# -----------------------------------

st.header("Historical Stock Price Analysis")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    data["Date"],
    data["Close"],
    color="blue",
    label="Closing Price"
)

ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.set_title("Historical Stock Closing Prices")
ax.legend()
ax.grid(True)

st.pyplot(fig)

plt.close(fig)


# -----------------------------------
# DATA PREPROCESSING
# -----------------------------------

st.header("Data Preprocessing")

# Create a copy of the original dataset
processed_data = data.copy()

# 1. Convert Date into datetime format
processed_data["Date"] = pd.to_datetime(
    processed_data["Date"],
    errors="coerce"
)

# 2. Convert price and volume columns into numeric format
numeric_columns = [
    "Open", "High", "Low", "Close", "Volume"
]

for column in numeric_columns:
    processed_data[column] = pd.to_numeric(
        processed_data[column],
        errors="coerce"
    )

# 3. Remove duplicate records
processed_data = processed_data.drop_duplicates()

# 4. Remove rows with missing important values
processed_data = processed_data.dropna(
    subset=["Date", "Close", "Volume"]
)

# 5. Arrange data by date
processed_data = processed_data.sort_values(
    "Date"
).reset_index(drop=True)

# 6. Create previous closing price columns
for i in range(1, 6):
    processed_data[f"Close_Lag_{i}"] = (
        processed_data["Close"].shift(i)
    )

# 7. Create previous day's volume column
processed_data["Volume_Lag_1"] = (
    processed_data["Volume"].shift(1)
)

# 8. Remove rows without previous-day information
processed_data = processed_data.dropna().reset_index(drop=True)

# Display results
st.success("Data preprocessing completed successfully!")

col1, col2, col3 = st.columns(3)

col1.metric("Original Records", len(data))
col2.metric("Processed Records", len(processed_data))
col3.metric(
    "Removed Records",
    len(data) - len(processed_data)
)

st.subheader("Processed Stock Market Data")

st.dataframe(
    processed_data.head(10),
    use_container_width=True
)

st.subheader("Features Created")

for feature in [
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_4",
    "Close_Lag_5",
    "Volume_Lag_1"
]:
    st.write("✅", feature)

# -----------------------------------
# LINEAR REGRESSION MODEL TRAINING
# -----------------------------------

st.header("Linear Regression Model Training")

# Select input features
features = [
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_4",
    "Close_Lag_5",
    "Volume_Lag_1"
]

# Input and target data
X = processed_data[features]
y = processed_data["Close"]

# Split data chronologically (80% training, 20% testing)
split_index = int(len(processed_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Create and train the model
model = LinearRegression()

model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

st.success("Linear Regression model trained successfully!")

st.write("Training records:", len(X_train))
st.write("Testing records:", len(X_test))

# -----------------------------------
# ACTUAL VS PREDICTED GRAPH
# -----------------------------------

st.header("Actual vs Predicted Stock Prices")

fig, ax = plt.subplots(figsize=(12, 5))

# Actual stock prices
ax.plot(
    processed_data["Date"].iloc[split_index:],
    y_test,
    label="Actual Price",
    color="blue"
)

# Predicted stock prices
ax.plot(
    processed_data["Date"].iloc[split_index:],
    predictions,
    label="Predicted Price",
    color="red"
)

ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")
ax.set_title("Actual vs Predicted Stock Prices")

ax.legend()
ax.grid(True)

st.pyplot(fig)

plt.close(fig)

st.success("Actual vs Predicted graph created successfully!")

# -----------------------------------
# NEXT STOCK PRICE PREDICTION
# -----------------------------------

st.header("Next Trading Session Prediction")

# Get the latest 5 closing prices
latest_closes = processed_data["Close"].tail(5).to_numpy()[::-1]

# Get the latest trading volume
latest_volume = processed_data["Volume"].iloc[-1]

# Prepare input for prediction
next_day_data = pd.DataFrame([{
    "Close_Lag_1": latest_closes[0],
    "Close_Lag_2": latest_closes[1],
    "Close_Lag_3": latest_closes[2],
    "Close_Lag_4": latest_closes[3],
    "Close_Lag_5": latest_closes[4],
    "Volume_Lag_1": latest_volume
}])

# Predict next closing price
next_price = model.predict(next_day_data[features])[0]

st.metric(
    "Predicted Next Closing Price",
    round(next_price, 2)
)

st.success("Next trading session prediction completed!")

# -----------------------------------
# DOWNLOAD PREDICTIONS
# -----------------------------------

st.header("Download Predicted Stock Prices")

# Create a results table
results = pd.DataFrame({
    "Date": processed_data["Date"].iloc[split_index:].values,
    "Actual Closing Price": y_test.values,
    "Predicted Closing Price": predictions
})

# Display prediction results
st.subheader("Prediction Results")
st.dataframe(results, use_container_width=True)

# Convert results into CSV
csv_data = results.to_csv(index=False).encode("utf-8")

# Download button
st.download_button(
    label="Download Predictions as CSV",
    data=csv_data,
    file_name="stock_price_predictions.csv",
    mime="text/csv"
)

st.success("Prediction results are ready to download!")
