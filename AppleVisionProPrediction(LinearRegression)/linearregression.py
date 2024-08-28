import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI()

# Load the data
apple_products = r'C:\Users\Anuval Mohan sai\Desktop\Project P0\AppleVisionProPrediction(LinearRegression)\apple_vision_pro_sales - Copy.csv'
df = pd.read_csv(apple_products)

# Display the data
print(df.head())

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Use only relevant columns for prediction
df = df[['Year', 'Month', 'Sales']]

# Display processed data
print(df.head())

# Ensure the 'plots' directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Plot historical sales data and display values
def plot_historical_sales():
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Month', y='Sales', hue='Year', marker='o')
    plt.title('Apple Vision Pro Sales Over Time')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.xticks(
        ticks=np.arange(1, 13),
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        rotation=45
    )
    plt.tight_layout()
    plt.savefig('plots/sales_over_time.png')  # Save the plot
    plt.close()
    
    # Print the historical sales data
    print("\n--- Historical Sales Data ---")
    print(df[['Year', 'Month', 'Sales']])

# Prepare data for prediction
X = df[['Year', 'Month']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Predict sales for next year (e.g., 2024)
next_year = 2024
future_months = pd.DataFrame({
    'Year': [next_year] * 12,
    'Month': list(range(1, 13))
})

future_sales = model.predict(future_months)

# Display predicted future sales
future_sales_df = future_months.copy()
future_sales_df['Predicted_Sales'] = future_sales

print(future_sales_df)

# Combine historical and future sales data for plotting
df['Type'] = 'Historical'
future_sales_df['Type'] = 'Predicted'
plot_data = pd.concat([df, future_sales_df.rename(columns={'Predicted_Sales': 'Sales'})])

# Plot future sales prediction and display values
def plot_future_sales():
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data, x='Month', y='Sales', hue='Type', marker='o')
    plt.title('Sales Prediction for Next Year')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.xticks(
        ticks=np.arange(1, 13),
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        rotation=45
    )
    plt.legend(title='Data Type')
    plt.tight_layout()
    plt.savefig('plots/sales_prediction.png')  # Save the plot
    plt.close()
    
    # Print the future sales predictions
    print("\n--- Predicted Sales Data for Next Year ---")
    print(future_sales_df)

# FastAPI endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Apple Vision Pro Sales Prediction API!"}

@app.get("/plot/historical")
def get_historical_plot():
    try:
        plot_historical_sales()
        return FileResponse("plots/sales_over_time.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating historical sales plot: {str(e)}")

@app.get("/plot/prediction")
def get_prediction_plot():
    try:
        plot_future_sales()
        return FileResponse("plots/sales_prediction.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sales prediction plot: {str(e)}")

# To run the app, use: uvicorn <filename_without_extension>:app --reload
# Example: uvicorn apple_vision_pro_sales:app --reload
