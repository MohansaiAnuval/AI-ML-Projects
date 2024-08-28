import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()

# Load the data
apple_products = r'C:\Users\Anuval Mohan sai\Desktop\Project P0\ApplevisionProPrediction(RandomForest)\apple_vision_pro_sales.csv'
df = pd.read_csv(apple_products)

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Promotion'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 12] else 0)
df['Prev_Month_Sales'] = df['Sales'].shift(1)
df = df.dropna()  # Drop rows with NaN values

# Ensure the 'plots' directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Historical Sales Plot with Values
def plot_historical_sales():
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='Month', y='Sales', hue='Year', marker='o')
    plt.title('Apple Vision Pro Sales Over Time')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.xticks(
        ticks=np.arange(1, 13),
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        rotation=45
    )

    # Annotate each point with the sales value
    for year in df['Year'].unique():
        yearly_data = df[df['Year'] == year]
        for x, y in zip(yearly_data['Month'], yearly_data['Sales']):
            plt.text(
                x,
                y,
                f'{y:.0f}',
                ha='center',
                va='bottom',
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )

    plt.tight_layout()
    plt.savefig('plots/sales_over_time.png')
    plt.close()

# Prepare Data for Prediction
X = df[['Year', 'Month', 'Promotion', 'Prev_Month_Sales']]
y = df['Sales']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

# Predict Sales for Next Year
next_year = 2024
future_months = pd.DataFrame({
    'Year': [next_year] * 12,
    'Month': list(range(1, 13)),
    'Promotion': [1 if month in [6, 7, 12] else 0 for month in range(1, 13)],
    'Prev_Month_Sales': [df['Sales'].iloc[-1]] * 12  # Use the last known sales value
})

future_sales = model.predict(future_months)

# Display predicted future sales
future_sales_df = future_months.copy()
future_sales_df['Predicted_Sales'] = future_sales

# Combine historical and future sales data for plotting
df['Type'] = 'Historical'
future_sales_df['Type'] = 'Predicted'
plot_data = pd.concat([df, future_sales_df.rename(columns={'Predicted_Sales': 'Sales'})])

# Future Sales Prediction Plot with Values
def plot_future_sales():
    plt.figure(figsize=(12, 7))
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

    # Annotate historical and predicted values
    for _, row in plot_data.iterrows():
        plt.text(
            row['Month'],
            row['Sales'],
            f'{row["Sales"]:.0f}',
            ha='center',
            va='bottom',
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

    plt.tight_layout()
    plt.savefig('plots/sales_prediction.png')
    plt.close()

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

# Run the app with: uvicorn apple_visionpro_Randomforest:app --reload
