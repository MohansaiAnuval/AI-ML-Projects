from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import io
import os
import base64

app = FastAPI()

# Load the data
apple_products = r'C:\Users\Anuval Mohan sai\Desktop\Project P0\apple_vision_pro_sales.csv'
df = pd.read_csv(apple_products)

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Promotion'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 12] else 0)
df['Prev_Month_Sales'] = df['Sales'].shift(1)
df = df.dropna()  # Drop rows with NaN values

# Prepare Data for Prediction
X = df[['Year', 'Month', 'Promotion', 'Prev_Month_Sales']]
y = df['Sales']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

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

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        # Explicitly set the Matplotlib backend
        plt.switch_backend('agg')
        
        # Generate historical sales plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Month', y='Sales', hue='Year', marker='o')
        plt.title('Apple Vision Pro Sales Over Time')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.xticks(ticks=np.arange(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
        plt.tight_layout()

        # Save historical plot to a BytesIO object
        buf_hist = io.BytesIO()
        plt.savefig(buf_hist, format='png')
        buf_hist.seek(0)
        plt.close()

        # Encode historical plot as base64
        plot_hist_base64 = base64.b64encode(buf_hist.read()).decode('utf-8')

        # Generate future sales prediction plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=plot_data, x='Month', y='Sales', hue='Type', marker='o')
        plt.title('Sales Prediction for Next Year')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.xticks(ticks=np.arange(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
        plt.legend(title='Data Type')
        plt.tight_layout()

        # Save prediction plot to a BytesIO object
        buf_pred = io.BytesIO()
        plt.savefig(buf_pred, format='png')
        buf_pred.seek(0)
        plt.close()

        # Encode prediction plot as base64
        plot_pred_base64 = base64.b64encode(buf_pred.read()).decode('utf-8')

        # Generate HTML table
        html_data = df.to_html(classes='table table-striped', index=False)
        
        # Embed plots in HTML
        html_content = f"""
        <html>
            <head>
                <title>Apple Vision Pro Sales</title>
                <style>
                    .table {{
                        width: 80%;
                        margin: 20px auto;
                        border-collapse: collapse;
                    }}
                    .table th, .table td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                    }}
                    .table th {{
                        background-color: #f4f4f4;
                    }}
                    .table-striped tbody tr:nth-of-type(odd) {{
                        background-color: #f9f9f9;
                    }}
                </style>
            </head>
            <body>
                <h1>Apple Vision Pro Sales Data and Predictions</h1>
                <h2>Historical Sales Data</h2>
                {html_data}
                <h2>Sales Over Time</h2>
                <img src="data:image/png;base64,{plot_hist_base64}" alt="Sales Over Time Plot" />
                <h2>Future Sales Predictions</h2>
                <img src="data:image/png;base64,{plot_pred_base64}" alt="Sales Prediction Plot" />
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating page: {str(e)}")

