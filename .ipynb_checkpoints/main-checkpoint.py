import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Salary API"}

# Load data once when the server starts
try:
    data = r"C:\Users\Anuval Mohan sai\Desktop\Project P0\Employees.csv"
    df = pd.read_csv(data)
    print(df.head())  # Check the first few rows of the DataFrame
except FileNotFoundError:
    df = pd.DataFrame()  # Create an empty DataFrame if the file is not found
    print("File not found. DataFrame is empty.")


# Print the columns of the DataFrame
print("Columns in the DataFrame:", df.columns.tolist())

# Data Preprocessing
def preprocess_data(df):
    # Drop rows with missing target values
    df = df.dropna(subset=['Monthly Salary'])
    
    # Ensure 'Monthly Salary' is numeric
    df['Monthly Salary'] = pd.to_numeric(df['Monthly Salary'], errors='coerce')

    # Encode categorical features
    df = pd.get_dummies(df, columns=['Gender', 'Department', 'Country', 'Center'], drop_first=True)

    # Feature and target selection
    X = df.drop(columns=['No', 'First Name', 'Last Name', 'Start Date', 'Monthly Salary', 'Annual Salary'])
    y = df['Monthly Salary']
    
    return X, y

# Train model when the server starts
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

@app.get("/salary_stats")
def salary_statistics():
    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")
    # Rest of the code remains unchanged

@app.get("/predict_salary")
def predict_salary(years: int, overtime_hours: int, gender: str, department: str, country: str, center: str):
    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")

    # Encode input data
    input_data = pd.DataFrame({
        'Years': [years],
        'Overtime Hours': [overtime_hours],
        'Gender_' + gender: [1],
        'Department_' + department: [1],
        'Country_' + country: [1],
        'Center_' + center: [1]
    })
    
    # Align with training data (to handle any missing columns)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Predict salary
    predicted_salary = model.predict(input_data)
    
    return {"predicted_salary": predicted_salary[0], "model_mse": mse}
