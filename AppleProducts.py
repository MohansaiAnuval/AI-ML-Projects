import pandas as pd

# Define the data in dictionary format
data_dict = {
    "Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01", "2023-11-01", "2023-12-01"],
    "Product Name": ["Apple Vision Pro"] * 12,
    "Product URL": ["https://example.com/apple-vision-pro"] * 12,
    "Brand": ["Apple"] * 12,
    "Sale Price": [2999] * 12,
    "Mrp": [3299] * 12,
    "Discount Percentage": [9.1] * 12,
    "Number Of Ratings": [150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205],
    "Number Of Reviews": [30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 55, 58],
    "Upc": ["123456789012"] * 12,
    "Star Rating": [4.5, 4.6, 4.6, 4.7, 4.7, 4.8, 4.8, 4.9, 4.9, 4.9, 5.0, 5.0],
    "Ram": ["12GB"] * 12,
    "Sales": [5000, 5200, 5300, 5500, 5800, 6000, 6200, 6300, 6400, 6500, 6600, 6700]
}

# Create DataFrame from the dictionary
df = pd.DataFrame(data_dict)

# Save DataFrame to CSV file
df.to_csv('Project P0/apple_vision_pro_sales.csv', index=False)

# Display DataFrame
print(df)
