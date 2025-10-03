
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime

# Load the dataset
file_path = 'Nat_Gas.csv'  # Change this to your file path if needed
gas_data = pd.read_csv(file_path)

# Convert 'Dates' column to datetime format
gas_data['Dates'] = pd.to_datetime(gas_data['Dates'])

# Plotting the historical price data
plt.figure(figsize=(12, 6))
plt.plot(gas_data['Dates'], gas_data['Prices'], marker='o', linestyle='-')
plt.title('Natural Gas Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# Prepare data for interpolation
dates_num = gas_data['Dates'].map(pd.Timestamp.toordinal)
prices = gas_data['Prices']

# Create interpolation function
interp_func = interp1d(dates_num, prices, kind='linear', fill_value="extrapolate")

# Function to estimate price for any given date
def estimate_gas_price(input_date_str):
    input_date = pd.to_datetime(input_date_str)
    max_date = gas_data['Dates'].max() + pd.DateOffset(years=1)
    if input_date > max_date:
        return "Date is beyond the extrapolation limit of one year."
    price_estimate = float(interp_func(input_date.toordinal()))
    return round(price_estimate, 2)

# Example test cases
test_dates = ['2021-06-30', '2024-12-31', '2025-10-01']
print("Natural Gas Price Estimates:")
for date in test_dates:
    print(f"{date}: {estimate_gas_price(date)}")

# Note for documentation
note = """
Note: This role demands a strong foundation in data analysis and machine learning. Python is a critical tool, 
extensively used at JPMorgan Chase—especially in quantitative research—to perform advanced computations, 
analyze vast datasets, and build robust predictive models for informed decision-making.
"""
print(note)
