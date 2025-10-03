import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Nat_Gas.csv'  # Change this to your file path if needed
gas_data = pd.read_csv(file_path)
gas_data['Dates'] = pd.to_datetime(gas_data['Dates'])

# Prepare interpolation function for prices
dates_num = gas_data['Dates'].map(pd.Timestamp.toordinal)
prices = gas_data['Prices']
interp_func = interp1d(dates_num, prices, kind='linear', fill_value="extrapolate")

# Function to estimate price at any date
def estimate_gas_price(input_date_str):
    input_date = pd.to_datetime(input_date_str)
    price_estimate = float(interp_func(input_date.toordinal()))
    return round(price_estimate, 2)

# Contract pricing function
def price_contract(injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, max_volume, storage_cost):
    volume = 0.0
    total_value = 0.0
    
    # Combine all dates and sort
    all_dates = sorted(set(injection_dates + withdrawal_dates))
    
    for date in all_dates:
        price = estimate_gas_price(date)
        
        # Handle injection
        if date in injection_dates:
            inject_amount = min(injection_rate, max_volume - volume)
            cost = inject_amount * price
            volume += inject_amount
            total_value -= cost  # Buying gas costs money
            print(f"{date}: Inject {inject_amount} units at price {price} -> Cost = {cost}")
        
        # Handle withdrawal
        if date in withdrawal_dates:
            withdraw_amount = min(withdrawal_rate, volume)
            revenue = withdraw_amount * price
            volume -= withdraw_amount
            total_value += revenue  # Selling gas brings money
            print(f"{date}: Withdraw {withdraw_amount} units at price {price} -> Revenue = {revenue}")
        
        # Storage cost for this period (if volume > 0)
        if volume > 0:
            cost = volume * storage_cost
            total_value -= cost
            print(f"{date}: Storage cost for {volume} units -> {cost}")
    
    return round(total_value, 2)

# Example test case
if __name__ == "__main__":
    injection_dates = ['2024-06-30', '2024-07-31']
    withdrawal_dates = ['2024-08-31', '2024-09-30']
    injection_rate = 50
    withdrawal_rate = 40
    max_volume = 100
    storage_cost = 0.05

    contract_value = price_contract(injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, max_volume, storage_cost)
    print("\nTotal Contract Value:", contract_value)
