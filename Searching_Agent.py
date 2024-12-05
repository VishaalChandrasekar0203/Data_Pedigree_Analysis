import pandas as pd
import numpy as np
import os
import requests
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(data_path, tracking_id):
    """
    Analyzes the uploaded data and provides comprehensive insights.
    """
    if not os.path.exists(data_path):
        print("Data file does not exist.")
        return
    else:
        print("Data file found. Analyzing...")

    # Load data into a pandas dataframe
    df = pd.read_csv(data_path)

    # Data cleaning and type conversion
    numeric_columns = ['tmax', 'tmin', 'tavg', 'departure', 'HDD', 'CDD', 'precipitation', 'new_snow', 'snow_depth']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace('T', '0.0'), errors='coerce')

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')

    # Basic insights
    print("\n--- Basic Insights ---")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)

    # Advanced insights
    print("\n--- Advanced Insights ---")
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Correlation analysis
    print("\nCorrelation Matrix:")
    corr_matrix = df.corr()
    print(corr_matrix)

    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Time series analysis
    print("\nTime Series Analysis:")
    df.set_index('date', inplace=True)
    monthly_avg = df['tavg'].resample('M').mean()
    print("Monthly Average Temperatures:")
    print(monthly_avg)

    # Visualize time series
    plt.figure(figsize=(12, 6))
    monthly_avg.plot()
    plt.title('Monthly Average Temperature')
    plt.xlabel('Date')
    plt.ylabel('Average Temperature')
    plt.savefig('monthly_avg_temp.png')
    plt.close()

    # Statistical tests
    print("\nStatistical Tests:")
    summer = df.loc[(df.index.month >= 6) & (df.index.month <= 8), 'tavg']
    winter = df.loc[(df.index.month <= 2) | (df.index.month == 12), 'tavg']
    t_stat, p_value = stats.ttest_ind(summer, winter)
    print(f"T-test between summer and winter temperatures: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    # Data tracking (simplified for demonstration)
    print("\n--- Data Tracking ---")
    payload = {
        "v": "1",
        "tid": tracking_id,
        "cid": "555",
        "t": "event",
        "ec": "data_analysis",
        "ea": "completed",
        "el": f"Analyzed {len(df)} records",
        "ev": 1
    }
    response = requests.post("https://www.google-analytics.com/collect", data=payload)
    
    if response.status_code == 200:
        status_message = "Success"
        details_message = f"Data on {len(df)} records was successfully tracked to Google Analytics with Tracking ID: {tracking_id}"
        location_message = f"Data sent to: Google Analytics"
        website_link_message = f"Tracker Website: Google Analytics (https://www.google-analytics.com)"
        
        print(f"Data tracking status: {status_message}")
        print(details_message)
        print(location_message)
        print(website_link_message)

    else:
        print("tracking failed")


# Example usage
analyze_data("/Users/vishaalchandrasekar/Desktop/nyc_temp.csv", "UA-123456789-1")