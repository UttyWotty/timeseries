# timeseries
using several methods to practise time series analysis using example data

- methods that has been used 

### 1. **Importing Libraries and Loading Data**
```python
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

2. Renaming the 'Unnamed' Column
python
Copy code
df.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
Renaming the column: The CSV file has an unnamed column that is renamed to 'Date'.
Converting Date column: The 'Date' column is then converted to a datetime format for proper time series manipulation.

3. Plotting the Closing Price
python
Copy code
df['close_USD'].plot(figsize=(12,6), title = 'Closing price(USD)', ylabel = 'Price')
plt.show()
This plots the closing price in USD over time. It uses the close_USD column from the dataset, and customizes the figure's size and labels.

4. Decomposing the Time Series
python
Copy code
result = seasonal_decompose(df['close_USD'], model='additive', period=30)
result.plot()
plt.show()
Seasonal decomposition: This decomposes the time series data into 3 components: Trend, Seasonality, and Residuals (Noise).
additive model: The additive model assumes that the observed data is the sum of the three components.
Period of 30: Assumes the data repeats in cycles of 30 (e.g., daily data with monthly seasonality).
Components Explanation:

Observed: The original time series.
Trend: The overall direction of the data (upward or downward).
Seasonal: The repeating patterns or cycles in the data.
Residual: The random fluctuations in the data after removing trend and seasonality.

5. Plotting Decomposed Components
python
Copy code
# Breaking down the plot into subplots for each component
plt.subplot(4,1,1)
plt.plot(result.observed, label='Observed', color='blue')
plt.title('Observed')
plt.legend(loc='upper left')

plt.subplot(4,1,2)
plt.plot(result.trend, label='Trend', color='green')
plt.title('Trend')
plt.legend(loc='upper left')

plt.subplot(4,1,3)
plt.plot(result.seasonal, label='Seasonal', color='red')
plt.title('Seasonal')
plt.legend(loc='upper left')

plt.subplot(4,1,4)
plt.plot(result.resid, label='Residual', color='black')
plt.title('Residual')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
This section plots the four decomposed components (Observed, Trend, Seasonal, and Residual) in separate subplots for better clarity.


6. Creating New Columns for Weekly and Monthly Data
python
Copy code
df['Week No.'] = df['Date'].dt.isocalendar().week
df = df[df['Week No.'] != 53]  # Removing incomplete weeks
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
This creates new columns:

Week No.: Extracts the week number from the date.
Month: Extracts the month.
Year: Extracts the year.
7. Plotting Weekly and Monthly Data
python
Copy code
plt.plot(weekly_data['Week No.'], weekly_data['close_SAR'], label='Weekly Close Price(SAR)', color='blue')
plt.plot(weekly_data['Week No.'], weekly_data['close_USD'], label='Weekly Close Price(USD)', color='red')
plt.show()
This section plots the weekly closing prices in SAR and USD, adjusting the x-axis to reflect time correctly.

8. Volume of Transactions per Week
python
Copy code
sns.barplot(x='Week No.', y="volume", data=weekly_data, hue='Year', palette='Oranges', alpha=0.7)
plt.show()
This creates a bar plot of the transaction volume per week, grouped by the year.

9. Monthly Data Analysis
python
Copy code
monthly_data['Month Name'] = monthly_data['Month'].apply(lambda x: pd.to_datetime(f"2020-{x:02d}-01").strftime('%b'))
sns.barplot(x='Month Name', y="volume", data=monthly_data, hue='Year', palette='Blues', alpha=0.9)
plt.show()
Monthly analysis: The code aggregates monthly data for SAR/USD prices and volumes.
Month Name Conversion: Converts month numbers to names (e.g., January to 'Jan') for better readability.
10. Plotting Monthly High USD Prices
python
Copy code
sns.barplot(x='Month Name', y="high_USD", data=monthly_data, hue='Year', palette='Set2', ci=None, dodge=True)
plt.show()
This section plots the average high USD prices per month using a bar plot.

11. Interactive Plotting with Plotly
python
Copy code
fig = px.bar(monthly_data, x='Month Name', y='high_USD', color='Year', barmode='group', labels={'high USD': 'High USD Price', 'Month Name': 'Month'}, title='Monthly High Prices(USD)')
fig.update_traces(texttemplate='%{y:.2f}', textposition='outside', hoverinfo='x + y')
fig.show()
Plotly: This section uses Plotly to create an interactive bar plot of the high USD prices by month.
12. Forecasting High USD Prices with ARIMA
python
Copy code
model = ARIMA(df['high_USD'], order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=100)
plt.plot(df.index, df['high_USD'], label='Historical Data', color='blue')
plt.plot(pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=101, freq='D')[1:], forecast, label='Forecast', color='red')
plt.legend()
plt.show()
ARIMA model: The ARIMA (AutoRegressive Integrated Moving Average) model is used to forecast the next 100 days of high USD prices based on historical data.
13. Anomaly Detection with Z-Score Method
python
Copy code
z_scores = df[columns_to_check].apply(zscore)
outliers = (np.abs(z_scores) > 3)
anomalies = df[outliers.any(axis=1)]
plt.plot(df.index, df['open_USD'], label='Open Price(USD)', color='blue')
plt.scatter(anomalies.index, anomalies['open_USD'], color='red', label='Anomalies')
plt.show()
This detects anomalies using the Z-score method (values with a Z-score greater than 3 are considered outliers).

14. Anomaly Detection with Isolation Forest
python
Copy code
iso_forest = IsolationForest(contamination=0.01)
df['anomaly'] = iso_forest.fit_predict(df[features])
anomalies_iso = df[df['anomaly'] == -1]
plt.plot(df.index, df['high_USD'], label='High Price(USD)', color='blue')
plt.scatter(anomalies_iso.index, anomalies_iso['high_USD'], color='red', label='Anomalies')
plt.show()
Isolation Forest: A machine learning technique used to detect anomalies. It assigns a label of -1 for anomalies and 1 for normal observations.
