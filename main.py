import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. Load data
df = pd.read_csv("data.csv")

# 2. Convert date column
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.set_index('date', inplace=True)

# 3. Print data
print("Data Preview:\n", df.head())

# 4. Plot original data
plt.figure(figsize=(10,5))
plt.plot(df['sales'])
plt.title("Sales Data")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# 5. Build ARIMA model
model = ARIMA(df['sales'], order=(1,1,1))
model_fit = model.fit()

# 6. Forecast next 5 days
forecast = model_fit.forecast(steps=5)

print("\nForecasted Sales:\n", forecast)

# 7. Plot forecast
plt.figure(figsize=(10,5))
plt.plot(df['sales'], label='Actual')
plt.plot(forecast, label='Forecast', color='red')
plt.title("Sales Forecast")
plt.legend()
plt.show()