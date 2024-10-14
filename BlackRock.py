#!/usr/bin/env python
# coding: utf-8

# In[24]:


get_ipython().system('pip install xgboost')


# In[26]:


get_ipython().system('pip install lightgbm')


# In[3]:


get_ipython().system('pip install yfinance')


# In[ ]:





# In[5]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# Fetch BlackRock stock data
blk = yf.Ticker("BLK")

# Get stock info
info = blk.info

# Fetch historical data for the past year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
hist = blk.history(start=start_date, end=end_date)

# Calculate daily returns
hist['Return'] = hist['Close'].pct_change()

# Calculate volatility (30-day rolling standard deviation of returns)
hist['Volatility'] = hist['Return'].rolling(window=30).std() * (252**0.5)  # Annualized

# Fetch financial data
financials = blk.financials
balance_sheet = blk.balance_sheet
cash_flow = blk.cashflow

# Calculate key financial ratios
current_price = info.get('currentPrice', None)
earnings_per_share = info.get('trailingEps', None)
price_to_earnings = current_price / earnings_per_share if current_price and earnings_per_share else None
price_to_book = None
if current_price and 'Total Stockholder Equity' in balance_sheet.index:
    total_equity = balance_sheet.loc['Total Stockholder Equity'][0]
    shares_outstanding = info.get('sharesOutstanding', None)
    if total_equity and shares_outstanding:
        price_to_book = current_price / (total_equity / shares_outstanding)
dividend_yield = info.get('dividendYield', None)

# Generate insights
print("BlackRock (BLK) Investment and Audit Analysis")
print("-----------------------------------------")
print(f"Current Stock Price: ${current_price:.2f}" if current_price else "Current Stock Price: Not available")
print(f"P/E Ratio: {price_to_earnings:.2f}" if price_to_earnings else "P/E Ratio: Not available")
print(f"P/B Ratio: {price_to_book:.2f}" if price_to_book else "P/B Ratio: Not available")
print(f"Dividend Yield: {dividend_yield:.2%}" if dividend_yield else "Dividend Yield: Not available")
print(f"1-Year Return: {((hist['Close'][-1] / hist['Close'][0]) - 1):.2%}" if not hist.empty else "1-Year Return: Not available")
print(f"Current Volatility: {hist['Volatility'][-1]:.2%}" if not hist.empty else "Current Volatility: Not available")

print("\nKey Financial Metrics (in millions USD):")
print(f"Revenue: ${financials.loc['Total Revenue'][0] / 1e6:.2f}M" if 'Total Revenue' in financials.index else "Revenue: Not available")
print(f"Net Income: ${financials.loc['Net Income'][0] / 1e6:.2f}M" if 'Net Income' in financials.index else "Net Income: Not available")
print(f"Total Assets: ${balance_sheet.loc['Total Assets'][0] / 1e6:.2f}M" if 'Total Assets' in balance_sheet.index else "Total Assets: Not available")
print(f"Total Liabilities: ${balance_sheet.loc['Total Liabilities Net Minority Interest'][0] / 1e6:.2f}M" if 'Total Liabilities Net Minority Interest' in balance_sheet.index else "Total Liabilities: Not available")

# Visualize stock price and volatility
if not hist.empty:
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(hist.index, hist['Close'], label='Stock Price', color='blue')
    ax2.plot(hist.index, hist['Volatility'], label='Volatility', color='red')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price ($)', color='blue')
    ax2.set_ylabel('Volatility', color='red')

    plt.title('BlackRock Stock Price and Volatility (1 Year)')
    plt.tight_layout()
    plt.show()
else:
    print("Unable to generate stock price and volatility chart due to insufficient data.")

# Audit considerations
print("\nAudit Considerations:")
print("1. Revenue Recognition: Ensure proper timing and measurement of management fees and performance fees.")
print("2. Valuation of Investments: Verify fair value measurements, especially for Level 3 assets.")
print("3. Regulatory Compliance: Review adherence to SEC regulations and other applicable laws.")
print("4. Internal Controls: Assess the effectiveness of internal controls over financial reporting.")
print("5. Related Party Transactions: Examine any transactions with related entities for proper disclosure and arm's length terms.")

# Risk analysis
print("\nRisk Analysis:")
if not hist.empty:
    print(f"1. Market Risk: Stock shows a volatility of {hist['Volatility'][-1]:.2%}, indicating moderate market risk.")
else:
    print("1. Market Risk: Unable to calculate volatility due to insufficient data.")

if 'Total Liabilities Net Minority Interest' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
    debt_to_equity = balance_sheet.loc['Total Liabilities Net Minority Interest'][0] / balance_sheet.loc['Total Stockholder Equity'][0]
    print(f"2. Financial Leverage: Debt-to-Equity ratio is {debt_to_equity:.2f}.")
else:
    print("2. Financial Leverage: Unable to calculate Debt-to-Equity ratio due to missing data.")

print("3. Regulatory Risk: As a major financial institution, BlackRock faces ongoing regulatory scrutiny and potential changes in financial regulations.")
print("4. Reputational Risk: Given its size and influence, any misconduct or perceived impropriety could significantly impact BlackRock's reputation and business.")
print("5. Operational Risk: Ensure robust systems and processes are in place to manage the vast amount of assets under management.")

# Investment thesis
print("\nInvestment Thesis:")
print("1. Market Leadership: BlackRock is the world's largest asset manager, providing economies of scale and a strong competitive position.")
print("2. Diverse Product Offering: From ETFs to alternative investments, BlackRock's wide range of products helps to mitigate risk and capture various market opportunities.")
if 'Net Income' in financials.index and 'Total Revenue' in financials.index:
    net_income_margin = financials.loc['Net Income'][0] / financials.loc['Total Revenue'][0]
    print(f"3. Profitability: With a net income margin of {net_income_margin:.2%}, BlackRock demonstrates strong profitability.")
else:
    print("3. Profitability: Unable to calculate net income margin due to missing data.")
print("4. Technology Edge: The Aladdin platform provides a technological advantage in risk management and portfolio construction.")
if dividend_yield:
    print(f"5. Shareholder Returns: A dividend yield of {dividend_yield:.2%} provides steady income for investors.")
else:
    print("5. Shareholder Returns: Dividend yield information not available.")

print("\nNote: This analysis is based on publicly available data and should be supplemented with thorough due diligence and professional advice before making investment decisions.")
print("\nCaution: Some data points were not available, which may affect the completeness of this analysis.")


# In[6]:


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist, stock.info

def calculate_metrics(df):
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)
    return df

def predict_future_price(df, days_to_predict=30):
    df['Prediction'] = df['Close'].shift(-1)
    df = df.dropna()
    X = np.array(df.index.astype(int).values).reshape(-1, 1)
    y = df['Prediction'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict)
    future_X = np.array(future_dates.astype(int)).reshape(-1, 1)
    
    future_prices = model.predict(future_X)
    
    return future_dates, future_prices

def analyze_blackrock():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data
    
    hist, info = fetch_stock_data("BLK", start_date, end_date)
    hist = calculate_metrics(hist)
    
    future_dates, future_prices = predict_future_price(hist)
    
    # Print analysis
    print(f"BlackRock (BLK) Analysis")
    print(f"Current Price: ${hist['Close'][-1]:.2f}")
    print(f"52-Week High: ${hist['Close'].tail(252).max():.2f}")
    print(f"52-Week Low: ${hist['Close'].tail(252).min():.2f}")
    print(f"Current Volatility: {hist['Volatility'][-1]*100:.2f}%")
    print(f"Average Volume (3 months): {hist['Volume'].tail(63).mean():.0f}")
    
    if 'trailingPE' in info:
        print(f"P/E Ratio: {info['trailingPE']:.2f}")
    if 'dividendYield' in info:
        print(f"Dividend Yield: {info['dividendYield']*100:.2f}%")
    if 'bookValue' in info:
        print(f"Price to Book Ratio: {hist['Close'][-1]/info['bookValue']:.2f}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index, hist['Close'], label='Actual Price')
    plt.plot(future_dates, future_prices, label='Predicted Price', linestyle='--')
    plt.title('BlackRock (BLK) Stock Price - Actual and Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.show()
    
    # Print future price prediction
    print(f"\nPredicted price after 30 days: ${future_prices[-1]:.2f}")
    print("\nNote: This is a simple linear prediction and should not be used as financial advice.")
    
    # Aladdin analysis (hypothetical data)
    print("\nAladdin Platform Analysis:")
    print("Assets on Platform: $21 trillion")
    print("Estimated Annual Revenue: $1 billion")
    print("YoY Growth: 15%")
    
    # Market risks
    print("\nMarket Risks:")
    print(f"Beta (5Y monthly): {info.get('beta', 'N/A')}")
    print(f"Volatility (Annual): {hist['Volatility'].mean()*100:.2f}%")

if __name__ == "__main__":
    analyze_blackrock()


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Simulate Aladdin data
def simulate_aladdin_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Assets under management (AUM) on Aladdin platform
    initial_aum = 21e12  # $21 trillion
    aum = initial_aum * (1 + np.cumsum(np.random.normal(0.00015, 0.001, len(date_range))))
    
    # Number of investment professionals using Aladdin
    initial_users = 55000
    users = initial_users * (1 + np.cumsum(np.random.normal(0.0001, 0.0005, len(date_range))))
    
    # Aladdin revenue (estimated)
    initial_revenue = 1e9  # $1 billion annually
    daily_revenue = initial_revenue / 365
    revenue = daily_revenue * (1 + np.cumsum(np.random.normal(0.0002, 0.002, len(date_range))))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'AUM': aum,
        'Users': users,
        'Revenue': revenue
    })
    
    return df

# Analyze Aladdin data
def analyze_aladdin(df):
    print("Aladdin Platform Analysis")
    print("--------------------------")
    print(f"Analysis Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Current AUM on Platform: ${df['AUM'].iloc[-1]/1e12:.2f} trillion")
    print(f"AUM Growth: {((df['AUM'].iloc[-1] / df['AUM'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Current Users: {df['Users'].iloc[-1]:,.0f}")
    print(f"User Growth: {((df['Users'].iloc[-1] / df['Users'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Estimated Annual Revenue: ${df['Revenue'].iloc[-1]*365/1e9:.2f} billion")
    print(f"Revenue Growth: {((df['Revenue'].iloc[-1] / df['Revenue'].iloc[0]) - 1) * 100:.2f}%")
    
    # Calculate correlations
    correlations = df[['AUM', 'Users', 'Revenue']].corr()
    print("\nCorrelations:")
    print(correlations)
    
    return correlations

# Visualize Aladdin data
def visualize_aladdin(df, correlations):
    # Plot AUM, Users, and Revenue over time
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    ax1.plot(df['Date'], df['AUM']/1e12)
    ax1.set_title('Assets Under Management on Aladdin')
    ax1.set_ylabel('AUM (Trillion $)')
    
    ax2.plot(df['Date'], df['Users'])
    ax2.set_title('Investment Professionals Using Aladdin')
    ax2.set_ylabel('Number of Users')
    
    ax3.plot(df['Date'], df['Revenue']*365/1e9)
    ax3.set_title('Estimated Annual Aladdin Revenue')
    ax3.set_ylabel('Revenue (Billion $)')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap: AUM, Users, and Revenue')
    plt.show()

# Predict future growth
def predict_growth(df, years=5):
    # Calculate average daily growth rates
    aum_growth = (df['AUM'].iloc[-1] / df['AUM'].iloc[0]) ** (1/len(df)) - 1
    users_growth = (df['Users'].iloc[-1] / df['Users'].iloc[0]) ** (1/len(df)) - 1
    revenue_growth = (df['Revenue'].iloc[-1] / df['Revenue'].iloc[0]) ** (1/len(df)) - 1
    
    # Predict future values
    future_aum = df['AUM'].iloc[-1] * (1 + aum_growth) ** (365 * years)
    future_users = df['Users'].iloc[-1] * (1 + users_growth) ** (365 * years)
    future_revenue = df['Revenue'].iloc[-1] * (1 + revenue_growth) ** (365 * years) * 365
    
    print(f"\nGrowth Predictions for {years} years:")
    print(f"Projected AUM: ${future_aum/1e12:.2f} trillion")
    print(f"Projected Users: {future_users:,.0f}")
    print(f"Projected Annual Revenue: ${future_revenue/1e9:.2f} billion")

# Main analysis function
def main_aladdin_analysis():
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    df = simulate_aladdin_data(start_date, end_date)
    correlations = analyze_aladdin(df)
    visualize_aladdin(df, correlations)
    predict_growth(df)

if __name__ == "__main__":
    main_aladdin_analysis()


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Simulate Aladdin data based on the provided results
def simulate_aladdin_data():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 10, 10)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    initial_aum = 21e12  # $21 trillion
    final_aum = 27.79e12  # $27.79 trillion
    initial_users = 55000
    final_users = 63535
    initial_revenue = 1e9  # $1 billion annually
    final_revenue = 1.29e9  # $1.29 billion annually

    aum = np.linspace(initial_aum, final_aum, len(date_range))
    users = np.linspace(initial_users, final_users, len(date_range))
    revenue = np.linspace(initial_revenue/365, final_revenue/365, len(date_range))  # Daily revenue

    df = pd.DataFrame({
        'Date': date_range,
        'AUM': aum,
        'Users': users,
        'Revenue': revenue
    })
    
    return df

def analyze_aladdin_data(df):
    print("Aladdin Platform Analysis")
    print("--------------------------")
    print(f"Analysis Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Current AUM on Platform: ${df['AUM'].iloc[-1]/1e12:.2f} trillion")
    print(f"AUM Growth: {((df['AUM'].iloc[-1] / df['AUM'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Current Users: {df['Users'].iloc[-1]:,.0f}")
    print(f"User Growth: {((df['Users'].iloc[-1] / df['Users'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Estimated Annual Revenue: ${df['Revenue'].iloc[-1]*365/1e9:.2f} billion")
    print(f"Revenue Growth: {((df['Revenue'].iloc[-1] / df['Revenue'].iloc[0]) - 1) * 100:.2f}%")
    
    correlations = df[['AUM', 'Users', 'Revenue']].corr()
    print("\nCorrelations:")
    print(correlations)
    
    return correlations

def visualize_aladdin_trends(df, correlations):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # AUM Trend
    axes[0, 0].plot(df['Date'], df['AUM']/1e12)
    axes[0, 0].set_title('Assets Under Management Trend')
    axes[0, 0].set_ylabel('AUM (Trillion $)')

    # Users Trend
    axes[0, 1].plot(df['Date'], df['Users'])
    axes[0, 1].set_title('Aladdin Users Trend')
    axes[0, 1].set_ylabel('Number of Users')

    # Revenue Trend
    axes[1, 0].plot(df['Date'], df['Revenue']*365/1e9)
    axes[1, 0].set_title('Estimated Annual Revenue Trend')
    axes[1, 0].set_ylabel('Revenue (Billion $)')

    # Correlation Heatmap
    sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.show()

def project_future_growth(df, years=5):
    last_date = df['Date'].iloc[-1]
    projection_end = last_date + timedelta(days=365*years)
    projection_range = pd.date_range(start=last_date, end=projection_end, freq='D')
    
    aum_growth_rate = (df['AUM'].iloc[-1] / df['AUM'].iloc[0]) ** (1/len(df)) - 1
    users_growth_rate = (df['Users'].iloc[-1] / df['Users'].iloc[0]) ** (1/len(df)) - 1
    revenue_growth_rate = (df['Revenue'].iloc[-1] / df['Revenue'].iloc[0]) ** (1/len(df)) - 1
    
    projected_aum = df['AUM'].iloc[-1] * (1 + aum_growth_rate) ** np.arange(len(projection_range))
    projected_users = df['Users'].iloc[-1] * (1 + users_growth_rate) ** np.arange(len(projection_range))
    projected_revenue = df['Revenue'].iloc[-1] * (1 + revenue_growth_rate) ** np.arange(len(projection_range))
    
    projection_df = pd.DataFrame({
        'Date': projection_range,
        'AUM': projected_aum,
        'Users': projected_users,
        'Revenue': projected_revenue
    })
    
    return projection_df

def visualize_projections(df, projection_df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    
    # AUM Projection
    axes[0].plot(df['Date'], df['AUM']/1e12, label='Historical')
    axes[0].plot(projection_df['Date'], projection_df['AUM']/1e12, label='Projected')
    axes[0].set_title('AUM: Historical and Projected')
    axes[0].set_ylabel('AUM (Trillion $)')
    axes[0].legend()

    # Users Projection
    axes[1].plot(df['Date'], df['Users'], label='Historical')
    axes[1].plot(projection_df['Date'], projection_df['Users'], label='Projected')
    axes[1].set_title('Users: Historical and Projected')
    axes[1].set_ylabel('Number of Users')
    axes[1].legend()

    # Revenue Projection
    axes[2].plot(df['Date'], df['Revenue']*365/1e9, label='Historical')
    axes[2].plot(projection_df['Date'], projection_df['Revenue']*365/1e9, label='Projected')
    axes[2].set_title('Annual Revenue: Historical and Projected')
    axes[2].set_ylabel('Revenue (Billion $)')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def main():
    df = simulate_aladdin_data()
    correlations = analyze_aladdin_data(df)
    visualize_aladdin_trends(df, correlations)
    
    projection_df = project_future_growth(df)
    visualize_projections(df, projection_df)
    
    print("\nProjected Values for 5 Years:")
    print(f"AUM: ${projection_df['AUM'].iloc[-1]/1e12:.2f} trillion")
    print(f"Users: {projection_df['Users'].iloc[-1]:,.0f}")
    print(f"Annual Revenue: ${projection_df['Revenue'].iloc[-1]*365/1e9:.2f} billion")

if __name__ == "__main__":
    main()


# In[9]:


def calculate_aum_growth_rate(initial_aum, final_aum, time_period_years):
    growth_rate = (final_aum / initial_aum) ** (1 / time_period_years) - 1
    return growth_rate

# Example usage
initial_aum = 21e12  # $21 trillion
final_aum = 27.79e12  # $27.79 trillion
time_period = 4.75  # years

aum_growth_rate = calculate_aum_growth_rate(initial_aum, final_aum, time_period)
print(f"AUM Annual Growth Rate: {aum_growth_rate:.2%}")


# In[10]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Close'].pct_change()
    data['Target'] = data['Returns'].shift(-1)  # Next day's return
    return data.dropna()

def create_features(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    return data.dropna()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def predict_best_times(model, data, threshold=0.01):
    predictions = model.predict(data)
    best_times = data[predictions > threshold]
    return best_times

def plot_results(data, best_times):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Stock Price')
    plt.scatter(best_times.index, data.loc[best_times.index, 'Close'], color='red', label='Buy Signal')
    plt.title('Stock Price with Buy Signals')
    plt.legend()
    plt.show()

# Main execution
ticker = "BLK"  # BlackRock stock
start_date = "2010-01-01"
end_date = "2024-10-10"

data = fetch_data(ticker, start_date, end_date)
data = create_features(data)

features = ['MA5', 'MA20', 'RSI', 'Volatility', 'Returns']
target = 'Target'

model, X_test, y_test = train_model(data[features], data[target])
mse = mean_squared_error(y_test, model.predict(X_test))
print(f"Model MSE: {mse}")

best_times = predict_best_times(model, data[features])
print("Best times to invest:")
print(best_times.index)

plot_results(data, best_times)

# Predict next 30 days
last_30_days = data[features].tail(30)
next_30_days_pred = model.predict(last_30_days)
print("\nPredicted returns for next 30 days:")
for date, pred in zip(last_30_days.index, next_30_days_pred):
    print(f"{date.date()}: {pred:.2%}")


# In[ ]:





# In[12]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Close'].pct_change()
    return data.dropna()

def create_features(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    return data.dropna()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(data):
    features = ['MA5', 'MA20', 'RSI', 'Volatility', 'Returns']
    target = 'Returns'
    
    X = data[features].fillna(method='ffill')
    y = data[target].shift(-1).fillna(method='ffill')  # Next day's return
    X = X[:-1]  # Remove last row
    y = y[:-1]  # Remove last row
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, data, days=30):
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    future_df = pd.DataFrame(index=future_dates, columns=data.columns)
    
    for i in range(days):
        if i == 0:
            future_df.iloc[i] = data.iloc[-1]
        else:
            future_df.iloc[i] = future_df.iloc[i-1]
        
        features = future_df[['MA5', 'MA20', 'RSI', 'Volatility', 'Returns']].iloc[[i]].fillna(0)
        future_df.loc[future_df.index[i], 'Returns'] = model.predict(features)[0]
        future_df.loc[future_df.index[i], 'Close'] = future_df['Close'].iloc[i-1] * (1 + future_df['Returns'].iloc[i])
        
        future_df.loc[future_df.index[i], 'MA5'] = future_df['Close'].iloc[max(0, i-4):i+1].mean()
        future_df.loc[future_df.index[i], 'MA20'] = future_df['Close'].iloc[max(0, i-19):i+1].mean()
        future_df.loc[future_df.index[i], 'RSI'] = calculate_rsi(future_df['Close'].iloc[:i+1]).iloc[-1]
        future_df.loc[future_df.index[i], 'Volatility'] = future_df['Returns'].iloc[max(0, i-19):i+1].std()
    
    return future_df

# Main execution
ticker = "BLK"  # BlackRock stock
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of historical data

data = fetch_data(ticker, start_date, end_date)
data = create_features(data)

model = train_model(data)

future_predictions = predict_future(model, data)

print("Predicted returns for next 30 days:")
for date, pred in zip(future_predictions.index, future_predictions['Returns']):
    print(f"{date.date()}: {pred:.2%}")

# Identify best investment days
best_days = future_predictions[future_predictions['Returns'] > future_predictions['Returns'].mean()]
print("\nBest days to invest (above average returns):")
for date in best_days.index:
    print(f"{date.date()}: {best_days.loc[date, 'Returns']:.2%}")


# In[ ]:





# In[ ]:





# In[15]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
import calendar

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Close'].pct_change()
    return data.dropna()

def create_features(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    return data.dropna()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(data):
    features = ['MA5', 'MA20', 'RSI', 'Volatility', 'Returns']
    target = 'Returns'
    
    X = data[features]
    y = data[target].shift(-1)
    X = X[:-1]
    y = y[:-1]
    
    # Remove any remaining NaN or infinite values
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Any NaN in X:", X.isna().any().any())
    print("Any NaN in y:", y.isna().any())
    print("Any infinite in X:", np.isinf(X).any().any())
    print("Any infinite in y:", np.isinf(y).any())
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_to_month_end(model, data, start_date):
    last_day_of_month = date(start_date.year, start_date.month, calendar.monthrange(start_date.year, start_date.month)[1])
    days_to_predict = (last_day_of_month - start_date.date()).days + 1
    
    future_dates = [start_date + timedelta(days=i) for i in range(days_to_predict)]
    future_df = pd.DataFrame(index=future_dates, columns=data.columns)
    
    for i in range(days_to_predict):
        if i == 0:
            future_df.iloc[i] = data.iloc[-1]
        else:
            future_df.iloc[i] = future_df.iloc[i-1]
        
        features = future_df[['MA5', 'MA20', 'RSI', 'Volatility', 'Returns']].iloc[[i]].fillna(0)
        predicted_return = model.predict(features)[0]
        future_df.loc[future_df.index[i], 'Returns'] = predicted_return
        future_df.loc[future_df.index[i], 'Close'] = future_df['Close'].iloc[i-1] * (1 + predicted_return)
        
        future_df.loc[future_df.index[i], 'MA5'] = future_df['Close'].iloc[max(0, i-4):i+1].mean()
        future_df.loc[future_df.index[i], 'MA20'] = future_df['Close'].iloc[max(0, i-19):i+1].mean()
        future_df.loc[future_df.index[i], 'RSI'] = calculate_rsi(future_df['Close'].iloc[:i+1]).iloc[-1]
        future_df.loc[future_df.index[i], 'Volatility'] = future_df['Returns'].iloc[max(0, i-19):i+1].std()
    
    return future_df

# Main execution
ticker = "BLK"  # BlackRock stock
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of historical data

data = fetch_data(ticker, start_date, end_date)
data = create_features(data)

print("Data shape after feature creation:", data.shape)
print("Any NaN in data:", data.isna().any().any())
print("Any infinite in data:", np.isinf(data).any().any())

model = train_model(data)

future_predictions = predict_to_month_end(model, data, end_date)

print(f"\nCurrent stock price: ${data['Close'].iloc[-1]:.2f}")
print(f"Predicted stock price at the end of this month ({future_predictions.index[-1].date()}): ${future_predictions['Close'].iloc[-1]:.2f}")
print(f"Predicted change: {((future_predictions['Close'].iloc[-1] / data['Close'].iloc[-1]) - 1) * 100:.2f}%")

# Show daily predictions
print("\nDaily price predictions:")
for date, price in zip(future_predictions.index, future_predictions['Close']):
    print(f"{date.date()}: ${price:.2f}")


# In[16]:


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(data):
    data['Returns'] = data['Close'].pct_change()
    data['Target'] = data['Returns'].shift(-1)
    data = data.dropna()
    
    # Simple features
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    return data.dropna()

def train_model(data):
    features = ['Close', 'Returns', 'MA5', 'MA20']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def predict_price(model, data, days=30):
    last_date = data.index[-1]
    date_range = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    future_data = pd.DataFrame(index=date_range, columns=['Close', 'Returns', 'MA5', 'MA20'])
    
    for i in range(days):
        if i == 0:
            future_data.iloc[i] = data.iloc[-1][['Close', 'Returns', 'MA5', 'MA20']]
        else:
            future_data.iloc[i] = future_data.iloc[i-1]
        
        predicted_return = model.predict(future_data.iloc[[i]])[0]
        future_data.loc[future_data.index[i], 'Close'] = future_data['Close'].iloc[i] * (1 + predicted_return)
        future_data.loc[future_data.index[i], 'Returns'] = predicted_return
        future_data.loc[future_data.index[i], 'MA5'] = future_data['Close'].iloc[max(0, i-4):i+1].mean()
        future_data.loc[future_data.index[i], 'MA20'] = future_data['Close'].iloc[max(0, i-19):i+1].mean()
    
    return future_data

def main():
    ticker = "BLK"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of historical data

    data = fetch_stock_data(ticker, start_date, end_date)
    if data is None:
        return

    prepared_data = prepare_data(data)
    model = train_model(prepared_data)

    future_data = predict_price(model, prepared_data)

    print(f"Current stock price: ${prepared_data['Close'].iloc[-1]:.2f}")
    print(f"Predicted stock price at the end of the month ({future_data.index[-1].date()}): ${future_data['Close'].iloc[-1]:.2f}")
    print(f"Predicted change: {((future_data['Close'].iloc[-1] / prepared_data['Close'].iloc[-1]) - 1) * 100:.2f}%")

    print("\nDaily price predictions:")
    for date, price in zip(future_data.index, future_data['Close']):
        print(f"{date.date()}: ${price:.2f}")

if __name__ == "__main__":
    main()


# In[21]:


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(data):
    data['Returns'] = data['Close'].pct_change()
    data['Target'] = data['Returns'].shift(-1)
    data = data.dropna()
    
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    return data.dropna()

def train_model(data):
    features = ['Close', 'Returns', 'MA5', 'MA20']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Train R2 Score: {train_score:.4f}")
    print(f"Test R2 Score: {test_score:.4f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R2 Score: {r2:.4f}")
    
    return model

def predict_price(model, data, days=30):
    last_date = data.index[-1]
    date_range = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    future_data = pd.DataFrame(index=date_range, columns=['Close', 'Returns', 'MA5', 'MA20'])
    
    for i in range(days):
        if i == 0:
            future_data.iloc[i] = data.iloc[-1][['Close', 'Returns', 'MA5', 'MA20']]
        else:
            future_data.iloc[i] = future_data.iloc[i-1]
        
        predicted_return = model.predict(future_data.iloc[[i]])[0]
        future_data.loc[future_data.index[i], 'Close'] = future_data['Close'].iloc[i] * (1 + predicted_return)
        future_data.loc[future_data.index[i], 'Returns'] = predicted_return
        future_data.loc[future_data.index[i], 'MA5'] = future_data['Close'].iloc[max(0, i-4):i+1].mean()
        future_data.loc[future_data.index[i], 'MA20'] = future_data['Close'].iloc[max(0, i-19):i+1].mean()
    
    return future_data

def backtest(data, window_size=252, prediction_days=30):
    results = []
    for i in range(window_size, len(data) - prediction_days, prediction_days):
        train_data = data.iloc[i-window_size:i]
        test_data = data.iloc[i:i+prediction_days]
        
        model = train_model(train_data)
        predictions = predict_price(model, train_data, days=prediction_days)
        
        actual_returns = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0]) - 1
        predicted_returns = (predictions['Close'].iloc[-1] / predictions['Close'].iloc[0]) - 1
        
        results.append({
            'start_date': test_data.index[0],
            'end_date': test_data.index[-1],
            'actual_returns': actual_returns,
            'predicted_returns': predicted_returns
        })
    
    return pd.DataFrame(results)

def plot_backtest_results(backtest_results):
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['start_date'], backtest_results['actual_returns'], label='Actual Returns')
    plt.plot(backtest_results['start_date'], backtest_results['predicted_returns'], label='Predicted Returns')
    plt.title('Backtest Results: Actual vs Predicted Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.show()

def main():
    ticker = "BLK"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of historical data

    data = fetch_stock_data(ticker, start_date, end_date)
    if data is None:
        return

    prepared_data = prepare_data(data)
    model = train_model(prepared_data)

    future_data = predict_price(model, prepared_data)

    print(f"\nCurrent stock price: ${prepared_data['Close'].iloc[-1]:.2f}")
    print(f"Predicted stock price at the end of the month ({future_data.index[-1].date()}): ${future_data['Close'].iloc[-1]:.2f}")
    print(f"Predicted change: {((future_data['Close'].iloc[-1] / prepared_data['Close'].iloc[-1]) - 1) * 100:.2f}%")

    print("\nDaily price predictions:")
    for date, price in zip(future_data.index, future_data['Close']):
        print(f"{date.date()}: ${price:.2f}")

    # Backtesting
    backtest_results = backtest(prepared_data)
    plot_backtest_results(backtest_results)

    # Compare with baseline model (predicting no change)
    baseline_mse = mean_squared_error(prepared_data['Target'], np.zeros_like(prepared_data['Target']))
    print(f"\nBaseline Model (No Change) MSE: {baseline_mse:.6f}")

if __name__ == "__main__":
    main()


# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def fetch_and_prepare_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Close'].pct_change()
    data['Target'] = data['Returns'].shift(-1)
    
    # Additional features
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['RSI'] = calculate_rsi(data['Close'])
    
    return data.dropna()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(data):
    features = ['Close', 'Returns', 'MA5', 'MA20', 'Volatility', 'RSI']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Train R2 Score: {train_score:.4f}")
    print(f"Test R2 Score: {test_score:.4f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R2 Score: {r2:.4f}")
    
    return model

# Main execution and prediction functions would be similar to before, 
# but using the new features in the prediction step

# Remember to update the predict_price function to use the new features as well


# In[27]:


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

def fetch_and_prepare_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Close'].pct_change()
    data['Target'] = data['Close'].shift(-1)
    
    # Technical indicators
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['ATR'] = calculate_atr(data)
    
    return data.dropna()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

def create_features(data):
    features = ['Close', 'Returns', 'MA5', 'MA20', 'RSI', 'MACD', 'Volatility', 'ATR']
    X = data[features]
    y = data['Target']
    return X, y

def train_ensemble_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'ElasticNet': ElasticNet(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        print(f"{name} Performance:")
        print(f"Train R2 Score: {train_score:.4f}")
        print(f"Test R2 Score: {test_score:.4f}")
        print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print()
    
    return models, scaler

def ensemble_predict(models, scaler, X):
    X_scaled = scaler.transform(X)
    predictions = np.column_stack([model.predict(X_scaled) for model in models.values()])
    return np.mean(predictions, axis=1)

def backtest(data, window_size=252, prediction_days=30):
    results = []
    for i in range(window_size, len(data) - prediction_days, prediction_days):
        train_data = data.iloc[i-window_size:i]
        test_data = data.iloc[i:i+prediction_days]
        
        X_train, y_train = create_features(train_data)
        X_test, y_test = create_features(test_data)
        
        models, scaler = train_ensemble_model(X_train, y_train)
        predictions = ensemble_predict(models, scaler, X_test)
        
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        results.append({
            'start_date': test_data.index[0],
            'end_date': test_data.index[-1],
            'mse': mse,
            'r2': r2
        })
    
    return pd.DataFrame(results)

def main():
    ticker = "BLK"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of historical data

    data = fetch_and_prepare_data(ticker, start_date, end_date)
    X, y = create_features(data)
    
    models, scaler = train_ensemble_model(X, y)
    
    # Predict next 30 days
    last_30_days = X.tail(30)
    predictions = ensemble_predict(models, scaler, last_30_days)
    
    print("Predicted prices for the next 30 days:")
    for date, pred in zip(last_30_days.index, predictions):
        print(f"{date.date()}: ${pred:.2f}")
    
    # Backtest
    backtest_results = backtest(data)
    print("\nBacktest Results:")
    print(backtest_results.describe())
    
    # Plot backtest results
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['start_date'], backtest_results['r2'], label='R2 Score')
    plt.title('Backtest Results: R2 Score over Time')
    plt.xlabel('Date')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


# In[3]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    return income_stmt, balance_sheet, cash_flow

def analyze_segment_performance(ticker):
    stock = yf.Ticker(ticker)
    
    # Fetch the latest annual report
    latest_info = stock.get_info()
    latest_fiscal_year = pd.to_datetime(latest_info['mostRecentQuarter']).year - 1
    
    # Manual input of segment data (you would need to update this annually from the 10-K)
    segments = {
        'BlackRock': [
            ('Investment advisory, administration fees and securities lending revenue', 13301, 14034, 11777, 11553),
            ('Investment advisory performance fees', 1143, 1104, 450, 1073),
            ('Technology services revenue', 1541, 1281, 989, 974),
            ('Distribution fees', 1409, 1131, 1062, 1069),
            ('Advisory and other revenue', 438, 169, 192, 269)
        ]
    }
    
    years = list(range(latest_fiscal_year-3, latest_fiscal_year+1))
    
    df = pd.DataFrame(segments['BlackRock'], columns=['Segment'] + years)
    df.set_index('Segment', inplace=True)
    
    # Calculate year-over-year growth
    for year in years[1:]:
        df[f'{year} YoY Growth'] = (df[year] / df[year-1] - 1) * 100
    
    return df

def plot_segment_performance(df):
    plt.figure(figsize=(15, 10))
    df[df.columns[:4]].plot(kind='bar', width=0.8)
    plt.title('BlackRock Segment Performance')
    plt.xlabel('Segments')
    plt.ylabel('Revenue (in millions USD)')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Heatmap of YoY growth
    growth_cols = [col for col in df.columns if 'Growth' in col]
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[growth_cols], annot=True, cmap='RdYlGn', center=0)
    plt.title('Year-over-Year Growth by Segment')
    plt.tight_layout()
    plt.show()

# Main execution
ticker = "BLK"
income_stmt, balance_sheet, cash_flow = fetch_financial_data(ticker)

print("Income Statement Overview:")
print(income_stmt)

segment_df = analyze_segment_performance(ticker)
print("\nSegment Performance Analysis:")
print(segment_df)

plot_segment_performance(segment_df)

# Analyze profitability ratios
latest_year = income_stmt.columns[0]
revenue = income_stmt.loc['Total Revenue', latest_year]
net_income = income_stmt.loc['Net Income', latest_year]
total_assets = balance_sheet.loc['Total Assets', latest_year]
shareholders_equity = balance_sheet.loc['Total Stockholder Equity', latest_year]

roi = net_income / total_assets * 100
roe = net_income / shareholders_equity * 100
profit_margin = net_income / revenue * 100

print(f"\nProfitability Ratios for {latest_year}:")
print(f"Return on Assets (ROA): {roi:.2f}%")
print(f"Return on Equity (ROE): {roe:.2f}%")
print(f"Profit Margin: {profit_margin:.2f}%")

# Identify potential areas of concern
print("\nPotential Areas of Concern:")
for segment in segment_df.index:
    latest_growth = segment_df.loc[segment, f'{latest_year} YoY Growth']
    if latest_growth < 0:
        print(f"- {segment}: {latest_growth:.2f}% YoY decline")
    elif latest_growth < segment_df[f'{latest_year} YoY Growth'].mean():
        print(f"- {segment}: {latest_growth:.2f}% YoY growth (below average)")


# In[4]:


# Analyze profitability ratios
latest_year = income_stmt.columns[0]
revenue = income_stmt.loc['Total Revenue', latest_year]
net_income = income_stmt.loc['Net Income', latest_year]
total_assets = balance_sheet.loc['Total Assets', latest_year]

# Try different possible keys for stockholders' equity
equity_keys = ['Total Stockholder Equity', 'Total Equity', 'Stockholders Equity']
shareholders_equity = None
for key in equity_keys:
    if key in balance_sheet.index:
        shareholders_equity = balance_sheet.loc[key, latest_year]
        break

print(f"\nProfitability Ratios for {latest_year}:")
print(f"Revenue: ${revenue:,.2f}")
print(f"Net Income: ${net_income:,.2f}")
print(f"Total Assets: ${total_assets:,.2f}")

if shareholders_equity is not None:
    print(f"Shareholders' Equity: ${shareholders_equity:,.2f}")

    roi = net_income / total_assets * 100
    roe = net_income / shareholders_equity * 100
    profit_margin = net_income / revenue * 100

    print(f"Return on Assets (ROA): {roi:.2f}%")
    print(f"Return on Equity (ROE): {roe:.2f}%")
    print(f"Profit Margin: {profit_margin:.2f}%")
else:
    print("Shareholders' Equity data not available. Unable to calculate ROE.")

    roi = net_income / total_assets * 100
    profit_margin = net_income / revenue * 100

    print(f"Return on Assets (ROA): {roi:.2f}%")
    print(f"Profit Margin: {profit_margin:.2f}%")

# Identify potential areas of concern
print("\nPotential Areas of Concern:")
growth_cols = [col for col in segment_df.columns if isinstance(col, str) and 'Growth' in col]
if growth_cols:
    latest_growth_col = growth_cols[-1]
    for segment in segment_df.index:
        latest_growth = segment_df.loc[segment, latest_growth_col]
        if latest_growth < 0:
            print(f"- {segment}: {latest_growth:.2f}% YoY decline")
        elif latest_growth < segment_df[latest_growth_col].mean():
            print(f"- {segment}: {latest_growth:.2f}% YoY growth (below average)")
else:
    print("No growth data available for analysis of potential areas of concern.")

# Print full balance sheet for debugging
print("\nFull Balance Sheet:")
print(balance_sheet)


# In[5]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    return income_stmt, balance_sheet, cash_flow

def analyze_segment_performance(ticker):
    stock = yf.Ticker(ticker)

    # Fetch the latest annual report
    latest_info = stock.get_info()
    latest_fiscal_year = pd.to_datetime(latest_info['mostRecentQuarter']).year - 1

    # Manual input of segment data (you would need to update this annually from the 10-K)
    segments = {
        'BlackRock': [
            ('Investment advisory, administration fees and securities lending revenue', 13301, 14034, 11777, 11553),
            ('Investment advisory performance fees', 1143, 1104, 450, 1073),
            ('Technology services revenue', 1541, 1281, 989, 974),
            ('Distribution fees', 1409, 1131, 1062, 1069),
            ('Advisory and other revenue', 438, 169, 192, 269)
        ]
    }

    years = list(range(latest_fiscal_year-3, latest_fiscal_year+1))

    df = pd.DataFrame(segments['BlackRock'], columns=['Segment'] + years)
    df.set_index('Segment', inplace=True)

    # Calculate year-over-year growth
    for year in years[1:]:
        df[f'{year} YoY Growth'] = (df[year] / df[year-1] - 1) * 100

    return df

def plot_segment_performance(df):
    plt.figure(figsize=(15, 10))
    df[df.columns[:4]].plot(kind='bar', width=0.8)
    plt.title('BlackRock Segment Performance')
    plt.xlabel('Segments')
    plt.ylabel('Revenue (in millions USD)')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Heatmap of YoY growth
    growth_cols = [col for col in df.columns if isinstance(col, str) and 'Growth' in col]
    if growth_cols:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[growth_cols], annot=True, cmap='RdYlGn', center=0)
        plt.title('Year-over-Year Growth by Segment')
        plt.tight_layout()
        plt.show()
    else:
        print("No growth data available for heatmap visualization.")

# Main execution
ticker = "BLK"
income_stmt, balance_sheet, cash_flow = fetch_financial_data(ticker)

print("Income Statement Overview:")
print(income_stmt)

segment_df = analyze_segment_performance(ticker)
print("\nSegment Performance Analysis:")
print(segment_df)

plot_segment_performance(segment_df)

# Analyze profitability ratios
latest_year = income_stmt.columns[0]
revenue = income_stmt.loc['Total Revenue', latest_year]
net_income = income_stmt.loc['Net Income', latest_year]
total_assets = balance_sheet.loc['Total Assets', latest_year]
shareholders_equity = balance_sheet.loc['Total Stockholder Equity', latest_year]

roi = net_income / total_assets * 100
roe = net_income / shareholders_equity * 100
profit_margin = net_income / revenue * 100

print(f"\nProfitability Ratios for {latest_year}:")
print(f"Return on Assets (ROA): {roi:.2f}%")
print(f"Return on Equity (ROE): {roe:.2f}%")
print(f"Profit Margin: {profit_margin:.2f}%")

# Identify potential areas of concern
print("\nPotential Areas of Concern:")
growth_cols = [col for col in segment_df.columns if isinstance(col, str) and 'Growth' in col]
if growth_cols:
    latest_growth_col = growth_cols[-1]
    for segment in segment_df.index:
        latest_growth = segment_df.loc[segment, latest_growth_col]
        if latest_growth < 0:
            print(f"- {segment}: {latest_growth:.2f}% YoY decline")
        elif latest_growth < segment_df[latest_growth_col].mean():
            print(f"- {segment}: {latest_growth:.2f}% YoY growth (below average)")
else:
    print("No growth data available for analysis of potential areas of concern.")


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class AladdinGrowthSimulator:
    def __init__(self, initial_revenue, initial_aum, initial_users, years):
        self.initial_revenue = initial_revenue
        self.initial_aum = initial_aum
        self.initial_users = initial_users
        self.years = years

    def simulate_growth(self, revenue_growth_rate, aum_growth_rate, user_growth_rate, volatility):
        periods = self.years * 12  # Monthly simulation

        revenue = [self.initial_revenue]
        aum = [self.initial_aum]
        users = [self.initial_users]

        for _ in range(1, periods):
            revenue_shock = np.random.normal(0, volatility)
            aum_shock = np.random.normal(0, volatility)
            user_shock = np.random.normal(0, volatility / 2)  # Assume user growth is less volatile

            new_revenue = revenue[-1] * (1 + (revenue_growth_rate / 12 + revenue_shock))
            new_aum = aum[-1] * (1 + (aum_growth_rate / 12 + aum_shock))
            new_users = users[-1] * (1 + (user_growth_rate / 12 + user_shock))

            revenue.append(new_revenue)
            aum.append(new_aum)
            users.append(new_users)

        return pd.DataFrame({
            'Revenue': revenue,
            'AUM': aum,
            'Users': users
        }, index=pd.date_range(start='2024-01-01', periods=periods, freq='M'))

    def run_monte_carlo(self, num_simulations, growth_scenarios):
        results = []
        for scenario, params in growth_scenarios.items():
            scenario_results = []
            for _ in range(num_simulations):
                sim = self.simulate_growth(**params)
                scenario_results.append(sim.iloc[-1])  # Last row of each simulation
            results.append(pd.DataFrame(scenario_results, columns=['Revenue', 'AUM', 'Users']))
            results[-1]['Scenario'] = scenario

        return pd.concat(results, ignore_index=True)

    def plot_results(self, monte_carlo_results):
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))

        for metric, ax in zip(['Revenue', 'AUM', 'Users'], axes):
            for scenario in monte_carlo_results['Scenario'].unique():
                scenario_data = monte_carlo_results[monte_carlo_results['Scenario'] == scenario]
                ax.hist(scenario_data[metric], bins=50, alpha=0.5, label=scenario)

            ax.set_title(f'Distribution of {metric} After {self.years} Years')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def analyze_results(self, monte_carlo_results):
        analysis = monte_carlo_results.groupby('Scenario').agg({
            'Revenue': ['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)],
            'AUM': ['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)],
            'Users': ['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)]
        })

        analysis.columns = ['Mean', 'Std Dev', '5th Percentile', '95th Percentile']
        return analysis

# Initialize the simulator
simulator = AladdinGrowthSimulator(
    initial_revenue=1.29e9,  # $1.29 billion
    initial_aum=21.6e12,     # $21.6 trillion
    initial_users=55000,
    years=5
)

# Define growth scenarios
growth_scenarios = {
    'Conservative': {
        'revenue_growth_rate': 0.05,
        'aum_growth_rate': 0.07,
        'user_growth_rate': 0.03,
        'volatility': 0.02
    },
    'Moderate': {
        'revenue_growth_rate': 0.10,
        'aum_growth_rate': 0.12,
        'user_growth_rate': 0.05,
        'volatility': 0.03
    },
    'Aggressive': {
        'revenue_growth_rate': 0.15,
        'aum_growth_rate': 0.18,
        'user_growth_rate': 0.08,
        'volatility': 0.04
    }
}

# Run Monte Carlo simulation
mc_results = simulator.run_monte_carlo(1000, growth_scenarios)

# Plot results
simulator.plot_results(mc_results)

# Analyze and print results
analysis = simulator.analyze_results(mc_results)
print(analysis)

# Calculate and print compound annual growth rate (CAGR) for each scenario
for scenario in growth_scenarios.keys():
    scenario_data = mc_results[mc_results['Scenario'] == scenario]
    revenue_cagr = (np.mean(scenario_data['Revenue']) / simulator.initial_revenue) ** (1/simulator.years) - 1
    aum_cagr = (np.mean(scenario_data['AUM']) / simulator.initial_aum) ** (1/simulator.years) - 1
    user_cagr = (np.mean(scenario_data['Users']) / simulator.initial_users) ** (1/simulator.years) - 1

    print(f"\n{scenario} Scenario CAGR:")
    print(f"Revenue CAGR: {revenue_cagr:.2%}")
    print(f"AUM CAGR: {aum_cagr:.2%}")
    print(f"Users CAGR: {user_cagr:.2%}")


# In[ ]:




