import yfinance as yf
import pandas as pd

def calculate_technical_indicators(stock_data):
    """
    Dummy implementation for technical indicators.
    Replace this with your actual indicator calculations.
    """
    # For demonstration, calculate simple moving averages for closing prices.
    tech_indicators = {
        'SMA20': stock_data['Close'].rolling(window=20).mean(),
        'SMA50': stock_data['Close'].rolling(window=50).mean(),
        'SMA200': stock_data['Close'].rolling(window=200).mean(),
        'EMA20': stock_data['Close'].ewm(span=20, adjust=False).mean(),
        # Dummy values for the rest
        'RSI': pd.Series([50] * len(stock_data), index=stock_data.index),
        'MACD': pd.Series([0] * len(stock_data), index=stock_data.index),
        'Signal': pd.Series([0] * len(stock_data), index=stock_data.index),
        'UpperBand': pd.Series([0] * len(stock_data), index=stock_data.index),
        'LowerBand': pd.Series([0] * len(stock_data), index=stock_data.index),
        'Stoch_K': pd.Series([0] * len(stock_data), index=stock_data.index),
        'Stoch_D': pd.Series([0] * len(stock_data), index=stock_data.index),
        'ATR': pd.Series([0] * len(stock_data), index=stock_data.index),
        'OBV': pd.Series([0] * len(stock_data), index=stock_data.index)
    }
    return pd.DataFrame(tech_indicators)

def get_stock_info(ticker):
    # Create a yfinance Ticker object
    stock = yf.Ticker(ticker)
    
    # Step 1: Get historical data
    stock_data = stock.history(period="1y", interval="1d", auto_adjust=True)
    
    if stock_data.empty or len(stock_data) == 0:
        # Try alternative extensions if data is empty
        if ticker.endswith(".NS"):
            alternative = ticker.replace(".NS", ".BO")
            stock = yf.Ticker(alternative)
            stock_data = stock.history(period="1y", interval="1d", auto_adjust=True)
            if not stock_data.empty and len(stock_data) > 0:
                ticker = alternative
        elif ticker.endswith(".BO"):
            alternative = ticker.replace(".BO", ".NS")
            stock = yf.Ticker(alternative)
            stock_data = stock.history(period="1y", interval="1d", auto_adjust=True)
            if not stock_data.empty and len(stock_data) > 0:
                ticker = alternative
        elif not ticker.startswith('^') and not ticker.endswith((".NS", ".BO")):
            # Try with NSE extension first
            alternative = f"{ticker}.NS"
            stock = yf.Ticker(alternative)
            stock_data = stock.history(period="1y", interval="1d", auto_adjust=True)
            if not stock_data.empty and len(stock_data) > 0:
                ticker = alternative
            else:
                # If NSE fails, try BSE
                alternative = f"{ticker}.BO"
                stock = yf.Ticker(alternative)
                stock_data = stock.history(period="1y", interval="1d", auto_adjust=True)
                if not stock_data.empty and len(stock_data) > 0:
                    ticker = alternative
        
    # Exit if still no data
    if stock_data.empty or len(stock_data) == 0:
        print(f"No historical data found for {ticker}")
        return {
            'longName': ticker,
            'currentPrice': 0,
            'dayChange': 0,
            'error': "No data available"
        }
    
    # Get the latest price data
    latest_data = stock_data.iloc[-1]
    prev_data = stock_data.iloc[-2] if len(stock_data) > 1 else latest_data
    
    current_price = latest_data['Close']
    previous_close = prev_data['Close']
    day_change_percent = ((current_price / previous_close) - 1) * 100 if previous_close > 0 else 0
    
    # Step 2: Get all available info from Yahoo Finance
    try:
        info = stock.info
    except Exception as info_error:
        print(f"Error getting detailed info: {str(info_error)}")
        info = {}
    
    # Step 3: Prepare technical indicators
    tech_data = calculate_technical_indicators(stock_data)
    latest_tech = tech_data.iloc[-1] if not tech_data.empty and len(tech_data) > 0 else None
    
    # Step 4: Create the complete stock information dictionary
    stock_info = {
        # Basic info
        'symbol': ticker,
        'longName': info.get('shortName', info.get('longName', ticker)),
        'exchange': info.get('exchange', 'N/A'),
        'currency': info.get('currency', 'INR'),
        'currentPrice': current_price,
        'dayChange': day_change_percent,
        'dayHigh': latest_data.get('High', 0),
        'dayLow': latest_data.get('Low', 0),
        'volume': latest_data.get('Volume', 0),
        
        # Fundamental data
        'marketCap': info.get('marketCap', 0),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'trailingPE': info.get('trailingPE', None),
        'forwardPE': info.get('forwardPE', None),
        'priceToBook': info.get('priceToBook', None),
        'bookValue': info.get('bookValue', None),
        'dividendYield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
        'profitMargins': info.get('profitMargins', None) * 100 if info.get('profitMargins') else None,
        'operatingMargins': info.get('operatingMargins', None) * 100 if info.get('operatingMargins') else None,
        'returnOnAssets': info.get('returnOnAssets', None) * 100 if info.get('returnOnAssets') else None,
        'returnOnEquity': info.get('returnOnEquity', None) * 100 if info.get('returnOnEquity') else None,
        'revenueGrowth': info.get('revenueGrowth', None) * 100 if info.get('revenueGrowth') else None,
        'earningsGrowth': info.get('earningsGrowth', None) * 100 if info.get('earningsGrowth') else None,
        'totalCash': info.get('totalCash', None),
        'totalDebt': info.get('totalDebt', None),
        'debtToEquity': info.get('debtToEquity', None),
        'currentRatio': info.get('currentRatio', None),
        
        # Market data
        'beta': info.get('beta', None),
        'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', stock_data['High'].max()),
        'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', stock_data['Low'].min()),
        'averageVolume': info.get('averageVolume', stock_data['Volume'].mean()),
        'averageVolume10days': info.get('averageVolume10days', None),
        
        # Technical indicators (if available)
        'technicalIndicators': {}
    }
    
    # Add technical indicators if available
    if latest_tech is not None:
        stock_info['technicalIndicators'] = {
            'sma20': round(latest_tech.get('SMA20', 0), 2),
            'sma50': round(latest_tech.get('SMA50', 0), 2),
            'sma200': round(latest_tech.get('SMA200', 0), 2),
            'ema20': round(latest_tech.get('EMA20', 0), 2),
            'rsi': round(latest_tech.get('RSI', 0), 2),
            'macd': round(latest_tech.get('MACD', 0), 2),
            'macdSignal': round(latest_tech.get('Signal', 0), 2),
            'upperBand': round(latest_tech.get('UpperBand', 0), 2),
            'lowerBand': round(latest_tech.get('LowerBand', 0), 2),
            'stochK': round(latest_tech.get('Stoch_K', 0), 2),
            'stochD': round(latest_tech.get('Stoch_D', 0), 2),
            'atr': round(latest_tech.get('ATR', 0), 2),
            'obv': latest_tech.get('OBV', 0)
        }
    
    print(f"Successfully retrieved comprehensive info for {ticker}")
    return stock_info

if __name__ == "__main__":
    # Replace with your desired ticker symbol; for example, "AAPL" for Apple Inc.
    test_ticker = "INFY.NS"
    info = get_stock_info(test_ticker)
    print("\nRetrieved Data:")
    for key, value in info.items():
        print(f"{key}: {value}")
