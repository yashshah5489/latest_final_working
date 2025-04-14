import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json

def get_stock_data(ticker, period="1mo", interval="1d"):
    """
    Fetch stock data from Yahoo Finance using the Ticker API
    
    Args:
        ticker (str): Stock ticker symbol (add .NS for NSE, .BO for BSE)
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        pandas.DataFrame: Stock price data
    """
    try:
        # Clean up the input ticker symbol first
        if ticker is None:
            print("No ticker provided")
            return None
            
        # Handle different notations and normalize ticker symbols
        if ticker == "Nifty50" or ticker == "NIFTY50" or ticker == "NIFTY 50" or (isinstance(ticker, str) and ticker.lower() == "nifty50"):
            ticker = "^NSEI"  # Use the official Yahoo Finance symbol
        elif ticker == "Sensex" or ticker == "SENSEX" or ticker == "BSE SENSEX" or (isinstance(ticker, str) and ticker.lower() == "sensex"):
            ticker = "^BSESN"  # Use the official Yahoo Finance symbol
        elif ticker == "NiftyBank" or ticker == "NIFTYBANK" or ticker == "NIFTY BANK" or (isinstance(ticker, str) and ticker.lower() == "niftybank"):
            ticker = "^NSEBANK"  # Use the official Yahoo Finance symbol
        
        # Fix special cases for stock dropdowns
        if "(" in ticker and ")" in ticker:
            # Extract the actual symbol from format like "RELIANCE (Nifty50)"
            if "Nifty50" in ticker:
                ticker = ticker.split(" (")[0] + ".NS"
            elif "Sensex" in ticker:
                ticker = ticker.split(" (")[0] + ".BO"  # Use BSE extension for Sensex stocks
            elif "^" in ticker or "NIFTY" in ticker.upper():
                # Extract index symbol from format like "Nifty 50 (^NSEI)"
                ticker = ticker.split(" (")[1].rstrip(")")
            
        print(f"Getting historical data for: {ticker}")
        
        # Use Ticker API instead of download function
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period=period, interval=interval, auto_adjust=True)
        
        if stock_data.empty:
            print(f"No data found for ticker {ticker}")
            
            # Try alternative for Indian stocks if no data found
            if ticker.endswith(".NS"):
                # If NSE ticker doesn't work, try BSE
                alternative = ticker.replace(".NS", ".BO")
                print(f"Trying BSE alternative: {alternative}")
                alt_stock = yf.Ticker(alternative)
                stock_data = alt_stock.history(period=period, interval=interval, auto_adjust=True)
                
                if not stock_data.empty and len(stock_data) > 0:
                    print(f"Found data with BSE extension")
                    ticker = alternative  # Update ticker to the working alternative
            
            elif ticker.endswith(".BO") or ticker.endswith(".BS"):
                # If BSE ticker doesn't work, try NSE
                if ticker.endswith(".BO"):
                    alternative = ticker.replace(".BO", ".NS")
                else:
                    alternative = ticker.replace(".BS", ".NS")
                print(f"Trying NSE alternative: {alternative}")
                alt_stock = yf.Ticker(alternative)
                stock_data = alt_stock.history(period=period, interval=interval, auto_adjust=True)
                
                if not stock_data.empty and len(stock_data) > 0:
                    print(f"Found data with NSE extension")
                    ticker = alternative  # Update ticker to the working alternative
            
            elif not ticker.startswith('^') and not ticker.endswith((".NS", ".BO", ".BS")):
                # For plain symbols without extensions, try first NSE then BSE
                # First try NSE (more liquid market usually)
                nse_alternative = f"{ticker}.NS"
                print(f"Trying with NSE extension: {nse_alternative}")
                nse_stock = yf.Ticker(nse_alternative)
                stock_data = nse_stock.history(period=period, interval=interval, auto_adjust=True)
                
                if not stock_data.empty and len(stock_data) > 0:
                    print(f"Found data with NSE extension")
                    ticker = nse_alternative  # Update ticker to the working alternative
                else:
                    # If NSE fails, try BSE
                    bse_alternative = f"{ticker}.BO"
                    print(f"Trying with BSE extension: {bse_alternative}")
                    bse_stock = yf.Ticker(bse_alternative)
                    stock_data = bse_stock.history(period=period, interval=interval, auto_adjust=True)
                    
                    if not stock_data.empty and len(stock_data) > 0:
                        print(f"Found data with BSE extension")
                        ticker = bse_alternative  # Update ticker to the working alternative
        
        if stock_data.empty or len(stock_data) == 0:
            print(f"Still no data found for {ticker} after trying all alternatives")
            return None
        
        # Check for expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in expected_columns if col not in stock_data.columns]
        
        # if missing_columns:
        #     print(f"Warning: Missing columns: {missing_columns}")
        #     # Add missing columns with reasonable defaults
        #     for col in missing_columns:
        #         if col == 'Volume':
        #             stock_data['Volume'] = 0  # Default volume
        #         else:
        #             # For OHLC, use available price columns
        #             price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in stock_data.columns]
        #             if price_cols:
        #                 stock_data[col] = stock_data[price_cols[0]]
        #             else:
        #                 print(f"No price data available for {ticker}")
        #                 return None

        # # Convert all columns to numeric values
        # for col in stock_data.columns:
        #     stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
        
        # # Fill any NaN values (using newer pandas methods instead of deprecated ones)
        # stock_data = stock_data.ffill().bfill()


        # Data handling for missing columns - CORRECTED
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            # Add missing columns with reasonable defaults
            for col in missing_columns:
                if col == 'Volume':
                    stock_data['Volume'] = 0  # Default volume
                elif col in ['Open', 'High', 'Low', 'Close']:
                    # For OHLC, use more appropriate substitution logic
                    available_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in stock_data.columns]
                    if available_cols:
                        # If Close is missing but other price data exists, use appropriate substitution
                        if col == 'Close' and 'Open' in available_cols:
                            stock_data['Close'] = stock_data['Open']
                        elif col == 'High' and 'Close' in available_cols:
                            stock_data['High'] = stock_data['Close']
                        elif col == 'Low' and 'Close' in available_cols:
                            stock_data['Low'] = stock_data['Close']
                        elif col == 'Open' and 'Close' in available_cols:
                            stock_data['Open'] = stock_data['Close']
                        else:
                            stock_data[col] = stock_data[available_cols[0]]
                    else:
                        print(f"No price data available for {ticker}")
                        return None

        # Convert all columns to numeric values - CORRECTED
        for col in stock_data.columns:
            # Keep track of non-numeric values before coercing
            non_numeric_count = stock_data[col].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x))).sum()
            if non_numeric_count > 0:
                print(f"Warning: {non_numeric_count} non-numeric values found in column {col}")
            
            # Convert with coercion but fill NaN values appropriately
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

        # Fill NaN values using forward fill method
        # stock_data = stock_data.fillna(method='ffill')
        stock_data = stock_data.ffill()
        # Then backward fill any remaining NaNs at the beginning
        # stock_data = stock_data.fillna(method='bfill')
        stock_data = stock_data.bfill()
        
        if len(stock_data) == 0:
            print(f"No valid data rows for {ticker}")
            return None
        
        print(f"Successfully downloaded data for {ticker}, shape: {stock_data.shape}")
        print(f"Columns: {stock_data.columns.tolist()}")
        return stock_data
        
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(stock_data):
    """
    Calculate technical indicators for the stock data
    
    Args:
        stock_data (pandas.DataFrame): Stock price data with OHLCV columns
    
    Returns:
        pandas.DataFrame: Stock data with technical indicators added
    """
    if stock_data is None or len(stock_data) < 14:
        # Need at least 14 days of data for basic indicators
        return stock_data
    
    try:
        # Create a working copy
        df = stock_data.copy()
        
        # 1. Moving Averages
        # Simple Moving Averages
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # 2. Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['MA20'] + (df['StdDev'] * 2)
        df['LowerBand'] = df['MA20'] - (df['StdDev'] * 2)
        
        # 4. MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # 5. Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # 6. Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # 7. On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        # Clean up NANs (using newer pandas methods)
        df = df.bfill().ffill()
        
        return df
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        return stock_data

def get_stock_info(ticker):
    """
    Get comprehensive stock information including fundamental and technical data
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Detailed stock information
    """
    try:
        # Clean up and normalize the ticker first
        if ticker is None:
            print("No ticker provided for stock info")
            return {
                'longName': "Unknown",
                'currentPrice': 0,
                'dayChange': 0,
                'marketCap': 0
            }
            
        # Handle different notations and normalize ticker symbols
        if ticker == "Nifty50" or ticker == "NIFTY50" or ticker == "NIFTY 50" or (isinstance(ticker, str) and ticker.lower() == "nifty50"):
            ticker = "^NSEI"  # Use the official Yahoo Finance symbol
        elif ticker == "Sensex" or ticker == "SENSEX" or ticker == "BSE SENSEX" or (isinstance(ticker, str) and ticker.lower() == "sensex"):
            ticker = "^BSESN"  # Use the official Yahoo Finance symbol
        elif ticker == "NiftyBank" or ticker == "NIFTYBANK" or ticker == "NIFTY BANK" or (isinstance(ticker, str) and ticker.lower() == "niftybank"):
            ticker = "^NSEBANK"  # Use the official Yahoo Finance symbol
        
        # Fix special cases for stock dropdowns
        if "(" in ticker and ")" in ticker:
            # Extract the actual symbol from format like "RELIANCE (Nifty50)"
            if "Nifty50" in ticker:
                ticker = ticker.split(" (")[0] + ".NS"
            elif "Sensex" in ticker:
                ticker = ticker.split(" (")[0] + ".BO"  # Fix: Use BSE extension for Sensex stocks
            elif "^" in ticker or "NIFTY" in ticker.upper():
                # Extract index symbol from format like "Nifty 50 (^NSEI)"
                ticker = ticker.split(" (")[1].rstrip(")")
                
        print(f"Getting comprehensive stock info for: {ticker}")
        
        # Use Ticker API for everything
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
        
    except Exception as e:
        print(f"Error getting stock info for {ticker}: {str(e)}")
        # Return minimal info
        return {
            'longName': ticker,
            'currentPrice': 0,
            'dayChange': 0,
            'error': str(e)
        }

def get_nifty50_list():
    """
    Get list of Nifty 50 stocks
    
    Returns:
        list: List of Nifty 50 stock symbols
    """
    try:
        # This is a simplified approach - in a production environment, you would fetch this from NSE API
        nifty50_stocks = [
            "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
            "BAJFINANCE","BAJAJFINSV","BEL","BHARTIARTL","CIPLA", "COALINDIA","DRREDDY",
            "EICHERMOT","ETERNAL","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO",
            "HINDALCO","HINDUNILVR","ICICIBANK", "ITC","INDUSINDBK","INFY","JSWSTEEL",
            "JIOFIN","KOTAKBANK","LT","M&M","MARUTI","NTPC","NESTLEIND","ONGC","POWERGRID",
            "RELIANCE","SBILIFE", "SHRIRAMFIN","SBIN","SUNPHARMA","TCS","TATACONSUM",
            "TATAMOTORS","TATASTEEL","TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO"
        ]
        return nifty50_stocks
    except:
        return []

def get_sensex_list():
    """
    Get list of Sensex stocks
    
    Returns:
        list: List of Sensex stock symbols
    """
    try:
        sensex_stocks = [
            "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJFINSV",
            "BAJFINANCE", "BHARTIARTL", "ETERNAL", "HCLTECH", "HDFCBANK",
            "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC","KOTAKBANK",
            "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND", "POWERGRID", 
            "RELIANCE", "SBIN", "SUNPHARMA", "TCS", "TATAMOTORS", "TATASTEEL", 
            "TECHM", "TITAN", "ULTRACEMCO"
        ]
        return sensex_stocks
    except:
        return []

def get_market_overview():
    """
    Get market overview data including major indices
    
    Returns:
        dict: Market overview data
    """
    try:
        # Get data using Ticker API for consistency
        nifty50_ticker = yf.Ticker("^NSEI")
        sensex_ticker = yf.Ticker("^BSESN")
        nifty_bank_ticker = yf.Ticker("^NSEBANK")
        
        # Get historical data for the last 2 days
        nifty50 = nifty50_ticker.history(period="2d", interval="1d")
        sensex = sensex_ticker.history(period="2d", interval="1d")
        nifty_bank = nifty_bank_ticker.history(period="2d", interval="1d")
        
        # Initialize market data with defaults
        market_data = {
            "nifty50": {
                "current": 0,
                "change": 0
            },
            "sensex": {
                "current": 0,
                "change": 0
            },
            "nifty_bank": {
                "current": 0,
                "change": 0
            }
        }
        
        # Calculate Nifty50 data if available
        if not nifty50.empty and len(nifty50) >= 2 and 'Close' in nifty50.columns:
            nifty50_current = nifty50['Close'].iloc[-1]
            nifty50_prev = nifty50['Close'].iloc[-2]
            nifty50_change = ((nifty50_current / nifty50_prev) - 1) * 100 if nifty50_prev > 0 else 0
            market_data["nifty50"] = {
                "current": nifty50_current,
                "change": nifty50_change
            }
        
        # Calculate Sensex data if available
        if not sensex.empty and len(sensex) >= 2 and 'Close' in sensex.columns:
            sensex_current = sensex['Close'].iloc[-1]
            sensex_prev = sensex['Close'].iloc[-2]
            sensex_change = ((sensex_current / sensex_prev) - 1) * 100 if sensex_prev > 0 else 0
            market_data["sensex"] = {
                "current": sensex_current,
                "change": sensex_change
            }
        
        # Calculate Nifty Bank data if available
        if not nifty_bank.empty and len(nifty_bank) >= 2 and 'Close' in nifty_bank.columns:
            nifty_bank_current = nifty_bank['Close'].iloc[-1]
            nifty_bank_prev = nifty_bank['Close'].iloc[-2]
            nifty_bank_change = ((nifty_bank_current / nifty_bank_prev) - 1) * 100 if nifty_bank_prev > 0 else 0
            market_data["nifty_bank"] = {
                "current": nifty_bank_current,
                "change": nifty_bank_change
            }
        
        return market_data
    except Exception as e:
        print(f"Error getting market overview: {str(e)}")
        # Return defaults
        return {
            "nifty50": {"current": 0, "change": 0},
            "sensex": {"current": 0, "change": 0},
            "nifty_bank": {"current": 0, "change": 0}
        }

def get_stock_recommendations(ticker):
    """
    Get analyst recommendations for a stock
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        pandas.DataFrame: Analyst recommendations
    """
    try:
        # Clean up the ticker symbol first
        if ticker is None:
            return None
            
        # Handle special cases in the same way as other functions
        if "(" in ticker and ")" in ticker:
            if "Nifty50" in ticker:
                ticker = ticker.split(" (")[0] + ".NS"
            elif "Sensex" in ticker:
                ticker = ticker.split(" (")[0] + ".BO"
            elif "^" in ticker:
                ticker = ticker.split(" (")[1].rstrip(")")
        
        # Get stock recommendations
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations is not None and not recommendations.empty:
            # Sort by date (most recent first)
            recommendations = recommendations.sort_index(ascending=False)
            return recommendations
        else:
            return None
    except Exception as e:
        print(f"Error getting recommendations for {ticker}: {str(e)}")
        return None

def calculate_moving_averages(stock_data):
    """
    Calculate moving averages for the stock data
    
    Args:
        stock_data (pandas.DataFrame): Stock price data
    
    Returns:
        pandas.DataFrame: Moving averages data
    """
    if stock_data is None or stock_data.empty:
        return None
    
    try:
        # Create a copy to avoid modifying the original
        ma_data = stock_data.copy()
        
        # Calculate moving averages of different periods
        ma_data['MA5'] = ma_data['Close'].rolling(window=5).mean()
        ma_data['MA10'] = ma_data['Close'].rolling(window=10).mean()
        ma_data['MA20'] = ma_data['Close'].rolling(window=20).mean()
        ma_data['MA50'] = ma_data['Close'].rolling(window=50).mean()
        ma_data['MA100'] = ma_data['Close'].rolling(window=100).mean()
        ma_data['MA200'] = ma_data['Close'].rolling(window=200).mean()
        
        # Calculate exponential moving averages
        ma_data['EMA12'] = ma_data['Close'].ewm(span=12, adjust=False).mean()
        ma_data['EMA26'] = ma_data['Close'].ewm(span=26, adjust=False).mean()
        
        return ma_data
    except Exception as e:
        print(f"Error calculating moving averages: {str(e)}")
        return stock_data