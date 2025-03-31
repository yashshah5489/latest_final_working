import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json

def get_stock_data(ticker, period="1mo", interval="1d"):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol (add .NS for NSE, .BO for BSE)
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        pandas.DataFrame: Stock price data
    """
    try:
        # Special handling for Nifty indices that aren't working with standard names
        if ticker == "NIFTY50" or ticker == "NIFTY 50":
            ticker = "^NSEI"  # Use the official Yahoo Finance symbol
        elif ticker == "SENSEX" or ticker == "BSE SENSEX":
            ticker = "^BSESN"  # Use the official Yahoo Finance symbol
        elif ticker == "NIFTYBANK" or ticker == "NIFTY BANK":
            ticker = "^NSEBANK"  # Use the official Yahoo Finance symbol
            
        # Fetch data with auto adjust turned off to avoid conversion issues
        stock_data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if stock_data.empty:
            print(f"No data found for ticker {ticker}")
            # Try alternative for Indian stocks if no data found
            if ticker.endswith(".NS"):
                alternative = ticker.replace(".NS", ".BO")
                print(f"Trying alternative symbol: {alternative}")
                stock_data = yf.download(alternative, period=period, interval=interval, progress=False, auto_adjust=False)
            elif not ticker.startswith('^') and not ticker.endswith((".NS", ".BO")):
                # Try with .NS extension
                alternative = f"{ticker}.NS"
                print(f"Trying with NSE extension: {alternative}")
                stock_data = yf.download(alternative, period=period, interval=interval, progress=False, auto_adjust=False)
                
        if stock_data.empty:
            return None
            
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {str(e)}")
        return None

def get_stock_info(ticker):
    """
    Get stock information including current price and key statistics
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Stock information
    """
    try:
        # Special handling for indices
        if ticker == "NIFTY50" or ticker == "NIFTY 50":
            ticker = "^NSEI"  # Use the official Yahoo Finance symbol
        elif ticker == "SENSEX" or ticker == "BSE SENSEX":
            ticker = "^BSESN"  # Use the official Yahoo Finance symbol
        elif ticker == "NIFTYBANK" or ticker == "NIFTY BANK":
            ticker = "^NSEBANK"  # Use the official Yahoo Finance symbol
        
        # First try to get the latest stock data
        stock_data = get_stock_data(ticker, period="5d")
        if stock_data is None or stock_data.empty:
            print(f"No recent data found for {ticker}")
            # Return default info for the ticker
            return {
                'longName': ticker,
                'currentPrice': 0,
                'dayChange': 0,
                'marketCap': 0,
                'fiftyTwoWeekHigh': 0,
                'fiftyTwoWeekLow': 0
            }
            
        # Get basic info from the stock data
        latest_data = stock_data.iloc[-1]
        prev_data = stock_data.iloc[-2] if len(stock_data) > 1 else latest_data
        
        current_price = latest_data['Close']
        previous_close = prev_data['Close']
        day_change_percent = ((current_price / previous_close) - 1) * 100 if previous_close > 0 else 0
        
        # Try to get more detailed info from ticker
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Create a dictionary with relevant information
            stock_info = {
                'longName': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'currentPrice': current_price,
                'dayChange': day_change_percent,
                'marketCap': info.get('marketCap', 0),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', stock_data['High'].max()),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', stock_data['Low'].min()),
                'trailingPE': info.get('trailingPE', None),
                'forwardPE': info.get('forwardPE', None),
                'pegRatio': info.get('pegRatio', None),
                'priceToBook': info.get('priceToBook', None),
                'dividendYield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'averageVolume': info.get('averageVolume', stock_data['Volume'].mean()),
                'averageVolume10days': info.get('averageVolume10days', stock_data['Volume'].mean())
            }
            
            return stock_info
        except Exception as ticker_error:
            print(f"Error getting detailed stock info: {str(ticker_error)}")
            # If detailed info fails, return basic info from the data we have
            return {
                'longName': ticker,
                'currentPrice': current_price,
                'dayChange': day_change_percent,
                'marketCap': 0,
                'fiftyTwoWeekHigh': stock_data['High'].max(),
                'fiftyTwoWeekLow': stock_data['Low'].min(),
                'trailingPE': None,
                'forwardPE': None
            }
    except Exception as e:
        print(f"Error getting any stock info for {ticker}: {str(e)}")
        # Return minimal info
        return {
            'longName': ticker,
            'currentPrice': 0,
            'dayChange': 0,
            'marketCap': 0,
            'fiftyTwoWeekHigh': 0,
            'fiftyTwoWeekLow': 0
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
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
            "SUNPHARMA", "TITAN", "ULTRACEMCO", "JSWSTEEL", "TATAMOTORS",
            "ADANIENT", "BAJAJFINSV", "WIPRO", "HCLTECH", "TECHM",
            "NTPC", "M&M", "POWERGRID", "TATASTEEL", "NESTLEIND",
            "GRASIM", "HINDALCO", "DRREDDY", "ADANIPORTS", "CIPLA",
            "SBILIFE", "DIVISLAB", "INDUSINDBK", "EICHERMOT", "TATACONSUM",
            "BRITANNIA", "HDFCLIFE", "APOLLOHOSP", "COALINDIA", "BAJAJ-AUTO",
            "UPL", "HEROMOTOCO", "BPCL", "ONGC", "SHREECEM"
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
        # This is a simplified approach - in a production environment, you would fetch this from BSE API
        sensex_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
            "BHARTIARTL", "KOTAKBANK", "LT", "HINDUNILVR", "SBIN",
            "AXISBANK", "BAJFINANCE", "ITC", "MARUTI", "SUNPHARMA",
            "TITAN", "ASIANPAINT", "ULTRACEMCO", "TATAMOTORS", "BAJAJFINSV",
            "NTPC", "TECHM", "M&M", "POWERGRID", "HCLTECH",
            "JSWSTEEL", "TATASTEEL", "NESTLEIND", "INDUSINDBK", "WIPRO"
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
        # Get Nifty 50 data
        nifty50 = yf.download("^NSEI", period="2d", interval="1d", progress=False)
        
        # Get Sensex data
        sensex = yf.download("^BSESN", period="2d", interval="1d", progress=False)
        
        # Get Nifty Bank data
        nifty_bank = yf.download("^NSEBANK", period="2d", interval="1d", progress=False)
        
        # Calculate daily changes
        nifty50_current = nifty50['Close'].iloc[-1]
        nifty50_prev = nifty50['Close'].iloc[-2]
        nifty50_change = ((nifty50_current / nifty50_prev) - 1) * 100
        
        sensex_current = sensex['Close'].iloc[-1]
        sensex_prev = sensex['Close'].iloc[-2]
        sensex_change = ((sensex_current / sensex_prev) - 1) * 100
        
        nifty_bank_current = nifty_bank['Close'].iloc[-1]
        nifty_bank_prev = nifty_bank['Close'].iloc[-2]
        nifty_bank_change = ((nifty_bank_current / nifty_bank_prev) - 1) * 100
        
        market_data = {
            "nifty50": {
                "current": nifty50_current,
                "change": nifty50_change
            },
            "sensex": {
                "current": sensex_current,
                "change": sensex_change
            },
            "nifty_bank": {
                "current": nifty_bank_current,
                "change": nifty_bank_change
            }
        }
        
        return market_data
    except Exception as e:
        print(f"Error getting market overview: {str(e)}")
        return {}
