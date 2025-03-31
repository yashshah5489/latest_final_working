import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.news_service import get_financial_news
from utils.stock_service import get_stock_data, get_stock_info, get_nifty50_list, get_sensex_list
from utils.llm_service import generate_llm_response
from utils.rag_service import get_financial_wisdom

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="India Financial Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

if "time_period" not in st.session_state:
    st.session_state.time_period = "1mo"
    
if "news_sources" not in st.session_state:
    st.session_state.news_sources = []
    
if "current_financial_wisdom" not in st.session_state:
    st.session_state.current_financial_wisdom = ""
    
if "rerun_flag" not in st.session_state:
    st.session_state.rerun_flag = False

def display_chat_interface():
    st.header("🇮🇳 Indian Financial Assistant")
    
    # Add a "New Chat" button in the top right
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("New Chat"):
            # Clear chat history and reset state
            st.session_state.chat_history = []
            st.session_state.current_financial_wisdom = ""
            st.session_state.rerun_flag = True
            # Rerun the app to refresh the interface
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_query = st.chat_input("Ask me about Indian finances, investments, or specific stocks...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Log user query for analytics
        print(f"USER QUERY: {user_query}")
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Display assistant response with loading spinner
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                try:
                    # Get relevant financial news and sources
                    news, news_sources = get_financial_news(user_query)
                    
                    # Store news sources in session state for display and immediately update the sidebar
                    st.session_state.news_sources = news_sources
                    
                    # Force an immediate sidebar refresh to show the updated news sources
                    with st.sidebar:
                        if news_sources:
                            st.empty()  # Clear any previous content
                            st.markdown("### 📰 Latest News Sources")
                            for i, source in enumerate(news_sources[:3]):  # Display top 3 sources
                                # Make sure we have all the required keys
                                title = source.get('title', 'Article')
                                url = source.get('url', '#')
                                date = source.get('date', '')
                                st.markdown(f"{i+1}. [{title}]({url}) {date}")
                    
                    # Get financial wisdom
                    wisdom = get_financial_wisdom(user_query)
                    
                    # Store the current financial wisdom for display in the sidebar
                    st.session_state.current_financial_wisdom = wisdom
                    
                    # Force an immediate sidebar refresh to show the updated wisdom
                    with st.sidebar:
                        if wisdom:
                            st.markdown("---")
                            with st.expander("💡 Financial Insights", expanded=True):
                                # Format and display the financial wisdom
                                wisdom_text = wisdom
                                # Limit the length to avoid overwhelming the sidebar
                                if len(wisdom_text) > 500:
                                    wisdom_text = wisdom_text[:500] + "..."
                                st.markdown(wisdom_text)
                    
                    # Check if query is stock-specific or index-specific
                    stock_data = None
                    stock_detected = False
                    index_detected = False
                    
                    # Define common indices and check for them first
                    indices = {
                        "nifty": "^NSEI",  # Nifty 50
                        "sensex": "^BSESN",  # BSE Sensex
                        "nifty bank": "^NSEBANK",  # Nifty Bank
                        "nifty 50": "^NSEI",
                        "nifty50": "^NSEI"
                    }
                    
                    # Check for indices first
                    query_lower = user_query.lower()
                    for index_name, index_symbol in indices.items():
                        if index_name in query_lower:
                            try:
                                # Try to get index data
                                index_data = get_stock_data(index_symbol, "1mo")
                                if index_data is not None and not index_data.empty:
                                    stock_data = index_data
                                    st.session_state.selected_stock = index_symbol
                                    st.session_state.stock_data = index_data
                                    index_detected = True
                                    break
                            except Exception as e:
                                print(f"Error fetching index data for {index_name}: {str(e)}")
                    
                    # If no index detected, try to find stocks
                    if not index_detected and any(keyword in query_lower for keyword in ["stock", "share", "nse", "bse", "price", "ltd", "limited"]):
                        # Try to extract stock name/symbol from query
                        import re
                        stock_matches = re.findall(r'\b[A-Za-z]+(?:\s[A-Za-z]+)*\b', user_query)
                        
                        # Define common words to ignore
                        common_words = ['the', 'and', 'for', 'about', 'what', 'how', 'stock', 'share', 
                                        'price', 'market', 'india', 'indian', 'should', 'could', 'would', 
                                        'invest', 'investment', 'trade', 'trading', 'buy', 'sell', 'tell', 
                                        'me', 'please', 'thanks', 'you', 'your', 'my', 'can', 'will']
                        
                        for potential_stock in stock_matches:
                            if len(potential_stock) > 2 and potential_stock.lower() not in common_words:
                                try:
                                    # Try NSE first
                                    stock_data = get_stock_data(potential_stock + ".NS", "1mo")
                                    if stock_data is not None and not stock_data.empty:
                                        st.session_state.selected_stock = potential_stock + ".NS"
                                        st.session_state.stock_data = stock_data
                                        stock_detected = True
                                        break
                                except Exception as e:
                                    print(f"NSE lookup failed for {potential_stock}: {str(e)}")
                                    try:
                                        # Try BSE if NSE fails
                                        stock_data = get_stock_data(potential_stock + ".BO", "1mo")
                                        if stock_data is not None and not stock_data.empty:
                                            st.session_state.selected_stock = potential_stock + ".BO"
                                            st.session_state.stock_data = stock_data
                                            stock_detected = True
                                            break
                                    except:
                                        pass
                    
                    # Generate LLM response with chat history context
                    response = generate_llm_response(
                        user_query, 
                        news, 
                        wisdom, 
                        stock_data, 
                        chat_history=st.session_state.chat_history[:-1]  # Exclude the current query
                    )
                    
                    # Log response for analytics
                    print(f"ASSISTANT RESPONSE GENERATED")
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Display response
                    st.markdown(response)
                    
                    # Suggest switching to stock tab if stock data was found
                    if stock_data is not None and not stock_data.empty:
                        if index_detected:
                            st.info("💡 Market index data found! Switch to the 'Stock Analysis' tab for detailed charts and analysis of this index.")
                        elif stock_detected:
                            st.info("💡 Stock data found! Switch to the 'Stock Analysis' tab for detailed charts and analysis of this stock.")
                
                except Exception as e:
                    error_msg = f"Error processing your query: {str(e)}"
                    st.error(error_msg)
                    print(f"ERROR: {error_msg}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"I apologize, but I encountered an error while processing your query: {str(e)}"})

def display_stock_analysis():
    st.header("📈 Stock Analysis Dashboard")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Stock selector with default lists and indices
        stock_options = ["Search for a stock..."]
        
        # Add major indices first
        indices = [
            "Nifty 50 (^NSEI)",
            "Sensex (^BSESN)",
            "Nifty Bank (^NSEBANK)",
            "Nifty IT (NIFTYIT.NS)",
            "Nifty Auto (NIFTYAUTO.NS)",
            "Nifty Pharma (NIFTYPHARMA.NS)"
        ]
        stock_options.extend(indices)
        
        # Add Nifty 50 and Sensex stocks
        try:
            nifty_stocks = get_nifty50_list()
            sensex_stocks = get_sensex_list()
            
            if nifty_stocks:
                stock_options.extend([f"{stock} (Nifty50)" for stock in nifty_stocks])
            
            if sensex_stocks:
                stock_options.extend([f"{stock} (Sensex)" for stock in sensex_stocks])
        except Exception as e:
            print(f"Error loading stock lists: {str(e)}")
        
        selected_option = st.selectbox("Select a stock or index:", stock_options)
        
        if selected_option != "Search for a stock...":
            try:
                print(f"User selected: {selected_option}")
                
                # Handle the different formats of stock options
                if "(" in selected_option and ")" in selected_option:
                    # Extract the parts of the selection
                    parts = selected_option.split(" (")
                    stock_name = parts[0].strip()
                    category = parts[1].strip().rstrip(")")
                    
                    # Case 1: Major indices with Yahoo Finance symbols
                    if category in ["^NSEI", "^BSESN", "^NSEBANK", "NIFTYIT.NS", "NIFTYAUTO.NS", "NIFTYPHARMA.NS"]:
                        selected_stock = category
                        print(f"Index symbol directly used: {selected_stock}")
                    
                    # Case 2: Nifty50 stocks
                    elif category == "Nifty50":
                        selected_stock = f"{stock_name}.NS"
                        print(f"Nifty50 stock selected: {selected_stock}")
                    
                    # Case 3: Sensex stocks
                    elif category == "Sensex":
                        selected_stock = f"{stock_name}.NS"
                        print(f"Sensex stock selected: {selected_stock}")
                    
                    # Fallback
                    else:
                        selected_stock = f"{stock_name}.NS"
                        print(f"Unknown category, using NSE: {selected_stock}")
                
                # Handle selections without parentheses
                else:
                    # Check if it's one of the known indices
                    if selected_option == "Nifty 50":
                        selected_stock = "^NSEI"
                    elif selected_option == "Sensex":
                        selected_stock = "^BSESN"
                    elif selected_option == "Nifty Bank":
                        selected_stock = "^NSEBANK"
                    elif selected_option == "Nifty IT":
                        selected_stock = "NIFTYIT.NS"
                    elif selected_option == "Nifty Auto":
                        selected_stock = "NIFTYAUTO.NS"
                    elif selected_option == "Nifty Pharma":
                        selected_stock = "NIFTYPHARMA.NS"
                    # For regular stocks, add NSE extension
                    else:
                        selected_stock = f"{selected_option}.NS"
                    
                    print(f"Direct selection: {selected_stock}")
                        
                st.session_state.selected_stock = selected_stock
            except Exception as e:
                print(f"Error processing selected option: {str(e)}")
                # Provide a default value
                st.session_state.selected_stock = "^NSEI"
        
        # Custom stock input
        # Help text with example of correct format for BSE stocks
        st.caption("For BSE stocks, use format: STOCKNAME.BO (example: RELIANCE.BO)")
        
        custom_stock = st.text_input("Or enter a specific stock symbol (add .NS for NSE or .BO for BSE):", 
                                     value="" if st.session_state.selected_stock in [entry for entry in stock_options if entry != "Search for a stock..."] else (st.session_state.selected_stock if st.session_state.selected_stock else ""))
        
        if custom_stock:
            # Only update if the user has actually entered something new
            if custom_stock != st.session_state.selected_stock:
                st.session_state.selected_stock = custom_stock

    with col2:
        # Time period selector
        time_periods = {"1 Week": "1wk", "1 Month": "1mo", "3 Months": "3mo", 
                       "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"}
        
        selected_period = st.selectbox("Select time period:", list(time_periods.keys()))
        st.session_state.time_period = time_periods[selected_period]
    
    with col3:
        # Chart type selector
        chart_types = ["Candlestick", "Line", "OHLC"]
        selected_chart = st.selectbox("Select chart type:", chart_types)
    
    # Display stock data if a stock is selected
    if st.session_state.selected_stock:
        try:
            # Fetch stock data
            stock_data = get_stock_data(st.session_state.selected_stock, st.session_state.time_period)
            
            if stock_data is not None and not stock_data.empty:
                st.session_state.stock_data = stock_data
                
                # Display stock info
                stock_info = get_stock_info(st.session_state.selected_stock)
                
                if stock_info:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{stock_info.get('currentPrice', 'N/A'):,.2f}", 
                                  f"{stock_info.get('dayChange', 0):.2f}%")
                    with col2:
                        st.metric("52-Week High", f"₹{stock_info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}")
                    with col3:
                        st.metric("52-Week Low", f"₹{stock_info.get('fiftyTwoWeekLow', 'N/A'):,.2f}")
                    with col4:
                        st.metric("Market Cap", f"₹{stock_info.get('marketCap', 0)/10000000:,.2f}Cr")
                
                # Debugging code can be uncommented if needed
                # with st.sidebar.expander("Technical Details", expanded=False):
                #     st.markdown("### Data Information")
                #     st.markdown(f"Stock data shape: {stock_data.shape}")
                #     st.markdown(f"Data columns: {list(stock_data.columns)}")
                #     st.markdown("First few rows:")
                #     st.dataframe(stock_data.head(3))
                
                # Create chart and handle errors
                fig = go.Figure()
                
                if stock_data.empty:
                    st.error("No data available for the selected stock and time period.")
                else:
                    # Check for required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    missing_columns = [col for col in required_columns if col not in stock_data.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        # Ensure index is datetime
                        if not isinstance(stock_data.index, pd.DatetimeIndex):
                            stock_data.index = pd.to_datetime(stock_data.index)
                        
                        try:
                            # Make sure all data is properly formatted for charts
                            # Convert any values to numeric if they aren't already
                            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                if col in stock_data.columns:
                                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                            
                            # Check for NaN values and clean them
                            stock_data = stock_data.dropna(subset=['Close'])
                            
                            # Create different chart types using numpy arrays to avoid Series.format errors
                            if selected_chart == "Candlestick":
                                fig.add_trace(go.Candlestick(
                                    x=stock_data.index,
                                    open=stock_data['Open'].values,  # Convert to numpy array
                                    high=stock_data['High'].values, 
                                    low=stock_data['Low'].values,
                                    close=stock_data['Close'].values,
                                    name='Price',
                                    increasing_line_color='#00BD9D',
                                    decreasing_line_color='#FF5252'
                                ))
                            elif selected_chart == "OHLC":
                                fig.add_trace(go.Ohlc(
                                    x=stock_data.index,
                                    open=stock_data['Open'].values,
                                    high=stock_data['High'].values,
                                    low=stock_data['Low'].values,
                                    close=stock_data['Close'].values,
                                    name='Price',
                                    increasing_line_color='#00BD9D',
                                    decreasing_line_color='#FF5252'
                                ))
                            else:  # Line chart
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['Close'].values,  # Convert to numpy array
                                    mode='lines',
                                    name='Close Price',
                                    line=dict(color='#00BD9D', width=2)
                                ))
                            
                            # Add volume chart with numpy array values
                            if 'Volume' in stock_data.columns and not stock_data['Volume'].isna().all():
                                volume_values = stock_data['Volume'].values  # Convert to numpy array
                                fig.add_trace(go.Bar(
                                    x=stock_data.index,
                                    y=volume_values,
                                    name='Volume',
                                    marker=dict(color='rgba(100, 100, 255, 0.3)'),
                                    yaxis='y2'
                                ))
                                
                        except Exception as e:
                            st.error(f"Error creating chart: {str(e)}")
                
                # Set layout
                display_name = st.session_state.selected_stock
                # Format the display name differently based on whether it's an index or stock
                if display_name.startswith('^'):
                    if display_name == '^NSEI':
                        display_name = 'Nifty 50'
                    elif display_name == '^BSESN':
                        display_name = 'BSE Sensex'
                    elif display_name == '^NSEBANK':
                        display_name = 'Nifty Bank'
                    title_text = f"{display_name} Index"
                else:
                    display_name = display_name.split('.')[0]
                    title_text = f"{display_name} Stock Price"
                    
                fig.update_layout(
                    title=title_text,
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    yaxis2=dict(
                        title='Volume',
                        title_font=dict(color='rgba(100, 100, 255, 0.8)'),
                        tickfont=dict(color='rgba(100, 100, 255, 0.8)'),
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)'
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    hovermode='x unified',
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=600
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional analysis tabs
                stock_analysis_tabs = st.tabs(["Key Metrics", "Technical Indicators", "Fundamental Data", "Moving Averages"])
                
                with stock_analysis_tabs[0]:
                    # Key Metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.subheader("Valuation Metrics")
                        metrics_data = {
                            "PE Ratio": stock_info.get('trailingPE', 'N/A'),
                            "Forward PE": stock_info.get('forwardPE', 'N/A'),
                            "PEG Ratio": stock_info.get('pegRatio', 'N/A'),
                            "Price to Book": stock_info.get('priceToBook', 'N/A'),
                            "Book Value": stock_info.get('bookValue', 'N/A'),
                            "Dividend Yield": f"{stock_info.get('dividendYield', 0):.2f}%" if stock_info.get('dividendYield') is not None else "N/A"
                        }
                        
                        # Avoid dataframe conversion issues by formatting the data directly with HTML
                        metrics_html = "<table width='100%'><tr><th>Metric</th><th>Value</th></tr>"
                        for metric, value in metrics_data.items():
                            metrics_html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                        metrics_html += "</table>"
                        
                        # Use markdown to display the table
                        st.markdown(metrics_html, unsafe_allow_html=True)
                        
                    with metrics_col2:
                        st.subheader("Market Metrics")
                        market_data = {
                            "Market Cap": f"₹{stock_info.get('marketCap', 0)/10000000:,.2f}Cr" if stock_info.get('marketCap') else 'N/A',
                            "Beta": f"{stock_info.get('beta', 'N/A')}",
                            "Volume": f"{stock_info.get('volume', 0):,}",
                            "Avg Volume": f"{stock_info.get('averageVolume', 0):,}",
                            "Exchange": stock_info.get('exchange', 'N/A'),
                            "Currency": stock_info.get('currency', 'INR')
                        }
                        
                        market_html = "<table width='100%'><tr><th>Metric</th><th>Value</th></tr>"
                        for metric, value in market_data.items():
                            market_html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                        market_html += "</table>"
                        
                        st.markdown(market_html, unsafe_allow_html=True)
                
                with stock_analysis_tabs[1]:
                    # Technical Indicators Tab
                    st.subheader("Technical Indicators")
                    
                    # Check if we have technical indicators available
                    if 'technicalIndicators' in stock_info and stock_info['technicalIndicators']:
                        tech_indicators = stock_info['technicalIndicators']
                        
                        # Create multiple columns for better display
                        t_col1, t_col2, t_col3 = st.columns(3)
                        
                        with t_col1:
                            # Moving Averages vs Price
                            st.markdown("#### Moving Averages")
                            ma_data = {
                                "SMA20": tech_indicators.get('sma20', 'N/A'),
                                "SMA50": tech_indicators.get('sma50', 'N/A'),
                                "SMA200": tech_indicators.get('sma200', 'N/A'),
                                "EMA20": tech_indicators.get('ema20', 'N/A'),
                                "Current Price": stock_info.get('currentPrice', 'N/A')
                            }
                            
                            # Color formatting for MAs vs price
                            ma_html = "<table width='100%'><tr><th>Indicator</th><th>Value</th></tr>"
                            price = stock_info.get('currentPrice', 0)
                            for indicator, value in ma_data.items():
                                if indicator == "Current Price":
                                    ma_html += f"<tr><td>{indicator}</td><td>{value:.2f}</td></tr>"
                                else:
                                    if isinstance(value, (int, float)) and price:
                                        color = "green" if value < price else "red" if value > price else "black"
                                        ma_html += f"<tr><td>{indicator}</td><td style='color:{color}'>{value:.2f}</td></tr>"
                                    else:
                                        ma_html += f"<tr><td>{indicator}</td><td>N/A</td></tr>"
                            ma_html += "</table>"
                            
                            st.markdown(ma_html, unsafe_allow_html=True)
                        
                        with t_col2:
                            # Momentum Indicators
                            st.markdown("#### Momentum Indicators")
                            
                            # RSI
                            rsi = tech_indicators.get('rsi', 'N/A')
                            if isinstance(rsi, (int, float)):
                                rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "black"
                                rsi_text = f"<span style='color:{rsi_color}'>{rsi:.2f}</span>"
                            else:
                                rsi_text = "N/A"
                                
                            # MACD
                            macd = tech_indicators.get('macd', 'N/A')
                            macd_signal = tech_indicators.get('macdSignal', 'N/A')
                            if isinstance(macd, (int, float)) and isinstance(macd_signal, (int, float)):
                                macd_color = "green" if macd > macd_signal else "red"
                                macd_text = f"<span style='color:{macd_color}'>{macd:.2f}</span>"
                                signal_text = f"{macd_signal:.2f}"
                            else:
                                macd_text = "N/A"
                                signal_text = "N/A"
                                
                            # Stochastic
                            stoch_k = tech_indicators.get('stochK', 'N/A')
                            stoch_d = tech_indicators.get('stochD', 'N/A')
                            if isinstance(stoch_k, (int, float)):
                                stoch_color = "red" if stoch_k > 80 else "green" if stoch_k < 20 else "black"
                                stoch_k_text = f"<span style='color:{stoch_color}'>{stoch_k:.2f}</span>"
                            else:
                                stoch_k_text = "N/A"
                                
                            momentum_html = "<table width='100%'><tr><th>Indicator</th><th>Value</th><th>Signal</th></tr>"
                            momentum_html += f"<tr><td>RSI (14)</td><td>{rsi_text}</td><td>Overbought >70, Oversold <30</td></tr>"
                            momentum_html += f"<tr><td>MACD</td><td>{macd_text}</td><td>{signal_text}</td></tr>"
                            momentum_html += f"<tr><td>Stochastic %K</td><td>{stoch_k_text}</td><td>{stoch_d:.2f if isinstance(stoch_d, (int, float)) else 'N/A'}</td></tr>"
                            momentum_html += "</table>"
                            
                            st.markdown(momentum_html, unsafe_allow_html=True)
                            
                        with t_col3:
                            # Volatility & Volume Indicators
                            st.markdown("#### Volatility & Volume")
                            
                            # Bollinger Bands
                            upper_band = tech_indicators.get('upperBand', 'N/A')
                            lower_band = tech_indicators.get('lowerBand', 'N/A')
                            price = stock_info.get('currentPrice', 0)
                            
                            if isinstance(upper_band, (int, float)) and isinstance(lower_band, (int, float)) and price:
                                band_status = "Near Upper Band" if price > (upper_band * 0.95) else "Near Lower Band" if price < (lower_band * 1.05) else "Middle"
                                band_color = "red" if band_status == "Near Upper Band" else "green" if band_status == "Near Lower Band" else "black"
                            else:
                                band_status = "N/A"
                                band_color = "black"
                                
                            # ATR - Volatility
                            atr = tech_indicators.get('atr', 'N/A')
                            
                            # OBV - Volume trend
                            obv = tech_indicators.get('obv', 'N/A')
                            
                            vol_html = "<table width='100%'><tr><th>Indicator</th><th>Value</th></tr>"
                            vol_html += f"<tr><td>BB Upper</td><td>{upper_band:.2f if isinstance(upper_band, (int, float)) else 'N/A'}</td></tr>"
                            vol_html += f"<tr><td>BB Lower</td><td>{lower_band:.2f if isinstance(lower_band, (int, float)) else 'N/A'}</td></tr>"
                            vol_html += f"<tr><td>Position</td><td style='color:{band_color}'>{band_status}</td></tr>"
                            vol_html += f"<tr><td>ATR (14)</td><td>{atr:.2f if isinstance(atr, (int, float)) else 'N/A'}</td></tr>"
                            vol_html += f"<tr><td>OBV</td><td>{obv:,.0f if isinstance(obv, (int, float)) else 'N/A'}</td></tr>"
                            vol_html += "</table>"
                            
                            st.markdown(vol_html, unsafe_allow_html=True)
                    else:
                        st.info("Technical indicators not available for this stock or timeframe. Try selecting a different time period.")
                
                with stock_analysis_tabs[2]:
                    # Fundamental Data Tab
                    st.subheader("Fundamental Data")
                    
                    # Create multiple columns for better display
                    f_col1, f_col2 = st.columns(2)
                    
                    with f_col1:
                        st.markdown("#### Profitability Metrics")
                        
                        profit_data = {
                            "Profit Margin": f"{stock_info.get('profitMargins', 'N/A'):.2f}%" if stock_info.get('profitMargins') is not None else "N/A",
                            "Operating Margin": f"{stock_info.get('operatingMargins', 'N/A'):.2f}%" if stock_info.get('operatingMargins') is not None else "N/A",
                            "Return on Assets": f"{stock_info.get('returnOnAssets', 'N/A'):.2f}%" if stock_info.get('returnOnAssets') is not None else "N/A",
                            "Return on Equity": f"{stock_info.get('returnOnEquity', 'N/A'):.2f}%" if stock_info.get('returnOnEquity') is not None else "N/A",
                            "Revenue Growth": f"{stock_info.get('revenueGrowth', 'N/A'):.2f}%" if stock_info.get('revenueGrowth') is not None else "N/A",
                            "Earnings Growth": f"{stock_info.get('earningsGrowth', 'N/A'):.2f}%" if stock_info.get('earningsGrowth') is not None else "N/A"
                        }
                        
                        profit_html = "<table width='100%'><tr><th>Metric</th><th>Value</th></tr>"
                        for metric, value in profit_data.items():
                            profit_html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                        profit_html += "</table>"
                        
                        st.markdown(profit_html, unsafe_allow_html=True)
                        
                    with f_col2:
                        st.markdown("#### Financial Health")
                        
                        health_data = {
                            "Total Cash": f"₹{stock_info.get('totalCash', 0)/10000000:,.2f}Cr" if stock_info.get('totalCash') else "N/A",
                            "Total Debt": f"₹{stock_info.get('totalDebt', 0)/10000000:,.2f}Cr" if stock_info.get('totalDebt') else "N/A",
                            "Debt to Equity": stock_info.get('debtToEquity', 'N/A'),
                            "Current Ratio": stock_info.get('currentRatio', 'N/A'),
                            "Sector": stock_info.get('sector', 'N/A'),
                            "Industry": stock_info.get('industry', 'N/A')
                        }
                        
                        health_html = "<table width='100%'><tr><th>Metric</th><th>Value</th></tr>"
                        for metric, value in health_data.items():
                            health_html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                        health_html += "</table>"
                        
                        st.markdown(health_html, unsafe_allow_html=True)
                    
                    with metrics_col2:
                        st.subheader("Technical Indicators")
                        # Calculate RSI
                        delta = stock_data['Close'].diff()
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        # Moving averages
                        ma20 = stock_data['Close'].rolling(window=20).mean().iloc[-1]
                        ma50 = stock_data['Close'].rolling(window=50).mean().iloc[-1]
                        ma200 = stock_data['Close'].rolling(window=200).mean().iloc[-1]
                        
                        last_close = stock_data['Close'].iloc[-1]
                        
                        tech_data = {
                            "RSI (14)": f"{rsi.iloc[-1]:.2f}",
                            "20-Day MA": f"₹{ma20:.2f}",
                            "50-Day MA": f"₹{ma50:.2f}",
                            "200-Day MA": f"₹{ma200:.2f}",
                            "Price vs 20-Day MA": f"{(last_close/ma20 - 1) * 100:.2f}%" if not pd.isna(ma20) else "N/A",
                            "Price vs 50-Day MA": f"{(last_close/ma50 - 1) * 100:.2f}%" if not pd.isna(ma50) else "N/A",
                            "Price vs 200-Day MA": f"{(last_close/ma200 - 1) * 100:.2f}%" if not pd.isna(ma200) else "N/A"
                        }
                        
                        # Avoid dataframe conversion issues by formatting the data directly with HTML
                        tech_html = "<table width='100%'><tr><th>Indicator</th><th>Value</th></tr>"
                        for indicator, value in tech_data.items():
                            # Color code percentage values
                            if "%" in str(value) and not value == "N/A":
                                if value.startswith("-"):
                                    tech_html += f"<tr><td>{indicator}</td><td style='color:#FF5252'>{value}</td></tr>"
                                else:
                                    tech_html += f"<tr><td>{indicator}</td><td style='color:#00BD9D'>{value}</td></tr>"
                            else:
                                tech_html += f"<tr><td>{indicator}</td><td>{value}</td></tr>"
                        tech_html += "</table>"
                        
                        # Use markdown to display the table
                        st.markdown(tech_html, unsafe_allow_html=True)
                
                with stock_analysis_tabs[1]:
                    # Moving Averages Chart
                    ma_fig = go.Figure()
                    
                    # Add price with numpy array values
                    ma_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].values,  # Convert to numpy array
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#FFFFFF', width=2)
                    ))
                    
                    # Add moving averages with numpy array values
                    ma20_series = stock_data['Close'].rolling(window=20).mean()
                    ma_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=ma20_series.values,  # Convert to numpy array
                        mode='lines',
                        name='20-Day MA',
                        line=dict(color='#00BD9D', width=1.5)
                    ))
                    
                    ma50_series = stock_data['Close'].rolling(window=50).mean()
                    ma_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=ma50_series.values,  # Convert to numpy array
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='#FF9800', width=1.5)
                    ))
                    
                    ma200_series = stock_data['Close'].rolling(window=200).mean()
                    ma_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=ma200_series.values,  # Convert to numpy array
                        mode='lines',
                        name='200-Day MA',
                        line=dict(color='#F44336', width=1.5)
                    ))
                    
                    # Set layout
                    ma_fig.update_layout(
                        title='Moving Averages Analysis',
                        xaxis_title='Date',
                        yaxis_title='Price (₹)',
                        plot_bgcolor='#1E1E1E',
                        paper_bgcolor='#1E1E1E',
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(255, 255, 255, 0.1)'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(255, 255, 255, 0.1)'
                        ),
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1
                        ),
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(ma_fig, use_container_width=True)
                    
                    # Interpretation
                    st.subheader("Moving Average Analysis")
                    last_close = stock_data['Close'].iloc[-1]
                    ma20_last = stock_data['Close'].rolling(window=20).mean().iloc[-1]
                    ma50_last = stock_data['Close'].rolling(window=50).mean().iloc[-1]
                    ma200_last = stock_data['Close'].rolling(window=200).mean().iloc[-1]
                    
                    # Trend analysis
                    trend_analysis = []
                    if last_close > ma20_last:
                        trend_analysis.append("• Price is above 20-Day MA, suggesting short-term bullish momentum.")
                    else:
                        trend_analysis.append("• Price is below 20-Day MA, suggesting short-term bearish pressure.")
                        
                    if last_close > ma50_last:
                        trend_analysis.append("• Price is above 50-Day MA, indicating medium-term bullish trend.")
                    else:
                        trend_analysis.append("• Price is below 50-Day MA, indicating medium-term bearish trend.")
                        
                    if last_close > ma200_last:
                        trend_analysis.append("• Price is above 200-Day MA, confirming long-term bullish trend.")
                    else:
                        trend_analysis.append("• Price is below 200-Day MA, confirming long-term bearish trend.")
                    
                    # Golden/Death cross check
                    if ma20_last > ma50_last and stock_data['Close'].rolling(window=20).mean().iloc[-2] <= stock_data['Close'].rolling(window=50).mean().iloc[-2]:
                        trend_analysis.append("• **Golden Cross Alert:** 20-Day MA has crossed above 50-Day MA, a bullish signal.")
                    elif ma20_last < ma50_last and stock_data['Close'].rolling(window=20).mean().iloc[-2] >= stock_data['Close'].rolling(window=50).mean().iloc[-2]:
                        trend_analysis.append("• **Death Cross Alert:** 20-Day MA has crossed below 50-Day MA, a bearish signal.")
                    
                    st.markdown("\n".join(trend_analysis))
                
                with stock_analysis_tabs[2]:
                    # Performance Analysis
                    st.subheader("Historical Performance")
                    
                    # Calculate different time period returns
                    current_price = stock_data['Close'].iloc[-1]
                    
                    try:
                        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                        month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                        quarter_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                        half_year_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
                        year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                        
                        week_price = stock_data.loc[stock_data.index <= week_ago, 'Close'].iloc[-1] if not stock_data.loc[stock_data.index <= week_ago, 'Close'].empty else None
                        month_price = stock_data.loc[stock_data.index <= month_ago, 'Close'].iloc[-1] if not stock_data.loc[stock_data.index <= month_ago, 'Close'].empty else None
                        quarter_price = stock_data.loc[stock_data.index <= quarter_ago, 'Close'].iloc[-1] if not stock_data.loc[stock_data.index <= quarter_ago, 'Close'].empty else None
                        half_year_price = stock_data.loc[stock_data.index <= half_year_ago, 'Close'].iloc[-1] if not stock_data.loc[stock_data.index <= half_year_ago, 'Close'].empty else None
                        year_price = stock_data.loc[stock_data.index <= year_ago, 'Close'].iloc[-1] if not stock_data.loc[stock_data.index <= year_ago, 'Close'].empty else None
                        
                        # Create periods and calculate returns as strings with percentage formatting
                        periods = ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "YTD"]
                        return_values = [
                            f"{((current_price / week_price) - 1) * 100:.2f}%" if week_price else "N/A",
                            f"{((current_price / month_price) - 1) * 100:.2f}%" if month_price else "N/A",
                            f"{((current_price / quarter_price) - 1) * 100:.2f}%" if quarter_price else "N/A",
                            f"{((current_price / half_year_price) - 1) * 100:.2f}%" if half_year_price else "N/A",
                            f"{((current_price / year_price) - 1) * 100:.2f}%" if year_price else "N/A",
                            f"{((current_price / stock_data[stock_data.index.year == datetime.now().year].iloc[0]['Close']) - 1) * 100:.2f}%" if not stock_data[stock_data.index.year == datetime.now().year].empty else "N/A"
                        ]
                        
                        # We don't need perf_df anymore, since we're using direct lists for plotting
                        
                        # Display returns as bar chart - convert to proper format for plotting
                        returns = []
                        for ret in return_values:
                            if ret == "N/A":
                                returns.append(0)
                            else:
                                try:
                                    # Remove the % sign and convert to float
                                    returns.append(float(ret.strip('%')))
                                except:
                                    returns.append(0)
                        
                        colors = ['#00BD9D' if r >= 0 else '#FF5252' for r in returns]
                        
                        perf_fig = go.Figure()
                        perf_fig.add_trace(go.Bar(
                            x=periods,  # Use the periods list directly
                            y=returns,  # Already numeric
                            marker_color=colors,
                            text=[f"{r:.2f}%" for r in returns],
                            textposition='auto'
                        ))
                        
                        perf_fig.update_layout(
                            title='Performance by Time Period',
                            xaxis_title='Time Period',
                            yaxis_title='Return (%)',
                            plot_bgcolor='#1E1E1E',
                            paper_bgcolor='#1E1E1E',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(
                                showgrid=False,
                                gridcolor='rgba(255, 255, 255, 0.1)'
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='rgba(255, 255, 255, 0.1)',
                                zeroline=True,
                                zerolinecolor='rgba(255, 255, 255, 0.3)'
                            ),
                            height=400
                        )
                        
                        st.plotly_chart(perf_fig, use_container_width=True)
                        
                        # Volatility analysis
                        st.subheader("Volatility Analysis")
                        
                        # Calculate daily returns and volatility
                        stock_data['Daily Return'] = stock_data['Close'].pct_change() * 100
                        daily_volatility = stock_data['Daily Return'].std()
                        annualized_volatility = daily_volatility * (252 ** 0.5)
                        
                        vol_col1, vol_col2 = st.columns(2)
                        
                        with vol_col1:
                            st.metric("Daily Volatility", f"{daily_volatility:.2f}%")
                            st.metric("Annualized Volatility", f"{annualized_volatility:.2f}%")
                        
                        with vol_col2:
                            # Create returns distribution
                            vol_fig = go.Figure()
                            # Use numpy array for histogram data
                            daily_returns = stock_data['Daily Return'].dropna().values
                            vol_fig.add_trace(go.Histogram(
                                x=daily_returns,
                                nbinsx=30,
                                marker_color='#00BD9D'
                            ))
                            
                            vol_fig.update_layout(
                                title='Distribution of Daily Returns',
                                xaxis_title='Daily Return (%)',
                                yaxis_title='Frequency',
                                plot_bgcolor='#1E1E1E',
                                paper_bgcolor='#1E1E1E',
                                font=dict(color='#FFFFFF'),
                                height=250
                            )
                            
                            st.plotly_chart(vol_fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error calculating performance metrics: {str(e)}")
                
            else:
                st.error(f"No data found for the selected stock: {st.session_state.selected_stock}")
        
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
    else:
        st.info("Please select a stock to view analysis.")

# Main app layout
st.title("🇮🇳 India-Focused Financial Assistant")

# Create placeholders in the sidebar for dynamic content
with st.sidebar:
    # Create a placeholder for the news sources section
    news_section = st.container()
    
    # Create a placeholder for the financial wisdom section
    wisdom_section = st.container()
    
    with news_section:
        if "news_sources" in st.session_state and st.session_state.news_sources:
            st.markdown("### 📰 Latest News Sources")
            for i, source in enumerate(st.session_state.news_sources[:3]):  # Display top 3 sources
                # Make sure we have all the required keys
                title = source.get('title', 'Article')
                url = source.get('url', '#')
                date = source.get('date', '')
                st.markdown(f"{i+1}. [{title}]({url}) {date}")
    
    with wisdom_section:
        # Display financial wisdom if available
        if "current_financial_wisdom" in st.session_state and st.session_state.current_financial_wisdom:
            st.markdown("---")
            with st.expander("💡 Financial Insights", expanded=True):
                # Format and display the financial wisdom in a scrollable container
                wisdom_text = st.session_state.current_financial_wisdom
                
                # Add custom CSS for the scrollable container
                st.markdown("""
                <style>
                .scrollable-container {
                    height: 200px;
                    overflow-y: auto;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: rgba(0, 0, 0, 0.1);
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display wisdom text in the scrollable container
                st.markdown(f"""<div class="scrollable-container">{wisdom_text}</div>""", unsafe_allow_html=True)
    
    st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["Financial Advice", "Stock Analysis"])

with tab1:
    display_chat_interface()

with tab2:
    display_stock_analysis()

# Footer
st.markdown("---")
st.markdown("### 🙏 Powered by Indian Financial Wisdom")
st.markdown("Data sources: Yahoo Finance, Groq LLM, Tavily, and Indian financial literature.")

# Handle any rerun flags set during the session
if st.session_state.rerun_flag:
    st.session_state.rerun_flag = False
    st.rerun()
