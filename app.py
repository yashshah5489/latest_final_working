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

def display_chat_interface():
    st.header("🇮🇳 Indian Financial Assistant")
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_query = st.chat_input("Ask me about Indian finances, investments, or specific stocks...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Display assistant response with loading spinner
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                try:
                    # Get relevant financial news
                    news = get_financial_news(user_query)
                    
                    # Get financial wisdom
                    wisdom = get_financial_wisdom(user_query)
                    
                    # Check if query is stock-specific
                    stock_data = None
                    if any(keyword in user_query.lower() for keyword in ["stock", "share", "nse", "bse", "sensex", "nifty"]):
                        # Try to extract stock name/symbol from query
                        import re
                        stock_matches = re.findall(r'\b[A-Za-z]+(?:\s[A-Za-z]+)*\b', user_query)
                        for potential_stock in stock_matches:
                            if len(potential_stock) > 2 and potential_stock.lower() not in ['the', 'and', 'for', 'about', 'what', 'how', 'stock', 'share', 'price']:
                                try:
                                    # Try to get stock data
                                    stock_data = get_stock_data(potential_stock + ".NS", "1mo")
                                    if stock_data is not None and not stock_data.empty:
                                        st.session_state.selected_stock = potential_stock + ".NS"
                                        st.session_state.stock_data = stock_data
                                        break
                                except:
                                    pass
                    
                    # Generate LLM response
                    response = generate_llm_response(user_query, news, wisdom, stock_data)
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Display response
                    st.markdown(response)
                    
                    # Suggest switching to stock tab if stock data was found
                    if stock_data is not None and not stock_data.empty:
                        st.info("💡 Stock data found! Switch to the 'Stock Analysis' tab for detailed charts and analysis.")
                
                except Exception as e:
                    st.error(f"Error processing your query: {str(e)}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"I apologize, but I encountered an error while processing your query: {str(e)}"})

def display_stock_analysis():
    st.header("📈 Stock Analysis Dashboard")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Stock selector with default lists
        stock_options = ["Search for a stock..."]
        
        # Add Nifty 50 and Sensex stocks
        try:
            nifty_stocks = get_nifty50_list()
            sensex_stocks = get_sensex_list()
            
            if nifty_stocks:
                stock_options.extend([f"{stock} (Nifty50)" for stock in nifty_stocks])
            
            if sensex_stocks:
                stock_options.extend([f"{stock} (Sensex)" for stock in sensex_stocks])
        except:
            pass
        
        selected_option = st.selectbox("Select a stock:", stock_options)
        
        if selected_option != "Search for a stock...":
            # Extract stock symbol from the selected option
            selected_stock = selected_option.split(" (")[0] + ".NS"
            st.session_state.selected_stock = selected_stock
        
        # Custom stock input
        custom_stock = st.text_input("Or enter a specific stock symbol (add .NS for NSE or .BO for BSE):", 
                                     value=st.session_state.selected_stock if st.session_state.selected_stock else "")
        
        if custom_stock:
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
                
                # Create chart
                fig = go.Figure()
                
                if selected_chart == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name='Price',
                        increasing_line_color='#00BD9D',
                        decreasing_line_color='#FF5252'
                    ))
                elif selected_chart == "OHLC":
                    fig.add_trace(go.Ohlc(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name='Price',
                        increasing_line_color='#00BD9D',
                        decreasing_line_color='#FF5252'
                    ))
                else:  # Line chart
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#00BD9D', width=2)
                    ))
                
                # Add volume chart
                fig.add_trace(go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    name='Volume',
                    marker=dict(color='rgba(100, 100, 255, 0.3)'),
                    yaxis='y2'
                ))
                
                # Set layout
                fig.update_layout(
                    title=f"{st.session_state.selected_stock.split('.')[0]} Stock Price",
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    yaxis2=dict(
                        title='Volume',
                        titlefont=dict(color='rgba(100, 100, 255, 0.8)'),
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
                stock_analysis_tabs = st.tabs(["Key Metrics", "Moving Averages", "Performance Analysis"])
                
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
                            "Dividend Yield": f"{stock_info.get('dividendYield', 0) * 100:.2f}%" if stock_info.get('dividendYield') else "N/A"
                        }
                        
                        metrics_df = pd.DataFrame([metrics_data]).T.reset_index()
                        metrics_df.columns = ["Metric", "Value"]
                        st.table(metrics_df)
                    
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
                        
                        tech_df = pd.DataFrame([tech_data]).T.reset_index()
                        tech_df.columns = ["Indicator", "Value"]
                        st.table(tech_df)
                
                with stock_analysis_tabs[1]:
                    # Moving Averages Chart
                    ma_fig = go.Figure()
                    
                    # Add price
                    ma_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#FFFFFF', width=2)
                    ))
                    
                    # Add moving averages
                    ma_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].rolling(window=20).mean(),
                        mode='lines',
                        name='20-Day MA',
                        line=dict(color='#00BD9D', width=1.5)
                    ))
                    
                    ma_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].rolling(window=50).mean(),
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='#FF9800', width=1.5)
                    ))
                    
                    ma_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].rolling(window=200).mean(),
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
                        
                        perf_data = {
                            "Period": ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "YTD"],
                            "Return": [
                                f"{((current_price / week_price) - 1) * 100:.2f}%" if week_price else "N/A",
                                f"{((current_price / month_price) - 1) * 100:.2f}%" if month_price else "N/A",
                                f"{((current_price / quarter_price) - 1) * 100:.2f}%" if quarter_price else "N/A",
                                f"{((current_price / half_year_price) - 1) * 100:.2f}%" if half_year_price else "N/A",
                                f"{((current_price / year_price) - 1) * 100:.2f}%" if year_price else "N/A",
                                f"{((current_price / stock_data[stock_data.index.year == datetime.now().year].iloc[0]['Close']) - 1) * 100:.2f}%" if not stock_data[stock_data.index.year == datetime.now().year].empty else "N/A"
                            ]
                        }
                        
                        perf_df = pd.DataFrame(perf_data)
                        
                        # Display returns as bar chart
                        returns = [float(ret.strip('%')) if ret != "N/A" else 0 for ret in perf_df['Return']]
                        colors = ['#00BD9D' if r >= 0 else '#FF5252' for r in returns]
                        
                        perf_fig = go.Figure()
                        perf_fig.add_trace(go.Bar(
                            x=perf_df['Period'],
                            y=returns,
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
                            vol_fig.add_trace(go.Histogram(
                                x=stock_data['Daily Return'].dropna(),
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

# Create tabs
tab1, tab2 = st.tabs(["Generic Advice", "Stock Analysis"])

with tab1:
    display_chat_interface()

with tab2:
    display_stock_analysis()

# Footer
st.markdown("---")
st.markdown("### 🙏 Powered by Indian Financial Wisdom")
st.markdown("Data sources: Yahoo Finance, Groq LLM, and Indian financial literature.")
