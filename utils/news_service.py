import os
import requests
from datetime import datetime, timedelta

def get_financial_news(query):
    """
    Get relevant financial news (currently using fallback system)
    
    Args:
        query (str): The user's query to search for relevant news
    
    Returns:
        str: A summary of relevant financial news
    """
    try:
        # For now, using the fallback news system
        # In a future update, a proper news API integration will be added
        
        # Parse query to customize the fallback news
        # Add common financial terms for better customization
        financial_terms = [
            "RBI", "SEBI", "NSE", "BSE", "Sensex", "Nifty", 
            "Indian economy", "fiscal policy", "monetary policy",
            "investment", "stocks", "mutual funds", "tax", "budget",
            "insurance", "retirement", "real estate", "gold"
        ]
        
        # Check if any financial terms are in the query
        mentioned_terms = []
        for term in financial_terms:
            if term.lower() in query.lower():
                mentioned_terms.append(term)
        
        return get_fallback_news(mentioned_terms, query)
    
    except Exception as e:
        print(f"Error in get_financial_news: {str(e)}")
        return get_fallback_news([], query)

def get_fallback_news(mentioned_terms=None, query=""):
    """
    Get fallback news with some contextual customization
    
    Args:
        mentioned_terms (list): Financial terms mentioned in the query
        query (str): The original user query
    
    Returns:
        str: A fallback news summary
    """
    current_date = datetime.now().strftime("%d %b %Y")
    
    # Default news items
    news_items = [
        {
            "title": "RBI Monetary Policy Update",
            "content": "The Reserve Bank of India has maintained its stance on current monetary policy. Experts suggest watching for interest rate changes in the coming quarters based on inflation trends."
        },
        {
            "title": "Indian Markets Overview",
            "content": "Indian stock markets have shown resilience despite global headwinds. Domestic institutional investors continue to provide support to the markets."
        },
        {
            "title": "Banking Sector Performance",
            "content": "Indian banks have reported improved asset quality with declining NPAs. Credit growth remains robust in retail and MSME segments."
        },
        {
            "title": "Government Financial Policies",
            "content": "Recent economic measures by the government focus on infrastructure development and boosting manufacturing through PLI schemes."
        },
        {
            "title": "Digital Payments Ecosystem",
            "content": "UPI transactions continue to set new records each month, highlighting India's rapid transition to digital payments."
        }
    ]
    
    # Special news items based on mentioned terms
    special_news = {
        "investment": {
            "title": "Investment Trends in India",
            "content": "Retail investor participation in the Indian stock market continues to grow. SIP investments have reached an all-time high, indicating increased interest in systematic investing."
        },
        "mutual funds": {
            "title": "Mutual Fund Industry Growth",
            "content": "The Indian mutual fund industry's assets under management (AUM) have seen significant growth. Equity funds continue to attract the majority of new investments."
        },
        "tax": {
            "title": "Tax Planning Considerations",
            "content": "As the financial year progresses, experts recommend reviewing tax-saving investments. ELSS funds, PPF, and NPS remain popular tax-saving instruments under Section 80C."
        },
        "retirement": {
            "title": "Retirement Planning Insights",
            "content": "Studies show that most Indians are under-prepared for retirement. Financial advisors recommend increasing retirement corpus allocations given increasing life expectancy and medical inflation."
        },
        "real estate": {
            "title": "Real Estate Market Developments",
            "content": "The Indian real estate market is showing signs of recovery in major metropolitan areas. Commercial real estate and REITs are gaining investor attention."
        },
        "gold": {
            "title": "Gold Investment Outlook",
            "content": "Gold continues to be a preferred asset for portfolio diversification. Sovereign Gold Bonds and Gold ETFs are becoming increasingly popular alternatives to physical gold."
        }
    }
    
    # Replace some default news with special news based on mentioned terms
    if mentioned_terms:
        replacement_count = 0
        for term in mentioned_terms:
            term_lower = term.lower()
            if term_lower in special_news and replacement_count < 3:
                news_items[replacement_count] = special_news[term_lower]
                replacement_count += 1
    
    # Format the news items
    formatted_news = f"## Relevant Financial News (as of {current_date}):\n\n"
    
    for i, news in enumerate(news_items):
        formatted_news += f"### {i+1}. {news['title']}\n"
        formatted_news += f"{news['content']}\n\n"
    
    return formatted_news
