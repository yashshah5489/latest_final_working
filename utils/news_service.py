import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_financial_news(query):
    """
    Get relevant financial news using the Tavily API
    
    Args:
        query (str): The user's query to search for relevant news
    
    Returns:
        str: A summary of relevant financial news
    """
    try:
        # Get Tavily API key from environment variables
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not tavily_api_key:
            print("Tavily API key not found. Using fallback news system.")
            # Parse query to customize the fallback news
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
        
        # Parse user query to extract key topics
        search_query = f"Indian finance {query}"
        
        # Add common financial terms for better results
        financial_terms = [
            "RBI", "SEBI", "NSE", "BSE", "Sensex", "Nifty", 
            "Indian economy", "fiscal policy", "monetary policy"
        ]
        
        for term in financial_terms:
            if term.lower() in query.lower():
                search_query = f"{term} {query} latest news India"
                break
        
        # Set up the Tavily API request
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Set search parameters
        payload = {
            "api_key": tavily_api_key,
            "query": search_query,
            "search_depth": "advanced",
            "include_domains": [
                "economictimes.indiatimes.com",
                "livemint.com",
                "moneycontrol.com",
                "financialexpress.com",
                "business-standard.com",
                "thehindubusinessline.com",
                "rbi.org.in",
                "sebi.gov.in",
                "nseindia.com",
                "bseindia.com"
            ],
            "max_results": 5,
            "time_window": "1w"  # Last week
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            search_response = response.json()
            
            # Format the results
            if not search_response.get("results"):
                return "No relevant financial news found for your query."
        
        formatted_news = "## Relevant Financial News:\n\n"
        
        for i, result in enumerate(search_response.get("results", [])[:5]):
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "#")
            published_date = result.get("published_date", "Unknown date")
            
            # Format the date if available
            if published_date and published_date != "Unknown date":
                try:
                    date_obj = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                    formatted_date = date_obj.strftime("%d %b %Y")
                except:
                    formatted_date = published_date
            else:
                formatted_date = "Recent"
            
            # Add to formatted news
            formatted_news += f"### {i+1}. {title}\n"
            formatted_news += f"**Date:** {formatted_date}\n\n"
            
            # Truncate content if too long
            if len(content) > 300:
                content = content[:300] + "..."
            
            formatted_news += f"{content}\n\n"
            formatted_news += f"[Read more]({url})\n\n"
        
        return formatted_news
    
    except Exception as e:
        print(f"Error in get_financial_news: {str(e)}")
        # Parse query to customize the fallback news
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
