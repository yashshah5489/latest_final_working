import os
import json
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

def generate_llm_response(query, news, financial_wisdom, stock_data=None):
    """
    Generate a response using the Groq LLM API
    
    Args:
        query (str): User's query
        news (str): Relevant financial news
        financial_wisdom (str): Extracted financial wisdom
        stock_data (pandas.DataFrame, optional): Stock data if available
    
    Returns:
        str: LLM-generated response
    """
    try:
        # Get API key from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            return """
## API Key Required

To provide personalized financial advice using advanced AI capabilities, I need a Groq API key.

Please set the `GROQ_API_KEY` in your .env file to enable the full functionality of this assistant. You can obtain a Groq API key by signing up at [groq.com](https://groq.com).

Once you have the key, add it to the .env file and restart the application.

### For Now

I can still help with stock analysis and provide general information about Indian finance based on the data available:

1. Try exploring stock data in the "Stock Analysis" tab
2. Ask general questions about financial concepts
3. Inquire about Indian market indices like Nifty50 or Sensex
            """
        
        # Initialize Groq client
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama3-70b-8192",  # Using Llama 3 70B model
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=4096,
            top_p=0.9
        )
        
        # Prepare system prompt
        system_prompt = """You are an expert Indian financial advisor and analyst specializing in Indian markets, economy, regulations, investments, and personal finance. 

Your task is to provide detailed, accurate, and actionable financial advice to users based on their queries. Your responses should:

1. Be tailored specifically to the Indian financial context (mentioning relevant Indian regulations, tax laws, market conditions, etc.)
2. Incorporate the latest financial news provided to you
3. Draw from established financial wisdom and best practices
4. Include stock-specific analysis when relevant
5. Always provide balanced perspectives, discussing both benefits and risks
6. Be clear, concise, and accessible to users with varying levels of financial literacy
7. Use relevant Indian examples, terms, and currency (₹) where appropriate
8. NEVER make up or fabricate information - if you're uncertain, acknowledge the limits of your knowledge

Always frame your advice within the context of Indian financial regulations, market conditions, and cultural factors.
"""

        # Prepare stock data section if available
        stock_data_text = ""
        if stock_data is not None and not stock_data.empty:
            # Format the stock data for the prompt
            latest_price = stock_data['Close'].iloc[-1]
            price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
            percent_change = (price_change / stock_data['Close'].iloc[0]) * 100
            
            stock_data_text = f"""
## Stock Data Summary:
- Latest Price: ₹{latest_price:.2f}
- Change over period: ₹{price_change:.2f} ({percent_change:.2f}%)
- Highest price in period: ₹{stock_data['High'].max():.2f}
- Lowest price in period: ₹{stock_data['Low'].min():.2f}
- Average trading volume: {int(stock_data['Volume'].mean()):,}
"""

        # Prepare the human message with the query, news, and financial wisdom
        human_prompt = f"""
# User Query:
{query}

# Relevant Financial News:
{news}

{stock_data_text}

# Financial Wisdom:
{financial_wisdom}

Please provide a comprehensive response that addresses the user's query with actionable insights, considering the Indian financial context, the provided news, financial wisdom, and stock data (if available). Your response should be well-structured, informative, and tailored specifically to the Indian financial landscape.
"""

        # Generate response
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        return response.content
    
    except Exception as e:
        # Fallback response in case of API failure
        fallback_response = f"""I apologize, but I'm having trouble connecting to my knowledge base at the moment ({str(e)}). 

Here's some general guidance based on your query:

{query}

For financial advice in India, it's generally recommended to:
1. Maintain an emergency fund covering 6-12 months of expenses
2. Consider diversifying investments across FDs, mutual funds, stocks, and government schemes
3. Start retirement planning early through EPF, PPF, and NPS
4. Ensure adequate insurance coverage (health and life)
5. Consult with a SEBI-registered financial advisor for personalized advice

Please try your query again later when the service is fully operational.
"""
        return fallback_response
