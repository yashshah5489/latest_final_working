import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_llm_response(query, news, financial_wisdom, stock_data=None, chat_history=None):
    """
    Generate a response using the Groq LLM API
    
    Args:
        query (str): User's query
        news (str): Relevant financial news
        financial_wisdom (str): Extracted financial wisdom
        stock_data (pandas.DataFrame, optional): Stock data if available
        chat_history (list, optional): List of previous chat messages in format [{"role": "user", "content": msg}, {"role": "assistant", "content": resp}]
    
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
        
        # Prepare system prompt
        system_prompt = """**Role Definition:**  
You are an expert Indian financial advisor and analyst specializing in Indian markets, economy, regulations, investments, and personal finance.

**Primary Objective:**  
Deliver detailed, accurate, and actionable financial advice tailored exclusively to the Indian financial context. Your guidance should be rooted in verified sources and data, making it accessible to users across a range of financial literacy levels.

**Guidelines for Your Responses:**

1. **Contextual Relevance:**  
   - Base every response on India’s financial landscape, with explicit references to Indian regulations, tax laws, market conditions, and cultural nuances.  
   - Utilize local examples and terminology, and ensure all monetary values are denoted in Indian Rupees (₹).

2. **Integration of Real-Time Data:**  
   - Incorporate the latest financial news when provided, using it to ensure the timeliness and relevancy of your advice.  
   - When available, integrate stock data detailed in the "Stock Data Summary" section for data-backed insights. Analyze metrics such as the latest price, price changes, high/low values, and average trading volumes.

3. **Analytical Rigor and Financial Wisdom:**  
   - Leverage established financial wisdom and best practices to offer balanced insights that weigh both benefits and risks.  
   - In stock-specific discussions, conduct a thorough analysis of all available data, clearly outlining potential rewards and associated risks.

4. **Accuracy and Transparency:**  
   - Do not fabricate or invent information. If there is any uncertainty or incomplete data, acknowledge the limits of your knowledge explicitly.  
   - Ensure every piece of advice is evidence-based and transparently grounded in current data.

5. **Communication Style:**  
   - Respond in a clear, concise, and organized manner that is accessible to users with varying levels of financial literacy.  
   - Structure your answers with appropriate headings and sections, ensuring clarity in addressing the user query, relevant news, stock data, and financial wisdom.

6. **Prioritization of Stock Data:**  
   - When stock data is available, give it higher priority over older financial news. In stock-related queries, center your analysis on up-to-date market metrics and price information, use the news items only as contextual support if necessary.

**Always:**  
Frame your advice within the current context of Indian financial regulations, market conditions, and cultural factors. Your recommendations should be client-focused, actionable, and designed to empower informed decision-making with the most accurate and timely information.


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

        # Prepare the user message with the query, news, and financial wisdom
        user_prompt = f"""
# User Query:
{query}

# Relevant Financial News:
{news}

{stock_data_text}

# Financial Wisdom:
{financial_wisdom}

Please provide a comprehensive response that addresses the user's query with actionable insights, considering the Indian financial context, the provided news, financial wisdom, and stock data (if available). Your response should be well-structured, informative, and tailored specifically to the Indian financial landscape.
"""

        # Set up the API call to Groq
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Start with the system message
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history if it exists
        if chat_history and len(chat_history) > 0:
            # Only include the most recent 10 messages to stay within context limits
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            
            # Convert the chat history to the format expected by the API
            for message in recent_history:
                if message["role"] in ["user", "assistant"]:
                    messages.append({"role": message["role"], "content": message["content"]})
        
        # Add the current user query
        messages.append({"role": "user", "content": user_prompt})
        
        # Prepare the API request
        data = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1024,
            "top_p": 0.9
        }
        
        # Make the API call
        response = requests.post(url, headers=headers, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    except Exception as e:
        # Fallback response in case of API failure
        error_message = str(e)
        print(f"LLM API Error: {error_message}")
        
        # Check if the query is about market indices
        index_keywords = ["nifty", "sensex", "index", "market", "nse", "bse"]
        is_index_query = any(keyword in query.lower() for keyword in index_keywords)
        
        if is_index_query:
            fallback_response = f"""I apologize, but I'm having trouble connecting to my knowledge base at the moment (unsupported format string passed to Series format). 

Here's some general guidance about Indian market indices based on your query:

{query}

### Market Indices in India:
1. **Nifty 50**: The National Stock Exchange's benchmark index representing 50 of the largest Indian companies
2. **Sensex**: The Bombay Stock Exchange's benchmark index tracking 30 financially sound companies
3. **Nifty Bank**: Index representing the 12 most liquid and large capitalized stocks from the banking sector
4. **Nifty Next 50**: Represents 50 companies from Nifty 100 after excluding Nifty 50 companies

### General Market Investment Strategies:
1. Consider index funds or ETFs tracking Nifty or Sensex for broad market exposure
2. Implement a systematic investment plan (SIP) approach for regular investing
3. Monitor sectoral trends and their impact on indices
4. Be aware of global factors that influence Indian markets
5. Stay updated with RBI and SEBI policies that affect market direction

💡 **Stock data found!** Switch to the 'Stock Analysis' tab for detailed charts and analysis of market indices.

Please try your query again later when the service is fully operational.
"""
        else:
            fallback_response = f"""I apologize, but I'm having trouble connecting to my knowledge base at the moment. 

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
