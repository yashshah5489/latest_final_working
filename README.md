# India-Focused Financial Assistant

A comprehensive financial assistant platform focused on the Indian market, providing personalized financial advice and interactive stock analysis.

## Features

- 🤖 Chat-based financial advisory powered by Groq LLM
- 📈 Interactive stock analysis for NSE and BSE stocks
- 📊 Advanced visualizations including candlestick charts, moving averages, and performance metrics
- 📰 Integration with real-time financial news
- 🧠 Customized financial wisdom for the Indian context

## Setup Instructions

### Prerequisites

- Python 3.8+
- Required packages (listed in `pyproject.toml`)

### Environment Variables

Create a `.env` file in the root directory and add the following API keys:

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

To obtain these API keys:
- GROQ API key: Sign up at [groq.com](https://groq.com)
- Tavily API key: Sign up at [tavily.com](https://tavily.com)

### Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   Or using `poetry`:
   ```
   poetry install
   ```

## Running the Application

### On Replit
The application is configured to run automatically on Replit.

### Locally

#### Option 1: Using the script
Run the provided shell script:
```
./run_local.sh
```

#### Option 2: Manual execution
Run the following command:
```
streamlit run app.py
```

The application will be available at:
- Local: http://localhost:8501
- Network: http://[YOUR-IP-ADDRESS]:8501

## Architecture

- **Frontend**: Streamlit with custom dark-themed UI
- **Data Sources**: Yahoo Finance (stock data), Tavily API (financial news)
- **NLP Engine**: Groq Cloud (LLM based on Llama3-70b-8192)
- **RAG System**: Custom implementation for Indian financial wisdom

## License

This project is licensed under the MIT License - see the LICENSE file for details.