#!/bin/bash

# Script to run the Financial Assistant application locally
echo "Starting India Financial Assistant application..."
echo "Checking for required environment variables..."

# Path to .env file
ENV_FILE=".env"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Warning: .env file not found. Creating a template .env file."
    echo "# API Keys for India Financial Assistant" > "$ENV_FILE"
    echo "GROQ_API_KEY=your_groq_api_key_here" >> "$ENV_FILE"
    echo "TAVILY_API_KEY=your_tavily_api_key_here" >> "$ENV_FILE"
    echo ""
    echo "Please edit the .env file and add your API keys before running the application again."
    echo "You can obtain these API keys at:"
    echo "- GROQ API key: https://groq.com"
    echo "- Tavily API key: https://tavily.com"
    exit 1
fi

# Load environment variables from .env file
echo "Loading environment variables from .env file..."
if [ -f "$ENV_FILE" ]; then
    # Export each line from .env file as environment variable
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Check if required API keys are set
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY is not set in .env file."
    echo "Please add your GROQ API key to the .env file."
    exit 1
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "Error: TAVILY_API_KEY is not set in .env file."
    echo "Please add your Tavily API key to the .env file."
    exit 1
fi

echo "Environment variables loaded successfully."
echo ""

# Detect platform and set the open command accordingly
platform=$(uname)
if [ "$platform" = "Darwin" ]; then
    # macOS
    open_cmd="open"
elif [ "$platform" = "Linux" ]; then
    # Linux
    open_cmd="xdg-open"
elif [[ "$platform" == *"MINGW"* || "$platform" == *"MSYS"* || "$platform" == *"CYGWIN"* ]]; then
    # Windows
    open_cmd="start"
else
    open_cmd="echo 'Please open a browser and navigate to'"
fi

# Start the Streamlit app
echo "Starting Streamlit server..."
echo "Once the server starts, it will be available at: http://localhost:8501"
echo ""
echo "To access from other devices on your local network, use: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
sleep 2

# Open browser after a brief delay to allow the server to start
(sleep 5 && $open_cmd "http://localhost:8501") &

# Run Streamlit with default port 8501 for local development
streamlit run app.py