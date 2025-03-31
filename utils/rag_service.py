import os
import re
import random

def get_financial_wisdom(query):
    """
    Retrieve relevant financial wisdom using a keyword-based approach
    
    Args:
        query (str): User's query
    
    Returns:
        str: Relevant financial wisdom
    """
    try:
        # Load financial wisdom data
        wisdom_text = load_financial_wisdom()
        
        # Split the text into chunks
        chunks = chunk_text(wisdom_text, chunk_size=300, overlap=50)
        
        if not chunks:
            return "No financial wisdom available."
        
        # Get the most relevant chunks using keyword matching
        relevant_chunks = retrieve_by_keywords(query, chunks, top_k=3)
        
        # Format the wisdom
        formatted_wisdom = "## Financial Wisdom:\n\n"
        
        for i, chunk in enumerate(relevant_chunks):
            formatted_wisdom += f"### Insight {i+1}:\n{chunk}\n\n"
        
        return formatted_wisdom
    
    except Exception as e:
        # Fallback wisdom in case of errors
        print(f"Error retrieving financial wisdom: {str(e)}")
        return get_fallback_wisdom()

def load_financial_wisdom():
    """
    Load financial wisdom from the data file
    
    Returns:
        str: Financial wisdom text
    """
    try:
        # Load from file
        wisdom_path = "data/financial_wisdom.txt"
        
        if os.path.exists(wisdom_path):
            with open(wisdom_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            # Return predefined wisdom if file doesn't exist
            return get_predefined_wisdom()
    except:
        return get_predefined_wisdom()

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split text into overlapping chunks
    
    Args:
        text (str): Text to split
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Add current chunk to chunks
            chunks.append(" ".join(current_chunk))
            
            # Keep overlap for next chunk
            overlap_size = 0
            overlap_chunk = []
            
            for s in reversed(current_chunk):
                s_size = len(s)
                if overlap_size + s_size <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_size += s_size
                else:
                    break
            
            # Start new chunk with overlap
            current_chunk = overlap_chunk
            current_size = overlap_size
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def retrieve_by_keywords(query, chunks, top_k=3):
    """
    Simple keyword-based retrieval
    
    Args:
        query (str): User's query
        chunks (list): List of text chunks
        top_k (int): Number of chunks to retrieve
    
    Returns:
        list: List of relevant chunks
    """
    # Extract keywords from query
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    
    # Add financial terms that might be relevant
    financial_terms = ["invest", "stock", "fund", "saving", "retirement", "tax", "insurance", 
                       "loan", "debt", "asset", "equity", "bond", "mutual", "fixed", "deposit", 
                       "budget", "income", "expense", "financial", "plan", "wealth", "money", 
                       "bank", "credit", "interest", "rate", "return", "risk", "market", "finance",
                       "RBI", "SEBI", "NSE", "BSE", "Sensex", "Nifty", "SIP", "PPF", "EPF", "NPS"]
    
    for term in financial_terms:
        if term.lower() in query.lower() and term.lower() not in keywords:
            keywords.append(term.lower())
    
    # Score chunks based on keyword matches
    chunk_scores = []
    for chunk in chunks:
        score = 0
        chunk_lower = chunk.lower()
        for keyword in keywords:
            if keyword in chunk_lower:
                score += 1
                # Bonus points for exact matches
                if f" {keyword} " in f" {chunk_lower} ":
                    score += 0.5
        chunk_scores.append(score)
    
    # Get top-k chunks
    top_k = min(top_k, len(chunks))
    if not any(chunk_scores):
        # If no matches, return random chunks
        return random.sample(chunks, top_k)
    
    # Get indices of top scores
    top_indices = sorted(range(len(chunk_scores)), key=lambda i: chunk_scores[i], reverse=True)[:top_k]
    relevant_chunks = [chunks[idx] for idx in top_indices]
    
    return relevant_chunks

def get_predefined_wisdom():
    """
    Get predefined financial wisdom for India
    
    Returns:
        str: Predefined financial wisdom
    """
    wisdom = """
# Indian Financial Wisdom

## Investment Principles
Always maintain an emergency fund covering 6-12 months of expenses before investing in market-linked products. This provides financial stability during unexpected events.

For long-term wealth creation in India, consider a diversified portfolio with a mix of equities, debt, and gold. Historically, equities have delivered superior returns over longer time horizons of 7+ years compared to traditional fixed deposits.

Tax-efficient investment planning is crucial in the Indian context. Utilize tax-saving instruments under Section 80C (ELSS, PPF, NPS) strategically not just for tax benefits but also for long-term wealth creation.

Dollar-cost averaging through SIPs (Systematic Investment Plans) in mutual funds helps mitigate market volatility and reduces the impact of timing the market, which is particularly valuable in the cyclical Indian market.

## Stock Market Wisdom
The Indian stock market has delivered approximately 12-15% CAGR over long periods despite short-term volatility. Focus on fundamentally strong companies with good governance rather than market timing.

In the Indian context, companies with consistent dividend history often indicate strong financial health and management confidence in future prospects.

Avoid concentrated sector bets in Indian markets. Sectors like IT, pharma, FMCG, banking, and manufacturing move in different cycles, providing natural diversification benefits.

Look beyond large-caps. The mid and small-cap segments of Indian markets often contain hidden gems that can deliver superior returns, though with higher volatility.

## Personal Finance Essentials
The power of compounding works most effectively when started early. A 25-year-old who invests ₹10,000 monthly until age 60 may accumulate significantly more than someone starting at 35 with the same monthly investment.

In India's high-inflation environment (averaging 5-6% historically), keeping money idle in savings accounts erodes purchasing power. Even conservative investors should consider debt mutual funds or corporate fixed deposits.

Medical insurance is not optional in India given the rising healthcare costs. A family floater policy of at least ₹10 lakh sum insured is considered prudent financial planning.

Life insurance should be purchased primarily for protection, not investment. Term insurance offering coverage of at least 10x annual income is recommended for breadwinners.

## Retirement Planning
India's demographic shift means relying only on children for old-age support is increasingly unsustainable. Building a retirement corpus is essential.

The National Pension System (NPS) offers an excellent combination of tax benefits, low costs, and market-linked returns for retirement planning in India.

Consider the rule of 4% for retirement planning - a diversified portfolio can sustainably support annual withdrawals of approximately 4% of the initial retirement corpus, adjusted for inflation.

The EPF (Employee Provident Fund) alone is rarely sufficient for retirement in India due to increasing life expectancy and inflation. Additional retirement savings through PPF, NPS, and mutual funds are typically necessary.

## Debt Management
In India, personal loans and credit card debt often carry interest rates of 12-36%, significantly higher than potential investment returns. Prioritize clearing high-interest debt before aggressive investing.

The debt-to-income ratio should ideally not exceed 40% in the Indian context, inclusive of all EMIs (home loan, car loan, personal loan, etc.).

Home loans in India offer dual benefits - tax deductions on both principal (Section 80C) and interest (Section 24), making them more efficient than many other forms of debt.

## Risk Management
Insurance should be purchased for risk management, not investment returns. In India, a comprehensive protection strategy typically includes term life insurance, health insurance, personal accident cover, and critical illness cover.

For salaried individuals in India, adequate term life insurance coverage is typically 10-15 times annual income, while business owners may require specialized solutions based on business valuation and succession planning.

Diversify across asset managers, not just asset classes. In the Indian context, having all mutual fund investments with a single AMC (Asset Management Company) introduces unnecessary concentration risk.

## Tax Planning
Tax planning should be a year-round activity, not a March rush. Strategic decisions about investments, home loans, and salary structure can optimize tax efficiency when planned well in advance.

HRA (House Rent Allowance) optimization and home loan tax benefits can together provide significant tax advantages for many Indian taxpayers.

Capital gains tax treatment varies significantly based on holding period and asset class in India. Equity investments held for more than one year qualify for favorable long-term capital gains tax treatment.

## Real Estate Investment
Real estate in India typically has high entry costs and low liquidity. REITs (Real Estate Investment Trusts) offer an alternative with lower entry barriers, professional management, and better liquidity.

Rental yields in major Indian cities typically range from 2-4%, significantly lower than residential mortgage rates. Real estate investment decisions should factor in this yield gap and not focus solely on potential appreciation.

Location remains paramount in Indian real estate values. Properties in areas with developing infrastructure, good connectivity, and proximity to employment hubs typically appreciate faster.

## Gold Investment
Gold has cultural significance in India but should be limited to 5-10% of the investment portfolio for diversification. Gold ETFs and Sovereign Gold Bonds offer more secure and efficient alternatives to physical gold.

Sovereign Gold Bonds in India provide additional benefits of 2.5% annual interest and potential tax advantages, making them superior to physical gold for investment purposes.

## Market Cycles
The Indian market follows its own cycles influenced by domestic factors, monsoons, government policies, and global cues. Understanding these cycles helps maintain a long-term perspective during market volatility.

In the Indian context, election years often introduce volatility but have limited long-term impact on fundamentally sound companies.

## Economic Context
India's demographic dividend with a young working population provides a long-term economic tailwind that creates structural growth opportunities across multiple sectors.

Rising financial inclusion in India, accelerated by initiatives like Jan Dhan accounts, UPI, and digital banking, is creating new markets and investment opportunities in financial services.
"""
    return wisdom

def get_fallback_wisdom():
    """
    Get fallback financial wisdom
    
    Returns:
        str: Fallback financial wisdom
    """
    fallback_wisdom = """## Financial Wisdom:

### Insight 1:
For long-term wealth creation in India, consider a diversified portfolio with a mix of equities, debt, and gold. Historically, equities have delivered superior returns over longer time horizons of 7+ years compared to traditional fixed deposits.

### Insight 2:
Tax-efficient investment planning is crucial in the Indian context. Utilize tax-saving instruments under Section 80C (ELSS, PPF, NPS) strategically not just for tax benefits but also for long-term wealth creation.

### Insight 3:
Always maintain an emergency fund covering 6-12 months of expenses before investing in market-linked products. This provides financial stability during unexpected events like medical emergencies or job loss, which is particularly important in the Indian economic context.
"""
    return fallback_wisdom
