"""Check how many UN reports are available for 2025."""

import requests
import json
from datetime import datetime
import time

def search_undl_reports_2025():
    """Search UN Digital Library for 2025 reports."""
    
    # Try different search approaches
    results = {}
    
    # Method 1: Direct search API (if available)
    print("🔍 Searching UN Digital Library for 2025 reports...")
    
    try:
        # Search for documents from 2025
        search_url = "https://digitallibrary.un.org/search"
        params = {
            'ln': 'en',
            'p': 'year:2025 AND subject:"Reports"',  # Search for reports from 2025
            'of': 'json',
            'rg': 100  # Get up to 100 results
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        print(f"Search API response: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                results['search_api'] = len(data.get('records', []))
            except json.JSONDecodeError:
                print("Search API returned non-JSON response")
        
    except Exception as e:
        print(f"Search API failed: {e}")
    
    # Method 2: Check recent record IDs for 2025 content
    print("\n🔍 Checking recent record IDs for 2025 content...")
    
    # Start from a recent ID and check backwards
    recent_records_2025 = 0
    sample_size = 50
    
    for record_id in range(4080000, 4080000 + sample_size):  # Sample recent IDs
        try:
            url = f"https://digitallibrary.un.org/record/{record_id}?of=recjson"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Check if it's from 2025
                    date_field = data.get('date', '')
                    if isinstance(date_field, list):
                        date_field = date_field[0] if date_field else ''
                    
                    if '2025' in str(date_field):
                        recent_records_2025 += 1
                        
                except json.JSONDecodeError:
                    pass
            
            # Rate limit
            time.sleep(0.1)
            
        except Exception:
            continue
    
    results['sampled_2025'] = recent_records_2025
    
    # Method 3: Estimate based on current year pattern
    print("\n📊 Estimating 2025 reports...")
    
    # The UN publishes thousands of documents per year
    # Rough estimates based on typical UN publication patterns:
    estimated_total_2025 = 5000  # Conservative estimate for full year
    current_month = datetime.now().month
    estimated_so_far_2025 = int(estimated_total_2025 * (current_month / 12))
    
    results['estimated_total_2025'] = estimated_total_2025
    results['estimated_current_2025'] = estimated_so_far_2025
    
    return results

def estimate_api_costs(num_documents):
    """Estimate OpenAI API costs for RAG system with given number of documents."""
    
    # Assumptions
    avg_doc_size_chars = 50000  # Average document size
    chars_per_chunk = 4000      # Chunk size
    chunks_per_doc = avg_doc_size_chars // chars_per_chunk
    
    total_chunks = num_documents * chunks_per_doc
    
    # OpenAI pricing (as of 2024)
    embedding_cost_per_1k_tokens = 0.00002  # text-embedding-3-small
    chat_cost_per_1k_input_tokens = 0.00015  # gpt-3.5-turbo input
    chat_cost_per_1k_output_tokens = 0.0002  # gpt-3.5-turbo output
    
    # Estimate tokens (rough: 1 token ≈ 4 chars)
    tokens_per_chunk = chars_per_chunk / 4
    
    # One-time indexing cost
    total_embedding_tokens = total_chunks * tokens_per_chunk
    indexing_cost = (total_embedding_tokens / 1000) * embedding_cost_per_1k_tokens
    
    # Per-query costs (assume 5 chunks retrieved per query, 500 token response)
    input_tokens_per_query = 5 * tokens_per_chunk + 100  # 5 chunks + query
    output_tokens_per_query = 500  # Response length
    
    cost_per_query = ((input_tokens_per_query / 1000) * chat_cost_per_1k_input_tokens + 
                     (output_tokens_per_query / 1000) * chat_cost_per_1k_output_tokens)
    
    # Usage scenarios
    scenarios = {
        "light_usage": {"queries_per_month": 50, "months": 12},
        "moderate_usage": {"queries_per_month": 200, "months": 12}, 
        "heavy_usage": {"queries_per_month": 500, "months": 12}
    }
    
    print(f"\n💰 API Cost Estimates for {num_documents:,} documents:")
    print(f"📊 Estimated chunks: {total_chunks:,}")
    print(f"🔧 One-time indexing cost: ${indexing_cost:.2f}")
    print(f"💬 Cost per query: ${cost_per_query:.4f}")
    
    print("\n📈 Annual usage scenarios:")
    for scenario, params in scenarios.items():
        annual_queries = params["queries_per_month"] * params["months"]
        annual_query_cost = annual_queries * cost_per_query
        total_annual_cost = indexing_cost + annual_query_cost
        
        print(f"  {scenario.replace('_', ' ').title()}:")
        print(f"    - {params['queries_per_month']} queries/month = {annual_queries} queries/year")
        print(f"    - Query costs: ${annual_query_cost:.2f}")
        print(f"    - Total annual cost: ${total_annual_cost:.2f}")

if __name__ == "__main__":
    print("🇺🇳 UN Reports 2025 Analysis")
    print("=" * 50)
    
    # Search for 2025 reports
    results = search_undl_reports_2025()
    
    print(f"\n📋 Results Summary:")
    for method, count in results.items():
        print(f"  {method}: {count}")
    
    # Use the most reliable estimate
    if 'estimated_current_2025' in results:
        current_estimate = results['estimated_current_2025']
        full_year_estimate = results['estimated_total_2025']
    else:
        current_estimate = 1000  # Fallback
        full_year_estimate = 5000
    
    print(f"\n🎯 Best Estimates:")
    print(f"  Reports available now (2025 so far): ~{current_estimate:,}")
    print(f"  Expected by end of 2025: ~{full_year_estimate:,}")
    
    # Estimate costs for both scenarios
    print("\n" + "=" * 50)
    print("COST ANALYSIS")
    print("=" * 50)
    
    print(f"\n🔍 Scenario 1: Current 2025 reports (~{current_estimate:,})")
    estimate_api_costs(current_estimate)
    
    print(f"\n🔍 Scenario 2: Full year 2025 reports (~{full_year_estimate:,})")  
    estimate_api_costs(full_year_estimate)