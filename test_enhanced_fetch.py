"""Test enhanced fetching with August 2025 reports."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from utils import load_config
from discover import UNReportDiscoverer
from fetch import UNReportFetcher

def test_enhanced_fetch():
    """Test the enhanced discovery and fetching system."""
    
    print("🇺🇳 Testing Enhanced UN Reports System")
    print("=" * 50)
    
    config = load_config()
    
    # Step 1: Get seed reports (skip slow API expansion)
    print("\n1. Getting seed reports...")
    discoverer = UNReportDiscoverer(config)
    seed_reports = discoverer._get_seed_reports()
    
    print(f"   Found {len(seed_reports)} seed reports")
    
    # Just test first 3 reports to validate system quickly
    test_reports = seed_reports[:3]
    
    # Create DataFrame and save as records
    records_df = pd.DataFrame(test_reports)
    records_file = "data/test_records.parquet"
    records_df.to_parquet(records_file)
    print(f"   Saved test records to {records_file}")
    
    # Display what we're testing
    print("\n📋 Testing these reports:")
    for i, report in enumerate(test_reports, 1):
        print(f"   {i}. {report['symbol']} - {report['title'][:60]}...")
        print(f"      URLs: {len(report['file_urls'])} fallback options")
    
    # Step 2: Test enhanced fetch
    print(f"\n2. Testing enhanced fetch with {len(test_reports)} reports...")
    fetcher = UNReportFetcher(config)
    result = fetcher.fetch_from_records(records_file)
    
    print(f"\n📊 Fetch Results:")
    print(f"   Success: {result.get('successful_downloads', 0)}")
    print(f"   Failed: {result.get('failed_downloads', 0)}")
    print(f"   Total files: {result.get('total_files', 0)}")
    
    # Show file details
    if result.get('successful_downloads', 0) > 0:
        print(f"\n✅ Successfully downloaded files:")
        raw_path = Path(config['paths']['raw_data'])
        for file_path in raw_path.glob("*.pdf"):
            size_kb = file_path.stat().st_size / 1024
            print(f"   - {file_path.name} ({size_kb:.1f} KB)")
    
    return result

if __name__ == "__main__":
    test_enhanced_fetch()