"""Test the improved UN reports discovery and fetch system."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from utils import load_config, setup_logging
from discover_improved import ImprovedUNReportDiscoverer
from fetch_improved import ImprovedUNReportFetcher

def test_improved_system():
    """Test the improved discovery and fetching system."""
    
    print("🇺🇳 Testing Improved UN Reports System")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    config = load_config()
    
    # Step 1: Test improved discovery
    print("\n1. Testing improved discovery...")
    discoverer = ImprovedUNReportDiscoverer(config)
    
    # Test with a specific query first
    print("   Testing search for 'UNCTAD 2025'...")
    try:
        reports = discoverer.client.search_reports("UNCTAD 2025", max_results=5)
        print(f"   Found {len(reports)} reports")
        
        if reports:
            print("   Sample report:")
            sample = reports[0]
            print(f"   - Symbol: {sample.get('symbol', 'N/A')}")
            print(f"   - Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"   - File URLs: {len(sample.get('file_urls', []))}")
            if sample.get('file_urls'):
                print(f"   - First URL: {sample.get('file_urls')[0]}")
        
    except Exception as e:
        print(f"   Discovery test failed: {e}")
        return False
    
    # Step 2: Test URL validation
    print("\n2. Testing URL validation...")
    fetcher = ImprovedUNReportFetcher(config)
    
    # Test with a known good URL
    test_url = "https://digitallibrary.un.org/record/4084949/files/UNCTAD_TIR_2025-EN.pdf"
    print(f"   Testing URL: {test_url}")
    
    try:
        validation = fetcher._validate_url_before_download(test_url)
        print(f"   Validation result: {validation['is_valid']}")
        if not validation['is_valid']:
            print(f"   Error: {validation['error_msg']}")
        else:
            print(f"   Content type: {validation['content_type']}")
            print(f"   File size: {validation['file_size']} bytes")
    except Exception as e:
        print(f"   URL validation test failed: {e}")
    
    # Step 3: Test with a few specific reports
    print("\n3. Testing with specific reports...")
    
    # Create a test records file with known good reports
    test_records = [
        {
            'symbol': 'UNCTAD/TIR/2025',
            'title': 'Technology and innovation report 2025',
            'file_urls': ['https://digitallibrary.un.org/record/4084949/files/UNCTAD_TIR_2025-EN.pdf'],
            'language': 'en'
        },
        {
            'symbol': 'JIU/REP/2025/1',
            'title': 'Review of management and administration in UNEP',
            'file_urls': ['https://digitallibrary.un.org/record/4087110/files/JIU_REP_2025_1_%5EExpanded_report%5E-EN.pdf'],
            'language': 'en'
        }
    ]
    
    test_records_file = "data/test_records_improved.parquet"
    test_df = pd.DataFrame(test_records)
    test_df.to_parquet(test_records_file)
    print(f"   Created test records file: {test_records_file}")
    
    # Test fetch with these records
    print("   Testing fetch with test records...")
    try:
        result = fetcher.fetch_from_records(test_records_file)
        print(f"   Fetch result: {result}")
        
        if result.get('successful_downloads', 0) > 0:
            print("   ✅ Successfully downloaded files:")
            raw_path = Path(config['paths']['raw_data'])
            for file_path in raw_path.glob("*.pdf"):
                if file_path.stat().st_size > 0:
                    size_kb = file_path.stat().st_size / 1024
                    print(f"      - {file_path.name} ({size_kb:.1f} KB)")
        else:
            print("   ❌ No files were downloaded successfully")
            
    except Exception as e:
        print(f"   Fetch test failed: {e}")
        return False
    
    # Step 4: Show final status
    print("\n4. Final status...")
    try:
        status = fetcher.get_download_status()
        print(f"   Total files: {status['total_files']}")
        print(f"   Successful: {status['successful']}")
        print(f"   Failed: {status['failed']}")
        print(f"   Total size: {status['total_size_mb']} MB")
    except Exception as e:
        print(f"   Status check failed: {e}")
    
    print("\n🎯 Test completed!")
    return True

def test_specific_url_validation():
    """Test URL validation with specific problematic URLs."""
    
    print("\n🔍 Testing specific URL validation...")
    
    setup_logging()
    config = load_config()
    fetcher = ImprovedUNReportFetcher(config)
    
    # Test URLs that might be problematic
    test_urls = [
        "https://digitallibrary.un.org/record/4084949/files/UNCTAD_TIR_2025-EN.pdf",
        "https://digitallibrary.un.org/record/4087110/files/JIU_REP_2025_1_%5EExpanded_report%5E-EN.pdf",
        "https://digitallibrary.un.org/record/999999/files/nonexistent.pdf",  # Should fail
        "https://digitallibrary.un.org/record/4084949",  # Record page, not PDF
    ]
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n   Test {i}: {url}")
        try:
            validation = fetcher._validate_url_before_download(url)
            print(f"      Valid: {validation['is_valid']}")
            print(f"      Content type: {validation['content_type']}")
            if validation['error_msg']:
                print(f"      Error: {validation['error_msg']}")
            if validation['redirects_to']:
                print(f"      Redirects to: {validation['redirects_to']}")
        except Exception as e:
            print(f"      Validation failed: {e}")

if __name__ == "__main__":
    print("Starting improved system tests...")
    
    # Run main test
    success = test_improved_system()
    
    if success:
        # Run additional URL validation tests
        test_specific_url_validation()
    
    print("\n🏁 All tests completed!")
