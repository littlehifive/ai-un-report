"""Test the robust UN reports discovery and fetch system."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from utils import load_config, setup_logging
from discover_robust import RobustUNReportDiscoverer
from fetch_improved import ImprovedUNReportFetcher

def test_robust_system():
    """Test the robust discovery and fetching system."""
    
    print("🇺🇳 Testing Robust UN Reports System")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    config = load_config()
    
    # Step 1: Test robust discovery
    print("\n1. Testing robust discovery...")
    discoverer = RobustUNReportDiscoverer(config)
    
    # Test with a specific query first
    print("   Testing robust search for 'UNCTAD 2025'...")
    try:
        reports = discoverer.client.search_reports_robust("UNCTAD 2025", max_results=5)
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
        print(f"   Robust discovery test failed: {e}")
        return False
    
    # Step 2: Test fallback methods
    print("\n2. Testing fallback methods...")
    
    # Test known patterns method
    print("   Testing known patterns method...")
    try:
        known_reports = discoverer.client._try_known_patterns("UNCTAD 2025", 3)
        print(f"   Known patterns found {len(known_reports)} reports")
        
        if known_reports:
            print("   Sample from known patterns:")
            sample = known_reports[0]
            print(f"   - Symbol: {sample.get('symbol', 'N/A')}")
            print(f"   - Title: {sample.get('title', 'N/A')[:60]}...")
        
    except Exception as e:
        print(f"   Known patterns test failed: {e}")
    
    # Step 3: Test URL validation with the improved fetcher
    print("\n3. Testing URL validation...")
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
            if validation['redirects_to']:
                print(f"   Redirects to: {validation['redirects_to']}")
    except Exception as e:
        print(f"   URL validation test failed: {e}")
    
    # Step 4: Test with discovered reports
    print("\n4. Testing with discovered reports...")
    
    # Use the reports we discovered
    if reports:
        # Create a test records file
        test_records_file = "data/test_records_robust.parquet"
        test_df = pd.DataFrame(reports)
        test_df.to_parquet(test_records_file)
        print(f"   Created test records file: {test_records_file}")
        print(f"   Records to test: {len(reports)}")
        
        # Test fetch with these records
        print("   Testing fetch with discovered records...")
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
    else:
        print("   No reports to test with")
    
    # Step 5: Show final status
    print("\n5. Final status...")
    try:
        status = fetcher.get_download_status()
        print(f"   Total files: {status['total_files']}")
        print(f"   Successful: {status['successful']}")
        print(f"   Failed: {status['failed']}")
        print(f"   Total size: {status['total_size_mb']} MB")
    except Exception as e:
        print(f"   Status check failed: {e}")
    
    print("\n🎯 Robust test completed!")
    return True

def test_url_validation_comprehensive():
    """Test comprehensive URL validation."""
    
    print("\n🔍 Testing comprehensive URL validation...")
    
    setup_logging()
    config = load_config()
    fetcher = ImprovedUNReportFetcher(config)
    
    # Test various types of URLs
    test_cases = [
        {
            'name': 'Valid PDF URL',
            'url': 'https://digitallibrary.un.org/record/4084949/files/UNCTAD_TIR_2025-EN.pdf',
            'should_be_valid': True
        },
        {
            'name': 'Another Valid PDF URL',
            'url': 'https://digitallibrary.un.org/record/4087110/files/JIU_REP_2025_1_%5EExpanded_report%5E-EN.pdf',
            'should_be_valid': True
        },
        {
            'name': 'Non-existent PDF',
            'url': 'https://digitallibrary.un.org/record/999999/files/nonexistent.pdf',
            'should_be_valid': False
        },
        {
            'name': 'Record page (not PDF)',
            'url': 'https://digitallibrary.un.org/record/4084949',
            'should_be_valid': False
        },
        {
            'name': 'Invalid domain',
            'url': 'https://example.com/file.pdf',
            'should_be_valid': False
        },
        {
            'name': 'Non-PDF file',
            'url': 'https://digitallibrary.un.org/record/4084949/files/document.txt',
            'should_be_valid': False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['name']}")
        print(f"      URL: {test_case['url']}")
        
        try:
            validation = fetcher._validate_url_before_download(test_case['url'])
            print(f"      Valid: {validation['is_valid']}")
            print(f"      Expected: {test_case['should_be_valid']}")
            
            if validation['is_valid'] == test_case['should_be_valid']:
                print(f"      ✅ PASS")
            else:
                print(f"      ❌ FAIL - Expected {test_case['should_be_valid']}, got {validation['is_valid']}")
            
            if validation['content_type']:
                print(f"      Content type: {validation['content_type']}")
            if validation['error_msg']:
                print(f"      Error: {validation['error_msg']}")
            if validation['redirects_to']:
                print(f"      Redirects to: {validation['redirects_to']}")
                
        except Exception as e:
            print(f"      Validation failed: {e}")
            print(f"      ❌ FAIL - Exception occurred")

def test_discovery_fallbacks():
    """Test the discovery fallback methods."""
    
    print("\n🔄 Testing discovery fallback methods...")
    
    setup_logging()
    config = load_config()
    discoverer = RobustUNReportDiscoverer(config)
    
    # Test each fallback method individually
    methods = [
        ('Official API', discoverer.client._try_official_api),
        ('Search Endpoint', discoverer.client._try_search_endpoint),
        ('Known Patterns', discoverer.client._try_known_patterns)
    ]
    
    for method_name, method_func in methods:
        print(f"\n   Testing {method_name}...")
        try:
            reports = method_func("UNCTAD 2025", 3)
            print(f"      Found {len(reports)} reports")
            
            if reports:
                print(f"      Sample: {reports[0].get('symbol', 'N/A')} - {reports[0].get('title', 'N/A')[:50]}...")
            else:
                print(f"      No reports found")
                
        except Exception as e:
            print(f"      Method failed: {e}")

if __name__ == "__main__":
    print("Starting robust system tests...")
    
    # Run main test
    success = test_robust_system()
    
    if success:
        # Run additional tests
        test_url_validation_comprehensive()
        test_discovery_fallbacks()
    
    print("\n🏁 All robust tests completed!")
