"""Improved Discovery module using proper UNDL API with correct URL extraction."""

import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import re

# Add the undl-main package to the path
undl_path = Path(__file__).parent.parent / "help" / "undl-main"
sys.path.insert(0, str(undl_path))

import pandas as pd
from undl.client import UNDLClient

from utils import load_config, get_date_window, RateLimiter, ensure_dir

logger = logging.getLogger(__name__)

class ImprovedUNDLClient:
    """Improved client using the undl-main package with proper URL extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = UNDLClient(verbose=True)
        self.rate_limiter = RateLimiter(config.get('delay_seconds', 5))
        
    def search_reports(self, query: str = "reports 2025", max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for UN reports using the proper UNDL client."""
        logger.info(f"Searching UNDL API for: {query}")
        
        try:
            # Use the proper UNDL client
            result = self.client.query(
                prompt=query,
                outputFormat="marcxml",
                lang="en"
            )
            
            if not result or 'records' not in result:
                logger.warning(f"No results found for query: {query}")
                return []
            
            logger.info(f"Found {len(result['records'])} records for query: {query}")
            
            # Convert to our format and filter valid reports
            valid_reports = []
            for record in result['records']:
                report_data = self._convert_to_report_format(record)
                if report_data and self._is_valid_report(report_data):
                    valid_reports.append(report_data)
            
            logger.info(f"Filtered to {len(valid_reports)} valid reports")
            return valid_reports
            
        except Exception as e:
            logger.error(f"UNDL API search failed: {e}")
            return []
    
    def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific record by ID."""
        try:
            result = self.client.queryById(record_id)
            if result and 'records' in result and len(result['records']) > 0:
                record = result['records'][0]
                report_data = self._convert_to_report_format(record)
                if report_data and self._is_valid_report(report_data):
                    return report_data
            return None
        except Exception as e:
            logger.error(f"Failed to get record {record_id}: {e}")
            return None
    
    def _convert_to_report_format(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert UNDL record format to our report format."""
        try:
            # Extract basic fields
            record_id = record.get('id')
            if not record_id:
                return None
            
            title = record.get('title')
            if not title:
                return None
            
            symbol = record.get('symbol')
            if not symbol:
                symbol = f"UNDoc-{record_id}"
            
            # Extract file URLs with proper validation
            file_urls = self._extract_valid_file_urls(record, record_id)
            
            # Only include if we have valid file URLs
            if not file_urls:
                logger.debug(f"No valid file URLs found for {symbol}")
                return None
            
            report_data = {
                'id': record_id,
                'title': title.strip(),
                'symbol': symbol,
                'date': record.get('publication_date', '2025-01-01'),
                'organ': self._extract_organ_from_symbol(symbol),
                'language': 'en',  # We're filtering for English
                'record_url': f'https://digitallibrary.un.org/record/{record_id}',
                'file_urls': file_urls,
                'summary': record.get('summary', ''),
                'authors': record.get('authors', [])
            }
            
            return report_data
            
        except Exception as e:
            logger.warning(f"Failed to convert record: {e}")
            return None
    
    def _extract_valid_file_urls(self, record: Dict[str, Any], record_id: str) -> List[str]:
        """Extract and validate file URLs from the record."""
        urls = []
        
        # Get downloads from the record
        downloads = record.get('downloads', {})
        
        # Process downloads dictionary
        for lang, url in downloads.items():
            if url and isinstance(url, str):
                # Validate URL format
                if self._is_valid_pdf_url(url):
                    urls.append(url)
                    logger.debug(f"Added valid URL for {lang}: {url}")
        
        # If no downloads found, try to construct from record ID
        if not urls:
            # Try common URL patterns
            potential_urls = [
                f"https://digitallibrary.un.org/record/{record_id}/files/{record_id}-EN.pdf",
                f"https://digitallibrary.un.org/record/{record_id}/files/{record_id}.pdf"
            ]
            
            for url in potential_urls:
                if self._test_url_accessibility(url):
                    urls.append(url)
                    logger.info(f"Found accessible URL: {url}")
                    break
        
        return urls
    
    def _is_valid_pdf_url(self, url: str) -> bool:
        """Check if URL appears to be a valid PDF URL."""
        if not url or not isinstance(url, str):
            return False
        
        # Must be HTTP/HTTPS
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Should contain digitallibrary.un.org
        if 'digitallibrary.un.org' not in url:
            return False
        
        # Should end with .pdf or contain /files/
        if not (url.endswith('.pdf') or '/files/' in url):
            return False
        
        return True
    
    def _test_url_accessibility(self, url: str) -> bool:
        """Test if a URL is actually accessible."""
        try:
            import requests
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    def _extract_organ_from_symbol(self, symbol: str) -> str:
        """Extract UN organ from document symbol."""
        if not symbol:
            return 'UN System'
        
        symbol_upper = symbol.upper()
        if symbol_upper.startswith('A/'):
            return 'General Assembly'
        elif symbol_upper.startswith('S/'):
            return 'Security Council'
        elif symbol_upper.startswith('E/'):
            return 'Economic and Social Council'
        elif symbol_upper.startswith('JIU/'):
            return 'Joint Inspection Unit'
        elif symbol_upper.startswith('UNCTAD/'):
            return 'UNCTAD'
        elif symbol_upper.startswith('UNW/'):
            return 'UN Women'
        else:
            return 'UN System'
    
    def _is_valid_report(self, report_data: Dict[str, Any]) -> bool:
        """Check if record is a valid report for our purposes."""
        title = report_data.get('title', '').lower()
        symbol = report_data.get('symbol', '')
        file_urls = report_data.get('file_urls', [])
        
        # Must have title and symbol
        if not title or not symbol:
            return False
        
        # Must have downloadable files
        if not file_urls:
            return False
        
        # Check for report-like content
        report_keywords = [
            'report', 'annual', 'progress', 'implementation', 
            'review', 'situation', 'analysis', 'study', 'overview'
        ]
        
        if any(keyword in title for keyword in report_keywords):
            return True
        
        # Accept documents with standard UN symbols
        if re.match(r'[A-Z]+/\d+', symbol):
            return True
        
        return False


class ImprovedUNReportDiscoverer:
    """Improved discoverer using the enhanced UNDL client."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = ImprovedUNDLClient(config)
        self.start_date, self.end_date = get_date_window(config['date_window_days'])
        
    def discover_all(self) -> List[Dict[str, Any]]:
        """Main discovery method using improved API."""
        logger.info("Starting improved UN reports discovery...")
        
        all_reports = []
        
        # Search for 2025 reports with specific queries
        queries = [
            "reports 2025 english",
            "General Assembly 2025",
            "Security Council 2025",
            "Secretary-General report 2025",
            "UNCTAD 2025",
            "UN Women 2025"
        ]
        
        for query in queries:
            try:
                reports = self.client.search_reports(query, max_results=30)
                logger.info(f"Query '{query}' found {len(reports)} reports")
                
                # Deduplicate by symbol
                existing_symbols = {r.get('symbol') for r in all_reports}
                new_reports = [r for r in reports if r.get('symbol') not in existing_symbols]
                all_reports.extend(new_reports)
                
                logger.info(f"Added {len(new_reports)} new reports from '{query}'")
                
                # Rate limiting between queries
                time.sleep(3)
                
            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                continue
        
        logger.info(f"Discovery complete. Found {len(all_reports)} total reports with valid file URLs.")
        return all_reports
    
    def discover_by_symbols(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Discover reports by specific document symbols."""
        logger.info(f"Discovering reports for {len(symbols)} specific symbols...")
        
        reports = []
        for symbol in symbols:
            try:
                # Try to find the record ID from the symbol
                # This is a simplified approach - in practice you might need to search
                record_id = self._extract_record_id_from_symbol(symbol)
                if record_id:
                    report = self.client.get_record_by_id(record_id)
                    if report:
                        reports.append(report)
                        logger.info(f"Found report for {symbol}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to discover {symbol}: {e}")
                continue
        
        return reports
    
    def _extract_record_id_from_symbol(self, symbol: str) -> Optional[str]:
        """Extract record ID from symbol - this is a placeholder for now."""
        # In practice, you might need to search for the symbol to get the record ID
        # For now, return None to indicate we need to search
        return None
    
    def save_records(self, reports: List[Dict[str, Any]], output_file: str) -> None:
        """Save discovered records to parquet file."""
        if not reports:
            logger.warning("No reports to save")
            return
            
        df = pd.DataFrame(reports)
        
        # Ensure output directory exists
        output_path = Path(output_file)
        ensure_dir(output_path.parent)
        
        # Save to parquet
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(reports)} records to {output_file}")
        
        # Also save a summary
        summary_file = output_path.with_suffix('.summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"UN Reports Discovery Summary\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Reports: {len(reports)}\n\n")
            
            for i, report in enumerate(reports, 1):
                f.write(f"{i}. {report['symbol']}\n")
                f.write(f"   Title: {report['title'][:80]}...\n")
                f.write(f"   Files: {len(report['file_urls'])} URL(s)\n")
                if report['file_urls']:
                    f.write(f"   First URL: {report['file_urls'][0]}\n")
                f.write("\n")


def main():
    """Main function for standalone execution."""
    from utils import setup_logging
    
    setup_logging()
    config = load_config()
    
    discoverer = ImprovedUNReportDiscoverer(config)
    reports = discoverer.discover_all()
    
    # Save results
    output_file = config['paths']['records_file']
    discoverer.save_records(reports, output_file)
    
    print(f"Discovery complete!")
    print(f"Reports found: {len(reports)}")
    print(f"Results saved to {output_file}")
    
    # Show sample of what we found
    if reports:
        print("\nSample reports:")
        for i, report in enumerate(reports[:5]):
            print(f"{i+1}. {report['symbol']}")
            print(f"   Title: {report['title'][:80]}...")
            print(f"   Files: {len(report['file_urls'])} URL(s)")
            if report['file_urls']:
                print(f"   First URL: {report['file_urls'][0]}")
            print()


if __name__ == "__main__":
    main()
