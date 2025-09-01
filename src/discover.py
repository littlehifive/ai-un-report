"""Fixed Discovery module using proper UNDL API with MARCXML."""

import logging
import requests
import pandas as pd
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import re

import pymarc
from pymarc import Record

from utils import load_config, get_date_window, RateLimiter, ensure_dir

logger = logging.getLogger(__name__)

class UNDLClient:
    """Client for the UN Digital Library API based on the uploaded wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limiter = RateLimiter(config.get('delay_seconds', 5))
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UN-RAG-Research-Bot/1.0 (Educational Purpose; Respectful Crawling)'
        })
        self.api_key = os.getenv("UN_API")
        
    def search_reports(self, query: str = "reports 2025", max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for UN reports using UNDL API."""
        logger.info(f"Searching UNDL API for: {query}")
        
        try:
            # Use official UNDL API
            params = {
                "p": query,
                "format": "xml",
                "ln": "en",
                "rg": min(max_results, 200)  # API limit
            }
            
            self.rate_limiter.wait()
            
            if self.api_key:
                # Use official API v1 with authentication
                url = "https://digitallibrary.un.org/api/v1/search"
                headers = {
                    "content-type": "application/xml",
                    "Authorization": f"Token {self.api_key}"
                }
                response = self.session.get(url, params=params, headers=headers, timeout=30)
            else:
                # Fallback to search endpoint (may have limitations)
                url = "https://digitallibrary.un.org/search"
                params["of"] = "xm"  # MARCXML format
                response = self.session.get(url, params=params, timeout=30)
            
            response.raise_for_status()
            logger.debug(f"API URL: {response.url}")
            
            # Parse MARCXML response
            return self._parse_marcxml_response(response.text)
            
        except Exception as e:
            logger.error(f"UNDL API search failed: {e}")
            return []
    
    def _parse_marcxml_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse MARCXML response to extract record data."""
        try:
            # Remove namespace to simplify parsing
            clean_xml = xml_text.replace('xmlns="http://www.loc.gov/MARC21/slim"', '')
            root = ET.fromstring(clean_xml)
            
            # Handle API v1 response structure
            collection_elem = root.find('collection')
            if collection_elem is None:
                collection_elem = root
            
            # Write temporary XML file for pymarc parsing
            temp_path = Path.home() / ".undl"
            temp_path.mkdir(exist_ok=True)
            temp_file = temp_path / "temp_records.xml"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(ET.tostring(collection_elem, encoding='unicode'))
            
            # Parse using pymarc
            records = pymarc.parse_xml_to_array(str(temp_file))
            logger.info(f"Parsed {len(records)} MARC records")
            
            # Convert to our format
            result = []
            for record in records:
                record_data = self._extract_record_data(record)
                if record_data and self._is_valid_report(record_data):
                    result.append(record_data)
            
            # Cleanup temp file
            temp_file.unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse MARCXML response: {e}")
            return []
    
    def _extract_record_data(self, record: Record) -> Optional[Dict[str, Any]]:
        """Extract structured data from a MARC record."""
        try:
            # Get record ID
            record_id = self._extract_field(record, "001")
            if not record_id:
                return None
            
            # Extract basic fields
            title = self._extract_field(record, "245")
            if not title:
                # Try alternative title field
                title = self._extract_field(record, "239")
            
            if not title:
                logger.debug(f"No title found for record {record_id}")
                return None
            
            symbol = self._get_symbol(record)
            downloads = self._get_downloads(record)
            
            record_data = {
                'id': record_id,
                'title': title.strip(),
                'symbol': symbol,
                'date': self._extract_date(record),
                'organ': self._extract_organ(record, symbol),
                'language': self._extract_language(record),
                'record_url': f'https://digitallibrary.un.org/record/{record_id}',
                'file_urls': self._extract_file_urls(downloads, record_id),
                'summary': self._extract_field(record, "520"),
                'authors': self._extract_field(record, "710", "a", collection=True)
            }
            
            return record_data
            
        except Exception as e:
            logger.warning(f"Failed to extract record data: {e}")
            return None
    
    def _extract_field(self, record: Record, field: str, subfield: Optional[str] = None, collection: bool = False) -> Optional[str]:
        """Extract field from MARC record using pymarc."""
        try:
            fields = record.get_fields(field)
            if not fields:
                return None
            
            if subfield:
                values = []
                for f in fields:
                    subfield_values = f.get_subfields(subfield)
                    values.extend(subfield_values)
                
                if collection:
                    return values if values else None
                else:
                    return values[0] if values else None
            else:
                # Get formatted field
                if collection:
                    return [f.format_field() for f in fields]
                else:
                    return fields[0].format_field()
                    
        except Exception as e:
            logger.debug(f"Could not extract field {field}: {e}")
            return None
    
    def _get_symbol(self, record: Record) -> str:
        """Extract UN document symbol from MARC record."""
        # Check field 191 first (primary symbol field)
        symbol = self._extract_field(record, "191", "a")
        if symbol:
            return symbol
        
        # Check field 791 (alternative symbol field)
        symbol = self._extract_field(record, "791", "a")
        if symbol:
            return symbol
        
        # Fallback to record ID
        record_id = self._extract_field(record, "001")
        return f"UNDoc-{record_id}" if record_id else "Unknown"
    
    def _get_downloads(self, record: Record) -> Dict[str, str]:
        """Extract download URLs from MARC record."""
        downloads = {}
        
        # Field 856 contains URLs
        url_fields = record.get_fields("856")
        for field in url_fields:
            url = field.get_subfields("u")  # URL subfield
            desc = field.get_subfields("y")  # Description subfield
            
            if url and desc:
                downloads[desc[0]] = url[0]
            elif url:
                # Guess language from URL pattern
                url_str = url[0]
                if "-EN." in url_str.upper():
                    downloads["English"] = url_str
                elif "-ES." in url_str.upper():
                    downloads["Spanish"] = url_str
                elif "-FR." in url_str.upper():
                    downloads["French"] = url_str
                else:
                    downloads["Document"] = url_str
        
        return downloads
    
    def _extract_date(self, record: Record) -> str:
        """Extract publication date from MARC record."""
        # Field 269 contains publication date
        date_str = self._extract_field(record, "269")
        if date_str:
            # Try to parse and normalize date
            try:
                # Handle various date formats
                if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    return date_str
                elif re.match(r'\d{4}', date_str):
                    return f"{date_str}-01-01"
            except:
                pass
        
        # Field 260 publication info
        pub_info = self._extract_field(record, "260", "c")
        if pub_info:
            try:
                year_match = re.search(r'(\d{4})', pub_info)
                if year_match:
                    return f"{year_match.group(1)}-01-01"
            except:
                pass
        
        # Default to current year for recent searches
        return "2025-01-01"
    
    def _extract_language(self, record: Record) -> str:
        """Extract language from MARC record."""
        # Check control field 008 positions 35-37 for language code
        control_008 = self._extract_field(record, "008")
        if control_008 and len(control_008) > 37:
            lang_code = control_008[35:38].strip()
            if lang_code == "eng":
                return "en"
            elif lang_code == "spa":
                return "es"
            elif lang_code == "fre":
                return "fr"
        
        # Default to English
        return "en"
    
    def _extract_organ(self, record: Record, symbol: str) -> str:
        """Extract UN organ from MARC record and symbol."""
        # Check field 710 for corporate bodies
        corporate = self._extract_field(record, "710", "a")
        if corporate:
            if "General Assembly" in corporate:
                return "General Assembly"
            elif "Security Council" in corporate:
                return "Security Council"
            elif "Economic and Social Council" in corporate:
                return "Economic and Social Council"
            elif "Secretary-General" in corporate:
                return "Secretary-General"
        
        # Determine from symbol pattern
        if symbol.startswith('A/'):
            return 'General Assembly'
        elif symbol.startswith('S/'):
            return 'Security Council'
        elif symbol.startswith('E/'):
            return 'Economic and Social Council'
        else:
            return 'UN System'
    
    def _extract_file_urls(self, downloads: Dict[str, str], record_id: str) -> List[str]:
        """Extract and prioritize file URLs."""
        urls = []
        
        # Prioritize English PDFs
        for desc, url in downloads.items():
            if "English" in desc or "-EN." in url.upper():
                urls.insert(0, url)  # Put English first
            elif ".pdf" in url.lower():
                urls.append(url)
        
        # Add record page as fallback
        if not urls:
            urls.append(f"https://digitallibrary.un.org/record/{record_id}")
        
        return urls
    
    def _is_valid_report(self, record_data: Dict[str, Any]) -> bool:
        """Check if record is a valid report for our purposes."""
        title = record_data.get('title', '').lower()
        symbol = record_data.get('symbol', '')
        
        # Must have title
        if not title:
            return False
        
        # Must be English
        if record_data.get('language') != 'en':
            return False
        
        # Skip if no downloadable files
        file_urls = record_data.get('file_urls', [])
        if not any('.pdf' in url.lower() for url in file_urls):
            logger.debug(f"Skipping {symbol} - no PDF files found")
            return False
        
        # Check for report-like content
        report_keywords = [
            'report', 'annual', 'progress', 'implementation', 
            'review', 'situation', 'analysis', 'study'
        ]
        
        if any(keyword in title for keyword in report_keywords):
            return True
        
        # Accept documents with standard UN symbols
        if re.match(r'[A-Z]/\d+', symbol):
            return True
        
        return False


class UNReportDiscoverer:
    """Enhanced discoverer using proper UNDL API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = UNDLClient(config)
        self.start_date, self.end_date = get_date_window(config['date_window_days'])
        
    def discover_all(self) -> List[Dict[str, Any]]:
        """Main discovery method with comprehensive search strategies."""
        logger.info("Starting comprehensive UN reports discovery...")
        
        all_reports = []
        
        # Greatly expanded search queries for broader coverage
        queries = [
            # Year-based searches
            "2025 english",
            "reports 2025",
            "documents 2025",
            
            # UN body searches
            "General Assembly 2025",
            "Security Council 2025", 
            "ECOSOC 2025",
            "Economic and Social Council 2025",
            "Secretary-General 2025",
            "Secretariat 2025",
            
            # Report type searches
            "annual report 2025",
            "progress report 2025", 
            "situation report 2025",
            "implementation 2025",
            "review 2025",
            
            # Thematic searches
            "sustainable development 2025",
            "climate 2025",
            "peacekeeping 2025",
            "humanitarian 2025",
            "human rights 2025",
            "decolonization 2025",
            
            # Document series searches
            "A/80/ 2025",
            "A/79/ 2025", 
            "S/2025/",
            "E/2025/",
            "DP/2025/",
        ]
        
        for query in queries:
            try:
                reports = self.client.search_reports(query, max_results=50)  # Increased per query
                logger.info(f"Query '{query}' found {len(reports)} reports")
                
                # Deduplicate by symbol
                existing_symbols = {r.get('symbol') for r in all_reports}
                new_reports = [r for r in reports if r.get('symbol') not in existing_symbols]
                all_reports.extend(new_reports)
                
                # Rate limiting between queries
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                continue
        
        logger.info(f"Discovery complete. Found {len(all_reports)} total reports with proper file URLs.")
        return all_reports
    
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


def main():
    """Main function for standalone execution."""
    from utils import setup_logging
    
    setup_logging()
    config = load_config()
    
    discoverer = UNReportDiscoverer(config)
    reports = discoverer.discover_all()
    
    # Save results
    output_file = config['paths']['records_file']
    discoverer.save_records(reports, output_file)
    
    print(f"Discovery complete! Found {len(reports)} reports with proper file URLs.")
    print(f"Results saved to {output_file}")
    
    # Show sample of what we found
    if reports:
        print("\nSample reports:")
        for i, report in enumerate(reports[:3]):
            print(f"{i+1}. {report['title'][:80]}...")
            print(f"   Symbol: {report['symbol']}")
            print(f"   Files: {len(report['file_urls'])} URL(s)")
            if report['file_urls']:
                english_files = [url for url in report['file_urls'] if '-EN.' in url.upper() or 'english' in url.lower()]
                if english_files:
                    print(f"   English PDF: {english_files[0]}")
            print()

if __name__ == "__main__":
    main()