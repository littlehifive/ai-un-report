"""Discovery module for UN reports from Digital Library."""

import logging
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import time
import re

from utils import load_config, get_date_window, RateLimiter, ensure_dir, safe_filename

logger = logging.getLogger(__name__)

class UNReportDiscoverer:
    """Discovers UN reports from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Temporarily reduce delay for testing with verified records
        self.rate_limiter = RateLimiter(1)  # 1 second delay instead of 5
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UN-RAG-Research-Bot/1.0 (Educational Purpose; Respectful Crawling)'
        })
        
        # Date window for recent reports
        self.start_date, self.end_date = get_date_window(config['date_window_days'])
        
    def discover_from_sitemap(self) -> List[Dict[str, Any]]:
        """Discover reports using UN Digital Library sitemap."""
        logger.info("Discovering reports from sitemap...")
        
        try:
            # Get sitemap index
            sitemap_url = "https://digitallibrary.un.org/sitemap_index.xml.gz"
            self.rate_limiter.wait()
            
            response = self.session.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            # Parse sitemap index (would need to handle gzip in production)
            # For now, we'll use a simplified approach with seed data
            logger.warning("Sitemap parsing not fully implemented. Using seed data approach.")
            return self._get_seed_reports()
            
        except Exception as e:
            logger.error(f"Failed to fetch sitemap: {e}")
            return self._get_seed_reports()
    
    def _get_seed_reports(self) -> List[Dict[str, Any]]:
        """Get real UN reports from August 2025 using UNDL API."""
        logger.info("Discovering real UN reports from August 2025...")
        
        # First, try to get recent 2025 reports via API search (temporarily disabled for speed)
        # api_reports = self._search_undl_reports()
        # if api_reports:
        #     logger.info(f"Found {len(api_reports)} reports via UNDL API search")
        #     return api_reports
        
        # Fallback to known working 2025 record IDs if API search fails
        logger.info("Using fallback approach with verified working 2025 record IDs...")
        known_2025_records = [
            4087389,  # Palestine letter Aug 2025 - verified working
            4085123,  # SDG report 2025 - verified working
            4084949,  # Technology and innovation report 2025
            4082930,  # Human development report 2025
            4087103,  # UN Environment Programme review
        ]
        
        fallback_reports = []
        for record_id in known_2025_records:
            try:
                record_data = self._fetch_record_metadata(record_id)
                if record_data and self._is_valid_report(record_data):
                    fallback_reports.append(record_data)
                self.rate_limiter.wait()
            except Exception as e:
                logger.warning(f"Failed to fetch record {record_id}: {e}")
                continue
        
        return fallback_reports
    
    def _search_undl_reports(self) -> List[Dict[str, Any]]:
        """Search UNDL for recent reports using their search API."""
        logger.info("Searching UNDL for August 2025 reports...")
        
        try:
            # Search for August 2025 reports  
            search_url = "https://digitallibrary.un.org/search"
            params = {
                'c': 'Reports',
                'cc': 'Reports', 
                'ln': 'en',
                'p': '2025-08',  # August 2025
                'f': 'datecreated',
                'rg': '20',  # Get 20 results
                'of': 'xm'   # XML format for parsing
            }
            
            self.rate_limiter.wait()
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response to extract record IDs
            record_ids = self._parse_search_xml(response.text)
            logger.info(f"Found {len(record_ids)} record IDs from search")
            
            # Fetch metadata for each record
            reports = []
            for record_id in record_ids[:10]:  # Limit to first 10
                try:
                    record_data = self._fetch_record_metadata(record_id)
                    if record_data and self._is_valid_report(record_data):
                        reports.append(record_data)
                    self.rate_limiter.wait()
                except Exception as e:
                    logger.warning(f"Failed to fetch record {record_id}: {e}")
                    continue
                    
            return reports
            
        except Exception as e:
            logger.error(f"UNDL API search failed: {e}")
            return []
    
    def _parse_search_xml(self, xml_text: str) -> List[int]:
        """Parse UNDL search XML response to extract record IDs."""
        record_ids = []
        try:
            # Parse XML and extract record IDs from controlfield 001
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            
            # Find all record elements
            for record in root.findall('.//{http://www.loc.gov/MARC21/slim}record'):
                controlfield = record.find('.//{http://www.loc.gov/MARC21/slim}controlfield[@tag="001"]')
                if controlfield is not None and controlfield.text:
                    try:
                        record_id = int(controlfield.text)
                        record_ids.append(record_id)
                    except ValueError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to parse search XML: {e}")
            
        return record_ids
    
    def _filter_reports_by_criteria(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter reports by date, language, and target bodies."""
        filtered_reports = []
        target_bodies = set(self.config.get('target_bodies', []))
        
        for report in reports:
            try:
                # Parse report date
                report_date = datetime.strptime(report['date'], '%Y-%m-%d')
                
                # Check if within date window (focus on 2025 reports)
                if report_date.year >= 2025:
                    # Check if from target body (if specified)
                    if not target_bodies or report['organ'] in target_bodies:
                        # Check language
                        if report['language'] in self.config.get('languages', ['en']):
                            filtered_reports.append(report)
            except Exception as e:
                logger.warning(f"Failed to filter report {report.get('symbol', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Found {len(filtered_reports)} reports matching criteria")
        return filtered_reports
    
    def expand_with_recent_searches(self) -> List[Dict[str, Any]]:
        """Expand seed data by searching for recent reports patterns."""
        logger.info("Expanding with pattern-based recent report discovery...")
        
        expanded_reports = []
        
        # Use UNDL Record API to check recent record IDs
        recent_record_ids = self._get_recent_record_ids()
        
        for record_id in recent_record_ids:
            try:
                record_data = self._fetch_record_metadata(record_id)
                if record_data and self._is_valid_report(record_data):
                    expanded_reports.append(record_data)
                    
                # Respect rate limits
                self.rate_limiter.wait()
                
            except Exception as e:
                logger.warning(f"Failed to fetch record {record_id}: {e}")
                continue
                
        logger.info(f"Found {len(expanded_reports)} additional reports via API discovery")
        return expanded_reports
    
    def _get_recent_record_ids(self) -> List[int]:
        """Get list of recent record IDs to check."""
        # Sample of recent record IDs near known records
        # This is a heuristic approach - in production, would use sitemap or search API
        base_ids = [4060789, 4025952, 4021147, 4007894, 4025121]  # Known record IDs
        
        candidate_ids = []
        for base_id in base_ids:
            # Check records around known IDs (smaller range for efficiency)
            for offset in range(-10, 11, 5):  # Reduced from ±50 to ±10
                candidate_id = base_id + offset
                if candidate_id > 0:
                    candidate_ids.append(candidate_id)
                    
        # Deduplicate and sort, limit to first 20 candidates
        return sorted(list(set(candidate_ids)))[:20]
    
    def _fetch_record_metadata(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a specific record ID using UNDL Record API."""
        try:
            # Use JSON API to get comprehensive metadata (no field restrictions)
            url = f"https://digitallibrary.un.org/record/{record_id}?of=recjson"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 404:
                return None  # Record doesn't exist
            elif response.status_code != 200:
                logger.warning(f"API returned {response.status_code} for record {record_id}")
                return None
                
            data = response.json()
            
            # Handle array response format
            if isinstance(data, list) and data:
                data = data[0]  # Take first item from array
            elif not isinstance(data, dict):
                logger.warning(f"Unexpected data format for record {record_id}")
                return None
            
            # Extract relevant information using improved extractors
            title = self._extract_title(data)
            if not title:
                logger.debug(f"No title found for record {record_id}, skipping")
                return None  # Skip records without titles
            
            record_info = {
                'title': title,
                'symbol': self._extract_symbol(data),
                'date': self._extract_date(data),
                'organ': self._extract_organ(data),
                'language': self._extract_language(data),
                'record_url': f'https://digitallibrary.un.org/record/{record_id}',
                'file_urls': self._extract_file_urls(record_id, data)
            }
            
            logger.debug(f"Successfully extracted metadata for record {record_id}: {record_info['title'][:50]}...")
            return record_info
            
        except Exception as e:
            logger.warning(f"Failed to fetch record {record_id}: {e}")
            return None
    
    def _extract_title(self, data: Dict) -> str:
        """Extract document title from record data."""
        # Try title field first (standard)
        if 'title' in data:
            if isinstance(data['title'], dict) and 'title' in data['title']:
                return data['title']['title']
            elif isinstance(data['title'], str):
                return data['title']
        
        # Try abstract summary as fallback
        if 'abstract' in data and isinstance(data['abstract'], dict):
            if 'summary' in data['abstract']:
                summary = data['abstract']['summary']
                # Use summary as title if it's not too long
                if len(summary) <= 200:
                    return summary
                else:
                    # Use first sentence of summary
                    sentences = summary.split('. ')
                    if sentences:
                        return sentences[0] + '.'
        
        # Try corporate name as last resort
        if 'corporate_name' in data and data['corporate_name']:
            corp_names = data['corporate_name']
            if isinstance(corp_names, list) and corp_names:
                corp_name = corp_names[0]
                if isinstance(corp_name, dict) and 'name' in corp_name:
                    return f"Document from {corp_name['name']}"
        
        return ''
    
    def _extract_language(self, data: Dict) -> str:
        """Extract language from record data."""
        if 'language' in data:
            if isinstance(data['language'], list) and data['language']:
                return data['language'][0]
            elif isinstance(data['language'], str):
                return data['language']
        return 'en'  # Default to English
    
    def _extract_symbol(self, data: Dict) -> str:
        """Extract UN document symbol from record data."""
        
        # First, check files for symbol patterns in filenames
        if 'files' in data and data['files']:
            files_list = data['files']
            if isinstance(files_list, list):
                for file_entry in files_list:
                    if isinstance(file_entry, dict):
                        filename = file_entry.get('full_name', '') or file_entry.get('name', '')
                        if filename:
                            # Parse filename like "A_ES-10_1043--S_2025_529-EN.pdf"
                            # Extract symbols by replacing underscores with slashes
                            base_name = filename.replace('-EN.pdf', '').replace('-ES.pdf', '').replace('-FR.pdf', '').replace('-AR.pdf', '').replace('-RU.pdf', '').replace('-ZH.pdf', '')
                            if '--' in base_name:  # Multiple symbols
                                symbols = base_name.split('--')
                                clean_symbols = []
                                for symbol in symbols:
                                    clean_symbol = symbol.replace('_', '/')
                                    if re.match(r'[A-Z]/.*', clean_symbol):
                                        clean_symbols.append(clean_symbol)
                                if clean_symbols:
                                    return ', '.join(clean_symbols)  # Return all symbols
                            else:
                                # Single symbol
                                clean_symbol = base_name.replace('_', '/')
                                if re.match(r'[A-Z]/.*', clean_symbol):
                                    return clean_symbol
        
        # Check report_number field (standard approach)
        if 'report_number' in data:
            report_nums = data['report_number']
            if isinstance(report_nums, list):
                for num in report_nums:
                    if isinstance(num, str) and re.match(r'[A-Z]/.*', num):
                        return num
            elif isinstance(report_nums, str) and re.match(r'[A-Z]/.*', report_nums):
                return report_nums
        
        # Check series field
        if 'series' in data:
            series = data['series']
            if isinstance(series, list):
                for s in series:
                    if isinstance(s, str) and re.match(r'[A-Z]/\d+', s):
                        return s
            elif isinstance(series, str) and re.match(r'[A-Z]/\d+', series):
                return series
        
        # Check title for embedded symbol
        title = self._extract_title(data)
        if title:
            # Look for pattern like "A/80/1" in title
            symbol_match = re.search(r'\b([A-Z]+/[A-Z]*\d+[/\w]*)\b', title)
            if symbol_match:
                return symbol_match.group(1)
        
        return f'UNDoc-{data.get("recid", "Unknown")}'  # Fallback using record ID
    
    def _extract_date(self, data: Dict) -> str:
        """Extract publication date from record data."""
        # Try imprint date first (most reliable for UNDL)
        if 'imprint' in data and isinstance(data['imprint'], dict):
            if 'date' in data['imprint']:
                imprint_date = data['imprint']['date']
                if isinstance(imprint_date, str):
                    try:
                        # Handle formats like "22 Aug. 2025"
                        if re.search(r'\d{1,2}\s+\w{3}\.\s+\d{4}', imprint_date):
                            # Parse "22 Aug. 2025" format
                            import datetime
                            date_obj = datetime.datetime.strptime(imprint_date.replace('.', ''), '%d %b %Y')
                            return date_obj.strftime('%Y-%m-%d')
                        # Handle YYYY-MM-DD format
                        elif re.match(r'\d{4}-\d{2}-\d{2}', imprint_date):
                            return imprint_date
                    except:
                        pass
        
        # Try prepublication date
        if 'prepublication' in data and isinstance(data['prepublication'], dict):
            if 'place' in data['prepublication']:  # Actually contains date in UNDL
                prep_date = data['prepublication']['place']
                if isinstance(prep_date, str) and re.match(r'\d{4}-\d{2}-\d{2}', prep_date):
                    return prep_date
        
        # Try standard date fields
        date_sources = ['date', 'publication_date', 'created']
        for date_field in date_sources:
            if date_field in data:
                date_value = data[date_field]
                if isinstance(date_value, list) and date_value:
                    date_value = date_value[0]
                    
                if isinstance(date_value, str) and date_value:
                    try:
                        if re.match(r'\d{4}-\d{2}-\d{2}', date_value):
                            return date_value
                        elif re.match(r'^\d{4}$', date_value):
                            return f"{date_value}-08-01"
                    except:
                        continue
        
        # Fallback to August 2025 for current discovery
        return '2025-08-01'
    
    def _extract_organ(self, data: Dict) -> str:
        """Extract UN organ/body from record data."""
        # Check imprint publisher name first
        if 'imprint' in data and isinstance(data['imprint'], dict):
            if 'publisher_name' in data['imprint']:
                pub_name = data['imprint']['publisher_name']
                if isinstance(pub_name, str):
                    if 'General Assembly' in pub_name:
                        return 'General Assembly'
                    elif 'Security Council' in pub_name:
                        return 'Security Council'
                    elif 'Economic and Social Council' in pub_name:
                        return 'Economic and Social Council'
                    elif 'UN' in pub_name:
                        # Generic UN publisher, determine by symbol
                        pass
        
        # Check corporate name for issuing body
        if 'corporate_name' in data and data['corporate_name']:
            corp_names = data['corporate_name']
            if isinstance(corp_names, list):
                for corp in corp_names:
                    if isinstance(corp, dict) and 'name' in corp:
                        name = corp['name']
                        if 'State of Palestine' in name:
                            return 'General Assembly'  # Palestine documents go to GA
                        elif 'Secretary-General' in name:
                            return 'Secretary-General'
                        elif 'Security Council' in name:
                            return 'Security Council'
        
        # Default based on symbol pattern
        symbol = self._extract_symbol(data)
        if symbol.startswith('A/'):
            return 'General Assembly'
        elif symbol.startswith('S/'):
            return 'Security Council'
        elif symbol.startswith('E/'):
            return 'Economic and Social Council'
        elif symbol.startswith('JIU/'):
            return 'Joint Inspection Unit'
        else:
            return 'UN System'
    
    def _extract_file_urls(self, record_id: int, data: Dict) -> List[str]:
        """Extract file download URLs for the record."""
        file_urls = []
        
        # First, check if 'files' field has actual filenames with URLs
        if 'files' in data and data['files']:
            files_list = data['files']
            if isinstance(files_list, list):
                for file_entry in files_list:
                    if isinstance(file_entry, dict):
                        # Check for direct URL in file entry
                        direct_url = file_entry.get('url')
                        if direct_url:
                            # Prefer English files
                            if '-EN.pdf' in direct_url:
                                file_urls.insert(0, direct_url)  # Put English first
                            elif '.pdf' in direct_url:
                                file_urls.append(direct_url)
                        else:
                            # Fallback: construct URL from filename
                            full_name = file_entry.get('full_name', '')
                            if full_name and '-EN.pdf' in full_name:
                                file_urls.insert(0, f"https://digitallibrary.un.org/record/{record_id}/files/{full_name}")
                            elif full_name and '.pdf' in full_name:
                                file_urls.append(f"https://digitallibrary.un.org/record/{record_id}/files/{full_name}")
        
        # If no files found, try symbol-based construction
        if not file_urls:
            symbol = self._extract_symbol(data)
            
            if symbol and not symbol.startswith('UNDoc-'):
                # Primary URL: Digital Library direct file link
                filename = symbol.replace('/', '_') + '-EN.pdf'
                file_urls.append(f"https://digitallibrary.un.org/record/{record_id}/files/{filename}")
                
                # Secondary URL: UNDOCS system
                file_urls.append(f"https://undocs.org/{symbol}")
            else:
                # Fallback URLs for records without clear symbols
                file_urls.append(f"https://digitallibrary.un.org/record/{record_id}/files/doc-EN.pdf")
                file_urls.append(f"https://digitallibrary.un.org/record/{record_id}")
        
        # Always add record page as final fallback
        if f"https://digitallibrary.un.org/record/{record_id}" not in file_urls:
            file_urls.append(f"https://digitallibrary.un.org/record/{record_id}")
        
        return file_urls
    
    def _is_valid_report(self, record_data: Dict) -> bool:
        """Check if record appears to be a valid UN report."""
        title = record_data.get('title', '').lower()
        symbol = record_data.get('symbol', '')
        date_str = record_data.get('date', '')
        
        # Must have title and symbol
        if not title or not symbol:
            return False
            
        # Must be recent (within configured window)
        try:
            record_date = datetime.strptime(date_str, '%Y-%m-%d')
            if record_date < self.start_date:
                return False
        except:
            # If date parsing fails, assume it's recent enough
            pass
            
        # Must be in target language
        language = record_data.get('language', 'en')
        if language not in self.config.get('languages', ['en']):
            return False
            
        # Title should suggest it's a report
        report_keywords = ['report', 'annual', 'progress', 'implementation', 'review', 'situation', 'analysis']
        if any(keyword in title for keyword in report_keywords):
            return True
            
        # Symbol patterns that typically indicate reports
        if re.match(r'[A-Z]/\d+/\d+', symbol) or re.match(r'[A-Z]/\d+', symbol):
            return True
            
        return False
    
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
        
    def discover_all(self) -> List[Dict[str, Any]]:
        """Main discovery method combining all sources."""
        logger.info("Starting UN reports discovery...")
        
        all_reports = []
        
        # Try sitemap first, fallback to seed data
        reports = self.discover_from_sitemap()
        all_reports.extend(reports)
        
        # Skip expansion for faster testing with verified records
        # if len(all_reports) < 10:  # If we have fewer than 10 reports, try to expand
        #     logger.info("Expanding corpus with API-based discovery...")
        #     expanded = self.expand_with_recent_searches()
        #     # Deduplicate by symbol
        #     existing_symbols = {r.get('symbol') for r in all_reports}
        #     for report in expanded:
        #         if report.get('symbol') not in existing_symbols:
        #             all_reports.append(report)
        
        logger.info(f"Discovery complete. Found {len(all_reports)} total reports.")
        return all_reports

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
    
    print(f"Discovery complete! Found {len(reports)} reports.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()