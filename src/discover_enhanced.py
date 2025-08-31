"""Enhanced Discovery module targeting high-impact 2025 UN reports."""

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

class EnhancedUNDLClient:
    """Enhanced client for strategic UN report discovery."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limiter = RateLimiter(config.get('delay_seconds', 5))
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UN-RAG-Research-Bot/1.0 (Educational Purpose; Respectful Crawling)'
        })
        self.api_key = os.getenv("UN_API")
        
        # Strategic targeting
        self.target_docs = config.get('corpus', {}).get('target_documents', 150)
        self.priority_keywords = config.get('priority_keywords', [])
        
    def search_strategic_reports(self) -> List[Dict[str, Any]]:
        """Search for high-impact UN reports strategically."""
        logger.info(f"Searching for {self.target_docs} high-impact 2025 reports...")
        
        all_reports = []
        
        # Strategic search queries for maximum impact
        search_strategies = [
            # High-priority themes
            ("climate change report 2025", 20),
            ("sustainable development goals 2025", 20),
            ("Secretary-General annual report 2025", 15),
            ("Security Council annual report 2025", 15),
            ("General Assembly report 2025", 20),
            ("peacekeeping operation 2025", 15),
            ("human rights report 2025", 15),
            ("humanitarian situation 2025", 15),
            
            # Specific high-impact documents
            ("World Economic Situation Prospects 2025", 10),
            ("Financing for Development 2025", 10),
            ("World Social Report 2025", 5),
            ("Global Compact 2025", 5),
        ]
        
        for query, max_results in search_strategies:
            try:
                reports = self._search_with_filters(query, max_results)
                logger.info(f"Query '{query}' found {len(reports)} reports")
                
                # Deduplicate by symbol
                existing_symbols = {r.get('symbol') for r in all_reports}
                new_reports = [r for r in reports if r.get('symbol') not in existing_symbols]
                all_reports.extend(new_reports)
                
                # Stop if we have enough high-quality reports
                if len(all_reports) >= self.target_docs:
                    logger.info(f"Reached target of {self.target_docs} documents")
                    break
                
                # Rate limiting between queries
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                continue
        
        # Sort by priority and take top N
        prioritized_reports = self._prioritize_reports(all_reports)
        final_reports = prioritized_reports[:self.target_docs]
        
        logger.info(f"Strategic discovery complete: {len(final_reports)} high-impact reports selected")
        return final_reports
    
    def _search_with_filters(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search UNDL with enhanced filtering."""
        try:
            # Enhanced search parameters
            params = {
                "p": f"{query} AND date:2025",
                "format": "xml",
                "ln": "en",
                "rg": min(max_results, 50)
            }
            
            self.rate_limiter.wait()
            
            if self.api_key:
                url = "https://digitallibrary.un.org/api/v1/search"
                headers = {
                    "content-type": "application/xml",
                    "Authorization": f"Token {self.api_key}"
                }
                response = self.session.get(url, params=params, headers=headers, timeout=30)
            else:
                url = "https://digitallibrary.un.org/search"
                params["of"] = "xm"
                response = self.session.get(url, params=params, timeout=30)
            
            response.raise_for_status()
            return self._parse_marcxml_response(response.text)
            
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []
    
    def _parse_marcxml_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse MARCXML response to extract record data."""
        try:
            clean_xml = xml_text.replace('xmlns="http://www.loc.gov/MARC21/slim"', '')
            root = ET.fromstring(clean_xml)
            
            collection_elem = root.find('collection')
            if collection_elem is None:
                collection_elem = root
            
            # Write temporary XML file for pymarc parsing
            temp_path = Path.home() / ".undl"
            temp_path.mkdir(exist_ok=True)
            temp_file = temp_path / "temp_enhanced.xml"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(ET.tostring(collection_elem, encoding='unicode'))
            
            # Parse using pymarc
            records = pymarc.parse_xml_to_array(str(temp_file))
            logger.debug(f"Parsed {len(records)} MARC records")
            
            # Convert to our format
            result = []
            for record in records:
                record_data = self._extract_record_data(record)
                if record_data and self._is_high_impact_report(record_data):
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
            record_id = self._extract_field(record, "001")
            if not record_id:
                return None
            
            title = self._extract_field(record, "245")
            if not title:
                title = self._extract_field(record, "239")
            
            if not title:
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
                'authors': self._extract_field(record, "710", "a", collection=True),
                'priority_score': self._calculate_priority_score(title, symbol)
            }
            
            return record_data
            
        except Exception as e:
            logger.warning(f"Failed to extract record data: {e}")
            return None
    
    def _calculate_priority_score(self, title: str, symbol: str) -> float:
        """Calculate priority score for strategic selection."""
        score = 0.0
        title_lower = title.lower()
        
        # High priority keywords
        priority_weights = {
            'annual report': 10.0,
            'secretary-general': 9.0,
            'climate': 8.0,
            'sustainable development': 8.0,
            'security council': 9.0,
            'general assembly': 8.0,
            'peacekeeping': 7.0,
            'human rights': 7.0,
            'humanitarian': 6.0,
            'progress report': 6.0,
            'situation': 5.0
        }
        
        for keyword, weight in priority_weights.items():
            if keyword in title_lower:
                score += weight
        
        # Symbol-based priorities
        if symbol.startswith('A/'):  # General Assembly
            score += 5.0
        elif symbol.startswith('S/'):  # Security Council
            score += 7.0
        elif symbol.startswith('E/'):  # Economic and Social Council
            score += 4.0
        
        # Recency bonus (2025 documents)
        if '2025' in title_lower:
            score += 3.0
        
        return score
    
    def _prioritize_reports(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort reports by strategic priority."""
        return sorted(reports, key=lambda r: r.get('priority_score', 0), reverse=True)
    
    def _is_high_impact_report(self, record_data: Dict[str, Any]) -> bool:
        """Check if record is a high-impact report."""
        title = record_data.get('title', '').lower()
        symbol = record_data.get('symbol', '')
        
        # Must have title and be English
        if not title or record_data.get('language') != 'en':
            return False
        
        # Must have downloadable files
        file_urls = record_data.get('file_urls', [])
        if not any('.pdf' in url.lower() for url in file_urls):
            return False
        
        # Priority score threshold
        priority_score = record_data.get('priority_score', 0)
        if priority_score >= 3.0:  # High-impact threshold
            return True
        
        # Fallback: standard report patterns
        report_patterns = [
            'annual report', 'progress report', 'situation report',
            'secretary-general', 'security council', 'general assembly'
        ]
        
        return any(pattern in title for pattern in report_patterns)
    
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
                if collection:
                    return [f.format_field() for f in fields]
                else:
                    return fields[0].format_field()
                    
        except Exception:
            return None
    
    def _get_symbol(self, record: Record) -> str:
        """Extract UN document symbol from MARC record."""
        symbol = self._extract_field(record, "191", "a")
        if symbol:
            return symbol
        
        symbol = self._extract_field(record, "791", "a")
        if symbol:
            return symbol
        
        record_id = self._extract_field(record, "001")
        return f"UNDoc-{record_id}" if record_id else "Unknown"
    
    def _get_downloads(self, record: Record) -> Dict[str, str]:
        """Extract download URLs from MARC record."""
        downloads = {}
        
        url_fields = record.get_fields("856")
        for field in url_fields:
            url = field.get_subfields("u")
            desc = field.get_subfields("y")
            
            if url and desc:
                downloads[desc[0]] = url[0]
            elif url:
                url_str = url[0]
                if "-EN." in url_str.upper():
                    downloads["English"] = url_str
                else:
                    downloads["Document"] = url_str
        
        return downloads
    
    def _extract_date(self, record: Record) -> str:
        """Extract publication date from MARC record."""
        date_str = self._extract_field(record, "269")
        if date_str and re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return date_str
        
        pub_info = self._extract_field(record, "260", "c")
        if pub_info:
            year_match = re.search(r'(\d{4})', pub_info)
            if year_match and year_match.group(1) == "2025":
                return f"2025-01-01"
        
        return "2025-01-01"
    
    def _extract_language(self, record: Record) -> str:
        """Extract language from MARC record."""
        control_008 = self._extract_field(record, "008")
        if control_008 and len(control_008) > 37:
            lang_code = control_008[35:38].strip()
            if lang_code == "eng":
                return "en"
        return "en"
    
    def _extract_organ(self, record: Record, symbol: str) -> str:
        """Extract UN organ from MARC record and symbol."""
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
                urls.insert(0, url)
            elif ".pdf" in url.lower():
                urls.append(url)
        
        if not urls:
            urls.append(f"https://digitallibrary.un.org/record/{record_id}")
        
        return urls


class StrategicUNReportDiscoverer:
    """Strategic discoverer targeting high-impact reports."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = EnhancedUNDLClient(config)
        
    def discover_all(self) -> List[Dict[str, Any]]:
        """Main discovery method for strategic expansion."""
        logger.info("Starting strategic UN reports discovery for optimal demo corpus...")
        
        # Get strategic reports
        reports = self.client.search_strategic_reports()
        
        logger.info(f"Strategic discovery complete: {len(reports)} high-impact reports")
        return reports
    
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
        logger.info(f"Saved {len(reports)} strategic reports to {output_file}")


def main():
    """Main function for strategic discovery."""
    from utils import setup_logging
    
    setup_logging()
    config = load_config()
    
    discoverer = StrategicUNReportDiscoverer(config)
    reports = discoverer.discover_all()
    
    # Save results
    output_file = config['paths']['records_file']
    discoverer.save_records(reports, output_file)
    
    print(f"Strategic discovery complete! Found {len(reports)} high-impact reports.")
    print(f"Results saved to {output_file}")
    
    # Show top strategic reports
    if reports:
        print("\n🎯 Top Strategic Reports:")
        sorted_reports = sorted(reports, key=lambda r: r.get('priority_score', 0), reverse=True)
        for i, report in enumerate(sorted_reports[:5]):
            print(f"{i+1:2d}. {report['title'][:70]}...")
            print(f"    Symbol: {report['symbol']}, Score: {report.get('priority_score', 0):.1f}")
            print()

if __name__ == "__main__":
    main()