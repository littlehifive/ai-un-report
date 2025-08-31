"""Hybrid Discovery combining original working approach with strategic targeting."""

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

class HybridUNDLClient:
    """Hybrid client combining working discovery with strategic targeting."""
    
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
        
    def discover_comprehensive_reports(self) -> List[Dict[str, Any]]:
        """Comprehensive discovery combining multiple approaches."""
        logger.info(f"Starting comprehensive discovery targeting {self.target_docs} reports...")
        
        all_reports = []
        
        # Phase 1: Use the working approach with broader queries
        working_reports = self._working_discovery_approach()
        all_reports.extend(working_reports)
        logger.info(f"Working approach found {len(working_reports)} reports")
        
        # Phase 2: Strategic high-impact searches (if we need more)
        if len(all_reports) < self.target_docs:
            strategic_reports = self._strategic_discovery_approach(all_reports)
            all_reports.extend(strategic_reports)
            logger.info(f"Strategic approach added {len(strategic_reports)} more reports")
        
        # Phase 3: Fill gaps with recent record scanning (if still need more)
        if len(all_reports) < self.target_docs:
            gap_reports = self._gap_filling_discovery(all_reports)
            all_reports.extend(gap_reports)
            logger.info(f"Gap filling added {len(gap_reports)} more reports")
        
        # Deduplicate and prioritize
        deduplicated = self._deduplicate_reports(all_reports)
        prioritized = self._prioritize_reports(deduplicated)
        
        # Take top N reports
        final_reports = prioritized[:self.target_docs]
        
        logger.info(f"Comprehensive discovery complete: {len(final_reports)} reports selected")
        return final_reports
    
    def _working_discovery_approach(self) -> List[Dict[str, Any]]:
        """Use the approach that was working well."""
        reports = []
        
        # Proven working queries with broader scope
        working_queries = [
            "reports 2025 english",
            "General Assembly 2025",
            "Secretary-General report 2025", 
            "Economic Social Council 2025",
            "Security Council 2025",
            "sustainable development 2025",
            "climate change 2025",
            "human rights 2025",
            "peacekeeping 2025",
            "humanitarian 2025"
        ]
        
        for query in working_queries:
            try:
                query_reports = self._search_with_standard_params(query, max_results=25)
                logger.info(f"Query '{query}' found {len(query_reports)} reports")
                
                # Deduplicate by symbol
                existing_symbols = {r.get('symbol') for r in reports}
                new_reports = [r for r in query_reports if r.get('symbol') not in existing_symbols]
                reports.extend(new_reports)
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                continue
        
        return reports
    
    def _strategic_discovery_approach(self, existing_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strategic searches for high-impact documents."""
        reports = []
        existing_symbols = {r.get('symbol') for r in existing_reports}
        
        # High-impact specific searches
        strategic_queries = [
            ("World Economic Situation Prospects", 5),
            ("Financing for Development", 5), 
            ("annual report Secretary-General", 10),
            ("situation report", 15),
            ("progress report", 15),
            ("implementation report", 10)
        ]
        
        for query, max_results in strategic_queries:
            try:
                query_reports = self._search_with_standard_params(query, max_results=max_results)
                
                # Filter out existing
                new_reports = [r for r in query_reports if r.get('symbol') not in existing_symbols]
                reports.extend(new_reports)
                
                # Update existing symbols
                existing_symbols.update(r.get('symbol') for r in new_reports)
                
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Strategic query '{query}' failed: {e}")
                continue
        
        return reports
    
    def _gap_filling_discovery(self, existing_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill remaining gaps with recent record scanning."""
        reports = []
        existing_symbols = {r.get('symbol') for r in existing_reports}
        
        # Known good recent record ID ranges (expand the working ranges)
        record_ranges = [
            (4070000, 4090000, 10),  # Recent 2025 records, sample 10
            (4050000, 4070000, 8),   # Mid-2025 records, sample 8
            (4030000, 4050000, 5),   # Earlier records, sample 5
        ]
        
        for start_id, end_id, sample_count in record_ranges:
            try:
                # Sample record IDs from the range
                import random
                sample_ids = random.sample(range(start_id, end_id), min(sample_count * 3, end_id - start_id))
                
                for record_id in sample_ids[:sample_count]:
                    try:
                        record_data = self._fetch_record_metadata(record_id)
                        if (record_data and 
                            self._is_valid_report(record_data) and 
                            record_data.get('symbol') not in existing_symbols):
                            
                            reports.append(record_data)
                            existing_symbols.add(record_data.get('symbol'))
                        
                        self.rate_limiter.wait()
                        
                    except Exception as e:
                        logger.debug(f"Failed to fetch record {record_id}: {e}")
                        continue
                
            except Exception as e:
                logger.warning(f"Gap filling for range {start_id}-{end_id} failed: {e}")
                continue
        
        return reports
    
    def _search_with_standard_params(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Standard search using the working parameters."""
        try:
            params = {
                "p": query,
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
            logger.error(f"Standard search failed for '{query}': {e}")
            return []
    
    def _fetch_record_metadata(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a specific record ID."""
        try:
            url = f"https://digitallibrary.un.org/record/{record_id}?of=recjson"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 404:
                return None
            elif response.status_code != 200:
                return None
                
            data = response.json()
            
            if isinstance(data, list) and data:
                data = data[0]
            elif not isinstance(data, dict):
                return None
            
            # Extract using the working extractors
            title = self._extract_title(data)
            if not title:
                return None
            
            record_info = {
                'id': str(record_id),
                'title': title,
                'symbol': self._extract_symbol(data),
                'date': self._extract_date(data),
                'organ': self._extract_organ(data),
                'language': self._extract_language(data),
                'record_url': f'https://digitallibrary.un.org/record/{record_id}',
                'file_urls': self._extract_file_urls(record_id, data),
                'priority_score': self._calculate_priority_score(title, self._extract_symbol(data))
            }
            
            return record_info
            
        except Exception as e:
            logger.debug(f"Failed to fetch record {record_id}: {e}")
            return None
    
    def _parse_marcxml_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse MARCXML response using working approach."""
        try:
            clean_xml = xml_text.replace('xmlns="http://www.loc.gov/MARC21/slim"', '')
            root = ET.fromstring(clean_xml)
            
            collection_elem = root.find('collection')
            if collection_elem is None:
                collection_elem = root
            
            # Write temporary XML file for pymarc parsing
            temp_path = Path.home() / ".undl"
            temp_path.mkdir(exist_ok=True)
            temp_file = temp_path / "temp_hybrid.xml"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(ET.tostring(collection_elem, encoding='unicode'))
            
            # Parse using pymarc
            records = pymarc.parse_xml_to_array(str(temp_file))
            logger.debug(f"Parsed {len(records)} MARC records")
            
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
        """Extract structured data from a MARC record using working approach."""
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
                'date': self._extract_date_from_marc(record),
                'organ': self._extract_organ_from_marc(record, symbol),
                'language': self._extract_language_from_marc(record),
                'record_url': f'https://digitallibrary.un.org/record/{record_id}',
                'file_urls': self._extract_file_urls_from_marc(downloads, record_id),
                'summary': self._extract_field(record, "520"),
                'authors': self._extract_field(record, "710", "a", collection=True),
                'priority_score': self._calculate_priority_score(title.strip(), symbol)
            }
            
            return record_data
            
        except Exception as e:
            logger.warning(f"Failed to extract record data: {e}")
            return None
    
    # Working extraction methods from the original discovery
    def _extract_title(self, data: Dict) -> str:
        """Extract document title from record data."""
        if 'title' in data:
            if isinstance(data['title'], dict) and 'title' in data['title']:
                return data['title']['title']
            elif isinstance(data['title'], str):
                return data['title']
        
        if 'abstract' in data and isinstance(data['abstract'], dict):
            if 'summary' in data['abstract']:
                summary = data['abstract']['summary']
                if len(summary) <= 200:
                    return summary
                else:
                    sentences = summary.split('. ')
                    if sentences:
                        return sentences[0] + '.'
        
        if 'corporate_name' in data and data['corporate_name']:
            corp_names = data['corporate_name']
            if isinstance(corp_names, list) and corp_names:
                corp_name = corp_names[0]
                if isinstance(corp_name, dict) and 'name' in corp_name:
                    return f"Document from {corp_name['name']}"
        
        return ''
    
    def _extract_symbol(self, data: Dict) -> str:
        """Extract UN document symbol from record data."""
        # Check files for symbol patterns in filenames
        if 'files' in data and data['files']:
            files_list = data['files']
            if isinstance(files_list, list):
                for file_entry in files_list:
                    if isinstance(file_entry, dict):
                        filename = file_entry.get('full_name', '') or file_entry.get('name', '')
                        if filename:
                            base_name = filename.replace('-EN.pdf', '').replace('-ES.pdf', '').replace('-FR.pdf', '').replace('-AR.pdf', '').replace('-RU.pdf', '').replace('-ZH.pdf', '')
                            if '--' in base_name:
                                symbols = base_name.split('--')
                                clean_symbols = []
                                for symbol in symbols:
                                    clean_symbol = symbol.replace('_', '/')
                                    if re.match(r'[A-Z]/.*', clean_symbol):
                                        clean_symbols.append(clean_symbol)
                                if clean_symbols:
                                    return ', '.join(clean_symbols)
                            else:
                                clean_symbol = base_name.replace('_', '/')
                                if re.match(r'[A-Z]/.*', clean_symbol):
                                    return clean_symbol
        
        # Standard fields
        if 'report_number' in data:
            report_nums = data['report_number']
            if isinstance(report_nums, list):
                for num in report_nums:
                    if isinstance(num, str) and re.match(r'[A-Z]/.*', num):
                        return num
            elif isinstance(report_nums, str) and re.match(r'[A-Z]/.*', report_nums):
                return report_nums
        
        return f'UNDoc-{data.get("recid", "Unknown")}'
    
    def _extract_date(self, data: Dict) -> str:
        """Extract publication date from record data."""
        # Try imprint date first
        if 'imprint' in data and isinstance(data['imprint'], dict):
            if 'date' in data['imprint']:
                imprint_date = data['imprint']['date']
                if isinstance(imprint_date, str):
                    try:
                        if re.search(r'\d{1,2}\s+\w{3}\.\s+\d{4}', imprint_date):
                            import datetime
                            date_obj = datetime.datetime.strptime(imprint_date.replace('.', ''), '%d %b %Y')
                            return date_obj.strftime('%Y-%m-%d')
                        elif re.match(r'\d{4}-\d{2}-\d{2}', imprint_date):
                            return imprint_date
                    except:
                        pass
        
        # Try other date fields
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
        
        return '2025-08-01'
    
    def _extract_organ(self, data: Dict) -> str:
        """Extract UN organ/body from record data."""
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
        
        # Check corporate name
        if 'corporate_name' in data and data['corporate_name']:
            corp_names = data['corporate_name']
            if isinstance(corp_names, list):
                for corp in corp_names:
                    if isinstance(corp, dict) and 'name' in corp:
                        name = corp['name']
                        if 'Secretary-General' in name:
                            return 'Secretary-General'
                        elif 'Security Council' in name:
                            return 'Security Council'
        
        # Default based on symbol
        symbol = self._extract_symbol(data)
        if symbol.startswith('A/'):
            return 'General Assembly'
        elif symbol.startswith('S/'):
            return 'Security Council'
        elif symbol.startswith('E/'):
            return 'Economic and Social Council'
        else:
            return 'UN System'
    
    def _extract_language(self, data: Dict) -> str:
        """Extract language from record data."""
        if 'language' in data:
            if isinstance(data['language'], list) and data['language']:
                return data['language'][0]
            elif isinstance(data['language'], str):
                return data['language']
        return 'en'
    
    def _extract_file_urls(self, record_id: int, data: Dict) -> List[str]:
        """Extract file download URLs for the record."""
        file_urls = []
        
        if 'files' in data and data['files']:
            files_list = data['files']
            if isinstance(files_list, list):
                for file_entry in files_list:
                    if isinstance(file_entry, dict):
                        direct_url = file_entry.get('url')
                        if direct_url:
                            if '-EN.pdf' in direct_url:
                                file_urls.insert(0, direct_url)
                            elif '.pdf' in direct_url:
                                file_urls.append(direct_url)
                        else:
                            full_name = file_entry.get('full_name', '')
                            if full_name and '-EN.pdf' in full_name:
                                file_urls.insert(0, f"https://digitallibrary.un.org/record/{record_id}/files/{full_name}")
                            elif full_name and '.pdf' in full_name:
                                file_urls.append(f"https://digitallibrary.un.org/record/{record_id}/files/{full_name}")
        
        # Fallback URLs
        if not file_urls:
            symbol = self._extract_symbol(data)
            if symbol and not symbol.startswith('UNDoc-'):
                filename = symbol.replace('/', '_') + '-EN.pdf'
                file_urls.append(f"https://digitallibrary.un.org/record/{record_id}/files/{filename}")
                file_urls.append(f"https://undocs.org/{symbol}")
            else:
                file_urls.append(f"https://digitallibrary.un.org/record/{record_id}/files/doc-EN.pdf")
        
        # Always add record page
        if f"https://digitallibrary.un.org/record/{record_id}" not in file_urls:
            file_urls.append(f"https://digitallibrary.un.org/record/{record_id}")
        
        return file_urls
    
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
            'situation': 5.0,
            'implementation': 4.0,
            'financing': 6.0,
            'development': 4.0
        }
        
        for keyword, weight in priority_weights.items():
            if keyword in title_lower:
                score += weight
        
        # Symbol-based priorities
        if symbol.startswith('A/'):
            score += 5.0
        elif symbol.startswith('S/'):
            score += 7.0
        elif symbol.startswith('E/'):
            score += 4.0
        
        # Recency bonus (2025 documents)
        if '2025' in title_lower:
            score += 3.0
        
        return score
    
    def _is_valid_report(self, record_data: Dict) -> bool:
        """Check if record is a valid UN report."""
        title = record_data.get('title', '').lower()
        symbol = record_data.get('symbol', '')
        
        # Must have title and symbol
        if not title or not symbol:
            return False
        
        # Must be English
        if record_data.get('language') != 'en':
            return False
        
        # Must have downloadable files
        file_urls = record_data.get('file_urls', [])
        if not any('.pdf' in url.lower() for url in file_urls):
            return False
        
        # Check for report-like content or high priority score
        priority_score = record_data.get('priority_score', 0)
        if priority_score >= 3.0:
            return True
        
        # Fallback: standard report patterns
        report_keywords = ['report', 'annual', 'progress', 'implementation', 'review', 'situation', 'analysis']
        return any(keyword in title for keyword in report_keywords)
    
    def _deduplicate_reports(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate reports by symbol."""
        seen_symbols = set()
        deduped = []
        
        for report in reports:
            symbol = report.get('symbol')
            if symbol and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                deduped.append(report)
        
        return deduped
    
    def _prioritize_reports(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort reports by strategic priority."""
        return sorted(reports, key=lambda r: r.get('priority_score', 0), reverse=True)
    
    # MARC extraction methods (for XML responses)
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
    
    def _extract_date_from_marc(self, record: Record) -> str:
        """Extract publication date from MARC record."""
        date_str = self._extract_field(record, "269")
        if date_str and re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return date_str
        
        pub_info = self._extract_field(record, "260", "c")
        if pub_info:
            year_match = re.search(r'(\d{4})', pub_info)
            if year_match and year_match.group(1) in ["2024", "2025"]:
                return f"{year_match.group(1)}-01-01"
        
        return "2025-01-01"
    
    def _extract_language_from_marc(self, record: Record) -> str:
        """Extract language from MARC record."""
        control_008 = self._extract_field(record, "008")
        if control_008 and len(control_008) > 37:
            lang_code = control_008[35:38].strip()
            if lang_code == "eng":
                return "en"
        return "en"
    
    def _extract_organ_from_marc(self, record: Record, symbol: str) -> str:
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
    
    def _extract_file_urls_from_marc(self, downloads: Dict[str, str], record_id: str) -> List[str]:
        """Extract and prioritize file URLs from MARC downloads."""
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


class HybridUNReportDiscoverer:
    """Hybrid discoverer combining all working approaches."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = HybridUNDLClient(config)
        
    def discover_all(self) -> List[Dict[str, Any]]:
        """Main discovery method using hybrid approach."""
        logger.info("Starting hybrid UN reports discovery...")
        
        # Use comprehensive discovery
        reports = self.client.discover_comprehensive_reports()
        
        logger.info(f"Hybrid discovery complete: {len(reports)} high-quality reports")
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
        logger.info(f"Saved {len(reports)} hybrid-discovered reports to {output_file}")


def main():
    """Main function for hybrid discovery."""
    from utils import setup_logging
    
    setup_logging()
    config = load_config()
    
    discoverer = HybridUNReportDiscoverer(config)
    reports = discoverer.discover_all()
    
    # Save results
    output_file = config['paths']['records_file']
    discoverer.save_records(reports, output_file)
    
    print(f"Hybrid discovery complete! Found {len(reports)} high-quality reports.")
    print(f"Results saved to {output_file}")
    
    # Show sample reports by priority
    if reports:
        print("\n🎯 Sample High-Priority Reports:")
        sorted_reports = sorted(reports, key=lambda r: r.get('priority_score', 0), reverse=True)
        for i, report in enumerate(sorted_reports[:8]):
            print(f"{i+1:2d}. {report['title'][:70]}...")
            print(f"    Symbol: {report['symbol']}, Score: {report.get('priority_score', 0):.1f}")
            print()

if __name__ == "__main__":
    main()