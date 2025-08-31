"""Improved Fetching module for downloading UN report files with proper URL validation."""

import logging
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
from urllib.parse import urlparse
import time
import re

from utils import load_config, RateLimiter, ensure_dir, safe_filename, get_file_hash

logger = logging.getLogger(__name__)

class ImprovedUNReportFetcher:
    """Improved downloader for UN report files with proper URL validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limiter = RateLimiter(config['throttle']['delay_seconds'])
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UN-RAG-Research-Bot/1.0 (Educational Purpose; Respectful Crawling)'
        })
        
        self.raw_data_path = Path(config['paths']['raw_data'])
        ensure_dir(self.raw_data_path)
        
        # Load existing fetch manifest if it exists
        self.manifest_file = self.raw_data_path / "files_manifest.parquet"
        self.manifest = self._load_manifest()
        
    def _load_manifest(self) -> pd.DataFrame:
        """Load existing fetch manifest or create empty one."""
        if self.manifest_file.exists():
            return pd.read_parquet(self.manifest_file)
        else:
            return pd.DataFrame(columns=[
                'symbol', 'url', 'filename', 'status', 'http_code', 
                'file_size', 'checksum', 'download_date', 'error_msg', 'validation_status'
            ])
    
    def _save_manifest(self) -> None:
        """Save fetch manifest to disk."""
        self.manifest.to_parquet(self.manifest_file, index=False)
        logger.info(f"Updated manifest with {len(self.manifest)} entries")
    
    def _is_already_downloaded(self, url: str, symbol: str) -> bool:
        """Check if file is already successfully downloaded."""
        if self.manifest.empty:
            return False
            
        existing = self.manifest[
            (self.manifest['url'] == url) & 
            (self.manifest['symbol'] == symbol) &
            (self.manifest['status'] == 'success')
        ]
        
        if not existing.empty:
            # Check if file still exists
            filename = existing.iloc[0]['filename']
            file_path = self.raw_data_path / filename
            if file_path.exists():
                return True
            else:
                # File was deleted, remove from manifest
                self.manifest = self.manifest[
                    ~((self.manifest['url'] == url) & (self.manifest['symbol'] == symbol))
                ]
        
        return False
    
    def _generate_filename(self, url: str, symbol: str, language: str = 'en') -> str:
        """Generate safe filename for downloaded file."""
        parsed_url = urlparse(url)
        original_filename = Path(parsed_url.path).name
        
        # If we have a symbol, use it as base
        if symbol:
            safe_symbol = safe_filename(symbol)
            if original_filename and '.' in original_filename:
                extension = Path(original_filename).suffix
                return f"{safe_symbol}.{language}{extension}"
            else:
                return f"{safe_symbol}.{language}.pdf"  # Default to PDF
        else:
            # Fallback to original filename or hash
            if original_filename:
                return safe_filename(original_filename)
            else:
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                return f"doc_{url_hash}.{language}.pdf"
    
    def _validate_url_before_download(self, url: str) -> Dict[str, Any]:
        """Validate URL before attempting download."""
        validation_result = {
            'is_valid': False,
            'error_msg': None,
            'content_type': None,
            'file_size': None,
            'redirects_to': None
        }
        
        try:
            # First, check if URL is accessible
            response = self.session.head(url, timeout=30, allow_redirects=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                validation_result['content_type'] = content_type
                
                # Check if it's actually a PDF
                if 'application/pdf' in content_type:
                    validation_result['is_valid'] = True
                    validation_result['file_size'] = response.headers.get('content-length')
                elif 'text/html' in content_type:
                    # Check if it's an error page
                    validation_result['error_msg'] = "URL returns HTML instead of PDF"
                else:
                    validation_result['error_msg'] = f"Unexpected content type: {content_type}"
                    
            elif response.status_code == 404:
                validation_result['error_msg'] = "File not found (HTTP 404)"
            else:
                validation_result['error_msg'] = f"HTTP {response.status_code}: {response.reason}"
                
            # Check for redirects
            if response.history:
                validation_result['redirects_to'] = response.url
                
        except requests.exceptions.Timeout:
            validation_result['error_msg'] = "Request timeout during validation"
        except requests.exceptions.RequestException as e:
            validation_result['error_msg'] = f"Request failed during validation: {e}"
        except Exception as e:
            validation_result['error_msg'] = f"Unexpected error during validation: {e}"
        
        return validation_result
    
    def download_file(self, url: str, symbol: str, language: str = 'en') -> Dict[str, Any]:
        """Download a single file with enhanced validation."""
        logger.info(f"Downloading {symbol}: {url}")
        
        # Check if already downloaded
        if self._is_already_downloaded(url, symbol):
            logger.info(f"File {symbol} already downloaded, skipping")
            existing_entry = self.manifest[
                (self.manifest['url'] == url) & (self.manifest['symbol'] == symbol)
            ].iloc[0]
            return existing_entry.to_dict()
        
        # Validate URL before download
        validation = self._validate_url_before_download(url)
        if not validation['is_valid']:
            logger.warning(f"URL validation failed for {symbol}: {validation['error_msg']}")
            result = {
                'symbol': symbol,
                'url': url,
                'filename': None,
                'status': 'error',
                'http_code': None,
                'file_size': None,
                'checksum': None,
                'download_date': pd.Timestamp.now(),
                'error_msg': f"URL validation failed: {validation['error_msg']}",
                'validation_status': 'failed'
            }
            return result
        
        # Generate filename
        filename = self._generate_filename(url, symbol, language)
        file_path = self.raw_data_path / filename
        
        # Rate limit
        self.rate_limiter.wait()
        
        result = {
            'symbol': symbol,
            'url': url,
            'filename': filename,
            'status': 'error',
            'http_code': None,
            'file_size': None,
            'checksum': None,
            'download_date': pd.Timestamp.now(),
            'error_msg': None,
            'validation_status': 'passed'
        }
        
        try:
            # Download file with proper headers
            headers = {
                'Accept': 'application/pdf,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://digitallibrary.un.org/'
            }
            
            response = self.session.get(url, timeout=60, stream=True, headers=headers)
            result['http_code'] = response.status_code
            
            if response.status_code == 200:
                # Download file in chunks
                total_size = 0
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                
                # Validate download
                if total_size < 1024:  # Less than 1KB is suspicious
                    result['error_msg'] = f"Downloaded file too small ({total_size} bytes), likely an error page"
                    logger.warning(f"Downloaded file {filename} is suspiciously small: {total_size} bytes")
                    file_path.unlink(missing_ok=True)  # Remove invalid file
                    return result
                
                # Verify it's actually a PDF by checking file header
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(4)
                        if header != b'%PDF':
                            result['error_msg'] = "Downloaded file is not a valid PDF"
                            logger.warning(f"Downloaded file {filename} is not a PDF")
                            file_path.unlink(missing_ok=True)
                            return result
                except Exception as e:
                    result['error_msg'] = f"Failed to verify PDF header: {e}"
                    file_path.unlink(missing_ok=True)
                    return result
                
                # Calculate file stats
                result['file_size'] = file_path.stat().st_size
                result['checksum'] = get_file_hash(file_path)
                result['status'] = 'success'
                
                logger.info(f"Successfully downloaded {filename} ({result['file_size']} bytes)")
                
            elif response.status_code == 404:
                result['error_msg'] = f"File not found (HTTP 404)"
                logger.warning(f"File not found: {url}")
                
            else:
                result['error_msg'] = f"HTTP {response.status_code}: {response.reason}"
                logger.warning(f"Failed to download {url}: {result['error_msg']}")
                
        except requests.exceptions.Timeout:
            result['error_msg'] = "Request timeout"
            logger.error(f"Timeout downloading {url}")
            
        except requests.exceptions.RequestException as e:
            result['error_msg'] = str(e)
            logger.error(f"Request failed for {url}: {e}")
            
        except Exception as e:
            result['error_msg'] = str(e)
            logger.error(f"Unexpected error downloading {url}: {e}")
        
        return result

    def download_file_with_fallbacks(self, file_urls: List[str], symbol: str, language: str = 'en') -> Dict[str, Any]:
        """Download a file trying multiple URLs with enhanced validation."""
        
        # Check if already successfully downloaded
        if self._is_already_downloaded_symbol(symbol):
            logger.info(f"File {symbol} already downloaded, skipping")
            existing_entry = self.manifest[self.manifest['symbol'] == symbol].iloc[0]
            return existing_entry.to_dict()
        
        # Generate filename
        filename = self._generate_filename_from_symbol(symbol, language)
        file_path = self.raw_data_path / filename
        
        result = {
            'symbol': symbol,
            'url': None,
            'filename': filename,
            'status': 'error',
            'http_code': None,
            'file_size': None,
            'checksum': None,
            'download_date': pd.Timestamp.now(),
            'error_msg': None,
            'attempts': 0,
            'successful_url': None,
            'validation_status': 'not_attempted'
        }
        
        # Try each URL with enhanced validation
        for attempt, url in enumerate(file_urls, 1):
            result['attempts'] = attempt
            result['url'] = url
            
            logger.info(f"Attempting download {attempt}/{len(file_urls)} for {symbol}: {url}")
            
            try:
                # Rate limit between attempts
                if attempt > 1:
                    wait_time = min(2 * (2 ** (attempt - 2)), 10)  # Exponential backoff, max 10s
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    self.rate_limiter.wait()
                
                # Validate URL first
                validation = self._validate_url_before_download(url)
                if not validation['is_valid']:
                    logger.warning(f"URL {attempt} validation failed for {symbol}: {validation['error_msg']}")
                    result['error_msg'] = f"URL validation failed: {validation['error_msg']}"
                    result['validation_status'] = 'failed'
                    continue
                
                # Download with enhanced headers
                headers = {
                    'Accept': 'application/pdf,*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://digitallibrary.un.org/'
                }
                
                response = self.session.get(url, timeout=60, stream=True, headers=headers)
                result['http_code'] = response.status_code
                
                if response.status_code == 200:
                    # Download file in chunks
                    total_size = 0
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                total_size += len(chunk)
                    
                    # Validate download
                    if total_size < 1024:
                        logger.warning(f"Downloaded file {filename} is too small ({total_size} bytes)")
                        file_path.unlink(missing_ok=True)
                        result['error_msg'] = f"File too small ({total_size} bytes)"
                        continue
                    
                    # Verify PDF header
                    try:
                        with open(file_path, 'rb') as f:
                            header = f.read(4)
                            if header != b'%PDF':
                                result['error_msg'] = "Downloaded file is not a valid PDF"
                                logger.warning(f"Downloaded file {filename} is not a PDF")
                                file_path.unlink(missing_ok=True)
                                continue
                    except Exception as e:
                        result['error_msg'] = f"Failed to verify PDF header: {e}"
                        file_path.unlink(missing_ok=True)
                        continue
                    
                    # Success!
                    result['file_size'] = total_size
                    result['checksum'] = get_file_hash(file_path)
                    result['status'] = 'success'
                    result['successful_url'] = url
                    result['validation_status'] = 'passed'
                    
                    logger.info(f"Successfully downloaded {symbol} from URL {attempt}/{len(file_urls)} ({total_size} bytes)")
                    return result
                    
                else:
                    logger.warning(f"URL {attempt} returned HTTP {response.status_code} for {symbol}")
                    result['error_msg'] = f"HTTP {response.status_code}: {response.reason}"
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"URL {attempt} timed out for {symbol}")
                result['error_msg'] = "Request timeout"
                continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"URL {attempt} failed for {symbol}: {e}")
                result['error_msg'] = str(e)
                continue
                
            except Exception as e:
                logger.warning(f"Unexpected error with URL {attempt} for {symbol}: {e}")
                result['error_msg'] = str(e)
                continue
        
        # All URLs failed
        logger.error(f"All {len(file_urls)} download attempts failed for {symbol}")
        result['error_msg'] = f"All download attempts failed. Last error: {result.get('error_msg', 'Unknown error')}"
        return result

    def _is_already_downloaded_symbol(self, symbol: str) -> bool:
        """Check if a file with this symbol was already successfully downloaded."""
        if self.manifest.empty:
            return False
        return ((self.manifest['symbol'] == symbol) & 
                (self.manifest['status'] == 'success')).any()

    def _generate_filename_from_symbol(self, symbol: str, language: str) -> str:
        """Generate filename from UN document symbol."""
        safe_symbol = symbol.replace('/', '_').replace('\\', '_')
        return f"{safe_symbol}.{language}.pdf"
    
    def fetch_from_records(self, records_file: str) -> Dict[str, Any]:
        """Fetch all files from discovered records with enhanced validation."""
        logger.info(f"Starting improved fetch from {records_file}")
        
        # Load records
        try:
            records_df = pd.read_parquet(records_file)
        except Exception as e:
            logger.error(f"Failed to load records from {records_file}: {e}")
            return {'success': False, 'error': str(e)}
        
        logger.info(f"Found {len(records_df)} records to process")
        
        # Track results
        download_results = []
        successful_downloads = 0
        failed_downloads = 0
        skipped_downloads = 0
        
        for idx, record in records_df.iterrows():
            symbol = record.get('symbol', f'record_{idx}')
            language = record.get('language', 'en')
            file_urls = record.get('file_urls', [])
            
            # Handle case where file_urls might be a pandas Series or array
            if hasattr(file_urls, 'tolist'):
                file_urls = file_urls.tolist()
            elif not isinstance(file_urls, list):
                file_urls = [file_urls] if file_urls else []
            
            if not file_urls or len(file_urls) == 0:
                logger.warning(f"No file URLs found for {symbol}")
                failed_downloads += 1
                continue
                
            # Clean file URLs list
            clean_urls = [url for url in file_urls if not pd.isna(url) and url.strip()]
            
            if not clean_urls:
                logger.warning(f"No valid file URLs found for {symbol}")
                failed_downloads += 1
                continue
                
            # Use enhanced download with fallbacks
            result = self.download_file_with_fallbacks(clean_urls, symbol, language)
            download_results.append(result)
            
            if result['status'] == 'success':
                successful_downloads += 1
                logger.info(f"✅ Downloaded {symbol} using {result.get('successful_url', 'unknown URL')}")
            else:
                failed_downloads += 1
                logger.warning(f"❌ Failed to download {symbol}: {result.get('error_msg', 'Unknown error')}")
        
        # Update manifest with new results
        new_manifest_entries = pd.DataFrame(download_results)
        if not new_manifest_entries.empty:
            # Remove old entries for same symbol/url combinations
            for _, new_entry in new_manifest_entries.iterrows():
                self.manifest = self.manifest[
                    ~((self.manifest['symbol'] == new_entry['symbol']) & 
                      (self.manifest['url'] == new_entry['url']))
                ]
            
            # Add new entries
            self.manifest = pd.concat([self.manifest, new_manifest_entries], ignore_index=True)
            self._save_manifest()
        
        summary = {
            'success': True,
            'total_records': len(records_df),
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'total_files': len(self.manifest[self.manifest['status'] == 'success'])
        }
        
        logger.info(f"Fetch complete: {summary}")
        return summary
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get current download status summary."""
        if self.manifest.empty:
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'total_size_mb': 0
            }
        
        status_counts = self.manifest['status'].value_counts().to_dict()
        successful_files = self.manifest[self.manifest['status'] == 'success']
        total_size = successful_files['file_size'].fillna(0).sum()
        
        return {
            'total_files': len(self.manifest),
            'successful': status_counts.get('success', 0),
            'failed': status_counts.get('error', 0),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'files_by_status': status_counts
        }
    
    def cleanup_failed_downloads(self) -> int:
        """Remove files for failed download entries."""
        if self.manifest.empty:
            return 0
            
        failed_entries = self.manifest[self.manifest['status'] == 'error']
        cleaned_count = 0
        
        for _, entry in failed_entries.iterrows():
            file_path = self.raw_data_path / entry['filename']
            if file_path.exists():
                try:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.info(f"Removed failed download: {entry['filename']}")
                except Exception as e:
                    logger.warning(f"Failed to remove {entry['filename']}: {e}")
        
        return cleaned_count


def main():
    """Main function for standalone execution."""
    from utils import setup_logging
    
    setup_logging()
    config = load_config()
    
    fetcher = ImprovedUNReportFetcher(config)
    
    # Fetch from discovered records
    records_file = config['paths']['records_file']
    result = fetcher.fetch_from_records(records_file)
    
    if result['success']:
        print(f"Fetch complete!")
        print(f"Records processed: {result['total_records']}")
        print(f"Successful downloads: {result['successful_downloads']}")
        print(f"Failed downloads: {result['failed_downloads']}")
        print(f"Total files available: {result['total_files']}")
    else:
        print(f"Fetch failed: {result.get('error')}")
    
    # Show status
    status = fetcher.get_download_status()
    print(f"\nDownload Status:")
    print(f"Total files: {status['total_files']}")
    print(f"Successful: {status['successful']}")
    print(f"Failed: {status['failed']}")
    print(f"Total size: {status['total_size_mb']} MB")


if __name__ == "__main__":
    main()
