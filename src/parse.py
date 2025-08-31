"""Parsing module for extracting text from UN report files."""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pymupdf  # PyMuPDF
import trafilatura
import re
from datetime import datetime

from utils import load_config, ensure_dir

logger = logging.getLogger(__name__)

class UNReportParser:
    """Parses downloaded UN report files into structured chunks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parsing_config = config.get('parsing', {})
        self.chunk_tokens = self.parsing_config.get('chunk_tokens', 1200)
        self.overlap_tokens = self.parsing_config.get('overlap_tokens', 150)
        self.min_chunk_length = self.parsing_config.get('min_chunk_length', 100)
        
        self.raw_data_path = Path(config['paths']['raw_data'])
        self.parsed_data_path = Path(config['paths']['parsed_data'])
        ensure_dir(self.parsed_data_path)
        
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 characters per token on average)."""
        return len(text) // 4
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and common PDF artifacts
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\f', '\n', text)  # Form feed characters
        
        # Clean up common UN document artifacts
        text = re.sub(r'United Nations\s+[A-Z]/\d+/\d+', '', text)
        text = re.sub(r'GE\.\d+-\d+.*?(?=\n|\Z)', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def parse_pdf(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse PDF file and extract text with structure."""
        logger.info(f"Parsing PDF: {file_path}")
        
        try:
            doc = pymupdf.open(file_path)
            full_text = ""
            sections = []
            current_section = {"title": "Introduction", "content": "", "page": 1}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                if not page_text.strip():
                    continue
                    
                # Clean page text
                page_text = self._clean_text(page_text)
                full_text += page_text + "\n"
                
                # Simple section detection (look for common heading patterns)
                lines = page_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if line looks like a heading (all caps, numbered, etc.)
                    if (len(line) < 100 and 
                        (line.isupper() or 
                         re.match(r'^[IVX]+\.\s', line) or 
                         re.match(r'^\d+\.\s', line) or
                         line.startswith('Chapter') or
                         line.startswith('Section'))):
                        
                        # Save previous section if it has content
                        if current_section["content"].strip():
                            sections.append(current_section.copy())
                        
                        # Start new section
                        current_section = {
                            "title": line,
                            "content": "",
                            "page": page_num + 1
                        }
                    else:
                        current_section["content"] += line + " "
            
            # Don't forget the last section
            if current_section["content"].strip():
                sections.append(current_section)
            
            doc.close()
            
            # If no sections were detected, create one big section
            if not sections:
                sections = [{
                    "title": "Document Content",
                    "content": full_text,
                    "page": 1
                }]
            
            logger.info(f"Extracted {len(full_text)} characters in {len(sections)} sections")
            return self._clean_text(full_text), sections
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            return "", []
    
    def parse_html(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse HTML file and extract text."""
        logger.info(f"Parsing HTML: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Use trafilatura for clean text extraction
            text = trafilatura.extract(html_content)
            
            if not text:
                logger.warning(f"No text extracted from {file_path}")
                return "", []
            
            cleaned_text = self._clean_text(text)
            
            # For HTML, create simple sections based on paragraphs
            sections = [{
                "title": "Document Content",
                "content": cleaned_text,
                "page": 1
            }]
            
            logger.info(f"Extracted {len(cleaned_text)} characters from HTML")
            return cleaned_text, sections
            
        except Exception as e:
            logger.error(f"Failed to parse HTML {file_path}: {e}")
            return "", []
    
    def create_chunks(self, text: str, sections: List[Dict[str, Any]], 
                     doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text with metadata."""
        if not text.strip():
            return []
        
        chunks = []
        chunk_id = 0
        
        # Convert token limits to character estimates
        chunk_chars = self.chunk_tokens * 4  # ~4 chars per token
        overlap_chars = self.overlap_tokens * 4
        
        # Process each section
        for section in sections:
            section_text = section["content"].strip()
            if len(section_text) < self.min_chunk_length:
                continue
                
            # Split section into chunks
            start = 0
            while start < len(section_text):
                end = min(start + chunk_chars, len(section_text))
                
                # Try to break at sentence or paragraph boundaries
                if end < len(section_text):
                    # Look for sentence break within last 200 characters
                    last_period = section_text.rfind('.', end - 200, end)
                    last_newline = section_text.rfind('\n', end - 200, end)
                    
                    break_point = max(last_period, last_newline)
                    if break_point > start + chunk_chars // 2:  # Don't make chunks too small
                        end = break_point + 1
                
                chunk_text = section_text[start:end].strip()
                
                if len(chunk_text) >= self.min_chunk_length:
                    chunk = {
                        'chunk_id': f"{doc_metadata.get('symbol', 'unknown')}_{chunk_id:03d}",
                        'doc_id': doc_metadata.get('symbol', 'unknown'),
                        'symbol': doc_metadata.get('symbol', ''),
                        'title': doc_metadata.get('title', ''),
                        'date': doc_metadata.get('date', ''),
                        'organ': doc_metadata.get('organ', ''),
                        'language': doc_metadata.get('language', 'en'),
                        'source_url': doc_metadata.get('record_url', ''),
                        'section_title': section.get('title', ''),
                        'section_page': section.get('page', 1),
                        'text': chunk_text,
                        'char_count': len(chunk_text),
                        'token_estimate': self._estimate_tokens(chunk_text),
                        'chunk_index': chunk_id
                    }
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Move start position with overlap
                if end >= len(section_text):
                    break
                start = max(start + chunk_chars - overlap_chars, end - overlap_chars)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def parse_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a single file and return chunks."""
        logger.info(f"Processing file: {file_path}")
        
        # Determine file type and parse accordingly
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            text, sections = self.parse_pdf(file_path)
        elif suffix in ['.html', '.htm']:
            text, sections = self.parse_html(file_path)
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return []
        
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return []
        
        # Create chunks
        chunks = self.create_chunks(text, sections, metadata)
        
        return chunks
    
    def parse_all_files(self, records_file: str, manifest_file: Optional[str] = None) -> Dict[str, Any]:
        """Parse all downloaded files and create chunks."""
        logger.info(f"Starting parsing from {records_file}")
        
        # Load records and manifest
        try:
            records_df = pd.read_parquet(records_file)
        except Exception as e:
            logger.error(f"Failed to load records: {e}")
            return {'success': False, 'error': str(e)}
        
        # Load manifest to map symbols to filenames
        manifest_df = None
        if manifest_file and Path(manifest_file).exists():
            try:
                manifest_df = pd.read_parquet(manifest_file)
                manifest_df = manifest_df[manifest_df['status'] == 'success']
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        
        all_chunks = []
        processed_files = 0
        failed_files = 0
        
        for idx, record in records_df.iterrows():
            symbol = record.get('symbol', f'record_{idx}')
            
            # Find corresponding file
            file_path = None
            if manifest_df is not None:
                matching_files = manifest_df[manifest_df['symbol'] == symbol]
                if not matching_files.empty:
                    filename = matching_files.iloc[0]['filename']
                    file_path = self.raw_data_path / filename
            
            if file_path is None or not file_path.exists():
                logger.warning(f"File not found for {symbol}")
                failed_files += 1
                continue
            
            # Parse file
            try:
                chunks = self.parse_file(file_path, record.to_dict())
                all_chunks.extend(chunks)
                processed_files += 1
                logger.info(f"Processed {symbol}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                failed_files += 1
        
        # Save chunks to parquet
        if all_chunks:
            chunks_df = pd.DataFrame(all_chunks)
            output_file = self.config['paths']['chunks_file']
            ensure_dir(Path(output_file).parent)
            
            chunks_df.to_parquet(output_file, index=False)
            logger.info(f"Saved {len(all_chunks)} chunks to {output_file}")
        
        summary = {
            'success': True,
            'total_records': len(records_df),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_chunks': len(all_chunks),
            'output_file': self.config['paths']['chunks_file']
        }
        
        logger.info(f"Parsing complete: {summary}")
        return summary
    
    def get_parsing_stats(self, chunks_file: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about parsed chunks."""
        chunks_file = chunks_file or self.config['paths']['chunks_file']
        
        if not Path(chunks_file).exists():
            return {'total_chunks': 0, 'total_documents': 0}
        
        try:
            chunks_df = pd.read_parquet(chunks_file)
            
            stats = {
                'total_chunks': len(chunks_df),
                'total_documents': chunks_df['doc_id'].nunique(),
                'avg_chunk_length': chunks_df['char_count'].mean(),
                'total_characters': chunks_df['char_count'].sum(),
                'languages': chunks_df['language'].value_counts().to_dict(),
                'organs': chunks_df['organ'].value_counts().to_dict(),
                'date_range': {
                    'earliest': chunks_df['date'].min(),
                    'latest': chunks_df['date'].max()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute stats: {e}")
            return {'error': str(e)}

def main():
    """Main function for standalone execution."""
    from utils import setup_logging
    
    setup_logging()
    config = load_config()
    
    parser = UNReportParser(config)
    
    # Parse all files
    records_file = config['paths']['records_file']
    manifest_file = Path(config['paths']['raw_data']) / "files_manifest.parquet"
    
    result = parser.parse_all_files(records_file, str(manifest_file))
    
    if result['success']:
        print(f"Parsing complete!")
        print(f"Total records: {result['total_records']}")
        print(f"Processed files: {result['processed_files']}")
        print(f"Failed files: {result['failed_files']}")
        print(f"Total chunks: {result['total_chunks']}")
        print(f"Output: {result['output_file']}")
    else:
        print(f"Parsing failed: {result.get('error')}")
    
    # Show stats
    stats = parser.get_parsing_stats()
    if 'error' not in stats:
        print(f"\nParsing Statistics:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Avg chunk length: {stats.get('avg_chunk_length', 0):.0f} chars")
        print(f"Languages: {list(stats.get('languages', {}).keys())}")

if __name__ == "__main__":
    main()