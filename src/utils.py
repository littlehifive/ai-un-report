"""Shared utilities for UN RAG system."""

import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import os
from datetime import datetime, timedelta

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/un_rag.log', mode='a')
        ]
    )

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    # If running from src/ directory, look in parent directory
    if not Path(config_path).exists() and Path(f"../{config_path}").exists():
        config_path = f"../{config_path}"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_date_window(days_back: int = 365) -> tuple[datetime, datetime]:
    """Get date range for the last N days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date

def rate_limit(delay_seconds: float) -> None:
    """Apply rate limiting delay."""
    time.sleep(delay_seconds)

def get_file_hash(file_path: Path) -> str:
    """Get SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def safe_filename(text: str, max_length: int = 100) -> str:
    """Create a safe filename from text."""
    # Remove/replace unsafe characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    safe_text = "".join(c if c in safe_chars else "_" for c in text)
    return safe_text[:max_length]

def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def load_openai_key() -> Optional[str]:
    """Load OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, delay_seconds: float = 5.0):
        self.delay_seconds = delay_seconds
        self.last_call = 0.0
    
    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_call
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self.last_call = time.time()