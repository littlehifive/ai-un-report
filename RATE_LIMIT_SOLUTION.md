# OpenAI API Rate Limiting Solution

## Problem Identified

You encountered this error:
```
ERROR:index:Failed to generate embeddings: Error code: 429 - {'error': {'message': 'Rate limit reached for text-embedding-3-small in organization org-GUi94cfJHNADVg6Zj65phcnJ on tokens per min (TPM): Limit 1000000, Used 974150, Requested 34558. Please try again in 522ms. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}
```

This indicates that the system was trying to process too many documents at once, exceeding OpenAI's rate limits:
- **TPM Limit**: 1,000,000 tokens per minute
- **Used**: 974,150 tokens
- **Requested**: 34,558 tokens
- **Result**: Rate limit exceeded

## Root Causes

1. **No rate limiting**: The original system processed all chunks without respecting API limits
2. **Large batch sizes**: Processing too many documents simultaneously
3. **No token estimation**: Not tracking token usage before making requests
4. **No fallback mechanisms**: No graceful handling when rate limits are hit
5. **No retry logic**: Failed requests weren't retried with appropriate delays

## Solutions Implemented

### 1. Rate-Limited Indexing System (`index_improved.py`)

#### Key Features:
- **Token estimation**: Estimates token count before making requests
- **Rate limit tracking**: Monitors TPM and RPM usage
- **Automatic waiting**: Waits when approaching rate limits
- **Small batch processing**: Processes documents in small batches (8 texts per batch)
- **Retry logic**: Automatically retries failed requests with exponential backoff
- **Fallback to local**: Uses local BGE embeddings when OpenAI is unavailable

#### Rate Limiting Logic:
```python
def _wait_for_rate_limit(self, estimated_tokens: int):
    """Wait if necessary to respect rate limits."""
    while not self._check_rate_limits(1, estimated_tokens):
        wait_time = self.reset_time - current_time + 1
        logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
```

### 2. Configuration Updates

Added rate limiting settings to `config.yaml`:
```yaml
rate_limiting:
  max_queries_per_hour: 20
  cooldown_seconds: 3
  enable_fallback: true
  # OpenAI API rate limits
  openai_tpm_limit: 1000000  # Tokens per minute
  openai_rpm_limit: 3500     # Requests per minute
  batch_size: 8              # Small batch size to avoid rate limits
```

### 3. Chunk Limiting

Added optional chunk limiting to prevent processing too many documents:
```python
def create_index(self, chunks_file: Optional[str] = None, max_chunks: Optional[int] = None):
    # Limit chunks if specified
    if max_chunks and len(chunks_df) > max_chunks:
        logger.info(f"Limiting chunks from {len(chunks_df)} to {max_chunks}")
        chunks_df = chunks_df.head(max_chunks)
```

## How the New System Works

### 1. Pre-Request Validation
1. **Estimate tokens** for the batch
2. **Check rate limits** (TPM and RPM)
3. **Wait if necessary** before making request
4. **Make request** with confidence it won't exceed limits

### 2. Post-Request Updates
1. **Update counters** with actual token usage
2. **Add cooldown** between batches
3. **Handle errors** with retry logic

### 3. Error Handling
1. **Rate limit errors**: Wait and retry
2. **Other errors**: Log and continue
3. **Fallback**: Use local embeddings if OpenAI fails

## Usage Instructions

### 1. Use the Rate-Limited Indexer
```python
from index_improved import RateLimitedUNReportIndexer

indexer = RateLimitedUNReportIndexer(config)

# Create index with chunk limit to avoid rate limits
max_chunks = 1000  # Limit to 1000 chunks
result = indexer.create_index(max_chunks=max_chunks)
```

### 2. Test the System
```bash
# Test rate-limited indexing
python test_rate_limited_indexing.py

# Test with small batches
python src/index_improved.py
```

### 3. Monitor Progress
The system provides detailed logging:
```
Processing batch 1/125 (8 texts, ~2000 tokens)
✅ Batch 1/125 completed successfully
Rate limit reached. Waiting 45.2 seconds...
Processing batch 2/125 (8 texts, ~1800 tokens)
```

## Configuration Options

### Rate Limiting Settings
- `batch_size`: Number of texts per batch (default: 8)
- `cooldown_seconds`: Delay between batches (default: 3)
- `openai_tpm_limit`: Tokens per minute limit (default: 1,000,000)
- `openai_rpm_limit`: Requests per minute limit (default: 3,500)

### Chunk Limiting
- `max_chunks`: Maximum number of chunks to process
- `target_documents`: Target number of documents (affects chunk count)

## Benefits of the New System

1. **No more rate limit errors**: Proper rate limiting prevents 429 errors
2. **Efficient processing**: Small batches with cooldowns
3. **Automatic retries**: Failed requests are retried automatically
4. **Fallback support**: Uses local embeddings when OpenAI is unavailable
5. **Progress tracking**: Clear logging of progress and rate limit status
6. **Configurable limits**: Adjust batch sizes and delays as needed

## Testing Results

The improved system successfully:
- ✅ **Processes small batches** without hitting rate limits
- ✅ **Estimates tokens** accurately before requests
- ✅ **Waits when needed** to respect rate limits
- ✅ **Retries failed requests** with appropriate delays
- ✅ **Falls back to local** embeddings when OpenAI is unavailable
- ✅ **Provides clear progress** logging

## Recommendations

### 1. Start Small
Begin with a small number of chunks to test the system:
```python
result = indexer.create_index(max_chunks=100)
```

### 2. Monitor Usage
Watch the logs to understand your rate limit usage:
```
Request count: 45
Token count: 125000
Can make request with 50000 tokens: True
```

### 3. Adjust Settings
Modify batch size and cooldown based on your needs:
```yaml
rate_limiting:
  batch_size: 4      # Smaller batches for more conservative approach
  cooldown_seconds: 5  # Longer delays between batches
```

### 4. Use Local Fallback
For large datasets, consider using local embeddings:
```yaml
retrieval:
  embedding_provider: "local_bge"
  fallback_to_local: true
```

## Next Steps

1. **Deploy the improved system** in production
2. **Monitor rate limit usage** to optimize settings
3. **Consider local embeddings** for large-scale processing
4. **Implement progressive indexing** for very large datasets
5. **Add monitoring** to track API usage and costs

## Conclusion

The new rate-limited indexing system successfully addresses the OpenAI API rate limiting issue by:
- ✅ **Implementing proper rate limiting** with token estimation
- ✅ **Using small batch processing** to avoid overwhelming the API
- ✅ **Adding automatic retry logic** for failed requests
- ✅ **Providing fallback mechanisms** when OpenAI is unavailable
- ✅ **Offering clear progress tracking** and error handling

You should now be able to process your UN reports without encountering rate limit errors.
