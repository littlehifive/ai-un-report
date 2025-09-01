"""Simple RAG evaluation for Python 3.7 compatibility."""

import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Our imports
from utils import load_config
from index_improved import RateLimitedUNReportIndexer as UNReportIndexer
from app import enhanced_search, get_chat_response, validate_citations

logger = logging.getLogger(__name__)

class SimpleRAGEvaluator:
    """Simple evaluation framework compatible with Python 3.7."""
    
    def __init__(self):
        self.config = load_config()
        self.indexer = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.load_indexer()
    
    def load_indexer(self):
        """Load the RAG indexer."""
        self.indexer = UNReportIndexer(self.config)
        result = self.indexer.load_index()
        if not result['success']:
            raise Exception(f"Failed to load index: {result['error']}")
        logger.info(f"Loaded index with {result['total_chunks']} chunks")
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """Create evaluation test cases."""
        return [
            {
                "question": "What recent reports discuss climate change impacts?",
                "expected_keywords": ["climate", "change", "impacts", "environment"],
                "query_type": "search",
                "expect_results": True
            },
            {
                "question": "Show me Statistical Commission reports from 2025",  
                "expected_keywords": ["statistical", "commission", "2025"],
                "query_type": "search",
                "expect_results": True
            },
            {
                "question": "What does the Technology and Innovation Report 2025 say about AI?",
                "expected_keywords": ["technology", "innovation", "artificial", "intelligence"],
                "query_type": "document_specific", 
                "expect_results": True
            },
            {
                "question": "What are the main challenges in peacekeeping mentioned in recent reports?",
                "expected_keywords": ["peacekeeping", "challenges", "security"],
                "query_type": "analytical",
                "expect_results": True
            },
            {
                "question": "How do recent economic reports assess global recovery?",
                "expected_keywords": ["economic", "recovery", "global"],
                "query_type": "analytical",
                "expect_results": True
            },
            {
                "question": "What is the point of UN reports?",
                "expected_keywords": ["transparency", "accountability", "purpose"],
                "query_type": "general",
                "expect_results": False  # Should handle with general knowledge
            },
            {
                "question": "Tell me about lunar colonization in UN reports",
                "expected_keywords": ["lunar", "colonization"],
                "query_type": "irrelevant",
                "expect_results": False
            }
        ]
    
    def calculate_keyword_overlap(self, text: str, keywords: List[str]) -> float:
        """Calculate overlap between response and expected keywords."""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return matches / len(keywords) if keywords else 0
    
    def calculate_response_quality(self, question: str, answer: str) -> Dict[str, float]:
        """Calculate basic response quality metrics."""
        metrics = {}
        
        # Length appropriateness (50-500 words is good)
        word_count = len(answer.split())
        if 50 <= word_count <= 500:
            metrics['length_score'] = 1.0
        elif word_count < 50:
            metrics['length_score'] = max(0, word_count / 50)
        else:
            metrics['length_score'] = max(0, 1 - (word_count - 500) / 500)
        
        # Citation presence (should have citations for factual queries)
        citation_pattern = r'\[\d+\]'
        citations = len(re.findall(citation_pattern, answer))
        metrics['citation_count'] = citations
        
        # Semantic similarity using TF-IDF
        try:
            docs = [question, answer]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            metrics['semantic_similarity'] = float(similarity)
        except:
            metrics['semantic_similarity'] = 0.0
        
        # Error detection
        error_indicators = ['error', 'failed', '❌', 'exception']
        metrics['has_error'] = any(indicator.lower() in answer.lower() for indicator in error_indicators)
        
        return metrics
    
    def evaluate_retrieval(self, question: str, results: List[Dict], expected_results: bool) -> Dict[str, Any]:
        """Evaluate retrieval performance."""
        metrics = {
            'retrieved_count': len(results),
            'expected_results': expected_results,
            'retrieval_success': (len(results) > 0) == expected_results
        }
        
        if results:
            # Average similarity score
            scores = [r.get('similarity_score', 0) for r in results]
            metrics['avg_similarity'] = sum(scores) / len(scores)
            metrics['min_similarity'] = min(scores)
            metrics['max_similarity'] = max(scores)
            
            # Document diversity (unique symbols)
            unique_symbols = set(r.get('symbol', '') for r in results if r.get('symbol'))
            metrics['document_diversity'] = len(unique_symbols)
        else:
            metrics['avg_similarity'] = 0
            metrics['min_similarity'] = 0  
            metrics['max_similarity'] = 0
            metrics['document_diversity'] = 0
        
        return metrics
    
    def evaluate_single_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single query."""
        question = test_case['question']
        expected_keywords = test_case['expected_keywords']
        query_type = test_case['query_type']
        expect_results = test_case['expect_results']
        
        start_time = time.time()
        
        try:
            # Get search results
            results = enhanced_search(self.indexer, question, [], top_k=5, min_threshold=0.3)
            
            # Generate response
            if results:
                answer = get_chat_response(question, results, self.config, [])
            else:
                # Handle no results case
                if any(word in question.lower() for word in ['what is', 'why', 'how', 'purpose', 'point']):
                    answer = f"I don't have specific information from recent UN reports about '{question}'. This appears to be a general question that would require broader knowledge."
                else:
                    answer = "I couldn't find relevant information in the UN reports corpus for your query."
            
            response_time = time.time() - start_time
            
            # Evaluate different aspects
            retrieval_metrics = self.evaluate_retrieval(question, results, expect_results)
            quality_metrics = self.calculate_response_quality(question, answer)
            keyword_overlap = self.calculate_keyword_overlap(answer, expected_keywords)
            
            # Check for citation hallucinations
            validated_answer = validate_citations(answer, results)
            citation_issues = len(answer) != len(validated_answer)
            
            return {
                'question': question,
                'answer': answer[:200] + "..." if len(answer) > 200 else answer,
                'query_type': query_type,
                'response_time_ms': response_time * 1000,
                'retrieval': retrieval_metrics,
                'quality': quality_metrics,
                'keyword_overlap': keyword_overlap,
                'citation_issues': citation_issues,
                'success': retrieval_metrics['retrieval_success'] and not quality_metrics['has_error']
            }
            
        except Exception as e:
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'query_type': query_type, 
                'response_time_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'success': False
            }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation."""
        logger.info("Starting RAG evaluation...")
        
        test_cases = self.create_test_cases()
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Evaluating {i}/{len(test_cases)}: {test_case['question'][:50]}...")
            result = self.evaluate_single_query(test_case)
            results.append(result)
        
        # Calculate aggregate metrics
        successful_queries = [r for r in results if r.get('success', False)]
        total_queries = len(results)
        
        # Retrieval metrics
        retrieval_success_rate = sum(1 for r in results if r.get('retrieval', {}).get('retrieval_success', False)) / total_queries
        avg_retrieved = sum(r.get('retrieval', {}).get('retrieved_count', 0) for r in results) / total_queries
        
        # Quality metrics
        avg_response_time = sum(r.get('response_time_ms', 0) for r in results) / total_queries
        citation_issues_count = sum(1 for r in results if r.get('citation_issues', False))
        
        # Keyword overlap
        avg_keyword_overlap = sum(r.get('keyword_overlap', 0) for r in results) / total_queries
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': total_queries,
            'successful_queries': len(successful_queries),
            'success_rate': len(successful_queries) / total_queries,
            'avg_response_time_ms': avg_response_time,
            'retrieval_success_rate': retrieval_success_rate,
            'avg_documents_retrieved': avg_retrieved,
            'citation_issues': citation_issues_count,
            'avg_keyword_overlap': avg_keyword_overlap
        }
        
        return {
            'summary': summary,
            'detailed_results': results,
            'test_cases': test_cases
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save evaluation results."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"simple_eval_{timestamp}.json"
        
        results_file = Path("data") / filename
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        return str(results_file)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print human-readable summary."""
        summary = results['summary']
        
        print("\n" + "="*60)
        print("RAG EVALUATION RESULTS")
        print("="*60)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Queries: {summary['total_queries']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Avg Response Time: {summary['avg_response_time_ms']:.0f}ms")
        print()
        
        print("RETRIEVAL METRICS:")
        print(f"  Retrieval Success Rate: {summary['retrieval_success_rate']:.2%}")
        print(f"  Avg Documents Retrieved: {summary['avg_documents_retrieved']:.1f}")
        print()
        
        print("QUALITY METRICS:")
        print(f"  Citation Issues: {summary['citation_issues']}/{summary['total_queries']}")
        print(f"  Keyword Overlap: {summary['avg_keyword_overlap']:.2%}")
        print()
        
        # Query type breakdown
        type_stats = {}
        for result in results['detailed_results']:
            qtype = result['query_type']
            if qtype not in type_stats:
                type_stats[qtype] = {'total': 0, 'success': 0}
            type_stats[qtype]['total'] += 1
            if result.get('success', False):
                type_stats[qtype]['success'] += 1
        
        print("QUERY TYPE BREAKDOWN:")
        for qtype, stats in type_stats.items():
            success_rate = stats['success'] / stats['total']
            print(f"  {qtype.title()}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
        
        print("="*60)

def main():
    """Run simple evaluation."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        evaluator = SimpleRAGEvaluator()
        results = evaluator.run_evaluation()
        evaluator.print_summary(results)
        
        # Save results
        results_file = evaluator.save_results(results)
        print(f"\nDetailed results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()