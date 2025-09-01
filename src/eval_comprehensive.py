#!/usr/bin/env python3
"""Comprehensive evaluation using 20 golden queries with Langfuse tracking."""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Our imports
from utils import load_config
from index_improved import RateLimitedUNReportIndexer as UNReportIndexer
from app import enhanced_search, get_chat_response, validate_citations

# Langfuse import
try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None

logger = logging.getLogger(__name__)

class ComprehensiveRAGEvaluator:
    """Comprehensive RAG evaluation with 20 golden queries and Langfuse tracking."""
    
    def __init__(self):
        """Initialize evaluator with Langfuse tracking."""
        self.config = load_config()
        self.indexer = None
        self.load_indexer()
        
        # Initialize Langfuse tracking
        self.langfuse = None
        self.evaluation_session_id = f"comprehensive_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if Langfuse and os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
            try:
                self.langfuse = Langfuse(
                    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
                    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
                    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                )
                print(f"🔗 Langfuse tracking enabled - Session: {self.evaluation_session_id}")
            except Exception as e:
                print(f"⚠️  Langfuse initialization failed: {e}")
                self.langfuse = None
        else:
            print("⚠️  Langfuse tracking disabled - add API keys to .env file")
    
    def load_indexer(self):
        """Load the RAG indexer."""
        self.indexer = UNReportIndexer(self.config)
        result = self.indexer.load_index()
        if not result['success']:
            raise Exception(f"Failed to load index: {result['error']}")
        logger.info(f"Loaded index with {result['total_chunks']} chunks")
        print(f"✅ Index loaded: {result['total_chunks']} chunks")
    
    def load_golden_dataset(self) -> List[Dict[str, Any]]:
        """Load the most recent golden dataset."""
        data_dir = Path("data")
        golden_files = list(data_dir.glob("golden_dataset_*.json"))
        
        if not golden_files:
            raise FileNotFoundError("No golden dataset found. Run create_golden_dataset.py first.")
        
        # Get the most recent file
        latest_file = max(golden_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            dataset = json.load(f)
        
        print(f"📂 Loaded golden dataset: {latest_file}")
        print(f"   Total queries: {dataset['metadata']['total_queries']}")
        print(f"   Categories: {', '.join(dataset['metadata']['categories'])}")
        
        return dataset['queries']
    
    def calculate_metrics(self, question: str, answer: str, ground_truth: str, 
                         retrieved_docs: int, expected_docs: List[str]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # 1. Retrieval Success Rate
        metrics['retrieval_success'] = 1.0 if retrieved_docs > 0 else 0.0
        
        # 2. Response Length Appropriateness (50-500 words)
        word_count = len(answer.split())
        if 50 <= word_count <= 500:
            metrics['length_score'] = 1.0
        elif word_count < 50:
            metrics['length_score'] = max(0, word_count / 50)
        else:
            metrics['length_score'] = max(0, 1 - (word_count - 500) / 500)
        
        # 3. Citation Count
        import re
        citations = len(re.findall(r'\\[[0-9]+\\]', answer))
        metrics['citation_count'] = citations
        metrics['has_citations'] = 1.0 if citations > 0 else 0.0
        
        # 4. Keyword Overlap with Ground Truth
        answer_words = set(answer.lower().split())
        ground_truth_words = set(ground_truth.lower().split())
        
        if len(ground_truth_words) > 0:
            overlap = len(answer_words.intersection(ground_truth_words))
            metrics['keyword_overlap'] = overlap / len(ground_truth_words)
        else:
            metrics['keyword_overlap'] = 0.0
        
        # 5. Error Detection
        error_indicators = ['error', 'failed', '❌', 'exception', 'could not', 'unable to']
        metrics['has_error'] = 1.0 if any(indicator.lower() in answer.lower() for indicator in error_indicators) else 0.0
        
        # 6. Relevance Score (based on retrieved documents)
        max_docs = 10  # Our typical top_k
        metrics['relevance_score'] = min(1.0, retrieved_docs / max_docs)
        
        return metrics
    
    def evaluate_single_query(self, query_data: Dict[str, Any], query_num: int, total_queries: int) -> Dict[str, Any]:
        """Evaluate a single query with comprehensive tracking."""
        question = query_data['question']
        ground_truth = query_data['ground_truth']
        query_type = query_data['query_type']
        category = query_data['category']
        expected_docs = query_data['expected_docs']
        
        print(f"📝 Query {query_num}/{total_queries}: {question[:80]}...")
        print(f"   Category: {category} | Type: {query_type}")
        
        start_time = time.time()
        
        # Create Langfuse trace
        trace_id = None
        if self.langfuse:
            try:
                trace_id = self.langfuse.create_trace_id()
                
                # Log query start
                self.langfuse.create_event(
                    name="comprehensive_eval_query_start",
                    input={
                        "question": question,
                        "ground_truth": ground_truth,
                        "query_type": query_type,
                        "category": category,
                        "expected_docs": expected_docs
                    },
                    metadata={
                        "session_id": self.evaluation_session_id,
                        "query_number": query_num,
                        "total_queries": total_queries
                    }
                )
            except Exception as e:
                print(f"   ⚠️  Langfuse query logging failed: {e}")
        
        try:
            # Perform enhanced search
            search_results = enhanced_search(self.indexer, question, [], top_k=10, min_threshold=0.3)
            print(f"   🔍 Retrieved: {len(search_results)} documents")
            
            # Generate response
            if search_results:
                answer = get_chat_response(question, search_results, self.config, [])
                
                # Validate citations
                validated_answer = validate_citations(answer, search_results)
                citation_issues = len(answer) != len(validated_answer)
                
                print(f"   💬 Generated: {len(answer)} chars")
                
            else:
                answer = f"I couldn't find relevant information in the UN reports corpus for your query about {category}."
                citation_issues = False
                
                print("   ❌ No relevant documents found")
            
            response_time = time.time() - start_time
            
            # Calculate comprehensive metrics
            metrics = self.calculate_metrics(
                question, answer, ground_truth, 
                len(search_results), expected_docs
            )
            
            # Log search and generation to Langfuse
            if self.langfuse and trace_id:
                try:
                    # Log search results
                    self.langfuse.create_event(
                        name="search_results", 
                        input={"query": question},
                        output={
                            "num_results": len(search_results),
                            "result_symbols": [r.get('symbol', '') for r in search_results[:5]],
                            "avg_similarity": sum(r.get('similarity_score', 0) for r in search_results) / len(search_results) if search_results else 0
                        },
                        metadata={"search_threshold": 0.3, "top_k": 10}
                    )
                    
                    # Log generation
                    self.langfuse.create_event(
                        name="response_generation",
                        input={
                            "question": question,
                            "context_docs": len(search_results)
                        },
                        output=answer[:500] + "..." if len(answer) > 500 else answer,
                        metadata={
                            "model": "gpt-4o-mini",
                            "response_time_ms": response_time * 1000,
                            "word_count": len(answer.split()),
                            "citation_issues": citation_issues
                        }
                    )
                    
                    # Log metrics as scores
                    for metric_name, score in metrics.items():
                        if isinstance(score, (int, float)) and metric_name != 'citation_count':
                            self.langfuse.create_score(
                                name=f"eval_{metric_name}",
                                value=float(score),
                                comment=f"{metric_name} for query: {question[:50]}..."
                            )
                    
                except Exception as e:
                    print(f"   ⚠️  Langfuse detailed logging failed: {e}")
            
            # Determine overall success
            success = (
                metrics['retrieval_success'] > 0 and
                metrics['has_error'] == 0 and
                metrics['length_score'] > 0.3
            )
            
            print(f"   ✅ Success: {success} | Time: {response_time:.1f}s")
            
            return {
                'question': question,
                'answer': answer,
                'ground_truth': ground_truth,
                'query_type': query_type,
                'category': category,
                'expected_docs': expected_docs,
                'response_time_ms': response_time * 1000,
                'retrieved_docs': len(search_results),
                'citation_issues': citation_issues,
                'metrics': metrics,
                'success': success,
                'trace_id': trace_id
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"   ❌ Error: {str(e)}")
            
            # Log error to Langfuse
            if self.langfuse and trace_id:
                try:
                    self.langfuse.create_event(
                        name="evaluation_error",
                        input={"question": question},
                        output={"error": str(e)},
                        metadata={"error_time_ms": error_time * 1000}
                    )
                except:
                    pass
            
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'ground_truth': ground_truth,
                'query_type': query_type,
                'category': category,
                'response_time_ms': error_time * 1000,
                'error': str(e),
                'success': False,
                'trace_id': trace_id
            }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on all 20 golden queries."""
        print("🚀 Starting Comprehensive RAG Evaluation")
        print("=" * 60)
        
        # Load golden dataset
        golden_queries = self.load_golden_dataset()
        
        print(f"\n🎯 Evaluating {len(golden_queries)} golden queries...")
        print("=" * 60)
        
        # Run evaluation on all queries
        results = []
        category_stats = {}
        type_stats = {}
        
        for i, query_data in enumerate(golden_queries, 1):
            result = self.evaluate_single_query(query_data, i, len(golden_queries))
            results.append(result)
            
            # Track category and type statistics
            category = result['category']
            query_type = result['query_type']
            
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'success': 0}
            if query_type not in type_stats:
                type_stats[query_type] = {'total': 0, 'success': 0}
                
            category_stats[category]['total'] += 1
            type_stats[query_type]['total'] += 1
            
            if result['success']:
                category_stats[category]['success'] += 1
                type_stats[query_type]['success'] += 1
            
            # Small delay to avoid rate limits
            time.sleep(1)
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r.get('success', False)]
        
        # Calculate average metrics from successful results
        avg_metrics = {}
        if successful_results:
            metric_names = successful_results[0].get('metrics', {}).keys()
            for metric in metric_names:
                values = [r['metrics'][metric] for r in successful_results if 'metrics' in r]
                avg_metrics[metric] = sum(values) / len(values) if values else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.evaluation_session_id,
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "avg_response_time_ms": sum(r.get('response_time_ms', 0) for r in results) / len(results),
            "avg_retrieved_docs": sum(r.get('retrieved_docs', 0) for r in results) / len(results),
            "citation_issues": sum(1 for r in results if r.get('citation_issues', False)),
            "avg_metrics": avg_metrics,
            "category_performance": {
                cat: {
                    "success_rate": stats['success'] / stats['total'],
                    "total": stats['total'],
                    "successful": stats['success']
                } for cat, stats in category_stats.items()
            },
            "query_type_performance": {
                qtype: {
                    "success_rate": stats['success'] / stats['total'],
                    "total": stats['total'], 
                    "successful": stats['success']
                } for qtype, stats in type_stats.items()
            }
        }
        
        # Send summary to Langfuse
        if self.langfuse:
            try:
                # Create summary scores
                self.langfuse.create_score(
                    name="overall_success_rate",
                    value=summary['success_rate'],
                    comment=f"Comprehensive evaluation session {self.evaluation_session_id}"
                )
                
                for category, perf in summary['category_performance'].items():
                    self.langfuse.create_score(
                        name=f"category_success_{category}",
                        value=perf['success_rate'],
                        comment=f"Success rate for {category} queries"
                    )
                
                print("   📊 Summary scores sent to Langfuse")
            except Exception as e:
                print(f"   ⚠️  Failed to send summary to Langfuse: {e}")
        
        return {
            "summary": summary,
            "detailed_results": results,
            "category_stats": category_stats,
            "query_type_stats": type_stats
        }
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save comprehensive evaluation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comprehensive_eval_{timestamp}.json"
        filepath = Path("data") / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filepath}")
        return str(filepath)
    
    def print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print detailed summary of comprehensive evaluation."""
        summary = results['summary']
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE RAG EVALUATION RESULTS (20 Golden Queries)")
        print("="*80)
        print(f"📅 Session: {summary['session_id']}")
        print(f"📊 Total Queries: {summary['total_queries']}")
        print(f"✅ Success Rate: {summary['success_rate']:.1%} ({summary['successful_queries']}/{summary['total_queries']})")
        print(f"⏱️  Avg Response Time: {summary['avg_response_time_ms']:.0f}ms")
        print(f"📄 Avg Documents Retrieved: {summary['avg_retrieved_docs']:.1f}")
        print(f"🚫 Citation Issues: {summary['citation_issues']}")
        
        if summary.get('avg_metrics'):
            print(f"\n📈 AVERAGE METRICS:")
            print("-" * 50)
            metrics = summary['avg_metrics']
            print(f"  🎯 Retrieval Success: {metrics.get('retrieval_success', 0):.1%}")
            print(f"  📝 Length Score: {metrics.get('length_score', 0):.1%}")
            print(f"  🔗 Has Citations: {metrics.get('has_citations', 0):.1%}")
            print(f"  🎪 Keyword Overlap: {metrics.get('keyword_overlap', 0):.1%}")
            print(f"  📊 Relevance Score: {metrics.get('relevance_score', 0):.1%}")
        
        print(f"\n🏷️  CATEGORY PERFORMANCE:")
        print("-" * 50)
        for category, perf in summary['category_performance'].items():
            emoji = "🟢" if perf['success_rate'] >= 0.8 else "🟡" if perf['success_rate'] >= 0.6 else "🔴"
            print(f"  {emoji} {category.title()}: {perf['success_rate']:.1%} ({perf['successful']}/{perf['total']})")
        
        print(f"\n📝 QUERY TYPE PERFORMANCE:")
        print("-" * 50)
        for qtype, perf in summary['query_type_performance'].items():
            emoji = "🟢" if perf['success_rate'] >= 0.8 else "🟡" if perf['success_rate'] >= 0.6 else "🔴"
            print(f"  {emoji} {qtype.replace('_', ' ').title()}: {perf['success_rate']:.1%} ({perf['successful']}/{perf['total']})")
        
        print(f"\n🔗 LANGFUSE DASHBOARD:")
        print("-" * 50)
        if self.langfuse:
            print(f"  ✅ Session tracked: {summary['session_id']}")
            print(f"  📊 View at: {os.getenv('LANGFUSE_HOST')}")
            print(f"  🔍 Search for: {summary['session_id']}")
            print(f"  📈 Check Traces, Events, and Scores tabs")
        else:
            print("  ❌ Langfuse tracking disabled")
        
        print("="*80)

def main():
    """Run comprehensive evaluation."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    try:
        evaluator = ComprehensiveRAGEvaluator()
        results = evaluator.run_comprehensive_evaluation()
        evaluator.print_comprehensive_summary(results)
        
        # Save results
        results_file = evaluator.save_results(results)
        
        # Flush Langfuse events
        if evaluator.langfuse:
            try:
                evaluator.langfuse.flush()
                print("\n🚀 All evaluation data sent to Langfuse dashboard!")
            except Exception as e:
                print(f"⚠️  Failed to flush Langfuse: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()