"""Evaluation module for testing UN Reports RAG system."""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

from utils import load_config, setup_logging
from index import UNReportIndexer
from app import get_chat_response

logger = logging.getLogger(__name__)

class UNRAGEvaluator:
    """Evaluates the UN Reports RAG system with test queries."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_queries = self._get_test_queries()
        
        # Initialize indexer
        self.indexer = UNReportIndexer(config)
        load_result = self.indexer.load_index()
        
        if not load_result['success']:
            raise RuntimeError(f"Failed to load index: {load_result['error']}")
    
    def _get_test_queries(self) -> List[Dict[str, Any]]:
        """Define test queries for evaluation."""
        return [
            {
                'query': 'What did the Secretary-General report on climate change?',
                'expected_topics': ['climate', 'environment', 'sustainable development'],
                'min_sources': 1
            },
            {
                'query': 'Recent Security Council resolutions on peacekeeping',
                'expected_topics': ['peacekeeping', 'security', 'peace'],
                'min_sources': 1
            },
            {
                'query': 'Economic and Social Council recommendations for development',
                'expected_topics': ['economic', 'social', 'development'],
                'min_sources': 1
            },
            {
                'query': 'UN reports on human rights violations',
                'expected_topics': ['human rights', 'violations', 'protection'],
                'min_sources': 1
            },
            {
                'query': 'What are the main challenges in recent UN reports?',
                'expected_topics': ['challenges', 'issues', 'problems'],
                'min_sources': 2
            },
            {
                'query': 'Secretary-General annual report findings',
                'expected_topics': ['annual', 'Secretary-General', 'findings'],
                'min_sources': 1
            },
            {
                'query': 'UN sustainable development goals progress',
                'expected_topics': ['sustainable', 'development', 'goals'],
                'min_sources': 1
            },
            {
                'query': 'Global economic situation and prospects',
                'expected_topics': ['economic', 'global', 'prospects'],
                'min_sources': 1
            },
            {
                'query': 'What does the UN say about food security?',
                'expected_topics': ['food', 'security', 'hunger'],
                'min_sources': 1
            },
            {
                'query': 'Recent UN reports on refugee situations',
                'expected_topics': ['refugee', 'displacement', 'humanitarian'],
                'min_sources': 1
            }
        ]
    
    def evaluate_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Evaluate retrieval quality for a single query."""
        logger.info(f"Evaluating retrieval for: '{query}'")
        
        results = self.indexer.search(query, top_k)
        
        evaluation = {
            'query': query,
            'retrieved_count': len(results),
            'has_results': len(results) > 0,
            'top_score': results[0]['similarity_score'] if results else 0.0,
            'avg_score': sum(r['similarity_score'] for r in results) / len(results) if results else 0.0,
            'unique_documents': len(set(r['doc_id'] for r in results)),
            'organs_covered': len(set(r['organ'] for r in results if r.get('organ'))),
            'results': results
        }
        
        return evaluation
    
    def evaluate_generation(self, query: str, retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate response generation quality."""
        logger.info(f"Evaluating generation for: '{query}'")
        
        if not retrieval_results:
            return {
                'query': query,
                'has_response': False,
                'response': "No relevant documents found.",
                'has_citations': False,
                'response_length': 0,
                'error': None
            }
        
        try:
            response = get_chat_response(query, retrieval_results, self.config)
            
            evaluation = {
                'query': query,
                'has_response': bool(response and response.strip()),
                'response': response,
                'has_citations': '[1]' in response or '[2]' in response,
                'response_length': len(response) if response else 0,
                'mentions_sources': any(chunk['symbol'] in response for chunk in retrieval_results[:3]) if response else False,
                'error': None
            }
            
            # Check for hallucination indicators
            if response and ("I don't have" in response or "not found" in response or "unable to find" in response):
                evaluation['appropriate_uncertainty'] = True
            else:
                evaluation['appropriate_uncertainty'] = False
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Generation failed for '{query}': {e}")
            return {
                'query': query,
                'has_response': False,
                'response': None,
                'has_citations': False,
                'response_length': 0,
                'error': str(e)
            }
    
    def evaluate_topic_coverage(self, query: str, expected_topics: List[str], 
                              retrieval_results: List[Dict[str, Any]], 
                              response: str) -> Dict[str, Any]:
        """Evaluate if retrieved content and response cover expected topics."""
        
        # Combine all text for topic analysis
        all_text = ""
        if response:
            all_text += response.lower()
        
        for result in retrieval_results[:3]:  # Top 3 results
            all_text += " " + result.get('text', '').lower()
        
        # Check topic coverage
        topics_found = []
        for topic in expected_topics:
            if topic.lower() in all_text:
                topics_found.append(topic)
        
        coverage_score = len(topics_found) / len(expected_topics) if expected_topics else 1.0
        
        return {
            'expected_topics': expected_topics,
            'topics_found': topics_found,
            'coverage_score': coverage_score,
            'full_coverage': coverage_score == 1.0
        }
    
    def run_single_evaluation(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation for a single test case."""
        query = test_case['query']
        expected_topics = test_case.get('expected_topics', [])
        min_sources = test_case.get('min_sources', 1)
        
        logger.info(f"Running evaluation for: '{query}'")
        
        # 1. Evaluate retrieval
        retrieval_eval = self.evaluate_retrieval(query, top_k=10)
        
        # 2. Evaluate generation
        generation_eval = self.evaluate_generation(query, retrieval_eval['results'])
        
        # 3. Evaluate topic coverage
        topic_eval = self.evaluate_topic_coverage(
            query, expected_topics, 
            retrieval_eval['results'], 
            generation_eval.get('response', '')
        )
        
        # 4. Overall assessment
        overall = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'retrieval_success': retrieval_eval['retrieved_count'] >= min_sources,
            'generation_success': generation_eval['has_response'] and generation_eval['has_citations'],
            'topic_success': topic_eval['coverage_score'] >= 0.5,  # At least 50% topic coverage
            'overall_pass': False
        }
        
        # Determine overall pass/fail
        overall['overall_pass'] = (
            overall['retrieval_success'] and 
            overall['generation_success'] and
            overall['topic_success']
        )
        
        # Combine all results
        result = {
            **overall,
            'retrieval': retrieval_eval,
            'generation': generation_eval,
            'topics': topic_eval
        }
        
        return result
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all test queries."""
        logger.info(f"Starting full evaluation with {len(self.test_queries)} test queries")
        
        results = []
        passed = 0
        failed = 0
        
        for test_case in self.test_queries:
            try:
                result = self.run_single_evaluation(test_case)
                results.append(result)
                
                if result['overall_pass']:
                    passed += 1
                    logger.info(f"✅ PASS: {result['query']}")
                else:
                    failed += 1
                    logger.warning(f"❌ FAIL: {result['query']}")
                    
            except Exception as e:
                logger.error(f"Evaluation failed for '{test_case['query']}': {e}")
                failed += 1
                results.append({
                    'query': test_case['query'],
                    'timestamp': datetime.now().isoformat(),
                    'overall_pass': False,
                    'error': str(e)
                })
        
        # Calculate summary statistics
        summary = {
            'total_queries': len(self.test_queries),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self.test_queries) if self.test_queries else 0.0,
            'retrieval_success_rate': sum(1 for r in results if r.get('retrieval_success', False)) / len(results),
            'generation_success_rate': sum(1 for r in results if r.get('generation_success', False)) / len(results),
            'topic_success_rate': sum(1 for r in results if r.get('topic_success', False)) / len(results),
        }
        
        return {
            'summary': summary,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_evaluation_report(self, evaluation_results: Dict[str, Any], 
                             output_file: str = "evaluation_report.json") -> None:
        """Save evaluation results to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
    
    def print_summary(self, evaluation_results: Dict[str, Any]) -> None:
        """Print evaluation summary to console."""
        summary = evaluation_results['summary']
        
        print("\n🧪 UN Reports RAG Evaluation Summary")
        print("=" * 40)
        print(f"Total queries:           {summary['total_queries']}")
        print(f"Passed:                  {summary['passed']}")
        print(f"Failed:                  {summary['failed']}")
        print(f"Overall pass rate:       {summary['pass_rate']:.1%}")
        print(f"Retrieval success rate:  {summary['retrieval_success_rate']:.1%}")
        print(f"Generation success rate: {summary['generation_success_rate']:.1%}")
        print(f"Topic coverage rate:     {summary['topic_success_rate']:.1%}")
        
        print(f"\n📊 Detailed Results:")
        print("-" * 20)
        
        for result in evaluation_results['results']:
            status = "✅ PASS" if result.get('overall_pass') else "❌ FAIL"
            query = result['query'][:60] + "..." if len(result['query']) > 60 else result['query']
            print(f"{status} {query}")
            
            if not result.get('overall_pass') and 'error' not in result:
                issues = []
                if not result.get('retrieval_success'):
                    issues.append("retrieval")
                if not result.get('generation_success'):
                    issues.append("generation")
                if not result.get('topic_success'):
                    issues.append("topics")
                
                if issues:
                    print(f"      Issues: {', '.join(issues)}")

def main():
    """Main function for standalone execution."""
    setup_logging()
    config = load_config()
    
    try:
        evaluator = UNRAGEvaluator(config)
        
        # Run full evaluation
        results = evaluator.run_full_evaluation()
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save detailed report
        output_file = "evaluation_report.json"
        evaluator.save_evaluation_report(results, output_file)
        
        print(f"\n📄 Detailed report saved to: {output_file}")
        
        # Return exit code based on pass rate
        if results['summary']['pass_rate'] >= 0.7:  # 70% pass rate threshold
            print("\n🎉 Evaluation PASSED (≥70% success rate)")
            return 0
        else:
            print(f"\n⚠️  Evaluation NEEDS IMPROVEMENT (<70% success rate)")
            return 1
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())