"""RAGAS evaluation for UN Reports RAG system."""

import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any
import asyncio
from datetime import datetime
import json

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset

# Our imports
from utils import load_config, load_openai_key
from index_improved import RateLimitedUNReportIndexer as UNReportIndexer
from app import enhanced_search, get_chat_response

logger = logging.getLogger(__name__)

class RAGASEvaluator:
    """RAGAS-based evaluation for our UN Reports RAG system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config()
        self.indexer = None
        self.load_indexer()
        
        # RAGAS metrics to evaluate
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity
        ]
    
    def load_indexer(self):
        """Load the RAG indexer."""
        self.indexer = UNReportIndexer(self.config)
        result = self.indexer.load_index()
        if not result['success']:
            raise Exception(f"Failed to load index: {result['error']}")
        logger.info(f"Loaded index with {result['total_chunks']} chunks")
    
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create evaluation test cases."""
        test_cases = [
            # Search-focused queries
            {
                "question": "What recent reports discuss climate change impacts?",
                "ground_truth": "Recent UN reports discuss climate change impacts including extreme weather events, adaptation needs, and policy recommendations.",
                "query_type": "search"
            },
            {
                "question": "Show me Statistical Commission reports from 2025",
                "ground_truth": "The Statistical Commission published reports in 2025 including the 56th session report covering statistical methodologies and data governance.",
                "query_type": "search"
            },
            {
                "question": "What does the Technology and Innovation Report 2025 say about AI?",
                "ground_truth": "The Technology and Innovation Report 2025 discusses inclusive artificial intelligence for development, covering AI applications and policy frameworks.",
                "query_type": "document_specific"
            },
            
            # Conversational queries
            {
                "question": "What are the main challenges in peacekeeping mentioned in recent reports?",
                "ground_truth": "Recent UN reports identify peacekeeping challenges including resource constraints, complex conflict environments, and coordination issues.",
                "query_type": "analytical"
            },
            {
                "question": "How do recent economic reports assess global recovery?",
                "ground_truth": "Recent UN economic reports assess global recovery focusing on post-pandemic resilience, sustainable development financing, and economic inequalities.",
                "query_type": "analytical"
            },
            
            # Edge cases
            {
                "question": "What is the point of UN reports?",
                "ground_truth": "UN reports serve transparency, accountability, and informed decision-making purposes within the UN system.",
                "query_type": "general"
            },
            {
                "question": "Tell me about lunar colonization in UN reports",
                "ground_truth": "No relevant information found in recent UN reports about lunar colonization.",
                "query_type": "irrelevant"
            }
        ]
        
        return test_cases
    
    async def evaluate_query(self, question: str, ground_truth: str, query_type: str) -> Dict[str, Any]:
        """Evaluate a single query using our RAG system."""
        try:
            # Get search results
            results = enhanced_search(self.indexer, question, [], top_k=5, min_threshold=0.3)
            
            # Generate response
            if results:
                answer = get_chat_response(question, results, self.config, [])
                contexts = [chunk['text'] for chunk in results]
            else:
                answer = "I couldn't find relevant information in the UN reports corpus for your query."
                contexts = []
            
            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "query_type": query_type,
                "num_retrieved": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating query '{question}': {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth,
                "query_type": query_type,
                "num_retrieved": 0
            }
    
    async def run_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """Run complete RAGAS evaluation."""
        logger.info("Starting RAGAS evaluation...")
        
        # Create test dataset
        test_cases = self.create_test_dataset()
        
        # Evaluate each query
        evaluation_data = []
        for test_case in test_cases:
            logger.info(f"Evaluating: {test_case['question'][:50]}...")
            result = await self.evaluate_query(
                test_case["question"], 
                test_case["ground_truth"],
                test_case["query_type"]
            )
            evaluation_data.append(result)
        
        # Convert to RAGAS dataset format
        dataset = Dataset.from_list(evaluation_data)
        
        # Run RAGAS evaluation
        logger.info("Running RAGAS metrics evaluation...")
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )
        
        # Process results
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(test_cases),
            "ragas_scores": dict(evaluation_result),
            "detailed_results": evaluation_data,
            "summary": {
                "avg_faithfulness": evaluation_result.get('faithfulness', 0),
                "avg_relevancy": evaluation_result.get('answer_relevancy', 0),
                "avg_context_precision": evaluation_result.get('context_precision', 0),
                "avg_context_recall": evaluation_result.get('context_recall', 0)
            }
        }
        
        if save_results:
            # Save results
            results_file = Path("data") / f"ragas_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results_file.parent.mkdir(exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {results_file}")
        
        return results
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary of results."""
        print("\n" + "="*60)
        print("RAGAS EVALUATION RESULTS")
        print("="*60)
        
        summary = results['summary']
        print(f"Total Queries Evaluated: {results['total_queries']}")
        print(f"Timestamp: {results['timestamp']}")
        print()
        
        print("RAGAS METRICS:")
        print(f"  Faithfulness:      {summary['avg_faithfulness']:.3f}")
        print(f"  Answer Relevancy:   {summary['avg_relevancy']:.3f}")
        print(f"  Context Precision:  {summary['avg_context_precision']:.3f}")
        print(f"  Context Recall:     {summary['avg_context_recall']:.3f}")
        print()
        
        # Query type breakdown
        type_breakdown = {}
        for result in results['detailed_results']:
            query_type = result['query_type']
            if query_type not in type_breakdown:
                type_breakdown[query_type] = {"count": 0, "retrieved": 0}
            type_breakdown[query_type]["count"] += 1
            type_breakdown[query_type]["retrieved"] += result['num_retrieved']
        
        print("QUERY TYPE BREAKDOWN:")
        for qtype, stats in type_breakdown.items():
            avg_retrieved = stats["retrieved"] / stats["count"]
            print(f"  {qtype.title()}: {stats['count']} queries, avg {avg_retrieved:.1f} docs retrieved")
        
        print("="*60)

async def main():
    """Run RAGAS evaluation."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        evaluator = RAGASEvaluator()
        results = await evaluator.run_evaluation()
        evaluator.print_results_summary(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())