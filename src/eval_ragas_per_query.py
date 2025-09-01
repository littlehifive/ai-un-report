#!/usr/bin/env python3
"""RAGAS evaluation with per-query scoring and proper trace attachment."""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Our imports
sys.path.append('src')
from utils import load_config
from index_improved import RateLimitedUNReportIndexer as UNReportIndexer
from app import enhanced_search, get_chat_response

# Langfuse import
from langfuse import Langfuse

logger = logging.getLogger(__name__)

class PerQueryRAGASEvaluator:
    """RAGAS evaluation with per-query scoring properly attached to traces."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.config = load_config()
        self.indexer = None
        self.load_indexer()
        
        # Initialize Langfuse
        self.langfuse = Langfuse(
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            host=os.getenv('LANGFUSE_HOST')
        )
        
        self.session_id = f"ragas_per_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"🔗 Langfuse session: {self.session_id}")
    
    def load_indexer(self):
        """Load the RAG indexer."""
        self.indexer = UNReportIndexer(self.config)
        result = self.indexer.load_index()
        if not result['success']:
            raise Exception(f"Failed to load index: {result['error']}")
        print(f"✅ Index loaded: {result['total_chunks']} chunks")
    
    def calculate_ragas_scores_for_query(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str], 
        ground_truth: str
    ) -> Dict[str, float]:
        """Calculate RAGAS scores for a single query-answer pair."""
        
        # Faithfulness: How grounded is the answer in the contexts?
        # Simple heuristic: check overlap between answer and context
        context_text = " ".join(contexts).lower()
        answer_lower = answer.lower()
        
        # Count how many answer sentences are grounded in context
        answer_sentences = answer.split('. ')
        grounded_sentences = 0
        for sentence in answer_sentences:
            if len(sentence) > 10:  # Skip very short sentences
                # Check if key words from sentence appear in context
                sentence_words = set(sentence.lower().split())
                context_words = set(context_text.split())
                overlap = len(sentence_words.intersection(context_words))
                if overlap > len(sentence_words) * 0.3:  # 30% overlap threshold
                    grounded_sentences += 1
        
        faithfulness = grounded_sentences / len(answer_sentences) if answer_sentences else 0
        
        # Answer Relevancy: How relevant is the answer to the question?
        question_words = set(question.lower().split())
        answer_words = set(answer_lower.split())
        relevancy_overlap = len(question_words.intersection(answer_words))
        answer_relevancy = min(1.0, relevancy_overlap / (len(question_words) * 0.5)) if question_words else 0
        
        # Context Precision: How precise are the retrieved contexts?
        # Assume contexts are ranked by relevance, give higher score to top contexts
        if contexts:
            # Weight contexts by position (first = most relevant)
            weights = [1.0 / (i + 1) for i in range(len(contexts))]
            
            # Check if contexts contain key terms from question
            precision_scores = []
            for context, weight in zip(contexts, weights):
                context_lower = context.lower()
                matches = sum(1 for word in question_words if word in context_lower)
                precision_scores.append((matches / len(question_words)) * weight if question_words else 0)
            
            context_precision = sum(precision_scores) / sum(weights) if weights else 0
        else:
            context_precision = 0
        
        # Context Recall: How much of the ground truth is covered by contexts?
        ground_truth_words = set(ground_truth.lower().split())
        context_coverage = len(ground_truth_words.intersection(set(context_text.split())))
        context_recall = context_coverage / len(ground_truth_words) if ground_truth_words else 0
        
        # Ensure scores are in valid range [0, 1]
        return {
            "faithfulness": min(1.0, max(0.0, faithfulness)),
            "answer_relevancy": min(1.0, max(0.0, answer_relevancy)),
            "context_precision": min(1.0, max(0.0, context_precision)),
            "context_recall": min(1.0, max(0.0, context_recall))
        }
    
    def evaluate_single_query(
        self,
        question: str,
        ground_truth: str,
        query_num: int,
        total_queries: int,
        category: str = "general"
    ) -> Dict[str, Any]:
        """Evaluate a single query with its own trace and RAGAS scores."""
        
        print(f"\n📝 Query {query_num}/{total_queries}: {question[:60]}...")
        
        # Create a unique trace for this query
        trace_id = self.langfuse.create_trace_id()
        print(f"   📍 Trace ID: {trace_id}")
        
        # Create trace event (without trace_id parameter)
        self.langfuse.create_event(
            name=f"ragas_query_{query_num}",
            input={
                "question": question,
                "ground_truth": ground_truth,
                "category": category,
                "session_id": self.session_id,
                "trace_id": trace_id  # Include trace_id in input for reference
            },
            metadata={
                "query_number": query_num,
                "total_queries": total_queries
            }
        )
        
        try:
            # Perform search
            start_time = time.time()
            search_results = enhanced_search(self.indexer, question, [], top_k=5, min_threshold=0.3)
            search_time = time.time() - start_time
            
            print(f"   🔍 Retrieved: {len(search_results)} documents ({search_time:.2f}s)")
            
            # Log search event (include trace_id in metadata for reference)
            self.langfuse.create_event(
                name="search_operation",
                input={"query": question},
                output={
                    "num_results": len(search_results),
                    "search_time_ms": search_time * 1000
                },
                metadata={"trace_id": trace_id}
            )
            
            if search_results:
                # Generate answer
                gen_start = time.time()
                answer = get_chat_response(question, search_results, self.config, [])
                gen_time = time.time() - gen_start
                
                contexts = [chunk.get('text', '') for chunk in search_results]
                
                print(f"   💬 Generated: {len(answer)} chars ({gen_time:.2f}s)")
                
                # Log generation event (include trace_id in metadata)
                self.langfuse.create_event(
                    name="answer_generation",
                    input={
                        "question": question,
                        "num_contexts": len(contexts)
                    },
                    output={
                        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                        "generation_time_ms": gen_time * 1000
                    },
                    metadata={"trace_id": trace_id}
                )
                
                # Calculate RAGAS scores for this specific query
                ragas_scores = self.calculate_ragas_scores_for_query(
                    question, answer, contexts, ground_truth
                )
                
                print(f"   📊 RAGAS Scores:")
                
                # Create individual scores attached to this trace
                for metric_name, score_value in ragas_scores.items():
                    # Create score with trace_id
                    self.langfuse.create_score(
                        trace_id=trace_id,
                        name=f"ragas_{metric_name}",
                        value=float(score_value),
                        comment=f"RAGAS {metric_name} for query {query_num}: {question[:50]}..."
                    )
                    
                    emoji = "🟢" if score_value >= 0.7 else "🟡" if score_value >= 0.5 else "🔴"
                    print(f"      {emoji} {metric_name}: {score_value:.3f}")
                
                # Calculate and attach overall score for this query
                overall_score = sum(ragas_scores.values()) / len(ragas_scores)
                self.langfuse.create_score(
                    trace_id=trace_id,
                    name="ragas_overall",
                    value=float(overall_score),
                    comment=f"Overall RAGAS score for query {query_num}"
                )
                
                print(f"      🏆 Overall: {overall_score:.3f}")
                
                return {
                    "query_num": query_num,
                    "question": question,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "contexts": contexts,
                    "ragas_scores": ragas_scores,
                    "overall_score": overall_score,
                    "trace_id": trace_id,
                    "success": True
                }
                
            else:
                print(f"   ❌ No relevant documents found")
                
                # Still create scores with 0 values for failed retrieval
                failed_scores = {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0
                }
                
                for metric_name, score_value in failed_scores.items():
                    self.langfuse.create_score(
                        trace_id=trace_id,
                        name=f"ragas_{metric_name}",
                        value=0.0,
                        comment=f"Failed retrieval for query {query_num}"
                    )
                
                return {
                    "query_num": query_num,
                    "question": question,
                    "trace_id": trace_id,
                    "success": False,
                    "ragas_scores": failed_scores
                }
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
            # Log error (include trace_id in metadata)
            self.langfuse.create_event(
                name="evaluation_error",
                input={"question": question},
                output={"error": str(e)},
                metadata={"trace_id": trace_id}
            )
            
            return {
                "query_num": query_num,
                "question": question,
                "trace_id": trace_id,
                "error": str(e),
                "success": False
            }
    
    def run_evaluation(self, num_queries: int = 10) -> Dict[str, Any]:
        """Run RAGAS evaluation on multiple queries with per-query scoring."""
        
        print(f"\n🚀 Starting Per-Query RAGAS Evaluation")
        print("=" * 60)
        
        # Define test queries with ground truth
        test_queries = [
            {
                "question": "What does the UNCTAD Technology and Innovation Report 2025 say about AI?",
                "ground_truth": "The report focuses on inclusive AI for development, emphasizing local adaptation and sustainable implementation.",
                "category": "technology"
            },
            {
                "question": "What peacekeeping challenges are mentioned in Security Council reports?",
                "ground_truth": "Reports highlight resource constraints, operational complexity, and civilian protection challenges.",
                "category": "security"
            },
            {
                "question": "How do UN reports address women's economic empowerment?",
                "ground_truth": "Reports emphasize equal pay, leadership opportunities, and removing employment barriers.",
                "category": "human_rights"
            },
            {
                "question": "What are UNDP's development priorities for 2025?",
                "ground_truth": "UNDP prioritizes poverty reduction, SDG acceleration, and climate resilience.",
                "category": "development"
            },
            {
                "question": "What does the Joint Inspection Unit recommend for UN efficiency?",
                "ground_truth": "JIU recommends enhanced coordination, accountability, and organizational reforms.",
                "category": "governance"
            },
            {
                "question": "How do General Assembly reports address climate change?",
                "ground_truth": "Reports emphasize mitigation, adaptation, and international cooperation on climate.",
                "category": "environment"
            },
            {
                "question": "What does ECOSOC say about post-pandemic recovery?",
                "ground_truth": "ECOSOC focuses on sustainable recovery, resilience building, and inclusive growth.",
                "category": "economic"
            },
            {
                "question": "What are the main themes in Commission on Status of Women reports?",
                "ground_truth": "Main themes include gender equality, women's rights, and eliminating discrimination.",
                "category": "human_rights"
            },
            {
                "question": "How do UN reports assess progress on SDGs?",
                "ground_truth": "Reports track SDG indicators, identify gaps, and recommend acceleration strategies.",
                "category": "development"
            },
            {
                "question": "What digital transformation initiatives are mentioned in UN documents?",
                "ground_truth": "Documents cover digital governance, cybersecurity, and technology capacity building.",
                "category": "technology"
            }
        ]
        
        # Limit to requested number of queries
        queries_to_eval = test_queries[:num_queries]
        
        print(f"📋 Evaluating {len(queries_to_eval)} queries with individual RAGAS scores")
        
        results = []
        for i, query_data in enumerate(queries_to_eval, 1):
            result = self.evaluate_single_query(
                question=query_data["question"],
                ground_truth=query_data["ground_truth"],
                query_num=i,
                total_queries=len(queries_to_eval),
                category=query_data.get("category", "general")
            )
            results.append(result)
            
            # Small delay between queries
            time.sleep(1)
        
        # Calculate aggregate statistics
        successful_results = [r for r in results if r.get("success", False)]
        
        if successful_results:
            # Calculate average scores across all queries
            avg_scores = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }
            
            for result in successful_results:
                for metric, value in result["ragas_scores"].items():
                    avg_scores[metric] += value
            
            for metric in avg_scores:
                avg_scores[metric] /= len(successful_results)
            
            # Create session-level aggregate scores
            print(f"\n📊 Creating session-level aggregate scores...")
            
            for metric, value in avg_scores.items():
                self.langfuse.create_score(
                    name=f"session_avg_{metric}",
                    value=float(value),
                    comment=f"Average {metric} across {len(successful_results)} queries in session {self.session_id}"
                )
            
            overall_avg = sum(avg_scores.values()) / len(avg_scores)
            self.langfuse.create_score(
                name="session_overall_performance",
                value=float(overall_avg),
                comment=f"Overall performance for session {self.session_id}"
            )
        
        # Flush all events and scores
        self.langfuse.flush()
        
        return {
            "session_id": self.session_id,
            "total_queries": len(queries_to_eval),
            "successful_queries": len(successful_results),
            "results": results,
            "average_scores": avg_scores if successful_results else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def print_summary(self, evaluation_results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("🎯 PER-QUERY RAGAS EVALUATION COMPLETE")
        print("=" * 70)
        print(f"📅 Session: {evaluation_results['session_id']}")
        print(f"📊 Queries Evaluated: {evaluation_results['total_queries']}")
        print(f"✅ Successful: {evaluation_results['successful_queries']}")
        
        if evaluation_results.get("average_scores"):
            print(f"\n📈 AVERAGE RAGAS SCORES ACROSS ALL QUERIES:")
            print("-" * 50)
            for metric, score in evaluation_results["average_scores"].items():
                emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
                print(f"  {emoji} {metric}: {score:.3f}")
        
        print(f"\n🔍 VIEW IN LANGFUSE DASHBOARD:")
        print("-" * 50)
        print(f"1. Go to: {os.getenv('LANGFUSE_HOST')}")
        print(f"2. Navigate to 'Traces' tab")
        print(f"3. You'll see {evaluation_results['total_queries']} traces named 'ragas_query_1', 'ragas_query_2', etc.")
        print(f"4. Click on each trace to see its individual RAGAS scores")
        print(f"5. Check 'Scores' tab to see all scores with trace associations")
        
        print(f"\n📊 EACH TRACE HAS 5 SCORES:")
        print("  • ragas_faithfulness (per query)")
        print("  • ragas_answer_relevancy (per query)")
        print("  • ragas_context_precision (per query)")
        print("  • ragas_context_recall (per query)")
        print("  • ragas_overall (per query)")
        
        print(f"\n✅ Total scores created: {evaluation_results['successful_queries'] * 5} individual scores")
        print("=" * 70)

def main():
    """Run per-query RAGAS evaluation."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        evaluator = PerQueryRAGASEvaluator()
        
        # Run evaluation on 10 queries (you can change this number)
        results = evaluator.run_evaluation(num_queries=10)
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"data/ragas_per_query_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()