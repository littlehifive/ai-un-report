#!/usr/bin/env python3
"""RAGAS evaluation with Langfuse integration for UN Reports RAG system."""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Our imports
from utils import load_config
from index_improved import RateLimitedUNReportIndexer as UNReportIndexer
from app import enhanced_search, get_chat_response
from langfuse_integration import langfuse_tracker

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision, 
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
except ImportError as e:
    print(f"❌ Missing RAGAS dependencies: {e}")
    print("Please run: pip install ragas datasets")
    sys.exit(1)

logger = logging.getLogger(__name__)

class UNRAGRagasEvaluator:
    """RAGAS evaluation with Langfuse tracking for UN Reports RAG."""
    
    def __init__(self):
        """Initialize evaluator with Langfuse tracking."""
        self.config = load_config()
        self.indexer = None
        self.load_indexer()
        
        # Initialize Langfuse tracking
        self.evaluation_session_id = f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.main_trace_id = None
        
        if langfuse_tracker.enabled:
            self.main_trace_id = langfuse_tracker.start_conversation(
                session_id=self.evaluation_session_id,
                user_id="evaluation_system"
            )
            print(f"🔗 Langfuse tracking enabled - Session: {self.evaluation_session_id}")
        else:
            print("⚠️  Langfuse tracking disabled - add API keys to .env file")
    
    def load_indexer(self):
        """Load the RAG indexer."""
        self.indexer = UNReportIndexer(self.config)
        result = self.indexer.load_index()
        if not result['success']:
            raise Exception(f"Failed to load index: {result['error']}")
        logger.info(f"Loaded index with {result['total_chunks']} chunks")
    
    def create_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Create comprehensive evaluation dataset for RAGAS."""
        return [
            {
                "question": "What recent reports discuss climate change impacts on global governance?",
                "ground_truth": "Recent UN reports address climate change impacts on governance, infrastructure resilience, and humanitarian action, with specific focus on climate adaptation and mitigation strategies.",
                "query_type": "analytical",
                "expected_documents": ["climate", "governance", "adaptation"]
            },
            {
                "question": "Show me the Statistical Commission reports from 2025 with their main findings",
                "ground_truth": "The Statistical Commission report from 2025 documents the fifty-sixth session held from March 4-7, 2025, covering statistical methods and data frameworks.",
                "query_type": "search",
                "expected_documents": ["statistical", "commission", "2025"]
            },
            {
                "question": "What does the Technology and Innovation Report 2025 say about artificial intelligence applications?",
                "ground_truth": "The Technology and Innovation Report 2025 focuses on AI's transformative potential for development, emphasizing inclusive growth and digital transformation strategies.",
                "query_type": "document_specific", 
                "expected_documents": ["technology", "innovation", "artificial", "intelligence"]
            },
            {
                "question": "What are the main peacekeeping challenges mentioned in recent Security Council reports?",
                "ground_truth": "Recent reports highlight peacekeeping challenges including resource constraints, operational complexity, civilian protection, and coordination between international actors.",
                "query_type": "analytical",
                "expected_documents": ["peacekeeping", "security", "challenges"]
            },
            {
                "question": "How do recent ECOSOC reports assess global economic recovery post-pandemic?",
                "ground_truth": "ECOSOC reports assess global recovery through multiple dimensions including sustainable development, economic resilience, and coordinated policy responses to ongoing challenges.",
                "query_type": "analytical",
                "expected_documents": ["economic", "recovery", "ECOSOC"]
            },
            {
                "question": "What initiatives does the Secretary-General propose for digital transformation in UN operations?",
                "ground_truth": "The Secretary-General's reports propose comprehensive digital transformation initiatives including modernized systems, enhanced cybersecurity, and improved digital service delivery.",
                "query_type": "document_specific",
                "expected_documents": ["secretary-general", "digital", "transformation"]
            },
            {
                "question": "How do General Assembly resolutions address sustainable development financing mechanisms?",
                "ground_truth": "General Assembly resolutions outline various financing mechanisms for sustainable development including innovative partnerships, blended finance, and enhanced international cooperation.",
                "query_type": "policy_analysis",
                "expected_documents": ["general assembly", "sustainable", "financing"]
            }
        ]
    
    def generate_rag_response(self, question: str, trace_id: str = None) -> Dict[str, Any]:
        """Generate RAG response with Langfuse tracking."""
        try:
            # Perform enhanced search
            results = enhanced_search(self.indexer, question, [], top_k=5, min_threshold=0.3)
            
            # Track search if Langfuse enabled
            if langfuse_tracker.enabled and trace_id:
                langfuse_tracker.track_search(
                    trace_id,
                    question,
                    results,
                    {"top_k": 5, "threshold": 0.3, "evaluation": True}
                )
            
            # Generate response
            if results:
                answer = get_chat_response(question, results, self.config, [])
                contexts = [chunk.get('text', '') for chunk in results]
                
                # Track generation if Langfuse enabled
                if langfuse_tracker.enabled and trace_id:
                    langfuse_tracker.track_generation(
                        trace_id,
                        question,
                        results,
                        answer,
                        {
                            "model": self.config.get("openai", {}).get("chat_model", "gpt-4o-mini"),
                            "temperature": 0.1,
                            "evaluation": True
                        }
                    )
                
                return {
                    "answer": answer,
                    "contexts": contexts,
                    "retrieved_docs": len(results),
                    "success": True
                }
            else:
                return {
                    "answer": "I couldn't find relevant information in the UN reports corpus for your query.",
                    "contexts": [],
                    "retrieved_docs": 0,
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "contexts": [],
                "retrieved_docs": 0,
                "success": False
            }
    
    def run_ragas_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive RAGAS evaluation."""
        logger.info("Starting RAGAS evaluation with Langfuse tracking...")
        
        # Create evaluation dataset
        eval_data = self.create_evaluation_dataset()
        
        # Generate responses for each question
        evaluation_results = []
        
        for i, item in enumerate(eval_data, 1):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            logger.info(f"Evaluating {i}/{len(eval_data)}: {question[:50]}...")
            
            # Create individual trace for this evaluation
            individual_trace_id = None
            if langfuse_tracker.enabled:
                individual_trace_id = langfuse_tracker.start_conversation(
                    session_id=f"{self.evaluation_session_id}_q{i}",
                    user_id="ragas_evaluator"
                )
            
            # Generate RAG response
            rag_result = self.generate_rag_response(question, individual_trace_id)
            
            evaluation_results.append({
                "question": question,
                "answer": rag_result["answer"],
                "contexts": rag_result["contexts"],
                "ground_truth": ground_truth,
                "query_type": item["query_type"],
                "retrieved_docs": rag_result["retrieved_docs"],
                "success": rag_result["success"]
            })
        
        # Create RAGAS dataset
        ragas_dataset = Dataset.from_list([
            {
                "question": item["question"],
                "answer": item["answer"], 
                "contexts": item["contexts"],
                "ground_truth": item["ground_truth"]
            }
            for item in evaluation_results
        ])
        
        # Define RAGAS metrics
        metrics = [
            faithfulness,           # Measures hallucination (0-1, higher better)
            answer_relevancy,       # How relevant answer is to question (0-1, higher better)
            context_precision,      # Precision of retrieved contexts (0-1, higher better) 
            context_recall,         # Recall of retrieved contexts (0-1, higher better)
            answer_similarity,      # Semantic similarity to ground truth (0-1, higher better)
            answer_correctness      # Factual correctness (0-1, higher better)
        ]
        
        print("🧮 Running RAGAS evaluation (this may take a few minutes)...")
        
        # Run RAGAS evaluation
        try:
            ragas_results = evaluate(
                dataset=ragas_dataset,
                metrics=metrics,
                llm=None,  # Will use default OpenAI
                embeddings=None  # Will use default
            )
            
            # Extract scores
            scores = dict(ragas_results)
            
            # Calculate aggregate metrics
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(evaluation_results),
                "successful_retrievals": sum(1 for r in evaluation_results if r["success"]),
                "avg_retrieved_docs": sum(r["retrieved_docs"] for r in evaluation_results) / len(evaluation_results),
                "ragas_scores": scores
            }
            
            # Track evaluation summary in Langfuse
            if langfuse_tracker.enabled and self.main_trace_id:
                # Add RAGAS scores to main trace
                for metric_name, score in scores.items():
                    langfuse_tracker.langfuse.score(
                        trace_id=self.main_trace_id,
                        name=f"ragas_{metric_name}",
                        value=float(score),
                        comment=f"RAGAS {metric_name} score from evaluation"
                    )
            
            return {
                "summary": summary,
                "detailed_results": evaluation_results,
                "ragas_scores": scores,
                "dataset_info": {
                    "num_questions": len(eval_data),
                    "question_types": [item["query_type"] for item in eval_data]
                }
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            
            # Track error in Langfuse
            if langfuse_tracker.enabled and self.main_trace_id:
                langfuse_tracker.track_error(
                    self.main_trace_id,
                    e,
                    {"evaluation_stage": "ragas_computation"}
                )
            
            raise
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save evaluation results."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ragas_evaluation_{timestamp}.json"
        
        results_file = Path("data") / filename
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        return str(results_file)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print human-readable summary of RAGAS results."""
        summary = results['summary']
        ragas_scores = results['ragas_scores']
        
        print("\n" + "="*70)
        print("🎯 RAGAS EVALUATION RESULTS WITH LANGFUSE TRACKING")
        print("="*70)
        print(f"📅 Timestamp: {summary['timestamp']}")
        print(f"📊 Total Queries: {summary['total_queries']}")
        print(f"✅ Successful Retrievals: {summary['successful_retrievals']}")
        print(f"📄 Avg Documents Retrieved: {summary['avg_retrieved_docs']:.1f}")
        print()
        
        print("🧮 RAGAS METRICS:")
        print("-" * 40)
        
        # Define score interpretations
        score_interpretations = {
            "faithfulness": "Factual accuracy (hallucination detection)",
            "answer_relevancy": "Answer relevance to question", 
            "context_precision": "Precision of retrieved contexts",
            "context_recall": "Recall of retrieved contexts",
            "answer_similarity": "Semantic similarity to ground truth",
            "answer_correctness": "Factual correctness of answer"
        }
        
        for metric, score in ragas_scores.items():
            interpretation = score_interpretations.get(metric, "Unknown metric")
            score_emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
            print(f"  {score_emoji} {metric}: {score:.3f} - {interpretation}")
        
        print()
        print("📈 PERFORMANCE ANALYSIS:")
        print("-" * 40)
        
        # Performance analysis
        faithfulness_score = ragas_scores.get('faithfulness', 0)
        relevancy_score = ragas_scores.get('answer_relevancy', 0) 
        precision_score = ragas_scores.get('context_precision', 0)
        
        if faithfulness_score >= 0.8:
            print("  ✅ Excellent factual accuracy - minimal hallucinations")
        elif faithfulness_score >= 0.6:
            print("  ⚠️  Good factual accuracy - some minor issues")  
        else:
            print("  ❌ Low factual accuracy - significant hallucinations detected")
        
        if relevancy_score >= 0.7:
            print("  ✅ High answer relevancy - responses match questions well")
        else:
            print("  ⚠️  Room for improvement in answer relevancy")
        
        if precision_score >= 0.6:
            print("  ✅ Good context precision - retrieved docs are relevant")
        else:
            print("  ⚠️  Context precision could be improved")
        
        print()
        print("🔗 LANGFUSE TRACKING:")
        print("-" * 40)
        if langfuse_tracker.enabled:
            print(f"  ✅ Session tracked: {self.evaluation_session_id}")
            print(f"  📊 View detailed results at: https://cloud.langfuse.com")
            print(f"  🔍 Look for traces with session ID: {self.evaluation_session_id}")
        else:
            print("  ❌ Langfuse tracking disabled - add API keys to .env")
        
        print("="*70)

def main():
    """Run RAGAS evaluation with Langfuse integration."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        evaluator = UNRAGRagasEvaluator()
        results = evaluator.run_ragas_evaluation()
        evaluator.print_summary(results)
        
        # Save results
        results_file = evaluator.save_results(results)
        print(f"\n📁 Detailed results saved to: {results_file}")
        
        # Flush Langfuse events
        if langfuse_tracker.enabled:
            langfuse_tracker.flush()
            print("🚀 All data sent to Langfuse dashboard!")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()