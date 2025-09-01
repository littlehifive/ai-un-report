#!/usr/bin/env python3
"""Simple RAGAS evaluation for UN Reports RAG system."""

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

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision, 
        context_recall
    )
    from datasets import Dataset
    from langfuse import Langfuse
except ImportError as e:
    print(f"❌ Missing RAGAS dependencies: {e}")
    print("Please run: pip install ragas datasets langfuse")
    sys.exit(1)

logger = logging.getLogger(__name__)

class UNRAGRagasEvaluator:
    """RAGAS evaluation for UN Reports RAG."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.config = load_config()
        self.indexer = None
        self.load_indexer()
        
        # Initialize Langfuse if credentials available
        self.langfuse = None
        if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
            try:
                self.langfuse = Langfuse(
                    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
                    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
                    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                )
                self.evaluation_session_id = f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    
    def create_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Create comprehensive evaluation dataset for RAGAS."""
        return [
            {
                "question": "What recent reports discuss climate change impacts on global governance?",
                "ground_truth": "Recent UN reports address climate change impacts on governance, infrastructure resilience, and humanitarian action, with specific focus on climate adaptation and mitigation strategies.",
            },
            {
                "question": "Show me the Statistical Commission reports from 2025 with their main findings",
                "ground_truth": "The Statistical Commission report from 2025 documents the fifty-sixth session held from March 4-7, 2025, covering statistical methods and data frameworks.",
            },
            {
                "question": "What does the Technology and Innovation Report 2025 say about artificial intelligence applications?",
                "ground_truth": "The Technology and Innovation Report 2025 focuses on AI's transformative potential for development, emphasizing inclusive growth and digital transformation strategies.",
            },
            {
                "question": "What are the main peacekeeping challenges mentioned in recent Security Council reports?", 
                "ground_truth": "Recent reports highlight peacekeeping challenges including resource constraints, operational complexity, civilian protection, and coordination between international actors.",
            },
            {
                "question": "How do recent ECOSOC reports assess global economic recovery post-pandemic?",
                "ground_truth": "ECOSOC reports assess global recovery through multiple dimensions including sustainable development, economic resilience, and coordinated policy responses to ongoing challenges.",
            }
        ]
    
    def generate_rag_response(self, question: str) -> Dict[str, Any]:
        """Generate RAG response."""
        try:
            # Perform enhanced search
            results = enhanced_search(self.indexer, question, [], top_k=5, min_threshold=0.3)
            
            # Generate response
            if results:
                answer = get_chat_response(question, results, self.config, [])
                contexts = [chunk.get('text', '') for chunk in results]
                
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
        print("🚀 Starting RAGAS evaluation...")
        print("=" * 60)
        
        # Create evaluation dataset
        eval_data = self.create_evaluation_dataset()
        
        # Generate responses for each question
        evaluation_results = []
        
        for i, item in enumerate(eval_data, 1):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            print(f"📝 Evaluating {i}/{len(eval_data)}: {question[:60]}...")
            
            # Track in Langfuse if available
            trace_id = None
            if self.langfuse:
                try:
                    trace_id = self.langfuse.create_trace_id()
                    
                    # Create basic event
                    event_data = {
                        "name": "ragas_evaluation_query",
                        "input": {"question": question, "ground_truth": ground_truth},
                        "metadata": {
                            "session_id": self.evaluation_session_id,
                            "query_number": i,
                            "total_queries": len(eval_data)
                        }
                    }
                    
                    # Try different event creation methods
                    try:
                        self.langfuse.create_event(trace_id=trace_id, **event_data)
                    except TypeError:
                        # Try alternative API
                        try:
                            self.langfuse.create_event(**event_data)
                        except:
                            pass  # Skip Langfuse logging if API is incompatible
                            
                except Exception as e:
                    print(f"   ⚠️  Langfuse tracking failed: {e}")
            
            # Generate RAG response
            rag_result = self.generate_rag_response(question)
            
            evaluation_results.append({
                "question": question,
                "answer": rag_result["answer"],
                "contexts": rag_result["contexts"],
                "ground_truth": ground_truth,
                "retrieved_docs": rag_result["retrieved_docs"],
                "success": rag_result["success"],
                "trace_id": trace_id
            })
            
            print(f"   ✅ Generated response ({rag_result['retrieved_docs']} docs retrieved)")
        
        # Create RAGAS dataset
        print()
        print("🧮 Creating RAGAS dataset...")
        ragas_dataset = Dataset.from_list([
            {
                "question": item["question"],
                "answer": item["answer"], 
                "contexts": item["contexts"],
                "ground_truth": item["ground_truth"]
            }
            for item in evaluation_results if item["success"]
        ])
        
        if len(ragas_dataset) == 0:
            print("❌ No successful responses to evaluate!")
            return {"error": "No successful responses generated"}
        
        print(f"   📊 Dataset created with {len(ragas_dataset)} samples")
        
        # Define RAGAS metrics
        metrics = [
            faithfulness,           # Measures hallucination (0-1, higher better)
            answer_relevancy,       # How relevant answer is to question (0-1, higher better)
            context_precision,      # Precision of retrieved contexts (0-1, higher better) 
            context_recall,         # Recall of retrieved contexts (0-1, higher better)
        ]
        
        print()
        print("🔬 Running RAGAS evaluation (this may take a few minutes)...")
        print("   This involves multiple OpenAI API calls for evaluation...")
        
        # Run RAGAS evaluation
        try:
            ragas_results = evaluate(
                dataset=ragas_dataset,
                metrics=metrics
            )
            
            # Extract scores
            scores = dict(ragas_results)
            
            # Send scores to Langfuse if available
            if self.langfuse:
                try:
                    for metric_name, score in scores.items():
                        self.langfuse.create_score(
                            name=f"ragas_{metric_name}",
                            value=float(score),
                            comment=f"RAGAS {metric_name} score from evaluation session {self.evaluation_session_id}"
                        )
                    print("   📤 Scores sent to Langfuse dashboard")
                except Exception as e:
                    print(f"   ⚠️  Failed to send scores to Langfuse: {e}")
            
            # Calculate aggregate metrics
            summary = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.evaluation_session_id,
                "total_queries": len(evaluation_results),
                "successful_retrievals": sum(1 for r in evaluation_results if r["success"]),
                "avg_retrieved_docs": sum(r["retrieved_docs"] for r in evaluation_results) / len(evaluation_results),
                "ragas_scores": scores
            }
            
            return {
                "summary": summary,
                "detailed_results": evaluation_results,
                "ragas_scores": scores,
                "dataset_info": {
                    "num_questions": len(eval_data),
                    "successful_evaluations": len(ragas_dataset)
                }
            }
            
        except Exception as e:
            print(f"❌ RAGAS evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": str(e),
                "summary": {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.evaluation_session_id,
                    "total_queries": len(evaluation_results),
                    "successful_retrievals": sum(1 for r in evaluation_results if r["success"]),
                    "error": "RAGAS evaluation failed"
                },
                "detailed_results": evaluation_results
            }
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save evaluation results."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ragas_evaluation_{timestamp}.json"
        
        results_file = Path("data") / filename
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_file}")
        return str(results_file)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print human-readable summary of RAGAS results."""
        if "error" in results:
            print(f"\n❌ Evaluation failed: {results['error']}")
            return
            
        summary = results['summary']
        ragas_scores = results.get('ragas_scores', {})
        
        print("\n" + "="*70)
        print("🎯 RAGAS EVALUATION RESULTS")
        print("="*70)
        print(f"📅 Timestamp: {summary['timestamp']}")
        print(f"📊 Total Queries: {summary['total_queries']}")
        print(f"✅ Successful Retrievals: {summary['successful_retrievals']}")
        print(f"📄 Avg Documents Retrieved: {summary.get('avg_retrieved_docs', 0):.1f}")
        print()
        
        if ragas_scores:
            print("🧮 RAGAS METRICS:")
            print("-" * 50)
            
            # Define score interpretations
            score_interpretations = {
                "faithfulness": "Factual accuracy (low hallucination)",
                "answer_relevancy": "Answer relevance to question", 
                "context_precision": "Precision of retrieved contexts",
                "context_recall": "Recall of retrieved contexts"
            }
            
            for metric, score in ragas_scores.items():
                interpretation = score_interpretations.get(metric, "Unknown metric")
                score_emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
                print(f"  {score_emoji} {metric}: {score:.3f} - {interpretation}")
            
            print()
            print("📈 PERFORMANCE ANALYSIS:")
            print("-" * 50)
            
            # Performance analysis
            faithfulness_score = ragas_scores.get('faithfulness', 0)
            relevancy_score = ragas_scores.get('answer_relevancy', 0) 
            
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
        
        print()
        print("🔗 LANGFUSE DASHBOARD:")
        print("-" * 50)
        if self.langfuse:
            print(f"  ✅ Session tracked: {summary.get('session_id', 'unknown')}")
            print(f"  📊 View results at: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
            print(f"  🔍 Search for session: {summary.get('session_id', 'unknown')}")
        else:
            print("  ❌ Langfuse tracking disabled")
        
        print("="*70)

def main():
    """Run RAGAS evaluation."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    try:
        evaluator = UNRAGRagasEvaluator()
        results = evaluator.run_ragas_evaluation()
        evaluator.print_summary(results)
        
        # Save results
        results_file = evaluator.save_results(results)
        
        # Flush Langfuse events
        if evaluator.langfuse:
            try:
                evaluator.langfuse.flush()
                print("\n🚀 All data sent to Langfuse dashboard!")
            except Exception as e:
                print(f"⚠️  Failed to flush Langfuse: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()