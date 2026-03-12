"""
Production-Grade Async Batch Evaluation

This module provides production-ready batch evaluation capabilities with:
- Async evaluation with concurrency control
- Rate limiting (100 API calls/minute)
- Retry logic (3 attempts with exponential backoff)
- Result persistence to disk
- Progress tracking
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime, timezone

# Import components
from src.rag_pipeline import RAGPipeline
from src.evaluation import ProductionEvaluator
from src.sample_data import SAMPLE_DOCUMENTS, SAMPLE_EVAL_DATA
from src.utils import configure_logging
from config import config


def print_section(title: str, emoji: str = "📋"):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"{emoji} {title}")
    print("="*80 + "\n")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print('─'*80)


async def main():
    """
    Production-grade async batch evaluation demonstration
    """
    
    # Configure logging
    configure_logging(level="INFO", log_file="./logs/batch_evaluation.log")
    
    print("\n" + "⚡"*40)
    print("  PRODUCTION-GRADE ASYNC BATCH EVALUATION")
    print("⚡"*40 + "\n")
    
    # =========================================================================
    # PART 1: RAG PIPELINE SETUP & DATA COLLECTION
    # =========================================================================
    print_section("PART 1: RAG PIPELINE SETUP & DATA COLLECTION", "🔧")
    
    print("Initializing RAG Pipeline with NESGEN...")
    rag = RAGPipeline(collection_name="batch_evaluation")
    print(f"✓ Pipeline initialized with model: {config.nesgen.model}")
    
    print(f"\nIngesting {len(SAMPLE_DOCUMENTS)} documents...")
    num_chunks = rag.ingest_documents(SAMPLE_DOCUMENTS)
    print(f"✓ Ingested {num_chunks} chunks into vector store")
    
    # Run sample queries to collect data for evaluation
    print("\nRunning RAG queries on sample questions...")
    eval_data = []
    
    for i, item in enumerate(SAMPLE_EVAL_DATA[:5], 1):
        question = item['question']
        ground_truth = item['ground_truth']
        
        print(f"\n[Query {i}] {question}")
        
        # Execute RAG query
        result = rag.query(question, top_k=5)
        
        print(f"  Answer: {result['answer'][:150]}...")
        print(f"  Retrieved: {len(result['contexts'])} contexts")
        
        # Collect for evaluation
        eval_data.append({
            'question': question,
            'answer': result['answer'],
            'contexts': result['contexts'],
            'ground_truth': ground_truth
        })
    
    print(f"\n✓ Completed {len(eval_data)} queries")
    
    # Extract data for evaluation
    questions = [r['question'] for r in eval_data]
    answers = [r['answer'] for r in eval_data]
    contexts = [r['contexts'] for r in eval_data]
    ground_truths = [r['ground_truth'] for r in eval_data]
    
    # =========================================================================
    # PART 2: PRODUCTION-GRADE ASYNC EVALUATION
    # =========================================================================
    print_section("PART 2: PRODUCTION-GRADE ASYNC EVALUATION", "⚡")
    
    print("Initializing Production Evaluator with enterprise features:")
    print("  • Async evaluation with concurrency control")
    print("  • Rate limiting (100 API calls/minute)")
    print("  • Retry logic (3 attempts with exponential backoff)")
    print("  • Result persistence to disk")
    print("  • Progress tracking")
    print()
    
    production_evaluator = ProductionEvaluator(
        framework="deepeval",
        max_concurrent=5,
        rate_limit_per_minute=100,
        enable_retry=True,
        max_retries=3,
        enable_persistence=True,
        storage_path="./evaluation_results",
        log_level="INFO"
    )
    
    # Define progress callback
    progress_updates = []
    def progress_callback(completed: int, total: int):
        percentage = (completed / total) * 100
        progress_updates.append((completed, total))
        print(f"  Progress: {completed}/{total} ({percentage:.1f}%)")
    
    print("Running async evaluation with progress tracking...")
    
    production_result = await production_evaluator.evaluate_async(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        metrics=["faithfulness", "answer_relevancy", "contextual_precision"],
        metadata={
            "environment": "production",
            "dataset": "ai_ml_sample",
            "model": config.nesgen.model,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        progress_callback=progress_callback
    )
    
    print_subsection("Production Evaluation Results")
    print(f"Request ID: {production_result.request_id}")
    print(f"Status: {production_result.status.value}")
    print(f"Duration: {production_result.duration_seconds:.2f} seconds")
    print(f"Items Evaluated: {len(questions)}")
    print()
    print("Scores:")
    for metric, score in sorted(production_result.scores.items()):
        if isinstance(score, (int, float)):
            print(f"  {metric:30s}: {score:.4f}")
    
    # =========================================================================
    # PART 3: EVALUATION STATISTICS & MONITORING
    # =========================================================================
    print_section("PART 3: EVALUATION STATISTICS & MONITORING", "📊")
    
    stats = production_evaluator.get_statistics()
    
    print("Production Evaluator Statistics:")
    print("─" * 80)
    print(f"Total Evaluations:     {stats['total_evaluations']}")
    print(f"Successful:            {stats['successful_evaluations']}")
    print(f"Failed:                {stats['failed_evaluations']}")
    print(f"Success Rate:          {stats['success_rate']*100:.1f}%")
    
    if stats['total_evaluations'] > 0:
        print(f"\nAverage Duration:      {production_result.duration_seconds:.2f} seconds")
        print(f"Items per Second:      {len(questions)/production_result.duration_seconds:.2f}")
    
    # =========================================================================
    # PART 4: PERSISTENCE & STORAGE
    # =========================================================================
    print_section("PART 4: PERSISTENCE & STORAGE", "💾")
    
    print("Results Storage:")
    print("─" * 80)
    print(f"✓ Production evaluation results: ./evaluation_results/")
    print(f"  Request ID: {production_result.request_id}")
    print(f"✓ Logs: ./logs/batch_evaluation.log")
    print()
    print("Files can be used for:")
    print("  • Audit trails and compliance")
    print("  • Historical analysis and trends")
    print("  • Sharing with team members")
    print("  • Integration with BI tools")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("BATCH EVALUATION COMPLETE", "✅")
    
    print("Summary:")
    print("─" * 80)
    print(f"""
✅ Async evaluation with concurrency control
✅ Rate limiting enabled (100 API calls/minute)
✅ Retry logic with exponential backoff
✅ Result persistence to disk
✅ Progress tracking and monitoring
✅ Production-ready evaluation statistics

Performance:
  • Questions Evaluated:   {len(questions)}
  • Total Duration:        {production_result.duration_seconds:.2f} seconds
  • Throughput:            {len(questions)/production_result.duration_seconds:.2f} items/second
  • Success Rate:          {stats['success_rate']*100:.1f}%
    """)
    
    print("─" * 80)
    print("Next Steps:")
    print("─" * 80)
    print("""
1. Review persisted results in ./evaluation_results/
2. Check logs in ./logs/batch_evaluation.log
3. Integrate this into your CI/CD pipeline
4. Set up regular batch evaluation runs
5. Configure alerts for quality degradation
6. Scale up with larger datasets
    """)
    
    print("\n" + "⚡"*40)
    print("  Production Batch Evaluation Complete!")
    print("⚡"*40 + "\n")


if __name__ == "__main__":
    # Run the batch evaluation
    asyncio.run(main())
