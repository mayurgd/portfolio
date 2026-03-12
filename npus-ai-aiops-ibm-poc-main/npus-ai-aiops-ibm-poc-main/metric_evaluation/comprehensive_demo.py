"""
Comprehensive RAG Evaluation Pipeline Demo

This single demo showcases ALL features of the framework:
1. RAG Pipeline with NESGEN Integration
2. Document Ingestion & Retrieval
3. Answer Generation
4. RAGAS Evaluation (6 metrics)
5. DeepEval Evaluation (14+ metrics including safety)
6. Performance Comparison (different configurations)
7. Quality Analysis & Recommendations
8. Framework Comparison (RAGAS vs DeepEval)
9. Langfuse Integration & Tracking
10. Result Persistence & Storage

Note: For Production-Grade Async Evaluation, see batch_evaluation.py
"""

import asyncio
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timezone

# Import all components
from src.rag_pipeline import RAGPipeline
from src.evaluation_metrics import RAGEvaluator
from src.deepeval_metrics import DeepEvalEvaluator
from src.sample_data import load_hf_eval_data, load_hf_corpus
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
    Comprehensive demonstration of all framework features
    """
    
    # Configure logging
    configure_logging(level="INFO", log_file="./logs/comprehensive_demo.log")
    
    print("\n" + "🌟"*40)
    print("  COMPREHENSIVE RAG EVALUATION PIPELINE DEMO")
    print("  Showcasing ALL Framework Features")
    print("🌟"*40 + "\n")
    
    # =========================================================================
    # PART 1: RAG PIPELINE SETUP
    # =========================================================================
    print_section("PART 1: RAG PIPELINE SETUP", "🔧")
    
    print("Initializing RAG Pipeline with NESGEN...")
    rag = RAGPipeline(collection_name="comprehensive_demo")
    print(f"✓ Pipeline initialized with model: {config.nesgen.model}")
    
    fiqa_corpus = load_hf_corpus(
        dataset_name="vibrantlabsai/fiqa",
        config_name="corpus",
        split="corpus",
        doc_column="doc",
        max_samples=200,
    )
    print(f"\nIngesting {len(fiqa_corpus)} FiQA financial documents...")
    num_chunks = rag.ingest_documents(fiqa_corpus)
    print(f"✓ Ingested {num_chunks} chunks into vector store")

    fiqa_eval = load_hf_eval_data(
        dataset_name="vibrantlabsai/fiqa",
        config_name="main",
        split="test",
        column_mapping={"questions": "question", "ground_truths": "ground_truths"},
        max_samples=5,
    )
    
    # =========================================================================
    # PART 2: RETRIEVAL & GENERATION DEMO
    # =========================================================================
    print_section("PART 2: RETRIEVAL & GENERATION DEMO", "🔍")
    
    # Run sample queries to collect data for evaluation
    print("Running RAG queries on sample questions...")
    eval_data = []
    
    for i, (question, ground_truth) in enumerate(
        zip(fiqa_eval["questions"], fiqa_eval["ground_truths"]), 1
    ):
        
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
    # PART 3: RAGAS EVALUATION (6 METRICS)
    # =========================================================================
    print_section("PART 3: RAGAS EVALUATION", "📊")
    
    print("Initializing RAGAS Evaluator...")
    ragas_evaluator = RAGEvaluator()
    
    ragas_metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_similarity",
        "answer_correctness"
    ]
    
    print(f"Evaluating with {len(ragas_metrics)} RAGAS metrics...")
    print("(This may take 1-2 minutes...)")
    
    ragas_scores = ragas_evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        metrics=ragas_metrics
    )
    
    print_subsection("RAGAS Results")
    print(ragas_evaluator.get_metrics_summary(ragas_scores))
    
    # =========================================================================
    # PART 4: DEEPEVAL EVALUATION (SAFETY + CORE METRICS)
    # =========================================================================
    print_section("PART 4: DEEPEVAL EVALUATION (with Safety Metrics)", "🛡️")
    
    print("Initializing DeepEval Evaluator...")
    deepeval_evaluator = DeepEvalEvaluator()
    
    deepeval_metrics = [
        # Core RAG metrics
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        # Safety metrics
        "hallucination",
        "bias",
        "toxicity"
    ]
    
    print(f"Evaluating with {len(deepeval_metrics)} DeepEval metrics (including safety)...")
    print("(This may take 1-2 minutes...)")
    
    deepeval_scores = deepeval_evaluator.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        metrics=deepeval_metrics
    )
    
    print_subsection("DeepEval Results (with Safety)")
    print(deepeval_evaluator.get_metrics_summary(deepeval_scores))
    
    # =========================================================================
    # PART 5: CONFIGURATION COMPARISON
    # =========================================================================
    print_section("PART 5: CONFIGURATION COMPARISON", "⚖️")
    
    print("Comparing different retrieval configurations (top_k values)...")
    comparison_results = []
    
    for top_k in [3, 5, 7]:
        print(f"\nEvaluating with top_k={top_k}...")
        
        # Run queries with different top_k
        eval_data_topk = []
        for question, ground_truth in zip(
            fiqa_eval["questions"][:3], fiqa_eval["ground_truths"][:3]
        ):
            result = rag.query(question, top_k=top_k)
            eval_data_topk.append({
                'question': question,
                'answer': result['answer'],
                'contexts': result['contexts'],
                'ground_truth': ground_truth
            })
        
        # Extract data
        q = [r['question'] for r in eval_data_topk]
        a = [r['answer'] for r in eval_data_topk]
        c = [r['contexts'] for r in eval_data_topk]
        g = [r['ground_truth'] for r in eval_data_topk]
        
        # Evaluate
        scores = ragas_evaluator.evaluate(
            questions=q,
            answers=a,
            contexts=c,
            ground_truths=g,
            metrics=["faithfulness", "answer_relevancy", "context_precision"]
        )
        
        comparison_results.append({
            'top_k': top_k,
            'faithfulness': scores.get('faithfulness', 0),
            'answer_relevancy': scores.get('answer_relevancy', 0),
            'context_precision': scores.get('context_precision', 0)
        })
    
    print_subsection("Configuration Comparison Results")
    df_comparison = pd.DataFrame(comparison_results)
    print(df_comparison.to_string(index=False))
    
    # Identify best configuration
    best_config = df_comparison.loc[df_comparison['faithfulness'].idxmax()]
    print(f"\n✓ Best configuration: top_k={int(best_config['top_k'])} "
          f"(faithfulness: {best_config['faithfulness']:.4f})")
    
    # =========================================================================
    # PART 6: BATCH EVALUATION WITH DETAILED ANALYSIS
    # =========================================================================
    print_section("PART 6: BATCH EVALUATION & DETAILED ANALYSIS", "📈")
    
    print("Running batch evaluation with per-question breakdown...")
    
    df_detailed = ragas_evaluator.evaluate_batch(
        eval_data,
        metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    )
    
    print_subsection("Per-Question Scores")
    print(df_detailed.to_string(index=False))
    
    print_subsection("Summary Statistics")
    numeric_cols = df_detailed.select_dtypes(include=['float64', 'int64']).columns
    stats = df_detailed[numeric_cols].describe()
    print(stats.to_string())
    
    # Export results
    output_file = "comprehensive_evaluation_results.csv"
    df_detailed.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results exported to: {output_file}")
    
    # =========================================================================
    # PART 7: QUALITY ANALYSIS & RECOMMENDATIONS
    # =========================================================================
    print_section("PART 7: QUALITY ANALYSIS & RECOMMENDATIONS", "🎯")
    
    # Define quality thresholds
    thresholds = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.70
    }
    
    print("Quality Threshold Analysis:")
    print(f"{'Metric':<25} {'Score':<10} {'Threshold':<12} {'Status':<10} {'Gap':<10}")
    print("─" * 80)
    
    recommendations = []
    
    for metric, threshold in thresholds.items():
        if metric in ragas_scores:
            score = ragas_scores[metric]
            status = "✅ PASS" if score >= threshold else "❌ FAIL"
            gap = score - threshold
            gap_str = f"{gap:+.4f}"
            print(f"{metric:<25} {score:<10.4f} {threshold:<12.2f} {status:<10} {gap_str:<10}")
            
            # Generate recommendations
            if gap < 0:
                if metric == "faithfulness":
                    recommendations.append(
                        "⚠️  LOW FAITHFULNESS: Model may be hallucinating\n"
                        "    → Lower temperature setting\n"
                        "    → Improve context grounding in prompt\n"
                        "    → Add explicit instructions to cite sources"
                    )
                elif metric == "answer_relevancy":
                    recommendations.append(
                        "⚠️  LOW ANSWER RELEVANCY: Answers may be off-topic\n"
                        "    → Improve prompt specificity\n"
                        "    → Better retrieval quality\n"
                        "    → Add question understanding step"
                    )
                elif metric == "context_precision":
                    recommendations.append(
                        "⚠️  LOW CONTEXT PRECISION: Too much irrelevant context\n"
                        "    → Adjust retrieval threshold\n"
                        "    → Improve chunking strategy\n"
                        "    → Use reranking"
                    )
                elif metric == "context_recall":
                    recommendations.append(
                        "⚠️  LOW CONTEXT RECALL: Missing relevant information\n"
                        "    → Increase top_k parameter\n"
                        "    → Improve embedding quality\n"
                        "    → Better document coverage"
                    )
    
    print("\n" + "─" * 80)
    print("Recommendations:")
    print("─" * 80)
    
    if recommendations:
        for rec in recommendations:
            print(f"\n{rec}")
    else:
        print("\n✅ All metrics above threshold!")
        print("   System performing well. Consider:")
        print("   → Monitor for performance regression")
        print("   → Set up alerts for quality degradation")
        print("   → Regular evaluation runs (daily/weekly)")
    
    # =========================================================================
    # PART 8: FRAMEWORK COMPARISON (RAGAS VS DEEPEVAL)
    # =========================================================================
    print_section("PART 8: FRAMEWORK COMPARISON (RAGAS vs DeepEval)", "⚖️")
    
    print("Comparing overlapping metrics between frameworks...")
    
    # Compare common metrics
    comparison_data = []
    
    common_metrics = {
        'faithfulness': ('faithfulness', 'faithfulness'),
        'answer_relevancy': ('answer_relevancy', 'answer_relevancy'),
    }
    
    print(f"\n{'Metric':<25} {'RAGAS':<15} {'DeepEval':<15} {'Difference':<15}")
    print("─" * 80)
    
    for display_name, (ragas_key, deepeval_key) in common_metrics.items():
        ragas_val = ragas_scores.get(ragas_key, 0)
        deepeval_val = deepeval_scores.get(deepeval_key, 0)
        diff = ragas_val - deepeval_val
        
        print(f"{display_name:<25} {ragas_val:<15.4f} {deepeval_val:<15.4f} {diff:+.4f}")
        
        comparison_data.append({
            'metric': display_name,
            'ragas': ragas_val,
            'deepeval': deepeval_val,
            'difference': diff
        })
    
    print("\n" + "─" * 80)
    print("Framework Insights:")
    print("─" * 80)
    print("""
RAGAS Strengths:
  ✓ Academically validated metrics
  ✓ Strong research backing
  ✓ Comprehensive retrieval analysis (precision & recall)
  ✓ Seamless LangChain integration

DeepEval Strengths:
  ✓ Faster evaluation performance
  ✓ More metrics (14+ vs 7)
  ✓ Safety metrics (bias, toxicity, hallucination)
  ✓ Better for production use cases
  ✓ Custom metric support with G-Eval

Recommendation:
  → Use RAGAS for research and academic work
  → Use DeepEval for production and safety-critical applications
  → Use BOTH for comprehensive validation
    """)
    
    # =========================================================================
    # PART 10: EVALUATION STATISTICS & MONITORING
    # =========================================================================
    # print_section("PART 10: EVALUATION STATISTICS & MONITORING", "📊")
    
    # stats = production_evaluator.get_statistics()
    
    # print("Production Evaluator Statistics:")
    # print("─" * 80)
    # print(f"Total Evaluations:     {stats['total_evaluations']}")
    # print(f"Successful:            {stats['successful_evaluations']}")
    # print(f"Failed:                {stats['failed_evaluations']}")
    # print(f"Success Rate:          {stats['success_rate']*100:.1f}%")
    
    # if stats['total_evaluations'] > 0:
    #     print(f"\nAverage Duration:      {production_result.duration_seconds:.2f} seconds")
    #     print(f"Items per Second:      {len(questions)/production_result.duration_seconds:.2f}")
    
    # =========================================================================
    # PART 9: LANGFUSE INTEGRATION & OBSERVABILITY
    # =========================================================================
    print_section("PART 9: LANGFUSE INTEGRATION & OBSERVABILITY", "🔍")
    
    if config.langfuse.enabled:
        print("✓ Langfuse Integration Active")
        print(f"  Host: {config.langfuse.host}")
        print()
        print("All operations have been traced and logged to Langfuse:")
        print("  • RAG queries (retrieval + generation)")
        print("  • Evaluation metrics and scores")
        print("  • Performance metrics (latency, tokens)")
        print("  • Custom metadata and tags")
        print()
        print("View your results at:")
        print(f"  → {config.langfuse.host}")
        print()
        print("In Langfuse Dashboard you can:")
        print("  • View full traces of each query")
        print("  • Analyze score distributions")
        print("  • Track performance trends")
        print("  • Monitor costs and token usage")
        print("  • Set up alerts for quality degradation")
    else:
        print("⚠️  Langfuse integration is disabled")
        print("   Enable it in .env to get full observability:")
        print("   → Set LANGFUSE_ENABLED=true")
        print("   → Add LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY")
    
    # =========================================================================
    # PART 10: PERSISTENCE & STORAGE
    # =========================================================================
    print_section("PART 10: PERSISTENCE & STORAGE", "💾")
    
    print("Results Storage:")
    print("─" * 80)
    print(f"✓ Detailed CSV export: {output_file}")
    print(f"✓ Logs: ./logs/comprehensive_demo.log")
    print()
    print("Files can be used for:")
    print("  • Audit trails and compliance")
    print("  • Historical analysis and trends")
    print("  • Sharing with team members")
    print("  • Integration with BI tools")
    print()
    print("For production-grade async evaluation with persistence, see:")
    print("  → batch_evaluation.py")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("COMPREHENSIVE DEMO COMPLETE", "✅")
    
    print("Summary of Demonstrated Features:")
    print("─" * 80)
    print("""
✅ RAG Pipeline with NESGEN Integration
✅ Document Ingestion (chunking, embedding, vector storage)
✅ Semantic Retrieval with ChromaDB
✅ Answer Generation with NESGEN gpt-4.1
✅ RAGAS Evaluation (6 metrics)
✅ DeepEval Evaluation (14+ metrics with safety)
✅ Configuration Comparison (top_k)
✅ Batch Processing & Per-Question Analysis
✅ Quality Analysis & Recommendations
✅ Framework Comparison (RAGAS vs DeepEval)
✅ Langfuse Integration & Tracking
✅ Result Persistence & Storage
✅ CSV Export for Further Analysis

For Production-Grade Async Evaluation, see batch_evaluation.py
    """)
    
    print("─" * 80)
    print("Next Steps:")
    print("─" * 80)
    print("""
1. Review the exported CSV file for detailed per-question analysis
2. Check Langfuse dashboard for traces and analytics
3. Examine logs in ./logs/comprehensive_demo.log
4. Adjust configuration in config.py based on recommendations
5. Integrate this pipeline into your application
6. Set up regular evaluation runs for monitoring
7. Configure alerts for quality degradation in Langfuse
8. For production-grade async evaluation, run batch_evaluation.py
    """)
    
    print("─" * 80)
    print("Performance Summary:")
    print("─" * 80)
    print(f"Total Questions Evaluated: {len(questions)}")
    print(f"Total Metrics Computed:    {len(ragas_metrics) + len(deepeval_metrics)}")
    
    print("\n" + "🌟"*40)
    print("  Thank you for using the RAG Evaluation Framework!")
    print("🌟"*40 + "\n")


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())
