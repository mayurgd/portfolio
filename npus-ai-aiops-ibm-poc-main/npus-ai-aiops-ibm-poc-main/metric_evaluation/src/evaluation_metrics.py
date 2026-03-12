"""
RAG Evaluation Metrics using RAGAS framework
"""
from typing import List, Dict, Any, Optional
import os
import json
import pandas as pd
from datetime import datetime
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from langfuse import Langfuse
from config import config
from src.nesgen_llm import NESGENChatModel, NESGENEmbeddings, NESGENRagasEmbeddings


class RAGEvaluator:
    """
    Comprehensive RAG Evaluation using RAGAS metrics
    
    Optimized for stability with slower APIs:
    - Reduced concurrency (max_workers=2) to prevent API overload
    - Increased timeouts (180s per request, 300s total) for slower responses
    - Enhanced error handling for individual metric failures
    
    Supported metrics:
    - faithfulness: Measures factual accuracy
    - answer_relevancy: Measures answer relevance to question
    - context_precision: Measures context relevance
    - context_recall: Measures context completeness
    - answer_correctness: Measures semantic + factual correctness
    - answer_similarity: Measures semantic similarity
    """
    
    def __init__(self, llm=None, embeddings=None):
        """
        Initialize RAG Evaluator
        
        Args:
            llm: Optional LLM for metrics that require it (uses NESGEN or OpenAI if not provided)
            embeddings: Optional embeddings model (uses NESGEN or OpenAI if not provided)
        """
        # Initialize Langfuse with proper credential check
        if config.langfuse.is_available():
            try:
                self.langfuse = Langfuse(
                    secret_key=config.langfuse.secret_key,
                    public_key=config.langfuse.public_key,
                    host=config.langfuse.host
                )
                print("✓ Langfuse initialized for RAGAS")
                self.use_file_fallback = False
            except Exception as e:
                print(f"⚠ Langfuse initialization failed: {e}")
                print("  Results will be saved to local files instead")
                self.langfuse = None
                self.use_file_fallback = True
        else:
            print("⚠ Langfuse not available - credentials not configured")
            print("  Results will be saved to local files instead")
            self.langfuse = None
            self.use_file_fallback = True
        
        # Create output directory for file fallback
        if self.use_file_fallback:
            self.output_dir = "evaluation_results"
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Use provided LLM or create a default one
        if llm is None:
            # Try NESGEN first, fall back to OpenAI
            try:
                if config.nesgen.client_id and config.nesgen.client_secret:
                    print("Using NESGEN LLM for evaluation...")
                    # RAGAS 0.2+ supports LangChain models directly
                    self.llm = NESGENChatModel(
                        temperature=0.0  # Use deterministic outputs for evaluation
                    )
                elif os.getenv("OPENAI_API_KEY"):
                    print("Using OpenAI LLM for evaluation...")
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0.0,
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                else:
                    print("Warning: No LLM credentials found. LLM functionality will be limited.")
                    self.llm = None
            except Exception as e:
                print(f"Warning: Could not initialize default LLM: {e}")
                import traceback
                traceback.print_exc()
                self.llm = None
        else:
            self.llm = llm
        
        # Use provided embeddings or create default ones
        if embeddings is None:
            try:
                if config.nesgen.client_id and config.nesgen.client_secret and config.nesgen.embedding_api_base:
                    print("Using NESGEN embeddings for evaluation...")
                    self.embeddings = NESGENRagasEmbeddings()
                elif os.getenv("OPENAI_API_KEY"):
                    print("Using OpenAI embeddings for evaluation...")
                    from langchain_openai import OpenAIEmbeddings
                    self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
                else:
                    print("Warning: No embeddings credentials found. Embeddings functionality will be limited.")
                    self.embeddings = None
            except Exception as e:
                print(f"Warning: Could not initialize default embeddings: {e}")
                import traceback
                traceback.print_exc()
                self.embeddings = None
        else:
            self.embeddings = embeddings
        
        # Define metric instances
        self.metrics_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity
        }
    
    @property
    def available_metrics(self):
        """Get list of available metric names"""
        return list(self.metrics_map.keys())
    
    def _get_metric_instances(self, metric_names: List[str]) -> List:
        """
        Get metric instances for the given metric names
        
        Args:
            metric_names: List of metric names
            
        Returns:
            List of metric objects
        """
        if not self.llm:
            raise ValueError("LLM is required for metrics but none was provided or could be initialized")
            
        instances = []
        # Metrics that need embeddings in addition to LLM
        embeddings_metrics = {"answer_relevancy", "context_precision", "answer_correctness", "answer_similarity"}
        
        print(f"⚙️ Configuring metrics with reduced concurrency for API stability...")
        
        for name in metric_names:
            if name in self.metrics_map:
                try:
                    # Check if metric needs embeddings
                    if name in embeddings_metrics:
                        if self.embeddings:
                            instances.append(self.metrics_map[name])
                        else:
                            print(f"⚠️ Metric '{name}' requires embeddings but none provided. Skipping.")
                    else:
                        # Metric only needs LLM
                        instances.append(self.metrics_map[name])
                except Exception as e:
                    print(f"⚠️ Could not get metric '{name}': {e}")
        
        if not instances:
            raise ValueError("No valid metrics could be initialized")
        
        print(f"✓ Initialized {len(instances)} metrics: {[m.name for m in instances]}")
        return instances
    
    def prepare_evaluation_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dataset:
        """
        Prepare dataset in RAGAS format
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists (each question has multiple contexts)
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dataset object for RAGAS evaluation
        """
        # Ensure all data is in the correct format
        # Convert to plain Python types to avoid Pydantic issues
        data = {
            "question": [str(q) for q in questions],
            "answer": [str(a) for a in answers],
            "contexts": [[str(c) for c in ctx_list] for ctx_list in contexts]
        }
        
        if ground_truths:
            data["ground_truth"] = [str(gt) for gt in ground_truths]
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system with specified metrics
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts for each question
            ground_truths: Optional ground truth answers
            metrics: List of metric names to evaluate (default: all)
            
        Returns:
            Dictionary with evaluation results
        """
        # Prepare dataset
        dataset = self.prepare_evaluation_dataset(
            questions, answers, contexts, ground_truths
        )
        
        # Select metrics
        if metrics is None:
            # Use a subset of metrics from config that are actually available
            config_metrics = getattr(config.evaluation, 'ragas_metrics', [])
            metrics = [m for m in config_metrics if m in self.metrics_map]
            if not metrics:
                # Fallback to all available metrics
                metrics = list(self.metrics_map.keys())
        
        selected_metrics = self._get_metric_instances(metrics)
        
        if not selected_metrics:
            raise ValueError("No valid metrics selected for evaluation")
        
        # Run evaluation
        print(f"Evaluating with metrics: {metrics}")
        
        # Configure RAGAS executor with increased timeout and reduced concurrency
        # This prevents timeout errors with slower APIs like NESGEN
        run_config = RunConfig(
            max_workers=2,  # Reduced concurrency to prevent API overload
            timeout=180,    # Increased timeout for slower responses (3 minutes)
            max_wait=300,   # Maximum wait time (5 minutes)
        )
        
        try:
            # Pass llm and embeddings to evaluate - ragas will use defaults if None
            results = evaluate(
                dataset, 
                metrics=selected_metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=run_config
            )
            
            # Convert to dict
            results_dict = results.to_pandas().to_dict('records')[0] if hasattr(results, 'to_pandas') else dict(results)
            
        except TimeoutError as e:
            print(f"\n⚠️ Evaluation timeout occurred. Consider:")
            print(f"   1. Reducing the number of test samples")
            print(f"   2. Using fewer metrics at once")
            print(f"   3. Increasing API timeout further")
            raise TimeoutError(f"RAGAS evaluation timed out: {e}") from e
        except Exception as e:
            print(f"\n❌ Evaluation failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Log to Langfuse or save to file
        if self.langfuse:
            self._log_to_langfuse(results_dict, questions, answers, contexts)
        elif self.use_file_fallback:
            self._save_to_file(results_dict, questions, answers, contexts)
        
        return results_dict
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair
        
        Args:
            question: Question text
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            metrics: List of metric names to evaluate
            
        Returns:
            Dictionary with metric scores
        """
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
            metrics=metrics
        )
    
    def evaluate_batch(
        self,
        eval_data: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate a batch of questions with detailed results
        
        Args:
            eval_data: List of dicts with 'question', 'answer', 'contexts', 'ground_truth'
            metrics: List of metric names to evaluate
            
        Returns:
            DataFrame with results for each question
        """
        results = []
        
        for item in eval_data:
            result = self.evaluate_single(
                question=item['question'],
                answer=item['answer'],
                contexts=item['contexts'],
                ground_truth=item.get('ground_truth'),
                metrics=metrics
            )
            
            # Add original data
            result['question'] = item['question']
            result['answer'] = item['answer']
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _log_to_langfuse(
        self,
        results: Dict[str, float],
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]]
    ):
        """Log evaluation results to Langfuse"""
        try:
            # Create a trace for evaluation
            trace = self.langfuse.trace(
                name="rag_evaluation",
                metadata={
                    "num_questions": len(questions),
                    "metrics": list(results.keys())
                }
            )
            
            # Log overall scores
            for metric_name, score in results.items():
                if isinstance(score, (int, float)):
                    trace.score(
                        name=metric_name,
                        value=score,
                        comment=f"RAG evaluation metric: {metric_name}"
                    )
            
            # Log individual examples
            for i, (q, a, c) in enumerate(zip(questions[:5], answers[:5], contexts[:5])):
                trace.span(
                    name=f"example_{i}",
                    input={"question": q, "contexts": c},
                    output={"answer": a}
                )
            
            print("✓ Results logged to Langfuse")
        except Exception as e:
            print(f"Warning: Could not log to Langfuse: {e}")
            if self.use_file_fallback:
                self._save_to_file(results, questions, answers, contexts)
    
    def _save_to_file(
        self,
        results: Dict[str, float],
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]]
    ):
        """Save evaluation results to a JSON file as fallback"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ragas_results_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            output_data = {
                "framework": "ragas",
                "timestamp": timestamp,
                "num_questions": len(questions),
                "metrics": results,
                "examples": [
                    {
                        "question": q,
                        "answer": a,
                        "contexts": c
                    }
                    for q, a, c in zip(questions[:5], answers[:5], contexts[:5])
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ RAGAS results saved to: {filepath}")
            
            # Also update the consolidated file
            self._update_consolidated_file(output_data)
            
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
    
    def _update_consolidated_file(self, new_result: Dict[str, Any]):
        """Update the consolidated RAGAS results file"""
        try:
            consolidated_path = os.path.join(self.output_dir, "ragas_consolidated_results.json")
            
            # Load existing consolidated file if it exists
            if os.path.exists(consolidated_path):
                with open(consolidated_path, 'r', encoding='utf-8') as f:
                    consolidated = json.load(f)
            else:
                # Create new consolidated structure
                consolidated = {
                    "metadata": {
                        "title": "Consolidated RAGAS Evaluation Results",
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat()
                    },
                    "evaluations": []
                }
            
            # Add new result
            consolidated["evaluations"].append(new_result)
            consolidated["metadata"]["last_updated"] = datetime.now().isoformat()
            consolidated["metadata"]["total_evaluations"] = len(consolidated["evaluations"])
            
            # Calculate aggregate statistics
            all_metrics = {}
            total_questions = 0
            
            for eval_result in consolidated["evaluations"]:
                total_questions += eval_result.get("num_questions", 0)
                metrics = eval_result.get("metrics", {})
                for metric_name, score in metrics.items():
                    if isinstance(score, (int, float)):
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(score)
            
            # Add statistics
            consolidated["statistics"] = {
                "total_evaluations": len(consolidated["evaluations"]),
                "total_questions": total_questions,
                "metrics": {}
            }
            
            for metric_name, scores in all_metrics.items():
                if scores:
                    consolidated["statistics"]["metrics"][metric_name] = {
                        "count": len(scores),
                        "mean": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores),
                        "median": sorted(scores)[len(scores) // 2]
                    }
            
            # Save consolidated file
            with open(consolidated_path, 'w', encoding='utf-8') as f:
                json.dump(consolidated, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Updated consolidated results: {consolidated_path}")
            
        except Exception as e:
            print(f"Warning: Could not update consolidated file: {e}")
    
    def get_metrics_summary(self, results: Dict[str, float]) -> str:
        """
        Get a formatted summary of evaluation metrics
        
        Args:
            results: Dictionary of metric scores
            
        Returns:
            Formatted string summary
        """
        summary = "\n" + "="*60 + "\n"
        summary += "RAG EVALUATION RESULTS\n"
        summary += "="*60 + "\n\n"
        
        # Group metrics by category
        quality_metrics = ["faithfulness", "answer_correctness"]
        relevancy_metrics = ["answer_relevancy"]
        retrieval_metrics = ["context_precision", "context_recall"]
        
        def print_metrics(metric_list, title):
            s = f"{title}:\n"
            for metric in metric_list:
                if metric in results:
                    score = results[metric]
                    if isinstance(score, (int, float)):
                        s += f"  • {metric.replace('_', ' ').title()}: {score:.4f}\n"
            return s
        
        summary += print_metrics(quality_metrics, "Quality Metrics")
        summary += "\n" + print_metrics(relevancy_metrics, "Relevancy Metrics")
        summary += "\n" + print_metrics(retrieval_metrics, "Retrieval Metrics")
        
        summary += "\n" + "="*60 + "\n"
        
        return summary
