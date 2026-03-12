"""
Configuration management for RAG Evaluation Project
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class NESGENConfig(BaseModel):
    """NESGEN (Nestle GenAI) Configuration"""
    client_id: str = Field(default_factory=lambda: os.getenv("NESGEN_CLIENT_ID", ""))
    client_secret: str = Field(default_factory=lambda: os.getenv("NESGEN_CLIENT_SECRET", ""))
    api_base: str = Field(default_factory=lambda: os.getenv("NESGEN_API_BASE", "https://eur-sdr-int-pub.nestle.com/api/dv-exp-sandbox-openai-api/1/genai/Azure"))
    model: str = Field(default_factory=lambda: os.getenv("NESGEN_MODEL", "gpt-4.1"))
    api_version: str = Field(default_factory=lambda: os.getenv("NESGEN_API_VERSION", "2024-02-01"))
    temperature: float = 0.0
    embedding_model: str = Field(default_factory=lambda: os.getenv("NESGEN_EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_api_base: str = Field(default_factory=lambda: os.getenv("NESGEN_EMBEDDING_API_BASE", "https://eur-sdr-int-pub.nestle.com/api/dv-exp-sandbox-openai-api/1"))
    embedding_api_version: str = Field(default_factory=lambda: os.getenv("NESGEN_EMBEDDING_API_VERSION", "2024-10-21"))
    # Fallback OpenAI key for embeddings if NESGEN doesn't provide them
    openai_api_key_fallback: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


class LangfuseConfig(BaseModel):
    """Langfuse Configuration"""
    secret_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    public_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    host: str = Field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    enabled: bool = True
    
    def is_available(self) -> bool:
        """Check if Langfuse credentials are properly configured"""
        return self.enabled and bool(self.secret_key) and bool(self.public_key)


class RAGConfig(BaseModel):
    """RAG Pipeline Configuration"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    retrieval_threshold: float = 0.5


class EvaluationConfig(BaseModel):
    """Evaluation Configuration"""
    # RAGAS metrics
    ragas_metrics: list[str] = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_similarity",
        "answer_correctness",
        "context_relevancy"
    ]
    # DeepEval metrics
    deepeval_metrics: list[str] = [
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "hallucination",
        "bias",
        "toxicity"
    ]
    batch_size: int = 10
    # Framework to use: "ragas", "deepeval", or "both"
    framework: str = "both"


class Config(BaseModel):
    """Main Configuration"""
    nesgen: NESGENConfig = Field(default_factory=NESGENConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# Global config instance
config = Config()
