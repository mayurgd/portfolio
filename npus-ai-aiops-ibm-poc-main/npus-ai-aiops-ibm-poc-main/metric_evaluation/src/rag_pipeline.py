"""
RAG Pipeline with retrieval and generation components
"""
from typing import List, Dict, Any, Optional
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from config import config
from src.nesgen_llm import NESGENChatModel, NESGENEmbeddings

# Conditional Langfuse import - only if configured
try:
    from langfuse import Langfuse, observe
    _LANGFUSE_AVAILABLE = config.langfuse.is_available()
except ImportError:
    _LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None


def conditional_observe(*args, **kwargs):
    """
    Decorator that conditionally applies Langfuse @observe based on configuration.
    If Langfuse is not configured, returns the function unchanged.
    """
    def decorator(func):
        if _LANGFUSE_AVAILABLE and observe is not None:
            return observe(*args, **kwargs)(func)
        return func
    return decorator


class RAGPipeline:
    """
    Complete RAG Pipeline with retrieval and generation
    """
    
    def __init__(self, collection_name: str = "rag_documents"):
        """
        Initialize RAG Pipeline with NESGEN
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.provider = "nesgen"
        
        # Initialize NESGEN provider
        print("🔵 Initializing NESGEN provider...")
        self.embeddings = NESGENEmbeddings()
        self.llm = NESGENChatModel()
        print(f"✓ Using NESGEN model: {config.nesgen.model}")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.chroma_client.create_collection(collection_name)
        
        # Initialize Langfuse with proper credential check
        if config.langfuse.is_available():
            try:
                self.langfuse = Langfuse(
                    secret_key=config.langfuse.secret_key,
                    public_key=config.langfuse.public_key,
                    host=config.langfuse.host
                )
                print("✓ Langfuse initialized successfully")
            except Exception as e:
                print(f"⚠ Langfuse initialization failed: {e}")
                self.langfuse = None
        else:
            print("⚠ Langfuse not available - credentials not configured")
            print("  Results will be saved to local files instead")
            self.langfuse = None
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap
        )
        
        # RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Answer the user's question based on the provided context.
            
Context:
{context}

Rules:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Be concise and accurate
4. Cite specific parts of the context when possible"""),
            ("human", "{question}")
        ])
    
    @conditional_observe(as_type="span")
    def ingest_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Ingest documents into the vector store
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts for each document
        """
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        for idx, doc in enumerate(documents):
            # Split into chunks
            chunks = self.text_splitter.split_text(doc)
            
            # Generate embeddings
            chunk_embeddings = self.embeddings.embed_documents(chunks)
            
            # Prepare metadata
            doc_metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            
            for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
                chunk_meta = {**doc_metadata, "chunk_id": chunk_idx, "doc_id": idx}
                all_metadatas.append(chunk_meta)
                all_ids.append(f"doc_{idx}_chunk_{chunk_idx}")
        
        # Add to ChromaDB
        self.collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        return len(all_chunks)
    
    @conditional_observe(as_type="span")
    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant document chunks
        """
        if top_k is None:
            top_k = config.rag.top_k
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Retrieve from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        contexts = results['documents'][0] if results['documents'] else []
        
        return contexts
    
    @conditional_observe(as_type="generation")
    def generate(self, query: str, contexts: List[str]) -> str:
        """
        Generate answer based on query and retrieved contexts
        
        Args:
            query: User query
            contexts: Retrieved context documents
            
        Returns:
            Generated answer
        """
        # Format context
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        # Create prompt
        messages = self.prompt_template.format_messages(
            context=context_text,
            question=query
        )
        
        # Generate response
        response = self.llm.invoke(messages)
        answer = response.content
        
        return answer
    
    @conditional_observe(name="rag_query")
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve and generate
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and contexts
        """
        # Retrieve contexts
        contexts = self.retrieve(question, top_k)
        
        # Generate answer
        answer = self.generate(question, contexts)
        
        result = {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }
        
        return result
