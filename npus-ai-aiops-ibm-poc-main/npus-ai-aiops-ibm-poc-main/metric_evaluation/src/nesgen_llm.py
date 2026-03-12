"""
Custom LLM wrapper for NESGEN (Nestle GenAI) API
"""
import json
import requests
from typing import Any, Dict, List, Optional, Iterator, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, BaseModel
from config import config

try:
    from ragas.embeddings import BaseRagasEmbeddings
except ImportError:
    # Fallback if BaseRagasEmbeddings is not available
    BaseRagasEmbeddings = object


class NESGENChatModel(BaseChatModel):
    """
    Custom Chat Model for NESGEN API
    
    Handles authentication via client_id and client_secret headers
    """
    
    client_id: str = Field(default="")
    client_secret: str = Field(default="")
    api_base: str = Field(default="")
    model: str = Field(default="gpt-4.1")
    api_version: str = Field(default="2024-02-01")
    temperature: float = Field(default=0.0)
    max_retries: int = Field(default=3)
    timeout: int = Field(default=180)  # Increased for RAGAS evaluation
    
    def __init__(self, **kwargs):
        """Initialize NESGEN Chat Model"""
        super().__init__(**kwargs)
        if not self.client_id:
            self.client_id = config.nesgen.client_id
        if not self.client_secret:
            self.client_secret = config.nesgen.client_secret
        if not self.api_base:
            self.api_base = config.nesgen.api_base
        if not self.model:
            self.model = config.nesgen.model
        if not self.api_version:
            self.api_version = config.nesgen.api_version
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for LLM type"""
        return "nesgen"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat completion using NESGEN API
        
        Args:
            messages: List of chat messages
            stop: Optional stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments (e.g., response_format for JSON mode)
            
        Returns:
            ChatResult with generated message
        """
        # Convert LangChain messages to NESGEN format
        nesgen_messages = []
        for msg in messages:
            if hasattr(msg, 'type'):
                role = msg.type
                if role == "human":
                    role = "user"
                elif role == "system":
                    role = "system"
                elif role == "ai":
                    role = "assistant"
            else:
                role = "user"
            
            nesgen_messages.append({
                "role": role,
                "content": msg.content
            })
        
        # Prepare request
        url = f"{self.api_base}/{self.model}/chat/completions"
        params = {"api-version": self.api_version}
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        body = {
            "messages": nesgen_messages,
            "temperature": self.temperature,
        }
        
        # Add optional parameters
        if stop:
            body["stop"] = stop
        
        # Add response_format for JSON mode if requested (for RAGAS structured outputs)
        if "response_format" in kwargs:
            body["response_format"] = kwargs["response_format"]
        
        # Make API request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    params=params,
                    headers=headers,
                    json=body,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                # Extract message content
                # Assuming Azure OpenAI-style response format
                if "choices" in result and len(result["choices"]) > 0:
                    message_content = result["choices"][0]["message"]["content"]
                else:
                    raise ValueError(f"Unexpected response format: {result}")
                
                # Create ChatGeneration
                message = AIMessage(content=message_content)
                generation = ChatGeneration(message=message)
                
                return ChatResult(generations=[generation])
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
                else:
                    raise Exception(f"NESGEN API request failed after {self.max_retries} attempts: {str(e)}")
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                raise Exception(f"Failed to parse NESGEN API response: {str(e)}")
        
        raise Exception(f"NESGEN API request failed: {str(last_error)}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation - currently delegates to sync version"""
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters"""
        return {
            "model": self.model,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "temperature": self.temperature,
        }


class NESGENEmbeddings:
    """
    Custom Embeddings for NESGEN API
    
    Note: If NESGEN doesn't provide embeddings endpoint, 
    you'll need to use OpenAI embeddings separately
    """
    
    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        api_base: str = "",
        model: str = "text-embedding-3-small",
        api_version: str = ""
    ):
        """Initialize NESGEN Embeddings"""
        self.client_id = client_id or config.nesgen.client_id
        self.client_secret = client_secret or config.nesgen.client_secret
        self.api_base = api_base or config.nesgen.embedding_api_base
        self.model = model or config.nesgen.embedding_model
        self.api_version = api_version or getattr(config.nesgen, 'embedding_api_version', '2024-10-21')
        
        # If no NESGEN embeddings endpoint, fall back to OpenAI
        if not self.api_base and config.nesgen.openai_api_key_fallback:
            from langchain_openai import OpenAIEmbeddings
            self._fallback = OpenAIEmbeddings(
                model=self.model,
                openai_api_key=config.nesgen.openai_api_key_fallback
            )
            self._use_fallback = True
        else:
            self._use_fallback = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if self._use_fallback:
            return self._fallback.embed_documents(texts)
        
        # Implement NESGEN embeddings API call
        # Format: https://eur-sdr-int-pub.nestle.com/api/dv-exp-sandbox-openai-api/1/openai/deployments/text-embedding-3-small/embeddings
        url = f"{self.api_base}/openai/deployments/{self.model}/embeddings"
        params = {"api-version": self.api_version}
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        body = {
            "input": texts,
            "model": self.model
        }
        
        try:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=body,
                timeout=180  # Increased for RAGAS evaluation
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract embeddings (assuming OpenAI-style format)
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
        except Exception as e:
            raise Exception(f"NESGEN embeddings API request failed: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        if self._use_fallback:
            return self._fallback.embed_query(text)
        
        embeddings = self.embed_documents([text])
        return embeddings[0]


class ChatCompletionMessage(BaseModel):
    """OpenAI-compatible message response"""
    role: str
    content: str
    
    def model_dump_json(self, **kwargs):
        """Serialize to JSON for compatibility"""
        return json.dumps({"role": self.role, "content": self.content})


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible choice response"""
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"
    
    def model_dump_json(self, **kwargs):
        """Serialize to JSON for compatibility"""
        return json.dumps({
            "index": self.index,
            "message": {"role": self.message.role, "content": self.message.content},
            "finish_reason": self.finish_reason
        })


class ChatCompletion(BaseModel):
    """OpenAI-compatible completion response"""
    id: str = "chatcmpl-nesgen"
    object: str = "chat.completion"
    created: int = 0
    model: str = "gpt-4.1"
    choices: List[ChatCompletionChoice]
    
    def model_dump_json(self, **kwargs):
        """Serialize to JSON for compatibility"""
        return json.dumps({
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": c.message.role, "content": c.message.content},
                    "finish_reason": c.finish_reason
                } for c in self.choices
            ]
        })


class NESGENInstructorClient:
    """
    OpenAI-compatible client wrapper for NESGEN API
    This allows NESGEN to work with ragas InstructorLLM
    """
    
    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        api_base: str = "",
        model: str = "gpt-4.1",
        api_version: str = "2024-02-01"
    ):
        self.client_id = client_id or config.nesgen.client_id
        self.client_secret = client_secret or config.nesgen.client_secret
        self.api_base = api_base or config.nesgen.api_base
        self.model = model or config.nesgen.model
        self.api_version = api_version or config.nesgen.api_version
        
        # Create a chat completions object for compatibility
        self.chat = self._ChatCompletions(self)
    
    class _ChatCompletions:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self._Completions(parent)
        
        class _Completions:
            def __init__(self, parent):
                self.parent = parent
            
            def create(
                self,
                model: str,
                messages: List[Dict[str, str]],
                temperature: float = 0.0,
                response_format: Dict = None,
                **kwargs
            ):
                """Create chat completion with JSON mode support for RAGAS"""
                url = f"{self.parent.api_base}/{self.parent.model}/chat/completions"
                params = {"api-version": self.parent.api_version}
                
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "client_id": self.parent.client_id,
                    "client_secret": self.parent.client_secret
                }
                
                body = {
                    "messages": messages,
                    "temperature": temperature,
                }
                
                # Add response_format for JSON mode if requested (for structured outputs)
                if response_format and response_format.get("type") == "json_object":
                    body["response_format"] = response_format
                
                try:
                    response = requests.post(
                        url,
                        params=params,
                        headers=headers,
                        json=body,
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Convert to Pydantic models for RAGAS compatibility
                    if "choices" in result and len(result["choices"]) > 0:
                        # Wrap in Pydantic models that have model_dump_json method
                        choices = []
                        for choice in result["choices"]:
                            msg = ChatCompletionMessage(
                                role=choice["message"]["role"],
                                content=choice["message"]["content"]
                            )
                            completion_choice = ChatCompletionChoice(
                                index=choice.get("index", 0),
                                message=msg,
                                finish_reason=choice.get("finish_reason", "stop")
                            )
                            choices.append(completion_choice)
                        
                        completion = ChatCompletion(
                            id=result.get("id", "chatcmpl-nesgen"),
                            object=result.get("object", "chat.completion"),
                            created=result.get("created", 0),
                            model=result.get("model", model),
                            choices=choices
                        )
                        return completion
                    else:
                        # Fallback to raw dict if structure is unexpected
                        return result
                        
                except Exception as e:
                    raise Exception(f"NESGEN API request failed: {str(e)}")


class NESGENRagasEmbeddings(BaseRagasEmbeddings):
    """
    Modern NESGEN Embeddings compatible with ragas BaseRagasEmbeddings
    """
    
    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        api_base: str = "",
        model: str = "text-embedding-3-small",
        api_version: str = ""
    ):
        super().__init__()
        self.client_id = client_id or config.nesgen.client_id
        self.client_secret = client_secret or config.nesgen.client_secret
        self.api_base = api_base or config.nesgen.embedding_api_base
        self.model = model or config.nesgen.embedding_model
        self.api_version = api_version or getattr(config.nesgen, 'embedding_api_version', '2024-10-21')
        self.cache = None  # For ragas compatibility
    
    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Call NESGEN embeddings API"""
        url = f"{self.api_base}/openai/deployments/{self.model}/embeddings"
        params = {"api-version": self.api_version}
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        body = {"input": texts}
        
        try:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=body,
                timeout=180  # Increased for RAGAS evaluation
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
        except Exception as e:
            raise Exception(f"NESGEN embeddings API request failed: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self._call_embedding_api(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        embeddings = self._call_embedding_api([text])
        return embeddings[0]
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text (alias for embed_query)"""
        return self.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (alias for embed_documents)"""
        return self.embed_documents(texts)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents - delegates to sync version"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query - delegates to sync version"""
        return self.embed_query(text)
    
    def set_run_config(self, config):
        """Set run configuration (for ragas compatibility)"""
        pass
