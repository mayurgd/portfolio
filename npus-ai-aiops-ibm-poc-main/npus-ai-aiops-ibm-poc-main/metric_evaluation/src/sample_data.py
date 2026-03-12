"""
Sample data for RAG evaluation

Also exposes ``load_hf_eval_data()`` and ``load_hf_corpus()`` — thin wrappers
around the generic ``src.hf_dataset.load_hf_dataset`` loader — so any script
that already imports from ``sample_data`` can switch to a HuggingFace dataset
(e.g. vibrantlabsai/fiqa) with a single call.
"""

from typing import Any

from src.hf_dataset import load_hf_dataset

# Sample documents about AI and Machine Learning
SAMPLE_DOCUMENTS = [
    """
    Machine Learning is a subset of artificial intelligence that focuses on building systems 
    that can learn from and make decisions based on data. Unlike traditional programming where 
    explicit instructions are provided, machine learning algorithms identify patterns in data 
    and use those patterns to make predictions or decisions without being explicitly programmed 
    to do so. The three main types of machine learning are supervised learning, unsupervised 
    learning, and reinforcement learning.
    """,
    
    """
    Deep Learning is a specialized subset of machine learning that uses artificial neural 
    networks with multiple layers (deep neural networks) to progressively extract higher-level 
    features from raw input. For example, in image processing, lower layers may identify edges, 
    while higher layers may identify concepts relevant to a human such as digits, letters, or 
    faces. Deep learning has achieved remarkable success in areas like computer vision, natural 
    language processing, and speech recognition.
    """,
    
    """
    Natural Language Processing (NLP) is a branch of artificial intelligence that helps 
    computers understand, interpret, and manipulate human language. NLP draws from many 
    disciplines, including computer science and computational linguistics, to bridge the gap 
    between human communication and computer understanding. Applications of NLP include 
    machine translation, sentiment analysis, chatbots, and text summarization.
    """,
    
    """
    Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of 
    retrieval-based and generation-based approaches. In RAG, a system first retrieves relevant 
    documents from a knowledge base using a retrieval mechanism, then uses a language model to 
    generate responses based on both the query and the retrieved documents. This approach helps 
    reduce hallucinations and provides more factual, grounded responses compared to pure 
    generation methods.
    """,
    
    """
    Large Language Models (LLMs) are neural networks trained on vast amounts of text data to 
    understand and generate human-like text. Models like GPT-4, Claude, and others have billions 
    of parameters and can perform a wide range of language tasks including question answering, 
    summarization, translation, and code generation. However, LLMs can sometimes generate 
    incorrect or misleading information, a phenomenon known as hallucination, which is why 
    techniques like RAG are important for grounding their responses in factual data.
    """,
    
    """
    Transfer Learning is a machine learning technique where a model developed for one task is 
    reused as the starting point for a model on a second task. This is particularly useful in 
    deep learning where training models from scratch can be computationally expensive and 
    time-consuming. Pre-trained models can be fine-tuned on specific datasets, allowing for 
    faster development and often better performance, especially when the target dataset is small.
    """,
    
    """
    Prompt Engineering is the practice of designing and optimizing prompts (input text) to 
    elicit desired responses from language models. Effective prompts can significantly improve 
    the quality, accuracy, and relevance of model outputs. Techniques include zero-shot prompting, 
    few-shot prompting with examples, chain-of-thought prompting for complex reasoning, and 
    role-based prompting where the model is instructed to take on a specific persona or expertise.
    """,
    
    """
    Photosynthesis is the process by which green plants and some other organisms use sunlight 
    to synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally 
    involves the green pigment chlorophyll and generates oxygen as a by-product. This process 
    is fundamental to life on Earth as it is the primary source of organic matter for nearly 
    all organisms, and it also releases the oxygen required for aerobic respiration in most 
    living things.
    """,
    
    """
    The Renaissance was a period in European history, spanning roughly from the 14th to the 
    17th century, marking the transition from the Middle Ages to modernity. It began in Italy 
    in the 14th century and later spread to the rest of Europe. The Renaissance was characterized 
    by a revival of interest in classical learning and values of ancient Greece and Rome. 
    Notable figures include Leonardo da Vinci, Michelangelo, and Raphael in art, as well as 
    Galileo and Copernicus in science.
    """,
    
    """
    Quantum mechanics is a fundamental theory in physics that describes the physical properties 
    of nature at the scale of atoms and subatomic particles. Unlike classical physics, quantum 
    mechanics introduces concepts like wave-particle duality, uncertainty principle, and quantum 
    entanglement. It forms the basis for understanding phenomena that cannot be explained by 
    classical physics, such as the behavior of semiconductors, lasers, and superconductors.
    """,
    
    """
    The water cycle, also known as the hydrological cycle, describes the continuous movement 
    of water on, above, and below the surface of the Earth. It involves processes such as 
    evaporation, condensation, precipitation, infiltration, runoff, and subsurface flow. The 
    water cycle is driven by solar energy and plays a crucial role in distributing heat around 
    the planet, supporting life, and maintaining Earth's climate system.
    """,
    
    """
    Shakespeare's plays are divided into three main categories: tragedies, comedies, and 
    histories. His tragedies, such as Hamlet, Macbeth, and Romeo and Juliet, explore themes 
    of human nature, ambition, and fate. His comedies, including A Midsummer Night's Dream 
    and Much Ado About Nothing, often feature mistaken identities, wordplay, and happy endings. 
    Shakespeare's influence on English literature and language is immeasurable, with many phrases 
    and words he coined still in common use today.
    """,
    
    """
    Climate change refers to long-term shifts in temperatures and weather patterns. While 
    climate change can occur naturally, human activities have been the main driver since the 
    1800s, primarily due to burning fossil fuels like coal, oil, and gas. This produces 
    greenhouse gases that trap heat in Earth's atmosphere. The effects include rising 
    temperatures, melting ice caps, rising sea levels, and more extreme weather events like 
    hurricanes, droughts, and floods.
    """,
    
    """
    The human digestive system is a complex series of organs and glands that processes food. 
    It breaks down food into nutrients, which the body uses for energy, growth, and cell repair. 
    The digestive system includes the mouth, esophagus, stomach, small intestine, large intestine, 
    rectum, and anus. Accessory organs such as the liver, pancreas, and gallbladder also play 
    important roles by producing enzymes and bile to aid in digestion.
    """,
    
    """
    Jazz is a music genre that originated in the African-American communities of New Orleans 
    in the late 19th and early 20th centuries. It is characterized by swing and blue notes, 
    complex chords, call and response vocals, polyrhythms, and improvisation. Jazz has evolved 
    through various styles including bebop, cool jazz, free jazz, and fusion. Influential jazz 
    musicians include Louis Armstrong, Duke Ellington, Miles Davis, and John Coltrane.
    """,
    
    """
    The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends from 
    the Arctic Ocean in the north to the Southern Ocean in the south and is bounded by Asia 
    and Australia in the west and the Americas in the east. The Pacific Ocean contains about 
    25,000 islands, more than the total number in the rest of the world's oceans combined. 
    It covers approximately 165 million square kilometers, which is about 46% of the world's 
    water surface.
    """,
    
    """
    Democracy is a form of government in which power is vested in the people, who exercise 
    this power directly or through freely elected representatives. The key principles of 
    democracy include political equality, freedom of speech and assembly, rule of law, and 
    protection of individual rights. Democratic systems can take various forms, including 
    direct democracy, representative democracy, and constitutional democracy. The concept 
    of democracy has evolved significantly since ancient Athens.
    """,
]

# Sample evaluation questions with ground truth answers
SAMPLE_EVAL_DATA = [
    {
        "question": "What is machine learning and how does it differ from traditional programming?",
        "ground_truth": "Machine learning is a subset of AI that focuses on building systems that learn from data and make decisions based on patterns, unlike traditional programming which requires explicit instructions."
    },
    {
        "question": "What are the main types of machine learning?",
        "ground_truth": "The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning."
    },
    {
        "question": "How does deep learning differ from regular machine learning?",
        "ground_truth": "Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input, achieving remarkable success in computer vision, NLP, and speech recognition."
    },
    {
        "question": "What is RAG and why is it useful?",
        "ground_truth": "RAG (Retrieval-Augmented Generation) is an AI framework that combines retrieval and generation by first retrieving relevant documents then using them to generate responses, which helps reduce hallucinations and provides more factual answers."
    },
    {
        "question": "What is the main advantage of transfer learning?",
        "ground_truth": "Transfer learning allows reusing a model trained for one task as a starting point for another task, which is faster and often performs better than training from scratch, especially with small datasets."
    },
    {
        "question": "What are some applications of Natural Language Processing?",
        "ground_truth": "Applications of NLP include machine translation, sentiment analysis, chatbots, and text summarization."
    },
    {
        "question": "What is prompt engineering?",
        "ground_truth": "Prompt engineering is the practice of designing and optimizing input prompts to elicit desired responses from language models, using techniques like zero-shot, few-shot, and chain-of-thought prompting."
    },
    {
        "question": "What is a hallucination in the context of LLMs?",
        "ground_truth": "Hallucination is when large language models generate incorrect or misleading information, which is why grounding techniques like RAG are important."
    },
]

# Additional test questions without ground truth
TEST_QUESTIONS = [
    "Explain the relationship between AI, machine learning, and deep learning.",
    "How can RAG help improve the accuracy of LLM responses?",
    "What role does transfer learning play in modern NLP applications?",
    "Compare and contrast supervised and unsupervised learning.",
    "Why is prompt engineering important for working with LLMs?",
]


# ---------------------------------------------------------------------------
# HuggingFace dataset helpers
# ---------------------------------------------------------------------------

def load_hf_eval_data(
    dataset_name: str,
    column_mapping: dict[str, str],
    config_name: str | None = None,
    split: str = "test",
    max_samples: int | None = None,
) -> dict[str, list[Any]]:
    """Load any HuggingFace dataset as evaluation-ready data.

    Returns a dict with the keys present in *column_mapping*:
    ``questions``, ``answers``, ``contexts``, ``ground_truths``.

    Example – FiQA pre-built evaluation split (answers + contexts included):

        data = load_hf_eval_data(
            dataset_name="vibrantlabsai/fiqa",
            config_name="ragas_eval_v3",
            split="baseline",
            column_mapping={
                "questions":     "user_input",
                "answers":       "response",
                "contexts":      "retrieved_contexts",
                "ground_truths": "reference",
            },
        )

    Example – FiQA questions-only (run through your own RAG pipeline):

        data = load_hf_eval_data(
            dataset_name="vibrantlabsai/fiqa",
            config_name="main",
            split="test",
            column_mapping={
                "questions":     "question",
                "ground_truths": "ground_truths",
            },
            max_samples=20,
        )
    """
    return load_hf_dataset(
        dataset_name=dataset_name,
        column_mapping=column_mapping,
        split=split,
        config_name=config_name,
        max_samples=max_samples,
    )


def load_hf_corpus(
    dataset_name: str,
    doc_column: str,
    config_name: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
) -> list[str]:
    """Load a HuggingFace dataset split as a plain list of document strings.

    The returned list can be passed directly to ``RAGPipeline.ingest_documents()``.

    Example – FiQA financial corpus:

        docs = load_hf_corpus(
            dataset_name="vibrantlabsai/fiqa",
            config_name="corpus",
            split="corpus",
            doc_column="doc",
            max_samples=500,
        )
        rag.ingest_documents(docs)
    """
    result = load_hf_dataset(
        dataset_name=dataset_name,
        column_mapping={"documents": doc_column},
        split=split,
        config_name=config_name,
        max_samples=max_samples,
    )
    return result.get("documents", [])
