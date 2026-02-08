"""Local embedding generation using sentence-transformers.

Provides semantic search capability without external API calls.
Uses lightweight models that run efficiently on CPU.
"""

import logging
import time
import warnings
from typing import Any

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Lazy load to avoid import cost when not using embeddings
_model = None
_model_name = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2", verbose: bool = False):
    """Get or initialize the sentence-transformers model.
    
    Uses lazy loading to avoid startup cost. The model is cached
    after first load.
    
    Args:
        model_name: Name of the sentence-transformers model to use.
                   Default is 'all-MiniLM-L6-v2' (384 dimensions, fast).
        verbose: Whether to show loading status messages.
                   
    Returns:
        SentenceTransformer model instance
    """
    global _model, _model_name
    
    if _model is not None and _model_name == model_name:
        return _model
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Temporarily suppress noisy library loggers
        noisy_loggers = [
            "sentence_transformers",
            "transformers",
            "transformers.modeling_utils", 
            "torch.distributed",
        ]
        old_levels = {}
        for name in noisy_loggers:
            lg = logging.getLogger(name)
            old_levels[name] = lg.level
            lg.setLevel(logging.ERROR)
        
        # Show spinner during model load
        start_time = time.perf_counter()
        with console.status(f"[cyan]Loading embedding model[/cyan] [bold]{model_name}[/bold]"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*layers were not sharded.*")
                warnings.filterwarnings("ignore", category=FutureWarning)
                _model = SentenceTransformer(model_name)
                _model_name = model_name
        
        # Restore log levels
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)
        
        elapsed = time.perf_counter() - start_time
        dim = _model.get_sentence_embedding_dimension()
        if verbose:
            console.print(
                f"[green]✓[/green] [bold]Embedding model loaded:[/bold] {model_name} "
                f"[dim]({dim}d, {elapsed:.1f}s)[/dim]"
            )
        logger.debug(f"Loaded embedding model {model_name} ({dim}d) in {elapsed:.1f}s")
        
        return _model
        
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Install with: "
            "uv pip install 'llm-bawt[memory]'"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        return None


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2", verbose: bool = False) -> list[float] | None:
    """Generate an embedding vector for the given text.
    
    Args:
        text: Text to embed
        model_name: Sentence-transformers model name
        verbose: Whether to show loading status messages
        
    Returns:
        List of floats representing the embedding, or None on failure
    """
    if not text or text.isspace():
        return None
    
    model = get_embedding_model(model_name, verbose=verbose)
    if model is None:
        return None
    
    try:
        # Generate embedding (returns numpy array, disable progress bar)
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None


def generate_embeddings_batch(texts: list[str], model_name: str = "all-MiniLM-L6-v2", verbose: bool = False) -> list[list[float] | None]:
    """Generate embeddings for multiple texts efficiently.
    
    Args:
        texts: List of texts to embed
        model_name: Sentence-transformers model name
        verbose: Whether to show loading status messages
        
    Returns:
        List of embeddings (or None for failed items)
    """
    if not texts:
        return []
    
    model = get_embedding_model(model_name, verbose=verbose)
    if model is None:
        return [None] * len(texts)
    
    try:
        # Filter out empty texts, keeping track of indices
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and not text.isspace():
                valid_indices.append(i)
                valid_texts.append(text)
        
        if not valid_texts:
            return [None] * len(texts)
        
        # Batch encode for efficiency (disable progress bar)
        embeddings = model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Reconstruct result list with None for invalid entries
        result = [None] * len(texts)
        for i, emb in zip(valid_indices, embeddings):
            result[i] = emb.tolist()
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        return [None] * len(texts)


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2", verbose: bool = False) -> int:
    """Get the embedding dimension for the specified model.
    
    Args:
        model_name: Sentence-transformers model name
        verbose: Whether to show loading status messages
        
    Returns:
        Embedding dimension (default 384 if model can't be loaded)
    """
    model = get_embedding_model(model_name, verbose=verbose)
    if model is None:
        return 384  # Default for MiniLM
    
    return model.get_sentence_embedding_dimension()
