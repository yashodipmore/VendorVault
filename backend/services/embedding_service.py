"""
Embedding Service - Text to Vector Conversion
Uses lightweight mock embeddings for demo (no heavy ML dependencies)
"""
import numpy as np
from typing import List
import time
import hashlib

# Configuration
EMBEDDING_DIM = 384  # Standard dimension for sentence-transformers


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a deterministic embedding for text.
    Uses hash-based approach for demo purposes.
    In production, this would use sentence-transformers.
    """
    # Create deterministic embedding from text hash
    text_bytes = text.encode('utf-8')
    
    # Use multiple hash iterations to fill 384 dimensions
    # Each SHA256 hash = 32 bytes = 8 floats (4 bytes each)
    # Need 384/8 = 48 iterations
    embeddings = []
    iteration = 0
    while len(embeddings) < EMBEDDING_DIM:
        hash_input = text_bytes + str(iteration).encode()
        hash_bytes = hashlib.sha256(hash_input).digest()
        # Convert bytes to floats (4 bytes per float)
        for j in range(0, len(hash_bytes), 4):
            if len(embeddings) < EMBEDDING_DIM:
                value = int.from_bytes(hash_bytes[j:j+4], 'little', signed=True)
                embeddings.append(value / (2**31))  # Normalize to [-1, 1]
        iteration += 1
    
    embedding = np.array(embeddings[:EMBEDDING_DIM], dtype=np.float32)
    
    # Normalize to unit length (important for cosine similarity)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def generate_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """
    Generate embeddings for multiple texts (batch processing)
    """
    return [generate_embedding(text) for text in texts]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better search
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c]  # Remove empty chunks


class EmbeddingService:
    """Service class for embedding operations"""
    
    def __init__(self):
        self.embedding_count = 0
        self.total_time = 0.0
    
    def embed_text(self, text: str) -> dict:
        """Embed text and return with metrics"""
        start = time.time()
        embedding = generate_embedding(text)
        elapsed = (time.time() - start) * 1000
        
        self.embedding_count += 1
        self.total_time += elapsed
        
        return {
            'embedding': embedding,
            'dimension': len(embedding),
            'processing_time_ms': round(elapsed, 2)
        }
    
    def embed_document(self, text: str, chunk_size: int = 500) -> List[dict]:
        """
        Embed a full document by chunking it first
        """
        chunks = chunk_text(text, chunk_size)
        results = []
        
        for i, chunk in enumerate(chunks):
            result = self.embed_text(chunk)
            result['chunk_id'] = i
            result['chunk_text'] = chunk[:200] + '...' if len(chunk) > 200 else chunk
            results.append(result)
        
        return results
    
    def get_stats(self) -> dict:
        return {
            'total_embeddings': self.embedding_count,
            'avg_time_ms': round(self.total_time / max(self.embedding_count, 1), 2),
            'model': 'hash-based-mock (demo)',
            'dimension': EMBEDDING_DIM
        }


# Global instance
embedding_service = EmbeddingService()
