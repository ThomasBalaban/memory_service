# memory_service/semantic_memory.py
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Tuple, Dict, Any

class SemanticMemoryRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"🧠 [Semantic] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.embedding_cache = {}  # {memory_id: embedding_tensor}
        print(f"✅ [Semantic] Model loaded (CPU Enforced).")

    def _get_embedding(self, text: str):
        return self.model.encode(text, convert_to_tensor=True)

    def rank_memories(self, query_text: str, memories: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
        """Ranks a list of memory dicts based on semantic similarity to the query."""
        if not memories or not query_text:
            return []
            
        # Clean cache of deleted memories
        current_ids = {m['id'] for m in memories}
        self.embedding_cache = {k: v for k, v in self.embedding_cache.items() if k in current_ids}
        
        memory_embeddings = []
        valid_memories = []
        
        for mem in memories:
            content = mem.get('memory_text') or mem.get('text', '')
            if not content: 
                continue
                
            mem_id = mem['id']
            if mem_id not in self.embedding_cache:
                self.embedding_cache[mem_id] = self._get_embedding(content)
            
            memory_embeddings.append(self.embedding_cache[mem_id])
            valid_memories.append(mem)
            
        if not valid_memories:
            return []

        query_embedding = self._get_embedding(query_text)
        
        # Calculate Cosine Similarities
        corpus_embeddings = torch.stack(memory_embeddings)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        results = []
        for idx, score in enumerate(cos_scores):
            results.append((float(score), valid_memories[idx]))
            
        return results