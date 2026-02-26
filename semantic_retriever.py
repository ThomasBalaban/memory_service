# memory_service/semantic_retriever.py

from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Tuple, Dict, Any

from models import MemoryRecord
from config import EMBEDDING_MODEL


class SemanticRetriever:
    """
    Wraps sentence-transformers to produce semantic similarity scores for
    MemoryRecord objects against a free-text query.

    Embeddings are cached by memory ID to avoid recomputing on every call.
    The cache is pruned to remove IDs that no longer exist in the store.
    """

    def __init__(self):
        print(f"🧠 [Retriever] Loading model: {EMBEDDING_MODEL} (CPU)…")
        # Force CPU to avoid Metal/MPS command-buffer conflicts on macOS.
        self.model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        self._cache: Dict[str, Any] = {}
        print("✅ [Retriever] Ready.")

    def rank(
        self,
        query: str,
        memories: List[MemoryRecord],
    ) -> List[Tuple[float, MemoryRecord]]:
        """
        Returns [(similarity_score, record), …] sorted by similarity descending.
        Empty list if query or memories are empty.
        """
        if not memories or not query.strip():
            return []

        # Prune stale cache entries
        live_ids = {m.id for m in memories}
        self._cache = {k: v for k, v in self._cache.items() if k in live_ids}

        # Build / retrieve embeddings
        embeddings = []
        valid: List[MemoryRecord] = []

        for mem in memories:
            content = mem.memory_text or mem.text
            if not content:
                continue
            if mem.id not in self._cache:
                self._cache[mem.id] = self.model.encode(
                    content, convert_to_tensor=True
                )
            embeddings.append(self._cache[mem.id])
            valid.append(mem)

        if not valid:
            return []

        query_emb = self.model.encode(query, convert_to_tensor=True)
        corpus = torch.stack(embeddings)
        scores = util.cos_sim(query_emb, corpus)[0]

        return [(float(scores[i]), valid[i]) for i in range(len(valid))]