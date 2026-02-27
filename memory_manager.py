# memory_service/memory_manager.py
import time
from typing import List, Dict, Any
from semantic_memory import SemanticMemoryRetriever

MEMORY_DECAY_RATE = 0.05  # Adjust as needed

class MemoryManager:
    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.semantic_retriever = SemanticMemoryRetriever()
        self.last_decay_time = time.time()

    def add_memory(self, memory_data: Dict[str, Any]):
        """Expects dict with: id, timestamp, text, memory_text, importance"""
        # Prevent duplicates
        if not any(m['id'] == memory_data['id'] for m in self.memories):
            # Ensure it has an importance score
            if 'importance' not in memory_data:
                memory_data['importance'] = 0.5 
            self.memories.append(memory_data)
            print(f"💾 [Storage] Saved memory: {memory_data.get('memory_text', memory_data.get('text'))[:40]}...")

    def decay_memories(self):
        now = time.time()
        minutes_passed = (now - self.last_decay_time) / 60.0
        
        if minutes_passed < 1.0: 
            return 
        
        decay_amount = minutes_passed * MEMORY_DECAY_RATE
        self.last_decay_time = now
        decayed_count = 0
        
        for mem in self.memories:
            old_score = mem['importance']
            factor = 0.5 if old_score > 0.9 else 1.0
            mem['importance'] = max(0.1, old_score - (decay_amount * factor))
            if mem['importance'] < old_score:
                decayed_count += 1
                
        if decayed_count > 0:
            print(f"🧠 [Memory] Decayed {decayed_count} memories.")

    def retrieve(self, query_context: str, limit: int = 5) -> List[Dict[str, Any]]:
        self.decay_memories()

        if not self.memories:
            return []
            
        if not query_context or len(query_context.strip()) < 5:
            # Fall back to importance-based retrieval
            return sorted(self.memories, key=lambda m: m['importance'], reverse=True)[:limit]

        # 1. Get Semantic Scores
        semantic_results = self.semantic_retriever.rank_memories(query_context, self.memories)
        semantic_map = {mem['id']: score for score, mem in semantic_results}
        
        scored_memories = []
        now = time.time()
        
        for mem in self.memories:
            sem_score = semantic_map.get(mem['id'], 0.0)
            imp_score = mem['importance']
            
            # Recency Score
            age_hours = (now - mem['timestamp']) / 3600.0
            recency_score = max(0.0, 1.0 - (age_hours / 24.0))
            
            # Hybrid Formula: Meaning > Importance > Time
            final_score = (sem_score * 0.6) + (imp_score * 0.3) + (recency_score * 0.1)
            
            if sem_score > 0.8:
                final_score += 0.2
            
            # Keyword boost
            query_lower = query_context.lower()
            mem_text_lower = (mem.get('memory_text') or mem.get('text', '')).lower()
            query_words = set(w for w in query_lower.split() if len(w) > 4)
            mem_words = set(w for w in mem_text_lower.split() if len(w) > 4)
            
            if query_words & mem_words:
                final_score += min(0.15, len(query_words & mem_words) * 0.05)
                
            scored_memories.append((final_score, mem))
            
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in scored_memories[:limit]]