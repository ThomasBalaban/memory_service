# memory_service/store.py

import threading
from typing import List, Optional, Dict, Any

from models import MemoryRecord


class MemoryStore:
    """
    Thread-safe, in-process storage for all persistent memory data.

    Three buckets:
      memories        – promoted EventItem snapshots (semantic recall targets)
      narrative_log   – compressed stream moments ("Earlier this stream…")
      ancient_history – further-compressed old narrative chunks ("Way earlier…")
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.memories: List[MemoryRecord] = []
        self.narrative_log: List[str] = []
        self.ancient_history_log: List[str] = []

    # ------------------------------------------------------------------ #
    #  Memories                                                            #
    # ------------------------------------------------------------------ #

    def add_memory(self, record: MemoryRecord) -> bool:
        """Add a memory if it doesn't already exist. Returns True if added."""
        with self._lock:
            if any(m.id == record.id for m in self.memories):
                return False
            self.memories.append(record)
            print(f"💾 [MemoryStore] Stored: {(record.memory_text or record.text)[:60]}...")
            return True

    def get_all(self) -> List[MemoryRecord]:
        with self._lock:
            return list(self.memories)

    def decay(self, decay_amount: float):
        """Reduce interestingness of every memory by decay_amount (floored at 0.1)."""
        with self._lock:
            for mem in self.memories:
                factor = 0.5 if mem.interestingness > 0.9 else 1.0
                mem.interestingness = max(0.1, mem.interestingness - decay_amount * factor)

    # ------------------------------------------------------------------ #
    #  Narrative log                                                       #
    # ------------------------------------------------------------------ #

    def add_narrative(self, text: str):
        with self._lock:
            self.narrative_log.append(text)
            print(f"📖 [MemoryStore] Narrative ({len(self.narrative_log)}): {text[:60]}...")

    def get_narrative(self) -> List[str]:
        with self._lock:
            return list(self.narrative_log)

    def narrative_len(self) -> int:
        with self._lock:
            return len(self.narrative_log)

    def pop_narrative_chunk(self, count: int) -> List[str]:
        """Remove and return the oldest `count` entries for ancient compression."""
        with self._lock:
            if len(self.narrative_log) < count:
                return []
            chunk = self.narrative_log[:count]
            self.narrative_log = self.narrative_log[count:]
            return chunk

    # ------------------------------------------------------------------ #
    #  Ancient history                                                     #
    # ------------------------------------------------------------------ #

    def add_ancient(self, text: str):
        with self._lock:
            self.ancient_history_log.append(text)
            print(f"📜 [MemoryStore] Ancient ({len(self.ancient_history_log)}): {text[:60]}...")

    def get_ancient(self) -> List[str]:
        with self._lock:
            return list(self.ancient_history_log)

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            top = sorted(self.memories, key=lambda m: m.interestingness, reverse=True)[:5]
            return {
                "total_memories": len(self.memories),
                "narrative_count": len(self.narrative_log),
                "ancient_count": len(self.ancient_history_log),
                "top_memories": [
                    {
                        "text": (m.memory_text or m.text)[:80],
                        "score": round(m.interestingness, 3),
                        "source": m.source,
                    }
                    for m in top
                ],
            }