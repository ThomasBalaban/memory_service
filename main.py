# memory_service/main.py

import time
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import (
    MEMORY_SERVICE_HOST,
    MEMORY_SERVICE_PORT,
    MEMORY_DECAY_RATE,
    DECAY_INTERVAL_SECONDS,
)
from models import (
    AddMemoryRequest,
    MemoryRecord,
    RetrieveRequest,
    CompressRequest,
    NarrativeAddRequest,
)
from store import MemoryStore
from semantic_retriever import SemanticRetriever
from compressor import Compressor


# --------------------------------------------------------------------------- #
#  Singletons                                                                  #
# --------------------------------------------------------------------------- #

store     = MemoryStore()
retriever = SemanticRetriever()          # loads sentence-transformer on init
compressor = Compressor()

_last_decay: float = time.time()


# --------------------------------------------------------------------------- #
#  App                                                                         #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"✅ Memory Service ready on port {MEMORY_SERVICE_PORT}")
    yield
    print("🛑 Memory Service shutting down.")


app = FastAPI(title="Nami Memory Service", lifespan=lifespan)


# --------------------------------------------------------------------------- #
#  Memories                                                                    #
# --------------------------------------------------------------------------- #

@app.post("/memories/add")
async def add_memory(req: AddMemoryRequest):
    """Persist a promoted EventItem snapshot."""
    record = MemoryRecord(**req.model_dump())
    added = store.add_memory(record)
    return {"status": "ok", "added": added}


@app.post("/memories/retrieve")
async def retrieve_memories(req: RetrieveRequest):
    """
    Hybrid-scored retrieval:
      final = (semantic * 0.6) + (importance * 0.3) + (recency * 0.1)
    Falls back to importance-only when query is empty/short.
    """
    memories = store.get_all()

    if not req.query or len(req.query.strip()) < 5:
        sorted_mems = sorted(memories, key=lambda m: m.interestingness, reverse=True)
        return {"memories": [m.model_dump() for m in sorted_mems[: req.limit]]}

    ranked = retriever.rank(req.query, memories)
    now    = time.time()
    scored = []

    for sem_score, mem in ranked:
        age_hours = (now - mem.timestamp) / 3600.0
        recency   = max(0.0, 1.0 - age_hours / 24.0)
        final     = (sem_score * 0.6) + (mem.interestingness * 0.3) + (recency * 0.1)

        # Boost strong semantic matches
        if sem_score > 0.8:
            final += 0.2

        # Keyword overlap boost
        q_words   = {w for w in req.query.lower().split() if len(w) > 4}
        mem_words = {w for w in (mem.memory_text or mem.text).lower().split() if len(w) > 4}
        overlap   = q_words & mem_words
        if overlap:
            final += min(0.15, len(overlap) * 0.05)

        scored.append((final, mem))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Debug top matches
    print(f"🧠 [Retrieve] Top results for '{req.query[:40]}…':")
    for score, mem in scored[:3]:
        print(f"   • {score:.3f}: {(mem.memory_text or mem.text)[:50]}…")

    return {"memories": [m.model_dump() for _, m in scored[: req.limit]]}


@app.post("/memories/decay")
async def decay_memories():
    """Decay all memory scores proportional to time elapsed since last decay."""
    global _last_decay
    now     = time.time()
    minutes = (now - _last_decay) / 60.0

    if minutes < DECAY_INTERVAL_SECONDS / 60.0:
        return {"status": "skipped", "reason": "too_soon", "minutes_since_last": round(minutes, 2)}

    decay_amount = minutes * MEMORY_DECAY_RATE
    store.decay(decay_amount)
    _last_decay = now
    print(f"🧠 [Decay] Applied {decay_amount:.4f} to all memories.")
    return {"status": "ok", "decay_amount": round(decay_amount, 4), "minutes_elapsed": round(minutes, 2)}


# --------------------------------------------------------------------------- #
#  Narrative & Ancient history                                                 #
# --------------------------------------------------------------------------- #

@app.post("/narrative/add")
async def add_narrative(req: NarrativeAddRequest):
    """Directly add a pre-formed narrative segment (used for manual pushes)."""
    store.add_narrative(req.text)
    return {"status": "ok", "narrative_count": store.narrative_len()}


@app.get("/narrative")
async def get_narrative():
    return {"narrative": store.get_narrative()}


@app.get("/ancient")
async def get_ancient():
    return {"ancient": store.get_ancient()}


# --------------------------------------------------------------------------- #
#  Compression                                                                 #
# --------------------------------------------------------------------------- #

@app.post("/compress")
async def compress(req: CompressRequest):
    """
    1. Compress the provided event batch into a narrative segment (Ollama).
    2. Auto-archive to ancient history if narrative_log is long enough.
    Returns the updated narrative and ancient lists so the director can sync
    its local caches in a single round-trip.
    """
    narrative_added = await compressor.compress_events(req.events, store)
    ancient_added   = await compressor.compress_ancient(store)

    return {
        "status":          "ok",
        "narrative_added": narrative_added,
        "ancient_added":   ancient_added,
        "narrative":       store.get_narrative(),
        "ancient":         store.get_ancient(),
    }


# --------------------------------------------------------------------------- #
#  Stats / Health                                                              #
# --------------------------------------------------------------------------- #

@app.get("/stats")
async def stats():
    return store.get_stats()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "memory_service", "port": 8009}


# --------------------------------------------------------------------------- #
#  Entry-point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run(
        "memory_service.main:app",
        host=MEMORY_SERVICE_HOST,
        port=MEMORY_SERVICE_PORT,
        log_level="warning",
    )