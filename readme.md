# Nami Memory Service

This microservice handles semantic embedding, storage, decay, and retrieval of long-term memories. It connects to the Nami Central Hub (port 8002) via Socket.IO.

## Architecture
1. **Director Engine** promotes an event to a memory and emits a `save_memory` event to the Hub.
2. **Memory Service** receives `save_memory`, generates a semantic embedding, and stores it in memory.
3. When formulating a prompt, **Director Engine** emits a `query_memories` event containing the current context.
4. **Memory Service** receives `query_memories`, runs a hybrid similarity search (Semantic + Recency + Importance), and emits a `memory_results` event back.
5. **Director Engine** catches `memory_results` and injects them into the Gemini prompt.

---
