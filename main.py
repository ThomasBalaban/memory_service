# memory_service/main.py
import asyncio
import socketio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from memory_manager import MemoryManager

HUB_URL = 'http://localhost:8002'

# Use AsyncClient so it plays nice with FastAPI's event loop
sio = socketio.AsyncClient()
manager = MemoryManager()

async def connect_to_hub():
    """Background task to keep the Hub connection alive."""
    while True:
        if not sio.connected:
            try:
                print(f"🔌 [Memory] Connecting to Hub at {HUB_URL}...")
                await sio.connect(HUB_URL)
            except Exception:
                await asyncio.sleep(5)
        await asyncio.sleep(2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the socket connection when Uvicorn boots the server
    loop = asyncio.get_running_loop()
    task = loop.create_task(connect_to_hub())
    yield
    # Clean up on shutdown
    if sio.connected:
        await sio.disconnect()
    task.cancel()

# This is the "app" that Uvicorn is looking for!
app = FastAPI(title="Nami Memory Service", lifespan=lifespan)

@app.get("/health")
async def health():
    """Endpoint the Launcher polls to see if we are online."""
    return {"status": "ok", "service": "memory_service", "hub_connected": sio.connected}

@sio.event
async def connect():
    print("✅ [Hub] Connected to Central Hub")

@sio.event
async def disconnect():
    print("❌ [Hub] Disconnected from Central Hub")

@sio.on("save_memory")
async def on_save_memory(data):
    manager.add_memory(data)

@sio.on("query_memories")
async def on_query_memories(data):
    query = data.get("query", "")
    limit = data.get("limit", 5)
    request_id = data.get("request_id", "unknown")
    
    print(f"🔍 [Query] Searching for: '{query[:30]}...'")
    results = manager.retrieve(query, limit)
    
    await sio.emit("memory_results", {
        "request_id": request_id,
        "memories": results
    })
    print(f"📤 [Query] Sent {len(results)} results back to Hub.")