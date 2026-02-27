# memory_service/main.py
import socketio
import time
from memory_manager import MemoryManager

HUB_URL = 'http://localhost:8002'

sio = socketio.Client()
manager = MemoryManager()

@sio.event
def connect():
    print("✅ [Hub] Connected to Central Hub")

@sio.event
def disconnect():
    print("❌ [Hub] Disconnected from Central Hub")

@sio.on("save_memory")
def on_save_memory(data):
    """
    Expected data format:
    {
      "id": "uuid",
      "timestamp": 170000000.0,
      "text": "Full text of event",
      "memory_text": "Summarized version",
      "importance": 0.85
    }
    """
    manager.add_memory(data)

@sio.on("query_memories")
def on_query_memories(data):
    """
    Expected data format:
    {
      "request_id": "uuid-for-tracking-response",
      "query": "The context text to search against",
      "limit": 5
    }
    """
    query = data.get("query", "")
    limit = data.get("limit", 5)
    request_id = data.get("request_id", "unknown")
    
    print(f"🔍 [Query] Searching for: '{query[:30]}...' (req: {request_id})")
    
    results = manager.retrieve(query, limit)
    
    # Send results back through the hub
    response_payload = {
        "request_id": request_id,
        "memories": results
    }
    sio.emit("memory_results", response_payload)
    print(f"📤 [Query] Sent {len(results)} results back to Hub.")

if __name__ == "__main__":
    print("🚀 Starting Memory Service...")
    
    while True:
        try:
            sio.connect(HUB_URL)
            sio.wait()
        except socketio.exceptions.ConnectionError:
            print("⚠️ Hub not found. Retrying in 5 seconds...")
            time.sleep(5)