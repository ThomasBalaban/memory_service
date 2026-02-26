# memory_service/config.py

MEMORY_SERVICE_HOST = "0.0.0.0"
MEMORY_SERVICE_PORT = 8009
MEMORY_SERVICE_URL  = "http://localhost:8009"   # used by director client

OLLAMA_MODEL = 'llama3.2:latest'
OLLAMA_HOST  = 'http://localhost:11434'

MEMORY_DECAY_RATE = 0.05          # per minute of decay
DECAY_INTERVAL_SECONDS = 60.0     # minimum gap between decay runs

COMPRESSION_INTERVAL = 30.0       # seconds between narrative compressions
ANCIENT_COMPRESSION_THRESHOLD = 10  # archive to ancient when narrative reaches this length
ANCIENT_COMPRESSION_CHUNK = 5       # how many narrative entries to compress at once

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'