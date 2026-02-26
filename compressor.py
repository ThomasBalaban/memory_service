# memory_service/compressor.py

import ollama
from typing import List, Dict

from config import (
    OLLAMA_MODEL,
    ANCIENT_COMPRESSION_THRESHOLD,
    ANCIENT_COMPRESSION_CHUNK,
)
from store import MemoryStore


# Map InputSource names to human-readable labels used in the compression prompt.
_SOURCE_LABELS: Dict[str, str] = {
    "MICROPHONE":        "Streamer said",
    "DIRECT_MICROPHONE": "Streamer said",
    "TWITCH_CHAT":       "Chat",
    "TWITCH_MENTION":    "Chat @Nami",
    "VISUAL_CHANGE":     "On screen",
    "AMBIENT_AUDIO":     "Audio",
    "SYSTEM_PATTERN":    "Event",
}


class Compressor:
    """
    Owns all Ollama-based compression logic.

    compress_events  – Turns a batch of raw stream events into one memorable
                       narrative sentence and appends it to the store's
                       narrative_log.

    compress_ancient – Takes the oldest ANCIENT_COMPRESSION_CHUNK entries from
                       narrative_log and rolls them into a single ancient-history
                       summary, triggered automatically when narrative_log reaches
                       ANCIENT_COMPRESSION_THRESHOLD entries.
    """

    # ------------------------------------------------------------------ #
    #  Narrative compression                                               #
    # ------------------------------------------------------------------ #

    async def compress_events(
        self, events: List[Dict[str, str]], store: MemoryStore
    ) -> bool:
        """
        Returns True if a narrative segment was successfully added.
        """
        if not events:
            return False

        context_lines = [
            f"- [{_SOURCE_LABELS.get(e.get('source', ''), 'Event')}] {e.get('text', '')}"
            for e in events
        ]
        context_text = "\n".join(context_lines)

        prompt = (
            "You are extracting memorable moments from a livestream for later callback.\n\n"
            f"Stream events:\n{context_text}\n\n"
            "Extract ONE specific memorable moment that Nami (the AI) could reference later.\n"
            "Format: '[What happened] - [The funny/notable detail]'\n\n"
            "Good examples:\n"
            "- 'Otter rage-quit after spinning a 1 three times in a row'\n"
            "- 'Chat convinced Otter to pick the Space Janitor job'\n"
            "- 'Otter celebrated winning then immediately got hit with 50k taxes'\n\n"
            "Bad examples (too vague):\n"
            "- 'The streamer played a game and had reactions'\n"
            "- 'Various events occurred during gameplay'\n\n"
            "Be specific. Use names/usernames when available. Include the emotion or punchline.\n"
            "If nothing memorable happened, write: [SKIP]\n\n"
            "Memorable moment:"
        )

        try:
            client = ollama.AsyncClient()
            response = await client.generate(model=OLLAMA_MODEL, prompt=prompt)
            narrative = response["response"].strip()

            if "[SKIP]" in narrative.upper() or not narrative:
                print("📖 [Compressor] Nothing memorable – skipping.")
                return False

            # Strip common AI preamble
            for phrase in [
                "here is", "here's", "the memorable moment is",
                "memorable moment:", "one memorable moment",
            ]:
                nl = narrative.lower()
                if nl.startswith(phrase):
                    narrative = narrative[len(phrase):].lstrip(":").strip()
                    break

            # Strip enclosing quotes
            narrative = narrative.strip("\"'")

            # Remove trailing notes
            if "\nNote:" in narrative:
                narrative = narrative.split("\nNote:")[0].strip()
            if narrative.lower().startswith("note:"):
                return False

            if len(narrative) > 10:
                store.add_narrative(narrative)
                print(f"📖 [Compressor] Added narrative: {narrative[:60]}…")
                return True

        except Exception as exc:
            print(f"❌ [Compressor] compress_events error: {exc}")

        return False

    # ------------------------------------------------------------------ #
    #  Ancient compression                                                 #
    # ------------------------------------------------------------------ #

    async def compress_ancient(self, store: MemoryStore) -> bool:
        """
        Rolls the oldest ANCIENT_COMPRESSION_CHUNK narrative entries into one
        ancient-history sentence.  Returns True if an entry was added.
        Only runs when narrative_log has at least ANCIENT_COMPRESSION_THRESHOLD entries.
        """
        if store.narrative_len() < ANCIENT_COMPRESSION_THRESHOLD:
            return False

        chunk = store.pop_narrative_chunk(ANCIENT_COMPRESSION_CHUNK)
        if not chunk:
            return False

        print(f"📚 [Compressor] Archiving {len(chunk)} narrative segments → ancient…")
        context_text = "\n".join(f"- {seg}" for seg in chunk)

        prompt = (
            "These are memorable moments from earlier in a livestream:\n"
            f"{context_text}\n\n"
            "Combine these into ONE sentence that captures the key callbacks.\n"
            "Focus on: names, specific events, running jokes, or notable fails/wins.\n"
            "Example: 'Earlier, Otter lost 3 games in a row, chat roasted him, and he blamed the RNG.'\n\n"
            "Combined summary:"
        )

        try:
            client = ollama.AsyncClient()
            response = await client.generate(model=OLLAMA_MODEL, prompt=prompt)
            summary = response["response"].strip()

            if summary.lower().startswith("combined summary:"):
                summary = summary[17:].strip()
            summary = summary.strip("\"'")

            if len(summary) > 15:
                store.add_ancient(summary)
                print(f"📜 [Compressor] Ancient archived: {summary[:60]}…")
                return True

        except Exception as exc:
            print(f"❌ [Compressor] compress_ancient error: {exc}")

        return False