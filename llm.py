"""
EquipAI — Phase 3: LLM Module
Handles all calls to Mistral via Ollama.
Supports both streaming (for UI) and blocking (for batch use).

Usage:
    from llm import generate, generate_stream

    # Blocking
    answer = generate(prompt)

    # Streaming (for Streamlit st.write_stream)
    for token in generate_stream(prompt):
        print(token, end="", flush=True)
"""

import logging
import requests
from typing import Generator

logger = logging.getLogger("equipai.llm")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
DEFAULT_MODEL = "mistral"
TEMPERATURE   = 0.2     # low = factual, high = creative
MAX_TOKENS    = 1024
TIMEOUT       = 120     # seconds


# ── Health check ──────────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def is_model_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if a specific model is pulled in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False


def get_available_models() -> list[str]:
    """Return list of all models pulled in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ── Blocking generation ───────────────────────────────────────────────────────

def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Send a prompt to Mistral and return the complete response.
    Blocks until generation is done.
    """
    if not is_ollama_running():
        return (
            "⚠️ **Ollama is not running.**\n\n"
            "Open a new terminal and run:\n```\nollama serve\n```\n"
            "Then ask your question again."
        )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": MAX_TOKENS,
                    "stop": ["TECHNICIAN'S QUESTION:", "MANUAL EXCERPTS:"],
                }
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return (
            "⚠️ **Cannot connect to Ollama.**\n\n"
            "Make sure Ollama is running:\n```\nollama serve\n```"
        )
    except requests.exceptions.Timeout:
        return (
            "⚠️ **Mistral timed out.** "
            "Try a shorter question or restart Ollama."
        )
    except Exception as e:
        logger.error("LLM error: %s", e)
        return f"⚠️ Error: {str(e)}"


# ── Streaming generation ──────────────────────────────────────────────────────

def generate_stream(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
) -> Generator[str, None, None]:
    """
    Stream tokens from Mistral one by one.
    Use with Streamlit's st.write_stream() for real-time output.

    Example:
        with st.chat_message("assistant"):
            answer = st.write_stream(generate_stream(prompt))
    """
    if not is_ollama_running():
        yield (
            "⚠️ **Ollama is not running.**\n\n"
            "Open a new terminal and run:\n```\nollama serve\n```"
        )
        return

    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": MAX_TOKENS,
                    "stop": ["TECHNICIAN'S QUESTION:", "MANUAL EXCERPTS:"],
                }
            },
            stream  = True,
            timeout = TIMEOUT,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                import json
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break

    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ Lost connection to Ollama."
    except Exception as e:
        logger.error("Stream error: %s", e)
        yield f"\n\n⚠️ Error: {str(e)}"


# ── Vision call (for Phase 4) ─────────────────────────────────────────────────

def describe_image(
    base64_image: str,
    prompt: str,
    model: str = "llava",
) -> str:
    """
    Send an image to LLaVA for part identification (Phase 4).
    Returns the description as a string.
    """
    if not is_ollama_running():
        return "⚠️ Ollama is not running. Run: ollama serve"

    if not is_model_available(model):
        return (
            f"⚠️ Vision model '{model}' not found.\n"
            f"Run: ollama pull {model}"
        )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 800,
                }
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except Exception as e:
        logger.error("Vision error: %s", e)
        return f"⚠️ Vision error: {str(e)}"


if __name__ == "__main__":
    print("Ollama running:", is_ollama_running())
    print("Models available:", get_available_models())

    if is_ollama_running():
        print("\nTest generation...")
        result = generate("What is 2 + 2? Answer in one word.")
        print("Response:", result)