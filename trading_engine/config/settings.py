"""Central configuration — all values loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------
ALPACA_API_KEY: str = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY: str = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL: str = os.getenv(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
)

# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------
ALPHAVANTAGE_API_KEY: str = os.environ["ALPHAVANTAGE_API_KEY"]

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_URL: str = os.environ["DB_URL"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
