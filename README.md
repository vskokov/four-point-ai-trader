# Four-Point AI Trader

An autonomous stock trading engine for Alpaca paper (and live) trading. The
system ingests real-time market data and news, detects market regimes, computes
ensemble signals, sizes positions with fractional Kelly criterion, enforces
mandatory risk controls, and routes orders — all driven by a local LLM
(Ollama / Gemma) for news sentiment.

```
                          ┌─────────────────────────────────────────┐
                          │            TradingEngine (orchestrator)  │
                          │                                         │
  Alpaca WebSocket ───────┤► bar_handler()                          │
                          │    │                                    │
  APScheduler ────────────┤    ├─► HMMRegimeDetector                │
    sentiment (4 h)        │    ├─► KalmanHedgeRatio + OUSpreadSignal│
    eod_job (16:05 ET)     │    ├─► LLM signal (from signal_log)    │
                          │    └─► MWUMetaAgent.scheduled_update()  │
                          │              │                          │
                          │              └─► OrderExecutor          │
                          │                    RiskManager          │
                          │                    circuit breaker      │
                          └─────────────────────────────────────────┘
```

---

## Features

| Layer | Capability |
|---|---|
| **Data** | TimescaleDB hypertables for OHLCV, signals, regimes, news; Alpaca market data + WebSocket bars; Alpha Vantage news with rate-limit tracking |
| **Regime detection** | 3-state Gaussian HMM (bear / neutral / bull) with deterministic post-hoc state labelling; online partial-fit every 20 bars |
| **Pairs trading** | Kalman-filter adaptive hedge ratio; Ornstein-Uhlenbeck spread signal with z-score thresholds; periodic Engle-Granger cointegration health checks |
| **LLM sentiment** | Local Ollama (Gemma 4 e4b) scores news headlines into directional signals; retries on malformed JSON; 60-second timeout → safe neutral fallback |
| **Meta-agent** | Multiplicative Weights Update (MWU) ensemble conditioned on HMM regime; per-regime weight isolation; online learning from realised price directions |
| **Backtesting** | Walk-forward vectorbt engine; Sharpe, CAGR, max-drawdown metrics; bias checks; CSV + PNG results |
| **Risk management** | Fractional Kelly criterion (¼ Kelly default); per-position cap (10 %); peak-drawdown circuit breaker (15 %); daily-loss circuit breaker (5 %) |
| **Execution** | Alpaca `TradingClient`; market orders (DAY); sell capped at held quantity; emergency `close_all_positions` |
| **Orchestration** | APScheduler (interval sentiment job + cron EOD job); SIGINT / SIGTERM graceful shutdown; atomic state persistence with SHA-256 checksum + 3-backup rotation |

---

## Architecture

### Module map

```
trading_engine/
├── config/
│   ├── settings.py              # All constants loaded from .env
│   └── av_rate_state.json       # Auto-created; Alpha Vantage daily call counter
├── data/
│   ├── storage.py               # TimescaleDB — OHLCV, signals, regimes, news
│   ├── alpaca_client.py         # AlpacaMarketData (bars, quotes, account, stream)
│   └── alphavantage_client.py   # AlphaVantageNewsClient (primary news source)
├── signals/
│   ├── hmm_regime.py            # GaussianHMM regime detector
│   ├── kalman_pairs.py          # Kalman adaptive hedge ratio
│   ├── mean_reversion.py        # CointegrationTest + OUSpreadSignal
│   └── llm_sentiment.py         # Ollama/Gemma news sentiment → ±1/0 signal
├── backtesting/
│   ├── backtest_engine.py       # BacktestEngine (vectorbt) + walk-forward
│   └── results/                 # Auto-created; CSV + PNG outputs
├── meta_agent/
│   └── mwu_agent.py             # MWUMetaAgent — regime-conditioned MWU ensemble
├── execution/
│   └── executor.py              # RiskManager + OrderExecutor
├── orchestrator/
│   ├── engine.py                # TradingEngine — top-level loop
│   └── state_manager.py         # Atomic JSON state persistence + checksum
├── models/                      # Auto-created; HMM .pkl, Kalman .pkl, MWU .npy
├── utils/
│   └── logging.py               # structlog factory (JSON file + console)
├── tests/                       # 327 unit tests — no live connections required
├── main.py                      # CLI entry point
├── requirements.txt
├── docker-compose.yml           # TimescaleDB container
└── .env.example                 # Template — copy to .env
```

### Signal pipeline (per bar)

```
New bar arrives
      │
      ▼
 Storage.insert_ohlcv()
      │
      ▼
 HMMRegimeDetector.partial_fit_online()
 HMMRegimeDetector.predict_regime()  →  {regime: 0/1/2, label, probs}
      │
      ├──► OUSpreadSignal.compute_signal()  →  {signal, z_score, ...}
      │         (for any pair containing this ticker)
      │
      ├──► query signal_log  →  most recent llm_sentiment (12 h window)
      │
      ▼
 signals = {
   "hmm_regime":    {signal: ±1/0, confidence},
   "ou_spread":     {signal: ±1/0, confidence},
   "llm_sentiment": {signal: ±1/0, confidence},
 }
      │
      ▼
 MWUMetaAgent.scheduled_update(ticker, signals, regime)
      │   ├── decide() → ensemble score = Σ w_r[k] * sig[k] * conf[k]
      │   └── update_weights() for any elapsed pending decisions
      │
      ▼
 final_signal ∈ {−1, 0, +1}
      │
      ├── if 0 → no-op
      │
      └── if ±1 →
            RiskManager.check_trade()
              ├── circuit breaker (drawdown / daily loss)
              └── position size limit
            RiskManager.kelly_size(win_rate, avg_win, avg_loss)
            size_usd = equity × kelly_f × confidence
            n_shares = floor(size_usd / mid_price)
            TradingClient.submit_order(MarketOrderRequest, DAY)
```

---

## Requirements

- Python 3.12
- Docker (for TimescaleDB)
- [Ollama](https://ollama.com/) with `gemma4:e4b` pulled
- Alpaca account (paper recommended for testing)
- Alpha Vantage API key (free tier: 25 req/day)

---

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/vskokov/four-point-ai-trader.git
cd four-point-ai-trader/trading_engine

cp .env.example .env
# Edit .env and fill in your credentials
```

### 2. Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ALPACA_API_KEY` | yes | — | Alpaca API key |
| `ALPACA_SECRET_KEY` | yes | — | Alpaca secret key |
| `ALPHAVANTAGE_API_KEY` | yes | — | Alpha Vantage API key |
| `DB_URL` | yes | — | `postgresql+psycopg2://trader:traderpass@localhost:5432/trading` |
| `ALPACA_BASE_URL` | no | `https://paper-api.alpaca.markets` | Use paper endpoint |
| `OLLAMA_HOST` | no | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | no | `gemma4:e4b` | Ollama model tag |
| `LOG_LEVEL` | no | `INFO` | `DEBUG` / `INFO` / `WARNING` |

### 3. Start TimescaleDB

```bash
# From trading_engine/
docker compose up -d
docker compose ps    # wait for "healthy"
```

Schema is auto-created on first connection — no migrations needed.

### 4. Pull the Ollama model

```bash
docker exec ollama ollama pull gemma4:e4b
docker exec ollama ollama list     # verify
```

### 5. Python environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

## Running

### Paper trading (default)

```bash
cd trading_engine
.venv/bin/python -m trading_engine.main \
    --tickers AAPL MSFT JPM BAC \
    --pairs JPM,BAC \
    --log-level INFO
```

### Live trading

```bash
.venv/bin/python -m trading_engine.main \
    --tickers AAPL MSFT \
    --pairs \
    --live \
    --log-file /var/log/trader.json
```

### CLI reference

```
usage: four-point-trader [--tickers TICKER [TICKER ...]]
                         [--pairs T1,T2 [T1,T2 ...]]
                         [--live]
                         [--log-level {DEBUG,INFO,WARNING,ERROR}]
                         [--log-file PATH]

options:
  --tickers   Equity symbols to trade (default: AAPL MSFT JPM BAC)
  --pairs     Cointegrated pairs for OU mean-reversion (e.g. JPM,BAC)
  --live      Connect to Alpaca live trading  [paper is default]
  --log-level Logging verbosity (default: INFO)
  --log-file  Optional path for newline-delimited JSON log file
```

Shutdown cleanly with `Ctrl-C` or `SIGTERM`. If the circuit breaker fires, all
positions are liquidated before exit.

---

## Risk controls

| Control | Default | Description |
|---|---|---|
| Max position per ticker | 10 % of equity | `RiskManager.max_position_pct` |
| Peak drawdown halt | 15 % | Triggers circuit breaker, liquidates all positions |
| Daily loss halt | 5 % | Triggers circuit breaker, liquidates all positions |
| Kelly fraction | ¼ Kelly | `RiskManager.kelly_fraction` |
| Order type | Market / DAY | No limit orders; sells capped at held quantity |

The circuit breaker fires **before** any order on every bar. If triggered it
sets an emergency flag, signals the shutdown event, and `close_all_positions`
is called during the shutdown sequence.

---

## Testing

```bash
cd trading_engine

# Unit tests — 327 tests, no live connections required
.venv/bin/pytest tests/test_alpaca_client.py \
                 tests/test_alphavantage_client.py \
                 tests/test_hmm_regime.py \
                 tests/test_mean_reversion.py \
                 tests/test_llm_sentiment.py \
                 tests/backtesting/test_backtest_engine.py \
                 tests/meta_agent/test_mwu_agent.py \
                 tests/execution/test_executor.py \
                 tests/test_engine.py -v

# Integration tests — require live TimescaleDB
TEST_DB_URL="postgresql+psycopg2://trader:traderpass@localhost:5432/trading" \
    .venv/bin/pytest tests/test_storage.py -v
```

| Test file | Tests | Scope |
|---|---|---|
| `test_alpaca_client.py` | 20 | Alpaca data + news clients |
| `test_alphavantage_client.py` | 30 | Alpha Vantage news + rate limiting |
| `test_hmm_regime.py` | 28 | HMM fit, predict, online update, persistence |
| `test_mean_reversion.py` | 36 | Cointegration, OU params, Kalman hedge ratio |
| `test_llm_sentiment.py` | 50 | LLM prompt, parse, retry, pipeline |
| `test_backtest_engine.py` | 27 | BacktestEngine, walk-forward, bias checks |
| `test_mwu_agent.py` | 49 | MWU weights, decide, update, scheduled_update |
| `test_executor.py` | 41 | RiskManager, Kelly sizing, OrderExecutor |
| `test_engine.py` | 46 | TradingEngine bar_handler, jobs, shutdown, StateManager |
| `test_storage.py` | — | Integration (requires `TEST_DB_URL`) |
| **Total (unit)** | **327** | |

---

## State persistence

Engine metadata is saved atomically to `models/engine_state.json` after each
EOD job and on clean shutdown:

- Format version, UTC timestamp, tickers, pairs, per-ticker Kelly stats, HMM
  fitted flags
- SHA-256 checksum embedded — load raises `ValueError` on corruption
- Rolling 3-backup rotation (`engine_state.json.bak1` … `.bak3`)

HMM models, Kalman filters, and MWU weights are persisted by their own modules
(`models/hmm_{ticker}.pkl`, `models/kalman_{t1}_{t2}.pkl`,
`models/mwu_weights.npy`).

---

## Scheduled jobs

| Job | Trigger | Action |
|---|---|---|
| `sentiment_job` | Every 4 hours (market hours only, 09:30–16:00 ET) | Alpha Vantage fetch → Gemma scoring → `signal_log` insert |
| `eod_job` | 16:05 ET Mon–Fri | P&L log, MWU performance report, Kelly stat refresh, state save |

---

## Development phases

| Phase | Scope | Status |
|---|---|---|
| 1 — Data layer | TimescaleDB, Alpaca client, Alpha Vantage client | Complete |
| 2 — Signals | HMM regime, Kalman pairs / OU mean-reversion, LLM sentiment | Complete |
| 3 — Backtesting | BacktestEngine (vectorbt), walk-forward, bias checks | Complete |
| 4 — Meta-agent | MWU ensemble conditioned on HMM regime | Complete |
| 5 — Execution | Fractional Kelly sizing, risk controls, order routing, orchestration | Complete |

---

## License

MIT
