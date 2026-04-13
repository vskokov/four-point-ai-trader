# Four-Point AI Trader

An autonomous stock trading engine for Alpaca paper (and live) trading. The
system ingests real-time market data and news, detects market regimes, computes
ensemble signals, sizes positions with fractional Kelly criterion, enforces
mandatory risk controls, optimises a portfolio daily via Black-Litterman, and
routes orders — all driven by a local LLM (Ollama / Gemma) for news sentiment.

```
                          ┌──────────────────────────────────────────────────┐
                          │              TradingEngine (orchestrator)         │
                          │                                                  │
  pair_scanner.py ───────►│  config/discovered_pairs.json                   │
  (run weekly)            │         │ (read at startup)                      │
                          │         ▼                                        │
  Alpaca WebSocket ───────┤► bar_handler()                                  │
                          │    │                                             │
  APScheduler ────────────┤    ├─► HMMRegimeDetector                        │
    sentiment (25/35 min)  │    ├─► KalmanHedgeRatio + OUSpreadSignal        │
    market_open (09:31 ET) │    ├─► LLM signal (from signal_log)            │
    eod_job (16:05 ET)     │    ├─► Analyst recs (yfinance, 24 h cache)     │
                          │    └─► MWUMetaAgent.scheduled_update()         │
                          │              │                                   │
                          │              └─► OrderExecutor                  │
                          │                    RiskManager (Kelly + CBs)    │
                          │                    PortfolioOptimizer (BL/MV)   │
                          └──────────────────────────────────────────────────┘
```

---

## Features

| Layer                | Capability                                                                                                                                                                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data**             | TimescaleDB hypertables for OHLCV, signals, regimes, news, trade decisions; Alpaca market data + WebSocket bars; Alpaca News (all tickers, single call per pipeline run); yfinance market-cap ranking, earnings dates, and analyst recommendations with 24 h cache                         |
| **Pair discovery**   | Standalone scanner (`pair_scanner.py`) scans a ticker universe for cointegrated pairs; correlation pre-filter on log-returns, Engle-Granger + Johansen tests, OU half-life filter; results written to JSON                                                                                 |
| **Regime detection** | 3-state Gaussian HMM (bear / neutral / bull) with deterministic post-hoc state labelling; online partial-fit every 20 bars                                                                                                                                                                 |
| **Pairs trading**    | Kalman-filter adaptive hedge ratio; Ornstein-Uhlenbeck spread signal with z-score thresholds; periodic cointegration health checks                                                                                                                                                         |
| **LLM sentiment**    | Local Ollama (Gemma 4 e4b) scores news headlines into directional signals; retries on malformed JSON; 60-second timeout → safe neutral fallback                                                                                                                                            |
| **Meta-agent**       | Multiplicative Weights Update (MWU) ensemble over 4 signals (HMM, OU, LLM, analyst recs) conditioned on HMM regime; analyst recs start at half weight (1/7 vs 2/7); per-regime weight isolation; online learning from realised price directions                                            |
| **Backtesting**      | Walk-forward vectorbt engine; Sharpe, CAGR, max-drawdown metrics; bias checks; CSV + PNG results                                                                                                                                                                                           |
| **Portfolio**        | Daily Black-Litterman optimisation at 09:31 ET with MWU scores as views; LedoitWolf covariance; min-variance fallback; rebalance execution (sells before buys, cash re-fetched after sells)                                                                                                |
| **Risk management**  | Fractional Kelly criterion (¼ Kelly default); per-position cap (10 % of equity); all buy sizing off **cash** (not equity or buying power) to enforce cash-only / no-margin trading; peak-drawdown circuit breaker (15 %); daily-loss circuit breaker (5 %)                                 |
| **Execution**        | Alpaca `TradingClient`; market orders (DAY); sell capped at held quantity; emergency `close_all_positions`; market-open guard (Alpaca clock API, 60 s cache) blocks orders on weekends, holidays, and early closes                                                                         |
| **Orchestration**    | APScheduler (4 cron jobs); SIGINT / SIGTERM graceful shutdown; atomic state persistence with SHA-256 checksum + 3-backup rotation                                                                                                                                                          |
| **Dashboard**        | 4-tab Streamlit app (`dashboard/app.py`): **Overview** (equity curve + open positions + trade log), **Ticker Detail** (candlestick with regime bands, confirmed-fill markers, OU z-score + MWU score subplots), **Signals** (MWU weight evolution, signal win rates via LATERAL JOIN, signal agreement matrix), **News** (filterable headlines with LLM direction); auto-refreshes every 60 s |
| **Decision quality** | Offline analysis framework (`analysis/`); joins `trade_log` + `ohlcv` for forward-return labels at 4 horizons; per-signal accuracy, MWU weight evolution, and parameter sensitivity sweeps for `hours_back`, `entry_z`, `min_confidence`, `eta`; Markdown report with auto-recommendations |

---

## Architecture

### Module map

```
trading_engine/
├── config/
│   ├── settings.py              # All constants loaded from .env
│   ├── av_rate_state.json       # Auto-created; Alpha Vantage daily call counter
│   └── discovered_pairs.json    # Written by pair_scanner.py; read at engine startup
├── data/
│   ├── storage.py               # TimescaleDB — OHLCV, signals, regimes, news
│   ├── alpaca_client.py         # AlpacaMarketData + AlpacaNewsClient (bars, quotes, news, stream)
│   ├── alphavantage_client.py   # AlphaVantageNewsClient (retained for reference; not used in pipeline)
│   └── fundamentals_client.py   # FundamentalsClient — yfinance market cap, earnings dates, analyst recs (24 h cache)
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
├── portfolio/
│   └── portfolio_optimizer.py   # PortfolioOptimizer (Black-Litterman + Min-Variance)
├── tools/
│   └── pair_scanner.py          # Standalone pair discovery CLI (run weekly)
├── scripts/                     # Connectivity / smoke-test scripts (load .env, real API calls)
│   ├── check_alphavantage.py    # AV budget + news fetch check
│   ├── check_alpaca.py          # Account, clock, quote, OHLCV, news check
│   └── check_yfinance.py        # Market cap fetch, cache timing, field preview
├── analysis/
│   ├── outcome_labeler.py        # Join trade_log + ohlcv → forward-return labels
│   ├── signal_quality.py         # Per-signal accuracy (hmm/ou/llm/analyst) by regime + confidence
│   ├── weight_evolution.py       # MWU weight trajectory from trade_log.mwu_weights
│   ├── parameter_sweep.py        # Sensitivity for hours_back, entry_z, min_confidence, eta
│   ├── report.py                 # Markdown report generator with auto-recommendations
│   └── run_analysis.py           # CLI entry point
├── dashboard/
│   └── app.py                   # Streamlit trade decision dashboard
├── models/                      # Auto-created; HMM .pkl, Kalman .pkl, MWU .npy
├── utils/
│   └── logging.py               # structlog factory (JSON file + console; colored regime banner)
├── tests/                       # 542 unit tests — no live connections required
│   ├── tools/
│   │   └── test_pair_scanner.py
│   └── ...
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
      ├──► FundamentalsClient.get_analyst_recommendations()  →  ±1/0 (24 h cache)
      │
      ▼
 signals = {
   "hmm_regime":    {signal: ±1/0, confidence},
   "ou_spread":     {signal: ±1/0, confidence},
   "llm_sentiment": {signal: ±1/0, confidence},
   "analyst_recs":  {signal: ±1/0, confidence},
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
            Storage.insert_trade_log()   ← full decision snapshot to DB
            AlpacaMarketData.is_market_open()  ← Alpaca clock API, 60 s cache
              └── if closed → skip order (bar still fully processed)
            RiskManager.check_trade()
              ├── circuit breaker (drawdown / daily loss)
              └── position size limit (max_size capped at available cash)
            RiskManager.kelly_size(win_rate, avg_win, avg_loss)
            size_usd = cash × kelly_f × confidence   ← cash, not equity
            n_shares = floor(size_usd / mid_price)
            [hard cap: n_shares × price ≤ cash]
            TradingClient.submit_order(MarketOrderRequest, DAY)
```

---

## Requirements

- Python 3.12
- Docker (for TimescaleDB)
- [Ollama](https://ollama.com/) with `gemma4:e4b` pulled
- Alpaca account (paper recommended for testing)
- Alpha Vantage API key (required by `settings.py` at startup, but not used for news — AV free tier was abandoned due to multi-ticker query restrictions)

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

| Variable               | Required | Default                            | Description                                                                 |
| ---------------------- | -------- | ---------------------------------- | --------------------------------------------------------------------------- |
| `ALPACA_API_KEY`       | yes      | —                                  | Alpaca API key                                                              |
| `ALPACA_SECRET_KEY`    | yes      | —                                  | Alpaca secret key                                                           |
| `ALPHAVANTAGE_API_KEY` | yes      | —                                  | Alpha Vantage API key (loaded at startup; AV not used for news in pipeline) |
| `DB_URL`               | yes      | —                                  | `postgresql+psycopg2://trader:traderpass@localhost:5432/trading`            |
| `ALPACA_BASE_URL`      | no       | `https://paper-api.alpaca.markets` | Use paper endpoint                                                          |
| `OLLAMA_HOST`          | no       | `http://localhost:11434`           | Ollama server URL                                                           |
| `OLLAMA_MODEL`         | no       | `gemma4:e4b`                       | Ollama model tag                                                            |
| `LOG_LEVEL`            | no       | `INFO`                             | `DEBUG` / `INFO` / `WARNING`                                                |

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

### Step 1 — Discover pairs (run weekly)

Before starting the engine for the first time, scan your ticker universe for
cointegrated pairs. Results are written to `config/discovered_pairs.json` and
loaded automatically at engine startup.

```bash
cd trading_engine
.venv/bin/python -m trading_engine.tools.pair_scanner \
    --tickers LMT NOC RTX GD BA MSFT GOOG AAPL NVDA AMD \
    --lookback-days 504 \
    --min-correlation 0.70 \
    --max-pvalue 0.05 \
    --min-half-life 5 \
    --max-half-life 60 \
    --max-pairs 10
```

To automate the weekly rescan, add a cron entry (`crontab -e`):

```
0 8 * * 1 cd /path/to/four-point-ai-trader && trading_engine/.venv/bin/python -m trading_engine.tools.pair_scanner --tickers AAPL MSFT ... --lookback-days 504 --min-correlation 0.50 --max-pvalue 0.05 --min-half-life 5 --max-half-life 60 --max-pairs 20 >> /tmp/pair_scanner.log 2>&1
```

This fires every Monday at 08:00. Replace the ticker list with your universe.
`discovered_pairs.json` is gitignored — each environment maintains its own.

| Flag                | Default                        | Description                                              |
| ------------------- | ------------------------------ | -------------------------------------------------------- |
| `--tickers`         | required                       | Ticker universe to scan                                  |
| `--lookback-days`   | 504                            | Calendar days of history (~2 trading years)              |
| `--output`          | `config/discovered_pairs.json` | Output file path                                         |
| `--min-correlation` | 0.70                           | Minimum log-return Pearson correlation                   |
| `--max-pvalue`      | 0.05                           | Maximum Engle-Granger p-value                            |
| `--min-half-life`   | 5                              | Minimum OU half-life in bars (too fast = noise)          |
| `--max-half-life`   | 60                             | Maximum OU half-life in bars (too slow = no opportunity) |
| `--max-pairs`       | 10                             | Maximum output pairs (ranked by EG p-value)              |

### Step 2 — Run the engine

```bash
cd trading_engine
.venv/bin/python -m trading_engine.main \
    --tickers AAPL MSFT JPM BAC \
    --log-level INFO
```

The engine reads `config/discovered_pairs.json` at startup and automatically
merges pair tickers into the subscription universe. If the file is missing the
engine still runs on HMM + LLM signals only.

### Live trading

```bash
.venv/bin/python -m trading_engine.main \
    --tickers AAPL MSFT \
    --live \
    --log-file /var/log/trader.json
```

### CLI reference

```
usage: four-point-trader [--tickers TICKER [TICKER ...]]
                         [--pairs-file PATH]
                         [--live]
                         [--log-level {DEBUG,INFO,WARNING,ERROR}]
                         [--log-file PATH]
                         [--update-kelly-stats]

options:
  --tickers            Equity symbols to trade (default: AAPL MSFT JPM BAC)
  --pairs-file         Path to discovered_pairs.json (default: config/discovered_pairs.json)
  --live               Connect to Alpaca live trading  [paper is default]
  --log-level          Logging verbosity (default: INFO)
  --log-file           Optional path for newline-delimited JSON log file
  --update-kelly-stats Recompute Kelly sizing stats from Alpaca confirmed fill P&L,
                       write updated engine_state.json, and exit (no trading)
```

Shutdown cleanly with `Ctrl-C` or `SIGTERM`. If the circuit breaker fires, all
positions are liquidated before exit.

### Step 3 — Verify connectivity (optional, one-time)

Before running the engine for the first time, confirm all external services
are reachable with real credentials:

```bash
cd trading_engine

# Alpaca — account info, market clock, latest quote, OHLCV, Alpaca news
.venv/bin/python scripts/check_alpaca.py

# yfinance — no API key needed; market cap table + cache timing
.venv/bin/python scripts/check_yfinance.py

# Alpha Vantage — connectivity only (AV not used for news; retained for reference)
.venv/bin/python scripts/check_alphavantage.py
```

Each script prints `✓ / ✗ / ⚠` per check and exits 0 on success.

### Step 5 — Run the dashboard (optional)

In a separate terminal, start the Streamlit monitoring dashboard:

```bash
cd trading_engine
.venv/bin/streamlit run dashboard/app.py
```

Open `http://localhost:8501`. Four tabs:
- **Overview** — equity curve with confirmed-fill markers, open positions table, collapsible trade log
- **Ticker Detail** — candlestick + regime background bands + OU z-score + MWU score subplots, ticker selector, period picker
- **Signals** — MWU weight evolution, per-signal win rates (1m/15m/1h horizons), signal agreement matrix
- **News** — filterable headlines table with LLM direction and sentiment scores

Auto-refreshes every 60 s.

---

## Risk controls

| Control                 | Default        | Description                                                                                          |
| ----------------------- | -------------- | ---------------------------------------------------------------------------------------------------- |
| Cash-only sizing        | —              | All buy orders sized off `account.cash`, never `equity` or `buying_power`; no margin used            |
| Max position per ticker | 10 % of equity | Position limit % measured against equity; dollar cap further constrained to available cash           |
| Peak drawdown halt      | 15 %           | Triggers circuit breaker, liquidates all positions                                                   |
| Daily loss halt         | 5 %            | Triggers circuit breaker, liquidates all positions                                                   |
| Kelly fraction          | ¼ Kelly        | `RiskManager.kelly_fraction`                                                                         |
| Order type              | Market / DAY   | No limit orders; sells capped at held quantity; sells are not cash-constrained                       |
| Rebalance cash gate     | —              | Cash re-fetched after sells; each buy deducts from `available_cash`; insufficient cash → buy skipped |

The circuit breaker fires **before** any order on every bar and before every
portfolio rebalance. If triggered it sets an emergency flag, signals the
shutdown event, and `close_all_positions` is called during the shutdown
sequence.

---

## Testing

```bash
cd trading_engine

# Unit tests — 542 tests, no live connections required
.venv/bin/pytest tests/test_alpaca_client.py \
                 tests/test_alphavantage_client.py \
                 tests/test_hmm_regime.py \
                 tests/test_mean_reversion.py \
                 tests/test_llm_sentiment.py \
                 tests/backtesting/test_backtest_engine.py \
                 tests/meta_agent/test_mwu_agent.py \
                 tests/execution/test_executor.py \
                 tests/test_engine.py \
                 tests/portfolio/ \
                 tests/tools/ \
                 tests/test_fundamentals_client.py \
                 tests/analysis/ -v

# Integration tests — require live TimescaleDB
TEST_DB_URL="postgresql+psycopg2://trader:traderpass@localhost:5432/trading" \
    .venv/bin/pytest tests/test_storage.py -v
```

| Test file                     | Tests   | Scope                                                                                                                                                                                       |
| ----------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_alpaca_client.py`       | 25      | Alpaca data + news clients; `is_market_open` cache                                                                                                                                          |
| `test_alphavantage_client.py` | 30      | Alpha Vantage news + rate limiting                                                                                                                                                          |
| `test_hmm_regime.py`          | 28      | HMM fit, predict, online update, persistence                                                                                                                                                |
| `test_mean_reversion.py`      | 36      | Cointegration, OU params, Kalman hedge ratio                                                                                                                                                |
| `test_llm_sentiment.py`       | 56      | LLM prompt, parse, retry, pipeline (Alpaca-only mode)                                                                                                                                       |
| `test_backtest_engine.py`     | 27      | BacktestEngine, walk-forward, bias checks                                                                                                                                                   |
| `test_mwu_agent.py`           | 53      | MWU weights (4-signal, half-weight init), decide, update, scheduled_update                                                                                                                  |
| `test_executor.py`            | 47      | RiskManager, Kelly sizing, OrderExecutor; cash-only enforcement; market-closed guard                                                                                                        |
| `test_engine.py`              | 106     | TradingEngine bar_handler, jobs, shutdown, StateManager, pairs loading; rebalance cash gate; HMM seeding; Alpaca-only sentiment routing; earnings guard; analyst signal logged to trade_log; Kelly stats from fills |
| `test_portfolio_optimizer.py` | 9       | Black-Litterman, min-variance, rebalance orders                                                                                                                                             |
| `test_pair_scanner.py`        | 21      | Pair scanner pipeline, filter stages, JSON output                                                                                                                                           |
| `test_fundamentals_client.py` | 32      | FundamentalsClient market cap, earnings dates, analyst recommendations — all with 24 h cache                                                                                                |
| `tests/analysis/`             | 68      | Offline analysis framework — outcome labels, per-signal accuracy, weight evolution, parameter sweeps; no DB required                                                                        |
| `test_storage.py`             | —       | Integration (requires `TEST_DB_URL`)                                                                                                                                                        |
| **Total (unit)**              | **543** |                                                                                                                                                                                             |

---

## State persistence

Engine metadata is saved atomically to `models/engine_state.json` after each
EOD job and on clean shutdown:

- Format version, UTC timestamp, tickers, pairs, per-ticker Kelly stats, HMM
  fitted flags
- SHA-256 checksum embedded — load raises `ValueError` on corruption
- Rolling 3-backup rotation (`engine_state.json.bak1` … `.bak3`)

**Kelly stats** (`win_rate`, `avg_win`, `avg_loss` per ticker) are recomputed
at EOD from Alpaca confirmed fill P&L using FIFO round-trip pairing (90-day
lookback, minimum 5 round-trips per ticker before deviating from conservative
defaults of 52% win rate). On startup, any `win_rate < 0.30` loaded from state
is automatically reset to defaults. To manually refresh after a stale state file:

```bash
.venv/bin/python -m trading_engine.main --tickers ... --update-kelly-stats
```

HMM models, Kalman filters, and MWU weights are persisted by their own modules
(`models/hmm_{ticker}.pkl`, `models/kalman_{t1}_{t2}.pkl`,
`models/mwu_weights_{ticker}.npy` — one file per ticker).

To force a clean restart (refit all models from scratch):

```bash
rm trading_engine/models/hmm_*.pkl
rm trading_engine/models/mwu_weights_*.npy
rm trading_engine/models/engine_state.json*
```

---

## Scheduled jobs

| Job                   | Trigger                              | Action                                                                                |
| --------------------- | ------------------------------------ | ------------------------------------------------------------------------------------- |
| `sentiment_job_early` | Every 25 min, 07:00–10:29 ET Mon–Fri | All tickers via Alpaca News (single fetch call) → Gemma scoring → `signal_log` insert |
| `sentiment_job_late`  | Every 35 min, 10:30–16:30 ET Mon–Fri | Same as above (lower cadence for later session)                                       |
| `market_open_job`     | 09:31 ET Mon–Fri                     | Black-Litterman portfolio optimisation → rebalance execution (cash-gated buys)        |
| `eod_job`             | 16:05 ET Mon–Fri                     | P&L log, MWU performance report, Kelly stat refresh, state save                       |

No Alpha Vantage calls are made during sentiment jobs. AV's free tier was abandoned
because it rejects multi-ticker NEWS_SENTIMENT queries with `"Invalid inputs"` and its
20-call/day quota is exhausted within a few hours. All news now routes through
`AlpacaNewsClient` with no daily call budget constraint.

---

## Development phases

| Phase                 | Scope                                                                                                                                                                                                                                                                   | Status   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1 — Data layer        | TimescaleDB, Alpaca client, Alpha Vantage client                                                                                                                                                                                                                        | Complete |
| 2 — Signals           | HMM regime, Kalman pairs / OU mean-reversion, LLM sentiment                                                                                                                                                                                                             | Complete |
| 3 — Backtesting       | BacktestEngine (vectorbt), walk-forward, bias checks                                                                                                                                                                                                                    | Complete |
| 4 — Meta-agent        | MWU ensemble conditioned on HMM regime                                                                                                                                                                                                                                  | Complete |
| 5 — Execution         | Fractional Kelly sizing, risk controls, order routing, orchestration                                                                                                                                                                                                    | Complete |
| 6 — Portfolio         | Black-Litterman + Min-Variance optimisation, daily rebalance                                                                                                                                                                                                            | Complete |
| 7 — Pair discovery    | Standalone cointegration scanner, JSON-driven pair loading                                                                                                                                                                                                              | Complete |
| 8 — Market-open guard | Alpaca clock API guard on all order paths; holiday / early-close safe                                                                                                                                                                                                   | Complete |
| 9 — Cash-only trading | Buy sizing off `cash` not `equity`; hard cash cap; rebalance cash gate                                                                                                                                                                                                  | Complete |
| 10 — Observability    | Trade decision dashboard (Streamlit); colored regime banner in logs; `trade_log` DB table; contributing headlines persisted; per-ticker MWU weight files; HMM history seeding at startup                                                                                | Complete |
| 11 — News routing     | `FundamentalsClient` (yfinance, 24 h cap cache); all tickers → Alpaca News (AV abandoned — free tier rejects multi-ticker queries); connectivity check scripts                                                                                                          | Complete |
| 12 — Analyst signal   | `FundamentalsClient.get_analyst_recommendations` (24 h cache); 4th MWU signal at half initial weight (1/7); `analyst_signal` + `analyst_confidence` columns in `trade_log`; auto-migration on bootstrap                                                                 | Complete |
| 13 — Decision quality | Offline analysis framework (`analysis/`); forward-return outcome labeling at 1 m / 15 m / 1 h / 4 h; per-signal accuracy and IC; MWU weight evolution; parameter sweeps for `hours_back`, `entry_z`, `min_confidence`, `eta`; Markdown report with auto-recommendations | Complete |
| 14 — Dashboard redesign | 4-tab Streamlit app; candlestick with regime bands + confirmed-fill markers; OU z-score + MWU score subplots; MWU weight evolution; signal win rates via LATERAL JOIN; signal agreement matrix; news tab | Complete |
| 15 — Kelly stats fix | Kelly sizing stats computed from Alpaca confirmed fill P&L (FIFO round-trip pairing, 90-day window, min 5 trips); decoupled from MWU win rates; startup sanity reset for stale state; `--update-kelly-stats` CLI flag | Complete |

---

## License

MIT
