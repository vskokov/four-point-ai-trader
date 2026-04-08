# CLAUDE.md — Four-Point AI Trader

## Project overview

Autonomous stock trading engine in Python. Ingests market data and news,
computes signals, detects market regimes, manages a portfolio, and executes
orders via Alpaca — with an LLM meta-agent (Ollama/Gemma) providing
directional sentiment from news headlines.

All runnable code lives under `trading_engine/`.

### Phase status

| Phase | Scope | Status |
|---|---|---|
| **1 — Data layer** | Storage (TimescaleDB), Alpaca client, Alpha Vantage client | **Complete** |
| **2 — Signals** | HMM regime detector, Kalman pairs / OU mean-reversion, LLM sentiment | **Complete** |
| **3 — Backtesting** | BacktestEngine (vectorbt), walk-forward validation, bias checks | **Complete** |
| **4 — Meta-agent** | MWU ensemble agent conditioned on HMM regime | **Complete** |
| **5 — Execution** | RiskManager (Kelly + circuit breakers), OrderExecutor, TradingEngine orchestrator, StateManager | **Complete** |
| **6 — Portfolio** | PortfolioOptimizer (Black-Litterman + Min-Variance, LedoitWolf, daily rebalance job) | **Complete** |

---

## Repository layout

```
trading_engine/
├── config/
│   ├── settings.py          # All constants; loads from .env via python-dotenv
│   └── av_rate_state.json   # Auto-created; tracks Alpha Vantage daily call count
├── data/
│   ├── storage.py           # TimescaleDB interface — COMPLETE
│   ├── alpaca_client.py     # AlpacaMarketData + AlpacaNewsClient — COMPLETE
│   └── alphavantage_client.py # AlphaVantageNewsClient (primary news) — COMPLETE
├── signals/
│   ├── hmm_regime.py        # GaussianHMM regime detector (bear/neutral/bull) — COMPLETE
│   ├── kalman_pairs.py      # Kalman adaptive hedge ratio for pairs — COMPLETE
│   ├── mean_reversion.py    # CointegrationTest + OUSpreadSignal — COMPLETE
│   └── llm_sentiment.py     # Ollama/Gemma news sentiment signal — COMPLETE
├── backtesting/
│   ├── backtest_engine.py   # BacktestEngine (vectorbt) — COMPLETE
│   └── results/             # Auto-created; CSV + PNG outputs written here
├── meta_agent/
│   └── mwu_agent.py         # MWUMetaAgent — MWU ensemble over signals — COMPLETE
├── execution/
│   └── executor.py          # RiskManager + OrderExecutor — COMPLETE
├── orchestrator/
│   ├── engine.py            # TradingEngine — top-level loop — COMPLETE
│   └── state_manager.py     # Atomic JSON state + checksum + 3-backup rotation — COMPLETE
├── portfolio/
│   └── portfolio_optimizer.py # PortfolioOptimizer (Black-Litterman + Min-Variance) — COMPLETE
├── utils/
│   └── logging.py           # structlog factory
├── main.py                  # CLI entry point — COMPLETE
├── tests/
│   ├── test_storage.py              # Integration tests (requires TEST_DB_URL)
│   ├── test_alpaca_client.py        # Unit tests — fully mocked (20 tests)
│   ├── test_alphavantage_client.py  # Unit tests — fully mocked (30 tests)
│   ├── test_hmm_regime.py           # Unit tests — fully mocked (28 tests)
│   ├── test_mean_reversion.py       # Unit tests — fully mocked (36 tests)
│   ├── test_llm_sentiment.py        # Unit tests — fully mocked (50 tests)
│   ├── backtesting/
│   │   └── test_backtest_engine.py  # Unit tests — synthetic data (27 tests)
│   ├── meta_agent/
│   │   └── test_mwu_agent.py        # Unit tests — no DB or network (49 tests)
│   ├── execution/
│   │   └── test_executor.py         # Unit tests — fully mocked (41 tests)
│   ├── portfolio/
│   │   └── test_portfolio_optimizer.py  # Unit tests — fully mocked (9 tests)
│   └── test_engine.py               # Unit tests — fully mocked (46 tests)
├── conftest.py              # Adds repo root to sys.path for pytest
├── requirements.txt
├── docker-compose.yml       # TimescaleDB container
└── .env.example             # Template — copy to .env, never commit .env
```

---

## Environment setup

```bash
cd trading_engine
cp .env.example .env        # fill in real credentials
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Required env vars (missing ones raise `KeyError` at import time)

| Variable | Description |
|---|---|
| `ALPACA_API_KEY` | Alpaca key |
| `ALPACA_SECRET_KEY` | Alpaca secret |
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage key |
| `DB_URL` | SQLAlchemy URL, e.g. `postgresql+psycopg2://trader:traderpass@localhost:5432/trading` |

### Optional env vars with defaults

| Variable | Default |
|---|---|
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` |
| `LOG_LEVEL` | `INFO` |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `gemma4:e4b` (installed tag in Docker) |

---

## Database

TimescaleDB runs in Docker. `vskokov` is in the `docker` group — no `sudo`
needed:

```bash
# Start
docker compose up -d

# Status
docker compose ps

# Stop
docker compose down
```

`Storage.__init__` auto-creates all tables and hypertables on first connect —
no separate migration step needed.

### Schema

| Table | Type | Partition | Notes |
|---|---|---|---|
| `ohlcv` | hypertable | `time` | |
| `signal_log` | hypertable | `time` | |
| `news` | regular table | — | Needs `UNIQUE headline_hash` — see gotcha #3 |
| `regime_log` | regular table | — | |

---

## Running tests

```bash
cd trading_engine

# Unit tests only — no DB or network required (336 tests across 9 files)
.venv/bin/pytest tests/test_alpaca_client.py tests/test_alphavantage_client.py \
    tests/test_hmm_regime.py tests/test_mean_reversion.py tests/test_llm_sentiment.py \
    tests/backtesting/test_backtest_engine.py tests/meta_agent/test_mwu_agent.py \
    tests/execution/test_executor.py tests/test_engine.py tests/portfolio/ -v

# Integration tests — require live TimescaleDB
TEST_DB_URL="postgresql+psycopg2://trader:traderpass@localhost:5432/trading" \
    .venv/bin/pytest tests/test_storage.py -v

# Full suite
TEST_DB_URL="postgresql+psycopg2://trader:traderpass@localhost:5432/trading" \
    .venv/bin/pytest tests/ -v
```

`test_storage.py` is skipped automatically when `TEST_DB_URL` is unset.
Teardown deletes all rows with `ticker = 'TEST'` after the module-scoped session.

---

## Coding conventions

- **Python 3.12**, type hints throughout, PEP 8.
- `from __future__ import annotations` in every module (enables forward refs).
- All credentials from env vars only — never hardcoded.
- Use `get_logger(__name__)` from `utils/logging.py` in every module.
- Log DB credentials stripped: `db_url.split("@")[-1]` before logging.
- SQLAlchemy 2.0 style: always `text()` for raw SQL, `conn.execute(text(...), params)`.
- For `INSERT ... ON CONFLICT DO NOTHING` with accurate count: use `RETURNING`
  and execute row-by-row (not executemany) — executemany rowcount is unreliable.
- Dict/JSONB columns (`metadata`, `regime_probs`) must be `json.dumps()`-ed
  before binding; psycopg2 does not auto-cast Python dicts to JSONB.

### Signals layer (`signals/`)

- All model persistence uses `joblib` to `trading_engine/models/`.
- The `models_dir` path is injectable in every class `__init__` — pass
  `tmp_path` in tests; the default resolves relative to the source file.
- Signal results are inserted into `signal_log` (value = the signal scalar)
  and/or `regime_log` via the `Storage` instance passed as a method argument,
  never imported at module level.
- `settings` IS imported at module level in signal modules (for default
  parameter values). Tests that need to override settings must either pass
  explicit constructor arguments or monkeypatch
  `trading_engine.signals.<module>.settings` before importing the class.

### Alpaca client (`data/alpaca_client.py`)

- `AlpacaNewsClient` is **optional** — not called in the default pipeline.
  `AlphaVantageNewsClient` is the primary news source (pre-computed sentiment).
- `fetch_news()` returns raw dicts and does **not** insert into the DB — the
  sentiment module scores them first, then calls `Storage.insert_news()`.
- `_with_retry` duck-types `status_code` via `getattr` (not `except APIError`)
  so it works with test fakes and stays robust across SDK versions.
- `NewsRequest.symbols` is `Optional[str]` (comma-separated), not a list.
  Use `",".join(tickers)`.

### Alpha Vantage client (`data/alphavantage_client.py`)

- **Primary news source** — always prefer over `AlpacaNewsClient`.
- Rate limit: 25 req/day free tier. Tracked in `config/av_rate_state.json`.
  Warn at 15 calls, hard-block at 20 (raises `RateLimitExceeded` before HTTP call).
- AV returns HTTP 200 for errors — detect `"Information"` and `"Note"` keys
  in the body and raise `AlphaVantageError`.
- `rate_state_path` is injectable in `__init__` — pass `tmp_path / "..."` in
  tests to avoid touching production state (no module patching needed).
- `is_market_hours()` uses `zoneinfo.ZoneInfo("America/New_York")` — ET/EDT
  handled automatically; window is 09:30 ≤ t < 16:00 weekdays only.

---

## Key gotchas

### Storage layer

1. **`structlog.stdlib.add_logger_name` requires a stdlib logger** — always use
   `stdlib.LoggerFactory()`, never `PrintLoggerFactory()`, or it crashes with
   `AttributeError: 'PrintLogger' object has no attribute 'name'`.

2. **PostgreSQL INTERVAL with bind params**: `INTERVAL ':n hours'` is invalid
   (param inside a string literal). Use `(:n * INTERVAL '1 hour')` instead.

3. **`news` cannot be a hypertable**: TimescaleDB requires the partition column
   in any `UNIQUE` constraint. `headline_hash` uniqueness can't include `time`,
   so `news` stays a regular table.

4. **Import path for pytest**: running `pytest` from inside `trading_engine/`
   means `trading_engine` is not on `sys.path`. `conftest.py` at the package
   root inserts the parent directory to fix this.

5. **Docker socket**: `vskokov` is not in the `docker` group — all
   `docker compose` commands require `sudo`. Claude cannot run these;
   user must execute and paste output.

### Alpaca client

6. **`MagicMock(spec=SomeClass)` is not an instance of `SomeClass`** — it won't
   be caught by `except SomeClass`. Use a plain `Exception` subclass with the
   right attributes as a test fake. `_with_retry` uses `except Exception` +
   `getattr(exc, 'status_code', None)` for this reason.

7. **`NewsRequest.symbols` is `Optional[str]`**, not a list — verified via
   `Model.model_fields`. Always inspect Pydantic fields before assuming types.

8. **`alpaca-trade-api` removed from requirements** — use `alpaca.data.*` and
   `alpaca.trading.*` exclusively. Never import from `alpaca_trade_api`.

### Alpha Vantage client

9. **AV non-standard errors**: rate-limit and auth errors come back as HTTP 200
   with `{"Information": "..."}` or `{"Note": "..."}`. Always call
   `_check_av_body_errors(body)` before parsing `feed`.

10. **AV time formats differ between request and response**: `time_from` request
    param uses `YYYYMMDDTHHMM` (no seconds); `time_published` in the response
    uses `YYYYMMDDTHHMMSS` (with seconds). Parser handles both via `len()` check.

11. **Patching `datetime` for market-hours tests**: patch `datetime` at the
    module level (`trading_engine.data.alphavantage_client.datetime`) with a
    subclass that overrides `now()`. Don't patch `time` or `os`.

### Signals layer

12. **HMM state ordering is arbitrary** — EM assigns state indices randomly
    each run. Always post-hoc rank states by mean log-return and store a
    `state_labels` dict (`HMMRegimeDetector._assign_state_labels`). Never
    assume state 0 = bear.

13. **Kalman gain absorbs P1 shocks when `p2 ≈ 100`** — with default
    `delta=1e-4`, `observation_noise=1e-3`, and `p2 ≈ 100`, the Kalman gain
    `K * p2 ≈ 1`, meaning nearly all measurement shocks are absorbed into
    beta. Threshold-firing tests for signal logic must control the spread
    directly (patch `get_spread`) rather than constructing adversarial price
    series.

14. **`KalmanHedgeRatio.get_spread` uses a fresh internal filter** — it does
    NOT mutate `self.beta` or `self.beta_history`. It stores the final batch
    beta in `self._batch_beta` as a side-effect. Callers (e.g.
    `OUSpreadSignal.compute_signal`) must read `_batch_beta`, not
    `beta_history[-1]`.

15. **OU half-life formula**: `half_life = ln(2) / kappa` where
    `kappa = 1 - b` from OLS of `Z_t ~ Z_{t-1}` with `dt = 1 bar`.
    Guard `kappa > 1e-8` to avoid div-by-zero on non-stationary spreads.

16. **`ollama.Client(host=host, timeout=60)`** — timeout is passed as an
    httpx kwarg. On timeout the client raises `httpx.ReadTimeout`; detect via
    `"timeout" in type(exc).__name__.lower()` so it works across httpx
    versions without importing httpx.

17. **LLM response may include markdown fences** — strip ` ```json … ``` `
    before `json.loads()`. Always validate all six schema keys plus direction
    ∈ {−1,0,1}, confidence ∈ [0,1], horizon ∈ {4h,8h,1d}.

18. **Ollama model tag is `gemma4:e4b`** (not `gemma4`). Ollama runs in
    Docker container named `ollama` on port 11434. Check available models
    with `docker exec ollama ollama list`.

### Backtesting layer

19. **vectorbt returns `inf` Sharpe for zero-variance equity** — a flat (no-trade)
    equity curve has std(returns) = 0, causing vectorbt to produce `inf`.
    `_safe_float` in `BacktestEngine` clamps any non-finite value to `0.0`.
    Test with `math.isfinite()` rather than comparing to `0.0` directly.

20. **vectorbt Sharpe diverges from manual `mean/std * √252`** — when the
    signal is only active for a fraction of the period, the manual computation
    inflates the denominator with zero-return flat bars while vectorbt uses a
    different convention.  Only assert sign agreement and order-of-magnitude
    match in tests (within factor of 3), not an exact percentage.

21. **`size_type='percent'` does not support position reversal in vectorbt** —
    `from_signals` with `entries`/`exits` + `size_type='percent'` is long-only.
    Short signals (−1) act only as exits.  Use `short_entries`/`short_exits`
    arrays if full long/short simulation is needed.

22. **Walk-forward train bar count uses `int()` truncation** — `n_train = int(window * frac)`
    truncates, so `int(25 * 0.5) = 12`, not 13.  Test cases must use window
    sizes exactly divisible by the denominator to get equal train/test splits
    (e.g. window=20 with frac=0.5 → 10/10).

23. **`run_all_signals` queries `signal_log` directly via `storage._engine`** —
    there is no public `query_signal` method on `Storage`.  The backtesting
    engine accesses `storage._engine` with a raw `text()` query, following the
    same SQLAlchemy 2.0 style as the rest of the codebase.

24. **Backtesting results are written to `backtesting/results/`** — the directory
    is auto-created by `_RESULTS_DIR.mkdir(parents=True, exist_ok=True)`.
    CSV files are named `backtest_{YYYYMMDDHHMMSS}.csv`; PNG files are
    `equity_curves_{YYYYMMDDHHMMSS}.png`.  Both use UTC timestamps.

### Meta-agent layer (`meta_agent/`)

25. **MWU weight matrix is `(n_regimes, n_signals)` — updates are regime-isolated** —
    `update_weights` only modifies `self.weights[regime_t]`.  The other two rows
    are never touched.  Tests must verify isolation explicitly.

26. **Neutral signal loss is 0.5, not 0 or 1** — a signal that fires 0 (unavailable
    or genuinely neutral) receives `loss = 0.5`.  This is between correct (0) and
    wrong (1), so neutral signals decay slowly rather than being rewarded or
    penalised strongly.

27. **`get_actual_direction` uses a strict inequality for the "after" slice** —
    bars at exactly `decision_time` fall into the "before" bucket (`<= decision_ts`).
    Test DataFrames must place the outcome bar strictly after `decision_time`
    (e.g. `decision_time + timedelta(minutes=1)`), not at it.

28. **`models/mwu_weights.npy` is created lazily** — the directory is created by
    `_save_weights` via `path.parent.mkdir(parents=True, exist_ok=True)`.  On a
    fresh clone there is no `models/` directory until the first `update_weights`
    call.  Pass `models_dir=tmp_path` in every test to avoid touching production
    state.

29. **`scheduled_update` pending horizon is in wall-clock minutes, not bars** —
    the elapsed-time check uses `timedelta(minutes=horizon_bars)`.  In production
    this means `horizon_bars=1` corresponds to 1 wall-clock minute.  For bar-level
    back-testing, call `update_weights` directly instead.

### Execution layer (`execution/executor.py`)

30. **Never retry order submission** — `_read_with_retry` is only for read-only
    calls (`get_all_positions`, etc.).  Retrying `submit_order` risks duplicate
    fills.  Let submission exceptions propagate and log them.

31. **`close_all_positions` uses Alpaca's built-in atomic call** —
    `TradingClient.close_all_positions(cancel_orders=True)` is used instead of
    looping individual sells.  It also cancels open orders first, which is
    essential for emergency liquidation.

32. **`RiskManager` tracks `_peak_equity` and `_daily_start_equity` internally** —
    both are updated on every `circuit_breaker()` call.  `_daily_start_equity`
    resets at UTC midnight (date comparison).  Tests that cross day boundaries
    must monkeypatch `trading_engine.execution.executor.datetime`.

33. **structlog + pytest `caplog` incompatibility** — structlog's
    `ConsoleRenderer` writes to stdout; `caplog.records` is always empty for
    structlog output.  Use `capsys.readouterr().out` to assert on log messages
    in tests.

### Orchestrator (`orchestrator/engine.py`, `orchestrator/state_manager.py`)

34. **`bar_handler` must never raise** — it runs in the Alpaca WebSocket
    stream's background thread.  All sub-calls (HMM predict, OU signal, order
    submission, account check) are wrapped in `try/except` with a `logger.error`
    or `logger.warning` fallback so the stream thread survives transient errors.

35. **LLM signal is queried from `signal_log`, not recomputed per bar** —
    `sentiment_job` (every 4 h) calls `LLMSentimentSignal.run_pipeline()` which
    inserts into `signal_log` with `signal_name = 'llm_sentiment'`.
    `bar_handler` queries the most recent row within a 12-hour window via raw
    SQL on `storage._engine` (same pattern as `BacktestEngine`).

36. **`MWUMetaAgent.scheduled_update()` is the only MWU call per bar** — it
    internally calls `decide()` and returns the decision dict.  Calling `decide()`
    separately before `scheduled_update()` would score the bar twice and
    pollute `_pending`.

37. **State file is written atomically via `.tmp` → `rename`** — `StateManager`
    writes to `engine_state.json.tmp` then calls `Path.replace()`.  A partial
    write never corrupts the current snapshot.  Checksum is SHA-256 of the JSON
    payload (excluding the `checksum` key itself, computed after all other fields
    are set).

38. **APScheduler `next_run_time` for sentiment job** — setting
    `next_run_time=datetime.now(tz=utc)` causes the job to fire immediately at
    engine startup (before the first 4-hour interval), ensuring fresh sentiment
    data is available before the first bars arrive.

### Portfolio layer (`portfolio/portfolio_optimizer.py`)

39. **`PortfolioOptimizer._get_return_matrix` instantiates `Storage` internally** —
    it does a deferred `from trading_engine.data.storage import Storage` and
    creates a fresh instance each call.  Tests must patch
    `trading_engine.data.storage.Storage` (not a module-level import) and also
    patch `trading_engine.portfolio.portfolio_optimizer.settings` to avoid
    requiring `DB_URL` in the test environment.

40. **`max_weight` must satisfy `max_weight >= 1/n_tickers`** — if the per-ticker
    ceiling is too tight, the weight-sum-to-1 constraint becomes infeasible.
    `PortfolioOptimizer` automatically relaxes the bound to `max(max_weight, 1/n)`
    so optimisation never crashes on a small universe.  The configured
    `max_weight` is still honoured when the ticker count is large enough
    (e.g. 10 tickers × 0.10 = 1.0 — exactly feasible).

41. **Black-Litterman view threshold** — only tickers with `abs(score) >= 0.3`
    AND `confidence >= 0.4` are included as absolute views.  All others are
    ignored; the prior absorbs them.  When no ticker passes this threshold the
    optimizer falls back to `compute_min_variance()` automatically.

42. **`MWUMetaAgent.last_decision` is set by `decide()`, not `scheduled_update()`** —
    `decide()` stores `self.last_decision = result` before returning.
    `market_open_job` reads `getattr(agent, 'last_decision', None)` to get the
    most recent ensemble output without triggering a new scoring round.

43. **`market_open_job` runs at 09:31 ET Mon–Fri via APScheduler cron** — it
    collects `last_decision` from every per-ticker `MWUMetaAgent`, converts the
    `weights` dict max value to a confidence proxy, and calls
    `compute_black_litterman`.  All exceptions are caught; a min-variance
    fallback is attempted on failure.  The engine now registers 4 APScheduler
    jobs (sentiment_early, sentiment_late, market_open, eod) — tests asserting
    job count must expect 4.

44. **`OrderExecutor.portfolio_optimizer` defaults to `None`** — when `None`,
    Kelly sizing is used unchanged.  When set, a non-zero `get_target_weight`
    overrides Kelly: `size_usd = equity * target_w * confidence`, capped at
    `max_position_pct`.  Tests that bypass `__init__` via `__new__` must
    explicitly set `executor.portfolio_optimizer = None`.

45. **Rebalance execution order: sells before buys** — `_execute_rebalance_orders`
    separates the order list from `get_rebalance_orders` into sells and buys, then
    processes `sells + buys` in that sequence.  This ensures capital freed by
    reducing positions is available before new purchases are submitted.
    Sell quantities are capped at currently-held shares (fetched once before the
    loop) to avoid "no position" rejections from Alpaca.

46. **Circuit-breaker gate on every rebalance** — `_execute_rebalance_orders`
    calls `self._risk.circuit_breaker(account_info)` before any order is
    submitted.  A tripped breaker logs a warning and returns immediately; no
    orders are placed.  The check uses the same `account_info` dict that was
    just fetched from Alpaca, so it reflects the current equity.

47. **Per-ticker error isolation in rebalance** — every order submission is
    wrapped in its own `try/except`.  A broker rejection or network error on
    one ticker logs an `engine.rebalance.order_failed` event and increments
    `n_errors`, but the loop continues to the next ticker.  A summary log
    (`engine.rebalance.summary`) is emitted at the end with `n_executed`,
    `n_skipped`, and `n_errors` counts.  Tests that verify isolation must check
    that `submit_order` was called N times even when one call raises.

48. **`run_pipeline` fetches news for all tickers in a single `fetch_news()` call** —
    articles are grouped by the `"ticker"` field already present in each dict from
    `_parse_feed`.  This keeps AV usage to 1 call per pipeline run instead of N.
    The `limit` param is set to `"200"` to accommodate the larger multi-ticker
    response.  Tests that previously used `fetch_news.side_effect` with one item
    per ticker must be updated to use `fetch_news.return_value` with a combined list.

49. **`_human_age()` formats headline age as a human-readable bracket prefix** —
    e.g. `[2 hours 30 minutes ago]`.  The `now` parameter exists for testability;
    production code omits it (defaults to `datetime.now(utc)`).  Tests must pass
    a fixed `now` to get deterministic assertions.  The function uses singular
    forms ("1 minute", "1 hour", "1 day") and omits the sub-unit when it is zero
    (e.g. `[1 hour ago]` not `[1 hour 0 minutes ago]`).

50. **Per-minute rate guard (`_enforce_per_minute_limit`) uses `time.monotonic()`
    timestamps** stored in `_recent_call_times`, pruned to a 60-second window.
    If 5+ calls exist in the window, it sleeps until the oldest entry expires.
    With the single-call-per-pipeline pattern, this should never trigger in normal
    operation — it is a safety net.  Tests must patch `trading_engine.data.alphavantage_client.time`
    (the module-level import) to control both `monotonic` and `sleep`.

51. **Sentiment scheduling uses two cron jobs** — `sentiment_job_early` (every
    25 min, 07:00–10:29 ET) and `sentiment_job_late` (every 35 min, 10:30–16:30 ET).
    Budget math: ~8 + ~10 = ~18 calls/day.  `sentiment_job()` has a soft budget
    check that skips the run if `get_daily_call_count() >= 20`; the hard limit
    (raises `RateLimitExceeded`) remains at 20 in `_check_and_increment()`.
    Total APScheduler jobs is now 4.  `sentiment_job()` no longer has its own
    market-hours guard — the cron windows handle that.  Tests must configure
    `engine._av_client.get_daily_call_count.return_value` to an int (default
    MagicMock comparison with `>= 20` is truthy and would skip the job).

52. **`hours_back` defaults to 2 (was 8)** — with runs every 25–35 minutes, the
    long lookback is unnecessary.  The in-process `_seen_hashes` dedup cache
    ensures headlines are not re-scored across overlapping windows.  On a fresh
    engine restart, `_seen_hashes` is empty so the first run may re-score recent
    headlines — this is acceptable and self-corrects after one cycle.
