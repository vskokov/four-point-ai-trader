# Adversarial Review Concerns and Fix Plan

This document captures the main concerns identified during an adversarial review
of the repository, along with a concrete implementation prompt and a detailed
fix plan for each issue.

The goal is not just to patch symptoms, but to make the behavior safer and more
observable under real trading conditions, partial outages, and multi-module
interactions.

## Review status after remediation

Status from follow-up review of the current code:

| Concern                                       | Status    | Notes                                                                                                                                                   |
| --------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Duplicate live-bar persistence                | Addressed | `stream_bars()` no longer writes live bars; `bar_handler()` remains the writer; `ohlcv` now also has duplicate cleanup + unique index defense-in-depth. |
| Cross-ticker news dedup corruption            | Addressed | Dedup cache is now keyed by `(ticker, headline_hash)` and DB uniqueness is now `(ticker, headline_hash)`.                                               |
| MWU penalized on missing outcome data         | Addressed | `get_actual_direction()` now returns `None` for unavailable data and `scheduled_update()` skips weight updates in that case.                            |
| Regime smoothing not affecting MWU regime row | Addressed | Smoothed regime index is now tracked and passed into MWU.                                                                                               |
| Rolling backups unused for recovery           | Addressed | `StateManager.load()` now falls back through backup files.                                                                                              |

Follow-up notes from the review:

- No new high-severity regressions were identified in the remediation itself.
- Relevant unit tests passed in this session for `test_alpaca_client.py`,
  `test_engine.py`, `test_mwu_agent.py`, and `test_llm_sentiment.py`.
- `test_storage.py` was skipped because `TEST_DB_URL` was not set, so the DB
  migration path for the new indexes/constraints is still only partially
  validated in this session.
- One minor follow-up remains worth considering: `Storage.insert_ohlcv()` still
  returns `len(rows)` even though `ON CONFLICT DO NOTHING` can now skip rows due
  to the new unique index, so its returned count can overstate actual inserts.

---

## Recommended implementation prompt

Use the following prompt when implementing the fixes:

> Perform a focused remediation of the adversarial review findings in this repository.
> Fix the issues without changing unrelated behavior:
>
> 1. Prevent duplicate OHLCV inserts for live streamed bars.
> 2. Correct shared-news handling so the same article can be persisted and scored per ticker without cross-ticker dedup corruption.
> 3. Prevent MWU weights from being penalized when price outcome data is unavailable or insufficient.
> 4. Make regime smoothing affect the regime row used by MWU, not just the displayed label/signal.
> 5. Add backup-based recovery for engine state corruption so valid backups are automatically tried before falling back to defaults.
>
> Requirements:
>
> - Reuse existing patterns and helpers.
> - Preserve current public behavior where it is not directly implicated.
> - Add or update tests that prove each bug is fixed.
> - Do not introduce silent fallbacks that hide unsafe states.
> - Keep changes surgical and production-safe.
>
> Validation:
>
> - Run the relevant existing unit tests before and after changes.
> - Add targeted tests for the newly covered failure modes.
> - Summarize any intentional behavior changes.

---

## 1. Duplicate live-bar persistence

### Concern

Live bars appear to be persisted twice:

- `trading_engine/data/alpaca_client.py` in `AlpacaMarketData.stream_bars()`
- `trading_engine/orchestrator/engine.py` in `TradingEngine.bar_handler()`

`ohlcv` currently has no uniqueness constraint that would suppress duplicate
writes, so the same live bar can be stored twice.

### Why this matters

This is a high-severity data integrity issue.

Duplicated bars can distort:

- HMM training and regime inference
- OU spread calculations
- outcome labeling and analytics
- backtests or any downstream metric built on `ohlcv`

Because the duplication happens in a normal success path, it is especially
dangerous: the system may look healthy while its market data slowly diverges
from reality.

### Evidence

- `trading_engine/data/alpaca_client.py`: stream handler persists the bar before
  invoking the callback.
- `trading_engine/orchestrator/engine.py`: `bar_handler()` persists the same bar
  again as its first step.
- `trading_engine/data/storage.py`: `ohlcv` has no unique constraint on
  `(ticker, time)`.

### Fix plan

Choose one authoritative insertion point for live bars and remove the other.

Recommended approach:

1. Make `bar_handler()` the single writer for live-bar persistence.
2. Remove the `insert_ohlcv()` call from the live stream callback in
   `AlpacaMarketData.stream_bars()`.
3. Keep historical fetch persistence unchanged.
4. Update tests so they verify:
   - stream callbacks forward bar payloads without writing to storage
   - `bar_handler()` persists once
   - no double insert happens in the live path

### Test additions

- Add a unit test for `stream_bars()` proving the callback receives the bar but
  storage is not written there.
- Add or update an engine test asserting one insert per live bar end-to-end.

---

## 2. Cross-ticker news dedup corruption for shared headlines

### Concern

The same article can be associated with multiple tickers by Alpaca news, but the
LLM pipeline deduplicates only by `headline_hash`, and storage also enforces a
global unique constraint on `headline_hash`.

This creates a mismatch:

- Alpaca expands one article into multiple `(ticker, article)` rows
- pipeline dedup cache collapses them by `headline_hash`
- storage also collapses them globally by `headline_hash`

Result: shared articles can be scored or persisted for the first ticker only,
while later tickers silently lose them.

### Why this matters

This is a high-severity signal integrity issue.

It can systematically bias sentiment coverage for tickers that share macro,
sector, ETF, M&A, legal, or earnings-related headlines. The engine may therefore
under-score or neutralize valid sentiment for affected symbols.

### Evidence

- `trading_engine/data/alpaca_client.py`: Alpaca news fan-outs a shared article
  to multiple matched tickers.
- `trading_engine/signals/llm_sentiment.py`: `_seen_hashes` and per-run dedup
  use only `headline_hash`.
- `trading_engine/data/storage.py`: `news.headline_hash` is globally unique.

### Fix plan

Make dedup semantics explicitly ticker-aware where the logical unit is
`(ticker, article)`.

Recommended approach:

1. Decide the intended storage model:
   - If `news` is per-ticker news, uniqueness should be `(ticker, headline_hash)`.
   - If `news` is global article storage, then ticker linkage should live in a
     separate mapping table.

For this codebase, the least disruptive fix is likely:

2. Change `news` uniqueness from global `headline_hash` to a composite uniqueness
   on `(ticker, headline_hash)`.
3. Update `insert_news()` so duplicate handling follows that composite key.
4. Change in-process LLM dedup from:
   - `headline_hash`
     to:
   - `(ticker, headline_hash)`
5. Ensure prompt-level dedup inside a single ticker still removes repeated
   copies of the same article for that ticker.
6. Preserve one LLM call per ticker in `run_pipeline()`.

### Test additions

- Add a test where one article is returned for both `AAPL` and `MSFT`; both
  tickers should receive it and both should be scored.
- Add a storage test verifying the same `headline_hash` can be inserted once per
  ticker.
- Add a dedup-cache test proving that processing `AAPL` does not suppress the
  same shared article for `MSFT`.

### Migration note

If the DB schema already exists in a real environment, this likely requires a
schema migration rather than only DDL bootstrap changes.

---

## 3. MWU weights are penalized when outcome data is missing

### Concern

`MWUMetaAgent.get_actual_direction()` returns `0` both for a real flat move and
for error-like states such as:

- no storage
- empty data
- insufficient bars

`update_weights()` then treats any non-zero signal against `actual_direction=0`
as wrong, causing the agent to penalize signals even when it had no valid market
outcome to learn from.

### Why this matters

This is a high-severity online-learning integrity issue.

During data gaps, DB outages, or delayed bar availability, the ensemble can
learn incorrect lessons and gradually degrade its weights. That can make the
system less reliable exactly when infrastructure is unstable.

### Evidence

- `trading_engine/meta_agent/mwu_agent.py`: `get_actual_direction()` returns `0`
  in true-flat and no-data cases.
- `trading_engine/meta_agent/mwu_agent.py`: `update_weights()` interprets
  `actual_direction=0` as a real outcome and penalizes directional signals.

### Fix plan

Separate “neutral market outcome” from “unknown outcome”.

Recommended approach:

1. Change `get_actual_direction()` to return a tri-state result such as:
   - `1`, `0`, `-1` for known outcomes
   - `None` for unavailable outcome data
2. Update `scheduled_update()` so it skips `update_weights()` when the outcome is
   unknown.
3. Log a clear informational or warning event when a pending decision expires
   without enough data to evaluate.
4. Keep real flat moves (`0`) as valid learnable outcomes.
5. Ensure pending entries are either:
   - removed with an explicit “skipped due to missing outcome data” log, or
   - retained temporarily if the design prefers another evaluation attempt

The simpler and safer option is to skip the update and remove the pending record
with a log entry.

### Test additions

- Add a test proving that empty or insufficient OHLCV data produces an unknown
  outcome, not a penalizing neutral.
- Add a test proving `scheduled_update()` does not call `update_weights()` when
  outcome data is unavailable.
- Add a test preserving current behavior for genuine flat outcomes.

---

## 4. Regime smoothing does not affect the MWU regime row

### Concern

The engine smooths the HMM label and converts that into a smoothed HMM signal,
but still passes the raw unsmoothed integer `regime` into the MWU meta-agent.

That means:

- the visible HMM label can say “bull”
- the HMM signal sent to MWU can be smoothed to bull
- but MWU still indexes weights using the raw regime, such as bear

### Why this matters

This is a medium-severity logic consistency bug.

It breaks the contract implied by regime smoothing. The system appears to use a
stable regime while the meta-agent can still update or read from a different
regime row.

### Evidence

- `trading_engine/orchestrator/engine.py`: smoothing modifies label and HMM
  signal only.
- `trading_engine/orchestrator/engine.py`: `scheduled_update(..., regime=regime)`
  still passes the original raw regime index.

### Fix plan

Make smoothing produce both:

- a stable label
- a stable regime index

Recommended approach:

1. Extend the smoothing logic so the stable state is tracked as both label and
   underlying regime index.
2. When smoothing suppresses a raw label change, pass the stable regime index
   into MWU.
3. Ensure `trade_log` records remain internally consistent:
   - if raw regime is important for debugging, log both raw and stable values
   - if only one regime is stored, store the one actually used for decisioning

Least disruptive option:

4. Use the smoothed regime index for MWU and trade logging, since that is the
   effective decision regime.

### Test additions

- Add a test asserting the `regime` argument passed to `scheduled_update()` is
  the smoothed regime, not just the smoothed signal.
- Add a test covering a one-bar outlier where stable label and stable regime row
  must both remain unchanged.

---

## 5. Rolling backups exist but are not used for recovery

### Concern

`StateManager` rotates and retains backup files, but `load()` reads only the
current state file. If the current file is corrupted, the engine logs the error
and falls back to defaults without attempting valid backups.

### Why this matters

This is a medium-severity resilience issue.

The system already pays the complexity cost of backup rotation, but does not
receive the recovery benefit. A single corrupt current file can discard useful
state even when `bak1` is valid.

### Evidence

- `trading_engine/orchestrator/state_manager.py`: backup rotation is implemented.
- `trading_engine/orchestrator/state_manager.py`: `load()` checks only the
  current file.
- `trading_engine/orchestrator/engine.py`: `_load_state()` catches failure and
  returns to defaults.

### Fix plan

Teach state loading to try the current file first, then fall back through
backups in order.

Recommended approach:

1. Refactor state reading into a helper that can validate an arbitrary path.
2. Make `load()` try:
   - current
   - `bak1`
   - `bak2`
   - `bak3`
3. On successful backup recovery:
   - log which file was used
   - optionally restore/copy it back to the current path
4. If all files fail validation, return `None` or raise a clear corruption
   exception depending on the existing recovery contract.

Given the current engine flow, the safest fit is:

5. Return `None` only after all candidates fail, with strong logging for each
   failure and for the final fallback.

### Test additions

- Add a test where current state is corrupted but `bak1` is valid; load should
  recover successfully.
- Add a test where current and `bak1` are corrupted but `bak2` is valid.
- Add a test where all files are invalid and the system falls back cleanly.

---

## Suggested implementation order

To minimize risk and reduce debugging churn, implement in this order:

1. **Duplicate live-bar persistence**
   - Smallest surface area
   - Highest data-integrity impact

2. **MWU missing-outcome handling**
   - Prevents silent model degradation during failures

3. **Regime smoothing consistency**
   - Keeps decision logic coherent

4. **Backup-based recovery**
   - Improves resilience without affecting trading math

5. **Cross-ticker news dedup/storage fix**
   - Highest schema impact
   - May require migration handling and broader test updates

---

## Validation plan

After implementation, run at minimum:

```bash
cd trading_engine
.venv/bin/pytest tests/meta_agent/test_mwu_agent.py tests/test_engine.py tests/test_llm_sentiment.py tests/execution/test_executor.py tests/test_alpaca_client.py -v
```

If storage schema changes are made, also run:

```bash
cd trading_engine
TEST_DB_URL="postgresql+psycopg2://trader:traderpass@localhost:5432/trading" \
    .venv/bin/pytest tests/test_storage.py -v
```

---

## Expected outcome after remediation

After all fixes are applied:

- live market bars should be persisted exactly once
- shared articles should remain usable per ticker
- MWU should learn only from real outcomes, not missing data
- smoothed regimes should affect both the displayed signal and the regime row
  used by MWU
- state corruption should recover from backups automatically when possible

That would materially improve data integrity, signal correctness, learning
stability, and operational resilience without changing the broader architecture.
