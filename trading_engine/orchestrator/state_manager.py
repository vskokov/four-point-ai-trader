"""
Engine state persistence.

StateManager saves and restores all engine metadata atomically, with SHA-256
checksum validation and a rolling window of the last 3 snapshots.

Model weights and fitted objects (HMM, Kalman) are persisted by their own
modules (joblib / numpy).  This file stores the lightweight metadata that
tracks which models are fitted, current signal statistics, etc.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_STATE_DIR = Path(__file__).parent.parent / "models"
_STATE_FILENAME = "engine_state.json"
_MAX_BACKUPS = 3
_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """
    Atomic save / load of engine metadata with checksum validation.

    Layout
    ------
    ``{state_dir}/engine_state.json``        — current state
    ``{state_dir}/engine_state.json.bak1``   — one snapshot behind
    ``{state_dir}/engine_state.json.bak2``   — two snapshots behind
    ``{state_dir}/engine_state.json.bak3``   — three snapshots behind

    Parameters
    ----------
    state_dir:
        Directory where the state file is written.  Defaults to
        ``trading_engine/models/``.  Pass ``tmp_path`` in tests.
    """

    def __init__(self, state_dir: Path | str | None = None) -> None:
        self._state_dir = Path(state_dir) if state_dir is not None else _DEFAULT_STATE_DIR

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return self._state_dir / _STATE_FILENAME

    def _backup_path(self, n: int) -> Path:
        return self._state_dir / f"{_STATE_FILENAME}.bak{n}"

    # ------------------------------------------------------------------
    # Checksum helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checksum(payload: dict[str, Any]) -> str:
        """SHA-256 of the JSON-serialised payload (checksum key excluded)."""
        body = {k: v for k, v in payload.items() if k != "checksum"}
        raw = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _verify(data: dict[str, Any]) -> bool:
        """Return True if the stored checksum matches the recomputed one."""
        stored = data.get("checksum", "")
        expected = StateManager._checksum(data)
        return stored == expected

    # ------------------------------------------------------------------
    # Rolling backup
    # ------------------------------------------------------------------

    def _rotate_backups(self) -> None:
        """
        Shift existing backups: bak2 → bak3, bak1 → bak2, current → bak1.
        The oldest backup (beyond _MAX_BACKUPS) is deleted.
        """
        current = self._state_path()
        if not current.exists():
            return

        # Drop the oldest backup beyond the window
        oldest = self._backup_path(_MAX_BACKUPS)
        if oldest.exists():
            oldest.unlink()

        # Shift bak(n-1) → bak(n)
        for n in range(_MAX_BACKUPS - 1, 0, -1):
            src = self._backup_path(n)
            dst = self._backup_path(n + 1)
            if src.exists():
                shutil.copy2(src, dst)

        # current → bak1
        shutil.copy2(current, self._backup_path(1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, state: dict[str, Any]) -> None:
        """
        Atomically save *state* to disk.

        The dict is extended with ``version``, ``saved_at``, and
        ``checksum`` fields.  Existing snapshots are rotated before writing.

        Parameters
        ----------
        state:
            Serialisable dict (must not contain a ``checksum`` key).
        """
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._rotate_backups()

        payload: dict[str, Any] = {
            "version":  _FORMAT_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **state,
        }
        payload["checksum"] = self._checksum(payload)

        # Write atomically via a tmp file + rename
        tmp = self._state_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._state_path())

        logger.info(
            "state_manager.saved",
            path=str(self._state_path()),
            tickers=state.get("tickers"),
        )

    def load(self) -> dict[str, Any] | None:
        """
        Load and validate the current state snapshot.

        Returns
        -------
        dict
            The stored state (without ``checksum`` and ``version`` keys), or
            *None* if the file does not exist.

        Raises
        ------
        ValueError
            If the checksum is invalid (file is corrupted).
        """
        path = self._state_path()
        if not path.exists():
            logger.info("state_manager.no_state_file", path=str(path))
            return None

        raw = path.read_text(encoding="utf-8")
        data: dict[str, Any] = json.loads(raw)

        if not self._verify(data):
            logger.error(
                "state_manager.checksum_mismatch",
                path=str(path),
                stored=data.get("checksum"),
            )
            raise ValueError(f"State file checksum mismatch: {path}")

        logger.info(
            "state_manager.loaded",
            path=str(path),
            version=data.get("version"),
            saved_at=data.get("saved_at"),
        )
        # Return without internal metadata
        return {
            k: v
            for k, v in data.items()
            if k not in {"checksum", "version", "saved_at"}
        }

    def list_backups(self) -> list[Path]:
        """Return existing backup paths in order (bak1 = most recent)."""
        return [
            p
            for n in range(1, _MAX_BACKUPS + 1)
            if (p := self._backup_path(n)).exists()
        ]
