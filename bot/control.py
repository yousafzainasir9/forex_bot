"""Runtime control flags shared between the bot and the web app.

The bot reads ``control.json`` each loop so the web app can flip the kill switch
live (block new entries) WITHOUT restarting the process. This is deliberately a
tiny JSON file in the log dir — no sockets, no shared memory, robust across
separate processes on the same machine.

Schema (all optional, sensible defaults):
  {
    "kill_switch": false,        # block all new orders when true
    "updated_utc": "..."         # last write time (informational)
  }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def control_path(log_dir) -> Path:
    return Path(log_dir) / "control.json"


def read_control(log_dir) -> Dict[str, Any]:
    """Return the control dict, defaulting to a safe (no-kill) state."""
    p = control_path(log_dir)
    default = {"kill_switch": False}
    if not p.exists():
        return default
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return default
        data.setdefault("kill_switch", False)
        return data
    except (json.JSONDecodeError, OSError):
        return default


def write_control(log_dir, **changes: Any) -> Dict[str, Any]:
    """Merge ``changes`` into control.json and return the new state."""
    p = control_path(log_dir)
    data = read_control(log_dir)
    data.update(changes)
    data["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def kill_active(log_dir) -> bool:
    """Convenience: True if the runtime kill switch is set."""
    return bool(read_control(log_dir).get("kill_switch", False))
