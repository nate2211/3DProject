from __future__ import annotations

import json
from typing import Any, Dict, List


def _coerce(v: str) -> Any:
    v = v.strip()
    low = v.lower()

    if low in ("true", "false"):
        return low == "true"

    # try JSON objects/arrays
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return json.loads(v)
        except Exception:
            pass

    # int
    try:
        if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            return int(v)
    except Exception:
        pass

    # float
    try:
        return float(v)
    except Exception:
        pass

    # quoted strings
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        return v[1:-1]

    return v


def parse_extras(items: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Parse repeated --extra key=val items.

    Supports:
      --extra all.foo=1
      --extra glctx.w=1280
      --extra stages="glctx|glprogram|glcube|glloop"   (treated as all.stages)

    Returns:
      {"all": {...}, "glctx": {...}, "glloop": {...}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for it in items or []:
        if "=" not in it:
            continue
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()

        if "." in k:
            group, key = k.split(".", 1)
        else:
            group, key = "all", k

        group = group.strip().lower()
        key = key.strip()

        out.setdefault(group, {})[key] = _coerce(v)
    return out
