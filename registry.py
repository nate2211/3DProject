from __future__ import annotations

import sys
from typing import Any, Dict, List, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from block import BaseBlock


class Registry:
    """Central registry for Block implementations."""
    def __init__(self) -> None:
        self._by_name: Dict[str, type["BaseBlock"]] = {}

    def register(self, name: str, cls: type["BaseBlock"]) -> None:
        key = name.strip().lower()
        if key in self._by_name:
            print(f"[Registry] Warning: Overwriting block '{key}'", file=sys.stderr)
        self._by_name[key] = cls

    def names(self) -> List[str]:
        return sorted(self._by_name.keys())

    def create(self, name: str, **kwargs: Any) -> "BaseBlock":
        key = name.strip().lower()
        if key not in self._by_name:
            raise KeyError(f"Unknown block '{name}'. Available: {', '.join(self.names()) or '(none)'}")
        return self._by_name[key](**kwargs)

    # ---- NEW: Introspection for GUI ----
    def get_class(self, name: str) -> Optional[type["BaseBlock"]]:
        """Return the class registered under 'name' (lowercased) or None."""
        return self._by_name.get(name.strip().lower())

    def items(self) -> List[tuple[str, type["BaseBlock"]]]:
        """List of (name, class) for registered blocks."""
        return sorted(self._by_name.items(), key=lambda x: x[0])


BLOCKS = Registry()
