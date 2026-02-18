# builder_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from registry import BLOCKS
@dataclass
class BaseBlock:
    """Abstract base class for all blocks."""
    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError

StageSpec = Union[
    str,                         # "glcube:c1"
    Tuple[str, Dict[str, Any]],   # ("glcube:c1", {"size": 2.0})
    Dict[str, Any],               # {"stage": "glcube:c1", "params": {...}}
]


def _stage_id_and_base(stage: str) -> Tuple[str, str]:
    s = (stage or "").strip()
    stage_id = s.lower()
    base = s.split(":", 1)[0].strip().lower()
    return stage_id, base


def _normalize_stage_spec(spec: StageSpec) -> Tuple[str, Dict[str, Any]]:
    if isinstance(spec, str):
        return spec.strip(), {}
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str) and isinstance(spec[1], dict):
        return spec[0].strip(), dict(spec[1])
    if isinstance(spec, dict):
        tok = (spec.get("stage") or spec.get("token") or "").strip()
        prm = spec.get("params") or {}
        return tok, dict(prm) if isinstance(prm, dict) else {}
    raise TypeError(f"Bad stage spec: {type(spec)}")


@dataclass
class BuilderBlock(BaseBlock, ABC):
    """
    Abstract BuilderBlock base class.

    Subclasses decide HOW to obtain stages/extras (GUI, CLI, file, database, etc.)
    by implementing build_plan().

    - build_plan(payload, params) -> (stages, extras)
    - pipeline(...) runs the plan using BLOCKS registry.
    """

    @abstractmethod
    def build_plan(
        self,
        payload: Any,
        *,
        params: Dict[str, Any],
    ) -> Tuple[List[StageSpec], Dict[str, Dict[str, Any]]]:
        """
        Return (stages, extras).

        extras is a dict-of-dicts merged as:
          all.* -> base.* -> stage_id.* -> local stage params
        """
        raise NotImplementedError

    # ---- shared implementation (non-abstract) ----

    def _merge_params(
        self,
        stage_id: str,
        base: str,
        *,
        extras: Dict[str, Dict[str, Any]],
        local_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        merged.update(extras.get("all", {}) or {})
        merged.update(extras.get(base, {}) or {})
        merged.update(extras.get(stage_id, {}) or {})
        merged.update(local_params or {})
        return merged

    def _run_stage(
        self,
        stage_token: str,
        payload: Any,
        *,
        extras: Dict[str, Dict[str, Any]],
        local_params: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        stage_id, base = _stage_id_and_base(stage_token)
        params = self._merge_params(stage_id, base, extras=extras, local_params=local_params)

        try:
            blk = BLOCKS.create(base)  # instantiate by BASE
        except KeyError:
            raise KeyError(f"Unknown stage '{base}'. Available: {', '.join(BLOCKS.names()) or '(none)'}")

        # optional identifiers for downstream blocks
        params = dict(params)
        params["_stage_id"] = stage_id
        params["_stage_base"] = base

        result, meta = blk.execute(payload, params=params)
        out_meta: Dict[str, Any] = {"stage": stage_id, "base": base}
        if isinstance(meta, dict):
            out_meta.update(meta)
        return result, out_meta

    def pipeline(
        self,
        payload: Any,
        stages: List[StageSpec],
        *,
        extras: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Instance pipeline runner.
        """
        extras2: Dict[str, Dict[str, Any]] = dict(extras or {})
        chain: List[Dict[str, Any]] = []
        cur = payload

        for spec in stages:
            tok, local = _normalize_stage_spec(spec)
            if not tok:
                continue
            cur, m = self._run_stage(tok, cur, extras=extras2, local_params=local)
            chain.append(m)

        return cur, {"type": "builder", "stages": [(_normalize_stage_spec(s)[0]) for s in stages], "chain": chain}

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        stages, extras = self.build_plan(payload, params=params)
        out, meta = self.pipeline(payload, stages, extras=extras)
        return out, meta