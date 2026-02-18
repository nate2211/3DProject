from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional
import os
import sys
import tempfile
from io import BytesIO

from PIL import Image
from block import BaseBlock
from registry import BLOCKS

try:
    import requests
except Exception:
    requests = None


def _parse_all_extras_from_argv() -> Dict[str, Dict[str, Any]]:
    extras_raw: List[str] = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--extra":
            if i + 1 >= len(argv):
                break
            extras_raw.append(argv[i + 1])
            i += 2
        else:
            i += 1

    def _coerce(v: str) -> Any:
        v = v.strip()
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
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

    out: Dict[str, Dict[str, Any]] = {}
    for item in extras_raw:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        if "." in k:
            group, key = k.split(".", 1)
        else:
            group, key = "all", k
        group = group.strip().lower()
        key = key.strip()
        out.setdefault(group, {})[key] = _coerce(v)
    return out


def _default_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }


def _fetch_image_to_temp(url: str, *, timeout: float = 15.0, headers: Optional[Dict[str, str]] = None) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed. Try: pip install requests")

    hdrs = _default_headers()
    if headers and isinstance(headers, dict):
        hdrs.update(headers)

    resp = requests.get(url, timeout=timeout, headers=hdrs, allow_redirects=True)
    resp.raise_for_status()
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if not ctype.startswith("image/"):
        raise ValueError(f"URL did not return an image (Content-Type='{ctype or 'unknown'}').")

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(tmp_path)
    return tmp_path


def _stage_id_and_base(stage: str) -> Tuple[str, str]:
    s = stage.strip()
    stage_id = s.lower()
    base = s.split(":", 1)[0].strip().lower()
    return stage_id, base


@dataclass
class PipelineBlock(BaseBlock):
    _temp_paths: List[str] = field(default_factory=list)

    def _stage_params(self, stage_id: str, base: str, global_extras: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge order (later wins):
          all.* -> base.* -> stage_id.*
        So glcube.* can set shared defaults, while glcube:c1.* overrides per-instance.
        """
        merged: Dict[str, Any] = {}
        merged.update(global_extras.get("all", {}))
        merged.update(global_extras.get(base, {}))
        merged.update(global_extras.get(stage_id, {}))
        return merged

    def _run_stage(self, stage_token: str, payload: Any, global_extras: Dict[str, Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        stage_id, base = _stage_id_and_base(stage_token)
        params = self._stage_params(stage_id, base, global_extras)

        # built-in adapter
        if base in ("imageurl", "image_url"):
            url = str(payload).strip()
            timeout = float(params.get("timeout", 15.0))
            headers = params.get("headers", None)
            if headers is not None and not isinstance(headers, dict):
                try:
                    import ast
                    headers = ast.literal_eval(str(headers))
                except Exception:
                    headers = None
            path = _fetch_image_to_temp(url, timeout=timeout, headers=headers)
            self._temp_paths.append(path)
            return path, {"stage": stage_id, "base": base, "emitted": "path", "path": path}

        # normal block: create from BASE, not stage_id
        try:
            blk = BLOCKS.create(base)
        except KeyError:
            raise KeyError(f"Unknown stage '{base}'. Available: {', '.join(BLOCKS.names()) or '(none)'}")

        # pass instance identifiers in params (optional use by blocks)
        params = dict(params)
        params["_stage_id"] = stage_id
        params["_stage_base"] = base

        result, meta = blk.execute(payload, params=params)
        out_meta = {"stage": stage_id, "base": base}
        if meta and isinstance(meta, dict):
            out_meta.update(meta)
        return result, out_meta

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        stages_str = params.get("stages") or params.get("pipeline") or params.get("pipe")
        if not stages_str or not isinstance(stages_str, str):
            raise ValueError("Missing pipeline.stages. Example: --extra pipeline.stages='glctx|glcube|glloop'")

        stages = [s for s in stages_str.split("|") if s.strip()]
        if not stages:
            raise ValueError("Empty pipeline.stages.")

        all_extras = _parse_all_extras_from_argv()

        meta_chain: List[Dict[str, Any]] = []
        current = payload

        try:
            for st in stages:
                current, m = self._run_stage(st, current, all_extras)
                meta_chain.append(m)
        finally:
            for p in self._temp_paths:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

        out = current if isinstance(current, str) else str(current)
        return out, {"type": "pipeline", "stages": stages, "chain": meta_chain}


BLOCKS.register("pipeline", PipelineBlock)
