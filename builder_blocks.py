# cube_structure.py
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, ClassVar, Optional

from block import BuilderBlock, StageSpec
from registry import BLOCKS


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _as_str(v: Any, default: str) -> str:
    if v is None:
        return default
    return str(v)


def _parse_listish(v: Any) -> List[Dict[str, Any]]:
    """
    Accept:
      - list[dict]
      - JSON string
      - Python literal string (via ast.literal_eval)
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            pass
    return []


def _parse_map_lines(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        lines = [str(x).rstrip("\n") for x in v]
    else:
        s = str(v)
        lines = s.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _cubes_from_map(
    lines: List[str],
    *,
    cell: float,
    cube_size: float,
    origin: Tuple[float, float, float],
    rotate_with_scene: bool,
    max_cubes: int,
    z_negative: bool,
) -> List[Dict[str, Any]]:
    ox, oy, oz = origin
    out: List[Dict[str, Any]] = []
    z_sign = -1.0 if z_negative else 1.0

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch in (" ", ".", "\t"):
                continue

            # digit means "stack height"
            if ch.isdigit():
                h = max(1, min(9, int(ch)))
            else:
                h = 1

            x = ox + (c * cell)
            z = oz + (r * cell * z_sign)

            for k in range(h):
                y = oy + (k * cell)
                out.append(
                    {
                        "size": cube_size,
                        "pos_x": x,
                        "pos_y": y,
                        "pos_z": z,
                        "rotate_with_scene": rotate_with_scene,
                    }
                )
                if len(out) >= max_cubes:
                    return out
    return out


def _cubes_from_plane(
    *,
    cell: float,
    cube_size: float,
    origin: Tuple[float, float, float],
    rotate_with_scene: bool,
    max_cubes: int,
    z_negative: bool,
    # plane controls
    plane_count: int,
    plane_w: int,
    plane_d: int,
    plane_h: int,
    fill_partial: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a Minecraft-like rectangular floor/volume.

    If plane_w/plane_d are 0, we auto-pack plane_count into a near-square plane.
    plane_h controls vertical height (1 = flat plane).
    If fill_partial is True and plane_w*plane_d > plane_count, we only place plane_count cells.
    """
    ox, oy, oz = origin
    out: List[Dict[str, Any]] = []

    plane_count = max(0, int(plane_count))
    plane_h = max(1, int(plane_h))

    # Decide width/depth
    w = int(plane_w)
    d = int(plane_d)
    if w <= 0 or d <= 0:
        if plane_count <= 0:
            w, d = 1, 1
            plane_count = 1
        else:
            w = max(1, int(math.ceil(math.sqrt(plane_count))))
            d = max(1, int(math.ceil(plane_count / float(w))))
    else:
        w = max(1, w)
        d = max(1, d)
        if plane_count <= 0:
            plane_count = w * d

    z_sign = -1.0 if z_negative else 1.0

    placed_cells = 0
    max_cells = w * d

    # If fill_partial: only place `plane_count` cells on the XZ plane.
    # If not fill_partial: always fill full rectangle w*d (plane_count ignored).
    target_cells = min(plane_count, max_cells) if fill_partial else max_cells

    for zi in range(d):
        for xi in range(w):
            if placed_cells >= target_cells:
                break

            x = ox + (xi * cell)
            z = oz + (zi * cell * z_sign)

            for yi in range(plane_h):
                y = oy + (yi * cell)
                out.append(
                    {
                        "size": cube_size,
                        "pos_x": x,
                        "pos_y": y,
                        "pos_z": z,
                        "rotate_with_scene": rotate_with_scene,
                    }
                )
                if len(out) >= max_cubes:
                    meta = {"plane_w": w, "plane_d": d, "plane_h": plane_h, "cells": placed_cells + 1, "truncated": True}
                    return out, meta

            placed_cells += 1

        if placed_cells >= target_cells:
            break

    meta = {"plane_w": w, "plane_d": d, "plane_h": plane_h, "cells": placed_cells, "truncated": False}
    return out, meta


def _center_cubes(
    cubes: List[Dict[str, Any]],
    *,
    target: Tuple[float, float, float],
    center_y: bool,
) -> None:
    if not cubes:
        return

    tx, ty, tz = target

    min_x = 1e30
    min_y = 1e30
    min_z = 1e30
    max_x = -1e30
    max_y = -1e30
    max_z = -1e30

    for c in cubes:
        s = float(c.get("size", 1.0)) * 0.5
        x = float(c.get("pos_x", 0.0))
        y = float(c.get("pos_y", 0.0))
        z = float(c.get("pos_z", 0.0))

        min_x = min(min_x, x - s)
        max_x = max(max_x, x + s)
        min_y = min(min_y, y - s)
        max_y = max(max_y, y + s)
        min_z = min(min_z, z - s)
        max_z = max(max_z, z + s)

    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    cz = (min_z + max_z) * 0.5

    dx = cx - tx
    dz = cz - tz
    dy = (cy - ty) if center_y else 0.0

    for c in cubes:
        c["pos_x"] = float(c.get("pos_x", 0.0)) - dx
        c["pos_z"] = float(c.get("pos_z", 0.0)) - dz
        if center_y:
            c["pos_y"] = float(c.get("pos_y", 0.0)) - dy


@dataclass
class CubeBuilder(BuilderBlock):
    """
    CubeBuilder: expands a "level description" into MANY glcube stages.

    Modes:
      - mode="plane" (Minecraft-ish floor/volume)
      - mode="map"   (ASCII map, digits = stack height)
      - mode="cubes" (JSON / Python list[dict] of per-cube params)

    Intended pipeline:
      glctx | glprogram | glstate | cube_structure | glloop
    """

    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {
        # mode
        "mode": {"type": "enum", "default": "plane", "choices": ["plane", "map", "cubes"], "ui": "enum"},

        # --- PLANE mode (Minecraft-ish) ---
        # Option A: set plane_w & plane_d (and optionally plane_h)
        "plane_w": {"type": "int", "default": 10, "min": 1, "max": 500, "step": 1, "ui": "int"},
        "plane_d": {"type": "int", "default": 10, "min": 1, "max": 500, "step": 1, "ui": "int"},
        "plane_h": {"type": "int", "default": 1, "min": 1, "max": 200, "step": 1, "ui": "int"},
        # Option B: set plane_count and leave plane_w/plane_d <= 0 to auto-pack near-square
        "plane_count": {"type": "int", "default": 100, "min": 1, "max": 20000, "step": 1, "ui": "int"},
        "fill_partial": {"type": "bool", "default": False, "ui": "bool"},
        # If fill_partial=True and w*d > plane_count, we place only plane_count cells (useful for "N cubes").

        # --- MAP mode ---
        "map": {"type": "str", "default": "###\n#.#\n###", "ui": "text"},

        # --- CUBES mode (or override) ---
        "cubes": {"type": "str", "default": "", "ui": "text"},  # JSON/Python list[dict]

        # placement
        "cell": {"type": "float", "default": 1.0, "min": 0.05, "max": 25.0, "step": 0.05, "ui": "float"},
        "cube_size": {"type": "float", "default": 1.0, "min": 0.05, "max": 25.0, "step": 0.05, "ui": "float"},
        "origin_x": {"type": "float", "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "ui": "float"},
        "origin_y": {"type": "float", "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "ui": "float"},
        "origin_z": {"type": "float", "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "ui": "float"},
        "center": {"type": "bool", "default": True, "ui": "bool"},
        "center_y": {"type": "bool", "default": False, "ui": "bool"},
        "z_negative": {"type": "bool", "default": False, "ui": "bool"},
        "rotate_with_scene": {"type": "bool", "default": True, "ui": "bool"},

        # optional extras
        "include_axes": {"type": "bool", "default": False, "ui": "bool"},
        "axes_length": {"type": "float", "default": 2.0, "min": 0.1, "max": 100.0, "step": 0.1, "ui": "float"},

        # safety
        "max_cubes": {"type": "int", "default": 5000, "min": 1, "max": 20000, "step": 1, "ui": "int"},
        "alias_prefix": {"type": "str", "default": "cs", "ui": "text"},
    }

    def build_plan(
        self,
        payload: Any,
        *,
        params: Dict[str, Any],
    ) -> Tuple[List[StageSpec], Dict[str, Dict[str, Any]]]:
        mode = _as_str(params.get("mode"), "plane").strip().lower()
        if mode not in ("plane", "map", "cubes"):
            mode = "plane"

        cell = _as_float(params.get("cell", 1.0), 1.0)
        cube_size = _as_float(params.get("cube_size", 1.0), 1.0)
        ox = _as_float(params.get("origin_x", 0.0), 0.0)
        oy = _as_float(params.get("origin_y", 0.0), 0.0)
        oz = _as_float(params.get("origin_z", 0.0), 0.0)

        center = _as_bool(params.get("center", True), True)
        center_y = _as_bool(params.get("center_y", False), False)
        z_negative = _as_bool(params.get("z_negative", False), False)
        rotate_with_scene = _as_bool(params.get("rotate_with_scene", True), True)

        max_cubes = _as_int(params.get("max_cubes", 5000), 5000)
        max_cubes = max(1, min(20000, max_cubes))

        alias_prefix = str(params.get("alias_prefix", "cs") or "cs").strip() or "cs"
        alias_prefix = "".join([ch for ch in alias_prefix if ch.isalnum() or ch in ("_", "-")]) or "cs"

        # If cubes text is provided, we treat that as explicit override regardless of mode.
        cubes_override = _parse_listish(params.get("cubes"))

        meta_mode: Dict[str, Any] = {"mode": mode}
        cubes: List[Dict[str, Any]] = []

        if cubes_override:
            mode = "cubes"
            cubes = cubes_override
            meta_mode["mode"] = "cubes"

        if mode == "plane":
            plane_w = _as_int(params.get("plane_w", 10), 10)
            plane_d = _as_int(params.get("plane_d", 10), 10)
            plane_h = _as_int(params.get("plane_h", 1), 1)
            plane_count = _as_int(params.get("plane_count", plane_w * plane_d), plane_w * plane_d)
            fill_partial = _as_bool(params.get("fill_partial", False), False)

            # If user wants "pick number of cubes", set plane_w or plane_d <= 0 and fill_partial=True.
            # But we also support: w*d rectangle always.
            cubes, plane_meta = _cubes_from_plane(
                cell=cell,
                cube_size=cube_size,
                origin=(ox, oy, oz),
                rotate_with_scene=rotate_with_scene,
                max_cubes=max_cubes,
                z_negative=z_negative,
                plane_count=plane_count,
                plane_w=plane_w,
                plane_d=plane_d,
                plane_h=plane_h,
                fill_partial=fill_partial,
            )
            meta_mode.update(plane_meta)

        elif mode == "map":
            lines = _parse_map_lines(params.get("map"))
            cubes = _cubes_from_map(
                lines,
                cell=cell,
                cube_size=cube_size,
                origin=(ox, oy, oz),
                rotate_with_scene=rotate_with_scene,
                max_cubes=max_cubes,
                z_negative=z_negative,
            )

        elif mode == "cubes":
            cubes = cubes_override

        # normalize cube dicts + defaults
        norm: List[Dict[str, Any]] = []
        for c in cubes[:max_cubes]:
            if not isinstance(c, dict):
                continue
            norm.append(
                {
                    "size": _as_float(c.get("size", cube_size), cube_size),
                    "pos_x": _as_float(c.get("pos_x", ox), ox),
                    "pos_y": _as_float(c.get("pos_y", oy), oy),
                    "pos_z": _as_float(c.get("pos_z", oz), oz),
                    "rotate_with_scene": _as_bool(c.get("rotate_with_scene", rotate_with_scene), rotate_with_scene),
                }
            )

        if not norm:
            norm = [{
                "size": cube_size,
                "pos_x": ox,
                "pos_y": oy,
                "pos_z": oz,
                "rotate_with_scene": rotate_with_scene,
            }]

        if center:
            _center_cubes(norm, target=(ox, oy, oz), center_y=center_y)

        # stages: optional axes + many glcube aliases
        stages: List[StageSpec] = []

        if _as_bool(params.get("include_axes", False), False):
            stages.append(
                (
                    f"glaxes:{alias_prefix}_axes",
                    {
                        "length": _as_float(params.get("axes_length", 2.0), 2.0),
                        "rotate_with_scene": False,
                        "pos_x": ox,
                        "pos_y": oy,
                        "pos_z": oz,
                    },
                )
            )

        for i, c in enumerate(norm[:max_cubes]):
            stages.append(
                (
                    f"glcube:{alias_prefix}_{i}",
                    {
                        "size": c["size"],
                        "pos_x": c["pos_x"],
                        "pos_y": c["pos_y"],
                        "pos_z": c["pos_z"],
                        "rotate_with_scene": c["rotate_with_scene"],
                    },
                )
            )

        extras: Dict[str, Dict[str, Any]] = {}
        return stages, extras

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        out, meta = super().execute(payload, params=params)
        meta = dict(meta or {})
        chain = meta.get("chain") or []
        if isinstance(chain, list):
            meta["cube_count"] = sum(1 for m in chain if isinstance(m, dict) and m.get("base") == "glcube")
            meta["added_axes"] = any(isinstance(m, dict) and m.get("base") == "glaxes" for m in chain)
        return out, meta


BLOCKS.register("cube_builder", CubeBuilder)
