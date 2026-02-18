from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, ClassVar
import ctypes
import math
import time

import pygame
from OpenGL import GL as gl

from block import BaseBlock
from registry import BLOCKS


# ---------------------------
# Small math helpers (mat4) column-major
# ---------------------------
def mat4_identity() -> List[float]:
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]


def mat4_copy(m: List[float]) -> List[float]:
    return list(m)


def mat4_mul(a: List[float], b: List[float]) -> List[float]:
    # c = a * b (column-major)
    c = [0.0] * 16
    for col in range(4):
        for row in range(4):
            c[col * 4 + row] = (
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3]
            )
    return c


def mat4_translate(x: float, y: float, z: float) -> List[float]:
    m = mat4_identity()
    m[12] = x
    m[13] = y
    m[14] = z
    return m


def mat4_rotate_y(a: float) -> List[float]:
    c = math.cos(a)
    s = math.sin(a)
    return [
        c,  0, -s, 0,
        0,  1,  0, 0,
        s,  0,  c, 0,
        0,  0,  0, 1,
    ]


def mat4_rotate_x(a: float) -> List[float]:
    c = math.cos(a)
    s = math.sin(a)
    return [
        1, 0,  0, 0,
        0, c,  s, 0,
        0, -s, c, 0,
        0, 0,  0, 1,
    ]


def mat4_perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> List[float]:
    fovy = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy / 2.0)
    nf = 1.0 / (znear - zfar)
    return [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (zfar + znear) * nf, -1,
        0, 0, (2 * zfar * znear) * nf, 0,
    ]


# ---------------------------
# small coercions
# ---------------------------
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


# ---------------------------
# shader helpers
# ---------------------------
def _compile_shader(src: str, shader_type: int) -> int:
    sid = gl.glCreateShader(shader_type)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)

    ok = gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS)
    if not ok:
        log = gl.glGetShaderInfoLog(sid).decode("utf-8", "replace")
        gl.glDeleteShader(sid)
        raise RuntimeError(f"Shader compile failed:\n{log}")
    return sid


def _link_program(vs: int, fs: int) -> int:
    pid = gl.glCreateProgram()
    gl.glAttachShader(pid, vs)
    gl.glAttachShader(pid, fs)
    gl.glLinkProgram(pid)

    ok = gl.glGetProgramiv(pid, gl.GL_LINK_STATUS)
    if not ok:
        log = gl.glGetProgramInfoLog(pid).decode("utf-8", "replace")
        gl.glDeleteProgram(pid)
        raise RuntimeError(f"Program link failed:\n{log}")

    gl.glDetachShader(pid, vs)
    gl.glDetachShader(pid, fs)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return pid


def _ensure_ctx(payload: Any, who: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{who} expects ctx dict from glctx")
    return payload


# ---------------------------
# Block: GL context (pygame)
# ---------------------------
@dataclass
class GLContextBlock(BaseBlock):
    """
    Creates pygame window + OpenGL context.
    Emits ctx dict used by later stages.
    """
    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "w": {"type": "int", "default": 1280, "min": 320, "max": 7680, "step": 1, "ui": "int"},
        "h": {"type": "int", "default": 720, "min": 240, "max": 4320, "step": 1, "ui": "int"},
        "title": {"type": "str", "default": "Nate's 3D Renderer", "ui": "text"},
        "fps": {"type": "int", "default": 60, "min": 1, "max": 1000, "step": 1, "ui": "int"},
        "major": {"type": "int", "default": 3, "min": 2, "max": 4, "step": 1, "ui": "int"},
        "minor": {"type": "int", "default": 3, "min": 0, "max": 6, "step": 1, "ui": "int"},
        "vsync": {"type": "bool", "default": True, "ui": "bool"},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        w = _as_int(params.get("w", 1280), 1280)
        h = _as_int(params.get("h", 720), 720)
        title = str(params.get("title", "Nate's 3D Renderer"))
        major = _as_int(params.get("major", 3), 3)
        minor = _as_int(params.get("minor", 3), 3)
        fps = _as_int(params.get("fps", 60), 60)
        vsync = _as_bool(params.get("vsync", True), True)

        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, major)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, minor)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        except Exception:
            pass

        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE

        try:
            pygame.display.set_mode((w, h), flags, vsync=1 if vsync else 0)
        except Exception:
            pygame.display.set_mode((w, h), flags)

        pygame.display.set_caption(title)

        gl.glViewport(0, 0, w, h)

        # sensible defaults
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        ctx: Dict[str, Any] = {
            "w": w,
            "h": h,
            "title": title,
            "fps": fps,
            "clock": pygame.time.Clock(),
            "start_t": time.perf_counter(),
            "program": None,
            "u_mvp": -1,

            # drawables: each can carry its own model_local + rotate_with_scene flag
            # {
            #   "vao": int, "count": int, "mode": int,
            #   "indexed": bool, "index_type": int,
            #   "model_local": List[float],
            #   "rotate_with_scene": bool
            # }
            "drawables": [],

            "_gl_buffers": [],  # list[int]
            "_gl_vaos": [],     # list[int]
        }

        meta = {
            "type": "glctx",
            "w": w,
            "h": h,
            "gl_version": gl.glGetString(gl.GL_VERSION).decode("utf-8", "replace"),
            "glsl_version": gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode("utf-8", "replace"),
        }
        return ctx, meta


# ---------------------------
# Block: program (basic MVP)
# ---------------------------
@dataclass
class GLProgramBlock(BaseBlock):
    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    VS = """#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_col;
uniform mat4 u_mvp;
out vec3 v_col;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_col = a_col;
}
"""
    FS = """#version 330 core
in vec3 v_col;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_col, 1.0);
}
"""

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ctx = _ensure_ctx(payload, "glprogram")

        vs_src = str(params.get("vs", self.VS))
        fs_src = str(params.get("fs", self.FS))

        vs = _compile_shader(vs_src, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
        pid = _link_program(vs, fs)

        ctx["program"] = int(pid)
        ctx["u_mvp"] = int(gl.glGetUniformLocation(pid, "u_mvp"))

        return ctx, {"type": "glprogram", "program": int(pid), "u_mvp": int(ctx["u_mvp"])}


# ---------------------------
# Block: GL state toggles
# ---------------------------
@dataclass
class GLStateBlock(BaseBlock):
    """
    Toggle useful GL states (depth/cull/wireframe).
    """
    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "depth_test": {"type": "bool", "default": True, "ui": "bool"},
        "cull_face": {"type": "bool", "default": True, "ui": "bool"},
        "wireframe": {"type": "bool", "default": False, "ui": "bool"},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ctx = _ensure_ctx(payload, "glstate")

        depth_test = _as_bool(params.get("depth_test", True), True)
        cull_face = _as_bool(params.get("cull_face", True), True)
        wireframe = _as_bool(params.get("wireframe", False), False)

        if depth_test:
            gl.glEnable(gl.GL_DEPTH_TEST)
        else:
            gl.glDisable(gl.GL_DEPTH_TEST)

        if cull_face:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        try:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if wireframe else gl.GL_FILL)
        except Exception:
            pass

        return ctx, {"type": "glstate", "depth_test": depth_test, "cull_face": cull_face, "wireframe": wireframe}


# ---------------------------
# Block: cube mesh (per-object transform support)
# ---------------------------
@dataclass
class GLCubeBlock(BaseBlock):
    """
    Creates a colored cube (VAO/VBO/EBO).

    Per-object params:
      size, pos_x, pos_y, pos_z
    """
    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "size":  {"type": "float", "default": 1.0, "min": 0.1, "max": 50.0, "step": 0.1, "ui": "float"},
        "pos_x": {"type": "float", "default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1, "ui": "float"},
        "pos_y": {"type": "float", "default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1, "ui": "float"},
        "pos_z": {"type": "float", "default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1, "ui": "float"},
        # legacy alias (your GUI may still send it)
        "length": {"type": "float", "default": 1.0, "min": 0.1, "max": 50.0, "step": 0.1, "ui": "float"},
        "rotate_with_scene": {"type": "bool", "default": True, "ui": "bool"},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ctx = _ensure_ctx(payload, "glcube")

        # accept size or legacy length
        size = _as_float(params.get("size", params.get("length", 1.0)), 1.0)
        px = _as_float(params.get("pos_x", 0.0), 0.0)
        py = _as_float(params.get("pos_y", 0.0), 0.0)
        pz = _as_float(params.get("pos_z", 0.0), 0.0)
        rotate_with_scene = _as_bool(params.get("rotate_with_scene", True), True)

        s = size * 0.5

        verts = [
            -s, -s, -s,   1.0, 0.2, 0.2,
             s, -s, -s,   0.2, 1.0, 0.2,
             s,  s, -s,   0.2, 0.2, 1.0,
            -s,  s, -s,   1.0, 1.0, 0.2,
            -s, -s,  s,   0.2, 1.0, 1.0,
             s, -s,  s,   1.0, 0.2, 1.0,
             s,  s,  s,   1.0, 1.0, 1.0,
            -s,  s,  s,   0.4, 0.4, 0.4,
        ]

        idx = [
            # +Z (front)
            4, 5, 6, 6, 7, 4,
            # -Z (back)
            0, 3, 2, 2, 1, 0,
            # -X (left)
            0, 4, 7, 7, 3, 0,
            # +X (right)
            1, 2, 6, 6, 5, 1,
            # +Y (top)
            3, 7, 6, 6, 2, 3,
            # -Y (bottom)
            0, 1, 5, 5, 4, 0,
        ]
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(vao)

        arr = (ctypes.c_float * len(verts))(*verts)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(arr), arr, gl.GL_STATIC_DRAW)

        iarr = (ctypes.c_uint * len(idx))(*idx)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(iarr), iarr, gl.GL_STATIC_DRAW)

        stride = 6 * ctypes.sizeof(ctypes.c_float)

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))

        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        model_local = mat4_translate(px, py, pz)

        ctx["drawables"].append({
            "vao": int(vao),
            "count": len(idx),
            "mode": int(gl.GL_TRIANGLES),
            "indexed": True,
            "index_type": int(gl.GL_UNSIGNED_INT),
            "model_local": model_local,
            "rotate_with_scene": rotate_with_scene,

            # --- for export ---
            "primitive": "triangles",
            "cpu_stride": 6,        # x y z r g b
            "cpu_verts": verts,     # flat float list
            "cpu_idx": idx,         # flat int list (triangles)
        })

        ctx["_gl_vaos"].append(int(vao))
        ctx["_gl_buffers"].extend([int(vbo), int(ebo)])

        return ctx, {
            "type": "glcube",
            "vao": int(vao),
            "vbo": int(vbo),
            "ebo": int(ebo),
            "count": len(idx),
            "size": size,
            "pos": [px, py, pz],
            "rotate_with_scene": rotate_with_scene,
        }


# ---------------------------
# Block: axes (XYZ) (per-object transform support)
# ---------------------------
@dataclass
class GLAxesBlock(BaseBlock):
    """
    Adds XYZ axes lines (GL_LINES).
    By default, axes DO NOT rotate with the scene, so they stay "world aligned".
    """
    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "length": {"type": "float", "default": 1.5, "min": 0.1, "max": 50.0, "step": 0.1, "ui": "float"},
        "rotate_with_scene": {"type": "bool", "default": False, "ui": "bool"},
        "pos_x": {"type": "float", "default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1, "ui": "float"},
        "pos_y": {"type": "float", "default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1, "ui": "float"},
        "pos_z": {"type": "float", "default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1, "ui": "float"},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ctx = _ensure_ctx(payload, "glaxes")

        L = _as_float(params.get("length", 1.5), 1.5)
        rotate_with_scene = _as_bool(params.get("rotate_with_scene", False), False)
        px = _as_float(params.get("pos_x", 0.0), 0.0)
        py = _as_float(params.get("pos_y", 0.0), 0.0)
        pz = _as_float(params.get("pos_z", 0.0), 0.0)

        # 6 vertices (3 lines): origin->X, origin->Y, origin->Z
        # pos + col
        verts = [
            0,0,0,  1,0,0,   L,0,0,  1,0,0,
            0,0,0,  0,1,0,   0,L,0,  0,1,0,
            0,0,0,  0,0,1,   0,0,L,  0,0,1,
        ]

        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(vao)

        arr = (ctypes.c_float * len(verts))(*verts)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(arr), arr, gl.GL_STATIC_DRAW)

        stride = 6 * ctypes.sizeof(ctypes.c_float)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))

        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        model_local = mat4_translate(px, py, pz)
        ctx["drawables"].append({
            "vao": int(vao),
            "count": 6,                 # 6 vertices
            "mode": int(gl.GL_LINES),
            "indexed": False,
            "model_local": model_local,
            "rotate_with_scene": rotate_with_scene,

            # --- for export ---
            "primitive": "lines",
            "cpu_stride": 6,        # x y z r g b
            "cpu_verts": verts,     # flat float list
            "cpu_idx": None,
        })
        ctx["_gl_vaos"].append(int(vao))
        ctx["_gl_buffers"].append(int(vbo))

        return ctx, {
            "type": "glaxes",
            "vao": int(vao),
            "vbo": int(vbo),
            "length": L,
            "pos": [px, py, pz],
            "rotate_with_scene": rotate_with_scene,
        }


# ---------------------------
# Block: loop (per-drawable MVP)
# ---------------------------
@dataclass
class GLLoopBlock(BaseBlock):
    """
    Runs render loop until closed / ESC.

    Key change:
      - computes MVP PER DRAWABLE using drawable["model_local"]
      - optionally applies scene rotation depending on drawable["rotate_with_scene"]
    """
    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "fps": {"type": "int", "default": 120, "min": 1, "max": 1000, "step": 1, "ui": "int"},

        "clear_r": {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "ui": "float"},
        "clear_g": {"type": "float", "default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01, "ui": "float"},
        "clear_b": {"type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "ui": "float"},

        "fovy": {"type": "float", "default": 60.0, "min": 20.0, "max": 120.0, "step": 1.0, "ui": "float"},
        "near": {"type": "float", "default": 0.10, "min": 0.01, "max": 10.0, "step": 0.01, "ui": "float"},
        "far":  {"type": "float", "default": 100.0, "min": 1.0, "max": 5000.0, "step": 1.0, "ui": "float"},

        "cam_dist": {"type": "float", "default": 3.0, "min": 0.5, "max": 50.0, "step": 0.1, "ui": "float"},
        "rot_x_speed": {"type": "float", "default": 0.8, "min": -10.0, "max": 10.0, "step": 0.1, "ui": "float"},
        "rot_y_speed": {"type": "float", "default": 1.1, "min": -10.0, "max": 10.0, "step": 0.1, "ui": "float"},
        "pause": {"type": "bool", "default": False, "ui": "bool"},
        "scene_rotate": {"type": "bool", "default": False, "ui": "bool"},
        # prints a single line at loop start so you can confirm params are applied
        "print_params": {"type": "bool", "default": True, "ui": "bool"},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        ctx = _ensure_ctx(payload, "glloop")

        pid = ctx.get("program")
        if not pid:
            raise RuntimeError("glloop requires program (run glprogram first)")

        fps = _as_int(params.get("fps", ctx.get("fps", 60)), 60)

        cr = _as_float(params.get("clear_r", 0.05), 0.05)
        cg = _as_float(params.get("clear_g", 0.06), 0.06)
        cb = _as_float(params.get("clear_b", 0.10), 0.10)

        fovy = _as_float(params.get("fovy", 60.0), 60.0)
        znear = _as_float(params.get("near", 0.1), 0.1)
        zfar = _as_float(params.get("far", 100.0), 100.0)

        cam_dist = _as_float(params.get("cam_dist", 3.0), 3.0)
        rot_x_speed = _as_float(params.get("rot_x_speed", 0.8), 0.8)
        rot_y_speed = _as_float(params.get("rot_y_speed", 1.1), 1.1)
        pause = _as_bool(params.get("pause", False), False)

        print_params = _as_bool(params.get("print_params", True), True)


        frames = 0
        t0 = time.perf_counter()

        if print_params:
            try:
                print(
                    f"[glloop] fps={fps} cam_dist={cam_dist} clear=({cr:.2f},{cg:.2f},{cb:.2f}) "
                    f"fovy={fovy} near={znear} far={zfar} drawables={len(ctx.get('drawables', []))}",
                    flush=True
                )
            except Exception:
                pass

        try:
            running = True
            while running:
                dt_s = ctx["clock"].tick(fps) / 1000.0
                if dt_s <= 0:
                    dt_s = 1.0 / max(1.0, float(fps))
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.type == pygame.VIDEORESIZE:
                        ctx["w"], ctx["h"] = ev.w, ev.h
                        gl.glViewport(0, 0, ev.w, ev.h)
                    elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F1:
                        p = ctx.get("player")
                        if isinstance(p, dict):
                            p["grab_mouse"] = not bool(p.get("grab_mouse", False))
                            try:
                                pygame.event.set_grab(bool(p["grab_mouse"]))
                                pygame.mouse.set_visible(not bool(p["grab_mouse"]))
                                pygame.mouse.get_rel()
                            except Exception:
                                pass
                    elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                        # RMB down = mouselook (WoW feel)
                        p = ctx.get("player")
                        if isinstance(p, dict):
                            p["mouselook"] = True
                            try:
                                pygame.event.set_grab(True)
                                pygame.mouse.set_visible(False)
                                pygame.mouse.get_rel()
                            except Exception:
                                pass

                    elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                        # RMB up = stop mouselook
                        p = ctx.get("player")
                        if isinstance(p, dict):
                            p["mouselook"] = False
                            # return to "grab_mouse" preference
                            want_grab = bool(p.get("grab_mouse", False))
                            try:
                                pygame.event.set_grab(want_grab)
                                pygame.mouse.set_visible(not want_grab)
                                pygame.mouse.get_rel()
                            except Exception:
                                pass

                    elif ev.type == pygame.MOUSEWHEEL:
                        # zoom in/out (wow/orbit)
                        p = ctx.get("player")
                        if isinstance(p, dict):
                            z = float(p.get("zoom_speed", 1.2))
                            p["orbit_dist"] = max(0.5, float(p.get("orbit_dist", 7.5)) - (ev.y * z))
                now = time.perf_counter()
                t = float(now - ctx["start_t"])
                if pause:
                    t = 0.0

                w = max(1, int(ctx["w"]))
                h = max(1, int(ctx["h"]))
                aspect = w / float(h)

                proj = mat4_perspective(fovy, aspect, znear, zfar)
                player = ctx.get("player")
                if isinstance(player, dict):
                    mode = str(player.get("mode", "wow"))
                    yaw = float(player.get("yaw", 0.0))
                    pitch = float(player.get("pitch", 0.15))

                    pivot = player.get("pivot", [0.0, 0.0, 0.0])
                    if not (isinstance(pivot, list) and len(pivot) == 3):
                        pivot = [0.0, 0.0, 0.0]

                    vel = player.get("vel", [0.0, 0.0, 0.0])
                    if not (isinstance(vel, list) and len(vel) == 3):
                        vel = [0.0, 0.0, 0.0]

                    up = [0.0, 1.0, 0.0]
                    fly = bool(player.get("fly", False))
                    lock_ground_y = bool(player.get("lock_ground_y", True))

                    # --- 1) EDGE AUTO-TURN (only when cursor visible / not grabbed)
                    if bool(player.get("edge_turn", False)) and not (
                            bool(player.get("mouselook", False)) or bool(player.get("grab_mouse", False))):
                        mx, my = pygame.mouse.get_pos()
                        w = max(1, int(ctx["w"]))
                        h = max(1, int(ctx["h"]))
                        edge = int(player.get("edge_px", 24))
                        spd = float(player.get("edge_turn_speed", 2.2))

                        if edge > 0:
                            if mx < edge:
                                amt = (edge - mx) / float(edge)
                                yaw -= spd * amt * dt_s
                            elif mx > (w - edge):
                                amt = (mx - (w - edge)) / float(edge)
                                yaw += spd * amt * dt_s

                            # optional pitch edge turn (subtle)
                            # if my < edge: pitch += 0.8 * spd * ((edge - my)/edge) * dt_s
                            # elif my > (h - edge): pitch -= 0.8 * spd * ((my - (h-edge))/edge) * dt_s

                    # --- 2) MOUSELOOK (WoW RMB)
                    if bool(player.get("mouselook", False)) or bool(player.get("grab_mouse", False)):
                        mx, my = pygame.mouse.get_rel()
                        sens = float(player.get("mouse_sens", 0.0022))
                        inv = -1.0 if bool(player.get("invert_y", False)) else 1.0

                        yaw += (mx * sens)
                        pitch += (my * sens) * inv  # move mouse up => look up (unless inverted)
                        pitch = clamp(pitch, -1.20, 1.20)

                    player["yaw"] = yaw
                    player["pitch"] = pitch

                    # movement basis from yaw (stable strafing, no “wall” feel)
                    fwd = v3_norm([math.sin(yaw), 0.0, -math.cos(yaw)])
                    right = v3_norm([math.cos(yaw), 0.0, math.sin(yaw)])

                    keys = pygame.key.get_pressed()
                    wish = [0.0, 0.0, 0.0]
                    if keys[pygame.K_w]: wish = v3_add(wish, fwd)
                    if keys[pygame.K_s]: wish = v3_sub(wish, fwd)
                    if keys[pygame.K_d]: wish = v3_add(wish, right)
                    if keys[pygame.K_a]: wish = v3_sub(wish, right)

                    if fly:
                        if keys[pygame.K_SPACE]: wish = v3_add(wish, up)
                        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]: wish = v3_sub(wish, up)

                    wish_dir = v3_norm(wish)

                    # --- 3) SMOOTH ACCEL/DECEL (production movement)
                    speed = float(player.get("move_speed", 6.0))
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        speed *= float(player.get("run_mul", 1.8))

                    desired_vel = v3_scale(wish_dir, speed) if v3_len(wish_dir) > 0 else [0.0, 0.0, 0.0]

                    accel = float(player.get("accel", 18.0))
                    decel = float(player.get("decel", 22.0))
                    rate = accel if v3_len(wish_dir) > 0 else decel
                    k = smooth_k(rate, dt_s)

                    vel = v3_lerp(vel, desired_vel, k)
                    pivot = v3_add(pivot, v3_scale(vel, dt_s))

                    if (not fly) and lock_ground_y:
                        # keep original ground y
                        base_y = float(player.get("pivot", [0.0, 0.0, 0.0])[1])
                        pivot[1] = base_y

                    player["vel"] = vel
                    player["pivot"] = pivot

                    # --- 4) CAMERA (WoW boom)
                    eye_h = float(player.get("eye_height", 1.6))
                    dist = float(player.get("orbit_dist", 7.5))
                    h_off = float(player.get("orbit_height", 0.7))

                    target = [pivot[0], pivot[1] + eye_h, pivot[2]]

                    # forward from yaw/pitch (for camera direction)
                    fwd3 = [
                        math.sin(yaw) * math.cos(pitch),
                        math.sin(pitch),
                        -math.cos(yaw) * math.cos(pitch),
                    ]
                    fwd3 = v3_norm(fwd3)

                    desired_cam = v3_sub(target, v3_scale(fwd3, dist))
                    desired_cam[1] += h_off

                    cam_pos = player.get("cam_pos")
                    if not (isinstance(cam_pos, list) and len(cam_pos) == 3):
                        cam_pos = desired_cam

                    cam_s = float(player.get("cam_smooth", 14.0))
                    cam_k = smooth_k(cam_s, dt_s)
                    cam_pos = v3_lerp(cam_pos, desired_cam, cam_k)

                    player["cam_pos"] = cam_pos

                    # view matrix
                    view = mat4_look_at(cam_pos, target, up)

                else:
                    view = mat4_translate(0.0, 0.0, -cam_dist)

                # shared scene rotation (applied only to drawables that opt-in)
                scene_rotate = _as_bool(params.get("scene_rotate", False), False)
                if scene_rotate and not pause:
                    scene_rot = mat4_mul(
                        mat4_rotate_y(t * rot_y_speed),
                        mat4_rotate_x(t * rot_x_speed),
                    )
                else:
                    scene_rot = mat4_identity()
                gl.glClearColor(cr, cg, cb, 1.0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                gl.glUseProgram(int(pid))

                u_mvp = int(ctx.get("u_mvp", -1))

                for d in ctx.get("drawables", []):
                    vao = int(d["vao"])
                    mode = int(d.get("mode", gl.GL_TRIANGLES))

                    model_local = d.get("model_local")
                    if not isinstance(model_local, list) or len(model_local) != 16:
                        model_local = mat4_identity()

                    if bool(d.get("rotate_with_scene", True)):
                        # rotate in place, then translate (because model_local is translation matrix)
                        model = mat4_mul(model_local, scene_rot)
                    else:
                        model = model_local

                    mvp = mat4_mul(proj, mat4_mul(view, model))

                    if u_mvp >= 0:
                        buf = (ctypes.c_float * 16)(*mvp)
                        gl.glUniformMatrix4fv(u_mvp, 1, gl.GL_FALSE, buf)

                    gl.glBindVertexArray(vao)

                    if d.get("indexed", True):
                        count = int(d["count"])
                        idx_t = int(d.get("index_type", gl.GL_UNSIGNED_INT))
                        gl.glDrawElements(mode, count, idx_t, ctypes.c_void_p(0))
                    else:
                        count = int(d["count"])
                        gl.glDrawArrays(mode, 0, count)

                gl.glBindVertexArray(0)
                gl.glUseProgram(0)

                pygame.display.flip()
                frames += 1

        finally:
            # Cleanup GPU objects
            for vao in ctx.get("_gl_vaos", []):
                try:
                    gl.glDeleteVertexArrays(1, [int(vao)])
                except Exception:
                    pass
            for bid in ctx.get("_gl_buffers", []):
                try:
                    gl.glDeleteBuffers(1, [int(bid)])
                except Exception:
                    pass
            try:
                gl.glDeleteProgram(int(pid))
            except Exception:
                pass
            try:
                pygame.quit()
            except Exception:
                pass

        dt = time.perf_counter() - t0
        avg_fps = (frames / dt) if dt > 0 else 0.0
        return "ok", {"type": "glloop", "frames": frames, "seconds": dt, "avg_fps": avg_fps}


# Register blocks
BLOCKS.register("glctx", GLContextBlock)
BLOCKS.register("glprogram", GLProgramBlock)
BLOCKS.register("glstate", GLStateBlock)
BLOCKS.register("glcube", GLCubeBlock)
BLOCKS.register("glaxes", GLAxesBlock)
BLOCKS.register("glloop", GLLoopBlock)

def v3(x: float, y: float, z: float) -> List[float]:
    return [float(x), float(y), float(z)]

def v3_add(a: List[float], b: List[float]) -> List[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

def v3_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

def v3_scale(a: List[float], s: float) -> List[float]:
    return [a[0] * s, a[1] * s, a[2] * s]

def v3_len(a: List[float]) -> float:
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def v3_norm(a: List[float]) -> List[float]:
    L = v3_len(a)
    if L <= 1e-9:
        return [0.0, 0.0, 0.0]
    return [a[0] / L, a[1] / L, a[2] / L]

def v3_cross(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
def v3_dot(a: List[float], b: List[float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def mat4_look_at(eye: List[float], target: List[float], up: List[float]) -> List[float]:
    # OpenGL RH lookAt, column-major
    f = v3_norm(v3_sub(target, eye))      # forward
    r = v3_norm(v3_cross(f, up))          # right
    u = v3_cross(r, f)                    # corrected up

    # column-major matrix:
    # [ r.x  u.x  -f.x  0
    #   r.y  u.y  -f.y  0
    #   r.z  u.z  -f.z  0
    #  -dot(r,eye) -dot(u,eye) dot(f,eye) 1 ]
    return [
        r[0], u[0], -f[0], 0.0,
        r[1], u[1], -f[1], 0.0,
        r[2], u[2], -f[2], 0.0,
        -v3_dot(r, eye), -v3_dot(u, eye), v3_dot(f, eye), 1.0,
    ]

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else (b if x > b else x)

def smooth_k(rate: float, dt: float) -> float:
    # stable exponential smoothing factor
    if rate <= 0:
        return 1.0
    return 1.0 - math.exp(-rate * dt)

def v3_lerp(a: List[float], b: List[float], t: float) -> List[float]:
    return [a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t]
@dataclass
class GLPlayerControllerBlock(BaseBlock):
    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "mode": {"type": "enum", "default": "wow", "choices": ["fps", "orbit", "wow"], "ui": "enum"},

        # pivot/anchor (player position)
        "pivot_x": {"type": "float", "default": 0.0, "min": -500.0, "max": 500.0, "step": 0.1, "ui": "float"},
        "pivot_y": {"type": "float", "default": 0.0, "min": -500.0, "max": 500.0, "step": 0.1, "ui": "float"},
        "pivot_z": {"type": "float", "default": 0.0, "min": -500.0, "max": 500.0, "step": 0.1, "ui": "float"},

        # look
        "yaw":   {"type": "float", "default": 0.0, "min": -999.0, "max": 999.0, "step": 0.01, "ui": "float"},
        "pitch": {"type": "float", "default": 0.15, "min": -1.20, "max": 1.20, "step": 0.01, "ui": "float"},

        # movement (production-feel)
        "move_speed": {"type": "float", "default": 6.0, "min": 0.1, "max": 50.0, "step": 0.1, "ui": "float"},
        "run_mul":    {"type": "float", "default": 1.8, "min": 1.0, "max": 10.0, "step": 0.1, "ui": "float"},
        "accel":      {"type": "float", "default": 18.0, "min": 0.0, "max": 200.0, "step": 0.5, "ui": "float"},
        "decel":      {"type": "float", "default": 22.0, "min": 0.0, "max": 200.0, "step": 0.5, "ui": "float"},

        # mouse
        "mouse_sens": {"type": "float", "default": 0.0022, "min": 0.0001, "max": 0.02, "step": 0.0001, "ui": "float"},
        "invert_y":   {"type": "bool",  "default": False, "ui": "bool"},

        # camera boom (wow/orbit)
        "eye_height":   {"type": "float", "default": 1.6, "min": 0.0, "max": 20.0, "step": 0.1, "ui": "float"},
        "orbit_dist":   {"type": "float", "default": 7.5, "min": 0.5, "max": 200.0, "step": 0.1, "ui": "float"},
        "orbit_height": {"type": "float", "default": 0.7, "min": -50.0, "max": 50.0, "step": 0.1, "ui": "float"},
        "zoom_speed":   {"type": "float", "default": 1.2, "min": 0.01, "max": 50.0, "step": 0.1, "ui": "float"},
        "cam_smooth":   {"type": "float", "default": 14.0, "min": 0.0, "max": 200.0, "step": 0.5, "ui": "float"},

        # edge auto-turn (only works when mouse not grabbed)
        "edge_turn":        {"type": "bool",  "default": True, "ui": "bool"},
        "edge_px":          {"type": "int",   "default": 24, "min": 0, "max": 300, "step": 1, "ui": "int"},
        "edge_turn_speed":  {"type": "float", "default": 2.2, "min": 0.0, "max": 20.0, "step": 0.1, "ui": "float"},

        # misc
        "fly": {"type": "bool", "default": False, "ui": "bool"},
        "lock_ground_y": {"type": "bool", "default": True, "ui": "bool"},
        "grab_mouse": {"type": "bool", "default": False, "ui": "bool"},  # wow-style: grab only while RMB held
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ctx = _ensure_ctx(payload, "glplayer")

        mode = str(params.get("mode", "wow") or "wow").strip().lower()
        if mode not in ("fps", "orbit", "wow"):
            mode = "wow"

        pivot = [
            _as_float(params.get("pivot_x", 0.0), 0.0),
            _as_float(params.get("pivot_y", 0.0), 0.0),
            _as_float(params.get("pivot_z", 0.0), 0.0),
        ]

        ctx["player"] = {
            "mode": mode,
            "pivot": pivot,
            "vel": [0.0, 0.0, 0.0],      # <- smooth movement
            "cam_pos": None,             # <- smoothed boom camera

            "yaw": _as_float(params.get("yaw", 0.0), 0.0),
            "pitch": _as_float(params.get("pitch", 0.15), 0.15),

            "move_speed": _as_float(params.get("move_speed", 6.0), 6.0),
            "run_mul": _as_float(params.get("run_mul", 1.8), 1.8),
            "accel": _as_float(params.get("accel", 18.0), 18.0),
            "decel": _as_float(params.get("decel", 22.0), 22.0),

            "mouse_sens": _as_float(params.get("mouse_sens", 0.0022), 0.0022),
            "invert_y": _as_bool(params.get("invert_y", False), False),

            "eye_height": _as_float(params.get("eye_height", 1.6), 1.6),
            "orbit_dist": _as_float(params.get("orbit_dist", 7.5), 7.5),
            "orbit_height": _as_float(params.get("orbit_height", 0.7), 0.7),
            "zoom_speed": _as_float(params.get("zoom_speed", 1.2), 1.2),
            "cam_smooth": _as_float(params.get("cam_smooth", 14.0), 14.0),

            "edge_turn": _as_bool(params.get("edge_turn", True), True),
            "edge_px": _as_int(params.get("edge_px", 24), 24),
            "edge_turn_speed": _as_float(params.get("edge_turn_speed", 2.2), 2.2),

            "fly": _as_bool(params.get("fly", False), False),
            "lock_ground_y": _as_bool(params.get("lock_ground_y", True), True),

            "grab_mouse": _as_bool(params.get("grab_mouse", False), False),
            "mouselook": False,  # <- RMB toggles this
        }

        return ctx, {"type": "glplayer", **ctx["player"]}


BLOCKS.register("glplayer", GLPlayerControllerBlock)
def mat4_mul_v3(m: List[float], v: List[float]) -> List[float]:
    # column-major, vec4(v,1)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    rx = m[0] * x + m[4] * y + m[8]  * z + m[12] * 1.0
    ry = m[1] * x + m[5] * y + m[9]  * z + m[13] * 1.0
    rz = m[2] * x + m[6] * y + m[10] * z + m[14] * 1.0
    rw = m[3] * x + m[7] * y + m[11] * z + m[15] * 1.0
    if abs(rw) > 1e-9 and rw != 1.0:
        rx /= rw
        ry /= rw
        rz /= rw
    return [rx, ry, rz]


@dataclass
class GLExportBlock(BaseBlock):
    """
    Export current ctx drawables to OBJ (CPU-side mesh data).

    Can be used:
      - as a tap (mid-pipeline): ...|glexport|glloop   (cleanup must be False)
      - as a sink:               ...|glexport         (cleanup can be True)
    """
    PARAMS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "path": {"type": "str", "default": "export.obj", "ui": "text"},
        "format": {"type": "enum", "default": "obj", "choices": ["obj"], "ui": "enum"},
        "apply_transforms": {"type": "bool", "default": True, "ui": "bool"},
        "include_colors": {"type": "bool", "default": True, "ui": "bool"},

        # ✅ important: default must NOT kill the GL runtime if glloop comes after
        "cleanup": {"type": "bool", "default": False, "ui": "bool"},

        # ✅ optional: if False, block returns "ok" instead of ctx
        "passthrough": {"type": "bool", "default": True, "ui": "bool"},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        ctx = _ensure_ctx(payload, "glexport")

        fmt = str(params.get("format", "obj") or "obj").strip().lower()
        if fmt != "obj":
            raise ValueError("glexport currently supports only format='obj'")

        path = str(params.get("path", "export.obj") or "export.obj").strip()
        if not path:
            path = "export.obj"
        root, ext = os.path.splitext(path)
        if not ext:
            path = path + ".obj"
        elif ext.lower() != ".obj":
            path = root + ".obj"

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        apply_tx = _as_bool(params.get("apply_transforms", True), True)
        include_col = _as_bool(params.get("include_colors", True), True)
        cleanup = _as_bool(params.get("cleanup", False), False)
        passthrough = _as_bool(params.get("passthrough", True), True)

        drawables = ctx.get("drawables", [])
        v_total = 0
        f_total = 0
        l_total = 0

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Nate's 3D Renderer - OBJ export\n")
            f.write("# vertex color extension: v x y z r g b (optional)\n\n")

            global_v_base = 1

            for di, d in enumerate(drawables):
                verts = d.get("cpu_verts")
                stride = int(d.get("cpu_stride", 6))
                prim = str(d.get("primitive", "triangles"))
                idx = d.get("cpu_idx")

                if not isinstance(verts, list) or len(verts) < stride:
                    continue

                model = d.get("model_local")
                if not (isinstance(model, list) and len(model) == 16):
                    model = mat4_identity()

                f.write(f"o drawable_{di}\n")

                n_verts = len(verts) // stride
                for vi in range(n_verts):
                    off = vi * stride
                    x, y, z = float(verts[off + 0]), float(verts[off + 1]), float(verts[off + 2])
                    pos = [x, y, z]
                    if apply_tx:
                        pos = mat4_mul_v3(model, pos)

                    if include_col and stride >= 6:
                        r = float(verts[off + 3]); g = float(verts[off + 4]); b = float(verts[off + 5])
                        f.write(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
                    else:
                        f.write(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

                if prim == "triangles":
                    if isinstance(idx, list) and len(idx) >= 3:
                        for ii in range(0, len(idx) - 2, 3):
                            a = int(idx[ii + 0]) + global_v_base
                            b = int(idx[ii + 1]) + global_v_base
                            c = int(idx[ii + 2]) + global_v_base
                            f.write(f"f {a} {b} {c}\n")
                            f_total += 1
                    else:
                        for ii in range(0, n_verts - 2, 3):
                            a = global_v_base + ii
                            b = global_v_base + ii + 1
                            c = global_v_base + ii + 2
                            f.write(f"f {a} {b} {c}\n")
                            f_total += 1

                elif prim == "lines":
                    for ii in range(0, n_verts - 1, 2):
                        a = global_v_base + ii
                        b = global_v_base + ii + 1
                        f.write(f"l {a} {b}\n")
                        l_total += 1

                f.write("\n")
                v_total += n_verts
                global_v_base += n_verts

        if cleanup:
            try:
                for vao in ctx.get("_gl_vaos", []):
                    try: gl.glDeleteVertexArrays(1, [int(vao)])
                    except Exception: pass
                for bid in ctx.get("_gl_buffers", []):
                    try: gl.glDeleteBuffers(1, [int(bid)])
                    except Exception: pass
                pid = ctx.get("program")
                if pid:
                    try: gl.glDeleteProgram(int(pid))
                    except Exception: pass
            finally:
                try: pygame.quit()
                except Exception: pass

        meta = {
            "type": "glexport",
            "path": os.path.abspath(path),
            "format": "obj",
            "vertices": int(v_total),
            "faces": int(f_total),
            "lines": int(l_total),
        }

        # ✅ critical: return ctx so later stages (like glloop) still work
        return (ctx if passthrough else "ok"), meta


# Register exporter block (and keep your existing registrations)
BLOCKS.register("glexport", GLExportBlock)
