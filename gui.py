# gui.py  (updated: dual library panes (Normal vs Builder) w/ shared Add buttons,
#         model export, and session persistence for stages + params + selection)

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QProcess, QStandardPaths
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget,
    QListWidget, QMessageBox, QGroupBox, QFormLayout, QSplitter, QSizePolicy,
    QCheckBox, QSlider, QScrollArea, QInputDialog, QFileDialog
)

from registry import BLOCKS
import pipeline  # registers pipeline
import blocks    # registers gl blocks
import builder_blocks  # registers builder blocks
import main as cli_entry  # <-- your CLI runner module (main.py)

def _maybe_run_cli_mode() -> None:
    # If this exe is invoked as: Nates3DRenderer.exe cli pipeline ...
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cli":
        argv = sys.argv[2:]  # drop the "cli"
        raise SystemExit(cli_entry.main(argv))
DARK_QSS = """
QWidget { background-color: #121417; color: #E7E9EA; font-size: 12px; }
QGroupBox { border: 1px solid #252B33; border-radius: 8px; margin-top: 10px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #C9D1D9; font-weight: 600; }
QLineEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0D0F12; border: 1px solid #252B33; border-radius: 8px; padding: 6px;
    selection-background-color: #2B5BFF; selection-color: white;
}
QPlainTextEdit { font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; }
QPushButton {
    background-color: #1A2028; border: 1px solid #2B3440; border-radius: 10px;
    padding: 8px 12px; font-weight: 600;
}
QPushButton:hover { background-color: #202835; }
QPushButton:pressed { background-color: #141A22; }
QPushButton:disabled { color: #6C7784; border-color: #222831; background-color: #101317; }
QTabWidget::pane { border: 1px solid #252B33; border-radius: 10px; padding: 6px; }
QTabBar::tab {
    background: #14181E; border: 1px solid #252B33; padding: 8px 12px;
    border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 4px;
}
QTabBar::tab:selected { background: #1B2230; border-bottom-color: #1B2230; }
QLabel#TitleLabel { font-size: 18px; font-weight: 800; color: #FFFFFF; }
QLabel#SubLabel { color: #AAB4BE; }
"""

SINK_BASES = {"glloop", "glloop_orbit"}


# ---------------------------
# persistence
# ---------------------------
APP_ID = "nates_3d_renderer"

def _state_path() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or os.path.expanduser("~")
    d = os.path.join(base, APP_ID)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "ui_state.json")

def _json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return str(x)

def load_ui_state() -> Dict[str, Any]:
    p = _state_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_ui_state(state: Dict[str, Any]) -> None:
    p = _state_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_json_safe(state), f, indent=2)
    except Exception:
        pass


# ---------------------------
# helpers
# ---------------------------
def stage_id_and_base(stage_token: str) -> Tuple[str, str]:
    s = stage_token.strip()
    base = s.split(":", 1)[0].strip().lower()
    return s.lower(), base

def parse_stage_chain(text: str) -> List[str]:
    return [t.strip() for t in (text or "").split("|") if t.strip()]

def block_schema_for_stage(stage_token: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    stage_id, base = stage_id_and_base(stage_token)
    cls = BLOCKS.get_class(base)
    if cls is None:
        return base, {}
    schema = getattr(cls, "PARAMS", {}) or {}
    if not isinstance(schema, dict):
        return base, {}
    return base, schema

def schema_defaults(schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, spec in (schema or {}).items():
        out[k] = (spec or {}).get("default", None)
    return out

def coerce_to_type(value: Any, t: str) -> Any:
    if t == "int":
        try: return int(value)
        except Exception: return 0
    if t == "float":
        try: return float(value)
        except Exception: return 0.0
    if t == "bool":
        if isinstance(value, bool): return value
        s = str(value).strip().lower()
        return s in ("1", "true", "yes", "y", "on")
    return str(value)

def normalize_pipeline(tokens: List[str]) -> List[str]:
    tokens = [t.strip() for t in tokens if t and t.strip()]
    if not tokens:
        return tokens

    # keep only ONE sink (the last one user specified), and force it to the end
    sinks: List[str] = []
    nons: List[str] = []
    for t in tokens:
        _sid, base = stage_id_and_base(t)
        if base in SINK_BASES:
            sinks.append(t)
        else:
            nons.append(t)
    if sinks:
        return nons + [sinks[-1]]
    return nons

def is_builder_block(name: str) -> bool:
    name = (name or "").strip().lower()
    if not name or name == "pipeline":
        return False
    cls = BLOCKS.get_class(name)
    if cls is None:
        return False
    mod = getattr(cls, "__module__", "") or ""
    return ("builder_blocks" in mod) or name.startswith("builder_") or name.startswith("build_")

def is_normal_block(name: str) -> bool:
    name = (name or "").strip().lower()
    if not name or name == "pipeline":
        return False
    return not is_builder_block(name)


# ---------------------------
# dynamic param controls
# ---------------------------
class ParamEditor(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._schema: Dict[str, Dict[str, Any]] = {}
        self._values: Dict[str, Any] = {}
        self._widgets: Dict[str, QWidget] = {}
        self._on_change = None

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(8)

        self.form_box = QGroupBox("Stage Parameters")
        self.form = QFormLayout(self.form_box)
        self.form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.form.setFormAlignment(Qt.AlignTop)
        self.layout().addWidget(self.form_box)

        self.hint = QLabel("")
        self.hint.setObjectName("SubLabel")
        self.layout().addWidget(self.hint)

        self.layout().addStretch(1)

    def set_on_change(self, fn) -> None:
        self._on_change = fn

    def clear(self) -> None:
        while self.form.rowCount() > 0:
            self.form.removeRow(0)
        self._schema = {}
        self._values = {}
        self._widgets = {}
        self.hint.setText("")

    def set_stage(self, stage_token: str, schema: Dict[str, Dict[str, Any]], values: Dict[str, Any]) -> None:
        self.clear()
        self._schema = schema or {}
        self._values = dict(values or {})

        if not self._schema:
            self.hint.setText("This stage exposes no editable parameters.")
            return

        for key in sorted(self._schema.keys()):
            spec = self._schema.get(key, {}) or {}
            t = str(spec.get("type", "str"))
            ui = str(spec.get("ui", t))
            label = QLabel(key)
            label.setToolTip(str(spec.get("help", "")))

            default = spec.get("default", None)
            cur = self._values.get(key, default)

            w = self._build_widget(key, t, ui, spec, cur)
            self._widgets[key] = w
            self.form.addRow(label, w)

        self.hint.setText(f"Editing: {stage_token}")

    def values(self) -> Dict[str, Any]:
        return dict(self._values)

    def _emit_change(self) -> None:
        if self._on_change:
            self._on_change()

    def _build_widget(self, key: str, t: str, ui: str, spec: Dict[str, Any], cur: Any) -> QWidget:
        if t == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(cur))
            cb.stateChanged.connect(lambda _=None: self._set_value(key, cb.isChecked()))
            return cb

        if t in ("int", "float"):
            mn = spec.get("min", None)
            mx = spec.get("max", None)
            step = spec.get("step", 1 if t == "int" else 0.01)

            if mn is not None and mx is not None:
                return self._slider_spin(key, t, float(mn), float(mx), float(step), cur)

            if t == "int":
                sp = QSpinBox()
                sp.setRange(-2**31, 2**31 - 1)
                sp.setValue(int(coerce_to_type(cur, "int")))
                sp.valueChanged.connect(lambda v: self._set_value(key, int(v)))
                return sp
            else:
                sp = QDoubleSpinBox()
                sp.setDecimals(4)
                sp.setRange(-1e9, 1e9)
                sp.setSingleStep(float(step))
                sp.setValue(float(coerce_to_type(cur, "float")))
                sp.valueChanged.connect(lambda v: self._set_value(key, float(v)))
                return sp

        if t == "enum":
            choices = spec.get("choices", []) or []
            combo = QComboBox()
            for c in choices:
                combo.addItem(str(c))
            if cur is not None and str(cur) in [str(c) for c in choices]:
                combo.setCurrentText(str(cur))
            combo.currentTextChanged.connect(lambda v: self._set_value(key, v))
            return combo

        le = QLineEdit()
        le.setText("" if cur is None else str(cur))
        le.textChanged.connect(lambda v: self._set_value(key, v))
        return le

    def _slider_spin(self, key: str, t: str, mn: float, mx: float, step: float, cur: Any) -> QWidget:
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        if step <= 0:
            step = 1.0
        steps = int(round((mx - mn) / step))
        if steps <= 0:
            steps = 1
        if steps > 5000:
            # too many slider steps -> spinbox only
            if t == "int":
                sp = QSpinBox()
                sp.setRange(int(mn), int(mx))
                sp.setValue(int(coerce_to_type(cur, "int")))
                sp.valueChanged.connect(lambda v: self._set_value(key, int(v)))
                lay.addWidget(sp, 1)
                return wrap
            else:
                sp = QDoubleSpinBox()
                sp.setDecimals(4)
                sp.setRange(mn, mx)
                sp.setSingleStep(step)
                sp.setValue(float(coerce_to_type(cur, "float")))
                sp.valueChanged.connect(lambda v: self._set_value(key, float(v)))
                lay.addWidget(sp, 1)
                return wrap

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, steps)

        if t == "int":
            spin = QSpinBox()
            spin.setRange(int(mn), int(mx))
            spin.setValue(int(coerce_to_type(cur, "int")))
        else:
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(mn, mx)
            spin.setSingleStep(step)
            spin.setValue(float(coerce_to_type(cur, "float")))

        cur_f = float(coerce_to_type(cur, "float" if t == "float" else "int"))
        cur_f = max(mn, min(mx, cur_f))
        slider.setValue(int(round((cur_f - mn) / step)))

        def slider_to_value(pos: int) -> float:
            return mn + (pos * step)

        def on_slider(pos: int) -> None:
            v = slider_to_value(pos)
            if t == "int":
                spin.blockSignals(True); spin.setValue(int(round(v))); spin.blockSignals(False)
                self._set_value(key, int(round(v)), emit=True)
            else:
                spin.blockSignals(True); spin.setValue(float(v)); spin.blockSignals(False)
                self._set_value(key, float(v), emit=True)

        def on_spin(v: Any) -> None:
            vf = float(v)
            vf = max(mn, min(mx, vf))
            slider.blockSignals(True)
            slider.setValue(int(round((vf - mn) / step)))
            slider.blockSignals(False)
            self._set_value(key, int(vf) if t == "int" else float(vf), emit=True)

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

        lay.addWidget(slider, 2)
        lay.addWidget(spin, 1)
        return wrap

    def _set_value(self, key: str, value: Any, emit: bool = True) -> None:
        self._values[key] = value
        if emit:
            self._emit_change()


# ---------------------------
# Main window
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Nate's 3D Renderer")
        self.resize(1200, 760)

        self.proc: Optional[QProcess] = None

        self.stage_values: Dict[str, Dict[str, Any]] = {}
        self.stage_schemas: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._updating_stages_edit = False

        # which library list is “active” for the shared Add buttons
        self._active_library: str = "normal"  # "normal" | "builder"

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 8, 8, 8)
        hl.setSpacing(12)

        title_col = QVBoxLayout()
        t = QLabel("Nate's 3D Renderer")
        t.setObjectName("TitleLabel")
        s = QLabel("Blocks • Dynamic Params • PyQt5 UI • pygame/OpenGL Runtime")
        s.setObjectName("SubLabel")
        title_col.addWidget(t)
        title_col.addWidget(s)
        hl.addLayout(title_col, 1)

        self.btn_run_top = QPushButton("▶ Launch")
        self.btn_run_top.clicked.connect(self.on_run)
        self.btn_stop_top = QPushButton("■ Stop")
        self.btn_stop_top.clicked.connect(self.on_stop)
        self.btn_stop_top.setEnabled(False)

        hl.addWidget(self.btn_run_top)
        hl.addWidget(self.btn_stop_top)

        root_layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(self._build_render_tab(), "Render")
        tabs.addTab(self._build_blocks_tab(), "Blocks")
        root_layout.addWidget(tabs, 1)

        self.setCentralWidget(root)

        app = QApplication.instance()
        if app:
            app.setStyleSheet(DARK_QSS)
            font = QFont()
            font.setPointSize(10)
            app.setFont(font)

        # defaults (can be overridden by restore)
        self.stages_edit.setText("glctx|glprogram|glstate|glaxes|glcube|glloop")
        self.preset.setCurrentIndex(0)

        # restore session state (stages + params + selected stage)
        self.restore_session()

        # build lists & param editor after restore
        self.rebuild_stages()
        self._refresh_libraries()

    # -------------------- Render tab --------------------
    def _build_render_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(10)

        left = QWidget()
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        l = QVBoxLayout(left)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(10)

        grp_pipe = QGroupBox("Pipeline")
        gf = QFormLayout(grp_pipe)
        gf.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.preset = QComboBox()
        self.preset.addItems([
            "Cube + Axes (Default)",
            "Cube Only (Minimal)",
            "Wireframe Debug",
            "Two Cubes (Aliased)",
            "Custom",
        ])
        self.preset.currentIndexChanged.connect(self.apply_preset)

        self.stages_edit = QLineEdit()
        self.stages_edit.setPlaceholderText("Example: glctx|glprogram|glcube|glloop")
        self.stages_edit.textChanged.connect(self._on_stages_edit_changed)

        gf.addRow(QLabel("Preset"), self.preset)
        gf.addRow(QLabel("Stages"), self.stages_edit)

        # ---- Dual Block Library (Normal vs Builder) w/ shared buttons ----
        grp_lib = QGroupBox("Block Library")
        lib_outer = QVBoxLayout(grp_lib)
        lib_outer.setContentsMargins(10, 10, 10, 10)
        lib_outer.setSpacing(8)

        split_lib = QSplitter(Qt.Horizontal)

        # Normal pane
        normal_pane = QWidget()
        nlay = QVBoxLayout(normal_pane)
        nlay.setContentsMargins(0, 0, 0, 0)
        nlay.setSpacing(6)

        ntitle = QLabel("Normal Blocks")
        ntitle.setObjectName("SubLabel")
        self.lib_search_normal = QLineEdit()
        self.lib_search_normal.setPlaceholderText("Search normal blocks…")
        self.lib_search_normal.textChanged.connect(self._refresh_libraries)

        self.lib_list_normal = QListWidget()
        self.lib_list_normal.itemDoubleClicked.connect(lambda _=None: self.on_add_block())
        self.lib_list_normal.currentRowChanged.connect(lambda _=None: self._set_active_library("normal"))
        self.lib_list_normal.itemClicked.connect(lambda _=None: self._set_active_library("normal"))

        nlay.addWidget(ntitle)
        nlay.addWidget(self.lib_search_normal)
        nlay.addWidget(self.lib_list_normal, 1)

        # Builder pane
        builder_pane = QWidget()
        blay = QVBoxLayout(builder_pane)
        blay.setContentsMargins(0, 0, 0, 0)
        blay.setSpacing(6)

        btitle = QLabel("Builder Blocks")
        btitle.setObjectName("SubLabel")
        self.lib_search_builder = QLineEdit()
        self.lib_search_builder.setPlaceholderText("Search builder blocks…")
        self.lib_search_builder.textChanged.connect(self._refresh_libraries)

        self.lib_list_builder = QListWidget()
        self.lib_list_builder.itemDoubleClicked.connect(lambda _=None: self.on_add_block())
        self.lib_list_builder.currentRowChanged.connect(lambda _=None: self._set_active_library("builder"))
        self.lib_list_builder.itemClicked.connect(lambda _=None: self._set_active_library("builder"))

        blay.addWidget(btitle)
        blay.addWidget(self.lib_search_builder)
        blay.addWidget(self.lib_list_builder, 1)

        split_lib.addWidget(normal_pane)
        split_lib.addWidget(builder_pane)
        split_lib.setStretchFactor(0, 1)
        split_lib.setStretchFactor(1, 1)

        lib_outer.addWidget(split_lib, 1)

        # Shared buttons row (works on whichever list is active)
        lib_btn_row = QWidget()
        lbr = QHBoxLayout(lib_btn_row)
        lbr.setContentsMargins(0, 0, 0, 0)
        lbr.setSpacing(8)

        self.btn_add = QPushButton("Add →")
        self.btn_add.clicked.connect(self.on_add_block)
        self.btn_add_alias = QPushButton("Add as Alias…")
        self.btn_add_alias.clicked.connect(self.on_add_block_alias)

        lbr.addWidget(self.btn_add, 1)
        lbr.addWidget(self.btn_add_alias, 1)
        lib_outer.addWidget(lib_btn_row)

        # ---- Stage list + edit buttons ----
        grp_list = QGroupBox("Stages")
        vl = QVBoxLayout(grp_list)
        vl.setContentsMargins(10, 10, 10, 10)
        vl.setSpacing(8)

        self.stage_list = QListWidget()
        self.stage_list.currentRowChanged.connect(self.on_stage_selected)
        vl.addWidget(self.stage_list, 1)

        stage_btns = QWidget()
        sbr = QHBoxLayout(stage_btns)
        sbr.setContentsMargins(0, 0, 0, 0)
        sbr.setSpacing(8)

        self.btn_up = QPushButton("↑ Up")
        self.btn_up.clicked.connect(self.on_stage_up)
        self.btn_down = QPushButton("↓ Down")
        self.btn_down.clicked.connect(self.on_stage_down)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self.on_stage_remove)

        sbr.addWidget(self.btn_up, 1)
        sbr.addWidget(self.btn_down, 1)
        sbr.addWidget(self.btn_remove, 1)
        vl.addWidget(stage_btns)

        # Launch/export buttons row
        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(8)

        self.btn_run = QPushButton("▶ Launch")
        self.btn_run.clicked.connect(self.on_run)

        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)

        self.btn_copy = QPushButton("Copy CLI")
        self.btn_copy.clicked.connect(self.on_copy_cli)

        self.btn_export = QPushButton("Export Model…")
        self.btn_export.clicked.connect(self.on_export_model)

        self.btn_reset_stage = QPushButton("Reset Stage")
        self.btn_reset_stage.clicked.connect(self.on_reset_stage)

        br.addWidget(self.btn_run, 1)
        br.addWidget(self.btn_stop, 1)
        br.addWidget(self.btn_copy, 1)
        br.addWidget(self.btn_export, 1)
        br.addWidget(self.btn_reset_stage, 1)

        l.addWidget(grp_pipe)
        l.addWidget(grp_lib, 2)
        l.addWidget(grp_list, 2)
        l.addWidget(btn_row)

        # Right: params + log
        right_split = QSplitter(Qt.Vertical)

        self.param_editor = ParamEditor()
        self.param_editor.set_on_change(self.on_params_changed)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.param_editor)

        scroll_box = QGroupBox("Parameters")
        sb = QVBoxLayout(scroll_box)
        sb.setContentsMargins(10, 10, 10, 10)
        sb.addWidget(scroll)

        right_split.addWidget(scroll_box)

        grp_log = QGroupBox("Output")
        log_layout = QVBoxLayout(grp_log)
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_layout.setSpacing(8)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Render process output will appear here…")
        log_layout.addWidget(self.log, 1)

        right_split.addWidget(grp_log)
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right_split)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        lay.addWidget(split, 1)
        return w

    # -------------------- Blocks tab --------------------
    def _build_blocks_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        title = QLabel("Registered Blocks")
        title.setObjectName("TitleLabel")
        lay.addWidget(title)

        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        ll.addWidget(QLabel("Normal"))
        self.blocks_list_normal = QListWidget()
        for name in sorted([n for n in BLOCKS.names() if is_normal_block(n)]):
            self.blocks_list_normal.addItem(name)
        ll.addWidget(self.blocks_list_normal, 1)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        rl.addWidget(QLabel("Builder"))
        self.blocks_list_builder = QListWidget()
        for name in sorted([n for n in BLOCKS.names() if is_builder_block(n)]):
            self.blocks_list_builder.addItem(name)
        rl.addWidget(self.blocks_list_builder, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        lay.addWidget(split, 1)

        self.block_info = QPlainTextEdit()
        self.block_info.setReadOnly(True)
        lay.addWidget(self.block_info, 1)

        self.blocks_list_normal.currentTextChanged.connect(self.on_block_info)
        self.blocks_list_builder.currentTextChanged.connect(self.on_block_info)
        if self.blocks_list_normal.count() > 0:
            self.blocks_list_normal.setCurrentRow(0)
        elif self.blocks_list_builder.count() > 0:
            self.blocks_list_builder.setCurrentRow(0)

        return w

    # -------------------- library (dual) --------------------
    def _set_active_library(self, which: str) -> None:
        self._active_library = "builder" if which == "builder" else "normal"

    def _refresh_libraries(self) -> None:
        # normal
        qn = (self.lib_search_normal.text() or "").strip().lower()
        self.lib_list_normal.blockSignals(True)
        self.lib_list_normal.clear()
        normal_names = sorted([n for n in BLOCKS.names() if is_normal_block(n)])
        for n in normal_names:
            if qn and qn not in n.lower():
                continue
            self.lib_list_normal.addItem(n)
        self.lib_list_normal.blockSignals(False)

        # builder
        qb = (self.lib_search_builder.text() or "").strip().lower()
        self.lib_list_builder.blockSignals(True)
        self.lib_list_builder.clear()
        builder_names = sorted([n for n in BLOCKS.names() if is_builder_block(n)])
        for n in builder_names:
            if qb and qb not in n.lower():
                continue
            self.lib_list_builder.addItem(n)
        self.lib_list_builder.blockSignals(False)

        # pick a selection if nothing selected
        if self._active_library == "normal":
            if self.lib_list_normal.count() > 0 and self.lib_list_normal.currentRow() < 0:
                self.lib_list_normal.setCurrentRow(0)
        else:
            if self.lib_list_builder.count() > 0 and self.lib_list_builder.currentRow() < 0:
                self.lib_list_builder.setCurrentRow(0)

    def _selected_library_block(self) -> Optional[str]:
        # shared selection: prefer active list; fallback to other if empty
        if self._active_library == "builder":
            it = self.lib_list_builder.currentItem()
            if it:
                return it.text().strip()
            it2 = self.lib_list_normal.currentItem()
            return it2.text().strip() if it2 else None
        else:
            it = self.lib_list_normal.currentItem()
            if it:
                return it.text().strip()
            it2 = self.lib_list_builder.currentItem()
            return it2.text().strip() if it2 else None

    # -------------------- presets --------------------
    def apply_preset(self) -> None:
        i = self.preset.currentIndex()
        if i == 0:
            self.stages_edit.setText("glctx|glprogram|glstate|glaxes|glcube|glloop")
        elif i == 1:
            self.stages_edit.setText("glctx|glprogram|glcube|glloop")
        elif i == 2:
            self.stages_edit.setText("glctx|glprogram|glstate|glcube|glloop")
        elif i == 3:
            self.stages_edit.setText("glctx|glprogram|glstate|glaxes|glcube:c1|glcube:c2|glloop")
        else:
            pass

        self.rebuild_stages()

        if i == 2:
            self.stage_values.setdefault("glstate", {})
            self.stage_values["glstate"]["wireframe"] = True
        if i == 3:
            self.stage_values.setdefault("glcube:c1", {})
            self.stage_values.setdefault("glcube:c2", {})
            self.stage_values["glcube:c1"]["size"] = 1.0
            self.stage_values["glcube:c2"]["size"] = 2.0

        self.refresh_selected_stage_editor()

    # -------------------- stage list editing --------------------
    def _set_pipeline_tokens(self, tokens: List[str], *, select_index: Optional[int] = None) -> None:
        tokens = normalize_pipeline(tokens)
        self._updating_stages_edit = True
        try:
            self.stages_edit.setText("|".join(tokens))
        finally:
            self._updating_stages_edit = False
        self.rebuild_stages()
        if select_index is not None and 0 <= select_index < self.stage_list.count():
            self.stage_list.setCurrentRow(select_index)

    def _insert_stage_token(self, token: str, *, alias: Optional[str] = None) -> None:
        token = (token or "").strip()
        if not token:
            return

        base = token.split(":", 1)[0].strip().lower()
        if BLOCKS.get_class(base) is None:
            QMessageBox.warning(self, "Unknown block", f"Block '{base}' is not registered.")
            return

        if alias:
            alias = alias.strip()
            if any(c in alias for c in ["|", ":", " "]):
                QMessageBox.warning(self, "Bad alias", "Alias cannot contain spaces, ':' or '|'.")
                return
            token = f"{base}:{alias}"

        tokens = normalize_pipeline(parse_stage_chain(self.stages_edit.text()))

        # insertion index: after currently selected stage
        insert_at = self.stage_list.currentRow()
        if insert_at < 0:
            insert_at = len(tokens) - 1
        insert_at = max(0, min(len(tokens), insert_at + 1))

        # if adding sink -> force to end, replace existing sink
        if base in SINK_BASES:
            tokens = [t for t in tokens if stage_id_and_base(t)[1] not in SINK_BASES]
            tokens.append(token)
            self._set_pipeline_tokens(tokens, select_index=len(tokens) - 1)
            return

        # if a sink exists, insert before sink
        sink_idx = None
        for i, t in enumerate(tokens):
            if stage_id_and_base(t)[1] in SINK_BASES:
                sink_idx = i
                break
        if sink_idx is not None:
            insert_at = min(insert_at, sink_idx)

        tokens.insert(insert_at, token)
        self._set_pipeline_tokens(tokens, select_index=insert_at)

    def on_add_block(self) -> None:
        base = self._selected_library_block()
        if not base:
            return
        self._insert_stage_token(base)

    def on_add_block_alias(self) -> None:
        base = self._selected_library_block()
        if not base:
            return
        alias, ok = QInputDialog.getText(self, "Add as Alias", "Alias (example: c1):")
        if not ok:
            return
        alias = (alias or "").strip()
        if not alias:
            self._insert_stage_token(base)
            return
        self._insert_stage_token(base, alias=alias)

    def on_stage_remove(self) -> None:
        row = self.stage_list.currentRow()
        if row < 0:
            return
        tokens = normalize_pipeline(parse_stage_chain(self.stages_edit.text()))
        if not (0 <= row < len(tokens)):
            return
        del tokens[row]
        self._set_pipeline_tokens(tokens, select_index=max(0, row - 1))

    def on_stage_up(self) -> None:
        row = self.stage_list.currentRow()
        if row <= 0:
            return
        tokens = normalize_pipeline(parse_stage_chain(self.stages_edit.text()))
        if stage_id_and_base(tokens[row])[1] in SINK_BASES:
            return
        if stage_id_and_base(tokens[row - 1])[1] in SINK_BASES:
            return
        tokens[row - 1], tokens[row] = tokens[row], tokens[row - 1]
        self._set_pipeline_tokens(tokens, select_index=row - 1)

    def on_stage_down(self) -> None:
        row = self.stage_list.currentRow()
        tokens = normalize_pipeline(parse_stage_chain(self.stages_edit.text()))
        if row < 0 or row >= len(tokens) - 1:
            return
        if stage_id_and_base(tokens[row])[1] in SINK_BASES:
            return
        if stage_id_and_base(tokens[row + 1])[1] in SINK_BASES:
            return
        tokens[row + 1], tokens[row] = tokens[row], tokens[row + 1]
        self._set_pipeline_tokens(tokens, select_index=row + 1)

    # -------------------- stage/model wiring --------------------
    def _on_stages_edit_changed(self) -> None:
        if self._updating_stages_edit:
            return
        self.rebuild_stages()

    def rebuild_stages(self) -> None:
        tokens = normalize_pipeline(parse_stage_chain(self.stages_edit.text()))
        # keep editor text normalized
        self._updating_stages_edit = True
        try:
            self.stages_edit.setText("|".join(tokens))
        finally:
            self._updating_stages_edit = False

        self.stage_list.blockSignals(True)
        self.stage_list.clear()

        existing = dict(self.stage_values)
        self.stage_values = {}
        self.stage_schemas = {}

        for tok in tokens:
            stage_id, _base = stage_id_and_base(tok)
            _, schema = block_schema_for_stage(tok)

            self.stage_schemas[stage_id] = schema
            defaults = schema_defaults(schema)

            vals = dict(defaults)
            if stage_id in existing:
                vals.update(existing[stage_id])
            self.stage_values[stage_id] = vals
            self.stage_list.addItem(tok)

        self.stage_list.blockSignals(False)

        if self.stage_list.count() > 0 and self.stage_list.currentRow() < 0:
            self.stage_list.setCurrentRow(0)

        self.refresh_selected_stage_editor()

    def current_stage_token(self) -> Optional[str]:
        item = self.stage_list.currentItem()
        return item.text().strip() if item else None

    def on_stage_selected(self, _row: int) -> None:
        self.refresh_selected_stage_editor()

    def refresh_selected_stage_editor(self) -> None:
        tok = self.current_stage_token()
        if not tok:
            self.param_editor.clear()
            return
        stage_id = tok.lower()
        schema = self.stage_schemas.get(stage_id, {})
        vals = self.stage_values.get(stage_id, {})
        self.param_editor.set_stage(tok, schema, vals)

    def on_params_changed(self) -> None:
        tok = self.current_stage_token()
        if not tok:
            return
        self.stage_values[tok.lower()] = self.param_editor.values()

    def on_reset_stage(self) -> None:
        tok = self.current_stage_token()
        if not tok:
            return
        stage_id = tok.lower()
        schema = self.stage_schemas.get(stage_id, {})
        self.stage_values[stage_id] = schema_defaults(schema)
        self.refresh_selected_stage_editor()

    # -------------------- build CLI extras --------------------
    def build_cli_args(self, *, override_stages: Optional[str] = None, extra_kv: Optional[Dict[str, Any]] = None) -> \
    List[str]:
        stages = (override_stages or self.stages_edit.text() or "").strip()
        if not stages:
            stages = "glctx|glprogram|glcube|glloop"

        # When frozen, we call the same EXE with: cli pipeline ...
        frozen = bool(getattr(sys, "frozen", False))

        if frozen:
            args: List[str] = [
                "cli",
                "pipeline",
                "",
                "--extra", f"pipeline.stages={stages}",
            ]
        else:
            args = [
                os.path.join(os.path.dirname(__file__), "main.py"),
                "pipeline",
                "",
                "--extra", f"pipeline.stages={stages}",
            ]

        for stage_id, schema in self.stage_schemas.items():
            vals = self.stage_values.get(stage_id, {})
            defaults = schema_defaults(schema)
            for k, v in vals.items():
                if v == defaults.get(k, None):
                    continue
                args += ["--extra", f"{stage_id}.{k}={v}"]

        if extra_kv:
            for k, v in extra_kv.items():
                args += ["--extra", f"{k}={v}"]

        return args

    # -------------------- export model --------------------
    def _find_export_blocks(self) -> List[str]:
        names = [n for n in BLOCKS.names() if n]
        cand: List[str] = []
        for n in names:
            nl = n.lower()
            if any(x in nl for x in ("export", "exporter", "writer", "save", "dump")):
                cand.append(n)
        cand = sorted(set(cand), key=lambda x: (0 if ("obj" in x.lower() or "gltf" in x.lower()) else 1, x))
        return cand

    def _guess_output_param_key(self, block_name: str) -> Optional[str]:
        cls = BLOCKS.get_class(block_name.lower())
        if cls is None:
            return None
        schema = getattr(cls, "PARAMS", {}) or {}
        if not isinstance(schema, dict):
            return None
        preferred = ["path", "out", "outfile", "filename", "file", "output", "out_path", "dest"]
        keys = list(schema.keys())
        for k in preferred:
            if k in keys:
                return k
        for k in keys:
            kl = k.lower()
            if "path" in kl or "file" in kl or "out" in kl:
                return k
        return None

    def export_preset_json(self, path: str) -> None:
        data = {
            "stages": self.stages_edit.text().strip(),
            "stage_values": self.stage_values,
            "preset_index": int(self.preset.currentIndex()),
            "selected_stage_row": int(self.stage_list.currentRow()),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_json_safe(data), f, indent=2)
            QMessageBox.information(self, "Exported", f"Saved preset:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def on_export_model(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model / Preset",
            os.path.expanduser("~"),
            "OBJ (*.obj);;glTF (*.gltf);;Preset (*.json);;All Files (*)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            self.export_preset_json(path)
            return

        exporters = self._find_export_blocks()
        if not exporters:
            QMessageBox.warning(
                self,
                "No exporter block found",
                "No exporter-like block is registered (name contains export/save/writer).\n\n"
                "I’ll export a preset JSON instead. Add an exporter block later to produce .obj/.gltf."
            )
            self.export_preset_json(os.path.splitext(path)[0] + ".json")
            return

        # pick best exporter by extension
        default_choice = exporters[0]
        if ext == ".obj":
            for n in exporters:
                if "obj" in n.lower():
                    default_choice = n
                    break
        elif ext == ".gltf":
            for n in exporters:
                if "gltf" in n.lower():
                    default_choice = n
                    break

        exporter = default_choice
        if len(exporters) > 1:
            choice, ok = QInputDialog.getItem(
                self, "Choose Export Block", "Exporter:", exporters, exporters.index(default_choice), False
            )
            if not ok or not choice:
                return
            exporter = str(choice).strip()

        exporter_id = exporter.lower()

        # Override pipeline: remove render sink(s), append exporter as sink.
        tokens = normalize_pipeline(parse_stage_chain(self.stages_edit.text()))
        tokens = [t for t in tokens if stage_id_and_base(t)[1] not in SINK_BASES]
        tokens.append(exporter_id)
        override_stages = "|".join(tokens)

        # Try to pass output path param to exporter
        extra: Dict[str, Any] = {}
        out_key = self._guess_output_param_key(exporter_id)
        if out_key:
            extra[f"{exporter_id}.{out_key}"] = path

        if self.proc is not None:
            QMessageBox.information(self, "Already running", "Stop the current process before exporting.")
            return

        args = self.build_cli_args(override_stages=override_stages, extra_kv=extra)
        extra[f"{exporter_id}.cleanup"] = True
        extra[f"{exporter_id}.passthrough"] = False  # optional, keeps CLI output small

        self.proc = QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setArguments(args)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_proc)
        self.proc.finished.connect(self._proc_finished)

        self._append(f"> Export: {sys.executable} " + " ".join(args))
        self.proc.start()
        self._set_running(True)

    # -------------------- run process --------------------
    def _append(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip("\n"))

    def on_copy_cli(self) -> None:
        args = self.build_cli_args()
        cmd = f'{sys.executable} ' + " ".join([f'"{a}"' if " " in a else a for a in args])
        QApplication.clipboard().setText(cmd)
        QMessageBox.information(self, "Copied", "CLI command copied to clipboard.")

    def on_run(self) -> None:
        if self.proc is not None:
            QMessageBox.information(self, "Already running", "A render process is already running.")
            return

        args = self.build_cli_args()

        self.proc = QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setArguments(args)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_proc)
        self.proc.finished.connect(self._proc_finished)

        self._append(f"> Launch: {sys.executable} " + " ".join(args))
        self.proc.start()
        self._set_running(True)

    def _read_proc(self) -> None:
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        if data.strip():
            self._append(data)

    def _proc_finished(self) -> None:
        self._append("> Process finished.")
        self.proc = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_run_top.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_stop_top.setEnabled(running)
        self.btn_export.setEnabled(not running)

    def on_stop(self) -> None:
        if not self.proc:
            return
        self._append("> Terminating process...")
        self.proc.terminate()
        if not self.proc.waitForFinished(2000):
            self.proc.kill()

    # -------------------- blocks tab info --------------------
    def on_block_info(self, name: str) -> None:
        name = (name or "").strip().lower()
        cls = BLOCKS.get_class(name)
        if cls is None:
            self.block_info.setPlainText("")
            return
        schema = getattr(cls, "PARAMS", {}) or {}
        lines = [f"Block: {name}", f"Module: {getattr(cls, '__module__', '')}", ""]
        if not schema:
            lines.append("(No PARAMS schema exposed)")
        else:
            for k in sorted(schema.keys()):
                spec = schema[k] or {}
                lines.append(
                    f"- {k}: type={spec.get('type')} default={spec.get('default')} "
                    f"min={spec.get('min')} max={spec.get('max')} step={spec.get('step')}"
                )
        self.block_info.setPlainText("\n".join(lines))

    # -------------------- session save/restore --------------------
    def restore_session(self) -> None:
        st = load_ui_state()
        if not st:
            return

        try:
            stages = str(st.get("stages", "") or "").strip()
            if stages:
                self.stages_edit.setText(stages)

            sv = st.get("stage_values", {})
            if isinstance(sv, dict):
                # stored keys might not be lower; normalize
                self.stage_values = {str(k).lower(): (v if isinstance(v, dict) else {}) for k, v in sv.items()}

            pi = st.get("preset_index", None)
            if isinstance(pi, int) and 0 <= pi < self.preset.count():
                self.preset.setCurrentIndex(pi)

            self._restore_selected_stage_row = st.get("selected_stage_row", None)
        except Exception:
            self._restore_selected_stage_row = None

    def save_session(self) -> None:
        st = {
            "stages": self.stages_edit.text().strip(),
            "stage_values": self.stage_values,
            "preset_index": int(self.preset.currentIndex()),
            "selected_stage_row": int(self.stage_list.currentRow()),
        }
        save_ui_state(st)

    def closeEvent(self, ev) -> None:
        try:
            self.save_session()
        finally:
            super().closeEvent(ev)


def gui_main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Nate's 3D Renderer")
    app.setStyleSheet(DARK_QSS)
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    win = MainWindow()
    win.show()

    try:
        row = getattr(win, "_restore_selected_stage_row", None)
        if isinstance(row, int) and 0 <= row < win.stage_list.count():
            win.stage_list.setCurrentRow(row)
    except Exception:
        pass

    sys.exit(app.exec_())


if __name__ == "__main__":
    _maybe_run_cli_mode()  # ✅ handle CLI mode FIRST
    gui_main()