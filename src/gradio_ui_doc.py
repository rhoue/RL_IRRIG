"""
Gradio-native documentation renderer.

This module reuses the existing documentation text from utils_ui_doc.py
by executing its functions with a lightweight Streamlit shim that captures
rendered blocks and then exposes them as Gradio components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import base64
import io

import importlib
import sys
import types

import gradio as gr


@dataclass
class DocBlock:
    kind: str
    content: Any
    meta: Dict[str, Any] = field(default_factory=dict)


class _ExpanderShim:
    def __init__(self, parent: "StreamlitDocShim", label: str, expanded: bool = False) -> None:
        self.parent = parent
        self.label = label
        self.expanded = expanded
        self.blocks: List[DocBlock] = []

    def __enter__(self) -> "StreamlitDocShim":
        self.parent._push(self)
        return self.parent

    def __exit__(self, exc_type, exc, tb) -> None:
        self.parent._pop()
        self.parent._append(DocBlock("expander", self.blocks, {"label": self.label, "expanded": self.expanded}))


class _ColumnShim:
    def __init__(self, parent: "StreamlitDocShim") -> None:
        self.parent = parent
        self.blocks: List[DocBlock] = []

    def __enter__(self) -> "StreamlitDocShim":
        self.parent._push(self)
        return self.parent

    def __exit__(self, exc_type, exc, tb) -> None:
        self.parent._pop()


class StreamlitDocShim:
    def __init__(self) -> None:
        self.blocks: List[DocBlock] = []
        self._stack: List[Any] = []

    def _append(self, block: DocBlock) -> None:
        if self._stack:
            self._stack[-1].blocks.append(block)
        else:
            self.blocks.append(block)

    def _push(self, ctx: Any) -> None:
        self._stack.append(ctx)

    def _pop(self) -> None:
        if self._stack:
            self._stack.pop()

    def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
        self._append(DocBlock("html" if unsafe_allow_html else "markdown", body))

    def caption(self, body: str) -> None:
        self._append(DocBlock("caption", body))

    def info(self, body: str, icon: Optional[str] = None) -> None:
        self._append(DocBlock("info", body))

    def divider(self) -> None:
        self._append(DocBlock("divider", ""))

    def expander(self, label: str, expanded: bool = False) -> _ExpanderShim:
        return _ExpanderShim(self, label, expanded)

    def columns(self, count: int) -> List[_ColumnShim]:
        cols = [_ColumnShim(self) for _ in range(count)]
        self._append(DocBlock("columns", cols))
        return cols

    def image(self, path: str, width: Optional[int] = None) -> None:
        style = f' style="width:{width}px;"' if width else ""
        html = f'<img src="{path}"{style} />'
        self._append(DocBlock("html", html))

    def pyplot(self, fig, clear_figure: bool = False) -> None:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        if clear_figure:
            fig.clf()
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        html = f'<img src="data:image/png;base64,{data}" style="max-width:100%;height:auto;" />'
        self._append(DocBlock("html", html))


def _load_doc_module(shim: StreamlitDocShim, module_path: str):
    module = types.ModuleType("streamlit")
    module.markdown = shim.markdown
    module.caption = shim.caption
    module.info = shim.info
    module.divider = shim.divider
    module.expander = shim.expander
    module.columns = shim.columns
    module.image = shim.image
    module.pyplot = shim.pyplot
    sys.modules["streamlit"] = module
    return importlib.import_module(module_path)


def _collect_blocks(module_path: str, fn_name: str, *args, **kwargs) -> List[DocBlock]:
    shim = StreamlitDocShim()
    module = _load_doc_module(shim, module_path)
    fn = getattr(module, fn_name)
    fn(*args, **kwargs)
    return shim.blocks


def _render_blocks(blocks: List[DocBlock]) -> None:
    for block in blocks:
        if block.kind == "markdown":
            gr.Markdown(block.content)
        elif block.kind == "html":
            gr.HTML(block.content)
        elif block.kind == "caption":
            gr.Markdown(f"<span style='color:#666;font-size:0.9em'>{block.content}</span>")
        elif block.kind == "info":
            gr.Markdown(f"> {block.content}")
        elif block.kind == "divider":
            gr.HTML("<hr/>")
        elif block.kind == "expander":
            label = block.meta.get("label", "Details")
            expanded = block.meta.get("expanded", False)
            with gr.Accordion(label, open=expanded):
                _render_blocks(block.content)
        elif block.kind == "columns":
            cols = block.content
            with gr.Row():
                for col in cols:
                    with gr.Column():
                        _render_blocks(col.blocks)


def render_doc(fn_name: str) -> None:
    blocks = _collect_blocks("src.utils_ui_doc", fn_name)
    _render_blocks(blocks)


def render_doc_from_module(module_path: str, fn_name: str, *args, **kwargs) -> None:
    blocks = _collect_blocks(module_path, fn_name, *args, **kwargs)
    _render_blocks(blocks)


def _blocks_to_markdown(blocks: List[DocBlock]) -> str:
    parts: List[str] = []
    for block in blocks:
        if block.kind == "markdown":
            parts.append(block.content)
        elif block.kind == "html":
            parts.append(block.content)
        elif block.kind == "caption":
            parts.append(f"<span style='color:#666;font-size:0.9em'>{block.content}</span>")
        elif block.kind == "info":
            parts.append(f"> {block.content}")
        elif block.kind == "divider":
            parts.append("<hr/>")
        elif block.kind == "expander":
            label = block.meta.get("label", "Details")
            expanded = " open" if block.meta.get("expanded", False) else ""
            inner = _blocks_to_markdown(block.content)
            parts.append(f"<details{expanded}><summary>{label}</summary>\n{inner}\n</details>")
        elif block.kind == "columns":
            cols = block.content
            col_html = []
            for col in cols:
                inner = _blocks_to_markdown(col.blocks)
                col_html.append(f"<div style='flex:1;padding:0 8px;'>{inner}</div>")
            parts.append("<div style='display:flex;gap:8px;flex-wrap:wrap;'>" + "".join(col_html) + "</div>")
    return "\n\n".join(parts)


def render_doc_markdown(module_path: str, fn_name: str, *args, **kwargs) -> str:
    blocks = _collect_blocks(module_path, fn_name, *args, **kwargs)
    return _blocks_to_markdown(blocks)
