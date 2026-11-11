"""
curved_air

interferograms

streamlit_components

This module contains a custom Streamlit component for selecting a line with optional
width from an image or 2D data array.
"""

import base64
import io
from typing import Any

import streamlit as st
import streamlit.components.v2 as components
from PIL import Image

_HTML = """
<div id="root">
    <canvas id="canvas"></canvas>
    <div id="hint"></div>
</div>
"""

_CSS = """
#root {
    position: relative;
    width: 100%;
    line-height: 0;
}

#canvas {
    width: 100%;
    height: auto;
    display: block;
    cursor: crosshair;
    border-radius: var(--st-base-radius, 4px);
    border: 0px;
}

#hint {
    margin-top: 0.35rem;
    line-height: 1.2;
    font-family: var(--st-font, sans-serif);
    font-size: 0.8rem;
    color: var(--st-text-color, #555555);
    opacity: 0.75;
}
"""

_JS = """
export default function (component) {
    const { parentElement, data, setStateValue } = component

    const canvas = parentElement.querySelector("#canvas")
    const hint = parentElement.querySelector("#hint")
    if (!canvas || !hint) return

    const ctx = canvas.getContext("2d")

    const nativeWidth = data.native_width
    const nativeHeight = data.native_height
    const maxHalfWidth = data.max_half_width

    canvas.width = nativeWidth
    canvas.height = nativeHeight

    const host = parentElement.host ?? parentElement
    const accentColor =
        getComputedStyle(host).getPropertyValue("--st-primary-color").trim() ||
        "#ff4b4b"

    // Per-instance interaction state lives on the canvas element itself, so it survives
    // the exported function being re-invoked (e.g. on unrelated Streamlit reruns), while
    // actual line/width values are re-hydrated below from `data`, which is always the
    // authoritative value from Python.

    if (!canvas._state) {
        canvas._state = { image: null, imageSrc: null, dragging: null }
    }
    const state = canvas._state
    state.line = data.line
    state.halfWidth = data.half_width ?? 0

    function toCanvasPos(event) {
        const rect = canvas.getBoundingClientRect()
        const scaleX = canvas.width / rect.width
        const scaleY = canvas.height / rect.height
        return {
            x: Math.min(Math.max((event.clientX - rect.left) * scaleX, 0), canvas.width - 1),
            y: Math.min(Math.max((event.clientY - rect.top) * scaleY, 0), canvas.height - 1),
        }
    }

    function nearHandle(pos, line) {
        const radius = Math.max(6, nativeWidth / 40)
        if (Math.hypot(pos.x - line.x1, pos.y - line.y1) <= radius) return "p1"
        if (Math.hypot(pos.x - line.x2, pos.y - line.y2) <= radius) return "p2"
        return null
    }

    function render() {
        if (!state.image) return
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height)

        const line = state.line
        if (!line) {
            hint.textContent = "Click and drag to draw a line. Scroll over the image to set the line width."
            return
        }

        const dx = line.x2 - line.x1
        const dy = line.y2 - line.y1
        const length = Math.hypot(dx, dy)
        const perpX = length > 0 ? -dy / length : 0
        const perpY = length > 0 ? dx / length : 0
        const strokeWidth = Math.max(1, nativeWidth / 300)

        if (state.halfWidth > 0) {
            ctx.save()
            ctx.strokeStyle = accentColor
            ctx.globalAlpha = 0.35
            ctx.lineWidth = strokeWidth
            const dash = Math.max(4, nativeWidth / 100)
            ctx.setLineDash([dash, dash])
            for (const sign of [-1, 1]) {
                const offsetX = perpX * state.halfWidth * sign
                const offsetY = perpY * state.halfWidth * sign
                ctx.beginPath()
                ctx.moveTo(line.x1 + offsetX, line.y1 + offsetY)
                ctx.lineTo(line.x2 + offsetX, line.y2 + offsetY)
                ctx.stroke()
            }
            ctx.restore()
        }

        ctx.save()
        ctx.strokeStyle = accentColor
        ctx.globalAlpha = 0.9
        ctx.lineWidth = strokeWidth
        ctx.setLineDash([])
        ctx.beginPath()
        ctx.moveTo(line.x1, line.y1)
        ctx.lineTo(line.x2, line.y2)
        ctx.stroke()
        ctx.restore()

        const handleRadius = Math.max(3, nativeWidth / 80)
        ctx.save()
        ctx.fillStyle = "white"
        ctx.strokeStyle = accentColor
        ctx.lineWidth = Math.max(1, handleRadius / 3)
        for (const point of [
            [line.x1, line.y1],
            [line.x2, line.y2],
        ]) {
            ctx.beginPath()
            ctx.arc(point[0], point[1], handleRadius, 0, 2 * Math.PI)
            ctx.fill()
            ctx.stroke()
        }
        ctx.restore()

        hint.textContent =
            state.halfWidth > 0
            ? `Width: \u00b1${state.halfWidth}px. Drag the handles to move the line, drag elsewhere for a new one, or scroll to change the width.`
            : "Drag the handles to move the line, drag elsewhere for a new one, or scroll over the image to set its width."
}

    function commitLine() {
        const line = state.line
        if (line && Math.hypot(line.x2 - line.x1, line.y2 - line.y1) > 1) {
            setStateValue("line", line)
        }
    }

    canvas.onmousedown = (event) => {
        const pos = toCanvasPos(event)
        const line = state.line
        const handle = line ? nearHandle(pos, line) : null
        if (handle) {
            state.dragging = handle
        } else {
              state.dragging = "new"
              state.line = { x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y }
        }
        render()
    }

    canvas.onmousemove = (event) => {
        if (!state.dragging) return
        const pos = toCanvasPos(event)
        if (state.dragging === "new" || state.dragging === "p2") {
                state.line.x2 = pos.x
                state.line.y2 = pos.y
        } else if (state.dragging === "p1") {
                state.line.x1 = pos.x
                state.line.y1 = pos.y
        }
        render()
    }

    window.addEventListener("mouseup", () => {
        if (!state.dragging) return
        state.dragging = null
        commitLine()
    })

    canvas.onwheel = (event) => {
        if (!state.line) return
        event.preventDefault()
        const delta = event.deltaY < 0 ? 1 : -1
        const next = Math.min(maxHalfWidth, Math.max(0, state.halfWidth + delta))
        if (next !== state.halfWidth) {
            state.halfWidth = next
            render()
            setStateValue("half_width", next)
        }
    }

    if (state.imageSrc !== data.image_src) {
        state.imageSrc = data.image_src
        const img = new Image()
        img.onload = () => {
            state.image = img
            render()
        }
        img.src = data.image_src
        } else {
            render()
        }
    }
"""

_LINE_SELECTOR = components.component(
    "interferograms.line_selector",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def profile_selector(
    image: Image.Image,
    *,
    key: str,
    max_half_width: int | None = None,
) -> dict[str, Any]:
    """
    Display `image` and let the user draw a line on it by clicking and dragging, and
    adjust the line's width (for band/median profiles) by scrolling while hovering over
    it.

    Returns a `ComponentResult` with two attributes:
    - `line`: `None`, or a dict `{"x1", "y1", "x2", "y2"}` with the endpoints of the
      selected line, in the pixel coordinates of `image`.
    - `half_width`: the currently selected half-width, in pixels.
    """

    if not max_half_width:
        max_half_width: int = max(image.width, image.height) // 2

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_src: str = "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    component_state: dict[str, Any] = st.session_state.get(key, {})
    current_line: dict[str, float] | None = component_state.get("line")
    current_half_width: int = component_state.get("half_width", 0)

    return _LINE_SELECTOR(
        key=key,
        data={
            "image_src": image_src,
            "native_width": image.width,
            "native_height": image.height,
            "line": current_line,
            "half_width": current_half_width,
            "max_half_width": max_half_width,
        },
        default={"line": current_line, "half_width": current_half_width},
        on_line_change=lambda: None,
        on_half_width_change=lambda: None,
    )
