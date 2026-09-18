"""ANSI terminal renderer: the Doom frame as half-block pixels plus a side panel."""

from __future__ import annotations

import shutil
import sys

import numpy as np

ESC = "\x1b["
HIDE, SHOW, HOME, CLEAR = ESC + "?25l", ESC + "?25h", ESC + "H", ESC + "2J"
RESET = ESC + "0m"


def _downsample(frame: np.ndarray, cols: int, rows_px: int) -> np.ndarray:
    H, W, _ = frame.shape
    ys = (np.arange(rows_px) * H // rows_px)
    xs = (np.arange(cols) * W // cols)
    return frame[ys][:, xs]


def frame_lines(frame: np.ndarray, cols: int) -> list[str]:
    """Render an RGB frame into `cols` columns of ▀ half-blocks, keeping the 4:3 aspect."""
    rows = max(2, int(cols * 3 / 4 / 2)) * 2  # pixel rows, even
    small = _downsample(frame, cols, rows).astype(int)
    out = []
    for y in range(0, rows, 2):
        top, bot = small[y], small[y + 1]
        parts = []
        for x in range(cols):
            tr, tg, tb = top[x]
            br, bg, bb = bot[x]
            parts.append(f"{ESC}38;2;{tr};{tg};{tb}m{ESC}48;2;{br};{bg};{bb}m▀")
        out.append("".join(parts) + RESET)
    return out


def bar(p: float, width: int) -> str:
    n = int(round(p * width))
    return "█" * n + "░" * (width - n)


class Screen:
    def __init__(self) -> None:
        self.out = sys.stdout
        self.first = True

    def __enter__(self) -> "Screen":
        self.out.write(HIDE + CLEAR)
        return self

    def __exit__(self, *exc) -> None:
        self.out.write(RESET + SHOW + "\n")
        self.out.flush()

    def draw(self, frame: np.ndarray, panel: list[str]) -> None:
        term = shutil.get_terminal_size((120, 40))
        panel_w = 44
        cols = max(40, min(96, term.columns - panel_w - 1))
        left = frame_lines(frame, cols)
        n = max(len(left), len(panel))
        lines = []
        for i in range(n):
            l = left[i] if i < len(left) else " " * cols
            r = panel[i] if i < len(panel) else ""
            lines.append(l + " " + r[: panel_w - 1] + ESC + "K")
        self.out.write(HOME + "\n".join(lines) + ESC + "J")
        self.out.flush()
