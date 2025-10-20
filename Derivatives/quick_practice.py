
#!/usr/bin/env python3
"""
Quick Derivatives Practice: Random functions with derivative stepping

Usage:
  python quick_practice.py

Controls (when the plot window is focused):
  Space / Enter / Right Arrow: advance view (f → f' → f'')
  Left Arrow / Backspace: go back (f'' → f' → f)
  N: new random function
  G: toggle grid
  S: save current view as PNG
  H or ?: show/hide help
  Q or Esc: quit

Notes:
  - Functions are randomized from several smooth families (polynomials, trig mixes, exp/log combos).
  - Derivatives are computed symbolically with SymPy and evaluated numerically with NumPy.
  - The domain is chosen to avoid obvious singularities, but not all edge-cases are removed.
    If a function is ill-behaved on the sampled grid, the generator retries a few times.
"""

import random
import math
import time
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List

import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt

# ========= Random Function Generator =========

x = sp.symbols('x')

@dataclass
class FnSpec:
    expr: sp.Expr
    f: Callable[[np.ndarray], np.ndarray]
    f1: Callable[[np.ndarray], np.ndarray]
    f2: Callable[[np.ndarray], np.ndarray]
    family: str
    desc: str
    domain: Tuple[float, float]


def _lamb(expr: sp.Expr) -> Callable[[np.ndarray], np.ndarray]:
    """Numpy-aware callable for expr(x)."""
    return sp.lambdify(x, expr, 'numpy')


def _finite_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


def _poly_family() -> Tuple[sp.Expr, str]:
    # Degree 2–5 polynomial with moderate coefficients
    deg = random.randint(2, 5)
    coeffs = [random.uniform(-3.0, 3.0) for _ in range(deg + 1)]
    expr = sum(coeffs[i] * x**i for i in range(deg + 1))
    desc = f"Polynomial deg {deg}: " + " + ".join([f"{coeffs[i]:+.2f}·x^{i}" for i in range(deg, -1, -1)])
    return sp.simplify(expr), desc


def _trig_family() -> Tuple[sp.Expr, str]:
    # a*sin(bx + c) + d*cos(ex + f) with small integers
    a = random.uniform(-3, 3)
    b = random.choice([1, 2, 3, 4])
    c = random.uniform(-math.pi, math.pi)
    d = random.uniform(-3, 3)
    e = random.choice([1, 2, 3, 4])
    f = random.uniform(-math.pi, math.pi)
    expr = a*sp.sin(b*x + c) + d*sp.cos(e*x + f)
    desc = f"Trig mix: {a:.2f}·sin({b}x{c:+.2f}) + {d:.2f}·cos({e}x{f:+.2f})"
    return sp.simplify(expr), desc


def _exp_log_family() -> Tuple[sp.Expr, str]:
    # Smooth combos like e^(ax) * sin(bx) or ln(x+shift) * sin / cos
    choice = random.choice(['exp_sin', 'exp_cos', 'log_sin', 'log_poly'])
    if choice in ('exp_sin', 'exp_cos'):
        a = random.uniform(-1.0, 1.0)
        b = random.choice([1, 2, 3])
        base = sp.exp(a*x)
        trig = sp.sin(b*x) if choice == 'exp_sin' else sp.cos(b*x)
        expr = base * trig
        desc = f"exp({a:.2f}x) * {'sin' if choice=='exp_sin' else 'cos'}({b}x)"
        # domain is all real numbers
        return sp.simplify(expr), desc
    elif choice == 'log_sin':
        # ln(x+shift) * sin(bx); ensure x+shift > 0 on domain by selecting shift and domain later
        b = random.choice([1, 2, 3])
        shift = random.uniform(0.5, 3.0)
        expr = sp.log(x + shift) * sp.sin(b*x)
        desc = f"ln(x+{shift:.2f}) * sin({b}x)"
        return sp.simplify(expr), desc
    else:  # log_poly
        shift = random.uniform(0.5, 3.0)
        k = random.uniform(-2.0, 2.0)
        m = random.choice([1, 2, 3])
        expr = k * sp.log(x + shift) + (x**m) / (m + 1)
        desc = f"{k:.2f}·ln(x+{shift:.2f}) + x^{m}/{m+1}"
        return sp.simplify(expr), desc


def _rational_family() -> Tuple[sp.Expr, str, Tuple[float, float]]:
    # (P(x)) / (Q(x)) with no obvious poles in chosen domain; pick domain after building Q
    deg_p = random.randint(1, 3)
    deg_q = random.randint(1, 2)
    P = sum(random.uniform(-2, 2)*x**i for i in range(deg_p + 1))
    # Keep Q away from zero by adding a positive offset and using small coefficients
    Q = sum(random.uniform(-1, 1)*x**i for i in range(deg_q + 1)) + random.uniform(1.5, 3.0)
    expr = sp.simplify(P / Q)
    desc = f"Rational: P_deg{deg_p}(x) / Q_deg{deg_q}(x)+offset"
    # Domain will be chosen later; return a default wide domain
    return expr, desc, (-5.0, 5.0)


def _choose_domain(expr: sp.Expr) -> Tuple[float, float]:
    # Decide a domain that avoids singularities for log-like terms.
    syms = list(expr.free_symbols)
    if sp.log(x) in sp.preorder_traversal(expr) or any(isinstance(node, sp.log) for node in sp.preorder_traversal(expr)):
        # keep x > lower bound - 10% margin
        # Find minimal shift from patterns ln(x+shift)
        min_shift = 0.5
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.log):
                arg = node.args[0]
                try:
                    # Roughly estimate where arg > 0. We choose a right-shifted domain.
                    # If arg is x + c, then x > -c
                    poly = sp.Poly(arg, x)
                    # If linear: ax + b > 0 -> choose a range satisfying it
                    if poly.degree() == 1:
                        a = float(poly.all_coeffs()[0])
                        b = float(poly.all_coeffs()[1])
                        if a > 0:
                            bound = -b / a + 0.5
                            min_shift = max(min_shift, bound)
                        else:
                            # a < 0 -> inequality flips; choose a small bounded left range
                            min_shift = max(min_shift, -10.0)
                except Exception:
                    pass
        return (min_shift, min_shift + 10.0)
    # default
    return (-5.0, 5.0)


def generate_function(max_retries: int = 10) -> FnSpec:
    """
    Try to build a random, numerically well-behaved function with its first two derivatives.
    """
    for _ in range(max_retries):
        family = random.choices(
            population=['poly', 'trig', 'exp_log', 'rational'],
            weights=[0.35, 0.30, 0.25, 0.10],
            k=1
        )[0]

        if family == 'poly':
            expr, desc = _poly_family()
            domain = (-5.0, 5.0)
        elif family == 'trig':
            expr, desc = _trig_family()
            domain = (-2*math.pi, 2*math.pi)
        elif family == 'exp_log':
            expr, desc = _exp_log_family()
            domain = _choose_domain(expr)
        else:
            expr, desc, domain = _rational_family()

        try:
            d1 = sp.diff(expr, x)
            d2 = sp.diff(d1, x)
            f = _lamb(expr)
            f1 = _lamb(d1)
            f2 = _lamb(d2)
        except Exception:
            continue

        # Quick numeric sanity check
        xs = np.linspace(domain[0], domain[1], 800)
        try:
            y = f(xs)
            y1 = f1(xs)
            y2 = f2(xs)
            ok = _finite_mask(y).mean() > 0.95 and _finite_mask(y1).mean() > 0.95 and _finite_mask(y2).mean() > 0.95
            if not ok:
                continue
            # Additional overflow sanity
            if np.nanmax(np.abs(y)) > 1e6 or np.nanmax(np.abs(y1)) > 1e7 or np.nanmax(np.abs(y2)) > 1e8:
                continue
        except Exception:
            continue

        return FnSpec(expr=expr, f=f, f1=f1, f2=f2, family=family, desc=desc, domain=domain)

    # Fallback simple polynomial if all retries fail
    expr = x**3 - 2*x
    d1 = sp.diff(expr, x)
    d2 = sp.diff(d1, x)
    return FnSpec(expr=expr, f=_lamb(expr), f1=_lamb(d1), f2=_lamb(d2),
                  family='fallback', desc='Fallback: x^3 - 2x', domain=(-5.0, 5.0))

# ========= Visualization App =========

class CalcVisualizer:
    def __init__(self):
        self.spec: FnSpec = generate_function()
        self.mode: int = 0  # 0=f, 1=f', 2=f''
        self.grid_on: bool = True
        self.help_on: bool = True
        self.fig, self.ax = plt.subplots()
        self.text_help = None
        self._plot_current()

        try:
            self.fig.canvas.manager.set_window_title("Derivatives (Quick Practice) | Denver Clark 2025 | seperet.com")
        except Exception:
            pass

        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def _title(self) -> str:
        name = {0: "f(x)", 1: "f'(x)", 2: "f''(x)"}[self.mode]
        return f"{name} — {self.spec.family} | {self.spec.desc}"

    def _compute(self, xs: np.ndarray) -> np.ndarray:
        if self.mode == 0:
            return self.spec.f(xs)
        elif self.mode == 1:
            return self.spec.f1(xs)
        else:
            return self.spec.f2(xs)

    def _plot_current(self):
        self.ax.clear()
        a, b = self.spec.domain
        xs = np.linspace(a, b, 1200)
        ys = self._compute(xs)

        # Mask non-finite values
        mask = np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]

        self.ax.plot(xs, ys, lw=2)
        self.ax.set_xlim(a, b)
        self.ax.grid(self.grid_on, which='both', alpha=0.4)
        self.ax.set_title(self._title())
        self.ax.set_xlabel("x")
        self.ax.set_ylabel({0: "f(x)", 1: "f'(x)", 2: "f''(x)"}[self.mode])

        if self.help_on:
            self._draw_help()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _draw_help(self):
        help_text = (
            "Controls:\n"
            "  Space/Enter/Right → next view   |   Left/Backspace → previous view\n"
            "  N → new random function         |   G → toggle grid\n"
            "  S → save PNG                    |   H/? → toggle help\n"
            "  Q/Esc → quit"
        )
        # Place a semi-transparent box in the corner
        self.text_help = self.ax.text(
            0.99, 0.02, help_text,
            transform=self.ax.transAxes,
            ha='right', va='bottom',
            fontsize=9,
            bbox=dict(boxstyle='round', alpha=0.15, ec='none', pad=0.4)
        )

    def _advance(self, step: int):
        self.mode = (self.mode + step) % 3
        self._plot_current()

    def _new_function(self):
        self.spec = generate_function()
        self.mode = 0
        self._plot_current()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in (' ', 'enter', 'right'):
            self._advance(1)
        elif key in ('left', 'backspace'):
            self._advance(-1)
        elif key == 'n':
            self._new_function()
        elif key == 'g':
            self.grid_on = not self.grid_on
            self._plot_current()
        elif key in ('h', '?'):
            self.help_on = not self.help_on
            self._plot_current()
        elif key == 's':
            ts = time.strftime("%Y%m%d-%H%M%S")
            name = {0: "f", 1: "f1", 2: "f2"}[self.mode]
            fname = f"calc_view_{name}_{ts}.png"
            self.fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"Saved {fname}")
        elif key in ('q', 'escape'):
            plt.close(self.fig)

def main():
    app = CalcVisualizer()
    plt.show()

if __name__ == "__main__":
    main()
