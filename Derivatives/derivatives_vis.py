
#!/usr/bin/env python3
"""
Derivatives Visualizer: 'derivatives_vis.py'

Run:
  python derivatives_vis.py

Adds:
- Point inspection: click to drop a probe and read f, f', f'' at that x. Toggle with I. U removes last, Shift+U clears all.
- Refined extrema/inflection: zero-finders use bisection on sign-change brackets for f' and f''.
- Preset families: hotkeys & buttons to force next function type (Poly, Trig, Exp/Log, Rational).
- Concept drills: randomized true/false statements about increasing/concavity/roots/local extrema. Toggle with J; answer Y/N.

Core views:
  0: f(x)
  1: f'(x)
  2: f''(x)
  3: ∫ f(x) dx  (numerical integral from left bound; F(a)=0)

Primary controls (focus the plot window):
  View: Space/Enter/→ next | ←/Backspace prev | 0/1/2/3 jump
  New: N   | Grid: G   | Help: H/?   | Context: C   | Save PNG: S   | Quit: Esc/Ctrl+W

  Tangent: T toggle | A/D move x0 (Shift for coarse) | Slider x0
  Markers: M toggle (extrema ●, inflection ◼ on f only; refined by bisection)
  Integral view: B shade area | sliders a/b | R reset

  Point inspect: I toggle | click to drop | U remove last | Shift+U clear all
  Preset families (for NEXT function): 1=Poly, 2=Trig, 3=Exp/Log, 4=Rational (or click side buttons)
  Quiz (f vs f' vs f''): Q toggle | then 1=f, 2=f', 3=f''
  Concept drills: J toggle | Answer Y/N | shows running score

No SciPy required.
"""
import random
import math
import time
from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---------------- Random function families ----------------

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

def _lamb(expr: sp.Expr):
    return sp.lambdify(x, expr, 'numpy')

def _poly_family():
    deg = random.randint(2, 5)
    coeffs = [random.uniform(-3.0, 3.0) for _ in range(deg+1)]
    expr = sum(coeffs[i] * x**i for i in range(deg+1))
    desc = f"Polynomial deg {deg}"
    return sp.simplify(expr), desc

def _trig_family():
    a = random.uniform(-3, 3); b = random.choice([1,2,3,4]); c = random.uniform(-math.pi, math.pi)
    d = random.uniform(-3, 3); e = random.choice([1,2,3,4]); f = random.uniform(-math.pi, math.pi)
    expr = a*sp.sin(b*x + c) + d*sp.cos(e*x + f)
    desc = "Trig mix"
    return sp.simplify(expr), desc

def _exp_log_family():
    choice = random.choice(['exp_sin','exp_cos','log_sin','log_poly'])
    if choice in ('exp_sin','exp_cos'):
        a = random.uniform(-1.0,1.0); b = random.choice([1,2,3])
        expr = sp.exp(a*x)*(sp.sin(b*x) if choice=='exp_sin' else sp.cos(b*x))
        desc = f"exp({a:.2f}x)·{'sin' if choice=='exp_sin' else 'cos'}({b}x)"
        return sp.simplify(expr), desc
    elif choice=='log_sin':
        shift = random.uniform(0.5, 3.0); b = random.choice([1,2,3])
        expr = sp.log(x+shift)*sp.sin(b*x)
        desc = f"ln(x+{shift:.2f})·sin({b}x)"
        return sp.simplify(expr), desc
    else:
        shift = random.uniform(0.5, 3.0); k = random.uniform(-2.0, 2.0); m = random.choice([1,2,3])
        expr = k*sp.log(x+shift) + (x**m)/(m+1)
        desc = f"{k:.2f}·ln(x+{shift:.2f}) + x^{m}/{m+1}"
        return sp.simplify(expr), desc

def _rational_family():
    deg_p = random.randint(1,3); deg_q = random.randint(1,2)
    P = sum(random.uniform(-2,2)*x**i for i in range(deg_p+1))
    Q = sum(random.uniform(-1,1)*x**i for i in range(deg_q+1)) + random.uniform(1.5,3.0)
    expr = sp.simplify(P/Q)
    desc = f"Rational P_deg{deg_p}/Q_deg{deg_q}+offset"
    return expr, desc

def _choose_domain(expr: sp.Expr):
    has_log = any(isinstance(node, sp.log) for node in sp.preorder_traversal(expr))
    if has_log:
        return (0.5, 10.5)
    if any(isinstance(node, (sp.sin, sp.cos)) for node in sp.preorder_traversal(expr)):
        return (-2*math.pi, 2*math.pi)
    return (-5.0, 5.0)

def _finite_mask(*arrays):
    ok = True
    for A in arrays:
        ok &= np.isfinite(A).mean() > 0.95
    return ok

def generate_function(force_family: Optional[str]=None, max_retries=12) -> FnSpec:
    for _ in range(max_retries):
        if force_family:
            family = force_family
        else:
            family = random.choices(['poly','trig','exp_log','rational'], weights=[0.35,0.30,0.25,0.10])[0]
        if family=='poly':
            expr, desc = _poly_family(); domain=(-5,5)
        elif family=='trig':
            expr, desc = _trig_family(); domain=(-2*math.pi, 2*math.pi)
        elif family=='exp_log':
            expr, desc = _exp_log_family(); domain=_choose_domain(expr)
        else:
            expr, desc = _rational_family(); domain=(-5,5)

        try:
            d1 = sp.diff(expr, x); d2 = sp.diff(d1, x)
            f  = _lamb(expr); f1 = _lamb(d1); f2 = _lamb(d2)
        except Exception:
            continue

        xs = np.linspace(domain[0], domain[1], 1000)
        try:
            y, y1, y2 = f(xs), f1(xs), f2(xs)
            if not _finite_mask(y, y1, y2): continue
            if np.nanmax(np.abs(y))>1e6 or np.nanmax(np.abs(y1))>1e7 or np.nanmax(np.abs(y2))>1e8:
                continue
            return FnSpec(expr=expr, f=f, f1=f1, f2=f2, family=family, desc=str(desc), domain=domain)
        except Exception:
            continue

    expr = x**3 - 2*x
    d1 = sp.diff(expr, x); d2 = sp.diff(d1, x)
    return FnSpec(expr=expr, f=_lamb(expr), f1=_lamb(d1), f2=_lamb(d2),
                  family='fallback', desc='x^3-2x', domain=(-5,5))

# ---------------- Utilities ----------------

def cumtrapz(y, x):
    dx = np.diff(x)
    mid = 0.5*(y[:-1]+y[1:])*dx
    F = np.zeros_like(y)
    F[1:] = np.cumsum(mid)
    return F

def zero_brackets(xs, ys):
    """Return brackets [x_i, x_{i+1}] where ys changes sign or hits 0."""
    signs = np.sign(ys)
    sdiff = signs[1:] * signs[:-1]
    idx = np.where(sdiff <= 0)[0]
    brackets = []
    for i in idx:
        y0, y1 = ys[i], ys[i+1]
        if not np.isfinite(y0) or not np.isfinite(y1): continue
        # If flat segment includes zero, make a tiny bracket
        if y0 == 0 and y1 == 0:
            xm = 0.5*(xs[i]+xs[i+1])
            eps = (xs[-1]-xs[0])*1e-4
            brackets.append((xm-eps, xm+eps))
        else:
            brackets.append((xs[i], xs[i+1]))
    return brackets

def bisection_refine(func, a, b, iters=20):
    """Refine root of func in [a,b] assuming sign change. Returns midpoint if invalid."""
    fa = func(a); fb = func(b)
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa*fb > 0:
        return 0.5*(a+b)
    lo, hi = a, b
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        fm = func(mid)
        if not np.isfinite(fm):
            break
        if fa*fm <= 0:
            hi = mid; fb = fm
        else:
            lo = mid; fa = fm
    return 0.5*(lo+hi)

# ---------------- App ----------------

class CalcVizUltra:
    def __init__(self):
        self.spec: FnSpec = generate_function()
        self.mode = 0  # 0:f, 1:f', 2:f'', 3:integral
        self.fig, self.ax = plt.subplots(figsize=(9.6,6.0))
        try:
            self.fig.canvas.manager.set_window_title("Derivatives Visualizer | Denver Clark 2025 | seperet.com")
        except Exception:
            pass
        plt.subplots_adjust(left=0.08, right=0.78, bottom=0.22, top=0.92)

        # state toggles
        self.grid_on = True
        self.show_help = True
        self.show_context = False
        self.show_markers = True
        self.show_tangent = False
        self.shade_on = False
        self.inspect_on = False
        self.quiz_mode = False
        self.drill_mode = False

        # quiz state
        self.quiz_answer = 0; self.quiz_score = 0; self.quiz_attempts = 0

        # drill state
        self.drill_score = 0; self.drill_attempts = 0
        self.drill_prompt = "" ; self.drill_truth = False

        # next function family override
        self.next_family: Optional[str] = None

        # domain & caches
        self.N = 1600
        self._rebuild_domain()

        # UI
        self._build_sliders()
        self._build_buttons()

        # Events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self._plot_all()

    # ----- Data prep -----

    def _rebuild_domain(self):
        a, b = self.spec.domain
        self.xs = np.linspace(a, b, self.N)
        self.y  = self.spec.f(self.xs)
        self.y1 = self.spec.f1(self.xs)
        self.y2 = self.spec.f2(self.xs)
        self.F  = cumtrapz(self.y, self.xs)

        self.x0 = 0.5*(a+b)
        self.a_int, self.b_int = a, b

        # refined markers
        ext_br = zero_brackets(self.xs, self.y1)
        inf_br = zero_brackets(self.xs, self.y2)
        self.extrema = [bisection_refine(lambda z: np.interp(z, self.xs, self.y1), a, b) for (a,b) in ext_br]
        self.inflex  = [bisection_refine(lambda z: np.interp(z, self.xs, self.y2), a, b) for (a,b) in inf_br]

        # inspection points
        self.inspect_pts: List[float] = []

    # ----- UI -----

    def _build_sliders(self):
        axcolor = 'lightgoldenrodyellow'
        self.ax_x0 = plt.axes([0.08, 0.12, 0.62, 0.03], facecolor=axcolor)
        self.s_x0 = Slider(self.ax_x0, 'x0', self.spec.domain[0], self.spec.domain[1], valinit=self.x0)

        self.ax_a = plt.axes([0.08, 0.08, 0.62, 0.03], facecolor=axcolor)
        self.s_a = Slider(self.ax_a, 'a', self.spec.domain[0], self.spec.domain[1], valinit=self.a_int)

        self.ax_b = plt.axes([0.08, 0.04, 0.62, 0.03], facecolor=axcolor)
        self.s_b = Slider(self.ax_b, 'b', self.spec.domain[0], self.spec.domain[1], valinit=self.b_int)

        self.s_x0.on_changed(self._on_slider_x0)
        self.s_a.on_changed(self._on_slider_bounds)
        self.s_b.on_changed(self._on_slider_bounds)

    def _build_buttons(self):
        # left column buttons (function controls)
        self.btn_new  = Button(plt.axes([0.81, 0.86, 0.16, 0.055]), 'New (N)')
        self.btn_help = Button(plt.axes([0.81, 0.79, 0.16, 0.055]), 'Help (H/?)')
        self.btn_ctx  = Button(plt.axes([0.81, 0.72, 0.16, 0.055]), 'Context (C)')
        self.btn_grid = Button(plt.axes([0.81, 0.65, 0.16, 0.055]), 'Grid (G)')
        self.btn_mark = Button(plt.axes([0.81, 0.58, 0.16, 0.055]), 'Markers (M)')
        self.btn_tan  = Button(plt.axes([0.81, 0.51, 0.16, 0.055]), 'Tangent (T)')
        self.btn_save = Button(plt.axes([0.81, 0.44, 0.16, 0.055]), 'Save PNG (S)')

        # middle column (families)
        self.btn_poly = Button(plt.axes([0.81, 0.34, 0.075, 0.055]), 'Poly (1)')
        self.btn_trig = Button(plt.axes([0.895,0.34, 0.075, 0.055]), 'Trig (2)')
        self.btn_exp  = Button(plt.axes([0.81, 0.27, 0.075, 0.055]), 'Exp (3)')
        self.btn_rat  = Button(plt.axes([0.895,0.27, 0.075, 0.055]), 'Rat (4)')

        # bottom row
        self.btn_full = Button(plt.axes([0.81, 0.20, 0.16, 0.055]), 'Reset Bounds (R)')
        self.btn_ins  = Button(plt.axes([0.81, 0.13, 0.16, 0.055]), 'Inspect (I)')
        self.btn_quiz = Button(plt.axes([0.81, 0.06, 0.075, 0.055]), 'Quiz (Q)')
        self.btn_drll = Button(plt.axes([0.895,0.06, 0.075, 0.055]), 'Drill (J)')

        # wire up
        self.btn_new.on_clicked(lambda e: self._new_function())
        self.btn_help.on_clicked(lambda e: self._toggle_help())
        self.btn_ctx.on_clicked(lambda e: self._toggle_context())
        self.btn_grid.on_clicked(lambda e: self._toggle_grid())
        self.btn_mark.on_clicked(lambda e: self._toggle_markers())
        self.btn_tan.on_clicked(lambda e: self._toggle_tangent())
        self.btn_save.on_clicked(lambda e: self._save_png())

        self.btn_poly.on_clicked(lambda e: self._set_family('poly'))
        self.btn_trig.on_clicked(lambda e: self._set_family('trig'))
        self.btn_exp.on_clicked(lambda e: self._set_family('exp_log'))
        self.btn_rat.on_clicked(lambda e: self._set_family('rational'))

        self.btn_full.on_clicked(lambda e: self._reset_bounds())
        self.btn_ins.on_clicked(lambda e: self._toggle_inspect())
        self.btn_quiz.on_clicked(lambda e: self._toggle_quiz())
        self.btn_drll.on_clicked(lambda e: self._toggle_drill())

    # ----- Plotting -----

    def _title(self):
        names = {0:"f(x)",1:"f'(x)",2:"f''(x)",3:"F(x)=∫f"}
        base = f"{names[self.mode]} — {self.spec.family} | {self.spec.desc}"
        if self.quiz_mode:
            base = "QUIZ — Guess: 1=f, 2=f', 3=f'' | " + base
        if self.drill_mode:
            base = "DRILL — Y/N | " + base
        return base

    def _compute_current(self):
        return {0:self.y, 1:self.y1, 2:self.y2, 3:self.F}[self.mode]

    def _plot_all(self):
        self.ax.clear()
        self.ax.grid(self.grid_on, which='both', alpha=0.35)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel({0:"f(x)",1:"f'(x)",2:"f''(x)",3:"F(x)"}[self.mode])
        self.ax.set_title(self._title())

        Y = self._compute_current()
        mask = np.isfinite(Y)
        self.ax.plot(self.xs[mask], Y[mask], lw=2)

        # Integral shading
        if self.mode==3 and self.shade_on:
            a, b = sorted([self.a_int, self.b_int])
            m = (self.xs>=a) & (self.xs<=b) & mask
            self.ax.fill_between(self.xs[m], Y[m], 0, alpha=0.15)

        # Tangent
        if self.show_tangent:
            if self.mode==0:
                slope = np.interp(self.x0, self.xs, self.y1); y0 = np.interp(self.x0, self.xs, self.y)
            elif self.mode==1:
                slope = np.interp(self.x0, self.xs, self.y2); y0 = np.interp(self.x0, self.xs, self.y1)
            elif self.mode==2:
                dy = np.gradient(self.y2, self.xs)
                slope = np.interp(self.x0, self.xs, dy); y0 = np.interp(self.x0, self.xs, self.y2)
            else:
                slope = np.interp(self.x0, self.xs, self.y); y0 = np.interp(self.x0, self.xs, self.F)
            xline = np.array([self.xs[0], self.xs[-1]])
            yline = y0 + slope*(xline - self.x0)
            self.ax.plot(xline, yline, lw=1.8, linestyle='--')
            self.ax.axvline(self.x0, lw=1, linestyle=':')
            self.ax.text(self.x0, y0, f"  x0={self.x0:.2f}\n  slope≈{slope:.3g}", fontsize=9,
                         ha='left', va='bottom', bbox=dict(boxstyle='round', alpha=0.12, ec='none'))

        # Markers
        if self.show_markers and self.mode==0:
            for r in self.extrema:
                yi = np.interp(r, self.xs, self.y)
                self.ax.plot(r, yi, marker='o')
            for r in self.inflex:
                yi = np.interp(r, self.xs, self.y)
                self.ax.plot(r, yi, marker='s')

        # Inspect points
        if self.inspect_pts:
            for xp in self.inspect_pts:
                fval  = np.interp(xp, self.xs, self.y)
                f1val = np.interp(xp, self.xs, self.y1)
                f2val = np.interp(xp, self.xs, self.y2)
                self.ax.plot([xp],[np.interp(xp,self.xs,Y)], marker='x', ms=8)
                self.ax.text(xp, np.interp(xp,self.xs,Y),
                             f"\nx={xp:.3g}\nf={fval:.3g}\nf'={f1val:.3g}\nf''={f2val:.3g}",
                             fontsize=8, ha='left', va='top',
                             bbox=dict(boxstyle='round', alpha=0.12, ec='none'))

        # Overlays
        if self.show_help:
            self._draw_help()
        if self.show_context:
            self._draw_context()
        if self.quiz_mode:
            self._draw_quiz_hud()
        if self.drill_mode:
            self._draw_drill_hud()

        self.fig.canvas.draw_idle()

    def _draw_help(self):
        text = (
            "View: Space/Enter/→ next, ←/Backspace prev, 0/1/2/3 jump\n"
            "Function: N new | G grid | H help | C context | S save\n"
            "Families for NEXT function: 1=Poly 2=Trig 3=Exp/Log 4=Rational\n"
            "Tangent: T toggle, A/D move x0 (Shift for coarse), slider x0\n"
            "Markers: M toggle (extrema ●, inflection ◼ on f only; refined)\n"
            "Integral: B shade area, sliders a/b, R reset\n"
            "Inspect: I toggle, click to drop | U remove last | Shift+U clear\n"
            "Quiz: Q toggle (then 1=f, 2=f', 3=f'')\n"
            "Drill: J toggle (True/False): answer Y/N"
        )
        self.ax.text(0.99, 0.02, text, transform=self.ax.transAxes,
                     ha='right', va='bottom', fontsize=9,
                     bbox=dict(boxstyle='round', alpha=0.15, ec='none', pad=0.4))

    def _draw_context(self):
        topics = [
            ("Derivatives & Tangents",
             ["Khan Academy: Derivatives — https://www.khanacademy.org/math/calculus-1/cs1-differentiation",
              "Paul's Notes: Tangent Lines — https://tutorial.math.lamar.edu/classes/calci/tangents.aspx"]),
            ("Critical & Inflection Points",
             ["Paul's Notes: Critical Points — https://tutorial.math.lamar.edu/classes/calci/criticalpoints.aspx",
              "Paul's Notes: Concavity/Inflection — https://tutorial.math.lamar.edu/classes/calci/concavity.aspx"]),
            ("Integrals & Area",
             ["Khan Academy: Integrals — https://www.khanacademy.org/math/calculus-1/cs1-integration",
              "MIT OCW: Single Variable Calculus — https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/"]),
        ]
        y = 0.98
        self.ax.text(0.78, 0.98, "Context & Resources", transform=self.ax.transAxes,
                     ha='left', va='top', fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round', alpha=0.15, ec='none', pad=0.5))
        for title, links in topics:
            y -= 0.08
            self.ax.text(0.78, y, title, transform=self.ax.transAxes,
                         ha='left', va='top', fontsize=9, fontweight='bold')
            for L in links:
                y -= 0.05
                self.ax.text(0.78, y, L, transform=self.ax.transAxes,
                             ha='left', va='top', fontsize=8)

    def _draw_quiz_hud(self):
        hud = f"Quiz Score: {self.quiz_score}/{self.quiz_attempts}"
        self.ax.text(0.01, 0.98, hud, transform=self.ax.transAxes,
                     ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', alpha=0.15, ec='none', pad=0.4))

    def _draw_drill_hud(self):
        hud = f"DRILL — {self.drill_prompt}  (Y/N)   Score: {self.drill_score}/{self.drill_attempts}"
        self.ax.text(0.01, 0.92, hud, transform=self.ax.transAxes,
                     ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', alpha=0.15, ec='none', pad=0.4))

    # ----- Events -----

    def on_key(self, ev):
        k = (ev.key or "").lower()
        if k in ('right','enter',' '): self._advance(1)
        elif k in ('left','backspace'): self._advance(-1)
        elif k == '0' and not self.quiz_mode: self.mode=0; self._plot_all()
        elif k == '1' and not self.quiz_mode: self.mode=1; self._plot_all()
        elif k == '2' and not self.quiz_mode: self.mode=2; self._plot_all()
        elif k == '3' and not self.quiz_mode: self.mode=3; self._plot_all()
        elif k == 'n': self._new_function()
        elif k == 'g': self._toggle_grid()
        elif k in ('h','?'): self._toggle_help()
        elif k == 'c': self._toggle_context()
        elif k == 'm': self._toggle_markers()
        elif k == 't': self._toggle_tangent()
        elif k == 'b' and self.mode==3: self._toggle_shade()
        elif k == 'r': self._reset_bounds()
        elif k == 'a': self._nudge_x0(-1, fine=True)
        elif k == 'd': self._nudge_x0(+1, fine=True)
        elif k == 'shift+a': self._nudge_x0(-1, fine=False)
        elif k == 'shift+d': self._nudge_x0(+1, fine=False)
        elif k == 'q': self._toggle_quiz()
        elif k == '1' and self.quiz_mode: self._quiz_guess(0)
        elif k == '2' and self.quiz_mode: self._quiz_guess(1)
        elif k == '3' and self.quiz_mode: self._quiz_guess(2)
        elif k == 'j': self._toggle_drill()
        elif k == 'y' and self.drill_mode: self._drill_answer(True)
        elif k == 'n' and self.drill_mode: self._drill_answer(False)
        elif k == 'i': self._toggle_inspect()
        elif k == 'u': self._inspect_pop(last_only=True)
        elif k == 'shift+u': self._inspect_pop(last_only=False)
        elif k == 's': self._save_png()
        elif k in ('escape','ctrl+w'): plt.close(self.fig)
        elif k == '1' and not self.quiz_mode and not self.drill_mode: self._set_family('poly')
        elif k == '2' and not self.quiz_mode and not self.drill_mode: self._set_family('trig')
        elif k == '3' and not self.quiz_mode and not self.drill_mode: self._set_family('exp_log')
        elif k == '4' and not self.quiz_mode and not self.drill_mode: self._set_family('rational')

    def on_click(self, ev):
        if not self.inspect_on: return
        if ev.inaxes != self.ax: return
        xp = float(ev.xdata)
        self.inspect_pts.append(xp)
        self._plot_all()

    # ----- Actions -----

    def _advance(self, step):
        if self.quiz_mode:
            self.quiz_answer = random.choice([0,1,2])
            self.mode = self.quiz_answer
        else:
            self.mode = (self.mode + step) % 4
        self._plot_all()

    def _new_function(self):
        fam = self.next_family
        self.next_family = None
        self.spec = generate_function(force_family=fam)
        self._rebuild_domain()
        # reset sliders
        self.s_x0.valmin, self.s_x0.valmax = self.spec.domain
        self.s_a.valmin,  self.s_a.valmax  = self.spec.domain
        self.s_b.valmin,  self.s_b.valmax  = self.spec.domain
        self.s_x0.set_val(self.x0)
        self.s_a.set_val(self.a_int)
        self.s_b.set_val(self.b_int)
        self._plot_all()

    def _toggle_grid(self):
        self.grid_on = not self.grid_on; self._plot_all()

    def _toggle_help(self):
        self.show_help = not self.show_help; self._plot_all()

    def _toggle_context(self):
        self.show_context = not self.show_context; self._plot_all()

    def _toggle_markers(self):
        self.show_markers = not self.show_markers; self._plot_all()

    def _toggle_tangent(self):
        self.show_tangent = not self.show_tangent; self._plot_all()

    def _toggle_shade(self):
        self.shade_on = not self.shade_on; self._plot_all()

    def _reset_bounds(self):
        self.a_int, self.b_int = self.spec.domain
        self.s_a.set_val(self.a_int); self.s_b.set_val(self.b_int); self._plot_all()

    def _on_slider_x0(self, val):
        self.x0 = float(val); self._plot_all()

    def _on_slider_bounds(self, val):
        self.a_int = float(self.s_a.val); self.b_int = float(self.s_b.val); self._plot_all()

    def _nudge_x0(self, direction, fine=True):
        step = (self.xs[-1]-self.xs[0]) * (0.005 if fine else 0.03)
        self.x0 = float(np.clip(self.x0 + direction*step, self.xs[0], self.xs[-1]))
        self.s_x0.set_val(self.x0)

    def _toggle_quiz(self):
        self.quiz_mode = not self.quiz_mode
        if self.quiz_mode:
            self.quiz_answer = random.choice([0,1,2])
            self.mode = self.quiz_answer
        self._plot_all()

    def _quiz_guess(self, guess_idx):
        self.quiz_attempts += 1
        if guess_idx == self.quiz_answer:
            self.quiz_score += 1; msg = "Correct!"
        else:
            names = {0:'f',1:"f'",2:"f''"}; msg = f"Nope — it was {names[self.quiz_answer]}"
        self.ax.text(0.50, 0.06, msg, transform=self.ax.transAxes,
                     ha='center', va='bottom', fontsize=11,
                     bbox=dict(boxstyle='round', alpha=0.12, ec='none'))
        self.fig.canvas.draw_idle()
        self.quiz_answer = random.choice([0,1,2]); self.mode = self.quiz_answer

    def _toggle_inspect(self):
        self.inspect_on = not self.inspect_on; self._plot_all()

    def _inspect_pop(self, last_only=True):
        if not self.inspect_pts: return
        if last_only:
            self.inspect_pts.pop()
        else:
            self.inspect_pts.clear()
        self._plot_all()

    def _set_family(self, fam: str):
        self.next_family = fam
        self.ax.text(0.50, 0.98, f"Next function → {fam}", transform=self.ax.transAxes,
                     ha='center', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', alpha=0.12, ec='none'))
        self.fig.canvas.draw_idle()

    # ----- Drill mode -----

    def _toggle_drill(self):
        self.drill_mode = not self.drill_mode
        if self.drill_mode:
            self._new_drill_prompt()
        self._plot_all()

    def _new_drill_prompt(self):
        # Build a random true/false statement
        a, b = self.spec.domain
        L = b - a
        # choose a random subinterval of moderate length
        w = L * random.uniform(0.15, 0.35)
        start = random.uniform(a, b - w)
        end = start + w

        typ = random.choice(['increasing','concave_up','root_exists','local_max'])
        if typ == 'increasing':
            seg = (self.xs>=start) & (self.xs<=end)
            val = np.nanmin(self.y1[seg]) > 0
            prompt = f"f is increasing on [{start:.2f}, {end:.2f}]"
        elif typ == 'concave_up':
            seg = (self.xs>=start) & (self.xs<=end)
            val = np.nanmin(self.y2[seg]) > 0
            prompt = f"f is concave up on [{start:.2f}, {end:.2f}]"
        elif typ == 'root_exists':
            seg = (self.xs>=start) & (self.xs<=end)
            ys = self.y[seg]
            val = np.nanmin(ys) <= 0 <= np.nanmax(ys)
            prompt = f"f has a root in [{start:.2f}, {end:.2f}]"
        else:  # local_max near x0
            # pick a candidate around some refined extremum if available
            if self.extrema:
                xc = random.choice(self.extrema)
            else:
                xc = a + 0.5*L
            # determine if it's a local max using sign of f''
            y2c = np.interp(xc, self.xs, self.y2)
            val = y2c < 0
            prompt = f"f has a local maximum near x≈{xc:.2f}"
        self.drill_prompt = prompt
        self.drill_truth = bool(val)

    def _drill_answer(self, user_true: bool):
        self.drill_attempts += 1
        if user_true == self.drill_truth:
            self.drill_score += 1; msg = "✓ Correct"
        else:
            msg = "✗ Incorrect"
        self.ax.text(0.50, 0.12, msg, transform=self.ax.transAxes,
                     ha='center', va='bottom', fontsize=11,
                     bbox=dict(boxstyle='round', alpha=0.12, ec='none'))
        self.fig.canvas.draw_idle()
        self._new_drill_prompt()
        # keep same function; just refresh overlay
        self._plot_all()

def main():
    app = CalcVizUltra()
    plt.show()

if __name__ == "__main__":
    main()
