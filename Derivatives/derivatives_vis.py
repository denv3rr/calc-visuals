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
        # |sin| to avoid log domain issues; add small eps
        expr = sp.log(sp.Abs(sp.sin(x)) + sp.Float('1e-3'))
        desc = "log(|sin x| + 1e-3)"
        return sp.simplify(expr), desc
    else:  # log_poly
        a = random.randint(1,3)
        expr = sp.log(x**2 + a)
        desc = f"log(x^2+{a})"
        return sp.simplify(expr), desc

def _rational_family():
    # random simple rational with vertical asymptote but bounded on domain
    pdeg = random.choice([1,2])
    qdeg = random.choice([1,2])
    p = sum(random.randint(-3,3) * x**i for i in range(pdeg+1))
    q = sum(random.randint(1,3) * x**i for i in range(qdeg+1))
    expr = sp.simplify(p / (q + 1))  # avoid zeros
    return expr, "Rational"

def _choose_domain(expr: sp.Expr) -> Tuple[float,float]:
    # basic heuristic domains per family
    if expr.has(sp.log):
        return (-5, 5)
    if expr.has(sp.exp):
        return (-5, 5)
    return (-5, 5)

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
    out = np.empty_like(x)
    out[0] = 0.0
    out[1:] = np.cumsum(mid)
    return out

def _finite_mask(*arrs):
    for a in arrs:
        if not np.all(np.isfinite(a)):
            return False
    return True

def zero_brackets(xs, ys):
    # find sign-change intervals
    s = np.sign(ys)
    idx = np.where(np.diff(s) != 0)[0]
    return [(xs[i], xs[i+1]) for i in idx]

def bisection_refine(f, a, b, iters=28):
    fa, fb = f(a), f(b)
    if np.isnan(fa) or np.isnan(fb): return 0.5*(a+b)
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0: return 0.5*(a+b)
    lo, hi = a, b
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        fm = f(mid)
        if np.isnan(fm): break
        if fa*fm <= 0:
            hi = mid; fb = fm
        else:
            lo = mid; fa = fm
    return 0.5*(lo+hi)

# ---------------- App ----------------

class DerivativesVis:
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

        # events
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._plot_all()

    def _rebuild_domain(self):
        self._update_latex()
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

        self.btn_exp  = Button(plt.axes([0.81, 0.27, 0.075, 0.055]), 'Exp/Log (3)')
        self.btn_rat  = Button(plt.axes([0.895,0.27, 0.075, 0.055]), 'Rational (4)')

        # right column (quiz/drill)
        self.btn_quiz = Button(plt.axes([0.81, 0.17, 0.16, 0.055]), 'View Quiz (Q)')
        self.btn_drill= Button(plt.axes([0.81, 0.10, 0.16, 0.055]), 'Concept Drill (J)')

        # callbacks
        self.btn_new.on_clicked(lambda e: self._new_function())
        self.btn_help.on_clicked(lambda e: self._toggle_help())
        self.btn_ctx.on_clicked(lambda e: self._toggle_context())
        self.btn_grid.on_clicked(lambda e: self._toggle_grid())
        self.btn_mark.on_clicked(lambda e: self._toggle_markers())
        self.btn_tan.on_clicked(lambda e: self._toggle_tangent())
        self.btn_save.on_clicked(lambda e: self._save_png())

        self.btn_poly.on_clicked(lambda e: self._force_family('poly'))
        self.btn_trig.on_clicked(lambda e: self._force_family('trig'))
        self.btn_exp.on_clicked(lambda e: self._force_family('exp_log'))
        self.btn_rat.on_clicked(lambda e: self._force_family('rational'))

        self.btn_quiz.on_clicked(lambda e: self._toggle_quiz())
        self.btn_drill.on_clicked(lambda e: self._toggle_drill())

    # ----------------- Header (formulas) -----------------

    def _update_latex(self):
        """Compute LaTeX for f, f', f'' using SymPy; displayed in the header."""
        try:
            d1 = sp.diff(self.spec.expr, x)
            d2 = sp.diff(d1, x)
            self._latex_f  = sp.latex(sp.simplify(self.spec.expr))
            self._latex_f1 = sp.latex(sp.simplify(d1))
            self._latex_f2 = sp.latex(sp.simplify(d2))
        except Exception:
            # fallback plain strings
            self._latex_f  = sp.latex(self.spec.expr)
            self._latex_f1 = "?"
            self._latex_f2 = "?"

    def _draw_formula_header(self):
        """Draw f, f', f'' formulas in mathtext at the top-left."""
        try:
            txt = (r"$f(x)=%s$" % self._latex_f) + "\n" + \
                  (r"$f'(x)=%s$" % self._latex_f1) + "\n" + \
                  (r"$f''(x)=%s$" % self._latex_f2)
            self.ax.text(0.01, 0.98, txt, transform=self.ax.transAxes,
                         ha='left', va='top', fontsize=10,
                         bbox=dict(boxstyle='round', alpha=0.10, ec='none', pad=0.4))
        except Exception:
            pass

    # ----------------- Events -----------------

    def _on_slider_x0(self, val):
        self.x0 = float(val); self._plot_all()

    def _on_slider_bounds(self, val):
        self.a_int = float(self.s_a.val); self.b_int = float(self.s_b.val)
        self._plot_all()

    def _on_click(self, ev):
        if not self.inspect_on: return
        if ev.inaxes != self.ax: return
        self.inspect_pts.append(ev.xdata)
        self._plot_all()

    def _on_key(self, ev):
        k = ev.key.lower() if ev.key else ''
        if k in ('escape', 'ctrl+w'):
            plt.close(self.fig); return

        if self.quiz_mode:
            if k in ('1','2','3'): self.quiz_answer = int(k); self._answer_quiz(); return
            if k in ('q',): self._toggle_quiz(); return
            return

        if self.drill_mode:
            if k in ('y','n'):
                self._drill_answer(k=='y')
                return
            if k in ('j',): self._toggle_drill(); return
            # allow navigation even in drill
        if True:
            if k in ('enter',' '):
                self.mode = (self.mode+1)%4; self._plot_all()
            elif k in ('backspace', 'left'):
                self.mode = (self.mode-1)%4; self._plot_all()
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
            elif k == 'b':
                # jump to Integral view and toggle shading
                if not self.quiz_mode:
                    if self.mode != 3:
                        self.mode = 3
                    self._toggle_shade()
            elif k == 'r': self._reset_bounds()
            elif k == 'a': self._nudge_x0(-1, fine=True)
            elif k == 'd': self._nudge_x0(+1, fine=True)
            elif k == 'shift+a': self._nudge_x0(-1, fine=False)
            elif k == 'shift+d': self._nudge_x0(+1, fine=False)
            elif k == 'i': self._toggle_inspect()
            elif k == 'u': self._inspect_undo()
            elif k == 'shift+u': self._inspect_clear()
            elif k == 'q': self._toggle_quiz()
            elif k == 'j': self._toggle_drill()
            elif k == 's': self._save_png()

    # ----- Feature toggles -----

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

    def _toggle_inspect(self):
        self.inspect_on = not self.inspect_on; self._plot_all()

    def _inspect_undo(self):
        if self.inspect_pts:
            self.inspect_pts.pop()
            self._plot_all()

    def _inspect_clear(self):
        self.inspect_pts = []; self._plot_all()

    def _force_family(self, fam: str):
        self.next_family = fam

    def _reset_bounds(self):
        self.a_int, self.b_int = self.spec.domain
        self.s_a.set_val(self.a_int); self.s_b.set_val(self.b_int)
        self._plot_all()

    def _nudge_x0(self, direction: int, fine=True):
        rng = self.spec.domain
        step = (rng[1]-rng[0]) * (0.005 if fine else 0.03)
        self.x0 = float(np.clip(self.x0 + direction*step, rng[0], rng[1]))
        self.s_x0.set_val(self.x0)
        self._plot_all()

    def _save_png(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"deriv_vis_{ts}.png"
        self.fig.savefig(fname, dpi=180, bbox_inches='tight')
        self.ax.text(0.5, 0.5, f"Saved {fname}", transform=self.ax.transAxes,
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', alpha=0.2, ec='none'))
        self.fig.canvas.draw_idle()

    # ----- Plotting -----

    def _title(self):
        vname = {0:"f(x)",1:"f'(x)",2:"f''(x)",3:"F(x)"}[self.mode]
        return f"{vname} | Family: {self.spec.family} ({self.spec.desc})"

    def _compute_current(self):
        if self.mode==0: return self.y
        if self.mode==1: return self.y1
        if self.mode==2: return self.y2
        return self.F

    def _plot_all(self):
        self.ax.clear()
        self.ax.grid(self.grid_on, which='both', alpha=0.35)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel({0:"f(x)",1:"f'(x)",2:"f''(x)",3:"F(x)"}[self.mode])
        self.ax.set_title(self._title())

        Y = self._compute_current()
        mask = np.isfinite(Y)
        self.ax.plot(self.xs[mask], Y[mask], lw=2)

        # Integral shading (definite integral of f from a to b)
        if self.mode==3 and self.shade_on:
            a, b = sorted([self.a_int, self.b_int])

            seg = (self.xs>=a) & (self.xs<=b)
            yf = self.y
            m_fill = seg & np.isfinite(yf)
            if np.any(m_fill):
                self.ax.fill_between(self.xs[m_fill], yf[m_fill], 0, alpha=0.15)
            # guide lines
            self.ax.axvline(a, linestyle=':', lw=1)
            self.ax.axvline(b, linestyle=':', lw=1)

            # compute area via fundamental theorem: F(b)-F(a)
            Fa = float(np.interp(a, self.xs, self.F))
            Fb = float(np.interp(b, self.xs, self.F))
            area = Fb - Fa
            self.ax.text(
                0.50, 0.94,
                f"$\\int_{{{a:.2f}}}^{{{b:.2f}}} f(x)\\,dx \\approx$ {area:.5g}",
                transform=self.ax.transAxes, ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round', alpha=0.15, ec='none', pad=0.4)
            )

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

        # Inspection probes
        if self.inspect_pts:
            for xp in self.inspect_pts:
                if self.mode==0:
                    yp = float(np.interp(xp, self.xs, self.y)); d1 = float(np.interp(xp, self.xs, self.y1)); d2 = float(np.interp(xp, self.xs, self.y2))
                elif self.mode==1:
                    yp = float(np.interp(xp, self.xs, self.y1)); d1 = float(np.interp(xp, self.xs, self.y2)); d2 = float(0.0)
                elif self.mode==2:
                    yp = float(np.interp(xp, self.xs, self.y2)); d1 = float(0.0); d2 = float(0.0)
                else:
                    yp = float(np.interp(xp, self.xs, self.F)); d1 = float(np.interp(xp, self.xs, self.y)); d2 = float(np.interp(xp, self.xs, self.y1))
                self.ax.plot([xp],[yp], marker='x')
                self.ax.text(xp, yp, f"\n x={xp:.3g}\n f≈{yp:.3g}\n f'≈{d1:.3g}\n f''≈{d2:.3g}",
                             fontsize=8, ha='left', va='bottom',
                             bbox=dict(boxstyle='round', alpha=0.12, ec='none'))

        # Header formulas
        self._draw_formula_header()

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

    # ----- Overlays -----

    def _draw_help(self):
        text = (
            "View: Space/Enter/→ next, ←/Backspace prev, 0/1/2/3 jump\n"
            "Function: N new | G grid | H help | C context | S save\n"
            "Families for NEXT function: 1=Poly 2=Trig 3=Exp/Log 4=Rational\n"
            "Tangent: T toggle, A/D move x0 (Shift for coarse), slider x0\n"
            "Markers: M toggle (extrema ●, inflection ◼ on f only; refined)\n"
            "Integral: B toggle (auto-jumps), sliders a/b, R reset\n"
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
             ["Khan Academy: Derivatives — https://www.khanacademy.org/math/calculus-1/cs1-derivatives",
              "3Blue1Brown: Essence of Calculus — https://www.3blue1brown.com/topics/calculus"]),
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
            for link in links:
                y -= 0.05
                self.ax.text(0.78, y, link, transform=self.ax.transAxes,
                             ha='left', va='top', fontsize=8)

    # ----- Quiz (identify views) -----

    def _toggle_quiz(self):
        self.quiz_mode = not self.quiz_mode
        if self.quiz_mode:
            self.quiz_answer = 0
        self._plot_all()

    def _draw_quiz_hud(self):
        self.ax.text(0.02, 0.98, "QUIZ: Which view is shown? Press 1=f, 2=f', 3=f''.",
                     transform=self.ax.transAxes, ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', alpha=0.15, ec='none'))
        if self.quiz_answer in (1,2,3):
            correct = self.mode+1 if self.mode<3 else 1
            ok = (self.quiz_answer == correct)
            self.quiz_attempts += 1
            self.quiz_score += 1 if ok else 0
            self.ax.text(0.02, 0.89, f"{'✓ Correct' if ok else '✗ Incorrect'}  (score {self.quiz_score}/{self.quiz_attempts})",
                         transform=self.ax.transAxes, fontsize=10,
                         bbox=dict(boxstyle='round', alpha=0.12, ec='none'))
            self.fig.canvas.draw_idle()

    # ----- Drill (True/False) -----

    def _toggle_drill(self):
        self.drill_mode = not self.drill_mode
        if self.drill_mode:
            self._new_drill_prompt()
        self._plot_all()

    def _draw_drill_hud(self):
        self.ax.text(0.50, 0.18, "DRILL: Y (True) / N (False)", transform=self.ax.transAxes,
                     ha='center', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round', alpha=0.12, ec='none'))
        self.ax.text(0.50, 0.14, self.drill_prompt, transform=self.ax.transAxes,
                     ha='center', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round', alpha=0.12, ec='none'))

    def _new_drill_prompt(self):
        # randomize among a few conceptual checks on f
        choice = random.choice(['increasing','concave','root','localmax'])
        L = self.xs[-1]-self.xs[0]
        a = self.xs[0] + 0.20*L
        b = self.xs[0] + 0.80*L
        if choice=='increasing':
            xc = random.uniform(a, b)
            val = np.interp(xc, self.xs, self.y1) > 0
            prompt = f"f is increasing near x≈{xc:.2f}"
        elif choice=='concave':
            xc = random.uniform(a, b)
            val = np.interp(xc, self.xs, self.y2) > 0
            prompt = f"f is concave up near x≈{xc:.2f}"
        elif choice=='root':
            xc = random.uniform(a, b)
            yi = np.interp(xc, self.xs, self.y)
            # small-magnitude heuristic
            val = abs(yi) < 0.1*np.nanmax(np.abs(self.y))
            prompt = f"x≈{xc:.2f} is near a root of f"
        else:  # localmax
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
    app = DerivativesVis()
    plt.show()

if __name__ == "__main__":
    main()
