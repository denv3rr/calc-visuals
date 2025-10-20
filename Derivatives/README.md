# Derivatives

Generates a random function, where:
- each 'ENTER' keypress steps the plot through f(x) → f′(x) → f″(x).
- 'T' toggles tangent slope line which can be dragged with slider 'x0' at the bottom.
- 'B' toggles integrals shading between 'a' and 'b' domain bounds.

> [!Note]
>
> Mostly for practice and personal use. As in... **there are probably bugs.**
>

## Controls (when the plot window is focused)

- Space / Enter / →: next view (f → f′ → f″)
- ← / Backspace: previous view
- N: new random function
- G: toggle grid
- B: toggle integral shading
  - *note:* a/b sliders at bottom when this is toggled
- S: save current view as PNG
- H or ?: toggle quick help overlay
- Q or Esc: quit

## Run

```sh
python <filename>
```

## Notes

What it currently generates:

- Random families: polynomials, trig mixes, exp/log combos, and some rational functions
- Symbolic derivatives via SymPy → fast NumPy lambdas for plotting
- Domains chosen to avoid obvious singularities
- Generator retries if a candidate acts incorrectly
- “B” key works from any view (auto-jumps to 'Integral view' and toggles shading).
- Integral shading now fills the region under f(x) between a and b, and shows the value
  - $\int_a^b f(x) \, dx$ ≈ F(b)−F(a)
 
- Notes from file comments (`derivatives_vis.py`):
  - ```
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
    ``` 
