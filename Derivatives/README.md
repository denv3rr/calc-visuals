# Derivatives

For practice. Generates a random function, and each keypress steps the plot through f(x) → f′(x) → f″(x).

## Controls (when the plot window is focused)

- Space / Enter / →: next view (f → f′ → f″)
- ← / Backspace: previous view
- N: new random function
- G: toggle grid
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
