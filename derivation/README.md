# The Hubbard atom away from half filling: derivation

`doped-gamma-derivation.pdf` derives the closed forms for the channel-irreducible vertex
Γ of the Hubbard atom at arbitrary filling, in all four channels, that `DopedHubbardAtom`
implements. Its source is `doped-gamma-derivation.typ` (Typst), with the bibliography
`doped-gamma-refs.yml`. Appendix F writes out the regular Γ formulas, including their
polynomial and thermal factors, from a single input panel, in the notation of Thunström et al.
(at half filling, their Eq. (19)). Its new Section F.6 states the closed forms with simultaneous
doping and Zeeman field for the four sectors of `FieldHubbardAtom`: coupled longitudinal,
transverse spin, oriented opposite-spin pairs and equal-spin pairs. It defines the physical bubbles,
normalizations, axis reductions and exceptional-set restrictions, without adding derivations.
Appendix G derives and plots the zero-field divergences (null vectors, the conditions at ω = 0
and exceptional points); Section G.5 adds the field-sector spectra, divergence conditions, genuine
and cancelling branches, null vectors, exceptional points and axis reductions at arbitrary transfer.
Appendix H lists the symbols, including the shared field notation. In the field parts, `h` is the
physical Zeeman field (splitting `2h`), and `κ` replaces the zero-field text's thermal constant `h`.

The longitudinal formula is stated for `μh(μ²-h²) ≠ 0` at regular roots. Its continuous extension on
`μ = ±h` is checked numerically, not derived; the separately known `h = 0` and `μ = 0` values use
zero-field and particle–hole (Shiba) identities. Unresolved exceptional-set extensions are marked
as such, and genuine divergences are not regularized. No field result is claimed to be formalized
in Lean.

The note's checks are listed in its Appendix D:

| File | Checks |
|---|---|
| `mathematica/doped_gamma.nb` | derives the susceptibility from the Hamiltonian, checks the steps of the proofs symbolically, and compares all theorems with finite-box inversion (Mathematica 15; 62 checks, about 20 s; generated from `mathematica/src/*.wl`) |
| `doped-gamma/verify_identities.py`, `verify_dm.py` | the rational identities (A1)–(A7), exact (SymPy), for readers without Mathematica |
| `doped-gamma/check_vertices.jl` | all theorems against `doped_gamma.jl`, a reference evaluator that computes Γ from residue sums and a finite linear solve, independently of the closed forms (256 bits), and against `HubbardAtom` at half filling |
| `doped-gamma/check_explicit.jl` | Appendix F's expanded formulas against the package at 256 bits, including diagonal entries, mirror fixed points, the static magnetic atomic root and independent square-root signs |
| `doped-gamma/check_divergences.jl` | Appendix G's zero-field divergence lines against the sign changes of real eigenvalues of χ on finite boxes, their null vectors, and the exceptional points (about 15 s with 6 threads) |
| `doped-gamma/check_field.jl` | independent transcription of every field vertex and physical bubble at 256 bits, both orientations, either sign of U, static/nonstatic transfers, axis limits, square-root signs and the longitudinal branch-ratio identity; field divergence/null-vector checks on growing package χ boxes, cancelled-branch controls, nonzero transverse spectra and transverse/longitudinal exceptional points (2 threads) |

In addition, `lean/` derives the input Eqs. (G), (chi-uu)–(pair) from the Hamiltonian and
formalizes Theorems "triplet", "singlet" and "density and magnetic" and Corollary "half filling" in
Lean 4 with Mathlib (except the lattice zeros of `e` at `μ ≠ 0`; at the other lattice zeros it verifies
the limiting inverses), together with the lattice sums and identities they use and some of the
exact identities behind the Float64 setup of `DopedHubbardAtom`. The table in Appendix E of the
note says what is formalized, statement by statement; `lean/README.md` lists the
Lean files and what is taken as given.

`doped-gamma/figures/` holds the figures, the script for their data (`figure_data.jl`) and the
plotting script (`make_figures.py`).

The note and its figures are licensed under CC BY 4.0, the scripts and the Lean code under MIT;
see `LICENSE`.

## Reproducing

```sh
cd mathematica
wolframscript -file replay.wl                              # evaluate the notebook, report the checks
wolframscript -file build.wl nb=doped_gamma.nb src/*.wl    # regenerate it from the sources

cd ../doped-gamma
python verify_identities.py                                # needs SymPy
python verify_dm.py d nonstatic                            # also: m, and static
julia --project=../.. check_vertices.jl
julia --project=../.. -t 1 check_explicit.jl
julia --project=../.. -t 4 check_divergences.jl
# From the repository root (not this subdirectory):
# OPENBLAS_NUM_THREADS=2 timeout 1200 julia -t 2 --project=. derivation/doped-gamma/check_field.jl

cd ../lean
lake exe cache get && lake build                           # Lean 4 and Mathlib (via elan)
./scripts/check.sh                                         # also: no sorry, standard axioms only
```

The rebuilt note has 45 pages. The field check's worst normalized vertex difference is
`6.4e-63` over 2882 comparisons; the longitudinal ratio identity has error at most `2.3e-60`.
At the nonstatic transverse/pair example the smallest box eigenvalue decreases from `3.7e-8`
to `3.8e-11` over 24–96 frequencies, with overlap deficit at most `3.2e-8`; restricted equal-spin
zeros are below `1.5e-18`. The exceptional pairs shrink to `3.4e-8` (transverse) and `2.6e-8`
(longitudinal) at 192 frequencies per spin. These are numerical checks, not rigorous bounds.
The check also tests the down-pair mirror relation and rejects the incorrect all-spin Shiba identity
at unchanged frequencies: spin interchange must be applied before the particle–hole map.

The note compiles with `typst compile doped-gamma-derivation.typ` (Typst 0.15). To redraw
the figures, run `julia --project=../../.. figure_data.jl` and then `python make_figures.py`
(NumPy, Matplotlib) in `doped-gamma/figures/`.
