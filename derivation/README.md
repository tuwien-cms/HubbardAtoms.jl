# The Hubbard atom away from half filling: derivation

`doped-gamma-derivation.pdf` derives the closed forms for the channel-irreducible vertex
Γ of the Hubbard atom at arbitrary filling, in all four channels, that `DopedHubbardAtom`
implements. Its source is `doped-gamma-derivation.typ` (Typst), with the bibliography
`doped-gamma-refs.yml`.

The note is checked three ways, as listed in its Appendix D:

| File | Checks |
|---|---|
| `mathematica/doped_gamma.nb` | derives the susceptibility from the Hamiltonian, checks the steps of the proofs symbolically, and compares all theorems with finite-box inversion (Mathematica 15; 62 checks, about 20 s; generated from `mathematica/src/*.wl`) |
| `doped-gamma/verify_identities.py`, `verify_dm.py` | the rational identities (A1)–(A7), exact (SymPy), for readers without Mathematica |
| `doped-gamma/check_vertices.jl` | all theorems against `doped_gamma.jl`, a reference evaluator that computes Γ from residue sums and a finite linear solve, independently of the closed forms (256 bits), and against `HubbardAtom` at half filling |

In addition, `lean/` formalizes Theorems "triplet", "singlet" and "density and magnetic" (the last
two in their generic cases) in Lean 4 with Mathlib, from the input Eqs. (chi-uu)–(pair), together
with the lattice sums and identities they use and the exact identities behind the Float64 setup of
`DopedHubbardAtom`. Appendix E of the note, together with a remark after each numbered statement,
says what is formalized; `lean/README.md` lists the Lean files and what is taken as given.

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

cd ../lean
lake exe cache get && lake build                           # Lean 4 and Mathlib (via elan)
./scripts/check.sh                                         # also: no sorry, standard axioms only
```

The note compiles with `typst compile doped-gamma-derivation.typ` (Typst 0.15). To redraw
the figures, run `julia --project=../../.. figure_data.jl` and then `python make_figures.py`
(NumPy, Matplotlib) in `doped-gamma/figures/`.
