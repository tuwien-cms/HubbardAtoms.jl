# The Hubbard atom away from half filling: derivation

`doped-gamma-derivation.pdf` derives the closed forms for the channel-irreducible vertex
Γ of the Hubbard atom at arbitrary filling, in all four channels, that `DopedHubbardAtom`
implements. Its source is `doped-gamma-derivation.typ` (Typst), with the bibliography
`doped-gamma-refs.yml`.

The note takes the generalized susceptibility of Eßl et al., Phys. Rev. Research 6, 033061
(2024), Eq. (D3), as input, and checks every rational identity it uses by computer algebra
and every result numerically.

`mathematica/doped_gamma.nb` replays the derivation in Mathematica 15: it derives the
susceptibility from the Lehmann representation of the atom, checks the steps of the proofs
symbolically, and compares all theorems with brute-force inversion on finite boxes (62 checks,
about 20 s). It is generated from `mathematica/src/*.wl`.

The scripts in `doped-gamma/` are cited in the note's appendices:

| Script | Checks |
|---|---|
| `verify_identities.py` | identities (A1)–(A5), exact (SymPy) |
| `verify_dm.py` | identities (A6), (A7) and Step 3 of the proof for the density and magnetic channels, exact (SymPy) |
| `check_formulas.jl` | the input against exact diagonalization; Matsubara sums and the Cauchy calculus against direct lattice sums; the singlet and triplet results against the reference evaluator |
| `check_dm.jl` | the density and magnetic result against the reference evaluator at 256 bits, its half-filling limit against `HubbardAtom`, and the steps of the proof |
| `check_exceptional.jl` | the removable singularities and a genuine divergence, against finite-box inversion |
| `indep_check.jl` | the reference evaluator against exact diagonalization and finite-box inversion, all channels |

`doped_gamma.jl` is the reference evaluator, which computes Γ from residue sums and a finite
linear solve, independently of the closed forms; `ed_doped.jl` is the exact diagonalization.
`figures/` holds the figures, their data (`figure_data.jl`) and the plotting script
(`make_figures.py`).

The note and its figures are licensed under CC BY 4.0, the scripts under MIT; see `LICENSE`.

## Reproducing

From `doped-gamma/`, with this repository as the Julia project:

```sh
python verify_identities.py                          # needs SymPy
python verify_dm.py d nonstatic                      # also: m, and static
julia --project=../.. check_formulas.jl
julia --project=../.. check_dm.jl
julia --project=../.. check_exceptional.jl
julia --project=../.. indep_check.jl                 # several minutes
```

From `mathematica/`:

```sh
wolframscript -file replay.wl                              # evaluate the notebook, report the checks
wolframscript -file build.wl nb=doped_gamma.nb src/*.wl    # regenerate it from the sources
```

The note compiles with `typst compile doped-gamma-derivation.typ` (Typst 0.15). To redraw
the figures, run `julia --project=../../.. figure_data.jl` and then `python make_figures.py`
(NumPy, Matplotlib) in `doped-gamma/figures/`.
