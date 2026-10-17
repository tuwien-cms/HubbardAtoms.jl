# HubbardAtomLean

A Lean 4 / Mathlib formalization of parts of the derivation in `../doped-gamma-derivation.typ` (the
irreducible vertex of the Hubbard atom away from half filling), and of the identities behind the
floating-point implementation `DopedHubbardAtom` of this package (`../../src/doped.jl`).

## What is proved

| Lean | Content | Note / package |
|---|---|---|
| `Basic.lean` | the atom, its Boltzmann weights from the Hamiltonian, the Matsubara lattice, the mirror `n̄ = -n - k - 1` | Section "Setup", Eqs. (weights), (weight-ids), (z) |
| `Thermal.lean` | the period shifts of `tanh` and `coth` by `kπi/2`; Eq. (shift), `S_k(u) = S₀(u - w/2)`; `S₀(±a)` from the weights; Lemma "thermal function" for even and odd `k`; the package's evaluation of `α₀` with its prefactor taken from the stored weights; the argument reduction of the singlet setup; the factorization of `𝒫(-Aₛ)` | Lemmas "basic sum" (Eq. shift), "thermal function"; `_α`, `_pp_setup_raw` |
| `Matsubara.lean` | Mathlib's cotangent expansion (`cot_series_rep'`) as a sum over `ℤ` along the symmetric boxes of `SummationFilter.symmetricIcc`/`symmetricIco`; from it `coth` and `tanh` as sums over the integer and half-integer lattice; `S_k(x) = PV ∑ₙ (zₙ - x)⁻¹` for even and odd `k`; `∑ₙ (tₙ - t)⁻¹ = -S_k(μ + ξ)/ξ`, absolutely convergent (Mathlib's summability lemmas for Eisenstein series, extended to complex shifts) | Lemmas "basic sum", "particle–particle sum" |
| `Triplet.lean` | `χ_t` from the input Eq. (chi-uu) via the particle–particle frequency map; `χ_t = B𝓜₊ + L_t𝓜₋`; a two-sided inverse exists; **every** right or left inverse `Y` gives `β²(Y - B⁻¹) = Γ_t` of Eq. (Gamma-t), so the vertex of Eq. (Gamma-def) is unique and equals the closed form; `Γ_t` vanishes at mirror fixed points; the Lehmann form of `G` | Theorem "triplet", Proposition "pp" (triplet part), Eq. (G) |
| `SingletInput.lean` | `χ_s` from the inputs Eqs. (chi-uu), (chi-ud), with the pair term and the twelve fractions of Appendix C; `χ_s = L_s𝓜₊ - B𝓜₋ + χ_s,reg` with the rank-two `χ_s,reg = γ₊g₊g₊ + γ₋g₋g₋` (identities (A1), (A3)) | Proposition "pp" (singlet part), Eq. (chi-s-reg) |
| `Singlet.lean` | Eq. (T-atomic) at `t±`; the Woodbury cancellation, including identity (A4); an explicit two-sided inverse `Y` of `χ_s`, and `β²(Y + B⁻¹) = Γ_s` of Eq. (Gamma-s), for `U ≠ 0`, `b ≠ 0, ±2a`, `d± ≠ 0`, `tₙ + Aₛ ≠ 0` for all `n`, `ξ ≠ 0` and `Kₛ ≠ 0` | Theorem "singlet" (generic case) |
| `StableForms.lean` | the reformulations used by the Float64 setup of Theorem "density and magnetic": the discriminant of `e` from the weights, the roots and Vieta products, the offsets of the roots from `t₊², t₋², a², w²/4`, `P(u)` as a product of offsets, the factors `z² - s = (x - δ)(x + w + δ)` of the entries, `AB = -ρM` at a root, the alternative forms of `K/ρ` and of `K` at `k = 0` (density), the divided-difference form of `Q(z, v)` off the diagonal. Not covered: `_Y_atomic`, `_tanhdiff`, and `_Q` on the diagonal | `_ph_setup_raw`, `_efactors`, `_Γraw`, `_Q` in `src/doped.jl` |

`scripts/check.sh` rebuilds the library from a cold cache (failing on any warning), checks every
declaration, private and auxiliary ones included, with the kernel for `sorry` and for axioms beyond
`propext`, `Classical.choice` and `Quot.sound` (`scripts/AxiomCheck.lean`), and checks that the linter profile of `lakefile.toml` fires
on its fixtures (`LinterFixtures.lean`).

## What is trusted

The statements: the definitions in `Basic.lean`, `Triplet.lean` and `SingletInput.lean`, the closed
forms `gammaT`, `Ks` and `gammaS`, and the assumptions `SingletGeneric`. That is, the Hamiltonian's
energies and weights, the frequency conventions, the inputs Eqs. (chi-uu), (chi-ud) with Eq. (pair)
and Appendix C (the susceptibilities of Eßl et al., Phys. Rev. Research 6, 033061 (2024), Eq. (D3),
taken as given, as in the note), the particle–particle form `χ_pp(ν, ν', ω) = χ(ν, ν', -ω - ν - ν')`,
the channel combinations `χ_t`, `χ_s`, the bubble of Eq. (B), and the definition
`Γ_r = β²(χ_r⁻¹ - η_r B⁻¹)` with infinite-matrix products as unconditional sums over ℤ. The singlet
inverse `invS` need not be trusted: the theorems show that it is a two-sided inverse.

## Not covered (yet)

* Theorem "density and magnetic" and Corollary "half filling", which need the algebra of the Cauchy
  matrix (Section "The Cauchy matrix" of the note) on top of the lattice sums here.
* The cases of Theorem "singlet" excluded above, which the note covers by continuity
  (Lemma "continuity"), and Proposition "finite boxes".
* That `χ_s` has no inverse other than the constructed one. For the block-diagonal `χ_t` this is
  automatic; the note does not claim it for `χ_s` either.
* The derivation of the input from the Hamiltonian (the note takes it from Eßl et al.; the
  Mathematica notebook in `../mathematica` derives it).
* Anything about rounding: `StableForms.lean` and `Thermal.lean` prove that the listed
  reformulations are exact identities, not that its loss estimates are right.

## Build

In this directory, with the toolchain of `lean-toolchain` (via elan) and Mathlib at the revision of
`lake-manifest.json`:

```sh
lake exe cache get   # prebuilt Mathlib
lake build
./scripts/check.sh   # about 1.5 min once Mathlib is built
```
