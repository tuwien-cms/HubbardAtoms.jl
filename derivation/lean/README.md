# HubbardAtomLean

A Lean 4 / Mathlib formalization of parts of the derivation in `../doped-gamma-derivation.typ` (the
irreducible vertex of the Hubbard atom away from half filling), and of the identities behind the
floating-point implementation `DopedHubbardAtom` of this package (`../../src/doped.jl`). Appendix E
of the note, together with a remark after each of its numbered statements, says which statements are
formalized and under which assumptions.

## What is proved

| Lean | Content | Note / package |
|---|---|---|
| `Basic.lean` | the atom, its Boltzmann weights from the Hamiltonian, the Matsubara lattice, the mirror `n̄ = -n - k - 1` | Section "Setup", Eqs. (weights), (weight-ids), (z) |
| `Thermal.lean` | the period shifts of `tanh` and `coth` by `kπi/2`; Eq. (shift), `S_k(u) = S₀(u - w/2)`; `S₀(±a)` from the weights; Lemma "thermal function" for even and odd `k`; the package's evaluation of `α₀` with its prefactor taken from the stored weights; the argument reduction of the singlet setup; the factorization of `𝒫(-Aₛ)` | Lemmas "basic sum" (Eq. shift), "thermal function"; `_α`, `_pp_setup_raw` |
| `Matsubara.lean` | Mathlib's cotangent expansion (`cot_series_rep'`) as a sum over `ℤ` along the symmetric boxes of `SummationFilter.symmetricIcc`/`symmetricIco`; from it `coth` and `tanh` as sums over the integer and half-integer lattice; `S_k(x) = PV ∑ₙ (zₙ - x)⁻¹` for even and odd `k`; `∑ₙ 1/((zₙ - x)(zₙ - y))` and `∑ₙ (zₙ - x)⁻² = S_k'(x) = S_k(x)² - β²/4`; `∑ₙ (tₙ - t)⁻¹ = -S_k(μ + ξ)/ξ`; the single-pole sums as principal values along these boxes, the others absolutely convergent (Mathlib's summability lemmas for Eisenstein series, extended to complex shifts) | Lemmas "basic sum", "particle–particle sum" |
| `Triplet.lean` | `χ_t` from the input Eq. (chi-uu) via the particle–particle frequency map; `χ_t = B𝓜₊ + L_t𝓜₋`; a two-sided inverse exists; **every** right or left inverse `Y` gives `β²(Y - B⁻¹) = Γ_t` of Eq. (Gamma-t), so the vertex of Eq. (Gamma-def) is unique and equals the closed form; `Γ_t` vanishes at mirror fixed points; the Lehmann form of `G` | Theorem "triplet", Proposition "pp" (triplet part), Eq. (G) |
| `SingletInput.lean` | `χ_s` from the inputs Eqs. (chi-uu), (chi-ud), with the pair term and the twelve fractions of Appendix C; `χ_s = L_s𝓜₊ - B𝓜₋ + χ_s,reg` with the rank-two `χ_s,reg = γ₊g₊g₊ + γ₋g₋g₋` (identities (A1), (A3)) | Proposition "pp" (singlet part), Eq. (chi-s-reg) |
| `Singlet.lean` | Eq. (T-atomic) at `t±`; the Woodbury cancellation, including identity (A4); an explicit two-sided inverse `Y` of `χ_s`, and `β²(Y + B⁻¹) = Γ_s` of Eq. (Gamma-s), for `U ≠ 0`, `b ≠ 0, ±2a`, `d± ≠ 0`, `tₙ + Aₛ ≠ 0` for all `n`, `ξ ≠ 0` and `Kₛ ≠ 0` | Theorem "singlet" (generic case) |
| `PHInput.lean` | `χ_r = χ↑↑ + εχ↑↓` (`r = d, m`) from the inputs; Lemma "compact input" (identity (A1)); Proposition "structure", `P(zₙ)χ_r,nm = -βH(zₙ)δₙₘ + λρ/(zₙ + zₘ) + N(zₙ, zₘ)/P(zₘ)` for `μ ≠ 0` | Lemma "compact input", Proposition "structure" |
| `PHSums.lean` | the lattice sums of the particle–hole channels, uniform in `k` by splitting `P = Q₊Q₋` (for `k = 0` the zeros of `P` are double): `PV ∑ zʲ/P(z)`, sums of cubics over `P`; `ρS_k(-zₙ) = c̄`; Lemmas "square" and "Cauchy calculus"; `α_k(x)α_k(-x) = h²`; Lemma "rank-two sums"; the constant sum of Step 2 | Lemmas "values on the reflected lattice", "square", "thermal function" (Eq. alpha), "Cauchy calculus", "rank-two sums" |
| `PHIdentities.lean` | identities (A6) and (A7) (Eq. source), with `S±` from the weights, for both channels and `k = 0`, `k ≠ 0` | (A6), (A7) |
| `DensityMagnetic.lean` | the algebra of Steps 1, 3 and 4; Steps 1–3 as lattice sums; an explicit two-sided inverse of `χ_r` and `β²(χ_r⁻¹ - B⁻¹) = Γ_r` of Eq. (Gamma-ph), with `DD(z, z) = (F(z) + zF'(z))/(2z)` on the diagonal, for `U ≠ 0`, `μ ≠ 0`, `e(zₙ) ≠ 0`, `s₁ ≠ s₂` and, for each root, `P(uⱼ) ≠ 0`, `Kⱼ ≠ 0` or the atomic alternative; for `r = m`, `k = 0` the atomic root `u = a` drops out | Theorem "density and magnetic" (generic case, including the static magnetic channel) |
| `StableForms.lean` | the reformulations used by the Float64 setup of Theorem "density and magnetic": the discriminant of `e` from the weights, the roots and Vieta products, the offsets of the roots from `t₊², t₋², a², w²/4`, `P(u)` as a product of offsets, the factors `z² - s = (x - δ)(x + w + δ)` of the entries, `AB = -ρM` at a root, the alternative forms of `K/ρ` and of `K` at `k = 0` (density), the divided-difference form of `Q(z, v)` off the diagonal. Not covered: `_Y_atomic`, `_tanhdiff`, and `_Q` on the diagonal | `_ph_setup_raw`, `_efactors`, `_Γraw`, `_Q` in `src/doped.jl` |

`scripts/check.sh` rebuilds the library from a cold cache (failing on any warning), checks every
declaration, private and auxiliary ones included, with the kernel for `sorry` and for axioms beyond
`propext`, `Classical.choice` and `Quot.sound` (`scripts/AxiomCheck.lean`), and checks that the linter profile of `lakefile.toml` fires
on its fixtures (`LinterFixtures.lean`).

## What is trusted

The statements: the definitions in `Basic.lean`, `Triplet.lean`, `SingletInput.lean` and
`PHInput.lean` (`chiR`, `Bph` and the polynomials of Eq. (Delta)), the closed forms `gammaT`, `Ks`,
`gammaS` and `gammaR` (with `DD`, `Fph`, `Zph`, `Kph`, `α`), and the assumptions `SingletGeneric`,
`DMGeneric` and `ModeOK`. That is, the Hamiltonian's energies and weights, the frequency
conventions, the inputs Eqs. (chi-uu), (chi-ud) with Eq. (pair) and Appendix C (the
susceptibilities of Eßl et al., Phys. Rev. Research 6, 033061 (2024), Eq. (D3), taken as given, as
in the note), the particle–particle form `χ_pp(ν, ν', ω) = χ(ν, ν', -ω - ν - ν')`, the channel
combinations `χ_t`, `χ_s`, `χ_d`, `χ_m`, the bubbles of Eq. (B), and the definition
`Γ_r = β²(χ_r⁻¹ - η_r B⁻¹)` with infinite-matrix products as unconditional sums over ℤ. The inverses
`invS` and `invR` need not be trusted: the theorems show that they are two-sided inverses. Neither
does `W(z; s)` of (A7), which only enters the proof.

## Not covered (yet)

* The cases that the note covers by continuity (Lemma "continuity"): those of Theorem "singlet"
  excluded above; in Theorem "density and magnetic" the confluent roots `s₁ = s₂`, the lattice roots
  `e(zₙ) = 0` and the other common zeros of `P` and `e`; and Corollary "half filling" (`μ = 0`),
  which the note obtains as the limit `μ → 0`. Proposition "finite boxes".
* That `χ_s` and `χ_r` have no inverses other than the constructed ones. For the block-diagonal `χ_t`
  this is automatic; the note does not claim it for the other channels either.
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
./scripts/check.sh   # about 2.5 min once Mathlib is built
```
