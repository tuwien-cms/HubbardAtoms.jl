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
| `OrderedIntegrals.lean` | the integral over `[0, β]³` (or `[0, β]²`) of an integrand that is continuous on each ordering of the times, as a sum of integrals over the ordered simplex (Fubini on rectangles and triangles); these integrals of exponentials in closed form, also where partial sums of the rates vanish, and through the Boltzmann weights | the Lehmann representation of Section "Setup" |
| `Lehmann.lean` | the atom from its Hamiltonian: `c↑`, `c↓` as 4×4 matrices (with the anticommutation relations), `H`, `O(τ) = e^{τH}Oe^{-τH}` (matrix exponential), the thermal average, the fermionic time ordering, and `G(ν)` and `χ_σσ'(ν, ν', ω)` as the integrals of the note; `H` is diagonal with the energies of the note and `⟨X⟩ = Σᵢ pᵢXᵢᵢ`; `χ` is a sum of six ordered terms minus the disconnected part, which factorizes, and each term is a sum over paths through the four states | Section "Setup", Eq. (H) |
| `LehmannInput.lean` | Eq. (G) and the input Eqs. (chi-uu), (chi-ud) with Eq. (pair) and Appendix C, derived from the Hamiltonian for all `U`, `μ`, `n`, `m`, `k` (including the pair coefficient `c_ℓ = βp₀δ_ℓ0` at `μ = 0`); the particle–particle forms and the bubbles of Eq. (B) through the Hamiltonian's `χ` and `G` | Eqs. (G), (chi-uu)–(pair), Appendix C |
| `Thermal.lean` | the period shifts of `tanh` and `coth` by `kπi/2`; Eq. (shift), `S_k(u) = S₀(u - w/2)`; `S₀(±a)` from the weights; Lemma "thermal function" for even and odd `k`; the package's evaluation of `α₀` with its prefactor taken from the stored weights; the argument reduction of the singlet setup; the factorization of `𝒫(-Aₛ)` | Lemmas "basic sum" (Eq. shift), "thermal function"; `_α`, `_pp_setup_raw` |
| `Matsubara.lean` | Mathlib's cotangent expansion (`cot_series_rep'`) as a sum over `ℤ` along the symmetric boxes of `SummationFilter.symmetricIcc`/`symmetricIco`; from it `coth` and `tanh` as sums over the integer and half-integer lattice; `S_k(x) = PV ∑ₙ (zₙ - x)⁻¹` for even and odd `k`; `∑ₙ 1/((zₙ - x)(zₙ - y))` and `∑ₙ (zₙ - x)⁻² = S_k'(x) = S_k(x)² - β²/4`; `∑ₙ (tₙ - t)⁻¹ = 𝒯_k(t)`, also at `ξ = 0`, and `∑ₙ (tₙ - t)⁻²`; the single-pole sums as principal values along these boxes, the others absolutely convergent (Mathlib's summability lemmas for Eisenstein series, extended to complex shifts) | Lemmas "basic sum", "particle–particle sum" |
| `Triplet.lean` | `χ_t` from the input Eq. (chi-uu) via the particle–particle frequency map; `χ_t = B𝓜₊ + L_t𝓜₋`; a two-sided inverse exists; **every** right or left inverse `Y` gives `β²(Y - B⁻¹) = Γ_t` of Eq. (Gamma-t), so the vertex of Eq. (Gamma-def) is unique and equals the closed form; `Γ_t` vanishes at mirror fixed points; the Lehmann form of `G` | Theorem "triplet", Proposition "pp" (triplet part), Eq. (G) |
| `SingletInput.lean` | `χ_s` from the inputs Eqs. (chi-uu), (chi-ud), with the pair term and the twelve fractions of Appendix C; `χ_s = L_s𝓜₊ - B𝓜₋ + χ_s,reg` with the rank-two `χ_s,reg = γ₊g₊g₊ + γ₋g₋g₋` (identities (A1), (A3)) | Proposition "pp" (singlet part), Eq. (chi-s-reg) |
| `Singlet.lean` | Eq. (T-atomic) at `t±`, and its value `β²/4` at `b = ±2a`; the Woodbury cancellation, including identity (A4), without `d± ≠ 0`; an explicit two-sided inverse `Y` of `χ_s`, and `β²(Y + B⁻¹) = Γ_s` of Eq. (Gamma-s), for `U ≠ 0`, `b ≠ 0`, `tₙ + Aₛ ≠ 0` for all `n` and `Kₛ ≠ 0` | Theorem "singlet" |
| `SingletZero.lean` | `b = 0` (`μ = 0`, `k = 0`), where the poles `t±` coincide and the regular part has double poles; the assumptions `SingletHyp` of Theorem "singlet" for all `b` | Theorem "singlet" at `b = 0` |
| `SingletLZ.lean` | a lattice zero `t_c + Aₛ = 0` (any `b`): the limiting inverse, with the orbit removed from `𝒯_k(-Aₛ)`, is a two-sided inverse, and `Γ_s` is Eq. (Gamma-s-lz) | Theorem "singlet", last sentence |
| `PHInput.lean` | `χ_r = χ↑↑ + εχ↑↓` (`r = d, m`) from the inputs; Lemma "compact input" (identity (A1)); Proposition "structure", `P(zₙ)χ_r,nm = -βH(zₙ)δₙₘ + λρ/(zₙ + zₘ) + N(zₙ, zₘ)/P(zₘ)` for `zₙ + zₘ ≠ 0` (all entries for `μ ≠ 0`), and its mirror entries at `μ = 0` | Lemma "compact input", Proposition "structure" |
| `PHSums.lean` | the lattice sums of the particle–hole channels, uniform in `k` by splitting `P = Q₊Q₋` (for `k = 0` the zeros of `P` are double): `PV ∑ zʲ/P(z)`, sums of cubics over `P`; `ρS_k(-zₙ) = c̄`; Lemmas "square" and "Cauchy calculus"; `α_k(x)α_k(-x) = h²`; Lemma "rank-two sums"; the constant sum of Step 2 | Lemmas "values on the reflected lattice", "square", "thermal function" (Eq. alpha), "Cauchy calculus", "rank-two sums" |
| `PHIdentities.lean` | identities (A6) and (A7) (Eq. source), with `S±` from the weights, for both channels and `k = 0`, `k ≠ 0` | (A6), (A7) |
| `DensityMagnetic.lean` | the algebra of Steps 1, 3 and 4; Steps 1–3 as lattice sums; an explicit two-sided inverse of `χ_r` and `β²(χ_r⁻¹ - B⁻¹) = Γ_r` of Eq. (Gamma-ph), with `DD(z, z) = (F(z) + zF'(z))/(2z)` on the diagonal, for `U ≠ 0`, `μ ≠ 0`, `e(zₙ) ≠ 0`, `s₁ ≠ s₂` and, for each root, `P(uⱼ) ≠ 0`, `Kⱼ ≠ 0` or the atomic alternative; for `r = m`, `k = 0` the atomic root `u = a` drops out | Theorem "density and magnetic" (`s₁ ≠ s₂`, including the static magnetic channel) |
| `DMLifts.lean` | the note's assumptions `DMHyp` (`Kⱼ ≠ 0` only): at a zero of `P`, `βe = c(u ∓ εa)K`, so a root with `P(u) = 0`, `K(u) ≠ 0` is the atomic pair; Eq. (branch) and the independence of Eq. (Gamma-ph) of the signs of `uⱼ` | Theorem "density and magnetic", Step 5 |
| `Confluent.lean` | a double root `s₁ = s₂ = u²` (including `u = 0`): Step 3 for every `u` up to a remainder `e(u)Ξ`, termwise differentiation of lattice series (from `Complex.hasSum_deriv_of_summable_norm`), and at `u = 0` difference quotients with Tannery's theorem; the inverse, the limit formula with `∂ₛZ`, which is shown to be the `s`-derivative of the even part `Z(s)`, and the assumptions `DMConfHyp` of the note | Theorem "density and magnetic", `s₁ = s₂` |
| `HalfFilling.lean` | `μ = 0`: Proposition "structure" with the mirror `h𝓜`; the sums `T̂ = ∑ ĝ` (also at `û = 0`), `ℓ₀`, `ℓ₂`; the inverse in orbit form (even and odd eigenvalues of the diagonal-plus-mirror part) and Eq. (Gamma-hf), under `zₙ² ≠ ŝ` and, off the mirror fixed points, `zₙ² ≠ ŝ'` (`HFHyp.of_eph`: implied by `e(zₙ) ≠ 0`); `P(û) ≠ 0` automatically | Corollary "half filling" |
| `HalfFillingLZ.lean` | `μ = 0` at a lattice zero `z_c² = ŝ` (including `ŝ = 0` at a fixed point): the limiting inverse, with the orbit removed from `T̂`, and `Γ_r` of Eq. (Gamma-hf-lz) | Corollary "half filling", remark after it |
| `UZero.lean` | `U = 0`: `χ↑↓ = 0`, every channel is diagonal or mirror-diagonal with its bubble, and `Γ_r = 0` for every right or left inverse, in all four channels | `U = 0` in the proofs of the theorems |
| `StableForms.lean` | the reformulations used by the Float64 setup of Theorem "density and magnetic": the discriminant of `e` from the weights, the roots and Vieta products, the offsets of the roots from `t₊², t₋², a², w²/4`, `P(u)` as a product of offsets, the factors `z² - s = (x - δ)(x + w + δ)` of the entries, `AB = -ρM` at a root, the alternative forms of `K/ρ` and of `K` at `k = 0` (density), the divided-difference form of `Q(z, v)` off the diagonal. Not covered: `_Y_atomic`, `_tanhdiff`, and `_Q` on the diagonal | `_ph_setup_raw`, `_efactors`, `_Γraw`, `_Q` in `src/doped.jl` |

`scripts/check.sh` rebuilds the library from a cold cache (failing on any warning), checks every
declaration, private and auxiliary ones included, with the kernel for `sorry` and for axioms beyond
`propext`, `Classical.choice` and `Quot.sound` (`scripts/AxiomCheck.lean`), and checks that the linter profile of `lakefile.toml` fires
on its fixtures (`LinterFixtures.lean`).

## What is trusted

The statements: the definitions in `Lehmann.lean` (`cUp`, `cDn`, `H`, `heis`, `thermal`,
`timeOrdered`, `propagator`, `chiLehmann`) and the parameters and frequencies of `Basic.lean`; the
channels built from them in `Triplet.lean`, `SingletInput.lean` and `PHInput.lean` (`chiPPUpUp`,
`chiPPUpDn`, `chiT`, `chiS`, `chiR`, `Bpp`, `Bph`, and the polynomials of Eq. (Delta)); the closed
forms `gammaT`, `Ks`, `gammaS`, `gammaSlz`, `gammaR`, `gammaRc`, `gammaHF`, `gammaHFnote` and
`gammaHFlz` (with `DD`, `Fph`, `Zph`, `Zsd`, `Kph`, `α`, `Tpp`, `Dhat`, `Tlz`, `That'`); and the
assumptions `SingletHyp`, `SingletLZ`, `DMHyp`, `DMConfHyp`, `HFHyp` and `HFLZ` (and the generic
`SingletGeneric`, `DMGeneric`, `ModeOK`, `DMConf`, which the note's assumptions imply). The channels are written with the closed
forms `chiUpUp`, `chiUpDn` (Eqs. (chi-uu), (chi-ud), (pair), Appendix C; the susceptibilities of
Eßl et al., Phys. Rev. Research 6, 033061 (2024), Eq. (D3)) and `G`, which need no trust:
`LehmannInput.lean` proves them equal to `chiLehmann` and `propagator`. That is, the Hamiltonian
and the definitions of `G` and `χ_σσ'` (Thunström et al. 2018), the frequency conventions, the
particle–particle form `χ_pp(ν, ν', ω) = χ(ν, ν', -ω - ν - ν')` (`chiPPUpUp_eq_lehmann`), the
channel combinations `χ_t`, `χ_s`, `χ_d`, `χ_m`, the bubbles of Eq. (B) (`Bpp_eq_propagator`,
`Bph_eq_propagator`), and the definition `Γ_r = β²(χ_r⁻¹ - η_r B⁻¹)` with infinite-matrix products
as unconditional sums over ℤ. The inverses
(`invS`, `invSlz`, `invR`, `invRc`, `invHF`, `invHFlz`) need not be trusted: the theorems show that
they are two-sided inverses. Neither
does `W(z; s)` of (A7), which only enters the proof.

## Not covered (yet)

* Lemma "continuity" and the limits themselves: at the lattice zeros of Theorem "singlet" and
  Corollary "half filling" the limiting inverses are written down and verified, but that they are
  the limits of the closed forms is not proved. The lattice zeros `e(zₙ) = 0` of Theorem "density
  and magnetic" at `μ ≠ 0`. Proposition "finite boxes".
* That `χ_s` and `χ_r` have no inverses other than the constructed ones. For the block-diagonal `χ_t`
  this is automatic; the note does not claim it for the other channels either.
* Anything about rounding: `StableForms.lean` and `Thermal.lean` prove that the listed
  reformulations are exact identities, not that its loss estimates are right.

## Build

In this directory, with the toolchain of `lean-toolchain` (via elan) and Mathlib at the revision of
`lake-manifest.json`:

```sh
lake exe cache get   # prebuilt Mathlib
lake build
./scripts/check.sh   # about 3 min once Mathlib is built
```
