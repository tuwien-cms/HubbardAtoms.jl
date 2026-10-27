# HubbardAtomLean

A Lean 4 / Mathlib formalization of parts of the derivation in `../doped-gamma-derivation.typ` (the
irreducible vertex of the Hubbard atom away from half filling), and of the identities behind the
floating-point implementation `DopedHubbardAtom` of this package (`../../src/doped.jl`). The table in
Appendix E of the note lists each numbered statement with what is formalized, under which
assumptions, and the Lean names; the formalized statements carry a "Lean" tag in the note.
The files `Field*.lean` formalize the transverse, opposite-spin and equal-spin pair vertices of
the atom with doping and Zeeman field (Section F.6 of the note, `FieldHubbardAtom` in
`../../src/field*.jl`),
and derive the input of the coupled longitudinal sector from the Hamiltonian;
the longitudinal vertex itself is not formalized.

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
| `SingletLZ.lean` | a lattice zero `t_c + Aₛ = 0` (any `b`): the limiting inverse, with the orbit removed from `𝒯_k(-Aₛ)`, is a two-sided inverse, and `Γ_s` is Eq. (Gamma-s-lz) | Proposition "removable singularities", item 1 |
| `PHInput.lean` | `χ_r = χ↑↑ + εχ↑↓` (`r = d, m`) from the inputs; Lemma "compact input" (identity (A1)); Proposition "structure", `P(zₙ)χ_r,nm = -βH(zₙ)δₙₘ + λρ/(zₙ + zₘ) + N(zₙ, zₘ)/P(zₘ)` for `zₙ + zₘ ≠ 0` (all entries for `μ ≠ 0`), and its mirror entries at `μ = 0` | Lemma "compact input", Proposition "structure" |
| `PHSums.lean` | the lattice sums of the particle–hole channels, uniform in `k` by splitting `P = Q₊Q₋` (for `k = 0` the zeros of `P` are double): `PV ∑ zʲ/P(z)`, sums of cubics over `P`; `ρS_k(-zₙ) = c̄`; Lemmas "square" and "Cauchy calculus"; `α_k(x)α_k(-x) = h²`; Lemma "rank-two sums"; the constant sum of Step 2 | Lemmas "values on the reflected lattice", "square", "thermal function" (Eq. alpha), "Cauchy calculus", "rank-two sums" |
| `PHIdentities.lean` | identities (A6) and (A7) (Eq. source), with `S±` from the weights, for both channels and `k = 0`, `k ≠ 0` | (A6), (A7) |
| `DensityMagnetic.lean` | the algebra of Steps 1, 3 and 4; Steps 1–3 as lattice sums; an explicit two-sided inverse of `χ_r` and `β²(χ_r⁻¹ - B⁻¹) = Γ_r` of Eq. (Gamma-ph), with `DD(z, z) = (F(z) + zF'(z))/(2z)` on the diagonal, for `U ≠ 0`, `μ ≠ 0`, `e(zₙ) ≠ 0`, `s₁ ≠ s₂` and, for each root, `P(uⱼ) ≠ 0`, `Kⱼ ≠ 0` or the atomic alternative; for `r = m`, `k = 0` the atomic root `u = a` drops out | Theorem "density and magnetic" (`s₁ ≠ s₂`, including the static magnetic channel) |
| `DMLifts.lean` | the note's assumptions `DMHyp` (`Kⱼ ≠ 0` only): at a zero of `P`, `βe = c(u ∓ εa)K`, so a root with `P(u) = 0`, `K(u) ≠ 0` is the atomic pair; Eq. (branch) and the independence of Eq. (Gamma-ph) of the signs of `uⱼ` | Theorem "density and magnetic", Step 5 |
| `DivergenceAlgebra.lean` | for every `k`, the even and odd parts of `α_k(±u)`, `K(u)`, `A(u)` in `D_k = cosh βu + (-1)^k cosh βμ`, `R_k` and `Φ_k = D_k + εa sinh(βu)/u`; `K_eA_e - K_oA_o = β²e(u)`; where `Φ_k = 0`, `R_k² = 1 + s sinh²(βμ)/a²` and `K_eA_e = β²e(u)`; at a root of `e`, `Z = uP(u)A_e/K_o` (even in `u`), `λρZ = -εUP(u)D_kA_e/(βΦ_k)`, and the branch `K_e = 0` where `Φ_k = 0`, `A_e ≠ 0` | Eqs. (div-alpha), (div-Phi), the selection in Eq. (div-ph), for every `k` |
| `Confluent.lean` | a double root `s₁ = s₂ = u²` (including `u = 0`): Step 3 for every `u` up to a remainder `e(u)Ξ`, termwise differentiation of lattice series (from `Complex.hasSum_deriv_of_summable_norm`), and at `u = 0` difference quotients with Tannery's theorem; the inverse, the limit formula with `∂ₛZ`, which is shown to be the `s`-derivative of the even part `Z(s)`, and the assumptions `DMConfHyp` of the note | Proposition "removable singularities", item 2 |
| `HalfFilling.lean` | `μ = 0`: Proposition "structure" with the mirror `h𝓜`; the sums `T̂ = ∑ ĝ` (also at `û = 0`), `ℓ₀`, `ℓ₂`; the inverse in orbit form (even and odd eigenvalues of the diagonal-plus-mirror part) and Eq. (Gamma-hf), under `zₙ² ≠ ŝ` and, off the mirror fixed points, `zₙ² ≠ ŝ'` (`HFHyp.of_eph`: implied by `e(zₙ) ≠ 0`); `P(û) ≠ 0` automatically | Corollary "half filling" |
| `HalfFillingLZ.lean` | `μ = 0` at a lattice zero `z_c² = ŝ` (including `ŝ = 0` at a fixed point): the limiting inverse, with the orbit removed from `T̂`, and `Γ_r` of Eq. (Gamma-hf-lz) | Proposition "removable singularities", item 4 |
| `UZero.lean` | `U = 0`: `χ↑↓ = 0`, every channel is diagonal or mirror-diagonal with its bubble, and `Γ_r = 0` for every right or left inverse, in all four channels | `U = 0` in the proofs of the theorems |
| `StableForms.lean` | the reformulations used by the Float64 setup of Theorem "density and magnetic": the discriminant of `e` from the weights, the roots and Vieta products, the offsets of the roots from `t₊², t₋², a², w²/4`, `P(u)` as a product of offsets, the factors `z² - s = (x - δ)(x + w + δ)` of the entries, `AB = -ρM` at a root, the alternative forms of `K/ρ` and of `K` at `k = 0` (density), the divided-difference form of `Q(z, v)` off the diagonal. Not covered: `_Y_atomic`, `_tanhdiff`, and `_Q` on the diagonal | `_ph_setup_raw`, `_efactors`, `_Γraw`, `_Q` in `src/doped.jl` |
| `FieldBasic.lean` | the atom with field, `H = U (n↑ - 1/2)(n↓ - 1/2) - μ (n↑ + n↓ - 1) - h (n↑ - n↓)`: parameters `FParams`, the weights `p₀, p↑, p↓, p₂`, the lattice of a centre `c` (`FParams.lat c`), and the parameters `(-U, β, h, μ)` of the Shiba image and `(U, β, μ, -h)` of the spin flip, with their weights | Section F.6, Eq. (field-weights) |
| `FieldLehmann.lean` | the objects of `Lehmann.lean` for the atom with field; `corr`, the full four-point function of any four operators, as six ordered terms; fermionic exchange (`corr_swap`); a unitary conjugation `V H V⁻¹` carries the four-point functions and the connected susceptibility (`corr_conj`, `chiLehmann_conj`); the Shiba map `c↓ ↦ c↓†` and the spin flip as `4 × 4` matrices, which send `H` to the atom at `(-U, h, μ)` and `(U, μ, -h)` | Section F.6 |
| `FieldInput.lean` | the sectors from the Hamiltonian: `chiTUp`, `chiTDn` (transverse), `chiAUp`, `chiADn` (oriented opposite-spin pairs), `chiEUp`, `chiEDn` (equal-spin pairs), `chiL` (coupled longitudinal), and the bubbles of Eq. (field-bubbles); `G↑`, `G↓` in closed form; the general Shiba map `χ_A↑(U, μ, h) = χ_T↑(-U, h, μ)` and `χ_A↓(U, μ, h) = χ_T↑(-U, -h, μ)`; the transverse input `P(zₙ)χ_T↑,nm = -βH(zₙ)δₙₘ + λρ/(zₙ + zₘ) + N(zₙ, zₘ)/P(zₘ)` with the complex transfer `W = iω_k - 2h` (`PT_mul_chiTUp`) | Eqs. (field-A), (field-bubbles), (field-T-data); Proposition "structure" with the transverse data |
| `FieldTAlg.lean` | the algebra of the six Fock-space paths of `χ_T↑`, split by the weights, also on the mirror at `μ = 0` | |
| `FieldTSums.lean`, `FieldTIdentities.lean` | the lattice sums of `PHSums.lean` for the transverse data (the weights of the field atom are a multiple of those of the zero-field atom at the same `μ`), the values of `S_k` at the four zeros of `P` from the weights, and identities (A6), (A7) for the transverse data | |
| `FieldTransverse.lean` | **the transverse vertex** `Γ_T↑`, Eq. (field-T) in the form of Eq. (Gamma-ph) with the transverse data: an explicit two-sided inverse of `χ_T↑` and `β²(χ_T↑⁻¹ - b_T↑⁻¹) = Γ_T↑`, under `TGeneric` | Eq. (field-T) |
| `FieldPair.lean` | the vertices of `A↑`, `T↓`, `A↓` through the Shiba map and the spin flip of the susceptibilities and the bubbles | Eq. (field-A) |
| `FieldEqualSpin.lean` | the equal-spin pair from the Hamiltonian, `χ_E↑ = (β/2)Q(xₙ, x_n̄)(δₙₘ - δ_{m n̄})`; its inverse and that of the bubble on the mirror-odd subspace, and **the equal-spin vertex** `Γ_E↑` of Eq. (field-E); `Γ_E↓` through the spin flip | Eq. (field-E) |
| `FieldTHalfFilling.lean` | the transverse input at `μ = 0`, where the Cauchy matrix is the mirror; **the transverse vertex at `μ = 0`**, Eq. (field-T-axis) (`gammaTHF_eq_note`), under `THFHyp`; the pair vertices at `h = 0` | Eq. (field-T-axis) |
| `FieldTZero.lean` | at `h = 0`, `k = 0`, where `W = 0`: from the Hamiltonian, `χ_T↑ = χ_m` of the zero-field atom (also at `μ = 0`) and `b_T↑ = B_m`, so Theorem "density and magnetic" and Corollary "half filling" give the transverse vertex there; the opposite-spin pair at `μ = 0`, `k = 0` through the Shiba map | Eq. (field-T) at `h = 0`, `k = 0` |
| `FieldLInput.lean` | the longitudinal input from the Hamiltonian: `P(z)χ↑↓(z, v) = λ[ρ/(z + v) - m/(z - v)] + (N₀↓(v) + z²N₂↓(v))/P(v)` off the axes `μ = 0`, `h = 0`, and `P(z)χ↑↑(z, v) = -βH↑(z)δ_zv + δ_k0 ζ↑(z² - a²)(v² - a²)/P(v)` for all `μ`, `h`; the rows of spin `↓` through the spin flip | Eq. (field-L-data), the input of Eq. (field-L) |
| `DividedDifferences.lean` | off the diagonal, each susceptibility as a divided difference of one function of one frequency, from the inputs: `(zₙ² - zₘ²)χ_r,nm = εU[f_r(zₙ) - f_r(zₘ)]` (density and magnetic), `(tₙ - tₘ)χ_s,nm = -U[f_s(n) - f_s(m)]` (singlet, `b ≠ 0`), `(zₙ² - zₘ²)χ_T↑,nm = -U[f_T(zₙ) - f_T(zₘ)]` (transverse, `W ≠ 0`), and the longitudinal `χ↑↓` (off the axes `μ = 0`, `h = 0`) and `χ↑↑` on the two spin lattices, with the rows of spin `↓` by the spin flip, all for `n ≠ m`; `f` from `G`: `[G(ν) - G(ν + ω)]/(iω)` for `k ≠ 0` (density and magnetic, longitudinal), `-[G(ν) + G(-ν - ω)]/(2b)` (singlet), `[G↑(ν) - G↓(ν + ω)]/W` (transverse); at `μ = 0` both sides vanish on the mirror, as `f` is even (singlet: on the mirror for all `μ`) | not in the note |
| `StaticResponse.lean` | at `k = 0` the functions `f` of `DividedDifferences.lean` as static responses of the propagator from the Hamiltonian, as derivatives along a line in `(μ, h)`: `f↑↑ = -∂G↑/∂μ↑`, `f↑↓ = -∂G↑/∂μ↓` (`μ↑ = μ + h`, `μ↓ = μ - h`), and at `h = 0` `f_d = -∂G/∂μ`, `f_m = -∂G↑/∂h` | not in the note |
| `WardIdentities.lean` | the functions `f` of `DividedDifferences.lean` as row sums of the susceptibilities from the Hamiltonian, `∑ₘ χ_nm = -βf(zₙ)`: density and magnetic (`U ≠ 0`, `μ ≠ 0`), singlet (`b ≠ 0`), transverse (`U ≠ 0`, `μ ≠ 0`, `W ≠ 0`), and all four longitudinal blocks, `∑ₘ χ_στ(z_σₙ, z_τₘ) = -βf_στ(z_σₙ)` (opposite spins off the axes), with the sums over the other spin's lattice through `S↓(±z↑ₙ)` (`coth βh`, `coth βμ`) and `S₀(±a)` of each spin lattice from the weights | not in the note |
| `SpectralIdentity.lean` | density and magnetic: for `g_s(n) = 1/(zₙ² - s)` with `s` off the lattice, `∑ₙ g_s(n) = 𝒯(s)`, and for `U ≠ 0`, `μ ≠ 0`, `∑ₘ χ_r,nm g_s(m) = ℓ(s)g_s(n) - β[1 - (εU/β)𝒯(s)]f_r(zₙ)g_s(n)`, `ℓ(s) = -εU ∑ₘ f_r(zₘ)g_s(m)`; where `1 - (εU/β)𝒯(s) = 0`, `g_s` is an eigenvector with eigenvalue `ℓ(s)` | not in the note |
| `SpectralLink.lean` | density and magnetic: `1 - (εU/β)𝒯(s) = Φ_k(s)/D_k(s)` for `u ≠ 0`, `D_k(u) ≠ 0`; for `U ≠ 0`, `P(s) ≠ 0`, the eigenvalue `ℓ(s) = -K_e(s)/P(s) + βf_e(u)[1 - (εU/β)𝒯(s)]` with `f_e(u) = (f_r(u) + f_r(-u))/2`; so for `μ ≠ 0` every `g_s` with `Φ_k(s) = 0` and `P(s) ≠ 0` is an eigenvector of `χ_r` with eigenvalue `-K_e(s)/P(s)` | not in the note |
| `ChiBounds.lean` | density and magnetic, `μ ≠ 0`: the entries of `χ_r` decay as those of `diag(-β/zₙ²)`, `\|χ_r,nm\| ≤ c\|zₙ\|⁻²\|zₘ\|⁻²` for `n ≠ m` and `\|χ_r,nn + β/zₙ²\| ≤ c\|zₙ\|⁻³` | proof of Proposition "finite boxes", from Proposition "structure" |
| `HilbertSchmidt.lean` | a matrix on `ℤ` with square-summable entries is a compact operator on `ℓ²(ℤ)`, the norm limit of its finite-rank row truncations | not in the note |
| `ChiCompact.lean` | density and magnetic, `μ ≠ 0`: `χ_r = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)` with `𝒟 = diag(-β/zₙ²)` and `𝒦` Hilbert–Schmidt, hence compact; the bubble `B = 𝒟^(1/2)(1 + 𝒦₀)𝒟^(1/2)` with `1 + 𝒦₀` diagonal and invertible on `ℓ²(ℤ)` | proof of Proposition "finite boxes" |
| `GammaOperator.lean` | density and magnetic, `μ ≠ 0`: `Γ_r = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)` defined entrywise; `1 + 𝒦` is invertible iff `ker(1 + 𝒦) = 0` (Fredholm alternative) iff `χ_r` has no nonzero null vector `v` with `∑ₙ \|vₙ/zₙ\|² < ∞`; such null vectors are bounded | not in the note |
| `GammaClosedForm.lean` | on the domain of Theorem "density and magnetic" (generic case): `1 + 𝒦` is invertible and the operator `Γ_r` of `GammaOperator.lean` is the closed form of Eq. (Gamma-ph); the columns of `χ_r⁻¹` tend to `2εa/β²` | Theorem "density and magnetic" |
| `GammaBoxes.lean` | density and magnetic, `μ ≠ 0`: the compressions of `1 + 𝒦` to the boxes `\|2n + 1 + k\| ≤ 2M + 1` converge in norm; wherever `1 + 𝒦` is invertible, the box matrices `χ_r\|_M` are invertible for large `M` and the box vertices `β²[(χ_r\|_M)⁻¹ - B⁻¹]` converge entrywise to `Γ_r`, on the domain of Theorem "density and magnetic" (generic case) to its closed form | Proposition "finite boxes" |
| `GammaContinuity.lean` | density and magnetic: `Γ_r` is continuous in `(U, β, μ)` wherever `μ ≠ 0` and `1 + 𝒦` is invertible, in particular on the domain of Theorem "density and magnetic" (generic case); `𝒦` is continuous in the Hilbert–Schmidt norm, by a uniform bound `\|𝒦_nm\|² ≤ C\|zₙ\|⁻²(\|zₘ\|⁻² + δₙₘ)` and dominated convergence | not in the note |

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

For the atom with field:
the definitions of `FieldLehmann.lean` (`FParams.H` and the objects built from it, as in
`Lehmann.lean`, and `corr`) and the parameters of `FieldBasic.lean`;
the sectors and bubbles of `FieldInput.lean`
(`chiTUp`, `chiTDn`, `chiAUp`, `chiADn`, `chiEUp`, `chiEDn`, `chiL`,
`bTUp`, `bTDn`, `bAUp`, `bADn`, `bE`, `bL`);
the closed forms `gammaTUp`, `gammaTHF`, `gammaTHFnote` and `gammaEUp`
(with the transverse data of `FieldInput.lean` and `DDT`, `ZT`, `KT`, `ThatT`, `DhatT`),
and the longitudinal input data of `FieldLInput.lean`;
and the assumptions `TGeneric`, `THFHyp` and that of `gammaEUp_eq`.
The matrices `shibaV`, `flipV` need no trust:
Lean checks how they conjugate `c↑`, `c↓` and `H`.

## Not covered (yet)

* Lemma "continuity" and the limits themselves: at the lattice zeros of Theorem "singlet" and
  Corollary "half filling" the limiting inverses are written down and verified, but that they are
  the limits of the closed forms is not proved. The lattice zeros `e(zₙ) = 0` of Theorem "density
  and magnetic" at `μ ≠ 0`. Proposition "finite boxes".
* That `χ_s` and `χ_r` have no inverses other than the constructed ones. For the block-diagonal `χ_t`
  this is automatic; the note does not claim it for the other channels either.
* Anything about rounding: `StableForms.lean` and `Thermal.lean` prove that the listed
  reformulations are exact identities, not that its loss estimates are right.
* With field:
  the coupled longitudinal vertex, Eq. (field-L), of which only the input is formalized.
  `TGeneric` assumes the conditions of Eq. (field-T)
  (`U ≠ 0`, `μ ≠ 0`, `e(zₙ) ≠ 0`, `s₁ ≠ s₂`, `Kⱼ ≠ 0` except for a root that is omitted)
  and `W ≠ 0` (the case `W = 0` is `FieldTZero.lean`);
  `THFHyp` assumes `W ≠ 0` besides the note's conditions.
  Roots at a zero of `P` are included,
  because the inverse carries `P(uⱼ)` (`P(û)` at `μ = 0`) as a factor:
  Step 3 is used multiplied by `P(u)` (`hasSum_step3`),
  and at `μ = 0` the image of `P(û)ĝ` (`THFHyp.hasSum_NT_ghat`).
  So is `W = ±2a` (`k = 0`, `|h| = |U|/2`), where `q₀ = a² - W²/4 = 0` and `P(t) = t²(t² - 4a²)`
  has a double zero:
  the lattice sums of `M/P` take the values of `hasSum_cubicT` at `q₀ = 0`,
  and identities (A6), (A7) hold with them (`q₀T_zero_data`).
  There `q₁ = -4a²p↑` with `p↑² = p₀p₂` (`h = -a`; `p↓` for `h = a`),
  so `e(0) = 0`, and one root is `u = 0` (`û = 0` at `μ = 0`), at the double zero of `P`.
  There also `K(0) = 0`, and Eq. (field-T) omits this root:
  like the atomic pair of `DMGeneric`, it satisfies the alternative of `ModeOKT`,
  `P(u) = 0` and `W(z; u²) = 0`, instead of `K(u) ≠ 0`
  (`q₀T_zero_root`, `TGeneric.of_q₀T_zero`),
  and its term in the inverse vanishes.
  The pair vertices of `FieldPair.lean` inherit all of this.
  The equal-spin vertex assumes `ℒ(tₙ) ≠ 0` for all `n ≠ n̄`,
  where the note states Eq. (field-E) entrywise.
  At `h = 0`, `k = 0` (`W = 0`) the transverse vertex is that of the zero-field magnetic channel
  (`FieldTZero.lean`), under the zero-field assumptions.
  Not covered are the transverse exceptional sets
  (confluent roots, lattice roots),
  the axis reductions of Eqs. (field-h0), (field-mu0) as identities between closed forms,
  and the divergences of Section G.5.

## Build

In this directory, with the toolchain of `lean-toolchain` (via elan) and Mathlib at the revision of
`lake-manifest.json`:

```sh
lake exe cache get   # prebuilt Mathlib
lake build
./scripts/check.sh   # about 4 min once Mathlib is built
```

CI does not build the Lean project:
green CI checks the Julia package only, and the proofs are checked by running `scripts/check.sh`
by hand.
Its memory peak, measured as the `memory.peak` of its cgroup, is about 8 GiB,
plus about 6 GiB of Mathlib's `.olean` files, which every Lean process maps and shares
(so plan for about 14 GiB).
Compiled alone, each module needs at most 4.3 GiB besides the mapped Mathlib files
(`FieldTIdentities.lean`; most need less than 1 GiB).
`SERIAL=1 ./scripts/check.sh` compiles the modules one at a time,
which lowers the peak to about 6 GiB plus the mapped files, in about 7 min.
