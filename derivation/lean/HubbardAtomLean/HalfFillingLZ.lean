import HubbardAtomLean.HalfFilling
import HubbardAtomLean.SingletLZ

/-!
# Corollary "half filling" at a lattice zero `z_c² = ŝ`

At `μ = 0` the zeros of `e(zₙ) = (zₙ² - ŝ)(zₙ² - ŝ')` with `zₙ² = ŝ'` and `n ≠ n̄` are genuine
divergences (the mirror-odd vector is a null vector), and those at a mirror fixed point are covered
by `HFHyp` (orbit form). Here the remaining zeros `z_c² = ŝ` on one orbit `O = {c, c̄}`
(`r ∈ {1, 2}` elements, including `ŝ = 0` at a fixed point): the even eigenvalue vanishes on
`O`, `û = z_c` and `D̂` diverge, and the separable term of Eq. (Gamma-hf) drops out off the orbit.
With `ĝ` and the inverse even eigenvalue `0` on `O` (`1/0 = 0`), `ιₙ` the indicator of `O`,
`T̂' = ∑_{l ∉ O} ĝ(zₗ)`, `D̂' = β - 2εaT̂'`:

`χ_r⁻¹ = (orbit inverse) + 2εa/β² + (P(z_c)/(βr))(ιₙĝ(zₘ) + ĝ(zₙ)ιₘ) + y ιₙιₘ`,
`y = P(z_c)D̂'/(2εaβr²) - (κ + 2ŝ)/(βr)`.

* `Params.HFLZ`, `Params.HFLZ.chiR_isRightInverse`, `Params.HFLZ.chiR_isLeftInverse`,
  `Params.HFLZ.gammaHF_eq`.
-/

open Complex

namespace HubbardAtom

/-! ### The cancellations, as rational identities -/

section LZ

variable {β a ε s κ q₀ r Tq c0 N2 l0 l2 z v : ℂ}

/-- `∑ₗ R_nl Z_lm` with `R_nl = N(z, zₗ)/(P(z)P(zₗ))`, `N(z, t) = c₀ + N₂t²`,
`P(t) = t⁴ + κt² + q₀²`, and the separable part `Z` of the inverse; `ιₘ`, `ĝₘ` the column's orbit
indicator and `ĝ`. -/
noncomputable def srz (β a ε s κ q₀ r Tq c0 N2 l0 l2 P ιm gm : ℂ) : ℂ :=
  (2 * ε * a / β ^ 2 * (c0 * l0 + N2 * l2) +
      (s ^ 2 + κ * s + q₀ ^ 2) / (β * r) * (gm * (r * (c0 + N2 * s) / (s ^ 2 + κ * s + q₀ ^ 2)) +
        ιm * ((c0 + N2 * s) / (s ^ 2 + κ * s + q₀ ^ 2) * Tq +
          (N2 - (c0 + N2 * s) / (s ^ 2 + κ * s + q₀ ^ 2) * (κ + s)) * l0 -
          (c0 + N2 * s) / (s ^ 2 + κ * s + q₀ ^ 2) * l2 -
          r * (N2 - (c0 + N2 * s) / (s ^ 2 + κ * s + q₀ ^ 2) * (κ + 2 * s)) /
            (s ^ 2 + κ * s + q₀ ^ 2))) +
      ((s ^ 2 + κ * s + q₀ ^ 2) * (β - 2 * ε * a * Tq) / (2 * ε * a * β * r ^ 2) -
        (κ + 2 * s) / (β * r)) * ιm * (r * (c0 + N2 * s) / (s ^ 2 + κ * s + q₀ ^ 2))) / P

variable (hβ : β ≠ 0) (ha : a ≠ 0) (hε : ε ≠ 0) (hr : r ≠ 0)
  (hPc : s ^ 2 + κ * s + q₀ ^ 2 ≠ 0)
include hβ ha hε hr hPc

/-- Both off the orbit. -/
theorem hflz_off (hA6 : -β * (z ^ 2 - s) + (c0 * l0 + N2 * l2) = β * N2 / (2 * ε * a))
    (hPz : z ^ 4 + κ * z ^ 2 + q₀ ^ 2 ≠ 0) (hv : v ^ 2 - s ≠ 0)
    (hPv : v ^ 4 + κ * v ^ 2 + q₀ ^ 2 ≠ 0) :
    -(β * (z ^ 2 - s)) / (z ^ 4 + κ * z ^ 2 + q₀ ^ 2) * (2 * ε * a / β ^ 2) +
        (c0 + N2 * v ^ 2) / ((z ^ 4 + κ * z ^ 2 + q₀ ^ 2) * (v ^ 4 + κ * v ^ 2 + q₀ ^ 2)) *
          -((v ^ 4 + κ * v ^ 2 + q₀ ^ 2) / (β * (v ^ 2 - s))) +
      srz β a ε s κ q₀ r Tq c0 N2 l0 l2 (z ^ 4 + κ * z ^ 2 + q₀ ^ 2) 0 (1 / (v ^ 2 - s)) = 0 := by
  have hS : c0 * l0 + N2 * l2 = β * N2 / (2 * ε * a) + β * (z ^ 2 - s) := by
    linear_combination hA6
  simp only [srz]
  rw [hS]
  generalize s ^ 2 + κ * s + q₀ ^ 2 = Pc at hPc ⊢
  generalize z ^ 4 + κ * z ^ 2 + q₀ ^ 2 = Pz at hPz ⊢
  generalize v ^ 4 + κ * v ^ 2 + q₀ ^ 2 = Pv at hPv ⊢
  field_simp
  ring

/-- The row on the orbit (`z² = s`), the column off it. -/
theorem hflz_row (hA6 : c0 * l0 + N2 * l2 = β * N2 / (2 * ε * a)) (hv : v ^ 2 - s ≠ 0)
    (hPv : v ^ 4 + κ * v ^ 2 + q₀ ^ 2 ≠ 0) :
    (c0 + N2 * v ^ 2) / ((s ^ 2 + κ * s + q₀ ^ 2) * (v ^ 4 + κ * v ^ 2 + q₀ ^ 2)) *
        -((v ^ 4 + κ * v ^ 2 + q₀ ^ 2) / (β * (v ^ 2 - s))) +
      srz β a ε s κ q₀ r Tq c0 N2 l0 l2 (s ^ 2 + κ * s + q₀ ^ 2) 0 (1 / (v ^ 2 - s)) = 0 := by
  simp only [srz]
  rw [hA6]
  generalize s ^ 2 + κ * s + q₀ ^ 2 = Pc at hPc ⊢
  generalize v ^ 4 + κ * v ^ 2 + q₀ ^ 2 = Pv at hPv ⊢
  field_simp
  ring

/-- The row off the orbit, the column on it. -/
theorem hflz_col (hA6 : -β * (z ^ 2 - s) + (c0 * l0 + N2 * l2) = β * N2 / (2 * ε * a))
    (hHF : 2 * ε * a * ((N2 * (s ^ 2 + κ * s + q₀ ^ 2) - (c0 + N2 * s) * (κ + s)) * l0 -
      (c0 + N2 * s) * l2) = 2 * ε * a * β * (s ^ 2 + κ * s + q₀ ^ 2) - β * (c0 + N2 * s))
    (hPz : z ^ 4 + κ * z ^ 2 + q₀ ^ 2 ≠ 0) (hz : z ^ 2 - s ≠ 0) :
    -(β * (z ^ 2 - s)) / (z ^ 4 + κ * z ^ 2 + q₀ ^ 2) *
        (2 * ε * a / β ^ 2 + (s ^ 2 + κ * s + q₀ ^ 2) / (β * r) * (1 / (z ^ 2 - s))) +
      srz β a ε s κ q₀ r Tq c0 N2 l0 l2 (z ^ 4 + κ * z ^ 2 + q₀ ^ 2) 1 0 = 0 := by
  have key : -(β * (z ^ 2 - s)) / (z ^ 4 + κ * z ^ 2 + q₀ ^ 2) *
        (2 * ε * a / β ^ 2 + (s ^ 2 + κ * s + q₀ ^ 2) / (β * r) * (1 / (z ^ 2 - s))) +
      srz β a ε s κ q₀ r Tq c0 N2 l0 l2 (z ^ 4 + κ * z ^ 2 + q₀ ^ 2) 1 0 =
      2 * ε * a / (β ^ 2 * (z ^ 4 + κ * z ^ 2 + q₀ ^ 2)) *
          (-β * (z ^ 2 - s) + (c0 * l0 + N2 * l2) - β * N2 / (2 * ε * a)) +
        1 / (2 * a * β * ε * r * (z ^ 4 + κ * z ^ 2 + q₀ ^ 2)) *
          (2 * ε * a * ((N2 * (s ^ 2 + κ * s + q₀ ^ 2) - (c0 + N2 * s) * (κ + s)) * l0 -
            (c0 + N2 * s) * l2) - (2 * ε * a * β * (s ^ 2 + κ * s + q₀ ^ 2) -
              β * (c0 + N2 * s))) := by
    simp only [srz]
    generalize s ^ 2 + κ * s + q₀ ^ 2 = Pc at hPc ⊢
    generalize z ^ 4 + κ * z ^ 2 + q₀ ^ 2 = Pz at hPz ⊢
    field_simp
    ring
  rw [key, hA6, hHF]
  ring

/-- Both on the orbit: the row sum is `1/r`. -/
theorem hflz_orb (hA6 : c0 * l0 + N2 * l2 = β * N2 / (2 * ε * a))
    (hHF : 2 * ε * a * ((N2 * (s ^ 2 + κ * s + q₀ ^ 2) - (c0 + N2 * s) * (κ + s)) * l0 -
      (c0 + N2 * s) * l2) = 2 * ε * a * β * (s ^ 2 + κ * s + q₀ ^ 2) - β * (c0 + N2 * s)) :
    srz β a ε s κ q₀ r Tq c0 N2 l0 l2 (s ^ 2 + κ * s + q₀ ^ 2) 1 0 = 1 / r := by
  have key : srz β a ε s κ q₀ r Tq c0 N2 l0 l2 (s ^ 2 + κ * s + q₀ ^ 2) 1 0 - 1 / r =
      2 * ε * a / (β ^ 2 * (s ^ 2 + κ * s + q₀ ^ 2)) *
          ((c0 * l0 + N2 * l2) - β * N2 / (2 * ε * a)) +
        1 / (2 * a * β * ε * r * (s ^ 2 + κ * s + q₀ ^ 2)) *
          (2 * ε * a * ((N2 * (s ^ 2 + κ * s + q₀ ^ 2) - (c0 + N2 * s) * (κ + s)) * l0 -
            (c0 + N2 * s) * l2) - (2 * ε * a * β * (s ^ 2 + κ * s + q₀ ^ 2) -
              β * (c0 + N2 * s))) := by
    simp only [srz]
    generalize s ^ 2 + κ * s + q₀ ^ 2 = Pc at hPc ⊢
    field_simp
    ring
  rw [hA6, hHF, sub_self, sub_self, mul_zero, mul_zero, add_zero] at key
  linear_combination key

end LZ

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ) (c : ℤ)

/-- The even eigenvalue `-β(z² - ŝ)/P(z)` of the diagonal-plus-mirror part. -/
noncomputable def bev (z : ℂ) : ℂ := -(A.β * (z ^ 2 - A.shat k ε)) / A.Pph k z

/-- The odd eigenvalue `-β(z² - ŝ')/P(z)`. -/
noncomputable def bodd (z : ℂ) : ℂ := -(A.β * (z ^ 2 - A.shat' k ε)) / A.Pph k z

/-- `T̂' = ∑_{l ∉ O} ĝ(zₗ)` (the orbit terms are `0`). -/
noncomputable def That' : ℂ := ∑' l, A.ghat k ε (A.z k l)

/-- `D̂' = β - 2εaT̂'`. -/
noncomputable def Dhat' : ℂ := A.β - 2 * ε * A.a * A.That' k ε

/-- `y = P(z_c)D̂'/(2εaβr²) - (κ + 2ŝ)/(βr)`. -/
noncomputable def yhf : ℂ :=
  A.Pph k (A.z k c) * A.Dhat' k ε / (2 * ε * A.a * A.β * orbSize k c ^ 2) -
    (A.κ k + 2 * A.shat k ε) / (A.β * orbSize k c)

/-- The separable part of the inverse at the lattice zero. -/
noncomputable def Zhf (n m : ℤ) : ℂ :=
  2 * ε * A.a / A.β ^ 2 + A.Pph k (A.z k c) / (A.β * orbSize k c) *
      (orb k c n * A.ghat k ε (A.z k m) + A.ghat k ε (A.z k n) * orb k c m) +
    A.yhf k ε c * orb k c n * orb k c m

/-- The inverse of `χ_r` at the lattice zero `z_c² = ŝ`. -/
noncomputable def invHFlz (n m : ℤ) : ℂ :=
  mirrorBlock k (fun j ↦ A.ievn k ε (A.z k j)) (fun j ↦ A.iodd k ε (A.z k j)) n m +
    A.Zhf k ε c n m

/-- The closed form at the lattice zero: Eq. (Gamma-hf) in orbit form without the separable term,
plus `βP(z_c)(ιₙĝ(zₘ) + ĝ(zₙ)ιₘ)/r + β²yιₙιₘ`. -/
noncomputable def gammaHFlz (n m : ℤ) : ℂ :=
  A.β * A.Pph k (A.z k n) / A.G₂ k (A.z k n) * kron n m +
    A.β ^ 2 * mirrorBlock k (fun j ↦ A.ievn k ε (A.z k j)) (fun j ↦ A.iodd k ε (A.z k j)) n m +
    A.β ^ 2 * A.Zhf k ε c n m

/-- The assumptions at a lattice zero `z_c² = ŝ` of Corollary "half filling" (with `U ≠ 0`); the
odd eigenvalue stays nonzero off the mirror fixed points. -/
structure HFLZ : Prop where
  ε_eq : ε = 1 ∨ ε = -1
  a_ne : (A.a : ℂ) ≠ 0
  μ_eq : A.μ = 0
  z_c : A.z k c ^ 2 - A.shat k ε = 0
  odd_ne : ∀ n, n ≠ mirror k n → A.z k n ^ 2 - A.shat' k ε ≠ 0

theorem bev_neg (z : ℂ) : A.bev k ε (-z) = A.bev k ε z := by
  rw [bev, bev, A.Pph_neg, neg_sq]

theorem bodd_neg (z : ℂ) : A.bodd k ε (-z) = A.bodd k ε z := by
  rw [bodd, bodd, A.Pph_neg, neg_sq]

/-- `χ_r = (bev 𝓜₊ + bodd 𝓜₋) + N(zₙ, zₘ)/(P(zₙ)P(zₘ))` at `μ = 0`. -/
theorem chiR_hf_eq (hμ : A.μ = 0) (n m : ℤ) :
    A.chiR k ε n m = mirrorBlock k (fun j ↦ A.bev k ε (A.z k j)) (fun j ↦ A.bodd k ε (A.z k j))
      n m + A.Nph k ε (A.z k n) (A.z k m) / (A.Pph k (A.z k n) * A.Pph k (A.z k m)) := by
  have hPn := A.Pph_z_ne_zero k n
  have hPm := A.Pph_z_ne_zero k m
  have h := A.Pph_mul_chiR_hf k ε hμ n m
  have hc : A.chiR k ε n m = (-A.β * A.Hph k ε (A.z k n) * kron n m +
      A.lam ε * (A.β * A.p₀) * kron m (mirror k n) +
      A.Nph k ε (A.z k n) (A.z k m) / A.Pph k (A.z k m)) / A.Pph k (A.z k n) := by
    rw [eq_div_iff hPn, mul_comm]
    exact h
  rw [hc, mirrorBlock, A.Hph_of_mu k ε hμ, A.lam_p₀ k ε]
  simp only [bev, bodd]
  field_simp
  ring

variable {A k ε c}

theorem HFLZ.zero_iff (h : A.HFLZ k ε c) {n : ℤ} :
    A.z k n ^ 2 - A.shat k ε = 0 ↔ n = c ∨ n = mirror k c := by
  have hzm := A.z_mirror_of_mu k h.μ_eq c
  constructor
  · intro h0
    have h1 : (A.z k n - A.z k c) * (A.z k n + A.z k c) = 0 := by
      linear_combination h0 - h.z_c
    rcases mul_eq_zero.mp h1 with h2 | h2
    · exact Or.inl (A.z_injective k (sub_eq_zero.mp h2))
    · exact Or.inr (A.z_injective k (by rw [hzm]; linear_combination h2))
  · rintro (rfl | rfl)
    · exact h.z_c
    · rw [hzm, neg_sq]
      exact h.z_c

theorem HFLZ.ne_of_not_mem (h : A.HFLZ k ε c) {n : ℤ} (hn : ¬(n = c ∨ n = mirror k c)) :
    A.z k n ^ 2 - A.shat k ε ≠ 0 := fun h0 ↦ hn (h.zero_iff.mp h0)

theorem HFLZ.ghat_orb (h : A.HFLZ k ε c) {n : ℤ} (hn : n = c ∨ n = mirror k c) :
    A.ghat k ε (A.z k n) = 0 := by
  rw [ghat, h.zero_iff.mpr hn, div_zero]

theorem HFLZ.ievn_orb (h : A.HFLZ k ε c) {n : ℤ} (hn : n = c ∨ n = mirror k c) :
    A.ievn k ε (A.z k n) = 0 := by
  rw [ievn, h.zero_iff.mpr hn, mul_zero, div_zero, neg_zero]

theorem HFLZ.bev_orb (h : A.HFLZ k ε c) {n : ℤ} (hn : n = c ∨ n = mirror k c) :
    A.bev k ε (A.z k n) = 0 := by
  rw [bev, h.zero_iff.mpr hn, mul_zero, neg_zero, zero_div]

/-- `∑_{l ∉ O} ĝ(zₗ)` converges: `ĝ(zₗ) = 1/((zₗ - z_c)(zₗ + z_c))`. -/
theorem HFLZ.summable_ghat (h : A.HFLZ k ε c) : Summable (fun l ↦ A.ghat k ε (A.z k l)) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hc : (2 * Real.pi / A.β * I : ℂ) ≠ 0 :=
    mul_ne_zero (div_ne_zero (mul_ne_zero two_ne_zero
      (ofReal_ne_zero.mpr Real.pi_ne_zero)) hβ) I_ne_zero
  have hs := summable_inv_mul_linear hc (A.μ + (1 + k) * Real.pi / A.β * I - A.z k c)
    (A.μ + (1 + k) * Real.pi / A.β * I + A.z k c)
  convert hs using 1
  funext l
  have hz : A.z k l = l * (2 * Real.pi / A.β * I) + (A.μ + (1 + k) * Real.pi / A.β * I) := by
    rw [A.z_eq]
    push_cast
    ring
  rw [ghat, show A.z k l ^ 2 - A.shat k ε = (A.z k l - A.z k c) * (A.z k l + A.z k c) by
    linear_combination h.z_c, hz, one_div]
  ring_nf

/-- `∑ₗ N(z, zₗ)ĝ(zₗ)/P(zₗ)` with the orbit removed, by the partial fractions of
`HFHyp.hasSum_Nph_ghat`. -/
theorem HFLZ.hasSum_Nph_ghat (h : A.HFLZ k ε c) (z : ℂ) :
    HasSum (fun l ↦ A.Nph k ε z (A.z k l) * A.ghat k ε (A.z k l) / A.Pph k (A.z k l))
      ((A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε) /
            (A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2) * A.That' k ε +
        (A.N₂ k ε z - (A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε) /
            (A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2) * (A.κ k + A.shat k ε)) * A.ℓ₀ k -
        (A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε) /
            (A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2) * A.ℓ₂ k -
        orbSize k c * (A.N₂ k ε z - (A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε) /
            (A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2) * (A.κ k + 2 * A.shat k ε)) /
          (A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2)) := by
  have hμ := h.μ_eq
  have ha := h.a_ne
  have hPc : A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 ≠ 0 := by
    have := A.Pph_z_ne_zero k c
    rwa [A.Pph_eq_poly, show A.z k c ^ 4 = (A.z k c ^ 2) ^ 2 by ring,
      show A.z k c ^ 2 = A.shat k ε by linear_combination h.z_c] at this
  set D := A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 with hD
  set cA := (A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε) / D with hcA
  have ho := hasSum_mul_orb k c (f := fun l ↦ (A.N₂ k ε z - cA * (A.κ k + 2 * A.shat k ε)) /
    A.Pph k (A.z k l)) (fun l ↦ by rw [A.z_mirror_of_mu k hμ, A.Pph_neg])
  convert ((((h.summable_ghat.hasSum.mul_left cA).add ((A.hasSum_ℓ₀ k ha).mul_left
    (A.N₂ k ε z - cA * (A.κ k + A.shat k ε)))).add
    ((A.hasSum_ℓ₂ k ha).mul_left (-cA))).sub ho) using 1
  · funext l
    have h₂ := A.Pph_z_ne_zero k l
    rw [A.Nph_of_mu k ε hμ, A.Pph_eq_poly] at *
    by_cases hl : l = c ∨ l = mirror k c
    · rw [h.ghat_orb hl, orb_of_mem hl, show A.z k l ^ 2 = A.shat k ε by
        linear_combination h.zero_iff.mpr hl]
      rw [show A.z k l ^ 4 = A.shat k ε ^ 2 by
        rw [show A.z k l ^ 4 = (A.z k l ^ 2) ^ 2 by ring,
          show A.z k l ^ 2 = A.shat k ε by linear_combination h.zero_iff.mpr hl]]
      rw [hcA]
      field_simp
      ring
    · have h₁ := h.ne_of_not_mem hl
      rw [orb_of_not_mem hl, ghat]
      rw [hcA]
      generalize hPl : A.z k l ^ 4 + A.κ k * A.z k l ^ 2 + A.q₀ k ^ 2 = Pl at h₂ ⊢
      generalize hG : A.z k l ^ 2 - A.shat k ε = G at h₁ ⊢
      field_simp
      rw [hD, ← hPl, ← hG]
      ring
  · have hPcD : A.Pph k (A.z k c) = D := by
      rw [A.Pph_eq_poly, hD, show A.z k c ^ 4 = (A.z k c ^ 2) ^ 2 by ring,
        show A.z k c ^ 2 = A.shat k ε by linear_combination h.z_c]
    rw [That', hPcD]
    ring

theorem Zhf_mirror (hμ : A.μ = 0) (l m : ℤ) : A.Zhf k ε c (mirror k l) m = A.Zhf k ε c l m := by
  unfold Zhf
  rw [orb_mirror, A.z_mirror_of_mu k hμ, A.ghat_neg]

theorem Zhf_symm (n m : ℤ) : A.Zhf k ε c n m = A.Zhf k ε c m n := by
  unfold Zhf
  ring

theorem HFLZ.invHFlz_symm (h : A.HFLZ k ε c) (n m : ℤ) :
    A.invHFlz k ε c n m = A.invHFlz k ε c m n := by
  rw [invHFlz, invHFlz, Zhf_symm, mirrorBlock_symm
    (fun j ↦ by rw [A.z_mirror_of_mu k h.μ_eq, A.ievn_neg])
    (fun j ↦ by rw [A.z_mirror_of_mu k h.μ_eq, A.iodd_neg])]

/-- **Corollary "half filling"** at a lattice zero `z_c² = ŝ`, the inverse: `χ_r invHFlz = 1`. -/
theorem HFLZ.chiR_isRightInverse (h : A.HFLZ k ε c) :
    IsRightInverse (A.chiR k ε) (A.invHFlz k ε c) := by
  intro n m
  have hμ := h.μ_eq
  have ha := h.a_ne
  have hβ := A.β_ne_zero
  have hε0 : ε ≠ 0 := by rcases h.ε_eq with rfl | rfl <;> norm_num
  have hr : orbSize k c ≠ 0 := by unfold orbSize; split_ifs <;> norm_num
  have hzc : A.z k c ^ 2 = A.shat k ε := by linear_combination h.z_c
  have hPcD : A.Pph k (A.z k c) = A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 := by
    rw [A.Pph_eq_poly, show A.z k c ^ 4 = (A.z k c ^ 2) ^ 2 by ring, hzc]
  have hPc : A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 ≠ 0 := by
    rw [← hPcD]
    exact A.Pph_z_ne_zero k c
  have hev : ∀ (f : ℂ → ℂ), (∀ z, f (-z) = f z) → ∀ j, f (A.z k (mirror k j)) = f (A.z k j) :=
    fun f hf j ↦ by rw [A.z_mirror_of_mu k hμ, hf]
  set X := mirrorBlock k (fun j ↦ A.bev k ε (A.z k j)) (fun j ↦ A.bodd k ε (A.z k j)) with hX
  set Xi := mirrorBlock k (fun j ↦ A.ievn k ε (A.z k j)) (fun j ↦ A.iodd k ε (A.z k j)) with hXi
  have h₁ : HasSum (fun l ↦ X n l * Xi l m) (mirrorBlock k
      (fun j ↦ A.bev k ε (A.z k j) * A.ievn k ε (A.z k j))
      (fun j ↦ A.bodd k ε (A.z k j) * A.iodd k ε (A.z k j)) n m) :=
    hasSum_mirrorBlock_mul (k := k) (B := fun j ↦ A.bev k ε (A.z k j))
      (L := fun j ↦ A.bodd k ε (A.z k j)) (B' := fun j ↦ A.ievn k ε (A.z k j))
      (L' := fun j ↦ A.iodd k ε (A.z k j)) (hev _ (A.ievn_neg k ε)) (hev _ (A.iodd_neg k ε)) n m
  have h₂ : HasSum (fun l ↦ X n l * A.Zhf k ε c l m) (A.bev k ε (A.z k n) * A.Zhf k ε c n m) :=
    hasSum_mirrorBlock_mul_even (v := fun l ↦ A.Zhf k ε c l m) (fun l ↦ Zhf_mirror hμ l m) n
  set R : ℤ → ℂ := fun l ↦ A.Nph k ε (A.z k n) (A.z k l) /
    (A.Pph k (A.z k n) * A.Pph k (A.z k l)) with hR
  have hRev : ∀ l, R (mirror k l) = R l := fun l ↦ by
    simp only [hR, A.z_mirror_of_mu k hμ, A.Nph_neg_of_mu k ε hμ, A.Pph_neg]
  have h₃ : HasSum (fun l ↦ R l * Xi l m) (A.ievn k ε (A.z k m) * R m) := by
    have := hasSum_mirrorBlock_mul_even (k := k) (B := fun j ↦ A.ievn k ε (A.z k j))
      (L := fun j ↦ A.iodd k ε (A.z k j)) (v := R) hRev m
    convert this using 1
    funext l
    rw [hXi, mirrorBlock_symm (hev _ (A.ievn_neg k ε)) (hev _ (A.iodd_neg k ε)) l m, mul_comm]
  -- the lattice sums
  have hPn := A.Pph_z_ne_zero k n
  have hS1 := (A.hasSum_Nph_div k ε ha hμ (A.z k n)).div_const (A.Pph k (A.z k n))
  have hSi := (hasSum_mul_orb k c (f := fun l ↦ A.Nph k ε (A.z k n) (A.z k l) / A.Pph k (A.z k l))
    (fun l ↦ by rw [A.z_mirror_of_mu k hμ, A.Nph_neg_of_mu k ε hμ, A.Pph_neg])).div_const
    (A.Pph k (A.z k n))
  have hSg := (h.hasSum_Nph_ghat (A.z k n)).div_const (A.Pph k (A.z k n))
  set ιm := orb k c m with hιm
  set gm := A.ghat k ε (A.z k m) with hgm
  set α := A.Pph k (A.z k c) / (A.β * orbSize k c) with hα
  have h₄ : HasSum (fun l ↦ R l * A.Zhf k ε c l m)
      (srz A.β A.a ε (A.shat k ε) (A.κ k) (A.q₀ k) (orbSize k c) (A.That' k ε)
        (A.cN₀ k ε (A.z k n)) (A.N₂ k ε (A.z k n)) (A.ℓ₀ k) (A.ℓ₂ k) (A.Pph k (A.z k n)) ιm
        gm) := by
    convert (((hS1.mul_left (2 * ε * A.a / A.β ^ 2)).add (hSi.mul_left (α * gm))).add
      (hSg.mul_left (α * ιm))).add (hSi.mul_left (A.yhf k ε c * ιm)) using 1
    · funext l
      simp only [hR, Zhf]
      ring
    · rw [srz, yhf, Dhat', hα, hPcD, A.Nph_of_mu k ε hμ, hzc]
      field_simp
      ring
  convert ((h₁.add h₂).add h₃).add h₄ using 1
  · funext l
    rw [A.chiR_hf_eq k ε hμ, invHFlz]
    ring
  · -- the remainder vanishes, block by block
    have hA6 := A.identity_A6_hf k ε h.ε_eq ha hμ (A.z k n)
    rw [show -(A.β : ℂ) * A.Hph k ε (A.z k n) + A.lam ε * (A.β * A.p₀) =
      -A.β * (A.z k n ^ 2 - A.shat k ε) by
        rw [← A.Hph_sub_of_mu k ε hμ]; ring] at hA6
    have hHF := A.identity_hf k ε h.ε_eq ha hμ (A.z k n)
    have hPz : A.Pph k (A.z k n) = A.z k n ^ 4 + A.κ k * A.z k n ^ 2 + A.q₀ k ^ 2 :=
      A.Pph_eq_poly k _
    have hPz' := hPn
    rw [hPz] at hPz'
    have hodd : ∀ j, j ≠ mirror k j →
        A.bodd k ε (A.z k j) * A.iodd k ε (A.z k j) = 1 := fun j hj ↦ by
      have h₁ := h.odd_ne j hj
      have h₂ := A.Pph_z_ne_zero k j
      rw [bodd, iodd]
      field_simp
    have hboth : ∀ j, ¬(j = c ∨ j = mirror k c) →
        A.bev k ε (A.z k j) * A.ievn k ε (A.z k j) = 1 := fun j hj ↦ by
      have h₁ := h.ne_of_not_mem hj
      have h₂ := A.Pph_z_ne_zero k j
      rw [bev, ievn]
      field_simp
    -- the diagonal-plus-mirror product off the orbit is `δₙₘ`
    have hmb_off : ¬(n = c ∨ n = mirror k c) → mirrorBlock k
        (fun j ↦ A.bev k ε (A.z k j) * A.ievn k ε (A.z k j))
        (fun j ↦ A.bodd k ε (A.z k j) * A.iodd k ε (A.z k j)) n m = kron n m := fun hn ↦ by
      by_cases hfix : mirror k n = n
      · rw [mirrorBlock_of_fixed hfix, hboth n hn, one_mul]
      · rw [mirrorBlock, hboth n hn, hodd n (Ne.symm hfix)]
        ring
    by_cases hn : n = c ∨ n = mirror k c <;> by_cases hm : m = c ∨ m = mirror k c
    · -- both on the orbit
      have hk := kron_add_kron_orb hn hm
      have hzn : A.z k n ^ 2 = A.shat k ε := by linear_combination h.zero_iff.mpr hn
      have hA6' : A.cN₀ k ε (A.z k n) * A.ℓ₀ k + A.N₂ k ε (A.z k n) * A.ℓ₂ k =
          A.β * A.N₂ k ε (A.z k n) / (2 * ε * A.a) := by
        rw [hzn, sub_self, mul_zero, zero_add] at hA6
        exact hA6
      have hS := hflz_orb (Tq := A.That' k ε) (r := orbSize k c) hβ ha hε0 hr hPc hA6' hHF
      have hPzn : A.Pph k (A.z k n) = A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 := by
        rw [hPz, show A.z k n ^ 4 = (A.z k n ^ 2) ^ 2 by ring, hzn]
      rw [hιm, hgm, orb_of_mem hm, h.ghat_orb hm, h.bev_orb hn, h.ievn_orb hm, hPzn, hS]
      by_cases hfix : mirror k n = n
      · rw [mirrorBlock_of_fixed hfix, h.bev_orb hn, zero_mul, zero_mul]
        rw [hfix, kron_comm m n] at hk
        linear_combination (1 / 2 : ℂ) * hk
      · rw [mirrorBlock, h.bev_orb hn, h.ievn_orb hn, hodd n (Ne.symm hfix)]
        linear_combination (1 / 2 : ℂ) * hk
    · -- `n` on the orbit, `m` off it
      have hzn : A.z k n ^ 2 = A.shat k ε := by linear_combination h.zero_iff.mpr hn
      have hA6' : A.cN₀ k ε (A.z k n) * A.ℓ₀ k + A.N₂ k ε (A.z k n) * A.ℓ₂ k =
          A.β * A.N₂ k ε (A.z k n) / (2 * ε * A.a) := by
        rw [hzn, sub_self, mul_zero, zero_add] at hA6
        exact hA6
      have hnm : n ≠ m := fun h0 ↦ hm (by rw [← h0]; exact hn)
      have hnm' : m ≠ mirror k n := by
        intro h0
        apply hm
        rcases hn with rfl | rfl
        · exact Or.inr h0
        · left
          rw [h0, mirror_mirror]
      have hgv := h.ne_of_not_mem hm
      have hPm := A.Pph_z_ne_zero k m
      rw [A.Pph_eq_poly] at hPm
      have hS := hflz_row (Tq := A.That' k ε) (r := orbSize k c) (l0 := A.ℓ₀ k) (l2 := A.ℓ₂ k)
        hβ ha hε0 hr hPc hA6' hgv hPm
      have hPzn : A.Pph k (A.z k n) = A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 := by
        rw [hPz, show A.z k n ^ 4 = (A.z k n ^ 2) ^ 2 by ring, hzn]
      rw [hιm, hgm, orb_of_not_mem hm, h.bev_orb hn, kron_of_ne hnm, hR]
      simp only [mirrorBlock, kron_of_ne hnm, kron_of_ne hnm', mul_zero, add_zero, zero_mul,
        zero_add]
      rw [A.Nph_of_mu k ε hμ, hPzn, A.Pph_eq_poly k (A.z k m), ievn, A.Pph_eq_poly k (A.z k m),
        ghat]
      linear_combination -hS
    · -- `n` off the orbit, `m` on it
      have hgn := h.ne_of_not_mem hn
      have hnm : n ≠ m := fun h0 ↦ hn (by rw [h0]; exact hm)
      have hS := hflz_col (Tq := A.That' k ε) (r := orbSize k c) hβ ha hε0 hr hPc hA6 hHF hPz' hgn
      rw [hιm, hgm, orb_of_mem hm, h.ghat_orb hm, h.ievn_orb hm, hmb_off hn, kron_of_ne hnm]
      simp only [Zhf, orb_of_not_mem hn, orb_of_mem hm, h.ghat_orb hm, mul_zero, zero_mul,
        add_zero, zero_add, mul_one]
      rw [hPz, bev, hPz, ghat, hPcD]
      linear_combination -hS
    · -- both off the orbit
      have hgn := h.ne_of_not_mem hn
      have hgv := h.ne_of_not_mem hm
      have hPm := A.Pph_z_ne_zero k m
      rw [A.Pph_eq_poly] at hPm
      have hS := hflz_off (Tq := A.That' k ε) (r := orbSize k c) hβ ha hε0 hr hPc hA6 hPz' hgv
        hPm
      rw [hιm, hgm, orb_of_not_mem hm, hmb_off hn]
      simp only [Zhf, orb_of_not_mem hn, orb_of_not_mem hm, mul_zero, zero_mul, add_zero]
      simp only [hR]
      rw [A.Nph_of_mu k ε hμ, hPz, bev, hPz, ievn, A.Pph_eq_poly k (A.z k m), ghat]
      linear_combination -hS

/-- **Corollary "half filling"** at a lattice zero, two-sidedness: `invHFlz χ_r = 1`. -/
theorem HFLZ.chiR_isLeftInverse (h : A.HFLZ k ε c) :
    IsLeftInverse (A.chiR k ε) (A.invHFlz k ε c) := by
  intro n m
  have := h.chiR_isRightInverse m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [A.chiR_symm_hf k ε h.μ_eq l m, h.invHFlz_symm n l, mul_comm]

/-- **Corollary "half filling"** at a lattice zero, the closed form:
`β²(χ_r⁻¹ - B⁻¹) = gammaHFlz`. -/
theorem HFLZ.gammaHF_eq (_h : A.HFLZ k ε c) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invHFlz k ε c n m - kron n m / A.Bph k n) = A.gammaHFlz k ε c n m := by
  have hβ := A.β_ne_zero
  have hPn := A.Pph_z_ne_zero k n
  have hGn := A.G₂_z_ne_zero k n
  rw [invHFlz, gammaHFlz, A.Bph_eq k n]
  field_simp
  ring

end Params

end HubbardAtom
