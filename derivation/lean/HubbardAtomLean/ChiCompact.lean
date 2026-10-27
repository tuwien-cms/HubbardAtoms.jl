import HubbardAtomLean.ChiBounds
import HubbardAtomLean.DensityMagnetic
import HubbardAtomLean.HilbertSchmidt

/-!
# `χ_r` as a compact perturbation of `𝒟 = diag(-β/zₙ²)`

For the density and magnetic channels and `μ ≠ 0`, with `𝒟^(1/2) = diag(dₙ)`, `dₙ = i√β/zₙ`
(`Dhalf_sq`: `dₙ² = -β/zₙ²`):

* `Params.chiR_eq_Dhalf`: `χ_r = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)` with `𝒦_nm = -(zₙzₘ/β)χ_r,nm - δ_nm`;
* `Params.isCompactOperator_Kr`: `𝒦` is Hilbert–Schmidt (`summable_Kr_sq`, from the bounds of
  `ChiBounds.lean`), hence compact on `ℓ²(ℤ)`;
* `Params.Bph_eq_Dhalf`: the bubble is `B = 𝒟^(1/2)(1 + 𝒦₀)𝒟^(1/2)` with the diagonal
  `1 + 𝒦₀ = diag(tₙ)`, `tₙ = zₙ²G₂(zₙ)/P(zₙ)`;
* `Params.isUnit_one_add_K0`: `1 + 𝒦₀` is invertible on `ℓ²(ℤ)` (the inverse is
  `1 + diag(1/tₙ - 1)`); both diagonals are square-summable, as `tₙ = 1 + O(|zₙ|⁻¹)`.
-/

open Complex Filter
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `dₙ = i√β/zₙ`, the diagonal of `𝒟^(1/2)`. -/
noncomputable def Dhalf (n : ℤ) : ℂ := I * (√A.β : ℝ) / A.z k n

/-- `dₙ² = -β/zₙ²`. -/
theorem Dhalf_sq (n : ℤ) : A.Dhalf k n ^ 2 = -A.β / A.z k n ^ 2 := by
  rw [Dhalf, div_pow, mul_pow, I_sq, ← ofReal_pow, Real.sq_sqrt A.β_pos.le]
  ring

/-- `dₙ ≠ 0` for `μ ≠ 0`. -/
theorem Dhalf_ne_zero (hμ : A.μ ≠ 0) (n : ℤ) : A.Dhalf k n ≠ 0 := by
  rw [Dhalf]
  exact div_ne_zero (mul_ne_zero I_ne_zero (ofReal_ne_zero.mpr
    (Real.sqrt_pos.mpr A.β_pos).ne')) (A.z_ne_zero k hμ n)

/-- `𝒦_nm = -(zₙzₘ/β)χ_r,nm - δ_nm`. -/
noncomputable def Kr (n m : ℤ) : ℂ := -(A.z k n * A.z k m / A.β) * A.chiR k ε n m - kron n m

/-- `χ_r = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)`, entrywise. -/
theorem chiR_eq_Dhalf (hμ : A.μ ≠ 0) (n m : ℤ) :
    A.chiR k ε n m = A.Dhalf k n * (kron n m + A.Kr k ε n m) * A.Dhalf k m := by
  have hz := A.z_ne_zero k hμ n
  have hv := A.z_ne_zero k hμ m
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hs : ((√A.β : ℝ) : ℂ) ^ 2 = A.β := by rw [← ofReal_pow, Real.sq_sqrt A.β_pos.le]
  have hd : A.Dhalf k n * A.Dhalf k m = -A.β / (A.z k n * A.z k m) := by
    rw [Dhalf, Dhalf, div_mul_div_comm, mul_mul_mul_comm, I_mul_I, ← sq, hs]
    ring
  rw [Kr, add_sub_cancel, show ∀ x y c : ℂ, x * c * y = x * y * c from fun _ _ _ ↦ by ring, hd]
  field_simp

/-- `∑ₙ |zₙ|⁻²` converges. -/
theorem summable_inv_norm_z_sq (hμ : A.μ ≠ 0) : Summable fun n ↦ ‖A.z k n‖⁻¹ ^ 2 := by
  refine (A.hasSum_sq (x := 0) fun n h ↦ A.z_ne_zero k hμ n h).summable.norm.congr fun n ↦ ?_
  simp

/-- A sequence with `|zₙfₙ|` bounded is square-summable. -/
theorem summable_sq_of_norm_mul_z_le (hμ : A.μ ≠ 0) {f : ℤ → ℂ} {C : ℝ}
    (h : ∀ n, ‖A.z k n * f n‖ ≤ C) : Summable fun n ↦ ‖f n‖ ^ 2 := by
  refine ((A.summable_inv_norm_z_sq k hμ).mul_left (C ^ 2)).of_nonneg_of_le
    (fun _ ↦ by positivity) fun n ↦ ?_
  have hz := norm_pos_iff.mpr (A.z_ne_zero k hμ n)
  rw [← mul_pow]
  refine pow_le_pow_left₀ (norm_nonneg _) ?_ 2
  rw [le_mul_inv_iff₀ hz, mul_comm, ← norm_mul]
  exact h n

/-- `|𝒦_nm|² ≤ C|zₙ|⁻²(|zₘ|⁻² + δ_nm)`: `|𝒦_nm| ≤ c|zₙ|⁻¹|zₘ|⁻¹` off the diagonal and
`|𝒦_nn| ≤ c|zₙ|⁻¹`, by `norm_chiR_le` and `norm_chiR_diag_le`. -/
theorem norm_Kr_sq_le (hμ : A.μ ≠ 0) : ∃ C, ∀ n m, ‖A.Kr k ε n m‖ ^ 2 ≤
    C * (‖A.z k n‖⁻¹ ^ 2 * (‖A.z k m‖⁻¹ ^ 2 + if n = m then 1 else 0)) := by
  obtain ⟨c₁, hc₁⟩ := A.norm_chiR_le k ε hμ
  obtain ⟨c₂, hc₂⟩ := A.norm_chiR_diag_le k ε hμ
  have hβ := A.β_pos
  refine ⟨(c₁ / A.β) ^ 2 + (c₂ / A.β) ^ 2, fun n m ↦ ?_⟩
  have hz := norm_pos_iff.mpr (A.z_ne_zero k hμ n)
  have hv := norm_pos_iff.mpr (A.z_ne_zero k hμ m)
  by_cases hnm : n = m
  · subst hnm
    -- `|zₙ𝒦_nn| = |zₙ|³|χ_r,nn + β/zₙ²|/β ≤ c₂/β`
    have h : A.z k n * A.Kr k ε n n =
        -(A.z k n ^ 3 / A.β) * (A.chiR k ε n n + A.β / A.z k n ^ 2) := by
      have := A.z_ne_zero k hμ n
      have hβ' : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr hβ.ne'
      rw [Kr, kron_self]
      field_simp
      ring
    have hb : ‖A.z k n * A.Kr k ε n n‖ ≤ c₂ / A.β := by
      rw [h, norm_mul, norm_neg, norm_div, norm_pow, norm_real, Real.norm_of_nonneg hβ.le,
        div_mul_eq_mul_div, div_le_div_iff_of_pos_right hβ, mul_comm,
        ← le_div_iff₀ (by positivity)]
      exact hc₂ n
    have hK : ‖A.Kr k ε n n‖ ≤ c₂ / A.β * ‖A.z k n‖⁻¹ := by
      rw [le_mul_inv_iff₀ hz, mul_comm, ← norm_mul]
      exact hb
    calc ‖A.Kr k ε n n‖ ^ 2 ≤ (c₂ / A.β * ‖A.z k n‖⁻¹) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hK 2
      _ ≤ _ := by
        simp only [ite_true]
        nlinarith [sq_nonneg (c₁ / A.β), sq_nonneg (c₂ / A.β), sq_nonneg ‖A.z k n‖⁻¹,
          mul_nonneg (sq_nonneg (c₁ / A.β + c₂ / A.β)) (pow_nonneg (sq_nonneg ‖A.z k n‖⁻¹) 2)]
  · have hK : ‖A.Kr k ε n m‖ ≤ c₁ / A.β * ‖A.z k n‖⁻¹ * ‖A.z k m‖⁻¹ := by
      rw [Kr, kron_of_ne hnm, sub_zero, norm_mul, norm_neg, norm_div, norm_mul, norm_real,
        Real.norm_of_nonneg hβ.le]
      calc ‖A.z k n‖ * ‖A.z k m‖ / A.β * ‖A.chiR k ε n m‖
          ≤ ‖A.z k n‖ * ‖A.z k m‖ / A.β * (c₁ / (‖A.z k n‖ ^ 2 * ‖A.z k m‖ ^ 2)) :=
            mul_le_mul_of_nonneg_left (hc₁ n m hnm) (by positivity)
        _ = c₁ / A.β * ‖A.z k n‖⁻¹ * ‖A.z k m‖⁻¹ := by field_simp
    calc ‖A.Kr k ε n m‖ ^ 2 ≤ (c₁ / A.β * ‖A.z k n‖⁻¹ * ‖A.z k m‖⁻¹) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hK 2
      _ ≤ _ := by
        simp only [hnm, ite_false, add_zero]
        nlinarith [mul_nonneg (sq_nonneg (c₂ / A.β))
          (mul_nonneg (sq_nonneg ‖A.z k n‖⁻¹) (sq_nonneg ‖A.z k m‖⁻¹))]

/-- **`𝒦` is Hilbert–Schmidt**, by `norm_Kr_sq_le`. -/
theorem summable_Kr_sq (hμ : A.μ ≠ 0) :
    Summable fun p : ℤ × ℤ ↦ ‖A.Kr k ε p.1 p.2‖ ^ 2 := by
  obtain ⟨C, hC⟩ := A.norm_Kr_sq_le k ε hμ
  have ha := A.summable_inv_norm_z_sq k hμ
  -- the diagonal part, through `(n, n)`
  have hd : Summable fun p : ℤ × ℤ ↦ ‖A.z k p.1‖⁻¹ ^ 2 * if p.1 = p.2 then (1 : ℝ) else 0 := by
    refine (summable_diag_sq (c := fun n ↦ (‖A.z k n‖⁻¹ : ℂ)) (by simpa using ha)).congr
      fun p ↦ ?_
    by_cases h : p.1 = p.2 <;> simp [diagK, h]
  refine (((ha.mul_of_nonneg ha (fun _ ↦ by positivity) fun _ ↦ by positivity).add hd).mul_left
    C).of_nonneg_of_le (fun _ ↦ by positivity) fun p ↦ (hC p.1 p.2).trans_eq (by ring)

/-- **`𝒦` is compact** on `ℓ²(ℤ)`. -/
theorem isCompactOperator_Kr (hμ : A.μ ≠ 0) :
    IsCompactOperator (hsOp (A.Kr k ε) (A.summable_Kr_sq k ε hμ)) :=
  isCompactOperator_hsOp _ _

/-- `tₙ = zₙ²G₂(zₙ)/P(zₙ)`, the diagonal of `1 + 𝒦₀`. -/
noncomputable def tB (n : ℤ) : ℂ := A.z k n ^ 2 * A.G₂ k (A.z k n) / A.Pph k (A.z k n)

/-- `tₙ ≠ 0` on the lattice. -/
theorem tB_ne_zero (hμ : A.μ ≠ 0) (n : ℤ) : A.tB k n ≠ 0 :=
  div_ne_zero (mul_ne_zero (pow_ne_zero 2 (A.z_ne_zero k hμ n)) (A.G₂_z_ne_zero k n))
    (A.Pph_z_ne_zero k n)

/-- `B = 𝒟^(1/2)(1 + 𝒦₀)𝒟^(1/2)`: `Bₙ = dₙ²tₙ`. -/
theorem Bph_eq_Dhalf (hμ : A.μ ≠ 0) (n : ℤ) : A.Bph k n = A.Dhalf k n ^ 2 * A.tB k n := by
  have hz := A.z_ne_zero k hμ n
  rw [A.Dhalf_sq, tB, A.Bph_eq]
  field_simp

/-- `tₙ - 1` is square-summable: `zₙ(tₙ - 1)` is a function of `1/zₙ` continuous at `0`. -/
theorem summable_tB_sub_one_sq (hμ : A.μ ≠ 0) : Summable fun n ↦ ‖A.tB k n - 1‖ ^ 2 := by
  obtain ⟨C, hC⟩ := A.bdd_of_inv_z k (f := fun n ↦ A.z k n * (A.tB k n - 1))
    (fun t ↦ (1 + (2 * A.q₀ k - 4 * A.a ^ 2) * t ^ 2 + A.q₀ k ^ 2 * t ^ 4)⁻¹ *
      (2 * A.a * A.ρ + (A.a ^ 2 * A.ρ ^ 2 - A.w k ^ 2 / 4 - (2 * A.q₀ k - 4 * A.a ^ 2)) * t -
        A.q₀ k ^ 2 * t ^ 3))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) fun n ↦ by
    have hz := A.z_ne_zero k hμ n
    have hP := A.Pph_z_ne_zero k n
    have hD : 1 + (2 * A.q₀ k - 4 * A.a ^ 2) * (A.z k n)⁻¹ ^ 2 + A.q₀ k ^ 2 * (A.z k n)⁻¹ ^ 4 =
        A.Pph k (A.z k n) / A.z k n ^ 4 := by
      rw [A.Pph_eq]
      field_simp
    rw [hD, tB]
    field_simp
    simp only [G₂, A.Pph_eq]
    ring
  exact A.summable_sq_of_norm_mul_z_le k hμ hC

/-- `1/tₙ - 1` is square-summable. -/
theorem summable_inv_tB_sub_one_sq (hμ : A.μ ≠ 0) :
    Summable fun n ↦ ‖(A.tB k n)⁻¹ - 1‖ ^ 2 := by
  obtain ⟨C, hC⟩ := A.bdd_of_inv_z k (f := fun n ↦ A.z k n * ((A.tB k n)⁻¹ - 1))
    (fun t ↦ (1 + 2 * A.a * A.ρ * t + (A.a ^ 2 * A.ρ ^ 2 - A.w k ^ 2 / 4) * t ^ 2)⁻¹ *
      (-(2 * A.a * A.ρ) - (A.a ^ 2 * A.ρ ^ 2 - A.w k ^ 2 / 4 - (2 * A.q₀ k - 4 * A.a ^ 2)) * t +
        A.q₀ k ^ 2 * t ^ 3))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) fun n ↦ by
    have hz := A.z_ne_zero k hμ n
    have hP := A.Pph_z_ne_zero k n
    have hG := A.G₂_z_ne_zero k n
    have hE : 1 + 2 * A.a * A.ρ * (A.z k n)⁻¹ + (A.a ^ 2 * A.ρ ^ 2 - A.w k ^ 2 / 4) *
        (A.z k n)⁻¹ ^ 2 = A.G₂ k (A.z k n) / A.z k n ^ 2 := by
      rw [G₂]
      field_simp
      ring
    rw [hE, tB]
    field_simp
    simp only [G₂, A.Pph_eq]
    ring
  exact A.summable_sq_of_norm_mul_z_le k hμ hC

/-- **`1 + 𝒦₀` is invertible** on `ℓ²(ℤ)`; the proof builds the inverse `1 + diag(1/tₙ - 1)`. -/
theorem isUnit_one_add_K0 (hμ : A.μ ≠ 0) :
    IsUnit (1 + hsOp (diagK fun n ↦ A.tB k n - 1)
      (summable_diag_sq (A.summable_tB_sub_one_sq k hμ))) := by
  refine ⟨⟨_, 1 + hsOp (diagK fun n ↦ (A.tB k n)⁻¹ - 1)
    (summable_diag_sq (A.summable_inv_tB_sub_one_sq k hμ)), ?_, ?_⟩, rfl⟩ <;>
  · ext v n
    have ht := A.tB_ne_zero k hμ n
    simp [hsOp_diag_apply (A.summable_tB_sub_one_sq k hμ),
      hsOp_diag_apply (A.summable_inv_tB_sub_one_sq k hμ)]
    field_simp
    ring

end Params

end HubbardAtom
