import HubbardAtomLean.ChiCompact
import Mathlib.Analysis.Normed.Operator.Compact.FredholmAlternative

/-!
# The vertex as an operator: definition and criterion

For the density and magnetic channels and `μ ≠ 0`, with `𝒟^(1/2) = diag(dₙ)` and `𝒦`, `𝒦₀` of
`ChiCompact.lean`:

* `Params.gammaOp`: `Γ_r = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)`, entrywise, with
  `Ring.inverse` (zero where `1 + 𝒦` is not invertible);
* `Params.isUnit_onePlusK_iff_injective`: `1 + 𝒦` is invertible iff `ker(1 + 𝒦) = 0`, by the
  Fredholm alternative for the compact `𝒦`;
* `Params.injective_onePlusK_iff`: `ker(1 + 𝒦) = 0` iff `χ_r` has no nonzero null vector `v` with
  `∑ₙ |vₙ/zₙ|² < ∞` (`hasSum_chiR_mul`: `χ_r v = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)v`); together,
  `isUnit_onePlusK_iff`;
* `Params.exists_bound_of_null`: such null vectors are bounded, as `v = -𝒟^(-1/2)𝒦𝒟^(1/2)v` and
  the rows of `𝒦` have norm `O(|zₙ|⁻¹)` (`norm_Kr_sq_le`).
-/

open Complex
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `1 + 𝒦` on `ℓ²(ℤ)`. -/
noncomputable def onePlusK (hμ : A.μ ≠ 0) : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (A.Kr k ε) (A.summable_Kr_sq k ε hμ)

/-- `1 + 𝒦₀ = diag(tₙ)` on `ℓ²(ℤ)`. -/
noncomputable def onePlusK0 (hμ : A.μ ≠ 0) : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (diagK fun n ↦ A.tB k n - 1) (summable_diag_sq (A.summable_tB_sub_one_sq k hμ))

/-- `Γ_r,nm = β²/(dₙdₘ) [(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]ₙₘ`, the entries taken on the unit vectors. -/
noncomputable def gammaOp (hμ : A.μ ≠ 0) (n m : ℤ) : ℂ :=
  A.β ^ 2 / (A.Dhalf k n * A.Dhalf k m) *
    ((Ring.inverse (A.onePlusK k ε hμ) - Ring.inverse (A.onePlusK0 k hμ))
      (lp.single 2 m 1) n)

/-- **Fredholm alternative**: `1 + 𝒦` is invertible iff it is injective. -/
theorem isUnit_onePlusK_iff_injective (hμ : A.μ ≠ 0) :
    IsUnit (A.onePlusK k ε hμ) ↔ Function.Injective (A.onePlusK k ε hμ) := by
  refine ⟨fun h ↦ (ContinuousLinearMap.isUnit_iff_bijective.mp h).1, fun hinj ↦ ?_⟩
  rcases (A.isCompactOperator_Kr k ε hμ).hasEigenvalue_or_mem_resolventSet (μ := -1)
    (by norm_num) with hev | hres
  · obtain ⟨x, hx⟩ := hev.exists_hasEigenvector
    refine absurd (hinj ?_) hx.2
    have h := hx.apply_eq_smul
    simp only [ContinuousLinearMap.coe_coe] at h
    simp [onePlusK, h]
  · rw [spectrum.mem_resolventSet_iff, Algebra.algebraMap_eq_smul_one,
      show (-1 : ℂ) • (1 : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ)) - hsOp (A.Kr k ε) (A.summable_Kr_sq k ε hμ) =
        -A.onePlusK k ε hμ by rw [onePlusK]; module, IsUnit.neg_iff] at hres
    exact hres

/-- For `∑ₙ |vₙ/zₙ|² < ∞`, `𝒟^(1/2)v ∈ ℓ²`. -/
noncomputable def DhalfVec {v : ℤ → ℂ} (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2) :
    ℓ²(ℤ, ℂ) :=
  ⟨fun n ↦ A.Dhalf k n * v n, memℓp_two_iff.mpr ((hv.mul_left A.β).congr fun n ↦ by
    rw [Dhalf, show I * (√A.β : ℝ) / A.z k n * v n = I * (√A.β : ℝ) * (v n / A.z k n) by ring,
      norm_mul, norm_mul, norm_I, norm_real, Real.norm_of_nonneg (Real.sqrt_nonneg _), one_mul,
      mul_pow, Real.sq_sqrt A.β_pos.le])⟩

/-- For `w ∈ ℓ²`, `vₙ = wₙ/dₙ` has `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem summable_div_Dhalf (hμ : A.μ ≠ 0) (w : ℓ²(ℤ, ℂ)) :
    Summable fun n ↦ ‖w n / A.Dhalf k n / A.z k n‖ ^ 2 := by
  refine ((memℓp_two_iff.mp (lp.memℓp w)).mul_left A.β⁻¹).congr fun n ↦ ?_
  have hz := A.z_ne_zero k hμ n
  rw [Dhalf, norm_div, norm_div, norm_div, norm_mul, norm_I, norm_real,
    Real.norm_of_nonneg (Real.sqrt_nonneg _)]
  field_simp
  rw [Real.sq_sqrt A.β_pos.le]

/-- `𝒟^(1/2)(w/d) = w`. -/
theorem DhalfVec_div (hμ : A.μ ≠ 0) (w : ℓ²(ℤ, ℂ)) :
    A.DhalfVec k (A.summable_div_Dhalf k hμ w) = w := by
  ext n
  simp [DhalfVec, mul_div_cancel₀ _ (A.Dhalf_ne_zero k hμ n)]

/-- `χ_r v = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)v`, row by row, for `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem hasSum_chiR_mul (hμ : A.μ ≠ 0) {v : ℤ → ℂ} (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2)
    (n : ℤ) : HasSum (fun m ↦ A.chiR k ε n m * v m)
      (A.Dhalf k n * A.onePlusK k ε hμ (A.DhalfVec k hv) n) := by
  have h := ((hasSum_ite_eq n (A.DhalfVec k hv n)).add
    (hasSum_hsOp (A.Kr k ε) (A.summable_Kr_sq k ε hμ) (A.DhalfVec k hv) n)).mul_left
    (A.Dhalf k n)
  convert h using 1
  · funext m
    rw [A.chiR_eq_Dhalf k ε hμ]
    by_cases hnm : n = m
    · subst hnm
      simp [kron_self, DhalfVec]
      ring
    · simp [kron_of_ne hnm, Ne.symm hnm, DhalfVec]
      ring
  · simp [onePlusK]

/-- A null vector of `χ_r` with `∑ₙ |vₙ/zₙ|² < ∞` gives `(1 + 𝒦)𝒟^(1/2)v = 0`. -/
theorem onePlusK_DhalfVec_eq_zero (hμ : A.μ ≠ 0) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) 0) :
    A.onePlusK k ε hμ (A.DhalfVec k hv) = 0 := by
  ext n
  have h := (A.hasSum_chiR_mul k ε hμ hv n).unique (hχ n)
  simpa [A.Dhalf_ne_zero k hμ n] using h

/-- **The criterion**: `ker(1 + 𝒦) = 0` iff `χ_r` has no nonzero null vector `v` with
`∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem injective_onePlusK_iff (hμ : A.μ ≠ 0) :
    Function.Injective (A.onePlusK k ε hμ) ↔ ∀ v : ℤ → ℂ,
      Summable (fun n ↦ ‖v n / A.z k n‖ ^ 2) →
        (∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) 0) → v = 0 := by
  have hd := A.Dhalf_ne_zero k hμ
  constructor
  · intro hinj v hv hχ
    have h := hinj ((A.onePlusK_DhalfVec_eq_zero k ε hμ hv hχ).trans (map_zero _).symm)
    funext n
    have hn := congrArg (fun w : ℓ²(ℤ, ℂ) ↦ w n) h
    simpa [DhalfVec, hd n] using hn
  · intro h
    rw [injective_iff_map_eq_zero]
    intro w hw
    have h0 := h _ (A.summable_div_Dhalf k hμ w) fun n ↦ by
      simpa [A.DhalfVec_div k hμ, hw] using
        A.hasSum_chiR_mul k ε hμ (A.summable_div_Dhalf k hμ w) n
    ext n
    have hn := congrFun h0 n
    simp only [Pi.zero_apply, div_eq_zero_iff, hd n, or_false] at hn
    simp [hn]

/-- **`Γ_r` exists iff `χ_r` has no null vector**: `1 + 𝒦` is invertible iff `χ_r` has no nonzero
null vector `v` with `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem isUnit_onePlusK_iff (hμ : A.μ ≠ 0) :
    IsUnit (A.onePlusK k ε hμ) ↔ ∀ v : ℤ → ℂ, Summable (fun n ↦ ‖v n / A.z k n‖ ^ 2) →
      (∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) 0) → v = 0 :=
  (A.isUnit_onePlusK_iff_injective k ε hμ).trans (A.injective_onePlusK_iff k ε hμ)

/-- **Null vectors are bounded**: a null vector `v` of `χ_r` with `∑ₙ |vₙ/zₙ|² < ∞` has
`|vₙ| ≤ C`, as `w = 𝒟^(1/2)v = -𝒦w` gives `|wₙ| ≤ ‖𝒦ₙ·‖‖w‖ = O(|zₙ|⁻¹)`. -/
theorem exists_bound_of_null (hμ : A.μ ≠ 0) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) 0) : ∃ C, ∀ n, ‖v n‖ ≤ C := by
  obtain ⟨C, hC⟩ := A.norm_Kr_sq_le k ε hμ
  have ha := A.summable_inv_norm_z_sq k hμ
  have hβ := A.β_pos
  set w := A.DhalfVec k hv with hwdef
  set S := ∑' m, ‖A.z k m‖⁻¹ ^ 2
  refine ⟨√(C * (S + 1) * ‖w‖ ^ 2 / A.β), fun n ↦ ?_⟩
  have hz := norm_pos_iff.mpr (A.z_ne_zero k hμ n)
  -- `wₙ = -⟪𝒦ₙ·, w⟫`
  have hw : w n = -inner ℂ (hsRow (A.Kr k ε) (A.summable_Kr_sq k ε hμ) n) w := by
    have h := congrArg (fun x : ℓ²(ℤ, ℂ) ↦ x n) (A.onePlusK_DhalfVec_eq_zero k ε hμ hv hχ)
    rw [← hwdef] at h
    simp only [onePlusK, _root_.add_apply, one_apply_eq_self, lp.coeFn_add, Pi.add_apply,
      hsOp_apply_eq_inner, lp.coeFn_zero, Pi.zero_apply] at h
    linear_combination h
  -- `‖𝒦ₙ·‖² ≤ C|zₙ|⁻²(S + 1)`
  have hrow : ‖hsRow (A.Kr k ε) (A.summable_Kr_sq k ε hμ) n‖ ^ 2 ≤
      C * (‖A.z k n‖⁻¹ ^ 2 * (S + 1)) := by
    have hδ : HasSum (fun m ↦ if n = m then (1 : ℝ) else 0) 1 := by
      simpa [eq_comm] using hasSum_ite_eq n (1 : ℝ)
    have hs : Summable fun m ↦ C * (‖A.z k n‖⁻¹ ^ 2 * (‖A.z k m‖⁻¹ ^ 2 +
        if n = m then 1 else 0)) := ((ha.add hδ.summable).mul_left _).mul_left _
    refine (lp_norm_sq_le hs fun m ↦ ?_).trans_eq ?_
    · simpa [hsRow] using hC n m
    · rw [tsum_mul_left, tsum_mul_left, ha.tsum_add hδ.summable, hδ.tsum_eq]
  have hv' : ‖v n‖ ^ 2 = ‖A.z k n‖ ^ 2 * ‖w n‖ ^ 2 / A.β := by
    have hwn : w n = A.Dhalf k n * v n := rfl
    rw [hwn, Dhalf, norm_mul, norm_div, norm_mul, norm_I, norm_real,
      Real.norm_of_nonneg (Real.sqrt_nonneg _)]
    field_simp
    rw [Real.sq_sqrt hβ.le]
  rw [← abs_norm]
  refine Real.abs_le_sqrt ?_
  rw [hv', hw, norm_neg]
  have hCS := norm_inner_sq_le (hsRow (A.Kr k ε) (A.summable_Kr_sq k ε hμ) n) w
  calc ‖A.z k n‖ ^ 2 * ‖inner ℂ (hsRow (A.Kr k ε) (A.summable_Kr_sq k ε hμ) n) w‖ ^ 2 / A.β
      ≤ ‖A.z k n‖ ^ 2 * (C * (‖A.z k n‖⁻¹ ^ 2 * (S + 1)) * ‖w‖ ^ 2) / A.β := by
        gcongr
        exact hCS.trans (mul_le_mul_of_nonneg_right hrow (by positivity))
    _ = C * (S + 1) * ‖w‖ ^ 2 / A.β := by
        field_simp

end Params

end HubbardAtom
