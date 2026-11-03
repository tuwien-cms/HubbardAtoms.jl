import HubbardAtomLean.GammaClosedForm
import HubbardAtomLean.Confluent

/-!
# The operator vertex from a right inverse, and at a double root

For the density and magnetic channels and `μ ≠ 0`:

* `Params.isUnit_onePlusK_of_isRightInverse`: if `χ_r` has a right inverse whose columns are in the
  class `∑ₙ |vₙ/zₙ|² < ∞`, then `1 + 𝒦` is invertible, so `Γ_r` exists;
  `Params.isUnit_onePlusK_iff_exists_isRightInverse`: and conversely, by `chiInv`;
* `Params.DMConf.gammaOp_eq`: at a double root `s₁ = s₂` (Proposition "removable singularities",
  item 2), the operator vertex `gammaOp` is the closed form `gammaRc`, as the columns of `invRc`
  are bounded (`DMConf.tendsto_invRc`).
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable {A : Params} {k : ℤ} {ε u : ℂ}

/-- **`1 + 𝒦` is invertible where `χ_r` has a right inverse in the class**: for a right inverse
`Y` with `∑ₙ |Y_nm/zₙ|² < ∞` for every `m`, `cₘ = 𝒟^(1/2)(Y eₘ)dₘ` has `(1 + 𝒦)cₘ = eₘ`, and for
`(1 + 𝒦)x = 0`, `xₘ = ∑ₙ xₙ((1 + 𝒦)cₘ)ₙ = ∑ₙ cₘ,ₙ((1 + 𝒦)x)ₙ = 0`, as `𝒦` is symmetric. -/
theorem isUnit_onePlusK_of_isRightInverse (hμ : A.μ ≠ 0) {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiR k ε) Y)
    (hcls : ∀ m, Summable fun n ↦ ‖Y n m / A.z k n‖ ^ 2) : IsUnit (A.onePlusK k ε hμ) := by
  have hcol : ∀ m, ∃ c : ℓ²(ℤ, ℂ), A.onePlusK k ε hμ c = lp.single 2 m 1 := fun m ↦ by
    have hv : Summable fun n ↦ ‖Y n m * A.Dhalf k m / A.z k n‖ ^ 2 :=
      ((hcls m).mul_left (‖A.Dhalf k m‖ ^ 2)).congr fun n ↦ by
        rw [mul_div_right_comm, norm_mul]
        ring
    refine ⟨A.DhalfVec k hv, ?_⟩
    ext n
    have h₁ := A.hasSum_chiR_mul k ε hμ hv n
    have h₂ := (hY n m).mul_right (A.Dhalf k m)
    simp only [mul_assoc] at h₂
    have h₃ := h₁.unique h₂
    have hd := A.Dhalf_ne_zero k hμ n
    rw [single_apply_eq_kron]
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self, one_mul] at h₃
      rw [kron_self]
      exact mul_left_cancel₀ hd (h₃.trans (mul_one _).symm)
    · rw [kron_of_ne hnm, zero_mul] at h₃
      rw [kron_of_ne hnm]
      exact (mul_eq_zero.mp h₃).resolve_left hd
  refine (A.isUnit_onePlusK_iff_injective k ε hμ).mpr ?_
  rw [injective_iff_map_eq_zero]
  intro x hx
  ext m
  obtain ⟨c, hc⟩ := hcol m
  have hpair := tsum_mul_one_add_hsOp_comm (A.Kr k ε) (A.summable_Kr_sq k ε hμ) (Kr_symm hμ) x c
  rw [show (1 + hsOp (A.Kr k ε) (A.summable_Kr_sq k ε hμ)) = A.onePlusK k ε hμ from rfl,
    hc, hx] at hpair
  simp only [single_apply_eq_kron, lp.coeFn_zero, Pi.zero_apply, mul_zero, tsum_zero, kron,
    mul_ite, mul_one] at hpair
  rw [tsum_ite_eq m] at hpair
  simpa using hpair

/-- **`Γ_r` exists iff `χ_r` has a right inverse in the class**: `1 + 𝒦` is invertible iff `χ_r`
has a right inverse whose columns satisfy `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem isUnit_onePlusK_iff_exists_isRightInverse (hμ : A.μ ≠ 0) :
    IsUnit (A.onePlusK k ε hμ) ↔ ∃ Y : ℤ → ℤ → ℂ, IsRightInverse (A.chiR k ε) Y ∧
      ∀ m, Summable fun n ↦ ‖Y n m / A.z k n‖ ^ 2 :=
  ⟨fun hK ↦ ⟨A.chiInv k ε hμ, chiInv_isRightInverse hμ hK, summable_chiInv hμ⟩,
    fun ⟨_, hY, hcls⟩ ↦ isUnit_onePlusK_of_isRightInverse hμ hY hcls⟩

/-- **The columns of `invRc` are bounded**: `invRc n m → 2εa/β²` as `|n| → ∞`, as
`(c₁t + c₂)/(t² - u²) = c₁[(t - u)⁻¹ + (t + u)⁻¹]/2 + c₂(t - u)⁻¹(t + u)⁻¹` at `t = zₙ`. -/
theorem DMConf.tendsto_invRc (h : A.DMConf k ε u) (m : ℤ) :
    Tendsto (fun n ↦ A.invRc k ε u n m) cofinite (𝓝 (2 * ε * A.a / A.β ^ 2)) := by
  have hk : Tendsto (fun n ↦ kron n m) cofinite (𝓝 0) :=
    tendsto_const_nhds.congr' ((eventually_cofinite_ne m).mono fun n hn ↦ (kron_of_ne hn).symm)
  have hi : ∀ x, Tendsto (fun n ↦ (A.z k n - x)⁻¹) cofinite (𝓝 0) := fun x ↦
    tendsto_inv₀_cobounded.comp (tendsto_norm_atTop_iff_cobounded.mp (A.tendsto_norm_z_sub k x))
  have hii : Tendsto (fun n ↦ (A.z k n - u)⁻¹ * (A.z k n - -u)⁻¹) cofinite (𝓝 0) := by
    simpa using (hi u).mul (hi (-u))
  have hq : ∀ c₁ c₂ : ℂ, Tendsto (fun n ↦ (c₁ * A.z k n + c₂) / (A.z k n ^ 2 - u ^ 2))
      cofinite (𝓝 0) := fun c₁ c₂ ↦ by
    have h0 := (((hi u).add (hi (-u))).const_mul (c₁ / 2)).add (hii.const_mul c₂)
    simp only [add_zero, mul_zero] at h0
    refine h0.congr fun n ↦ ?_
    have h1 : A.z k n - u ≠ 0 := sub_ne_zero.mpr (h.z_ne n)
    have h2 : A.z k n + u ≠ 0 := fun h0 ↦ h.z_ne_neg n (by linear_combination h0)
    have h2' : A.z k n - -u ≠ 0 := sub_ne_zero.mpr (h.z_ne_neg n)
    have h3 : A.z k n ^ 2 - u ^ 2 ≠ 0 := h.z_sq_ne n
    field_simp
    ring
  have hq2 : ∀ c₁ c₂ : ℂ, Tendsto (fun n ↦ (c₁ * A.z k n + c₂) / (A.z k n ^ 2 - u ^ 2) ^ 2)
      cofinite (𝓝 0) := fun c₁ c₂ ↦ by
    have h0 := (hq c₁ c₂).mul hii
    simp only [zero_mul] at h0
    refine h0.congr fun n ↦ ?_
    have h1 : A.z k n - u ≠ 0 := sub_ne_zero.mpr (h.z_ne n)
    have h2 : A.z k n + u ≠ 0 := fun h0 ↦ h.z_ne_neg n (by linear_combination h0)
    have h2' : A.z k n - -u ≠ 0 := sub_ne_zero.mpr (h.z_ne_neg n)
    have h3 : A.z k n ^ 2 - u ^ 2 ≠ 0 := h.z_sq_ne n
    field_simp
    ring
  have hM := (((hq (A.κ k + 2 * u ^ 2) (A.Zsd k ε u)).add
    (hq2 (A.Pph k u) (A.Zph k ε u))).mul_const (1 / (A.z k m ^ 2 - u ^ 2))).add
    ((hq (A.Pph k u) (A.Zph k ε u)).mul_const ((1 / (A.z k m ^ 2 - u ^ 2)) ^ 2))
  have h0 := (((hk.const_mul (A.Dv k ε (A.z k m))).add
    ((hi (-A.z k m)).const_mul (A.Ev k ε (A.z k m)))).add_const (2 * ε * A.a / A.β ^ 2)).add
    (hM.const_mul (A.lam ε * A.ρ / A.β ^ 2))
  simp only [mul_zero, add_zero, zero_add, zero_mul] at h0
  refine h0.congr fun n ↦ ?_
  simp only [invRc, M₁, modeΦ, sub_neg_eq_add, div_eq_mul_inv]

/-- The columns of `invRc` are in the class `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem DMConf.summable_invRc (h : A.DMConf k ε u) (m : ℤ) :
    Summable fun n ↦ ‖A.invRc k ε u n m / A.z k n‖ ^ 2 := by
  obtain ⟨C, hC⟩ := (h.tendsto_invRc m).norm.bddAbove_range_of_cofinite
  exact A.summable_sq_of_norm_mul_z_le k h.μ_ne (C := C) fun n ↦ by
    rw [mul_div_cancel₀ _ (A.z_ne_zero k h.μ_ne n)]
    exact hC ⟨n, rfl⟩

/-- At a double root `1 + 𝒦` is invertible: `invRc` is a right inverse in the class. -/
theorem DMConf.isUnit_onePlusK (h : A.DMConf k ε u) : IsUnit (A.onePlusK k ε h.μ_ne) :=
  isUnit_onePlusK_of_isRightInverse h.μ_ne h.chiR_isRightInverse h.summable_invRc

/-- **`χ_r⁻¹` is `invRc`** at a double root. -/
theorem DMConf.invRc_eq_chiInv (h : A.DMConf k ε u) (n m : ℤ) :
    A.invRc k ε u n m = A.chiInv k ε h.μ_ne n m :=
  eq_chiInv_of_hasSum h.μ_ne h.isUnit_onePlusK (h.summable_invRc m)
    (fun n ↦ h.chiR_isRightInverse n m) n

/-- **The operator vertex at a double root** (Proposition "removable singularities", item 2):
`Γ_r = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)` is the closed form `gammaRc`. -/
theorem DMConf.gammaOp_eq (h : A.DMConf k ε u) (n m : ℤ) :
    A.gammaOp k ε h.μ_ne n m = A.gammaRc k ε u n m := by
  rw [gammaOp_eq_chiInv, ← h.invRc_eq_chiInv, h.gammaR_eq]

end Params

end HubbardAtom
