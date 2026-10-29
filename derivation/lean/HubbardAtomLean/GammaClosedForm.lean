import HubbardAtomLean.GammaOperator

/-!
# The operator vertex is the closed form

On the domain of Theorem "density and magnetic" (generic case, `DMGeneric`), the vertex
`Γ_r = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)` of `GammaOperator.lean` exists and is the
closed form `gammaR` of Eq. (Gamma-ph):

* `Params.tendsto_invR`: each column of the inverse `invR` tends to `2εa/β²`, so it is bounded,
  and `cₘ = 𝒟^(1/2)(invR eₘ)dₘ ∈ ℓ²` (`invCol`);
* `DMGeneric.onePlusK_invCol`: `(1 + 𝒦)cₘ = eₘ`, from `χ_r invR = 1`;
* `DMGeneric.isUnit_onePlusK`: `1 + 𝒦` is invertible, as `xₘ = ∑ₙ xₙ((1 + 𝒦)cₘ)ₙ =
  ∑ₙ cₘ,ₙ((1 + 𝒦)x)ₙ` for the symmetric `𝒦` (`tsum_mul_one_add_hsOp_comm`);
* `DMGeneric.gammaOp_eq`: `gammaOp = gammaR`.

For every `μ ≠ 0`, with `χ_r⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)` (`chiInv`):

* `Params.gammaOp_eq_chiInv`: `Γ_r = β²(χ_r⁻¹ - B⁻¹)`;
* where `1 + 𝒦` is invertible, `χ_r⁻¹` is a right inverse of `χ_r` (`hasSum_chiR_mul_chiInv`),
  the only one with columns in the class `∑ₙ |vₙ/zₙ|² < ∞` (`eq_chiInv_of_hasSum`), symmetric
  (`chiInv_symm`), hence also a left inverse (`hasSum_chiInv_mul_chiR`).
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable {A : Params} {k : ℤ} {ε u₁ u₂ : ℂ}

/-- `𝒦` is symmetric, as `χ_r` is. -/
theorem Kr_symm (hμ : A.μ ≠ 0) (n m : ℤ) : A.Kr k ε n m = A.Kr k ε m n := by
  rw [Kr, Kr, A.chiR_symm k ε hμ, kron_comm, mul_comm (A.z k n)]

variable (A k ε) in
/-- **The columns of `invR` are bounded**: `invR n m → 2εa/β²` as `|n| → ∞`. -/
theorem tendsto_invR (u₁ u₂ : ℂ) (m : ℤ) :
    Tendsto (fun n ↦ A.invR k ε u₁ u₂ n m) cofinite (𝓝 (2 * ε * A.a / A.β ^ 2)) := by
  have hk : Tendsto (fun n ↦ kron n m) cofinite (𝓝 0) :=
    tendsto_const_nhds.congr' ((eventually_cofinite_ne m).mono fun n hn ↦ (kron_of_ne hn).symm)
  have hi : ∀ x, Tendsto (fun n ↦ (A.z k n - x)⁻¹) cofinite (𝓝 0) := fun x ↦
    tendsto_inv₀_cobounded.comp (tendsto_norm_atTop_iff_cobounded.mp (A.tendsto_norm_z_sub k x))
  have hφ : ∀ u c, Tendsto (fun n ↦ c * (A.β * A.Hph k ε (-u) * (A.z k n - u)⁻¹ +
      A.lam ε * A.α k u * (A.z k n - -u)⁻¹)) cofinite (𝓝 0) := fun u c ↦ by
    simpa using (((hi u).const_mul _).add ((hi (-u)).const_mul _)).const_mul c
  have h := ((((hk.const_mul (A.Dv k ε (A.z k m))).add
    ((hi (-A.z k m)).const_mul (A.Ev k ε (A.z k m)))).add_const (2 * ε * A.a / A.β ^ 2)).add
    (hφ u₁ (A.cmode k ε u₁ u₂ (A.z k m)))).add (hφ u₂ (A.cmode k ε u₂ u₁ (A.z k m)))
  simp only [mul_zero, add_zero, zero_add] at h
  refine h.congr fun n ↦ ?_
  simp only [invR, mode, div_eq_mul_inv, sub_neg_eq_add]

/-- The column `m` of `invR` times `dₘ`, `vₙ = invR n m · dₘ`, has `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem DMGeneric.summable_invCol (h : A.DMGeneric k ε u₁ u₂) (m : ℤ) :
    Summable fun n ↦ ‖A.invR k ε u₁ u₂ n m * A.Dhalf k m / A.z k n‖ ^ 2 := by
  obtain ⟨C, hC⟩ := (A.tendsto_invR k ε u₁ u₂ m).norm.bddAbove_range_of_cofinite
  refine A.summable_sq_of_norm_mul_z_le k h.μ_ne (C := C * ‖A.Dhalf k m‖) fun n ↦ ?_
  rw [mul_div_cancel₀ _ (A.z_ne_zero k h.μ_ne n), norm_mul]
  exact mul_le_mul_of_nonneg_right (hC ⟨n, rfl⟩) (norm_nonneg _)

/-- `cₘ = 𝒟^(1/2)(invR eₘ)dₘ ∈ ℓ²`, `cₘ,ₙ = dₙ invR n m dₘ`. -/
noncomputable def DMGeneric.invCol (h : A.DMGeneric k ε u₁ u₂) (m : ℤ) : ℓ²(ℤ, ℂ) :=
  A.DhalfVec k (h.summable_invCol m)

/-- `cₘ,ₙ = dₙ(invR n m · dₘ)`. -/
theorem DMGeneric.invCol_apply (h : A.DMGeneric k ε u₁ u₂) (m n : ℤ) :
    h.invCol m n = A.Dhalf k n * (A.invR k ε u₁ u₂ n m * A.Dhalf k m) :=
  rfl

/-- `eₘ`, with entries `δₙₘ`. -/
theorem single_apply_eq_kron (m n : ℤ) : (lp.single 2 m (1 : ℂ) : ℤ → ℂ) n = kron n m := by
  rw [lp.single_apply, Pi.single_apply, kron]

/-- **`(1 + 𝒦)cₘ = eₘ`**, from `χ_r invR = 1`. -/
theorem DMGeneric.onePlusK_invCol (h : A.DMGeneric k ε u₁ u₂) (m : ℤ) :
    A.onePlusK k ε h.μ_ne (h.invCol m) = lp.single 2 m 1 := by
  ext n
  have h₁ := A.hasSum_chiR_mul k ε h.μ_ne (h.summable_invCol m) n
  have h₂ := (h.chiR_isRightInverse n m).mul_right (A.Dhalf k m)
  simp only [mul_assoc] at h₂
  have h₃ := h₁.unique h₂
  have hd := A.Dhalf_ne_zero k h.μ_ne n
  rw [single_apply_eq_kron]
  change A.onePlusK k ε h.μ_ne (h.invCol m) n = kron n m
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self, one_mul] at h₃
    rw [kron_self]
    exact mul_left_cancel₀ hd (h₃.trans (mul_one _).symm)
  · rw [kron_of_ne hnm, zero_mul] at h₃
    rw [kron_of_ne hnm]
    exact (mul_eq_zero.mp h₃).resolve_left hd

/-- **`1 + 𝒦` is invertible** on the domain of Theorem "density and magnetic": for
`(1 + 𝒦)x = 0`, `xₘ = ∑ₙ xₙ((1 + 𝒦)cₘ)ₙ = ∑ₙ cₘ,ₙ((1 + 𝒦)x)ₙ = 0`. -/
theorem DMGeneric.isUnit_onePlusK (h : A.DMGeneric k ε u₁ u₂) :
    IsUnit (A.onePlusK k ε h.μ_ne) := by
  refine (A.isUnit_onePlusK_iff_injective k ε h.μ_ne).mpr ?_
  rw [injective_iff_map_eq_zero]
  intro x hx
  ext m
  have hpair := tsum_mul_one_add_hsOp_comm (A.Kr k ε) (A.summable_Kr_sq k ε h.μ_ne)
    (Kr_symm h.μ_ne) x (h.invCol m)
  rw [show (1 + hsOp (A.Kr k ε) (A.summable_Kr_sq k ε h.μ_ne)) = A.onePlusK k ε h.μ_ne from rfl,
    h.onePlusK_invCol m, hx] at hpair
  simp only [single_apply_eq_kron, lp.coeFn_zero, Pi.zero_apply, mul_zero, tsum_zero, kron,
    mul_ite, mul_one] at hpair
  rw [tsum_ite_eq m] at hpair
  simpa using hpair

/-- `(1 + 𝒦₀)(eₘ/tₘ) = eₘ`. -/
theorem onePlusK0_single (hμ : A.μ ≠ 0) (m : ℤ) :
    A.onePlusK0 k hμ ((A.tB k m)⁻¹ • lp.single 2 m 1) = lp.single 2 m 1 := by
  ext n
  simp only [onePlusK0, _root_.add_apply, one_apply_eq_self, lp.coeFn_add, Pi.add_apply,
    hsOp_diag_apply (A.summable_tB_sub_one_sq k hμ), lp.coeFn_smul, Pi.smul_apply, smul_eq_mul,
    single_apply_eq_kron]
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp [A.tB_ne_zero k hμ n]
    ring
  · simp [kron_of_ne hnm]

/-- The inverse of a unit applied to `T c`. -/
theorem inverse_apply_of_eq {T : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ)} (hT : IsUnit T)
    {c e : ℓ²(ℤ, ℂ)} (hc : T c = e) : Ring.inverse T e = c := by
  rw [← hc]
  change (Ring.inverse T * T) c = c
  rw [Ring.inverse_mul_cancel _ hT]
  rfl

/-- A unit applied to its inverse. -/
theorem apply_inverse_of_isUnit {T : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ)} (hT : IsUnit T) (e : ℓ²(ℤ, ℂ)) :
    T (Ring.inverse T e) = e := by
  change (T * Ring.inverse T) e = e
  rw [Ring.mul_inverse_cancel _ hT]
  rfl

/-- **The operator vertex is the closed form**: on the domain of Theorem "density and magnetic"
(generic case), `Γ_r = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)` is `gammaR` of
Eq. (Gamma-ph). -/
theorem DMGeneric.gammaOp_eq (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    A.gammaOp k ε h.μ_ne n m = A.gammaR k ε u₁ u₂ n m := by
  have hμ := h.μ_ne
  rw [← h.gammaR_eq, gammaOp, _root_.sub_apply,
    inverse_apply_of_eq h.isUnit_onePlusK (h.onePlusK_invCol m),
    inverse_apply_of_eq (show IsUnit (A.onePlusK0 k hμ) from A.isUnit_one_add_K0 k hμ)
      (onePlusK0_single hμ m), lp.coeFn_sub,
    Pi.sub_apply, h.invCol_apply, lp.coeFn_smul, Pi.smul_apply, smul_eq_mul,
    single_apply_eq_kron, A.Bph_eq_Dhalf k hμ]
  have hdn := A.Dhalf_ne_zero k hμ n
  have hdm := A.Dhalf_ne_zero k hμ m
  have htn := A.tB_ne_zero k hμ n
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp
  · rw [kron_of_ne hnm]
    field_simp
    ring

variable (A k ε) in
/-- `cₘ = (1 + 𝒦)⁻¹eₘ/dₘ ∈ ℓ²`, so `𝒟^(-1/2)cₘ` is column `m` of `χ_r⁻¹` (`chiInv_eq`). -/
noncomputable def invColK (hμ : A.μ ≠ 0) (m : ℤ) : ℓ²(ℤ, ℂ) :=
  (A.Dhalf k m)⁻¹ • Ring.inverse (A.onePlusK k ε hμ) (lp.single 2 m 1)

variable (A k ε) in
/-- `χ_r⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, entrywise, with `Ring.inverse` as in `gammaOp`. -/
noncomputable def chiInv (hμ : A.μ ≠ 0) (n m : ℤ) : ℂ :=
  Ring.inverse (A.onePlusK k ε hμ) (lp.single 2 m 1) n / (A.Dhalf k n * A.Dhalf k m)

/-- `χ_r⁻¹,nm = cₘ,ₙ/dₙ`. -/
theorem chiInv_eq (hμ : A.μ ≠ 0) (n m : ℤ) :
    A.chiInv k ε hμ n m = A.invColK k ε hμ m n / A.Dhalf k n := by
  rw [chiInv, invColK, lp.coeFn_smul, Pi.smul_apply, smul_eq_mul]
  field_simp

/-- **`Γ_r = β²(χ_r⁻¹ - B⁻¹)`**, entrywise, for every `μ ≠ 0`. Where `1 + 𝒦` is not invertible,
`χ_r⁻¹ = 0` (`Ring.inverse`), so `Γ_r = -β²δₙₘ/Bₙ` there. -/
theorem gammaOp_eq_chiInv (hμ : A.μ ≠ 0) (n m : ℤ) :
    A.gammaOp k ε hμ n m = A.β ^ 2 * (A.chiInv k ε hμ n m - kron n m / A.Bph k n) := by
  rw [gammaOp, chiInv, _root_.sub_apply,
    inverse_apply_of_eq (show IsUnit (A.onePlusK0 k hμ) from A.isUnit_one_add_K0 k hμ)
      (onePlusK0_single hμ m), lp.coeFn_sub, Pi.sub_apply, lp.coeFn_smul, Pi.smul_apply,
    smul_eq_mul, single_apply_eq_kron, A.Bph_eq_Dhalf k hμ]
  have hdn := A.Dhalf_ne_zero k hμ n
  have hdm := A.Dhalf_ne_zero k hμ m
  have htn := A.tB_ne_zero k hμ n
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp
  · rw [kron_of_ne hnm]
    field_simp
    ring

/-- The columns of `χ_r⁻¹` are in the class `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem summable_chiInv (hμ : A.μ ≠ 0) (m : ℤ) :
    Summable fun n ↦ ‖A.chiInv k ε hμ n m / A.z k n‖ ^ 2 := by
  simpa only [chiInv_eq hμ] using A.summable_div_Dhalf k hμ (A.invColK k ε hμ m)

/-- **`χ_r⁻¹` is a right inverse** where `1 + 𝒦` is invertible: `∑ₗ χ_r,nl χ_r⁻¹,lm = δₙₘ`. -/
theorem hasSum_chiR_mul_chiInv (hμ : A.μ ≠ 0) (hK : IsUnit (A.onePlusK k ε hμ)) (n m : ℤ) :
    HasSum (fun l ↦ A.chiR k ε n l * A.chiInv k ε hμ l m) (kron n m) := by
  have h := A.hasSum_chiR_mul k ε hμ (A.summable_div_Dhalf k hμ (A.invColK k ε hμ m)) n
  have hT : A.onePlusK k ε hμ (A.invColK k ε hμ m) = (A.Dhalf k m)⁻¹ • lp.single 2 m 1 := by
    rw [invColK, map_smul, apply_inverse_of_isUnit hK]
  rw [A.DhalfVec_div k hμ, hT, lp.coeFn_smul, Pi.smul_apply, smul_eq_mul,
    single_apply_eq_kron] at h
  convert h using 1
  · funext l
    rw [chiInv_eq hμ]
  by_cases hnm : n = m
  · subst hnm
    field_simp [A.Dhalf_ne_zero k hμ n]
  · simp [kron_of_ne hnm]

/-- **`χ_r⁻¹` is the only right inverse in the class**: where `1 + 𝒦` is invertible, a `v` with
`∑ₙ |vₙ/zₙ|² < ∞` and `∑ₗ χ_r,nl vₗ = δₙₘ` is column `m` of `χ_r⁻¹`. -/
theorem eq_chiInv_of_hasSum (hμ : A.μ ≠ 0) (hK : IsUnit (A.onePlusK k ε hμ)) {m : ℤ}
    {v : ℤ → ℂ} (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun l ↦ A.chiR k ε n l * v l) (kron n m)) (n : ℤ) :
    v n = A.chiInv k ε hμ n m := by
  have hT : A.onePlusK k ε hμ (A.DhalfVec k hv) = (A.Dhalf k m)⁻¹ • lp.single 2 m 1 := by
    ext l
    have h := (A.hasSum_chiR_mul k ε hμ hv l).unique (hχ l)
    have hd := A.Dhalf_ne_zero k hμ l
    rw [lp.coeFn_smul, Pi.smul_apply, smul_eq_mul, single_apply_eq_kron]
    by_cases hlm : l = m
    · subst hlm
      rw [kron_self] at h ⊢
      field_simp
      linear_combination h
    · rw [kron_of_ne hlm] at h ⊢
      simpa [hd] using h
  have hc := inverse_apply_of_eq hK hT
  rw [map_smul] at hc
  rw [chiInv_eq hμ, invColK, hc]
  simp [DhalfVec, A.Dhalf_ne_zero k hμ n]

/-- **`χ_r⁻¹` is symmetric** where `1 + 𝒦` is invertible, as `𝒦` is: with `R = (1 + 𝒦)⁻¹`,
`(Reₘ)ₙ = ∑ₗ (Reₘ)ₗ((1 + 𝒦)Reₙ)ₗ = ∑ₗ (Reₙ)ₗ((1 + 𝒦)Reₘ)ₗ = (Reₙ)ₘ`. -/
theorem chiInv_symm (hμ : A.μ ≠ 0) (hK : IsUnit (A.onePlusK k ε hμ)) (n m : ℤ) :
    A.chiInv k ε hμ n m = A.chiInv k ε hμ m n := by
  have hpair := tsum_mul_one_add_hsOp_comm (A.Kr k ε) (A.summable_Kr_sq k ε hμ) (Kr_symm hμ)
    (Ring.inverse (A.onePlusK k ε hμ) (lp.single 2 m 1))
    (Ring.inverse (A.onePlusK k ε hμ) (lp.single 2 n 1))
  rw [show (1 + hsOp (A.Kr k ε) (A.summable_Kr_sq k ε hμ)) = A.onePlusK k ε hμ from rfl,
    apply_inverse_of_isUnit hK, apply_inverse_of_isUnit hK] at hpair
  simp only [single_apply_eq_kron, kron, mul_ite, mul_one, mul_zero, tsum_ite_eq] at hpair
  rw [chiInv, chiInv, hpair, mul_comm (A.Dhalf k n)]

/-- **`χ_r⁻¹` is a left inverse** where `1 + 𝒦` is invertible: `∑ₗ χ_r⁻¹,nl χ_r,lm = δₙₘ`. -/
theorem hasSum_chiInv_mul_chiR (hμ : A.μ ≠ 0) (hK : IsUnit (A.onePlusK k ε hμ)) (n m : ℤ) :
    HasSum (fun l ↦ A.chiInv k ε hμ n l * A.chiR k ε l m) (kron n m) := by
  convert A.hasSum_chiR_mul_chiInv hμ hK m n using 1
  · funext l
    rw [chiInv_symm hμ hK n l, A.chiR_symm k ε hμ l m, mul_comm]
  · exact kron_comm n m

end Params

end HubbardAtom
