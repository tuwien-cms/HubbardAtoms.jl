import HubbardAtomLean.GammaBoxes

/-!
# The operator vertex for a general `χ` on a fermionic lattice

The proofs of items 6(b)–(d) for the density and magnetic channels (`ChiCompact.lean`,
`GammaOperator.lean`, `GammaClosedForm.lean`, `GammaBoxes.lean`) use only the bounds of
item 6(a) on `χ_r` and on the bubble. This file repeats them for a general matrix `X` and
diagonal `b` on the lattice `zₙ = μ + i(νₙ + ω_k/2)` of `A : Params`, `μ ≠ 0`, so that the field
sectors only need their own bounds:

* `Params.ChiDecay`: `|X_nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m` and `|X_nn + β/zₙ²| ≤ c|zₙ|⁻³`;
  `Params.BubbleDecay`: `bₙ ≠ 0` and `zₙ(tₙ - 1)` bounded, with `tₙ = -zₙ²bₙ/β`;
* `Params.X_eq_Dhalf`, `Params.summable_KX_sq`: `X = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)` with `𝒦`
  Hilbert–Schmidt; `Params.b_eq_Dhalf`, `Params.isUnit_onePlusK0X`: `b = 𝒟^(1/2)(1 + 𝒦₀)𝒟^(1/2)`
  with `1 + 𝒦₀ = diag(tₙ)` invertible;
* `Params.gammaOpX`: `Γ = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)`, entrywise;
  `Params.isUnit_onePlusKX_iff`: `1 + 𝒦` is invertible iff `X` has no nonzero null vector `v`
  with `∑ₙ |vₙ/zₙ|² < ∞`; `Params.exists_bound_of_nullX`: such null vectors are bounded;
* `Params.gammaOpX_eq_chiInvX`: `Γ = β²(X⁻¹ - b⁻¹)` with `X⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`
  (`chiInvX`), which, where `1 + 𝒦` is invertible, is a right inverse of `X`
  (`chiInvX_isRightInverse`), the only one with columns in the class (`eq_chiInvX_of_hasSum`), and
  for a symmetric `X` also a left inverse (`chiInvX_isLeftInverse`);
* `Params.gammaOpX_eq_of_isRightInverse`: for a symmetric `X` with a right inverse `Y` whose
  columns are bounded, `1 + 𝒦` is invertible and `Γ = β²(Y - b⁻¹)`;
* `Params.tendsto_gammaBoxX`: wherever `1 + 𝒦` is invertible, the box vertices
  `β²[(X|_M)⁻¹ - b⁻¹]` converge to `Γ` entrywise.

`Params.chiDecay_chiR`: the density and magnetic `χ_r` is an instance.
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (X : ℤ → ℤ → ℂ) (b : ℤ → ℂ)

/-! ### The bounds -/

/-- The bounds of item 6(a) on a matrix `X`: `|X_nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m`, and
`|X_nn + β/zₙ²| ≤ c|zₙ|⁻³`, i.e. `X_nn = -βzₙ⁻²[1 + O(|zₙ|⁻¹)]`. -/
structure ChiDecay : Prop where
  off : ∃ c, ∀ n m, n ≠ m → ‖X n m‖ ≤ c / (‖A.z k n‖ ^ 2 * ‖A.z k m‖ ^ 2)
  diag : ∃ c, ∀ n, ‖X n n + A.β / A.z k n ^ 2‖ ≤ c / ‖A.z k n‖ ^ 3

/-- `tₙ = -zₙ²bₙ/β`, the diagonal of `1 + 𝒦₀` for the bubble `b`. -/
noncomputable def tX (n : ℤ) : ℂ := -(A.z k n ^ 2 / A.β) * b n

/-- The bubble `b` decays as `-β/zₙ²`: `bₙ ≠ 0` and `|zₙ(tₙ - 1)| ≤ C`. -/
structure BubbleDecay : Prop where
  ne : ∀ n, b n ≠ 0
  bdd : ∃ C, ∀ n, ‖A.z k n * (A.tX k b n - 1)‖ ≤ C

/-- The density and magnetic `χ_r` satisfies the bounds, by `ChiBounds.lean`. -/
theorem chiDecay_chiR (ε : ℂ) (hμ : A.μ ≠ 0) : A.ChiDecay k (A.chiR k ε) :=
  ⟨A.norm_chiR_le k ε hμ, A.norm_chiR_diag_le k ε hμ⟩

/-! ### `X = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)` with `𝒦` Hilbert–Schmidt -/

/-- `𝒦_nm = -(zₙzₘ/β)X_nm - δ_nm`. -/
noncomputable def KX (n m : ℤ) : ℂ := -(A.z k n * A.z k m / A.β) * X n m - kron n m

/-- `X = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)`, entrywise. -/
theorem X_eq_Dhalf (hμ : A.μ ≠ 0) (n m : ℤ) :
    X n m = A.Dhalf k n * (kron n m + A.KX k X n m) * A.Dhalf k m := by
  have hz := A.z_ne_zero k hμ n
  have hv := A.z_ne_zero k hμ m
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hs : ((√A.β : ℝ) : ℂ) ^ 2 = A.β := by rw [← ofReal_pow, Real.sq_sqrt A.β_pos.le]
  have hd : A.Dhalf k n * A.Dhalf k m = -A.β / (A.z k n * A.z k m) := by
    rw [Dhalf, Dhalf, div_mul_div_comm, mul_mul_mul_comm, I_mul_I, ← sq, hs]
    ring
  rw [KX, add_sub_cancel, show ∀ x y c : ℂ, x * c * y = x * y * c from fun _ _ _ ↦ by ring, hd]
  field_simp

/-- `|𝒦_nm|² ≤ C|zₙ|⁻²(|zₘ|⁻² + δ_nm)`, by the bounds on `X`. -/
theorem norm_KX_sq_le (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) : ∃ C, ∀ n m, ‖A.KX k X n m‖ ^ 2 ≤
    C * (‖A.z k n‖⁻¹ ^ 2 * (‖A.z k m‖⁻¹ ^ 2 + if n = m then 1 else 0)) := by
  obtain ⟨c₁, hc₁⟩ := hX.off
  obtain ⟨c₂, hc₂⟩ := hX.diag
  have hβ := A.β_pos
  refine ⟨(c₁ / A.β) ^ 2 + (c₂ / A.β) ^ 2, fun n m ↦ ?_⟩
  have hz := norm_pos_iff.mpr (A.z_ne_zero k hμ n)
  have hv := norm_pos_iff.mpr (A.z_ne_zero k hμ m)
  by_cases hnm : n = m
  · subst hnm
    -- `|zₙ𝒦_nn| = |zₙ|³|X_nn + β/zₙ²|/β ≤ c₂/β`
    have h : A.z k n * A.KX k X n n =
        -(A.z k n ^ 3 / A.β) * (X n n + A.β / A.z k n ^ 2) := by
      have := A.z_ne_zero k hμ n
      have hβ' : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr hβ.ne'
      rw [KX, kron_self]
      field_simp
      ring
    have hb : ‖A.z k n * A.KX k X n n‖ ≤ c₂ / A.β := by
      rw [h, norm_mul, norm_neg, norm_div, norm_pow, norm_real, Real.norm_of_nonneg hβ.le,
        div_mul_eq_mul_div, div_le_div_iff_of_pos_right hβ, mul_comm,
        ← le_div_iff₀ (by positivity)]
      exact hc₂ n
    have hK : ‖A.KX k X n n‖ ≤ c₂ / A.β * ‖A.z k n‖⁻¹ := by
      rw [le_mul_inv_iff₀ hz, mul_comm, ← norm_mul]
      exact hb
    calc ‖A.KX k X n n‖ ^ 2 ≤ (c₂ / A.β * ‖A.z k n‖⁻¹) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hK 2
      _ ≤ _ := by
        simp only [ite_true]
        nlinarith [sq_nonneg (c₁ / A.β), sq_nonneg (c₂ / A.β), sq_nonneg ‖A.z k n‖⁻¹,
          mul_nonneg (sq_nonneg (c₁ / A.β + c₂ / A.β)) (pow_nonneg (sq_nonneg ‖A.z k n‖⁻¹) 2)]
  · have hK : ‖A.KX k X n m‖ ≤ c₁ / A.β * ‖A.z k n‖⁻¹ * ‖A.z k m‖⁻¹ := by
      rw [KX, kron_of_ne hnm, sub_zero, norm_mul, norm_neg, norm_div, norm_mul, norm_real,
        Real.norm_of_nonneg hβ.le]
      calc ‖A.z k n‖ * ‖A.z k m‖ / A.β * ‖X n m‖
          ≤ ‖A.z k n‖ * ‖A.z k m‖ / A.β * (c₁ / (‖A.z k n‖ ^ 2 * ‖A.z k m‖ ^ 2)) :=
            mul_le_mul_of_nonneg_left (hc₁ n m hnm) (by positivity)
        _ = c₁ / A.β * ‖A.z k n‖⁻¹ * ‖A.z k m‖⁻¹ := by field_simp
    calc ‖A.KX k X n m‖ ^ 2 ≤ (c₁ / A.β * ‖A.z k n‖⁻¹ * ‖A.z k m‖⁻¹) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hK 2
      _ ≤ _ := by
        simp only [hnm, ite_false, add_zero]
        nlinarith [mul_nonneg (sq_nonneg (c₂ / A.β))
          (mul_nonneg (sq_nonneg ‖A.z k n‖⁻¹) (sq_nonneg ‖A.z k m‖⁻¹))]

/-- **`𝒦` is Hilbert–Schmidt**, by `norm_KX_sq_le`. -/
theorem summable_KX_sq (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) :
    Summable fun p : ℤ × ℤ ↦ ‖A.KX k X p.1 p.2‖ ^ 2 := by
  obtain ⟨C, hC⟩ := A.norm_KX_sq_le k X hμ hX
  exact ((A.summable_majorant k hμ).mul_left C).of_nonneg_of_le (fun _ ↦ by positivity)
    fun p ↦ (hC p.1 p.2).trans_eq (by ring)

/-- **`𝒦` is compact** on `ℓ²(ℤ)`. -/
theorem isCompactOperator_KX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) :
    IsCompactOperator (hsOp (A.KX k X) (A.summable_KX_sq k X hμ hX)) :=
  isCompactOperator_hsOp _ _

/-- `𝒦` is symmetric where `X` is. -/
theorem KX_symm (hs : ∀ n m, X n m = X m n) (n m : ℤ) : A.KX k X n m = A.KX k X m n := by
  rw [KX, KX, hs, kron_comm, mul_comm (A.z k n)]

/-! ### The bubble -/

/-- `tₙ ≠ 0`. -/
theorem tX_ne_zero (hμ : A.μ ≠ 0) (hb : A.BubbleDecay k b) (n : ℤ) : A.tX k b n ≠ 0 := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  exact mul_ne_zero (neg_ne_zero.mpr (div_ne_zero (pow_ne_zero 2 (A.z_ne_zero k hμ n)) hβ))
    (hb.ne n)

/-- `b = 𝒟^(1/2)(1 + 𝒦₀)𝒟^(1/2)`: `bₙ = dₙ²tₙ`. -/
theorem b_eq_Dhalf (hμ : A.μ ≠ 0) (n : ℤ) : b n = A.Dhalf k n ^ 2 * A.tX k b n := by
  have hz := A.z_ne_zero k hμ n
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  rw [A.Dhalf_sq, tX]
  field_simp

/-- `tₙ → 1`. -/
theorem tendsto_tX (hμ : A.μ ≠ 0) (hb : A.BubbleDecay k b) :
    Tendsto (A.tX k b) cofinite (𝓝 1) := by
  obtain ⟨C, hC⟩ := hb.bdd
  rw [tendsto_iff_norm_sub_tendsto_zero]
  have h0 : Tendsto (fun n ↦ C * ‖(A.z k n)⁻¹‖) cofinite (𝓝 0) := by
    simpa using (A.tendsto_inv_z k).norm.const_mul C
  refine squeeze_zero (fun _ ↦ norm_nonneg _) (fun n ↦ ?_) h0
  have hz := A.z_ne_zero k hμ n
  calc ‖A.tX k b n - 1‖ = ‖(A.z k n)⁻¹‖ * ‖A.z k n * (A.tX k b n - 1)‖ := by
        rw [← norm_mul, inv_mul_cancel_left₀ hz]
    _ ≤ ‖(A.z k n)⁻¹‖ * C := mul_le_mul_of_nonneg_left (hC n) (norm_nonneg _)
    _ = C * ‖(A.z k n)⁻¹‖ := mul_comm _ _

/-- `tₙ - 1` is square-summable. -/
theorem summable_tX_sub_one_sq (hμ : A.μ ≠ 0) (hb : A.BubbleDecay k b) :
    Summable fun n ↦ ‖A.tX k b n - 1‖ ^ 2 := by
  obtain ⟨C, hC⟩ := hb.bdd
  exact A.summable_sq_of_norm_mul_z_le k hμ hC

/-- `1/tₙ - 1` is square-summable, as `tₙ → 1` and `tₙ ≠ 0` keep `1/tₙ` bounded. -/
theorem summable_inv_tX_sub_one_sq (hμ : A.μ ≠ 0) (hb : A.BubbleDecay k b) :
    Summable fun n ↦ ‖(A.tX k b n)⁻¹ - 1‖ ^ 2 := by
  obtain ⟨C, hC⟩ := hb.bdd
  obtain ⟨D, hD⟩ :=
    ((A.tendsto_tX k b hμ hb).inv₀ one_ne_zero).norm.bddAbove_range_of_cofinite
  refine A.summable_sq_of_norm_mul_z_le k hμ (C := D * C) fun n ↦ ?_
  have ht := A.tX_ne_zero k b hμ hb n
  have hDn : ‖(A.tX k b n)⁻¹‖ ≤ D := hD ⟨n, rfl⟩
  rw [show A.z k n * ((A.tX k b n)⁻¹ - 1) =
      -((A.tX k b n)⁻¹ * (A.z k n * (A.tX k b n - 1))) by field_simp; ring, norm_neg, norm_mul]
  exact mul_le_mul hDn (hC n) (norm_nonneg _) ((norm_nonneg _).trans hDn)

/-- `1 + 𝒦₀ = diag(tₙ)` on `ℓ²(ℤ)`. -/
noncomputable def onePlusK0X (hμ : A.μ ≠ 0) (hb : A.BubbleDecay k b) :
    ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (diagK fun n ↦ A.tX k b n - 1) (summable_diag_sq (A.summable_tX_sub_one_sq k b hμ hb))

/-- **`1 + 𝒦₀` is invertible** on `ℓ²(ℤ)`, with the inverse `1 + diag(1/tₙ - 1)`. -/
theorem isUnit_onePlusK0X (hμ : A.μ ≠ 0) (hb : A.BubbleDecay k b) :
    IsUnit (A.onePlusK0X k b hμ hb) := by
  refine ⟨⟨_, 1 + hsOp (diagK fun n ↦ (A.tX k b n)⁻¹ - 1)
    (summable_diag_sq (A.summable_inv_tX_sub_one_sq k b hμ hb)), ?_, ?_⟩, rfl⟩ <;>
  · ext v n
    have ht := A.tX_ne_zero k b hμ hb n
    simp [hsOp_diag_apply (A.summable_tX_sub_one_sq k b hμ hb),
      hsOp_diag_apply (A.summable_inv_tX_sub_one_sq k b hμ hb)]
    field_simp
    ring

/-- `(1 + 𝒦₀)(eₘ/tₘ) = eₘ`. -/
theorem onePlusK0X_single (hμ : A.μ ≠ 0) (hb : A.BubbleDecay k b) (m : ℤ) :
    A.onePlusK0X k b hμ hb ((A.tX k b m)⁻¹ • lp.single 2 m 1) = lp.single 2 m 1 := by
  ext n
  simp only [onePlusK0X, _root_.add_apply, one_apply_eq_self, lp.coeFn_add, Pi.add_apply,
    hsOp_diag_apply (A.summable_tX_sub_one_sq k b hμ hb), lp.coeFn_smul, Pi.smul_apply,
    smul_eq_mul, single_apply_eq_kron]
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp [A.tX_ne_zero k b hμ hb n]
    ring
  · simp [kron_of_ne hnm]

/-- `[(1 + 𝒦₀)⁻¹eₘ]ₙ = δₙₘ/tₘ`. -/
theorem inverse_onePlusK0X_single (hμ : A.μ ≠ 0) (hb : A.BubbleDecay k b) (n m : ℤ) :
    Ring.inverse (A.onePlusK0X k b hμ hb) (lp.single 2 m 1) n = kron n m / A.tX k b m := by
  rw [inverse_apply_of_eq (A.isUnit_onePlusK0X k b hμ hb) (A.onePlusK0X_single k b hμ hb m),
    lp.coeFn_smul, Pi.smul_apply, smul_eq_mul, single_apply_eq_kron, div_eq_inv_mul]

/-! ### The operator `Γ` and the criterion -/

/-- `1 + 𝒦` on `ℓ²(ℤ)`. -/
noncomputable def onePlusKX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (A.KX k X) (A.summable_KX_sq k X hμ hX)

/-- `Γ_nm = β²/(dₙdₘ) [(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]ₙₘ`, the entries taken on the unit vectors, with
`Ring.inverse` (zero where `1 + 𝒦` is not invertible). -/
noncomputable def gammaOpX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (hb : A.BubbleDecay k b)
    (n m : ℤ) : ℂ :=
  A.β ^ 2 / (A.Dhalf k n * A.Dhalf k m) *
    ((Ring.inverse (A.onePlusKX k X hμ hX) - Ring.inverse (A.onePlusK0X k b hμ hb))
      (lp.single 2 m 1) n)

/-- **Fredholm alternative**: `1 + 𝒦` is invertible iff it is injective. -/
theorem isUnit_onePlusKX_iff_injective (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) :
    IsUnit (A.onePlusKX k X hμ hX) ↔ Function.Injective (A.onePlusKX k X hμ hX) := by
  refine ⟨fun h ↦ (ContinuousLinearMap.isUnit_iff_bijective.mp h).1, fun hinj ↦ ?_⟩
  rcases (A.isCompactOperator_KX k X hμ hX).hasEigenvalue_or_mem_resolventSet (μ := -1)
    (by norm_num) with hev | hres
  · obtain ⟨x, hx⟩ := hev.exists_hasEigenvector
    refine absurd (hinj ?_) hx.2
    have h := hx.apply_eq_smul
    simp only [ContinuousLinearMap.coe_coe] at h
    simp [onePlusKX, h]
  · rw [spectrum.mem_resolventSet_iff, Algebra.algebraMap_eq_smul_one,
      show (-1 : ℂ) • (1 : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ)) - hsOp (A.KX k X) (A.summable_KX_sq k X hμ hX) =
        -A.onePlusKX k X hμ hX by rw [onePlusKX]; module, IsUnit.neg_iff] at hres
    exact hres

/-- `X v = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)v`, row by row, for `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem hasSum_X_mul (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2) (n : ℤ) :
    HasSum (fun m ↦ X n m * v m) (A.Dhalf k n * A.onePlusKX k X hμ hX (A.DhalfVec k hv) n) := by
  have h := ((hasSum_ite_eq n (A.DhalfVec k hv n)).add
    (hasSum_hsOp (A.KX k X) (A.summable_KX_sq k X hμ hX) (A.DhalfVec k hv) n)).mul_left
    (A.Dhalf k n)
  convert h using 1
  · funext m
    rw [A.X_eq_Dhalf k X hμ]
    by_cases hnm : n = m
    · subst hnm
      simp [kron_self, DhalfVec]
      ring
    · simp [kron_of_ne hnm, Ne.symm hnm, DhalfVec]
      ring
  · simp [onePlusKX]

/-- A null vector of `X` with `∑ₙ |vₙ/zₙ|² < ∞` gives `(1 + 𝒦)𝒟^(1/2)v = 0`. -/
theorem onePlusKX_DhalfVec_eq_zero (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2) (hχ : ∀ n, HasSum (fun m ↦ X n m * v m) 0) :
    A.onePlusKX k X hμ hX (A.DhalfVec k hv) = 0 := by
  ext n
  have h := (A.hasSum_X_mul k X hμ hX hv n).unique (hχ n)
  simpa [A.Dhalf_ne_zero k hμ n] using h

/-- **The criterion**: `ker(1 + 𝒦) = 0` iff `X` has no nonzero null vector `v` with
`∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem injective_onePlusKX_iff (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) :
    Function.Injective (A.onePlusKX k X hμ hX) ↔ ∀ v : ℤ → ℂ,
      Summable (fun n ↦ ‖v n / A.z k n‖ ^ 2) →
        (∀ n, HasSum (fun m ↦ X n m * v m) 0) → v = 0 := by
  have hd := A.Dhalf_ne_zero k hμ
  constructor
  · intro hinj v hv hχ
    have h := hinj ((A.onePlusKX_DhalfVec_eq_zero k X hμ hX hv hχ).trans (map_zero _).symm)
    funext n
    have hn := congrArg (fun w : ℓ²(ℤ, ℂ) ↦ w n) h
    simpa [DhalfVec, hd n] using hn
  · intro h
    rw [injective_iff_map_eq_zero]
    intro w hw
    have h0 := h _ (A.summable_div_Dhalf k hμ w) fun n ↦ by
      simpa [A.DhalfVec_div k hμ, hw] using
        A.hasSum_X_mul k X hμ hX (A.summable_div_Dhalf k hμ w) n
    ext n
    have hn := congrFun h0 n
    simp only [Pi.zero_apply, div_eq_zero_iff, hd n, or_false] at hn
    simp [hn]

/-- **`Γ` exists iff `X` has no null vector**: `1 + 𝒦` is invertible iff `X` has no nonzero null
vector `v` with `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem isUnit_onePlusKX_iff (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) :
    IsUnit (A.onePlusKX k X hμ hX) ↔ ∀ v : ℤ → ℂ, Summable (fun n ↦ ‖v n / A.z k n‖ ^ 2) →
      (∀ n, HasSum (fun m ↦ X n m * v m) 0) → v = 0 :=
  (A.isUnit_onePlusKX_iff_injective k X hμ hX).trans (A.injective_onePlusKX_iff k X hμ hX)

/-- **Null vectors are bounded**: a null vector `v` of `X` with `∑ₙ |vₙ/zₙ|² < ∞` has
`|vₙ| ≤ C`, as `w = 𝒟^(1/2)v = -𝒦w` gives `|wₙ| ≤ ‖𝒦ₙ·‖‖w‖ = O(|zₙ|⁻¹)`. -/
theorem exists_bound_of_nullX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2) (hχ : ∀ n, HasSum (fun m ↦ X n m * v m) 0) :
    ∃ C, ∀ n, ‖v n‖ ≤ C := by
  obtain ⟨C, hC⟩ := A.norm_KX_sq_le k X hμ hX
  have ha := A.summable_inv_norm_z_sq k hμ
  have hβ := A.β_pos
  set w := A.DhalfVec k hv with hwdef
  set S := ∑' m, ‖A.z k m‖⁻¹ ^ 2
  refine ⟨√(C * (S + 1) * ‖w‖ ^ 2 / A.β), fun n ↦ ?_⟩
  have hz := norm_pos_iff.mpr (A.z_ne_zero k hμ n)
  -- `wₙ = -⟪𝒦ₙ·, w⟫`
  have hw : w n = -inner ℂ (hsRow (A.KX k X) (A.summable_KX_sq k X hμ hX) n) w := by
    have h := congrArg (fun x : ℓ²(ℤ, ℂ) ↦ x n) (A.onePlusKX_DhalfVec_eq_zero k X hμ hX hv hχ)
    rw [← hwdef] at h
    simp only [onePlusKX, _root_.add_apply, one_apply_eq_self, lp.coeFn_add, Pi.add_apply,
      hsOp_apply_eq_inner, lp.coeFn_zero, Pi.zero_apply] at h
    linear_combination h
  -- `‖𝒦ₙ·‖² ≤ C|zₙ|⁻²(S + 1)`
  have hrow : ‖hsRow (A.KX k X) (A.summable_KX_sq k X hμ hX) n‖ ^ 2 ≤
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
  have hCS := norm_inner_sq_le (hsRow (A.KX k X) (A.summable_KX_sq k X hμ hX) n) w
  calc ‖A.z k n‖ ^ 2 * ‖inner ℂ (hsRow (A.KX k X) (A.summable_KX_sq k X hμ hX) n) w‖ ^ 2 / A.β
      ≤ ‖A.z k n‖ ^ 2 * (C * (‖A.z k n‖⁻¹ ^ 2 * (S + 1)) * ‖w‖ ^ 2) / A.β := by
        gcongr
        exact hCS.trans (mul_le_mul_of_nonneg_right hrow (by positivity))
    _ = C * (S + 1) * ‖w‖ ^ 2 / A.β := by
        field_simp

/-! ### `Γ = β²(X⁻¹ - b⁻¹)` -/

/-- `cₘ = (1 + 𝒦)⁻¹eₘ/dₘ ∈ ℓ²`, so `𝒟^(-1/2)cₘ` is column `m` of `X⁻¹` (`chiInvX_eq`). -/
noncomputable def invColKX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (m : ℤ) : ℓ²(ℤ, ℂ) :=
  (A.Dhalf k m)⁻¹ • Ring.inverse (A.onePlusKX k X hμ hX) (lp.single 2 m 1)

/-- `X⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, entrywise, with `Ring.inverse` as in `gammaOpX`. -/
noncomputable def chiInvX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (n m : ℤ) : ℂ :=
  Ring.inverse (A.onePlusKX k X hμ hX) (lp.single 2 m 1) n / (A.Dhalf k n * A.Dhalf k m)

/-- `X⁻¹_nm = cₘ,ₙ/dₙ`. -/
theorem chiInvX_eq (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (n m : ℤ) :
    A.chiInvX k X hμ hX n m = A.invColKX k X hμ hX m n / A.Dhalf k n := by
  rw [chiInvX, invColKX, lp.coeFn_smul, Pi.smul_apply, smul_eq_mul]
  field_simp

/-- **`Γ = β²(X⁻¹ - b⁻¹)`**, entrywise. Where `1 + 𝒦` is not invertible, `X⁻¹ = 0`
(`Ring.inverse`), so `Γ = -β²δₙₘ/bₙ` there. -/
theorem gammaOpX_eq_chiInvX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (hb : A.BubbleDecay k b)
    (n m : ℤ) :
    A.gammaOpX k X b hμ hX hb n m = A.β ^ 2 * (A.chiInvX k X hμ hX n m - kron n m / b n) := by
  rw [gammaOpX, chiInvX, _root_.sub_apply, lp.coeFn_sub, Pi.sub_apply,
    A.inverse_onePlusK0X_single k b hμ hb, A.b_eq_Dhalf k b hμ]
  have hdn := A.Dhalf_ne_zero k hμ n
  have hdm := A.Dhalf_ne_zero k hμ m
  have htn := A.tX_ne_zero k b hμ hb n
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp
  · rw [kron_of_ne hnm]
    field_simp
    ring

/-- The columns of `X⁻¹` are in the class `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem summable_chiInvX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (m : ℤ) :
    Summable fun n ↦ ‖A.chiInvX k X hμ hX n m / A.z k n‖ ^ 2 := by
  simpa only [A.chiInvX_eq k X hμ hX] using A.summable_div_Dhalf k hμ (A.invColKX k X hμ hX m)

/-- **`X⁻¹` is a right inverse** where `1 + 𝒦` is invertible: `∑ₗ X_nl X⁻¹_lm = δₙₘ`. -/
theorem chiInvX_isRightInverse (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X)
    (hK : IsUnit (A.onePlusKX k X hμ hX)) : IsRightInverse X (A.chiInvX k X hμ hX) := by
  intro n m
  have h := A.hasSum_X_mul k X hμ hX (A.summable_div_Dhalf k hμ (A.invColKX k X hμ hX m)) n
  have hT : A.onePlusKX k X hμ hX (A.invColKX k X hμ hX m) =
      (A.Dhalf k m)⁻¹ • lp.single 2 m 1 := by
    rw [invColKX, map_smul, apply_inverse_of_isUnit hK]
  rw [A.DhalfVec_div k hμ, hT, lp.coeFn_smul, Pi.smul_apply, smul_eq_mul,
    single_apply_eq_kron] at h
  convert h using 1
  · funext l
    rw [A.chiInvX_eq k X hμ hX]
  by_cases hnm : n = m
  · subst hnm
    field_simp [A.Dhalf_ne_zero k hμ n]
  · simp [kron_of_ne hnm]

/-- **`X⁻¹` is the only right inverse in the class**: where `1 + 𝒦` is invertible, a `v` with
`∑ₙ |vₙ/zₙ|² < ∞` and `∑ₗ X_nl vₗ = δₙₘ` is column `m` of `X⁻¹`. -/
theorem eq_chiInvX_of_hasSum (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X)
    (hK : IsUnit (A.onePlusKX k X hμ hX)) {m : ℤ} {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / A.z k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun l ↦ X n l * v l) (kron n m)) (n : ℤ) :
    v n = A.chiInvX k X hμ hX n m := by
  have hT : A.onePlusKX k X hμ hX (A.DhalfVec k hv) = (A.Dhalf k m)⁻¹ • lp.single 2 m 1 := by
    ext l
    have h := (A.hasSum_X_mul k X hμ hX hv l).unique (hχ l)
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
  rw [A.chiInvX_eq k X hμ hX, invColKX, hc]
  simp [DhalfVec, A.Dhalf_ne_zero k hμ n]

/-- **`X⁻¹` is symmetric** where `1 + 𝒦` is invertible and `X` is symmetric. -/
theorem chiInvX_symm (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (hs : ∀ n m, X n m = X m n)
    (hK : IsUnit (A.onePlusKX k X hμ hX)) (n m : ℤ) :
    A.chiInvX k X hμ hX n m = A.chiInvX k X hμ hX m n := by
  have hpair := tsum_mul_one_add_hsOp_comm (A.KX k X) (A.summable_KX_sq k X hμ hX)
    (A.KX_symm k X hs) (Ring.inverse (A.onePlusKX k X hμ hX) (lp.single 2 m 1))
    (Ring.inverse (A.onePlusKX k X hμ hX) (lp.single 2 n 1))
  rw [show (1 + hsOp (A.KX k X) (A.summable_KX_sq k X hμ hX)) = A.onePlusKX k X hμ hX from rfl,
    apply_inverse_of_isUnit hK, apply_inverse_of_isUnit hK] at hpair
  simp only [single_apply_eq_kron, kron, mul_ite, mul_one, mul_zero, tsum_ite_eq] at hpair
  rw [chiInvX, chiInvX, hpair, mul_comm (A.Dhalf k n)]

/-- **`X⁻¹` is a left inverse** where `1 + 𝒦` is invertible and `X` is symmetric. -/
theorem chiInvX_isLeftInverse (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (hs : ∀ n m, X n m = X m n)
    (hK : IsUnit (A.onePlusKX k X hμ hX)) : IsLeftInverse X (A.chiInvX k X hμ hX) := by
  intro n m
  convert A.chiInvX_isRightInverse k X hμ hX hK m n using 1
  · funext l
    rw [A.chiInvX_symm k X hμ hX hs hK n l, hs l m, mul_comm]
  · exact kron_comm n m

/-! ### A right inverse with bounded columns -/

/-- A column of `Y` with `|Y_nm| ≤ C` is in the class `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem summable_col_of_bdd (hμ : A.μ ≠ 0) {Y : ℤ → ℤ → ℂ} {m : ℤ} {C : ℝ}
    (hC : ∀ n, ‖Y n m‖ ≤ C) (c : ℂ) : Summable fun n ↦ ‖Y n m * c / A.z k n‖ ^ 2 := by
  refine A.summable_sq_of_norm_mul_z_le k hμ (C := C * ‖c‖) fun n ↦ ?_
  rw [mul_div_cancel₀ _ (A.z_ne_zero k hμ n), norm_mul]
  exact mul_le_mul_of_nonneg_right (hC n) (norm_nonneg _)

/-- **`1 + 𝒦` is invertible** where the symmetric `X` has a right inverse `Y` with bounded
columns: `cₘ = 𝒟^(1/2)(Y eₘ)dₘ` has `(1 + 𝒦)cₘ = eₘ`, and for `(1 + 𝒦)x = 0`,
`xₘ = ∑ₙ xₙ((1 + 𝒦)cₘ)ₙ = ∑ₙ cₘ,ₙ((1 + 𝒦)x)ₙ = 0`. -/
theorem isUnit_onePlusKX_of_isRightInverse (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X)
    (hs : ∀ n m, X n m = X m n) {Y : ℤ → ℤ → ℂ} (hY : IsRightInverse X Y)
    (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) : IsUnit (A.onePlusKX k X hμ hX) := by
  have hcol : ∀ m, ∃ c : ℓ²(ℤ, ℂ), A.onePlusKX k X hμ hX c = lp.single 2 m 1 := fun m ↦ by
    obtain ⟨C, hC⟩ := hbdd m
    have hv := A.summable_col_of_bdd k hμ hC (A.Dhalf k m)
    refine ⟨A.DhalfVec k hv, ?_⟩
    ext n
    have h₁ := A.hasSum_X_mul k X hμ hX hv n
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
  refine (A.isUnit_onePlusKX_iff_injective k X hμ hX).mpr ?_
  rw [injective_iff_map_eq_zero]
  intro x hx
  ext m
  obtain ⟨c, hc⟩ := hcol m
  have hpair := tsum_mul_one_add_hsOp_comm (A.KX k X) (A.summable_KX_sq k X hμ hX)
    (A.KX_symm k X hs) x c
  rw [show (1 + hsOp (A.KX k X) (A.summable_KX_sq k X hμ hX)) = A.onePlusKX k X hμ hX from rfl,
    hc, hx] at hpair
  simp only [single_apply_eq_kron, lp.coeFn_zero, Pi.zero_apply, mul_zero, tsum_zero, kron,
    mul_ite, mul_one] at hpair
  rw [tsum_ite_eq m] at hpair
  simpa using hpair

/-- **The right inverse is `X⁻¹`**: for a symmetric `X` with a right inverse `Y` whose columns
are bounded, `Y = X⁻¹`. -/
theorem eq_chiInvX_of_isRightInverse (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X)
    (hs : ∀ n m, X n m = X m n) {Y : ℤ → ℤ → ℂ} (hY : IsRightInverse X Y)
    (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) (n m : ℤ) :
    Y n m = A.chiInvX k X hμ hX n m := by
  obtain ⟨C, hC⟩ := hbdd m
  have hv : Summable fun n ↦ ‖Y n m / A.z k n‖ ^ 2 := by
    simpa using A.summable_col_of_bdd k hμ hC 1
  exact A.eq_chiInvX_of_hasSum k X hμ hX
    (A.isUnit_onePlusKX_of_isRightInverse k X hμ hX hs hY hbdd) hv (fun n ↦ hY n m) n

/-- **The operator vertex is the closed form**: for a symmetric `X` with a right inverse `Y`
whose columns are bounded, `Γ = β²(Y - b⁻¹)`. -/
theorem gammaOpX_eq_of_isRightInverse (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X)
    (hb : A.BubbleDecay k b) (hs : ∀ n m, X n m = X m n) {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse X Y) (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) (n m : ℤ) :
    A.gammaOpX k X b hμ hX hb n m = A.β ^ 2 * (Y n m - kron n m / b n) := by
  rw [A.gammaOpX_eq_chiInvX k X b hμ hX hb,
    A.eq_chiInvX_of_isRightInverse k X hμ hX hs hY hbdd n m]

/-! ### Finite boxes -/

/-- `1 + 𝒦_s`, the compression of `1 + 𝒦` to `s × s`. -/
noncomputable def onePlusKX_box (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (s : Finset ℤ) :
    ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (truncK (A.KX k X) s) (summable_truncK_sq _ (A.summable_KX_sq k X hμ hX) s)

/-- **The compressions converge in norm**: `1 + 𝒦_M → 1 + 𝒦`. -/
theorem tendsto_onePlusKX_box (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) :
    Tendsto (fun M ↦ A.onePlusKX_box k X hμ hX (box k M)) atTop (𝓝 (A.onePlusKX k X hμ hX)) :=
  ((tendsto_hsOp_truncK _ (A.summable_KX_sq k X hμ hX)).comp (tendsto_box k)).const_add 1

/-- The box matrix `X|_s`. -/
noncomputable def chiBoxX (s : Finset ℤ) : Matrix s s ℂ := Matrix.of fun i j ↦ X i j

/-- Where `1 + 𝒦_s` is invertible, `X|_s` has the right inverse `𝒟^(-1/2)(1 + 𝒦_s)⁻¹𝒟^(-1/2)`. -/
theorem chiBoxX_mul (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) {s : Finset ℤ}
    (hu : IsUnit (A.onePlusKX_box k X hμ hX s)) :
    chiBoxX X s * Matrix.of (fun i j : s ↦
      Ring.inverse (A.onePlusKX_box k X hμ hX s) (lp.single 2 (j : ℤ) 1) i /
        (A.Dhalf k i * A.Dhalf k j)) = 1 := by
  set T := A.onePlusKX_box k X hμ hX s
  set y : ℤ → ℓ²(ℤ, ℂ) := fun m ↦ Ring.inverse T (lp.single 2 m 1)
  have hy : ∀ m n, n ∈ s → y m n + ∑ l ∈ s, A.KX k X n l * y m l = kron n m := fun m n hn ↦ by
    have hT : T (y m) = lp.single 2 m 1 := apply_inverse_of_isUnit hu _
    have h := congrArg (fun w : ℓ²(ℤ, ℂ) ↦ w n) hT
    simp only [T, onePlusKX_box, _root_.add_apply, one_apply_eq_self, lp.coeFn_add,
      Pi.add_apply] at h
    rw [hsOp_truncK_apply _ (A.summable_KX_sq k X hμ hX), single_apply_eq_kron] at h
    simpa only [hn, ite_true] using h
  have hd := A.Dhalf_ne_zero k hμ
  ext i j
  simp only [chiBoxX, Matrix.mul_apply, Matrix.of_apply, Matrix.one_apply]
  have h₁ : ∀ l : s, X i l * (y j l / (A.Dhalf k l * A.Dhalf k j)) =
      A.Dhalf k i / A.Dhalf k j * (kron i l * y j l + A.KX k X i l * y j l) := fun l ↦ by
    rw [A.X_eq_Dhalf k X hμ i l]
    field_simp [hd l, hd j]
  simp only [y] at h₁ hy
  simp only [h₁, ← Finset.mul_sum, Finset.sum_add_distrib]
  rw [Finset.sum_coe_sort s (fun l ↦ kron i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l),
    Finset.sum_coe_sort s (fun l ↦ A.KX k X i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l)]
  have hk : ∑ l ∈ s, kron i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l =
      Ring.inverse T (lp.single 2 (j : ℤ) 1) i := by
    simp only [kron, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq, i.2, ite_true]
  rw [hk, hy j i i.2]
  by_cases hij : i = j
  · subst hij
    rw [kron_self, ite_eq_left rfl, mul_one, div_self (hd i)]
  · rw [kron_of_ne fun h ↦ hij (Subtype.ext h), ite_eq_right hij, mul_zero]

/-- **The box matrix is invertible** where `1 + 𝒦_s` is. -/
theorem isUnit_chiBoxX_det (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) {s : Finset ℤ}
    (hu : IsUnit (A.onePlusKX_box k X hμ hX s)) : IsUnit (chiBoxX X s).det :=
  Matrix.isUnit_det_of_right_inverse (A.chiBoxX_mul k X hμ hX hu)

/-- **Finite boxes**: wherever `1 + 𝒦` is invertible, the box matrices are invertible for all
sufficiently large `M`. -/
theorem eventually_isUnit_chiBoxX_det (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X)
    (hu : IsUnit (A.onePlusKX k X hμ hX)) : ∀ᶠ M in atTop, IsUnit (chiBoxX X (box k M)).det :=
  ((A.tendsto_onePlusKX_box k X hμ hX).eventually (Units.isOpen.mem_nhds hu)).mono fun _ hM ↦
    A.isUnit_chiBoxX_det k X hμ hX hM

/-- `(X|_s)⁻¹ᵢⱼ = [(1 + 𝒦_s)⁻¹]ᵢⱼ/(dᵢdⱼ)`. -/
theorem chiBoxX_inv (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) {s : Finset ℤ}
    (hu : IsUnit (A.onePlusKX_box k X hμ hX s)) (i j : s) : (chiBoxX X s)⁻¹ i j =
      Ring.inverse (A.onePlusKX_box k X hμ hX s) (lp.single 2 (j : ℤ) 1) i /
        (A.Dhalf k i * A.Dhalf k j) := by
  rw [Matrix.inv_eq_right_inv (A.chiBoxX_mul k X hμ hX hu), Matrix.of_apply]

/-- The box vertex `β²[(X|_s)⁻¹ - b⁻¹]ₙₘ` for `n, m ∈ s`, zero outside the box. -/
noncomputable def gammaBoxX (s : Finset ℤ) (n m : ℤ) : ℂ :=
  if h : n ∈ s ∧ m ∈ s then A.β ^ 2 * ((chiBoxX X s)⁻¹ ⟨n, h.1⟩ ⟨m, h.2⟩ - kron n m / b n)
  else 0

/-- **The box vertices converge to `Γ`** wherever `1 + 𝒦` is invertible: `1 + 𝒦_M → 1 + 𝒦` in
norm, so `1 + 𝒦_M` is eventually invertible and `(1 + 𝒦_M)⁻¹ → (1 + 𝒦)⁻¹`. -/
theorem tendsto_gammaBoxX (hμ : A.μ ≠ 0) (hX : A.ChiDecay k X) (hb : A.BubbleDecay k b)
    (hu : IsUnit (A.onePlusKX k X hμ hX)) (n m : ℤ) :
    Tendsto (fun M ↦ A.gammaBoxX X b (box k M) n m) atTop
      (𝓝 (A.gammaOpX k X b hμ hX hb n m)) := by
  have hT := A.tendsto_onePlusKX_box k X hμ hX
  have hinv : Tendsto (fun M ↦ Ring.inverse (A.onePlusKX_box k X hμ hX (box k M))) atTop
      (𝓝 (Ring.inverse (A.onePlusKX k X hμ hX))) := by
    have h := NormedRing.inverse_continuousAt hu.unit
    rw [hu.unit_spec] at h
    exact h.tendsto.comp hT
  have he := ((lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 n).continuous.tendsto _).comp
    (((ContinuousLinearMap.apply ℂ _ (lp.single 2 m (1 : ℂ))).continuous.tendsto _).comp hinv)
  have hlim := (he.sub_const (Ring.inverse (A.onePlusK0X k b hμ hb) (lp.single 2 m 1) n)).const_mul
    (A.β ^ 2 / (A.Dhalf k n * A.Dhalf k m))
  rw [show A.gammaOpX k X b hμ hX hb n m = A.β ^ 2 / (A.Dhalf k n * A.Dhalf k m) *
      (Ring.inverse (A.onePlusKX k X hμ hX) (lp.single 2 m 1) n -
        Ring.inverse (A.onePlusK0X k b hμ hb) (lp.single 2 m 1) n) from by
    rw [gammaOpX, _root_.sub_apply, lp.coeFn_sub, Pi.sub_apply]]
  refine hlim.congr' ?_
  filter_upwards [(tendsto_box k).eventually (eventually_ge_atTop {n, m}),
    hT.eventually (Units.isOpen.mem_nhds hu)] with M hM hU
  have hn : n ∈ box k M := hM (by simp)
  have hm : m ∈ box k M := hM (by simp)
  have hev : ∀ f : ℓ²(ℤ, ℂ), lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 n f = f n := fun _ ↦ rfl
  simp only [Function.comp_apply, ContinuousLinearMap.apply_apply, hev]
  rw [gammaBoxX, dite_eq_left ⟨hn, hm⟩, A.chiBoxX_inv k X hμ hX hU,
    A.inverse_onePlusK0X_single k b hμ hb, A.b_eq_Dhalf k b hμ]
  have hdn := A.Dhalf_ne_zero k hμ n
  have hdm := A.Dhalf_ne_zero k hμ m
  have htn := A.tX_ne_zero k b hμ hb n
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp
  · rw [kron_of_ne hnm]
    field_simp
    ring

end Params

end HubbardAtom
