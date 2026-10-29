import HubbardAtomLean.GammaBoxes

/-!
# The operator vertex for a general `χ` on a lattice

The proofs of items 6(b)–(d) for the density and magnetic channels (`ChiCompact.lean`,
`GammaOperator.lean`, `GammaClosedForm.lean`, `GammaBoxes.lean`) use only the bounds of
item 6(a) on `χ_r` and on the bubble, and of the lattice only `zₙ ≠ 0` and `∑ₙ |zₙ|⁻² < ∞`. This
file repeats them for a general matrix `X` and diagonal `b` on a general lattice (`OpLattice`), so
that the field sectors only need their own bounds; the longitudinal sector, on two spin lattices,
enumerates them as one.

* `OpLattice`: `β > 0` and `zₙ ≠ 0` with `∑ₙ |zₙ|⁻² < ∞`; `Params.opLattice`: the lattice
  `zₙ = μ + i(νₙ + ω_k/2)` for `μ ≠ 0`;
* `OpLattice.ChiDecay`: `|X_nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m` and `|X_nn + β/zₙ²| ≤ c|zₙ|⁻³`;
  `OpLattice.BubbleDecay`: `bₙ ≠ 0` and `zₙ(tₙ - 1)` bounded, with `tₙ = -zₙ²bₙ/β`;
* `OpLattice.X_eq_Dhalf`, `OpLattice.summable_KX_sq`: `X = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)` with
  `𝒟 = diag(-β/zₙ²)` and `𝒦` Hilbert–Schmidt; `OpLattice.b_eq_Dhalf`,
  `OpLattice.isUnit_onePlusK0X`: `b = 𝒟^(1/2)(1 + 𝒦₀)𝒟^(1/2)` with `1 + 𝒦₀ = diag(tₙ)` invertible;
* `OpLattice.gammaOpX`: `Γ = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)`, entrywise;
  `OpLattice.isUnit_onePlusKX_iff`: `1 + 𝒦` is invertible iff `X` has no nonzero null vector `v`
  with `∑ₙ |vₙ/zₙ|² < ∞`; `OpLattice.exists_bound_of_nullX`: such null vectors are bounded;
* `OpLattice.gammaOpX_eq_chiInvX`: `Γ = β²(X⁻¹ - b⁻¹)` with `X⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`
  (`chiInvX`), which, where `1 + 𝒦` is invertible, is a right inverse of `X`
  (`chiInvX_isRightInverse`), the only one with columns in the class (`eq_chiInvX_of_hasSum`), and
  for a symmetric `X` also a left inverse (`chiInvX_isLeftInverse`);
* `OpLattice.gammaOpX_eq_of_isRightInverse`: for a symmetric `X` with a right inverse `Y` whose
  columns are bounded, `1 + 𝒦` is invertible and `Γ = β²(Y - b⁻¹)`;
* `OpLattice.tendsto_gammaBoxX`: wherever `1 + 𝒦` is invertible, the vertices
  `β²[(X|_S)⁻¹ - b⁻¹]` of finite boxes `S` exhausting `ℤ` converge to `Γ` entrywise.

`Params.chiDecay_chiR`: the density and magnetic `χ_r` is an instance.
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

/-- A lattice for the operator vertex: `β > 0` and points `zₙ ≠ 0` with `∑ₙ |zₙ|⁻² < ∞`. -/
structure OpLattice where
  β : ℝ
  β_pos : 0 < β
  z : ℤ → ℂ
  z_ne_zero : ∀ n, z n ≠ 0
  summable_inv_sq : Summable fun n ↦ ‖z n‖⁻¹ ^ 2

/-- The lattice `zₙ = μ + i(νₙ + ω_k/2)` of `Params`, for `μ ≠ 0`. -/
noncomputable def Params.opLattice (A : Params) (k : ℤ) (hμ : A.μ ≠ 0) : OpLattice :=
  ⟨A.β, A.β_pos, A.z k, A.z_ne_zero k hμ, A.summable_inv_norm_z_sq k hμ⟩

namespace OpLattice

open Params (single_apply_eq_kron inverse_apply_of_eq apply_inverse_of_isUnit)

variable (L : OpLattice) (X : ℤ → ℤ → ℂ) (b : ℤ → ℂ)

/-! ### The lattice -/

/-- `1/zₙ → 0` along the cofinite filter. -/
theorem tendsto_inv_z : Tendsto (fun n ↦ (L.z n)⁻¹) cofinite (𝓝 0) := by
  have h : Tendsto (fun n ↦ ‖L.z n‖⁻¹) cofinite (𝓝 0) := by
    simpa [Real.sqrt_sq_eq_abs] using L.summable_inv_sq.tendsto_cofinite_zero.sqrt
  exact tendsto_zero_iff_norm_tendsto_zero.mpr (by simpa using h)

/-- A sequence with `|zₙfₙ|` bounded is square-summable. -/
theorem summable_sq_of_norm_mul_z_le {f : ℤ → ℂ} {C : ℝ} (h : ∀ n, ‖L.z n * f n‖ ≤ C) :
    Summable fun n ↦ ‖f n‖ ^ 2 := by
  refine (L.summable_inv_sq.mul_left (C ^ 2)).of_nonneg_of_le (fun _ ↦ by positivity)
    fun n ↦ ?_
  have hz := norm_pos_iff.mpr (L.z_ne_zero n)
  rw [← mul_pow]
  refine pow_le_pow_left₀ (norm_nonneg _) ?_ 2
  rw [le_mul_inv_iff₀ hz, mul_comm, ← norm_mul]
  exact h n

/-- `∑ₙₘ |zₙ|⁻²(|zₘ|⁻² + δₙₘ)` converges. -/
theorem summable_majorant :
    Summable fun p : ℤ × ℤ ↦
      ‖L.z p.1‖⁻¹ ^ 2 * (‖L.z p.2‖⁻¹ ^ 2 + if p.1 = p.2 then (1 : ℝ) else 0) := by
  have ha := L.summable_inv_sq
  have hδ : Summable fun p : ℤ × ℤ ↦ ‖L.z p.1‖⁻¹ ^ 2 * if p.1 = p.2 then (1 : ℝ) else 0 := by
    refine (summable_diag_sq (c := fun n ↦ (‖L.z n‖⁻¹ : ℂ)) (by simpa using ha)).congr
      fun p ↦ ?_
    by_cases h : p.1 = p.2 <;> simp [diagK, h]
  exact ((ha.mul_of_nonneg ha (fun _ ↦ by positivity) fun _ ↦ by positivity).add hδ).congr
    fun p ↦ by ring

/-- `dₙ = i√β/zₙ`, the diagonal of `𝒟^(1/2)`. -/
noncomputable def Dhalf (n : ℤ) : ℂ := I * (√L.β : ℝ) / L.z n

/-- `dₙ² = -β/zₙ²`. -/
theorem Dhalf_sq (n : ℤ) : L.Dhalf n ^ 2 = -L.β / L.z n ^ 2 := by
  rw [Dhalf, div_pow, mul_pow, I_sq, ← ofReal_pow, Real.sq_sqrt L.β_pos.le]
  ring

/-- `dₙ ≠ 0`. -/
theorem Dhalf_ne_zero (n : ℤ) : L.Dhalf n ≠ 0 := by
  rw [Dhalf]
  exact div_ne_zero (mul_ne_zero I_ne_zero (ofReal_ne_zero.mpr
    (Real.sqrt_pos.mpr L.β_pos).ne')) (L.z_ne_zero n)

/-- For `∑ₙ |vₙ/zₙ|² < ∞`, `𝒟^(1/2)v ∈ ℓ²`. -/
noncomputable def DhalfVec {v : ℤ → ℂ} (hv : Summable fun n ↦ ‖v n / L.z n‖ ^ 2) :
    ℓ²(ℤ, ℂ) :=
  ⟨fun n ↦ L.Dhalf n * v n, memℓp_two_iff.mpr ((hv.mul_left L.β).congr fun n ↦ by
    rw [Dhalf, show I * (√L.β : ℝ) / L.z n * v n = I * (√L.β : ℝ) * (v n / L.z n) by ring,
      norm_mul, norm_mul, norm_I, norm_real, Real.norm_of_nonneg (Real.sqrt_nonneg _), one_mul,
      mul_pow, Real.sq_sqrt L.β_pos.le])⟩

/-- For `w ∈ ℓ²`, `vₙ = wₙ/dₙ` has `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem summable_div_Dhalf (w : ℓ²(ℤ, ℂ)) :
    Summable fun n ↦ ‖w n / L.Dhalf n / L.z n‖ ^ 2 := by
  refine ((memℓp_two_iff.mp (lp.memℓp w)).mul_left L.β⁻¹).congr fun n ↦ ?_
  have hz := L.z_ne_zero n
  rw [Dhalf, norm_div, norm_div, norm_div, norm_mul, norm_I, norm_real,
    Real.norm_of_nonneg (Real.sqrt_nonneg _)]
  field_simp
  rw [Real.sq_sqrt L.β_pos.le]

/-- `𝒟^(1/2)(w/d) = w`. -/
theorem DhalfVec_div (w : ℓ²(ℤ, ℂ)) : L.DhalfVec (L.summable_div_Dhalf w) = w := by
  ext n
  simp [DhalfVec, mul_div_cancel₀ _ (L.Dhalf_ne_zero n)]

/-! ### The bounds -/

/-- The bounds of item 6(a) on a matrix `X`: `|X_nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m`, and
`|X_nn + β/zₙ²| ≤ c|zₙ|⁻³`, i.e. `X_nn = -βzₙ⁻²[1 + O(|zₙ|⁻¹)]`. -/
structure ChiDecay : Prop where
  off : ∃ c, ∀ n m, n ≠ m → ‖X n m‖ ≤ c / (‖L.z n‖ ^ 2 * ‖L.z m‖ ^ 2)
  diag : ∃ c, ∀ n, ‖X n n + L.β / L.z n ^ 2‖ ≤ c / ‖L.z n‖ ^ 3

/-- `tₙ = -zₙ²bₙ/β`, the diagonal of `1 + 𝒦₀` for the bubble `b`. -/
noncomputable def tX (n : ℤ) : ℂ := -(L.z n ^ 2 / L.β) * b n

/-- The bubble `b` decays as `-β/zₙ²`: `bₙ ≠ 0` and `|zₙ(tₙ - 1)| ≤ C`. -/
structure BubbleDecay : Prop where
  ne : ∀ n, b n ≠ 0
  bdd : ∃ C, ∀ n, ‖L.z n * (L.tX b n - 1)‖ ≤ C

/-! ### `X = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)` with `𝒦` Hilbert–Schmidt -/

/-- `𝒦_nm = -(zₙzₘ/β)X_nm - δ_nm`. -/
noncomputable def KX (n m : ℤ) : ℂ := -(L.z n * L.z m / L.β) * X n m - kron n m

/-- `X = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)`, entrywise. -/
theorem X_eq_Dhalf (n m : ℤ) :
    X n m = L.Dhalf n * (kron n m + L.KX X n m) * L.Dhalf m := by
  have hz := L.z_ne_zero n
  have hv := L.z_ne_zero m
  have hβ : (L.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr L.β_pos.ne'
  have hs : ((√L.β : ℝ) : ℂ) ^ 2 = L.β := by rw [← ofReal_pow, Real.sq_sqrt L.β_pos.le]
  have hd : L.Dhalf n * L.Dhalf m = -L.β / (L.z n * L.z m) := by
    rw [Dhalf, Dhalf, div_mul_div_comm, mul_mul_mul_comm, I_mul_I, ← sq, hs]
    ring
  rw [KX, add_sub_cancel, show ∀ x y c : ℂ, x * c * y = x * y * c from fun _ _ _ ↦ by ring, hd]
  field_simp

/-- `|𝒦_nm|² ≤ ((c₁/β)² + (c₂/β)²)|zₙ|⁻²(|zₘ|⁻² + δ_nm)`, from the bounds of item 6(a) with the
constants `c₁`, `c₂`. -/
theorem norm_KX_sq_le_of {c₁ c₂ : ℝ}
    (hc₁ : ∀ n m, n ≠ m → ‖X n m‖ ≤ c₁ / (‖L.z n‖ ^ 2 * ‖L.z m‖ ^ 2))
    (hc₂ : ∀ n, ‖X n n + L.β / L.z n ^ 2‖ ≤ c₂ / ‖L.z n‖ ^ 3) (n m : ℤ) :
    ‖L.KX X n m‖ ^ 2 ≤ ((c₁ / L.β) ^ 2 + (c₂ / L.β) ^ 2) *
      (‖L.z n‖⁻¹ ^ 2 * (‖L.z m‖⁻¹ ^ 2 + if n = m then 1 else 0)) := by
  have hβ := L.β_pos
  have hz := norm_pos_iff.mpr (L.z_ne_zero n)
  have hv := norm_pos_iff.mpr (L.z_ne_zero m)
  by_cases hnm : n = m
  · subst hnm
    -- `|zₙ𝒦_nn| = |zₙ|³|X_nn + β/zₙ²|/β ≤ c₂/β`
    have h : L.z n * L.KX X n n =
        -(L.z n ^ 3 / L.β) * (X n n + L.β / L.z n ^ 2) := by
      have := L.z_ne_zero n
      have hβ' : (L.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr hβ.ne'
      rw [KX, kron_self]
      field_simp
      ring
    have hb : ‖L.z n * L.KX X n n‖ ≤ c₂ / L.β := by
      rw [h, norm_mul, norm_neg, norm_div, norm_pow, norm_real, Real.norm_of_nonneg hβ.le,
        div_mul_eq_mul_div, div_le_div_iff_of_pos_right hβ, mul_comm,
        ← le_div_iff₀ (by positivity)]
      exact hc₂ n
    have hK : ‖L.KX X n n‖ ≤ c₂ / L.β * ‖L.z n‖⁻¹ := by
      rw [le_mul_inv_iff₀ hz, mul_comm, ← norm_mul]
      exact hb
    calc ‖L.KX X n n‖ ^ 2 ≤ (c₂ / L.β * ‖L.z n‖⁻¹) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hK 2
      _ ≤ _ := by
        simp only [ite_true]
        nlinarith [sq_nonneg (c₁ / L.β), sq_nonneg (c₂ / L.β), sq_nonneg ‖L.z n‖⁻¹,
          mul_nonneg (sq_nonneg (c₁ / L.β + c₂ / L.β)) (pow_nonneg (sq_nonneg ‖L.z n‖⁻¹) 2)]
  · have hK : ‖L.KX X n m‖ ≤ c₁ / L.β * ‖L.z n‖⁻¹ * ‖L.z m‖⁻¹ := by
      rw [KX, kron_of_ne hnm, sub_zero, norm_mul, norm_neg, norm_div, norm_mul, norm_real,
        Real.norm_of_nonneg hβ.le]
      calc ‖L.z n‖ * ‖L.z m‖ / L.β * ‖X n m‖
          ≤ ‖L.z n‖ * ‖L.z m‖ / L.β * (c₁ / (‖L.z n‖ ^ 2 * ‖L.z m‖ ^ 2)) :=
            mul_le_mul_of_nonneg_left (hc₁ n m hnm) (by positivity)
        _ = c₁ / L.β * ‖L.z n‖⁻¹ * ‖L.z m‖⁻¹ := by field_simp
    calc ‖L.KX X n m‖ ^ 2 ≤ (c₁ / L.β * ‖L.z n‖⁻¹ * ‖L.z m‖⁻¹) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hK 2
      _ ≤ _ := by
        simp only [hnm, ite_false, add_zero]
        nlinarith [mul_nonneg (sq_nonneg (c₂ / L.β))
          (mul_nonneg (sq_nonneg ‖L.z n‖⁻¹) (sq_nonneg ‖L.z m‖⁻¹))]

/-- `|𝒦_nm|² ≤ C|zₙ|⁻²(|zₘ|⁻² + δ_nm)`, by the bounds on `X`. -/
theorem norm_KX_sq_le (hX : L.ChiDecay X) : ∃ C, ∀ n m, ‖L.KX X n m‖ ^ 2 ≤
    C * (‖L.z n‖⁻¹ ^ 2 * (‖L.z m‖⁻¹ ^ 2 + if n = m then 1 else 0)) := by
  obtain ⟨c₁, hc₁⟩ := hX.off
  obtain ⟨c₂, hc₂⟩ := hX.diag
  exact ⟨_, L.norm_KX_sq_le_of X hc₁ hc₂⟩

/-- **`𝒦` is Hilbert–Schmidt**, by `norm_KX_sq_le`. -/
theorem summable_KX_sq (hX : L.ChiDecay X) :
    Summable fun p : ℤ × ℤ ↦ ‖L.KX X p.1 p.2‖ ^ 2 := by
  obtain ⟨C, hC⟩ := L.norm_KX_sq_le X hX
  exact ((L.summable_majorant).mul_left C).of_nonneg_of_le (fun _ ↦ by positivity)
    fun p ↦ (hC p.1 p.2).trans_eq (by ring)

/-- **`𝒦` is compact** on `ℓ²(ℤ)`. -/
theorem isCompactOperator_KX (hX : L.ChiDecay X) :
    IsCompactOperator (hsOp (L.KX X) (L.summable_KX_sq X hX)) :=
  isCompactOperator_hsOp _ _

/-- The rows of `X` are absolutely summable against a bounded vector. -/
theorem summable_norm_X_mul (hX : L.ChiDecay X) {v : ℤ → ℂ} {C : ℝ} (hv : ∀ m, ‖v m‖ ≤ C)
    (n : ℤ) : Summable fun m ↦ ‖X n m * v m‖ := by
  obtain ⟨c, hc⟩ := hX.off
  refine (((L.summable_inv_sq.mul_left (|c| / ‖L.z n‖ ^ 2 * |C|)).add
    (hasSum_ite_eq n ‖X n n * v n‖).summable)).of_nonneg_of_le (fun _ ↦ norm_nonneg _)
    fun m ↦ ?_
  by_cases hnm : n = m
  · subst hnm
    rw [ite_eq_left rfl]
    have : 0 ≤ |c| / ‖L.z n‖ ^ 2 * |C| * ‖L.z n‖⁻¹ ^ 2 := by positivity
    linarith
  · have hz := norm_pos_iff.mpr (L.z_ne_zero n)
    have hm := norm_pos_iff.mpr (L.z_ne_zero m)
    rw [ite_eq_right (Ne.symm hnm), add_zero, norm_mul]
    calc ‖X n m‖ * ‖v m‖ ≤ |c| / (‖L.z n‖ ^ 2 * ‖L.z m‖ ^ 2) * |C| :=
          mul_le_mul ((hc n m hnm).trans (div_le_div_of_nonneg_right (le_abs_self c)
            (by positivity))) ((hv m).trans (le_abs_self C)) (norm_nonneg _) (by positivity)
      _ = |c| / ‖L.z n‖ ^ 2 * |C| * ‖L.z m‖⁻¹ ^ 2 := by field_simp

/-- `𝒦` is symmetric where `X` is. -/
theorem KX_symm (hs : ∀ n m, X n m = X m n) (n m : ℤ) : L.KX X n m = L.KX X m n := by
  rw [KX, KX, hs, kron_comm, mul_comm (L.z n)]

/-! ### The bubble -/

/-- `tₙ ≠ 0`. -/
theorem tX_ne_zero (hb : L.BubbleDecay b) (n : ℤ) : L.tX b n ≠ 0 := by
  have hβ : (L.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr L.β_pos.ne'
  exact mul_ne_zero (neg_ne_zero.mpr (div_ne_zero (pow_ne_zero 2 (L.z_ne_zero n)) hβ))
    (hb.ne n)

/-- `b = 𝒟^(1/2)(1 + 𝒦₀)𝒟^(1/2)`: `bₙ = dₙ²tₙ`. -/
theorem b_eq_Dhalf (n : ℤ) : b n = L.Dhalf n ^ 2 * L.tX b n := by
  have hz := L.z_ne_zero n
  have hβ : (L.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr L.β_pos.ne'
  rw [L.Dhalf_sq, tX]
  field_simp

/-- `tₙ → 1`. -/
theorem tendsto_tX (hb : L.BubbleDecay b) :
    Tendsto (L.tX b) cofinite (𝓝 1) := by
  obtain ⟨C, hC⟩ := hb.bdd
  rw [tendsto_iff_norm_sub_tendsto_zero]
  have h0 : Tendsto (fun n ↦ C * ‖(L.z n)⁻¹‖) cofinite (𝓝 0) := by
    simpa using (L.tendsto_inv_z).norm.const_mul C
  refine squeeze_zero (fun _ ↦ norm_nonneg _) (fun n ↦ ?_) h0
  have hz := L.z_ne_zero n
  calc ‖L.tX b n - 1‖ = ‖(L.z n)⁻¹‖ * ‖L.z n * (L.tX b n - 1)‖ := by
        rw [← norm_mul, inv_mul_cancel_left₀ hz]
    _ ≤ ‖(L.z n)⁻¹‖ * C := mul_le_mul_of_nonneg_left (hC n) (norm_nonneg _)
    _ = C * ‖(L.z n)⁻¹‖ := mul_comm _ _

/-- `tₙ - 1` is square-summable. -/
theorem summable_tX_sub_one_sq (hb : L.BubbleDecay b) :
    Summable fun n ↦ ‖L.tX b n - 1‖ ^ 2 := by
  obtain ⟨C, hC⟩ := hb.bdd
  exact L.summable_sq_of_norm_mul_z_le hC

/-- `1/tₙ - 1` is square-summable, as `tₙ → 1` and `tₙ ≠ 0` keep `1/tₙ` bounded. -/
theorem summable_inv_tX_sub_one_sq (hb : L.BubbleDecay b) :
    Summable fun n ↦ ‖(L.tX b n)⁻¹ - 1‖ ^ 2 := by
  obtain ⟨C, hC⟩ := hb.bdd
  obtain ⟨D, hD⟩ :=
    ((L.tendsto_tX b hb).inv₀ one_ne_zero).norm.bddAbove_range_of_cofinite
  refine L.summable_sq_of_norm_mul_z_le (C := D * C) fun n ↦ ?_
  have ht := L.tX_ne_zero b hb n
  have hDn : ‖(L.tX b n)⁻¹‖ ≤ D := hD ⟨n, rfl⟩
  rw [show L.z n * ((L.tX b n)⁻¹ - 1) =
      -((L.tX b n)⁻¹ * (L.z n * (L.tX b n - 1))) by field_simp; ring, norm_neg, norm_mul]
  exact mul_le_mul hDn (hC n) (norm_nonneg _) ((norm_nonneg _).trans hDn)

/-- `1 + 𝒦₀ = diag(tₙ)` on `ℓ²(ℤ)`. -/
noncomputable def onePlusK0X (hb : L.BubbleDecay b) :
    ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (diagK fun n ↦ L.tX b n - 1) (summable_diag_sq (L.summable_tX_sub_one_sq b hb))

/-- **`1 + 𝒦₀` is invertible** on `ℓ²(ℤ)`, with the inverse `1 + diag(1/tₙ - 1)`. -/
theorem isUnit_onePlusK0X (hb : L.BubbleDecay b) :
    IsUnit (L.onePlusK0X b hb) := by
  refine ⟨⟨_, 1 + hsOp (diagK fun n ↦ (L.tX b n)⁻¹ - 1)
    (summable_diag_sq (L.summable_inv_tX_sub_one_sq b hb)), ?_, ?_⟩, rfl⟩ <;>
  · ext v n
    have ht := L.tX_ne_zero b hb n
    simp [hsOp_diag_apply (L.summable_tX_sub_one_sq b hb),
      hsOp_diag_apply (L.summable_inv_tX_sub_one_sq b hb)]
    field_simp
    ring

/-- `(1 + 𝒦₀)(eₘ/tₘ) = eₘ`. -/
theorem onePlusK0X_single (hb : L.BubbleDecay b) (m : ℤ) :
    L.onePlusK0X b hb ((L.tX b m)⁻¹ • lp.single 2 m 1) = lp.single 2 m 1 := by
  ext n
  simp only [onePlusK0X, _root_.add_apply, one_apply_eq_self, lp.coeFn_add, Pi.add_apply,
    hsOp_diag_apply (L.summable_tX_sub_one_sq b hb), lp.coeFn_smul, Pi.smul_apply,
    smul_eq_mul, single_apply_eq_kron]
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp [L.tX_ne_zero b hb n]
    ring
  · simp [kron_of_ne hnm]

/-- `[(1 + 𝒦₀)⁻¹eₘ]ₙ = δₙₘ/tₘ`. -/
theorem inverse_onePlusK0X_single (hb : L.BubbleDecay b) (n m : ℤ) :
    Ring.inverse (L.onePlusK0X b hb) (lp.single 2 m 1) n = kron n m / L.tX b m := by
  rw [inverse_apply_of_eq (L.isUnit_onePlusK0X b hb) (L.onePlusK0X_single b hb m),
    lp.coeFn_smul, Pi.smul_apply, smul_eq_mul, single_apply_eq_kron, div_eq_inv_mul]

/-! ### The operator `Γ` and the criterion -/

/-- `1 + 𝒦` on `ℓ²(ℤ)`. -/
noncomputable def onePlusKX (hX : L.ChiDecay X) : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (L.KX X) (L.summable_KX_sq X hX)

/-- `Γ_nm = β²/(dₙdₘ) [(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]ₙₘ`, the entries taken on the unit vectors, with
`Ring.inverse` (zero where `1 + 𝒦` is not invertible). -/
noncomputable def gammaOpX (hX : L.ChiDecay X) (hb : L.BubbleDecay b)
    (n m : ℤ) : ℂ :=
  L.β ^ 2 / (L.Dhalf n * L.Dhalf m) *
    ((Ring.inverse (L.onePlusKX X hX) - Ring.inverse (L.onePlusK0X b hb))
      (lp.single 2 m 1) n)

/-- **Fredholm alternative**: `1 + 𝒦` is invertible iff it is injective. -/
theorem isUnit_onePlusKX_iff_injective (hX : L.ChiDecay X) :
    IsUnit (L.onePlusKX X hX) ↔ Function.Injective (L.onePlusKX X hX) := by
  refine ⟨fun h ↦ (ContinuousLinearMap.isUnit_iff_bijective.mp h).1, fun hinj ↦ ?_⟩
  rcases (L.isCompactOperator_KX X hX).hasEigenvalue_or_mem_resolventSet (μ := -1)
    (by norm_num) with hev | hres
  · obtain ⟨x, hx⟩ := hev.exists_hasEigenvector
    refine absurd (hinj ?_) hx.2
    have h := hx.apply_eq_smul
    simp only [ContinuousLinearMap.coe_coe] at h
    simp [onePlusKX, h]
  · rw [spectrum.mem_resolventSet_iff, Algebra.algebraMap_eq_smul_one,
      show (-1 : ℂ) • (1 : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ)) - hsOp (L.KX X) (L.summable_KX_sq X hX) =
        -L.onePlusKX X hX by rw [onePlusKX]; module, IsUnit.neg_iff] at hres
    exact hres

/-- `X v = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)v`, row by row, for `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem hasSum_X_mul (hX : L.ChiDecay X) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / L.z n‖ ^ 2) (n : ℤ) :
    HasSum (fun m ↦ X n m * v m) (L.Dhalf n * L.onePlusKX X hX (L.DhalfVec hv) n) := by
  have h := ((hasSum_ite_eq n (L.DhalfVec hv n)).add
    (hasSum_hsOp (L.KX X) (L.summable_KX_sq X hX) (L.DhalfVec hv) n)).mul_left
    (L.Dhalf n)
  convert h using 1
  · funext m
    rw [L.X_eq_Dhalf X]
    by_cases hnm : n = m
    · subst hnm
      simp [kron_self, DhalfVec]
      ring
    · simp [kron_of_ne hnm, Ne.symm hnm, DhalfVec]
      ring
  · simp [onePlusKX]

/-- A null vector of `X` with `∑ₙ |vₙ/zₙ|² < ∞` gives `(1 + 𝒦)𝒟^(1/2)v = 0`. -/
theorem onePlusKX_DhalfVec_eq_zero (hX : L.ChiDecay X) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / L.z n‖ ^ 2) (hχ : ∀ n, HasSum (fun m ↦ X n m * v m) 0) :
    L.onePlusKX X hX (L.DhalfVec hv) = 0 := by
  ext n
  have h := (L.hasSum_X_mul X hX hv n).unique (hχ n)
  simpa [L.Dhalf_ne_zero n] using h

/-- **The criterion**: `ker(1 + 𝒦) = 0` iff `X` has no nonzero null vector `v` with
`∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem injective_onePlusKX_iff (hX : L.ChiDecay X) :
    Function.Injective (L.onePlusKX X hX) ↔ ∀ v : ℤ → ℂ,
      Summable (fun n ↦ ‖v n / L.z n‖ ^ 2) →
        (∀ n, HasSum (fun m ↦ X n m * v m) 0) → v = 0 := by
  have hd := L.Dhalf_ne_zero
  constructor
  · intro hinj v hv hχ
    have h := hinj ((L.onePlusKX_DhalfVec_eq_zero X hX hv hχ).trans (map_zero _).symm)
    funext n
    have hn := congrArg (fun w : ℓ²(ℤ, ℂ) ↦ w n) h
    simpa [DhalfVec, hd n] using hn
  · intro h
    rw [injective_iff_map_eq_zero]
    intro w hw
    have h0 := h _ (L.summable_div_Dhalf w) fun n ↦ by
      simpa [L.DhalfVec_div, hw] using
        L.hasSum_X_mul X hX (L.summable_div_Dhalf w) n
    ext n
    have hn := congrFun h0 n
    simp only [Pi.zero_apply, div_eq_zero_iff, hd n, or_false] at hn
    simp [hn]

/-- **`Γ` exists iff `X` has no null vector**: `1 + 𝒦` is invertible iff `X` has no nonzero null
vector `v` with `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem isUnit_onePlusKX_iff (hX : L.ChiDecay X) :
    IsUnit (L.onePlusKX X hX) ↔ ∀ v : ℤ → ℂ, Summable (fun n ↦ ‖v n / L.z n‖ ^ 2) →
      (∀ n, HasSum (fun m ↦ X n m * v m) 0) → v = 0 :=
  (L.isUnit_onePlusKX_iff_injective X hX).trans (L.injective_onePlusKX_iff X hX)

/-- **Null vectors are bounded**: a null vector `v` of `X` with `∑ₙ |vₙ/zₙ|² < ∞` has
`|vₙ| ≤ C`, as `w = 𝒟^(1/2)v = -𝒦w` gives `|wₙ| ≤ ‖𝒦ₙ·‖‖w‖ = O(|zₙ|⁻¹)`. -/
theorem exists_bound_of_nullX (hX : L.ChiDecay X) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / L.z n‖ ^ 2) (hχ : ∀ n, HasSum (fun m ↦ X n m * v m) 0) :
    ∃ C, ∀ n, ‖v n‖ ≤ C := by
  obtain ⟨C, hC⟩ := L.norm_KX_sq_le X hX
  have ha := L.summable_inv_sq
  have hβ := L.β_pos
  set w := L.DhalfVec hv with hwdef
  set S := ∑' m, ‖L.z m‖⁻¹ ^ 2
  refine ⟨√(C * (S + 1) * ‖w‖ ^ 2 / L.β), fun n ↦ ?_⟩
  have hz := norm_pos_iff.mpr (L.z_ne_zero n)
  -- `wₙ = -⟪𝒦ₙ·, w⟫`
  have hw : w n = -inner ℂ (hsRow (L.KX X) (L.summable_KX_sq X hX) n) w := by
    have h := congrArg (fun x : ℓ²(ℤ, ℂ) ↦ x n) (L.onePlusKX_DhalfVec_eq_zero X hX hv hχ)
    rw [← hwdef] at h
    simp only [onePlusKX, _root_.add_apply, one_apply_eq_self, lp.coeFn_add, Pi.add_apply,
      hsOp_apply_eq_inner, lp.coeFn_zero, Pi.zero_apply] at h
    linear_combination h
  -- `‖𝒦ₙ·‖² ≤ C|zₙ|⁻²(S + 1)`
  have hrow : ‖hsRow (L.KX X) (L.summable_KX_sq X hX) n‖ ^ 2 ≤
      C * (‖L.z n‖⁻¹ ^ 2 * (S + 1)) := by
    have hδ : HasSum (fun m ↦ if n = m then (1 : ℝ) else 0) 1 := by
      simpa [eq_comm] using hasSum_ite_eq n (1 : ℝ)
    have hs : Summable fun m ↦ C * (‖L.z n‖⁻¹ ^ 2 * (‖L.z m‖⁻¹ ^ 2 +
        if n = m then 1 else 0)) := ((ha.add hδ.summable).mul_left _).mul_left _
    refine (lp_norm_sq_le hs fun m ↦ ?_).trans_eq ?_
    · simpa [hsRow] using hC n m
    · rw [tsum_mul_left, tsum_mul_left, ha.tsum_add hδ.summable, hδ.tsum_eq]
  have hv' : ‖v n‖ ^ 2 = ‖L.z n‖ ^ 2 * ‖w n‖ ^ 2 / L.β := by
    have hwn : w n = L.Dhalf n * v n := rfl
    rw [hwn, Dhalf, norm_mul, norm_div, norm_mul, norm_I, norm_real,
      Real.norm_of_nonneg (Real.sqrt_nonneg _)]
    field_simp
    rw [Real.sq_sqrt hβ.le]
  rw [← abs_norm]
  refine Real.abs_le_sqrt ?_
  rw [hv', hw, norm_neg]
  have hCS := norm_inner_sq_le (hsRow (L.KX X) (L.summable_KX_sq X hX) n) w
  calc ‖L.z n‖ ^ 2 * ‖inner ℂ (hsRow (L.KX X) (L.summable_KX_sq X hX) n) w‖ ^ 2 / L.β
      ≤ ‖L.z n‖ ^ 2 * (C * (‖L.z n‖⁻¹ ^ 2 * (S + 1)) * ‖w‖ ^ 2) / L.β := by
        gcongr
        exact hCS.trans (mul_le_mul_of_nonneg_right hrow (by positivity))
    _ = C * (S + 1) * ‖w‖ ^ 2 / L.β := by
        field_simp

/-! ### `Γ = β²(X⁻¹ - b⁻¹)` -/

/-- `cₘ = (1 + 𝒦)⁻¹eₘ/dₘ ∈ ℓ²`, so `𝒟^(-1/2)cₘ` is column `m` of `X⁻¹` (`chiInvX_eq`). -/
noncomputable def invColKX (hX : L.ChiDecay X) (m : ℤ) : ℓ²(ℤ, ℂ) :=
  (L.Dhalf m)⁻¹ • Ring.inverse (L.onePlusKX X hX) (lp.single 2 m 1)

/-- `X⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, entrywise, with `Ring.inverse` as in `gammaOpX`. -/
noncomputable def chiInvX (hX : L.ChiDecay X) (n m : ℤ) : ℂ :=
  Ring.inverse (L.onePlusKX X hX) (lp.single 2 m 1) n / (L.Dhalf n * L.Dhalf m)

/-- `X⁻¹_nm = cₘ,ₙ/dₙ`. -/
theorem chiInvX_eq (hX : L.ChiDecay X) (n m : ℤ) :
    L.chiInvX X hX n m = L.invColKX X hX m n / L.Dhalf n := by
  rw [chiInvX, invColKX, lp.coeFn_smul, Pi.smul_apply, smul_eq_mul]
  field_simp

/-- **`Γ = β²(X⁻¹ - b⁻¹)`**, entrywise. Where `1 + 𝒦` is not invertible, `X⁻¹ = 0`
(`Ring.inverse`), so `Γ = -β²δₙₘ/bₙ` there. -/
theorem gammaOpX_eq_chiInvX (hX : L.ChiDecay X) (hb : L.BubbleDecay b)
    (n m : ℤ) :
    L.gammaOpX X b hX hb n m = L.β ^ 2 * (L.chiInvX X hX n m - kron n m / b n) := by
  rw [gammaOpX, chiInvX, _root_.sub_apply, lp.coeFn_sub, Pi.sub_apply,
    L.inverse_onePlusK0X_single b hb, L.b_eq_Dhalf b]
  have hdn := L.Dhalf_ne_zero n
  have hdm := L.Dhalf_ne_zero m
  have htn := L.tX_ne_zero b hb n
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp
  · rw [kron_of_ne hnm]
    field_simp
    ring

/-- The columns of `X⁻¹` are in the class `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem summable_chiInvX (hX : L.ChiDecay X) (m : ℤ) :
    Summable fun n ↦ ‖L.chiInvX X hX n m / L.z n‖ ^ 2 := by
  simpa only [L.chiInvX_eq X hX] using L.summable_div_Dhalf (L.invColKX X hX m)

/-- **`X⁻¹` is a right inverse** where `1 + 𝒦` is invertible: `∑ₗ X_nl X⁻¹_lm = δₙₘ`. -/
theorem chiInvX_isRightInverse (hX : L.ChiDecay X)
    (hK : IsUnit (L.onePlusKX X hX)) : IsRightInverse X (L.chiInvX X hX) := by
  intro n m
  have h := L.hasSum_X_mul X hX (L.summable_div_Dhalf (L.invColKX X hX m)) n
  have hT : L.onePlusKX X hX (L.invColKX X hX m) =
      (L.Dhalf m)⁻¹ • lp.single 2 m 1 := by
    rw [invColKX, map_smul, apply_inverse_of_isUnit hK]
  rw [L.DhalfVec_div, hT, lp.coeFn_smul, Pi.smul_apply, smul_eq_mul,
    single_apply_eq_kron] at h
  convert h using 1
  · funext l
    rw [L.chiInvX_eq X hX]
  by_cases hnm : n = m
  · subst hnm
    field_simp [L.Dhalf_ne_zero n]
  · simp [kron_of_ne hnm]

/-- **`X⁻¹` is the only right inverse in the class**: where `1 + 𝒦` is invertible, a `v` with
`∑ₙ |vₙ/zₙ|² < ∞` and `∑ₗ X_nl vₗ = δₙₘ` is column `m` of `X⁻¹`. -/
theorem eq_chiInvX_of_hasSum (hX : L.ChiDecay X)
    (hK : IsUnit (L.onePlusKX X hX)) {m : ℤ} {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / L.z n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun l ↦ X n l * v l) (kron n m)) (n : ℤ) :
    v n = L.chiInvX X hX n m := by
  have hT : L.onePlusKX X hX (L.DhalfVec hv) = (L.Dhalf m)⁻¹ • lp.single 2 m 1 := by
    ext l
    have h := (L.hasSum_X_mul X hX hv l).unique (hχ l)
    have hd := L.Dhalf_ne_zero l
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
  rw [L.chiInvX_eq X hX, invColKX, hc]
  simp [DhalfVec, L.Dhalf_ne_zero n]

/-- **`X⁻¹` is symmetric** where `1 + 𝒦` is invertible and `X` is symmetric. -/
theorem chiInvX_symm (hX : L.ChiDecay X) (hs : ∀ n m, X n m = X m n)
    (hK : IsUnit (L.onePlusKX X hX)) (n m : ℤ) :
    L.chiInvX X hX n m = L.chiInvX X hX m n := by
  have hpair := tsum_mul_one_add_hsOp_comm (L.KX X) (L.summable_KX_sq X hX)
    (L.KX_symm X hs) (Ring.inverse (L.onePlusKX X hX) (lp.single 2 m 1))
    (Ring.inverse (L.onePlusKX X hX) (lp.single 2 n 1))
  rw [show (1 + hsOp (L.KX X) (L.summable_KX_sq X hX)) = L.onePlusKX X hX from rfl,
    apply_inverse_of_isUnit hK, apply_inverse_of_isUnit hK] at hpair
  simp only [single_apply_eq_kron, kron, mul_ite, mul_one, mul_zero, tsum_ite_eq] at hpair
  rw [chiInvX, chiInvX, hpair, mul_comm (L.Dhalf n)]

/-- **`X⁻¹` is a left inverse** where `1 + 𝒦` is invertible and `X` is symmetric. -/
theorem chiInvX_isLeftInverse (hX : L.ChiDecay X) (hs : ∀ n m, X n m = X m n)
    (hK : IsUnit (L.onePlusKX X hX)) : IsLeftInverse X (L.chiInvX X hX) := by
  intro n m
  convert L.chiInvX_isRightInverse X hX hK m n using 1
  · funext l
    rw [L.chiInvX_symm X hX hs hK n l, hs l m, mul_comm]
  · exact kron_comm n m

/-! ### A right inverse with bounded columns -/

/-- A column of `Y` with `|Y_nm| ≤ C` is in the class `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem summable_col_of_bdd {Y : ℤ → ℤ → ℂ} {m : ℤ} {C : ℝ}
    (hC : ∀ n, ‖Y n m‖ ≤ C) (c : ℂ) : Summable fun n ↦ ‖Y n m * c / L.z n‖ ^ 2 := by
  refine L.summable_sq_of_norm_mul_z_le (C := C * ‖c‖) fun n ↦ ?_
  rw [mul_div_cancel₀ _ (L.z_ne_zero n), norm_mul]
  exact mul_le_mul_of_nonneg_right (hC n) (norm_nonneg _)

/-- **`1 + 𝒦` is invertible** where the symmetric `X` has a right inverse `Y` with bounded
columns: `cₘ = 𝒟^(1/2)(Y eₘ)dₘ` has `(1 + 𝒦)cₘ = eₘ`, and for `(1 + 𝒦)x = 0`,
`xₘ = ∑ₙ xₙ((1 + 𝒦)cₘ)ₙ = ∑ₙ cₘ,ₙ((1 + 𝒦)x)ₙ = 0`. -/
theorem isUnit_onePlusKX_of_isRightInverse (hX : L.ChiDecay X)
    (hs : ∀ n m, X n m = X m n) {Y : ℤ → ℤ → ℂ} (hY : IsRightInverse X Y)
    (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) : IsUnit (L.onePlusKX X hX) := by
  have hcol : ∀ m, ∃ c : ℓ²(ℤ, ℂ), L.onePlusKX X hX c = lp.single 2 m 1 := fun m ↦ by
    obtain ⟨C, hC⟩ := hbdd m
    have hv := L.summable_col_of_bdd hC (L.Dhalf m)
    refine ⟨L.DhalfVec hv, ?_⟩
    ext n
    have h₁ := L.hasSum_X_mul X hX hv n
    have h₂ := (hY n m).mul_right (L.Dhalf m)
    simp only [mul_assoc] at h₂
    have h₃ := h₁.unique h₂
    have hd := L.Dhalf_ne_zero n
    rw [single_apply_eq_kron]
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self, one_mul] at h₃
      rw [kron_self]
      exact mul_left_cancel₀ hd (h₃.trans (mul_one _).symm)
    · rw [kron_of_ne hnm, zero_mul] at h₃
      rw [kron_of_ne hnm]
      exact (mul_eq_zero.mp h₃).resolve_left hd
  refine (L.isUnit_onePlusKX_iff_injective X hX).mpr ?_
  rw [injective_iff_map_eq_zero]
  intro x hx
  ext m
  obtain ⟨c, hc⟩ := hcol m
  have hpair := tsum_mul_one_add_hsOp_comm (L.KX X) (L.summable_KX_sq X hX)
    (L.KX_symm X hs) x c
  rw [show (1 + hsOp (L.KX X) (L.summable_KX_sq X hX)) = L.onePlusKX X hX from rfl,
    hc, hx] at hpair
  simp only [single_apply_eq_kron, lp.coeFn_zero, Pi.zero_apply, mul_zero, tsum_zero, kron,
    mul_ite, mul_one] at hpair
  rw [tsum_ite_eq m] at hpair
  simpa using hpair

/-- **The right inverse is `X⁻¹`**: for a symmetric `X` with a right inverse `Y` whose columns
are bounded, `Y = X⁻¹`. -/
theorem eq_chiInvX_of_isRightInverse (hX : L.ChiDecay X)
    (hs : ∀ n m, X n m = X m n) {Y : ℤ → ℤ → ℂ} (hY : IsRightInverse X Y)
    (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) (n m : ℤ) :
    Y n m = L.chiInvX X hX n m := by
  obtain ⟨C, hC⟩ := hbdd m
  have hv : Summable fun n ↦ ‖Y n m / L.z n‖ ^ 2 := by
    simpa using L.summable_col_of_bdd hC 1
  exact L.eq_chiInvX_of_hasSum X hX
    (L.isUnit_onePlusKX_of_isRightInverse X hX hs hY hbdd) hv (fun n ↦ hY n m) n

/-- **The operator vertex is the closed form**: for a symmetric `X` with a right inverse `Y`
whose columns are bounded, `Γ = β²(Y - b⁻¹)`. -/
theorem gammaOpX_eq_of_isRightInverse (hX : L.ChiDecay X)
    (hb : L.BubbleDecay b) (hs : ∀ n m, X n m = X m n) {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse X Y) (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) (n m : ℤ) :
    L.gammaOpX X b hX hb n m = L.β ^ 2 * (Y n m - kron n m / b n) := by
  rw [L.gammaOpX_eq_chiInvX X b hX hb,
    L.eq_chiInvX_of_isRightInverse X hX hs hY hbdd n m]

/-! ### Finite boxes -/

/-- `1 + 𝒦_s`, the compression of `1 + 𝒦` to `s × s`. -/
noncomputable def onePlusKX_box (hX : L.ChiDecay X) (s : Finset ℤ) :
    ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (truncK (L.KX X) s) (summable_truncK_sq _ (L.summable_KX_sq X hX) s)

/-- **The compressions converge in norm**: `1 + 𝒦_M → 1 + 𝒦`. -/
theorem tendsto_onePlusKX_box (hX : L.ChiDecay X) {S : ℕ → Finset ℤ}
    (hS : Tendsto S atTop atTop) :
    Tendsto (fun M ↦ L.onePlusKX_box X hX (S M)) atTop (𝓝 (L.onePlusKX X hX)) :=
  ((tendsto_hsOp_truncK _ (L.summable_KX_sq X hX)).comp hS).const_add 1

/-- The box matrix `X|_s`. -/
noncomputable def chiBoxX (s : Finset ℤ) : Matrix s s ℂ := Matrix.of fun i j ↦ X i j

/-- Where `1 + 𝒦_s` is invertible, `X|_s` has the right inverse `𝒟^(-1/2)(1 + 𝒦_s)⁻¹𝒟^(-1/2)`. -/
theorem chiBoxX_mul (hX : L.ChiDecay X) {s : Finset ℤ}
    (hu : IsUnit (L.onePlusKX_box X hX s)) :
    chiBoxX X s * Matrix.of (fun i j : s ↦
      Ring.inverse (L.onePlusKX_box X hX s) (lp.single 2 (j : ℤ) 1) i /
        (L.Dhalf i * L.Dhalf j)) = 1 := by
  set T := L.onePlusKX_box X hX s
  set y : ℤ → ℓ²(ℤ, ℂ) := fun m ↦ Ring.inverse T (lp.single 2 m 1)
  have hy : ∀ m n, n ∈ s → y m n + ∑ l ∈ s, L.KX X n l * y m l = kron n m := fun m n hn ↦ by
    have hT : T (y m) = lp.single 2 m 1 := apply_inverse_of_isUnit hu _
    have h := congrArg (fun w : ℓ²(ℤ, ℂ) ↦ w n) hT
    simp only [T, onePlusKX_box, _root_.add_apply, one_apply_eq_self, lp.coeFn_add,
      Pi.add_apply] at h
    rw [hsOp_truncK_apply _ (L.summable_KX_sq X hX), single_apply_eq_kron] at h
    simpa only [hn, ite_true] using h
  have hd := L.Dhalf_ne_zero
  ext i j
  simp only [chiBoxX, Matrix.mul_apply, Matrix.of_apply, Matrix.one_apply]
  have h₁ : ∀ l : s, X i l * (y j l / (L.Dhalf l * L.Dhalf j)) =
      L.Dhalf i / L.Dhalf j * (kron i l * y j l + L.KX X i l * y j l) := fun l ↦ by
    rw [L.X_eq_Dhalf X i l]
    field_simp [hd l, hd j]
  simp only [y] at h₁ hy
  simp only [h₁, ← Finset.mul_sum, Finset.sum_add_distrib]
  rw [Finset.sum_coe_sort s (fun l ↦ kron i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l),
    Finset.sum_coe_sort s (fun l ↦ L.KX X i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l)]
  have hk : ∑ l ∈ s, kron i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l =
      Ring.inverse T (lp.single 2 (j : ℤ) 1) i := by
    simp only [kron, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq, i.2, ite_true]
  rw [hk, hy j i i.2]
  by_cases hij : i = j
  · subst hij
    rw [kron_self, ite_eq_left rfl, mul_one, div_self (hd i)]
  · rw [kron_of_ne fun h ↦ hij (Subtype.ext h), ite_eq_right hij, mul_zero]

/-- **The box matrix is invertible** where `1 + 𝒦_s` is. -/
theorem isUnit_chiBoxX_det (hX : L.ChiDecay X) {s : Finset ℤ}
    (hu : IsUnit (L.onePlusKX_box X hX s)) : IsUnit (chiBoxX X s).det :=
  Matrix.isUnit_det_of_right_inverse (L.chiBoxX_mul X hX hu)

/-- **Finite boxes**: wherever `1 + 𝒦` is invertible, the box matrices `X|_{S M}` are invertible
for all sufficiently large `M`, for boxes `S M` exhausting `ℤ`. -/
theorem eventually_isUnit_chiBoxX_det (hX : L.ChiDecay X) {S : ℕ → Finset ℤ}
    (hS : Tendsto S atTop atTop) (hu : IsUnit (L.onePlusKX X hX)) :
    ∀ᶠ M in atTop, IsUnit (chiBoxX X (S M)).det :=
  ((L.tendsto_onePlusKX_box X hX hS).eventually (Units.isOpen.mem_nhds hu)).mono fun _ hM ↦
    L.isUnit_chiBoxX_det X hX hM

/-- `(X|_s)⁻¹ᵢⱼ = [(1 + 𝒦_s)⁻¹]ᵢⱼ/(dᵢdⱼ)`. -/
theorem chiBoxX_inv (hX : L.ChiDecay X) {s : Finset ℤ}
    (hu : IsUnit (L.onePlusKX_box X hX s)) (i j : s) : (chiBoxX X s)⁻¹ i j =
      Ring.inverse (L.onePlusKX_box X hX s) (lp.single 2 (j : ℤ) 1) i /
        (L.Dhalf i * L.Dhalf j) := by
  rw [Matrix.inv_eq_right_inv (L.chiBoxX_mul X hX hu), Matrix.of_apply]

/-- The box vertex `β²[(X|_s)⁻¹ - b⁻¹]ₙₘ` for `n, m ∈ s`, zero outside the box. -/
noncomputable def gammaBoxX (s : Finset ℤ) (n m : ℤ) : ℂ :=
  if h : n ∈ s ∧ m ∈ s then L.β ^ 2 * ((chiBoxX X s)⁻¹ ⟨n, h.1⟩ ⟨m, h.2⟩ - kron n m / b n)
  else 0

/-- **The box vertices converge to `Γ`** wherever `1 + 𝒦` is invertible: `1 + 𝒦_M → 1 + 𝒦` in
norm, so `1 + 𝒦_M` is eventually invertible and `(1 + 𝒦_M)⁻¹ → (1 + 𝒦)⁻¹`. -/
theorem tendsto_gammaBoxX (hX : L.ChiDecay X) (hb : L.BubbleDecay b) {S : ℕ → Finset ℤ}
    (hS : Tendsto S atTop atTop) (hu : IsUnit (L.onePlusKX X hX)) (n m : ℤ) :
    Tendsto (fun M ↦ L.gammaBoxX X b (S M) n m) atTop
      (𝓝 (L.gammaOpX X b hX hb n m)) := by
  have hT := L.tendsto_onePlusKX_box X hX hS
  have hinv : Tendsto (fun M ↦ Ring.inverse (L.onePlusKX_box X hX (S M))) atTop
      (𝓝 (Ring.inverse (L.onePlusKX X hX))) := by
    have h := NormedRing.inverse_continuousAt hu.unit
    rw [hu.unit_spec] at h
    exact h.tendsto.comp hT
  have he := ((lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 n).continuous.tendsto _).comp
    (((ContinuousLinearMap.apply ℂ _ (lp.single 2 m (1 : ℂ))).continuous.tendsto _).comp hinv)
  have hlim := (he.sub_const (Ring.inverse (L.onePlusK0X b hb) (lp.single 2 m 1) n)).const_mul
    (L.β ^ 2 / (L.Dhalf n * L.Dhalf m))
  rw [show L.gammaOpX X b hX hb n m = L.β ^ 2 / (L.Dhalf n * L.Dhalf m) *
      (Ring.inverse (L.onePlusKX X hX) (lp.single 2 m 1) n -
        Ring.inverse (L.onePlusK0X b hb) (lp.single 2 m 1) n) from by
    rw [gammaOpX, _root_.sub_apply, lp.coeFn_sub, Pi.sub_apply]]
  refine hlim.congr' ?_
  filter_upwards [hS.eventually (eventually_ge_atTop {n, m}),
    hT.eventually (Units.isOpen.mem_nhds hu)] with M hM hU
  have hn : n ∈ S M := hM (by simp)
  have hm : m ∈ S M := hM (by simp)
  have hev : ∀ f : ℓ²(ℤ, ℂ), lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 n f = f n := fun _ ↦ rfl
  simp only [Function.comp_apply, ContinuousLinearMap.apply_apply, hev]
  rw [gammaBoxX, dite_eq_left ⟨hn, hm⟩, L.chiBoxX_inv X hX hU,
    L.inverse_onePlusK0X_single b hb, L.b_eq_Dhalf b]
  have hdn := L.Dhalf_ne_zero n
  have hdm := L.Dhalf_ne_zero m
  have htn := L.tX_ne_zero b hb n
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp
  · rw [kron_of_ne hnm]
    field_simp
    ring

end OpLattice

/-- The density and magnetic `χ_r` satisfies the bounds, by `ChiBounds.lean`. -/
theorem Params.chiDecay_chiR (A : Params) (k : ℤ) (ε : ℂ) (hμ : A.μ ≠ 0) :
    (A.opLattice k hμ).ChiDecay (A.chiR k ε) :=
  ⟨A.norm_chiR_le k ε hμ, A.norm_chiR_diag_le k ε hμ⟩

end HubbardAtom
