import HubbardAtomLean.SingletBounds
import HubbardAtomLean.SingletLZ

/-!
# The singlet vertex as an operator

Items 6(b)–(c) and the boxes of 6(d) for the singlet channel, by `ChiOperator.lean` applied to
`X = 2χ_s` and `b = -2B` with the bounds of `SingletBounds.lean`. For every `μ`, on the lattice
`z'ₙ` of `OpLatticeAll.lean` with `𝒟 = diag(-β/z'ₙ²)`, `2χ_s = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)` with `𝒦`
Hilbert–Schmidt, and `Γ_s = β²(χ_s⁻¹ + B⁻¹)` (Eq. (Gamma-def), `η_s = -1`) is twice the vertex
`β²(X⁻¹ - b⁻¹)` of `ChiOperator.lean`:

* `Params.gammaOpS`: `Γ_s = 2β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)`, entrywise;
* `Params.isUnit_onePlusKS_iff`: `1 + 𝒦` is invertible iff `χ_s` has no nonzero null vector `v`
  with `∑ₙ |vₙ/z'ₙ|² < ∞`; `Params.exists_bound_of_nullS`: such null vectors are bounded;
* `Params.gammaOpS_eq_chiInvS`: `Γ_s = β²(χ_s⁻¹ + B⁻¹)` with `χ_s⁻¹ = 2𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`,
  which, where `1 + 𝒦` is invertible, is a two-sided inverse of `χ_s` and the only right inverse
  with columns in the class;
* `Params.gammaOpS_eq_of_isRightInverse`: for every right inverse `Y` of `χ_s` with bounded
  columns, `1 + 𝒦` is invertible and `Γ_s = β²(Y + B⁻¹)`;
* `Params.SingletHyp.gammaOpS_eq`: under the assumptions of Theorem "singlet", `Γ_s` is the closed
  form `gammaS`; `Params.SingletLZ.gammaOpS_eq`: at a lattice zero `t_c + Aₛ = 0`, it is the
  continuation `gammaSlz` (item 1 of Proposition "special");
* `Params.tendsto_gammaBoxS`: wherever `1 + 𝒦` is invertible, the box matrices `χ_s|_M` are
  invertible for large `M` (`eventually_isUnit_chiBoxS_det`), and the box vertices
  `β²[(χ_s|_M)⁻¹ + B⁻¹]` on `|2n + 1 + k| ≤ 2M + 1` converge to `Γ_s`.
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ)

/-! ### The operator and the criterion -/

/-- `1 + 𝒦` for `2χ_s` on `ℓ²(ℤ)`. -/
noncomputable def onePlusKS : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  (A.opLatticeAll k).onePlusKX (A.chiS₂ k) (A.chiDecay_chiS k)

/-- **The singlet vertex as an operator**: `Γ_s,nm = 2β²/(dₙdₘ) [(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]ₙₘ`, with
`Ring.inverse` (zero where `1 + 𝒦` is not invertible). -/
noncomputable def gammaOpS (n m : ℤ) : ℂ :=
  2 * (A.opLatticeAll k).gammaOpX (A.chiS₂ k) (A.Bpp₂ k) (A.chiDecay_chiS k)
    (A.bubbleDecay_Bpp k) n m

/-- `χ_s⁻¹ = 2𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, entrywise, with `Ring.inverse` as in `gammaOpS`. -/
noncomputable def chiInvS (n m : ℤ) : ℂ :=
  2 * (A.opLatticeAll k).chiInvX (A.chiS₂ k) (A.chiDecay_chiS k) n m

/-- Null vectors of `2χ_s` and of `χ_s` agree. -/
theorem hasSum_chiS₂_iff {v : ℤ → ℂ} {n : ℤ} {c : ℂ} :
    HasSum (fun m ↦ A.chiS₂ k n m * v m) (2 * c) ↔ HasSum (fun m ↦ A.chiS k n m * v m) c := by
  simp only [chiS₂, mul_assoc]
  exact hasSum_mul_left_iff two_ne_zero

/-- **`Γ_s` exists iff `χ_s` has no null vector**: `1 + 𝒦` is invertible iff `χ_s` has no nonzero
null vector `v` with `∑ₙ |vₙ/z'ₙ|² < ∞`. -/
theorem isUnit_onePlusKS_iff :
    IsUnit (A.onePlusKS k) ↔ ∀ v : ℤ → ℂ, Summable (fun n ↦ ‖v n / A.zOp k n‖ ^ 2) →
      (∀ n, HasSum (fun m ↦ A.chiS k n m * v m) 0) → v = 0 := by
  rw [onePlusKS, (A.opLatticeAll k).isUnit_onePlusKX_iff]
  refine forall_congr' fun v ↦ imp_congr_right fun _ ↦ imp_congr_left (forall_congr' fun n ↦ ?_)
  rw [← A.hasSum_chiS₂_iff k, mul_zero]

/-- **Null vectors are bounded**: a null vector `v` of `χ_s` with `∑ₙ |vₙ/z'ₙ|² < ∞` has
`|vₙ| ≤ C`. -/
theorem exists_bound_of_nullS {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / A.zOp k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun m ↦ A.chiS k n m * v m) 0) : ∃ C, ∀ n, ‖v n‖ ≤ C :=
  (A.opLatticeAll k).exists_bound_of_nullX _ (A.chiDecay_chiS k) hv fun n ↦ by
    simpa using (A.hasSum_chiS₂_iff k (c := 0)).mpr (hχ n)

/-! ### `Γ_s = β²(χ_s⁻¹ + B⁻¹)` -/

/-- **`Γ_s = β²(χ_s⁻¹ + B⁻¹)`**, entrywise, for every `μ`. Where `1 + 𝒦` is not invertible,
`χ_s⁻¹ = 0`, so `Γ_s = β²δₙₘ/B(νₙ)` there. -/
theorem gammaOpS_eq_chiInvS (n m : ℤ) :
    A.gammaOpS k n m = A.β ^ 2 * (A.chiInvS k n m + kron n m / A.Bpp k n) := by
  rw [gammaOpS, (A.opLatticeAll k).gammaOpX_eq_chiInvX, chiInvS, Bpp₂]
  have hB := A.Bpp_ne_zero k n
  change (2 : ℂ) * (((A.β : ℝ) : ℂ) ^ 2 * _) = _
  field_simp
  ring

/-- `Y` is a right inverse of `2χ_s` iff `2Y` is one of `χ_s`. -/
theorem isRightInverse_chiS₂_iff {Y : ℤ → ℤ → ℂ} :
    IsRightInverse (A.chiS₂ k) Y ↔ IsRightInverse (A.chiS k) (fun n m ↦ 2 * Y n m) := by
  refine forall₂_congr fun n m ↦ ?_
  congr! 1
  funext l
  simp only [chiS₂]
  ring

/-- `Y` is a left inverse of `2χ_s` iff `2Y` is one of `χ_s`. -/
theorem isLeftInverse_chiS₂_iff {Y : ℤ → ℤ → ℂ} :
    IsLeftInverse (A.chiS₂ k) Y ↔ IsLeftInverse (A.chiS k) (fun n m ↦ 2 * Y n m) := by
  refine forall₂_congr fun n m ↦ ?_
  congr! 1
  funext l
  simp only [chiS₂]
  ring

theorem chiS₂_symm (n m : ℤ) : A.chiS₂ k n m = A.chiS₂ k m n := by
  rw [chiS₂, chiS₂, A.chiS_symm']

/-- **`χ_s⁻¹` is a right inverse** where `1 + 𝒦` is invertible. -/
theorem chiInvS_isRightInverse (hK : IsUnit (A.onePlusKS k)) :
    IsRightInverse (A.chiS k) (A.chiInvS k) :=
  (A.isRightInverse_chiS₂_iff k).mp ((A.opLatticeAll k).chiInvX_isRightInverse _ _ hK)

/-- **`χ_s⁻¹` is a left inverse** where `1 + 𝒦` is invertible, as `χ_s` is symmetric. -/
theorem chiInvS_isLeftInverse (hK : IsUnit (A.onePlusKS k)) :
    IsLeftInverse (A.chiS k) (A.chiInvS k) :=
  (A.isLeftInverse_chiS₂_iff k).mp
    ((A.opLatticeAll k).chiInvX_isLeftInverse _ _ (A.chiS₂_symm k) hK)

/-- **`χ_s⁻¹` is the only right inverse in the class**: where `1 + 𝒦` is invertible, a `v` with
`∑ₙ |vₙ/z'ₙ|² < ∞` and `∑ₗ χ_s,nl vₗ = δₙₘ` is column `m` of `χ_s⁻¹`. -/
theorem eq_chiInvS_of_hasSum (hK : IsUnit (A.onePlusKS k)) {m : ℤ}
    {v : ℤ → ℂ} (hv : Summable fun n ↦ ‖v n / A.zOp k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun l ↦ A.chiS k n l * v l) (kron n m)) (n : ℤ) :
    v n = A.chiInvS k n m := by
  have hv' : Summable fun n ↦ ‖v n / 2 / A.zOp k n‖ ^ 2 := by
    refine (hv.mul_left ((1 / 2) ^ 2)).congr fun n ↦ ?_
    simp only [norm_div, Complex.norm_two]
    ring
  have h := (A.opLatticeAll k).eq_chiInvX_of_hasSum _ (A.chiDecay_chiS k) hK hv' (m := m)
    fun n ↦ by
      have := (A.hasSum_chiS₂_iff k (v := fun l ↦ v l / 2) (c := kron n m / 2)).mpr
        (by simpa [mul_div_assoc'] using (hχ n).div_const 2)
      rwa [mul_div_cancel₀ _ two_ne_zero] at this
  rw [chiInvS, ← h]
  ring

/-! ### The closed form under the assumptions of Theorem "singlet" -/

/-- `1/(tₙ + c) → 0` along the cofinite filter. -/
theorem tendsto_inv_t_add (c : ℂ) :
    Tendsto (fun n ↦ 1 / (A.t k n + c)) cofinite (𝓝 0) := by
  have hq : ContinuousAt (fun s ↦ s ^ 2 * (A.tS k s + c * s ^ 2)⁻¹) 0 := by
    unfold tS
    exact ContinuousAt.mul (by fun_prop) (ContinuousAt.inv₀ (by fun_prop) (by simp))
  have h := hq.tendsto.comp (A.tendsto_inv_z k)
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_mul] at h
  refine h.congr' ((A.eventually_zOp_eq_ne k).mono fun n ⟨_, _, hz⟩ ↦ ?_)
  simp only [Function.comp_apply]
  rw [A.t_eq_tS k hz, one_div, show A.z k n ^ 2 * A.tS k (A.z k n)⁻¹ + c =
    A.z k n ^ 2 * (A.tS k (A.z k n)⁻¹ + c * (A.z k n)⁻¹ ^ 2) by field_simp, mul_inv, inv_pow]

/-- A matrix `mirrorBlock + Z` has bounded columns if the columns of `Z` converge: off `{m, m̄}`,
column `m` is that of `Z`. -/
theorem bdd_mirrorBlock_add {B L : ℤ → ℂ} {Z : ℤ → ℤ → ℂ} {m : ℤ} {l : ℂ}
    (hZ : Tendsto (fun n ↦ Z n m) cofinite (𝓝 l)) :
    ∃ C, ∀ n, ‖mirrorBlock k B L n m + Z n m‖ ≤ C := by
  have hev : (fun n ↦ Z n m) =ᶠ[cofinite] fun n ↦ mirrorBlock k B L n m + Z n m := by
    filter_upwards [eventually_cofinite_ne m, eventually_cofinite_ne (mirror k m)] with n h₁ h₂
    rw [mirrorBlock_of_ne h₁ (fun h ↦ h₂ (by rw [h, mirror_mirror])), zero_add]
  obtain ⟨C, hC⟩ := (hZ.congr' hev).norm.bddAbove_range_of_cofinite
  exact ⟨C, fun n ↦ hC ⟨n, rfl⟩⟩

/-- **`1 + 𝒦` is invertible** if `χ_s` has a right inverse `Y` with bounded columns, as `χ_s` is
symmetric. -/
theorem isUnit_onePlusKS_of_isRightInverse {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiS k) Y) (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) :
    IsUnit (A.onePlusKS k) :=
  (A.opLatticeAll k).isUnit_onePlusKX_of_isRightInverse _ _ (A.chiS₂_symm k)
    (Y := fun n m ↦ Y n m / 2)
    ((A.isRightInverse_chiS₂_iff k).mpr (by simpa [mul_div_cancel₀] using hY))
    fun m ↦ (hbdd m).imp fun C hC n ↦ by
      rw [norm_div, Complex.norm_two]
      linarith [hC n, norm_nonneg (Y n m)]

/-- **`χ_s⁻¹` is every right inverse with bounded columns**. -/
theorem eq_chiInvS_of_isRightInverse {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiS k) Y) (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) (n m : ℤ) :
    Y n m = A.chiInvS k n m := by
  have h := (A.opLatticeAll k).eq_chiInvX_of_isRightInverse _ (A.chiDecay_chiS k)
    (A.chiS₂_symm k) (Y := fun n m ↦ Y n m / 2)
    ((A.isRightInverse_chiS₂_iff k).mpr (by simpa [mul_div_cancel₀] using hY))
    (fun m ↦ (hbdd m).imp fun C hC n ↦ by
      rw [norm_div, Complex.norm_two]
      linarith [hC n, norm_nonneg (Y n m)]) n m
  rw [chiInvS, ← h]
  ring

/-- **The vertex of a right inverse with bounded columns**: `Γ_s = β²(Y + B⁻¹)`. -/
theorem gammaOpS_eq_of_isRightInverse {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiS k) Y) (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) (n m : ℤ) :
    A.gammaOpS k n m = A.β ^ 2 * (Y n m + kron n m / A.Bpp k n) := by
  rw [A.gammaOpS_eq_chiInvS k, A.eq_chiInvS_of_isRightInverse k hY hbdd]

variable {k} in
/-- The columns of `invS` are bounded: off `{m, m̄}` they are the separable term, which tends to
`4a/β²`. -/
theorem bdd_invS (ξ : ℂ) (m : ℤ) : ∃ C, ∀ n, ‖A.invS k ξ n m‖ ≤ C :=
  bdd_mirrorBlock_add k (Z := A.Zs k ξ) (by
    unfold Zs
    exact (((tendsto_const_nhds.mul (A.tendsto_inv_t_add k (A.As k))).mul
      tendsto_const_nhds).div_const _).const_add _ |>.const_mul _)

variable {k} in
/-- The columns of `invSlz` are bounded: off `{m, m̄}` they are `Zlz`, which tends to `4a/β²`. -/
theorem bdd_invSlz (c m : ℤ) : ∃ C, ∀ n, ‖A.invSlz k c n m‖ ≤ C := by
  have ho : Tendsto (fun n ↦ orb k c n) cofinite (𝓝 0) :=
    tendsto_const_nhds.congr' (by
      filter_upwards [eventually_cofinite_ne c, eventually_cofinite_ne (mirror k c)] with n h₁ h₂
      simp [orb, h₁, h₂])
  exact bdd_mirrorBlock_add k (Z := A.Zlz k c) (by
    unfold Zlz
    exact ((tendsto_const_nhds.sub (tendsto_const_nhds.mul ((ho.mul tendsto_const_nhds).add
      ((A.tendsto_inv_t_add k (A.As k)).mul tendsto_const_nhds)))).add
      ((tendsto_const_nhds.mul ho).mul tendsto_const_nhds)))

variable {A k}

/-- **`1 + 𝒦` is invertible** under the assumptions of Theorem "singlet". -/
theorem SingletHyp.isUnit_onePlusKS {ξ : ℂ} (h : A.SingletHyp k ξ) :
    IsUnit (A.onePlusKS k) :=
  A.isUnit_onePlusKS_of_isRightInverse k h.chiS_isRightInverse (A.bdd_invS ξ)

/-- **`χ_s⁻¹` is `invS`** under the assumptions of Theorem "singlet". -/
theorem SingletHyp.invS_eq_chiInvS {ξ : ℂ} (h : A.SingletHyp k ξ) (n m : ℤ) :
    A.invS k ξ n m = A.chiInvS k n m :=
  A.eq_chiInvS_of_isRightInverse k h.chiS_isRightInverse (A.bdd_invS ξ) n m

/-- **The operator vertex is the closed form**: under the assumptions of Theorem "singlet",
`Γ_s = 2β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)` is `gammaS`, Eq. (Gamma-s). -/
theorem SingletHyp.gammaOpS_eq {ξ : ℂ} (h : A.SingletHyp k ξ) (n m : ℤ) :
    A.gammaOpS k n m = A.gammaS k ξ n m := by
  rw [A.gammaOpS_eq_of_isRightInverse k h.chiS_isRightInverse (A.bdd_invS ξ), h.gammaS_eq]

/-- **`1 + 𝒦` is invertible** at a lattice zero `t_c + Aₛ = 0` (item 1 of Proposition
"special"). -/
theorem SingletLZ.isUnit_onePlusKS {c : ℤ} (h : A.SingletLZ k c) :
    IsUnit (A.onePlusKS k) :=
  A.isUnit_onePlusKS_of_isRightInverse k h.chiS_isRightInverse (A.bdd_invSlz c)

/-- **The operator vertex at a lattice zero** is the continuation `gammaSlz`, Eq. (Gamma-s-lz). -/
theorem SingletLZ.gammaOpS_eq {c : ℤ} (h : A.SingletLZ k c) (n m : ℤ) :
    A.gammaOpS k n m = A.gammaSlz k c n m := by
  rw [A.gammaOpS_eq_of_isRightInverse k h.chiS_isRightInverse (A.bdd_invSlz c),
    h.gammaS_eq]

/-! ### Finite boxes -/

variable (A k)

/-- The box vertex `β²[(χ_s|_s)⁻¹ + B⁻¹]ₙₘ` for `n, m ∈ s`, zero outside the box. -/
noncomputable def gammaBoxS (s : Finset ℤ) (n m : ℤ) : ℂ :=
  if h : n ∈ s ∧ m ∈ s then
    A.β ^ 2 * ((OpLattice.chiBoxX (A.chiS k) s)⁻¹ ⟨n, h.1⟩ ⟨m, h.2⟩ + kron n m / A.Bpp k n)
  else 0

/-- `(2χ_s)|_s = 2 • χ_s|_s`. -/
theorem chiBoxX_chiS₂ (s : Finset ℤ) :
    OpLattice.chiBoxX (A.chiS₂ k) s = (2 : ℂ) • OpLattice.chiBoxX (A.chiS k) s := by
  ext i j
  simp [OpLattice.chiBoxX, chiS₂]

/-- `(2χ_s|_s)⁻¹ = (χ_s|_s)⁻¹/2`, also where `χ_s|_s` is singular (both sides vanish). -/
theorem inv_chiBoxX_chiS₂ (s : Finset ℤ) :
    (OpLattice.chiBoxX (A.chiS₂ k) s)⁻¹ = (2 : ℂ)⁻¹ • (OpLattice.chiBoxX (A.chiS k) s)⁻¹ := by
  rw [A.chiBoxX_chiS₂ k]
  by_cases hd : IsUnit (OpLattice.chiBoxX (A.chiS k) s).det
  · have h := Matrix.inv_smul' _ (Units.mk0 (2 : ℂ) two_ne_zero) hd
    rwa [Units.smul_def, Units.smul_def, Units.val_inv_eq_inv_val, Units.val_mk0] at h
  · have hd' : ¬IsUnit ((2 : ℂ) • OpLattice.chiBoxX (A.chiS k) s).det := by
      rw [Matrix.det_smul, isUnit_iff_ne_zero, not_ne_iff] at *
      rw [hd, mul_zero]
    rw [Matrix.nonsing_inv_apply_not_isUnit _ hd, Matrix.nonsing_inv_apply_not_isUnit _ hd',
      smul_zero]

/-- The box vertex is twice that of `ChiOperator.lean` for `2χ_s`, `-2B`. -/
theorem gammaBoxS_eq (s : Finset ℤ) (n m : ℤ) :
    A.gammaBoxS k s n m =
      2 * (A.opLatticeAll k).gammaBoxX (A.chiS₂ k) (A.Bpp₂ k) s n m := by
  unfold gammaBoxS OpLattice.gammaBoxX
  split_ifs with h
  · have hB := A.Bpp_ne_zero k n
    rw [A.inv_chiBoxX_chiS₂ k, Matrix.smul_apply, smul_eq_mul, Bpp₂]
    change _ = (2 : ℂ) * (((A.β : ℝ) : ℂ) ^ 2 * _)
    field_simp
    ring
  · rw [mul_zero]

/-- **Finite boxes**: wherever `1 + 𝒦` is invertible, the box matrices `χ_s|_M` on
`|2n + 1 + k| ≤ 2M + 1` are invertible for all sufficiently large `M`. -/
theorem eventually_isUnit_chiBoxS_det (hu : IsUnit (A.onePlusKS k)) :
    ∀ᶠ M in atTop, IsUnit (OpLattice.chiBoxX (A.chiS k) (box k M)).det := by
  filter_upwards [(A.opLatticeAll k).eventually_isUnit_chiBoxX_det _ (A.chiDecay_chiS k)
    (tendsto_box k) hu] with M hM
  rw [A.chiBoxX_chiS₂ k, Matrix.det_smul, isUnit_iff_ne_zero] at hM
  exact isUnit_iff_ne_zero.mpr (right_ne_zero_of_mul hM)

/-- **The box vertices converge to `Γ_s`** wherever `1 + 𝒦` is invertible. -/
theorem tendsto_gammaBoxS (hu : IsUnit (A.onePlusKS k)) (n m : ℤ) :
    Tendsto (fun M ↦ A.gammaBoxS k (box k M) n m) atTop (𝓝 (A.gammaOpS k n m)) := by
  simp only [A.gammaBoxS_eq k, gammaOpS]
  exact ((A.opLatticeAll k).tendsto_gammaBoxX _ _ _ _ (tendsto_box k) hu n m).const_mul 2

variable {A k}

/-- **Boxes under the assumptions of Theorem "singlet"**: the box vertices converge to the closed
form `gammaS`. -/
theorem SingletHyp.tendsto_gammaBoxS {ξ : ℂ} (h : A.SingletHyp k ξ) (n m : ℤ) :
    Tendsto (fun M ↦ A.gammaBoxS k (box k M) n m) atTop (𝓝 (A.gammaS k ξ n m)) := by
  rw [← h.gammaOpS_eq]
  exact A.tendsto_gammaBoxS k h.isUnit_onePlusKS n m

end Params

end HubbardAtom
