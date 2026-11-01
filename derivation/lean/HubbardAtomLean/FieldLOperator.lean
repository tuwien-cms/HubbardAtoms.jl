import HubbardAtomLean.FieldLBounds

/-!
# The longitudinal vertex as an operator

Items 6(b)–(d) for the longitudinal sector, off the axes (`μ ≠ 0`, `h ≠ 0`, `μ ≠ ±h`), by
`ChiOperator.lean` and the bounds of `FieldLBounds.lean`. The two spin lattices are enumerated as
one by `encL : Bool × ℤ ≃ ℤ`, `(σ, n) ↦ 2n + [σ]` (`opLatL`), and `χ_L` and the bubble are read
through it (`chiLe`, `bLe`). With `𝒟 = diag(-β/z_σn²)`:

* `FParams.gammaOpL`: `Γ_L = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)`, entrywise;
* `FParams.isUnit_onePlusKL_iff`: `1 + 𝒦` is invertible iff `χ_L` has no nonzero null vector `v`
  on `Bool × ℤ` with `∑_{σn} |v_σn/z_σn|² < ∞`; `FParams.exists_bound_of_nullL`: such null
  vectors are bounded;
* `FParams.gammaOpL_eq_chiInvL`: `Γ_L = β²(χ_L⁻¹ - b⁻¹)` with `χ_L⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`,
  which, where `1 + 𝒦` is invertible, is a two-sided inverse of `χ_L` (`chiInvL_isRightInverse`,
  `chiInvL_isLeftInverse`);
* `FParams.LGeneric.gammaOpL_eq`: on the domain of Eq. (field-L) (`LGeneric` for both column
  spins), `1 + 𝒦` is invertible and `Γ_L` is the closed form `gammaL`, as the columns of `invL`
  are bounded (`tendsto_invL`);
* `FParams.tendsto_gammaBoxL`: wherever `1 + 𝒦` is invertible, the vertices of the boxes
  `|2n + 1 + k| ≤ 2M + 1` (both spins) converge to `Γ_L`, on the domain of Eq. (field-L) to
  `gammaL` (`LGeneric.tendsto_gammaBoxL`).
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace FParams

/-! ### The two spin lattices as one -/

/-- `Bool × ℤ ≃ ℤ`, `(σ, n) ↦ 2n + [σ]`. -/
def encL : Bool × ℤ ≃ ℤ where
  toFun p := 2 * p.2 + bif p.1 then 1 else 0
  invFun j := (decide (j % 2 = 1), j / 2)
  left_inv := by
    rintro ⟨σ, n⟩
    cases σ <;> refine Prod.ext ?_ ?_ <;>
      simp only [Bool.cond_true, Bool.cond_false, decide_eq_true_eq, decide_eq_false_iff_not] <;>
      omega
  right_inv := by
    intro j
    by_cases h : j % 2 = 1 <;> simp [h] <;> omega

/-- `δ` on the enumerated lattice. -/
theorem kron_encL (p q : Bool × ℤ) :
    kron (encL p) (encL q) = if p.1 = q.1 then kron p.2 q.2 else 0 := by
  obtain ⟨σ, n⟩ := p
  obtain ⟨τ, m⟩ := q
  by_cases hσ : σ = τ
  · subst hσ
    by_cases hnm : n = m
    · subst hnm
      simp [kron_self]
    · rw [kron_of_ne fun h ↦ hnm (Prod.ext_iff.mp (encL.injective h)).2, ite_eq_left rfl,
        kron_of_ne hnm]
  · rw [kron_of_ne fun h ↦ hσ (Prod.ext_iff.mp (encL.injective h)).1, ite_eq_right hσ]

variable (F : FParams) (k : ℤ)

/-- The point of `encL.symm j` on the lattice of its spin. -/
noncomputable def zLe (j : ℤ) : ℂ := (F.BL (encL.symm j).1).z k (encL.symm j).2

@[simp] theorem zLe_encL (p : Bool × ℤ) : F.zLe k (encL p) = (F.BL p.1).z k p.2 := by
  simp [zLe]

/-- The lattices of the two spins, enumerated by `encL`, for `μ_σ ≠ 0`. -/
noncomputable def opLatL (hc : ∀ σ, F.cL σ ≠ 0) : OpLattice where
  β := F.β
  β_pos := F.β_pos
  z := F.zLe k
  z_ne_zero j := (F.BL _).z_ne_zero k (hc _) _
  summable_inv_sq := by
    refine (encL.symm.summable_iff (f := fun p : Bool × ℤ ↦ ‖(F.BL p.1).z k p.2‖⁻¹ ^ 2)).mpr ?_
    refine (summable_prod_of_nonneg fun _ ↦ by positivity).mpr
      ⟨fun σ ↦ (F.BL σ).summable_inv_norm_z_sq k (hc σ), Summable.of_finite⟩

@[simp] theorem opLatL_z (hc : ∀ σ, F.cL σ ≠ 0) (p : Bool × ℤ) :
    (F.opLatL k hc).z (encL p) = (F.BL p.1).z k p.2 :=
  F.zLe_encL k p

@[simp] theorem opLatL_β (hc : ∀ σ, F.cL σ ≠ 0) : (F.opLatL k hc).β = F.β := rfl

/-- `χ_L` on the enumerated lattice. -/
noncomputable def chiLe (i j : ℤ) : ℂ :=
  F.chiL (encL.symm i).1 (encL.symm j).1 k (encL.symm i).2 (encL.symm j).2

/-- The bubble on the enumerated lattice. -/
noncomputable def bLe (j : ℤ) : ℂ := F.bL (encL.symm j).1 k (encL.symm j).2

@[simp] theorem chiLe_encL (p q : Bool × ℤ) :
    F.chiLe k (encL p) (encL q) = F.chiL p.1 q.1 k p.2 q.2 := by
  simp [chiLe]

@[simp] theorem bLe_encL (p : Bool × ℤ) : F.bLe k (encL p) = F.bL p.1 k p.2 := by
  simp [bLe]

/-- `χ_L` is symmetric off the axes. -/
theorem chiLe_symm (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (i j : ℤ) : F.chiLe k i j = F.chiLe k j i :=
  F.chiL_symm k hμ hh _ _ _ _

/-! ### The bounds -/

/-- The off-the-axes conditions `μ ≠ 0`, `h ≠ 0`, `μ ± h ≠ 0`. -/
structure OffAxes : Prop where
  μ_ne : F.μ ≠ 0
  h_ne : F.h ≠ 0
  c_ne : ∀ σ, F.cL σ ≠ 0

/-- **The bounds of item 6(a)** for `χ_L`, off the axes. -/
theorem chiDecay_chiLe (ho : F.OffAxes) : (F.opLatL k ho.c_ne).ChiDecay (F.chiLe k) := by
  obtain ⟨c₁, hc₁⟩ := F.norm_chiL_self_le k (ho.c_ne true)
  obtain ⟨c₂, hc₂⟩ := F.norm_chiL_self_le k (ho.c_ne false)
  obtain ⟨c₃, hc₃⟩ := F.norm_chiL_opp_le k ho.μ_ne ho.h_ne (ho.c_ne true) (ho.c_ne (!true))
  obtain ⟨c₄, hc₄⟩ := F.norm_chiL_opp_le k ho.μ_ne ho.h_ne (ho.c_ne false) (ho.c_ne (!false))
  obtain ⟨d₁, hd₁⟩ := F.norm_chiL_diag_le k (ho.c_ne true)
  obtain ⟨d₂, hd₂⟩ := F.norm_chiL_diag_le k (ho.c_ne false)
  refine ⟨⟨max (max c₁ c₂) (max c₃ c₄), fun i j hij ↦ ?_⟩, ⟨max d₁ d₂, fun i ↦ ?_⟩⟩
  · obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
    obtain ⟨⟨τ, m⟩, rfl⟩ := encL.surjective j
    rw [chiLe_encL, opLatL_z, opLatL_z]
    dsimp only
    have hD : 0 ≤ ‖(F.BL σ).z k n‖ ^ 2 * ‖(F.BL τ).z k m‖ ^ 2 := by positivity
    by_cases hστ : τ = σ
    · subst hστ
      have hnm : n ≠ m := fun h ↦ hij (by rw [h])
      refine le_trans ?_ (div_le_div_of_nonneg_right (le_max_left _ _) hD)
      cases τ
      · exact (hc₂ n m hnm).trans (div_le_div_of_nonneg_right (le_max_right _ _) hD)
      · exact (hc₁ n m hnm).trans (div_le_div_of_nonneg_right (le_max_left _ _) hD)
    · obtain rfl : τ = !σ := by cases σ <;> cases τ <;> simp_all
      refine le_trans ?_ (div_le_div_of_nonneg_right (le_max_right _ _) hD)
      cases σ
      · exact (hc₄ n m).trans (div_le_div_of_nonneg_right (le_max_right _ _) hD)
      · exact (hc₃ n m).trans (div_le_div_of_nonneg_right (le_max_left _ _) hD)
  · obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
    rw [chiLe_encL, opLatL_z, opLatL_β]
    dsimp only
    have hD : 0 ≤ ‖(F.BL σ).z k n‖ ^ 3 := by positivity
    cases σ
    · exact (hd₂ n).trans (div_le_div_of_nonneg_right (le_max_right _ _) hD)
    · exact (hd₁ n).trans (div_le_div_of_nonneg_right (le_max_left _ _) hD)

/-- **The bubble** decays as `-β/z²` on both lattices. -/
theorem bubbleDecay_bLe (ho : F.OffAxes) : (F.opLatL k ho.c_ne).BubbleDecay (F.bLe k) := by
  obtain ⟨C₁, hC₁⟩ := F.norm_zL_mul_tL_le k (ho.c_ne true)
  obtain ⟨C₂, hC₂⟩ := F.norm_zL_mul_tL_le k (ho.c_ne false)
  refine ⟨fun j ↦ F.bL_ne_zero k _ _, max C₁ C₂, fun j ↦ ?_⟩
  obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective j
  rw [OpLattice.tX, opLatL_z, opLatL_β, bLe_encL]
  dsimp only
  cases σ
  · exact (hC₂ n).trans (le_max_right _ _)
  · exact (hC₁ n).trans (le_max_left _ _)

/-! ### The operator and the criterion -/

/-- `1 + 𝒦` for `χ_L`. -/
noncomputable def onePlusKL (ho : F.OffAxes) : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  (F.opLatL k ho.c_ne).onePlusKX (F.chiLe k) (F.chiDecay_chiLe k ho)

/-- **The longitudinal vertex as an operator**, entrywise on `Bool × ℤ`:
`Γ_L,(σn)(τm) = β²/(d_σn d_τm) [(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]`, with `Ring.inverse` (zero where
`1 + 𝒦` is not invertible). -/
noncomputable def gammaOpL (ho : F.OffAxes) (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) : ℂ :=
  (F.opLatL k ho.c_ne).gammaOpX (F.chiLe k) (F.bLe k) (F.chiDecay_chiLe k ho)
    (F.bubbleDecay_bLe k ho) (encL (σ, n)) (encL (τ, m))

/-- `χ_L⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, entrywise on `Bool × ℤ`. -/
noncomputable def chiInvL (ho : F.OffAxes) (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) : ℂ :=
  (F.opLatL k ho.c_ne).chiInvX (F.chiLe k) (F.chiDecay_chiLe k ho) (encL (σ, n)) (encL (τ, m))

/-- A sum over `ℤ` read through `encL` is a sum over `Bool × ℤ`. -/
theorem hasSum_encL_iff {f : Bool × ℤ → ℂ} {a : ℂ} :
    HasSum (fun j ↦ f (encL.symm j)) a ↔ HasSum f a := by
  rw [← encL.symm.hasSum_iff]
  rfl

/-- The class `∑ |vⱼ/zⱼ|² < ∞` on the enumerated lattice is the class on `Bool × ℤ`. -/
theorem summable_opLatL_iff (hc : ∀ σ, F.cL σ ≠ 0) {v : Bool × ℤ → ℂ} :
    Summable (fun j ↦ ‖v (encL.symm j) / (F.opLatL k hc).z j‖ ^ 2) ↔
      Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).z k p.2‖ ^ 2) := by
  rw [← encL.summable_iff]
  simp only [Function.comp_def, Equiv.symm_apply_apply, opLatL_z]

/-- A row of `χ_L` against `v` on the enumerated lattice is the row on `Bool × ℤ`. -/
theorem hasSum_row_iff (σ : Bool) (n : ℤ) {v : Bool × ℤ → ℂ} {a : ℂ} :
    HasSum (fun j ↦ F.chiLe k (encL (σ, n)) j * v (encL.symm j)) a ↔
      HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) a := by
  rw [← encL.hasSum_iff]
  simp only [Function.comp_def, Equiv.symm_apply_apply, chiLe_encL]

/-- **`Γ_L` exists iff `χ_L` has no null vector**: off the axes, `1 + 𝒦` is invertible iff
`χ_L` has no nonzero null vector `v` on `Bool × ℤ` with `∑_{σn} |v_σn/z_σn|² < ∞`. -/
theorem isUnit_onePlusKL_iff (ho : F.OffAxes) :
    IsUnit (F.onePlusKL k ho) ↔ ∀ v : Bool × ℤ → ℂ,
      Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).z k p.2‖ ^ 2) →
        (∀ σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) 0) → v = 0 := by
  rw [onePlusKL, OpLattice.isUnit_onePlusKX_iff]
  constructor
  · intro h v hv hχ
    have h0 := h (fun j ↦ v (encL.symm j)) ((F.summable_opLatL_iff k ho.c_ne).mpr hv)
      fun i ↦ by
        obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
        exact (F.hasSum_row_iff k σ n).mpr (hχ σ n)
    funext p
    simpa using congrFun h0 (encL p)
  · intro h w hw hχ
    have h0 := h (fun p ↦ w (encL p))
      ((F.summable_opLatL_iff k ho.c_ne).mp (by simpa using hw)) fun σ n ↦
        (F.hasSum_row_iff k σ n).mp (by simpa using hχ (encL (σ, n)))
    funext j
    obtain ⟨p, rfl⟩ := encL.surjective j
    exact congrFun h0 p

/-- **Null vectors are bounded**: a null vector `v` of `χ_L` on `Bool × ℤ` with
`∑_{σn} |v_σn/z_σn|² < ∞` has `|v_σn| ≤ C`. -/
theorem exists_bound_of_nullL (ho : F.OffAxes) {v : Bool × ℤ → ℂ}
    (hv : Summable fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).z k p.2‖ ^ 2)
    (hχ : ∀ σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) 0) :
    ∃ C, ∀ p, ‖v p‖ ≤ C := by
  obtain ⟨C, hC⟩ := (F.opLatL k ho.c_ne).exists_bound_of_nullX (F.chiLe k)
    (F.chiDecay_chiLe k ho) (v := fun j ↦ v (encL.symm j))
    ((F.summable_opLatL_iff k ho.c_ne).mpr hv) fun i ↦ by
      obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
      exact (F.hasSum_row_iff k σ n).mpr (hχ σ n)
  exact ⟨C, fun p ↦ by simpa using hC (encL p)⟩

/-! ### `Γ_L = β²(χ_L⁻¹ - b⁻¹)` -/

/-- **`Γ_L = β²(χ_L⁻¹ - b⁻¹)`**, entrywise, off the axes. Where `1 + 𝒦` is not invertible,
`χ_L⁻¹ = 0`. -/
theorem gammaOpL_eq_chiInvL (ho : F.OffAxes) (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) :
    F.gammaOpL k ho σ n τ m =
      F.β ^ 2 * (F.chiInvL k ho σ n τ m - if σ = τ then kron n m / F.bL σ k n else 0) := by
  rw [gammaOpL, OpLattice.gammaOpX_eq_chiInvX, kron_encL, bLe_encL]
  split_ifs <;> simp [chiInvL]

/-- A right inverse of `χ_L` on `Bool × ℤ` (as in `chiL_isRightInverse`) whose rows are summable
gives a right inverse of `chiLe`. -/
theorem isRightInverse_chiLe_of (ho : F.OffAxes) {Y : Bool → ℤ → Bool → ℤ → ℂ}
    (hbdd : ∀ t m, ∃ C, ∀ τ l, ‖Y τ l t m‖ ≤ C)
    (hY : ∀ σ n t m, HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * Y τ l t m)
      (if σ = t then kron n m else 0)) :
    IsRightInverse (F.chiLe k)
      (fun i j ↦ Y (encL.symm i).1 (encL.symm i).2 (encL.symm j).1 (encL.symm j).2) := by
  intro i j
  obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
  obtain ⟨⟨t, m⟩, rfl⟩ := encL.surjective j
  obtain ⟨C, hC⟩ := hbdd t m
  set g : Bool × ℤ → ℂ := fun p ↦ F.chiL σ p.1 k n p.2 * Y p.1 p.2 t m
  have hs : Summable g := by
    have h := (F.opLatL k ho.c_ne).summable_norm_X_mul (F.chiLe k) (F.chiDecay_chiLe k ho)
      (v := fun l ↦ Y (encL.symm l).1 (encL.symm l).2 t m) (C := C) (fun l ↦ hC _ _) (encL (σ, n))
    refine Summable.of_norm (encL.summable_iff.mpr h |>.congr fun p ↦ ?_)
    simp [g, chiLe]
  have h₁ : HasSum (fun q : ℤ × Bool ↦ g q.swap) (∑' p, g p) :=
    (Equiv.prodComm ℤ Bool).hasSum_iff.mpr hs.hasSum
  have h₂ : HasSum (fun l ↦ ∑ τ, g (τ, l)) (∑' p, g p) :=
    h₁.prod_fiberwise fun l ↦ hasSum_fintype _
  have h₃ := h₂.unique (hY σ n t m)
  have h₄ : HasSum g (if σ = t then kron n m else 0) := h₃ ▸ hs.hasSum
  have h₅ : HasSum (fun j ↦ g (encL.symm j)) (if σ = t then kron n m else 0) :=
    hasSum_encL_iff.mpr h₄
  simp only [kron_encL]
  convert h₅ using 1
  funext l
  simp [g, chiLe]

/-! ### The closed form on the domain of Eq. (field-L) -/

variable {k} in
/-- **The columns of `invL` are bounded**: `invL τ l t m → 0` for `τ = t` and `→ 2a/β²` for
`τ ≠ t`, as `|l| → ∞`. -/
theorem tendsto_invL (u : Fin 4 → ℂ) (τ t : Bool) (m : ℤ) :
    Tendsto (fun l ↦ F.invL k u τ l t m) cofinite
      (𝓝 (if τ = t then 0 else 2 * F.a / F.β ^ 2)) := by
  have hk : Tendsto (fun l ↦ kron l m) cofinite (𝓝 0) :=
    tendsto_const_nhds.congr' ((eventually_cofinite_ne m).mono fun n hn ↦ (kron_of_ne hn).symm)
  have hi : ∀ σ x, Tendsto (fun l ↦ ((F.BL σ).z k l - x)⁻¹) cofinite (𝓝 0) := fun σ x ↦
    tendsto_inv₀_cobounded.comp
      (tendsto_norm_atTop_iff_cobounded.mp ((F.BL σ).tendsto_norm_z_sub k x))
  have hmode : ∀ (c u : ℂ), Tendsto (fun l ↦ c * F.modeL k t τ u l) cofinite (𝓝 0) :=
    fun c u ↦ by
      have h := (((hi τ u).const_mul (F.fMp k t τ u)).add
        ((hi τ (-u)).const_mul (F.fMm k t τ u))).const_mul (F.PL k u)
      simp only [mul_zero, add_zero] at h
      simpa using (h.congr fun l ↦ by simp [modeL, div_eq_mul_inv, sub_neg_eq_add]).const_mul c
  have hsum : Tendsto (fun l ↦ ∑ j, F.cMode k t u j / ((F.BL t).z k m ^ 2 - u j ^ 2) *
      F.modeL k t τ (u j) l) cofinite (𝓝 0) := by
    simpa using tendsto_finsetSum Finset.univ fun j _ ↦ hmode _ (u j)
  have hd : Tendsto (fun l ↦ F.denseL k t τ m l) cofinite
      (𝓝 (if τ = t then 0 else 2 * F.a / F.β ^ 2)) := by
    by_cases hτ : τ = t
    · subst hτ
      have h := ((hk.const_mul (F.dhL k τ ((F.BL τ).z k m))).add
        ((hi τ (-(F.BL τ).z k m)).const_mul (F.AhL k τ ((F.BL τ).z k m)))).const_mul
          (F.PL k ((F.BL τ).z k m))
      simp only [mul_zero, add_zero] at h
      simpa [denseL, div_eq_mul_inv, sub_neg_eq_add] using h
    · have hτ' : τ = !t := by cases τ <;> cases t <;> simp_all
      subst hτ'
      have h := ((((hi (!t) ((F.BL t).z k m)).const_mul (F.BhL k t ((F.BL t).z k m))).add
        ((hi (!t) (-(F.BL t).z k m)).const_mul (F.ChL k t ((F.BL t).z k m)))).const_mul
          (F.PL k ((F.BL t).z k m))).add_const (2 * F.a / F.β ^ 2 : ℂ)
      simp only [mul_zero, add_zero, zero_add] at h
      simpa [denseL, hτ, div_eq_mul_inv, sub_neg_eq_add] using h
  simpa [invL] using hd.sub hsum

variable {k} in
/-- The columns of `invL` are bounded. -/
theorem bdd_invL (u : Fin 4 → ℂ) (t : Bool) (m : ℤ) : ∃ C, ∀ τ l, ‖F.invL k u τ l t m‖ ≤ C := by
  obtain ⟨C₁, hC₁⟩ := (F.tendsto_invL u true t m).norm.bddAbove_range_of_cofinite
  obtain ⟨C₂, hC₂⟩ := (F.tendsto_invL u false t m).norm.bddAbove_range_of_cofinite
  refine ⟨max C₁ C₂, fun τ l ↦ ?_⟩
  cases τ
  · exact (hC₂ ⟨l, rfl⟩).trans (le_max_right _ _)
  · exact (hC₁ ⟨l, rfl⟩).trans (le_max_left _ _)

variable {F k}

/-- The assumptions of Eq. (field-L) for both column spins are off the axes. -/
theorem LGeneric.offAxes {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) : F.OffAxes :=
  ⟨(hg true).μ_ne, (hg true).h_ne, fun σ ↦ (hg σ).c_ne⟩

/-- `invL` on the enumerated lattice is a right inverse of `chiLe`. -/
theorem LGeneric.isRightInverse_invL {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) :
    IsRightInverse (F.chiLe k)
      (fun i j ↦ F.invL k u (encL.symm i).1 (encL.symm i).2 (encL.symm j).1 (encL.symm j).2) :=
  F.isRightInverse_chiLe_of k (LGeneric.offAxes hg) (F.bdd_invL u)
    fun σ n t m ↦ chiL_isRightInverse hg σ n t m

/-- **`1 + 𝒦` is invertible** on the domain of Eq. (field-L). -/
theorem LGeneric.isUnit_onePlusKL {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) :
    IsUnit (F.onePlusKL k (LGeneric.offAxes hg)) :=
  (F.opLatL k _).isUnit_onePlusKX_of_isRightInverse _ _
    (F.chiLe_symm k (LGeneric.offAxes hg).μ_ne (LGeneric.offAxes hg).h_ne)
    (LGeneric.isRightInverse_invL hg) fun j ↦ by
      obtain ⟨C, hC⟩ := F.bdd_invL u (encL.symm j).1 (encL.symm j).2
      exact ⟨C, fun i ↦ hC _ _⟩

/-- **`χ_L⁻¹` is `invL`** on the domain of Eq. (field-L). -/
theorem LGeneric.invL_eq_chiInvL {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) : F.invL k u σ n t m = F.chiInvL k (LGeneric.offAxes hg) σ n t m := by
  have h := (F.opLatL k _).eq_chiInvX_of_isRightInverse _ (F.chiDecay_chiLe k (LGeneric.offAxes hg))
    (F.chiLe_symm k (LGeneric.offAxes hg).μ_ne (LGeneric.offAxes hg).h_ne)
    (LGeneric.isRightInverse_invL hg) (fun j ↦ by
      obtain ⟨C, hC⟩ := F.bdd_invL u (encL.symm j).1 (encL.symm j).2
      exact ⟨C, fun i ↦ hC _ _⟩) (encL (σ, n)) (encL (t, m))
  simpa [chiInvL] using h

/-- **The operator vertex is the closed form**: on the domain of Eq. (field-L), `Γ_L` is
`gammaL`. -/
theorem LGeneric.gammaOpL_eq {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) :
    F.gammaOpL k (LGeneric.offAxes hg) σ n t m = F.gammaL k u σ n t m := by
  rw [gammaOpL_eq_chiInvL, ← LGeneric.invL_eq_chiInvL hg, (hg t).gammaL_eq σ n m]

/-- **`χ_L⁻¹` is a right inverse** where `1 + 𝒦` is invertible. -/
theorem chiInvL_isRightInverse (ho : F.OffAxes) (hK : IsUnit (F.onePlusKL k ho)) (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * F.chiInvL k ho p.1 p.2 t m)
      (if σ = t then kron n m else 0) := by
  have h := (F.opLatL k ho.c_ne).chiInvX_isRightInverse _ _ hK (encL (σ, n)) (encL (t, m))
  rw [kron_encL] at h
  refine hasSum_encL_iff.mp ?_
  simpa [chiLe, chiInvL] using h

/-- **`χ_L⁻¹` is a left inverse** where `1 + 𝒦` is invertible. -/
theorem chiInvL_isLeftInverse (ho : F.OffAxes) (hK : IsUnit (F.onePlusKL k ho)) (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun p : Bool × ℤ ↦ F.chiInvL k ho σ n p.1 p.2 * F.chiL p.1 t k p.2 m)
      (if σ = t then kron n m else 0) := by
  have h := (F.opLatL k ho.c_ne).chiInvX_isLeftInverse _ _ (F.chiLe_symm k ho.μ_ne ho.h_ne) hK
    (encL (σ, n)) (encL (t, m))
  rw [kron_encL] at h
  refine hasSum_encL_iff.mp ?_
  simpa [chiLe, chiInvL] using h

/-! ### Finite boxes -/

variable (F k)

/-- The boxes `|2n + 1 + k| ≤ 2M + 1` of both spins, enumerated. -/
def boxL (M : ℕ) : Finset ℤ := (Finset.univ ×ˢ Params.box k M).map encL.toEmbedding

/-- The boxes exhaust `ℤ`. -/
theorem tendsto_boxL : Tendsto (boxL k) atTop atTop := by
  refine tendsto_atTop_finset_of_monotone (fun M M' h j hj ↦ ?_) fun j ↦ ?_
  · simp only [boxL, Finset.mem_map, Finset.mem_product, Finset.mem_univ, true_and,
      Equiv.coe_toEmbedding] at hj ⊢
    obtain ⟨p, hp, rfl⟩ := hj
    exact ⟨p, (Params.mem_box k).mpr (by
      have := (Params.mem_box k).mp hp
      have : (M : ℤ) ≤ M' := by exact_mod_cast h
      omega), rfl⟩
  · obtain ⟨M, hM⟩ := ((Params.tendsto_box k).eventually
      (eventually_ge_atTop {(encL.symm j).2})).exists
    refine ⟨M, ?_⟩
    simp only [boxL, Finset.mem_map, Finset.mem_product, Finset.mem_univ, true_and,
      Equiv.coe_toEmbedding]
    exact ⟨encL.symm j, hM (by simp), by simp⟩

/-- The box vertex `β²[(χ_L|_M)⁻¹ - b⁻¹]` on `Bool × ℤ`, zero outside the box. -/
noncomputable def gammaBoxL (ho : F.OffAxes) (M : ℕ) (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) : ℂ :=
  (F.opLatL k ho.c_ne).gammaBoxX (F.chiLe k) (F.bLe k) (boxL k M) (encL (σ, n)) (encL (τ, m))

/-- **The box vertices converge to `Γ_L`** wherever `1 + 𝒦` is invertible. -/
theorem tendsto_gammaBoxL (ho : F.OffAxes) (hu : IsUnit (F.onePlusKL k ho)) (σ : Bool) (n : ℤ)
    (τ : Bool) (m : ℤ) :
    Tendsto (fun M ↦ F.gammaBoxL k ho M σ n τ m) atTop (𝓝 (F.gammaOpL k ho σ n τ m)) :=
  (F.opLatL k ho.c_ne).tendsto_gammaBoxX _ _ _ _ (tendsto_boxL k) hu _ _

/-- **Finite boxes**: wherever `1 + 𝒦` is invertible, the box matrices `χ_L|_M` are invertible
for all sufficiently large `M`. -/
theorem eventually_isUnit_chiBoxL_det (ho : F.OffAxes) (hu : IsUnit (F.onePlusKL k ho)) :
    ∀ᶠ M in atTop, IsUnit (OpLattice.chiBoxX (F.chiLe k) (boxL k M)).det :=
  (F.opLatL k ho.c_ne).eventually_isUnit_chiBoxX_det _ _ (tendsto_boxL k) hu

variable {F k}

/-- **Boxes on the domain of Eq. (field-L)**: the box vertices converge to the closed form
`gammaL`. -/
theorem LGeneric.tendsto_gammaBoxL {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    Tendsto (fun M ↦ F.gammaBoxL k (LGeneric.offAxes hg) M σ n t m) atTop
      (𝓝 (F.gammaL k u σ n t m)) := by
  rw [← LGeneric.gammaOpL_eq hg]
  exact F.tendsto_gammaBoxL k _ (LGeneric.isUnit_onePlusKL hg) σ n t m

end FParams

end HubbardAtom
