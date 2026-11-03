import HubbardAtomLean.FieldLAxisBounds

/-!
# The longitudinal vertex as an operator, for every `μ`, `h`

Items 6(b)–(c) and the boxes of 6(d) for the longitudinal sector through the axes, by
`ChiOperator.lean` on the lattices `z'` of `opLatLAll` (`FieldLAxisBounds.lean`).
`FieldLOperator.lean` does the same off the axes on the lattices `z`; the definitions and theorems
here carry the suffix `All`. With `𝒟 = diag(-β/z'_σn²)`:

* `FParams.chiL_symm_all`: `χ_L` is symmetric, `χ_στ(νₙ, νₘ) = χ_τσ(νₘ, νₙ)`, for every `μ`, `h`.
  For opposite spins this follows from the input `PL_mul_chiL_opp_all`, as the kernels satisfy
  `K_P(z + v) = ρ` and `K_S(z - v) = σm` also on their special pairs (`kerPL_mul_zL_add_opp`,
  `kerSLs_mul_zL_sub_opp`), and are symmetric (`kerPL_symm`, `kerSLs_not`);
* `FParams.gammaOpLAll`: `Γ_L = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)`, entrywise;
* `FParams.isUnit_onePlusKLAll_iff`: `1 + 𝒦` is invertible iff `χ_L` has no nonzero null vector
  `v` on `Bool × ℤ` with `∑_{σn} |v_σn/z'_σn|² < ∞`; `FParams.exists_bound_of_nullLAll`: such
  null vectors are bounded;
* `FParams.gammaOpLAll_eq_chiInvLAll`: `Γ_L = β²(χ_L⁻¹ - b⁻¹)`, with `χ_L⁻¹` a two-sided inverse
  of `χ_L` where `1 + 𝒦` is invertible (`chiInvLAll_isRightInverse`, `chiInvLAll_isLeftInverse`);
* `FParams.LGeneric.gammaOpLAll_eq`: on the domain of Eq. (field-L), `1 + 𝒦` is invertible
  (`LGeneric.isUnit_onePlusKLAll`) and `Γ_L` is the closed form `gammaL`;
* `FParams.tendsto_gammaBoxLAll`: wherever `1 + 𝒦` is invertible, the vertices of the boxes
  `|2n + 1 + k| ≤ 2M + 1` (both spins) converge to `Γ_L`, on the axes too, and the box matrices are
  eventually invertible (`eventually_isUnit_chiBoxL_det_all`); on the domain of Eq. (field-L) they
  converge to `gammaL` (`LGeneric.tendsto_gammaBoxLAll`);
* `FParams.gammaOpLAll_eq_gammaOpL`: off the axes, `Γ_L` is the operator vertex `gammaOpL` of
  `FieldLOperator.lean`; moving the fixed point changes neither the class nor `Γ_L`
  (`summable_zOpL_iff`, `isUnit_onePlusKLAll_iff_onePlusKL`).
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### `χ_L` is symmetric -/

/-- `K_P(z + v) = ρ` for the pair kernel, on the mirror pairs too, where `z + v = 2μ`. -/
theorem kerPL_mul_zL_add_opp (σ : Bool) (n m : ℤ) :
    F.kerPL k n m * ((F.BL σ).z k n + (F.BL (!σ)).z k m) = F.ρ := by
  by_cases hm : m = mirror k n
  · subst hm
    rw [F.kerPL_mirror, F.zL_add_opp_mirror, ← F.two_mul_mu_mul_κPL]
    push_cast
    ring
  · rw [F.kerPL_of_ne_mirror k σ hm]
    field_simp [F.zL_add_opp_ne_zero_of_ne k σ hm]

/-- `K_S(z - v) = σm` for the spin-flip kernel, at `n = m` too, where `z - v = μ_σ - μ_{-σ}`. -/
theorem kerSLs_mul_zL_sub_opp (σ : Bool) (n m : ℤ) :
    F.kerSLs k σ n m * ((F.BL σ).z k n - (F.BL (!σ)).z k m) = F.mL σ := by
  by_cases hnm : n = m
  · subst hnm
    rw [F.kerSLs_self, F.zL_sub_opp_self, F.mL_eq_mul_κSL]
    push_cast
    ring
  · rw [F.kerSLs_of_ne k σ hnm]
    field_simp [F.zL_sub_opp_ne_zero_of_ne k σ hnm]

/-- The pair kernel is symmetric. -/
theorem kerPL_symm (n m : ℤ) : F.kerPL k m n = F.kerPL k n m := by
  rw [kerPL, kerPL, add_comm (F.Bup.z k m), F.Bdn_z_add_Bup_z, add_comm m n]

/-- The spin-flip kernels of the two rows agree: `K_{S,-σ}(m, n) = K_{S,σ}(n, m)`. -/
theorem kerSLs_not (σ : Bool) (n m : ℤ) : F.kerSLs k (!σ) m n = F.kerSLs k σ n m := by
  by_cases hnm : n = m
  · subst hnm
    rw [F.kerSLs_self, F.kerSLs_self]
  · rw [F.kerSLs_of_ne k (!σ) (Ne.symm hnm), F.kerSLs_of_ne k σ hnm, Bool.not_not, F.mL_not,
      show (F.BL (!σ)).z k m - (F.BL σ).z k n = -((F.BL σ).z k n - (F.BL (!σ)).z k m) by ring]
    push_cast
    rw [neg_div_neg_eq]

/-- **`χ_L` is symmetric**, for every `μ`, `h`: `χ_στ(νₙ, νₘ) = χ_τσ(νₘ, νₙ)`. -/
theorem chiL_symm_all (σ τ : Bool) (n m : ℤ) : F.chiL σ τ k n m = F.chiL τ σ k m n := by
  by_cases hστ : τ = σ
  · subst hστ
    rcases eq_or_ne n m with rfl | hnm
    · rfl
    have hPn := F.PL_BL_ne k τ n
    have hPm := F.PL_BL_ne k τ m
    have h₁ := F.PL_mul_chiL_self k τ n m
    have h₂ := F.PL_mul_chiL_self k τ m n
    rw [kron_of_ne hnm] at h₁
    rw [kron_of_ne (Ne.symm hnm)] at h₂
    rw [(eq_div_iff hPn).mpr ((mul_comm _ _).trans h₁),
      (eq_div_iff hPm).mpr ((mul_comm _ _).trans h₂)]
    field_simp
    ring
  · obtain rfl : τ = !σ := by cases σ <;> cases τ <;> simp_all
    have hPn := F.PL_BL_ne k σ n
    have hPm := F.PL_BL_ne k (!σ) m
    have h₁ := F.PL_mul_chiL_opp_all k σ n m
    have h₂ := F.PL_mul_chiL_opp_all k (!σ) m n
    simp only [Bool.not_not] at h₂
    have hP := F.kerPL_mul_zL_add_opp k σ n m
    have hS := F.kerSLs_mul_zL_sub_opp k σ n m
    rw [F.kerPL_symm, F.kerSLs_not] at h₂
    rw [(eq_div_iff hPn).mpr ((mul_comm _ _).trans h₁),
      (eq_div_iff hPm).mpr ((mul_comm _ _).trans h₂), div_eq_div_iff hPn hPm, add_mul, add_mul,
      div_mul_cancel₀ _ hPm, div_mul_cancel₀ _ hPn]
    have hr' : (F.rL (!σ) : ℂ) = F.ρ + F.mL σ := by rw [F.rL_not σ]; push_cast; ring
    generalize F.kerPL k n m = KP at hP ⊢
    generalize F.kerSLs k σ n m = KS at hS ⊢
    generalize (F.BL σ).z k n = z at *
    generalize (F.BL (!σ)).z k m = v at *
    simp only [hr', F.rL_cast σ, N₀L, N₂L, HL, PL, q₀L, lamL] at hP hS ⊢
    linear_combination (4 * (F.a : ℂ) ^ 2 * (v - z) *
        (v ^ 2 + z ^ 2 + 2 * (F.a ^ 2 - F.wL k ^ 2 / 4) - 4 * F.a ^ 2)) * hP +
      (4 * (F.a : ℂ) ^ 2 * (v + z) *
        (v ^ 2 + z ^ 2 + 2 * (F.a ^ 2 - F.wL k ^ 2 / 4) - 4 * F.a ^ 2)) * hS

/-- `χ_L` on the enumerated lattice is symmetric, for every `μ`, `h`. -/
theorem chiLe_symm_all (i j : ℤ) : F.chiLe k i j = F.chiLe k j i :=
  F.chiL_symm_all k _ _ _ _

/-! ### The operator and the criterion -/

/-- `1 + 𝒦` for `χ_L` on `opLatLAll`, for every `μ`, `h`. -/
noncomputable def onePlusKLAll : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  (F.opLatLAll k).onePlusKX (F.chiLe k) (F.chiDecay_chiLe_all k)

/-- **The longitudinal vertex as an operator for every `μ`, `h`**, entrywise on `Bool × ℤ`:
`Γ_L,(σn)(τm) = β²/(d_σn d_τm) [(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]`, with `Ring.inverse` (zero where
`1 + 𝒦` is not invertible). -/
noncomputable def gammaOpLAll (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) : ℂ :=
  (F.opLatLAll k).gammaOpX (F.chiLe k) (F.bLe k) (F.chiDecay_chiLe_all k)
    (F.bubbleDecay_bLe_all k) (encL (σ, n)) (encL (τ, m))

/-- `χ_L⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, entrywise on `Bool × ℤ`. -/
noncomputable def chiInvLAll (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) : ℂ :=
  (F.opLatLAll k).chiInvX (F.chiLe k) (F.chiDecay_chiLe_all k) (encL (σ, n)) (encL (τ, m))

/-- The class `∑ |vⱼ/z'ⱼ|² < ∞` on the enumerated lattice is the class on `Bool × ℤ`. -/
theorem summable_opLatLAll_iff {v : Bool × ℤ → ℂ} :
    Summable (fun j ↦ ‖v (encL.symm j) / (F.opLatLAll k).z j‖ ^ 2) ↔
      Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2) := by
  rw [← encL.summable_iff]
  simp only [Function.comp_def, Equiv.symm_apply_apply, opLatLAll_z]

/-- **`Γ_L` exists iff `χ_L` has no null vector**, for every `μ`, `h`: `1 + 𝒦` is invertible
iff `χ_L` has no nonzero null vector `v` on `Bool × ℤ` with `∑_{σn} |v_σn/z'_σn|² < ∞`. -/
theorem isUnit_onePlusKLAll_iff :
    IsUnit (F.onePlusKLAll k) ↔ ∀ v : Bool × ℤ → ℂ,
      Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2) →
        (∀ σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) 0) → v = 0 := by
  rw [onePlusKLAll, OpLattice.isUnit_onePlusKX_iff]
  constructor
  · intro h v hv hχ
    have h0 := h (fun j ↦ v (encL.symm j)) ((F.summable_opLatLAll_iff k).mpr hv)
      fun i ↦ by
        obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
        exact (F.hasSum_row_iff k σ n).mpr (hχ σ n)
    funext p
    simpa using congrFun h0 (encL p)
  · intro h w hw hχ
    have h0 := h (fun p ↦ w (encL p))
      ((F.summable_opLatLAll_iff k).mp (by simpa using hw)) fun σ n ↦
        (F.hasSum_row_iff k σ n).mp (by simpa using hχ (encL (σ, n)))
    funext j
    obtain ⟨p, rfl⟩ := encL.surjective j
    exact congrFun h0 p

/-- **Null vectors are bounded**: a null vector `v` of `χ_L` on `Bool × ℤ` with
`∑_{σn} |v_σn/z'_σn|² < ∞` has `|v_σn| ≤ C`. -/
theorem exists_bound_of_nullLAll {v : Bool × ℤ → ℂ}
    (hv : Summable fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2)
    (hχ : ∀ σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) 0) :
    ∃ C, ∀ p, ‖v p‖ ≤ C := by
  obtain ⟨C, hC⟩ := (F.opLatLAll k).exists_bound_of_nullX (F.chiLe k)
    (F.chiDecay_chiLe_all k) (v := fun j ↦ v (encL.symm j))
    ((F.summable_opLatLAll_iff k).mpr hv) fun i ↦ by
      obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
      exact (F.hasSum_row_iff k σ n).mpr (hχ σ n)
  exact ⟨C, fun p ↦ by simpa using hC (encL p)⟩

/-! ### `Γ_L = β²(χ_L⁻¹ - b⁻¹)` -/

/-- **`Γ_L = β²(χ_L⁻¹ - b⁻¹)`**, entrywise, for every `μ`, `h`. Where `1 + 𝒦` is not invertible,
`χ_L⁻¹ = 0`. -/
theorem gammaOpLAll_eq_chiInvLAll (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n τ m =
      F.β ^ 2 * (F.chiInvLAll k σ n τ m - if σ = τ then kron n m / F.bL σ k n else 0) := by
  rw [gammaOpLAll, OpLattice.gammaOpX_eq_chiInvX, kron_encL, bLe_encL]
  split_ifs <;> simp [chiInvLAll]

/-! ### The closed form on the domain of Eq. (field-L) -/

variable {F k}

/-- **`1 + 𝒦` is invertible** on the domain of Eq. (field-L). -/
theorem LGeneric.isUnit_onePlusKLAll {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) :
    IsUnit (F.onePlusKLAll k) :=
  (F.opLatLAll k).isUnit_onePlusKX_of_isRightInverse _ _
    (F.chiLe_symm_all k) (LGeneric.isRightInverse_invL hg) fun j ↦ by
      obtain ⟨C, hC⟩ := F.bdd_invL u (encL.symm j).1 (encL.symm j).2
      exact ⟨C, fun i ↦ hC _ _⟩

/-- **`χ_L⁻¹` is `invL`** on the domain of Eq. (field-L). -/
theorem LGeneric.invL_eq_chiInvLAll {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) : F.invL k u σ n t m = F.chiInvLAll k σ n t m := by
  have h := (F.opLatLAll k).eq_chiInvX_of_isRightInverse _ (F.chiDecay_chiLe_all k)
    (F.chiLe_symm_all k) (LGeneric.isRightInverse_invL hg) (fun j ↦ by
      obtain ⟨C, hC⟩ := F.bdd_invL u (encL.symm j).1 (encL.symm j).2
      exact ⟨C, fun i ↦ hC _ _⟩) (encL (σ, n)) (encL (t, m))
  simpa [chiInvLAll] using h

/-- **The operator vertex is the closed form**: on the domain of Eq. (field-L), `Γ_L` is
`gammaL`. -/
theorem LGeneric.gammaOpLAll_eq {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n t m = F.gammaL k u σ n t m := by
  rw [gammaOpLAll_eq_chiInvLAll, ← LGeneric.invL_eq_chiInvLAll hg, (hg t).gammaL_eq σ n m]

/-- **`χ_L⁻¹` is a right inverse** where `1 + 𝒦` is invertible. -/
theorem chiInvLAll_isRightInverse (hK : IsUnit (F.onePlusKLAll k)) (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * F.chiInvLAll k p.1 p.2 t m)
      (if σ = t then kron n m else 0) := by
  have h := (F.opLatLAll k).chiInvX_isRightInverse _ _ hK (encL (σ, n)) (encL (t, m))
  rw [kron_encL] at h
  refine hasSum_encL_iff.mp ?_
  simpa [chiLe, chiInvLAll] using h

/-- **`χ_L⁻¹` is a left inverse** where `1 + 𝒦` is invertible. -/
theorem chiInvLAll_isLeftInverse (hK : IsUnit (F.onePlusKLAll k)) (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun p : Bool × ℤ ↦ F.chiInvLAll k σ n p.1 p.2 * F.chiL p.1 t k p.2 m)
      (if σ = t then kron n m else 0) := by
  have h := (F.opLatLAll k).chiInvX_isLeftInverse _ _ (F.chiLe_symm_all k) hK
    (encL (σ, n)) (encL (t, m))
  rw [kron_encL] at h
  refine hasSum_encL_iff.mp ?_
  simpa [chiLe, chiInvLAll] using h

/-! ### Finite boxes -/

variable (F k)

/-- The box vertex `β²[(χ_L|_M)⁻¹ - b⁻¹]` on `Bool × ℤ`, zero outside the box. -/
noncomputable def gammaBoxLAll (M : ℕ) (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) : ℂ :=
  (F.opLatLAll k).gammaBoxX (F.chiLe k) (F.bLe k) (boxL k M) (encL (σ, n)) (encL (τ, m))

/-- **The box vertices converge to `Γ_L`** wherever `1 + 𝒦` is invertible. -/
theorem tendsto_gammaBoxLAll (hu : IsUnit (F.onePlusKLAll k)) (σ : Bool) (n : ℤ)
    (τ : Bool) (m : ℤ) :
    Tendsto (fun M ↦ F.gammaBoxLAll k M σ n τ m) atTop (𝓝 (F.gammaOpLAll k σ n τ m)) :=
  (F.opLatLAll k).tendsto_gammaBoxX _ _ _ _ (tendsto_boxL k) hu _ _

/-- **Finite boxes**: wherever `1 + 𝒦` is invertible, the box matrices `χ_L|_M` are invertible
for all sufficiently large `M`. -/
theorem eventually_isUnit_chiBoxL_det_all (hu : IsUnit (F.onePlusKLAll k)) :
    ∀ᶠ M in atTop, IsUnit (OpLattice.chiBoxX (F.chiLe k) (boxL k M)).det :=
  (F.opLatLAll k).eventually_isUnit_chiBoxX_det _ _ (tendsto_boxL k) hu

variable {F k}

/-- **Boxes on the domain of Eq. (field-L)**: the box vertices converge to the closed form
`gammaL`. -/
theorem LGeneric.tendsto_gammaBoxLAll {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    Tendsto (fun M ↦ F.gammaBoxLAll k M σ n t m) atTop
      (𝓝 (F.gammaL k u σ n t m)) := by
  rw [← LGeneric.gammaOpLAll_eq hg]
  exact F.tendsto_gammaBoxLAll k (LGeneric.isUnit_onePlusKLAll hg) σ n t m

/-! ### Agreement with `FieldLOperator.lean` off the axes -/

variable (F k)

/-- The classes of `z'` and `z` agree: they differ only at the fixed point. -/
theorem summable_zOpL_iff {v : Bool × ℤ → ℂ} :
    Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2) ↔
      Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).z k p.2‖ ^ 2) := by
  have h : (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2) =ᶠ[cofinite]
      fun p ↦ ‖v p / (F.BL p.1).z k p.2‖ ^ 2 := by
    refine Filter.eventually_cofinite.mpr
      ((Set.finite_univ.prod (Set.finite_singleton ((-k - 1) / 2))).subset ?_)
    rintro ⟨σ, j⟩ hj
    refine ⟨Set.mem_univ _, ?_⟩
    by_contra hne
    have hj' : mirror k j ≠ j := fun h ↦ hne (Set.mem_singleton_iff.mpr (eq_of_mirror_eq_self h))
    exact hj (by simp only [(F.BL σ).zOp_of_ne k hj'])
  exact ⟨fun hs ↦ hs.congr_cofinite h, fun hs ↦ hs.congr_cofinite h.symm⟩

/-- Off the axes, `1 + 𝒦` on `z'` is invertible iff `1 + 𝒦` on `z` is. -/
theorem isUnit_onePlusKLAll_iff_onePlusKL (ho : F.OffAxes) :
    IsUnit (F.onePlusKLAll k) ↔ IsUnit (F.onePlusKL k ho) := by
  rw [F.isUnit_onePlusKLAll_iff, F.isUnit_onePlusKL_iff k ho]
  exact forall_congr' fun v ↦ imp_congr (F.summable_zOpL_iff k) Iff.rfl

/-- **Off the axes, `Γ_L` is `gammaOpL`**: the vertex does not depend on the scaling `𝒟`. Where
`1 + 𝒦` is invertible both are limits of the same box vertices; elsewhere both are `-β²δ/b`. -/
theorem gammaOpLAll_eq_gammaOpL (ho : F.OffAxes) (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n τ m = F.gammaOpL k ho σ n τ m := by
  by_cases hu : IsUnit (F.onePlusKLAll k)
  · exact tendsto_nhds_unique (F.tendsto_gammaBoxLAll k hu σ n τ m)
      (F.tendsto_gammaBoxL k ho ((F.isUnit_onePlusKLAll_iff_onePlusKL k ho).mp hu) σ n τ m)
  · have hu' : ¬IsUnit (F.onePlusKL k ho) :=
      fun h ↦ hu ((F.isUnit_onePlusKLAll_iff_onePlusKL k ho).mpr h)
    have h₁ : Ring.inverse ((F.opLatLAll k).onePlusKX (F.chiLe k) (F.chiDecay_chiLe_all k)) = 0 :=
      Ring.inverse_non_unit _ hu
    have h₂ : Ring.inverse ((F.opLatL k ho.c_ne).onePlusKX (F.chiLe k) (F.chiDecay_chiLe k ho)) =
        0 := Ring.inverse_non_unit _ hu'
    simp [gammaOpLAll_eq_chiInvLAll, gammaOpL_eq_chiInvL, chiInvLAll, chiInvL,
      OpLattice.chiInvX, h₁, h₂]

end FParams

end HubbardAtom
