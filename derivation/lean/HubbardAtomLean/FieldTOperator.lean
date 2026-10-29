import HubbardAtomLean.FieldTBounds

/-!
# The transverse vertex as an operator

Items 6(b)–(d) for the transverse sector, by `ChiOperator.lean` and the bounds of
`FieldTBounds.lean`. For `μ ≠ 0` and `W = iω_k - 2h ≠ 0`, on the lattice `zₙ = μ + i(νₙ + ω_k/2)`
with `𝒟 = diag(-β/zₙ²)`:

* `FParams.gammaOpT`: `Γ_T↑ = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)`, entrywise, with
  `χ_T↑ = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)`, `𝒦` Hilbert–Schmidt, and `b_T↑ = 𝒟^(1/2)(1 + 𝒦₀)𝒟^(1/2)`;
* `FParams.isUnit_onePlusKT_iff`: `1 + 𝒦` is invertible iff `χ_T↑` has no nonzero null vector `v`
  with `∑ₙ |vₙ/zₙ|² < ∞`; `FParams.exists_bound_of_nullT`: such null vectors are bounded;
* `FParams.gammaOpT_eq_chiInvT`: `Γ_T↑ = β²(χ_T↑⁻¹ - b_T↑⁻¹)` with
  `χ_T↑⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, which, where `1 + 𝒦` is invertible, is a two-sided
  inverse of `χ_T↑` (`chiInvT_isRightInverse`, `chiInvT_isLeftInverse`), the only right inverse
  with columns in the class (`eq_chiInvT_of_hasSum`);
* `FParams.TGeneric.gammaOpT_eq`: on the domain of Eq. (field-T), `1 + 𝒦` is invertible and
  `Γ_T↑` is the closed form `gammaTUp`, as the columns of `invTUp` tend to `-2a/β²`
  (`tendsto_invTUp`); `TGeneric.invTUp_eq_chiInvT`;
* `FParams.tendsto_gammaBoxT`: wherever `1 + 𝒦` is invertible, the box matrices `χ_T↑|_M` on
  `|2n + 1 + k| ≤ 2M + 1` are invertible for large `M` (`eventually_isUnit_chiBoxT_det`), and the
  box vertices `β²[(χ_T↑|_M)⁻¹ - b_T↑⁻¹]` converge to `Γ_T↑`, on the domain of Eq. (field-T) to
  `gammaTUp` (`TGeneric.tendsto_gammaBoxT`).
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The operator and the criterion -/

/-- `1 + 𝒦` for `χ_T↑` on `ℓ²(ℤ)`. -/
noncomputable def onePlusKT (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  (F.opLatT k hμ).onePlusKX (F.chiTUp k) (F.chiDecay_chiTUp k hμ hW)

/-- **The transverse vertex as an operator**: `Γ_T↑,nm = β²/(dₙdₘ) [(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]ₙₘ`,
with `Ring.inverse` (zero where `1 + 𝒦` is not invertible). -/
noncomputable def gammaOpT (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) (n m : ℤ) : ℂ :=
  (F.opLatT k hμ).gammaOpX (F.chiTUp k) (F.bTUp k) (F.chiDecay_chiTUp k hμ hW)
    (F.bubbleDecay_bTUp k hμ) n m

/-- `χ_T↑⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, entrywise, with `Ring.inverse` as in `gammaOpT`. -/
noncomputable def chiInvT (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) (n m : ℤ) : ℂ :=
  (F.opLatT k hμ).chiInvX (F.chiTUp k) (F.chiDecay_chiTUp k hμ hW) n m

/-- **`Γ_T↑` exists iff `χ_T↑` has no null vector**: `1 + 𝒦` is invertible iff `χ_T↑` has no
nonzero null vector `v` with `∑ₙ |vₙ/zₙ|² < ∞`. -/
theorem isUnit_onePlusKT_iff (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) :
    IsUnit (F.onePlusKT k hμ hW) ↔ ∀ v : ℤ → ℂ, Summable (fun n ↦ ‖v n / F.zT k n‖ ^ 2) →
      (∀ n, HasSum (fun m ↦ F.chiTUp k n m * v m) 0) → v = 0 :=
  (F.opLatT k hμ).isUnit_onePlusKX_iff _ _

/-- **Null vectors are bounded**: a null vector `v` of `χ_T↑` with `∑ₙ |vₙ/zₙ|² < ∞` has
`|vₙ| ≤ C`. -/
theorem exists_bound_of_nullT (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / F.zT k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun m ↦ F.chiTUp k n m * v m) 0) : ∃ C, ∀ n, ‖v n‖ ≤ C :=
  (F.opLatT k hμ).exists_bound_of_nullX _ (F.chiDecay_chiTUp k hμ hW) hv hχ

/-! ### `Γ_T↑ = β²(χ_T↑⁻¹ - b_T↑⁻¹)` -/

/-- **`Γ_T↑ = β²(χ_T↑⁻¹ - b_T↑⁻¹)`**, entrywise, for every `μ ≠ 0`, `W ≠ 0`. Where `1 + 𝒦` is
not invertible, `χ_T↑⁻¹ = 0`, so `Γ_T↑ = -β²δₙₘ/b_T↑(νₙ)` there. -/
theorem gammaOpT_eq_chiInvT (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) (n m : ℤ) :
    F.gammaOpT k hμ hW n m = F.β ^ 2 * (F.chiInvT k hμ hW n m - kron n m / F.bTUp k n) :=
  (F.opLatT k hμ).gammaOpX_eq_chiInvX _ _ _ _ n m

/-- **`χ_T↑⁻¹` is a right inverse** where `1 + 𝒦` is invertible. -/
theorem chiInvT_isRightInverse (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0)
    (hK : IsUnit (F.onePlusKT k hμ hW)) : IsRightInverse (F.chiTUp k) (F.chiInvT k hμ hW) :=
  (F.opLatT k hμ).chiInvX_isRightInverse _ _ hK

/-- **`χ_T↑⁻¹` is a left inverse** where `1 + 𝒦` is invertible, as `χ_T↑` is symmetric
(`chiTUp_symm`). -/
theorem chiInvT_isLeftInverse (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0)
    (hK : IsUnit (F.onePlusKT k hμ hW)) : IsLeftInverse (F.chiTUp k) (F.chiInvT k hμ hW) :=
  (F.opLatT k hμ).chiInvX_isLeftInverse _ _ (F.chiTUp_symm k hμ hW) hK

/-- **`χ_T↑⁻¹` is the only right inverse in the class**: where `1 + 𝒦` is invertible, a `v` with
`∑ₙ |vₙ/zₙ|² < ∞` and `∑ₗ χ_T↑,nl vₗ = δₙₘ` is column `m` of `χ_T↑⁻¹`. -/
theorem eq_chiInvT_of_hasSum (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0)
    (hK : IsUnit (F.onePlusKT k hμ hW)) {m : ℤ} {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / F.zT k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun l ↦ F.chiTUp k n l * v l) (kron n m)) (n : ℤ) :
    v n = F.chiInvT k hμ hW n m :=
  (F.opLatT k hμ).eq_chiInvX_of_hasSum _ _ hK hv hχ n

/-! ### The closed form on the domain of Eq. (field-T) -/

variable {k} in
/-- **The columns of `invTUp` tend to `-2a/β²`** as `|n| → ∞`. -/
theorem tendsto_invTUp (u₁ u₂ : ℂ) (m : ℤ) :
    Tendsto (fun n ↦ F.invTUp k u₁ u₂ n m) cofinite (𝓝 (2 * (-F.a) / F.β ^ 2)) := by
  have hk : Tendsto (fun n ↦ kron n m) cofinite (𝓝 0) :=
    tendsto_const_nhds.congr' ((eventually_cofinite_ne m).mono fun n hn ↦ (kron_of_ne hn).symm)
  have hi : ∀ x, Tendsto (fun n ↦ (F.zT k n - x)⁻¹) cofinite (𝓝 0) := fun x ↦
    tendsto_inv₀_cobounded.comp
      (tendsto_norm_atTop_iff_cobounded.mp ((F.lat F.μ).tendsto_norm_z_sub k x))
  have hφ : ∀ u c, Tendsto (fun n ↦ c * (F.β * F.HT k (-u) * (F.zT k n - u)⁻¹ +
      F.lamT * F.αT k u * (F.zT k n - -u)⁻¹)) cofinite (𝓝 0) := fun u c ↦ by
    simpa using (((hi u).const_mul _).add ((hi (-u)).const_mul _)).const_mul c
  have h := ((((hk.const_mul (F.DvT k (F.zT k m))).add
    ((hi (-F.zT k m)).const_mul (F.EvT k (F.zT k m)))).add_const (2 * (-F.a) / F.β ^ 2 : ℂ)).add
    (hφ u₁ (F.cmodeT k u₁ u₂ (F.zT k m)))).add (hφ u₂ (F.cmodeT k u₂ u₁ (F.zT k m)))
  simp only [mul_zero, add_zero, zero_add] at h
  refine h.congr fun n ↦ ?_
  simp only [invTUp, modeT, div_eq_mul_inv, sub_neg_eq_add]

variable {k} in
/-- The columns of `invTUp` are bounded. -/
theorem bdd_invTUp (u₁ u₂ : ℂ) (m : ℤ) : ∃ C, ∀ n, ‖F.invTUp k u₁ u₂ n m‖ ≤ C := by
  obtain ⟨C, hC⟩ := (F.tendsto_invTUp u₁ u₂ m).norm.bddAbove_range_of_cofinite
  exact ⟨C, fun n ↦ hC ⟨n, rfl⟩⟩

variable {F k}

/-- **`1 + 𝒦` is invertible** on the domain of Eq. (field-T), as `χ_T↑` is symmetric and has the
right inverse `invTUp` with bounded columns. -/
theorem TGeneric.isUnit_onePlusKT {u₁ u₂ : ℂ} (h : F.TGeneric k u₁ u₂) :
    IsUnit (F.onePlusKT k h.μ_ne h.W_ne) :=
  (F.opLatT k h.μ_ne).isUnit_onePlusKX_of_isRightInverse _ _ (F.chiTUp_symm k h.μ_ne h.W_ne)
    h.chiTUp_isRightInverse (F.bdd_invTUp u₁ u₂)

/-- **`χ_T↑⁻¹` is `invTUp`** on the domain of Eq. (field-T). -/
theorem TGeneric.invTUp_eq_chiInvT {u₁ u₂ : ℂ} (h : F.TGeneric k u₁ u₂) (n m : ℤ) :
    F.invTUp k u₁ u₂ n m = F.chiInvT k h.μ_ne h.W_ne n m :=
  (F.opLatT k h.μ_ne).eq_chiInvX_of_isRightInverse _ _ (F.chiTUp_symm k h.μ_ne h.W_ne)
    h.chiTUp_isRightInverse (F.bdd_invTUp u₁ u₂) n m

/-- **The operator vertex is the closed form**: on the domain of Eq. (field-T),
`Γ_T↑ = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)` is `gammaTUp`. -/
theorem TGeneric.gammaOpT_eq {u₁ u₂ : ℂ} (h : F.TGeneric k u₁ u₂) (n m : ℤ) :
    F.gammaOpT k h.μ_ne h.W_ne n m = F.gammaTUp k u₁ u₂ n m := by
  rw [← h.gammaTUp_eq n m]
  exact (F.opLatT k h.μ_ne).gammaOpX_eq_of_isRightInverse _ _ _ _
    (F.chiTUp_symm k h.μ_ne h.W_ne) h.chiTUp_isRightInverse (F.bdd_invTUp u₁ u₂) n m

/-! ### Finite boxes -/

variable (F k)

/-- The box vertex `β²[(χ_T↑|_s)⁻¹ - b_T↑⁻¹]ₙₘ` for `n, m ∈ s`, zero outside the box. -/
noncomputable def gammaBoxT (hμ : F.μ ≠ 0) (s : Finset ℤ) (n m : ℤ) : ℂ :=
  (F.opLatT k hμ).gammaBoxX (F.chiTUp k) (F.bTUp k) s n m

/-- **Finite boxes**: wherever `1 + 𝒦` is invertible, the box matrices `χ_T↑|_M` on
`|2n + 1 + k| ≤ 2M + 1` are invertible for all sufficiently large `M`. -/
theorem eventually_isUnit_chiBoxT_det (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0)
    (hu : IsUnit (F.onePlusKT k hμ hW)) :
    ∀ᶠ M in atTop, IsUnit (OpLattice.chiBoxX (F.chiTUp k) (Params.box k M)).det :=
  (F.opLatT k hμ).eventually_isUnit_chiBoxX_det _ _ (Params.tendsto_box k) hu

/-- **The box vertices converge to `Γ_T↑`** wherever `1 + 𝒦` is invertible. -/
theorem tendsto_gammaBoxT (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) (hu : IsUnit (F.onePlusKT k hμ hW))
    (n m : ℤ) :
    Tendsto (fun M ↦ F.gammaBoxT k hμ (Params.box k M) n m) atTop
      (𝓝 (F.gammaOpT k hμ hW n m)) :=
  (F.opLatT k hμ).tendsto_gammaBoxX _ _ _ _ (Params.tendsto_box k) hu n m

variable {F k}

/-- **Boxes on the domain of Eq. (field-T)**: the box vertices converge to the closed form
`gammaTUp`. -/
theorem TGeneric.tendsto_gammaBoxT {u₁ u₂ : ℂ} (h : F.TGeneric k u₁ u₂) (n m : ℤ) :
    Tendsto (fun M ↦ F.gammaBoxT k h.μ_ne (Params.box k M) n m) atTop
      (𝓝 (F.gammaTUp k u₁ u₂ n m)) := by
  rw [← h.gammaOpT_eq]
  exact F.tendsto_gammaBoxT k h.μ_ne h.W_ne h.isUnit_onePlusKT n m

end FParams

end HubbardAtom
