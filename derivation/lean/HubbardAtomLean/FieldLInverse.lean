import HubbardAtomLean.FieldLImage

/-!
# The longitudinal inverse

The inverse of the coupled longitudinal susceptibility `χ_L` on `Bool × ℤ`, Eq. (field-L) of the
note (App. F.6.4): the column `(t, m)`, `v = z_m` on the lattice of the spin `t`, is the dense part
`denseL` minus the modes `modeL` of the four roots `u_j` of `E₈`, with the coefficients
`𝔠ⱼ/(v² - u_j²)` of the completion (`FieldLCompl.lean`).

* `FParams.LGeneric`: the assumptions for the column spin `t`: `U ≠ 0`, `μ ≠ 0`, `h ≠ 0`,
  `μ_t ≠ 0`, four roots `u_j` of `E₈` with distinct `u_j²`, `E₈(zₘ) ≠ 0` on the lattice of `t`,
  and at each non-atomic root (all but `u_j² = a²` at `k = 0`) `u_j` off both lattices and
  `K_t(u_j) ≠ 0`.
* `FParams.invL`: the inverse; `LGeneric.hasSum_row_self`, `LGeneric.hasSum_row_opp`: its column
  `(t, m)` under the rows of the spins `t`, `-t`; `chiL_isRightInverse`: `χ_L invL = 1`.
-/

open Complex

namespace HubbardAtom

private theorem sum_bool_eq {M : Type*} [AddCommMonoid M] (f : Bool → M) (t : Bool) :
    ∑ τ, f τ = f t + f (!t) := by
  cases t <;> simp [add_comm]

namespace FParams

variable (F : FParams) (k : ℤ)

theorem DL_ne {t : Bool} (hc : F.cL t ≠ 0) : (F.DL t : ℂ) ≠ 0 := by
  have hE : Real.exp (2 * F.β * F.cL t) ≠ 1 := by
    rw [Ne, Real.exp_eq_one_iff]
    exact mul_ne_zero (mul_ne_zero two_ne_zero F.β_pos.ne') hc
  have h0 : 0 < F.p₀ * F.pL (!t) := mul_pos F.p₀_pos (F.pL_pos (!t))
  rw [ofReal_ne_zero, DL, F.p₂_mul_pL t,
    show F.p₀ * F.pL (!t) * Real.exp (2 * F.β * F.cL t) - F.p₀ * F.pL (!t) =
      F.p₀ * F.pL (!t) * (Real.exp (2 * F.β * F.cL t) - 1) by ring]
  exact mul_ne_zero h0.ne' (sub_ne_zero.mpr hE)

theorem q₀L_ne (ha : (F.a : ℂ) ≠ 0) : F.q₀L k ≠ 0 := by
  have e : F.q₀L k = ((F.a ^ 2 + F.ω k ^ 2 / 4 : ℝ) : ℂ) := by
    simp only [q₀L, wL, mul_pow, I_sq]
    push_cast
    ring
  have ha' : F.a ≠ 0 := by exact_mod_cast ha
  rw [e, ofReal_ne_zero]
  positivity

/-! ### The assumptions and the inverse -/

/-- The assumptions of the longitudinal inverse for the column spin `t`, with four roots `u_j` of
`E₈`. -/
structure LGeneric (t : Bool) (u : Fin 4 → ℂ) : Prop where
  a_ne : (F.a : ℂ) ≠ 0
  μ_ne : F.μ ≠ 0
  h_ne : F.h ≠ 0
  c_ne : F.cL t ≠ 0
  root : ∀ j, F.E8L k (u j) = 0
  sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0
  off : ∀ j, ¬F.Atomic k (u j) → ∀ σ l, (F.BL σ).z k l ≠ u j ∧ (F.BL σ).z k l ≠ -u j
  K_ne : ∀ j, ¬F.Atomic k (u j) → F.KL k t t (u j) ≠ 0

variable {F k}

/-- The roots are off the lattice of `t`, also squared: the non-atomic ones by `off`, the atomic one
`u² = a²` (`k = 0`) since `Im zₘ = νₘ ≠ 0`. -/
theorem LGeneric.lat_ne {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) (m : ℤ) (j : Fin 4) :
    (F.BL t).z k m ^ 2 - u j ^ 2 ≠ 0 := by
  rw [sq_sub_sq]
  by_cases hj : F.Atomic k (u j)
  · obtain ⟨rfl, hu⟩ := hj
    have hν := (F.BL t).ν_ne_zero m
    have hz : ((F.BL t).z 0 m).im = (F.BL t).ν m := by
      simp [Params.z, Params.x, Params.w, Params.ω]
    have hu' : (u j).im = 0 := by
      rcases sq_eq_sq_iff_eq_or_eq_neg.1 hu with h | h <;> simp [h]
    refine mul_ne_zero ?_ ?_ <;> intro h <;> apply hν <;>
      simpa [hz, hu'] using congrArg Complex.im h
  · obtain ⟨h₁, h₂⟩ := hg.off j hj t m
    exact mul_ne_zero (fun h ↦ h₂ (eq_neg_of_add_eq_zero_left h)) (sub_ne_zero.2 h₁)

theorem LGeneric.colNe {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) : F.ColNe k t :=
  ⟨hg.a_ne, F.rho_ne_zeroT hg.μ_ne, F.mL_ne_zero hg.h_ne t, F.DL_ne hg.c_ne, F.q₀L_ne k hg.a_ne⟩

theorem LGeneric.roots {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) : F.Roots k t u :=
  ⟨hg.root, hg.sep, fun j hj ↦ ModeRel.of_off F k hg.c_ne (fun σ l ↦ (hg.off j hj σ l).1)
    (fun σ l ↦ (hg.off j hj σ l).2), hg.K_ne⟩

theorem LGeneric.E8L_ne {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) (m : ℤ) :
    F.E8L k ((F.BL t).z k m) ≠ 0 := by
  have hv := hg.lat_ne m
  rw [F.E8L_eq_prod k hg.root hg.sep]
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero (hv 0) (hv 1)) (hv 2)) (hv 3)

/-- The completion in the row of the column spin, for all `k`. -/
theorem LGeneric.compl_s {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) (m : ℤ) :
    (F.Ds0 k t ((F.BL t).z k m) = ∑ j, F.cMode k t u j * F.Ms0 k t (u j) /
        ((F.BL t).z k m ^ 2 - u j ^ 2)) ∧
      (F.Ds1 k t ((F.BL t).z k m) = ∑ j, F.cMode k t u j * F.Ms1 k t (u j) /
        ((F.BL t).z k m ^ 2 - u j ^ 2)) ∧
      F.Ds2 k t ((F.BL t).z k m) = ∑ j, F.cMode k t u j * F.Ms2 k t (u j) /
        ((F.BL t).z k m ^ 2 - u j ^ 2) := by
  by_cases hk : k = 0
  · exact ⟨F.compl_s0_Z k hk hg.colNe hg.roots (hg.lat_ne m),
      F.compl_s1_Z k hk hg.colNe hg.roots (hg.lat_ne m),
      F.compl_s2_Z k hk hg.colNe hg.roots (hg.lat_ne m)⟩
  · exact ⟨F.compl_s0_W k hk hg.colNe hg.roots (hg.lat_ne m),
      F.compl_s1_W k hk hg.colNe hg.roots (hg.lat_ne m),
      F.compl_s2_W k hk hg.colNe hg.roots (hg.lat_ne m)⟩

/-- The completion in the row of `-t`, for all `k`. -/
theorem LGeneric.compl_o {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) (m : ℤ) :
    (F.Do0 k t ((F.BL t).z k m) = ∑ j, F.cMode k t u j * F.Mo0 k t (u j) /
        ((F.BL t).z k m ^ 2 - u j ^ 2)) ∧
      (F.Do1 k t ((F.BL t).z k m) = ∑ j, F.cMode k t u j * F.Mo1 k t (u j) /
        ((F.BL t).z k m ^ 2 - u j ^ 2)) ∧
      F.Do2 k t ((F.BL t).z k m) = ∑ j, F.cMode k t u j * F.Mo2 k t (u j) /
        ((F.BL t).z k m ^ 2 - u j ^ 2) := by
  by_cases hk : k = 0
  · exact ⟨F.compl_o0_Z k hk hg.colNe hg.roots (hg.lat_ne m),
      F.compl_o1_Z k hk hg.colNe hg.roots (hg.lat_ne m),
      F.compl_o2_Z k hk hg.colNe hg.roots (hg.lat_ne m)⟩
  · exact ⟨F.compl_o0_W k hk hg.colNe hg.roots (hg.lat_ne m),
      F.compl_o1_W k hk hg.colNe hg.roots (hg.lat_ne m),
      F.compl_o2_W k hk hg.colNe hg.roots (hg.lat_ne m)⟩

variable (F k)

/-- **The longitudinal inverse**: the entry `((τ, l), (t, m))`, `v = z_m` on the lattice of `t`,
is the dense part minus the modes of the roots `u_j` with the coefficients `𝔠ⱼ/(v² - u_j²)`. -/
noncomputable def invL (u : Fin 4 → ℂ) (τ : Bool) (l : ℤ) (t : Bool) (m : ℤ) : ℂ :=
  F.denseL k t τ m l -
    ∑ j, F.cMode k t u j / ((F.BL t).z k m ^ 2 - u j ^ 2) * F.modeL k t τ (u j) l

variable {F k}

/-- **The column `(t, m)` in the rows of the spin `t`**: `∑_{τ, l} χ_{tτ}(n, l) invL = δ_nm`. -/
theorem LGeneric.hasSum_row_self {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) (n m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.chiL t τ k n l * F.invL k u τ l t m) (kron n m) := by
  have hcol := hg.colNe
  have hR := hg.roots
  have hv := hg.lat_ne m
  have hPn := F.PL_BL_ne k t n
  have hD := F.hasSum_dense_s k hg.a_ne hg.μ_ne hg.h_ne hg.c_ne hcol (n := n) (hg.E8L_ne m)
  have hM : ∀ j ∈ Finset.univ, HasSum (fun l ↦ F.cMode k t u j / ((F.BL t).z k m ^ 2 - u j ^ 2) *
      (F.PL k ((F.BL t).z k n) * F.chiL t t k n l * F.modeL k t t (u j) l +
        F.PL k ((F.BL t).z k n) * F.chiL t (!t) k n l * F.modeL k t (!t) (u j) l))
      (F.cMode k t u j / ((F.BL t).z k m ^ 2 - u j ^ 2) * (F.Ms0 k t (u j) +
        (F.BL t).z k n * F.Ms1 k t (u j) + (F.BL t).z k n ^ 2 * F.Ms2 k t (u j))) := by
    intro j _
    by_cases ha : F.Atomic k (u j)
    · simp only [cMode, ha, ↓reduceIte, zero_div, zero_mul]
      exact hasSum_zero
    · exact (F.hasSum_mode_s k hg.a_ne hg.μ_ne hg.h_ne hcol (hR.rel j ha) (hg.root j)
        (fun σ l ↦ (hg.off j ha σ l).1) (fun σ l ↦ (hg.off j ha σ l).2) n).mul_left _
  have hT := (hD.sub (hasSum_sum hM)).mul_left (F.PL k ((F.BL t).z k n))⁻¹
  obtain ⟨c0, c1, c2⟩ := hg.compl_s m
  convert hT using 1
  · funext l
    rw [sum_bool_eq _ t]
    simp only [invL, Fin.sum_univ_four]
    field_simp
    ring
  · rw [c0, c1, c2]
    simp only [Fin.sum_univ_four]
    rcases eq_or_ne n m with rfl | hnm
    · have := hv 0; have := hv 1; have := hv 2; have := hv 3
      simp only [kron_self]
      field_simp
      ring
    · simp only [kron_of_ne hnm]
      ring

/-- **The column `(t, m)` in the rows of the spin `-t`**: `∑_{τ, l} χ_{-t,τ}(n, l) invL = 0`. -/
theorem LGeneric.hasSum_row_opp {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) (n m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.chiL (!t) τ k n l * F.invL k u τ l t m) 0 := by
  have hcol := hg.colNe
  have hR := hg.roots
  have hPn := F.PL_BL_ne k (!t) n
  have hD := F.hasSum_dense_o k hg.a_ne hg.μ_ne hg.h_ne hg.c_ne hcol (n := n) (m := m)
  have hM : ∀ j ∈ Finset.univ, HasSum (fun l ↦ F.cMode k t u j / ((F.BL t).z k m ^ 2 - u j ^ 2) *
      (F.PL k ((F.BL (!t)).z k n) * F.chiL (!t) (!t) k n l * F.modeL k t (!t) (u j) l +
        F.PL k ((F.BL (!t)).z k n) * F.chiL (!t) t k n l * F.modeL k t t (u j) l))
      (F.cMode k t u j / ((F.BL t).z k m ^ 2 - u j ^ 2) * (F.Mo0 k t (u j) +
        (F.BL (!t)).z k n * F.Mo1 k t (u j) + (F.BL (!t)).z k n ^ 2 * F.Mo2 k t (u j))) := by
    intro j _
    by_cases ha : F.Atomic k (u j)
    · simp only [cMode, ha, ↓reduceIte, zero_div, zero_mul]
      exact hasSum_zero
    · exact (F.hasSum_mode_o k hg.a_ne hg.μ_ne hg.h_ne hcol (hR.rel j ha)
        (fun σ l ↦ (hg.off j ha σ l).1) (fun σ l ↦ (hg.off j ha σ l).2) n).mul_left _
  have hT := (hD.sub (hasSum_sum hM)).mul_left (F.PL k ((F.BL (!t)).z k n))⁻¹
  obtain ⟨c0, c1, c2⟩ := hg.compl_o m
  convert hT using 1
  · funext l
    rw [sum_bool_eq _ (!t), Bool.not_not]
    simp only [invL, Fin.sum_univ_four]
    field_simp
    ring
  · rw [c0, c1, c2]
    simp only [Fin.sum_univ_four]
    ring

/-- **The longitudinal inverse**, right inverse: `∑_{τ, l} χ_{στ}(n, l) invL_{(τ, l), (t, m)} =
δ_στ δ_nm`, under the assumptions for both column spins. -/
theorem chiL_isRightInverse {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * F.invL k u τ l t m)
      (if σ = t then kron n m else 0) := by
  by_cases hσ : σ = t
  · subst hσ
    simpa using (hg σ).hasSum_row_self n m
  · obtain rfl : σ = !t := by cases σ <;> cases t <;> simp_all
    simpa using (hg t).hasSum_row_opp n m

end FParams

end HubbardAtom
