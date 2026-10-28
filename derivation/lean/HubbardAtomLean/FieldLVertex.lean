import HubbardAtomLean.FieldLInverse

/-!
# The coupled longitudinal vertex

* `FParams.bL_eq`: the bubble, `b_σ(νₙ) = -β((z + ar_σ)² + ω²/4)/P(z)`.
* `FParams.gammaL`, `LGeneric.gammaL_eq`: **the longitudinal vertex**,
  `β²(χ_L⁻¹ - b⁻¹) = Γ_L` of Eq. (field-L), with `χ_L⁻¹ = invL`; `U = 2a`, `w = iω`.
* `FParams.chiL_symm`: `χ_L` is symmetric, `χ_στ(ν, ν') = χ_τσ(ν', ν)`, off the axes;
  `chiL_isLeftInverse_transpose`: hence the transpose of `invL` is a left inverse.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The bubble -/

/-- `(z + ar_σ)² - w²/4 = (x + ar_σ)(x + w + ar_σ) ≠ 0`, `x = z - w/2` nonreal. -/
theorem bL_num_ne (σ : Bool) (n : ℤ) :
    ((F.BL σ).z k n + F.a * F.rL σ) ^ 2 - F.wL k ^ 2 / 4 ≠ 0 := by
  have h₁ := (F.BL σ).x_add_ofReal_ne_zero n (F.a * F.rL σ)
  have h₂ := (F.BL σ).x_add_ofReal_ne_zero (n + k) (F.a * F.rL σ)
  rw [← (F.BL σ).x_add_w] at h₂
  rw [Params.z, show (F.BL σ).w k = F.wL k from rfl] at *
  push_cast at h₁ h₂
  rw [show ((F.BL σ).x n + F.wL k / 2 + F.a * F.rL σ) ^ 2 - F.wL k ^ 2 / 4 =
    ((F.BL σ).x n + F.a * F.rL σ) * ((F.BL σ).x n + F.wL k + F.a * F.rL σ) by ring]
  exact mul_ne_zero h₁ h₂

/-- **The longitudinal bubble**: `b_σ(νₙ) = -βG_σ(ν)G_σ(ν + ω) = -β((z + ar_σ)² - w²/4)/P(z)`,
`z = zₙ` on the lattice of `σ`. -/
theorem bL_eq (σ : Bool) (n : ℤ) :
    F.bL σ k n = -F.β * (((F.BL σ).z k n + F.a * F.rL σ) ^ 2 - F.wL k ^ 2 / 4) /
      F.PL k ((F.BL σ).z k n) := by
  have hG : ∀ j, F.Gs σ (F.ν j) = F.Gc (F.rL σ) ((F.BL σ).x j) := by
    intro j
    cases σ
    · simpa using F.propagator_dn j
    · simpa using F.propagator_up j
  have h₁ := (F.BL σ).x_sq_sub_ne_zero n
  have h₂ := (F.BL σ).x_sq_sub_ne_zero (n + k)
  rw [← (F.BL σ).x_add_w] at h₂
  simp only [lat_a] at h₁ h₂
  rw [bL, F.ν_add_ω, hG, hG, ← (F.BL σ).x_add_w, Params.z, PL, Gc, Gc,
    show (F.BL σ).w k = F.wL k from rfl]
  rw [show (F.BL σ).w k = F.wL k from rfl] at h₂
  rw [show (((F.BL σ).x n + F.wL k / 2 + F.a) ^ 2 - F.wL k ^ 2 / 4) *
      (((F.BL σ).x n + F.wL k / 2 - F.a) ^ 2 - F.wL k ^ 2 / 4) =
    ((F.BL σ).x n ^ 2 - F.a ^ 2) * (((F.BL σ).x n + F.wL k) ^ 2 - F.a ^ 2) by ring]
  field_simp
  ring

/-! ### The vertex -/

/-- **The longitudinal vertex**, Eq. (field-L): `z = zₙ` on the lattice of `σ`, `v = zₘ` on the
lattice of `t`, `𝔠ⱼ = P(u_j) cMode_j`, `U = 2a`. The column `t` uses the mode normalized with `t`
(`fMp k t`, `fMm k t`); the note normalizes with `↑`, and rescaling a mode does not change `Γ_L`. -/
noncomputable def gammaL (u : Fin 4 → ℂ) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) : ℂ :=
  (if σ = t then
      F.β * F.PL k ((F.BL t).z k n) *
          (1 / (((F.BL t).z k n + F.a * F.rL t) ^ 2 - F.wL k ^ 2 / 4) -
            F.QL k t ((F.BL t).z k n) / F.E8L k ((F.BL t).z k n)) * kron n m +
        2 * F.lamL ^ 2 * F.DL t * ((F.BL t).z k m ^ 2 + F.q₀L k) * F.PL k ((F.BL t).z k m) /
          (F.E8L k ((F.BL t).z k m) * ((F.BL t).z k n + (F.BL t).z k m))
    else
      2 * F.a - F.lamL * F.PL k ((F.BL t).z k m) / F.E8L k ((F.BL t).z k m) *
        (F.mL t * (F.HL k (F.rL t) (-(F.BL t).z k m) * F.HL k (F.rL (!t)) (-(F.BL t).z k m) +
              F.lamL ^ 2 * (F.piP - F.piS)) / ((F.BL σ).z k n - (F.BL t).z k m) +
          F.ρ * (F.HL k (F.rL t) (-(F.BL t).z k m) * F.HL k (F.rL (!t)) ((F.BL t).z k m) -
              F.lamL ^ 2 * (F.piP - F.piS)) / ((F.BL σ).z k n + (F.BL t).z k m))) -
    F.β ^ 2 * ∑ j, F.cMode k t u j * F.PL k (u j) / ((F.BL t).z k m ^ 2 - u j ^ 2) *
      (F.fMp k t σ (u j) / ((F.BL σ).z k n - u j) + F.fMm k t σ (u j) / ((F.BL σ).z k n + u j))

variable {F k}

/-- **The longitudinal vertex**, the closed form: `β²(χ_L⁻¹ - b⁻¹) = Γ_L` with `χ_L⁻¹ = invL`. -/
theorem LGeneric.gammaL_eq {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) (σ : Bool)
    (n m : ℤ) :
    (F.β : ℂ) ^ 2 * (F.invL k u σ n t m - if σ = t then kron n m / F.bL σ k n else 0) =
      F.gammaL k u σ n t m := by
  have hβ := β_ne_zeroT (F := F)
  have hEm := hg.E8L_ne m
  have hs : ∑ j, F.cMode k t u j / ((F.BL t).z k m ^ 2 - u j ^ 2) * F.modeL k t σ (u j) n =
      ∑ j, F.cMode k t u j * F.PL k (u j) / ((F.BL t).z k m ^ 2 - u j ^ 2) *
        (F.fMp k t σ (u j) / ((F.BL σ).z k n - u j) +
          F.fMm k t σ (u j) / ((F.BL σ).z k n + u j)) :=
    Finset.sum_congr rfl fun j _ ↦ by simp only [modeL]; ring
  rw [invL, gammaL, hs]
  by_cases hσ : σ = t
  · subst hσ
    have hPn := F.PL_BL_ne k σ n
    have hb := F.bL_num_ne k σ n
    have hzv := F.zL_add_self_ne k hg.c_ne n m
    simp only [↓reduceIte, denseL, F.bL_eq k σ n, dhL, AhL]
    rcases eq_or_ne n m with rfl | hnm
    · simp only [kron_self]
      field_simp
      ring
    · simp only [kron_of_ne hnm]
      field_simp
      ring
  · obtain rfl : σ = !t := by cases σ <;> cases t <;> simp_all
    have hzm : (F.BL (!t)).z k n - (F.BL t).z k m ≠ 0 := by
      simpa using F.zL_sub_opp_ne k hg.h_ne (!t) n m
    have hzp : (F.BL (!t)).z k n + (F.BL t).z k m ≠ 0 := by
      simpa using F.zL_add_opp_ne k hg.μ_ne (!t) n m
    simp only [Bool.not_ne_self, ↓reduceIte, denseL, BhL, ChL, sub_zero]
    field_simp
    ring

/-! ### Symmetry and the left inverse -/

variable (F k)

/-- **`χ_L` is symmetric**, off the axes: `χ_στ(νₙ, νₘ) = χ_τσ(νₘ, νₙ)`, by the input
(`PL_mul_chiL_self`, `PL_mul_chiL_opp`). -/
theorem chiL_symm (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (σ τ : Bool) (n m : ℤ) :
    F.chiL σ τ k n m = F.chiL τ σ k m n := by
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
    have h₁ := F.PL_mul_chiL_opp k hμ hh σ n m
    have h₂ := F.PL_mul_chiL_opp k hμ hh (!σ) m n
    simp only [Bool.not_not] at h₂
    have hzm := F.zL_sub_opp_ne k hh σ n m
    have hzp := F.zL_add_opp_ne k hμ σ n m
    have hzm' : (F.BL (!σ)).z k m - (F.BL σ).z k n ≠ 0 := fun h0 ↦ hzm (by linear_combination -h0)
    have hzp' : (F.BL (!σ)).z k m + (F.BL σ).z k n ≠ 0 := fun h0 ↦ hzp (by linear_combination h0)
    rw [(eq_div_iff hPn).mpr ((mul_comm _ _).trans h₁),
      (eq_div_iff hPm).mpr ((mul_comm _ _).trans h₂), div_eq_div_iff hPn hPm, add_mul, add_mul,
      div_mul_cancel₀ _ hPm, div_mul_cancel₀ _ hPn]
    have hm' : (F.mL (!σ) : ℂ) = -F.mL σ := by rw [F.mL_not σ]; push_cast; ring
    have hr' : (F.rL (!σ) : ℂ) = F.ρ + F.mL σ := by rw [F.rL_not σ]; push_cast; ring
    simp only [hm', hr', F.rL_cast σ, N₀L, N₂L, HL, PL, q₀L, lamL]
    field_simp
    ring

variable {F k}

/-- **The left inverse**: the transpose of `invL` is a left inverse of `χ_L`,
`∑_{τ, l} invL_{(τ, l), (σ, n)} χ_{τt}(l, m) = δ_σt δ_nm`. -/
theorem chiL_isLeftInverse_transpose {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.invL k u τ l σ n * F.chiL τ t k l m)
      (if σ = t then kron n m else 0) := by
  have h := chiL_isRightInverse hg t m σ n
  have hμ := (hg σ).μ_ne
  have hh := (hg σ).h_ne
  rw [show (if t = σ then kron m n else 0) = if σ = t then kron n m else 0 by
    by_cases hσ : σ = t
    · subst hσ; simp [kron_comm]
    · simp [hσ, Ne.symm hσ]] at h
  convert h using 1
  funext l
  exact Finset.sum_congr rfl fun τ _ ↦ by rw [F.chiL_symm k hμ hh τ t l m, mul_comm]

end FParams

end HubbardAtom
