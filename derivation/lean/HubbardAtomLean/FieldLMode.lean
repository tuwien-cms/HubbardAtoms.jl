import HubbardAtomLean.FieldLConnZo

/-!
# The image of the mode of a root

The image of the pole-pair mode `P(u)Φ` of a root `u` of `E₈` under `P(z)χ_L`, in the rows of the
column spin `t` (`s`) and of `-t` (`o`): its residues at `z = ±u` vanish, and its polynomial part
is `M_c(u, S_t(u))` of `FieldLCertData.lean`. The thermal values at `±u` on both lattices enter
through `S_t(u)` alone, by the product relations `αSL_mul_αSL`, `αPL_mul_αPL`, `αL_mul_αL_neg`.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

-- the identifications below are large `field_simp`/`ring` computations: one at a time
set_option Elab.async false

/-! ### The image of the mode -/

/-- `(u² - a²)S_σ(±u) + R̂_S(±u)`: the rank-one part (`k = 0`) of the same-spin block. -/
noncomputable def WmL (σ : Bool) (x : ℂ) : ℂ :=
  (x ^ 2 - F.a ^ 2) * (F.BL σ).S k x + F.RhatSL k σ x

/-- The coefficient of `z⁰` of the image of `P(u)Φ` in row `t`. -/
noncomputable def Ms0 (t : Bool) (u : ℂ) : ℂ :=
  -F.β * F.PL k u * (F.fMp k t t u * (u + 2 * F.a * F.rL t) +
        F.fMm k t t u * (-u + 2 * F.a * F.rL t)) -
      F.a ^ 2 * F.dζ k t * (F.fMp k t t u * F.WmL k t u + F.fMm k t t u * F.WmL k t (-u)) +
    F.fMp k t (!t) u * (F.N₀L k (F.rL (!t)) u * (F.BL (!t)).S k u + F.RhatL0 k t u) +
    F.fMm k t (!t) u * (F.N₀L k (F.rL (!t)) (-u) * (F.BL (!t)).S k (-u) + F.RhatL0 k t (-u))

/-- The coefficient of `z¹` of the image of `P(u)Φ` in row `t`. -/
noncomputable def Ms1 (t : Bool) (u : ℂ) : ℂ :=
  -F.β * F.PL k u * (F.fMp k t t u + F.fMm k t t u)

/-- The coefficient of `z²` of the image of `P(u)Φ` in row `t`. -/
noncomputable def Ms2 (t : Bool) (u : ℂ) : ℂ :=
  F.dζ k t * (F.fMp k t t u * F.WmL k t u + F.fMm k t t u * F.WmL k t (-u)) +
    F.fMp k t (!t) u * (F.N₂L k (F.rL (!t)) u * (F.BL (!t)).S k u + F.RhatL2 k t u) +
    F.fMm k t (!t) u * (F.N₂L k (F.rL (!t)) (-u) * (F.BL (!t)).S k (-u) + F.RhatL2 k t (-u))

/-- The coefficient of `z⁰` of the image of `P(u)Φ` in row `-t`. -/
noncomputable def Mo0 (t : Bool) (u : ℂ) : ℂ :=
  -F.β * F.PL k u * (F.fMp k t (!t) u * (u + 2 * F.a * F.rL (!t)) +
        F.fMm k t (!t) u * (-u + 2 * F.a * F.rL (!t))) -
      F.a ^ 2 * F.dζ k (!t) *
        (F.fMp k t (!t) u * F.WmL k (!t) u + F.fMm k t (!t) u * F.WmL k (!t) (-u)) +
    F.fMp k t t u * (F.N₀L k (F.rL t) u * (F.BL t).S k u + F.RhatL0 k (!t) u) +
    F.fMm k t t u * (F.N₀L k (F.rL t) (-u) * (F.BL t).S k (-u) + F.RhatL0 k (!t) (-u))

/-- The coefficient of `z¹` of the image of `P(u)Φ` in row `-t`. -/
noncomputable def Mo1 (t : Bool) (u : ℂ) : ℂ :=
  -F.β * F.PL k u * (F.fMp k t (!t) u + F.fMm k t (!t) u)

/-- The coefficient of `z²` of the image of `P(u)Φ` in row `-t`. -/
noncomputable def Mo2 (t : Bool) (u : ℂ) : ℂ :=
  F.dζ k (!t) * (F.fMp k t (!t) u * F.WmL k (!t) u + F.fMm k t (!t) u * F.WmL k (!t) (-u)) +
    F.fMp k t t u * (F.N₂L k (F.rL t) u * (F.BL t).S k u + F.RhatL2 k (!t) u) +
    F.fMm k t t u * (F.N₂L k (F.rL t) (-u) * (F.BL t).S k (-u) + F.RhatL2 k (!t) (-u))

/-! ### The thermal values at a root through `s = S_t(u)` -/

/-- The product relations at `±u`, for `±u` off both lattices and `μ_t ≠ 0`: the hypotheses under
which the image of the mode is a polynomial in `s = S_t(u)`. -/
structure ModeRel (t : Bool) (u : ℂ) : Prop where
  rS : F.αSL k t u * F.αSL k (!t) u = F.β ^ 2 * F.pUp * F.pDn
  rP : F.αPL k (!t) u * F.αPL k t (-u) = F.β ^ 2 * F.p₀ * F.p₂
  rα : F.αL k t u * F.αL k t (-u) = F.β ^ 2 * (F.p₀ * F.p₂) * (F.pUp * F.pDn)

theorem ModeRel.of_off {t : Bool} (hc : F.cL t ≠ 0) {u : ℂ}
    (hu : ∀ σ n, (F.BL σ).z k n ≠ u) (hu' : ∀ σ n, (F.BL σ).z k n ≠ -u) : F.ModeRel k t u := by
  refine ⟨?_, ?_, F.αL_mul_αL_neg k hc (hu t) (hu' t)⟩
  · exact F.αSL_mul_αSL k t (hu true) (hu false)
  · have := F.αPL_mul_αPL k (!t) (x := u) (hu' (!t)) (by simpa using hu t)
    simpa using this

/-- The thermal values at `±u` through `s = S_t(u)`, from the product relations. -/
theorem ModeRel.values {t : Bool} {u : ℂ} (hr : F.ModeRel k t u) (h : F.ColNe k t) :
    F.mL t * (F.BL t).S k u - F.cS ≠ 0 ∧ F.ρ * (F.BL t).S k u - F.cP ≠ 0 ∧
      F.DL t * (F.BL t).S k u - F.β * F.cLw t / 2 ≠ 0 ∧
      (F.BL (!t)).S k u = -(F.β ^ 2 * (F.pUp * F.pDn) /
        (F.mL t * (F.BL t).S k u - F.cS) + F.cS) / F.mL t ∧
      (F.BL (!t)).S k (-u) = (F.β ^ 2 * (F.p₀ * F.p₂) /
        (F.ρ * (F.BL t).S k u - F.cP) + F.cP) / F.ρ ∧
      (F.BL t).S k (-u) = (F.β ^ 2 * (F.p₀ * F.p₂) * (F.pUp * F.pDn) /
        (F.DL t * (F.BL t).S k u - F.β * F.cLw t / 2) + F.β * F.cLw t / 2) / F.DL t := by
  have hβ := β_ne_zeroT (F := F)
  have h₀ : (F.p₀ : ℂ) * F.p₂ ≠ 0 := by
    have := F.p₀_pos; have := F.p₂_pos
    exact_mod_cast (mul_pos ‹0 < F.p₀› ‹0 < F.p₂›).ne'
  have h₁ : (F.pUp : ℂ) * F.pDn ≠ 0 := by
    have := F.pUp_pos; have := F.pDn_pos
    exact_mod_cast (mul_pos ‹0 < F.pUp› ‹0 < F.pDn›).ne'
  obtain ⟨rS, rP, rα⟩ := hr
  simp only [αSL, αPL, αL, Bool.not_not, F.mL_not t] at rS rP rα
  push_cast at rS rP rα
  have hm := h.m_ne
  have hρ := h.ρ_ne
  have hD := h.D_ne
  have nS : F.mL t * (F.BL t).S k u - F.cS ≠ 0 := fun h0 ↦ by
    have : (F.β : ℂ) ^ 2 * F.pUp * F.pDn = 0 := by
      rw [← rS]; linear_combination (-(F.mL t * (F.BL (!t)).S k u) - F.cS) * h0
    exact mul_ne_zero (mul_ne_zero (pow_ne_zero 2 hβ) (left_ne_zero_of_mul h₁))
      (right_ne_zero_of_mul h₁) this
  have nP : F.ρ * (F.BL t).S k u - F.cP ≠ 0 := fun h0 ↦ by
    have : (F.β : ℂ) ^ 2 * F.p₀ * F.p₂ = 0 := by
      rw [← rP]; linear_combination (F.ρ * (F.BL (!t)).S k (-u) - F.cP) * h0
    exact mul_ne_zero (mul_ne_zero (pow_ne_zero 2 hβ) (left_ne_zero_of_mul h₀))
      (right_ne_zero_of_mul h₀) this
  have nα : F.DL t * (F.BL t).S k u - F.β * F.cLw t / 2 ≠ 0 := fun h0 ↦ by
    have : (F.β : ℂ) ^ 2 * (F.p₀ * F.p₂) * (F.pUp * F.pDn) = 0 := by
      rw [← rα]; linear_combination (F.DL t * (F.BL t).S k (-u) - F.β * F.cLw t / 2) * h0
    exact mul_ne_zero (mul_ne_zero (pow_ne_zero 2 hβ) h₀) h₁ this
  refine ⟨nS, nP, nα, ?_, ?_, ?_⟩
  · field_simp
    linear_combination -rS
  · field_simp
    linear_combination rP
  · field_simp
    linear_combination rα

end FParams

end HubbardAtom
