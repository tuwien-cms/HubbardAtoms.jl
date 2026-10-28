import HubbardAtomLean.FieldLMode

/-!
# The residues of the image of the mode

The image of the mode `P(u)Φ` of a root `u` of `E₈` under `P(z)χ_L` has no poles: its residues at
`z = ±u` vanish in both rows. The residue at `z = u` in the row of the column spin is
`β²P(u)E₈(u)`; the other three vanish identically, by the product relations.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-- `f_{-t,+} = ν₊α^S_{-t}(u)` with `ν₊ = -λ[H_t(-u)H_{-t}(-u) + λ²d_π]`. -/
theorem fMp_opp (t : Bool) (u : ℂ) :
    F.fMp k t (!t) u = -(F.lamL * (F.HL k (F.rL t) (-u) * F.HL k (F.rL (!t)) (-u) +
      F.lamL ^ 2 * (F.piP - F.piS))) * F.αSL k (!t) u := by
  simp only [fMp, Bool.not_ne_self, ↓reduceIte]

/-- `f_{-t,-} = ν₋α^P_{-t}(u)` with `ν₋ = -λ[H_t(-u)H_{-t}(u) - λ²d_π]`. -/
theorem fMm_opp (t : Bool) (u : ℂ) :
    F.fMm k t (!t) u = -(F.lamL * (F.HL k (F.rL t) (-u) * F.HL k (F.rL (!t)) u -
      F.lamL ^ 2 * (F.piP - F.piS))) * F.αPL k (!t) u := by
  simp only [fMm, Bool.not_ne_self, ↓reduceIte]

/-- **Residue at `z = u`, row `t`**: `β²P(u)E₈(u)`, zero at a root. -/
theorem res_self_pos {t : Bool} {u : ℂ} (hr : F.ModeRel k t u) (hE : F.E8L k u = 0) :
    -F.β * F.HL k (F.rL t) u * F.PL k u * F.fMp k t t u +
        F.lamL * F.PL k u * (F.fMp k t (!t) u * F.αSL k t u +
          F.fMm k t (!t) u * F.αPL k t (-u)) = 0 := by
  have rS := hr.rS
  have rP := hr.rP
  rw [F.fMp_opp, F.fMm_opp,
    show ∀ X Y : ℂ, X * F.αSL k (!t) u * Y = X * (Y * F.αSL k (!t) u) from fun _ _ ↦ by ring,
    show ∀ X Y : ℂ, X * F.αPL k (!t) u * Y = X * (F.αPL k (!t) u * Y) from fun _ _ ↦ by ring,
    rS, rP]
  rw [F.E8L_eq k t] at hE
  simp only [fMp, ↓reduceIte, QL, F.piS_eq t] at hE ⊢
  rw [F.pUp_mul_pDn t]
  linear_combination (F.β ^ 2 * F.PL k u) * hE

/-- **Residue at `z = -u`, row `t`**: zero, by the spin-flip and pair relations. -/
theorem res_self_neg {t : Bool} {u : ℂ} (h : F.ColNe k t) (hr : F.ModeRel k t u) :
    -F.β * F.HL k (F.rL t) (-u) * F.PL k u * F.fMm k t t u +
        F.lamL * F.PL k u * (F.fMp k t (!t) u * F.αPL k t u +
          F.fMm k t (!t) u * F.αSL k t (-u)) = 0 := by
  have e₁ : F.fMp k t (!t) u * F.αPL k t u =
      F.ρ * (F.fMp k t (!t) u * (F.BL (!t)).S k u) - F.cP * F.fMp k t (!t) u := by
    simp only [αPL]; ring
  have e₂ : F.fMm k t (!t) u * F.αSL k t (-u) =
      -(F.mL t * (F.fMm k t (!t) u * (F.BL (!t)).S k (-u))) - F.cS * F.fMm k t (!t) u := by
    simp only [αSL]; ring
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  rw [e₁, e₂, F.fS_opp k hr h.m_ne, F.fS_opp_neg k hr h.ρ_ne]
  simp only [fMp, fMm, Bool.not_ne_self, ↓reduceIte, αSL, αPL, αL, Bool.not_not, F.mL_not t,
    HL, QL, lamL, piP, F.piS_eq t, F.pUp_mul_pDn t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw]
  push_cast
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM]
  rw [hR] at h
  have hm := h.m_ne
  have hρ := h.ρ_ne
  rw [hM] at hm
  rw [hR] at hρ
  field_simp
  ring

/-- **Residue at `z = u`, row `-t`**: zero, by the same-spin relation. -/
theorem res_opp_pos {t : Bool} {u : ℂ} (h : F.ColNe k t) (hr : F.ModeRel k t u) :
    -F.β * F.HL k (F.rL (!t)) u * F.PL k u * F.fMp k t (!t) u +
        F.lamL * F.PL k u * (F.fMp k t t u * F.αSL k (!t) u +
          F.fMm k t t u * F.αPL k (!t) (-u)) = 0 := by
  have e₂ : F.fMm k t t u * F.αPL k (!t) (-u) =
      F.ρ * (F.fMm k t t u * (F.BL t).S k (-u)) - F.cP * F.fMm k t t u := by
    simp only [αPL, Bool.not_not]; ring
  have hβ := β_ne_zeroT (F := F)
  rw [e₂, F.fS_self_neg k hr h.D_ne]
  simp only [fMp, fMm, Bool.not_ne_self, ↓reduceIte, αSL, αPL, αL, Bool.not_not, F.mL_not t,
    HL, QL, lamL, piP, F.piS_eq t, F.pUp_mul_pDn t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw]
  push_cast
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  have hD := h.D_ne
  simp only [DL] at hD
  push_cast at hD
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM] at hD ⊢
  field_simp
  ring

/-- **Residue at `z = -u`, row `-t`**: zero, by the same-spin relation. -/
theorem res_opp_neg {t : Bool} {u : ℂ} (h : F.ColNe k t) (hr : F.ModeRel k t u) :
    -F.β * F.HL k (F.rL (!t)) (-u) * F.PL k u * F.fMm k t (!t) u +
        F.lamL * F.PL k u * (F.fMp k t t u * F.αPL k (!t) u +
          F.fMm k t t u * F.αSL k (!t) (-u)) = 0 := by
  have e₂ : F.fMm k t t u * F.αSL k (!t) (-u) =
      F.mL t * (F.fMm k t t u * (F.BL t).S k (-u)) - F.cS * F.fMm k t t u := by
    simp only [αSL, Bool.not_not, F.mL_not t]; push_cast; ring
  have hβ := β_ne_zeroT (F := F)
  rw [e₂, F.fS_self_neg k hr h.D_ne]
  simp only [fMp, fMm, Bool.not_ne_self, ↓reduceIte, αSL, αPL, αL, Bool.not_not, F.mL_not t,
    HL, QL, lamL, piP, F.piS_eq t, F.pUp_mul_pDn t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw]
  push_cast
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  have hD := h.D_ne
  simp only [DL] at hD
  push_cast at hD
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM] at hD ⊢
  field_simp
  ring

end FParams

end HubbardAtom
