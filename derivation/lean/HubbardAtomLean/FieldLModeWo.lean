import HubbardAtomLean.FieldLModeWs

/-!
# The image of the mode: `k ≠ 0`, row `-t`

As `FieldLModeWs.lean`, in the row `-t`.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

-- large `field_simp`/`ring` computations: one at a time
set_option Elab.async false

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `Mo0` is `McW10(u, S_t(u))`, `k ≠ 0`. -/
theorem Mo0_W (hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.Mo0 k t u = LCert.McW10 u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell k t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell k (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [Mo0, MoA0, MoB, MoC0, MoD0, F.fS_opp k hr h.m_ne, F.fS_opp_neg k hr h.ρ_ne,
      F.fS_self_neg k hr h.D_ne]
  simp only [dζ, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, zero_add, add_zero, fMp,
      fMm, Bool.not_ne_self, ↓reduceIte, RhatSL, RhatL0,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S k u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw,
    ζL, F.pUp_mul_pDn t, kron_of_ne hk, zero_mul, sub_zero, add_zero]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.McW10]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `Mo1` is `McW11(u, S_t(u))`, `k ≠ 0`. -/
theorem Mo1_W (_hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (_hr : F.ModeRel k t u) :
    F.Mo1 k t u = LCert.McW11 u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell k t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell k (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [Mo1]
  simp only [fMp, fMm, Bool.not_ne_self, ↓reduceIte,

    Bool.not_not, αSL, αPL, F.mL_not t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S k u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [HL, PL, lamL, rL_not, q₀L, F.cS_eq t, cP]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.McW11]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `Mo2` is `McW12(u, S_t(u))`, `k ≠ 0`. -/
theorem Mo2_W (hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.Mo2 k t u = LCert.McW12 u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell k t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell k (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [Mo2, MoB, MoC2, MoD2, F.fS_opp k hr h.m_ne, F.fS_opp_neg k hr h.ρ_ne,
      F.fS_self_neg k hr h.D_ne]
  simp only [dζ, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, zero_add, add_zero, fMp,
      fMm, Bool.not_ne_self, ↓reduceIte, RhatSL,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  generalize (F.BL t).S k u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, DL, cLw,
    ζL, F.pUp_mul_pDn t, kron_of_ne hk, zero_mul, add_zero]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.McW12]
  ring

end FParams

end HubbardAtom
