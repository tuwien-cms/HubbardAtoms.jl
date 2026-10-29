import HubbardAtomLean.FieldLModeWo
import HubbardAtomLean.FieldLModeData

/-!
# The image of the mode: `k = 0`, row `t`

The coefficients of the image of the mode in the row of the column spin at `k = 0`: block by
block against the pieces of `FieldLModeData.lean`, then summed to `McZ` of `FieldLCertData.lean`.
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
/-- `MsA0` is `MpZ00A(u, S_t(u))`, `k = 0`. -/
theorem MsA0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MsA0 k t u = LCert.MpZ00A u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell 0 t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell 0 (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [MsA0]
  simp only [fMp, fMm, ↓reduceIte,

    αL]
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, DL, cLw,
    hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ00A]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MsB` is `MpZ02B(u, S_t(u))`, `k = 0`. -/
theorem MsB_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MsB k t u = LCert.MpZ02B u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell 0 t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell 0 (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [MsB, F.fS_self_neg 0 hr h.D_ne]
  simp only [dζ, kron_self, one_mul, fMp, fMm, ↓reduceIte, WmL, RhatSL,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hq, hκ, hP, lat_a, lat_β,
    αL, F.pUp_mul_pDn t]
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, DL, cLw,
    ζLs, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ02B]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MsC0` is `MpZ00C(u, S_t(u))`, `k = 0`. -/
theorem MsC0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MsC0 k t u = LCert.MpZ00C u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell 0 t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell 0 (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [MsC0, F.fS_opp 0 hr h.m_ne]
  simp only [kron_self, one_mul, mul_one, fMp, Bool.not_ne_self, ↓reduceIte, RhatL0,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm,
    hSp', hSm', hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, lamL, rL_not, q₀L, F.cS_eq t,
    ζL, F.pUp_mul_pDn t, kron_self, one_mul, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ00C]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MsD0` is `MpZ00D(u, S_t(u))`, `k = 0`. -/
theorem MsD0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MsD0 k t u = LCert.MpZ00D u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell 0 t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell 0 (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [MsD0, F.fS_opp_neg 0 hr h.ρ_ne]
  simp only [kron_self, one_mul, mul_one, fMm, Bool.not_ne_self, ↓reduceIte, RhatL0,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm,
    hSp', hSm', hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αPL, piP, F.piS_eq t]
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, lamL, rL_not, q₀L, cP,
    ζL, F.pUp_mul_pDn t, kron_self, one_mul, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ00D]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MsC2` is `MpZ02C(u, S_t(u))`, `k = 0`. -/
theorem MsC2_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MsC2 k t u = LCert.MpZ02C u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell 0 t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell 0 (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [MsC2, F.fS_opp 0 hr h.m_ne]
  simp only [kron_self, one_mul, mul_one, fMp, Bool.not_ne_self, ↓reduceIte,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm,
    hSp', hSm', hq', hκ', hP', lat_a, lat_β,
    Bool.not_not, αSL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₂L, HL, PL, lamL, rL_not, q₀L, F.cS_eq t,
    ζL, F.pUp_mul_pDn t, kron_self, one_mul, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ02C]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MsD2` is `MpZ02D(u, S_t(u))`, `k = 0`. -/
theorem MsD2_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MsD2 k t u = LCert.MpZ02D u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell 0 t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell 0 (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [MsD2, F.fS_opp_neg 0 hr h.ρ_ne]
  simp only [kron_self, one_mul, mul_one, fMm, Bool.not_ne_self, ↓reduceIte,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm,
    hSp', hSm', hq', hκ', hP', lat_a, lat_β,
    Bool.not_not, αPL, piP, F.piS_eq t]
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₂L, HL, PL, lamL, rL_not, q₀L, cP,
    ζL, F.pUp_mul_pDn t, kron_self, one_mul, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ02D]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `Ms1` is `McZ01(u, S_t(u))`, `k = 0`. -/
theorem Ms1_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.Ms1 k t u = LCert.McZ01 u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨w₁, w₂, w₃, w₄⟩ := F.weight_sums_ne t
  obtain ⟨hSp, hSm, hq, hκ, hP⟩ := F.BL_ell 0 t
  obtain ⟨hSp', hSm', hq', hκ', hP'⟩ := F.BL_ell 0 (!t)
  rw [Bool.not_not] at hSm'
  have ha := h.a_ne
  have hρ := h.ρ_ne
  have hm := h.m_ne
  have hD := h.D_ne
  have hq0 := h.q_ne
  simp only [Ms1]
  simp only [fMp, fMm, ↓reduceIte,

    αL]
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, DL, cLw,
    hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.McZ01]
  ring

set_option maxHeartbeats 4000000 in
-- `ring` on the generated polynomials
/-- `Ms0` is `McZ00(u, S_t(u))`, `k = 0`: the sum of its pieces. -/
theorem Ms0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.Ms0 k t u = LCert.McZ00 u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  rw [Ms0, F.MsA0_Z k hk t h hr, F.MsB_Z k hk t h hr, F.MsC0_Z k hk t h hr, F.MsD0_Z k hk t h hr]
  unfold LCert.McZ00 LCert.MpZ00A LCert.MpZ02B LCert.MpZ00C LCert.MpZ00D
  ring

set_option maxHeartbeats 4000000 in
-- `ring` on the generated polynomials
/-- `Ms2` is `McZ02(u, S_t(u))`, `k = 0`: the sum of its pieces. -/
theorem Ms2_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.Ms2 k t u = LCert.McZ02 u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  rw [Ms2, F.MsB_Z k hk t h hr, F.MsC2_Z k hk t h hr, F.MsD2_Z k hk t h hr]
  unfold LCert.McZ02 LCert.MpZ02B LCert.MpZ02C LCert.MpZ02D
  ring

end FParams

end HubbardAtom
