import HubbardAtomLean.FieldLModeZs

/-!
# The image of the mode: `k = 0`, row `-t`

As `FieldLModeZs.lean`, in the row `-t`.
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
/-- `MoA0` is `MpZ10A(u, S_t(u))`, `k = 0`. -/
theorem MoA0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MoA0 k t u = LCert.MpZ10A u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
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
  simp only [MoA0, F.fS_opp 0 hr h.m_ne, F.fS_opp_neg 0 hr h.ρ_ne, F.fS_self_neg 0 hr h.D_ne]
  simp only [dζ, kron_self, one_mul, mul_one, fMp, fMm, Bool.not_ne_self, ↓reduceIte, WmL, RhatSL, RhatL0,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw,
    ζL, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one, ζLs, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ10A]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MoB` is `MpZ12B(u, S_t(u))`, `k = 0`. -/
theorem MoB_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MoB k t u = LCert.MpZ12B u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
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
  simp only [MoB, F.fS_opp 0 hr h.m_ne, F.fS_opp_neg 0 hr h.ρ_ne, F.fS_self_neg 0 hr h.D_ne]
  simp only [dζ, kron_self, one_mul, mul_one, fMp, fMm, Bool.not_ne_self, ↓reduceIte, WmL, RhatSL, RhatL0,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw,
    ζL, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one, ζLs, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ12B]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MoC0` is `MpZ10C(u, S_t(u))`, `k = 0`. -/
theorem MoC0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MoC0 k t u = LCert.MpZ10C u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
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
  simp only [MoC0, F.fS_opp 0 hr h.m_ne, F.fS_opp_neg 0 hr h.ρ_ne, F.fS_self_neg 0 hr h.D_ne]
  simp only [dζ, kron_self, one_mul, mul_one, fMp, fMm, Bool.not_ne_self, ↓reduceIte, WmL, RhatSL, RhatL0,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw,
    ζL, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one, ζLs, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ10C]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MoD0` is `MpZ10D(u, S_t(u))`, `k = 0`. -/
theorem MoD0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MoD0 k t u = LCert.MpZ10D u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
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
  simp only [MoD0, F.fS_opp 0 hr h.m_ne, F.fS_opp_neg 0 hr h.ρ_ne, F.fS_self_neg 0 hr h.D_ne]
  simp only [dζ, kron_self, one_mul, mul_one, fMp, fMm, Bool.not_ne_self, ↓reduceIte, WmL, RhatSL, RhatL0,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw,
    ζL, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one, ζLs, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ10D]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MoC2` is `MpZ12C(u, S_t(u))`, `k = 0`. -/
theorem MoC2_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MoC2 k t u = LCert.MpZ12C u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
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
  simp only [MoC2, F.fS_opp 0 hr h.m_ne, F.fS_opp_neg 0 hr h.ρ_ne, F.fS_self_neg 0 hr h.D_ne]
  simp only [dζ, kron_self, one_mul, mul_one, fMp, fMm, Bool.not_ne_self, ↓reduceIte, WmL, RhatSL, RhatL0,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw,
    ζL, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one, ζLs, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ12C]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `MoD2` is `MpZ12D(u, S_t(u))`, `k = 0`. -/
theorem MoD2_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.MoD2 k t u = LCert.MpZ12D u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
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
  simp only [MoD2, F.fS_opp 0 hr h.m_ne, F.fS_opp_neg 0 hr h.ρ_ne, F.fS_self_neg 0 hr h.D_ne]
  simp only [dζ, kron_self, one_mul, mul_one, fMp, fMm, Bool.not_ne_self, ↓reduceIte, WmL, RhatSL, RhatL0,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw,
    ζL, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one, ζLs, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.MpZ12D]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the image of the mode
set_option maxRecDepth 100000 in
/-- `Mo1` is `McZ11(u, S_t(u))`, `k = 0`. -/
theorem Mo1_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.Mo1 k t u = LCert.McZ11 u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
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
  simp only [Mo1, F.fS_opp 0 hr h.m_ne, F.fS_opp_neg 0 hr h.ρ_ne, F.fS_self_neg 0 hr h.D_ne]
  simp only [dζ, kron_self, one_mul, mul_one, fMp, fMm, Bool.not_ne_self, ↓reduceIte, WmL, RhatSL, RhatL0,
    RhatL2, Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp,
    hSm, hSp', hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β,
    Bool.not_not, αSL, αPL, αL, F.mL_not t, F.pUp_mul_pDn t, piP, F.piS_eq t]
  push_cast
  generalize (F.BL t).S 0 u = s
  field_simp
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw,
    ζL, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one, ζLs, hw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.McZ11]
  ring

set_option maxHeartbeats 4000000 in
-- `ring` on the generated polynomials
/-- `Mo0` is `McZ10(u, S_t(u))`, `k = 0`: the sum of its pieces. -/
theorem Mo0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.Mo0 k t u = LCert.McZ10 u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  rw [Mo0, F.MoA0_Z k hk t h hr, F.MoB_Z k hk t h hr, F.MoC0_Z k hk t h hr, F.MoD0_Z k hk t h hr]
  unfold LCert.McZ10 LCert.MpZ10A LCert.MpZ12B LCert.MpZ10C LCert.MpZ10D
  ring

set_option maxHeartbeats 4000000 in
-- `ring` on the generated polynomials
/-- `Mo2` is `McZ12(u, S_t(u))`, `k = 0`: the sum of its pieces. -/
theorem Mo2_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    F.Mo2 k t u = LCert.McZ12 u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  rw [Mo2, F.MoB_Z k hk t h hr, F.MoC2_Z k hk t h hr, F.MoD2_Z k hk t h hr]
  unfold LCert.McZ12 LCert.MpZ12B LCert.MpZ12C LCert.MpZ12D
  ring

end FParams

end HubbardAtom
