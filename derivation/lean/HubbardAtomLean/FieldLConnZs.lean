import HubbardAtomLean.FieldLConn

/-!
# The completed defect at `k = 0`, row `t`

The identification of the completed defect of the row `t` at `k = 0` with the generated
polynomials (`FieldLConn.lean` has `k ≠ 0`); split off to bound the memory of the build.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

-- large `field_simp`/`ring` computations: one at a time
set_option Elab.async false

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Ds0` is `NcZ00(v²)/(βE₈(v))`, `k = 0`. -/
theorem Ds0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Ds0 k t v = LCert.NcZ00 (v ^ 2) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Ds0, dζ, kron_self, dhL, AhL, BhL, ChL, WsL, sReflL, RhatSL, RhatL0,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL,
      ζLs, hw, F.pUp_mul_pDn t, kron_self, one_mul]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcZ00, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Ds1` is `NcZ01(v²)/(βE₈(v))`, `k = 0`. -/
theorem Ds1_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Ds1 k t v = LCert.NcZ01 (v ^ 2) F.a (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Ds1, AhL]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [PL, lamL, q₀L, DL, hw]
  push_cast
  simp only [h₀, h₂, ht, hn, hR, hM, LCert.NcZ01]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Ds2` is `NcZ02(v²)/(βE₈(v))`, `k = 0`. -/
theorem Ds2_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Ds2 k t v = LCert.NcZ02 (v ^ 2) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Ds2, dζ, kron_self, dhL, AhL, BhL, ChL, WsL, sReflL, RhatSL, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', lat_a, lat_β]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, ζLs,
      hw, F.pUp_mul_pDn t, kron_self, one_mul]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcZ02, LCert.E8W]
  ring

end FParams

end HubbardAtom
