import HubbardAtomLean.FieldLAlg
import HubbardAtomLean.FieldLCertZ

/-!
# The completion of the longitudinal inverse

The polynomial parts of the images under `P(z)χ_L` of the column `(t, v)` of the inverse and of the
mode of a root `u`, as coefficients of `zᶜ`, and their identification with the generated
polynomials of `FieldLCertData.lean`: the completed defect is `N_c(v²)/(βE₈(v))`
(`DsL_eq`, `DoL_eq`), the image of the mode is `M_c(u, S_t(u))` (`MsL_eq`, `MoL_eq`).

Rows: `s` is the row of the column spin `t`, `o` the row of `-t`.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

-- the identifications below are large `field_simp`/`ring` computations: one at a time
set_option Elab.async false

/-! ### The parameters of a column -/

/-- The weights of the column `t` through `P = p₀ + p₂`, `ρ`, `M = p_t - p_{-t}`. -/
theorem col_params (t : Bool) : ∃ P R M : ℂ,
    (F.p₀ : ℂ) = (P - R) / 2 ∧ (F.p₂ : ℂ) = (P + R) / 2 ∧ (F.pL t : ℂ) = (1 - P + M) / 2 ∧
      (F.pL (!t) : ℂ) = (1 - P - M) / 2 ∧ (F.ρ : ℂ) = R ∧ (F.mL t : ℂ) = M ∧
      (F.p₀ : ℂ) + F.p₂ = P := by
  have hs := F.weights_sumL t
  refine ⟨F.p₀ + F.p₂, F.ρ, F.mL t, ?_, ?_, ?_, ?_, rfl, rfl, rfl⟩
  · rw [ρ]; push_cast; ring
  · rw [ρ]; push_cast; ring
  · rw [mL]; push_cast; linear_combination hs / 2
  · rw [mL]; push_cast; linear_combination hs / 2

theorem pUp_mul_pDn (t : Bool) : (F.pUp : ℂ) * F.pDn = F.pL t * F.pL (!t) := by
  have := F.piS_eq t
  simp only [piS] at this
  exact_mod_cast this

/-! ### The completed defect of the column `(t, v)` -/

/-- `S_t(-v) = βc_t/(2D_t)` for `v` on the lattice of spin `t`. -/
noncomputable def sReflL (t : Bool) : ℂ := F.β * F.cLw t / (2 * F.DL t)

/-- `δ_k0 ζ_σ`. -/
noncomputable def dζ (σ : Bool) : ℂ := kron k 0 * F.ζLs σ

/-- `W = (v² - a²)S_t(-v) + R̂_S(-v)`: the rank-one part of row `t` on `1/(z + v)`. -/
noncomputable def WsL (t : Bool) (v : ℂ) : ℂ :=
  (v ^ 2 - F.a ^ 2) * F.sReflL t + F.RhatSL k t (-v)

/-- The coefficient of `z⁰` of the completed defect in row `t`. -/
noncomputable def Ds0 (t : Bool) (v : ℂ) : ℂ :=
  -F.a ^ 2 * F.dζ k t * F.dhL k t v * (v ^ 2 - F.a ^ 2) -
      F.β * F.AhL k t v * F.PL k v * (-v + 2 * F.a * F.rL t) -
    F.a ^ 2 * F.dζ k t * F.AhL k t v * F.WsL k t v +
      F.BhL k t v * (F.N₀L k (F.rL (!t)) v * -(F.cS / F.mL t) + F.RhatL0 k t v) +
    F.ChL k t v * (F.N₀L k (F.rL (!t)) (-v) * (F.cP / F.ρ) + F.RhatL0 k t (-v)) -
      F.a ^ 2 * F.dζ k t / F.β

/-- The coefficient of `z¹` of the completed defect in row `t`. -/
noncomputable def Ds1 (t : Bool) (v : ℂ) : ℂ := -F.β * F.AhL k t v * F.PL k v

/-- The coefficient of `z²` of the completed defect in row `t`. -/
noncomputable def Ds2 (t : Bool) (v : ℂ) : ℂ :=
  F.dζ k t * F.dhL k t v * (v ^ 2 - F.a ^ 2) + F.dζ k t * F.AhL k t v * F.WsL k t v +
      F.BhL k t v * (F.N₂L k (F.rL (!t)) v * -(F.cS / F.mL t) + F.RhatL2 k t v) +
    F.ChL k t v * (F.N₂L k (F.rL (!t)) (-v) * (F.cP / F.ρ) + F.RhatL2 k t (-v)) +
      F.dζ k t / F.β

/-- `(v² - a²)S_{-t}(v) + R̂_S(v)` on the lattice of `-t`. -/
noncomputable def WoBL (t : Bool) (v : ℂ) : ℂ :=
  (v ^ 2 - F.a ^ 2) * -(F.cS / F.mL t) + F.RhatSL k (!t) v

/-- `(v² - a²)S_{-t}(-v) + R̂_S(-v)` on the lattice of `-t`. -/
noncomputable def WoCL (t : Bool) (v : ℂ) : ℂ :=
  (v ^ 2 - F.a ^ 2) * (F.cP / F.ρ) + F.RhatSL k (!t) (-v)

/-- The coefficient of `z⁰` of the completed defect in row `-t`. -/
noncomputable def Do0 (t : Bool) (v : ℂ) : ℂ :=
  -F.β * F.PL k v * (F.BhL k t v * (v + 2 * F.a * F.rL (!t)) +
        F.ChL k t v * (-v + 2 * F.a * F.rL (!t))) -
      F.a ^ 2 * F.dζ k (!t) * (F.BhL k t v * F.WoBL k t v + F.ChL k t v * F.WoCL k t v) +
    F.dhL k t v * F.N₀L k (F.rL t) v +
      F.AhL k t v * (F.N₀L k (F.rL t) (-v) * F.sReflL t + F.RhatL0 k (!t) (-v)) +
    (-2 * F.a * F.q₀L k - F.a ^ 2 * kron k 0 * F.ζL) / F.β

/-- The coefficient of `z¹` of the completed defect in row `-t`. -/
noncomputable def Do1 (t : Bool) (v : ℂ) : ℂ :=
  -F.β * F.PL k v * (F.BhL k t v + F.ChL k t v) - 4 * F.a ^ 2 * F.rL (!t) / F.β

/-- The coefficient of `z²` of the completed defect in row `-t`. -/
noncomputable def Do2 (t : Bool) (v : ℂ) : ℂ :=
  F.dζ k (!t) * (F.BhL k t v * F.WoBL k t v + F.ChL k t v * F.WoCL k t v) +
      F.dhL k t v * F.N₂L k (F.rL t) v +
    F.AhL k t v * (F.N₂L k (F.rL t) (-v) * F.sReflL t + F.RhatL2 k (!t) (-v)) +
      (-2 * F.a + kron k 0 * F.ζL) / F.β

/-! ### The identification with the generated polynomials -/

theorem E8L_W (t : Bool) (x : ℂ) :
    F.E8L k x = LCert.E8W x F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  rw [F.E8L_eq k t]
  simp only [HL, lamL, piP, F.piS_eq t, rL_not, q₀L]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.E8W]
  ring

/-- The nonvanishing denominators of the completion, for `U ≠ 0`, `μ ≠ 0`, `h ≠ 0`, `μ_t ≠ 0`. -/
structure ColNe (t : Bool) : Prop where
  a_ne : (F.a : ℂ) ≠ 0
  ρ_ne : (F.ρ : ℂ) ≠ 0
  m_ne : (F.mL t : ℂ) ≠ 0
  D_ne : (F.DL t : ℂ) ≠ 0
  q_ne : F.q₀L k ≠ 0

/-- The weights in the denominators of the thermal values `S₀(±a)`. -/
theorem weight_sums_ne (t : Bool) :
    (F.p₀ : ℂ) + F.pL t ≠ 0 ∧ (F.p₀ : ℂ) + F.pL (!t) ≠ 0 ∧ (F.p₂ : ℂ) + F.pL t ≠ 0 ∧
      (F.p₂ : ℂ) + F.pL (!t) ≠ 0 := by
  have := F.pL_pos t
  have := F.pL_pos (!t)
  have := F.p₀_pos
  have := F.p₂_pos
  refine ⟨?_, ?_, ?_, ?_⟩ <;> exact_mod_cast (by positivity : (0 : ℝ) < _).ne'

/-- The constants `ℓⱼ` of the lattice of spin `τ` through the weights. -/
theorem BL_ell (τ : Bool) :
    (F.BL τ).Sp = F.β * (F.pL τ - F.p₀) / (2 * (F.p₀ + F.pL τ)) ∧
      (F.BL τ).Sm = F.β * (F.p₂ - F.pL (!τ)) / (2 * (F.p₂ + F.pL (!τ))) ∧
      (F.BL τ).q₀ k = F.q₀L k ∧ (F.BL τ).κ k = 2 * F.q₀L k - 4 * F.a ^ 2 ∧
      (F.BL τ).Pph k = F.PL k := by
  refine ⟨F.BL_S₀_neg_a τ, F.BL_S₀_a τ, rfl, rfl, rfl⟩

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Ds0` is `NcW00(v²)/(βE₈(v))`, `k ≠ 0`. -/
theorem Ds0_W (hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Ds0 k t v = LCert.NcW00 (v ^ 2) F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Ds0, dζ, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, zero_add, add_zero, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, Bool.not_not, mL_not, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, add_zero, zero_add]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcW00, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Ds1` is `NcW01(v²)/(βE₈(v))`, `k ≠ 0`. -/
theorem Ds1_W (hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Ds1 k t v = LCert.NcW01 (v ^ 2) F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Ds1, dζ, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, zero_add, add_zero, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, Bool.not_not, mL_not, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, add_zero, zero_add]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcW01, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Ds2` is `NcW02(v²)/(βE₈(v))`, `k ≠ 0`. -/
theorem Ds2_W (hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Ds2 k t v = LCert.NcW02 (v ^ 2) F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Ds2, dζ, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, zero_add, add_zero, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, Bool.not_not, mL_not, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, add_zero, zero_add]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcW02, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Do0` is `NcW10(v²)/(βE₈(v))`, `k ≠ 0`. -/
theorem Do0_W (hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Do0 k t v = LCert.NcW10 (v ^ 2) F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Do0, dζ, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, zero_add, add_zero, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, Bool.not_not, mL_not, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, add_zero, zero_add]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcW10, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Do1` is `NcW11(v²)/(βE₈(v))`, `k ≠ 0`. -/
theorem Do1_W (hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Do1 k t v = LCert.NcW11 (v ^ 2) F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Do1, dζ, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, zero_add, add_zero, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, Bool.not_not, mL_not, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, add_zero, zero_add]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcW11, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Do2` is `NcW12(v²)/(βE₈(v))`, `k ≠ 0`. -/
theorem Do2_W (hk : k ≠ 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Do2 k t v = LCert.NcW12 (v ^ 2) F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) /
      (F.β * F.E8L k v) := by
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
  simp only [Do2, dζ, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, zero_add, add_zero, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, Bool.not_not, mL_not, kron_of_ne hk, zero_mul, mul_zero, zero_div, sub_zero, add_zero, zero_add]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcW12, LCert.E8W]
  ring

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
  simp only [Ds0, dζ, kron_self, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, ζLs, hw, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one]
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
  simp only [Ds1, dζ, kron_self, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, ζLs, hw, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcZ01, LCert.E8W]
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
  simp only [Ds2, dζ, kron_self, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, ζLs, hw, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcZ02, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Do0` is `NcZ10(v²)/(βE₈(v))`, `k = 0`. -/
theorem Do0_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Do0 k t v = LCert.NcZ10 (v ^ 2) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) /
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
  simp only [Do0, dζ, kron_self, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, ζLs, hw, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcZ10, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Do1` is `NcZ11(v²)/(βE₈(v))`, `k = 0`. -/
theorem Do1_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Do1 k t v = LCert.NcZ11 (v ^ 2) F.a (F.p₀ + F.p₂) F.ρ (F.mL t) /
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
  simp only [Do1, dζ, kron_self, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, ζLs, hw, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcZ11, LCert.E8W]
  ring

set_option maxHeartbeats 8000000 in
-- `field_simp` and `ring` on the rational functions of the completed defect
set_option maxRecDepth 100000 in
/-- `Do2` is `NcZ12(v²)/(βE₈(v))`, `k = 0`. -/
theorem Do2_Z (hk : k = 0) (t : Bool) (h : F.ColNe k t) {v : ℂ} (hE : F.E8L k v ≠ 0) :
    F.Do2 k t v = LCert.NcZ12 (v ^ 2) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) /
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
  simp only [Do2, dζ, kron_self, dhL, AhL, BhL, ChL, WsL, WoBL, WoCL, sReflL, RhatSL, RhatL0, RhatL2,
    Params.RhatC, Params.ℓ₀, Params.ℓ₁, Params.ℓ₂, Params.ℓ₃, Params.Tp, Params.Tm, hSp, hSm, hSp',
    hSm', hq, hκ, hP, hq', hκ', hP', cN0L, cN1L, cN2L, cN3L, lat_a, lat_β, Bool.not_not]
  rw [eq_div_iff (mul_ne_zero hβ hE)]
  field_simp
  try rw [F.E8L_W _ t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [N₀L, N₂L, HL, PL, QL, lamL, piP, F.piS_eq t, rL_not, q₀L, F.cS_eq t, cP, DL, cLw, ζL, ζLs, hw, F.pUp_mul_pDn t, Bool.not_not, mL_not, kron_self, one_mul, mul_one]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.NcZ12, LCert.E8W]
  ring

end FParams

end HubbardAtom
