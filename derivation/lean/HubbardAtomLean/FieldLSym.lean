import HubbardAtomLean.FieldLVertex
import HubbardAtomLean.FieldLSymCert

/-!
# The longitudinal inverse is two-sided

`Γ_L` of Eq. (field-L) is symmetric, `Γ_{L,σt}(ν, ν') = Γ_{L,tσ}(ν', ν)` (`gammaL_symm`), hence so
is `invL` (`invL_symm`), and `invL` is a left inverse of `χ_L` as well (`chiL_isLeftInverse`).

The dense part of `Γ_L` is a divided difference of `P(x)X(x)/E₈(x)`, `P(x)Y(x)/E₈(x)` (rows of
opposite spin) or `(x² + q₀)P(x)/E₈(x)` (equal spins); in `y = x²` these are partial fractions over
the roots `s_j` (`pf_E8`), and the difference of `Γ_L` and its transpose is a sum over the roots.
Each term vanishes (`sym_root`, `sym_root_self`): for equal spins by the definition of `𝔠ⱼ`, for
opposite spins by three identities at the root (`res_z`, `res_v`, `res_c`, from the certificates of
`FieldLSymCert.lean`), the first of which is Eq. (field-L-quotient).
-/

open Complex

namespace HubbardAtom

/-- **One root of the symmetry, opposite spins**: the term of the root `u` in
`Γ_L(-t, z; t, v) - Γ_L(t, v; -t, z)` vanishes, given the three identities at the root. -/
theorem sym_root {lam m ρ a q β z v u d p Pi Xi Kt Kp fp fm gp gm aT aN : ℂ} (hβ : β ≠ 0)
    (hd : d ≠ 0) (hKt : Kt ≠ 0) (hKp : Kp ≠ 0) (hzv : z - v ≠ 0) (hzv' : z + v ≠ 0)
    (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0)
    (hvu : v - u ≠ 0) (hvu' : v + u ≠ 0)
    (ez : aT * (fp + fm) = -(lam * (ρ * Xi + m * Pi)) * Kt)
    (ev : aN * (gp + gm) = -(lam * (ρ * Xi - m * Pi)) * Kp)
    (ec : 8 * a * lam * ρ * m * u ^ 2 * (u ^ 2 + q) * Kt * Kp - aT * (fp - fm) * u * Kp +
      aN * (gp - gm) * u * Kt = 0) :
    -(lam * m) * ((p * Pi / d - 4 * a * ρ * v * (p * (u ^ 2 + q) / d)) / (v ^ 2 - u ^ 2) -
          (p * Pi / d - 4 * a * ρ * z * (p * (u ^ 2 + q) / d)) / (z ^ 2 - u ^ 2)) / (z - v) -
        lam * ρ * ((p * Xi / d + 4 * a * m * v * (p * (u ^ 2 + q) / d)) / (v ^ 2 - u ^ 2) -
          (p * Xi / d + 4 * a * m * -z * (p * (u ^ 2 + q) / d)) / (z ^ 2 - u ^ 2)) / (z + v) -
      β ^ 2 * (aT / (β ^ 2 * Kt * d) * p / (v ^ 2 - u ^ 2) * (fp / (z - u) + fm / (z + u))) +
      β ^ 2 * (aN / (β ^ 2 * Kp * d) * p / (z ^ 2 - u ^ 2) *
        (gp / (v - u) + gm / (v + u))) = 0 := by
  have key : -(lam * m) * ((p * Pi / d - 4 * a * ρ * v * (p * (u ^ 2 + q) / d)) / (v ^ 2 - u ^ 2) -
          (p * Pi / d - 4 * a * ρ * z * (p * (u ^ 2 + q) / d)) / (z ^ 2 - u ^ 2)) / (z - v) -
        lam * ρ * ((p * Xi / d + 4 * a * m * v * (p * (u ^ 2 + q) / d)) / (v ^ 2 - u ^ 2) -
          (p * Xi / d + 4 * a * m * -z * (p * (u ^ 2 + q) / d)) / (z ^ 2 - u ^ 2)) / (z + v) -
      β ^ 2 * (aT / (β ^ 2 * Kt * d) * p / (v ^ 2 - u ^ 2) * (fp / (z - u) + fm / (z + u))) +
      β ^ 2 * (aN / (β ^ 2 * Kp * d) * p / (z ^ 2 - u ^ 2) * (gp / (v - u) + gm / (v + u))) =
      p / (d * (v ^ 2 - u ^ 2) * (z ^ 2 - u ^ 2) * Kt * Kp) *
        (z * Kp * (-(lam * (ρ * Xi + m * Pi)) * Kt - aT * (fp + fm)) +
          v * Kt * (aN * (gp + gm) + lam * (ρ * Xi - m * Pi) * Kp) +
          (8 * a * lam * ρ * m * u ^ 2 * (u ^ 2 + q) * Kt * Kp - aT * (fp - fm) * u * Kp +
            aN * (gp - gm) * u * Kt)) := by
    have hz2 : z ^ 2 - u ^ 2 = (z - u) * (z + u) := by ring
    have hv2 : v ^ 2 - u ^ 2 = (v - u) * (v + u) := by ring
    rw [hz2, hv2]
    field_simp
    ring
  rw [key, ec, show -(lam * (ρ * Xi + m * Pi)) * Kt - aT * (fp + fm) = 0 by linear_combination -ez,
    show aN * (gp + gm) + lam * (ρ * Xi - m * Pi) * Kp = 0 by linear_combination ev]
  ring

/-- **One root of the symmetry, equal spins**: the term of the root `u` in
`Γ_L(t, z; t, v) - Γ_L(t, v; t, z)` vanishes. -/
theorem sym_root_self {lam D q β z v u d p K fp fm : ℂ} (hβ : β ≠ 0) (hd : d ≠ 0) (hK : K ≠ 0)
    (hK' : fp + fm = K) (hzv : z + v ≠ 0) (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0)
    (hvu : v - u ≠ 0) (hvu' : v + u ≠ 0) :
    2 * lam ^ 2 * D * (p * (u ^ 2 + q) / d / (v ^ 2 - u ^ 2) -
        p * (u ^ 2 + q) / d / (z ^ 2 - u ^ 2)) /
          (z + v) -
        β ^ 2 * (2 * lam ^ 2 * D * (u ^ 2 + q) / (β ^ 2 * K * d) * p / (v ^ 2 - u ^ 2) *
          (fp / (z - u) + fm / (z + u))) +
      β ^ 2 * (2 * lam ^ 2 * D * (u ^ 2 + q) / (β ^ 2 * K * d) * p / (z ^ 2 - u ^ 2) *
          (fp / (v - u) + fm / (v + u))) = 0 := by
  have hz2 : z ^ 2 - u ^ 2 = (z - u) * (z + u) := by ring
  have hv2 : v ^ 2 - u ^ 2 = (v - u) * (v + u) := by ring
  rw [← hK'] at hK ⊢
  rw [hz2, hv2]
  field_simp
  ring

namespace FParams

variable (F : FParams) (k : ℤ)

-- the connections are `field_simp`/`ring` computations: one at a time
set_option Elab.async false

/-! ### The data in `y = x²` -/

/-- `p(y)`, with `P(x) = p(x²)`. -/
noncomputable def pY (y : ℂ) : ℂ := (y + F.q₀L k) ^ 2 - 4 * F.a ^ 2 * y

/-- `Π(y) = (y + q₀)² + 4a²r↑r↓y + λ²d_π`. -/
noncomputable def PiY (y : ℂ) : ℂ :=
  (y + F.q₀L k) ^ 2 + 4 * F.a ^ 2 * F.rUp * F.rDn * y + F.lamL ^ 2 * (F.piP - F.piS)

/-- `Ξ(y) = (y + q₀)² - 4a²r↑r↓y - λ²d_π`. -/
noncomputable def XiY (y : ℂ) : ℂ :=
  (y + F.q₀L k) ^ 2 - 4 * F.a ^ 2 * F.rUp * F.rDn * y - F.lamL ^ 2 * (F.piP - F.piS)

/-- `E(y)`, with `E₈(x) = E(x²)`. -/
noncomputable def EY (y : ℂ) : ℂ :=
  y ^ 4 + F.E8b k 3 * y ^ 3 + F.E8b k 2 * y ^ 2 + F.E8b k 1 * y + F.E8b k 0

/-- `pΠ - E`, a cubic. -/
noncomputable def R1Y (y : ℂ) : ℂ := F.pY k y * F.PiY k y - F.EY k y

/-- `p(y)(y + q₀)`, a cubic. -/
noncomputable def R2Y (y : ℂ) : ℂ := F.pY k y * (y + F.q₀L k)

/-- `pΞ - E`, a cubic. -/
noncomputable def R3Y (y : ℂ) : ℂ := F.pY k y * F.XiY k y - F.EY k y

theorem PL_pY (x : ℂ) : F.PL k x = F.pY k (x ^ 2) := by
  simp only [PL, pY, q₀L]
  ring

theorem E8L_EY (x : ℂ) : F.E8L k x = F.EY k (x ^ 2) := F.E8L_eq_quartic k x

theorem rL_mul_not (t : Bool) : (F.rL t : ℂ) * F.rL (!t) = F.rUp * F.rDn := by
  cases t <;> simp [mul_comm]

theorem rL_add_not (t : Bool) : (F.rL t : ℂ) + F.rL (!t) = 2 * F.ρ := by
  rw [F.rL_cast t, F.rL_not t]
  push_cast
  ring

theorem rL_not_sub (t : Bool) : (F.rL (!t) : ℂ) - F.rL t = 2 * F.mL t := by
  rw [F.rL_cast t, F.rL_not t]
  push_cast
  ring

/-- `X(x) = H_t(-x)H_{-t}(-x) + λ²d_π = Π(x²) - 4aρx(x² + q₀)`. -/
theorem X_split (t : Bool) (x : ℂ) :
    F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) (-x) + F.lamL ^ 2 * (F.piP - F.piS) =
      F.PiY k (x ^ 2) - 4 * F.a * F.ρ * x * (x ^ 2 + F.q₀L k) := by
  simp only [HL, PiY]
  linear_combination (-2 * F.a * x * (x ^ 2 + F.q₀L k)) * F.rL_add_not t +
    4 * F.a ^ 2 * x ^ 2 * F.rL_mul_not t

/-- `Y(x) = H_t(-x)H_{-t}(x) - λ²d_π = Ξ(x²) + 4amx(x² + q₀)`. -/
theorem Y_split (t : Bool) (x : ℂ) :
    F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) x - F.lamL ^ 2 * (F.piP - F.piS) =
      F.XiY k (x ^ 2) + 4 * F.a * F.mL t * x * (x ^ 2 + F.q₀L k) := by
  simp only [HL, XiY]
  linear_combination (2 * F.a * x * (x ^ 2 + F.q₀L k)) * F.rL_not_sub t -
    4 * F.a ^ 2 * x ^ 2 * F.rL_mul_not t

theorem R1Y_cubic (y : ℂ) : F.R1Y k y = fd3 (F.R1Y k) 0 + fd3 (F.R1Y k) 1 * y +
    fd3 (F.R1Y k) 2 * y ^ 2 + fd3 (F.R1Y k) 3 * y ^ 3 := by
  simp only [fd3, R1Y, pY, PiY, EY]
  ring

theorem R2Y_cubic (y : ℂ) : F.R2Y k y = fd3 (F.R2Y k) 0 + fd3 (F.R2Y k) 1 * y +
    fd3 (F.R2Y k) 2 * y ^ 2 + fd3 (F.R2Y k) 3 * y ^ 3 := by
  simp only [fd3, R2Y, pY]
  ring

theorem R3Y_cubic (y : ℂ) : F.R3Y k y = fd3 (F.R3Y k) 0 + fd3 (F.R3Y k) 1 * y +
    fd3 (F.R3Y k) 2 * y ^ 2 + fd3 (F.R3Y k) 3 * y ^ 3 := by
  simp only [fd3, R3Y, pY, XiY, EY]
  ring

/-! ### Partial fractions over the roots -/

/-- **Partial fractions** of a cubic in `x²` over `E₈(x) = ∏ⱼ(x² - s_j)`. -/
theorem pf_E8 {u : Fin 4 → ℂ} (root : ∀ j, F.E8L k (u j) = 0)
    (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) {N : ℂ → ℂ}
    (hN : ∀ y, N y = fd3 N 0 + fd3 N 1 * y + fd3 N 2 * y ^ 2 + fd3 N 3 * y ^ 3) {x : ℂ}
    (hx : ∀ j, x ^ 2 - u j ^ 2 ≠ 0) :
    N (x ^ 2) / F.E8L k x = ∑ j, N (u j ^ 2) / dprod (fun i ↦ u i ^ 2) j / (x ^ 2 - u j ^ 2) := by
  rw [F.E8L_eq_prod k root sep, pf4 hN (sep 0 1 (by decide)) (sep 0 2 (by decide))
    (sep 0 3 (by decide)) (sep 1 2 (by decide)) (sep 1 3 (by decide)) (sep 2 3 (by decide))
    (hx 0) (hx 1) (hx 2) (hx 3)]
  simp only [Fin.sum_univ_four, dprod]

theorem E8L_ne_of {u : Fin 4 → ℂ} (root : ∀ j, F.E8L k (u j) = 0)
    (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) {x : ℂ} (hx : ∀ j, x ^ 2 - u j ^ 2 ≠ 0) :
    F.E8L k x ≠ 0 := by
  rw [F.E8L_eq_prod k root sep]
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero (hx 0) (hx 1)) (hx 2)) (hx 3)

/-- `P(x)X(x)/E₈(x) = 1 + ∑ⱼ (R₁(s_j) - 4aρxR₂(s_j))/(E'ⱼ(x² - s_j))`. -/
theorem X_pf (t : Bool) {u : Fin 4 → ℂ} (root : ∀ j, F.E8L k (u j) = 0)
    (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) {x : ℂ} (hx : ∀ j, x ^ 2 - u j ^ 2 ≠ 0) :
    F.PL k x * (F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) (-x) + F.lamL ^ 2 * (F.piP - F.piS)) /
        F.E8L k x =
      1 + ∑ j, (F.R1Y k (u j ^ 2) - 4 * F.a * F.ρ * x * F.R2Y k (u j ^ 2)) /
        dprod (fun i ↦ u i ^ 2) j / (x ^ 2 - u j ^ 2) := by
  have hE := F.E8L_ne_of k root sep hx
  have h1 := (div_eq_iff hE).mp (F.pf_E8 k root sep (F.R1Y_cubic k) hx)
  have h2 := (div_eq_iff hE).mp (F.pf_E8 k root sep (F.R2Y_cubic k) hx)
  have e : F.PL k x * (F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) (-x) +
      F.lamL ^ 2 * (F.piP - F.piS)) =
      F.E8L k x + F.R1Y k (x ^ 2) - 4 * F.a * F.ρ * x * F.R2Y k (x ^ 2) := by
    rw [F.X_split k t x, F.PL_pY k x, F.E8L_EY k x]
    simp only [R1Y, R2Y]
    ring
  rw [div_eq_iff hE, e]
  simp only [Fin.sum_univ_four] at h1 h2 ⊢
  linear_combination h1 - 4 * F.a * F.ρ * x * h2

/-- `P(x)Y(x)/E₈(x) = 1 + ∑ⱼ (R₃(s_j) + 4amxR₂(s_j))/(E'ⱼ(x² - s_j))`. -/
theorem Y_pf (t : Bool) {u : Fin 4 → ℂ} (root : ∀ j, F.E8L k (u j) = 0)
    (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) {x : ℂ} (hx : ∀ j, x ^ 2 - u j ^ 2 ≠ 0) :
    F.PL k x * (F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) x - F.lamL ^ 2 * (F.piP - F.piS)) /
        F.E8L k x =
      1 + ∑ j, (F.R3Y k (u j ^ 2) + 4 * F.a * F.mL t * x * F.R2Y k (u j ^ 2)) /
        dprod (fun i ↦ u i ^ 2) j / (x ^ 2 - u j ^ 2) := by
  have hE := F.E8L_ne_of k root sep hx
  have h1 := (div_eq_iff hE).mp (F.pf_E8 k root sep (F.R3Y_cubic k) hx)
  have h2 := (div_eq_iff hE).mp (F.pf_E8 k root sep (F.R2Y_cubic k) hx)
  have e : F.PL k x * (F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) x -
      F.lamL ^ 2 * (F.piP - F.piS)) =
      F.E8L k x + F.R3Y k (x ^ 2) + 4 * F.a * F.mL t * x * F.R2Y k (x ^ 2) := by
    rw [F.Y_split k t x, F.PL_pY k x, F.E8L_EY k x]
    simp only [R3Y, R2Y]
    ring
  rw [div_eq_iff hE, e]
  simp only [Fin.sum_univ_four] at h1 h2 ⊢
  linear_combination h1 + 4 * F.a * F.mL t * x * h2

/-- `(x² + q₀)P(x)/E₈(x) = ∑ⱼ R₂(s_j)/(E'ⱼ(x² - s_j))`. -/
theorem R2_pf {u : Fin 4 → ℂ} (root : ∀ j, F.E8L k (u j) = 0)
    (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) {x : ℂ} (hx : ∀ j, x ^ 2 - u j ^ 2 ≠ 0) :
    (x ^ 2 + F.q₀L k) * F.PL k x / F.E8L k x =
      ∑ j, F.R2Y k (u j ^ 2) / dprod (fun i ↦ u i ^ 2) j / (x ^ 2 - u j ^ 2) := by
  rw [← F.pf_E8 k root sep (F.R2Y_cubic k) hx, F.PL_pY k x, R2Y]
  ring

/-! ### The identities at a root -/

theorem fnp_W (t : Bool) (u : ℂ) :
    F.fMp k t (!t) u =
      LCert.FnpW u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [fMp, Bool.not_ne_self, ↓reduceIte, αSL, Bool.not_not, F.mL_not t, HL, lamL, piP,
    F.piS_eq t, rL_not, q₀L, F.cS_eq t]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.FnpW]
  ring

theorem fnm_W (t : Bool) (u : ℂ) :
    F.fMm k t (!t) u =
      LCert.FnmW u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [fMm, Bool.not_ne_self, ↓reduceIte, αPL, Bool.not_not, HL, lamL, piP, F.piS_eq t,
    rL_not, q₀L, cP]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.FnmW]
  ring

theorem Kn_W (t : Bool) (u : ℂ) :
    F.KL k t (!t) u =
      LCert.KnW u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  rw [KL, F.fnp_W k t u, F.fnm_W k t u]
  unfold LCert.KnW LCert.FnpW LCert.FnmW
  ring

theorem aAL_not_W (t : Bool) (u : ℂ) :
    F.aAL k (!t) u = LCert.anW u F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [aAL, lamL, q₀L, DL, Bool.not_not]
  push_cast
  simp only [h₀, h₂, ht, hn, hR, hM, LCert.anW]
  ring

theorem bW_W (t : Bool) (u : ℂ) :
    -(F.lamL * (F.ρ * F.XiY k (u ^ 2) + F.mL t * F.PiY k (u ^ 2))) =
      LCert.bWW u F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  have hr : (F.rUp : ℂ) * F.rDn = (R - M) * (R + M) := by
    rw [← F.rL_mul_not t, F.rL_cast t, F.rL_not t]
    push_cast
    rw [hR, hM]
  simp only [XiY, PiY, lamL, piP, F.piS_eq t, q₀L]
  rw [h₀, h₂, ht, hn, hR, hM]
  unfold LCert.bWW
  linear_combination 16 * (F.a : ℂ) ^ 4 * u ^ 2 * (R - M) * hr

theorem bWn_W (t : Bool) (u : ℂ) :
    -(F.lamL * (F.ρ * F.XiY k (u ^ 2) - F.mL t * F.PiY k (u ^ 2))) =
      LCert.bWnW u F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  have hr : (F.rUp : ℂ) * F.rDn = (R - M) * (R + M) := by
    rw [← F.rL_mul_not t, F.rL_cast t, F.rL_not t]
    push_cast
    rw [hR, hM]
  simp only [XiY, PiY, lamL, piP, F.piS_eq t, q₀L]
  rw [h₀, h₂, ht, hn, hR, hM]
  unfold LCert.bWnW
  linear_combination 16 * (F.a : ℂ) ^ 4 * u ^ 2 * (R + M) * hr

set_option maxHeartbeats 1000000 in
-- `field_simp` and `ring` on the mode normalized with `-t`
theorem KnS_W {t : Bool} (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    (F.mL t * (F.BL t).S k u - F.cS) * F.KL k (!t) (!t) u =
      LCert.KnSW u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨nS, -, -, hS, -, -⟩ := ModeRel.values F k hr h
  have hβ := β_ne_zeroT (F := F)
  have hm := h.m_ne
  simp only [KL, fMp, fMm, ↓reduceIte, QL, αL, hS, Bool.not_not, HL, lamL, piP, F.piS_eq t,
    rL_not, q₀L, DL, cLw, F.pUp_mul_pDn t]
  field_simp
  rw [F.cS_eq t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.KnSW]
  ring

set_option maxHeartbeats 1000000 in
-- `field_simp` and `ring` on the mode normalized with `-t`
theorem GpS_W {t : Bool} (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    (F.mL t * (F.BL t).S k u - F.cS) * F.fMp k (!t) t u =
      LCert.GpSW u F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨nS, -, -, hS, -, -⟩ := ModeRel.values F k hr h
  have hβ := β_ne_zeroT (F := F)
  have hm := h.m_ne
  simp only [fMp, Bool.self_ne_not, ↓reduceIte, αSL, hS, Bool.not_not, HL, lamL, piP, F.piS_eq t,
    rL_not, q₀L, F.pUp_mul_pDn t]
  field_simp
  rw [F.cS_eq t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.GpSW]
  ring

set_option maxHeartbeats 1000000 in
-- `field_simp` and `ring` on the mode normalized with `-t`
theorem GmS_W {t : Bool} (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) :
    (F.mL t * (F.BL t).S k u - F.cS) * F.fMm k (!t) t u =
      LCert.GmSW u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨nS, -, -, hS, -, -⟩ := ModeRel.values F k hr h
  have hβ := β_ne_zeroT (F := F)
  have hm := h.m_ne
  simp only [fMm, Bool.self_ne_not, ↓reduceIte, αPL, hS, Bool.not_not, HL, lamL, piP, F.piS_eq t,
    rL_not, q₀L, cP, F.pUp_mul_pDn t]
  field_simp
  rw [F.cS_eq t]
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.GmSW]
  ring

/-- **The quotient identity** of the column spin `t` at a root, Eq. (field-L-quotient):
`a_tK_{-t} = -λ(ρΞ + mΠ)K_t`. -/
theorem res_z (t : Bool) {u : ℂ} (hE : F.E8L k u = 0) :
    F.aAL k t u * (F.fMp k t (!t) u + F.fMm k t (!t) u) =
      -(F.lamL * (F.ρ * F.XiY k (u ^ 2) + F.mL t * F.PiY k (u ^ 2))) * F.KL k t t u := by
  have c := LCert.sym_z u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
    ((F.E8L_W k t u).symm.trans hE)
  rw [← KL, F.Kn_W, F.aAL_W, F.bW_W, F.KL_W]
  linear_combination c

/-- The quotient identity of the spin `-t`, with the mode normalized with `-t`. -/
theorem res_v {t : Bool} (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) (hE : F.E8L k u = 0) :
    F.aAL k (!t) u * (F.fMp k (!t) t u + F.fMm k (!t) t u) =
      -(F.lamL * (F.ρ * F.XiY k (u ^ 2) - F.mL t * F.PiY k (u ^ 2))) * F.KL k (!t) (!t) u := by
  have c := LCert.sym_v u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
    ((F.E8L_W k t u).symm.trans hE)
  have nS := (ModeRel.values F k hr h).1
  rw [← F.aAL_not_W, ← F.GpS_W k h hr, ← F.GmS_W k h hr, ← F.bWn_W, ← F.KnS_W k h hr] at c
  apply mul_left_cancel₀ nS
  linear_combination c

/-- The constant part of the residue of `Γ_L(-t, z; t, v) - Γ_L(t, v; -t, z)` at a root. -/
theorem res_c {t : Bool} (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) (hE : F.E8L k u = 0) :
    8 * F.a * F.lamL * F.ρ * F.mL t * u ^ 2 * (u ^ 2 + F.q₀L k) * F.KL k t t u *
          F.KL k (!t) (!t) u -
        F.aAL k t u * (F.fMp k t (!t) u - F.fMm k t (!t) u) * u * F.KL k (!t) (!t) u +
      F.aAL k (!t) u * (F.fMp k (!t) t u - F.fMm k (!t) t u) * u * F.KL k t t u = 0 := by
  have c := LCert.sym_c u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
    ((F.E8L_W k t u).symm.trans hE)
  have nS := (ModeRel.values F k hr h).1
  rw [← F.KL_W, ← F.aAL_W, ← F.fnp_W, ← F.fnm_W, ← F.aAL_not_W, ← F.GpS_W k h hr,
    ← F.GmS_W k h hr, ← F.KnS_W k h hr] at c
  apply mul_left_cancel₀ nS
  simp only [lamL, q₀L]
  linear_combination c

/-! ### The symmetry of `Γ_L` -/

theorem R1Y_root {u : ℂ} (hE : F.E8L k u = 0) : F.R1Y k (u ^ 2) = F.PL k u * F.PiY k (u ^ 2) := by
  rw [R1Y, ← F.E8L_EY, hE, ← F.PL_pY, sub_zero]

theorem R3Y_root {u : ℂ} (hE : F.E8L k u = 0) : F.R3Y k (u ^ 2) = F.PL k u * F.XiY k (u ^ 2) := by
  rw [R3Y, ← F.E8L_EY, hE, ← F.PL_pY, sub_zero]

theorem R2Y_eq (u : ℂ) : F.R2Y k (u ^ 2) = F.PL k u * (u ^ 2 + F.q₀L k) := by
  rw [R2Y, ← F.PL_pY]

/-- The term of the root `u_j` in `Γ_L(-t, z; t, v) - Γ_L(t, v; -t, z)`, `z = zₙ` on the lattice of
`-t`, `v = zₘ` on that of `t`. -/
noncomputable def symOpp (t : Bool) (u : Fin 4 → ℂ) (n m : ℤ) (j : Fin 4) : ℂ :=
  let z := (F.BL (!t)).z k n
  let v := (F.BL t).z k m
  let d := dprod (fun i ↦ u i ^ 2) j
  (-(F.lamL * F.mL t)) * ((F.R1Y k (u j ^ 2) - 4 * F.a * F.ρ * v * F.R2Y k (u j ^ 2)) / d /
          (v ^ 2 - u j ^ 2) -
        (F.R1Y k (u j ^ 2) - 4 * F.a * F.ρ * z * F.R2Y k (u j ^ 2)) / d / (z ^ 2 - u j ^ 2)) /
        (z - v) -
      F.lamL * F.ρ * ((F.R3Y k (u j ^ 2) + 4 * F.a * F.mL t * v * F.R2Y k (u j ^ 2)) / d /
          (v ^ 2 - u j ^ 2) -
        (F.R3Y k (u j ^ 2) + 4 * F.a * F.mL t * -z * F.R2Y k (u j ^ 2)) / d / (z ^ 2 - u j ^ 2)) /
        (z + v) -
    F.β ^ 2 * (F.cMode k t u j * F.PL k (u j) / (v ^ 2 - u j ^ 2) *
      (F.fMp k t (!t) (u j) / (z - u j) + F.fMm k t (!t) (u j) / (z + u j))) +
    F.β ^ 2 * (F.cMode k (!t) u j * F.PL k (u j) / (z ^ 2 - u j ^ 2) *
      (F.fMp k (!t) t (u j) / (v - u j) + F.fMm k (!t) t (u j) / (v + u j)))

/-- The term of the root `u_j` in `Γ_L(t, z; t, v) - Γ_L(t, v; t, z)`, `z = zₙ`, `v = zₘ` on the
lattice of `t`. -/
noncomputable def symSelf (t : Bool) (u : Fin 4 → ℂ) (n m : ℤ) (j : Fin 4) : ℂ :=
  let z := (F.BL t).z k n
  let v := (F.BL t).z k m
  let d := dprod (fun i ↦ u i ^ 2) j
  2 * F.lamL ^ 2 * F.DL t * (F.R2Y k (u j ^ 2) / d / (v ^ 2 - u j ^ 2) -
        F.R2Y k (u j ^ 2) / d / (z ^ 2 - u j ^ 2)) / (z + v) -
    F.β ^ 2 * (F.cMode k t u j * F.PL k (u j) / (v ^ 2 - u j ^ 2) *
      (F.fMp k t t (u j) / (z - u j) + F.fMm k t t (u j) / (z + u j))) +
    F.β ^ 2 * (F.cMode k t u j * F.PL k (u j) / (z ^ 2 - u j ^ 2) *
      (F.fMp k t t (u j) / (v - u j) + F.fMm k t t (u j) / (v + u j)))

variable {F k}

theorem PL_atomic {u : ℂ} (ha : F.Atomic k u) : F.PL k u = 0 := by
  obtain ⟨rfl, h2⟩ := ha
  rw [F.PL_zero, h2, sub_self]
  ring

theorem LGeneric.symOpp_eq_zero {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (t : Bool)
    (n m : ℤ) (j : Fin 4) : F.symOpp k t u n m j = 0 := by
  have h := hg t
  have h' := hg (!t)
  have hβ := β_ne_zeroT (F := F)
  have hd := dprod_ne h.sep j
  have hzv : (F.BL (!t)).z k n - (F.BL t).z k m ≠ 0 := by
    simpa using F.zL_sub_opp_ne k h.h_ne (!t) n m
  have hzv' : (F.BL (!t)).z k n + (F.BL t).z k m ≠ 0 := by
    simpa using F.zL_add_opp_ne k h.μ_ne (!t) n m
  simp only [symOpp, F.R1Y_root k (h.root j), F.R3Y_root k (h.root j), F.R2Y_eq k]
  by_cases ha : F.Atomic k (u j)
  · simp [cMode, ha, PL_atomic ha]
  · have hz := (h.off j ha (!t) n)
    have hv := (h.off j ha t m)
    have e := sym_root (p := F.PL k (u j)) hβ hd (h.K_ne j ha) (h'.K_ne j ha) hzv hzv'
      (sub_ne_zero.mpr hz.1)
      (fun h0 ↦ hz.2 (by linear_combination h0)) (sub_ne_zero.mpr hv.1)
      (fun h0 ↦ hv.2 (by linear_combination h0)) (F.res_z k t (h.root j))
      (F.res_v k h.colNe (h.roots.rel j ha) (h.root j))
      (F.res_c k h.colNe (h.roots.rel j ha) (h.root j))
    simp only [cMode, ha, ↓reduceIte, aAL]
    simp only [aAL] at e
    linear_combination e

theorem LGeneric.symSelf_eq_zero {t : Bool} {u : Fin 4 → ℂ} (h : F.LGeneric k t u) (n m : ℤ)
    (j : Fin 4) : F.symSelf k t u n m j = 0 := by
  have hβ := β_ne_zeroT (F := F)
  have hd := dprod_ne h.sep j
  have hzv := F.zL_add_self_ne k h.c_ne n m
  simp only [symSelf, F.R2Y_eq k]
  by_cases ha : F.Atomic k (u j)
  · simp [cMode, ha, PL_atomic ha]
  · have hz := (h.off j ha t n)
    have hv := (h.off j ha t m)
    have e := sym_root_self (lam := F.lamL) (D := F.DL t) (q := F.q₀L k) (p := F.PL k (u j))
      hβ hd (h.K_ne j ha) rfl hzv (sub_ne_zero.mpr hz.1)
      (fun h0 ↦ hz.2 (by linear_combination h0)) (sub_ne_zero.mpr hv.1)
      (fun h0 ↦ hv.2 (by linear_combination h0))
    simp only [cMode, ha, ↓reduceIte, aAL]
    linear_combination e

/-- **`Γ_L` is symmetric, opposite spins**: `Γ_{L,-t,t}(νₙ, νₘ) = Γ_{L,t,-t}(νₘ, νₙ)`. -/
theorem LGeneric.gammaL_symm_opp {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (t : Bool)
    (n m : ℤ) : F.gammaL k u (!t) n t m = F.gammaL k u t m (!t) n := by
  have h := hg t
  have h' := hg (!t)
  have hT := fun j ↦ LGeneric.symOpp_eq_zero hg t n m j
  have hX1 := F.X_pf k t h.root h.sep (h.lat_ne m)
  have hX2 := F.X_pf k t h.root h.sep (h'.lat_ne n)
  have hY1 := F.Y_pf k t h.root h.sep (h.lat_ne m)
  have hY2 := F.Y_pf k t h.root h.sep (x := -(F.BL (!t)).z k n)
    (by simpa only [neg_sq] using h'.lat_ne n)
  simp only [F.PL_neg, F.E8L_neg, neg_neg, neg_sq] at hY2
  have hm' : (F.mL (!t) : ℂ) = -F.mL t := by rw [F.mL_not t]; push_cast; ring
  simp only [symOpp] at hT
  simp only [gammaL, Bool.not_ne_self, Bool.self_ne_not, ↓reduceIte, Bool.not_not, hm']
  rw [show (F.BL t).z k m - (F.BL (!t)).z k n = -((F.BL (!t)).z k n - (F.BL t).z k m) by ring,
    show (F.BL t).z k m + (F.BL (!t)).z k n = (F.BL (!t)).z k n + (F.BL t).z k m by ring]
  simp only [div_neg, neg_mul, neg_div, neg_neg]
  simp only [Fin.sum_univ_four] at hX1 hX2 hY1 hY2 ⊢
  linear_combination (-(F.lamL * F.mL t) / ((F.BL (!t)).z k n - (F.BL t).z k m)) * hX1 +
    F.lamL * F.mL t / ((F.BL (!t)).z k n - (F.BL t).z k m) * hX2 +
    (-(F.lamL * F.ρ) / ((F.BL (!t)).z k n + (F.BL t).z k m)) * hY1 +
    F.lamL * F.ρ / ((F.BL (!t)).z k n + (F.BL t).z k m) * hY2 + hT 0 + hT 1 + hT 2 + hT 3

/-- **`Γ_L` is symmetric, equal spins**: `Γ_{L,tt}(νₙ, νₘ) = Γ_{L,tt}(νₘ, νₙ)`. -/
theorem LGeneric.gammaL_symm_self {t : Bool} {u : Fin 4 → ℂ} (h : F.LGeneric k t u) (n m : ℤ) :
    F.gammaL k u t n t m = F.gammaL k u t m t n := by
  rcases eq_or_ne n m with rfl | hnm
  · rfl
  have hT := fun j ↦ h.symSelf_eq_zero n m j
  have hRv := F.R2_pf k h.root h.sep (h.lat_ne m)
  have hRz := F.R2_pf k h.root h.sep (h.lat_ne n)
  simp only [symSelf] at hT
  simp only [gammaL, ↓reduceIte, kron_of_ne hnm, kron_of_ne (Ne.symm hnm), mul_zero, zero_add]
  rw [show (F.BL t).z k m + (F.BL t).z k n = (F.BL t).z k n + (F.BL t).z k m by ring]
  simp only [← div_div, Fin.sum_univ_four] at hRv hRz ⊢
  linear_combination 2 * F.lamL ^ 2 * F.DL t / ((F.BL t).z k n + (F.BL t).z k m) * hRv -
    2 * F.lamL ^ 2 * F.DL t / ((F.BL t).z k n + (F.BL t).z k m) * hRz +
    hT 0 + hT 1 + hT 2 + hT 3

/-- **`Γ_L` is symmetric**: `Γ_{L,στ}(νₙ, νₘ) = Γ_{L,τσ}(νₘ, νₙ)`. -/
theorem gammaL_symm {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) : F.gammaL k u σ n t m = F.gammaL k u t m σ n := by
  by_cases hσ : σ = t
  · subst hσ
    exact (hg σ).gammaL_symm_self n m
  · obtain rfl : σ = !t := by cases σ <;> cases t <;> simp_all
    exact LGeneric.gammaL_symm_opp hg t n m

/-- **The longitudinal inverse is symmetric**: `invL_{(τ,l),(t,m)} = invL_{(t,m),(τ,l)}`. -/
theorem invL_symm {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (τ : Bool) (l : ℤ) (t : Bool)
    (m : ℤ) : F.invL k u τ l t m = F.invL k u t m τ l := by
  have e1 := (hg t).gammaL_eq τ l m
  have e2 := (hg τ).gammaL_eq t m l
  have hs := gammaL_symm hg τ l t m
  have hδ : (if τ = t then kron l m / F.bL τ k l else 0) =
      (if t = τ then kron m l / F.bL t k m else 0) := by
    by_cases hτ : τ = t
    · subst hτ
      rcases eq_or_ne l m with rfl | hlm
      · rfl
      · simp [kron_of_ne hlm, kron_of_ne (Ne.symm hlm)]
    · simp [hτ, Ne.symm hτ]
  have hβ : (F.β : ℂ) ^ 2 ≠ 0 := pow_ne_zero 2 β_ne_zeroT
  apply mul_left_cancel₀ hβ
  linear_combination e1 - e2 + hs + (F.β : ℂ) ^ 2 * hδ

/-- **The longitudinal inverse is two-sided**: `invL χ_L = 1` as well,
`∑_{τ, l} invL_{(σ,n),(τ,l)} χ_{τt}(l, m) = δ_σt δ_nm`. -/
theorem chiL_isLeftInverse {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.invL k u σ n τ l * F.chiL τ t k l m)
      (if σ = t then kron n m else 0) := by
  convert chiL_isLeftInverse_transpose hg σ n t m using 1
  funext l
  exact Finset.sum_congr rfl fun τ _ ↦ by rw [invL_symm hg σ n τ l]

end FParams

end HubbardAtom
