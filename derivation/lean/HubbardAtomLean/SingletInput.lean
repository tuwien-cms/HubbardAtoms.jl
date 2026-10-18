import HubbardAtomLean.Triplet

/-!
# The singlet susceptibility: the input and Proposition "pp" (singlet part)

* `Rreg'`: the regular part of Eq. (pair), the twelve simple fractions of Appendix C
  [Eßl et al., Eq. (D3)], and `piPair'`, `Params.pairC`: the pair term `c_ℓ Π(x, y)`.
* `Params.chiUpDn`: the input Eq. (chi-ud), taken as given; `Params.chiS`: the singlet
  susceptibility `χ_s = (-χ_pp↑↑ + 2χ_pp↑↓ - 2χ_0,pp)/4`.
* `regular_pp`: the regular part at the particle–particle transfer is the rank-two term
  `γ₊g₊g₊ + γ₋g₋g₋` (identities (A1) and (A3) of the note, combined).
* `Params.chiS_eq`: Proposition "pp", singlet part: `χ_s = L_s 𝓜₊ - B 𝓜₋ + χ_s,reg` for `b ≠ 0`.
-/

open Complex

namespace HubbardAtom

/-- The twelve simple fractions of Appendix C, with `xw = x + w` and `yw = y + w` as separate
arguments (the weights `q₋ = p₀ + p₁`, `q₊ = p₁ + p₂`). -/
noncomputable def Rreg' (a qm qp x y xw yw : ℂ) : ℂ :=
  qm * (-1 / ((yw + a) * (y + a) * (x - a)) + 1 / ((xw - a) * (x - a) * (y + a)) +
      1 / ((xw + a) * (y - a) * (x - a)) - 1 / ((xw + a) * (x - a) * (yw + a)) +
      1 / ((y + a) * (xw + a) * (yw + a)) + 1 / ((y + a) * (xw + a) * (x + a)) +
      1 / ((y - a) * (x + a) * (yw - a)) - 1 / ((y - a) * (xw + a) * (x + a)) +
      1 / ((xw - a) * (x + a) * (yw - a)) - 1 / ((y + a) * (xw - a) * (x + a))) -
    qp * (1 / ((y - a) * (xw - a) * (yw - a)) + 1 / ((y - a) * (xw - a) * (x - a)))

/-- `Π(x, y) = (1/(x + w - a) + 1/(y - a))(1/(y + w - a) + 1/(x - a))` of Eq. (pair). -/
noncomputable def piPair' (a x y xw yw : ℂ) : ℂ :=
  (1 / (xw - a) + 1 / (y - a)) * (1 / (yw - a) + 1 / (x - a))

/-- Identities (A1) and (A3) at the particle–particle transfer `w' = b - x - y`, where
`x + w' = b - y` and `y + w' = b - x`: the regular part `(c Π + R_reg)/2` with `c = ρ/b` is
`γ₊g₊(t)g₊(t') + γ₋g₋(t)g₋(t')`, with `t - t₊ = (x - a)(b - x - a)`, `t - t₋ = (x + a)(b - x + a)`
(Eq. pp-ids) and `γ± = a(ρ ± 1)(2a ∓ b)/(2b)`. -/
theorem regular_pp {a b x y qm qp : ℂ} (hq : qm + qp = 1) (hb : b ≠ 0)
    (h₁ : x - a ≠ 0) (h₂ : x + a ≠ 0) (h₃ : b - x - a ≠ 0) (h₄ : b - x + a ≠ 0)
    (h₅ : y - a ≠ 0) (h₆ : y + a ≠ 0) (h₇ : b - y - a ≠ 0) (h₈ : b - y + a ≠ 0) :
    ((qp - qm) / b * piPair' a x y (b - y) (b - x) + Rreg' a qm qp x y (b - y) (b - x)) / 2 =
      a * (qp - qm + 1) * (2 * a - b) / (2 * b) / ((x - a) * (b - x - a)) /
          ((y - a) * (b - y - a)) +
        a * (qp - qm - 1) * (2 * a + b) / (2 * b) / ((x + a) * (b - x + a)) /
          ((y + a) * (b - y + a)) := by
  obtain rfl : qp = 1 - qm := by linear_combination hq
  unfold piPair' Rreg'
  field_simp
  ring

namespace Params

variable (A : Params)

/-- The pair coefficient `c_ℓ = ρ/(2μ + iΩ_ℓ)` of Eq. (pair); at `μ = 0` its limit `βp₀ δ_ℓ0`. -/
noncomputable def pairC (ℓ : ℤ) : ℂ :=
  if A.μ = 0 ∧ ℓ = 0 then (A.β * A.p₀ : ℝ) else A.ρ / (2 * A.μ + A.ω ℓ * I)

/-- The input, Eq. (chi-ud): the opposite-spin generalized susceptibility `χ↑↓(ν_n, ν_m, ω_k)`
[Eßl et al., Eq. (D3)], taken as given. Here `x_n + w = x_{n+k}` (`x_add_w`). -/
noncomputable def chiUpDn (k n m : ℤ) : ℂ :=
  -4 * A.β * A.a ^ 2 * A.p₁ * A.J (A.x n) * A.J (A.x (n + k)) * kron n m +
    4 * A.β * A.a ^ 2 * (A.p₀ * A.p₂ - A.p₁ ^ 2) * A.J (A.x n) * A.J (A.x m) * kron k 0 +
    A.pairC (n + m + k + 1) * piPair' A.a (A.x n) (A.x m) (A.x (n + k)) (A.x (m + k)) +
    Rreg' A.a A.qm A.qp (A.x n) (A.x m) (A.x (n + k)) (A.x (m + k))

/-- The particle–particle form `χ_pp↑↓(ν, ν', ω) = χ↑↓(ν, ν', -ω - ν - ν')`. -/
noncomputable def chiPPUpDn (k n m : ℤ) : ℂ := A.chiUpDn (-k - n - m - 1) n m

/-- The singlet susceptibility `χ_s = (-χ_pp↑↑ + 2χ_pp↑↓ - 2χ_0,pp)/4`, `χ_0,pp = δₙₘ B(ν_n)`. -/
noncomputable def chiS (k n m : ℤ) : ℂ :=
  (-A.chiPPUpUp k n m + 2 * A.chiPPUpDn k n m - 2 * (kron n m * A.Bpp k n)) / 4

/-! ### The scalars of the singlet channel -/

variable (k : ℤ)

/-- `t₊ = -a² + ab`, `t₋ = -a² - ab`, the zeros of `𝒫`. -/
noncomputable def tp : ℂ := -A.a ^ 2 + A.a * A.b k
/-- `t₋ = -a² - ab`. -/
noncomputable def tm : ℂ := -A.a ^ 2 - A.a * A.b k
/-- `Aₛ = aρb + a²(1 - 8p₁)`. -/
noncomputable def As : ℂ := A.a * A.ρ * A.b k + A.a ^ 2 * (1 - 8 * A.p₁)
/-- `L_s(t) = (β/2)(t + Aₛ)/𝒫(t)`. -/
noncomputable def Ls (t : ℂ) : ℂ := (A.β / 2 : ℂ) * (t + A.As k) / A.calP k t
/-- `γ₊ = a(ρ + 1)(2a - b)/(2b)`. -/
noncomputable def γp : ℂ := A.a * (A.ρ + 1) * (2 * A.a - A.b k) / (2 * A.b k)
/-- `γ₋ = a(ρ - 1)(2a + b)/(2b)`. -/
noncomputable def γm : ℂ := A.a * (A.ρ - 1) * (2 * A.a + A.b k) / (2 * A.b k)
/-- The rank-two regular part `χ_s,reg = γ₊g₊g₊ + γ₋g₋g₋` of Eq. (chi-s-reg), `g± = 1/(t - t±)`. -/
noncomputable def chiReg (n m : ℤ) : ℂ :=
  A.γp k / (A.t k n - A.tp k) / (A.t k m - A.tp k) +
    A.γm k / (A.t k n - A.tm k) / (A.t k m - A.tm k)

/-! ### From the input -/

/-- Eq. (pp-ids): `t_n - t₊ = (x_n - a)(x_{n̄} - a)`. -/
theorem t_sub_tp (n : ℤ) : A.t k n - A.tp k = (A.x n - A.a) * (A.x (mirror k n) - A.a) := by
  rw [t, tp, A.x_mirror]
  ring

/-- Eq. (pp-ids): `t_n - t₋ = (x_n + a)(x_{n̄} + a)`. -/
theorem t_sub_tm (n : ℤ) : A.t k n - A.tm k = (A.x n + A.a) * (A.x (mirror k n) + A.a) := by
  rw [t, tm, A.x_mirror]
  ring

/-- The particle–particle same-spin susceptibility, from the input (proof of Proposition "pp"):
`χ_pp↑↑ = 2Bδₙₘ - βa²(1 - ρ²)/𝒫 (δₙₘ - δ_{m n̄})`. -/
theorem chiPPUpUp_eq (n m : ℤ) :
    A.chiPPUpUp k n m = 2 * A.Bpp k n * kron n m -
      A.β * A.a ^ 2 * (1 - A.ρ ^ 2) / A.calP k (A.t k n) * (kron n m - kron m (mirror k n)) := by
  have hJJ := A.J_mul_J_mirror k n
  have hB := A.Bpp_eq k n
  by_cases hm : m = n
  · subst hm
    rw [chiPPUpUp, chiUpUp, kron_self, A.x_add_w]
    have hx : m + (-k - m - m - 1) = mirror k m := by unfold mirror; ring
    rw [hx]
    by_cases hfix : mirror k m = m
    · have h0 : -k - m - m - 1 = 0 := by unfold mirror at hfix; omega
      rw [h0, kron_self, hfix, kron_self]
      rw [hfix] at hB
      rw [hB]
      ring
    · have h0 : -k - m - m - 1 ≠ 0 := by unfold mirror at hfix; omega
      rw [kron_of_ne h0, kron_of_ne (Ne.symm hfix)]
      linear_combination (-2 : ℂ) * hB - A.β * A.a ^ 2 * (1 - A.ρ ^ 2) * hJJ
  · by_cases hm' : m = mirror k n
    · subst hm'
      have h0 : -k - n - mirror k n - 1 = 0 := by unfold mirror; ring
      rw [chiPPUpUp, chiUpUp, kron_of_ne (Ne.symm hm), h0, kron_self, A.w_zero, add_zero, kron_self]
      linear_combination (A.β * A.a ^ 2 * (1 - A.ρ ^ 2) : ℂ) * hJJ
    · have h0 : -k - n - m - 1 ≠ 0 := fun h ↦ hm' (by unfold mirror; omega)
      rw [chiPPUpUp, chiUpUp, kron_of_ne (Ne.symm hm), kron_of_ne h0, kron_of_ne hm']
      ring

/-- `c_{-k} = ρ/b` for `b ≠ 0`. -/
theorem pairC_neg (hb : A.b k ≠ 0) : A.pairC (-k) = A.ρ / A.b k := by
  rw [pairC, ite_eq_right]
  · congr 1
    unfold b w ω
    push_cast
    ring
  · rintro ⟨hμ, hk⟩
    apply hb
    have hk' : k = 0 := by omega
    rw [b, hk', A.w_zero, hμ]
    simp

theorem kron_pp (n m : ℤ) : kron (-k - n - m - 1) 0 = kron m (mirror k n) := by
  unfold kron mirror
  by_cases h : m = -n - k - 1
  · rw [ite_eq_left (by omega), ite_eq_left h]
  · rw [ite_eq_right (by omega), ite_eq_right h]

/-- The particle–particle opposite-spin susceptibility, from the input (proof of Proposition "pp"),
for `b ≠ 0`: the `δₙₘ` and `δ_{m n̄}` terms, and the pair and regular terms at the transfer
`w' = b - x_n - x_m`, where `x_n + w' = x_{m̄} = b - x_m`. -/
theorem chiPPUpDn_eq (hb : A.b k ≠ 0) (n m : ℤ) :
    A.chiPPUpDn k n m =
      -4 * A.β * A.a ^ 2 * A.p₁ / A.calP k (A.t k n) * kron n m +
        4 * A.β * A.a ^ 2 * (A.p₀ * A.p₂ - A.p₁ ^ 2) / A.calP k (A.t k n) * kron m (mirror k n) +
        A.ρ / A.b k * piPair' A.a (A.x n) (A.x m) (A.b k - A.x m) (A.b k - A.x n) +
        Rreg' A.a A.qm A.qp (A.x n) (A.x m) (A.b k - A.x m) (A.b k - A.x n) := by
  have hJJ := A.J_mul_J_mirror k n
  have hn : n + (-k - n - m - 1) = mirror k m := by unfold mirror; ring
  have hm : m + (-k - n - m - 1) = mirror k n := by unfold mirror; ring
  have hc : n + m + (-k - n - m - 1) + 1 = -k := by ring
  rw [chiPPUpDn, chiUpDn, hn, hm, hc, A.pairC_neg k hb, A.x_mirror, A.x_mirror, kron_pp k]
  congr 3
  · by_cases hnm : m = n
    · subst hnm
      rw [← A.x_mirror, kron_self, mul_one, mul_one]
      linear_combination (-4 * A.β * A.a ^ 2 * A.p₁ : ℂ) * hJJ
    · rw [kron_of_ne (Ne.symm hnm), mul_zero, mul_zero]
  · by_cases hm' : m = mirror k n
    · rw [hm', kron_self, mul_one, mul_one]
      linear_combination (4 * A.β * A.a ^ 2 * (A.p₀ * A.p₂ - A.p₁ ^ 2) : ℂ) * hJJ
    · rw [kron_of_ne hm', mul_zero, mul_zero]

/-- `L_s(t_n) + B_n = βa²(1 - 8p₁ - ρ²)/(2𝒫(t_n))`. -/
theorem Ls_add_Bpp (n : ℤ) :
    A.Ls k (A.t k n) + A.Bpp k n =
      A.β * A.a ^ 2 * (1 - 8 * A.p₁ - A.ρ ^ 2) / (2 * A.calP k (A.t k n)) := by
  rw [Ls, A.Bpp_eq_calG_div, As, calG]
  ring

/-- `p₀p₂ - p₁² = (1 - 4p₁ - ρ²)/4`. -/
theorem p₀p₂_sub (A : Params) : A.p₀ * A.p₂ - A.p₁ ^ 2 = (1 - 4 * A.p₁ - A.ρ ^ 2) / 4 := by
  have h := A.weights_sum
  have h' := A.rho_sq_add
  linear_combination (1 / 4 : ℝ) * h' + ((A.p₀ + A.p₂ + 1 - 2 * A.p₁) / 4) * h

/-- **Proposition "pp"**, singlet part, from the input: `χ_s = L_s 𝓜₊ - B 𝓜₋ + χ_s,reg` for `b ≠ 0`,
with the rank-two `χ_s,reg = γ₊g₊g₊ + γ₋g₋g₋` of Eq. (chi-s-reg). -/
theorem chiS_eq (hb : A.b k ≠ 0) (n m : ℤ) :
    A.chiS k n m = mirrorBlock k (fun j ↦ A.Ls k (A.t k j)) (fun j ↦ -A.Bpp k j) n m +
      A.chiReg k n m := by
  have hq : ((A.qm + A.qp : ℝ) : ℂ) = 1 := by
    rw [qm, qp, show A.p₀ + A.p₁ + (A.p₁ + A.p₂) = A.p₀ + 2 * A.p₁ + A.p₂ by ring, A.weights_sum,
      ofReal_one]
  push_cast at hq
  have hreg := regular_pp (a := A.a) (b := A.b k) (x := A.x n) (y := A.x m) (qm := A.qm)
    (qp := A.qp) hq hb
    (A.x_sub_ofReal_ne_zero n _) (A.x_add_ofReal_ne_zero n _)
    (by rw [← A.x_mirror]; exact A.x_sub_ofReal_ne_zero _ _)
    (by rw [← A.x_mirror]; exact A.x_add_ofReal_ne_zero _ _)
    (A.x_sub_ofReal_ne_zero m _) (A.x_add_ofReal_ne_zero m _)
    (by rw [← A.x_mirror]; exact A.x_sub_ofReal_ne_zero _ _)
    (by rw [← A.x_mirror]; exact A.x_add_ofReal_ne_zero _ _)
  have hρ : (A.qp : ℂ) - A.qm = A.ρ := by rw [qp, qm, ρ]; push_cast; ring
  rw [hρ] at hreg
  have hLB := A.Ls_add_Bpp k n
  have hw : (A.p₀ : ℂ) * A.p₂ - A.p₁ ^ 2 = (1 - 4 * A.p₁ - A.ρ ^ 2) / 4 := by
    exact_mod_cast A.p₀p₂_sub
  have hreg' : (A.ρ / A.b k * piPair' A.a (A.x n) (A.x m) (A.b k - A.x m) (A.b k - A.x n) +
      Rreg' A.a A.qm A.qp (A.x n) (A.x m) (A.b k - A.x m) (A.b k - A.x n)) / 2 =
        A.chiReg k n m := by
    rw [hreg, chiReg, A.t_sub_tp, A.t_sub_tp, A.t_sub_tm, A.t_sub_tm, A.x_mirror, A.x_mirror,
      γp, γm]
  rw [chiS, A.chiPPUpUp_eq, A.chiPPUpDn_eq k hb, ← hreg']
  unfold mirrorBlock
  rw [hw]
  linear_combination (-(kron n m + kron m (mirror k n)) / 2) * hLB

end Params

end HubbardAtom
