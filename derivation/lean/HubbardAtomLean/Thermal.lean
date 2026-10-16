import HubbardAtomLean.Basic

/-!
# The thermal functions `S_k` and `α_k`, and the period shifts of `tanh`

Lemmas "basic sum" (Eq. shift) and "thermal function" of the note, in the forms used by the
Float64 setup of HubbardAtoms.jl (`src/doped.jl`, PR #12).

* `tanh_eq_exp`, `coth_eq_exp`, `tanh_add_int_mul`, `coth_add_int_mul`: `tanh` and
  `coth = cosh/sinh` under shifts of the argument by multiples of `iπ/2`.
* `Params.S_shift`: Eq. (shift), `S_k(u) = S₀(u - w/2)`; the package evaluates `S_k` and `α_k`
  at `δ = u - w/2`, which is small at large `|k|`, instead of at `u`.
* `Params.S₀_a`, `Params.S₀_neg_a`: `S₀(±a)` from the weights (`S∓` of the note).
* `Params.αeven_eq`, `Params.αodd_eq`: Lemma "thermal function", `α_k(x) = ρS_k(x) - c̄` as
  `-h cosh(y₊)/cosh(y₋)` resp. `-h sinh(y₊)/sinh(y₋)`.
* `Params.αeven_eq_package`: the package's evaluation of `α₀`, whose prefactor
  `h e^{σ₊y₊ - σ₋y₋}` is `βp₂`, `βp₀` or `h e^{±βx}` from the stored weights.
* `singlet_tanh_reduction`, `singlet_coth_reduction`: the argument reduction of `tanh` in the
  singlet setup, `tanh(βξ/2)` (`coth` for odd `k`) `= tanh Y` with `Y = β(Aₛ/(ξ ± b/2) ± μ)/2`.
* `calP_neg_As`: `𝒫(-Aₛ) = 4a²(4ap₁ - bq₊)(4ap₁ + bq₋)`.
-/

open Complex

namespace HubbardAtom

/-- `coth z = cosh z / sinh z` (`0` where `sinh z = 0`, like `tanh` where `cosh z = 0`). -/
noncomputable def coth (z : ℂ) : ℂ := cosh z / sinh z

theorem tanh_eq_exp (z : ℂ) : tanh z = (exp (2 * z) - 1) / (exp (2 * z) + 1) := by
  have he : exp z ≠ 0 := exp_ne_zero z
  have h2 : exp (2 * z) = exp z * exp z := by rw [two_mul, exp_add]
  rw [tanh_eq_sinh_div_cosh, Complex.sinh, Complex.cosh, exp_neg, h2,
    ← mul_div_mul_right _ _ (mul_ne_zero he (two_ne_zero' ℂ))]
  congr 1 <;> field_simp

theorem coth_eq_exp (z : ℂ) : coth z = (exp (2 * z) + 1) / (exp (2 * z) - 1) := by
  have he : exp z ≠ 0 := exp_ne_zero z
  have h2 : exp (2 * z) = exp z * exp z := by rw [two_mul, exp_add]
  rw [coth, Complex.sinh, Complex.cosh, exp_neg, h2,
    ← mul_div_mul_right _ _ (mul_ne_zero he (two_ne_zero' ℂ))]
  congr 1 <;> field_simp

/-- `e^{2 · kπi/2} = (-1)^k`. -/
theorem exp_two_mul_int_mul (k : ℤ) : exp (2 * (k * (Real.pi / 2) * I)) = (-1) ^ k := by
  rw [show 2 * ((k : ℂ) * (Real.pi / 2) * I) = k * (Real.pi * I) by ring, exp_int_mul,
    exp_pi_mul_I]

/-- `tanh(z + kπi/2) = tanh z` for even `k` and `coth z` for odd `k`. -/
theorem tanh_add_int_mul (z : ℂ) (k : ℤ) :
    tanh (z + k * (Real.pi / 2) * I) = if Even k then tanh z else coth z := by
  rw [tanh_eq_exp, mul_add, exp_add, exp_two_mul_int_mul]
  split_ifs with hk
  · rw [hk.neg_one_zpow, mul_one, tanh_eq_exp]
  · rw [(Int.not_even_iff_odd.mp hk).neg_one_zpow, coth_eq_exp]
    rw [show exp (2 * z) * -1 - 1 = -(exp (2 * z) + 1) by ring,
      show exp (2 * z) * -1 + 1 = -(exp (2 * z) - 1) by ring, neg_div_neg_eq]

/-- `coth(z + kπi/2) = coth z` for even `k` and `tanh z` for odd `k`. -/
theorem coth_add_int_mul (z : ℂ) (k : ℤ) :
    coth (z + k * (Real.pi / 2) * I) = if Even k then coth z else tanh z := by
  rw [coth_eq_exp, mul_add, exp_add, exp_two_mul_int_mul]
  split_ifs with hk
  · rw [hk.neg_one_zpow, mul_one, coth_eq_exp]
  · rw [(Int.not_even_iff_odd.mp hk).neg_one_zpow, tanh_eq_exp]
    rw [show exp (2 * z) * -1 + 1 = -(exp (2 * z) - 1) by ring,
      show exp (2 * z) * -1 - 1 = -(exp (2 * z) + 1) by ring, neg_div_neg_eq]

theorem even_zero' : Even (0 : ℤ) := ⟨0, rfl⟩

namespace Params

variable (A : Params)

/-- The Matsubara sum `S_k(x) = PV ∑ₙ 1/(zₙ - x)` of Eq. (S): `-(β/2) tanh(β(x - μ)/2)` for even
`k`, `-(β/2) coth(β(x - μ)/2)` for odd `k`. -/
noncomputable def S (k : ℤ) (x : ℂ) : ℂ :=
  -((A.β : ℂ) / 2) * if Even k then tanh (A.β * (x - A.μ) / 2) else coth (A.β * (x - A.μ) / 2)

/-- Eq. (shift): `S_k(u) = S₀(u - w/2)` for every `k`. -/
theorem S_shift (k : ℤ) (u : ℂ) : A.S k u = A.S 0 (u - A.w k / 2) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hz : (A.β : ℂ) * (u - A.μ) / 2 =
      A.β * (u - A.w k / 2 - A.μ) / 2 + k * (Real.pi / 2) * I := by
    rw [w, ω]
    push_cast
    field_simp
    ring
  rw [S, S, ite_eq_left even_zero', hz]
  by_cases hk : Even k
  · rw [ite_eq_left hk, tanh_add_int_mul, ite_eq_left hk]
  · rw [ite_eq_right hk, coth_add_int_mul, ite_eq_right hk]

theorem exp_βa_sub_μ : A.p₂ * Real.exp (A.β * (A.a - A.μ)) = A.p₁ := A.p₁_eq_p₂_mul.symm

/-- `S₀(a) = β(p₂ - p₁)/(2(p₁ + p₂))`, the value `S₋` of `S_k` at the zeros of `Q₋`. -/
theorem S₀_a : A.S 0 A.a = A.β * (A.p₂ - A.p₁) / (2 * (A.p₁ + A.p₂)) := by
  have hq : (A.p₁ : ℂ) + A.p₂ ≠ 0 := by
    have := add_pos A.p₁_pos A.p₂_pos
    exact_mod_cast this.ne'
  have hp₂ : (A.p₂ : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.p₂_pos.ne'
  have h : exp (2 * (A.β * (A.a - A.μ) / 2 : ℂ)) = A.p₁ / A.p₂ := by
    rw [eq_div_iff hp₂, show 2 * ((A.β : ℂ) * (A.a - A.μ) / 2) = ((A.β * (A.a - A.μ) : ℝ) : ℂ) by
        push_cast; ring, ← ofReal_exp, ← ofReal_mul]
    congr 1
    rw [mul_comm, A.exp_βa_sub_μ]
  rw [S, ite_eq_left even_zero', tanh_eq_exp, h]
  field_simp
  ring

/-- `S₀(-a) = β(p₁ - p₀)/(2(p₀ + p₁))`, the value `S₊` of `S_k` at the zeros of `Q₊`. -/
theorem S₀_neg_a : A.S 0 (-A.a) = A.β * (A.p₁ - A.p₀) / (2 * (A.p₀ + A.p₁)) := by
  have hq : (A.p₀ : ℂ) + A.p₁ ≠ 0 := by
    have := add_pos A.p₀_pos A.p₁_pos
    exact_mod_cast this.ne'
  have hp₁ : (A.p₁ : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.p₁_pos.ne'
  have h : exp (2 * (A.β * (-A.a - A.μ) / 2 : ℂ)) = A.p₀ / A.p₁ := by
    rw [eq_div_iff hp₁,
      show 2 * ((A.β : ℂ) * (-A.a - A.μ) / 2) = ((-(A.β * (A.a + A.μ)) : ℝ) : ℂ) by
        push_cast; ring, ← ofReal_exp, ← ofReal_mul, A.p₁_eq_p₀_mul]
    congr 1
    rw [mul_comm A.p₀, ← mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, one_mul]
  rw [S, ite_eq_left even_zero', tanh_eq_exp, h]
  field_simp
  ring

/-! ### The thermal function -/

/-- `c̄ = β(p₀ + p₂)/2`. -/
noncomputable def cbar : ℝ := A.β * (A.p₀ + A.p₂) / 2

/-- `h = β√(p₀p₂)`. -/
noncomputable def hth : ℝ := A.β * Real.sqrt (A.p₀ * A.p₂)

/-- The thermal function `α_k(x) = ρS_k(x) - c̄` of Eq. (alpha-def). -/
noncomputable def α (k : ℤ) (x : ℂ) : ℂ := A.ρ * A.S k x - A.cbar

/-- `√(p₀p₂) = p₀ e^{βμ}`. -/
theorem sqrt_p₀p₂ : Real.sqrt (A.p₀ * A.p₂) = A.p₀ * Real.exp (A.β * A.μ) := by
  rw [A.p₂_eq_p₀_mul, show 2 * A.β * A.μ = A.β * A.μ + A.β * A.μ by ring, Real.exp_add,
    show A.p₀ * (A.p₀ * (Real.exp (A.β * A.μ) * Real.exp (A.β * A.μ))) =
      (A.p₀ * Real.exp (A.β * A.μ)) ^ 2 by ring]
  exact Real.sqrt_sq (mul_pos A.p₀_pos (Real.exp_pos _)).le

/-- `p₂ = √(p₀p₂) e^{βμ}` and `p₀ = √(p₀p₂) e^{-βμ}`, as complex numbers. -/
theorem p₂_eq_sqrt : (A.p₂ : ℂ) = Real.sqrt (A.p₀ * A.p₂) * exp (A.β * A.μ) := by
  rw [A.sqrt_p₀p₂, ← ofReal_mul, ← ofReal_exp, ← ofReal_mul, mul_assoc, ← Real.exp_add,
    ← two_mul, ← mul_assoc, ← A.p₂_eq_p₀_mul]

theorem p₀_eq_sqrt : (A.p₀ : ℂ) = Real.sqrt (A.p₀ * A.p₂) * exp (-(A.β * A.μ)) := by
  rw [A.sqrt_p₀p₂, ← ofReal_mul, ← ofReal_neg, ← ofReal_exp, ← ofReal_mul, mul_assoc,
    ← Real.exp_add, add_neg_cancel, Real.exp_zero, mul_one]

/-- Lemma "thermal function", even `k`: `α_k(x) = -h cosh(y₊)/cosh(y₋)`, `y± = β(x ± μ)/2`, which
avoids the cancellation of `ρS` and `c̄` at low temperature. -/
theorem αeven_eq {k : ℤ} (hk : Even k) (x : ℂ) (hc : cosh (A.β * (x - A.μ) / 2) ≠ 0) :
    A.α k x = -(A.hth : ℂ) * cosh (A.β * (x + A.μ) / 2) / cosh (A.β * (x - A.μ) / 2) := by
  have hyp : (A.β : ℂ) * (x + A.μ) / 2 = A.β * (x - A.μ) / 2 + A.β * A.μ := by ring
  rw [α, S, ite_eq_left hk, ρ, cbar, hth, hyp, tanh_eq_sinh_div_cosh, Complex.cosh_add]
  push_cast
  rw [A.p₂_eq_sqrt, A.p₀_eq_sqrt]
  generalize (A.β : ℂ) * (x - A.μ) / 2 = y at hc ⊢
  generalize (A.β : ℂ) * A.μ = c
  have hf := exp_ne_zero c
  rw [show cosh c = (exp c + (exp c)⁻¹) / 2 by rw [Complex.cosh, exp_neg],
    show sinh c = (exp c - (exp c)⁻¹) / 2 by rw [Complex.sinh, exp_neg], exp_neg]
  field_simp
  ring

/-- Lemma "thermal function", odd `k`: `α_k(x) = -h sinh(y₊)/sinh(y₋)`. -/
theorem αodd_eq {k : ℤ} (hk : ¬Even k) (x : ℂ) (hs : sinh (A.β * (x - A.μ) / 2) ≠ 0) :
    A.α k x = -(A.hth : ℂ) * sinh (A.β * (x + A.μ) / 2) / sinh (A.β * (x - A.μ) / 2) := by
  have hyp : (A.β : ℂ) * (x + A.μ) / 2 = A.β * (x - A.μ) / 2 + A.β * A.μ := by ring
  rw [α, S, ite_eq_right hk, ρ, cbar, hth, hyp, coth, Complex.sinh_add]
  push_cast
  rw [A.p₂_eq_sqrt, A.p₀_eq_sqrt]
  generalize (A.β : ℂ) * (x - A.μ) / 2 = y at hs ⊢
  generalize (A.β : ℂ) * A.μ = c
  have hf := exp_ne_zero c
  rw [show cosh c = (exp c + (exp c)⁻¹) / 2 by rw [Complex.cosh, exp_neg],
    show sinh c = (exp c - (exp c)⁻¹) / 2 by rw [Complex.sinh, exp_neg], exp_neg]
  field_simp
  ring

/-- `cosh z = e^{σz}(1 + e^{-2σz})/2` for `σ = ±1`. -/
theorem _root_.HubbardAtom.cosh_eq_exp_mul {σ : ℂ} (hσ : σ = 1 ∨ σ = -1) (z : ℂ) :
    cosh z = exp (σ * z) * (1 + exp (-2 * σ * z)) / 2 := by
  rw [Complex.cosh, mul_add, mul_one, ← exp_add]
  rcases hσ with rfl | rfl
  · ring_nf
  · ring_nf

/-- The package's evaluation of `α₀` (`_α` in `src/doped.jl`): for signs `σ₊, σ₋ = ±1` (chosen
there as the signs of `Re y±`, so that no exponential overflows),
`α₀(x) = -pre (1 + e^{-2σ₊y₊})/(1 + e^{-2σ₋y₋})`, with the prefactor `pre = h e^{σ₊y₊ - σ₋y₋}`
taken from the stored weights: `βp₂` if `σ₊ = σ₋ = 1`, `βp₀` if `σ₊ = σ₋ = -1`, and
`β√(p₀p₂) e^{σ₊βx}` otherwise. -/
theorem αeven_eq_package (x : ℂ) {σp σm : ℂ} (hσp : σp = 1 ∨ σp = -1) (hσm : σm = 1 ∨ σm = -1)
    (hc : cosh (A.β * (x - A.μ) / 2) ≠ 0) :
    A.α 0 x = -(A.β * (if σp = σm then (if σp = 1 then (A.p₂ : ℂ) else A.p₀) else
          Real.sqrt (A.p₀ * A.p₂) * exp (σp * A.β * x))) *
        (1 + exp (-2 * σp * (A.β * (x + A.μ) / 2))) /
          (1 + exp (-2 * σm * (A.β * (x - A.μ) / 2))) := by
  have hcm := hc
  rw [cosh_eq_exp_mul hσm] at hcm
  have hE := exp_ne_zero (σm * (A.β * (x - A.μ) / 2))
  have h₁ : 1 + exp (-2 * σm * (A.β * (x - A.μ) / 2)) ≠ 0 := by
    intro h0
    apply hcm
    rw [h0, mul_zero, zero_div]
  -- the prefactor `h e^{σ₊y₊ - σ₋y₋}`
  have hpre : (A.hth : ℂ) * exp (σp * (A.β * (x + A.μ) / 2)) / exp (σm * (A.β * (x - A.μ) / 2)) =
      A.β * (if σp = σm then (if σp = 1 then (A.p₂ : ℂ) else A.p₀) else
        Real.sqrt (A.p₀ * A.p₂) * exp (σp * A.β * x)) := by
    rw [mul_div_assoc, ← exp_sub, hth]
    push_cast
    rcases hσp with rfl | rfl <;> rcases hσm with rfl | rfl
    · rw [ite_eq_left rfl, ite_eq_left rfl, A.p₂_eq_sqrt,
        show (1 : ℂ) * (A.β * (x + A.μ) / 2) - 1 * (A.β * (x - A.μ) / 2) = A.β * A.μ by ring]
      ring
    · rw [ite_eq_right (by norm_num),
        show (1 : ℂ) * (A.β * (x + A.μ) / 2) - -1 * (A.β * (x - A.μ) / 2) = 1 * A.β * x by ring]
      ring
    · rw [ite_eq_right (by norm_num),
        show (-1 : ℂ) * (A.β * (x + A.μ) / 2) - 1 * (A.β * (x - A.μ) / 2) = -1 * A.β * x by ring]
      ring
    · rw [ite_eq_left rfl, ite_eq_right (by norm_num), A.p₀_eq_sqrt,
        show (-1 : ℂ) * (A.β * (x + A.μ) / 2) - -1 * (A.β * (x - A.μ) / 2) = -(A.β * A.μ) by ring]
      ring
  rw [A.αeven_eq even_zero' x hc, cosh_eq_exp_mul hσp (A.β * (x + A.μ) / 2),
    cosh_eq_exp_mul hσm (A.β * (x - A.μ) / 2), ← hpre]
  field_simp

end Params

/-! ### The singlet setup -/

/-- The argument reduction of the singlet setup (`_pp_setup_raw`): for `ξ² = b²/4 + Aₛ`,
`b = 2μ - w`, `w = iω_k`, and `sg = ±1` with `ξ + sg b/2 ≠ 0`, the argument `βξ/2` of `tanh`
(`coth` for odd `k`) differs from `Y = β(Aₛ/(ξ + sg b/2) + sg μ)/2` by `-sg kπi/2`. At large `|k|`,
`Y` is small while `βξ/2` is not, and its rounding would cost `log₂|βξ/2|` bits. -/
theorem singlet_arg_eq {β μ ω As ξ sg : ℂ} {k : ℤ} (hβ : β ≠ 0) (hω : ω = 2 * k * Real.pi / β)
    (hξ : ξ ^ 2 = (2 * μ - ω * I) ^ 2 / 4 + As) (hsg : sg = 1 ∨ sg = -1)
    (hd : ξ + sg * (2 * μ - ω * I) / 2 ≠ 0) :
    β * ξ / 2 = β * (As / (ξ + sg * (2 * μ - ω * I) / 2) + sg * μ) / 2 +
      (-sg * k) * (Real.pi / 2) * I := by
  have hsq : sg ^ 2 = 1 := by rcases hsg with rfl | rfl <;> ring
  have hAs : As = (ξ - sg * (2 * μ - ω * I) / 2) * (ξ + sg * (2 * μ - ω * I) / 2) := by
    linear_combination (-1 : ℂ) * hξ + ((2 * μ - ω * I) ^ 2 / 4) * hsq
  rw [hAs, mul_div_cancel_right₀ _ hd, hω]
  field_simp
  ring

theorem singlet_tanh_reduction {β μ ω As ξ : ℂ} {k : ℤ} (hk : Even k) (hβ : β ≠ 0)
    (hω : ω = 2 * k * Real.pi / β) (hξ : ξ ^ 2 = (2 * μ - ω * I) ^ 2 / 4 + As) {sg : ℂ}
    (hsg : sg = 1 ∨ sg = -1) (hd : ξ + sg * (2 * μ - ω * I) / 2 ≠ 0) :
    tanh (β * ξ / 2) = tanh (β * (As / (ξ + sg * (2 * μ - ω * I) / 2) + sg * μ) / 2) := by
  rw [singlet_arg_eq hβ hω hξ hsg hd]
  rcases hsg with rfl | rfl
  · have := tanh_add_int_mul (β * (As / (ξ + 1 * (2 * μ - ω * I) / 2) + 1 * μ) / 2) (-k)
    rw [ite_eq_left hk.neg] at this
    convert this using 3
    push_cast
    ring
  · have := tanh_add_int_mul (β * (As / (ξ + -1 * (2 * μ - ω * I) / 2) + -1 * μ) / 2) k
    rw [ite_eq_left hk] at this
    convert this using 3
    ring

theorem singlet_coth_reduction {β μ ω As ξ : ℂ} {k : ℤ} (hk : ¬Even k) (hβ : β ≠ 0)
    (hω : ω = 2 * k * Real.pi / β) (hξ : ξ ^ 2 = (2 * μ - ω * I) ^ 2 / 4 + As) {sg : ℂ}
    (hsg : sg = 1 ∨ sg = -1) (hd : ξ + sg * (2 * μ - ω * I) / 2 ≠ 0) :
    coth (β * ξ / 2) = tanh (β * (As / (ξ + sg * (2 * μ - ω * I) / 2) + sg * μ) / 2) := by
  rw [singlet_arg_eq hβ hω hξ hsg hd]
  rcases hsg with rfl | rfl
  · have := coth_add_int_mul (β * (As / (ξ + 1 * (2 * μ - ω * I) / 2) + 1 * μ) / 2) (-k)
    rw [ite_eq_right (fun h ↦ hk (by simpa using h.neg))] at this
    convert this using 3
    push_cast
    ring
  · have := coth_add_int_mul (β * (As / (ξ + -1 * (2 * μ - ω * I) / 2) + -1 * μ) / 2) k
    rw [ite_eq_right hk] at this
    convert this using 3
    ring

/-- `𝒫(-Aₛ) = 4a²(4ap₁ - bq₊)(4ap₁ + bq₋)` for `𝒫(t) = (t + a²)² - a²b²` and
`Aₛ = aρb + a²(1 - 8p₁)`: the package computes `𝒫(-Aₛ)` in this factored form. -/
theorem calP_neg_As {a b p₀ p₁ p₂ : ℂ} (hs : p₀ + 2 * p₁ + p₂ = 1) :
    (-(a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁)) + a ^ 2) ^ 2 - a ^ 2 * b ^ 2 =
      4 * a ^ 2 * (4 * a * p₁ - b * (p₁ + p₂)) * (4 * a * p₁ + b * (p₀ + p₁)) := by
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  ring

end HubbardAtom
