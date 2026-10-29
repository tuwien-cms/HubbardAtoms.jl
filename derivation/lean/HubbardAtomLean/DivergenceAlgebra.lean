import HubbardAtomLean.DensityMagnetic

/-!
# The algebra of the divergence condition

For the density and magnetic channels, the even and odd parts in `u` (`u² = s`) of the thermal
denominator `K(u) = βH(-u) + λα_k(u)` and of `A(u) = βH(-u) - λα_k(u)`, with
`D_k = cosh βu + (-1)^k cosh βμ`, `R_k = (cosh βμ cosh βu + (-1)^k)/D_k` and
`Φ_k = D_k + εa sinh(βu)/u`, with `sinh(βu)/u` the entire `shc β u` (so `Φ_k(0) = D_k(0) + εaβ`,
`Phik_zero`): Eqs. (div-alpha), (div-Phi) and the selection of the branch
`K_e = 0` behind Eq. (div-ph) of the note, which treats `k = 0`, here for every `k`.

* `Params.αe_eq`, `αo_eq`: `α_k(±u) = -hR_k ∓ ρβ sinh(βu)/(2D_k)`, `h = β√(p₀p₂)`;
* `Params.Ke_eq`, `Ae_eq`, `Ko_eq`, `Ao_eq`: `K_e = β(s + q₁) - λhR_k`, `A_e = β(s + q₁) + λhR_k`,
  `K_o = -2aβρuΦ_k/D_k`, `A_o = -2aβρu(2 - Φ_k/D_k)`;
* `Params.Ke_mul_Ae_sub`: `K_eA_e - K_oA_o = β²e(u)`;
* `Params.Rk_sq_of_Phi`, `Ke_mul_Ae_of_Phi`: where `Φ_k = 0`, `R_k² = 1 + s sinh²(βμ)/a²` and
  `K_eA_e = β²e(u)`;
* `Params.Zph_eq_Ae`, `Zph_neg`, `lam_rho_Zph_eq`: at a root `u` of `e`, `K_eA_e = K_oA_o`, so
  `Z = uP(u)A_e/K_o`, even in `u`, and `γ = λρZ/(s - s') = -εUP(u)D_kA_e/(β(s - s')Φ_k)` where
  `Φ_k ≠ 0`; `Ke_eq_zero_of_Phi`: where `Φ_k = 0` and `A_e ≠ 0`, `K_e = 0`. The divergence itself
  (`K = 0` at a root) is not formalized here.
-/

open Complex

namespace HubbardAtom

/-- `(-1)^k`. -/
noncomputable def τk (k : ℤ) : ℂ := if Even k then 1 else -1

theorem τk_sq (k : ℤ) : τk k ^ 2 = 1 := by rw [τk]; split_ifs <;> norm_num

/-- `sinh(βu)/u`, with its limit `β` at `u = 0`: an entire function of `u`. -/
noncomputable def shc (β u : ℂ) : ℂ := dslope (fun x ↦ sinh (β * x)) 0 u

/-- `u shc(β, u) = sinh(βu)`, for all `u`. -/
theorem mul_shc (β u : ℂ) : u * shc β u = sinh (β * u) := by
  have h := sub_smul_dslope (fun x ↦ sinh (β * x)) 0 u
  simpa [shc] using h

theorem shc_of_ne {β u : ℂ} (hu : u ≠ 0) : shc β u = sinh (β * u) / u := by
  rw [eq_div_iff hu, mul_comm, mul_shc]

/-- `shc(β, 0) = β`. -/
theorem shc_zero (β : ℂ) : shc β 0 = β := by
  rw [shc, dslope_same, deriv_csinh (by fun_prop), deriv_const_mul _ (by fun_prop), deriv_id'']
  simp

theorem differentiable_shc (β : ℂ) : Differentiable ℂ (shc β) := by
  change Differentiable ℂ (dslope (fun x ↦ sinh (β * x)) 0)
  rw [← differentiableOn_univ, Complex.differentiableOn_dslope Filter.univ_mem]
  fun_prop

/-- `shc` is even in `u`. -/
theorem shc_neg (β u : ℂ) : shc β (-u) = shc β u := by
  by_cases hu : u = 0
  · rw [hu, neg_zero]
  · rw [shc_of_ne hu, shc_of_ne (neg_ne_zero.mpr hu), mul_neg, Complex.sinh_neg, neg_div_neg_eq]

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `D_k = cosh βu + (-1)^k cosh βμ`. -/
noncomputable def Dk (u : ℂ) : ℂ := cosh (A.β * u) + τk k * cosh (A.β * A.μ)

/-- `R_k = (cosh βμ cosh βu + (-1)^k)/D_k`. -/
noncomputable def Rk (u : ℂ) : ℂ := (cosh (A.β * A.μ) * cosh (A.β * u) + τk k) / A.Dk k u

/-- `Φ_k = D_k + εa sinh(βu)/u`, with `sinh(βu)/u` as the entire `shc` (`Phik_zero`). -/
noncomputable def Phik (u : ℂ) : ℂ := A.Dk k u + ε * A.a * shc A.β u

/-- `Φ_k(0) = D_k(0) + εaβ`. -/
theorem Phik_zero : A.Phik k ε 0 = A.Dk k 0 + ε * A.a * A.β := by rw [Phik, shc_zero]

/-- `Φ_k` is entire. -/
theorem differentiable_Phik : Differentiable ℂ (A.Phik k ε) := by
  unfold Phik Dk
  have := differentiable_shc A.β
  fun_prop

/-- `h sinh(βμ) = βρ/2` (Eq. weight-hyp). -/
theorem hth_mul_sinh : (A.hth : ℂ) * sinh (A.β * A.μ) = A.β * A.ρ / 2 := by
  rw [hth, ρ, Complex.sinh, exp_neg]
  push_cast
  rw [A.p₂_eq_sqrt, A.p₀_eq_sqrt, exp_neg]
  ring

/-- **Lemma "thermal function" in `D_k`, `R_k`**: `α_k(u) = -hR_k(u) - ρβ sinh(βu)/(2D_k(u))` for
`D_k(u) ≠ 0`, from `αeven_eq`, `αodd_eq` with `D_k = 2cosh(y₊)cosh(y₋)` resp.
`2sinh(y₊)sinh(y₋)`. -/
theorem α_eq_Rk {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.α k u = -A.hth * A.Rk k u - A.ρ * A.β * sinh (A.β * u) / (2 * A.Dk k u) := by
  have hρ : (A.ρ : ℂ) * A.β = 2 * A.hth * sinh (A.β * A.μ) := by
    linear_combination -2 * A.hth_mul_sinh
  have h₁ : (A.β : ℂ) * u = A.β * (u + A.μ) / 2 + A.β * (u - A.μ) / 2 := by ring
  have h₂ : (A.β : ℂ) * A.μ = A.β * (u + A.μ) / 2 - A.β * (u - A.μ) / 2 := by ring
  have hp := cosh_sq_sub_sinh_sq (A.β * (u + A.μ) / 2)
  have hm := cosh_sq_sub_sinh_sq (A.β * (u - A.μ) / 2)
  rw [Dk, τk] at hD
  rw [Rk, Dk, hρ, τk]
  rw [h₁, h₂] at hD ⊢
  simp only [cosh_add, cosh_sub, sinh_add, sinh_sub] at hD ⊢
  by_cases hk : Even k
  · rw [ite_eq_left hk] at hD ⊢
    rw [A.αeven_eq hk u (by intro h0; apply hD; rw [h0]; ring)]
    generalize cosh (A.β * (u + A.μ) / 2) = cp at *
    generalize cosh (A.β * (u - A.μ) / 2) = cm at *
    generalize sinh (A.β * (u + A.μ) / 2) = sp at *
    generalize sinh (A.β * (u - A.μ) / 2) = sm at *
    rw [show cp * cm + sp * sm + 1 * (cp * cm - sp * sm) = 2 * cp * cm by ring] at hD ⊢
    obtain ⟨hcp, hcm⟩ := mul_ne_zero_iff.mp hD
    have hcp' := (mul_ne_zero_iff.mp hcp).2
    field_simp
    linear_combination (A.hth : ℂ) * (cp ^ 2 + sp ^ 2) * hm - A.hth * hp
  · rw [ite_eq_right hk] at hD ⊢
    rw [A.αodd_eq hk u (by intro h0; apply hD; rw [h0]; ring)]
    generalize cosh (A.β * (u + A.μ) / 2) = cp at *
    generalize cosh (A.β * (u - A.μ) / 2) = cm at *
    generalize sinh (A.β * (u + A.μ) / 2) = sp at *
    generalize sinh (A.β * (u - A.μ) / 2) = sm at *
    rw [show cp * cm + sp * sm + -1 * (cp * cm - sp * sm) = 2 * sp * sm by ring] at hD ⊢
    obtain ⟨hsp, hsm⟩ := mul_ne_zero_iff.mp hD
    have hsp' := (mul_ne_zero_iff.mp hsp).2
    field_simp
    linear_combination (A.hth : ℂ) * (cp ^ 2 + sp ^ 2) * hm + A.hth * hp

theorem Dk_neg (u : ℂ) : A.Dk k (-u) = A.Dk k u := by
  rw [Dk, Dk, mul_neg, Complex.cosh_neg]

theorem Rk_neg (u : ℂ) : A.Rk k (-u) = A.Rk k u := by
  rw [Rk, Rk, A.Dk_neg, mul_neg, Complex.cosh_neg]

/-- **The even part of `α_k`**: `(α_k(u) + α_k(-u))/2 = -hR_k(u)`. -/
theorem αe_eq {u : ℂ} (hD : A.Dk k u ≠ 0) :
    (A.α k u + A.α k (-u)) / 2 = -A.hth * A.Rk k u := by
  rw [A.α_eq_Rk k hD, A.α_eq_Rk k (by rwa [A.Dk_neg]), A.Dk_neg, A.Rk_neg, mul_neg,
    Complex.sinh_neg]
  ring

/-- **The odd part of `α_k`**: `(α_k(u) - α_k(-u))/2 = -ρβ sinh(βu)/(2D_k(u))`. -/
theorem αo_eq {u : ℂ} (hD : A.Dk k u ≠ 0) :
    (A.α k u - A.α k (-u)) / 2 = -(A.ρ * A.β * sinh (A.β * u) / (2 * A.Dk k u)) := by
  rw [A.α_eq_Rk k hD, A.α_eq_Rk k (by rwa [A.Dk_neg]), A.Dk_neg, A.Rk_neg, mul_neg,
    Complex.sinh_neg]
  ring

/-- `R_k² - 1 = sinh²(βμ) sinh²(βu)/D_k²`. -/
theorem Rk_sq_sub {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.Rk k u ^ 2 - 1 = sinh (A.β * A.μ) ^ 2 * sinh (A.β * u) ^ 2 / A.Dk k u ^ 2 := by
  have hτ := τk_sq k
  have h₁ := Complex.cosh_sq_sub_sinh_sq (A.β * (A.μ : ℂ))
  have h₂ := Complex.cosh_sq_sub_sinh_sq (A.β * u)
  rw [Rk, div_pow, div_sub_one (pow_ne_zero 2 hD), div_left_inj' (pow_ne_zero 2 hD), Dk]
  linear_combination (cosh (A.β * u) ^ 2 - 1) * h₁ + sinh (A.β * A.μ) ^ 2 * h₂ +
    (1 - cosh (A.β * A.μ) ^ 2) * hτ

/-- `h² sinh²(βμ) = β²ρ²/4`. -/
theorem hth_sq_mul_sinh_sq : (A.hth : ℂ) ^ 2 * sinh (A.β * A.μ) ^ 2 = A.β ^ 2 * A.ρ ^ 2 / 4 := by
  rw [← mul_pow, A.hth_mul_sinh]
  ring

/-- `α_k(u)α_k(-u) = h² = β²p₀p₂` for `D_k(u) ≠ 0`. -/
theorem α_mul_α_neg_of_Dk {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.α k u * A.α k (-u) = A.β ^ 2 * A.p₀ * A.p₂ := by
  have hR := A.Rk_sq_sub k hD
  have hs := A.hth_sq_mul_sinh_sq
  have hq : (Real.sqrt (A.p₀ * A.p₂) : ℂ) ^ 2 = A.p₀ * A.p₂ := by
    rw [← ofReal_pow, Real.sq_sqrt (mul_pos A.p₀_pos A.p₂_pos).le]
    push_cast
    ring
  have hh : (A.hth : ℂ) ^ 2 = A.β ^ 2 * A.p₀ * A.p₂ := by
    rw [hth]
    push_cast
    rw [mul_pow, hq]
    ring
  rw [A.α_eq_Rk k hD, A.α_eq_Rk k (by rwa [A.Dk_neg]), A.Dk_neg, A.Rk_neg, mul_neg,
    Complex.sinh_neg]
  rw [eq_div_iff (pow_ne_zero 2 hD)] at hR
  field_simp
  linear_combination 4 * A.hth ^ 2 * hR + 4 * sinh (A.β * u) ^ 2 * hs + 4 * A.Dk k u ^ 2 * hh

/-! ### The thermal denominators -/

/-- `A(u) = βH(-u) - λα_k(u)`. -/
noncomputable def Aph (u : ℂ) : ℂ := A.β * A.Hph k ε (-u) - A.lam ε * A.α k u

/-- The even part `K_e(u) = (K(u) + K(-u))/2`. -/
noncomputable def Ke (u : ℂ) : ℂ := (A.Kph k ε u + A.Kph k ε (-u)) / 2

/-- The odd part `K_o(u) = (K(u) - K(-u))/2`. -/
noncomputable def Ko (u : ℂ) : ℂ := (A.Kph k ε u - A.Kph k ε (-u)) / 2

/-- The even part `A_e(u) = (A(u) + A(-u))/2`. -/
noncomputable def Ae (u : ℂ) : ℂ := (A.Aph k ε u + A.Aph k ε (-u)) / 2

/-- The odd part `A_o(u) = (A(u) - A(-u))/2`. -/
noncomputable def Ao (u : ℂ) : ℂ := (A.Aph k ε u - A.Aph k ε (-u)) / 2

/-- `K_e = β(s + q₁) - λhR_k`, `q₁ = q₀ + 4εa²p₁`. -/
theorem Ke_eq {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.Ke k ε u =
      A.β * (u ^ 2 + (A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁)) - A.lam ε * A.hth * A.Rk k u := by
  have := A.αe_eq k hD
  rw [Ke, Kph, Kph, Hph, Hph, neg_neg]
  linear_combination A.lam ε * this

/-- `A_e = β(s + q₁) + λhR_k`. -/
theorem Ae_eq {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.Ae k ε u =
      A.β * (u ^ 2 + (A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁)) + A.lam ε * A.hth * A.Rk k u := by
  have := A.αe_eq k hD
  rw [Ae, Aph, Aph, Hph, Hph, neg_neg]
  linear_combination -A.lam ε * this

/-- `K_o = -2aβρuΦ_k/D_k`. -/
theorem Ko_eq {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.Ko k ε u = -2 * A.a * A.β * A.ρ * u * A.Phik k ε u / A.Dk k u := by
  have hα : A.α k u = A.α k (-u) - A.ρ * A.β * sinh (A.β * u) / A.Dk k u := by
    linear_combination 2 * A.αo_eq k hD
  rw [Ko, Kph, Kph, Hph, Hph, neg_neg, Phik, lam, hα, ← mul_shc A.β u]
  field_simp
  ring
/-- `A_o = -2aβρu(2 - Φ_k/D_k)`. -/
theorem Ao_eq {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.Ao k ε u = -2 * A.a * A.β * A.ρ * u * (2 - A.Phik k ε u / A.Dk k u) := by
  have hα : A.α k u = A.α k (-u) - A.ρ * A.β * sinh (A.β * u) / A.Dk k u := by
    linear_combination 2 * A.αo_eq k hD
  rw [Ao, Aph, Aph, Hph, Hph, neg_neg, Phik, lam, hα, ← mul_shc A.β u]
  field_simp
  ring

/-- **`K_eA_e - K_oA_o = β²e(u)`**, for `D_k(u) ≠ 0`: the left side is the mean of `K(u)A(-u)` and
`K(-u)A(u)`, which is `β²H(u)H(-u) - λ²α_k(u)α_k(-u)`. -/
theorem Ke_mul_Ae_sub {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.Ke k ε u * A.Ae k ε u - A.Ko k ε u * A.Ao k ε u = A.β ^ 2 * A.eph k ε u := by
  have hα := A.α_mul_α_neg_of_Dk k hD
  rw [Ke, Ko, Ae, Ao, Kph, Kph, Aph, Aph, eph, neg_neg]
  linear_combination -(A.lam ε ^ 2) * hα

/-- Where `Φ_k(u) = 0`, `R_k² = 1 + s sinh²(βμ)/a²` (`Φ_k = 0` forces `U ≠ 0`). -/
theorem Rk_sq_of_Phi (hε : ε = 1 ∨ ε = -1) {u : ℂ} (hD : A.Dk k u ≠ 0)
    (hΦ : A.Phik k ε u = 0) :
    A.Rk k u ^ 2 = 1 + u ^ 2 * sinh (A.β * A.μ) ^ 2 / A.a ^ 2 := by
  have ha : (A.a : ℂ) ≠ 0 := fun h0 ↦ hD (by rw [Phik, h0] at hΦ; simpa using hΦ)
  have hR := A.Rk_sq_sub k hD
  have hDs : A.Dk k u = -(ε * A.a * shc A.β u) := by
    rw [Phik] at hΦ
    linear_combination hΦ
  have hε2 : ε ^ 2 = 1 := by rcases hε with rfl | rfl <;> norm_num
  have hsh : shc A.β u ≠ 0 := fun h0 ↦ hD (by rw [hDs, h0]; simp)
  rw [hDs, ← mul_shc A.β u] at hR
  field_simp at hR
  rw [hε2, div_one] at hR
  field_simp
  linear_combination hR

/-- Where `Φ_k(u) = 0`, `K_o = 0` and `K_eA_e = β²e(u)`. -/
theorem Ke_mul_Ae_of_Phi {u : ℂ} (hD : A.Dk k u ≠ 0) (hΦ : A.Phik k ε u = 0) :
    A.Ke k ε u * A.Ae k ε u = A.β ^ 2 * A.eph k ε u := by
  have h := A.Ke_mul_Ae_sub k ε hD
  rw [A.Ko_eq k ε hD, hΦ] at h
  simpa using h

/-- At a root `u` of `e`, `K_eA_e = K_oA_o`. -/
theorem Ke_mul_Ae_eq {u : ℂ} (hD : A.Dk k u ≠ 0) (he : A.eph k ε u = 0) :
    A.Ke k ε u * A.Ae k ε u = A.Ko k ε u * A.Ao k ε u := by
  have h := A.Ke_mul_Ae_sub k ε hD
  rw [he, mul_zero] at h
  linear_combination h

/-- At a root `u` of `e`, `Z(u) = uP(u)A_e(u)/K_o(u)`, for `K(u) ≠ 0`,
`K_o(u) ≠ 0`. -/
theorem Zph_eq_Ae {u : ℂ} (hD : A.Dk k u ≠ 0) (he : A.eph k ε u = 0) (hK : A.Kph k ε u ≠ 0)
    (hKo : A.Ko k ε u ≠ 0) :
    A.Zph k ε u = u * A.Pph k u * A.Ae k ε u / A.Ko k ε u := by
  have h := A.Ke_mul_Ae_eq k ε hD he
  have hKs : A.Kph k ε u = A.Ke k ε u + A.Ko k ε u := by rw [Ke, Ko]; ring
  have hAs : A.β * A.Hph k ε (-u) - A.lam ε * A.α k u = A.Ae k ε u + A.Ao k ε u := by
    rw [Ae, Ao, Aph, Aph]; ring
  rw [Zph, hAs, div_eq_div_iff hK hKo, hKs]
  linear_combination -(u * A.Pph k u) * h

theorem Ae_neg (u : ℂ) : A.Ae k ε (-u) = A.Ae k ε u := by rw [Ae, Ae, neg_neg]; ring

theorem Ko_neg (u : ℂ) : A.Ko k ε (-u) = -A.Ko k ε u := by rw [Ko, Ko, neg_neg]; ring

/-- `Z` is even at a root, `Z(-u) = Z(u)` (both signs of `u` regular). -/
theorem Zph_neg {u : ℂ} (hD : A.Dk k u ≠ 0) (he : A.eph k ε u = 0) (hK : A.Kph k ε u ≠ 0)
    (hK' : A.Kph k ε (-u) ≠ 0) (hKo : A.Ko k ε u ≠ 0) :
    A.Zph k ε (-u) = A.Zph k ε u := by
  have hKo' : A.Ko k ε (-u) ≠ 0 := by rw [A.Ko_neg]; exact neg_ne_zero.mpr hKo
  rw [A.Zph_eq_Ae k ε (by rwa [A.Dk_neg]) (by rwa [A.eph_neg]) hK' hKo',
    A.Zph_eq_Ae k ε hD he hK hKo, A.Ae_neg, A.Ko_neg, A.Pph_neg]
  field_simp

/-- `lam_rho_Zph_eq` for `aρu ≠ 0`. -/
private theorem lam_rho_Zph_eq_ne {u : ℂ} (ha : (A.a : ℂ) ≠ 0) (hρ : (A.ρ : ℂ) ≠ 0) (hu : u ≠ 0)
    (hD : A.Dk k u ≠ 0) (he : A.eph k ε u = 0) (hK : A.Kph k ε u ≠ 0)
    (hΦ : A.Phik k ε u ≠ 0) :
    A.lam ε * A.ρ * A.Zph k ε u =
      -(ε * A.U * A.Pph k u * A.Dk k u * A.Ae k ε u / (A.β * A.Phik k ε u)) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hKo : A.Ko k ε u ≠ 0 := by
    rw [A.Ko_eq k ε hD]
    exact div_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero
      (by norm_num) ha) hβ) hρ) hu) hΦ) hD
  have hU : (A.U : ℂ) = 2 * A.a := by rw [a]; push_cast; ring
  rw [A.Zph_eq_Ae k ε hD he hK hKo, A.Ko_eq k ε hD, lam, hU]
  field_simp
  ring

/-- **The weight of the separable term**, `λρZ(u) = -εUP(u)D_k(u)A_e(u)/(βΦ_k(u))` for
`Φ_k(u) ≠ 0` and `K(u) ≠ 0`. If `aρu = 0`, then `K_o = 0`, so `A_e = 0` and both sides vanish. -/
theorem lam_rho_Zph_eq {u : ℂ} (hD : A.Dk k u ≠ 0) (he : A.eph k ε u = 0) (hK : A.Kph k ε u ≠ 0)
    (hΦ : A.Phik k ε u ≠ 0) :
    A.lam ε * A.ρ * A.Zph k ε u =
      -(ε * A.U * A.Pph k u * A.Dk k u * A.Ae k ε u / (A.β * A.Phik k ε u)) := by
  by_cases h0 : (A.a : ℂ) * A.ρ * u = 0
  · have hKo : A.Ko k ε u = 0 := by
      rw [A.Ko_eq k ε hD, show -2 * A.a * A.β * A.ρ * u * A.Phik k ε u =
        (A.a * A.ρ * u) * (-2 * A.β * A.Phik k ε u) by ring, h0, zero_mul, zero_div]
    have hKe : A.Ke k ε u ≠ 0 := by
      intro hKe
      apply hK
      rw [show A.Kph k ε u = A.Ke k ε u + A.Ko k ε u by rw [Ke, Ko]; ring, hKe, hKo, add_zero]
    have h := A.Ke_mul_Ae_eq k ε hD he
    rw [hKo, zero_mul] at h
    rw [(mul_eq_zero.mp h).resolve_left hKe, show A.lam ε * A.ρ * A.Zph k ε u = (A.a * A.ρ * u) *
      (4 * ε * A.a * A.Pph k u * (A.β * A.Hph k ε (-u) - A.lam ε * A.α k u) / A.Kph k ε u) by
        rw [Zph, lam]; ring, h0]
    simp
  have ha : (A.a : ℂ) ≠ 0 := fun h ↦ h0 (by rw [h]; ring)
  have hρ : (A.ρ : ℂ) ≠ 0 := fun h ↦ h0 (by rw [h]; ring)
  have hu : u ≠ 0 := fun h ↦ h0 (by rw [h]; ring)
  exact A.lam_rho_Zph_eq_ne k ε ha hρ hu hD he hK hΦ

/-- **The divergence condition**: at a root `u` of `e` with `Φ_k(u) = 0` and `A_e(u) ≠ 0`,
`K_e(u) = 0`: the divergence condition selects the branch `K_e = 0`. -/
theorem Ke_eq_zero_of_Phi {u : ℂ} (hD : A.Dk k u ≠ 0) (he : A.eph k ε u = 0)
    (hΦ : A.Phik k ε u = 0) (hAe : A.Ae k ε u ≠ 0) : A.Ke k ε u = 0 := by
  have h := A.Ke_mul_Ae_of_Phi k ε hD hΦ
  rw [he, mul_zero] at h
  exact (mul_eq_zero.mp h).resolve_right hAe

end Params

end HubbardAtom
