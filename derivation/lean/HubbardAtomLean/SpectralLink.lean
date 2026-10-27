import HubbardAtomLean.SpectralIdentity
import HubbardAtomLean.DivergenceAlgebra

/-!
# The spectral identity in terms of `Φ_k`, `D_k` and `K_e`

The coefficients of the spectral identity (`SpectralIdentity.lean`) in the notation of
`DivergenceAlgebra.lean`:

* `Params.one_sub_Tg`: `1 - (εU/β)𝒯(s) = Φ_k(s)/D_k(s)` for `u ≠ 0`, `D_k(u) ≠ 0`, from
  `S_k(u) - S_k(-u) = -β sinh(βu)/D_k(u)` (`S_sub_S_neg`);
* `Params.ell_eq`: for `U ≠ 0` and `P(s) ≠ 0`, the eigenvalue
  `ℓ(s) = -K_e(s)/P(s) + βf_e(u)[1 - (εU/β)𝒯(s)]`, `f_e(u) = (f_r(u) + f_r(-u))/2`, by partial
  fractions; so `ℓ(s) = -K_e(s)/P(s)` where `Φ_k(s) = 0` (`ell_eq_of_Phi`);
* `Params.hasSum_chiR_mul_g_of_Phi`: every `g_s` with `Φ_k(s) = 0` and `P(s) ≠ 0` is an
  eigenvector of `χ_r` with eigenvalue `-K_e(s)/P(s)`.
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `S_k(u) - S_k(-u) = -β sinh(βu)/D_k(u)`, with `D_k = 2cosh(y₊)cosh(y₋)` for even `k` and
`2sinh(y₊)sinh(y₋)` for odd `k`, `y± = β(u ± μ)/2`. -/
theorem S_sub_S_neg {u : ℂ} (hD : A.Dk k u ≠ 0) :
    A.S k u - A.S k (-u) = -(A.β * sinh (A.β * u) / A.Dk k u) := by
  have h₁ : (A.β : ℂ) * u = A.β * (u + A.μ) / 2 + A.β * (u - A.μ) / 2 := by ring
  have h₂ : (A.β : ℂ) * A.μ = A.β * (u + A.μ) / 2 - A.β * (u - A.μ) / 2 := by ring
  have h₃ : (A.β : ℂ) * (-u - A.μ) / 2 = -(A.β * (u + A.μ) / 2) := by ring
  rw [Dk, τk] at hD ⊢
  rw [S, S, h₃, h₁, h₂]
  rw [h₁, h₂] at hD
  simp only [cosh_add, cosh_sub, sinh_add, coth, Complex.tanh, cosh_neg, sinh_neg]
    at hD ⊢
  by_cases hk : Even k
  · simp only [hk, ↓reduceIte] at hD ⊢
    generalize cosh (A.β * (u + A.μ) / 2) = cp at *
    generalize cosh (A.β * (u - A.μ) / 2) = cm at *
    generalize sinh (A.β * (u + A.μ) / 2) = sp at *
    generalize sinh (A.β * (u - A.μ) / 2) = sm at *
    rw [show cp * cm + sp * sm + 1 * (cp * cm - sp * sm) = 2 * cp * cm by ring] at hD ⊢
    obtain ⟨hcp, hcm⟩ := mul_ne_zero_iff.mp hD
    have hcp' := (mul_ne_zero_iff.mp hcp).2
    field_simp
    ring
  · simp only [hk, ↓reduceIte] at hD ⊢
    generalize cosh (A.β * (u + A.μ) / 2) = cp at *
    generalize cosh (A.β * (u - A.μ) / 2) = cm at *
    generalize sinh (A.β * (u + A.μ) / 2) = sp at *
    generalize sinh (A.β * (u - A.μ) / 2) = sm at *
    rw [show cp * cm + sp * sm + -1 * (cp * cm - sp * sm) = 2 * sp * sm by ring] at hD ⊢
    obtain ⟨hsp, hsm⟩ := mul_ne_zero_iff.mp hD
    have hsp' := (mul_ne_zero_iff.mp hsp).2
    field_simp
    ring

/-- `1 - (εU/β)𝒯(s) = Φ_k(s)/D_k(s)`, for `u ≠ 0` and `D_k(u) ≠ 0`. -/
theorem one_sub_Tg {u : ℂ} (hu : u ≠ 0) (hD : A.Dk k u ≠ 0) :
    1 - ε * A.U / A.β * A.Tg k u = A.Phik k ε u / A.Dk k u := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hU : (A.U : ℂ) = 2 * A.a := by rw [a]; push_cast; ring
  rw [Tg, ite_eq_right hu, A.S_sub_S_neg k hD, Phik, hU]
  field_simp
  ring

/-- `f_r = N/P` with the quadratic `N(z) = q₋Q₋(z) + q₊Q₊(z) - δ_k0 εζ_r(z² - a²)/(2a)`, where
`P(z) ≠ 0` (for `k = 0`, `P(z) = (z² - a²)²`). -/
theorem fR_eq_div {z : ℂ} (hP : A.Pph k z ≠ 0) :
    A.fR k ε z = (A.qm * A.Qm k z + A.qp * A.Qp k z -
      kron k 0 * ε * A.ζ ε * (z ^ 2 - A.a ^ 2) / (2 * A.a)) / A.Pph k z := by
  rw [Pph] at hP
  have hp := left_ne_zero_of_mul hP
  have hm := right_ne_zero_of_mul hP
  rw [fR, Pph]
  by_cases hk : k = 0
  · subst hk
    simp only [Qp, Qm, A.w_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div,
      sub_zero, kron_self, one_mul] at hp hm ⊢
    have hp' : z + A.a ≠ 0 := pow_ne_zero_iff two_ne_zero |>.mp hp
    have hm' : z - A.a ≠ 0 := pow_ne_zero_iff two_ne_zero |>.mp hm
    have hza : z ^ 2 - A.a ^ 2 ≠ 0 := by
      rw [show z ^ 2 - A.a ^ 2 = (z + A.a) * (z - A.a) by ring]
      exact mul_ne_zero hp' hm'
    rw [show (z + A.a) ^ 2 * (z - A.a) ^ 2 = (z ^ 2 - A.a ^ 2) ^ 2 by ring]
    field_simp
    ring
  · rw [kron_of_ne hk]
    field_simp
    ring

/-- `zₗg_s(l) - 1/(zₗ - y)` sums to `(S_k(u) + S_k(-u))/2 - S_k(y)`, for `±u`, `y` off the
lattice. -/
theorem hasSum_z_mul_gs {u y : ℂ} (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u)
    (hy : ∀ m, A.z k m ≠ y) :
    HasSum (fun l ↦ A.z k l * A.gs k u l - 1 / (A.z k l - y))
      ((A.S k u + A.S k (-u)) / 2 - A.S k y) := by
  convert ((A.hasSum_sub_inv k hu hy).add (A.hasSum_sub_inv k hu' hy)).div_const 2 using 1
  · funext l
    have h₁ := sub_ne_zero.mpr (hu l)
    have h₂ := sub_ne_zero.mpr (hu' l)
    have h₃ := sub_ne_zero.mpr (hy l)
    rw [gs, show A.z k l ^ 2 - u ^ 2 = (A.z k l - u) * (A.z k l - -u) by ring]
    field_simp
    ring
  · ring

/-- **The eigenvalue in closed form**: for `U ≠ 0`, `P(u) ≠ 0` and `±u` off the lattice,
`ℓ(s) = -εU ∑ₘ f_r(zₘ)g_s(m) = -K_e(s)/P(s) + βf_e(u)[1 - (εU/β)𝒯(s)]`, with
`f_e(u) = (f_r(u) + f_r(-u))/2`. By partial fractions, `f_r g_s = A₃/P + (c₀ + c₁z)g_s` with a
cubic `A₃`, `c₀ = f_e(u)` and `c₁ = 2aρ/P(u)`; `hasSum_cubic`, `hasSum_g` and `hasSum_z_mul_gs`
sum the three parts. -/
theorem ell_eq (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) {u : ℂ} (hP : A.Pph k u ≠ 0)
    (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) :
    -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m =
      -(A.Ke k ε u / A.Pph k u) +
        A.β * ((A.fR k ε u + A.fR k ε (-u)) / 2) * (1 - ε * A.U / A.β * A.Tg k u) := by
  have hP' : A.Pph k (-u) ≠ 0 := by rwa [Pph_neg]
  obtain ⟨c₀, hc₀⟩ : ∃ c, c = (A.fR k ε u + A.fR k ε (-u)) / 2 := ⟨_, rfl⟩
  obtain ⟨c₁, hc₁⟩ : ∃ c, c = 2 * A.a * A.ρ / A.Pph k u := ⟨_, rfl⟩
  obtain ⟨σ, hσ⟩ : ∃ c, c = u ^ 2 - 2 * A.a ^ 2 - A.w k ^ 2 / 2 := ⟨_, rfl⟩
  obtain ⟨n₂, hn₂⟩ : ∃ c, c = A.qm + A.qp - kron k 0 * ε * A.ζ ε / (2 * A.a) := ⟨_, rfl⟩
  rw [← hc₀]
  have h := ((A.hasSum_cubic k ha (n₂ - c₀ * σ) (-(c₁ * σ)) (-c₀) (-c₁) hu).add
    ((A.hasSum_g k hu hu').mul_left c₀)).add ((A.hasSum_z_mul_gs k hu hu' hu).mul_left c₁)
  have hfg : (fun m ↦ A.fR k ε (A.z k m) * A.gs k u m) = fun l ↦
      ((n₂ - c₀ * σ) + -(c₁ * σ) * A.z k l + -c₀ * A.z k l ^ 2 + -c₁ * A.z k l ^ 3) /
        A.Pph k (A.z k l) - -c₁ / (A.z k l - u) + c₀ * A.gs k u l +
        c₁ * (A.z k l * A.gs k u l - 1 / (A.z k l - u)) := by
    funext l
    have hPz := A.Pph_z_ne_zero k l
    have h₁ := sub_ne_zero.mpr (hu l)
    have h₂ : A.z k l + u ≠ 0 := fun h0 ↦ hu' l (by linear_combination h0)
    rw [hc₀, hc₁, hσ, hn₂, A.fR_eq_div k ε hPz, A.fR_eq_div k ε hP, A.fR_eq_div k ε hP', Pph_neg,
      gs, show A.z k l ^ 2 - u ^ 2 = (A.z k l - u) * (A.z k l + u) by ring]
    field_simp
    simp only [Pph, Qp, Qm, ρ, qm, qp]
    push_cast
    ring
  rw [hfg, h.tsum_eq]
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hq := A.q₀_ne_zero k ha
  have h₀₁ : (A.p₀ : ℂ) + A.p₁ ≠ 0 := by exact_mod_cast (add_pos A.p₀_pos A.p₁_pos).ne'
  have h₁₂ : (A.p₁ : ℂ) + A.p₂ ≠ 0 := by exact_mod_cast (add_pos A.p₁_pos A.p₂_pos).ne'
  have hU : (A.U : ℂ) = 2 * A.a := by rw [a]; push_cast; ring
  have hp₂ : (A.p₂ : ℂ) = 1 - A.p₀ - 2 * A.p₁ := by
    have := A.weights_sum
    rw [show A.p₂ = 1 - A.p₀ - 2 * A.p₁ by linarith]
    push_cast
    ring
  rw [hc₀, hc₁, hσ, hn₂, A.fR_eq_div k ε hP, A.fR_eq_div k ε hP', Pph_neg]
  simp only [Ke, Kph, neg_neg, Hph, α, lam, cbar, ℓ₀, ℓ₁, ℓ₂, ℓ₃, Tp, Tm, Sp, Sm, S₀_a, S₀_neg_a,
    hU]
  generalize A.S k u = S₁
  generalize A.S k (-u) = S₂
  generalize A.Tg k u = T
  by_cases hk : k = 0
  · subst hk
    simp only [A.w_zero, kron_self] at hq hP ⊢
    rcases hε with rfl | rfl
    · field_simp
      simp only [q₀, Pph, Qp, Qm, ζ, ρ, qm, qp, A.w_zero]
      push_cast
      rw [hp₂]
      ring
    · field_simp
      simp only [q₀, Pph, Qp, Qm, ζ, ρ, qm, qp, A.w_zero]
      push_cast
      rw [hp₂]
      ring
  · simp only [kron_of_ne hk, zero_mul, zero_div, sub_zero, mul_zero, add_zero]
    field_simp
    simp only [q₀, Pph, Qp, Qm, ρ, qm, qp]
    push_cast
    rw [hp₂]
    ring

/-- `D_k(u) ≠ 0` for `±u` off the lattice: its factors are those of `S_k(±u)`. -/
theorem Dk_ne_zero_of_off {u : ℂ} (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) :
    A.Dk k u ≠ 0 := by
  have h₁ : (A.β : ℂ) * u = A.β * (u - A.μ) / 2 + -(A.β * (-u - A.μ) / 2) := by ring
  have h₂ : (A.β : ℂ) * A.μ = -(A.β * (-u - A.μ) / 2) - A.β * (u - A.μ) / 2 := by ring
  rw [Dk, τk, h₁, h₂]
  simp only [cosh_add, cosh_sub, cosh_neg, sinh_neg]
  obtain ⟨j, hj | hj⟩ := Int.even_or_odd' k
  · have hc := A.cosh_ne_zero_of_off hj hu
    have hc' := A.cosh_ne_zero_of_off hj hu'
    rw [ite_eq_left ⟨j, by omega⟩]
    convert mul_ne_zero (mul_ne_zero two_ne_zero hc) hc' using 1
    ring
  · have hj' : k = 2 * (j + 1) - 1 := by omega
    have hs := A.sinh_ne_zero_of_off hj' hu
    have hs' := A.sinh_ne_zero_of_off hj' hu'
    rw [ite_eq_right (by rintro ⟨r, hr⟩; omega)]
    convert mul_ne_zero (mul_ne_zero two_ne_zero hs) (neg_ne_zero.mpr hs') using 1
    ring

/-- At a root of `Φ_k` with `±u` off the lattice, `1 - (εU/β)𝒯(s) = 0` (`u = 0` is no root,
since `D_k(0) ≠ 0`). -/
theorem one_sub_Tg_of_Phi {u : ℂ} (hΦ : A.Phik k ε u = 0) (hu : ∀ m, A.z k m ≠ u)
    (hu' : ∀ m, A.z k m ≠ -u) : 1 - ε * A.U / A.β * A.Tg k u = 0 := by
  have hD := A.Dk_ne_zero_of_off k hu hu'
  have hu0 : u ≠ 0 := by
    rintro rfl
    simp [Phik, hD] at hΦ
  rw [A.one_sub_Tg k ε hu0 hD, hΦ, zero_div]

/-- At a root of `Φ_k`, `ℓ(s) = -K_e(s)/P(s)`. -/
theorem ell_eq_of_Phi (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) {u : ℂ}
    (hΦ : A.Phik k ε u = 0) (hP : A.Pph k u ≠ 0) (hu : ∀ m, A.z k m ≠ u)
    (hu' : ∀ m, A.z k m ≠ -u) :
    -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m = -(A.Ke k ε u / A.Pph k u) := by
  rw [A.ell_eq k ε hε ha hP hu hu', A.one_sub_Tg_of_Phi k ε hΦ hu hu', mul_zero, add_zero]

/-- **Eigenvectors at the roots of `Φ_k`**: for `U ≠ 0`, `μ ≠ 0`, `P(u) ≠ 0`, `±u` off the
lattice and `Φ_k(u) = 0`, `χ_r g_s = -(K_e(s)/P(s)) g_s`. -/
theorem hasSum_chiR_mul_g_of_Phi (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {u : ℂ} (hΦ : A.Phik k ε u = 0) (hP : A.Pph k u ≠ 0) (hu : ∀ m, A.z k m ≠ u)
    (hu' : ∀ m, A.z k m ≠ -u) (n : ℤ) :
    HasSum (fun m ↦ A.chiR k ε n m * A.gs k u m) (-(A.Ke k ε u / A.Pph k u) * A.gs k u n) := by
  rw [← A.ell_eq_of_Phi k ε hε ha hΦ hP hu hu']
  exact A.hasSum_chiR_mul_g_of k ε hε ha hμ hu hu' (A.one_sub_Tg_of_Phi k ε hΦ hu hu') n

end Params

end HubbardAtom
