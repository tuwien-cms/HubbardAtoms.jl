import HubbardAtomLean.FieldEqualSpin

/-!
# The longitudinal input from the Hamiltonian

Proposition "longitudinal input" of the companion derivation (the input of Eq. (field-L) of the
note), derived from the Hamiltonian with doping and field, which the companion obtains by exact
computer algebra. The spin `σ` row lives on the lattice `z = μ_σ + i(νₙ + ω_k/2)` of its own spin,
`μ_σ = μ + σh`.

* `FParams.PL_mul_chiL_updn`: off the axes `μ = 0`, `h = 0`,
  `P(z)χ↑↓(z, v) = λ[ρ/(z + v) - m/(z - v)] + (N₀↓(v) + z²N₂↓(v))/P(v)`, `λ = U²`, with the
  pair kernel (gap `2μ`) and the spin-flip kernel (gap `2h`).
* `FParams.PL_mul_chiL_upup`: `P(z)χ↑↑(z, v) = -βH↑(z)δ_zv + δ_k0 ζ↑(z² - a²)(v² - a²)/P(v)`, for
  all `μ`, `h`.
* `FParams.chiL_flip`: the spin flip maps `χ_σt(U, μ, h)` to `χ_{-σ,-t}(U, μ, -h)`.
-/

open Complex Matrix

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-- The lattice of spin `↓`: `x_n = μ - h + iν_n`. -/
noncomputable abbrev Bdn : Params := F.lat (F.μ - F.h)

/-- `w = iω_k`. -/
noncomputable def wL : ℂ := F.ω k * I

/-- `q₀ = a² - w²/4`. -/
noncomputable def q₀L : ℂ := F.a ^ 2 - F.wL k ^ 2 / 4

/-- `P(z) = ((z + a)² - w²/4)((z - a)² - w²/4)`. -/
noncomputable def PL (z : ℂ) : ℂ :=
  ((z + F.a) ^ 2 - F.wL k ^ 2 / 4) * ((z - F.a) ^ 2 - F.wL k ^ 2 / 4)

/-- `H_r(z) = z² + 2arz + q₀`, with `r = r_σ` for spin `σ`. -/
noncomputable def HL (r : ℝ) (z : ℂ) : ℂ := z ^ 2 + 2 * F.a * r * z + F.q₀L k

/-- `λ = U² = 4a²`. -/
noncomputable def lamL : ℂ := 4 * F.a ^ 2

/-- `ζ = 4βa²(p₀p₂ - p↑p↓)`. -/
noncomputable def ζL : ℂ := 4 * F.β * F.a ^ 2 * (F.p₀ * F.p₂ - F.pUp * F.pDn)

/-- `ζ↑ = 4βa²(p₀ + p↑)(p↓ + p₂)`. -/
noncomputable def ζUp : ℂ := 4 * F.β * F.a ^ 2 * (F.p₀ + F.pUp) * (F.pDn + F.p₂)

/-- `N₂(v) = -2aH_r(v) + δ_k0 ζ(v² - a²)`. -/
noncomputable def N₂L (r : ℝ) (v : ℂ) : ℂ :=
  -2 * F.a * F.HL k r v + kron k 0 * F.ζL * (v ^ 2 - F.a ^ 2)

/-- `N₀(v) = (v² + 2q₀ - 4a²)N₂(v) + (2a - δ_k0 ζ)P(v)`. -/
noncomputable def N₀L (r : ℝ) (v : ℂ) : ℂ :=
  (v ^ 2 + 2 * F.q₀L k - 4 * F.a ^ 2) * F.N₂L k r v + (2 * F.a - kron k 0 * F.ζL) * F.PL k v

/-- `P(z)` on the lattice of spin `↑` through its linear factors. -/
theorem PL_up (j : ℤ) : F.PL k (F.Bup.z k j) =
    (F.Bup.x j + F.a) * (F.Bup.x j + F.wL k + F.a) *
      ((F.Bup.x j - F.a) * (F.Bup.x j + F.wL k - F.a)) := by
  rw [PL, Params.z, show F.Bup.w k = F.wL k from rfl]
  ring

/-- `P(z)` on the lattice of spin `↓` through its linear factors. -/
theorem PL_dn (j : ℤ) : F.PL k (F.Bdn.z k j) =
    (F.Bdn.x j + F.a) * (F.Bdn.x j + F.wL k + F.a) *
      ((F.Bdn.x j - F.a) * (F.Bdn.x j + F.wL k - F.a)) := by
  rw [PL, Params.z, show F.Bdn.w k = F.wL k from rfl]
  ring

set_option maxHeartbeats 8000000 in
-- six paths and the disconnected part: rational expressions of high degree in eight variables
set_option maxRecDepth 4000 in
/-- **The longitudinal input, opposite spins**, from the Hamiltonian, off the axes:
`P(z)χ↑↓(z, v) = λ[ρ/(z + v) - m/(z - v)] + (N₀↓(v) + z²N₂↓(v))/P(v)`, `z = z↑ₙ`, `v = z↓ₘ`. -/
theorem PL_mul_chiL_updn (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (n m : ℤ) :
    F.PL k (F.Bup.z k n) * F.chiL true false k n m =
      F.lamL * (F.ρ / (F.Bup.z k n + F.Bdn.z k m) - F.mf / (F.Bup.z k n - F.Bdn.z k m)) +
        (F.N₀L k F.rDn (F.Bdn.z k m) + F.Bup.z k n ^ 2 * F.N₂L k F.rDn (F.Bdn.z k m)) /
          F.PL k (F.Bdn.z k m) := by
  set BU := F.Bup with hBU
  set BD := F.Bdn with hBD
  have h1 : cexp ((((F.μ + F.h : ℝ) : ℂ) - BU.x n) * F.β) = -1 := BU.cexp_mu_sub_x n
  have h2 : cexp ((BU.x n + BU.w k - ((F.μ + F.h : ℝ) : ℂ)) * F.β) = -1 := by
    rw [BU.x_add_w]
    exact BU.cexp_x_sub_mu (n + k)
  have h3 : cexp ((((F.μ - F.h : ℝ) : ℂ) - BD.x m - BU.w k) * F.β) = -1 := by
    rw [sub_sub, show BU.w k = BD.w k from rfl, BD.x_add_w]
    exact BD.cexp_mu_sub_x (m + k)
  have hφ : ∀ τ₁ τ₂ τ₃ : ℝ,
      (((-F.ν n * τ₁ + (F.ν n + F.ω k) * τ₂ - (F.ν m + F.ω k) * τ₃ : ℝ) : ℂ) * I) =
        (((F.μ + F.h : ℝ) : ℂ) - BU.x n) * τ₁ + (BU.x n + BU.w k - ((F.μ + F.h : ℝ) : ℂ)) * τ₂ +
          (((F.μ - F.h : ℝ) : ℂ) - BD.x m - BU.w k) * τ₃ := by
    intro τ₁ τ₂ τ₃
    simp only [hBU, hBD, Params.x, Params.w, lat_μ, lat_ν, lat_ω]
    push_cast
    ring
  rw [chiL, cs, cs, ite_eq_left rfl, ite_eq_right Bool.false_ne_true,
    F.chiLehmann_eq _ _ _ _ _ hφ]
  simp only [orderedTerm_eq, triangleTerm_eq, lineTerm_eq]
  simp [Fin.sum_univ_four, cUp_eq, cDn_eq]
  have hk := BU.w_eq_zero_iff k
  have hnm : BU.x n - BD.x m = 0 ↔ False := by
    rw [iff_false]
    intro h0
    have := congrArg Complex.re h0
    simp [hBU, hBD, Params.x] at this
    exact hh (by linarith)
  have hp : -(BU.x n + BD.x m + BU.w k) = 0 ↔ False := by
    rw [iff_false]
    intro h0
    have := congrArg Complex.re h0
    simp [hBU, hBD, Params.x, Params.w] at this
    exact hμ (by linarith)
  push_cast at h1 h2 h3
  rw [F.pathTerm_eq 3 2 3 1 h1 h2 h3 (r₁ := -(BU.x n - F.a)) (r₂ := BU.x n + BU.w k - F.a)
      (r₃ := -(BD.x m + BU.w k - F.a)) (r₁₂ := BU.w k) (r₂₃ := BU.x n - BD.x m)
      (r₁₂₃ := -(BD.x m - F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hnm,
    F.pathTerm_eq 3 2 0 1 h1 h3 h2 (r₁ := -(BU.x n - F.a)) (r₂ := -(BD.x m + BU.w k + F.a))
      (r₃ := BU.x n + BU.w k + F.a) (r₁₂ := -(BU.x n + BD.x m + BU.w k)) (r₂₃ := BU.x n - BD.x m)
      (r₁₂₃ := -(BD.x m - F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hp hnm,
    F.pathTerm_eq 3 1 0 1 h3 h1 h2 (r₁ := -(BD.x m + BU.w k - F.a)) (r₂ := -(BU.x n + F.a))
      (r₃ := BU.x n + BU.w k + F.a) (r₁₂ := -(BU.x n + BD.x m + BU.w k)) (r₂₃ := BU.w k)
      (r₁₂₃ := -(BD.x m - F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hp hk,
    F.pathTerm_eq 2 3 2 0 h2 h1 h3 (r₁ := BU.x n + BU.w k - F.a) (r₂ := -(BU.x n - F.a))
      (r₃ := -(BD.x m + BU.w k + F.a)) (r₁₂ := BU.w k) (r₂₃ := -(BU.x n + BD.x m + BU.w k))
      (r₁₂₃ := -(BD.x m + F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hp,
    F.pathTerm_eq 2 3 1 0 h2 h3 h1 (r₁ := BU.x n + BU.w k - F.a) (r₂ := -(BD.x m + BU.w k - F.a))
      (r₃ := -(BU.x n + F.a)) (r₁₂ := BU.x n - BD.x m) (r₂₃ := -(BU.x n + BD.x m + BU.w k))
      (r₁₂₃ := -(BD.x m + F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hp,
    F.pathTerm_eq 2 0 1 0 h3 h2 h1 (r₁ := -(BD.x m + BU.w k + F.a)) (r₂ := BU.x n + BU.w k + F.a)
      (r₃ := -(BU.x n + F.a)) (r₁₂ := BU.x n - BD.x m) (r₂₃ := BU.w k)
      (r₁₂₃ := -(BD.x m + F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hk,
    F.pathTerm₂_eq 1 0 h1 h2 (r₁ := -(BU.x n + F.a)) (r₂ := BU.x n + BU.w k + F.a) (r₁₂ := BU.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    F.pathTerm₂_eq 3 2 h1 h2 (r₁ := -(BU.x n - F.a)) (r₂ := BU.x n + BU.w k - F.a) (r₁₂ := BU.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    F.pathTerm₂_eq 0 1 h2 h1 (r₁ := BU.x n + BU.w k + F.a) (r₂ := -(BU.x n + F.a)) (r₁₂ := BU.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    F.pathTerm₂_eq 2 3 h2 h1 (r₁ := BU.x n + BU.w k - F.a) (r₂ := -(BU.x n - F.a)) (r₁₂ := BU.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    F.pathTerm₁_eq 2 0 h3 (r := -(BD.x m + BU.w k + F.a)) (by simp [rate, energy]; ring),
    F.pathTerm₁_eq 3 1 h3 (r := -(BD.x m + BU.w k - F.a)) (by simp [rate, energy]; ring)]

  have hw := F.weights_sum'
  have X₁ := BU.x_add_ofReal_ne_zero n F.a
  have X₂ := BU.x_sub_ofReal_ne_zero n F.a
  have Y₁ := BD.x_add_ofReal_ne_zero m F.a
  have Y₂ := BD.x_sub_ofReal_ne_zero m F.a
  have XW₁ : BU.x n + BU.w k + F.a ≠ 0 := by
    rw [BU.x_add_w]; exact BU.x_add_ofReal_ne_zero _ _
  have XW₂ : BU.x n + BU.w k - F.a ≠ 0 := by
    rw [BU.x_add_w]; exact BU.x_sub_ofReal_ne_zero _ _
  have YW₁ : BD.x m + BU.w k + F.a ≠ 0 := by
    rw [show BU.w k = BD.w k from rfl, BD.x_add_w]; exact BD.x_add_ofReal_ne_zero _ _
  have YW₂ : BD.x m + BU.w k - F.a ≠ 0 := by
    rw [show BU.w k = BD.w k from rfl, BD.x_add_w]; exact BD.x_sub_ofReal_ne_zero _ _
  have hXY : BU.x n - BD.x m ≠ 0 := fun h0 ↦ hnm.mp h0
  have hS : BU.x n + BD.x m + BU.w k ≠ 0 := fun h0 ↦ hp.mp (by rw [h0, neg_zero])
  have hzU : BU.z k n = BU.x n + BU.w k / 2 := rfl
  have hzD : BD.z k m = BD.x m + BU.w k / 2 := rfl
  rw [show BU.z k n + BD.z k m = BU.x n + BD.x m + BU.w k by rw [hzU, hzD]; ring,
    show BU.z k n - BD.z k m = BU.x n - BD.x m by rw [hzU, hzD]; ring]
  simp only [N₀L]
  rw [hBU, hBD, F.PL_up, F.PL_dn, ← hBU, ← hBD]
  simp only [simplexW, triangleW, weight, Matrix.cons_val, Matrix.cons_val_zero,
    Matrix.cons_val_one, N₂L, HL, q₀L, lamL, ζL, rDn, ρ, mf, hzU, hzD,
    show F.wL k = BU.w k from rfl]
  push_cast
  by_cases hk0 : k = 0
  · subst hk0
    have hw0 : BU.w 0 = 0 := hk.mpr rfl
    simp only [ite_true, kron_self, hw0, add_zero, sub_zero, zero_div, ne_eq,
      OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow] at XW₁ XW₂ YW₁ YW₂ hS ⊢
    generalize BU.x n = X at *
    generalize BD.x m = Y at *
    generalize (F.a : ℂ) = a at *
    generalize (F.β : ℂ) = β at *
    generalize (F.p₂ : ℂ) = p₂ at *
    obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hw
    field_simp
    ring
  · have hW : BU.w k ≠ 0 := mt hk.mp hk0
    simp only [hk0, ite_false, kron_of_ne hk0, zero_mul, sub_zero, add_zero]
    generalize BU.x n = X at *
    generalize BD.x m = Y at *
    generalize BU.w k = W at *
    generalize (F.a : ℂ) = a at *
    generalize (F.β : ℂ) = β at *
    generalize (F.p₂ : ℂ) = p₂ at *
    obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hw
    field_simp
    ring

set_option maxHeartbeats 8000000 in
-- four paths and the disconnected part: rational expressions of high degree in eight variables
set_option maxRecDepth 4000 in
/-- **The longitudinal input, equal spins**, from the Hamiltonian, for all `μ`, `h`:
`P(z)χ↑↑(z, v) = -βH↑(z)δ_zv + δ_k0 ζ↑(z² - a²)(v² - a²)/P(v)`. -/
theorem PL_mul_chiL_upup (n m : ℤ) :
    F.PL k (F.Bup.z k n) * F.chiL true true k n m =
      -F.β * F.HL k F.rUp (F.Bup.z k n) * kron n m +
        kron k 0 * F.ζUp * ((F.Bup.z k n ^ 2 - F.a ^ 2) * (F.Bup.z k m ^ 2 - F.a ^ 2)) /
          F.PL k (F.Bup.z k m) := by
  set BU := F.Bup with hBU
  have h1 : cexp ((((F.μ + F.h : ℝ) : ℂ) - BU.x n) * F.β) = -1 := BU.cexp_mu_sub_x n
  have h2 : cexp ((BU.x n + BU.w k - ((F.μ + F.h : ℝ) : ℂ)) * F.β) = -1 := by
    rw [BU.x_add_w]
    exact BU.cexp_x_sub_mu (n + k)
  have h3 : cexp ((((F.μ + F.h : ℝ) : ℂ) - BU.x m - BU.w k) * F.β) = -1 := by
    rw [sub_sub, BU.x_add_w]
    exact BU.cexp_mu_sub_x (m + k)
  have hφ : ∀ τ₁ τ₂ τ₃ : ℝ,
      (((-F.ν n * τ₁ + (F.ν n + F.ω k) * τ₂ - (F.ν m + F.ω k) * τ₃ : ℝ) : ℂ) * I) =
        (((F.μ + F.h : ℝ) : ℂ) - BU.x n) * τ₁ + (BU.x n + BU.w k - ((F.μ + F.h : ℝ) : ℂ)) * τ₂ +
          (((F.μ + F.h : ℝ) : ℂ) - BU.x m - BU.w k) * τ₃ :=
    BU.freq_eq k n m
  rw [chiL, cs, ite_eq_left rfl, F.chiLehmann_eq _ _ _ _ _ hφ]
  simp only [orderedTerm_eq, triangleTerm_eq, lineTerm_eq]
  simp [Fin.sum_univ_four, cUp_eq]
  have hk := BU.w_eq_zero_iff k
  have hnm := BU.x_sub_x_eq_zero_iff n m
  push_cast at h1 h2 h3
  rw [F.pathTerm_eq 1 0 1 0 h1 h2 h3 (r₁ := -(BU.x n + F.a)) (r₂ := BU.x n + BU.w k + F.a)
      (r₃ := -(BU.x m + BU.w k + F.a)) (r₁₂ := BU.w k) (r₂₃ := BU.x n - BU.x m)
      (r₁₂₃ := -(BU.x m + F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hnm,
    F.pathTerm_eq 3 2 3 2 h1 h2 h3 (r₁ := -(BU.x n - F.a)) (r₂ := BU.x n + BU.w k - F.a)
      (r₃ := -(BU.x m + BU.w k - F.a)) (r₁₂ := BU.w k) (r₂₃ := BU.x n - BU.x m)
      (r₁₂₃ := -(BU.x m - F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hnm,
    F.pathTerm_eq 1 0 1 0 h3 h2 h1 (r₁ := -(BU.x m + BU.w k + F.a)) (r₂ := BU.x n + BU.w k + F.a)
      (r₃ := -(BU.x n + F.a)) (r₁₂ := BU.x n - BU.x m) (r₂₃ := BU.w k)
      (r₁₂₃ := -(BU.x m + F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hk,
    F.pathTerm_eq 3 2 3 2 h3 h2 h1 (r₁ := -(BU.x m + BU.w k - F.a)) (r₂ := BU.x n + BU.w k - F.a)
      (r₃ := -(BU.x n - F.a)) (r₁₂ := BU.x n - BU.x m) (r₂₃ := BU.w k)
      (r₁₂₃ := -(BU.x m - F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hk,
    F.pathTerm₂_eq 1 0 h1 h2 (r₁ := -(BU.x n + F.a)) (r₂ := BU.x n + BU.w k + F.a) (r₁₂ := BU.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    F.pathTerm₂_eq 3 2 h1 h2 (r₁ := -(BU.x n - F.a)) (r₂ := BU.x n + BU.w k - F.a) (r₁₂ := BU.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    F.pathTerm₂_eq 0 1 h2 h1 (r₁ := BU.x n + BU.w k + F.a) (r₂ := -(BU.x n + F.a)) (r₁₂ := BU.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    F.pathTerm₂_eq 2 3 h2 h1 (r₁ := BU.x n + BU.w k - F.a) (r₂ := -(BU.x n - F.a)) (r₁₂ := BU.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    F.pathTerm₁_eq 1 0 h3 (r := -(BU.x m + BU.w k + F.a)) (by simp [rate, energy]; ring),
    F.pathTerm₁_eq 3 2 h3 (r := -(BU.x m + BU.w k - F.a)) (by simp [rate, energy]; ring)]
  have hw := F.weights_sum'
  have X₁ := BU.x_add_ofReal_ne_zero n F.a
  have X₂ := BU.x_sub_ofReal_ne_zero n F.a
  have Y₁ := BU.x_add_ofReal_ne_zero m F.a
  have Y₂ := BU.x_sub_ofReal_ne_zero m F.a
  have XW₁ : BU.x n + BU.w k + F.a ≠ 0 := by
    rw [BU.x_add_w]; exact BU.x_add_ofReal_ne_zero _ _
  have XW₂ : BU.x n + BU.w k - F.a ≠ 0 := by
    rw [BU.x_add_w]; exact BU.x_sub_ofReal_ne_zero _ _
  have YW₁ : BU.x m + BU.w k + F.a ≠ 0 := by
    rw [BU.x_add_w]; exact BU.x_add_ofReal_ne_zero _ _
  have YW₂ : BU.x m + BU.w k - F.a ≠ 0 := by
    rw [BU.x_add_w]; exact BU.x_sub_ofReal_ne_zero _ _
  have hzU : ∀ j, BU.z k j = BU.x j + BU.w k / 2 := fun _ ↦ rfl
  rw [hBU, F.PL_up, F.PL_up, ← hBU]
  simp only [simplexW, triangleW, weight, Matrix.cons_val, Matrix.cons_val_zero,
    Matrix.cons_val_one, HL, q₀L, ζUp, rUp, ρ, mf, hzU, show F.wL k = BU.w k from rfl, kron]
  push_cast
  by_cases hk0 : k = 0
  · subst hk0
    have hw0 : BU.w 0 = 0 := hk.mpr rfl
    simp only [ite_true, hw0, add_zero, sub_zero, zero_div, ne_eq, OfNat.ofNat_ne_zero,
      not_false_eq_true, zero_pow] at XW₁ XW₂ YW₁ YW₂ ⊢
    by_cases hnm0 : n = m
    · subst hnm0
      simp only [ite_true]
      generalize BU.x n = X at *
      generalize (F.a : ℂ) = a at *
      generalize (F.β : ℂ) = β at *
      generalize (F.p₂ : ℂ) = p₂ at *
      obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hw
      field_simp
      ring
    · have hXY : BU.x n - BU.x m ≠ 0 := mt hnm.mp hnm0
      simp only [hnm0, ite_false]
      generalize BU.x n = X at *
      generalize BU.x m = Y at *
      generalize (F.a : ℂ) = a at *
      generalize (F.β : ℂ) = β at *
      generalize (F.p₂ : ℂ) = p₂ at *
      obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hw
      field_simp
      ring
  · have hW : BU.w k ≠ 0 := mt hk.mp hk0
    simp only [hk0, ite_false, zero_mul, zero_div, add_zero]
    by_cases hnm0 : n = m
    · subst hnm0
      simp only [ite_true]
      generalize BU.x n = X at *
      generalize BU.w k = W at *
      generalize (F.a : ℂ) = a at *
      generalize (F.β : ℂ) = β at *
      generalize (F.p₂ : ℂ) = p₂ at *
      obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hw
      field_simp
      ring
    · have hXY : BU.x n - BU.x m ≠ 0 := mt hnm.mp hnm0
      simp only [hnm0, ite_false]
      generalize BU.x n = X at *
      generalize BU.x m = Y at *
      generalize BU.w k = W at *
      generalize (F.a : ℂ) = a at *
      field_simp
      ring

/-- The spin flip: `χ_σt(U, μ, h) = χ_{-σ,-t}(U, μ, -h)`, so the rows of spin `↓` follow from
those of spin `↑` at `-h`. -/
theorem chiL_flip (σ t : Bool) (n m : ℤ) :
    F.chiL σ t k n m = F.flip.chiL (!σ) (!t) k n m := by
  have key : ∀ c c' : Op, (flipV * c * flipV)ᴴ = flipV * cᴴ * flipV →
      (flipV * c' * flipV)ᴴ = flipV * c'ᴴ * flipV →
      F.flip.chiLehmann (flipV * c * flipV) (flipV * c' * flipV) (F.ν n) (F.ν m) (F.ω k) =
        F.chiLehmann c c' (F.ν n) (F.ν m) (F.ω k) := fun c c' hc hc' ↦
    chiLehmann_conj (A := F) (B := F.flip) (V := flipV) (W := flipV) flipV_mul_self rfl
      F.flip_H hc hc' _ _ _
  have hu : (flipV * cUp * flipV)ᴴ = flipV * cUpᴴ * flipV := by rw [flipV_cUp, flipV_cUpDag]
  have hd : (flipV * cDn * flipV)ᴴ = flipV * cDnᴴ * flipV := by rw [flipV_cDn, flipV_cDnDag]
  have e : ∀ s : Bool, cs (!s) = flipV * cs s * flipV := by
    intro s
    cases s
    · rw [cs, cs]; simp [flipV_cDn]
    · rw [cs, cs]; simp [flipV_cUp]
  have e' : ∀ s : Bool, (flipV * cs s * flipV)ᴴ = flipV * (cs s)ᴴ * flipV := by
    intro s
    cases s
    · simpa [cs] using hd
    · simpa [cs] using hu
  rw [chiL, chiL, e, e, F.flip_ν, F.flip_ν, F.flip_ω, key _ _ (e' σ) (e' t)]

end FParams

end HubbardAtom
