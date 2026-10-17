import Mathlib

/-!
# Setup: the atom, its thermal weights, and the Matsubara lattice

Section "Setup" of the note `derivation/doped-gamma-derivation.typ` of HubbardAtoms.jl.

* `Params`: the Hubbard atom `H = U (n↑ - 1/2)(n↓ - 1/2) - μ (n↑ + n↓ - 1)` at inverse
  temperature `β > 0`, with `μ` measured from half filling.
* `Params.p₀`, `Params.p₁`, `Params.p₂`: the Boltzmann weights of `|0⟩`, of each singly occupied
  state, and of `|↑↓⟩`; `weights_sum` and `p₁_eq_p₀_mul`, `p₁_eq_p₂_mul`, `p₂_eq_p₀_mul` are
  Eq. (weights), `one_sub_rho_sq` and `rho_sq_add` part of Eq. (weight-ids).
* `Params.x`, `Params.w`, `Params.b`, `Params.t`, `Params.z`, `mirror`: the frequencies
  `x_n = μ + iν_n`, `w = iω_k`, `b = 2μ - w`, `t_n = x_n (b - x_n)`, the centered lattice
  `z_n = x_n + w/2` and the mirror index `n̄ = -n - k - 1` (Section "Lattice and mirror"), with
  `x_mirror` and `t_mirror`.
-/

open Complex

namespace HubbardAtom

/-- The Kronecker delta on `ℤ`, with values in `ℂ`. -/
def kron (i j : ℤ) : ℂ := if i = j then 1 else 0

@[simp] theorem kron_self (i : ℤ) : kron i i = 1 := ite_eq_left rfl

theorem kron_of_ne {i j : ℤ} (h : i ≠ j) : kron i j = 0 := ite_eq_right h

theorem kron_comm (i j : ℤ) : kron i j = kron j i := by
  unfold kron
  by_cases h : i = j
  · rw [ite_eq_left h, ite_eq_left h.symm]
  · rw [ite_eq_right h, ite_eq_right (Ne.symm h)]

/-- The mirror index `n̄ = -n - k - 1` at bosonic transfer `k`. -/
def mirror (k n : ℤ) : ℤ := -n - k - 1

@[simp] theorem mirror_mirror (k n : ℤ) : mirror k (mirror k n) = n := by
  unfold mirror
  ring

theorem mirror_eq_iff {k n m : ℤ} : m = mirror k n ↔ n = mirror k m := by
  unfold mirror
  omega

/-- Parameters of the Hubbard atom: interaction `U`, inverse temperature `β > 0`, and chemical
potential `μ`, measured from half filling. -/
structure Params where
  /-- the Hubbard interaction -/
  U : ℝ
  /-- the inverse temperature -/
  β : ℝ
  /-- the chemical potential, measured from half filling -/
  μ : ℝ
  β_pos : 0 < β

namespace Params

variable (A : Params)

/-- `a = U/2`. -/
noncomputable def a : ℝ := A.U / 2

/-- The Boltzmann factor of `|0⟩`, whose energy is `U/4 + μ`. -/
noncomputable def boltz₀ : ℝ := Real.exp (-A.β * (A.U / 4 + A.μ))

/-- The Boltzmann factor of each singly occupied state, whose energy is `-U/4`. -/
noncomputable def boltz₁ : ℝ := Real.exp (-A.β * (-A.U / 4))

/-- The Boltzmann factor of `|↑↓⟩`, whose energy is `U/4 - μ`. -/
noncomputable def boltz₂ : ℝ := Real.exp (-A.β * (A.U / 4 - A.μ))

/-- The partition function. -/
noncomputable def Z : ℝ := A.boltz₀ + 2 * A.boltz₁ + A.boltz₂

/-- The weight of `|0⟩`. -/
noncomputable def p₀ : ℝ := A.boltz₀ / A.Z

/-- The weight of each singly occupied state. -/
noncomputable def p₁ : ℝ := A.boltz₁ / A.Z

/-- The weight of `|↑↓⟩`. -/
noncomputable def p₂ : ℝ := A.boltz₂ / A.Z

/-- `ρ = p₂ - p₀ = ⟨n↑ + n↓⟩ - 1`. -/
noncomputable def ρ : ℝ := A.p₂ - A.p₀

/-- `q₊ = p₁ + p₂ = ⟨n↑⟩`. -/
noncomputable def qp : ℝ := A.p₁ + A.p₂

/-- `q₋ = p₀ + p₁ = 1 - ⟨n↑⟩`. -/
noncomputable def qm : ℝ := A.p₀ + A.p₁

theorem Z_pos : 0 < A.Z := by
  unfold Z boltz₀ boltz₁ boltz₂
  positivity

theorem p₀_pos : 0 < A.p₀ := div_pos (Real.exp_pos _) A.Z_pos

theorem p₁_pos : 0 < A.p₁ := div_pos (Real.exp_pos _) A.Z_pos

theorem p₂_pos : 0 < A.p₂ := div_pos (Real.exp_pos _) A.Z_pos

/-- Eq. (weights): the weights are normalized. -/
theorem weights_sum : A.p₀ + 2 * A.p₁ + A.p₂ = 1 := by
  have hZ : A.Z ≠ 0 := A.Z_pos.ne'
  calc A.p₀ + 2 * A.p₁ + A.p₂ = A.Z / A.Z := by
        unfold p₀ p₁ p₂ Z
        ring
    _ = 1 := div_self hZ

/-- Eq. (weights): `p₁/p₀ = e^{β(a + μ)}`. -/
theorem p₁_eq_p₀_mul : A.p₁ = A.p₀ * Real.exp (A.β * (A.a + A.μ)) := by
  unfold p₀ p₁ boltz₀ boltz₁ a
  rw [div_mul_eq_mul_div, ← Real.exp_add]
  congr 2
  ring

/-- Eq. (weights): `p₁/p₂ = e^{β(a - μ)}`. -/
theorem p₁_eq_p₂_mul : A.p₁ = A.p₂ * Real.exp (A.β * (A.a - A.μ)) := by
  unfold p₂ p₁ boltz₂ boltz₁ a
  rw [div_mul_eq_mul_div, ← Real.exp_add]
  congr 2
  ring

/-- Eq. (weights): `p₂/p₀ = e^{2βμ}`. -/
theorem p₂_eq_p₀_mul : A.p₂ = A.p₀ * Real.exp (2 * A.β * A.μ) := by
  unfold p₀ p₂ boltz₀ boltz₂
  rw [div_mul_eq_mul_div, ← Real.exp_add]
  congr 2
  ring

/-- Part of Eq. (weight-ids): `1 - ρ² = 4 q₊ q₋`, as `q₊ + q₋ = 1` and `q₊ - q₋ = ρ`. -/
theorem one_sub_rho_sq : 1 - A.ρ ^ 2 = 4 * A.qp * A.qm := by
  have h := A.weights_sum
  unfold ρ qp qm
  linear_combination (-(A.p₀ + 2 * A.p₁ + A.p₂) - 1) * h

/-- Eq. (weight-ids): `ρ² + 4 p₀ p₂ = (p₀ + p₂)²`. -/
theorem rho_sq_add : A.ρ ^ 2 + 4 * A.p₀ * A.p₂ = (A.p₀ + A.p₂) ^ 2 := by
  unfold ρ
  ring

/-! ### The Matsubara lattice -/

/-- The fermionic Matsubara frequency `ν_n = (2n + 1)π/β`. -/
noncomputable def ν (n : ℤ) : ℝ := (2 * n + 1) * Real.pi / A.β

/-- The bosonic Matsubara frequency `ω_k = 2kπ/β`. -/
noncomputable def ω (k : ℤ) : ℝ := 2 * k * Real.pi / A.β

/-- `x_n = μ + iν_n`. -/
noncomputable def x (n : ℤ) : ℂ := A.μ + A.ν n * I

/-- `w = iω_k`. -/
noncomputable def w (k : ℤ) : ℂ := A.ω k * I

/-- `b = 2μ - w`. -/
noncomputable def b (k : ℤ) : ℂ := 2 * A.μ - A.w k

/-- `t_n = x_n (b - x_n)`. -/
noncomputable def t (k n : ℤ) : ℂ := A.x n * (A.b k - A.x n)

@[simp] theorem w_zero : A.w 0 = 0 := by
  simp [w, ω]

theorem ν_ne_zero (n : ℤ) : A.ν n ≠ 0 := by
  have h2 : (2 * n + 1 : ℤ) ≠ 0 := by omega
  have h : (2 * (n : ℝ) + 1) ≠ 0 := by exact_mod_cast h2
  exact div_ne_zero (mul_ne_zero h Real.pi_ne_zero) A.β_pos.ne'

theorem ν_add_ω (n k : ℤ) : A.ν n + A.ω k = A.ν (n + k) := by
  unfold ν ω
  push_cast
  ring

/-- `x_n + w = μ + i(ν_n + ω_k) = x_{n + k}`. -/
theorem x_add_w (n k : ℤ) : A.x n + A.w k = A.x (n + k) := by
  unfold x w
  rw [← A.ν_add_ω]
  push_cast
  ring

/-- Eq. (z): `ν_{n̄} = -ν_n - ω_k`. -/
theorem ν_mirror (k n : ℤ) : A.ν (mirror k n) = -A.ν n - A.ω k := by
  unfold ν ω mirror
  push_cast
  ring

/-- Eq. (z): `x_{n̄} = 2μ - w - x_n = b - x_n`. -/
theorem x_mirror (k n : ℤ) : A.x (mirror k n) = A.b k - A.x n := by
  unfold x b w
  rw [A.ν_mirror]
  push_cast
  ring

/-- `t_{n̄} = t_n`. -/
theorem t_mirror (k n : ℤ) : A.t k (mirror k n) = A.t k n := by
  unfold t
  rw [A.x_mirror]
  ring

/-- The centered lattice `z_n = x_n + w/2 = μ + iπ(2n + 1 + k)/β` of Eq. (z). -/
noncomputable def z (k n : ℤ) : ℂ := A.x n + A.w k / 2

theorem z_eq (k n : ℤ) : A.z k n = A.μ + ((2 * n + 1 + k) * Real.pi / A.β : ℝ) * I := by
  unfold z x w ν ω
  push_cast
  ring

/-- `t_n = b²/4 - (z_n - μ)²`. -/
theorem t_eq_z (k n : ℤ) : A.t k n = A.b k ^ 2 / 4 - (A.z k n - A.μ) ^ 2 := by
  unfold t z b
  ring

@[simp] theorem x_im (n : ℤ) : (A.x n).im = A.ν n := by
  simp [x]

/-- No `x_n` is real: `x_n - r ≠ 0` for real `r`, as `Im x_n = ν_n ≠ 0`. -/
theorem x_sub_ofReal_ne_zero (n : ℤ) (r : ℝ) : A.x n - r ≠ 0 := by
  intro h
  have := congrArg Complex.im h
  simp only [sub_im, x_im, ofReal_im, sub_zero, zero_im] at this
  exact A.ν_ne_zero n this

theorem x_add_ofReal_ne_zero (n : ℤ) (r : ℝ) : A.x n + r ≠ 0 := by
  have h := A.x_sub_ofReal_ne_zero n (-r)
  rwa [ofReal_neg, sub_neg_eq_add] at h

end Params

end HubbardAtom
