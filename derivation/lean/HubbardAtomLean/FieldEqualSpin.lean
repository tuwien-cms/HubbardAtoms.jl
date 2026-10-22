import HubbardAtomLean.FieldPair
import HubbardAtomLean.Triplet

/-!
# Equal-spin pairing

The equal-spin vertex, Eq. (field-E) of the note (App. F.6.3). The
equal-spin pair `χ_Eσ` is `-1/2` times the full same-spin four-point function at transfer
`-ω - ν - ν'`. Only `c_σ` enters, and `n_{-σ}` is conserved, so the four-point function has four
Fock-space paths, and `χ_Eσ` is diagonal on the mirror orbits, with eigenvalue `0` on the
mirror-even and `βQ(xₙ, x_n̄)` on the mirror-odd vectors (`chiEUp_eq`), where
`Q(x, x̄) = (p₀ + p↑)/((x + a)(x̄ + a)) + (p↓ + p₂)/((x - a)(x̄ - a))` and `x = μ↑ + iνₙ`,
`μ↑ = μ + h`. The bubble has the same form with `G↑(x)G↑(x̄)`.

The vertex is defined on the mirror-odd (Pauli-allowed) subspace `𝓜₋`: the restricted inverses
`Y` satisfy `χY = Yχ = 𝓜₋` and `Y = Y𝓜₋`, and `Γ_E↑ = β²(Y_χ - Y_b)` is Eq. (field-E).

* `FParams.chiEUp_eq`, `bEUp_eq`: the susceptibility and the bubble from the Hamiltonian.
* `FParams.gammaEUp_eq`: **the equal-spin vertex** `Γ_E↑`, for `ℒ(tₙ) ≠ 0` off the fixed points.
* `FParams.gammaEDn_eq`: `Γ_E↓(U, μ, h) = Γ_E↑(U, μ, -h)`.

The inverse needs `ℒ(tₙ) ≠ 0` on every orbit off the mirror fixed points; the note states
Eq. (field-E) entrywise.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-- The lattice of spin `↑`: `x_n = μ + h + iν_n`. -/
noncomputable abbrev Bup : Params := F.lat (F.μ + F.h)

/-- `Q(x, x̄) = (p₀ + p↑)/((x + a)(x̄ + a)) + (p↓ + p₂)/((x - a)(x̄ - a))` at `x = xₙ`, `x̄ = x_n̄`:
the mixture over `n↓ = 0, 1` of free pair propagators. -/
noncomputable def QEUp (n : ℤ) : ℂ :=
  (F.p₀ + F.pUp) / ((F.Bup.x n + F.a) * (F.Bup.x (mirror k n) + F.a)) +
    (F.pDn + F.p₂) / ((F.Bup.x n - F.a) * (F.Bup.x (mirror k n) - F.a))

/-- `G↑(xₙ)G↑(x_n̄)`. -/
noncomputable def GGEUp (n : ℤ) : ℂ := F.Gc F.rUp (F.Bup.x n) * F.Gc F.rUp (F.Bup.x (mirror k n))

theorem QEUp_mirror (n : ℤ) : F.QEUp k (mirror k n) = F.QEUp k n := by
  rw [QEUp, QEUp, mirror_mirror]
  ring

theorem GGEUp_mirror (n : ℤ) : F.GGEUp k (mirror k n) = F.GGEUp k n := by
  rw [GGEUp, GGEUp, mirror_mirror, mul_comm]

/-! ### The input from the Hamiltonian -/

-- the four paths of `χ_E↑` make the `field_simp` terms deep
set_option maxRecDepth 4000 in
/-- The equal-spin pair from the Hamiltonian: `χ_E↑,nm = (β/2)Q(xₙ, x_n̄)(δₙₘ - δ_{m n̄})`, a
matrix on the mirror orbits with eigenvalues `0` (even) and `βQ` (odd). -/
theorem chiEUp_eq (n m : ℤ) :
    F.chiEUp k n m = mirrorBlock k (fun _ ↦ 0) (fun j ↦ F.β * F.QEUp k j) n m := by
  set B := F.Bup with hB
  set k' := -k - n - m - 1 with hk'
  have hω : -F.ω k - F.ν n - F.ν m = F.ω k' := (F.ω_pp k n m).symm
  have h1 : cexp ((((F.μ + F.h : ℝ) : ℂ) - B.x n) * F.β) = -1 := B.cexp_mu_sub_x n
  have h2 : cexp ((B.x n + B.w k' - ((F.μ + F.h : ℝ) : ℂ)) * F.β) = -1 := by
    rw [B.x_add_w]
    exact B.cexp_x_sub_mu (n + k')
  have h3 : cexp ((((F.μ + F.h : ℝ) : ℂ) - B.x m - B.w k') * F.β) = -1 := by
    rw [sub_sub, B.x_add_w]
    exact B.cexp_mu_sub_x (m + k')
  have hφ : ∀ τ₁ τ₂ τ₃ : ℝ,
      (((-F.ν n * τ₁ + (F.ν n + F.ω k') * τ₂ - (F.ν m + F.ω k') * τ₃ : ℝ) : ℂ) * I) =
        (((F.μ + F.h : ℝ) : ℂ) - B.x n) * τ₁ + (B.x n + B.w k' - ((F.μ + F.h : ℝ) : ℂ)) * τ₂ +
          (((F.μ + F.h : ℝ) : ℂ) - B.x m - B.w k') * τ₃ :=
    B.freq_eq k' n m
  rw [chiEUp, hω, F.corr_eq _ _ _ _ _ _ _ hφ]
  simp only [orderedTerm_eq]
  simp [Fin.sum_univ_four, cUp_eq]
  have hk := B.w_eq_zero_iff k'
  have hnm := B.x_sub_x_eq_zero_iff n m
  push_cast at h1 h2 h3
  rw [F.pathTerm_eq 1 0 1 0 h1 h2 h3 (r₁ := -(B.x n + F.a)) (r₂ := B.x n + B.w k' + F.a)
      (r₃ := -(B.x m + B.w k' + F.a)) (r₁₂ := B.w k') (r₂₃ := B.x n - B.x m)
      (r₁₂₃ := -(B.x m + F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hnm,
    F.pathTerm_eq 3 2 3 2 h1 h2 h3 (r₁ := -(B.x n - F.a)) (r₂ := B.x n + B.w k' - F.a)
      (r₃ := -(B.x m + B.w k' - F.a)) (r₁₂ := B.w k') (r₂₃ := B.x n - B.x m)
      (r₁₂₃ := -(B.x m - F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hnm,
    F.pathTerm_eq 1 0 1 0 h3 h2 h1 (r₁ := -(B.x m + B.w k' + F.a)) (r₂ := B.x n + B.w k' + F.a)
      (r₃ := -(B.x n + F.a)) (r₁₂ := B.x n - B.x m) (r₂₃ := B.w k')
      (r₁₂₃ := -(B.x m + F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hk,
    F.pathTerm_eq 3 2 3 2 h3 h2 h1 (r₁ := -(B.x m + B.w k' - F.a)) (r₂ := B.x n + B.w k' - F.a)
      (r₃ := -(B.x n - F.a)) (r₁₂ := B.x n - B.x m) (r₂₃ := B.w k')
      (r₁₂₃ := -(B.x m - F.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hk]
  have hkm : k' = 0 ↔ m = mirror k n := by
    rw [hk']
    unfold mirror
    omega
  have hw := F.weights_sum'
  have lx : ∀ (j : ℤ) (c : ℝ), B.x j + c ≠ 0 := fun j c ↦ B.x_add_ofReal_ne_zero j c
  have lx' : ∀ (j : ℤ) (c : ℝ), B.x j - c ≠ 0 := fun j c ↦ B.x_sub_ofReal_ne_zero j c
  have lxw : ∀ (j : ℤ) (c : ℝ), B.x j + B.w k' + c ≠ 0 := fun j c ↦ by
    rw [B.x_add_w]
    exact B.x_add_ofReal_ne_zero _ c
  have lxw' : ∀ (j : ℤ) (c : ℝ), B.x j + B.w k' - c ≠ 0 := fun j c ↦ by
    rw [B.x_add_w]
    exact B.x_sub_ofReal_ne_zero _ c
  have n₁ := lx n F.a
  have n₂ := lx' n F.a
  have n₃ := lxw n F.a
  have n₄ := lxw' n F.a
  have m₁ := lx m F.a
  have m₂ := lx' m F.a
  have m₃ := lxw m F.a
  have m₄ := lxw' m F.a
  simp only [simplexW, weight, Matrix.cons_val, Matrix.cons_val_zero, Matrix.cons_val_one]
  unfold mirrorBlock QEUp
  by_cases hnm0 : n = m
  · subst hnm0
    simp only [ite_true, kron_self]
    by_cases hk0 : k' = 0
    · -- a mirror fixed point: both sides vanish
      have hfix : n = mirror k n := hkm.mp hk0
      have hw0 : B.w k' = 0 := hk.mpr hk0
      simp only [eq_true hk0, ite_true]
      rw [← hfix, kron_self]
      rw [hw0] at n₃ n₄ ⊢
      simp only [add_zero] at n₃ n₄ ⊢
      generalize B.x n = X at *
      generalize (F.a : ℂ) = a at *
      field_simp
      ring
    · have hfix : n ≠ mirror k n := fun h ↦ hk0 (hkm.mpr h)
      have hxm : B.x (mirror k n) = B.x n + B.w k' := by
        rw [B.x_add_w]
        congr 1
        rw [hk']
        unfold mirror
        ring
      have hW : B.w k' ≠ 0 := mt hk.mp hk0
      simp only [eq_false hk0, ite_false]
      rw [kron_of_ne hfix, hxm]
      generalize B.x n = X at *
      generalize B.w k' = W at *
      generalize (F.a : ℂ) = a at *
      generalize (F.β : ℂ) = β at *
      generalize (F.p₂ : ℂ) = p₂ at *
      obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hw
      field_simp
      ring
  · simp only [eq_false hnm0, ite_false]
    rw [kron_of_ne hnm0]
    have hXY : B.x n - B.x m ≠ 0 := mt hnm.mp hnm0
    by_cases hk0 : k' = 0
    · have hm : m = mirror k n := hkm.mp hk0
      have hw0 : B.w k' = 0 := hk.mpr hk0
      simp only [eq_true hk0, ite_true]
      rw [← hm, kron_self]
      rw [hw0] at n₃ n₄ m₃ m₄ ⊢
      simp only [add_zero] at n₃ n₄ m₃ m₄ ⊢
      generalize B.x n = X at *
      generalize B.x m = Y at *
      generalize (F.a : ℂ) = a at *
      generalize (F.β : ℂ) = β at *
      generalize (F.p₂ : ℂ) = p₂ at *
      obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hw
      field_simp
      ring
    · have hm : m ≠ mirror k n := fun h ↦ hk0 (hkm.mpr h)
      have hW : B.w k' ≠ 0 := mt hk.mp hk0
      simp only [eq_false hk0, ite_false]
      rw [kron_of_ne hm]
      generalize B.x n = X at *
      generalize B.x m = Y at *
      generalize B.w k' = W at *
      generalize (F.a : ℂ) = a at *
      field_simp
      ring

/-! ### The bubble -/

/-- The equal-spin bubble from the Hamiltonian: `(β/2)G↑(νₙ)G↑(-νₙ - ω_k) = (β/2)G↑(xₙ)G↑(x_n̄)`. -/
theorem bEUp_eq (n : ℤ) : F.bE true k n = F.β / 2 * F.GGEUp k n := by
  have hm : -F.ν n - F.ω k = F.ν (mirror k n) := ((F.lat 0).ν_mirror k n).symm
  rw [bE, hm, F.propagator_up, F.propagator_up, GGEUp]
  ring

/-- The bubble matrix `b_E↑,nm = (β/2)G↑G↑(δₙₘ - δ_{m n̄})` on the mirror orbits. -/
theorem bEUp_mat (n m : ℤ) : F.bE true k n * (kron n m - kron m (mirror k n)) =
    mirrorBlock k (fun _ ↦ 0) (fun j ↦ F.β * F.GGEUp k j) n m := by
  rw [F.bEUp_eq, mirrorBlock]
  ring

/-! ### The polynomials of the particle–particle variables -/

/-- `𝒫(t) = (t + a²)² - a²b²` with `b = b↑ = 2μ↑ - iω_k`. -/
noncomputable def calPE (t : ℂ) : ℂ := (t + F.a ^ 2) ^ 2 - F.a ^ 2 * F.Bup.b k ^ 2

/-- `𝒢(t) = t + arb + a²r²` with `r = r↑`. -/
noncomputable def calGE (t : ℂ) : ℂ := t + F.a * F.rUp * F.Bup.b k + F.a ^ 2 * F.rUp ^ 2

/-- `ℒ(t) = t + arb + a²`. -/
noncomputable def calLE (t : ℂ) : ℂ := t + F.a * F.rUp * F.Bup.b k + F.a ^ 2

/-- The closed form of Eq. (field-E):
`Γ_E↑,nm = -βa²(1 - r²)𝒫(tₙ)/(2𝒢(tₙ)ℒ(tₙ)) (δₙₘ - δ_{m n̄})`, `tₙ = xₙ(b - xₙ)`. -/
noncomputable def gammaEUp (n m : ℤ) : ℂ :=
  -(F.β * F.a ^ 2 * (1 - F.rUp ^ 2) * F.calPE k (F.Bup.t k n)) /
      (2 * F.calGE k (F.Bup.t k n) * F.calLE k (F.Bup.t k n)) *
    (kron n m - kron m (mirror k n))

theorem x_mirror_up (n : ℤ) : F.Bup.x (mirror k n) = F.Bup.b k - F.Bup.x n := F.Bup.x_mirror k n

/-- `Q = ℒ/𝒫` and `G↑G↑ = 𝒢/𝒫` at `t = tₙ`. -/
theorem QEUp_eq (n : ℤ) : F.QEUp k n = F.calLE k (F.Bup.t k n) / F.calPE k (F.Bup.t k n) := by
  have hw := F.weights_sum'
  have h₁ := F.Bup.x_add_ofReal_ne_zero n F.a
  have h₂ := F.Bup.x_sub_ofReal_ne_zero n F.a
  have h₃ := F.Bup.x_add_ofReal_ne_zero (mirror k n) F.a
  have h₄ := F.Bup.x_sub_ofReal_ne_zero (mirror k n) F.a
  rw [F.x_mirror_up] at h₃ h₄
  rw [QEUp, F.x_mirror_up, calLE, calPE, Params.t, rUp, ρ, mf]
  push_cast
  generalize F.Bup.x n = X at *
  generalize F.Bup.b k = b at *
  generalize (F.a : ℂ) = a at *
  generalize (F.p₂ : ℂ) = p₂ at *
  obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hw
  have hP : (X * (b - X) + a ^ 2) ^ 2 - a ^ 2 * b ^ 2 ≠ 0 := by
    rw [show (X * (b - X) + a ^ 2) ^ 2 - a ^ 2 * b ^ 2 =
      (X + a) * (b - X + a) * ((X - a) * (b - X - a)) by ring]
    exact mul_ne_zero (mul_ne_zero h₁ h₃) (mul_ne_zero h₂ h₄)
  field_simp
  ring

theorem GGEUp_eq (n : ℤ) : F.GGEUp k n = F.calGE k (F.Bup.t k n) / F.calPE k (F.Bup.t k n) := by
  have h₁ := F.Bup.x_add_ofReal_ne_zero n F.a
  have h₂ := F.Bup.x_sub_ofReal_ne_zero n F.a
  have h₃ := F.Bup.x_add_ofReal_ne_zero (mirror k n) F.a
  have h₄ := F.Bup.x_sub_ofReal_ne_zero (mirror k n) F.a
  rw [F.x_mirror_up] at h₃ h₄
  rw [GGEUp, F.x_mirror_up, calGE, calPE, Params.t]
  simp only [Gc]
  generalize F.Bup.x n = X at *
  generalize F.Bup.b k = b at *
  generalize (F.a : ℂ) = a at *
  have g₁ : X ^ 2 - a ^ 2 ≠ 0 := by rw [sq_sub_sq]; exact mul_ne_zero h₁ h₂
  have g₂ : (b - X) ^ 2 - a ^ 2 ≠ 0 := by rw [sq_sub_sq]; exact mul_ne_zero h₃ h₄
  have hP : (X * (b - X) + a ^ 2) ^ 2 - a ^ 2 * b ^ 2 ≠ 0 := by
    rw [show (X * (b - X) + a ^ 2) ^ 2 - a ^ 2 * b ^ 2 =
      (X + a) * (b - X + a) * ((X - a) * (b - X - a)) by ring]
    exact mul_ne_zero (mul_ne_zero h₁ h₃) (mul_ne_zero h₂ h₄)
  field_simp
  ring

theorem calPE_ne_zero (n : ℤ) : F.calPE k (F.Bup.t k n) ≠ 0 := by
  have h₁ := F.Bup.x_add_ofReal_ne_zero n F.a
  have h₂ := F.Bup.x_sub_ofReal_ne_zero n F.a
  have h₃ := F.Bup.x_add_ofReal_ne_zero (mirror k n) F.a
  have h₄ := F.Bup.x_sub_ofReal_ne_zero (mirror k n) F.a
  rw [F.x_mirror_up] at h₃ h₄
  rw [calPE, Params.t, show (F.Bup.x n * (F.Bup.b k - F.Bup.x n) + (F.a : ℂ) ^ 2) ^ 2 -
      (F.a : ℂ) ^ 2 * F.Bup.b k ^ 2 = (F.Bup.x n + F.a) * (F.Bup.b k - F.Bup.x n + F.a) *
      ((F.Bup.x n - F.a) * (F.Bup.b k - F.Bup.x n - F.a)) by ring]
  exact mul_ne_zero (mul_ne_zero h₁ h₃) (mul_ne_zero h₂ h₄)

theorem calGE_ne_zero (n : ℤ) : F.calGE k (F.Bup.t k n) ≠ 0 := by
  have h₁ := F.Bup.x_add_ofReal_ne_zero n (F.a * F.rUp)
  have h₂ := F.Bup.x_add_ofReal_ne_zero (mirror k n) (F.a * F.rUp)
  rw [F.x_mirror_up] at h₂
  rw [calGE, Params.t, show F.Bup.x n * (F.Bup.b k - F.Bup.x n) + F.a * F.rUp * F.Bup.b k +
      (F.a : ℂ) ^ 2 * F.rUp ^ 2 = (F.Bup.x n + ((F.a * F.rUp : ℝ) : ℂ)) *
      (F.Bup.b k - F.Bup.x n + ((F.a * F.rUp : ℝ) : ℂ)) by push_cast; ring]
  exact mul_ne_zero h₁ h₂

/-! ### The vertex on the mirror-odd subspace -/

theorem mirrorBlock_congr_odd {B L L' : ℤ → ℂ} (hL : ∀ j, j ≠ mirror k j → L j = L' j)
    (n m : ℤ) : mirrorBlock k B L n m = mirrorBlock k B L' n m := by
  by_cases hfix : mirror k n = n
  · rw [mirrorBlock_of_fixed hfix, mirrorBlock_of_fixed hfix]
  · unfold mirrorBlock
    rw [hL n (Ne.symm hfix)]

/-- The projector `𝓜₋ = (1 - 𝓜)/2` onto the mirror-odd vectors. -/
noncomputable def Mminus (n m : ℤ) : ℂ := mirrorBlock k (fun _ ↦ 0) (fun _ ↦ 1) n m

/-- The inverse of `χ_E↑` on `𝓜₋`, extended by zero: eigenvalue `1/(βQ)` on the odd vectors. -/
noncomputable def invEUp (n m : ℤ) : ℂ :=
  mirrorBlock k (fun _ ↦ 0) (fun j ↦ 1 / (F.β * F.QEUp k j)) n m

/-- The inverse of the bubble `b_E↑` on `𝓜₋`. -/
noncomputable def invbEUp (n m : ℤ) : ℂ :=
  mirrorBlock k (fun _ ↦ 0) (fun j ↦ 1 / (F.β * F.GGEUp k j)) n m

/-- **The equal-spin vertex**: for `ℒ(tₙ) ≠ 0` off the mirror fixed points, `invEUp` is the
inverse of `χ_E↑` on `𝓜₋` (`χY = Yχ = 𝓜₋`, `Y𝓜₋ = Y`), `invbEUp` that of the bubble
(`bY = Yb = 𝓜₋`), and
`β²(Y_χ - Y_b) = Γ_E↑` of Eq. (field-E). -/
theorem gammaEUp_eq (hL : ∀ n, n ≠ mirror k n → F.calLE k (F.Bup.t k n) ≠ 0) :
    (∀ n m, HasSum (fun l ↦ F.chiEUp k n l * F.invEUp k l m) (Mminus k n m)) ∧
      (∀ n m, HasSum (fun l ↦ F.invEUp k n l * F.chiEUp k l m) (Mminus k n m)) ∧
      (∀ n m, HasSum (fun l ↦ F.invEUp k n l * Mminus k l m) (F.invEUp k n m)) ∧
      (∀ n m, HasSum (fun l ↦ F.bE true k n * (kron n l - kron l (mirror k n)) *
        F.invbEUp k l m) (Mminus k n m)) ∧
      (∀ n m, HasSum (fun l ↦ F.invbEUp k n l *
        (F.bE true k l * (kron l m - kron m (mirror k l)))) (Mminus k n m)) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * (F.invEUp k n m - F.invbEUp k n m) = F.gammaEUp k n m := by
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  have hQ : ∀ j, j ≠ mirror k j → F.QEUp k j ≠ 0 := fun j hj ↦ by
    rw [F.QEUp_eq]
    exact div_ne_zero (hL j hj) (F.calPE_ne_zero k j)
  have hGG : ∀ j, F.GGEUp k j ≠ 0 := fun j ↦ by
    rw [F.GGEUp_eq]
    exact div_ne_zero (F.calGE_ne_zero k j) (F.calPE_ne_zero k j)
  have sQ : ∀ j, F.β * F.QEUp k (mirror k j) = F.β * F.QEUp k j := fun j ↦ by
    rw [F.QEUp_mirror]
  have sQ' : ∀ j, 1 / (F.β * F.QEUp k (mirror k j)) = 1 / (F.β * F.QEUp k j) := fun j ↦ by
    rw [F.QEUp_mirror]
  have sG' : ∀ j, 1 / (F.β * F.GGEUp k (mirror k j)) = 1 / (F.β * F.GGEUp k j) := fun j ↦ by
    rw [F.GGEUp_mirror]
  have sG : ∀ j, F.β * F.GGEUp k (mirror k j) = F.β * F.GGEUp k j := fun j ↦ by
    rw [F.GGEUp_mirror]
  have s0 : ∀ j : ℤ, (fun _ : ℤ ↦ (0 : ℂ)) (mirror k j) = (fun _ : ℤ ↦ (0 : ℂ)) j :=
    fun _ ↦ rfl
  have s1 : ∀ j : ℤ, (fun _ : ℤ ↦ (1 : ℂ)) (mirror k j) = (fun _ : ℤ ↦ (1 : ℂ)) j :=
    fun _ ↦ rfl
  refine ⟨fun n m ↦ ?_, fun n m ↦ ?_, fun n m ↦ ?_, fun n m ↦ ?_, fun n m ↦ ?_, fun n m ↦ ?_⟩
  · have h := hasSum_mirrorBlock_mul (k := k) (B := fun _ ↦ 0)
      (L := fun j ↦ F.β * F.QEUp k j) (B' := fun _ ↦ 0) (L' := fun j ↦ 1 / (F.β * F.QEUp k j))
      s0 sQ' n m
    simp only [invEUp, Mminus, F.chiEUp_eq]
    convert h using 1
    simp only [mul_zero]
    exact mirrorBlock_congr_odd k (fun j hj ↦ by field_simp [hQ j hj]) n m
  · have h := hasSum_mirrorBlock_mul (k := k) (B := fun _ ↦ 0)
      (L := fun j ↦ 1 / (F.β * F.QEUp k j)) (B' := fun _ ↦ 0) (L' := fun j ↦ F.β * F.QEUp k j)
      s0 sQ n m
    simp only [invEUp, Mminus, F.chiEUp_eq]
    convert h using 1
    simp only [mul_zero]
    exact mirrorBlock_congr_odd k (fun j hj ↦ by field_simp [hQ j hj]) n m
  · have h := hasSum_mirrorBlock_mul (k := k) (B := fun _ ↦ 0)
      (L := fun j ↦ 1 / (F.β * F.QEUp k j)) (B' := fun _ ↦ 0) (L' := fun _ ↦ 1) s0 s1 n m
    simp only [invEUp, Mminus]
    convert h using 1
    simp only [mul_zero, mul_one]
  · have h := hasSum_mirrorBlock_mul (k := k) (B := fun _ ↦ 0)
      (L := fun j ↦ F.β * F.GGEUp k j) (B' := fun _ ↦ 0)
      (L' := fun j ↦ 1 / (F.β * F.GGEUp k j)) s0 sG' n m
    simp only [invbEUp, Mminus, F.bEUp_mat]
    convert h using 1
    simp only [mul_zero]
    exact mirrorBlock_congr_odd k (fun j _ ↦ by field_simp [hGG j]) n m
  · have h := hasSum_mirrorBlock_mul (k := k) (B := fun _ ↦ 0)
      (L := fun j ↦ 1 / (F.β * F.GGEUp k j)) (B' := fun _ ↦ 0)
      (L' := fun j ↦ F.β * F.GGEUp k j) s0 sG n m
    simp only [invbEUp, Mminus, F.bEUp_mat]
    convert h using 1
    simp only [mul_zero]
    exact mirrorBlock_congr_odd k (fun j _ ↦ by field_simp [hGG j]) n m
  · rw [invEUp, invbEUp, gammaEUp]
    by_cases hfix : mirror k n = n
    · rw [mirrorBlock_of_fixed hfix, mirrorBlock_of_fixed hfix, hfix, kron_comm m n]
      ring
    · have hne : n ≠ mirror k n := Ne.symm hfix
      have hLn := hL n hne
      have hGn := F.calGE_ne_zero k n
      have hPn := F.calPE_ne_zero k n
      unfold mirrorBlock
      simp only [F.QEUp_eq, F.GGEUp_eq, calLE, calGE, calPE] at hLn hGn hPn ⊢
      field_simp
      ring

theorem bEDn_eq_flip (n : ℤ) : F.bE false k n = F.flip.bE true k n := by
  have hm : -F.ν n - F.ω k = F.ν (mirror k n) := ((F.lat 0).ν_mirror k n).symm
  rw [bE, bE, show F.flip.ν n = F.ν n from rfl, flip_ω, hm, flip_Gs_true, flip_Gs_true]
  rfl

/-- **The equal-spin vertex `E↓`**: `Γ_E↓(U, μ, h) = Γ_E↑(U, μ, -h)`, as `χ_E↓ = χ_E↑(U, μ, -h)`
and the bubbles agree. -/
theorem gammaEDn_eq (hL : ∀ n, n ≠ mirror k n → F.flip.calLE k (F.flip.Bup.t k n) ≠ 0) :
    (∀ n m, HasSum (fun l ↦ F.chiEDn k n l * F.flip.invEUp k l m) (Mminus k n m)) ∧
      (∀ n m, HasSum (fun l ↦ F.flip.invEUp k n l * F.chiEDn k l m) (Mminus k n m)) ∧
      (∀ n m, HasSum (fun l ↦ F.flip.invEUp k n l * Mminus k l m) (F.flip.invEUp k n m)) ∧
      (∀ n m, HasSum (fun l ↦ F.bE false k n * (kron n l - kron l (mirror k n)) *
        F.flip.invbEUp k l m) (Mminus k n m)) ∧
      (∀ n m, HasSum (fun l ↦ F.flip.invbEUp k n l *
        (F.bE false k l * (kron l m - kron m (mirror k l)))) (Mminus k n m)) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * (F.flip.invEUp k n m - F.flip.invbEUp k n m) =
        F.flip.gammaEUp k n m := by
  have hχ : F.chiEDn k = F.flip.chiEUp k := funext fun n ↦ funext fun m ↦ F.chiEDn_eq k n m
  have hb : ∀ n, F.bE false k n = F.flip.bE true k n := F.bEDn_eq_flip k
  obtain ⟨h₁, h₂, h₃, h₄, h₅, h₆⟩ := F.flip.gammaEUp_eq k hL
  refine ⟨hχ ▸ h₁, hχ ▸ h₂, h₃, fun n m ↦ ?_, fun n m ↦ ?_, h₆⟩
  · simp only [hb]
    exact h₄ n m
  · simp only [hb]
    exact h₅ n m

end FParams

end HubbardAtom
