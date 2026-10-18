import HubbardAtomLean.Basic

/-!
# The triplet channel: Theorem "triplet"

Sections "Singlet and triplet" and "Proofs for the singlet and triplet channels" of the note.

* `Params.chiUpUp`: the input, Eq. (chi-uu), the same-spin generalized susceptibility of
  Ref. [Eßl et al., PRR 6, 033061 (2024), Eq. (D3)], taken as given.
* `Params.chiT`: the triplet susceptibility `χ_t = (χ_pp↑↑ + 2χ_0,pp)/4`, with the
  particle–particle form `χ_pp(ν, ν', ω) = χ(ν, ν', -ω - ν - ν')` and the bubble of Eq. (B).
* `Params.chiT_eq_mirrorBlock`: the triplet part of Proposition "pp", `χ_t = B 𝓜₊ + L_t 𝓜₋`.
* `Params.chiT_isRightInverse`, `Params.chiT_isLeftInverse`: `χ_t` has the two-sided inverse
  `B⁻¹ 𝓜₊ + L_t⁻¹ 𝓜₋` if no `L_t(t_n)` vanishes.
* `Params.gammaT_of_isRightInverse`, `Params.gammaT_of_isLeftInverse`: **Theorem "triplet"**.
  Every right or left inverse `Y` of `χ_t` gives `β²(Y - B⁻¹) = Γ_t` of Eq. (Gamma-t). So the
  vertex of Eq. (Gamma-def) is well defined, and it is the closed form.
* `Params.gammaT_eq_zero_of_fixed`: `Γ_t` vanishes in the row and column of a mirror fixed point.

Infinite matrices are functions `ℤ → ℤ → ℂ`, and their products are the (unconditional) sums
`∑ₗ X n l * Y l m` of `HasSum`.
-/

open Complex

namespace HubbardAtom

/-- `Y` is a right inverse of the infinite matrix `X`: `∑ₗ X n l Y l m = δₙₘ` for all `n, m`. -/
def IsRightInverse (X Y : ℤ → ℤ → ℂ) : Prop :=
  ∀ n m, HasSum (fun l ↦ X n l * Y l m) (kron n m)

/-- `Y` is a left inverse of the infinite matrix `X`: `∑ₗ Y n l X l m = δₙₘ` for all `n, m`. -/
def IsLeftInverse (X Y : ℤ → ℤ → ℂ) : Prop :=
  ∀ n m, HasSum (fun l ↦ Y n l * X l m) (kron n m)

/-! ### Matrices acting on mirror orbits -/

/-- The matrix `B 𝓜₊ + L 𝓜₋` with `𝓜± = (1 ± 𝓜)/2` and the mirror `𝓜ₙₘ = δ_{m n̄}`. If `B` and
`L` are mirror symmetric, it acts on each orbit `{n, n̄}` with eigenvalue `B n` on the even and
`L n` on the odd vector. -/
noncomputable def mirrorBlock (k : ℤ) (B L : ℤ → ℂ) (n m : ℤ) : ℂ :=
  (B n + L n) / 2 * kron n m + (B n - L n) / 2 * kron m (mirror k n)

section MirrorBlock

variable {k : ℤ} {B L : ℤ → ℂ}

theorem mirrorBlock_of_fixed {n : ℤ} (h : mirror k n = n) (m : ℤ) :
    mirrorBlock k B L n m = B n * kron n m := by
  unfold mirrorBlock
  rw [h, kron_comm m n]
  ring

theorem mirrorBlock_self {n : ℤ} (h : n ≠ mirror k n) :
    mirrorBlock k B L n n = (B n + L n) / 2 := by
  unfold mirrorBlock
  rw [kron_self, kron_of_ne h]
  ring

theorem mirrorBlock_mirror {n : ℤ} (h : n ≠ mirror k n) :
    mirrorBlock k B L n (mirror k n) = (B n - L n) / 2 := by
  unfold mirrorBlock
  rw [kron_self, kron_of_ne h]
  ring

theorem mirrorBlock_mirror_self {n : ℤ} (h : n ≠ mirror k n) :
    mirrorBlock k B L (mirror k n) n = (B (mirror k n) - L (mirror k n)) / 2 := by
  unfold mirrorBlock
  rw [mirror_mirror, kron_self, kron_of_ne (Ne.symm h)]
  ring

theorem mirrorBlock_of_ne {n m : ℤ} (h₁ : n ≠ m) (h₂ : m ≠ mirror k n) :
    mirrorBlock k B L n m = 0 := by
  unfold mirrorBlock
  rw [kron_of_ne h₁, kron_of_ne h₂]
  ring

/-- `mirrorBlock k B L n l` vanishes unless `l ∈ {n, n̄}`. -/
theorem mirrorBlock_eq_zero {n l : ℤ} (hl : l ∉ ({n, mirror k n} : Finset ℤ)) :
    mirrorBlock k B L n l = 0 := by
  simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hl
  exact mirrorBlock_of_ne (Ne.symm hl.1) hl.2

/-- Matrices acting on mirror orbits multiply orbitwise: the sum over `l` has at most two
nonzero terms, `l = n` and `l = n̄`. -/
theorem hasSum_mirrorBlock_mul {B' L' : ℤ → ℂ} (hB' : ∀ n, B' (mirror k n) = B' n)
    (hL' : ∀ n, L' (mirror k n) = L' n) (n m : ℤ) :
    HasSum (fun l ↦ mirrorBlock k B L n l * mirrorBlock k B' L' l m)
      (mirrorBlock k (fun j ↦ B j * B' j) (fun j ↦ L j * L' j) n m) := by
  have hsum := hasSum_sum_of_ne_finset_zero (L := SummationFilter.unconditional ℤ)
    (s := ({n, mirror k n} : Finset ℤ))
    (f := fun l ↦ mirrorBlock k B L n l * mirrorBlock k B' L' l m)
    (fun l hl ↦ by simp only [mirrorBlock_eq_zero hl, zero_mul])
  convert hsum using 1
  by_cases hfix : mirror k n = n
  · -- the orbit is `{n}`
    rw [hfix, Finset.insert_eq_of_mem (Finset.mem_singleton_self n), Finset.sum_singleton,
      mirrorBlock_of_fixed hfix, mirrorBlock_of_fixed hfix, mirrorBlock_of_fixed hfix, kron_self]
    ring
  · have hne : n ≠ mirror k n := Ne.symm hfix
    have hne' : mirror k n ≠ mirror k (mirror k n) := by rw [mirror_mirror]; exact hfix
    rw [Finset.sum_pair hne, mirrorBlock_self hne, mirrorBlock_mirror hne]
    unfold mirrorBlock
    rw [mirror_mirror, hB', hL']
    by_cases hm : m = n
    · subst hm
      simp only [kron_self, kron_of_ne hne, kron_of_ne hfix]
      ring
    by_cases hm' : m = mirror k n
    · subst hm'
      simp only [kron_self, kron_of_ne hne, kron_of_ne hfix]
      ring
    · simp only [kron_of_ne (Ne.symm hm), kron_of_ne hm', kron_of_ne (fun h ↦ hm' h.symm),
        kron_of_ne hm]
      ring

/-- A matrix acting on mirror orbits with mirror-symmetric `B`, `L` is symmetric. -/
theorem mirrorBlock_symm (hB : ∀ n, B (mirror k n) = B n) (hL : ∀ n, L (mirror k n) = L n)
    (n m : ℤ) : mirrorBlock k B L n m = mirrorBlock k B L m n := by
  by_cases hm : m = n
  · rw [hm]
  by_cases hm' : m = mirror k n
  · subst hm'
    have hne : n ≠ mirror k n := Ne.symm hm
    rw [mirrorBlock_mirror hne, mirrorBlock_mirror_self hne, hB, hL]
  · rw [mirrorBlock_of_ne (Ne.symm hm) hm',
      mirrorBlock_of_ne hm (fun h ↦ hm' (mirror_eq_iff.mpr h))]

/-- The `2 × 2` system of an orbit: `(B + L)/2 y + (B - L)/2 y' = c` and the mirrored equation have
the unique solution `y = (1/B + 1/L)/2 c + (1/B - 1/L)/2 c'`. -/
theorem orbit_solve {B L y y' c c' : ℂ} (hB : B ≠ 0) (hL : L ≠ 0)
    (E₁ : (B + L) / 2 * y + (B - L) / 2 * y' = c) (E₂ : (B + L) / 2 * y' + (B - L) / 2 * y = c') :
    y = (1 / B + 1 / L) / 2 * c + (1 / B - 1 / L) / 2 * c' := by
  have h : 2 * B * L * y = (B + L) * c + (L - B) * c' := by
    linear_combination (B + L) * E₁ - (B - L) * E₂
  field_simp
  linear_combination h

/-- `mirrorBlock k B L` is the identity if `B = 1`, and `L = 1` off the fixed points. -/
theorem mirrorBlock_eq_kron (hB : ∀ j, B j = 1) (hL : ∀ j, j ≠ mirror k j → L j = 1)
    (n m : ℤ) : mirrorBlock k B L n m = kron n m := by
  by_cases hfix : mirror k n = n
  · rw [mirrorBlock_of_fixed hfix, hB, one_mul]
  · unfold mirrorBlock
    rw [hB, hL n (Ne.symm hfix)]
    ring

end MirrorBlock

/-! ### The triplet susceptibility -/

namespace Params

variable (A : Params)

/-- The propagator `G(x) = (x + aρ)/(x² - a²)` of Eq. (G). -/
noncomputable def G (z : ℂ) : ℂ := (z + A.a * A.ρ) / (z ^ 2 - A.a ^ 2)

/-- Eq. (G): the Lehmann representation `G(x) = (p₀ + p₁)/(x + a) + (p₁ + p₂)/(x - a)`, with weight
`p₀ + p₁` for `|0⟩ → |↑⟩` and `p₁ + p₂` for `|↓⟩ → |↑↓⟩`. -/
theorem G_eq_lehmann {z : ℂ} (hp : z + A.a ≠ 0) (hm : z - A.a ≠ 0) :
    A.G z = A.qm / (z + A.a) + A.qp / (z - A.a) := by
  have hw : ((A.p₀ + 2 * A.p₁ + A.p₂ : ℝ) : ℂ) = 1 := by rw [A.weights_sum, ofReal_one]
  push_cast at hw
  have hsq : z ^ 2 - (A.a : ℂ) ^ 2 = (z + A.a) * (z - A.a) := by ring
  rw [G, hsq, qm, qp, ρ]
  field_simp
  push_cast
  linear_combination (-z) * hw

/-- `J(x) = 1/(x² - a²)`. -/
noncomputable def J (z : ℂ) : ℂ := 1 / (z ^ 2 - A.a ^ 2)

/-- The input, Eq. (chi-uu): the same-spin generalized susceptibility
`χ↑↑(ν_n, ν_m, ω_k)` [Eßl et al., Eq. (D3)], taken as given. -/
noncomputable def chiUpUp (k n m : ℤ) : ℂ :=
  -(A.β : ℂ) * A.G (A.x n) * A.G (A.x n + A.w k) * kron n m +
    A.β * A.a ^ 2 * (1 - A.ρ ^ 2) * A.J (A.x n + A.w k) * A.J (A.x m) * (kron k 0 - kron n m)

/-- The particle–particle form `χ_pp↑↑(ν, ν', ω) = χ↑↑(ν, ν', -ω - ν - ν')`, whose bosonic index
is `-k - n - m - 1` (`ω_pp`). -/
noncomputable def chiPPUpUp (k n m : ℤ) : ℂ := A.chiUpUp (-k - n - m - 1) n m

theorem ω_pp (k n m : ℤ) : A.ω (-k - n - m - 1) = -A.ω k - A.ν n - A.ν m := by
  unfold ω ν
  push_cast
  ring

/-- The particle–particle bubble of Eq. (B), `B(ν_n) = -(β/2) G(ν_n) G(-ν_n - ω_k)`. -/
noncomputable def Bpp (k n : ℤ) : ℂ :=
  -((A.β : ℂ) / 2) * A.G (A.μ + A.ν n * I) * A.G (A.μ + (-A.ν n - A.ω k) * I)

/-- The triplet susceptibility `χ_t = (χ_pp↑↑ + 2χ_0,pp)/4`, `χ_0,pp = δₙₘ B(ν_n)`. -/
noncomputable def chiT (k n m : ℤ) : ℂ := (A.chiPPUpUp k n m + 2 * (kron n m * A.Bpp k n)) / 4

/-- `𝒫(t) = (t + a²)² - a²b²`. -/
noncomputable def calP (k : ℤ) (t : ℂ) : ℂ := (t + A.a ^ 2) ^ 2 - A.a ^ 2 * A.b k ^ 2

/-- `𝒢(t) = t + aρb + a²ρ²`. -/
noncomputable def calG (k : ℤ) (t : ℂ) : ℂ := t + A.a * A.ρ * A.b k + A.a ^ 2 * A.ρ ^ 2

/-- `L_t(t) = -(β/2)(t + aρb + a²)/𝒫(t)`. -/
noncomputable def Lt (k : ℤ) (t : ℂ) : ℂ :=
  -((A.β : ℂ) / 2) * (t + A.a * A.ρ * A.b k + A.a ^ 2) / A.calP k t

/-- The closed form of Eq. (Gamma-t). -/
noncomputable def gammaT (k n m : ℤ) : ℂ :=
  A.β * A.a ^ 2 * (1 - A.ρ ^ 2) * A.calP k (A.t k n) /
      (A.calG k (A.t k n) * (A.t k n + A.a * A.ρ * A.b k + A.a ^ 2)) *
    (kron n m - kron m (mirror k n))

/-- The inverse `B⁻¹ 𝓜₊ + L_t⁻¹ 𝓜₋` of `χ_t`. -/
noncomputable def invT (k : ℤ) : ℤ → ℤ → ℂ :=
  mirrorBlock k (fun j ↦ 1 / A.Bpp k j) (fun j ↦ 1 / A.Lt k (A.t k j))

/-! #### Algebra on the lattice -/

variable (k : ℤ)

theorem Bpp_eq (n : ℤ) : A.Bpp k n = -((A.β : ℂ) / 2) * A.G (A.x n) * A.G (A.x (mirror k n)) := by
  rw [Bpp, Params.x, Params.x, A.ν_mirror]
  push_cast
  ring

/-- Eq. (pp-ids): `(x² - a²)(x̄² - a²) = 𝒫(t)` for `x̄ = b - x`, `t = x x̄`. -/
theorem calP_t (n : ℤ) :
    A.calP k (A.t k n) = (A.x n ^ 2 - A.a ^ 2) * (A.x (mirror k n) ^ 2 - A.a ^ 2) := by
  rw [calP, t, A.x_mirror]
  ring

/-- Eq. (pp-ids): `(x + aρ)(x̄ + aρ) = 𝒢(t)`. -/
theorem calG_t (n : ℤ) :
    A.calG k (A.t k n) = (A.x n + A.a * A.ρ) * (A.x (mirror k n) + A.a * A.ρ) := by
  rw [calG, t, A.x_mirror]
  ring

theorem x_sq_sub_ne_zero (n : ℤ) : A.x n ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
  have h : A.x n ^ 2 - (A.a : ℂ) ^ 2 = (A.x n - A.a) * (A.x n + A.a) := by ring
  rw [h]
  exact mul_ne_zero (A.x_sub_ofReal_ne_zero n _) (A.x_add_ofReal_ne_zero n _)

theorem x_add_a_rho_ne_zero (n : ℤ) : A.x n + (A.a : ℂ) * A.ρ ≠ 0 := by
  have h := A.x_add_ofReal_ne_zero n (A.a * A.ρ)
  rwa [ofReal_mul] at h

theorem calP_ne_zero (n : ℤ) : A.calP k (A.t k n) ≠ 0 := by
  rw [A.calP_t]
  exact mul_ne_zero (A.x_sq_sub_ne_zero n) (A.x_sq_sub_ne_zero _)

theorem calG_ne_zero (n : ℤ) : A.calG k (A.t k n) ≠ 0 := by
  rw [A.calG_t]
  exact mul_ne_zero (A.x_add_a_rho_ne_zero n) (A.x_add_a_rho_ne_zero _)

/-- Eq. (B-pp): `B_n = -(β/2) 𝒢(t_n)/𝒫(t_n)`. -/
theorem Bpp_eq_calG_div (n : ℤ) :
    A.Bpp k n = -((A.β : ℂ) / 2) * A.calG k (A.t k n) / A.calP k (A.t k n) := by
  rw [A.Bpp_eq, A.calG_t, A.calP_t, G, G]
  field_simp

theorem Bpp_ne_zero (n : ℤ) : A.Bpp k n ≠ 0 := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  rw [A.Bpp_eq_calG_div]
  exact div_ne_zero (mul_ne_zero (neg_ne_zero.mpr (div_ne_zero hβ two_ne_zero))
    (A.calG_ne_zero k n)) (A.calP_ne_zero k n)

theorem Bpp_mirror (n : ℤ) : A.Bpp k (mirror k n) = A.Bpp k n := by
  rw [A.Bpp_eq, A.Bpp_eq, mirror_mirror]
  ring

theorem Lt_ne_zero {n : ℤ} (h : A.t k n + A.a * A.ρ * A.b k + A.a ^ 2 ≠ 0) :
    A.Lt k (A.t k n) ≠ 0 := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  exact div_ne_zero (mul_ne_zero (neg_ne_zero.mpr (div_ne_zero hβ two_ne_zero)) h)
    (A.calP_ne_zero k n)

/-- `J(x_n) J(x_{n̄}) = 1/𝒫(t_n)`. -/
theorem J_mul_J_mirror (n : ℤ) :
    A.J (A.x n) * A.J (A.x (mirror k n)) = 1 / A.calP k (A.t k n) := by
  rw [J, J, A.calP_t]
  field_simp

/-- `B_n - L_t(t_n) = β a² (1 - ρ²)/(2𝒫(t_n))`. -/
theorem Bpp_sub_Lt (n : ℤ) :
    A.Bpp k n - A.Lt k (A.t k n) =
      A.β * A.a ^ 2 * (1 - A.ρ ^ 2) / (2 * A.calP k (A.t k n)) := by
  have hP := A.calP_ne_zero k n
  rw [A.Bpp_eq_calG_div, Lt, calG]
  field_simp
  ring

/-- The triplet part of Proposition "pp": `χ_t = B 𝓜₊ + L_t 𝓜₋`, from the input. -/
theorem chiT_eq_mirrorBlock (n m : ℤ) :
    A.chiT k n m = mirrorBlock k (A.Bpp k) (fun j ↦ A.Lt k (A.t k j)) n m := by
  have hBL := A.Bpp_sub_Lt k n
  have hJJ := A.J_mul_J_mirror k n
  have hB := A.Bpp_eq k n
  by_cases hm : m = n
  · subst hm
    rw [chiT, chiPPUpUp, chiUpUp, kron_self, A.x_add_w]
    by_cases hfix : mirror k m = m
    · -- the bosonic index of the pp form is `0`
      have h0 : -k - m - m - 1 = 0 := by unfold mirror at hfix; omega
      have hx : m + (-k - m - m - 1) = mirror k m := by unfold mirror; ring
      rw [hx, h0, mirrorBlock_of_fixed hfix, kron_self, kron_self]
      rw [hfix] at hB ⊢
      rw [hB]
      ring
    · have h0 : -k - m - m - 1 ≠ 0 := by unfold mirror at hfix; omega
      have hx : m + (-k - m - m - 1) = mirror k m := by unfold mirror; ring
      rw [mirrorBlock_self (Ne.symm hfix), kron_of_ne h0, hx]
      linear_combination (-1 / 2 : ℂ) * hB - (A.β * A.a ^ 2 * (1 - A.ρ ^ 2) / 4 : ℂ) * hJJ +
        (1 / 2 : ℂ) * hBL
  · by_cases hm' : m = mirror k n
    · subst hm'
      have h0 : -k - n - mirror k n - 1 = 0 := by unfold mirror; ring
      rw [chiT, chiPPUpUp, chiUpUp, mirrorBlock_mirror (Ne.symm hm), kron_of_ne (Ne.symm hm), h0,
        kron_self, A.w_zero, add_zero]
      linear_combination (A.β * A.a ^ 2 * (1 - A.ρ ^ 2) / 4 : ℂ) * hJJ - (1 / 2 : ℂ) * hBL
    · have h0 : -k - n - m - 1 ≠ 0 := by
        intro h
        apply hm'
        unfold mirror
        omega
      rw [chiT, chiPPUpUp, chiUpUp, mirrorBlock_of_ne (Ne.symm hm) hm', kron_of_ne (Ne.symm hm),
        kron_of_ne h0]
      ring

theorem chiT_eq : A.chiT k = mirrorBlock k (A.Bpp k) (fun j ↦ A.Lt k (A.t k j)) := by
  funext n m
  exact A.chiT_eq_mirrorBlock k n m

/-! #### The inverse -/

variable {A k}

/-- The assumption of Theorem "triplet": `L_t(t_n) ≠ 0`, i.e. `t_n + aρb + a² ≠ 0`, off the
mirror fixed points. -/
def TripletNondegenerate (A : Params) (k : ℤ) : Prop :=
  ∀ n, n ≠ mirror k n → A.t k n + A.a * A.ρ * A.b k + A.a ^ 2 ≠ 0

theorem chiT_isRightInverse (hL : TripletNondegenerate A k) :
    IsRightInverse (A.chiT k) (A.invT k) := by
  intro n m
  rw [A.chiT_eq k, invT]
  convert hasSum_mirrorBlock_mul (k := k) (B := A.Bpp k) (L := fun j ↦ A.Lt k (A.t k j))
    (B' := fun j ↦ 1 / A.Bpp k j) (L' := fun j ↦ 1 / A.Lt k (A.t k j))
    (fun j ↦ by simp only [A.Bpp_mirror]) (fun j ↦ by simp only [A.t_mirror]) n m using 1
  refine (mirrorBlock_eq_kron (fun j ↦ ?_) (fun j hj ↦ ?_) n m).symm
  · exact mul_one_div_cancel (A.Bpp_ne_zero k j)
  · exact mul_one_div_cancel (A.Lt_ne_zero k (hL j hj))

theorem chiT_isLeftInverse (hL : TripletNondegenerate A k) :
    IsLeftInverse (A.chiT k) (A.invT k) := by
  intro n m
  rw [A.chiT_eq k, invT]
  convert hasSum_mirrorBlock_mul (B := fun j ↦ 1 / A.Bpp k j) (L := fun j ↦ 1 / A.Lt k (A.t k j))
    (B' := A.Bpp k) (L' := fun j ↦ A.Lt k (A.t k j)) (A.Bpp_mirror k)
    (fun j ↦ by simp only [A.t_mirror]) n m using 1
  refine (mirrorBlock_eq_kron (fun j ↦ ?_) (fun j hj ↦ ?_) n m).symm
  · exact one_div_mul_cancel (A.Bpp_ne_zero k j)
  · exact one_div_mul_cancel (A.Lt_ne_zero k (hL j hj))

/-- Row `n` of `χ_t Y = 1` as a finite sum over the orbit `{n, n̄}`. -/
theorem chiT_row {Y : ℤ → ℤ → ℂ} (hY : IsRightInverse (A.chiT k) Y) (n m : ℤ) :
    ∑ l ∈ ({n, mirror k n} : Finset ℤ), A.chiT k n l * Y l m = kron n m := by
  refine (hasSum_sum_of_ne_finset_zero (fun l hl ↦ ?_)).unique (hY n m)
  rw [A.chiT_eq_mirrorBlock, mirrorBlock_eq_zero hl, zero_mul]

/-- A right inverse of `χ_t` is `invT`: on each orbit, `χ_t Y = 1` is a regular linear system. -/
theorem eq_invT_of_isRightInverse (hL : TripletNondegenerate A k) {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiT k) Y) (n m : ℤ) : Y n m = A.invT k n m := by
  have hB := A.Bpp_ne_zero k n
  have E₁ := A.chiT_row hY n m
  rw [invT]
  by_cases hfix : mirror k n = n
  · rw [hfix, Finset.insert_eq_of_mem (Finset.mem_singleton_self n), Finset.sum_singleton,
      A.chiT_eq_mirrorBlock, mirrorBlock_of_fixed hfix, kron_self, mul_one] at E₁
    rw [mirrorBlock_of_fixed hfix]
    field_simp
    linear_combination E₁
  · have hne : n ≠ mirror k n := Ne.symm hfix
    have hLn := A.Lt_ne_zero k (hL n hne)
    have E₁' : (A.Bpp k n + A.Lt k (A.t k n)) / 2 * Y n m +
        (A.Bpp k n - A.Lt k (A.t k n)) / 2 * Y (mirror k n) m = kron n m := by
      rw [← E₁, Finset.sum_pair hne, A.chiT_eq_mirrorBlock, A.chiT_eq_mirrorBlock,
        mirrorBlock_self hne, mirrorBlock_mirror hne]
    have E₂' : (A.Bpp k n + A.Lt k (A.t k n)) / 2 * Y (mirror k n) m +
        (A.Bpp k n - A.Lt k (A.t k n)) / 2 * Y n m = kron (mirror k n) m := by
      have E₂ := A.chiT_row hY (mirror k n) m
      rw [mirror_mirror, Finset.sum_pair hfix, A.chiT_eq_mirrorBlock, A.chiT_eq_mirrorBlock,
        mirrorBlock_self (by rw [mirror_mirror]; exact hfix), mirrorBlock_mirror_self hne,
        A.Bpp_mirror, A.t_mirror] at E₂
      exact E₂
    unfold mirrorBlock
    rw [kron_comm m (mirror k n)]
    exact orbit_solve hB hLn E₁' E₂'

theorem chiT_symm (n m : ℤ) : A.chiT k n m = A.chiT k m n := by
  rw [A.chiT_eq_mirrorBlock, A.chiT_eq_mirrorBlock]
  exact mirrorBlock_symm (A.Bpp_mirror k) (fun j ↦ by simp only [A.t_mirror]) n m

theorem invT_symm (n m : ℤ) : A.invT k n m = A.invT k m n := by
  rw [invT]
  exact mirrorBlock_symm (fun j ↦ by simp only [A.Bpp_mirror]) (fun j ↦ by simp only [A.t_mirror])
    n m

/-- A left inverse of `χ_t` is `invT`: its transpose is a right inverse, as `χ_t` is symmetric. -/
theorem eq_invT_of_isLeftInverse (hL : TripletNondegenerate A k) {Y : ℤ → ℤ → ℂ}
    (hY : IsLeftInverse (A.chiT k) Y) (n m : ℤ) : Y n m = A.invT k n m := by
  have hYt : IsRightInverse (A.chiT k) (fun i j ↦ Y j i) := by
    intro i j
    have h := hY j i
    rw [kron_comm] at h
    convert h using 2 with l
    rw [A.chiT_symm, mul_comm]
  rw [A.eq_invT_of_isRightInverse hL hYt m n, A.invT_symm]

/-- The algebra of Theorem "triplet": `β²(χ_t⁻¹ - B⁻¹) = Γ_t` for the inverse `invT`. -/
theorem beta_sq_mul_invT_sub (hL : TripletNondegenerate A k) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invT k n m - kron n m / A.Bpp k n) = A.gammaT k n m := by
  by_cases hfix : mirror k n = n
  · rw [invT, mirrorBlock_of_fixed hfix, gammaT, hfix, kron_comm m n]
    ring
  · have hN := hL n (Ne.symm hfix)
    have hP := A.calP_ne_zero k n
    have hG : A.t k n + A.a * A.ρ * A.b k + A.a ^ 2 * A.ρ ^ 2 ≠ 0 := A.calG_ne_zero k n
    have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
    have key : (A.β : ℂ) ^ 2 * (1 / A.Lt k (A.t k n) - 1 / A.Bpp k n) / 2 =
        A.β * A.a ^ 2 * (1 - A.ρ ^ 2) * A.calP k (A.t k n) /
          (A.calG k (A.t k n) * (A.t k n + A.a * A.ρ * A.b k + A.a ^ 2)) := by
      rw [Lt, A.Bpp_eq_calG_div]
      simp only [calG]
      field_simp
      ring
    rw [gammaT, ← key]
    simp only [invT, mirrorBlock]
    ring

/-- **Theorem "triplet"**, existence: `χ_t` has a two-sided inverse. -/
theorem exists_inverse_chiT (hL : TripletNondegenerate A k) :
    ∃ Y, IsRightInverse (A.chiT k) Y ∧ IsLeftInverse (A.chiT k) Y :=
  ⟨A.invT k, A.chiT_isRightInverse hL, A.chiT_isLeftInverse hL⟩

/-- **Theorem "triplet"**: for every right inverse `Y` of `χ_t`, the vertex
`Γ_t = β²(χ_t⁻¹ - B⁻¹)` of Eq. (Gamma-def) is the closed form of Eq. (Gamma-t). -/
theorem gammaT_of_isRightInverse (hL : TripletNondegenerate A k) {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiT k) Y) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (Y n m - kron n m / A.Bpp k n) = A.gammaT k n m := by
  rw [A.eq_invT_of_isRightInverse hL hY n m]
  exact A.beta_sq_mul_invT_sub hL n m

/-- **Theorem "triplet"**, for left inverses. -/
theorem gammaT_of_isLeftInverse (hL : TripletNondegenerate A k) {Y : ℤ → ℤ → ℂ}
    (hY : IsLeftInverse (A.chiT k) Y) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (Y n m - kron n m / A.Bpp k n) = A.gammaT k n m := by
  rw [A.eq_invT_of_isLeftInverse hL hY n m]
  exact A.beta_sq_mul_invT_sub hL n m

/-- Theorem "triplet", at a mirror fixed point `n = n̄`: `Γ_t` vanishes in row and column `n`. -/
theorem gammaT_eq_zero_of_fixed {n : ℤ} (h : mirror k n = n) (m : ℤ) :
    A.gammaT k n m = 0 ∧ A.gammaT k m n = 0 := by
  refine ⟨?_, ?_⟩
  · rw [gammaT, h, kron_comm m n, sub_self, mul_zero]
  · by_cases hm : m = n
    · rw [hm, gammaT, h, sub_self, mul_zero]
    · have hm' : n ≠ mirror k m := fun h' ↦ hm (by rw [mirror_eq_iff.mpr h', h])
      rw [gammaT, kron_of_ne hm, kron_of_ne hm', sub_self, mul_zero]

end Params

end HubbardAtom
