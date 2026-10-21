import HubbardAtomLean.FieldLehmann
import HubbardAtomLean.LehmannInput
import HubbardAtomLean.FieldTAlg

/-!
# The field sectors: definitions and input from the Hamiltonian

The four sectors of the atom with doping and field (App. F.6 of the note), defined from the
Hamiltonian of `FieldLehmann.lean`, and their inputs in closed form.

* `FParams.chiTUp`, `chiTDn`: the transverse susceptibilities, the full four-point functions of
  `c↑†(τ₁) c↓(τ₂) c↓†(τ₃) c↑(0)` and of its spin flip; `chiAUp`, `chiADn`: the oriented
  opposite-spin pairs, `χ_Aσ(ν, ν', ω) = -χ_Tσ(ν, ν', -ω - ν - ν')`; `chiEUp`, `chiEDn`: the
  equal-spin pairs, `-1/2` times the full same-spin four-point function at transfer `-ω - ν - ν'`;
  `chiL`: the coupled longitudinal (connected) susceptibility `χ_σt`.
* `FParams.bT`, `bA`, `bE`, `bL`: the physical bubbles of Eq. (field-bubbles).
* `FParams.propagator_up`, `propagator_dn`: `G_σ(ν) = (x + a r_σ)/(x² - a²)`, `x = iν + μ + σh`,
  `r_σ = ρ - σm`.
* `FParams.chiAUp_eq`, `chiTDn_eq`, `chiADn_eq`, `chiEDn_eq`: the general Shiba map
  `χ_A↑(U, μ, h) = χ_T↑(-U, h, μ)` and the spin flip, from the conjugations of `FieldLehmann.lean`.
-/

open Complex Matrix

namespace HubbardAtom

namespace FParams

variable (F : FParams)

/-! ### The sectors -/

/-- The transverse susceptibility `χ_T↑(ν_n, ν_m, ω_k)`: the full four-point function of
`c↑†(τ₁) c↓(τ₂) c↓†(τ₃) c↑(0)` (there is no disconnected part, as `⟨c↑† c↓⟩ = 0`). -/
noncomputable def chiTUp (k n m : ℤ) : ℂ := F.corr cUpᴴ cDn cDnᴴ cUp (F.ν n) (F.ν m) (F.ω k)

/-- The transverse susceptibility `χ_T↓`, with the spins interchanged. -/
noncomputable def chiTDn (k n m : ℤ) : ℂ := F.corr cDnᴴ cUp cUpᴴ cDn (F.ν n) (F.ν m) (F.ω k)

/-- The oriented opposite-spin pair `χ_A↑(ν, ν', ω) = -χ_T↑(ν, ν', -ω - ν - ν')`. -/
noncomputable def chiAUp (k n m : ℤ) : ℂ :=
  -F.corr cUpᴴ cDn cDnᴴ cUp (F.ν n) (F.ν m) (-F.ω k - F.ν n - F.ν m)

/-- The oriented opposite-spin pair `χ_A↓`. -/
noncomputable def chiADn (k n m : ℤ) : ℂ :=
  -F.corr cDnᴴ cUp cUpᴴ cDn (F.ν n) (F.ν m) (-F.ω k - F.ν n - F.ν m)

/-- The equal-spin pair `χ_E↑`: `-1/2` times the full same-spin four-point function at transfer
`-ω - ν - ν'`. -/
noncomputable def chiEUp (k n m : ℤ) : ℂ :=
  -(1 / 2) * F.corr cUpᴴ cUp cUpᴴ cUp (F.ν n) (F.ν m) (-F.ω k - F.ν n - F.ν m)

/-- The equal-spin pair `χ_E↓`. -/
noncomputable def chiEDn (k n m : ℤ) : ℂ :=
  -(1 / 2) * F.corr cDnᴴ cDn cDnᴴ cDn (F.ν n) (F.ν m) (-F.ω k - F.ν n - F.ν m)

/-- The annihilator of spin `σ` (`true` for `↑`). -/
def cs (σ : Bool) : Op := if σ then cUp else cDn

/-- The coupled longitudinal susceptibility `χ_σt(ν_n, ν_m, ω_k)` (connected). -/
noncomputable def chiL (σ t : Bool) (k n m : ℤ) : ℂ :=
  F.chiLehmann (cs σ) (cs t) (F.ν n) (F.ν m) (F.ω k)

/-- The propagator `G_σ(ν)`. -/
noncomputable def Gs (σ : Bool) (ν : ℝ) : ℂ := F.propagator (cs σ) ν

/-- The transverse bubble `b_T↑ = -βG↑(ν)G↓(ν + ω)`. -/
noncomputable def bTUp (k n : ℤ) : ℂ :=
  -(F.β : ℂ) * F.Gs true (F.ν n) * F.Gs false (F.ν n + F.ω k)

/-- The transverse bubble `b_T↓ = -βG↓(ν)G↑(ν + ω)`. -/
noncomputable def bTDn (k n : ℤ) : ℂ :=
  -(F.β : ℂ) * F.Gs false (F.ν n) * F.Gs true (F.ν n + F.ω k)

/-- The pair bubble `b_A↑ = βG↑(ν)G↓(-ν - ω)`. -/
noncomputable def bAUp (k n : ℤ) : ℂ :=
  (F.β : ℂ) * F.Gs true (F.ν n) * F.Gs false (-F.ν n - F.ω k)

/-- The pair bubble `b_A↓ = βG↓(ν)G↑(-ν - ω)`. -/
noncomputable def bADn (k n : ℤ) : ℂ :=
  (F.β : ℂ) * F.Gs false (F.ν n) * F.Gs true (-F.ν n - F.ω k)

/-- The equal-spin factor `(β/2)G_σ(ν)G_σ(-ν - ω)` of the bubble
`b_Eσ = (β/2)G_σ(ν)G_σ(-ν - ω)[δ_νν' - δ_ν(-ν'-ω)]`. -/
noncomputable def bE (σ : Bool) (k n : ℤ) : ℂ :=
  (F.β : ℂ) / 2 * F.Gs σ (F.ν n) * F.Gs σ (-F.ν n - F.ω k)

/-- The longitudinal bubble `b_σ = -βG_σ(ν)G_σ(ν + ω)`. -/
noncomputable def bL (σ : Bool) (k n : ℤ) : ℂ :=
  -(F.β : ℂ) * F.Gs σ (F.ν n) * F.Gs σ (F.ν n + F.ω k)

/-! ### Symmetries -/

theorem ν_add_ω (n k : ℤ) : F.ν n + F.ω k = F.ν (n + k) := (F.lat 0).ν_add_ω n k

theorem ω_pp (k n m : ℤ) : F.ω (-k - n - m - 1) = -F.ω k - F.ν n - F.ν m :=
  (F.lat 0).ω_pp k n m

/-- **General Shiba map**: `χ_A↑(U, μ, h) = χ_T↑(-U, h, μ)`, at the same frequencies. -/
theorem chiAUp_eq (k n m : ℤ) : F.chiAUp k n m = F.shiba.chiTUp k n m := by
  have h := F.corr_shiba cUpᴴ cDn cDnᴴ cUp (F.ν n) (F.ν m) (-F.ω k - F.ν n - F.ν m)
  rw [shibaV_cUpDag, shibaV_cDn, shibaV_cDnDag, shibaV_cUp,
    show -F.ω k - F.ν n - F.ν m = -(F.ω k) - F.ν n - F.ν m by ring, F.shiba.corr_swap] at h
  rw [chiAUp, chiTUp, ← h, neg_neg]
  rfl

/-- The spin flip: `χ_T↓(U, μ, h) = χ_T↑(U, μ, -h)`. -/
theorem chiTDn_eq (k n m : ℤ) : F.chiTDn k n m = F.flip.chiTUp k n m := by
  have h := F.corr_flip cDnᴴ cUp cUpᴴ cDn (F.ν n) (F.ν m) (F.ω k)
  rw [flipV_cDnDag, flipV_cUp, flipV_cUpDag, flipV_cDn] at h
  rw [chiTDn, chiTUp, ← h]
  rfl

/-- The spin flip of the pair: `χ_A↓(U, μ, h) = χ_A↑(U, μ, -h)`. -/
theorem chiADn_eq_flip (k n m : ℤ) : F.chiADn k n m = F.flip.chiAUp k n m := by
  have h := F.corr_flip cDnᴴ cUp cUpᴴ cDn (F.ν n) (F.ν m) (-F.ω k - F.ν n - F.ν m)
  rw [flipV_cDnDag, flipV_cUp, flipV_cUpDag, flipV_cDn] at h
  rw [chiADn, chiAUp, ← h]
  rfl

/-- `χ_A↓(U, μ, h) = χ_T↑(-U, -h, μ)`: spin interchange before the particle–hole map. -/
theorem chiADn_eq (k n m : ℤ) : F.chiADn k n m = F.flip.shiba.chiTUp k n m := by
  rw [chiADn_eq_flip, chiAUp_eq]

/-- The spin flip of the equal-spin pair: `χ_E↓(U, μ, h) = χ_E↑(U, μ, -h)`. -/
theorem chiEDn_eq (k n m : ℤ) : F.chiEDn k n m = F.flip.chiEUp k n m := by
  have h := F.corr_flip cDnᴴ cDn cDnᴴ cDn (F.ν n) (F.ν m) (-F.ω k - F.ν n - F.ν m)
  rw [flipV_cDnDag, flipV_cDn] at h
  rw [chiEDn, chiEUp, ← h]
  rfl

/-! ### The propagators -/

/-- `r↑ = ρ - m = 2⟨n↓⟩ - 1`. -/
noncomputable def rUp : ℝ := F.ρ - F.mf

/-- `r↓ = ρ + m = 2⟨n↑⟩ - 1`. -/
noncomputable def rDn : ℝ := F.ρ + F.mf

/-- The propagator in closed form, `G(y) = (y + ar)/(y² - a²)`. -/
noncomputable def Gc (r : ℝ) (y : ℂ) : ℂ := (y + F.a * r) / (y ^ 2 - F.a ^ 2)

/-- `G↑(ν_n) = (x + ar↑)/(x² - a²)` with `x = iν_n + μ + h`. -/
theorem propagator_up (n : ℤ) : F.Gs true (F.ν n) = F.Gc F.rUp ((F.lat (F.μ + F.h)).x n) := by
  set B := F.lat (F.μ + F.h)
  have hψ : cexp ((B.x n - (F.μ + F.h)) * F.β) = -1 := by
    have := B.cexp_x_sub_mu n
    simp only [B, lat_μ, lat_β] at this
    push_cast at this
    exact this
  have h : F.Gs true (F.ν n) = -∫ τ in (0 : ℝ)..F.β,
      cexp ((B.x n - (F.μ + F.h)) * τ) * F.thermal (F.heis τ cUp * cUpᴴ) := by
    rw [Gs, cs, ite_eq_left rfl, propagator]
    congr 1
    refine integral_congr_Ioo F.β_pos.le fun τ hτ ↦ ?_
    rw [F.timeOrdered_12 _ _ hτ.1, heis_zero,
      show (F.ν n : ℂ) * τ * I = (B.x n - (F.μ + F.h)) * τ by
        simp only [B, Params.x, lat_μ, lat_ν]
        push_cast
        ring]
  rw [h, lineTerm_eq]
  simp [Fin.sum_univ_four, cUp_eq]
  have h₁ := B.x_add_ofReal_ne_zero n F.a
  have h₂ := B.x_sub_ofReal_ne_zero n F.a
  rw [F.pathTerm₁_eq 2 3 hψ (r := B.x n - F.a) (by simp [rate, energy]; ring),
    F.pathTerm₁_eq 0 1 hψ (r := B.x n + F.a) (by simp [rate, energy]; ring)]
  have hw := F.weights_sum'
  simp only [Gc, rUp, ρ, mf, weight]
  have hx : B.x n ^ 2 - (F.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero h₁ h₂
  push_cast
  field_simp
  linear_combination B.x n * (B.x n ^ 2 - (F.a : ℂ) ^ 2) * hw

/-- `G↓(ν_n) = (x + ar↓)/(x² - a²)` with `x = iν_n + μ - h`. -/
theorem propagator_dn (n : ℤ) : F.Gs false (F.ν n) = F.Gc F.rDn ((F.lat (F.μ - F.h)).x n) := by
  set B := F.lat (F.μ - F.h)
  have hψ : cexp ((B.x n - (F.μ - F.h)) * F.β) = -1 := by
    have := B.cexp_x_sub_mu n
    simp only [B, lat_μ, lat_β] at this
    push_cast at this
    exact this
  have h : F.Gs false (F.ν n) = -∫ τ in (0 : ℝ)..F.β,
      cexp ((B.x n - (F.μ - F.h)) * τ) * F.thermal (F.heis τ cDn * cDnᴴ) := by
    rw [Gs, cs, ite_eq_right Bool.false_ne_true, propagator]
    congr 1
    refine integral_congr_Ioo F.β_pos.le fun τ hτ ↦ ?_
    rw [F.timeOrdered_12 _ _ hτ.1, heis_zero,
      show (F.ν n : ℂ) * τ * I = (B.x n - (F.μ - F.h)) * τ by
        simp only [B, Params.x, lat_μ, lat_ν]
        push_cast
        ring]
  rw [h, lineTerm_eq]
  simp [Fin.sum_univ_four, cDn_eq]
  have h₁ := B.x_add_ofReal_ne_zero n F.a
  have h₂ := B.x_sub_ofReal_ne_zero n F.a
  rw [F.pathTerm₁_eq 1 3 hψ (r := B.x n - F.a) (by simp [rate, energy]; ring),
    F.pathTerm₁_eq 0 2 hψ (r := B.x n + F.a) (by simp [rate, energy]; ring)]
  have hw := F.weights_sum'
  simp only [Gc, rDn, ρ, mf, weight]
  have hx : B.x n ^ 2 - (F.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero h₁ h₂
  push_cast
  field_simp
  linear_combination B.x n * (B.x n ^ 2 - (F.a : ℂ) ^ 2) * hw

/-! ### The transverse data -/

variable (k : ℤ)

/-- The complex transfer `W = iω_k - 2h`. -/
noncomputable def WT : ℂ := F.ω k * I - 2 * F.h

/-- `q₀ = a² - W²/4`. -/
noncomputable def q₀T : ℂ := F.a ^ 2 - F.WT k ^ 2 / 4

/-- `q₁ = q₀ - 2a²(p↑ + p↓) - aWm`. -/
noncomputable def q₁T : ℂ := F.q₀T k - 2 * F.a ^ 2 * (F.pUp + F.pDn) - F.a * F.WT k * F.mf

/-- `λ = -4a²`. -/
noncomputable def lamT : ℂ := -4 * F.a ^ 2

/-- `H(z) = z² + 2aρz + q₁`. -/
noncomputable def HT (z : ℂ) : ℂ := z ^ 2 + 2 * F.a * F.ρ * z + F.q₁T k

/-- `Q₊(z) = (z + a)² - W²/4`. -/
noncomputable def QpT (z : ℂ) : ℂ := (z + F.a) ^ 2 - F.WT k ^ 2 / 4

/-- `Q₋(z) = (z - a)² - W²/4`. -/
noncomputable def QmT (z : ℂ) : ℂ := (z - F.a) ^ 2 - F.WT k ^ 2 / 4

/-- `P = Q₊Q₋`. -/
noncomputable def PT (z : ℂ) : ℂ := F.QpT k z * F.QmT k z

/-- `G₂(z) = (z + aρ)² - (W/2 + am)²`. -/
noncomputable def G₂T (z : ℂ) : ℂ := (z + F.a * F.ρ) ^ 2 - (F.WT k / 2 + F.a * F.mf) ^ 2

/-- `e(z) = H(z)H(-z) - λ²p₀p₂`. -/
noncomputable def eT (z : ℂ) : ℂ := F.HT k z * F.HT k (-z) - F.lamT ^ 2 * F.p₀ * F.p₂

/-- `ζ = -4a²m/W`. -/
noncomputable def ζT : ℂ := -4 * F.a ^ 2 * F.mf / F.WT k

/-- `N₂(v) = 2a(v² + 2aρv + q₀) + ζ(v² - q₀)`. -/
noncomputable def N₂T (v : ℂ) : ℂ :=
  2 * F.a * (v ^ 2 + 2 * F.a * F.ρ * v + F.q₀T k) + F.ζT k * (v ^ 2 - F.q₀T k)

/-- `N₀(v) = (v² + 2q₀ - 4a²)N₂(v) - (2a + ζ)P(v)`. -/
noncomputable def N₀T (v : ℂ) : ℂ :=
  (v ^ 2 + 2 * F.q₀T k - 4 * F.a ^ 2) * F.N₂T k v - (2 * F.a + F.ζT k) * F.PT k v

/-- `N(z, v) = N₀(v) + z²N₂(v)`. -/
noncomputable def NT (z v : ℂ) : ℂ := F.N₀T k v + z ^ 2 * F.N₂T k v

/-- The transverse lattice `z_n = μ + i(ν_n + ω_k/2)`. -/
noncomputable def zT (n : ℤ) : ℂ := (F.lat F.μ).z k n

/-- The six Fock-space paths of `χ_T↑`: the conditions say whether `n = m` and whether the pair
rate `xₙ + xₘ + w` vanishes (at `μ = 0`, `m = n̄`). -/
theorem chiTUp_eq_pathsTD (hW0 : F.WT k ≠ 0) (n m : ℤ) :
    F.chiTUp k n m = pathsTD F.β F.a F.h ((F.lat F.μ).x n) ((F.lat F.μ).x m) ((F.lat F.μ).w k)
      F.p₀ F.pUp F.pDn F.p₂ (n = m) (F.μ = 0 ∧ n + m + k + 1 = 0) := by
  set B := F.lat F.μ with hB
  have h1 : cexp ((F.μ - B.x n) * F.β) = -1 := B.cexp_mu_sub_x n
  have h2 : cexp ((B.x n + B.w k - F.μ) * F.β) = -1 := by
    rw [B.x_add_w]
    exact B.cexp_x_sub_mu (n + k)
  have h3 : cexp ((F.μ - B.x m - B.w k) * F.β) = -1 := by
    rw [sub_sub, B.x_add_w]
    exact B.cexp_mu_sub_x (m + k)
  have hφ : ∀ τ₁ τ₂ τ₃ : ℝ,
      (((-F.ν n * τ₁ + (F.ν n + F.ω k) * τ₂ - (F.ν m + F.ω k) * τ₃ : ℝ) : ℂ) * I) =
        (F.μ - B.x n) * τ₁ + (B.x n + B.w k - F.μ) * τ₂ + (F.μ - B.x m - B.w k) * τ₃ :=
    B.freq_eq k n m
  rw [chiTUp, F.corr_eq _ _ _ _ _ _ _ hφ]
  simp only [orderedTerm_eq]
  simp [Fin.sum_univ_four, cUp_eq, cDn_eq]
  have hnm := B.x_sub_x_eq_zero_iff n m
  have hp : -B.x n - B.x m - B.w k = 0 ↔ (F.μ = 0 ∧ n + m + k + 1 = 0) := by
    have hpe := B.pair_eq_zero_iff k n m
    exact ⟨fun h0 ↦ hpe.mp (by linear_combination -h0),
      fun h0 ↦ by linear_combination -hpe.mpr h0⟩
  have hW : -2 * F.h + B.w k = 0 ↔ False := by
    rw [iff_false]
    intro h0
    exact hW0 (by rw [show F.WT k = B.w k - 2 * F.h from rfl]; linear_combination h0)
  rw [
    F.pathTerm_eq 1 0 2 0 h1 h2 h3 (r₁ := -B.x n - F.a - F.h)
      (r₂ := B.x n + F.a - F.h + B.w k) (r₃ := -B.x m - F.a + F.h - B.w k)
      (r₁₂ := -2 * F.h + B.w k) (r₂₃ := B.x n - B.x m) (r₁₂₃ := -B.x m - F.a - F.h)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by ring) (by ring) (by ring) hW hnm,
    F.pathTerm_eq 3 2 0 2 h1 h3 h2 (r₁ := -B.x n + F.a - F.h)
      (r₂ := -B.x m - F.a + F.h - B.w k) (r₃ := B.x n + F.a - F.h + B.w k)
      (r₁₂ := -B.x n - B.x m - B.w k) (r₂₃ := B.x n - B.x m) (r₁₂₃ := -B.x m + F.a - F.h)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by ring) (by ring) (by ring) hp hnm,
    F.pathTerm_eq 3 1 0 2 h3 h1 h2 (r₁ := -B.x m + F.a + F.h - B.w k)
      (r₂ := -B.x n - F.a - F.h) (r₃ := B.x n + F.a - F.h + B.w k)
      (r₁₂ := -B.x n - B.x m - B.w k) (r₂₃ := -2 * F.h + B.w k) (r₁₂₃ := -B.x m + F.a - F.h)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by ring) (by ring) (by ring) hp hW,
    F.pathTerm_eq 1 3 2 0 h2 h1 h3 (r₁ := B.x n - F.a - F.h + B.w k)
      (r₂ := -B.x n + F.a - F.h) (r₃ := -B.x m - F.a + F.h - B.w k)
      (r₁₂ := -2 * F.h + B.w k) (r₂₃ := -B.x n - B.x m - B.w k) (r₁₂₃ := -B.x m - F.a - F.h)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by ring) (by ring) (by ring) hW hp,
    F.pathTerm_eq 1 3 1 0 h2 h3 h1 (r₁ := B.x n - F.a - F.h + B.w k)
      (r₂ := -B.x m + F.a + F.h - B.w k) (r₃ := -B.x n - F.a - F.h)
      (r₁₂ := B.x n - B.x m) (r₂₃ := -B.x n - B.x m - B.w k) (r₁₂₃ := -B.x m - F.a - F.h)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by ring) (by ring) (by ring) hnm hp,
    F.pathTerm_eq 3 1 3 2 h3 h2 h1 (r₁ := -B.x m + F.a + F.h - B.w k)
      (r₂ := B.x n - F.a - F.h + B.w k) (r₃ := -B.x n + F.a - F.h)
      (r₁₂ := B.x n - B.x m) (r₂₃ := -2 * F.h + B.w k) (r₁₂₃ := -B.x m + F.a - F.h)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by ring) (by ring) (by ring) hnm hW
]
  simp only [weight, Matrix.cons_val, Matrix.cons_val_zero, Matrix.cons_val_one]
  rfl

/-- **Proposition "transverse input"**, from the Hamiltonian, off the pair resonance and for
`W ≠ 0`:
`P(zₙ)χ_T↑,nm = -βH(zₙ)δₙₘ + λρ/(zₙ + zₘ) + N(zₙ, zₘ)/P(zₘ)`. -/
theorem PT_mul_chiTUp' (hW0 : F.WT k ≠ 0) {n m : ℤ} (hD : ¬(F.μ = 0 ∧ n + m + k + 1 = 0)) :
    F.PT k (F.zT k n) * F.chiTUp k n m =
      -F.β * F.HT k (F.zT k n) * kron n m + F.lamT * F.ρ / (F.zT k n + F.zT k m) +
        F.NT k (F.zT k n) (F.zT k m) / F.PT k (F.zT k m) := by
  set B := F.lat F.μ with hB
  have hnm := B.x_sub_x_eq_zero_iff n m
  have hp : -B.x n - B.x m - B.w k = 0 ↔ False := by
    rw [iff_false]
    intro h0
    exact hD ((B.pair_eq_zero_iff k n m).mp (by linear_combination -h0))
  rw [F.chiTUp_eq_pathsTD k hW0, pathsTD_congr _ _ _ _ _ _ _ _ _ _ Iff.rfl
    (iff_false_intro hD), ← pathsT_eq_pathsTD]
  have hw := F.weights_sum'
  have hWT : F.WT k = B.w k - 2 * F.h := rfl
  have hWne : B.w k - 2 * (F.h : ℂ) ≠ 0 := hW0
  have hPz : ∀ j, F.PT k (F.zT k j) = PTf F.a F.h (B.x j) (B.w k) := by
    intro j
    simp only [PT, QpT, QmT, zT, Params.z, hWT, PTf]
    ring
  have hzz : F.zT k n + F.zT k m = B.x n + B.x m + B.w k := by
    simp only [zT, Params.z]
    ring
  -- the linear factors of `P` on the lattice
  have lx : ∀ (j : ℤ) (c : ℝ), B.x j + c ≠ 0 := fun j c ↦ B.x_add_ofReal_ne_zero j c
  have lxw : ∀ (j : ℤ) (c : ℝ), B.x j + B.w k + c ≠ 0 := fun j c ↦ by
    rw [B.x_add_w]
    exact B.x_add_ofReal_ne_zero _ c
  have f₁ : ∀ j, B.x j + F.a + F.h ≠ 0 := fun j ↦ by
    have := lx j (F.a + F.h); push_cast at this; rwa [← add_assoc] at this
  have f₂ : ∀ j, B.x j - F.a + F.h ≠ 0 := fun j ↦ by
    have := lx j (-F.a + F.h); push_cast at this
    rwa [show B.x j + (-(F.a : ℂ) + F.h) = B.x j - F.a + F.h by ring] at this
  have f₃ : ∀ j, B.x j + B.w k + F.a - F.h ≠ 0 := fun j ↦ by
    have := lxw j (F.a - F.h); push_cast at this; rwa [← add_sub_assoc] at this
  have f₄ : ∀ j, B.x j + B.w k - F.a - F.h ≠ 0 := fun j ↦ by
    have := lxw j (-F.a - F.h); push_cast at this
    rwa [show B.x j + B.w k + (-(F.a : ℂ) - F.h) = B.x j + B.w k - F.a - F.h by ring] at this
  have hpair : B.x n + B.x m + B.w k ≠ 0 := fun h0 ↦ hp.mp (by linear_combination -h0)
  rw [hPz, hzz]
  have hρ : (F.ρ : ℂ) = F.p₂ - F.p₀ := by rw [ρ]; push_cast; ring
  have hm : (F.mf : ℂ) = F.pUp - F.pDn := by rw [mf]; push_cast; ring
  by_cases hnm0 : n = m
  · subst hnm0
    have hX : B.x n + B.x n + B.w k ≠ 0 := hpair
    rw [kron_self, pathsT_of_true rfl, pathsT_diagonal hw (f₁ n) (f₂ n) (f₃ n) (f₄ n) hWne hX]
    simp only [HTf, rhsT, Nf, N₂f, HT, NT, N₀T, N₂T, ζT, q₀T, q₁T, lamT, hρ, hm, hPz]
    simp only [zT, Params.z, hWT]
    ring
  · rw [kron_of_ne hnm0]
    have hXY : B.x n - B.x m ≠ 0 := mt hnm.mp hnm0
    rw [pathsT_of_false hnm0,
      pathsT_offdiag hw (f₁ n) (f₂ n) (f₃ n) (f₄ n) (f₁ m) (f₂ m) (f₃ m) (f₄ m) hWne hpair hXY]
    simp only [rhsT, Nf, N₂f, NT, N₀T, N₂T, ζT, q₀T, lamT, hρ, hm, hPz]
    simp only [zT, Params.z, hWT]
    ring

/-- **Proposition "transverse input"** for `μ ≠ 0`, where no pair rate vanishes. -/
theorem PT_mul_chiTUp (hμ : F.μ ≠ 0) (hW0 : F.WT k ≠ 0) (n m : ℤ) :
    F.PT k (F.zT k n) * F.chiTUp k n m =
      -F.β * F.HT k (F.zT k n) * kron n m + F.lamT * F.ρ / (F.zT k n + F.zT k m) +
        F.NT k (F.zT k n) (F.zT k m) / F.PT k (F.zT k m) :=
  F.PT_mul_chiTUp' k hW0 fun h0 ↦ hμ h0.1

end FParams

end HubbardAtom
