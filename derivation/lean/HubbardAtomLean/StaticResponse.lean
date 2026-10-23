import HubbardAtomLean.DividedDifferences
import HubbardAtomLean.FieldTZero

/-!
# The static functions of the divided differences as responses of `G`

At `k = 0` the functions `f` of `DividedDifferences.lean` are the static responses of the
propagator from the Hamiltonian, `G↑(ν) = Gs true ν`, to the chemical potentials: with
`μ↑ = μ + h`, `μ↓ = μ - h`,

* `FParams.hasDerivAt_Gs_up_μUp`: `f↑↑ = -∂G↑/∂μ↑`;
* `FParams.hasDerivAt_Gs_up_μDn`: `f↑↓ = -∂G↑/∂μ↓`;
* `FParams.hasDerivAt_Gs_up_μ`, `hasDerivAt_Gs_up_h`: at `h = 0`, `f_d = -∂G/∂μ` and
  `f_m = -∂G↑/∂h` (`f_r` of the zero-field atom).

The derivatives are taken along the line `(μ + αt, h + γt)` of `FParams.line`, where
`G↑ = (x + ar↑)/(x² - a²)` with `x = iν + μ + h` and `r↑ = p₂ + p↓ - p₀ - p↑`
(`hasDerivAt_rUp`, `hasDerivAt_Gc_rUp`).
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams)

/-- The atom moved along a line in `(μ, h)`: `(μ + αt, h + γt)`. -/
def line (α γ t : ℝ) : FParams := ⟨F.U, F.β, F.μ + α * t, F.h + γ * t, F.β_pos⟩

theorem line_zero (α γ : ℝ) : F.line α γ 0 = F := by
  simp [line]

theorem line_ν (α γ t : ℝ) (n : ℤ) : (F.line α γ t).ν n = F.ν n := rfl

/-- `∂r↑/∂t = β[(αp₀ - γp↑ - γp↓ + αp₂) - r↑(-αp₀ + γp↑ - γp↓ + αp₂)]`. -/
noncomputable def rUp' (α γ : ℝ) : ℝ :=
  F.β * ((α * F.p₀ - γ * F.pUp - γ * F.pDn + α * F.p₂) -
    F.rUp * (-α * F.p₀ + γ * F.pUp - γ * F.pDn + α * F.p₂))

theorem hasDerivAt_rUp (α γ : ℝ) : HasDerivAt (fun t ↦ (F.line α γ t).rUp) (F.rUp' α γ) 0 := by
  have hE : ∀ (f : ℝ → ℝ) (c d : ℝ), (∀ t, f t = c + d * t) →
      HasDerivAt (fun t ↦ Real.exp (f t)) (Real.exp c * d) 0 := by
    intro f c d hf
    have := (((hasDerivAt_id (0 : ℝ)).const_mul d).const_add c).exp
    simp only [id, mul_zero, add_zero, mul_one] at this
    rwa [show (fun t ↦ Real.exp (f t)) = fun t ↦ Real.exp (c + d * t) from
      funext fun t ↦ by rw [hf]]
  have h₀ := hE (fun t ↦ -F.β * (F.U / 4 + (F.μ + α * t))) (-F.β * (F.U / 4 + F.μ)) (-F.β * α)
    fun t ↦ by ring
  have h₁ := hE (fun t ↦ -F.β * (-F.U / 4 - (F.h + γ * t))) (-F.β * (-F.U / 4 - F.h)) (F.β * γ)
    fun t ↦ by ring
  have h₂ := hE (fun t ↦ -F.β * (-F.U / 4 + (F.h + γ * t))) (-F.β * (-F.U / 4 + F.h))
    (-(F.β * γ)) fun t ↦ by ring
  have h₃ := hE (fun t ↦ -F.β * (F.U / 4 - (F.μ + α * t))) (-F.β * (F.U / 4 - F.μ)) (F.β * α)
    fun t ↦ by ring
  have hZ : F.Z ≠ 0 := F.Z_pos.ne'
  convert (((h₃.add h₂).sub h₀).sub h₁).div (((h₀.add h₁).add h₂).add h₃) (by
    simp only [Pi.add_apply, mul_zero, add_zero]
    positivity) using 1
  · funext t
    simp only [line, rUp, ρ, mf, p₀, pUp, pDn, p₂, Z, boltz₀, boltzUp, boltzDn, boltz₂,
      Pi.add_apply, Pi.sub_apply, Pi.div_apply]
    ring
  · simp only [Z, boltz₀, boltzUp, boltzDn, boltz₂] at hZ
    simp only [rUp', rUp, ρ, mf, p₀, pUp, pDn, p₂, Z, boltz₀, boltzUp, boltzDn, boltz₂, mul_zero,
      add_zero, Pi.add_apply, Pi.sub_apply]
    generalize Real.exp (-F.β * (F.U / 4 + F.μ)) = B₀ at *
    generalize Real.exp (-F.β * (-F.U / 4 - F.h)) = B₁ at *
    generalize Real.exp (-F.β * (-F.U / 4 + F.h)) = B₂ at *
    generalize Real.exp (-F.β * (F.U / 4 - F.μ)) = B₃ at *
    field_simp
    ring

/-- `∂G↑/∂t` along the line, with `G↑ = (x + ar↑)/(x² - a²)` and `x = y + (α + γ)t`. -/
noncomputable def GUp' (α γ : ℝ) (y : ℂ) : ℂ :=
  ((α + γ : ℝ) + F.a * F.rUp' α γ) / (y ^ 2 - F.a ^ 2) -
    (y + F.a * F.rUp) * (2 * y * (α + γ : ℝ)) / (y ^ 2 - F.a ^ 2) ^ 2

theorem hasDerivAt_Gc_rUp (α γ : ℝ) {y : ℂ} (hy : y ^ 2 - F.a ^ 2 ≠ 0) :
    HasDerivAt (fun t ↦ (F.line α γ t).Gc (F.line α γ t).rUp (y + ((α + γ) * t : ℝ)))
      (F.GUp' α γ y) 0 := by
  have hr := (F.hasDerivAt_rUp α γ).ofReal_comp
  have hx : HasDerivAt (fun t : ℝ ↦ y + (((α + γ) * t : ℝ) : ℂ)) ((α + γ : ℝ) : ℂ) 0 := by
    have := (((hasDerivAt_id (0 : ℝ)).const_mul (α + γ)).ofReal_comp).const_add y
    simpa using this
  convert (hx.add (hr.const_mul (F.a : ℂ))).div ((hx.pow 2).sub_const ((F.a : ℂ) ^ 2))
    (by simpa using hy) using 1
  · funext t
    simp only [Gc, line, a, Pi.add_apply, Pi.div_apply, Pi.pow_apply]
  · simp only [GUp', mul_zero, ofReal_zero, add_zero, Pi.pow_apply, Pi.add_apply, line_zero]
    field_simp
    ring

/-- `G↑(ν_n)` of the atom on the line, from the Hamiltonian, at `x = x↑ₙ + (α + γ)t`. -/
theorem hasDerivAt_Gs_up (α γ : ℝ) (n : ℤ) :
    HasDerivAt (fun t ↦ (F.line α γ t).Gs true (F.ν n)) (F.GUp' α γ (F.Bup.x n)) 0 := by
  convert F.hasDerivAt_Gc_rUp α γ (F.Bup.x_sq_sub_ne_zero n) using 1
  funext t
  rw [show F.ν n = (F.line α γ t).ν n from rfl, propagator_up]
  congr 1
  simp only [Params.x, lat_μ, lat_ν, line_ν]
  simp only [line]
  push_cast
  ring

theorem Bup_z_zero (n : ℤ) : F.Bup.z 0 n = F.Bup.x n := by simp [Params.z]

/-- **`f↑↑ = -∂G↑/∂μ↑`** at `k = 0`: the line `(μ + t/2, h + t/2)` moves `μ↑` alone. -/
theorem hasDerivAt_Gs_up_μUp (n : ℤ) :
    HasDerivAt (fun t ↦ (F.line (1 / 2) (1 / 2) t).Gs true (F.ν n))
      (-F.fL 0 F.rUp (F.Bup.z 0 n)) 0 := by
  convert F.hasDerivAt_Gs_up (1 / 2) (1 / 2) n using 1
  have hy : F.Bup.x n ^ 2 - F.a ^ 2 ≠ 0 := F.Bup.x_sq_sub_ne_zero n
  have hs := F.weights_sum'
  rw [F.Bup_z_zero, fL, GUp', rUp', kron_self, ζL, show F.wL 0 = 0 from (F.Bup.w_eq_zero_iff 0).mpr
    rfl, rUp, ρ, mf]
  push_cast
  rw [show (F.p₂ : ℂ) = 1 - F.p₀ - F.pUp - F.pDn by linear_combination hs]
  generalize F.Bup.x n = y at *
  generalize (F.a : ℂ) = a at *
  rw [sq_sub_sq] at hy ⊢
  obtain ⟨h₁, h₂⟩ := mul_ne_zero_iff.mp hy
  by_cases ha : a = 0
  · subst ha
    simp
    field_simp
    ring
  field_simp
  ring

/-- **`f↑↓ = -∂G↑/∂μ↓`** at `k = 0`: the line `(μ + t/2, h - t/2)` moves `μ↓` alone. -/
theorem hasDerivAt_Gs_up_μDn (n : ℤ) :
    HasDerivAt (fun t ↦ (F.line (1 / 2) (-(1 / 2)) t).Gs true (F.ν n))
      (-F.fLUpDn 0 (F.Bup.z 0 n)) 0 := by
  convert F.hasDerivAt_Gs_up (1 / 2) (-(1 / 2)) n using 1
  have hy : F.Bup.x n ^ 2 - F.a ^ 2 ≠ 0 := F.Bup.x_sq_sub_ne_zero n
  have hs := F.weights_sum'
  rw [F.Bup_z_zero, fLUpDn, GUp', rUp', kron_self, ζUp, rUp, ρ, mf]
  push_cast
  rw [show (F.p₂ : ℂ) = 1 - F.p₀ - F.pUp - F.pDn by linear_combination hs]
  generalize F.Bup.x n = y at *
  generalize (F.a : ℂ) = a at *
  rw [sq_sub_sq] at hy ⊢
  obtain ⟨h₁, h₂⟩ := mul_ne_zero_iff.mp hy
  by_cases ha : a = 0
  · subst ha
    simp
  field_simp
  ring

/-- The value of `∂G↑/∂t` at `h = 0` along `μ` (`ε = 1`) or `h` (`ε = -1`): `-f_r`. -/
private theorem GUp'_static (hh : F.h = 0) {α γ : ℝ} {ε : ℂ}
    (h : (α = 1 ∧ γ = 0 ∧ ε = 1) ∨ (α = 0 ∧ γ = 1 ∧ ε = -1)) (n : ℤ) :
    F.GUp' α γ (F.Bup.x n) = -(F.lat F.μ).fR 0 ε ((F.lat F.μ).z 0 n) := by
  have hy : F.Bup.x n ^ 2 - F.a ^ 2 ≠ 0 := F.Bup.x_sq_sub_ne_zero n
  have hs := F.weights_sum'
  have hx : (F.lat F.μ).z 0 n = F.Bup.x n := by
    rw [Params.z, ((F.lat F.μ).w_eq_zero_iff 0).mpr rfl, zero_div, add_zero, Bup, hh, add_zero]
  have hd : (F.pDn : ℂ) = F.pUp := by rw [← F.lat_p₁_dn hh, ← F.lat_p₁_up hh]
  rw [hx, Params.fR, GUp', rUp', kron_self, Params.ζ, Params.Qp, Params.Qm, Params.qm, Params.qp,
    Params.ρ, F.lat_p₀ hh, F.lat_p₂ hh, F.lat_p₁_up hh, ((F.lat F.μ).w_eq_zero_iff 0).mpr rfl,
    rUp, ρ, mf]
  push_cast
  simp only [lat_a, lat_β, hd] at hs ⊢
  rw [show (F.p₂ : ℂ) = 1 - F.p₀ - 2 * F.pUp by linear_combination hs]
  generalize F.Bup.x n = y at *
  generalize (F.a : ℂ) = a at *
  rw [sq_sub_sq] at hy ⊢
  obtain ⟨h₁, h₂⟩ := mul_ne_zero_iff.mp hy
  rcases h with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;>
  · simp only [ofReal_one, ofReal_zero]
    by_cases ha : a = 0
    · subst ha
      simp only [add_zero] at h₁
      simp
      field_simp
      ring
    · field_simp
      ring

/-- **`f_d = -∂G/∂μ`** at `k = 0`, `h = 0` (`f_r` of the zero-field atom, `r = d`). -/
theorem hasDerivAt_Gs_up_μ (hh : F.h = 0) (n : ℤ) :
    HasDerivAt (fun t ↦ (F.line 1 0 t).Gs true (F.ν n))
      (-(F.lat F.μ).fR 0 1 ((F.lat F.μ).z 0 n)) 0 :=
  F.GUp'_static hh (Or.inl ⟨rfl, rfl, rfl⟩) n ▸ F.hasDerivAt_Gs_up 1 0 n

/-- **`f_m = -∂G↑/∂h`** at `k = 0`, `h = 0` (`f_r` of the zero-field atom, `r = m`). -/
theorem hasDerivAt_Gs_up_h (hh : F.h = 0) (n : ℤ) :
    HasDerivAt (fun t ↦ (F.line 0 1 t).Gs true (F.ν n))
      (-(F.lat F.μ).fR 0 (-1) ((F.lat F.μ).z 0 n)) 0 :=
  F.GUp'_static hh (Or.inr ⟨rfl, rfl, rfl⟩) n ▸ F.hasDerivAt_Gs_up 0 1 n

end FParams

end HubbardAtom
