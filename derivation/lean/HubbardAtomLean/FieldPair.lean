import HubbardAtomLean.FieldTransverse

/-!
# Orientations and opposite-spin pairing

The other orientation and the opposite-spin pair, Eq. (field-A) of the note (App. F.6.2): the
general Shiba map `c↓ ↦ c↓†` sends the atom `(U, μ, h)` to `(-U, h, μ)` and the oriented
opposite-spin pair to the transverse sector, `χ_A↑(U, μ, h) = χ_T↑(-U, h, μ)` (`chiAUp_eq`); the
spin flip sends `↑` to `↓` and `h` to `-h`. Both maps also relate the physical bubbles, so the
transverse theorem gives the vertices of all four oriented sectors.

* `FParams.bAUp_eq`, `bTDn_eq`, `bADn_eq`: the bubbles under the two maps.
* `FParams.gammaAUp_eq`, `gammaTDn_eq`, `gammaADn_eq`: the vertices `β²(χ⁻¹ - b⁻¹)` of `A↑`, `T↓`,
  `A↓`, with two-sided inverses, under `TGeneric` of the image parameters:
  `Γ_A↑(U, μ, h) = Γ_T↑(-U, h, μ)`, `Γ_T↓(U, μ, h) = Γ_T↑(U, μ, -h)`,
  `Γ_A↓(U, μ, h) = Γ_T↑(-U, -h, μ)` (spin interchange before the particle–hole map).
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

theorem shiba_ν (n : ℤ) : F.shiba.ν n = F.ν n := rfl

theorem flip_ν (n : ℤ) : F.flip.ν n = F.ν n := rfl

theorem shiba_ω : F.shiba.ω k = F.ω k := rfl

theorem flip_ω : F.flip.ω k = F.ω k := rfl

theorem shiba_a : F.shiba.a = -F.a := by
  rw [a, a, shiba_U]
  ring

theorem shiba_ρ : F.shiba.ρ = F.mf := by rw [ρ, mf, shiba_p₂, shiba_p₀]

theorem shiba_mf : F.shiba.mf = F.ρ := by rw [ρ, mf, shiba_pUp, shiba_pDn]

theorem flip_ρ : F.flip.ρ = F.ρ := by rw [ρ, ρ, flip_p₂, flip_p₀]

theorem flip_mf : F.flip.mf = -F.mf := by
  rw [mf, mf, flip_pUp, flip_pDn]
  ring

/-- The lattice point `x = c + iν_n` as a complex number. -/
theorem lat_x (c : ℝ) (n : ℤ) : (F.lat c).x n = c + F.ν n * I := rfl

/-- The spin flip exchanges the propagators at the fermionic frequencies. -/
theorem flip_Gs_true (n : ℤ) : F.flip.Gs true (F.ν n) = F.Gs false (F.ν n) := by
  have h₁ := F.flip.propagator_up n
  rw [flip_ν] at h₁
  rw [h₁, F.propagator_dn]
  simp only [Gc, rUp, rDn, flip_ρ, flip_mf, lat_x, flip_μ, flip_h, flip_ν]
  rw [show F.flip.a = F.a from rfl]
  push_cast
  ring

theorem flip_Gs_false (n : ℤ) : F.flip.Gs false (F.ν n) = F.Gs true (F.ν n) := by
  have h₁ := F.flip.propagator_dn n
  rw [flip_ν] at h₁
  rw [h₁, F.propagator_up]
  simp only [Gc, rUp, rDn, flip_ρ, flip_mf, lat_x, flip_μ, flip_h, flip_ν]
  rw [show F.flip.a = F.a from rfl]
  push_cast
  ring

/-- The bubble of `T↓` is that of `T↑` with `h ↦ -h`. -/
theorem bTDn_eq (n : ℤ) : F.bTDn k n = F.flip.bTUp k n := by
  rw [bTDn, bTUp, flip_ω, show F.flip.ν n = F.ν n from rfl, F.ν_add_ω, flip_Gs_true,
    flip_Gs_false]
  rfl

/-- The bubble of `A↑` is that of `T↑` at `(-U, h, μ)`: `βG↑(ν)G↓(-ν - ω) = -βG'↑(ν)G'↓(ν + ω)`. -/
theorem bAUp_eq (n : ℤ) : F.bAUp k n = F.shiba.bTUp k n := by
  have hm : -F.ν n - F.ω k = F.ν (mirror k n) := ((F.lat 0).ν_mirror k n).symm
  have hs₁ := F.shiba.propagator_up n
  have hs₂ := F.shiba.propagator_dn (n + k)
  rw [shiba_ν] at hs₁ hs₂
  rw [bAUp, bTUp, hm, shiba_ω, shiba_ν, F.ν_add_ω, hs₁, hs₂, F.propagator_up, F.propagator_dn]
  simp only [Gc, rUp, rDn, shiba_ρ, shiba_mf, shiba_a, lat_x, shiba_μ, shiba_h, shiba_β, shiba_ν]
  have hν : F.ν (mirror k n) = -F.ν (n + k) := by
    rw [← hm, ← F.ν_add_ω]
    ring
  rw [hν]
  push_cast
  ring

/-- The bubble of `A↓` is that of `A↑` with `h ↦ -h`. -/
theorem bADn_eq (n : ℤ) : F.bADn k n = F.flip.shiba.bTUp k n := by
  rw [← F.flip.bAUp_eq, bADn, bAUp, flip_ω, show F.flip.ν n = F.ν n from rfl]
  have hm : -F.ν n - F.ω k = F.ν (mirror k n) := ((F.lat 0).ν_mirror k n).symm
  rw [hm, flip_Gs_true, flip_Gs_false]
  rfl

/-! ### The vertices -/

variable {F k} {u₁ u₂ : ℂ}

/-- **The opposite-spin pair `A↑`**: under the assumptions of the transverse vertex at
`(-U, h, μ)`, its inverse is a two-sided inverse of `χ_A↑`, and
`β²(χ_A↑⁻¹ - b_A↑⁻¹) = Γ_T↑(-U, h, μ)`. -/
theorem gammaAUp_eq (h : F.shiba.TGeneric k u₁ u₂) :
    IsRightInverse (F.chiAUp k) (F.shiba.invTUp k u₁ u₂) ∧
      IsLeftInverse (F.chiAUp k) (F.shiba.invTUp k u₁ u₂) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * (F.shiba.invTUp k u₁ u₂ n m - kron n m / F.bAUp k n) =
        F.shiba.gammaTUp k u₁ u₂ n m := by
  have hχ : F.chiAUp k = F.shiba.chiTUp k := funext fun n ↦ funext fun m ↦ F.chiAUp_eq k n m
  refine ⟨hχ ▸ h.chiTUp_isRightInverse, hχ ▸ h.chiTUp_isLeftInverse, fun n m ↦ ?_⟩
  rw [F.bAUp_eq k n]
  exact h.gammaTUp_eq n m

/-- **The transverse sector `T↓`**: `Γ_T↓(U, μ, h) = Γ_T↑(U, μ, -h)`. -/
theorem gammaTDn_eq (h : F.flip.TGeneric k u₁ u₂) :
    IsRightInverse (F.chiTDn k) (F.flip.invTUp k u₁ u₂) ∧
      IsLeftInverse (F.chiTDn k) (F.flip.invTUp k u₁ u₂) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * (F.flip.invTUp k u₁ u₂ n m - kron n m / F.bTDn k n) =
        F.flip.gammaTUp k u₁ u₂ n m := by
  have hχ : F.chiTDn k = F.flip.chiTUp k := funext fun n ↦ funext fun m ↦ F.chiTDn_eq k n m
  refine ⟨hχ ▸ h.chiTUp_isRightInverse, hχ ▸ h.chiTUp_isLeftInverse, fun n m ↦ ?_⟩
  rw [F.bTDn_eq k n]
  exact h.gammaTUp_eq n m

/-- **The opposite-spin pair `A↓`**: `Γ_A↓(U, μ, h) = Γ_T↑(-U, -h, μ)`, the spin interchange
applied before the particle–hole map. -/
theorem gammaADn_eq (h : F.flip.shiba.TGeneric k u₁ u₂) :
    IsRightInverse (F.chiADn k) (F.flip.shiba.invTUp k u₁ u₂) ∧
      IsLeftInverse (F.chiADn k) (F.flip.shiba.invTUp k u₁ u₂) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * (F.flip.shiba.invTUp k u₁ u₂ n m - kron n m / F.bADn k n) =
        F.flip.shiba.gammaTUp k u₁ u₂ n m := by
  have hχ : F.chiADn k = F.flip.shiba.chiTUp k :=
    funext fun n ↦ funext fun m ↦ F.chiADn_eq k n m
  refine ⟨hχ ▸ h.chiTUp_isRightInverse, hχ ▸ h.chiTUp_isLeftInverse, fun n m ↦ ?_⟩
  rw [F.bADn_eq k n]
  exact h.gammaTUp_eq n m

end FParams

end HubbardAtom
