import HubbardAtomLean.DividedDifferences
import HubbardAtomLean.FieldLMu0
import HubbardAtomLean.FieldEqualSpin

/-!
# Divided differences of the opposite-spin longitudinal blocks on the axes

`DividedDifferences.lean` proves `(z↑ₙ² - z↓ₘ²)χ↑↓,nm = U[f↑↑(z↑ₙ) - f↓↓(z↓ₘ)]` off the axes
`μ = 0`, `h = 0`, where the two spin lattices neither meet nor mirror each other. On the axes they
do: at `h = 0`, `z↑ₙ = z↓ₙ`, and at `μ = 0`, `z↑ₙ = -z↓_{n̄}`. The identity holds there too, and
at these degenerate pairs both sides vanish:

* `FParams.fL_of_h`: at `h = 0`, `f↑↑ = f↓↓ = (f_d + f_m)/2` of the zero-field atom;
* `FParams.chiL_updn_divided_difference_of_h`: at `h = 0`, from `χ↑↓ = (χ_d - χ_m)/2` of the
  zero-field atom (`chiL_opp_of_h`) and `chiR_divided_difference`; on the diagonal both sides
  vanish;
* `FParams.shiba_fL`, `fL_neg`: `f^shiba(r, z) = f(-r, z)` and `f(r, -z) = f(-r, z)`;
* `FParams.chiL_updn_divided_difference_of_mu`: at `μ = 0`, through the Shiba map
  `χ↑↓,nm = -χ^shiba↑↓,n m̄` (`chiL_shiba`) onto `h = 0`;
* `FParams.chiL_updn_divided_difference'`, `chiL_dnup_divided_difference'`: the identity for all
  `μ`, `h`, `n`, `m`; `fL_updn_eq_of_sq_eq`: at a degenerate pair `z↑ₙ² = z↓ₘ²` (`U ≠ 0`),
  `f↑↑(z↑ₙ) = f↓↓(z↓ₘ)`.

The other sectors need nothing new: `chiR_divided_difference` and
`chiTUp_divided_difference` hold for all `n ≠ m`, the mirror pairs at `μ = 0` included (`fR_neg`,
`fT_neg`), `chiS_divided_difference` on the mirror (`f_s(n̄) = f_s(n)`), and
`chiL_upup_divided_difference` for all `n ≠ m` (`f↑↓` is even).

The pair sectors of `FieldPair.lean`, `FieldEqualSpin.lean`:

* `FParams.chiAUp_divided_difference`, `chiADn_divided_difference`: the opposite-spin pairs,
  through the Shiba map, `χ_A↑(U, μ, h) = χ_T↑(-U, h, μ)` and `χ_A↓(U, μ, h) = χ_T↑(-U, -h, μ)`,
  for `iω_k - 2μ ≠ 0` and `n ≠ m`; `shiba_fT_zT`: `f_A↑(z'ₙ) = [G↑(νₙ) + G↓(-νₙ - ω_k)]/(iω_k - 2μ)`;
* `FParams.chiEUp_divided_difference`, `chiEDn_divided_difference`: the equal-spin pairs,
  `(tₙ - tₘ)χ_E,nm = 0`, where both sides vanish.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-- At `h = 0`, `f↑↑ = f↓↓ = (f_d + f_m)/2` of the zero-field atom. -/
theorem fL_of_h (hh : F.h = 0) {r : ℝ} (hr : r = F.rUp ∨ r = F.rDn) (z : ℂ) :
    F.fL k r z = ((F.lat F.μ).fR k 1 z + (F.lat F.μ).fR k (-1) z) / 2 := by
  have hr' : r = (F.lat F.μ).ρ := by
    rcases hr with rfl | rfl <;> simp [rUp, rDn, F.mf_of_h hh, F.lat_ρ hh]
  have hs := (F.lat F.μ).weights_sum'
  rw [hr', fL, Params.fR, Params.fR, Params.Qp, Params.Qm, ζL, Params.ζ, Params.ζ, Params.qm,
    Params.qp, Params.ρ, ← F.lat_p₀ hh, ← F.lat_p₂ hh, ← F.lat_p₁_up hh, ← F.lat_p₁_dn hh]
  simp only [lat_a, lat_β]
  rw [show F.wL k = (F.lat F.μ).w k from rfl]
  push_cast
  rw [show ((F.lat F.μ).p₂ : ℂ) = 1 - (F.lat F.μ).p₀ - 2 * (F.lat F.μ).p₁ by
    linear_combination hs]
  ring

/-- **Opposite spins at `h = 0`**: `(z↑ₙ² - z↓ₘ²)χ↑↓,nm = U[f↑↑(z↑ₙ) - f↓↓(z↓ₘ)]` for all `n`, `m`;
on the diagonal, where `z↑ₙ = z↓ₙ`, both sides vanish. -/
theorem chiL_updn_divided_difference_of_h (hh : F.h = 0) (n m : ℤ) :
    (F.Bup.z k n ^ 2 - F.Bdn.z k m ^ 2) * F.chiL true false k n m =
      F.U * (F.fL k F.rUp (F.Bup.z k n) - F.fL k F.rDn (F.Bdn.z k m)) := by
  have hup : F.Bup = F.lat F.μ := by rw [Bup, hh, add_zero]
  have hdn : F.Bdn = F.lat F.μ := by rw [Bdn, hh, sub_zero]
  have hχ := F.chiL_opp_of_h k hh true n m
  rw [Bool.not_true] at hχ
  rw [hup, hdn, F.fL_of_h k hh (Or.inl rfl), F.fL_of_h k hh (Or.inr rfl), hχ]
  by_cases hnm : n = m
  · subst hnm
    ring
  · have hd := (F.lat F.μ).chiR_divided_difference k 1 (Or.inl rfl) hnm
    have hm := (F.lat F.μ).chiR_divided_difference k (-1) (Or.inr rfl) hnm
    simp only [Params.chiR, lat_U] at hd hm
    linear_combination (hd - hm) / 2

theorem shiba_rUp : F.shiba.rUp = -F.rUp := by
  rw [rUp, rUp, shiba_ρ, shiba_mf]
  ring

theorem shiba_rDn : F.shiba.rDn = F.rDn := by
  rw [rDn, rDn, shiba_ρ, shiba_mf, add_comm]

/-- Under the Shiba map, `a ↦ -a` and `ζ_L ↦ -ζ_L`, so `f^shiba(r, z) = f(-r, z)`. -/
theorem shiba_fL (r : ℝ) (z : ℂ) : F.shiba.fL k r z = F.fL k (-r) z := by
  have ha : (F.shiba.a : ℂ) = -F.a := by rw [shiba_a]; push_cast; ring
  have hζ : F.shiba.ζL = -F.ζL := by
    rw [ζL, ζL, ha, shiba_p₀, shiba_p₂, shiba_pUp, shiba_pDn, shiba_β]
    ring
  rw [fL, fL, ha, hζ, show F.shiba.wL k = F.wL k from rfl,
    show 2 * -(F.a : ℂ) * (z ^ 2 - (-F.a) ^ 2) = -(2 * F.a * (z ^ 2 - F.a ^ 2)) by ring, div_neg]
  push_cast
  ring

/-- `f(r, -z) = f(-r, z)`. -/
theorem fL_neg (r : ℝ) (z : ℂ) : F.fL k r (-z) = F.fL k (-r) z := by
  rw [fL, fL, neg_sq, show (-z + F.a) ^ 2 = (z - F.a) ^ 2 by ring,
    show (-z - F.a) ^ 2 = (z + F.a) ^ 2 by ring]
  push_cast
  ring

/-- **Opposite spins at `μ = 0`**: `(z↑ₙ² - z↓ₘ²)χ↑↓,nm = U[f↑↑(z↑ₙ) - f↓↓(z↓ₘ)]` for all `n`, `m`,
through the Shiba map onto `h = 0` (`z↓ₘ = -z^shiba↓_m̄`); on the pairs `z↑ₙ = -z↓ₘ` both sides
vanish. -/
theorem chiL_updn_divided_difference_of_mu (hμ : F.μ = 0) (n m : ℤ) :
    (F.Bup.z k n ^ 2 - F.Bdn.z k m ^ 2) * F.chiL true false k n m =
      F.U * (F.fL k F.rUp (F.Bup.z k n) - F.fL k F.rDn (F.Bdn.z k m)) := by
  have h := F.shiba.chiL_updn_divided_difference_of_h k (by rw [shiba_h, hμ]) n (RL k false m)
  have hup : F.shiba.Bup.z k n = F.Bup.z k n := by
    simp only [Params.z, Params.x, Params.w, lat_μ, lat_ν, lat_ω, shiba_μ, shiba_h, shiba_ν,
      shiba_ω, hμ, zero_add]
    push_cast
    ring
  have hdn : F.shiba.Bdn.z k (RL k false m) = -F.Bdn.z k m := by
    simp only [RL, Bool.false_eq_true, ↓reduceIte, Params.z, Params.x, Params.w, lat_μ, lat_ν,
      lat_ω, shiba_μ, shiba_h, shiba_ν, shiba_ω, hμ, F.ν_bar]
    push_cast
    ring
  rw [hup, hdn, neg_sq, shiba_U, shiba_fL, shiba_fL, shiba_rUp, shiba_rDn, fL_neg] at h
  rw [F.chiL_shiba k true false n m]
  simp only [εL, RL, ↓reduceIte, Bool.false_eq_true, neg_neg] at h ⊢
  push_cast at h
  linear_combination -h

/-- **Opposite spins, all `μ`, `h`**: `(z↑ₙ² - z↓ₘ²)χ↑↓,nm = U[f↑↑(z↑ₙ) - f↓↓(z↓ₘ)]` for all `n`,
`m`. -/
theorem chiL_updn_divided_difference' (n m : ℤ) :
    (F.Bup.z k n ^ 2 - F.Bdn.z k m ^ 2) * F.chiL true false k n m =
      F.U * (F.fL k F.rUp (F.Bup.z k n) - F.fL k F.rDn (F.Bdn.z k m)) := by
  by_cases hμ : F.μ = 0
  · exact F.chiL_updn_divided_difference_of_mu k hμ n m
  by_cases hh : F.h = 0
  · exact F.chiL_updn_divided_difference_of_h k hh n m
  exact F.chiL_updn_divided_difference k hμ hh n m

/-- **Opposite spins, rows `↓`, all `μ`, `h`**, by the spin flip. -/
theorem chiL_dnup_divided_difference' (n m : ℤ) :
    (F.Bdn.z k n ^ 2 - F.Bup.z k m ^ 2) * F.chiL false true k n m =
      F.U * (F.fL k F.rDn (F.Bdn.z k n) - F.fL k F.rUp (F.Bup.z k m)) := by
  rw [F.chiL_flip]
  simpa [flip_Bup, flip_Bdn, flip_fL, flip_rUp, flip_rDn] using
    F.flip.chiL_updn_divided_difference' k n m

/-- **Degenerate pairs**: for `U ≠ 0`, where `z↑ₙ² = z↓ₘ²` (`n = m` at `h = 0`, `z↑ₙ = -z↓ₘ` at
`μ = 0`), `f↑↑(z↑ₙ) = f↓↓(z↓ₘ)`. -/
theorem fL_updn_eq_of_sq_eq (hU : F.U ≠ 0) {n m : ℤ} (h : F.Bup.z k n ^ 2 = F.Bdn.z k m ^ 2) :
    F.fL k F.rUp (F.Bup.z k n) = F.fL k F.rDn (F.Bdn.z k m) := by
  have hd := F.chiL_updn_divided_difference' k n m
  rw [h, sub_self, zero_mul, eq_comm, mul_eq_zero, sub_eq_zero] at hd
  exact hd.resolve_left (ofReal_ne_zero.mpr hU)

/-! ### The opposite-spin and equal-spin pairs -/

/-- **Divided differences, opposite-spin pair `A↑`**: for `W' = iω_k - 2μ ≠ 0` and `n ≠ m`, on the
lattice `z'ₙ = h + i(νₙ + ω_k/2)` of the Shiba image `(-U, h, μ)`,
`(z'ₙ² - z'ₘ²)χ_A↑,nm = U[f_A(z'ₙ) - f_A(z'ₘ)]` with `f_A = f_T` of the image (`chiAUp_eq`); on
the mirror at `h = 0` both sides vanish (`fT_neg` of the image). -/
theorem chiAUp_divided_difference (hW : F.shiba.WT k ≠ 0) {n m : ℤ} (hnm : n ≠ m) :
    (F.shiba.zT k n ^ 2 - F.shiba.zT k m ^ 2) * F.chiAUp k n m =
      F.U * (F.shiba.fT k (F.shiba.zT k n) - F.shiba.fT k (F.shiba.zT k m)) := by
  rw [F.chiAUp_eq, F.shiba.chiTUp_divided_difference k hW hnm, shiba_U]
  push_cast
  ring

/-- `f_A(z'ₙ) = [G↑(νₙ) + G↓(-νₙ - ω_k)]/(iω_k - 2μ)`: the Shiba image has `G'↑ = G↑` and
`G'↓(ν) = -G↓(-ν)`. -/
theorem shiba_fT_zT (n : ℤ) :
    F.shiba.fT k (F.shiba.zT k n) =
      (F.Gs true (F.ν n) + F.Gs false (F.ν (mirror k n))) / F.shiba.WT k := by
  have hm : -F.ν n - F.ω k = F.ν (mirror k n) := ((F.lat 0).ν_mirror k n).symm
  have hν : F.ν (mirror k n) = -F.ν (n + k) := by
    rw [← hm, ← F.ν_add_ω]
    ring
  rw [F.shiba.fT_zT, F.shiba.propagator_up, F.shiba.propagator_dn, F.propagator_up,
    F.propagator_dn]
  simp only [Gc, shiba_rUp, shiba_rDn, shiba_a, lat_x, shiba_μ, shiba_h, shiba_ν]
  rw [hν]
  push_cast
  ring

/-- **Divided differences, opposite-spin pair `A↓`**: `χ_A↓(U, μ, h) = χ_T↑(-U, -h, μ)`
(`chiADn_eq`), so for `iω_k - 2μ ≠ 0` and `n ≠ m` the identity of `T↑` at `(-U, -h, μ)`. -/
theorem chiADn_divided_difference (hW : F.flip.shiba.WT k ≠ 0) {n m : ℤ} (hnm : n ≠ m) :
    (F.flip.shiba.zT k n ^ 2 - F.flip.shiba.zT k m ^ 2) * F.chiADn k n m =
      F.U * (F.flip.shiba.fT k (F.flip.shiba.zT k n) - F.flip.shiba.fT k (F.flip.shiba.zT k m)) := by
  rw [F.chiADn_eq, F.flip.shiba.chiTUp_divided_difference k hW hnm, shiba_U, flip_U]
  push_cast
  ring

/-- **Divided differences, equal-spin pair `E↑`**: `(tₙ - tₘ)χ_E↑,nm = 0` for all `n`, `m`
(`tₙ = xₙx_n̄` on the lattice of `↑`), the divided difference of `f = 0`: `χ_E↑` lives on the
diagonal and the mirror (`chiEUp_eq`), where `tₙ = tₘ`. -/
theorem chiEUp_divided_difference (n m : ℤ) :
    (F.Bup.t k n - F.Bup.t k m) * F.chiEUp k n m = 0 := by
  rw [F.chiEUp_eq, mirrorBlock]
  by_cases h₁ : n = m
  · subst h₁
    ring
  by_cases h₂ : m = mirror k n
  · subst h₂
    rw [F.Bup.t_mirror]
    ring
  · rw [kron_of_ne h₁, kron_of_ne h₂]
    ring

/-- **Divided differences, equal-spin pair `E↓`**, by the spin flip (`chiEDn_eq`). -/
theorem chiEDn_divided_difference (n m : ℤ) :
    (F.Bdn.t k n - F.Bdn.t k m) * F.chiEDn k n m = 0 := by
  rw [F.chiEDn_eq]
  exact F.flip.chiEUp_divided_difference k n m

end FParams

end HubbardAtom
