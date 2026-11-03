import HubbardAtomLean.FieldLSums

/-!
# The longitudinal input on the axes, both rows

`FParams.PL_mul_chiL_updn_all` of `FieldLInput.lean` gives the opposite-spin input of the row `↑`
for all `μ`, `h`, with the pair kernel `kerPL` and the spin-flip kernel `kerSL`. Where their rates
vanish, at `μ = 0`, `ν' = -ν - ω` and at `h = 0`, `ν' = ν`, they take the values `βp₀` and `βp↓`
of the limits of Eq. (field-kernel-limits). This file gives both rows, by the spin flip.

* `FParams.kerSLs`: the spin-flip kernel `σm/(z_σₙ - z_{-σ}ₘ)` of the row `σ`, and `βp_{-σ}` at
  `h = 0`, `n = m`. The note writes the limit as `m/(z - v) → σβp↓`; with `σm` in place of `m` it
  is `βp↓ = βp_{-σ}`, since `p↑ = p↓` at `h = 0`.
* `FParams.PL_mul_chiL_opp_all`: for all `μ`, `h`,
  `P(z)χ_{σ,-σ}(z, v) = λ[kerPL - kerSLs_σ] + (N₀_{-σ}(v) + z²N₂_{-σ}(v))/P(v)`. The pair kernel is
  the same in both rows, as `z_σₙ + z_{-σ}ₘ = 2μ + i(νₙ + νₘ + ω_k)` (`zL_add_opp`).
* `FParams.zL_add_opp_eq_zero_iff`, `zL_sub_opp_eq_zero_iff`: the conditions of the two kernels
  are where `z + v = 0` and `z = v`.

That `βp₀` and `βp_{-σ}` are the limits of the kernels is not proved here.
-/

open Complex Matrix

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-- The spin-flip kernel of `(n, m)` in the row `σ`: `σm/(z_σₙ - z_{-σ}ₘ)`, and `βp_{-σ}` where the
spin-flip rate vanishes (`h = 0`, `n = m`, where `z_σₙ = z_{-σ}ₘ`). -/
noncomputable def kerSLs (σ : Bool) (n m : ℤ) : ℂ :=
  if F.h = 0 ∧ n = m then F.β * F.pL (!σ) else F.mL σ / ((F.BL σ).z k n - (F.BL (!σ)).z k m)

theorem kerSLs_true (n m : ℤ) : F.kerSLs k true n m = F.kerSL k n m := rfl

/-- `z↓ₙ + z↑ₘ = z↑ₙ + z↓ₘ`: both are `2μ + i(νₙ + νₘ + ω_k)`. -/
theorem Bdn_z_add_Bup_z (n m : ℤ) : F.Bdn.z k n + F.Bup.z k m = F.Bup.z k n + F.Bdn.z k m := by
  simp only [Params.z, Params.x, Params.w, lat_μ, lat_ν, lat_ω]
  push_cast
  ring

theorem zL_add_opp (σ : Bool) (n m : ℤ) :
    (F.BL σ).z k n + (F.BL (!σ)).z k m = F.Bup.z k n + F.Bdn.z k m := by
  cases σ
  · exact F.Bdn_z_add_Bup_z k n m
  · rfl

@[simp] theorem flip_kerPL (n m : ℤ) : F.flip.kerPL k n m = F.kerPL k n m := by
  simp only [kerPL, flip_μ, flip_β, flip_p₀, flip_ρ, flip_Bup, flip_Bdn, F.Bdn_z_add_Bup_z]

@[simp] theorem flip_kerSLs (σ : Bool) (n m : ℤ) :
    F.flip.kerSLs k σ n m = F.kerSLs k (!σ) n m := by
  simp only [kerSLs, flip_h, neg_eq_zero, flip_β, flip_pL, flip_mL, flip_BL, Bool.not_not]

/-- **The longitudinal input, opposite spins**, for both rows and all `μ`, `h`:
`P(z)χ_{σ,-σ}(z, v) = λ[kerPL - kerSLs_σ] + (N₀_{-σ}(v) + z²N₂_{-σ}(v))/P(v)`, `z = z_σₙ`,
`v = z_{-σ}ₘ`. -/
theorem PL_mul_chiL_opp_all (σ : Bool) (n m : ℤ) :
    F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n m =
      F.lamL * (F.kerPL k n m - F.kerSLs k σ n m) +
        (F.N₀L k (F.rL (!σ)) ((F.BL (!σ)).z k m) +
            (F.BL σ).z k n ^ 2 * F.N₂L k (F.rL (!σ)) ((F.BL (!σ)).z k m)) /
          F.PL k ((F.BL (!σ)).z k m) := by
  cases σ
  · have h := F.flip.PL_mul_chiL_updn_all k n m
    have e : F.chiL false true k n m = F.flip.chiL true false k n m := F.chiL_flip k _ _ n m
    rw [Bool.not_false, e]
    have hr : F.flip.rDn = F.rL true := by rw [← rL_false, flip_rL]; rfl
    rw [flip_kerPL, ← kerSLs_true, flip_kerSLs, Bool.not_true, hr, flip_PL, flip_PL, flip_N₀L,
      flip_N₂L, flip_lamL, flip_Bup, flip_Bdn] at h
    exact h
  · rw [Bool.not_true, BL_true, BL_false, rL_false, kerSLs_true]
    exact F.PL_mul_chiL_updn_all k n m

/-! ### Where the kernels take their limits -/

/-- `z_σₙ + z_{-σ}ₘ = 0` iff `μ = 0` and `ν_m = -νₙ - ω_k`. -/
theorem zL_add_opp_eq_zero_iff (σ : Bool) (n m : ℤ) :
    (F.BL σ).z k n + (F.BL (!σ)).z k m = 0 ↔ F.μ = 0 ∧ n + m + k + 1 = 0 := by
  rw [zL_add_opp, Params.z, Params.z, show F.Bdn.w k = F.Bup.w k from rfl,
    show F.Bup.x n + F.Bup.w k / 2 + (F.Bdn.x m + F.Bup.w k / 2) =
      F.Bup.x n + F.Bdn.x m + F.Bup.w k by ring]
  exact F.pairL_eq_zero_iff k n m

/-- `z_σₙ - z_{-σ}ₘ = 0` iff `h = 0` and `n = m`. -/
theorem zL_sub_opp_eq_zero_iff (σ : Bool) (n m : ℤ) :
    (F.BL σ).z k n - (F.BL (!σ)).z k m = 0 ↔ F.h = 0 ∧ n = m := by
  have hzm : ∀ i j : ℤ, F.Bup.z k i - F.Bdn.z k j = F.Bup.x i - F.Bdn.x j := by
    intro i j
    rw [Params.z, Params.z, show F.Bdn.w k = F.Bup.w k from rfl]
    ring
  cases σ
  · rw [Bool.not_false, BL_true, BL_false, ← neg_sub, neg_eq_zero, hzm, F.flipL_eq_zero_iff,
      eq_comm (a := m)]
  · rw [Bool.not_true, BL_true, BL_false, hzm, F.flipL_eq_zero_iff]

end FParams

end HubbardAtom
