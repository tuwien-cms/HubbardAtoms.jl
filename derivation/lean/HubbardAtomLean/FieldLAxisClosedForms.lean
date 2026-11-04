import HubbardAtomLean.FieldLAxisFormula
import HubbardAtomLean.FieldLAxisValues
import HubbardAtomLean.GammaClosedForm

/-!
# The axis formula on `h = 0` and `μ = 0` and the zero-field closed forms

Section "Reductions, continuous extensions and exceptional sets" of the note describes the
longitudinal vertex on the axes `h = 0` and `μ = 0` in two ways: by the continuation `gammaLAll` of
Eq. (field-L) with Eqs. (field-kernel-limits), (field-L-quotient) (`FieldLAxisFormula.lean`), and
by Eqs. (field-h0), (field-mu0), which reduce it to the zero-field density and magnetic vertices,
with the closed forms `gammaR` of Theorem "density and magnetic". Here the two are compared,
through the operator vertex `Γ_L`, which both equal (`LAxisGeneric.gammaOpLAll_eq`;
`gammaOpLAll_h0`, `gammaOpLAll_mu0`; `DMGeneric.gammaOp_eq`):

* `FParams.LAxisGeneric.gammaLAll_h0`: at `h = 0`, `μ ≠ 0`, where `LAxisGeneric` holds and
  `1 + 𝒦_d`, `1 + 𝒦_m` of the zero-field atom `(U, β, μ)` are invertible,
  `gammaLAll_στ = (Γ_d + s_στΓ_m)/2`, entrywise, with the operator vertices `Γ_r`
  (`GammaOperator.lean`);
* `FParams.LAxisGeneric.gammaLAll_h0_closed`: **the axis formula is Eq. (field-h0) with the closed
  forms**: where in addition both zero-field channels satisfy the assumptions of Theorem "density
  and magnetic" (`DMGeneric`), the same with `gammaR`; in the note's form
  `gammaLAll_σσ = (Γ_d + Γ_m)/2` and `gammaLAll_σ(-σ) = (Γ_d - Γ_m)/2`
  (`LAxisGeneric.gammaLAll_h0_closed_self`, `LAxisGeneric.gammaLAll_h0_closed_not`);
* `FParams.LAxisGeneric.gammaLAll_mu0`, `LAxisGeneric.gammaLAll_mu0_closed`: **the axis formula is
  Eq. (field-mu0)**: at `μ = 0`, `h ≠ 0`, the same for the zero-field source `D = (-U, β, h)`,
  `gammaLAll_στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)`, with the operator vertices
  and, under `DMGeneric` for both channels of `D`, with the closed forms.

The invertibility of `1 + 𝒦` that `LAxisGeneric.gammaOpLAll_eq` needs is not assumed: it follows
from that of the zero-field channels (`isUnit_onePlusKLAll_iff_onePlusK_h0`,
`isUnit_onePlusKLAll_iff_onePlusK_mu0`), which `DMGeneric` gives (`DMGeneric.isUnit_onePlusK`).

Assumed: `LAxisGeneric` at the axis point, and the invertibility of `1 + 𝒦_d`, `1 + 𝒦_m` (for the
closed forms, `DMGeneric` for both zero-field channels). Not shown: that these hold at given axis
points; the corner `μ = h = 0`, where `LAxisGeneric` never holds (`not_lAxisGeneric_corner`) and
`DMGeneric` of the zero-field channels, which assumes `μ ≠ 0` for `(U, β, μ)` and `h ≠ 0` for `D`,
does not apply; and an identity between the closed forms without invertibility, which would have to
match the roots and mode coefficients of `gammaLAll` (the roots of `E₈`, which the note factors as
`e_d e_m` at `h = 0`) with those of `gammaR`.
Not in the note: the comparison as a proof; the note gives Eqs. (field-h0), (field-mu0) as the
values on these axes.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable {F : FParams} {k : ℤ}

/-! ### `h = 0` -/

/-- **The axis formula at `h = 0`, Eq. (field-h0) with the operator vertices**: for `μ ≠ 0`,
where `LAxisGeneric` holds and `1 + 𝒦_d`, `1 + 𝒦_m` of the zero-field atom `(U, β, μ)` are
invertible, `gammaLAll_στ = (Γ_d + s_στΓ_m)/2`, entrywise. Both sides are `Γ_L`
(`LAxisGeneric.gammaOpLAll_eq`, with `1 + 𝒦` invertible by `isUnit_onePlusKLAll_iff_onePlusK_h0`,
and `gammaOpLAll_h0`). -/
theorem LAxisGeneric.gammaLAll_h0 {u₀ : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u₀) (hh : F.h = 0)
    (hμ : F.μ ≠ 0) (hd : IsUnit ((F.lat F.μ).onePlusK k 1 hμ))
    (hm : IsUnit ((F.lat F.μ).onePlusK k (-1) hμ)) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.gammaLAll k u₀ σ n t m =
      ((F.lat F.μ).gammaOp k 1 hμ n m + sL σ t * (F.lat F.μ).gammaOp k (-1) hμ n m) / 2 := by
  rw [← h₀.gammaOpLAll_eq ((isUnit_onePlusKLAll_iff_onePlusK_h0 hh hμ).mpr ⟨hd, hm⟩)]
  exact gammaOpLAll_h0 hh hμ hd hm σ n t m

/-- **The axis formula at `h = 0` is Eq. (field-h0) with the closed forms**: where `LAxisGeneric`
holds and both zero-field channels of `(U, β, μ)` satisfy the assumptions of Theorem "density and
magnetic", `gammaLAll_στ = (Γ_d + s_στΓ_m)/2`, entrywise, with the closed forms `gammaR` of
Eq. (Gamma-ph) (`LAxisGeneric.gammaLAll_h0`, `DMGeneric.gammaOp_eq`). -/
theorem LAxisGeneric.gammaLAll_h0_closed {u₀ : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u₀)
    (hh : F.h = 0) {u₁ u₂ u₃ u₄ : ℂ} (hd : (F.lat F.μ).DMGeneric k 1 u₁ u₂)
    (hm : (F.lat F.μ).DMGeneric k (-1) u₃ u₄) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.gammaLAll k u₀ σ n t m =
      ((F.lat F.μ).gammaR k 1 u₁ u₂ n m + sL σ t * (F.lat F.μ).gammaR k (-1) u₃ u₄ n m) / 2 := by
  rw [h₀.gammaLAll_h0 hh hd.μ_ne hd.isUnit_onePlusK hm.isUnit_onePlusK, hd.gammaOp_eq,
    hm.gammaOp_eq]

/-- **Equal spins at `h = 0`**: under the assumptions of `LAxisGeneric.gammaLAll_h0_closed`,
`gammaLAll_σσ = (Γ_d + Γ_m)/2`, the first part of Eq. (field-h0). -/
theorem LAxisGeneric.gammaLAll_h0_closed_self {u₀ : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u₀)
    (hh : F.h = 0) {u₁ u₂ u₃ u₄ : ℂ} (hd : (F.lat F.μ).DMGeneric k 1 u₁ u₂)
    (hm : (F.lat F.μ).DMGeneric k (-1) u₃ u₄) (σ : Bool) (n m : ℤ) :
    F.gammaLAll k u₀ σ n σ m =
      ((F.lat F.μ).gammaR k 1 u₁ u₂ n m + (F.lat F.μ).gammaR k (-1) u₃ u₄ n m) / 2 := by
  rw [h₀.gammaLAll_h0_closed hh hd hm, sL, ite_eq_left rfl, one_mul]

/-- **Opposite spins at `h = 0`**: under the assumptions of `LAxisGeneric.gammaLAll_h0_closed`,
`gammaLAll_σ(-σ) = (Γ_d - Γ_m)/2`, the second part of Eq. (field-h0). -/
theorem LAxisGeneric.gammaLAll_h0_closed_not {u₀ : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u₀)
    (hh : F.h = 0) {u₁ u₂ u₃ u₄ : ℂ} (hd : (F.lat F.μ).DMGeneric k 1 u₁ u₂)
    (hm : (F.lat F.μ).DMGeneric k (-1) u₃ u₄) (σ : Bool) (n m : ℤ) :
    F.gammaLAll k u₀ σ n (!σ) m =
      ((F.lat F.μ).gammaR k 1 u₁ u₂ n m - (F.lat F.μ).gammaR k (-1) u₃ u₄ n m) / 2 := by
  rw [h₀.gammaLAll_h0_closed hh hd hm, sL, ite_eq_right (by cases σ <;> simp)]
  ring

/-! ### `μ = 0` -/

/-- **The axis formula at `μ = 0`, Eq. (field-mu0) with the operator vertices**: for `h ≠ 0`,
where `LAxisGeneric` holds and `1 + 𝒦_d^D`, `1 + 𝒦_m^D` of the zero-field source
`D = (-U, β, h)` are invertible,
`gammaLAll_στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)`, entrywise. Both sides are `Γ_L`
(`LAxisGeneric.gammaOpLAll_eq`, with `1 + 𝒦` invertible by
`isUnit_onePlusKLAll_iff_onePlusK_mu0`, and `gammaOpLAll_mu0`). -/
theorem LAxisGeneric.gammaLAll_mu0 {u₀ : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u₀) (hμ0 : F.μ = 0)
    (hh : F.h ≠ 0) (hd : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k 1 hh))
    (hm : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k (-1) hh)) (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    F.gammaLAll k u₀ σ n t m =
      εL σ * εL t / 2 * ((F.shiba.lat F.shiba.μ).gammaOp k 1 hh (RL k σ n) (RL k t m) +
        εL σ * εL t * (F.shiba.lat F.shiba.μ).gammaOp k (-1) hh (RL k σ n) (RL k t m)) := by
  rw [← h₀.gammaOpLAll_eq ((isUnit_onePlusKLAll_iff_onePlusK_mu0 hμ0 hh).mpr ⟨hd, hm⟩)]
  exact gammaOpLAll_mu0 hμ0 hh hd hm σ n t m

/-- **The axis formula at `μ = 0` is Eq. (field-mu0) with the closed forms**: where
`LAxisGeneric` holds and both zero-field channels of the source `D = (-U, β, h)` satisfy the
assumptions of Theorem "density and magnetic",
`gammaLAll_στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)`, entrywise, with the closed
forms `gammaR` of Eq. (Gamma-ph) (`LAxisGeneric.gammaLAll_mu0`, `DMGeneric.gammaOp_eq`). -/
theorem LAxisGeneric.gammaLAll_mu0_closed {u₀ : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u₀)
    (hμ0 : F.μ = 0) {u₁ u₂ u₃ u₄ : ℂ} (hd : (F.shiba.lat F.shiba.μ).DMGeneric k 1 u₁ u₂)
    (hm : (F.shiba.lat F.shiba.μ).DMGeneric k (-1) u₃ u₄) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.gammaLAll k u₀ σ n t m =
      εL σ * εL t / 2 * ((F.shiba.lat F.shiba.μ).gammaR k 1 u₁ u₂ (RL k σ n) (RL k t m) +
        εL σ * εL t * (F.shiba.lat F.shiba.μ).gammaR k (-1) u₃ u₄ (RL k σ n) (RL k t m)) := by
  rw [h₀.gammaLAll_mu0 hμ0 hd.μ_ne hd.isUnit_onePlusK hm.isUnit_onePlusK, hd.gammaOp_eq,
    hm.gammaOp_eq]

end FParams

end HubbardAtom
