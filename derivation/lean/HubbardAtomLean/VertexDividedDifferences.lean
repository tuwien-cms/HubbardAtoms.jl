import HubbardAtomLean.FieldTransverse
import HubbardAtomLean.Singlet

/-!
# The vertex as a divided difference of one function

Off the diagonal, each closed-form vertex is a divided difference of one function of one frequency,
as the susceptibilities are (`DividedDifferences.lean`):

* `Params.gammaR_divided_difference`: density and magnetic, `(zₙ² - zₘ²)Γ_r,nm = W_r(zₙ) - W_r(zₘ)`
  with `W_r(z) = εUz² - λρzF(z) - ∑ⱼ γⱼ/(z² - sⱼ)` (`λρ = εU²ρ`, `γⱼ = λρZⱼ/(sⱼ - s_ℓ)`), for
  `zₙ², zₘ² ∉ {s₁, s₂}`;
* `Params.gammaS_divided_difference`: singlet, `(tₙ - tₘ)Γ_s,nm = W_s(tₙ) - W_s(tₘ)` with
  `W_s(t) = 2Ut + 2Ucₛ/(t + Aₛ)`, `cₛ = 𝒫(-Aₛ)/Kₛ`, for `tₙ + Aₛ ≠ 0`, `tₘ + Aₛ ≠ 0`;
* `FParams.gammaTUp_divided_difference`: transverse, `(zₙ² - zₘ²)Γ_T↑,nm = W_T(zₙ) - W_T(zₘ)` with
  `W_T(z) = -Uz² - λρzF(z) - ∑ⱼ γⱼ/(z² - sⱼ)` and the transverse data (`λρ = -U²ρ`).

All entries are covered: on the diagonal (and, for the singlet, on the mirror) both sides vanish,
and so they do on the mirror `zₘ = -zₙ` at `μ = 0`, where `ρ = 0`. On the domains of the theorems
the assumptions hold: `DMGeneric.z_sq_ne₁`, `SingletGeneric.t_add_As_ne`, and the same for
`TGeneric`.
-/

open Complex

namespace HubbardAtom

/-- A constant, a dense term and two separable terms form the divided difference in `z²` of
`cz² - LzF(z) - ∑ⱼ γⱼ/(z² - sⱼ)`. -/
theorem dd_dense_two {c L z v Fz Fv γ₁ γ₂ s₁ s₂ : ℂ} (hzv : z ^ 2 - v ^ 2 ≠ 0)
    (hz₁ : z ^ 2 - s₁ ≠ 0) (hz₂ : z ^ 2 - s₂ ≠ 0) (hv₁ : v ^ 2 - s₁ ≠ 0)
    (hv₂ : v ^ 2 - s₂ ≠ 0) :
    (z ^ 2 - v ^ 2) * (c - L * ((z * Fz - v * Fv) / (z ^ 2 - v ^ 2)) +
        γ₁ * (1 / (z ^ 2 - s₁)) * (1 / (v ^ 2 - s₁)) +
        γ₂ * (1 / (z ^ 2 - s₂)) * (1 / (v ^ 2 - s₂))) =
      (c * z ^ 2 - L * (z * Fz) - γ₁ / (z ^ 2 - s₁) - γ₂ / (z ^ 2 - s₂)) -
        (c * v ^ 2 - L * (v * Fv) - γ₁ / (v ^ 2 - s₁) - γ₂ / (v ^ 2 - s₂)) := by
  field_simp
  ring

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `W_r(z) = εUz² - λρzF(z) - ∑ⱼ γⱼ/(z² - sⱼ)`, `γⱼ = λρZⱼ/(sⱼ - s_ℓ)`, written with `εU = 2εa`. -/
noncomputable def Wr (u₁ u₂ z : ℂ) : ℂ :=
  2 * ε * A.a * z ^ 2 - A.lam ε * A.ρ * (z * A.Fph k ε z) -
      A.lam ε * A.ρ * A.Zph k ε u₁ / (u₁ ^ 2 - u₂ ^ 2) / (z ^ 2 - u₁ ^ 2) -
    A.lam ε * A.ρ * A.Zph k ε u₂ / (u₂ ^ 2 - u₁ ^ 2) / (z ^ 2 - u₂ ^ 2)

/-- **The density and magnetic vertex as a divided difference**:
`(zₙ² - zₘ²)Γ_r,nm = W_r(zₙ) - W_r(zₘ)` for the closed form of Eq. (Gamma-ph), if `zₙ²` and `zₘ²`
are not roots of `e`; on the mirror at `μ = 0` both sides vanish, as `ρ = 0`. -/
theorem gammaR_divided_difference {u₁ u₂ : ℂ} {n m : ℤ}
    (hn₁ : A.z k n ^ 2 - u₁ ^ 2 ≠ 0) (hn₂ : A.z k n ^ 2 - u₂ ^ 2 ≠ 0)
    (hm₁ : A.z k m ^ 2 - u₁ ^ 2 ≠ 0) (hm₂ : A.z k m ^ 2 - u₂ ^ 2 ≠ 0) :
    (A.z k n ^ 2 - A.z k m ^ 2) * A.gammaR k ε u₁ u₂ n m =
      A.Wr k ε u₁ u₂ (A.z k n) - A.Wr k ε u₁ u₂ (A.z k m) := by
  by_cases he : A.z k n = A.z k m
  · rw [he, sub_self, zero_mul, sub_self]
  have hnm : n ≠ m := fun h ↦ he (by rw [h])
  by_cases hzv : A.z k n + A.z k m = 0
  · have hμ : A.μ = 0 := by
      have h := congrArg re hzv
      simp only [A.z_eq, add_re, ofReal_re, mul_re, ofReal_im, I_re, I_im, zero_re] at h
      linarith
    have hρ : (A.ρ : ℂ) = 0 := by rw [ρ, p₂_eq_p₀_of_mu hμ, sub_self, ofReal_zero]
    rw [show A.z k m = -A.z k n by linear_combination hzv, neg_sq, sub_self, zero_mul, Wr, Wr, hρ,
      neg_sq]
    ring
  have hsq : A.z k n ^ 2 - A.z k m ^ 2 ≠ 0 := by
    rw [show A.z k n ^ 2 - A.z k m ^ 2 = (A.z k n - A.z k m) * (A.z k n + A.z k m) by ring]
    exact mul_ne_zero (sub_ne_zero.mpr he) hzv
  rw [gammaR, DD, ite_eq_right he, kron_of_ne hnm, mul_zero, zero_add, Wr, Wr]
  convert dd_dense_two hsq hn₁ hn₂ hm₁ hm₂ using 1

/-- `W_s(t) = 2Ut + 2Ucₛ/(t + Aₛ)`, `cₛ = 𝒫(-Aₛ)/Kₛ`, written with `2U = 4a`. -/
noncomputable def Ws (ξ t : ℂ) : ℂ :=
  4 * A.a * t + 4 * A.a * (A.calP k (-A.As k) / A.Ks k ξ) / (t + A.As k)

/-- **The singlet vertex as a divided difference**: `(tₙ - tₘ)Γ_s,nm = W_s(tₙ) - W_s(tₘ)` for the
closed form of Eq. (Gamma-s), if `tₙ + Aₛ ≠ 0` and `tₘ + Aₛ ≠ 0`; on the diagonal and the mirror,
`tₙ = tₘ`, both sides vanish. -/
theorem gammaS_divided_difference {ξ : ℂ} {n m : ℤ} (hn : A.t k n + A.As k ≠ 0)
    (hm : A.t k m + A.As k ≠ 0) :
    (A.t k n - A.t k m) * A.gammaS k ξ n m = A.Ws k ξ (A.t k n) - A.Ws k ξ (A.t k m) := by
  by_cases he : A.t k n = A.t k m
  · rw [he, sub_self, zero_mul, sub_self]
  have hnm : n ≠ m := fun h ↦ he (by rw [h])
  have hmn : m ≠ mirror k n := fun h ↦ he (by rw [h, A.t_mirror])
  have hd := sub_ne_zero.mpr he
  rw [gammaS, kron_of_ne hnm, kron_of_ne hmn, add_zero, mul_zero, zero_add, Ws, Ws]
  field_simp
  ring

end Params

namespace FParams

variable (F : FParams) (k : ℤ)

/-- `W_T(z) = -Uz² - λρzF(z) - ∑ⱼ γⱼ/(z² - sⱼ)` with the transverse data, `γⱼ = λρZⱼ/(sⱼ - s_ℓ)`,
written with `-U = 2(-a)`. -/
noncomputable def WTup (u₁ u₂ z : ℂ) : ℂ :=
  2 * (-F.a) * z ^ 2 - F.lamT * F.ρ * (z * F.FT k z) -
      F.lamT * F.ρ * F.ZT k u₁ / (u₁ ^ 2 - u₂ ^ 2) / (z ^ 2 - u₁ ^ 2) -
    F.lamT * F.ρ * F.ZT k u₂ / (u₂ ^ 2 - u₁ ^ 2) / (z ^ 2 - u₂ ^ 2)

/-- **The transverse vertex as a divided difference**:
`(zₙ² - zₘ²)Γ_T↑,nm = W_T(zₙ) - W_T(zₘ)` for the closed form of Eq. (field-T), if `zₙ²` and `zₘ²`
are not roots of `e`; on the mirror at `μ = 0` both sides vanish, as `ρ = 0`. -/
theorem gammaTUp_divided_difference {u₁ u₂ : ℂ} {n m : ℤ}
    (hn₁ : F.zT k n ^ 2 - u₁ ^ 2 ≠ 0) (hn₂ : F.zT k n ^ 2 - u₂ ^ 2 ≠ 0)
    (hm₁ : F.zT k m ^ 2 - u₁ ^ 2 ≠ 0) (hm₂ : F.zT k m ^ 2 - u₂ ^ 2 ≠ 0) :
    (F.zT k n ^ 2 - F.zT k m ^ 2) * F.gammaTUp k u₁ u₂ n m =
      F.WTup k u₁ u₂ (F.zT k n) - F.WTup k u₁ u₂ (F.zT k m) := by
  by_cases he : F.zT k n = F.zT k m
  · rw [he, sub_self, zero_mul, sub_self]
  have hnm : n ≠ m := fun h ↦ he (by rw [h])
  by_cases hzv : F.zT k n + F.zT k m = 0
  · have hμ : F.μ = 0 := by
      have h := congrArg re hzv
      simp only [zT, Params.z_eq, add_re, ofReal_re, mul_re, ofReal_im, I_re, I_im, zero_re,
        lat_μ] at h
      linarith
    have hρ : (F.ρ : ℂ) = 0 := by
      rw [ρ, F.p₂_eq_p₀_mul, hμ, mul_zero, Real.exp_zero, mul_one, sub_self, ofReal_zero]
    rw [show F.zT k m = -F.zT k n by linear_combination hzv, neg_sq, sub_self, zero_mul, WTup,
      WTup, hρ, neg_sq]
    ring
  have hsq : F.zT k n ^ 2 - F.zT k m ^ 2 ≠ 0 := by
    rw [show F.zT k n ^ 2 - F.zT k m ^ 2 = (F.zT k n - F.zT k m) * (F.zT k n + F.zT k m) by ring]
    exact mul_ne_zero (sub_ne_zero.mpr he) hzv
  rw [gammaTUp, DDT, ite_eq_right he, kron_of_ne hnm, mul_zero, zero_add, WTup, WTup]
  convert dd_dense_two hsq hn₁ hn₂ hm₁ hm₂ using 1

end FParams

end HubbardAtom
