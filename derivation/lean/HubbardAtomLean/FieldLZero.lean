import HubbardAtomLean.FieldLVertex
import HubbardAtomLean.FieldTZero
import HubbardAtomLean.GammaClosedForm

/-!
# The longitudinal vertex at `h = 0`

At `h = 0` the longitudinal sector is the zero-field density and magnetic channels. From the
Hamiltonian, `χ_L,σσ = χ↑↑` and `χ_L,σ(-σ) = χ↑↓` of the zero-field atom `(U, β, μ)`
(`chiL_self_of_h`, `chiL_opp_of_h`), so `χ_L,στ = (χ_d + s_στχ_m)/2` with `s_στ = ±1`
(`chiL_of_h`), and the bubble is `B` (`bL_of_h`). For `μ ≠ 0`, with `χ_r⁻¹` and `Γ_r` of
`GammaClosedForm.lean`, `GammaOperator.lean`:

* `FParams.invLh0`: `Y_στ = (χ_d⁻¹ + s_στχ_m⁻¹)/2`; where `1 + 𝒦_d` and `1 + 𝒦_m` are invertible,
  it is a two-sided inverse of `χ_L` (`hasSum_chiL_mul_invLh0`, `hasSum_invLh0_mul_chiL`) and the
  only right inverse with columns in the class `∑_l |v_l/z_l|² < ∞` (`eq_invLh0_of_hasSum`);
* `FParams.gammaL_h0`: **Eq. (field-h0)**, `β²(Y - b⁻¹)_στ = (Γ_d + s_στΓ_m)/2`, entrywise. The
  identity holds unconditionally, both sides through `Ring.inverse`; `Y` is the inverse of `χ_L`
  where `1 + 𝒦_d` and `1 + 𝒦_m` are invertible.

The closed form of Eq. (field-L) needs `h ≠ 0` (`LGeneric`), so it is not compared with `Y` here.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The zero-field atom -/

/-- At `h = 0` the Hamiltonian is that of the zero-field atom. -/
theorem H_of_h (hh : F.h = 0) : F.H = (F.lat F.μ).H := by
  simp only [H, Params.H, lat_U, lat_μ, hh, ofReal_zero, zero_smul, sub_zero]

/-- At `h = 0` the susceptibilities are those of the zero-field atom. -/
theorem chiLehmann_of_h (hh : F.h = 0) (c c' : Op) (ν ν' ω : ℝ) :
    F.chiLehmann c c' ν ν' ω = (F.lat F.μ).chiLehmann c c' ν ν' ω := by
  simp only [chiLehmann, Params.chiLehmann, thermal, Params.thermal, timeOrdered,
    Params.timeOrdered, heis, Params.heis, F.H_of_h hh, lat_β]

/-- At `h = 0` the spin flip is the identity. -/
theorem flip_of_h (hh : F.h = 0) : F.flip = F := by
  cases F
  simp only [flip, mk.injEq, true_and]
  simp only at hh
  rw [hh, neg_zero]

/-- **Equal spins at `h = 0`**: `χ_L,σσ = χ↑↑` of the zero-field atom. -/
theorem chiL_self_of_h (hh : F.h = 0) (σ : Bool) (n m : ℤ) :
    F.chiL σ σ k n m = (F.lat F.μ).chiUpUp k n m := by
  have hup : F.chiL true true k n m = (F.lat F.μ).chiUpUp k n m := by
    rw [chiL, F.chiLehmann_of_h hh]
    exact (F.lat F.μ).chiLehmann_upUp k n m
  cases σ
  · rw [F.chiL_flip k, F.flip_of_h hh]
    exact hup
  · exact hup

/-- **Opposite spins at `h = 0`**: `χ_L,σ(-σ) = χ↑↓` of the zero-field atom. -/
theorem chiL_opp_of_h (hh : F.h = 0) (σ : Bool) (n m : ℤ) :
    F.chiL σ (!σ) k n m = (F.lat F.μ).chiUpDn k n m := by
  have hup : F.chiL true false k n m = (F.lat F.μ).chiUpDn k n m := by
    rw [chiL, F.chiLehmann_of_h hh]
    exact (F.lat F.μ).chiLehmann_upDn k n m
  cases σ
  · rw [F.chiL_flip k, F.flip_of_h hh]
    exact hup
  · exact hup

/-- `s_στ = 1` for equal spins and `-1` for opposite spins. -/
def sL (σ t : Bool) : ℂ := if σ = t then 1 else -1

/-- **`χ_L` at `h = 0`**: `χ_L,στ = (χ_d + s_στχ_m)/2`, with `χ_r = χ↑↑ + εχ↑↓`. -/
theorem chiL_of_h (hh : F.h = 0) (σ t : Bool) (n m : ℤ) :
    F.chiL σ t k n m =
      ((F.lat F.μ).chiR k 1 n m + sL σ t * (F.lat F.μ).chiR k (-1) n m) / 2 := by
  by_cases hσ : σ = t
  · subst hσ
    rw [F.chiL_self_of_h k hh, sL, ite_eq_left rfl, Params.chiR, Params.chiR]
    ring
  · obtain rfl : t = !σ := by cases σ <;> cases t <;> simp_all
    rw [F.chiL_opp_of_h k hh, sL, ite_eq_right hσ, Params.chiR, Params.chiR]
    ring

/-- **The bubble at `h = 0`** is that of the zero-field atom. -/
theorem bL_of_h (hh : F.h = 0) (σ : Bool) (n : ℤ) : F.bL σ k n = (F.lat F.μ).Bph k n := by
  have hr : ∀ r : ℝ, r = F.ρ - F.mf ∨ r = F.ρ + F.mf → (r : ℂ) = (F.lat F.μ).ρ := by
    rintro r (rfl | rfl) <;> simp [F.mf_of_h hh, F.lat_ρ hh]
  have hG : ∀ j, F.Gs σ (F.ν j) = (F.lat F.μ).G ((F.lat F.μ).x j) := fun j ↦ by
    cases σ
    · rw [F.propagator_dn, Gc, Params.G, hr F.rDn (Or.inr rfl), hh, sub_zero]
      rfl
    · rw [F.propagator_up, Gc, Params.G, hr F.rUp (Or.inl rfl), hh, add_zero]
      rfl
  rw [bL, F.ν_add_ω, hG, hG, Params.Bph, ← (F.lat F.μ).x_add_w]
  rfl

/-! ### The inverse and the vertex -/

variable {F k}
variable (hh : F.h = 0) (hμ : F.μ ≠ 0)

variable (F k) in
/-- `Y_στ = (χ_d⁻¹ + s_στχ_m⁻¹)/2`, with the inverses `χ_r⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)` of the
zero-field atom. -/
noncomputable def invLh0 (hμ : F.μ ≠ 0) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) : ℂ :=
  ((F.lat F.μ).chiInv k 1 hμ n m + sL σ t * (F.lat F.μ).chiInv k (-1) hμ n m) / 2

include hh in
/-- **`Y` is a right inverse of `χ_L`** at `h = 0`, where `1 + 𝒦_d` and `1 + 𝒦_m` are invertible:
`∑_{l, τ} χ_L,στ(n, l)Y_τt(l, m) = δ_σtδ_nm`. -/
theorem hasSum_chiL_mul_invLh0 (hd : IsUnit ((F.lat F.μ).onePlusK k 1 hμ))
    (hm : IsUnit ((F.lat F.μ).onePlusK k (-1) hμ)) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * F.invLh0 k hμ τ l t m)
      (if σ = t then kron n m else 0) := by
  have h₁ := Params.chiInv_isRightInverse hμ hd n m
  have h₂ := Params.chiInv_isRightInverse hμ hm n m
  have h := (h₁.add (h₂.mul_left (sL σ t))).div_const 2
  convert h using 1
  · funext l
    simp only [Fintype.sum_bool, F.chiL_of_h k hh, invLh0]
    cases σ <;> cases t <;>
      simp only [sL, Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte] <;> ring
  · cases σ <;> cases t <;> simp [sL]

include hh in
/-- **`Y` is a left inverse of `χ_L`** at `h = 0`, where `1 + 𝒦_d` and `1 + 𝒦_m` are invertible:
`∑_{l, τ} Y_στ(n, l)χ_L,τt(l, m) = δ_σtδ_nm`. -/
theorem hasSum_invLh0_mul_chiL (hd : IsUnit ((F.lat F.μ).onePlusK k 1 hμ))
    (hm : IsUnit ((F.lat F.μ).onePlusK k (-1) hμ)) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.invLh0 k hμ σ n τ l * F.chiL τ t k l m)
      (if σ = t then kron n m else 0) := by
  have h₁ := Params.chiInv_isLeftInverse hμ hd n m
  have h₂ := Params.chiInv_isLeftInverse hμ hm n m
  have h := (h₁.add (h₂.mul_left (sL σ t))).div_const 2
  convert h using 1
  · funext l
    simp only [Fintype.sum_bool, F.chiL_of_h k hh, invLh0]
    cases σ <;> cases t <;>
      simp only [sL, Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte] <;> ring
  · cases σ <;> cases t <;> simp [sL]

include hh in
/-- **`Y` is the only right inverse in the class**: at `h = 0`, where `1 + 𝒦_d` and `1 + 𝒦_m`
are invertible, a column `(t, m)` with `∑_l |Y'_τl/z_l|² < ∞` for both spins and
`∑_{l, τ} χ_L,στ(n, l)Y'_τl = δ_σtδ_nm` is the column of `Y`. -/
theorem eq_invLh0_of_hasSum (hd : IsUnit ((F.lat F.μ).onePlusK k 1 hμ))
    (hm : IsUnit ((F.lat F.μ).onePlusK k (-1) hμ)) {t : Bool} {m : ℤ} {Y : Bool → ℤ → ℂ}
    (hY : ∀ τ, Summable fun l ↦ ‖Y τ l / (F.lat F.μ).z k l‖ ^ 2)
    (hχ : ∀ σ n, HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * Y τ l)
      (if σ = t then kron n m else 0)) (σ : Bool) (n : ℤ) :
    Y σ n = F.invLh0 k hμ σ n t m := by
  -- `Y↑ ± Y↓` are columns of `χ_d⁻¹` and `±χ_m⁻¹`
  have hcls : ∀ c : ℂ, Summable fun l ↦ ‖(Y true l + c * Y false l) / (F.lat F.μ).z k l‖ ^ 2 :=
    fun c ↦ (((hY true).mul_left 2).add (((hY false).mul_left (2 * ‖c‖ ^ 2)))).of_nonneg_of_le
      (fun _ ↦ by positivity) fun l ↦ by
        rw [add_div, mul_div_assoc]
        set x := ‖Y true l / (F.lat F.μ).z k l‖
        set y := ‖c‖ * ‖Y false l / (F.lat F.μ).z k l‖
        have h1 := norm_add_le (Y true l / (F.lat F.μ).z k l) (c * (Y false l / (F.lat F.μ).z k l))
        rw [norm_mul] at h1
        have h2 := pow_le_pow_left₀ (norm_nonneg _) h1 2
        have hy : y ^ 2 = ‖c‖ ^ 2 * ‖Y false l / (F.lat F.μ).z k l‖ ^ 2 := mul_pow _ _ _
        nlinarith [sq_nonneg (x - y)]
  have hD : ∀ n, HasSum (fun l ↦ (F.lat F.μ).chiR k 1 n l * (Y true l + 1 * Y false l))
      (kron n m) := fun n ↦ by
    have h := (hχ true n).add (hχ false n)
    convert h using 1
    · funext l
      simp only [Fintype.sum_bool, F.chiL_of_h k hh, sL]
      norm_num
      ring
    · cases t <;> simp
  have hM : ∀ n, HasSum (fun l ↦ (F.lat F.μ).chiR k (-1) n l * (Y true l + -1 * Y false l))
      (sL true t * kron n m) := fun n ↦ by
    have h := (hχ true n).sub (hχ false n)
    convert h using 1
    · funext l
      simp only [Fintype.sum_bool, F.chiL_of_h k hh, sL]
      norm_num
      ring
    · cases t <;> simp [sL]
  have eD := fun n ↦ Params.eq_chiInv_of_hasSum hμ hd (hcls 1) hD n
  have eM : ∀ n, Y true n + -1 * Y false n = sL true t * (F.lat F.μ).chiInv k (-1) hμ n m := by
    intro n
    have hst : sL true t * sL true t = 1 := by cases t <;> simp [sL]
    have hM' : ∀ n, HasSum (fun l ↦ (F.lat F.μ).chiR k (-1) n l *
        (sL true t * (Y true l + -1 * Y false l))) (kron n m) := fun n ↦ by
      convert (hM n).mul_left (sL true t) using 1
      · funext l
        ring
      · rw [← mul_assoc, hst, one_mul]
    have hcls' : Summable fun l ↦ ‖sL true t * (Y true l + -1 * Y false l) /
        (F.lat F.μ).z k l‖ ^ 2 := by
      have hs1 : ‖sL true t‖ = 1 := by cases t <;> simp [sL]
      simpa [mul_div_assoc, norm_mul, hs1] using hcls (-1)
    have e := Params.eq_chiInv_of_hasSum hμ hm hcls' hM' n
    rw [← e, ← mul_assoc, hst, one_mul]
  have e1 := eD n
  have e2 := eM n
  simp only [invLh0]
  cases σ
  · have : Y false n = ((Y true n + 1 * Y false n) - (Y true n + -1 * Y false n)) / 2 := by ring
    rw [this, e1, e2]
    cases t
    · simp [sL]
    · simp [sL]
      ring
  · have : Y true n = ((Y true n + 1 * Y false n) + (Y true n + -1 * Y false n)) / 2 := by ring
    rw [this, e1, e2]

include hh in
/-- **The longitudinal vertex at `h = 0`, Eq. (field-h0)**: `β²(Y - b⁻¹)_στ = (Γ_d + s_στΓ_m)/2`,
entrywise, with `Γ_r` of the zero-field atom (`GammaOperator.lean`) and `bₙ = B(νₙ) ≠ 0`. Both sides
are defined through `Ring.inverse`; where `1 + 𝒦_d` and `1 + 𝒦_m` are invertible, `Y` is the
inverse of `χ_L` (`hasSum_chiL_mul_invLh0`, `hasSum_invLh0_mul_chiL`, `eq_invLh0_of_hasSum`). -/
theorem gammaL_h0 (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    (F.β : ℂ) ^ 2 * (F.invLh0 k hμ σ n t m - if σ = t then kron n m / F.bL σ k n else 0) =
      ((F.lat F.μ).gammaOp k 1 hμ n m + sL σ t * (F.lat F.μ).gammaOp k (-1) hμ n m) / 2 := by
  rw [Params.gammaOp_eq_chiInv hμ, Params.gammaOp_eq_chiInv hμ, F.bL_of_h k hh, invLh0]
  have hβ : (F.lat F.μ).β = F.β := rfl
  rw [hβ]
  cases σ <;> cases t <;> simp [sL] <;> ring

end FParams

end HubbardAtom
