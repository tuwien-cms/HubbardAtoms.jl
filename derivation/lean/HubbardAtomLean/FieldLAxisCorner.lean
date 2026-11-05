import HubbardAtomLean.DMOperator
import HubbardAtomLean.FieldLAxisContinuity
import HubbardAtomLean.FieldLAxisValues

/-!
# `1 + 𝒦` and `Γ_L` on `h = 0` and `μ = 0` through the corner `μ = h = 0`

`FieldLZero.lean`, `FieldLMu0.lean` and `FieldLAxisValues.lean` reduce `χ_L` on `h = 0` (`μ = 0`)
to the zero-field density and magnetic channels of `(U, β, μ)` (of the Shiba source `(-U, β, h)`)
through their operators on the lattice `z` of `GammaOperator.lean`, which needs `μ ≠ 0` (`h ≠ 0`).
Here the same argument runs on the operators of `DMOperator.lean` on the lattice `z'` of
`OpLatticeAll.lean` (`onePlusKR`, `chiInvR`, `gammaOpR`), which exist for every `μ`. At `h = 0`
both spin lattices of `FieldLAxisOperator.lean` are that zero-field lattice `z'` (`BL_of_h`), so
the classes agree as they are.

* `FParams.isUnit_onePlusKLAll_iff_onePlusKR_h0`: at `h = 0`, for every `μ`, `1 + 𝒦` is invertible
  iff `1 + 𝒦_d` and `1 + 𝒦_m` of the zero-field atom `(U, β, μ)` on `z'` are: a null vector `v` of
  `χ_L` in the class gives the null vectors `v↑ ± v↓` of `χ_d`, `χ_m`, and conversely `(w, ±w)`;
* `FParams.invLh0R`: `Y_στ = (χ_d⁻¹ + s_στχ_m⁻¹)/2` with `chiInvR`; where `1 + 𝒦_d` and `1 + 𝒦_m`
  are invertible, it is a two-sided inverse of `χ_L` (`hasSum_chiL_mul_invLh0R`,
  `hasSum_invLh0R_mul_chiL`), the only right inverse in the class (`eq_invLh0R_of_hasSum`), and
  `χ_L⁻¹ = Y` (`chiInvLAll_eq_invLh0R`);
* `FParams.gammaOpLAll_h0R`: **Eq. (field-h0) for the operator vertex, for every `μ`**,
  `Γ_L,στ = (Γ_d + s_στΓ_m)/2` with `gammaOpR`, where `1 + 𝒦_d` and `1 + 𝒦_m` are invertible
  (`gammaL_h0R` is the identity behind it, through `Ring.inverse`);
* `FParams.chiInvLAll_shiba`, `FParams.gammaOpLAll_shiba`: for every `μ`, `h`,
  `Γ_L,στ(n, m) = ε_σε_τ Γ_L,στ^shiba(R_σn, R_τm)` with the Shiba image `(-U, β, h, μ)`, and the
  same for `χ_L⁻¹` (by the uniqueness of the right inverse in the class,
  `eq_chiInvLAll_of_hasSum`); so at `μ = 0`, for every `h`, `1 + 𝒦` is invertible iff `1 + 𝒦_d^D`,
  `1 + 𝒦_m^D` of the source `D = (-U, β, h)` on `z'` are (`isUnit_onePlusKLAll_iff_onePlusKR_mu0`),
  and then **Eq. (field-mu0) holds for the operator vertex**,
  `Γ_L,στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)` with `gammaOpR` (`gammaOpLAll_mu0R`);
* `FParams.gammaOpLAll_corner`: **at the corner `μ = h = 0`**, under the assumptions of Corollary
  "half filling" for both channels of `(U, β, 0)` (`HFHyp`), `1 + 𝒦` is invertible
  (`isUnit_onePlusKLAll_corner`; in general iff `1 + 𝒦_d`, `1 + 𝒦_m` of `(U, β, 0)` are,
  `isUnit_onePlusKLAll_iff_corner`), `χ_L⁻¹ = (invHF_d + s_στ invHF_m)/2` (`chiInvLAll_corner`) and
  `Γ_L,στ = (Γ_d + s_στΓ_m)/2` with the closed forms `gammaHF` of Eq. (Gamma-hf); with a lattice
  zero `z_c² = ŝ` in one or both channels (`HFLZ`), the same with `gammaHFlz` of Eq. (Gamma-hf-lz)
  in that channel (`gammaOpLAll_corner_lz_d`, `gammaOpLAll_corner_lz_m`, `gammaOpLAll_corner_lz`);
  under `HFHyp` for both channels of the Shiba source `(-U, β, 0)` instead, Eq. (field-mu0) with
  its closed forms `gammaHF` (`gammaOpLAll_corner_mu0`);
* `FParams.tendsto_gammaL_corner`: along a family `(U, β, μ, h)(x)` continuous at a corner point
  `x₀` where both channels of `(U, β, 0)` satisfy `HFHyp`, along any approach through the domain of
  Eq. (field-L), **`gammaL → (gammaHF_d + s_στ gammaHF_m)/2`** (by `tendsto_gammaL_of_LGeneric`),
  and `Γ_L` is continuous at `x₀` (`continuousAt_gammaOpLAll_corner`).

Assumed: for `gammaOpLAll_h0R`, `gammaOpLAll_mu0R`, the invertibility of `1 + 𝒦_d`, `1 + 𝒦_m` on
`z'`; at the corner, `HFHyp` (with `U ≠ 0`) or `HFLZ` for each channel. For the limit, approaches
through the domain of Eq. (field-L) to the corner are not constructed: `LAxisGeneric`, which gives
them elsewhere on the axes (`FieldLAxisGeneric.lean`), never holds at `μ = h = 0`
(`not_lAxisGeneric_corner`), so the approach may be empty. Not shown: a value of the explicit axis
formula `gammaLAll` at the corner (its mode coefficients are `0/0` there, `FieldLAxisFormula.lean`).
No identity between the closed forms of `(U, β, 0)` and of `(-U, β, 0)` is stated: where both sets
of assumptions hold it follows from `gammaOpLAll_corner` and `gammaOpLAll_corner_mu0`; a direct
proof, or one where only one set holds, is not given. Not in the note: the operator statements, the
Shiba relation for `Γ_L`, and the proofs; at `μ = h = 0` the note takes Eqs. (field-h0),
(field-mu0) as the values.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace FParams

variable {F : FParams} {k : ℤ}

/-! ### The class and the rows at `h = 0` -/

variable (F k) in
/-- The columns of `χ_L⁻¹` are in the class `∑_l |v_τl/z'_τl|² < ∞` on the lattice `z'` of each
spin `τ`. -/
theorem summable_chiInvLAll_zOp (t : Bool) (m : ℤ) (τ : Bool) :
    Summable fun l ↦ ‖F.chiInvLAll k τ l t m / (F.BL τ).zOp k l‖ ^ 2 := by
  have h := (F.opLatLAll k).summable_chiInvX (F.chiLe k) (F.chiDecay_chiLe_all k) (encL (t, m))
  have h' : Summable fun p : Bool × ℤ ↦
      ‖F.chiInvLAll k p.1 p.2 t m / (F.BL p.1).zOp k p.2‖ ^ 2 := by
    refine (F.summable_opLatLAll_iff k (v := fun p ↦ F.chiInvLAll k p.1 p.2 t m)).mp
      (h.congr fun j ↦ ?_)
    simp only [chiInvLAll, Prod.mk.eta, Equiv.apply_symm_apply]
  exact (summable_prod_iff_spin (z := fun τ ↦ (F.BL τ).zOp k)).mp h' τ

/-- At `h = 0` the class on `Bool × ℤ` is the class of the zero-field lattice `z'` for each
spin. -/
theorem summable_zOpL_iff_h0 (hh : F.h = 0) {v : Bool × ℤ → ℂ} :
    Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2) ↔
      ∀ τ, Summable fun l ↦ ‖v (τ, l) / (F.lat F.μ).zOp k l‖ ^ 2 := by
  simp only [BL_of_h hh]
  exact summable_prod_iff_spin (z := fun _ ↦ (F.lat F.μ).zOp k)

/-- `χ_r w = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)w` converges row by row for `w` in the class of `z'`, for
every `μ`. -/
private theorem _root_.HubbardAtom.Params.hasSum_chiR_mul_zOp (A : Params) (k : ℤ) (ε : ℂ)
    {w : ℤ → ℂ} (hw : Summable fun l ↦ ‖w l / A.zOp k l‖ ^ 2) (n : ℤ) :
    HasSum (fun l ↦ A.chiR k ε n l * w l) (∑' l, A.chiR k ε n l * w l) :=
  ((A.opLatticeAll k).hasSum_X_mul _ (A.chiDecay_chiR_all k ε) hw n).summable.hasSum

/-! ### `1 + 𝒦` at `h = 0` -/

/-- **`1 + 𝒦` at `h = 0`, for every `μ`**: `1 + 𝒦` of `χ_L` is invertible iff `1 + 𝒦_d` and
`1 + 𝒦_m` of the zero-field atom `(U, β, μ)` on the lattice `z'` (`DMOperator.lean`) are. A null
vector `v` of `χ_L` with `∑_{σn} |v_σn/z'_σn|² < ∞` gives the null vectors `v↑ ± v↓` of `χ_d`,
`χ_m` in their class, and a null vector `w` of `χ_d` (of `χ_m`) gives the null vector `(w, w)`
(`(w, -w)`) of `χ_L`. -/
theorem isUnit_onePlusKLAll_iff_onePlusKR_h0 (hh : F.h = 0) :
    IsUnit (F.onePlusKLAll k) ↔
      IsUnit ((F.lat F.μ).onePlusKR k 1) ∧ IsUnit ((F.lat F.μ).onePlusKR k (-1)) := by
  -- a row of `χ_L` against `(c↑w, c↓w)`
  have hrow : ∀ w : ℤ → ℂ, Summable (fun l ↦ ‖w l / (F.lat F.μ).zOp k l‖ ^ 2) →
      ∀ (c : Bool → ℂ) σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * (c p.1 * w p.2))
        (∑ τ, c τ * ((∑' l, (F.lat F.μ).chiR k 1 n l * w l) +
          sL σ τ * ∑' l, (F.lat F.μ).chiR k (-1) n l * w l) / 2) := by
    intro w hw c σ n
    refine hasSum_prod_of_spin fun τ ↦ ?_
    have hD := (F.lat F.μ).hasSum_chiR_mul_zOp k 1 hw n
    have hM := (F.lat F.μ).hasSum_chiR_mul_zOp k (-1) hw n
    convert ((hD.add (hM.mul_left (sL σ τ))).mul_left (c τ)).div_const 2 using 1
    funext l
    rw [F.chiL_of_h k hh]
    ring
  rw [F.isUnit_onePlusKLAll_iff k, (F.lat F.μ).isUnit_onePlusKR_iff k 1,
    (F.lat F.μ).isUnit_onePlusKR_iff k (-1)]
  constructor
  · intro hL
    refine ⟨fun w hw hχ ↦ ?_, fun w hw hχ ↦ ?_⟩
    · have h0 := hL (fun p ↦ 1 * w p.2) ((summable_zOpL_iff_h0 hh).mpr fun τ ↦ by simpa using hw)
        fun σ n ↦ by
          convert hrow w hw (fun _ ↦ 1) σ n using 1
          rw [(hχ n).tsum_eq]
          cases σ <;> simp [sL] <;> ring
      funext l
      simpa using congrFun h0 (true, l)
    · have hs : ∀ τ, ‖sL true τ‖ = 1 := fun τ ↦ by cases τ <;> simp [sL]
      have h0 := hL (fun p ↦ sL true p.1 * w p.2)
        ((summable_zOpL_iff_h0 hh).mpr fun τ ↦ by simpa [mul_div_assoc, hs] using hw)
        fun σ n ↦ by
          convert hrow w hw (sL true) σ n using 1
          rw [(hχ n).tsum_eq]
          simp [sL]
          ring
      funext l
      simpa [sL] using congrFun h0 (true, l)
  · rintro ⟨hd, hm⟩ v hv hχ
    have hv' := (summable_zOpL_iff_h0 hh).mp hv
    have hs : ∀ σ n, HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * v (τ, l)) 0 :=
      fun σ n ↦ by
        have h := hasSum_spin_of_prod (hχ σ n)
        exact h
    have hD := hd (fun l ↦ v (true, l) + 1 * v (false, l))
      (summable_add_mul_div_sq (hv' true) (hv' false) 1) fun n ↦ by
        convert (hs true n).add (hs false n) using 1
        · funext l
          simp only [Fintype.sum_bool, F.chiL_of_h k hh, sL]
          norm_num
          ring
        · simp
    have hM := hm (fun l ↦ v (true, l) + -1 * v (false, l))
      (summable_add_mul_div_sq (hv' true) (hv' false) (-1)) fun n ↦ by
        convert (hs true n).sub (hs false n) using 1
        · funext l
          simp only [Fintype.sum_bool, F.chiL_of_h k hh, sL]
          norm_num
          ring
        · simp
    funext ⟨τ, l⟩
    have h1 := congrFun hD l
    have h2 := congrFun hM l
    simp only [Pi.zero_apply] at h1 h2 ⊢
    cases τ
    · linear_combination (h1 - h2) / 2
    · linear_combination (h1 + h2) / 2

/-! ### The inverse and the vertex at `h = 0` -/

variable (F k) in
/-- `Y_στ = (χ_d⁻¹ + s_στχ_m⁻¹)/2`, with the inverses `χ_r⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)` of the
zero-field atom on the lattice `z'` (`chiInvR`), for every `μ`. -/
noncomputable def invLh0R (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) : ℂ :=
  ((F.lat F.μ).chiInvR k 1 n m + sL σ t * (F.lat F.μ).chiInvR k (-1) n m) / 2

/-- **`Y` is a right inverse of `χ_L`** at `h = 0`, for every `μ`, where `1 + 𝒦_d` and `1 + 𝒦_m`
are invertible: `∑_{l, τ} χ_L,στ(n, l)Y_τt(l, m) = δ_σtδ_nm`. -/
theorem hasSum_chiL_mul_invLh0R (hh : F.h = 0) (hd : IsUnit ((F.lat F.μ).onePlusKR k 1))
    (hm : IsUnit ((F.lat F.μ).onePlusKR k (-1))) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * F.invLh0R k τ l t m)
      (if σ = t then kron n m else 0) := by
  have h₁ := (F.lat F.μ).chiInvR_isRightInverse k 1 hd n m
  have h₂ := (F.lat F.μ).chiInvR_isRightInverse k (-1) hm n m
  have h := (h₁.add (h₂.mul_left (sL σ t))).div_const 2
  convert h using 1
  · funext l
    simp only [Fintype.sum_bool, F.chiL_of_h k hh, invLh0R]
    cases σ <;> cases t <;>
      simp only [sL, Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte] <;> ring
  · cases σ <;> cases t <;> simp [sL]

/-- **`Y` is a left inverse of `χ_L`** at `h = 0`, for every `μ`, where `1 + 𝒦_d` and `1 + 𝒦_m`
are invertible: `∑_{l, τ} Y_στ(n, l)χ_L,τt(l, m) = δ_σtδ_nm`. -/
theorem hasSum_invLh0R_mul_chiL (hh : F.h = 0) (hd : IsUnit ((F.lat F.μ).onePlusKR k 1))
    (hm : IsUnit ((F.lat F.μ).onePlusKR k (-1))) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.invLh0R k σ n τ l * F.chiL τ t k l m)
      (if σ = t then kron n m else 0) := by
  have h₁ := (F.lat F.μ).chiInvR_isLeftInverse k 1 hd n m
  have h₂ := (F.lat F.μ).chiInvR_isLeftInverse k (-1) hm n m
  have h := (h₁.add (h₂.mul_left (sL σ t))).div_const 2
  convert h using 1
  · funext l
    simp only [Fintype.sum_bool, F.chiL_of_h k hh, invLh0R]
    cases σ <;> cases t <;>
      simp only [sL, Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte] <;> ring
  · cases σ <;> cases t <;> simp [sL]

/-- **`Y` is the only right inverse in the class**: at `h = 0`, for every `μ`, where `1 + 𝒦_d`
and `1 + 𝒦_m` are invertible, a column `(t, m)` with `∑_l |Y'_τl/z'_l|² < ∞` for both spins and
`∑_{l, τ} χ_L,στ(n, l)Y'_τl = δ_σtδ_nm` is the column of `Y`. -/
theorem eq_invLh0R_of_hasSum (hh : F.h = 0) (hd : IsUnit ((F.lat F.μ).onePlusKR k 1))
    (hm : IsUnit ((F.lat F.μ).onePlusKR k (-1))) {t : Bool} {m : ℤ} {Y : Bool → ℤ → ℂ}
    (hY : ∀ τ, Summable fun l ↦ ‖Y τ l / (F.lat F.μ).zOp k l‖ ^ 2)
    (hχ : ∀ σ n, HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * Y τ l)
      (if σ = t then kron n m else 0)) (σ : Bool) (n : ℤ) :
    Y σ n = F.invLh0R k σ n t m := by
  -- `Y↑ ± Y↓` are columns of `χ_d⁻¹` and `±χ_m⁻¹`
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
  have eD := fun n ↦ (F.lat F.μ).eq_chiInvR_of_hasSum k 1 hd
    (summable_add_mul_div_sq (hY true) (hY false) 1) hD n
  have eM : ∀ n, Y true n + -1 * Y false n = sL true t * (F.lat F.μ).chiInvR k (-1) n m := by
    intro n
    have hst : sL true t * sL true t = 1 := by cases t <;> simp [sL]
    have hM' : ∀ n, HasSum (fun l ↦ (F.lat F.μ).chiR k (-1) n l *
        (sL true t * (Y true l + -1 * Y false l))) (kron n m) := fun n ↦ by
      convert (hM n).mul_left (sL true t) using 1
      · funext l
        ring
      · rw [← mul_assoc, hst, one_mul]
    have hcls : Summable fun l ↦ ‖sL true t * (Y true l + -1 * Y false l) /
        (F.lat F.μ).zOp k l‖ ^ 2 := by
      have hs1 : ‖sL true t‖ = 1 := by cases t <;> simp [sL]
      simpa [mul_div_assoc, norm_mul, hs1] using
        summable_add_mul_div_sq (hY true) (hY false) (-1)
    have e := (F.lat F.μ).eq_chiInvR_of_hasSum k (-1) hm hcls hM' n
    rw [← e, ← mul_assoc, hst, one_mul]
  have e1 := eD n
  have e2 := eM n
  simp only [invLh0R]
  cases σ
  · have : Y false n = ((Y true n + 1 * Y false n) - (Y true n + -1 * Y false n)) / 2 := by ring
    rw [this, e1, e2]
    cases t
    · simp [sL]
    · simp [sL]
      ring
  · have : Y true n = ((Y true n + 1 * Y false n) + (Y true n + -1 * Y false n)) / 2 := by ring
    rw [this, e1, e2]

/-- **`χ_L⁻¹` at `h = 0` is `Y`**, for every `μ`: where `1 + 𝒦_d` and `1 + 𝒦_m` of the zero-field
atom on `z'` are invertible, `χ_L⁻¹ = (χ_d⁻¹ + s_στχ_m⁻¹)/2` (`invLh0R`), entrywise. -/
theorem chiInvLAll_eq_invLh0R (hh : F.h = 0) (hd : IsUnit ((F.lat F.μ).onePlusKR k 1))
    (hm : IsUnit ((F.lat F.μ).onePlusKR k (-1))) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.chiInvLAll k σ n t m = F.invLh0R k σ n t m := by
  have hK := (isUnit_onePlusKLAll_iff_onePlusKR_h0 hh).mpr ⟨hd, hm⟩
  exact eq_invLh0R_of_hasSum hh hd hm (Y := fun τ l ↦ F.chiInvLAll k τ l t m)
    (fun τ ↦ by simpa only [BL_of_h hh] using F.summable_chiInvLAll_zOp k t m τ)
    (fun σ n ↦ hasSum_chiL_mul_chiInvLAll hK σ n t m) σ n

/-- **The identity behind Eq. (field-h0)**, for every `μ`: `β²(Y - b⁻¹)_στ = (Γ_d + s_στΓ_m)/2`,
entrywise, with `Γ_r` of the zero-field atom on `z'` (`gammaOpR`) and `bₙ = B(νₙ)`. Both sides are
defined through `Ring.inverse`; where `1 + 𝒦_d` and `1 + 𝒦_m` are invertible, `Y` is the inverse
of `χ_L`. -/
theorem gammaL_h0R (hh : F.h = 0) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    (F.β : ℂ) ^ 2 * (F.invLh0R k σ n t m - if σ = t then kron n m / F.bL σ k n else 0) =
      ((F.lat F.μ).gammaOpR k 1 n m + sL σ t * (F.lat F.μ).gammaOpR k (-1) n m) / 2 := by
  rw [Params.gammaOpR_eq_chiInvR, Params.gammaOpR_eq_chiInvR, F.bL_of_h k hh, invLh0R]
  have hβ : (F.lat F.μ).β = F.β := rfl
  rw [hβ]
  cases σ <;> cases t <;> simp [sL] <;> ring

/-- **The operator vertex at `h = 0`, Eq. (field-h0), for every `μ`**: where `1 + 𝒦_d` and
`1 + 𝒦_m` of the zero-field atom `(U, β, μ)` on `z'` are invertible (equivalently `1 + 𝒦`,
`isUnit_onePlusKLAll_iff_onePlusKR_h0`), `Γ_L,στ = (Γ_d + s_στΓ_m)/2`, entrywise, with the
operator vertices `gammaOpR` of `DMOperator.lean`. -/
theorem gammaOpLAll_h0R (hh : F.h = 0) (hd : IsUnit ((F.lat F.μ).onePlusKR k 1))
    (hm : IsUnit ((F.lat F.μ).onePlusKR k (-1))) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      ((F.lat F.μ).gammaOpR k 1 n m + sL σ t * (F.lat F.μ).gammaOpR k (-1) n m) / 2 := by
  rw [gammaOpLAll_eq_chiInvLAll, chiInvLAll_eq_invLh0R hh hd hm]
  exact gammaL_h0R hh σ n t m

/-! ### The Shiba map and `μ = 0` for every `h` -/

/-- `χ_L⁻¹ = 0` where `1 + 𝒦` is not invertible. -/
theorem chiInvLAll_of_not_isUnit (hK : ¬IsUnit (F.onePlusKLAll k)) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) : F.chiInvLAll k σ n t m = 0 := by
  have h : Ring.inverse ((F.opLatLAll k).onePlusKX (F.chiLe k) (F.chiDecay_chiLe_all k)) = 0 :=
    Ring.inverse_non_unit _ hK
  simp [chiInvLAll, OpLattice.chiInvX, h]

/-- **`χ_L⁻¹` is the only right inverse in the class**, for every `μ`, `h`: where `1 + 𝒦` is
invertible, a column `(t, m)` on `Bool × ℤ` with `∑_{τl} |v_τl/z'_τl|² < ∞` and
`∑_{τl} χ_L,στ(n, l)v_τl = δ_σtδ_nm` is the column of `χ_L⁻¹`. -/
theorem eq_chiInvLAll_of_hasSum (hK : IsUnit (F.onePlusKLAll k)) {t : Bool} {m : ℤ}
    {v : Bool × ℤ → ℂ} (hv : Summable fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2)
    (hχ : ∀ σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p)
      (if σ = t then kron n m else 0)) (σ : Bool) (n : ℤ) :
    v (σ, n) = F.chiInvLAll k σ n t m := by
  have h := (F.opLatLAll k).eq_chiInvX_of_hasSum (F.chiLe k) (F.chiDecay_chiLe_all k) hK
    (m := encL (t, m)) (v := fun j ↦ v (encL.symm j)) ((F.summable_opLatLAll_iff k).mpr hv)
    (fun i ↦ by
      obtain ⟨⟨σ', n'⟩, rfl⟩ := encL.surjective i
      rw [kron_encL]
      exact (F.hasSum_row_iff k σ' n').mpr (hχ σ' n')) (encL (σ, n))
  simpa [chiInvLAll] using h

variable (F k) in
/-- **`χ_L⁻¹` under the Shiba map**, for every `μ`, `h`:
`χ_L⁻¹_στ(n, m) = ε_σε_τ (χ_L^shiba)⁻¹_στ(R_σn, R_τm)`, entrywise, with the Shiba image
`(-U, β, h, μ)`. Where `1 + 𝒦` is invertible, the reflected column of the image is a right inverse
in the class (`hasSum_row_shiba`, `norm_BL_z_shiba`); elsewhere both sides are `0`
(`isUnit_onePlusKLAll_shiba`). -/
theorem chiInvLAll_shiba (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.chiInvLAll k σ n t m = εL σ * εL t * F.shiba.chiInvLAll k σ (RL k σ n) t (RL k t m) := by
  by_cases hK : IsUnit (F.onePlusKLAll k)
  · have hK' := (F.isUnit_onePlusKLAll_shiba k).mp hK
    have hε : ∀ τ, ‖εL τ‖ = 1 := fun τ ↦ by cases τ <;> simp [εL]
    -- the column `(t, R_t m)` of the image, reflected back
    set w : Bool × ℤ → ℂ := fun p ↦ F.shiba.chiInvLAll k p.1 p.2 t (RL k t m)
    have hv : Summable fun p : Bool × ℤ ↦
        ‖εL t * εL p.1 * w (p.1, RL k p.1 p.2) / (F.BL p.1).zOp k p.2‖ ^ 2 := by
      rw [F.summable_zOpL_iff k]
      refine (summable_prod_iff_spin (v := fun p ↦ εL t * εL p.1 * w (p.1, RL k p.1 p.2))
        (z := fun τ ↦ (F.BL τ).z k)).mpr fun τ ↦ ?_
      refine ((RLe k τ).summable_iff.mpr (F.shiba.summable_chiInvLAll k t (RL k t m) τ)).congr
        fun l ↦ ?_
      simp only [Function.comp_apply, RLe, Equiv.coe_fn_mk, w, norm_div, norm_mul, hε, one_mul,
        F.norm_BL_z_shiba k τ l]
    have e := eq_chiInvLAll_of_hasSum hK (t := t) (m := m)
      (v := fun p ↦ εL t * εL p.1 * w (p.1, RL k p.1 p.2)) hv (fun σ' n' ↦ by
        refine (F.hasSum_row_shiba k σ' n').mpr ?_
        have h := (chiInvLAll_isRightInverse hK' σ' (RL k σ' n') t (RL k t m)).mul_left (εL t)
        convert h using 1
        · funext p
          simp only [w, RL_RL]
          linear_combination (εL t * F.shiba.chiL σ' p.1 k (RL k σ' n') p.2 *
            F.shiba.chiInvLAll k p.1 p.2 t (RL k t m)) * εL_mul_self p.1
        · by_cases hσ : σ' = t
          · rw [ite_eq_left hσ, ite_eq_left hσ, hσ, kron_RL, RL_RL]
          · rw [ite_eq_right hσ, ite_eq_right hσ, mul_zero, mul_zero]) σ n
    simp only [w] at e
    rw [← e]
    ring
  · have hK' : ¬IsUnit (F.shiba.onePlusKLAll k) :=
      fun h ↦ hK ((F.isUnit_onePlusKLAll_shiba k).mpr h)
    rw [chiInvLAll_of_not_isUnit hK, chiInvLAll_of_not_isUnit hK', mul_zero]

variable (F k) in
/-- **`Γ_L` under the Shiba map**, for every `μ`, `h`:
`Γ_L,στ(n, m) = ε_σε_τ Γ_L,στ^shiba(R_σn, R_τm)`, entrywise, with the Shiba image `(-U, β, h, μ)`
(`chiInvLAll_shiba`, and `bL_shiba` for the bubble). -/
theorem gammaOpLAll_shiba (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      εL σ * εL t * F.shiba.gammaOpLAll k σ (RL k σ n) t (RL k t m) := by
  rw [gammaOpLAll_eq_chiInvLAll, gammaOpLAll_eq_chiInvLAll, F.chiInvLAll_shiba k,
    F.bL_shiba k σ n]
  have hβ : (F.shiba.β : ℂ) = F.β := rfl
  rw [hβ]
  by_cases hσ : σ = t
  · rw [ite_eq_left hσ, ite_eq_left hσ, ← hσ, kron_RL, RL_RL]
    linear_combination ((F.β : ℂ) ^ 2 * (kron n m / F.shiba.bL σ k (RL k σ n))) *
      εL_mul_self σ
  · rw [ite_eq_right hσ, ite_eq_right hσ]
    ring

/-- **`1 + 𝒦` at `μ = 0`, for every `h`**: `1 + 𝒦` is invertible iff `1 + 𝒦_d` and `1 + 𝒦_m` of
the zero-field source `D = (-U, β, h)` on the lattice `z'` are (`isUnit_onePlusKLAll_shiba`, and
`isUnit_onePlusKLAll_iff_onePlusKR_h0` for the Shiba image). -/
theorem isUnit_onePlusKLAll_iff_onePlusKR_mu0 (hμ0 : F.μ = 0) :
    IsUnit (F.onePlusKLAll k) ↔ IsUnit ((F.shiba.lat F.shiba.μ).onePlusKR k 1) ∧
      IsUnit ((F.shiba.lat F.shiba.μ).onePlusKR k (-1)) :=
  (F.isUnit_onePlusKLAll_shiba k).trans (isUnit_onePlusKLAll_iff_onePlusKR_h0 (F := F.shiba) hμ0)

/-- **The operator vertex at `μ = 0`, Eq. (field-mu0), for every `h`**: where `1 + 𝒦_d^D` and
`1 + 𝒦_m^D` of the zero-field source `D = (-U, β, h)` on `z'` are invertible (equivalently
`1 + 𝒦`, `isUnit_onePlusKLAll_iff_onePlusKR_mu0`),
`Γ_L,στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)`, entrywise, with the operator
vertices `gammaOpR` of `DMOperator.lean` (`gammaOpLAll_shiba`, and `gammaOpLAll_h0R` for the
image). -/
theorem gammaOpLAll_mu0R (hμ0 : F.μ = 0) (hd : IsUnit ((F.shiba.lat F.shiba.μ).onePlusKR k 1))
    (hm : IsUnit ((F.shiba.lat F.shiba.μ).onePlusKR k (-1))) (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      εL σ * εL t / 2 * ((F.shiba.lat F.shiba.μ).gammaOpR k 1 (RL k σ n) (RL k t m) +
        εL σ * εL t * (F.shiba.lat F.shiba.μ).gammaOpR k (-1) (RL k σ n) (RL k t m)) := by
  rw [F.gammaOpLAll_shiba k, gammaOpLAll_h0R (F := F.shiba) hμ0 hd hm]
  have hs : sL σ t = εL σ * εL t := by cases σ <;> cases t <;> simp [sL, εL]
  rw [hs]
  ring

/-! ### The corner `μ = h = 0` -/

/-- At `μ = 0` the zero-field atom is `(U, β, 0)`. -/
private theorem lat_mu_of_mu (hμ : F.μ = 0) : F.lat F.μ = F.lat 0 := by
  rw [hμ]

/-- **`1 + 𝒦` at the corner**: at `μ = h = 0`, `1 + 𝒦` is invertible iff `1 + 𝒦_d` and
`1 + 𝒦_m` of `(U, β, 0)` are. -/
theorem isUnit_onePlusKLAll_iff_corner (hμ : F.μ = 0) (hh : F.h = 0) :
    IsUnit (F.onePlusKLAll k) ↔
      IsUnit ((F.lat 0).onePlusKR k 1) ∧ IsUnit ((F.lat 0).onePlusKR k (-1)) := by
  rw [isUnit_onePlusKLAll_iff_onePlusKR_h0 hh, lat_mu_of_mu hμ]

/-- **`1 + 𝒦` is invertible at the corner** under the assumptions of Corollary "half filling" for
both channels of `(U, β, 0)`. -/
theorem isUnit_onePlusKLAll_corner (hμ : F.μ = 0) (hh : F.h = 0) {u u' : ℂ}
    (hd : (F.lat 0).HFHyp k 1 u) (hm : (F.lat 0).HFHyp k (-1) u') :
    IsUnit (F.onePlusKLAll k) :=
  (isUnit_onePlusKLAll_iff_corner hμ hh).mpr ⟨hd.isUnit_onePlusKR, hm.isUnit_onePlusKR⟩

/-- **`χ_L⁻¹` at the corner**: at `μ = h = 0`, under the assumptions of Corollary "half filling"
for both channels of `(U, β, 0)`, `χ_L⁻¹ = (invHF_d + s_στ invHF_m)/2`, entrywise. -/
theorem chiInvLAll_corner (hμ : F.μ = 0) (hh : F.h = 0) {u u' : ℂ}
    (hd : (F.lat 0).HFHyp k 1 u) (hm : (F.lat 0).HFHyp k (-1) u') (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    F.chiInvLAll k σ n t m =
      ((F.lat 0).invHF k 1 u n m + sL σ t * (F.lat 0).invHF k (-1) u' n m) / 2 := by
  have hd' := hd.isUnit_onePlusKR
  have hm' := hm.isUnit_onePlusKR
  rw [← lat_mu_of_mu hμ] at hd' hm'
  rw [chiInvLAll_eq_invLh0R hh hd' hm', invLh0R, lat_mu_of_mu hμ, hd.invHF_eq_chiInvR,
    hm.invHF_eq_chiInvR]

/-- **The longitudinal vertex at the corner `μ = h = 0`, Eq. (field-h0) with the closed forms**:
under the assumptions of Corollary "half filling" for both channels of the zero-field atom
`(U, β, 0)`, `1 + 𝒦` is invertible (`isUnit_onePlusKLAll_corner`) and
`Γ_L,στ = (Γ_d + s_στΓ_m)/2`, entrywise, with the closed forms `gammaHF` of Eq. (Gamma-hf). -/
theorem gammaOpLAll_corner (hμ : F.μ = 0) (hh : F.h = 0) {u u' : ℂ}
    (hd : (F.lat 0).HFHyp k 1 u) (hm : (F.lat 0).HFHyp k (-1) u') (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      ((F.lat 0).gammaHF k 1 u n m + sL σ t * (F.lat 0).gammaHF k (-1) u' n m) / 2 := by
  have hd' := hd.isUnit_onePlusKR
  have hm' := hm.isUnit_onePlusKR
  rw [← lat_mu_of_mu hμ] at hd' hm'
  rw [gammaOpLAll_h0R hh hd' hm', lat_mu_of_mu hμ, hd.gammaOpR_eq, hm.gammaOpR_eq]

/-- **The corner with a lattice zero in the density channel**: at `μ = h = 0`, with a lattice zero
`z_c² = ŝ` of the density channel of `(U, β, 0)` (`HFLZ`) and the magnetic channel under `HFHyp`,
`Γ_L,στ = (Γ_d + s_στΓ_m)/2` with `gammaHFlz` of Eq. (Gamma-hf-lz) and `gammaHF`. -/
theorem gammaOpLAll_corner_lz_d (hμ : F.μ = 0) (hh : F.h = 0) {c : ℤ} {u' : ℂ}
    (hd : (F.lat 0).HFLZ k 1 c) (hm : (F.lat 0).HFHyp k (-1) u') (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      ((F.lat 0).gammaHFlz k 1 c n m + sL σ t * (F.lat 0).gammaHF k (-1) u' n m) / 2 := by
  have hd' := hd.isUnit_onePlusKR
  have hm' := hm.isUnit_onePlusKR
  rw [← lat_mu_of_mu hμ] at hd' hm'
  rw [gammaOpLAll_h0R hh hd' hm', lat_mu_of_mu hμ, hd.gammaOpR_eq, hm.gammaOpR_eq]

/-- **The corner with a lattice zero in the magnetic channel**: at `μ = h = 0`, with the density
channel of `(U, β, 0)` under `HFHyp` and a lattice zero `z_c² = ŝ` of the magnetic channel
(`HFLZ`), `Γ_L,στ = (Γ_d + s_στΓ_m)/2` with `gammaHF` and `gammaHFlz` of Eq. (Gamma-hf-lz). -/
theorem gammaOpLAll_corner_lz_m (hμ : F.μ = 0) (hh : F.h = 0) {u : ℂ} {c : ℤ}
    (hd : (F.lat 0).HFHyp k 1 u) (hm : (F.lat 0).HFLZ k (-1) c) (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      ((F.lat 0).gammaHF k 1 u n m + sL σ t * (F.lat 0).gammaHFlz k (-1) c n m) / 2 := by
  have hd' := hd.isUnit_onePlusKR
  have hm' := hm.isUnit_onePlusKR
  rw [← lat_mu_of_mu hμ] at hd' hm'
  rw [gammaOpLAll_h0R hh hd' hm', lat_mu_of_mu hμ, hd.gammaOpR_eq, hm.gammaOpR_eq]

/-- **The corner with lattice zeros in both channels**: at `μ = h = 0`, with lattice zeros
`z_c² = ŝ_d`, `z_c'² = ŝ_m` of both channels of `(U, β, 0)` (`HFLZ`),
`Γ_L,στ = (Γ_d + s_στΓ_m)/2` with `gammaHFlz` of Eq. (Gamma-hf-lz) in both. -/
theorem gammaOpLAll_corner_lz (hμ : F.μ = 0) (hh : F.h = 0) {c c' : ℤ}
    (hd : (F.lat 0).HFLZ k 1 c) (hm : (F.lat 0).HFLZ k (-1) c') (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      ((F.lat 0).gammaHFlz k 1 c n m + sL σ t * (F.lat 0).gammaHFlz k (-1) c' n m) / 2 := by
  have hd' := hd.isUnit_onePlusKR
  have hm' := hm.isUnit_onePlusKR
  rw [← lat_mu_of_mu hμ] at hd' hm'
  rw [gammaOpLAll_h0R hh hd' hm', lat_mu_of_mu hμ, hd.gammaOpR_eq, hm.gammaOpR_eq]

/-- **The longitudinal vertex at the corner, Eq. (field-mu0) with the closed forms**: at
`μ = h = 0`, under the assumptions of Corollary "half filling" for both channels of the Shiba
source `D = (-U, β, 0)`, `Γ_L,στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)`, entrywise,
with the closed forms `gammaHF` of `D`. -/
theorem gammaOpLAll_corner_mu0 (hμ : F.μ = 0) (hh : F.h = 0) {u u' : ℂ}
    (hd : (F.shiba.lat 0).HFHyp k 1 u) (hm : (F.shiba.lat 0).HFHyp k (-1) u') (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      εL σ * εL t / 2 * ((F.shiba.lat 0).gammaHF k 1 u (RL k σ n) (RL k t m) +
        εL σ * εL t * (F.shiba.lat 0).gammaHF k (-1) u' (RL k σ n) (RL k t m)) := by
  have hl : F.shiba.lat F.shiba.μ = F.shiba.lat 0 := lat_mu_of_mu (F := F.shiba) hh
  have hd' := hd.isUnit_onePlusKR
  have hm' := hm.isUnit_onePlusKR
  rw [← hl] at hd' hm'
  rw [gammaOpLAll_mu0R hμ hd' hm', hl, hd.gammaOpR_eq, hm.gammaOpR_eq]

/-! ### Continuity at the corner -/

section Family

variable {P : Type*} [TopologicalSpace P] {U β μ h : P → ℝ} {hβ : ∀ x, 0 < β x} {x₀ : P}

/-- **`Γ_L` is continuous at the corner**: along a family `(U, β, μ, h)(x)` continuous at `x₀`
with `μ(x₀) = h(x₀) = 0`, if both channels of `(U(x₀), β(x₀), 0)` satisfy the assumptions of
Corollary "half filling", `Γ_L,(σn)(τm)` is continuous at `x₀`. -/
theorem continuousAt_gammaOpLAll_corner (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀)
    (hm : ContinuousAt μ x₀) (hh : ContinuousAt h x₀) (hμ₀ : μ x₀ = 0) (hh₀ : h x₀ = 0)
    {u u' : ℂ} (hd : ((famF U β μ h hβ x₀).lat 0).HFHyp k 1 u)
    (hmg : ((famF U β μ h hβ x₀).lat 0).HFHyp k (-1) u') (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).gammaOpLAll k σ n τ m) x₀ :=
  continuousAt_gammaOpLAll hU hb hm hh
    (isUnit_onePlusKLAll_corner (F := famF U β μ h hβ x₀) hμ₀ hh₀ hd hmg) σ n τ m

/-- **The closed form tends to Eq. (field-h0) at the corner**: along a family `(U, β, μ, h)(x)`
continuous at `x₀` with `μ(x₀) = h(x₀) = 0`, if both channels of `(U(x₀), β(x₀), 0)` satisfy the
assumptions of Corollary "half filling", then along any approach `x i → x₀` through points where
Eq. (field-L) holds, `gammaL_στ → (gammaHF_d + s_στ gammaHF_m)/2` of `(U(x₀), β(x₀), 0)`.
`LAxisGeneric`, which gives such approaches elsewhere on the axes, never holds at the corner
(`not_lAxisGeneric_corner`); they exist through the axis points near it
(`exists_tendsto_gammaL_corner`, `FieldLAxisCornerK.lean`). -/
theorem tendsto_gammaL_corner (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀)
    (hm : ContinuousAt μ x₀) (hh : ContinuousAt h x₀) (hμ₀ : μ x₀ = 0) (hh₀ : h x₀ = 0)
    {u u' : ℂ} (hd : ((famF U β μ h hβ x₀).lat 0).HFHyp k 1 u)
    (hmg : ((famF U β μ h hβ x₀).lat 0).HFHyp k (-1) u') {ι : Type*} {l : Filter ι}
    {x : ι → P} {v : ι → Fin 4 → ℂ} (hx : Tendsto x l (𝓝 x₀))
    (hg : ∀ᶠ i in l, ∀ t, (famF U β μ h hβ (x i)).LGeneric k t (v i)) (σ : Bool) (n : ℤ)
    (τ : Bool) (m : ℤ) :
    Tendsto (fun i ↦ (famF U β μ h hβ (x i)).gammaL k (v i) σ n τ m) l
      (𝓝 ((((famF U β μ h hβ x₀).lat 0).gammaHF k 1 u n m +
        sL σ τ * ((famF U β μ h hβ x₀).lat 0).gammaHF k (-1) u' n m) / 2)) := by
  rw [← gammaOpLAll_corner (F := famF U β μ h hβ x₀) hμ₀ hh₀ hd hmg σ n τ m]
  exact tendsto_gammaL_of_LGeneric hU hb hm hh
    (isUnit_onePlusKLAll_corner (F := famF U β μ h hβ x₀) hμ₀ hh₀ hd hmg) hx hg σ n τ m

end Family

end FParams

end HubbardAtom
