import HubbardAtomLean.FieldLAxisFormula
import HubbardAtomLean.FieldLAxisValues

/-!
# `1 + 𝒦` and `Γ_L` on the axes `μ = ±h ≠ 0`

On the axis `μ_t = μ + th = 0` (`μ = h` for `t = ↓`, `μ = -h` for `t = ↑`), `μ ≠ 0`, the column
spin `t` of Eq. (field-L) is `0/0` (`D_t = 0`), but the column spin `-t` can be regular. Here the
regular columns and the spin-`t` block of `χ_L` together show that `1 + 𝒦` of
`FieldLAxisOperator.lean` is invertible where `LGeneric k (!t) u` holds at the point (and, at
`k = 0`, a condition that is also necessary), so that `Γ_L` exists there, and then, with
`FieldLAxisFormula.lean`, that it is the axis formula `gammaLAll`. Not in the note.

* `OpLattice.eq_zero_of_null_of_col`: for a symmetric `X`, a bounded `y` with `Xy = eₘ` makes every
  null vector `v` of `X` in the class vanish at `m` (the one-column form of
  `isUnit_onePlusKX_of_isRightInverse`).
* `FParams.LGeneric.null_not_eq_zero`: where `LGeneric k (!t) u` holds at the point (for every
  `μ_t`, `μ_t = 0` included), the bounded columns `(-t, m)` of `invL` (`bdd_invL`) are
  right-inverse columns of `χ_L` (`LGeneric.hasSum_row_self`, `LGeneric.hasSum_row_opp`), so a
  null vector of `χ_L` in the class vanishes on spin `-t`.
* The spin-`t` block, for every `μ`, `h`, by `PL_mul_chiL_self`: for `k ≠ 0` it is diagonal and has
  no null vector where `H_t(zₙ) ≠ 0` (`eq_zero_of_chiL_self_of_ne`). At `k = 0` it is diagonal plus
  rank one: a null vector is `Sw⁰` with `w⁰ₙ = ζ_t(zₙ² - a²)/(βH_t(zₙ))` (`w0L`) and
  `S(1 - T_t) = 0` (`eq_smul_w0L_of_chiL_self`), `T_t = (ζ_t/β) ∑ₙ 1/H_t(zₙ)` (`TL`, a convergent
  sum, `summable_inv_HL`), and `w⁰` is one where `T_t = 1` (`hasSum_chiL_self_w0L`).
* `FParams.LGeneric.isUnit_onePlusKLAll_of_HL_ne`: for every `μ_t`, `μ_t = 0` included,
  `LGeneric k (!t) u`, `H_t(zₙ) ≠ 0` on the lattice of `t`, and, at `k = 0`, `1 - T_t ≠ 0` or a row
  of spin `-t` that does not vanish on `w⁰`, give that `1 + 𝒦` is invertible. At `k = 0` the
  condition is also necessary (`not_isUnit_onePlusKLAll_of_TL_eq_one`: otherwise `(w⁰, 0)` is a
  null vector in the class; `LGeneric.isUnit_onePlusKLAll_iff_of_HL_ne`).
* `FParams.HL_ne_of_cL`: on `μ_t = 0`, `μ ≠ 0`, `U ≠ 0`, `H_t(zₙ) ≠ 0` on the lattice of `t`: there
  `zₙ = iy` and `Im H_t(iy) = 2ar_ty`, with `r_t = p₀(1 + e^{βa})(e^{2βμ} - 1) ≠ 0`
  (`rL_eq_of_cL`), and `H_t(0) = q₀ ≠ 0`.
* `FParams.TL_eq_of_cL`: on `μ_t = 0`, `T_t = aβ[tanh(βx₋/2) - tanh(βx₊/2)]/(4i cosh(βμ))` with
  `x± = -a tanh(βμ) ± ia/cosh(βμ)`, the roots of `H_t`, as `r_t = tanh(βμ)` (`rL_eq_tanh_of_cL`)
  and `ζ_t/β = a²/cosh²(βμ)` (`ζLs_div_β_of_cL`); so `T_t` depends on `U` only through `a²`.
* `FParams.isUnit_onePlusKLAll_of_cL_eq_zero`: **`1 + 𝒦` is invertible on `μ = ±h ≠ 0` under
  `LGeneric` for the regular column spin**: where `LGeneric k (!t) u` holds at the point and, at
  `k = 0`, the condition above; at `k = 0` it is iff (`isUnit_onePlusKLAll_iff_of_cL_eq_zero`).
* `FParams.gammaOpLAll_eq_gammaLAll_of_cL_eq_zero`: **there `Γ_L = gammaLAll`**, if in addition the
  condition of `LAxisGeneric` for the column spin `t` holds (`LGeneric.lAxisGeneric_of_not`:
  `LGeneric k (!t) u` already gives the rest of `LAxisGeneric`). This proves on `μ = ±h ≠ 0` the
  agreement that the note states as numerically verified, under these assumptions.

Assumed: `LGeneric k (!t) u` at the axis point (the roots of `E₈` there, off both lattices, with
`K_{-t}(u_j) ≠ 0`), and the condition at `k = 0`. `1 - T_t` changes sign along the axis
(numerically, e.g. at `β = 2`, `U = ±10`), but at the zero of `1 - T_t` that was checked
(`β = 2`, `U = 10`, `μ = h ≈ 0.0878`) a row of spin `-t` does not vanish on `w⁰`, and `1 + 𝒦` is
invertible there. Not proved: invertibility on `μ = ±h ≠ 0` where `LGeneric k (!t) u` fails, and
that `LGeneric k (!t) u` holds at given axis points (where it holds and the condition at `k = 0`
fails, `1 + 𝒦` is not invertible). `Γ_L` does not exist at every point of `μ = ±h ≠ 0`:
numerically `1 + 𝒦` is singular at a point with `K_{-t}(u_j) = 0` (`β = 2`, `U = 10`,
`μ = h ≈ 0.1178`, where `1 - T_t ≈ 0.649`).
-/

open Complex
open scoped lp

namespace HubbardAtom

namespace OpLattice

open Params (single_apply_eq_kron)

variable (L : OpLattice) (X : ℤ → ℤ → ℂ)

/-- **A bounded right-inverse column kills null vectors at its index**: for a symmetric `X` and a
bounded `y` with `∑ₗ X_nl yₗ = δₙₘ`, every null vector `v` of `X` with `∑ₙ |vₙ/zₙ|² < ∞` has
`vₘ = 0`: `cₘ = 𝒟^(1/2)y dₘ` has `(1 + 𝒦)cₘ = eₘ`, and `x = 𝒟^(1/2)v` has `(1 + 𝒦)x = 0`, so
`xₘ = ∑ₙ xₙ((1 + 𝒦)cₘ)ₙ = ∑ₙ cₘ,ₙ((1 + 𝒦)x)ₙ = 0`. The one-column form of
`isUnit_onePlusKX_of_isRightInverse`. -/
theorem eq_zero_of_null_of_col (hX : L.ChiDecay X) (hs : ∀ n m, X n m = X m n) {m : ℤ}
    {y : ℤ → ℂ} {C : ℝ} (hC : ∀ n, ‖y n‖ ≤ C)
    (hy : ∀ n, HasSum (fun l ↦ X n l * y l) (kron n m)) {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / L.z n‖ ^ 2) (hχ : ∀ n, HasSum (fun l ↦ X n l * v l) 0) :
    v m = 0 := by
  have hyv : Summable fun n ↦ ‖y n * L.Dhalf m / L.z n‖ ^ 2 :=
    L.summable_col_of_bdd (Y := fun n _ ↦ y n) (m := m) hC (L.Dhalf m)
  have hc : L.onePlusKX X hX (L.DhalfVec hyv) = lp.single 2 m 1 := by
    ext n
    have h₁ := L.hasSum_X_mul X hX hyv n
    have h₂ := (hy n).mul_right (L.Dhalf m)
    simp only [mul_assoc] at h₂
    have h₃ := h₁.unique h₂
    have hd := L.Dhalf_ne_zero n
    rw [single_apply_eq_kron]
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self, one_mul] at h₃
      rw [kron_self]
      exact mul_left_cancel₀ hd (h₃.trans (mul_one _).symm)
    · rw [kron_of_ne hnm, zero_mul] at h₃
      rw [kron_of_ne hnm]
      exact (mul_eq_zero.mp h₃).resolve_left hd
  have hpair := tsum_mul_one_add_hsOp_comm (L.KX X) (L.summable_KX_sq X hX)
    (L.KX_symm X hs) (L.DhalfVec hv) (L.DhalfVec hyv)
  rw [show (1 + hsOp (L.KX X) (L.summable_KX_sq X hX)) = L.onePlusKX X hX from rfl,
    hc, L.onePlusKX_DhalfVec_eq_zero X hX hv hχ] at hpair
  simp only [single_apply_eq_kron, lp.coeFn_zero, Pi.zero_apply, mul_zero, tsum_zero, kron,
    mul_ite, mul_one] at hpair
  rw [tsum_ite_eq m] at hpair
  have h : L.Dhalf m * v m = 0 := hpair
  exact (mul_eq_zero.mp h).resolve_left (L.Dhalf_ne_zero m)

end OpLattice

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### Rows of `χ_L` against bounded vectors -/

/-- A row of `χ_L` against a bounded `y` on `Bool × ℤ` is absolutely summable, for every `μ`, `h`:
the bounds of `chiDecay_chiLe_all`. -/
theorem summable_chiL_row_of_bdd {y : Bool × ℤ → ℂ} {C : ℝ} (hC : ∀ p, ‖y p‖ ≤ C) (σ : Bool)
    (n : ℤ) : Summable fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * y p := by
  have h := (F.opLatLAll k).summable_norm_X_mul (F.chiLe k) (F.chiDecay_chiLe_all k)
    (v := fun l ↦ y (encL.symm l)) (C := C) (fun l ↦ hC _) (encL (σ, n))
  refine Summable.of_norm (encL.summable_iff.mpr h |>.congr fun p ↦ ?_)
  simp [chiLe]

/-- A row of `χ_L`, summed spin by spin, against a bounded `y` on `Bool × ℤ`, is the row of `chiLe`
on the enumerated lattice, for every `μ`, `h` (`summable_chiL_row_of_bdd`). The one-column form of
`isRightInverse_chiLe_of`. -/
theorem hasSum_chiLe_of_spin {y : Bool × ℤ → ℂ} {C : ℝ} (hC : ∀ p, ‖y p‖ ≤ C) (σ : Bool) (n : ℤ)
    {a : ℂ} (h : HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * y (τ, l)) a) :
    HasSum (fun j ↦ F.chiLe k (encL (σ, n)) j * y (encL.symm j)) a := by
  have hs := F.summable_chiL_row_of_bdd k hC σ n
  have h₂ := hasSum_spin_of_prod hs.hasSum
  have h₄ : HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * y p) a := h₂.unique h ▸ hs.hasSum
  exact (F.hasSum_row_iff k σ n).mpr h₄

variable {F k}

/-- A row of `χ_L` against a `v` that vanishes on spin `-t` is its sum over spin `t`. -/
private theorem hasSum_row_iff_of_not_eq_zero {t : Bool} {v : Bool × ℤ → ℂ}
    (hv : ∀ m, v (!t, m) = 0) {σ : Bool} {n : ℤ} {a : ℂ} :
    HasSum (fun l ↦ F.chiL σ t k n l * v (t, l)) a ↔
      HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) a := by
  have hinj : Function.Injective (fun l : ℤ ↦ (t, l)) := fun _ _ h ↦ (Prod.mk.inj h).2
  refine hinj.hasSum_iff (f := fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) fun p hp ↦ ?_
  obtain ⟨τ, l⟩ := p
  have hτ : τ = !t := by
    by_contra hne
    exact hp ⟨l, by cases τ <;> cases t <;> simp_all⟩
  simp only [hτ, hv, mul_zero]

/-! ### The spin `-t` entries of a null vector -/

/-- **The spin `-t` entries of a null vector vanish** where the column spin `-t` is regular at the
point (`LGeneric k (!t) u`), for every `μ_t`, `μ_t = 0` included: the columns `(-t, m)` of `invL`
are bounded (`bdd_invL`) right-inverse columns of `χ_L` (`LGeneric.hasSum_row_self`,
`LGeneric.hasSum_row_opp`), and `χ_L` is symmetric (`OpLattice.eq_zero_of_null_of_col`). -/
theorem LGeneric.null_not_eq_zero {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k (!t) u)
    {v : Bool × ℤ → ℂ} (hv : Summable fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2)
    (hχ : ∀ σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) 0) (m : ℤ) :
    v (!t, m) = 0 := by
  obtain ⟨C, hC⟩ := F.bdd_invL u (!t) m
  have hy : ∀ i, HasSum (fun j ↦ F.chiLe k i j *
      F.invL k u (encL.symm j).1 (encL.symm j).2 (!t) m) (kron i (encL (!t, m))) := by
    intro i
    obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
    rw [kron_encL]
    refine F.hasSum_chiLe_of_spin k (y := fun p ↦ F.invL k u p.1 p.2 (!t) m)
      (fun p ↦ hC _ _) σ n ?_
    by_cases hσ : σ = !t
    · subst hσ
      simpa using hg.hasSum_row_self n m
    · obtain rfl : σ = t := by cases σ <;> cases t <;> simp_all
      simpa using hg.hasSum_row_opp n m
  have h := (F.opLatLAll k).eq_zero_of_null_of_col (F.chiLe k) (F.chiDecay_chiLe_all k)
    (F.chiLe_symm_all k) (fun j ↦ hC _ _) hy (v := fun j ↦ v (encL.symm j))
    ((F.summable_opLatLAll_iff k).mpr hv) fun i ↦ by
      obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
      exact (F.hasSum_row_iff k σ n).mpr (hχ σ n)
  simpa using h

/-! ### The spin-`t` block -/

variable (F k)

/-- `ζ_σ ≠ 0` for `U ≠ 0`. -/
private theorem ζLs_ne (ha : (F.a : ℂ) ≠ 0) (σ : Bool) : F.ζLs σ ≠ 0 := by
  have ha' : F.a ≠ 0 := by exact_mod_cast ha
  have h0 := F.p₀_pos
  have h2 := F.p₂_pos
  have hs := F.pL_pos σ
  have hn := F.pL_pos (!σ)
  have hβ := F.β_pos
  have e : F.ζLs σ = ((4 * F.β * F.a ^ 2 * (F.p₀ + F.pL σ) * (F.pL (!σ) + F.p₂) : ℝ) : ℂ) := by
    simp only [ζLs]
    push_cast
    ring
  rw [e, ofReal_ne_zero]
  positivity

/-- At `k = 0`, `zₙ² - a² ≠ 0` on every lattice, as `P(zₙ) = (zₙ² - a²)² ≠ 0`. -/
private theorem sq_sub_ne (t : Bool) (n : ℤ) : (F.BL t).z 0 n ^ 2 - F.a ^ 2 ≠ 0 := fun h ↦
  F.PL_BL_ne 0 t n (by rw [F.PL_zero, h]; ring)

/-- `∑ₙ 1/H_r(zₙ)` converges on every lattice: `H_r(z) = (z - x₁)(z - x₂)` is quadratic. -/
theorem summable_inv_HL (r c : ℝ) : Summable fun n ↦ (F.HL k r ((F.lat c).z k n))⁻¹ := by
  obtain ⟨s, hs⟩ := IsAlgClosed.exists_pow_nat_eq ((F.a * r : ℂ) ^ 2 - F.q₀L k) two_pos
  have hc : (2 * Real.pi / F.β * I : ℂ) ≠ 0 :=
    mul_ne_zero (div_ne_zero (mul_ne_zero two_ne_zero (ofReal_ne_zero.mpr Real.pi_ne_zero))
      (ofReal_ne_zero.mpr F.β_pos.ne')) I_ne_zero
  refine (summable_inv_mul_linear hc (c + (1 + k) * Real.pi / F.β * I + F.a * r - s)
    (c + (1 + k) * Real.pi / F.β * I + F.a * r + s)).congr fun n ↦ ?_
  congr 1
  rw [HL, Params.z_eq, lat_μ, lat_β]
  push_cast
  linear_combination (-1 : ℂ) * hs

/-- **`T_t = (ζ_t/β) ∑ₙ 1/H_t(zₙ)`** at `k = 0`, on the lattice of `t` (a convergent sum,
`summable_inv_HL`): a null vector `Sw⁰` of the spin-`t` block at `k = 0` has `S = T_tS`
(`eq_smul_w0L_of_chiL_self`). Not in the note. -/
noncomputable def TL (t : Bool) : ℂ :=
  F.ζLs t / F.β * ∑' n, (F.HL 0 (F.rL t) ((F.BL t).z 0 n))⁻¹

/-- **The vector `w⁰` of the spin-`t` block at `k = 0`**: `w⁰ₙ = ζ_t(zₙ² - a²)/(βH_t(zₙ))` on the
lattice of `t`. Where `H_t(zₙ) ≠ 0`, every null vector of the block is a multiple `Sw⁰` with
`S(1 - T_t) = 0` (`eq_smul_w0L_of_chiL_self`), and `w⁰` is one where `T_t = 1`
(`hasSum_chiL_self_w0L`). Not in the note. -/
noncomputable def w0L (t : Bool) (n : ℤ) : ℂ :=
  F.ζLs t * ((F.BL t).z 0 n ^ 2 - F.a ^ 2) / (F.β * F.HL 0 (F.rL t) ((F.BL t).z 0 n))

/-- `w⁰ₙ ≠ 0`. -/
private theorem w0L_ne {t : Bool} (ha : (F.a : ℂ) ≠ 0)
    (hH : ∀ n, F.HL 0 (F.rL t) ((F.BL t).z 0 n) ≠ 0) (n : ℤ) : F.w0L t n ≠ 0 :=
  div_ne_zero (mul_ne_zero (F.ζLs_ne ha t) (F.sq_sub_ne t n))
    (mul_ne_zero (ofReal_ne_zero.mpr F.β_pos.ne') (hH n))

/-- `w⁰` is bounded: `w⁰ₙ = q(1/zₙ)` with `q(y) = ζ_t(1 - a²y²)/(β(1 + 2ar_ty + q₀y²))` continuous
at `0`. -/
private theorem bdd_w0L (t : Bool) (hH : ∀ n, F.HL 0 (F.rL t) ((F.BL t).z 0 n) ≠ 0) :
    ∃ C, ∀ n, ‖F.w0L t n‖ ≤ C := by
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  obtain ⟨C, -, hC⟩ := F.bdd_inv_zL 0 t (fun y ↦ F.ζLs t * (1 - F.a ^ 2 * y ^ 2) /
      (F.β * (1 + 2 * F.a * F.rL t * y + F.q₀L 0 * y ^ 2)))
    (ContinuousAt.div (by fun_prop) (by fun_prop) (by simp [hβ]))
  refine ⟨C, fun n ↦ ?_⟩
  have hm : mirror 0 n ≠ n := by unfold mirror; omega
  have hz := (F.BL t).z_ne_zero_of_ne 0 hm
  have hH' := hH n
  convert hC n hm using 2
  rw [HL] at hH'
  rw [w0L, HL]
  field_simp

variable {F k}

/-- **The spin-`t` block at `k = 0`**: where `H_t(zₙ) ≠ 0` on the lattice of `t`, a `w` with
`∑ₗ χ_tt(n, l)wₗ = 0` for all `n` is `w = Sw⁰` with `S(1 - T_t) = 0`, for every `μ`, `h`. By
`PL_mul_chiL_self`, `P(zₙ)χ_tt(n, l) = -βH_t(zₙ)δₙₗ + ζ_t(zₙ² - a²)(zₗ² - a²)/P(zₗ)`, so
`wₙ = Sw⁰ₙ` with `S = ∑ₗ (zₗ² - a²)wₗ/P(zₗ)`, and then `S = T_tS` as `P(z) = (z² - a²)²`. -/
theorem eq_smul_w0L_of_chiL_self {t : Bool} (ha : (F.a : ℂ) ≠ 0)
    (hH : ∀ n, F.HL 0 (F.rL t) ((F.BL t).z 0 n) ≠ 0) {w : ℤ → ℂ}
    (hw : ∀ n, HasSum (fun l ↦ F.chiL t t 0 n l * w l) 0) :
    ∃ S : ℂ, S * (1 - F.TL t) = 0 ∧ ∀ n, w n = S * F.w0L t n := by
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  have hza := F.sq_sub_ne t
  have hζ := F.ζLs_ne ha t
  -- each row gives `∑ₗ gₗ = βH_t(zₙ)wₙ/(ζ_t(zₙ² - a²))`
  have hrow : ∀ n, HasSum (fun l ↦ ((F.BL t).z 0 l ^ 2 - F.a ^ 2) / F.PL 0 ((F.BL t).z 0 l) *
      w l) (F.β * F.HL 0 (F.rL t) ((F.BL t).z 0 n) * w n /
        (F.ζLs t * ((F.BL t).z 0 n ^ 2 - F.a ^ 2))) := by
    intro n
    have hB : F.ζLs t * ((F.BL t).z 0 n ^ 2 - F.a ^ 2) ≠ 0 := mul_ne_zero hζ (hza n)
    have h1 := (hw n).mul_left (F.PL 0 ((F.BL t).z 0 n))
    have h2 : HasSum (fun l ↦ -F.β * F.HL 0 (F.rL t) ((F.BL t).z 0 n) * kron n l * w l)
        (-F.β * F.HL 0 (F.rL t) ((F.BL t).z 0 n) * w n) := by
      convert hasSum_ite_eq n (-F.β * F.HL 0 (F.rL t) ((F.BL t).z 0 n) * w n) using 1
      funext l
      by_cases hl : l = n
      · subst hl
        simp
      · simp [kron_of_ne (Ne.symm hl), hl]
    have h3 := h1.sub h2
    rw [mul_zero, zero_sub] at h3
    refine (hasSum_mul_left_iff hB).mp ?_
    convert h3 using 1
    · funext l
      have hl := F.PL_mul_chiL_self 0 t n l
      rw [kron_self] at hl
      linear_combination (-w l) * hl
    · have := hza n
      field_simp
  set S := ∑' l, ((F.BL t).z 0 l ^ 2 - F.a ^ 2) / F.PL 0 ((F.BL t).z 0 l) * w l with hSdef
  have hS : ∀ n, w n = S * F.w0L t n := by
    intro n
    rw [w0L, hSdef, (hrow n).tsum_eq]
    have hB : F.ζLs t * ((F.BL t).z 0 n ^ 2 - F.a ^ 2) ≠ 0 := mul_ne_zero hζ (hza n)
    have := hH n
    have := hza n
    field_simp
  refine ⟨S, ?_, hS⟩
  have hg : HasSum (fun n ↦ F.ζLs t / F.β * (F.HL 0 (F.rL t) ((F.BL t).z 0 n))⁻¹ * S) S := by
    convert (hrow 0).summable.hasSum using 1
    funext n
    rw [hS n, w0L, F.PL_zero]
    have := hH n
    have := hza n
    field_simp
  by_cases hS0 : S = 0
  · rw [hS0, zero_mul]
  · have h1 : HasSum (fun n ↦ F.ζLs t / F.β * (F.HL 0 (F.rL t) ((F.BL t).z 0 n))⁻¹) 1 := by
      have := hg.div_const S
      simpa [mul_div_assoc, div_self hS0] using this
    rw [TL, ← tsum_mul_left, h1.tsum_eq, sub_self, mul_zero]

/-- **The spin-`t` block for `k ≠ 0` is diagonal**, `P(zₙ)χ_tt(n, l) = -βH_t(zₙ)δₙₗ`
(`PL_mul_chiL_self`), so where `H_t(zₙ) ≠ 0` on the lattice of `t` it has no null vector, for
every `μ`, `h`. -/
theorem eq_zero_of_chiL_self_of_ne {t : Bool} (hk : k ≠ 0)
    (hH : ∀ n, F.HL k (F.rL t) ((F.BL t).z k n) ≠ 0) {w : ℤ → ℂ}
    (hw : ∀ n, HasSum (fun l ↦ F.chiL t t k n l * w l) 0) : w = 0 := by
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  funext n
  have hP := F.PL_BL_ne k t n
  have hoff : ∀ l ≠ n, F.chiL t t k n l * w l = 0 := by
    intro l hl
    have h := F.PL_mul_chiL_self k t n l
    rw [kron_of_ne hk, kron_of_ne (Ne.symm hl)] at h
    simp only [mul_zero, zero_mul, zero_div, add_zero] at h
    rw [(mul_eq_zero.mp h).resolve_left hP, zero_mul]
  have h := (hasSum_single n hoff).unique (hw n)
  have hd := F.PL_mul_chiL_self k t n n
  rw [kron_self, kron_of_ne hk] at hd
  simp only [mul_one, zero_mul, zero_div, add_zero] at hd
  have hχ : F.chiL t t k n n =
      -F.β * F.HL k (F.rL t) ((F.BL t).z k n) / F.PL k ((F.BL t).z k n) := by
    rw [eq_div_iff hP]
    linear_combination hd
  rw [hχ] at h
  simpa [hβ, hH n, hP] using h

/-- **`w⁰` under the spin-`t` block**, at `k = 0`: where `H_t(zₙ) ≠ 0` on the lattice of `t`,
`∑ₗ χ_tt(n, l)w⁰ₗ = ζ_t(zₙ² - a²)(T_t - 1)/P(zₙ)`, so `w⁰` is a null vector of the block where
`T_t = 1`, for every `μ`, `h`. -/
theorem hasSum_chiL_self_w0L {t : Bool} (hH : ∀ n, F.HL 0 (F.rL t) ((F.BL t).z 0 n) ≠ 0)
    (n : ℤ) : HasSum (fun l ↦ F.chiL t t 0 n l * F.w0L t l)
      (F.ζLs t * ((F.BL t).z 0 n ^ 2 - F.a ^ 2) * (F.TL t - 1) / F.PL 0 ((F.BL t).z 0 n)) := by
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  have hP := F.PL_BL_ne 0 t n
  have hza := F.sq_sub_ne t
  have h1 : HasSum (fun l ↦ -F.β * F.HL 0 (F.rL t) ((F.BL t).z 0 n) * kron n l * F.w0L t l)
      (-F.β * F.HL 0 (F.rL t) ((F.BL t).z 0 n) * F.w0L t n) := by
    convert hasSum_ite_eq n (-F.β * F.HL 0 (F.rL t) ((F.BL t).z 0 n) * F.w0L t n) using 1
    funext l
    by_cases hl : l = n
    · subst hl
      simp
    · simp [kron_of_ne (Ne.symm hl), hl]
  have h2 : HasSum (fun l ↦ F.ζLs t * ((F.BL t).z 0 n ^ 2 - F.a ^ 2) *
      (F.ζLs t / F.β * (F.HL 0 (F.rL t) ((F.BL t).z 0 l))⁻¹))
      (F.ζLs t * ((F.BL t).z 0 n ^ 2 - F.a ^ 2) * F.TL t) := by
    rw [TL]
    exact ((F.summable_inv_HL 0 (F.rL t) (F.cL t)).hasSum.mul_left _).mul_left _
  have h3 := (h1.add h2).div_const (F.PL 0 ((F.BL t).z 0 n))
  convert h3 using 1
  · funext l
    have hl := F.PL_mul_chiL_self 0 t n l
    rw [kron_self] at hl
    have hid : ((F.BL t).z 0 l ^ 2 - F.a ^ 2) / F.PL 0 ((F.BL t).z 0 l) * F.w0L t l =
        F.ζLs t / F.β * (F.HL 0 (F.rL t) ((F.BL t).z 0 l))⁻¹ := by
      rw [w0L, F.PL_zero]
      have := hH l
      have := hza l
      field_simp
    rw [eq_div_iff hP]
    linear_combination (F.w0L t l) * hl + (F.ζLs t * ((F.BL t).z 0 n ^ 2 - F.a ^ 2)) * hid
  · rw [w0L]
    have := hH n
    have := hza n
    field_simp
    ring

/-! ### `1 + 𝒦` with one regular column spin -/

/-- **`1 + 𝒦` is invertible with one regular column spin**, for every `μ_t`, `μ_t = 0` included:
if the column spin `-t` satisfies the assumptions of Eq. (field-L) at the point
(`LGeneric k (!t) u`), `H_t(zₙ) ≠ 0` on the lattice of `t`, and, at `k = 0`, `1 - T_t ≠ 0` or a row
of spin `-t` does not vanish on `w⁰`, then `χ_L` has no nonzero null vector in the class: it
vanishes on spin `-t` (`LGeneric.null_not_eq_zero`); on spin `t` it is `0` for `k ≠ 0`
(`eq_zero_of_chiL_self_of_ne`) and `Sw⁰` with `S(1 - T_t) = 0` at `k = 0`
(`eq_smul_w0L_of_chiL_self`), where the rows of spin `-t` then give `S = 0`. Compared with
`LGeneric.isUnit_onePlusKLAll`, which needs `LGeneric` for both column spins, this needs it for one
column spin, plus `H_t ≠ 0` and the condition at `k = 0`; neither implies the other. At `k = 0` the
condition is also necessary (`LGeneric.isUnit_onePlusKLAll_iff_of_HL_ne`). Not in the note. -/
theorem LGeneric.isUnit_onePlusKLAll_of_HL_ne {t : Bool} {u : Fin 4 → ℂ}
    (hg : F.LGeneric k (!t) u) (hH : ∀ n, F.HL k (F.rL t) ((F.BL t).z k n) ≠ 0)
    (hT : k = 0 → 1 - F.TL t ≠ 0 ∨ ∃ n, ∑' l, F.chiL (!t) t 0 n l * F.w0L t l ≠ 0) :
    IsUnit (F.onePlusKLAll k) := by
  refine (F.isUnit_onePlusKLAll_iff k).mpr fun v hv hχ ↦ ?_
  have h0 := hg.null_not_eq_zero hv hχ
  have hrow : ∀ σ n, HasSum (fun l ↦ F.chiL σ t k n l * v (t, l)) 0 :=
    fun σ n ↦ (hasSum_row_iff_of_not_eq_zero h0).mpr (hχ σ n)
  have h1 : ∀ l, v (t, l) = 0 := by
    by_cases hk : k = 0
    · subst hk
      obtain ⟨S, hS, hw⟩ := eq_smul_w0L_of_chiL_self hg.a_ne hH fun n ↦ hrow t n
      suffices hS0 : S = 0 by
        intro l
        rw [hw l, hS0, zero_mul]
      rcases hT rfl with h | ⟨n, hn⟩
      · exact (mul_eq_zero.mp hS).resolve_right h
      · by_contra hS0
        apply hn
        have h2 := (hrow (!t) n).div_const S
        rw [zero_div] at h2
        have h3 : HasSum (fun l ↦ F.chiL (!t) t 0 n l * F.w0L t l) 0 := by
          convert h2 using 1
          funext l
          rw [hw l]
          field_simp
        exact h3.tsum_eq
    · exact fun l ↦ congrFun (eq_zero_of_chiL_self_of_ne hk hH fun n ↦ hrow t n) l
  funext ⟨τ, l⟩
  by_cases hτ : τ = t
  · subst hτ
    exact h1 l
  · obtain rfl : τ = !t := by cases τ <;> cases t <;> simp_all
    exact h0 l

/-- **The converse at `k = 0`**: where `H_t(zₙ) ≠ 0` on the lattice of `t`, `T_t = 1`, and every
row of spin `-t` vanishes on `w⁰`, `(w⁰, 0)` (`w⁰` on spin `t`) is a nonzero null vector of `χ_L`
in the class (`w⁰` is bounded), so `1 + 𝒦` is not invertible, for every `μ`, `h` with `U ≠ 0`. -/
theorem not_isUnit_onePlusKLAll_of_TL_eq_one {t : Bool} (ha : (F.a : ℂ) ≠ 0)
    (hH : ∀ n, F.HL 0 (F.rL t) ((F.BL t).z 0 n) ≠ 0) (hT : F.TL t = 1)
    (hrow : ∀ n, ∑' l, F.chiL (!t) t 0 n l * F.w0L t l = 0) : ¬IsUnit (F.onePlusKLAll 0) := by
  intro hu
  obtain ⟨C, hC⟩ := F.bdd_w0L t hH
  set v : Bool × ℤ → ℂ := fun p ↦ if p.1 = t then F.w0L t p.2 else 0 with hvdef
  have hvt : ∀ l, v (t, l) = F.w0L t l := fun l ↦ by simp [v]
  have hv0 : ∀ m, v (!t, m) = 0 := fun m ↦ by simp [v]
  have hvb : ∀ p, ‖v p‖ ≤ max C 0 := by
    rintro ⟨τ, l⟩
    by_cases hτ : τ = t
    · subst hτ
      rw [hvt]
      exact (hC l).trans (le_max_left _ _)
    · simp [v, hτ]
  have hcls : Summable fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp 0 p.2‖ ^ 2 := by
    refine (summable_prod_iff_spin (z := fun τ ↦ (F.BL τ).zOp 0)).mpr fun τ ↦ ?_
    refine (((F.BL τ).summable_inv_norm_zOp_sq 0).mul_left (max C 0 ^ 2)).of_nonneg_of_le
      (fun _ ↦ by positivity) fun l ↦ ?_
    rw [norm_div, div_pow, div_eq_mul_inv, inv_pow]
    exact mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (norm_nonneg _) (hvb _) 2) (by positivity)
  have hχ : ∀ σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 0 n p.2 * v p) 0 := by
    intro σ n
    refine (hasSum_row_iff_of_not_eq_zero hv0).mp ?_
    simp only [hvt]
    by_cases hσ : σ = t
    · subst hσ
      simpa [hT] using hasSum_chiL_self_w0L hH n
    · obtain rfl : σ = !t := by cases σ <;> cases t <;> simp_all
      have h := (hasSum_row_iff_of_not_eq_zero hv0).mpr
        (F.summable_chiL_row_of_bdd 0 hvb (!t) n).hasSum
      simp only [hvt] at h
      rwa [h.tsum_eq.symm.trans (hrow n)] at h
  have h0 := (F.isUnit_onePlusKLAll_iff 0).mp hu v hcls hχ
  have h1 := congrFun h0 (t, 0)
  rw [Pi.zero_apply, hvt] at h1
  exact F.w0L_ne ha hH 0 h1

/-- **`1 + 𝒦` with one regular column spin at `k = 0`, iff**: for every `μ_t`, `μ_t = 0` included,
under `LGeneric 0 (!t) u` and `H_t(zₙ) ≠ 0` on the lattice of `t`, `1 + 𝒦` is invertible iff
`1 - T_t ≠ 0` or some row of spin `-t` does not vanish on `w⁰`
(`LGeneric.isUnit_onePlusKLAll_of_HL_ne`, `not_isUnit_onePlusKLAll_of_TL_eq_one`). Not in the
note. -/
theorem LGeneric.isUnit_onePlusKLAll_iff_of_HL_ne {t : Bool} {u : Fin 4 → ℂ}
    (hg : F.LGeneric 0 (!t) u) (hH : ∀ n, F.HL 0 (F.rL t) ((F.BL t).z 0 n) ≠ 0) :
    IsUnit (F.onePlusKLAll 0) ↔
      1 - F.TL t ≠ 0 ∨ ∃ n, ∑' l, F.chiL (!t) t 0 n l * F.w0L t l ≠ 0 := by
  refine ⟨fun hu ↦ ?_, fun hT ↦ hg.isUnit_onePlusKLAll_of_HL_ne hH fun _ ↦ hT⟩
  by_contra h
  push Not at h
  exact not_isUnit_onePlusKLAll_of_TL_eq_one hg.a_ne hH (sub_eq_zero.mp h.1).symm h.2 hu

/-! ### The axes `μ = ±h ≠ 0` -/

variable (F) in
/-- On the axis `μ_t = 0`, `r_t = p₀(1 + e^{βa})(e^{2βμ} - 1)`. -/
theorem rL_eq_of_cL {t : Bool} (ht : F.cL t = 0) :
    F.rL t = F.p₀ * (1 + Real.exp (F.β * F.a)) * (Real.exp (2 * F.β * F.μ) - 1) := by
  cases t
  · have hh : F.h = F.μ := by simp only [cL, Bool.cond_false] at ht; linarith
    simp only [rL, mL, ρ, pL_false, pL_true, Bool.not_false]
    rw [F.p₂_eq_p₀_mul, F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, hh,
      show F.β * (F.a + F.μ + F.μ) = F.β * F.a + 2 * F.β * F.μ by ring,
      show F.β * (F.a + F.μ - F.μ) = F.β * F.a by ring, Real.exp_add]
    ring
  · have hh : F.h = -F.μ := by simp only [cL, Bool.cond_true] at ht; linarith
    simp only [rL, mL, ρ, pL_false, pL_true, Bool.not_true]
    rw [F.p₂_eq_p₀_mul, F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, hh,
      show F.β * (F.a + F.μ - -F.μ) = F.β * F.a + 2 * F.β * F.μ by ring,
      show F.β * (F.a + F.μ + -F.μ) = F.β * F.a by ring, Real.exp_add]
    ring

variable (F) in
/-- On the axis `μ_t = 0`, `r_t ≠ 0` for `μ ≠ 0`. -/
theorem rL_ne_of_cL {t : Bool} (ht : F.cL t = 0) (hμ : F.μ ≠ 0) : F.rL t ≠ 0 := by
  rw [F.rL_eq_of_cL ht]
  have h1 : Real.exp (2 * F.β * F.μ) ≠ 1 := by
    rw [Ne, Real.exp_eq_one_iff]
    exact mul_ne_zero (mul_ne_zero two_ne_zero F.β_pos.ne') hμ
  exact mul_ne_zero (mul_ne_zero F.p₀_pos.ne' (by positivity)) (sub_ne_zero.mpr h1)

variable (F k) in
/-- **`H_t ≠ 0` on the lattice of `t` on the axis `μ_t = 0`**, for `U ≠ 0`, `μ ≠ 0`: there
`zₙ = iy` and `H_t(iy) = q₀ - y² + 2ar_ty i`, whose imaginary part vanishes only at `y = 0`
(`rL_ne_of_cL`), where `H_t(0) = q₀ ≠ 0`. -/
theorem HL_ne_of_cL {t : Bool} (ht : F.cL t = 0) (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (n : ℤ) :
    F.HL k (F.rL t) ((F.BL t).z k n) ≠ 0 := by
  have hr := F.rL_ne_of_cL ht hμ
  have ha' : F.a ≠ 0 := by exact_mod_cast ha
  have hz : (F.BL t).z k n = (((2 * n + 1 + k) * Real.pi / F.β : ℝ) : ℂ) * I := by
    rw [Params.z_eq, lat_μ, lat_β, ht, ofReal_zero, zero_add]
  rw [hz]
  generalize ((2 * n + 1 + k) * Real.pi / F.β : ℝ) = y
  by_cases hy : y = 0
  · rw [hy, ofReal_zero, zero_mul]
    simpa [HL] using F.q₀L_ne k ha
  · intro h
    have him := congrArg Complex.im h
    simp [HL, q₀L, wL, pow_two] at him
    simp_all

/-! ### A closed form of `T_t` on the axis -/

variable (F) in
/-- On the axis `μ_t = 0`, `p₀(1 + e^{βa})(1 + e^{2βμ}) = 1`. -/
private theorem p₀_mul_of_cL {t : Bool} (ht : F.cL t = 0) :
    F.p₀ * (1 + Real.exp (F.β * F.a)) * (1 + Real.exp (2 * F.β * F.μ)) = 1 := by
  have hw := F.weights_sum
  cases t
  · have hh : F.h = F.μ := by simp only [cL, Bool.cond_false] at ht; linarith
    rw [F.p₂_eq_p₀_mul, F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, hh,
      show F.β * (F.a + F.μ + F.μ) = F.β * F.a + 2 * F.β * F.μ by ring,
      show F.β * (F.a + F.μ - F.μ) = F.β * F.a by ring, Real.exp_add] at hw
    linear_combination hw
  · have hh : F.h = -F.μ := by simp only [cL, Bool.cond_true] at ht; linarith
    rw [F.p₂_eq_p₀_mul, F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, hh,
      show F.β * (F.a + F.μ - -F.μ) = F.β * F.a + 2 * F.β * F.μ by ring,
      show F.β * (F.a + F.μ + -F.μ) = F.β * F.a by ring, Real.exp_add] at hw
    linear_combination hw

variable (F) in
/-- **On the axis `μ_t = 0`, `r_t = tanh(βμ)`**. -/
theorem rL_eq_tanh_of_cL {t : Bool} (ht : F.cL t = 0) : F.rL t = Real.tanh (F.β * F.μ) := by
  have h1 := F.p₀_mul_of_cL ht
  have hE := Real.exp_pos (F.β * F.μ)
  have he : Real.exp (2 * F.β * F.μ) = Real.exp (F.β * F.μ) ^ 2 := by
    rw [sq, ← Real.exp_add]
    ring_nf
  rw [F.rL_eq_of_cL ht, Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq, Real.exp_neg, he]
  rw [he] at h1
  field_simp
  linear_combination (Real.exp (F.β * F.μ) ^ 2 - 1) * h1

variable (F) in
/-- **On the axis `μ_t = 0`, `ζ_t/β = a²/cosh²(βμ)`**. -/
theorem ζLs_div_β_of_cL {t : Bool} (ht : F.cL t = 0) :
    F.ζLs t / F.β = F.a ^ 2 / (Real.cosh (F.β * F.μ) : ℂ) ^ 2 := by
  have h1 := F.p₀_mul_of_cL ht
  have hE := Real.exp_pos (F.β * F.μ)
  have he : Real.exp (2 * F.β * F.μ) = Real.exp (F.β * F.μ) ^ 2 := by
    rw [sq, ← Real.exp_add]
    ring_nf
  rw [he] at h1
  have hP : F.p₀ * (1 + Real.exp (F.β * F.a)) = 1 / (1 + Real.exp (F.β * F.μ) ^ 2) := by
    rw [eq_div_iff (by positivity)]
    exact h1
  have hcosh : 1 / Real.cosh (F.β * F.μ) ^ 2 =
      4 * Real.exp (F.β * F.μ) ^ 2 / (1 + Real.exp (F.β * F.μ) ^ 2) ^ 2 := by
    rw [Real.cosh_eq, Real.exp_neg]
    field_simp
    ring
  have hr : 4 * (F.p₀ + F.pL t) * (F.pL (!t) + F.p₂) = 1 / Real.cosh (F.β * F.μ) ^ 2 := by
    rw [hcosh]
    cases t
    · have hh : F.h = F.μ := by simp only [cL, Bool.cond_false] at ht; linarith
      simp only [pL_false, pL_true, Bool.not_false]
      rw [F.p₂_eq_p₀_mul, F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, hh,
        show F.β * (F.a + F.μ + F.μ) = F.β * F.a + 2 * F.β * F.μ by ring,
        show F.β * (F.a + F.μ - F.μ) = F.β * F.a by ring, Real.exp_add, he]
      calc _ = 4 * (F.p₀ * (1 + Real.exp (F.β * F.a))) ^ 2 * Real.exp (F.β * F.μ) ^ 2 := by ring
        _ = _ := by rw [hP]; field_simp
    · have hh : F.h = -F.μ := by simp only [cL, Bool.cond_true] at ht; linarith
      simp only [pL_false, pL_true, Bool.not_true]
      rw [F.p₂_eq_p₀_mul, F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, hh,
        show F.β * (F.a + F.μ - -F.μ) = F.β * F.a + 2 * F.β * F.μ by ring,
        show F.β * (F.a + F.μ + -F.μ) = F.β * F.a by ring, Real.exp_add, he]
      calc _ = 4 * (F.p₀ * (1 + Real.exp (F.β * F.a))) ^ 2 * Real.exp (F.β * F.μ) ^ 2 := by ring
        _ = _ := by rw [hP]; field_simp
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  have hr' : 4 * ((F.p₀ : ℂ) + F.pL t) * (F.pL (!t) + F.p₂) =
      1 / (Real.cosh (F.β * F.μ) : ℂ) ^ 2 := by exact_mod_cast hr
  rw [ζLs]
  rw [show 4 * (F.β : ℂ) * F.a ^ 2 * (F.p₀ + F.pL t) * (F.pL (!t) + F.p₂) / F.β =
      F.a ^ 2 * (4 * (F.p₀ + F.pL t) * (F.pL (!t) + F.p₂)) by field_simp, hr']
  ring

/-- **A closed form of `T_t` on the axis `μ_t = 0`**, for `U ≠ 0`, `μ ≠ 0`: `H_t` has the roots
`x± = -a tanh(βμ) ± ia/cosh(βμ)` (`r_t = tanh(βμ)`, `rL_eq_tanh_of_cL`), `ζ_t/β = a²/cosh²(βμ)`
(`ζLs_div_β_of_cL`), and `∑ₙ 1/((zₙ - x₊)(zₙ - x₋)) = (S(x₊) - S(x₋))/(x₊ - x₋)`
(`Params.hasSum_pair`) with `S(x) = -(β/2)tanh(βx/2)` on the lattice of `t`, so
`T_t = aβ[tanh(βx₋/2) - tanh(βx₊/2)]/(4i cosh(βμ))`. So `T_t` depends on `U` only through `a²`:
`a ↦ -a` maps `x± ↦ -x∓`, and `tanh` is odd. Not in the note. -/
theorem TL_eq_of_cL {t : Bool} (ht : F.cL t = 0) (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) :
    F.TL t = F.a * F.β / (4 * I * Real.cosh (F.β * F.μ)) *
      (Complex.tanh
          (F.β * (-F.a * Real.tanh (F.β * F.μ) - I * F.a / Real.cosh (F.β * F.μ)) / 2) -
        Complex.tanh
          (F.β * (-F.a * Real.tanh (F.β * F.μ) + I * F.a / Real.cosh (F.β * F.μ)) / 2)) := by
  have hc : (Real.cosh (F.β * F.μ) : ℂ) ≠ 0 := ofReal_ne_zero.mpr (Real.cosh_pos _).ne'
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  have hth : (Real.tanh (F.β * F.μ) : ℂ) ^ 2 + 1 / (Real.cosh (F.β * F.μ) : ℂ) ^ 2 = 1 := by
    have hr : Real.tanh (F.β * F.μ) ^ 2 + 1 / Real.cosh (F.β * F.μ) ^ 2 = 1 := by
      have := Real.cosh_sq_sub_sinh_sq (F.β * F.μ)
      have hc' := (Real.cosh_pos (F.β * F.μ)).ne'
      rw [Real.tanh_eq_sinh_div_cosh]
      field_simp
      linarith
    exact_mod_cast hr
  set c : ℂ := (Real.cosh (F.β * F.μ) : ℂ) with hcdef
  set th : ℂ := (Real.tanh (F.β * F.μ) : ℂ) with hthdef
  set xp : ℂ := -F.a * th + I * F.a / c with hxpdef
  set xm : ℂ := -F.a * th - I * F.a / c with hxmdef
  have hH : ∀ z, F.HL 0 (F.rL t) z = (z - xp) * (z - xm) := by
    intro z
    rw [HL, F.rL_eq_tanh_of_cL ht, q₀L, wL]
    simp only [ω, Params.ω, Int.cast_zero, mul_zero, zero_mul, zero_div, ofReal_zero, ne_eq,
      OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sub_zero]
    rw [← hthdef, hxpdef, hxmdef]
    linear_combination (-(F.a : ℂ) ^ 2) * hth + (F.a ^ 2 / c ^ 2) * I_sq
  have hne := F.HL_ne_of_cL 0 ht ha hμ
  have hxp : ∀ n, (F.BL t).z 0 n ≠ xp := fun n h ↦ hne n (by rw [hH, h, sub_self, zero_mul])
  have hxm : ∀ n, (F.BL t).z 0 n ≠ xm := fun n h ↦ hne n (by rw [hH, h, sub_self, mul_zero])
  have hpm : xp - xm = 2 * I * F.a / c := by
    rw [hxpdef, hxmdef]
    ring
  have hpm' : xp ≠ xm := fun h ↦ by
    have h0 : 2 * I * F.a / c = 0 := by rw [← hpm, h, sub_self]
    simp [ha, hc, I_ne_zero] at h0
  have hsum := (F.BL t).hasSum_pair hpm' hxp hxm
  rw [TL, F.ζLs_div_β_of_cL ht, ← hcdef]
  have htsum : ∑' n, (F.HL 0 (F.rL t) ((F.BL t).z 0 n))⁻¹ =
      ((F.BL t).S 0 xp - (F.BL t).S 0 xm) / (xp - xm) := by
    rw [← hsum.tsum_eq]
    exact tsum_congr fun n ↦ by rw [hH, one_div]
  rw [htsum, hpm, Params.S, Params.S, lat_μ, lat_β, ht, ite_eq_left even_zero',
    ite_eq_left even_zero', ofReal_zero, sub_zero, sub_zero]
  field_simp
  ring

/-- **`1 + 𝒦` is invertible on the axes `μ = ±h ≠ 0`** under `LGeneric` for the regular column
spin: at `μ_t = 0` (`μ = h` for `t = ↓`, `μ = -h` for `t = ↑`), if the column spin `-t` satisfies
the assumptions of Eq. (field-L) at the point (`LGeneric k (!t) u`, which contains `μ ≠ 0`,
`h ≠ 0`, `U ≠ 0`) and, at `k = 0`, `1 - T_t ≠ 0` or a row of spin `-t` does not vanish on `w⁰`,
then `1 + 𝒦` is invertible, so `Γ_L` exists. At `k = 0` the condition is also necessary
(`isUnit_onePlusKLAll_iff_of_cL_eq_zero`). -/
theorem isUnit_onePlusKLAll_of_cL_eq_zero {t : Bool} (ht : F.cL t = 0) {u : Fin 4 → ℂ}
    (hg : F.LGeneric k (!t) u)
    (hT : k = 0 → 1 - F.TL t ≠ 0 ∨ ∃ n, ∑' l, F.chiL (!t) t 0 n l * F.w0L t l ≠ 0) :
    IsUnit (F.onePlusKLAll k) :=
  hg.isUnit_onePlusKLAll_of_HL_ne (F.HL_ne_of_cL k ht hg.a_ne hg.μ_ne) hT

/-- **`1 + 𝒦` on the axes `μ = ±h ≠ 0` at `k = 0`, iff**: at `μ_t = 0`, under
`LGeneric 0 (!t) u`, `1 + 𝒦` is invertible iff `1 - T_t ≠ 0` or some row of spin `-t` does not
vanish on `w⁰`. -/
theorem isUnit_onePlusKLAll_iff_of_cL_eq_zero {t : Bool} (ht : F.cL t = 0) {u : Fin 4 → ℂ}
    (hg : F.LGeneric 0 (!t) u) :
    IsUnit (F.onePlusKLAll 0) ↔
      1 - F.TL t ≠ 0 ∨ ∃ n, ∑' l, F.chiL (!t) t 0 n l * F.w0L t l ≠ 0 :=
  hg.isUnit_onePlusKLAll_iff_of_HL_ne (F.HL_ne_of_cL 0 ht hg.a_ne hg.μ_ne)

/-- `LGeneric` for the column spin `-t` gives `LAxisGeneric`, given its condition for the column
spin `t`: `K_t(u_j) ≠ 0`, or `K_{t,-t}(u_j) ≠ 0` and `u_j² + q₀ ≠ 0`, at the non-atomic roots. -/
theorem LGeneric.lAxisGeneric_of_not {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k (!t) u)
    (hK : ∀ j, ¬F.Atomic k (u j) →
      F.KL k t t (u j) ≠ 0 ∨ F.KL k t (!t) (u j) ≠ 0 ∧ u j ^ 2 + F.q₀L k ≠ 0) :
    F.LAxisGeneric k u :=
  ⟨hg.a_ne, hg.root, hg.sep, hg.off, fun τ j hj ↦ by
    by_cases hτ : τ = t
    · subst hτ
      exact hK j hj
    · obtain rfl : τ = !t := by cases τ <;> cases t <;> simp_all
      exact Or.inl (hg.K_ne j hj)⟩

/-- **`Γ_L` on the axes `μ = ±h ≠ 0` is the note's axis formula** under `LGeneric` for the regular
column spin: at `μ_t = 0`, if the column spin `-t` satisfies `LGeneric k (!t) u` at the point, the
column spin `t` the condition of `LAxisGeneric` (`K_t(u_j) ≠ 0`, or `K_{t,-t}(u_j) ≠ 0` and
`u_j² + q₀ ≠ 0`, at the non-atomic roots), and, at `k = 0`, `1 - T_t ≠ 0` or a row of spin `-t`
does not vanish on `w⁰`, then `Γ_L = gammaLAll`, entrywise (`isUnit_onePlusKLAll_of_cL_eq_zero`,
`LAxisGeneric.gammaOpLAll_eq`). -/
theorem gammaOpLAll_eq_gammaLAll_of_cL_eq_zero {t : Bool} (ht : F.cL t = 0) {u : Fin 4 → ℂ}
    (hg : F.LGeneric k (!t) u)
    (hK : ∀ j, ¬F.Atomic k (u j) →
      F.KL k t t (u j) ≠ 0 ∨ F.KL k t (!t) (u j) ≠ 0 ∧ u j ^ 2 + F.q₀L k ≠ 0)
    (hT : k = 0 → 1 - F.TL t ≠ 0 ∨ ∃ n, ∑' l, F.chiL (!t) t 0 n l * F.w0L t l ≠ 0)
    (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n τ m = F.gammaLAll k u σ n τ m :=
  (hg.lAxisGeneric_of_not hK).gammaOpLAll_eq (isUnit_onePlusKLAll_of_cL_eq_zero ht hg hT) σ n τ m

end FParams

end HubbardAtom
