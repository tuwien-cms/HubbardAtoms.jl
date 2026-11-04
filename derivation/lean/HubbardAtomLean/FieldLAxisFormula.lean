import HubbardAtomLean.FieldLAxisGeneric

/-!
# The longitudinal vertex on the axes: the explicit formula and its agreement with `Γ_L`

Section "Reductions, continuous extensions and exceptional sets" of the note continues
Eq. (field-L) to the axes `μ = 0`, `h = 0`, `μ = ±h`: the three kernels that are `0/0` on their
special pairs take the limits of Eq. (field-kernel-limits), and a mode coefficient that is `0/0`
(on `μ_t = 0`, where `D_t = 0`) takes the quotient form of Eq. (field-L-quotient); the note states
the agreement with the vertex on `μ = ±h ≠ 0` as numerically verified, not derived. Here that
formula is defined, and at an axis point where `LAxisGeneric` holds it is the limit of
Eq. (field-L) and, if `1 + 𝒦` is invertible there, the operator vertex `Γ_L`.

* `FParams.kerDL`: the kernel `D_t/(z + v)` of the equal-spin row, `βp₀p_{-t}` on a mirror pair at
  `μ_t = 0`; on a mirror pair it is `κ_D = D_t/(2μ_t) = β sinhc(βμ_t)/Z²` for every `μ_t`
  (`kerDL_mirror`, `κDL_of_cL`, `two_mul_cL_mul_κDL`). With `kerPL` (`ρ/(z + v)`, `βp₀`) and
  `-kerSLs σ` (`tm/(z - v)` at `σ = -t`, continued by `-βp_t`, the note's `tσβp↓`) these are the
  three kernels of Eq. (field-kernel-limits).
* `FParams.cModeQ`: the quotient form of Eq. (field-L-quotient) (without its factor `P(u_j)`);
  `cModeAll`: `cMode` where `K_t(u_j) ≠ 0`, `cModeQ` where `K_t(u_j) = 0`; `cMode_eq_cModeQ`: the
  two agree at a root where `K_t` and `K_{t,-t}` are both nonzero, by the quotient identity `res_z`.
* `FParams.gammaLc`: the right side of Eq. (field-L) with these kernels and given mode
  coefficients; `FParams.gammaLAll`: **the axis formula**, `gammaLc` with `cModeAll`. On the domain
  of Eq. (field-L) it is `gammaL` (`gammaL_eq_gammaLc`, `LGeneric.gammaLAll_eq`). It is meaningful
  where `K_t(u_j) ≠ 0` or `K_{t,-t}(u_j) ≠ 0` at the non-atomic roots; at `μ = h = 0` both vanish
  on two roots and the quotient divides by `0`.
* `FParams.LAxisGeneric.tendsto_gammaL_of_eventually`: along a family `(U, β, μ, h)(x)` continuous
  at `x₀`, through points of the domain of Eq. (field-L) with roots `u(x)` continuous at `x₀`,
  **`gammaL` tends to `gammaLAll(x₀, u(x₀))` if `LAxisGeneric` holds at `x₀`**, term by term: the
  kernels are continuous (`continuousAt_famF_kerDL`, and `continuousAt_famF_kerPL`,
  `continuousAt_famF_kerSLs`), `E₈ ≠ 0` on the lattices and the roots are off them at `x₀`
  (`LAxisGeneric.E8L_ne`, `LAxisGeneric.z_ne`), the atomic root stays atomic
  (`eventually_famF_atomic`), and `cMode → cModeAll` (`tendsto_famF_cMode`): where `K_t(u_j) = 0` at
  `x₀`, `K_{t,-t}(u_j) ≠ 0`, and nearby `cMode` is the quotient form, which is continuous.
  `LAxisGeneric.tendsto_gammaL_gammaLAll`: the same along the points off the axes, with the roots
  of `LAxisGeneric.exists_eventually_LGeneric`; `LAxisGeneric.tendsto_gammaL_lineL_gammaLAll`: the
  same along the line `lineL`, a concrete approach. No invertibility of `1 + 𝒦` is used.
* `FParams.LAxisGeneric.gammaOpLAll_eq`: **where `LAxisGeneric` holds and `1 + 𝒦` is invertible,
  `Γ_L = gammaLAll`**: along the line `lineL`, both are limits of `gammaL`.
* `FParams.not_lAxisGeneric_corner`: at `μ = h = 0`, `LAxisGeneric` never holds: the roots
  `u² = -q₀ ± λ(p₀ - p↑)` of `E₈` have `K_t = K_{t,-t} = 0`, the simultaneous `0/0` of the note,
  which uses Eqs. (field-h0), (field-mu0) there.

Assumed: `LAxisGeneric` at the axis point, and for the agreement that `1 + 𝒦` is invertible there.
Not shown: that these hold at given axis points (on `h = 0`, `μ ≠ 0` and on `μ = 0`, `h ≠ 0` the
invertibility reduces to the zero-field channels, `FieldLAxisValues.lean`; on `μ = ±h ≠ 0` it
holds where `LGeneric` holds for the regular column spin, under a condition at `k = 0`,
`FieldLAxisUnitPM.lean`). The comparison of `gammaLAll` with Eqs. (field-h0), (field-mu0) as
closed forms is `FieldLAxisClosedForms.lean`. Not in the note: `LAxisGeneric`, the limits and the
agreement as proofs, and the corner statement.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The kernel `D_t/(z + v)` of the equal-spin row -/

/-- `p₀p_{-t} = e^{-βμ_t}/Z²`. -/
theorem p₀_mul_pL_not (t : Bool) : F.p₀ * F.pL (!t) = Real.exp (-(F.β * F.cL t)) / F.Z ^ 2 := by
  cases t <;>
  · simp only [p₀, pL, pUp, pDn, boltz₀, boltzUp, boltzDn, cL, Bool.not_true, Bool.not_false,
      Bool.cond_true, Bool.cond_false]
    rw [div_mul_div_comm, ← Real.exp_add, sq]
    congr 2
    ring

/-- `D_t = p₂p_t - p₀p_{-t} = 2 sinh(βμ_t)/Z²`. -/
theorem DL_eq_sinh (t : Bool) : F.DL t = 2 * Real.sinh (F.β * F.cL t) / F.Z ^ 2 := by
  have hZ := F.Z_pos.ne'
  have e : Real.exp (2 * F.β * F.cL t) = Real.exp (F.β * F.cL t) ^ 2 := by
    rw [sq, ← Real.exp_add]
    ring_nf
  rw [DL, F.p₂_mul_pL t, F.p₀_mul_pL_not t, Real.sinh_eq, e, Real.exp_neg]
  field_simp

/-- The kernel `D_t/(z + v)` on a mirror pair of the lattice of `t`: `κ_D = D_t/(2μ_t)`, continued
by `βp₀p_{-t}` at `μ_t = 0`, as `β sinhc(βμ_t)/Z²` with `sinhc = dslope sinh 0`. -/
noncomputable def κDL (t : Bool) : ℝ := F.β * dslope Real.sinh 0 (F.β * F.cL t) / F.Z ^ 2

/-- At `μ_t = 0`, `κ_D = βp₀p_{-t}`. -/
theorem κDL_of_cL {t : Bool} (ht : F.cL t = 0) : F.κDL t = F.β * F.p₀ * F.pL (!t) := by
  rw [κDL, ht, mul_zero, dslope_same, Real.deriv_sinh, Real.cosh_zero, mul_assoc,
    F.p₀_mul_pL_not, ht, mul_zero, neg_zero, Real.exp_zero]
  ring

/-- `2μ_tκ_D = D_t`. -/
theorem two_mul_cL_mul_κDL (t : Bool) : 2 * F.cL t * F.κDL t = F.DL t := by
  by_cases ht : F.cL t = 0
  · rw [ht, mul_zero, zero_mul, DL_eq_sinh, ht, mul_zero, Real.sinh_zero, mul_zero, zero_div]
  · have hβ := F.β_pos.ne'
    have hZ := F.Z_pos.ne'
    rw [κDL, dslope_of_ne _ (mul_ne_zero hβ ht), slope_def_field, sub_zero, Real.sinh_zero,
      sub_zero, DL_eq_sinh]
    field_simp

/-- The kernel `D_t/(zₙ + zₘ)` of the equal-spin row of Eq. (field-L), `zₙ`, `zₘ` on the lattice
of `t`, and `βp₀p_{-t}` where the denominator vanishes (`μ_t = 0`, `m = -n - k - 1`): the first
limit of Eq. (field-kernel-limits). -/
noncomputable def kerDL (t : Bool) (n m : ℤ) : ℂ :=
  if F.cL t = 0 ∧ n + m + k + 1 = 0 then F.β * F.p₀ * F.pL (!t)
  else F.DL t / ((F.BL t).z k n + (F.BL t).z k m)

/-- **The kernel on a mirror pair** is `κ_D`, for every `μ_t`. -/
theorem kerDL_mirror (t : Bool) (n : ℤ) : F.kerDL k t n (mirror k n) = F.κDL t := by
  have hidx : n + mirror k n + k + 1 = 0 := by unfold mirror; ring
  by_cases ht : F.cL t = 0
  · rw [kerDL, ite_eq_left ⟨ht, hidx⟩, F.κDL_of_cL ht]
    push_cast
    ring
  · have e : (F.BL t).z k n + (F.BL t).z k (mirror k n) = 2 * F.cL t :=
      (F.BL t).z_add_z_mirror k n
    have ht' : (F.cL t : ℂ) ≠ 0 := ofReal_ne_zero.mpr ht
    rw [kerDL, ite_eq_right (fun h ↦ ht h.1), e, ← F.two_mul_cL_mul_κDL]
    push_cast
    field_simp

/-- The kernel off the mirror pairs. -/
theorem kerDL_of_ne_mirror (t : Bool) {n m : ℤ} (hm : m ≠ mirror k n) :
    F.kerDL k t n m = F.DL t / ((F.BL t).z k n + (F.BL t).z k m) := by
  rw [kerDL, ite_eq_right (fun h ↦ hm (by have := h.2; unfold mirror; omega))]

/-- The kernel for `μ_t ≠ 0`. -/
theorem kerDL_of_cL_ne {t : Bool} (hc : F.cL t ≠ 0) (n m : ℤ) :
    F.kerDL k t n m = F.DL t / ((F.BL t).z k n + (F.BL t).z k m) := by
  rw [kerDL, ite_eq_right (fun h ↦ hc h.1)]

/-! ### The mode coefficients -/

open Classical in
/-- The quotient form of the mode coefficient, Eq. (field-L-quotient) without its factor `P(u_j)`:
`-λ(ρΞ_j + m_tΠ_j)/(β²K_{t,-t}(u_j)E₈'(s_j))`, zero at the atomic root. As in `gammaL`, the mode
is normalized with the column spin `t` (`fMp k t`), so `K_{t,-t}` is `KL k t (!t)`, the row `-t`
of the `t`-normalized mode; the note normalizes with `↑`, which differs by a rescaling of the
mode. Where `K_{t,-t}(u_j) = 0` it divides by `0`. -/
noncomputable def cModeQ (t : Bool) (u : Fin 4 → ℂ) (j : Fin 4) : ℂ :=
  if F.Atomic k (u j) then 0
  else -(F.lamL * (F.ρ * F.XiY k (u j ^ 2) + F.mL t * F.PiY k (u j ^ 2))) /
    (F.β ^ 2 * F.KL k t (!t) (u j) * dprod (fun i ↦ u i ^ 2) j)

open Classical in
/-- The mode coefficient of the root `u_j` on and off the axes: `cMode` of Eq. (field-L-mode)
where `K_t(u_j) ≠ 0`, and the quotient form `cModeQ` of Eq. (field-L-quotient) where
`K_t(u_j) = 0`. Meaningful where `K_t(u_j) ≠ 0` or `K_{t,-t}(u_j) ≠ 0`; otherwise (at `μ = h = 0`)
the quotient divides by `0`. -/
noncomputable def cModeAll (t : Bool) (u : Fin 4 → ℂ) (j : Fin 4) : ℂ :=
  if F.KL k t t (u j) = 0 then F.cModeQ k t u j else F.cMode k t u j

variable {F k}

/-- **The two forms of the mode coefficient agree** at a root where both `K_t` and `K_{t,-t}` are
nonzero, by the quotient identity `res_z`. -/
theorem cMode_eq_cModeQ {t : Bool} {u : Fin 4 → ℂ} {j : Fin 4} (hE : F.E8L k (u j) = 0)
    (hK : F.KL k t t (u j) ≠ 0) (hK' : F.KL k t (!t) (u j) ≠ 0) :
    F.cMode k t u j = F.cModeQ k t u j := by
  unfold cMode cModeQ
  split_ifs with ha
  · rfl
  · have hβ := β_ne_zeroT (F := F)
    by_cases hd : dprod (fun i ↦ u i ^ 2) j = 0
    · simp [hd]
    have h : F.aAL k t (u j) * F.KL k t (!t) (u j) =
        -(F.lamL * (F.ρ * F.XiY k (u j ^ 2) + F.mL t * F.PiY k (u j ^ 2))) * F.KL k t t (u j) :=
      F.res_z k t hE
    rw [div_eq_div_iff (mul_ne_zero (mul_ne_zero (pow_ne_zero 2 hβ) hK) hd)
      (mul_ne_zero (mul_ne_zero (pow_ne_zero 2 hβ) hK') hd)]
    linear_combination (F.β ^ 2 * dprod (fun i ↦ u i ^ 2) j) * h

/-- Where `K_t ≠ 0` at the non-atomic roots, `cModeAll` is `cMode`. -/
theorem cModeAll_eq_cMode {t : Bool} {u : Fin 4 → ℂ} {j : Fin 4}
    (hK : ¬F.Atomic k (u j) → F.KL k t t (u j) ≠ 0) : F.cModeAll k t u j = F.cMode k t u j := by
  unfold cModeAll
  split_ifs with h0
  · by_cases ha : F.Atomic k (u j)
    · simp [cModeQ, cMode, ha]
    · exact absurd h0 (hK ha)
  · rfl

/-- Where `K_{t,-t} ≠ 0` at a root, `cModeAll` is the quotient form. -/
theorem cModeAll_eq_cModeQ {t : Bool} {u : Fin 4 → ℂ} {j : Fin 4} (hE : F.E8L k (u j) = 0)
    (hK' : F.KL k t (!t) (u j) ≠ 0) : F.cModeAll k t u j = F.cModeQ k t u j := by
  unfold cModeAll
  split_ifs with h0
  · rfl
  · exact cMode_eq_cModeQ hE h0 hK'

variable (F k)

/-! ### The vertex -/

/-- The right side of Eq. (field-L) with the three kernels of Eq. (field-kernel-limits) and given
mode coefficients `c j` (in place of `𝔠ⱼ/P(u_j)`): `z = zₙ` on the lattice of `σ`, `v = zₘ` on that
of `t`. The kernels are `kerDL` (`D_t/(z + v)`, equal spins), `-kerSLs σ` (`t m/(z - v)`, `σ = -t`,
continued by `-βp_t`, the note's `tσβp↓`) and `kerPL` (`ρ/(z + v)`). As in `gammaL`, the modes
`fMp k t`, `fMm k t` are normalized with the column spin `t`. -/
noncomputable def gammaLc (u c : Fin 4 → ℂ) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) : ℂ :=
  (if σ = t then
      F.β * F.PL k ((F.BL t).z k n) *
          (1 / (((F.BL t).z k n + F.a * F.rL t) ^ 2 - F.wL k ^ 2 / 4) -
            F.QL k t ((F.BL t).z k n) / F.E8L k ((F.BL t).z k n)) * kron n m +
        2 * F.lamL ^ 2 * ((F.BL t).z k m ^ 2 + F.q₀L k) * F.PL k ((F.BL t).z k m) /
          F.E8L k ((F.BL t).z k m) * F.kerDL k t n m
    else
      2 * F.a - F.lamL * F.PL k ((F.BL t).z k m) / F.E8L k ((F.BL t).z k m) *
        (-F.kerSLs k σ n m * (F.HL k (F.rL t) (-(F.BL t).z k m) *
              F.HL k (F.rL (!t)) (-(F.BL t).z k m) + F.lamL ^ 2 * (F.piP - F.piS)) +
          F.kerPL k n m * (F.HL k (F.rL t) (-(F.BL t).z k m) *
              F.HL k (F.rL (!t)) ((F.BL t).z k m) - F.lamL ^ 2 * (F.piP - F.piS)))) -
    F.β ^ 2 * ∑ j, c j * (F.PL k (u j) / ((F.BL t).z k m ^ 2 - u j ^ 2) *
      (F.fMp k t σ (u j) / ((F.BL σ).z k n - u j) + F.fMm k t σ (u j) / ((F.BL σ).z k n + u j)))

/-- **The note's continuation of Eq. (field-L) to the axes**, Eqs. (field-kernel-limits),
(field-L-quotient): Eq. (field-L) with the kernels of Eq. (field-kernel-limits) and the mode
coefficients `cModeAll`. On the domain of Eq. (field-L) it is `gammaL` (`LGeneric.gammaLAll_eq`).
Where `LAxisGeneric` holds it is the limit of `gammaL` (`LAxisGeneric.tendsto_gammaL_gammaLAll`),
and `Γ_L` if `1 + 𝒦` is invertible there (`LAxisGeneric.gammaOpLAll_eq`); `LAxisGeneric` never
holds at `μ = h = 0` (`not_lAxisGeneric_corner`). -/
noncomputable def gammaLAll (u : Fin 4 → ℂ) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) : ℂ :=
  F.gammaLc k u (F.cModeAll k t u) σ n t m

variable {F k}

/-- Off the axes, `gammaL` is `gammaLc` with the coefficients `cMode`: the kernels are the
quotients themselves. -/
theorem gammaL_eq_gammaLc (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {t : Bool} (hc : F.cL t ≠ 0)
    (u : Fin 4 → ℂ) (σ : Bool) (n m : ℤ) :
    F.gammaL k u σ n t m = F.gammaLc k u (F.cMode k t u) σ n t m := by
  have hs : ∑ j, F.cMode k t u j * F.PL k (u j) / ((F.BL t).z k m ^ 2 - u j ^ 2) *
        (F.fMp k t σ (u j) / ((F.BL σ).z k n - u j) +
          F.fMm k t σ (u j) / ((F.BL σ).z k n + u j)) =
      ∑ j, F.cMode k t u j * (F.PL k (u j) / ((F.BL t).z k m ^ 2 - u j ^ 2) *
        (F.fMp k t σ (u j) / ((F.BL σ).z k n - u j) +
          F.fMm k t σ (u j) / ((F.BL σ).z k n + u j))) :=
    Finset.sum_congr rfl fun j _ ↦ by ring
  rw [gammaL, gammaLc, hs]
  by_cases hσ : σ = t
  · subst hσ
    simp only [↓reduceIte, F.kerDL_of_cL_ne k hc, div_mul_eq_div_div]
    ring
  · obtain rfl : σ = !t := by cases σ <;> cases t <;> simp_all
    have eP : F.kerPL k n m = F.ρ / ((F.BL (!t)).z k n + (F.BL t).z k m) := by
      rw [kerPL, ite_eq_right (fun h ↦ hμ h.1)]
      have e := F.zL_add_opp k (!t) n m
      rw [Bool.not_not] at e
      rw [e]
    have eS : F.kerSLs k (!t) n m = F.mL (!t) / ((F.BL (!t)).z k n - (F.BL t).z k m) := by
      rw [kerSLs, ite_eq_right (fun h ↦ hh h.1), Bool.not_not]
    have hm : (F.mL (!t) : ℂ) = -F.mL t := by rw [F.mL_not t]; push_cast; ring
    simp only [Bool.not_ne_self, ↓reduceIte, eP, eS, hm]
    ring

/-- **Off the axes, the axis formula is Eq. (field-L)**: on the domain of Eq. (field-L) for the
column spin `t`, `gammaLAll = gammaL`. -/
theorem LGeneric.gammaLAll_eq {t : Bool} {u : Fin 4 → ℂ} (hg : F.LGeneric k t u) (σ : Bool)
    (n m : ℤ) : F.gammaLAll k u σ n t m = F.gammaL k u σ n t m := by
  have e : F.cModeAll k t u = F.cMode k t u := funext fun j ↦ cModeAll_eq_cMode (hg.K_ne j)
  rw [gammaL_eq_gammaLc hg.μ_ne hg.h_ne hg.c_ne, gammaLAll, e]

/-! ### The roots at an axis point -/

/-- Under `LAxisGeneric`, every root is off both lattices, the atomic one too (it is real, and
`Im zₙ ≠ 0` at `k = 0`). -/
theorem LAxisGeneric.z_ne {u : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u) (σ : Bool) (l : ℤ)
    (j : Fin 4) : (F.BL σ).z k l ≠ u j ∧ (F.BL σ).z k l ≠ -u j := by
  by_cases hj : F.Atomic k (u j)
  · obtain ⟨rfl, hu⟩ := hj
    have hν := (F.BL σ).ν_ne_zero l
    have hz : ((F.BL σ).z 0 l).im = (F.BL σ).ν l := by
      simp [Params.z, Params.x, Params.w, Params.ω]
    have hu' : (u j).im = 0 := by
      rcases sq_eq_sq_iff_eq_or_eq_neg.1 hu with h | h <;> simp [h]
    refine ⟨fun h ↦ hν ?_, fun h ↦ hν ?_⟩ <;> simpa [hz, hu'] using congrArg Complex.im h
  · exact h₀.off j hj σ l

/-- Under `LAxisGeneric`, `zₗ² - u_j² ≠ 0` on both lattices. -/
theorem LAxisGeneric.sq_sub_ne {u : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u) (σ : Bool) (l : ℤ)
    (j : Fin 4) : (F.BL σ).z k l ^ 2 - u j ^ 2 ≠ 0 := by
  obtain ⟨h₁, h₂⟩ := h₀.z_ne σ l j
  rw [sq_sub_sq]
  exact mul_ne_zero (fun h ↦ h₂ (eq_neg_of_add_eq_zero_left h)) (sub_ne_zero.2 h₁)

/-- Under `LAxisGeneric`, `E₈ ≠ 0` on both lattices. -/
theorem LAxisGeneric.E8L_ne {u : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u) (σ : Bool) (l : ℤ) :
    F.E8L k ((F.BL σ).z k l) ≠ 0 := by
  have hv := h₀.sq_sub_ne σ l
  rw [F.E8L_eq_prod k h₀.root h₀.sep]
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero (hv 0) (hv 1)) (hv 2)) (hv 3)

/-- `E₈'(s_j) = ∏_{i ≠ j}(s_j - s_i)` is continuous in the roots. -/
theorem _root_.HubbardAtom.continuousAt_dprod {X : Type*} [TopologicalSpace X] {y₀ : X}
    {u : X → Fin 4 → ℂ} (hu : ContinuousAt u y₀) (j : Fin 4) :
    ContinuousAt (fun y ↦ dprod (fun i ↦ u y i ^ 2) j) y₀ := by
  have h0 := continuousAt_pi.1 hu 0
  have h1 := continuousAt_pi.1 hu 1
  have h2 := continuousAt_pi.1 hu 2
  have h3 := continuousAt_pi.1 hu 3
  fin_cases j <;> simp only [dprod] <;> fun_prop

/-! ### Along a family -/

section Family

variable {P : Type*} [TopologicalSpace P] {U β μ h : P → ℝ} {hβ : ∀ x, 0 < β x} {k : ℤ}
  {x₀ : P} (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)
  (hh : ContinuousAt h x₀)

include hU in
/-- A root non-atomic at `x₀` is non-atomic near `x₀`. -/
theorem eventually_famF_not_atomic {v : P → ℂ} (hv : ContinuousAt v x₀)
    (hj : ¬(famF U β μ h hβ x₀).Atomic k (v x₀)) :
    ∀ᶠ x in 𝓝 x₀, ¬(famF U β μ h hβ x).Atomic k (v x) := by
  by_cases hk : k = 0
  · have ha := continuousAt_famF_a hU (β := β) (μ := μ) (h := h) (hβ := hβ) (x₀ := x₀)
    have h0 : v x₀ ^ 2 - ((famF U β μ h hβ x₀).a : ℂ) ^ 2 ≠ 0 :=
      fun h0 ↦ hj ⟨hk, sub_eq_zero.1 h0⟩
    exact ((hv.pow 2).sub (ha.pow 2)).eventually_ne h0 |>.mono fun x hx hxa ↦
      hx (sub_eq_zero.2 hxa.2)
  · exact Eventually.of_forall fun x hxa ↦ hk hxa.1

include hU in
/-- **The atomic root stays atomic**: if the roots `u(x)` of `E₈` near `x₀` (along `𝓝[S] x₀`) have
distinct squares, and those at `x₀` too, a root atomic at `x₀` is atomic near `x₀`, since `±a(x)`
is a root of `E₈` (`E8L_of_atomic`) and the other roots stay away from it. -/
theorem eventually_famF_atomic {S : Set P} {u : P → Fin 4 → ℂ} (hu : ContinuousAt u x₀)
    (hsep : ∀ i j, i ≠ j → u x₀ i ^ 2 - u x₀ j ^ 2 ≠ 0) {t : Bool}
    (hg : ∀ᶠ x in 𝓝[S] x₀, (famF U β μ h hβ x).LGeneric k t (u x)) {j : Fin 4}
    (hj : (famF U β μ h hβ x₀).Atomic k (u x₀ j)) :
    ∀ᶠ x in 𝓝[S] x₀, (famF U β μ h hβ x).Atomic k (u x j) := by
  have ha := continuousAt_famF_a hU (β := β) (μ := μ) (h := h) (hβ := hβ) (x₀ := x₀)
  have hne : ∀ᶠ x in 𝓝 x₀, ∀ i, i ≠ j → ((famF U β μ h hβ x).a : ℂ) ^ 2 - u x i ^ 2 ≠ 0 :=
    eventually_all.2 fun i ↦ eventually_imp_distrib_left.2 fun hij ↦
      ((ha.pow 2).sub ((continuousAt_pi.1 hu i).pow 2)).eventually_ne (by
        change ((famF U β μ h hβ x₀).a : ℂ) ^ 2 - u x₀ i ^ 2 ≠ 0
        rw [← hj.2]
        exact hsep j i (Ne.symm hij))
  filter_upwards [hg, nhdsWithin_le_nhds hne] with x hx hxn
  refine ⟨hj.1, ?_⟩
  have hE := (famF U β μ h hβ x).E8L_of_atomic k (u := (famF U β μ h hβ x).a) ⟨hj.1, rfl⟩
  rw [(famF U β μ h hβ x).E8L_eq_prod k hx.root hx.sep] at hE
  have hp : ∏ i, (((famF U β μ h hβ x).a : ℂ) ^ 2 - u x i ^ 2) = 0 := by
    rw [Fin.prod_univ_four]
    exact hE
  obtain ⟨i, -, hi⟩ := Finset.prod_eq_zero_iff.1 hp
  by_cases hij : i = j
  · subst hij
    exact (sub_eq_zero.1 hi).symm
  · exact absurd hi (hxn i hij)

include hU hb hm hh

/-- `κ_D = β sinhc(βμ_t)/Z²` is continuous. -/
theorem continuousAt_famF_κDL (t : Bool) :
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).κDL t : ℂ)) x₀ := by
  have hs := Params.continuous_dslope_sinh
  have hr : ContinuousAt (fun x ↦ (famF U β μ h hβ x).κDL t) x₀ := by
    cases t <;>
      simp only [κDL, cL, famF, Z, boltz₀, boltzUp, boltzDn, boltz₂, Bool.cond_true,
        Bool.cond_false] <;>
      fun_prop (disch := positivity)
  exact continuous_ofReal.continuousAt.comp hr

/-- `D_t` is continuous. -/
theorem continuousAt_famF_DL (t : Bool) :
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).DL t : ℂ)) x₀ := by
  have p₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have pu := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have pd := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have p₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  cases t <;> simp only [DL, pL_true, pL_false, Bool.not_true, Bool.not_false] <;> push_cast <;>
    fun_prop

/-- **The kernel `D_t/(z + v)` is continuous**, for fixed indices: on a mirror pair it is `κ_D`,
elsewhere `D_t/(z + v)` with `z + v ≠ 0`. -/
theorem continuousAt_famF_kerDL (t : Bool) (n m : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).kerDL k t n m) x₀ := by
  by_cases hmir : m = mirror k n
  · subst hmir
    simp only [kerDL_mirror]
    exact continuousAt_famF_κDL hU hb hm hh t
  · simp only [kerDL_of_ne_mirror _ _ t hmir]
    exact (continuousAt_famF_DL hU hb hm hh t).div
      ((continuousAt_famF_zL (U := U) hb hm hh t n).add
        (continuousAt_famF_zL (U := U) hb hm hh t m))
      (((famF U β μ h hβ x₀).BL t).z_add_z_ne_zero_of_ne k hmir)

/-- **`f_{t,σ±}(v(x))` are continuous**, at a moving point `v(x)` off the lattice of `t` at `x₀`. -/
theorem continuousAt_famF_fM (t σ : Bool) {v : P → ℂ} (hv : ContinuousAt v x₀)
    (hoff : ∀ n, ((famF U β μ h hβ x₀).BL t).z k n ≠ v x₀) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).fMp k t σ (v x)) x₀ ∧
      ContinuousAt (fun x ↦ (famF U β μ h hβ x).fMm k t σ (v x)) x₀ := by
  have hS := continuousAt_famF_S hb hm hh t hv hoff
  obtain ⟨ha, hβc, -, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hq := continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k)
  have p₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have pu := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have pd := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have p₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  constructor <;>
  cases t <;> cases σ <;>
    simp only [fMp, fMm, QL, αL, αSL, αPL, HL, lamL, piP, piS, DL, cLw, cS, cP, mL, rL, ρ,
      pL_true, pL_false, Bool.not_true, Bool.not_false, Bool.false_eq_true, Bool.true_eq_false,
      ↓reduceIte] at hS ⊢ <;>
    push_cast <;> fun_prop

/-- `E₈(v(x))` is continuous. -/
theorem continuousAt_famF_E8L {v : P → ℂ} (hv : ContinuousAt v x₀) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).E8L k (v x)) x₀ := by
  have hc := continuousAt_famF_E8c hU hb hm hh (hβ := hβ) (k := k)
  have h0 := continuousAt_pi.1 hc 0
  have h1 := continuousAt_pi.1 hc 1
  have h2 := continuousAt_pi.1 hc 2
  have h3 := continuousAt_pi.1 hc 3
  have h4 := continuousAt_pi.1 hc 4
  simp only [E8L_eq_sum, Fin.sum_univ_five]
  fun_prop

/-- The polynomial data at a moving point: `a_t(v)`, `Ξ(v²)`, `Π(v²)`, `P(v)`. -/
theorem continuousAt_famF_poly (t : Bool) {v : P → ℂ} (hv : ContinuousAt v x₀) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).aAL k t (v x)) x₀ ∧
      ContinuousAt (fun x ↦ (famF U β μ h hβ x).XiY k (v x ^ 2)) x₀ ∧
      ContinuousAt (fun x ↦ (famF U β μ h hβ x).PiY k (v x ^ 2)) x₀ ∧
      ContinuousAt (fun x ↦ (famF U β μ h hβ x).PL k (v x)) x₀ := by
  obtain ⟨ha, -, hw, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hq := continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k)
  have hD := continuousAt_famF_DL hU hb hm hh (hβ := hβ) t
  have p₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have pu := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have pd := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have p₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  have hρ := continuousAt_famF_ρ hU hb hm hh (hβ := hβ)
  have hmf := continuousAt_famF_mf hU hb hm hh (hβ := hβ)
  have hrU : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).rUp : ℂ)) x₀ := by
    simp only [rUp]
    push_cast
    exact hρ.sub hmf
  have hrD : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).rDn : ℂ)) x₀ := by
    simp only [rDn]
    push_cast
    exact hρ.add hmf
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [aAL, lamL]
    fun_prop
  · simp only [XiY, lamL, piP, piS]
    fun_prop
  · simp only [PiY, lamL, piP, piS]
    fun_prop
  · simp only [PL]
    fun_prop

/-- **The mode coefficients converge to `cModeAll`**: along `𝓝[S] x₀`, through points of the domain
of Eq. (field-L) for the column spin `t`, with roots `u(x)` continuous at `x₀` where `LAxisGeneric`
holds, `cMode(x) → cModeAll(x₀)`. At the atomic root both vanish; where `K_t(u_j) ≠ 0` at `x₀`,
`cMode` is continuous; where `K_t(u_j) = 0`, `K_{t,-t}(u_j) ≠ 0`, and near `x₀` `cMode` is the
quotient form (`cMode_eq_cModeQ`), which is continuous. -/
theorem tendsto_famF_cMode {S : Set P} {u : P → Fin 4 → ℂ} (hu : ContinuousAt u x₀)
    (h₀ : (famF U β μ h hβ x₀).LAxisGeneric k (u x₀)) {t : Bool}
    (hg : ∀ᶠ x in 𝓝[S] x₀, (famF U β μ h hβ x).LGeneric k t (u x)) (j : Fin 4) :
    Tendsto (fun x ↦ (famF U β μ h hβ x).cMode k t (u x) j) (𝓝[S] x₀)
      (𝓝 ((famF U β μ h hβ x₀).cModeAll k t (u x₀) j)) := by
  have huj := continuousAt_pi.1 hu j
  by_cases hat : (famF U β μ h hβ x₀).Atomic k (u x₀ j)
  · have e₀ : (famF U β μ h hβ x₀).cModeAll k t (u x₀) j = 0 := by
      unfold cModeAll
      split_ifs <;> simp [cModeQ, cMode, hat]
    rw [e₀]
    refine tendsto_const_nhds.congr' ?_
    filter_upwards [eventually_famF_atomic hU hu h₀.sep hg hat] with x hx
    simp [cMode, hx]
  have hna := nhdsWithin_le_nhds (s := S) (eventually_famF_not_atomic hU huj hat)
  have hoff : ∀ l, ((famF U β μ h hβ x₀).BL t).z k l ≠ u x₀ j := fun l ↦ (h₀.z_ne t l j).1
  have hd := continuousAt_dprod hu j
  have hd0 : dprod (fun i ↦ u x₀ i ^ 2) j ≠ 0 := dprod_ne h₀.sep j
  have hβc := (continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)).2.1
  have hβ0 : ((famF U β μ h hβ x₀).β : ℂ) ^ 2 ≠ 0 := pow_ne_zero 2 β_ne_zeroT
  have hKt := continuousAt_famF_KL hU hb hm hh t t huj hoff
  obtain ⟨hA, hXi, hPi, -⟩ := continuousAt_famF_poly hU hb hm hh (hβ := hβ) (k := k) t huj
  by_cases hK : (famF U β μ h hβ x₀).KL k t t (u x₀ j) = 0
  · have hK' := ((h₀.K_ne t j hat).resolve_left (fun h ↦ h hK)).1
    have hKn := continuousAt_famF_KL hU hb hm hh t (!t) huj hoff
    have hρ := continuousAt_famF_ρ hU hb hm hh (hβ := hβ)
    have hmL := continuousAt_famF_mL hU hb hm hh (hβ := hβ) t
    have hlam : ContinuousAt (fun x ↦ (famF U β μ h hβ x).lamL) x₀ := by
      obtain ⟨ha, -, -, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
      simp only [lamL]
      fun_prop
    have hQ := ((hlam.mul ((hρ.mul hXi).add (hmL.mul hPi))).neg).div
      ((hβc.pow 2).mul hKn |>.mul hd) (mul_ne_zero (mul_ne_zero hβ0 hK') hd0)
    have e₀ : (famF U β μ h hβ x₀).cModeAll k t (u x₀) j =
        (famF U β μ h hβ x₀).cModeQ k t (u x₀) j := by
      rw [cModeAll, ite_eq_left hK]
    rw [e₀, cModeQ, ite_eq_right hat]
    refine (hQ.tendsto.mono_left nhdsWithin_le_nhds).congr' ?_
    filter_upwards [hg, nhdsWithin_le_nhds (hKn.eventually_ne hK'), hna] with x hx hx' hxa
    rw [cMode_eq_cModeQ (hx.root j) (hx.K_ne j hxa) hx', cModeQ, ite_eq_right hxa]
    rfl
  · have hC := hA.div ((hβc.pow 2).mul hKt |>.mul hd) (mul_ne_zero (mul_ne_zero hβ0 hK) hd0)
    have e₀ : (famF U β μ h hβ x₀).cModeAll k t (u x₀) j =
        (famF U β μ h hβ x₀).cMode k t (u x₀) j := by
      rw [cModeAll, ite_eq_right hK]
    rw [e₀, cMode, ite_eq_right hat]
    refine (hC.tendsto.mono_left nhdsWithin_le_nhds).congr' ?_
    filter_upwards [hna] with x hxa
    rw [cMode, ite_eq_right hxa]
    rfl

/-- **`gammaLc` is continuous in the parameters, the roots and the coefficients**: with roots
`u(x)` continuous at `x₀` where `LAxisGeneric` holds and coefficients `c(x) → c₀` along `l ≤ 𝓝 x₀`,
`gammaLc(x, u(x), c(x)) → gammaLc(x₀, u(x₀), c₀)`. The kernels are continuous through the axes,
`E₈ ≠ 0` on the lattices and the roots are off them at `x₀`. -/
theorem tendsto_famF_gammaLc {l : Filter P} (hl : l ≤ 𝓝 x₀) {u : P → Fin 4 → ℂ}
    (hu : ContinuousAt u x₀) (h₀ : (famF U β μ h hβ x₀).LAxisGeneric k (u x₀))
    {c : P → Fin 4 → ℂ} {c₀ : Fin 4 → ℂ} (hc : Tendsto c l (𝓝 c₀)) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) :
    Tendsto (fun x ↦ (famF U β μ h hβ x).gammaLc k (u x) (c x) σ n t m) l
      (𝓝 ((famF U β μ h hβ x₀).gammaLc k (u x₀) c₀ σ n t m)) := by
  obtain ⟨ha, hβc, hw, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hq := continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k)
  have p₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have pu := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have pd := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have p₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  have hrt := continuousAt_famF_rL hU hb hm hh (hβ := hβ) t
  have hrn := continuousAt_famF_rL hU hb hm hh (hβ := hβ) (!t)
  have hz := continuousAt_famF_zL (U := U) hb hm hh (hβ := hβ) (k := k) σ n
  have hv := continuousAt_famF_zL (U := U) hb hm hh (hβ := hβ) (k := k) t m
  have hEv := continuousAt_famF_E8L hU hb hm hh (hβ := hβ) (k := k) hv
  have hEv0 := h₀.E8L_ne t m
  have hPv := (continuousAt_famF_poly hU hb hm hh (hβ := hβ) (k := k) t hv).2.2.2
  unfold gammaLc
  refine Tendsto.sub ((ContinuousAt.tendsto ?_).mono_left hl)
    (Tendsto.mul ((hβc.pow 2).tendsto.mono_left hl)
      (tendsto_finsetSum _ fun j _ ↦ (tendsto_pi_nhds.1 hc j).mul
        ((ContinuousAt.tendsto ?_).mono_left hl)))
  · by_cases hσ : σ = t
    · simp only [hσ, ↓reduceIte]
      have hz' := continuousAt_famF_zL (U := U) hb hm hh (hβ := hβ) (k := k) t n
      have hEz := continuousAt_famF_E8L hU hb hm hh (hβ := hβ) (k := k) hz'
      have hEz0 := h₀.E8L_ne t n
      have hb0 := (famF U β μ h hβ x₀).bL_num_ne k t n
      have hPz := (continuousAt_famF_poly hU hb hm hh (hβ := hβ) (k := k) t hz').2.2.2
      have hD := continuousAt_famF_kerDL hU hb hm hh (hβ := hβ) (k := k) t n m
      have hQ : ContinuousAt (fun x ↦ (famF U β μ h hβ x).QL k t
          (((famF U β μ h hβ x).BL t).z k n)) x₀ := by
        simp only [QL, HL, lamL, piP, piS]
        fun_prop
      have hlam : ContinuousAt (fun x ↦ (famF U β μ h hβ x).lamL) x₀ := by
        simp only [lamL]
        fun_prop
      fun_prop (disch := assumption)
    · simp only [hσ, ↓reduceIte]
      have hP := continuousAt_famF_kerPL hU hb hm hh (hβ := hβ) (k := k) n m
      have hS := continuousAt_famF_kerSLs hU hb hm hh (hβ := hβ) (k := k) σ n m
      have hX : ContinuousAt (fun x ↦ (famF U β μ h hβ x).HL k ((famF U β μ h hβ x).rL t)
            (-((famF U β μ h hβ x).BL t).z k m) * (famF U β μ h hβ x).HL k
              ((famF U β μ h hβ x).rL (!t)) (-((famF U β μ h hβ x).BL t).z k m) +
          (famF U β μ h hβ x).lamL ^ 2 * ((famF U β μ h hβ x).piP - (famF U β μ h hβ x).piS))
          x₀ := by
        simp only [HL, lamL, piP, piS]
        fun_prop
      have hY : ContinuousAt (fun x ↦ (famF U β μ h hβ x).HL k ((famF U β μ h hβ x).rL t)
            (-((famF U β μ h hβ x).BL t).z k m) * (famF U β μ h hβ x).HL k
              ((famF U β μ h hβ x).rL (!t)) (((famF U β μ h hβ x).BL t).z k m) -
          (famF U β μ h hβ x).lamL ^ 2 * ((famF U β μ h hβ x).piP - (famF U β μ h hβ x).piS))
          x₀ := by
        simp only [HL, lamL, piP, piS]
        fun_prop
      have hlam : ContinuousAt (fun x ↦ (famF U β μ h hβ x).lamL) x₀ := by
        simp only [lamL]
        fun_prop
      fun_prop (disch := assumption)
  · have huj := continuousAt_pi.1 hu j
    have hoff : ∀ l, ((famF U β μ h hβ x₀).BL t).z k l ≠ u x₀ j := fun l ↦ (h₀.z_ne t l j).1
    obtain ⟨hfp, hfm⟩ := continuousAt_famF_fM hU hb hm hh (hβ := hβ) (k := k) t σ huj hoff
    have hPu := (continuousAt_famF_poly hU hb hm hh (hβ := hβ) (k := k) t huj).2.2.2
    have h1 := h₀.sq_sub_ne t m j
    have h2 : ((famF U β μ h hβ x₀).BL σ).z k n - u x₀ j ≠ 0 :=
      sub_ne_zero.2 (h₀.z_ne σ n j).1
    have h3 : ((famF U β μ h hβ x₀).BL σ).z k n + u x₀ j ≠ 0 :=
      fun h0 ↦ (h₀.z_ne σ n j).2 (eq_neg_of_add_eq_zero_left h0)
    fun_prop (disch := assumption)

/-- **The closed form tends to the axis formula**: along `𝓝[S] x₀`, through points where
Eq. (field-L) holds for the column spin `t` with roots `u(x)` continuous at `x₀`, if `LAxisGeneric`
holds at `x₀` with the roots `u(x₀)`, then `gammaL → gammaLAll(x₀)`. No invertibility of `1 + 𝒦`
is assumed. -/
theorem LAxisGeneric.tendsto_gammaL_of_eventually {S : Set P} {u : P → Fin 4 → ℂ}
    (hu : ContinuousAt u x₀) (h₀ : (famF U β μ h hβ x₀).LAxisGeneric k (u x₀)) {t : Bool}
    (hg : ∀ᶠ x in 𝓝[S] x₀, (famF U β μ h hβ x).LGeneric k t (u x)) (σ : Bool) (n m : ℤ) :
    Tendsto (fun x ↦ (famF U β μ h hβ x).gammaL k (u x) σ n t m) (𝓝[S] x₀)
      (𝓝 ((famF U β μ h hβ x₀).gammaLAll k (u x₀) σ n t m)) := by
  have hc : Tendsto (fun x ↦ (famF U β μ h hβ x).cMode k t (u x)) (𝓝[S] x₀)
      (𝓝 ((famF U β μ h hβ x₀).cModeAll k t (u x₀))) :=
    tendsto_pi_nhds.2 fun j ↦ tendsto_famF_cMode hU hb hm hh hu h₀ hg j
  refine (tendsto_famF_gammaLc hU hb hm hh nhdsWithin_le_nhds hu h₀ hc σ n t m).congr' ?_
  filter_upwards [hg] with x hx
  exact (gammaL_eq_gammaLc hx.μ_ne hx.h_ne hx.c_ne _ σ n m).symm

/-- **The axis formula is the limit of Eq. (field-L)**: under `LAxisGeneric` at `x₀` (on the axes
`μ = 0`, `h = 0`, `μ = ±h` included), with the roots `u(x)` of
`LAxisGeneric.exists_eventually_LGeneric`, near `x₀` the points off the axes are in the domain of
Eq. (field-L), and there `gammaL → gammaLAll(x₀, u₀)`. -/
theorem LAxisGeneric.tendsto_gammaL_gammaLAll {u₀ : Fin 4 → ℂ}
    (h₀ : (famF U β μ h hβ x₀).LAxisGeneric k u₀) :
    ∃ u : P → Fin 4 → ℂ, ContinuousAt u x₀ ∧ u x₀ = u₀ ∧
      (∀ᶠ x in 𝓝[{x | (famF U β μ h hβ x).OffAxes}] x₀,
        ∀ t, (famF U β μ h hβ x).LGeneric k t (u x)) ∧
      ∀ σ n τ m, Tendsto (fun x ↦ (famF U β μ h hβ x).gammaL k (u x) σ n τ m)
        (𝓝[{x | (famF U β μ h hβ x).OffAxes}] x₀)
        (𝓝 ((famF U β μ h hβ x₀).gammaLAll k u₀ σ n τ m)) := by
  obtain ⟨u, huc, hu₀, hev⟩ := h₀.exists_eventually_LGeneric hU hb hm hh
  have hg := eventually_nhdsWithin_iff.2 hev
  subst hu₀
  exact ⟨u, huc, rfl, hg, fun σ n τ m ↦
    h₀.tendsto_gammaL_of_eventually hU hb hm hh huc (hg.mono fun _ hx ↦ hx τ) σ n m⟩

end Family

/-! ### Agreement with the operator vertex -/

/-- **The axis formula is the vertex**: at a point where `LAxisGeneric` holds (on the axes
`μ = 0`, `h = 0`, `μ = ±h` included) and `1 + 𝒦` is invertible, `Γ_L = gammaLAll`, entrywise.
Along the line `lineL` both are limits of `gammaL` (`LAxisGeneric.tendsto_gammaL_lineL`,
`LAxisGeneric.tendsto_gammaL_of_eventually`), and limits along `𝓝[≠] 0` are unique. -/
theorem LAxisGeneric.gammaOpLAll_eq {u₀ : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u₀)
    (hu : IsUnit (F.onePlusKLAll k)) (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n τ m = F.gammaLAll k u₀ σ n τ m := by
  have e : famF (fun _ ↦ F.U) (fun _ ↦ F.β) (fun s ↦ F.μ + s) (fun s ↦ F.h + 2 * s)
      (fun _ ↦ F.β_pos) 0 = F := F.lineL_zero
  have hU : ContinuousAt (fun _ : ℝ ↦ F.U) 0 := continuousAt_const
  have hb : ContinuousAt (fun _ : ℝ ↦ F.β) 0 := continuousAt_const
  have hm : ContinuousAt (fun s : ℝ ↦ F.μ + s) 0 := by fun_prop
  have hh : ContinuousAt (fun s : ℝ ↦ F.h + 2 * s) 0 := by fun_prop
  obtain ⟨u, huc, hu₀, hg, ht⟩ := h₀.tendsto_gammaL_lineL hu
  have h₀' : (famF (fun _ ↦ F.U) (fun _ ↦ F.β) (fun s ↦ F.μ + s) (fun s ↦ F.h + 2 * s)
      (fun _ ↦ F.β_pos) 0).LAxisGeneric k (u 0) := by
    rw [e, hu₀]
    exact h₀
  have h₁ := h₀'.tendsto_gammaL_of_eventually hU hb hm hh (S := {0}ᶜ) huc
    (hg.mono fun _ hx ↦ hx τ) σ n m
  rw [e, hu₀] at h₁
  exact tendsto_nhds_unique (ht σ n τ m) h₁

/-- **The axis formula is the limit along the line through an axis point**, without invertibility
of `1 + 𝒦`: if `LAxisGeneric` holds at `F`, there are roots `u(s)` of `E₈` along the line `lineL`,
continuous at `0` with `u(0) = u₀`, such that for `s ≠ 0` small `(F.lineL s, u(s))` is in the
domain of Eq. (field-L) for both column spins, and `gammaL → gammaLAll(F, u₀)` as `s → 0`,
`s ≠ 0`. -/
theorem LAxisGeneric.tendsto_gammaL_lineL_gammaLAll {u₀ : Fin 4 → ℂ} (h₀ : F.LAxisGeneric k u₀) :
    ∃ u : ℝ → Fin 4 → ℂ, ContinuousAt u 0 ∧ u 0 = u₀ ∧
      (∀ᶠ s in 𝓝[≠] (0 : ℝ), ∀ t, (F.lineL s).LGeneric k t (u s)) ∧
      ∀ σ n τ m, Tendsto (fun s ↦ (F.lineL s).gammaL k (u s) σ n τ m) (𝓝[≠] 0)
        (𝓝 (F.gammaLAll k u₀ σ n τ m)) := by
  have e : famF (fun _ ↦ F.U) (fun _ ↦ F.β) (fun s ↦ F.μ + s) (fun s ↦ F.h + 2 * s)
      (fun _ ↦ F.β_pos) 0 = F := F.lineL_zero
  have hU : ContinuousAt (fun _ : ℝ ↦ F.U) 0 := continuousAt_const
  have hb : ContinuousAt (fun _ : ℝ ↦ F.β) 0 := continuousAt_const
  have hm : ContinuousAt (fun s : ℝ ↦ F.μ + s) 0 := by fun_prop
  have hh : ContinuousAt (fun s : ℝ ↦ F.h + 2 * s) 0 := by fun_prop
  rw [← e] at h₀
  obtain ⟨u, huc, hu₀, hev⟩ := h₀.exists_eventually_LGeneric hU hb hm hh
  have hg : ∀ᶠ s in 𝓝[≠] (0 : ℝ), ∀ t, (F.lineL s).LGeneric k t (u s) := by
    filter_upwards [nhdsWithin_le_nhds hev, F.eventually_offAxes_lineL] with s hs ho
    exact hs ho
  refine ⟨u, huc, hu₀, hg, fun σ n τ m ↦ ?_⟩
  rw [← hu₀] at h₀
  have ht := h₀.tendsto_gammaL_of_eventually hU hb hm hh (S := {0}ᶜ) huc
    (hg.mono fun _ hx ↦ hx τ) σ n m
  rwa [e, hu₀] at ht

/-! ### The corner `μ = h = 0` -/

section Corner

variable (hμ : F.μ = 0) (hh : F.h = 0)
include hμ hh

/-- At `μ = h = 0`, `p₂ = p₀` and `p↓ = p↑`. -/
theorem corner_weights : F.p₂ = F.p₀ ∧ F.pDn = F.pUp := by
  refine ⟨?_, ?_⟩
  · rw [F.p₂_eq_p₀_mul, hμ, mul_zero, Real.exp_zero, mul_one]
  · rw [F.pDn_eq_p₀_mul, F.pUp_eq_p₀_mul, hh, sub_zero, add_zero]

/-- At `μ = h = 0`, `E₈(v) = ((v² + q₀)² - λ²(p₀ + p↑)²)((v² + q₀)² - λ²(p₀ - p↑)²)`. -/
theorem E8L_corner (v : ℂ) :
    F.E8L k v = ((v ^ 2 + F.q₀L k) ^ 2 - F.lamL ^ 2 * (F.p₀ + F.pUp) ^ 2) *
      ((v ^ 2 + F.q₀L k) ^ 2 - F.lamL ^ 2 * (F.p₀ - F.pUp) ^ 2) := by
  obtain ⟨h2, hd⟩ := corner_weights hμ hh
  have hrU : F.rUp = 0 := by simp [rUp, ρ, mf, h2, hd]
  have hrD : F.rDn = 0 := by simp [rDn, ρ, mf, h2, hd]
  simp only [E8L, HL, hrU, hrD, piP, piS, h2, hd]
  push_cast
  ring

/-- At `μ = h = 0`, `K_t(v) = -β(v² + q₀)((v² + q₀)² - λ²(p₀ - p↑)²)`: `D_t = 0`, `r_t = 0`. -/
theorem KL_self_corner (t : Bool) (v : ℂ) :
    F.KL k t t v = -F.β * (v ^ 2 + F.q₀L k) *
      ((v ^ 2 + F.q₀L k) ^ 2 - F.lamL ^ 2 * (F.p₀ - F.pUp) ^ 2) := by
  obtain ⟨h2, hd⟩ := corner_weights hμ hh
  have hpt : F.pL t = F.pUp := by cases t <;> simp [hd]
  have hpn : F.pL (!t) = F.pUp := by cases t <;> simp [hd]
  have hr : F.rL t = 0 := by simp [rL, mL, ρ, hpt, hpn, h2]
  have hrn : F.rL (!t) = 0 := by simp [rL, mL, ρ, hpn, h2, hpt]
  have hD : F.DL t = 0 := by simp [DL, h2, hpt, hpn]
  simp only [KL, fMp, fMm, ↓reduceIte, QL, αL, HL, hr, hrn, hD, cLw, hpt, hpn, h2, piP, piS, hd]
  push_cast
  ring

/-- At `μ = h = 0`, `K_{t,-t}(v) = λβ(p₀ + p↑)((v² + q₀)² - λ²(p₀ - p↑)²)`: `ρ = m = 0`. -/
theorem KL_opp_corner (t : Bool) (v : ℂ) :
    F.KL k t (!t) v = F.lamL * F.β * (F.p₀ + F.pUp) *
      ((v ^ 2 + F.q₀L k) ^ 2 - F.lamL ^ 2 * (F.p₀ - F.pUp) ^ 2) := by
  obtain ⟨h2, hd⟩ := corner_weights hμ hh
  have hpt : F.pL t = F.pUp := by cases t <;> simp [hd]
  have hpn : F.pL (!t) = F.pUp := by cases t <;> simp [hd]
  have hρ : F.ρ = 0 := by simp [ρ, h2]
  have hm : F.mL (!t) = 0 := by simp [mL, hpt, hpn]
  have hr : F.rL t = 0 := by simp [rL, mL, ρ, hpt, hpn, h2]
  have hrn : F.rL (!t) = 0 := by simp [rL, mL, ρ, hpn, h2, hpt]
  simp only [KL, fMp, fMm, Bool.not_ne_self, ↓reduceIte, αSL, αPL, HL, hr, hrn, hρ, hm, cS, cP,
    h2, hd, piP, piS]
  push_cast
  ring

/-- **At `μ = h = 0`, `LAxisGeneric` never holds**: `E₈` has the roots `u² = -q₀ ± λ(p₀ - p↑)`,
distinct as `U ≠ 0`, and at both `K_t = K_{t,-t} = 0` for both `t` (`KL_self_corner`,
`KL_opp_corner`), the simultaneous `0/0` of the note; at most one of them is atomic. So
`LAxisGeneric.gammaOpLAll_eq` says nothing at the corner, where the note uses
Eqs. (field-h0), (field-mu0) instead. -/
theorem not_lAxisGeneric_corner (u : Fin 4 → ℂ) : ¬F.LAxisGeneric k u := by
  intro h₀
  obtain ⟨h2, hd⟩ := corner_weights hμ hh
  set d : ℂ := F.lamL * (F.p₀ - F.pUp) with hd_def
  have hroot : ∀ y : ℂ, (y + F.q₀L k) ^ 2 = d ^ 2 → ∃ j, u j ^ 2 = y ∧
      F.KL k true true (u j) = 0 ∧ F.KL k true (!true) (u j) = 0 := by
    intro y hy
    obtain ⟨v, hv⟩ : ∃ v : ℂ, v ^ 2 = y := ⟨y ^ (2⁻¹ : ℂ), by simp⟩
    have hE : F.E8L k v = 0 := by
      rw [E8L_corner hμ hh, hv, hy, hd_def]
      ring
    rw [F.E8L_eq_prod k h₀.root h₀.sep, hv] at hE
    have hp : ∏ j, (y - u j ^ 2) = 0 := by
      rw [Fin.prod_univ_four]
      exact hE
    obtain ⟨j, -, hj⟩ := Finset.prod_eq_zero_iff.1 hp
    have hj' : u j ^ 2 = y := (sub_eq_zero.1 hj).symm
    refine ⟨j, hj', ?_, ?_⟩
    · rw [KL_self_corner hμ hh, hj', hy, hd_def]
      ring
    · rw [KL_opp_corner hμ hh, hj', hy, hd_def]
      ring
  obtain ⟨j₁, e₁, K₁, K₁'⟩ := hroot (-F.q₀L k + d) (by ring)
  obtain ⟨j₂, e₂, K₂, K₂'⟩ := hroot (-F.q₀L k - d) (by ring)
  have hd0 : d ≠ 0 := by
    have ha : F.a ≠ 0 := by exact_mod_cast h₀.a_ne
    have hp : F.p₀ - F.pUp ≠ 0 := by
      intro h0
      rw [F.pUp_eq_p₀_mul, hμ, hh, add_zero, add_zero] at h0
      have h1 : Real.exp (F.β * F.a) = 1 := by
        have h3 : F.p₀ * (1 - Real.exp (F.β * F.a)) = 0 := by linear_combination h0
        linarith [(mul_eq_zero.1 h3).resolve_left F.p₀_pos.ne']
      rw [Real.exp_eq_one_iff] at h1
      exact mul_ne_zero F.β_pos.ne' ha h1
    rw [hd_def, lamL]
    exact mul_ne_zero (mul_ne_zero (by norm_num) (pow_ne_zero 2 (by exact_mod_cast ha)))
      (by exact_mod_cast hp)
  have hna : ¬F.Atomic k (u j₁) ∨ ¬F.Atomic k (u j₂) := by
    by_contra hc
    simp only [not_or, not_not] at hc
    apply hd0
    have := e₁.symm.trans (hc.1.2.trans (hc.2.2.symm.trans e₂))
    linear_combination this / 2
  rcases hna with hn | hn
  · rcases h₀.K_ne true j₁ hn with h | ⟨h, -⟩
    · exact h K₁
    · exact h K₁'
  · rcases h₀.K_ne true j₂ hn with h | ⟨h, -⟩
    · exact h K₂
    · exact h K₂'

end Corner

end FParams

end HubbardAtom
