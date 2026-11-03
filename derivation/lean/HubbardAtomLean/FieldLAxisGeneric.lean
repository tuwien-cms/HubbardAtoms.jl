import HubbardAtomLean.FieldLAxisContinuity
import HubbardAtomLean.FieldLSym
import HubbardAtomLean.RootContinuity

/-!
# The domain of Eq. (field-L) accumulates at the axis points where `LAxisGeneric` holds

`FieldLAxisContinuity.lean` shows that along any approach to `x₀` through the domain of
Eq. (field-L) (`LGeneric` for both column spins) the closed form `gammaL` tends to `Γ_L(x₀)`. Here
such approaches are constructed at a point `x₀` on the axes `μ = 0`, `h = 0`, `μ = ±h`, under
the nondegeneracy condition `LAxisGeneric` at `x₀`. That `LAxisGeneric` holds at any given axis
point (the roots of `E₈` there and the values of `K`) is not shown; where it fails, the statements
here are vacuous; at `μ = h = 0` it never holds (`not_lAxisGeneric_corner`). Where it holds, the
note's explicit formula on the axes is the limit of `gammaL` and, if `1 + 𝒦` is invertible, `Γ_L`
(`FieldLAxisFormula.lean`). Not in the note.

* `exists_continuousAt_sqrt_root_of_simple₀`: `exists_continuousAt_sqrt_root_of_simple` without
  `u₀ ≠ 0` (at `u₀ = 0` the root is `s^{1/2}` of the moving simple root `s → 0`).
* `FParams.E8c`, `E8L_eq_sum`: `E₈(v) = ∑ᵢ cᵢ v²ⁱ`; `sum_E8c_deriv`: at four roots with distinct
  squares `s_j`, the derivative in `s` is `∏_{i ≠ j}(s_j - s_i) ≠ 0`, so each root is simple;
  `E8L_of_atomic`: at `k = 0` the atomic root `u² = a²` is a root for all parameters.
* `FParams.LAxisGeneric`: the condition at `x₀`: `U ≠ 0`, four roots of `E₈` with distinct squares,
  the non-atomic ones off both lattices, with `K_t ≠ 0`, or `K_{t,-t} ≠ 0` and `u² + q₀ ≠ 0`, for
  each column spin `t`. It does not ask `μ`, `h`, `μ ± h ≠ 0`, nor `K_t ≠ 0` on the axis `μ_t = 0`:
  there `D_t = 0`, and the quotient identity `res_z` forces `K_t = 0` unless `ρΞ + m_tΠ = 0`. The
  second alternative is the note's case of Eq. (field-L-quotient), `0/0` but `K_{j,-t} ≠ 0`. It
  holds on the domain of Eq. (field-L) (`LGeneric.lAxisGeneric`).
* `FParams.continuousAt_famF_KL`: along a family `famF` continuous at `x₀`, `K_{t,σ}(v(x))` is
  continuous at a moving point `v(x)` off the lattice of `t`; `exists_famF_E8L_root`: a simple root
  of `E₈` moves continuously, and the atomic root moves as `±a(x)` and stays atomic.
* `FParams.LAxisGeneric.exists_eventually_LGeneric`: **under `LAxisGeneric` at `x₀` there are roots
  `u(x)` of `E₈`, continuous at `x₀` with `u(x₀) = u₀`, such that near `x₀` every point off the axes
  is in the domain of Eq. (field-L) with the roots `u(x)`**, for both column spins: the `u_j²` stay
  distinct and the roots off the lattices by continuity, `K_t ≠ 0` by continuity or, from
  `K_{t,-t} ≠ 0` and `u² + q₀ ≠ 0`, by the quotient identity (`KL_opp_eq_zero`).
* `FParams.LAxisGeneric.tendsto_gammaL`: then, if `1 + 𝒦` is invertible at `x₀`, `gammaL` tends to
  `Γ_L(x₀)` along the points off the axes; `LAxisGeneric.tendsto_gammaL_lineL`: the same along the
  line `lineL`, `s ↦ (U, β, μ + s, h + 2s)`, which leaves every axis for `s ≠ 0` small
  (`eventually_offAxes_lineL`), as `s → 0`, `s ≠ 0`. So at such an axis point the vertex is the
  limit of the closed form along a concrete approach.
-/

open Complex Filter Topology

namespace HubbardAtom

/-- **A square root of a simple root moves continuously, also at `u₀ = 0`**: for coefficients
`c x` continuous at `x₀` and `u₀` with `u₀²` a simple root of `∑ᵢ cᵢ(x₀) sⁱ`, there is a root
`u(x)` of `∑ᵢ cᵢ(x) u²ⁱ` near `x₀`, continuous at `x₀` with `u x₀ = u₀`. For `u₀ ≠ 0` this is
`exists_continuousAt_sqrt_root_of_simple`; for `u₀ = 0`, `u = s^{1/2}` of the root `s → 0`. -/
theorem exists_continuousAt_sqrt_root_of_simple₀ {X : Type*} [TopologicalSpace X] {x₀ : X}
    {d : ℕ} {c : X → Fin (d + 1) → ℂ} (hc : ContinuousAt c x₀) {u₀ : ℂ}
    (hroot : ∑ i, c x₀ i * (u₀ ^ 2) ^ (i : ℕ) = 0)
    (hsimple : ∑ i : Fin (d + 1), (i : ℕ) * c x₀ i * (u₀ ^ 2) ^ ((i : ℕ) - 1) ≠ 0) :
    ∃ u : X → ℂ, u x₀ = u₀ ∧ ContinuousAt u x₀ ∧
      ∀ᶠ x in 𝓝 x₀, ∑ i, c x i * (u x ^ 2) ^ (i : ℕ) = 0 := by
  by_cases hu₀ : u₀ = 0
  · subst hu₀
    have e : (0 : ℂ) ^ 2 = 0 := by norm_num
    rw [e] at hroot hsimple
    obtain ⟨s, hs₀, hs, hroot', -⟩ := exists_continuousAt_root_of_simple hc hroot hsimple
    refine ⟨fun x ↦ s x ^ (2⁻¹ : ℂ), ?_, ?_, ?_⟩
    · simp [hs₀]
    · exact (continuousAt_cpow_const_of_re_pos (by simp [hs₀]) (by norm_num)).comp hs
    · filter_upwards [hroot'] with x hx
      have h2 : (s x ^ (2⁻¹ : ℂ)) ^ 2 = s x := by simp
      rwa [h2]
  · obtain ⟨u, hu₀', hu, hroot', -⟩ := exists_continuousAt_sqrt_root_of_simple hc hu₀ hroot hsimple
    exact ⟨u, hu₀', hu, hroot'⟩

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### `E₈` as a polynomial in `s = v²` -/

/-- At `k = 0`, `E₈(u) = 0` at the atomic root `u² = a²`. -/
theorem E8L_of_atomic {u : ℂ} (ha : F.Atomic k u) : F.E8L k u = 0 := by
  obtain ⟨rfl, hu⟩ := ha
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  have e : ∀ P R M : ℂ, LCert.E8W u F.a 0 P R M = (u ^ 2 - F.a ^ 2) * LCert.E6Z u F.a P R M :=
    fun _ _ _ ↦ by unfold LCert.E8W LCert.E6Z; ring
  rw [F.E8L_W 0 true u, hw, e, hu, sub_self, zero_mul]

/-- The coefficients of `E₈` as a polynomial in `s = v²`, from the constant one to the leading
`1`. -/
noncomputable def E8c : Fin 5 → ℂ := ![F.E8b k 0, F.E8b k 1, F.E8b k 2, F.E8b k 3, 1]

theorem sum_E8c (s : ℂ) :
    ∑ i, F.E8c k i * s ^ (i : ℕ) =
      s ^ 4 + F.E8b k 3 * s ^ 3 + F.E8b k 2 * s ^ 2 + F.E8b k 1 * s + F.E8b k 0 := by
  simp [Fin.sum_univ_five, E8c]
  ring

/-- `E₈(v) = ∑ᵢ cᵢ v²ⁱ`. -/
theorem E8L_eq_sum (v : ℂ) : F.E8L k v = ∑ i, F.E8c k i * (v ^ 2) ^ (i : ℕ) := by
  rw [sum_E8c, E8L_eq_quartic]

/-- **The roots are simple in `s`**: for four roots of `E₈` with distinct squares `s_j = u_j²`, the
derivative of `∑ᵢ cᵢ sⁱ = ∏ᵢ(s - s_i)` at `s_j` is `∏_{i ≠ j}(s_j - s_i)`. -/
theorem sum_E8c_deriv {u : Fin 4 → ℂ} (root : ∀ j, F.E8L k (u j) = 0)
    (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) (j : Fin 4) :
    ∑ i : Fin 5, (i : ℕ) * F.E8c k i * (u j ^ 2) ^ ((i : ℕ) - 1) = dprod (fun i ↦ u i ^ 2) j := by
  have hQ : (fun y ↦ ∑ i, F.E8c k i * y ^ (i : ℕ)) =
      fun y ↦ (y - u 0 ^ 2) * (y - u 1 ^ 2) * (y - u 2 ^ 2) * (y - u 3 ^ 2) := by
    funext y
    rw [sum_E8c]
    exact quartic_eq_prod
      (Q := fun y ↦ y ^ 4 + F.E8b k 3 * y ^ 3 + F.E8b k 2 * y ^ 2 + F.E8b k 1 * y + F.E8b k 0)
      (fun _ ↦ rfl) (sep 0 1 (by decide)) (sep 0 2 (by decide)) (sep 0 3 (by decide))
      (sep 1 2 (by decide)) (sep 1 3 (by decide)) (sep 2 3 (by decide))
      (by simpa only [F.E8L_eq_quartic] using root 0)
      (by simpa only [F.E8L_eq_quartic] using root 1)
      (by simpa only [F.E8L_eq_quartic] using root 2)
      (by simpa only [F.E8L_eq_quartic] using root 3) y
  have d1 := hasDerivAt_sum_mul_pow (F.E8c k) (u j ^ 2)
  rw [hQ] at d1
  have d2 := ((((hasDerivAt_id' (u j ^ 2)).sub_const (u 0 ^ 2)).fun_mul
    ((hasDerivAt_id' (u j ^ 2)).sub_const (u 1 ^ 2))).fun_mul
      ((hasDerivAt_id' (u j ^ 2)).sub_const (u 2 ^ 2))).fun_mul
        ((hasDerivAt_id' (u j ^ 2)).sub_const (u 3 ^ 2))
  rw [d1.unique d2]
  match j with
  | 0 => simp only [dprod]; ring
  | 1 => simp only [dprod]; ring
  | 2 => simp only [dprod]; ring
  | 3 => simp only [dprod]; ring

/-! ### The assumptions at an axis point -/

/-- The assumptions at `F`, on the axes included, under which the points off the axes near `F` are
in the domain of Eq. (field-L) (`LAxisGeneric.exists_eventually_LGeneric`), so that the domain
accumulates at `F`, e.g. along `lineL`: `U ≠ 0`; four roots `u_j` of `E₈` with distinct `u_j²`; at
each non-atomic root, `u_j` off both lattices and, for each column spin `t`, either
`K_t(u_j) ≠ 0` or `K_{t,-t}(u_j) ≠ 0` and `u_j² + q₀ ≠ 0`. The second alternative is the one for
the axis `μ_t = 0`: there `D_t = 0`, and the quotient identity `res_z` forces `K_t(u_j) = 0` unless
`ρΞ(u_j²) + m_tΠ(u_j²) = 0`. Wherever `μ_t ≠ 0`, it gives back `K_t(u_j) ≠ 0` (`KL_opp_eq_zero`).
`LGeneric` for both column spins implies it (`LGeneric.lAxisGeneric`). Whether it holds at a
given axis point is not shown. -/
structure LAxisGeneric (u : Fin 4 → ℂ) : Prop where
  a_ne : (F.a : ℂ) ≠ 0
  root : ∀ j, F.E8L k (u j) = 0
  sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0
  off : ∀ j, ¬F.Atomic k (u j) → ∀ σ l, (F.BL σ).z k l ≠ u j ∧ (F.BL σ).z k l ≠ -u j
  K_ne : ∀ t j, ¬F.Atomic k (u j) →
    F.KL k t t (u j) ≠ 0 ∨ F.KL k t (!t) (u j) ≠ 0 ∧ u j ^ 2 + F.q₀L k ≠ 0

variable {F k} in
/-- The domain of Eq. (field-L), for both column spins, lies in that of `LAxisGeneric`. -/
theorem LGeneric.lAxisGeneric {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) :
    F.LAxisGeneric k u :=
  ⟨(hg true).a_ne, (hg true).root, (hg true).sep, (hg true).off,
    fun t j hj ↦ Or.inl ((hg t).K_ne j hj)⟩

/-! ### Along a family -/

section Family

variable {P : Type*} [TopologicalSpace P] {U β μ h : P → ℝ} {hβ : ∀ x, 0 < β x} {k : ℤ}
  {x₀ : P} (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)
  (hh : ContinuousAt h x₀)

include hU hb hm hh

/-- The coefficients `cᵢ` of `E₈` in `s = v²` are continuous. -/
theorem continuousAt_famF_E8c : ContinuousAt (fun x ↦ (famF U β μ h hβ x).E8c k) x₀ := by
  obtain ⟨ha, -, -, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hq := continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k)
  have p₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have pu := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have pd := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have p₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  refine continuousAt_pi.2 fun i ↦ ?_
  fin_cases i <;> simp [E8c, E8b, lamL, piS, piP, rUp, rDn, ρ, mf] <;> fun_prop

omit hU in
/-- `S_k` on the lattice of the spin `σ`, at a moving point off that lattice at `x₀`, is
continuous. -/
theorem continuousAt_famF_S (σ : Bool) {v : P → ℂ} (hv : ContinuousAt v x₀)
    (hoff : ∀ n, ((famF U β μ h hβ x₀).BL σ).z k n ≠ v x₀) :
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).BL σ).S k (v x)) x₀ :=
  Params.continuousAt_S (fun x ↦ (famF U β μ h hβ x).BL σ) k (continuousAt_famF_cL hm hh σ) hb hv
    hoff

/-- **`K_{t,σ}(v(x))` is continuous**, at a moving point `v(x)` off the lattice of `t` at `x₀`:
`K_{t,σ}` is a polynomial in the data and in `S_t`. -/
theorem continuousAt_famF_KL (t σ : Bool) {v : P → ℂ} (hv : ContinuousAt v x₀)
    (hoff : ∀ n, ((famF U β μ h hβ x₀).BL t).z k n ≠ v x₀) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).KL k t σ (v x)) x₀ := by
  have hS := continuousAt_famF_S hb hm hh t hv hoff
  obtain ⟨ha, hβc, -, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hq := continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k)
  have p₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have pu := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have pd := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have p₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  cases t <;> cases σ <;>
    simp only [KL, fMp, fMm, QL, αL, αSL, αPL, HL, lamL, piP, piS, DL, cLw, cS, cP, mL, rL, ρ,
      pL_true, pL_false, Bool.not_true, Bool.not_false, Bool.false_eq_true, Bool.true_eq_false,
      ↓reduceIte] at hS ⊢ <;>
    push_cast <;> fun_prop

/-- **A root of `E₈` along the family**: a root `u₀` of `E₈` at `x₀`, simple in `s = u²` (as each
of four roots with distinct squares is, `sum_E8c_deriv`), is the value at `x₀` of a root `v(x)`
continuous at `x₀`; at the atomic root `u₀ = ±a` (`k = 0`) the root `v = ±a` stays atomic. -/
theorem exists_famF_E8L_root {u₀ : ℂ} (ha : ((famF U β μ h hβ x₀).a : ℂ) ≠ 0)
    (hr : (famF U β μ h hβ x₀).E8L k u₀ = 0)
    (hs : ∑ i : Fin 5, (i : ℕ) * (famF U β μ h hβ x₀).E8c k i * (u₀ ^ 2) ^ ((i : ℕ) - 1) ≠ 0) :
    ∃ v : P → ℂ, v x₀ = u₀ ∧ ContinuousAt v x₀ ∧ ∀ᶠ x in 𝓝 x₀,
      (famF U β μ h hβ x).E8L k (v x) = 0 ∧
        ((famF U β μ h hβ x₀).Atomic k u₀ → (famF U β μ h hβ x).Atomic k (v x)) := by
  by_cases hat : (famF U β μ h hβ x₀).Atomic k u₀
  · have hA := continuousAt_famF_a hU (β := β) (μ := μ) (h := h) (hβ := hβ) (x₀ := x₀)
    refine ⟨fun x ↦ u₀ / (famF U β μ h hβ x₀).a * (famF U β μ h hβ x).a, ?_,
      continuousAt_const.mul hA, Eventually.of_forall fun x ↦ ?_⟩
    · exact div_mul_cancel₀ u₀ ha
    · have hx : (famF U β μ h hβ x).Atomic k
          (u₀ / (famF U β μ h hβ x₀).a * (famF U β μ h hβ x).a) := by
        refine ⟨hat.1, ?_⟩
        rw [mul_pow, div_pow, hat.2, div_self (pow_ne_zero 2 ha), one_mul]
      exact ⟨(famF U β μ h hβ x).E8L_of_atomic k hx, fun _ ↦ hx⟩
  · obtain ⟨v, hv₀, hv, hroot⟩ := exists_continuousAt_sqrt_root_of_simple₀
      (continuousAt_famF_E8c hU hb hm hh (hβ := hβ) (k := k)) (by rwa [← E8L_eq_sum]) hs
    exact ⟨v, hv₀, hv, hroot.mono fun x hx ↦ ⟨by rwa [E8L_eq_sum], fun hA ↦ absurd hA hat⟩⟩

/-- **The domain of Eq. (field-L) accumulates at `x₀`**: along a family `(U, β, μ, h)(x)` continuous
at `x₀`, if `LAxisGeneric` holds at `x₀` with the roots `u₀` (on the axes `μ = 0`, `h = 0`,
`μ = ±h` included), then there are roots `u(x)` of `E₈`, continuous at `x₀` with `u(x₀) = u₀`, such
that near `x₀` every point off the axes is in the domain of Eq. (field-L) with the roots `u(x)`, for
both column spins. -/
theorem LAxisGeneric.exists_eventually_LGeneric {u₀ : Fin 4 → ℂ}
    (h₀ : (famF U β μ h hβ x₀).LAxisGeneric k u₀) :
    ∃ u : P → Fin 4 → ℂ, ContinuousAt u x₀ ∧ u x₀ = u₀ ∧ ∀ᶠ x in 𝓝 x₀,
      (famF U β μ h hβ x).OffAxes → ∀ t, (famF U β μ h hβ x).LGeneric k t (u x) := by
  have hpath : ∀ j, ∃ v : P → ℂ, v x₀ = u₀ j ∧ ContinuousAt v x₀ ∧ ∀ᶠ x in 𝓝 x₀,
      (famF U β μ h hβ x).E8L k (v x) = 0 ∧
        ((famF U β μ h hβ x₀).Atomic k (u₀ j) → (famF U β μ h hβ x).Atomic k (v x)) := fun j ↦
    exists_famF_E8L_root hU hb hm hh h₀.a_ne (h₀.root j)
      (by rw [sum_E8c_deriv _ k h₀.root h₀.sep j]; exact dprod_ne h₀.sep j)
  choose v hv₀ hvc hvr using hpath
  refine ⟨fun x j ↦ v j x, continuousAt_pi.2 hvc, funext hv₀, ?_⟩
  -- `u_j²` stay distinct
  have e_sep : ∀ᶠ x in 𝓝 x₀, ∀ i j, i ≠ j → v i x ^ 2 - v j x ^ 2 ≠ 0 :=
    eventually_all.2 fun i ↦ eventually_all.2 fun j ↦ eventually_imp_distrib_left.2 fun hij ↦
      (((hvc i).pow 2).sub ((hvc j).pow 2)).eventually_ne (by
        change v i x₀ ^ 2 - v j x₀ ^ 2 ≠ 0
        rw [hv₀, hv₀]
        exact h₀.sep i j hij)
  -- the roots non-atomic at `x₀` stay off the lattices
  have e_off : ∀ᶠ x in 𝓝 x₀, ∀ j, ¬(famF U β μ h hβ x₀).Atomic k (u₀ j) → ∀ σ l,
      ((famF U β μ h hβ x).BL σ).z k l ≠ v j x ∧ ((famF U β μ h hβ x).BL σ).z k l ≠ -v j x := by
    refine eventually_all.2 fun j ↦ eventually_imp_distrib_left.2 fun hj ↦
      eventually_all.2 fun σ ↦ ?_
    have h1 := Params.eventually_z_ne_of_off (fun x ↦ (famF U β μ h hβ x).BL σ) k
      (continuousAt_famF_cL hm hh σ) hb (hvc j) fun n ↦ by
        rw [hv₀]
        exact (h₀.off j hj σ n).1
    have h2 := Params.eventually_z_ne_of_off (fun x ↦ (famF U β μ h hβ x).BL σ) k
      (continuousAt_famF_cL hm hh σ) hb (hvc j).neg fun n ↦ by
        rw [Pi.neg_apply, hv₀]
        exact (h₀.off j hj σ n).2
    filter_upwards [h1, h2] with x hx1 hx2 l
    exact ⟨hx1 l, hx2 l⟩
  -- the alternatives for `K`
  have e_K : ∀ᶠ x in 𝓝 x₀, ∀ t j, ¬(famF U β μ h hβ x₀).Atomic k (u₀ j) →
      (famF U β μ h hβ x).KL k t t (v j x) ≠ 0 ∨ (famF U β μ h hβ x).KL k t (!t) (v j x) ≠ 0 ∧
        v j x ^ 2 + (famF U β μ h hβ x).q₀L k ≠ 0 := by
    refine eventually_all.2 fun t ↦ eventually_all.2 fun j ↦ eventually_imp_distrib_left.2
      fun hj ↦ ?_
    have hoff₀ : ∀ n, ((famF U β μ h hβ x₀).BL t).z k n ≠ v j x₀ := fun n ↦ by
      rw [hv₀]
      exact (h₀.off j hj t n).1
    rcases h₀.K_ne t j hj with hK | ⟨hK, hq⟩
    · refine ((continuousAt_famF_KL hU hb hm hh t t (hvc j) hoff₀).eventually_ne ?_).mono
        fun _ hx ↦ Or.inl hx
      rwa [hv₀]
    · have h1 := (continuousAt_famF_KL hU hb hm hh t (!t) (hvc j) hoff₀).eventually_ne
        (by rwa [hv₀])
      have h2 := (((hvc j).pow 2).add
        (continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k))).eventually_ne (by
          change v j x₀ ^ 2 + (famF U β μ h hβ x₀).q₀L k ≠ 0
          rw [hv₀]
          exact hq)
      filter_upwards [h1, h2] with x hx1 hx2
      exact Or.inr ⟨hx1, hx2⟩
  filter_upwards [eventually_all.2 hvr, (continuousAt_famF_a hU).eventually_ne h₀.a_ne, e_sep,
    e_off, e_K] with x hr ha hsep hoff hK hx t
  -- a root non-atomic at `x` is non-atomic at `x₀`
  have hna : ∀ j, ¬(famF U β μ h hβ x).Atomic k (v j x) →
      ¬(famF U β μ h hβ x₀).Atomic k (u₀ j) := fun j hj h ↦ hj ((hr j).2 h)
  refine ⟨ha, hx.μ_ne, hx.h_ne, hx.c_ne t, fun j ↦ (hr j).1, hsep, fun j hj ↦ hoff j (hna j hj),
    fun j hj ↦ ?_⟩
  rcases hK t j (hna j hj) with hK | ⟨hK, hq⟩
  · exact hK
  · exact fun h0 ↦ hK ((famF U β μ h hβ x).KL_opp_eq_zero k t ha (hx.c_ne t) hq (hr j).1 h0)

/-- **The closed form tends to the vertex at an axis point**: if `LAxisGeneric` holds at `x₀` and
`1 + 𝒦` is invertible there, then with the roots `u(x)` of `exists_eventually_LGeneric`, near `x₀`
the points off the axes are in the domain of Eq. (field-L), and there `gammaL` tends to `Γ_L(x₀)`
as `x → x₀`. This is vacuous unless points off the axes accumulate at `x₀` along the family;
`tendsto_gammaL_lineL` is a non-vacuous instance. -/
theorem LAxisGeneric.tendsto_gammaL {u₀ : Fin 4 → ℂ} (h₀ : (famF U β μ h hβ x₀).LAxisGeneric k u₀)
    (hu : IsUnit ((famF U β μ h hβ x₀).onePlusKLAll k)) :
    ∃ u : P → Fin 4 → ℂ, ContinuousAt u x₀ ∧ u x₀ = u₀ ∧
      (∀ᶠ x in 𝓝[{x | (famF U β μ h hβ x).OffAxes}] x₀,
        ∀ t, (famF U β μ h hβ x).LGeneric k t (u x)) ∧
      ∀ σ n τ m, Tendsto (fun x ↦ (famF U β μ h hβ x).gammaL k (u x) σ n τ m)
        (𝓝[{x | (famF U β μ h hβ x).OffAxes}] x₀)
        (𝓝 ((famF U β μ h hβ x₀).gammaOpLAll k σ n τ m)) := by
  obtain ⟨u, huc, hu₀, hev⟩ := h₀.exists_eventually_LGeneric hU hb hm hh
  have hg := eventually_nhdsWithin_iff.2 hev
  exact ⟨u, huc, hu₀, hg, fun σ n τ m ↦ tendsto_gammaL_of_LGeneric hU hb hm hh hu (x := id)
    (tendsto_id.mono_left nhdsWithin_le_nhds) hg σ n τ m⟩

end Family

/-! ### Along a line through an axis point -/

/-- The line `s ↦ (U, β, μ + s, h + 2s)` through `F`. For `s ≠ 0` small it is off the axes
(`eventually_offAxes_lineL`), wherever `F` is. -/
def lineL (F : FParams) (s : ℝ) : FParams := ⟨F.U, F.β, F.μ + s, F.h + 2 * s, F.β_pos⟩

theorem lineL_zero (F : FParams) : F.lineL 0 = F := by
  cases F
  simp only [lineL, add_zero, mul_zero]

/-- `c + ds ≠ 0` for `s ≠ 0` small, if `d ≠ 0`. -/
private theorem eventually_add_mul_ne (c : ℝ) {d : ℝ} (hd : d ≠ 0) :
    ∀ᶠ s in 𝓝[≠] (0 : ℝ), c + d * s ≠ 0 := by
  by_cases hc : c = 0
  · subst hc
    filter_upwards [self_mem_nhdsWithin] with s hs
    rw [zero_add]
    exact mul_ne_zero hd hs
  · have hcont : ContinuousAt (fun s : ℝ ↦ c + d * s) 0 := by fun_prop
    exact nhdsWithin_le_nhds (hcont.eventually_ne (by simpa using hc))

/-- **The line leaves every axis**: `μ + s`, `h + 2s`, `μ + h + 3s`, `μ - h - s` are nonzero for
`s ≠ 0` small. -/
theorem eventually_offAxes_lineL (F : FParams) : ∀ᶠ s in 𝓝[≠] (0 : ℝ), (F.lineL s).OffAxes := by
  filter_upwards [eventually_add_mul_ne F.μ one_ne_zero, eventually_add_mul_ne F.h two_ne_zero,
    eventually_add_mul_ne (F.μ + F.h) three_ne_zero,
    eventually_add_mul_ne (F.μ - F.h) (neg_ne_zero.2 one_ne_zero)] with s h1 h2 h3 h4
  refine ⟨by simpa [lineL] using h1, by simpa [lineL] using h2, fun σ ↦ ?_⟩
  cases σ
  · simp only [cL, lineL, Bool.cond_false]
    convert h4 using 1
    ring
  · simp only [cL, lineL, Bool.cond_true]
    convert h3 using 1
    ring

/-- **The closed form tends to the vertex along the line through an axis point**: if
`LAxisGeneric` holds at `F` (on the axes `μ = 0`, `h = 0`, `μ = ±h` included) and `1 + 𝒦` is
invertible there, then there are roots `u(s)` of `E₈` along the line `lineL`, continuous at `0` with
`u(0) = u₀`, such that for `s ≠ 0` small `(F.lineL s, u(s))` is in the domain of Eq. (field-L) for
both column spins, and `gammaL → Γ_L(F)` as `s → 0`, `s ≠ 0`. -/
theorem LAxisGeneric.tendsto_gammaL_lineL {F : FParams} {k : ℤ} {u₀ : Fin 4 → ℂ}
    (h₀ : F.LAxisGeneric k u₀) (hu : IsUnit (F.onePlusKLAll k)) :
    ∃ u : ℝ → Fin 4 → ℂ, ContinuousAt u 0 ∧ u 0 = u₀ ∧
      (∀ᶠ s in 𝓝[≠] (0 : ℝ), ∀ t, (F.lineL s).LGeneric k t (u s)) ∧
      ∀ σ n τ m, Tendsto (fun s ↦ (F.lineL s).gammaL k (u s) σ n τ m) (𝓝[≠] 0)
        (𝓝 (F.gammaOpLAll k σ n τ m)) := by
  have e : famF (fun _ ↦ F.U) (fun _ ↦ F.β) (fun s ↦ F.μ + s) (fun s ↦ F.h + 2 * s)
      (fun _ ↦ F.β_pos) 0 = F := F.lineL_zero
  have hU : ContinuousAt (fun _ : ℝ ↦ F.U) 0 := continuousAt_const
  have hb : ContinuousAt (fun _ : ℝ ↦ F.β) 0 := continuousAt_const
  have hm : ContinuousAt (fun s : ℝ ↦ F.μ + s) 0 := by fun_prop
  have hh : ContinuousAt (fun s : ℝ ↦ F.h + 2 * s) 0 := by fun_prop
  rw [← e] at h₀ hu
  obtain ⟨u, huc, hu₀, hev⟩ := h₀.exists_eventually_LGeneric hU hb hm hh
  have hg : ∀ᶠ s in 𝓝[≠] (0 : ℝ), ∀ t, (F.lineL s).LGeneric k t (u s) := by
    filter_upwards [nhdsWithin_le_nhds hev, F.eventually_offAxes_lineL] with s hs ho
    exact hs ho
  refine ⟨u, huc, hu₀, hg, fun σ n τ m ↦ ?_⟩
  have ht := tendsto_gammaL_of_LGeneric hU hb hm hh hu (x := id)
    (tendsto_id.mono_left nhdsWithin_le_nhds) hg σ n τ m
  rwa [e] at ht

end FParams

end HubbardAtom
