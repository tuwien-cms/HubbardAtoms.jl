import HubbardAtomLean.FieldLAxisOperator
import HubbardAtomLean.FieldTContinuity
import HubbardAtomLean.SingletContinuity

/-!
# Continuity of the longitudinal vertex for every `μ`, `h`

The second half of item 6(d) for the longitudinal sector, through the axes: along a family of
parameters `(U, β, μ, h)(x)` continuous at `x₀` (`famF`), the vertex `Γ_L` of
`FieldLAxisOperator.lean` is continuous at `x₀` wherever `1 + 𝒦` is invertible there, on the axes
`μ = 0`, `h = 0`, `μ = ±h` included.

* The lattice of spin `σ` along the family is the zero-field family at `μ_σ` (`BL_famF`), so the
  lattice lemmas of `GammaContinuity.lean`, `DMContinuity.lean` and `SingletContinuity.lean`
  apply to each spin.
* **Each entry of `χ_L` and of the bubble is continuous** (`continuousAt_famF_chiL`,
  `continuousAt_famF_bL`). The kernels of `PL_mul_chiL_opp_all` are, for fixed indices, either
  `ρ/(z + v)`, `σm/(z - v)` with a denominator that never vanishes, or `κ_P`, `κ_S`, which are
  continuous through `μ = 0`, `h = 0` (`continuousAt_famF_kerPL`, `continuousAt_famF_kerSLs`).
* **The bounds of `FieldLAxisBounds.lean` hold uniformly near `x₀`**
  (`exists_eventually_chiLe_bounds`): the scalar constants are continuous, the factors off the
  fixed point are bounded by `exists_eventually_bdd_of_nonfixed`, which avoids the moved point
  (where `P(z')` may vanish), and `N₀`, `N₂` and the diagonal at the fixed point are continuous.
* `FParams.contHyp_LAll`: the hypotheses of `ChiContinuity.lean`;
  `FParams.continuousAt_gammaOpLAll`: **`Γ_L` is continuous** at `x₀` wherever `1 + 𝒦` is
  invertible there, in particular on the domain of Eq. (field-L)
  (`continuousAt_gammaOpLAll_of_LGeneric`);
* `FParams.tendsto_gammaL_of_LGeneric`: then along any approach to `x₀` through points of the
  domain of Eq. (field-L), the closed form `gammaL` tends to `Γ_L(x₀)`. On an axis point this is
  the vertex as the limit of the closed form; such approaches exist under the nondegeneracy
  condition `LAxisGeneric` at `x₀`, e.g. along the line `lineL`
  (`LAxisGeneric.tendsto_gammaL_lineL`, `FieldLAxisGeneric.lean`); there `Γ_L(x₀)` is the note's
  explicit axis formula (`LAxisGeneric.gammaOpLAll_eq`, `FieldLAxisFormula.lean`).
-/

open Complex Filter Topology

namespace HubbardAtom

namespace FParams

section Family

variable {P : Type*} [TopologicalSpace P] {U β μ h : P → ℝ} {hβ : ∀ x, 0 < β x} {k : ℤ}
  {x₀ : P} (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)
  (hh : ContinuousAt h x₀)

/-! ### The lattices of the two spins along the family -/

omit [TopologicalSpace P] in
/-- The lattice of spin `σ` along the family is the zero-field family at `μ_σ`. -/
theorem BL_famF (σ : Bool) (x : P) :
    (famF U β μ h hβ x).BL σ = Params.fam U β (fun y ↦ (famF U β μ h hβ y).cL σ) hβ x :=
  rfl

include hm hh in
/-- `μ_σ = μ ± h` is continuous. -/
theorem continuousAt_famF_cL (σ : Bool) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).cL σ) x₀ := by
  cases σ <;> simp only [famF, cL, Bool.cond_true, Bool.cond_false] <;> fun_prop

include hb hm hh in
/-- The lattice points `z_σₙ` are continuous. -/
theorem continuousAt_famF_zL (σ : Bool) (n : ℤ) :
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).BL σ).z k n) x₀ :=
  Params.continuousAt_z (U := U) (hβ := hβ) k hb (continuousAt_famF_cL hm hh σ) n

include hb hm hh in
/-- The moved lattice points `z'_σₙ` are continuous. -/
theorem continuousAt_famF_zOpL (σ : Bool) (n : ℤ) :
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).BL σ).zOp k n) x₀ :=
  Params.continuousAt_fam_zOp (U := U) (hβ := hβ) k hb (continuousAt_famF_cL hm hh σ) n

/-! ### The kernels along the family -/

include hU hb hm hh

/-- `κ_P = e^{-βU/4}β sinhc(βμ)/Z` is continuous. -/
theorem continuousAt_famF_κPL : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).κPL : ℂ)) x₀ := by
  have hs := Params.continuous_dslope_sinh
  have hr : ContinuousAt (fun x ↦ (famF U β μ h hβ x).κPL) x₀ := by
    simp only [κPL, famF, Z, boltz₀, boltzUp, boltzDn, boltz₂]
    fun_prop (disch := positivity)
  exact continuous_ofReal.continuousAt.comp hr

/-- `κ_S = e^{βU/4}β sinhc(βh)/Z` is continuous. -/
theorem continuousAt_famF_κSL : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).κSL : ℂ)) x₀ := by
  have hs := Params.continuous_dslope_sinh
  have hr : ContinuousAt (fun x ↦ (famF U β μ h hβ x).κSL) x₀ := by
    simp only [κSL, famF, Z, boltz₀, boltzUp, boltzDn, boltz₂]
    fun_prop (disch := positivity)
  exact continuous_ofReal.continuousAt.comp hr

/-- `σm` is continuous. -/
theorem continuousAt_famF_mL (σ : Bool) :
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).mL σ : ℂ)) x₀ := by
  have hm' := continuousAt_famF_mf hU hb hm hh (hβ := hβ)
  cases σ
  · simp only [mL_false, ofReal_neg]
    exact hm'.neg
  · simp only [mL_true]
    exact hm'

/-- **The pair kernel is continuous**, for fixed indices: on a mirror pair it is `κ_P`, elsewhere
`ρ/(z + v)` with `z + v ≠ 0`. -/
theorem continuousAt_famF_kerPL (n m : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).kerPL k n m) x₀ := by
  by_cases hmir : m = mirror k n
  · subst hmir
    simp only [kerPL_mirror]
    exact continuousAt_famF_κPL hU hb hm hh
  · have e : ∀ x, (famF U β μ h hβ x).kerPL k n m = (famF U β μ h hβ x).ρ /
        (((famF U β μ h hβ x).BL true).z k n + ((famF U β μ h hβ x).BL (!true)).z k m) :=
      fun x ↦ (famF U β μ h hβ x).kerPL_of_ne_mirror k true hmir
    simp only [e]
    exact (continuousAt_famF_ρ hU hb hm hh).div
      ((continuousAt_famF_zL (U := U) hb hm hh true n).add
        (continuousAt_famF_zL (U := U) hb hm hh (!true) m))
      ((famF U β μ h hβ x₀).zL_add_opp_ne_zero_of_ne k true hmir)

/-- **The spin-flip kernel is continuous**, for fixed indices: at `n = m` it is `κ_S`, elsewhere
`σm/(z - v)` with `z - v ≠ 0`. -/
theorem continuousAt_famF_kerSLs (σ : Bool) (n m : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).kerSLs k σ n m) x₀ := by
  by_cases hnm : n = m
  · subst hnm
    simp only [kerSLs_self]
    exact continuousAt_famF_κSL hU hb hm hh
  · have e : ∀ x, (famF U β μ h hβ x).kerSLs k σ n m = (famF U β μ h hβ x).mL σ /
        (((famF U β μ h hβ x).BL σ).z k n - ((famF U β μ h hβ x).BL (!σ)).z k m) :=
      fun x ↦ (famF U β μ h hβ x).kerSLs_of_ne k σ hnm
    simp only [e]
    exact (continuousAt_famF_mL hU hb hm hh σ).div
      ((continuousAt_famF_zL (U := U) hb hm hh σ n).sub
        (continuousAt_famF_zL (U := U) hb hm hh (!σ) m))
      ((famF U β μ h hβ x₀).zL_sub_opp_ne_zero_of_ne k σ hnm)

/-! ### The entries along the family -/

/-- `r_σ = ρ - σm` is continuous. -/
theorem continuousAt_famF_rL (σ : Bool) :
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).rL σ : ℂ)) x₀ := by
  simp only [rL_cast]
  exact (continuousAt_famF_ρ hU hb hm hh).sub (continuousAt_famF_mL hU hb hm hh σ)

/-- The data of the input are continuous: `a`, `β`, `w`, `ζ`, `ζ_σ`. -/
theorem continuousAt_famF_data :
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).a : ℂ)) x₀ ∧
    ContinuousAt (fun x ↦ ((famF U β μ h hβ x).β : ℂ)) x₀ ∧
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).wL k) x₀ ∧
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).ζL) x₀ ∧
    ∀ σ, ContinuousAt (fun x ↦ (famF U β μ h hβ x).ζLs σ) x₀ := by
  have ha := continuousAt_famF_a hU (β := β) (μ := μ) (h := h) (hβ := hβ)
  have hβc := continuousAt_famF_β hb (U := U) (μ := μ) (h := h) (hβ := hβ)
  have p₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have pu := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have pd := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have p₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  refine ⟨ha, hβc, ?_, ?_, fun σ ↦ ?_⟩
  · have hb0 : β x₀ ≠ 0 := (hβ x₀).ne'
    simp only [wL, ω, famF, Params.ω, lat]
    fun_prop
  · simp only [ζL]
    fun_prop
  · cases σ <;> simp only [ζLs, pL_true, pL_false, Bool.not_true, Bool.not_false] <;> fun_prop

/-- **Each entry of `χ_L` is continuous** along the family, for every `μ`, `h`: by the input,
`χ_στ(n, m)` is a rational function of continuous data with denominators `P(z) ≠ 0`, `P(v) ≠ 0`. -/
theorem continuousAt_famF_chiL (σ τ : Bool) (n m : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).chiL σ τ k n m) x₀ := by
  obtain ⟨ha, hβc, hw, hζ, hζs⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hz := continuousAt_famF_zL (U := U) hb hm hh (hβ := hβ) (k := k) σ n
  have hPz : ContinuousAt (fun x ↦ (famF U β μ h hβ x).PL k
      (((famF U β μ h hβ x).BL σ).z k n)) x₀ := by
    simp only [PL]
    fun_prop
  have hPz0 := (famF U β μ h hβ x₀).PL_BL_ne k σ n
  by_cases hστ : τ = σ
  · subst hστ
    have hv := continuousAt_famF_zL (U := U) hb hm hh (hβ := hβ) (k := k) τ m
    have hPv : ContinuousAt (fun x ↦ (famF U β μ h hβ x).PL k
        (((famF U β μ h hβ x).BL τ).z k m)) x₀ := by
      simp only [PL]
      fun_prop
    have hPv0 := (famF U β μ h hβ x₀).PL_BL_ne k τ m
    have hr := continuousAt_famF_rL hU hb hm hh (hβ := hβ) τ
    have e : ∀ x, (famF U β μ h hβ x).chiL τ τ k n m =
        (-(famF U β μ h hβ x).β * (famF U β μ h hβ x).HL k ((famF U β μ h hβ x).rL τ)
            (((famF U β μ h hβ x).BL τ).z k n) * kron n m +
          kron k 0 * (famF U β μ h hβ x).ζLs τ * ((((famF U β μ h hβ x).BL τ).z k n ^ 2 -
            (famF U β μ h hβ x).a ^ 2) * (((famF U β μ h hβ x).BL τ).z k m ^ 2 -
              (famF U β μ h hβ x).a ^ 2)) /
            (famF U β μ h hβ x).PL k (((famF U β μ h hβ x).BL τ).z k m)) /
        (famF U β μ h hβ x).PL k (((famF U β μ h hβ x).BL τ).z k n) := fun x ↦ by
      rw [eq_div_iff ((famF U β μ h hβ x).PL_BL_ne k τ n), mul_comm,
        (famF U β μ h hβ x).PL_mul_chiL_self k τ n m]
    simp only [e, HL, q₀L]
    have hζτ := hζs τ
    fun_prop (disch := assumption)
  · obtain rfl : τ = !σ := by cases σ <;> cases τ <;> simp_all
    have hv := continuousAt_famF_zL (U := U) hb hm hh (hβ := hβ) (k := k) (!σ) m
    have hPv : ContinuousAt (fun x ↦ (famF U β μ h hβ x).PL k
        (((famF U β μ h hβ x).BL (!σ)).z k m)) x₀ := by
      simp only [PL]
      fun_prop
    have hPv0 := (famF U β μ h hβ x₀).PL_BL_ne k (!σ) m
    have hr := continuousAt_famF_rL hU hb hm hh (hβ := hβ) (!σ)
    have hP := continuousAt_famF_kerPL hU hb hm hh (hβ := hβ) (k := k) n m
    have hS := continuousAt_famF_kerSLs hU hb hm hh (hβ := hβ) (k := k) σ n m
    have e : ∀ x, (famF U β μ h hβ x).chiL σ (!σ) k n m =
        ((famF U β μ h hβ x).lamL * ((famF U β μ h hβ x).kerPL k n m -
            (famF U β μ h hβ x).kerSLs k σ n m) +
          ((famF U β μ h hβ x).N₀L k ((famF U β μ h hβ x).rL (!σ))
              (((famF U β μ h hβ x).BL (!σ)).z k m) +
            ((famF U β μ h hβ x).BL σ).z k n ^ 2 * (famF U β μ h hβ x).N₂L k
              ((famF U β μ h hβ x).rL (!σ)) (((famF U β μ h hβ x).BL (!σ)).z k m)) /
          (famF U β μ h hβ x).PL k (((famF U β μ h hβ x).BL (!σ)).z k m)) /
        (famF U β μ h hβ x).PL k (((famF U β μ h hβ x).BL σ).z k n) := fun x ↦ by
      rw [eq_div_iff ((famF U β μ h hβ x).PL_BL_ne k σ n), mul_comm,
        (famF U β μ h hβ x).PL_mul_chiL_opp_all k σ n m]
    simp only [e, N₀L, N₂L, HL, q₀L, lamL]
    fun_prop (disch := assumption)

/-- **Each entry of the bubble is continuous** along the family. -/
theorem continuousAt_famF_bL (σ : Bool) (n : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).bL σ k n) x₀ := by
  obtain ⟨ha, hβc, hw, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hz := continuousAt_famF_zL (U := U) hb hm hh (hβ := hβ) (k := k) σ n
  have hr := continuousAt_famF_rL hU hb hm hh (hβ := hβ) σ
  have hP0 := (famF U β μ h hβ x₀).PL_BL_ne k σ n
  simp only [bL_eq, PL]
  fun_prop (disch := assumption)

/-! ### Uniform bounds near `x₀` -/

omit hU hb hm hh in
/-- A real function continuous at `x₀` is bounded near `x₀`. -/
private theorem eventually_abs_le_of_continuousAt {f : P → ℝ} (hf : ContinuousAt f x₀) :
    ∀ᶠ x in 𝓝 x₀, |f x| ≤ |f x₀| + 1 :=
  (hf.abs.eventually (gt_mem_nhds (lt_add_one _))).mono fun _ h ↦ h.le

omit hU hb hm hh in
/-- A complex function continuous at `x₀` is bounded near `x₀`. -/
private theorem eventually_norm_le_of_continuousAt {f : P → ℂ} (hf : ContinuousAt f x₀) :
    ∀ᶠ x in 𝓝 x₀, ‖f x‖ ≤ ‖f x₀‖ + 1 :=
  (hf.norm.eventually (gt_mem_nhds (lt_add_one _))).mono fun _ h ↦ h.le

omit hU hb hm hh in
/-- A function of `x` is continuous at `(x₀, s₀)` as a function of `(x, s)`. -/
private theorem continuousAt_fst_of {f : P → ℂ} (s₀ : ℂ) (hf : ContinuousAt f x₀) :
    ContinuousAt (fun p : P × ℂ ↦ f p.1) (x₀, s₀) :=
  ContinuousAt.comp (x := ((x₀, s₀) : P × ℂ)) hf continuousAt_fst

/-- `q₀ = a² - w²/4` is continuous. -/
theorem continuousAt_famF_q₀L : ContinuousAt (fun x ↦ (famF U β μ h hβ x).q₀L k) x₀ := by
  obtain ⟨ha, -, hw, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  simp only [q₀L]
  fun_prop

/-- `1/d(s)` is continuous in `(x, s)` at `s₀ = 0` and at `s₀ = 1/z_σⱼ(x₀)`, as
`d(1/z) = P(z)/z⁴`. -/
theorem continuousAt_famF_inv_dL (σ : Bool) (s₀ : ℂ)
    (hs₀ : s₀ = 0 ∨ ∃ j, mirror k j ≠ j ∧ s₀ = (((famF U β μ h hβ x₀).BL σ).z k j)⁻¹) :
    ContinuousAt (fun p : P × ℂ ↦ ((famF U β μ h hβ p.1).dL k p.2)⁻¹) (x₀, s₀) := by
  obtain ⟨ha, -, -, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have ha' := continuousAt_fst_of s₀ ha
  have hq' := continuousAt_fst_of s₀ (continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k))
  have hd0 : (famF U β μ h hβ x₀).dL k s₀ ≠ 0 := by
    rcases hs₀ with rfl | ⟨j, hj, rfl⟩
    · rw [dL_zero]
      exact one_ne_zero
    · have hz := ((famF U β μ h hβ x₀).BL σ).z_ne_zero_of_ne k hj
      have hP := (famF U β μ h hβ x₀).PL_BL_ne k σ j
      rw [(famF U β μ h hβ x₀).PL_eq_dL k hz] at hP
      exact right_ne_zero_of_mul hP
  refine ContinuousAt.inv₀ ?_ hd0
  simp only [dL]
  fun_prop

omit hU in
/-- **A factor `q(x, 1/z)` is bounded near `x₀`, uniformly off the fixed point**, on the lattice of
either spin. -/
theorem exists_eventually_bdd_zL (σ : Bool) {q : P → ℂ → ℂ}
    (hq : ∀ s₀, (s₀ = 0 ∨ ∃ j, mirror k j ≠ j ∧ s₀ = (((famF U β μ h hβ x₀).BL σ).z k j)⁻¹) →
      ContinuousAt (fun p : P × ℂ ↦ q p.1 p.2) (x₀, s₀)) :
    ∃ C, 0 ≤ C ∧ ∀ᶠ x in 𝓝 x₀, ∀ j, mirror k j ≠ j →
      ‖q x (((famF U β μ h hβ x).BL σ).z k j)⁻¹‖ ≤ C :=
  Params.exists_eventually_bdd_of_nonfixed (U := U) (hβ := hβ) k hb (continuousAt_famF_cL hm hh σ)
    hq

/-- The scalar bounds near `x₀`: `β/π ≤ R`; `π/β`, `|μ|`, `|h|`, `|μ_σ| ≤ M`; `|λ| ≤ Λ`; and
`|ρ|`, `|m|`, `|r_σ|`, `|κ_P|`, `|κ_S| ≤ K`. -/
theorem exists_eventually_scalars : ∃ R M Λ K : ℝ, ∀ᶠ x in 𝓝 x₀,
    (famF U β μ h hβ x).β / Real.pi ≤ R ∧ Real.pi / (famF U β μ h hβ x).β ≤ M ∧
    |(famF U β μ h hβ x).μ| ≤ M ∧ |(famF U β μ h hβ x).h| ≤ M ∧
    (∀ τ, |(famF U β μ h hβ x).cL τ| ≤ M) ∧ ‖(famF U β μ h hβ x).lamL‖ ≤ Λ ∧
    |(famF U β μ h hβ x).ρ| ≤ K ∧ |(famF U β μ h hβ x).mf| ≤ K ∧
    (∀ τ, |(famF U β μ h hβ x).rL τ| ≤ K) ∧ |(famF U β μ h hβ x).κPL| ≤ K ∧
    |(famF U β μ h hβ x).κSL| ≤ K := by
  have hb0 : β x₀ ≠ 0 := (hβ x₀).ne'
  have ha := continuousAt_famF_a hU (β := β) (μ := μ) (h := h) (hβ := hβ)
  have hR := eventually_abs_le_of_continuousAt (hb.div_const Real.pi)
  have hpi : ContinuousAt (fun x ↦ Real.pi / β x) x₀ := continuousAt_const.div hb hb0
  have hM := eventually_abs_le_of_continuousAt hpi
  have hμ := eventually_abs_le_of_continuousAt hm
  have hh' := eventually_abs_le_of_continuousAt hh
  have hΛ := eventually_norm_le_of_continuousAt
    (show ContinuousAt (fun x ↦ (famF U β μ h hβ x).lamL) x₀ by simp only [lamL]; fun_prop)
  have hρ := eventually_norm_le_of_continuousAt (continuousAt_famF_ρ hU hb hm hh (hβ := hβ))
  have hm' := eventually_norm_le_of_continuousAt (continuousAt_famF_mf hU hb hm hh (hβ := hβ))
  have hr₁ := eventually_norm_le_of_continuousAt (continuousAt_famF_rL hU hb hm hh (hβ := hβ) true)
  have hr₂ := eventually_norm_le_of_continuousAt
    (continuousAt_famF_rL hU hb hm hh (hβ := hβ) false)
  have hP := eventually_norm_le_of_continuousAt (continuousAt_famF_κPL hU hb hm hh (hβ := hβ))
  have hS := eventually_norm_le_of_continuousAt (continuousAt_famF_κSL hU hb hm hh (hβ := hβ))
  set F₀ := famF U β μ h hβ x₀
  refine ⟨|β x₀ / Real.pi| + 1, |μ x₀| + |h x₀| + |Real.pi / β x₀| + 3, ‖F₀.lamL‖ + 1,
    ‖(F₀.ρ : ℂ)‖ + 1 + (‖(F₀.mf : ℂ)‖ + 1) + (‖(F₀.rL true : ℂ)‖ + 1) +
      (‖(F₀.rL false : ℂ)‖ + 1) + (‖(F₀.κPL : ℂ)‖ + 1) + (‖(F₀.κSL : ℂ)‖ + 1), ?_⟩
  filter_upwards [hR, hM, hμ, hh', hΛ, hρ, hm', hr₁, hr₂, hP, hS] with x eR eM eμ eh eΛ eρ em er₁
    er₂ eP eS
  simp only [norm_real, Real.norm_eq_abs] at eρ em er₁ er₂ eP eS ⊢
  have n₁ := abs_nonneg (F₀.ρ)
  have n₂ := abs_nonneg (F₀.mf)
  have n₃ := abs_nonneg (F₀.rL true)
  have n₄ := abs_nonneg (F₀.rL false)
  have n₅ := abs_nonneg (F₀.κPL)
  have n₆ := abs_nonneg (F₀.κSL)
  have m₁ := abs_nonneg (μ x₀)
  have m₂ := abs_nonneg (h x₀)
  have m₃ := abs_nonneg (Real.pi / β x₀)
  have hcx : ∀ τ, |(famF U β μ h hβ x).cL τ| ≤ |μ x| + |h x| :=
    fun τ ↦ (famF U β μ h hβ x).abs_cL_le τ
  refine ⟨(le_abs_self _).trans eR, (le_abs_self _).trans (eM.trans (by linarith)), ?_, ?_,
    fun τ ↦ (hcx τ).trans (by linarith), eΛ, by linarith, by linarith, fun τ ↦ ?_, by linarith,
    by linarith⟩
  · change |μ x| ≤ _
    linarith
  · change |h x| ≤ _
    linarith
  · cases τ <;> linarith

/-- `N₀`, `N₂` at the fixed point of the lattice of `-σ` are continuous, hence bounded near `x₀`. -/
theorem exists_eventually_N_fixed (σ : Bool) : ∃ C, 0 ≤ C ∧ ∀ᶠ x in 𝓝 x₀, ∀ j, mirror k j = j →
    ‖(famF U β μ h hβ x).N₀L k ((famF U β μ h hβ x).rL (!σ))
        (((famF U β μ h hβ x).BL (!σ)).z k j)‖ ≤ C ∧
      ‖(famF U β μ h hβ x).N₂L k ((famF U β μ h hβ x).rL (!σ))
        (((famF U β μ h hβ x).BL (!σ)).z k j)‖ ≤ C := by
  obtain ⟨j₀, hj₀⟩ : ∃ j₀ : ℤ, j₀ = (-k - 1) / 2 := ⟨_, rfl⟩
  obtain ⟨ha, -, hw, hζ, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hz := continuousAt_famF_zL (U := U) hb hm hh (hβ := hβ) (k := k) (!σ) j₀
  have hr := continuousAt_famF_rL hU hb hm hh (hβ := hβ) (!σ)
  have h₀ : ContinuousAt (fun x ↦ (famF U β μ h hβ x).N₀L k ((famF U β μ h hβ x).rL (!σ))
      (((famF U β μ h hβ x).BL (!σ)).z k j₀)) x₀ := by
    simp only [N₀L, N₂L, HL, PL, q₀L]
    fun_prop
  have h₂ : ContinuousAt (fun x ↦ (famF U β μ h hβ x).N₂L k ((famF U β μ h hβ x).rL (!σ))
      (((famF U β μ h hβ x).BL (!σ)).z k j₀)) x₀ := by
    simp only [N₂L, HL, q₀L]
    fun_prop
  obtain ⟨a₀, ha₀⟩ : ∃ c, c = ‖(famF U β μ h hβ x₀).N₀L k ((famF U β μ h hβ x₀).rL (!σ))
      (((famF U β μ h hβ x₀).BL (!σ)).z k j₀)‖ := ⟨_, rfl⟩
  obtain ⟨a₂, ha₂⟩ : ∃ c, c = ‖(famF U β μ h hβ x₀).N₂L k ((famF U β μ h hβ x₀).rL (!σ))
      (((famF U β μ h hβ x₀).BL (!σ)).z k j₀)‖ := ⟨_, rfl⟩
  have h0 : 0 ≤ a₀ := by rw [ha₀]; exact norm_nonneg _
  refine ⟨max a₀ a₂ + 1, by linarith [le_max_left a₀ a₂], ?_⟩
  filter_upwards [eventually_norm_le_of_continuousAt h₀, eventually_norm_le_of_continuousAt h₂]
    with x e₀ e₂ j hj
  rw [← ha₀] at e₀
  rw [← ha₂] at e₂
  rw [eq_of_mirror_eq_self hj, ← hj₀]
  exact ⟨e₀.trans (by linarith [le_max_left a₀ a₂]), e₂.trans (by linarith [le_max_right a₀ a₂])⟩

/-- **Opposite spins on `z'`, uniformly near `x₀`**, from `norm_chiL_opp_le_all_of`. -/
theorem exists_eventually_chiL_opp_bound (σ : Bool) : ∃ c, ∀ᶠ x in 𝓝 x₀, ∀ n m,
    ‖((famF U β μ h hβ x).BL σ).zOp k n‖ ^ 2 * ‖((famF U β μ h hβ x).BL (!σ)).zOp k m‖ ^ 2 *
      ‖(famF U β μ h hβ x).chiL σ (!σ) k n m‖ ≤ c := by
  obtain ⟨ha, -, -, hζ, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hq₀ := continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k)
  have hr := continuousAt_famF_rL hU hb hm hh (hβ := hβ) (!σ)
  obtain ⟨R, M, Λ, K, hs⟩ := exists_eventually_scalars hU hb hm hh (hβ := hβ)
  obtain ⟨R', M', B, Λ', Cp', hBp⟩ := Params.exists_eventually_params (U := U) (hβ := hβ) (k := k)
    (ε := 1) hU hb (continuousAt_famF_cL (U := U) (β := β) (hβ := hβ) hm hh σ)
  have hB0 : 0 ≤ B := (norm_nonneg _).trans (hBp.self_of_nhds.2.2.2.1 0)
  obtain ⟨C₁, hC₁0, hC₁⟩ := exists_eventually_bdd_zL (U := U) hb hm hh (hβ := hβ) (k := k) (!σ)
    (q := fun x s ↦ ((famF U β μ h hβ x).dL k s)⁻¹ * (-2 * (famF U β μ h hβ x).a *
      (1 + 2 * (famF U β μ h hβ x).a * (famF U β μ h hβ x).rL (!σ) * s +
        (famF U β μ h hβ x).q₀L k * s ^ 2) +
      kron k 0 * (famF U β μ h hβ x).ζL * (1 - (famF U β μ h hβ x).a ^ 2 * s ^ 2)))
    fun s₀ hs₀ ↦ by
      have ha' := continuousAt_fst_of s₀ ha
      have hq' := continuousAt_fst_of s₀ hq₀
      have hr' := continuousAt_fst_of s₀ hr
      have hζ' := continuousAt_fst_of s₀ hζ
      exact (continuousAt_famF_inv_dL hU hb hm hh (!σ) s₀ hs₀).mul (by fun_prop)
  obtain ⟨C₂, hC₂0, hC₂⟩ := exists_eventually_bdd_zL (U := U) hb hm hh (hβ := hβ) (k := k) (!σ)
    (q := fun x s ↦ ((famF U β μ h hβ x).dL k s)⁻¹ *
      ((2 * (famF U β μ h hβ x).q₀L k - 4 * (famF U β μ h hβ x).a ^ 2) * s +
        (famF U β μ h hβ x).q₀L k ^ 2 * s ^ 3))
    fun s₀ hs₀ ↦ by
      have ha' := continuousAt_fst_of s₀ ha
      have hq' := continuousAt_fst_of s₀ hq₀
      exact (continuousAt_famF_inv_dL hU hb hm hh (!σ) s₀ hs₀).mul (by fun_prop)
  obtain ⟨C₃, hC₃0, hC₃⟩ := exists_eventually_bdd_zL (U := U) hb hm hh (hβ := hβ) (k := k) (!σ)
    (q := fun x s ↦ ((famF U β μ h hβ x).dL k s)⁻¹ * ((famF U β μ h hβ x).cN₂L k +
      (famF U β μ h hβ x).cN₁L k ((famF U β μ h hβ x).rL (!σ)) * s +
        (famF U β μ h hβ x).cN₀L k * s ^ 2))
    fun s₀ hs₀ ↦ by
      have ha' := continuousAt_fst_of s₀ ha
      have hq' := continuousAt_fst_of s₀ hq₀
      have hr' := continuousAt_fst_of s₀ hr
      have hζ' := continuousAt_fst_of s₀ hζ
      refine (continuousAt_famF_inv_dL hU hb hm hh (!σ) s₀ hs₀).mul ?_
      simp only [cN₂L, cN₁L, cN₀L, lamL]
      fun_prop
  obtain ⟨CN, hCN0, hN⟩ := exists_eventually_N_fixed hU hb hm hh (hβ := hβ) (k := k) σ
  refine ⟨cOppL B R M Λ K C₁ C₂ C₃ CN, ?_⟩
  filter_upwards [hs, hBp, hC₁, hC₂, hC₃, hN] with x ⟨eR, eMπ, eμ, eh, ec, eΛ, eρ, em, er, eP, eS⟩
    eB eC₁ eC₂ eC₃ eN n m
  set F := famF U β μ h hβ x
  exact F.norm_chiL_opp_le_all_of k σ eR eMπ eμ eh ec (fun j _ ↦ eB.2.2.2.1 j) hB0 eΛ eρ em
    (er (!σ)) eP eS
    (fun j hj ↦ by rw [F.G₁L_eq k σ ((F.BL (!σ)).z_ne_zero_of_ne k hj)]; exact eC₁ j hj)
    (fun j hj ↦ by rw [F.G₂L_eq k ((F.BL (!σ)).z_ne_zero_of_ne k hj)]; exact eC₂ j hj)
    (fun j hj ↦ by rw [F.G₃L_eq k σ ((F.BL (!σ)).z_ne_zero_of_ne k hj)]; exact eC₃ j hj)
    (fun j hj ↦ (eN j hj).1) (fun j hj ↦ (eN j hj).2) hC₁0 hC₂0 hC₃0 hCN0 n m

/-- **Equal spins off the diagonal on `z'`, uniformly near `x₀`**. -/
theorem exists_eventually_chiL_self_bound (σ : Bool) : ∃ c, ∀ᶠ x in 𝓝 x₀, ∀ n m, n ≠ m →
    ‖((famF U β μ h hβ x).BL σ).zOp k n‖ ^ 2 * ‖((famF U β μ h hβ x).BL σ).zOp k m‖ ^ 2 *
      ‖(famF U β μ h hβ x).chiL σ σ k n m‖ ≤ c := by
  obtain ⟨ha, -, -, -, hζs⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  obtain ⟨CE, hCE0, hCE⟩ := exists_eventually_bdd_zL (U := U) hb hm hh (hβ := hβ) (k := k) σ
    (q := fun x s ↦ ((famF U β μ h hβ x).dL k s)⁻¹ * (1 - (famF U β μ h hβ x).a ^ 2 * s ^ 2))
    fun s₀ hs₀ ↦ by
      have ha' := continuousAt_fst_of s₀ ha
      exact (continuousAt_famF_inv_dL hU hb hm hh σ s₀ hs₀).mul (by fun_prop)
  refine ⟨(‖(famF U β μ h hβ x₀).ζLs σ‖ + 1) * CE * CE, ?_⟩
  filter_upwards [eventually_norm_le_of_continuousAt (hζs σ), hCE] with x eZ eCE n m hnm
  exact (famF U β μ h hβ x).norm_chiL_self_le_all_of k σ eZ (fun j hj ↦ by
    rw [(famF U β μ h hβ x).EzL_eq k (((famF U β μ h hβ x).BL σ).z_ne_zero_of_ne k hj)]
    exact eCE j hj) hCE0 hnm

/-- **The diagonal on `z'`, uniformly near `x₀`**; at the fixed point by continuity. -/
theorem exists_eventually_chiL_diag_bound (σ : Bool) : ∃ c, ∀ᶠ x in 𝓝 x₀, ∀ n,
    ‖((famF U β μ h hβ x).BL σ).zOp k n‖ ^ 3 * ‖(famF U β μ h hβ x).chiL σ σ k n n +
      (famF U β μ h hβ x).β / ((famF U β μ h hβ x).BL σ).zOp k n ^ 2‖ ≤ c := by
  obtain ⟨ha, hβc, -, -, hζs⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hq₀ := continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k)
  have hr := continuousAt_famF_rL hU hb hm hh (hβ := hβ) σ
  obtain ⟨R, M, Λ, K, hs⟩ := exists_eventually_scalars hU hb hm hh (hβ := hβ)
  obtain ⟨Cd, hCd0, hCd⟩ := exists_eventually_bdd_zL (U := U) hb hm hh (hβ := hβ) (k := k) σ
    (q := fun x s ↦ (famF U β μ h hβ x).dgL k σ s) fun s₀ hs₀ ↦ by
      have hd := continuousAt_famF_inv_dL hU hb hm hh σ s₀ hs₀
      have ha' := continuousAt_fst_of s₀ ha
      have hβ' := continuousAt_fst_of s₀ hβc
      have hq' := continuousAt_fst_of s₀ hq₀
      have hr' := continuousAt_fst_of s₀ hr
      have hζ' := continuousAt_fst_of s₀ (hζs σ)
      simp only [dgL]
      fun_prop
  obtain ⟨j₀, hj₀⟩ : ∃ j₀ : ℤ, j₀ = (-k - 1) / 2 := ⟨_, rfl⟩
  have hf : ContinuousAt (fun x ↦ (famF U β μ h hβ x).chiL σ σ k j₀ j₀ +
      ((famF U β μ h hβ x).β : ℂ) / ((famF U β μ h hβ x).BL σ).zOp k j₀ ^ 2) x₀ :=
    (continuousAt_famF_chiL hU hb hm hh σ σ j₀ j₀).add (hβc.div
      ((continuousAt_famF_zOpL (U := U) hb hm hh σ j₀).pow 2)
      (pow_ne_zero 2 (((famF U β μ h hβ x₀).BL σ).zOp_ne_zero k j₀)))
  obtain ⟨Cf, hCf⟩ : ∃ c, c = ‖(famF U β μ h hβ x₀).chiL σ σ k j₀ j₀ +
      ((famF U β μ h hβ x₀).β : ℂ) / ((famF U β μ h hβ x₀).BL σ).zOp k j₀ ^ 2‖ := ⟨_, rfl⟩
  have hCf0 : 0 ≤ Cf + 1 := by rw [hCf]; positivity
  refine ⟨Cd + 8 * M ^ 3 * (Cf + 1), ?_⟩
  filter_upwards [hs, hCd, eventually_norm_le_of_continuousAt hf] with x
    ⟨_, eMπ, _, _, ec, _⟩ eCd ef n
  rw [← hCf] at ef
  exact (famF U β μ h hβ x).norm_chiL_diag_le_all_of k σ eMπ (ec σ) eCd
    (fun j hj ↦ by rw [eq_of_mirror_eq_self hj, ← hj₀]; exact ef) hCd0 hCf0 n

/-- **The bounds of item 6(a) for `χ_L` on `opLatLAll`, uniform near `x₀`**. -/
theorem exists_eventually_chiLe_bounds : ∃ c₁ c₂ : ℝ, ∀ᶠ x in 𝓝 x₀,
    (∀ i j, i ≠ j → ‖(famF U β μ h hβ x).chiLe k i j‖ ≤
      c₁ / (‖((famF U β μ h hβ x).opLatLAll k).z i‖ ^ 2 *
        ‖((famF U β μ h hβ x).opLatLAll k).z j‖ ^ 2)) ∧
    ∀ i, ‖(famF U β μ h hβ x).chiLe k i i + ((famF U β μ h hβ x).opLatLAll k).β /
      ((famF U β μ h hβ x).opLatLAll k).z i ^ 2‖ ≤
        c₂ / ‖((famF U β μ h hβ x).opLatLAll k).z i‖ ^ 3 := by
  obtain ⟨c₁, hc₁⟩ := exists_eventually_chiL_self_bound hU hb hm hh (hβ := hβ) (k := k) true
  obtain ⟨c₂, hc₂⟩ := exists_eventually_chiL_self_bound hU hb hm hh (hβ := hβ) (k := k) false
  obtain ⟨c₃, hc₃⟩ := exists_eventually_chiL_opp_bound hU hb hm hh (hβ := hβ) (k := k) true
  obtain ⟨c₄, hc₄⟩ := exists_eventually_chiL_opp_bound hU hb hm hh (hβ := hβ) (k := k) false
  obtain ⟨d₁, hd₁⟩ := exists_eventually_chiL_diag_bound hU hb hm hh (hβ := hβ) (k := k) true
  obtain ⟨d₂, hd₂⟩ := exists_eventually_chiL_diag_bound hU hb hm hh (hβ := hβ) (k := k) false
  refine ⟨max (max c₁ c₂) (max c₃ c₄), max d₁ d₂, ?_⟩
  filter_upwards [hc₁, hc₂, hc₃, hc₄, hd₁, hd₂] with x e₁ e₂ e₃ e₄ f₁ f₂
  set F := famF U β μ h hβ x
  refine ⟨fun i j hij ↦ ?_, fun i ↦ ?_⟩
  · obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
    obtain ⟨⟨τ, m⟩, rfl⟩ := encL.surjective j
    rw [chiLe_encL, opLatLAll_z, opLatLAll_z]
    dsimp only
    have hz := (F.BL σ).zOp_ne_zero k n
    have hv := (F.BL τ).zOp_ne_zero k m
    rw [le_div_iff₀ (by positivity), mul_comm]
    by_cases hστ : τ = σ
    · subst hστ
      have hnm : n ≠ m := fun h ↦ hij (by rw [h])
      cases τ
      · exact (e₂ n m hnm).trans ((le_max_right _ _).trans (le_max_left _ _))
      · exact (e₁ n m hnm).trans ((le_max_left _ _).trans (le_max_left _ _))
    · obtain rfl : τ = !σ := by cases σ <;> cases τ <;> simp_all
      cases σ
      · exact (e₄ n m).trans ((le_max_right _ _).trans (le_max_right _ _))
      · exact (e₃ n m).trans ((le_max_left _ _).trans (le_max_right _ _))
  · obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
    rw [chiLe_encL, opLatLAll_z, opLatLAll_β]
    dsimp only
    have hz := (F.BL σ).zOp_ne_zero k n
    rw [le_div_iff₀ (by positivity), mul_comm]
    cases σ
    · exact (f₂ n).trans (le_max_right _ _)
    · exact (f₁ n).trans (le_max_left _ _)

omit hU in
/-- `|z'|² ≥ |z'(x₀)|²/2` near `x₀`, on both lattices. -/
theorem eventually_norm_zLeOp_sq_ge : ∀ᶠ x in 𝓝 x₀, ∀ j,
    1 / 2 * ‖((famF U β μ h hβ x₀).opLatLAll k).z j‖ ^ 2 ≤
      ‖((famF U β μ h hβ x).opLatLAll k).z j‖ ^ 2 := by
  have h₁ := Params.eventually_norm_zOp_sq_ge (U := U) (hβ := hβ) (k := k) hb
    (continuousAt_famF_cL (U := U) (β := β) (hβ := hβ) hm hh true)
  have h₂ := Params.eventually_norm_zOp_sq_ge (U := U) (hβ := hβ) (k := k) hb
    (continuousAt_famF_cL (U := U) (β := β) (hβ := hβ) hm hh false)
  filter_upwards [h₁, h₂] with x e₁ e₂ j
  obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective j
  rw [opLatLAll_z, opLatLAll_z]
  cases σ
  · exact e₂ n
  · exact e₁ n

/-- **The hypotheses of `ChiContinuity.lean`** for `χ_L` on `opLatLAll`. -/
theorem contHyp_LAll :
    OpLattice.ContHyp (fun x ↦ (famF U β μ h hβ x).opLatLAll k)
      (fun x ↦ (famF U β μ h hβ x).chiLe k) (fun x ↦ (famF U β μ h hβ x).bLe k) x₀ := by
  have hβ₀ := hβ x₀
  have hβc := (continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)).2.1
  obtain ⟨c₁, c₂, hc⟩ := exists_eventually_chiLe_bounds hU hb hm hh (hβ := hβ) (k := k)
  have hz : ∀ j, ContinuousAt (fun x ↦ ((famF U β μ h hβ x).opLatLAll k).z j) x₀ := fun j ↦
    continuousAt_famF_zOpL (U := U) hb hm hh _ _
  refine ⟨Filter.Eventually.of_forall fun x ↦ (famF U β μ h hβ x).chiDecay_chiLe_all k,
    Filter.Eventually.of_forall fun x ↦ (famF U β μ h hβ x).bubbleDecay_bLe_all k, hb, hz,
    fun j ↦ continuousAt_famF_bL hU hb hm hh _ _, fun i j ↦ ?_,
    OpLattice.dom_of_uniform (c := 1 / 2) (c₁ := c₁) (c₂ := c₂) (by norm_num)
      (hc.mono fun x hx ↦ hx.1) (hc.mono fun x hx ↦ hx.2) hb
      (eventually_norm_zLeOp_sq_ge (U := U) hb hm hh (hβ := hβ) (k := k))⟩
  have hχ := continuousAt_famF_chiL hU hb hm hh (hβ := hβ) (k := k) (encL.symm i).1
    (encL.symm j).1 (encL.symm i).2 (encL.symm j).2
  have hzi := hz i
  have hzj := hz j
  have hK : ContinuousAt (fun x ↦ -(((famF U β μ h hβ x).opLatLAll k).z i *
      ((famF U β μ h hβ x).opLatLAll k).z j / ((famF U β μ h hβ x).β : ℂ)) *
        (famF U β μ h hβ x).chiL (encL.symm i).1 (encL.symm j).1 k (encL.symm i).2
          (encL.symm j).2 - kron i j) x₀ := by
    fun_prop (disch := exact ofReal_ne_zero.mpr hβ₀.ne')
  exact hK

/-- **`Γ_L` is continuous in the parameters, for every `μ`, `h`**: along a family
`(U, β, μ, h)(x)` continuous at `x₀`, if `1 + 𝒦` is invertible at `x₀`, then `Γ_L,(σn)(τm)` is
continuous at `x₀`, on the axes included. -/
theorem continuousAt_gammaOpLAll (hu : IsUnit ((famF U β μ h hβ x₀).onePlusKLAll k)) (σ : Bool)
    (n : ℤ) (τ : Bool) (m : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).gammaOpLAll k σ n τ m) x₀ := by
  have hH := contHyp_LAll hU hb hm hh (hβ := hβ) (k := k)
  have h₁ := OpLattice.continuousAt_gammaOpFam hH ((famF U β μ h hβ x₀).chiDecay_chiLe_all k) hu
    (encL (σ, n)) (encL (τ, m))
  refine h₁.congr (Filter.Eventually.of_forall fun x ↦ ?_)
  dsimp only
  rw [OpLattice.gammaOpFam_eq (fun x ↦ (famF U β μ h hβ x).opLatLAll k)
    (fun x ↦ (famF U β μ h hβ x).chiLe k) (fun x ↦ (famF U β μ h hβ x).bLe k)
    ((famF U β μ h hβ x).chiDecay_chiLe_all k) ((famF U β μ h hβ x).bubbleDecay_bLe_all k)]
  rfl

/-- **The closed form tends to the operator vertex**: if `1 + 𝒦` is invertible at `x₀`, then along
any approach `x i → x₀` through points where Eq. (field-L) holds, `gammaL → Γ_L(x₀)`. On an axis
point, this is the vertex as the limit of the closed form. Without such approaches the statement
is vacuous; that they exist, i.e. that the domain of Eq. (field-L) accumulates at `x₀`, is shown
under the nondegeneracy condition `LAxisGeneric` at `x₀`: near `x₀` the points off the axes are in
the domain (`LAxisGeneric.exists_eventually_LGeneric`), and the line `lineL` reaches `x₀` through
them (`LAxisGeneric.tendsto_gammaL_lineL`, `FieldLAxisGeneric.lean`). -/
theorem tendsto_gammaL_of_LGeneric (hu : IsUnit ((famF U β μ h hβ x₀).onePlusKLAll k))
    {ι : Type*} {l : Filter ι} {x : ι → P} {u : ι → Fin 4 → ℂ} (hx : Tendsto x l (𝓝 x₀))
    (hg : ∀ᶠ i in l, ∀ t, (famF U β μ h hβ (x i)).LGeneric k t (u i)) (σ : Bool) (n : ℤ)
    (τ : Bool) (m : ℤ) :
    Tendsto (fun i ↦ (famF U β μ h hβ (x i)).gammaL k (u i) σ n τ m) l
      (𝓝 ((famF U β μ h hβ x₀).gammaOpLAll k σ n τ m)) :=
  ((continuousAt_gammaOpLAll hU hb hm hh hu σ n τ m).tendsto.comp hx).congr'
    (hg.mono fun _ hi ↦ LGeneric.gammaOpLAll_eq hi σ n τ m)


/-- **On the domain of Eq. (field-L)** at `x₀`, `Γ_L` is continuous at `x₀`. -/
theorem continuousAt_gammaOpLAll_of_LGeneric {u : Fin 4 → ℂ}
    (hg : ∀ t, (famF U β μ h hβ x₀).LGeneric k t u) (σ : Bool) (n : ℤ) (τ : Bool) (m : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).gammaOpLAll k σ n τ m) x₀ :=
  continuousAt_gammaOpLAll hU hb hm hh (LGeneric.isUnit_onePlusKLAll hg) σ n τ m

end Family

end FParams

end HubbardAtom
