import HubbardAtomLean.ChiContinuity
import HubbardAtomLean.FieldTOperator

/-!
# Continuity of the transverse vertex in the parameters

The second half of item 6(d) for the transverse sector: along a family of parameters
`x ↦ (U, β, μ, h)(x)` continuous at `x₀`, `Γ_T↑` is continuous at `x₀` wherever `μ ≠ 0`,
`W = iω_k - 2h ≠ 0` and `1 + 𝒦` is invertible there, in particular on the domain of Eq. (field-T).

The proof is `ChiContinuity.lean`. The bounds of `FieldTBounds.lean` hold with constants that are
uniform near `x₀`: their factors are rational functions of `t = 1/zₙ` jointly continuous in
`(x, t)` (`OpLattice.eventually_bdd_of_inv_z`), and `1/|μ|` stays bounded.

* `FParams.famF`, `FParams.latTFam`: the parameters and the transverse lattice along the family;
* `FParams.gammaOpTFam`: `Γ_T↑,nm` along the family (`gammaOpT` where `μ ≠ 0`, `W ≠ 0`);
* `FParams.continuousAt_gammaOpTFam`: **`Γ_T↑,nm` is continuous at `x₀`** where `1 + 𝒦` is
  invertible; `continuousAt_gammaOpTFam_of_TGeneric`: on the domain of Eq. (field-T).
-/

open Complex Filter Topology

namespace HubbardAtom

namespace FParams

section Family

variable {P : Type*} [TopologicalSpace P] (U β μ h : P → ℝ) (hβ : ∀ x, 0 < β x) (k : ℤ)

/-- The parameters `(U(x), β(x), μ(x), h(x))`. -/
def famF (x : P) : FParams := ⟨U x, β x, μ x, h x, hβ x⟩

/-- A fixed lattice, used along a family where `μ = 0`. -/
noncomputable def latDefault : OpLattice := (⟨0, 1, 1, one_pos⟩ : Params).opLattice 0 one_ne_zero

open Classical in
/-- The transverse lattice along the family, `latDefault` where `μ = 0`. -/
noncomputable def latTFam (x : P) : OpLattice :=
  if hμ : (famF U β μ h hβ x).μ ≠ 0 then (famF U β μ h hβ x).opLatT k hμ else latDefault

open Classical in
/-- `Γ_T↑,nm` along the family: `gammaOpT` where `μ ≠ 0` and `W ≠ 0`, and `0` elsewhere. -/
noncomputable def gammaOpTFam (x : P) (n m : ℤ) : ℂ :=
  if hx : (famF U β μ h hβ x).μ ≠ 0 ∧ (famF U β μ h hβ x).WT k ≠ 0 then
    (famF U β μ h hβ x).gammaOpT k hx.1 hx.2 n m
  else 0

variable {U β μ h hβ k}

omit [TopologicalSpace P] in
theorem latTFam_eq {x : P} (hμ : (famF U β μ h hβ x).μ ≠ 0) :
    latTFam U β μ h hβ k x = (famF U β μ h hβ x).opLatT k hμ := by
  classical
  exact dite_eq_left hμ

variable {x₀ : P} (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)
  (hh : ContinuousAt h x₀)

/-! ### The parameters along the family -/

section Params
include hU

/-- `a = U/2` is continuous. -/
theorem continuousAt_famF_a : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).a : ℂ)) x₀ := by
  simp only [famF, a]
  fun_prop

include hb hm hh

/-- The weights `p₀`, `p↑`, `p↓`, `p₂` are continuous. -/
theorem continuousAt_famF_p (sel : FParams → ℝ)
    (hsel : sel ∈ ({FParams.p₀, FParams.pUp, FParams.pDn, FParams.p₂} : Set (FParams → ℝ))) :
    ContinuousAt (fun x ↦ ((sel (famF U β μ h hβ x) : ℝ) : ℂ)) x₀ := by
  rcases hsel with rfl | rfl | rfl | rfl <;>
  · simp only [famF, p₀, pUp, pDn, p₂, Z, boltz₀, boltzUp, boltzDn, boltz₂]
    fun_prop (disch := positivity)

/-- `ρ = p₂ - p₀` is continuous. -/
theorem continuousAt_famF_ρ : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).ρ : ℂ)) x₀ := by
  have h₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have h₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  simp only [ρ]
  push_cast
  exact h₂.sub h₀

/-- `m = p↑ - p↓` is continuous. -/
theorem continuousAt_famF_mf : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).mf : ℂ)) x₀ := by
  have h₁ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have h₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  simp only [mf]
  push_cast
  exact h₁.sub h₂

omit hU hm hh in
/-- `β` is continuous. -/
theorem continuousAt_famF_β : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).β : ℂ)) x₀ := by
  simp only [famF]
  fun_prop

omit hU hm in
/-- `W = iω_k - 2h` is continuous. -/
theorem continuousAt_famF_WT : ContinuousAt (fun x ↦ (famF U β μ h hβ x).WT k) x₀ := by
  have hb0 : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [famF, WT, ω, Params.ω, lat]
  fun_prop (disch := assumption)

omit hm in
/-- `q₀ = a² - W²/4` is continuous. -/
theorem continuousAt_famF_q₀T : ContinuousAt (fun x ↦ (famF U β μ h hβ x).q₀T k) x₀ := by
  have ha := continuousAt_famF_a hU (h := h) (hβ := hβ) (β := β) (μ := μ) (x₀ := x₀)
  have hW := continuousAt_famF_WT (k := k) hb hh (U := U) (μ := μ) (hβ := hβ)
  simp only [q₀T]
  fun_prop

/-- `q₁` is continuous. -/
theorem continuousAt_famF_q₁T : ContinuousAt (fun x ↦ (famF U β μ h hβ x).q₁T k) x₀ := by
  have ha := continuousAt_famF_a hU (h := h) (hβ := hβ) (β := β) (μ := μ) (x₀ := x₀)
  have hW := continuousAt_famF_WT (k := k) hb hh (U := U) (μ := μ) (hβ := hβ)
  have hq := continuousAt_famF_q₀T (k := k) hU hb hh (μ := μ) (hβ := hβ)
  have h₁ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have h₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have hmf := continuousAt_famF_mf hU hb hm hh (hβ := hβ)
  simp only [q₁T]
  fun_prop

/-- `ζ = -4a²m/W` is continuous where `W ≠ 0`. -/
theorem continuousAt_famF_ζT (hW₀ : (famF U β μ h hβ x₀).WT k ≠ 0) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).ζT k) x₀ := by
  have ha := continuousAt_famF_a hU (h := h) (hβ := hβ) (β := β) (μ := μ) (x₀ := x₀)
  have hW := continuousAt_famF_WT (k := k) hb hh (U := U) (μ := μ) (hβ := hβ)
  have hmf := continuousAt_famF_mf hU hb hm hh (hβ := hβ)
  simp only [ζT]
  fun_prop (disch := assumption)

omit hU hh in
/-- `zₙ = μ + i(νₙ + ω_k/2)` is continuous. -/
theorem continuousAt_famF_zT (n : ℤ) : ContinuousAt (fun x ↦ (famF U β μ h hβ x).zT k n) x₀ := by
  have hb0 : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [zT, Params.z_eq, famF, lat]
  fun_prop (disch := assumption)

/-- `P(zₙ)` is continuous. -/
theorem continuousAt_famF_PT (n : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).PT k ((famF U β μ h hβ x).zT k n)) x₀ := by
  have ha := continuousAt_famF_a hU (h := h) (hβ := hβ) (β := β) (μ := μ) (x₀ := x₀)
  have hW := continuousAt_famF_WT (k := k) hb hh (U := U) (μ := μ) (hβ := hβ)
  have hz := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
  simp only [PT, QpT, QmT]
  fun_prop

/-- The bubble `b_T↑(νₙ) = -βG₂(zₙ)/P(zₙ)` is continuous. -/
theorem continuousAt_famF_bTUp (n : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).bTUp k n) x₀ := by
  have ha := continuousAt_famF_a hU (h := h) (hβ := hβ) (β := β) (μ := μ) (x₀ := x₀)
  have hW := continuousAt_famF_WT (k := k) hb hh (U := U) (μ := μ) (hβ := hβ)
  have hz := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
  have hP := continuousAt_famF_PT (k := k) hU hb hm hh (hβ := hβ) n
  have hρ := continuousAt_famF_ρ hU hb hm hh (hβ := hβ)
  have hmf := continuousAt_famF_mf hU hb hm hh (hβ := hβ)
  have hβc := continuousAt_famF_β hb (U := U) (μ := μ) (h := h) (hβ := hβ)
  have hP0 := (famF U β μ h hβ x₀).PT_zT_ne_zero k n
  simp only [fun x ↦ (famF U β μ h hβ x).bTUp_eq k n, G₂T]
  fun_prop (disch := assumption)

end Params

/-! ### The factors of the bounds, at `t = 1/zₙ` and jointly in `(x, t)` -/

section Factors

variable (hμ₀ : (famF U β μ h hβ x₀).μ ≠ 0) (hW₀ : (famF U β μ h hβ x₀).WT k ≠ 0)
include hU hb hm hh hμ₀ hW₀

/-- The factors of the bounds at `t = 1/zₙ` are continuous at `x₀`, and jointly continuous in
`(x, t)` at `(x₀, 0)`. -/
theorem continuousAt_factors :
    (∀ n, ContinuousAt (fun x ↦ ((famF U β μ h hβ x).dT k ((famF U β μ h hβ x).zT k n)⁻¹)⁻¹) x₀) ∧
    (∀ n, ContinuousAt (fun x ↦ (famF U β μ h hβ x).n2T k ((famF U β μ h hβ x).zT k n)⁻¹) x₀) ∧
    (∀ n, ContinuousAt (fun x ↦ (famF U β μ h hβ x).n0T k ((famF U β μ h hβ x).zT k n)⁻¹) x₀) ∧
    (∀ n, ContinuousAt (fun x ↦ (famF U β μ h hβ x).r3T k ((famF U β μ h hβ x).zT k n)⁻¹) x₀) ∧
    (∀ n, ContinuousAt (fun x ↦ (famF U β μ h hβ x).dgT k ((famF U β μ h hβ x).zT k n)⁻¹) x₀) ∧
    ContinuousAt (fun p : P × ℂ ↦ ((famF U β μ h hβ p.1).dT k p.2)⁻¹) (x₀, 0) ∧
    ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).n2T k p.2) (x₀, 0) ∧
    ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).n0T k p.2) (x₀, 0) ∧
    ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).r3T k p.2) (x₀, 0) ∧
    ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).dgT k p.2) (x₀, 0) := by
  have ha := continuousAt_famF_a hU (h := h) (hβ := hβ) (β := β) (μ := μ) (x₀ := x₀)
  have hρ := continuousAt_famF_ρ hU hb hm hh (hβ := hβ)
  have hq₀ := continuousAt_famF_q₀T (k := k) hU hb hh (μ := μ) (hβ := hβ)
  have hq₁ := continuousAt_famF_q₁T (k := k) hU hb hm hh (hβ := hβ)
  have hζ := continuousAt_famF_ζT hU hb hm hh hW₀
  have hβc := continuousAt_famF_β hb (U := U) (μ := μ) (h := h) (hβ := hβ)
  -- at `t = 1/zₙ`
  have hd : ∀ n, ContinuousAt (fun x ↦ (famF U β μ h hβ x).dT k ((famF U β μ h hβ x).zT k n)⁻¹)
      x₀ := fun n ↦ by
    have hz := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
    have hz0 := (famF U β μ h hβ x₀).zT_ne_zero k hμ₀ n
    simp only [dT]
    fun_prop (disch := assumption)
  have hd0 : ∀ n, (famF U β μ h hβ x₀).dT k ((famF U β μ h hβ x₀).zT k n)⁻¹ ≠ 0 := fun n ↦ by
    have hz0 := (famF U β μ h hβ x₀).zT_ne_zero k hμ₀ n
    rw [(famF U β μ h hβ x₀).dT_inv k hz0]
    exact div_ne_zero ((famF U β μ h hβ x₀).PT_zT_ne_zero k n) (pow_ne_zero _ hz0)
  have hn2 : ∀ n, ContinuousAt (fun x ↦ (famF U β μ h hβ x).n2T k ((famF U β μ h hβ x).zT k n)⁻¹)
      x₀ := fun n ↦ by
    have hz := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
    have hz0 := (famF U β μ h hβ x₀).zT_ne_zero k hμ₀ n
    simp only [n2T]
    fun_prop (disch := assumption)
  have hn0 : ∀ n, ContinuousAt (fun x ↦ (famF U β μ h hβ x).n0T k ((famF U β μ h hβ x).zT k n)⁻¹)
      x₀ := fun n ↦ by
    have hz := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
    have hz0 := (famF U β μ h hβ x₀).zT_ne_zero k hμ₀ n
    simp only [n0T, lamT, cT₀, cT₁, cT₂]
    fun_prop (disch := assumption)
  have hr3 : ∀ n, ContinuousAt (fun x ↦ (famF U β μ h hβ x).r3T k ((famF U β μ h hβ x).zT k n)⁻¹)
      x₀ := fun n ↦ by
    have hz := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
    have hz0 := (famF U β μ h hβ x₀).zT_ne_zero k hμ₀ n
    simp only [r3T, lamT, cT₀, cT₁, cT₂]
    fun_prop (disch := assumption)
  -- jointly at `(x₀, 0)`
  have ha' : ContinuousAt (fun p : P × ℂ ↦ ((famF U β μ h hβ p.1).a : ℂ)) (x₀, 0) :=
    ContinuousAt.comp (x := ((x₀, 0) : P × ℂ)) ha continuousAt_fst
  have hρ' : ContinuousAt (fun p : P × ℂ ↦ ((famF U β μ h hβ p.1).ρ : ℂ)) (x₀, 0) :=
    ContinuousAt.comp (x := ((x₀, 0) : P × ℂ)) hρ continuousAt_fst
  have hq₀' : ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).q₀T k) (x₀, 0) :=
    ContinuousAt.comp (x := ((x₀, 0) : P × ℂ)) (g := fun x ↦ (famF U β μ h hβ x).q₀T k) hq₀
      continuousAt_fst
  have hq₁' : ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).q₁T k) (x₀, 0) :=
    ContinuousAt.comp (x := ((x₀, 0) : P × ℂ)) (g := fun x ↦ (famF U β μ h hβ x).q₁T k) hq₁
      continuousAt_fst
  have hζ' : ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).ζT k) (x₀, 0) :=
    ContinuousAt.comp (x := ((x₀, 0) : P × ℂ)) (g := fun x ↦ (famF U β μ h hβ x).ζT k) hζ
      continuousAt_fst
  have hβ' : ContinuousAt (fun p : P × ℂ ↦ ((famF U β μ h hβ p.1).β : ℂ)) (x₀, 0) :=
    ContinuousAt.comp (x := ((x₀, 0) : P × ℂ)) hβc continuousAt_fst
  have hdJ : ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).dT k p.2) (x₀, 0) := by
    simp only [dT]
    fun_prop
  have hdJ0 : (famF U β μ h hβ x₀).dT k 0 ≠ 0 := by simp [dT]
  have hn2J : ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).n2T k p.2) (x₀, 0) := by
    simp only [n2T]
    fun_prop
  have hn0J : ContinuousAt (fun p : P × ℂ ↦ (famF U β μ h hβ p.1).n0T k p.2) (x₀, 0) := by
    simp only [n0T, lamT, cT₀, cT₁, cT₂]
    fun_prop
  refine ⟨fun n ↦ (hd n).inv₀ (hd0 n), hn2, hn0, hr3, fun n ↦ ?_, hdJ.inv₀ hdJ0, hn2J, hn0J,
    ?_, ?_⟩
  · have hdn := hd n
    have hd0n := hd0 n
    have hn2n := hn2 n
    have hn0n := hn0 n
    have hz := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
    have hz0 := (famF U β μ h hβ x₀).zT_ne_zero k hμ₀ n
    simp only [dgT, lamT]
    fun_prop (disch := assumption)
  · simp only [r3T, lamT, cT₀, cT₁, cT₂]
    fun_prop
  · simp only [dgT, lamT]
    fun_prop (disch := exact hdJ0)

/-- **`χ_T↑,nm` is continuous** at `x₀`, through the transverse input. -/
theorem continuousAt_famF_chiTUp (n m : ℤ) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).chiTUp k n m) x₀ := by
  have ha := continuousAt_famF_a hU (h := h) (hβ := hβ) (β := β) (μ := μ) (x₀ := x₀)
  have hρ := continuousAt_famF_ρ hU hb hm hh (hβ := hβ)
  have hq₀ := continuousAt_famF_q₀T (k := k) hU hb hh (μ := μ) (hβ := hβ)
  have hq₁ := continuousAt_famF_q₁T (k := k) hU hb hm hh (hβ := hβ)
  have hζ := continuousAt_famF_ζT hU hb hm hh hW₀
  have hβc := continuousAt_famF_β hb (U := U) (μ := μ) (h := h) (hβ := hβ)
  have hzn := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
  have hzm := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) m
  have hPn := continuousAt_famF_PT (k := k) hU hb hm hh (hβ := hβ) n
  have hPm := continuousAt_famF_PT (k := k) hU hb hm hh (hβ := hβ) m
  have hPn0 := (famF U β μ h hβ x₀).PT_zT_ne_zero k n
  have hPm0 := (famF U β μ h hβ x₀).PT_zT_ne_zero k m
  have hzz0 := (famF U β μ h hβ x₀).zT_add_zT_ne_zero k hμ₀ n m
  have hrhs : ContinuousAt (fun x ↦ (-((famF U β μ h hβ x).β : ℂ) *
      (famF U β μ h hβ x).HT k ((famF U β μ h hβ x).zT k n) * kron n m +
      (famF U β μ h hβ x).lamT * (famF U β μ h hβ x).ρ /
        ((famF U β μ h hβ x).zT k n + (famF U β μ h hβ x).zT k m) +
      (famF U β μ h hβ x).NT k ((famF U β μ h hβ x).zT k n) ((famF U β μ h hβ x).zT k m) /
        (famF U β μ h hβ x).PT k ((famF U β μ h hβ x).zT k m)) /
      (famF U β μ h hβ x).PT k ((famF U β μ h hβ x).zT k n)) x₀ := by
    simp only [HT, NT, N₀T, N₂T, lamT]
    fun_prop (disch := assumption)
  refine hrhs.congr ?_
  filter_upwards [hm.eventually_ne hμ₀,
    (continuousAt_famF_WT (k := k) hb hh (U := U) (μ := μ) (hβ := hβ)).eventually_ne hW₀]
    with x hμx hWx
  exact ((eq_div_iff ((famF U β μ h hβ x).PT_zT_ne_zero k n)).mpr
    (by rw [mul_comm]; exact (famF U β μ h hβ x).PT_mul_chiTUp k hμx hWx n m)).symm

/-- **The hypotheses of `ChiContinuity.lean`** for the transverse sector at `x₀`. -/
theorem contHyp_T :
    OpLattice.ContHyp (latTFam U β μ h hβ k) (fun x ↦ (famF U β μ h hβ x).chiTUp k)
      (fun x ↦ (famF U β μ h hβ x).bTUp k) x₀ := by
  have hμev : ∀ᶠ x in 𝓝 x₀, (famF U β μ h hβ x).μ ≠ 0 := hm.eventually_ne hμ₀
  have hWev : ∀ᶠ x in 𝓝 x₀, (famF U β μ h hβ x).WT k ≠ 0 :=
    (continuousAt_famF_WT (k := k) hb hh (U := U) (μ := μ) (hβ := hβ)).eventually_ne hW₀
  have hβc := continuousAt_famF_β hb (U := U) (μ := μ) (h := h) (hβ := hβ)
  obtain ⟨hd, hn2, hn0, hr3, hdg, hdJ, hn2J, hn0J, hr3J, hdgJ⟩ :=
    continuousAt_factors hU hb hm hh hμ₀ hW₀
  have hu := Params.eventually_norm_inv_z_lt
    (fun x ↦ (famF U β μ h hβ x).lat (famF U β μ h hβ x).μ) k hb
  obtain ⟨C₁, hC₁⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (famF U β μ h hβ x).zT k n)
    (q := fun x t ↦ ((famF U β μ h hβ x).dT k t)⁻¹) hdJ hd hu
  obtain ⟨C₂, hC₂⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (famF U β μ h hβ x).zT k n)
    (q := fun x t ↦ (famF U β μ h hβ x).n2T k t) hn2J hn2 hu
  obtain ⟨C₃, hC₃⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (famF U β μ h hβ x).zT k n)
    (q := fun x t ↦ (famF U β μ h hβ x).n0T k t) hn0J hn0 hu
  obtain ⟨C₄, hC₄⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (famF U β μ h hβ x).zT k n)
    (q := fun x t ↦ (famF U β μ h hβ x).r3T k t) hr3J hr3 hu
  obtain ⟨Cd, hCd⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (famF U β μ h hβ x).zT k n)
    (q := fun x t ↦ (famF U β μ h hβ x).dgT k t) hdgJ hdg hu
  have hμ₀' : 0 < |μ x₀| := abs_pos.mpr hμ₀
  have hMev : ∀ᶠ x in 𝓝 x₀, |(famF U β μ h hβ x).μ|⁻¹ ≤ 2 * |μ x₀|⁻¹ := by
    filter_upwards [(continuous_abs.continuousAt.comp hm).eventually
      (lt_mem_nhds (half_lt_self hμ₀'))] with x hx
    have hx' : |μ x₀| / 2 < |μ x| := hx
    calc |(famF U β μ h hβ x).μ|⁻¹ = |μ x|⁻¹ := rfl
      _ ≤ (|μ x₀| / 2)⁻¹ := inv_anti₀ (by positivity) hx'.le
      _ = 2 * |μ x₀|⁻¹ := by rw [inv_div, div_eq_mul_inv]
  refine ⟨?_, ?_, ?_, fun n ↦ ?_, fun n ↦ continuousAt_famF_bTUp hU hb hm hh n,
    fun n m ↦ ?_, ?_⟩
  · filter_upwards [hμev, hWev] with x hμx hWx
    rw [latTFam_eq hμx]
    exact (famF U β μ h hβ x).chiDecay_chiTUp k hμx hWx
  · filter_upwards [hμev] with x hμx
    rw [latTFam_eq hμx]
    exact (famF U β μ h hβ x).bubbleDecay_bTUp k hμx
  · refine hb.congr ?_
    filter_upwards [hμev] with x hμx
    rw [latTFam_eq hμx]
    rfl
  · refine (continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n).congr ?_
    filter_upwards [hμev] with x hμx
    rw [latTFam_eq hμx]
    rfl
  · have hzn := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) n
    have hzm := continuousAt_famF_zT (k := k) hb hm (U := U) (h := h) (hβ := hβ) m
    have hχ := continuousAt_famF_chiTUp hU hb hm hh hμ₀ hW₀ n m
    have hβ0 : ((famF U β μ h hβ x₀).β : ℂ) ≠ 0 := ofReal_ne_zero.mpr (hβ x₀).ne'
    have hK : ContinuousAt (fun x ↦ -((famF U β μ h hβ x).zT k n * (famF U β μ h hβ x).zT k m /
        ((famF U β μ h hβ x).β : ℂ)) * (famF U β μ h hβ x).chiTUp k n m - kron n m) x₀ := by
      fun_prop (disch := assumption)
    refine hK.congr ?_
    filter_upwards [hμev] with x hμx
    rw [latTFam_eq hμx]
    rfl
  · refine OpLattice.dom_of_uniform (c := 1 / 2) (by norm_num) (c₁ := C₁ * C₁ *
      (C₂ + (C₃ + C₄ * (2 * |μ x₀|⁻¹)) * (2 * |μ x₀|⁻¹ + 2 * |μ x₀|⁻¹))) (c₂ := Cd) ?_ ?_ ?_ ?_
    · filter_upwards [hμev, hWev, hMev, hC₁, hC₂, hC₃, hC₄] with x hμx hWx hMx h₁ h₂ h₃ h₄
      rw [latTFam_eq hμx]
      exact (famF U β μ h hβ x).norm_chiTUp_le_of k hμx hWx hMx h₁ h₂ h₃ h₄
    · filter_upwards [hμev, hWev, hCd] with x hμx hWx hdx
      rw [latTFam_eq hμx]
      exact (famF U β μ h hβ x).norm_chiTUp_diag_le_of k hμx hWx hdx
    · refine hb.congr ?_
      filter_upwards [hμev] with x hμx
      rw [latTFam_eq hμx]
      rfl
    · filter_upwards [Params.eventually_norm_z_sq_ge
        (fun x ↦ (famF U β μ h hβ x).lat (famF U β μ h hβ x).μ) k hm hb hμ₀, hμev]
        with x hx hμx n
      rw [latTFam_eq hμx, latTFam_eq hμ₀]
      exact hx n

/-- **`Γ_T↑` is continuous in the parameters**: along a family continuous at `x₀`, with `μ ≠ 0`,
`W ≠ 0` and `1 + 𝒦` invertible at `x₀`, `Γ_T↑,nm` is continuous at `x₀`. -/
theorem continuousAt_gammaOpTFam (hu : IsUnit ((famF U β μ h hβ x₀).onePlusKT k hμ₀ hW₀))
    (n m : ℤ) : ContinuousAt (fun x ↦ gammaOpTFam U β μ h hβ k x n m) x₀ := by
  have hH := contHyp_T hU hb hm hh hμ₀ hW₀
  have hμev : ∀ᶠ x in 𝓝 x₀, (famF U β μ h hβ x).μ ≠ 0 := hm.eventually_ne hμ₀
  have hWev : ∀ᶠ x in 𝓝 x₀, (famF U β μ h hβ x).WT k ≠ 0 :=
    (continuousAt_famF_WT (k := k) hb hh (U := U) (μ := μ) (hβ := hβ)).eventually_ne hW₀
  have hX₀ := hH.chi.self_of_nhds
  have hu' : IsUnit ((latTFam U β μ h hβ k x₀).onePlusKX _ hX₀) := by
    rw [OpLattice.onePlusKX_congr (latTFam_eq hμ₀) hX₀
      ((famF U β μ h hβ x₀).chiDecay_chiTUp k hμ₀ hW₀)]
    exact hu
  refine (OpLattice.continuousAt_gammaOpFam hH hX₀ hu' n m).congr ?_
  filter_upwards [hμev, hWev, hH.chi, hH.bub] with x hμx hWx hXx hbx
  classical
  rw [OpLattice.gammaOpFam_eq _ _ _ hXx hbx, gammaOpTFam, dite_eq_left ⟨hμx, hWx⟩, gammaOpT]
  exact OpLattice.gammaOpX_congr (latTFam_eq hμx) _ _ _ _ n m

omit hμ₀ hW₀ in
/-- **On the domain of Eq. (field-T)**, `Γ_T↑` (there the closed form `gammaTUp`) is continuous
in the parameters. -/
theorem continuousAt_gammaOpTFam_of_TGeneric {u₁ u₂ : ℂ}
    (hg : (famF U β μ h hβ x₀).TGeneric k u₁ u₂) (n m : ℤ) :
    ContinuousAt (fun x ↦ gammaOpTFam U β μ h hβ k x n m) x₀ :=
  continuousAt_gammaOpTFam hU hb hm hh hg.μ_ne hg.W_ne hg.isUnit_onePlusKT n m

end Factors

end Family

end FParams

end HubbardAtom
