import Mathlib

/-!
# Iterated integrals over the cube and its ordered simplices

The imaginary-time integrals of the correlation functions run over the cube `[0, β]³`, and their
integrands change with the ordering of the times.

* `integral_cube`, `integral_square`: an iterated integral over `[0, β]³` (or `[0, β]²`) whose
  integrand agrees with a continuous function on each ordering of the times is a sum of integrals
  over the ordered simplex `β > s₁ > s₂ > s₃ > 0` (`simplexIntegral`), one for each ordering
  (or over the triangle, `triangleIntegral`). The proof splits the inner integrals at the other
  times and reorders by Fubini on rectangles and triangles (`integral_swap_rect`,
  `integral_swap_triangle`, from Mathlib's `intervalIntegral_intervalIntegral_swap`).
* `integral_simplex2`, `integral_simplex3`: `∫ e^{a s₁ + b s₂ (+ c s₃)}` over the ordered simplex,
  in closed form (`simplex2`, `simplex3`), also when partial sums of the rates vanish;
  `expInt r t = ∫₀ᵗ e^{rs} ds`.
* `mul_simplex2`, `mul_simplex3`: with a Boltzmann weight in front, the same integrals through
  the weights reached along the path (`triangleW`, `simplexW`), without exponentials.
-/

open Complex MeasureTheory Set intervalIntegral

namespace HubbardAtom

/-- `∫₀ᵗ e^{rs} ds`: `(e^{rt} - 1)/r`, and `t` for `r = 0`. -/
noncomputable def expInt (r : ℂ) (t : ℝ) : ℂ := if r = 0 then t else (cexp (r * t) - 1) / r

theorem hasDerivAt_cexp_mul (r : ℂ) (x : ℝ) :
    HasDerivAt (fun y : ℝ ↦ cexp (r * y)) (r * cexp (r * x)) x := by
  have h : HasDerivAt (fun y : ℝ ↦ r * (y : ℂ)) r x := by
    simpa using ((hasDerivAt_id (x : ℂ)).const_mul r).comp_ofReal
  simpa [mul_comm] using h.cexp

theorem integral_cexp_mul (r : ℂ) (t : ℝ) : ∫ s in (0 : ℝ)..t, cexp (r * s) = expInt r t := by
  by_cases hr : r = 0
  · simp [expInt, hr]
  · rw [expInt, ite_eq_right hr, integral_exp_mul_complex hr]
    simp

/-- `∫₀ᵗ s e^{as} ds = (t e^{at} - expInt a t)/a` for `a ≠ 0`. -/
theorem integral_mul_cexp_mul {a : ℂ} (ha : a ≠ 0) (t : ℝ) :
    ∫ s in (0 : ℝ)..t, (s : ℂ) * cexp (a * s) = (t * cexp (a * t) - expInt a t) / a := by
  have hD : ∀ x : ℝ, HasDerivAt (fun y : ℝ ↦ ((y : ℂ) / a - 1 / a ^ 2) * cexp (a * y))
      ((x : ℂ) * cexp (a * x)) x := by
    intro x
    have h1 : HasDerivAt (fun y : ℝ ↦ (y : ℂ) / a - 1 / a ^ 2) (1 / a) x := by
      have := ((hasDerivAt_id (x : ℂ)).div_const a).comp_ofReal
      simpa using this.sub_const (1 / a ^ 2)
    convert h1.mul (hasDerivAt_cexp_mul a x) using 1
    field_simp
    ring
  rw [integral_eq_sub_of_hasDerivAt (fun x _ ↦ hD x)
    ((by fun_prop : Continuous fun s : ℝ ↦ (s : ℂ) * cexp (a * s)).intervalIntegrable _ _),
    expInt, ite_eq_right ha]
  simp only [ofReal_zero, mul_zero, Complex.exp_zero]
  field_simp
  ring

/-- The integral of `e^{a s₁ + b s₂}` over the ordered triangle `β > s₁ > s₂ > 0`. -/
noncomputable def simplex2 (a b : ℂ) (β : ℝ) : ℂ := (expInt (a + b) β - expInt a β) / b

theorem integral_simplex2 {a b : ℂ} (hb : b ≠ 0) (β : ℝ) :
    ∫ s₁ in (0 : ℝ)..β, ∫ s₂ in (0 : ℝ)..s₁, cexp (a * s₁ + b * s₂) = simplex2 a b β := by
  have inner : ∀ s₁ : ℝ, ∫ s₂ in (0 : ℝ)..s₁, cexp (a * s₁ + b * s₂) =
      (cexp ((a + b) * s₁) - cexp (a * s₁)) / b := by
    intro s₁
    simp_rw [Complex.exp_add]
    rw [intervalIntegral.integral_const_mul, integral_exp_mul_complex hb]
    simp only [ofReal_zero, mul_zero, Complex.exp_zero, add_mul, Complex.exp_add]
    ring
  simp_rw [inner]
  rw [intervalIntegral.integral_div, intervalIntegral.integral_sub
    ((by fun_prop : Continuous fun s : ℝ ↦ cexp ((a + b) * s)).intervalIntegrable _ _)
    ((by fun_prop : Continuous fun s : ℝ ↦ cexp (a * s)).intervalIntegrable _ _),
    integral_cexp_mul, integral_cexp_mul, simplex2]

/-- The integral of `e^{a s₁ + b s₂ + c s₃}` over the ordered simplex `β > s₁ > s₂ > s₃ > 0`, for
nonzero `a`, `b`, `c`; `a + b` or `b + c` may vanish. -/
noncomputable def simplex3 (a b c : ℂ) (β : ℝ) : ℂ :=
  ((if b + c = 0 then (β * cexp (a * β) - expInt a β) / a
      else (expInt (a + b + c) β - expInt a β) / (b + c)) -
    (expInt (a + b) β - expInt a β) / b) / c

theorem integral_simplex3 {a b c : ℂ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (β : ℝ) :
    ∫ s₁ in (0 : ℝ)..β, ∫ s₂ in (0 : ℝ)..s₁, ∫ s₃ in (0 : ℝ)..s₂,
      cexp (a * s₁ + b * s₂ + c * s₃) = simplex3 a b c β := by
  -- the innermost integral
  have inner : ∀ s₁ s₂ : ℝ, ∫ s₃ in (0 : ℝ)..s₂, cexp (a * s₁ + b * s₂ + c * s₃) =
      (cexp (a * s₁) * cexp ((b + c) * s₂) - cexp (a * s₁) * cexp (b * s₂)) / c := by
    intro s₁ s₂
    simp_rw [Complex.exp_add]
    rw [intervalIntegral.integral_const_mul, integral_exp_mul_complex hc]
    simp only [ofReal_zero, mul_zero, Complex.exp_zero, add_mul, Complex.exp_add]
    ring
  -- the middle integral
  have middle : ∀ s₁ : ℝ, ∫ s₂ in (0 : ℝ)..s₁,
      (cexp (a * s₁) * cexp ((b + c) * s₂) - cexp (a * s₁) * cexp (b * s₂)) / c =
      (cexp (a * s₁) * expInt (b + c) s₁ - cexp (a * s₁) * ((cexp (b * s₁) - 1) / b)) / c := by
    intro s₁
    rw [intervalIntegral.integral_div, intervalIntegral.integral_sub
      ((by fun_prop : Continuous fun s : ℝ ↦ cexp (a * s₁) * cexp ((b + c) * s)).intervalIntegrable
        _ _)
      ((by fun_prop : Continuous fun s : ℝ ↦ cexp (a * s₁) * cexp (b * s)).intervalIntegrable _ _),
      intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul, integral_cexp_mul,
      integral_cexp_mul, expInt, expInt, ite_eq_right hb]
  simp_rw [inner, middle]
  by_cases hbc : b + c = 0
  · -- `b + c = 0`: the middle integral gives `s₁ e^{a s₁}`
    have h : ∀ s₁ : ℝ, (cexp (a * s₁) * expInt (b + c) s₁ -
        cexp (a * s₁) * ((cexp (b * s₁) - 1) / b)) / c =
          ((s₁ : ℂ) * cexp (a * s₁) - (cexp ((a + b) * s₁) - cexp (a * s₁)) / b) / c := by
      intro s₁
      rw [expInt, ite_eq_left hbc, add_mul, Complex.exp_add]
      ring
    simp_rw [h]
    rw [intervalIntegral.integral_div, intervalIntegral.integral_sub
      ((by fun_prop : Continuous fun s : ℝ ↦ (s : ℂ) * cexp (a * s)).intervalIntegrable _ _)
      ((by fun_prop : Continuous fun s : ℝ ↦
        (cexp ((a + b) * s) - cexp (a * s)) / b).intervalIntegrable _ _),
      integral_mul_cexp_mul ha, intervalIntegral.integral_div, intervalIntegral.integral_sub
      ((by fun_prop : Continuous fun s : ℝ ↦ cexp ((a + b) * s)).intervalIntegrable _ _)
      ((by fun_prop : Continuous fun s : ℝ ↦ cexp (a * s)).intervalIntegrable _ _),
      integral_cexp_mul, integral_cexp_mul, simplex3, ite_eq_left hbc]
  · have h : ∀ s₁ : ℝ, (cexp (a * s₁) * expInt (b + c) s₁ -
        cexp (a * s₁) * ((cexp (b * s₁) - 1) / b)) / c =
          ((cexp ((a + b + c) * s₁) - cexp (a * s₁)) / (b + c) -
            (cexp ((a + b) * s₁) - cexp (a * s₁)) / b) / c := by
      intro s₁
      rw [expInt, ite_eq_right hbc, show (a + b + c) * (s₁ : ℂ) = a * s₁ + (b + c) * s₁ by ring,
        Complex.exp_add, add_mul a b, Complex.exp_add]
      field_simp
    simp_rw [h]
    rw [intervalIntegral.integral_div, intervalIntegral.integral_sub
      ((by fun_prop : Continuous fun s : ℝ ↦
        (cexp ((a + b + c) * s) - cexp (a * s)) / (b + c)).intervalIntegrable _ _)
      ((by fun_prop : Continuous fun s : ℝ ↦
        (cexp ((a + b) * s) - cexp (a * s)) / b).intervalIntegrable _ _),
      intervalIntegral.integral_div, intervalIntegral.integral_sub
      ((by fun_prop : Continuous fun s : ℝ ↦ cexp ((a + b + c) * s)).intervalIntegrable _ _)
      ((by fun_prop : Continuous fun s : ℝ ↦ cexp (a * s)).intervalIntegrable _ _),
      intervalIntegral.integral_div, intervalIntegral.integral_sub
      ((by fun_prop : Continuous fun s : ℝ ↦ cexp ((a + b) * s)).intervalIntegrable _ _)
      ((by fun_prop : Continuous fun s : ℝ ↦ cexp (a * s)).intervalIntegrable _ _),
      integral_cexp_mul, integral_cexp_mul, integral_cexp_mul, simplex3, ite_eq_right hbc]

/-! ### Changing the order of integration -/

/-- Fubini on a rectangle, for a continuous integrand. -/
theorem integral_swap_rect {F : ℝ → ℝ → ℂ} (hF : Continuous (Function.uncurry F)) (a b c d : ℝ) :
    ∫ x in a..b, ∫ y in c..d, F x y = ∫ y in c..d, ∫ x in a..b, F x y :=
  intervalIntegral_intervalIntegral_swap
    ((hF.continuousOn.integrableOn_compact (isCompact_uIcc.prod isCompact_uIcc)).mono_set
      (prod_mono uIoc_subset_uIcc uIoc_subset_uIcc))

/-- Fubini on a triangle: `∫₀ᶜ dx ∫₀ˣ dy F = ∫₀ᶜ dy ∫ᵧᶜ dx F` for continuous `F` and `0 ≤ c`. -/
theorem integral_swap_triangle {F : ℝ → ℝ → ℂ} (hF : Continuous (Function.uncurry F)) {c : ℝ}
    (hc : 0 ≤ c) :
    ∫ x in (0 : ℝ)..c, ∫ y in (0 : ℝ)..x, F x y = ∫ y in (0 : ℝ)..c, ∫ x in y..c, F x y := by
  set G : ℝ → ℝ → ℂ := fun x y ↦ if y ≤ x then F x y else 0 with hG
  have hGint : IntegrableOn (Function.uncurry G) (uIoc 0 c ×ˢ uIoc 0 c) := by
    have h : Function.uncurry G = {p : ℝ × ℝ | p.2 ≤ p.1}.indicator (Function.uncurry F) := by
      funext p
      simp [G, Set.indicator, Function.uncurry]
    rw [h]
    exact ((hF.continuousOn.integrableOn_compact (isCompact_uIcc.prod isCompact_uIcc)).mono_set
      (prod_mono uIoc_subset_uIcc uIoc_subset_uIcc)).indicator
      (measurableSet_le measurable_snd measurable_fst)
  -- the rows: `∫₀ᶜ G(x, y) dy = ∫₀ˣ F(x, y) dy`
  have hrow : ∀ x ∈ uIcc 0 c, ∫ y in (0 : ℝ)..c, G x y = ∫ y in (0 : ℝ)..x, F x y := by
    intro x hx
    rw [uIcc_of_le hc] at hx
    have hGx : G x = (Iic x).indicator (F x) := by
      funext y
      simp [G, Set.indicator]
    rw [hGx, intervalIntegral.integral_of_le hc, intervalIntegral.integral_of_le hx.1,
      setIntegral_indicator measurableSet_Iic,
      show Ioc 0 c ∩ Iic x = Ioc 0 x from by
        ext y
        simp only [mem_inter_iff, mem_Ioc, mem_Iic]
        constructor
        · rintro ⟨⟨h1, _⟩, h3⟩
          exact ⟨h1, h3⟩
        · rintro ⟨h1, h2⟩
          exact ⟨⟨h1, h2.trans hx.2⟩, h2⟩]
  -- the columns: `∫₀ᶜ G(x, y) dx = ∫ᵧᶜ F(x, y) dx`, for `y > 0`
  have hcol : ∀ y ∈ Ioc 0 c, ∫ x in (0 : ℝ)..c, G x y = ∫ x in y..c, F x y := by
    intro y hy
    have hGy : (fun x ↦ G x y) = (Ici y).indicator (fun x ↦ F x y) := by
      funext x
      simp [G, Set.indicator]
    rw [hGy, intervalIntegral.integral_of_le hc, intervalIntegral.integral_of_le hy.2,
      setIntegral_indicator measurableSet_Ici,
      show Ioc 0 c ∩ Ici y = Icc y c from by
        ext x
        simp only [mem_inter_iff, mem_Ioc, mem_Ici, mem_Icc]
        constructor
        · rintro ⟨⟨_, h2⟩, h3⟩
          exact ⟨h3, h2⟩
        · rintro ⟨h1, h2⟩
          exact ⟨⟨hy.1.trans_le h1, h2⟩, h1⟩,
      integral_Icc_eq_integral_Ioc]
  calc ∫ x in (0 : ℝ)..c, ∫ y in (0 : ℝ)..x, F x y
      = ∫ x in (0 : ℝ)..c, ∫ y in (0 : ℝ)..c, G x y :=
        intervalIntegral.integral_congr fun x hx ↦ (hrow x hx).symm
    _ = ∫ y in (0 : ℝ)..c, ∫ x in (0 : ℝ)..c, G x y := intervalIntegral_intervalIntegral_swap hGint
    _ = ∫ y in (0 : ℝ)..c, ∫ x in y..c, F x y := by
        refine intervalIntegral.integral_congr_ae (Filter.Eventually.of_forall fun y hy ↦ ?_)
        rw [uIoc_of_le hc] at hy
        exact hcol y hy

/-- Continuity of `x ↦ ∫_{l(x)}^{u(x)} f(x, t) dt` for continuous `f`, `l`, `u`. -/
theorem continuous_intervalIntegral_of_continuous {X : Type*} [TopologicalSpace X]
    {f : X → ℝ → ℂ} (hf : Continuous (Function.uncurry f)) {l u : X → ℝ} (hl : Continuous l)
    (hu : Continuous u) : Continuous fun x ↦ ∫ t in l x..u x, f x t := by
  have h : (fun x ↦ ∫ t in l x..u x, f x t) =
      fun x ↦ (∫ t in (0 : ℝ)..u x, f x t) - ∫ t in (0 : ℝ)..l x, f x t := by
    funext x
    have hfx : Continuous (f x) := hf.comp (Continuous.prodMk_right x)
    rw [intervalIntegral.integral_interval_sub_left (hfx.intervalIntegrable _ _)
      (hfx.intervalIntegrable _ _)]
  rw [h]
  exact (intervalIntegral.continuous_parametric_intervalIntegral_of_continuous hf hu).sub
    (intervalIntegral.continuous_parametric_intervalIntegral_of_continuous hf hl)

/-! ### Splitting an integral where the integrand changes form -/

theorem integral_congr_Ioo {f g : ℝ → ℂ} {a b : ℝ} (hab : a ≤ b)
    (h : ∀ t ∈ Ioo a b, f t = g t) : ∫ t in a..b, f t = ∫ t in a..b, g t := by
  rw [intervalIntegral.integral_of_le hab, intervalIntegral.integral_of_le hab,
    integral_Ioc_eq_integral_Ioo, integral_Ioc_eq_integral_Ioo]
  exact setIntegral_congr_fun measurableSet_Ioo h

theorem intervalIntegrable_of_eqOn_Ioo {f g : ℝ → ℂ} {a b : ℝ} (hab : a ≤ b) (hg : Continuous g)
    (h : ∀ t ∈ Ioo a b, f t = g t) : IntervalIntegrable f volume a b := by
  rw [intervalIntegrable_iff_integrableOn_Ioo_of_le hab]
  exact (hg.integrableOn_Icc.mono_set Ioo_subset_Icc_self).congr_fun (fun t ht ↦ (h t ht).symm)
    measurableSet_Ioo

theorem integral_split2 {f g₁ g₂ : ℝ → ℂ} {p q r : ℝ} (hpq : p ≤ q) (hqr : q ≤ r)
    (c₁ : Continuous g₁) (c₂ : Continuous g₂)
    (e₁ : ∀ t ∈ Ioo p q, f t = g₁ t) (e₂ : ∀ t ∈ Ioo q r, f t = g₂ t) :
    ∫ t in p..r, f t = (∫ t in p..q, g₁ t) + ∫ t in q..r, g₂ t := by
  rw [← intervalIntegral.integral_add_adjacent_intervals
    (intervalIntegrable_of_eqOn_Ioo hpq c₁ e₁) (intervalIntegrable_of_eqOn_Ioo hqr c₂ e₂),
    integral_congr_Ioo hpq e₁, integral_congr_Ioo hqr e₂]

theorem integral_split3 {f g₁ g₂ g₃ : ℝ → ℂ} {p q r s : ℝ} (hpq : p ≤ q) (hqr : q ≤ r)
    (hrs : r ≤ s) (c₁ : Continuous g₁) (c₂ : Continuous g₂) (c₃ : Continuous g₃)
    (e₁ : ∀ t ∈ Ioo p q, f t = g₁ t) (e₂ : ∀ t ∈ Ioo q r, f t = g₂ t)
    (e₃ : ∀ t ∈ Ioo r s, f t = g₃ t) :
    ∫ t in p..s, f t = (∫ t in p..q, g₁ t) + (∫ t in q..r, g₂ t) + ∫ t in r..s, g₃ t := by
  have i₂ := intervalIntegrable_of_eqOn_Ioo hqr c₂ e₂
  have i₃ := intervalIntegrable_of_eqOn_Ioo hrs c₃ e₃
  rw [← intervalIntegral.integral_add_adjacent_intervals
    (intervalIntegrable_of_eqOn_Ioo hpq c₁ e₁) (i₂.trans i₃),
    ← intervalIntegral.integral_add_adjacent_intervals i₂ i₃, integral_congr_Ioo hpq e₁,
    integral_congr_Ioo hqr e₂, integral_congr_Ioo hrs e₃, add_assoc]

theorem integral_add₃ {f g h : ℝ → ℂ} (hf : Continuous f) (hg : Continuous g) (hh : Continuous h)
    (a b : ℝ) : ∫ x in a..b, f x + g x + h x =
      (∫ x in a..b, f x) + (∫ x in a..b, g x) + ∫ x in a..b, h x := by
  have hfg : Continuous fun x ↦ f x + g x := hf.add hg
  rw [intervalIntegral.integral_add (hfg.intervalIntegrable _ _) (hh.intervalIntegrable _ _),
    intervalIntegral.integral_add (hf.intervalIntegrable _ _) (hg.intervalIntegrable _ _)]

/-! ### The six orderings of three times -/

/-- Joint continuity of a function of three real variables. -/
def Continuous₃ (g : ℝ → ℝ → ℝ → ℂ) : Prop := Continuous fun p : ℝ × ℝ × ℝ ↦ g p.1 p.2.1 p.2.2

theorem Continuous₃.comp {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) {X : Type*}
    [TopologicalSpace X] {f₁ f₂ f₃ : X → ℝ} (h₁ : Continuous f₁) (h₂ : Continuous f₂)
    (h₃ : Continuous f₃) : Continuous fun x ↦ g (f₁ x) (f₂ x) (f₃ x) :=
  Continuous.comp (g := fun p : ℝ × ℝ × ℝ ↦ g p.1 p.2.1 p.2.2) hg (h₁.prodMk (h₂.prodMk h₃))

/-- `(x, y) ↦ ∫_{l(x, y)}^{u(x, y)} g(x, y, t) dt` is continuous. -/
theorem Continuous₃.integral {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) {l u : ℝ × ℝ → ℝ}
    (hl : Continuous l) (hu : Continuous u) :
    Continuous fun p : ℝ × ℝ ↦ ∫ t in l p..u p, g p.1 p.2 t :=
  continuous_intervalIntegral_of_continuous (f := fun p t ↦ g p.1 p.2 t)
    (hg.comp (by fun_prop) (by fun_prop) (by fun_prop)) hl hu

/-- `x ↦ ∫_{l'(x)}^{u'(x)} dy ∫_{l(x, y)}^{u(x, y)} g(x, y, t) dt` is continuous. -/
theorem Continuous₃.integral₂ {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) {l u : ℝ × ℝ → ℝ}
    (hl : Continuous l) (hu : Continuous u) {l' u' : ℝ → ℝ} (hl' : Continuous l')
    (hu' : Continuous u') :
    Continuous fun x ↦ ∫ y in l' x..u' x, ∫ t in l (x, y)..u (x, y), g x y t :=
  continuous_intervalIntegral_of_continuous
    (f := fun x y ↦ ∫ t in l (x, y)..u (x, y), g x y t) (hg.integral hl hu) hl' hu'

/-- The iterated integral over the ordered simplex `β > s₁ > s₂ > s₃ > 0`. -/
noncomputable def simplexIntegral (β : ℝ) (g : ℝ → ℝ → ℝ → ℂ) : ℂ :=
  ∫ s₁ in (0 : ℝ)..β, ∫ s₂ in (0 : ℝ)..s₁, ∫ s₃ in (0 : ℝ)..s₂, g s₁ s₂ s₃

theorem region_132 {β : ℝ} {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) (hβ : 0 ≤ β) :
    ∫ τ₁ in (0 : ℝ)..β, ∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₂..τ₁, g τ₁ τ₂ τ₃ =
      simplexIntegral β fun s₁ s₂ s₃ ↦ g s₁ s₃ s₂ := by
  refine intervalIntegral.integral_congr fun τ₁ hτ₁ ↦ ?_
  rw [uIcc_of_le hβ] at hτ₁
  exact (integral_swap_triangle (F := fun x y ↦ g τ₁ y x)
    (hg.comp continuous_const continuous_snd continuous_fst) hτ₁.1).symm

theorem region_312 {β : ℝ} {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) (hβ : 0 ≤ β) :
    ∫ τ₁ in (0 : ℝ)..β, ∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₁..β, g τ₁ τ₂ τ₃ =
      simplexIntegral β fun s₁ s₂ s₃ ↦ g s₂ s₃ s₁ := by
  rw [intervalIntegral.integral_congr (g := fun τ₁ ↦ ∫ τ₃ in τ₁..β, ∫ τ₂ in (0 : ℝ)..τ₁,
    g τ₁ τ₂ τ₃) fun τ₁ _ ↦ integral_swap_rect (F := fun τ₂ τ₃ ↦ g τ₁ τ₂ τ₃)
      (hg.comp continuous_const continuous_fst continuous_snd) 0 τ₁ τ₁ β]
  exact (integral_swap_triangle (F := fun x y ↦ ∫ τ₂ in (0 : ℝ)..y, g y τ₂ x)
    (continuous_intervalIntegral_of_continuous (f := fun p : ℝ × ℝ ↦ fun t ↦ g p.2 t p.1)
      (hg.comp (by fun_prop) (by fun_prop) (by fun_prop)) continuous_const continuous_snd)
    hβ).symm

theorem region_213 {β : ℝ} {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) (hβ : 0 ≤ β) :
    ∫ τ₁ in (0 : ℝ)..β, ∫ τ₂ in τ₁..β, ∫ τ₃ in (0 : ℝ)..τ₁, g τ₁ τ₂ τ₃ =
      simplexIntegral β fun s₁ s₂ s₃ ↦ g s₂ s₁ s₃ :=
  (integral_swap_triangle (F := fun x y ↦ ∫ τ₃ in (0 : ℝ)..y, g y x τ₃)
    (continuous_intervalIntegral_of_continuous (f := fun p : ℝ × ℝ ↦ fun t ↦ g p.2 p.1 t)
      (hg.comp (by fun_prop) (by fun_prop) (by fun_prop)) continuous_const continuous_snd)
    hβ).symm

theorem region_231 {β : ℝ} {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) (hβ : 0 ≤ β) :
    ∫ τ₁ in (0 : ℝ)..β, ∫ τ₂ in τ₁..β, ∫ τ₃ in τ₁..τ₂, g τ₁ τ₂ τ₃ =
      simplexIntegral β fun s₁ s₂ s₃ ↦ g s₃ s₁ s₂ := by
  rw [← integral_swap_triangle (F := fun x y ↦ ∫ τ₃ in y..x, g y x τ₃)
    (continuous_intervalIntegral_of_continuous (f := fun p : ℝ × ℝ ↦ fun t ↦ g p.2 p.1 t)
      (hg.comp (by fun_prop) (by fun_prop) (by fun_prop)) continuous_snd continuous_fst) hβ]
  refine intervalIntegral.integral_congr fun τ₂ hτ₂ ↦ ?_
  rw [uIcc_of_le hβ] at hτ₂
  exact (integral_swap_triangle (F := fun x y ↦ g y τ₂ x)
    (hg.comp continuous_snd continuous_const continuous_fst) hτ₂.1).symm

theorem region_321 {β : ℝ} {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) (hβ : 0 ≤ β) :
    ∫ τ₁ in (0 : ℝ)..β, ∫ τ₂ in τ₁..β, ∫ τ₃ in τ₂..β, g τ₁ τ₂ τ₃ =
      simplexIntegral β fun s₁ s₂ s₃ ↦ g s₃ s₂ s₁ := by
  rw [← integral_swap_triangle (F := fun x y ↦ ∫ τ₃ in x..β, g y x τ₃)
    (continuous_intervalIntegral_of_continuous (f := fun p : ℝ × ℝ ↦ fun t ↦ g p.2 p.1 t)
      (hg.comp (by fun_prop) (by fun_prop) (by fun_prop)) continuous_fst continuous_const) hβ]
  rw [intervalIntegral.integral_congr (g := fun τ₂ ↦ ∫ τ₃ in τ₂..β, ∫ τ₁ in (0 : ℝ)..τ₂,
    g τ₁ τ₂ τ₃) fun τ₂ _ ↦ integral_swap_rect (F := fun τ₁ τ₃ ↦ g τ₁ τ₂ τ₃)
      (hg.comp continuous_fst continuous_const continuous_snd) 0 τ₂ τ₂ β]
  exact (integral_swap_triangle (F := fun x y ↦ ∫ τ₁ in (0 : ℝ)..y, g τ₁ y x)
    (continuous_intervalIntegral_of_continuous (f := fun p : ℝ × ℝ ↦ fun t ↦ g t p.2 p.1)
      (hg.comp (by fun_prop) (by fun_prop) (by fun_prop)) continuous_const continuous_snd)
    hβ).symm

/-- The triple integral over the cube `[0, β]³` of an integrand that agrees with a continuous
function on each ordering of the three times is a sum of six integrals over the ordered simplex.
The function `g₃₁₂` is the one for `τ₃ > τ₁ > τ₂`, and so on. -/
theorem integral_cube {β : ℝ} (hβ : 0 ≤ β) {F : ℝ → ℝ → ℝ → ℂ}
    {g₁₂₃ g₁₃₂ g₃₁₂ g₂₁₃ g₂₃₁ g₃₂₁ : ℝ → ℝ → ℝ → ℂ}
    (c₁₂₃ : Continuous₃ g₁₂₃) (c₁₃₂ : Continuous₃ g₁₃₂) (c₃₁₂ : Continuous₃ g₃₁₂)
    (c₂₁₃ : Continuous₃ g₂₁₃) (c₂₃₁ : Continuous₃ g₂₃₁) (c₃₂₁ : Continuous₃ g₃₂₁)
    (e₁₂₃ : ∀ τ₁ τ₂ τ₃, 0 < τ₃ → τ₃ < τ₂ → τ₂ < τ₁ → τ₁ < β → F τ₁ τ₂ τ₃ = g₁₂₃ τ₁ τ₂ τ₃)
    (e₁₃₂ : ∀ τ₁ τ₂ τ₃, 0 < τ₂ → τ₂ < τ₃ → τ₃ < τ₁ → τ₁ < β → F τ₁ τ₂ τ₃ = g₁₃₂ τ₁ τ₂ τ₃)
    (e₃₁₂ : ∀ τ₁ τ₂ τ₃, 0 < τ₂ → τ₂ < τ₁ → τ₁ < τ₃ → τ₃ < β → F τ₁ τ₂ τ₃ = g₃₁₂ τ₁ τ₂ τ₃)
    (e₂₁₃ : ∀ τ₁ τ₂ τ₃, 0 < τ₃ → τ₃ < τ₁ → τ₁ < τ₂ → τ₂ < β → F τ₁ τ₂ τ₃ = g₂₁₃ τ₁ τ₂ τ₃)
    (e₂₃₁ : ∀ τ₁ τ₂ τ₃, 0 < τ₁ → τ₁ < τ₃ → τ₃ < τ₂ → τ₂ < β → F τ₁ τ₂ τ₃ = g₂₃₁ τ₁ τ₂ τ₃)
    (e₃₂₁ : ∀ τ₁ τ₂ τ₃, 0 < τ₁ → τ₁ < τ₂ → τ₂ < τ₃ → τ₃ < β → F τ₁ τ₂ τ₃ = g₃₂₁ τ₁ τ₂ τ₃) :
    ∫ τ₁ in (0 : ℝ)..β, ∫ τ₂ in (0 : ℝ)..β, ∫ τ₃ in (0 : ℝ)..β, F τ₁ τ₂ τ₃ =
      simplexIntegral β g₁₂₃ + simplexIntegral β (fun s₁ s₂ s₃ ↦ g₁₃₂ s₁ s₃ s₂) +
        simplexIntegral β (fun s₁ s₂ s₃ ↦ g₃₁₂ s₂ s₃ s₁) +
        simplexIntegral β (fun s₁ s₂ s₃ ↦ g₂₁₃ s₂ s₁ s₃) +
        simplexIntegral β (fun s₁ s₂ s₃ ↦ g₂₃₁ s₃ s₁ s₂) +
        simplexIntegral β (fun s₁ s₂ s₃ ↦ g₃₂₁ s₃ s₂ s₁) := by
  have ci : ∀ {g : ℝ → ℝ → ℝ → ℂ}, Continuous₃ g → ∀ {l u : ℝ × ℝ → ℝ}, Continuous l →
      Continuous u → ∀ τ₁ : ℝ, Continuous fun τ₂ ↦ ∫ τ₃ in l (τ₁, τ₂)..u (τ₁, τ₂), g τ₁ τ₂ τ₃ :=
    fun hg _ _ hl hu τ₁ ↦ (hg.integral hl hu).comp (continuous_const.prodMk continuous_id)
  have cc : ∀ {g : ℝ → ℝ → ℝ → ℂ}, Continuous₃ g → ∀ τ₁ τ₂ : ℝ, Continuous (g τ₁ τ₂) :=
    fun hg τ₁ τ₂ ↦ hg.comp continuous_const continuous_const continuous_id
  -- the two inner integrals, for `τ₁ ∈ (0, β)`
  have inner : ∀ τ₁ ∈ Ioo 0 β, ∫ τ₂ in (0 : ℝ)..β, ∫ τ₃ in (0 : ℝ)..β, F τ₁ τ₂ τ₃ =
      (∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in (0 : ℝ)..τ₂, g₁₂₃ τ₁ τ₂ τ₃) +
      (∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₂..τ₁, g₁₃₂ τ₁ τ₂ τ₃) +
      (∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₁..β, g₃₁₂ τ₁ τ₂ τ₃) +
      (∫ τ₂ in τ₁..β, ∫ τ₃ in (0 : ℝ)..τ₁, g₂₁₃ τ₁ τ₂ τ₃) +
      (∫ τ₂ in τ₁..β, ∫ τ₃ in τ₁..τ₂, g₂₃₁ τ₁ τ₂ τ₃) +
      ∫ τ₂ in τ₁..β, ∫ τ₃ in τ₂..β, g₃₂₁ τ₁ τ₂ τ₃ := by
    rintro τ₁ ⟨h0, hβ'⟩
    have a₁ : Continuous fun τ₂ ↦ ∫ τ₃ in (0 : ℝ)..τ₂, g₁₂₃ τ₁ τ₂ τ₃ :=
      ci c₁₂₃ (l := fun _ ↦ 0) (u := fun p ↦ p.2) continuous_const continuous_snd τ₁
    have a₂ : Continuous fun τ₂ ↦ ∫ τ₃ in τ₂..τ₁, g₁₃₂ τ₁ τ₂ τ₃ :=
      ci c₁₃₂ (l := fun p ↦ p.2) (u := fun p ↦ p.1) continuous_snd continuous_fst τ₁
    have a₃ : Continuous fun τ₂ ↦ ∫ τ₃ in τ₁..β, g₃₁₂ τ₁ τ₂ τ₃ :=
      ci c₃₁₂ (l := fun p ↦ p.1) (u := fun _ ↦ β) continuous_fst continuous_const τ₁
    have b₁ : Continuous fun τ₂ ↦ ∫ τ₃ in (0 : ℝ)..τ₁, g₂₁₃ τ₁ τ₂ τ₃ :=
      ci c₂₁₃ (l := fun _ ↦ 0) (u := fun p ↦ p.1) continuous_const continuous_fst τ₁
    have b₂ : Continuous fun τ₂ ↦ ∫ τ₃ in τ₁..τ₂, g₂₃₁ τ₁ τ₂ τ₃ :=
      ci c₂₃₁ (l := fun p ↦ p.1) (u := fun p ↦ p.2) continuous_fst continuous_snd τ₁
    have b₃ : Continuous fun τ₂ ↦ ∫ τ₃ in τ₂..β, g₃₂₁ τ₁ τ₂ τ₃ :=
      ci c₃₂₁ (l := fun p ↦ p.2) (u := fun _ ↦ β) continuous_snd continuous_const τ₁
    have a : Continuous fun τ₂ ↦ (∫ τ₃ in (0 : ℝ)..τ₂, g₁₂₃ τ₁ τ₂ τ₃) +
        (∫ τ₃ in τ₂..τ₁, g₁₃₂ τ₁ τ₂ τ₃) + ∫ τ₃ in τ₁..β, g₃₁₂ τ₁ τ₂ τ₃ := (a₁.add a₂).add a₃
    have b : Continuous fun τ₂ ↦ (∫ τ₃ in (0 : ℝ)..τ₁, g₂₁₃ τ₁ τ₂ τ₃) +
        (∫ τ₃ in τ₁..τ₂, g₂₃₁ τ₁ τ₂ τ₃) + ∫ τ₃ in τ₂..β, g₃₂₁ τ₁ τ₂ τ₃ := (b₁.add b₂).add b₃
    rw [integral_split2 h0.le hβ'.le a b]
    · rw [integral_add₃ a₁ a₂ a₃, integral_add₃ b₁ b₂ b₃]
      ring
    · rintro τ₂ ⟨h1, h2⟩
      exact integral_split3 h1.le h2.le hβ'.le (cc c₁₂₃ τ₁ τ₂) (cc c₁₃₂ τ₁ τ₂) (cc c₃₁₂ τ₁ τ₂)
        (fun τ₃ h ↦ e₁₂₃ τ₁ τ₂ τ₃ h.1 h.2 h2 hβ') (fun τ₃ h ↦ e₁₃₂ τ₁ τ₂ τ₃ h1 h.1 h.2 hβ')
        (fun τ₃ h ↦ e₃₁₂ τ₁ τ₂ τ₃ h1 h2 h.1 h.2)
    · rintro τ₂ ⟨h1, h2⟩
      exact integral_split3 h0.le h1.le h2.le (cc c₂₁₃ τ₁ τ₂) (cc c₂₃₁ τ₁ τ₂) (cc c₃₂₁ τ₁ τ₂)
        (fun τ₃ h ↦ e₂₁₃ τ₁ τ₂ τ₃ h.1 h.2 h1 h2) (fun τ₃ h ↦ e₂₃₁ τ₁ τ₂ τ₃ h0 h.1 h.2 h2)
        (fun τ₃ h ↦ e₃₂₁ τ₁ τ₂ τ₃ h0 h1 h.1 h.2)
  rw [integral_congr_Ioo hβ inner]
  -- linearity in `τ₁`
  have d₁ : Continuous fun τ₁ ↦ ∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in (0 : ℝ)..τ₂, g₁₂₃ τ₁ τ₂ τ₃ :=
    c₁₂₃.integral₂ (l := fun _ ↦ 0) (u := fun p ↦ p.2) continuous_const continuous_snd
      (l' := fun _ ↦ 0) (u' := id) continuous_const continuous_id
  have d₂ : Continuous fun τ₁ ↦ ∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₂..τ₁, g₁₃₂ τ₁ τ₂ τ₃ :=
    c₁₃₂.integral₂ (l := fun p ↦ p.2) (u := fun p ↦ p.1) continuous_snd continuous_fst
      (l' := fun _ ↦ 0) (u' := id) continuous_const continuous_id
  have d₃ : Continuous fun τ₁ ↦ ∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₁..β, g₃₁₂ τ₁ τ₂ τ₃ :=
    c₃₁₂.integral₂ (l := fun p ↦ p.1) (u := fun _ ↦ β) continuous_fst continuous_const
      (l' := fun _ ↦ 0) (u' := id) continuous_const continuous_id
  have d₄ : Continuous fun τ₁ ↦ ∫ τ₂ in τ₁..β, ∫ τ₃ in (0 : ℝ)..τ₁, g₂₁₃ τ₁ τ₂ τ₃ :=
    c₂₁₃.integral₂ (l := fun _ ↦ 0) (u := fun p ↦ p.1) continuous_const continuous_fst
      (l' := id) (u' := fun _ ↦ β) continuous_id continuous_const
  have d₅ : Continuous fun τ₁ ↦ ∫ τ₂ in τ₁..β, ∫ τ₃ in τ₁..τ₂, g₂₃₁ τ₁ τ₂ τ₃ :=
    c₂₃₁.integral₂ (l := fun p ↦ p.1) (u := fun p ↦ p.2) continuous_fst continuous_snd
      (l' := id) (u' := fun _ ↦ β) continuous_id continuous_const
  have d₆ : Continuous fun τ₁ ↦ ∫ τ₂ in τ₁..β, ∫ τ₃ in τ₂..β, g₃₂₁ τ₁ τ₂ τ₃ :=
    c₃₂₁.integral₂ (l := fun p ↦ p.2) (u := fun _ ↦ β) continuous_snd continuous_const
      (l' := id) (u' := fun _ ↦ β) continuous_id continuous_const
  have d₁₂₃ : Continuous fun τ₁ ↦ (∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in (0 : ℝ)..τ₂, g₁₂₃ τ₁ τ₂ τ₃) +
      (∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₂..τ₁, g₁₃₂ τ₁ τ₂ τ₃) +
      ∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₁..β, g₃₁₂ τ₁ τ₂ τ₃ := (d₁.add d₂).add d₃
  have d₁₂₃₄ : Continuous fun τ₁ ↦ (∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in (0 : ℝ)..τ₂, g₁₂₃ τ₁ τ₂ τ₃) +
      (∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₂..τ₁, g₁₃₂ τ₁ τ₂ τ₃) +
      (∫ τ₂ in (0 : ℝ)..τ₁, ∫ τ₃ in τ₁..β, g₃₁₂ τ₁ τ₂ τ₃) +
      ∫ τ₂ in τ₁..β, ∫ τ₃ in (0 : ℝ)..τ₁, g₂₁₃ τ₁ τ₂ τ₃ := d₁₂₃.add d₄
  rw [integral_add₃ d₁₂₃₄ d₅ d₆,
    intervalIntegral.integral_add (d₁₂₃.intervalIntegrable _ _) (d₄.intervalIntegrable _ _),
    integral_add₃ d₁ d₂ d₃,
    region_132 c₁₃₂ hβ, region_312 c₃₁₂ hβ, region_213 c₂₁₃ hβ, region_231 c₂₃₁ hβ,
    region_321 c₃₂₁ hβ]
  rfl

/-! ### Linearity of the simplex integral -/

theorem Continuous₃.inner {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) (s₁ s₂ : ℝ) :
    Continuous (g s₁ s₂) :=
  hg.comp continuous_const continuous_const continuous_id

theorem Continuous₃.mid {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) (s₁ : ℝ) :
    Continuous fun s₂ ↦ ∫ s₃ in (0 : ℝ)..s₂, g s₁ s₂ s₃ :=
  (hg.integral (l := fun _ ↦ 0) (u := fun p ↦ p.2) continuous_const continuous_snd).comp
    (continuous_const.prodMk continuous_id)

theorem Continuous₃.outer {g : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g) :
    Continuous fun s₁ ↦ ∫ s₂ in (0 : ℝ)..s₁, ∫ s₃ in (0 : ℝ)..s₂, g s₁ s₂ s₃ :=
  hg.integral₂ (l := fun _ ↦ 0) (u := fun p ↦ p.2) continuous_const continuous_snd
    (l' := fun _ ↦ 0) (u' := id) continuous_const continuous_id

theorem simplexIntegral_const_mul (β : ℝ) (c : ℂ) (g : ℝ → ℝ → ℝ → ℂ) :
    simplexIntegral β (fun s₁ s₂ s₃ ↦ c * g s₁ s₂ s₃) = c * simplexIntegral β g := by
  simp only [simplexIntegral, intervalIntegral.integral_const_mul]

theorem simplexIntegral_sub (β : ℝ) {g h : ℝ → ℝ → ℝ → ℂ} (hg : Continuous₃ g)
    (hh : Continuous₃ h) :
    simplexIntegral β (fun s₁ s₂ s₃ ↦ g s₁ s₂ s₃ - h s₁ s₂ s₃) =
      simplexIntegral β g - simplexIntegral β h := by
  unfold simplexIntegral
  have e₃ : ∀ s₁ s₂ : ℝ, ∫ s₃ in (0 : ℝ)..s₂, g s₁ s₂ s₃ - h s₁ s₂ s₃ =
      (∫ s₃ in (0 : ℝ)..s₂, g s₁ s₂ s₃) - ∫ s₃ in (0 : ℝ)..s₂, h s₁ s₂ s₃ := fun s₁ s₂ ↦
    intervalIntegral.integral_sub ((hg.inner s₁ s₂).intervalIntegrable _ _)
      ((hh.inner s₁ s₂).intervalIntegrable _ _)
  simp_rw [e₃]
  have e₂ : ∀ s₁ : ℝ, ∫ s₂ in (0 : ℝ)..s₁, (∫ s₃ in (0 : ℝ)..s₂, g s₁ s₂ s₃) -
      ∫ s₃ in (0 : ℝ)..s₂, h s₁ s₂ s₃ = (∫ s₂ in (0 : ℝ)..s₁, ∫ s₃ in (0 : ℝ)..s₂, g s₁ s₂ s₃) -
        ∫ s₂ in (0 : ℝ)..s₁, ∫ s₃ in (0 : ℝ)..s₂, h s₁ s₂ s₃ := fun s₁ ↦
    intervalIntegral.integral_sub ((hg.mid s₁).intervalIntegrable _ _)
      ((hh.mid s₁).intervalIntegrable _ _)
  simp_rw [e₂]
  exact intervalIntegral.integral_sub (hg.outer.intervalIntegrable _ _)
    (hh.outer.intervalIntegrable _ _)

theorem simplexIntegral_finsetSum {ι : Type*} (S : Finset ι) (β : ℝ)
    {g : ι → ℝ → ℝ → ℝ → ℂ} (hg : ∀ i, Continuous₃ (g i)) :
    simplexIntegral β (fun s₁ s₂ s₃ ↦ ∑ i ∈ S, g i s₁ s₂ s₃) =
      ∑ i ∈ S, simplexIntegral β (g i) := by
  unfold simplexIntegral
  have e₃ : ∀ s₁ s₂ : ℝ, ∫ s₃ in (0 : ℝ)..s₂, ∑ i ∈ S, g i s₁ s₂ s₃ =
      ∑ i ∈ S, ∫ s₃ in (0 : ℝ)..s₂, g i s₁ s₂ s₃ := fun s₁ s₂ ↦
    intervalIntegral.integral_finsetSum fun i _ ↦ ((hg i).inner s₁ s₂).intervalIntegrable _ _
  simp_rw [e₃]
  have e₂ : ∀ s₁ : ℝ, ∫ s₂ in (0 : ℝ)..s₁, ∑ i ∈ S, ∫ s₃ in (0 : ℝ)..s₂, g i s₁ s₂ s₃ =
      ∑ i ∈ S, ∫ s₂ in (0 : ℝ)..s₁, ∫ s₃ in (0 : ℝ)..s₂, g i s₁ s₂ s₃ := fun s₁ ↦
    intervalIntegral.integral_finsetSum fun i _ ↦ ((hg i).mid s₁).intervalIntegrable _ _
  simp_rw [e₂]
  exact intervalIntegral.integral_finsetSum fun i _ ↦ (hg i).outer.intervalIntegrable _ _

theorem simplexIntegral_cexp {β : ℝ} {a b c : ℂ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
    simplexIntegral β (fun s₁ s₂ s₃ ↦ cexp (a * s₁ + b * s₂ + c * s₃)) = simplex3 a b c β :=
  integral_simplex3 ha hb hc β

/-! ### Two times -/

/-- The iterated integral over the ordered triangle `β > s₁ > s₂ > 0`. -/
noncomputable def triangleIntegral (β : ℝ) (g : ℝ → ℝ → ℂ) : ℂ :=
  ∫ s₁ in (0 : ℝ)..β, ∫ s₂ in (0 : ℝ)..s₁, g s₁ s₂

/-- The double integral over the square `[0, β]²` of an integrand that agrees with a continuous
function on each ordering of the two times. -/
theorem integral_square {β : ℝ} (hβ : 0 ≤ β) {F g₁₂ g₂₁ : ℝ → ℝ → ℂ}
    (c₁₂ : Continuous (Function.uncurry g₁₂)) (c₂₁ : Continuous (Function.uncurry g₂₁))
    (e₁₂ : ∀ τ₁ τ₂, 0 < τ₂ → τ₂ < τ₁ → τ₁ < β → F τ₁ τ₂ = g₁₂ τ₁ τ₂)
    (e₂₁ : ∀ τ₁ τ₂, 0 < τ₁ → τ₁ < τ₂ → τ₂ < β → F τ₁ τ₂ = g₂₁ τ₁ τ₂) :
    ∫ τ₁ in (0 : ℝ)..β, ∫ τ₂ in (0 : ℝ)..β, F τ₁ τ₂ =
      triangleIntegral β g₁₂ + triangleIntegral β fun s₁ s₂ ↦ g₂₁ s₂ s₁ := by
  have inner : ∀ τ₁ ∈ Ioo 0 β, ∫ τ₂ in (0 : ℝ)..β, F τ₁ τ₂ =
      (∫ τ₂ in (0 : ℝ)..τ₁, g₁₂ τ₁ τ₂) + ∫ τ₂ in τ₁..β, g₂₁ τ₁ τ₂ := by
    rintro τ₁ ⟨h0, hβ'⟩
    exact integral_split2 h0.le hβ'.le (c₁₂.comp (continuous_const.prodMk continuous_id))
      (c₂₁.comp (continuous_const.prodMk continuous_id)) (fun τ₂ h ↦ e₁₂ τ₁ τ₂ h.1 h.2 hβ')
      (fun τ₂ h ↦ e₂₁ τ₁ τ₂ h0 h.1 h.2)
  have d₁ : Continuous fun τ₁ ↦ ∫ τ₂ in (0 : ℝ)..τ₁, g₁₂ τ₁ τ₂ :=
    continuous_intervalIntegral_of_continuous c₁₂ continuous_const continuous_id
  have d₂ : Continuous fun τ₁ ↦ ∫ τ₂ in τ₁..β, g₂₁ τ₁ τ₂ :=
    continuous_intervalIntegral_of_continuous c₂₁ continuous_id continuous_const
  rw [integral_congr_Ioo hβ inner,
    intervalIntegral.integral_add (d₁.intervalIntegrable _ _) (d₂.intervalIntegrable _ _)]
  congr 1
  exact (integral_swap_triangle (F := fun x y ↦ g₂₁ y x) (c₂₁.comp continuous_swap) hβ).symm

theorem triangleIntegral_const_mul (β : ℝ) (c : ℂ) (g : ℝ → ℝ → ℂ) :
    triangleIntegral β (fun s₁ s₂ ↦ c * g s₁ s₂) = c * triangleIntegral β g := by
  simp only [triangleIntegral, intervalIntegral.integral_const_mul]

theorem triangleIntegral_finsetSum {ι : Type*} (S : Finset ι) (β : ℝ)
    {g : ι → ℝ → ℝ → ℂ} (hg : ∀ i, Continuous (Function.uncurry (g i))) :
    triangleIntegral β (fun s₁ s₂ ↦ ∑ i ∈ S, g i s₁ s₂) = ∑ i ∈ S, triangleIntegral β (g i) := by
  unfold triangleIntegral
  have e₂ : ∀ s₁ : ℝ, ∫ s₂ in (0 : ℝ)..s₁, ∑ i ∈ S, g i s₁ s₂ =
      ∑ i ∈ S, ∫ s₂ in (0 : ℝ)..s₁, g i s₁ s₂ := fun s₁ ↦
    intervalIntegral.integral_finsetSum fun i _ ↦
      ((hg i).comp (continuous_const.prodMk continuous_id)).intervalIntegrable _ _
  simp_rw [e₂]
  exact intervalIntegral.integral_finsetSum fun i _ ↦
    (continuous_intervalIntegral_of_continuous (hg i) continuous_const
      continuous_id).intervalIntegrable _ _

theorem triangleIntegral_cexp {β : ℝ} {a b : ℂ} (hb : b ≠ 0) :
    triangleIntegral β (fun s₁ s₂ ↦ cexp (a * s₁ + b * s₂)) = simplex2 a b β :=
  integral_simplex2 hb β

theorem simplexIntegral_sum₄ {ι : Type*} [Fintype ι] (β : ℝ)
    {g : ι → ι → ι → ι → ℝ → ℝ → ℝ → ℂ} (hg : ∀ i j k l, Continuous₃ (g i j k l)) :
    simplexIntegral β (fun s₁ s₂ s₃ ↦ ∑ i, ∑ j, ∑ k, ∑ l, g i j k l s₁ s₂ s₃) =
      ∑ i, ∑ j, ∑ k, ∑ l, simplexIntegral β (g i j k l) := by
  have c₁ : ∀ i j k, Continuous₃ fun s₁ s₂ s₃ ↦ ∑ l, g i j k l s₁ s₂ s₃ := fun i j k ↦
    continuous_finsetSum _ fun l _ ↦ hg i j k l
  have c₂ : ∀ i j, Continuous₃ fun s₁ s₂ s₃ ↦ ∑ k, ∑ l, g i j k l s₁ s₂ s₃ := fun i j ↦
    continuous_finsetSum _ fun k _ ↦ c₁ i j k
  have c₃ : ∀ i, Continuous₃ fun s₁ s₂ s₃ ↦ ∑ j, ∑ k, ∑ l, g i j k l s₁ s₂ s₃ := fun i ↦
    continuous_finsetSum _ fun j _ ↦ c₂ i j
  rw [simplexIntegral_finsetSum _ _ c₃]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [simplexIntegral_finsetSum _ _ (c₂ i)]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  rw [simplexIntegral_finsetSum _ _ (c₁ i j)]
  exact Finset.sum_congr rfl fun k _ ↦ simplexIntegral_finsetSum _ _ (hg i j k)

theorem triangleIntegral_sum₂ {ι : Type*} [Fintype ι] (β : ℝ) {g : ι → ι → ℝ → ℝ → ℂ}
    (hg : ∀ i j, Continuous (Function.uncurry (g i j))) :
    triangleIntegral β (fun s₁ s₂ ↦ ∑ i, ∑ j, g i j s₁ s₂) =
      ∑ i, ∑ j, triangleIntegral β (g i j) := by
  have c₁ : ∀ i, Continuous (Function.uncurry fun s₁ s₂ ↦ ∑ j, g i j s₁ s₂) := fun i ↦ by
    simpa [Function.uncurry_def] using continuous_finsetSum _ fun j _ ↦ hg i j
  rw [triangleIntegral_finsetSum _ _ c₁]
  exact Finset.sum_congr rfl fun i _ ↦ triangleIntegral_finsetSum _ _ (hg i)

/-! ### Weighted closed forms

In the correlation functions the simplex integrals come with a Boltzmann weight `P`, and
`P e^{rβ}` is again a weight (up to sign) for every partial sum `r` of the rates. These forms
express the integrals through the weights, without exponentials. -/

/-- `P · expInt(r, β)` when `P e^{rβ} = -Q`. -/
theorem mul_expInt {β : ℝ} {P Q r : ℂ} (h : P * cexp (r * β) = -Q) (hr : r ≠ 0) :
    P * expInt r β = (-Q - P) / r := by
  rw [expInt, ite_eq_right hr, ← h]
  ring

/-- `P ∫_{β>s₁>s₂>0} e^{r₁s₁ + r₂s₂}` through the weights `P e^{r₁β} = -Q₁` and
`P e^{r₁₂β} = Q₂`, where `r₁₂ = r₁ + r₂`; the condition `C₁₂` says whether `r₁₂` vanishes. -/
noncomputable def triangleW (β P Q₁ Q₂ r₁ r₂ r₁₂ : ℂ) (C₁₂ : Prop) [Decidable C₁₂] : ℂ :=
  ((if C₁₂ then P * β else (Q₂ - P) / r₁₂) - (-Q₁ - P) / r₁) / r₂

theorem mul_simplex2 {β : ℝ} {P Q₁ Q₂ r₁ r₂ r₁₂ : ℂ} {C₁₂ : Prop} [Decidable C₁₂]
    (e₁₂ : r₁ + r₂ = r₁₂) (h₁ : P * cexp (r₁ * β) = -Q₁) (h₁₂ : P * cexp (r₁₂ * β) = Q₂)
    (n₁ : r₁ ≠ 0) (c₁₂ : r₁₂ = 0 ↔ C₁₂) :
    P * simplex2 r₁ r₂ β = triangleW β P Q₁ Q₂ r₁ r₂ r₁₂ C₁₂ := by
  subst e₁₂
  rw [simplex2, triangleW, ← mul_div_assoc, mul_sub, mul_expInt h₁ n₁]
  by_cases h : C₁₂
  · rw [ite_eq_left h, expInt, ite_eq_left (c₁₂.mpr h)]
  · rw [ite_eq_right h, mul_expInt (Q := -Q₂) (by rw [h₁₂, neg_neg]) (mt c₁₂.mp h)]
    ring

/-- `P ∫_{β>s₁>s₂>s₃>0} e^{r₁s₁ + r₂s₂ + r₃s₃}` through the weights `P e^{r₁β} = -Q₁`,
`P e^{r₁₂β} = Q₂`, `P e^{r₁₂₃β} = -Q₃`, where `r₁₂`, `r₂₃`, `r₁₂₃` are the partial sums of the
rates; the conditions `C₁₂`, `C₂₃` say whether `r₁₂`, `r₂₃` vanish. -/
noncomputable def simplexW (β P Q₁ Q₂ Q₃ r₁ r₂ r₃ r₁₂ r₂₃ r₁₂₃ : ℂ) (C₁₂ C₂₃ : Prop)
    [Decidable C₁₂] [Decidable C₂₃] : ℂ :=
  ((if C₂₃ then (-β * Q₁ - (-Q₁ - P) / r₁) / r₁
      else ((-Q₃ - P) / r₁₂₃ - (-Q₁ - P) / r₁) / r₂₃) -
    ((if C₁₂ then P * β else (Q₂ - P) / r₁₂) - (-Q₁ - P) / r₁) / r₂) / r₃

theorem mul_simplex3 {β : ℝ} {P Q₁ Q₂ Q₃ r₁ r₂ r₃ r₁₂ r₂₃ r₁₂₃ : ℂ} {C₁₂ C₂₃ : Prop}
    [Decidable C₁₂] [Decidable C₂₃] (e₁₂ : r₁ + r₂ = r₁₂) (e₂₃ : r₂ + r₃ = r₂₃)
    (e₁₂₃ : r₁ + r₂ + r₃ = r₁₂₃) (h₁ : P * cexp (r₁ * β) = -Q₁)
    (h₁₂ : P * cexp (r₁₂ * β) = Q₂) (h₁₂₃ : P * cexp (r₁₂₃ * β) = -Q₃) (n₁ : r₁ ≠ 0)
    (n₁₂₃ : r₁₂₃ ≠ 0) (c₁₂ : r₁₂ = 0 ↔ C₁₂) (c₂₃ : r₂₃ = 0 ↔ C₂₃) :
    P * simplex3 r₁ r₂ r₃ β = simplexW β P Q₁ Q₂ Q₃ r₁ r₂ r₃ r₁₂ r₂₃ r₁₂₃ C₁₂ C₂₃ := by
  subst e₁₂ e₂₃ e₁₂₃
  have e₁₂ : P * expInt (r₁ + r₂) β = if C₁₂ then P * β else (Q₂ - P) / (r₁ + r₂) := by
    by_cases h : C₁₂
    · rw [ite_eq_left h, expInt, ite_eq_left (c₁₂.mpr h)]
    · rw [ite_eq_right h, mul_expInt (Q := -Q₂) (by rw [h₁₂, neg_neg]) (mt c₁₂.mp h)]
      ring
  have e₂₃ : P * (if r₂ + r₃ = 0 then (β * cexp (r₁ * β) - expInt r₁ β) / r₁
      else (expInt (r₁ + r₂ + r₃) β - expInt r₁ β) / (r₂ + r₃)) =
      if C₂₃ then (-β * Q₁ - (-Q₁ - P) / r₁) / r₁
        else ((-Q₃ - P) / (r₁ + r₂ + r₃) - (-Q₁ - P) / r₁) / (r₂ + r₃) := by
    by_cases h : C₂₃
    · rw [ite_eq_left (c₂₃.mpr h), ite_eq_left h, ← mul_div_assoc, mul_sub, mul_expInt h₁ n₁,
        mul_left_comm, h₁]
      ring
    · rw [ite_eq_right (mt c₂₃.mp h), ite_eq_right h, ← mul_div_assoc, mul_sub,
        mul_expInt h₁ n₁, mul_expInt h₁₂₃ n₁₂₃]
  rw [simplex3, simplexW, ← mul_div_assoc, mul_sub, e₂₃, ← mul_div_assoc, mul_sub, e₁₂,
    mul_expInt h₁ n₁]

end HubbardAtom
