import Mathlib

/-!
# Products over the roots of a polynomial with analytic coefficients

Not in the note.
-/

open Complex Filter Topology Polynomial Finset Metric

namespace HubbardAtom

/-! ### Newton's identities on a multiset -/

/-- **Newton's identities** for the elements of a multiset `s`:
`k e_k = (-1)^{k+1} ∑_{i + j = k, i < k} (-1)^i e_i p_j`, with `p_j = ∑_{x ∈ s} x^j`. -/
theorem multiset_mul_esymm_eq_sum (s : Multiset ℂ) (k : ℕ) :
    k * s.esymm k = (-1) ^ (k + 1) *
      ∑ a ∈ antidiagonal k with a.1 < k, (-1) ^ a.1 * s.esymm a.1 * (s.map (· ^ a.2)).sum := by
  obtain ⟨l, rfl⟩ : ∃ l : List ℂ, (l : Multiset ℂ) = s := ⟨s.toList, Multiset.coe_toList s⟩
  have h := congr_arg (MvPolynomial.aeval l.get) (MvPolynomial.mul_esymm_eq_sum (Fin l.length) ℂ k)
  have hl : (univ.val.map l.get : Multiset ℂ) = (l : Multiset ℂ) := by
    rw [Fin.univ_val_map, List.ofFn_get]
  simp only [map_mul, map_natCast, map_pow, map_neg, map_one, map_sum,
    MvPolynomial.aeval_esymm_eq_multiset_esymm, hl, MvPolynomial.psum, MvPolynomial.aeval_X] at h
  convert h using 4 with a
  rw [← hl, Multiset.map_map, Finset.sum_map_val]
  rfl

/-- If the power sums of a family of multisets are analytic at `w₀`, so are its elementary
symmetric functions, by Newton's identities. -/
theorem analyticAt_esymm_of_psum {V : ℂ → Multiset ℂ} {w₀ : ℂ}
    (h : ∀ j, AnalyticAt ℂ (fun w ↦ ((V w).map (· ^ j)).sum) w₀) (k : ℕ) :
    AnalyticAt ℂ (fun w ↦ (V w).esymm k) w₀ := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    rcases k.eq_zero_or_pos with rfl | hk
    · simpa using analyticAt_const
    have e : (fun w ↦ (V w).esymm k) = fun w ↦ (k : ℂ)⁻¹ * ((-1) ^ (k + 1) *
        ∑ a ∈ antidiagonal k with a.1 < k,
          (-1) ^ a.1 * (V w).esymm a.1 * ((V w).map (· ^ a.2)).sum) := by
      funext w
      rw [← multiset_mul_esymm_eq_sum, ← mul_assoc, inv_mul_cancel₀ (by exact_mod_cast hk.ne'),
        one_mul]
    rw [e]
    refine analyticAt_const.mul (analyticAt_const.mul ?_)
    have := Finset.analyticAt_sum (𝕜 := ℂ) (c := w₀) (filter (fun a ↦ a.1 < k) (antidiagonal k))
      (f := fun a w ↦ (-1) ^ a.1 * (V w).esymm a.1 * ((V w).map (· ^ a.2)).sum) fun a ha ↦ ?_
    · convert this using 1
      ext w
      simp
    exact (analyticAt_const.mul (ih a.1 (mem_filter.1 ha).2)).mul (h a.2)

/-- If the power sums of a family of multisets of constant cardinality are analytic at `w₀`, so is
the product of its elements. -/
theorem analyticAt_prod_of_psum {V : ℂ → Multiset ℂ} {w₀ : ℂ} {n : ℕ}
    (hn : ∀ w, (V w).card = n) (h : ∀ j, AnalyticAt ℂ (fun w ↦ ((V w).map (· ^ j)).sum) w₀) :
    AnalyticAt ℂ (fun w ↦ (V w).prod) w₀ := by
  have e : (fun w ↦ (V w).prod) = fun w ↦ (V w).esymm n := by
    funext w
    rw [← hn w, Multiset.esymm, Multiset.powersetCard_self]
    simp
  rw [e]
  exact analyticAt_esymm_of_psum h n

/-! ### The power sums over the roots as contour integrals -/

/-- The logarithmic derivative of `∏_{r ∈ s} (X - r)` at `x`, off the `r`: `∑_{r ∈ s} (x - r)⁻¹`. -/
theorem eval_derivative_div_eval_prod (s : Multiset ℂ) {x : ℂ}
    (hx : ((s.map fun r ↦ X - C r).prod).eval x ≠ 0) :
    ((s.map fun r ↦ X - C r).prod).derivative.eval x / ((s.map fun r ↦ X - C r).prod).eval x =
      (s.map fun r ↦ (x - r)⁻¹).sum := by
  induction s using Multiset.induction_on with
  | empty => simp
  | cons a s ih =>
    simp only [Multiset.map_cons, Multiset.prod_cons, Multiset.sum_cons, derivative_mul,
      derivative_sub, derivative_X, derivative_C, sub_zero, one_mul, eval_add, eval_mul, eval_sub,
      eval_X, eval_C] at hx ⊢
    have ha : x - a ≠ 0 := left_ne_zero_of_mul hx
    have hs := right_ne_zero_of_mul hx
    rw [← ih hs]
    field_simp

/-- The logarithmic derivative of a monic polynomial over `ℂ` at `x`, off its roots, is
`∑_r (x - r)⁻¹` over its roots counted with multiplicity. -/
theorem eval_derivative_div_eval {p : ℂ[X]} (hp : p.Monic) {x : ℂ} (hx : p.eval x ≠ 0) :
    p.derivative.eval x / p.eval x = (p.roots.map fun r ↦ (x - r)⁻¹).sum := by
  have e := prod_multiset_X_sub_C_of_monic_of_roots_card_eq hp IsAlgClosed.card_roots_eq_natDegree
  have h := eval_derivative_div_eval_prod p.roots (x := x) (by rwa [e])
  rwa [e] at h

/-- **A root bound**: a monic polynomial of degree `n` whose other coefficients have norm at most
`B` has no roots with `n B + 1 ≤ ‖x‖`. -/
theorem eval_ne_zero_of_coeff_le {p : ℂ[X]} (hp : p.Monic) {B : ℝ}
    (hB : ∀ i < p.natDegree, ‖p.coeff i‖ ≤ B) {x : ℂ} (hx : p.natDegree * B + 1 ≤ ‖x‖) :
    p.eval x ≠ 0 := by
  set n := p.natDegree with hn
  rcases Nat.eq_zero_or_pos n with h0 | hpos
  · rw [(Polynomial.Monic.natDegree_eq_zero hp).1 (hn ▸ h0)]
    simp
  have hB0 : 0 ≤ B := (norm_nonneg _).trans (hB 0 hpos)
  have h1 : 1 ≤ ‖x‖ := le_trans (by nlinarith [show (0 : ℝ) ≤ n by positivity]) hx
  rw [eval_eq_sum_range, ← hn, Finset.sum_range_succ, hp.coeff_natDegree, one_mul]
  intro h0
  have hlt : ‖∑ i ∈ range n, p.coeff i * x ^ i‖ < ‖x ^ n‖ := by
    calc ‖∑ i ∈ range n, p.coeff i * x ^ i‖
        ≤ ∑ i ∈ range n, B * ‖x‖ ^ (n - 1) := by
          refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun i hi ↦ ?_)
          rw [norm_mul, norm_pow]
          exact mul_le_mul (hB i (Finset.mem_range.1 hi))
            (pow_le_pow_right₀ h1 (by have := Finset.mem_range.1 hi; omega)) (by positivity) hB0
      _ = n * B * ‖x‖ ^ (n - 1) := by simp [mul_assoc]
      _ < ‖x‖ * ‖x‖ ^ (n - 1) := by
          refine mul_lt_mul_of_pos_right (by linarith) (by positivity)
      _ = ‖x ^ n‖ := by
          rw [norm_pow, ← pow_succ']
          congr 1
          omega
  rw [add_eq_zero_iff_eq_neg] at h0
  rw [h0, norm_neg] at hlt
  exact lt_irrefl _ hlt

/-- A contour integral of a finite sum (over a multiset) of circle-integrable functions is the sum
of the contour integrals. -/
theorem circleIntegral_multiset_sum {ι : Type*} (s : Multiset ι) (F : ι → ℂ → ℂ) {c : ℂ} {R : ℝ}
    (h : ∀ i ∈ s, CircleIntegrable (F i) c R) :
    CircleIntegrable (fun x ↦ (s.map fun i ↦ F i x).sum) c R ∧
      (∮ x in C(c, R), (s.map fun i ↦ F i x).sum) = (s.map fun i ↦ ∮ x in C(c, R), F i x).sum := by
  induction s using Multiset.induction_on with
  | empty => simp [CircleIntegrable, circleIntegral]
  | cons a s ih =>
    have ha := h a (Multiset.mem_cons_self a s)
    obtain ⟨hi, he⟩ := ih fun i hi ↦ h i (Multiset.mem_cons_of_mem hi)
    simp only [Multiset.map_cons, Multiset.sum_cons]
    exact ⟨ha.add hi, by rw [circleIntegral.integral_add ha hi, he]⟩

/-- **The power sums over the roots as contour integrals**: for a monic polynomial `p` over `ℂ`
whose roots lie in the open disc of radius `R`, and an entire `f`,
`∮_{|x| = R} f(x) p'(x)/p(x) dx = 2πi ∑_r f(r)` over the roots counted with multiplicity. -/
theorem circleIntegral_mul_logDeriv {p : ℂ[X]} (hp : p.Monic) {R : ℝ} (hR : 0 ≤ R)
    (hr : ∀ r ∈ p.roots, ‖r‖ < R) {f : ℂ → ℂ} (hf : Differentiable ℂ f) :
    (∮ x in C(0, R), f x * (p.derivative.eval x / p.eval x)) =
      2 * Real.pi * I * (p.roots.map f).sum := by
  rw [circleIntegral.integral_congr hR (g := fun x ↦ (p.roots.map fun r ↦ (x - r)⁻¹ • f x).sum)]
  · have hi : ∀ r ∈ p.roots, CircleIntegrable (fun x ↦ (x - r)⁻¹ • f x) 0 R := fun r hr' ↦ by
      refine ContinuousOn.circleIntegrable hR ?_
      refine ContinuousOn.smul (ContinuousOn.inv₀ (continuousOn_id.sub continuousOn_const)
        fun x hx ↦ ?_) hf.continuous.continuousOn
      simp only [Pi.sub_apply, id, sub_ne_zero]
      rintro rfl
      exact lt_irrefl _ ((mem_sphere_zero_iff_norm.1 hx) ▸ hr x hr')
    rw [(circleIntegral_multiset_sum _ (fun r x ↦ (x - r)⁻¹ • f x) hi).2,
      ← Multiset.sum_map_mul_left]
    refine congr_arg _ (Multiset.map_congr rfl fun r hr' ↦ ?_)
    rw [hf.differentiableOn.circleIntegral_sub_inv_smul (mem_ball_zero_iff.2 (hr r hr')),
      smul_eq_mul]
  · intro x hx
    have hx0 : p.eval x ≠ 0 := fun h ↦ by
      have := hr x ((mem_roots hp.ne_zero).2 h)
      rw [mem_sphere_zero_iff_norm.1 hx] at this
      exact lt_irrefl _ this
    simp only [eval_derivative_div_eval hp hx0, ← Multiset.sum_map_mul_left, smul_eq_mul]
    exact congr_arg _ (Multiset.map_congr rfl fun r _ ↦ mul_comm _ _)

/-! ### Analyticity in a parameter -/

/-- **A contour integral analytic in a parameter**: if `(w, x) ↦ h w x` is analytic at every
`(w, x)` with `w` near `w₀` and `x` on the circle, then `w ↦ ∮_{|x - c| = R} h w x dx` is analytic
at `w₀`, by differentiation under the integral; the partial derivative in `w` is bounded on a
compact neighborhood of `{w₀} × circle`. -/
theorem analyticAt_circleIntegral {h : ℂ → ℂ → ℂ} {c w₀ : ℂ} {R : ℝ}
    (hh : ∀ᶠ w in 𝓝 w₀, ∀ x ∈ sphere c |R|, AnalyticAt ℂ (Function.uncurry h) (w, x)) :
    AnalyticAt ℂ (fun w ↦ ∮ x in C(c, R), h w x) w₀ := by
  obtain ⟨ε, hε, hball⟩ := Metric.eventually_nhds_iff_ball.1 hh
  set S := {z : ℂ × ℂ | AnalyticAt ℂ (Function.uncurry h) z}
  have hS : IsOpen S := isOpen_analyticAt ℂ _
  have hA : AnalyticOnNhd ℂ (Function.uncurry h) S := fun z hz ↦ hz
  -- the partial derivative in `w`
  set D : ℂ × ℂ → ℂ := fun z ↦ fderiv ℂ (Function.uncurry h) z (1, 0)
  have hD : ContinuousOn D S := hA.fderiv.continuousOn.clm_apply continuousOn_const
  have hε' : 0 < ε / 2 := half_pos hε
  have hmem : ∀ w ∈ closedBall w₀ (ε / 2), ∀ θ : ℝ, (w, circleMap c R θ) ∈ S := fun w hw θ ↦
    hball w (closedBall_subset_ball (half_lt_self hε) hw) _ (circleMap_mem_sphere' c R θ)
  have hK : closedBall w₀ (ε / 2) ×ˢ sphere c |R| ⊆ S := fun z hz ↦
    hball z.1 (closedBall_subset_ball (half_lt_self hε) hz.1) z.2 hz.2
  obtain ⟨C, hC⟩ := ((isCompact_closedBall w₀ (ε / 2)).prod
    (isCompact_sphere c |R|)).exists_bound_of_continuousOn (hD.mono hK)
  have hdc : Continuous fun θ ↦ deriv (circleMap c R) θ := by
    simp only [deriv_circleMap]
    fun_prop
  have hcont : ∀ w ∈ closedBall w₀ (ε / 2),
      Continuous fun θ ↦ deriv (circleMap c R) θ • h w (circleMap c R θ) := fun w hw ↦
    hdc.smul (continuous_iff_continuousAt.2 fun θ ↦ (hA _ (hmem w hw θ)).continuousAt.comp
      (f := fun θ ↦ (w, circleMap c R θ))
      (continuous_const.prodMk (continuous_circleMap c R)).continuousAt)
  have hderiv : ∀ w ∈ closedBall w₀ (ε / 2), ∀ θ : ℝ,
      HasDerivAt (fun w ↦ h w (circleMap c R θ)) (D (w, circleMap c R θ)) w := fun w hw θ ↦
    ((hA _ (hmem w hw θ)).differentiableAt.hasFDerivAt).comp_hasDerivAt
      (f := fun w ↦ (w, circleMap c R θ)) w ((hasDerivAt_id w).prodMk (hasDerivAt_const w _))
  refine DifferentiableOn.analyticAt (s := ball w₀ (ε / 2)) (fun w₁ hw₁ ↦ ?_)
    (ball_mem_nhds w₀ hε')
  have hw₁' := ball_subset_closedBall hw₁
  have H := (intervalIntegral.hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (a := 0) (b := 2 * Real.pi) (μ := MeasureTheory.volume) (s := ball w₀ (ε / 2))
    (bound := fun _ ↦ |R| * C)
    (F := fun w θ ↦ deriv (circleMap c R) θ • h w (circleMap c R θ))
    (F' := fun w θ ↦ deriv (circleMap c R) θ • D (w, circleMap c R θ))
    (isOpen_ball.mem_nhds hw₁) ?_ ((hcont w₁ hw₁').intervalIntegrable _ _) ?_ ?_
    intervalIntegrable_const ?_).2
  · exact H.differentiableAt.differentiableWithinAt
  · filter_upwards [isOpen_ball.mem_nhds hw₁] with w hw
    exact (hcont w (ball_subset_closedBall hw)).aestronglyMeasurable
  · refine (hdc.smul (continuous_iff_continuousAt.2 fun θ ↦ ?_)).aestronglyMeasurable
    exact (hD.continuousAt (hS.mem_nhds (hmem w₁ hw₁' θ))).comp (f := fun θ ↦ (w₁, circleMap c R θ))
      (continuous_const.prodMk (continuous_circleMap c R)).continuousAt
  · refine Eventually.of_forall fun θ _ w hw ↦ ?_
    rw [norm_smul, deriv_circleMap, norm_mul, norm_circleMap_zero, norm_I, mul_one]
    exact mul_le_mul_of_nonneg_left
      (hC _ ⟨ball_subset_closedBall hw, circleMap_mem_sphere' c R θ⟩) (abs_nonneg R)
  · exact Eventually.of_forall fun θ _ w hw ↦ by
      simpa only [smul_eq_mul] using
        (hderiv w (ball_subset_closedBall hw) θ).const_mul (deriv (circleMap c R) θ)

/-! ### Products over the roots -/

/-- A finite sum `∑ᵢ cᵢ₊ₘ(w) aᵢ xⁱ` of coefficients of `p w` is analytic in `(w, x)` where the
coefficients are analytic in `w`. -/
theorem analyticAt_sum_coeff_mul_pow {p : ℂ → ℂ[X]} {N m : ℕ} (a : ℕ → ℂ) {w x : ℂ}
    (ha : ∀ i ∈ range N, AnalyticAt ℂ (fun w ↦ (p w).coeff (i + m)) w) :
    AnalyticAt ℂ (fun z : ℂ × ℂ ↦ ∑ i ∈ range N, (p z.1).coeff (i + m) * a i * z.2 ^ i) (w, x) := by
  have := Finset.analyticAt_sum (𝕜 := ℂ) (c := (w, x)) (range N)
    (f := fun i (z : ℂ × ℂ) ↦ (p z.1).coeff (i + m) * a i * z.2 ^ i) fun i hi ↦
      ((AnalyticAt.comp (g := fun w ↦ (p w).coeff (i + m)) (f := fun z : ℂ × ℂ ↦ z.1)
        (x := (w, x)) (ha i hi) analyticAt_fst).mul analyticAt_const).mul (analyticAt_snd.pow i)
  convert this using 1
  ext z
  simp

/-- **A product over the roots, analytic in a parameter**: for monic polynomials `p w` of degree
`n` whose coefficients are analytic at `w₀`, and `g` analytic at every `(w, x)` with `w` near `w₀`,
`w ↦ ∏_r g w r` over the roots of `p w` counted with multiplicity is analytic at `w₀`. The power
sums of the `g w r` are contour integrals (`circleIntegral_mul_logDeriv`) over a circle that
encloses the roots for all `w` near `w₀` (`eval_ne_zero_of_coeff_le`), analytic in `w`
(`analyticAt_circleIntegral`), and the product follows by Newton's identities
(`analyticAt_prod_of_psum`). -/
theorem analyticAt_prod_roots {p : ℂ → ℂ[X]} {g : ℂ → ℂ → ℂ} {w₀ : ℂ} {n : ℕ}
    (hm : ∀ w, (p w).Monic) (hn : ∀ w, (p w).natDegree = n)
    (hc : ∀ i, AnalyticAt ℂ (fun w ↦ (p w).coeff i) w₀)
    (hg : ∀ᶠ w in 𝓝 w₀, ∀ x, AnalyticAt ℂ (Function.uncurry g) (w, x)) :
    AnalyticAt ℂ (fun w ↦ ((p w).roots.map (g w)).prod) w₀ := by
  refine analyticAt_prod_of_psum (n := n)
    (fun w ↦ by rw [Multiset.card_map, IsAlgClosed.card_roots_eq_natDegree, hn]) fun j ↦ ?_
  -- a bound on the coefficients near `w₀`, so a circle of radius `R` that encloses the roots
  obtain ⟨B, hB0, hB⟩ : ∃ B : ℝ, 0 ≤ B ∧ ∀ᶠ w in 𝓝 w₀, ∀ i < n, ‖(p w).coeff i‖ ≤ B := by
    refine ⟨∑ i ∈ range n, (‖(p w₀).coeff i‖ + 1), sum_nonneg fun _ _ ↦ by positivity, ?_⟩
    have h1 : ∀ i, ∀ᶠ w in 𝓝 w₀, ‖(p w).coeff i‖ < ‖(p w₀).coeff i‖ + 1 := fun i ↦
      (hc i).continuousAt.norm.eventually (gt_mem_nhds (lt_add_one _))
    filter_upwards [(eventually_all_finset (range n)).2 fun i _ ↦ h1 i] with w hw i hi
    exact (hw i (mem_range.2 hi)).le.trans
      (single_le_sum (f := fun i ↦ ‖(p w₀).coeff i‖ + 1) (fun _ _ ↦ by positivity) (mem_range.2 hi))
  obtain ⟨R, hRdef⟩ : ∃ R : ℝ, R = n * B + 1 := ⟨_, rfl⟩
  have hR : 0 ≤ R := by rw [hRdef]; positivity
  have hev : ∀ᶠ w in 𝓝 w₀, ∀ x, R ≤ ‖x‖ → (p w).eval x ≠ 0 := by
    filter_upwards [hB] with w hw x hx
    exact eval_ne_zero_of_coeff_le (hm w) (by rwa [hn]) (by rwa [hn, ← hRdef])
  -- `p` and `p'` as sums, analytic in `(w, x)`
  have hpe : ∀ w x, (p w).eval x = ∑ i ∈ range (n + 1), (p w).coeff i * x ^ i := fun w x ↦
    eval_eq_sum_range' (by rw [hn]; omega) x
  have hde : ∀ w x, (p w).derivative.eval x =
      ∑ i ∈ range (n + 1), (p w).coeff (i + 1) * (i + 1) * x ^ i := fun w x ↦ by
    rw [eval_eq_sum_range' (n := n + 1) ((natDegree_derivative_le _).trans_lt (by rw [hn]; omega))]
    simp [coeff_derivative]
  have hca : ∀ᶠ w in 𝓝 w₀, ∀ i ∈ range (n + 2), AnalyticAt ℂ (fun w ↦ (p w).coeff i) w :=
    (eventually_all_finset _).2 fun i _ ↦ (hc i).eventually_analyticAt
  set H : ℂ → ℂ → ℂ := fun w x ↦ g w x ^ j * ((p w).derivative.eval x / (p w).eval x)
  have he : (fun w ↦ (((p w).roots.map (g w)).map (· ^ j)).sum) =ᶠ[𝓝 w₀]
      fun w ↦ (2 * Real.pi * I)⁻¹ * ∮ x in C(0, R), H w x := by
    filter_upwards [hev, hg] with w hw hgw
    have hd : Differentiable ℂ fun x ↦ g w x ^ j := fun x ↦
      (((hgw x).comp (f := fun x ↦ (w, x)) (by fun_prop)).differentiableAt).pow j
    rw [circleIntegral_mul_logDeriv (hm w) hR (fun r hr ↦ ?_) hd, Multiset.map_map, ← mul_assoc,
      inv_mul_cancel₀ (by simp [Real.pi_ne_zero, I_ne_zero]), one_mul]
    · rfl
    · by_contra hle
      exact hw r (not_lt.1 hle) ((mem_roots (hm w).ne_zero).1 hr)
  refine (analyticAt_const.mul (analyticAt_circleIntegral ?_)).congr he.symm
  filter_upwards [hg, hev, hca] with w hgw hw hcw x hx
  have hP := analyticAt_sum_coeff_mul_pow (N := n + 1) (x := x) (fun _ ↦ 1) (m := 0)
    fun i hi ↦ hcw i (mem_range.2 (by have := mem_range.1 hi; omega))
  have hD := analyticAt_sum_coeff_mul_pow (N := n + 1) (x := x) (fun i ↦ (i : ℂ) + 1) (m := 1)
    fun i hi ↦ hcw (i + 1) (mem_range.2 (by have := mem_range.1 hi; omega))
  simp only [add_zero, mul_one] at hP
  have hx0 : (p w).eval x ≠ 0 := hw x (by rw [mem_sphere_zero_iff_norm.1 hx]; exact le_abs_self R)
  have e : Function.uncurry H = fun z : ℂ × ℂ ↦ Function.uncurry g z ^ j *
      ((∑ i ∈ range (n + 1), (p z.1).coeff (i + 1) * (i + 1) * z.2 ^ i) /
        ∑ i ∈ range (n + 1), (p z.1).coeff i * z.2 ^ i) := by
    ext z
    simp only [Function.uncurry, H, hpe, hde]
  rw [e]
  refine ((hgw x).pow j).mul (hD.div hP ?_)
  rwa [← hpe]

end HubbardAtom
