import HubbardAtomLean.ConverseCircles

/-!
# The converse of the spectral identity: eigenvectors are combinations of `g_s`

For the density and magnetic channels, `U ≠ 0`, `μ ≠ 0`, and a bounded `v` with `χ_r v = ℓv`
(`Params.eq_sum_gs_of_eigen`): if the roots of `Φ_k` are simple in `s = u²` (of order `≤ 2` in
`u` at `u = 0`) and finitely many of them, `s = ρ²` for `ρ ∈ S`, have `ℓ(s) = ℓ`, then
`vₙ = ∑_{ρ ∈ S} c_ρ g_{ρ²}(n)`.

The proof divides entire functions instead of integrating over contours. With
`Q(u) = ∏_{ρ ∈ S} (u² - ρ²)`:

* `E·Q = F·Φ_k` with `F` entire (`exists_differentiable_mul_eq`), as `E` vanishes at the other
  roots (`Ehat_eq_zero_of_Phi`) and every root is simple; `F` can be taken even;
* `F - P = K·Q` with `K` entire, where `P(u) = ∑_ρ c_ρ ∏_{σ ≠ ρ} (u² - σ²)` interpolates `F` at
  `S`;
* on the circles of `ConverseCircles.lean`, `K = E/Φ_k - ∑_ρ c_ρ/(u² - ρ²)` is `O(δ_j + R_j⁻²)`,
  so `K = 0` by the maximum modulus principle (`eq_zero_of_norm_le_sphere`);
* at `u = zₙ`, `E/Φ_k = -βvₙ/(2εa)` (`Ehat_z`).
-/

open Complex Filter Topology Metric

namespace HubbardAtom

/-- **Division of entire functions**: if `g` vanishes nowhere to higher order than `f`, then
`f = hg` with `h` entire. The quotient `f/g` is meromorphic of nonnegative order everywhere,
and its normal form is `h`. -/
theorem exists_differentiable_mul_eq {f g : ℂ → ℂ} (hf : Differentiable ℂ f)
    (hg : Differentiable ℂ g)
    (hord : ∀ x, g x = 0 → analyticOrderAt g x ≤ analyticOrderAt f x) :
    ∃ h : ℂ → ℂ, Differentiable ℂ h ∧ ∀ u, f u = h u * g u := by
  have hfa := hf.analyticAt
  have hga := hg.analyticAt
  have hq : MeromorphicOn (f / g) Set.univ := fun x _ ↦
    (hfa x).meromorphicAt.div (hga x).meromorphicAt
  set h := toMeromorphicNFOn (f / g) Set.univ
  have hnf := meromorphicNFOn_toMeromorphicNFOn (f / g) Set.univ
  have hnear : ∀ x, h =ᶠ[𝓝[≠] x] f / g := fun x ↦
    hq.toMeromorphicNFOn_eq_self_on_nhdsNE (Set.mem_univ x)
  -- `0 ≤ ord f - ord g`
  have hnonneg : ∀ x, 0 ≤ meromorphicOrderAt (f / g) x := fun x ↦ by
    rw [meromorphicOrderAt_div (hfa x).meromorphicAt (hga x).meromorphicAt,
      (hfa x).meromorphicOrderAt_eq, (hga x).meromorphicOrderAt_eq]
    by_cases hgx : g x = 0
    · have hle := hord x hgx
      cases hg' : analyticOrderAt g x with
      | top => simp
      | coe n =>
        cases hf' : analyticOrderAt f x with
        | top => simp
        | coe m =>
          rw [hg', hf'] at hle
          have hnm : (0 : ℤ) ≤ m - n := by
            have : n ≤ m := by exact_mod_cast hle
            omega
          rw [ENat.map_natCast, ENat.map_natCast]
          exact_mod_cast hnm
    · rw [analyticOrderAt_eq_zero.mpr (Or.inr hgx)]
      simp
  have han : ∀ x, AnalyticAt ℂ h x := fun x ↦
    (hnf (Set.mem_univ x)).meromorphicOrderAt_nonneg_iff_analyticAt.mp
      ((meromorphicOrderAt_toMeromorphicNFOn hq (Set.mem_univ x)).symm ▸ hnonneg x)
  refine ⟨h, fun x ↦ (han x).differentiableAt, fun x ↦ ?_⟩
  rcases (hga x).eventually_eq_zero_or_eventually_ne_zero with h0 | hne
  · -- `g = 0` near `x`, so `f = 0` near `x`
    have hg0 : analyticOrderAt g x = ⊤ := analyticOrderAt_eq_top.mpr h0
    have hf0 := hord x h0.self_of_nhds
    rw [hg0, top_le_iff] at hf0
    rw [(analyticOrderAt_eq_top.mp hf0).self_of_nhds, h0.self_of_nhds, mul_zero]
  · have hev : f =ᶠ[𝓝[≠] x] fun u ↦ h u * g u := by
      filter_upwards [hnear x, hne] with u hu hgu
      rw [hu, Pi.div_apply, div_mul_cancel₀ _ hgu]
    exact ((ContinuousAt.eventuallyEq_nhds_iff_eventuallyEq_nhdsNE (hfa x).continuousAt
      ((han x).continuousAt.mul (hga x).continuousAt)).mp hev).self_of_nhds

/-- An even analytic function vanishing at `0` vanishes there to order at least `2`. -/
theorem two_le_analyticOrderAt_of_even {f : ℂ → ℂ} (hf : Differentiable ℂ f)
    (heven : ∀ u, f (-u) = f u) (h0 : f 0 = 0) : 2 ≤ analyticOrderAt f 0 := by
  rw [show (2 : ℕ∞) = ((2 : ℕ) : ℕ∞) by rfl,
    natCast_le_analyticOrderAt_iff_iteratedDeriv_eq_zero (hf.analyticAt 0)]
  intro i hi
  interval_cases i
  · simpa using h0
  · have h := deriv_comp_neg f 0
    simp only [heven, neg_zero] at h
    simp only [iteratedDeriv_one]
    linear_combination h / 2

/-- The order bound used twice below: if the roots of `g` are simple in `u²` and an even `f`
vanishes at them, then `g` vanishes nowhere to higher order than `f`. -/
theorem analyticOrderAt_le_of_even {f g : ℂ → ℂ} (hf : Differentiable ℂ f)
    (heven : ∀ u, f (-u) = f u) (hg : ∀ x, g x = 0 → analyticOrderAt g x ≤ if x = 0 then 2 else 1)
    (hfg : ∀ x, g x = 0 → f x = 0) (x : ℂ) (hx : g x = 0) :
    analyticOrderAt g x ≤ analyticOrderAt f x := by
  refine (hg x hx).trans ?_
  split_ifs with h0
  · subst h0
    exact two_le_analyticOrderAt_of_even hf heven (hfg 0 hx)
  · exact Order.one_le_iff_ne_zero.mpr (analyticOrderAt_ne_zero.mpr ⟨hf.analyticAt x, hfg x hx⟩)

/-- **Maximum modulus on spheres**: an entire `K` with `|K| ≤ εⱼ` on the spheres `|u| = Rⱼ`,
`Rⱼ → ∞`, `εⱼ → 0`, vanishes. -/
theorem eq_zero_of_norm_le_sphere {K : ℂ → ℂ} (hK : Differentiable ℂ K) {R ε : ℕ → ℝ}
    (hR : Tendsto R atTop atTop) (hε : Tendsto ε atTop (𝓝 0))
    (hb : ∀ᶠ j in atTop, ∀ u : ℂ, ‖u‖ = R j → ‖K u‖ ≤ ε j) (u : ℂ) : K u = 0 := by
  have h : ∀ᶠ j in atTop, ‖K u‖ ≤ ε j := by
    filter_upwards [hb, hR.eventually_gt_atTop ‖u‖] with j hj hRj
    have hR0 : R j ≠ 0 := ((norm_nonneg u).trans_lt hRj).ne'
    refine Complex.norm_le_of_forall_mem_frontier_norm_le isBounded_ball hK.diffContOnCl
      (fun w hw ↦ ?_) (subset_closure (mem_ball_zero_iff.mpr hRj))
    rw [frontier_ball 0 hR0, mem_sphere_zero_iff_norm] at hw
    exact hj w hw
  exact norm_le_zero_iff.mp (ge_of_tendsto hε h)

/-- `Q(u) = ∏_{ρ ∈ S} (u² - ρ²)`. -/
noncomputable def prodSq (S : Finset ℂ) (u : ℂ) : ℂ := ∏ ρ ∈ S, (u ^ 2 - ρ ^ 2)

theorem differentiable_prodSq (S : Finset ℂ) : Differentiable ℂ (prodSq S) := by
  unfold prodSq
  fun_prop

theorem prodSq_neg (S : Finset ℂ) (u : ℂ) : prodSq S (-u) = prodSq S u := by
  simp [prodSq]

/-- `∏_{σ ∈ S∖ρ} (ρ² - σ²) ≠ 0` for distinct `σ²`. -/
theorem prod_erase_ne_zero {S : Finset ℂ} (hS : Set.InjOn (fun ρ : ℂ ↦ ρ ^ 2) S) {ρ x : ℂ}
    (hρ : ρ ∈ S) (hx : x ^ 2 = ρ ^ 2) : ∏ σ ∈ S.erase ρ, (x ^ 2 - σ ^ 2) ≠ 0 := by
  refine Finset.prod_ne_zero_iff.mpr fun σ hσ h ↦ ?_
  rw [hx, sub_eq_zero] at h
  exact Finset.ne_of_mem_erase hσ (hS (Finset.mem_of_mem_erase hσ) hρ h.symm)

/-- The roots of `Q` are simple in `u²`: `±ρ` of order `1` for `ρ ≠ 0`, and `0` of order `2`. -/
theorem analyticOrderAt_prodSq_le {S : Finset ℂ} (hS : Set.InjOn (fun ρ : ℂ ↦ ρ ^ 2) S) (x : ℂ)
    (hx : prodSq S x = 0) : analyticOrderAt (prodSq S) x ≤ if x = 0 then 2 else 1 := by
  obtain ⟨ρ, hρ, hxρ⟩ := Finset.prod_eq_zero_iff.mp hx
  rw [sub_eq_zero] at hxρ
  have hsplit : prodSq S = (fun u ↦ u - ρ) * (fun u ↦ u - -ρ) *
      fun u ↦ ∏ σ ∈ S.erase ρ, (u ^ 2 - σ ^ 2) := by
    funext u
    rw [prodSq, ← Finset.mul_prod_erase S _ hρ, Pi.mul_apply, Pi.mul_apply]
    ring
  have hr : analyticOrderAt (fun u ↦ ∏ σ ∈ S.erase ρ, (u ^ 2 - σ ^ 2)) x = 0 :=
    analyticOrderAt_eq_zero.mpr (Or.inr (prod_erase_ne_zero hS hρ hxρ))
  rw [hsplit, analyticOrderAt_mul (by fun_prop) (by fun_prop), hr, add_zero,
    analyticOrderAt_mul (by fun_prop) (by fun_prop)]
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp hxρ with rfl | rfl
  · by_cases h0 : x = 0
    · subst h0
      simp only [neg_zero, analyticOrderAt_id_sub_const_self, ite_true]
      rfl
    · rw [analyticOrderAt_id_sub_const_self, analyticOrderAt_id_sub_const_of_ne
        (fun h ↦ h0 (by linear_combination h / 2)), add_zero]
      simp [h0]
  · by_cases h0 : ρ = 0
    · subst h0
      simp only [neg_zero, analyticOrderAt_id_sub_const_self, ite_true]
      rfl
    · rw [analyticOrderAt_id_sub_const_of_ne (fun h ↦ h0 (by linear_combination -h / 2)),
        analyticOrderAt_id_sub_const_self, zero_add]
      simp [h0]

/-- `P(u)/Q(u) = ∑_ρ c_ρ/(u² - ρ²)` for `P(u) = ∑_ρ c_ρ ∏_{σ ≠ ρ} (u² - σ²)`. -/
theorem sum_div_prodSq {S : Finset ℂ} (c : ℂ → ℂ) {u : ℂ} (hu : ∀ ρ ∈ S, u ^ 2 - ρ ^ 2 ≠ 0) :
    (∑ ρ ∈ S, c ρ * ∏ σ ∈ S.erase ρ, (u ^ 2 - σ ^ 2)) / prodSq S u =
      ∑ ρ ∈ S, c ρ / (u ^ 2 - ρ ^ 2) := by
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun ρ hρ ↦ ?_
  have hr : ∏ σ ∈ S.erase ρ, (u ^ 2 - σ ^ 2) ≠ 0 :=
    Finset.prod_ne_zero_iff.mpr fun σ hσ ↦ hu σ (Finset.mem_of_mem_erase hσ)
  rw [prodSq, ← Finset.mul_prod_erase S _ hρ]
  field_simp [hu ρ hρ]

namespace Params

variable (A : Params) (k : ℤ)

theorem differentiable_Phik (ε : ℂ) : Differentiable ℂ (A.Phik k ε) := by
  unfold Phik Dk
  have := differentiable_shc A.β
  fun_prop

/-- **The converse of the spectral identity**: for `U ≠ 0`, `μ ≠ 0` and a bounded `v` with
`χ_r v = ℓv`, if the roots of `Φ_k` are simple in `s = u²` (of order `≤ 2` in `u` at `u = 0`)
and those with `ℓ(s) = ℓ` are `s = ρ²` for `ρ` in a finite set `S` (both assumed), then
`v = ∑_{ρ ∈ S} c_ρ g_{ρ²}`. -/
theorem eq_sum_gs_of_eigen {ε : ℂ} (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {v : ℤ → ℂ} {M : ℝ} (hv : ∀ n, ‖v n‖ ≤ M) {ℓ : ℂ}
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) (ℓ * v n))
    (hsimple : ∀ ρ, A.Phik k ε ρ = 0 →
      analyticOrderAt (A.Phik k ε) ρ ≤ if ρ = 0 then 2 else 1)
    {S : Finset ℂ} (hS : Set.InjOn (fun ρ : ℂ ↦ ρ ^ 2) S) (hSΦ : ∀ ρ ∈ S, A.Phik k ε ρ = 0)
    (hSℓ : ∀ ρ, A.Phik k ε ρ = 0 → ℓ = -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k ρ m →
      ∃ σ ∈ S, σ ^ 2 = ρ ^ 2) :
    ∃ c : ℂ → ℂ, ∀ n, v n = ∑ ρ ∈ S, c ρ * A.gs k ρ n := by
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  set E := A.Ehat k v
  set Φ := A.Phik k ε
  set Q := prodSq S
  have hE : Differentiable ℂ E := A.differentiable_Ehat hμ hv
  have hΦ : Differentiable ℂ Φ := A.differentiable_Phik k ε
  have hQ : Differentiable ℂ Q := differentiable_prodSq S
  -- 1. `E·Q = F₀·Φ_k`
  have hvan : ∀ x, Φ x = 0 → E x * Q x = 0 := fun x hx ↦ by
    by_cases hℓ : ℓ = -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k x m
    · obtain ⟨σ, hσ, hσx⟩ := hSℓ x hx hℓ
      rw [show Q x = 0 from Finset.prod_eq_zero hσ (by rw [hσx, sub_self]), mul_zero]
    · rw [show E x = 0 from A.Ehat_eq_zero_of_Phi hε ha hμ hv hχ hx hℓ, zero_mul]
  obtain ⟨F₀, hF₀, hEQ₀⟩ := exists_differentiable_mul_eq (hE.mul hQ) hΦ
    (analyticOrderAt_le_of_even (hE.mul hQ)
      (fun u ↦ by simp only [Pi.mul_apply, E, Q, A.Ehat_neg, prodSq_neg]) hsimple hvan)
  -- the even part `F` of `F₀` has `E·Q = F·Φ_k` too
  set F : ℂ → ℂ := fun u ↦ (F₀ u + F₀ (-u)) / 2
  have hF : Differentiable ℂ F := by
    have := hF₀.comp differentiable_neg
    simp only [Function.comp_def] at this
    fun_prop
  have hFe : ∀ u, F (-u) = F u := fun u ↦ by simp only [F, neg_neg, add_comm]
  have hEQ : ∀ u, E u * Q u = F u * Φ u := fun u ↦ by
    have h₁ : E u * Q u = F₀ u * Φ u := hEQ₀ u
    have h₂ : E (-u) * Q (-u) = F₀ (-u) * Φ (-u) := hEQ₀ (-u)
    rw [show E (-u) = E u from Ehat_neg v u, show Q (-u) = Q u from prodSq_neg S u,
      show Φ (-u) = Φ u from A.Phik_neg k ε u] at h₂
    simp only [F]
    linear_combination h₁ / 2 + h₂ / 2
  -- 2. `F - P = K·Q`, `P` interpolating `F` on `S`
  set c : ℂ → ℂ := fun ρ ↦ F ρ / ∏ σ ∈ S.erase ρ, (ρ ^ 2 - σ ^ 2)
  set P : ℂ → ℂ := fun u ↦ ∑ ρ ∈ S, c ρ * ∏ σ ∈ S.erase ρ, (u ^ 2 - σ ^ 2)
  have hP : Differentiable ℂ P := by
    simp only [P]
    fun_prop
  have hPe : ∀ u, P (-u) = P u := fun u ↦ by simp only [P, neg_sq]
  have hPρ : ∀ ρ ∈ S, P ρ = F ρ := fun ρ hρ ↦ by
    simp only [P]
    rw [Finset.sum_eq_single ρ (fun ρ' hρ' hne ↦ by
      rw [Finset.prod_eq_zero (f := fun σ ↦ ρ ^ 2 - σ ^ 2)
        (Finset.mem_erase.mpr ⟨Ne.symm hne, hρ⟩) (sub_self _), mul_zero])
      (fun h ↦ absurd hρ h)]
    exact div_mul_cancel₀ _ (prod_erase_ne_zero hS hρ rfl)
  have hH : Differentiable ℂ (fun u ↦ F u - P u) := hF.sub hP
  have hHvan : ∀ x, Q x = 0 → F x - P x = 0 := fun x hx ↦ by
    obtain ⟨ρ, hρ, hxρ⟩ := Finset.prod_eq_zero_iff.mp hx
    rw [sub_eq_zero] at hxρ
    rcases sq_eq_sq_iff_eq_or_eq_neg.mp hxρ with rfl | rfl
    · rw [hPρ x hρ, sub_self]
    · rw [hFe, hPe, hPρ ρ hρ, sub_self]
  obtain ⟨K, hK, hHK⟩ := exists_differentiable_mul_eq hH hQ
    (analyticOrderAt_le_of_even hH (fun u ↦ by simp only [hFe, hPe])
      (analyticOrderAt_prodSq_le hS) hHvan)
  -- 3. `K = 0` by the maximum modulus principle on the circles
  set C := ∑ ρ ∈ S, ‖c ρ‖
  have hR := A.tendsto_circR k
  have hR2 : Tendsto (fun j ↦ A.circR k j ^ 2) atTop atTop :=
    (tendsto_pow_atTop two_ne_zero).comp hR
  have hlim : Tendsto (fun j ↦ 2 * M * A.circδ k j + 2 * C / A.circR k j ^ 2) atTop (𝓝 0) := by
    have h := ((A.tendsto_circδ k).const_mul (2 * M)).add
      ((tendsto_const_nhds (x := 2 * C)).div_atTop hR2)
    simpa using h
  have hK0 : ∀ u, K u = 0 := eq_zero_of_norm_le_sphere hK hR hlim (by
      filter_upwards [A.norm_Ehat_le_circ k hμ ε, A.Phik_ne_zero_circ k ε,
        hR.eventually_ge_atTop 1, hR2.eventually_ge_atTop (2 * ∑ ρ ∈ S, ‖ρ‖ ^ 2)]
        with j hEj hΦj hR1 hRS u hu
      have hR2' : ∀ ρ ∈ S, 2 * ‖ρ‖ ^ 2 ≤ A.circR k j ^ 2 := fun ρ hρ ↦
        (mul_le_mul_of_nonneg_left (Finset.single_le_sum (f := fun ρ ↦ ‖ρ‖ ^ 2)
          (fun _ _ ↦ by positivity) hρ) zero_le_two).trans hRS
      have hfac : ∀ ρ ∈ S, A.circR k j ^ 2 / 2 ≤ ‖u ^ 2 - ρ ^ 2‖ := fun ρ hρ ↦ by
        have := norm_sub_norm_le (u ^ 2) (ρ ^ 2)
        rw [norm_pow, norm_pow, hu] at this
        linarith [hR2' ρ hρ]
      have hRpos : 0 < A.circR k j ^ 2 / 2 := by positivity
      have hfac0 : ∀ ρ ∈ S, u ^ 2 - ρ ^ 2 ≠ 0 := fun ρ hρ h ↦ by
        have := hfac ρ hρ
        rw [h, norm_zero] at this
        linarith
      have hQu : Q u ≠ 0 := Finset.prod_ne_zero_iff.mpr hfac0
      have hΦu := hΦj u hu
      have hKu : K u = E u / Φ u - ∑ ρ ∈ S, c ρ / (u ^ 2 - ρ ^ 2) := by
        have hPQ : P u / Q u = ∑ ρ ∈ S, c ρ / (u ^ 2 - ρ ^ 2) := sum_div_prodSq c hfac0
        have hFQ : F u / Q u = E u / Φ u := by
          rw [div_eq_div_iff hQu hΦu]
          exact (hEQ u).symm
        rw [← hPQ, ← hFQ, ← sub_div, hHK u, mul_div_cancel_right₀ _ hQu]
      have hEu : ‖E u / Φ u‖ ≤ 2 * M * A.circδ k j := by
        rw [norm_div, div_le_iff₀ (norm_pos_iff.mpr hΦu)]
        exact hEj v M hv u hu
      have hsum : ‖∑ ρ ∈ S, c ρ / (u ^ 2 - ρ ^ 2)‖ ≤ 2 * C / A.circR k j ^ 2 := by
        calc ‖∑ ρ ∈ S, c ρ / (u ^ 2 - ρ ^ 2)‖ ≤ ∑ ρ ∈ S, ‖c ρ‖ * (2 / A.circR k j ^ 2) :=
              (norm_sum_le _ _).trans (Finset.sum_le_sum fun ρ hρ ↦ by
                rw [norm_div]
                refine (div_le_div_of_nonneg_left (norm_nonneg _) hRpos (hfac ρ hρ)).trans_eq ?_
                rw [div_div_eq_mul_div]
                ring)
          _ = 2 * C / A.circR k j ^ 2 := by
              rw [← Finset.sum_mul]
              ring
      rw [hKu]
      exact (norm_sub_le _ _).trans (add_le_add hEu hsum))
  -- 4. at `u = zₙ`, `E/Φ_k = -βvₙ/(2εa)`
  refine ⟨fun ρ ↦ -(2 * ε * A.a / A.β) * c ρ, fun n ↦ ?_⟩
  have hz := A.z_ne_zero k hμ n
  have hΦz := A.Phik_z_ne_zero k ε hε0 ha hμ n
  have hfz : ∀ ρ ∈ S, A.z k n ^ 2 - ρ ^ 2 ≠ 0 := fun ρ hρ h ↦ by
    rcases sq_eq_sq_iff_eq_or_eq_neg.mp (sub_eq_zero.mp h) with h' | h'
    · exact hΦz (h' ▸ hSΦ ρ hρ)
    · exact hΦz (by rw [h', A.Phik_neg k]; exact hSΦ ρ hρ)
  have hQz : Q (A.z k n) ≠ 0 := Finset.prod_ne_zero_iff.mpr hfz
  have hFP : F (A.z k n) = P (A.z k n) := by
    have := hHK (A.z k n)
    rw [hK0, zero_mul, sub_eq_zero] at this
    exact this
  have hid := hEQ (A.z k n)
  have hΦz' : Φ (A.z k n) = ε * A.a * sinh (A.β * A.z k n) / A.z k n := by
    simp only [Φ]
    rw [A.Phik_of_ne k ε hz, A.Dk_z k, zero_add]
  have hs : sinh (A.β * A.z k n) ≠ 0 := fun h ↦ hΦz (by
    change Φ (A.z k n) = 0
    rw [hΦz', h, mul_zero, zero_div])
  simp only [E] at hid
  rw [Ehat_z hμ v n, hFP, hΦz'] at hid
  have hPQ : P (A.z k n) / Q (A.z k n) = ∑ ρ ∈ S, c ρ / (A.z k n ^ 2 - ρ ^ 2) :=
    sum_div_prodSq c hfz
  have hsum : ∑ ρ ∈ S, -(2 * ε * A.a / A.β) * c ρ * A.gs k ρ n =
      -(2 * ε * A.a / A.β) * (P (A.z k n) / Q (A.z k n)) := by
    rw [hPQ, Finset.mul_sum]
    refine Finset.sum_congr rfl fun ρ _ ↦ ?_
    rw [gs]
    ring
  rw [hsum]
  field_simp at hid ⊢
  linear_combination -hid

end Params

end HubbardAtom
