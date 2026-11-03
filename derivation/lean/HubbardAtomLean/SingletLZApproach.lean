import HubbardAtomLean.SingletNull
import HubbardAtomLean.LZLimits

/-!
# The singlet lattice zeros are approached in `μ`

`tendsto_gammaS_of_SingletLZ` (`LZLimits.lean`) gives `gammaS → gammaSlz` along any approach to
a lattice zero `t_c + Aₛ = 0` through points where the assumptions of Theorem "singlet" hold. Here
these points are shown to exist: at fixed `U`, `β`, `k`, the assumptions hold at every `μ ≠ μ₀`
near `μ₀`, so item 1 of Proposition "special" holds as a limit in `μ`.

* `Params.eventually_t_add_As_ne`: for every `μ₀`, `tₙ + Aₛ ≠ 0` for all `n` at every `μ ≠ μ₀`
  near `μ₀`. Each `tₙ + Aₛ` is analytic in `μ` (`analyticAt_t_add_As`) and nonzero for large
  `μ`, as `Re(tₙ + Aₛ) ≥ μ² + (νₙ + ω_k/2)² - ω_k²/4 - 2|a||μ| - 3a²` (`re_t_add_As_ge`), so its
  zeros are isolated; for all but finitely many `n` the same bound excludes zeros near `μ₀`;
* `Params.SingletLZ.eventually_singletHyp`: near a lattice zero, `Kₛ ≠ 0` as well, since `1 + 𝒦`
  stays invertible (`SingletLZ.eventually_isUnit`, by `ContHyp.eventually_isUnit`) and
  `𝒫(-Aₛ) = 𝒫(t_c) ≠ 0` there, so `isUnit_onePlusKS_iff_Ks` applies;
* `Params.SingletLZ.tendsto_gammaS`: **`gammaS → gammaSlz` as `μ → μ₀`, `μ ≠ μ₀`**, at fixed `U`,
  `β`, `k`, with `ξ = xiMu`.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ)

/-- `A` with the chemical potential `μ`. -/
def withMu (μ : ℝ) : Params := ⟨A.U, A.β, μ, A.β_pos⟩

@[simp] theorem withMu_μ (μ : ℝ) : (A.withMu μ).μ = μ := rfl
@[simp] theorem withMu_a (μ : ℝ) : (A.withMu μ).a = A.a := rfl
@[simp] theorem withMu_ν (μ : ℝ) (n : ℤ) : (A.withMu μ).ν n = A.ν n := rfl
@[simp] theorem withMu_ω (μ : ℝ) : (A.withMu μ).ω k = A.ω k := rfl

/-- `Re(tₙ + Aₛ) = μ² + νₙ(νₙ + ω_k) + 2aρμ + a²(1 - 8p₁)`. -/
theorem re_t_add_As (n : ℤ) : (A.t k n + A.As k).re =
    A.μ ^ 2 + A.ν n * (A.ν n + A.ω k) + 2 * A.a * A.ρ * A.μ + A.a ^ 2 * (1 - 8 * A.p₁) := by
  simp [t, x, b, w, As, ← ofReal_pow]
  ring

/-- A lower bound on `Re(tₙ + Aₛ)`, uniform in the weights (`|ρ| ≤ 1`, `p₁ ≤ 1/2`). -/
theorem re_t_add_As_ge (n : ℤ) : A.μ ^ 2 + (A.ν n + A.ω k / 2) ^ 2 - A.ω k ^ 2 / 4
    - 2 * |A.a| * |A.μ| - 3 * A.a ^ 2 ≤ (A.t k n + A.As k).re := by
  rw [re_t_add_As]
  have hs := A.weights_sum
  have h0 := A.p₀_pos
  have h1 := A.p₁_pos
  have h2 := A.p₂_pos
  have hρ : |A.ρ| ≤ 1 := by rw [ρ, abs_le]; constructor <;> linarith
  have hm : |A.a * A.ρ * A.μ| ≤ |A.a| * |A.μ| := by
    rw [abs_mul, abs_mul]
    exact mul_le_mul_of_nonneg_right (mul_le_of_le_one_right (abs_nonneg _) hρ) (abs_nonneg _)
  have := neg_abs_le (A.a * A.ρ * A.μ)
  nlinarith [sq_nonneg A.a, sq_nonneg A.μ]

/-- A real analytic function, as a complex one. -/
theorem analyticAt_ofReal_comp {f : ℝ → ℝ} {x : ℝ} (hf : AnalyticAt ℝ f x) :
    AnalyticAt ℝ (fun y ↦ (f y : ℂ)) x :=
  (Complex.ofRealCLM.analyticAt _).comp hf

/-- `p₁` and `ρ` are analytic in `μ`. -/
theorem analyticAt_p₁_ρ (μ₀ : ℝ) :
    AnalyticAt ℝ (fun μ ↦ (A.withMu μ).p₁) μ₀ ∧ AnalyticAt ℝ (fun μ ↦ (A.withMu μ).ρ) μ₀ := by
  have hZ := (A.withMu μ₀).Z_pos.ne'
  simp only [withMu, Z, boltz₀, boltz₁, boltz₂] at hZ
  simp only [withMu, p₁, ρ, p₂, p₀, Z, boltz₀, boltz₁, boltz₂]
  constructor <;> fun_prop (disch := exact hZ)

/-- `b = 2μ - iω_k` is analytic in `μ`. -/
theorem analyticAt_b (μ₀ : ℝ) : AnalyticAt ℝ (fun μ ↦ (A.withMu μ).b k) μ₀ := by
  have hμ : AnalyticAt ℝ (fun μ : ℝ ↦ (μ : ℂ)) μ₀ := analyticAt_ofReal_comp analyticAt_id
  simp only [b, w, withMu_μ, withMu_ω]
  exact (analyticAt_const.mul hμ).sub analyticAt_const

/-- `Aₛ` is analytic in `μ`. -/
theorem analyticAt_As (μ₀ : ℝ) : AnalyticAt ℝ (fun μ ↦ (A.withMu μ).As k) μ₀ := by
  obtain ⟨hp, hρ⟩ := A.analyticAt_p₁_ρ μ₀
  have hp' := analyticAt_ofReal_comp hp
  have hρ' := analyticAt_ofReal_comp hρ
  have hb := A.analyticAt_b k μ₀
  simp only [As, withMu_a]
  exact ((analyticAt_const.mul hρ').mul hb).add
    (analyticAt_const.mul (analyticAt_const.sub (analyticAt_const.mul hp')))

/-- `tₙ + Aₛ` is analytic in `μ`. -/
theorem analyticAt_t_add_As (n : ℤ) (μ₀ : ℝ) :
    AnalyticAt ℝ (fun μ ↦ (A.withMu μ).t k n + (A.withMu μ).As k) μ₀ := by
  have hμ : AnalyticAt ℝ (fun μ : ℝ ↦ (μ : ℂ)) μ₀ := analyticAt_ofReal_comp analyticAt_id
  have hx : AnalyticAt ℝ (fun μ ↦ (A.withMu μ).x n) μ₀ := by
    simp only [x, withMu_μ, withMu_ν]
    exact hμ.add analyticAt_const
  exact (hx.mul ((A.analyticAt_b k μ₀).sub hx)).add (A.analyticAt_As k μ₀)

/-- `|zₙ - μ| = |νₙ + ω_k/2|`. -/
theorem norm_z_sub_mu (n : ℤ) : ‖A.z k n - A.μ‖ = |A.ν n + A.ω k / 2| := by
  rw [show A.z k n - A.μ = ((A.ν n + A.ω k / 2 : ℝ) : ℂ) * I by simp [z, x, w]; ring]
  rw [norm_mul, norm_I, mul_one, Complex.norm_real, Real.norm_eq_abs]

/-- Only finitely many `n` have `(νₙ + ω_k/2)² ≤ C`. -/
theorem finite_setOf_sq_le (C : ℝ) : {n : ℤ | (A.ν n + A.ω k / 2) ^ 2 ≤ C}.Finite := by
  have h := (A.tendsto_norm_z_sub k A.μ).eventually_gt_atTop (Real.sqrt (max C 0))
  refine (Filter.eventually_cofinite.1 h).subset fun n hn ↦ ?_
  simp only [Set.mem_ofPred_eq, not_lt, A.norm_z_sub_mu] at hn ⊢
  rw [← Real.sqrt_sq_eq_abs]
  exact Real.sqrt_le_sqrt (hn.trans (le_max_left _ _))

/-- **At fixed `U`, `β`, `k`, every `tₙ + Aₛ` is nonzero for `μ ≠ μ₀` near `μ₀`**: each is analytic
in `μ` and nonzero for large `μ`, so its zeros are isolated, and for all but finitely many `n` it
has no zero near `μ₀` at all. -/
theorem eventually_t_add_As_ne (μ₀ : ℝ) :
    ∀ᶠ μ in 𝓝[≠] μ₀, ∀ n, (A.withMu μ).t k n + (A.withMu μ).As k ≠ 0 := by
  set C : ℝ := A.ω k ^ 2 / 4 + 2 * |A.a| * (|μ₀| + 1) + 3 * A.a ^ 2
  have hnear : ∀ n, ∀ᶠ μ in 𝓝[≠] μ₀, (A.withMu μ).t k n + (A.withMu μ).As k ≠ 0 := by
    intro n
    by_contra hne
    simp only [Filter.not_eventually, not_not] at hne
    set μ₁ : ℝ := 3 * |A.a| + |A.ω k| + 1
    have h0 := AnalyticOnNhd.eqOn_zero_of_preconnected_of_frequently_eq_zero
      (fun μ _ ↦ A.analyticAt_t_add_As k n μ) isPreconnected_univ (Set.mem_univ μ₀) hne
      (Set.mem_univ μ₁)
    have hre := (A.withMu μ₁).re_t_add_As_ge k n
    simp only [withMu_μ, withMu_a, withMu_ν, withMu_ω] at hre
    simp only [Pi.zero_apply] at h0
    rw [h0, zero_re] at hre
    have hμ₁ : |μ₁| = μ₁ := abs_of_nonneg (by positivity)
    rw [hμ₁] at hre
    nlinarith [abs_nonneg A.a, abs_nonneg (A.ω k), sq_abs A.a, sq_abs (A.ω k),
      sq_nonneg (A.ν n + A.ω k / 2)]
  have hball : ∀ᶠ μ in 𝓝[≠] μ₀, |μ| ≤ |μ₀| + 1 := nhdsWithin_le_nhds <|
    (eventually_abs_sub_lt μ₀ one_pos).mono fun μ hμ ↦ by linarith [abs_sub_abs_le_abs_sub μ μ₀]
  filter_upwards [hball, (eventually_all_finite (A.finite_setOf_sq_le k C)).2
    fun n _ ↦ hnear n] with μ hμ hF n
  by_cases hn : n ∈ {n : ℤ | (A.ν n + A.ω k / 2) ^ 2 ≤ C}
  · exact hF n hn
  · intro h0
    have hre := (A.withMu μ).re_t_add_As_ge k n
    simp only [withMu_μ, withMu_a, withMu_ν, withMu_ω] at hre
    rw [h0, zero_re] at hre
    simp only [Set.mem_ofPred_eq, not_le, C] at hn
    nlinarith [abs_nonneg A.a, sq_nonneg μ, mul_le_mul_of_nonneg_left hμ (abs_nonneg A.a)]

/-- A square root `ξ(μ)` of `b²/4 + Aₛ` at `A.withMu μ`, as in Theorem "singlet". -/
noncomputable def xiMu (μ : ℝ) : ℂ :=
  ((A.withMu μ).b k ^ 2 / 4 + (A.withMu μ).As k) ^ (2⁻¹ : ℂ)

theorem xiMu_sq (μ : ℝ) : A.xiMu k μ ^ 2 = (A.withMu μ).b k ^ 2 / 4 + (A.withMu μ).As k := by
  simp [xiMu]

variable {A k}

/-- Near a lattice zero `t_c + Aₛ = 0`, in `μ` at fixed `U`, `β`, `k`, `1 + 𝒦` stays invertible
(`SingletLZ.isUnit_onePlusKS`, `ContHyp.eventually_isUnit`). -/
theorem SingletLZ.eventually_isUnit {c : ℤ} (h : A.SingletLZ k c) :
    ∀ᶠ μ in 𝓝 A.μ, IsUnit ((A.withMu μ).onePlusKS k) := by
  have hH := contHyp_S (U := fun _ ↦ A.U) (β := fun _ ↦ A.β) (μ := id) (hβ := fun _ ↦ A.β_pos)
    (x₀ := A.μ) (k := k) continuousAt_const continuousAt_const continuousAt_id
  filter_upwards [hH.eventually_isUnit (A.chiDecay_chiS k) h.isUnit_onePlusKS] with μ hμ
  exact hμ ((A.withMu μ).chiDecay_chiS k)

/-- **The assumptions of Theorem "singlet" hold near a lattice zero**: at `t_c + Aₛ = 0`, in `μ`
at fixed `U`, `β`, `k`, they hold at every `μ ≠ μ₀` near `μ₀`, with `ξ = xiMu`. -/
theorem SingletLZ.eventually_singletHyp {c : ℤ} (h : A.SingletLZ k c) :
    ∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).SingletHyp k (A.xiMu k μ) := by
  have hP : ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).calP k (-(A.withMu μ).As k) ≠ 0 := by
    have hc : ContinuousAt (fun μ ↦ (A.withMu μ).calP k (-(A.withMu μ).As k)) A.μ := by
      have hAs := (A.analyticAt_As k A.μ).continuousAt
      have hb := (A.analyticAt_b k A.μ).continuousAt
      simp only [calP, withMu_a]
      fun_prop
    refine hc.eventually_ne ?_
    change A.calP k (-A.As k) ≠ 0
    rw [show -A.As k = A.t k c by linear_combination (-1 : ℂ) * h.t_c, calP_eq_mul]
    exact mul_ne_zero (A.t_sub_tp_ne k c) (A.t_sub_tm_ne k c)
  filter_upwards [A.eventually_t_add_As_ne k A.μ, nhdsWithin_le_nhds hP,
    nhdsWithin_le_nhds h.eventually_isUnit] with μ hne hPμ hu
  have ha : ((A.withMu μ).a : ℂ) ≠ 0 := h.a_ne
  exact ⟨ha, hne, A.xiMu_sq k μ,
    ((A.withMu μ).isUnit_onePlusKS_iff_Ks ha hne (A.xiMu_sq k μ) hPμ).mp hu⟩

/-- **Item 1 of Proposition "special" as a limit in `μ`**: at a lattice zero `t_c + Aₛ = 0`,
`gammaS → gammaSlz` as `μ → μ₀`, `μ ≠ μ₀`, at fixed `U`, `β`, `k` (`𝓝[≠] μ₀` is not empty). -/
theorem SingletLZ.tendsto_gammaS {c : ℤ} (h : A.SingletLZ k c) (n m : ℤ) :
    Tendsto (fun μ ↦ (A.withMu μ).gammaS k (A.xiMu k μ) n m) (𝓝[≠] A.μ)
      (𝓝 (A.gammaSlz k c n m)) :=
  tendsto_gammaS_of_SingletLZ (U := fun _ ↦ A.U) (β := fun _ ↦ A.β) (μ := id)
    (hβ := fun _ ↦ A.β_pos) k continuousAt_const continuousAt_const continuousAt_id
    nhdsWithin_le_nhds h h.eventually_singletHyp n m

end Params

end HubbardAtom
