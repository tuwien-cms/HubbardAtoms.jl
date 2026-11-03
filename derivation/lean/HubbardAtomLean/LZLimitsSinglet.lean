import HubbardAtomLean.LZLimits

/-!
# The singlet approach of `LZLimits.lean` is nonempty

`tendsto_gammaS_of_SingletLZ` holds along every filter on which the assumptions of Theorem
"singlet" eventually hold. Here they are shown to hold on the punctured neighbourhood `𝓝[≠] μ₀`
in `μ`, at fixed `U`, `β`, `k`, around a lattice zero `t_c + Aₛ = 0` at `μ₀`. As `𝓝[≠] μ₀` is
nontrivial, item 1 of Proposition "special" is a statement about limits
(`tendsto_gammaS_of_SingletLZ_nhdsNE`).

* With `s = b²/4 + Aₛ` (`= ξ²`) and `yₙ = (2n + 1 + k)π/β`, `tₙ + Aₛ = s + yₙ²` (`t_add_As_eq`):
  the lattice conditions are the one function `s(μ)` avoiding the points `-yₙ²`, which are at
  least `(π/β)²` apart unless `n` lies in the orbit of `c`.
* `Re s = μ² - ω²/4 + 2aρμ + a²(1 - 8p₁)` is real-analytic in `μ` and grows like `μ²`, so it takes
  its value at `μ₀` only at isolated `μ` (`eventually_gS_ne`).
* `Kₛ = 1 + (2a/β)∑ₙ (s + yₙ²)⁻¹` blows up at `μ₀`: the terms of the orbit of `c` are `1/(s - s₀)`,
  with the same sign, and the others stay bounded (`eventually_Ks_ne`).
-/

open Complex Filter Topology

namespace HubbardAtom

namespace Params

/-! ### `tₙ + Aₛ` through `s = b²/4 + Aₛ` -/

variable (A : Params) (k : ℤ)

/-- `s = b²/4 + Aₛ`, the value of `ξ²` in Theorem "singlet". -/
noncomputable def sS : ℂ := A.b k ^ 2 / 4 + A.As k

/-- A square root of `s`. -/
noncomputable def ξS : ℂ := A.sS k ^ ((2 : ℂ)⁻¹)

theorem ξS_sq : A.ξS k ^ 2 = A.sS k := by
  have h := Complex.cpow_nat_inv_pow (A.sS k) (two_ne_zero : (2 : ℕ) ≠ 0)
  rw [ξS]
  simpa only [Nat.cast_ofNat] using h

/-- `yₙ = (2n + 1 + k)π/β`, so that `zₙ = μ + iyₙ`. -/
noncomputable def yS (n : ℤ) : ℝ := (2 * n + 1 + k) * Real.pi / A.β

/-- **`tₙ + Aₛ = s + yₙ²`**, as `tₙ = b²/4 - (zₙ - μ)²`. -/
theorem t_add_As_eq (n : ℤ) : A.t k n + A.As k = A.sS k + (A.yS k n : ℂ) ^ 2 := by
  rw [A.t_eq_z, sS, A.z_eq, yS]
  linear_combination (-(((2 * n + 1 + k) * Real.pi / A.β : ℝ) : ℂ) ^ 2) * I_sq

/-- `s` in real and imaginary parts. -/
theorem sS_eq : A.sS k = ((A.μ ^ 2 - A.ω k ^ 2 / 4 + 2 * A.a * A.ρ * A.μ +
    A.a ^ 2 * (1 - 8 * A.p₁) : ℝ) : ℂ) + ((-(A.ω k) * (A.μ + A.a * A.ρ) : ℝ) : ℂ) * I := by
  simp only [sS, As, b, w]
  push_cast
  linear_combination ((A.ω k : ℂ) ^ 2 / 4) * I_sq

/-! ### Along `μ`, at fixed `U`, `β` -/

/-- `Z` as a function of `μ`. -/
noncomputable def ZS (U β μ : ℝ) : ℝ :=
  Real.exp (-β * (U / 4 + μ)) + 2 * Real.exp (-β * (-U / 4)) + Real.exp (-β * (U / 4 - μ))

/-- `ρ` as a function of `μ`. -/
noncomputable def ρS (U β μ : ℝ) : ℝ :=
  Real.exp (-β * (U / 4 - μ)) / ZS U β μ - Real.exp (-β * (U / 4 + μ)) / ZS U β μ

/-- `p₁` as a function of `μ`. -/
noncomputable def p₁S (U β μ : ℝ) : ℝ := Real.exp (-β * (-U / 4)) / ZS U β μ

/-- `Re s + ω²/4 = μ² + 2aρμ + a²(1 - 8p₁)` as a function of `μ`. -/
noncomputable def gS (U β μ : ℝ) : ℝ :=
  μ ^ 2 + 2 * (U / 2) * ρS U β μ * μ + (U / 2) ^ 2 * (1 - 8 * p₁S U β μ)

theorem ZS_pos (U β μ : ℝ) : 0 < ZS U β μ := by
  unfold ZS
  positivity

theorem ρ_eq_ρS {U β μ : ℝ} (hβ : 0 < β) : (⟨U, β, μ, hβ⟩ : Params).ρ = ρS U β μ := rfl

theorem p₁_eq_p₁S {U β μ : ℝ} (hβ : 0 < β) : (⟨U, β, μ, hβ⟩ : Params).p₁ = p₁S U β μ := rfl

theorem analyticAt_gS (U β x : ℝ) : AnalyticAt ℝ (gS U β) x := by
  have hZ : AnalyticAt ℝ (ZS U β) x := by
    unfold ZS
    fun_prop
  have hZ0 := (ZS_pos U β x).ne'
  unfold gS ρS p₁S
  fun_prop (disch := exact hZ0)

theorem continuous_ρS (U β : ℝ) : Continuous (ρS U β) := by
  have hZ : Continuous (ZS U β) := by
    unfold ZS
    fun_prop
  unfold ρS
  exact ((Real.continuous_exp.comp (by fun_prop)).div hZ fun x ↦ (ZS_pos U β x).ne').sub
    ((Real.continuous_exp.comp (by fun_prop)).div hZ fun x ↦ (ZS_pos U β x).ne')

theorem continuous_p₁S (U β : ℝ) : Continuous (p₁S U β) := by
  have hZ : Continuous (ZS U β) := by
    unfold ZS
    fun_prop
  unfold p₁S
  exact continuous_const.div hZ fun x ↦ (ZS_pos U β x).ne'

/-- `gS(μ) ≥ μ² - 2|a||μ| - 3a²`, as `|ρ| ≤ 1` and `p₁ ≤ 1/2`. -/
theorem gS_ge {U β : ℝ} (hβ : 0 < β) (μ : ℝ) :
    μ ^ 2 - 2 * |U / 2| * |μ| - 3 * (U / 2) ^ 2 ≤ gS U β μ := by
  set A : Params := ⟨U, β, μ, hβ⟩
  have hs := A.weights_sum
  have h0 := A.p₀_pos
  have h1 : 0 < A.p₁ := div_pos (Real.exp_pos _) A.Z_pos
  have h2 : 0 < A.p₂ := div_pos (Real.exp_pos _) A.Z_pos
  have hρ : |ρS U β μ| ≤ 1 := by
    rw [← ρ_eq_ρS hβ]
    change |A.p₂ - A.p₀| ≤ 1
    exact abs_le.mpr ⟨by linarith, by linarith⟩
  have hp : p₁S U β μ ≤ 1 / 2 := by
    rw [← p₁_eq_p₁S hβ]
    linarith
  have hm : -(2 * |U / 2| * |μ|) ≤ 2 * (U / 2) * ρS U β μ * μ := by
    have : |2 * (U / 2) * ρS U β μ * μ| ≤ 2 * |U / 2| * |μ| := by
      rw [abs_mul, abs_mul, abs_mul, abs_two]
      calc 2 * |U / 2| * |ρS U β μ| * |μ| ≤ 2 * |U / 2| * 1 * |μ| := by gcongr
        _ = 2 * |U / 2| * |μ| := by ring
    exact (neg_abs_le _).trans' (neg_le_neg this)
  have hq : -(3 * (U / 2) ^ 2) ≤ (U / 2) ^ 2 * (1 - 8 * p₁S U β μ) := by
    nlinarith [sq_nonneg (U / 2)]
  unfold gS
  linarith

/-- **`Re s` takes its value at `μ₀` only at isolated `μ`**: `gS` is real-analytic and not constant,
as it grows like `μ²`. -/
theorem eventually_gS_ne {U β : ℝ} (hβ : 0 < β) (μ₀ : ℝ) :
    ∀ᶠ μ in 𝓝[≠] μ₀, gS U β μ ≠ gS U β μ₀ := by
  have hA : AnalyticOnNhd ℝ (fun μ ↦ gS U β μ - gS U β μ₀) Set.univ := fun x _ ↦
    (analyticAt_gS U β x).sub analyticAt_const
  rcases (hA μ₀ (Set.mem_univ _)).eventually_eq_zero_or_eventually_ne_zero with h0 | hne
  · exfalso
    have hE := hA.eqOn_zero_of_preconnected_of_eventuallyEq_zero isPreconnected_univ
      (Set.mem_univ μ₀) h0
    obtain ⟨M, hMd⟩ : ∃ M, M = |gS U β μ₀| + 2 * |U / 2| + 3 * (U / 2) ^ 2 + 1 := ⟨_, rfl⟩
    have hM : 1 ≤ M := by
      rw [hMd]
      have := abs_nonneg (gS U β μ₀)
      have := abs_nonneg (U / 2)
      have := sq_nonneg (U / 2)
      linarith
    have hge := gS_ge (U := U) hβ M
    rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ M)] at hge
    have hEM := hE (Set.mem_univ M)
    simp only [Pi.zero_apply, sub_eq_zero] at hEM
    have hg := le_abs_self (gS U β μ₀)
    have hpos : 0 ≤ |gS U β μ₀| + 3 * (U / 2) ^ 2 + 1 := by positivity
    have e : M ^ 2 - 2 * |U / 2| * M - 3 * (U / 2) ^ 2 =
        M * (|gS U β μ₀| + 3 * (U / 2) ^ 2 + 1) - 3 * (U / 2) ^ 2 := by
      rw [hMd]
      ring
    have key := le_mul_of_one_le_left hpos hM
    linarith
  · exact hne.mono fun μ h ↦ sub_ne_zero.mp h

/-! ### The family `μ ↦ (U, β, μ)` -/

/-- The parameters `(U, β, μ)`. -/
abbrev pμ (U β : ℝ) (hβ : 0 < β) (μ : ℝ) : Params := ⟨U, β, μ, hβ⟩

variable {U β : ℝ} (hβ : 0 < β)

theorem ρ_pμ (μ : ℝ) : (pμ U β hβ μ).ρ = ρS U β μ := rfl

theorem p₁_pμ (μ : ℝ) : (pμ U β hβ μ).p₁ = p₁S U β μ := rfl

/-- `s` along the family, with `ω = 2kπ/β` independent of `μ`. -/
theorem sS_pμ (μ : ℝ) : (pμ U β hβ μ).sS k =
    ((gS U β μ - (2 * k * Real.pi / β) ^ 2 / 4 : ℝ) : ℂ) +
      ((-(2 * k * Real.pi / β) * (μ + U / 2 * ρS U β μ) : ℝ) : ℂ) * I := by
  rw [sS_eq, ρ_pμ, p₁_pμ]
  simp only [gS, ω, a]
  push_cast
  ring

theorem re_sS_pμ (μ : ℝ) :
    ((pμ U β hβ μ).sS k).re = gS U β μ - (2 * k * Real.pi / β) ^ 2 / 4 := by
  rw [sS_pμ]
  simp only [add_re, ofReal_re, mul_re, ofReal_im, I_re, I_im]
  ring

theorem continuousAt_sS_pμ (μ₀ : ℝ) : ContinuousAt (fun μ ↦ (pμ U β hβ μ).sS k) μ₀ := by
  have h1 := (analyticAt_gS U β μ₀).continuousAt
  have h2 := (continuous_ρS U β).continuousAt (x := μ₀)
  simp only [sS_pμ]
  fun_prop

/-- `yₙ² - y_c² = ((2n + 1 + k)² - (2c + 1 + k)²)(π/β)²`. -/
theorem yS_sq_sub (A : Params) (n c : ℤ) : A.yS k n ^ 2 - A.yS k c ^ 2 =
    (((2 * n + 1 + k : ℤ) : ℝ) ^ 2 - ((2 * c + 1 + k : ℤ) : ℝ) ^ 2) * (Real.pi / A.β) ^ 2 := by
  simp only [yS]
  push_cast
  ring

/-- `(2n + 1 + k)² = (2c + 1 + k)²` iff `n` is in the orbit of `c`. -/
theorem sq_eq_sq_iff_mem_orbit (n c : ℤ) :
    (2 * n + 1 + k) ^ 2 = (2 * c + 1 + k) ^ 2 ↔ n ∈ ({c, mirror k c} : Finset ℤ) := by
  rw [sq_eq_sq_iff_eq_or_eq_neg, Finset.mem_insert, Finset.mem_singleton]
  unfold mirror
  omega

variable {k} {μ₀ : ℝ} {c : ℤ}

/-- At the lattice zero, `s₀ = -y_c²`. -/
theorem SingletLZ.sS_eq (h₀ : (pμ U β hβ μ₀).SingletLZ k c) :
    (pμ U β hβ μ₀).sS k = -((pμ U β hβ μ₀).yS k c : ℂ) ^ 2 := by
  have h := h₀.t_c
  rw [t_add_As_eq] at h
  linear_combination h

/-- **The lattice conditions near the lattice zero**: for `μ ≠ μ₀` near `μ₀`, `tₙ + Aₛ ≠ 0` for
all `n`. -/
theorem eventually_t_add_As_ne (h₀ : (pμ U β hβ μ₀).SingletLZ k c) :
    ∀ᶠ μ in 𝓝[≠] μ₀, ∀ n, (pμ U β hβ μ).t k n + (pμ U β hβ μ).As k ≠ 0 := by
  have hδ : 0 < (Real.pi / β) ^ 2 := by have := Real.pi_pos; positivity
  have hnear : ∀ᶠ μ in 𝓝 μ₀,
      ‖(pμ U β hβ μ).sS k - (pμ U β hβ μ₀).sS k‖ < (Real.pi / β) ^ 2 := by
    have h : ContinuousAt (fun μ ↦ ‖(pμ U β hβ μ).sS k - (pμ U β hβ μ₀).sS k‖) μ₀ :=
      ((continuousAt_sS_pμ (U := U) k hβ μ₀).sub continuousAt_const).norm
    exact h.eventually (gt_mem_nhds (by simp only [sub_self, norm_zero]; exact hδ))
  filter_upwards [nhdsWithin_le_nhds hnear, eventually_gS_ne (U := U) hβ μ₀] with μ hμ hg n
  have hs₀ := h₀.sS_eq hβ
  have hy : ∀ j, (pμ U β hβ μ).yS k j = (pμ U β hβ μ₀).yS k j := fun _ ↦ rfl
  rw [t_add_As_eq, hy n]
  intro h0
  have key : ((pμ U β hβ μ₀).yS k n : ℂ) ^ 2 - ((pμ U β hβ μ₀).yS k c : ℂ) ^ 2 =
      -((pμ U β hβ μ).sS k - (pμ U β hβ μ₀).sS k) := by
    linear_combination h0 - hs₀
  have hd := yS_sq_sub k (pμ U β hβ μ₀) n c
  by_cases hm : (2 * n + 1 + k) ^ 2 = (2 * c + 1 + k) ^ 2
  · -- in the orbit: `s = s₀`, so `Re s = Re s₀`
    have hm' : (((2 * n + 1 + k : ℤ) : ℝ)) ^ 2 = ((2 * c + 1 + k : ℤ) : ℝ) ^ 2 := by
      exact_mod_cast hm
    rw [hm', sub_self, zero_mul] at hd
    have hss : (pμ U β hβ μ).sS k = (pμ U β hβ μ₀).sS k := by
      have : ((pμ U β hβ μ₀).yS k n : ℂ) ^ 2 - ((pμ U β hβ μ₀).yS k c : ℂ) ^ 2 = 0 := by
        exact_mod_cast hd
      linear_combination key - this
    have := congrArg Complex.re hss
    rw [re_sS_pμ, re_sS_pμ] at this
    exact hg (by linarith)
  · -- off the orbit: `|yₙ² - y_c²| ≥ (π/β)² > |s - s₀|`
    have h1 : (1 : ℝ) ≤ |(((2 * n + 1 + k : ℤ) : ℝ)) ^ 2 - ((2 * c + 1 + k : ℤ) : ℝ) ^ 2| := by
      have : (1 : ℤ) ≤ |(2 * n + 1 + k) ^ 2 - (2 * c + 1 + k) ^ 2| :=
        Int.one_le_abs (sub_ne_zero.mpr hm)
      exact_mod_cast this
    have hge : (Real.pi / β) ^ 2 ≤
        |(pμ U β hβ μ₀).yS k n ^ 2 - (pμ U β hβ μ₀).yS k c ^ 2| := by
      rw [hd, abs_mul, abs_of_pos hδ]
      exact le_mul_of_one_le_left hδ.le h1
    have hn : ‖((pμ U β hβ μ₀).yS k n : ℂ) ^ 2 - ((pμ U β hβ μ₀).yS k c : ℂ) ^ 2‖ =
        |(pμ U β hβ μ₀).yS k n ^ 2 - (pμ U β hβ μ₀).yS k c ^ 2| := by
      have e : (((pμ U β hβ μ₀).yS k n ^ 2 - (pμ U β hβ μ₀).yS k c ^ 2 : ℝ) : ℂ) =
          ((pμ U β hβ μ₀).yS k n : ℂ) ^ 2 - ((pμ U β hβ μ₀).yS k c : ℂ) ^ 2 := by push_cast; ring
      rw [← e, Complex.norm_real, Real.norm_eq_abs]
    rw [key, norm_neg] at hn
    linarith

/-! ### `Kₛ` near the lattice zero -/

variable (k)

/-- `dₙ = yₙ² - y_c²`, independent of `μ`. -/
noncomputable def dS (β : ℝ) (hβ : 0 < β) (c n : ℤ) : ℝ :=
  (pμ 0 β hβ 0).yS k n ^ 2 - (pμ 0 β hβ 0).yS k c ^ 2

/-- The majorant of the terms off the orbit: `2/|dₙ|` off the orbit, `0` on it. -/
noncomputable def GS (β : ℝ) (hβ : 0 < β) (c n : ℤ) : ℝ :=
  if n ∈ ({c, mirror k c} : Finset ℤ) then 0 else 2 / |dS k β hβ c n|

variable {k}

theorem dS_eq (c n : ℤ) : dS k β hβ c n =
    (((2 * n + 1 + k : ℤ) : ℝ) ^ 2 - ((2 * c + 1 + k : ℤ) : ℝ) ^ 2) * (Real.pi / β) ^ 2 :=
  yS_sq_sub k _ n c

theorem dS_eq_zero_of_mem {c n : ℤ} (hn : n ∈ ({c, mirror k c} : Finset ℤ)) :
    dS k β hβ c n = 0 := by
  have hm := (sq_eq_sq_iff_mem_orbit k n c).mpr hn
  have hm' : (((2 * n + 1 + k : ℤ) : ℝ)) ^ 2 = ((2 * c + 1 + k : ℤ) : ℝ) ^ 2 := by
    exact_mod_cast hm
  rw [dS_eq, hm', sub_self, zero_mul]

/-- Off the orbit, `|dₙ| ≥ (π/β)²`. -/
theorem le_abs_dS {c n : ℤ} (hn : n ∉ ({c, mirror k c} : Finset ℤ)) :
    (Real.pi / β) ^ 2 ≤ |dS k β hβ c n| := by
  have hδ : 0 < (Real.pi / β) ^ 2 := by have := Real.pi_pos; positivity
  have hm : (2 * n + 1 + k) ^ 2 ≠ (2 * c + 1 + k) ^ 2 := fun h ↦
    hn ((sq_eq_sq_iff_mem_orbit k n c).mp h)
  have h1 : (1 : ℝ) ≤ |(((2 * n + 1 + k : ℤ) : ℝ)) ^ 2 - ((2 * c + 1 + k : ℤ) : ℝ) ^ 2| := by
    have : (1 : ℤ) ≤ |(2 * n + 1 + k) ^ 2 - (2 * c + 1 + k) ^ 2| :=
      Int.one_le_abs (sub_ne_zero.mpr hm)
    exact_mod_cast this
  rw [dS_eq, abs_mul, abs_of_pos hδ]
  exact le_mul_of_one_le_left hδ.le h1

theorem GS_nonneg (c n : ℤ) : 0 ≤ GS k β hβ c n := by
  unfold GS
  split_ifs
  · exact le_rfl
  · positivity

/-- **The majorant is summable**: for `|n| ≥ N = |k| + 1 + |2c + 1 + k|`,
`(2n + 1 + k)² - (2c + 1 + k)² ≥ n²`, so `GSₙ ≤ 2(β/π)²/n²`. -/
theorem summable_GS (c : ℤ) : Summable (GS k β hβ c) := by
  have hδ : 0 < (Real.pi / β) ^ 2 := by have := Real.pi_pos; positivity
  set N : ℤ := |k| + 1 + |2 * c + 1 + k|
  have hev : ∀ᶠ n in cofinite, N ≤ |n| := by
    refine (Set.finite_Ioo (-N) N).subset fun n hn ↦ ?_
    have hn' : ¬N ≤ |n| := hn
    exact Set.mem_Ioo.mpr (abs_lt.mp (not_le.mp hn'))
  refine Summable.of_norm_bounded_eventually
    ((Real.summable_one_div_int_pow.mpr one_lt_two).mul_left (2 / (Real.pi / β) ^ 2)) ?_
  filter_upwards [hev] with n hn
  have hN : (1 : ℤ) ≤ N := by
    have := abs_nonneg k
    have := abs_nonneg (2 * c + 1 + k)
    omega
  have hn0 : n ≠ 0 := abs_pos.mp (lt_of_lt_of_le one_pos (hN.trans hn))
  have hnR : (0 : ℝ) < (n : ℝ) ^ 2 := by
    have : (n : ℝ) ≠ 0 := by exact_mod_cast hn0
    positivity
  -- `(2n + 1 + k)² - (2c + 1 + k)² ≥ n²`, as `|2n + 1 + k| ≥ |n| + |2c + 1 + k|`
  have h1 : |n| + |2 * c + 1 + k| ≤ |2 * n + 1 + k| := by
    have h := abs_sub_abs_le_abs_sub (2 * n) (-(1 + k))
    rw [show 2 * n - -(1 + k) = 2 * n + 1 + k by ring, abs_neg, abs_mul, abs_two] at h
    have := abs_add_le (1 : ℤ) k
    rw [abs_one] at this
    omega
  have h2 := pow_le_pow_left₀ (add_nonneg (abs_nonneg n) (abs_nonneg _)) h1 2
  have h3 : n ^ 2 + (2 * c + 1 + k) ^ 2 ≤ (2 * n + 1 + k) ^ 2 := by
    rw [sq_abs] at h2
    nlinarith [sq_abs n, sq_abs (2 * c + 1 + k), abs_nonneg n, abs_nonneg (2 * c + 1 + k)]
  have h3R : (n : ℝ) ^ 2 ≤ (((2 * n + 1 + k : ℤ) : ℝ)) ^ 2 - ((2 * c + 1 + k : ℤ) : ℝ) ^ 2 := by
    have : n ^ 2 ≤ (2 * n + 1 + k) ^ 2 - (2 * c + 1 + k) ^ 2 := by linarith
    exact_mod_cast this
  have hnO : n ∉ ({c, mirror k c} : Finset ℤ) := fun h ↦ by
    have := (sq_eq_sq_iff_mem_orbit k n c).mpr h
    have : (n : ℤ) ^ 2 ≤ 0 := by linarith
    exact hn0 (pow_eq_zero_iff two_ne_zero |>.mp (le_antisymm this (sq_nonneg n)))
  have hd : (n : ℝ) ^ 2 * (Real.pi / β) ^ 2 ≤ |dS k β hβ c n| := by
    rw [dS_eq, abs_mul, abs_of_pos hδ, abs_of_pos (hnR.trans_le h3R)]
    exact mul_le_mul_of_nonneg_right h3R hδ.le
  rw [Real.norm_of_nonneg (GS_nonneg hβ c n), GS, ite_eq_right hnO]
  calc 2 / |dS k β hβ c n| ≤ 2 / ((n : ℝ) ^ 2 * (Real.pi / β) ^ 2) :=
        div_le_div_of_nonneg_left zero_le_two (by positivity) hd
    _ = 2 / (Real.pi / β) ^ 2 * (1 / (n : ℝ) ^ 2) := by
        field_simp

/-- **`Kₛ ≠ 0` near the lattice zero**: `Kₛ = 1 + (2a/β)𝒯` with `𝒯 = |O|/(s - s₀) + R`, `|R| ≤ ∑ GS`,
and `s - s₀ → 0`. -/
theorem eventually_Ks_ne (h₀ : (pμ U β hβ μ₀).SingletLZ k c) :
    ∀ᶠ μ in 𝓝[≠] μ₀, (pμ U β hβ μ).Ks k ((pμ U β hβ μ).ξS k) ≠ 0 := by
  have hπ := Real.pi_pos
  have hδ : 0 < (Real.pi / β) ^ 2 := by positivity
  set O : Finset ℤ := {c, mirror k c}
  set C₀ := ∑' n, GS k β hβ c n
  have hC₀ : 0 ≤ C₀ := tsum_nonneg (GS_nonneg hβ c)
  have ha : (U / 2 : ℝ) ≠ 0 := by
    have := h₀.a_ne
    exact_mod_cast this
  have ha' : 0 < |U / 2| := abs_pos.mpr ha
  set η := min ((Real.pi / β) ^ 2 / 2) (1 / (C₀ + β / |U / 2|))
  have hη : 0 < η := lt_min (by positivity) (by positivity)
  have hsmall : ∀ᶠ μ in 𝓝 μ₀, ‖(pμ U β hβ μ).sS k - (pμ U β hβ μ₀).sS k‖ < η := by
    have h : ContinuousAt (fun μ ↦ ‖(pμ U β hβ μ).sS k - (pμ U β hβ μ₀).sS k‖) μ₀ :=
      ((continuousAt_sS_pμ (U := U) k hβ μ₀).sub continuousAt_const).norm
    exact h.eventually (gt_mem_nhds (by simp only [sub_self, norm_zero]; exact hη))
  filter_upwards [eventually_t_add_As_ne hβ h₀, nhdsWithin_le_nhds hsmall] with μ hl hsm
  set A := pμ U β hβ μ
  set e := A.sS k - (pμ U β hβ μ₀).sS k with he
  have hs₀ := h₀.sS_eq hβ
  have hy : ∀ j, A.yS k j = (pμ 0 β hβ 0).yS k j := fun _ ↦ rfl
  have hy₀ : ∀ j, (pμ U β hβ μ₀).yS k j = (pμ 0 β hβ 0).yS k j := fun _ ↦ rfl
  -- `tₙ + Aₛ = e + dₙ`
  have hte : ∀ n, A.t k n + A.As k = e + (dS k β hβ c n : ℂ) := fun n ↦ by
    rw [t_add_As_eq, dS, hy n]
    rw [hy₀ c] at hs₀
    push_cast
    linear_combination hs₀ - he
  have he0 : e ≠ 0 := by
    have := hl c
    rwa [hte, dS_eq_zero_of_mem hβ (Finset.mem_insert_self c _), ofReal_zero, add_zero] at this
  -- the lattice sum
  have hT := A.hasSum_Tpp (k := k) (t := -A.As k) (ξ := A.ξS k)
    (by rw [ξS_sq, sS]; ring) (fun n h0 ↦ hl n (by rw [h0]; ring))
  have hT' : HasSum (fun n ↦ 1 / (e + (dS k β hβ c n : ℂ))) (A.Tpp k (A.ξS k)) := by
    refine hT.congr_fun fun n ↦ ?_
    rw [sub_neg_eq_add, hte]
  -- the orbit and the rest
  set f : ℤ → ℂ := fun n ↦ 1 / (e + (dS k β hβ c n : ℂ))
  have hO : HasSum (fun n ↦ if n ∈ O then f n else 0) (∑ n ∈ O, (if n ∈ O then f n else 0)) :=
    hasSum_sum_of_ne_finset_zero fun n hn ↦ ite_eq_right hn
  have hsumO : ∑ n ∈ O, (if n ∈ O then f n else 0) = (O.card : ℂ) * (1 / e) := by
    rw [Finset.sum_congr rfl fun n hn ↦ by rw [ite_eq_left hn, show f n = 1 / e by
      simp only [f]; rw [dS_eq_zero_of_mem hβ hn, ofReal_zero, add_zero]], Finset.sum_const,
      nsmul_eq_mul]
  have hR : HasSum (fun n ↦ if n ∈ O then 0 else f n)
      (A.Tpp k (A.ξS k) - (O.card : ℂ) * (1 / e)) := by
    rw [← hsumO]
    refine (hT'.sub hO).congr_fun fun n ↦ ?_
    by_cases hn : n ∈ O <;> simp [hn, f]
  have hbd : ∀ n, ‖(if n ∈ O then 0 else f n : ℂ)‖ ≤ GS k β hβ c n := fun n ↦ by
    by_cases hn : n ∈ O
    · simp [hn, GS, O]
    · rw [ite_eq_right hn, GS, ite_eq_right hn]
      have hd := le_abs_dS hβ hn
      have he : ‖e‖ ≤ |dS k β hβ c n| / 2 := by
        have := min_le_left ((Real.pi / β) ^ 2 / 2) (1 / (C₀ + β / |U / 2|))
        linarith
      have hlow : |dS k β hβ c n| / 2 ≤ ‖e + (dS k β hβ c n : ℂ)‖ := by
        have := norm_sub_norm_le ((dS k β hβ c n : ℂ)) (-e)
        rw [norm_neg, sub_neg_eq_add, add_comm, Complex.norm_real, Real.norm_eq_abs] at this
        linarith
      have hpos : 0 < |dS k β hβ c n| / 2 := by linarith
      simp only [f, norm_div, norm_one]
      rw [div_le_div_iff₀ (hpos.trans_le hlow) (by linarith)]
      linarith
  have hRn := hR.norm_le_of_bounded (summable_GS hβ c).hasSum hbd
  -- `‖𝒯‖ ≥ 1/‖e‖ - C₀`
  have hcard : (1 : ℝ) ≤ O.card := by
    have : 1 ≤ O.card := (Finset.card_pos (s := O)).mpr ⟨c, Finset.mem_insert_self c _⟩
    exact_mod_cast this
  have heN : 0 < ‖e‖ := norm_pos_iff.mpr he0
  have hO1 : 1 / ‖e‖ ≤ ‖(O.card : ℂ) * (1 / e)‖ := by
    rw [norm_mul, Complex.norm_natCast, norm_div, norm_one]
    exact le_mul_of_one_le_left (by positivity) hcard
  have hTn : 1 / ‖e‖ - C₀ ≤ ‖A.Tpp k (A.ξS k)‖ := by
    have := norm_sub_norm_le ((O.card : ℂ) * (1 / e)) (A.Tpp k (A.ξS k))
    rw [norm_sub_rev] at hRn
    rw [norm_sub_rev] at this
    linarith [norm_sub_rev (A.Tpp k (A.ξS k)) ((O.card : ℂ) * (1 / e))]
  -- `1/‖e‖ > C₀ + β/|a|`
  have hinv : C₀ + β / |U / 2| < 1 / ‖e‖ := by
    have hlt : ‖e‖ < 1 / (C₀ + β / |U / 2|) := hsm.trans_le (min_le_right _ _)
    rw [lt_div_iff₀ heN]
    rw [lt_div_iff₀ (by positivity)] at hlt
    linarith
  -- `‖Kₛ‖ ≥ (2|a|/β)‖𝒯‖ - 1 > 1`
  intro hK
  have hKe : (2 * (A.a : ℂ) * A.Tpp k (A.ξS k) / (A.β : ℂ)) = -1 := by
    simp only [Ks] at hK
    linear_combination hK
  have hn := congrArg norm hKe
  rw [norm_neg, norm_one, norm_div, norm_mul, norm_mul, Complex.norm_two, Complex.norm_real,
    Complex.norm_real, Real.norm_eq_abs, Real.norm_of_nonneg hβ.le] at hn
  change 2 * |U / 2| * ‖A.Tpp k (A.ξS k)‖ / β = 1 at hn
  have h2 : 2 < 2 * |U / 2| * ‖A.Tpp k (A.ξS k)‖ / β := by
    rw [lt_div_iff₀ hβ]
    have : β / |U / 2| < ‖A.Tpp k (A.ξS k)‖ := by linarith
    rw [div_lt_iff₀ ha'] at this
    linarith
  linarith

/-- **The assumptions of Theorem "singlet" hold near the lattice zero**, `μ ≠ μ₀`. -/
theorem eventually_singletHyp (h₀ : (pμ U β hβ μ₀).SingletLZ k c) :
    ∀ᶠ μ in 𝓝[≠] μ₀, (pμ U β hβ μ).SingletHyp k ((pμ U β hβ μ).ξS k) := by
  filter_upwards [eventually_t_add_As_ne hβ h₀, eventually_Ks_ne hβ h₀] with μ hl hK
  exact ⟨h₀.a_ne, hl, by rw [ξS_sq]; rfl, hK⟩

/-- **Item 1 of Proposition "special" as a limit**: at a lattice zero `t_c + Aₛ = 0` at `μ₀`,
`gammaS → gammaSlz` as `μ → μ₀`, `μ ≠ μ₀`, at fixed `U`, `β`. As `𝓝[≠] μ₀` is nontrivial, the
closed form is defined at points arbitrarily close to `μ₀` and this is a limit. -/
theorem tendsto_gammaS_of_SingletLZ_nhdsNE (h₀ : (pμ U β hβ μ₀).SingletLZ k c) (n m : ℤ) :
    Tendsto (fun μ ↦ (pμ U β hβ μ).gammaS k ((pμ U β hβ μ).ξS k) n m) (𝓝[≠] μ₀)
      (𝓝 ((pμ U β hβ μ₀).gammaSlz k c n m)) :=
  tendsto_gammaS_of_SingletLZ (X := ℝ) (U := fun _ ↦ U) (β := fun _ ↦ β) (μ := id)
    (hβ := fun _ ↦ hβ) k continuousAt_const continuousAt_const continuousAt_id nhdsWithin_le_nhds
    h₀ (eventually_singletHyp hβ h₀) n m

end Params

end HubbardAtom
