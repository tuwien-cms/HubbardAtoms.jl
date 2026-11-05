import HubbardAtomLean.LZLimits

/-!
# The half-filling lattice zeros are approached in `β`

`tendsto_gammaHF_of_HFLZ` (`LZLimits.lean`) gives `gammaHF → gammaHFlz` along any approach to a
lattice zero `z_c² = ŝ` of Corollary "half filling" through points where its assumptions hold.
Here these points are shown to exist: at fixed `U`, `k` and `μ = 0`, the assumptions hold at every
`β ≠ β₀` near `β₀`, so item 4 of Proposition "special" holds as a limit in `β`, as in the note's
proof.

* `γ = -β²ŝ/π²` (`gHF`) is analytic in `β` on all of `ℝ`, and `zₙ² = ŝ` iff `γ = (2n + 1 + k)²`
  (`z_sq_sub_shat`). As `γ(0) = k² ≠ γ(β₀)`, `γ ≠ γ(β₀)` at every `β ≠ β₀` near `β₀`
  (`HFLZ.eventually_gHF_ne`), so there `γ` is no integer and `zₙ² ≠ ŝ` for all `n`
  (`HFLZ.eventually_z_sq_ne`); `zₙ² ≠ ŝ'` off the mirror fixed points stays true near `β₀`
  (`HFLZ.eventually_odd_ne`);
* `HFLZ.eventually_Dhat_ne`: `D̂ ≠ 0` at every `β ≠ β₀` near `β₀`, as the orbit terms of
  `T̂ = ∑ₗ ĝ(zₗ)` blow up and the others converge by dominated convergence (`HFLZ.tendsto_rhoHF`);
* `HFLZ.tendsto_gammaHF`: **`gammaHF → gammaHFlz` as `β → β₀`, `β ≠ β₀`**, at fixed `U`, `k` and
  `μ = 0`, with `û = uBeta`.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ)

/-- `A` with the inverse temperature `β` (for `β > 0`; `A.β` otherwise). -/
noncomputable def withBeta (β : ℝ) : Params :=
  ⟨A.U, if 0 < β then β else A.β, A.μ, by split_ifs with h; exacts [h, A.β_pos]⟩

theorem withBeta_β {β : ℝ} (hβ : 0 < β) : (A.withBeta β).β = β := ite_eq_left hβ
@[simp] theorem withBeta_a (β : ℝ) : (A.withBeta β).a = A.a := rfl
@[simp] theorem withBeta_μ (β : ℝ) : (A.withBeta β).μ = A.μ := rfl

/-- `p₀ + sp₁` as a function of every real `β`, at fixed `U`, `μ` (used for `s = ±1`). -/
noncomputable def pBeta (s β : ℝ) : ℝ :=
  (Real.exp (-β * (A.U / 4 + A.μ)) + s * Real.exp (-β * (-A.U / 4))) /
    (Real.exp (-β * (A.U / 4 + A.μ)) + 2 * Real.exp (-β * (-A.U / 4)) +
      Real.exp (-β * (A.U / 4 - A.μ)))

theorem pBeta_eq {β : ℝ} (hβ : 0 < β) (s : ℝ) :
    A.pBeta s β = (A.withBeta β).p₀ + s * (A.withBeta β).p₁ := by
  simp only [pBeta, p₀, p₁, Z, boltz₀, boltz₁, boltz₂, withBeta, ite_eq_left hβ]
  ring

theorem analyticAt_pBeta (s β : ℝ) : AnalyticAt ℝ (A.pBeta s) β := by
  have hZ : Real.exp (-β * (A.U / 4 + A.μ)) + 2 * Real.exp (-β * (-A.U / 4)) +
      Real.exp (-β * (A.U / 4 - A.μ)) ≠ 0 := by positivity
  unfold pBeta
  fun_prop (disch := exact hZ)

/-- `-β²(4ea²p - q₀)/π²` for every real `β`, where `q₀ = a² + (kπ/β)²` at `β > 0`. -/
noncomputable def levBeta (e p β : ℝ) : ℝ :=
  (β ^ 2 * A.a ^ 2 + k ^ 2 * Real.pi ^ 2 - 4 * e * A.a ^ 2 * β ^ 2 * p) / Real.pi ^ 2

/-- At `μ = 0`, `zₙ² - (4ea²p - q₀) = (π/β)²(levBeta - (2n + 1 + k)²)`. -/
theorem z_sq_sub_eq {β : ℝ} (hβ : 0 < β) (hμ : A.μ = 0) (e p : ℝ) (n : ℤ) :
    (A.withBeta β).z k n ^ 2 - (4 * e * (A.a : ℂ) ^ 2 * p - (A.withBeta β).q₀ k) =
      ((Real.pi : ℂ) / β) ^ 2 *
        ((A.levBeta k e p β - ((2 * n + 1 + k) ^ 2 : ℤ) : ℝ) : ℂ) := by
  have hβ' : (β : ℂ) ≠ 0 := ofReal_ne_zero.mpr hβ.ne'
  have hπ : (Real.pi : ℂ) ≠ 0 := ofReal_ne_zero.mpr Real.pi_ne_zero
  rw [z_eq]
  simp only [q₀, w, ω, levBeta, withBeta_μ, withBeta_a, A.withBeta_β hβ, hμ]
  push_cast
  field_simp
  ring_nf
  rw [I_sq]
  ring

theorem withBeta_self : A.withBeta A.β = A := by
  simp [withBeta, A.β_pos]

variable (ε : ℂ)

/-- `γ = -β²ŝ/π²` as a function of every real `β`, for `ε = ±1` (as `Re ε`): at `μ = 0`, `zₙ² = ŝ`
iff `γ = (2n + 1 + k)²` (`z_sq_sub_shat`). -/
noncomputable def gHF (β : ℝ) : ℝ := A.levBeta k ε.re (A.pBeta (-1) β) β

/-- `γ' = -β²ŝ'/π²` as a function of every real `β`, for `ε = ±1` (`z_sq_sub_shat'`). -/
noncomputable def gHF' (β : ℝ) : ℝ := A.levBeta k (-ε.re) (A.pBeta 1 β) β

variable {ε}

theorem ofReal_re_of_sign (hε : ε = 1 ∨ ε = -1) : (ε.re : ℂ) = ε := by
  rcases hε with rfl | rfl <;> simp

theorem z_sq_sub_shat (hε : ε = 1 ∨ ε = -1) (hμ : A.μ = 0) {β : ℝ} (hβ : 0 < β) (n : ℤ) :
    (A.withBeta β).z k n ^ 2 - (A.withBeta β).shat k ε =
      ((Real.pi : ℂ) / β) ^ 2 * ((A.gHF k ε β - ((2 * n + 1 + k) ^ 2 : ℤ) : ℝ) : ℂ) := by
  rw [gHF, ← z_sq_sub_eq _ _ hβ hμ, shat, A.pBeta_eq hβ, withBeta_a]
  push_cast [ofReal_re_of_sign hε]
  ring

theorem z_sq_sub_shat' (hε : ε = 1 ∨ ε = -1) (hμ : A.μ = 0) {β : ℝ} (hβ : 0 < β) (n : ℤ) :
    (A.withBeta β).z k n ^ 2 - (A.withBeta β).shat' k ε =
      ((Real.pi : ℂ) / β) ^ 2 * ((A.gHF' k ε β - ((2 * n + 1 + k) ^ 2 : ℤ) : ℝ) : ℂ) := by
  rw [gHF', ← z_sq_sub_eq _ _ hβ hμ, shat', A.pBeta_eq hβ, withBeta_a]
  push_cast [ofReal_re_of_sign hε]
  ring

theorem analyticAt_gHF (β : ℝ) : AnalyticAt ℝ (A.gHF k ε) β := by
  have hp := A.analyticAt_pBeta (-1) β
  unfold gHF levBeta
  fun_prop

theorem continuous_gHF' : Continuous (A.gHF' k ε) := by
  have hp : Continuous (A.pBeta 1) := continuous_iff_continuousAt.mpr fun β ↦
    (A.analyticAt_pBeta 1 β).continuousAt
  unfold gHF' levBeta
  fun_prop

theorem pi_div_sq_ne {β : ℝ} (hβ : 0 < β) : ((Real.pi : ℂ) / β) ^ 2 ≠ 0 :=
  pow_ne_zero _ (div_ne_zero (ofReal_ne_zero.mpr Real.pi_ne_zero) (ofReal_ne_zero.mpr hβ.ne'))

variable {A k} {c : ℤ}

/-- `zₙ² = ŝ` at `β` iff `γ(β) = (2n + 1 + k)²`. -/
theorem z_sq_sub_shat_eq_zero (hε : ε = 1 ∨ ε = -1) (hμ : A.μ = 0) {β : ℝ} (hβ : 0 < β) {n : ℤ}
    (h0 : (A.withBeta β).z k n ^ 2 - (A.withBeta β).shat k ε = 0) :
    A.gHF k ε β = ((2 * n + 1 + k) ^ 2 : ℤ) := by
  rw [A.z_sq_sub_shat k hε hμ hβ] at h0
  exact sub_eq_zero.mp (ofReal_eq_zero.mp ((mul_eq_zero.mp h0).resolve_left (pi_div_sq_ne hβ)))

/-- At a lattice zero, `γ(β₀) = (2c + 1 + k)²`. -/
theorem HFLZ.gHF_eq (h : A.HFLZ k ε c) : A.gHF k ε A.β = ((2 * c + 1 + k) ^ 2 : ℤ) := by
  have h0 := h.z_c
  rw [← A.withBeta_self] at h0
  exact z_sq_sub_shat_eq_zero h.ε_eq h.μ_eq A.β_pos h0

/-- At a lattice zero, `γ ≠ γ(β₀)` at every `β ≠ β₀` near `β₀`: `γ` is analytic, and
`γ(0) = k² ≠ (2c + 1 + k)²`. -/
theorem HFLZ.eventually_gHF_ne (h : A.HFLZ k ε c) :
    ∀ᶠ β in 𝓝[≠] A.β, A.gHF k ε β ≠ A.gHF k ε A.β := by
  by_contra hne
  simp only [Filter.not_eventually, not_not] at hne
  have h0 := AnalyticOnNhd.eqOn_of_preconnected_of_frequently_eq
    (fun β _ ↦ A.analyticAt_gHF k β) (fun β _ ↦ analyticAt_const) isPreconnected_univ
    (Set.mem_univ A.β) hne (Set.mem_univ 0)
  rw [h.gHF_eq] at h0
  simp only at h0
  have hk : ((k ^ 2 : ℤ) : ℝ) = ((2 * c + 1 + k) ^ 2 : ℤ) := by
    rw [← h0, gHF, levBeta]
    field_simp
    push_cast
    ring
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp (Int.cast_inj.mp hk) with h1 | h1 <;> omega

/-- At a lattice zero, `zₙ² ≠ ŝ` for all `n` at every `β ≠ β₀` near `β₀`: there
`0 < |γ - γ(β₀)| < 1`, and `γ(β₀)` is an integer. -/
theorem HFLZ.eventually_z_sq_ne (h : A.HFLZ k ε c) :
    ∀ᶠ β in 𝓝[≠] A.β, ∀ n, (A.withBeta β).z k n ^ 2 - (A.withBeta β).shat k ε ≠ 0 := by
  have hlt := (A.analyticAt_gHF k A.β (ε := ε)).continuousAt.tendsto.eventually
    (eventually_abs_sub_lt _ one_pos)
  filter_upwards [h.eventually_gHF_ne, nhdsWithin_le_nhds hlt,
    nhdsWithin_le_nhds (eventually_gt_nhds A.β_pos)] with β hne hlt hβ n h0
  rw [z_sq_sub_shat_eq_zero h.ε_eq h.μ_eq hβ h0, h.gHF_eq] at hne hlt
  have hj : |(2 * n + 1 + k) ^ 2 - (2 * c + 1 + k) ^ 2| < 1 := by exact_mod_cast hlt
  exact hne (by rw [sub_eq_zero.mp (Int.abs_lt_one_iff.mp hj)])

/-- At a lattice zero, `zₙ² ≠ ŝ'` for all `n ≠ mirror k n` at every `β` near `β₀`: `γ'` is
continuous, and the squares `(2n + 1 + k)²` form a closed set. -/
theorem HFLZ.eventually_odd_ne (h : A.HFLZ k ε c) :
    ∀ᶠ β in 𝓝 A.β, ∀ n, n ≠ mirror k n →
      (A.withBeta β).z k n ^ 2 - (A.withBeta β).shat' k ε ≠ 0 := by
  set S : Set ℝ := (↑) '' {j : ℤ | ∃ n, n ≠ mirror k n ∧ j = (2 * n + 1 + k) ^ 2}
  have hS : IsClosed S := Int.isClosedEmbedding_coe_real.isClosedMap _ (isClosed_discrete _)
  have hmem : ∀ {β : ℝ}, 0 < β → ∀ n, n ≠ mirror k n →
      (A.withBeta β).z k n ^ 2 - (A.withBeta β).shat' k ε = 0 → A.gHF' k ε β ∈ S := by
    intro β hβ n hn h0
    rw [A.z_sq_sub_shat' k h.ε_eq h.μ_eq hβ] at h0
    exact ⟨_, ⟨n, hn, rfl⟩, (sub_eq_zero.mp (ofReal_eq_zero.mp
      ((mul_eq_zero.mp h0).resolve_left (pi_div_sq_ne hβ)))).symm⟩
  have h₀ : A.gHF' k ε A.β ∉ S := by
    rintro ⟨_, ⟨n, hn, rfl⟩, he⟩
    refine h.odd_ne n hn ?_
    rw [← A.withBeta_self, A.z_sq_sub_shat' k h.ε_eq h.μ_eq A.β_pos, ← he, sub_self,
      ofReal_zero, mul_zero]
  filter_upwards [(A.continuous_gHF' k (ε := ε)).continuousAt.eventually_mem
    (hS.isOpen_compl.mem_nhds h₀), eventually_gt_nhds A.β_pos] with β hSβ hβ n hn h0
  exact hSβ (hmem hβ n hn h0)

variable (A k ε) in
/-- `(γ(β) - (2l + 1 + k)²)⁻¹`, which is `(π/β)²ĝ(zₗ)` at `β > 0`. -/
noncomputable def fHF (β : ℝ) (l : ℤ) : ℂ :=
  ((A.gHF k ε β - ((2 * l + 1 + k) ^ 2 : ℤ) : ℝ) : ℂ)⁻¹

variable (k c) in
/-- The orbit `{c, mirror k c}`. -/
def orbF : Finset ℤ := {c, mirror k c}

variable (A k ε c) in
/-- `∑_{l ∉ {c, c̄}} fHF β l`, the terms of `(π/β)²T̂` off the orbit. -/
noncomputable def rhoHF (β : ℝ) : ℂ := ∑' l, if l ∈ orbF k c then 0 else A.fHF k ε β l

theorem mem_orbF {l : ℤ} : l ∈ orbF k c ↔ ((2 * l + 1 + k) ^ 2 : ℤ) = (2 * c + 1 + k) ^ 2 := by
  rw [sq_eq_sq_iff_eq_or_eq_neg]
  simp only [orbF, Finset.mem_insert, Finset.mem_singleton, mirror]
  omega

/-- `ĝ(zₗ) = (β/π)² fHF β l` at `β`. -/
theorem ghat_z_eq (hε : ε = 1 ∨ ε = -1) (hμ : A.μ = 0) {β : ℝ} (hβ : 0 < β) (l : ℤ) :
    (A.withBeta β).ghat k ε ((A.withBeta β).z k l) =
      (((Real.pi : ℂ) / β) ^ 2)⁻¹ * A.fHF k ε β l := by
  rw [ghat, A.z_sq_sub_shat k hε hμ hβ, one_div, mul_inv, fHF]

theorem HFLZ.fHF_eq_of_mem (h : A.HFLZ k ε c) (β : ℝ) {l : ℤ} (hl : l ∈ orbF k c) :
    A.fHF k ε β l = ((A.gHF k ε β - A.gHF k ε A.β : ℝ) : ℂ)⁻¹ := by
  rw [fHF, mem_orbF.mp hl, h.gHF_eq]

theorem HFLZ.gHF_sub_ne (h : A.HFLZ k ε c) {l : ℤ} (hl : l ∉ orbF k c) :
    A.gHF k ε A.β - ((2 * l + 1 + k) ^ 2 : ℤ) ≠ 0 := by
  rw [h.gHF_eq, sub_ne_zero, Ne, Int.cast_inj]
  exact fun h0 ↦ hl (mem_orbF.mpr h0.symm)

/-- `∑ₗ fHF β₀ l` converges: it is `(π/β₀)²T̂'` with `T̂' = ∑_{l ∉ O} ĝ(zₗ)`. -/
theorem HFLZ.summable_fHF (h : A.HFLZ k ε c) : Summable (A.fHF k ε A.β) := by
  convert h.summable_ghat.mul_left (((Real.pi : ℂ) / A.β) ^ 2) using 1
  funext l
  have hg := ghat_z_eq (A := A) (k := k) h.ε_eq h.μ_eq A.β_pos l
  rw [withBeta_self] at hg
  rw [hg, ← mul_assoc, mul_inv_cancel₀ (pi_div_sq_ne A.β_pos), one_mul]

/-- The terms off the orbit converge as `β → β₀`, dominated by `2‖fHF β₀ l‖`: there
`|γ - γ(β₀)| < 1/2` and `|γ(β₀) - (2l + 1 + k)²| ≥ 1`. -/
theorem HFLZ.tendsto_rhoHF (h : A.HFLZ k ε c) :
    Tendsto (A.rhoHF k ε c) (𝓝[≠] A.β) (𝓝 (∑' l, A.fHF k ε A.β l)) := by
  have hg := (A.analyticAt_gHF k A.β (ε := ε)).continuousAt
  have hlt := hg.tendsto.eventually (eventually_abs_sub_lt _ (one_half_pos (α := ℝ)))
  refine tendsto_tsum_of_dominated_convergence (bound := fun l ↦ 2 * ‖A.fHF k ε A.β l‖)
    (h.summable_fHF.norm.mul_left 2) (fun l ↦ ?_) (nhdsWithin_le_nhds (hlt.mono fun β hβ l ↦ ?_))
  · by_cases hl : l ∈ orbF k c
    · simp only [hl, ↓reduceIte]
      rw [h.fHF_eq_of_mem A.β hl, sub_self, ofReal_zero, inv_zero]
      exact tendsto_const_nhds
    · simp only [hl, ↓reduceIte]
      exact (((continuous_ofReal.continuousAt.comp (hg.sub continuousAt_const)).inv₀
        (ofReal_ne_zero.mpr (h.gHF_sub_ne hl))).tendsto).mono_left nhdsWithin_le_nhds
  · by_cases hl : l ∈ orbF k c
    · simp only [hl, ↓reduceIte, norm_zero]
      positivity
    · simp only [hl, ↓reduceIte, fHF, norm_inv, Complex.norm_real, Real.norm_eq_abs]
      set m : ℝ := (((2 * l + 1 + k) ^ 2 : ℤ) : ℝ)
      have h1 : (1 : ℝ) ≤ |A.gHF k ε A.β - m| := by
        have hne := h.gHF_sub_ne hl
        rw [h.gHF_eq, ← Int.cast_sub] at hne ⊢
        exact_mod_cast Int.one_le_abs (by exact_mod_cast hne)
      have h2 : |A.gHF k ε A.β - m| / 2 ≤ |A.gHF k ε β - m| := by
        have := abs_sub_abs_le_abs_sub (A.gHF k ε A.β - m) (A.gHF k ε β - m)
        rw [show A.gHF k ε A.β - m - (A.gHF k ε β - m) = -(A.gHF k ε β - A.gHF k ε A.β) by ring,
          abs_neg] at this
        linarith
      rw [show 2 * |A.gHF k ε A.β - m|⁻¹ = (|A.gHF k ε A.β - m| / 2)⁻¹ by rw [inv_div]; ring]
      exact inv_anti₀ (by linarith) h2

/-- `T̂ = (β/π)²(r/(γ - γ(β₀)) + rhoHF β)` at `β`, with `r` the size of the orbit. -/
theorem HFLZ.That_eq (h : A.HFLZ k ε c) {β : ℝ} (hβ : 0 < β) {u : ℂ}
    (hu : u ^ 2 = (A.withBeta β).shat k ε)
    (hz : ∀ n, (A.withBeta β).z k n ^ 2 - (A.withBeta β).shat k ε ≠ 0) :
    (A.withBeta β).That k u = (((Real.pi : ℂ) / β) ^ 2)⁻¹ * ((orbF k c).card *
      ((A.gHF k ε β - A.gHF k ε A.β : ℝ) : ℂ)⁻¹ + A.rhoHF k ε c β) := by
  have hS := hasSum_That_of (A := A.withBeta β) h.μ_eq hu hz
  simp only [ghat_z_eq h.ε_eq h.μ_eq hβ] at hS
  have hsf : Summable (A.fHF k ε β) := by
    have := hS.summable.mul_left (((Real.pi : ℂ) / β) ^ 2)
    simpa only [← mul_assoc, mul_inv_cancel₀ (pi_div_sq_ne hβ), one_mul] using this
  rw [← hS.tsum_eq, tsum_mul_left]
  congr 1
  have h1 : Summable fun l ↦ if l ∈ orbF k c then A.fHF k ε β l else 0 :=
    summable_of_ne_finset_zero (s := orbF k c) fun l hl ↦ ite_eq_right hl
  have h2 : Summable fun l ↦ if l ∈ orbF k c then 0 else A.fHF k ε β l := by
    convert hsf.sub h1 using 1
    funext l
    split_ifs <;> simp
  rw [show A.fHF k ε β = fun l ↦ (if l ∈ orbF k c then A.fHF k ε β l else 0) +
      (if l ∈ orbF k c then 0 else A.fHF k ε β l) by funext l; split_ifs <;> simp,
    h1.tsum_add h2, tsum_eq_sum (s := orbF k c) fun l hl ↦ ite_eq_right hl, rhoHF]
  congr 1
  rw [Finset.sum_congr rfl fun l hl ↦ by rw [ite_eq_left hl, h.fHF_eq_of_mem β hl],
    Finset.sum_const, nsmul_eq_mul]

/-- At a lattice zero, `D̂ ≠ 0` at every `β ≠ β₀` near `β₀`: by `HFLZ.That_eq`,
`(γ - γ(β₀))D̂ = (γ - γ(β₀))β - 2εa(β/π)²(r + (γ - γ(β₀))rhoHF β)`, which tends to
`-2εar(β₀/π)² ≠ 0` by `HFLZ.tendsto_rhoHF`. -/
theorem HFLZ.eventually_Dhat_ne (h : A.HFLZ k ε c) :
    ∀ᶠ β in 𝓝[≠] A.β, ∀ u, u ^ 2 = (A.withBeta β).shat k ε →
      (A.withBeta β).Dhat k ε u ≠ 0 := by
  set t₀ := ∑' l, A.fHF k ε A.β l
  set Φ : ℝ × ℂ → ℂ := fun p ↦ ((A.gHF k ε p.1 - A.gHF k ε A.β : ℝ) : ℂ) * p.1 -
    2 * ε * A.a * (((Real.pi : ℂ) / p.1) ^ 2)⁻¹ *
      ((orbF k c).card + ((A.gHF k ε p.1 - A.gHF k ε A.β : ℝ) : ℂ) * p.2)
  have hc : ContinuousAt Φ (A.β, t₀) := by
    have hX : ContinuousAt (fun p : ℝ × ℂ ↦ ((A.gHF k ε p.1 - A.gHF k ε A.β : ℝ) : ℂ)) (A.β, t₀) :=
      continuous_ofReal.continuousAt.comp_of_eq
        (((A.analyticAt_gHF k A.β).continuousAt.comp_of_eq continuousAt_fst rfl).sub
          continuousAt_const) rfl
    have hB : ContinuousAt (fun p : ℝ × ℂ ↦ (p.1 : ℂ)) (A.β, t₀) :=
      (continuous_ofReal.comp continuous_fst).continuousAt
    have hP : ContinuousAt (fun p : ℝ × ℂ ↦ (((Real.pi : ℂ) / p.1) ^ 2)⁻¹) (A.β, t₀) :=
      ((continuousAt_const.div hB (ofReal_ne_zero.mpr A.β_pos.ne')).pow 2).inv₀
        (pi_div_sq_ne A.β_pos)
    exact (hX.mul hB).sub ((continuousAt_const.mul hP).mul
      (continuousAt_const.add (hX.mul continuousAt_snd)))
  have hlim : Tendsto (fun β ↦ Φ (β, A.rhoHF k ε c β)) (𝓝[≠] A.β) (𝓝 (Φ (A.β, t₀))) :=
    hc.tendsto.comp ((tendsto_nhdsWithin_of_tendsto_nhds tendsto_id).prodMk_nhds h.tendsto_rhoHF)
  have hL : Φ (A.β, t₀) ≠ 0 := by
    have hε : ε ≠ 0 := by rcases h.ε_eq with hε | hε <;> rw [hε] <;> norm_num
    have hr : ((orbF k c).card : ℂ) ≠ 0 :=
      Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr ⟨c, by simp [orbF]⟩)
    simp only [Φ, sub_self, ofReal_zero, zero_mul, zero_sub, add_zero, neg_ne_zero]
    exact mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero hε) h.a_ne)
      (inv_ne_zero (pi_div_sq_ne A.β_pos))) hr
  filter_upwards [hlim.eventually_ne hL, h.eventually_gHF_ne, h.eventually_z_sq_ne,
    nhdsWithin_le_nhds (eventually_gt_nhds A.β_pos)] with β hΦ hG hz hβ u hu hD
  apply hΦ
  have hG' : ((A.gHF k ε β - A.gHF k ε A.β : ℝ) : ℂ) ≠ 0 := ofReal_ne_zero.mpr (sub_ne_zero.mpr hG)
  rw [← mul_zero ((A.gHF k ε β - A.gHF k ε A.β : ℝ) : ℂ), ← hD, Dhat, h.That_eq hβ hu hz,
    A.withBeta_β hβ, withBeta_a]
  simp only [Φ]
  field_simp

variable (A k ε) in
/-- A square root `û(β)` of `ŝ` at `A.withBeta β`, as in Corollary "half filling". -/
noncomputable def uBeta (β : ℝ) : ℂ := (A.withBeta β).shat k ε ^ (2⁻¹ : ℂ)

theorem uBeta_sq (β : ℝ) : A.uBeta k ε β ^ 2 = (A.withBeta β).shat k ε := by
  simp [uBeta]

/-- **The assumptions of Corollary "half filling" hold near a lattice zero**: at `z_c² = ŝ`, in
`β` at fixed `U`, `k` and `μ = 0`, they hold at every `β ≠ β₀` near `β₀`, with `û = uBeta`. -/
theorem HFLZ.eventually_hfHyp (h : A.HFLZ k ε c) :
    ∀ᶠ β in 𝓝[≠] A.β, (A.withBeta β).HFHyp k ε (A.uBeta k ε β) := by
  filter_upwards [h.eventually_z_sq_ne, nhdsWithin_le_nhds h.eventually_odd_ne,
    h.eventually_Dhat_ne] with β hz hodd hD
  exact ⟨h.ε_eq, h.a_ne, h.μ_eq, hz, hodd, uBeta_sq β, hD _ (uBeta_sq β)⟩

/-- **Item 4 of Proposition "special" as a limit in `β`**: at a lattice zero `z_c² = ŝ` of
Corollary "half filling", `gammaHF → gammaHFlz` as `β → β₀`, `β ≠ β₀`, at fixed `U`, `k` and
`μ = 0` (`𝓝[≠] β₀` is not empty). -/
theorem HFLZ.tendsto_gammaHF (h : A.HFLZ k ε c) (n m : ℤ) :
    Tendsto (fun β ↦ (A.withBeta β).gammaHF k ε (A.uBeta k ε β) n m) (𝓝[≠] A.β)
      (𝓝 (A.gammaHFlz k ε c n m)) := by
  have h₀ : (A.withBeta A.β).HFLZ k ε c := A.withBeta_self.symm ▸ h
  have hb : ContinuousAt (fun β : ℝ ↦ if 0 < β then β else A.β) A.β :=
    continuousAt_id.congr ((eventually_gt_nhds A.β_pos).mono fun β hβ ↦ (ite_eq_left hβ).symm)
  have := tendsto_gammaHF_of_HFLZ (U := fun _ ↦ A.U) (β := fun β ↦ if 0 < β then β else A.β)
    (μ := fun _ ↦ A.μ) (hβ := fun β ↦ (A.withBeta β).β_pos) k continuousAt_const hb
    continuousAt_const nhdsWithin_le_nhds ε h₀ h.eventually_hfHyp n m
  change Tendsto (fun β ↦ (A.withBeta β).gammaHF k ε (A.uBeta k ε β) n m) _
    (𝓝 ((A.withBeta A.β).gammaHFlz k ε c n m)) at this
  rwa [withBeta_self] at this

end Params

end HubbardAtom
