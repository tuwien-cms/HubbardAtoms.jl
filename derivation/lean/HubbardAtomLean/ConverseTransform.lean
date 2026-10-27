import HubbardAtomLean.GammaContinuity
import HubbardAtomLean.SpectralLink

/-!
# The converse of the spectral identity: the transform identity

For the density and magnetic channels, `U ≠ 0`, `μ ≠ 0`, and a bounded `v` with `χ_r v = ℓv`, the Cauchy
transforms `v̂(s) = ∑ₙ vₙg_s(n)` and `ψ̂(s) = ∑ₙ f_r(zₙ)vₙg_s(n)` (`vhat`, `psihat`, with
`s = u²`) satisfy, off the lattice,

  `[ℓ - ℓ(s)]v̂(s) = -β[1 - (εU/β)𝒯(s)]ψ̂(s)`, `ℓ(s) = -εU ∑ₙ f_r(zₙ)g_s(n)`

(`Params.eigen_transform`), with `1 - (εU/β)𝒯 = Φ_k/D_k` (`eigen_transform_Phi`). So at a root of
`Φ_k` with `ℓ(s) ≠ ℓ`, `v̂(s) = 0` (`vhat_eq_zero_of_Phi`). The proof sums
`vₙχ_r,nm g_s(m)` over `n` and `m` in both orders: by rows with the spectral identity
(`hasSum_chiR_mul_g`), by columns with `χ_r v = ℓv` and the symmetry of `χ_r`. The double sum
converges absolutely by the bound `|χ_r,nm| ≤ C|zₙ|⁻²(|zₘ|⁻² + δₙₘ)` (`norm_chiR_le_majorant`).
-/

open Complex Filter

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- The Cauchy transform `v̂(s) = ∑ₙ vₙg_s(n)`, `s = u²`. -/
noncomputable def vhat (v : ℤ → ℂ) (u : ℂ) : ℂ := ∑' n, v n * A.gs k u n

/-- The Cauchy transform `ψ̂(s) = ∑ₙ f_r(zₙ)vₙg_s(n)`, `s = u²`. -/
noncomputable def psihat (v : ℤ → ℂ) (u : ℂ) : ℂ := ∑' n, A.fR k ε (A.z k n) * v n * A.gs k u n

/-- `|χ_r,nm| ≤ C|zₙ|⁻²(|zₘ|⁻² + δₙₘ)` for `μ ≠ 0`, by `norm_chiR_le` and
`norm_chiR_diag_le` (`|zₙ| ≥ |μ|`). -/
theorem norm_chiR_le_majorant (hμ : A.μ ≠ 0) : ∃ C, ∀ n m, ‖A.chiR k ε n m‖ ≤
    C * (‖A.z k n‖⁻¹ ^ 2 * (‖A.z k m‖⁻¹ ^ 2 + if n = m then 1 else 0)) := by
  obtain ⟨c₁, hc₁⟩ := A.norm_chiR_le k ε hμ
  obtain ⟨c₂, hc₂⟩ := A.norm_chiR_diag_le k ε hμ
  have hμ' := abs_pos.mpr hμ
  have hβ := A.β_pos
  have hz0 := norm_pos_iff.mpr (A.z_ne_zero k hμ 0)
  have hz1 := norm_pos_iff.mpr (A.z_ne_zero k hμ 1)
  have hc₁0 : 0 ≤ c₁ := by
    simpa using (le_div_iff₀ (by positivity)).mp ((norm_nonneg _).trans (hc₁ 0 1 (by norm_num)))
  have hc₂0 : 0 ≤ c₂ := by
    simpa using (le_div_iff₀ (by positivity)).mp ((norm_nonneg _).trans (hc₂ 0))
  refine ⟨c₁ + c₂ / |A.μ| + A.β, fun n m ↦ ?_⟩
  have hz := norm_pos_iff.mpr (A.z_ne_zero k hμ n)
  have hzμ := A.abs_mu_le_norm_z k n
  by_cases hnm : n = m
  · subst hnm
    rw [ite_eq_left rfl]
    -- `|χ_nn| ≤ |χ_nn + β/zₙ²| + β/|zₙ|² ≤ (c₂/|μ| + β)|zₙ|⁻²`
    have h1 : ‖A.chiR k ε n n‖ ≤ c₂ / ‖A.z k n‖ ^ 3 + A.β * ‖A.z k n‖⁻¹ ^ 2 := by
      have := norm_sub_le (A.chiR k ε n n + A.β / A.z k n ^ 2) (A.β / A.z k n ^ 2)
      rw [add_sub_cancel_right, norm_div, norm_pow, norm_real, Real.norm_of_nonneg hβ.le] at this
      refine this.trans (add_le_add (hc₂ n) (le_of_eq ?_))
      rw [inv_pow, div_eq_mul_inv]
    have h2 : c₂ / ‖A.z k n‖ ^ 3 ≤ c₂ / |A.μ| * ‖A.z k n‖⁻¹ ^ 2 :=
      calc c₂ / ‖A.z k n‖ ^ 3 = c₂ * ‖A.z k n‖⁻¹ ^ 2 / ‖A.z k n‖ := by field_simp
        _ ≤ c₂ * ‖A.z k n‖⁻¹ ^ 2 / |A.μ| :=
          div_le_div_of_nonneg_left (by positivity) hμ' hzμ
        _ = c₂ / |A.μ| * ‖A.z k n‖⁻¹ ^ 2 := by ring
    have h3 : 0 ≤ ‖A.z k n‖⁻¹ ^ 2 := by positivity
    nlinarith [mul_nonneg hc₁0 (mul_nonneg h3 (add_nonneg h3 zero_le_one)),
      mul_nonneg (add_nonneg (div_nonneg hc₂0 hμ'.le) hβ.le) (mul_nonneg h3 h3)]
  · rw [ite_eq_right hnm, add_zero]
    have hv := norm_pos_iff.mpr (A.z_ne_zero k hμ m)
    have h := hc₁ n m hnm
    rw [inv_pow, inv_pow, ← mul_inv, ← div_eq_mul_inv] at *
    refine h.trans (div_le_div_of_nonneg_right ?_ (by positivity))
    linarith [div_nonneg hc₂0 hμ'.le]

/-- **The transform identity**: for a bounded `v` with `χ_r v = ℓv`, off the lattice,
`[ℓ - ℓ(s)]v̂(s) = -β[1 - (εU/β)𝒯(s)]ψ̂(s)` with `ℓ(s) = -εU ∑ₙ f_r(zₙ)g_s(n)`. -/
theorem eigen_transform (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {v : ℤ → ℂ} {M : ℝ} (hv : ∀ n, ‖v n‖ ≤ M) {ℓ : ℂ}
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) (ℓ * v n)) {u : ℂ}
    (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) :
    (ℓ + ε * A.U * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) * A.vhat k v u =
      -(A.β * (1 - ε * A.U / A.β * A.Tg k u)) * A.psihat k ε v u := by
  have hM : 0 ≤ M := (norm_nonneg _).trans (hv 0)
  have hg := (A.hasSum_g k hu hu').summable
  have hgn : Summable fun n ↦ ‖A.gs k u n‖ := summable_norm_iff.mpr hg
  have hfg := A.summable_fR_mul_gs k ε hε ha hμ hu hu'
  obtain ⟨G, hG⟩ := hg.tendsto_cofinite_zero.norm.bddAbove_range_of_cofinite
  have hG' : ∀ m, ‖A.gs k u m‖ ≤ G := fun m ↦ hG ⟨m, rfl⟩
  -- the summabilities of `v̂` and `ψ̂`
  have hvg : Summable fun n ↦ v n * A.gs k u n :=
    Summable.of_norm_bounded (hgn.mul_left M) fun n ↦ by
      rw [norm_mul]
      exact mul_le_mul_of_nonneg_right (hv n) (norm_nonneg _)
  have hfvg : Summable fun n ↦ A.fR k ε (A.z k n) * v n * A.gs k u n :=
    Summable.of_norm_bounded ((summable_norm_iff.mpr hfg).mul_left M) fun n ↦ by
      rw [show A.fR k ε (A.z k n) * v n * A.gs k u n =
        v n * (A.fR k ε (A.z k n) * A.gs k u n) by ring, norm_mul]
      exact mul_le_mul_of_nonneg_right (hv n) (norm_nonneg _)
  -- the double sum converges absolutely
  obtain ⟨C, hC⟩ := A.norm_chiR_le_majorant k ε hμ
  set f : ℤ → ℤ → ℂ := fun n m ↦ v n * (A.chiR k ε n m * A.gs k u m)
  have hf : Summable (Function.uncurry f) := by
    refine Summable.of_norm_bounded (((A.summable_majorant k hμ).mul_left (M * C * G))) fun p ↦ ?_
    simp only [Function.uncurry, f, norm_mul]
    have h1 := hv p.1
    have h2 := hC p.1 p.2
    have h3 := hG' p.2
    have h4 : 0 ≤ C * (‖A.z k p.1‖⁻¹ ^ 2 * (‖A.z k p.2‖⁻¹ ^ 2 + if p.1 = p.2 then 1 else 0)) :=
      (norm_nonneg _).trans h2
    calc ‖v p.1‖ * (‖A.chiR k ε p.1 p.2‖ * ‖A.gs k u p.2‖)
        ≤ M * ((C * (‖A.z k p.1‖⁻¹ ^ 2 * (‖A.z k p.2‖⁻¹ ^ 2 + if p.1 = p.2 then 1 else 0))) *
          G) := by gcongr
      _ = _ := by ring
  -- by rows: the spectral identity
  have hrow : ∀ n, HasSum (fun m ↦ f n m) (v n *
      ((-(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) * A.gs k u n -
        A.β * (1 - ε * A.U / A.β * A.Tg k u) * A.fR k ε (A.z k n) * A.gs k u n)) :=
    fun n ↦ (A.hasSum_chiR_mul_g k ε hε ha hμ hu hu' n).mul_left (v n)
  -- by columns: `χ_r v = ℓv` and the symmetry
  have hcol : ∀ m, HasSum (fun n ↦ f n m) (ℓ * v m * A.gs k u m) := fun m ↦ by
    convert (hχ m).mul_right (A.gs k u m) using 1
    funext n
    simp only [f, A.chiR_symm k ε hμ n m]
    ring
  have e1 : ∑' n, ∑' m, f n m = ∑' n, v n *
      ((-(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) * A.gs k u n -
        A.β * (1 - ε * A.U / A.β * A.Tg k u) * A.fR k ε (A.z k n) * A.gs k u n) :=
    tsum_congr fun n ↦ (hrow n).tsum_eq
  have e2 : ∑' m, ∑' n, f n m = ∑' m, ℓ * v m * A.gs k u m := tsum_congr fun m ↦ (hcol m).tsum_eq
  have hL : HasSum (fun m ↦ ℓ * v m * A.gs k u m) (ℓ * A.vhat k v u) := by
    simpa only [mul_assoc, vhat] using hvg.hasSum.mul_left ℓ
  have hR : HasSum (fun n ↦ v n *
      ((-(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) * A.gs k u n -
        A.β * (1 - ε * A.U / A.β * A.Tg k u) * A.fR k ε (A.z k n) * A.gs k u n))
      ((-(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) * A.vhat k v u -
        A.β * (1 - ε * A.U / A.β * A.Tg k u) * A.psihat k ε v u) := by
    convert (hvg.hasSum.mul_left (-(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m)).sub
      (hfvg.hasSum.mul_left (A.β * (1 - ε * A.U / A.β * A.Tg k u))) using 1
    · funext n
      ring
    · simp only [vhat, psihat]
  have h := hf.tsum_comm
  rw [e1, e2, hL.tsum_eq, hR.tsum_eq] at h
  linear_combination h

/-- The transform identity in the form of the issue: `[ℓ - ℓ(s)]v̂(s) = -β(Φ_k/D_k)(s)ψ̂(s)`, for
`u ≠ 0` (`one_sub_Tg`; `D_k(u) ≠ 0` off the lattice). -/
theorem eigen_transform_Phi (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {v : ℤ → ℂ} {M : ℝ} (hv : ∀ n, ‖v n‖ ≤ M) {ℓ : ℂ}
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) (ℓ * v n)) {u : ℂ}
    (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) (hu0 : u ≠ 0) :
    (ℓ + ε * A.U * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) * A.vhat k v u =
      -(A.β * (A.Phik k ε u / A.Dk k u)) * A.psihat k ε v u := by
  rw [← A.one_sub_Tg k ε hu0 (A.Dk_ne_zero_of_off k hu hu')]
  exact A.eigen_transform k ε hε ha hμ hv hχ hu hu'

/-- **`v̂` vanishes at the other roots of `Φ_k`**: at `Φ_k(s) = 0` with `ℓ(s) ≠ ℓ`, `v̂(s) = 0`. -/
theorem vhat_eq_zero_of_Phi (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {v : ℤ → ℂ} {M : ℝ} (hv : ∀ n, ‖v n‖ ≤ M) {ℓ : ℂ}
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) (ℓ * v n)) {u : ℂ}
    (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) (hΦ : A.Phik k ε u = 0)
    (hℓ : ℓ ≠ -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) :
    A.vhat k v u = 0 := by
  have h := A.eigen_transform k ε hε ha hμ hv hχ hu hu'
  rw [A.one_sub_Tg_of_Phi k ε hΦ hu hu', mul_zero, neg_zero, zero_mul] at h
  refine (mul_eq_zero.mp h).resolve_left fun h0 ↦ hℓ ?_
  linear_combination h0

end Params

end HubbardAtom
