import HubbardAtomLean.ChiOperator
import HubbardAtomLean.GammaContinuity

/-!
# Continuity of the operator vertex along a family

The second half of item 6(d), for a general `χ` (`ChiOperator.lean`): along a family
`x ↦ (L x, X x, b x)` of lattices, matrices and bubbles over a topological space, `Γ` is
continuous at `x₀` wherever `1 + 𝒦` is invertible there, provided

* the bounds of item 6(a) hold near `x₀`;
* `β`, the lattice points `zₙ`, the bubble `bₙ` and the entries of `𝒦` are continuous at `x₀`;
* near `x₀`, `|𝒦_nm|² ≤ C|zₙ(x₀)|⁻²(|zₘ(x₀)|⁻² + δₙₘ)`, uniformly (`ContHyp`).

Then `𝒦` is continuous in the Hilbert–Schmidt norm (dominated convergence), hence in the operator
norm, and inversion is continuous at units. The proof is that of `GammaContinuity.lean` for the
density and magnetic channels, which also supplies the domination there; each field sector supplies
its own.

* `OpLattice.gammaOpFam`: `Γ` along the family, `0` where the bounds of 6(a) fail;
* `OpLattice.tendsto_tsum_norm_KX_sub_sq`: `∑ₙₘ |𝒦_nm(x) - 𝒦_nm(x₀)|² → 0`;
* `OpLattice.continuousAt_gammaOpFam`: **`Γ_nm` is continuous at `x₀`**.
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace OpLattice

open Params (single_apply_eq_kron inverse_apply_of_eq)

variable {P : Type*} [TopologicalSpace P] (L : P → OpLattice) (X : P → ℤ → ℤ → ℂ)
  (b : P → ℤ → ℂ)

open Classical in
/-- `Γ` along the family, `0` where the bounds of item 6(a) fail. -/
noncomputable def gammaOpFam (x : P) (n m : ℤ) : ℂ :=
  if h : (L x).ChiDecay (X x) ∧ (L x).BubbleDecay (b x) then
    (L x).gammaOpX (X x) (b x) h.1 h.2 n m
  else 0

omit [TopologicalSpace P] in
open Classical in
theorem gammaOpFam_eq {x : P} (hX : (L x).ChiDecay (X x)) (hb : (L x).BubbleDecay (b x))
    (n m : ℤ) : gammaOpFam L X b x n m = (L x).gammaOpX (X x) (b x) hX hb n m :=
  dite_eq_left (c := (L x).ChiDecay (X x) ∧ (L x).BubbleDecay (b x)) ⟨hX, hb⟩

/-- `Γ` does not depend on the proofs of the bounds, and only on the lattice up to equality. -/
theorem gammaOpX_congr {L₁ L₂ : OpLattice} (hL : L₁ = L₂) {X : ℤ → ℤ → ℂ}
    {b : ℤ → ℂ} (hX₁ : L₁.ChiDecay X) (hb₁ : L₁.BubbleDecay b) (hX₂ : L₂.ChiDecay X)
    (hb₂ : L₂.BubbleDecay b) (n m : ℤ) :
    L₁.gammaOpX X b hX₁ hb₁ n m = L₂.gammaOpX X b hX₂ hb₂ n m := by
  subst hL
  rfl

/-- `1 + 𝒦` does not depend on the proof of the bounds, and only on the lattice up to equality. -/
theorem onePlusKX_congr {L₁ L₂ : OpLattice} (hL : L₁ = L₂) {X : ℤ → ℤ → ℂ}
    (hX₁ : L₁.ChiDecay X) (hX₂ : L₂.ChiDecay X) : L₁.onePlusKX X hX₁ = L₂.onePlusKX X hX₂ := by
  subst hL
  rfl

/-- The hypotheses of `continuousAt_gammaOpFam` at `x₀`. -/
structure ContHyp (x₀ : P) : Prop where
  chi : ∀ᶠ x in 𝓝 x₀, (L x).ChiDecay (X x)
  bub : ∀ᶠ x in 𝓝 x₀, (L x).BubbleDecay (b x)
  β : ContinuousAt (fun x ↦ (L x).β) x₀
  z : ∀ n, ContinuousAt (fun x ↦ (L x).z n) x₀
  b : ∀ n, ContinuousAt (fun x ↦ b x n) x₀
  K : ∀ n m, ContinuousAt (fun x ↦ (L x).KX (X x) n m) x₀
  dom : ∃ C, ∀ᶠ x in 𝓝 x₀, ∀ n m, ‖(L x).KX (X x) n m‖ ^ 2 ≤
    C * (‖(L x₀).z n‖⁻¹ ^ 2 * (‖(L x₀).z m‖⁻¹ ^ 2 + if n = m then 1 else 0))

variable {L X b} {x₀ : P}

/-- **`𝒦` is continuous in the Hilbert–Schmidt norm**: `∑ₙₘ |𝒦_nm(x) - 𝒦_nm(x₀)|² → 0`, by
dominated convergence. -/
theorem tendsto_tsum_norm_KX_sub_sq (h : ContHyp L X b x₀) :
    Tendsto (fun x ↦ ∑' p : ℤ × ℤ,
      ‖(L x).KX (X x) p.1 p.2 - (L x₀).KX (X x₀) p.1 p.2‖ ^ 2) (𝓝 x₀) (𝓝 0) := by
  obtain ⟨C, hC⟩ := h.dom
  have hC₀ := hC.self_of_nhds
  have hsum := (L x₀).summable_majorant.mul_left (4 * C)
  have h' := tendsto_tsum_of_dominated_convergence (f := fun x (p : ℤ × ℤ) ↦
      ‖(L x).KX (X x) p.1 p.2 - (L x₀).KX (X x₀) p.1 p.2‖ ^ 2)
    (g := fun p ↦ ‖(L x₀).KX (X x₀) p.1 p.2 - (L x₀).KX (X x₀) p.1 p.2‖ ^ 2) hsum
    (fun p ↦ ((h.K p.1 p.2).sub_const _).norm.pow 2)
    (hC.mono fun x hx p ↦ by
      rw [Real.norm_of_nonneg (sq_nonneg _)]
      have h1 := hx p.1 p.2
      have h2 := hC₀ p.1 p.2
      have h3 := norm_sub_le ((L x).KX (X x) p.1 p.2) ((L x₀).KX (X x₀) p.1 p.2)
      have h4 : 0 ≤ ‖(L x₀).z p.1‖⁻¹ ^ 2 * (‖(L x₀).z p.2‖⁻¹ ^ 2 + if p.1 = p.2 then 1 else 0) := by
        split_ifs <;> positivity
      nlinarith [pow_le_pow_left₀ (norm_nonneg _) h3 2,
        sq_nonneg (‖(L x).KX (X x) p.1 p.2‖ - ‖(L x₀).KX (X x₀) p.1 p.2‖)])
  simpa using h'

/-- `dₙ` is continuous along the family. -/
theorem ContHyp.continuousAt_Dhalf (h : ContHyp L X b x₀) (n : ℤ) :
    ContinuousAt (fun x ↦ (L x).Dhalf n) x₀ := by
  have hz := h.z n
  have hβ := h.β
  have hz0 := (L x₀).z_ne_zero n
  simp only [Dhalf]
  fun_prop (disch := assumption)

/-- `tₙ = -zₙ²bₙ/β` is continuous along the family. -/
theorem ContHyp.continuousAt_tX (h : ContHyp L X b x₀) (n : ℤ) :
    ContinuousAt (fun x ↦ (L x).tX (b x) n) x₀ := by
  have hz := h.z n
  have hβ := h.β
  have hbn := h.b n
  have hβ0 : ((L x₀).β : ℂ) ≠ 0 := ofReal_ne_zero.mpr (L x₀).β_pos.ne'
  simp only [tX]
  fun_prop (disch := assumption)

/-- **`Γ` is continuous along the family** at `x₀`, wherever `1 + 𝒦` is invertible at `x₀`:
`𝒦` is continuous in norm (`tendsto_tsum_norm_KX_sub_sq`, `norm_hsOp_le`), and inversion is
continuous at units. -/
theorem continuousAt_gammaOpFam (h : ContHyp L X b x₀) (hX₀ : (L x₀).ChiDecay (X x₀))
    (hu : IsUnit ((L x₀).onePlusKX (X x₀) hX₀)) (n m : ℤ) :
    ContinuousAt (fun x ↦ gammaOpFam L X b x n m) x₀ := by
  classical
  have hb₀ := h.bub.self_of_nhds
  let T : P → (ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ)) := fun x ↦
    if hx : (L x).ChiDecay (X x) then (L x).onePlusKX (X x) hx else 1
  have hTx : ∀ x (hx : (L x).ChiDecay (X x)), T x = (L x).onePlusKX (X x) hx :=
    fun x hx ↦ dite_eq_left hx
  have hT : Tendsto T (𝓝 x₀) (𝓝 (T x₀)) := by
    rw [tendsto_iff_norm_sub_tendsto_zero]
    have ht := (tendsto_tsum_norm_KX_sub_sq h).sqrt
    rw [Real.sqrt_zero] at ht
    refine squeeze_zero' (Eventually.of_forall fun _ ↦ norm_nonneg _) ?_ ht
    filter_upwards [h.chi] with x hx
    have hs := (L x).summable_KX_sq (X x) hx
    have hs₀ := (L x₀).summable_KX_sq (X x₀) hX₀
    have hd : Summable fun p : ℤ × ℤ ↦ ‖(L x).KX (X x) p.1 p.2 -
        (L x₀).KX (X x₀) p.1 p.2‖ ^ 2 :=
      ((hs.mul_left 2).add (hs₀.mul_left 2)).of_nonneg_of_le (fun _ ↦ by positivity) fun p ↦ by
        nlinarith [pow_le_pow_left₀ (norm_nonneg _) (norm_sub_le ((L x).KX (X x) p.1 p.2)
          ((L x₀).KX (X x₀) p.1 p.2)) 2,
          sq_nonneg (‖(L x).KX (X x) p.1 p.2‖ - ‖(L x₀).KX (X x₀) p.1 p.2‖)]
    rw [hTx x hx, hTx x₀ hX₀]
    unfold onePlusKX
    rw [add_sub_add_left_eq_sub, hsOp_sub ((L x).KX (X x)) hs ((L x₀).KX (X x₀)) hs₀ hd]
    exact norm_hsOp_le (fun n m ↦ (L x).KX (X x) n m - (L x₀).KX (X x₀) n m) hd
  have hinv : Tendsto (fun x ↦ Ring.inverse (T x)) (𝓝 x₀) (𝓝 (Ring.inverse (T x₀))) := by
    have hu' : IsUnit (T x₀) := (hTx x₀ hX₀).symm ▸ hu
    have h' := NormedRing.inverse_continuousAt hu'.unit
    rw [hu'.unit_spec] at h'
    exact h'.tendsto.comp hT
  have he := ((lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 n).continuous.tendsto _).comp
    (((ContinuousLinearMap.apply ℂ _ (lp.single 2 m (1 : ℂ))).continuous.tendsto _).comp hinv)
  have hβc : ContinuousAt (fun x ↦ ((L x).β : ℂ)) x₀ := by
    have := h.β
    fun_prop
  have hlim := ((hβc.tendsto.pow 2).div ((h.continuousAt_Dhalf n).tendsto.mul
    (h.continuousAt_Dhalf m).tendsto) (mul_ne_zero ((L x₀).Dhalf_ne_zero n)
      ((L x₀).Dhalf_ne_zero m))).mul
    (he.sub ((tendsto_const_nhds : Tendsto (fun _ : P ↦ kron n m) (𝓝 x₀) _).div
      (h.continuousAt_tX m).tendsto ((L x₀).tX_ne_zero (b x₀) hb₀ m)))
  refine ContinuousAt.congr (f := fun x ↦ ((L x).β : ℂ) ^ 2 / ((L x).Dhalf n * (L x).Dhalf m) *
      (Ring.inverse (T x) (lp.single 2 m 1) n - kron n m / (L x).tX (b x) m)) hlim ?_
  filter_upwards [h.chi, h.bub] with x hx hbx
  rw [gammaOpFam_eq L X b hx hbx, gammaOpX, _root_.sub_apply, lp.coeFn_sub, Pi.sub_apply,
    (L x).inverse_onePlusK0X_single (b x) hbx, hTx x hx]

/-! ### Uniform bounds near `x₀` -/

omit [TopologicalSpace P] in
/-- **Bounded near `x₀`, uniformly in `n`**: if `q` is jointly continuous at `(x₀, 0)`, each
`x ↦ q x (1/zₙ(x))` is continuous at `x₀`, and `1/zₙ(x) → 0` as `|n| → ∞` uniformly near `x₀`,
then `q x (1/zₙ(x))` is bounded near `x₀`, uniformly in `n`: the large `|n|` by continuity at
`(x₀, 0)`, the finitely many others one by one. -/
theorem eventually_bdd_of_inv_z [TopologicalSpace P] {zf : P → ℤ → ℂ} {q : P → ℂ → ℂ}
    (hq : ContinuousAt (fun p : P × ℂ ↦ q p.1 p.2) (x₀, 0))
    (hn : ∀ n, ContinuousAt (fun x ↦ q x (zf x n)⁻¹) x₀)
    (hu : ∀ ε > 0, ∃ N : ℕ, ∀ᶠ x in 𝓝 x₀, ∀ n : ℤ, (N : ℤ) ≤ |n| → ‖(zf x n)⁻¹‖ < ε) :
    ∃ C, ∀ᶠ x in 𝓝 x₀, ∀ n, ‖q x (zf x n)⁻¹‖ ≤ C := by
  have h0 : ∀ᶠ p : P × ℂ in 𝓝 (x₀, 0), ‖q p.1 p.2‖ < ‖q x₀ 0‖ + 1 :=
    hq.norm.eventually (gt_mem_nhds (lt_add_one _))
  rw [nhds_prod_eq] at h0
  obtain ⟨pa, hpa, pb, hpb, hab⟩ := Filter.eventually_prod_iff.mp h0
  obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.mp hpb
  obtain ⟨N, hN⟩ := hu δ hδ
  set I := Finset.Ioo (-(N : ℤ)) N
  have hfin : ∀ᶠ x in 𝓝 x₀, ∀ n ∈ I, ‖q x (zf x n)⁻¹‖ < ‖q x₀ (zf x₀ n)⁻¹‖ + 1 :=
    (Filter.eventually_all_finset I).mpr fun n _ ↦
      (hn n).norm.eventually (gt_mem_nhds (lt_add_one _))
  refine ⟨‖q x₀ 0‖ + 1 + ∑ n ∈ I, (‖q x₀ (zf x₀ n)⁻¹‖ + 1), ?_⟩
  filter_upwards [hpa, hN, hfin] with x hxa hxN hxf n
  have hs : 0 ≤ ∑ n ∈ I, (‖q x₀ (zf x₀ n)⁻¹‖ + 1) :=
    Finset.sum_nonneg fun _ _ ↦ by positivity
  by_cases hn' : (N : ℤ) ≤ |n|
  · have h := hab hxa (hball (y := (zf x n)⁻¹) (by rw [dist_zero_right]; exact hxN n hn'))
    dsimp only at h
    linarith
  · have hmem : n ∈ I := by
      rw [Finset.mem_Ioo]
      exact abs_lt.mp (not_le.mp hn')
    have h1 := hxf n hmem
    have h2 := Finset.single_le_sum (f := fun n ↦ ‖q x₀ (zf x₀ n)⁻¹‖ + 1)
      (fun _ _ ↦ by positivity) hmem
    have := norm_nonneg (q x₀ 0)
    linarith

/-- **The domination of `ContHyp` from uniform bounds**: if, near `x₀`, the bounds of item 6(a)
hold with the same constants and `c|zₙ(x₀)|² ≤ |zₙ(x)|²`, and `β` is continuous, then near `x₀`,
`|𝒦_nm|² ≤ C|zₙ(x₀)|⁻²(|zₘ(x₀)|⁻² + δ_nm)`. -/
theorem dom_of_uniform {c₁ c₂ c : ℝ} (hc : 0 < c)
    (h₁ : ∀ᶠ x in 𝓝 x₀, ∀ n m, n ≠ m →
      ‖X x n m‖ ≤ c₁ / (‖(L x).z n‖ ^ 2 * ‖(L x).z m‖ ^ 2))
    (h₂ : ∀ᶠ x in 𝓝 x₀, ∀ n, ‖X x n n + (L x).β / (L x).z n ^ 2‖ ≤ c₂ / ‖(L x).z n‖ ^ 3)
    (hβ : ContinuousAt (fun x ↦ (L x).β) x₀)
    (hz : ∀ᶠ x in 𝓝 x₀, ∀ n, c * ‖(L x₀).z n‖ ^ 2 ≤ ‖(L x).z n‖ ^ 2) :
    ∃ C, ∀ᶠ x in 𝓝 x₀, ∀ n m, ‖(L x).KX (X x) n m‖ ^ 2 ≤
      C * (‖(L x₀).z n‖⁻¹ ^ 2 * (‖(L x₀).z m‖⁻¹ ^ 2 + if n = m then 1 else 0)) := by
  have hβ₀ := (L x₀).β_pos
  have hβev : ∀ᶠ x in 𝓝 x₀, (L x₀).β / 2 < (L x).β :=
    hβ.eventually (lt_mem_nhds (half_lt_self hβ₀))
  refine ⟨4 * (c₁ ^ 2 + c₂ ^ 2) / (L x₀).β ^ 2 * (c⁻¹ * (c⁻¹ + 1)), ?_⟩
  filter_upwards [h₁, h₂, hz, hβev] with x hx₁ hx₂ hxz hxβ n m
  refine ((L x).norm_KX_sq_le_of (X x) hx₁ hx₂ n m).trans ?_
  have hβx := (L x).β_pos
  have hK : (c₁ / (L x).β) ^ 2 + (c₂ / (L x).β) ^ 2 ≤ 4 * (c₁ ^ 2 + c₂ ^ 2) / (L x₀).β ^ 2 := by
    rw [div_pow, div_pow, ← add_div, div_le_div_iff₀ (by positivity) (by positivity)]
    have h4 : (L x₀).β ^ 2 ≤ 4 * (L x).β ^ 2 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left h4 (add_nonneg (sq_nonneg c₁) (sq_nonneg c₂))]
  have hinv : ∀ j, ‖(L x).z j‖⁻¹ ^ 2 ≤ c⁻¹ * ‖(L x₀).z j‖⁻¹ ^ 2 := fun j ↦ by
    have hz0 := norm_pos_iff.mpr ((L x₀).z_ne_zero j)
    rw [inv_pow, inv_pow, ← mul_inv]
    exact inv_anti₀ (by positivity) (hxz j)
  have hδ : (0 : ℝ) ≤ if n = m then 1 else 0 := by split_ifs <;> norm_num
  have ha := hinv n
  have hb := hinv m
  have hn0 : 0 ≤ ‖(L x₀).z n‖⁻¹ ^ 2 := by positivity
  have hm0 : 0 ≤ ‖(L x₀).z m‖⁻¹ ^ 2 := by positivity
  have hc' : 0 < c⁻¹ := inv_pos.mpr hc
  calc ((c₁ / (L x).β) ^ 2 + (c₂ / (L x).β) ^ 2) *
        (‖(L x).z n‖⁻¹ ^ 2 * (‖(L x).z m‖⁻¹ ^ 2 + if n = m then 1 else 0))
      ≤ 4 * (c₁ ^ 2 + c₂ ^ 2) / (L x₀).β ^ 2 *
        (c⁻¹ * ‖(L x₀).z n‖⁻¹ ^ 2 * (c⁻¹ * ‖(L x₀).z m‖⁻¹ ^ 2 + if n = m then 1 else 0)) := by
        gcongr
    _ ≤ _ := by
        have key : c⁻¹ * ‖(L x₀).z n‖⁻¹ ^ 2 * (c⁻¹ * ‖(L x₀).z m‖⁻¹ ^ 2 +
              if n = m then 1 else 0) ≤
            c⁻¹ * (c⁻¹ + 1) * (‖(L x₀).z n‖⁻¹ ^ 2 * (‖(L x₀).z m‖⁻¹ ^ 2 +
              if n = m then 1 else 0)) := by
          nlinarith [mul_nonneg (mul_nonneg hc'.le hc'.le) (mul_nonneg hn0 hδ),
            mul_nonneg hc'.le (mul_nonneg hn0 hm0)]
        exact (mul_le_mul_of_nonneg_left key (by positivity)).trans_eq (by ring)

end OpLattice

/-! ### Lattices of `Params` along a family -/

namespace Params

variable {P : Type*} [TopologicalSpace P] {x₀ : P}

/-- `1/zₙ → 0` as `|n| → ∞`, uniformly near `x₀`, for a family of `Params` with `β` continuous:
`|zₙ| ≥ |Im zₙ| = |2n + 1 + k|π/β`. -/
theorem eventually_norm_inv_z_lt (A : P → Params) (k : ℤ)
    (hβ : ContinuousAt (fun x ↦ (A x).β) x₀) :
    ∀ ε > 0, ∃ N : ℕ, ∀ᶠ x in 𝓝 x₀, ∀ n : ℤ, (N : ℤ) ≤ |n| → ‖((A x).z k n)⁻¹‖ < ε := by
  intro ε hε
  have hπ := Real.pi_pos
  have hβ₀ := (A x₀).β_pos
  set B := 2 * (A x₀).β / (Real.pi * ε)
  have hB : 0 ≤ B := by positivity
  refine ⟨⌈B + |(k : ℝ)| + 1⌉₊, ?_⟩
  filter_upwards [hβ.eventually (gt_mem_nhds (lt_two_mul_self hβ₀))] with x hx n hn
  have hβx := (A x).β_pos
  have hN : B + |(k : ℝ)| + 1 ≤ (⌈B + |(k : ℝ)| + 1⌉₊ : ℝ) := Nat.le_ceil _
  have hn' : (⌈B + |(k : ℝ)| + 1⌉₊ : ℝ) ≤ |(n : ℝ)| := by exact_mod_cast hn
  -- `|2n + 1 + k| ≥ 2|n| - 1 - |k| > B`
  have hs : B < |(2 * n + 1 + k : ℝ)| := by
    have h1 : 2 * |(n : ℝ)| ≤ |(2 * n + 1 + k : ℝ)| + 1 + |(k : ℝ)| := by
      have := abs_add_le (2 * n + 1 + k : ℝ) (-(1 + k))
      rw [show (2 * n + 1 + k : ℝ) + -(1 + k) = 2 * n by ring, abs_mul, abs_two,
        abs_neg] at this
      have h1k := abs_add_le (1 : ℝ) k
      rw [abs_one] at h1k
      linarith
    have hk0 := abs_nonneg (k : ℝ)
    linarith
  have hs0 : 0 < |(2 * n + 1 + k : ℝ)| := hB.trans_lt hs
  have him : ((A x).z k n).im = (2 * n + 1 + k) * Real.pi / (A x).β := by
    rw [z_eq]
    simp
  have hz : |(2 * n + 1 + k : ℝ)| * Real.pi / (A x).β ≤ ‖(A x).z k n‖ := by
    have h := abs_im_le_norm ((A x).z k n)
    rwa [him, abs_div, abs_mul, abs_of_pos hπ, abs_of_pos hβx] at h
  rw [norm_inv]
  refine (inv_anti₀ (by positivity) hz).trans_lt ?_
  rw [inv_div, div_lt_iff₀ (mul_pos hs0 hπ)]
  have hB' : B * (Real.pi * ε) = 2 * (A x₀).β := by
    simp only [B]
    field_simp
  nlinarith [mul_lt_mul_of_pos_right hs (mul_pos hπ hε)]

/-- `|zₙ(x)|² ≥ |zₙ(x₀)|²/2` near `x₀`, for a family of `Params` with `μ`, `β` continuous and
`μ(x₀) ≠ 0`. -/
theorem eventually_norm_z_sq_ge (A : P → Params) (k : ℤ)
    (hμ : ContinuousAt (fun x ↦ (A x).μ) x₀) (hβ : ContinuousAt (fun x ↦ (A x).β) x₀)
    (hμ₀ : (A x₀).μ ≠ 0) :
    ∀ᶠ x in 𝓝 x₀, ∀ n, 1 / 2 * ‖(A x₀).z k n‖ ^ 2 ≤ ‖(A x).z k n‖ ^ 2 := by
  have hβ₀ := (A x₀).β_pos
  have hμ2 : 0 < (A x₀).μ ^ 2 := by positivity
  have hp1 : 1 / 2 * (A x₀).μ ^ 2 < (A x₀).μ ^ 2 := by linarith
  have hp2 : (A x₀).β ^ 2 < 2 * (A x₀).β ^ 2 := by nlinarith
  have h1 : ∀ᶠ x in 𝓝 x₀, 1 / 2 * (A x₀).μ ^ 2 < (A x).μ ^ 2 :=
    (hμ.pow 2).eventually (lt_mem_nhds hp1)
  have h2 : ∀ᶠ x in 𝓝 x₀, (A x).β ^ 2 < 2 * (A x₀).β ^ 2 :=
    (hβ.pow 2).eventually (gt_mem_nhds hp2)
  filter_upwards [h1, h2] with x hx1 hx2 n
  refine (A x).norm_z_sq_ge k (A x₀) hx1.le ?_ n
  have hβx := (A x).β_pos
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith

end Params

end HubbardAtom
