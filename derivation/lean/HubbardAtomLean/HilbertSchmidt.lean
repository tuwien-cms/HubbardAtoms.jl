import Mathlib.Analysis.InnerProductSpace.l2Space
import Mathlib.Analysis.Normed.Operator.Compact.Basic

/-!
# Hilbert–Schmidt matrices on `ℓ²(ℤ)`

A matrix `K : ℤ → ℤ → ℂ` with square-summable entries acts on `ℓ²(ℤ)` as a compact operator:

* `hsOp K hK`: `(Kv)ₙ = ∑ₘ Kₙₘvₘ` (`hsOp_apply`), with `‖Kv‖² ≤ ∑ₙ‖Kₙ·‖²‖v‖²`;
* `isCompactOperator_hsOp`: `hsOp K hK` is compact, as the norm limit of its restrictions to
  finitely many rows, which have finite rank;
* `hsOp_diag_apply`: a diagonal matrix `diagK c` acts by multiplication;
* `tsum_mul_one_add_hsOp_comm`: for a symmetric `K`, `∑ₙ xₙ((1 + K)c)ₙ = ∑ₙ cₙ((1 + K)x)ₙ`;
* `tendsto_hsOp_truncK`: the restrictions `truncK K s` of `K` to `s × s` converge in norm as `s`
  grows, by the Hilbert–Schmidt bound `norm_hsOp_le`.

Row `n` acts as the inner product with `m ↦ conj Kₙₘ`, so linearity and the Cauchy–Schwarz bound
come from `ℓ²(ℤ)`.
-/

open Filter Topology
open scoped lp ComplexConjugate

namespace HubbardAtom

/-- `v ∈ ℓ²` iff `∑ₙ ‖vₙ‖²` converges. -/
theorem memℓp_two_iff {f : ℤ → ℂ} : Memℓp f 2 ↔ Summable fun n ↦ ‖f n‖ ^ 2 := by
  rw [memℓp_gen_iff (by norm_num)]
  norm_num

/-- `‖v‖² = ∑ₙ ‖vₙ‖²`. -/
theorem lp_norm_sq_eq_tsum (v : ℓ²(ℤ, ℂ)) : ‖v‖ ^ 2 = ∑' n, ‖v n‖ ^ 2 := by
  have h := lp.norm_rpow_eq_tsum (p := 2) (by norm_num) v
  norm_num at h
  exact_mod_cast h

/-- A componentwise bound by a summable sequence bounds `‖v‖²`. -/
theorem lp_norm_sq_le {v : ℓ²(ℤ, ℂ)} {c : ℤ → ℝ} (hc : Summable c) (h : ∀ n, ‖v n‖ ^ 2 ≤ c n) :
    ‖v‖ ^ 2 ≤ ∑' n, c n := by
  rw [lp_norm_sq_eq_tsum]
  exact (memℓp_two_iff.mp (lp.memℓp v)).tsum_le_tsum h hc

/-- `x² ≤ Cy²` gives `x ≤ √C y`, for `x, y, C ≥ 0`. -/
theorem le_sqrt_mul_of_sq_le {x y C : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hC : 0 ≤ C)
    (h : x ^ 2 ≤ C * y ^ 2) : x ≤ √C * y := by
  rw [← Real.sqrt_sq hx, ← Real.sqrt_sq hy, ← Real.sqrt_mul hC]
  exact Real.sqrt_le_sqrt h

/-- Cauchy–Schwarz, squared: `|⟪x, y⟫|² ≤ ‖x‖²‖y‖²`. -/
theorem norm_inner_sq_le (x y : ℓ²(ℤ, ℂ)) : ‖inner ℂ x y‖ ^ 2 ≤ ‖x‖ ^ 2 * ‖y‖ ^ 2 := by
  rw [← mul_pow]
  exact pow_le_pow_left₀ (norm_nonneg _) (norm_inner_le_norm _ _) 2

variable (K : ℤ → ℤ → ℂ) (hK : Summable fun p : ℤ × ℤ ↦ ‖K p.1 p.2‖ ^ 2)

/-- Row `n` of `K`, conjugated, as an element of `ℓ²`. -/
noncomputable def hsRow (n : ℤ) : ℓ²(ℤ, ℂ) :=
  ⟨fun m ↦ conj (K n m), memℓp_two_iff.mpr (by simpa using hK.prod_factor n)⟩

/-- `‖Kₙ·‖²` is summable over the rows. -/
theorem summable_norm_hsRow_sq : Summable fun n ↦ ‖hsRow K hK n‖ ^ 2 := by
  have h := ((summable_prod_of_nonneg fun _ ↦ by positivity).mp hK).2
  refine h.congr fun n ↦ ?_
  rw [lp_norm_sq_eq_tsum]
  simp [hsRow]

/-- `Kv ∈ ℓ²`, with `|(Kv)ₙ| ≤ ‖Kₙ·‖‖v‖`. -/
theorem memℓp_hsRow_inner (v : ℓ²(ℤ, ℂ)) :
    Memℓp (fun n ↦ inner ℂ (hsRow K hK n) v) 2 := by
  refine memℓp_two_iff.mpr (((summable_norm_hsRow_sq K hK).mul_right (‖v‖ ^ 2)).of_nonneg_of_le
    (fun _ ↦ by positivity) fun n ↦ norm_inner_sq_le _ _)

/-- The operator of a Hilbert–Schmidt matrix. -/
noncomputable def hsOp : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  LinearMap.mkContinuous
    { toFun := fun v ↦ ⟨fun n ↦ inner ℂ (hsRow K hK n) v, memℓp_hsRow_inner K hK v⟩
      map_add' := fun v w ↦ by ext n; exact inner_add_right _ _ _
      map_smul' := fun c v ↦ by ext n; exact inner_smul_right _ _ _ }
    √(∑' n, ‖hsRow K hK n‖ ^ 2) fun v ↦ by
      refine le_sqrt_mul_of_sq_le (norm_nonneg _) (norm_nonneg _)
        (tsum_nonneg fun _ ↦ by positivity) ?_
      exact (lp_norm_sq_le ((summable_norm_hsRow_sq K hK).mul_right (‖v‖ ^ 2)) fun n ↦
        norm_inner_sq_le _ _).trans (by rw [tsum_mul_right])

/-- Row `n` of `Kv` is the inner product with row `n` of `K`, conjugated. -/
theorem hsOp_apply_eq_inner (v : ℓ²(ℤ, ℂ)) (n : ℤ) : hsOp K hK v n = inner ℂ (hsRow K hK n) v :=
  rfl

/-- `(Kv)ₙ = ∑ₘ Kₙₘvₘ`, absolutely convergent. -/
theorem hasSum_hsOp (v : ℓ²(ℤ, ℂ)) (n : ℤ) : HasSum (fun m ↦ K n m * v m) (hsOp K hK v n) := by
  rw [hsOp_apply_eq_inner]
  simpa [hsRow, RCLike.inner_apply, mul_comm] using lp.hasSum_inner (𝕜 := ℂ) (hsRow K hK n) v

/-- `(Kv)ₙ = ∑ₘ Kₙₘvₘ`. -/
theorem hsOp_apply (v : ℓ²(ℤ, ℂ)) (n : ℤ) : hsOp K hK v n = ∑' m, K n m * v m :=
  (hasSum_hsOp K hK v n).tsum_eq.symm

/-- `∑ₙ xₙyₙ` converges for `x, y ∈ ℓ²`. -/
theorem summable_mul_lp (x y : ℓ²(ℤ, ℂ)) : Summable fun n ↦ x n * y n :=
  (lp.summable_inner (star x) y).congr fun n ↦ by simp [lp.star_apply, mul_comm]

/-- **The bilinear pairing is symmetric for a symmetric `K`**: `∑ₙ xₙ(Kc)ₙ = ∑ₙ cₙ(Kx)ₙ`. The
double sum `∑ₙₗ xₙKₙₗcₗ` converges absolutely, as `|xₙKₙₗcₗ| ≤ (|xₙ|²|cₗ|² + |Kₙₗ|²)/2`. -/
theorem tsum_mul_hsOp_comm (hsym : ∀ n m, K n m = K m n) (x c : ℓ²(ℤ, ℂ)) :
    ∑' n, x n * hsOp K hK c n = ∑' n, c n * hsOp K hK x n := by
  have hx := memℓp_two_iff.mp (lp.memℓp x)
  have hc := memℓp_two_iff.mp (lp.memℓp c)
  have hf : Summable (Function.uncurry fun n m ↦ x n * (K n m * c m)) := by
    refine Summable.of_norm_bounded
      (((hx.mul_of_nonneg hc (fun _ ↦ by positivity) fun _ ↦ by positivity).add hK).div_const 2)
      fun p ↦ ?_
    simp only [Function.uncurry, norm_mul]
    nlinarith [sq_nonneg (‖x p.1‖ * ‖c p.2‖ - ‖K p.1 p.2‖), norm_nonneg (x p.1),
      norm_nonneg (c p.2), norm_nonneg (K p.1 p.2)]
  simp only [hsOp_apply, ← tsum_mul_left]
  rw [← hf.tsum_comm]
  exact tsum_congr fun m ↦ tsum_congr fun n ↦ by rw [hsym n m]; ring

/-- `∑ₙ xₙ((1 + K)c)ₙ = ∑ₙ cₙ((1 + K)x)ₙ` for a symmetric `K`. -/
theorem tsum_mul_one_add_hsOp_comm (hsym : ∀ n m, K n m = K m n) (x c : ℓ²(ℤ, ℂ)) :
    ∑' n, x n * (1 + hsOp K hK) c n = ∑' n, c n * (1 + hsOp K hK) x n := by
  simp only [_root_.add_apply, one_apply_eq_self, lp.coeFn_add, Pi.add_apply, mul_add]
  rw [(summable_mul_lp x c).tsum_add (summable_mul_lp x _),
    (summable_mul_lp c x).tsum_add (summable_mul_lp c _), tsum_mul_hsOp_comm K hK hsym]
  simp only [mul_comm (x _)]

/-- The coordinates in `s`, as a map to `s → ℂ`. -/
noncomputable def finEval (s : Finset ℤ) : ℓ²(ℤ, ℂ) →L[ℂ] (s → ℂ) :=
  ContinuousLinearMap.pi fun i : s ↦ lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 i.1

/-- `finEval s v` takes the coordinates of `v`. -/
theorem finEval_apply (s : Finset ℤ) (v : ℓ²(ℤ, ℂ)) (i : s) : finEval s v i = v i :=
  rfl

/-- The vector of `ℓ²` with the coordinates in `s` given, zero elsewhere. -/
noncomputable def finSpan (s : Finset ℤ) : (s → ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  ∑ i : s, (lp.singleContinuousLinearMap ℂ (fun _ : ℤ ↦ ℂ) 2 i.1).comp
    (ContinuousLinearMap.proj i)

/-- The projection of `ℓ²` onto the coordinates in `s`, of finite rank. -/
noncomputable def finProj (s : Finset ℤ) : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  (finSpan s).comp (finEval s)

/-- `finProj s` zeroes the coordinates outside `s`. -/
theorem finProj_apply (s : Finset ℤ) (v : ℓ²(ℤ, ℂ)) (n : ℤ) :
    finProj s v n = if n ∈ s then v n else 0 := by
  simp only [finProj, finSpan, ContinuousLinearMap.comp_apply,
    FunLike.coe_sum, Finset.sum_apply, lp.coeFn_sum]
  simp only [ContinuousLinearMap.proj_apply, lp.singleContinuousLinearMap_apply, lp.single_apply,
    finEval_apply]
  rw [Finset.sum_coe_sort s (fun i ↦ Pi.single i (v i) n)]
  exact Finset.sum_pi_single n (fun i ↦ v i) s

/-- **Hilbert–Schmidt matrices are compact.** -/
theorem isCompactOperator_hsOp : IsCompactOperator (hsOp K hK) := by
  have hc := summable_norm_hsRow_sq K hK
  refine isCompactOperator_of_tendsto (l := atTop) (F := fun s ↦ (finProj s).comp (hsOp K hK))
    ?_ (Eventually.of_forall fun s ↦ ?_)
  · rw [tendsto_iff_norm_sub_tendsto_zero]
    have ht := ((tendsto_tsum_compl_atTop_zero fun n ↦ ‖hsRow K hK n‖ ^ 2).sqrt)
    rw [Real.sqrt_zero] at ht
    refine squeeze_zero (fun _ ↦ norm_nonneg _) (fun s ↦ ?_) ht
    refine ContinuousLinearMap.opNorm_le_bound _ (Real.sqrt_nonneg _) fun v ↦ ?_
    refine le_sqrt_mul_of_sq_le (norm_nonneg _) (norm_nonneg _)
      (tsum_nonneg fun _ ↦ by positivity) ?_
    rw [show (∑' n : {n // n ∉ s}, ‖hsRow K hK n‖ ^ 2) =
        ∑' n, ({n | n ∉ s} : Set ℤ).indicator (fun n ↦ ‖hsRow K hK n‖ ^ 2) n from
      tsum_subtype ({n | n ∉ s} : Set ℤ) fun n ↦ ‖hsRow K hK n‖ ^ 2, ← tsum_mul_right]
    refine lp_norm_sq_le ((hc.indicator _).mul_right _) fun n ↦ ?_
    simp only [_root_.sub_apply, ContinuousLinearMap.comp_apply, lp.coeFn_sub,
      Pi.sub_apply, finProj_apply, Set.indicator, Set.mem_ofPred_eq]
    by_cases hn : n ∈ s
    · simp [hn]
    · simp only [hn, ite_false, zero_sub, norm_neg, not_false_eq_true, ite_true]
      exact norm_inner_sq_le _ _
  · rw [show (finProj s).comp (hsOp K hK) = (finSpan s).comp ((finEval s).comp (hsOp K hK))
      from rfl, ContinuousLinearMap.coe_comp]
    exact (isCompactOperator_of_locallyCompactSpace_dom _).clm_comp _

/-- The diagonal matrix `diag(c)`. -/
def diagK (c : ℤ → ℂ) (n m : ℤ) : ℂ := if n = m then c n else 0

/-- A diagonal matrix with square-summable diagonal is Hilbert–Schmidt. -/
theorem summable_diag_sq {c : ℤ → ℂ} (hc : Summable fun n ↦ ‖c n‖ ^ 2) :
    Summable fun p : ℤ × ℤ ↦ ‖diagK c p.1 p.2‖ ^ 2 := by
  refine (Function.Injective.summable_iff (g := fun n : ℤ ↦ (n, n))
    (fun a b h ↦ (Prod.ext_iff.mp h).1) fun p hp ↦ ?_).mp ?_
  · have h : p.1 ≠ p.2 := fun h ↦ hp ⟨p.1, Prod.ext rfl h⟩
    simp [diagK, h]
  · simpa [Function.comp_def, diagK] using hc

/-- A diagonal matrix acts by multiplication. -/
theorem hsOp_diag_apply {c : ℤ → ℂ} (hc : Summable fun n ↦ ‖c n‖ ^ 2) (v : ℓ²(ℤ, ℂ)) (n : ℤ) :
    hsOp (diagK c) (summable_diag_sq hc) v n = c n * v n := by
  rw [hsOp_apply]
  simp_rw [diagK, ite_mul, zero_mul, @eq_comm _ n]
  exact tsum_ite_eq n fun m ↦ c n * v m

/-- `‖hsOp K‖ ≤ (∑ₙₘ |Kₙₘ|²)^(1/2)`, the Hilbert–Schmidt norm. -/
theorem norm_hsOp_le : ‖hsOp K hK‖ ≤ √(∑' p : ℤ × ℤ, ‖K p.1 p.2‖ ^ 2) := by
  have hrows : ∑' n, ‖hsRow K hK n‖ ^ 2 = ∑' p : ℤ × ℤ, ‖K p.1 p.2‖ ^ 2 := by
    rw [hK.tsum_prod]
    exact tsum_congr fun n ↦ by rw [lp_norm_sq_eq_tsum]; simp [hsRow]
  refine ContinuousLinearMap.opNorm_le_bound _ (Real.sqrt_nonneg _) fun v ↦
    le_sqrt_mul_of_sq_le (norm_nonneg _) (norm_nonneg _) (tsum_nonneg fun _ ↦ by positivity) ?_
  rw [← hrows, ← tsum_mul_right]
  exact lp_norm_sq_le ((summable_norm_hsRow_sq K hK).mul_right _) fun n ↦ by
    rw [hsOp_apply_eq_inner]
    exact norm_inner_sq_le _ _

/-- `hsOp` is additive in the matrix: `hsOp K - hsOp K' = hsOp (K - K')`. -/
theorem hsOp_sub (K' : ℤ → ℤ → ℂ) (hK' : Summable fun p : ℤ × ℤ ↦ ‖K' p.1 p.2‖ ^ 2)
    (hd : Summable fun p : ℤ × ℤ ↦ ‖K p.1 p.2 - K' p.1 p.2‖ ^ 2) :
    hsOp K hK - hsOp K' hK' = hsOp (fun n m ↦ K n m - K' n m) hd := by
  ext v n
  have h := hasSum_hsOp (fun n m ↦ K n m - K' n m) hd v n
  simp only [sub_mul] at h
  rw [_root_.sub_apply, lp.coeFn_sub, Pi.sub_apply]
  exact ((hasSum_hsOp K hK v n).sub (hasSum_hsOp K' hK' v n)).unique h

/-- `K` restricted to `s × s`, zero elsewhere. -/
def truncK (s : Finset ℤ) (n m : ℤ) : ℂ := if n ∈ s ∧ m ∈ s then K n m else 0

include hK in
/-- The restriction of `K` to `s × s` is Hilbert–Schmidt. -/
theorem summable_truncK_sq (s : Finset ℤ) :
    Summable fun p : ℤ × ℤ ↦ ‖truncK K s p.1 p.2‖ ^ 2 :=
  hK.of_nonneg_of_le (fun _ ↦ by positivity) fun p ↦ by
    unfold truncK
    split_ifs <;> simp

/-- The restriction of `K` acts on the coordinates in `s`: `(K_s v)ₙ = ∑_{l ∈ s} Kₙₗvₗ` for
`n ∈ s`, and `0` otherwise. -/
theorem hsOp_truncK_apply (s : Finset ℤ) (v : ℓ²(ℤ, ℂ)) (n : ℤ) :
    hsOp (truncK K s) (summable_truncK_sq K hK s) v n =
      if n ∈ s then ∑ l ∈ s, K n l * v l else 0 := by
  rw [hsOp_apply, tsum_eq_sum (s := s) fun l hl ↦ by simp [truncK, hl]]
  split_ifs with hn
  · exact Finset.sum_congr rfl fun l hl ↦ by simp [truncK, hn, hl]
  · simp [truncK, hn]

/-- **The restrictions to finite sets converge in norm**: `‖K - K_s‖` is bounded by the
Hilbert–Schmidt norm of the entries outside `s × s`, which tends to `0`. -/
theorem tendsto_hsOp_truncK :
    Tendsto (fun s ↦ hsOp (truncK K s) (summable_truncK_sq K hK s)) atTop (𝓝 (hsOp K hK)) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  have ht := ((tendsto_tsum_compl_atTop_zero fun p : ℤ × ℤ ↦ ‖K p.1 p.2‖ ^ 2).comp
    (tendsto_finsetProd_atTop.comp tendsto_atTop_diagonal)).sqrt
  rw [Real.sqrt_zero] at ht
  refine squeeze_zero (fun _ ↦ norm_nonneg _) (fun s ↦ ?_) ht
  have hd : Summable fun p : ℤ × ℤ ↦ ‖truncK K s p.1 p.2 - K p.1 p.2‖ ^ 2 :=
    hK.of_nonneg_of_le (fun _ ↦ by positivity) fun p ↦ by
      unfold truncK
      split_ifs <;> simp
  rw [hsOp_sub (truncK K s) (summable_truncK_sq K hK s) K hK hd]
  refine (norm_hsOp_le (fun n m ↦ truncK K s n m - K n m) hd).trans
    (Real.sqrt_le_sqrt (le_of_eq ?_))
  simp only [Function.comp_apply]
  rw [show (∑' p : {p // p ∉ s ×ˢ s}, ‖K p.1.1 p.1.2‖ ^ 2) =
      ∑' p, ({p | p ∉ s ×ˢ s} : Set (ℤ × ℤ)).indicator (fun p ↦ ‖K p.1 p.2‖ ^ 2) p from
    tsum_subtype ({p | p ∉ s ×ˢ s} : Set (ℤ × ℤ)) fun p ↦ ‖K p.1 p.2‖ ^ 2]
  refine tsum_congr fun p ↦ ?_
  by_cases hp : p ∈ s ×ˢ s <;> rw [Finset.mem_product] at hp <;>
    simp [truncK, hp, Set.indicator, Finset.mem_product]

end HubbardAtom
