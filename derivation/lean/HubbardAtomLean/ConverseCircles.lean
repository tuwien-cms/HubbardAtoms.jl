import HubbardAtomLean.ConverseAnalytic

/-!
# The converse of the spectral identity: estimates on circles

On the circles `|u| = R_j`, `R_j² = μ² + ((2j + k)π/β)²`, which pass between the lattice values
`|zₘ|² = μ² + ((2m + 1 + k)π/β)²` (`2j + k` and `2m + 1 + k` have opposite parity), the Cauchy
transform of a bounded `v` tends to zero uniformly, and `E = v̂D_k` is small against `Φ_k`:

* `Params.norm_gs_le_circ`: `|g_s(m)| ≤ 2(β/π)²[φ_C(m - j) + φ_C(m + j + k)]` on the circle,
  with `φ_C(i) = 1/(|2i + 1|(|2i + 1| + C))` and `C = |2j + k|`, from
  `|zₘ² - u²| ≥ ||zₘ|² - R_j²| = (π/β)²|2(m - j) + 1||2(m + j + k) + 1|`;
* `Params.norm_vhat_le_circ`, `tendsto_circδ`: `|v̂(u²)| ≤ M δ_j` for `|vₘ| ≤ M`, with
  `δ_j = 4(β/π)² ∑ᵢ φ_C(i) → 0` by dominated convergence;
* `Params.norm_Ehat_le_circ`: for large `j`, `|E(u)| ≤ 2Mδ_j|Φ_k(u)|` on the circle, as
  `Φ_k = D_k[1 - (εU/β)𝒯]` with `𝒯 = ∑ₙ g_s(n) → 0` there; in particular `Φ_k ≠ 0` on these
  circles (`Phik_ne_zero_circ`).
-/

open Complex Filter Topology

namespace HubbardAtom

/-- `φ_C(i) = 1/(|2i + 1|(|2i + 1| + C))`. -/
noncomputable def circPhi (C : ℝ) (i : ℤ) : ℝ := 1 / (|(2 * i + 1 : ℝ)| * (|(2 * i + 1 : ℝ)| + C))

theorem one_le_abs_two_mul_add_one (i : ℤ) : 1 ≤ |(2 * i + 1 : ℝ)| := by
  have h : (1 : ℤ) ≤ |2 * i + 1| := Int.one_le_abs (by omega)
  exact_mod_cast h

theorem circPhi_le {C : ℝ} (hC : 0 ≤ C) (i : ℤ) : circPhi C i ≤ 1 / ((2 * i + 1 : ℤ) : ℝ) ^ 2 := by
  have h := one_le_abs_two_mul_add_one i
  rw [circPhi, ← sq_abs (((2 * i + 1 : ℤ) : ℝ)), Int.cast_add, Int.cast_mul, Int.cast_ofNat,
    Int.cast_one]
  exact one_div_le_one_div_of_le (by positivity) (by nlinarith)

theorem summable_inv_odd_sq : Summable fun i : ℤ ↦ 1 / ((2 * i + 1 : ℤ) : ℝ) ^ 2 :=
  (Real.summable_one_div_int_pow.mpr (by norm_num : 1 < 2)).comp_injective
    (f := fun n : ℤ ↦ 1 / (n : ℝ) ^ 2) fun a b h ↦ by simpa using h

theorem summable_circPhi {C : ℝ} (hC : 0 ≤ C) : Summable (circPhi C) :=
  summable_inv_odd_sq.of_nonneg_of_le (fun i ↦ by
    have := one_le_abs_two_mul_add_one i
    rw [circPhi]
    positivity) (circPhi_le hC)

/-- `∑ᵢ φ_C(i) → 0` as `C = |2j + k| → ∞`, by dominated convergence. -/
theorem tendsto_tsum_circPhi (k : ℤ) :
    Tendsto (fun j : ℕ ↦ ∑' i, circPhi |(2 * j + k : ℝ)| i) atTop (𝓝 0) := by
  have hC : Tendsto (fun j : ℕ ↦ |(2 * j + k : ℝ)|) atTop atTop := by
    refine tendsto_atTop_mono (fun j ↦ le_abs_self _) (tendsto_atTop_add_const_right _ _ ?_)
    exact tendsto_natCast_atTop_atTop.const_mul_atTop two_pos
  have h := tendsto_tsum_of_dominated_convergence (𝓕 := atTop)
    (f := fun (j : ℕ) i ↦ circPhi |(2 * j + k : ℝ)| i) (g := fun _ ↦ (0 : ℝ))
    (bound := fun i : ℤ ↦ 1 / ((2 * i + 1 : ℤ) : ℝ) ^ 2) summable_inv_odd_sq (fun i ↦ ?_)
    (Eventually.of_forall fun j i ↦ ?_)
  · simpa using h
  · have ha := one_le_abs_two_mul_add_one i
    simp only [circPhi, one_div]
    exact tendsto_inv_atTop_zero.comp
      ((tendsto_atTop_add_const_left _ _ hC).const_mul_atTop (by positivity))
  · rw [Real.norm_of_nonneg (by
      have := one_le_abs_two_mul_add_one i
      rw [circPhi]
      positivity)]
    exact circPhi_le (abs_nonneg _) i

/-- `1/(ee') ≤ 2/(e(e + C)) + 2/(e'(e' + C))` for `e, e' > 0` with `e + e' ≥ 2C`: the larger of
`e`, `e'` is at least `C`. -/
private theorem one_div_mul_le {e e' C : ℝ} (he : 0 < e) (he' : 0 < e') (hC : 0 ≤ C)
    (h : 2 * C ≤ e + e') : 1 / (e * e') ≤ 2 / (e * (e + C)) + 2 / (e' * (e' + C)) := by
  rcases le_total e e' with h' | h'
  · calc 1 / (e * e') ≤ 2 / (e * (e + C)) := by
          rw [div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith
      _ ≤ _ := le_add_of_nonneg_right (by positivity)
  · calc 1 / (e * e') ≤ 2 / (e' * (e' + C)) := by
          rw [div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith
      _ ≤ _ := le_add_of_nonneg_left (by positivity)

namespace Params

variable (A : Params) (k : ℤ)

/-- The radius `R_j`, `R_j² = μ² + ((2j + k)π/β)²`. -/
noncomputable def circR (j : ℕ) : ℝ := √(A.μ ^ 2 + ((2 * j + k) * Real.pi / A.β) ^ 2)

/-- `δ_j = 4(β/π)² ∑ᵢ φ_C(i)`, `C = |2j + k|`. -/
noncomputable def circδ (j : ℕ) : ℝ :=
  4 * (A.β / Real.pi) ^ 2 * ∑' i, circPhi |(2 * j + k : ℝ)| i

theorem tendsto_circR : Tendsto (A.circR k) atTop atTop := by
  have hπβ : 0 < Real.pi / A.β := div_pos Real.pi_pos A.β_pos
  have h : Tendsto (fun j : ℕ ↦ |(2 * j + k : ℝ)| * (Real.pi / A.β)) atTop atTop := by
    refine Tendsto.atTop_mul_const hπβ (tendsto_atTop_mono (fun j ↦ le_abs_self _)
      (tendsto_atTop_add_const_right _ _ ?_))
    exact tendsto_natCast_atTop_atTop.const_mul_atTop two_pos
  refine tendsto_atTop_mono (fun j ↦ ?_) h
  rw [circR, ← abs_of_pos hπβ, ← abs_mul, mul_div_assoc]
  exact Real.abs_le_sqrt (le_add_of_nonneg_left (sq_nonneg _))

theorem tendsto_circδ : Tendsto (A.circδ k) atTop (𝓝 0) := by
  unfold circδ
  simpa using (tendsto_tsum_circPhi k).const_mul (4 * (A.β / Real.pi) ^ 2)

/-- On the circle, `|zₘ² - u²| ≥ (π/β)²|2(m - j) + 1||2(m + j + k) + 1|`. -/
theorem norm_z_sq_sub_ge_circ (j : ℕ) {u : ℂ} (hu : ‖u‖ = A.circR k j) (m : ℤ) :
    (Real.pi / A.β) ^ 2 * (|(2 * (m - j : ℤ) + 1 : ℝ)| * |(2 * (m + j + k : ℤ) + 1 : ℝ)|) ≤
      ‖A.z k m ^ 2 - u ^ 2‖ := by
  have hβ : A.β ≠ 0 := A.β_pos.ne'
  have hz : ‖A.z k m‖ ^ 2 = A.μ ^ 2 + ((2 * m + 1 + k) * Real.pi / A.β) ^ 2 := by
    rw [Complex.sq_norm, A.z_eq, normSq_add_mul_I]
  have hR : ‖u‖ ^ 2 = A.μ ^ 2 + ((2 * j + k) * Real.pi / A.β) ^ 2 := by
    rw [hu, circR, Real.sq_sqrt (by positivity)]
  refine le_of_eq_of_le ?_ (abs_norm_sub_norm_le _ _)
  rw [norm_pow, norm_pow, hz, hR, ← abs_mul, ← abs_of_nonneg (sq_nonneg (Real.pi / A.β)),
    ← abs_mul]
  congr 1
  push_cast
  field_simp
  ring

/-- **`g_s` on the circle**: `|g_s(m)| ≤ 2(β/π)²[φ_C(m - j) + φ_C(m + j + k)]`, `C = |2j + k|`. -/
theorem norm_gs_le_circ (j : ℕ) {u : ℂ} (hu : ‖u‖ = A.circR k j) (m : ℤ) :
    ‖A.gs k u m‖ ≤ 2 * (A.β / Real.pi) ^ 2 *
      (circPhi |(2 * j + k : ℝ)| (m - j) + circPhi |(2 * j + k : ℝ)| (m + j + k)) := by
  have h := A.norm_z_sq_sub_ge_circ k j hu m
  have hp := one_le_abs_two_mul_add_one (m - j)
  have hq := one_le_abs_two_mul_add_one (m + j + k)
  have hπβ : 0 < Real.pi / A.β := div_pos Real.pi_pos A.β_pos
  push_cast at h hp hq
  have hsum : 2 * |(2 * j + k : ℝ)| ≤ |2 * (m - j : ℝ) + 1| + |2 * (m + j + k : ℝ) + 1| := by
    calc 2 * |(2 * j + k : ℝ)| = |(2 * (m + j + k : ℝ) + 1) - (2 * (m - j : ℝ) + 1)| := by
          rw [show (2 * (m + j + k : ℝ) + 1) - (2 * (m - j : ℝ) + 1) = 2 * (2 * j + k) by ring,
            abs_mul, abs_two]
      _ ≤ _ := (abs_sub _ _).trans_eq (add_comm _ _)
  have hb := one_div_mul_le (by positivity) (by positivity) (abs_nonneg _) hsum
  rw [gs, norm_div, norm_one]
  calc 1 / ‖A.z k m ^ 2 - u ^ 2‖
      ≤ 1 / ((Real.pi / A.β) ^ 2 * (|2 * (m - j : ℝ) + 1| * |2 * (m + j + k : ℝ) + 1|)) :=
        one_div_le_one_div_of_le (by positivity) h
    _ = (A.β / Real.pi) ^ 2 * (1 / (|2 * (m - j : ℝ) + 1| * |2 * (m + j + k : ℝ) + 1|)) := by
        rw [div_pow, div_pow]
        field_simp
    _ ≤ (A.β / Real.pi) ^ 2 * (2 / (|2 * (m - j : ℝ) + 1| * (|2 * (m - j : ℝ) + 1| +
          |(2 * j + k : ℝ)|)) + 2 / (|2 * (m + j + k : ℝ) + 1| * (|2 * (m + j + k : ℝ) + 1| +
          |(2 * j + k : ℝ)|))) := mul_le_mul_of_nonneg_left hb (by positivity)
    _ = _ := by
        simp only [circPhi]
        push_cast
        ring

/-- The circle avoids the lattice: `zₘ² ≠ u²`. -/
theorem z_sq_sub_ne_zero_circ (j : ℕ) {u : ℂ} (hu : ‖u‖ = A.circR k j) (m : ℤ) :
    A.z k m ^ 2 - u ^ 2 ≠ 0 := by
  have h := A.norm_z_sq_sub_ge_circ k j hu m
  have hp := one_le_abs_two_mul_add_one (m - j)
  have hq := one_le_abs_two_mul_add_one (m + j + k)
  have hπβ : 0 < Real.pi / A.β := div_pos Real.pi_pos A.β_pos
  push_cast at h hp hq
  exact norm_pos_iff.mp (lt_of_lt_of_le (by positivity) h)

theorem z_ne_circ (j : ℕ) {u : ℂ} (hu : ‖u‖ = A.circR k j) (m : ℤ) : A.z k m ≠ u := fun h ↦
  A.z_sq_sub_ne_zero_circ k j hu m (by rw [h, sub_self])

theorem z_ne_neg_circ (j : ℕ) {u : ℂ} (hu : ‖u‖ = A.circR k j) (m : ℤ) : A.z k m ≠ -u := fun h ↦
  A.z_sq_sub_ne_zero_circ k j hu m (by rw [h, neg_sq, sub_self])

/-- **`v̂` on the circle**: `|v̂(u²)| ≤ Mδ_j` for `|vₘ| ≤ M`. -/
theorem norm_vhat_le_circ {v : ℤ → ℂ} {M : ℝ} (hv : ∀ n, ‖v n‖ ≤ M) (j : ℕ) {u : ℂ}
    (hu : ‖u‖ = A.circR k j) : ‖A.vhat k v u‖ ≤ M * A.circδ k j := by
  have hM : 0 ≤ M := (norm_nonneg _).trans (hv 0)
  set C := |(2 * j + k : ℝ)|
  have hs := (summable_circPhi (abs_nonneg (2 * j + k : ℝ))).hasSum
  have h₁ : HasSum (fun m : ℤ ↦ circPhi C (m - j)) (∑' i, circPhi C i) :=
    (Equiv.subRight (j : ℤ)).hasSum_iff.mpr hs
  have h₂ : HasSum (fun m : ℤ ↦ circPhi C (m + j + k)) (∑' i, circPhi C i) := by
    have := (Equiv.addRight ((j : ℤ) + k)).hasSum_iff.mpr hs
    simpa only [Function.comp_def, Equiv.coe_addRight, add_assoc] using this
  have hg : HasSum (fun m : ℤ ↦ M * (2 * (A.β / Real.pi) ^ 2 *
      (circPhi C (m - j) + circPhi C (m + j + k)))) (M * A.circδ k j) := by
    convert ((h₁.add h₂).mul_left (2 * (A.β / Real.pi) ^ 2)).mul_left M using 1
    rw [circδ]
    ring
  refine tsum_of_norm_bounded hg fun m ↦ ?_
  rw [norm_mul]
  exact mul_le_mul (hv m) (A.norm_gs_le_circ k j hu m) (norm_nonneg _) hM
    |>.trans_eq (by ring)

/-- `𝒯` on the circle: `|𝒯(u)| ≤ δ_j`. -/
theorem norm_Tg_le_circ (j : ℕ) {u : ℂ} (hu : ‖u‖ = A.circR k j) : ‖A.Tg k u‖ ≤ A.circδ k j := by
  have h := A.norm_vhat_le_circ k (v := fun _ ↦ 1) (M := 1) (fun _ ↦ by simp) j hu
  have hT : A.vhat k (fun _ ↦ 1) u = A.Tg k u := by
    rw [vhat, ← (A.hasSum_g k (A.z_ne_circ k j hu) (A.z_ne_neg_circ k j hu)).tsum_eq]
    simp
  rwa [hT, one_mul] at h

/-- For large `j`, `|D_k| ≤ 2|Φ_k|` on the circle, as `|(εU/β)𝒯| ≤ 1/2` there. -/
theorem eventually_norm_Dk_le_circ (ε : ℂ) : ∀ᶠ j in atTop, ∀ u : ℂ, ‖u‖ = A.circR k j →
    ‖A.Dk k u‖ ≤ 2 * ‖A.Phik k ε u‖ := by
  have h := ((A.tendsto_circδ k).const_mul ‖ε * A.U / A.β‖).eventually
    (Iic_mem_nhds (by norm_num : (‖ε * A.U / A.β‖ * 0 : ℝ) < 1 / 2))
  filter_upwards [h] with j hj u hu
  have hD := A.Dk_ne_zero_of_off k (A.z_ne_circ k j hu) (A.z_ne_neg_circ k j hu)
  have hΦ : A.Phik k ε u = A.Dk k u * (1 - ε * A.U / A.β * A.Tg k u) := by
    rw [A.one_sub_Tg k ε hD]
    field_simp
  have hx : ‖ε * A.U / A.β * A.Tg k u‖ ≤ 1 / 2 := by
    rw [norm_mul]
    exact (mul_le_mul_of_nonneg_left (A.norm_Tg_le_circ k j hu) (norm_nonneg _)).trans hj
  have h1 : 1 / 2 ≤ ‖1 - ε * A.U / A.β * A.Tg k u‖ := by
    have := norm_sub_norm_le (1 : ℂ) (ε * A.U / A.β * A.Tg k u)
    rw [norm_one] at this
    linarith
  rw [hΦ, norm_mul]
  nlinarith [norm_nonneg (A.Dk k u)]

/-- For large `j`, `Φ_k ≠ 0` on the circle. -/
theorem Phik_ne_zero_circ (ε : ℂ) : ∀ᶠ j in atTop, ∀ u : ℂ, ‖u‖ = A.circR k j →
    A.Phik k ε u ≠ 0 := by
  filter_upwards [A.eventually_norm_Dk_le_circ k ε] with j hj u hu hΦ
  have h := hj u hu
  rw [hΦ, norm_zero, mul_zero] at h
  exact A.Dk_ne_zero_of_off k (A.z_ne_circ k j hu) (A.z_ne_neg_circ k j hu) (norm_le_zero_iff.mp h)

/-- **`E` on the circle**: for large `j` and `|vₘ| ≤ M`, `|E(u)| ≤ 2Mδ_j|Φ_k(u)|`. -/
theorem norm_Ehat_le_circ (hμ : A.μ ≠ 0) (ε : ℂ) : ∀ᶠ j in atTop, ∀ (v : ℤ → ℂ) (M : ℝ),
    (∀ n, ‖v n‖ ≤ M) → ∀ u : ℂ, ‖u‖ = A.circR k j →
      ‖A.Ehat k v u‖ ≤ 2 * M * A.circδ k j * ‖A.Phik k ε u‖ := by
  filter_upwards [A.eventually_norm_Dk_le_circ k ε] with j hj v M hv u hu
  have hM : 0 ≤ M * A.circδ k j := (norm_nonneg _).trans (A.norm_vhat_le_circ k hv j hu)
  rw [Ehat_eq hμ v (A.z_ne_circ k j hu) (A.z_ne_neg_circ k j hu), norm_mul]
  calc ‖A.vhat k v u‖ * ‖A.Dk k u‖ ≤ M * A.circδ k j * (2 * ‖A.Phik k ε u‖) :=
        mul_le_mul (A.norm_vhat_le_circ k hv j hu) (hj u hu) (norm_nonneg _) hM
    _ = _ := by ring

end Params

end HubbardAtom
