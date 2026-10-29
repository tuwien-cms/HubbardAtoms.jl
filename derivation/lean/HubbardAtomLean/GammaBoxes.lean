import HubbardAtomLean.GammaClosedForm

/-!
# Finite boxes converge to the vertex

For the density and magnetic channels and `μ ≠ 0`, on the boxes `|2n + 1 + k| ≤ 2M + 1`
(`box k M`):

* `Params.tendsto_onePlusK_box`: the compressions `1 + 𝒦_M` of `1 + 𝒦` to the boxes converge in
  norm (`tendsto_hsOp_truncK`);
* `Params.chiBox_inv`: where `1 + 𝒦_M` is invertible, so is the box matrix `χ_r|_M`, with
  `(χ_r|_M)⁻¹ = 𝒟^(-1/2)(1 + 𝒦_M)⁻¹𝒟^(-1/2)` on the box (`isUnit_chiBox_det`); wherever
  `1 + 𝒦` is invertible, this holds for all large `M` (`eventually_isUnit_chiBox_det`);
* `Params.tendsto_gammaBox`: wherever `1 + 𝒦` is invertible, the box vertices
  `β²[(χ_r|_M)⁻¹ - B⁻¹]ₙₘ` converge to `Γ_r,nm` (`gammaOp`), as inversion is continuous at units;
* `Params.DMGeneric.tendsto_gammaBox`: on the domain of Theorem "density and magnetic" (generic
  case), they converge to the closed form `gammaR`.
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- The box `|2n + 1 + k| ≤ 2M + 1`. -/
def box (M : ℕ) : Finset ℤ := Finset.Icc (-((2 * M + 2 + k) / 2)) ((2 * M - k) / 2)

/-- `n ∈ box k M` iff `|2n + 1 + k| ≤ 2M + 1`. -/
theorem mem_box {M : ℕ} {n : ℤ} : n ∈ box k M ↔ |2 * n + 1 + k| ≤ 2 * M + 1 := by
  rw [box, Finset.mem_Icc, abs_le]
  omega

/-- The boxes exhaust `ℤ`. -/
theorem tendsto_box : Tendsto (box k) atTop atTop := by
  refine tendsto_atTop_finset_of_monotone (fun M M' h n hn ↦ ?_) fun n ↦
    ⟨(|2 * n + 1 + k|).toNat, ?_⟩
  · rw [mem_box] at hn ⊢
    have : (M : ℤ) ≤ M' := by exact_mod_cast h
    omega
  · rw [mem_box, Int.toNat_of_nonneg (abs_nonneg _)]
    have := abs_nonneg (2 * n + 1 + k)
    omega

/-- `1 + 𝒦_s`, the compression of `1 + 𝒦` to `s × s`. -/
noncomputable def onePlusK_box (hμ : A.μ ≠ 0) (s : Finset ℤ) : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  1 + hsOp (truncK (A.Kr k ε) s) (summable_truncK_sq _ (A.summable_Kr_sq k ε hμ) s)

/-- **The compressions converge in norm**: `1 + 𝒦_M → 1 + 𝒦`. -/
theorem tendsto_onePlusK_box (hμ : A.μ ≠ 0) :
    Tendsto (fun M ↦ A.onePlusK_box k ε hμ (box k M)) atTop (𝓝 (A.onePlusK k ε hμ)) :=
  ((tendsto_hsOp_truncK _ (A.summable_Kr_sq k ε hμ)).comp (tendsto_box k)).const_add 1

/-- The box matrix `χ_r|_s`. -/
noncomputable def chiBox (s : Finset ℤ) : Matrix s s ℂ := Matrix.of fun i j ↦ A.chiR k ε i j

/-- Where `1 + 𝒦_s` is invertible, `χ_r|_s` has the right inverse
`𝒟^(-1/2)(1 + 𝒦_s)⁻¹𝒟^(-1/2)`: the vector `y = (1 + 𝒦_s)⁻¹eₘ` has
`yₙ + ∑_{l ∈ s} 𝒦ₙₗyₗ = δₙₘ` for `n ∈ s`. -/
theorem chiBox_mul (hμ : A.μ ≠ 0) {s : Finset ℤ} (hu : IsUnit (A.onePlusK_box k ε hμ s)) :
    A.chiBox k ε s * Matrix.of (fun i j : s ↦
      Ring.inverse (A.onePlusK_box k ε hμ s) (lp.single 2 (j : ℤ) 1) i /
        (A.Dhalf k i * A.Dhalf k j)) = 1 := by
  set T := A.onePlusK_box k ε hμ s
  set y : ℤ → ℓ²(ℤ, ℂ) := fun m ↦ Ring.inverse T (lp.single 2 m 1)
  have hy : ∀ m n, n ∈ s → y m n + ∑ l ∈ s, A.Kr k ε n l * y m l = kron n m := fun m n hn ↦ by
    have hT : T (y m) = lp.single 2 m 1 := apply_inverse_of_isUnit hu _
    have h := congrArg (fun w : ℓ²(ℤ, ℂ) ↦ w n) hT
    simp only [T, onePlusK_box, _root_.add_apply, one_apply_eq_self, lp.coeFn_add,
      Pi.add_apply] at h
    rw [hsOp_truncK_apply _ (A.summable_Kr_sq k ε hμ), single_apply_eq_kron] at h
    simpa only [hn, ite_true] using h
  have hd := A.Dhalf_ne_zero k hμ
  ext i j
  simp only [chiBox, Matrix.mul_apply, Matrix.of_apply, Matrix.one_apply]
  have h₁ : ∀ l : s, A.chiR k ε i l * (y j l / (A.Dhalf k l * A.Dhalf k j)) =
      A.Dhalf k i / A.Dhalf k j * (kron i l * y j l + A.Kr k ε i l * y j l) := fun l ↦ by
    rw [A.chiR_eq_Dhalf k ε hμ]
    field_simp [hd l, hd j]
  simp only [y] at h₁ hy
  simp only [h₁, ← Finset.mul_sum, Finset.sum_add_distrib]
  rw [Finset.sum_coe_sort s (fun l ↦ kron i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l),
    Finset.sum_coe_sort s (fun l ↦ A.Kr k ε i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l)]
  have hk : ∑ l ∈ s, kron i l * Ring.inverse T (lp.single 2 (j : ℤ) 1) l =
      Ring.inverse T (lp.single 2 (j : ℤ) 1) i := by
    simp only [kron, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq, i.2, ite_true]
  rw [hk, hy j i i.2]
  by_cases hij : i = j
  · subst hij
    rw [kron_self, ite_eq_left rfl, mul_one, div_self (hd i)]
  · rw [kron_of_ne fun h ↦ hij (Subtype.ext h), ite_eq_right hij, mul_zero]

/-- **The box matrix is invertible** where `1 + 𝒦_s` is. -/
theorem isUnit_chiBox_det (hμ : A.μ ≠ 0) {s : Finset ℤ}
    (hu : IsUnit (A.onePlusK_box k ε hμ s)) : IsUnit (A.chiBox k ε s).det :=
  Matrix.isUnit_det_of_right_inverse (A.chiBox_mul k ε hμ hu)

/-- **Proposition "finite boxes"**: wherever `1 + 𝒦` is invertible, the box matrices are invertible
for all sufficiently large `M`. -/
theorem eventually_isUnit_chiBox_det (hμ : A.μ ≠ 0) (hu : IsUnit (A.onePlusK k ε hμ)) :
    ∀ᶠ M in atTop, IsUnit (A.chiBox k ε (box k M)).det :=
  ((A.tendsto_onePlusK_box k ε hμ).eventually (Units.isOpen.mem_nhds hu)).mono fun _ hM ↦
    A.isUnit_chiBox_det k ε hμ hM

/-- `(χ_r|_s)⁻¹ᵢⱼ = [(1 + 𝒦_s)⁻¹]ᵢⱼ/(dᵢdⱼ)`. -/
theorem chiBox_inv (hμ : A.μ ≠ 0) {s : Finset ℤ} (hu : IsUnit (A.onePlusK_box k ε hμ s))
    (i j : s) : (A.chiBox k ε s)⁻¹ i j =
      Ring.inverse (A.onePlusK_box k ε hμ s) (lp.single 2 (j : ℤ) 1) i /
        (A.Dhalf k i * A.Dhalf k j) := by
  rw [Matrix.inv_eq_right_inv (A.chiBox_mul k ε hμ hu), Matrix.of_apply]

/-- The box vertex `β²[(χ_r|_s)⁻¹ - B⁻¹]ₙₘ` for `n, m ∈ s`, zero outside the box. -/
noncomputable def gammaBox (s : Finset ℤ) (n m : ℤ) : ℂ :=
  if h : n ∈ s ∧ m ∈ s then
    A.β ^ 2 * ((A.chiBox k ε s)⁻¹ ⟨n, h.1⟩ ⟨m, h.2⟩ - kron n m / A.Bph k n)
  else 0

/-- **The box vertices converge to `Γ_r`** wherever `1 + 𝒦` is invertible: `1 + 𝒦_M → 1 + 𝒦` in
norm, so `1 + 𝒦_M` is eventually invertible and `(1 + 𝒦_M)⁻¹ → (1 + 𝒦)⁻¹`. -/
theorem tendsto_gammaBox (hμ : A.μ ≠ 0) (hu : IsUnit (A.onePlusK k ε hμ)) (n m : ℤ) :
    Tendsto (fun M ↦ A.gammaBox k ε (box k M) n m) atTop (𝓝 (A.gammaOp k ε hμ n m)) := by
  have hT := A.tendsto_onePlusK_box k ε hμ
  have hinv : Tendsto (fun M ↦ Ring.inverse (A.onePlusK_box k ε hμ (box k M))) atTop
      (𝓝 (Ring.inverse (A.onePlusK k ε hμ))) := by
    have h := NormedRing.inverse_continuousAt hu.unit
    rw [hu.unit_spec] at h
    exact h.tendsto.comp hT
  have he := ((lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 n).continuous.tendsto _).comp
    (((ContinuousLinearMap.apply ℂ _ (lp.single 2 m (1 : ℂ))).continuous.tendsto _).comp hinv)
  have hlim := (he.sub_const (Ring.inverse (A.onePlusK0 k hμ) (lp.single 2 m 1) n)).const_mul
    (A.β ^ 2 / (A.Dhalf k n * A.Dhalf k m))
  have h0 : Ring.inverse (A.onePlusK0 k hμ) (lp.single 2 m 1) n = kron n m / A.tB k m := by
    rw [inverse_apply_of_eq (show IsUnit (A.onePlusK0 k hμ) from A.isUnit_one_add_K0 k hμ)
      (onePlusK0_single hμ m), lp.coeFn_smul, Pi.smul_apply, smul_eq_mul, single_apply_eq_kron,
      div_eq_inv_mul]
  rw [show A.gammaOp k ε hμ n m = A.β ^ 2 / (A.Dhalf k n * A.Dhalf k m) *
      (Ring.inverse (A.onePlusK k ε hμ) (lp.single 2 m 1) n -
        Ring.inverse (A.onePlusK0 k hμ) (lp.single 2 m 1) n) from by
    rw [gammaOp, _root_.sub_apply, lp.coeFn_sub, Pi.sub_apply]]
  refine hlim.congr' ?_
  filter_upwards [(tendsto_box k).eventually (eventually_ge_atTop {n, m}),
    hT.eventually (Units.isOpen.mem_nhds hu)] with M hM hU
  have hn : n ∈ box k M := hM (by simp)
  have hm : m ∈ box k M := hM (by simp)
  have hev : ∀ f : ℓ²(ℤ, ℂ), lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 n f = f n := fun _ ↦ rfl
  simp only [Function.comp_apply, ContinuousLinearMap.apply_apply, hev]
  rw [gammaBox, dite_eq_left ⟨hn, hm⟩, A.chiBox_inv k ε hμ hU, h0, A.Bph_eq_Dhalf k hμ]
  have hdn := A.Dhalf_ne_zero k hμ n
  have hdm := A.Dhalf_ne_zero k hμ m
  have htn := A.tB_ne_zero k hμ n
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self]
    field_simp
  · rw [kron_of_ne hnm]
    field_simp
    ring

/-- **Boxes on the domain of the theorem**: under the assumptions of Theorem "density and
magnetic" (generic case), the box vertices converge to the closed form `gammaR`. -/
theorem DMGeneric.tendsto_gammaBox {u₁ u₂ : ℂ} (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    Tendsto (fun M ↦ A.gammaBox k ε (box k M) n m) atTop (𝓝 (A.gammaR k ε u₁ u₂ n m)) := by
  rw [← h.gammaOp_eq]
  exact A.tendsto_gammaBox k ε h.μ_ne h.isUnit_onePlusK n m

end Params

end HubbardAtom
