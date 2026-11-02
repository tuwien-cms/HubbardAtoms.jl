import HubbardAtomLean.ConverseExpansion

/-!
# The divergence criterion

For the density and magnetic channels, `U ≠ 0`, `μ ≠ 0`:

* `Params.Phik_ne_zero_of_Pph`: the roots of `Φ_k` avoid the zeros `±a ± w/2` of `P`, where the
  real part of `uΦ_k(u)` is `±a(e^{εβa} + cosh βμ)`;
* `Params.finite_roots_ell`: for every `ℓ`, the values `s = u²` at the roots of `Φ_k` with
  eigenvalue `ℓ(s) = ℓ` form a finite set. There `ℓ(s) = -K_e(s)/P(s)`
  (`ell_eq_of_Phi`) with `K_e = β(s + q₁) - λhR_k` and `R_k² = 1 + s sinh²(βμ)/a²`
  (`Ke_eq`, `Rk_sq_of_Phi`), so `[β(s + q₁) + ℓP(s)]² = λ²h²[1 + s sinh²(βμ)/a²]`, a nonzero
  polynomial equation in `s` (of degree `8` for `ℓ ≠ 0`, `2` for `ℓ = 0`);
* `Params.eq_sum_gs_of_eigen'`: hence `eq_sum_gs_of_eigen` without its finiteness assumption: a
  bounded `v` with `χ_r v = ℓv` is a finite combination of the `g_s` with `Φ_k(s) = 0` and
  `ℓ(s) = ℓ`, if the roots of `Φ_k` are simple;
* `Params.not_isUnit_onePlusK_of_Ke`, `exists_Ke_of_not_isUnit`, `not_isUnit_onePlusK_iff`:
  **`Γ_r` diverges iff `Φ_k(s) = K_e(s) = 0` for some `s`**, i.e. `1 + 𝒦` of `GammaOperator.lean`
  is not invertible iff `Φ_k(u) = K_e(u) = 0` for some `u`, if the roots of `Φ_k` are simple (for
  the converse direction). At such a root `g_s` is a null vector (`hasSum_chiR_mul_g_of_Phi`);
  conversely a null vector is bounded (`exists_bound_of_null`), so a combination of `g_s` with
  `ℓ(s) = -K_e(s)/P(s) = 0`.
-/

open Complex Polynomial

namespace HubbardAtom

/-- `cos(jπ) = (-1)^j`, `sin(jπ) = 0`. -/
private theorem cos_sin_int_mul_pi (j : ℤ) :
    cos ((j : ℂ) * Real.pi) = τk j ∧ sin ((j : ℂ) * Real.pi) = 0 := by
  refine ⟨?_, sin_int_mul_pi _⟩
  obtain ⟨i, hi | hi⟩ := Int.even_or_odd' j
  · rw [show ((j : ℂ)) * Real.pi = (i : ℤ) * (2 * Real.pi) by rw [hi]; push_cast; ring,
      cos_int_mul_two_pi, τk, ite_eq_left ⟨i, by omega⟩]
  · rw [show ((j : ℂ)) * Real.pi = (i : ℤ) * (2 * Real.pi) + Real.pi by rw [hi]; push_cast; ring,
      cos_int_mul_two_pi_add_pi, τk, ite_eq_right (by rintro ⟨r, hr⟩; omega)]

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `Φ_k(a + jπi/β) ≠ 0` for `(-1)^j = (-1)^k`: `uΦ_k(u) = (-1)^k[u(cosh βa + cosh βμ) +
εa sinh βa]`, whose real part is `(-1)^k a(e^{εβa} + cosh βμ)`. -/
theorem Phik_a_add_ne_zero (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) {j : ℤ}
    (hj : τk j = τk k) : A.Phik k ε (A.a + ((j * Real.pi / A.β : ℝ) : ℂ) * I) ≠ 0 := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  set y : ℝ := j * Real.pi / A.β
  set u : ℂ := A.a + (y : ℂ) * I
  have hu : (A.β : ℂ) * u = ((A.β * A.a : ℝ) : ℂ) + (j : ℂ) * Real.pi * I := by
    simp only [u, y]
    push_cast
    field_simp
  obtain ⟨hc, hs⟩ := cos_sin_int_mul_pi j
  have hcosh : cosh ((A.β : ℂ) * u) = τk k * Real.cosh (A.β * A.a) := by
    rw [hu, cosh_add, cosh_mul_I, sinh_mul_I, hc, hs, hj, ofReal_cosh]
    ring
  have hsinh : sinh ((A.β : ℂ) * u) = τk k * Real.sinh (A.β * A.a) := by
    rw [hu, sinh_add, cosh_mul_I, sinh_mul_I, hc, hs, hj, ofReal_sinh]
    ring
  have hμ : cosh ((A.β : ℂ) * A.μ) = Real.cosh (A.β * A.μ) := by rw [ofReal_cosh]; push_cast; rfl
  obtain ⟨e, he, hpos⟩ : ∃ e : ℝ, (e : ℂ) = ε ∧
      0 < Real.cosh (A.β * A.a) + e * Real.sinh (A.β * A.a) := by
    rcases hε with rfl | rfl
    · exact ⟨1, by simp, by rw [one_mul, Real.cosh_add_sinh]; positivity⟩
    · exact ⟨-1, by simp, by rw [neg_one_mul, ← sub_eq_add_neg, Real.cosh_sub_sinh]; positivity⟩
  have hτ : τk k ≠ 0 := by rw [τk]; split_ifs <;> norm_num
  intro h0
  have h := A.mul_Phik k ε u
  rw [h0, mul_zero, Dk, hcosh, hsinh, hμ, ← he] at h
  have hX : ((A.a * (Real.cosh (A.β * A.a) + e * Real.sinh (A.β * A.a) +
      Real.cosh (A.β * A.μ)) : ℝ) : ℂ) +
      ((y * (Real.cosh (A.β * A.a) + Real.cosh (A.β * A.μ)) : ℝ) : ℂ) * I = 0 := by
    apply mul_left_cancel₀ hτ
    simp only [u] at h
    simp only [ofReal_add, ofReal_mul]
    linear_combination -h
  have hre := congrArg re hX
  simp only [add_re, ofReal_re, mul_re, ofReal_im, I_re, I_im, mul_zero, zero_mul, sub_zero,
    add_zero, zero_re] at hre
  have ha' : A.a ≠ 0 := by exact_mod_cast ha
  exact mul_ne_zero ha' (by linarith [Real.cosh_pos (A.β * A.μ)]) hre

/-- **The roots of `Φ_k` avoid the zeros of `P`**: `P(u) = 0` means `±u = a + jπi/β` with
`j = ±k`, where `Φ_k ≠ 0` (`Phik_a_add_ne_zero`). -/
theorem Phik_ne_zero_of_Pph (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) {u : ℂ}
    (hP : A.Pph k u = 0) : A.Phik k ε u ≠ 0 := by
  have hw : A.w k / 2 = ((k * Real.pi / A.β : ℝ) : ℂ) * I := by
    rw [w, ω]
    push_cast
    ring
  have hw' : -(A.w k / 2) = (((-k : ℤ) * Real.pi / A.β : ℝ) : ℂ) * I := by
    rw [hw]
    push_cast
    ring
  have hτ : τk (-k) = τk k := by simp only [τk, even_neg]
  rw [Pph, Qp, Qm, mul_eq_zero] at hP
  rcases hP with hP | hP
  · -- `-u = a ∓ w/2`
    rw [← A.Phik_neg k ε]
    have : (-u - (A.a + -(A.w k / 2))) * (-u - (A.a + A.w k / 2)) = 0 := by
      linear_combination hP
    rcases mul_eq_zero.mp this with h | h
    · rw [sub_eq_zero.mp h, hw']
      exact A.Phik_a_add_ne_zero k ε hε ha hτ
    · rw [sub_eq_zero.mp h, hw]
      exact A.Phik_a_add_ne_zero k ε hε ha rfl
  · have : (u - (A.a + A.w k / 2)) * (u - (A.a + -(A.w k / 2))) = 0 := by
      linear_combination hP
    rcases mul_eq_zero.mp this with h | h
    · rw [sub_eq_zero.mp h, hw]
      exact A.Phik_a_add_ne_zero k ε hε ha rfl
    · rw [sub_eq_zero.mp h, hw']
      exact A.Phik_a_add_ne_zero k ε hε ha hτ

/-- **Finitely many roots carry a given eigenvalue**: for `ε = ±1`, `U ≠ 0`, `μ ≠ 0` and every
`ℓ`, the set of `s = u²` with `Φ_k(u) = 0` and `ℓ(s) = -εU ∑ₘ f_r(zₘ)g_s(m) = ℓ` is finite. -/
theorem finite_roots_ell (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) (ℓ : ℂ) :
    {s : ℂ | ∃ u, u ^ 2 = s ∧ A.Phik k ε u = 0 ∧
      ℓ = -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m}.Finite := by
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  set q₁ := A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁
  set c := A.lam ε * A.hth
  -- the equation for `ℓ(s) = ℓ`
  set pA : ℂ[X] := C ℓ * X ^ 2 + C (A.β + ℓ * A.κ k) * X + C (A.β * q₁ + ℓ * A.q₀ k ^ 2)
  set pB : ℂ[X] := C (c ^ 2 * (sinh (A.β * A.μ) ^ 2 / A.a ^ 2)) * X + C (c ^ 2)
  have hA : 1 ≤ pA.natDegree := by
    by_cases hℓ : ℓ = 0
    · have e : pA = C (A.β : ℂ) * X + C (A.β * q₁) := by
        simp only [pA, hℓ, C_0, zero_mul, zero_add, add_zero]
      rw [e, natDegree_linear hβ]
    · rw [natDegree_quadratic hℓ]
      norm_num
  have hQ0 : pA ^ 2 - pB ≠ 0 := by
    have hB : pB.natDegree < (pA ^ 2).natDegree := by
      rw [natDegree_pow]
      exact natDegree_linear_le.trans_lt (by omega)
    apply ne_zero_of_natDegree_gt (n := 1)
    rw [natDegree_sub_eq_left_of_natDegree_lt hB, natDegree_pow]
    omega
  refine (finite_setOfPred_isRoot hQ0).subset ?_
  rintro s ⟨u, rfl, hΦ, hℓ⟩
  have hu : ∀ m, A.z k m ≠ u := fun m h ↦ A.Phik_z_ne_zero k ε hε0 ha hμ m (h ▸ hΦ)
  have hu' : ∀ m, A.z k m ≠ -u := fun m h ↦
    A.Phik_z_ne_zero k ε hε0 ha hμ m (by rw [h, A.Phik_neg k ε, hΦ])
  have hP : A.Pph k u ≠ 0 := fun h ↦ A.Phik_ne_zero_of_Pph k ε hε ha h hΦ
  have hD := A.Dk_ne_zero_of_off k hu hu'
  have hK : A.Ke k ε u = -(ℓ * A.Pph k u) := by
    rw [hℓ, A.ell_eq_of_Phi k ε hε ha hΦ hP hu hu']
    field_simp
  have hR := A.Rk_sq_of_Phi k ε hε hD hΦ
  rw [A.Ke_eq k ε hD, A.Pph_eq_poly k] at hK
  simp only [Set.mem_ofPred_eq, IsRoot, pA, pB, eval_sub, eval_add, eval_mul, eval_C, eval_pow,
    eval_X]
  linear_combination (A.β * (u ^ 2 + q₁) + ℓ * (u ^ 4 + A.κ k * u ^ 2 + A.q₀ k ^ 2) +
    c * A.Rk k u) * hK + c ^ 2 * hR

/-- **The converse of the spectral identity, without the finiteness assumption**: for `U ≠ 0`,
`μ ≠ 0` and a bounded `v` with `χ_r v = ℓv`, if the roots of `Φ_k` are simple in `s = u²` (of
order `≤ 2` in `u` at `u = 0`), then `v = ∑_{ρ ∈ S} c_ρ g_{ρ²}` for a finite set `S` of roots of
`Φ_k` with `ℓ(ρ²) = ℓ`, by `finite_roots_ell`. -/
theorem eq_sum_gs_of_eigen' (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {v : ℤ → ℂ} {M : ℝ} (hv : ∀ n, ‖v n‖ ≤ M) {ℓ : ℂ}
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) (ℓ * v n))
    (hsimple : ∀ ρ, A.Phik k ε ρ = 0 →
      analyticOrderAt (A.Phik k ε) ρ ≤ if ρ = 0 then 2 else 1) :
    ∃ (S : Finset ℂ) (c : ℂ → ℂ), (∀ ρ ∈ S, A.Phik k ε ρ = 0 ∧
        ℓ = -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k ρ m) ∧
      ∀ n, v n = ∑ ρ ∈ S, c ρ * A.gs k ρ n := by
  classical
  set T := {s : ℂ | ∃ u, u ^ 2 = s ∧ A.Phik k ε u = 0 ∧
    ℓ = -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m}
  have hT : T.Finite := A.finite_roots_ell k ε hε ha hμ ℓ
  -- one square root of each `s ∈ T`
  set f : ℂ → ℂ := fun s ↦ if h : s ∈ T then h.choose else 0
  have hf : ∀ s ∈ T, f s ^ 2 = s ∧ A.Phik k ε (f s) = 0 ∧
      ℓ = -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k (f s) m := fun s hs ↦ by
    simp only [f, hs, ↓reduceDIte]
    exact hs.choose_spec
  set S := hT.toFinset.image f
  have hmem : ∀ ρ ∈ S, ∃ s ∈ T, f s = ρ := fun ρ hρ ↦ by
    obtain ⟨s, hs, rfl⟩ := Finset.mem_image.mp hρ
    exact ⟨s, hT.mem_toFinset.mp hs, rfl⟩
  have hS : Set.InjOn (fun ρ : ℂ ↦ ρ ^ 2) S := by
    intro x hx y hy hxy
    obtain ⟨s, hs, rfl⟩ := hmem x hx
    obtain ⟨t, ht, rfl⟩ := hmem y hy
    have : s = t := by rw [← (hf s hs).1, ← (hf t ht).1]; exact hxy
    rw [this]
  have hSℓ : ∀ ρ, A.Phik k ε ρ = 0 → ℓ = -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k ρ m →
      ∃ σ ∈ S, σ ^ 2 = ρ ^ 2 := fun ρ hΦ hℓ ↦ by
    have hs : ρ ^ 2 ∈ T := ⟨ρ, rfl, hΦ, hℓ⟩
    exact ⟨f (ρ ^ 2), Finset.mem_image_of_mem f (hT.mem_toFinset.mpr hs), (hf _ hs).1⟩
  have hSΦ : ∀ ρ ∈ S, A.Phik k ε ρ = 0 ∧
      ℓ = -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k ρ m := fun ρ hρ ↦ by
    obtain ⟨s, hs, rfl⟩ := hmem ρ hρ
    exact (hf s hs).2
  obtain ⟨c, hc⟩ := A.eq_sum_gs_of_eigen k hε ha hμ hv hχ hsimple hS (fun ρ hρ ↦ (hSΦ ρ hρ).1)
    hSℓ
  exact ⟨S, c, hSΦ, hc⟩

/-- **A divergence at each root with `K_e = 0`**: if `Φ_k(u) = K_e(u) = 0`, then `g_s` is a null
vector of `χ_r` in the class, and `1 + 𝒦` is not invertible. -/
theorem not_isUnit_onePlusK_of_Ke (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {u : ℂ} (hΦ : A.Phik k ε u = 0) (hK : A.Ke k ε u = 0) : ¬IsUnit (A.onePlusK k ε hμ) := by
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  have hP : A.Pph k u ≠ 0 := fun h ↦ A.Phik_ne_zero_of_Pph k ε hε ha h hΦ
  have hu : ∀ m, A.z k m ≠ u := fun m h ↦ A.Phik_z_ne_zero k ε hε0 ha hμ m (h ▸ hΦ)
  have hu' : ∀ m, A.z k m ≠ -u := fun m h ↦
    A.Phik_z_ne_zero k ε hε0 ha hμ m (by rw [h, A.Phik_neg k ε, hΦ])
  rw [A.isUnit_onePlusK_iff k ε hμ]
  intro h
  have hg := (A.hasSum_g k hu hu').summable.norm
  have h0 := congrFun (h (A.gs k u) ?_ fun n ↦ by
    simpa [hK] using A.hasSum_chiR_mul_g_of_Phi k ε hε ha hμ hΦ hP hu hu' n) 0
  · exact one_div_ne_zero (A.z_sq_sub_ne_zero k hu hu' 0) h0
  · refine A.summable_sq_of_norm_mul_z_le k hμ (C := ∑' n, ‖A.gs k u n‖) fun n ↦ ?_
    rw [mul_div_cancel₀ _ (A.z_ne_zero k hμ n)]
    exact hg.le_tsum n fun _ _ ↦ norm_nonneg _

/-- **A divergence needs a root with `K_e = 0`**: if the roots of `Φ_k` are simple and `1 + 𝒦` is
not invertible, then `Φ_k(u) = K_e(u) = 0` for some `u`. -/
theorem exists_Ke_of_not_isUnit (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    (hsimple : ∀ ρ, A.Phik k ε ρ = 0 →
      analyticOrderAt (A.Phik k ε) ρ ≤ if ρ = 0 then 2 else 1)
    (h : ¬IsUnit (A.onePlusK k ε hμ)) :
    ∃ u, A.Phik k ε u = 0 ∧ A.Ke k ε u = 0 := by
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  rw [A.isUnit_onePlusK_iff k ε hμ] at h
  push Not at h
  obtain ⟨v, hv, hχ, hv0⟩ := h
  obtain ⟨M, hM⟩ := A.exists_bound_of_null k ε hμ hv hχ
  obtain ⟨S, c, hS, hvS⟩ := A.eq_sum_gs_of_eigen' k ε hε ha hμ hM (ℓ := 0)
    (fun n ↦ by simpa using hχ n) hsimple
  obtain ⟨ρ, hρ⟩ : S.Nonempty := by
    rw [Finset.nonempty_iff_ne_empty]
    rintro rfl
    exact hv0 (funext fun n ↦ by simpa using hvS n)
  obtain ⟨hΦ, hℓ⟩ := hS ρ hρ
  refine ⟨ρ, hΦ, ?_⟩
  have hP : A.Pph k ρ ≠ 0 := fun h ↦ A.Phik_ne_zero_of_Pph k ε hε ha h hΦ
  have hu : ∀ m, A.z k m ≠ ρ := fun m h ↦ A.Phik_z_ne_zero k ε hε0 ha hμ m (h ▸ hΦ)
  have hu' : ∀ m, A.z k m ≠ -ρ := fun m h ↦
    A.Phik_z_ne_zero k ε hε0 ha hμ m (by rw [h, A.Phik_neg k ε, hΦ])
  rw [A.ell_eq_of_Phi k ε hε ha hΦ hP hu hu', eq_comm, neg_eq_zero, div_eq_zero_iff] at hℓ
  exact hℓ.resolve_right hP

/-- **The divergence criterion**: for `U ≠ 0`, `μ ≠ 0`, if the roots of `Φ_k` are simple, then
`Γ_r` diverges (`1 + 𝒦` is not invertible) iff `Φ_k(u) = K_e(u) = 0` for some `u`. -/
theorem not_isUnit_onePlusK_iff (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    (hsimple : ∀ ρ, A.Phik k ε ρ = 0 →
      analyticOrderAt (A.Phik k ε) ρ ≤ if ρ = 0 then 2 else 1) :
    ¬IsUnit (A.onePlusK k ε hμ) ↔ ∃ u, A.Phik k ε u = 0 ∧ A.Ke k ε u = 0 :=
  ⟨A.exists_Ke_of_not_isUnit k ε hε ha hμ hsimple, fun ⟨_, hΦ, hK⟩ ↦
    A.not_isUnit_onePlusK_of_Ke k ε hε ha hμ hΦ hK⟩

end Params

end HubbardAtom
