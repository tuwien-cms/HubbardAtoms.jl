import HubbardAtomLean.Matsubara

/-!
# Continuity of simple roots, and of `S_k` at a moving point off the lattice

Tools for the continuity of the closed forms in the parameters, where the roots of a polynomial
(`e(u)`, a quadratic in `u²`, of the density and magnetic channels, `E₈` of the longitudinal
sector) and the values of `S_k` at these roots enter. Not in the note.

* `exists_continuousAt_zero_of_hasDerivAt`: the implicit function theorem (Mathlib's
  `HasStrictFDerivAt.implicitFunctionOfProdDomain`) for `F : E × ℂ → ℂ` `C¹` at `(c x₀, s₀)` (`E` a
  real Banach space) along a family `c : X → E` continuous at `x₀`: a simple zero `s₀` of
  `F (c x₀, ·)` moves continuously, and near `(x₀, s₀)` it is the only zero.
* `hasDerivAt_sum_mul_pow`: the derivative of `∑ᵢ cᵢ sⁱ` in `s`.
* `exists_continuousAt_root_of_simple` (and `exists_continuousAt_root_of_simple'`, with each
  coefficient continuous at `x₀`): for a polynomial `∑ᵢ cᵢ(x) sⁱ` with coefficients continuous at
  `x₀`, a simple root `s₀` at `x₀` is the value at `x₀` of a root `s(x)` continuous at `x₀`, the
  only root near `(x₀, s₀)`.
* `exists_continuousAt_sqrt_root_of_simple`: the same for a square root `u₀ ≠ 0` of a simple root
  `s₀ = u₀²`, a root `u(x)` of `∑ᵢ cᵢ(x) u²ⁱ`.
* `Params.eventually_z_eq_imp`: along a family `A : X → Params` with `μ`, `β` continuous at `x₀` and
  a point `u(x)` continuous at `x₀`, near `x₀` the point `u(x)` lies on the lattice `{zₙ}` of `A x`
  only at indices `n` where `u(x₀) = zₙ` at `x₀`, uniformly in `n` (`Params.z_eq_iff`: `zₙ = w` iff
  `Re w = μ` and `(β Im w/π - 1 - k)/2 = n`); so (`Params.eventually_z_ne_of_off`) if `u(x₀)` is
  off the lattice `{zₙ}` of `A x₀`, then `u(x)` is off the lattice of `A x` near `x₀`.
* `Params.continuousAt_S`: along such a family, `S_k(u(x))` is continuous at `x₀` if `u(x₀)` is off
  the lattice `{zₙ}` of `A x₀`, which is where the denominator `cosh` (even `k`) or `sinh` (odd `k`)
  of `S_k` does not vanish.
-/

open Complex Filter Topology

namespace HubbardAtom

/-- The implicit function theorem for a simple zero: if `F : E × ℂ → ℂ` is `C¹` at `(c x₀, s₀)`,
`F (c x₀, s₀) = 0` and the complex derivative of `F (c x₀, ·)` at `s₀` is nonzero, then there is
`s : X → ℂ` continuous at `x₀` with `s x₀ = s₀` and `F (c x, s x) = 0` near `x₀`, and near
`(x₀, s₀)` the zeros of `F (c x, ·)` are exactly `s x`. -/
theorem exists_continuousAt_zero_of_hasDerivAt {E X : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [CompleteSpace E] [TopologicalSpace X] {x₀ : X} {c : X → E}
    (hc : ContinuousAt c x₀) {F : E × ℂ → ℂ} {s₀ F' : ℂ} (hF : ContDiffAt ℝ 1 F (c x₀, s₀))
    (hroot : F (c x₀, s₀) = 0) (hF' : HasDerivAt (fun s ↦ F (c x₀, s)) F' s₀) (hF'0 : F' ≠ 0) :
    ∃ s : X → ℂ, s x₀ = s₀ ∧ ContinuousAt s x₀ ∧ (∀ᶠ x in 𝓝 x₀, F (c x, s x) = 0) ∧
      ∀ᶠ p in 𝓝 (x₀, s₀), F (c p.1, p.2) = 0 ↔ p.2 = s p.1 := by
  have hd := hF.hasStrictFDerivAt one_ne_zero
  -- the partial derivative in `s` is multiplication by `F' ≠ 0`
  have hinv : (fderiv ℝ F (c x₀, s₀) ∘L ContinuousLinearMap.inr ℝ E ℂ).IsInvertible := by
    have h1 := hd.hasFDerivAt.comp s₀ (hasFDerivAt_prodMk_right (𝕜 := ℝ) (c x₀) s₀)
    rw [h1.unique (hF'.hasFDerivAt.restrictScalars ℝ)]
    refine ⟨(ContinuousLinearEquiv.unitsEquivAut ℂ (Units.mk0 F' hF'0)).restrictScalars ℝ, ?_⟩
    ext x
    simp [mul_comm]
  have hiff := hd.eventually_apply_eq_iff_implicitFunctionOfProdDomain hinv
  set ψ := hd.implicitFunctionOfProdDomain hinv
  have hψ₀ : ψ (c x₀) = s₀ := hiff.self_of_nhds.mp rfl
  have hψ : ContinuousAt ψ (c x₀) := by
    have := hd.tendsto_implicitFunctionOfProdDomain hinv
    rwa [ContinuousAt, hψ₀]
  refine ⟨ψ ∘ c, hψ₀, hψ.comp hc, ?_, ?_⟩
  · have := hd.eventually_apply_implicitFunctionOfProdDomain hinv
    rw [hroot] at this
    exact hc.eventually this
  · have hcont : ContinuousAt (fun p : X × ℂ ↦ (c p.1, p.2)) (x₀, s₀) :=
      (hc.comp continuousAt_fst).prodMk continuousAt_snd
    filter_upwards [hcont.eventually hiff] with p hp
    rw [← hroot]
    exact hp.trans eq_comm

/-- The derivative of the polynomial `∑ᵢ cᵢ sⁱ` in `s`. -/
theorem hasDerivAt_sum_mul_pow {d : ℕ} (c : Fin (d + 1) → ℂ) (s : ℂ) :
    HasDerivAt (fun s ↦ ∑ i, c i * s ^ (i : ℕ))
      (∑ i : Fin (d + 1), (i : ℕ) * c i * s ^ ((i : ℕ) - 1)) s := by
  refine (HasDerivAt.fun_sum fun (i : Fin (d + 1)) _ ↦
    (hasDerivAt_pow (i : ℕ) s).const_mul (c i)).congr_deriv ?_
  exact Finset.sum_congr rfl fun i _ ↦ by ring

/-- **A simple root moves continuously**: for coefficients `c x` continuous at `x₀` and a simple
root `s₀` of `∑ᵢ cᵢ(x₀) sⁱ`, there is a root `s(x)` of `∑ᵢ cᵢ(x) sⁱ` near `x₀`, continuous at `x₀`
with `s x₀ = s₀`, and near `(x₀, s₀)` it is the only root. -/
theorem exists_continuousAt_root_of_simple {X : Type*} [TopologicalSpace X] {x₀ : X} {d : ℕ}
    {c : X → Fin (d + 1) → ℂ} (hc : ContinuousAt c x₀) {s₀ : ℂ}
    (hroot : ∑ i, c x₀ i * s₀ ^ (i : ℕ) = 0)
    (hsimple : ∑ i : Fin (d + 1), (i : ℕ) * c x₀ i * s₀ ^ ((i : ℕ) - 1) ≠ 0) :
    ∃ s : X → ℂ, s x₀ = s₀ ∧ ContinuousAt s x₀ ∧
      (∀ᶠ x in 𝓝 x₀, ∑ i, c x i * s x ^ (i : ℕ) = 0) ∧
      ∀ᶠ p in 𝓝 (x₀, s₀), ∑ i, c p.1 i * p.2 ^ (i : ℕ) = 0 ↔ p.2 = s p.1 := by
  have hF : ContDiff ℝ 1 fun p : (Fin (d + 1) → ℂ) × ℂ ↦ ∑ i, p.1 i * p.2 ^ (i : ℕ) := by
    fun_prop
  exact exists_continuousAt_zero_of_hasDerivAt hc hF.contDiffAt hroot
    (hasDerivAt_sum_mul_pow (c x₀) s₀) hsimple

/-- `exists_continuousAt_root_of_simple` with each coefficient `cᵢ` continuous at `x₀`. -/
theorem exists_continuousAt_root_of_simple' {X : Type*} [TopologicalSpace X] {x₀ : X} {d : ℕ}
    {c : X → Fin (d + 1) → ℂ} (hc : ∀ i, ContinuousAt (fun x ↦ c x i) x₀) {s₀ : ℂ}
    (hroot : ∑ i, c x₀ i * s₀ ^ (i : ℕ) = 0)
    (hsimple : ∑ i : Fin (d + 1), (i : ℕ) * c x₀ i * s₀ ^ ((i : ℕ) - 1) ≠ 0) :
    ∃ s : X → ℂ, s x₀ = s₀ ∧ ContinuousAt s x₀ ∧
      (∀ᶠ x in 𝓝 x₀, ∑ i, c x i * s x ^ (i : ℕ) = 0) ∧
      ∀ᶠ p in 𝓝 (x₀, s₀), ∑ i, c p.1 i * p.2 ^ (i : ℕ) = 0 ↔ p.2 = s p.1 :=
  exists_continuousAt_root_of_simple (continuousAt_pi.mpr hc) hroot hsimple

/-- **A square root of a simple root moves continuously**: for coefficients `c x` continuous at
`x₀`, a simple root `s₀ = u₀²` of `∑ᵢ cᵢ(x₀) sⁱ` and `u₀ ≠ 0`, there is a root `u(x)` of
`∑ᵢ cᵢ(x) u²ⁱ` near `x₀`, continuous at `x₀` with `u x₀ = u₀`, and near `(x₀, u₀)` it is the only
root. -/
theorem exists_continuousAt_sqrt_root_of_simple {X : Type*} [TopologicalSpace X] {x₀ : X} {d : ℕ}
    {c : X → Fin (d + 1) → ℂ} (hc : ContinuousAt c x₀) {u₀ : ℂ} (hu₀ : u₀ ≠ 0)
    (hroot : ∑ i, c x₀ i * (u₀ ^ 2) ^ (i : ℕ) = 0)
    (hsimple : ∑ i : Fin (d + 1), (i : ℕ) * c x₀ i * (u₀ ^ 2) ^ ((i : ℕ) - 1) ≠ 0) :
    ∃ u : X → ℂ, u x₀ = u₀ ∧ ContinuousAt u x₀ ∧
      (∀ᶠ x in 𝓝 x₀, ∑ i, c x i * (u x ^ 2) ^ (i : ℕ) = 0) ∧
      ∀ᶠ p in 𝓝 (x₀, u₀), ∑ i, c p.1 i * (p.2 ^ 2) ^ (i : ℕ) = 0 ↔ p.2 = u p.1 := by
  have hF : ContDiff ℝ 1 fun p : (Fin (d + 1) → ℂ) × ℂ ↦ ∑ i, p.1 i * (p.2 ^ 2) ^ (i : ℕ) := by
    fun_prop
  have hF' := (hasDerivAt_sum_mul_pow (c x₀) (u₀ ^ 2)).comp u₀ (hasDerivAt_pow 2 u₀)
  refine exists_continuousAt_zero_of_hasDerivAt hc hF.contDiffAt hroot hF' (mul_ne_zero hsimple ?_)
  simpa using hu₀

namespace Params

/-- `zₙ = w` iff `Re w = μ` and `(β Im w/π - 1 - k)/2 = n`. -/
theorem z_eq_iff (A : Params) (k n : ℤ) (w : ℂ) :
    A.z k n = w ↔ w.re = A.μ ∧ (w.im * A.β / Real.pi - 1 - k) / 2 = n := by
  have hβ := A.β_pos.ne'
  have hπ := Real.pi_ne_zero
  rw [A.z_eq, Complex.ext_iff]
  simp only [add_re, ofReal_re, mul_re, I_re, mul_zero, ofReal_im, I_im, mul_one, sub_self,
    add_zero, add_im, mul_im, zero_add]
  constructor <;> rintro ⟨h1, h2⟩ <;> refine ⟨h1.symm, ?_⟩ <;> rw [← h2] <;> field_simp <;> ring

/-- **The lattice points hit by a moving point**: along a family `A : X → Params` with `μ`, `β`
continuous at `x₀`, and `u(x)` continuous at `x₀`, near `x₀` the point `u(x)` is `zₙ` of `A x` only
for indices `n` with `u(x₀) = zₙ` at `x₀`, uniformly in `n`. -/
theorem eventually_z_eq_imp {X : Type*} [TopologicalSpace X] {x₀ : X} (A : X → Params) (k : ℤ)
    (hμ : ContinuousAt (fun x ↦ (A x).μ) x₀) (hβ : ContinuousAt (fun x ↦ (A x).β) x₀)
    {u : X → ℂ} (hu : ContinuousAt u x₀) :
    ∀ᶠ x in 𝓝 x₀, ∀ n, (A x).z k n = u x → (A x₀).z k n = u x₀ := by
  -- `u = zₙ` iff `g = (0, n)`
  set g : X → ℝ × ℝ := fun x ↦ ((u x).re - (A x).μ, ((u x).im * (A x).β / Real.pi - 1 - k) / 2)
  have hg : ContinuousAt g x₀ := by fun_prop
  -- the indices that `u(x₀)` misses form a closed set of `ℝ`, as every subset of `ℤ` does
  set B : Set (ℝ × ℝ) := {0} ×ˢ ((↑) '' {n : ℤ | (A x₀).z k n ≠ u x₀})
  have hB : IsClosed B :=
    isClosed_singleton.prod (Real.isClosedEmbedding_intCast.isClosedMap _ (isClosed_discrete _))
  have h₀' : g x₀ ∉ B := by
    rintro ⟨h1, n, hn, hgn⟩
    exact hn (((A x₀).z_eq_iff k n _).mpr ⟨sub_eq_zero.mp h1, hgn.symm⟩)
  filter_upwards [hg.eventually_mem (hB.isOpen_compl.mem_nhds h₀')] with x hx n hn
  by_contra hne
  obtain ⟨h1, h2⟩ := ((A x).z_eq_iff k n _).mp hn
  exact hx ⟨sub_eq_zero.mpr h1, n, hne, h2.symm⟩

/-- **Off the lattice is an open condition, uniformly in `n`**: along a family `A : X → Params`
with `μ`, `β` continuous at `x₀`, a point `u(x)` continuous at `x₀` with `u(x₀) ≠ zₙ` for all `n`
at `x₀` satisfies `u(x) ≠ zₙ` for all `n` near `x₀`. -/
theorem eventually_z_ne_of_off {X : Type*} [TopologicalSpace X] {x₀ : X} (A : X → Params) (k : ℤ)
    (hμ : ContinuousAt (fun x ↦ (A x).μ) x₀) (hβ : ContinuousAt (fun x ↦ (A x).β) x₀)
    {u : X → ℂ} (hu : ContinuousAt u x₀) (h₀ : ∀ n, (A x₀).z k n ≠ u x₀) :
    ∀ᶠ x in 𝓝 x₀, ∀ n, (A x).z k n ≠ u x :=
  (eventually_z_eq_imp A k hμ hβ hu).mono fun _ h n hn ↦ h₀ n (h n hn)

/-- **`S_k` at a moving point off the lattice is continuous**: along a family `A : X → Params` with
`μ`, `β` continuous at `x₀`, and `u(x)` continuous at `x₀` with `u(x₀) ≠ zₙ` for all `n` at `x₀`,
`S_k(u(x))` is continuous at `x₀`. -/
theorem continuousAt_S {X : Type*} [TopologicalSpace X] {x₀ : X} (A : X → Params) (k : ℤ)
    (hμ : ContinuousAt (fun x ↦ (A x).μ) x₀) (hβ : ContinuousAt (fun x ↦ (A x).β) x₀)
    {u : X → ℂ} (hu : ContinuousAt u x₀) (h₀ : ∀ n, (A x₀).z k n ≠ u x₀) :
    ContinuousAt (fun x ↦ (A x).S k (u x)) x₀ := by
  have hβ' := Complex.continuous_ofReal.continuousAt.comp hβ
  have hμ' := Complex.continuous_ofReal.continuousAt.comp hμ
  have hy : ContinuousAt (fun x ↦ ((A x).β : ℂ) * (u x - (A x).μ) / 2) x₀ :=
    (hβ'.mul (hu.sub hμ')).div_const 2
  have hpre : ContinuousAt (fun x ↦ -(((A x).β : ℂ) / 2)) x₀ := (hβ'.div_const 2).neg
  have hsh := Complex.continuous_sinh.continuousAt.comp hy
  have hch := Complex.continuous_cosh.continuousAt.comp hy
  unfold S
  by_cases hk : Even k
  · simp only [hk, ↓reduceIte, tanh_eq_sinh_div_cosh]
    obtain ⟨j, hj⟩ := hk
    exact hpre.mul (hsh.div hch ((A x₀).cosh_ne_zero_of_off (j := j) (by omega) h₀))
  · simp only [hk, ↓reduceIte, coth]
    obtain ⟨j, hj⟩ := Int.not_even_iff_odd.mp hk
    exact hpre.mul (hch.div hsh ((A x₀).sinh_ne_zero_of_off (j := j + 1) (by omega) h₀))

end Params

end HubbardAtom
