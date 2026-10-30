import HubbardAtomLean.ChiOperator

/-!
# The lattice of `Params` for every `μ`, as an `OpLattice`

`Params.opLattice` needs `μ ≠ 0`: at `μ = 0` and odd `k` the mirror fixed point `2n + 1 + k = 0`
has `zₙ = μ = 0`. Here that one point is moved to `μ + iπ/β`, for every `μ`, so that the lattice
points stay continuous in `μ`:

* `Params.zOp`: `z'ₙ = zₙ` except at the mirror fixed point, where `z'ₙ = μ + iπ/β`;
* `Params.zOp_ne_zero`, `Params.summable_inv_norm_zOp_sq`, `Params.opLatticeAll`: the
  `OpLattice` with these points;
* `Params.eventually_zOp_eq`, `Params.bdd_of_inv_zOp`: `z'ₙ = zₙ` for all but at most one `n`, so a
  function of `1/zₙ` continuous at `0` is bounded on the lattice as soon as it agrees with `fₙ`
  for all but finitely many `n`.

The bounds of item 6(a) change only at the moved point. The class `∑ₙ |vₙ/z'ₙ|² < ∞` of
`ChiOperator.lean` is that of `zₙ` wherever both are defined.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ)

/-- The lattice points for every `μ`: `z'ₙ = zₙ`, except at the mirror fixed point (odd `k`,
`2n + 1 + k = 0`, `zₙ = μ`), where `z'ₙ = μ + iπ/β`. -/
noncomputable def zOp (n : ℤ) : ℂ :=
  if mirror k n = n then A.μ + (Real.pi / A.β : ℝ) * I else A.z k n

theorem zOp_of_ne {n : ℤ} (h : mirror k n ≠ n) : A.zOp k n = A.z k n := by simp [zOp, h]

/-- `z'ₙ = zₙ` for all `n` but the mirror fixed point. -/
theorem eventually_zOp_eq : ∀ᶠ n in cofinite, A.zOp k n = A.z k n := by
  filter_upwards [eventually_cofinite_ne ((-k - 1) / 2)] with n hn
  exact A.zOp_of_ne k (by unfold mirror; omega)

/-- `z'ₙ ≠ 0`: its imaginary part is `π/β` at the fixed point and `π(2n + 1 + k)/β ≠ 0` off it. -/
theorem zOp_ne_zero (n : ℤ) : A.zOp k n ≠ 0 := by
  have hβ := A.β_pos
  intro h
  have him := congrArg Complex.im h
  unfold zOp at him
  split_ifs at him with hn
  · simp only [add_im, ofReal_im, mul_im, ofReal_re, I_im, mul_one, I_re, mul_zero, add_zero,
      zero_add, zero_im] at him
    exact (div_pos Real.pi_pos hβ).ne' him
  · rw [A.z_eq] at him
    simp only [add_im, ofReal_im, mul_im, ofReal_re, I_im, mul_one, I_re, mul_zero, add_zero,
      zero_add, zero_im] at him
    have h0 : (2 * n + 1 + k : ℝ) ≠ 0 := by
      have : 2 * n + 1 + k ≠ 0 := fun h0 ↦ hn (by unfold mirror; omega)
      exact_mod_cast this
    exact div_ne_zero (mul_ne_zero h0 Real.pi_ne_zero) hβ.ne' him

/-- `∑ₙ |z'ₙ|⁻²` converges: `z'ₙ = zₙ` for all but at most one `n`. -/
theorem summable_inv_norm_zOp_sq : Summable fun n ↦ ‖A.zOp k n‖⁻¹ ^ 2 := by
  have h : Summable fun n ↦ ‖A.z k n‖⁻¹ ^ 2 :=
    (A.summable_inv_z_sub_sq k 0).norm.congr fun n ↦ by simp
  exact h.congr_cofinite ((A.eventually_zOp_eq k).mono fun n h ↦ by simp only [h])

/-- **The lattice of `Params` for every `μ`** as an `OpLattice`, with the mirror fixed point
moved to `μ + iπ/β`. -/
noncomputable def opLatticeAll : OpLattice :=
  ⟨A.β, A.β_pos, A.zOp k, A.zOp_ne_zero k, A.summable_inv_norm_zOp_sq k⟩

/-- A function of `1/z'ₙ` continuous at `0` is bounded on the lattice, as soon as it agrees with
`fₙ` for all but finitely many `n`. -/
theorem bdd_of_inv_zOp {f : ℤ → ℂ} (q : ℂ → ℂ) (hq : ContinuousAt q 0)
    (hf : ∀ᶠ n in cofinite, f n = q (A.zOp k n)⁻¹) : ∃ C, ∀ n, ‖f n‖ ≤ C := by
  obtain ⟨C, hC⟩ := ((hq.tendsto.comp (A.opLatticeAll k).tendsto_inv_z).congr'
    (hf.mono fun n h ↦ h.symm)).norm.bddAbove_range_of_cofinite
  exact ⟨C, fun n ↦ hC ⟨n, rfl⟩⟩

end Params

end HubbardAtom
