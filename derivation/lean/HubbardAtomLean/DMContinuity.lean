import HubbardAtomLean.DMOperator
import HubbardAtomLean.GammaContinuity
import HubbardAtomLean.ChiContinuity

/-!
# Continuity of the density and magnetic vertex for every `μ`

The second half of item 6(d) for the density and magnetic channels, through half filling: along a
family of parameters `(U, β, μ)` continuous at `x₀`, the vertex `Γ_r` of `DMOperator.lean` is
continuous at `x₀` wherever `1 + 𝒦` is invertible there, `μ(x₀) = 0` included.

`GammaContinuity.lean` covers `μ(x₀) ≠ 0` on the lattice `zₙ`; its bounds use `|1/zₙ| ≤ 1/|μ|`
and `|1/(zₙ + zₘ)| ≤ 1/(2|μ|)`. Here the lattice is `z'ₙ` of `OpLatticeAll.lean`, and the bounds
are uniform through `μ = 0`:

* off the mirror fixed point, `|1/zₙ| ≤ β/π` (`norm_inv_z_le_of_ne`), and off the mirror pairs,
  `|1/(zₙ + zₘ)| ≤ β/(2π)` (`norm_inv_z_add_z_le_of_ne`); always `|P(zₙ)| ≥ (π/β)⁴`;
* on a mirror pair, Proposition "structure" holds with the pair coefficient `c₀ = ρ/(2μ)`,
  continued by `βp₀` at `μ = 0` (`Pph_mul_chiR_mirror_all`), and `c₀` is continuous in the
  parameters (`pairC_zero_eq`: `c₀ = e^{-βU/4}β sinhc(βμ)/Z`);
* off the mirror pairs, `zₘ²P(zₙ)χ_r,nm = (zₘ⁴/P(zₘ))[r₃(tₘ) + zₙ²n₂(tₘ)] - λρzₙ + λρzₙ²/(zₙ + zₘ)`
  (`z_sq_mul_Pph_mul_chiR`), bounded also in the row of the fixed point, where `zₙ = μ`.

* `Params.norm_chiR_le_all_of`, `Params.norm_chiR_diag_le_all_of`: the bounds of item 6(a) on
  `z'ₙ`, with constants;
* `Params.continuousAt_gammaOpR`: **`Γ_r,nm` is continuous** at `x₀` wherever `1 + 𝒦` is
  invertible there.
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-! ### The lattice -/

/-- `zₙ + z_n̄ = 2μ`. -/
theorem z_add_z_mirror (n : ℤ) : A.z k n + A.z k (mirror k n) = 2 * A.μ := by
  rw [A.z_eq, A.z_eq]
  unfold mirror
  push_cast
  ring

/-- `Im zₙ = (2n + 1 + k)π/β`. -/
theorem z_im (n : ℤ) : (A.z k n).im = (2 * n + 1 + k) * Real.pi / A.β := by
  rw [A.z_eq]
  simp

/-- `|1/zₙ| ≤ β/π` off the mirror fixed point, where `|Im zₙ| ≥ π/β`. -/
theorem norm_inv_z_le_of_ne {n : ℤ} (hn : mirror k n ≠ n) : ‖(A.z k n)⁻¹‖ ≤ A.β / Real.pi := by
  have hβ := A.β_pos
  have hπ := Real.pi_pos
  have h1 : (1 : ℝ) ≤ |(2 * n + 1 + k : ℝ)| := by
    have : (1 : ℤ) ≤ |2 * n + 1 + k| := by
      have : 2 * n + 1 + k ≠ 0 := fun h0 ↦ hn (by unfold mirror; omega)
      rcases abs_cases (2 * n + 1 + k) with ⟨h, _⟩ | ⟨h, _⟩ <;> omega
    exact_mod_cast this
  have hz : Real.pi / A.β ≤ ‖A.z k n‖ := by
    refine le_trans ?_ (abs_im_le_norm _)
    rw [A.z_im, abs_div, abs_mul, abs_of_pos hπ, abs_of_pos hβ]
    exact div_le_div_of_nonneg_right (le_mul_of_one_le_left hπ.le h1) hβ.le
  rw [norm_inv]
  calc ‖A.z k n‖⁻¹ ≤ (Real.pi / A.β)⁻¹ := inv_anti₀ (by positivity) hz
    _ = A.β / Real.pi := inv_div _ _

/-- `|1/(zₙ + zₘ)| ≤ β/(2π)` off the mirror pairs, where `|Im(zₙ + zₘ)| ≥ 2π/β`. -/
theorem norm_inv_z_add_z_le_of_ne {n m : ℤ} (hnm : m ≠ mirror k n) :
    ‖(A.z k n + A.z k m)⁻¹‖ ≤ A.β / (2 * Real.pi) := by
  have hβ := A.β_pos
  have hπ := Real.pi_pos
  have h1 : (1 : ℝ) ≤ |(n + m + k + 1 : ℝ)| := by
    have : (1 : ℤ) ≤ |n + m + k + 1| := by
      have : n + m + k + 1 ≠ 0 := fun h0 ↦ hnm (by unfold mirror; omega)
      rcases abs_cases (n + m + k + 1) with ⟨h, _⟩ | ⟨h, _⟩ <;> omega
    exact_mod_cast this
  have him : (A.z k n + A.z k m).im = 2 * ((n + m + k + 1) * Real.pi / A.β) := by
    rw [add_im, A.z_im, A.z_im]
    ring
  have hz : 2 * Real.pi / A.β ≤ ‖A.z k n + A.z k m‖ := by
    refine le_trans ?_ (abs_im_le_norm _)
    rw [him, abs_mul, abs_two, abs_div, abs_mul, abs_of_pos hπ, abs_of_pos hβ, mul_div_assoc']
    exact div_le_div_of_nonneg_right (by nlinarith) hβ.le
  rw [norm_inv]
  calc ‖A.z k n + A.z k m‖⁻¹ ≤ (2 * Real.pi / A.β)⁻¹ := inv_anti₀ (by positivity) hz
    _ = A.β / (2 * Real.pi) := inv_div _ _

/-- `|xⱼ + r| ≥ π/β` for real `r`, as `|Im xⱼ| = |νⱼ| ≥ π/β`. -/
theorem norm_x_add_ge (j : ℤ) (r : ℝ) : Real.pi / A.β ≤ ‖A.x j + r‖ := by
  have hβ := A.β_pos
  have hπ := Real.pi_pos
  have h1 : (1 : ℝ) ≤ |(2 * j + 1 : ℝ)| := by
    have : (1 : ℤ) ≤ |2 * j + 1| := by
      rcases abs_cases (2 * j + 1) with ⟨h, _⟩ | ⟨h, _⟩ <;> omega
    exact_mod_cast this
  refine le_trans ?_ (abs_im_le_norm _)
  have him : (A.x j + r).im = (2 * j + 1) * Real.pi / A.β := by simp [x, ν]
  rw [him, abs_div, abs_mul, abs_of_pos hπ, abs_of_pos hβ]
  exact div_le_div_of_nonneg_right (le_mul_of_one_le_left hπ.le h1) hβ.le

/-- **`|1/P(zₙ)| ≤ (β/π)⁴`**, from `P(zₙ) = (xₙ + a)(x_{n+k} + a)(xₙ - a)(x_{n+k} - a)`. -/
theorem norm_inv_Pph_z_le (n : ℤ) : ‖(A.Pph k (A.z k n))⁻¹‖ ≤ (A.β / Real.pi) ^ 4 := by
  have hβ := A.β_pos
  have hπ := Real.pi_pos
  have hP : A.Pph k (A.z k n) = (A.x n + A.a) * (A.x (n + k) + A.a) *
      ((A.x n + (-A.a : ℝ)) * (A.x (n + k) + (-A.a : ℝ))) := by
    rw [Pph, Qp_z, Qm_z]
    push_cast
    ring
  have h₁ := A.norm_x_add_ge n A.a
  have h₂ := A.norm_x_add_ge (n + k) A.a
  have h₃ := A.norm_x_add_ge n (-A.a)
  have h₄ := A.norm_x_add_ge (n + k) (-A.a)
  have hc : 0 < Real.pi / A.β := by positivity
  have hle : (Real.pi / A.β) ^ 4 ≤ ‖A.Pph k (A.z k n)‖ := by
    rw [hP, norm_mul, norm_mul, norm_mul]
    calc (Real.pi / A.β) ^ 4 = Real.pi / A.β * (Real.pi / A.β) *
        (Real.pi / A.β * (Real.pi / A.β)) := by ring
      _ ≤ _ := by gcongr
  rw [norm_inv, show A.β / Real.pi = (Real.pi / A.β)⁻¹ from (inv_div _ _).symm, inv_pow]
  exact inv_anti₀ (by positivity) hle

/-! ### The pair coefficient at `ℓ = 0` -/

/-- **The pair coefficient at `ℓ = 0` for every `μ`**: `c₀ = ρ/(2μ)`, continued by `βp₀` at `μ = 0`,
is `e^{-βU/4}β sinhc(βμ)/Z` with `sinhc = dslope sinh 0`, continuous in the parameters. -/
theorem pairC_zero_eq : A.pairC 0 =
    ((Real.exp (-A.β * (A.U / 4)) * A.β * dslope Real.sinh 0 (A.β * A.μ) / A.Z : ℝ) : ℂ) := by
  have hZ := A.Z_pos.ne'
  have hβ := A.β_pos.ne'
  by_cases hμ : A.μ = 0
  · rw [pairC, ite_eq_left ⟨hμ, rfl⟩, hμ, mul_zero, dslope_same, Real.deriv_sinh, Real.cosh_zero]
    congr 1
    rw [p₀, boltz₀, hμ, add_zero]
    field_simp
  · have hω : A.ω 0 = 0 := by simp [ω]
    have hbm : A.β * A.μ ≠ 0 := mul_ne_zero hβ hμ
    rw [pairC, ite_eq_right (fun h ↦ hμ h.1), hω, ofReal_zero, zero_mul, add_zero,
      dslope_of_ne _ hbm, slope_def_field, sub_zero, Real.sinh_zero, sub_zero,
      show (2 : ℂ) * A.μ = ((2 * A.μ : ℝ) : ℂ) by push_cast; ring, ← ofReal_div]
    congr 1
    have e2 : Real.exp (-A.β * (A.U / 4 - A.μ)) =
        Real.exp (-A.β * (A.U / 4)) * Real.exp (A.β * A.μ) := by
      rw [← Real.exp_add]
      ring_nf
    have e0 : Real.exp (-A.β * (A.U / 4 + A.μ)) =
        Real.exp (-A.β * (A.U / 4)) * Real.exp (-(A.β * A.μ)) := by
      rw [← Real.exp_add]
      ring_nf
    rw [ρ, p₂, p₀, boltz₂, boltz₀, e2, e0, Real.sinh_eq]
    field_simp

/-! ### Proposition "structure" for every `μ` -/

/-- **Proposition "structure" on the mirror, for every `μ`**: with `c₀` the pair coefficient at
`ℓ = 0`, `P(zₙ)χ_r,nn̄ = -βH(zₙ)δₙₙ̄ + λc₀ + N(zₙ, zₙ̄)/P(zₙ̄)`. -/
theorem Pph_mul_chiR_mirror_all (n : ℤ) :
    A.Pph k (A.z k n) * A.chiR k ε n (mirror k n) =
      -A.β * A.Hph k ε (A.z k n) * kron n (mirror k n) + A.lam ε * A.pairC 0 +
        A.Nph k ε (A.z k n) (A.z k (mirror k n)) / A.Pph k (A.z k (mirror k n)) := by
  by_cases hμ : A.μ = 0
  · rw [A.Pph_mul_chiR_mirror k ε hμ n, pairC, ite_eq_left ⟨hμ, rfl⟩]
    push_cast
    ring
  · have hzz : A.z k n + A.z k (mirror k n) ≠ 0 := by
      rw [A.z_add_z_mirror]
      exact mul_ne_zero two_ne_zero (ofReal_ne_zero.mpr hμ)
    have hω : A.ω 0 = 0 := by simp [ω]
    rw [A.Pph_mul_chiR' k ε hzz, A.z_add_z_mirror, pairC, ite_eq_right (fun h ↦ hμ h.1), hω,
      ofReal_zero, zero_mul, add_zero, mul_div_assoc]

/-- **Off the mirror pairs, for every `μ`**: for `n ≠ m`, `zₘ ≠ 0`, `zₙ + zₘ ≠ 0`,
`zₘ²P(zₙ)χ_r,nm = (zₘ⁴/P(zₘ))[r₃(tₘ) + zₙ²n₂(tₘ)] - λρzₙ + λρzₙ²/(zₙ + zₘ)`, `tₘ = 1/zₘ`. -/
theorem z_sq_mul_Pph_mul_chiR {n m : ℤ} (hnm : n ≠ m) (hv : A.z k m ≠ 0)
    (hzv : A.z k n + A.z k m ≠ 0) :
    A.z k m ^ 2 * (A.Pph k (A.z k n) * A.chiR k ε n m) =
      A.z k m ^ 4 / A.Pph k (A.z k m) *
          (A.r3t k ε (A.z k m)⁻¹ + A.z k n ^ 2 * A.n2t k ε (A.z k m)⁻¹) -
        A.lam ε * A.ρ * A.z k n + A.lam ε * A.ρ * A.z k n ^ 2 / (A.z k n + A.z k m) := by
  have hPv := A.Pph_z_ne_zero k m
  rw [A.Pph_mul_chiR' k ε hzv, kron_of_ne hnm]
  simp only [Nph, N₀, N₂, n2t, r3t, coefN₂, coefN₀, lam]
  have eQ := A.Pph_eq k (A.z k m)
  generalize A.Pph k (A.z k m) = Pv at hPv eQ ⊢
  field_simp
  subst eQ
  ring

/-- **At a fixed point in the column**: for `n ≠ m`, `zₙ ≠ 0`, `zₙ + zₘ ≠ 0`,
`zₙ²χ_r,nm = (zₙ⁴/P(zₙ))[λρtₙ²/(zₙ + zₘ) + (tₙ²N₀(zₘ) + N₂(zₘ))/P(zₘ)]`, `tₙ = 1/zₙ`. -/
theorem z_sq_mul_chiR_col {n m : ℤ} (hnm : n ≠ m) (hz : A.z k n ≠ 0)
    (hzv : A.z k n + A.z k m ≠ 0) :
    A.z k n ^ 2 * A.chiR k ε n m = A.z k n ^ 4 / A.Pph k (A.z k n) *
      (A.lam ε * A.ρ * (A.z k n)⁻¹ ^ 2 / (A.z k n + A.z k m) +
        ((A.z k n)⁻¹ ^ 2 * A.N₀ k ε (A.z k m) + A.N₂ k ε (A.z k m)) / A.Pph k (A.z k m)) := by
  have hPz := A.Pph_z_ne_zero k n
  have hPv := A.Pph_z_ne_zero k m
  have hχ : A.chiR k ε n m = (A.lam ε * A.ρ / (A.z k n + A.z k m) +
      A.Nph k ε (A.z k n) (A.z k m) / A.Pph k (A.z k m)) / A.Pph k (A.z k n) := by
    rw [eq_div_iff hPz, mul_comm, A.Pph_mul_chiR' k ε hzv, kron_of_ne hnm]
    ring
  rw [hχ, Nph]
  field_simp

/-- **On a mirror pair**: for `zₙ, zₙ̄ ≠ 0`, `n ≠ n̄`, with `v = zₙ̄`, `t = 1/v`,
`zₙ²v²χ_r,nn̄ = (zₙ⁴/P(zₙ))(v/zₙ)²[λc₀ + (v⁴/P(v))(t n₀(t) + (zₙ/v)²n₂(t))]`. -/
theorem z_sq_mul_chiR_mirror {n : ℤ} (hn : mirror k n ≠ n) (hz : A.z k n ≠ 0)
    (hv : A.z k (mirror k n) ≠ 0) :
    A.z k n ^ 2 * A.z k (mirror k n) ^ 2 * A.chiR k ε n (mirror k n) =
      A.z k n ^ 4 / A.Pph k (A.z k n) * (A.z k (mirror k n) / A.z k n) ^ 2 *
        (A.lam ε * A.pairC 0 + A.z k (mirror k n) ^ 4 / A.Pph k (A.z k (mirror k n)) *
          ((A.z k (mirror k n))⁻¹ * A.n0t k ε (A.z k (mirror k n))⁻¹ +
            (A.z k n / A.z k (mirror k n)) ^ 2 * A.n2t k ε (A.z k (mirror k n))⁻¹)) := by
  have hPz := A.Pph_z_ne_zero k n
  have hPv := A.Pph_z_ne_zero k (mirror k n)
  have hχ : A.chiR k ε n (mirror k n) = (A.lam ε * A.pairC 0 +
      A.Nph k ε (A.z k n) (A.z k (mirror k n)) / A.Pph k (A.z k (mirror k n))) /
        A.Pph k (A.z k n) := by
    rw [eq_div_iff hPz, mul_comm, A.Pph_mul_chiR_mirror_all k ε n, kron_of_ne (Ne.symm hn)]
    ring
  rw [hχ]
  simp only [Nph, N₀, N₂, n0t, n2t, coefN₂, coefN₀, lam]
  have eQ := A.Pph_eq k (A.z k (mirror k n))
  generalize A.Pph k (A.z k (mirror k n)) = Pv at hPv eQ ⊢
  field_simp
  subst eQ
  ring

/-- **On the diagonal**, for `zₙ ≠ 0`:
`zₙ³(χ_r,nn + β/zₙ²) = (zₙ⁴/P(zₙ))[dgt(tₙ) + (zₙ⁴/P(zₙ))(tₙ²n₀(tₙ) + tₙn₂(tₙ))]`. -/
theorem z_cube_mul_chiR_diag {n : ℤ} (hz : A.z k n ≠ 0) :
    A.z k n ^ 3 * (A.chiR k ε n n + A.β / A.z k n ^ 2) =
      A.z k n ^ 4 / A.Pph k (A.z k n) * (A.dgt k ε (A.z k n)⁻¹ +
        A.z k n ^ 4 / A.Pph k (A.z k n) *
          ((A.z k n)⁻¹ ^ 2 * A.n0t k ε (A.z k n)⁻¹ + (A.z k n)⁻¹ * A.n2t k ε (A.z k n)⁻¹)) := by
  have hPz := A.Pph_z_ne_zero k n
  have hzz : A.z k n + A.z k n ≠ 0 := by
    rw [← two_mul]
    exact mul_ne_zero two_ne_zero hz
  have hχ : A.chiR k ε n n = (-A.β * A.Hph k ε (A.z k n) + A.lam ε * A.ρ / (2 * A.z k n) +
      A.Nph k ε (A.z k n) (A.z k n) / A.Pph k (A.z k n)) / A.Pph k (A.z k n) := by
    rw [eq_div_iff hPz, mul_comm, A.Pph_mul_chiR' k ε hzz, kron_self, two_mul]
    ring
  rw [hχ]
  simp only [Nph, N₀, N₂, n2t, n0t, dgt, coefN₂, coefN₀, Hph, lam]
  have eP := A.Pph_eq k (A.z k n)
  generalize A.Pph k (A.z k n) = Pz at hPz eP ⊢
  field_simp
  subst eP
  ring

/-! ### The bounds of item 6(a) on `z'ₙ`, with constants -/

private theorem nmul {a b : ℂ} {A B : ℝ} (ha : ‖a‖ ≤ A) (hb : ‖b‖ ≤ B) (hA : 0 ≤ A) :
    ‖a * b‖ ≤ A * B := by
  rw [norm_mul]
  exact mul_le_mul ha hb (norm_nonneg _) hA

private theorem nadd {a b : ℂ} {A B : ℝ} (ha : ‖a‖ ≤ A) (hb : ‖b‖ ≤ B) : ‖a + b‖ ≤ A + B :=
  (norm_add_le _ _).trans (add_le_add ha hb)

private theorem nsub {a b : ℂ} {A B : ℝ} (ha : ‖a‖ ≤ A) (hb : ‖b‖ ≤ B) : ‖a - b‖ ≤ A + B :=
  (norm_sub_le _ _).trans (add_le_add ha hb)

/-- `zₙ + zₘ ≠ 0` off the mirror pairs. -/
theorem z_add_z_ne_zero_of_ne {n m : ℤ} (hnm : m ≠ mirror k n) : A.z k n + A.z k m ≠ 0 := by
  intro h
  have hz := congrArg Complex.im h
  rw [add_im, A.z_im, A.z_im, zero_im] at hz
  have : n + m + k + 1 ≠ 0 := fun h0 ↦ hnm (by unfold mirror; omega)
  have h2 : (n + m + k + 1 : ℝ) ≠ 0 := by exact_mod_cast this
  have hβ := A.β_pos.ne'
  have : (n + m + k + 1 : ℝ) * (2 * Real.pi / A.β) = 0 := by
    rw [← hz]
    field_simp
    ring
  rcases mul_eq_zero.mp this with h0 | h0
  · exact h2 h0
  · exact div_ne_zero (mul_ne_zero two_ne_zero Real.pi_ne_zero) hβ h0

/-- `zₙ ≠ 0` off the mirror fixed point. -/
theorem z_ne_zero_of_ne {n : ℤ} (hn : mirror k n ≠ n) : A.z k n ≠ 0 := by
  intro h
  have h' := A.norm_inv_z_le_of_ne k hn
  have hz := congrArg Complex.im h
  rw [A.z_im, zero_im] at hz
  have : 2 * n + 1 + k ≠ 0 := fun h0 ↦ hn (by unfold mirror; omega)
  have h2 : (2 * n + 1 + k : ℝ) ≠ 0 := by exact_mod_cast this
  exact div_ne_zero (mul_ne_zero h2 Real.pi_ne_zero) A.β_pos.ne' hz

/-- At the fixed point, `zₙ = μ`. -/
theorem z_of_fixed {n : ℤ} (hn : mirror k n = n) : A.z k n = A.μ := by
  have h : (2 * n + 1 + k : ℝ) = 0 := by
    have : 2 * n + 1 + k = 0 := by unfold mirror at hn; omega
    exact_mod_cast this
  rw [A.z_eq, h]
  simp

/-- At the fixed point, `|z'ₙ| ≤ |μ| + π/β`. -/
theorem norm_zOp_fixed_le {n : ℤ} (hn : mirror k n = n) :
    ‖A.zOp k n‖ ≤ |A.μ| + Real.pi / A.β := by
  have hc : 0 ≤ Real.pi / A.β := by have := A.β_pos; positivity
  rw [zOp, ite_eq_left hn]
  refine (norm_add_le _ _).trans ?_
  rw [norm_mul, norm_I, mul_one, norm_real, norm_real, Real.norm_eq_abs,
    Real.norm_of_nonneg hc]

/-- `|zₙ̄/zₙ| ≤ 1 + 2MR` off the fixed point, for `|μ| ≤ M`, `β/π ≤ R`: `zₙ̄ = 2μ - zₙ`. -/
theorem norm_z_mirror_div_le {n : ℤ} (hn : mirror k n ≠ n) {M R : ℝ} (hM : |A.μ| ≤ M)
    (hR : A.β / Real.pi ≤ R) : ‖A.z k (mirror k n) / A.z k n‖ ≤ 1 + 2 * M * R := by
  have hz0 := A.z_ne_zero_of_ne k hn
  have ht := A.norm_inv_z_le_of_ne k hn
  have hv : A.z k (mirror k n) = ((2 * A.μ : ℝ) : ℂ) - A.z k n := by
    have := A.z_add_z_mirror k n
    push_cast
    linear_combination this
  have h2 : ‖((2 * A.μ : ℝ) : ℂ)‖ ≤ 2 * M := by
    rw [norm_real, Real.norm_eq_abs, abs_mul, abs_two]
    linarith
  rw [hv, sub_div, div_self hz0, div_eq_mul_inv]
  calc ‖((2 * A.μ : ℝ) : ℂ) * (A.z k n)⁻¹ - 1‖ ≤ ‖((2 * A.μ : ℝ) : ℂ)‖ * ‖(A.z k n)⁻¹‖ + 1 := by
        refine (norm_sub_le _ _).trans ?_
        rw [norm_mul, norm_one]
    _ ≤ 2 * M * R + 1 := by
        have := mul_le_mul h2 (ht.trans hR) (norm_nonneg _) (by linarith [abs_nonneg A.μ])
        linarith
    _ = 1 + 2 * M * R := by ring

/-- **Off-diagonal entries on `z'ₙ`, with constants**: for `n ≠ m`,
`|z'ₙ|²|z'ₘ|²|χ_r,nm|` is bounded by the constants below. -/
theorem norm_chiR_le_all_of {B R M Λ Cp C₂ C₃ C₄ CN : ℝ}
    (hR : A.β / Real.pi ≤ R) (hM : |A.μ| ≤ M) (hMπ : Real.pi / A.β ≤ M)
    (hB : ∀ j, ‖A.z k j ^ 4 / A.Pph k (A.z k j)‖ ≤ B)
    (hΛ : ‖A.lam ε‖ ≤ Λ) (hΛρ : ‖A.lam ε * A.ρ‖ ≤ Λ) (hp : ‖A.pairC 0‖ ≤ Cp)
    (hC₂ : ∀ j, mirror k j ≠ j → ‖A.n2t k ε (A.z k j)⁻¹‖ ≤ C₂)
    (hC₃ : ∀ j, mirror k j ≠ j → ‖A.n0t k ε (A.z k j)⁻¹‖ ≤ C₃)
    (hC₄ : ∀ j, mirror k j ≠ j → ‖A.r3t k ε (A.z k j)⁻¹‖ ≤ C₄)
    (hN₀ : ∀ j, mirror k j = j → ‖A.N₀ k ε (A.z k j)‖ ≤ CN)
    (hN₂ : ∀ j, mirror k j = j → ‖A.N₂ k ε (A.z k j)‖ ≤ CN)
    (hC₂0 : 0 ≤ C₂) (hC₃0 : 0 ≤ C₃) (hC₄0 : 0 ≤ C₄) (hCN0 : 0 ≤ CN)
    (n m : ℤ) (hnm : n ≠ m) :
    ‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2 * ‖A.chiR k ε n m‖ ≤
      B * (1 + 2 * M * R) ^ 2 * (Λ * Cp + B * (R * C₃ + (1 + 2 * M * R) ^ 2 * C₂)) +
      B * (B * (R ^ 2 * C₄ + C₂) + Λ * R + Λ * R) +
      4 * M ^ 2 * (R ^ 4 * (B * (C₄ + M ^ 2 * C₂) + Λ * M + Λ * M ^ 2 * R)) +
      4 * M ^ 2 * (B * (Λ * R ^ 2 * R + (R ^ 2 * CN + CN) * R ^ 4)) := by
  have hβ := A.β_pos
  have hπ := Real.pi_pos
  have hR0 : 0 ≤ R := (div_nonneg hβ.le hπ.le).trans hR
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hM
  have hB0 : 0 ≤ B := (norm_nonneg _).trans (hB 0)
  have hΛ0 : 0 ≤ Λ := (norm_nonneg _).trans hΛ
  have hCp0 : 0 ≤ Cp := (norm_nonneg _).trans hp
  have hs2 : A.β / (2 * Real.pi) ≤ R :=
    (div_le_div_of_nonneg_left hβ.le hπ (by linarith)).trans hR
  have hP4 : ∀ j, ‖(A.Pph k (A.z k j))⁻¹‖ ≤ R ^ 4 := fun j ↦
    (A.norm_inv_Pph_z_le k j).trans (pow_le_pow_left₀ (by positivity) hR 4)
  have hfix : ∀ j, mirror k j = j → ‖A.zOp k j‖ ^ 2 ≤ 4 * M ^ 2 := fun j hj ↦ by
    have h := A.norm_zOp_fixed_le k hj
    have h0 := norm_nonneg (A.zOp k j)
    nlinarith
  -- the four cases are each bounded by one summand
  set c₁ := B * (1 + 2 * M * R) ^ 2 * (Λ * Cp + B * (R * C₃ + (1 + 2 * M * R) ^ 2 * C₂))
  set c₂ := B * (B * (R ^ 2 * C₄ + C₂) + Λ * R + Λ * R)
  set c₃ := 4 * M ^ 2 * (R ^ 4 * (B * (C₄ + M ^ 2 * C₂) + Λ * M + Λ * M ^ 2 * R))
  set c₄ := 4 * M ^ 2 * (B * (Λ * R ^ 2 * R + (R ^ 2 * CN + CN) * R ^ 4))
  have hc₁ : 0 ≤ c₁ := by positivity
  have hc₂ : 0 ≤ c₂ := by positivity
  have hc₃ : 0 ≤ c₃ := by positivity
  have hc₄ : 0 ≤ c₄ := by positivity
  by_cases hnf : mirror k n = n
  · -- the row of the fixed point, `zₙ = μ`
    have hmf : mirror k m ≠ m := fun h ↦ hnm (by unfold mirror at hnf h; omega)
    have hmm : m ≠ mirror k n := by rw [hnf]; exact Ne.symm hnm
    have hv0 := A.z_ne_zero_of_ne k hmf
    have hzv := A.z_add_z_ne_zero_of_ne k hmm
    have hPz := A.Pph_z_ne_zero k n
    have h := A.z_sq_mul_Pph_mul_chiR k ε hnm hv0 hzv
    have e : A.z k m ^ 2 * A.chiR k ε n m = (A.Pph k (A.z k n))⁻¹ *
        (A.z k m ^ 2 * (A.Pph k (A.z k n) * A.chiR k ε n m)) := by
      field_simp
    rw [h] at e
    have hzn : ‖A.z k n‖ ≤ M := by
      rw [A.z_of_fixed k hnf, norm_real, Real.norm_eq_abs]
      exact hM
    have hzn2 : ‖A.z k n ^ 2‖ ≤ M ^ 2 := by
      rw [norm_pow]
      exact pow_le_pow_left₀ (norm_nonneg _) hzn 2
    have hs : ‖(A.z k n + A.z k m)⁻¹‖ ≤ R := (A.norm_inv_z_add_z_le_of_ne k hmm).trans hs2
    have hrhs : ‖A.z k m ^ 4 / A.Pph k (A.z k m) *
          (A.r3t k ε (A.z k m)⁻¹ + A.z k n ^ 2 * A.n2t k ε (A.z k m)⁻¹) -
        A.lam ε * A.ρ * A.z k n + A.lam ε * A.ρ * A.z k n ^ 2 / (A.z k n + A.z k m)‖ ≤
        B * (C₄ + M ^ 2 * C₂) + Λ * M + Λ * M ^ 2 * R := by
      rw [div_eq_mul_inv (A.lam ε * A.ρ * A.z k n ^ 2)]
      exact nadd (nsub (nmul (hB m) (nadd (hC₄ m hmf) (nmul hzn2 (hC₂ m hmf) (by positivity)))
        hB0) (nmul hΛρ hzn hΛ0)) (nmul (nmul hΛρ hzn2 hΛ0) hs (by positivity))
    have hb : ‖A.z k m‖ ^ 2 * ‖A.chiR k ε n m‖ ≤
        R ^ 4 * (B * (C₄ + M ^ 2 * C₂) + Λ * M + Λ * M ^ 2 * R) := by
      rw [← norm_pow, ← norm_mul, e]
      exact nmul (hP4 n) hrhs (by positivity)
    have hzOm : ‖A.zOp k m‖ = ‖A.z k m‖ := by rw [A.zOp_of_ne k hmf]
    calc ‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2 * ‖A.chiR k ε n m‖ =
        ‖A.zOp k n‖ ^ 2 * (‖A.z k m‖ ^ 2 * ‖A.chiR k ε n m‖) := by rw [hzOm]; ring
      _ ≤ c₃ := mul_le_mul (hfix n hnf) hb (by positivity) (by positivity)
      _ ≤ c₁ + c₂ + c₃ + c₄ := by linarith
  have hz0 := A.z_ne_zero_of_ne k hnf
  have htn : ‖(A.z k n)⁻¹‖ ≤ R := (A.norm_inv_z_le_of_ne k hnf).trans hR
  have hzOn : ‖A.zOp k n‖ = ‖A.z k n‖ := by rw [A.zOp_of_ne k hnf]
  by_cases hmf : mirror k m = m
  · -- the column of the fixed point, `zₘ = μ`
    have hmm : m ≠ mirror k n := fun h ↦ hnm (by
      have := congrArg (mirror k) h
      rw [mirror_mirror, hmf] at this
      exact this.symm)
    have hzv := A.z_add_z_ne_zero_of_ne k hmm
    have h := A.z_sq_mul_chiR_col k ε hnm hz0 hzv
    have hs : ‖(A.z k n + A.z k m)⁻¹‖ ≤ R := (A.norm_inv_z_add_z_le_of_ne k hmm).trans hs2
    have ht2 : ‖(A.z k n)⁻¹ ^ 2‖ ≤ R ^ 2 := by
      rw [norm_pow]
      exact pow_le_pow_left₀ (norm_nonneg _) htn 2
    have hrhs : ‖A.lam ε * A.ρ * (A.z k n)⁻¹ ^ 2 / (A.z k n + A.z k m) +
        ((A.z k n)⁻¹ ^ 2 * A.N₀ k ε (A.z k m) + A.N₂ k ε (A.z k m)) / A.Pph k (A.z k m)‖ ≤
        Λ * R ^ 2 * R + (R ^ 2 * CN + CN) * R ^ 4 := by
      rw [div_eq_mul_inv (A.lam ε * A.ρ * (A.z k n)⁻¹ ^ 2), div_eq_mul_inv (_ + _)]
      exact nadd (nmul (nmul hΛρ ht2 hΛ0) hs (by positivity))
        (nmul (nadd (nmul ht2 (hN₀ m hmf) (by positivity)) (hN₂ m hmf)) (hP4 m) (by positivity))
    have hb : ‖A.z k n‖ ^ 2 * ‖A.chiR k ε n m‖ ≤
        B * (Λ * R ^ 2 * R + (R ^ 2 * CN + CN) * R ^ 4) := by
      rw [← norm_pow, ← norm_mul, h]
      exact nmul (hB n) hrhs hB0
    calc ‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2 * ‖A.chiR k ε n m‖ =
        ‖A.zOp k m‖ ^ 2 * (‖A.z k n‖ ^ 2 * ‖A.chiR k ε n m‖) := by rw [hzOn]; ring
      _ ≤ c₄ := mul_le_mul (hfix m hmf) hb (by positivity) (by positivity)
      _ ≤ c₁ + c₂ + c₃ + c₄ := by linarith
  have hv0 := A.z_ne_zero_of_ne k hmf
  have htm : ‖(A.z k m)⁻¹‖ ≤ R := (A.norm_inv_z_le_of_ne k hmf).trans hR
  have hzOm : ‖A.zOp k m‖ = ‖A.z k m‖ := by rw [A.zOp_of_ne k hmf]
  have hlhs : ‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2 * ‖A.chiR k ε n m‖ =
      ‖A.z k n ^ 2 * A.z k m ^ 2 * A.chiR k ε n m‖ := by
    rw [hzOn, hzOm, norm_mul, norm_mul, norm_pow, norm_pow]
  by_cases hmir : m = mirror k n
  · -- a mirror pair
    subst hmir
    have h := A.z_sq_mul_chiR_mirror k ε hnf hz0 hv0
    have hq := A.norm_z_mirror_div_le k hnf hM hR
    have hn' : mirror k (mirror k n) ≠ mirror k n := by rw [mirror_mirror]; exact Ne.symm hnf
    have hq' := A.norm_z_mirror_div_le k hn' hM hR
    rw [mirror_mirror] at hq'
    have hq2 : ‖(A.z k (mirror k n) / A.z k n) ^ 2‖ ≤ (1 + 2 * M * R) ^ 2 := by
      rw [norm_pow]
      exact pow_le_pow_left₀ (norm_nonneg _) hq 2
    have hq2' : ‖(A.z k n / A.z k (mirror k n)) ^ 2‖ ≤ (1 + 2 * M * R) ^ 2 := by
      rw [norm_pow]
      exact pow_le_pow_left₀ (norm_nonneg _) hq' 2
    rw [hlhs, h]
    exact nmul (nmul (hB n) hq2 hB0) (nadd (nmul hΛ hp hΛ0) (nmul (hB (mirror k n))
      (nadd (nmul htm (hC₃ _ hmf) hR0) (nmul hq2' (hC₂ _ hmf) (by positivity))) hB0))
      (by positivity) |>.trans (by linarith)
  · -- a generic pair
    have hzv := A.z_add_z_ne_zero_of_ne k hmir
    have hPz := A.Pph_z_ne_zero k n
    have hPv := A.Pph_z_ne_zero k m
    have h := A.z_sq_mul_Pph_mul_chiR k ε hnm hv0 hzv
    have key : A.z k n ^ 2 * A.z k m ^ 2 * A.chiR k ε n m = A.z k n ^ 4 / A.Pph k (A.z k n) *
        (A.z k m ^ 4 / A.Pph k (A.z k m) *
            ((A.z k n)⁻¹ ^ 2 * A.r3t k ε (A.z k m)⁻¹ + A.n2t k ε (A.z k m)⁻¹) -
          A.lam ε * A.ρ * (A.z k n)⁻¹ + A.lam ε * A.ρ * (A.z k n + A.z k m)⁻¹) := by
      have e : A.z k n ^ 2 * A.z k m ^ 2 * A.chiR k ε n m = A.z k n ^ 2 / A.Pph k (A.z k n) *
          (A.z k m ^ 2 * (A.Pph k (A.z k n) * A.chiR k ε n m)) := by
        field_simp
      rw [e, h]
      field_simp
    have hs : ‖(A.z k n + A.z k m)⁻¹‖ ≤ R := (A.norm_inv_z_add_z_le_of_ne k hmir).trans hs2
    have ht2 : ‖(A.z k n)⁻¹ ^ 2‖ ≤ R ^ 2 := by
      rw [norm_pow]
      exact pow_le_pow_left₀ (norm_nonneg _) htn 2
    rw [hlhs, key]
    exact nmul (hB n) (nadd (nsub (nmul (hB m) (nadd (nmul ht2 (hC₄ m hmf) (by positivity))
      (hC₂ m hmf)) hB0) (nmul hΛρ htn hΛ0)) (nmul hΛρ hs hΛ0)) hB0 |>.trans (by linarith)

/-- **Diagonal entries on `z'ₙ`, with constants**: `|z'ₙ|³|χ_r,nn + β/z'ₙ²|` is bounded by the
constants below, `Cf` bounding it at the fixed point. -/
theorem norm_chiR_diag_le_all_of {B R M Cd C₂ C₃ Cf : ℝ}
    (hR : A.β / Real.pi ≤ R) (hM : |A.μ| ≤ M) (hMπ : Real.pi / A.β ≤ M)
    (hB : ∀ j, ‖A.z k j ^ 4 / A.Pph k (A.z k j)‖ ≤ B)
    (hd : ∀ j, mirror k j ≠ j → ‖A.dgt k ε (A.z k j)⁻¹‖ ≤ Cd)
    (hC₂ : ∀ j, mirror k j ≠ j → ‖A.n2t k ε (A.z k j)⁻¹‖ ≤ C₂)
    (hC₃ : ∀ j, mirror k j ≠ j → ‖A.n0t k ε (A.z k j)⁻¹‖ ≤ C₃)
    (hf : ∀ j, mirror k j = j → ‖A.chiR k ε j j + A.β / A.zOp k j ^ 2‖ ≤ Cf)
    (hC₂0 : 0 ≤ C₂) (hC₃0 : 0 ≤ C₃) (hCd0 : 0 ≤ Cd) (hCf0 : 0 ≤ Cf) (n : ℤ) :
    ‖A.zOp k n‖ ^ 3 * ‖A.chiR k ε n n + A.β / A.zOp k n ^ 2‖ ≤
      B * (Cd + B * (R ^ 2 * C₃ + R * C₂)) + 8 * M ^ 3 * Cf := by
  have hβ := A.β_pos
  have hπ := Real.pi_pos
  have hR0 : 0 ≤ R := (div_nonneg hβ.le hπ.le).trans hR
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hM
  have hB0 : 0 ≤ B := (norm_nonneg _).trans (hB 0)
  have h₁ : 0 ≤ B * (Cd + B * (R ^ 2 * C₃ + R * C₂)) := by positivity
  have h₂ : 0 ≤ 8 * M ^ 3 * Cf := by positivity
  by_cases hnf : mirror k n = n
  · have h := A.norm_zOp_fixed_le k hnf
    have h3 : ‖A.zOp k n‖ ^ 3 ≤ 8 * M ^ 3 := by
      have := pow_le_pow_left₀ (norm_nonneg _) (h.trans (show |A.μ| + Real.pi / A.β ≤ 2 * M by linarith)) 3
      linarith [show (2 * M) ^ 3 = 8 * M ^ 3 by ring]
    calc _ ≤ 8 * M ^ 3 * Cf := mul_le_mul h3 (hf n hnf) (norm_nonneg _) (by positivity)
      _ ≤ _ := by linarith
  · have hz0 := A.z_ne_zero_of_ne k hnf
    have ht : ‖(A.z k n)⁻¹‖ ≤ R := (A.norm_inv_z_le_of_ne k hnf).trans hR
    have ht2 : ‖(A.z k n)⁻¹ ^ 2‖ ≤ R ^ 2 := by
      rw [norm_pow]
      exact pow_le_pow_left₀ (norm_nonneg _) ht 2
    rw [A.zOp_of_ne k hnf, ← norm_pow, ← norm_mul, A.z_cube_mul_chiR_diag k ε hz0]
    exact (nmul (hB n) (nadd (hd n hnf) (nmul (hB n) (nadd (nmul ht2 (hC₃ n hnf) (by positivity))
      (nmul ht (hC₂ n hnf) hR0)) hB0)) hB0).trans (by linarith)

/-! ### Along a family of parameters -/

/-- `sinhc = dslope sinh 0` is continuous. -/
theorem continuous_dslope_sinh : Continuous (dslope Real.sinh 0) := by
  refine continuous_iff_continuousAt.mpr fun y ↦ ?_
  by_cases hy : y = 0
  · subst hy
    exact continuousAt_dslope_same.mpr (Real.differentiable_sinh 0)
  · exact (continuousAt_dslope_of_ne hy).mpr Real.continuous_sinh.continuousAt

section Family

variable {X : Type*} [TopologicalSpace X] {U β μ : X → ℝ} {hβ : ∀ x, 0 < β x} {x₀ : X}

section Cont

variable (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)

include hb hm in
theorem continuousAt_fam_x (j : ℤ) : ContinuousAt (fun x ↦ (fam U β μ hβ x).x j) x₀ := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [x, ν, fam]
  fun_prop (disch := assumption)

include hU in
theorem continuousAt_fam_a : ContinuousAt (fun x ↦ ((fam U β μ hβ x).a : ℂ)) x₀ := by
  simp only [fam, a]
  fun_prop

include hb in
theorem continuousAt_fam_β : ContinuousAt (fun x ↦ ((fam U β μ hβ x).β : ℂ)) x₀ := by
  simp only [fam]
  fun_prop

include hU hb hm

/-- The weights are continuous. -/
theorem continuousAt_fam_p :
    ContinuousAt (fun x ↦ ((fam U β μ hβ x).p₀ : ℂ)) x₀ ∧
      ContinuousAt (fun x ↦ ((fam U β μ hβ x).p₁ : ℂ)) x₀ ∧
      ContinuousAt (fun x ↦ ((fam U β μ hβ x).p₂ : ℂ)) x₀ := by
  refine ⟨?_, ?_, ?_⟩ <;>
  · simp only [fam, p₀, p₁, p₂, Z, boltz₀, boltz₁, boltz₂]
    fun_prop (disch := positivity)

theorem continuousAt_fam_ρ : ContinuousAt (fun x ↦ ((fam U β μ hβ x).ρ : ℂ)) x₀ := by
  obtain ⟨h₀, -, h₂⟩ := continuousAt_fam_p (hβ := hβ) hU hb hm
  simp only [ρ]
  push_cast
  fun_prop

/-- **The pair coefficient is continuous**, at `ℓ = 0` through `μ = 0` (`pairC_zero_eq`). -/
theorem continuousAt_fam_pairC (ℓ : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).pairC ℓ) x₀ := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  by_cases hℓ : ℓ = 0
  · subst hℓ
    have hs := continuous_dslope_sinh
    simp only [pairC_zero_eq, fam, Z, boltz₀, boltz₁, boltz₂]
    fun_prop (disch := positivity)
  · have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
    have e : ∀ x, (fam U β μ hβ x).pairC ℓ = ((fam U β μ hβ x).ρ : ℂ) /
        (2 * (fam U β μ hβ x).μ + (fam U β μ hβ x).ω ℓ * I) := fun x ↦ by
      rw [pairC, ite_eq_right (fun h ↦ hℓ h.2)]
    have h0 : 2 * ((fam U β μ hβ x₀).μ : ℂ) + (fam U β μ hβ x₀).ω ℓ * I ≠ 0 := by
      intro h
      have him := congrArg Complex.im h
      simp only [add_im, mul_im, ofReal_re, ofReal_im, I_re, I_im, mul_zero, mul_one,
        zero_add, add_zero, zero_im, re_ofNat, im_ofNat, zero_mul] at him
      rw [ω] at him
      have : (ℓ : ℝ) ≠ 0 := by exact_mod_cast hℓ
      have hβx := hβ x₀
      have : 2 * (ℓ : ℝ) * Real.pi / (fam U β μ hβ x₀).β ≠ 0 :=
        div_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero this) Real.pi_ne_zero)
          (fam U β μ hβ x₀).β_pos.ne'
      exact this him
    simp only [e]
    revert h0
    simp only [fam, ω]
    intro h0
    fun_prop (disch := assumption)

/-- **The entries of `χ_r` are continuous** in the parameters, for every `μ`. -/
theorem continuousAt_fam_chiR (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).chiR k ε n m) x₀ := by
  have hx := continuousAt_fam_x (U := U) (hβ := hβ) hb hm
  have hxn := hx n
  have hxm := hx m
  have hxnk := hx (n + k)
  have hxmk := hx (m + k)
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  obtain ⟨hp₀, hp₁, hp₂⟩ := continuousAt_fam_p (hβ := hβ) hU hb hm
  have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
  have hc := continuousAt_fam_pairC (hβ := hβ) hU hb hm (n + m + k + 1)
  have hqm : ContinuousAt (fun x ↦ ((fam U β μ hβ x).qm : ℂ)) x₀ := by
    simp only [qm]
    push_cast
    fun_prop
  have hqp : ContinuousAt (fun x ↦ ((fam U β μ hβ x).qp : ℂ)) x₀ := by
    simp only [qp]
    push_cast
    fun_prop
  simp only [chiR, chiUpUp, chiUpDn, piPair', Rreg', G, J, Params.x_add_w]
  fun_prop (disch := apply_rules [mul_ne_zero, Params.x_add_ofReal_ne_zero,
    Params.x_sub_ofReal_ne_zero, Params.x_sq_sub_ne_zero])

/-- The bubble `B(νₙ)` is continuous. -/
theorem continuousAt_fam_Bph (n : ℤ) : ContinuousAt (fun x ↦ (fam U β μ hβ x).Bph k n) x₀ := by
  have hx := continuousAt_fam_x (U := U) (hβ := hβ) hb hm
  have hxn := hx n
  have hxnk := hx (n + k)
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
  simp only [Bph, G, Params.x_add_w]
  fun_prop (disch := apply_rules [Params.x_sq_sub_ne_zero])

omit hU in
/-- The lattice points `z'ₙ` are continuous. -/
theorem continuousAt_fam_zOp (n : ℤ) : ContinuousAt (fun x ↦ (fam U β μ hβ x).zOp k n) x₀ := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  by_cases hn : mirror k n = n
  · simp only [zOp, ite_eq_left hn, fam]
    fun_prop (disch := assumption)
  · simp only [zOp, ite_eq_right hn]
    exact continuousAt_z (U := U) k hb hm n

end Cont

section Bounds

variable (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)

/-- `|z'ₙ|² = μ² + (π/β)²` at the fixed point. -/
theorem norm_zOp_fixed_sq (A : Params) {n : ℤ} (hn : mirror k n = n) :
    ‖A.zOp k n‖ ^ 2 = A.μ ^ 2 + (Real.pi / A.β) ^ 2 := by
  rw [zOp, ite_eq_left hn, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  simp
  ring

/-- `|zₙ|² = μ² + ((2n + 1 + k)π/β)²`. -/
theorem norm_z_sq_eq (A : Params) (n : ℤ) :
    ‖A.z k n‖ ^ 2 = A.μ ^ 2 + ((2 * n + 1 + k) * Real.pi / A.β) ^ 2 := by
  rw [A.z_eq, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]
  simp
  ring

include hU hb hm

omit hU hm in
/-- `1/z'ₙ → 0` as `|n| → ∞`, uniformly near `x₀`. -/
theorem eventually_norm_inv_zOp_lt :
    ∀ e > 0, ∃ N : ℕ, ∀ᶠ x in 𝓝 x₀, ∀ n : ℤ, (N : ℤ) ≤ |n| →
      ‖((fam U β μ hβ x).zOp k n)⁻¹‖ < e := by
  intro e he
  obtain ⟨N, hN⟩ := eventually_norm_inv_z_lt (fun x ↦ fam U β μ hβ x) k hb e he
  refine ⟨N + k.natAbs + 1, ?_⟩
  filter_upwards [hN] with x hx n hn
  have hk := Int.natCast_natAbs k
  push_cast at hn
  have hnf : mirror k n ≠ n := by
    unfold mirror
    rcases abs_cases n with ⟨h1, _⟩ | ⟨h1, _⟩ <;> rcases abs_cases k with ⟨h2, _⟩ | ⟨h2, _⟩ <;>
      omega
  rw [zOp_of_ne _ k hnf]
  exact hx n (by linarith [abs_nonneg k])

omit hU hb hm in
/-- `(c/β₀)² ≤ 2(c/β)²` for `β² ≤ 2β₀²`. -/
private theorem sq_div_le_two_mul {c b₀ b : ℝ} (hb₀ : 0 < b₀) (hb : 0 < b)
    (h : b ^ 2 ≤ 2 * b₀ ^ 2) : (c / b₀) ^ 2 ≤ 2 * (c / b) ^ 2 := by
  rw [div_pow, div_pow, div_le_iff₀ (by positivity)]
  field_simp
  nlinarith [mul_le_mul_of_nonneg_left h (sq_nonneg c)]

omit hU hb hm in
/-- A function continuous at `x₀` is bounded near `x₀`. -/
private theorem eventually_norm_le_of_continuousAt {f : X → ℂ} (hf : ContinuousAt f x₀) :
    ∀ᶠ x in 𝓝 x₀, ‖f x‖ ≤ ‖f x₀‖ + 1 :=
  (hf.norm.eventually (gt_mem_nhds (lt_add_one _))).mono fun _ h ↦ h.le

/-- The factors `n₂`, `n₀`, `r₃`, `dgt` at `t = 1/z'ₙ` are bounded near `x₀`, uniformly in `n`. -/
theorem exists_eventually_factors : ∃ C₂ C₃ C₄ Cd : ℝ, 0 ≤ C₂ ∧ 0 ≤ C₃ ∧ 0 ≤ C₄ ∧ 0 ≤ Cd ∧
    ∀ᶠ x in 𝓝 x₀, ∀ n, ‖(fam U β μ hβ x).n2t k ε ((fam U β μ hβ x).zOp k n)⁻¹‖ ≤ C₂ ∧
      ‖(fam U β μ hβ x).n0t k ε ((fam U β μ hβ x).zOp k n)⁻¹‖ ≤ C₃ ∧
      ‖(fam U β μ hβ x).r3t k ε ((fam U β μ hβ x).zOp k n)⁻¹‖ ≤ C₄ ∧
      ‖(fam U β μ hβ x).dgt k ε ((fam U β μ hβ x).zOp k n)⁻¹‖ ≤ Cd := by
  have hz := continuousAt_fam_zOp (U := U) (hβ := hβ) (k := k) hb hm
  have hcomp : ∀ {q : X → ℂ → ℂ}, (∀ t, ContinuousAt (fun p : X × ℂ ↦ q p.1 p.2) (x₀, t)) →
      ∀ n, ContinuousAt (fun x ↦ q x ((fam U β μ hβ x).zOp k n)⁻¹) x₀ := fun hq n ↦
    (hq ((fam U β μ hβ x₀).zOp k n)⁻¹).comp (f := fun x ↦ (x, ((fam U β μ hβ x).zOp k n)⁻¹))
      (continuousAt_id.prodMk ((hz n).inv₀ ((fam U β μ hβ x₀).zOp_ne_zero k n)))
  have hu := eventually_norm_inv_zOp_lt (U := U) (μ := μ) (hβ := hβ) (k := k) hb
  have h2J := continuousAt_n2t (hβ := hβ) k ε hU hb hm
  have h0J := continuousAt_n0t (hβ := hβ) k ε hU hb hm
  have h3J := continuousAt_r3t (hβ := hβ) k ε hU hb hm
  have hdJ := continuousAt_dgt (hβ := hβ) k ε hU hb hm
  obtain ⟨C₂, hC₂⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (fam U β μ hβ x).zOp k n) (h2J 0)
    (hcomp (q := fun x t ↦ (fam U β μ hβ x).n2t k ε t) h2J) hu
  obtain ⟨C₃, hC₃⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (fam U β μ hβ x).zOp k n) (h0J 0)
    (hcomp (q := fun x t ↦ (fam U β μ hβ x).n0t k ε t) h0J) hu
  obtain ⟨C₄, hC₄⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (fam U β μ hβ x).zOp k n) (h3J 0)
    (hcomp (q := fun x t ↦ (fam U β μ hβ x).r3t k ε t) h3J) hu
  obtain ⟨Cd, hCd⟩ := OpLattice.eventually_bdd_of_inv_z
    (zf := fun x n ↦ (fam U β μ hβ x).zOp k n) (hdJ 0)
    (hcomp (q := fun x t ↦ (fam U β μ hβ x).dgt k ε t) hdJ) hu
  have c0 : ∀ {C : ℝ} {f : X → ℤ → ℂ}, (∀ᶠ x in 𝓝 x₀, ∀ n, ‖f x n‖ ≤ C) → 0 ≤ C :=
    fun h ↦ (norm_nonneg _).trans (h.self_of_nhds 0)
  refine ⟨C₂, C₃, C₄, Cd, c0 hC₂, c0 hC₃, c0 hC₄, c0 hCd, ?_⟩
  filter_upwards [hC₂, hC₃, hC₄, hCd] with x e₂ e₃ e₄ ed n
  exact ⟨e₂ n, e₃ n, e₄ n, ed n⟩

/-- At the mirror fixed point, `N₀(zₙ)`, `N₂(zₙ)` and `χ_r,nn + β/z'ₙ²` are bounded near `x₀`. -/
theorem exists_eventually_fixed : ∃ CN Cf : ℝ, 0 ≤ CN ∧ 0 ≤ Cf ∧ ∀ᶠ x in 𝓝 x₀, ∀ j,
    mirror k j = j → ‖(fam U β μ hβ x).N₀ k ε ((fam U β μ hβ x).z k j)‖ ≤ CN ∧
      ‖(fam U β μ hβ x).N₂ k ε ((fam U β μ hβ x).z k j)‖ ≤ CN ∧
      ‖(fam U β μ hβ x).chiR k ε j j + ((fam U β μ hβ x).β : ℂ) /
        (fam U β μ hβ x).zOp k j ^ 2‖ ≤ Cf := by
  have hβ₀ := hβ x₀
  obtain ⟨jf, hjfd⟩ : ∃ j : ℤ, j = (-k - 1) / 2 := ⟨_, rfl⟩
  have hjf : ∀ j, mirror k j = j → j = jf := fun j hj ↦ by
    rw [hjfd]
    unfold mirror at hj
    omega
  have hz := continuousAt_fam_zOp (U := U) (hβ := hβ) (k := k) hb hm
  have hχ := continuousAt_fam_chiR (hβ := hβ) (k := k) (ε := ε) hU hb hm
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  obtain ⟨hp₀, hp₁, hp₂⟩ := continuousAt_fam_p (hβ := hβ) hU hb hm
  have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
  have hw : ContinuousAt (fun x ↦ (fam U β μ hβ x).w k) x₀ := by
    have hb' : β x₀ ≠ 0 := hβ₀.ne'
    simp only [w, ω, fam]
    fun_prop (disch := assumption)
  have hzf := continuousAt_z (U := U) (hβ := hβ) k hb hm jf
  have hN₀c : ContinuousAt (fun x ↦ (fam U β μ hβ x).N₀ k ε ((fam U β μ hβ x).z k jf)) x₀ := by
    simp only [N₀, N₂, Pph, Qp, Qm, q₀, ζ]
    fun_prop
  have hN₂c : ContinuousAt (fun x ↦ (fam U β μ hβ x).N₂ k ε ((fam U β μ hβ x).z k jf)) x₀ := by
    simp only [N₂, q₀, ζ]
    fun_prop
  have hfc : ContinuousAt (fun x ↦ (fam U β μ hβ x).chiR k ε jf jf +
      ((fam U β μ hβ x).β : ℂ) / (fam U β μ hβ x).zOp k jf ^ 2) x₀ := by
    have hz0 := (fam U β μ hβ x₀).zOp_ne_zero k jf
    have hχf := hχ jf jf
    have hzj := hz jf
    fun_prop (disch := exact pow_ne_zero _ hz0)
  refine ⟨‖(fam U β μ hβ x₀).N₀ k ε ((fam U β μ hβ x₀).z k jf)‖ +
    ‖(fam U β μ hβ x₀).N₂ k ε ((fam U β μ hβ x₀).z k jf)‖ + 1,
    ‖(fam U β μ hβ x₀).chiR k ε jf jf + ((fam U β μ hβ x₀).β : ℂ) /
      (fam U β μ hβ x₀).zOp k jf ^ 2‖ + 1, by positivity, by positivity, ?_⟩
  filter_upwards [eventually_norm_le_of_continuousAt hN₀c, eventually_norm_le_of_continuousAt hN₂c,
    eventually_norm_le_of_continuousAt hfc] with x e₀ e₂ ef j hj
  rw [hjf j hj]
  have h₀ := norm_nonneg ((fam U β μ hβ x₀).N₀ k ε ((fam U β μ hβ x₀).z k jf))
  have h₂ := norm_nonneg ((fam U β μ hβ x₀).N₂ k ε ((fam U β μ hβ x₀).z k jf))
  exact ⟨e₀.trans (by linarith), e₂.trans (by linarith), ef⟩

/-- The parameters and the pair coefficient are bounded near `x₀`. -/
theorem exists_eventually_params : ∃ R M B Λ Cp : ℝ, ∀ᶠ x in 𝓝 x₀,
    (fam U β μ hβ x).β / Real.pi ≤ R ∧ |(fam U β μ hβ x).μ| ≤ M ∧
      Real.pi / (fam U β μ hβ x).β ≤ M ∧
      (∀ j, ‖(fam U β μ hβ x).z k j ^ 4 / (fam U β μ hβ x).Pph k ((fam U β μ hβ x).z k j)‖ ≤ B) ∧
      ‖(fam U β μ hβ x).lam ε‖ ≤ Λ ∧ ‖(fam U β μ hβ x).lam ε * (fam U β μ hβ x).ρ‖ ≤ Λ ∧
      ‖(fam U β μ hβ x).pairC 0‖ ≤ Cp := by
  have hβ₀ := hβ x₀
  have hπ := Real.pi_pos
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
  have hlam : ContinuousAt (fun x ↦ (fam U β μ hβ x).lam ε) x₀ := by
    simp only [lam]
    fun_prop
  have hlr : ContinuousAt (fun x ↦ (fam U β μ hβ x).lam ε * (fam U β μ hβ x).ρ) x₀ := by
    simp only [lam]
    fun_prop
  have hpc := continuousAt_fam_pairC (hβ := hβ) hU hb hm 0
  have hL : ∀ᶠ x in 𝓝 x₀, 1 + |μ x| * β x / Real.pi + |(k : ℝ)| <
      1 + |μ x₀| * β x₀ / Real.pi + |(k : ℝ)| + 1 := by
    have : ContinuousAt (fun x ↦ 1 + |μ x| * β x / Real.pi + |(k : ℝ)|) x₀ := by fun_prop
    exact this.eventually (gt_mem_nhds (lt_add_one _))
  have hμe : ∀ᶠ x in 𝓝 x₀, |μ x| < |μ x₀| + 1 :=
    (continuous_abs.continuousAt.comp hm).eventually (gt_mem_nhds (lt_add_one _))
  have hβ1 : ∀ᶠ x in 𝓝 x₀, β x₀ / 2 < β x := hb.eventually (lt_mem_nhds (half_lt_self hβ₀))
  have hβ2 : ∀ᶠ x in 𝓝 x₀, β x < 2 * β x₀ := hb.eventually (gt_mem_nhds (by linarith))
  refine ⟨2 * β x₀ / Real.pi, |μ x₀| + 1 + 2 * Real.pi / β x₀,
    (1 + |μ x₀| * β x₀ / Real.pi + |(k : ℝ)| + 1) ^ 4,
    ‖(fam U β μ hβ x₀).lam ε‖ + ‖(fam U β μ hβ x₀).lam ε * (fam U β μ hβ x₀).ρ‖ + 1,
    ‖(fam U β μ hβ x₀).pairC 0‖ + 1, ?_⟩
  filter_upwards [eventually_norm_le_of_continuousAt hlam, eventually_norm_le_of_continuousAt hlr,
    eventually_norm_le_of_continuousAt hpc, hL, hμe, hβ1, hβ2] with x el elr ep eL eμ eβ1 eβ2
  have hβx : 0 < β x := hβ x
  have h2π : 0 < 2 * Real.pi / β x₀ := by positivity
  have hπβ : Real.pi / β x ≤ 2 * Real.pi / β x₀ := by
    rw [div_le_div_iff₀ hβx hβ₀]
    nlinarith
  have hl₀ := norm_nonneg ((fam U β μ hβ x₀).lam ε)
  have hlr₀ := norm_nonneg ((fam U β μ hβ x₀).lam ε * (fam U β μ hβ x₀).ρ)
  refine ⟨div_le_div_of_nonneg_right eβ2.le hπ.le, ?_, ?_, fun j ↦ ?_, by linarith, by linarith,
    ep⟩
  · change |μ x| ≤ _
    linarith
  · change Real.pi / β x ≤ _
    linarith [abs_nonneg (μ x₀)]
  · refine ((fam U β μ hβ x).norm_z_pow_div_Pph_le k j).trans (pow_le_pow_left₀ ?_ eL.le 4)
    have := mul_nonneg (abs_nonneg (μ x)) hβx.le
    have := div_nonneg this hπ.le
    have := abs_nonneg (k : ℝ)
    change 0 ≤ 1 + |μ x| * β x / Real.pi + |(k : ℝ)|
    linarith

/-- **The bounds of item 6(a) on `z'ₙ`, uniform near `x₀`**. -/
theorem exists_eventually_chiR_bounds : ∃ c₁ c₂ : ℝ, ∀ᶠ x in 𝓝 x₀,
    (∀ n m, n ≠ m → ‖(fam U β μ hβ x).chiR k ε n m‖ ≤
      c₁ / (‖(fam U β μ hβ x).zOp k n‖ ^ 2 * ‖(fam U β μ hβ x).zOp k m‖ ^ 2)) ∧
    ∀ n, ‖(fam U β μ hβ x).chiR k ε n n + ((fam U β μ hβ x).β : ℂ) /
      (fam U β μ hβ x).zOp k n ^ 2‖ ≤ c₂ / ‖(fam U β μ hβ x).zOp k n‖ ^ 3 := by
  obtain ⟨C₂, C₃, C₄, Cd, hC₂0, hC₃0, hC₄0, hCd0, hF⟩ :=
    exists_eventually_factors (hβ := hβ) (k := k) (ε := ε) hU hb hm
  obtain ⟨CN, Cf, hCN0, hCf0, hX⟩ := exists_eventually_fixed (hβ := hβ) (k := k) (ε := ε) hU hb hm
  obtain ⟨R, M, B, Λ, Cp, hP⟩ := exists_eventually_params (hβ := hβ) (k := k) (ε := ε) hU hb hm
  refine ⟨B * (1 + 2 * M * R) ^ 2 * (Λ * Cp + B * (R * C₃ + (1 + 2 * M * R) ^ 2 * C₂)) +
      B * (B * (R ^ 2 * C₄ + C₂) + Λ * R + Λ * R) +
      4 * M ^ 2 * (R ^ 4 * (B * (C₄ + M ^ 2 * C₂) + Λ * M + Λ * M ^ 2 * R)) +
      4 * M ^ 2 * (B * (Λ * R ^ 2 * R + (R ^ 2 * CN + CN) * R ^ 4)),
    B * (Cd + B * (R ^ 2 * C₃ + R * C₂)) + 8 * M ^ 3 * Cf, ?_⟩
  filter_upwards [hF, hX, hP] with x eF eX ⟨hR, hM, hMπ, hB, hΛ, hΛρ, hp⟩
  set A := fam U β μ hβ x
  have off : ∀ j, mirror k j ≠ j → ‖A.n2t k ε (A.z k j)⁻¹‖ ≤ C₂ ∧
      ‖A.n0t k ε (A.z k j)⁻¹‖ ≤ C₃ ∧ ‖A.r3t k ε (A.z k j)⁻¹‖ ≤ C₄ ∧
      ‖A.dgt k ε (A.z k j)⁻¹‖ ≤ Cd := fun j hj ↦ by
    have := eF j
    rwa [zOp_of_ne _ k hj] at this
  refine ⟨fun n m hnm ↦ ?_, fun n ↦ ?_⟩
  · have h := A.norm_chiR_le_all_of k ε hR hM hMπ hB hΛ hΛρ hp (fun j hj ↦ (off j hj).1)
      (fun j hj ↦ (off j hj).2.1) (fun j hj ↦ (off j hj).2.2.1) (fun j hj ↦ (eX j hj).1)
      (fun j hj ↦ (eX j hj).2.1) hC₂0 hC₃0 hC₄0 hCN0 n m hnm
    have h1 := norm_pos_iff.mpr (A.zOp_ne_zero k n)
    have h2 := norm_pos_iff.mpr (A.zOp_ne_zero k m)
    rw [le_div_iff₀ (by positivity)]
    linarith
  · have h := A.norm_chiR_diag_le_all_of k ε hR hM hMπ hB (fun j hj ↦ (off j hj).2.2.2)
      (fun j hj ↦ (off j hj).1) (fun j hj ↦ (off j hj).2.1) (fun j hj ↦ (eX j hj).2.2)
      hC₂0 hC₃0 hCd0 hCf0 n
    have h1 := norm_pos_iff.mpr (A.zOp_ne_zero k n)
    rw [le_div_iff₀ (by positivity)]
    linarith

omit hU in
/-- `|z'ₙ(x)|² ≥ |z'ₙ(x₀)|²/2` near `x₀`, for every `μ(x₀)`. -/
theorem eventually_norm_zOp_sq_ge : ∀ᶠ x in 𝓝 x₀, ∀ n,
    1 / 2 * ‖(fam U β μ hβ x₀).zOp k n‖ ^ 2 ≤ ‖(fam U β μ hβ x).zOp k n‖ ^ 2 := by
  have hβ₀ := hβ x₀
  have hb2 : ContinuousAt (fun x ↦ β x ^ 2) x₀ := hb.pow 2
  have hβ3 : ∀ᶠ x in 𝓝 x₀, β x ^ 2 < 2 * β x₀ ^ 2 :=
    hb2.eventually (gt_mem_nhds (show β x₀ ^ 2 < 2 * β x₀ ^ 2 by nlinarith))
  have hμ2 : ∀ᶠ x in 𝓝 x₀, 1 / 2 * μ x₀ ^ 2 ≤ μ x ^ 2 := by
    by_cases hμ₀ : μ x₀ = 0
    · exact Filter.Eventually.of_forall fun x ↦ by rw [hμ₀]; nlinarith [sq_nonneg (μ x)]
    · have hm2 : ContinuousAt (fun x ↦ μ x ^ 2) x₀ := hm.pow 2
      have : 1 / 2 * μ x₀ ^ 2 < μ x₀ ^ 2 := by
        have := pow_pos (abs_pos.mpr hμ₀) 2
        rw [sq_abs] at this
        linarith
      exact (hm2.eventually (lt_mem_nhds this)).mono fun _ h ↦ h.le
  filter_upwards [hμ2, hβ3] with x eμ eβ n
  have hβx := hβ x
  by_cases hn : mirror k n = n
  · rw [norm_zOp_fixed_sq k _ hn, norm_zOp_fixed_sq k _ hn]
    have := sq_div_le_two_mul (c := Real.pi) hβ₀ hβx eβ.le
    change 1 / 2 * (μ x₀ ^ 2 + (Real.pi / β x₀) ^ 2) ≤ μ x ^ 2 + (Real.pi / β x) ^ 2
    linarith
  · rw [zOp_of_ne _ k hn, zOp_of_ne _ k hn, norm_z_sq_eq k, norm_z_sq_eq k]
    have := sq_div_le_two_mul (c := (2 * n + 1 + k) * Real.pi) hβ₀ hβx eβ.le
    change 1 / 2 * (μ x₀ ^ 2 + ((2 * n + 1 + k) * Real.pi / β x₀) ^ 2) ≤
      μ x ^ 2 + ((2 * n + 1 + k) * Real.pi / β x) ^ 2
    linarith

/-- **The hypotheses of `ChiContinuity.lean`** for the density and magnetic channels on `z'ₙ`, at
every `x₀`, `μ(x₀) = 0` included. -/
theorem contHyp_R :
    OpLattice.ContHyp (fun x ↦ (fam U β μ hβ x).opLatticeAll k)
      (fun x ↦ (fam U β μ hβ x).chiR k ε) (fun x ↦ (fam U β μ hβ x).Bph k) x₀ := by
  have hβ₀ := hβ x₀
  have hz := continuousAt_fam_zOp (U := U) (hβ := hβ) (k := k) hb hm
  have hχ := continuousAt_fam_chiR (hβ := hβ) (k := k) (ε := ε) hU hb hm
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  obtain ⟨c₁, c₂, hc⟩ := exists_eventually_chiR_bounds (hβ := hβ) (k := k) (ε := ε) hU hb hm
  refine ⟨Filter.Eventually.of_forall fun x ↦ (fam U β μ hβ x).chiDecay_chiR_all k ε,
    Filter.Eventually.of_forall fun x ↦ (fam U β μ hβ x).bubbleDecay_Bph k, hb, hz,
    fun n ↦ continuousAt_fam_Bph (hβ := hβ) k hU hb hm n, fun n m ↦ ?_,
    OpLattice.dom_of_uniform (c := 1 / 2) (c₁ := c₁) (c₂ := c₂) (by norm_num)
      (hc.mono fun x hx ↦ hx.1) (hc.mono fun x hx ↦ hx.2) hb
      (eventually_norm_zOp_sq_ge (U := U) (hβ := hβ) (k := k) hb hm)⟩
  have hzn := hz n
  have hzm := hz m
  have hχnm := hχ n m
  have hK : ContinuousAt (fun x ↦ -((fam U β μ hβ x).zOp k n * (fam U β μ hβ x).zOp k m /
      ((fam U β μ hβ x).β : ℂ)) * (fam U β μ hβ x).chiR k ε n m - kron n m) x₀ := by
    fun_prop (disch := exact ofReal_ne_zero.mpr hβ₀.ne')
  exact hK

/-- **`Γ_r` is continuous in the parameters, for every `μ`**: along a family `(U, β, μ)(x)`
continuous at `x₀`, if `1 + 𝒦` is invertible at `x₀`, then `Γ_r,nm` is continuous at `x₀`;
`μ(x₀) = 0` included. -/
theorem continuousAt_gammaOpR (hu : IsUnit ((fam U β μ hβ x₀).onePlusKR k ε)) (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).gammaOpR k ε n m) x₀ := by
  have hH := contHyp_R (hβ := hβ) (k := k) (ε := ε) hU hb hm
  refine (OpLattice.continuousAt_gammaOpFam hH ((fam U β μ hβ x₀).chiDecay_chiR_all k ε) hu
    n m).congr (Filter.Eventually.of_forall fun x ↦ ?_)
  exact OpLattice.gammaOpFam_eq (fun x ↦ (fam U β μ hβ x).opLatticeAll k)
    (fun x ↦ (fam U β μ hβ x).chiR k ε) (fun x ↦ (fam U β μ hβ x).Bph k)
    ((fam U β μ hβ x).chiDecay_chiR_all k ε) ((fam U β μ hβ x).bubbleDecay_Bph k) n m

/-- **Through half filling**: under the assumptions of Corollary "half filling" at `x₀`
(`μ(x₀) = 0`), `Γ_r,nm` is continuous at `x₀`; there it is the closed form `gammaHF`. -/
theorem continuousAt_gammaOpR_of_HFHyp {u : ℂ} (h : (fam U β μ hβ x₀).HFHyp k ε u) (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).gammaOpR k ε n m) x₀ :=
  continuousAt_gammaOpR k ε hU hb hm h.isUnit_onePlusKR n m

/-- **On the domain of Theorem "density and magnetic"** at `x₀`, `Γ_r,nm` is continuous at `x₀`. -/
theorem continuousAt_gammaOpR_of_DMGeneric {u₁ u₂ : ℂ}
    (h : (fam U β μ hβ x₀).DMGeneric k ε u₁ u₂) (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).gammaOpR k ε n m) x₀ :=
  continuousAt_gammaOpR k ε hU hb hm ((fam U β μ hβ x₀).isUnit_onePlusKR_of_isRightInverse k ε
    h.chiR_isRightInverse fun m ↦ by
      obtain ⟨C, hC⟩ :=
        ((fam U β μ hβ x₀).tendsto_invR k ε u₁ u₂ m).norm.bddAbove_range_of_cofinite
      exact ⟨C, fun n ↦ hC ⟨n, rfl⟩⟩) n m

end Bounds

end Family

end Params

end HubbardAtom
