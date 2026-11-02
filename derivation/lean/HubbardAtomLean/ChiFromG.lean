import HubbardAtomLean.WardIdentities

/-!
# The density and magnetic susceptibilities from `G`, and their commutation with `X - (εU/β)𝟙𝟙ᵀ`

For `U ≠ 0`, `μ ≠ 0`, with `X = diag(zₙ²)` and `A = X - (εU/β)𝟙𝟙ᵀ` (`Params.Amat`), as infinite
matrices with products as absolutely convergent sums over `ℤ`:

* `Params.hasSum_Amat_mul_chiR`: `(Aχ_r)_nm = zₙ²χ_r,nm + εUf_r(zₘ)`, the column sums of `χ_r`
  being its row sums (`chiR_symm`, Ward identity `hasSum_chiR`);
* `Params.hasSum_chiR_mul_Amat`: `(χ_rA)_nm = χ_r,nm zₘ² + εUf_r(zₙ)`;
* `Params.commute_Amat_chiR`: `[A, χ_r] = 0`, both products converging absolutely to the same
  entries, by the divided differences (`chiR_divided_difference`).

And `χ_r` from `G`, as one statement (`U ≠ 0`, `μ ≠ 0`):

* `Params.chiR_eq_fR`: off the diagonal `χ_r,nm = εU[f_r(zₙ) - f_r(zₘ)]/(zₙ² - zₘ²)`
  (`DividedDifferences.lean`), on it `χ_r,nn = -βf_r(zₙ) - ∑_{l ≠ n} χ_r,nl` (`WardIdentities.lean`);
* `Params.chiR_eq_G`: for `k ≠ 0`, the same with `f_r(zⱼ) = [G(xⱼ) - G(xⱼ + w)]/w`. At `k = 0`,
  `f_r` is the static response of `G` (`StaticResponse.lean`).
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `A = X - (εU/β)𝟙𝟙ᵀ` with `X = diag(zₙ²)`. -/
noncomputable def Amat (n m : ℤ) : ℂ := A.z k n ^ 2 * kron n m - ε * A.U / A.β

/-- `(Aχ_r)_nm = zₙ²χ_r,nm + εUf_r(zₘ)`. -/
theorem hasSum_Amat_mul_chiR (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    (n m : ℤ) :
    HasSum (fun l ↦ A.Amat k ε n l * A.chiR k ε l m)
      (A.z k n ^ 2 * A.chiR k ε n m + ε * A.U * A.fR k ε (A.z k m)) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hcol : HasSum (fun l ↦ A.chiR k ε l m) (-A.β * A.fR k ε (A.z k m)) := by
    simpa only [A.chiR_symm k ε hμ m] using A.hasSum_chiR k ε hε ha hμ m
  have h₁ := hasSum_single (f := fun l ↦ A.z k n ^ 2 * kron n l * A.chiR k ε l m) n
    (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_mul])
  convert h₁.sub (hcol.mul_left (ε * A.U / A.β)) using 1
  · funext l
    rw [Amat]
    ring
  · rw [kron_self]
    field_simp
    ring

/-- `(χ_rA)_nm = χ_r,nm zₘ² + εUf_r(zₙ)`. -/
theorem hasSum_chiR_mul_Amat (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    (n m : ℤ) :
    HasSum (fun l ↦ A.chiR k ε n l * A.Amat k ε l m)
      (A.chiR k ε n m * A.z k m ^ 2 + ε * A.U * A.fR k ε (A.z k n)) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have h₁ := hasSum_single (f := fun l ↦ A.chiR k ε n l * (A.z k l ^ 2 * kron l m)) m
    (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero])
  convert h₁.sub ((A.hasSum_chiR k ε hε ha hμ n).mul_right (ε * A.U / A.β)) using 1
  · funext l
    rw [Amat]
    ring
  · rw [kron_self]
    field_simp
    ring

/-- **`[A, χ_r] = 0`**: for `U ≠ 0`, `μ ≠ 0`, the products `Aχ_r` and `χ_rA` converge absolutely
and have the same entries `zₙ²χ_r,nm + εUf_r(zₘ) = χ_r,nm zₘ² + εUf_r(zₙ)`. -/
theorem commute_Amat_chiR (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    (n m : ℤ) :
    HasSum (fun l ↦ A.Amat k ε n l * A.chiR k ε l m)
        (A.z k n ^ 2 * A.chiR k ε n m + ε * A.U * A.fR k ε (A.z k m)) ∧
      HasSum (fun l ↦ A.chiR k ε n l * A.Amat k ε l m)
        (A.z k n ^ 2 * A.chiR k ε n m + ε * A.U * A.fR k ε (A.z k m)) ∧
      (Summable fun l ↦ ‖A.Amat k ε n l * A.chiR k ε l m‖) ∧
      Summable fun l ↦ ‖A.chiR k ε n l * A.Amat k ε l m‖ := by
  have h₁ := A.hasSum_Amat_mul_chiR k ε hε ha hμ n m
  have h₂ := A.hasSum_chiR_mul_Amat k ε hε ha hμ n m
  have he : A.chiR k ε n m * A.z k m ^ 2 + ε * A.U * A.fR k ε (A.z k n) =
      A.z k n ^ 2 * A.chiR k ε n m + ε * A.U * A.fR k ε (A.z k m) := by
    by_cases hnm : n = m
    · subst hnm
      ring
    · linear_combination -A.chiR_divided_difference k ε hε hnm
  rw [he] at h₂
  exact ⟨h₁, h₂, summable_norm_iff.mpr h₁.summable, summable_norm_iff.mpr h₂.summable⟩

/-! ### `χ_r` from `G` -/

theorem z_sq_sub_z_sq_ne_zero (hμ : A.μ ≠ 0) {n m : ℤ} (hnm : n ≠ m) :
    A.z k n ^ 2 - A.z k m ^ 2 ≠ 0 := by
  rw [show A.z k n ^ 2 - A.z k m ^ 2 = (A.z k n - A.z k m) * (A.z k n + A.z k m) by ring]
  exact mul_ne_zero (sub_ne_zero.mpr fun h ↦ hnm (A.z_injective k h))
    (A.z_add_z_ne_zero k hμ n m)

/-- **`χ_r` from `f_r`**: for `U ≠ 0`, `μ ≠ 0`, off the diagonal
`χ_r,nm = εU[f_r(zₙ) - f_r(zₘ)]/(zₙ² - zₘ²)` (`chiR_divided_difference`, with `zₙ² ≠ zₘ²`), and on
it `χ_r,nn = -βf_r(zₙ) - ∑_{l ≠ n} χ_r,nl` (Ward identity `hasSum_chiR`). -/
theorem chiR_eq_fR (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) (n m : ℤ) :
    A.chiR k ε n m =
      if n = m then
        -A.β * A.fR k ε (A.z k n) - ∑' l, if l = n then 0 else
          ε * A.U * (A.fR k ε (A.z k n) - A.fR k ε (A.z k l)) / (A.z k n ^ 2 - A.z k l ^ 2)
      else ε * A.U * (A.fR k ε (A.z k n) - A.fR k ε (A.z k m)) / (A.z k n ^ 2 - A.z k m ^ 2) := by
  have hoff : ∀ l, l ≠ n → A.chiR k ε n l =
      ε * A.U * (A.fR k ε (A.z k n) - A.fR k ε (A.z k l)) / (A.z k n ^ 2 - A.z k l ^ 2) :=
    fun l hl ↦ by
      rw [eq_div_iff (A.z_sq_sub_z_sq_ne_zero k hμ (Ne.symm hl)), mul_comm]
      exact A.chiR_divided_difference k ε hε (Ne.symm hl)
  split_ifs with hnm
  · subst hnm
    have h := hasSum_ite_sub_hasSum (A.hasSum_chiR k ε hε ha hμ n) n
    rw [(h.congr_fun fun l ↦ ?_).tsum_eq]
    · ring
    · split_ifs with hl
      · rfl
      · exact (hoff l hl).symm
  · exact hoff m (Ne.symm hnm)

/-- **`χ_r` from `G`**, `k ≠ 0`: `chiR_eq_fR` with `f_r(zⱼ) = [G(xⱼ) - G(xⱼ + w)]/w`
(`fR_eq_G`), so the entries of `χ_r` are fixed by `G` alone. -/
theorem chiR_eq_G (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) (hk : k ≠ 0)
    (n m : ℤ) :
    A.chiR k ε n m =
      if n = m then
        -A.β * ((A.G (A.x n) - A.G (A.x n + A.w k)) / A.w k) - ∑' l, if l = n then 0 else
          ε * A.U * ((A.G (A.x n) - A.G (A.x n + A.w k)) / A.w k -
            (A.G (A.x l) - A.G (A.x l + A.w k)) / A.w k) / (A.z k n ^ 2 - A.z k l ^ 2)
      else ε * A.U * ((A.G (A.x n) - A.G (A.x n + A.w k)) / A.w k -
        (A.G (A.x m) - A.G (A.x m + A.w k)) / A.w k) / (A.z k n ^ 2 - A.z k m ^ 2) := by
  simpa only [A.fR_eq_G k ε hk] using A.chiR_eq_fR k ε hε ha hμ n m

end Params

end HubbardAtom
