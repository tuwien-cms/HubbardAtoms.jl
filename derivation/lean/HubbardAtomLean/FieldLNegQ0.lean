import HubbardAtomLean.FieldLAxisGeneric

/-!
# `E₈` at `s = -q₀`: no simple root has `u² = -q₀`

`E₈(v)` is a monic quartic in `s = v²` (`E8L_eq_sum`). At `s = -q₀` it is a perfect square and
its `s`-derivative carries the same factor:

* `FParams.cNegQ0`: `c_k = 4a²(r↑r↓ - 4d_π) + ω_k²r↑r↓`, `d_π = π_P - π_S`; equivalently
  `c_k = 4(q₀r↑r↓ - λd_π)` (`cNegQ0_eq`), as `q₀ = a² + ω_k²/4` (`q₀L_eq_omega`);
* `FParams.E8L_neg_q0`: **`E₈(v) = a⁴c_k²` where `v² = -q₀`**;
* `FParams.sum_E8c_deriv_neg_q0`: **`E₈'(-q₀) = -8a⁴r↑r↓c_k`**, the derivative in `s`;
* `FParams.sum_E8c_deriv_eq_zero_of_neg_q0`: so if `E₈(v) = 0` at `v² = -q₀`, then
  `(a⁴c_k)² = a⁴E₈(v) = 0` and `E₈'(v²) = 0`: `s = -q₀` is a root of `E₈` only as a multiple root;
* `FParams.sq_add_q₀L_ne_of_sep`: **at four roots `u_j` of `E₈` with distinct squares,
  `u_j² + q₀ ≠ 0`**, as each `u_j²` is a simple root (`sum_E8c_deriv`). Neither `U ≠ 0` nor any
  relation between the weights is assumed: the identities hold for any `a`, `q₀`, `r↑`, `r↓`,
  `π_S`, `π_P`, and at `a = 0`, `E₈ = (s + q₀)⁴` has no simple root;
* `FParams.LAxisGeneric.mk'`, `FParams.lAxisGeneric_iff`: so the conjunct `u_j² + q₀ ≠ 0` in the
  second alternative of `LAxisGeneric.K_ne` is redundant: `LAxisGeneric` holds iff `U ≠ 0`, the
  roots, `sep`, `off`, and `K_t(u_j) ≠ 0` or `K_{t,-t}(u_j) ≠ 0` at each non-atomic root;
* `FParams.KL_opp_eq_zero_of_sep`: the quotient identity's consequence `KL_opp_eq_zero`
  (`K_t = 0` at a root forces `K_{t,-t} = 0`, off `μ_t = 0`) without `u² + q₀ ≠ 0`, at four roots
  with distinct squares.

What this simplifies: the note's remark that at `s_j = -q₀` the whole mode vanishes (Lean
appendix, with `KL_opp_eq_zero`) concerns a case that does not occur at simple roots, which are
the roots of `LGeneric` and `LAxisGeneric`. Not in the note: `c_k`, the two identities and their
consequences.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The identities at `s = -q₀` -/

/-- `q₀ = a² + ω_k²/4`, as `w = iω_k`. -/
theorem q₀L_eq_omega : F.q₀L k = F.a ^ 2 + (F.ω k : ℂ) ^ 2 / 4 := by
  rw [q₀L, wL, mul_pow, I_sq]
  ring

/-- `c_k = 4a²(r↑r↓ - 4d_π) + ω_k²r↑r↓` with `d_π = π_P - π_S`: `E₈(-q₀) = a⁴c_k²`
(`E8L_neg_q0`). Not in the note. -/
noncomputable def cNegQ0 : ℂ :=
  4 * F.a ^ 2 * (F.rUp * F.rDn - 4 * (F.piP - F.piS)) + (F.ω k : ℂ) ^ 2 * F.rUp * F.rDn

/-- `c_k = 4(q₀r↑r↓ - λd_π)`. -/
theorem cNegQ0_eq : F.cNegQ0 k = 4 * (F.q₀L k * F.rUp * F.rDn - F.lamL * (F.piP - F.piS)) := by
  rw [cNegQ0, q₀L_eq_omega, lamL]
  ring

/-- **`E₈` at `s = -q₀`**: `E₈(v) = a⁴c_k²` where `v² = -q₀`. Not in the note. -/
theorem E8L_neg_q0 {v : ℂ} (hv : v ^ 2 = -F.q₀L k) : F.E8L k v = F.a ^ 4 * F.cNegQ0 k ^ 2 := by
  rw [E8L_eq_quartic, hv, cNegQ0_eq]
  simp only [E8b, lamL]
  ring

/-- **The `s`-derivative of `E₈` at `s = -q₀`**: `E₈'(-q₀) = -8a⁴r↑r↓c_k`. Not in the note. -/
theorem sum_E8c_deriv_neg_q0 :
    ∑ i : Fin 5, (i : ℕ) * F.E8c k i * (-F.q₀L k) ^ ((i : ℕ) - 1) =
      -8 * F.a ^ 4 * F.rUp * F.rDn * F.cNegQ0 k := by
  rw [cNegQ0_eq]
  simp [Fin.sum_univ_five, E8c]
  simp only [E8b, lamL]
  ring

/-- **`s = -q₀` is a root of `E₈` only as a multiple root**: if `E₈(v) = 0` at `v² = -q₀`, the
`s`-derivative vanishes there, as `(a⁴c_k)² = a⁴E₈(v) = 0`. Not in the note. -/
theorem sum_E8c_deriv_eq_zero_of_neg_q0 {v : ℂ} (hv : v ^ 2 = -F.q₀L k) (hE : F.E8L k v = 0) :
    ∑ i : Fin 5, (i : ℕ) * F.E8c k i * (v ^ 2) ^ ((i : ℕ) - 1) = 0 := by
  have h : (F.a ^ 4 * F.cNegQ0 k) ^ 2 = (F.a : ℂ) ^ 4 * F.E8L k v := by
    rw [F.E8L_neg_q0 k hv]
    ring
  rw [hE, mul_zero] at h
  have hc : F.a ^ 4 * F.cNegQ0 k = 0 := (pow_eq_zero_iff two_ne_zero).1 h
  rw [hv, sum_E8c_deriv_neg_q0]
  linear_combination (-8 * F.rUp * F.rDn : ℂ) * hc

/-! ### Simple roots are off `s = -q₀` -/

/-- **At four roots of `E₈` with distinct squares, `u_j² + q₀ ≠ 0`**: each `u_j²` is a simple root
(`sum_E8c_deriv`), and `-q₀` is a root only as a multiple one
(`sum_E8c_deriv_eq_zero_of_neg_q0`). Neither `U ≠ 0` nor a relation between the weights is
assumed. Not in the note. -/
theorem sq_add_q₀L_ne_of_sep {u : Fin 4 → ℂ} (root : ∀ j, F.E8L k (u j) = 0)
    (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) (j : Fin 4) : u j ^ 2 + F.q₀L k ≠ 0 := by
  intro hq
  have h := F.sum_E8c_deriv_eq_zero_of_neg_q0 k (eq_neg_of_add_eq_zero_left hq) (root j)
  rw [F.sum_E8c_deriv k root sep j] at h
  exact dprod_ne sep j h

/-- **`LAxisGeneric` without `u_j² + q₀ ≠ 0`**: the conjunct of the second alternative of `K_ne`
follows from `root` and `sep` (`sq_add_q₀L_ne_of_sep`). -/
theorem LAxisGeneric.mk' {u : Fin 4 → ℂ} (a_ne : (F.a : ℂ) ≠ 0)
    (root : ∀ j, F.E8L k (u j) = 0) (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0)
    (off : ∀ j, ¬F.Atomic k (u j) → ∀ σ l, (F.BL σ).z k l ≠ u j ∧ (F.BL σ).z k l ≠ -u j)
    (K_ne : ∀ t j, ¬F.Atomic k (u j) → F.KL k t t (u j) ≠ 0 ∨ F.KL k t (!t) (u j) ≠ 0) :
    F.LAxisGeneric k u :=
  ⟨a_ne, root, sep, off, fun t j hj ↦
    (K_ne t j hj).imp_right fun hK ↦ ⟨hK, F.sq_add_q₀L_ne_of_sep k root sep j⟩⟩

/-- **The conjunct `u_j² + q₀ ≠ 0` of `LAxisGeneric` is redundant**: `LAxisGeneric` holds iff
`U ≠ 0`, `root`, `sep`, `off`, and `K_t(u_j) ≠ 0` or `K_{t,-t}(u_j) ≠ 0` at each non-atomic root,
for each column spin `t`. -/
theorem lAxisGeneric_iff {u : Fin 4 → ℂ} :
    F.LAxisGeneric k u ↔ (F.a : ℂ) ≠ 0 ∧ (∀ j, F.E8L k (u j) = 0) ∧
      (∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) ∧
      (∀ j, ¬F.Atomic k (u j) → ∀ σ l, (F.BL σ).z k l ≠ u j ∧ (F.BL σ).z k l ≠ -u j) ∧
      ∀ t j, ¬F.Atomic k (u j) → F.KL k t t (u j) ≠ 0 ∨ F.KL k t (!t) (u j) ≠ 0 :=
  ⟨fun h ↦ ⟨h.a_ne, h.root, h.sep, h.off, fun t j hj ↦ (h.K_ne t j hj).imp_right And.left⟩,
    fun ⟨a_ne, root, sep, off, K_ne⟩ ↦ LAxisGeneric.mk' F k a_ne root sep off K_ne⟩

/-- **`KL_opp_eq_zero` at four roots with distinct squares, without `u² + q₀ ≠ 0`**: off
`μ_t = 0`, `K_t(u_j) = 0` forces `K_{t,-t}(u_j) = 0`. -/
theorem KL_opp_eq_zero_of_sep (t : Bool) (ha : (F.a : ℂ) ≠ 0) (hc : F.cL t ≠ 0) {u : Fin 4 → ℂ}
    (root : ∀ j, F.E8L k (u j) = 0) (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) (j : Fin 4)
    (hK : F.KL k t t (u j) = 0) : F.KL k t (!t) (u j) = 0 :=
  F.KL_opp_eq_zero k t ha hc (F.sq_add_q₀L_ne_of_sep k root sep j) (root j) hK

end FParams

end HubbardAtom
