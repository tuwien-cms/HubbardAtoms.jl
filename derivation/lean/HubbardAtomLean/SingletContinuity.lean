import HubbardAtomLean.DMContinuity
import HubbardAtomLean.SingletOperator

/-!
# Continuity of the singlet vertex

The second half of item 6(d) for the singlet channel: along a family of parameters `(U, β, μ)`
continuous at `x₀`, the vertex `Γ_s` of `SingletOperator.lean` (on the lattice `z'ₙ` for every `μ`
and `k`) is continuous at `x₀` wherever `1 + 𝒦` is invertible there.

The bounds of `SingletBounds.lean` hold with constants uniform near `x₀`. The regular part of
Proposition "pp" has two forms, `γ₊g₊g₊ + γ₋g₋g₋` for `b ≠ 0` with `γ± ∝ 1/b`, and the
partial fractions at `b = 0`; both are
`χ_s,reg = (a²r - a/2)(g₊g₊ + g₋g₋) + (2a³ - a²ρb)(g₊g₋g₊ + g₋g₊g₋)` with `r = ρ/b`, continued by
`βp₀` at `b = 0` (`chiRegS_eq_uniform`), as `g₊ - g₋ = 2ab g₊g₋`. At `k = 0`, `r` is the pair
coefficient `c₀` of `DMContinuity.lean`, continuous through `μ = 0`; at `k ≠ 0`, `b ≠ 0`.

* `Params.norm_chiS_le_all_of`, `Params.norm_chiS_diag_le_all_of`: the bounds of item 6(a) for
  `2χ_s` on `z'ₙ`, with constants;
* `Params.continuousAt_gammaOpS`: **`Γ_s,nm` is continuous** at `x₀` wherever `1 + 𝒦` is
  invertible there.
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ)

/-! ### The regular part, uniformly in `b` -/

/-- `r = ρ/b`, continued by `βp₀` at `b = 0`. -/
noncomputable def rhoB : ℂ := if A.b k = 0 then ((A.β * A.p₀ : ℝ) : ℂ) else A.ρ / A.b k

/-- **The regular part of `χ_s`, uniformly in `b`**: with `g± = 1/(t - t±)`,
`χ_s,reg = (a²r - a/2)(g₊g₊ + g₋g₋) + (2a³ - a²ρb)(g₊g₋g₊ + g₋g₊g₋)`. -/
theorem chiRegS_eq_uniform (n m : ℤ) : A.chiRegS k n m =
    (A.a ^ 2 * A.rhoB k - A.a / 2) * (1 / (A.t k n - A.tp k) / (A.t k m - A.tp k) +
        1 / (A.t k n - A.tm k) / (A.t k m - A.tm k)) +
      (2 * A.a ^ 3 - A.a ^ 2 * A.ρ * A.b k) *
        (1 / (A.t k n - A.tp k) / (A.t k n - A.tm k) / (A.t k m - A.tp k) +
          1 / (A.t k n - A.tm k) / (A.t k m - A.tp k) / (A.t k m - A.tm k)) := by
  have h1 := A.t_sub_tp_ne k n
  have h2 := A.t_sub_tm_ne k n
  have h3 := A.t_sub_tp_ne k m
  have h4 := A.t_sub_tm_ne k m
  unfold chiRegS rhoB
  split_ifs with hb
  · have htm : A.tm k = A.tp k := by rw [tm, tp, hb]; ring
    rw [chiReg₀, g₀, g₀, htm, hb]
    rw [htm] at h2 h4
    push_cast
    field_simp
    ring
  · rw [chiReg, γp, γm]
    have e1 : A.t k n - A.tm k = (A.t k n - A.tp k) + 2 * A.a * A.b k := by rw [tp, tm]; ring
    have e2 : A.t k m - A.tm k = (A.t k m - A.tp k) + 2 * A.a * A.b k := by rw [tp, tm]; ring
    rw [e1] at h2 ⊢
    rw [e2] at h4 ⊢
    generalize A.t k n - A.tp k = u at h1 h2 ⊢
    generalize A.t k m - A.tp k = v at h3 h4 ⊢
    have hbb : A.b k * (A.b k)⁻¹ = 1 := mul_inv_cancel₀ hb
    have Rn : 1 / u - 1 / (u + 2 * A.a * A.b k) =
        2 * A.a * A.b k * (1 / u * (1 / (u + 2 * A.a * A.b k))) := by
      field_simp
      ring
    have Rm : 1 / v - 1 / (v + 2 * A.a * A.b k) =
        2 * A.a * A.b k * (1 / v * (1 / (v + 2 * A.a * A.b k))) := by
      field_simp
      ring
    -- `γ₊P + γ₋Q = S(P + Q) + (D/2)(P - Q)` with `D = γ₊ - γ₋`, and
    -- `P - Q = 2ab(pqr + qrs)` by `Rn`, `Rm`
    linear_combination
      ((A.a * (A.ρ + 1) * (2 * A.a - A.b k) / (2 * A.b k) -
          A.a * (A.ρ - 1) * (2 * A.a + A.b k) / (2 * A.b k)) / 2 * (1 / v)) * Rn +
      ((A.a * (A.ρ + 1) * (2 * A.a - A.b k) / (2 * A.b k) -
          A.a * (A.ρ - 1) * (2 * A.a + A.b k) / (2 * A.b k)) / 2 *
          (1 / (u + 2 * A.a * A.b k))) * Rm +
      (-(A.a / 2) * (1 / u * (1 / v) + 1 / (u + 2 * A.a * A.b k) * (1 / (v + 2 * A.a * A.b k))) +
        (2 * A.a ^ 3 - A.a ^ 2 * A.ρ * A.b k) *
          (1 / u * (1 / (u + 2 * A.a * A.b k)) * (1 / v) +
            1 / (u + 2 * A.a * A.b k) * (1 / v) * (1 / (v + 2 * A.a * A.b k)))) * hbb

/-- At `k = 0`, `r = ρ/(2μ)` is the pair coefficient `c₀`. -/
theorem rhoB_zero : A.rhoB 0 = A.pairC 0 := by
  have hb : A.b 0 = 2 * A.μ := by simp [b, w, ω]
  by_cases hμ : A.μ = 0
  · rw [rhoB, ite_eq_left (by rw [hb, hμ]; simp), pairC, ite_eq_left ⟨hμ, rfl⟩]
  · have hb0 : A.b 0 ≠ 0 := by
      rw [hb]
      exact mul_ne_zero two_ne_zero (ofReal_ne_zero.mpr hμ)
    rw [rhoB, ite_eq_right hb0, pairC, ite_eq_right (fun h ↦ hμ h.1), hb]
    simp [ω]

/-- At `k ≠ 0`, `b ≠ 0`: `Im b = -ω_k`. -/
theorem b_ne_zero_of_k_ne (hk : k ≠ 0) : A.b k ≠ 0 := by
  intro h
  have him := congrArg Complex.im h
  simp only [b, w, sub_im, mul_im, ofReal_re, ofReal_im, I_re, I_im, mul_zero, mul_one,
    zero_add, zero_im, re_ofNat, im_ofNat, zero_mul, ω] at him
  have : (k : ℝ) ≠ 0 := by exact_mod_cast hk
  have : 2 * (k : ℝ) * Real.pi / A.β ≠ 0 :=
    div_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero this) Real.pi_ne_zero) A.β_pos.ne'
  exact this (by linarith)

/-! ### The entries through `1/z'ₙ` -/

/-- `|1/z'ₙ| ≤ β/π` for all `n`. -/
theorem norm_inv_zOp_le (n : ℤ) : ‖(A.zOp k n)⁻¹‖ ≤ A.β / Real.pi := by
  by_cases hn : mirror k n = n
  · have hβ := A.β_pos
    have hπ := Real.pi_pos
    have hz : Real.pi / A.β ≤ ‖A.zOp k n‖ := by
      refine le_trans ?_ (abs_im_le_norm _)
      rw [zOp, ite_eq_left hn]
      simp only [add_im, ofReal_im, mul_im, ofReal_re, I_im, mul_one, I_re, mul_zero, add_zero,
        zero_add]
      rw [abs_of_pos (div_pos hπ hβ)]
    rw [norm_inv]
    calc ‖A.zOp k n‖⁻¹ ≤ (Real.pi / A.β)⁻¹ := inv_anti₀ (by positivity) hz
      _ = A.β / Real.pi := inv_div _ _
  · rw [A.zOp_of_ne k hn]
    exact A.norm_inv_z_le_of_ne k hn

/-- `K = βa²(1 - 8p₁ - ρ²)/2`, the mirror entries `2χ_s,nn̄ - 2χ_s,reg,nn̄ = K/𝒫(tₙ)`. -/
noncomputable def mirK : ℂ := A.β * A.a ^ 2 * (1 - 8 * A.p₁ - A.ρ ^ 2) / 2

/-- **On a mirror pair**: `2χ_s,nn̄ = K/𝒫(tₙ) + 2χ_s,reg,nn̄`. -/
theorem two_chiS_mirror {n : ℤ} (hn : n ≠ mirror k n) :
    2 * A.chiS k n (mirror k n) = A.mirK / A.calP k (A.t k n) +
      2 * A.chiRegS k n (mirror k n) := by
  have hP := A.calP_ne_zero k n
  rw [A.chiS_eq_regS k, mirrorBlock_mirror hn, sub_neg_eq_add, A.Ls_add_Bpp, mirK]
  field_simp

/-- `zₙ³[β(tₙ + c/2)/𝒫(tₙ) + β/zₙ²]` as a function of `s = 1/zₙ` (`SingletBounds.bdd_of_diag`). -/
noncomputable def qDiag (c s : ℂ) : ℂ :=
  A.β * (A.tS k s * ((A.b k ^ 2 / 4 - A.μ ^ 2) * s + 2 * A.μ) + 2 * A.a ^ 2 * s * A.tS k s +
    (A.a ^ 4 - A.a ^ 2 * A.b k ^ 2) * s ^ 3 + c * s / 2) * (A.calPS k s)⁻¹

/-- **On the diagonal, off the fixed point**:
`zₙ³[2(χ_s,nn - χ_s,reg,nn) + β/zₙ²] = qDiag(c, 1/zₙ)`, `c = Aₛ + aρb + a²ρ²`. -/
theorem z_cube_two_chiS_diag {n : ℤ} (hn : mirror k n ≠ n) :
    A.z k n ^ 3 * (2 * (A.chiS k n n - A.chiRegS k n n) + A.β / A.z k n ^ 2) =
      A.qDiag k (A.As k + A.a * A.ρ * A.b k + A.a ^ 2 * A.ρ ^ 2) (A.z k n)⁻¹ := by
  have hz := A.z_ne_zero_of_ne k hn
  have hP := A.calP_ne_zero k n
  have e : 2 * (A.chiS k n n - A.chiRegS k n n) = A.β * (A.t k n +
      (A.As k + A.a * A.ρ * A.b k + A.a ^ 2 * A.ρ ^ 2) / 2) / A.calP k (A.t k n) := by
    rw [A.chiS_eq_regS, add_sub_cancel_right, mirrorBlock_self (Ne.symm hn), Ls,
      A.Bpp_eq_calG_div, calG]
    field_simp
    ring
  rw [e, qDiag, A.calPS_inv k hz, tS]
  field_simp
  rw [calP, A.t_eq_z]
  ring

/-! ### The bounds of item 6(a) for `2χ_s` on `z'ₙ`, with constants -/

private theorem nmul {a b : ℂ} {A B : ℝ} (ha : ‖a‖ ≤ A) (hb : ‖b‖ ≤ B) (hA : 0 ≤ A) :
    ‖a * b‖ ≤ A * B := by
  rw [norm_mul]
  exact mul_le_mul ha hb (norm_nonneg _) hA

private theorem nadd {a b : ℂ} {A B : ℝ} (ha : ‖a‖ ≤ A) (hb : ‖b‖ ≤ B) : ‖a + b‖ ≤ A + B :=
  (norm_add_le _ _).trans (add_le_add ha hb)

/-- **The regular part on `z'ₙ`**: `|z'ₙ|²|z'ₘ|²|χ_s,reg,nm|` is bounded, for all `n`, `m`. -/
theorem norm_chiRegS_le_of {R Pp Pm Ca Cb : ℝ} (hR : A.β / Real.pi ≤ R)
    (hPp : ∀ j, ‖A.zOp k j ^ 2 / (A.t k j - A.tp k)‖ ≤ Pp)
    (hPm : ∀ j, ‖A.zOp k j ^ 2 / (A.t k j - A.tm k)‖ ≤ Pm)
    (hCa : ‖A.a ^ 2 * A.rhoB k - A.a / 2‖ ≤ Ca)
    (hCb : ‖2 * A.a ^ 3 - A.a ^ 2 * A.ρ * A.b k‖ ≤ Cb) (n m : ℤ) :
    ‖A.zOp k n ^ 2 * A.zOp k m ^ 2 * A.chiRegS k n m‖ ≤
      Ca * (Pp * Pp + Pm * Pm) + Cb * (R ^ 2 * Pp * Pm * Pp + R ^ 2 * Pm * Pp * Pm) := by
  have hR0 : 0 ≤ R := (div_nonneg A.β_pos.le Real.pi_pos.le).trans hR
  have hPp0 := (norm_nonneg _).trans (hPp 0)
  have hPm0 := (norm_nonneg _).trans (hPm 0)
  have hCa0 := (norm_nonneg _).trans hCa
  have hi : ∀ j, ‖(A.zOp k j)⁻¹ ^ 2‖ ≤ R ^ 2 := fun j ↦ by
    rw [norm_pow]
    exact pow_le_pow_left₀ (norm_nonneg _) ((A.norm_inv_zOp_le k j).trans hR) 2
  have hz := A.zOp_ne_zero k
  have h1 := A.t_sub_tp_ne k n
  have h2 := A.t_sub_tm_ne k n
  have h3 := A.t_sub_tp_ne k m
  have h4 := A.t_sub_tm_ne k m
  have key : A.zOp k n ^ 2 * A.zOp k m ^ 2 * A.chiRegS k n m =
      (A.a ^ 2 * A.rhoB k - A.a / 2) *
          (A.zOp k n ^ 2 / (A.t k n - A.tp k) * (A.zOp k m ^ 2 / (A.t k m - A.tp k)) +
            A.zOp k n ^ 2 / (A.t k n - A.tm k) * (A.zOp k m ^ 2 / (A.t k m - A.tm k))) +
        (2 * A.a ^ 3 - A.a ^ 2 * A.ρ * A.b k) *
          ((A.zOp k n)⁻¹ ^ 2 * (A.zOp k n ^ 2 / (A.t k n - A.tp k)) *
              (A.zOp k n ^ 2 / (A.t k n - A.tm k)) * (A.zOp k m ^ 2 / (A.t k m - A.tp k)) +
            (A.zOp k m)⁻¹ ^ 2 * (A.zOp k n ^ 2 / (A.t k n - A.tm k)) *
              (A.zOp k m ^ 2 / (A.t k m - A.tp k)) * (A.zOp k m ^ 2 / (A.t k m - A.tm k))) := by
    rw [A.chiRegS_eq_uniform]
    have := hz n
    have := hz m
    field_simp
  rw [key]
  exact nadd (nmul hCa (nadd (nmul (hPp n) (hPp m) hPp0) (nmul (hPm n) (hPm m) hPm0)) hCa0)
    (nmul hCb (nadd (nmul (nmul (nmul (hi n) (hPp n) (by positivity)) (hPm n) (by positivity))
      (hPp m) (by positivity)) (nmul (nmul (nmul (hi m) (hPm n) (by positivity)) (hPp m)
        (by positivity)) (hPm m) (by positivity))) ((norm_nonneg _).trans hCb))

/-- **Off-diagonal entries of `2χ_s` on `z'ₙ`, with constants**: for `n ≠ m`,
`|z'ₙ|²|z'ₘ|²|2χ_s,nm|` is bounded by the constants below. -/
theorem norm_chiS_le_all_of {R M Pp Pm Ca Cb CP CK : ℝ} (hR : A.β / Real.pi ≤ R)
    (hM : |A.μ| ≤ M) (hPp : ∀ j, ‖A.zOp k j ^ 2 / (A.t k j - A.tp k)‖ ≤ Pp)
    (hPm : ∀ j, ‖A.zOp k j ^ 2 / (A.t k j - A.tm k)‖ ≤ Pm)
    (hCa : ‖A.a ^ 2 * A.rhoB k - A.a / 2‖ ≤ Ca)
    (hCb : ‖2 * A.a ^ 3 - A.a ^ 2 * A.ρ * A.b k‖ ≤ Cb)
    (hCP : ∀ j, ‖A.zOp k j ^ 4 / A.calP k (A.t k j)‖ ≤ CP) (hCK : ‖A.mirK‖ ≤ CK)
    (n m : ℤ) (hnm : n ≠ m) :
    ‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2 * ‖2 * A.chiS k n m‖ ≤
      (1 + 2 * M * R) ^ 2 * CP * CK +
        2 * (Ca * (Pp * Pp + Pm * Pm) + Cb * (R ^ 2 * Pp * Pm * Pp + R ^ 2 * Pm * Pp * Pm)) := by
  have hR0 : 0 ≤ R := (div_nonneg A.β_pos.le Real.pi_pos.le).trans hR
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hM
  have hCP0 := (norm_nonneg _).trans (hCP 0)
  have hCK0 := (norm_nonneg _).trans hCK
  have hreg := A.norm_chiRegS_le_of k hR hPp hPm hCa hCb n m
  set CR := Ca * (Pp * Pp + Pm * Pm) + Cb * (R ^ 2 * Pp * Pm * Pp + R ^ 2 * Pm * Pp * Pm)
  have hlhs : ‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2 * ‖2 * A.chiS k n m‖ =
      ‖A.zOp k n ^ 2 * A.zOp k m ^ 2 * (2 * A.chiS k n m)‖ := by
    simp only [norm_mul, norm_pow]
  rw [hlhs]
  by_cases hmir : m = mirror k n
  · subst hmir
    have hn : mirror k n ≠ n := Ne.symm hnm
    have hn' : mirror k (mirror k n) ≠ mirror k n := by rw [mirror_mirror]; exact hnm
    have hz := A.zOp_ne_zero k n
    have hP := A.calP_ne_zero k n
    have hq : ‖A.zOp k (mirror k n) / A.zOp k n‖ ≤ 1 + 2 * M * R := by
      rw [A.zOp_of_ne k hn, A.zOp_of_ne k hn']
      exact A.norm_z_mirror_div_le k hn hM hR
    have key : A.zOp k n ^ 2 * A.zOp k (mirror k n) ^ 2 * (2 * A.chiS k n (mirror k n)) =
        (A.zOp k (mirror k n) / A.zOp k n) ^ 2 * (A.zOp k n ^ 4 / A.calP k (A.t k n)) * A.mirK +
          2 * (A.zOp k n ^ 2 * A.zOp k (mirror k n) ^ 2 * A.chiRegS k n (mirror k n)) := by
      rw [A.two_chiS_mirror k hnm]
      field_simp
    rw [key]
    refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
    · rw [norm_mul, norm_mul, norm_pow]
      exact mul_le_mul (mul_le_mul (pow_le_pow_left₀ (norm_nonneg _) hq 2) (hCP n)
        (norm_nonneg _) (by positivity)) hCK (norm_nonneg _) (by positivity)
    · rw [norm_mul, Complex.norm_two]
      exact mul_le_mul_of_nonneg_left hreg zero_le_two
  · rw [A.chiS_eq_regS k, mirrorBlock_of_ne hnm hmir, zero_add, mul_left_comm, norm_mul,
      Complex.norm_two]
    have : 0 ≤ (1 + 2 * M * R) ^ 2 * CP * CK := by positivity
    linarith

/-- **Diagonal entries of `2χ_s` on `z'ₙ`, with constants**, `Cf` bounding them at the fixed
point. -/
theorem norm_chiS_diag_le_all_of {R M Pp Pm Ca Cb Cq Cf : ℝ} (hR : A.β / Real.pi ≤ R)
    (hM : |A.μ| ≤ M) (hMπ : Real.pi / A.β ≤ M)
    (hPp : ∀ j, ‖A.zOp k j ^ 2 / (A.t k j - A.tp k)‖ ≤ Pp)
    (hPm : ∀ j, ‖A.zOp k j ^ 2 / (A.t k j - A.tm k)‖ ≤ Pm)
    (hCa : ‖A.a ^ 2 * A.rhoB k - A.a / 2‖ ≤ Ca)
    (hCb : ‖2 * A.a ^ 3 - A.a ^ 2 * A.ρ * A.b k‖ ≤ Cb)
    (hq : ∀ j, mirror k j ≠ j →
      ‖A.qDiag k (A.As k + A.a * A.ρ * A.b k + A.a ^ 2 * A.ρ ^ 2) (A.z k j)⁻¹‖ ≤ Cq)
    (hf : ∀ j, mirror k j = j → ‖2 * A.chiS k j j + A.β / A.zOp k j ^ 2‖ ≤ Cf)
    (hCq0 : 0 ≤ Cq) (hCf0 : 0 ≤ Cf) (n : ℤ) :
    ‖A.zOp k n‖ ^ 3 * ‖2 * A.chiS k n n + A.β / A.zOp k n ^ 2‖ ≤
      Cq + 2 * R * (Ca * (Pp * Pp + Pm * Pm) + Cb * (R ^ 2 * Pp * Pm * Pp + R ^ 2 * Pm * Pp * Pm)) +
        8 * M ^ 3 * Cf := by
  have hR0 : 0 ≤ R := (div_nonneg A.β_pos.le Real.pi_pos.le).trans hR
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hM
  have hreg := A.norm_chiRegS_le_of k hR hPp hPm hCa hCb n n
  set CR := Ca * (Pp * Pp + Pm * Pm) + Cb * (R ^ 2 * Pp * Pm * Pp + R ^ 2 * Pm * Pp * Pm)
  have hCR0 : 0 ≤ CR := (norm_nonneg _).trans hreg
  by_cases hnf : mirror k n = n
  · have h := A.norm_zOp_fixed_le k hnf
    have h3 : ‖A.zOp k n‖ ^ 3 ≤ 8 * M ^ 3 := by
      have := pow_le_pow_left₀ (norm_nonneg _)
        (h.trans (show |A.μ| + Real.pi / A.β ≤ 2 * M by linarith)) 3
      linarith [show (2 * M) ^ 3 = 8 * M ^ 3 by ring]
    calc _ ≤ 8 * M ^ 3 * Cf := mul_le_mul h3 (hf n hnf) (norm_nonneg _) (by positivity)
      _ ≤ _ := by
        have : 0 ≤ 2 * R * CR := by positivity
        linarith
  · have hz := A.z_ne_zero_of_ne k hnf
    rw [A.zOp_of_ne k hnf] at hreg ⊢
    have key : A.z k n ^ 3 * (2 * A.chiS k n n + A.β / A.z k n ^ 2) =
        A.qDiag k (A.As k + A.a * A.ρ * A.b k + A.a ^ 2 * A.ρ ^ 2) (A.z k n)⁻¹ +
          2 * (A.z k n)⁻¹ * (A.z k n ^ 2 * A.z k n ^ 2 * A.chiRegS k n n) := by
      rw [← A.z_cube_two_chiS_diag k hnf]
      field_simp
      ring
    rw [← norm_pow, ← norm_mul, key]
    have ht : ‖(A.z k n)⁻¹‖ ≤ R := (A.norm_inv_z_le_of_ne k hnf).trans hR
    refine (norm_add_le _ _).trans ?_
    rw [norm_mul, norm_mul, Complex.norm_two]
    have h1 := mul_le_mul ht hreg (norm_nonneg _) hR0
    have : 0 ≤ 8 * M ^ 3 * Cf := by positivity
    nlinarith [hq n hnf, norm_nonneg ((A.z k n)⁻¹)]

/-! ### Along a family of parameters -/

section Family

variable {X : Type*} [TopologicalSpace X] {U β μ : X → ℝ} {hβ : ∀ x, 0 < β x} {x₀ : X}
  (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)

omit hU hb hm in
/-- A function continuous at `x₀` is bounded near `x₀`. -/
private theorem eventually_norm_le_of_continuousAt' {f : X → ℂ} (hf : ContinuousAt f x₀) :
    ∀ᶠ x in 𝓝 x₀, ‖f x‖ ≤ ‖f x₀‖ + 1 :=
  (hf.norm.eventually (gt_mem_nhds (lt_add_one _))).mono fun _ h ↦ h.le

include hb hm in
theorem continuousAt_fam_b : ContinuousAt (fun x ↦ (fam U β μ hβ x).b k) x₀ := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [b, w, ω, fam]
  fun_prop (disch := assumption)

include hb hm in
theorem continuousAt_fam_t (n : ℤ) : ContinuousAt (fun x ↦ (fam U β μ hβ x).t k n) x₀ := by
  have hx := continuousAt_fam_x (U := U) (hβ := hβ) hb hm n
  have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
  simp only [t]
  fun_prop

include hU hb hm

/-- `r = ρ/b` (`βp₀` at `b = 0`) is continuous. -/
theorem continuousAt_fam_rhoB : ContinuousAt (fun x ↦ (fam U β μ hβ x).rhoB k) x₀ := by
  by_cases hk : k = 0
  · subst hk
    simp only [rhoB_zero]
    exact continuousAt_fam_pairC (hβ := hβ) hU hb hm 0
  · have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
    have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
    have hb0 := (fam U β μ hβ x₀).b_ne_zero_of_k_ne k hk
    have e : ∀ x, (fam U β μ hβ x).rhoB k = (fam U β μ hβ x).ρ / (fam U β μ hβ x).b k :=
      fun x ↦ by rw [rhoB, ite_eq_right ((fam U β μ hβ x).b_ne_zero_of_k_ne k hk)]
    simp only [e]
    fun_prop (disch := assumption)

/-- **The entries of `χ_s` are continuous** in the parameters. -/
theorem continuousAt_fam_chiS (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).chiS k n m) x₀ := by
  have e : ∀ (A : Params) (k' : ℤ), A.chiUpUp k' n m =
      (A.chiR k' 1 n m + A.chiR k' (-1) n m) / 2 ∧
      A.chiUpDn k' n m = (A.chiR k' 1 n m - A.chiR k' (-1) n m) / 2 := fun A k' ↦ by
    simp only [chiR]
    constructor <;> ring
  have h1 := continuousAt_fam_chiR (hβ := hβ) (k := -k - n - m - 1) (ε := 1) hU hb hm n m
  have h2 := continuousAt_fam_chiR (hβ := hβ) (k := -k - n - m - 1) (ε := -1) hU hb hm n m
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
  have ht := continuousAt_fam_t (U := U) (hβ := hβ) (k := k) hb hm n
  have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
  have hP0 := (fam U β μ hβ x₀).calP_ne_zero k n
  simp only [chiS, chiPPUpUp, chiPPUpDn, (e _ _).1, (e _ _).2, Params.Bpp_eq_calG_div, calG, calP] at hP0 ⊢
  fun_prop (disch := assumption)

/-- The bubble `B(νₙ)` of the particle–particle channel is continuous. -/
theorem continuousAt_fam_Bpp (n : ℤ) : ContinuousAt (fun x ↦ (fam U β μ hβ x).Bpp k n) x₀ := by
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
  have ht := continuousAt_fam_t (U := U) (hβ := hβ) (k := k) hb hm n
  have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
  have hP0 := (fam U β μ hβ x₀).calP_ne_zero k n
  simp only [Params.Bpp_eq_calG_div, calG, calP] at hP0 ⊢
  fun_prop (disch := assumption)

/-- The lattice for the bounds through `1/zₙ`: `zₙ`, with the fixed point replaced by its
neighbour `n + 1`, which is not fixed. -/
noncomputable def zSh (A : Params) (k n : ℤ) : ℂ := A.z k (if mirror k n = n then n + 1 else n)

omit hU hb hm in
theorem zSh_of_ne (A : Params) {n : ℤ} (hn : mirror k n ≠ n) : zSh A k n = A.z k n := by
  simp [zSh, hn]

omit hU hb hm in
theorem not_fixed_zSh_index (n : ℤ) :
    mirror k (if mirror k n = n then n + 1 else n) ≠ (if mirror k n = n then n + 1 else n) := by
  split_ifs with h
  · unfold mirror at h ⊢
    omega
  · exact h

omit hU hm in
/-- `1/zSh → 0` as `|n| → ∞`, uniformly near `x₀`. -/
theorem eventually_norm_inv_zSh_lt :
    ∀ e > 0, ∃ N : ℕ, ∀ᶠ x in 𝓝 x₀, ∀ n : ℤ, (N : ℤ) ≤ |n| →
      ‖(zSh (fam U β μ hβ x) k n)⁻¹‖ < e := by
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
  rw [zSh_of_ne k _ hnf]
  exact hx n (by linarith [abs_nonneg k])

omit hU in
/-- A function `q(x, s)` jointly continuous at `(x₀, 0)` and at each `(x₀, 1/zₙ(x₀))`, `n` not
fixed, is bounded near `x₀` at `s = 1/zₙ`, uniformly in the non-fixed `n`. -/
theorem exists_eventually_bdd_of_nonfixed {q : X → ℂ → ℂ}
    (hq : ∀ s₀, (s₀ = 0 ∨ ∃ j, mirror k j ≠ j ∧ s₀ = ((fam U β μ hβ x₀).z k j)⁻¹) →
      ContinuousAt (fun p : X × ℂ ↦ q p.1 p.2) (x₀, s₀)) :
    ∃ C, 0 ≤ C ∧ ∀ᶠ x in 𝓝 x₀, ∀ j, mirror k j ≠ j → ‖q x ((fam U β μ hβ x).z k j)⁻¹‖ ≤ C := by
  have hn : ∀ n, ContinuousAt (fun x ↦ q x (zSh (fam U β μ hβ x) k n)⁻¹) x₀ := fun n ↦ by
    have hj := not_fixed_zSh_index k n
    set j := if mirror k n = n then n + 1 else n
    exact (hq _ (Or.inr ⟨j, hj, rfl⟩)).comp (f := fun x ↦ (x, ((fam U β μ hβ x).z k j)⁻¹))
      (continuousAt_id.prodMk ((continuousAt_z (U := U) (hβ := hβ) k hb hm j).inv₀
        ((fam U β μ hβ x₀).z_ne_zero_of_ne k hj)))
  obtain ⟨C, hC⟩ := OpLattice.eventually_bdd_of_inv_z (zf := fun x n ↦ zSh (fam U β μ hβ x) k n)
    (hq 0 (Or.inl rfl)) hn (eventually_norm_inv_zSh_lt (U := U) (μ := μ) (hβ := hβ) (k := k) hb)
  refine ⟨C, (norm_nonneg _).trans (hC.self_of_nhds 0), ?_⟩
  filter_upwards [hC] with x hx j hj
  have := hx j
  rwa [zSh_of_ne k _ hj] at this

include hU hb hm

omit hU hb hm in
/-- The parameters of `fam` composed with the first projection are continuous at `(x₀, s₀)`. -/
private theorem fst_cont {f : X → ℂ} (s₀ : ℂ) (hf : ContinuousAt f x₀) :
    ContinuousAt (fun p : X × ℂ ↦ f p.1) (x₀, s₀) :=
  ContinuousAt.comp (x := ((x₀, s₀) : X × ℂ)) hf continuousAt_fst

omit hU in
/-- `z'ⱼ²/(tⱼ - c)` is bounded near `x₀`, uniformly in `j`, for `c` continuous with
`tⱼ - c ≠ 0`. -/
theorem exists_eventually_sq_div_t_sub {c : X → ℂ} (hcc : ContinuousAt c x₀)
    (hne : ∀ x j, (fam U β μ hβ x).t k j - c x ≠ 0) : ∃ P : ℝ, 0 ≤ P ∧ ∀ᶠ x in 𝓝 x₀, ∀ j,
      ‖(fam U β μ hβ x).zOp k j ^ 2 / ((fam U β μ hβ x).t k j - c x)‖ ≤ P := by
  have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
  have hμc : ContinuousAt (fun x ↦ ((fam U β μ hβ x).μ : ℂ)) x₀ := by
    simp only [fam]
    fun_prop
  -- off the fixed point, through `q(x, s) = (T(s) - c s²)⁻¹` at `s = 1/zⱼ`
  have e : ∀ x j, (fam U β μ hβ x).z k j ≠ 0 →
      (fam U β μ hβ x).z k j ^ 2 / ((fam U β μ hβ x).t k j - c x) =
        ((fam U β μ hβ x).tS k ((fam U β μ hβ x).z k j)⁻¹ -
          c x * ((fam U β μ hβ x).z k j)⁻¹ ^ 2)⁻¹ := fun x j hz ↦ by
    rw [(fam U β μ hβ x).t_eq_tS k hz, ← inv_div]
    congr 1
    field_simp
  obtain ⟨C, hC0, hC⟩ := exists_eventually_bdd_of_nonfixed (U := U) (μ := μ) (hβ := hβ) (k := k)
    hb hm (q := fun x s ↦ ((fam U β μ hβ x).tS k s - c x * s ^ 2)⁻¹) fun s₀ hs₀ ↦ by
      have h1 := fst_cont s₀ hbc
      have h2 := fst_cont s₀ hμc
      have h3 := fst_cont s₀ hcc
      have hden : (fam U β μ hβ x₀).tS k s₀ - c x₀ * s₀ ^ 2 ≠ 0 := by
        rcases hs₀ with rfl | ⟨j, hj, rfl⟩
        · simp [tS]
        · have hz := (fam U β μ hβ x₀).z_ne_zero_of_ne k hj
          have := e x₀ j hz
          intro h0
          rw [h0, inv_zero] at this
          exact div_ne_zero (pow_ne_zero 2 hz) (hne x₀ j) this
      refine ContinuousAt.inv₀ ?_ hden
      have hs : ContinuousAt (fun p : X × ℂ ↦ p.2) (x₀, s₀) := continuousAt_snd
      simp only [tS]
      exact (((((h1.pow 2).div_const 4).sub (h2.pow 2)).mul (hs.pow 2)).add
        ((continuousAt_const.mul h2).mul hs)).sub continuousAt_const |>.sub (h3.mul (hs.pow 2))
  -- at the fixed point, by continuity
  obtain ⟨jf, hjfd⟩ : ∃ j : ℤ, j = (-k - 1) / 2 := ⟨_, rfl⟩
  have hfc : ContinuousAt (fun x ↦ (fam U β μ hβ x).zOp k jf ^ 2 /
      ((fam U β μ hβ x).t k jf - c x)) x₀ :=
    ((continuousAt_fam_zOp (U := U) (hβ := hβ) (k := k) hb hm jf).pow 2).div
      ((continuousAt_fam_t (U := U) (hβ := hβ) (k := k) hb hm jf).sub hcc) (hne x₀ jf)
  refine ⟨C + (‖(fam U β μ hβ x₀).zOp k jf ^ 2 / ((fam U β μ hβ x₀).t k jf - c x₀)‖ + 1),
    by positivity, ?_⟩
  filter_upwards [hC, (hfc.norm.eventually (gt_mem_nhds (lt_add_one _)))] with x hx hf j
  have h0 := norm_nonneg ((fam U β μ hβ x₀).zOp k jf ^ 2 / ((fam U β μ hβ x₀).t k jf - c x₀))
  by_cases hj : mirror k j = j
  · have : j = jf := by rw [hjfd]; unfold mirror at hj; omega
    subst this
    linarith
  · have hz := (fam U β μ hβ x).z_ne_zero_of_ne k hj
    rw [zOp_of_ne _ k hj, e x j hz]
    have := hx j hj
    linarith

/-- `z'ⱼ⁴/𝒫(tⱼ)` is bounded near `x₀`, uniformly in `j`. -/
theorem exists_eventually_pow_div_calP : ∃ P : ℝ, 0 ≤ P ∧ ∀ᶠ x in 𝓝 x₀, ∀ j,
    ‖(fam U β μ hβ x).zOp k j ^ 4 / (fam U β μ hβ x).calP k ((fam U β μ hβ x).t k j)‖ ≤ P := by
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
  have hμc : ContinuousAt (fun x ↦ ((fam U β μ hβ x).μ : ℂ)) x₀ := by
    simp only [fam]
    fun_prop
  obtain ⟨C, hC0, hC⟩ := exists_eventually_bdd_of_nonfixed (U := U) (μ := μ) (hβ := hβ) (k := k)
    hb hm (q := fun x s ↦ ((fam U β μ hβ x).calPS k s)⁻¹) fun s₀ hs₀ ↦ by
      have h1 := fst_cont s₀ hbc
      have h2 := fst_cont s₀ hμc
      have h3 := fst_cont s₀ ha
      have hden : (fam U β μ hβ x₀).calPS k s₀ ≠ 0 := by
        rcases hs₀ with rfl | ⟨j, hj, rfl⟩
        · simp [calPS, tS]
        · have hz := (fam U β μ hβ x₀).z_ne_zero_of_ne k hj
          rw [(fam U β μ hβ x₀).calPS_inv k hz]
          exact div_ne_zero ((fam U β μ hβ x₀).calP_ne_zero k j) (pow_ne_zero 4 hz)
      refine ContinuousAt.inv₀ ?_ hden
      have hs : ContinuousAt (fun p : X × ℂ ↦ p.2) (x₀, s₀) := continuousAt_snd
      have hT : ContinuousAt (fun p : X × ℂ ↦ (fam U β μ hβ p.1).tS k p.2) (x₀, s₀) := by
        simp only [tS]
        exact (((((h1.pow 2).div_const 4).sub (h2.pow 2)).mul (hs.pow 2)).add
          ((continuousAt_const.mul h2).mul hs)).sub continuousAt_const
      simp only [calPS]
      exact ((hT.add ((h3.pow 2).mul (hs.pow 2))).pow 2).sub
        (((h3.pow 2).mul (h1.pow 2)).mul (hs.pow 4))
  obtain ⟨jf, hjfd⟩ : ∃ j : ℤ, j = (-k - 1) / 2 := ⟨_, rfl⟩
  have hfc : ContinuousAt (fun x ↦ (fam U β μ hβ x).zOp k jf ^ 4 /
      (fam U β μ hβ x).calP k ((fam U β μ hβ x).t k jf)) x₀ := by
    have hz := continuousAt_fam_zOp (U := U) (hβ := hβ) (k := k) hb hm jf
    have ht := continuousAt_fam_t (U := U) (hβ := hβ) (k := k) hb hm jf
    have h0 := (fam U β μ hβ x₀).calP_ne_zero k jf
    simp only [calP] at h0 ⊢
    fun_prop (disch := assumption)
  refine ⟨C + (‖(fam U β μ hβ x₀).zOp k jf ^ 4 /
      (fam U β μ hβ x₀).calP k ((fam U β μ hβ x₀).t k jf)‖ + 1), by positivity, ?_⟩
  filter_upwards [hC, (hfc.norm.eventually (gt_mem_nhds (lt_add_one _)))] with x hx hf j
  by_cases hj : mirror k j = j
  · have : j = jf := by rw [hjfd]; unfold mirror at hj; omega
    subst this
    linarith
  · have hz := (fam U β μ hβ x).z_ne_zero_of_ne k hj
    rw [zOp_of_ne _ k hj, ← inv_div, ← (fam U β μ hβ x).calPS_inv k hz]
    have := hx j hj
    have := norm_nonneg ((fam U β μ hβ x₀).zOp k jf ^ 4 /
      (fam U β μ hβ x₀).calP k ((fam U β μ hβ x₀).t k jf))
    linarith

/-- `qDiag(c, 1/zⱼ)` is bounded near `x₀`, uniformly in the non-fixed `j`. -/
theorem exists_eventually_qDiag : ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ x in 𝓝 x₀, ∀ j, mirror k j ≠ j →
    ‖(fam U β μ hβ x).qDiag k ((fam U β μ hβ x).As k +
      (fam U β μ hβ x).a * (fam U β μ hβ x).ρ * (fam U β μ hβ x).b k +
        (fam U β μ hβ x).a ^ 2 * (fam U β μ hβ x).ρ ^ 2) ((fam U β μ hβ x).z k j)⁻¹‖ ≤ C := by
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
  obtain ⟨-, hp₁, -⟩ := continuousAt_fam_p (hβ := hβ) hU hb hm
  have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  have hμc : ContinuousAt (fun x ↦ ((fam U β μ hβ x).μ : ℂ)) x₀ := by
    simp only [fam]
    fun_prop
  have hcc : ContinuousAt (fun x ↦ (fam U β μ hβ x).As k +
      (fam U β μ hβ x).a * (fam U β μ hβ x).ρ * (fam U β μ hβ x).b k +
        (fam U β μ hβ x).a ^ 2 * (fam U β μ hβ x).ρ ^ 2) x₀ := by
    simp only [As]
    fun_prop
  exact exists_eventually_bdd_of_nonfixed (U := U) (μ := μ) (hβ := hβ) (k := k) hb hm
    (q := fun x s ↦ (fam U β μ hβ x).qDiag k ((fam U β μ hβ x).As k +
      (fam U β μ hβ x).a * (fam U β μ hβ x).ρ * (fam U β μ hβ x).b k +
        (fam U β μ hβ x).a ^ 2 * (fam U β μ hβ x).ρ ^ 2) s) fun s₀ hs₀ ↦ by
      have h1 := fst_cont s₀ hbc
      have h2 := fst_cont s₀ hμc
      have h3 := fst_cont s₀ ha
      have h4 := fst_cont s₀ hβc
      have h5 := fst_cont s₀ hcc
      have hs : ContinuousAt (fun p : X × ℂ ↦ p.2) (x₀, s₀) := continuousAt_snd
      have hden : (fam U β μ hβ x₀).calPS k s₀ ≠ 0 := by
        rcases hs₀ with rfl | ⟨j, hj, rfl⟩
        · simp [calPS, tS]
        · have hz := (fam U β μ hβ x₀).z_ne_zero_of_ne k hj
          rw [(fam U β μ hβ x₀).calPS_inv k hz]
          exact div_ne_zero ((fam U β μ hβ x₀).calP_ne_zero k j) (pow_ne_zero 4 hz)
      have hT : ContinuousAt (fun p : X × ℂ ↦ (fam U β μ hβ p.1).tS k p.2) (x₀, s₀) := by
        simp only [tS]
        exact (((((h1.pow 2).div_const 4).sub (h2.pow 2)).mul (hs.pow 2)).add
          ((continuousAt_const.mul h2).mul hs)).sub continuousAt_const
      have hP : ContinuousAt (fun p : X × ℂ ↦ (fam U β μ hβ p.1).calPS k p.2) (x₀, s₀) := by
        simp only [calPS]
        exact ((hT.add ((h3.pow 2).mul (hs.pow 2))).pow 2).sub
          (((h3.pow 2).mul (h1.pow 2)).mul (hs.pow 4))
      simp only [qDiag]
      exact (h4.mul ((((hT.mul ((((h1.pow 2).div_const 4).sub (h2.pow 2)).mul hs |>.add
        (continuousAt_const.mul h2))).add (((continuousAt_const.mul (h3.pow 2)).mul hs).mul hT)).add
        (((h3.pow 4).sub ((h3.pow 2).mul (h1.pow 2))).mul (hs.pow 3))).add
        ((h5.mul hs).div_const 2))).mul (hP.inv₀ hden)

/-- At the fixed point, `2χ_s,jj + β/z'ⱼ²` is bounded near `x₀`. -/
theorem exists_eventually_singlet_fixed : ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ x in 𝓝 x₀, ∀ j, mirror k j = j →
    ‖2 * (fam U β μ hβ x).chiS k j j + ((fam U β μ hβ x).β : ℂ) /
      (fam U β μ hβ x).zOp k j ^ 2‖ ≤ C := by
  obtain ⟨jf, hjfd⟩ : ∃ j : ℤ, j = (-k - 1) / 2 := ⟨_, rfl⟩
  have hχ := continuousAt_fam_chiS (hβ := hβ) (k := k) hU hb hm jf jf
  have hz := continuousAt_fam_zOp (U := U) (hβ := hβ) (k := k) hb hm jf
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  have hfc : ContinuousAt (fun x ↦ 2 * (fam U β μ hβ x).chiS k jf jf +
      ((fam U β μ hβ x).β : ℂ) / (fam U β μ hβ x).zOp k jf ^ 2) x₀ :=
    (continuousAt_const.mul hχ).add (hβc.div (hz.pow 2)
      (pow_ne_zero 2 ((fam U β μ hβ x₀).zOp_ne_zero k jf)))
  refine ⟨‖2 * (fam U β μ hβ x₀).chiS k jf jf + ((fam U β μ hβ x₀).β : ℂ) /
      (fam U β μ hβ x₀).zOp k jf ^ 2‖ + 1, by positivity,
    (hfc.norm.eventually (gt_mem_nhds (lt_add_one _))).mono fun x hx j hj ↦ ?_⟩
  have : j = jf := by rw [hjfd]; unfold mirror at hj; omega
  subst this
  exact hx.le

/-- The coefficients `a²r - a/2`, `2a³ - a²ρb` and `K` are bounded near `x₀`. -/
theorem exists_eventually_singlet_coeffs : ∃ Ca Cb CK : ℝ, ∀ᶠ x in 𝓝 x₀,
    ‖(fam U β μ hβ x).a ^ 2 * (fam U β μ hβ x).rhoB k - (fam U β μ hβ x).a / 2‖ ≤ Ca ∧
    ‖2 * (fam U β μ hβ x).a ^ 3 - (fam U β μ hβ x).a ^ 2 * (fam U β μ hβ x).ρ *
      (fam U β μ hβ x).b k‖ ≤ Cb ∧ ‖(fam U β μ hβ x).mirK‖ ≤ CK := by
  have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
  have hρ := continuousAt_fam_ρ (hβ := hβ) hU hb hm
  obtain ⟨-, hp₁, -⟩ := continuousAt_fam_p (hβ := hβ) hU hb hm
  have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  have hr := continuousAt_fam_rhoB (hβ := hβ) (k := k) hU hb hm
  have h1 : ContinuousAt (fun x ↦ (fam U β μ hβ x).a ^ 2 * (fam U β μ hβ x).rhoB k -
      (fam U β μ hβ x).a / 2) x₀ := by fun_prop
  have h2 : ContinuousAt (fun x ↦ 2 * (fam U β μ hβ x).a ^ 3 - (fam U β μ hβ x).a ^ 2 *
      (fam U β μ hβ x).ρ * (fam U β μ hβ x).b k) x₀ := by fun_prop
  have h3 : ContinuousAt (fun x ↦ (fam U β μ hβ x).mirK) x₀ := by
    simp only [mirK]
    fun_prop
  refine ⟨‖(fam U β μ hβ x₀).a ^ 2 * (fam U β μ hβ x₀).rhoB k - (fam U β μ hβ x₀).a / 2‖ + 1,
    ‖2 * (fam U β μ hβ x₀).a ^ 3 - (fam U β μ hβ x₀).a ^ 2 * (fam U β μ hβ x₀).ρ *
      (fam U β μ hβ x₀).b k‖ + 1, ‖(fam U β μ hβ x₀).mirK‖ + 1, ?_⟩
  filter_upwards [h1.norm.eventually (gt_mem_nhds (lt_add_one _)),
    h2.norm.eventually (gt_mem_nhds (lt_add_one _)),
    h3.norm.eventually (gt_mem_nhds (lt_add_one _))] with x e1 e2 e3
  exact ⟨e1.le, e2.le, e3.le⟩

/-- **The bounds of item 6(a) for `2χ_s` on `z'ₙ`, uniform near `x₀`**. -/
theorem exists_eventually_chiS_bounds : ∃ c₁ c₂ : ℝ, ∀ᶠ x in 𝓝 x₀,
    (∀ n m, n ≠ m → ‖(fam U β μ hβ x).chiS₂ k n m‖ ≤
      c₁ / (‖(fam U β μ hβ x).zOp k n‖ ^ 2 * ‖(fam U β μ hβ x).zOp k m‖ ^ 2)) ∧
    ∀ n, ‖(fam U β μ hβ x).chiS₂ k n n + ((fam U β μ hβ x).β : ℂ) /
      (fam U β μ hβ x).zOp k n ^ 2‖ ≤ c₂ / ‖(fam U β μ hβ x).zOp k n‖ ^ 3 := by
  obtain ⟨Pp, -, hPp⟩ := exists_eventually_sq_div_t_sub (U := U) (hβ := hβ) (k := k) hb hm
    (c := fun x ↦ (fam U β μ hβ x).tp k) (by
      have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
      have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
      simp only [tp]
      fun_prop) fun x j ↦ (fam U β μ hβ x).t_sub_tp_ne k j
  obtain ⟨Pm, -, hPm⟩ := exists_eventually_sq_div_t_sub (U := U) (hβ := hβ) (k := k) hb hm
    (c := fun x ↦ (fam U β μ hβ x).tm k) (by
      have ha := continuousAt_fam_a (β := β) (μ := μ) (hβ := hβ) hU
      have hbc := continuousAt_fam_b (U := U) (hβ := hβ) (k := k) hb hm
      simp only [tm]
      fun_prop) fun x j ↦ (fam U β μ hβ x).t_sub_tm_ne k j
  obtain ⟨CP, -, hCP⟩ := exists_eventually_pow_div_calP (hβ := hβ) (k := k) hU hb hm
  obtain ⟨Cq, hCq0, hCq⟩ := exists_eventually_qDiag (hβ := hβ) (k := k) hU hb hm
  obtain ⟨Cf, hCf0, hCf⟩ := exists_eventually_singlet_fixed (hβ := hβ) (k := k) hU hb hm
  obtain ⟨Ca, Cb, CK, hC⟩ := exists_eventually_singlet_coeffs (hβ := hβ) (k := k) hU hb hm
  obtain ⟨R, M, B, Λ, Cp, hP⟩ := exists_eventually_params (hβ := hβ) (k := k) (ε := 1) hU hb hm
  refine ⟨(1 + 2 * M * R) ^ 2 * CP * CK +
      2 * (Ca * (Pp * Pp + Pm * Pm) + Cb * (R ^ 2 * Pp * Pm * Pp + R ^ 2 * Pm * Pp * Pm)),
    Cq + 2 * R * (Ca * (Pp * Pp + Pm * Pm) + Cb * (R ^ 2 * Pp * Pm * Pp + R ^ 2 * Pm * Pp * Pm)) +
      8 * M ^ 3 * Cf, ?_⟩
  filter_upwards [hPp, hPm, hCP, hCq, hCf, hC, hP] with x ePp ePm eCP eCq eCf ⟨eCa, eCb, eCK⟩
    ⟨hR, hM, hMπ, _hB, _hΛ, _hΛρ, _hp⟩
  set A := fam U β μ hβ x
  refine ⟨fun n m hnm ↦ ?_, fun n ↦ ?_⟩
  · have h := A.norm_chiS_le_all_of k hR hM ePp ePm eCa eCb eCP eCK n m hnm
    have h1 := norm_pos_iff.mpr (A.zOp_ne_zero k n)
    have h2 := norm_pos_iff.mpr (A.zOp_ne_zero k m)
    rw [chiS₂, le_div_iff₀ (by positivity)]
    linarith
  · have h := A.norm_chiS_diag_le_all_of k hR hM hMπ ePp ePm eCa eCb eCq eCf hCq0 hCf0 n
    have h1 := norm_pos_iff.mpr (A.zOp_ne_zero k n)
    rw [chiS₂, le_div_iff₀ (by positivity)]
    linarith

/-- **The hypotheses of `ChiContinuity.lean`** for the singlet channel on `z'ₙ`. -/
theorem contHyp_S :
    OpLattice.ContHyp (fun x ↦ (fam U β μ hβ x).opLatticeAll k)
      (fun x ↦ (fam U β μ hβ x).chiS₂ k) (fun x ↦ (fam U β μ hβ x).Bpp₂ k) x₀ := by
  have hβ₀ := hβ x₀
  have hz := continuousAt_fam_zOp (U := U) (hβ := hβ) (k := k) hb hm
  have hχ := continuousAt_fam_chiS (hβ := hβ) (k := k) hU hb hm
  have hβc := continuousAt_fam_β (U := U) (μ := μ) (hβ := hβ) hb
  obtain ⟨c₁, c₂, hc⟩ := exists_eventually_chiS_bounds (hβ := hβ) (k := k) hU hb hm
  refine ⟨Filter.Eventually.of_forall fun x ↦ (fam U β μ hβ x).chiDecay_chiS k,
    Filter.Eventually.of_forall fun x ↦ (fam U β μ hβ x).bubbleDecay_Bpp k, hb, hz,
    fun n ↦ continuousAt_const.mul (continuousAt_fam_Bpp (hβ := hβ) k hU hb hm n), fun n m ↦ ?_,
    OpLattice.dom_of_uniform (c := 1 / 2) (c₁ := c₁) (c₂ := c₂) (by norm_num)
      (hc.mono fun x hx ↦ hx.1) (hc.mono fun x hx ↦ hx.2) hb
      (eventually_norm_zOp_sq_ge (U := U) (hβ := hβ) (k := k) hb hm)⟩
  have hzn := hz n
  have hzm := hz m
  have hχnm := hχ n m
  have hK : ContinuousAt (fun x ↦ -((fam U β μ hβ x).zOp k n * (fam U β μ hβ x).zOp k m /
      ((fam U β μ hβ x).β : ℂ)) * (2 * (fam U β μ hβ x).chiS k n m) - kron n m) x₀ := by
    fun_prop (disch := exact ofReal_ne_zero.mpr hβ₀.ne')
  exact hK

/-- **`Γ_s` is continuous in the parameters**: along a family `(U, β, μ)(x)` continuous at `x₀`,
if `1 + 𝒦` is invertible at `x₀`, then `Γ_s,nm` is continuous at `x₀`, for every `μ` and `k`. -/
theorem continuousAt_gammaOpS (hu : IsUnit ((fam U β μ hβ x₀).onePlusKS k)) (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).gammaOpS k n m) x₀ := by
  have hH := contHyp_S (hβ := hβ) (k := k) hU hb hm
  have h := OpLattice.continuousAt_gammaOpFam hH ((fam U β μ hβ x₀).chiDecay_chiS k) hu n m
  have h2 : ContinuousAt (fun x ↦ (2 : ℂ) * OpLattice.gammaOpFam
      (fun x ↦ (fam U β μ hβ x).opLatticeAll k) (fun x ↦ (fam U β μ hβ x).chiS₂ k)
      (fun x ↦ (fam U β μ hβ x).Bpp₂ k) x n m) x₀ := continuousAt_const.mul h
  refine h2.congr (Filter.Eventually.of_forall fun x ↦ ?_)
  dsimp only
  rw [OpLattice.gammaOpFam_eq (fun x ↦ (fam U β μ hβ x).opLatticeAll k)
    (fun x ↦ (fam U β μ hβ x).chiS₂ k) (fun x ↦ (fam U β μ hβ x).Bpp₂ k)
    ((fam U β μ hβ x).chiDecay_chiS k) ((fam U β μ hβ x).bubbleDecay_Bpp k) n m]
  rfl

/-- **Under the assumptions of Theorem "singlet"** at `x₀`, `Γ_s,nm` is continuous at `x₀`. -/
theorem continuousAt_gammaOpS_of_SingletHyp {ξ : ℂ} (h : (fam U β μ hβ x₀).SingletHyp k ξ)
    (n m : ℤ) : ContinuousAt (fun x ↦ (fam U β μ hβ x).gammaOpS k n m) x₀ :=
  continuousAt_gammaOpS k hU hb hm h.isUnit_onePlusKS n m

end Family

end Params

end HubbardAtom
