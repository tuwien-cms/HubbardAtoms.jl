import HubbardAtomLean.OpLatticeAll
import HubbardAtomLean.SingletZero

/-!
# Bounds on the entries of `χ_s`

Item 6(a) for the singlet channel: for every `μ`, the entries of `2χ_s` on the lattice `z'ₙ` of
`OpLatticeAll.lean` (`zₙ = μ + i(νₙ + ω_k/2)`, with the mirror fixed point `zₙ = μ` moved) decay
as those of the bubble `diag(-β/z'ₙ²)`:

* `Params.norm_chiS_le`: `|2χ_s,nm| ≤ c|z'ₙ|⁻²|z'ₘ|⁻²` for `n ≠ m`;
* `Params.norm_chiS_diag_le`: `|2χ_s,nn + β/z'ₙ²| ≤ c|z'ₙ|⁻³`;
* `Params.chiDecay_chiS`, `Params.bubbleDecay_Bpp`: the bounds of `ChiOperator.lean`, for
  `X = 2χ_s` (`chiS₂`) and for `b = -2B` (`Bpp₂`).

By Proposition "pp" (`chiS_eq`, `chiS_eq_zero`), `χ_s = L_s 𝓜₊ - B 𝓜₋ + χ_s,reg`. No split into
mirror subspaces is needed: the mirror entries `(L_s + B)/2 = βa²(1 - 8p₁ - ρ²)/(4𝒫(tₙ))` decay as
`|zₙ|⁻⁴`, and `z_{n̄} = 2μ - zₙ`. With `tₙ = zₙ²T(1/zₙ)` and `𝒫(tₙ) = zₙ⁴𝒫̃(1/zₙ)`, every other
factor is, for all `n` but the mirror fixed point, a function of `1/z'ₙ` continuous at `0`
(`bdd_of_inv_zOp`).
-/

open Complex Filter Topology

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ)

/-- `T(s) = (b²/4 - μ²)s² + 2μs - 1`, so that `tₙ = zₙ²T(1/zₙ)`. -/
noncomputable def tS (s : ℂ) : ℂ := (A.b k ^ 2 / 4 - A.μ ^ 2) * s ^ 2 + 2 * A.μ * s - 1

/-- `𝒫̃(s) = (T(s) + a²s²)² - a²b²s⁴`, so that `𝒫(tₙ) = zₙ⁴𝒫̃(1/zₙ)`. -/
noncomputable def calPS (s : ℂ) : ℂ :=
  (A.tS k s + A.a ^ 2 * s ^ 2) ^ 2 - A.a ^ 2 * A.b k ^ 2 * s ^ 4

theorem t_eq_tS {n : ℤ} (hz : A.z k n ≠ 0) : A.t k n = A.z k n ^ 2 * A.tS k (A.z k n)⁻¹ := by
  rw [A.t_eq_z, tS]
  field_simp
  ring

theorem calPS_inv {n : ℤ} (hz : A.z k n ≠ 0) :
    A.calPS k (A.z k n)⁻¹ = A.calP k (A.t k n) / A.z k n ^ 4 := by
  rw [calPS, A.t_eq_z, tS, calP]
  field_simp
  ring

theorem continuousAt_inv_calPS : ContinuousAt (fun s ↦ (A.calPS k s)⁻¹) 0 :=
  ContinuousAt.inv₀ (by unfold calPS tS; fun_prop) (by simp [calPS, tS])

/-- Off the mirror fixed point, `z'ₙ = zₙ ≠ 0`. -/
theorem eventually_zOp_eq_ne : ∀ᶠ n in cofinite, mirror k n ≠ n ∧ A.zOp k n = A.z k n ∧
    A.z k n ≠ 0 := by
  filter_upwards [eventually_cofinite_ne ((-k - 1) / 2)] with n hn
  have h : mirror k n ≠ n := by unfold mirror; omega
  exact ⟨h, A.zOp_of_ne k h, A.zOp_of_ne k h ▸ A.zOp_ne_zero k n⟩

/-- `z'ₙ⁴/𝒫(tₙ)` is bounded. -/
theorem bdd_zOp_pow_div_calP : ∃ C, ∀ n, ‖A.zOp k n ^ 4 / A.calP k (A.t k n)‖ ≤ C :=
  A.bdd_of_inv_zOp k _ (A.continuousAt_inv_calPS k) <| (A.eventually_zOp_eq_ne k).mono
    fun n ⟨_, h, hz⟩ ↦ by rw [h, A.calPS_inv k hz, inv_div]

/-- `z'ₙ²/(tₙ - t₀)` is bounded. -/
theorem bdd_zOp_sq_div_t_sub (t₀ : ℂ) : ∃ C, ∀ n, ‖A.zOp k n ^ 2 / (A.t k n - t₀)‖ ≤ C :=
  A.bdd_of_inv_zOp k (fun s ↦ (A.tS k s - t₀ * s ^ 2)⁻¹)
    (ContinuousAt.inv₀ (by unfold tS; fun_prop) (by simp [tS])) <|
    (A.eventually_zOp_eq_ne k).mono fun n ⟨_, h, hz⟩ ↦ by
      rw [h, A.t_eq_tS k hz, ← inv_div]
      congr 1
      field_simp

/-- `1/z'ₙ` is bounded. -/
theorem bdd_inv_zOp : ∃ C, ∀ n, ‖(A.zOp k n)⁻¹‖ ≤ C :=
  A.bdd_of_inv_zOp k id continuousAt_id (Eventually.of_forall fun _ ↦ rfl)

/-- `z_{n̄} = 2μ - zₙ`. -/
theorem z_mirror (n : ℤ) : A.z k (mirror k n) = 2 * A.μ - A.z k n := by
  rw [z, z, A.x_mirror, b]
  ring

/-- `z'_{n̄}/z'ₙ` is bounded: off the fixed point it is `2μ/zₙ - 1`. -/
theorem bdd_zOp_mirror_div : ∃ C, ∀ n, ‖A.zOp k (mirror k n) / A.zOp k n‖ ≤ C :=
  A.bdd_of_inv_zOp k (fun s ↦ 2 * A.μ * s - 1) (by fun_prop) <|
    (A.eventually_zOp_eq_ne k).mono fun n ⟨hn, h, hz⟩ ↦ by
      have hm : mirror k (mirror k n) ≠ mirror k n := by rw [mirror_mirror]; exact Ne.symm hn
      rw [A.zOp_of_ne k hm, h, A.z_mirror]
      field_simp

/-- The regular part of `χ_s`: `χ_s,reg` of Eq. (chi-s-reg) for `b ≠ 0`, the partial fractions
of Eq. (chi-s-num) at `b = 0`. -/
noncomputable def chiRegS (n m : ℤ) : ℂ := if A.b k = 0 then A.chiReg₀ k n m else A.chiReg k n m

/-- **Proposition "pp"**, singlet part, for all `b`: `χ_s = L_s 𝓜₊ - B 𝓜₋ + χ_s,reg`. -/
theorem chiS_eq_regS (n m : ℤ) :
    A.chiS k n m = mirrorBlock k (fun j ↦ A.Ls k (A.t k j)) (fun j ↦ -A.Bpp k j) n m +
      A.chiRegS k n m := by
  unfold chiRegS
  split_ifs with hb
  exacts [A.chiS_eq_zero k hb n m, A.chiS_eq k hb n m]

/-- `z'ₙ²z'ₘ²χ_s,reg,nm` is bounded. -/
theorem bdd_zOp_sq_mul_chiRegS :
    ∃ C, ∀ n m, ‖A.zOp k n ^ 2 * A.zOp k m ^ 2 * A.chiRegS k n m‖ ≤ C := by
  obtain ⟨Cp, hCp⟩ := A.bdd_zOp_sq_div_t_sub k (A.tp k)
  obtain ⟨Cm, hCm⟩ := A.bdd_zOp_sq_div_t_sub k (A.tm k)
  obtain ⟨Ci, hCi⟩ := A.bdd_inv_zOp k
  -- `gₙ = z'ₙ⁻² (z'ₙ² gₙ)` and `z'ₙ²gₙ` are bounded, for `g = 1/(t - t±)`
  set p : ℤ → ℂ := fun n ↦ A.zOp k n ^ 2 / (A.t k n - A.tp k)
  set q : ℤ → ℂ := fun n ↦ A.zOp k n ^ 2 / (A.t k n - A.tm k)
  set i : ℤ → ℂ := fun n ↦ (A.zOp k n)⁻¹
  have hz := A.zOp_ne_zero k
  have hp : ∀ n, 1 / (A.t k n - A.tp k) = i n ^ 2 * p n := fun n ↦ by
    have := hz n; simp only [i, p]; field_simp
  -- the bound is a polynomial in the bounds of `p`, `q`, `i`
  have hCp0 := (norm_nonneg _).trans (hCp 0)
  have hCm0 := (norm_nonneg _).trans (hCm 0)
  have hCi0 := (norm_nonneg _).trans (hCi 0)
  refine ⟨‖(2 * A.a ^ 2 * A.β * A.p₀ - A.a : ℂ)‖ * (Cp * Cp) + ‖(2 * A.a ^ 3 : ℂ)‖ *
    (2 * (Ci ^ 2 * Cp ^ 3)) +
    (‖A.γp k‖ * (Cp * Cp) + ‖A.γm k‖ * (Cm * Cm)), fun n m ↦ ?_⟩
  have hpn := hCp n; have hpm := hCp m
  have hqn := hCm n; have hqm := hCm m
  have hin := hCi n; have him := hCi m
  unfold chiRegS
  split_ifs with hb0
  · have h : A.zOp k n ^ 2 * A.zOp k m ^ 2 * A.chiReg₀ k n m =
        (2 * A.a ^ 2 * A.β * A.p₀ - A.a) * (p n * p m) +
          2 * A.a ^ 3 * (i n ^ 2 * p n ^ 2 * p m + i m ^ 2 * p n * p m ^ 2) := by
      rw [chiReg₀, g₀, g₀, hp, hp]
      simp only [i, p]
      field_simp
    rw [h]
    refine ((norm_add_le _ _).trans (add_le_add ?_ ?_)).trans (le_add_of_nonneg_right
      (by positivity))
    · rw [norm_mul, norm_mul]
      gcongr
    · rw [norm_mul]
      gcongr
      refine (norm_add_le _ _).trans ?_
      simp only [norm_mul, norm_pow]
      calc ‖i n‖ ^ 2 * ‖p n‖ ^ 2 * ‖p m‖ + ‖i m‖ ^ 2 * ‖p n‖ * ‖p m‖ ^ 2
          ≤ Ci ^ 2 * Cp ^ 2 * Cp + Ci ^ 2 * Cp * Cp ^ 2 := by gcongr
        _ = 2 * (Ci ^ 2 * Cp ^ 3) := by ring
  · have h : A.zOp k n ^ 2 * A.zOp k m ^ 2 * A.chiReg k n m =
        A.γp k * (p n * p m) + A.γm k * (q n * q m) := by
      rw [chiReg]
      simp only [p, q]
      ring
    rw [h]
    refine ((norm_add_le _ _).trans (add_le_add ?_ ?_)).trans (le_add_of_nonneg_left
      (by positivity)) <;> rw [norm_mul, norm_mul] <;> gcongr

/-- A bound on `zₙ²zₘ²x` is a bound `|x| ≤ C|zₙ|⁻²|zₘ|⁻²`. -/
private theorem norm_le_div_of_norm_mul_le {x z v : ℂ} {C : ℝ} (hz : z ≠ 0) (hv : v ≠ 0)
    (h : ‖z ^ 2 * v ^ 2 * x‖ ≤ C) : ‖x‖ ≤ C / (‖z‖ ^ 2 * ‖v‖ ^ 2) := by
  rw [norm_mul, norm_mul, norm_pow, norm_pow] at h
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- **Off-diagonal entries**: `|2χ_s,nm| ≤ c|z'ₙ|⁻²|z'ₘ|⁻²` for `n ≠ m`. -/
theorem norm_chiS_le :
    ∃ c, ∀ n m, n ≠ m → ‖2 * A.chiS k n m‖ ≤ c / (‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2) := by
  obtain ⟨CP, hCP⟩ := A.bdd_zOp_pow_div_calP k
  obtain ⟨CR, hCR⟩ := A.bdd_zOp_sq_mul_chiRegS k
  obtain ⟨CM, hCM⟩ := A.bdd_zOp_mirror_div k
  have hCP0 := (norm_nonneg _).trans (hCP 0)
  set K : ℂ := A.β * A.a ^ 2 * (1 - 8 * A.p₁ - A.ρ ^ 2) / 2
  refine ⟨CM ^ 2 * CP * ‖K‖ + 2 * CR, fun n m hnm ↦
    norm_le_div_of_norm_mul_le (A.zOp_ne_zero k n) (A.zOp_ne_zero k m) ?_⟩
  rw [A.chiS_eq_regS k]
  by_cases hm : m = mirror k n
  · subst hm
    rw [mirrorBlock_mirror hnm, mul_add, mul_add]
    have h : A.zOp k n ^ 2 * A.zOp k (mirror k n) ^ 2 *
        (2 * ((A.Ls k (A.t k n) - -A.Bpp k n) / 2)) =
        (A.zOp k (mirror k n) / A.zOp k n) ^ 2 * (A.zOp k n ^ 4 / A.calP k (A.t k n)) * K := by
      have hz := A.zOp_ne_zero k n
      have hP := A.calP_ne_zero k n
      rw [sub_neg_eq_add, A.Ls_add_Bpp]
      simp only [K]
      field_simp
    refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
    · rw [h, norm_mul, norm_mul, norm_pow]
      have : ‖A.zOp k (mirror k n) / A.zOp k n‖ ^ 2 ≤ CM ^ 2 := by
        have := hCM n
        gcongr
      exact mul_le_mul_of_nonneg_right (mul_le_mul this (hCP n) (norm_nonneg _) (by positivity))
        (norm_nonneg _)
    · rw [mul_left_comm, norm_mul, Complex.norm_two]
      exact mul_le_mul_of_nonneg_left (hCR _ _) zero_le_two
  · rw [mirrorBlock_of_ne hnm hm, zero_add, mul_left_comm, norm_mul, Complex.norm_two]
    have := hCR n m
    have : 0 ≤ CM ^ 2 * CP * ‖K‖ := by positivity
    linarith

/-- A sequence that agrees with `z'ₙ³[β(tₙ + c/2)/𝒫(tₙ) + β/z'ₙ²]` for all but finitely many `n`
is bounded: off the fixed point that is a function of `1/zₙ` continuous at `0`, as
`T(0) + 𝒫̃(0) = 0`. The diagonal of `2χ_s` and the bubble have this form. -/
theorem bdd_of_diag (c : ℂ) {f : ℤ → ℂ} (hf : ∀ᶠ n in cofinite, f n = A.zOp k n ^ 3 *
    (A.β * (A.t k n + c / 2) / A.calP k (A.t k n) + A.β / A.zOp k n ^ 2)) :
    ∃ C, ∀ n, ‖f n‖ ≤ C :=
  A.bdd_of_inv_zOp k (fun s ↦ A.β * (A.tS k s * ((A.b k ^ 2 / 4 - A.μ ^ 2) * s + 2 * A.μ) +
      2 * A.a ^ 2 * s * A.tS k s + (A.a ^ 4 - A.a ^ 2 * A.b k ^ 2) * s ^ 3 + c * s / 2) *
      (A.calPS k s)⁻¹)
    (by unfold tS; exact ContinuousAt.mul (by fun_prop) (A.continuousAt_inv_calPS k)) <|
    (hf.and (A.eventually_zOp_eq_ne k)).mono fun n ⟨hfn, _, h, hz⟩ ↦ by
      have hP := A.calP_ne_zero k n
      rw [hfn, h, A.calPS_inv k hz, tS]
      field_simp
      rw [calP, A.t_eq_z]
      ring

/-- **Diagonal entries**: `|2χ_s,nn + β/z'ₙ²| ≤ c|z'ₙ|⁻³`, i.e.
`2χ_s,nn = -βz'ₙ⁻²[1 + O(|z'ₙ|⁻¹)]`. Off the mirror fixed point
`2χ_s,nn = L_s - B + 2χ_s,reg,nn`. -/
theorem norm_chiS_diag_le :
    ∃ c, ∀ n, ‖2 * A.chiS k n n + A.β / A.zOp k n ^ 2‖ ≤ c / ‖A.zOp k n‖ ^ 3 := by
  obtain ⟨C₁, hC₁⟩ := A.bdd_of_diag k (A.As k + A.a * A.ρ * A.b k + A.a ^ 2 * A.ρ ^ 2)
    (f := fun n ↦ A.zOp k n ^ 3 * (2 * (A.chiS k n n - A.chiRegS k n n) + A.β / A.zOp k n ^ 2))
    ((A.eventually_zOp_eq_ne k).mono fun n ⟨hn, _⟩ ↦ by
      have hz := A.zOp_ne_zero k n
      have hP := A.calP_ne_zero k n
      rw [A.chiS_eq_regS, add_sub_cancel_right, mirrorBlock_self (Ne.symm hn), Ls,
        A.Bpp_eq_calG_div, calG]
      field_simp
      ring)
  obtain ⟨CR, hCR⟩ := A.bdd_zOp_sq_mul_chiRegS k
  obtain ⟨Ci, hCi⟩ := A.bdd_inv_zOp k
  have hCi0 := (norm_nonneg _).trans (hCi 0)
  refine ⟨C₁ + 2 * Ci * CR, fun n ↦ ?_⟩
  have hz := A.zOp_ne_zero k n
  have hz0 := norm_pos_iff.mpr hz
  rw [le_div_iff₀ (by positivity), ← norm_pow, mul_comm, ← norm_mul]
  have hreg : ‖A.zOp k n ^ 3 * (2 * A.chiRegS k n n)‖ ≤ 2 * Ci * CR := by
    have h : A.zOp k n ^ 3 * (2 * A.chiRegS k n n) =
        2 * (A.zOp k n)⁻¹ * (A.zOp k n ^ 2 * A.zOp k n ^ 2 * A.chiRegS k n n) := by
      field_simp
    rw [h, norm_mul, norm_mul, Complex.norm_two]
    gcongr
    exacts [hCi n, hCR n n]
  have h : A.zOp k n ^ 3 * (2 * A.chiS k n n + A.β / A.zOp k n ^ 2) =
      A.zOp k n ^ 3 * (2 * (A.chiS k n n - A.chiRegS k n n) + A.β / A.zOp k n ^ 2) +
        A.zOp k n ^ 3 * (2 * A.chiRegS k n n) := by ring
  rw [h]
  exact (norm_add_le _ _).trans (add_le_add (hC₁ n) hreg)

/-- `2χ_s`, the matrix handed to `ChiOperator.lean`. -/
noncomputable def chiS₂ (n m : ℤ) : ℂ := 2 * A.chiS k n m

/-- `-2B`, the bubble handed to `ChiOperator.lean`. -/
noncomputable def Bpp₂ (n : ℤ) : ℂ := -2 * A.Bpp k n

/-- **The bounds of item 6(a)** for `2χ_s`, as used by `ChiOperator.lean`. -/
theorem chiDecay_chiS : (A.opLatticeAll k).ChiDecay (A.chiS₂ k) :=
  ⟨A.norm_chiS_le k, A.norm_chiS_diag_le k⟩

/-- **The bubble** `-2B = β𝒢(tₙ)/𝒫(tₙ)` decays as `-β/z'ₙ²`: `z'ₙ(tₙ - 1)`, with
`tₙ = -z'ₙ²(-2Bₙ)/β`, has the form of `bdd_of_diag`. -/
theorem bubbleDecay_Bpp : (A.opLatticeAll k).BubbleDecay (A.Bpp₂ k) := by
  refine ⟨fun n ↦ mul_ne_zero (by norm_num) (A.Bpp_ne_zero k n), ?_⟩
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  -- `-βz'ₙ(tₙ - 1) = z'ₙ³[β(tₙ + aρb + a²ρ²)/𝒫(tₙ) + β/z'ₙ²]`
  obtain ⟨C, hC⟩ := A.bdd_of_diag k (2 * (A.a * A.ρ * A.b k + A.a ^ 2 * A.ρ ^ 2))
    (f := fun n ↦ -A.β * (A.zOp k n * (-(A.zOp k n ^ 2 / A.β) * (-2 * A.Bpp k n) - 1)))
    (Eventually.of_forall fun n ↦ by
      have hz := A.zOp_ne_zero k n
      have hP := A.calP_ne_zero k n
      rw [A.Bpp_eq_calG_div, calG]
      field_simp
      ring)
  refine ⟨A.β⁻¹ * C, fun n ↦ ?_⟩
  have h := hC n
  change ‖A.zOp k n * (-(A.zOp k n ^ 2 / A.β) * (-2 * A.Bpp k n) - 1)‖ ≤ _
  rw [norm_mul, norm_neg, Complex.norm_real, Real.norm_of_nonneg A.β_pos.le] at h
  rw [inv_mul_eq_div, le_div_iff₀ A.β_pos]
  linarith

end Params

end HubbardAtom
