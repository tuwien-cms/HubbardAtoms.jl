import HubbardAtomLean.FieldLNegQ0
import HubbardAtomLean.FieldLAxisFormula
import HubbardAtomLean.SingletLZApproach

/-!
# On the axes, distinct roots of `E₈` and roots off the lattices fail only at isolated points

`LAxisGeneric` (`FieldLAxisGeneric.lean`) asks, at an axis point, for four roots `u_j` of `E₈` with
distinct `u_j²` (`sep`), off both spin lattices (`off`), and for the `K` conditions. Here, on each
axis `h = 0`, `μ = 0`, `μ = h`, `μ = -h`, at fixed `U ≠ 0`, `β`, `k`, `sep` and `off` are shown to
fail only at isolated points: near every axis point `s₀`, they hold at every other point. So there
`LAxisGeneric` reduces to the `K` conditions, which are not shown. Not in the note.

* `quarticDisc`, `quarticDisc_prod`: the discriminant of a monic quartic, `∏_{i<j}(sᵢ - sⱼ)²` at
  its roots (one `ring` identity); `quartic_coeffs`, `quartic_ne_zero_of_norm` (a root bound),
  `exists_quartic_eq_prod` (a quartic over `ℂ` splits).
* `FParams.discE8`: the discriminant of `E₈` in `s = v²`. Where `E₈(v) = ∏ⱼ(v² - s_j)` (such `s_j`
  exist, `exists_E8L_eq_prod`), `discE8 = ∏_{i<j}(s_i - s_j)²` (`discE8_eq_prod`), so `discE8 ≠ 0`
  gives distinct `s_j` (`sub_ne_of_discE8_ne`); `discE8_corner_ne`: at `μ = h = 0` and `U ≠ 0`,
  `discE8 ≠ 0`, as the roots `-q₀ ± λ(p₀ ± p↑)` of `E8L_corner` are distinct.
* `FParams.Axis`, `FParams.axis`: the axes as lines `s ↦ (U, β, s d_μ, s d_h)` through the corner
  `s = 0`; every axis point is one (`axis_self`). The lattice of `σ` is
  `z_l = s(d_μ ± d_h) + i(ν_l + ω_k/2)` (`axis_z`); it is fixed only for `↓` on `μ = h` and
  `↑` on `μ = -h` (`Axis.slope_eq_zero_iff`). The weights, the coefficients of `E₈`, `discE8`
  and `E₈(z_{σ,l})` are analytic in `s` (`analyticAt_axis_p`, `analyticAt_axis_E8b`,
  `analyticAt_axis_discE8`, `analyticAt_axis_E8L_z`).
* `FParams.eventually_discE8_ne_axis`: **for every `s₀`, `discE8 ≠ 0` at every `s ≠ s₀` near
  `s₀`**, by isolated zeros (`eventually_ne_zero_of_analyticAt`) from the corner.
* `FParams.exists_E8L_ne_of_le`: a root bound for `E₈` uniform in `μ`, `h`: `E₈(v) ≠ 0` where
  `‖v‖² ≥ M`. The coefficients are continuous in `(r↑, r↓, π_S, π_P)` (`E8bx`), which lies in the
  closed unit ball, a compact set (`exists_bound_E8bx`, `norm_E8x_le`).
* `FParams.eventually_E8L_lattice_ne_axis`: **for every `s₀`, `E₈(z_{σ,l}) ≠ 0` for all `σ`, `l`
  at every `s ≠ s₀` near `s₀`**. The root bound gives `E₈(z_{σ,l}) ≠ 0` everywhere for all but
  finitely many `l`; for each other `l`, `E₈(z_{σ,l}(s))` is analytic and not identically zero
  (`eventually_axis_E8L_z_ne`): on a moving lattice it is nonzero at large `|s|` by the root
  bound (`exists_axis_E8L_z_ne_moving`); on the fixed one, as `s → ∞`, `π_S, π_P → 0` and
  `E₈(z_l) → |H_{r↑}(z_l)|²|H_{r↓}(z_l)|² > 0` with `r↑, r↓ ≠ 0` in the limit
  (`exists_axis_E8L_z_ne_fixed`, `E8bx_iy_ne`).
* `FParams.E8L_ne_iff_of_prod`: at the roots counted with multiplicity, `E₈(z) ≠ 0` iff
  `z ≠ ±u_j` for all `j`; so the lattice statement is `off` for every root.
* `FParams.eventually_sep_off_axis`, `FParams.eventually_lAxisGeneric_iff_axis`: **at every
  `s ≠ s₀` near any axis point `s₀`, the roots of `E₈` counted with multiplicity satisfy `sep` and
  `off`, and `LAxisGeneric` holds iff the `u_j²` are these roots and the `K` conditions hold**
  (`K_t(u_j) ≠ 0` or `K_{t,-t}(u_j) ≠ 0` at each non-atomic root, for each column spin `t`).

The exceptional points of `sep` and `off` thus do not accumulate anywhere on an axis; at the
corner `s₀ = 0` this says that `sep` and `off` hold at every axis point near the corner other than
the corner itself (where `sep` holds too, `discE8_corner_ne`, and `LAxisGeneric` never holds,
`not_lAxisGeneric_corner`). `U ≠ 0` is needed: at `U = 0`, `E₈ = (s + q₀)⁴`. Not shown here: the
`K` conditions (near the corner `FieldLAxisCornerK.lean`, outside isolated points of an axis where
they hold at one point `FieldLAxisKIsolated.lean`), the exceptional points themselves, and
statements off the axes, with `U` or `β` varying, or uniformly in `k`.
-/

open Complex Filter Topology

namespace HubbardAtom

/-! ### The discriminant of a monic quartic -/

/-- The discriminant of the monic quartic `s⁴ + b₃s³ + b₂s² + b₁s + b₀`. -/
noncomputable def quarticDisc (b₃ b₂ b₁ b₀ : ℂ) : ℂ :=
  256 * b₀ ^ 3 - 192 * b₃ * b₁ * b₀ ^ 2 - 128 * b₂ ^ 2 * b₀ ^ 2 + 144 * b₂ * b₁ ^ 2 * b₀ -
    27 * b₁ ^ 4 + 144 * b₃ ^ 2 * b₂ * b₀ ^ 2 - 6 * b₃ ^ 2 * b₁ ^ 2 * b₀ -
    80 * b₃ * b₂ ^ 2 * b₁ * b₀ + 18 * b₃ * b₂ * b₁ ^ 3 + 16 * b₂ ^ 4 * b₀ - 4 * b₂ ^ 3 * b₁ ^ 2 -
    27 * b₃ ^ 4 * b₀ ^ 2 + 18 * b₃ ^ 3 * b₂ * b₁ * b₀ - 4 * b₃ ^ 3 * b₁ ^ 3 -
    4 * b₃ ^ 2 * b₂ ^ 3 * b₀ + b₃ ^ 2 * b₂ ^ 2 * b₁ ^ 2

/-- **The discriminant at the roots**: for `∏ᵢ(s - sᵢ)`, whose coefficients are the elementary
symmetric functions of the `sᵢ`, `quarticDisc` is `∏_{i<j}(sᵢ - sⱼ)²`. -/
theorem quarticDisc_prod (s₀ s₁ s₂ s₃ : ℂ) :
    quarticDisc (-(s₀ + s₁ + s₂ + s₃))
        (s₀ * s₁ + s₀ * s₂ + s₀ * s₃ + s₁ * s₂ + s₁ * s₃ + s₂ * s₃)
        (-(s₀ * s₁ * s₂ + s₀ * s₁ * s₃ + s₀ * s₂ * s₃ + s₁ * s₂ * s₃)) (s₀ * s₁ * s₂ * s₃) =
      ((s₀ - s₁) * (s₀ - s₂) * (s₀ - s₃) * (s₁ - s₂) * (s₁ - s₃) * (s₂ - s₃)) ^ 2 := by
  unfold quarticDisc
  ring

/-- The coefficients of a monic quartic `∏ᵢ(s - sᵢ)` are the elementary symmetric functions of
the `sᵢ`, from its values at `s = 0, ±1, 2`. -/
theorem quartic_coeffs {b₃ b₂ b₁ b₀ s₀ s₁ s₂ s₃ : ℂ}
    (h : ∀ y, y ^ 4 + b₃ * y ^ 3 + b₂ * y ^ 2 + b₁ * y + b₀ =
      (y - s₀) * (y - s₁) * (y - s₂) * (y - s₃)) :
    b₃ = -(s₀ + s₁ + s₂ + s₃) ∧
      b₂ = s₀ * s₁ + s₀ * s₂ + s₀ * s₃ + s₁ * s₂ + s₁ * s₃ + s₂ * s₃ ∧
      b₁ = -(s₀ * s₁ * s₂ + s₀ * s₁ * s₃ + s₀ * s₂ * s₃ + s₁ * s₂ * s₃) ∧
      b₀ = s₀ * s₁ * s₂ * s₃ := by
  have h0 := h 0
  have h1 := h 1
  have hm := h (-1)
  have h2 := h 2
  refine ⟨?_, ?_, ?_, ?_⟩
  · linear_combination (h2 - 3 * h1 - hm + 3 * h0) / 6
  · linear_combination (h1 + hm) / 2 - h0
  · linear_combination (h1 - hm) / 2 - (h2 - 3 * h1 - hm + 3 * h0) / 6
  · linear_combination h0

/-- **A root bound for a monic quartic**: if `‖bᵢ‖ ≤ B` and `‖s‖ ≥ 4B + 1`, then
`s⁴ + b₃s³ + b₂s² + b₁s + b₀ ≠ 0`. -/
theorem quartic_ne_zero_of_norm {B : ℝ} {b₃ b₂ b₁ b₀ y : ℂ} (h₃ : ‖b₃‖ ≤ B) (h₂ : ‖b₂‖ ≤ B)
    (h₁ : ‖b₁‖ ≤ B) (h₀ : ‖b₀‖ ≤ B) (hy : 4 * B + 1 ≤ ‖y‖) :
    y ^ 4 + b₃ * y ^ 3 + b₂ * y ^ 2 + b₁ * y + b₀ ≠ 0 := by
  intro h
  have hB : 0 ≤ B := (norm_nonneg _).trans h₀
  set t := ‖y‖ with ht
  have ht1 : 1 ≤ t := by linarith
  have e : y ^ 4 = -(b₃ * y ^ 3 + b₂ * y ^ 2 + b₁ * y + b₀) := by linear_combination h
  have hn : t ^ 4 ≤ B * t ^ 3 + B * t ^ 2 + B * t + B := by
    have h1 := congrArg norm e
    rw [norm_neg, norm_pow] at h1
    rw [← ht] at h1
    rw [h1]
    calc ‖b₃ * y ^ 3 + b₂ * y ^ 2 + b₁ * y + b₀‖
        ≤ ‖b₃ * y ^ 3‖ + ‖b₂ * y ^ 2‖ + ‖b₁ * y‖ + ‖b₀‖ := norm_add₄_le
      _ = ‖b₃‖ * t ^ 3 + ‖b₂‖ * t ^ 2 + ‖b₁‖ * t + ‖b₀‖ := by
          simp only [norm_mul, norm_pow, ht]
      _ ≤ B * t ^ 3 + B * t ^ 2 + B * t + B := by
          gcongr
  have h2 : t ^ 2 ≤ t ^ 3 := pow_le_pow_right₀ ht1 (by norm_num)
  have h3 : t ≤ t ^ 3 := by nlinarith
  have h4 : 1 ≤ t ^ 3 := by nlinarith
  have h5 : 4 * B * t ^ 3 + t ^ 3 ≤ t ^ 4 := by nlinarith
  nlinarith

/-- **Isolated zeros on the line**: a function analytic on all of `ℝ` that is nonzero at
some point vanishes, near any `s₀`, at most at `s₀` itself. -/
theorem eventually_ne_zero_of_analyticAt {f : ℝ → ℂ} (hf : ∀ s, AnalyticAt ℝ f s) {s₁ : ℝ}
    (h₁ : f s₁ ≠ 0) (s₀ : ℝ) : ∀ᶠ s in 𝓝[≠] s₀, f s ≠ 0 := by
  by_contra hne
  simp only [Filter.not_eventually, not_not] at hne
  exact h₁ (AnalyticOnNhd.eqOn_zero_of_preconnected_of_frequently_eq_zero (fun s _ ↦ hf s)
    isPreconnected_univ (Set.mem_univ s₀) hne (Set.mem_univ s₁))

open Polynomial in
/-- **A monic quartic splits**: `s⁴ + b₃s³ + b₂s² + b₁s + b₀ = ∏ᵢ(s - sᵢ)` for some `sᵢ`, as `ℂ`
is algebraically closed. -/
theorem exists_quartic_eq_prod (b₃ b₂ b₁ b₀ : ℂ) :
    ∃ s : Fin 4 → ℂ, ∀ y, y ^ 4 + b₃ * y ^ 3 + b₂ * y ^ 2 + b₁ * y + b₀ = ∏ j, (y - s j) := by
  set p : ℂ[X] := X ^ 4 + C b₃ * X ^ 3 + C b₂ * X ^ 2 + C b₁ * X + C b₀ with hp
  have hm : p.Monic := by rw [hp]; monicity!
  have hd : p.natDegree = 4 := by rw [hp]; compute_degree!
  have hs : p.Splits := IsAlgClosed.splits p
  have hc : Multiset.card p.roots = 4 := by rw [← hs.natDegree_eq_card_roots, hd]
  obtain ⟨x₀, x₁, x₂, x₃, hr⟩ := Multiset.card_eq_four.1 hc
  refine ⟨![x₀, x₁, x₂, x₃], fun y ↦ ?_⟩
  have e := hs.eval_eq_prod_roots_of_monic hm y
  rw [hr] at e
  simp only [hp, eval_add, eval_mul, eval_pow, eval_C, eval_X] at e
  rw [e, Fin.prod_univ_four]
  simp [Multiset.insert_eq_cons, mul_assoc]

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The discriminant of `E₈` -/

/-- The discriminant of `E₈` as a monic quartic in `s = v²` (`E8L_eq_quartic`). Not in the
note. -/
noncomputable def discE8 : ℂ := quarticDisc (F.E8b k 3) (F.E8b k 2) (F.E8b k 1) (F.E8b k 0)

/-- **The discriminant at the roots**: if `E₈(v) = ∏ⱼ(v² - s_j)`, the `s_j` being the roots of
`E₈` in `s = v²` counted with multiplicity, then `discE8 = ∏_{i<j}(s_i - s_j)²`. -/
theorem discE8_eq_prod {s : Fin 4 → ℂ} (h : ∀ v, F.E8L k v = ∏ j, (v ^ 2 - s j)) :
    F.discE8 k = ((s 0 - s 1) * (s 0 - s 2) * (s 0 - s 3) * (s 1 - s 2) * (s 1 - s 3) *
      (s 2 - s 3)) ^ 2 := by
  have hq : ∀ y : ℂ, y ^ 4 + F.E8b k 3 * y ^ 3 + F.E8b k 2 * y ^ 2 + F.E8b k 1 * y +
      F.E8b k 0 = (y - s 0) * (y - s 1) * (y - s 2) * (y - s 3) := by
    intro y
    have hy : (y ^ (2⁻¹ : ℂ)) ^ 2 = y := by simp
    have e := h (y ^ (2⁻¹ : ℂ))
    rwa [E8L_eq_quartic, hy, Fin.prod_univ_four] at e
  obtain ⟨h3, h2, h1, h0⟩ := quartic_coeffs hq
  rw [discE8, h3, h2, h1, h0, quarticDisc_prod]

/-- **The four roots of `E₈`**, counted with multiplicity, exist at every point:
`E₈(v) = ∏ⱼ(v² - u_j²)` for some `u` (`exists_quartic_eq_prod`, `u_j = s_j^{1/2}`). -/
theorem exists_E8L_eq_prod : ∃ u : Fin 4 → ℂ, ∀ v, F.E8L k v = ∏ j, (v ^ 2 - u j ^ 2) := by
  obtain ⟨s, hs⟩ := exists_quartic_eq_prod (F.E8b k 3) (F.E8b k 2) (F.E8b k 1) (F.E8b k 0)
  refine ⟨fun j ↦ s j ^ (2⁻¹ : ℂ), fun v ↦ ?_⟩
  have e : ∀ j, (s j ^ (2⁻¹ : ℂ)) ^ 2 = s j := fun j ↦ by simp
  simp only [e]
  rw [E8L_eq_quartic, hs]

/-- **Off the roots is `E₈ ≠ 0`**: if `E₈(v) = ∏ⱼ(v² - u_j²)`, then `E₈(z) ≠ 0` iff `z ≠ ±u_j`
for all `j`. So `E₈ ≠ 0` on both lattices is `off` for every root. -/
theorem E8L_ne_iff_of_prod {u : Fin 4 → ℂ} (hu : ∀ v, F.E8L k v = ∏ j, (v ^ 2 - u j ^ 2))
    (z : ℂ) : F.E8L k z ≠ 0 ↔ ∀ j, z ≠ u j ∧ z ≠ -u j := by
  rw [hu, Ne, Finset.prod_eq_zero_iff]
  simp only [Finset.mem_univ, true_and, not_exists]
  refine forall_congr' fun j ↦ ?_
  rw [sub_eq_zero, sq_eq_sq_iff_eq_or_eq_neg, not_or]

/-- **A nonzero discriminant separates the roots**: if `E₈(v) = ∏ⱼ(v² - s_j)` and
`discE8 ≠ 0`, the `s_j` are pairwise distinct. -/
theorem sub_ne_of_discE8_ne {s : Fin 4 → ℂ} (h : ∀ v, F.E8L k v = ∏ j, (v ^ 2 - s j))
    (hd : F.discE8 k ≠ 0) : ∀ i j, i ≠ j → s i - s j ≠ 0 := by
  rw [F.discE8_eq_prod k h] at hd
  intro i j hij hs
  apply hd
  have e : s i = s j := sub_eq_zero.1 hs
  fin_cases i <;> fin_cases j <;>
    simp only [Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk] at e hij ⊢ <;>
    first | exact absurd rfl hij | (rw [e]; ring)

/-- **The discriminant at the corner** `μ = h = 0`, for `U ≠ 0`: there
`E₈(v) = ((v² + q₀)² - λ²(p₀ + p↑)²)((v² + q₀)² - λ²(p₀ - p↑)²)` (`E8L_corner`), whose roots
`s = -q₀ ± λ(p₀ ± p↑)` in `s = v²` are pairwise distinct. -/
theorem discE8_corner_ne (hμ : F.μ = 0) (hh : F.h = 0) (ha : F.a ≠ 0) : F.discE8 k ≠ 0 := by
  set A : ℂ := F.p₀ + F.pUp with hA
  set B : ℂ := F.p₀ - F.pUp with hB
  set s : Fin 4 → ℂ := ![-F.q₀L k + F.lamL * A, -F.q₀L k - F.lamL * A, -F.q₀L k + F.lamL * B,
    -F.q₀L k - F.lamL * B] with hs
  have hprod : ∀ v, F.E8L k v = ∏ j, (v ^ 2 - s j) := by
    intro v
    rw [E8L_corner hμ hh, Fin.prod_univ_four]
    simp only [hs, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.cons_val_three, Matrix.head_cons, Matrix.tail_cons, hA, hB]
    ring
  rw [F.discE8_eq_prod k hprod]
  have e : (s 0 - s 1) * (s 0 - s 2) * (s 0 - s 3) * (s 1 - s 2) * (s 1 - s 3) * (s 2 - s 3) =
      4 * F.lamL ^ 6 * A * B * (2 * F.pUp) ^ 2 * (2 * F.p₀) ^ 2 := by
    simp only [hs, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.cons_val_three, Matrix.head_cons, Matrix.tail_cons, hA, hB]
    ring
  rw [e]
  have hlam : F.lamL ≠ 0 := by
    rw [lamL]
    exact mul_ne_zero (by norm_num) (pow_ne_zero 2 (by exact_mod_cast ha))
  have hA0 : A ≠ 0 := by
    rw [hA]
    exact_mod_cast (add_pos F.p₀_pos F.pUp_pos).ne'
  have hB0 : B ≠ 0 := by
    rw [hB]
    have hp : F.p₀ - F.pUp ≠ 0 := by
      intro h0
      rw [F.pUp_eq_p₀_mul, hμ, hh, add_zero, add_zero] at h0
      have h1 : Real.exp (F.β * F.a) = 1 := by
        have h3 : F.p₀ * (1 - Real.exp (F.β * F.a)) = 0 := by linear_combination h0
        linarith [(mul_eq_zero.1 h3).resolve_left F.p₀_pos.ne']
      rw [Real.exp_eq_one_iff] at h1
      exact mul_ne_zero F.β_pos.ne' ha h1
    exact_mod_cast hp
  have hu : (2 * F.pUp : ℂ) ≠ 0 := by exact_mod_cast (mul_pos two_pos F.pUp_pos).ne'
  have h0 : (2 * F.p₀ : ℂ) ≠ 0 := by exact_mod_cast (mul_pos two_pos F.p₀_pos).ne'
  exact pow_ne_zero 2 (mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero
    (by norm_num) (pow_ne_zero 6 hlam)) hA0) hB0) (pow_ne_zero 2 hu)) (pow_ne_zero 2 h0))

/-! ### The axes through the corner -/

/-- The axes `h = 0`, `μ = 0`, `μ = h`, `μ = -h` of the `(μ, h)` plane, each a line through the
corner `μ = h = 0`. -/
inductive Axis
  /-- `h = 0`. -/
  | h0
  /-- `μ = 0`. -/
  | mu0
  /-- `μ = h`. -/
  | plus
  /-- `μ = -h`. -/
  | minus

namespace Axis

/-- The `μ` of the axis point at `s = 1`. -/
def dμ : Axis → ℝ
  | h0 => 1
  | mu0 => 0
  | plus => 1
  | minus => 1

/-- The `h` of the axis point at `s = 1`. -/
def dh : Axis → ℝ
  | h0 => 0
  | mu0 => 1
  | plus => 1
  | minus => -1

/-- The slope `d_μ ± d_h` of the center `μ_σ = μ ± h` of the lattice of the spin `σ` along the
axis. It vanishes only for `↓` on `μ = h` and `↑` on `μ = -h` (`slope_eq_zero_iff`). -/
def slope (d : Axis) (σ : Bool) : ℝ := bif σ then d.dμ + d.dh else d.dμ - d.dh

/-- The lattices that do not move: `↓` on `μ = h` and `↑` on `μ = -h`. -/
theorem slope_eq_zero_iff (d : Axis) (σ : Bool) :
    d.slope σ = 0 ↔ d = plus ∧ σ = false ∨ d = minus ∧ σ = true := by
  cases d <;> cases σ <;> norm_num [slope, dμ, dh]

end Axis

/-- The point `s` of the axis `d` at the `U`, `β` of `F`: `(U, β, μ, h) = (U, β, s d_μ, s d_h)`.
The `μ`, `h` of `F` are not used; at `s = 0` it is the corner. -/
def axis (d : Axis) (s : ℝ) : FParams := ⟨F.U, F.β, s * d.dμ, s * d.dh, F.β_pos⟩

section Axis

variable (d : Axis)

/-! The data that depend only on `U`, `β`, `k` are those of `F`. -/

@[simp] theorem axis_U (s : ℝ) : (F.axis d s).U = F.U := rfl
@[simp] theorem axis_β (s : ℝ) : (F.axis d s).β = F.β := rfl
@[simp] theorem axis_μ (s : ℝ) : (F.axis d s).μ = s * d.dμ := rfl
@[simp] theorem axis_h (s : ℝ) : (F.axis d s).h = s * d.dh := rfl
@[simp] theorem axis_a (s : ℝ) : (F.axis d s).a = F.a := rfl
@[simp] theorem axis_ν (s : ℝ) (n : ℤ) : (F.axis d s).ν n = F.ν n := rfl
@[simp] theorem axis_ω (s : ℝ) : (F.axis d s).ω k = F.ω k := rfl
@[simp] theorem axis_q₀L (s : ℝ) : (F.axis d s).q₀L k = F.q₀L k := rfl
@[simp] theorem axis_lamL (s : ℝ) : (F.axis d s).lamL = F.lamL := rfl

/-- Every point of an axis is an axis point: on `h = 0` at `s = μ`, on `μ = 0` at `s = h`, on
`μ = ±h` at `s = μ`. -/
theorem axis_self : (F.h = 0 → F.axis .h0 F.μ = F) ∧ (F.μ = 0 → F.axis .mu0 F.h = F) ∧
    (F.μ = F.h → F.axis .plus F.μ = F) ∧ (F.μ = -F.h → F.axis .minus F.μ = F) := by
  obtain ⟨U, β, μ, h, hβ⟩ := F
  refine ⟨fun e ↦ ?_, fun e ↦ ?_, fun e ↦ ?_, fun e ↦ ?_⟩ <;>
    simp only at e <;> simp [axis, Axis.dμ, Axis.dh, e]

/-- The center of the lattice of the spin `σ` moves as `s(d_μ ± d_h)`. -/
theorem axis_cL (σ : Bool) (s : ℝ) : (F.axis d s).cL σ = s * d.slope σ := by
  cases σ <;> simp only [cL, axis_μ, axis_h, Axis.slope, Bool.cond_true, Bool.cond_false] <;>
    ring

/-- The lattice of the spin `σ` at the axis point `s`: `z_l = s(d_μ ± d_h) + i(ν_l + ω_k/2)`. -/
theorem axis_z (σ : Bool) (l : ℤ) (s : ℝ) :
    ((F.axis d s).BL σ).z k l =
      ((s * d.slope σ : ℝ) : ℂ) + ((F.ν l + F.ω k / 2 : ℝ) : ℂ) * I := by
  simp only [Params.z, Params.x, Params.w, lat_μ, lat_ν, lat_ω, axis_ν, axis_ω, axis_cL]
  push_cast
  ring

/-! ### Analyticity along the axes -/

/-- The weights are analytic along the axis. -/
theorem analyticAt_axis_p (s₀ : ℝ) :
    AnalyticAt ℝ (fun s ↦ ((F.axis d s).p₀ : ℂ)) s₀ ∧
      AnalyticAt ℝ (fun s ↦ ((F.axis d s).pUp : ℂ)) s₀ ∧
      AnalyticAt ℝ (fun s ↦ ((F.axis d s).pDn : ℂ)) s₀ ∧
      AnalyticAt ℝ (fun s ↦ ((F.axis d s).p₂ : ℂ)) s₀ := by
  have hZ := (F.axis d s₀).Z_pos.ne'
  simp only [Z, boltz₀, boltzUp, boltzDn, boltz₂, axis_U, axis_β, axis_μ, axis_h] at hZ
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
  · refine Params.analyticAt_ofReal_comp ?_
    simp only [p₀, pUp, pDn, p₂, Z, boltz₀, boltzUp, boltzDn, boltz₂, axis_U, axis_β, axis_μ,
      axis_h]
    fun_prop (disch := exact hZ)

/-- The coefficients of `E₈` in `s = v²` are analytic along the axis. -/
theorem analyticAt_axis_E8b (j : Fin 4) (s₀ : ℝ) :
    AnalyticAt ℝ (fun s ↦ (F.axis d s).E8b k j) s₀ := by
  obtain ⟨h0, hu, hd, h2⟩ := F.analyticAt_axis_p d s₀
  fin_cases j <;> simp [E8b, piS, piP, rUp, rDn, ρ, mf] <;> fun_prop

/-- The discriminant of `E₈` is analytic along the axis. -/
theorem analyticAt_axis_discE8 (s₀ : ℝ) : AnalyticAt ℝ (fun s ↦ (F.axis d s).discE8 k) s₀ := by
  have h3 := F.analyticAt_axis_E8b k d 3 s₀
  have h2 := F.analyticAt_axis_E8b k d 2 s₀
  have h1 := F.analyticAt_axis_E8b k d 1 s₀
  have h0 := F.analyticAt_axis_E8b k d 0 s₀
  simp only [discE8, quarticDisc]
  fun_prop

/-- `E₈` at the lattice point `z_{σ,l}` of the axis point is analytic along the axis. -/
theorem analyticAt_axis_E8L_z (σ : Bool) (l : ℤ) (s₀ : ℝ) :
    AnalyticAt ℝ (fun s ↦ (F.axis d s).E8L k (((F.axis d s).BL σ).z k l)) s₀ := by
  have hz : AnalyticAt ℝ (fun s ↦ ((F.axis d s).BL σ).z k l) s₀ := by
    simp only [axis_z]
    exact (Params.analyticAt_ofReal_comp (by fun_prop)).add analyticAt_const
  have h3 := F.analyticAt_axis_E8b k d 3 s₀
  have h2 := F.analyticAt_axis_E8b k d 2 s₀
  have h1 := F.analyticAt_axis_E8b k d 1 s₀
  have h0 := F.analyticAt_axis_E8b k d 0 s₀
  simp only [E8L_eq_quartic]
  fun_prop

/-! ### Distinct roots -/

/-- **On each axis, the discriminant of `E₈` vanishes only at isolated points**: at fixed
`U ≠ 0`, `β`, `k`, for every `s₀`, `discE8 ≠ 0` at every `s ≠ s₀` near `s₀`. It is analytic
along the axis and nonzero at the corner `s = 0` (`discE8_corner_ne`). -/
theorem eventually_discE8_ne_axis (hU : F.U ≠ 0) (s₀ : ℝ) :
    ∀ᶠ s in 𝓝[≠] s₀, (F.axis d s).discE8 k ≠ 0 := by
  have ha : (F.axis d 0).a ≠ 0 := by
    rw [axis_a, a]
    exact div_ne_zero hU two_ne_zero
  exact eventually_ne_zero_of_analyticAt (F.analyticAt_axis_discE8 k d) (s₁ := 0)
    ((F.axis d 0).discE8_corner_ne k (by simp) (by simp) ha) s₀

/-! ### A root bound for `E₈`, uniform in `μ`, `h` -/

/-- The coefficients `E8b` of `E₈` in `s = v²` as functions of `x = (r↑, r↓, π_S, π_P)`, at given
`a`, `q₀` (`E8b_eq_E8bx`). -/
noncomputable def E8bx (a q : ℂ) (x : Fin 4 → ℂ) (j : Fin 4) : ℂ :=
  let Au := 2 * q - 4 * a ^ 2 * x 0 ^ 2
  let Ad := 2 * q - 4 * a ^ 2 * x 1 ^ 2
  let c := 4 * a ^ 2 * x 0 * x 1
  let lam := 4 * a ^ 2
  match j with
  | 3 => Au + Ad
  | 2 => Au * Ad + 2 * q ^ 2 - 2 * lam ^ 2 * (x 2 + x 3)
  | 1 => q ^ 2 * (Au + Ad) - 2 * lam ^ 2 * (2 * q * (x 2 + x 3) + c * (x 2 - x 3))
  | 0 => q ^ 4 - 2 * lam ^ 2 * q ^ 2 * (x 2 + x 3) + lam ^ 4 * (x 3 - x 2) ^ 2

/-- `E8b` is `E8bx` at `x = (r↑, r↓, π_S, π_P)`. -/
private theorem E8b_eq_E8bx (j : Fin 4) :
    F.E8b k j = E8bx F.a (F.q₀L k) ![(F.rUp : ℂ), F.rDn, F.piS, F.piP] j := by
  fin_cases j <;> rfl

/-- The coefficients `E8bx` are polynomials in `x`, so continuous. -/
private theorem continuous_E8bx (a q : ℂ) (j : Fin 4) : Continuous fun x ↦ E8bx a q x j := by
  fin_cases j <;> simp only [E8bx] <;> fun_prop

/-- On the closed unit ball the coefficients `E8bx` are bounded. -/
theorem exists_bound_E8bx (a q : ℂ) :
    ∃ B, ∀ x : Fin 4 → ℂ, ‖x‖ ≤ 1 → ∀ j, ‖E8bx a q x j‖ ≤ B := by
  have hc : Continuous fun x : Fin 4 → ℂ ↦ fun j ↦ E8bx a q x j :=
    continuous_pi fun j ↦ continuous_E8bx a q j
  obtain ⟨B, hB⟩ := (isCompact_closedBall (0 : Fin 4 → ℂ) 1).exists_bound_of_continuousOn
    hc.continuousOn
  exact ⟨B, fun x hx j ↦ (norm_le_pi_norm (fun j ↦ E8bx a q x j) j).trans
    (hB x (mem_closedBall_zero_iff.2 hx))⟩

/-- `|r↑|, |r↓|, |π_S|, |π_P| ≤ 1`. -/
theorem norm_E8x_le : ‖(![(F.rUp : ℂ), F.rDn, F.piS, F.piP] : Fin 4 → ℂ)‖ ≤ 1 := by
  refine (pi_norm_le_iff_of_nonneg zero_le_one).2 fun j ↦ ?_
  have hs := F.weights_sum
  have h0 := F.p₀_pos
  have hu := F.pUp_pos
  have hd := F.pDn_pos
  have h2 := F.p₂_pos
  fin_cases j
  · simp only [Fin.zero_eta, Matrix.cons_val_zero, Complex.norm_real, Real.norm_eq_abs, rUp, ρ,
      mf, abs_le]
    constructor <;> linarith
  · simp only [Fin.mk_one, Matrix.cons_val_one, Matrix.cons_val_zero, Complex.norm_real,
      Real.norm_eq_abs, rDn, ρ, mf, abs_le]
    constructor <;> linarith
  · simp only [Fin.reduceFinMk, Matrix.cons_val, piS, norm_mul, Complex.norm_real,
      Real.norm_eq_abs, abs_of_pos hu, abs_of_pos hd]
    nlinarith
  · simp only [Fin.reduceFinMk, Matrix.cons_val, piP, norm_mul, Complex.norm_real,
      Real.norm_eq_abs, abs_of_pos h0, abs_of_pos h2]
    nlinarith

/-- `E₈` at `G` with the `U`, `β` of `F`, through `E8bx` at the `a`, `q₀` of `F`. -/
private theorem E8L_eq_E8bx {G : FParams} (hU : G.U = F.U) (hβ : G.β = F.β) (v : ℂ) :
    G.E8L k v = (v ^ 2) ^ 4 +
      E8bx F.a (F.q₀L k) ![(G.rUp : ℂ), G.rDn, G.piS, G.piP] 3 * (v ^ 2) ^ 3 +
      E8bx F.a (F.q₀L k) ![(G.rUp : ℂ), G.rDn, G.piS, G.piP] 2 * (v ^ 2) ^ 2 +
      E8bx F.a (F.q₀L k) ![(G.rUp : ℂ), G.rDn, G.piS, G.piP] 1 * v ^ 2 +
      E8bx F.a (F.q₀L k) ![(G.rUp : ℂ), G.rDn, G.piS, G.piP] 0 := by
  have ha : G.a = F.a := by simp only [a, hU]
  have hq : G.q₀L k = F.q₀L k := by
    simp only [q₀L, wL, FParams.ω, Params.ω, lat_β, ha, hβ]
  simp only [E8L_eq_quartic, E8b_eq_E8bx, ha, hq]

/-- **A root bound for `E₈`, uniform in `μ`, `h`**: at fixed `U`, `β`, `k` there is `M` with
`E₈(v) ≠ 0` wherever `‖v‖² ≥ M`, at every `μ`, `h`. The coefficients of `E₈` in `s = v²`
are continuous in `(r↑, r↓, π_S, π_P)`, which lies in the closed unit ball (`norm_E8x_le`), so
they are bounded (`exists_bound_E8bx`), and `quartic_ne_zero_of_norm` applies. -/
theorem exists_E8L_ne_of_le :
    ∃ M : ℝ, ∀ G : FParams, G.U = F.U → G.β = F.β → ∀ v : ℂ, M ≤ ‖v‖ ^ 2 → G.E8L k v ≠ 0 := by
  obtain ⟨B, hB⟩ := exists_bound_E8bx (F.a : ℂ) (F.q₀L k)
  refine ⟨4 * B + 1, fun G hU hβ v hv ↦ ?_⟩
  rw [F.E8L_eq_E8bx k hU hβ]
  have hb := hB _ G.norm_E8x_le
  refine quartic_ne_zero_of_norm (hb 3) (hb 2) (hb 1) (hb 0) ?_
  rwa [norm_pow]

/-- **`E₈` at `v = iy` with `π_S = π_P = 0`** is `|H_{r₁}(iy)|²|H_{r₂}(iy)|²`, with
`|H_r(iy)|² = (q₀ - y²)² + 4a²r²y²`; for `a ≠ 0`, `r₁, r₂ ≠ 0` it is positive. -/
theorem E8bx_iy_ne (ha : F.a ≠ 0) {r₁ r₂ : ℝ} (h₁ : r₁ ≠ 0) (h₂ : r₂ ≠ 0) (y : ℝ) :
    (((y : ℂ) * I) ^ 2) ^ 4 +
        E8bx F.a (F.q₀L k) ![(r₁ : ℂ), r₂, 0, 0] 3 * (((y : ℂ) * I) ^ 2) ^ 3 +
        E8bx F.a (F.q₀L k) ![(r₁ : ℂ), r₂, 0, 0] 2 * (((y : ℂ) * I) ^ 2) ^ 2 +
        E8bx F.a (F.q₀L k) ![(r₁ : ℂ), r₂, 0, 0] 1 * ((y : ℂ) * I) ^ 2 +
        E8bx F.a (F.q₀L k) ![(r₁ : ℂ), r₂, 0, 0] 0 ≠ 0 := by
  set q : ℝ := F.a ^ 2 + F.ω k ^ 2 / 4 with hq
  have hq' : F.q₀L k = (q : ℂ) := by rw [q₀L_eq_omega, hq]; push_cast; ring
  have hpos : ∀ r : ℝ, r ≠ 0 → 0 < (q - y ^ 2) ^ 2 + 4 * F.a ^ 2 * r ^ 2 * y ^ 2 := by
    intro r hr
    by_cases hy : y = 0
    · have : 0 < q := by rw [hq]; positivity
      simp [hy]
      positivity
    · have : 0 < 4 * F.a ^ 2 * r ^ 2 * y ^ 2 := by positivity
      nlinarith [sq_nonneg (q - y ^ 2)]
  have e : (((y : ℂ) * I) ^ 2) ^ 4 +
        E8bx F.a (F.q₀L k) ![(r₁ : ℂ), r₂, 0, 0] 3 * (((y : ℂ) * I) ^ 2) ^ 3 +
        E8bx F.a (F.q₀L k) ![(r₁ : ℂ), r₂, 0, 0] 2 * (((y : ℂ) * I) ^ 2) ^ 2 +
        E8bx F.a (F.q₀L k) ![(r₁ : ℂ), r₂, 0, 0] 1 * ((y : ℂ) * I) ^ 2 +
        E8bx F.a (F.q₀L k) ![(r₁ : ℂ), r₂, 0, 0] 0 =
      (((q - y ^ 2) ^ 2 + 4 * F.a ^ 2 * r₁ ^ 2 * y ^ 2) *
        ((q - y ^ 2) ^ 2 + 4 * F.a ^ 2 * r₂ ^ 2 * y ^ 2) : ℝ) := by
    have hy : ((y : ℂ) * I) ^ 2 = -(y : ℂ) ^ 2 := by rw [mul_pow, I_sq]; ring
    simp only [E8bx, hq', hy, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.cons_val_three, Matrix.head_cons, Matrix.tail_cons]
    push_cast
    ring
  rw [e]
  exact_mod_cast (mul_pos (hpos r₁ h₁) (hpos r₂ h₂)).ne'

/-! ### Off the lattices -/

/-- The real part of the lattice point is the center `s(d_μ ± d_h)`, its imaginary part
`ν_l + ω_k/2`. -/
private theorem axis_z_re_im (σ : Bool) (l : ℤ) (s : ℝ) :
    (((F.axis d s).BL σ).z k l).re = s * d.slope σ ∧
      (((F.axis d s).BL σ).z k l).im = F.ν l + F.ω k / 2 := by
  rw [axis_z]
  simp

/-- On an axis with `d_μ = 1` (`h = 0`, `μ = ±h`), `μ = s` and `p₀ → 0` as `s → ∞`:
`p₀ = p₂e^{-2βs} ≤ e^{-2βs}`. -/
private theorem tendsto_axis_p₀ (hd : d.dμ = 1) :
    Tendsto (fun s ↦ (F.axis d s).p₀) atTop (𝓝 0) := by
  have hβ := F.β_pos
  have hexp : Tendsto (fun s : ℝ ↦ Real.exp (-(2 * F.β * s))) atTop (𝓝 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero.comp (tendsto_id.const_mul_atTop (by positivity))
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hexp
    (fun s ↦ (F.axis d s).p₀_pos.le) fun s ↦ ?_
  have h2 := (F.axis d s).p₂_eq_p₀_mul
  rw [axis_μ, axis_β, hd, mul_one] at h2
  have hs := (F.axis d s).weights_sum
  have hp := (F.axis d s).pUp_pos
  have hq := (F.axis d s).pDn_pos
  have hp0 := (F.axis d s).p₀_pos
  have hle : (F.axis d s).p₂ ≤ 1 := by linarith
  have e : (F.axis d s).p₀ = (F.axis d s).p₂ * Real.exp (-(2 * F.β * s)) := by
    rw [h2, mul_assoc, ← Real.exp_add, add_neg_cancel, Real.exp_zero, mul_one]
  change (F.axis d s).p₀ ≤ Real.exp (-(2 * F.β * s))
  rw [e]
  exact mul_le_of_le_one_left (Real.exp_pos _).le hle

/-- **`E₈` at `iy` is eventually nonzero** along a family `G(s)` with the `U`, `β` of `F`, as
`s → ∞`, if `(r↑, r↓, π_S, π_P) → (r₁, r₂, 0, 0)` with `r₁, r₂ ≠ 0` (`E8bx_iy_ne`). -/
private theorem exists_E8L_iy_ne_of_tendsto (ha : F.a ≠ 0) {r₁ r₂ : ℝ} (h₁ : r₁ ≠ 0) (h₂ : r₂ ≠ 0)
    (y : ℝ) {G : ℝ → FParams} (hU : ∀ s, (G s).U = F.U) (hβ : ∀ s, (G s).β = F.β)
    (hx : Tendsto (fun s ↦ (![((G s).rUp : ℂ), (G s).rDn, (G s).piS, (G s).piP] : Fin 4 → ℂ))
      atTop (𝓝 ![(r₁ : ℂ), r₂, 0, 0])) :
    ∃ s, (G s).E8L k ((y : ℂ) * I) ≠ 0 := by
  have h3 := continuous_E8bx F.a (F.q₀L k) 3
  have h2 := continuous_E8bx F.a (F.q₀L k) 2
  have h1 := continuous_E8bx F.a (F.q₀L k) 1
  have h0 := continuous_E8bx F.a (F.q₀L k) 0
  have hc : Continuous fun x ↦ (((y : ℂ) * I) ^ 2) ^ 4 +
      E8bx F.a (F.q₀L k) x 3 * (((y : ℂ) * I) ^ 2) ^ 3 +
      E8bx F.a (F.q₀L k) x 2 * (((y : ℂ) * I) ^ 2) ^ 2 +
      E8bx F.a (F.q₀L k) x 1 * ((y : ℂ) * I) ^ 2 + E8bx F.a (F.q₀L k) x 0 := by
    fun_prop
  obtain ⟨s, hs⟩ := (((hc.tendsto _).comp hx).eventually_ne (F.E8bx_iy_ne k ha h₁ h₂ y)).exists
  exact ⟨s, by rwa [F.E8L_eq_E8bx k (hU s) (hβ s)]⟩

/-- The limits of the weights give those of `(r↑, r↓, π_S, π_P)`. -/
private theorem tendsto_E8x {G : ℝ → FParams} {wu wd w₂ : ℝ}
    (h0 : Tendsto (fun s ↦ (G s).p₀) atTop (𝓝 0))
    (hu : Tendsto (fun s ↦ (G s).pUp) atTop (𝓝 wu))
    (hd : Tendsto (fun s ↦ (G s).pDn) atTop (𝓝 wd))
    (h2 : Tendsto (fun s ↦ (G s).p₂) atTop (𝓝 w₂)) :
    Tendsto (fun s ↦ (![((G s).rUp : ℂ), (G s).rDn, (G s).piS, (G s).piP] : Fin 4 → ℂ)) atTop
      (𝓝 ![((w₂ - wu + wd : ℝ) : ℂ), ((w₂ + wu - wd : ℝ) : ℂ), ((wu * wd : ℝ) : ℂ), 0]) := by
  have c : ∀ {f : ℝ → ℝ} {w : ℝ}, Tendsto f atTop (𝓝 w) →
      Tendsto (fun s ↦ (f s : ℂ)) atTop (𝓝 (w : ℂ)) :=
    fun h ↦ (continuous_ofReal.tendsto _).comp h
  refine tendsto_pi_nhds.2 fun j ↦ ?_
  fin_cases j
  · simp only [Fin.zero_eta, Matrix.cons_val_zero]
    convert c (((h2.sub h0).sub (hu.sub hd))) using 2
    · simp only [rUp, ρ, mf]
    · ring_nf
  · simp only [Fin.mk_one, Matrix.cons_val_one, Matrix.cons_val_zero]
    convert c (((h2.sub h0).add (hu.sub hd))) using 2
    · simp only [rDn, ρ, mf]
    · ring_nf
  · simp only [Fin.reduceFinMk, Matrix.cons_val]
    convert c (hu.mul hd) using 2
    simp only [piS, ofReal_mul]
  · simp only [Fin.reduceFinMk, Matrix.cons_val]
    convert c (h0.mul h2) using 2
    · simp only [piP, ofReal_mul]
    · simp

/-- **On the fixed lattice `E₈` is not identically zero**: on `μ = h` the lattice of `↓`, on
`μ = -h` that of `↑`, stays at `z_l = i(ν_l + ω_k/2)`. As `s → ∞`, `p₀ → 0`
(`tendsto_axis_p₀`), and with `E = e^{βa}`, `P = 1/(1 + E)`: on `μ = h`, `p↓ = Ep₀ → 0`,
`p₂ → P`, `p↑ = Ep₂ → EP`, so `(r↑, r↓) → ((1 - E)P, 1)` and `π_S, π_P → 0`; on `μ = -h` the
spins are exchanged. So `E₈(z_l) → |H_{r↑}(z_l)|²|H_{r↓}(z_l)|² > 0` (`E8bx_iy_ne`). -/
theorem exists_axis_E8L_z_ne_fixed (hU : F.U ≠ 0) {σ : Bool}
    (hfix : d = .plus ∧ σ = false ∨ d = .minus ∧ σ = true) (l : ℤ) :
    ∃ s, (F.axis d s).E8L k (((F.axis d s).BL σ).z k l) ≠ 0 := by
  have ha : F.a ≠ 0 := div_ne_zero hU two_ne_zero
  have hsl : d.slope σ = 0 := (d.slope_eq_zero_iff σ).2 hfix
  have hdμ : d.dμ = 1 := by rcases hfix with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> rfl
  have hz : ∀ s, ((F.axis d s).BL σ).z k l = ((F.ν l + F.ω k / 2 : ℝ) : ℂ) * I := fun s ↦ by
    rw [axis_z, hsl, mul_zero, ofReal_zero, zero_add]
  simp only [hz]
  set E : ℝ := Real.exp (F.β * F.a) with hE
  have hE0 : 0 < E := Real.exp_pos _
  have hE1 : E ≠ 1 := by
    rw [hE, Ne, Real.exp_eq_one_iff]
    exact mul_ne_zero F.β_pos.ne' ha
  set P : ℝ := (1 + E)⁻¹ with hP
  have hP0 : 0 < P := by positivity
  have h0 := F.tendsto_axis_p₀ d hdμ
  -- `p₂ = P - p₀` from `(1 + E)(p₀ + p₂) = 1`
  have hp₂ : ∀ s, (F.axis d s).pUp + (F.axis d s).pDn =
      E * ((F.axis d s).p₀ + (F.axis d s).p₂) → (F.axis d s).p₂ = P - (F.axis d s).p₀ := by
    intro s h
    have hs := (F.axis d s).weights_sum
    have e : (1 + E) * ((F.axis d s).p₀ + (F.axis d s).p₂) = 1 := by linear_combination hs - h
    have e' := eq_inv_of_mul_eq_one_right e
    rw [hP]
    linarith
  have hcU : ∀ s, (F.axis d s).U = F.U := fun _ ↦ rfl
  have hcβ : ∀ s, (F.axis d s).β = F.β := fun _ ↦ rfl
  rcases hfix with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · -- `μ = h = s`: `p↓ = Ep₀`, `p↑ = Ep₂`
    have hd : ∀ s, (F.axis .plus s).pDn = (F.axis .plus s).p₀ * E := fun s ↦ by
      rw [pDn_eq_p₀_mul]
      simp [Axis.dμ, Axis.dh, hE]
    have hu : ∀ s, (F.axis .plus s).pUp = (F.axis .plus s).p₂ * E := fun s ↦ by
      rw [pUp_eq_p₂_mul]
      simp [Axis.dμ, Axis.dh, hE]
    have h2 : ∀ s, (F.axis .plus s).p₂ = P - (F.axis .plus s).p₀ := fun s ↦
      hp₂ s (by rw [hd, hu]; ring)
    have t2 : Tendsto (fun s ↦ (F.axis .plus s).p₂) atTop (𝓝 P) := by
      simpa only [h2, sub_zero] using tendsto_const_nhds.sub h0
    have td : Tendsto (fun s ↦ (F.axis .plus s).pDn) atTop (𝓝 0) := by
      simpa only [hd, zero_mul] using h0.mul_const E
    have tu : Tendsto (fun s ↦ (F.axis .plus s).pUp) atTop (𝓝 (P * E)) := by
      simpa only [hu] using t2.mul_const E
    have hx := tendsto_E8x h0 tu td t2
    simp only [mul_zero, ofReal_zero] at hx
    refine F.exists_E8L_iy_ne_of_tendsto k ha ?_ ?_ _ hcU hcβ hx
    · rw [add_zero, ← mul_one_sub]
      exact mul_ne_zero hP0.ne' (sub_ne_zero.2 (Ne.symm hE1))
    · rw [sub_zero]
      positivity
  · -- `μ = -h = s`: `p↑ = Ep₀`, `p↓ = Ep₂`
    have hu : ∀ s, (F.axis .minus s).pUp = (F.axis .minus s).p₀ * E := fun s ↦ by
      rw [pUp_eq_p₀_mul]
      simp [Axis.dμ, Axis.dh, hE]
    have hd : ∀ s, (F.axis .minus s).pDn = (F.axis .minus s).p₂ * E := fun s ↦ by
      rw [pDn_eq_p₂_mul]
      simp [Axis.dμ, Axis.dh, hE]
    have h2 : ∀ s, (F.axis .minus s).p₂ = P - (F.axis .minus s).p₀ := fun s ↦
      hp₂ s (by rw [hd, hu]; ring)
    have t2 : Tendsto (fun s ↦ (F.axis .minus s).p₂) atTop (𝓝 P) := by
      simpa only [h2, sub_zero] using tendsto_const_nhds.sub h0
    have tu : Tendsto (fun s ↦ (F.axis .minus s).pUp) atTop (𝓝 0) := by
      simpa only [hu, zero_mul] using h0.mul_const E
    have td : Tendsto (fun s ↦ (F.axis .minus s).pDn) atTop (𝓝 (P * E)) := by
      simpa only [hd] using t2.mul_const E
    have hx := tendsto_E8x h0 tu td t2
    simp only [zero_mul, ofReal_zero] at hx
    refine F.exists_E8L_iy_ne_of_tendsto k ha ?_ ?_ _ hcU hcβ hx
    · rw [sub_zero]
      positivity
    · rw [add_zero, ← mul_one_sub]
      exact mul_ne_zero hP0.ne' (sub_ne_zero.2 (Ne.symm hE1))

/-- **On a moving lattice `E₈` is not identically zero**: if the center `s(d_μ ± d_h)` of the
lattice of `σ` moves, then at large `|s|`, `‖z_l‖ ≥ |s(d_μ ± d_h)|` exceeds the root bound
(`exists_E8L_ne_of_le`). -/
theorem exists_axis_E8L_z_ne_moving {σ : Bool} (hsl : d.slope σ ≠ 0) (l : ℤ) :
    ∃ s, (F.axis d s).E8L k (((F.axis d s).BL σ).z k l) ≠ 0 := by
  obtain ⟨M, hM⟩ := F.exists_E8L_ne_of_le k
  set s₁ := (|M| + 1) / d.slope σ
  refine ⟨s₁, hM (F.axis d s₁) rfl rfl _ ?_⟩
  have hre := (F.axis_z_re_im k d σ l s₁).1
  rw [div_mul_cancel₀ _ hsl] at hre
  have h1 := abs_re_le_norm (((F.axis d s₁).BL σ).z k l)
  rw [hre, abs_of_pos (by positivity)] at h1
  have h2 : (|M| + 1) ^ 2 ≤ ‖((F.axis d s₁).BL σ).z k l‖ ^ 2 :=
    pow_le_pow_left₀ (by positivity) h1 2
  have h3 : M ≤ (|M| + 1) ^ 2 := by nlinarith [abs_nonneg M, le_abs_self M]
  exact h3.trans h2

/-- **`E₈` at each lattice point vanishes only at isolated points of the axis**: for `U ≠ 0`,
`s ↦ E₈(z_{σ,l}(s))` is analytic (`analyticAt_axis_E8L_z`) and not identically zero
(`exists_axis_E8L_z_ne_moving`, `exists_axis_E8L_z_ne_fixed`). -/
theorem eventually_axis_E8L_z_ne (hU : F.U ≠ 0) (σ : Bool) (l : ℤ) (s₀ : ℝ) :
    ∀ᶠ s in 𝓝[≠] s₀, (F.axis d s).E8L k (((F.axis d s).BL σ).z k l) ≠ 0 := by
  obtain ⟨s₁, h₁⟩ : ∃ s, (F.axis d s).E8L k (((F.axis d s).BL σ).z k l) ≠ 0 := by
    by_cases hsl : d.slope σ = 0
    · exact F.exists_axis_E8L_z_ne_fixed k d hU ((d.slope_eq_zero_iff σ).1 hsl) l
    · exact F.exists_axis_E8L_z_ne_moving k d hsl l
  exact eventually_ne_zero_of_analyticAt (F.analyticAt_axis_E8L_z k d σ l) h₁ s₀

/-- **On each axis, `E₈` is nonzero on both spin lattices except at isolated points**: at fixed
`U ≠ 0`, `β`, `k`, for every `s₀`, `E₈(z_{σ,l}) ≠ 0` for all `σ`, `l` at every `s ≠ s₀` near
`s₀`. For `(ν_l + ω_k/2)² ≥ M`, the uniform root bound (`exists_E8L_ne_of_le`), `‖z_l‖² ≥ M` and
`E₈(z_l) ≠ 0` everywhere; the finitely many other `l` vanish only at isolated points
(`eventually_axis_E8L_z_ne`). -/
theorem eventually_E8L_lattice_ne_axis (hU : F.U ≠ 0) (s₀ : ℝ) :
    ∀ᶠ s in 𝓝[≠] s₀, ∀ σ l, (F.axis d s).E8L k (((F.axis d s).BL σ).z k l) ≠ 0 := by
  obtain ⟨M, hM⟩ := F.exists_E8L_ne_of_le k
  have hfin : {l : ℤ | (F.ν l + F.ω k / 2) ^ 2 ≤ M}.Finite := (F.lat 0).finite_setOf_sq_le k M
  filter_upwards [(eventually_all_finite hfin).2 fun l _ ↦ eventually_all.2 fun σ ↦
    F.eventually_axis_E8L_z_ne k d hU σ l s₀] with s hs σ l
  by_cases hl : (F.ν l + F.ω k / 2) ^ 2 ≤ M
  · exact hs l hl σ
  · refine hM (F.axis d s) rfl rfl _ ?_
    have him := (F.axis_z_re_im k d σ l s).2
    have h1 := abs_im_le_norm (((F.axis d s).BL σ).z k l)
    rw [him] at h1
    have h2 := sq_le_sq' (neg_le_of_abs_le h1) (le_of_abs_le h1)
    linarith [not_le.1 hl]

/-! ### `LAxisGeneric` near the axis points -/

/-- **Near every axis point, the roots of `E₈` are distinct and off the lattices**: at fixed
`U ≠ 0`, `β`, `k`, at every `s ≠ s₀` near `s₀` on the axis `d`, the four roots of `E₈` counted
with multiplicity, `E₈(v) = ∏ⱼ(v² - u_j²)`, have distinct squares (`sep`) and lie off both
lattices (`off`, for every root, atomic or not). -/
theorem eventually_sep_off_axis (hU : F.U ≠ 0) (s₀ : ℝ) :
    ∀ᶠ s in 𝓝[≠] s₀, ∀ u : Fin 4 → ℂ, (∀ v, (F.axis d s).E8L k v = ∏ j, (v ^ 2 - u j ^ 2)) →
      (∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) ∧
        ∀ j σ l, ((F.axis d s).BL σ).z k l ≠ u j ∧ ((F.axis d s).BL σ).z k l ≠ -u j := by
  filter_upwards [F.eventually_discE8_ne_axis k d hU s₀,
    F.eventually_E8L_lattice_ne_axis k d hU s₀] with s hd hl u hu
  exact ⟨(F.axis d s).sub_ne_of_discE8_ne k (s := fun j ↦ u j ^ 2) hu hd,
    fun j σ l ↦ ((F.axis d s).E8L_ne_iff_of_prod k hu _).1 (hl σ l) j⟩

/-- **Near every axis point, `LAxisGeneric` reduces to its `K` conditions**: at fixed `U ≠ 0`,
`β`, `k`, at every `s ≠ s₀` near `s₀` on the axis `d`, `LAxisGeneric` holds with the roots `u`
iff the `u_j²` are the roots of `E₈` in `s = v²` counted with multiplicity and, at each
non-atomic root, `K_t(u_j) ≠ 0` or `K_{t,-t}(u_j) ≠ 0`, for each column spin `t`
(`eventually_sep_off_axis`, `lAxisGeneric_iff`). -/
theorem eventually_lAxisGeneric_iff_axis (hU : F.U ≠ 0) (s₀ : ℝ) :
    ∀ᶠ s in 𝓝[≠] s₀, ∀ u : Fin 4 → ℂ, (F.axis d s).LAxisGeneric k u ↔
      (∀ v, (F.axis d s).E8L k v = ∏ j, (v ^ 2 - u j ^ 2)) ∧
        ∀ t j, ¬(F.axis d s).Atomic k (u j) →
          (F.axis d s).KL k t t (u j) ≠ 0 ∨ (F.axis d s).KL k t (!t) (u j) ≠ 0 := by
  filter_upwards [F.eventually_sep_off_axis k d hU s₀] with s hs u
  have ha : ((F.axis d s).a : ℂ) ≠ 0 := by
    rw [axis_a, a]
    exact_mod_cast div_ne_zero hU two_ne_zero
  rw [lAxisGeneric_iff]
  constructor
  · rintro ⟨-, root, sep, -, hK⟩
    refine ⟨fun v ↦ ?_, hK⟩
    rw [(F.axis d s).E8L_eq_prod k root sep v, Fin.prod_univ_four]
  · rintro ⟨hu, hK⟩
    obtain ⟨sep, off⟩ := hs u hu
    have root : ∀ j, (F.axis d s).E8L k (u j) = 0 := fun j ↦ by
      rw [hu, Finset.prod_eq_zero_iff]
      exact ⟨j, Finset.mem_univ _, sub_self _⟩
    exact ⟨ha, root, sep, fun j _ σ l ↦ off j σ l, hK⟩

end Axis

end FParams

end HubbardAtom
