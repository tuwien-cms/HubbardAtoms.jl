import HubbardAtomLean.PHInput
import HubbardAtomLean.Matsubara

/-!
# Lattice sums for the density and magnetic channels

The sums of Section "Proofs for the density and magnetic channels", for `μ ≠ 0`. Lemma "residue
rule" is needed here only for denominators `P(t)` and `P(t)(t - x)`; instead of residues we split
`P = Q₊Q₋` and sum over each factor, which gives one formula for all `k` (for `k = 0` the zeros of
`P` are double).

* `Params.hasSum_Q_sub`, `Params.hasSum_inv_Q`: `PV ∑ₗ (zₗ + c)/Q(zₗ) = S₀(-c)` and
  `∑ₗ 1/Q(zₗ) = δ_k0 (S₀(-c)² - β²/4)` for `Q(t) = (t + c)² - w²/4`, `c` real; for `k ≠ 0` the
  second sum telescopes to zero.
* `Params.hasSum_cubic`: the lattice sum of `M/P` for cubic `M`, through the constants
  `ℓⱼ = PV ∑ₗ zₗʲ/P(zₗ)` (`Params.ℓ₀` … `Params.ℓ₃`).
* `Params.rho_mul_S_neg_z`: Lemma "values on the reflected lattice", `ρS_k(-zₙ) = c̄`.
* `Params.hasSum_cauchy`, `Params.hasSum_square`: Lemmas "Cauchy calculus" and "square".
* `Params.α_mul_α_neg`: Eq. (alpha), `α_k(x)α_k(-x) = h²`.
* `Params.hasSum_rank_two`, `Params.hasSum_const`: Lemma "rank-two sums",
  `I(z; x) = (N(z, x)S_k(x) + R̂(z, x))/P(x)`, and the constant sum of Step 2.
-/

open Complex Filter Topology

namespace HubbardAtom

/-- Partial fractions of `(t + c)/((t + c)² - d²)`. -/
theorem pf_Q_sub {t c d y : ℂ} (h₁ : t - (-c + d) ≠ 0) (h₂ : t - (-c - d) ≠ 0) (h₃ : t - y ≠ 0) :
    (t + c) / ((t + c) ^ 2 - d ^ 2) - 1 / (t - y) =
      1 / 2 * (1 / (t - (-c + d)) - 1 / (t - y) + (1 / (t - (-c - d)) - 1 / (t - y))) := by
  rw [show t - (-c + d) = t + c - d by ring] at h₁ ⊢
  rw [show t - (-c - d) = t + c + d by ring] at h₂ ⊢
  rw [show (t + c) ^ 2 - d ^ 2 = (t + c - d) * (t + c + d) by ring]
  field_simp
  ring

/-- Partial fractions of `1/((t + c)² - d²)`. -/
theorem pf_inv_Q {t c d : ℂ} (hd : d ≠ 0) (h₁ : t - (-c + d) ≠ 0) (h₂ : t - (-c - d) ≠ 0) :
    1 / ((t + c) ^ 2 - d ^ 2) = 1 / (2 * d) * (1 / (t - (-c + d)) - 1 / (t - (-c - d))) := by
  rw [show t - (-c + d) = t + c - d by ring] at h₁ ⊢
  rw [show t - (-c - d) = t + c + d by ring] at h₂ ⊢
  rw [show (t + c) ^ 2 - d ^ 2 = (t + c - d) * (t + c + d) by ring]
  field_simp
  ring

/-- The split of a cubic over `P = Q₊Q₋`, `Q±(t) = (t ± a)² - d²`:
`M = (a₊(t + a) + b₊)Q₋ + (a₋(t - a) + b₋)Q₊`, with `M/P - m₃/(t - y)` written through the
absolutely convergent combinations of `hasSum_Q_sub`, `hasSum_inv_Q`. -/
theorem cubic_pf {t a d y m₀ m₁ m₂ m₃ da sb db : ℂ}
    (hda : da = (m₀ / (a ^ 2 - d ^ 2) - m₂) / (2 * a)) (hsb : sb = (m₂ + m₀ / (a ^ 2 - d ^ 2)) / 2)
    (hdb : db = ((a ^ 2 - d ^ 2 - 2 * a ^ 2) * m₃ - m₁) / (2 * a)) (ha : a ≠ 0)
    (hq : a ^ 2 - d ^ 2 ≠ 0) (hp : (t + a) ^ 2 - d ^ 2 ≠ 0) (hm : (t - a) ^ 2 - d ^ 2 ≠ 0)
    (hy : t - y ≠ 0) :
    (m₀ + m₁ * t + m₂ * t ^ 2 + m₃ * t ^ 3) / (((t + a) ^ 2 - d ^ 2) * ((t - a) ^ 2 - d ^ 2)) -
        m₃ / (t - y) =
      (m₃ + da) / 2 * ((t + a) / ((t + a) ^ 2 - d ^ 2) - 1 / (t - y)) +
          (m₃ - da) / 2 * ((t - a) / ((t - a) ^ 2 - d ^ 2) - 1 / (t - y)) +
        (sb + db) / 2 * (1 / ((t + a) ^ 2 - d ^ 2)) +
      (sb - db) / 2 * (1 / ((t - a) ^ 2 - d ^ 2)) := by
  subst hda hsb hdb
  field_simp
  ring

/-- The rank-two summand as a cubic over `P` minus a simple pole: for `P(t) = t⁴ + κt² + q²` and
`N(t) = n₀ + n₁t + n₂t² + n₃t³`, `P(x)N(t)/(P(t)(t - x)) = M̃(t)/P(t) + N(x)/(t - x)` with the cubic
`M̃(t) = (P(x)N(t) - N(x)P(t))/(t - x)`. -/
theorem rank_two_pf {t x n₀ n₁ n₂ n₃ κ q Pt Px : ℂ} (hPt : Pt = t ^ 4 + κ * t ^ 2 + q ^ 2)
    (hPx : Px = x ^ 4 + κ * x ^ 2 + q ^ 2) (hPt0 : Pt ≠ 0) (hPx0 : Px ≠ 0) (htx : t - x ≠ 0) :
    (n₀ + n₁ * t + n₂ * t ^ 2 + n₃ * t ^ 3) / (Pt * (t - x)) =
      ((Px * (n₁ + n₂ * x + n₃ * x ^ 2) -
            (n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) * (x ^ 3 + κ * x) +
          (Px * (n₂ + n₃ * x) - (n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) * (x ^ 2 + κ)) * t +
          (Px * n₃ - (n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) * x) * t ^ 2 +
          -(n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) * t ^ 3) / Pt -
        -(n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) / (t - x)) / Px := by
  field_simp
  rw [hPt, hPx]
  ring

namespace Params

variable (A : Params) (k : ℤ)

/-- Eq. (shift), the other half: `S_k(u) = S₀(u + w/2)`. -/
theorem S_shift' (u : ℂ) : A.S k u = A.S 0 (u + A.w k / 2) := by
  have hβ := A.β_ne_zero
  have hz : (A.β : ℂ) * (u - A.μ) / 2 =
      A.β * (u + A.w k / 2 - A.μ) / 2 + ((-k : ℤ) : ℂ) * (Real.pi / 2) * I := by
    rw [w, ω]
    push_cast
    field_simp
    ring
  rw [S, S, ite_eq_left even_zero', hz]
  by_cases hk : Even k
  · rw [ite_eq_left hk, tanh_add_int_mul, ite_eq_left hk.neg]
  · rw [ite_eq_right hk, coth_add_int_mul, ite_eq_right (fun h ↦ hk (even_neg.mp h))]

/-- `1/(zₙ - x) - 1/(zₙ - y)` sums to `S_k(x) - S_k(y)`, absolutely, for `x`, `y` off the
lattice. -/
theorem hasSum_sub_inv {x y : ℂ} (hx : ∀ n, A.z k n ≠ x) (hy : ∀ n, A.z k n ≠ y) :
    HasSum (fun n ↦ 1 / (A.z k n - x) - 1 / (A.z k n - y)) (A.S k x - A.S k y) := by
  by_cases hxy : x = y
  · subst hxy
    simp only [sub_self]
    exact hasSum_zero
  · have hxy' := sub_ne_zero.mpr hxy
    convert (A.hasSum_pair hxy hx hy).mul_left (x - y) using 1
    · funext n
      have h₁ := sub_ne_zero.mpr (hx n)
      have h₂ := sub_ne_zero.mpr (hy n)
      field_simp
      ring
    · field_simp

/-- The zeros `-c ± w/2` of `(t + c)² - w²/4` are off the lattice, for real `c`: there
`xₙ + c = 0` resp. `x_{n+k} + c = 0`. -/
theorem z_ne_neg_add (c : ℝ) (n : ℤ) : A.z k n ≠ -c + A.w k / 2 := by
  intro h
  apply A.x_add_ofReal_ne_zero n c
  rw [x_eq_z A k, h]
  ring

theorem z_ne_neg_sub (c : ℝ) (n : ℤ) : A.z k n ≠ -c - A.w k / 2 := by
  intro h
  apply A.x_add_ofReal_ne_zero (n + k) c
  rw [← A.x_add_w, x_eq_z A k, h]
  ring

theorem w_ne_zero (hk : k ≠ 0) : A.w k ≠ 0 := by
  rw [w, ω]
  have hβ := A.β_pos.ne'
  have hπ := Real.pi_ne_zero
  have hk' : (k : ℝ) ≠ 0 := Int.cast_ne_zero.mpr hk
  exact mul_ne_zero (ofReal_ne_zero.mpr (div_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero hk')
    hπ) hβ)) I_ne_zero

/-- `PV ∑ₗ (zₗ + c)/((zₗ + c)² - w²/4) = S₀(-c)`, written as an absolutely convergent sum by
subtracting `1/(zₗ - y)`, for real `c`. For `k ≠ 0` the fraction is `(1/(t - r₁) + 1/(t - r₂))/2`
with `r₁,₂ = -c ± w/2`, where `S_k(r₁,₂) = S₀(-c)` by Eq. (shift); for `k = 0` it is `1/(t + c)`. -/
theorem hasSum_Q_sub (c : ℝ) {y : ℂ} (hy : ∀ n, A.z k n ≠ y) :
    HasSum (fun l ↦ (A.z k l + c) / ((A.z k l + c) ^ 2 - A.w k ^ 2 / 4) - 1 / (A.z k l - y))
      (A.S 0 (-c) - A.S k y) := by
  by_cases hk : k = 0
  · subst hk
    have h := A.hasSum_sub_inv 0 (x := -c) (fun n ↦ by simpa using A.z_ne_neg_add 0 c n) hy
    convert h using 1
    funext l
    have hl : A.z 0 l + c ≠ 0 := fun h0 ↦ A.z_ne_neg_add 0 c l (by simp; linear_combination h0)
    simp only [A.w_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div,
      sub_zero, sub_neg_eq_add]
    rw [sq, div_mul_cancel_left₀ hl]
    simp only [one_div]
  · have hw := A.w_ne_zero k hk
    have hS₁ : A.S k (-c + A.w k / 2) = A.S 0 (-c) := by
      rw [S_shift]
      ring_nf
    have hS₂ : A.S k (-c - A.w k / 2) = A.S 0 (-c) := by
      rw [S_shift']
      ring_nf
    have h := ((A.hasSum_sub_inv k (A.z_ne_neg_add k c) hy).add
      (A.hasSum_sub_inv k (A.z_ne_neg_sub k c) hy)).mul_left (1 / 2)
    rw [hS₁, hS₂] at h
    convert h using 1
    · funext l
      rw [show A.w k ^ 2 / 4 = (A.w k / 2) ^ 2 by ring]
      exact pf_Q_sub (sub_ne_zero.mpr (A.z_ne_neg_add k c l))
        (sub_ne_zero.mpr (A.z_ne_neg_sub k c l)) (sub_ne_zero.mpr (hy l))
    · ring

/-- `∑ₗ 1/((zₗ + c)² - w²/4) = δ_k0 (S₀(-c)² - β²/4)` for real `c`: for `k = 0` this is
`∑ₗ 1/(zₗ + c)²`; for `k ≠ 0` the terms `(1/(zₗ - r₁) - 1/(zₗ - r₂))/w` sum to `0`. -/
theorem hasSum_inv_Q (c : ℝ) :
    HasSum (fun l ↦ 1 / ((A.z k l + c) ^ 2 - A.w k ^ 2 / 4))
      (kron k 0 * (A.S 0 (-c) ^ 2 - A.β ^ 2 / 4)) := by
  by_cases hk : k = 0
  · subst hk
    have h := A.hasSum_sq (k := 0) (x := -c) (fun n ↦ by simpa using A.z_ne_neg_add 0 c n)
    rw [kron_self, one_mul]
    convert h using 1
    funext l
    simp only [A.w_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div,
      sub_zero, sub_neg_eq_add]
  · have hw := A.w_ne_zero k hk
    have hS : A.S k (-c + A.w k / 2) - A.S k (-c - A.w k / 2) = 0 := by
      rw [A.S_shift k (-c + A.w k / 2), A.S_shift' k (-c - A.w k / 2)]
      ring_nf
    have h := (A.hasSum_sub_inv k (A.z_ne_neg_add k c) (A.z_ne_neg_sub k c)).mul_left
      (1 / A.w k)
    rw [hS, mul_zero] at h
    rw [kron_of_ne hk, zero_mul]
    convert h using 1
    funext l
    rw [show A.w k ^ 2 / 4 = (A.w k / 2) ^ 2 by ring, show 1 / A.w k = 1 / (2 * (A.w k / 2)) by
      ring]
    exact pf_inv_Q (div_ne_zero hw two_ne_zero) (sub_ne_zero.mpr (A.z_ne_neg_add k c l))
      (sub_ne_zero.mpr (A.z_ne_neg_sub k c l))

/-! ### The lattice constants of `P` -/

/-- `S₊ = S₀(-a)`, the value of `S_k` at the zeros `-a ± w/2` of `Q₊` (Lemma "basic sum"). -/
noncomputable def Sp : ℂ := A.S 0 (-A.a)

/-- `S₋ = S₀(a)`, the value of `S_k` at the zeros `a ± w/2` of `Q₋`. -/
noncomputable def Sm : ℂ := A.S 0 A.a

/-- `T₊ = ∑ₗ 1/Q₊(zₗ) = δ_k0 (S₊² - β²/4)`. -/
noncomputable def Tp : ℂ := kron k 0 * (A.Sp ^ 2 - A.β ^ 2 / 4)

/-- `T₋ = ∑ₗ 1/Q₋(zₗ) = δ_k0 (S₋² - β²/4)`. -/
noncomputable def Tm : ℂ := kron k 0 * (A.Sm ^ 2 - A.β ^ 2 / 4)

/-- `ℓ₀ = ∑ₗ 1/P(zₗ)`. -/
noncomputable def ℓ₀ : ℂ :=
  (A.Sp - A.Sm) / (4 * A.a * A.q₀ k) + (A.Tp k + A.Tm k) / (4 * A.q₀ k)

/-- `ℓ₁ = ∑ₗ zₗ/P(zₗ)`. -/
noncomputable def ℓ₁ : ℂ := (A.Tm k - A.Tp k) / (4 * A.a)

/-- `ℓ₂ = ∑ₗ zₗ²/P(zₗ)`. -/
noncomputable def ℓ₂ : ℂ := (A.Sm - A.Sp) / (4 * A.a) + (A.Tp k + A.Tm k) / 4

/-- `ℓ₃ = PV ∑ₗ zₗ³/P(zₗ)`. -/
noncomputable def ℓ₃ : ℂ :=
  (A.Sp + A.Sm) / 2 + (A.q₀ k - 2 * A.a ^ 2) * (A.Tp k - A.Tm k) / (4 * A.a)

/-- `q₀ = a² + ω²/4 ≠ 0` for `U ≠ 0`. -/
theorem q₀_ne_zero (ha : (A.a : ℂ) ≠ 0) : A.q₀ k ≠ 0 := by
  have ha' : A.a ≠ 0 := by exact_mod_cast ha
  have h : A.q₀ k = ((A.a ^ 2 + A.ω k ^ 2 / 4 : ℝ) : ℂ) := by
    rw [q₀, w]
    push_cast
    ring_nf
    rw [I_sq]
    ring
  rw [h, ofReal_ne_zero]
  positivity

/-- The lattice sum of `M/P` for a cubic `M = m₀ + m₁t + m₂t² + m₃t³`, made absolutely convergent
by subtracting `m₃/(t - y)`: `∑ₗ m₀ℓ₀ + m₁ℓ₁ + m₂ℓ₂ + m₃(ℓ₃ - S_k(y))`. The proof splits
`M = (a₊(t + a) + b₊)Q₋ + (a₋(t - a) + b₋)Q₊` (`Q₊`, `Q₋` are coprime for `U ≠ 0`). -/
theorem hasSum_cubic (ha : (A.a : ℂ) ≠ 0) (m₀ m₁ m₂ m₃ : ℂ) {y : ℂ} (hy : ∀ n, A.z k n ≠ y) :
    HasSum (fun l ↦ (m₀ + m₁ * A.z k l + m₂ * A.z k l ^ 2 + m₃ * A.z k l ^ 3) /
        A.Pph k (A.z k l) - m₃ / (A.z k l - y))
      (m₀ * A.ℓ₀ k + m₁ * A.ℓ₁ k + m₂ * A.ℓ₂ k + m₃ * (A.ℓ₃ k - A.S k y)) := by
  have hq := A.q₀_ne_zero k ha
  have hq' : (A.a : ℂ) ^ 2 - (A.w k / 2) ^ 2 ≠ 0 := by
    rw [show (A.a : ℂ) ^ 2 - (A.w k / 2) ^ 2 = A.q₀ k by rw [q₀]; ring]
    exact hq
  have hp₁ := A.hasSum_Q_sub k A.a hy
  have hp₂ := A.hasSum_Q_sub k (-A.a) hy
  have hp₃ := A.hasSum_inv_Q k A.a
  have hp₄ := A.hasSum_inv_Q k (-A.a)
  simp only [ofReal_neg, ← sub_eq_add_neg, neg_neg] at hp₂ hp₄
  set da := (m₀ / ((A.a : ℂ) ^ 2 - (A.w k / 2) ^ 2) - m₂) / (2 * A.a) with hda
  set sb := (m₂ + m₀ / ((A.a : ℂ) ^ 2 - (A.w k / 2) ^ 2)) / 2 with hsb
  set db := (((A.a : ℂ) ^ 2 - (A.w k / 2) ^ 2 - 2 * A.a ^ 2) * m₃ - m₁) / (2 * A.a) with hdb
  convert (((hp₁.mul_left ((m₃ + da) / 2)).add (hp₂.mul_left ((m₃ - da) / 2))).add
    (hp₃.mul_left ((sb + db) / 2))).add (hp₄.mul_left ((sb - db) / 2)) using 1
  · funext l
    have hQp := A.Qp_z_ne_zero k l
    have hQm := A.Qm_z_ne_zero k l
    rw [Qp] at hQp
    rw [Qm] at hQm
    rw [Pph, Qp, Qm]
    rw [show A.w k ^ 2 / 4 = (A.w k / 2) ^ 2 by ring] at hQp hQm ⊢
    exact cubic_pf hda hsb hdb ha hq' hQp hQm (sub_ne_zero.mpr (hy l))
  · rw [ℓ₀, ℓ₁, ℓ₂, ℓ₃, Tp, Tm, Sp, Sm, hda, hsb, hdb, q₀]
    rw [q₀] at hq
    field_simp
    ring

/-! ### The reflected lattice -/

/-- `-zₙ` is off the lattice for `μ ≠ 0`. -/
theorem z_ne_neg_z (hμ : A.μ ≠ 0) (n m : ℤ) : A.z k m ≠ -A.z k n := fun h ↦
  A.z_add_z_ne_zero k hμ m n (by rw [h]; ring)

theorem z_injective {n m : ℤ} (h : A.z k n = A.z k m) : n = m := by
  have h' := congrArg im h
  rw [A.z_eq, A.z_eq] at h'
  simp only [add_im, ofReal_im, mul_im, ofReal_re, I_re, I_im, mul_zero, mul_one, zero_add,
    add_zero] at h'
  have hβ := A.β_pos.ne'
  have hπ := Real.pi_pos.ne'
  rw [div_left_inj' hβ] at h'
  have h'' := mul_right_cancel₀ hπ h'
  have : (n : ℝ) = m := by linarith
  exact_mod_cast this

/-- Lemma "values on the reflected lattice": `ρS_k(-zₙ) = c̄` for `μ ≠ 0`, as
`S_k(-zₙ) = S₀(-xₙ) = (β/2)coth(βμ)` and `ρ coth(βμ) = p₀ + p₂` (Eq. weight-hyp). -/
theorem rho_mul_S_neg_z (hμ : A.μ ≠ 0) (n : ℤ) : (A.ρ : ℂ) * A.S k (-A.z k n) = A.cbar := by
  have hβ := A.β_ne_zero
  rw [A.S_shift' k, show -A.z k n + A.w k / 2 = -A.x n by rw [z]; ring, S, ite_eq_left even_zero']
  have harg : (A.β : ℂ) * (-A.x n - A.μ) / 2 =
      ((-(A.β * A.μ) : ℝ) : ℂ) + ((-(2 * n + 1) : ℤ) : ℂ) * (Real.pi / 2) * I := by
    rw [x, ν]
    push_cast
    field_simp
    ring
  have hodd : ¬Even (-(2 * n + 1)) := by
    rw [even_neg, Int.not_even_iff_odd]
    exact odd_two_mul_add_one n
  rw [harg, tanh_add_int_mul, ite_eq_right hodd, coth_eq_exp]
  set E := Real.exp (2 * A.β * A.μ) with hE
  have hE1 : E ≠ 1 := by
    rw [hE, Ne, Real.exp_eq_one_iff]
    exact mul_ne_zero (mul_ne_zero two_ne_zero A.β_pos.ne') hμ
  have hexp : exp (2 * ((-(A.β * A.μ) : ℝ) : ℂ)) = ((E⁻¹ : ℝ) : ℂ) := by
    rw [← ofReal_ofNat, ← ofReal_mul, ← ofReal_exp, hE, ← Real.exp_neg]
    congr 2
    ring
  have hE0 : (E : ℂ) ≠ 0 := ofReal_ne_zero.mpr (Real.exp_pos _).ne'
  have hE1' : (E : ℂ) - 1 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast hE1)
  have hE1'' : (1 : ℂ) - E ≠ 0 := sub_ne_zero.mpr (Ne.symm (by exact_mod_cast hE1))
  have hE1''' : (E : ℂ)⁻¹ - 1 ≠ 0 := by
    rw [show (E : ℂ)⁻¹ - 1 = (1 - E) / E by field_simp]
    exact div_ne_zero hE1'' hE0
  rw [hexp, ρ, cbar, A.p₂_eq_p₀_mul, ← hE]
  push_cast
  field_simp
  ring

/-- Lemma "Cauchy calculus": `∑ₗ ρ/((zₙ + zₗ)(zₗ - x)) = α_k(x)/(zₙ + x)` for `x` off the lattice,
`x ≠ -zₙ`, absolutely convergent. -/
theorem hasSum_cauchy (hμ : A.μ ≠ 0) (n : ℤ) {x : ℂ} (hx : ∀ l, A.z k l ≠ x)
    (hxn : x ≠ -A.z k n) :
    HasSum (fun l ↦ A.ρ / (A.z k n + A.z k l) * (1 / (A.z k l - x)))
      (A.α k x / (A.z k n + x)) := by
  have h := (A.hasSum_pair (k := k) (x := -A.z k n) (y := x) (Ne.symm hxn)
    (fun l ↦ A.z_ne_neg_z k hμ n l) hx).mul_left (A.ρ : ℂ)
  have hxn' : A.z k n + x ≠ 0 := fun h0 ↦ hxn (by linear_combination h0)
  have hxn'' : -A.z k n - x ≠ 0 := fun h0 ↦ hxn' (by linear_combination -h0)
  convert h using 1
  · funext l
    have h₁ := A.z_add_z_ne_zero k hμ n l
    have h₂ := sub_ne_zero.mpr (hx l)
    rw [sub_neg_eq_add, add_comm (A.z k l)]
    field_simp
  · rw [α, ← A.rho_mul_S_neg_z k hμ n]
    field_simp
    ring

/-- Lemma "square": `∑ₗ C_nl C_lm = h²δₙₘ` with `C_nm = ρ/(zₙ + zₘ)`, `h² = β²p₀p₂`. -/
theorem hasSum_square (hμ : A.μ ≠ 0) (n m : ℤ) :
    HasSum (fun l ↦ A.ρ / (A.z k n + A.z k l) * (A.ρ / (A.z k l + A.z k m)))
      (A.β ^ 2 * A.p₀ * A.p₂ * kron n m) := by
  have hSn := A.rho_mul_S_neg_z k hμ n
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self, mul_one]
    have h := (A.hasSum_sq (k := k) (x := -A.z k n) (fun l ↦ A.z_ne_neg_z k hμ n l)).mul_left
      ((A.ρ : ℂ) ^ 2)
    convert h using 1
    · funext l
      have := A.z_add_z_ne_zero k hμ n l
      rw [sub_neg_eq_add, add_comm (A.z k l)]
      field_simp
    · rw [show (A.ρ : ℂ) ^ 2 * (A.S k (-A.z k n) ^ 2 - A.β ^ 2 / 4) =
        (A.ρ * A.S k (-A.z k n)) ^ 2 - A.ρ ^ 2 * A.β ^ 2 / 4 by ring, hSn, cbar, ρ]
      push_cast
      ring
  · rw [kron_of_ne hnm, mul_zero]
    have hzz : -A.z k n ≠ -A.z k m := fun h ↦ hnm (A.z_injective k (neg_inj.mp h))
    have h := (A.hasSum_pair hzz (fun l ↦ A.z_ne_neg_z k hμ n l)
      (fun l ↦ A.z_ne_neg_z k hμ m l)).mul_left ((A.ρ : ℂ) ^ 2)
    convert h using 1
    · funext l
      rw [sub_neg_eq_add, sub_neg_eq_add, add_comm (A.z k n), one_div, mul_inv]
      ring
    · rw [show (A.ρ : ℂ) ^ 2 * ((A.S k (-A.z k n) - A.S k (-A.z k m)) / (-A.z k n - -A.z k m)) =
        A.ρ * (A.ρ * A.S k (-A.z k n) - A.ρ * A.S k (-A.z k m)) / (-A.z k n - -A.z k m) by ring,
        hSn, A.rho_mul_S_neg_z k hμ m, sub_self, mul_zero, zero_div]

/-- Eq. (alpha): `α_k(x)α_k(-x) = h² = β²p₀p₂` for `x`, `-x` off the lattice. -/
theorem α_mul_α_neg {x : ℂ} (hx : ∀ n, A.z k n ≠ x) (hx' : ∀ n, A.z k n ≠ -x) :
    A.α k x * A.α k (-x) = A.β ^ 2 * A.p₀ * A.p₂ := by
  have hh : (A.hth : ℂ) ^ 2 = A.β ^ 2 * A.p₀ * A.p₂ := by
    have hs : Real.sqrt (A.p₀ * A.p₂) ^ 2 = A.p₀ * A.p₂ :=
      Real.sq_sqrt (mul_pos A.p₀_pos A.p₂_pos).le
    rw [hth, ofReal_mul, mul_pow, ← ofReal_pow (Real.sqrt (A.p₀ * A.p₂)) 2, hs]
    push_cast
    ring
  have hy₁ : (A.β : ℂ) * (-x + A.μ) / 2 = -(A.β * (x - A.μ) / 2) := by ring
  have hy₂ : (A.β : ℂ) * (-x - A.μ) / 2 = -(A.β * (x + A.μ) / 2) := by ring
  obtain ⟨j, hj | hj⟩ := Int.even_or_odd' k
  · have hk : Even k := ⟨j, by omega⟩
    have hc := A.cosh_ne_zero_of_off hj hx
    have hc' := A.cosh_ne_zero_of_off hj hx'
    rw [A.αeven_eq hk x hc, A.αeven_eq hk (-x) hc', hy₁, hy₂, cosh_neg, cosh_neg]
    rw [hy₂, cosh_neg] at hc'
    field_simp
    linear_combination hh
  · have hk : ¬Even k := by rintro ⟨r, hr⟩; omega
    have hj' : k = 2 * (j + 1) - 1 := by omega
    have hs := A.sinh_ne_zero_of_off hj' hx
    have hs' := A.sinh_ne_zero_of_off hj' hx'
    rw [A.αodd_eq hk x hs, A.αodd_eq hk (-x) hs', hy₁, hy₂, sinh_neg, sinh_neg]
    rw [hy₂, sinh_neg, neg_ne_zero] at hs'
    field_simp
    linear_combination hh

/-! ### The rank-two sums -/

variable (ε : ℂ)

/-- `κ = 2q₀ - 4a²`, so that `P(t) = t⁴ + κt² + q₀²`. -/
noncomputable def κ : ℂ := 2 * A.q₀ k - 4 * A.a ^ 2

/-- The coefficient of `t³` in `N(z, t)`: `-λρ`. -/
noncomputable def cN₃ : ℂ := -4 * ε * A.a ^ 2 * A.ρ

/-- The coefficient of `t²` in `N(z, t)`. -/
noncomputable def cN₂ (z : ℂ) : ℂ :=
  -2 * ε * A.a * A.q₀ k - kron k 0 * A.ζ ε * A.a ^ 2 + z ^ 2 * (-2 * ε * A.a + kron k 0 * A.ζ ε)

/-- The coefficient of `t` in `N(z, t)`. -/
noncomputable def cN₁ (z : ℂ) : ℂ := (A.κ k + z ^ 2) * A.cN₃ ε

/-- The coefficient of `1` in `N(z, t)`. -/
noncomputable def cN₀ (z : ℂ) : ℂ :=
  (A.κ k + z ^ 2) * (-2 * ε * A.a * A.q₀ k - kron k 0 * A.ζ ε * A.a ^ 2) +
    (2 * ε * A.a - kron k 0 * A.ζ ε) * A.q₀ k ^ 2

theorem Pph_eq_poly (t : ℂ) : A.Pph k t = t ^ 4 + A.κ k * t ^ 2 + A.q₀ k ^ 2 := by
  rw [Pph, Qp, Qm, κ, q₀]
  ring

theorem Nph_eq_coeffs (z t : ℂ) :
    A.Nph k ε z t = A.cN₀ k ε z + A.cN₁ k ε z * t + A.cN₂ k ε z * t ^ 2 + A.cN₃ ε * t ^ 3 := by
  rw [Nph, N₀, N₂, A.Pph_eq_poly, cN₀, cN₁, cN₂, cN₃, κ]
  ring

/-- `P(x)Ī(z; x) - c̄N(z, x)/ρ`: the part of the rank-two sum `I(z; x)` that comes from the zeros
of `P`, i.e. the lattice sum of the cubic `(P(x)N(z, t) - N(z, x)P(t))/(t - x)` over `P(t)`. -/
noncomputable def Rhat (z x : ℂ) : ℂ :=
  (A.Pph k x * (A.cN₁ k ε z + A.cN₂ k ε z * x + A.cN₃ ε * x ^ 2) -
        A.Nph k ε z x * (x ^ 3 + A.κ k * x)) * A.ℓ₀ k +
      (A.Pph k x * (A.cN₂ k ε z + A.cN₃ ε * x) - A.Nph k ε z x * (x ^ 2 + A.κ k)) * A.ℓ₁ k +
    (A.Pph k x * A.cN₃ ε - A.Nph k ε z x * x) * A.ℓ₂ k - A.Nph k ε z x * A.ℓ₃ k

/-- Lemma "rank-two sums": `I(z; x) = ∑ₗ N(z, zₗ)/(P(zₗ)(zₗ - x)) = (N(z, x)S_k(x) + R̂(z, x))/P(x)`
for `x` off the lattice with `P(x) ≠ 0`, absolutely convergent. -/
theorem hasSum_rank_two (ha : (A.a : ℂ) ≠ 0) (z : ℂ) {x : ℂ} (hx : ∀ l, A.z k l ≠ x)
    (hPx : A.Pph k x ≠ 0) :
    HasSum (fun l ↦ A.Nph k ε z (A.z k l) / (A.Pph k (A.z k l) * (A.z k l - x)))
      ((A.Nph k ε z x * A.S k x + A.Rhat k ε z x) / A.Pph k x) := by
  have h := (A.hasSum_cubic k ha
    (A.Pph k x * (A.cN₁ k ε z + A.cN₂ k ε z * x + A.cN₃ ε * x ^ 2) -
      A.Nph k ε z x * (x ^ 3 + A.κ k * x))
    (A.Pph k x * (A.cN₂ k ε z + A.cN₃ ε * x) - A.Nph k ε z x * (x ^ 2 + A.κ k))
    (A.Pph k x * A.cN₃ ε - A.Nph k ε z x * x) (-A.Nph k ε z x) hx).div_const (A.Pph k x)
  convert h using 1
  · funext l
    rw [A.Nph_eq_coeffs, A.Nph_eq_coeffs]
    exact rank_two_pf (A.Pph_eq_poly k _) (A.Pph_eq_poly k x) (A.Pph_z_ne_zero k l) hPx
      (sub_ne_zero.mpr (hx l))
  · rw [Rhat]
    field_simp
    ring

/-- The constant sum of Step 2: `∑ₗ [λρ/(zₙ + zₗ) + N(zₙ, zₗ)/P(zₗ)] = λc̄ + ∑ⱼ cNⱼ(zₙ)ℓⱼ`,
absolutely convergent (the `1/zₗ` tails cancel, as the `t³` coefficient of `N` is `-λρ`). -/
theorem hasSum_const (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) (n : ℤ) :
    HasSum (fun l ↦ A.lam ε * A.ρ / (A.z k n + A.z k l) +
        A.Nph k ε (A.z k n) (A.z k l) / A.Pph k (A.z k l))
      (A.lam ε * A.cbar + (A.cN₀ k ε (A.z k n) * A.ℓ₀ k + A.cN₁ k ε (A.z k n) * A.ℓ₁ k +
        A.cN₂ k ε (A.z k n) * A.ℓ₂ k + A.cN₃ ε * A.ℓ₃ k)) := by
  have h := A.hasSum_cubic k ha (A.cN₀ k ε (A.z k n)) (A.cN₁ k ε (A.z k n))
    (A.cN₂ k ε (A.z k n)) (A.cN₃ ε) (fun l ↦ A.z_ne_neg_z k hμ n l)
  convert h using 1
  · funext l
    rw [A.Nph_eq_coeffs, cN₃, lam, sub_neg_eq_add, add_comm (A.z k l)]
    ring
  · rw [← A.rho_mul_S_neg_z k hμ n, cN₃, lam]
    ring

end Params

end HubbardAtom
