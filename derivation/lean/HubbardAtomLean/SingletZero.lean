import HubbardAtomLean.Singlet

/-!
# Theorem "singlet" at `b = 0`

At `μ = 0`, `k = 0` (`b = 0`) the two poles `t₊ = t₋ = -a²` coincide and the pair coefficient is
`c₀ = βp₀`. Then the regular part of `χ_s` is the rank-two form
`c₁ g₀(t)g₀(t') + c₂ (g₀(t)²g₀(t') + g₀(t)g₀(t')²)`, `g₀ = 1/(t + a²)`, `c₁ = 2a²βp₀ - a`,
`c₂ = 2a³` (`chiS_eq_zero`), and the inverse of Theorem "singlet" is the same `invS`.

* `regular_pp_zero`: the regular part at `b = 0` (a rational identity).
* `Params.exists_hasSum_sq_mul_κ`: `∑ₙ g₀²κ` with the double-pole sum `∑ₙ g₀²`.
* `woodbury_identity_zero`: the cancellation, as a rational identity.
* `Params.SingletHyp`: the assumptions of Theorem "singlet" without `b ≠ 0`, and
  `Params.SingletHyp.chiS_isRightInverse`, `chiS_isLeftInverse`, `gammaS_eq`.
-/

open Complex

namespace HubbardAtom

/-- The regular part at `b = 0` (`μ = 0`, `k = 0`): with `q₊ = q₋ = 1/2` and the pair coefficient
`βp₀`, `(βp₀Π + R_reg)/2 = c₁g₀(x)g₀(y) + c₂(g₀(x)²g₀(y) + g₀(x)g₀(y)²)` at the transfer
`w' = -x - y`, with `g₀(x) = 1/((x - a)(-x - a))`. -/
theorem regular_pp_zero {a β p₀ x y : ℂ} (h₁ : x - a ≠ 0) (h₂ : x + a ≠ 0) (h₃ : y - a ≠ 0)
    (h₄ : y + a ≠ 0) :
    (β * p₀ * piPair' a x y (-y) (-x) + Rreg' a (1 / 2) (1 / 2) x y (-y) (-x)) / 2 =
      (2 * a ^ 2 * β * p₀ - a) * (1 / ((x - a) * (-x - a))) * (1 / ((y - a) * (-y - a))) +
        2 * a ^ 3 * ((1 / ((x - a) * (-x - a))) ^ 2 * (1 / ((y - a) * (-y - a))) +
          (1 / ((x - a) * (-x - a))) * (1 / ((y - a) * (-y - a))) ^ 2) := by
  have h₅ : -x - a ≠ 0 := fun h ↦ h₂ (by linear_combination -h)
  have h₆ : -y - a ≠ 0 := fun h ↦ h₄ (by linear_combination -h)
  have h₇ : -y + a ≠ 0 := fun h ↦ h₃ (by linear_combination -h)
  have h₈ : -x + a ≠ 0 := fun h ↦ h₁ (by linear_combination -h)
  unfold piPair' Rreg'
  field_simp
  ring

/-- Woodbury's cancellation at `b = 0`, where `t₊ = t₋ = -a²` and `Aₛ = a²(1 - 8p₁)`: given the
sums `T₁ = ∑ g₀ = -S/a`, `T₂ = ∑ g₀² = (S² - β²/4)/(2a²) - S/(2a³)` with `S = S₀(a)` from the
weights, `T₀ = ∑ κ` with `Kₛ = 1 + 2aT₀/β`, and `V₁ = ∑ g₀κ`, `V₂ = ∑ g₀²κ` with `dV₁ = T₁ - T₀`
and `d²V₂ = dT₂ - T₁ + T₀` (`d = Aₛ + t₊`), the remainder of `(χ_s invS)ₙₘ - δₙₘ` vanishes in
`t = tₙ`, `t' = tₘ`. -/
theorem woodbury_identity_zero {a β p₀ p₁ T₀ T₁ T₂ V₁ V₂ Ks S t t' As tp tm : ℂ}
    (hs : p₀ + 2 * p₁ + p₀ = 1) (hS : S = β * (p₀ - p₁) / (2 * (p₁ + p₀)))
    (hT₁ : T₁ = -S / a) (hT₂ : T₂ = (S ^ 2 - β ^ 2 / 4) / (2 * a ^ 2) - S / (2 * a ^ 3))
    (htp : tp = -a ^ 2) (htm : tm = -a ^ 2) (hAs : As = a ^ 2 * (1 - 8 * p₁))
    (hV₁ : (As + tp) * V₁ = T₁ - T₀) (hV₂ : (As + tp) ^ 2 * V₂ = (As + tp) * T₂ - T₁ + T₀)
    (hKs : Ks = 1 + 2 * a * T₀ / β) (ha : a ≠ 0) (hβ : β ≠ 0) (hq : p₁ + p₀ ≠ 0) (hK : Ks ≠ 0)
    (ht : t - tp ≠ 0) (ht' : t' - tp ≠ 0) (hA : t + As ≠ 0) (hA' : t' + As ≠ 0) :
    β / 2 * (t + As) / ((t - tp) * (t - tm)) *
          (-(4 / β ^ 2) * (-a + a * (As + tp) * (As + tm) * (1 / (t + As)) * (1 / (t' + As)) /
            Ks)) +
        ((2 * a ^ 2 * β * p₀ - a) * (1 / (t - tp)) * (1 / (t' - tp)) +
            2 * a ^ 3 * ((1 / (t - tp)) ^ 2 * (1 / (t' - tp)) +
              (1 / (t - tp)) * (1 / (t' - tp)) ^ 2)) /
          (β / 2 * (t' + As) / ((t' - tp) * (t' - tm))) +
      -(4 / β ^ 2) *
        ((2 * a ^ 2 * β * p₀ - a) * (1 / (t - tp)) *
            (-a * T₁ + a * (As + tp) * (As + tm) * (1 / (t' + As)) / Ks * V₁) +
          2 * a ^ 3 * ((1 / (t - tp)) ^ 2 *
              (-a * T₁ + a * (As + tp) * (As + tm) * (1 / (t' + As)) / Ks * V₁) +
            (1 / (t - tp)) *
              (-a * T₂ + a * (As + tp) * (As + tm) * (1 / (t' + As)) / Ks * V₂))) = 0 := by
  have hm : As + tm = As + tp := by rw [htp, htm]
  have e₁ : a * (As + tp) * (As + tm) * (1 / (t' + As)) / Ks * V₁ =
      a * (1 / (t' + As)) / Ks * ((As + tp) * (T₁ - T₀)) := by
    rw [hm, ← hV₁]
    ring
  have e₂ : a * (As + tp) * (As + tm) * (1 / (t' + As)) / Ks * V₂ =
      a * (1 / (t' + As)) / Ks * ((As + tp) * T₂ - T₁ + T₀) := by
    rw [hm, ← hV₂]
    ring
  rw [e₁, e₂]
  have hT₀ : T₀ = (Ks - 1) * β / (2 * a) := by
    rw [hKs]
    field_simp
    ring
  have htm' : t - tm ≠ 0 := by rwa [htm, ← htp]
  have htm'' : t' - tm ≠ 0 := by rwa [htm, ← htp]
  subst htp htm hAs
  obtain rfl : p₀ = (1 - 2 * p₁) / 2 := by linear_combination hs / 2
  rw [hT₀, hT₁, hT₂, hS]
  field_simp
  ring

namespace Params

variable (A : Params) (k : ℤ)

theorem b_eq_zero_iff : A.b k = 0 ↔ A.μ = 0 ∧ k = 0 := by
  constructor
  · intro h
    have hre := congrArg Complex.re h
    have him := congrArg Complex.im h
    simp only [b, w, ω, sub_re, sub_im, mul_re, mul_im, ofReal_re, ofReal_im, I_re, I_im,
      mul_zero, mul_one, sub_zero, zero_mul, re_ofNat, im_ofNat, zero_sub, add_zero, zero_re,
      zero_im, neg_eq_zero] at hre him
    have hβ := A.β_pos.ne'
    refine ⟨by linarith, ?_⟩
    have : (2 * (k : ℝ)) * Real.pi = 0 := by
      field_simp at him
      linarith
    have hk : (k : ℝ) = 0 := by
      rcases mul_eq_zero.mp this with h | h
      · linarith
      · exact absurd h Real.pi_ne_zero
    exact_mod_cast hk
  · rintro ⟨hμ, rfl⟩
    rw [b, A.w_zero, hμ]
    simp

/-- The particle–particle opposite-spin susceptibility, from the input, with the pair coefficient
`c_{-k}` kept (for `b ≠ 0` it is `ρ/b`, `chiPPUpDn_eq`). -/
theorem chiPPUpDn_eq' (n m : ℤ) :
    A.chiPPUpDn k n m =
      -4 * A.β * A.a ^ 2 * A.p₁ / A.calP k (A.t k n) * kron n m +
        4 * A.β * A.a ^ 2 * (A.p₀ * A.p₂ - A.p₁ ^ 2) / A.calP k (A.t k n) * kron m (mirror k n) +
        A.pairC (-k) * piPair' A.a (A.x n) (A.x m) (A.b k - A.x m) (A.b k - A.x n) +
        Rreg' A.a A.qm A.qp (A.x n) (A.x m) (A.b k - A.x m) (A.b k - A.x n) := by
  have hJJ := A.J_mul_J_mirror k n
  have hn : n + (-k - n - m - 1) = mirror k m := by unfold mirror; ring
  have hm : m + (-k - n - m - 1) = mirror k n := by unfold mirror; ring
  have hc : n + m + (-k - n - m - 1) + 1 = -k := by ring
  rw [chiPPUpDn, chiUpDn, hn, hm, hc, A.x_mirror, A.x_mirror, kron_pp k]
  congr 3
  · by_cases hnm : m = n
    · subst hnm
      rw [← A.x_mirror, kron_self, mul_one, mul_one]
      linear_combination (-4 * A.β * A.a ^ 2 * A.p₁ : ℂ) * hJJ
    · rw [kron_of_ne (Ne.symm hnm), mul_zero, mul_zero]
  · by_cases hm' : m = mirror k n
    · rw [hm', kron_self, mul_one, mul_one]
      linear_combination (4 * A.β * A.a ^ 2 * (A.p₀ * A.p₂ - A.p₁ ^ 2) : ℂ) * hJJ
    · rw [kron_of_ne hm', mul_zero, mul_zero]

/-- `g₀ = 1/(t - t₊)`. -/
noncomputable def g₀ (n : ℤ) : ℂ := 1 / (A.t k n - A.tp k)

/-- The regular part of `χ_s` at `b = 0`: `c₁g₀g₀ + c₂(g₀²g₀ + g₀g₀²)`, `c₁ = 2a²βp₀ - a`,
`c₂ = 2a³`. -/
noncomputable def chiReg₀ (n m : ℤ) : ℂ :=
  (2 * A.a ^ 2 * A.β * A.p₀ - A.a) * A.g₀ k n * A.g₀ k m +
    2 * A.a ^ 3 * (A.g₀ k n ^ 2 * A.g₀ k m + A.g₀ k n * A.g₀ k m ^ 2)

/-- **Proposition "pp"**, singlet part, at `b = 0`: `χ_s = L_s 𝓜₊ - B 𝓜₋ + χ_s,reg` with the
regular part `chiReg₀`. -/
theorem chiS_eq_zero (hb : A.b k = 0) (n m : ℤ) :
    A.chiS k n m = mirrorBlock k (fun j ↦ A.Ls k (A.t k j)) (fun j ↦ -A.Bpp k j) n m +
      A.chiReg₀ k n m := by
  obtain ⟨hμ, rfl⟩ := (A.b_eq_zero_iff k).mp hb
  have hp₂ := p₂_eq_p₀_of_mu hμ
  have hqm : (A.qm : ℂ) = 1 / 2 := by
    have := A.weights_sum
    rw [qm]
    push_cast
    rw [hp₂] at this
    have h' : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₀ = 1 := by exact_mod_cast this
    linear_combination h' / 2
  have hqp : (A.qp : ℂ) = 1 / 2 := by
    rw [qp, hp₂, add_comm]
    exact hqm
  have hc : A.pairC (-0) = (A.β * A.p₀ : ℝ) := by
    rw [neg_zero, pairC, ite_eq_left ⟨hμ, rfl⟩]
  have hreg := regular_pp_zero (a := A.a) (β := A.β) (p₀ := A.p₀) (x := A.x n) (y := A.x m)
    (A.x_sub_ofReal_ne_zero n _) (A.x_add_ofReal_ne_zero n _) (A.x_sub_ofReal_ne_zero m _)
    (A.x_add_ofReal_ne_zero m _)
  have hLB := A.Ls_add_Bpp 0 n
  have hw : (A.p₀ : ℂ) * A.p₂ - A.p₁ ^ 2 = (1 - 4 * A.p₁ - A.ρ ^ 2) / 4 := by
    exact_mod_cast A.p₀p₂_sub
  have hbx : ∀ j, A.b 0 - A.x j = -A.x j := fun j ↦ by rw [hb, zero_sub]
  have hg : ∀ j, A.g₀ 0 j = 1 / ((A.x j - A.a) * (-A.x j - A.a)) := fun j ↦ by
    rw [g₀, A.t_sub_tp, A.x_mirror, hbx]
  rw [chiS, A.chiPPUpUp_eq, A.chiPPUpDn_eq', hc, hqm, hqp, hbx, hbx]
  unfold mirrorBlock
  rw [hw, chiReg₀, hg, hg]
  push_cast
  linear_combination (-(kron n m + kron m (mirror 0 n)) / 2) * hLB + hreg

/-- `∑ₙ g²κ` is some `V` with `d²V = dT₂ - T₁ + T₀`, `d = Aₛ + t₀`, where `T₁ = ∑ g`,
`T₂ = ∑ g²`, `T₀ = ∑ κ`, `g = 1/(t - t₀)`; for `d = 0` it is `∑ κ³`. -/
theorem exists_hasSum_sq_mul_κ {t₀ T₀ T₁ T₂ : ℂ} (hT₁ : HasSum (fun n ↦ 1 / (A.t k n - t₀)) T₁)
    (hT₂ : HasSum (fun n ↦ 1 / (A.t k n - t₀) ^ 2) T₂)
    (hT₀ : HasSum (fun n ↦ 1 / (A.t k n + A.As k)) T₀) (ht : ∀ n, A.t k n - t₀ ≠ 0)
    (hAs : ∀ n, A.t k n + A.As k ≠ 0) :
    ∃ V, HasSum (fun n ↦ (1 / (A.t k n - t₀)) ^ 2 * (1 / (A.t k n + A.As k))) V ∧
      (A.As k + t₀) ^ 2 * V = (A.As k + t₀) * T₂ - T₁ + T₀ := by
  by_cases hd : A.As k + t₀ = 0
  · obtain rfl : t₀ = -A.As k := by linear_combination hd
    simp only [sub_neg_eq_add] at hT₁ ⊢
    have hs : Summable fun n ↦ (1 / (A.t k n + A.As k)) ^ 2 * (1 / (A.t k n + A.As k)) := by
      have hn : Summable fun n ↦ ‖1 / (A.t k n + A.As k)‖ := summable_norm_iff.mpr hT₀.summable
      have h0 := hT₀.summable.tendsto_cofinite_zero
      refine Summable.of_norm_bounded_eventually hn ?_
      filter_upwards [(Metric.tendsto_nhds.mp h0) 1 one_pos] with n hn'
      rw [dist_zero_right] at hn'
      rw [norm_mul, norm_pow]
      exact mul_le_of_le_one_left (norm_nonneg _) (pow_le_one₀ (norm_nonneg _) hn'.le)
    exact ⟨_, hs.hasSum, by rw [hd, zero_pow two_ne_zero, zero_mul, zero_mul, zero_sub,
      hT₁.unique hT₀, neg_add_cancel]⟩
  · refine ⟨((A.As k + t₀) * T₂ - T₁ + T₀) / (A.As k + t₀) ^ 2, ?_, by field_simp⟩
    convert ((hT₂.div_const (A.As k + t₀)).sub (hT₁.div_const ((A.As k + t₀) ^ 2))).add
      (hT₀.div_const ((A.As k + t₀) ^ 2)) using 1
    · funext n
      have h₁ := ht n
      have h₂ := hAs n
      field_simp
      ring
    · field_simp

theorem chiReg₀_mirror (n l : ℤ) : A.chiReg₀ k n (mirror k l) = A.chiReg₀ k n l := by
  unfold chiReg₀ g₀
  rw [A.t_mirror]

theorem chiReg₀_symm (n m : ℤ) : A.chiReg₀ k n m = A.chiReg₀ k m n := by
  unfold chiReg₀
  ring

variable {A k} {ξ : ℂ}

/-- **Theorem "singlet"** at `b = 0`, the inverse: `χ_s invS = 1`. -/
theorem chiS_isRightInverse_zero (ha : (A.a : ℂ) ≠ 0) (hb : A.b k = 0)
    (hAs : ∀ n, A.t k n + A.As k ≠ 0) (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k)
    (hK : A.Ks k ξ ≠ 0) : IsRightInverse (A.chiS k) (A.invS k ξ) := by
  intro n m
  have hβ := A.β_ne_zero
  have hLs : ∀ j, A.Ls k (A.t k j) ≠ 0 := fun j ↦
    div_ne_zero (mul_ne_zero (div_ne_zero hβ two_ne_zero) (hAs j)) (A.calP_ne_zero k j)
  set X := mirrorBlock k (fun j ↦ A.Ls k (A.t k j)) (fun j ↦ -A.Bpp k j) with hX
  set Xi := mirrorBlock k (fun j ↦ 1 / A.Ls k (A.t k j)) (fun j ↦ -(1 / A.Bpp k j)) with hXi
  have h₁ : HasSum (fun l ↦ X n l * Xi l m) (kron n m) := by
    convert hasSum_mirrorBlock_mul (k := k) (B := fun j ↦ A.Ls k (A.t k j))
      (L := fun j ↦ -A.Bpp k j) (B' := fun j ↦ 1 / A.Ls k (A.t k j))
      (L' := fun j ↦ -(1 / A.Bpp k j)) (fun j ↦ by simp only [A.t_mirror])
      (fun j ↦ by simp only [A.Bpp_mirror]) n m using 1
    refine (mirrorBlock_eq_kron (fun j ↦ ?_) (fun j _ ↦ ?_) n m).symm
    · exact mul_one_div_cancel (hLs j)
    · rw [neg_mul_neg]
      exact mul_one_div_cancel (A.Bpp_ne_zero k j)
  have h₂ : HasSum (fun l ↦ X n l * A.Zs k ξ l m) (A.Ls k (A.t k n) * A.Zs k ξ n m) :=
    hasSum_mirrorBlock_mul_even (v := fun l ↦ A.Zs k ξ l m) (fun l ↦ A.Zs_mirror l m) n
  have h₃ : HasSum (fun l ↦ A.chiReg₀ k n l * Xi l m)
      (1 / A.Ls k (A.t k m) * A.chiReg₀ k n m) := by
    have := hasSum_mirrorBlock_mul_even (k := k) (B := fun j ↦ 1 / A.Ls k (A.t k j))
      (L := fun j ↦ -(1 / A.Bpp k j)) (v := fun l ↦ A.chiReg₀ k n l)
      (fun l ↦ A.chiReg₀_mirror k n l) m
    convert this using 1
    funext l
    rw [hXi, mirrorBlock_symm (fun j ↦ by simp only [A.t_mirror])
      (fun j ↦ by simp only [A.Bpp_mirror]) l m, mul_comm]
  -- the lattice sums
  have hκ : HasSum (fun l ↦ 1 / (A.t k l + A.As k)) (A.Tpp k ξ) := by
    have := A.hasSum_Tpp (k := k) (t := -A.As k) (ξ := ξ) (by rw [hξ]; ring)
      (fun l h0 ↦ hAs l (by rw [h0]; ring))
    simpa [sub_neg_eq_add] using this
  have hT₁ := A.hasSum_gp k
  have ha2 : (A.a : ℂ) - A.b k / 2 ≠ 0 := by rw [hb, zero_div, sub_zero]; exact ha
  have hT₂ := A.hasSum_one_div_t_sub_sq (k := k) (t := A.tp k) (ξ := A.a - A.b k / 2)
    (by rw [tp]; ring) ha2 (fun n h0 ↦ A.t_sub_tp_ne k n (by rw [h0, sub_self]))
  obtain ⟨V₁, hV₁, hV₁'⟩ := A.exists_hasSum_mul_κ k hT₁ hκ (A.t_sub_tp_ne k) hAs
  obtain ⟨V₂, hV₂, hV₂'⟩ := A.exists_hasSum_sq_mul_κ k hT₁ hT₂ hκ (A.t_sub_tp_ne k) hAs
  set c := A.a * (A.As k + A.tp k) * (A.As k + A.tm k) * (1 / (A.t k m + A.As k)) / A.Ks k ξ
    with hc
  have h₄ : HasSum (fun l ↦ A.chiReg₀ k n l * A.Zs k ξ l m)
      (-(4 / A.β ^ 2 : ℂ) *
        ((2 * A.a ^ 2 * A.β * A.p₀ - A.a) * A.g₀ k n *
            (-A.a * A.Tpp k (A.a - A.b k / 2) + c * V₁) +
          2 * A.a ^ 3 * (A.g₀ k n ^ 2 * (-A.a * A.Tpp k (A.a - A.b k / 2) + c * V₁) +
            A.g₀ k n * (-A.a * ((A.S k (A.μ + (A.a - A.b k / 2)) ^ 2 - (A.β : ℂ) ^ 2 / 4) /
              (2 * (A.a - A.b k / 2) ^ 2) - A.S k (A.μ + (A.a - A.b k / 2)) /
              (2 * (A.a - A.b k / 2) ^ 3)) + c * V₂)))) := by
    have := (((((hT₁.mul_left (-(A.a : ℂ))).add (hV₁.mul_left c)).mul_left
      ((2 * A.a ^ 2 * A.β * A.p₀ - A.a) * A.g₀ k n)).add
      ((((hT₁.mul_left (-(A.a : ℂ))).add (hV₁.mul_left c)).mul_left
        (2 * A.a ^ 3 * A.g₀ k n ^ 2)).add
      (((hT₂.mul_left (-(A.a : ℂ))).add (hV₂.mul_left c)).mul_left
        (2 * A.a ^ 3 * A.g₀ k n)))).mul_left (-(4 / A.β ^ 2 : ℂ)))
    convert this using 1
    · funext l
      simp only [chiReg₀, Zs, hc, g₀, one_div, inv_pow]
      ring
    · ring
  have htot := ((h₁.add h₂).add h₃).add h₄
  convert htot using 1
  · funext l
    rw [A.chiS_eq_zero k hb, invS]
    ring
  · -- the remainder vanishes by `woodbury_identity_zero`
    obtain ⟨hμ, rfl⟩ := (A.b_eq_zero_iff k).mp hb
    have hp₂ := p₂_eq_p₀_of_mu hμ
    have hs : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₀ = 1 := by
      have := A.weights_sum
      rw [hp₂] at this
      exact_mod_cast this
    have hq : (A.p₁ : ℂ) + A.p₀ ≠ 0 := by exact_mod_cast (add_pos A.p₁_pos A.p₀_pos).ne'
    have hρ : (A.ρ : ℂ) = 0 := by rw [ρ, hp₂, sub_self, ofReal_zero]
    have hSa : A.S 0 (A.μ + (A.a - A.b 0 / 2)) = A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀)) := by
      rw [hb, zero_div, sub_zero, hμ, ofReal_zero, zero_add, A.S₀_a, hp₂]
    have hTp : A.Tpp 0 (A.a - A.b 0 / 2) = -(A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀))) / A.a := by
      rw [Tpp, ite_eq_right ha2, hSa, hb, zero_div, sub_zero]
    have htp : A.tp 0 = -A.a ^ 2 := by rw [tp, hb, mul_zero, add_zero]
    have htm : A.tm 0 = -A.a ^ 2 := by rw [tm, hb, mul_zero, sub_zero]
    have hAs0 : A.As 0 = A.a ^ 2 * (1 - 8 * A.p₁) := by rw [As, hρ, hb]; ring
    have hW := woodbury_identity_zero (a := A.a) (β := A.β) (p₀ := A.p₀) (p₁ := A.p₁)
      (T₀ := A.Tpp 0 ξ) (T₁ := A.Tpp 0 (A.a - A.b 0 / 2))
      (T₂ := (A.S 0 (A.μ + (A.a - A.b 0 / 2)) ^ 2 - (A.β : ℂ) ^ 2 / 4) /
        (2 * (A.a - A.b 0 / 2) ^ 2) - A.S 0 (A.μ + (A.a - A.b 0 / 2)) /
        (2 * (A.a - A.b 0 / 2) ^ 3))
      (V₁ := V₁) (V₂ := V₂) (Ks := A.Ks 0 ξ) (S := A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀)))
      (t := A.t 0 n) (t' := A.t 0 m) (As := A.As 0) (tp := A.tp 0) (tm := A.tm 0) hs rfl hTp
      (by rw [hSa, hb, zero_div, sub_zero]) htp htm hAs0 hV₁' hV₂' rfl ha hβ hq hK
      (A.t_sub_tp_ne 0 n) (A.t_sub_tp_ne 0 m) (hAs n) (hAs m)
    rw [Ls, Ls, A.calP_eq_mul, A.calP_eq_mul, Zs, chiReg₀, hc]
    simp only [g₀]
    linear_combination -hW

variable (A k ξ) in
/-- The assumptions of Theorem "singlet" (with `U ≠ 0`, off the lattice zeros `tₙ + Aₛ = 0`):
`ξ² = b²/4 + Aₛ` and the theorem's `Kₛ ≠ 0`. -/
structure SingletHyp : Prop where
  a_ne : (A.a : ℂ) ≠ 0
  t_add_As_ne : ∀ n, A.t k n + A.As k ≠ 0
  ξ_sq : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k
  Ks_ne : A.Ks k ξ ≠ 0

theorem chiS_symm' (n m : ℤ) : A.chiS k n m = A.chiS k m n := by
  have hmb := mirrorBlock_symm (k := k) (B := fun j ↦ A.Ls k (A.t k j)) (L := fun j ↦ -A.Bpp k j)
    (fun j ↦ by simp only [A.t_mirror]) (fun j ↦ by simp only [A.Bpp_mirror]) n m
  by_cases hb : A.b k = 0
  · rw [A.chiS_eq_zero k hb, A.chiS_eq_zero k hb, A.chiReg₀_symm k n m, hmb]
  · rw [A.chiS_eq k hb, A.chiS_eq k hb, chiReg_symm, hmb]

/-- **Theorem "singlet"**, the inverse: `χ_s invS = 1`, for all `b`. -/
theorem SingletHyp.chiS_isRightInverse (h : A.SingletHyp k ξ) :
    IsRightInverse (A.chiS k) (A.invS k ξ) := by
  by_cases hb : A.b k = 0
  · exact chiS_isRightInverse_zero h.a_ne hb h.t_add_As_ne h.ξ_sq h.Ks_ne
  · exact HubbardAtom.Params.chiS_isRightInverse (A := A) (k := k)
      ⟨h.a_ne, hb, h.t_add_As_ne, h.ξ_sq, h.Ks_ne⟩

/-- **Theorem "singlet"**, two-sidedness: `invS χ_s = 1`. -/
theorem SingletHyp.chiS_isLeftInverse (h : A.SingletHyp k ξ) :
    IsLeftInverse (A.chiS k) (A.invS k ξ) := by
  intro n m
  have := h.chiS_isRightInverse m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [chiS_symm' (k := k) l m, invS_symm (k := k) n l, mul_comm]

/-- **Theorem "singlet"**: `β²(χ_s⁻¹ + B⁻¹) = Γ_s` of Eq. (Gamma-s), for all `b`. -/
theorem SingletHyp.gammaS_eq (h : A.SingletHyp k ξ) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invS k ξ n m + kron n m / A.Bpp k n) = A.gammaS k ξ n m :=
  gammaS_eq_of h.t_add_As_ne h.Ks_ne n m

end Params

end HubbardAtom
