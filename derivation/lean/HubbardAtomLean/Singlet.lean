import HubbardAtomLean.SingletInput
import HubbardAtomLean.Matsubara

/-!
# Theorem "singlet"

Theorem "singlet" of the note in its generic case, under the assumptions `Params.SingletGeneric`:
`U ≠ 0`, `b ≠ 0, ±2a`, `d± ≠ 0`, `tₙ + Aₛ ≠ 0`, `ξ ≠ 0`, and the theorem's `Kₛ ≠ 0`. The excluded
values, where the note takes limits, are not covered here.

* `Params.hasSum_gp`, `Params.hasSum_gm`, `Params.hasSum_mul_κ`: the lattice sums `∑ₙ 1/(tₙ - t±)`
  (Eq. T-atomic) and `∑ₙ 1/((tₙ - t₀)(tₙ + Aₛ))`, from Lemma "particle–particle sum".
* `woodbury_identity`: the cancellation behind the Woodbury formula of the proof, as a rational
  identity (it contains identity (A4) of the note).
* `Params.chiS_isRightInverse`, `Params.chiS_isLeftInverse`: `χ_s` has the two-sided inverse
  `invS = L_s⁻¹ 𝓜₊ - B⁻¹ 𝓜₋ - (4/β²)(-a + a d₊d₋ κₙκₘ/Kₛ)`, `κₙ = 1/(tₙ + Aₛ)`.
* `Params.gammaS_eq`: **Theorem "singlet"**, `β²(χ_s⁻¹ + B⁻¹) = Γ_s` of Eq. (Gamma-s) for this
  inverse.
-/

open Complex

namespace HubbardAtom

/-- Identity (A4) of the note at `t₊`: `a d₊/(t₊ - t₋) + γ₊(1 + 2aT₊/β) = 0` for
`T₊ = -2S₋/(2a - b)` (Eq. T-atomic) with `S₋ = β(p₂ - p₁)/(2q₊)`. -/
theorem singlet_A4p {a b β p₀ p₁ p₂ : ℂ} (hs : p₀ + 2 * p₁ + p₂ = 1) (ha : a ≠ 0) (hb : b ≠ 0)
    (hβ : β ≠ 0) (hb₁ : 2 * a - b ≠ 0) (hqp : p₁ + p₂ ≠ 0) :
    a * (a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁) + (-a ^ 2 + a * b)) /
          ((-a ^ 2 + a * b) - (-a ^ 2 - a * b)) +
        a * (p₂ - p₀ + 1) * (2 * a - b) / (2 * b) *
          (1 + 2 * a * (-2 * (β * (p₂ - p₁) / (2 * (p₁ + p₂))) / (2 * a - b)) / β) = 0 := by
  have h2ab : (-a ^ 2 + a * b) - (-a ^ 2 - a * b) = 2 * a * b := by ring
  have hρp : p₂ - p₀ + 1 = 2 * (p₁ + p₂) := by linear_combination (-1 : ℂ) * hs
  -- the form in which `field_simp` meets the denominator
  have hb₁' : a * 2 - b ≠ 0 := by rwa [mul_comm]
  rw [h2ab, hρp]
  -- `d₊/(2b) + (a/b)[q₊(2a - b) - 2a(p₂ - p₁)] = 0`
  have e₁ : a * (a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁) + (-a ^ 2 + a * b)) / (2 * a * b) =
      (a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁) + (-a ^ 2 + a * b)) / (2 * b) := by
    field_simp
  have e₂ : a * (2 * (p₁ + p₂)) * (2 * a - b) / (2 * b) *
        (1 + 2 * a * (-2 * (β * (p₂ - p₁) / (2 * (p₁ + p₂))) / (2 * a - b)) / β) =
      a / b * ((p₁ + p₂) * (2 * a - b) - 2 * a * (p₂ - p₁)) := by
    field_simp
    ring
  rw [e₁, e₂]
  field_simp
  linear_combination (-a * b) * hs

/-- Identity (A4) of the note at `t₋`: `-a d₋/(t₊ - t₋) + γ₋(1 + 2aT₋/β) = 0` for
`T₋ = 2S₊/(2a + b)` (Eq. T-atomic) with `S₊ = β(p₁ - p₀)/(2q₋)`. -/
theorem singlet_A4m {a b β p₀ p₁ p₂ : ℂ} (hs : p₀ + 2 * p₁ + p₂ = 1) (ha : a ≠ 0) (hb : b ≠ 0)
    (hβ : β ≠ 0) (hb₂ : 2 * a + b ≠ 0) (hqm : p₀ + p₁ ≠ 0) :
    -a * (a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁) + (-a ^ 2 - a * b)) /
          ((-a ^ 2 + a * b) - (-a ^ 2 - a * b)) +
        a * (p₂ - p₀ - 1) * (2 * a + b) / (2 * b) *
          (1 + 2 * a * (2 * (β * (p₁ - p₀) / (2 * (p₀ + p₁))) / (2 * a + b)) / β) = 0 := by
  have h2ab : (-a ^ 2 + a * b) - (-a ^ 2 - a * b) = 2 * a * b := by ring
  have hρm : p₂ - p₀ - 1 = -2 * (p₀ + p₁) := by linear_combination hs
  have hb₂' : a * 2 + b ≠ 0 := by rwa [mul_comm]
  rw [h2ab, hρm]
  have e₁ : -a * (a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁) + (-a ^ 2 - a * b)) / (2 * a * b) =
      -(a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁) + (-a ^ 2 - a * b)) / (2 * b) := by
    field_simp
  have e₂ : a * (-2 * (p₀ + p₁)) * (2 * a + b) / (2 * b) *
        (1 + 2 * a * (2 * (β * (p₁ - p₀) / (2 * (p₀ + p₁))) / (2 * a + b)) / β) =
      -(a / b) * ((p₀ + p₁) * (2 * a + b) + 2 * a * (p₁ - p₀)) := by
    field_simp
  rw [e₁, e₂]
  field_simp
  linear_combination (-a * b) * hs

/-- Identity (A4) of the note: `±a d±/(t₊ - t₋) + γ±(1 + 2aT±/β) = 0` for
`T₊ = -2S₋/(2a - b)` and `T₋ = 2S₊/(2a + b)` (Eq. T-atomic), with `S₋ = β(p₂ - p₁)/(2q₊)` and
`S₊ = β(p₁ - p₀)/(2q₋)`. -/
theorem singlet_A4 {a b β p₀ p₁ p₂ : ℂ} (hs : p₀ + 2 * p₁ + p₂ = 1) (ha : a ≠ 0) (hb : b ≠ 0)
    (hβ : β ≠ 0) (hb₁ : 2 * a - b ≠ 0) (hb₂ : 2 * a + b ≠ 0) (hqp : p₁ + p₂ ≠ 0)
    (hqm : p₀ + p₁ ≠ 0) :
    a * (a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁) + (-a ^ 2 + a * b)) /
          ((-a ^ 2 + a * b) - (-a ^ 2 - a * b)) +
        a * (p₂ - p₀ + 1) * (2 * a - b) / (2 * b) *
          (1 + 2 * a * (-2 * (β * (p₂ - p₁) / (2 * (p₁ + p₂))) / (2 * a - b)) / β) = 0 ∧
      -a * (a * (p₂ - p₀) * b + a ^ 2 * (1 - 8 * p₁) + (-a ^ 2 - a * b)) /
          ((-a ^ 2 + a * b) - (-a ^ 2 - a * b)) +
        a * (p₂ - p₀ - 1) * (2 * a + b) / (2 * b) *
          (1 + 2 * a * (2 * (β * (p₁ - p₀) / (2 * (p₀ + p₁))) / (2 * a + b)) / β) = 0 :=
  ⟨singlet_A4p hs ha hb hβ hb₁ hqp, singlet_A4m hs ha hb hβ hb₂ hqm⟩

/-- The cancellation in the Woodbury formula of the proof of Theorem "singlet": given identity (A4)
at `t±` (`hIp`, `hIm`), any `T₀` with `Kₛ = 1 + 2aT₀/β`, and `V±` with `d±V± = T± - T₀` (the sums
`∑ₙ g±κ`), the three terms of `(χ_s Y)ₙₘ - δₙₘ` cancel identically in `t = tₙ`, `t' = tₘ`: they are
`(2/β)[g₊(t)(1 - d₋κ'/Kₛ) I₊ + g₋(t)(1 - d₊κ'/Kₛ) I₋]` with the vanishing `I±`,
`κ' = 1/(t' + Aₛ)`. No `d±` needs to be nonzero. -/
theorem woodbury_identity {a β T₀ t t' As tp tm dp dm Tp Tm Vp Vm Ks γp γm : ℂ}
    (hdp : dp = As + tp) (hdm : dm = As + tm) (hVp : dp * Vp = Tp - T₀) (hVm : dm * Vm = Tm - T₀)
    (hKs : Ks = 1 + 2 * a * T₀ / β)
    (hIp : a * dp / (tp - tm) + γp * (1 + 2 * a * Tp / β) = 0)
    (hIm : -a * dm / (tp - tm) + γm * (1 + 2 * a * Tm / β) = 0)
    (ha : a ≠ 0) (hβ : β ≠ 0) (hpm : tp - tm ≠ 0) (hK : Ks ≠ 0)
    (ht₁ : t - tp ≠ 0) (ht₂ : t - tm ≠ 0) (ht₃ : t + As ≠ 0)
    (ht₁' : t' - tp ≠ 0) (ht₂' : t' - tm ≠ 0) (ht₃' : t' + As ≠ 0) :
    β / 2 * (t + As) / ((t - tp) * (t - tm)) *
          (-(4 / β ^ 2) * (-a + a * dp * dm * (1 / (t + As)) * (1 / (t' + As)) / Ks)) +
        (γp / (t - tp) / (t' - tp) + γm / (t - tm) / (t' - tm)) /
          (β / 2 * (t' + As) / ((t' - tp) * (t' - tm))) +
      -(4 / β ^ 2) *
        (γp / (t - tp) * (-a * Tp + a * dp * dm * (1 / (t' + As)) / Ks * Vp) +
          γm / (t - tm) * (-a * Tm + a * dp * dm * (1 / (t' + As)) / Ks * Vm)) = 0 := by
  -- `T₀` in terms of `Kₛ`
  have hT₀ : T₀ = (Ks - 1) * β / (2 * a) := by
    rw [hKs]
    field_simp
    ring
  have e₁ : a * dp * dm * (1 / (t' + As)) / Ks * Vp =
      a * dm * (1 / (t' + As)) / Ks * (Tp - T₀) := by
    rw [← hVp]
    ring
  have e₂ : a * dp * dm * (1 / (t' + As)) / Ks * Vm =
      a * dp * (1 / (t' + As)) / Ks * (Tm - T₀) := by
    rw [← hVm]
    ring
  rw [e₁, e₂]
  calc _ = 2 / β * (1 / (t - tp)) * (1 - dm * (1 / (t' + As)) / Ks) *
          (a * dp / (tp - tm) + γp * (1 + 2 * a * Tp / β)) +
        2 / β * (1 / (t - tm)) * (1 - dp * (1 / (t' + As)) / Ks) *
          (-a * dm / (tp - tm) + γm * (1 + 2 * a * Tm / β)) := by
        rw [hT₀, hdp, hdm]
        field_simp
        ring
    _ = 0 := by
        rw [hIp, hIm]
        ring

/-- A matrix acting on mirror orbits, applied to a mirror-even vector, multiplies it by the even
eigenvalue: `∑ₗ (B𝓜₊ + L𝓜₋)ₙₗ vₗ = Bₙ vₙ`. -/
theorem hasSum_mirrorBlock_mul_even {k : ℤ} {B L v : ℤ → ℂ} (hv : ∀ l, v (mirror k l) = v l)
    (n : ℤ) : HasSum (fun l ↦ mirrorBlock k B L n l * v l) (B n * v n) := by
  have hsum := hasSum_sum_of_ne_finset_zero (L := SummationFilter.unconditional ℤ)
    (s := ({n, mirror k n} : Finset ℤ)) (f := fun l ↦ mirrorBlock k B L n l * v l)
    (fun l hl ↦ by simp only [mirrorBlock_eq_zero hl, zero_mul])
  convert hsum using 1
  by_cases hfix : mirror k n = n
  · rw [hfix, Finset.insert_eq_of_mem (Finset.mem_singleton_self n), Finset.sum_singleton,
      mirrorBlock_of_fixed hfix, kron_self, mul_one]
  · rw [Finset.sum_pair (Ne.symm hfix), mirrorBlock_self (Ne.symm hfix),
      mirrorBlock_mirror (Ne.symm hfix), hv]
    ring

namespace Params

variable (A : Params) (k : ℤ)

/-- `𝒫(t) = (t - t₊)(t - t₋)`. -/
theorem calP_eq_mul (t : ℂ) : A.calP k t = (t - A.tp k) * (t - A.tm k) := by
  rw [calP, tp, tm]
  ring

theorem t_sub_tp_ne (n : ℤ) : A.t k n - A.tp k ≠ 0 := by
  rw [A.t_sub_tp]
  exact mul_ne_zero (A.x_sub_ofReal_ne_zero n _) (A.x_sub_ofReal_ne_zero _ _)

theorem t_sub_tm_ne (n : ℤ) : A.t k n - A.tm k ≠ 0 := by
  rw [A.t_sub_tm]
  exact mul_ne_zero (A.x_add_ofReal_ne_zero n _) (A.x_add_ofReal_ne_zero _ _)

/-- Lemma "particle–particle sum" at `t₊`: `∑ₙ 1/(tₙ - t₊) = 𝒯_k(t₊)`, `ξ = a - b/2`. -/
theorem hasSum_gp :
    HasSum (fun n ↦ 1 / (A.t k n - A.tp k)) (A.Tpp k (A.a - A.b k / 2)) :=
  A.hasSum_Tpp (by rw [tp]; ring) (fun n h0 ↦ A.t_sub_tp_ne k n (by rw [h0, sub_self]))

/-- Lemma "particle–particle sum" at `t₋`: `∑ₙ 1/(tₙ - t₋) = 𝒯_k(t₋)`, `ξ = -a - b/2`. -/
theorem hasSum_gm :
    HasSum (fun n ↦ 1 / (A.t k n - A.tm k)) (A.Tpp k (-A.a - A.b k / 2)) :=
  A.hasSum_Tpp (by rw [tm]; ring) (fun n h0 ↦ A.t_sub_tm_ne k n (by rw [h0, sub_self]))

/-- Eq. (T-atomic) at `t₊`: `𝒯_k(t₊) = -2S₋/(2a - b)` with `S₋ = S₀(a)`, for `b ≠ 2a`. -/
theorem Tpp_tp (hb : A.b k ≠ 2 * A.a) :
    A.Tpp k (A.a - A.b k / 2) = -2 * A.S 0 A.a / (2 * A.a - A.b k) := by
  have hξ0 : (A.a : ℂ) - A.b k / 2 ≠ 0 := fun h0 ↦ hb (by linear_combination -2 * h0)
  have hb₁ : 2 * (A.a : ℂ) - A.b k ≠ 0 := fun h0 ↦ hb (by linear_combination -h0)
  have hμ : (A.μ : ℂ) + (A.a - A.b k / 2) - A.w k / 2 = A.a := by rw [b]; ring
  rw [Tpp, ite_eq_right hξ0, A.S_shift, hμ]
  field_simp

/-- Eq. (T-atomic) at `t₋`: `𝒯_k(t₋) = 2S₊/(2a + b)` with `S₊ = S₀(-a)`, for `b ≠ -2a`. -/
theorem Tpp_tm (hb : A.b k ≠ -2 * A.a) :
    A.Tpp k (-A.a - A.b k / 2) = 2 * A.S 0 (-A.a) / (2 * A.a + A.b k) := by
  have hξ0 : -(A.a : ℂ) - A.b k / 2 ≠ 0 := fun h0 ↦ hb (by linear_combination -2 * h0)
  have hb₂ : 2 * (A.a : ℂ) + A.b k ≠ 0 := fun h0 ↦ hb (by linear_combination h0)
  have hμ : (A.μ : ℂ) + (-A.a - A.b k / 2) - A.w k / 2 = -A.a := by rw [b]; ring
  rw [Tpp, ite_eq_right hξ0, A.S_shift, hμ, div_eq_div_iff hξ0 hb₂]
  ring

/-- `∑ₙ 1/((tₙ - t₀)(tₙ + Aₛ)) = (T - T₀)/(Aₛ + t₀)`, by partial fractions. -/
theorem hasSum_mul_κ {t₀ T T₀ : ℂ} (hT : HasSum (fun n ↦ 1 / (A.t k n - t₀)) T)
    (hT₀ : HasSum (fun n ↦ 1 / (A.t k n + A.As k)) T₀) (hd : A.As k + t₀ ≠ 0)
    (ht : ∀ n, A.t k n - t₀ ≠ 0) (hAs : ∀ n, A.t k n + A.As k ≠ 0) :
    HasSum (fun n ↦ 1 / (A.t k n - t₀) * (1 / (A.t k n + A.As k))) ((T - T₀) / (A.As k + t₀)) := by
  convert (hT.sub hT₀).div_const (A.As k + t₀) using 1
  funext n
  have h₁ := ht n
  have h₂ := hAs n
  field_simp
  ring

/-- `∑ₙ 1/((tₙ - t₀)(tₙ + Aₛ))` is some `V` with `(Aₛ + t₀)V = T - T₀`; for `t₀ = -Aₛ` it is
`∑ₙ (tₙ + Aₛ)⁻²`. -/
theorem exists_hasSum_mul_κ {t₀ T T₀ : ℂ} (hT : HasSum (fun n ↦ 1 / (A.t k n - t₀)) T)
    (hT₀ : HasSum (fun n ↦ 1 / (A.t k n + A.As k)) T₀) (ht : ∀ n, A.t k n - t₀ ≠ 0)
    (hAs : ∀ n, A.t k n + A.As k ≠ 0) :
    ∃ V, HasSum (fun n ↦ 1 / (A.t k n - t₀) * (1 / (A.t k n + A.As k))) V ∧
      (A.As k + t₀) * V = T - T₀ := by
  by_cases hd : A.As k + t₀ = 0
  · obtain rfl : t₀ = -A.As k := by linear_combination hd
    simp only [sub_neg_eq_add] at hT ⊢
    -- the squares are summable: `κₙ → 0` and `∑ |κₙ|` converges
    have hs : Summable fun n ↦ 1 / (A.t k n + A.As k) * (1 / (A.t k n + A.As k)) := by
      have hn : Summable fun n ↦ ‖1 / (A.t k n + A.As k)‖ := summable_norm_iff.mpr hT₀.summable
      have h0 := hT₀.summable.tendsto_cofinite_zero
      refine Summable.of_norm_bounded_eventually hn ?_
      filter_upwards [(Metric.tendsto_nhds.mp h0) 1 one_pos] with n hn'
      rw [dist_zero_right] at hn'
      rw [norm_mul]
      exact mul_le_of_le_one_left (norm_nonneg _) hn'.le
    exact ⟨_, hs.hasSum, by rw [hd, zero_mul, hT.unique hT₀, sub_self]⟩
  · exact ⟨_, A.hasSum_mul_κ k hT hT₀ hd ht hAs, by field_simp⟩

/-- `Kₛ = 1 + (2a/β) 𝒯_k(-Aₛ)` of Theorem "singlet", `ξ² = b²/4 + Aₛ`. -/
noncomputable def Ks (ξ : ℂ) : ℂ := 1 + 2 * A.a * A.Tpp k ξ / A.β

/-- The assumption `Kₛ ≠ 0` of Theorem "singlet", and those of its generic case. The values
`b = ±2a`, `d± = 0` and `ξ = 0`, which the note reaches by continuity, are included. -/
structure SingletGeneric (ξ : ℂ) : Prop where
  a_ne : (A.a : ℂ) ≠ 0
  b_ne : A.b k ≠ 0
  t_add_As_ne : ∀ n, A.t k n + A.As k ≠ 0
  ξ_sq : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k
  Ks_ne : A.Ks k ξ ≠ 0

/-- At `b = 2a` (so `k = 0`, `μ = a`, and `p₁ = p₂`), `d₊ = Aₛ + t₊ = 0`. -/
theorem dp_eq_zero_of_b (hb : A.b k = 2 * A.a) : A.As k + A.tp k = 0 := by
  have hre := congrArg Complex.re hb
  simp only [b, w, sub_re, mul_re, ofReal_re, ofReal_im, I_re, I_im, mul_zero, sub_zero,
    re_ofNat, im_ofNat, mul_one, sub_self] at hre
  have hμ : A.μ = A.a := by linarith
  have hp : A.p₁ = A.p₂ := by rw [A.p₁_eq_p₂_mul, hμ, sub_self, mul_zero, Real.exp_zero, mul_one]
  have hs : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₂ = 1 := by exact_mod_cast A.weights_sum
  rw [As, tp, hb, ρ]
  push_cast
  rw [hp] at hs ⊢
  linear_combination (-2 * (A.a : ℂ) ^ 2) * hs

/-- At `b = -2a` (so `k = 0`, `μ = -a`, and `p₁ = p₀`), `d₋ = Aₛ + t₋ = 0`. -/
theorem dm_eq_zero_of_b (hb : A.b k = -2 * A.a) : A.As k + A.tm k = 0 := by
  have hre := congrArg Complex.re hb
  simp only [b, w, sub_re, mul_re, ofReal_re, ofReal_im, I_re, I_im, mul_zero, sub_zero,
    re_ofNat, im_ofNat, mul_one, sub_self, neg_mul, neg_re] at hre
  have hμ : A.μ = -A.a := by linarith
  have hp : A.p₁ = A.p₀ := by
    rw [A.p₁_eq_p₀_mul, hμ, add_neg_cancel, mul_zero, Real.exp_zero, mul_one]
  have hs : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₂ = 1 := by exact_mod_cast A.weights_sum
  rw [As, tm, hb, ρ]
  push_cast
  rw [hp] at hs ⊢
  linear_combination (-2 * (A.a : ℂ) ^ 2) * hs

/-- Identity (A4) at `t₊`, for all `b ≠ 0`: at `b = 2a` both terms vanish (`d₊ = γ₊ = 0`). -/
theorem singlet_Ip (ha : (A.a : ℂ) ≠ 0) (hb : A.b k ≠ 0) :
    A.a * (A.As k + A.tp k) / (A.tp k - A.tm k) +
      A.γp k * (1 + 2 * A.a * A.Tpp k (A.a - A.b k / 2) / A.β) = 0 := by
  by_cases hb2 : A.b k = 2 * A.a
  · rw [A.dp_eq_zero_of_b k hb2, γp, hb2]
    ring
  · have hβ := A.β_ne_zero
    have hb₁ : 2 * (A.a : ℂ) - A.b k ≠ 0 := fun h0 ↦ hb2 (by linear_combination -h0)
    have hs : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₂ = 1 := by exact_mod_cast A.weights_sum
    have hqp : (A.p₁ : ℂ) + A.p₂ ≠ 0 := by exact_mod_cast (add_pos A.p₁_pos A.p₂_pos).ne'
    have h := singlet_A4p hs ha hb hβ hb₁ hqp
    rw [A.Tpp_tp k hb2, A.S₀_a, γp, As, tp, tm, ρ]
    push_cast
    linear_combination h

/-- Identity (A4) at `t₋`, for all `b ≠ 0`: at `b = -2a` both terms vanish (`d₋ = γ₋ = 0`). -/
theorem singlet_Im (ha : (A.a : ℂ) ≠ 0) (hb : A.b k ≠ 0) :
    -A.a * (A.As k + A.tm k) / (A.tp k - A.tm k) +
      A.γm k * (1 + 2 * A.a * A.Tpp k (-A.a - A.b k / 2) / A.β) = 0 := by
  by_cases hb2 : A.b k = -2 * A.a
  · rw [A.dm_eq_zero_of_b k hb2, γm, hb2]
    ring
  · have hβ := A.β_ne_zero
    have hb₂ : 2 * (A.a : ℂ) + A.b k ≠ 0 := fun h0 ↦ hb2 (by linear_combination h0)
    have hs : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₂ = 1 := by exact_mod_cast A.weights_sum
    have hqm : (A.p₀ : ℂ) + A.p₁ ≠ 0 := by exact_mod_cast (add_pos A.p₀_pos A.p₁_pos).ne'
    have h := singlet_A4m hs ha hb hβ hb₂ hqm
    rw [A.Tpp_tm k hb2, A.S₀_neg_a, γm, As, tp, tm, ρ]
    push_cast
    linear_combination h

/-- The separable part `Zₙₘ = -(4/β²)(-a + a d₊d₋ κₙκₘ/Kₛ)` of the inverse, `κₙ = 1/(tₙ + Aₛ)`. -/
noncomputable def Zs (ξ : ℂ) (n m : ℤ) : ℂ :=
  -(4 / A.β ^ 2 : ℂ) * (-A.a + A.a * (A.As k + A.tp k) * (A.As k + A.tm k) *
    (1 / (A.t k n + A.As k)) * (1 / (A.t k m + A.As k)) / A.Ks k ξ)

/-- The inverse of `χ_s`: `L_s⁻¹𝓜₊ - B⁻¹𝓜₋ + Z`. -/
noncomputable def invS (ξ : ℂ) (n m : ℤ) : ℂ :=
  mirrorBlock k (fun j ↦ 1 / A.Ls k (A.t k j)) (fun j ↦ -(1 / A.Bpp k j)) n m + A.Zs k ξ n m

/-- The closed form of Eq. (Gamma-s). -/
noncomputable def gammaS (ξ : ℂ) (n m : ℤ) : ℂ :=
  A.β * A.a ^ 2 * (A.ρ ^ 2 - 1 + 8 * A.p₁) * A.calP k (A.t k n) /
      (A.calG k (A.t k n) * (A.t k n + A.As k)) * (kron n m + kron m (mirror k n)) +
    4 * A.a - 4 * A.a * A.calP k (-A.As k) / (A.Ks k ξ * (A.t k n + A.As k) * (A.t k m + A.As k))

/-! ### The inverse -/

variable {A k} {ξ : ℂ}

theorem Ls_ne (h : A.SingletGeneric k ξ) (n : ℤ) : A.Ls k (A.t k n) ≠ 0 := by
  have hβ := A.β_ne_zero
  exact div_ne_zero (mul_ne_zero (div_ne_zero hβ two_ne_zero) (h.t_add_As_ne n))
    (A.calP_ne_zero k n)

theorem chiReg_symm (n m : ℤ) : A.chiReg k n m = A.chiReg k m n := by
  unfold chiReg
  ring

theorem chiReg_mirror (n l : ℤ) : A.chiReg k n (mirror k l) = A.chiReg k n l := by
  unfold chiReg
  rw [A.t_mirror]

theorem Zs_symm (n m : ℤ) : A.Zs k ξ n m = A.Zs k ξ m n := by
  unfold Zs
  ring

theorem Zs_mirror (l m : ℤ) : A.Zs k ξ (mirror k l) m = A.Zs k ξ l m := by
  unfold Zs
  rw [A.t_mirror]

theorem chiS_symm (h : A.SingletGeneric k ξ) (n m : ℤ) : A.chiS k n m = A.chiS k m n := by
  rw [A.chiS_eq k h.b_ne, A.chiS_eq k h.b_ne, chiReg_symm,
    mirrorBlock_symm (fun j ↦ by simp only [A.t_mirror]) (fun j ↦ by simp only [A.Bpp_mirror])]

theorem invS_symm (n m : ℤ) : A.invS k ξ n m = A.invS k ξ m n := by
  rw [invS, invS, Zs_symm,
    mirrorBlock_symm (fun j ↦ by simp only [A.t_mirror]) (fun j ↦ by simp only [A.Bpp_mirror])]

/-- **Theorem "singlet"**, the inverse: `χ_s invS = 1`. The sum over `l` splits into the parts
`L_s𝓜₊ - B𝓜₋` (finite on each orbit) and the rank-two `χ_s,reg`, whose products with the
separable `Z` are lattice sums (`hasSum_gp`, `hasSum_gm`, `hasSum_mul_κ`); the remainder vanishes by
`woodbury_identity`. -/
theorem chiS_isRightInverse (h : A.SingletGeneric k ξ) :
    IsRightInverse (A.chiS k) (A.invS k ξ) := by
  intro n m
  have hβ := A.β_ne_zero
  set X := mirrorBlock k (fun j ↦ A.Ls k (A.t k j)) (fun j ↦ -A.Bpp k j) with hX
  set Xi := mirrorBlock k (fun j ↦ 1 / A.Ls k (A.t k j)) (fun j ↦ -(1 / A.Bpp k j)) with hXi
  -- `X Xi = 1`
  have h₁ : HasSum (fun l ↦ X n l * Xi l m) (kron n m) := by
    convert hasSum_mirrorBlock_mul (k := k) (B := fun j ↦ A.Ls k (A.t k j))
      (L := fun j ↦ -A.Bpp k j) (B' := fun j ↦ 1 / A.Ls k (A.t k j))
      (L' := fun j ↦ -(1 / A.Bpp k j)) (fun j ↦ by simp only [A.t_mirror])
      (fun j ↦ by simp only [A.Bpp_mirror]) n m using 1
    refine (mirrorBlock_eq_kron (fun j ↦ ?_) (fun j _ ↦ ?_) n m).symm
    · exact mul_one_div_cancel (A.Ls_ne h j)
    · rw [neg_mul_neg]
      exact mul_one_div_cancel (A.Bpp_ne_zero k j)
  -- `X Z = L_s Z`, `χ_reg Xi = χ_reg/L_s`: both factors act on mirror-even vectors
  have h₂ : HasSum (fun l ↦ X n l * A.Zs k ξ l m) (A.Ls k (A.t k n) * A.Zs k ξ n m) :=
    hasSum_mirrorBlock_mul_even (v := fun l ↦ A.Zs k ξ l m) (fun l ↦ A.Zs_mirror l m) n
  have h₃ : HasSum (fun l ↦ A.chiReg k n l * Xi l m) (1 / A.Ls k (A.t k m) * A.chiReg k n m) := by
    have := hasSum_mirrorBlock_mul_even (k := k) (B := fun j ↦ 1 / A.Ls k (A.t k j))
      (L := fun j ↦ -(1 / A.Bpp k j)) (v := fun l ↦ A.chiReg k n l) (fun l ↦ A.chiReg_mirror n l) m
    convert this using 1
    funext l
    rw [hXi, mirrorBlock_symm (fun j ↦ by simp only [A.t_mirror])
      (fun j ↦ by simp only [A.Bpp_mirror]) l m, mul_comm]
  -- `χ_reg Z`: lattice sums
  set T₀ := A.Tpp k ξ with hT₀
  have hκ : HasSum (fun l ↦ 1 / (A.t k l + A.As k)) T₀ := by
    have := A.hasSum_Tpp (k := k) (t := -A.As k) (ξ := ξ) (by rw [h.ξ_sq]; ring)
      (fun l h0 ↦ h.t_add_As_ne l (by rw [h0]; ring))
    simpa [sub_neg_eq_add] using this
  have hgp := A.hasSum_gp k
  have hgm := A.hasSum_gm k
  obtain ⟨Vp, hgpκ, hVp⟩ := A.exists_hasSum_mul_κ k hgp hκ (A.t_sub_tp_ne k) h.t_add_As_ne
  obtain ⟨Vm, hgmκ, hVm⟩ := A.exists_hasSum_mul_κ k hgm hκ (A.t_sub_tm_ne k) h.t_add_As_ne
  set c := A.a * (A.As k + A.tp k) * (A.As k + A.tm k) * (1 / (A.t k m + A.As k)) / A.Ks k ξ with hc
  have h₄ : HasSum (fun l ↦ A.chiReg k n l * A.Zs k ξ l m)
      (-(4 / A.β ^ 2 : ℂ) *
        (A.γp k / (A.t k n - A.tp k) * (-A.a * A.Tpp k (A.a - A.b k / 2) + c * Vp) +
          A.γm k / (A.t k n - A.tm k) * (-A.a * A.Tpp k (-A.a - A.b k / 2) + c * Vm))) := by
    have := ((((hgp.mul_left (-(A.a : ℂ))).add (hgpκ.mul_left c)).mul_left
      (A.γp k / (A.t k n - A.tp k))).add
      (((hgm.mul_left (-(A.a : ℂ))).add (hgmκ.mul_left c)).mul_left
      (A.γm k / (A.t k n - A.tm k)))).mul_left (-(4 / A.β ^ 2 : ℂ))
    convert this using 1
    funext l
    rw [chiReg, Zs, hc]
    ring
  have htot := ((h₁.add h₂).add h₃).add h₄
  convert htot using 1
  · funext l
    rw [A.chiS_eq k h.b_ne, invS]
    ring
  · -- the remainder vanishes
    have hpm : A.tp k - A.tm k ≠ 0 := by
      rw [tp, tm, show -(A.a : ℂ) ^ 2 + A.a * A.b k - (-A.a ^ 2 - A.a * A.b k) =
        2 * A.a * A.b k by ring]
      exact mul_ne_zero (mul_ne_zero two_ne_zero h.a_ne) h.b_ne
    have hW := woodbury_identity (a := A.a) (β := A.β) (T₀ := T₀) (t := A.t k n)
      (t' := A.t k m) (As := A.As k) (tp := A.tp k) (tm := A.tm k) (dp := A.As k + A.tp k)
      (dm := A.As k + A.tm k) (Tp := A.Tpp k (A.a - A.b k / 2))
      (Tm := A.Tpp k (-A.a - A.b k / 2)) (Vp := Vp) (Vm := Vm) (Ks := A.Ks k ξ)
      (γp := A.γp k) (γm := A.γm k) rfl rfl hVp hVm (by rw [Ks, hT₀])
      (A.singlet_Ip k h.a_ne h.b_ne) (A.singlet_Im k h.a_ne h.b_ne) h.a_ne hβ hpm h.Ks_ne
      (A.t_sub_tp_ne k n) (A.t_sub_tm_ne k n) (h.t_add_As_ne n) (A.t_sub_tp_ne k m)
      (A.t_sub_tm_ne k m) (h.t_add_As_ne m)
    rw [Ls, Ls, A.calP_eq_mul, A.calP_eq_mul, Zs, chiReg, hc]
    linear_combination -hW

theorem chiS_isLeftInverse (h : A.SingletGeneric k ξ) :
    IsLeftInverse (A.chiS k) (A.invS k ξ) := by
  intro n m
  have := A.chiS_isRightInverse h m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [A.chiS_symm h, A.invS_symm, mul_comm]

/-- `β²(invS + B⁻¹) = Γ_s` of Eq. (Gamma-s), an identity of the closed forms. -/
theorem gammaS_eq_of (hAs : ∀ n, A.t k n + A.As k ≠ 0) (hK : A.Ks k ξ ≠ 0) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invS k ξ n m + kron n m / A.Bpp k n) = A.gammaS k ξ n m := by
  have hβ := A.β_ne_zero
  have hP := A.calP_ne_zero k n
  have hA := hAs n
  have hA' := hAs m
  have hD : (A.As k + A.tp k) * (A.As k + A.tm k) = A.calP k (-A.As k) := by
    rw [calP, tp, tm]
    ring
  -- the diagonal and mirror part
  have key : (A.β : ℂ) ^ 2 * (1 / A.Ls k (A.t k n) + 1 / A.Bpp k n) / 2 =
      A.β * A.a ^ 2 * (A.ρ ^ 2 - 1 + 8 * A.p₁) * A.calP k (A.t k n) /
        (A.calG k (A.t k n) * (A.t k n + A.As k)) := by
    rw [Ls, A.Bpp_eq_calG_div]
    have hG' := A.calG_ne_zero k n
    field_simp
    simp only [As, calG]
    ring
  rw [gammaS, ← key, invS, Zs, ← hD]
  by_cases hfix : mirror k n = n
  · rw [mirrorBlock_of_fixed hfix, hfix, kron_comm m n]
    field_simp
    ring
  · unfold mirrorBlock
    field_simp
    ring

/-- **Theorem "singlet"** (generic case): for the inverse `invS` of `χ_s`,
`β²(χ_s⁻¹ + B⁻¹) = Γ_s` of Eq. (Gamma-s) (Eq. Gamma-def with `η_s = -1`). -/
theorem gammaS_eq (h : A.SingletGeneric k ξ) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invS k ξ n m + kron n m / A.Bpp k n) = A.gammaS k ξ n m :=
  gammaS_eq_of h.t_add_As_ne h.Ks_ne n m

end Params

end HubbardAtom
