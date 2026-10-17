import HubbardAtomLean.SingletZero

/-!
# Theorem "singlet" at a lattice zero `t_c + Aₛ = 0`

If `t_c + Aₛ = 0`, the orbit `O = {c, c̄}` (with `r ∈ {1, 2}` elements) is where `L_s(tₙ)`
vanishes, and `𝒯_k(-Aₛ)` and `Kₛ` diverge. The note takes limits in `μ`; here the limiting
inverse is written down and checked directly. With `κₙ = 1/(tₙ + Aₛ)` off the orbit (`0` on it),
`ιₙ` the indicator of `O`, `T' = ∑_{l ∉ O} κₗ` and `K' = 1 + 2aT'/β`:

`χ_s⁻¹ = L_s⁻¹𝓜₊ - B⁻¹𝓜₋ + 4a/β² - (2𝒫(-Aₛ)/(βr))(ιₙκₘ + κₙιₘ) + y ιₙιₘ`,
`y = 𝒫(-Aₛ)K'/(ar²) + 2𝒫'(-Aₛ)/(βr)` (`L_s⁻¹ = 0` on `O`).

* `Params.SingletLZ`: the assumptions (`U ≠ 0`, `b ≠ 0`, `t_c + Aₛ = 0`); no condition on `Kₛ`.
* `Params.SingletLZ.chiS_isRightInverse`, `Params.SingletLZ.chiS_isLeftInverse`,
  `Params.SingletLZ.gammaS_eq`: the inverse is two-sided, and `Γ_s = β²(χ_s⁻¹ + B⁻¹)` is
  `gammaSlz`: Eq. (Gamma-s) without the separable term off the orbit, and
  `4a - 2β𝒫(-Aₛ)/(r(tₘ + Aₛ))` (`n ∈ O`, `m ∉ O`) and
  `4a + (2β/r)(𝒫'(-Aₛ) - 𝒫(-Aₛ)/𝒢(-Aₛ)) + β²𝒫(-Aₛ)K'/(ar²)` (`n, m ∈ O`) on it.
-/

open Complex

namespace HubbardAtom

/-- The indicator `ιₙ` of the orbit `{c, c̄}`. -/
noncomputable def orb (k c n : ℤ) : ℂ := if n = c ∨ n = mirror k c then 1 else 0

/-- The number `r` of elements of the orbit `{c, c̄}`. -/
noncomputable def orbSize (k c : ℤ) : ℂ := if mirror k c = c then 1 else 2

theorem orb_mirror (k c n : ℤ) : orb k c (mirror k n) = orb k c n := by
  unfold orb
  congr 1
  apply propext
  constructor
  · rintro (h | h)
    · right
      rw [← h, mirror_mirror]
    · left
      rw [← mirror_mirror k n, h, mirror_mirror]
  · rintro (h | h)
    · right
      rw [h]
    · left
      rw [h, mirror_mirror]

theorem orb_of_mem {k c n : ℤ} (h : n = c ∨ n = mirror k c) : orb k c n = 1 := by
  rw [orb, ite_eq_left h]

theorem orb_of_not_mem {k c n : ℤ} (h : ¬(n = c ∨ n = mirror k c)) : orb k c n = 0 := by
  rw [orb, ite_eq_right h]

/-- `∑ₗ fₗ ιₗ = r f_c` for a mirror-even `f`. -/
theorem hasSum_mul_orb (k c : ℤ) {f : ℤ → ℂ} (hf : ∀ l, f (mirror k l) = f l) :
    HasSum (fun l ↦ f l * orb k c l) (orbSize k c * f c) := by
  have hsum := hasSum_sum_of_ne_finset_zero (L := SummationFilter.unconditional ℤ)
    (s := ({c, mirror k c} : Finset ℤ)) (f := fun l ↦ f l * orb k c l) (fun l hl ↦ by
      simp only [Finset.mem_insert, Finset.mem_singleton] at hl
      rw [orb_of_not_mem hl, mul_zero])
  convert hsum using 1
  by_cases hfix : mirror k c = c
  · rw [hfix, Finset.insert_eq_of_mem (Finset.mem_singleton_self c), Finset.sum_singleton,
      orbSize, ite_eq_left hfix, orb_of_mem (Or.inl rfl)]
    ring
  · rw [Finset.sum_pair (Ne.symm hfix), orbSize, ite_eq_right hfix,
      orb_of_mem (Or.inl rfl), orb_of_mem (Or.inr rfl), hf]
    ring

/-! ### The cancellations at a lattice zero, as rational identities

With `d± = Aₛ + t±`, `t_c = -Aₛ`, `g±(t) = 1/(t - t±)`, `κ = 1/(t + Aₛ)`, the four blocks of
`χ_s invSlz - 1` (`n`, `m` on or off the orbit) are combinations of identity (A4) at `t±`. -/

section LZ

variable {a β γp γm tp tm As Tp Tm Tq r t t' : ℂ}

/-- The separable part of the inverse at the lattice zero, as a function of `ιₙ, κₙ, ιₘ, κₘ`. -/
noncomputable def zlz (a β tp tm As Tq r ιn κn ιm κm : ℂ) : ℂ :=
  4 * a / β ^ 2 - 2 * ((As + tp) * (As + tm)) / (β * r) * (ιn * κm + κn * ιm) +
    ((As + tp) * (As + tm) * (1 + 2 * a * Tq / β) / (a * r ^ 2) -
      2 * ((As + tp) + (As + tm)) / (β * r)) * ιn * ιm

/-- `∑ₗ χ_reg,nl Z_lm` at the lattice zero, by `hasSum_gp`, `hasSum_mul_orb`, `hasSum_g_kappa`. -/
noncomputable def slz (a β γp γm tp tm As Tp Tm Tq r t ιm κm : ℂ) : ℂ :=
  γp / (t - tp) * (4 * a / β ^ 2 * Tp -
      2 * ((As + tp) * (As + tm)) / (β * r) * κm * (r * (1 / (-As - tp))) -
      2 * ((As + tp) * (As + tm)) / (β * r) * ιm * ((Tp - r * (1 / (-As - tp)) - Tq) / (As + tp)) +
      ((As + tp) * (As + tm) * (1 + 2 * a * Tq / β) / (a * r ^ 2) -
        2 * ((As + tp) + (As + tm)) / (β * r)) * ιm * (r * (1 / (-As - tp)))) +
    γm / (t - tm) * (4 * a / β ^ 2 * Tm -
      2 * ((As + tp) * (As + tm)) / (β * r) * κm * (r * (1 / (-As - tm))) -
      2 * ((As + tp) * (As + tm)) / (β * r) * ιm * ((Tm - r * (1 / (-As - tm)) - Tq) / (As + tm)) +
      ((As + tp) * (As + tm) * (1 + 2 * a * Tq / β) / (a * r ^ 2) -
        2 * ((As + tp) + (As + tm)) / (β * r)) * ιm * (r * (1 / (-As - tm))))

variable (hIp : a * (As + tp) / (tp - tm) + γp * (1 + 2 * a * Tp / β) = 0)
  (hIm : -a * (As + tm) / (tp - tm) + γm * (1 + 2 * a * Tm / β) = 0)
  (ha : a ≠ 0) (hβ : β ≠ 0) (hr : r ≠ 0) (hpm : tp - tm ≠ 0) (hdp : As + tp ≠ 0)
  (hdm : As + tm ≠ 0)
include hIp hIm ha hβ hr hpm hdp hdm

/-- Both `n`, `m` off the orbit. -/
theorem woodbury_lz_off (h₁ : t - tp ≠ 0) (h₂ : t - tm ≠ 0) (h₃ : t + As ≠ 0)
    (h₁' : t' - tp ≠ 0) (h₂' : t' - tm ≠ 0) (h₃' : t' + As ≠ 0) :
    β / 2 * (t + As) / ((t - tp) * (t - tm)) * zlz a β tp tm As Tq r 0 (1 / (t + As)) 0
        (1 / (t' + As)) +
      1 / (β / 2 * (t' + As) / ((t' - tp) * (t' - tm))) *
        (γp / (t - tp) / (t' - tp) + γm / (t - tm) / (t' - tm)) +
      slz a β γp γm tp tm As Tp Tm Tq r t 0 (1 / (t' + As)) = 0 := by
  have hdp' : -As - tp ≠ 0 := fun h0 ↦ hdp (by linear_combination -h0)
  have hdm' : -As - tm ≠ 0 := fun h0 ↦ hdm (by linear_combination -h0)
  have key : β / 2 * (t + As) / ((t - tp) * (t - tm)) * zlz a β tp tm As Tq r 0 (1 / (t + As)) 0
        (1 / (t' + As)) +
      1 / (β / 2 * (t' + As) / ((t' - tp) * (t' - tm))) *
        (γp / (t - tp) / (t' - tp) + γm / (t - tm) / (t' - tm)) +
      slz a β γp γm tp tm As Tp Tm Tq r t 0 (1 / (t' + As)) =
      2 / (β * (t - tp)) * (a * (As + tp) / (tp - tm) + γp * (1 + 2 * a * Tp / β)) +
        2 / (β * (t - tm)) * (-a * (As + tm) / (tp - tm) + γm * (1 + 2 * a * Tm / β)) := by
    simp only [zlz, slz]
    field_simp
    ring
  rw [key, hIp, hIm]
  ring

/-- `n` on the orbit (`tₙ = -Aₛ`, `L_s = 0`, `κₙ = 0`), `m` off it. -/
theorem woodbury_lz_row (h₁' : t' - tp ≠ 0) (h₂' : t' - tm ≠ 0) (h₃' : t' + As ≠ 0) :
    1 / (β / 2 * (t' + As) / ((t' - tp) * (t' - tm))) *
        (γp / (-As - tp) / (t' - tp) + γm / (-As - tm) / (t' - tm)) +
      slz a β γp γm tp tm As Tp Tm Tq r (-As) 0 (1 / (t' + As)) = 0 := by
  have hdp' : -As - tp ≠ 0 := fun h0 ↦ hdp (by linear_combination -h0)
  have hdm' : -As - tm ≠ 0 := fun h0 ↦ hdm (by linear_combination -h0)
  have key : 1 / (β / 2 * (t' + As) / ((t' - tp) * (t' - tm))) *
        (γp / (-As - tp) / (t' - tp) + γm / (-As - tm) / (t' - tm)) +
      slz a β γp γm tp tm As Tp Tm Tq r (-As) 0 (1 / (t' + As)) =
      -2 / (β * (As + tp)) * (a * (As + tp) / (tp - tm) + γp * (1 + 2 * a * Tp / β)) +
        -2 / (β * (As + tm)) * (-a * (As + tm) / (tp - tm) + γm * (1 + 2 * a * Tm / β)) := by
    simp only [slz]
    field_simp
    ring
  rw [key, hIp, hIm]
  ring

/-- `n` off the orbit, `m` on it (`L_s⁻¹ = 0`, `κₘ = 0`). -/
theorem woodbury_lz_col (h₁ : t - tp ≠ 0) (h₂ : t - tm ≠ 0) (h₃ : t + As ≠ 0) :
    β / 2 * (t + As) / ((t - tp) * (t - tm)) * zlz a β tp tm As Tq r 0 (1 / (t + As)) 1 0 +
      slz a β γp γm tp tm As Tp Tm Tq r t 1 0 = 0 := by
  have hdp' : -As - tp ≠ 0 := fun h0 ↦ hdp (by linear_combination -h0)
  have hdm' : -As - tm ≠ 0 := fun h0 ↦ hdm (by linear_combination -h0)
  have key : β / 2 * (t + As) / ((t - tp) * (t - tm)) * zlz a β tp tm As Tq r 0 (1 / (t + As)) 1 0 +
      slz a β γp γm tp tm As Tp Tm Tq r t 1 0 =
      (-As * β + 2 * a * r - β * tm) / (a * β * r * (t - tp)) *
          (a * (As + tp) / (tp - tm) + γp * (1 + 2 * a * Tp / β)) +
        (-As * β + 2 * a * r - β * tp) / (a * β * r * (t - tm)) *
          (-a * (As + tm) / (tp - tm) + γm * (1 + 2 * a * Tm / β)) := by
    simp only [zlz, slz]
    field_simp
    ring
  rw [key, hIp, hIm]
  ring

/-- Both `n`, `m` on the orbit: `∑ₗ χ_reg,nl Z_lm = 1/r`. -/
theorem woodbury_lz_orb :
    slz a β γp γm tp tm As Tp Tm Tq r (-As) 1 0 = 1 / r := by
  have hdp' : -As - tp ≠ 0 := fun h0 ↦ hdp (by linear_combination -h0)
  have hdm' : -As - tm ≠ 0 := fun h0 ↦ hdm (by linear_combination -h0)
  have key : slz a β γp γm tp tm As Tp Tm Tq r (-As) 1 0 - 1 / r =
      -(-As * β + 2 * a * r - β * tm) / (a * β * r * (As + tp)) *
          (a * (As + tp) / (tp - tm) + γp * (1 + 2 * a * Tp / β)) +
        -(-As * β + 2 * a * r - β * tp) / (a * β * r * (As + tm)) *
          (-a * (As + tm) / (tp - tm) + γm * (1 + 2 * a * Tm / β)) := by
    simp only [slz]
    field_simp
    ring
  rw [hIp, hIm, mul_zero, mul_zero, add_zero] at key
  linear_combination key

end LZ

section LZ0

/-- `∑ₗ χ_s,reg,nl Z_lm` at a lattice zero with `b = 0` (`chiReg₀`, `t₊ = t₋`), with
`T₁ = ∑ g₀`, `T₂ = ∑ g₀²`. -/
noncomputable def slz0 (a β p₀ tp As T₁ T₂ Tq r t ιm κm : ℂ) : ℂ :=
  (2 * a ^ 2 * β * p₀ - a) * (1 / (t - tp)) *
      (4 * a / β ^ 2 * T₁ - 2 * ((As + tp) * (As + tp)) / (β * r) *
          (κm * (r * (1 / (-As - tp))) + ιm * ((T₁ - r * (1 / (-As - tp)) - Tq) / (As + tp))) +
        ((As + tp) * (As + tp) * (1 + 2 * a * Tq / β) / (a * r ^ 2) -
          2 * ((As + tp) + (As + tp)) / (β * r)) * ιm * (r * (1 / (-As - tp)))) +
    2 * a ^ 3 * ((1 / (t - tp)) ^ 2 *
      (4 * a / β ^ 2 * T₁ - 2 * ((As + tp) * (As + tp)) / (β * r) *
          (κm * (r * (1 / (-As - tp))) + ιm * ((T₁ - r * (1 / (-As - tp)) - Tq) / (As + tp))) +
        ((As + tp) * (As + tp) * (1 + 2 * a * Tq / β) / (a * r ^ 2) -
          2 * ((As + tp) + (As + tp)) / (β * r)) * ιm * (r * (1 / (-As - tp)))) +
      (1 / (t - tp)) *
      (4 * a / β ^ 2 * T₂ - 2 * ((As + tp) * (As + tp)) / (β * r) *
          (κm * (r * (1 / (-As - tp)) ^ 2) + ιm * ((T₂ - r * (1 / (-As - tp)) ^ 2) / (As + tp) -
            (T₁ - r * (1 / (-As - tp))) / (As + tp) ^ 2 + Tq / (As + tp) ^ 2)) +
        ((As + tp) * (As + tp) * (1 + 2 * a * Tq / β) / (a * r ^ 2) -
          2 * ((As + tp) + (As + tp)) / (β * r)) * ιm * (r * (1 / (-As - tp)) ^ 2)))

section Zero

variable {a β p₀ p₁ tp As T₁ T₂ S Tq r t t' : ℂ}
variable (hs : p₀ + 2 * p₁ + p₀ = 1) (hS : S = β * (p₀ - p₁) / (2 * (p₁ + p₀)))
  (hT₁ : T₁ = -S / a) (hT₂ : T₂ = (S ^ 2 - β ^ 2 / 4) / (2 * a ^ 2) - S / (2 * a ^ 3))
  (htp : tp = -a ^ 2) (hAs : As = a ^ 2 * (1 - 8 * p₁)) (ha : a ≠ 0) (hβ : β ≠ 0)
  (hr : r ≠ 0) (hq : p₁ + p₀ ≠ 0) (hd : As + tp ≠ 0)
include hs hS hT₁ hT₂ htp hAs ha hβ hr hq hd

theorem woodbury_lz0_off (h₁ : t - tp ≠ 0) (h₃ : t + As ≠ 0) (h₁' : t' - tp ≠ 0)
    (h₃' : t' + As ≠ 0) :
    β / 2 * (t + As) / ((t - tp) * (t - tp)) * zlz a β tp tp As Tq r 0 (1 / (t + As)) 0
        (1 / (t' + As)) +
      1 / (β / 2 * (t' + As) / ((t' - tp) * (t' - tp))) *
        ((2 * a ^ 2 * β * p₀ - a) * (1 / (t - tp)) * (1 / (t' - tp)) +
          2 * a ^ 3 * ((1 / (t - tp)) ^ 2 * (1 / (t' - tp)) + 1 / (t - tp) * (1 / (t' - tp)) ^ 2)) +
      slz0 a β p₀ tp As T₁ T₂ Tq r t 0 (1 / (t' + As)) = 0 := by
  have hd' : -As - tp ≠ 0 := fun h0 ↦ hd (by linear_combination -h0)
  simp only [zlz, slz0]
  subst htp hT₁ hT₂ hS
  obtain rfl : p₀ = (1 - 2 * p₁) / 2 := by linear_combination hs / 2
  obtain rfl : p₁ = (a ^ 2 - As) / (8 * a ^ 2) := by
    field_simp
    linear_combination hAs
  field_simp
  ring

theorem woodbury_lz0_row (h₁' : t' - tp ≠ 0) (h₃' : t' + As ≠ 0) :
    1 / (β / 2 * (t' + As) / ((t' - tp) * (t' - tp))) *
        ((2 * a ^ 2 * β * p₀ - a) * (1 / (-As - tp)) * (1 / (t' - tp)) +
          2 * a ^ 3 * ((1 / (-As - tp)) ^ 2 * (1 / (t' - tp)) +
            1 / (-As - tp) * (1 / (t' - tp)) ^ 2)) +
      slz0 a β p₀ tp As T₁ T₂ Tq r (-As) 0 (1 / (t' + As)) = 0 := by
  have hd' : -As - tp ≠ 0 := fun h0 ↦ hd (by linear_combination -h0)
  simp only [slz0]
  subst htp hT₁ hT₂ hS
  obtain rfl : p₀ = (1 - 2 * p₁) / 2 := by linear_combination hs / 2
  obtain rfl : p₁ = (a ^ 2 - As) / (8 * a ^ 2) := by
    field_simp
    linear_combination hAs
  field_simp
  ring

theorem woodbury_lz0_col (h₁ : t - tp ≠ 0) (h₃ : t + As ≠ 0) :
    β / 2 * (t + As) / ((t - tp) * (t - tp)) * zlz a β tp tp As Tq r 0 (1 / (t + As)) 1 0 +
      slz0 a β p₀ tp As T₁ T₂ Tq r t 1 0 = 0 := by
  have hd' : -As - tp ≠ 0 := fun h0 ↦ hd (by linear_combination -h0)
  simp only [zlz, slz0]
  subst htp hT₁ hT₂ hS
  obtain rfl : p₀ = (1 - 2 * p₁) / 2 := by linear_combination hs / 2
  obtain rfl : p₁ = (a ^ 2 - As) / (8 * a ^ 2) := by
    field_simp
    linear_combination hAs
  field_simp
  ring

theorem woodbury_lz0_orb : slz0 a β p₀ tp As T₁ T₂ Tq r (-As) 1 0 = 1 / r := by
  have hd' : -As - tp ≠ 0 := fun h0 ↦ hd (by linear_combination -h0)
  simp only [slz0]
  subst htp hT₁ hT₂ hS
  obtain rfl : p₀ = (1 - 2 * p₁) / 2 := by linear_combination hs / 2
  obtain rfl : p₁ = (a ^ 2 - As) / (8 * a ^ 2) := by
    field_simp
    linear_combination hAs
  field_simp
  ring

end Zero

end LZ0

namespace Params

variable (A : Params) (k : ℤ)

/-! ### Lattice facts -/

theorem x_injective {n m : ℤ} (h : A.x n = A.x m) : n = m := by
  have h' := congrArg im h
  simp only [x, add_im, ofReal_im, mul_im, ofReal_re, I_re, I_im, mul_zero, mul_one,
    zero_add] at h'
  have hβ := A.β_pos.ne'
  have hπ := Real.pi_pos.ne'
  rw [ν, ν, add_zero, add_zero, div_left_inj' hβ] at h'
  have h'' := mul_right_cancel₀ hπ h'
  have : (n : ℝ) = m := by linarith
  exact_mod_cast this

theorem t_sub_t (n m : ℤ) : A.t k n - A.t k m = (A.x n - A.x m) * (A.x (mirror k m) - A.x n) := by
  rw [t, t, A.x_mirror]
  ring

theorem t_eq_t_iff {n m : ℤ} : A.t k n = A.t k m ↔ n = m ∨ n = mirror k m := by
  constructor
  · intro h
    have h0 := A.t_sub_t k n m
    rw [h, sub_self] at h0
    rcases mul_eq_zero.mp h0.symm with h1 | h1
    · exact Or.inl (A.x_injective (sub_eq_zero.mp h1))
    · exact Or.inr (A.x_injective (sub_eq_zero.mp h1).symm)
  · rintro (rfl | rfl)
    · rfl
    · exact A.t_mirror k m

variable (c : ℤ)

/-! ### The lattice zero -/

/-- `T' = ∑_{l ∉ O} 1/(t_l + Aₛ)` (the orbit terms are `0`). -/
noncomputable def Tlz : ℂ := ∑' l, 1 / (A.t k l + A.As k)

/-- `K' = 1 + 2aT'/β`. -/
noncomputable def Klz : ℂ := 1 + 2 * A.a * A.Tlz k / A.β

/-- `y = 𝒫(-Aₛ)K'/(ar²) + 2𝒫'(-Aₛ)/(βr)`, with `𝒫'(-Aₛ) = -(d₊ + d₋)`. -/
noncomputable def ylz : ℂ :=
  A.calP k (-A.As k) * A.Klz k / (A.a * orbSize k c ^ 2) -
    2 * ((A.As k + A.tp k) + (A.As k + A.tm k)) / (A.β * orbSize k c)

/-- The separable part of the inverse at the lattice zero. -/
noncomputable def Zlz (n m : ℤ) : ℂ :=
  4 * A.a / A.β ^ 2 - 2 * A.calP k (-A.As k) / (A.β * orbSize k c) *
      (orb k c n * (1 / (A.t k m + A.As k)) + 1 / (A.t k n + A.As k) * orb k c m) +
    A.ylz k c * orb k c n * orb k c m

/-- The inverse of `χ_s` at the lattice zero: `L_s⁻¹𝓜₊ - B⁻¹𝓜₋ + Zlz` (`L_s⁻¹ = 0` on `O`). -/
noncomputable def invSlz (n m : ℤ) : ℂ :=
  mirrorBlock k (fun j ↦ 1 / A.Ls k (A.t k j)) (fun j ↦ -(1 / A.Bpp k j)) n m + A.Zlz k c n m

/-- The closed form at the lattice zero: Eq. (Gamma-s) without the separable term (the first term
vanishes on `O`, where `β²(δₙₘ + δ_{m n̄})/(2Bₙ)` replaces it), plus the orbit terms. -/
noncomputable def gammaSlz (n m : ℤ) : ℂ :=
  A.β * A.a ^ 2 * (A.ρ ^ 2 - 1 + 8 * A.p₁) * A.calP k (A.t k n) /
        (A.calG k (A.t k n) * (A.t k n + A.As k)) * (kron n m + kron m (mirror k n)) +
      orb k c n * (A.β ^ 2 * (kron n m + kron m (mirror k n)) / (2 * A.Bpp k n)) +
    A.β ^ 2 * A.Zlz k c n m

/-- The assumptions of Theorem "singlet" at a lattice zero `t_c + Aₛ = 0` (with `U ≠ 0`). -/
structure SingletLZ : Prop where
  a_ne : (A.a : ℂ) ≠ 0
  t_c : A.t k c + A.As k = 0

variable {A k c}

theorem SingletLZ.zero_iff (h : A.SingletLZ k c) {n : ℤ} :
    A.t k n + A.As k = 0 ↔ n = c ∨ n = mirror k c := by
  rw [← A.t_eq_t_iff k]
  constructor
  · intro h0
    linear_combination h0 - h.t_c
  · intro h0
    rw [h0]
    exact h.t_c

theorem SingletLZ.kappa_orb (h : A.SingletLZ k c) {n : ℤ} (hn : n = c ∨ n = mirror k c) :
    1 / (A.t k n + A.As k) = 0 := by
  rw [h.zero_iff.mpr hn, div_zero]

theorem SingletLZ.ne_of_not_mem (h : A.SingletLZ k c) {n : ℤ} (hn : ¬(n = c ∨ n = mirror k c)) :
    A.t k n + A.As k ≠ 0 := fun h0 ↦ hn (h.zero_iff.mp h0)

theorem SingletLZ.dp_ne (h : A.SingletLZ k c) : A.As k + A.tp k ≠ 0 := by
  intro h0
  apply A.t_sub_tp_ne k c
  linear_combination h.t_c - h0

theorem SingletLZ.dm_ne (h : A.SingletLZ k c) : A.As k + A.tm k ≠ 0 := by
  intro h0
  apply A.t_sub_tm_ne k c
  linear_combination h.t_c - h0

/-- `∑_{l ∉ O} 1/(t_l + Aₛ)` converges: `t_l + Aₛ = -(x_l - x_c)(x_l - x_{c̄})`. -/
theorem SingletLZ.summable_kappa (h : A.SingletLZ k c) :
    Summable (fun l ↦ 1 / (A.t k l + A.As k)) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hc : (2 * Real.pi / A.β * I : ℂ) ≠ 0 :=
    mul_ne_zero (div_ne_zero (mul_ne_zero two_ne_zero
      (ofReal_ne_zero.mpr Real.pi_ne_zero)) hβ) I_ne_zero
  have hs := (summable_inv_mul_linear hc (A.μ + Real.pi / A.β * I - A.x c)
    (A.μ + Real.pi / A.β * I - A.x (mirror k c))).neg
  convert hs using 1
  funext l
  have hfac : A.t k l + A.As k = -((l * (2 * Real.pi / A.β * I) + (A.μ + Real.pi / A.β * I -
      A.x c)) * (l * (2 * Real.pi / A.β * I) + (A.μ + Real.pi / A.β * I - A.x (mirror k c)))) := by
    have h₁ := A.t_sub_t k l c
    have hx : A.x l = l * (2 * Real.pi / A.β * I) + (A.μ + Real.pi / A.β * I) := by
      rw [x, ν]
      push_cast
      ring
    rw [show A.t k l + A.As k = A.t k l - A.t k c + (A.t k c + A.As k) by ring, h.t_c, add_zero,
      h₁, hx]
    ring
  rw [hfac, one_div, inv_neg]

theorem SingletLZ.hasSum_kappa (h : A.SingletLZ k c) :
    HasSum (fun l ↦ 1 / (A.t k l + A.As k)) (A.Tlz k) :=
  h.summable_kappa.hasSum

/-- `∑ₗ κₗ/(tₗ - t₀) = (T₀ - r/(t_c - t₀) - T')/(Aₛ + t₀)` (the orbit is removed). -/
theorem SingletLZ.hasSum_g_kappa (h : A.SingletLZ k c) {t₀ T₀ : ℂ}
    (hT : HasSum (fun n ↦ 1 / (A.t k n - t₀)) T₀) (hd : A.As k + t₀ ≠ 0)
    (ht : ∀ n, A.t k n - t₀ ≠ 0) :
    HasSum (fun l ↦ 1 / (A.t k l - t₀) * (1 / (A.t k l + A.As k)))
      ((T₀ - orbSize k c * (1 / (A.t k c - t₀)) - A.Tlz k) / (A.As k + t₀)) := by
  have ho := hasSum_mul_orb k c (f := fun l ↦ 1 / (A.t k l - t₀)) (fun l ↦ by rw [A.t_mirror])
  convert ((hT.sub h.hasSum_kappa).sub ho).div_const (A.As k + t₀) using 1
  · funext l
    have h₁ := ht l
    by_cases hl : l = c ∨ l = mirror k c
    · rw [h.kappa_orb hl, orb_of_mem hl]
      field_simp
      ring
    · have h₂ := h.ne_of_not_mem hl
      rw [orb_of_not_mem hl]
      field_simp
      ring
  · ring

theorem SingletLZ.t_eq (h : A.SingletLZ k c) {n : ℤ} (hn : n = c ∨ n = mirror k c) :
    A.t k n = -A.As k := by
  linear_combination h.zero_iff.mpr hn

/-- On the orbit, `δₙₘ + δ_{m n̄} = 2/r`. -/
theorem kron_add_kron_orb {n m : ℤ} (hn : n = c ∨ n = mirror k c) (hm : m = c ∨ m = mirror k c) :
    kron n m + kron m (mirror k n) = 2 / orbSize k c := by
  by_cases hfix : mirror k c = c
  · rw [hfix, or_self] at hn hm
    subst hn hm
    rw [hfix, kron_self, orbSize, ite_eq_left hfix]
    ring
  · rw [orbSize, ite_eq_right hfix]
    rcases hn with rfl | rfl <;> rcases hm with rfl | rfl
    · rw [kron_self, kron_of_ne (Ne.symm hfix)]
      ring
    · rw [kron_of_ne (Ne.symm hfix), kron_self]
      ring
    · rw [mirror_mirror, kron_of_ne hfix, kron_self]
      ring
    · rw [kron_self, mirror_mirror, kron_of_ne hfix]
      ring

theorem Zlz_mirror (l m : ℤ) : A.Zlz k c (mirror k l) m = A.Zlz k c l m := by
  unfold Zlz
  rw [orb_mirror, A.t_mirror]

theorem Zlz_symm (n m : ℤ) : A.Zlz k c n m = A.Zlz k c m n := by
  unfold Zlz
  ring

theorem invSlz_symm (n m : ℤ) : A.invSlz k c n m = A.invSlz k c m n := by
  rw [invSlz, invSlz, Zlz_symm,
    mirrorBlock_symm (fun j ↦ by simp only [A.t_mirror]) (fun j ↦ by simp only [A.Bpp_mirror])]

theorem Zlz_eq (n m : ℤ) : A.Zlz k c n m = zlz A.a A.β (A.tp k) (A.tm k) (A.As k) (A.Tlz k)
    (orbSize k c) (orb k c n) (1 / (A.t k n + A.As k)) (orb k c m) (1 / (A.t k m + A.As k)) := by
  rw [Zlz, zlz, ylz, Klz, calP, tp, tm]
  ring

theorem SingletLZ.hasSum_g2_kappa (h : A.SingletLZ k c) {t₀ T₁ T₂ : ℂ}
    (hT₁ : HasSum (fun n ↦ 1 / (A.t k n - t₀)) T₁)
    (hT₂ : HasSum (fun n ↦ 1 / (A.t k n - t₀) ^ 2) T₂) (hd : A.As k + t₀ ≠ 0)
    (ht : ∀ n, A.t k n - t₀ ≠ 0) :
    HasSum (fun l ↦ (1 / (A.t k l - t₀)) ^ 2 * (1 / (A.t k l + A.As k)))
      ((T₂ - orbSize k c * (1 / (A.t k c - t₀)) ^ 2) / (A.As k + t₀) -
        (T₁ - orbSize k c * (1 / (A.t k c - t₀))) / (A.As k + t₀) ^ 2 +
          A.Tlz k / (A.As k + t₀) ^ 2) := by
  have ho₁ := hasSum_mul_orb k c (f := fun l ↦ 1 / (A.t k l - t₀)) (fun l ↦ by rw [A.t_mirror])
  have ho₂ := hasSum_mul_orb k c (f := fun l ↦ (1 / (A.t k l - t₀)) ^ 2)
    (fun l ↦ by rw [A.t_mirror])
  convert (((hT₂.div_const (A.As k + t₀)).sub (hT₁.div_const ((A.As k + t₀) ^ 2))).add
    (h.hasSum_kappa.div_const ((A.As k + t₀) ^ 2))).sub
    ((ho₂.div_const (A.As k + t₀)).sub (ho₁.div_const ((A.As k + t₀) ^ 2))) using 1
  · funext l
    have h₁ := ht l
    by_cases hl : l = c ∨ l = mirror k c
    · rw [h.kappa_orb hl, orb_of_mem hl]
      field_simp
      ring
    · have h₂ := h.ne_of_not_mem hl
      rw [orb_of_not_mem hl]
      field_simp
      ring
  · ring

/-- **Theorem "singlet"** at a lattice zero with `b ≠ 0`, the inverse: `χ_s invSlz = 1`. -/
theorem SingletLZ.chiS_isRightInverse_of_b_ne (h : A.SingletLZ k c) (hb : A.b k ≠ 0) :
    IsRightInverse (A.chiS k) (A.invSlz k c) := by
  intro n m
  have hβ := A.β_ne_zero
  have hr : orbSize k c ≠ 0 := by unfold orbSize; split_ifs <;> norm_num
  have hpm : A.tp k - A.tm k ≠ 0 := by
    rw [tp, tm, show -(A.a : ℂ) ^ 2 + A.a * A.b k - (-A.a ^ 2 - A.a * A.b k) =
      2 * A.a * A.b k by ring]
    exact mul_ne_zero (mul_ne_zero two_ne_zero h.a_ne) hb
  set X := mirrorBlock k (fun j ↦ A.Ls k (A.t k j)) (fun j ↦ -A.Bpp k j) with hX
  set Xi := mirrorBlock k (fun j ↦ 1 / A.Ls k (A.t k j)) (fun j ↦ -(1 / A.Bpp k j)) with hXi
  have h₁ : HasSum (fun l ↦ X n l * Xi l m) (mirrorBlock k
      (fun j ↦ A.Ls k (A.t k j) * (1 / A.Ls k (A.t k j))) (fun j ↦ -A.Bpp k j * -(1 / A.Bpp k j))
      n m) :=
    hasSum_mirrorBlock_mul (k := k) (B := fun j ↦ A.Ls k (A.t k j))
      (L := fun j ↦ -A.Bpp k j) (B' := fun j ↦ 1 / A.Ls k (A.t k j))
      (L' := fun j ↦ -(1 / A.Bpp k j)) (fun j ↦ by simp only [A.t_mirror])
      (fun j ↦ by simp only [A.Bpp_mirror]) n m
  have h₂ : HasSum (fun l ↦ X n l * A.Zlz k c l m) (A.Ls k (A.t k n) * A.Zlz k c n m) :=
    hasSum_mirrorBlock_mul_even (v := fun l ↦ A.Zlz k c l m) (fun l ↦ A.Zlz_mirror l m) n
  have h₃ : HasSum (fun l ↦ A.chiReg k n l * Xi l m) (1 / A.Ls k (A.t k m) * A.chiReg k n m) := by
    have := hasSum_mirrorBlock_mul_even (k := k) (B := fun j ↦ 1 / A.Ls k (A.t k j))
      (L := fun j ↦ -(1 / A.Bpp k j)) (v := fun l ↦ A.chiReg k n l) (fun l ↦ A.chiReg_mirror n l) m
    convert this using 1
    funext l
    rw [hXi, mirrorBlock_symm (fun j ↦ by simp only [A.t_mirror])
      (fun j ↦ by simp only [A.Bpp_mirror]) l m, mul_comm]
  -- the lattice sums of the separable part
  have htc : A.t k c = -A.As k := h.t_eq (Or.inl rfl)
  have hgp := A.hasSum_gp k
  have hgm := A.hasSum_gm k
  have hop := hasSum_mul_orb k c (f := fun l ↦ 1 / (A.t k l - A.tp k)) (fun l ↦ by rw [A.t_mirror])
  have hom := hasSum_mul_orb k c (f := fun l ↦ 1 / (A.t k l - A.tm k)) (fun l ↦ by rw [A.t_mirror])
  have hκp := h.hasSum_g_kappa hgp h.dp_ne (A.t_sub_tp_ne k)
  have hκm := h.hasSum_g_kappa hgm h.dm_ne (A.t_sub_tm_ne k)
  simp only [htc] at hop hom hκp hκm
  set c₀ := 2 * A.calP k (-A.As k) / (A.β * orbSize k c) with hc₀
  set κm := 1 / (A.t k m + A.As k) with hκm_def
  set ιm := orb k c m with hιm
  have hsp := (((hgp.mul_left (4 * (A.a : ℂ) / (A.β : ℂ) ^ 2)).sub
    (hop.mul_left (c₀ * κm))).sub (hκp.mul_left (c₀ * ιm))).add (hop.mul_left (A.ylz k c * ιm))
  have hsm := (((hgm.mul_left (4 * (A.a : ℂ) / (A.β : ℂ) ^ 2)).sub
    (hom.mul_left (c₀ * κm))).sub (hκm.mul_left (c₀ * ιm))).add (hom.mul_left (A.ylz k c * ιm))
  have h₄ : HasSum (fun l ↦ A.chiReg k n l * A.Zlz k c l m)
      (slz A.a A.β (A.γp k) (A.γm k) (A.tp k) (A.tm k) (A.As k) (A.Tpp k (A.a - A.b k / 2))
        (A.Tpp k (-A.a - A.b k / 2)) (A.Tlz k) (orbSize k c) (A.t k n) ιm κm) := by
    convert (hsp.mul_left (A.γp k / (A.t k n - A.tp k))).add
      (hsm.mul_left (A.γm k / (A.t k n - A.tm k))) using 1
    · funext l
      rw [chiReg, Zlz]
      ring
    · rw [slz, ylz, Klz, hc₀, calP, tp, tm]
      ring
  convert ((h₁.add h₂).add h₃).add h₄ using 1
  · funext l
    rw [A.chiS_eq k hb, invSlz]
    ring
  · -- the remainder vanishes, block by block
    have hIp := A.singlet_Ip k h.a_ne hb
    have hIm := A.singlet_Im k h.a_ne hb
    have hLs : ∀ j, A.Ls k (A.t k j) = A.β / 2 * (A.t k j + A.As k) /
        ((A.t k j - A.tp k) * (A.t k j - A.tm k)) := fun j ↦ by rw [Ls, A.calP_eq_mul]
    have hmb : ∀ j, (fun j ↦ -A.Bpp k j * -(1 / A.Bpp k j)) j = 1 := fun j ↦ by
      simp only [neg_mul_neg]
      exact mul_one_div_cancel (A.Bpp_ne_zero k j)
    by_cases hn : n = c ∨ n = mirror k c <;> by_cases hm : m = c ∨ m = mirror k c
    · -- both on the orbit
      have hLn : A.Ls k (A.t k n) = 0 := by rw [Ls, h.zero_iff.mpr hn]; ring
      have hLm : A.Ls k (A.t k m) = 0 := by rw [Ls, h.zero_iff.mpr hm]; ring
      have hk := kron_add_kron_orb hn hm
      have hS := woodbury_lz_orb (γp := A.γp k) (γm := A.γm k) (Tq := A.Tlz k) hIp hIm h.a_ne
        hβ hr hpm h.dp_ne h.dm_ne
      rw [hιm, hκm_def, orb_of_mem hm, h.kappa_orb hm]
      simp only [mirrorBlock]
      simp only [hLn, hLm, hmb, zero_mul, div_zero, zero_add, zero_sub, mul_zero, add_zero]
      rw [h.t_eq hn, hS]
      linear_combination (1 / 2 : ℂ) * hk
    · -- `n` on the orbit, `m` off it
      have hLn : A.Ls k (A.t k n) = 0 := by rw [Ls, h.zero_iff.mpr hn]; ring
      have hnm : n ≠ m := fun h0 ↦ hm (by rw [← h0]; exact hn)
      have hnm' : m ≠ mirror k n := by
        intro h0
        apply hm
        rcases hn with rfl | rfl
        · exact Or.inr h0
        · left
          rw [h0, mirror_mirror]
      have hS := woodbury_lz_row (γp := A.γp k) (γm := A.γm k) (Tq := A.Tlz k)
        (r := orbSize k c) hIp hIm h.a_ne hβ hr hpm h.dp_ne h.dm_ne (A.t_sub_tp_ne k m)
        (A.t_sub_tm_ne k m) (h.ne_of_not_mem hm)
      rw [hιm, hκm_def, orb_of_not_mem hm, kron_of_ne hnm]
      simp only [mirrorBlock]
      rw [kron_of_ne hnm, kron_of_ne hnm', hLn, hLs m]
      simp only [chiReg, h.t_eq hn, hmb, zero_mul, mul_zero, zero_add, add_zero]
      linear_combination -hS
    · -- `n` off the orbit, `m` on it
      have hLm : A.Ls k (A.t k m) = 0 := by rw [Ls, h.zero_iff.mpr hm]; ring
      have hLn' : A.Ls k (A.t k n) * (1 / A.Ls k (A.t k n)) = 1 := by
        rw [hLs n]
        have := h.ne_of_not_mem hn
        have := A.t_sub_tp_ne k n
        have := A.t_sub_tm_ne k n
        field_simp
      have hnm : n ≠ m := fun h0 ↦ hn (by rw [h0]; exact hm)
      have hS := woodbury_lz_col (γp := A.γp k) (γm := A.γm k) (Tp := A.Tpp k (A.a - A.b k / 2))
        (Tm := A.Tpp k (-A.a - A.b k / 2)) (Tq := A.Tlz k) (r := orbSize k c) hIp hIm h.a_ne hβ
        hr hpm h.dp_ne h.dm_ne (A.t_sub_tp_ne k n) (A.t_sub_tm_ne k n) (h.ne_of_not_mem hn)
      rw [hιm, hκm_def, orb_of_mem hm, h.kappa_orb hm, kron_of_ne hnm, A.Zlz_eq,
        orb_of_not_mem hn, orb_of_mem hm, h.kappa_orb hm]
      simp only [mirrorBlock]
      rw [hLn', kron_of_ne hnm, hLm, hLs n]
      simp only [hmb, div_zero, zero_mul, mul_zero, add_zero, sub_self, zero_div]
      linear_combination -hS
    · -- both off the orbit
      have hLn' : A.Ls k (A.t k n) * (1 / A.Ls k (A.t k n)) = 1 := by
        rw [hLs n]
        have := h.ne_of_not_mem hn
        have := A.t_sub_tp_ne k n
        have := A.t_sub_tm_ne k n
        field_simp
      have hS := woodbury_lz_off (γp := A.γp k) (γm := A.γm k) (Tq := A.Tlz k)
        (r := orbSize k c) hIp hIm h.a_ne hβ hr hpm h.dp_ne h.dm_ne (A.t_sub_tp_ne k n)
        (A.t_sub_tm_ne k n) (h.ne_of_not_mem hn) (A.t_sub_tp_ne k m) (A.t_sub_tm_ne k m)
        (h.ne_of_not_mem hm)
      rw [hιm, hκm_def, orb_of_not_mem hm, A.Zlz_eq, orb_of_not_mem hn, orb_of_not_mem hm]
      simp only [mirrorBlock]
      rw [hLn', hLs n, hLs m]
      simp only [hmb, chiReg, sub_self, zero_div, zero_mul, add_zero]
      linear_combination -hS

/-- **Theorem "singlet"** at a lattice zero with `b = 0` (`μ = 0`, `k = 0`), the inverse. -/
theorem SingletLZ.chiS_isRightInverse_of_b_eq (h : A.SingletLZ k c) (hb : A.b k = 0) :
    IsRightInverse (A.chiS k) (A.invSlz k c) := by
  intro n m
  have hβ := A.β_ne_zero
  have ha := h.a_ne
  have hr : orbSize k c ≠ 0 := by unfold orbSize; split_ifs <;> norm_num
  set X := mirrorBlock k (fun j ↦ A.Ls k (A.t k j)) (fun j ↦ -A.Bpp k j) with hX
  set Xi := mirrorBlock k (fun j ↦ 1 / A.Ls k (A.t k j)) (fun j ↦ -(1 / A.Bpp k j)) with hXi
  have h₁ : HasSum (fun l ↦ X n l * Xi l m) (mirrorBlock k
      (fun j ↦ A.Ls k (A.t k j) * (1 / A.Ls k (A.t k j))) (fun j ↦ -A.Bpp k j * -(1 / A.Bpp k j))
      n m) :=
    hasSum_mirrorBlock_mul (k := k) (B := fun j ↦ A.Ls k (A.t k j))
      (L := fun j ↦ -A.Bpp k j) (B' := fun j ↦ 1 / A.Ls k (A.t k j))
      (L' := fun j ↦ -(1 / A.Bpp k j)) (fun j ↦ by simp only [A.t_mirror])
      (fun j ↦ by simp only [A.Bpp_mirror]) n m
  have h₂ : HasSum (fun l ↦ X n l * A.Zlz k c l m) (A.Ls k (A.t k n) * A.Zlz k c n m) :=
    hasSum_mirrorBlock_mul_even (v := fun l ↦ A.Zlz k c l m) (fun l ↦ A.Zlz_mirror l m) n
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
  have htc : A.t k c = -A.As k := h.t_eq (Or.inl rfl)
  have hT₁ := A.hasSum_gp k
  have ha2 : (A.a : ℂ) - A.b k / 2 ≠ 0 := by rw [hb, zero_div, sub_zero]; exact ha
  have hT₂ := A.hasSum_one_div_t_sub_sq (k := k) (t := A.tp k) (ξ := A.a - A.b k / 2)
    (by rw [tp]; ring) ha2 (fun n h0 ↦ A.t_sub_tp_ne k n (by rw [h0, sub_self]))
  have ho₁ := hasSum_mul_orb k c (f := fun l ↦ 1 / (A.t k l - A.tp k)) (fun l ↦ by rw [A.t_mirror])
  have ho₂ := hasSum_mul_orb k c (f := fun l ↦ (1 / (A.t k l - A.tp k)) ^ 2)
    (fun l ↦ by rw [A.t_mirror])
  have hκ₁ := h.hasSum_g_kappa hT₁ h.dp_ne (A.t_sub_tp_ne k)
  have hκ₂ := h.hasSum_g2_kappa hT₁ hT₂ h.dp_ne (A.t_sub_tp_ne k)
  simp only [htc] at ho₁ ho₂ hκ₁ hκ₂
  set T₂ := (A.S k (A.μ + (A.a - A.b k / 2)) ^ 2 - (A.β : ℂ) ^ 2 / 4) /
    (2 * (A.a - A.b k / 2) ^ 2) - A.S k (A.μ + (A.a - A.b k / 2)) / (2 * (A.a - A.b k / 2) ^ 3)
    with hT₂_def
  set c₀ := 2 * ((A.As k + A.tp k) * (A.As k + A.tp k)) / (A.β * orbSize k c) with hc₀
  set y := (A.As k + A.tp k) * (A.As k + A.tp k) * A.Klz k / (A.a * orbSize k c ^ 2) -
    2 * ((A.As k + A.tp k) + (A.As k + A.tp k)) / (A.β * orbSize k c) with hy
  set κm := 1 / (A.t k m + A.As k) with hκm_def
  set ιm := orb k c m with hιm
  have htm : A.tm k = A.tp k := by rw [tm, tp, hb]; ring
  have hZ : ∀ l, A.Zlz k c l m = 4 * A.a / A.β ^ 2 - c₀ * (orb k c l * κm +
      1 / (A.t k l + A.As k) * ιm) + y * orb k c l * ιm := fun l ↦ by
    simp only [Zlz, ylz, calP, hc₀, hy, htm, tp, hb]
    ring
  have hs₁ := (((hT₁.mul_left (4 * (A.a : ℂ) / (A.β : ℂ) ^ 2)).sub
    ((ho₁.mul_left κm).add (hκ₁.mul_left ιm) |>.mul_left c₀)).add (ho₁.mul_left (y * ιm)))
  have hs₂ := (((hT₂.mul_left (4 * (A.a : ℂ) / (A.β : ℂ) ^ 2)).sub
    ((ho₂.mul_left κm).add (hκ₂.mul_left ιm) |>.mul_left c₀)).add (ho₂.mul_left (y * ιm)))
  have h₄ : HasSum (fun l ↦ A.chiReg₀ k n l * A.Zlz k c l m)
      (slz0 A.a A.β A.p₀ (A.tp k) (A.As k) (A.Tpp k (A.a - A.b k / 2)) T₂ (A.Tlz k)
        (orbSize k c) (A.t k n) ιm κm) := by
    convert (((hs₁.mul_left ((2 * A.a ^ 2 * A.β * A.p₀ - A.a) * A.g₀ k n)).add
      (hs₁.mul_left (2 * A.a ^ 3 * A.g₀ k n ^ 2))).add
      (hs₂.mul_left (2 * A.a ^ 3 * A.g₀ k n))) using 1
    · funext l
      rw [chiReg₀, hZ, g₀, g₀]
      simp only [one_div, inv_pow]
      ring
    · rw [slz0, hc₀, hy, Klz, g₀]
      ring
  convert ((h₁.add h₂).add h₃).add h₄ using 1
  · funext l
    rw [A.chiS_eq_zero k hb, invSlz]
    ring
  · -- the remainder vanishes, block by block
    obtain ⟨hμ, rfl⟩ := (A.b_eq_zero_iff k).mp hb
    have hp₂ := p₂_eq_p₀_of_mu hμ
    have hs : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₀ = 1 := by
      have := A.weights_sum
      rw [hp₂] at this
      exact_mod_cast this
    have hq : (A.p₁ : ℂ) + A.p₀ ≠ 0 := by exact_mod_cast (add_pos A.p₁_pos A.p₀_pos).ne'
    have hρ : (A.ρ : ℂ) = 0 := by rw [ρ, hp₂, sub_self, ofReal_zero]
    have hSa : A.S 0 (A.μ + (A.a - A.b 0 / 2)) =
        A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀)) := by
      rw [hb, zero_div, sub_zero, hμ, ofReal_zero, zero_add, A.S₀_a, hp₂]
    have hTp : A.Tpp 0 (A.a - A.b 0 / 2) =
        -(A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀))) / A.a := by
      rw [Tpp, ite_eq_right ha2, hSa, hb, zero_div, sub_zero]
    have hT₂' : T₂ = ((A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀))) ^ 2 - (A.β : ℂ) ^ 2 / 4) /
        (2 * A.a ^ 2) - A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀)) / (2 * A.a ^ 3) := by
      rw [hT₂_def, hSa, hb, zero_div, sub_zero]
    have htp : A.tp 0 = -A.a ^ 2 := by rw [tp, hb, mul_zero, add_zero]
    have hAs0 : A.As 0 = A.a ^ 2 * (1 - 8 * A.p₁) := by rw [As, hρ, hb]; ring
    have hLs : ∀ j, A.Ls 0 (A.t 0 j) = A.β / 2 * (A.t 0 j + A.As 0) /
        ((A.t 0 j - A.tp 0) * (A.t 0 j - A.tp 0)) := fun j ↦ by rw [Ls, A.calP_eq_mul, htm]
    have hmb : ∀ j, (fun j ↦ -A.Bpp 0 j * -(1 / A.Bpp 0 j)) j = 1 := fun j ↦ by
      simp only [neg_mul_neg]
      exact mul_one_div_cancel (A.Bpp_ne_zero 0 j)
    have hd := h.dp_ne
    by_cases hn : n = c ∨ n = mirror 0 c <;> by_cases hm : m = c ∨ m = mirror 0 c
    · have hLn : A.Ls 0 (A.t 0 n) = 0 := by rw [Ls, h.zero_iff.mpr hn]; ring
      have hLm : A.Ls 0 (A.t 0 m) = 0 := by rw [Ls, h.zero_iff.mpr hm]; ring
      have hk := kron_add_kron_orb hn hm
      have hS := woodbury_lz0_orb (Tq := A.Tlz 0) hs rfl hTp hT₂' htp hAs0 ha hβ hr hq hd
      rw [hιm, hκm_def, orb_of_mem hm, h.kappa_orb hm]
      simp only [mirrorBlock, hLn, hLm, hmb, zero_mul, div_zero, zero_add, zero_sub, mul_zero,
        add_zero]
      rw [h.t_eq hn, hS]
      linear_combination (1 / 2 : ℂ) * hk
    · have hLn : A.Ls 0 (A.t 0 n) = 0 := by rw [Ls, h.zero_iff.mpr hn]; ring
      have hnm : n ≠ m := fun h0 ↦ hm (by rw [← h0]; exact hn)
      have hnm' : m ≠ mirror 0 n := by
        intro h0
        apply hm
        rcases hn with rfl | rfl
        · exact Or.inr h0
        · left
          rw [h0, mirror_mirror]
      have hS := woodbury_lz0_row (Tq := A.Tlz 0) (r := orbSize 0 c) hs rfl hTp hT₂' htp hAs0 ha
        hβ hr hq hd (A.t_sub_tp_ne 0 m) (h.ne_of_not_mem hm)
      rw [hιm, hκm_def, orb_of_not_mem hm, kron_of_ne hnm]
      simp only [mirrorBlock]
      rw [kron_of_ne hnm, kron_of_ne hnm', hLn, hLs m]
      simp only [chiReg₀, g₀, h.t_eq hn, hmb, zero_mul, mul_zero, zero_add, add_zero]
      linear_combination -hS
    · have hLm : A.Ls 0 (A.t 0 m) = 0 := by rw [Ls, h.zero_iff.mpr hm]; ring
      have hLn' : A.Ls 0 (A.t 0 n) * (1 / A.Ls 0 (A.t 0 n)) = 1 := by
        rw [hLs n]
        have := h.ne_of_not_mem hn
        have := A.t_sub_tp_ne 0 n
        field_simp
      have hnm : n ≠ m := fun h0 ↦ hn (by rw [h0]; exact hm)
      have hS := woodbury_lz0_col (T₁ := A.Tpp 0 (A.a - A.b 0 / 2)) (T₂ := T₂) (Tq := A.Tlz 0)
        (r := orbSize 0 c) hs rfl hTp hT₂' htp hAs0 ha hβ hr hq hd (A.t_sub_tp_ne 0 n)
        (h.ne_of_not_mem hn)
      rw [hιm, hκm_def, orb_of_mem hm, h.kappa_orb hm, kron_of_ne hnm, A.Zlz_eq,
        orb_of_not_mem hn, orb_of_mem hm, h.kappa_orb hm, htm]
      simp only [mirrorBlock]
      rw [hLn', kron_of_ne hnm, hLm, hLs n]
      simp only [hmb, div_zero, zero_mul, mul_zero, add_zero, sub_self, zero_div]
      linear_combination -hS
    · have hLn' : A.Ls 0 (A.t 0 n) * (1 / A.Ls 0 (A.t 0 n)) = 1 := by
        rw [hLs n]
        have := h.ne_of_not_mem hn
        have := A.t_sub_tp_ne 0 n
        field_simp
      have hS := woodbury_lz0_off (Tq := A.Tlz 0) (r := orbSize 0 c) hs rfl hTp hT₂' htp hAs0 ha
        hβ hr hq hd (A.t_sub_tp_ne 0 n) (h.ne_of_not_mem hn) (A.t_sub_tp_ne 0 m)
        (h.ne_of_not_mem hm)
      rw [hιm, hκm_def, orb_of_not_mem hm, A.Zlz_eq, orb_of_not_mem hn, orb_of_not_mem hm, htm]
      simp only [mirrorBlock]
      rw [hLn', hLs n, hLs m]
      simp only [hmb, chiReg₀, g₀, sub_self, zero_div, zero_mul, add_zero]
      linear_combination -hS

/-- **Theorem "singlet"** at a lattice zero, the inverse: `χ_s invSlz = 1`. -/
theorem SingletLZ.chiS_isRightInverse (h : A.SingletLZ k c) :
    IsRightInverse (A.chiS k) (A.invSlz k c) := by
  by_cases hb : A.b k = 0
  · exact h.chiS_isRightInverse_of_b_eq hb
  · exact h.chiS_isRightInverse_of_b_ne hb

theorem SingletLZ.chiS_isLeftInverse (h : A.SingletLZ k c) :
    IsLeftInverse (A.chiS k) (A.invSlz k c) := by
  intro n m
  have := h.chiS_isRightInverse m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [chiS_symm' (k := k) l m, invSlz_symm n l, mul_comm]

/-- **Theorem "singlet"** at a lattice zero, the closed form: `β²(χ_s⁻¹ + B⁻¹) = gammaSlz`. -/
theorem SingletLZ.gammaS_eq (h : A.SingletLZ k c) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invSlz k c n m + kron n m / A.Bpp k n) = A.gammaSlz k c n m := by
  have hβ := A.β_ne_zero
  have hP := A.calP_ne_zero k n
  have hB := A.Bpp_ne_zero k n
  rw [invSlz, gammaSlz]
  by_cases hn : n = c ∨ n = mirror k c
  · -- on the orbit `L_s = 0` and the first term of Eq. (Gamma-s) is `0`
    have hLn : A.Ls k (A.t k n) = 0 := by rw [Ls, h.zero_iff.mpr hn]; ring
    rw [orb_of_mem hn, h.zero_iff.mpr hn, mul_zero, div_zero, zero_mul, zero_add]
    by_cases hfix : mirror k n = n
    · rw [mirrorBlock_of_fixed hfix, hfix, kron_comm m n, hLn, div_zero]
      field_simp
      ring
    · simp only [mirrorBlock, hLn, div_zero]
      field_simp
      ring
  · have hA := h.ne_of_not_mem hn
    have key : (A.β : ℂ) ^ 2 * (1 / A.Ls k (A.t k n) + 1 / A.Bpp k n) / 2 =
        A.β * A.a ^ 2 * (A.ρ ^ 2 - 1 + 8 * A.p₁) * A.calP k (A.t k n) /
          (A.calG k (A.t k n) * (A.t k n + A.As k)) := by
      rw [Ls, A.Bpp_eq_calG_div]
      have hG' := A.calG_ne_zero k n
      field_simp
      simp only [As, calG]
      ring
    rw [orb_of_not_mem hn, zero_mul, add_zero, ← key]
    by_cases hfix : mirror k n = n
    · rw [mirrorBlock_of_fixed hfix, hfix, kron_comm m n]
      field_simp
      ring
    · simp only [mirrorBlock]
      field_simp
      ring

end Params

end HubbardAtom
