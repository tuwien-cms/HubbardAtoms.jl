import HubbardAtomLean.FieldLRes

/-!
# Partial fractions in `s = v²`

The completed defect is `N(v²)/(βE₈(v))` with `N` of degree at most three in `v²`
(`FieldLConn.lean`); with the four roots `s_j = u_j²` of `E₈` it splits into
`∑ⱼ N(s_j)/(βE₈'(s_j)(v² - s_j))`.

* `interp4`: Lagrange interpolation of a cubic at four distinct nodes.
* `quartic_eq_prod`: a monic quartic with four distinct roots is `∏ⱼ(x - s_j)`.
* `FParams.E8L_eq_quartic`, `E8L_eq_prod`: `E₈(v)` as a monic quartic in `v²`, hence
  `∏ⱼ(v² - s_j)`.
* `pf4`: the partial fractions of `N(x)/∏ⱼ(x - s_j)`.
-/

open Complex

namespace HubbardAtom

/-- **Lagrange interpolation** of a cubic at four distinct nodes. -/
theorem interp4 {N : ℂ → ℂ} {c₀ c₁ c₂ c₃ : ℂ}
    (hN : ∀ y, N y = c₀ + c₁ * y + c₂ * y ^ 2 + c₃ * y ^ 3)
    {s₁ s₂ s₃ s₄ : ℂ} (h₁₂ : s₁ - s₂ ≠ 0) (h₁₃ : s₁ - s₃ ≠ 0) (h₁₄ : s₁ - s₄ ≠ 0)
    (h₂₃ : s₂ - s₃ ≠ 0) (h₂₄ : s₂ - s₄ ≠ 0) (h₃₄ : s₃ - s₄ ≠ 0) (x : ℂ) :
    N x = N s₁ * ((x - s₂) * (x - s₃) * (x - s₄)) / ((s₁ - s₂) * (s₁ - s₃) * (s₁ - s₄)) +
      N s₂ * ((x - s₁) * (x - s₃) * (x - s₄)) / ((s₂ - s₁) * (s₂ - s₃) * (s₂ - s₄)) +
      N s₃ * ((x - s₁) * (x - s₂) * (x - s₄)) / ((s₃ - s₁) * (s₃ - s₂) * (s₃ - s₄)) +
      N s₄ * ((x - s₁) * (x - s₂) * (x - s₃)) / ((s₄ - s₁) * (s₄ - s₂) * (s₄ - s₃)) := by
  have h₂₁ : s₂ - s₁ ≠ 0 := fun h ↦ h₁₂ (by linear_combination -h)
  have h₃₁ : s₃ - s₁ ≠ 0 := fun h ↦ h₁₃ (by linear_combination -h)
  have h₄₁ : s₄ - s₁ ≠ 0 := fun h ↦ h₁₄ (by linear_combination -h)
  have h₃₂ : s₃ - s₂ ≠ 0 := fun h ↦ h₂₃ (by linear_combination -h)
  have h₄₂ : s₄ - s₂ ≠ 0 := fun h ↦ h₂₄ (by linear_combination -h)
  have h₄₃ : s₄ - s₃ ≠ 0 := fun h ↦ h₃₄ (by linear_combination -h)
  rw [hN, hN, hN, hN, hN]
  field_simp
  ring

/-- **A monic quartic with four distinct roots** is `∏ⱼ(x - s_j)`. -/
theorem quartic_eq_prod {Q : ℂ → ℂ} {b₀ b₁ b₂ b₃ : ℂ}
    (hQ : ∀ y, Q y = y ^ 4 + b₃ * y ^ 3 + b₂ * y ^ 2 + b₁ * y + b₀)
    {s₁ s₂ s₃ s₄ : ℂ} (h₁₂ : s₁ - s₂ ≠ 0) (h₁₃ : s₁ - s₃ ≠ 0) (h₁₄ : s₁ - s₄ ≠ 0)
    (h₂₃ : s₂ - s₃ ≠ 0) (h₂₄ : s₂ - s₄ ≠ 0) (h₃₄ : s₃ - s₄ ≠ 0)
    (r₁ : Q s₁ = 0) (r₂ : Q s₂ = 0) (r₃ : Q s₃ = 0) (r₄ : Q s₄ = 0) (x : ℂ) :
    Q x = (x - s₁) * (x - s₂) * (x - s₃) * (x - s₄) := by
  set d : ℂ → ℂ := fun y ↦ Q y - (y - s₁) * (y - s₂) * (y - s₃) * (y - s₄) with hd
  have hd' : ∀ y, d y = (b₀ - s₁ * s₂ * s₃ * s₄) +
      (b₁ + (s₁ * s₂ * s₃ + s₁ * s₂ * s₄ + s₁ * s₃ * s₄ + s₂ * s₃ * s₄)) * y +
      (b₂ - (s₁ * s₂ + s₁ * s₃ + s₁ * s₄ + s₂ * s₃ + s₂ * s₄ + s₃ * s₄)) * y ^ 2 +
      (b₃ + (s₁ + s₂ + s₃ + s₄)) * y ^ 3 := by
    intro y
    simp only [hd, hQ]
    ring
  have h := interp4 hd' h₁₂ h₁₃ h₁₄ h₂₃ h₂₄ h₃₄ x
  have z₁ : d s₁ = 0 := by simp only [hd, r₁]; ring
  have z₂ : d s₂ = 0 := by simp only [hd, r₂]; ring
  have z₃ : d s₃ = 0 := by simp only [hd, r₃]; ring
  have z₄ : d s₄ = 0 := by simp only [hd, r₄]; ring
  rw [z₁, z₂, z₃, z₄] at h
  simp only [hd, zero_mul, zero_div, add_zero] at h
  exact sub_eq_zero.mp h

/-- **Partial fractions** of a cubic over four distinct linear factors. -/
theorem pf4 {N : ℂ → ℂ} {c₀ c₁ c₂ c₃ : ℂ} (hN : ∀ y, N y = c₀ + c₁ * y + c₂ * y ^ 2 + c₃ * y ^ 3)
    {s₁ s₂ s₃ s₄ : ℂ} (h₁₂ : s₁ - s₂ ≠ 0) (h₁₃ : s₁ - s₃ ≠ 0) (h₁₄ : s₁ - s₄ ≠ 0)
    (h₂₃ : s₂ - s₃ ≠ 0) (h₂₄ : s₂ - s₄ ≠ 0) (h₃₄ : s₃ - s₄ ≠ 0) {x : ℂ}
    (x₁ : x - s₁ ≠ 0) (x₂ : x - s₂ ≠ 0) (x₃ : x - s₃ ≠ 0) (x₄ : x - s₄ ≠ 0) :
    N x / ((x - s₁) * (x - s₂) * (x - s₃) * (x - s₄)) =
      N s₁ / ((s₁ - s₂) * (s₁ - s₃) * (s₁ - s₄)) / (x - s₁) +
      N s₂ / ((s₂ - s₁) * (s₂ - s₃) * (s₂ - s₄)) / (x - s₂) +
      N s₃ / ((s₃ - s₁) * (s₃ - s₂) * (s₃ - s₄)) / (x - s₃) +
      N s₄ / ((s₄ - s₁) * (s₄ - s₂) * (s₄ - s₃)) / (x - s₄) := by
  have h₂₁ : s₂ - s₁ ≠ 0 := fun h ↦ h₁₂ (by linear_combination -h)
  have h₃₁ : s₃ - s₁ ≠ 0 := fun h ↦ h₁₃ (by linear_combination -h)
  have h₄₁ : s₄ - s₁ ≠ 0 := fun h ↦ h₁₄ (by linear_combination -h)
  have h₃₂ : s₃ - s₂ ≠ 0 := fun h ↦ h₂₃ (by linear_combination -h)
  have h₄₂ : s₄ - s₂ ≠ 0 := fun h ↦ h₂₄ (by linear_combination -h)
  have h₄₃ : s₄ - s₃ ≠ 0 := fun h ↦ h₃₄ (by linear_combination -h)
  rw [interp4 hN h₁₂ h₁₃ h₁₄ h₂₃ h₂₄ h₃₄ x]
  field_simp

namespace FParams

variable (F : FParams) (k : ℤ)

/-- The coefficients of `E₈` as a monic quartic in `s = v²`: with
`H_r(v)H_r(-v) = s² + A_r s + q₀²`, `A_r = 2q₀ - 4a²r²`. -/
noncomputable def E8b (j : Fin 4) : ℂ :=
  let Au := 2 * F.q₀L k - 4 * F.a ^ 2 * (F.rUp : ℂ) ^ 2
  let Ad := 2 * F.q₀L k - 4 * F.a ^ 2 * (F.rDn : ℂ) ^ 2
  let q := F.q₀L k
  let c := 4 * F.a ^ 2 * (F.rUp : ℂ) * F.rDn
  match j with
  | 3 => Au + Ad
  | 2 => Au * Ad + 2 * q ^ 2 - 2 * F.lamL ^ 2 * (F.piS + F.piP)
  | 1 => q ^ 2 * (Au + Ad) - 2 * F.lamL ^ 2 * (2 * q * (F.piS + F.piP) + c * (F.piS - F.piP))
  | 0 => q ^ 4 - 2 * F.lamL ^ 2 * q ^ 2 * (F.piS + F.piP) + F.lamL ^ 4 * (F.piP - F.piS) ^ 2

/-- `E₈(v)` is a monic quartic in `v²`. -/
theorem E8L_eq_quartic (v : ℂ) :
    F.E8L k v = (v ^ 2) ^ 4 + F.E8b k 3 * (v ^ 2) ^ 3 + F.E8b k 2 * (v ^ 2) ^ 2 +
      F.E8b k 1 * v ^ 2 + F.E8b k 0 := by
  simp only [E8L, E8b, HL]
  ring

end FParams

end HubbardAtom
