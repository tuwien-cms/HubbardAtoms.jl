import HubbardAtomLean.FieldTSums

/-!
# The rational identities of the transverse sector

Identities (A6) and (A7) of the proof of Theorem "density and magnetic" for the transverse data,
with the values of `S_k` at the zeros of `P` from the weights (`sT₁_eq` … `sT₄_eq`).

* `FParams.identity_A6T`: the image of the constant, `-βH(z) + λc̄ + ∑ⱼ cNⱼ(z)ℓⱼ = -βN₂(z)/(2a)`.
* `FParams.identity_A7T`: Eq. (source), with the remainder `W(z; s) = W₀(s) + z²W₂(s)`, affine in
  `s`; its coefficients are polynomials in `a`, `W` and the weights, divided by `W`.
* `FParams.q₀T_zero_data`: at `q₀ = 0` (`k = 0`, `h = ±a`) the thermal values `S₀(0)`, `S₀(±2a)`
  of the sums `ℓ₀` … `ℓ₃` in terms of the weights, which satisfy `p↑² = p₀p₂` or `p↓² = p₀p₂`.
  Both identities hold there too, with the same `W(z; s)`.
-/

open Complex

namespace HubbardAtom

set_option maxHeartbeats 1000000 in
-- elaborating a polynomial with a hundred terms
/-- `W · W₀(s)` of the transverse sector, with `p₂ = 1 - p₀ - p↑ - p↓`. -/
noncomputable def W₀Tpoly (a W p₀ pu pd s : ℂ) : ℂ :=
  3 * W ^ 5 * a * p₀ ^ 2/4 + 3 * W ^ 5 * a * p₀ * pd/4 + 3 * W ^ 5 * a * p₀ * pu/4 - 
    3 * W ^ 5 * a * p₀/4 + W ^ 5 * a * pd ^ 2/4 + W ^ 5 * a * pd * pu/4 - W ^ 5 * a * pd/4 + 
    W ^ 5 * a * pu ^ 2/4 - W ^ 5 * a * pu/4 - W ^ 4 * a ^ 2 * pd ^ 3/4 + 
    3 * W ^ 4 * a ^ 2 * pd ^ 2 * pu/4 - 3 * W ^ 4 * a ^ 2 * pd ^ 2/4 - 
    3 * W ^ 4 * a ^ 2 * pd * pu ^ 2/4 + W ^ 4 * a ^ 2 * pd + W ^ 4 * a ^ 2 * pu ^ 3/4 + 
    3 * W ^ 4 * a ^ 2 * pu ^ 2/4 - W ^ 4 * a ^ 2 * pu + W ^ 3 * a ^ 3 * pd ^ 3 - 
    W ^ 3 * a ^ 3 * pd ^ 2 * pu - W ^ 3 * a ^ 3 * pd ^ 2 - W ^ 3 * a ^ 3 * pd * pu ^ 2 + 
    2 * W ^ 3 * a ^ 3 * pd * pu + W ^ 3 * a ^ 3 * pu ^ 3 - W ^ 3 * a ^ 3 * pu ^ 2 - 
    3 * W ^ 3 * a * p₀ ^ 2 * s - 3 * W ^ 3 * a * p₀ * pd * s - 3 * W ^ 3 * a * p₀ * pu * s + 
    3 * W ^ 3 * a * p₀ * s - W ^ 3 * a * pd ^ 2 * s - W ^ 3 * a * pd * pu * s + 
    W ^ 3 * a * pd * s - W ^ 3 * a * pu ^ 2 * s + W ^ 3 * a * pu * s - 
    4 * W ^ 2 * a ^ 4 * p₀ ^ 2 * pd + 4 * W ^ 2 * a ^ 4 * p₀ ^ 2 * pu - 
    4 * W ^ 2 * a ^ 4 * p₀ * pd ^ 2 + 4 * W ^ 2 * a ^ 4 * p₀ * pd + 
    4 * W ^ 2 * a ^ 4 * p₀ * pu ^ 2 - 4 * W ^ 2 * a ^ 4 * p₀ * pu - 
    4 * W ^ 2 * a ^ 4 * pd ^ 2 * pu + 4 * W ^ 2 * a ^ 4 * pd ^ 2 + 
    4 * W ^ 2 * a ^ 4 * pd * pu ^ 2 - 4 * W ^ 2 * a ^ 4 * pd - 4 * W ^ 2 * a ^ 4 * pu ^ 2 + 
    4 * W ^ 2 * a ^ 4 * pu + 4 * W ^ 2 * a ^ 2 * p₀ ^ 2 * pd * s - 
    4 * W ^ 2 * a ^ 2 * p₀ ^ 2 * pu * s + 4 * W ^ 2 * a ^ 2 * p₀ * pd ^ 2 * s - 
    4 * W ^ 2 * a ^ 2 * p₀ * pd * s - 4 * W ^ 2 * a ^ 2 * p₀ * pu ^ 2 * s + 
    4 * W ^ 2 * a ^ 2 * p₀ * pu * s + W ^ 2 * a ^ 2 * pd ^ 3 * s + 
    W ^ 2 * a ^ 2 * pd ^ 2 * pu * s - W ^ 2 * a ^ 2 * pd ^ 2 * s - 
    W ^ 2 * a ^ 2 * pd * pu ^ 2 * s - W ^ 2 * a ^ 2 * pu ^ 3 * s + W ^ 2 * a ^ 2 * pu ^ 2 * s - 
    12 * W * a ^ 5 * p₀ ^ 2 - 12 * W * a ^ 5 * p₀ * pd - 12 * W * a ^ 5 * p₀ * pu + 
    12 * W * a ^ 5 * p₀ - 4 * W * a ^ 5 * pd ^ 3 + 4 * W * a ^ 5 * pd ^ 2 * pu + 
    4 * W * a ^ 5 * pd * pu ^ 2 - 12 * W * a ^ 5 * pd * pu + 4 * W * a ^ 5 * pd - 
    4 * W * a ^ 5 * pu ^ 3 + 4 * W * a ^ 5 * pu + 12 * W * a ^ 3 * p₀ ^ 2 * s + 
    12 * W * a ^ 3 * p₀ * pd * s + 12 * W * a ^ 3 * p₀ * pu * s - 12 * W * a ^ 3 * p₀ * s + 
    4 * W * a ^ 3 * pd ^ 2 * s + 4 * W * a ^ 3 * pd * pu * s - 4 * W * a ^ 3 * pd * s + 
    4 * W * a ^ 3 * pu ^ 2 * s - 4 * W * a ^ 3 * pu * s + 16 * a ^ 6 * p₀ ^ 2 * pd - 
    16 * a ^ 6 * p₀ ^ 2 * pu + 16 * a ^ 6 * p₀ * pd ^ 2 - 16 * a ^ 6 * p₀ * pd - 
    16 * a ^ 6 * p₀ * pu ^ 2 + 16 * a ^ 6 * p₀ * pu + 4 * a ^ 6 * pd ^ 3 + 
    4 * a ^ 6 * pd ^ 2 * pu - 4 * a ^ 6 * pd ^ 2 - 4 * a ^ 6 * pd * pu ^ 2 - 4 * a ^ 6 * pu ^ 3 + 
    4 * a ^ 6 * pu ^ 2 - 16 * a ^ 4 * p₀ ^ 2 * pd * s + 16 * a ^ 4 * p₀ ^ 2 * pu * s - 
    16 * a ^ 4 * p₀ * pd ^ 2 * s + 16 * a ^ 4 * p₀ * pd * s + 16 * a ^ 4 * p₀ * pu ^ 2 * s - 
    16 * a ^ 4 * p₀ * pu * s - 4 * a ^ 4 * pd ^ 3 * s - 4 * a ^ 4 * pd ^ 2 * pu * s + 
    4 * a ^ 4 * pd ^ 2 * s + 4 * a ^ 4 * pd * pu ^ 2 * s + 4 * a ^ 4 * pu ^ 3 * s - 
    4 * a ^ 4 * pu ^ 2 * s

set_option maxHeartbeats 1000000 in
-- elaborating a polynomial with a hundred terms
/-- `W · W₂(s)` of the transverse sector, with `p₂ = 1 - p₀ - p↑ - p↓`. -/
noncomputable def W₂Tpoly (a W p₀ pu pd s : ℂ) : ℂ :=
  -W ^ 3 * a * p₀ ^ 2 - W ^ 3 * a * p₀ * pd - W ^ 3 * a * p₀ * pu + W ^ 3 * a * p₀ - 
    W ^ 3 * a * pd * pu - W ^ 2 * a ^ 2 * pd ^ 3 + 3 * W ^ 2 * a ^ 2 * pd ^ 2 * pu + 
    W ^ 2 * a ^ 2 * pd ^ 2 - 3 * W ^ 2 * a ^ 2 * pd * pu ^ 2 + W ^ 2 * a ^ 2 * pu ^ 3 - 
    W ^ 2 * a ^ 2 * pu ^ 2 - 4 * W * a ^ 3 * p₀ ^ 2 - 4 * W * a ^ 3 * p₀ * pd - 
    4 * W * a ^ 3 * p₀ * pu + 4 * W * a ^ 3 * p₀ + 4 * W * a ^ 3 * pd ^ 3 - 
    4 * W * a ^ 3 * pd ^ 2 * pu - 4 * W * a ^ 3 * pd ^ 2 - 4 * W * a ^ 3 * pd * pu ^ 2 + 
    4 * W * a ^ 3 * pd * pu + 4 * W * a ^ 3 * pu ^ 3 - 4 * W * a ^ 3 * pu ^ 2 + 
    4 * W * a * p₀ ^ 2 * s + 4 * W * a * p₀ * pd * s + 4 * W * a * p₀ * pu * s - 
    4 * W * a * p₀ * s + 4 * W * a * pd * pu * s - 16 * a ^ 4 * p₀ ^ 2 * pd + 
    16 * a ^ 4 * p₀ ^ 2 * pu - 16 * a ^ 4 * p₀ * pd ^ 2 + 16 * a ^ 4 * p₀ * pd + 
    16 * a ^ 4 * p₀ * pu ^ 2 - 16 * a ^ 4 * p₀ * pu - 4 * a ^ 4 * pd ^ 3 - 
    4 * a ^ 4 * pd ^ 2 * pu + 4 * a ^ 4 * pd ^ 2 + 4 * a ^ 4 * pd * pu ^ 2 + 4 * a ^ 4 * pu ^ 3 - 
    4 * a ^ 4 * pu ^ 2 + 16 * a ^ 2 * p₀ ^ 2 * pd * s - 16 * a ^ 2 * p₀ ^ 2 * pu * s + 
    16 * a ^ 2 * p₀ * pd ^ 2 * s - 16 * a ^ 2 * p₀ * pd * s - 16 * a ^ 2 * p₀ * pu ^ 2 * s + 
    16 * a ^ 2 * p₀ * pu * s + 4 * a ^ 2 * pd ^ 3 * s + 4 * a ^ 2 * pd ^ 2 * pu * s - 
    4 * a ^ 2 * pd ^ 2 * s - 4 * a ^ 2 * pd * pu ^ 2 * s - 4 * a ^ 2 * pu ^ 3 * s + 
    4 * a ^ 2 * pu ^ 2 * s

namespace FParams

variable (F : FParams) (k : ℤ)

/-- `W₀(s)`. -/
noncomputable def W₀T (s : ℂ) : ℂ := W₀Tpoly F.a (F.WT k) F.p₀ F.pUp F.pDn s / F.WT k

/-- `W₂(s)`. -/
noncomputable def W₂T (s : ℂ) : ℂ := W₂Tpoly F.a (F.WT k) F.p₀ F.pUp F.pDn s / F.WT k

/-- `W(z; s) = W₀(s) + z²W₂(s)`. -/
noncomputable def WphT (z s : ℂ) : ℂ := F.W₀T k s + z ^ 2 * F.W₂T k s

/-- `q₀T_zero_data`, case by case: `h = a` or `h = -a`. -/
private theorem q₀T_zero_cases (ha : (F.a : ℂ) ≠ 0) (hq : F.q₀T k = 0) :
    k = 0 ∧
      ((F.WT k = 2 * F.a ∧ F.sZT = F.β * (F.pUp - F.p₀) / (2 * (F.pUp + F.p₀)) ∧
          F.sMT = F.β * (F.pDn - F.p₀) / (2 * (F.pDn + F.p₀)) ∧
          F.sPT = F.β * (F.p₂ - F.pDn) / (2 * (F.p₂ + F.pDn)) ∧
          (F.pUp : ℂ) ^ 2 = F.p₀ * F.p₂) ∨
        (F.WT k = -(2 * F.a) ∧ F.sZT = F.β * (F.pDn - F.p₀) / (2 * (F.pDn + F.p₀)) ∧
          F.sMT = F.β * (F.pUp - F.p₀) / (2 * (F.pUp + F.p₀)) ∧
          F.sPT = F.β * (F.p₂ - F.pUp) / (2 * (F.p₂ + F.pUp)) ∧
          (F.pDn : ℂ) ^ 2 = F.p₀ * F.p₂)) := by
  obtain ⟨rfl, hh⟩ := F.q₀T_eq_zero k ha hq
  have hW : F.WT 0 = -2 * F.h := by simp [WT, ω, Params.ω]
  refine ⟨rfl, ?_⟩
  rcases hh with hh | hh
  · right
    have e₀ : ((-F.a + F.h : ℝ) : ℂ) = 0 := by rw [hh]; simp
    refine ⟨by rw [hW, hh]; ring, ?_, ?_, ?_, ?_⟩
    · rw [← F.sT₂_eq, sT₂, sZT]
      congr 1
      push_cast at e₀ ⊢
      rw [e₀]
    · rw [← F.sT₁_eq, sT₁, sMT, hh]
      congr 1
      ring
    · rw [← F.sT₄_eq, sT₄, sPT, hh]
      congr 1
      ring
    · have : F.pDn ^ 2 = F.p₀ * F.p₂ := by
        rw [F.pDn_eq_p₀_mul, F.p₂_eq_p₀_mul, hh, mul_pow, sq (Real.exp _), ← Real.exp_add]
        ring_nf
      exact_mod_cast this
  · left
    have e₀ : ((-F.a - F.h : ℝ) : ℂ) = 0 := by rw [hh]; simp
    refine ⟨by rw [hW, hh]; push_cast; ring, ?_, ?_, ?_, ?_⟩
    · rw [← F.sT₁_eq, sT₁, sZT]
      congr 1
      push_cast at e₀ ⊢
      rw [e₀]
    · rw [← F.sT₂_eq, sT₂, sMT, hh]
      congr 1
      push_cast
      ring
    · rw [← F.sT₃_eq, sT₃, sPT, hh]
      congr 1
      push_cast
      ring
    · have : F.pUp ^ 2 = F.p₀ * F.p₂ := by
        rw [F.pUp_eq_p₀_mul, F.p₂_eq_p₀_mul, hh, mul_pow, sq (Real.exp _), ← Real.exp_add]
        ring_nf
      exact_mod_cast this

/-- Identity (A6) for `q₀ ≠ 0`, where `P` has four simple zeros. -/
private theorem identity_A6T_ne (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0)
    (z : ℂ) :
    -F.β * F.HT k z + F.lamT * F.cbarT +
        (F.cN₀T k z * F.ℓ₀T k + F.cN₁T k z * F.ℓ₁T k + F.cN₂T k z * F.ℓ₂T k +
          F.cN₃T * F.ℓ₃T k) =
      -(F.β * F.N₂T k z / (2 * F.a)) := by
  have hβ := (F.lat F.μ).β_ne_zero
  simp only [lat_β] at hβ
  have h₁ : (F.pUp : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos F.pUp_pos F.p₀_pos).ne'
  have h₂ : (F.pDn : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos F.pDn_pos F.p₀_pos).ne'
  have h₃ : (F.p₂ : ℂ) + F.pDn ≠ 0 := by exact_mod_cast (add_pos F.p₂_pos F.pDn_pos).ne'
  have h₄ : (F.p₂ : ℂ) + F.pUp ≠ 0 := by exact_mod_cast (add_pos F.p₂_pos F.pUp_pos).ne'
  have hs := F.weights_sum'
  simp only [HT, q₁T, lamT, cbarT, cN₀T, cN₁T, cN₂T, cN₃T, ℓ₀T, ℓ₁T, ℓ₂T, ℓ₃T, TpT, TmT, SpT,
    SmT, F.sT₁_eq, F.sT₂_eq, F.sT₃_eq, F.sT₄_eq, N₂T, ζT, κT, ρ, mf, hq, ite_false]
  rw [q₀T] at hq ⊢
  push_cast
  generalize F.WT k = W at *
  generalize (F.a : ℂ) = a at *
  generalize (F.β : ℂ) = β at *
  generalize (F.p₂ : ℂ) = p₂ at *
  obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hs
  have hq4 : a ^ 2 * 4 - W ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
  field_simp
  ring

set_option maxHeartbeats 40000000 in
-- a rational identity of high degree in nine variables
set_option maxRecDepth 100000 in
/-- Identity (A7) for `q₀ ≠ 0`. -/
private theorem identity_A7T_ne (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0)
    (z v : ℂ) :
    F.lamT * F.ρ * (z - v + 2 * F.a * F.ρ) * F.PT k v - F.HT k (-v) * F.NT k z v -
        F.lamT / F.β * (F.cbarT * F.NT k z (-v) + F.ρ * F.RhatT k z (-v)) =
      -F.eT k v * F.N₂T k z + F.lamT * F.ρ * z * (F.PT k v - F.eT k v) +
        F.lamT * F.WphT k z (v ^ 2) := by
  have hβ := (F.lat F.μ).β_ne_zero
  simp only [lat_β] at hβ
  have h₁ : (F.pUp : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos F.pUp_pos F.p₀_pos).ne'
  have h₂ : (F.pDn : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos F.pDn_pos F.p₀_pos).ne'
  have h₃ : (F.p₂ : ℂ) + F.pDn ≠ 0 := by exact_mod_cast (add_pos F.p₂_pos F.pDn_pos).ne'
  have h₄ : (F.p₂ : ℂ) + F.pUp ≠ 0 := by exact_mod_cast (add_pos F.p₂_pos F.pUp_pos).ne'
  have hs := F.weights_sum'
  simp only [RhatT, F.NT_eq_coeffs, F.PT_eq_poly, eT, HT, q₁T, WphT, W₀T, W₂T, W₀Tpoly, W₂Tpoly,
    lamT, cbarT, cN₀T, cN₁T, cN₂T, cN₃T, ℓ₀T, ℓ₁T, ℓ₂T, ℓ₃T, TpT, TmT, SpT, SmT, F.sT₁_eq,
    F.sT₂_eq, F.sT₃_eq, F.sT₄_eq, N₂T, ζT, κT, ρ, mf, hq, ite_false]
  rw [q₀T] at hq ⊢
  push_cast
  generalize F.WT k = W at *
  generalize (F.a : ℂ) = a at *
  generalize (F.β : ℂ) = β at *
  generalize (F.p₂ : ℂ) = p₂ at *
  obtain rfl : p₂ = 1 - F.p₀ - F.pUp - F.pDn := by linear_combination hs
  -- `field_simp` would normalize `q₀ = a² - W²/4` to forms it cannot discharge
  generalize hQ : a ^ 2 - W ^ 2 / 4 = Q at hq ⊢
  field_simp
  subst hQ
  ring

/-- At `q₀ = 0` (`k = 0`, `h = -σa`, so `W = 2σa`, `σ = ±1`) the thermal values `S₀(0)`,
`S₀(±2a)` are those at the zeros of `P`, through the weights `X = p↑`, `Y = p↓` (`σ = 1`) or
`X = p↓`, `Y = p↑` (`σ = -1`), which satisfy `X² = p₀p₂`. -/
theorem q₀T_zero_data (ha : (F.a : ℂ) ≠ 0) (hq : F.q₀T k = 0) :
    k = 0 ∧ ∃ (σ : ℂ) (X Y : ℝ), (σ = 1 ∨ σ = -1) ∧ 0 < X ∧ 0 < Y ∧ F.WT k = σ * (2 * F.a) ∧
      (F.pUp : ℂ) = ((1 + σ) * X + (1 - σ) * Y) / 2 ∧
      (F.pDn : ℂ) = ((1 - σ) * X + (1 + σ) * Y) / 2 ∧
      F.sZT = F.β * (X - F.p₀) / (2 * (X + F.p₀)) ∧
      F.sMT = F.β * (Y - F.p₀) / (2 * (Y + F.p₀)) ∧
      F.sPT = F.β * (F.p₂ - Y) / (2 * (F.p₂ + Y)) ∧ (X : ℂ) ^ 2 = F.p₀ * F.p₂ := by
  obtain ⟨hk, ⟨hW, h₁, h₂, h₃, hc⟩ | ⟨hW, h₁, h₂, h₃, hc⟩⟩ := F.q₀T_zero_cases k ha hq
  · exact ⟨hk, 1, F.pUp, F.pDn, Or.inl rfl, F.pUp_pos, F.pDn_pos, by rw [hW]; ring, by ring,
      by ring, h₁, h₂, h₃, hc⟩
  · exact ⟨hk, -1, F.pDn, F.pUp, Or.inr rfl, F.pDn_pos, F.pUp_pos, by rw [hW]; ring, by ring,
      by ring, h₁, h₂, h₃, hc⟩

/-- Identity (A6) at `q₀ = 0`, with `X = tp₀`, `p₂ = t²p₀` (`t = e^{βμ}`) and both signs `σ`. -/
private theorem identity_A6T_zero (ha : (F.a : ℂ) ≠ 0) (hq : F.q₀T k = 0) (z : ℂ) :
    -F.β * F.HT k z + F.lamT * F.cbarT +
        (F.cN₀T k z * F.ℓ₀T k + F.cN₁T k z * F.ℓ₁T k + F.cN₂T k z * F.ℓ₂T k +
          F.cN₃T * F.ℓ₃T k) =
      -(F.β * F.N₂T k z / (2 * F.a)) := by
  obtain ⟨rfl, σ, X, Y, hσ, hX, hY, hW, hpu, hpd, h₁, h₂, h₃, hc⟩ := F.q₀T_zero_data k ha hq
  have hβ := (F.lat F.μ).β_ne_zero
  simp only [lat_β] at hβ
  have hp0 : (F.p₀ : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.p₀_pos.ne'
  have e₁ : (X : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos hX F.p₀_pos).ne'
  have e₂ : (Y : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos hY F.p₀_pos).ne'
  have e₃ : (F.p₂ : ℂ) + Y ≠ 0 := by exact_mod_cast (add_pos F.p₂_pos hY).ne'
  have hs := F.weights_sum'
  simp only [HT, q₁T, lamT, cbarT, cN₀T, cN₁T, cN₂T, cN₃T, ℓ₀T, ℓ₁T, ℓ₂T, ℓ₃T, N₂T, ζT, κT,
    ρ, mf, hq, ite_true, h₁, h₂, h₃, hW]
  push_cast
  simp only [hpu, hpd] at hs ⊢
  generalize (F.a : ℂ) = a at *
  generalize (F.β : ℂ) = β at *
  generalize (X : ℂ) = X at *
  generalize (Y : ℂ) = Y at *
  generalize (F.p₂ : ℂ) = p₂ at *
  generalize (F.p₀ : ℂ) = p₀ at *
  -- `X = t p₀`, `p₂ = t² p₀` with `t = e^{βμ}`
  generalize ht : X / p₀ = t
  obtain rfl : X = t * p₀ := by rw [← ht]; field_simp
  obtain rfl : p₂ = t ^ 2 * p₀ := by
    apply mul_left_cancel₀ hp0
    linear_combination -hc
  obtain rfl : Y = 1 - p₀ - t * p₀ - t ^ 2 * p₀ := by linear_combination hs
  have ht : 1 + t ≠ 0 := fun h0 ↦ e₁ (by linear_combination p₀ * h0)
  have ht' : t + 1 ≠ 0 := by rwa [add_comm]
  rcases hσ with rfl | rfl <;> field_simp <;> ring

set_option maxHeartbeats 40000000 in
-- a rational identity of high degree
set_option maxRecDepth 100000 in
/-- Identity (A7) at `q₀ = 0`. -/
private theorem identity_A7T_zero (ha : (F.a : ℂ) ≠ 0) (hq : F.q₀T k = 0) (z v : ℂ) :
    F.lamT * F.ρ * (z - v + 2 * F.a * F.ρ) * F.PT k v - F.HT k (-v) * F.NT k z v -
        F.lamT / F.β * (F.cbarT * F.NT k z (-v) + F.ρ * F.RhatT k z (-v)) =
      -F.eT k v * F.N₂T k z + F.lamT * F.ρ * z * (F.PT k v - F.eT k v) +
        F.lamT * F.WphT k z (v ^ 2) := by
  obtain ⟨rfl, σ, X, Y, hσ, hX, hY, hW, hpu, hpd, h₁, h₂, h₃, hc⟩ := F.q₀T_zero_data k ha hq
  have hβ := (F.lat F.μ).β_ne_zero
  simp only [lat_β] at hβ
  have hp0 : (F.p₀ : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.p₀_pos.ne'
  have e₁ : (X : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos hX F.p₀_pos).ne'
  have e₂ : (Y : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos hY F.p₀_pos).ne'
  have e₃ : (F.p₂ : ℂ) + Y ≠ 0 := by exact_mod_cast (add_pos F.p₂_pos hY).ne'
  have hs := F.weights_sum'
  simp only [RhatT, F.NT_eq_coeffs, F.PT_eq_poly, eT, HT, q₁T, WphT, W₀T, W₂T, W₀Tpoly,
    W₂Tpoly, lamT, cbarT, cN₀T, cN₁T, cN₂T, cN₃T, ℓ₀T, ℓ₁T, ℓ₂T, ℓ₃T, N₂T, ζT, κT, ρ, mf, hq,
    ite_true, h₁, h₂, h₃, hW]
  push_cast
  simp only [hpu, hpd] at hs ⊢
  generalize (F.a : ℂ) = a at *
  generalize (F.β : ℂ) = β at *
  generalize (X : ℂ) = X at *
  generalize (Y : ℂ) = Y at *
  generalize (F.p₂ : ℂ) = p₂ at *
  generalize (F.p₀ : ℂ) = p₀ at *
  -- `X = t p₀`, `p₂ = t² p₀` with `t = e^{βμ}`
  generalize ht : X / p₀ = t
  obtain rfl : X = t * p₀ := by rw [← ht]; field_simp
  obtain rfl : p₂ = t ^ 2 * p₀ := by
    apply mul_left_cancel₀ hp0
    linear_combination -hc
  obtain rfl : Y = 1 - p₀ - t * p₀ - t ^ 2 * p₀ := by linear_combination hs
  have ht : 1 + t ≠ 0 := fun h0 ↦ e₁ (by linear_combination p₀ * h0)
  have ht' : t + 1 ≠ 0 := by rwa [add_comm]
  rcases hσ with rfl | rfl <;> field_simp <;> ring

/-- Identity (A6): `-βH(z) + λc̄ + ∑ⱼ cNⱼ(z)ℓⱼ = -βN₂(z)/(2a)`. -/
theorem identity_A6T (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (z : ℂ) :
    -F.β * F.HT k z + F.lamT * F.cbarT +
        (F.cN₀T k z * F.ℓ₀T k + F.cN₁T k z * F.ℓ₁T k + F.cN₂T k z * F.ℓ₂T k +
          F.cN₃T * F.ℓ₃T k) =
      -(F.β * F.N₂T k z / (2 * F.a)) := by
  by_cases hq : F.q₀T k = 0
  · exact F.identity_A6T_zero k ha hq z
  · exact F.identity_A6T_ne k ha hW hq z

/-- Identity (A7), Eq. (source): with `P(v)Ī(z; -v) = c̄N(z, -v)/ρ + R̂(z, -v)` (`P` is even),
`λρ(z - v + 2aρ)P(v) - H(-v)N(z, v) - (λρ/β)P(v)Ī(z; -v)
  = -e(v)N₂(z) + λρz[P(v) - e(v)] + λW(z; v²)`. -/
theorem identity_A7T (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (z v : ℂ) :
    F.lamT * F.ρ * (z - v + 2 * F.a * F.ρ) * F.PT k v - F.HT k (-v) * F.NT k z v -
        F.lamT / F.β * (F.cbarT * F.NT k z (-v) + F.ρ * F.RhatT k z (-v)) =
      -F.eT k v * F.N₂T k z + F.lamT * F.ρ * z * (F.PT k v - F.eT k v) +
        F.lamT * F.WphT k z (v ^ 2) := by
  by_cases hq : F.q₀T k = 0
  · exact F.identity_A7T_zero k ha hq z v
  · exact F.identity_A7T_ne k ha hW hq z v

end FParams

end HubbardAtom
