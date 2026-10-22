import HubbardAtomLean.FieldTSums

/-!
# The rational identities of the transverse sector

Identities (A6) and (A7) of the proof of Theorem "density and magnetic" for the transverse data,
with the values of `S_k` at the zeros of `P` from the weights (`sT₁_eq` … `sT₄_eq`).

* `FParams.identity_A6T`: the image of the constant, `-βH(z) + λc̄ + ∑ⱼ cNⱼ(z)ℓⱼ = -βN₂(z)/(2a)`.
* `FParams.identity_A7T`: Eq. (source), with the remainder `W(z; s) = W₀(s) + z²W₂(s)`, affine in
  `s`; its coefficients are polynomials in `a`, `W` and the weights, divided by `W`.
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

/-- Identity (A6): `-βH(z) + λc̄ + ∑ⱼ cNⱼ(z)ℓⱼ = -βN₂(z)/(2a)`. -/
theorem identity_A6T (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0) (z : ℂ) :
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
    SmT, F.sT₁_eq, F.sT₂_eq, F.sT₃_eq, F.sT₄_eq, N₂T, ζT, κT, ρ, mf]
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
/-- Identity (A7), Eq. (source): with `P(v)Ī(z; -v) = c̄N(z, -v)/ρ + R̂(z, -v)` (`P` is even),
`λρ(z - v + 2aρ)P(v) - H(-v)N(z, v) - (λρ/β)P(v)Ī(z; -v)
  = -e(v)N₂(z) + λρz[P(v) - e(v)] + λW(z; v²)`. -/
theorem identity_A7T (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0) (z v : ℂ) :
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
    F.sT₂_eq, F.sT₃_eq, F.sT₄_eq, N₂T, ζT, κT, ρ, mf]
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

end FParams

end HubbardAtom
