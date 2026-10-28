import HubbardAtomLean.WardIdentities

/-!
# Lattice sums of the longitudinal sector

The sums behind the inverse of the coupled longitudinal susceptibility, Eq. (field-L) of the note
(App. F.6.4). The row of spin `σ` lives on the lattice `L_σ = {z_{μ_σ}(ν)}`, `μ_σ = μ + σh`, and
its opposite-spin block acts on functions on `L_{-σ}`; everything is stated for a general spin
`σ : Bool` (`true` for `↑`), with the rows of spin `↓` from the spin flip.

* `FParams.cL`, `BL`, `pL`, `mL`, `rL`: the centre `μ_σ`, the lattice, the weight `p_σ`,
  `σm = p_σ - p_{-σ}` and `r_σ = ρ - σm` of spin `σ`.
* `FParams.PL_mul_chiL_self`, `PL_mul_chiL_opp`: the longitudinal input for both rows.
* `FParams.S_opp_z`, `S_opp_neg_z`, `S_self_neg_z`: the bosonic values of the thermal sums at the
  points of a spin lattice, `S_{-σ}(z) = -c_S/(σm)`, `S_{-σ}(-z) = c_P/ρ`, `S_σ(-z) = βc_σ/(2D_σ)`.
* `FParams.hasSum_kernelL`: the opposite-spin kernels on a simple pole,
  `∑ₗ [ρ/(z + l) - σm/(z - l)]/(l - x) = α^P_σ(x)/(z + x) + α^S_σ(x)/(z - x)`.
* `FParams.αSL_mul_αSL`, `αPL_mul_αPL`, `αL_mul_αL_neg`: the product relations
  `α^S_σ(x)α^S_{-σ}(x) = β²π_S`, `α^P_σ(x)α^P_{-σ}(-x) = β²π_P`, `α_σ(x)α_σ(-x) = β²π_Pπ_S`.
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ)

/-- The part of the rank-two sum of a cubic `M = m₀ + m₁t + m₂t² + m₃t³` that comes from the zeros
of `P`: the lattice sum of `(P(x)M(t) - M(x)P(t))/(t - x)` over `P(t)`. -/
noncomputable def RhatC (m₀ m₁ m₂ m₃ x : ℂ) : ℂ :=
  (A.Pph k x * (m₁ + m₂ * x + m₃ * x ^ 2) -
        (m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3) * (x ^ 3 + A.κ k * x)) * A.ℓ₀ k +
      (A.Pph k x * (m₂ + m₃ * x) - (m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3) * (x ^ 2 + A.κ k)) *
        A.ℓ₁ k +
    (A.Pph k x * m₃ - (m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3) * x) * A.ℓ₂ k -
      (m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3) * A.ℓ₃ k

/-- Lemma "rank-two sums" for a cubic `M`, times `P(x)`, which holds also where `P(x) = 0`:
`∑ₗ P(x)M(zₗ)/(P(zₗ)(zₗ - x)) = M(x)S_k(x) + R̂(x)`. -/
theorem hasSum_cubic_mul (ha : (A.a : ℂ) ≠ 0) (m₀ m₁ m₂ m₃ : ℂ) {x : ℂ}
    (hx : ∀ l, A.z k l ≠ x) :
    HasSum (fun l ↦ A.Pph k x * ((m₀ + m₁ * A.z k l + m₂ * A.z k l ^ 2 + m₃ * A.z k l ^ 3) /
        (A.Pph k (A.z k l) * (A.z k l - x))))
      ((m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3) * A.S k x + A.RhatC k m₀ m₁ m₂ m₃ x) := by
  convert A.hasSum_cubic k ha
    (A.Pph k x * (m₁ + m₂ * x + m₃ * x ^ 2) -
      (m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3) * (x ^ 3 + A.κ k * x))
    (A.Pph k x * (m₂ + m₃ * x) - (m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3) * (x ^ 2 + A.κ k))
    (A.Pph k x * m₃ - (m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3) * x)
    (-(m₀ + m₁ * x + m₂ * x ^ 2 + m₃ * x ^ 3)) hx using 1
  · funext l
    exact rank_two_pf_mul (A.Pph_eq_poly k _) (A.Pph_eq_poly k x) (A.Pph_z_ne_zero k l)
      (sub_ne_zero.mpr (hx l))
  · rw [RhatC]
    ring

end Params

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The data of a spin -/

/-- The centre `μ_σ = μ + σh` of the lattice of spin `σ`. -/
noncomputable def cL (σ : Bool) : ℝ := bif σ then F.μ + F.h else F.μ - F.h

/-- The lattice `L_σ` of spin `σ`: `z = μ_σ + i(νₙ + ω/2)`. -/
noncomputable abbrev BL (σ : Bool) : Params := F.lat (F.cL σ)

/-- The weight `p_σ`. -/
noncomputable def pL (σ : Bool) : ℝ := bif σ then F.pUp else F.pDn

/-- `σm = p_σ - p_{-σ}`. -/
noncomputable def mL (σ : Bool) : ℝ := F.pL σ - F.pL (!σ)

/-- `r_σ = ρ - σm`. -/
noncomputable def rL (σ : Bool) : ℝ := F.ρ - F.mL σ

/-- `D_σ = p₂p_σ - p₀p_{-σ}`. -/
noncomputable def DL (σ : Bool) : ℝ := F.p₂ * F.pL σ - F.p₀ * F.pL (!σ)

/-- `c_σ = p₂p_σ + p₀p_{-σ}`. -/
noncomputable def cLw (σ : Bool) : ℝ := F.p₂ * F.pL σ + F.p₀ * F.pL (!σ)

/-- `ζ_σ = 4βa²(p₀ + p_σ)(p_{-σ} + p₂)`, the weight of the rank-one same-spin term at `k = 0`. -/
noncomputable def ζLs (σ : Bool) : ℂ :=
  4 * F.β * F.a ^ 2 * (F.p₀ + F.pL σ) * (F.pL (!σ) + F.p₂)

/-- `c_P = β(p₀ + p₂)/2`. -/
noncomputable def cP : ℝ := F.β * (F.p₀ + F.p₂) / 2

/-- `c_S = β(p↑ + p↓)/2`. -/
noncomputable def cS : ℝ := F.β * (F.pUp + F.pDn) / 2

@[simp] theorem BL_true : F.BL true = F.Bup := rfl

@[simp] theorem BL_false : F.BL false = F.Bdn := rfl

@[simp] theorem pL_true : F.pL true = F.pUp := rfl

@[simp] theorem pL_false : F.pL false = F.pDn := rfl

theorem cL_add (σ : Bool) : F.cL σ + F.cL (!σ) = 2 * F.μ := by
  cases σ <;>
    simp only [cL, Bool.not_false, Bool.not_true, Bool.cond_true, Bool.cond_false] <;> ring

theorem pL_add (σ : Bool) : (F.pL σ : ℂ) + F.pL (!σ) = F.pUp + F.pDn := by
  cases σ <;> simp only [pL_true, pL_false, Bool.not_false, Bool.not_true, add_comm]

theorem pL_pos (σ : Bool) : 0 < F.pL σ := by
  cases σ
  · exact F.pDn_pos
  · exact F.pUp_pos

theorem weights_sumL (σ : Bool) : (F.p₀ : ℂ) + F.pL σ + F.pL (!σ) + F.p₂ = 1 := by
  rw [add_assoc (F.p₀ : ℂ), F.pL_add]
  linear_combination F.weights_sum'

theorem mL_true : F.mL true = F.mf := rfl

theorem mL_false : F.mL false = -F.mf := by
  simp only [mL, pL_false, Bool.not_false, pL_true, mf]
  ring

@[simp] theorem rL_true : F.rL true = F.rUp := rfl

@[simp] theorem rL_false : F.rL false = F.rDn := by
  rw [rL, mL_false, rDn]
  ring

theorem mL_ne_zero (hh : F.h ≠ 0) (σ : Bool) : (F.mL σ : ℂ) ≠ 0 := by
  have hm := F.mf_ne_zero hh
  cases σ
  · rw [mL_false]
    push_cast
    exact neg_ne_zero.mpr hm
  · exact hm

/-! ### The spin flip -/

@[simp] theorem flip_cL (σ : Bool) : F.flip.cL σ = F.cL (!σ) := by
  cases σ <;> simp only [cL, flip_μ, flip_h, Bool.not_false, Bool.not_true, Bool.cond_true,
    Bool.cond_false] <;> ring

@[simp] theorem flip_BL (σ : Bool) : F.flip.BL σ = F.BL (!σ) := by
  simp only [BL, flip_cL, lat, flip_U, flip_β]

@[simp] theorem flip_pL (σ : Bool) : F.flip.pL σ = F.pL (!σ) := by
  cases σ <;> simp only [pL_true, pL_false, flip_pUp, flip_pDn, Bool.not_false, Bool.not_true]

@[simp] theorem flip_mL (σ : Bool) : F.flip.mL σ = F.mL (!σ) := by
  simp only [mL, flip_pL, Bool.not_not]

@[simp] theorem flip_rL (σ : Bool) : F.flip.rL σ = F.rL (!σ) := by
  simp only [rL, flip_mL, flip_ρ]

@[simp] theorem flip_a : F.flip.a = F.a := rfl

@[simp] theorem flip_wL : F.flip.wL k = F.wL k := rfl

@[simp] theorem flip_q₀L : F.flip.q₀L k = F.q₀L k := rfl

@[simp] theorem flip_PL (z : ℂ) : F.flip.PL k z = F.PL k z := rfl

theorem flip_HL (r : ℝ) (z : ℂ) : F.flip.HL k r z = F.HL k r z := rfl

theorem flip_ζL : F.flip.ζL = F.ζL := by
  simp only [ζL, flip_β, flip_a, flip_p₀, flip_p₂, flip_pUp, flip_pDn]
  ring

theorem flip_ζLs (σ : Bool) : F.flip.ζLs σ = F.ζLs (!σ) := by
  simp only [ζLs, flip_β, flip_a, flip_p₀, flip_p₂, flip_pL, Bool.not_not]

theorem flip_lamL : F.flip.lamL = F.lamL := rfl

theorem flip_N₂L (r : ℝ) (v : ℂ) : F.flip.N₂L k r v = F.N₂L k r v := by
  simp only [N₂L, flip_HL, flip_ζL, flip_a]

theorem flip_N₀L (r : ℝ) (v : ℂ) : F.flip.N₀L k r v = F.N₀L k r v := by
  simp only [N₀L, flip_N₂L, flip_q₀L, flip_ζL, flip_a, flip_PL]

/-! ### The input for both rows -/

theorem ζUp_eq : F.ζUp = F.ζLs true := rfl

/-- **The longitudinal input, equal spins**, for both rows and all `μ`, `h`:
`P(z)χ_σσ(z, v) = -βH_σ(z)δ_zv + δ_k0 ζ_σ(z² - a²)(v² - a²)/P(v)`. -/
theorem PL_mul_chiL_self (σ : Bool) (n m : ℤ) :
    F.PL k ((F.BL σ).z k n) * F.chiL σ σ k n m =
      -F.β * F.HL k (F.rL σ) ((F.BL σ).z k n) * kron n m +
        kron k 0 * F.ζLs σ * (((F.BL σ).z k n ^ 2 - F.a ^ 2) * ((F.BL σ).z k m ^ 2 - F.a ^ 2)) /
          F.PL k ((F.BL σ).z k m) := by
  cases σ
  · have h := F.flip.PL_mul_chiL_upup k n m
    have e : F.chiL false false k n m = F.flip.chiL true true k n m := F.chiL_flip k _ _ n m
    rw [e]
    have hζ : F.flip.ζUp = F.ζLs false := by rw [ζUp_eq, flip_ζLs]; rfl
    have hr : F.flip.rUp = F.rL false := by rw [← rL_true, flip_rL]; rfl
    rw [hζ, hr, flip_PL, flip_PL, flip_HL, flip_β, flip_a, flip_Bup] at h
    exact h
  · exact F.PL_mul_chiL_upup k n m

/-- **The longitudinal input, opposite spins**, for both rows, off the axes:
`P(z)χ_{σ,-σ}(z, v) = λ[ρ/(z + v) - σm/(z - v)] + (N₀_{-σ}(v) + z²N₂_{-σ}(v))/P(v)`. -/
theorem PL_mul_chiL_opp (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (σ : Bool) (n m : ℤ) :
    F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n m =
      F.lamL * (F.ρ / ((F.BL σ).z k n + (F.BL (!σ)).z k m) -
          F.mL σ / ((F.BL σ).z k n - (F.BL (!σ)).z k m)) +
        (F.N₀L k (F.rL (!σ)) ((F.BL (!σ)).z k m) +
            (F.BL σ).z k n ^ 2 * F.N₂L k (F.rL (!σ)) ((F.BL (!σ)).z k m)) /
          F.PL k ((F.BL (!σ)).z k m) := by
  cases σ
  · have h := F.flip.PL_mul_chiL_updn k hμ (by simpa using hh) n m
    have e : F.chiL false true k n m = F.flip.chiL true false k n m := F.chiL_flip k _ _ n m
    rw [Bool.not_false, e]
    have hm : (F.flip.mf : ℂ) = F.mL false := by
      rw [flip_mf, mL_false]
    have hr : F.flip.rDn = F.rL true := by rw [← rL_false, flip_rL]; rfl
    rw [hm, flip_ρ, hr, flip_PL, flip_PL, flip_N₀L, flip_N₂L, flip_lamL, flip_Bup, flip_Bdn] at h
    exact h
  · rw [Bool.not_true, BL_true, BL_false, rL_false, mL_true]
    exact F.PL_mul_chiL_updn k hμ hh n m

/-! ### The lattices of the two spins -/

theorem zL_sub_opp_ne (hh : F.h ≠ 0) (σ : Bool) (n m : ℤ) :
    (F.BL σ).z k n - (F.BL (!σ)).z k m ≠ 0 := by
  intro h0
  have := congrArg Complex.re h0
  cases σ <;> simp [Params.z, Params.x, Params.w] at this <;> exact hh (by linarith)

theorem zL_add_opp_ne (hμ : F.μ ≠ 0) (σ : Bool) (n m : ℤ) :
    (F.BL σ).z k n + (F.BL (!σ)).z k m ≠ 0 := by
  intro h0
  have := congrArg Complex.re h0
  cases σ <;> simp [Params.z, Params.x, Params.w] at this <;> exact hμ (by linarith)

theorem zL_add_self_ne {σ : Bool} (hc : F.cL σ ≠ 0) (n m : ℤ) :
    (F.BL σ).z k n + (F.BL σ).z k m ≠ 0 :=
  (F.BL σ).z_add_z_ne_zero k hc n m

/-! ### The bosonic values of the thermal sums -/

/-- `S_{-σ}(z) = -c_S/(σm)` on the lattice of spin `σ` (`Bdn_S_Bup_z`). -/
theorem S_opp_z (hh : F.h ≠ 0) (σ : Bool) (n : ℤ) :
    (F.BL (!σ)).S k ((F.BL σ).z k n) = -(F.cS / F.mL σ) := by
  cases σ
  · have h := F.flip.Bdn_S_Bup_z (by simpa using hh) k n
    simp only [flip_Bdn, flip_Bup, flip_pUp, flip_pDn, flip_mf, flip_β] at h
    rw [Bool.not_false, BL_true, BL_false, h, cS, mL_false]
    push_cast
    ring
  · rw [Bool.not_true, BL_true, BL_false, F.Bdn_S_Bup_z hh, cS, mL_true]
    push_cast
    ring

/-- `S_{-σ}(-z) = c_P/ρ` on the lattice of spin `σ` (`Bdn_S_neg_Bup_z`). -/
theorem S_opp_neg_z (hμ : F.μ ≠ 0) (σ : Bool) (n : ℤ) :
    (F.BL (!σ)).S k (-(F.BL σ).z k n) = F.cP / F.ρ := by
  cases σ
  · have h := F.flip.Bdn_S_neg_Bup_z hμ k n
    simp only [flip_Bdn, flip_Bup, flip_p₀, flip_p₂, flip_ρ, flip_β] at h
    rw [BL_false, Bool.not_false, BL_true, h, cP]
    push_cast
    ring
  · rw [Bool.not_true, BL_true, BL_false, F.Bdn_S_neg_Bup_z hμ, cP]
    push_cast
    ring

/-- `p₂p_σ = p₀p_{-σ}e^{2βμ_σ}`. -/
theorem p₂_mul_pL (σ : Bool) :
    F.p₂ * F.pL σ = F.p₀ * F.pL (!σ) * Real.exp (2 * F.β * F.cL σ) := by
  have key : ∀ A B C D : ℝ, A + B = C + D →
      F.p₀ * Real.exp A * (F.p₀ * Real.exp B) = F.p₀ * (F.p₀ * Real.exp C) * Real.exp D := by
    intro A B C D h
    rw [show F.p₀ * Real.exp A * (F.p₀ * Real.exp B) = F.p₀ * F.p₀ * Real.exp (A + B) by
      rw [Real.exp_add]; ring, h, Real.exp_add]
    ring
  rw [F.p₂_eq_p₀_mul]
  cases σ
  · rw [pL_false, Bool.not_false, pL_true, F.pDn_eq_p₀_mul, F.pUp_eq_p₀_mul]
    exact key _ _ _ _ (by simp only [cL, Bool.cond_false]; ring)
  · rw [pL_true, Bool.not_true, pL_false, F.pDn_eq_p₀_mul, F.pUp_eq_p₀_mul]
    exact key _ _ _ _ (by simp only [cL, Bool.cond_true]; ring)

/-- `S_σ(-z) = βc_σ/(2D_σ) = (β/2)coth βμ_σ` on the lattice of spin `σ`, for `μ_σ ≠ 0`
(Lemma "values on the reflected lattice" of the zero-field atom at `μ_σ`). -/
theorem S_self_neg_z {σ : Bool} (hc : F.cL σ ≠ 0) (n : ℤ) :
    (F.BL σ).S k (-(F.BL σ).z k n) = F.β * F.cLw σ / (2 * F.DL σ) := by
  set B := F.BL σ
  have h := B.rho_mul_S_neg_z k hc n
  have hB := B.p₂_eq_p₀_mul
  have hp := F.p₂_mul_pL σ
  simp only [B, lat_β, lat_μ] at hB
  have hE : Real.exp (2 * F.β * F.cL σ) ≠ 1 := by
    rw [Ne, Real.exp_eq_one_iff]
    exact mul_ne_zero (mul_ne_zero two_ne_zero F.β_pos.ne') hc
  have hB0 : (B.p₀ : ℂ) ≠ 0 := ofReal_ne_zero.mpr B.p₀_pos.ne'
  have h0 : (F.p₀ : ℂ) * F.pL (!σ) ≠ 0 := by
    have := F.pL_pos (!σ)
    have := F.p₀_pos
    exact_mod_cast (mul_pos ‹0 < F.p₀› ‹0 < F.pL (!σ)›).ne'
  rw [Params.ρ, Params.cbar, hB] at h
  rw [cLw, DL, show F.p₂ * F.pL σ = F.p₀ * F.pL (!σ) * Real.exp (2 * F.β * F.cL σ) from hp]
  generalize Real.exp (2 * F.β * F.cL σ) = E at *
  have hE' : (E : ℂ) - 1 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast hE)
  simp only [B, lat_β] at h
  push_cast at h ⊢
  generalize (F.BL σ).S k (-(F.BL σ).z k n) = S at *
  rw [eq_div_iff (mul_ne_zero two_ne_zero (by
    rw [show (F.p₀ : ℂ) * F.pL (!σ) * E - F.p₀ * F.pL (!σ) = F.p₀ * F.pL (!σ) * (E - 1) by ring]
    exact mul_ne_zero h0 hE'))]
  have hS : S = F.β * (1 + E) / (2 * (E - 1)) := by
    rw [eq_div_iff (mul_ne_zero two_ne_zero hE')]
    apply mul_left_cancel₀ hB0
    linear_combination 2 * h
  rw [hS]
  field_simp
  ring

/-! ### The opposite-spin kernels -/

/-- `α^P_σ(x) = ρS_{-σ}(x) - c_P`. -/
noncomputable def αPL (σ : Bool) (x : ℂ) : ℂ := F.ρ * (F.BL (!σ)).S k x - F.cP

/-- `α^S_σ(x) = -σmS_{-σ}(x) - c_S`. -/
noncomputable def αSL (σ : Bool) (x : ℂ) : ℂ := -(F.mL σ * (F.BL (!σ)).S k x) - F.cS

/-- `α_σ(x) = D_σS_σ(x) - βc_σ/2`. -/
noncomputable def αL (σ : Bool) (x : ℂ) : ℂ := F.DL σ * (F.BL σ).S k x - F.β * F.cLw σ / 2

/-- **The opposite-spin kernels on a simple pole**: for `x` off `L_{-σ}`, `x ≠ ±z`,
`∑ₗ [ρ/(z + l) - σm/(z - l)]/(l - x) = α^P_σ(x)/(z + x) + α^S_σ(x)/(z - x)`. -/
theorem hasSum_kernelL (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (σ : Bool) (n : ℤ) {x : ℂ}
    (hx : ∀ l, (F.BL (!σ)).z k l ≠ x) (hxp : x ≠ -(F.BL σ).z k n)
    (hxm : x ≠ (F.BL σ).z k n) :
    HasSum (fun l ↦ (F.ρ / ((F.BL σ).z k n + (F.BL (!σ)).z k l) -
        F.mL σ / ((F.BL σ).z k n - (F.BL (!σ)).z k l)) * (1 / ((F.BL (!σ)).z k l - x)))
      (F.αPL k σ x / ((F.BL σ).z k n + x) + F.αSL k σ x / ((F.BL σ).z k n - x)) := by
  have hS₁ := F.S_opp_neg_z k hμ σ n
  have hS₂ := F.S_opp_z k hh σ n
  have hρ := F.rho_ne_zeroT hμ
  have hmL := F.mL_ne_zero hh σ
  rw [αPL, αSL]
  set B := F.BL (!σ)
  set z := (F.BL σ).z k n
  have hp : ∀ l, B.z k l ≠ -z := fun l h0 ↦ F.zL_add_opp_ne k hμ σ n l (by
    rw [h0]; ring)
  have hm : ∀ l, B.z k l ≠ z := fun l h0 ↦ F.zL_sub_opp_ne k hh σ n l (by rw [h0]; ring)
  have h₁ := (B.hasSum_pair (k := k) (Ne.symm hxp) hp hx).mul_left (F.ρ : ℂ)
  have h₂ := (B.hasSum_pair (k := k) (Ne.symm hxm) hm hx).mul_left (F.mL σ : ℂ)
  have hzx : z + x ≠ 0 := fun h0 ↦ hxp (by linear_combination h0)
  have hzx' : z - x ≠ 0 := fun h0 ↦ hxm (by linear_combination -h0)
  convert h₁.add h₂ using 1
  · funext l
    have e₁ : z + B.z k l ≠ 0 := fun h0 ↦ hp l (by linear_combination h0)
    have e₂ : z - B.z k l ≠ 0 := fun h0 ↦ hm l (by linear_combination -h0)
    have e₃ := sub_ne_zero.mpr (hx l)
    have e₄ : B.z k l - -z ≠ 0 := by rw [sub_neg_eq_add, add_comm]; exact e₁
    have e₅ : B.z k l - z ≠ 0 := fun h0 ↦ e₂ (by linear_combination -h0)
    field_simp
    ring
  · rw [hS₁, hS₂]
    have hzx'' : -z - x ≠ 0 := fun h0 ↦ hzx (by linear_combination -h0)
    field_simp
    ring

/-- **The opposite-spin kernels on a pole of the own lattice**: for `z = z_σₙ`, `v = z_σₘ` and
`μ_σ ≠ 0`, `∑ₗ [ρ/(z + l) - σm/(z - l)]/(l - v) = (ρS - c_P)/(z + v) + δₙₘσm(S² - β²/4)` with
`S = S_{-σ}(v) = -c_S/(σm)`: the pole at `z = v` is absent, `S_{-σ}(z) = S_{-σ}(v)`. -/
theorem hasSum_kernelL_sub (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {σ : Bool} (hc : F.cL σ ≠ 0)
    (n m : ℤ) :
    HasSum (fun l ↦ (F.ρ / ((F.BL σ).z k n + (F.BL (!σ)).z k l) -
        F.mL σ / ((F.BL σ).z k n - (F.BL (!σ)).z k l)) * (1 / ((F.BL (!σ)).z k l -
          (F.BL σ).z k m)))
      ((-(F.ρ * F.cS / F.mL σ) - F.cP) / ((F.BL σ).z k n + (F.BL σ).z k m) +
        kron n m * (F.mL σ * ((F.cS / F.mL σ) ^ 2 - F.β ^ 2 / 4))) := by
  have hρ := F.rho_ne_zeroT hμ
  have hmL := F.mL_ne_zero hh σ
  have hSn' := F.S_opp_neg_z k hμ σ n
  have hSm := F.S_opp_z k hh σ m
  have hzv := F.zL_add_self_ne k hc n m
  have hp : ∀ l, (F.BL (!σ)).z k l ≠ -(F.BL σ).z k n := fun l h0 ↦
    F.zL_add_opp_ne k hμ σ n l (by rw [h0]; ring)
  have hmv : ∀ l, (F.BL (!σ)).z k l ≠ (F.BL σ).z k m := fun l h0 ↦
    F.zL_sub_opp_ne k hh σ m l (by rw [h0]; ring)
  have hzv' : -(F.BL σ).z k n ≠ (F.BL σ).z k m := fun h0 ↦ hzv (by rw [← h0]; ring)
  have hzv'' : -(F.BL σ).z k n - (F.BL σ).z k m ≠ 0 := fun h0 ↦ hzv (by linear_combination -h0)
  have h₁ := ((F.BL (!σ)).hasSum_pair (k := k) hzv' hp hmv).mul_left (F.ρ : ℂ)
  by_cases hnm : n = m
  · subst hnm
    have h₂ := ((F.BL (!σ)).hasSum_sq (k := k) hmv).mul_left (F.mL σ : ℂ)
    convert h₁.add h₂ using 1
    · funext l
      have e₁ : (F.BL σ).z k n + (F.BL (!σ)).z k l ≠ 0 := fun h0 ↦ hp l (by linear_combination h0)
      have e₂ := sub_ne_zero.mpr (hmv l)
      have e₄ : (F.BL (!σ)).z k l - -(F.BL σ).z k n ≠ 0 := by
        rw [sub_neg_eq_add, add_comm]; exact e₁
      have e₅ : (F.BL σ).z k n - (F.BL (!σ)).z k l ≠ 0 := fun h0 ↦ e₂ (by linear_combination -h0)
      field_simp
      ring
    · rw [kron_self, hSn', hSm]
      simp only [lat_β]
      field_simp
      ring
  · have hSn := F.S_opp_z k hh σ n
    have hmz : ∀ l, (F.BL (!σ)).z k l ≠ (F.BL σ).z k n := fun l h0 ↦
      F.zL_sub_opp_ne k hh σ n l (by rw [h0]; ring)
    have hne : (F.BL σ).z k n ≠ (F.BL σ).z k m := fun h0 ↦ hnm ((F.BL σ).z_injective k h0)
    have h₂ := ((F.BL (!σ)).hasSum_pair (k := k) hne hmz hmv).mul_left (F.mL σ : ℂ)
    convert h₁.add h₂ using 1
    · funext l
      have e₁ : (F.BL σ).z k n + (F.BL (!σ)).z k l ≠ 0 := fun h0 ↦ hp l (by linear_combination h0)
      have e₂ := sub_ne_zero.mpr (hmv l)
      have e₃ := sub_ne_zero.mpr (hmz l)
      have e₄ : (F.BL (!σ)).z k l - -(F.BL σ).z k n ≠ 0 := by
        rw [sub_neg_eq_add, add_comm]; exact e₁
      have e₅ : (F.BL σ).z k n - (F.BL (!σ)).z k l ≠ 0 := fun h0 ↦ e₃ (by linear_combination -h0)
      field_simp
      ring
    · rw [kron_of_ne hnm, hSn', hSn, hSm, sub_self, zero_div, mul_zero]
      field_simp
      ring

/-- **The opposite-spin kernels on a reflected pole of the own lattice**: for `z = z_σₙ`,
`v = z_σₘ` and `μ_σ ≠ 0`,
`∑ₗ [ρ/(z + l) - σm/(z - l)]/(l + v) = δₙₘρ(S'² - β²/4) + σm(S - S')/(z + v)` with
`S = S_{-σ}(z) = -c_S/(σm)`, `S' = S_{-σ}(-v) = c_P/ρ`. -/
theorem hasSum_kernelL_add (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {σ : Bool} (hc : F.cL σ ≠ 0)
    (n m : ℤ) :
    HasSum (fun l ↦ (F.ρ / ((F.BL σ).z k n + (F.BL (!σ)).z k l) -
        F.mL σ / ((F.BL σ).z k n - (F.BL (!σ)).z k l)) * (1 / ((F.BL (!σ)).z k l +
          (F.BL σ).z k m)))
      (kron n m * (F.ρ * ((F.cP / F.ρ) ^ 2 - F.β ^ 2 / 4)) +
        F.mL σ * (-(F.cS / F.mL σ) - F.cP / F.ρ) / ((F.BL σ).z k n + (F.BL σ).z k m)) := by
  have hρ := F.rho_ne_zeroT hμ
  have hmL := F.mL_ne_zero hh σ
  have hSn := F.S_opp_z k hh σ n
  have hSm' := F.S_opp_neg_z k hμ σ m
  have hzv := F.zL_add_self_ne k hc n m
  have hmz : ∀ l, (F.BL (!σ)).z k l ≠ (F.BL σ).z k n := fun l h0 ↦
    F.zL_sub_opp_ne k hh σ n l (by rw [h0]; ring)
  have hpv : ∀ l, (F.BL (!σ)).z k l ≠ -(F.BL σ).z k m := fun l h0 ↦
    F.zL_add_opp_ne k hμ σ m l (by rw [h0]; ring)
  have hzv' : (F.BL σ).z k n ≠ -(F.BL σ).z k m := fun h0 ↦ hzv (by rw [h0]; ring)
  have hzv'' : (F.BL σ).z k n - -(F.BL σ).z k m ≠ 0 := by rw [sub_neg_eq_add]; exact hzv
  have h₂ := ((F.BL (!σ)).hasSum_pair (k := k) hzv' hmz hpv).mul_left (F.mL σ : ℂ)
  by_cases hnm : n = m
  · subst hnm
    have h₁ := ((F.BL (!σ)).hasSum_sq (k := k) hpv).mul_left (F.ρ : ℂ)
    convert h₁.add h₂ using 1
    · funext l
      have e₁ := sub_ne_zero.mpr (hpv l)
      have e₂ : (F.BL σ).z k n + (F.BL (!σ)).z k l ≠ 0 := fun h0 ↦ e₁ (by linear_combination h0)
      have e₃ := sub_ne_zero.mpr (hmz l)
      have e₅ : (F.BL σ).z k n - (F.BL (!σ)).z k l ≠ 0 := fun h0 ↦ e₃ (by linear_combination -h0)
      have e₆ : (F.BL (!σ)).z k l + (F.BL σ).z k n ≠ 0 := fun h0 ↦ e₁ (by linear_combination h0)
      field_simp
      ring
    · rw [kron_self, hSm', hSn]
      simp only [lat_β]
      field_simp
      ring
  · have hSn' := F.S_opp_neg_z k hμ σ n
    have hp : ∀ l, (F.BL (!σ)).z k l ≠ -(F.BL σ).z k n := fun l h0 ↦
      F.zL_add_opp_ne k hμ σ n l (by rw [h0]; ring)
    have hne : -(F.BL σ).z k n ≠ -(F.BL σ).z k m := fun h0 ↦
      hnm ((F.BL σ).z_injective k (neg_inj.mp h0))
    have h₁ := ((F.BL (!σ)).hasSum_pair (k := k) hne hp hpv).mul_left (F.ρ : ℂ)
    convert h₁.add h₂ using 1
    · funext l
      have e₁ := sub_ne_zero.mpr (hpv l)
      have e₀ := sub_ne_zero.mpr (hp l)
      have e₂ : (F.BL σ).z k n + (F.BL (!σ)).z k l ≠ 0 := fun h0 ↦ e₀ (by linear_combination h0)
      have e₃ := sub_ne_zero.mpr (hmz l)
      have e₅ : (F.BL σ).z k n - (F.BL (!σ)).z k l ≠ 0 := fun h0 ↦ e₃ (by linear_combination -h0)
      have e₆ : (F.BL (!σ)).z k l + (F.BL σ).z k m ≠ 0 := fun h0 ↦ e₁ (by linear_combination h0)
      field_simp
      ring
    · rw [kron_of_ne hnm, hSn', hSm', hSn, sub_self, zero_div, mul_zero, zero_mul, zero_add]
      field_simp
      ring

/-! ### The thermal values at the zeros of `P` -/

/-- `S₀(-a) = β(p_σ - p₀)/(2(p₀ + p_σ))` on the lattice of spin `σ`. -/
theorem BL_S₀_neg_a (σ : Bool) :
    (F.BL σ).S 0 (-F.a) = F.β * (F.pL σ - F.p₀) / (2 * (F.p₀ + F.pL σ)) := by
  cases σ
  · simpa only [BL_false, pL_false] using F.Bdn_S₀_neg_a
  · simpa only [BL_true, pL_true] using F.Bup_S₀_neg_a

/-- `S₀(a) = β(p₂ - p_{-σ})/(2(p₂ + p_{-σ}))` on the lattice of spin `σ`. -/
theorem BL_S₀_a (σ : Bool) :
    (F.BL σ).S 0 F.a = F.β * (F.p₂ - F.pL (!σ)) / (2 * (F.p₂ + F.pL (!σ))) := by
  cases σ
  · simpa only [BL_false, pL_true, Bool.not_false] using F.Bdn_S₀_a
  · simpa only [BL_true, pL_false, Bool.not_true] using F.Bup_S₀_a

/-! ### The product relations -/

/-- `e^{2y} + 1 = 2e^y cosh y`. -/
theorem exp_two_mul_add_one (y : ℂ) : exp (2 * y) + 1 = 2 * exp y * cosh y := by
  have := exp_ne_zero y
  rw [Complex.cosh, exp_neg, show 2 * y = y + y by ring, exp_add]
  field_simp

/-- `e^{2y} - 1 = 2e^y sinh y`. -/
theorem exp_two_mul_sub_one (y : ℂ) : exp (2 * y) - 1 = 2 * exp y * sinh y := by
  have := exp_ne_zero y
  rw [Complex.sinh, exp_neg, show 2 * y = y + y by ring, exp_add]
  field_simp

/-- The thermal sum on a lattice of centre `c` through `X = e^{β(x - μ)}` and `E = e^{β(μ - c)}`:
`S(x) = -(β/2)(XE - 1)/(XE + 1)` for even `k`, the inverse fraction for odd `k`. -/
theorem S_lat_eq_exp (c : ℝ) {x X E : ℂ} (hX : exp (F.β * (x - F.μ)) = X)
    (hE : exp ((F.β * (F.μ - c) : ℝ) : ℂ) = E) :
    (F.lat c).S k x = -((F.β : ℂ) / 2) *
      (if Even k then (X * E - 1) / (X * E + 1) else (X * E + 1) / (X * E - 1)) := by
  have e : 2 * ((F.β : ℂ) * (x - c) / 2) = F.β * (x - F.μ) + ((F.β * (F.μ - c) : ℝ) : ℂ) := by
    push_cast
    ring
  rw [Params.S, lat_β, lat_μ]
  split_ifs
  · rw [tanh_eq_exp, e, exp_add, hX, hE]
  · rw [coth_eq_exp, e, exp_add, hX, hE]

/-- The off-lattice condition through `X = e^{β(x - μ)}` and `E = e^{β(μ - c)}`: `XE + 1 ≠ 0` for
even `k`, `XE - 1 ≠ 0` for odd `k`. -/
theorem exp_off_lat {c : ℝ} {x X E : ℂ} (hX : exp (F.β * (x - F.μ)) = X)
    (hE : exp ((F.β * (F.μ - c) : ℝ) : ℂ) = E) (hx : ∀ n, (F.lat c).z k n ≠ x) :
    (if Even k then X * E + 1 else X * E - 1) ≠ 0 := by
  have e : X * E = exp (2 * ((F.lat c).β * (x - (F.lat c).μ) / 2)) := by
    rw [← hX, ← hE, ← exp_add, lat_β, lat_μ]
    push_cast
    ring_nf
  rw [e]
  obtain ⟨j, hj | hj⟩ := Int.even_or_odd' k
  · have hk : Even k := ⟨j, by omega⟩
    rw [ite_eq_left hk, exp_two_mul_add_one]
    exact mul_ne_zero (mul_ne_zero two_ne_zero (exp_ne_zero _))
      ((F.lat c).cosh_ne_zero_of_off hj hx)
  · have hk : ¬Even k := by rintro ⟨r, hr⟩; omega
    rw [ite_eq_right hk, exp_two_mul_sub_one]
    exact mul_ne_zero (mul_ne_zero two_ne_zero (exp_ne_zero _))
      ((F.lat c).sinh_ne_zero_of_off (by omega : k = 2 * (j + 1) - 1) hx)

/-- `p↑ = p↓e^{2βh}`. -/
theorem pUp_eq_pDn_mul : (F.pUp : ℂ) = F.pDn * exp (F.β * F.h) ^ 2 := by
  rw [← exp_nat_mul, F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul]
  push_cast
  rw [mul_assoc, ← exp_add]
  ring_nf

/-- `p₂ = p₀e^{2βμ}`. -/
theorem p₂_eq_p₀_mul_sq : (F.p₂ : ℂ) = F.p₀ * exp (F.β * F.μ) ^ 2 := by
  rw [← exp_nat_mul, F.p₂_eq_p₀_mul]
  push_cast
  ring_nf

theorem exp_up : exp ((F.β * (F.μ - (F.μ + F.h)) : ℝ) : ℂ) = (exp (F.β * F.h))⁻¹ := by
  rw [← exp_neg]
  push_cast
  ring_nf

theorem exp_dn : exp ((F.β * (F.μ - (F.μ - F.h)) : ℝ) : ℂ) = exp (F.β * F.h) := by
  push_cast
  ring_nf

/-- `(AB⁻¹ - 1)/(AB⁻¹ + 1) = (A - B)/(A + B)`, also with the fraction inverted. -/
theorem frac_mul_inv {A B : ℂ} (hB : B ≠ 0) :
    (A * B⁻¹ - 1) / (A * B⁻¹ + 1) = (A - B) / (A + B) ∧
      (A * B⁻¹ + 1) / (A * B⁻¹ - 1) = (A + B) / (A - B) := by
  constructor <;> rw [div_eq_div_iff_comm] <;> field_simp

theorem add_ne_of_mul_inv {A B : ℂ} (hB : B ≠ 0) (h : A * B⁻¹ + 1 ≠ 0) : A + B ≠ 0 :=
  fun h0 ↦ h (by rw [show A = -B by linear_combination h0]; field_simp; ring)

theorem sub_ne_of_mul_inv {A B : ℂ} (hB : B ≠ 0) (h : A * B⁻¹ - 1 ≠ 0) : A - B ≠ 0 :=
  fun h0 ↦ h (by rw [show A = B by linear_combination h0]; field_simp; ring)

/-- **Product relation, spin flip**: `α^S↑(x)α^S↓(x) = β²p↑p↓` for `x` off both lattices. -/
theorem αSL_mul_αSL_true {x : ℂ} (hx₁ : ∀ n, F.Bup.z k n ≠ x) (hx₂ : ∀ n, F.Bdn.z k n ≠ x) :
    F.αSL k true x * F.αSL k false x = F.β ^ 2 * F.pUp * F.pDn := by
  have h₁ := F.exp_off_lat k rfl F.exp_up hx₁
  have h₂ := F.exp_off_lat k rfl F.exp_dn hx₂
  have hq := F.pUp_eq_pDn_mul
  rw [αSL, αSL, Bool.not_true, Bool.not_false, BL_false, BL_true,
    F.S_lat_eq_exp k _ rfl F.exp_dn, F.S_lat_eq_exp k _ rfl F.exp_up, mL_true, mL_false, cS, mf]
  push_cast
  rw [hq]
  generalize exp (F.β * (x - F.μ)) = X at *
  have hq0 := exp_ne_zero ((F.β : ℂ) * F.h)
  generalize exp ((F.β : ℂ) * F.h) = q at *
  obtain ⟨e₁, e₂⟩ := frac_mul_inv (A := X) hq0
  split_ifs at h₁ h₂ ⊢
  · have h₁' := add_ne_of_mul_inv hq0 h₁
    rw [e₁]
    generalize hD₁ : X + q = D₁ at h₁' ⊢
    generalize hD₂ : X * q + 1 = D₂ at h₂ ⊢
    field_simp
    subst hD₁ hD₂
    ring
  · have h₁' := sub_ne_of_mul_inv hq0 h₁
    rw [e₂]
    generalize hD₁ : X - q = D₁ at h₁' ⊢
    generalize hD₂ : X * q - 1 = D₂ at h₂ ⊢
    field_simp
    subst hD₁ hD₂
    ring

/-- **Product relation, spin flip**: `α^S_σ(x)α^S_{-σ}(x) = β²p↑p↓`. -/
theorem αSL_mul_αSL (σ : Bool) {x : ℂ} (hx₁ : ∀ n, F.Bup.z k n ≠ x)
    (hx₂ : ∀ n, F.Bdn.z k n ≠ x) :
    F.αSL k σ x * F.αSL k (!σ) x = F.β ^ 2 * F.pUp * F.pDn := by
  cases σ
  · rw [Bool.not_false, mul_comm]
    exact F.αSL_mul_αSL_true k hx₁ hx₂
  · exact F.αSL_mul_αSL_true k hx₁ hx₂

/-- **Product relation, pair**: `α^P↑(x)α^P↓(-x) = β²p₀p₂` for `x` off `L↓`, `-x` off `L↑`. -/
theorem αPL_mul_αPL_true {x : ℂ} (hx₁ : ∀ n, F.Bup.z k n ≠ -x) (hx₂ : ∀ n, F.Bdn.z k n ≠ x) :
    F.αPL k true x * F.αPL k false (-x) = F.β ^ 2 * F.p₀ * F.p₂ := by
  have hX : exp (F.β * (-x - F.μ)) =
      (exp (F.β * (x - F.μ)) * exp (F.β * F.μ) ^ 2 * exp (F.β * F.h))⁻¹ * exp (F.β * F.h) := by
    rw [← exp_nat_mul, ← exp_add, ← exp_add, ← exp_neg, ← exp_add]
    push_cast
    ring_nf
  have h₁ := F.exp_off_lat k hX F.exp_up hx₁
  have h₂ := F.exp_off_lat k rfl F.exp_dn hx₂
  rw [αPL, αPL, Bool.not_true, Bool.not_false, BL_false, BL_true,
    F.S_lat_eq_exp k _ rfl F.exp_dn, F.S_lat_eq_exp k _ hX F.exp_up, cP, ρ]
  push_cast
  rw [F.p₂_eq_p₀_mul_sq]
  have hX0 := exp_ne_zero (F.β * (x - F.μ))
  generalize exp (F.β * (x - F.μ)) = X at *
  have he := exp_ne_zero ((F.β : ℂ) * F.μ)
  generalize exp ((F.β : ℂ) * F.μ) = e at *
  have hq0 := exp_ne_zero ((F.β : ℂ) * F.h)
  generalize exp ((F.β : ℂ) * F.h) = q at *
  have hY : X * e ^ 2 * q ≠ 0 := mul_ne_zero (mul_ne_zero hX0 (pow_ne_zero 2 he)) hq0
  rw [show (X * e ^ 2 * q)⁻¹ * q * q⁻¹ = (X * e ^ 2 * q)⁻¹ by field_simp] at h₁ ⊢
  obtain ⟨e₁, e₂⟩ := frac_mul_inv (A := 1) hY
  rw [one_mul] at e₁ e₂
  split_ifs at h₁ h₂ ⊢
  · have h₁' := add_ne_of_mul_inv hY (by rwa [one_mul])
    rw [e₁]
    generalize hD₁ : 1 + X * e ^ 2 * q = D₁ at h₁' ⊢
    generalize hD₂ : X * q + 1 = D₂ at h₂ ⊢
    field_simp
    subst hD₁ hD₂
    ring
  · have h₁' := sub_ne_of_mul_inv hY (by rwa [one_mul])
    rw [e₂]
    generalize hD₁ : 1 - X * e ^ 2 * q = D₁ at h₁' ⊢
    generalize hD₂ : X * q - 1 = D₂ at h₂ ⊢
    field_simp
    subst hD₁ hD₂
    ring

/-- **Product relation, pair**: `α^P_σ(x)α^P_{-σ}(-x) = β²p₀p₂`. -/
theorem αPL_mul_αPL (σ : Bool) {x : ℂ} (hx₁ : ∀ n, (F.BL σ).z k n ≠ -x)
    (hx₂ : ∀ n, (F.BL (!σ)).z k n ≠ x) :
    F.αPL k σ x * F.αPL k (!σ) (-x) = F.β ^ 2 * F.p₀ * F.p₂ := by
  cases σ
  · have := F.αPL_mul_αPL_true k (x := -x) (by simpa using hx₂) (by simpa using hx₁)
    rw [neg_neg] at this
    rw [Bool.not_false, mul_comm, this]
  · exact F.αPL_mul_αPL_true k hx₁ hx₂

/-! ### The blocks of `Pχ_L` on simple poles -/

/-- The coefficients of `N(z, t) = N₀_{-σ}(t) + z²N₂_{-σ}(t)` in `t` (`NL_cubic`). -/
noncomputable def cN0L (z : ℂ) : ℂ :=
  4 * F.a ^ 4 * kron k 0 * F.ζL + 8 * F.a ^ 3 * F.q₀L k -
    2 * F.a ^ 2 * kron k 0 * F.q₀L k * F.ζL - F.a ^ 2 * kron k 0 * z ^ 2 * F.ζL -
    2 * F.a * F.q₀L k ^ 2 - 2 * F.a * F.q₀L k * z ^ 2 - kron k 0 * F.q₀L k ^ 2 * F.ζL

noncomputable def cN1L (σ : Bool) (z : ℂ) : ℂ :=
  -4 * F.a ^ 2 * F.rL (!σ) * (-4 * F.a ^ 2 + 2 * F.q₀L k + z ^ 2)

noncomputable def cN2L (z : ℂ) : ℂ :=
  -(F.a ^ 2 * kron k 0 * F.ζL) - 2 * F.a * F.q₀L k - 2 * F.a * z ^ 2 + kron k 0 * z ^ 2 * F.ζL

noncomputable def cN3L (σ : Bool) : ℂ := -4 * F.a ^ 2 * F.rL (!σ)

theorem NL_eq_cubic (σ : Bool) (z t : ℂ) :
    F.N₀L k (F.rL (!σ)) t + z ^ 2 * F.N₂L k (F.rL (!σ)) t =
      F.cN0L k z + F.cN1L k σ z * t + F.cN2L k z * t ^ 2 + F.cN3L σ * t ^ 3 := by
  rw [F.NL_cubic, cN0L, cN1L, cN2L, cN3L]

/-- `R̂(z, x)` of the opposite-spin rank-two part of row `σ`, on the lattice of spin `-σ`. -/
noncomputable def RhatL (σ : Bool) (z x : ℂ) : ℂ :=
  (F.BL (!σ)).RhatC k (F.cN0L k z) (F.cN1L k σ z) (F.cN2L k z) (F.cN3L σ) x

/-- `R̂(x)` of the same-spin rank-one part (`k = 0`) of row `σ`, for `t² - a²`. -/
noncomputable def RhatSL (σ : Bool) (x : ℂ) : ℂ := (F.BL σ).RhatC k (-F.a ^ 2) 0 1 0 x

theorem PL_BL (σ : Bool) (z : ℂ) : (F.BL σ).Pph k z = F.PL k z := rfl

theorem PL_BL_ne (σ : Bool) (j : ℤ) : F.PL k ((F.BL σ).z k j) ≠ 0 :=
  (F.BL σ).Pph_z_ne_zero k j

/-- **The same-spin block on a simple pole**, times `P(x)`: for `x` off `L_σ`,
`∑ₗ P(z)χ_σσ(z, l)P(x)/(l - x) = -βH_σ(z)P(x)/(z - x) + δ_k0 ζ_σ(z² - a²)[(x² - a²)S_σ(x) + R̂]`. -/
theorem hasSum_self_pole (ha : (F.a : ℂ) ≠ 0) (σ : Bool) (n : ℤ) {x : ℂ}
    (hx : ∀ l, (F.BL σ).z k l ≠ x) :
    HasSum (fun l ↦ F.PL k ((F.BL σ).z k n) * F.chiL σ σ k n l *
        (F.PL k x / ((F.BL σ).z k l - x)))
      (-F.β * F.HL k (F.rL σ) ((F.BL σ).z k n) * F.PL k x / ((F.BL σ).z k n - x) +
        kron k 0 * F.ζLs σ * ((F.BL σ).z k n ^ 2 - F.a ^ 2) *
          ((x ^ 2 - F.a ^ 2) * (F.BL σ).S k x + F.RhatSL k σ x)) := by
  have h₁ := hasSum_single (f := fun l ↦ -F.β * F.HL k (F.rL σ) ((F.BL σ).z k n) * kron n l *
    (F.PL k x / ((F.BL σ).z k l - x))) n
    (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_mul])
  have h₂ := ((F.BL σ).hasSum_cubic_mul k ha (-F.a ^ 2) 0 1 0 hx).mul_left
    (kron k 0 * F.ζLs σ * ((F.BL σ).z k n ^ 2 - F.a ^ 2))
  simp only [PL_BL] at h₂
  convert h₁.add h₂ using 1
  · funext l
    have hPl := F.PL_BL_ne k σ l
    rw [F.PL_mul_chiL_self k σ n l]
    by_cases hnl : n = l
    · subst hnl
      rw [kron_self]
      field_simp
      ring
    · rw [kron_of_ne hnl]
      field_simp
      ring
  · rw [kron_self, RhatSL]
    ring

/-- **The opposite-spin block on a simple pole**, times `P(x)`: for `x` off `L_{-σ}`, `x ≠ ±z`,
`∑ₗ P(z)χ_{σ,-σ}(z, l)P(x)/(l - x) = λP(x)[α^P_σ(x)/(z + x) + α^S_σ(x)/(z - x)] +
N(z, x)S_{-σ}(x) + R̂(z, x)`. -/
theorem hasSum_opp_pole (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (σ : Bool) (n : ℤ)
    {x : ℂ} (hx : ∀ l, (F.BL (!σ)).z k l ≠ x) (hxp : x ≠ -(F.BL σ).z k n)
    (hxm : x ≠ (F.BL σ).z k n) :
    HasSum (fun l ↦ F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n l *
        (F.PL k x / ((F.BL (!σ)).z k l - x)))
      (F.lamL * F.PL k x * (F.αPL k σ x / ((F.BL σ).z k n + x) +
          F.αSL k σ x / ((F.BL σ).z k n - x)) +
        ((F.N₀L k (F.rL (!σ)) x + (F.BL σ).z k n ^ 2 * F.N₂L k (F.rL (!σ)) x) *
          (F.BL (!σ)).S k x + F.RhatL k σ ((F.BL σ).z k n) x)) := by
  have h₁ := (F.hasSum_kernelL k hμ hh σ n hx hxp hxm).mul_left (F.lamL * F.PL k x)
  have h₂ := (F.BL (!σ)).hasSum_cubic_mul k ha (F.cN0L k ((F.BL σ).z k n))
    (F.cN1L k σ ((F.BL σ).z k n)) (F.cN2L k ((F.BL σ).z k n)) (F.cN3L σ) hx
  simp only [PL_BL] at h₂
  convert h₁.add h₂ using 1
  · funext l
    have hPl := F.PL_BL_ne k (!σ) l
    have e₁ := sub_ne_zero.mpr (hx l)
    have e₂ := F.zL_add_opp_ne k hμ σ n l
    have e₃ := F.zL_sub_opp_ne k hh σ n l
    rw [F.PL_mul_chiL_opp k hμ hh σ n l, F.NL_eq_cubic]
    field_simp
  · rw [F.NL_eq_cubic, RhatL]

/-- The opposite-spin block on the pole at `v = z_σₘ` of the own lattice, times `P(v)`. -/
theorem hasSum_opp_pole_sub (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {σ : Bool}
    (hc : F.cL σ ≠ 0) (n m : ℤ) :
    HasSum (fun l ↦ F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n l *
        (F.PL k ((F.BL σ).z k m) / ((F.BL (!σ)).z k l - (F.BL σ).z k m)))
      (F.lamL * F.PL k ((F.BL σ).z k m) *
          ((-(F.ρ * F.cS / F.mL σ) - F.cP) / ((F.BL σ).z k n + (F.BL σ).z k m) +
            kron n m * (F.mL σ * ((F.cS / F.mL σ) ^ 2 - F.β ^ 2 / 4))) +
        ((F.N₀L k (F.rL (!σ)) ((F.BL σ).z k m) +
            (F.BL σ).z k n ^ 2 * F.N₂L k (F.rL (!σ)) ((F.BL σ).z k m)) * -(F.cS / F.mL σ) +
          F.RhatL k σ ((F.BL σ).z k n) ((F.BL σ).z k m))) := by
  have hx : ∀ l, (F.BL (!σ)).z k l ≠ (F.BL σ).z k m := fun l h0 ↦
    F.zL_sub_opp_ne k hh σ m l (by rw [h0]; ring)
  have h₁ := (F.hasSum_kernelL_sub k hμ hh hc n m).mul_left (F.lamL * F.PL k ((F.BL σ).z k m))
  have h₂ := (F.BL (!σ)).hasSum_cubic_mul k ha (F.cN0L k ((F.BL σ).z k n))
    (F.cN1L k σ ((F.BL σ).z k n)) (F.cN2L k ((F.BL σ).z k n)) (F.cN3L σ) hx
  rw [F.S_opp_z k hh σ m] at h₂
  simp only [PL_BL] at h₂
  convert h₁.add h₂ using 1
  · funext l
    have hPl := F.PL_BL_ne k (!σ) l
    have e₁ := sub_ne_zero.mpr (hx l)
    have e₂ := F.zL_add_opp_ne k hμ σ n l
    have e₃ := F.zL_sub_opp_ne k hh σ n l
    rw [F.PL_mul_chiL_opp k hμ hh σ n l, F.NL_eq_cubic]
    field_simp
  · rw [F.NL_eq_cubic, RhatL]

/-- The opposite-spin block on the reflected pole at `-v`, `v = z_σₘ`, times `P(v)`. -/
theorem hasSum_opp_pole_add (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {σ : Bool}
    (hc : F.cL σ ≠ 0) (n m : ℤ) :
    HasSum (fun l ↦ F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n l *
        (F.PL k ((F.BL σ).z k m) / ((F.BL (!σ)).z k l + (F.BL σ).z k m)))
      (F.lamL * F.PL k ((F.BL σ).z k m) *
          (kron n m * (F.ρ * ((F.cP / F.ρ) ^ 2 - F.β ^ 2 / 4)) +
            F.mL σ * (-(F.cS / F.mL σ) - F.cP / F.ρ) / ((F.BL σ).z k n + (F.BL σ).z k m)) +
        ((F.N₀L k (F.rL (!σ)) (-(F.BL σ).z k m) +
            (F.BL σ).z k n ^ 2 * F.N₂L k (F.rL (!σ)) (-(F.BL σ).z k m)) * (F.cP / F.ρ) +
          F.RhatL k σ ((F.BL σ).z k n) (-(F.BL σ).z k m))) := by
  have hx : ∀ l, (F.BL (!σ)).z k l ≠ -(F.BL σ).z k m := fun l h0 ↦
    F.zL_add_opp_ne k hμ σ m l (by rw [h0]; ring)
  have h₁ := (F.hasSum_kernelL_add k hμ hh hc n m).mul_left (F.lamL * F.PL k ((F.BL σ).z k m))
  have h₂ := (F.BL (!σ)).hasSum_cubic_mul k ha (F.cN0L k ((F.BL σ).z k n))
    (F.cN1L k σ ((F.BL σ).z k n)) (F.cN2L k ((F.BL σ).z k n)) (F.cN3L σ) hx
  rw [F.S_opp_neg_z k hμ σ m] at h₂
  have hPv : F.PL k (-(F.BL σ).z k m) = F.PL k ((F.BL σ).z k m) := by simp only [PL]; ring
  simp only [PL_BL] at h₂
  rw [hPv] at h₂
  convert h₁.add h₂ using 1
  · funext l
    have hPl := F.PL_BL_ne k (!σ) l
    have e₁ := sub_ne_zero.mpr (hx l)
    have e₁' : (F.BL (!σ)).z k l + (F.BL σ).z k m ≠ 0 := by rwa [sub_neg_eq_add] at e₁
    have e₂ := F.zL_add_opp_ne k hμ σ n l
    have e₃ := F.zL_sub_opp_ne k hh σ n l
    rw [F.PL_mul_chiL_opp k hμ hh σ n l, F.NL_eq_cubic]
    field_simp
    ring
  · rw [F.NL_eq_cubic, RhatL]

/-- **Product relation, same spin**: `α_σ(x)α_σ(-x) = β²p₀p₂p↑p↓` for `±x` off the lattice of spin
`σ`, `μ_σ ≠ 0`: `α_σ = (D_σ/ρ_B)α` with `α` of the zero-field atom at `μ_σ` (Eq. (alpha)). -/
theorem αL_mul_αL_neg {σ : Bool} (hc : F.cL σ ≠ 0) {x : ℂ} (hx : ∀ n, (F.BL σ).z k n ≠ x)
    (hx' : ∀ n, (F.BL σ).z k n ≠ -x) :
    F.αL k σ x * F.αL k σ (-x) = F.β ^ 2 * (F.p₀ * F.p₂) * (F.pUp * F.pDn) := by
  have h := (F.BL σ).α_mul_α_neg k hx hx'
  have hB := (F.BL σ).p₂_eq_p₀_mul
  have hp := F.p₂_mul_pL σ
  have hB0 : ((F.BL σ).p₀ : ℂ) ≠ 0 := ofReal_ne_zero.mpr (F.BL σ).p₀_pos.ne'
  have hpσ : (F.pL σ : ℂ) ≠ 0 := ofReal_ne_zero.mpr (F.pL_pos σ).ne'
  have hpp : (F.pUp : ℂ) * F.pDn = F.pL σ * F.pL (!σ) := by
    cases σ
    · simp only [pL_false, Bool.not_false, pL_true]
      ring
    · simp only [pL_true, Bool.not_true, pL_false]
  have hE : Real.exp (2 * F.β * F.cL σ) ≠ 1 := by
    rw [Ne, Real.exp_eq_one_iff]
    exact mul_ne_zero (mul_ne_zero two_ne_zero F.β_pos.ne') hc
  simp only [lat_β, lat_μ] at hB
  rw [Params.α, Params.α, Params.ρ, Params.cbar, hB] at h
  simp only [lat_β] at h
  rw [αL, αL, cLw, DL, hpp]
  generalize Real.exp (2 * F.β * F.cL σ) = E at hB hp hE h
  have hX : (F.p₂ : ℂ) * F.pL σ = F.p₀ * F.pL (!σ) * E := by exact_mod_cast hp
  push_cast at h ⊢
  rw [show (F.β : ℂ) ^ 2 * (F.p₀ * F.p₂) * (F.pL σ * F.pL (!σ)) =
    F.β ^ 2 * F.p₀ * F.pL (!σ) * (F.p₂ * F.pL σ) by ring, hX]
  generalize (F.BL σ).S k x = S₁ at *
  generalize (F.BL σ).S k (-x) = S₂ at *
  have hS : (S₁ * (E - 1) - (1 + E) * F.β / 2) * (S₂ * (E - 1) - (1 + E) * F.β / 2) =
      F.β ^ 2 * E := by
    apply mul_left_cancel₀ (pow_ne_zero 2 hB0)
    linear_combination h
  linear_combination (F.p₀ * F.pL (!σ)) ^ 2 * hS

end FParams

end HubbardAtom
