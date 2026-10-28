import HubbardAtomLean.FieldLCompl

/-!
# The images of the column of the longitudinal inverse

The column `(t, m)` of the inverse, `v = z_m` on the lattice of the spin `t`, is a dense part
`P(v)[d̂δ_lm + Â/(l + v)]` on the spin `t` and `P(v)[B̂/(l - v) + Ĉ/(l + v)] + U/β²` on the spin
`-t` (`denseL`), minus the modes `P(u)[f₊/(l - u) + f₋/(l + u)]` of the roots of `E₈` (`modeL`).
Their images under `P(z)χ_L` in the row `n` of the spin `t` (`s`) and of the spin `-t` (`o`):

* `hasSum_const_opp`, `hasSum_const_self`: the constant column, by the Ward identities
  (`hasSum_chiL_updn`, …);
* `hasSum_dense_s`, `hasSum_dense_o`: the dense part is `P(v)δ_nm + ∑_c zᶜD_c(v)`, the residues at
  `z = ∓v` vanishing by `dense_pole`, `dense_pole_v`, `dense_pole_neg_v`, the diagonal by
  `dense_delta`;
* `hasSum_mode_s`, `hasSum_mode_o`: the mode of a root `u` is `∑_c zᶜM_c(u)`, the residues at
  `z = ±u` vanishing by `res_self_pos`, … .
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

-- the value identities are `field_simp`/`ring` computations: one at a time
set_option Elab.async false

theorem PL_neg (z : ℂ) : F.PL k (-z) = F.PL k z := by
  simp only [PL]
  ring

/-- At `k = 0`, `P(z) = (z² - a²)²`. -/
theorem PL_zero (z : ℂ) : F.PL 0 z = (z ^ 2 - F.a ^ 2) ^ 2 := by
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  simp only [PL, hw]
  ring

/-! ### The constant column -/

/-- **The constant column on the opposite spin**, by the Ward identity: off the axes,
`∑ₗ P(z)χ_{σ,-σ}(z, l) · 2a/β² = δ_k0 ζ_σ(z² - a²)/β`. -/
theorem hasSum_const_opp (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (σ : Bool) (n : ℤ) :
    HasSum (fun l ↦ F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n l * (2 * F.a / F.β ^ 2))
      (kron k 0 * F.ζLs σ * ((F.BL σ).z k n ^ 2 - F.a ^ 2) / F.β) := by
  have hβ := β_ne_zeroT (F := F)
  have hP := F.PL_BL_ne k σ n
  have hW : HasSum (fun l ↦ F.chiL σ (!σ) k n l)
      (F.β * (kron k 0 * F.ζLs σ / (2 * F.a * ((F.BL σ).z k n ^ 2 - F.a ^ 2)))) := by
    cases σ
    · have h := F.hasSum_chiL_dnup k hμ hh n
      simp only [fLUpDn, ζUp_eq, flip_ζLs, flip_a, Bool.not_true] at h
      simp only [Bool.not_false, BL_false]
      convert h using 1
      ring
    · have h := F.hasSum_chiL_updn k hμ hh n
      simp only [fLUpDn, ζUp_eq] at h
      simp only [Bool.not_true, BL_true]
      convert h using 1
      ring
  convert (hW.mul_left (F.PL k ((F.BL σ).z k n))).mul_right (2 * (F.a : ℂ) / (F.β : ℂ) ^ 2) using 1
  by_cases hk : k = 0
  · subst hk
    rw [F.PL_zero] at hP ⊢
    have hz : (F.BL σ).z 0 n ^ 2 - F.a ^ 2 ≠ 0 := fun h0 ↦ hP (by rw [h0]; ring)
    field_simp
  · simp only [kron_of_ne hk, zero_mul, zero_div, mul_zero]

/-- **The constant column on the same spin**, by the Ward identity:
`∑ₗ P(z)χ_σσ(z, l) · 2a/β² = -(2a/β)H_σ(z) + δ_k0 ζ(z² - a²)/β`. -/
theorem hasSum_const_self (ha : (F.a : ℂ) ≠ 0) (σ : Bool) (n : ℤ) :
    HasSum (fun l ↦ F.PL k ((F.BL σ).z k n) * F.chiL σ σ k n l * (2 * F.a / F.β ^ 2))
      (-(2 * F.a / F.β) * F.HL k (F.rL σ) ((F.BL σ).z k n) +
        kron k 0 * F.ζL * ((F.BL σ).z k n ^ 2 - F.a ^ 2) / F.β) := by
  have hβ := β_ne_zeroT (F := F)
  have hP := F.PL_BL_ne k σ n
  have hW : HasSum (fun l ↦ F.chiL σ σ k n l) (-F.β * F.fL k (F.rL σ) ((F.BL σ).z k n)) := by
    cases σ
    · simpa using F.hasSum_chiL_dndn k n
    · simpa using F.hasSum_chiL_upup k n
  convert (hW.mul_left (F.PL k ((F.BL σ).z k n))).mul_right (2 * (F.a : ℂ) / (F.β : ℂ) ^ 2) using 1
  by_cases hk : k = 0
  · subst hk
    have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
    rw [F.PL_zero] at hP ⊢
    have hz : (F.BL σ).z 0 n ^ 2 - F.a ^ 2 ≠ 0 := fun h0 ↦ hP (by rw [h0]; ring)
    have h₁ : ((F.BL σ).z 0 n + F.a) ^ 2 ≠ 0 :=
      pow_ne_zero 2 fun h0 ↦ hz (by linear_combination ((F.BL σ).z 0 n - F.a) * h0)
    have h₂ : ((F.BL σ).z 0 n - F.a) ^ 2 ≠ 0 :=
      pow_ne_zero 2 fun h0 ↦ hz (by linear_combination ((F.BL σ).z 0 n + F.a) * h0)
    simp only [fL, HL, q₀L, hw, kron_self]
    field_simp
    ring
  · simp only [PL] at hP
    have h₁ := left_ne_zero_of_mul hP
    have h₂ := right_ne_zero_of_mul hP
    simp only [fL, kron_of_ne hk, zero_mul, zero_div, sub_zero, PL]
    generalize hA : ((F.BL σ).z k n + F.a) ^ 2 - F.wL k ^ 2 / 4 = A at *
    generalize hB : ((F.BL σ).z k n - F.a) ^ 2 - F.wL k ^ 2 / 4 = B at *
    have hH : F.HL k (F.rL σ) ((F.BL σ).z k n) =
        (1 - F.rL σ) / 2 * B + (1 + F.rL σ) / 2 * A := by
      rw [← hA, ← hB]
      simp only [HL, q₀L]
      ring
    rw [hH]
    field_simp
    ring

/-! ### The column of the inverse -/

/-- The dense part of the column `(t, m)` of the inverse on the spin `τ`, `v = z_m` on the
lattice of `t`: `P(v)[d̂δ_lm + Â/(l + v)]` on `t`, `P(v)[B̂/(l - v) + Ĉ/(l + v)] + U/β²` on `-t`. -/
noncomputable def denseL (t τ : Bool) (m l : ℤ) : ℂ :=
  if τ = t then
    F.PL k ((F.BL t).z k m) * (F.dhL k t ((F.BL t).z k m) * kron l m +
      F.AhL k t ((F.BL t).z k m) / ((F.BL t).z k l + (F.BL t).z k m))
  else
    F.PL k ((F.BL t).z k m) * (F.BhL k t ((F.BL t).z k m) / ((F.BL (!t)).z k l - (F.BL t).z k m) +
      F.ChL k t ((F.BL t).z k m) / ((F.BL (!t)).z k l + (F.BL t).z k m)) + 2 * F.a / F.β ^ 2

/-- The mode of the root `u` on the spin `τ`, normalized with the column spin `t`:
`P(u)[f_{τ+}/(l - u) + f_{τ-}/(l + u)]`. -/
noncomputable def modeL (t τ : Bool) (u : ℂ) (l : ℤ) : ℂ :=
  F.PL k u * (F.fMp k t τ u / ((F.BL τ).z k l - u) + F.fMm k t τ u / ((F.BL τ).z k l + u))

/-! ### The images of the dense part -/

set_option maxHeartbeats 1000000 in
-- `field_simp` and `ring` on the value
/-- **The dense part, row `t`**: `P(v)δ_nm + D₀(v) + zD₁(v) + z²D₂(v)`. -/
theorem hasSum_dense_s (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {t : Bool}
    (hc : F.cL t ≠ 0) (h : F.ColNe k t) {n m : ℤ} (hE : F.E8L k ((F.BL t).z k m) ≠ 0) :
    HasSum (fun l ↦ F.PL k ((F.BL t).z k n) * F.chiL t t k n l * F.denseL k t t m l +
        F.PL k ((F.BL t).z k n) * F.chiL t (!t) k n l * F.denseL k t (!t) m l)
      (F.PL k ((F.BL t).z k m) * kron n m + F.Ds0 k t ((F.BL t).z k m) +
        (F.BL t).z k n * F.Ds1 k t ((F.BL t).z k m) +
        (F.BL t).z k n ^ 2 * F.Ds2 k t ((F.BL t).z k m)) := by
  have hβ := β_ne_zeroT (F := F)
  have hP := F.PL_BL_ne k t m
  have hm := h.m_ne
  have hρ := h.ρ_ne
  have hD := h.D_ne
  have hzv : (F.BL t).z k n + (F.BL t).z k m ≠ 0 := F.zL_add_self_ne k hc n m
  have h1 := hasSum_single (f := fun l ↦ F.PL k ((F.BL t).z k n) * F.chiL t t k n l *
    (F.PL k ((F.BL t).z k m) * F.dhL k t ((F.BL t).z k m) * kron l m)) m
    (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero])
  have h2 := (F.hasSum_self_pole k ha t n (x := -(F.BL t).z k m)
    (fun l h0 ↦ F.zL_add_self_ne k hc l m (by rw [h0]; ring))).mul_left (F.AhL k t ((F.BL t).z k m))
  have h3 := (F.hasSum_opp_pole_sub k ha hμ hh hc n m).mul_left (F.BhL k t ((F.BL t).z k m))
  have h4 := (F.hasSum_opp_pole_add k ha hμ hh hc n m).mul_left (F.ChL k t ((F.BL t).z k m))
  have h5 := F.hasSum_const_opp k ha hμ hh t n
  convert (h1.add h2).add ((h3.add h4).add h5) using 1
  · funext l
    simp only [denseL, ↓reduceIte, Bool.not_ne_self, F.PL_neg, sub_neg_eq_add]
    ring
  have e1 := F.dense_delta k t hE hm hρ
  have e2 := F.dense_pole k t (v := (F.BL t).z k m) hm hρ
  have hS : (F.BL t).S k (-(F.BL t).z k m) = F.sReflL t := F.S_self_neg_z k hc m
  simp only [kron_self, mul_one, F.PL_mul_chiL_self, F.PL_neg, sub_neg_eq_add, hS, F.RhatL_split,
    Ds0, Ds1, Ds2, WsL, dζ]
  rcases eq_or_ne n m with rfl | hnm
  · simp only [kron_self]
    simp only [HL] at e1 e2 ⊢
    have hz0 : (F.BL t).z k n ≠ 0 := fun h0 ↦ hzv (by rw [h0]; ring)
    linear_combination (norm := skip) -F.PL k ((F.BL t).z k n) * e1 -
      F.PL k ((F.BL t).z k n) / ((F.BL t).z k n + (F.BL t).z k n) * e2
    field_simp
    ring
  · simp only [kron_of_ne hnm]
    simp only [HL] at e2 ⊢
    linear_combination (norm := skip)
      -F.PL k ((F.BL t).z k m) / ((F.BL t).z k n + (F.BL t).z k m) * e2
    field_simp
    ring

set_option maxHeartbeats 1000000 in
-- `field_simp` and `ring` on the value
/-- **The dense part, row `-t`**: `D₀(v) + zD₁(v) + z²D₂(v)`. -/
theorem hasSum_dense_o (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {t : Bool}
    (hc : F.cL t ≠ 0) (h : F.ColNe k t) {n m : ℤ} :
    HasSum (fun l ↦ F.PL k ((F.BL (!t)).z k n) * F.chiL (!t) (!t) k n l * F.denseL k t (!t) m l +
        F.PL k ((F.BL (!t)).z k n) * F.chiL (!t) t k n l * F.denseL k t t m l)
      (F.Do0 k t ((F.BL t).z k m) + (F.BL (!t)).z k n * F.Do1 k t ((F.BL t).z k m) +
        (F.BL (!t)).z k n ^ 2 * F.Do2 k t ((F.BL t).z k m)) := by
  have hβ := β_ne_zeroT (F := F)
  have hP := F.PL_BL_ne k t m
  have hm := h.m_ne
  have hρ := h.ρ_ne
  have hD := h.D_ne
  have hzm : (F.BL (!t)).z k n - (F.BL t).z k m ≠ 0 := by
    simpa using F.zL_sub_opp_ne k hh (!t) n m
  have hzp : (F.BL (!t)).z k n + (F.BL t).z k m ≠ 0 := by
    simpa using F.zL_add_opp_ne k hμ (!t) n m
  have b1 := (F.hasSum_self_pole k ha (!t) n (x := (F.BL t).z k m)
    (fun l h0 ↦ F.zL_sub_opp_ne k hh t m l (by rw [h0]; ring))).mul_left (F.BhL k t ((F.BL t).z k m))
  have b2 := (F.hasSum_self_pole k ha (!t) n (x := -(F.BL t).z k m)
    (fun l h0 ↦ F.zL_add_opp_ne k hμ t m l (by rw [h0]; ring))).mul_left
      (F.ChL k t ((F.BL t).z k m))
  have b3 := F.hasSum_const_self k ha (!t) n
  have b4 := hasSum_single (f := fun l ↦ F.PL k ((F.BL (!t)).z k n) * F.chiL (!t) t k n l *
    (F.PL k ((F.BL t).z k m) * F.dhL k t ((F.BL t).z k m) * kron l m)) m
    (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero])
  have b5 := (F.hasSum_opp_pole k ha hμ hh (!t) n (x := -(F.BL t).z k m)
    (fun l h0 ↦ F.zL_add_self_ne k hc l m (by simp only [Bool.not_not] at h0; rw [h0]; ring))
    (fun h0 ↦ F.zL_sub_opp_ne k hh t m n (by rw [neg_inj.mp h0]; ring))
    (fun h0 ↦ F.zL_add_opp_ne k hμ t m n (by rw [← h0]; ring))).mul_left
      (F.AhL k t ((F.BL t).z k m))
  simp only [Bool.not_not] at b5
  convert ((b1.add b2).add b3).add (b4.add b5) using 1
  · funext l
    simp only [denseL, ↓reduceIte, Bool.not_ne_self, F.PL_neg, sub_neg_eq_add]
    ring
  have e3 := F.dense_pole_v k t (v := (F.BL t).z k m) hD
  have e4 := F.dense_pole_neg_v k t (v := (F.BL t).z k m) hD
  have hS : (F.BL t).S k (-(F.BL t).z k m) = F.sReflL t := F.S_self_neg_z k hc m
  have hm' : (F.mL (!t) : ℂ) = -F.mL t := by rw [F.mL_not t]; push_cast; ring
  have hc4 := F.PL_mul_chiL_opp k hμ hh (!t) n m
  simp only [Bool.not_not] at hc4
  simp only [kron_self, mul_one, hc4, F.PL_neg, sub_neg_eq_add, ← sub_eq_add_neg, hS,
    F.RhatL_split, F.S_opp_z k hh,
    F.S_opp_neg_z k hμ, αPL, αSL, Bool.not_not, hm', Do0, Do1, Do2, WoBL, WoCL, dζ, sReflL]
  simp only [HL] at e3 e4 ⊢
  linear_combination (norm := skip)
    -F.PL k ((F.BL t).z k m) / ((F.BL (!t)).z k n - (F.BL t).z k m) * e3 -
      F.PL k ((F.BL t).z k m) / ((F.BL (!t)).z k n + (F.BL t).z k m) * e4
  field_simp
  ring

/-! ### The images of the mode -/

set_option maxHeartbeats 1000000 in
-- `field_simp` and `ring` on the value
/-- **The mode of a root, row `t`**: `M₀(u) + zM₁(u) + z²M₂(u)`. -/
theorem hasSum_mode_s (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {t : Bool}
    (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u) (hE : F.E8L k u = 0)
    (hu : ∀ σ l, (F.BL σ).z k l ≠ u) (hu' : ∀ σ l, (F.BL σ).z k l ≠ -u) (n : ℤ) :
    HasSum (fun l ↦ F.PL k ((F.BL t).z k n) * F.chiL t t k n l * F.modeL k t t u l +
        F.PL k ((F.BL t).z k n) * F.chiL t (!t) k n l * F.modeL k t (!t) u l)
      (F.Ms0 k t u + (F.BL t).z k n * F.Ms1 k t u + (F.BL t).z k n ^ 2 * F.Ms2 k t u) := by
  have hzm : (F.BL t).z k n - u ≠ 0 := sub_ne_zero.mpr (hu t n)
  have hzp : (F.BL t).z k n + u ≠ 0 := fun h0 ↦ hu' t n (by linear_combination h0)
  have g1 := (F.hasSum_self_pole k ha t n (hu t)).mul_left (F.fMp k t t u)
  have g2 := (F.hasSum_self_pole k ha t n (hu' t)).mul_left (F.fMm k t t u)
  have g3 := (F.hasSum_opp_pole k ha hμ hh t n (hu (!t))
    (fun h0 ↦ hu' t n (by rw [h0]; ring)) (fun h0 ↦ hu t n h0.symm)).mul_left (F.fMp k t (!t) u)
  have g4 := (F.hasSum_opp_pole k ha hμ hh t n (hu' (!t))
    (fun h0 ↦ hu t n (by linear_combination h0)) (fun h0 ↦ hu' t n h0.symm)).mul_left
      (F.fMm k t (!t) u)
  convert (g1.add g2).add (g3.add g4) using 1
  · funext l
    simp only [modeL, F.PL_neg, sub_neg_eq_add]
    ring
  have r1 := F.res_self_pos k hr hE
  have r2 := F.res_self_neg k h hr
  simp only [F.PL_neg, sub_neg_eq_add, ← sub_eq_add_neg, F.RhatL_split, Ms0, Ms1, Ms2, MsA0, MsB,
    MsC0, MsD0, MsC2, MsD2, WmL, dζ]
  simp only [HL] at r1 r2 ⊢
  linear_combination (norm := skip)
    -1 / ((F.BL t).z k n - u) * r1 - 1 / ((F.BL t).z k n + u) * r2
  field_simp
  ring

set_option maxHeartbeats 1000000 in
-- `field_simp` and `ring` on the value
/-- **The mode of a root, row `-t`**: `M₀(u) + zM₁(u) + z²M₂(u)`. -/
theorem hasSum_mode_o (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {t : Bool}
    (h : F.ColNe k t) {u : ℂ} (hr : F.ModeRel k t u)
    (hu : ∀ σ l, (F.BL σ).z k l ≠ u) (hu' : ∀ σ l, (F.BL σ).z k l ≠ -u) (n : ℤ) :
    HasSum (fun l ↦ F.PL k ((F.BL (!t)).z k n) * F.chiL (!t) (!t) k n l * F.modeL k t (!t) u l +
        F.PL k ((F.BL (!t)).z k n) * F.chiL (!t) t k n l * F.modeL k t t u l)
      (F.Mo0 k t u + (F.BL (!t)).z k n * F.Mo1 k t u + (F.BL (!t)).z k n ^ 2 * F.Mo2 k t u) := by
  have hzm : (F.BL (!t)).z k n - u ≠ 0 := sub_ne_zero.mpr (hu (!t) n)
  have hzp : (F.BL (!t)).z k n + u ≠ 0 := fun h0 ↦ hu' (!t) n (by linear_combination h0)
  have g1 := (F.hasSum_self_pole k ha (!t) n (hu (!t))).mul_left (F.fMp k t (!t) u)
  have g2 := (F.hasSum_self_pole k ha (!t) n (hu' (!t))).mul_left (F.fMm k t (!t) u)
  have g3 := (F.hasSum_opp_pole k ha hμ hh (!t) n (hu (!(!t)))
    (fun h0 ↦ hu' (!t) n (by rw [h0]; ring)) (fun h0 ↦ hu (!t) n h0.symm)).mul_left
      (F.fMp k t t u)
  have g4 := (F.hasSum_opp_pole k ha hμ hh (!t) n (hu' (!(!t)))
    (fun h0 ↦ hu (!t) n (by linear_combination h0)) (fun h0 ↦ hu' (!t) n h0.symm)).mul_left
      (F.fMm k t t u)
  simp only [Bool.not_not] at g3 g4
  convert (g1.add g2).add (g3.add g4) using 1
  · funext l
    simp only [modeL, F.PL_neg, sub_neg_eq_add]
    ring
  have r1 := F.res_opp_pos k h hr
  have r2 := F.res_opp_neg k h hr
  simp only [F.PL_neg, sub_neg_eq_add, ← sub_eq_add_neg, F.RhatL_split, Mo0, Mo1, Mo2, MoA0, MoB,
    MoC0, MoD0, MoC2, MoD2, dζ]
  simp only [HL] at r1 r2 ⊢
  linear_combination (norm := skip)
    -1 / ((F.BL (!t)).z k n - u) * r1 - 1 / ((F.BL (!t)).z k n + u) * r2
  field_simp
  ring

end FParams

end HubbardAtom
