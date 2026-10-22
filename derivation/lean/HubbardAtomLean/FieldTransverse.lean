import HubbardAtomLean.FieldTIdentities
import HubbardAtomLean.DensityMagnetic

/-!
# The transverse vertex with doping and field

The transverse vertex, Eq. (field-T) of the note (App. F.6.2): the
transverse problem is the doped magnetic problem with the complex transfer `W = iω_k - 2h`, and the
proof of Theorem "density and magnetic" goes through with the data of `FieldInput.lean`. The algebra
of Steps 1, 3 and 4 is the free-variable algebra of `DensityMagnetic.lean` (`step1_off`,
`step1_diag`, `step3_identity`, `step4_identity`); the lattice sums are those of `FieldTSums.lean`
and the rational identities those of `FieldTIdentities.lean`.

* `FParams.TGeneric`: the assumptions: `U ≠ 0`, `μ ≠ 0`, `W ≠ 0`,
  `e(zₙ) ≠ 0` for all `n`, two roots `u₁`, `u₂` of `e` with `u₁² ≠ u₂²`, and `Kⱼ ≠ 0`; a root
  may lie at a zero of `P`.
* `FParams.invTUp`: the inverse; `TGeneric.chiTUp_isRightInverse`, `chiTUp_isLeftInverse`.
* `FParams.gammaTUp`, `TGeneric.gammaTUp_eq`: **the transverse vertex**,
  `β²(χ_T↑⁻¹ - b_T↑⁻¹) = Γ_T↑` of Eq. (field-T).
-/

open Complex Filter Topology

namespace HubbardAtom

/-- `step3_identity` times `P = P(u)`, which needs no `P ≠ 0`: the image of `P(u)φ_u` is
`-βK(u)[P(u)z + W(z; u²)/ρ]`. -/
theorem step3_identity_mul {β lam ρ a q₁ z u P W αu αmu Nu Nmu Ru Rmu cb : ℂ} (hβ : β ≠ 0)
    (hlam : lam ≠ 0) (hρ : ρ ≠ 0) (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0) (hαu : αu ≠ 0)
    (hrel : β ^ 2 * (u ^ 2 + 2 * a * ρ * u + q₁) * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) =
      lam ^ 2 * (αu * αmu))
    (hp : lam / β * (cb * Nu + ρ * Ru) = lam * ρ * (z + u + 2 * a * ρ) * P -
      (u ^ 2 + 2 * a * ρ * u + q₁) * Nmu - lam * ρ * z * P - lam * W)
    (hm : lam / β * (cb * Nmu + ρ * Rmu) = lam * ρ * (z - u + 2 * a * ρ) * P -
      ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) * Nu - lam * ρ * z * P - lam * W) :
    P * (-β * (z ^ 2 + 2 * a * ρ * z + q₁) * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) / (z - u) +
          lam * αu / (z + u)) +
        lam * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁)) * (αu / (z + u)) +
        lam * (lam * αu) * (αmu / (z + -u))) +
        β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) * (Nu * ((αu + cb) / ρ) + Ru) +
      lam * αu * (Nmu * ((αmu + cb) / ρ) + Rmu) =
    -β * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) + lam * αu) * (P * z + W / ρ) := by
  have hYp : cb * Nu + ρ * Ru = β / lam * (lam * ρ * (z + u + 2 * a * ρ) * P -
      (u ^ 2 + 2 * a * ρ * u + q₁) * Nmu - lam * ρ * z * P - lam * W) := by
    rw [← hp]
    field_simp
  have hYm : cb * Nmu + ρ * Rmu = β / lam * (lam * ρ * (z - u + 2 * a * ρ) * P -
      ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) * Nu - lam * ρ * z * P - lam * W) := by
    rw [← hm]
    field_simp
  have hαmu : αmu = β ^ 2 * (u ^ 2 + 2 * a * ρ * u + q₁) * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) /
      (lam ^ 2 * αu) := by
    rw [hrel]
    field_simp
  have e₁ : Nu * ((αu + cb) / ρ) + Ru = (αu * Nu + (cb * Nu + ρ * Ru)) / ρ := by
    field_simp
    ring
  have e₂ : Nmu * ((αmu + cb) / ρ) + Rmu = (αmu * Nmu + (cb * Nmu + ρ * Rmu)) / ρ := by
    field_simp
    ring
  have hzu'' : z + -u ≠ 0 := by rwa [← sub_eq_add_neg]
  rw [e₁, e₂, hYp, hYm, hαmu]
  field_simp
  ring

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The inverse -/

/-- `K(u) = βH(-u) + λα_k(u)`. -/
noncomputable def KT (u : ℂ) : ℂ := F.β * F.HT k (-u) + F.lamT * F.αT k u

/-- `Z(u) = uP(u)[βH(-u) - λα_k(u)]/K(u)`. -/
noncomputable def ZT (u : ℂ) : ℂ :=
  u * F.PT k u * (F.β * F.HT k (-u) - F.lamT * F.αT k u) / F.KT k u

/-- `F(z) = P(z)/e(z)`. -/
noncomputable def FT (z : ℂ) : ℂ := F.PT k z / F.eT k z

/-- Eq. (mode): `φ_u(t) = βH(-u)/(t - u) + λα_k(u)/(t + u)`. -/
noncomputable def modeT (u t : ℂ) : ℂ :=
  F.β * F.HT k (-u) / (t - u) + F.lamT * F.αT k u / (t + u)

/-- `D_v = -H(-v)P(v)/(βe(v))` of Step 1. -/
noncomputable def DvT (v : ℂ) : ℂ := -(F.HT k (-v) * F.PT k v / (F.β * F.eT k v))

/-- `E_v = -λρF(v)/β²` of Step 1. -/
noncomputable def EvT (v : ℂ) : ℂ := -(F.lamT * F.ρ * F.FT k v) / F.β ^ 2

/-- The coefficient `λρFⱼgⱼ(v)/(β²Kⱼ)` of `φ_{uⱼ}` in the column `y_v`. -/
noncomputable def cmodeT (u u' v : ℂ) : ℂ :=
  F.lamT * F.ρ * (F.PT k u / (u ^ 2 - u' ^ 2)) / (v ^ 2 - u ^ 2) / (F.β ^ 2 * F.KT k u)

/-- The inverse of `χ_T↑`: its column `m` is
`y_v = D_v δ + E_v/(t + v) + (-2a + λρ ∑ⱼ Fⱼgⱼ(v)φⱼ/Kⱼ)/β²`, `v = zₘ`. -/
noncomputable def invTUp (u₁ u₂ : ℂ) (n m : ℤ) : ℂ :=
  F.DvT k (F.zT k m) * kron n m + F.EvT k (F.zT k m) / (F.zT k n + F.zT k m) +
      2 * (-F.a) / F.β ^ 2 + F.cmodeT k u₁ u₂ (F.zT k m) * F.modeT k u₁ (F.zT k n) +
    F.cmodeT k u₂ u₁ (F.zT k m) * F.modeT k u₂ (F.zT k n)

/-- The row `P(zₙ)χ_T↑,nl` of the transverse input (`PT_mul_chiTUp`). -/
noncomputable def RrowT (n l : ℤ) : ℂ :=
  -F.β * F.HT k (F.zT k n) * kron n l + F.lamT * F.ρ / (F.zT k n + F.zT k l) +
    F.NT k (F.zT k n) (F.zT k l) / F.PT k (F.zT k l)

/-- The assumptions of the transverse vertex, Eq. (field-T), for the roots `u₁² = s₁`, `u₂² = s₂`
of `e`. -/
structure TGeneric (u₁ u₂ : ℂ) : Prop where
  a_ne : (F.a : ℂ) ≠ 0
  μ_ne : F.μ ≠ 0
  W_ne : F.WT k ≠ 0
  e_ne : ∀ n, F.eT k (F.zT k n) ≠ 0
  e_u₁ : F.eT k u₁ = 0
  e_u₂ : F.eT k u₂ = 0
  s_ne : u₁ ^ 2 ≠ u₂ ^ 2
  K_u₁ : F.KT k u₁ ≠ 0
  K_u₂ : F.KT k u₂ ≠ 0

/-! ### Consequences of the assumptions -/

theorem eT_neg (t : ℂ) : F.eT k (-t) = F.eT k t := by
  rw [eT, eT, neg_neg]
  ring

theorem PT_neg (t : ℂ) : F.PT k (-t) = F.PT k t := by
  rw [F.PT_eq_poly, F.PT_eq_poly]
  ring

theorem rho_ne_zeroT (hμ : F.μ ≠ 0) : (F.ρ : ℂ) ≠ 0 := by
  rw [F.ρ_eq_cw]
  push_cast
  exact mul_ne_zero (ofReal_ne_zero.mpr F.cw_pos.ne') ((F.lat F.μ).rho_ne_zero hμ)

theorem lamT_ne_zero (ha : (F.a : ℂ) ≠ 0) : F.lamT ≠ 0 := by
  rw [lamT]
  exact mul_ne_zero (by norm_num) (pow_ne_zero 2 ha)

theorem zT_add_zT_ne_zero (hμ : F.μ ≠ 0) (n m : ℤ) : F.zT k n + F.zT k m ≠ 0 :=
  (F.lat F.μ).z_add_z_ne_zero k hμ n m

theorem zT_ne_neg_zT (hμ : F.μ ≠ 0) (n m : ℤ) : F.zT k m ≠ -F.zT k n :=
  (F.lat F.μ).z_ne_neg_z k hμ n m

theorem zT_injective {n m : ℤ} (h : F.zT k n = F.zT k m) : n = m :=
  (F.lat F.μ).z_injective k h

/-- `e(t) = (t² - u₁²)(t² - u₂²)`. -/
theorem eT_eq_mul {u₁ u₂ : ℂ} (h₁ : F.eT k u₁ = 0) (h₂ : F.eT k u₂ = 0)
    (hs : u₁ ^ 2 ≠ u₂ ^ 2) (t : ℂ) : F.eT k t = (t ^ 2 - u₁ ^ 2) * (t ^ 2 - u₂ ^ 2) := by
  have hpoly : ∀ t, F.eT k t =
      t ^ 4 + (F.eT k 1 - 1 - F.eT k 0) * t ^ 2 + F.eT k 0 := by
    intro t
    simp only [eT, HT]
    ring
  have e₁ := hpoly u₁
  have e₂ := hpoly u₂
  rw [h₁] at e₁
  rw [h₂] at e₂
  have hs' : u₁ ^ 2 - u₂ ^ 2 ≠ 0 := sub_ne_zero.mpr hs
  have hC₁ : F.eT k 1 - 1 - F.eT k 0 = -(u₁ ^ 2 + u₂ ^ 2) := by
    apply mul_left_cancel₀ hs'
    linear_combination e₂ - e₁
  have hC₀ : F.eT k 0 = u₁ ^ 2 * u₂ ^ 2 := by linear_combination -e₁ - u₁ ^ 2 * hC₁
  rw [hpoly t, hC₁, hC₀]
  ring

theorem WphT_affine (z s : ℂ) :
    F.WphT k z s = F.WphT k z 0 + (F.WphT k z 1 - F.WphT k z 0) * s := by
  simp only [WphT, W₀T, W₂T, W₀Tpoly, W₂Tpoly]
  ring

theorem PT_sub_eT (t : ℂ) : F.PT k t - F.eT k t =
    ((F.PT k 1 - F.eT k 1) - (F.PT k 0 - F.eT k 0)) * t ^ 2 + (F.PT k 0 - F.eT k 0) := by
  simp only [F.PT_eq_poly, eT, HT]
  ring

variable {F k} {u₁ u₂ : ℂ}

theorem TGeneric.z_ne (h : F.TGeneric k u₁ u₂) {u : ℂ} (hu : F.eT k u = 0) (n : ℤ) :
    F.zT k n ≠ u := fun h0 ↦ h.e_ne n (by rw [h0, hu])

theorem TGeneric.z_ne_neg (h : F.TGeneric k u₁ u₂) {u : ℂ} (hu : F.eT k u = 0) (n : ℤ) :
    F.zT k n ≠ -u := fun h0 ↦ h.e_ne n (by rw [h0, F.eT_neg, hu])

theorem TGeneric.eT_eq (h : F.TGeneric k u₁ u₂) (t : ℂ) :
    F.eT k t = (t ^ 2 - u₁ ^ 2) * (t ^ 2 - u₂ ^ 2) :=
  F.eT_eq_mul k h.e_u₁ h.e_u₂ h.s_ne t

theorem TGeneric.z_sq_ne₁ (h : F.TGeneric k u₁ u₂) (n : ℤ) : F.zT k n ^ 2 - u₁ ^ 2 ≠ 0 :=
  fun h0 ↦ h.e_ne n (by rw [h.eT_eq, h0, zero_mul])

theorem TGeneric.z_sq_ne₂ (h : F.TGeneric k u₁ u₂) (n : ℤ) : F.zT k n ^ 2 - u₂ ^ 2 ≠ 0 :=
  fun h0 ↦ h.e_ne n (by rw [h.eT_eq, h0, mul_zero])

/-- At a root `u` of `e`: `β²H(u)H(-u) = λ²α(u)α(-u)`, both being `λ²β²p₀p₂`. -/
theorem TGeneric.hrel (h : F.TGeneric k u₁ u₂) {u : ℂ} (hu : F.eT k u = 0) :
    (F.β : ℂ) ^ 2 * F.HT k u * F.HT k (-u) = F.lamT ^ 2 * (F.αT k u * F.αT k (-u)) := by
  rw [F.αT_mul_αT_neg k (h.z_ne hu) (h.z_ne_neg hu)]
  rw [eT] at hu
  linear_combination (F.β : ℂ) ^ 2 * hu

theorem TGeneric.α_ne (h : F.TGeneric k u₁ u₂) {u : ℂ} (hu : F.eT k u = 0) :
    F.αT k u ≠ 0 := by
  intro h0
  have := F.αT_mul_αT_neg k (h.z_ne hu) (h.z_ne_neg hu)
  rw [h0, zero_mul] at this
  have hp : (F.β : ℂ) ^ 2 * F.p₀ * F.p₂ ≠ 0 := by
    have := (F.lat F.μ).β_ne_zero
    have := ofReal_ne_zero.mpr F.p₀_pos.ne'
    have := ofReal_ne_zero.mpr F.p₂_pos.ne'
    simp only [lat_β] at *
    positivity
  exact hp this.symm

theorem β_ne_zeroT : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'

/-! ### Steps 1 and 2 -/

/-- Step 1 (the Cauchy column). -/
theorem TGeneric.hasSum_step1 (h : F.TGeneric k u₁ u₂) (n m : ℤ) :
    HasSum (fun l ↦ F.RrowT k n l *
        (F.DvT k (F.zT k m) * kron l m + F.EvT k (F.zT k m) / (F.zT k l + F.zT k m)))
      (F.PT k (F.zT k m) * kron n m + (-F.N₂T k (F.zT k n) +
        F.lamT * F.ρ * F.zT k n * (F.FT k (F.zT k m) - 1) +
        F.lamT * F.WphT k (F.zT k n) (F.zT k m ^ 2) / F.eT k (F.zT k m)) / F.β) := by
  have hμ := h.μ_ne
  have ha := h.a_ne
  have hβ := F.β_ne_zeroT
  have hρ := F.rho_ne_zeroT hμ
  have hlam := F.lamT_ne_zero ha
  have hPm := F.PT_zT_ne_zero k m
  have hem := h.e_ne m
  have hzv := F.zT_add_zT_ne_zero k hμ n m
  have hS : (F.lat F.μ).S k (-F.zT k m) = F.cbarT / F.ρ := by
    rw [eq_div_iff hρ, mul_comm]
    exact F.rho_mul_S_neg_zT k hμ m
  have h₁ : HasSum (fun l ↦ F.RrowT k n l * F.DvT k (F.zT k m) * kron l m)
      (F.RrowT k n m * F.DvT k (F.zT k m)) := by
    convert hasSum_single (f := fun l ↦ F.RrowT k n l * F.DvT k (F.zT k m) * kron l m) m
      (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero]) using 1
    rw [kron_self, mul_one]
  have h₂ : HasSum (fun l ↦ -F.β * F.HT k (F.zT k n) * F.EvT k (F.zT k m) * kron n l /
      (F.zT k l + F.zT k m)) (-F.β * F.HT k (F.zT k n) * F.EvT k (F.zT k m) /
      (F.zT k n + F.zT k m)) := by
    convert hasSum_single (f := fun l ↦ -F.β * F.HT k (F.zT k n) * F.EvT k (F.zT k m) *
      kron n l / (F.zT k l + F.zT k m)) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_div]) using 1
    rw [kron_self, mul_one]
  have h₃ := (F.hasSum_squareT k hμ n m).mul_left (F.lamT * F.EvT k (F.zT k m) / F.ρ)
  have h₄ := (F.hasSum_rank_twoT k ha h.W_ne (F.zT k n) (x := -F.zT k m)
    (fun l ↦ F.zT_ne_neg_zT k hμ m l) (by rw [F.PT_neg]; exact hPm)).mul_left
      (F.EvT k (F.zT k m))
  convert ((h₁.add h₂).add h₃).add h₄ using 1
  · funext l
    have h₅ := F.zT_add_zT_ne_zero k hμ n l
    have h₆ := F.zT_add_zT_ne_zero k hμ l m
    have h₇ := F.PT_zT_ne_zero k l
    rw [RrowT, sub_neg_eq_add]
    field_simp
    ring
  · rw [F.PT_neg, hS]
    have h7 := F.identity_A7T k ha h.W_ne (F.zT k n) (F.zT k m)
    simp only [RrowT, DvT, EvT, FT, HT] at h7 ⊢
    have hR := h7
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self]
      have hoff := step1_off (a := F.a) (q₁ := F.q₁T k) hβ hlam hρ hPm hem
        hzv (N := F.NT k (F.zT k n) (F.zT k n)) (Nm := F.NT k (F.zT k n) (-F.zT k n))
        (R := F.RhatT k (F.zT k n) (-F.zT k n)) (cb := F.cbarT) (N2z := F.N₂T k (F.zT k n))
        (W := F.WphT k (F.zT k n) (F.zT k n ^ 2)) (by linear_combination -hR)
      have hdiag := step1_diag (Hv := F.HT k (F.zT k n)) (Hmv := F.HT k (-F.zT k n))
        (Pv := F.PT k (F.zT k n)) (lam := F.lamT) (p₀ := F.p₀) (p₂ := F.p₂) hβ hρ hem
        (by rw [eT])
      simp only [HT] at hdiag
      linear_combination -hoff - hdiag
    · rw [kron_of_ne hnm]
      have hoff := step1_off (a := F.a) (q₁ := F.q₁T k) hβ hlam hρ hPm hem
        hzv (N := F.NT k (F.zT k n) (F.zT k m)) (Nm := F.NT k (F.zT k n) (-F.zT k m))
        (R := F.RhatT k (F.zT k n) (-F.zT k m)) (cb := F.cbarT) (N2z := F.N₂T k (F.zT k n))
        (W := F.WphT k (F.zT k n) (F.zT k m ^ 2)) (by linear_combination -hR)
      linear_combination -hoff

/-- Step 2 (the constant): `∑ₗ P(zₙ)χ_T↑,nl = -βN₂(zₙ)/(2a)`, by `hasSum_constT` and (A6). -/
theorem TGeneric.hasSum_step2 (h : F.TGeneric k u₁ u₂) (n : ℤ) :
    HasSum (fun l ↦ F.RrowT k n l) (-(F.β * F.N₂T k (F.zT k n) / (2 * F.a))) := by
  have h₁ : HasSum (fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l)
      (-F.β * F.HT k (F.zT k n)) := by
    convert hasSum_single (f := fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero]) using 1
    rw [kron_self, mul_one]
  convert h₁.add (F.hasSum_constT k h.a_ne h.W_ne h.μ_ne n) using 1
  · funext l
    rw [RrowT]
    ring
  · rw [← F.identity_A6T k h.a_ne h.W_ne]
    ring

/-! ### Step 3 -/

/-- Step 3 (pole pairs), Eq. (mode-image), times `P(u)`, for a root `u` of `e`:
`∑ₗ P(zₙ)χ_T↑,nl P(u)φ_u(zₗ) = -βK(u)[P(u)zₙ + W(zₙ; u²)/ρ]`. Multiplying by `P(u)` removes
`P(u)` from the denominators, so that roots at a zero of `P` need no separate treatment. -/
theorem TGeneric.hasSum_step3 (h : F.TGeneric k u₁ u₂) {u : ℂ} (hu : F.eT k u = 0) (n : ℤ) :
    HasSum (fun l ↦ F.RrowT k n l * (F.PT k u * F.modeT k u (F.zT k l)))
      (-F.β * F.KT k u * (F.PT k u * F.zT k n + F.WphT k (F.zT k n) (u ^ 2) / F.ρ)) := by
  have hμ := h.μ_ne
  have ha := h.a_ne
  have hβ := F.β_ne_zeroT
  have hρ := F.rho_ne_zeroT hμ
  have hlam := F.lamT_ne_zero ha
  have hoff := h.z_ne hu
  have hoff' := h.z_ne_neg hu
  have hzu : F.zT k n - u ≠ 0 := sub_ne_zero.mpr (hoff n)
  have hzu' : F.zT k n + u ≠ 0 := fun h0 ↦ hoff' n (by linear_combination h0)
  have hun : u ≠ -F.zT k n := fun h0 ↦ hoff' n (by rw [h0, neg_neg])
  have hun' : -u ≠ -F.zT k n := fun h0 ↦ hoff n (neg_inj.mp h0).symm
  have h₁ : HasSum (fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l * F.modeT k u (F.zT k l))
      (-F.β * F.HT k (F.zT k n) * F.modeT k u (F.zT k n)) := by
    convert hasSum_single (f := fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l *
      F.modeT k u (F.zT k l)) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_mul]) using 1
    rw [kron_self, mul_one]
  have h₂ := (F.hasSum_cauchyT k hμ n hoff hun).mul_left (F.lamT * (F.β * F.HT k (-u)))
  have h₃ := (F.hasSum_cauchyT k hμ n hoff' hun').mul_left (F.lamT * (F.lamT * F.αT k u))
  have h₄ := (F.hasSum_rank_two_mulT k ha h.W_ne (F.zT k n) hoff).mul_left
    (F.β * F.HT k (-u))
  have h₅ := (F.hasSum_rank_two_mulT k ha h.W_ne (F.zT k n) hoff').mul_left
    (F.lamT * F.αT k u)
  rw [F.PT_neg] at h₅
  convert ((((h₁.add h₂).add h₃).mul_left (F.PT k u)).add h₄).add h₅ using 1
  · funext l
    have h₆ := sub_ne_zero.mpr (hoff l)
    have h₇ : F.zT k l + u ≠ 0 := fun h0 ↦ hoff' l (by linear_combination h0)
    have h₈ := F.zT_add_zT_ne_zero k hμ n l
    have h₉ := F.PT_zT_ne_zero k l
    simp only [RrowT, modeT, sub_neg_eq_add]
    field_simp
    ring
  · have hSu : (F.lat F.μ).S k u = (F.αT k u + F.cbarT) / F.ρ := by
      rw [αT]
      field_simp
      ring
    have hSmu : (F.lat F.μ).S k (-u) = (F.αT k (-u) + F.cbarT) / F.ρ := by
      rw [αT]
      field_simp
      ring
    rw [hSu, hSmu]
    have h7p := F.identity_A7T k ha h.W_ne (F.zT k n) (-u)
    have h7m := F.identity_A7T k ha h.W_ne (F.zT k n) u
    rw [neg_neg, F.PT_neg, F.eT_neg, hu, neg_sq] at h7p
    rw [hu] at h7m
    have hrel := h.hrel hu
    simp only [KT, modeT, HT] at h7p h7m hrel ⊢
    have key := step3_identity_mul (a := F.a) (q₁ := F.q₁T k)
      (z := F.zT k n) (Nu := F.NT k (F.zT k n) u) (Nmu := F.NT k (F.zT k n) (-u))
      (Ru := F.RhatT k (F.zT k n) u) (Rmu := F.RhatT k (F.zT k n) (-u)) (cb := F.cbarT)
      (W := F.WphT k (F.zT k n) (u ^ 2)) (P := F.PT k u) hβ hlam hρ hzu hzu' (h.α_ne hu) hrel
      (by linear_combination -h7p) (by linear_combination -h7m)
    linear_combination -key

/-! ### Step 4 and the right inverse -/

/-- The contribution of one root pair to the column. -/
theorem TGeneric.hasSum_mode (h : F.TGeneric k u₁ u₂) {u u' : ℂ} (hu : F.eT k u = 0)
    (hK : F.KT k u ≠ 0) (hs : u ^ 2 - u' ^ 2 ≠ 0) (n m : ℤ)
    (hv : F.zT k m ^ 2 - u ^ 2 ≠ 0) :
    HasSum (fun l ↦ F.cmodeT k u u' (F.zT k m) * (F.RrowT k n l * F.modeT k u (F.zT k l)))
      (-(F.lamT / F.β) * (F.ρ * F.zT k n * F.PT k u + F.WphT k (F.zT k n) (u ^ 2)) /
        ((u ^ 2 - u' ^ 2) * (F.zT k m ^ 2 - u ^ 2))) := by
  have hβ := F.β_ne_zeroT
  have hρ := F.rho_ne_zeroT h.μ_ne
  convert (h.hasSum_step3 hu n).mul_left
    (F.lamT * F.ρ / ((u ^ 2 - u' ^ 2) * (F.zT k m ^ 2 - u ^ 2) * (F.β ^ 2 * F.KT k u))) using 1
  · funext l
    rw [cmodeT]
    field_simp
  · field_simp

/-- **The transverse vertex**, the inverse: `χ_T↑ invTUp = 1`. -/
theorem TGeneric.chiTUp_isRightInverse (h : F.TGeneric k u₁ u₂) :
    IsRightInverse (F.chiTUp k) (F.invTUp k u₁ u₂) := by
  intro n m
  have hμ := h.μ_ne
  have ha := h.a_ne
  have hβ := F.β_ne_zeroT
  have hPn := F.PT_zT_ne_zero k n
  have hs := sub_ne_zero.mpr h.s_ne
  have hs' : u₂ ^ 2 - u₁ ^ 2 ≠ 0 := fun h0 ↦ hs (by linear_combination -h0)
  have hS1 := h.hasSum_step1 n m
  have hS2 := (h.hasSum_step2 n).mul_left (2 * (-(F.a : ℂ)) / (F.β : ℂ) ^ 2)
  have hS3 := h.hasSum_mode h.e_u₁ h.K_u₁ hs n m (h.z_sq_ne₁ m)
  have hS4 := h.hasSum_mode h.e_u₂ h.K_u₂ hs' n m (h.z_sq_ne₂ m)
  convert (((hS1.add hS2).add hS3).add hS4).div_const (F.PT k (F.zT k n)) using 1
  · funext l
    have hc : F.chiTUp k n l = F.RrowT k n l / F.PT k (F.zT k n) := by
      rw [eq_div_iff hPn, mul_comm, RrowT]
      exact F.PT_mul_chiTUp k hμ h.W_ne n l
    rw [hc, invTUp]
    field_simp
  · have hεa : -(F.a : ℂ) ≠ 0 := neg_ne_zero.mpr ha
    have h4 := step4_identity (β := F.β) (lam := F.lamT) (ρ := F.ρ) (z := F.zT k n)
      (N2z := F.N₂T k (F.zT k n)) (Pv := F.PT k (F.zT k m)) (ev := F.eT k (F.zT k m))
      (P₁ := F.PT k u₁) (P₂ := F.PT k u₂) (W₀ := F.WphT k (F.zT k n) 0)
      (W₁ := F.WphT k (F.zT k n) 1 - F.WphT k (F.zT k n) 0) hβ hεa hs (h.z_sq_ne₁ m)
      (h.z_sq_ne₂ m) (h.eT_eq _) (F.PT_sub_eT k _)
      (by linear_combination F.PT_sub_eT k u₁ + h.e_u₁)
      (by linear_combination F.PT_sub_eT k u₂ + h.e_u₂)
    rw [← F.WphT_affine k, ← F.WphT_affine k, ← F.WphT_affine k] at h4
    simp only [FT] at h4 ⊢
    rw [eq_div_iff hPn]
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self]
      linear_combination -h4
    · rw [kron_of_ne hnm]
      linear_combination -h4

/-! ### The closed form -/

variable (F k) in
/-- `DD(z, v) = (zF(z) - vF(v))/(z² - v²)` and `DD(z, z) = (F(z) + zF'(z))/(2z)`. -/
noncomputable def DDT (z v : ℂ) : ℂ :=
  if z = v then (F.FT k z + z * deriv (F.FT k) z) / (2 * z)
  else (z * F.FT k z - v * F.FT k v) / (z ^ 2 - v ^ 2)

variable (F k) in
/-- The closed form of Eq. (field-T): `βP(zₙ)[1/G₂(zₙ) - H(-zₙ)/e(zₙ)]δₙₘ - U - λρDD(zₙ, zₘ)
+ ∑ⱼ γⱼgⱼ(zₙ)gⱼ(zₘ)`, with `γⱼ = λρZⱼ/(sⱼ - s_ℓ)`, `λ = -4a²`; written with `2εa = -2a = -U`. -/
noncomputable def gammaTUp (u₁ u₂ : ℂ) (n m : ℤ) : ℂ :=
  F.β * F.PT k (F.zT k n) *
        (1 / F.G₂T k (F.zT k n) - F.HT k (-F.zT k n) / F.eT k (F.zT k n)) * kron n m +
      2 * (-F.a) - F.lamT * F.ρ * F.DDT k (F.zT k n) (F.zT k m) +
    F.lamT * F.ρ * F.ZT k u₁ / (u₁ ^ 2 - u₂ ^ 2) * (1 / (F.zT k n ^ 2 - u₁ ^ 2)) *
      (1 / (F.zT k m ^ 2 - u₁ ^ 2)) +
    F.lamT * F.ρ * F.ZT k u₂ / (u₂ ^ 2 - u₁ ^ 2) * (1 / (F.zT k n ^ 2 - u₂ ^ 2)) *
      (1 / (F.zT k m ^ 2 - u₂ ^ 2))

variable (F k) in
/-- The bubble: `b_T↑(νₙ) = -βG↑(νₙ)G↓(νₙ + ω_k) = -βG₂(zₙ)/P(zₙ)`. -/
theorem bTUp_eq (n : ℤ) : F.bTUp k n = -F.β * F.G₂T k (F.zT k n) / F.PT k (F.zT k n) := by
  have hP := F.PT_zT_ne_zero k n
  have hx : (F.lat (F.μ + F.h)).x n = F.zT k n - F.WT k / 2 := by
    simp only [zT, Params.z, Params.x, WT_eq, Params.w, lat_μ, lat_ν, lat_ω]
    push_cast
    ring
  have hx' : (F.lat (F.μ - F.h)).x (n + k) = F.zT k n + F.WT k / 2 := by
    rw [← (F.lat (F.μ - F.h)).x_add_w]
    simp only [zT, Params.z, Params.x, WT_eq, Params.w, lat_μ, lat_ν, lat_ω]
    push_cast
    ring
  have h₁ := (F.lat (F.μ + F.h)).x_sq_sub_ne_zero n
  have h₂ := (F.lat (F.μ - F.h)).x_sq_sub_ne_zero (n + k)
  simp only [lat_a] at h₁ h₂
  rw [hx] at h₁
  rw [hx'] at h₂
  have hPe : F.PT k (F.zT k n) = ((F.zT k n - F.WT k / 2) ^ 2 - F.a ^ 2) *
      ((F.zT k n + F.WT k / 2) ^ 2 - F.a ^ 2) := by
    simp only [PT, QpT, QmT]
    ring
  rw [bTUp, F.ν_add_ω, F.propagator_up, F.propagator_dn, hx, hx', hPe]
  simp only [Gc, G₂T, rUp, rDn]
  push_cast
  field_simp
  ring

variable (F k) in
/-- `G₂(zₙ) ≠ 0`: it is `(x↑ + ar↑)(x↓' + ar↓)` with nonreal `x`'s. -/
theorem G₂T_zT_ne_zero (n : ℤ) : F.G₂T k (F.zT k n) ≠ 0 := by
  have h₁ := (F.lat (F.μ + F.h)).x_add_ofReal_ne_zero n (F.a * F.rUp)
  have h₂ := (F.lat (F.μ - F.h)).x_add_ofReal_ne_zero (n + k) (F.a * F.rDn)
  rw [← (F.lat (F.μ - F.h)).x_add_w] at h₂
  have e : F.G₂T k (F.zT k n) = ((F.lat (F.μ + F.h)).x n + ((F.a * F.rUp : ℝ) : ℂ)) *
      ((F.lat (F.μ - F.h)).x n + (F.lat (F.μ - F.h)).w k + ((F.a * F.rDn : ℝ) : ℂ)) := by
    simp only [G₂T, zT, Params.z, Params.x, WT_eq, Params.w, lat_μ, lat_ν, lat_ω, rUp, rDn]
    push_cast
    ring
  rw [e]
  exact mul_ne_zero h₁ h₂

/-- The partial fractions of Step 4: `F(t) = 1 + F₁/(t² - s₁) + F₂/(t² - s₂)`. -/
theorem TGeneric.FT_eq_pf (h : F.TGeneric k u₁ u₂) (t : ℂ) (h₁ : t ^ 2 - u₁ ^ 2 ≠ 0)
    (h₂ : t ^ 2 - u₂ ^ 2 ≠ 0) :
    F.FT k t = 1 + (F.PT k u₁ / (u₁ ^ 2 - u₂ ^ 2) / (t ^ 2 - u₁ ^ 2) +
      F.PT k u₂ / (u₂ ^ 2 - u₁ ^ 2) / (t ^ 2 - u₂ ^ 2)) := by
  have hP₁ := F.PT_sub_eT k u₁
  have hP₂ := F.PT_sub_eT k u₂
  rw [h.e_u₁, sub_zero] at hP₁
  rw [h.e_u₂, sub_zero] at hP₂
  have hPt : F.PT k t = F.eT k t + (((F.PT k 1 - F.eT k 1) -
      (F.PT k 0 - F.eT k 0)) * t ^ 2 + (F.PT k 0 - F.eT k 0)) := by
    linear_combination F.PT_sub_eT k t
  have hs := sub_ne_zero.mpr h.s_ne
  have hs' : u₂ ^ 2 - u₁ ^ 2 ≠ 0 := fun h0 ↦ hs (by linear_combination -h0)
  rw [FT, hP₁, hP₂, hPt, h.eT_eq t]
  field_simp
  ring

/-- `F' = -2z(F₁/(z² - s₁)² + F₂/(z² - s₂)²)` at lattice points. -/
theorem TGeneric.hasDerivAt_FT (h : F.TGeneric k u₁ u₂) (n : ℤ) :
    HasDerivAt (F.FT k)
      (-2 * F.zT k n * (F.PT k u₁ / (u₁ ^ 2 - u₂ ^ 2) / (F.zT k n ^ 2 - u₁ ^ 2) ^ 2 +
        F.PT k u₂ / (u₂ ^ 2 - u₁ ^ 2) / (F.zT k n ^ 2 - u₂ ^ 2) ^ 2)) (F.zT k n) := by
  have h₁ := h.z_sq_ne₁ n
  have h₂ := h.z_sq_ne₂ n
  have hd : ∀ c P : ℂ, F.zT k n ^ 2 - c ≠ 0 →
      HasDerivAt (fun t ↦ P / (t ^ 2 - c)) (P * (-(2 * F.zT k n) / (F.zT k n ^ 2 - c) ^ 2))
        (F.zT k n) := by
    intro c P hc
    have := ((hasDerivAt_pow 2 (F.zT k n)).sub_const c).inv hc
    convert this.const_mul P using 1
    · funext t
      rw [div_eq_mul_inv]
      rfl
    · push_cast
      ring
  have hg := ((hd (u₁ ^ 2) (F.PT k u₁ / (u₁ ^ 2 - u₂ ^ 2)) h₁).add
    (hd (u₂ ^ 2) (F.PT k u₂ / (u₂ ^ 2 - u₁ ^ 2)) h₂)).const_add 1
  have hev : ∀ᶠ t in 𝓝 (F.zT k n), F.FT k t = 1 +
      (F.PT k u₁ / (u₁ ^ 2 - u₂ ^ 2) / (t ^ 2 - u₁ ^ 2) +
        F.PT k u₂ / (u₂ ^ 2 - u₁ ^ 2) / (t ^ 2 - u₂ ^ 2)) := by
    have hc : ContinuousAt (fun t : ℂ ↦ (t ^ 2 - u₁ ^ 2) * (t ^ 2 - u₂ ^ 2)) (F.zT k n) := by
      fun_prop
    filter_upwards [hc.eventually_ne (mul_ne_zero h₁ h₂)] with t ht
    exact h.FT_eq_pf t (left_ne_zero_of_mul ht) (right_ne_zero_of_mul ht)
  convert hg.congr_of_eventuallyEq hev using 1
  ring

/-- Eq. (defect-dd): `DD(z, v) = F(v)/(z + v) - z ∑ⱼ Fⱼgⱼ(z)gⱼ(v)` on the lattice. -/
theorem TGeneric.DDT_eq (h : F.TGeneric k u₁ u₂) (n m : ℤ) :
    F.DDT k (F.zT k n) (F.zT k m) = F.FT k (F.zT k m) / (F.zT k n + F.zT k m) -
      F.zT k n * (F.PT k u₁ / (u₁ ^ 2 - u₂ ^ 2) * (1 / (F.zT k n ^ 2 - u₁ ^ 2)) *
          (1 / (F.zT k m ^ 2 - u₁ ^ 2)) +
        F.PT k u₂ / (u₂ ^ 2 - u₁ ^ 2) * (1 / (F.zT k n ^ 2 - u₂ ^ 2)) *
          (1 / (F.zT k m ^ 2 - u₂ ^ 2))) := by
  have hzv := F.zT_add_zT_ne_zero k h.μ_ne n m
  have hs := sub_ne_zero.mpr h.s_ne
  have hs' : u₂ ^ 2 - u₁ ^ 2 ≠ 0 := fun h0 ↦ hs (by linear_combination -h0)
  have hn₁ := h.z_sq_ne₁ n
  have hn₂ := h.z_sq_ne₂ n
  have hm₁ := h.z_sq_ne₁ m
  have hm₂ := h.z_sq_ne₂ m
  by_cases hnm : n = m
  · subst hnm
    have hz0 : F.zT k n ≠ 0 := fun h0 ↦ hzv (by rw [h0]; ring)
    rw [DDT, ite_eq_left rfl, (h.hasDerivAt_FT n).deriv, h.FT_eq_pf _ hn₁ hn₂]
    field_simp
    ring
  · have hne : F.zT k n ≠ F.zT k m := fun h0 ↦ hnm (F.zT_injective k h0)
    have hsq : F.zT k n ^ 2 - F.zT k m ^ 2 ≠ 0 := by
      rw [show F.zT k n ^ 2 - F.zT k m ^ 2 = (F.zT k n - F.zT k m) * (F.zT k n + F.zT k m) by
        ring]
      exact mul_ne_zero (sub_ne_zero.mpr hne) hzv
    rw [DDT, ite_eq_right hne, h.FT_eq_pf _ hn₁ hn₂, h.FT_eq_pf _ hm₁ hm₂]
    field_simp
    ring

/-- `β²cⱼφⱼ(z)` splits into the `DD` part and the `γⱼ` part. -/
theorem cmodeT_mul_modeT {u u' z v : ℂ} (hK : F.KT k u ≠ 0) (hs : u ^ 2 - u' ^ 2 ≠ 0)
    (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0) (hv : v ^ 2 - u ^ 2 ≠ 0) :
    (F.β : ℂ) ^ 2 * (F.cmodeT k u u' v * F.modeT k u z) =
      F.lamT * F.ρ * (F.PT k u / (u ^ 2 - u' ^ 2)) * z * (1 / (z ^ 2 - u ^ 2)) *
          (1 / (v ^ 2 - u ^ 2)) +
        F.lamT * F.ρ * F.ZT k u / (u ^ 2 - u' ^ 2) * (1 / (z ^ 2 - u ^ 2)) *
          (1 / (v ^ 2 - u ^ 2)) := by
  have hβ := F.β_ne_zeroT
  have hz : z ^ 2 - u ^ 2 ≠ 0 := by
    rw [show z ^ 2 - u ^ 2 = (z - u) * (z + u) by ring]
    exact mul_ne_zero hzu hzu'
  rw [KT] at hK
  simp only [cmodeT, modeT, ZT, KT]
  field_simp
  ring

/-- **The transverse vertex**, the closed form: `β²(χ_T↑⁻¹ - b_T↑⁻¹) = Γ_T↑`. -/
theorem TGeneric.gammaTUp_eq (h : F.TGeneric k u₁ u₂) (n m : ℤ) :
    (F.β : ℂ) ^ 2 * (F.invTUp k u₁ u₂ n m - kron n m / F.bTUp k n) =
      F.gammaTUp k u₁ u₂ n m := by
  have hβ := F.β_ne_zeroT
  have hs := sub_ne_zero.mpr h.s_ne
  have hs' : u₂ ^ 2 - u₁ ^ 2 ≠ 0 := fun h0 ↦ hs (by linear_combination -h0)
  have hzv := F.zT_add_zT_ne_zero k h.μ_ne n m
  have hPn := F.PT_zT_ne_zero k n
  have hGn := F.G₂T_zT_ne_zero k n
  have hen := h.e_ne n
  have hem := h.e_ne m
  have hn₁ := h.z_sq_ne₁ n
  have hn₂ := h.z_sq_ne₂ n
  have hm₁ := h.z_sq_ne₁ m
  have hm₂ := h.z_sq_ne₂ m
  have e₁ := cmodeT_mul_modeT (F := F) (k := k) h.K_u₁ hs (sub_ne_zero.mpr (h.z_ne h.e_u₁ n))
    (fun h0 ↦ h.z_ne_neg h.e_u₁ n (by linear_combination h0)) hm₁
  have e₂ := cmodeT_mul_modeT (F := F) (k := k) h.K_u₂ hs' (sub_ne_zero.mpr (h.z_ne h.e_u₂ n))
    (fun h0 ↦ h.z_ne_neg h.e_u₂ n (by linear_combination h0)) hm₂
  have hsplit : (F.β : ℂ) ^ 2 * (F.invTUp k u₁ u₂ n m - kron n m / F.bTUp k n) =
      F.β ^ 2 * (F.DvT k (F.zT k m) * kron n m + F.EvT k (F.zT k m) / (F.zT k n + F.zT k m) -
        kron n m / F.bTUp k n) + 2 * (-F.a) +
        F.β ^ 2 * (F.cmodeT k u₁ u₂ (F.zT k m) * F.modeT k u₁ (F.zT k n)) +
        F.β ^ 2 * (F.cmodeT k u₂ u₁ (F.zT k m) * F.modeT k u₂ (F.zT k n)) := by
    rw [invTUp]
    field_simp
    ring
  rw [hsplit, e₁, e₂, gammaTUp, h.DDT_eq n m, F.bTUp_eq k n]
  simp only [DvT, EvT]
  by_cases hnm : n = m
  · subst hnm
    have hz0 : F.zT k n ≠ 0 := fun h0 ↦ hzv (by rw [h0]; ring)
    rw [kron_self]
    field_simp
    ring
  · rw [kron_of_ne hnm]
    field_simp
    ring

/-! ### Two-sidedness -/

variable (F k) in
theorem NT_sub_swap (z v : ℂ) : F.NT k z v - F.NT k v z =
    -(F.lamT * F.ρ) * (v - z) * (v ^ 2 + z ^ 2 + F.κT k) := by
  rw [F.NT_eq_coeffs, F.NT_eq_coeffs]
  simp only [cN₀T, cN₁T, cN₂T, cN₃T, lamT]
  ring

variable (F k) in
theorem PT_sub_PT (z v : ℂ) :
    F.PT k v - F.PT k z = (v - z) * (v + z) * (v ^ 2 + z ^ 2 + F.κT k) := by
  rw [F.PT_eq_poly, F.PT_eq_poly]
  ring

variable (F k) in
/-- `χ_T↑` is symmetric, for `μ ≠ 0` and `W ≠ 0` (via the transverse input, `PT_mul_chiTUp`). -/
theorem chiTUp_symm (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) (n m : ℤ) :
    F.chiTUp k n m = F.chiTUp k m n := by
  by_cases hnm : n = m
  · rw [hnm]
  have hPn := F.PT_zT_ne_zero k n
  have hPm := F.PT_zT_ne_zero k m
  have h₁ := F.PT_mul_chiTUp k hμ hW n m
  have h₂ := F.PT_mul_chiTUp k hμ hW m n
  rw [kron_of_ne hnm, mul_zero, zero_add] at h₁
  rw [kron_of_ne (Ne.symm hnm), mul_zero, zero_add] at h₂
  have hc₁ : F.chiTUp k n m = (F.lamT * F.ρ / (F.zT k n + F.zT k m) +
      F.NT k (F.zT k n) (F.zT k m) / F.PT k (F.zT k m)) / F.PT k (F.zT k n) := by
    rw [eq_div_iff hPn, mul_comm]
    exact h₁
  have hc₂ : F.chiTUp k m n = (F.lamT * F.ρ / (F.zT k m + F.zT k n) +
      F.NT k (F.zT k m) (F.zT k n) / F.PT k (F.zT k n)) / F.PT k (F.zT k m) := by
    rw [eq_div_iff hPm, mul_comm]
    exact h₂
  rw [hc₁, hc₂]
  exact symm_alg hPn hPm (F.zT_add_zT_ne_zero k hμ n m) (F.PT_sub_PT k _ _)
    (F.NT_sub_swap k _ _)

variable (F k) in
theorem DDT_symm (z v : ℂ) : F.DDT k z v = F.DDT k v z := by
  unfold DDT
  by_cases hzv : z = v
  · subst hzv
    rfl
  · rw [ite_eq_right hzv, ite_eq_right (Ne.symm hzv),
      show v * F.FT k v - z * F.FT k z = -(z * F.FT k z - v * F.FT k v) by ring,
      show v ^ 2 - z ^ 2 = -(z ^ 2 - v ^ 2) by ring, neg_div_neg_eq]

variable (F k) in
/-- `Γ_T↑` of Eq. (field-T) is symmetric. -/
theorem gammaTUp_symm (u₁ u₂ : ℂ) (n m : ℤ) :
    F.gammaTUp k u₁ u₂ n m = F.gammaTUp k u₁ u₂ m n := by
  by_cases hnm : n = m
  · rw [hnm]
  · rw [gammaTUp, gammaTUp, kron_of_ne hnm, kron_of_ne (Ne.symm hnm), F.DDT_symm k]
    ring

theorem TGeneric.invTUp_symm (h : F.TGeneric k u₁ u₂) (n m : ℤ) :
    F.invTUp k u₁ u₂ n m = F.invTUp k u₁ u₂ m n := by
  have hβ : (F.β : ℂ) ^ 2 ≠ 0 := pow_ne_zero 2 F.β_ne_zeroT
  have hk : kron n m / F.bTUp k n = kron m n / F.bTUp k m := by
    by_cases hnm : n = m
    · rw [hnm]
    · rw [kron_of_ne hnm, kron_of_ne (Ne.symm hnm), zero_div, zero_div]
  apply mul_left_cancel₀ hβ
  linear_combination h.gammaTUp_eq n m - h.gammaTUp_eq m n + F.gammaTUp_symm k u₁ u₂ n m +
    (F.β : ℂ) ^ 2 * hk

/-- **The transverse vertex**, two-sidedness: `invTUp χ_T↑ = 1`. -/
theorem TGeneric.chiTUp_isLeftInverse (h : F.TGeneric k u₁ u₂) :
    IsLeftInverse (F.chiTUp k) (F.invTUp k u₁ u₂) := by
  intro n m
  have := h.chiTUp_isRightInverse m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [F.chiTUp_symm k h.μ_ne h.W_ne l m, h.invTUp_symm n l, mul_comm]

end FParams

end HubbardAtom
