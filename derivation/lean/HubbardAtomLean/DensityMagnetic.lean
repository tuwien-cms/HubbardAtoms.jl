import HubbardAtomLean.PHIdentities

/-!
# Theorem "density and magnetic"

Theorem "density and magnetic" of the note in its generic case, under the assumptions
`Params.DMGeneric`: `ε = ±1`, `U ≠ 0`, `μ ≠ 0`, `e(zₙ) ≠ 0` for all `n`, two roots `u₁`, `u₂` of `e`
with `u₁² ≠ u₂²` (`s₁ ≠ s₂`), and for each root either `P(uⱼ) ≠ 0` and `Kⱼ ≠ 0`, or the atomic
alternative `P(uⱼ) = 0`, `W(z; sⱼ) = 0` (`Params.ModeOK`). The proof follows Steps 1–5 of the note.

* `step1_off`, `step1_diag`, `step3_identity`, `step4_identity`: the algebra of Steps 1, 3 and 4,
  over free variables.
* `Params.invR`: the inverse constructed in the proof; its column `m` is
  `y_v = D_v δ + E_v/(t + v) + (2εa + λρ ∑ⱼ Fⱼgⱼ(v)φⱼ/Kⱼ)/β²`, `v = zₘ`.
* `Params.DMGeneric.hasSum_step1`, `hasSum_step2`, `hasSum_step3`: Steps 1–3 as lattice sums
  of the row `P(zₙ)χ_r,n·` of Proposition "structure".
* `Params.DMGeneric.chiR_isRightInverse`, `Params.DMGeneric.chiR_isLeftInverse`: `invR` is a
  two-sided inverse of `χ_r`.
* `Params.gammaR`, `Params.DMGeneric.gammaR_eq`: **Theorem "density and magnetic"**,
  `β²(invR - B⁻¹) = Γ_r` of Eq. (Gamma-ph), with `DD(z, z) = (F(z) + zF'(z))/(2z)` on the diagonal.
* `Params.atomic_root`, `Params.dmGeneric_static_magnetic`: for `r = m`, `k = 0` the atomic root
  `u₁ = a` drops out, so the theorem covers the static magnetic channel.

The note's assumption `uⱼ ≠ 0` (Steps 1–4) is not needed here. Not covered: the confluent roots
`s₁ = s₂`, the lattice roots `e(zₙ) = 0`, common zeros of `P` and `e` other than the atomic pair,
and `μ = 0` (Corollary "half filling"); the note treats them by limits.
-/

open Complex Filter Topology

namespace HubbardAtom

/-! ### The algebra of Steps 1, 3 and 4 -/

/-- Step 1, off the diagonal: with `D = -H(-v)P(v)/(βe(v))` and `E = -λρF(v)/β²`, the terms of the
column that are not proportional to `δ_zv` are `(-N₂(z) + λρz(F(v) - 1) + λW(z; v²)/e(v))/β`,
given Eq. (source) at `v` (`hR`). -/
theorem step1_off {β lam ρ a q₁ z v Pv ev N Nm R cb N2z W : ℂ} (hβ : β ≠ 0) (hlam : lam ≠ 0)
    (hρ : ρ ≠ 0) (hP : Pv ≠ 0) (he : ev ≠ 0) (hzv : z + v ≠ 0)
    (hR : lam / β * (cb * Nm + ρ * R) = lam * ρ * (z - v + 2 * a * ρ) * Pv -
      ((-v) ^ 2 + 2 * a * ρ * (-v) + q₁) * N + ev * N2z - lam * ρ * z * (Pv - ev) - lam * W) :
    -(((-v) ^ 2 + 2 * a * ρ * (-v) + q₁) * Pv / (β * ev)) * (lam * ρ / (z + v) + N / Pv) +
        -β * (z ^ 2 + 2 * a * ρ * z + q₁) * (-(lam * ρ * (Pv / ev)) / β ^ 2) / (z + v) +
      -(lam * ρ * (Pv / ev)) / β ^ 2 * ((Nm * (cb / ρ) + R) / Pv) =
    (-N2z + lam * ρ * z * (Pv / ev - 1) + lam * W / ev) / β := by
  have hY : cb * Nm + ρ * R = β / lam * (lam * ρ * (z - v + 2 * a * ρ) * Pv -
      ((-v) ^ 2 + 2 * a * ρ * (-v) + q₁) * N + ev * N2z - lam * ρ * z * (Pv - ev) - lam * W) := by
    rw [← hR]
    field_simp
  have e₁ : (Nm * (cb / ρ) + R) / Pv = (cb * Nm + ρ * R) / (ρ * Pv) := by
    field_simp
  rw [e₁, hY]
  field_simp
  ring

/-- Step 1 on the diagonal: `-βH(v)D + λEh²/ρ = P(v)`, as `e = H(v)H(-v) - λ²h²/β²`. -/
theorem step1_diag {β lam ρ Hv Hmv Pv ev p₀ p₂ : ℂ} (hβ : β ≠ 0) (hρ : ρ ≠ 0) (he : ev ≠ 0)
    (hev : ev = Hv * Hmv - lam ^ 2 * p₀ * p₂) :
    -β * Hv * (-(Hmv * Pv / (β * ev))) + lam * (-(lam * ρ * (Pv / ev)) / β ^ 2) / ρ *
      (β ^ 2 * p₀ * p₂) = Pv := by
  have hPv : Pv = Pv * ev / ev := by field_simp
  rw [hPv, hev]
  field_simp
  ring

/-- Step 3, Eq. (mode-image): for a root `u` of `e` (`β²H(u)H(-u) = λ²α(u)α(-u)`), the image of
`φ = βH(-u)/(t - u) + λα(u)/(t + u)` under `P(z)χ_r` is `-βK[z + W(z; u²)/(ρP(u))]`; the poles at
`±u` cancel, and the rank-two sums are evaluated by Eq. (source) at `v = ∓u` (`hp`, `hm`). -/
theorem step3_identity {β lam ρ a q₁ z u P W αu αmu Nu Nmu Ru Rmu cb : ℂ} (hβ : β ≠ 0)
    (hlam : lam ≠ 0) (hρ : ρ ≠ 0) (hP : P ≠ 0) (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0)
    (hαu : αu ≠ 0)
    (hrel : β ^ 2 * (u ^ 2 + 2 * a * ρ * u + q₁) * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) =
      lam ^ 2 * (αu * αmu))
    (hp : lam / β * (cb * Nu + ρ * Ru) = lam * ρ * (z + u + 2 * a * ρ) * P -
      (u ^ 2 + 2 * a * ρ * u + q₁) * Nmu - lam * ρ * z * P - lam * W)
    (hm : lam / β * (cb * Nmu + ρ * Rmu) = lam * ρ * (z - u + 2 * a * ρ) * P -
      ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) * Nu - lam * ρ * z * P - lam * W) :
    -β * (z ^ 2 + 2 * a * ρ * z + q₁) * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) / (z - u) +
          lam * αu / (z + u)) +
        lam * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁)) * (αu / (z + u)) +
        lam * (lam * αu) * (αmu / (z + -u)) +
        β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) * ((Nu * ((αu + cb) / ρ) + Ru) / P) +
      lam * αu * ((Nmu * ((αmu + cb) / ρ) + Rmu) / P) =
    -β * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) + lam * αu) * (z + W / (ρ * P)) := by
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
  have e₁ : (Nu * ((αu + cb) / ρ) + Ru) / P = (αu * Nu + (cb * Nu + ρ * Ru)) / (ρ * P) := by
    field_simp
    ring
  have e₂ : (Nmu * ((αmu + cb) / ρ) + Rmu) / P = (αmu * Nmu + (cb * Nmu + ρ * Rmu)) / (ρ * P) := by
    field_simp
    ring
  have hzu'' : z + -u ≠ 0 := by rwa [← sub_eq_add_neg]
  rw [e₁, e₂, hYp, hYm, hαmu]
  field_simp
  ring

/-- Partial fractions of an affine function of `s` over `(s - s₁)(s - s₂)`. -/
theorem pf_affine {g₀ g₁ s₁ s₂ s : ℂ} (h12 : s₁ - s₂ ≠ 0) (h₁ : s - s₁ ≠ 0) (h₂ : s - s₂ ≠ 0) :
    (g₀ + g₁ * s) / ((s - s₁) * (s - s₂)) =
      (g₀ + g₁ * s₁) / ((s₁ - s₂) * (s - s₁)) + (g₀ + g₁ * s₂) / ((s₂ - s₁) * (s - s₂)) := by
  have h21 : s₂ - s₁ ≠ 0 := fun h ↦ h12 (by linear_combination -h)
  field_simp
  ring

/-- Step 4: the defect of Step 1 is removed by the constant (Step 2) and the contributions
`-(λ/β)(ρzP(uⱼ) + W(z; sⱼ))gⱼ(v)/(sⱼ - s_ℓ)` of the two root pairs (Step 3), by partial fractions in
`s = v²` of the affine `P - e` and `W(z; ·)`. -/
theorem step4_identity {β lam ρ εa z vv s₁ s₂ P₁ P₂ Pv ev N2z cA cB W₀ W₁ : ℂ} (hβ : β ≠ 0)
    (hεa : εa ≠ 0) (h12 : s₁ - s₂ ≠ 0) (hv₁ : vv - s₁ ≠ 0) (hv₂ : vv - s₂ ≠ 0)
    (hev : ev = (vv - s₁) * (vv - s₂)) (hPv : Pv - ev = cA * vv + cB) (hP₁' : P₁ = cA * s₁ + cB)
    (hP₂' : P₂ = cA * s₂ + cB) :
    (-N2z + lam * ρ * z * (Pv / ev - 1) + lam * (W₀ + W₁ * vv) / ev) / β +
        2 * εa / β ^ 2 * (β * N2z / (2 * εa)) +
        -(lam / β) * (ρ * z * P₁ + (W₀ + W₁ * s₁)) / ((s₁ - s₂) * (vv - s₁)) +
      -(lam / β) * (ρ * z * P₂ + (W₀ + W₁ * s₂)) / ((s₂ - s₁) * (vv - s₂)) = 0 := by
  have h21 : s₂ - s₁ ≠ 0 := fun h ↦ h12 (by linear_combination -h)
  have hPv' : Pv = cA * vv + cB + (vv - s₁) * (vv - s₂) := by linear_combination hPv + hev
  subst hev hPv' hP₁' hP₂'
  field_simp
  ring

/-- The symmetry of `χ_r` from Proposition "structure": `P(v) - P(z) = (v - z)(v + z)D` and
`N(z, v) - N(v, z) = -λρ(v - z)D` make `(λρ/(z + v) + N(z, v)/P(v))/P(z)` symmetric. -/
theorem symm_alg {z v c Pz Pv Nzv Nvz D : ℂ} (hPz : Pz ≠ 0) (hPv : Pv ≠ 0) (hzv : z + v ≠ 0)
    (hP : Pv - Pz = (v - z) * (v + z) * D) (hN : Nzv - Nvz = -c * (v - z) * D) :
    (c / (z + v) + Nzv / Pv) / Pz = (c / (v + z) + Nvz / Pz) / Pv := by
  have hNzv : Nzv = Nvz - c * (v - z) * D := by linear_combination hN
  rw [hNzv, add_comm v z, div_eq_div_iff hPz hPv]
  field_simp
  linear_combination c * hP

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-! ### The inverse -/

/-- `K(u) = βH(-u) + λα_k(u)`. -/
noncomputable def Kph (u : ℂ) : ℂ := A.β * A.Hph k ε (-u) + A.lam ε * A.α k u

/-- `Z(u) = uP(u)[βH(-u) - λα_k(u)]/K(u)`. -/
noncomputable def Zph (u : ℂ) : ℂ :=
  u * A.Pph k u * (A.β * A.Hph k ε (-u) - A.lam ε * A.α k u) / A.Kph k ε u

/-- `F(z) = P(z)/e(z)`. -/
noncomputable def Fph (z : ℂ) : ℂ := A.Pph k z / A.eph k ε z

/-- Eq. (mode): `φ_u(t) = βH(-u)/(t - u) + λα_k(u)/(t + u)`. -/
noncomputable def mode (u t : ℂ) : ℂ :=
  A.β * A.Hph k ε (-u) / (t - u) + A.lam ε * A.α k u / (t + u)

/-- `D_v = -H(-v)P(v)/(βe(v))` of Step 1. -/
noncomputable def Dv (v : ℂ) : ℂ := -(A.Hph k ε (-v) * A.Pph k v / (A.β * A.eph k ε v))

/-- `E_v = -λρF(v)/β²` of Step 1. -/
noncomputable def Ev (v : ℂ) : ℂ := -(A.lam ε * A.ρ * A.Fph k ε v) / A.β ^ 2

/-- The coefficient `λρFⱼgⱼ(v)/(β²Kⱼ)` of `φ_{uⱼ}` in the column `y_v` of Step 4, with
`Fⱼ = P(uⱼ)/(uⱼ² - u'²)` and `u'` the other root. -/
noncomputable def cmode (u u' v : ℂ) : ℂ :=
  A.lam ε * A.ρ * (A.Pph k u / (u ^ 2 - u' ^ 2)) / (v ^ 2 - u ^ 2) / (A.β ^ 2 * A.Kph k ε u)

/-- The inverse of `χ_r` constructed in Steps 1–4: its column `m` is
`y_v = D_v δ + E_v/(t + v) + (2εa + λρ ∑ⱼ Fⱼgⱼ(v)φⱼ/Kⱼ)/β²`, `v = zₘ`. -/
noncomputable def invR (u₁ u₂ : ℂ) (n m : ℤ) : ℂ :=
  A.Dv k ε (A.z k m) * kron n m + A.Ev k ε (A.z k m) / (A.z k n + A.z k m) +
      2 * ε * A.a / A.β ^ 2 + A.cmode k ε u₁ u₂ (A.z k m) * A.mode k ε u₁ (A.z k n) +
    A.cmode k ε u₂ u₁ (A.z k m) * A.mode k ε u₂ (A.z k n)

/-- The row `P(zₙ)χ_r,nl` of Proposition "structure". -/
noncomputable def Rrow (n l : ℤ) : ℂ :=
  -A.β * A.Hph k ε (A.z k n) * kron n l + A.lam ε * A.ρ / (A.z k n + A.z k l) +
    A.Nph k ε (A.z k n) (A.z k l) / A.Pph k (A.z k l)

/-- The condition on a root `u` of `e` in Theorem "density and magnetic": either `P(u) ≠ 0` and
`K(u) ≠ 0` (a pole pair of Step 3), or `P(u) = 0` and `W(z; u²) = 0` for all `z` (the atomic pair of
`r = m`, `k = 0`, which drops out). -/
def ModeOK (u : ℂ) : Prop :=
  (A.Pph k u ≠ 0 ∧ A.Kph k ε u ≠ 0) ∨ (A.Pph k u = 0 ∧ ∀ z, A.Wph k ε z (u ^ 2) = 0)

/-- The assumptions of the generic case of Theorem "density and magnetic", for the roots
`u₁² = s₁`, `u₂² = s₂` of `e`. -/
structure DMGeneric (u₁ u₂ : ℂ) : Prop where
  ε_eq : ε = 1 ∨ ε = -1
  a_ne : (A.a : ℂ) ≠ 0
  μ_ne : A.μ ≠ 0
  e_ne : ∀ n, A.eph k ε (A.z k n) ≠ 0
  e_u₁ : A.eph k ε u₁ = 0
  e_u₂ : A.eph k ε u₂ = 0
  s_ne : u₁ ^ 2 ≠ u₂ ^ 2
  mode₁ : A.ModeOK k ε u₁
  mode₂ : A.ModeOK k ε u₂

/-! ### Consequences of the assumptions -/

theorem eph_neg (t : ℂ) : A.eph k ε (-t) = A.eph k ε t := by
  rw [eph, eph, neg_neg]
  ring

theorem Pph_neg (t : ℂ) : A.Pph k (-t) = A.Pph k t := by
  rw [A.Pph_eq_poly, A.Pph_eq_poly]
  ring

theorem rho_ne_zero (hμ : A.μ ≠ 0) : (A.ρ : ℂ) ≠ 0 := by
  rw [ρ, A.p₂_eq_p₀_mul, ofReal_ne_zero, show A.p₀ * Real.exp (2 * A.β * A.μ) - A.p₀ =
    A.p₀ * (Real.exp (2 * A.β * A.μ) - 1) by ring]
  refine mul_ne_zero A.p₀_pos.ne' (sub_ne_zero.mpr ?_)
  rw [Ne, Real.exp_eq_one_iff]
  exact mul_ne_zero (mul_ne_zero two_ne_zero A.β_pos.ne') hμ

theorem lam_ne_zero (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) : A.lam ε ≠ 0 := by
  rw [lam]
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  exact mul_ne_zero (mul_ne_zero (by norm_num) hε0) (pow_ne_zero 2 ha)

/-- `e(t) = (t² - u₁²)(t² - u₂²)`: `e` is a monic quadratic in `t²` with the two roots `uⱼ²`. -/
theorem eph_eq_mul {u₁ u₂ : ℂ} (h₁ : A.eph k ε u₁ = 0) (h₂ : A.eph k ε u₂ = 0)
    (hs : u₁ ^ 2 ≠ u₂ ^ 2) (t : ℂ) : A.eph k ε t = (t ^ 2 - u₁ ^ 2) * (t ^ 2 - u₂ ^ 2) := by
  have hpoly : ∀ t, A.eph k ε t =
      t ^ 4 + (A.eph k ε 1 - 1 - A.eph k ε 0) * t ^ 2 + A.eph k ε 0 := by
    intro t
    simp only [eph, Hph]
    ring
  have e₁ := hpoly u₁
  have e₂ := hpoly u₂
  rw [h₁] at e₁
  rw [h₂] at e₂
  have hs' : u₁ ^ 2 - u₂ ^ 2 ≠ 0 := sub_ne_zero.mpr hs
  have hC₁ : A.eph k ε 1 - 1 - A.eph k ε 0 = -(u₁ ^ 2 + u₂ ^ 2) := by
    apply mul_left_cancel₀ hs'
    linear_combination e₂ - e₁
  have hC₀ : A.eph k ε 0 = u₁ ^ 2 * u₂ ^ 2 := by linear_combination -e₁ - u₁ ^ 2 * hC₁
  rw [hpoly t, hC₁, hC₀]
  ring

theorem Wph_affine (z s : ℂ) :
    A.Wph k ε z s = A.Wph k ε z 0 + (A.Wph k ε z 1 - A.Wph k ε z 0) * s := by
  simp only [Wph, W₀, W₂, ψ]
  ring

theorem Pph_sub_eph (t : ℂ) : A.Pph k t - A.eph k ε t =
    ((A.Pph k 1 - A.eph k ε 1) - (A.Pph k 0 - A.eph k ε 0)) * t ^ 2 +
      (A.Pph k 0 - A.eph k ε 0) := by
  simp only [A.Pph_eq_poly, eph, Hph]
  ring

variable {A k ε} {u₁ u₂ : ℂ}

theorem DMGeneric.z_ne (h : A.DMGeneric k ε u₁ u₂) {u : ℂ} (hu : A.eph k ε u = 0) (n : ℤ) :
    A.z k n ≠ u := fun h0 ↦ h.e_ne n (by rw [h0, hu])

theorem DMGeneric.z_ne_neg (h : A.DMGeneric k ε u₁ u₂) {u : ℂ} (hu : A.eph k ε u = 0) (n : ℤ) :
    A.z k n ≠ -u := fun h0 ↦ h.e_ne n (by rw [h0, A.eph_neg, hu])

theorem DMGeneric.eph_eq (h : A.DMGeneric k ε u₁ u₂) (t : ℂ) :
    A.eph k ε t = (t ^ 2 - u₁ ^ 2) * (t ^ 2 - u₂ ^ 2) :=
  A.eph_eq_mul k ε h.e_u₁ h.e_u₂ h.s_ne t

theorem DMGeneric.z_sq_ne₁ (h : A.DMGeneric k ε u₁ u₂) (n : ℤ) : A.z k n ^ 2 - u₁ ^ 2 ≠ 0 :=
  fun h0 ↦ h.e_ne n (by rw [h.eph_eq, h0, zero_mul])

theorem DMGeneric.z_sq_ne₂ (h : A.DMGeneric k ε u₁ u₂) (n : ℤ) : A.z k n ^ 2 - u₂ ^ 2 ≠ 0 :=
  fun h0 ↦ h.e_ne n (by rw [h.eph_eq, h0, mul_zero])

/-- At a root `u` of `e`: `β²H(u)H(-u) = λ²α(u)α(-u)`, both being `λ²β²p₀p₂`. -/
theorem DMGeneric.hrel (h : A.DMGeneric k ε u₁ u₂) {u : ℂ} (hu : A.eph k ε u = 0) :
    (A.β : ℂ) ^ 2 * A.Hph k ε u * A.Hph k ε (-u) = A.lam ε ^ 2 * (A.α k u * A.α k (-u)) := by
  rw [A.α_mul_α_neg k (h.z_ne hu) (h.z_ne_neg hu)]
  rw [eph] at hu
  linear_combination (A.β : ℂ) ^ 2 * hu

theorem DMGeneric.α_ne (h : A.DMGeneric k ε u₁ u₂) {u : ℂ} (hu : A.eph k ε u = 0) :
    A.α k u ≠ 0 := by
  intro h0
  have := A.α_mul_α_neg k (h.z_ne hu) (h.z_ne_neg hu)
  rw [h0, zero_mul] at this
  have hp : (A.β : ℂ) ^ 2 * A.p₀ * A.p₂ ≠ 0 := by
    have := A.β_ne_zero
    have := ofReal_ne_zero.mpr A.p₀_pos.ne'
    have := ofReal_ne_zero.mpr A.p₂_pos.ne'
    positivity
  exact hp this.symm

/-! ### Steps 1 and 2 -/

variable (A k ε) in
/-- Step 1 (the Cauchy column): `∑ₗ P(zₙ)χ_r,nl (D_v δ_lm + E_v/(zₗ + v)) =
P(v)δₙₘ + (-N₂(z) + λρz(F(v) - 1) + λW(z; v²)/e(v))/β` with `z = zₙ`, `v = zₘ`. -/
theorem hasSum_step1_of (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) {n m : ℤ}
    (hem : A.eph k ε (A.z k m) ≠ 0) :
    HasSum (fun l ↦ A.Rrow k ε n l *
        (A.Dv k ε (A.z k m) * kron l m + A.Ev k ε (A.z k m) / (A.z k l + A.z k m)))
      (A.Pph k (A.z k m) * kron n m + (-A.N₂ k ε (A.z k n) +
        A.lam ε * A.ρ * A.z k n * (A.Fph k ε (A.z k m) - 1) +
        A.lam ε * A.Wph k ε (A.z k n) (A.z k m ^ 2) / A.eph k ε (A.z k m)) / A.β) := by
  have hβ := A.β_ne_zero
  have hρ := A.rho_ne_zero hμ
  have hlam := A.lam_ne_zero ε hε ha
  have hPm := A.Pph_z_ne_zero k m
  have hzv := A.z_add_z_ne_zero k hμ n m
  have hS : A.S k (-A.z k m) = A.cbar / A.ρ := by
    rw [eq_div_iff hρ, mul_comm]
    exact A.rho_mul_S_neg_z k hμ m
  have h₁ : HasSum (fun l ↦ A.Rrow k ε n l * A.Dv k ε (A.z k m) * kron l m)
      (A.Rrow k ε n m * A.Dv k ε (A.z k m)) := by
    convert hasSum_single (f := fun l ↦ A.Rrow k ε n l * A.Dv k ε (A.z k m) * kron l m) m
      (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero]) using 1
    rw [kron_self, mul_one]
  have h₂ : HasSum (fun l ↦ -A.β * A.Hph k ε (A.z k n) * A.Ev k ε (A.z k m) * kron n l /
      (A.z k l + A.z k m)) (-A.β * A.Hph k ε (A.z k n) * A.Ev k ε (A.z k m) /
      (A.z k n + A.z k m)) := by
    convert hasSum_single (f := fun l ↦ -A.β * A.Hph k ε (A.z k n) * A.Ev k ε (A.z k m) *
      kron n l / (A.z k l + A.z k m)) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_div]) using 1
    rw [kron_self, mul_one]
  have h₃ := (A.hasSum_square k hμ n m).mul_left (A.lam ε * A.Ev k ε (A.z k m) / A.ρ)
  have h₄ := (A.hasSum_rank_two k ε ha (A.z k n) (x := -A.z k m)
    (fun l ↦ A.z_ne_neg_z k hμ m l) (by rw [A.Pph_neg]; exact hPm)).mul_left (A.Ev k ε (A.z k m))
  convert ((h₁.add h₂).add h₃).add h₄ using 1
  · funext l
    have h₅ := A.z_add_z_ne_zero k hμ n l
    have h₆ := A.z_add_z_ne_zero k hμ l m
    have h₇ := A.Pph_z_ne_zero k l
    rw [Rrow, sub_neg_eq_add]
    field_simp
    ring
  · rw [A.Pph_neg, hS]
    have h7 := A.identity_A7 k ε hε ha (A.z k n) (A.z k m)
    simp only [Rrow, Dv, Ev, Fph, Hph] at h7 ⊢
    have hR := h7
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self]
      have hoff := step1_off (a := A.a) (q₁ := A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁) hβ hlam hρ hPm hem
        hzv (N := A.Nph k ε (A.z k n) (A.z k n)) (Nm := A.Nph k ε (A.z k n) (-A.z k n))
        (R := A.Rhat k ε (A.z k n) (-A.z k n)) (cb := A.cbar) (N2z := A.N₂ k ε (A.z k n))
        (W := A.Wph k ε (A.z k n) (A.z k n ^ 2)) (by linear_combination -hR)
      have hdiag := step1_diag (Hv := A.Hph k ε (A.z k n)) (Hmv := A.Hph k ε (-A.z k n))
        (Pv := A.Pph k (A.z k n)) (lam := A.lam ε) (p₀ := A.p₀) (p₂ := A.p₂) hβ hρ hem
        (by rw [eph])
      simp only [Hph] at hdiag
      linear_combination -hoff - hdiag
    · rw [kron_of_ne hnm]
      have hoff := step1_off (a := A.a) (q₁ := A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁) hβ hlam hρ hPm hem
        hzv (N := A.Nph k ε (A.z k n) (A.z k m)) (Nm := A.Nph k ε (A.z k n) (-A.z k m))
        (R := A.Rhat k ε (A.z k n) (-A.z k m)) (cb := A.cbar) (N2z := A.N₂ k ε (A.z k n))
        (W := A.Wph k ε (A.z k n) (A.z k m ^ 2)) (by linear_combination -hR)
      linear_combination -hoff

theorem DMGeneric.hasSum_step1 (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l *
        (A.Dv k ε (A.z k m) * kron l m + A.Ev k ε (A.z k m) / (A.z k l + A.z k m)))
      (A.Pph k (A.z k m) * kron n m + (-A.N₂ k ε (A.z k n) +
        A.lam ε * A.ρ * A.z k n * (A.Fph k ε (A.z k m) - 1) +
        A.lam ε * A.Wph k ε (A.z k n) (A.z k m ^ 2) / A.eph k ε (A.z k m)) / A.β) :=
  A.hasSum_step1_of k ε h.ε_eq h.a_ne h.μ_ne (h.e_ne m)

variable (A k ε) in
/-- Step 2 (the constant): `∑ₗ P(zₙ)χ_r,nl = βN₂(zₙ)/(2εa)`, by `hasSum_const` and (A6). -/
theorem hasSum_step2_of (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l) (A.β * A.N₂ k ε (A.z k n) / (2 * ε * A.a)) := by
  have h₁ : HasSum (fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l)
      (-A.β * A.Hph k ε (A.z k n)) := by
    convert hasSum_single (f := fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero]) using 1
    rw [kron_self, mul_one]
  convert h₁.add (A.hasSum_const k ε ha hμ n) using 1
  · funext l
    rw [Rrow]
    ring
  · rw [← A.identity_A6 k ε hε ha]
    ring

theorem DMGeneric.hasSum_step2 (h : A.DMGeneric k ε u₁ u₂) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l) (A.β * A.N₂ k ε (A.z k n) / (2 * ε * A.a)) :=
  A.hasSum_step2_of k ε h.ε_eq h.a_ne h.μ_ne n

/-! ### Step 3 -/

/-- Step 3 (pole pairs), Eq. (mode-image): for a root `u` of `e` with `P(u) ≠ 0`,
`∑ₗ P(zₙ)χ_r,nl φ_u(zₗ) = -βK(u)[zₙ + W(zₙ; u²)/(ρP(u))]`. -/
theorem DMGeneric.hasSum_step3 (h : A.DMGeneric k ε u₁ u₂) {u : ℂ} (hu : A.eph k ε u = 0)
    (hPu : A.Pph k u ≠ 0) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l * A.mode k ε u (A.z k l))
      (-A.β * A.Kph k ε u * (A.z k n + A.Wph k ε (A.z k n) (u ^ 2) / (A.ρ * A.Pph k u))) := by
  have hμ := h.μ_ne
  have ha := h.a_ne
  have hβ := A.β_ne_zero
  have hρ := A.rho_ne_zero hμ
  have hlam := A.lam_ne_zero ε h.ε_eq ha
  have hoff := h.z_ne hu
  have hoff' := h.z_ne_neg hu
  have hzu : A.z k n - u ≠ 0 := sub_ne_zero.mpr (hoff n)
  have hzu' : A.z k n + u ≠ 0 := fun h0 ↦ hoff' n (by linear_combination h0)
  have hun : u ≠ -A.z k n := fun h0 ↦ hoff' n (by rw [h0, neg_neg])
  have hun' : -u ≠ -A.z k n := fun h0 ↦ hoff n (neg_inj.mp h0).symm
  have hPmu : A.Pph k (-u) ≠ 0 := by rwa [A.Pph_neg]
  have h₁ : HasSum (fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l * A.mode k ε u (A.z k l))
      (-A.β * A.Hph k ε (A.z k n) * A.mode k ε u (A.z k n)) := by
    convert hasSum_single (f := fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l *
      A.mode k ε u (A.z k l)) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_mul]) using 1
    rw [kron_self, mul_one]
  have h₂ := (A.hasSum_cauchy k hμ n hoff hun).mul_left (A.lam ε * (A.β * A.Hph k ε (-u)))
  have h₃ := (A.hasSum_cauchy k hμ n hoff' hun').mul_left (A.lam ε * (A.lam ε * A.α k u))
  have h₄ := (A.hasSum_rank_two k ε ha (A.z k n) hoff hPu).mul_left (A.β * A.Hph k ε (-u))
  have h₅ := (A.hasSum_rank_two k ε ha (A.z k n) hoff' hPmu).mul_left (A.lam ε * A.α k u)
  convert (((h₁.add h₂).add h₃).add h₄).add h₅ using 1
  · funext l
    have h₆ := sub_ne_zero.mpr (hoff l)
    have h₇ : A.z k l + u ≠ 0 := fun h0 ↦ hoff' l (by linear_combination h0)
    have h₈ := A.z_add_z_ne_zero k hμ n l
    have h₉ := A.Pph_z_ne_zero k l
    simp only [Rrow, mode, sub_neg_eq_add]
    field_simp
    ring
  · have hSu : A.S k u = (A.α k u + A.cbar) / A.ρ := by
      rw [α]
      field_simp
      ring
    have hSmu : A.S k (-u) = (A.α k (-u) + A.cbar) / A.ρ := by
      rw [α]
      field_simp
      ring
    rw [hSu, hSmu, A.Pph_neg]
    have h7p := A.identity_A7 k ε h.ε_eq ha (A.z k n) (-u)
    have h7m := A.identity_A7 k ε h.ε_eq ha (A.z k n) u
    rw [neg_neg, A.Pph_neg, A.eph_neg, hu, neg_sq] at h7p
    rw [hu] at h7m
    have hrel := h.hrel hu
    simp only [Kph, mode, Hph] at h7p h7m hrel ⊢
    have key := step3_identity (a := A.a) (q₁ := A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁)
      (z := A.z k n) (Nu := A.Nph k ε (A.z k n) u) (Nmu := A.Nph k ε (A.z k n) (-u))
      (Ru := A.Rhat k ε (A.z k n) u) (Rmu := A.Rhat k ε (A.z k n) (-u)) (cb := A.cbar)
      (W := A.Wph k ε (A.z k n) (u ^ 2)) hβ hlam hρ hPu hzu hzu' (h.α_ne hu) hrel
      (by linear_combination -h7p) (by linear_combination -h7m)
    linear_combination -key

/-! ### Step 4 and the right inverse -/

/-- The contribution of one root pair to the column: `cⱼ ∑ₗ P(zₙ)χ_r,nl φⱼ(zₗ) =
-(λ/β)(ρzₙP(uⱼ) + W(zₙ; sⱼ))/((sⱼ - s_ℓ)(v² - sⱼ))`, by Step 3; for the atomic pair both sides
vanish. -/
theorem DMGeneric.hasSum_mode (h : A.DMGeneric k ε u₁ u₂) {u u' : ℂ} (hu : A.eph k ε u = 0)
    (hmode : A.ModeOK k ε u) (hs : u ^ 2 - u' ^ 2 ≠ 0) (n m : ℤ)
    (hv : A.z k m ^ 2 - u ^ 2 ≠ 0) :
    HasSum (fun l ↦ A.cmode k ε u u' (A.z k m) * (A.Rrow k ε n l * A.mode k ε u (A.z k l)))
      (-(A.lam ε / A.β) * (A.ρ * A.z k n * A.Pph k u + A.Wph k ε (A.z k n) (u ^ 2)) /
        ((u ^ 2 - u' ^ 2) * (A.z k m ^ 2 - u ^ 2))) := by
  rcases hmode with ⟨hP, hK⟩ | ⟨hP, hW⟩
  · have hβ := A.β_ne_zero
    have hρ := A.rho_ne_zero h.μ_ne
    convert (h.hasSum_step3 hu hP n).mul_left (A.cmode k ε u u' (A.z k m)) using 1
    rw [cmode]
    field_simp
  · have hc : A.cmode k ε u u' (A.z k m) = 0 := by
      rw [cmode, hP, zero_div, mul_zero, zero_div, zero_div]
    simp only [hc, zero_mul, hP, mul_zero, hW, add_zero, zero_div, mul_zero]
    exact hasSum_zero

/-- **Theorem "density and magnetic"**, the inverse: `χ_r invR = 1`. The column `m` of `invR` is
`y_v` of the proof; by Proposition "structure", `P(zₙ)(χ_r y_v)ₙ` is the sum of Steps 1–3, which
add up to `P(v)δₙₘ` (Step 4). -/
theorem DMGeneric.chiR_isRightInverse (h : A.DMGeneric k ε u₁ u₂) :
    IsRightInverse (A.chiR k ε) (A.invR k ε u₁ u₂) := by
  intro n m
  have hμ := h.μ_ne
  have ha := h.a_ne
  have hβ := A.β_ne_zero
  have hPn := A.Pph_z_ne_zero k n
  have hs := sub_ne_zero.mpr h.s_ne
  have hs' : u₂ ^ 2 - u₁ ^ 2 ≠ 0 := fun h0 ↦ hs (by linear_combination -h0)
  have hS1 := h.hasSum_step1 n m
  have hS2 := (h.hasSum_step2 n).mul_left (2 * ε * A.a / A.β ^ 2)
  have hS3 := h.hasSum_mode h.e_u₁ h.mode₁ hs n m (h.z_sq_ne₁ m)
  have hS4 := h.hasSum_mode h.e_u₂ h.mode₂ hs' n m (h.z_sq_ne₂ m)
  convert (((hS1.add hS2).add hS3).add hS4).div_const (A.Pph k (A.z k n)) using 1
  · funext l
    have hc : A.chiR k ε n l = A.Rrow k ε n l / A.Pph k (A.z k n) := by
      rw [eq_div_iff hPn, mul_comm, Rrow]
      exact A.Pph_mul_chiR k ε hμ n l
    rw [hc, invR]
    field_simp
  · have hε0 : ε * (A.a : ℂ) ≠ 0 := by
      have : ε ≠ 0 := by rcases h.ε_eq with rfl | rfl <;> norm_num
      exact mul_ne_zero this ha
    have h4 := step4_identity (β := A.β) (lam := A.lam ε) (ρ := A.ρ) (z := A.z k n)
      (N2z := A.N₂ k ε (A.z k n)) (Pv := A.Pph k (A.z k m)) (ev := A.eph k ε (A.z k m))
      (P₁ := A.Pph k u₁) (P₂ := A.Pph k u₂) (W₀ := A.Wph k ε (A.z k n) 0)
      (W₁ := A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0) hβ hε0 hs (h.z_sq_ne₁ m)
      (h.z_sq_ne₂ m) (h.eph_eq _) (A.Pph_sub_eph k ε _)
      (by linear_combination A.Pph_sub_eph k ε u₁ + h.e_u₁)
      (by linear_combination A.Pph_sub_eph k ε u₂ + h.e_u₂)
    rw [← A.Wph_affine k ε, ← A.Wph_affine k ε, ← A.Wph_affine k ε] at h4
    simp only [Fph] at h4 ⊢
    rw [eq_div_iff hPn]
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self]
      linear_combination -h4
    · rw [kron_of_ne hnm]
      linear_combination -h4

/-! ### The closed form -/

variable (A k ε) in
/-- `DD(z, v) = (zF(z) - vF(v))/(z² - v²)` and `DD(z, z) = (F(z) + zF'(z))/(2z)`. -/
noncomputable def DD (z v : ℂ) : ℂ :=
  if z = v then (A.Fph k ε z + z * deriv (A.Fph k ε) z) / (2 * z)
  else (z * A.Fph k ε z - v * A.Fph k ε v) / (z ^ 2 - v ^ 2)

variable (A k ε) in
/-- The closed form of Eq. (Gamma-ph), for the roots `u₁`, `u₂`:
`βP(zₙ)[1/G₂(zₙ) - H(-zₙ)/e(zₙ)]δₙₘ + 2εa - λρDD(zₙ, zₘ) + ∑ⱼ γⱼ gⱼ(zₙ)gⱼ(zₘ)` with
`γⱼ = λρZⱼ/(sⱼ - s_ℓ)` and `gⱼ(z) = 1/(z² - sⱼ)`. -/
noncomputable def gammaR (u₁ u₂ : ℂ) (n m : ℤ) : ℂ :=
  A.β * A.Pph k (A.z k n) *
        (1 / A.G₂ k (A.z k n) - A.Hph k ε (-A.z k n) / A.eph k ε (A.z k n)) * kron n m +
      2 * ε * A.a - A.lam ε * A.ρ * A.DD k ε (A.z k n) (A.z k m) +
    A.lam ε * A.ρ * A.Zph k ε u₁ / (u₁ ^ 2 - u₂ ^ 2) * (1 / (A.z k n ^ 2 - u₁ ^ 2)) *
      (1 / (A.z k m ^ 2 - u₁ ^ 2)) +
    A.lam ε * A.ρ * A.Zph k ε u₂ / (u₂ ^ 2 - u₁ ^ 2) * (1 / (A.z k n ^ 2 - u₂ ^ 2)) *
      (1 / (A.z k m ^ 2 - u₂ ^ 2))

variable (A k) in
/-- `G₂(zₙ) = (xₙ + aρ)(x_{n+k} + aρ) ≠ 0`. -/
theorem G₂_z_ne_zero (n : ℤ) : A.G₂ k (A.z k n) ≠ 0 := by
  rw [show A.G₂ k (A.z k n) = (A.x n + A.a * A.ρ) * (A.x (n + k) + A.a * A.ρ) by
    rw [G₂, ← A.x_add_w, x_eq_z A k]; ring]
  exact mul_ne_zero (A.x_add_a_rho_ne_zero n) (A.x_add_a_rho_ne_zero (n + k))

variable (A k) in
/-- Eq. (B) for `r = d, m`: `B(νₙ) = -βG₂(zₙ)/P(zₙ)` (Eqs. G, Qfact). -/
theorem Bph_eq (n : ℤ) : A.Bph k n = -A.β * A.G₂ k (A.z k n) / A.Pph k (A.z k n) := by
  have hP := A.Pph_z_ne_zero k n
  have h₁ := A.x_sq_sub_ne_zero n
  have h₂ : (A.x n + A.w k) ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [A.x_add_w]
    exact A.x_sq_sub_ne_zero _
  rw [eq_div_iff hP, Bph, G, G, G₂, Pph, Qp, Qm, z]
  field_simp
  ring

/-- The partial fractions of Step 4: `F(t) = 1 + F₁/(t² - s₁) + F₂/(t² - s₂)` with
`Fⱼ = P(uⱼ)/(sⱼ - s_ℓ)`, as `P - e` is affine in `t²`. -/
theorem DMGeneric.Fph_eq_pf (h : A.DMGeneric k ε u₁ u₂) (t : ℂ) (h₁ : t ^ 2 - u₁ ^ 2 ≠ 0)
    (h₂ : t ^ 2 - u₂ ^ 2 ≠ 0) :
    A.Fph k ε t = 1 + (A.Pph k u₁ / (u₁ ^ 2 - u₂ ^ 2) / (t ^ 2 - u₁ ^ 2) +
      A.Pph k u₂ / (u₂ ^ 2 - u₁ ^ 2) / (t ^ 2 - u₂ ^ 2)) := by
  have hP₁ := A.Pph_sub_eph k ε u₁
  have hP₂ := A.Pph_sub_eph k ε u₂
  rw [h.e_u₁, sub_zero] at hP₁
  rw [h.e_u₂, sub_zero] at hP₂
  have hPt : A.Pph k t = A.eph k ε t + (((A.Pph k 1 - A.eph k ε 1) -
      (A.Pph k 0 - A.eph k ε 0)) * t ^ 2 + (A.Pph k 0 - A.eph k ε 0)) := by
    linear_combination A.Pph_sub_eph k ε t
  have hs := sub_ne_zero.mpr h.s_ne
  have hs' : u₂ ^ 2 - u₁ ^ 2 ≠ 0 := fun h0 ↦ hs (by linear_combination -h0)
  rw [Fph, hP₁, hP₂, hPt, h.eph_eq t]
  field_simp
  ring

/-- `F' = -2z(F₁/(z² - s₁)² + F₂/(z² - s₂)²)` at lattice points. -/
theorem DMGeneric.hasDerivAt_Fph (h : A.DMGeneric k ε u₁ u₂) (n : ℤ) :
    HasDerivAt (A.Fph k ε)
      (-2 * A.z k n * (A.Pph k u₁ / (u₁ ^ 2 - u₂ ^ 2) / (A.z k n ^ 2 - u₁ ^ 2) ^ 2 +
        A.Pph k u₂ / (u₂ ^ 2 - u₁ ^ 2) / (A.z k n ^ 2 - u₂ ^ 2) ^ 2)) (A.z k n) := by
  have h₁ := h.z_sq_ne₁ n
  have h₂ := h.z_sq_ne₂ n
  have hd : ∀ c F : ℂ, A.z k n ^ 2 - c ≠ 0 →
      HasDerivAt (fun t ↦ F / (t ^ 2 - c)) (F * (-(2 * A.z k n) / (A.z k n ^ 2 - c) ^ 2))
        (A.z k n) := by
    intro c F hc
    have := ((hasDerivAt_pow 2 (A.z k n)).sub_const c).inv hc
    convert this.const_mul F using 1
    · funext t
      rw [div_eq_mul_inv]
      rfl
    · push_cast
      ring
  have hg := ((hd (u₁ ^ 2) (A.Pph k u₁ / (u₁ ^ 2 - u₂ ^ 2)) h₁).add
    (hd (u₂ ^ 2) (A.Pph k u₂ / (u₂ ^ 2 - u₁ ^ 2)) h₂)).const_add 1
  have hev : ∀ᶠ t in 𝓝 (A.z k n), A.Fph k ε t = 1 +
      (A.Pph k u₁ / (u₁ ^ 2 - u₂ ^ 2) / (t ^ 2 - u₁ ^ 2) +
        A.Pph k u₂ / (u₂ ^ 2 - u₁ ^ 2) / (t ^ 2 - u₂ ^ 2)) := by
    have hc : ContinuousAt (fun t : ℂ ↦ (t ^ 2 - u₁ ^ 2) * (t ^ 2 - u₂ ^ 2)) (A.z k n) := by
      fun_prop
    filter_upwards [hc.eventually_ne (mul_ne_zero h₁ h₂)] with t ht
    exact h.Fph_eq_pf t (left_ne_zero_of_mul ht) (right_ne_zero_of_mul ht)
  convert hg.congr_of_eventuallyEq hev using 1
  ring

/-- Eq. (defect-dd): `DD(z, v) = F(v)/(z + v) - z ∑ⱼ Fⱼgⱼ(z)gⱼ(v)` on the lattice, including the
diagonal. -/
theorem DMGeneric.DD_eq (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    A.DD k ε (A.z k n) (A.z k m) = A.Fph k ε (A.z k m) / (A.z k n + A.z k m) -
      A.z k n * (A.Pph k u₁ / (u₁ ^ 2 - u₂ ^ 2) * (1 / (A.z k n ^ 2 - u₁ ^ 2)) *
          (1 / (A.z k m ^ 2 - u₁ ^ 2)) +
        A.Pph k u₂ / (u₂ ^ 2 - u₁ ^ 2) * (1 / (A.z k n ^ 2 - u₂ ^ 2)) *
          (1 / (A.z k m ^ 2 - u₂ ^ 2))) := by
  have hzv := A.z_add_z_ne_zero k h.μ_ne n m
  have hs := sub_ne_zero.mpr h.s_ne
  have hs' : u₂ ^ 2 - u₁ ^ 2 ≠ 0 := fun h0 ↦ hs (by linear_combination -h0)
  have hn₁ := h.z_sq_ne₁ n
  have hn₂ := h.z_sq_ne₂ n
  have hm₁ := h.z_sq_ne₁ m
  have hm₂ := h.z_sq_ne₂ m
  by_cases hnm : n = m
  · subst hnm
    have hz0 : A.z k n ≠ 0 := fun h0 ↦ hzv (by rw [h0]; ring)
    rw [DD, ite_eq_left rfl, (h.hasDerivAt_Fph n).deriv, h.Fph_eq_pf _ hn₁ hn₂]
    field_simp
    ring
  · have hne : A.z k n ≠ A.z k m := fun h0 ↦ hnm (A.z_injective k h0)
    have hsq : A.z k n ^ 2 - A.z k m ^ 2 ≠ 0 := by
      rw [show A.z k n ^ 2 - A.z k m ^ 2 = (A.z k n - A.z k m) * (A.z k n + A.z k m) by ring]
      exact mul_ne_zero (sub_ne_zero.mpr hne) hzv
    rw [DD, ite_eq_right hne, h.Fph_eq_pf _ hn₁ hn₂, h.Fph_eq_pf _ hm₁ hm₂]
    field_simp
    ring

/-- `β²cⱼφⱼ(z)` splits into the `DD` part `λρFⱼ z gⱼ(z)gⱼ(v)` and the `γⱼ` part, by
`φ(z) = (Kz + u(βH(-u) - λα(u)))/(z² - u²)`. -/
theorem cmode_mul_mode {u u' z v : ℂ} (hK : A.Kph k ε u ≠ 0) (hs : u ^ 2 - u' ^ 2 ≠ 0)
    (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0) (hv : v ^ 2 - u ^ 2 ≠ 0) :
    (A.β : ℂ) ^ 2 * (A.cmode k ε u u' v * A.mode k ε u z) =
      A.lam ε * A.ρ * (A.Pph k u / (u ^ 2 - u' ^ 2)) * z * (1 / (z ^ 2 - u ^ 2)) *
          (1 / (v ^ 2 - u ^ 2)) +
        A.lam ε * A.ρ * A.Zph k ε u / (u ^ 2 - u' ^ 2) * (1 / (z ^ 2 - u ^ 2)) *
          (1 / (v ^ 2 - u ^ 2)) := by
  have hβ := A.β_ne_zero
  have hz : z ^ 2 - u ^ 2 ≠ 0 := by
    rw [show z ^ 2 - u ^ 2 = (z - u) * (z + u) by ring]
    exact mul_ne_zero hzu hzu'
  rw [Kph] at hK
  simp only [cmode, mode, Zph, Kph]
  field_simp
  ring

/-- `cmode_mul_mode` for a root satisfying `ModeOK`; for the atomic pair both sides vanish. -/
theorem DMGeneric.mode_split (h : A.DMGeneric k ε u₁ u₂) {u u' : ℂ} (hu : A.eph k ε u = 0)
    (hmode : A.ModeOK k ε u) (hs : u ^ 2 - u' ^ 2 ≠ 0) (n m : ℤ)
    (hv : A.z k m ^ 2 - u ^ 2 ≠ 0) :
    (A.β : ℂ) ^ 2 * (A.cmode k ε u u' (A.z k m) * A.mode k ε u (A.z k n)) =
      A.lam ε * A.ρ * (A.Pph k u / (u ^ 2 - u' ^ 2)) * A.z k n * (1 / (A.z k n ^ 2 - u ^ 2)) *
          (1 / (A.z k m ^ 2 - u ^ 2)) +
        A.lam ε * A.ρ * A.Zph k ε u / (u ^ 2 - u' ^ 2) * (1 / (A.z k n ^ 2 - u ^ 2)) *
          (1 / (A.z k m ^ 2 - u ^ 2)) := by
  rcases hmode with ⟨hP, hK⟩ | ⟨hP, hW⟩
  · exact cmode_mul_mode hK hs (sub_ne_zero.mpr (h.z_ne hu n))
      (fun h0 ↦ h.z_ne_neg hu n (by linear_combination h0)) hv
  · simp only [cmode, Zph, hP, zero_div, mul_zero, zero_mul, add_zero]

/-- **Theorem "density and magnetic"** (generic case), the closed form: for the inverse `invR` of
`χ_r`, `β²(χ_r⁻¹ - B⁻¹) = Γ_r` of Eq. (Gamma-ph) (Eq. Gamma-def with `η_r = 1`). -/
theorem DMGeneric.gammaR_eq (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invR k ε u₁ u₂ n m - kron n m / A.Bph k n) =
      A.gammaR k ε u₁ u₂ n m := by
  have hβ := A.β_ne_zero
  have hs := sub_ne_zero.mpr h.s_ne
  have hs' : u₂ ^ 2 - u₁ ^ 2 ≠ 0 := fun h0 ↦ hs (by linear_combination -h0)
  have hzv := A.z_add_z_ne_zero k h.μ_ne n m
  have hPn := A.Pph_z_ne_zero k n
  have hGn := A.G₂_z_ne_zero k n
  have hen := h.e_ne n
  have hem := h.e_ne m
  have hn₁ := h.z_sq_ne₁ n
  have hn₂ := h.z_sq_ne₂ n
  have hm₁ := h.z_sq_ne₁ m
  have hm₂ := h.z_sq_ne₂ m
  have e₁ := h.mode_split h.e_u₁ h.mode₁ hs n m hm₁
  have e₂ := h.mode_split h.e_u₂ h.mode₂ hs' n m hm₂
  have hsplit : (A.β : ℂ) ^ 2 * (A.invR k ε u₁ u₂ n m - kron n m / A.Bph k n) =
      A.β ^ 2 * (A.Dv k ε (A.z k m) * kron n m + A.Ev k ε (A.z k m) / (A.z k n + A.z k m) -
        kron n m / A.Bph k n) + 2 * ε * A.a +
        A.β ^ 2 * (A.cmode k ε u₁ u₂ (A.z k m) * A.mode k ε u₁ (A.z k n)) +
        A.β ^ 2 * (A.cmode k ε u₂ u₁ (A.z k m) * A.mode k ε u₂ (A.z k n)) := by
    rw [invR]
    field_simp
    ring
  rw [hsplit, e₁, e₂, gammaR, h.DD_eq n m, A.Bph_eq k n]
  simp only [Dv, Ev]
  by_cases hnm : n = m
  · subst hnm
    have hz0 : A.z k n ≠ 0 := fun h0 ↦ hzv (by rw [h0]; ring)
    rw [kron_self]
    field_simp
    ring
  · rw [kron_of_ne hnm]
    field_simp
    ring

/-! ### Two-sidedness -/

variable (A k ε) in
theorem Nph_sub_swap (z v : ℂ) : A.Nph k ε z v - A.Nph k ε v z =
    -(A.lam ε * A.ρ) * (v - z) * (v ^ 2 + z ^ 2 + A.κ k) := by
  rw [A.Nph_eq_coeffs, A.Nph_eq_coeffs]
  simp only [cN₀, cN₁, cN₂, cN₃, lam]
  ring

variable (A k) in
theorem Pph_sub_Pph (z v : ℂ) :
    A.Pph k v - A.Pph k z = (v - z) * (v + z) * (v ^ 2 + z ^ 2 + A.κ k) := by
  rw [A.Pph_eq_poly, A.Pph_eq_poly]
  ring

variable (A k ε) in
/-- `χ_r` is symmetric, for `μ ≠ 0` (via Proposition "structure"). -/
theorem chiR_symm (hμ : A.μ ≠ 0) (n m : ℤ) : A.chiR k ε n m = A.chiR k ε m n := by
  by_cases hnm : n = m
  · rw [hnm]
  have hPn := A.Pph_z_ne_zero k n
  have hPm := A.Pph_z_ne_zero k m
  have h₁ := A.Pph_mul_chiR k ε hμ n m
  have h₂ := A.Pph_mul_chiR k ε hμ m n
  rw [kron_of_ne hnm, mul_zero, zero_add] at h₁
  rw [kron_of_ne (Ne.symm hnm), mul_zero, zero_add] at h₂
  have hc₁ : A.chiR k ε n m = (A.lam ε * A.ρ / (A.z k n + A.z k m) +
      A.Nph k ε (A.z k n) (A.z k m) / A.Pph k (A.z k m)) / A.Pph k (A.z k n) := by
    rw [eq_div_iff hPn, mul_comm]
    exact h₁
  have hc₂ : A.chiR k ε m n = (A.lam ε * A.ρ / (A.z k m + A.z k n) +
      A.Nph k ε (A.z k m) (A.z k n) / A.Pph k (A.z k n)) / A.Pph k (A.z k m) := by
    rw [eq_div_iff hPm, mul_comm]
    exact h₂
  rw [hc₁, hc₂]
  exact symm_alg hPn hPm (A.z_add_z_ne_zero k hμ n m) (A.Pph_sub_Pph k _ _)
    (A.Nph_sub_swap k ε _ _)

variable (A k ε) in
theorem DD_symm (z v : ℂ) : A.DD k ε z v = A.DD k ε v z := by
  unfold DD
  by_cases hzv : z = v
  · subst hzv
    rfl
  · rw [ite_eq_right hzv, ite_eq_right (Ne.symm hzv),
      show v * A.Fph k ε v - z * A.Fph k ε z = -(z * A.Fph k ε z - v * A.Fph k ε v) by ring,
      show v ^ 2 - z ^ 2 = -(z ^ 2 - v ^ 2) by ring, neg_div_neg_eq]

variable (A k ε) in
/-- `Γ_r` of Eq. (Gamma-ph) is symmetric. -/
theorem gammaR_symm (u₁ u₂ : ℂ) (n m : ℤ) :
    A.gammaR k ε u₁ u₂ n m = A.gammaR k ε u₁ u₂ m n := by
  by_cases hnm : n = m
  · rw [hnm]
  · rw [gammaR, gammaR, kron_of_ne hnm, kron_of_ne (Ne.symm hnm), A.DD_symm k ε]
    ring

theorem DMGeneric.invR_symm (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    A.invR k ε u₁ u₂ n m = A.invR k ε u₁ u₂ m n := by
  have hβ : (A.β : ℂ) ^ 2 ≠ 0 := pow_ne_zero 2 A.β_ne_zero
  have hk : kron n m / A.Bph k n = kron m n / A.Bph k m := by
    by_cases hnm : n = m
    · rw [hnm]
    · rw [kron_of_ne hnm, kron_of_ne (Ne.symm hnm), zero_div, zero_div]
  apply mul_left_cancel₀ hβ
  linear_combination h.gammaR_eq n m - h.gammaR_eq m n + A.gammaR_symm k ε u₁ u₂ n m +
    (A.β : ℂ) ^ 2 * hk

/-- **Theorem "density and magnetic"**, two-sidedness (Step 5): `invR χ_r = 1`, as `χ_r` and
`invR` are symmetric. -/
theorem DMGeneric.chiR_isLeftInverse (h : A.DMGeneric k ε u₁ u₂) :
    IsLeftInverse (A.chiR k ε) (A.invR k ε u₁ u₂) := by
  intro n m
  have := h.chiR_isRightInverse m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [A.chiR_symm k ε h.μ_ne l m, h.invR_symm n l, mul_comm]

/-! ### The static magnetic channel -/

variable (A) in
/-- The atomic pair of `r = m`, `k = 0`: `e(a) = 0`, as `H(a) = 4a²p₂` and `H(-a) = 4a²p₀`
(identity (A5)); `a` is a double zero of `P`, and `W(z; a²) = 0` ((A7)). So the root `u = a`
satisfies `ModeOK` and drops out of Theorem "density and magnetic". -/
theorem atomic_root : A.eph 0 (-1) A.a = 0 ∧ A.ModeOK 0 (-1) A.a := by
  have hs := A.weights_sum'
  have hρ := A.ρ_eq
  have hw := A.w_zero
  refine ⟨?_, Or.inr ⟨?_, fun z ↦ ?_⟩⟩
  · simp only [eph, Hph, lam, q₀, hw]
    rw [hρ]
    generalize (A.p₂ : ℂ) = p₂ at hs ⊢
    obtain rfl : p₂ = 1 - A.p₀ - 2 * A.p₁ := by linear_combination hs
    ring
  · rw [Pph, Qp, Qm, hw]
    ring
  · simp only [Wph, W₀, W₂, ψ, q₀, hw, kron_self]
    ring

variable (A) in
/-- **Theorem "density and magnetic"** applies to the static magnetic channel (`r = m`, `k = 0`),
with the atomic root `u₁ = a` and the other root `u₂`, under the assumptions on `u₂` alone. -/
theorem dmGeneric_static_magnetic {u₂ : ℂ} (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    (he : ∀ n, A.eph 0 (-1) (A.z 0 n) ≠ 0) (hu₂ : A.eph 0 (-1) u₂ = 0)
    (hs : (A.a : ℂ) ^ 2 ≠ u₂ ^ 2)
    (hP : A.Pph 0 u₂ ≠ 0) (hK : A.Kph 0 (-1) u₂ ≠ 0) : A.DMGeneric 0 (-1) A.a u₂ :=
  ⟨Or.inr rfl, ha, hμ, he, (A.atomic_root).1, hu₂, hs, (A.atomic_root).2, Or.inl ⟨hP, hK⟩⟩

end Params

end HubbardAtom
