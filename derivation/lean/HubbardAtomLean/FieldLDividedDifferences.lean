import HubbardAtomLean.FieldLSym

/-!
# The longitudinal vertex as a divided difference

Off the diagonal, the longitudinal vertex of Eq. (field-L) is a divided difference of a `2 × 2`
matrix of functions of one frequency, `W_στ` (`FParams.WL`), on the two spin lattices:

* `FParams.gammaL_divided_difference`: `(X_σn - X_tm)Γ_{L,σt}(νₙ, νₘ) = W_σt(z) - W_tσ(v)`, with
  `X = z²` on the lattice of each spin, `z = z_σn`, `v = z_tm`, for all `σ, n, t, m`, under
  `LGeneric` for both column spins. Here
  `W_σt(z) = δ_{σ,-t}[Uz² - λr_σz + aλ(r_σ² - r_t²)] + β²∑ⱼ 𝔠ⱼₜ[f_{jσ+}/(z - uⱼ) +
  f_{jσ-}/(z + uⱼ)]`, the mode of each root with the coefficient `𝔠ⱼₜ` of the column spin `t`: the residue of `W_σt`
  at `z = uⱼ` is `β²𝔠ⱼₜf_{jσ+}`.

The dense part is reduced by the partial fractions of `FieldLSym.lean` (`X_pf`, `Y_pf`, `R2_pf`);
each root then contributes a difference of its own (`dd_root_opp`, `dd_root_self`), for opposite
spins by the three identities at the root that give the symmetry of `Γ_L` (`res_z`, `res_v`,
`res_c`). On the diagonal both sides vanish.
-/

open Complex

namespace HubbardAtom

/-- **One root, opposite spins**: the term of the root `u` in `(z² - v²)Γ_L(-t, z; t, v)` is the
difference of the terms of the modes, up to the constant `8aλρm R₂(u²)/E'`, given the three
identities at the root. -/
private theorem dd_root_opp {lam m ρ a q β z v u d p Pi Xi Kt Kp fp fm gp gm aT aN : ℂ}
    (hβ : β ≠ 0) (hd : d ≠ 0) (hKt : Kt ≠ 0) (hKp : Kp ≠ 0) (hzv : z - v ≠ 0) (hzv' : z + v ≠ 0)
    (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0) (hvu : v - u ≠ 0) (hvu' : v + u ≠ 0)
    (ez : aT * (fp + fm) = -(lam * (ρ * Xi + m * Pi)) * Kt)
    (ev : aN * (gp + gm) = -(lam * (ρ * Xi - m * Pi)) * Kp)
    (ec : 8 * a * lam * ρ * m * u ^ 2 * (u ^ 2 + q) * Kt * Kp - aT * (fp - fm) * u * Kp +
      aN * (gp - gm) * u * Kt = 0) :
    (z ^ 2 - v ^ 2) * (-(lam * m) * ((p * Pi - 4 * a * ρ * v * (p * (u ^ 2 + q))) / d /
          (v ^ 2 - u ^ 2)) / (z - v) -
        lam * ρ * ((p * Xi + 4 * a * m * v * (p * (u ^ 2 + q))) / d / (v ^ 2 - u ^ 2)) / (z + v) -
        β ^ 2 * (aT / (β ^ 2 * Kt * d) * p / (v ^ 2 - u ^ 2) * (fp / (z - u) + fm / (z + u)))) =
      β ^ 2 * (aT / (β ^ 2 * Kt * d) * p * (fp / (z - u) + fm / (z + u))) -
        β ^ 2 * (aN / (β ^ 2 * Kp * d) * p * (gp / (v - u) + gm / (v + u))) +
      8 * a * lam * ρ * m * (p * (u ^ 2 + q) / d) := by
  have hvu2 : v ^ 2 - u ^ 2 ≠ 0 := by
    rw [show v ^ 2 - u ^ 2 = (v - u) * (v + u) by ring]
    exact mul_ne_zero hvu hvu'
  rw [← sub_eq_zero]
  have key : (z ^ 2 - v ^ 2) * (-(lam * m) * ((p * Pi - 4 * a * ρ * v * (p * (u ^ 2 + q))) / d /
          (v ^ 2 - u ^ 2)) / (z - v) -
        lam * ρ * ((p * Xi + 4 * a * m * v * (p * (u ^ 2 + q))) / d / (v ^ 2 - u ^ 2)) / (z + v) -
        β ^ 2 * (aT / (β ^ 2 * Kt * d) * p / (v ^ 2 - u ^ 2) * (fp / (z - u) + fm / (z + u)))) -
      (β ^ 2 * (aT / (β ^ 2 * Kt * d) * p * (fp / (z - u) + fm / (z + u))) -
        β ^ 2 * (aN / (β ^ 2 * Kp * d) * p * (gp / (v - u) + gm / (v + u))) +
      8 * a * lam * ρ * m * (p * (u ^ 2 + q) / d)) =
      p / (d * (v ^ 2 - u ^ 2) * Kt * Kp) *
        (-z * Kp * (aT * (fp + fm) + lam * (ρ * Xi + m * Pi) * Kt) +
          v * Kt * (aN * (gp + gm) + lam * (ρ * Xi - m * Pi) * Kp) +
          (8 * a * lam * ρ * m * u ^ 2 * (u ^ 2 + q) * Kt * Kp - aT * (fp - fm) * u * Kp +
            aN * (gp - gm) * u * Kt)) := by
    field_simp
    ring
  rw [key, ec, show aT * (fp + fm) + lam * (ρ * Xi + m * Pi) * Kt = 0 by linear_combination ez,
    show aN * (gp + gm) + lam * (ρ * Xi - m * Pi) * Kp = 0 by linear_combination ev]
  ring

/-- **One root, equal spins**: the term of the root `u` in `(z² - v²)Γ_L(t, z; t, v)` is the
difference of the terms of the mode. -/
private theorem dd_root_self {lam D q β z v u d p K fp fm : ℂ} (hβ : β ≠ 0) (hd : d ≠ 0)
    (hK : K ≠ 0) (hK' : fp + fm = K) (hzv : z + v ≠ 0) (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0)
    (hvu : v - u ≠ 0) (hvu' : v + u ≠ 0) :
    (z ^ 2 - v ^ 2) * (2 * lam ^ 2 * D * (p * (u ^ 2 + q) / d / (v ^ 2 - u ^ 2)) / (z + v) -
        β ^ 2 * (2 * lam ^ 2 * D * (u ^ 2 + q) / (β ^ 2 * K * d) * p / (v ^ 2 - u ^ 2) *
          (fp / (z - u) + fm / (z + u)))) =
      β ^ 2 * (2 * lam ^ 2 * D * (u ^ 2 + q) / (β ^ 2 * K * d) * p *
          (fp / (z - u) + fm / (z + u))) -
        β ^ 2 * (2 * lam ^ 2 * D * (u ^ 2 + q) / (β ^ 2 * K * d) * p *
          (fp / (v - u) + fm / (v + u))) := by
  have hv2 : v ^ 2 - u ^ 2 = (v - u) * (v + u) := by ring
  rw [← hK'] at hK ⊢
  rw [hv2]
  field_simp
  ring

/-- The residues of a monic cubic over four distinct nodes sum to one. -/
private theorem sum_monic_div_dprod {c₀ c₁ c₂ : ℂ} {s : Fin 4 → ℂ}
    (hs : ∀ i j, i ≠ j → s i - s j ≠ 0) :
    ∑ j, (c₀ + c₁ * s j + c₂ * s j ^ 2 + s j ^ 3) / dprod s j = 1 := by
  have h₀₁ := hs 0 1 (by decide)
  have h₀₂ := hs 0 2 (by decide)
  have h₀₃ := hs 0 3 (by decide)
  have h₁₂ := hs 1 2 (by decide)
  have h₁₃ := hs 1 3 (by decide)
  have h₂₃ := hs 2 3 (by decide)
  have h₁₀ := hs 1 0 (by decide)
  have h₂₀ := hs 2 0 (by decide)
  have h₃₀ := hs 3 0 (by decide)
  have h₂₁ := hs 2 1 (by decide)
  have h₃₁ := hs 3 1 (by decide)
  have h₃₂ := hs 3 2 (by decide)
  simp only [Fin.sum_univ_four, dprod]
  field_simp
  ring

namespace FParams

variable (F : FParams) (k : ℤ)

-- the per-root identities are `field_simp`/`ring` computations: one at a time
set_option Elab.async false

/-- `W_σt(z) = δ_{σ,-t}[Uz² - λr_σz + aλ(r_σ² - r_t²)] + β²∑ⱼ 𝔠ⱼₜ[f_{jσ+}/(z - uⱼ) +
f_{jσ-}/(z + uⱼ)]`, the mode normalized with the column spin `t` and `𝔠ⱼₜ = P(uⱼ) cMode_j`;
written with `U = 2a`, `r_σ = ρ + m_t` and `r_σ² - r_t² = 4ρm_t` for `σ = -t`. -/
noncomputable def WL (u : Fin 4 → ℂ) (σ t : Bool) (z : ℂ) : ℂ :=
  (if σ = t then 0
    else 2 * F.a * z ^ 2 - F.lamL * (F.mL t + F.ρ) * z + 4 * F.a * F.lamL * F.ρ * F.mL t) +
    F.β ^ 2 * ∑ j, F.cMode k t u j * F.PL k (u j) *
      (F.fMp k t σ (u j) / (z - u j) + F.fMm k t σ (u j) / (z + u j))

theorem R2Y_cubic_monic (y : ℂ) :
    F.R2Y k y = F.q₀L k ^ 3 + (3 * F.q₀L k ^ 2 - 4 * F.a ^ 2 * F.q₀L k) * y +
      (3 * F.q₀L k - 4 * F.a ^ 2) * y ^ 2 + y ^ 3 := by
  simp only [R2Y, pY]
  ring

/-- `∑ⱼ R₂(s_j)/E'ⱼ = 1`, as `R₂ = p(y)(y + q₀)` is a monic cubic. -/
theorem sum_R2Y_div {u : Fin 4 → ℂ} (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) :
    ∑ j, F.R2Y k (u j ^ 2) / dprod (fun i ↦ u i ^ 2) j = 1 := by
  simp only [F.R2Y_cubic_monic k]
  exact sum_monic_div_dprod (s := fun i ↦ u i ^ 2) sep

variable {F k}

/-- The term of the root `u_j` in `(z² - v²)Γ_L(-t, z; t, v)`, `z = zₙ` on the lattice of `-t`,
`v = zₘ` on that of `t`. -/
theorem LGeneric.dd_opp_root {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (t : Bool) (n m : ℤ)
    (j : Fin 4) :
    let z := (F.BL (!t)).z k n
    let v := (F.BL t).z k m
    let d := dprod (fun i ↦ u i ^ 2) j
    (z ^ 2 - v ^ 2) * (-(F.lamL * F.mL t) * ((F.R1Y k (u j ^ 2) -
            4 * F.a * F.ρ * v * F.R2Y k (u j ^ 2)) / d / (v ^ 2 - u j ^ 2)) / (z - v) -
          F.lamL * F.ρ * ((F.R3Y k (u j ^ 2) + 4 * F.a * F.mL t * v * F.R2Y k (u j ^ 2)) / d /
            (v ^ 2 - u j ^ 2)) / (z + v) -
          F.β ^ 2 * (F.cMode k t u j * F.PL k (u j) / (v ^ 2 - u j ^ 2) *
            (F.fMp k t (!t) (u j) / (z - u j) + F.fMm k t (!t) (u j) / (z + u j)))) =
      F.β ^ 2 * (F.cMode k t u j * F.PL k (u j) *
          (F.fMp k t (!t) (u j) / (z - u j) + F.fMm k t (!t) (u j) / (z + u j))) -
        F.β ^ 2 * (F.cMode k (!t) u j * F.PL k (u j) *
          (F.fMp k (!t) t (u j) / (v - u j) + F.fMm k (!t) t (u j) / (v + u j))) +
        8 * F.a * F.lamL * F.ρ * F.mL t * (F.R2Y k (u j ^ 2) / d) := by
  intro z v d
  have h := hg t
  have h' := hg (!t)
  have hβ := β_ne_zeroT (F := F)
  have hd := dprod_ne h.sep j
  have hzv : z - v ≠ 0 := by simpa using F.zL_sub_opp_ne k h.h_ne (!t) n m
  have hzv' : z + v ≠ 0 := by simpa using F.zL_add_opp_ne k h.μ_ne (!t) n m
  simp only [F.R1Y_root k (h.root j), F.R3Y_root k (h.root j), F.R2Y_eq k]
  by_cases ha : F.Atomic k (u j)
  · simp [cMode, ha, PL_atomic ha]
  · have hz := h.off j ha (!t) n
    have hv := h.off j ha t m
    have e := dd_root_opp (p := F.PL k (u j)) hβ hd (h.K_ne j ha) (h'.K_ne j ha) hzv hzv'
      (sub_ne_zero.mpr hz.1) (fun h0 ↦ hz.2 (by linear_combination h0)) (sub_ne_zero.mpr hv.1)
      (fun h0 ↦ hv.2 (by linear_combination h0)) (F.res_z k t (h.root j))
      (F.res_v k h.colNe (h.roots.rel j ha) (h.root j))
      (F.res_c k h.colNe (h.roots.rel j ha) (h.root j))
    simp only [cMode, ha, ↓reduceIte, aAL]
    simp only [aAL] at e
    linear_combination e

/-- The term of the root `u_j` in `(z² - v²)Γ_L(t, z; t, v)`, `z = zₙ`, `v = zₘ` on the lattice
of `t`. -/
theorem LGeneric.dd_self_root {t : Bool} {u : Fin 4 → ℂ} (h : F.LGeneric k t u) (n m : ℤ)
    (j : Fin 4) :
    let z := (F.BL t).z k n
    let v := (F.BL t).z k m
    (z ^ 2 - v ^ 2) * (2 * F.lamL ^ 2 * F.DL t *
          (F.R2Y k (u j ^ 2) / dprod (fun i ↦ u i ^ 2) j / (v ^ 2 - u j ^ 2)) / (z + v) -
        F.β ^ 2 * (F.cMode k t u j * F.PL k (u j) / (v ^ 2 - u j ^ 2) *
          (F.fMp k t t (u j) / (z - u j) + F.fMm k t t (u j) / (z + u j)))) =
      F.β ^ 2 * (F.cMode k t u j * F.PL k (u j) *
          (F.fMp k t t (u j) / (z - u j) + F.fMm k t t (u j) / (z + u j))) -
        F.β ^ 2 * (F.cMode k t u j * F.PL k (u j) *
          (F.fMp k t t (u j) / (v - u j) + F.fMm k t t (u j) / (v + u j))) := by
  intro z v
  have hβ := β_ne_zeroT (F := F)
  have hd := dprod_ne h.sep j
  have hzv : z + v ≠ 0 := F.zL_add_self_ne k h.c_ne n m
  simp only [F.R2Y_eq k]
  by_cases ha : F.Atomic k (u j)
  · simp [cMode, ha, PL_atomic ha]
  · have hz := h.off j ha t n
    have hv := h.off j ha t m
    have e := dd_root_self (lam := F.lamL) (D := F.DL t) (q := F.q₀L k) (p := F.PL k (u j))
      hβ hd (h.K_ne j ha) rfl hzv (sub_ne_zero.mpr hz.1)
      (fun h0 ↦ hz.2 (by linear_combination h0)) (sub_ne_zero.mpr hv.1)
      (fun h0 ↦ hv.2 (by linear_combination h0))
    simp only [cMode, ha, ↓reduceIte, aAL]
    linear_combination e

/-- **The longitudinal vertex as a divided difference, opposite spins**:
`(z² - v²)Γ_{L,-t,t}(νₙ, νₘ) = W_{-t,t}(z) - W_{t,-t}(v)`, `z = zₙ` on the lattice of `-t`,
`v = zₘ` on that of `t`. -/
theorem LGeneric.gammaL_dd_opp {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (t : Bool)
    (n m : ℤ) :
    ((F.BL (!t)).z k n ^ 2 - (F.BL t).z k m ^ 2) * F.gammaL k u (!t) n t m =
      F.WL k u (!t) t ((F.BL (!t)).z k n) - F.WL k u t (!t) ((F.BL t).z k m) := by
  have h := hg t
  have hzv : (F.BL (!t)).z k n - (F.BL t).z k m ≠ 0 := by
    simpa using F.zL_sub_opp_ne k h.h_ne (!t) n m
  have hzv' : (F.BL (!t)).z k n + (F.BL t).z k m ≠ 0 := by
    simpa using F.zL_add_opp_ne k h.μ_ne (!t) n m
  have hT := fun j ↦ LGeneric.dd_opp_root hg t n m j
  have hX := F.X_pf k t h.root h.sep (h.lat_ne m)
  have hY := F.Y_pf k t h.root h.sep (h.lat_ne m)
  have hs := F.sum_R2Y_div k h.sep
  have hm' : (F.mL (!t) : ℂ) = -F.mL t := by rw [F.mL_not t]; push_cast; ring
  have c₁ : ((F.BL (!t)).z k n ^ 2 - (F.BL t).z k m ^ 2) /
      ((F.BL (!t)).z k n - (F.BL t).z k m) = (F.BL (!t)).z k n + (F.BL t).z k m := by
    field_simp
    ring
  have c₂ : ((F.BL (!t)).z k n ^ 2 - (F.BL t).z k m ^ 2) /
      ((F.BL (!t)).z k n + (F.BL t).z k m) = (F.BL (!t)).z k n - (F.BL t).z k m := by
    field_simp
    ring
  simp only at hT
  simp only [gammaL, WL, Bool.not_ne_self, Bool.self_ne_not, ↓reduceIte, hm']
  simp only [Fin.sum_univ_four] at hX hY hs ⊢
  linear_combination
    (-(F.lamL * F.mL t) * ((F.BL (!t)).z k n ^ 2 - (F.BL t).z k m ^ 2) /
      ((F.BL (!t)).z k n - (F.BL t).z k m)) * hX +
    (-(F.lamL * F.ρ) * ((F.BL (!t)).z k n ^ 2 - (F.BL t).z k m ^ 2) /
      ((F.BL (!t)).z k n + (F.BL t).z k m)) * hY -
    F.lamL * F.mL t * c₁ - F.lamL * F.ρ * c₂ + 8 * F.a * F.lamL * F.ρ * F.mL t * hs +
    hT 0 + hT 1 + hT 2 + hT 3

/-- **The longitudinal vertex as a divided difference, equal spins**:
`(z² - v²)Γ_{L,tt}(νₙ, νₘ) = W_tt(z) - W_tt(v)`, `z = zₙ`, `v = zₘ` on the lattice of `t`. -/
theorem LGeneric.gammaL_dd_self {t : Bool} {u : Fin 4 → ℂ} (h : F.LGeneric k t u) (n m : ℤ) :
    ((F.BL t).z k n ^ 2 - (F.BL t).z k m ^ 2) * F.gammaL k u t n t m =
      F.WL k u t t ((F.BL t).z k n) - F.WL k u t t ((F.BL t).z k m) := by
  rcases eq_or_ne n m with rfl | hnm
  · rw [sub_self, zero_mul, sub_self]
  have hT := fun j ↦ h.dd_self_root n m j
  have hR := F.R2_pf k h.root h.sep (h.lat_ne m)
  simp only at hT
  simp only [gammaL, WL, ↓reduceIte, kron_of_ne hnm, mul_zero, zero_add]
  simp only [← div_div, Fin.sum_univ_four] at hR hT ⊢
  linear_combination
    2 * F.lamL ^ 2 * F.DL t * ((F.BL t).z k n ^ 2 - (F.BL t).z k m ^ 2) /
      ((F.BL t).z k n + (F.BL t).z k m) * hR + hT 0 + hT 1 + hT 2 + hT 3

/-- **The longitudinal vertex as a divided difference**:
`(X_σn - X_tm)Γ_{L,σt}(νₙ, νₘ) = W_σt(z_σn) - W_tσ(z_tm)`, `X = z²` on the lattice of each spin. -/
theorem gammaL_divided_difference {u : Fin 4 → ℂ} (hg : ∀ t, F.LGeneric k t u) (σ : Bool)
    (n : ℤ) (t : Bool) (m : ℤ) :
    ((F.BL σ).z k n ^ 2 - (F.BL t).z k m ^ 2) * F.gammaL k u σ n t m =
      F.WL k u σ t ((F.BL σ).z k n) - F.WL k u t σ ((F.BL t).z k m) := by
  by_cases hσ : σ = t
  · subst hσ
    exact (hg σ).gammaL_dd_self n m
  · obtain rfl : σ = !t := by cases σ <;> cases t <;> simp_all
    exact LGeneric.gammaL_dd_opp hg t n m

end FParams

end HubbardAtom
