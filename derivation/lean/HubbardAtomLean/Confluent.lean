import HubbardAtomLean.DMLifts

/-!
# Theorem "density and magnetic" at confluent roots

At a double root `s₁ = s₂ = u²` of `e` the sum over `j` in Eq. (Gamma-ph) is replaced by its limit
`λρ ∂ₛ[Z(s)gₛ(zₙ)gₛ(zₘ)]`, `gₛ(z) = 1/(z² - s)`. The proof is direct: the two modes of Step 3 are
replaced by the mode `Φ(u, t) = P(u)φᵤ(t)/K(u) = (P(u)t + Z(u))/(t² - u²)` and its `s`-derivative.

* `step3_general`: Step 3 for every `u`, with a remainder proportional to `e(u)`; at a double root
  it vanishes to second order, so the image of `∂ᵤΦ` is the derivative of the image of `Φ`
  (`hasSum_deriv_of_bound`, by `Complex.hasSum_deriv_of_summable_norm`); at `u = 0`, from
  difference quotients by Tannery's theorem.
* `Params.DMConf`: the assumptions at a double root `u` (including `u = 0`), and
  `Params.DMConf.chiR_isRightInverse`, `Params.DMConf.chiR_isLeftInverse`,
  `Params.DMConf.gammaR_eq`: **Theorem "density and magnetic"** for `s₁ = s₂`.
* `Params.DMConfHyp`, `Params.DMConfHyp.toDMConf`: the note's assumptions, which imply
  `P(u) ≠ 0`; `Params.DMConf.tendsto_Zeven`: the coefficient `Zsd` is `∂ₛZ` of the even part.
-/

open Complex Filter Topology Metric

namespace HubbardAtom

/-- Termwise differentiation of a series of holomorphic functions bounded on a ball by a summable
sequence (`hasSum_deriv_of_summable_norm`): `∑ₗ fₗ'(u₀) = G'(u₀)` for `∑ₗ fₗ = G` on the ball. -/
theorem hasSum_deriv_of_bound {f : ℤ → ℂ → ℂ} {f' : ℤ → ℂ} {G : ℂ → ℂ} {G' u₀ : ℂ} {b : ℤ → ℝ}
    {r : ℝ} (hr : 0 < r) (hb : Summable b) (hdiff : ∀ l, DifferentiableOn ℂ (f l) (ball u₀ r))
    (hd : ∀ l, HasDerivAt (f l) (f' l) u₀) (hbound : ∀ l, ∀ u ∈ ball u₀ r, ‖f l u‖ ≤ b l)
    (hG : ∀ u ∈ ball u₀ r, HasSum (fun l ↦ f l u) (G u)) (hG' : HasDerivAt G G' u₀) :
    HasSum f' G' := by
  have h := hasSum_deriv_of_summable_norm hb hdiff isOpen_ball hbound (mem_ball_self hr)
  have heq : (fun w ↦ ∑' l, f l w) =ᶠ[𝓝 u₀] G := by
    filter_upwards [isOpen_ball.mem_nhds (mem_ball_self hr)] with w hw
    exact (hG w hw).tsum_eq
  rw [heq.deriv_eq, hG'.deriv] at h
  convert h using 1
  funext l
  exact (hd l).deriv.symm

/-- `(x - x₀)²h(x)` has derivative `0` at `x₀` if `h` is continuous at `x₀`. -/
theorem hasDerivAt_sq_mul {h : ℂ → ℂ} {x₀ : ℂ} (hc : ContinuousAt h x₀) :
    HasDerivAt (fun x ↦ (x - x₀) ^ 2 * h x) 0 x₀ := by
  rw [hasDerivAt_iff_tendsto_slope]
  have hlim : Tendsto (fun x ↦ (x - x₀) * h x) (𝓝[≠] x₀) (𝓝 0) := by
    have : Tendsto (fun x ↦ (x - x₀) * h x) (𝓝 x₀) (𝓝 ((x₀ - x₀) * h x₀)) :=
      ((continuousAt_id.sub continuousAt_const).mul hc).tendsto
    rw [sub_self, zero_mul] at this
    exact this.mono_left nhdsWithin_le_nhds
  refine hlim.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with x hx
  have hx' : x - x₀ ≠ 0 := sub_ne_zero.mpr hx
  rw [slope_def_field]
  field_simp
  ring

/-- Step 3 for every `u` (not only the roots of `e`): the image of `φᵤ` under `P(z)χ_r` is
`-βK[z + W(z; u²)/(ρP(u))] + e(u)Ξ` with `Ξ = β(K(N₂(z) + λρz) - βN(z, -u))/(λρP(u)) - β²/(z - u)`.
The inputs are `α(u)α(-u) = β²p₀p₂` with `e = H(u)H(-u) - λ²p₀p₂` (`hrel`), and Eq. (source) at
`v = ∓u` (`hp`, `hm`). -/
theorem step3_general {β lam ρ a q₁ z u P W αu αmu Nu Nmu Ru Rmu cb N2z e : ℂ} (hβ : β ≠ 0)
    (hlam : lam ≠ 0) (hρ : ρ ≠ 0) (hP : P ≠ 0) (hzu : z - u ≠ 0) (hzu' : z + u ≠ 0)
    (hαu : αu ≠ 0)
    (hrel : β ^ 2 * ((u ^ 2 + 2 * a * ρ * u + q₁) * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) - e) =
      lam ^ 2 * (αu * αmu))
    (hp : lam / β * (cb * Nu + ρ * Ru) = lam * ρ * (z + u + 2 * a * ρ) * P -
      (u ^ 2 + 2 * a * ρ * u + q₁) * Nmu + e * N2z - lam * ρ * z * (P - e) - lam * W)
    (hm : lam / β * (cb * Nmu + ρ * Rmu) = lam * ρ * (z - u + 2 * a * ρ) * P -
      ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) * Nu + e * N2z - lam * ρ * z * (P - e) - lam * W) :
    -β * (z ^ 2 + 2 * a * ρ * z + q₁) * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) / (z - u) +
          lam * αu / (z + u)) +
        lam * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁)) * (αu / (z + u)) +
        lam * (lam * αu) * (αmu / (z + -u)) +
        β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) * ((Nu * ((αu + cb) / ρ) + Ru) / P) +
      lam * αu * ((Nmu * ((αmu + cb) / ρ) + Rmu) / P) =
    -β * (β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) + lam * αu) * (z + W / (ρ * P)) +
      e * (β * ((β * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) + lam * αu) * (N2z + lam * ρ * z) -
        β * Nmu) / (lam * ρ * P) - β ^ 2 / (z - u)) := by
  have hYp : cb * Nu + ρ * Ru = β / lam * (lam * ρ * (z + u + 2 * a * ρ) * P -
      (u ^ 2 + 2 * a * ρ * u + q₁) * Nmu + e * N2z - lam * ρ * z * (P - e) - lam * W) := by
    rw [← hp]
    field_simp
  have hYm : cb * Nmu + ρ * Rmu = β / lam * (lam * ρ * (z - u + 2 * a * ρ) * P -
      ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) * Nu + e * N2z - lam * ρ * z * (P - e) - lam * W) := by
    rw [← hm]
    field_simp
  have hαmu : αmu = β ^ 2 * ((u ^ 2 + 2 * a * ρ * u + q₁) * ((-u) ^ 2 + 2 * a * ρ * (-u) + q₁) -
      e) / (lam ^ 2 * αu) := by
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

/-- Step 4 at a double root `e = (v² - u²)²`: the defect of Step 1 is removed by the constant
(Step 2) and by the images `-β(P(u)z + W(z; u²)/ρ)` of `Φ` and `-β(P'(u)z + 2uW₁/ρ)` of `∂ᵤΦ`
(Step 3), now by partial fractions in `s = v²` with a double pole (`P - e` and `W(z; ·)` affine). -/
theorem step4_conf {β lam ρ εa z vv u Pu Pv ev N2z cA cB W₀ W₁ : ℂ} (hβ : β ≠ 0)
    (hρ : ρ ≠ 0) (hεa : εa ≠ 0) (hv : vv - u ^ 2 ≠ 0) (hev : ev = (vv - u ^ 2) ^ 2)
    (hPv : Pv - ev = cA * vv + cB) (hPu : Pu = cA * u ^ 2 + cB) :
    (-N2z + lam * ρ * z * (Pv / ev - 1) + lam * (W₀ + W₁ * vv) / ev) / β +
        2 * εa / β ^ 2 * (β * N2z / (2 * εa)) +
      lam * ρ / β ^ 2 * (-β * (cA * z + W₁ / ρ) * (1 / (vv - u ^ 2)) +
        -β * (Pu * z + (W₀ + W₁ * u ^ 2) / ρ) * (1 / (vv - u ^ 2)) ^ 2) = 0 := by
  have hPv' : Pv = cA * vv + cB + (vv - u ^ 2) ^ 2 := by linear_combination hPv + hev
  subst hev hPv' hPu
  field_simp
  ring

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- The remainder `Ξ = β(K(u)(N₂(zₙ) + λρzₙ) - βN(zₙ, -u))/(λρP(u)) - β²/(zₙ - u)` of
`step3_general`. -/
noncomputable def Xi (n : ℤ) (u : ℂ) : ℂ :=
  A.β * (A.Kph k ε u * (A.N₂ k ε (A.z k n) + A.lam ε * A.ρ * A.z k n) -
      A.β * A.Nph k ε (A.z k n) (-u)) / (A.lam ε * A.ρ * A.Pph k u) - A.β ^ 2 / (A.z k n - u)

/-- Step 3 for every `u` off the lattice with `P(u) ≠ 0`, `α(u) ≠ 0`:
`∑ₗ P(zₙ)χ_r,nl φᵤ(zₗ) = -βK(u)[zₙ + W(zₙ; u²)/(ρP(u))] + e(u)Ξ`. -/
theorem hasSum_mode_gen (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) {u : ℂ}
    (hoff : ∀ l, A.z k l ≠ u) (hoff' : ∀ l, A.z k l ≠ -u) (hPu : A.Pph k u ≠ 0)
    (hαu : A.α k u ≠ 0) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l * A.mode k ε u (A.z k l))
      (-A.β * A.Kph k ε u * (A.z k n + A.Wph k ε (A.z k n) (u ^ 2) / (A.ρ * A.Pph k u)) +
        A.eph k ε u * A.Xi k ε n u) := by
  have hβ := A.β_ne_zero
  have hρ := A.rho_ne_zero hμ
  have hlam := A.lam_ne_zero ε hε ha
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
    have h7p := A.identity_A7 k ε hε ha (A.z k n) (-u)
    have h7m := A.identity_A7 k ε hε ha (A.z k n) u
    rw [neg_neg, A.Pph_neg, A.eph_neg, neg_sq] at h7p
    have hrel : (A.β : ℂ) ^ 2 * (A.Hph k ε u * A.Hph k ε (-u) - A.eph k ε u) =
        A.lam ε ^ 2 * (A.α k u * A.α k (-u)) := by
      rw [A.α_mul_α_neg k hoff hoff', eph]
      ring
    simp only [Kph, mode, Hph, Xi] at h7p h7m hrel ⊢
    have key := step3_general (a := A.a) (q₁ := A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁)
      (z := A.z k n) (Nu := A.Nph k ε (A.z k n) u) (Nmu := A.Nph k ε (A.z k n) (-u))
      (Ru := A.Rhat k ε (A.z k n) u) (Rmu := A.Rhat k ε (A.z k n) (-u)) (cb := A.cbar)
      (W := A.Wph k ε (A.z k n) (u ^ 2)) (N2z := A.N₂ k ε (A.z k n)) (e := A.eph k ε u) hβ hlam hρ
      hPu hzu hzu' hαu hrel (by linear_combination -h7p) (by linear_combination -h7m)
    linear_combination -key

/-! ### The mode at a double root and its derivative -/

theorem hasDerivAt_Pph (u : ℂ) : HasDerivAt (A.Pph k) (4 * u ^ 3 + 2 * A.κ k * u) u := by
  have h : A.Pph k = fun t ↦ t ^ 4 + A.κ k * t ^ 2 + A.q₀ k ^ 2 := funext (A.Pph_eq_poly k)
  rw [h]
  have := (((hasDerivAt_pow 4 u).add ((hasDerivAt_pow 2 u).const_mul (A.κ k))).add_const
    (A.q₀ k ^ 2))
  convert this using 1
  push_cast
  ring

theorem hasDerivAt_Hph_neg (u : ℂ) :
    HasDerivAt (fun t ↦ A.Hph k ε (-t)) (2 * u - 2 * A.a * A.ρ) u := by
  have h : (fun t ↦ A.Hph k ε (-t)) =
      fun t ↦ t ^ 2 - 2 * A.a * A.ρ * t + (A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁) := by
    funext t
    rw [Hph]
    ring
  rw [h]
  refine (((hasDerivAt_pow 2 u).fun_sub ((hasDerivAt_id' u).const_mul
    (2 * (A.a : ℂ) * A.ρ))).add_const (A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁)).congr_deriv ?_
  norm_num

theorem hasDerivAt_α {u : ℂ} (hoff : ∀ l, A.z k l ≠ u) :
    HasDerivAt (A.α k) (A.ρ * (A.S k u ^ 2 - A.β ^ 2 / 4)) u := by
  have h : A.α k = fun t ↦ A.ρ * A.S k t - A.cbar := by
    funext t
    rw [α]
  rw [h]
  exact ((A.hasDerivAt_S hoff).const_mul (A.ρ : ℂ)).sub_const _

/-- `f(u) = P(u)[βH(-u) - λα(u)]/K(u)`, so that `Z(u) = uf(u)`. -/
noncomputable def fZ (u : ℂ) : ℂ :=
  A.Pph k u * (A.β * A.Hph k ε (-u) - A.lam ε * A.α k u) / A.Kph k ε u

/-- `f'(u)` (`hasDerivAt_fZ`). -/
noncomputable def fZd (u : ℂ) : ℂ :=
  (4 * u ^ 3 + 2 * A.κ k * u) * (A.β * A.Hph k ε (-u) - A.lam ε * A.α k u) / A.Kph k ε u +
    2 * A.lam ε * A.Pph k u * (A.α k u * (A.β * (2 * u - 2 * A.a * A.ρ)) -
      A.ρ * (A.S k u ^ 2 - A.β ^ 2 / 4) * (A.β * A.Hph k ε (-u))) / A.Kph k ε u ^ 2

/-- `Z'(u) = f(u) + uf'(u)` (`hasDerivAt_Zph`). -/
noncomputable def Zder (u : ℂ) : ℂ := A.fZ k ε u + u * A.fZd k ε u

theorem Zph_eq (u : ℂ) : A.Zph k ε u = u * A.fZ k ε u := by
  rw [Zph, fZ]
  ring

theorem hasDerivAt_fZ {u : ℂ} (hoff : ∀ l, A.z k l ≠ u) (hK : A.Kph k ε u ≠ 0) :
    HasDerivAt (A.fZ k ε) (A.fZd k ε u) u := by
  have hH := A.hasDerivAt_Hph_neg k ε u
  have hα := A.hasDerivAt_α k hoff
  have hY := (hH.const_mul (A.β : ℂ)).fun_sub (hα.const_mul (A.lam ε))
  have hKd := (hH.const_mul (A.β : ℂ)).fun_add (hα.const_mul (A.lam ε))
  have hZ := ((A.hasDerivAt_Pph k u).fun_mul hY).fun_div hKd hK
  have hfun : A.fZ k ε = fun t ↦ A.Pph k t *
      ((A.β : ℂ) * A.Hph k ε (-t) - A.lam ε * A.α k t) /
      ((A.β : ℂ) * A.Hph k ε (-t) + A.lam ε * A.α k t) := by
    funext t
    rw [fZ, Kph]
  rw [hfun]
  refine hZ.congr_deriv ?_
  rw [fZd, Kph] at *
  field_simp
  ring

theorem hasDerivAt_Zph {u : ℂ} (hoff : ∀ l, A.z k l ≠ u) (hK : A.Kph k ε u ≠ 0) :
    HasDerivAt (A.Zph k ε) (A.Zder k ε u) u := by
  have h : A.Zph k ε = fun t ↦ t * A.fZ k ε t := funext (A.Zph_eq k ε)
  rw [h]
  refine ((hasDerivAt_id' u).fun_mul (A.hasDerivAt_fZ k ε hoff hK)).congr_deriv ?_
  rw [Zder]
  ring

/-- `Φ(u, t) = (P(u)t + Z(u))/(t² - u²) = P(u)φᵤ(t)/K(u)` (`modeΦ_eq`). -/
noncomputable def modeΦ (u t : ℂ) : ℂ := (A.Pph k u * t + A.Zph k ε u) / (t ^ 2 - u ^ 2)

/-- `∂ᵤΦ(u, t)`. -/
noncomputable def dmodeΦ (u t : ℂ) : ℂ :=
  ((4 * u ^ 3 + 2 * A.κ k * u) * t + A.Zder k ε u) / (t ^ 2 - u ^ 2) +
    2 * u * (A.Pph k u * t + A.Zph k ε u) / (t ^ 2 - u ^ 2) ^ 2

theorem modeΦ_eq {u t : ℂ} (hK : A.Kph k ε u ≠ 0) (htu : t - u ≠ 0) (htu' : t + u ≠ 0) :
    A.modeΦ k ε u t = A.Pph k u * A.mode k ε u t / A.Kph k ε u := by
  have ht : t ^ 2 - u ^ 2 ≠ 0 := by
    rw [show t ^ 2 - u ^ 2 = (t - u) * (t + u) by ring]
    exact mul_ne_zero htu htu'
  rw [Kph] at hK
  simp only [modeΦ, Zph, mode, Kph]
  field_simp
  ring

theorem hasDerivAt_modeΦ {u t : ℂ} (hoff : ∀ l, A.z k l ≠ u) (hK : A.Kph k ε u ≠ 0)
    (ht : t ^ 2 - u ^ 2 ≠ 0) :
    HasDerivAt (fun v ↦ A.modeΦ k ε v t) (A.dmodeΦ k ε u t) u := by
  have hnum := ((A.hasDerivAt_Pph k u).mul_const t).fun_add (A.hasDerivAt_Zph k ε hoff hK)
  have hden := (hasDerivAt_pow 2 u).const_sub (t ^ 2)
  refine (hnum.fun_div hden ht).congr_deriv ?_
  rw [dmodeΦ]
  push_cast
  field_simp
  ring

/-! ### The image of the mode and of its derivative -/

/-- `Jgen(u) = -β(P(u)zₙ + W(zₙ; u²)/ρ)`: the image of `Φ(u, ·)` at a root `u` of `e`. -/
noncomputable def Jgen (n : ℤ) (u : ℂ) : ℂ :=
  -A.β * (A.Pph k u * A.z k n + A.Wph k ε (A.z k n) (u ^ 2) / A.ρ)

/-- `∑ₗ P(zₙ)χ_r,nl/(zₗ - x)` converges absolutely for `x` off the lattice, `x ≠ -zₙ`,
`P(x) ≠ 0`. -/
theorem summable_Rrow_div (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) {n : ℤ} {x : ℂ}
    (hx : ∀ l, A.z k l ≠ x)
    (hxn : x ≠ -A.z k n) (hPx : A.Pph k x ≠ 0) :
    Summable (fun l ↦ A.Rrow k ε n l * (1 / (A.z k l - x))) := by
  have h₁ : Summable (fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l * (1 / (A.z k l - x))) :=
    (hasSum_single (f := fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l * (1 / (A.z k l - x))) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_mul])).summable
  have h₂ := ((A.hasSum_cauchy k hμ n hx hxn).mul_left (A.lam ε)).summable
  have h₃ := (A.hasSum_rank_two k ε ha (A.z k n) hx hPx).summable
  convert (h₁.add h₂).add h₃ using 1
  funext l
  have h₄ := sub_ne_zero.mpr (hx l)
  have h₅ := A.z_add_z_ne_zero k hμ n l
  have h₆ := A.Pph_z_ne_zero k l
  simp only [Rrow]
  field_simp

/-- The image of `Φ(u, ·)` for `u` near a root: `Jgen(u) + e(u)P(u)Ξ/K(u)`. -/
theorem hasSum_modeΦ_gen (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) {u : ℂ}
    (hoff : ∀ l, A.z k l ≠ u) (hoff' : ∀ l, A.z k l ≠ -u) (hPu : A.Pph k u ≠ 0)
    (hαu : A.α k u ≠ 0) (hK : A.Kph k ε u ≠ 0) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l * A.modeΦ k ε u (A.z k l))
      (A.Jgen k ε n u + A.eph k ε u * (A.Pph k u / A.Kph k ε u * A.Xi k ε n u)) := by
  have hρ := A.rho_ne_zero hμ
  convert (A.hasSum_mode_gen k ε hε ha hμ hoff hoff' hPu hαu n).mul_left
    (A.Pph k u / A.Kph k ε u) using 1
  · funext l
    rw [A.modeΦ_eq k ε hK (sub_ne_zero.mpr (hoff l))
      (fun h0 ↦ hoff' l (by linear_combination h0))]
    ring
  · rw [Jgen]
    field_simp

/-- The assumptions of Theorem "density and magnetic" at a double root `s₁ = s₂ = u²` (with
`K(u) ≠ 0` and `P(u) ≠ 0`). -/
structure DMConf (u : ℂ) : Prop where
  ε_eq : ε = 1 ∨ ε = -1
  a_ne : (A.a : ℂ) ≠ 0
  μ_ne : A.μ ≠ 0
  e_ne : ∀ n, A.eph k ε (A.z k n) ≠ 0
  e_eq : ∀ t, A.eph k ε t = (t ^ 2 - u ^ 2) ^ 2
  K_ne : A.Kph k ε u ≠ 0
  P_ne : A.Pph k u ≠ 0

variable {A k ε} {u : ℂ}

theorem DMConf.z_ne (h : A.DMConf k ε u) (n : ℤ) : A.z k n ≠ u := fun h0 ↦
  h.e_ne n (by rw [h.e_eq, h0, sub_self, zero_pow two_ne_zero])

theorem DMConf.z_ne_neg (h : A.DMConf k ε u) (n : ℤ) : A.z k n ≠ -u := fun h0 ↦
  h.e_ne n (by rw [h.e_eq, h0, neg_sq, sub_self, zero_pow two_ne_zero])

theorem DMConf.z_sq_ne (h : A.DMConf k ε u) (n : ℤ) : A.z k n ^ 2 - u ^ 2 ≠ 0 := fun h0 ↦
  h.e_ne n (by rw [h.e_eq, h0, zero_pow two_ne_zero])

theorem DMConf.α_ne (h : A.DMConf k ε u) : A.α k u ≠ 0 := by
  intro h0
  have := A.α_mul_α_neg k h.z_ne h.z_ne_neg
  rw [h0, zero_mul] at this
  have hp : (A.β : ℂ) ^ 2 * A.p₀ * A.p₂ ≠ 0 := by
    have := A.β_ne_zero
    have := ofReal_ne_zero.mpr A.p₀_pos.ne'
    have := ofReal_ne_zero.mpr A.p₂_pos.ne'
    positivity
  exact hp this.symm

theorem DMConf.e_u (h : A.DMConf k ε u) : A.eph k ε u = 0 := by
  rw [h.e_eq, sub_self, zero_pow two_ne_zero]

/-- Step 3 at the double root: the image of `Φ(u, ·)` is `Jgen(u)`. -/
theorem DMConf.hasSum_modeΦ (h : A.DMConf k ε u) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l * A.modeΦ k ε u (A.z k l)) (A.Jgen k ε n u) := by
  have := A.hasSum_modeΦ_gen k ε h.ε_eq h.a_ne h.μ_ne h.z_ne h.z_ne_neg h.P_ne h.α_ne h.K_ne n
  rwa [h.e_u, zero_mul, add_zero] at this

variable (A k ε) in
theorem continuousAt_Kph {u : ℂ} (hoff : ∀ l, A.z k l ≠ u) : ContinuousAt (A.Kph k ε) u := by
  have h : A.Kph k ε = fun t ↦ A.β * A.Hph k ε (-t) + A.lam ε * A.α k t := by
    funext t
    rw [Kph]
  rw [h]
  exact (((A.hasDerivAt_Hph_neg k ε u).continuousAt).const_mul _).add
    (((A.hasDerivAt_α k hoff).continuousAt).const_mul _)

variable (A k ε) in
theorem continuousAt_Xi {n : ℤ} {u : ℂ} (hoff : ∀ l, A.z k l ≠ u) (hP : A.Pph k u ≠ 0) :
    ContinuousAt (A.Xi k ε n) u := by
  have hcN : ContinuousAt (fun v ↦ A.Nph k ε (A.z k n) (-v)) u := by
    have h : (fun v ↦ A.Nph k ε (A.z k n) (-v)) = fun v ↦ A.cN₀ k ε (A.z k n) +
        A.cN₁ k ε (A.z k n) * (-v) + A.cN₂ k ε (A.z k n) * (-v) ^ 2 + A.cN₃ ε * (-v) ^ 3 := by
      funext v
      rw [A.Nph_eq_coeffs]
    rw [h]
    fun_prop
  have hcP := (A.hasDerivAt_Pph k u).continuousAt
  have hcK := A.continuousAt_Kph k ε hoff
  have hzu : A.z k n - u ≠ 0 := sub_ne_zero.mpr (hoff n)
  have h : A.Xi k ε n = fun v ↦ A.β * (A.Kph k ε v * (A.N₂ k ε (A.z k n) +
      A.lam ε * A.ρ * A.z k n) - A.β * A.Nph k ε (A.z k n) (-v)) /
      (A.lam ε * A.ρ * A.Pph k v) - A.β ^ 2 / (A.z k n - v) := by
    funext v
    rw [Xi]
  rw [h]
  by_cases hlr : A.lam ε * A.ρ = 0
  · simp only [hlr, zero_mul, div_zero, zero_sub]
    exact (continuousAt_const.div (continuousAt_const.sub continuousAt_id) hzu).neg
  · refine ContinuousAt.sub ?_ (continuousAt_const.div (continuousAt_const.sub continuousAt_id) hzu)
    exact ((continuousAt_const.mul ((hcK.mul continuousAt_const).sub
      (continuousAt_const.mul hcN))).div (continuousAt_const.mul hcP) (mul_ne_zero hlr hP))

/-- Step 3 for the derivative: at the double root, `∑ₗ P(zₙ)χ_r,nl ∂ᵤΦ(u, zₗ) = Jgen'(u)
= -β(P'(u)zₙ + 2uW₁/ρ)`, `W₁ = W(zₙ; 1) - W(zₙ; 0)`. The remainder `e(u)P(u)Ξ/K(u)` of
`hasSum_modeΦ_gen` has a double zero at `u`, and the series is differentiated termwise. -/
theorem DMConf.hasSum_dmodeΦ (h : A.DMConf k ε u) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l * A.dmodeΦ k ε u (A.z k l))
      (-A.β * ((4 * u ^ 3 + 2 * A.κ k * u) * A.z k n +
        2 * u * (A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0) / A.ρ)) := by
  have hμ := h.μ_ne
  have ha := h.a_ne
  have hρ := A.rho_ne_zero hμ
  obtain ⟨r₁, hr₁, hd₁⟩ := A.exists_lattice_dist h.z_ne
  obtain ⟨r₂, hr₂, hd₂⟩ := A.exists_lattice_dist h.z_ne_neg
  -- the coefficients of `Φ = c₁/(t - v) + c₂/(t + v)`
  set c₁ : ℂ → ℂ := fun v ↦ A.Pph k v * (A.β * A.Hph k ε (-v)) / A.Kph k ε v with hc₁
  set c₂ : ℂ → ℂ := fun v ↦ A.Pph k v * (A.lam ε * A.α k v) / A.Kph k ε v with hc₂
  have hcP := (A.hasDerivAt_Pph k u).continuousAt
  have hcα := (A.hasDerivAt_α k h.z_ne).continuousAt
  have hcK := A.continuousAt_Kph k ε h.z_ne
  have hcc₁ : ContinuousAt c₁ u :=
    (hcP.mul (((A.hasDerivAt_Hph_neg k ε u).continuousAt).const_mul _)).div hcK h.K_ne
  have hcc₂ : ContinuousAt c₂ u := (hcP.mul (hcα.const_mul _)).div hcK h.K_ne
  have hev : ∀ᶠ v in 𝓝 u, A.Pph k v ≠ 0 ∧ A.α k v ≠ 0 ∧ A.Kph k ε v ≠ 0 ∧
      ‖c₁ v‖ < ‖c₁ u‖ + 1 ∧ ‖c₂ v‖ < ‖c₂ u‖ + 1 :=
    (hcP.eventually_ne h.P_ne).and ((hcα.eventually_ne h.α_ne).and ((hcK.eventually_ne h.K_ne).and
      ((hcc₁.norm.eventually_lt continuousAt_const (lt_add_one _)).and
        (hcc₂.norm.eventually_lt continuousAt_const (lt_add_one _)))))
  obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.mp hev
  set r := min δ (min (r₁ / 2) (r₂ / 2)) with hr_def
  have hr : 0 < r := lt_min hδ (lt_min (half_pos hr₁) (half_pos hr₂))
  -- the lattice stays away from `±v` on the ball
  have hfar₁ : ∀ v ∈ ball u r, ∀ l, ‖A.z k l - u‖ ≤ 2 * ‖A.z k l - v‖ := by
    intro v hv l
    have h₁ : ‖v - u‖ < r₁ / 2 := (mem_ball_iff_norm.mp hv).trans_le
      ((min_le_right _ _).trans (min_le_left _ _))
    have h₂ : ‖A.z k l - u‖ ≤ ‖A.z k l - v‖ + ‖v - u‖ := by
      rw [show A.z k l - u = (A.z k l - v) + (v - u) by ring]
      exact norm_add_le _ _
    linarith [hd₁ l]
  have hfar₂ : ∀ v ∈ ball u r, ∀ l, ‖A.z k l - -u‖ ≤ 2 * ‖A.z k l - -v‖ := by
    intro v hv l
    have h₁ : ‖v - u‖ < r₂ / 2 := (mem_ball_iff_norm.mp hv).trans_le
      ((min_le_right _ _).trans (min_le_right _ _))
    have h₂ : ‖A.z k l - -u‖ ≤ ‖A.z k l - -v‖ + ‖v - u‖ := by
      rw [show A.z k l - -u = (A.z k l - -v) + (u - v) by ring, ← norm_neg (v - u), neg_sub]
      exact norm_add_le _ _
    linarith [hd₂ l]
  have hoffv : ∀ v ∈ ball u r, ∀ l, A.z k l ≠ v := by
    intro v hv l h0
    have h₁ := hfar₁ v hv l
    have h₂ := hd₁ l
    rw [h0, sub_self, norm_zero, mul_zero] at h₁
    rw [h0] at h₂
    linarith
  have hoffv' : ∀ v ∈ ball u r, ∀ l, A.z k l ≠ -v := by
    intro v hv l h0
    have h₁ := hfar₂ v hv l
    have h₂ := hd₂ l
    rw [h0, sub_self, norm_zero, mul_zero] at h₁
    rw [h0] at h₂
    linarith
  have hgen : ∀ v ∈ ball u r, A.Pph k v ≠ 0 ∧ A.α k v ≠ 0 ∧ A.Kph k ε v ≠ 0 ∧
      ‖c₁ v‖ < ‖c₁ u‖ + 1 ∧ ‖c₂ v‖ < ‖c₂ u‖ + 1 := fun v hv ↦
    hball ((mem_ball.mp hv).trans_le (min_le_left _ _))
  -- the summable bound
  have hs₁ := summable_norm_iff.mpr (A.summable_Rrow_div k ε ha hμ (n := n) h.z_ne
    (fun h0 ↦ h.z_ne_neg n (by rw [h0, neg_neg])) h.P_ne)
  have hs₂ := summable_norm_iff.mpr (A.summable_Rrow_div k ε ha hμ (n := n) h.z_ne_neg
    (fun h0 ↦ h.z_ne n (neg_inj.mp h0).symm) (by rw [A.Pph_neg]; exact h.P_ne))
  set b : ℤ → ℝ := fun l ↦ 2 * (‖c₁ u‖ + 1) * ‖A.Rrow k ε n l * (1 / (A.z k l - u))‖ +
    2 * (‖c₂ u‖ + 1) * ‖A.Rrow k ε n l * (1 / (A.z k l - -u))‖ with hb_def
  have hb : Summable b := (hs₁.mul_left _).add (hs₂.mul_left _)
  -- the closed form near `u`
  set G : ℂ → ℂ := fun v ↦ A.Jgen k ε n v +
    (v - u) ^ 2 * ((v + u) ^ 2 * (A.Pph k v / A.Kph k ε v * A.Xi k ε n v)) with hG_def
  have hGeq : ∀ v, A.Jgen k ε n v + A.eph k ε v * (A.Pph k v / A.Kph k ε v * A.Xi k ε n v) =
      G v := by
    intro v
    rw [hG_def, h.e_eq v]
    ring
  refine hasSum_deriv_of_bound (f := fun l v ↦ A.Rrow k ε n l * A.modeΦ k ε v (A.z k l))
    (G := G) (u₀ := u) (r := r) hr hb ?_ ?_ ?_ ?_ ?_
  · -- differentiability on the ball
    intro l v hv
    obtain ⟨-, -, hKv, -, -⟩ := hgen v hv
    have ht : A.z k l ^ 2 - v ^ 2 ≠ 0 := by
      rw [show A.z k l ^ 2 - v ^ 2 = (A.z k l - v) * (A.z k l + v) by ring]
      exact mul_ne_zero (sub_ne_zero.mpr (hoffv v hv l))
        (fun h0 ↦ hoffv' v hv l (by linear_combination h0))
    exact (((A.hasDerivAt_modeΦ k ε (hoffv v hv) hKv ht).const_mul _).differentiableAt
      ).differentiableWithinAt
  · -- the derivative at `u`
    intro l
    exact (A.hasDerivAt_modeΦ k ε h.z_ne h.K_ne (h.z_sq_ne l)).const_mul _
  · -- the bound
    intro l v hv
    obtain ⟨-, -, hKv, hn₁, hn₂⟩ := hgen v hv
    have h₁ := sub_ne_zero.mpr (hoffv v hv l)
    have h₂ : A.z k l + v ≠ 0 := fun h0 ↦ hoffv' v hv l (by linear_combination h0)
    have hΦ : A.modeΦ k ε v (A.z k l) =
        c₁ v * (1 / (A.z k l - v)) + c₂ v * (1 / (A.z k l - -v)) := by
      rw [A.modeΦ_eq k ε hKv h₁ h₂, hc₁, hc₂]
      simp only [mode, sub_neg_eq_add]
      field_simp
    have hu₁ : ‖1 / (A.z k l - v)‖ ≤ 2 * ‖1 / (A.z k l - u)‖ := by
      rw [norm_div, norm_div, norm_one, div_le_iff₀ (norm_pos_iff.mpr h₁)]
      have := hfar₁ v hv l
      rw [show 2 * (1 / ‖A.z k l - u‖) * ‖A.z k l - v‖ = 2 * ‖A.z k l - v‖ / ‖A.z k l - u‖ by ring,
        le_div_iff₀ (norm_pos_iff.mpr (sub_ne_zero.mpr (h.z_ne l)))]
      linarith
    have hu₂ : ‖1 / (A.z k l - -v)‖ ≤ 2 * ‖1 / (A.z k l - -u)‖ := by
      have h₂' : A.z k l - -v ≠ 0 := by rwa [sub_neg_eq_add]
      have h₃ : A.z k l - -u ≠ 0 := sub_ne_zero.mpr (h.z_ne_neg l)
      rw [norm_div, norm_div, norm_one, div_le_iff₀ (norm_pos_iff.mpr h₂')]
      have := hfar₂ v hv l
      rw [show 2 * (1 / ‖A.z k l - -u‖) * ‖A.z k l - -v‖ =
        2 * ‖A.z k l - -v‖ / ‖A.z k l - -u‖ by ring, le_div_iff₀ (norm_pos_iff.mpr h₃)]
      linarith
    rw [hΦ, mul_add, hb_def]
    refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
    · rw [show A.Rrow k ε n l * (c₁ v * (1 / (A.z k l - v))) =
        c₁ v * (A.Rrow k ε n l * (1 / (A.z k l - v))) by ring, norm_mul, norm_mul, norm_mul]
      have := norm_nonneg (A.Rrow k ε n l)
      calc ‖c₁ v‖ * (‖A.Rrow k ε n l‖ * ‖1 / (A.z k l - v)‖)
          ≤ (‖c₁ u‖ + 1) * (‖A.Rrow k ε n l‖ * (2 * ‖1 / (A.z k l - u)‖)) := by
            gcongr
        _ = _ := by ring
    · rw [show A.Rrow k ε n l * (c₂ v * (1 / (A.z k l - -v))) =
        c₂ v * (A.Rrow k ε n l * (1 / (A.z k l - -v))) by ring, norm_mul, norm_mul, norm_mul]
      have := norm_nonneg (A.Rrow k ε n l)
      calc ‖c₂ v‖ * (‖A.Rrow k ε n l‖ * ‖1 / (A.z k l - -v)‖)
          ≤ (‖c₂ u‖ + 1) * (‖A.Rrow k ε n l‖ * (2 * ‖1 / (A.z k l - -u)‖)) := by
            gcongr
        _ = _ := by ring
  · -- the sums on the ball
    intro v hv
    obtain ⟨hPv, hαv, hKv, -, -⟩ := hgen v hv
    rw [← hGeq]
    exact A.hasSum_modeΦ_gen k ε h.ε_eq ha hμ (hoffv v hv) (hoffv' v hv) hPv hαv hKv n
  · -- the derivative of the closed form: the remainder has a double zero
    have hJ : HasDerivAt (A.Jgen k ε n) (-A.β * ((4 * u ^ 3 + 2 * A.κ k * u) * A.z k n +
        2 * u * (A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0) / A.ρ)) u := by
      have hJ' : A.Jgen k ε n = fun v ↦ -A.β * (A.Pph k v * A.z k n +
          (A.Wph k ε (A.z k n) 0 + (A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0) * v ^ 2) /
            A.ρ) := by
        funext v
        rw [Jgen, A.Wph_affine k ε _ (v ^ 2)]
      rw [hJ']
      refine ((((A.hasDerivAt_Pph k u).mul_const (A.z k n)).fun_add
        ((((hasDerivAt_pow 2 u).const_mul (A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0)
          ).const_add (A.Wph k ε (A.z k n) 0)).div_const (A.ρ : ℂ))).const_mul
            (-(A.β : ℂ))).congr_deriv ?_
      push_cast
      field_simp
    have hR : ContinuousAt (fun v ↦ (v + u) ^ 2 * (A.Pph k v / A.Kph k ε v * A.Xi k ε n v)) u :=
      ((continuousAt_id.add continuousAt_const).pow 2).mul
        ((hcP.div hcK h.K_ne).mul (A.continuousAt_Xi k ε h.z_ne h.P_ne))
    have := hJ.add (hasDerivAt_sq_mul hR)
    rw [add_zero] at this
    exact this

variable (A k ε) in
/-- `∂ₛZ(s)` at `s = u²`, `Z(s)` the even part of `Z(u)`: `Z'(u)/(2u)`, and `f'(0)` at `u = 0`. -/
noncomputable def Zsd (u : ℂ) : ℂ := if u = 0 then A.fZd k ε 0 else A.Zder k ε u / (2 * u)

variable (A k ε) in
/-- `M₁(t) = ∂ₛ[(P̃(s)t + Z(s))/(t² - s)]` at `s = u²`, with `P̃(s) = s² + κs + q₀²`. -/
noncomputable def M₁ (u t : ℂ) : ℂ :=
  ((A.κ k + 2 * u ^ 2) * t + A.Zsd k ε u) / (t ^ 2 - u ^ 2) +
    (A.Pph k u * t + A.Zph k ε u) / (t ^ 2 - u ^ 2) ^ 2

/-- `M₁ = ∂ᵤΦ/(2u)` for `u ≠ 0`. -/
theorem M₁_eq {t : ℂ} (hu : u ≠ 0) (ht : t ^ 2 - u ^ 2 ≠ 0) :
    A.M₁ k ε u t = A.dmodeΦ k ε u t / (2 * u) := by
  rw [M₁, dmodeΦ, Zsd, ite_eq_right hu]
  field_simp
  ring

/-- A bound for the difference quotients at `u = 0`: `‖t² - v²‖ ≥ 3‖t‖²/4` for `‖v‖ ≤ ‖t‖/2`. -/
theorem norm_sq_sub_sq_ge {t v : ℂ} (hv : 2 * ‖v‖ ≤ ‖t‖) : 3 / 4 * ‖t‖ ^ 2 ≤ ‖t ^ 2 - v ^ 2‖ := by
  have h₁ : ‖t ^ 2‖ - ‖v ^ 2‖ ≤ ‖t ^ 2 - v ^ 2‖ := norm_sub_norm_le _ _
  rw [norm_pow, norm_pow] at h₁
  nlinarith [norm_nonneg v]

/-- At `u = 0`: `∑ₗ P(zₙ)χ_r,nl M₁(zₗ) = -β(κzₙ + W₁/ρ)`. `M₁(t)` is the limit `v → 0` of the
difference quotients `[E(v, t) - E(0, t)]/v²` of the even part `E(v, t) = [Φ(v, t) + Φ(-v, t)]/2`,
whose images follow from `hasSum_modeΦ_gen` at `±v`; the limit is taken by Tannery's theorem. -/
theorem DMConf.hasSum_M₁_zero (h : A.DMConf k ε 0) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l * A.M₁ k ε 0 (A.z k l))
      (-A.β * (A.κ k * A.z k n + (A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0) / A.ρ)) := by
  have hμ := h.μ_ne
  have ha := h.a_ne
  have hβ := A.β_ne_zero
  have hρ := A.rho_ne_zero hμ
  have hz0 : ∀ l, A.z k l ≠ 0 := h.z_ne
  obtain ⟨d, hd, hdl⟩ := A.exists_lattice_dist hz0
  simp only [sub_zero] at hdl
  -- continuity at `0`
  have hcP := (A.hasDerivAt_Pph k 0).continuousAt
  have hcα := (A.hasDerivAt_α k hz0).continuousAt
  have hcK := A.continuousAt_Kph k ε hz0
  have hcαn : ContinuousAt (fun v ↦ A.α k (-v)) 0 := by
    have hc : ContinuousAt (A.α k) (-0) := by rwa [neg_zero]
    exact hc.comp continuousAt_neg
  have hcKn : ContinuousAt (fun v ↦ A.Kph k ε (-v)) 0 := by
    have hc : ContinuousAt (A.Kph k ε) (-0) := by rwa [neg_zero]
    exact hc.comp continuousAt_neg
  have hα0 : A.α k 0 ≠ 0 := h.α_ne
  have hαn0 : A.α k (-0) ≠ 0 := by rwa [neg_zero]
  have hKn0 : A.Kph k ε (-0) ≠ 0 := by rw [neg_zero]; exact h.K_ne
  set hR : ℂ → ℂ := fun w ↦ A.Pph k w / A.Kph k ε w * A.Xi k ε n w with hR_def
  have hcR : ContinuousAt hR 0 := (hcP.div hcK h.K_ne).mul (A.continuousAt_Xi k ε hz0 h.P_ne)
  have hcRn : ContinuousAt (fun v ↦ hR (-v)) 0 := by
    have hc : ContinuousAt hR (-0) := by rwa [neg_zero]
    exact hc.comp continuousAt_neg
  -- the symmetric slopes of `f`
  have hneg : Tendsto (fun v : ℂ ↦ -v) (𝓝[≠] 0) (𝓝[≠] 0) := by
    refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ ?_ ?_
    · have := (continuous_neg (G := ℂ)).tendsto 0
      rw [neg_zero] at this
      exact this.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with v hv
      exact neg_ne_zero.mpr hv
  have hslope := (hasDerivAt_iff_tendsto_slope.mp (A.hasDerivAt_fZ k ε hz0 h.K_ne))
  set Q : ℂ → ℂ := fun v ↦ (A.fZ k ε v - A.fZ k ε (-v)) / (2 * v) with hQ_def
  have hQ : Tendsto Q (𝓝[≠] 0) (𝓝 (A.fZd k ε 0)) := by
    have h₂ := (hslope.add (hslope.comp hneg)).div_const 2
    rw [show (A.fZd k ε 0 + A.fZd k ε 0) / 2 = A.fZd k ε 0 by ring] at h₂
    refine h₂.congr' ?_
    filter_upwards [self_mem_nhdsWithin] with v hv
    have hv' : v ≠ 0 := hv
    have hv'' : -v ≠ 0 := neg_ne_zero.mpr hv'
    simp only [Function.comp, slope_def_field, sub_zero, hQ_def]
    field_simp
    ring
  -- eventually: the generic conditions at `±v`, and the bounds
  have hev : ∀ᶠ v in 𝓝[≠] (0 : ℂ), v ≠ 0 ∧ A.Pph k v ≠ 0 ∧ A.α k v ≠ 0 ∧ A.α k (-v) ≠ 0 ∧
      A.Kph k ε v ≠ 0 ∧ A.Kph k ε (-v) ≠ 0 ∧ 2 * ‖v‖ < d ∧ ‖Q v‖ < ‖A.fZd k ε 0‖ + 1 := by
    have hsmall : ∀ᶠ v in 𝓝 (0 : ℂ), 2 * ‖v‖ < d := by
      have : ContinuousAt (fun v : ℂ ↦ 2 * ‖v‖) 0 := by fun_prop
      exact this.eventually_lt continuousAt_const (by simpa using hd)
    filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds (hcP.eventually_ne h.P_ne),
      nhdsWithin_le_nhds (hcα.eventually_ne hα0), nhdsWithin_le_nhds (hcαn.eventually_ne hαn0),
      nhdsWithin_le_nhds (hcK.eventually_ne h.K_ne), nhdsWithin_le_nhds (hcKn.eventually_ne hKn0),
      nhdsWithin_le_nhds hsmall, hQ.norm.eventually (gt_mem_nhds (lt_add_one _))]
      with v h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈
    exact ⟨h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈⟩
  -- the difference quotients and their images
  set D : ℂ → ℂ → ℂ := fun v t ↦ ((A.κ k + v ^ 2) * t + Q v) / (t ^ 2 - v ^ 2) +
    A.Pph k 0 / (t * (t ^ 2 - v ^ 2)) with hD_def
  set W₁ := A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0 with hW₁
  set V : ℂ → ℂ := fun v ↦ -A.β * ((A.κ k + v ^ 2) * A.z k n + W₁ / A.ρ) +
    v ^ 2 * (hR v + hR (-v)) / 2 with hV_def
  have hsum : ∀ᶠ v in 𝓝[≠] (0 : ℂ), HasSum (fun l ↦ A.Rrow k ε n l * D v (A.z k l)) (V v) := by
    filter_upwards [hev] with v hv'
    obtain ⟨hv, hPv, hαv, hαnv, hKv, hKnv, hvd, -⟩ := hv'
    have hoff : ∀ l, A.z k l ≠ v := by
      intro l h0
      have := hdl l
      rw [h0] at this
      linarith [norm_nonneg v]
    have hoffn : ∀ l, A.z k l ≠ -v := by
      intro l h0
      have := hdl l
      rw [h0, norm_neg] at this
      linarith [norm_nonneg v]
    have hPnv : A.Pph k (-v) ≠ 0 := by rwa [A.Pph_neg]
    have Sp := A.hasSum_modeΦ_gen k ε h.ε_eq ha hμ hoff hoffn hPv hαv hKv n
    have Sm := A.hasSum_modeΦ_gen k ε h.ε_eq ha hμ hoffn (by simpa using hoff) hPnv hαnv hKnv n
    have S0 := h.hasSum_modeΦ n
    convert (((Sp.add Sm).div_const 2).sub S0).div_const (v ^ 2) using 1
    · funext l
      have h₁ := sub_ne_zero.mpr (hoff l)
      have h₂ : A.z k l + v ≠ 0 := fun h0 ↦ hoffn l (by linear_combination h0)
      have h₃ : A.z k l ^ 2 - v ^ 2 ≠ 0 := by
        rw [show A.z k l ^ 2 - v ^ 2 = (A.z k l - v) * (A.z k l + v) by ring]
        exact mul_ne_zero h₁ h₂
      have h₄ := hz0 l
      simp only [hD_def, hQ_def, modeΦ, A.Zph_eq, A.Pph_neg, neg_sq]
      rw [show A.Pph k v = A.Pph k 0 + (A.κ k + v ^ 2) * v ^ 2 by
        rw [A.Pph_eq_poly, A.Pph_eq_poly]; ring]
      field_simp
      ring
    · simp only [hV_def, Jgen, hR_def, h.e_eq, neg_sq, A.Pph_neg]
      rw [A.Wph_affine k ε (A.z k n) (v ^ 2), A.Wph_affine k ε (A.z k n) (0 ^ 2)]
      rw [show A.Pph k v = A.Pph k 0 + (A.κ k + v ^ 2) * v ^ 2 by
        rw [A.Pph_eq_poly, A.Pph_eq_poly]; ring]
      field_simp
      ring
  -- the limits
  have hVlim : Tendsto V (𝓝[≠] 0) (𝓝 (-A.β * (A.κ k * A.z k n + W₁ / A.ρ))) := by
    have hc : ContinuousAt V 0 := by
      simp only [hV_def]
      exact ((continuousAt_const.mul (((continuousAt_const.add
        (continuousAt_id.pow 2)).mul continuousAt_const).add continuousAt_const)).add
        (((continuousAt_id.pow 2).mul (hcR.add hcRn)).div_const 2))
    have := hc.tendsto.mono_left (nhdsWithin_le_nhds (s := {0}ᶜ))
    simpa [hV_def] using this
  have hDlim : ∀ l, Tendsto (fun v ↦ A.Rrow k ε n l * D v (A.z k l)) (𝓝[≠] 0)
      (𝓝 (A.Rrow k ε n l * A.M₁ k ε 0 (A.z k l))) := by
    intro l
    set t := A.z k l
    have h₄ : t ≠ 0 := hz0 l
    set F : ℂ × ℂ → ℂ := fun p ↦ A.Rrow k ε n l *
      (((A.κ k + p.1 ^ 2) * t + p.2) / (t ^ 2 - p.1 ^ 2) + A.Pph k 0 / (t * (t ^ 2 - p.1 ^ 2)))
      with hF_def
    have hsq : t ^ 2 - ((0 : ℂ), A.fZd k ε 0).1 ^ 2 ≠ 0 := by
      simp only [zero_pow two_ne_zero, sub_zero]
      exact pow_ne_zero 2 h₄
    have hden : ContinuousAt (fun p : ℂ × ℂ ↦ t ^ 2 - p.1 ^ 2) (0, A.fZd k ε 0) :=
      continuousAt_const.sub (continuousAt_fst.pow 2)
    have hF : ContinuousAt F (0, A.fZd k ε 0) :=
      continuousAt_const.mul (((((continuousAt_const.add (continuousAt_fst.pow 2)).mul
        continuousAt_const).add continuousAt_snd).div hden hsq).add
        (continuousAt_const.div (continuousAt_const.mul hden) (mul_ne_zero h₄ hsq)))
    have hpair : Tendsto (fun v ↦ (v, Q v)) (𝓝[≠] 0) (𝓝 ((0 : ℂ), A.fZd k ε 0)) :=
      (tendsto_nhdsWithin_iff.mp tendsto_id |>.1).prodMk_nhds hQ
    have hval : F (0, A.fZd k ε 0) = A.Rrow k ε n l * A.M₁ k ε 0 t := by
      rw [hF_def, M₁, Zsd, ite_eq_left rfl, A.Zph_eq, zero_mul, add_zero]
      simp only [zero_pow two_ne_zero, sub_zero, add_zero, mul_zero]
      field_simp
    rw [← hval]
    exact (hF.tendsto.comp hpair).congr (fun v ↦ by simp only [Function.comp_apply, hF_def, hD_def])
  -- the bound
  have hs₀ := summable_norm_iff.mpr (A.summable_Rrow_div k ε ha hμ (n := n) hz0
    (fun h0 ↦ hz0 n (neg_eq_zero.mp h0.symm)) h.P_ne)
  set C : ℝ := 4 / 3 * (d ^ 2 + ‖A.κ k‖) + 4 / (3 * d) * (‖A.fZd k ε 0‖ + 1) +
    4 / (3 * d ^ 2) * ‖A.Pph k 0‖ with hC_def
  have hbound : ∀ᶠ v in 𝓝[≠] (0 : ℂ), ∀ l,
      ‖A.Rrow k ε n l * D v (A.z k l)‖ ≤ C * ‖A.Rrow k ε n l * (1 / (A.z k l - 0))‖ := by
    filter_upwards [hev] with v hv' l
    obtain ⟨-, -, -, -, -, -, hvd, hQv⟩ := hv'
    set t := A.z k l
    have ht := hdl l
    have ht0 : t ≠ 0 := hz0 l
    have htn : 0 < ‖t‖ := norm_pos_iff.mpr ht0
    have hv2 : 2 * ‖v‖ ≤ ‖t‖ := by linarith
    have hge := norm_sq_sub_sq_ge hv2
    have hpos : 0 < ‖t ^ 2 - v ^ 2‖ := lt_of_lt_of_le (by positivity) hge
    have htv : t ^ 2 - v ^ 2 ≠ 0 := norm_pos_iff.mp hpos
    -- `D(v, t) t = (κ + v²) t²/(t² - v²) + Q(v) t/(t² - v²) + P(0)/(t² - v²)`
    have hDt : D v t * t = (A.κ k + v ^ 2) * (t ^ 2 / (t ^ 2 - v ^ 2)) +
        Q v * (t / (t ^ 2 - v ^ 2)) + A.Pph k 0 * (1 / (t ^ 2 - v ^ 2)) := by
      simp only [hD_def]
      field_simp
    have hb₁ : ‖t ^ 2 / (t ^ 2 - v ^ 2)‖ ≤ 4 / 3 := by
      rw [norm_div, norm_pow, div_le_iff₀ hpos]
      linarith
    have hb₂ : ‖t / (t ^ 2 - v ^ 2)‖ ≤ 4 / (3 * d) := by
      rw [norm_div, div_le_iff₀ hpos, div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
      nlinarith
    have hb₃ : ‖1 / (t ^ 2 - v ^ 2)‖ ≤ 4 / (3 * d ^ 2) := by
      rw [norm_div, norm_one, div_le_iff₀ hpos, div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
      nlinarith [mul_le_mul ht ht hd.le (norm_nonneg t)]
    have hκv : ‖A.κ k + v ^ 2‖ ≤ d ^ 2 + ‖A.κ k‖ := by
      refine (norm_add_le _ _).trans ?_
      rw [norm_pow]
      nlinarith [norm_nonneg v]
    have hDt' : ‖D v t * t‖ ≤ C := by
      rw [hDt, hC_def]
      refine (norm_add_le _ _).trans (add_le_add ((norm_add_le _ _).trans (add_le_add ?_ ?_)) ?_)
      · rw [norm_mul]
        calc _ ≤ (d ^ 2 + ‖A.κ k‖) * (4 / 3) :=
              mul_le_mul hκv hb₁ (norm_nonneg _) (by positivity)
          _ = _ := by ring
      · rw [norm_mul]
        calc _ ≤ (‖A.fZd k ε 0‖ + 1) * (4 / (3 * d)) :=
              mul_le_mul hQv.le hb₂ (norm_nonneg _) (by positivity)
          _ = _ := by ring
      · rw [norm_mul]
        calc _ ≤ ‖A.Pph k 0‖ * (4 / (3 * d ^ 2)) :=
              mul_le_mul le_rfl hb₃ (norm_nonneg _) (norm_nonneg _)
          _ = _ := by ring
    rw [show A.Rrow k ε n l * D v t = A.Rrow k ε n l * (1 / (t - 0)) * (D v t * t) by
      rw [sub_zero]; field_simp, norm_mul, mul_comm C]
    exact mul_le_mul_of_nonneg_left hDt' (norm_nonneg _)
  -- Tannery's theorem
  have hT := tendsto_tsum_of_dominated_convergence (hs₀.mul_left C) hDlim hbound
  have hT' : Tendsto (fun v ↦ ∑' l, A.Rrow k ε n l * D v (A.z k l)) (𝓝[≠] 0)
      (𝓝 (-A.β * (A.κ k * A.z k n + W₁ / A.ρ))) := by
    refine hVlim.congr' ?_
    filter_upwards [hsum] with v hv
    exact hv.tsum_eq.symm
  have heq := tendsto_nhds_unique hT hT'
  -- summability of the limit
  have hsM : Summable (fun l ↦ A.Rrow k ε n l * A.M₁ k ε 0 (A.z k l)) := by
    refine Summable.of_norm_bounded (hs₀.mul_left C) (fun l ↦ ?_)
    exact le_of_tendsto (tendsto_norm.comp (hDlim l))
      (hbound.mono fun v hv ↦ hv l)
  rw [← heq]
  exact hsM.hasSum

/-- Step 3 for `M₁`: `∑ₗ P(zₙ)χ_r,nl M₁(zₗ) = ∂ₛJ = -β((κ + 2u²)zₙ + W₁/ρ)` at a double root. -/
theorem DMConf.hasSum_M₁ (h : A.DMConf k ε u) (n : ℤ) :
    HasSum (fun l ↦ A.Rrow k ε n l * A.M₁ k ε u (A.z k l))
      (-A.β * ((A.κ k + 2 * u ^ 2) * A.z k n +
        (A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0) / A.ρ)) := by
  by_cases hu : u = 0
  · subst hu
    convert h.hasSum_M₁_zero n using 2
    ring
  · have hρ := A.rho_ne_zero h.μ_ne
    convert (h.hasSum_dmodeΦ n).div_const (2 * u) using 1
    · funext l
      rw [M₁_eq (A := A) (k := k) (ε := ε) hu (h.z_sq_ne l)]
      ring
    · field_simp
      ring

/-! ### The inverse and the closed form -/

variable (A k ε) in
/-- The inverse of `χ_r` at the double root: its column `m` is
`D_v δ + E_v/(t + v) + (2εa + λρ[M₁(t)gᵤ(v) + Φ(u, t)gᵤ(v)²])/β²`, `v = zₘ`,
`gᵤ(v) = 1/(v² - u²)`: the limit of the column of Step 4 as `s₂ → s₁`. -/
noncomputable def invRc (u : ℂ) (n m : ℤ) : ℂ :=
  A.Dv k ε (A.z k m) * kron n m + A.Ev k ε (A.z k m) / (A.z k n + A.z k m) +
      2 * ε * A.a / A.β ^ 2 +
    A.lam ε * A.ρ / A.β ^ 2 * (A.M₁ k ε u (A.z k n) * (1 / (A.z k m ^ 2 - u ^ 2)) +
      A.modeΦ k ε u (A.z k n) * (1 / (A.z k m ^ 2 - u ^ 2)) ^ 2)



/-- `P(t) - e(t) = (κ + 2u²)t² + q₀² - u⁴` at a double root. -/
theorem DMConf.Pph_sub_eph (h : A.DMConf k ε u) (t : ℂ) :
    A.Pph k t - A.eph k ε t = (A.κ k + 2 * u ^ 2) * t ^ 2 + (A.q₀ k ^ 2 - u ^ 4) := by
  rw [h.e_eq, A.Pph_eq_poly]
  ring

/-- **Theorem "density and magnetic"** at a double root, the inverse: `χ_r invRc = 1`. -/
theorem DMConf.chiR_isRightInverse (h : A.DMConf k ε u) :
    IsRightInverse (A.chiR k ε) (A.invRc k ε u) := by
  intro n m
  have hμ := h.μ_ne
  have ha := h.a_ne
  have hβ := A.β_ne_zero
  have hρ := A.rho_ne_zero hμ
  have hPn := A.Pph_z_ne_zero k n
  have hgm := h.z_sq_ne m
  have hS1 := A.hasSum_step1_of k ε h.ε_eq ha hμ (n := n) (h.e_ne m)
  have hS2 := (A.hasSum_step2_of k ε h.ε_eq ha hμ n).mul_left (2 * ε * A.a / A.β ^ 2)
  have hS3 := (h.hasSum_M₁ n).mul_left
    (A.lam ε * A.ρ / A.β ^ 2 * (1 / (A.z k m ^ 2 - u ^ 2)))
  have hS4 := (h.hasSum_modeΦ n).mul_left
    (A.lam ε * A.ρ / A.β ^ 2 * (1 / (A.z k m ^ 2 - u ^ 2)) ^ 2)
  convert (((hS1.add hS2).add hS3).add hS4).div_const (A.Pph k (A.z k n)) using 1
  · funext l
    have hc : A.chiR k ε n l = A.Rrow k ε n l / A.Pph k (A.z k n) := by
      rw [eq_div_iff hPn, mul_comm, Rrow]
      exact A.Pph_mul_chiR k ε hμ n l
    rw [hc, invRc]
    field_simp
    ring
  · have hε0 : ε * (A.a : ℂ) ≠ 0 := by
      have : ε ≠ 0 := by rcases h.ε_eq with rfl | rfl <;> norm_num
      exact mul_ne_zero this ha
    have h4 := step4_conf (β := A.β) (lam := A.lam ε) (ρ := A.ρ) (z := A.z k n)
      (N2z := A.N₂ k ε (A.z k n)) (Pv := A.Pph k (A.z k m)) (ev := A.eph k ε (A.z k m))
      (Pu := A.Pph k u) (cA := A.κ k + 2 * u ^ 2)
      (cB := A.q₀ k ^ 2 - u ^ 4) (W₀ := A.Wph k ε (A.z k n) 0)
      (W₁ := A.Wph k ε (A.z k n) 1 - A.Wph k ε (A.z k n) 0) hβ hρ hε0 hgm (h.e_eq _)
      (h.Pph_sub_eph _) (by rw [A.Pph_eq_poly]; ring)
    rw [← A.Wph_affine k ε, ← A.Wph_affine k ε] at h4
    simp only [Fph, Jgen] at h4 ⊢
    rw [eq_div_iff hPn]
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self]
      linear_combination -h4
    · rw [kron_of_ne hnm]
      linear_combination -h4

/-- The partial fractions at a double root: `F(t) = 1 + (κ + 2u²)gᵤ(t) + P(u)gᵤ(t)²`. -/
theorem DMConf.Fph_eq (h : A.DMConf k ε u) {t : ℂ} (ht : t ^ 2 - u ^ 2 ≠ 0) :
    A.Fph k ε t = 1 + (A.κ k + 2 * u ^ 2) * (1 / (t ^ 2 - u ^ 2)) +
      A.Pph k u * (1 / (t ^ 2 - u ^ 2)) ^ 2 := by
  have hPt : A.Pph k t = A.eph k ε t + ((A.κ k + 2 * u ^ 2) * t ^ 2 + (A.q₀ k ^ 2 - u ^ 4)) := by
    linear_combination h.Pph_sub_eph t
  rw [Fph, hPt, h.e_eq t, A.Pph_eq_poly]
  field_simp
  ring

theorem DMConf.hasDerivAt_Fph (h : A.DMConf k ε u) (n : ℤ) :
    HasDerivAt (A.Fph k ε) (-2 * A.z k n * ((A.κ k + 2 * u ^ 2) * (1 / (A.z k n ^ 2 - u ^ 2)) ^ 2 +
      2 * A.Pph k u * (1 / (A.z k n ^ 2 - u ^ 2)) ^ 3)) (A.z k n) := by
  have h₁ := h.z_sq_ne n
  have hg : HasDerivAt (fun t ↦ 1 / (t ^ 2 - u ^ 2)) (-(2 * A.z k n) / (A.z k n ^ 2 - u ^ 2) ^ 2)
      (A.z k n) := by
    have := ((hasDerivAt_pow 2 (A.z k n)).sub_const (u ^ 2)).inv h₁
    convert this using 1
    · funext t
      rw [one_div]
      rfl
    · push_cast
      ring
  have hF := ((hg.const_mul (A.κ k + 2 * u ^ 2)).fun_add
    ((hg.pow 2).const_mul (A.Pph k u))).const_add 1
  have hev : ∀ᶠ t in 𝓝 (A.z k n), A.Fph k ε t = 1 + ((A.κ k + 2 * u ^ 2) * (1 / (t ^ 2 - u ^ 2)) +
      A.Pph k u * (1 / (t ^ 2 - u ^ 2)) ^ 2) := by
    have hc : ContinuousAt (fun t : ℂ ↦ t ^ 2 - u ^ 2) (A.z k n) := by fun_prop
    filter_upwards [hc.eventually_ne h₁] with t ht
    rw [h.Fph_eq ht]
    ring
  convert hF.congr_of_eventuallyEq hev using 1
  have hD : (A.z k n ^ 2 - u ^ 2)⁻¹ * (A.z k n ^ 2 - u ^ 2) = 1 := inv_mul_cancel₀ h₁
  field_simp
  linear_combination (2 * A.z k n * A.Pph k u) * hD

/-- Eq. (defect-dd) at a double root: `DD(z, v) = F(v)/(z + v) - z[(κ + 2u²)g(z)g(v) +
P(u)(g(z)²g(v) + g(z)g(v)²)]` on the lattice, `g = gᵤ`, including the diagonal. -/
theorem DMConf.DD_eq (h : A.DMConf k ε u) (n m : ℤ) :
    A.DD k ε (A.z k n) (A.z k m) = A.Fph k ε (A.z k m) / (A.z k n + A.z k m) -
      A.z k n * ((A.κ k + 2 * u ^ 2) * (1 / (A.z k n ^ 2 - u ^ 2)) * (1 / (A.z k m ^ 2 - u ^ 2)) +
        A.Pph k u * ((1 / (A.z k n ^ 2 - u ^ 2)) ^ 2 * (1 / (A.z k m ^ 2 - u ^ 2)) +
          (1 / (A.z k n ^ 2 - u ^ 2)) * (1 / (A.z k m ^ 2 - u ^ 2)) ^ 2)) := by
  have hzv := A.z_add_z_ne_zero k h.μ_ne n m
  have hn := h.z_sq_ne n
  have hm := h.z_sq_ne m
  by_cases hnm : n = m
  · subst hnm
    have hz0 : A.z k n ≠ 0 := fun h0 ↦ hzv (by rw [h0]; ring)
    rw [DD, ite_eq_left rfl, (h.hasDerivAt_Fph n).deriv, h.Fph_eq hn]
    field_simp
    ring
  · have hne : A.z k n ≠ A.z k m := fun h0 ↦ hnm (A.z_injective k h0)
    have hsq : A.z k n ^ 2 - A.z k m ^ 2 ≠ 0 := by
      rw [show A.z k n ^ 2 - A.z k m ^ 2 = (A.z k n - A.z k m) * (A.z k n + A.z k m) by ring]
      exact mul_ne_zero (sub_ne_zero.mpr hne) hzv
    rw [DD, ite_eq_right hne, h.Fph_eq hn, h.Fph_eq hm]
    field_simp
    ring

variable (A k ε) in
/-- The closed form of Eq. (Gamma-ph) at a double root `s₁ = s₂ = u²`:
`βP(zₙ)[1/G₂(zₙ) - H(-zₙ)/e(zₙ)]δₙₘ + 2εa - λρDD(zₙ, zₘ) + λρ ∂ₛ[Z(s)gₛ(zₙ)gₛ(zₘ)]`, with
`∂ₛZ = Zsd`. -/
noncomputable def gammaRc (u : ℂ) (n m : ℤ) : ℂ :=
  A.β * A.Pph k (A.z k n) *
        (1 / A.G₂ k (A.z k n) - A.Hph k ε (-A.z k n) / A.eph k ε (A.z k n)) * kron n m +
      2 * ε * A.a - A.lam ε * A.ρ * A.DD k ε (A.z k n) (A.z k m) +
    A.lam ε * A.ρ * (A.Zsd k ε u * (1 / (A.z k n ^ 2 - u ^ 2)) *
        (1 / (A.z k m ^ 2 - u ^ 2)) +
      A.Zph k ε u * ((1 / (A.z k n ^ 2 - u ^ 2)) ^ 2 * (1 / (A.z k m ^ 2 - u ^ 2)) +
        (1 / (A.z k n ^ 2 - u ^ 2)) * (1 / (A.z k m ^ 2 - u ^ 2)) ^ 2))

/-- **Theorem "density and magnetic"** at a double root, the closed form:
`β²(χ_r⁻¹ - B⁻¹) = Γ_r` for the inverse `invRc`. -/
theorem DMConf.gammaR_eq (h : A.DMConf k ε u) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invRc k ε u n m - kron n m / A.Bph k n) = A.gammaRc k ε u n m := by
  have hβ := A.β_ne_zero
  have hzv := A.z_add_z_ne_zero k h.μ_ne n m
  have hPn := A.Pph_z_ne_zero k n
  have hGn := A.G₂_z_ne_zero k n
  have hen := h.e_ne n
  have hem := h.e_ne m
  have hn := h.z_sq_ne n
  have hm := h.z_sq_ne m
  rw [invRc, gammaRc, h.DD_eq n m, A.Bph_eq k n]
  simp only [Dv, Ev, modeΦ, M₁]
  by_cases hnm : n = m
  · subst hnm
    have hz0 : A.z k n ≠ 0 := fun h0 ↦ hzv (by rw [h0]; ring)
    rw [kron_self]
    field_simp
    ring
  · rw [kron_of_ne hnm]
    field_simp
    ring

variable (A k ε) in
theorem gammaRc_symm (u : ℂ) (n m : ℤ) : A.gammaRc k ε u n m = A.gammaRc k ε u m n := by
  by_cases hnm : n = m
  · rw [hnm]
  · rw [gammaRc, gammaRc, kron_of_ne hnm, kron_of_ne (Ne.symm hnm), A.DD_symm k ε]
    ring

theorem DMConf.invRc_symm (h : A.DMConf k ε u) (n m : ℤ) :
    A.invRc k ε u n m = A.invRc k ε u m n := by
  have hβ : (A.β : ℂ) ^ 2 ≠ 0 := pow_ne_zero 2 A.β_ne_zero
  have hk : kron n m / A.Bph k n = kron m n / A.Bph k m := by
    by_cases hnm : n = m
    · rw [hnm]
    · rw [kron_of_ne hnm, kron_of_ne (Ne.symm hnm), zero_div, zero_div]
  apply mul_left_cancel₀ hβ
  linear_combination h.gammaR_eq n m - h.gammaR_eq m n + A.gammaRc_symm k ε u n m +
    (A.β : ℂ) ^ 2 * hk

/-- **Theorem "density and magnetic"** at a double root, two-sidedness: `invRc χ_r = 1`. -/
theorem DMConf.chiR_isLeftInverse (h : A.DMConf k ε u) :
    IsLeftInverse (A.chiR k ε) (A.invRc k ε u) := by
  intro n m
  have := h.chiR_isRightInverse m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [A.chiR_symm k ε h.μ_ne l m, h.invRc_symm n l, mul_comm]

/-! ### The assumptions of the note, and `∂ₛZ` -/

variable (A k ε) in
/-- At a double root `u` with `K(u) ≠ 0`, `P(u) ≠ 0`: otherwise `u` is the atomic pair of `r = m`,
`k = 0` (`atomic_of_Pph_eq_zero`), and `e = (t² - a²)²` would force `ρ² = 1 - 2p₁`, i.e.
`4p₀p₂ = -2p₁(p₀ + p₂) < 0`. -/
theorem Pph_ne_of_double (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) {u : ℂ}
    (he : ∀ t, A.eph k ε t = (t ^ 2 - u ^ 2) ^ 2) (hK : A.Kph k ε u ≠ 0) : A.Pph k u ≠ 0 := by
  intro hP
  have heu : A.eph k ε u = 0 := by rw [he, sub_self, zero_pow two_ne_zero]
  obtain ⟨rfl, hw, hu⟩ := A.atomic_of_Pph_eq_zero k ε hε ha heu hK hP
  -- the `t²` coefficient of `e`: `2q₁ - 4a²ρ² = -2a²`
  have h₀ := he 0
  have h₁ := he 1
  simp only [eph, Hph, lam, q₀, hw, hu] at h₀ h₁
  have hc : (A.a : ℂ) ^ 2 * (4 - 8 * A.p₁ - 4 * A.ρ ^ 2) = 0 := by linear_combination h₁ - h₀
  have hc' : (4 - 8 * A.p₁ - 4 * A.ρ ^ 2 : ℝ) = 0 := by
    have := (mul_eq_zero.mp hc).resolve_left (pow_ne_zero 2 ha)
    exact_mod_cast this
  have hs := A.weights_sum
  have hρ : A.ρ = A.p₂ - A.p₀ := rfl
  have := A.p₀_pos
  have := A.p₁_pos
  have := A.p₂_pos
  rw [hρ] at hc'
  nlinarith [mul_pos A.p₀_pos A.p₂_pos, mul_pos A.p₁_pos (add_pos A.p₀_pos A.p₂_pos)]

variable (A k ε) in
/-- The assumptions of Theorem "density and magnetic" at a double root `s₁ = s₂ = u²`, as in the
note: `μ ≠ 0`, `e(zₙ) ≠ 0` for all `n` and `K(u) ≠ 0` (with `U ≠ 0`). -/
structure DMConfHyp (u : ℂ) : Prop where
  ε_eq : ε = 1 ∨ ε = -1
  a_ne : (A.a : ℂ) ≠ 0
  μ_ne : A.μ ≠ 0
  e_ne : ∀ n, A.eph k ε (A.z k n) ≠ 0
  e_eq : ∀ t, A.eph k ε t = (t ^ 2 - u ^ 2) ^ 2
  K_ne : A.Kph k ε u ≠ 0

theorem DMConfHyp.toDMConf (h : A.DMConfHyp k ε u) : A.DMConf k ε u :=
  ⟨h.ε_eq, h.a_ne, h.μ_ne, h.e_ne, h.e_eq, h.K_ne,
    A.Pph_ne_of_double k ε h.ε_eq h.a_ne h.e_eq h.K_ne⟩

/-- At a double root, `K(-u) ≠ 0` too: `βH(u) = -λα(-u)` and `β²H(u)H(-u) = λ²α(u)α(-u)` would give
`βH(-u) = -λα(u)`. -/
theorem DMConf.Kph_neg_ne (h : A.DMConf k ε u) : A.Kph k ε (-u) ≠ 0 := by
  have hrel : (A.β : ℂ) ^ 2 * A.Hph k ε u * A.Hph k ε (-u) =
      A.lam ε ^ 2 * (A.α k u * A.α k (-u)) := by
    rw [A.α_mul_α_neg k h.z_ne h.z_ne_neg]
    have := h.e_u
    rw [eph] at this
    linear_combination (A.β : ℂ) ^ 2 * this
  have hαn : A.α k (-u) ≠ 0 := by
    intro h0
    have := A.α_mul_α_neg k h.z_ne h.z_ne_neg
    rw [h0, mul_zero] at this
    have hp : (A.β : ℂ) ^ 2 * A.p₀ * A.p₂ ≠ 0 := by
      have := A.β_ne_zero
      have := ofReal_ne_zero.mpr A.p₀_pos.ne'
      have := ofReal_ne_zero.mpr A.p₂_pos.ne'
      positivity
    exact hp this.symm
  have hK := h.K_ne
  intro h0
  apply hK
  simp only [Kph, neg_neg] at h0 ⊢
  have hlam := A.lam_ne_zero ε h.ε_eq h.a_ne
  apply mul_left_cancel₀ (mul_ne_zero hlam hαn)
  linear_combination (A.β * A.Hph k ε (-u)) * h0 - hrel

variable (A k ε) in
/-- The even part `Z(s) = [Z(u) + Z(-u)]/2` of `Z`, as a function of `u`. -/
noncomputable def Zeven (u : ℂ) : ℂ := (A.Zph k ε u + A.Zph k ε (-u)) / 2

variable (A k ε) in
/-- Eq. (branch): `[Z(u) - Z(-u)]/2 = β²uP(u)e(u)/(K(u)K(-u))` off the lattice. -/
theorem Zph_sub_Zph_neg {u : ℂ} (hoff : ∀ l, A.z k l ≠ u) (hoff' : ∀ l, A.z k l ≠ -u)
    (hK : A.Kph k ε u ≠ 0) (hK' : A.Kph k ε (-u) ≠ 0) :
    (A.Zph k ε u - A.Zph k ε (-u)) / 2 =
      A.β ^ 2 * u * A.Pph k u * A.eph k ε u / (A.Kph k ε u * A.Kph k ε (-u)) := by
  have hα := A.α_mul_α_neg k hoff hoff'
  have e₁ : (A.Zph k ε u - A.Zph k ε (-u)) / 2 = u * A.Pph k u * (A.β ^ 2 * A.Hph k ε (-u) *
      A.Hph k ε u - A.lam ε ^ 2 * (A.α k u * A.α k (-u))) / (A.Kph k ε u * A.Kph k ε (-u)) := by
    simp only [Kph, neg_neg] at hK hK' ⊢
    simp only [Zph, Kph, A.Pph_neg, neg_neg]
    field_simp
    ring
  rw [e₁, hα, eph]
  ring

/-- `Zsd = ∂ₛZ(s)` at `s = u²`: the difference quotients of the even part `Z(s)` in `s = v²`
converge to `Zsd`. (For `u ≠ 0`, the odd part has a double zero at `u`, Eq. branch.) -/
theorem DMConf.tendsto_Zeven (h : A.DMConf k ε u) :
    Tendsto (fun v ↦ (A.Zeven k ε v - A.Zeven k ε u) / (v ^ 2 - u ^ 2)) (𝓝[≠] u)
      (𝓝 (A.Zsd k ε u)) := by
  by_cases hu : u = 0
  · subst hu
    have hslope := (hasDerivAt_iff_tendsto_slope.mp (A.hasDerivAt_fZ k ε h.z_ne h.K_ne))
    have hneg : Tendsto (fun v : ℂ ↦ -v) (𝓝[≠] 0) (𝓝[≠] 0) := by
      refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ ?_ ?_
      · have := (continuous_neg (G := ℂ)).tendsto 0
        rw [neg_zero] at this
        exact this.mono_left nhdsWithin_le_nhds
      · filter_upwards [self_mem_nhdsWithin] with v hv
        exact neg_ne_zero.mpr hv
    have h₂ := (hslope.add (hslope.comp hneg)).div_const 2
    rw [show (A.fZd k ε 0 + A.fZd k ε 0) / 2 = A.fZd k ε 0 by ring] at h₂
    rw [Zsd, ite_eq_left rfl]
    refine h₂.congr' ?_
    filter_upwards [self_mem_nhdsWithin] with v hv
    have hv' : v ≠ 0 := hv
    have hv'' : -v ≠ 0 := neg_ne_zero.mpr hv'
    simp only [Function.comp, slope_def_field, sub_zero, Zeven, A.Zph_eq, neg_zero, zero_mul,
      add_zero, zero_div, neg_mul]
    field_simp
    ring
  · -- the odd part `β²vP(v)e(v)/(K(v)K(-v))` has a double zero at `u`
    have hK' := h.Kph_neg_ne
    have hcKn : ContinuousAt (fun v ↦ A.Kph k ε (-v)) u :=
      (A.continuousAt_Kph k ε (u := -u) (fun l h0 ↦ h.z_ne_neg l h0)).comp continuousAt_neg
    have hcP := (A.hasDerivAt_Pph k u).continuousAt
    have hcK := A.continuousAt_Kph k ε h.z_ne
    have hR : ContinuousAt (fun v ↦ (v + u) ^ 2 * (A.β ^ 2 * v * A.Pph k v /
        (A.Kph k ε v * A.Kph k ε (-v)))) u :=
      ((continuousAt_id.add continuousAt_const).pow 2).mul
        (((continuousAt_const.mul continuousAt_id).mul hcP).div (hcK.mul hcKn)
          (mul_ne_zero h.K_ne hK'))
    have hodd := hasDerivAt_sq_mul hR
    -- near `u`, `Z(v) = Z(-v) + 2(odd part)`
    obtain ⟨r₁, hr₁, hd₁⟩ := A.exists_lattice_dist h.z_ne
    obtain ⟨r₂, hr₂, hd₂⟩ := A.exists_lattice_dist h.z_ne_neg
    have hev : ∀ᶠ v in 𝓝 u, A.Zeven k ε v = A.Zph k ε v -
        (v - u) ^ 2 * ((v + u) ^ 2 * (A.β ^ 2 * v * A.Pph k v /
          (A.Kph k ε v * A.Kph k ε (-v)))) := by
      have hball : ∀ᶠ v in 𝓝 u, ‖v - u‖ < min r₁ r₂ := by
        have : ContinuousAt (fun v : ℂ ↦ ‖v - u‖) u := by fun_prop
        exact this.eventually_lt continuousAt_const (by simpa using lt_min hr₁ hr₂)
      filter_upwards [hball, hcK.eventually_ne h.K_ne, hcKn.eventually_ne hK'] with v hv hKv hKnv
      have hoff : ∀ l, A.z k l ≠ v := by
        intro l h0
        have h₁ := hd₁ l
        rw [h0] at h₁
        linarith [min_le_left r₁ r₂]
      have hoff' : ∀ l, A.z k l ≠ -v := by
        intro l h0
        have h₂ := hd₂ l
        rw [h0, show -v - -u = -(v - u) by ring, norm_neg] at h₂
        linarith [min_le_right r₁ r₂]
      have hb := A.Zph_sub_Zph_neg k ε hoff hoff' hKv hKnv
      rw [h.e_eq v] at hb
      rw [Zeven]
      have e₁ : A.Zph k ε (-v) = A.Zph k ε v - 2 * ((A.Zph k ε v - A.Zph k ε (-v)) / 2) := by ring
      rw [e₁, hb]
      field_simp
      ring
    have hZe : HasDerivAt (A.Zeven k ε) (A.Zder k ε u - 0) u :=
      ((A.hasDerivAt_Zph k ε h.z_ne h.K_ne).sub hodd).congr_of_eventuallyEq hev
    rw [sub_zero] at hZe
    have hs := hasDerivAt_iff_tendsto_slope.mp hZe
    have hq : Tendsto (fun v : ℂ ↦ 1 / (v + u)) (𝓝[≠] u) (𝓝 (1 / (u + u))) := by
      have hc : ContinuousAt (fun v : ℂ ↦ 1 / (v + u)) u :=
        continuousAt_const.div (continuousAt_id.add continuousAt_const)
          (show id u + u ≠ 0 by simp only [id]; rw [← two_mul]; exact mul_ne_zero two_ne_zero hu)
      exact hc.tendsto.mono_left nhdsWithin_le_nhds
    have hlim := hs.mul hq
    have hval : A.Zder k ε u * (1 / (u + u)) = A.Zsd k ε u := by
      rw [Zsd, ite_eq_right hu]
      field_simp
      ring
    rw [hval] at hlim
    refine hlim.congr' ?_
    filter_upwards [self_mem_nhdsWithin, eventually_ne_nhdsWithin (show u ≠ -u from
      fun h0 ↦ hu (by linear_combination h0 / 2))] with v hv hvn
    have h₁ : v - u ≠ 0 := sub_ne_zero.mpr hv
    have h₂ : v + u ≠ 0 := fun h0 ↦ hvn (by linear_combination h0)
    have h₃ : v ^ 2 - u ^ 2 ≠ 0 := by
      rw [show v ^ 2 - u ^ 2 = (v - u) * (v + u) by ring]
      exact mul_ne_zero h₁ h₂
    rw [slope_def_field]
    field_simp
    ring

end Params

end HubbardAtom
