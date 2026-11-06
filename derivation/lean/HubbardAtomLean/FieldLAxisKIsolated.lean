import HubbardAtomLean.FieldLAxisCornerK
import HubbardAtomLean.RootProducts

/-!
# On the axes, the `K` conditions of `LAxisGeneric` fail only at isolated points

Not in the note.
-/

open Complex Filter Topology Polynomial

namespace HubbardAtom

/-! ### `K` as a function of the weights -/

/-- `K_{t,σ}(u)` of Eq. (field-L) as a function of the data `β`, `a`, `q₀` and the weights
`p = (p₀, p↑, p↓, p₂)`, with `S_t(u)` replaced by `S` (`FParams.KL_eq_KLx`). -/
noncomputable def KLx (β a q : ℂ) (p : Fin 4 → ℂ) (t σ : Bool) (u S : ℂ) : ℂ :=
  let pL : Bool → ℂ := fun t ↦ bif t then p 1 else p 2
  let ρ := p 3 - p 0
  let mL : Bool → ℂ := fun t ↦ pL t - pL (!t)
  let r : Bool → ℂ := fun t ↦ ρ - mL t
  let H : ℂ → ℂ → ℂ := fun r z ↦ z ^ 2 + 2 * a * r * z + q
  let lam := 4 * a ^ 2
  let πP := p 0 * p 3
  let πS := p 1 * p 2
  if σ = t then
    -β * (H (r t) (-u) * H (r (!t)) u * H (r (!t)) (-u) -
        lam ^ 2 * (πP * H (r (!t)) (-u) + πS * H (r (!t)) u)) +
      2 * lam ^ 2 * (u ^ 2 + q) *
        ((p 3 * pL t - p 0 * pL (!t)) * S - β * (p 3 * pL t + p 0 * pL (!t)) / 2)
  else
    -(lam * (H (r t) (-u) * H (r (!t)) (-u) + lam ^ 2 * (πP - πS))) *
        (-(mL (!t) * S) - β * (p 1 + p 2) / 2) -
      lam * (H (r t) (-u) * H (r (!t)) u - lam ^ 2 * (πP - πS)) * (ρ * S - β * (p 0 + p 3) / 2)

/-- `KLx` is affine in `S`. -/
theorem KLx_affine (β a q : ℂ) (p : Fin 4 → ℂ) (t σ : Bool) (u S : ℂ) :
    KLx β a q p t σ u S =
      KLx β a q p t σ u 0 + S * (KLx β a q p t σ u 1 - KLx β a q p t σ u 0) := by
  unfold KLx
  split_ifs <;> ring

/-- The numerator of `S_k`: `sinh` for even `k`, `cosh` for odd `k`. -/
noncomputable def thNum (k : ℤ) (z : ℂ) : ℂ := if Even k then sinh z else cosh z

/-- The denominator of `S_k`: `cosh` for even `k`, `sinh` for odd `k`. -/
noncomputable def thDen (k : ℤ) (z : ℂ) : ℂ := if Even k then cosh z else sinh z

/-- `S_k(x) = -(β/2) thNum/thDen` at `z = β(x - μ)/2`. -/
theorem Params.S_eq_thNum_div (A : Params) (k : ℤ) (x : ℂ) :
    A.S k x = -((A.β : ℂ) / 2) * (thNum k (A.β * (x - A.μ) / 2) / thDen k (A.β * (x - A.μ) / 2)) := by
  unfold Params.S thNum thDen
  split_ifs <;> simp [tanh_eq_sinh_div_cosh, coth]

/-- Off the lattice, the denominator of `S_k` does not vanish. -/
theorem Params.thDen_ne_zero_of_off (A : Params) (k : ℤ) {x : ℂ} (hx : ∀ n, A.z k n ≠ x) :
    thDen k (A.β * (x - A.μ) / 2) ≠ 0 := by
  unfold thDen
  split_ifs with hk
  · obtain ⟨j, hj⟩ := hk
    exact A.cosh_ne_zero_of_off (j := j) (by omega) hx
  · obtain ⟨j, hj⟩ := Int.not_even_iff_odd.mp hk
    exact A.sinh_ne_zero_of_off (j := j + 1) (by omega) hx

/-- `K_{t,t} + λK_{t,-t}` as a function of the data, with `S_t(u)` replaced by `S`. -/
noncomputable def KLc (β a q : ℂ) (p : Fin 4 → ℂ) (lam : ℂ) (t : Bool) (u S : ℂ) : ℂ :=
  KLx β a q p t t u S + lam * KLx β a q p t (!t) u S

/-- `KLc` with the poles of `S_t` cleared: the value at `S = -(β/2) thNum/thDen` times `thDen`,
at the centre `c` of the lattice of `t`. -/
noncomputable def KLcd (β a q : ℂ) (k : ℤ) (p : Fin 4 → ℂ) (lam : ℂ) (t : Bool) (c u : ℂ) : ℂ :=
  KLc β a q p lam t u 0 * thDen k (β * (u - c) / 2) +
    (KLc β a q p lam t u 1 - KLc β a q p lam t u 0) * (-(β / 2) * thNum k (β * (u - c) / 2))

theorem KLc_affine (β a q : ℂ) (p : Fin 4 → ℂ) (lam : ℂ) (t : Bool) (u S : ℂ) :
    KLc β a q p lam t u S =
      KLc β a q p lam t u 0 + S * (KLc β a q p lam t u 1 - KLc β a q p lam t u 0) := by
  unfold KLc
  rw [KLx_affine β a q p t t u S, KLx_affine β a q p t (!t) u S]
  ring

/-- Where the denominator does not vanish, `KLcd = KLc(S) thDen`. -/
theorem KLcd_eq (β a q : ℂ) (k : ℤ) (p : Fin 4 → ℂ) (lam : ℂ) (t : Bool) (c u : ℂ)
    (hd : thDen k (β * (u - c) / 2) ≠ 0) :
    KLcd β a q k p lam t c u =
      KLc β a q p lam t u (-(β / 2) * (thNum k (β * (u - c) / 2) / thDen k (β * (u - c) / 2))) *
        thDen k (β * (u - c) / 2) := by
  rw [KLc_affine, KLcd]
  field_simp

/-- The function whose product over the roots is analytic along the axes:
`f(u) + μ' f(-u)` with `f = KLcd↑ · KLcd↓`, at the centres `c σ` of the lattices. -/
noncomputable def gK (β a q : ℂ) (k : ℤ) (p : Fin 4 → ℂ) (c : Bool → ℂ) (lam μ' u : ℂ) : ℂ :=
  KLcd β a q k p lam true (c true) u * KLcd β a q k p lam false (c false) u +
    μ' * (KLcd β a q k p lam true (c true) (-u) * KLcd β a q k p lam false (c false) (-u))

/-! ### Even polynomials in `v` from their coefficients in `s = v²` -/

/-- `v⁸ + b₃v⁶ + b₂v⁴ + b₁v² + b₀`. -/
noncomputable def evenPoly8 (b : Fin 4 → ℂ) : ℂ[X] :=
  X ^ 8 + C (b 3) * X ^ 6 + C (b 2) * X ^ 4 + C (b 1) * X ^ 2 + C (b 0)

/-- The quotient of `evenPoly8 b` by `v² - A` (synthetic division in `s = v²`):
`v⁶ + c₂v⁴ + c₁v² + c₀` with `c₂ = b₃ + A`, `c₁ = b₂ + Ac₂`, `c₀ = b₁ + Ac₁`. -/
noncomputable def evenPoly6 (A : ℂ) (b : Fin 4 → ℂ) : ℂ[X] :=
  X ^ 6 + C (b 3 + A) * X ^ 4 + C (b 2 + A * (b 3 + A)) * X ^ 2 +
    C (b 1 + A * (b 2 + A * (b 3 + A)))

theorem evenPoly8_monic (b : Fin 4 → ℂ) : (evenPoly8 b).Monic := by
  unfold evenPoly8
  monicity!

theorem evenPoly8_natDegree (b : Fin 4 → ℂ) : (evenPoly8 b).natDegree = 8 := by
  unfold evenPoly8
  compute_degree!

theorem evenPoly6_monic (A : ℂ) (b : Fin 4 → ℂ) : (evenPoly6 A b).Monic := by
  unfold evenPoly6
  monicity!

theorem evenPoly6_natDegree (A : ℂ) (b : Fin 4 → ℂ) : (evenPoly6 A b).natDegree = 6 := by
  unfold evenPoly6
  compute_degree!

theorem evenPoly8_eval (b : Fin 4 → ℂ) (v : ℂ) :
    (evenPoly8 b).eval v = (v ^ 2) ^ 4 + b 3 * (v ^ 2) ^ 3 + b 2 * (v ^ 2) ^ 2 + b 1 * v ^ 2 + b 0 := by
  simp only [evenPoly8, eval_add, eval_mul, eval_pow, eval_C, eval_X]
  ring

/-- `evenPoly8 b = (v² - A) evenPoly6 A b` plus the remainder `evenPoly8 b` at `v² = A`. -/
theorem evenPoly8_eval_eq (A : ℂ) (b : Fin 4 → ℂ) (v : ℂ) :
    (evenPoly8 b).eval v = (v ^ 2 - A) * (evenPoly6 A b).eval v +
      (A ^ 4 + b 3 * A ^ 3 + b 2 * A ^ 2 + b 1 * A + b 0) := by
  simp only [evenPoly8, evenPoly6, eval_add, eval_mul, eval_pow, eval_C, eval_X]
  ring

/-- The coefficients of `evenPoly8 (b w)` are analytic where the `b w j` are. -/
theorem analyticAt_evenPoly8_coeff {b : ℂ → Fin 4 → ℂ} {w₀ : ℂ}
    (hb : ∀ j, AnalyticAt ℂ (fun w ↦ b w j) w₀) (i : ℕ) :
    AnalyticAt ℂ (fun w ↦ (evenPoly8 (b w)).coeff i) w₀ := by
  simp only [evenPoly8, coeff_add, coeff_C_mul, coeff_X_pow, coeff_C]
  have h3 := hb 3
  have h2 := hb 2
  have h1 := hb 1
  have h0 := hb 0
  split_ifs <;> fun_prop

/-- The coefficients of `evenPoly6 A (b w)` are analytic where the `b w j` are. -/
theorem analyticAt_evenPoly6_coeff (A : ℂ) {b : ℂ → Fin 4 → ℂ} {w₀ : ℂ}
    (hb : ∀ j, AnalyticAt ℂ (fun w ↦ b w j) w₀) (i : ℕ) :
    AnalyticAt ℂ (fun w ↦ (evenPoly6 A (b w)).coeff i) w₀ := by
  simp only [evenPoly6, coeff_add, coeff_C_mul, coeff_X_pow, coeff_C]
  have h3 := hb 3
  have h2 := hb 2
  have h1 := hb 1
  split_ifs <;> fun_prop

namespace FParams

variable (F : FParams) (k : ℤ)

/-- The weights `(p₀, p↑, p↓, p₂)`. -/
noncomputable def pw : Fin 4 → ℂ := ![F.p₀, F.pUp, F.pDn, F.p₂]

@[simp] theorem pw_zero : F.pw 0 = F.p₀ := rfl
@[simp] theorem pw_one : F.pw 1 = F.pUp := rfl
@[simp] theorem pw_two : F.pw 2 = F.pDn := rfl
@[simp] theorem pw_three : F.pw 3 = F.p₂ := rfl

/-- `K_{t,σ}(u) = KLx(…, S_t(u))`. -/
theorem KL_eq_KLx (t σ : Bool) (u : ℂ) :
    F.KL k t σ u = KLx F.β F.a (F.q₀L k) F.pw t σ u ((F.BL t).S k u) := by
  cases t <;> cases σ <;>
  · simp only [KL, fMp, fMm, KLx, QL, HL, αL, αSL, αPL, DL, cLw, cP, cS, mL, rL, pL, ρ, piP, piS,
      lamL, pw_zero, pw_one, pw_two, pw_three, Bool.not_true, Bool.not_false,
      Bool.cond_true, Bool.cond_false, reduceCtorEq, ite_true, ite_false]
    push_cast
    ring

/-! ### The axes with complex `s` -/

variable (d : Axis)

/-- The energies of the four states at the axis point `s`, continued to complex `s = w`. -/
noncomputable def axisE (w : ℂ) : Fin 4 → ℂ :=
  ![F.U / 4 + w * d.dμ, -F.U / 4 - w * d.dh, -F.U / 4 + w * d.dh, F.U / 4 - w * d.dμ]

/-- The partition function of the axis point `s`, continued to complex `s = w`. -/
noncomputable def axisZ (w : ℂ) : ℂ :=
  cexp (-F.β * (F.U / 4 + w * d.dμ)) + cexp (-F.β * (-F.U / 4 - w * d.dh)) +
    cexp (-F.β * (-F.U / 4 + w * d.dh)) + cexp (-F.β * (F.U / 4 - w * d.dμ))

/-- The weights of the axis point `s`, continued to complex `s = w`. -/
noncomputable def axisP (w : ℂ) (i : Fin 4) : ℂ := cexp (-F.β * F.axisE d w i) / F.axisZ d w

theorem axisZ_ofReal (s : ℝ) : F.axisZ d s = (F.axis d s).Z := by
  simp only [axisZ, Z, boltz₀, boltzUp, boltzDn, boltz₂, axis_U, axis_β, axis_μ, axis_h]
  push_cast
  rfl

/-- At real `s`, the continued weights are the weights. -/
theorem axisP_ofReal (s : ℝ) : F.axisP d s = (F.axis d s).pw := by
  ext i
  rw [axisP, axisZ_ofReal]
  fin_cases i <;>
  · simp only [axisE, pw, p₀, pUp, pDn, p₂, boltz₀, boltzUp, boltzDn, boltz₂, axis_U, axis_β,
      axis_μ, axis_h]
    push_cast
    rfl

/-- The continued weights are analytic near the real axis. -/
theorem analyticAt_axisP (s₀ : ℝ) (i : Fin 4) : AnalyticAt ℂ (fun w ↦ F.axisP d w i) s₀ := by
  have hZ : F.axisZ d s₀ ≠ 0 := by
    rw [axisZ_ofReal]
    exact_mod_cast (F.axis d s₀).Z_pos.ne'
  simp only [axisP, axisZ] at hZ ⊢
  fin_cases i <;> simp [axisE] at hZ ⊢ <;>
    fun_prop (disch := exact hZ)

/-- The arguments `(r↑, r↓, π_S, π_P)` of `E8bx` from the weights `(p₀, p↑, p↓, p₂)`. -/
noncomputable def E8xOf (p : Fin 4 → ℂ) : Fin 4 → ℂ :=
  ![p 3 - p 0 - (p 1 - p 2), p 3 - p 0 + (p 1 - p 2), p 1 * p 2, p 0 * p 3]

/-- The coefficients of `E₈` in `s = v²` from the weights. -/
theorem E8b_eq_E8xOf (G : FParams) (j : Fin 4) :
    G.E8b k j = E8bx G.a (G.q₀L k) (E8xOf G.pw) j := by
  have e : E8xOf G.pw = ![(G.rUp : ℂ), G.rDn, G.piS, G.piP] := by
    ext i
    fin_cases i <;> simp [E8xOf, rUp, rDn, ρ, mf, piS, piP]
  rw [e]
  fin_cases j <;> rfl

/-- `E₈` along the axis, continued to complex `s = w`, as an even polynomial in `v`; at `k = 0`
divided by `v² - a²`. -/
noncomputable def axisPoly (w : ℂ) : ℂ[X] :=
  if k = 0 then evenPoly6 (F.a ^ 2) fun j ↦ E8bx F.a (F.q₀L k) (E8xOf (F.axisP d w)) j
  else evenPoly8 fun j ↦ E8bx F.a (F.q₀L k) (E8xOf (F.axisP d w)) j

theorem axisPoly_monic (w : ℂ) : (F.axisPoly k d w).Monic := by
  unfold axisPoly
  split_ifs
  exacts [evenPoly6_monic _ _, evenPoly8_monic _]

theorem axisPoly_natDegree (w : ℂ) :
    (F.axisPoly k d w).natDegree = if k = 0 then 6 else 8 := by
  unfold axisPoly
  split_ifs
  exacts [evenPoly6_natDegree _ _, evenPoly8_natDegree _]

theorem analyticAt_axisPoly_coeff (s₀ : ℝ) (i : ℕ) :
    AnalyticAt ℂ (fun w ↦ (F.axisPoly k d w).coeff i) s₀ := by
  have hp := F.analyticAt_axisP d s₀
  have hb : ∀ j, AnalyticAt ℂ (fun w ↦ E8bx F.a (F.q₀L k) (E8xOf (F.axisP d w)) j) s₀ := by
    have h0 := hp 0
    have h1 := hp 1
    have h2 := hp 2
    have h3 := hp 3
    intro j
    fin_cases j <;> simp only [E8bx, E8xOf] <;> simp <;> fun_prop
  unfold axisPoly
  split_ifs
  exacts [analyticAt_evenPoly6_coeff _ hb i, analyticAt_evenPoly8_coeff hb i]

/-- At real `s`, `E₈ = axisPoly` (`k ≠ 0`) or `E₈ = (v² - a²) axisPoly` (`k = 0`). -/
theorem E8L_eq_axisPoly (s : ℝ) (v : ℂ) :
    (F.axis d s).E8L k v = if k = 0 then (v ^ 2 - F.a ^ 2) * (F.axisPoly k d s).eval v
      else (F.axisPoly k d s).eval v := by
  have hb : ∀ j, E8bx F.a (F.q₀L k) (E8xOf (F.axisP d s)) j = (F.axis d s).E8b k j := fun j ↦ by
    rw [axisP_ofReal, E8b_eq_E8xOf]
    rfl
  rw [E8L_eq_quartic]
  unfold axisPoly
  split_ifs with hk
  · subst hk
    have h0 := (F.axis d s).E8L_of_atomic 0 (u := F.a) ⟨rfl, rfl⟩
    rw [E8L_eq_quartic] at h0
    rw [← add_zero ((v ^ 2 - _) * _), ← h0]
    simp only [← hb]
    rw [← evenPoly8_eval, evenPoly8_eval_eq (F.a ^ 2)]
  · simp only [evenPoly8_eval, hb]

/-! ### The product over the roots -/

attribute [local fun_prop] analyticAt_fst analyticAt_snd

/-- The product of `gK` over the roots of `axisPoly`, at complex `s = w`. -/
noncomputable def axisG (lam μ' w : ℂ) : ℂ :=
  ((F.axisPoly k d w).roots.map
    (gK F.β F.a (F.q₀L k) k (F.axisP d w) (fun σ ↦ w * d.slope σ) lam μ')).prod

/-- **The product over the roots is analytic near the real axis** (`analyticAt_prod_roots`). -/
theorem analyticAt_axisG (lam μ' : ℂ) (s₀ : ℝ) : AnalyticAt ℂ (F.axisG k d lam μ') s₀ := by
  refine analyticAt_prod_roots (F.axisPoly_monic k d) (F.axisPoly_natDegree k d)
    (F.analyticAt_axisPoly_coeff k d s₀) ?_
  have hp : ∀ i, ∀ᶠ w in 𝓝 (s₀ : ℂ), AnalyticAt ℂ (fun w ↦ F.axisP d w i) w := fun i ↦
    (F.analyticAt_axisP d s₀ i).eventually_analyticAt
  filter_upwards [hp 0, hp 1, hp 2, hp 3] with w h0 h1 h2 h3 x
  change AnalyticAt ℂ (fun z : ℂ × ℂ ↦ gK F.β F.a (F.q₀L k) k (F.axisP d z.1)
    (fun σ ↦ z.1 * d.slope σ) lam μ' z.2) (w, x)
  simp only [gK, KLcd, KLc, KLx, thNum, thDen, Bool.not_true, Bool.not_false, Bool.cond_true,
    Bool.cond_false, ite_true, reduceCtorEq, ite_false]
  split_ifs <;> fun_prop

end FParams

end HubbardAtom
