import HubbardAtomLean.FieldInput
import HubbardAtomLean.PHIdentities

/-!
# Lattice sums of the transverse sector

The sums of the proof of Theorem "density and magnetic" for the transverse data (App. F.6), whose
lattice is that of the zero-field atom `B = (U, β, μ)` (`FParams.lat`) and whose polynomials carry
the complex transfer `W = iω_k - 2h`.

* `FParams.cw`: the weights of the field atom are `p₀ = cw B.p₀`, `p₂ = cw B.p₂`, as
  `p₂/p₀ = e^{2βμ}` does not depend on `h`; so `ρ`, `c̄`, the Cauchy matrix and `α` are `cw`
  times those of `B` (`rho_mul_S_neg_zT`, `hasSum_cauchyT`, `hasSum_squareT`, `αT_mul_αT_neg`).
* `FParams.S₀_of_exp`, `sT₁` … `sT₄`: the values of `S_k` at the four zeros `∓a ± W/2` of `P`, from
  the weights.
* `FParams.hasSum_Q_subT`, `hasSum_inv_QT`, `hasSum_cubicT`: the lattice sums of `M/P` for cubic
  `M`, also at `q₀ = 0` (`k = 0`, `h = ±a`, `q₀T_eq_zero`), where `P(t) = t²(t² - 4a²)` has a
  double zero and the sums `ℓ₀` … `ℓ₃` are expressed through `S₀(0)` (`sZT`) and `S₀(±2a)`
  (`sPT`, `sMT`).
* `FParams.hasSum_rank_two_mulT`, `hasSum_rank_twoT`, `hasSum_constT`: Lemma "rank-two sums",
  times `P(x)` (which holds also where `P(x) = 0`) and divided by it, and the constant sum.
-/

open Complex Filter Topology

namespace HubbardAtom

/-- `S₀(x) = β(Q - P)/(2(Q + P))` if `P = Q e^{β(x - μ)}`: the value of the thermal sum at a real
point through two Boltzmann weights. -/
theorem Params.S₀_of_exp (B : Params) {x P Q : ℝ} (hP : 0 < P) (hQ : 0 < Q)
    (h : P = Q * Real.exp (B.β * (x - B.μ))) :
    B.S 0 x = B.β * (Q - P) / (2 * (Q + P)) := by
  have hq : (Q : ℂ) + P ≠ 0 := by exact_mod_cast (add_pos hQ hP).ne'
  have hQ' : (Q : ℂ) ≠ 0 := ofReal_ne_zero.mpr hQ.ne'
  have hq' : (P : ℂ) + Q ≠ 0 := by rwa [add_comm]
  have he : exp (2 * (B.β * (x - B.μ) / 2 : ℂ)) = P / Q := by
    rw [eq_div_iff hQ', show 2 * ((B.β : ℂ) * (x - B.μ) / 2) = ((B.β * (x - B.μ) : ℝ) : ℂ) by
        push_cast; ring, ← ofReal_exp, ← ofReal_mul, h]
    push_cast
    ring
  rw [Params.S, ite_eq_left even_zero', tanh_eq_exp, he]
  field_simp
  ring

/-- `rank_two_pf` times `P(x)`, which needs no `P(x) ≠ 0`. -/
theorem rank_two_pf_mul {t x n₀ n₁ n₂ n₃ κ q Pt Px : ℂ} (hPt : Pt = t ^ 4 + κ * t ^ 2 + q ^ 2)
    (hPx : Px = x ^ 4 + κ * x ^ 2 + q ^ 2) (hPt0 : Pt ≠ 0) (htx : t - x ≠ 0) :
    Px * ((n₀ + n₁ * t + n₂ * t ^ 2 + n₃ * t ^ 3) / (Pt * (t - x))) =
      (Px * (n₁ + n₂ * x + n₃ * x ^ 2) -
            (n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) * (x ^ 3 + κ * x) +
          (Px * (n₂ + n₃ * x) - (n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) * (x ^ 2 + κ)) * t +
          (Px * n₃ - (n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) * x) * t ^ 2 +
          -(n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) * t ^ 3) / Pt -
        -(n₀ + n₁ * x + n₂ * x ^ 2 + n₃ * x ^ 3) / (t - x) := by
  field_simp
  rw [hPt, hPx]
  ring

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The weights relative to the zero-field atom -/

/-- `cw = p₀/B.p₀`, with `B = (U, β, μ)`. -/
noncomputable def cw : ℝ := F.p₀ / (F.lat F.μ).p₀

theorem cw_pos : 0 < F.cw := div_pos F.p₀_pos (F.lat F.μ).p₀_pos

theorem p₀_eq_cw : F.p₀ = F.cw * (F.lat F.μ).p₀ := by
  rw [cw, div_mul_cancel₀ _ (F.lat F.μ).p₀_pos.ne']

theorem p₂_eq_cw : F.p₂ = F.cw * (F.lat F.μ).p₂ := by
  rw [F.p₂_eq_p₀_mul, (F.lat F.μ).p₂_eq_p₀_mul, F.p₀_eq_cw]
  simp only [lat_β, lat_μ]
  ring

theorem ρ_eq_cw : F.ρ = F.cw * (F.lat F.μ).ρ := by
  rw [ρ, Params.ρ, F.p₀_eq_cw, F.p₂_eq_cw]
  ring

/-- `c̄ = β(p₀ + p₂)/2`. -/
noncomputable def cbarT : ℝ := F.β * (F.p₀ + F.p₂) / 2

theorem cbarT_eq_cw : F.cbarT = F.cw * (F.lat F.μ).cbar := by
  rw [cbarT, Params.cbar, F.p₀_eq_cw, F.p₂_eq_cw]
  simp only [lat_β]
  ring

/-- The thermal function `α_k(x) = ρS_k(x) - c̄`. -/
noncomputable def αT (x : ℂ) : ℂ := F.ρ * (F.lat F.μ).S k x - F.cbarT

theorem αT_eq_cw (x : ℂ) : F.αT k x = F.cw * (F.lat F.μ).α k x := by
  rw [αT, Params.α, F.ρ_eq_cw, F.cbarT_eq_cw]
  push_cast
  ring

/-- Lemma "values on the reflected lattice": `ρS_k(-zₙ) = c̄` for `μ ≠ 0`. -/
theorem rho_mul_S_neg_zT (hμ : F.μ ≠ 0) (n : ℤ) :
    (F.ρ : ℂ) * (F.lat F.μ).S k (-F.zT k n) = F.cbarT := by
  rw [F.ρ_eq_cw, F.cbarT_eq_cw, zT]
  push_cast
  rw [mul_assoc, (F.lat F.μ).rho_mul_S_neg_z k hμ n]

/-- Lemma "Cauchy calculus": `∑ₗ ρ/((zₙ + zₗ)(zₗ - x)) = α_k(x)/(zₙ + x)`. -/
theorem hasSum_cauchyT (hμ : F.μ ≠ 0) (n : ℤ) {x : ℂ} (hx : ∀ l, F.zT k l ≠ x)
    (hxn : x ≠ -F.zT k n) :
    HasSum (fun l ↦ F.ρ / (F.zT k n + F.zT k l) * (1 / (F.zT k l - x)))
      (F.αT k x / (F.zT k n + x)) := by
  have h := ((F.lat F.μ).hasSum_cauchy k hμ n hx hxn).mul_left (F.cw : ℂ)
  convert h using 1
  · funext l
    rw [F.ρ_eq_cw, zT, zT]
    push_cast
    ring
  · rw [F.αT_eq_cw, zT]
    ring

/-- Lemma "square": `∑ₗ C_nl C_lm = β²p₀p₂δₙₘ`. -/
theorem hasSum_squareT (hμ : F.μ ≠ 0) (n m : ℤ) :
    HasSum (fun l ↦ F.ρ / (F.zT k n + F.zT k l) * (F.ρ / (F.zT k l + F.zT k m)))
      (F.β ^ 2 * F.p₀ * F.p₂ * kron n m) := by
  have h := ((F.lat F.μ).hasSum_square k hμ n m).mul_left ((F.cw : ℂ) ^ 2)
  convert h using 1
  · funext l
    rw [F.ρ_eq_cw, zT, zT, zT]
    push_cast
    ring
  · rw [F.p₀_eq_cw, F.p₂_eq_cw]
    push_cast
    simp only [lat_β]
    ring

/-- Eq. (alpha): `α_k(x)α_k(-x) = β²p₀p₂`. -/
theorem αT_mul_αT_neg {x : ℂ} (hx : ∀ n, F.zT k n ≠ x) (hx' : ∀ n, F.zT k n ≠ -x) :
    F.αT k x * F.αT k (-x) = F.β ^ 2 * F.p₀ * F.p₂ := by
  rw [F.αT_eq_cw, F.αT_eq_cw, F.p₀_eq_cw, F.p₂_eq_cw]
  have h := (F.lat F.μ).α_mul_α_neg k hx hx'
  push_cast
  simp only [lat_β] at h
  linear_combination (F.cw : ℂ) ^ 2 * h

/-! ### The thermal values at the zeros of `P` -/

/-- `S_k(-a + W/2) = S₀(-a - h) = β(p↑ - p₀)/(2(p↑ + p₀))`. -/
noncomputable def sT₁ : ℂ := (F.lat F.μ).S 0 (-F.a - F.h)

/-- `S_k(-a - W/2) = S₀(-a + h) = β(p↓ - p₀)/(2(p↓ + p₀))`. -/
noncomputable def sT₂ : ℂ := (F.lat F.μ).S 0 (-F.a + F.h)

/-- `S_k(a + W/2) = S₀(a - h) = β(p₂ - p↓)/(2(p₂ + p↓))`. -/
noncomputable def sT₃ : ℂ := (F.lat F.μ).S 0 (F.a - F.h)

/-- `S_k(a - W/2) = S₀(a + h) = β(p₂ - p↑)/(2(p₂ + p↑))`. -/
noncomputable def sT₄ : ℂ := (F.lat F.μ).S 0 (F.a + F.h)

theorem sT₁_eq : F.sT₁ = F.β * (F.pUp - F.p₀) / (2 * (F.pUp + F.p₀)) := by
  have := (F.lat F.μ).S₀_of_exp (x := -F.a - F.h) F.p₀_pos F.pUp_pos (by
    rw [F.pUp_eq_p₀_mul, mul_assoc, ← Real.exp_add]
    simp only [lat_β, lat_μ]
    rw [show F.β * (F.a + F.μ + F.h) + F.β * (-F.a - F.h - F.μ) = 0 by ring, Real.exp_zero,
      mul_one])
  push_cast at this
  rw [sT₁, this]
  rfl

theorem sT₂_eq : F.sT₂ = F.β * (F.pDn - F.p₀) / (2 * (F.pDn + F.p₀)) := by
  have := (F.lat F.μ).S₀_of_exp (x := -F.a + F.h) F.p₀_pos F.pDn_pos (by
    rw [F.pDn_eq_p₀_mul, mul_assoc, ← Real.exp_add]
    simp only [lat_β, lat_μ]
    rw [show F.β * (F.a + F.μ - F.h) + F.β * (-F.a + F.h - F.μ) = 0 by ring, Real.exp_zero,
      mul_one])
  push_cast at this
  rw [sT₂, this]
  rfl

theorem sT₃_eq : F.sT₃ = F.β * (F.p₂ - F.pDn) / (2 * (F.p₂ + F.pDn)) := by
  have := (F.lat F.μ).S₀_of_exp (x := F.a - F.h) F.pDn_pos F.p₂_pos (by
    rw [F.pDn_eq_p₂_mul]
    simp only [lat_β, lat_μ]
    ring_nf)
  push_cast at this
  rw [sT₃, this]
  rfl

theorem sT₄_eq : F.sT₄ = F.β * (F.p₂ - F.pUp) / (2 * (F.p₂ + F.pUp)) := by
  have := (F.lat F.μ).S₀_of_exp (x := F.a + F.h) F.pUp_pos F.p₂_pos (by
    rw [F.pUp_eq_p₂_mul]
    simp only [lat_β, lat_μ]
    ring_nf)
  push_cast at this
  rw [sT₄, this]
  rfl

/-! ### The lattice sums of `M/P` -/

theorem WT_eq : F.WT k = (F.lat F.μ).w k - 2 * F.h := rfl

/-- The linear factors of `P(zₙ)`, the denominators of the propagators, do not vanish. -/
theorem lin₁ (j : ℤ) : (F.lat F.μ).x j + F.a + F.h ≠ 0 := by
  have := (F.lat F.μ).x_add_ofReal_ne_zero j (F.a + F.h)
  push_cast at this
  rwa [← add_assoc] at this

theorem lin₂ (j : ℤ) : (F.lat F.μ).x j - F.a + F.h ≠ 0 := by
  have := (F.lat F.μ).x_add_ofReal_ne_zero j (-F.a + F.h)
  push_cast at this
  rwa [show (F.lat F.μ).x j + (-(F.a : ℂ) + F.h) = (F.lat F.μ).x j - F.a + F.h by ring] at this

theorem lin₃ (j : ℤ) : (F.lat F.μ).x j + (F.lat F.μ).w k + F.a - F.h ≠ 0 := by
  have := (F.lat F.μ).x_add_ofReal_ne_zero (j + k) (F.a - F.h)
  rw [← (F.lat F.μ).x_add_w] at this
  push_cast at this
  rwa [← add_sub_assoc] at this

theorem lin₄ (j : ℤ) : (F.lat F.μ).x j + (F.lat F.μ).w k - F.a - F.h ≠ 0 := by
  have := (F.lat F.μ).x_add_ofReal_ne_zero (j + k) (-F.a - F.h)
  rw [← (F.lat F.μ).x_add_w] at this
  push_cast at this
  rwa [show (F.lat F.μ).x j + (F.lat F.μ).w k + (-(F.a : ℂ) - F.h) =
    (F.lat F.μ).x j + (F.lat F.μ).w k - F.a - F.h by ring] at this

/-- `P(zₙ)` through its linear factors. -/
theorem PT_zT (j : ℤ) : F.PT k (F.zT k j) = PTf F.a F.h ((F.lat F.μ).x j) ((F.lat F.μ).w k) := by
  simp only [PT, QpT, QmT, zT, Params.z, WT_eq, PTf]
  ring

/-- `P` has no zeros on the lattice. -/
theorem PT_zT_ne_zero (j : ℤ) : F.PT k (F.zT k j) ≠ 0 := by
  rw [PT_zT, PTf]
  exact mul_ne_zero (mul_ne_zero (F.lin₁ j) (F.lin₃ k j)) (mul_ne_zero (F.lin₂ j) (F.lin₄ k j))

/-- `PV ∑ₗ (zₗ + c)/((zₗ + c)² - W²/4) = (S₀(-c - h) + S₀(-c + h))/2`, made absolutely convergent
by subtracting `1/(zₗ - y)`, for real `c`. -/
theorem hasSum_Q_subT (c : ℝ) {y : ℂ} (hy : ∀ n, F.zT k n ≠ y) :
    HasSum (fun l ↦ (F.zT k l + c) / ((F.zT k l + c) ^ 2 - F.WT k ^ 2 / 4) - 1 / (F.zT k l - y))
      (((F.lat F.μ).S 0 (-c - F.h) + (F.lat F.μ).S 0 (-c + F.h)) / 2 - (F.lat F.μ).S k y) := by
  set B := F.lat F.μ
  have hS₁ : B.S k (-((c + F.h : ℝ) : ℂ) + B.w k / 2) = B.S 0 (-c - F.h) := by
    rw [B.S_shift]
    push_cast
    ring_nf
  have hS₂ : B.S k (-((c - F.h : ℝ) : ℂ) - B.w k / 2) = B.S 0 (-c + F.h) := by
    rw [B.S_shift']
    push_cast
    ring_nf
  have h := ((B.hasSum_sub_inv k (B.z_ne_neg_add k (c + F.h)) hy).add
    (B.hasSum_sub_inv k (B.z_ne_neg_sub k (c - F.h)) hy)).mul_left (1 / 2)
  rw [hS₁, hS₂] at h
  convert h using 1
  · funext l
    have h₁ := sub_ne_zero.mpr (B.z_ne_neg_add k (c + F.h) l)
    have h₂ := sub_ne_zero.mpr (B.z_ne_neg_sub k (c - F.h) l)
    rw [show F.WT k ^ 2 / 4 = (B.w k / 2 - F.h) ^ 2 by rw [WT_eq]; ring]
    have e₁ : B.z k l - (-((c + F.h : ℝ) : ℂ) + B.w k / 2) =
        F.zT k l - (-c + (B.w k / 2 - F.h)) := by
      rw [zT]; push_cast; ring
    have e₂ : B.z k l - (-((c - F.h : ℝ) : ℂ) - B.w k / 2) =
        F.zT k l - (-c - (B.w k / 2 - F.h)) := by
      rw [zT]; push_cast; ring
    rw [e₁] at h₁ ⊢
    rw [e₂] at h₂ ⊢
    exact pf_Q_sub h₁ h₂ (sub_ne_zero.mpr (hy l))
  · ring

/-- `∑ₗ 1/((zₗ + c)² - W²/4) = (S₀(-c - h) - S₀(-c + h))/W` for real `c` and `W ≠ 0`. -/
theorem hasSum_inv_QT (hW : F.WT k ≠ 0) (c : ℝ) :
    HasSum (fun l ↦ 1 / ((F.zT k l + c) ^ 2 - F.WT k ^ 2 / 4))
      (((F.lat F.μ).S 0 (-c - F.h) - (F.lat F.μ).S 0 (-c + F.h)) / F.WT k) := by
  set B := F.lat F.μ
  have hS₁ : B.S k (-((c + F.h : ℝ) : ℂ) + B.w k / 2) = B.S 0 (-c - F.h) := by
    rw [B.S_shift]
    push_cast
    ring_nf
  have hS₂ : B.S k (-((c - F.h : ℝ) : ℂ) - B.w k / 2) = B.S 0 (-c + F.h) := by
    rw [B.S_shift']
    push_cast
    ring_nf
  have h := (B.hasSum_sub_inv k (B.z_ne_neg_add k (c + F.h)) (B.z_ne_neg_sub k (c - F.h))).mul_left
    (1 / F.WT k)
  rw [hS₁, hS₂] at h
  convert h using 1
  · funext l
    have h₁ := sub_ne_zero.mpr (B.z_ne_neg_add k (c + F.h) l)
    have h₂ := sub_ne_zero.mpr (B.z_ne_neg_sub k (c - F.h) l)
    have e₁ : B.z k l - (-((c + F.h : ℝ) : ℂ) + B.w k / 2) = F.zT k l - (-c + F.WT k / 2) := by
      rw [zT, WT_eq]; push_cast; ring
    have e₂ : B.z k l - (-((c - F.h : ℝ) : ℂ) - B.w k / 2) = F.zT k l - (-c - F.WT k / 2) := by
      rw [zT, WT_eq]; push_cast; ring
    rw [e₁] at h₁ ⊢
    rw [e₂] at h₂ ⊢
    rw [show F.WT k ^ 2 / 4 = (F.WT k / 2) ^ 2 by ring,
      show 1 / F.WT k = 1 / (2 * (F.WT k / 2)) by ring]
    exact pf_inv_Q (div_ne_zero hW two_ne_zero) h₁ h₂
  · ring

/-- `S₊ = (S_k(-a + W/2) + S_k(-a - W/2))/2`, the mean value at the zeros of `Q₊`. -/
noncomputable def SpT : ℂ := (F.sT₁ + F.sT₂) / 2

/-- `S₋ = (S_k(a + W/2) + S_k(a - W/2))/2`, the mean value at the zeros of `Q₋`. -/
noncomputable def SmT : ℂ := (F.sT₃ + F.sT₄) / 2

/-- `T₊ = ∑ₗ 1/Q₊(zₗ)`. -/
noncomputable def TpT : ℂ := (F.sT₁ - F.sT₂) / F.WT k

/-- `T₋ = ∑ₗ 1/Q₋(zₗ)`. -/
noncomputable def TmT : ℂ := (F.sT₃ - F.sT₄) / F.WT k

/-- `S₀(0)`, `S₀(2a)`, `S₀(-2a)`: at `q₀ = 0` (`k = 0`, `h = ±a`) the zeros of
`P(t) = t²(t² - 4a²)` are `0` (double) and `±2a`. -/
noncomputable def sZT : ℂ := (F.lat F.μ).S 0 0

noncomputable def sPT : ℂ := (F.lat F.μ).S 0 (2 * F.a)

noncomputable def sMT : ℂ := (F.lat F.μ).S 0 (-(2 * F.a))

/-- `ℓ₀ = ∑ₗ 1/P(zₗ)`; at `q₀ = 0`, `(T₂ - T₀)/(4a²)` with `T₂ = ∑ₗ 1/(zₗ² - 4a²)` and
`T₀ = ∑ₗ 1/zₗ² = S₀(0)² - β²/4`. -/
noncomputable def ℓ₀T : ℂ :=
  if F.q₀T k = 0 then ((F.sPT - F.sMT) / (4 * F.a) - (F.sZT ^ 2 - F.β ^ 2 / 4)) / (4 * F.a ^ 2)
  else (F.SpT - F.SmT) / (4 * F.a * F.q₀T k) + (F.TpT k + F.TmT k) / (4 * F.q₀T k)

/-- `ℓ₁ = ∑ₗ zₗ/P(zₗ)`. -/
noncomputable def ℓ₁T : ℂ :=
  if F.q₀T k = 0 then (F.sPT + F.sMT - 2 * F.sZT) / (8 * F.a ^ 2)
  else (F.TmT k - F.TpT k) / (4 * F.a)

/-- `ℓ₂ = ∑ₗ zₗ²/P(zₗ)`. -/
noncomputable def ℓ₂T : ℂ :=
  if F.q₀T k = 0 then (F.sPT - F.sMT) / (4 * F.a)
  else (F.SmT - F.SpT) / (4 * F.a) + (F.TpT k + F.TmT k) / 4

/-- `ℓ₃ = PV ∑ₗ zₗ³/P(zₗ)`. -/
noncomputable def ℓ₃T : ℂ :=
  if F.q₀T k = 0 then (F.sPT + F.sMT) / 2
  else (F.SpT + F.SmT) / 2 + (F.q₀T k - 2 * F.a ^ 2) * (F.TpT k - F.TmT k) / (4 * F.a)

/-- `q₀ = a² - W²/4` vanishes only at `k = 0`, `h = ±a`. -/
theorem q₀T_eq_zero (ha : (F.a : ℂ) ≠ 0) (hq : F.q₀T k = 0) : k = 0 ∧ (F.h = F.a ∨ F.h = -F.a) := by
  have ha' : F.a ≠ 0 := by exact_mod_cast ha
  have hre := congrArg Complex.re hq
  have him := congrArg Complex.im hq
  simp [q₀T, WT, pow_two] at hre him
  have hω : F.ω k = 0 := by
    by_contra hω
    have hh : F.h = 0 := by
      have : F.h * F.ω k = 0 := by linarith
      exact (mul_eq_zero.mp this).resolve_right hω
    rw [hh] at hre
    nlinarith [sq_nonneg (F.ω k), mul_self_pos.mpr ha']
  refine ⟨?_, ?_⟩
  · have hβ := F.β_pos.ne'
    simp only [ω, Params.ω, lat_β] at hω
    have : (k : ℝ) = 0 := by
      rcases mul_eq_zero.mp ((div_eq_zero_iff.mp hω).resolve_right hβ) with h0 | h0
      · rcases mul_eq_zero.mp h0 with h1 | h1
        · norm_num at h1
        · exact h1
      · exact absurd h0 Real.pi_ne_zero
    exact_mod_cast this
  · rw [hω] at hre
    have : (F.h - F.a) * (F.h + F.a) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h0 | h0
    · exact Or.inl (by linarith)
    · exact Or.inr (by linarith)

/-- The lattice sum of `M/P` at `q₀ = 0`, through `P(t) = t²(t² - 4a²)`. -/
private theorem hasSum_cubicT_zero (ha : (F.a : ℂ) ≠ 0) (hq : F.q₀T k = 0)
    (m₀ m₁ m₂ m₃ : ℂ) {y : ℂ} (hy : ∀ n, F.zT k n ≠ y) :
    HasSum (fun l ↦ (m₀ + m₁ * F.zT k l + m₂ * F.zT k l ^ 2 + m₃ * F.zT k l ^ 3) /
        F.PT k (F.zT k l) - m₃ / (F.zT k l - y))
      (m₀ * F.ℓ₀T k + m₁ * F.ℓ₁T k + m₂ * F.ℓ₂T k + m₃ * (F.ℓ₃T k - (F.lat F.μ).S k y)) := by
  obtain ⟨rfl, -⟩ := F.q₀T_eq_zero k ha hq
  have hz : ∀ (r : ℝ) (n : ℤ), (F.lat F.μ).z 0 n ≠ r := fun r n h0 ↦
    (F.lat F.μ).x_sub_ofReal_ne_zero n r (by rw [← h0, Params.z, (F.lat F.μ).w_zero]; ring)
  have h₀ : ∀ n, (F.lat F.μ).z 0 n ≠ 0 := fun n ↦ by simpa using hz 0 n
  have hp : ∀ n, (F.lat F.μ).z 0 n ≠ 2 * F.a := fun n ↦ by simpa using hz (2 * F.a) n
  have hm : ∀ n, (F.lat F.μ).z 0 n ≠ -(2 * F.a) := fun n ↦ by simpa using hz (-(2 * F.a)) n
  have s₁ := (F.lat F.μ).hasSum_sq (k := 0) h₀
  have s₂ := (F.lat F.μ).hasSum_sub_inv 0 h₀ hy
  have s₃ := (F.lat F.μ).hasSum_sub_inv 0 hp hy
  have s₄ := (F.lat F.μ).hasSum_sub_inv 0 hm hy
  have hW2 : F.WT 0 ^ 2 / 4 = (F.a : ℂ) ^ 2 := by rw [q₀T] at hq; linear_combination -hq
  have hPz : ∀ t, F.PT 0 t = t ^ 2 * ((t - 2 * F.a) * (t + 2 * F.a)) := fun t ↦ by
    rw [PT, QpT, QmT, hW2]
    ring
  set M : ℂ → ℂ := fun t ↦ m₀ + m₁ * t + m₂ * t ^ 2 + m₃ * t ^ 3 with hM
  set C := M (2 * F.a) / (16 * F.a ^ 3) with hC
  set D := -M (-(2 * F.a)) / (16 * F.a ^ 3) with hD
  convert (((s₁.mul_left (-m₀ / (4 * F.a ^ 2))).add (s₂.mul_left (m₃ - C - D))).add
    (s₃.mul_left C)).add (s₄.mul_left D) using 1
  · funext l
    have e₀ := h₀ l
    have e₁ := sub_ne_zero.mpr (hp l)
    have e₂ : (F.lat F.μ).z 0 l + 2 * F.a ≠ 0 := fun h0 ↦ hm l (by linear_combination h0)
    have e₃ := sub_ne_zero.mpr (hy l)
    rw [hPz, hC, hD, hM]
    simp only [zT, sub_zero, sub_neg_eq_add]
    field_simp
    ring
  · simp only [ℓ₀T, ℓ₁T, ℓ₂T, ℓ₃T, hq, ite_true, sZT, sPT, sMT, hC, hD, hM, lat_β]
    field_simp
    ring

/-- The lattice sum of `M/P` for a cubic `M`, made absolutely convergent by subtracting
`m₃/(t - y)`, for `a ≠ 0` and `W ≠ 0`; for `q₀ ≠ 0` the zeros of `P` are simple, and
`P = Q₊Q₋` with coprime factors. -/
theorem hasSum_cubicT (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0)
    (m₀ m₁ m₂ m₃ : ℂ) {y : ℂ} (hy : ∀ n, F.zT k n ≠ y) :
    HasSum (fun l ↦ (m₀ + m₁ * F.zT k l + m₂ * F.zT k l ^ 2 + m₃ * F.zT k l ^ 3) /
        F.PT k (F.zT k l) - m₃ / (F.zT k l - y))
      (m₀ * F.ℓ₀T k + m₁ * F.ℓ₁T k + m₂ * F.ℓ₂T k + m₃ * (F.ℓ₃T k - (F.lat F.μ).S k y)) := by
  by_cases hq : F.q₀T k = 0
  · exact F.hasSum_cubicT_zero k ha hq m₀ m₁ m₂ m₃ hy
  · have hq' : (F.a : ℂ) ^ 2 - (F.WT k / 2) ^ 2 ≠ 0 := by
      rw [show (F.a : ℂ) ^ 2 - (F.WT k / 2) ^ 2 = F.q₀T k by rw [q₀T]; ring]
      exact hq
    have hp₁ := F.hasSum_Q_subT k F.a hy
    have hp₂ := F.hasSum_Q_subT k (-F.a) hy
    have hp₃ := F.hasSum_inv_QT k hW F.a
    have hp₄ := F.hasSum_inv_QT k hW (-F.a)
    simp only [ofReal_neg, ← sub_eq_add_neg, neg_neg] at hp₂ hp₄
    set da := (m₀ / ((F.a : ℂ) ^ 2 - (F.WT k / 2) ^ 2) - m₂) / (2 * F.a) with hda
    set sb := (m₂ + m₀ / ((F.a : ℂ) ^ 2 - (F.WT k / 2) ^ 2)) / 2 with hsb
    set db := (((F.a : ℂ) ^ 2 - (F.WT k / 2) ^ 2 - 2 * F.a ^ 2) * m₃ - m₁) / (2 * F.a) with hdb
    convert (((hp₁.mul_left ((m₃ + da) / 2)).add (hp₂.mul_left ((m₃ - da) / 2))).add
      (hp₃.mul_left ((sb + db) / 2))).add (hp₄.mul_left ((sb - db) / 2)) using 1
    · funext l
      have hQp : (F.zT k l + F.a) ^ 2 - (F.WT k / 2) ^ 2 ≠ 0 := by
        rw [show (F.zT k l + F.a) ^ 2 - (F.WT k / 2) ^ 2 =
            ((F.lat F.μ).x l + F.a + F.h) * ((F.lat F.μ).x l + (F.lat F.μ).w k + F.a - F.h) by
          rw [zT, Params.z, WT_eq]; ring]
        exact mul_ne_zero (F.lin₁ l) (F.lin₃ k l)
      have hQm : (F.zT k l - F.a) ^ 2 - (F.WT k / 2) ^ 2 ≠ 0 := by
        rw [show (F.zT k l - F.a) ^ 2 - (F.WT k / 2) ^ 2 =
            ((F.lat F.μ).x l - F.a + F.h) * ((F.lat F.μ).x l + (F.lat F.μ).w k - F.a - F.h) by
          rw [zT, Params.z, WT_eq]; ring]
        exact mul_ne_zero (F.lin₂ l) (F.lin₄ k l)
      rw [PT, QpT, QmT]
      rw [show F.WT k ^ 2 / 4 = (F.WT k / 2) ^ 2 by ring]
      exact cubic_pf hda hsb hdb ha hq' hQp hQm (sub_ne_zero.mpr (hy l))
    · simp only [ℓ₀T, ℓ₁T, ℓ₂T, ℓ₃T, hq, ite_false]
      rw [TpT, TmT, SpT, SmT, sT₁, sT₂, sT₃, sT₄, hda, hsb, hdb]
      rw [q₀T] at hq ⊢
      generalize (F.lat F.μ).S 0 (-F.a - F.h) = S₁
      generalize (F.lat F.μ).S 0 (-F.a + F.h) = S₂
      generalize (F.lat F.μ).S 0 (F.a - F.h) = S₃
      generalize (F.lat F.μ).S 0 (F.a + F.h) = S₄
      generalize (F.lat F.μ).S k y = Sy
      have hq4 : (F.a : ℂ) ^ 2 * 4 - F.WT k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
      field_simp
      ring

/-! ### The rank-two sums -/

/-- `κ = 2q₀ - 4a²`, so that `P(t) = t⁴ + κt² + q₀²`. -/
noncomputable def κT : ℂ := 2 * F.q₀T k - 4 * F.a ^ 2

/-- The coefficient of `t³` in `N(z, t)`: `-λρ = 4a²ρ`. -/
noncomputable def cN₃T : ℂ := 4 * F.a ^ 2 * F.ρ

/-- The coefficient of `t²` in `N(z, t)`. -/
noncomputable def cN₂T (z : ℂ) : ℂ :=
  (2 * F.a - F.ζT k) * F.q₀T k + z ^ 2 * (2 * F.a + F.ζT k)

/-- The coefficient of `t` in `N(z, t)`. -/
noncomputable def cN₁T (z : ℂ) : ℂ := (F.κT k + z ^ 2) * F.cN₃T

/-- The coefficient of `1` in `N(z, t)`. -/
noncomputable def cN₀T (z : ℂ) : ℂ :=
  (F.κT k + z ^ 2) * ((2 * F.a - F.ζT k) * F.q₀T k) - (2 * F.a + F.ζT k) * F.q₀T k ^ 2

theorem PT_eq_poly (t : ℂ) : F.PT k t = t ^ 4 + F.κT k * t ^ 2 + F.q₀T k ^ 2 := by
  rw [PT, QpT, QmT, κT, q₀T]
  ring

theorem NT_eq_coeffs (z t : ℂ) :
    F.NT k z t = F.cN₀T k z + F.cN₁T k z * t + F.cN₂T k z * t ^ 2 + F.cN₃T * t ^ 3 := by
  rw [NT, N₀T, N₂T, F.PT_eq_poly, cN₀T, cN₁T, cN₂T, cN₃T, κT]
  ring

/-- `P(x)Ī(z; x) - c̄N(z, x)/ρ`: the part of the rank-two sum that comes from the zeros of `P`. -/
noncomputable def RhatT (z x : ℂ) : ℂ :=
  (F.PT k x * (F.cN₁T k z + F.cN₂T k z * x + F.cN₃T * x ^ 2) -
        F.NT k z x * (x ^ 3 + F.κT k * x)) * F.ℓ₀T k +
      (F.PT k x * (F.cN₂T k z + F.cN₃T * x) - F.NT k z x * (x ^ 2 + F.κT k)) * F.ℓ₁T k +
    (F.PT k x * F.cN₃T - F.NT k z x * x) * F.ℓ₂T k - F.NT k z x * F.ℓ₃T k

/-- Lemma "rank-two sums" times `P(x)`, which holds also where `P(x) = 0`:
`∑ₗ P(x)N(z, zₗ)/(P(zₗ)(zₗ - x)) = N(z, x)S_k(x) + R̂(z, x)`. -/
theorem hasSum_rank_two_mulT (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0)
    (z : ℂ) {x : ℂ} (hx : ∀ l, F.zT k l ≠ x) :
    HasSum (fun l ↦ F.PT k x * (F.NT k z (F.zT k l) / (F.PT k (F.zT k l) * (F.zT k l - x))))
      (F.NT k z x * (F.lat F.μ).S k x + F.RhatT k z x) := by
  convert F.hasSum_cubicT k ha hW
    (F.PT k x * (F.cN₁T k z + F.cN₂T k z * x + F.cN₃T * x ^ 2) -
      F.NT k z x * (x ^ 3 + F.κT k * x))
    (F.PT k x * (F.cN₂T k z + F.cN₃T * x) - F.NT k z x * (x ^ 2 + F.κT k))
    (F.PT k x * F.cN₃T - F.NT k z x * x) (-F.NT k z x) hx using 1
  · funext l
    rw [F.NT_eq_coeffs, F.NT_eq_coeffs]
    exact rank_two_pf_mul (F.PT_eq_poly k _) (F.PT_eq_poly k x) (F.PT_zT_ne_zero k l)
      (sub_ne_zero.mpr (hx l))
  · rw [RhatT]
    ring

/-- Lemma "rank-two sums": `∑ₗ N(z, zₗ)/(P(zₗ)(zₗ - x)) = (N(z, x)S_k(x) + R̂(z, x))/P(x)`. -/
theorem hasSum_rank_twoT (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (z : ℂ)
    {x : ℂ} (hx : ∀ l, F.zT k l ≠ x) (hPx : F.PT k x ≠ 0) :
    HasSum (fun l ↦ F.NT k z (F.zT k l) / (F.PT k (F.zT k l) * (F.zT k l - x)))
      ((F.NT k z x * (F.lat F.μ).S k x + F.RhatT k z x) / F.PT k x) := by
  convert (F.hasSum_rank_two_mulT k ha hW z hx).div_const (F.PT k x) using 1
  funext l
  field_simp

/-- The constant sum of Step 2: `∑ₗ [λρ/(zₙ + zₗ) + N(zₙ, zₗ)/P(zₗ)] = λc̄ + ∑ⱼ cNⱼ(zₙ)ℓⱼ`. -/
theorem hasSum_constT (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0)
    (hμ : F.μ ≠ 0) (n : ℤ) :
    HasSum (fun l ↦ F.lamT * F.ρ / (F.zT k n + F.zT k l) +
        F.NT k (F.zT k n) (F.zT k l) / F.PT k (F.zT k l))
      (F.lamT * F.cbarT + (F.cN₀T k (F.zT k n) * F.ℓ₀T k + F.cN₁T k (F.zT k n) * F.ℓ₁T k +
        F.cN₂T k (F.zT k n) * F.ℓ₂T k + F.cN₃T * F.ℓ₃T k)) := by
  have h := F.hasSum_cubicT k ha hW (F.cN₀T k (F.zT k n)) (F.cN₁T k (F.zT k n))
    (F.cN₂T k (F.zT k n)) F.cN₃T (fun l ↦ (F.lat F.μ).z_ne_neg_z k hμ n l)
  convert h using 1
  · funext l
    rw [F.NT_eq_coeffs, cN₃T, lamT, sub_neg_eq_add, add_comm (F.zT k l)]
    simp only [zT]
    ring
  · rw [← F.rho_mul_S_neg_zT k hμ n, cN₃T, lamT, zT]
    ring

end FParams

end HubbardAtom
