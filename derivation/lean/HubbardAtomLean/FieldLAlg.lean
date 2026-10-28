import HubbardAtomLean.FieldLSums

/-!
# The algebra of the longitudinal inverse

The data of Eq. (field-L) of the note (App. F.6.4) for a column of spin `t`, with the moving-pole
system that fixes the dense part of the inverse.

* `FParams.E8L`, `QL`: the polynomials `E₈` and `Q_t` of Eq. (field-L-data).
* `FParams.dhL`, `AhL`, `BhL`, `ChL`: the coefficients `d_t`, `A_t`, `B_t`, `C_t` of the column
  `(t, v)` of the inverse, divided by `P(v)`.
* `FParams.dense_delta`, `dense_pole`, `dense_pole_v`, `dense_pole_neg_v`: the moving-pole system
  `𝓑_t(v)(d, â, b̂, ĉ) = P(v)e₁`: the delta term of the row `t` is `P(v)`, and the poles at
  `z = -v` (row `t`) and `z = ±v` (row `-t`) cancel.
* `FParams.fMp`, `fMm`, `KL`: the pole-pair mode of a root `u` of `E₈`, normalized with the column
  spin `t` (Eq. (field-L-mode) has `t = ↑`), and its thermal denominators `K_{t'} = f_{t'+} + f_{t'-}`.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The polynomials -/

/-- `π_P = p₀p₂`. -/
noncomputable def piP : ℂ := F.p₀ * F.p₂

/-- `π_S = p↑p↓`. -/
noncomputable def piS : ℂ := F.pUp * F.pDn

/-- `E₈(x)` of Eq. (field-L-data). -/
noncomputable def E8L (x : ℂ) : ℂ :=
  F.HL k F.rUp x * F.HL k F.rUp (-x) * F.HL k F.rDn x * F.HL k F.rDn (-x) -
      F.lamL ^ 2 * (F.piS * (F.HL k F.rUp x * F.HL k F.rDn x +
          F.HL k F.rUp (-x) * F.HL k F.rDn (-x)) +
        F.piP * (F.HL k F.rUp x * F.HL k F.rDn (-x) + F.HL k F.rUp (-x) * F.HL k F.rDn x)) +
    F.lamL ^ 4 * (F.piP - F.piS) ^ 2

/-- `Q_t(x) = H_t(-x)H_{-t}(x)H_{-t}(-x) - λ²[π_P H_{-t}(-x) + π_S H_{-t}(x)]`. -/
noncomputable def QL (t : Bool) (x : ℂ) : ℂ :=
  F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) x * F.HL k (F.rL (!t)) (-x) -
    F.lamL ^ 2 * (F.piP * F.HL k (F.rL (!t)) (-x) + F.piS * F.HL k (F.rL (!t)) x)

/-- `E₈` is symmetric in the spins. -/
theorem E8L_eq (t : Bool) (x : ℂ) :
    F.E8L k x = F.HL k (F.rL t) x * F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) x *
          F.HL k (F.rL (!t)) (-x) -
        F.lamL ^ 2 * (F.piS * (F.HL k (F.rL t) x * F.HL k (F.rL (!t)) x +
            F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) (-x)) +
          F.piP * (F.HL k (F.rL t) x * F.HL k (F.rL (!t)) (-x) +
            F.HL k (F.rL t) (-x) * F.HL k (F.rL (!t)) x)) +
      F.lamL ^ 4 * (F.piP - F.piS) ^ 2 := by
  cases t
  · simp only [Bool.not_false, rL_false, rL_true, E8L]
    ring
  · simp only [Bool.not_true, rL_false, rL_true, E8L]

theorem E8L_neg (x : ℂ) : F.E8L k (-x) = F.E8L k x := by
  rw [E8L, E8L, neg_neg]
  ring

/-! ### The column of the inverse -/

/-- `d̂ = -Q_t(v)/(βE₈(v))`: the delta coefficient of the column `(t, v)`, divided by `P(v)`. -/
noncomputable def dhL (t : Bool) (v : ℂ) : ℂ := -(F.QL k t v / (F.β * F.E8L k v))

/-- `Â = 2λ²D_t(v² + q₀)/(β²E₈(v))`: the coefficient of `1/(z + v)` in row `t`, over `P(v)`. -/
noncomputable def AhL (t : Bool) (v : ℂ) : ℂ :=
  2 * F.lamL ^ 2 * F.DL t * (v ^ 2 + F.q₀L k) / (F.β ^ 2 * F.E8L k v)

/-- `B̂`: the coefficient of `1/(z - v)` in row `-t`, over `P(v)`. -/
noncomputable def BhL (t : Bool) (v : ℂ) : ℂ :=
  -(F.mL t * F.lamL * (F.HL k (F.rL t) (-v) * F.HL k (F.rL (!t)) (-v) +
      F.lamL ^ 2 * (F.piP - F.piS)) / (F.β ^ 2 * F.E8L k v))

/-- `Ĉ`: the coefficient of `1/(z + v)` in row `-t`, over `P(v)`. -/
noncomputable def ChL (t : Bool) (v : ℂ) : ℂ :=
  -(F.ρ * F.lamL * (F.HL k (F.rL t) (-v) * F.HL k (F.rL (!t)) v -
      F.lamL ^ 2 * (F.piP - F.piS)) / (F.β ^ 2 * F.E8L k v))

/-! ### The weights of a column -/

theorem mL_not (t : Bool) : F.mL (!t) = -F.mL t := by
  simp only [mL, Bool.not_not]
  ring

theorem rL_not (t : Bool) : F.rL (!t) = F.ρ + F.mL t := by
  rw [rL, mL_not]
  ring

theorem rL_cast (t : Bool) : (F.rL t : ℂ) = F.ρ - F.mL t := by
  rw [rL]
  push_cast
  ring

theorem piS_eq (t : Bool) : F.piS = F.pL t * F.pL (!t) := by
  cases t
  · simp only [piS, pL_false, Bool.not_false, pL_true]
    ring
  · simp only [piS, pL_true, Bool.not_true, pL_false]

theorem cS_eq (t : Bool) : (F.cS : ℂ) = F.β * (F.pL t + F.pL (!t)) / 2 := by
  rw [cS, F.pL_add t]
  push_cast
  ring

/-! ### The moving-pole system -/

variable {F k} in
/-- The weights of a column through `pₜ = p_t`, `pₙ = p_{-t}`: the hypotheses under which the
identities of the moving-pole system are proved, after `generalize`. -/
theorem weights_col (t : Bool) :
    (F.ρ : ℂ) = F.p₂ - F.p₀ ∧ (F.mL t : ℂ) = F.pL t - F.pL (!t) ∧
      (F.cS : ℂ) = F.β * (F.pL t + F.pL (!t)) / 2 ∧ (F.cP : ℂ) = F.β * (F.p₀ + F.p₂) / 2 ∧
      F.piS = F.pL t * F.pL (!t) ∧ F.piP = F.p₀ * F.p₂ ∧
      (F.DL t : ℂ) = F.p₂ * F.pL t - F.p₀ * F.pL (!t) ∧
      (F.cLw t : ℂ) = F.p₂ * F.pL t + F.p₀ * F.pL (!t) := by
  refine ⟨by rw [ρ]; push_cast; ring, by rw [mL]; push_cast; ring, F.cS_eq t,
    by rw [cP]; push_cast; ring, F.piS_eq t, rfl, by rw [DL]; push_cast; ring,
    by rw [cLw]; push_cast; ring⟩

/-- **Moving-pole system, delta term** (row `t`, `z = v`):
`-βH_t(v)d̂ + λ[σm(S² - β²/4)B̂ + ρ(S'² - β²/4)Ĉ] = 1` with `S = -c_S/(σm)`, `S' = c_P/ρ`. -/
theorem dense_delta (t : Bool) {v : ℂ} (hE : F.E8L k v ≠ 0) (hm : (F.mL t : ℂ) ≠ 0)
    (hρ : (F.ρ : ℂ) ≠ 0) :
    -F.β * F.HL k (F.rL t) v * F.dhL k t v +
        F.lamL * (F.BhL k t v * (F.mL t * ((F.cS / F.mL t) ^ 2 - F.β ^ 2 / 4)) +
          F.ChL k t v * (F.ρ * ((F.cP / F.ρ) ^ 2 - F.β ^ 2 / 4))) = 1 := by
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨e₁, e₂, e₃, e₄, e₅, e₆, -, -⟩ := weights_col (F := F) t
  simp only [dhL, BhL, ChL]
  field_simp
  rw [F.E8L_eq k t]
  simp only [QL, rL_not, HL, lamL]
  push_cast
  simp only [rL_cast, e₁, e₂, e₃, e₄, e₅, e₆]
  ring

/-- **Moving-pole system, pole at `z = -v`** (row `t`):
`-βH_t(-v)Â + λ[(-ρc_S/(σm) - c_P)B̂ + σm(-c_S/(σm) - c_P/ρ)Ĉ] = 0`. -/
theorem dense_pole (t : Bool) {v : ℂ} (hm : (F.mL t : ℂ) ≠ 0)
    (hρ : (F.ρ : ℂ) ≠ 0) :
    -F.β * F.HL k (F.rL t) (-v) * F.AhL k t v +
        F.lamL * (F.BhL k t v * (-(F.ρ * F.cS / F.mL t) - F.cP) +
          F.ChL k t v * (F.mL t * (-(F.cS / F.mL t) - F.cP / F.ρ))) = 0 := by
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨e₁, e₂, e₃, e₄, e₅, e₆, e₇, -⟩ := weights_col (F := F) t
  simp only [AhL, BhL, ChL, F.E8L_eq k t, rL_not, HL, lamL]
  push_cast
  simp only [rL_cast]
  simp only [e₁, e₂, e₃, e₄, e₅, e₆, e₇] at hm hρ ⊢
  generalize (F.p₀ : ℂ) = p₀ at *
  generalize (F.p₂ : ℂ) = p₂ at *
  generalize (F.pL t : ℂ) = pt at *
  generalize (F.pL (!t) : ℂ) = pn at *
  field_simp
  ring

/-- **Moving-pole system, pole at `z = v`** (row `-t`):
`-βH_{-t}(v)B̂ + λ[σm d̂ + (ρS_t(-v) - c_P)Â] = 0` with `S_t(-v) = βc_t/(2D_t)`. -/
theorem dense_pole_v (t : Bool) {v : ℂ} (hD : (F.DL t : ℂ) ≠ 0) :
    -F.β * F.HL k (F.rL (!t)) v * F.BhL k t v +
        F.lamL * (F.dhL k t v * F.mL t +
          F.AhL k t v * (F.ρ * (F.β * F.cLw t / (2 * F.DL t)) - F.cP)) = 0 := by
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨e₁, e₂, e₃, e₄, e₅, e₆, e₇, e₈⟩ := weights_col (F := F) t
  simp only [dhL, AhL, BhL, QL, F.E8L_eq k t, rL_not, HL, lamL]
  push_cast
  simp only [rL_cast]
  simp only [e₁, e₂, e₄, e₅, e₆, e₇, e₈] at hD ⊢
  generalize (F.p₀ : ℂ) = p₀ at *
  generalize (F.p₂ : ℂ) = p₂ at *
  generalize (F.pL t : ℂ) = pt at *
  generalize (F.pL (!t) : ℂ) = pn at *
  field_simp
  ring

/-- **Moving-pole system, pole at `z = -v`** (row `-t`):
`-βH_{-t}(-v)Ĉ + λ[ρd̂ + (σmS_t(-v) - c_S)Â] = 0`. -/
theorem dense_pole_neg_v (t : Bool) {v : ℂ} (hD : (F.DL t : ℂ) ≠ 0) :
    -F.β * F.HL k (F.rL (!t)) (-v) * F.ChL k t v +
        F.lamL * (F.dhL k t v * F.ρ +
          F.AhL k t v * (F.mL t * (F.β * F.cLw t / (2 * F.DL t)) - F.cS)) = 0 := by
  have hβ := β_ne_zeroT (F := F)
  obtain ⟨e₁, e₂, e₃, e₄, e₅, e₆, e₇, e₈⟩ := weights_col (F := F) t
  simp only [dhL, AhL, ChL, QL, F.E8L_eq k t, rL_not, HL, lamL]
  push_cast
  simp only [rL_cast]
  simp only [e₁, e₂, e₃, e₅, e₆, e₇, e₈] at hD ⊢
  generalize (F.p₀ : ℂ) = p₀ at *
  generalize (F.p₂ : ℂ) = p₂ at *
  generalize (F.pL t : ℂ) = pt at *
  generalize (F.pL (!t) : ℂ) = pn at *
  field_simp
  ring

/-! ### The mode of a root -/

/-- `f_{σ+}` of the mode of a root `u` of `E₈`, normalized with the column spin `t`:
`-βQ_t(u)` on the spin `t`, `-λ[H_t(-u)H_{-t}(-u) + λ²d_π]α^S_{-t}(u)` on the spin `-t`. -/
noncomputable def fMp (t σ : Bool) (u : ℂ) : ℂ :=
  if σ = t then -F.β * F.QL k t u
  else -(F.lamL * (F.HL k (F.rL t) (-u) * F.HL k (F.rL (!t)) (-u) +
    F.lamL ^ 2 * (F.piP - F.piS))) * F.αSL k (!t) u

/-- `f_{σ-}`: `2λ²(u² + q₀)α_t(u)` on the spin `t`, `-λ[H_t(-u)H_{-t}(u) - λ²d_π]α^P_{-t}(u)` on
the spin `-t`. -/
noncomputable def fMm (t σ : Bool) (u : ℂ) : ℂ :=
  if σ = t then 2 * F.lamL ^ 2 * (u ^ 2 + F.q₀L k) * F.αL k t u
  else -(F.lamL * (F.HL k (F.rL t) (-u) * F.HL k (F.rL (!t)) u -
    F.lamL ^ 2 * (F.piP - F.piS))) * F.αPL k (!t) u

/-- `K_σ = f_{σ+} + f_{σ-}`. -/
noncomputable def KL (t σ : Bool) (u : ℂ) : ℂ := F.fMp k t σ u + F.fMm k t σ u

/-! ### The rank-two parts by powers of `z` -/

/-- The part of `R̂(z, x)` without `z`. -/
noncomputable def RhatL0 (σ : Bool) (x : ℂ) : ℂ :=
  (F.BL (!σ)).RhatC k (F.cN0L k 0) (F.cN1L k σ 0) (F.cN2L k 0) (F.cN3L σ) x

/-- The coefficient of `z²` in `R̂(z, x)`. -/
noncomputable def RhatL2 (σ : Bool) (x : ℂ) : ℂ :=
  (F.BL (!σ)).RhatC k (-(F.a ^ 2 * kron k 0 * F.ζL) - 2 * F.a * F.q₀L k)
    (-4 * F.a ^ 2 * F.rL (!σ)) (-2 * F.a + kron k 0 * F.ζL) 0 x

theorem RhatL_split (σ : Bool) (z x : ℂ) :
    F.RhatL k σ z x = F.RhatL0 k σ x + z ^ 2 * F.RhatL2 k σ x := by
  simp only [RhatL, RhatL0, RhatL2, Params.RhatC, cN0L, cN1L, cN2L, cN3L]
  ring

end FParams

end HubbardAtom
