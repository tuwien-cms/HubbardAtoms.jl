import HubbardAtomLean.SingletInput

/-!
# The density and magnetic susceptibilities: Proposition "structure"

The particle–hole channels `r = d, m` (`ε = 1, -1`) of Section "Density and magnetic":

* the polynomials `Q±`, `P`, `G₂`, `H`, `e`, `N₀`, `N₂` of Eqs. (Delta) and (N), and the
  susceptibility `χ_r = χ↑↑ + εχ↑↓` from the inputs Eqs. (chi-uu), (chi-ud);
* `compact_input`: Lemma "compact input", the pair term and the twelve fractions of Appendix C in
  the form of Eq. (input), for `μ ≠ 0` (identity (A1));
* `Params.Pph_mul_chiR`: **Proposition "structure"**,
  `P(zₙ)χ_r,nm = -βH(zₙ)δₙₘ + λρ/(zₙ + zₘ) + N(zₙ, zₘ)/P(zₘ)` for `μ ≠ 0`.
-/

open Complex

namespace HubbardAtom

/-- Lemma "compact input": for `c = ρ/(z + v)` (`μ ≠ 0`, with `ρ = q₊ - q₋`), the pair term
`cΠ(x, y)` and the twelve fractions of Appendix C are
`(4a²c - 2a)/(Q₋(z)Q₋(v)) + 4a²(1 - ρ)(z + v + 2a)(zv + q₀)/(P(z)P(v))`, where `z = x + w/2`,
`v = y + w/2`, `Q₋(z) = (x - a)(x + w - a)`, `Q₊(z) = (x + a)(x + w + a)`. -/
theorem compact_input {a w x y qm qp : ℂ} (hq : qm + qp = 1) (hxy : x + y + w ≠ 0)
    (h₁ : x - a ≠ 0) (h₂ : x + a ≠ 0) (h₃ : x + w - a ≠ 0) (h₄ : x + w + a ≠ 0)
    (h₅ : y - a ≠ 0) (h₆ : y + a ≠ 0) (h₇ : y + w - a ≠ 0) (h₈ : y + w + a ≠ 0) :
    (qp - qm) / (x + y + w) * piPair' a x y (x + w) (y + w) + Rreg' a qm qp x y (x + w) (y + w) =
      (4 * a ^ 2 * ((qp - qm) / (x + y + w)) - 2 * a) /
          ((x - a) * (x + w - a) * ((y - a) * (y + w - a))) +
        4 * a ^ 2 * (1 - (qp - qm)) * (x + y + w + 2 * a) *
            ((x + w / 2) * (y + w / 2) + (a ^ 2 - w ^ 2 / 4)) /
          ((x + a) * (x + w + a) * ((x - a) * (x + w - a)) *
            ((y + a) * (y + w + a) * ((y - a) * (y + w - a)))) := by
  obtain rfl : qp = 1 - qm := by linear_combination hq
  unfold piPair' Rreg'
  field_simp
  ring

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `q₀ = a² - w²/4 = (U² + ω²)/4`. -/
noncomputable def q₀ : ℂ := A.a ^ 2 - A.w k ^ 2 / 4

/-- `Q₊(z) = (z + a)² - w²/4`. -/
noncomputable def Qp (z : ℂ) : ℂ := (z + A.a) ^ 2 - A.w k ^ 2 / 4

/-- `Q₋(z) = (z - a)² - w²/4`. -/
noncomputable def Qm (z : ℂ) : ℂ := (z - A.a) ^ 2 - A.w k ^ 2 / 4

/-- `P = Q₊Q₋`. -/
noncomputable def Pph (z : ℂ) : ℂ := A.Qp k z * A.Qm k z

/-- `G₂(z) = (z + aρ)² - w²/4`. -/
noncomputable def G₂ (z : ℂ) : ℂ := (z + A.a * A.ρ) ^ 2 - A.w k ^ 2 / 4

/-- `λ = 4εa²`. -/
noncomputable def lam : ℂ := 4 * ε * A.a ^ 2

/-- `H(z) = z² + 2aρz + q₁` with `q₁ = q₀ + 4εa²p₁`. -/
noncomputable def Hph (z : ℂ) : ℂ := z ^ 2 + 2 * A.a * A.ρ * z + (A.q₀ k + 4 * ε * A.a ^ 2 * A.p₁)

/-- `e(z) = H(z)H(-z) - λ²h²/β²` with `h² = β²p₀p₂`, Eq. (Delta). -/
noncomputable def eph (z : ℂ) : ℂ := A.Hph k ε z * A.Hph k ε (-z) - A.lam ε ^ 2 * A.p₀ * A.p₂

/-- `ζ_r = βa²[1 - ρ² + 4ε(p₀p₂ - p₁²)]`. -/
noncomputable def ζ : ℂ := A.β * A.a ^ 2 * (1 - A.ρ ^ 2 + 4 * ε * (A.p₀ * A.p₂ - A.p₁ ^ 2))

/-- `N₂(v) = -2εa(v² + 2aρv + q₀) + δ_k0 ζ_r(v² - a²)`, Eq. (N). -/
noncomputable def N₂ (v : ℂ) : ℂ :=
  -2 * ε * A.a * (v ^ 2 + 2 * A.a * A.ρ * v + A.q₀ k) + kron k 0 * A.ζ ε * (v ^ 2 - A.a ^ 2)

/-- `N₀(v) = (v² + 2q₀ - 4a²)N₂(v) + (2εa - δ_k0 ζ_r)P(v)`, Eq. (N). -/
noncomputable def N₀ (v : ℂ) : ℂ :=
  (v ^ 2 + 2 * A.q₀ k - 4 * A.a ^ 2) * A.N₂ k ε v + (2 * ε * A.a - kron k 0 * A.ζ ε) * A.Pph k v

/-- `N(z, v) = N₀(v) + z²N₂(v)`. -/
noncomputable def Nph (z v : ℂ) : ℂ := A.N₀ k ε v + z ^ 2 * A.N₂ k ε v

/-- The density (`ε = 1`) and magnetic (`ε = -1`) susceptibility `χ_r = χ↑↑ + εχ↑↓`. -/
noncomputable def chiR (n m : ℤ) : ℂ := A.chiUpUp k n m + ε * A.chiUpDn k n m

/-- The particle–hole bubble of Eq. (B), `B(ν_n) = -βG(ν_n)G(ν_n + ω_k)`. -/
noncomputable def Bph (n : ℤ) : ℂ := -(A.β : ℂ) * A.G (A.x n) * A.G (A.x n + A.w k)

/-! ### Lattice facts -/

theorem x_eq_z (n : ℤ) : A.x n = A.z k n - A.w k / 2 := by
  rw [z]
  ring

theorem x_add_w_eq_z (n : ℤ) : A.x n + A.w k = A.z k n + A.w k / 2 := by
  rw [z]
  ring

/-- `Q₋(zₙ) = (xₙ - a)(xₙ + w - a)` and `Q₊(zₙ) = (xₙ + a)(xₙ + w + a)`, Eq. (Qfact). -/
theorem Qm_z (n : ℤ) : A.Qm k (A.z k n) = (A.x n - A.a) * (A.x (n + k) - A.a) := by
  rw [Qm, ← A.x_add_w, x_eq_z A k]
  ring

theorem Qp_z (n : ℤ) : A.Qp k (A.z k n) = (A.x n + A.a) * (A.x (n + k) + A.a) := by
  rw [Qp, ← A.x_add_w, x_eq_z A k]
  ring

theorem Qm_z_ne_zero (n : ℤ) : A.Qm k (A.z k n) ≠ 0 := by
  rw [Qm_z]
  exact mul_ne_zero (A.x_sub_ofReal_ne_zero _ _) (A.x_sub_ofReal_ne_zero _ _)

theorem Qp_z_ne_zero (n : ℤ) : A.Qp k (A.z k n) ≠ 0 := by
  rw [Qp_z]
  exact mul_ne_zero (A.x_add_ofReal_ne_zero _ _) (A.x_add_ofReal_ne_zero _ _)

/-- `P` has no zeros on the lattice. -/
theorem Pph_z_ne_zero (n : ℤ) : A.Pph k (A.z k n) ≠ 0 :=
  mul_ne_zero (A.Qp_z_ne_zero k n) (A.Qm_z_ne_zero k n)

/-- `zₙ + zₘ = 2μ + iΩ_{n+m+k+1}` has real part `2μ`. -/
theorem z_add_z_ne_zero (hμ : A.μ ≠ 0) (n m : ℤ) : A.z k n + A.z k m ≠ 0 := by
  intro h0
  have h := congrArg re h0
  simp only [A.z_eq, add_re, ofReal_re, mul_re, ofReal_im, I_re, I_im, zero_re] at h
  apply hμ
  linarith

/-- The pair coefficient on the lattice: `c_{n+m+k+1} = ρ/(zₙ + zₘ)` for `μ ≠ 0`, Eq. (C). -/
theorem pairC_eq (hμ : A.μ ≠ 0) (n m : ℤ) :
    A.pairC (n + m + k + 1) = A.ρ / (A.z k n + A.z k m) := by
  rw [pairC, ite_eq_right (fun h ↦ hμ h.1)]
  congr 1
  rw [A.z_eq, A.z_eq, ω]
  push_cast
  ring

-- the twelve fractions of Appendix C make the `field_simp` terms deep
set_option maxRecDepth 4000 in
/-- **Proposition "structure"**: `P(zₙ)χ_r,nm = -βH(zₙ)δₙₘ + λρ/(zₙ + zₘ) + N(zₙ, zₘ)/P(zₘ)` for
`μ ≠ 0`, from the inputs Eqs. (chi-uu), (chi-ud) (identities (A1), (A2), (A5)). -/
theorem Pph_mul_chiR (hμ : A.μ ≠ 0) (n m : ℤ) :
    A.Pph k (A.z k n) * A.chiR k ε n m =
      -A.β * A.Hph k ε (A.z k n) * kron n m + A.lam ε * A.ρ / (A.z k n + A.z k m) +
        A.Nph k ε (A.z k n) (A.z k m) / A.Pph k (A.z k m) := by
  have hq : (A.qm : ℂ) + A.qp = 1 := by
    rw [qm, qp]
    exact_mod_cast (show A.p₀ + A.p₁ + (A.p₁ + A.p₂) = 1 by linarith [A.weights_sum])
  have hρ : (A.ρ : ℂ) = A.qp - A.qm := by rw [ρ, qp, qm]; push_cast; ring
  have hρs : (A.ρ : ℂ) = A.p₂ - A.p₀ := by rw [ρ]; push_cast; ring
  have hzv : A.x n + A.x m + A.w k ≠ 0 := by
    have := A.z_add_z_ne_zero k hμ n m
    rwa [z, z, show A.x n + A.w k / 2 + (A.x m + A.w k / 2) = A.x n + A.x m + A.w k by ring]
      at this
  -- the linear factors
  have f₁ := A.x_sub_ofReal_ne_zero n A.a
  have f₂ := A.x_add_ofReal_ne_zero n A.a
  have f₃ : A.x n + A.w k - A.a ≠ 0 := by rw [A.x_add_w]; exact A.x_sub_ofReal_ne_zero _ _
  have f₄ : A.x n + A.w k + A.a ≠ 0 := by rw [A.x_add_w]; exact A.x_add_ofReal_ne_zero _ _
  have f₅ := A.x_sub_ofReal_ne_zero m A.a
  have f₆ := A.x_add_ofReal_ne_zero m A.a
  have f₇ : A.x m + A.w k - A.a ≠ 0 := by rw [A.x_add_w]; exact A.x_sub_ofReal_ne_zero _ _
  have f₈ : A.x m + A.w k + A.a ≠ 0 := by rw [A.x_add_w]; exact A.x_add_ofReal_ne_zero _ _
  have g₁ : A.x n ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := A.x_sq_sub_ne_zero n
  have g₂ : (A.x n + A.w k) ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by rw [A.x_add_w]; exact A.x_sq_sub_ne_zero _
  have g₃ : A.x m ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := A.x_sq_sub_ne_zero m
  have hci := compact_input hq hzv f₁ f₂ f₃ f₄ f₅ f₆ f₇ f₈
  rw [← hρ] at hci
  have hc : A.pairC (n + m + k + 1) = A.ρ / (A.x n + A.x m + A.w k) := by
    rw [pairC_eq A k hμ, z, z, show A.x n + A.w k / 2 + (A.x m + A.w k / 2) =
      A.x n + A.x m + A.w k by ring]
  have hPn : A.Pph k (A.z k n) =
      (A.x n + A.a) * (A.x n + A.w k + A.a) * ((A.x n - A.a) * (A.x n + A.w k - A.a)) := by
    rw [Pph, Qp_z, Qm_z, ← A.x_add_w]
  have hPm : A.Pph k (A.z k m) =
      (A.x m + A.a) * (A.x m + A.w k + A.a) * ((A.x m - A.a) * (A.x m + A.w k - A.a)) := by
    rw [Pph, Qp_z, Qm_z, ← A.x_add_w]
  rw [chiR, chiUpUp, chiUpDn, ← A.x_add_w, ← A.x_add_w, hc,
    show ∀ X Y : ℂ, X + Y + A.ρ / (A.x n + A.x m + A.w k) *
        piPair' A.a (A.x n) (A.x m) (A.x n + A.w k) (A.x m + A.w k) +
        Rreg' A.a A.qm A.qp (A.x n) (A.x m) (A.x n + A.w k) (A.x m + A.w k) =
      X + Y + (A.ρ / (A.x n + A.x m + A.w k) *
        piPair' A.a (A.x n) (A.x m) (A.x n + A.w k) (A.x m + A.w k) +
        Rreg' A.a A.qm A.qp (A.x n) (A.x m) (A.x n + A.w k) (A.x m + A.w k)) from
      fun X Y ↦ by ring, hci, hPn, hPm]
  simp only [Hph, Nph, N₀, N₂, ζ, lam, q₀, Pph, Qp, Qm, G, J, z]
  rw [show A.x n + A.w k / 2 + (A.x m + A.w k / 2) = A.x n + A.x m + A.w k by ring]
  by_cases hk0 : k = 0
  · -- `w = 0`, and the `δ_k0` terms are present
    subst hk0
    clear hci hc hPn hPm hzv
    have hw0 := A.w_zero
    simp only [hw0, add_zero, sub_zero, zero_div, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
      zero_pow, kron_self, mul_one] at f₃ f₄ f₇ f₈ g₂ ⊢
    by_cases hnm : n = m
    · subst hnm
      have hx0 : A.x n ≠ 0 := by simpa using A.x_sub_ofReal_ne_zero n 0
      rw [kron_self]
      field_simp
      rw [hρs]
      ring
    · have hxy : A.x n + A.x m ≠ 0 := by
        have := A.z_add_z_ne_zero 0 hμ n m
        simpa [z, hw0] using this
      rw [kron_of_ne hnm]
      field_simp
      rw [hρs]
      ring
  · rw [kron_of_ne hk0]
    by_cases hnm : n = m
    · subst hnm
      have h2 : 2 * A.x n + A.w k ≠ 0 := by
        rwa [show 2 * A.x n + A.w k = A.x n + A.x n + A.w k by ring]
      have h2' : A.x n * 2 + A.w k ≠ 0 := by rwa [mul_comm]
      have h2'' : A.w k + 2 * A.x n ≠ 0 := by rwa [add_comm]
      rw [← two_mul] at hci hc ⊢
      rw [kron_self]
      field_simp
      rw [hρs]
      ring
    · rw [kron_of_ne hnm]
      field_simp
      rw [hρs]
      ring

end Params

end HubbardAtom
