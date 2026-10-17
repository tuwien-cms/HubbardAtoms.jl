import HubbardAtomLean.DMLifts

/-!
# Corollary "half filling"

The density and magnetic channels at `μ = 0`. There `ρ = 0`, `H` is even, and the Cauchy matrix of
Proposition "structure" becomes the mirror `h𝓜` with `h = βp₀` (`Pph_mul_chiR_hf`). The inverse is
the inverse of the diagonal-plus-mirror part, plus a constant and one separable term in
`ĝ(z) = 1/(z² - ŝ)`; the lattice sums are `T̂ = ∑ₗ ĝ(zₗ)` (`HFHyp.hasSum_That`), `ℓ₀`, `ℓ₂`.

* `Params.HFHyp`: the assumptions of Corollary "half filling": `U ≠ 0`, `μ = 0`, `e(zₙ) ≠ 0` for
  all `n`, `û² = ŝ` and `D̂ ≠ 0`. Then `P(û) ≠ 0` (`Pph_ne_of_shat`).
* `Params.HFHyp.chiR_isRightInverse`, `Params.HFHyp.chiR_isLeftInverse`,
  `Params.HFHyp.gammaHF_eq`: **Corollary "half filling"**, Eq. (Gamma-hf), proved directly (the
  note takes the limit `μ → 0` of Theorem "density and magnetic"). The case `û = 0` (even `k`) is
  included, with `D̂ = β(1 + εaβ/2)`.
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `ŝ = 4εa²(p₀ - p₁) - q₀`: at `μ = 0`, `e(z) = (z² - ŝ)(H(z) + λp₀)`. -/
noncomputable def shat : ℂ := 4 * ε * A.a ^ 2 * (A.p₀ - A.p₁) - A.q₀ k

/-- `ĝ(z) = 1/(z² - ŝ)`. -/
noncomputable def ghat (z : ℂ) : ℂ := 1 / (z ^ 2 - A.shat k ε)

/-- `T̂ = S_k(û)/û`, with the value `-β²/4` at `û = 0`; it is `∑ₗ ĝ(zₗ)` (`HFHyp.hasSum_That`). -/
noncomputable def That (u : ℂ) : ℂ := if u = 0 then -(A.β : ℂ) ^ 2 / 4 else A.S k u / u

/-- `D̂ = β - 2εaS_k(û)/û`, with the value `β(1 + εaβ/2)` at `û = 0`. -/
noncomputable def Dhat (u : ℂ) : ℂ := A.β - 2 * ε * A.a * A.That k u

/-- `ŝ' = -4εa²(p₀ + p₁) - q₀`, the other root: at `μ = 0`, `e(z) = (z² - ŝ)(z² - ŝ')`. -/
noncomputable def shat' : ℂ := -4 * ε * A.a ^ 2 * (A.p₀ + A.p₁) - A.q₀ k

/-- The inverse of the mirror-even eigenvalue `-β(z² - ŝ)/P(z)` of the diagonal-plus-mirror
part. -/
noncomputable def ievn (z : ℂ) : ℂ := -(A.Pph k z / (A.β * (z ^ 2 - A.shat k ε)))

/-- The inverse of the mirror-odd eigenvalue `-β(z² - ŝ')/P(z)`. -/
noncomputable def iodd (z : ℂ) : ℂ := -(A.Pph k z / (A.β * (z ^ 2 - A.shat' k ε)))

/-- The inverse of `χ_r` at `μ = 0`: the inverse of the diagonal-plus-mirror part on the mirror
orbits, plus `2εa/β² - 2εaP(û)ĝ(zₙ)ĝ(zₘ)/(βD̂)`. -/
noncomputable def invHF (u : ℂ) (n m : ℤ) : ℂ :=
  mirrorBlock k (fun j ↦ A.ievn k ε (A.z k j)) (fun j ↦ A.iodd k ε (A.z k j)) n m +
      2 * ε * A.a / A.β ^ 2 -
    2 * ε * A.a * A.Pph k u / (A.β * A.Dhat k ε u) * A.ghat k ε (A.z k n) *
      A.ghat k ε (A.z k m)

/-- The closed form of Eq. (Gamma-hf) in orbit form: `βP(zₙ)δₙₘ/G₂(zₙ) + β²(invHF part on the
orbits) + 2εa - 2εaβP(û)ĝ(zₙ)ĝ(zₘ)/D̂`; it is Eq. (Gamma-hf) where `e(zₙ) ≠ 0`
(`gammaHF_eq_note`), and holds also at a mirror fixed point where `βH(0) = -λh`. -/
noncomputable def gammaHF (u : ℂ) (n m : ℤ) : ℂ :=
  A.β * A.Pph k (A.z k n) / A.G₂ k (A.z k n) * kron n m +
      A.β ^ 2 * mirrorBlock k (fun j ↦ A.ievn k ε (A.z k j)) (fun j ↦ A.iodd k ε (A.z k j)) n m +
      2 * ε * A.a -
    2 * ε * A.a * A.β * A.Pph k u / A.Dhat k ε u * A.ghat k ε (A.z k n) * A.ghat k ε (A.z k m)

/-- Eq. (Gamma-hf) as printed:
`βP(zₙ)[1/G₂(zₙ) - H(zₙ)/e(zₙ)]δₙₘ + 2εa - λhF(zₙ)δ_{m n̄} - 2εaβP(û)ĝ(zₙ)ĝ(zₘ)/D̂`. -/
noncomputable def gammaHFnote (u : ℂ) (n m : ℤ) : ℂ :=
  A.β * A.Pph k (A.z k n) *
        (1 / A.G₂ k (A.z k n) - A.Hph k ε (A.z k n) / A.eph k ε (A.z k n)) * kron n m +
      2 * ε * A.a - A.lam ε * (A.β * A.p₀) * A.Fph k ε (A.z k n) * kron m (mirror k n) -
    2 * ε * A.a * A.β * A.Pph k u / A.Dhat k ε u * A.ghat k ε (A.z k n) * A.ghat k ε (A.z k m)

/-- The assumptions of Corollary "half filling" (with `U ≠ 0`), for a root `û² = ŝ`: `zₙ² ≠ ŝ`, and
`zₙ² ≠ ŝ'` off the mirror fixed points (together weaker than `e(zₙ) ≠ 0`, `HFHyp.of_eph`), and
`D̂ ≠ 0`. -/
structure HFHyp (u : ℂ) : Prop where
  ε_eq : ε = 1 ∨ ε = -1
  a_ne : (A.a : ℂ) ≠ 0
  μ_eq : A.μ = 0
  z_sq_ne : ∀ n, A.z k n ^ 2 - A.shat k ε ≠ 0
  odd_ne : ∀ n, n ≠ mirror k n → A.z k n ^ 2 - A.shat' k ε ≠ 0
  u_sq : u ^ 2 = A.shat k ε
  D_ne : A.Dhat k ε u ≠ 0

/-! ### The parity at `μ = 0` -/

theorem rho_eq_zero_of_mu (hμ : A.μ = 0) : (A.ρ : ℂ) = 0 := by
  rw [ρ, p₂_eq_p₀_of_mu hμ, sub_self, ofReal_zero]

theorem Hph_neg_of_mu (hμ : A.μ = 0) (z : ℂ) : A.Hph k ε (-z) = A.Hph k ε z := by
  simp only [Hph, A.rho_eq_zero_of_mu hμ]
  ring

theorem Nph_neg_of_mu (hμ : A.μ = 0) (z t : ℂ) : A.Nph k ε z (-t) = A.Nph k ε z t := by
  rw [A.Nph_eq_coeffs, A.Nph_eq_coeffs]
  simp only [cN₁, cN₃, A.rho_eq_zero_of_mu hμ]
  ring

/-- `N(z, t) = N₀(z) + t²N₂(z)` at `μ = 0`: `N` is symmetric. -/
theorem Nph_of_mu (hμ : A.μ = 0) (z t : ℂ) :
    A.Nph k ε z t = A.cN₀ k ε z + A.N₂ k ε z * t ^ 2 := by
  rw [A.Nph_eq_coeffs]
  simp only [cN₁, cN₂, cN₃, N₂, A.rho_eq_zero_of_mu hμ]
  ring

theorem Fph_neg (z : ℂ) : A.Fph k ε (-z) = A.Fph k ε z := by
  rw [Fph, Fph, A.Pph_neg, A.eph_neg]

theorem ghat_neg (z : ℂ) : A.ghat k ε (-z) = A.ghat k ε z := by
  rw [ghat, ghat, neg_sq]

theorem Hph_sub_of_mu (hμ : A.μ = 0) (z : ℂ) :
    A.Hph k ε z - A.lam ε * A.p₀ = z ^ 2 - A.shat k ε := by
  simp only [Hph, lam, shat, A.rho_eq_zero_of_mu hμ]
  ring

theorem Hph_add_of_mu (hμ : A.μ = 0) (z : ℂ) :
    A.Hph k ε z + A.lam ε * A.p₀ = z ^ 2 - A.shat' k ε := by
  simp only [Hph, lam, shat', A.rho_eq_zero_of_mu hμ]
  ring

theorem eph_of_mu (hμ : A.μ = 0) (z : ℂ) :
    A.eph k ε z = (A.Hph k ε z - A.lam ε * A.p₀) * (A.Hph k ε z + A.lam ε * A.p₀) := by
  rw [eph, A.Hph_neg_of_mu k ε hμ, p₂_eq_p₀_of_mu hμ]
  ring

theorem eph_of_mu' (hμ : A.μ = 0) (z : ℂ) :
    A.eph k ε z = (z ^ 2 - A.shat k ε) * (z ^ 2 - A.shat' k ε) := by
  rw [A.eph_of_mu k ε hμ, A.Hph_sub_of_mu k ε hμ, A.Hph_add_of_mu k ε hμ]

theorem Hph_of_mu (hμ : A.μ = 0) (z : ℂ) :
    A.Hph k ε z = z ^ 2 - (A.shat k ε + A.shat' k ε) / 2 := by
  simp only [Hph, shat, shat', A.rho_eq_zero_of_mu hμ]
  ring

theorem lam_p₀ : A.lam ε * (A.β * A.p₀) = A.β * ((A.shat k ε - A.shat' k ε) / 2) := by
  simp only [lam, shat, shat']
  ring

theorem ievn_neg (z : ℂ) : A.ievn k ε (-z) = A.ievn k ε z := by
  rw [ievn, ievn, A.Pph_neg, neg_sq]

theorem iodd_neg (z : ℂ) : A.iodd k ε (-z) = A.iodd k ε z := by
  rw [iodd, iodd, A.Pph_neg, neg_sq]

theorem z_add_z_ne_zero_of_ne_mirror (hμ : A.μ = 0) {n m : ℤ} (hm : m ≠ mirror k n) :
    A.z k n + A.z k m ≠ 0 := by
  intro h
  apply hm
  apply A.z_injective k
  rw [A.z_mirror_of_mu k hμ]
  linear_combination h

/-- **Proposition "structure"** at `μ = 0`, where the Cauchy matrix is the mirror `h𝓜`:
`P(zₙ)χ_r,nm = -βH(zₙ)δₙₘ + λβp₀δ_{m n̄} + N(zₙ, zₘ)/P(zₘ)`. -/
theorem Pph_mul_chiR_hf (hμ : A.μ = 0) (n m : ℤ) :
    A.Pph k (A.z k n) * A.chiR k ε n m =
      -A.β * A.Hph k ε (A.z k n) * kron n m + A.lam ε * (A.β * A.p₀) * kron m (mirror k n) +
        A.Nph k ε (A.z k n) (A.z k m) / A.Pph k (A.z k m) := by
  by_cases hm : m = mirror k n
  · subst hm
    rw [kron_self, mul_one]
    exact A.Pph_mul_chiR_mirror k ε hμ n
  · rw [A.Pph_mul_chiR' k ε (A.z_add_z_ne_zero_of_ne_mirror k hμ hm), kron_of_ne hm,
      A.rho_eq_zero_of_mu hμ]
    ring

/-- `P(û) = 4a²[a²(2(p₀ - p₁) - ε)² + ω²/4] ≠ 0` at `μ = 0`, as `|2(p₀ - p₁)| = |4p₀ - 1| < 1`. -/
theorem Pph_ne_of_shat (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0) {u : ℂ}
    (hu : u ^ 2 = A.shat k ε) : A.Pph k u ≠ 0 := by
  have hs : A.p₀ + A.p₁ = 1 / 2 := by
    have := A.weights_sum
    rw [p₂_eq_p₀_of_mu hμ] at this
    linarith
  have hp₀ := A.p₀_pos
  have hp₁ := A.p₁_pos
  have ha' : A.a ≠ 0 := by exact_mod_cast ha
  obtain ⟨e, he, rfl⟩ : ∃ e : ℝ, (e = 1 ∨ e = -1) ∧ ε = e := by
    rcases hε with rfl | rfl
    · exact ⟨1, Or.inl rfl, by norm_num⟩
    · exact ⟨-1, Or.inr rfl, by norm_num⟩
  have hd : 2 * (A.p₀ - A.p₁) - e ≠ 0 := by
    rcases he with rfl | rfl <;> intro h <;> linarith
  have hpos : 0 < (A.a * (2 * (A.p₀ - A.p₁) - e)) ^ 2 + A.ω k ^ 2 / 4 :=
    add_pos_of_pos_of_nonneg
      (lt_of_le_of_ne (sq_nonneg _) (pow_ne_zero 2 (mul_ne_zero ha' hd)).symm) (by positivity)
  have he2 : (e : ℂ) ^ 2 = 1 := by rcases he with rfl | rfl <;> norm_num
  have hP : A.Pph k u = 4 * A.a ^ 2 *
      (((A.a * (2 * (A.p₀ - A.p₁) - e)) ^ 2 + A.ω k ^ 2 / 4 : ℝ) : ℂ) := by
    rw [A.Pph_eq_poly, show u ^ 4 = (u ^ 2) ^ 2 by ring, hu]
    simp only [shat, κ, q₀, w]
    push_cast
    linear_combination (-(A.a : ℂ) ^ 2 * (A.ω k : ℂ) ^ 2) * I_sq +
      (16 * (A.a : ℂ) ^ 4 * (A.p₀ - A.p₁) ^ 2 - 4 * A.a ^ 4) * he2
  rw [hP]
  exact mul_ne_zero (mul_ne_zero (by norm_num) (pow_ne_zero 2 ha)) (ofReal_ne_zero.mpr hpos.ne')

/-- The note's assumptions `e(zₙ) ≠ 0` (with `μ = 0`, `û² = ŝ`, `D̂ ≠ 0`) imply `HFHyp`. -/
theorem HFHyp.of_eph {u : ℂ} (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0)
    (he : ∀ n, A.eph k ε (A.z k n) ≠ 0) (hu : u ^ 2 = A.shat k ε) (hD : A.Dhat k ε u ≠ 0) :
    A.HFHyp k ε u :=
  ⟨hε, ha, hμ, fun n h0 ↦ he n (by rw [A.eph_of_mu' k ε hμ, h0, zero_mul]),
    fun n _ h0 ↦ he n (by rw [A.eph_of_mu' k ε hμ, h0, mul_zero]), hu, hD⟩

/-! ### The lattice sums -/

theorem hasSum_ℓ₀ (ha : (A.a : ℂ) ≠ 0) : HasSum (fun l ↦ 1 / A.Pph k (A.z k l)) (A.ℓ₀ k) := by
  convert A.hasSum_cubic k ha 1 0 0 0 (A.z_ne_neg_add k 0) using 1
  · funext l
    ring
  · ring

theorem hasSum_ℓ₂ (ha : (A.a : ℂ) ≠ 0) :
    HasSum (fun l ↦ A.z k l ^ 2 / A.Pph k (A.z k l)) (A.ℓ₂ k) := by
  convert A.hasSum_cubic k ha 0 0 1 0 (A.z_ne_neg_add k 0) using 1
  · funext l
    ring
  · ring

/-- `∑ₗ N(z, zₗ)/P(zₗ) = N₀(z)ℓ₀ + N₂(z)ℓ₂` at `μ = 0`, absolutely convergent. -/
theorem hasSum_Nph_div (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0) (z : ℂ) :
    HasSum (fun l ↦ A.Nph k ε z (A.z k l) / A.Pph k (A.z k l))
      (A.cN₀ k ε z * A.ℓ₀ k + A.N₂ k ε z * A.ℓ₂ k) := by
  convert ((A.hasSum_ℓ₀ k ha).mul_left (A.cN₀ k ε z)).add
    ((A.hasSum_ℓ₂ k ha).mul_left (A.N₂ k ε z)) using 1
  funext l
  rw [A.Nph_of_mu k ε hμ]
  ring

variable {A k ε} {u : ℂ}

/-- `∑ₗ ĝ(zₗ) = T̂`: `S_k(û)/û` by Lemma "basic sum", and `∑ₗ zₗ⁻² = -β²/4` at `û = 0`. -/
theorem HFHyp.hasSum_That (h : A.HFHyp k ε u) :
    HasSum (fun l ↦ A.ghat k ε (A.z k l)) (A.That k u) := by
  have hμ := h.μ_eq
  by_cases hu0 : u = 0
  · -- then `k` is even: for odd `k`, `z = 0` is a lattice point
    have hs : A.shat k ε = 0 := by rw [← h.u_sq, hu0]; ring
    have hk : Even k := by
      rcases Int.even_or_odd k with hk | hk
      · exact hk
      · obtain ⟨n, hn⟩ := A.exists_z_eq_mu_of_odd hk
        exact absurd (by rw [hn, hμ, hs]; simp) (h.z_sq_ne n)
    have hz : ∀ l, A.z k l ≠ 0 := fun l h0 ↦ h.z_sq_ne l (by rw [h0, hs]; ring)
    have hS : A.S k 0 = 0 := by
      have := A.S_mu_of_even hk
      rwa [hμ, ofReal_zero] at this
    have h₀ := A.hasSum_sq hz
    rw [hS] at h₀
    rw [That, ite_eq_left hu0]
    convert h₀ using 1
    · funext l
      rw [ghat, hs, sub_zero, sub_zero]
    · ring
  · have hzu : ∀ l, A.z k l ≠ u := fun l h0 ↦ h.z_sq_ne l (by rw [h0, h.u_sq]; ring)
    have hzu' : ∀ l, A.z k l ≠ -u := fun l h0 ↦ h.z_sq_ne l (by rw [h0, neg_sq, h.u_sq]; ring)
    have hS : A.S k (-u) = -A.S k u := by
      have := A.S_mu_sub k u
      rwa [hμ, ofReal_zero, zero_sub, zero_add] at this
    rw [That, ite_eq_right hu0]
    convert (A.hasSum_sub_inv k hzu hzu').div_const (2 * u) using 1
    · funext l
      have h₁ := sub_ne_zero.mpr (hzu l)
      have h₂ : A.z k l + u ≠ 0 := fun h0 ↦ hzu' l (by linear_combination h0)
      have h₃ : A.z k l ^ 2 - u ^ 2 ≠ 0 := by
        rw [show A.z k l ^ 2 - u ^ 2 = (A.z k l - u) * (A.z k l + u) by ring]
        exact mul_ne_zero h₁ h₂
      rw [ghat, ← h.u_sq, sub_neg_eq_add, div_sub_div _ _ h₁ h₂, div_div,
        eq_div_iff (mul_ne_zero (mul_ne_zero h₁ h₂) (mul_ne_zero two_ne_zero hu0))]
      field_simp
      ring
    · rw [hS]
      field_simp
      ring

variable (A k ε) in
/-- The rational identity behind `HFHyp.hasSum_Nph_ghat`, with `P(û) = ŝ² + κŝ + q₀²` and
`N(z, û) = N₀(z) + N₂(z)ŝ`: `2εa[(N₂(z)P(û) - N(z, û)(κ + ŝ))ℓ₀ - N(z, û)ℓ₂]
= 2εaβP(û) - βN(z, û)`. -/
theorem identity_hf (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0) (z : ℂ) :
    2 * ε * A.a * ((A.N₂ k ε z * (A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2) -
        (A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε) * (A.κ k + A.shat k ε)) * A.ℓ₀ k -
      (A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε) * A.ℓ₂ k) =
    2 * ε * A.a * A.β * (A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2) -
      A.β * (A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε) := by
  have hq := A.q₀_ne_zero k ha
  have hs := A.weights_sum'
  have hρ := A.rho_eq_zero_of_mu hμ
  have hp₂ := p₂_eq_p₀_of_mu hμ
  rw [hp₂] at hs
  have hpp : (A.p₀ : ℂ) + A.p₁ = 1 / 2 := by linear_combination hs / 2
  have hSp : A.Sp = A.β * (A.p₁ - A.p₀) := by
    rw [Sp, A.S₀_neg_a, hpp]
    ring
  have hSm : A.Sm = A.β * (A.p₀ - A.p₁) := by
    rw [Sm, A.S₀_a, hp₂, add_comm (A.p₁ : ℂ), hpp]
    ring
  simp only [cN₀, N₂, ℓ₀, ℓ₂, Tp, Tm, hSp, hSm, ζ, κ, shat, hρ, hp₂]
  rw [q₀] at hq ⊢
  generalize (A.p₁ : ℂ) = p₁ at *
  obtain rfl : p₁ = 1 / 2 - A.p₀ := by linear_combination hpp
  rcases hε with rfl | rfl <;> by_cases hk : k = 0
  · subst hk
    simp only [kron_self, A.w_zero] at hq ⊢
    field_simp
    ring
  · rw [kron_of_ne hk]
    have hq4 : (A.a : ℂ) ^ 2 * 4 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    have hq4' : 4 * (A.a : ℂ) ^ 2 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    field_simp
    ring
  · subst hk
    simp only [kron_self, A.w_zero] at hq ⊢
    field_simp
    ring
  · rw [kron_of_ne hk]
    have hq4 : (A.a : ℂ) ^ 2 * 4 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    have hq4' : 4 * (A.a : ℂ) ^ 2 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    field_simp
    ring

/-- `∑ₗ N(z, zₗ)ĝ(zₗ)/P(zₗ) = β - N(z, û)D̂/(2εaP(û))`, by partial fractions in `zₗ²`:
`N(z, t)ĝ(t)/P(t) = cĝ(t) + (N₂(z) - c(κ + ŝ) - ct²)/P(t)` with `c = N(z, û)/P(û)`. -/
theorem HFHyp.hasSum_Nph_ghat (h : A.HFHyp k ε u) (z : ℂ) :
    HasSum (fun l ↦ A.Nph k ε z (A.z k l) * A.ghat k ε (A.z k l) / A.Pph k (A.z k l))
      (A.β - A.Nph k ε z u * A.Dhat k ε u / (2 * ε * A.a * A.Pph k u)) := by
  have hμ := h.μ_eq
  have ha := h.a_ne
  have hε0 : ε ≠ 0 := by rcases h.ε_eq with rfl | rfl <;> norm_num
  have hPu := A.Pph_ne_of_shat k ε h.ε_eq ha hμ h.u_sq
  have hPu' : A.Pph k u = A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 := by
    rw [A.Pph_eq_poly, show u ^ 4 = (u ^ 2) ^ 2 by ring, h.u_sq]
  have hNu : A.Nph k ε z u = A.cN₀ k ε z + A.N₂ k ε z * A.shat k ε := by
    rw [A.Nph_of_mu k ε hμ, h.u_sq]
  convert ((h.hasSum_That.mul_left (A.Nph k ε z u / A.Pph k u)).add
    ((A.hasSum_ℓ₀ k ha).mul_left
      (A.N₂ k ε z - A.Nph k ε z u / A.Pph k u * (A.κ k + A.shat k ε)))).add
    ((A.hasSum_ℓ₂ k ha).mul_left (-(A.Nph k ε z u / A.Pph k u))) using 1
  · funext l
    have h₁ := h.z_sq_ne l
    have h₂ := A.Pph_z_ne_zero k l
    rw [A.Pph_eq_poly] at h₂ ⊢
    rw [hPu'] at hPu ⊢
    rw [A.Nph_of_mu k ε hμ, ghat, hNu]
    -- clear the denominators as atoms, then put the polynomials back
    generalize hD : A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 = D at hPu ⊢
    generalize hPl : A.z k l ^ 4 + A.κ k * A.z k l ^ 2 + A.q₀ k ^ 2 = Pl at h₂ ⊢
    generalize hG : A.z k l ^ 2 - A.shat k ε = G at h₁ ⊢
    field_simp
    rw [← hD, ← hPl, ← hG]
    ring
  · have key := A.identity_hf k ε h.ε_eq ha hμ z
    rw [hPu'] at hPu ⊢
    rw [hNu, Dhat]
    generalize A.shat k ε ^ 2 + A.κ k * A.shat k ε + A.q₀ k ^ 2 = D at hPu key ⊢
    field_simp
    linear_combination -key

/-! ### The right inverse -/

variable (A k ε) in
/-- The row `P(zₙ)χ_r,nl` at `μ = 0` (`Pph_mul_chiR_hf`). -/
noncomputable def RrowHF (n l : ℤ) : ℂ :=
  -A.β * A.Hph k ε (A.z k n) * kron n l + A.lam ε * (A.β * A.p₀) * kron l (mirror k n) +
    A.Nph k ε (A.z k n) (A.z k l) / A.Pph k (A.z k l)

variable (A k ε) in
/-- Identity (A6) at `μ = 0`: `-βH(z) + λh + N₀(z)ℓ₀ + N₂(z)ℓ₂ = βN₂(z)/(2εa)`. -/
theorem identity_A6_hf (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0) (z : ℂ) :
    -A.β * A.Hph k ε z + A.lam ε * (A.β * A.p₀) + (A.cN₀ k ε z * A.ℓ₀ k + A.N₂ k ε z * A.ℓ₂ k) =
      A.β * A.N₂ k ε z / (2 * ε * A.a) := by
  have h6 := A.identity_A6 k ε hε ha z
  have hρ := A.rho_eq_zero_of_mu hμ
  have hc : (A.cbar : ℂ) = A.β * A.p₀ := by
    rw [cbar, p₂_eq_p₀_of_mu hμ]
    push_cast
    ring
  have h₂ : A.cN₂ k ε z = A.N₂ k ε z := by
    simp only [cN₂, N₂, hρ]
    ring
  rw [hc, h₂] at h6
  simp only [cN₁, cN₃, hρ] at h6
  linear_combination h6

variable (A k) in
theorem mul_kron_mirror_eq (hμ : A.μ = 0) (f : ℂ → ℂ) (hf : ∀ z, f (-z) = f z) (n m : ℤ) :
    f (A.z k m) * kron n (mirror k m) = f (A.z k n) * kron n (mirror k m) := by
  by_cases h : n = mirror k m
  · rw [h, A.z_mirror_of_mu k hμ, hf]
  · rw [kron_of_ne h, mul_zero, mul_zero]

/-- The sum of the row, Step 2 at `μ = 0`: `∑ₗ P(zₙ)χ_r,nl = βN₂(zₙ)/(2εa)`. -/
theorem HFHyp.hasSum_RrowHF (h : A.HFHyp k ε u) (n : ℤ) :
    HasSum (fun l ↦ A.RrowHF k ε n l) (A.β * A.N₂ k ε (A.z k n) / (2 * ε * A.a)) := by
  have h₁ : HasSum (fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l)
      (-A.β * A.Hph k ε (A.z k n)) := by
    convert hasSum_single (f := fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero]) using 1
    rw [kron_self, mul_one]
  have h₂ : HasSum (fun l ↦ A.lam ε * (A.β * A.p₀) * kron l (mirror k n))
      (A.lam ε * (A.β * A.p₀)) := by
    convert hasSum_single (f := fun l ↦ A.lam ε * (A.β * A.p₀) * kron l (mirror k n))
      (mirror k n) (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero]) using 1
    rw [kron_self, mul_one]
  convert (h₁.add h₂).add (A.hasSum_Nph_div k ε h.a_ne h.μ_eq (A.z k n)) using 1
  · funext l
    rw [RrowHF]
  · exact (A.identity_A6_hf k ε h.ε_eq h.a_ne h.μ_eq (A.z k n)).symm

/-- The image of `ĝ`: `∑ₗ P(zₙ)χ_r,nl ĝ(zₗ) = -N(zₙ, û)D̂/(2εaP(û))`. -/
theorem HFHyp.hasSum_RrowHF_ghat (h : A.HFHyp k ε u) (n : ℤ) :
    HasSum (fun l ↦ A.RrowHF k ε n l * A.ghat k ε (A.z k l))
      (-(A.Nph k ε (A.z k n) u * A.Dhat k ε u / (2 * ε * A.a * A.Pph k u))) := by
  have hμ := h.μ_eq
  have h₁ : HasSum (fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l * A.ghat k ε (A.z k l))
      (-A.β * A.Hph k ε (A.z k n) * A.ghat k ε (A.z k n)) := by
    convert hasSum_single (f := fun l ↦ -A.β * A.Hph k ε (A.z k n) * kron n l *
      A.ghat k ε (A.z k l)) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_mul]) using 1
    rw [kron_self, mul_one]
  have h₂ : HasSum (fun l ↦ A.lam ε * (A.β * A.p₀) * kron l (mirror k n) * A.ghat k ε (A.z k l))
      (A.lam ε * (A.β * A.p₀) * A.ghat k ε (A.z k n)) := by
    convert hasSum_single (f := fun l ↦ A.lam ε * (A.β * A.p₀) * kron l (mirror k n) *
      A.ghat k ε (A.z k l)) (mirror k n)
      (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero, zero_mul]) using 1
    rw [kron_self, mul_one, A.z_mirror_of_mu k hμ, A.ghat_neg]
  have hdiag : -A.β * A.Hph k ε (A.z k n) * A.ghat k ε (A.z k n) +
      A.lam ε * (A.β * A.p₀) * A.ghat k ε (A.z k n) = -A.β := by
    have hg := h.z_sq_ne n
    have hHs := A.Hph_sub_of_mu k ε hμ (A.z k n)
    rw [ghat]
    field_simp
    linear_combination -(A.β : ℂ) * hHs
  convert (h₁.add h₂).add (h.hasSum_Nph_ghat (A.z k n)) using 1
  · funext l
    rw [RrowHF]
    ring
  · linear_combination -hdiag

/-- The diagonal and mirror part of the column `m`: with `ε±(v)` the inverse even and odd
eigenvalues, `P(zₙ)χ_r,nm (ε₊ + ε₋)/2 + P(zₙ)χ_r,nm̄ (ε₊ - ε₋)/2 = P(zₙ)δₙₘ - N(zₙ, v)ĝ(v)/β`,
`v = zₘ`; at a mirror fixed point only `ε₊` enters. -/
theorem HFHyp.delta_part (h : A.HFHyp k ε u) (n m : ℤ) :
    A.RrowHF k ε n m * ((A.ievn k ε (A.z k m) + A.iodd k ε (A.z k m)) / 2) +
        A.RrowHF k ε n (mirror k m) * ((A.ievn k ε (A.z k m) - A.iodd k ε (A.z k m)) / 2) =
      A.Pph k (A.z k n) * kron n m -
        A.Nph k ε (A.z k n) (A.z k m) * A.ghat k ε (A.z k m) / A.β := by
  have hμ := h.μ_eq
  have hβ := A.β_ne_zero
  have hPm := A.Pph_z_ne_zero k m
  have hgm := h.z_sq_ne m
  have hzm := A.z_mirror_of_mu k hμ m
  simp only [RrowHF, ievn, iodd, ghat, hzm, A.Nph_neg_of_mu k ε hμ, A.Pph_neg,
    kron_mirror_mirror, kron_comm m n, ← kron_mirror_comm k n m, A.lam_p₀ k ε]
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self, A.Hph_of_mu k ε hμ]
    by_cases hfix : n = mirror k n
    · -- a mirror fixed point: the odd eigenvalue drops out
      rw [← hfix, kron_self]
      field_simp
      ring
    · have hodd := h.odd_ne n hfix
      rw [kron_of_ne hfix]
      field_simp
      ring
  · rw [kron_of_ne hnm]
    by_cases hmir : n = mirror k m
    · -- `zₙ = -zₘ`
      have hm' : m ≠ mirror k m := fun h0 ↦ hnm (by rw [hmir, ← h0])
      have hodd := h.odd_ne m hm'
      rw [hmir, kron_self, hzm, A.Hph_neg_of_mu k ε hμ, A.Hph_of_mu k ε hμ]
      field_simp
      ring
    · rw [kron_of_ne hmir]
      field_simp
      ring

/-- **Corollary "half filling"**, the inverse: `χ_r invHF = 1`. By Proposition "structure" at
`μ = 0`, `P(zₙ)(χ_r invHF)ₙₘ` is `delta_part` plus `2εa/β²` times the row sum plus the image of `ĝ`,
which add up to `P(zₙ)δₙₘ`, as `N(zₙ, zₘ) - N(zₙ, û) = N₂(zₙ)(zₘ² - ŝ)`. -/
theorem HFHyp.chiR_isRightInverse (h : A.HFHyp k ε u) :
    IsRightInverse (A.chiR k ε) (A.invHF k ε u) := by
  intro n m
  have hμ := h.μ_eq
  have ha := h.a_ne
  have hβ := A.β_ne_zero
  have hε0 : ε ≠ 0 := by rcases h.ε_eq with rfl | rfl <;> norm_num
  have hPn := A.Pph_z_ne_zero k n
  have hPu := A.Pph_ne_of_shat k ε h.ε_eq ha hμ h.u_sq
  have hD := h.D_ne
  have hgm := h.z_sq_ne m
  set ep := (A.ievn k ε (A.z k m) + A.iodd k ε (A.z k m)) / 2 with hep
  set em := (A.ievn k ε (A.z k m) - A.iodd k ε (A.z k m)) / 2 with hem
  have s₁ : HasSum (fun l ↦ A.RrowHF k ε n l * (ep * kron l m)) (A.RrowHF k ε n m * ep) := by
    convert hasSum_single (f := fun l ↦ A.RrowHF k ε n l * (ep * kron l m)) m
      (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero]) using 1
    rw [kron_self, mul_one]
  have s₂ : HasSum (fun l ↦ A.RrowHF k ε n l * (em * kron l (mirror k m)))
      (A.RrowHF k ε n (mirror k m) * em) := by
    convert hasSum_single (f := fun l ↦ A.RrowHF k ε n l * (em * kron l (mirror k m)))
      (mirror k m) (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero]) using 1
    rw [kron_self, mul_one]
  have s₃ := (h.hasSum_RrowHF n).mul_left (2 * ε * A.a / A.β ^ 2)
  have s₄ := (h.hasSum_RrowHF_ghat n).mul_left
    (-(2 * ε * A.a * A.Pph k u / (A.β * A.Dhat k ε u)) * A.ghat k ε (A.z k m))
  convert (((s₁.add s₂).add s₃).add s₄).div_const (A.Pph k (A.z k n)) using 1
  · funext l
    have hc : A.chiR k ε n l = A.RrowHF k ε n l / A.Pph k (A.z k n) := by
      rw [eq_div_iff hPn, mul_comm, RrowHF]
      exact A.Pph_mul_chiR_hf k ε hμ n l
    have hmb : mirrorBlock k (fun j ↦ A.ievn k ε (A.z k j)) (fun j ↦ A.iodd k ε (A.z k j)) l m =
        ep * kron l m + em * kron l (mirror k m) := by
      rw [mirrorBlock, hep, hem]
      have h₁ := mul_kron_eq (fun j ↦ (A.ievn k ε (A.z k j) + A.iodd k ε (A.z k j)) / 2) l m
      have h₂ : (A.ievn k ε (A.z k l) - A.iodd k ε (A.z k l)) / 2 * kron m (mirror k l) =
          (A.ievn k ε (A.z k m) - A.iodd k ε (A.z k m)) / 2 * kron l (mirror k m) := by
        rw [kron_mirror_comm k m l]
        have := A.mul_kron_mirror_eq k hμ (fun z ↦ (A.ievn k ε z - A.iodd k ε z) / 2)
          (fun z ↦ by rw [A.ievn_neg, A.iodd_neg]) l m
        simpa using this.symm
      rw [← h₁, h₂]
    rw [hc, invHF, hmb]
    field_simp
    ring
  · rw [hep, hem, h.delta_part n m, A.Nph_of_mu k ε hμ, A.Nph_of_mu k ε hμ, h.u_sq, ghat]
    field_simp
    ring

/-- **Corollary "half filling"**, the closed form: `β²(χ_r⁻¹ - B⁻¹) = Γ_r` of Eq. (Gamma-hf). -/
theorem HFHyp.gammaHF_eq (h : A.HFHyp k ε u) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (A.invHF k ε u n m - kron n m / A.Bph k n) = A.gammaHF k ε u n m := by
  have hμ := h.μ_eq
  have hβ := A.β_ne_zero
  have hPn := A.Pph_z_ne_zero k n
  have hGn := A.G₂_z_ne_zero k n
  have hD := h.D_ne
  rw [invHF, gammaHF, A.Bph_eq k n]
  field_simp
  ring

variable (A k ε) in
/-- Where `e(zₙ) ≠ 0`, the orbit form `gammaHF` is Eq. (Gamma-hf) as printed. -/
theorem gammaHF_eq_note (hμ : A.μ = 0) {u : ℂ} {n : ℤ} (he : A.eph k ε (A.z k n) ≠ 0)
    (m : ℤ) : A.gammaHF k ε u n m = A.gammaHFnote k ε u n m := by
  have hβ := A.β_ne_zero
  have hGn := A.G₂_z_ne_zero k n
  rw [A.eph_of_mu' k ε hμ] at he
  have h₁ := left_ne_zero_of_mul he
  have h₂ := right_ne_zero_of_mul he
  rw [gammaHF, gammaHFnote, mirrorBlock, Fph, A.eph_of_mu' k ε hμ, A.Hph_of_mu k ε hμ,
    A.lam_p₀ k ε]
  simp only [ievn, iodd]
  field_simp
  ring

variable (A k ε) in
/-- `χ_r` is symmetric at `μ = 0` (via Proposition "structure"). -/
theorem chiR_symm_hf (hμ : A.μ = 0) (n m : ℤ) : A.chiR k ε n m = A.chiR k ε m n := by
  by_cases hnm : n = m
  · rw [hnm]
  have hPn := A.Pph_z_ne_zero k n
  have hPm := A.Pph_z_ne_zero k m
  have h₁ := A.Pph_mul_chiR_hf k ε hμ n m
  have h₂ := A.Pph_mul_chiR_hf k ε hμ m n
  rw [kron_of_ne hnm, mul_zero, zero_add] at h₁
  rw [kron_of_ne (Ne.symm hnm), mul_zero, zero_add, ← kron_mirror_comm] at h₂
  have hN : A.Nph k ε (A.z k m) (A.z k n) = A.Nph k ε (A.z k n) (A.z k m) := by
    rw [A.Nph_of_mu k ε hμ, A.Nph_of_mu k ε hμ]
    simp only [cN₀, N₂, A.rho_eq_zero_of_mu hμ]
    ring
  have hPd := A.mul_kron_mirror_eq k hμ (A.Pph k) (A.Pph_neg k) m n
  apply mul_left_cancel₀ (mul_ne_zero hPn hPm)
  rw [hN] at h₂
  field_simp at h₁ h₂
  linear_combination h₁ - h₂ - A.lam ε * (A.β * A.p₀) * hPd

theorem HFHyp.invHF_symm (h : A.HFHyp k ε u) (n m : ℤ) :
    A.invHF k ε u n m = A.invHF k ε u m n := by
  rw [invHF, invHF, mirrorBlock_symm (fun j ↦ by rw [A.z_mirror_of_mu k h.μ_eq, A.ievn_neg])
    (fun j ↦ by rw [A.z_mirror_of_mu k h.μ_eq, A.iodd_neg])]
  ring

/-- **Corollary "half filling"**, two-sidedness: `invHF χ_r = 1`, as `χ_r` and `invHF` are
symmetric. -/
theorem HFHyp.chiR_isLeftInverse (h : A.HFHyp k ε u) :
    IsLeftInverse (A.chiR k ε) (A.invHF k ε u) := by
  intro n m
  have := h.chiR_isRightInverse m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [A.chiR_symm_hf k ε h.μ_eq l m, h.invHF_symm n l, mul_comm]

end Params

end HubbardAtom
