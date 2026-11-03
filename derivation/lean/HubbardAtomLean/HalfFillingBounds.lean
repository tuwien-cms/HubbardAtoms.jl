import HubbardAtomLean.HalfFilling
import HubbardAtomLean.SingletBounds

/-!
# Bounds on the entries of `χ_r` for every `μ`

Item 6(a) for the density and magnetic channels: for every `μ`, the entries of `χ_r` on the
lattice `z'ₙ` of `OpLatticeAll.lean` decay as those of the bubble `diag(-β/z'ₙ²)`.

* `OpLattice.ChiDecay.of_eventuallyEq`: the bounds carry over to a lattice whose points differ in
  finitely many places, so those of `ChiBounds.lean` hold on `z'ₙ` for `μ ≠ 0`;
* `Params.norm_chiR_le_hf`, `Params.norm_chiR_diag_le_hf`: at `μ = 0`,
  `|χ_r,nm| ≤ c|z'ₙ|⁻²|z'ₘ|⁻²` for `n ≠ m` and `|χ_r,nn + β/z'ₙ²| ≤ c|z'ₙ|⁻³`;
* `Params.chiDecay_chiR_all`, `Params.bubbleDecay_Bph`: the bounds of `ChiOperator.lean` for `χ_r`
  and for the bubble `B`.

At `μ = 0` the bounds follow from Proposition "structure" with the Cauchy matrix the mirror
(`Pph_mul_chiR_hf`): the mirror term `λβp₀/P(zₙ)` decays as `|zₙ|⁻⁴`, `z'_{n̄}/z'ₙ` is bounded, and
`N(z, v) = N₀(v) + z²N₂(v)` with `N₂` quadratic and, as `ρ = 0`, `N₀` quadratic.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace OpLattice

variable {L L' : OpLattice} {X : ℤ → ℤ → ℂ}

/-- A real sequence bounded for all but finitely many `n` is bounded. -/
theorem exists_le_of_eventually_le {f : ℤ → ℝ} {C : ℝ} (h : ∀ᶠ n in cofinite, f n ≤ C) :
    ∃ C', ∀ n, f n ≤ C' := by
  obtain ⟨C', hC'⟩ := (isBoundedUnder_of_eventually_le h).bddAbove_range_of_cofinite
  exact ⟨C', fun n ↦ hC' ⟨n, rfl⟩⟩

/-- **The bounds carry over** to a lattice with the same `β` whose points differ in finitely many
places. -/
theorem ChiDecay.of_eventuallyEq (hβ : L'.β = L.β) (hz : ∀ᶠ n in cofinite, L'.z n = L.z n)
    (hX : L.ChiDecay X) : L'.ChiDecay X := by
  -- `|z'ₙ| ≤ R|zₙ|`
  obtain ⟨R, hR⟩ := exists_le_of_eventually_le (f := fun n ↦ ‖L'.z n‖ / ‖L.z n‖) (C := 1)
    (hz.mono fun n h ↦ by simp [h, L.z_ne_zero n])
  have hR' : ∀ n, ‖L'.z n‖ ≤ R * ‖L.z n‖ := fun n ↦
    (div_le_iff₀ (norm_pos_iff.mpr (L.z_ne_zero n))).mp (hR n)
  obtain ⟨c, hc⟩ := hX.off
  obtain ⟨c₂, hc₂⟩ := hX.diag
  obtain ⟨C, hC⟩ := exists_le_of_eventually_le (C := c₂)
    (f := fun n ↦ ‖L'.z n‖ ^ 3 * ‖X n n + L'.β / L'.z n ^ 2‖) (hz.mono fun n h ↦ by
      have := norm_pos_iff.mpr (L.z_ne_zero n)
      have h2 := hc₂ n
      rw [le_div_iff₀ (by positivity)] at h2
      simp only [h, hβ]
      linarith)
  refine ⟨⟨R ^ 4 * c, fun n m hnm ↦ ?_⟩, ⟨C, fun n ↦ ?_⟩⟩
  · have := norm_pos_iff.mpr (L.z_ne_zero n)
    have := norm_pos_iff.mpr (L.z_ne_zero m)
    have := norm_pos_iff.mpr (L'.z_ne_zero n)
    have := norm_pos_iff.mpr (L'.z_ne_zero m)
    have h := hc n m hnm
    rw [le_div_iff₀ (by positivity)] at h ⊢
    calc ‖X n m‖ * (‖L'.z n‖ ^ 2 * ‖L'.z m‖ ^ 2)
        ≤ ‖X n m‖ * ((R * ‖L.z n‖) ^ 2 * (R * ‖L.z m‖) ^ 2) := by
          gcongr
          exacts [hR' n, hR' m]
      _ = R ^ 4 * (‖X n m‖ * (‖L.z n‖ ^ 2 * ‖L.z m‖ ^ 2)) := by ring
      _ ≤ R ^ 4 * c := by gcongr
  · have := norm_pos_iff.mpr (L'.z_ne_zero n)
    rw [le_div_iff₀ (by positivity), mul_comm]
    exact hC n

end OpLattice

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `z'ₙ⁴/P(zₙ)` is bounded. -/
theorem bdd_zOp_pow_div_Pph : ∃ C, ∀ n, ‖A.zOp k n ^ 4 / A.Pph k (A.z k n)‖ ≤ C :=
  A.bdd_of_inv_zOp k (fun t ↦ (1 + (2 * A.q₀ k - 4 * A.a ^ 2) * t ^ 2 + A.q₀ k ^ 2 * t ^ 4)⁻¹)
    (ContinuousAt.inv₀ (by fun_prop) (by simp)) <|
    (A.eventually_zOp_eq_ne k).mono fun n ⟨_, h, hz⟩ ↦ by
      rw [h, A.Pph_eq]
      field_simp

/-- `zₙ/z'ₙ` is bounded. -/
theorem bdd_z_div_zOp : ∃ C, ∀ n, ‖A.z k n / A.zOp k n‖ ≤ C :=
  A.bdd_of_inv_zOp k (fun _ ↦ 1) continuousAt_const <|
    (A.eventually_zOp_eq_ne k).mono fun n ⟨_, h, hz⟩ ↦ by rw [h, div_self hz]

/-- `N₂(zₙ)/z'ₙ²` is bounded. -/
theorem bdd_N₂_div_zOp_sq : ∃ C, ∀ n, ‖A.N₂ k ε (A.z k n) / A.zOp k n ^ 2‖ ≤ C :=
  A.bdd_of_inv_zOp k (fun t ↦ -2 * ε * A.a * (1 + 2 * A.a * A.ρ * t + A.q₀ k * t ^ 2) +
      kron k 0 * A.ζ ε * (1 - A.a ^ 2 * t ^ 2)) (by fun_prop) <|
    (A.eventually_zOp_eq_ne k).mono fun n ⟨_, h, hz⟩ ↦ by
      rw [h]
      simp only [N₂]
      field_simp

/-- At `μ = 0`, `N₀` is quadratic, so `N₀(zₙ)/z'ₙ²` is bounded. -/
theorem bdd_N₀_div_zOp_sq_hf (hμ : A.μ = 0) :
    ∃ C, ∀ n, ‖A.N₀ k ε (A.z k n) / A.zOp k n ^ 2‖ ≤ C := by
  obtain ⟨c₀, c₁, c₂, hN₀⟩ := A.N₀_cubic k ε
  have hρ := A.rho_eq_zero_of_mu hμ
  exact A.bdd_of_inv_zOp k (fun t ↦ c₂ + c₁ * t + c₀ * t ^ 2) (by fun_prop) <|
    (A.eventually_zOp_eq_ne k).mono fun n ⟨_, h, hz⟩ ↦ by
      rw [h, hN₀, hρ]
      field_simp
      ring

/-- **Off-diagonal entries** at `μ = 0`: `|χ_r,nm| ≤ c|z'ₙ|⁻²|z'ₘ|⁻²` for `n ≠ m`. -/
theorem norm_chiR_le_hf (hμ : A.μ = 0) :
    ∃ c, ∀ n m, n ≠ m → ‖A.chiR k ε n m‖ ≤ c / (‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2) := by
  obtain ⟨CP, hCP⟩ := A.bdd_zOp_pow_div_Pph k
  obtain ⟨Ci, hCi⟩ := A.bdd_inv_zOp k
  obtain ⟨CW, hCW⟩ := A.bdd_z_div_zOp k
  obtain ⟨C₀, hC₀⟩ := A.bdd_N₀_div_zOp_sq_hf k ε hμ
  obtain ⟨C₂, hC₂⟩ := A.bdd_N₂_div_zOp_sq k ε
  obtain ⟨CM, hCM⟩ := A.bdd_zOp_mirror_div k
  have hCP0 := (norm_nonneg _).trans (hCP 0)
  set K : ℂ := A.lam ε * (A.β * A.p₀)
  refine ⟨CP * (‖K‖ * CM ^ 2 + Ci ^ 2 * CP * C₀ + CW ^ 2 * CP * C₂), fun n m hnm ↦ ?_⟩
  have hz := A.zOp_ne_zero k n
  have hv := A.zOp_ne_zero k m
  have hPz := A.Pph_z_ne_zero k n
  have hPv := A.Pph_z_ne_zero k m
  have hχ : A.chiR k ε n m = (K * kron m (mirror k n) +
      (A.N₀ k ε (A.z k m) + A.z k n ^ 2 * A.N₂ k ε (A.z k m)) / A.Pph k (A.z k m)) /
        A.Pph k (A.z k n) := by
    rw [eq_div_iff hPz, mul_comm, A.Pph_mul_chiR_hf k ε hμ, kron_of_ne hnm, Nph]
    ring
  -- `z'ₙ²z'ₘ²χ_r,nm = pₙ[K(z'ₘ/z'ₙ)²δ_{m n̄} + z'ₙ⁻²pₘN₀(zₘ)/z'ₘ² + (zₙ/z'ₙ)²pₘN₂(zₘ)/z'ₘ²]`
  -- with `pₙ = z'ₙ⁴/P(zₙ)`
  have key : A.zOp k n ^ 2 * A.zOp k m ^ 2 * A.chiR k ε n m =
      A.zOp k n ^ 4 / A.Pph k (A.z k n) *
        (K * (A.zOp k m / A.zOp k n) ^ 2 * kron m (mirror k n) +
          (A.zOp k n)⁻¹ ^ 2 * (A.zOp k m ^ 4 / A.Pph k (A.z k m)) *
            (A.N₀ k ε (A.z k m) / A.zOp k m ^ 2) +
          (A.z k n / A.zOp k n) ^ 2 * (A.zOp k m ^ 4 / A.Pph k (A.z k m)) *
            (A.N₂ k ε (A.z k m) / A.zOp k m ^ 2)) := by
    rw [hχ]
    field_simp
    ring
  have hmir : ‖K * (A.zOp k m / A.zOp k n) ^ 2 * kron m (mirror k n)‖ ≤ ‖K‖ * CM ^ 2 := by
    by_cases hm : m = mirror k n
    · subst hm
      rw [kron_self, mul_one, norm_mul, norm_pow]
      gcongr
      exact hCM n
    · rw [kron_of_ne hm, mul_zero, norm_zero]
      positivity
  have hb : ‖A.zOp k n ^ 2 * A.zOp k m ^ 2 * A.chiR k ε n m‖ ≤
      CP * (‖K‖ * CM ^ 2 + Ci ^ 2 * CP * C₀ + CW ^ 2 * CP * C₂) := by
    rw [key, norm_mul]
    refine mul_le_mul (hCP n) ((norm_add_le _ _).trans (add_le_add
      ((norm_add_le _ _).trans (add_le_add hmir ?_)) ?_)) (norm_nonneg _) hCP0 <;>
      rw [norm_mul, norm_mul, norm_pow] <;> gcongr
    exacts [hCi n, hCP m, hC₀ m, hCW n, hCP m, hC₂ m]
  have := norm_pos_iff.mpr hz
  have := norm_pos_iff.mpr hv
  rw [le_div_iff₀ (by positivity)]
  calc ‖A.chiR k ε n m‖ * (‖A.zOp k n‖ ^ 2 * ‖A.zOp k m‖ ^ 2)
      = ‖A.zOp k n ^ 2 * A.zOp k m ^ 2 * A.chiR k ε n m‖ := by
        rw [norm_mul, norm_mul, norm_pow, norm_pow]
        ring
    _ ≤ _ := hb

/-- **Diagonal entries** at `μ = 0`: `|χ_r,nn + β/z'ₙ²| ≤ c|z'ₙ|⁻³`. Off the mirror fixed point
`z'ₙ³(χ_r,nn + β/z'ₙ²)` is the function of `1/zₙ` of `norm_chiR_diag_le` at `ρ = 0`. -/
theorem norm_chiR_diag_le_hf (hμ : A.μ = 0) :
    ∃ c, ∀ n, ‖A.chiR k ε n n + A.β / A.zOp k n ^ 2‖ ≤ c / ‖A.zOp k n‖ ^ 3 := by
  obtain ⟨c₀, c₁, c₂, hN₀⟩ := A.N₀_cubic k ε
  have hρ := A.rho_eq_zero_of_mu hμ
  obtain ⟨C, hC⟩ := A.bdd_of_inv_zOp k
    (f := fun n ↦ A.zOp k n ^ 3 * (A.chiR k ε n n + A.β / A.zOp k n ^ 2))
    (fun t ↦ (1 + (2 * A.q₀ k - 4 * A.a ^ 2) * t ^ 2 + A.q₀ k ^ 2 * t ^ 4)⁻¹ *
      (A.β * ((A.q₀ k - 4 * A.a ^ 2 - 4 * ε * A.a ^ 2 * A.p₁) * t + A.q₀ k ^ 2 * t ^ 3) +
        (1 + (2 * A.q₀ k - 4 * A.a ^ 2) * t ^ 2 + A.q₀ k ^ 2 * t ^ 4)⁻¹ *
          (t ^ 2 * (c₂ * t + c₁ * t ^ 2 + c₀ * t ^ 3) +
            t * (-2 * ε * A.a * (1 + A.q₀ k * t ^ 2) +
              kron k 0 * A.ζ ε * (1 - A.a ^ 2 * t ^ 2)))))
    (by fun_prop (disch := simp)) <|
    (A.eventually_zOp_eq_ne k).mono fun n ⟨hn, h, hz⟩ ↦ by
      have hP := A.Pph_z_ne_zero k n
      have hD : 1 + (2 * A.q₀ k - 4 * A.a ^ 2) * (A.z k n)⁻¹ ^ 2 +
          A.q₀ k ^ 2 * (A.z k n)⁻¹ ^ 4 = A.Pph k (A.z k n) / A.z k n ^ 4 := by
        rw [A.Pph_eq]
        field_simp
      have hχ : A.chiR k ε n n = (-A.β * A.Hph k ε (A.z k n) +
          A.Nph k ε (A.z k n) (A.z k n) / A.Pph k (A.z k n)) / A.Pph k (A.z k n) := by
        rw [eq_div_iff hP, mul_comm, A.Pph_mul_chiR_hf k ε hμ, kron_self, kron_of_ne (Ne.symm hn)]
        ring
      simp only [h, hD, hχ, Nph, hN₀, hρ]
      field_simp
      simp only [Hph, N₂, A.Pph_eq, hρ]
      ring
  refine ⟨C, fun n ↦ ?_⟩
  have hz := norm_pos_iff.mpr (A.zOp_ne_zero k n)
  have h := hC n
  rw [norm_mul, norm_pow] at h
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- **The bounds of item 6(a)** for `χ_r`, for every `μ`: from `ChiBounds.lean` for `μ ≠ 0`, and
from `norm_chiR_le_hf`, `norm_chiR_diag_le_hf` at `μ = 0`. -/
theorem chiDecay_chiR_all : (A.opLatticeAll k).ChiDecay (A.chiR k ε) := by
  by_cases hμ : A.μ = 0
  · exact ⟨A.norm_chiR_le_hf k ε hμ, A.norm_chiR_diag_le_hf k ε hμ⟩
  · exact (A.chiDecay_chiR k ε hμ).of_eventuallyEq (L' := A.opLatticeAll k)
      (L := A.opLattice k hμ) rfl (A.eventually_zOp_eq k)

/-- `B ≠ 0` on the lattice. -/
theorem Bph_ne_zero (n : ℤ) : A.Bph k n ≠ 0 := by
  rw [A.Bph_eq]
  exact div_ne_zero (mul_ne_zero (neg_ne_zero.mpr A.β_ne_zero) (A.G₂_z_ne_zero k n))
    (A.Pph_z_ne_zero k n)

/-- **The bubble** `B = -βG₂(zₙ)/P(zₙ)` decays as `-β/z'ₙ²`, for every `μ`: off the mirror fixed
point, `z'ₙ(tₙ - 1)` with `tₙ = zₙ²G₂(zₙ)/P(zₙ)` is a function of `1/zₙ` continuous at `0`. -/
theorem bubbleDecay_Bph : (A.opLatticeAll k).BubbleDecay (A.Bph k) := by
  refine ⟨A.Bph_ne_zero k, A.bdd_of_inv_zOp k
    (fun t ↦ (1 + (2 * A.q₀ k - 4 * A.a ^ 2) * t ^ 2 + A.q₀ k ^ 2 * t ^ 4)⁻¹ *
      (2 * A.a * A.ρ + (A.a ^ 2 * A.ρ ^ 2 - A.w k ^ 2 / 4 - (2 * A.q₀ k - 4 * A.a ^ 2)) * t -
        A.q₀ k ^ 2 * t ^ 3))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) <|
    (A.eventually_zOp_eq_ne k).mono fun n ⟨_, h, hz⟩ ↦ ?_⟩
  have hP := A.Pph_z_ne_zero k n
  have hβ := A.β_ne_zero
  have hD : 1 + (2 * A.q₀ k - 4 * A.a ^ 2) * (A.z k n)⁻¹ ^ 2 + A.q₀ k ^ 2 * (A.z k n)⁻¹ ^ 4 =
      A.Pph k (A.z k n) / A.z k n ^ 4 := by
    rw [A.Pph_eq]
    field_simp
  change A.zOp k n * (-(A.zOp k n ^ 2 / A.β) * A.Bph k n - 1) = _
  rw [h, hD, A.Bph_eq]
  field_simp
  simp only [G₂, A.Pph_eq]
  ring

end Params

end HubbardAtom
