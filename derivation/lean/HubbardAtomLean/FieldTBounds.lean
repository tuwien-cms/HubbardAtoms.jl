import HubbardAtomLean.ChiOperator
import HubbardAtomLean.FieldTransverse

/-!
# Bounds on the entries of `χ_T↑`

Item 6(a) for the transverse sector: for `μ ≠ 0` and `W = iω_k - 2h ≠ 0`, the entries of `χ_T↑`
on its lattice `zₙ = μ + i(νₙ + ω_k/2)` decay as those of the bubble `diag(-β/zₙ²)`:

* `FParams.norm_chiTUp_le`: `|χ_T↑,nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m`;
* `FParams.norm_chiTUp_diag_le`: `|χ_T↑,nn + β/zₙ²| ≤ c|zₙ|⁻³`;
* `FParams.chiDecay_chiTUp`, `FParams.bubbleDecay_bTUp`: the bounds of `ChiOperator.lean`, for
  `χ_T↑` and for the bubble `b_T↑ = -βG₂(zₙ)/P(zₙ)`.

The transverse input (`PT_mul_chiTUp`) has the form of Proposition "structure" for the density and
magnetic channels, with `P`, `H`, `N`, `λ` of `FieldInput.lean`, so the proofs are those of
`ChiBounds.lean`.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-- `zₙ ≠ 0` for `μ ≠ 0`. -/
theorem zT_ne_zero (hμ : F.μ ≠ 0) (n : ℤ) : F.zT k n ≠ 0 := (F.lat F.μ).z_ne_zero k hμ n

/-- The transverse lattice is the lattice of `F.lat F.μ`. -/
theorem lat_z_eq_zT (n : ℤ) : (F.lat F.μ).z k n = F.zT k n := rfl

/-- The transverse lattice `zₙ = μ + i(νₙ + ω_k/2)` as an `OpLattice`, for `μ ≠ 0`. -/
noncomputable def opLatT (hμ : F.μ ≠ 0) : OpLattice := (F.lat F.μ).opLattice k hμ

/-- `P(v) = v⁴ + (2q₀ - 4a²)v² + q₀²`. -/
theorem PT_eq (v : ℂ) :
    F.PT k v = v ^ 4 + (2 * F.q₀T k - 4 * F.a ^ 2) * v ^ 2 + F.q₀T k ^ 2 := by
  simp only [PT, QpT, QmT, q₀T]
  ring

/-- `N₀` is cubic with leading coefficient `-λρ`. -/
theorem N₀T_cubic : ∃ c₀ c₁ c₂ : ℂ, ∀ v,
    F.N₀T k v = -(F.lamT * F.ρ) * v ^ 3 + c₂ * v ^ 2 + c₁ * v + c₀ :=
  ⟨(2 * F.q₀T k - 4 * F.a ^ 2) * (2 * F.a - F.ζT k) * F.q₀T k -
      (2 * F.a + F.ζT k) * F.q₀T k ^ 2,
    4 * F.a ^ 2 * F.ρ * (2 * F.q₀T k - 4 * F.a ^ 2), (2 * F.a - F.ζT k) * F.q₀T k,
    fun v ↦ by simp only [N₀T, N₂T, F.PT_eq, lamT]; ring⟩

/-- `zₙ⁴/P(zₙ)` is bounded. -/
theorem bdd_zT_pow_div_PT (hμ : F.μ ≠ 0) : ∃ C, ∀ n, ‖F.zT k n ^ 4 / F.PT k (F.zT k n)‖ ≤ C := by
  refine (F.lat F.μ).bdd_of_inv_z k
    (fun t ↦ (1 + (2 * F.q₀T k - 4 * F.a ^ 2) * t ^ 2 + F.q₀T k ^ 2 * t ^ 4)⁻¹)
    (ContinuousAt.inv₀ (by fun_prop) (by simp)) fun n ↦ ?_
  have hz := F.zT_ne_zero k hμ n
  simp only [F.lat_z_eq_zT k]
  rw [F.PT_eq]
  field_simp

/-- **Off-diagonal entries**: for `μ ≠ 0`, `W ≠ 0`, `|χ_T↑,nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m`. -/
theorem norm_chiTUp_le (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) :
    ∃ c, ∀ n m, n ≠ m → ‖F.chiTUp k n m‖ ≤ c / (‖F.zT k n‖ ^ 2 * ‖F.zT k m‖ ^ 2) := by
  obtain ⟨c₀, c₁, c₂, hN₀⟩ := F.N₀T_cubic k
  obtain ⟨C₁, hC₁⟩ := F.bdd_zT_pow_div_PT k hμ
  -- `N₂(v)/v²`, `N₀(v)/v³` and `R₃(v)/v³` with `R₃(v) = λρP(v) + vN₀(v)` (cubic)
  obtain ⟨C₂, hC₂⟩ := (F.lat F.μ).bdd_of_inv_z k (f := fun n ↦ F.N₂T k (F.zT k n) / F.zT k n ^ 2)
    (fun t ↦ 2 * F.a * (1 + 2 * F.a * F.ρ * t + F.q₀T k * t ^ 2) +
      F.ζT k * (1 - F.q₀T k * t ^ 2)) (by fun_prop) fun n ↦ by
    have hz := F.zT_ne_zero k hμ n
    simp only [F.lat_z_eq_zT k]
    simp only [N₂T]
    field_simp
  obtain ⟨C₃, hC₃⟩ := (F.lat F.μ).bdd_of_inv_z k (f := fun n ↦ F.N₀T k (F.zT k n) / F.zT k n ^ 3)
    (fun t ↦ -(F.lamT * F.ρ) + c₂ * t + c₁ * t ^ 2 + c₀ * t ^ 3) (by fun_prop) fun n ↦ by
    have hz := F.zT_ne_zero k hμ n
    simp only [F.lat_z_eq_zT k]
    simp only [hN₀]
    field_simp
  obtain ⟨C₄, hC₄⟩ := (F.lat F.μ).bdd_of_inv_z k
    (f := fun n ↦ (F.lamT * F.ρ * F.PT k (F.zT k n) + F.zT k n * F.N₀T k (F.zT k n)) /
      F.zT k n ^ 3)
    (fun t ↦ c₂ + (F.lamT * F.ρ * (2 * F.q₀T k - 4 * F.a ^ 2) + c₁) * t + c₀ * t ^ 2 +
      F.lamT * F.ρ * F.q₀T k ^ 2 * t ^ 3) (by fun_prop) fun n ↦ by
    have hz := F.zT_ne_zero k hμ n
    simp only [F.lat_z_eq_zT k]
    simp only [hN₀, F.PT_eq]
    field_simp
    ring
  have hμ' : 0 < |F.μ| := abs_pos.mpr hμ
  have hC₁0 := (norm_nonneg _).trans (hC₁ 0)
  have hC₃0 := (norm_nonneg _).trans (hC₃ 0)
  have hC₄0 := (norm_nonneg _).trans (hC₄ 0)
  refine ⟨C₁ * C₁ * (C₂ + (C₃ + C₄ * |F.μ|⁻¹) * (|F.μ|⁻¹ + (2 * |F.μ|)⁻¹)),
    fun n m hnm ↦ ?_⟩
  have hz := F.zT_ne_zero k hμ n
  have hv := F.zT_ne_zero k hμ m
  have hzv := F.zT_add_zT_ne_zero k hμ n m
  have hPz := F.PT_zT_ne_zero k n
  have hPv := F.PT_zT_ne_zero k m
  have hχ : F.chiTUp k n m = (F.lamT * F.ρ / (F.zT k n + F.zT k m) +
      F.NT k (F.zT k n) (F.zT k m) / F.PT k (F.zT k m)) / F.PT k (F.zT k n) := by
    rw [eq_div_iff hPz, mul_comm, F.PT_mul_chiTUp k hμ hW, kron_of_ne hnm]
    ring
  have key : F.zT k n ^ 2 * F.zT k m ^ 2 * F.chiTUp k n m =
      F.zT k n ^ 4 / F.PT k (F.zT k n) * (F.zT k m ^ 4 / F.PT k (F.zT k m)) *
        (F.N₂T k (F.zT k m) / F.zT k m ^ 2 + (F.N₀T k (F.zT k m) / F.zT k m ^ 3 +
          (F.lamT * F.ρ * F.PT k (F.zT k m) + F.zT k m * F.N₀T k (F.zT k m)) /
            F.zT k m ^ 3 * (F.zT k n)⁻¹) * ((F.zT k n)⁻¹ - (F.zT k n + F.zT k m)⁻¹)) := by
    rw [hχ, NT]
    field_simp
    ring
  have hb : ‖F.zT k n ^ 2 * F.zT k m ^ 2 * F.chiTUp k n m‖ ≤
      C₁ * C₁ * (C₂ + (C₃ + C₄ * |F.μ|⁻¹) * (|F.μ|⁻¹ + (2 * |F.μ|)⁻¹)) := by
    rw [key, norm_mul, norm_mul]
    refine mul_le_mul (mul_le_mul (hC₁ n) (hC₁ m) (norm_nonneg _) hC₁0) ?_ (norm_nonneg _)
      (mul_nonneg hC₁0 hC₁0)
    refine (norm_add_le _ _).trans (add_le_add (hC₂ m) ?_)
    rw [norm_mul]
    refine mul_le_mul ?_ ((norm_sub_le _ _).trans (add_le_add
      ((F.lat F.μ).norm_inv_z_le k hμ n) ((F.lat F.μ).norm_inv_z_add_z_le k hμ n m)))
      (norm_nonneg _) (add_nonneg hC₃0 (mul_nonneg hC₄0 (inv_nonneg.mpr hμ'.le)))
    refine (norm_add_le _ _).trans (add_le_add (hC₃ m) ?_)
    rw [norm_mul]
    exact mul_le_mul (hC₄ m) ((F.lat F.μ).norm_inv_z_le k hμ n) (norm_nonneg _) hC₄0
  rw [norm_mul, norm_mul, norm_pow, norm_pow] at hb
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- **Diagonal entries**: for `μ ≠ 0`, `W ≠ 0`, `|χ_T↑,nn + β/zₙ²| ≤ c|zₙ|⁻³`, i.e.
`χ_T↑,nn = -βzₙ⁻²[1 + O(|zₙ|⁻¹)]`. -/
theorem norm_chiTUp_diag_le (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) :
    ∃ c, ∀ n, ‖F.chiTUp k n n + F.β / F.zT k n ^ 2‖ ≤ c / ‖F.zT k n‖ ^ 3 := by
  obtain ⟨c₀, c₁, c₂, hN₀⟩ := F.N₀T_cubic k
  -- with `D(t) = P(z)/z⁴`, `z³(χ_nn + β/z²) = D⁻¹[β(P - z²H)/z³ + λρ/(2z²) + D⁻¹N(z, z)/z⁵]`
  obtain ⟨C, hC⟩ := (F.lat F.μ).bdd_of_inv_z k
    (f := fun n ↦ F.zT k n ^ 3 * (F.chiTUp k n n + F.β / F.zT k n ^ 2))
    (fun t ↦ (1 + (2 * F.q₀T k - 4 * F.a ^ 2) * t ^ 2 + F.q₀T k ^ 2 * t ^ 4)⁻¹ *
      (F.β * (-2 * F.a * F.ρ + (2 * F.q₀T k - 4 * F.a ^ 2 - F.q₁T k) * t +
          F.q₀T k ^ 2 * t ^ 3) + F.lamT * F.ρ / 2 * t ^ 2 +
        (1 + (2 * F.q₀T k - 4 * F.a ^ 2) * t ^ 2 + F.q₀T k ^ 2 * t ^ 4)⁻¹ *
          (t ^ 2 * (-(F.lamT * F.ρ) + c₂ * t + c₁ * t ^ 2 + c₀ * t ^ 3) +
            t * (2 * F.a * (1 + 2 * F.a * F.ρ * t + F.q₀T k * t ^ 2) +
              F.ζT k * (1 - F.q₀T k * t ^ 2)))))
    (by fun_prop (disch := simp)) fun n ↦ by
    have hz := F.zT_ne_zero k hμ n
    have hP := F.PT_zT_ne_zero k n
    simp only [F.lat_z_eq_zT k]
    have hD : 1 + (2 * F.q₀T k - 4 * F.a ^ 2) * (F.zT k n)⁻¹ ^ 2 +
        F.q₀T k ^ 2 * (F.zT k n)⁻¹ ^ 4 = F.PT k (F.zT k n) / F.zT k n ^ 4 := by
      rw [F.PT_eq]
      field_simp
    have hχ : F.chiTUp k n n = (-F.β * F.HT k (F.zT k n) + F.lamT * F.ρ / (2 * F.zT k n) +
        F.NT k (F.zT k n) (F.zT k n) / F.PT k (F.zT k n)) / F.PT k (F.zT k n) := by
      rw [eq_div_iff hP, mul_comm, F.PT_mul_chiTUp k hμ hW, kron_self, two_mul]
      ring
    simp only [hD, hχ, NT, hN₀]
    field_simp
    simp only [HT, N₂T, F.PT_eq]
    ring
  refine ⟨C, fun n ↦ ?_⟩
  have hz := norm_pos_iff.mpr (F.zT_ne_zero k hμ n)
  have h := hC n
  rw [norm_mul, norm_pow] at h
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- **The bounds of item 6(a)** for `χ_T↑`, as used by `ChiOperator.lean`. -/
theorem chiDecay_chiTUp (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) :
    (F.opLatT k hμ).ChiDecay (F.chiTUp k) :=
  ⟨F.norm_chiTUp_le k hμ hW, F.norm_chiTUp_diag_le k hμ hW⟩

/-- **The bubble** `b_T↑ = -βG₂(zₙ)/P(zₙ)` decays as `-β/zₙ²`: `tₙ = -zₙ²bₙ/β = zₙ²G₂(zₙ)/P(zₙ)` and
`zₙ(tₙ - 1)` is a function of `1/zₙ` continuous at `0`. -/
theorem bubbleDecay_bTUp (hμ : F.μ ≠ 0) : (F.opLatT k hμ).BubbleDecay (F.bTUp k) := by
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  refine ⟨fun n ↦ ?_, ?_⟩
  · rw [F.bTUp_eq]
    exact div_ne_zero (mul_ne_zero (neg_ne_zero.mpr hβ) (F.G₂T_zT_ne_zero k n))
      (F.PT_zT_ne_zero k n)
  obtain ⟨C, hC⟩ := (F.lat F.μ).bdd_of_inv_z k
    (f := fun n ↦ F.zT k n * (-(F.zT k n ^ 2 / F.β) * F.bTUp k n - 1))
    (fun t ↦ (1 + (2 * F.q₀T k - 4 * F.a ^ 2) * t ^ 2 + F.q₀T k ^ 2 * t ^ 4)⁻¹ *
      (2 * F.a * F.ρ + (F.a ^ 2 * F.ρ ^ 2 - (F.WT k / 2 + F.a * F.mf) ^ 2 -
        (2 * F.q₀T k - 4 * F.a ^ 2)) * t - F.q₀T k ^ 2 * t ^ 3))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) fun n ↦ by
    have hz := F.zT_ne_zero k hμ n
    have hP := F.PT_zT_ne_zero k n
    simp only [F.lat_z_eq_zT k]
    have hD : 1 + (2 * F.q₀T k - 4 * F.a ^ 2) * (F.zT k n)⁻¹ ^ 2 +
        F.q₀T k ^ 2 * (F.zT k n)⁻¹ ^ 4 = F.PT k (F.zT k n) / F.zT k n ^ 4 := by
      rw [F.PT_eq]
      field_simp
    rw [hD, F.bTUp_eq]
    field_simp
    simp only [G₂T, F.PT_eq]
    ring
  exact ⟨C, hC⟩

end FParams

end HubbardAtom
