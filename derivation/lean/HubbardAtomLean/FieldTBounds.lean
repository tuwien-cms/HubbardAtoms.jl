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

/-! ### The entries through `t = 1/z` -/

/-- The coefficient `c₂` of `N₀(v) = -λρv³ + c₂v² + c₁v + c₀`. -/
noncomputable def cT₂ : ℂ := (2 * F.a - F.ζT k) * F.q₀T k

/-- The coefficient `c₁` of `N₀`. -/
noncomputable def cT₁ : ℂ := 4 * F.a ^ 2 * F.ρ * (2 * F.q₀T k - 4 * F.a ^ 2)

/-- The coefficient `c₀` of `N₀`. -/
noncomputable def cT₀ : ℂ :=
  (2 * F.q₀T k - 4 * F.a ^ 2) * (2 * F.a - F.ζT k) * F.q₀T k - (2 * F.a + F.ζT k) * F.q₀T k ^ 2

/-- `N₀` is cubic with leading coefficient `-λρ`. -/
theorem N₀T_eq (v : ℂ) :
    F.N₀T k v = -(F.lamT * F.ρ) * v ^ 3 + F.cT₂ k * v ^ 2 + F.cT₁ k * v + F.cT₀ k := by
  simp only [N₀T, N₂T, F.PT_eq, lamT, cT₀, cT₁, cT₂]
  ring

/-- `D(t) = P(z)/z⁴` at `t = 1/z`. -/
noncomputable def dT (t : ℂ) : ℂ := 1 + (2 * F.q₀T k - 4 * F.a ^ 2) * t ^ 2 + F.q₀T k ^ 2 * t ^ 4

/-- `N₂(z)/z²` at `t = 1/z`. -/
noncomputable def n2T (t : ℂ) : ℂ :=
  2 * F.a * (1 + 2 * F.a * F.ρ * t + F.q₀T k * t ^ 2) + F.ζT k * (1 - F.q₀T k * t ^ 2)

/-- `N₀(z)/z³` at `t = 1/z`. -/
noncomputable def n0T (t : ℂ) : ℂ :=
  -(F.lamT * F.ρ) + F.cT₂ k * t + F.cT₁ k * t ^ 2 + F.cT₀ k * t ^ 3

/-- `[λρP(z) + zN₀(z)]/z³` at `t = 1/z` (the quartic terms cancel). -/
noncomputable def r3T (t : ℂ) : ℂ :=
  F.cT₂ k + (F.lamT * F.ρ * (2 * F.q₀T k - 4 * F.a ^ 2) + F.cT₁ k) * t + F.cT₀ k * t ^ 2 +
    F.lamT * F.ρ * F.q₀T k ^ 2 * t ^ 3

/-- `z³(χ_nn + β/z²) = D⁻¹[β(P - z²H)/z³ + λρ/(2z²) + D⁻¹N(z, z)/z⁵]` at `t = 1/z`. -/
noncomputable def dgT (t : ℂ) : ℂ :=
  (F.dT k t)⁻¹ * (F.β * (-2 * F.a * F.ρ + (2 * F.q₀T k - 4 * F.a ^ 2 - F.q₁T k) * t +
    F.q₀T k ^ 2 * t ^ 3) + F.lamT * F.ρ / 2 * t ^ 2 +
      (F.dT k t)⁻¹ * (t ^ 2 * F.n0T k t + t * F.n2T k t))

/-- `D(1/z) = P(z)/z⁴`. -/
theorem dT_inv {z : ℂ} (hz : z ≠ 0) : F.dT k z⁻¹ = F.PT k z / z ^ 4 := by
  rw [dT, F.PT_eq]
  field_simp

/-- `z⁴/P(z) = D(1/z)⁻¹`. -/
theorem pow_div_PT {z : ℂ} (hz : z ≠ 0) : z ^ 4 / F.PT k z = (F.dT k z⁻¹)⁻¹ := by
  rw [F.dT_inv k hz, inv_div]

/-- `N₂(z)/z² = n2T(1/z)`. -/
theorem N₂T_div {z : ℂ} (hz : z ≠ 0) : F.N₂T k z / z ^ 2 = F.n2T k z⁻¹ := by
  simp only [N₂T, n2T]
  field_simp

/-- `N₀(z)/z³ = n0T(1/z)`. -/
theorem N₀T_div {z : ℂ} (hz : z ≠ 0) : F.N₀T k z / z ^ 3 = F.n0T k z⁻¹ := by
  simp only [F.N₀T_eq, n0T]
  field_simp

/-- `[λρP(z) + zN₀(z)]/z³ = r3T(1/z)`. -/
theorem R₃T_div {z : ℂ} (hz : z ≠ 0) :
    (F.lamT * F.ρ * F.PT k z + z * F.N₀T k z) / z ^ 3 = F.r3T k z⁻¹ := by
  simp only [F.N₀T_eq, F.PT_eq, r3T]
  field_simp
  ring

/-- **Off-diagonal entries, with constants**: from bounds on `z⁴/P`, `N₂/z²`, `N₀/z³` and
`[λρP + zN₀]/z³` on the lattice, and `1/|μ| ≤ M`,
`|χ_T↑,nm| ≤ C₁²(C₂ + (C₃ + C₄M)(2M))|zₙ|⁻²|zₘ|⁻²` for `n ≠ m`. -/
theorem norm_chiTUp_le_of (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) {C₁ C₂ C₃ C₄ M : ℝ}
    (hM : |F.μ|⁻¹ ≤ M) (hC₁ : ∀ n, ‖(F.dT k (F.zT k n)⁻¹)⁻¹‖ ≤ C₁)
    (hC₂ : ∀ n, ‖F.n2T k (F.zT k n)⁻¹‖ ≤ C₂) (hC₃ : ∀ n, ‖F.n0T k (F.zT k n)⁻¹‖ ≤ C₃)
    (hC₄ : ∀ n, ‖F.r3T k (F.zT k n)⁻¹‖ ≤ C₄) (n m : ℤ) (hnm : n ≠ m) :
    ‖F.chiTUp k n m‖ ≤ C₁ * C₁ * (C₂ + (C₃ + C₄ * M) * (M + M)) /
      (‖F.zT k n‖ ^ 2 * ‖F.zT k m‖ ^ 2) := by
  have hμ' : 0 < |F.μ| := abs_pos.mpr hμ
  have hM0 : 0 ≤ M := (inv_nonneg.mpr hμ'.le).trans hM
  have hC₁0 := (norm_nonneg _).trans (hC₁ 0)
  have hC₃0 := (norm_nonneg _).trans (hC₃ 0)
  have hC₄0 := (norm_nonneg _).trans (hC₄ 0)
  have hz := F.zT_ne_zero k hμ n
  have hv := F.zT_ne_zero k hμ m
  have hzv := F.zT_add_zT_ne_zero k hμ n m
  have hPz := F.PT_zT_ne_zero k n
  have hPv := F.PT_zT_ne_zero k m
  have ht : ‖(F.zT k n)⁻¹‖ ≤ M := ((F.lat F.μ).norm_inv_z_le k hμ n).trans hM
  have hp : ‖(F.zT k n + F.zT k m)⁻¹‖ ≤ M :=
    ((F.lat F.μ).norm_inv_z_add_z_le k hμ n m).trans
      ((inv_anti₀ hμ' (by rw [lat_μ]; linarith)).trans hM)
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
  rw [F.pow_div_PT k hz, F.pow_div_PT k hv, F.N₂T_div k hv, F.N₀T_div k hv, F.R₃T_div k hv]
    at key
  have hb : ‖F.zT k n ^ 2 * F.zT k m ^ 2 * F.chiTUp k n m‖ ≤
      C₁ * C₁ * (C₂ + (C₃ + C₄ * M) * (M + M)) := by
    rw [key, norm_mul, norm_mul]
    refine mul_le_mul (mul_le_mul (hC₁ n) (hC₁ m) (norm_nonneg _) hC₁0) ?_ (norm_nonneg _)
      (mul_nonneg hC₁0 hC₁0)
    refine (norm_add_le _ _).trans (add_le_add (hC₂ m) ?_)
    rw [norm_mul]
    refine mul_le_mul ?_ ((norm_sub_le _ _).trans (add_le_add ht hp)) (norm_nonneg _)
      (add_nonneg hC₃0 (mul_nonneg hC₄0 hM0))
    refine (norm_add_le _ _).trans (add_le_add (hC₃ m) ?_)
    rw [norm_mul]
    exact mul_le_mul (hC₄ m) ht (norm_nonneg _) hC₄0
  rw [norm_mul, norm_mul, norm_pow, norm_pow] at hb
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- **Off-diagonal entries**: for `μ ≠ 0`, `W ≠ 0`, `|χ_T↑,nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m`. -/
theorem norm_chiTUp_le (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) :
    ∃ c, ∀ n m, n ≠ m → ‖F.chiTUp k n m‖ ≤ c / (‖F.zT k n‖ ^ 2 * ‖F.zT k m‖ ^ 2) := by
  have h₁ : ContinuousAt (fun t ↦ (F.dT k t)⁻¹) 0 := by
    refine ContinuousAt.inv₀ ?_ (by simp [dT])
    unfold dT
    fun_prop
  have h₂ : ContinuousAt (fun t ↦ F.n2T k t) 0 := by simp only [n2T]; fun_prop
  have h₃ : ContinuousAt (fun t ↦ F.n0T k t) 0 := by simp only [n0T]; fun_prop
  have h₄ : ContinuousAt (fun t ↦ F.r3T k t) 0 := by simp only [r3T]; fun_prop
  obtain ⟨C₁, hC₁⟩ := (F.lat F.μ).bdd_of_inv_z k _ h₁ (f := fun n ↦ (F.dT k (F.zT k n)⁻¹)⁻¹)
    fun _ ↦ rfl
  obtain ⟨C₂, hC₂⟩ := (F.lat F.μ).bdd_of_inv_z k _ h₂ (f := fun n ↦ F.n2T k (F.zT k n)⁻¹)
    fun _ ↦ rfl
  obtain ⟨C₃, hC₃⟩ := (F.lat F.μ).bdd_of_inv_z k _ h₃ (f := fun n ↦ F.n0T k (F.zT k n)⁻¹)
    fun _ ↦ rfl
  obtain ⟨C₄, hC₄⟩ := (F.lat F.μ).bdd_of_inv_z k _ h₄ (f := fun n ↦ F.r3T k (F.zT k n)⁻¹)
    fun _ ↦ rfl
  exact ⟨_, F.norm_chiTUp_le_of k hμ hW le_rfl hC₁ hC₂ hC₃ hC₄⟩

/-- `z³(χ_nn + β/z²) = dgT(1/z)`, for `μ ≠ 0`, `W ≠ 0`. -/
theorem dgT_inv (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) (n : ℤ) :
    F.zT k n ^ 3 * (F.chiTUp k n n + F.β / F.zT k n ^ 2) = F.dgT k (F.zT k n)⁻¹ := by
  have hz := F.zT_ne_zero k hμ n
  have hP := F.PT_zT_ne_zero k n
  have hD := F.dT_inv k hz
  have hχ : F.chiTUp k n n = (-F.β * F.HT k (F.zT k n) + F.lamT * F.ρ / (2 * F.zT k n) +
      F.NT k (F.zT k n) (F.zT k n) / F.PT k (F.zT k n)) / F.PT k (F.zT k n) := by
    rw [eq_div_iff hP, mul_comm, F.PT_mul_chiTUp k hμ hW, kron_self, two_mul]
    ring
  simp only [dgT, n0T, n2T, hD, hχ, NT, F.N₀T_eq]
  field_simp
  simp only [HT, N₂T, F.PT_eq]
  ring

/-- **Diagonal entries, with a constant**: from `|z³(χ_nn + β/z²)| ≤ C`. -/
theorem norm_chiTUp_diag_le_of (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) {C : ℝ}
    (hC : ∀ n, ‖F.dgT k (F.zT k n)⁻¹‖ ≤ C) (n : ℤ) :
    ‖F.chiTUp k n n + F.β / F.zT k n ^ 2‖ ≤ C / ‖F.zT k n‖ ^ 3 := by
  have hz := norm_pos_iff.mpr (F.zT_ne_zero k hμ n)
  have h := hC n
  rw [← F.dgT_inv k hμ hW n, norm_mul, norm_pow] at h
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- **Diagonal entries**: for `μ ≠ 0`, `W ≠ 0`, `|χ_T↑,nn + β/zₙ²| ≤ c|zₙ|⁻³`, i.e.
`χ_T↑,nn = -βzₙ⁻²[1 + O(|zₙ|⁻¹)]`. -/
theorem norm_chiTUp_diag_le (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) :
    ∃ c, ∀ n, ‖F.chiTUp k n n + F.β / F.zT k n ^ 2‖ ≤ c / ‖F.zT k n‖ ^ 3 := by
  have hq : ContinuousAt (fun t ↦ F.dgT k t) 0 := by
    simp only [dgT, n0T, n2T, dT]
    fun_prop (disch := simp)
  obtain ⟨C, hC⟩ := (F.lat F.μ).bdd_of_inv_z k _ hq (f := fun n ↦ F.dgT k (F.zT k n)⁻¹)
    fun _ ↦ rfl
  exact ⟨C, F.norm_chiTUp_diag_le_of k hμ hW hC⟩

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
