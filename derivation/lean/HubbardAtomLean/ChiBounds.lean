import HubbardAtomLean.Matsubara
import HubbardAtomLean.PHInput

/-!
# Bounds on the entries of `χ_r`

For the density and magnetic channels and `μ ≠ 0`, the entries of `χ_r` decay as those of the
bubble `diag(-β/zₙ²)`, as used in the proof of Proposition "finite boxes":

* `Params.norm_chiR_le`: `|χ_r,nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m`;
* `Params.norm_chiR_diag_le`: `|χ_r,nn + β/zₙ²| ≤ c|zₙ|⁻³`, i.e.
  `χ_r,nn = -βzₙ⁻²[1 + O(|zₙ|⁻¹)]`.

Both follow from Proposition "structure" (`Pph_mul_chiR`): every factor of the entries is a
function of `1/zₙ` continuous at `0` (`bdd_of_inv_z`), or `1/zₙ`, `1/(zₙ + zₘ)`, which are
bounded by `1/|μ|` and `1/(2|μ|)`.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `1/zₙ → 0` along the cofinite filter. -/
theorem tendsto_inv_z : Tendsto (fun n ↦ (A.z k n)⁻¹) cofinite (𝓝 0) := by
  have h := A.tendsto_norm_z_sub k 0
  simp only [sub_zero] at h
  exact tendsto_inv₀_cobounded.comp (tendsto_norm_atTop_iff_cobounded.mp h)

/-- A function of `1/zₙ` that is continuous at `0` is bounded on the lattice. -/
theorem bdd_of_inv_z {f : ℤ → ℂ} (q : ℂ → ℂ) (hq : ContinuousAt q 0)
    (hf : ∀ n, f n = q (A.z k n)⁻¹) : ∃ C, ∀ n, ‖f n‖ ≤ C := by
  obtain ⟨C, hC⟩ := (hq.tendsto.comp (A.tendsto_inv_z k)).norm.bddAbove_range_of_cofinite
  exact ⟨C, fun n ↦ (hf n).symm ▸ hC ⟨n, rfl⟩⟩

/-- `|μ| ≤ |zₙ|`, as `Re zₙ = μ`. -/
theorem abs_mu_le_norm_z (n : ℤ) : |A.μ| ≤ ‖A.z k n‖ := by
  have h := abs_re_le_norm (A.z k n)
  simpa [A.z_eq] using h

/-- `|1/zₙ| ≤ 1/|μ|`. -/
theorem norm_inv_z_le (hμ : A.μ ≠ 0) (n : ℤ) : ‖(A.z k n)⁻¹‖ ≤ |A.μ|⁻¹ := by
  rw [norm_inv]
  exact inv_anti₀ (abs_pos.mpr hμ) (A.abs_mu_le_norm_z k n)

/-- `|1/(zₙ + zₘ)| ≤ 1/(2|μ|)`, as `Re(zₙ + zₘ) = 2μ`. -/
theorem norm_inv_z_add_z_le (hμ : A.μ ≠ 0) (n m : ℤ) :
    ‖(A.z k n + A.z k m)⁻¹‖ ≤ (2 * |A.μ|)⁻¹ := by
  rw [norm_inv]
  refine inv_anti₀ (by positivity) ?_
  have h := abs_re_le_norm (A.z k n + A.z k m)
  have hre : (A.z k n + A.z k m).re = 2 * A.μ := by simp [A.z_eq]; ring
  rwa [hre, abs_mul, abs_two] at h

/-- `zₙ ≠ 0` for `μ ≠ 0`. -/
theorem z_ne_zero (hμ : A.μ ≠ 0) (n : ℤ) : A.z k n ≠ 0 :=
  norm_pos_iff.mp ((abs_pos.mpr hμ).trans_le (A.abs_mu_le_norm_z k n))

/-- `P(v) = v⁴ + (2q₀ - 4a²)v² + q₀²`. -/
theorem Pph_eq (v : ℂ) :
    A.Pph k v = v ^ 4 + (2 * A.q₀ k - 4 * A.a ^ 2) * v ^ 2 + A.q₀ k ^ 2 := by
  simp only [Pph, Qp, Qm, q₀]
  ring

/-- `N₀` is cubic with leading coefficient `-λρ`. -/
theorem N₀_cubic : ∃ c₀ c₁ c₂ : ℂ, ∀ v,
    A.N₀ k ε v = -(A.lam ε * A.ρ) * v ^ 3 + c₂ * v ^ 2 + c₁ * v + c₀ :=
  ⟨(2 * A.q₀ k - 4 * A.a ^ 2) * (-2 * ε * A.a * A.q₀ k - kron k 0 * A.ζ ε * A.a ^ 2) -
      (kron k 0 * A.ζ ε - 2 * ε * A.a) * A.q₀ k ^ 2,
    -(2 * A.q₀ k - 4 * A.a ^ 2) * (A.lam ε * A.ρ),
    -2 * ε * A.a * A.q₀ k - kron k 0 * A.ζ ε * A.a ^ 2,
    fun v ↦ by simp only [N₀, N₂, A.Pph_eq, lam]; ring⟩

/-- `zₙ⁴/P(zₙ)` is bounded. -/
theorem bdd_z_pow_div_Pph (hμ : A.μ ≠ 0) : ∃ C, ∀ n, ‖A.z k n ^ 4 / A.Pph k (A.z k n)‖ ≤ C := by
  refine A.bdd_of_inv_z k (fun t ↦ (1 + (2 * A.q₀ k - 4 * A.a ^ 2) * t ^ 2 + A.q₀ k ^ 2 * t ^ 4)⁻¹)
    (ContinuousAt.inv₀ (by fun_prop) (by simp)) fun n ↦ ?_
  have hz := A.z_ne_zero k hμ n
  rw [A.Pph_eq]
  field_simp

/-- **Off-diagonal entries**: for `μ ≠ 0`, `|χ_r,nm| ≤ c|zₙ|⁻²|zₘ|⁻²` for `n ≠ m`. -/
theorem norm_chiR_le (hμ : A.μ ≠ 0) :
    ∃ c, ∀ n m, n ≠ m → ‖A.chiR k ε n m‖ ≤ c / (‖A.z k n‖ ^ 2 * ‖A.z k m‖ ^ 2) := by
  obtain ⟨c₀, c₁, c₂, hN₀⟩ := A.N₀_cubic k ε
  obtain ⟨C₁, hC₁⟩ := A.bdd_z_pow_div_Pph k hμ
  -- `N₂(v)/v²`, `N₀(v)/v³` and `R₃(v)/v³` with `R₃(v) = λρP(v) + vN₀(v)` (cubic)
  obtain ⟨C₂, hC₂⟩ := A.bdd_of_inv_z k (f := fun n ↦ A.N₂ k ε (A.z k n) / A.z k n ^ 2)
    (fun t ↦ -2 * ε * A.a * (1 + 2 * A.a * A.ρ * t + A.q₀ k * t ^ 2) +
      kron k 0 * A.ζ ε * (1 - A.a ^ 2 * t ^ 2)) (by fun_prop) fun n ↦ by
    have hz := A.z_ne_zero k hμ n
    simp only [N₂]
    field_simp
  obtain ⟨C₃, hC₃⟩ := A.bdd_of_inv_z k (f := fun n ↦ A.N₀ k ε (A.z k n) / A.z k n ^ 3)
    (fun t ↦ -(A.lam ε * A.ρ) + c₂ * t + c₁ * t ^ 2 + c₀ * t ^ 3) (by fun_prop) fun n ↦ by
    have hz := A.z_ne_zero k hμ n
    simp only [hN₀]
    field_simp
  obtain ⟨C₄, hC₄⟩ := A.bdd_of_inv_z k
    (f := fun n ↦ (A.lam ε * A.ρ * A.Pph k (A.z k n) + A.z k n * A.N₀ k ε (A.z k n)) /
      A.z k n ^ 3)
    (fun t ↦ c₂ + (A.lam ε * A.ρ * (2 * A.q₀ k - 4 * A.a ^ 2) + c₁) * t + c₀ * t ^ 2 +
      A.lam ε * A.ρ * A.q₀ k ^ 2 * t ^ 3) (by fun_prop) fun n ↦ by
    have hz := A.z_ne_zero k hμ n
    simp only [hN₀, A.Pph_eq]
    field_simp
    ring
  have hμ' : 0 < |A.μ| := abs_pos.mpr hμ
  have hC₁0 := (norm_nonneg _).trans (hC₁ 0)
  have hC₃0 := (norm_nonneg _).trans (hC₃ 0)
  have hC₄0 := (norm_nonneg _).trans (hC₄ 0)
  refine ⟨C₁ * C₁ * (C₂ + (C₃ + C₄ * |A.μ|⁻¹) * (|A.μ|⁻¹ + (2 * |A.μ|)⁻¹)),
    fun n m hnm ↦ ?_⟩
  have hz := A.z_ne_zero k hμ n
  have hv := A.z_ne_zero k hμ m
  have hzv := A.z_add_z_ne_zero k hμ n m
  have hPz := A.Pph_z_ne_zero k n
  have hPv := A.Pph_z_ne_zero k m
  have hχ : A.chiR k ε n m = (A.lam ε * A.ρ / (A.z k n + A.z k m) +
      A.Nph k ε (A.z k n) (A.z k m) / A.Pph k (A.z k m)) / A.Pph k (A.z k n) := by
    rw [eq_div_iff hPz, mul_comm, A.Pph_mul_chiR k ε hμ, kron_of_ne hnm]
    ring
  have key : A.z k n ^ 2 * A.z k m ^ 2 * A.chiR k ε n m =
      A.z k n ^ 4 / A.Pph k (A.z k n) * (A.z k m ^ 4 / A.Pph k (A.z k m)) *
        (A.N₂ k ε (A.z k m) / A.z k m ^ 2 + (A.N₀ k ε (A.z k m) / A.z k m ^ 3 +
          (A.lam ε * A.ρ * A.Pph k (A.z k m) + A.z k m * A.N₀ k ε (A.z k m)) /
            A.z k m ^ 3 * (A.z k n)⁻¹) * ((A.z k n)⁻¹ - (A.z k n + A.z k m)⁻¹)) := by
    rw [hχ, Nph]
    field_simp
    ring
  have hb : ‖A.z k n ^ 2 * A.z k m ^ 2 * A.chiR k ε n m‖ ≤
      C₁ * C₁ * (C₂ + (C₃ + C₄ * |A.μ|⁻¹) * (|A.μ|⁻¹ + (2 * |A.μ|)⁻¹)) := by
    rw [key, norm_mul, norm_mul]
    refine mul_le_mul (mul_le_mul (hC₁ n) (hC₁ m) (norm_nonneg _) hC₁0) ?_ (norm_nonneg _)
      (mul_nonneg hC₁0 hC₁0)
    refine (norm_add_le _ _).trans (add_le_add (hC₂ m) ?_)
    rw [norm_mul]
    refine mul_le_mul ?_ ((norm_sub_le _ _).trans (add_le_add (A.norm_inv_z_le k hμ n)
      (A.norm_inv_z_add_z_le k hμ n m))) (norm_nonneg _)
      (add_nonneg hC₃0 (mul_nonneg hC₄0 (inv_nonneg.mpr hμ'.le)))
    refine (norm_add_le _ _).trans (add_le_add (hC₃ m) ?_)
    rw [norm_mul]
    exact mul_le_mul (hC₄ m) (A.norm_inv_z_le k hμ n) (norm_nonneg _) hC₄0
  rw [norm_mul, norm_mul, norm_pow, norm_pow] at hb
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- **Diagonal entries**: for `μ ≠ 0`, `|χ_r,nn + β/zₙ²| ≤ c|zₙ|⁻³`, i.e.
`χ_r,nn = -βzₙ⁻²[1 + O(|zₙ|⁻¹)]`. -/
theorem norm_chiR_diag_le (hμ : A.μ ≠ 0) :
    ∃ c, ∀ n, ‖A.chiR k ε n n + A.β / A.z k n ^ 2‖ ≤ c / ‖A.z k n‖ ^ 3 := by
  obtain ⟨c₀, c₁, c₂, hN₀⟩ := A.N₀_cubic k ε
  -- with `D(t) = P(z)/z⁴`, `z³(χ_r,nn + β/z²) = D⁻¹[β(P - z²H)/z³ + λρ/(2z²) + D⁻¹N(z, z)/z⁵]`
  obtain ⟨C, hC⟩ := A.bdd_of_inv_z k
    (f := fun n ↦ A.z k n ^ 3 * (A.chiR k ε n n + A.β / A.z k n ^ 2))
    (fun t ↦ (1 + (2 * A.q₀ k - 4 * A.a ^ 2) * t ^ 2 + A.q₀ k ^ 2 * t ^ 4)⁻¹ *
      (A.β * (-2 * A.a * A.ρ + (A.q₀ k - 4 * A.a ^ 2 - 4 * ε * A.a ^ 2 * A.p₁) * t +
          A.q₀ k ^ 2 * t ^ 3) + A.lam ε * A.ρ / 2 * t ^ 2 +
        (1 + (2 * A.q₀ k - 4 * A.a ^ 2) * t ^ 2 + A.q₀ k ^ 2 * t ^ 4)⁻¹ *
          (t ^ 2 * (-(A.lam ε * A.ρ) + c₂ * t + c₁ * t ^ 2 + c₀ * t ^ 3) +
            t * (-2 * ε * A.a * (1 + 2 * A.a * A.ρ * t + A.q₀ k * t ^ 2) +
              kron k 0 * A.ζ ε * (1 - A.a ^ 2 * t ^ 2)))))
    (by fun_prop (disch := simp)) fun n ↦ by
    have hz := A.z_ne_zero k hμ n
    have hP := A.Pph_z_ne_zero k n
    have hD : 1 + (2 * A.q₀ k - 4 * A.a ^ 2) * (A.z k n)⁻¹ ^ 2 + A.q₀ k ^ 2 * (A.z k n)⁻¹ ^ 4 =
        A.Pph k (A.z k n) / A.z k n ^ 4 := by
      rw [A.Pph_eq]
      field_simp
    have hχ : A.chiR k ε n n = (-A.β * A.Hph k ε (A.z k n) + A.lam ε * A.ρ / (2 * A.z k n) +
        A.Nph k ε (A.z k n) (A.z k n) / A.Pph k (A.z k n)) / A.Pph k (A.z k n) := by
      rw [eq_div_iff hP, mul_comm, A.Pph_mul_chiR k ε hμ, kron_self, two_mul]
      ring
    simp only [hD, hχ, Nph, hN₀]
    field_simp
    simp only [Hph, N₂, A.Pph_eq]
    ring
  refine ⟨C, fun n ↦ ?_⟩
  have hz := norm_pos_iff.mpr (A.z_ne_zero k hμ n)
  have h := hC n
  rw [norm_mul, norm_pow] at h
  rw [le_div_iff₀ (by positivity)]
  linarith

end Params

end HubbardAtom
