import HubbardAtomLean.ChiOperator
import HubbardAtomLean.FieldLVertex

/-!
# Bounds on the entries of `χ_L`

Item 6(a) for the longitudinal sector, off the axes: on the lattices `z = μ_σ + i(νₙ + ω_k/2)` of
the two spins, `μ_σ = μ + σh ≠ 0`, the entries of `χ_L` decay as those of the bubble
`diag(-β/z²)`. With `z = z_σn`, `v = z_τm`:

* `FParams.norm_chiL_self_le`, `FParams.norm_chiL_opp_le`: `|χ_στ(n, m)| ≤ c|z|⁻²|v|⁻²` for
  `(σ, n) ≠ (τ, m)`, equal spins by the input `PL_mul_chiL_self` and opposite spins (for
  `μ ≠ 0`, `h ≠ 0`) by `PL_mul_chiL_opp`, whose poles `1/(z + v)` and `1/(z - v)` are bounded by
  `1/(2|μ|)` and `1/(2|h|)` (`norm_inv_zL_add_opp_le`, `norm_inv_zL_sub_opp_le`);
* `FParams.norm_chiL_diag_le`: `|χ_σσ(n, n) + β/z²| ≤ c|z|⁻³`;
* `FParams.norm_zL_mul_tL_le`: the bubble `b_σ = -β((z + ar_σ)² - w²/4)/P(z)` has
  `|z(-z²b_σ/β - 1)| ≤ C`.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The two lattices -/

/-- `Re z_σn = μ_σ`. -/
theorem re_zL (σ : Bool) (n : ℤ) : ((F.BL σ).z k n).re = F.cL σ := by
  simp [Params.z_eq]

/-- `|μ_σ - μ_{-σ}| = 2|h|`. -/
theorem abs_cL_sub (σ : Bool) : |F.cL σ - F.cL (!σ)| = 2 * |F.h| := by
  cases σ <;> simp only [cL, Bool.not_false, Bool.not_true, Bool.cond_true, Bool.cond_false]
  · rw [show F.μ - F.h - (F.μ + F.h) = -(2 * F.h) by ring, abs_neg, abs_mul, abs_two]
  · rw [show F.μ + F.h - (F.μ - F.h) = 2 * F.h by ring, abs_mul, abs_two]

/-- `|1/(z_σn - z_{-σ}m)| ≤ 1/(2|h|)`, as `Re(z_σn - z_{-σ}m) = ±2h`. -/
theorem norm_inv_zL_sub_opp_le (hh : F.h ≠ 0) (σ : Bool) (n m : ℤ) :
    ‖((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹‖ ≤ (2 * |F.h|)⁻¹ := by
  rw [norm_inv]
  refine inv_anti₀ (mul_pos two_pos (abs_pos.mpr hh)) ?_
  have h := abs_re_le_norm ((F.BL σ).z k n - (F.BL (!σ)).z k m)
  rwa [sub_re, F.re_zL, F.re_zL, F.abs_cL_sub] at h

/-- `|1/(z_σn + z_{-σ}m)| ≤ 1/(2|μ|)`, as `Re(z_σn + z_{-σ}m) = 2μ`. -/
theorem norm_inv_zL_add_opp_le (hμ : F.μ ≠ 0) (σ : Bool) (n m : ℤ) :
    ‖((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹‖ ≤ (2 * |F.μ|)⁻¹ := by
  rw [norm_inv]
  refine inv_anti₀ (mul_pos two_pos (abs_pos.mpr hμ)) ?_
  have h := abs_re_le_norm ((F.BL σ).z k n + (F.BL (!σ)).z k m)
  rwa [add_re, F.re_zL, F.re_zL, F.cL_add, abs_mul, abs_two] at h

/-! ### The data -/

/-- `P(v) = v⁴ + (2q₀ - 4a²)v² + q₀²`. -/
theorem PL_eq (v : ℂ) :
    F.PL k v = v ^ 4 + (2 * F.q₀L k - 4 * F.a ^ 2) * v ^ 2 + F.q₀L k ^ 2 := by
  simp only [PL, q₀L]
  ring

/-- `N₀_r` is cubic with leading coefficient `-λr`. -/
theorem N₀L_cubic (r : ℝ) : ∃ c₀ c₁ c₂ : ℂ, ∀ v,
    F.N₀L k r v = -(F.lamL * r) * v ^ 3 + c₂ * v ^ 2 + c₁ * v + c₀ :=
  ⟨(2 * F.q₀L k - 4 * F.a ^ 2) * (-2 * F.a * F.q₀L k - kron k 0 * F.ζL * F.a ^ 2) -
      (-2 * F.a + kron k 0 * F.ζL) * F.q₀L k ^ 2,
    -((2 * F.q₀L k - 4 * F.a ^ 2) * F.lamL * r),
    -2 * F.a * F.q₀L k - kron k 0 * F.ζL * F.a ^ 2,
    fun v ↦ by simp only [N₀L, N₂L, HL, F.PL_eq, lamL]; ring⟩

/-- `z⁴/P(z)` is bounded on the lattice of `σ`, for `μ_σ ≠ 0`. -/
theorem bdd_zL_pow_div_PL {σ : Bool} (hc : F.cL σ ≠ 0) :
    ∃ C, ∀ n, ‖(F.BL σ).z k n ^ 4 / F.PL k ((F.BL σ).z k n)‖ ≤ C := by
  refine (F.BL σ).bdd_of_inv_z k
    (fun t ↦ (1 + (2 * F.q₀L k - 4 * F.a ^ 2) * t ^ 2 + F.q₀L k ^ 2 * t ^ 4)⁻¹)
    (ContinuousAt.inv₀ (by fun_prop) (by simp)) fun n ↦ ?_
  have hz := (F.BL σ).z_ne_zero k hc n
  rw [F.PL_eq]
  field_simp

/-! ### Equal spins -/

/-- **Equal spins, off the diagonal**: for `μ_σ ≠ 0`, `|χ_σσ(n, m)| ≤ c|z|⁻²|v|⁻²` for `n ≠ m`. -/
theorem norm_chiL_self_le {σ : Bool} (hc : F.cL σ ≠ 0) :
    ∃ c, ∀ n m, n ≠ m →
      ‖F.chiL σ σ k n m‖ ≤ c / (‖(F.BL σ).z k n‖ ^ 2 * ‖(F.BL σ).z k m‖ ^ 2) := by
  -- `z²(z² - a²)/P(z)`
  obtain ⟨C, hC⟩ := (F.BL σ).bdd_of_inv_z k
    (f := fun n ↦ (F.BL σ).z k n ^ 2 * ((F.BL σ).z k n ^ 2 - F.a ^ 2) /
      F.PL k ((F.BL σ).z k n))
    (fun t ↦ (1 + (2 * F.q₀L k - 4 * F.a ^ 2) * t ^ 2 + F.q₀L k ^ 2 * t ^ 4)⁻¹ *
      (1 - F.a ^ 2 * t ^ 2))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) fun n ↦ by
    have hz := (F.BL σ).z_ne_zero k hc n
    rw [F.PL_eq]
    field_simp
  have hC0 := (norm_nonneg _).trans (hC 0)
  refine ⟨‖kron k 0 * F.ζLs σ‖ * C * C, fun n m hnm ↦ ?_⟩
  have hz := (F.BL σ).z_ne_zero k hc n
  have hv := (F.BL σ).z_ne_zero k hc m
  have hPz := F.PL_BL_ne k σ n
  have hPv := F.PL_BL_ne k σ m
  have key : (F.BL σ).z k n ^ 2 * (F.BL σ).z k m ^ 2 * F.chiL σ σ k n m =
      kron k 0 * F.ζLs σ *
        ((F.BL σ).z k n ^ 2 * ((F.BL σ).z k n ^ 2 - F.a ^ 2) / F.PL k ((F.BL σ).z k n)) *
        ((F.BL σ).z k m ^ 2 * ((F.BL σ).z k m ^ 2 - F.a ^ 2) / F.PL k ((F.BL σ).z k m)) := by
    have h := F.PL_mul_chiL_self k σ n m
    rw [kron_of_ne hnm, mul_zero, zero_add] at h
    rw [show F.chiL σ σ k n m = F.PL k ((F.BL σ).z k n) * F.chiL σ σ k n m /
      F.PL k ((F.BL σ).z k n) by field_simp, h]
    field_simp
  have hb : ‖(F.BL σ).z k n ^ 2 * (F.BL σ).z k m ^ 2 * F.chiL σ σ k n m‖ ≤
      ‖kron k 0 * F.ζLs σ‖ * C * C := by
    rw [key, norm_mul, norm_mul]
    exact mul_le_mul (mul_le_mul_of_nonneg_left (hC n) (norm_nonneg _)) (hC m) (norm_nonneg _)
      (mul_nonneg (norm_nonneg _) hC0)
  rw [norm_mul, norm_mul, norm_pow, norm_pow] at hb
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- **Equal spins, the diagonal**: for `μ_σ ≠ 0`, `|χ_σσ(n, n) + β/z²| ≤ c|z|⁻³`. -/
theorem norm_chiL_diag_le {σ : Bool} (hc : F.cL σ ≠ 0) :
    ∃ c, ∀ n, ‖F.chiL σ σ k n n + F.β / (F.BL σ).z k n ^ 2‖ ≤ c / ‖(F.BL σ).z k n‖ ^ 3 := by
  -- with `D(t) = P(z)/z⁴`, `z³(χ_nn + β/z²) = D⁻¹β(P - z²H)/z³ + D⁻²ζ(1 - a²t²)²t`
  obtain ⟨C, hC⟩ := (F.BL σ).bdd_of_inv_z k
    (f := fun n ↦ (F.BL σ).z k n ^ 3 * (F.chiL σ σ k n n + F.β / (F.BL σ).z k n ^ 2))
    (fun t ↦ (1 + (2 * F.q₀L k - 4 * F.a ^ 2) * t ^ 2 + F.q₀L k ^ 2 * t ^ 4)⁻¹ *
        (F.β * (-2 * F.a * F.rL σ + (2 * F.q₀L k - 4 * F.a ^ 2 - F.q₀L k) * t +
          F.q₀L k ^ 2 * t ^ 3)) +
      ((1 + (2 * F.q₀L k - 4 * F.a ^ 2) * t ^ 2 + F.q₀L k ^ 2 * t ^ 4)⁻¹) ^ 2 *
        (kron k 0 * F.ζLs σ * ((1 - F.a ^ 2 * t ^ 2) ^ 2 * t)))
    (by fun_prop (disch := simp)) fun n ↦ by
    have hz := (F.BL σ).z_ne_zero k hc n
    have hP := F.PL_BL_ne k σ n
    have hD : 1 + (2 * F.q₀L k - 4 * F.a ^ 2) * ((F.BL σ).z k n)⁻¹ ^ 2 +
        F.q₀L k ^ 2 * ((F.BL σ).z k n)⁻¹ ^ 4 = F.PL k ((F.BL σ).z k n) / (F.BL σ).z k n ^ 4 := by
      rw [F.PL_eq]
      field_simp
    have hχ : F.chiL σ σ k n n = (-F.β * F.HL k (F.rL σ) ((F.BL σ).z k n) +
        kron k 0 * F.ζLs σ * (((F.BL σ).z k n ^ 2 - F.a ^ 2) ^ 2 / F.PL k ((F.BL σ).z k n))) /
          F.PL k ((F.BL σ).z k n) := by
      rw [eq_div_iff hP, mul_comm, F.PL_mul_chiL_self k σ n n, kron_self]
      ring
    rw [hD, hχ]
    field_simp
    simp only [HL, F.PL_eq]
    ring
  refine ⟨C, fun n ↦ ?_⟩
  have hz := norm_pos_iff.mpr ((F.BL σ).z_ne_zero k hc n)
  have h := hC n
  rw [norm_mul, norm_pow] at h
  rw [le_div_iff₀ (by positivity)]
  linarith

/-! ### Opposite spins -/

/-- The norm of `Z(G₁ + t²(aG₂ + G₃) - b(t - p) - c(q - t))`, term by term. -/
theorem norm_opp_shape_le (Z G₁ G₂ G₃ t p q a b c : ℂ) :
    ‖Z * (G₁ + t ^ 2 * (a * G₂ + G₃) - b * (t - p) - c * (q - t))‖ ≤
      ‖Z‖ * (‖G₁‖ + ‖t‖ ^ 2 * (‖a‖ * ‖G₂‖ + ‖G₃‖) + ‖b‖ * (‖t‖ + ‖p‖) +
        ‖c‖ * (‖q‖ + ‖t‖)) := by
  rw [norm_mul]
  gcongr
  refine (norm_sub_le _ _).trans (add_le_add ((norm_sub_le _ _).trans (add_le_add
    ((norm_add_le _ _).trans (add_le_add le_rfl ?_)) ?_)) ?_)
  · rw [norm_mul, norm_pow]
    gcongr
    exact (norm_add_le _ _).trans (add_le_add (norm_mul _ _).le le_rfl)
  · rw [norm_mul]
    gcongr
    exact norm_sub_le _ _
  · rw [norm_mul]
    gcongr
    exact norm_sub_le _ _

/-- **Opposite spins**: for `μ ≠ 0`, `h ≠ 0` and `μ_{±σ} ≠ 0`, `|χ_{σ,-σ}(n, m)| ≤ c|z|⁻²|v|⁻²`.
With `t = 1/z`, `N₀(v) = -λ(ρ + σm)v³ + Q(v)`, the input gives
`z²v²χ = (z⁴/P(z))[v²N₂/P + t²(λ(ρ + σm)v(P - v⁴)/P + v²Q/P) - λρ(t - 1/(z + v))
- λσm(1/(z - v) - t)]`, with every factor bounded. -/
theorem norm_chiL_opp_le (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) {σ : Bool} (hc : F.cL σ ≠ 0)
    (hc' : F.cL (!σ) ≠ 0) :
    ∃ c, ∀ n m,
      ‖F.chiL σ (!σ) k n m‖ ≤ c / (‖(F.BL σ).z k n‖ ^ 2 * ‖(F.BL (!σ)).z k m‖ ^ 2) := by
  obtain ⟨c₀, c₁, c₂, hN₀⟩ := F.N₀L_cubic k (F.rL (!σ))
  obtain ⟨CZ, hCZ⟩ := F.bdd_zL_pow_div_PL k hc
  -- `v²N₂(v)/P(v)`, `v(P(v) - v⁴)/P(v)` and `v²Q(v)/P(v)` on the lattice of `-σ`
  obtain ⟨C₁, hC₁⟩ := (F.BL (!σ)).bdd_of_inv_z k
    (f := fun m ↦ (F.BL (!σ)).z k m ^ 2 * F.N₂L k (F.rL (!σ)) ((F.BL (!σ)).z k m) /
      F.PL k ((F.BL (!σ)).z k m))
    (fun s ↦ (1 + (2 * F.q₀L k - 4 * F.a ^ 2) * s ^ 2 + F.q₀L k ^ 2 * s ^ 4)⁻¹ *
      (-2 * F.a * (1 + 2 * F.a * F.rL (!σ) * s + F.q₀L k * s ^ 2) +
        kron k 0 * F.ζL * (1 - F.a ^ 2 * s ^ 2)))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) fun m ↦ by
    have hv := (F.BL (!σ)).z_ne_zero k hc' m
    rw [F.PL_eq]
    field_simp
    simp only [N₂L, HL]
    ring
  obtain ⟨C₂, hC₂⟩ := (F.BL (!σ)).bdd_of_inv_z k
    (f := fun m ↦ (F.BL (!σ)).z k m * (F.PL k ((F.BL (!σ)).z k m) - (F.BL (!σ)).z k m ^ 4) /
      F.PL k ((F.BL (!σ)).z k m))
    (fun s ↦ (1 + (2 * F.q₀L k - 4 * F.a ^ 2) * s ^ 2 + F.q₀L k ^ 2 * s ^ 4)⁻¹ *
      ((2 * F.q₀L k - 4 * F.a ^ 2) * s + F.q₀L k ^ 2 * s ^ 3))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) fun m ↦ by
    have hv := (F.BL (!σ)).z_ne_zero k hc' m
    rw [F.PL_eq]
    field_simp
    ring
  obtain ⟨C₃, hC₃⟩ := (F.BL (!σ)).bdd_of_inv_z k
    (f := fun m ↦ (F.BL (!σ)).z k m ^ 2 * (c₂ * (F.BL (!σ)).z k m ^ 2 +
      c₁ * (F.BL (!σ)).z k m + c₀) / F.PL k ((F.BL (!σ)).z k m))
    (fun s ↦ (1 + (2 * F.q₀L k - 4 * F.a ^ 2) * s ^ 2 + F.q₀L k ^ 2 * s ^ 4)⁻¹ *
      (c₂ + c₁ * s + c₀ * s ^ 2))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) fun m ↦ by
    have hv := (F.BL (!σ)).z_ne_zero k hc' m
    rw [F.PL_eq]
    field_simp
  have hCZ0 := (norm_nonneg _).trans (hCZ 0)
  set T := |F.cL σ|⁻¹
  refine ⟨CZ * (C₁ + T ^ 2 * (‖F.lamL * (F.ρ + F.mL σ)‖ * C₂ + C₃) +
    ‖F.lamL * F.ρ‖ * (T + (2 * |F.μ|)⁻¹) + ‖F.lamL * F.mL σ‖ * ((2 * |F.h|)⁻¹ + T)),
    fun n m ↦ ?_⟩
  set z := (F.BL σ).z k n with hzdef
  set v := (F.BL (!σ)).z k m with hvdef
  have hz : z ≠ 0 := (F.BL σ).z_ne_zero k hc n
  have hv : v ≠ 0 := (F.BL (!σ)).z_ne_zero k hc' m
  have hzp : z + v ≠ 0 := F.zL_add_opp_ne k hμ σ n m
  have hzm : z - v ≠ 0 := F.zL_sub_opp_ne k hh σ n m
  have hPz : F.PL k z ≠ 0 := F.PL_BL_ne k σ n
  have hPv : F.PL k v ≠ 0 := F.PL_BL_ne k (!σ) m
  have hr : (F.rL (!σ) : ℂ) = F.ρ + F.mL σ := by rw [F.rL_not σ]; push_cast; ring
  have hχ : F.chiL σ (!σ) k n m = (F.lamL * (F.ρ / (z + v) - F.mL σ / (z - v)) +
      (F.N₀L k (F.rL (!σ)) v + z ^ 2 * F.N₂L k (F.rL (!σ)) v) / F.PL k v) / F.PL k z := by
    rw [eq_div_iff hPz, mul_comm, F.PL_mul_chiL_opp k hμ hh σ n m]
  have key : z ^ 2 * v ^ 2 * F.chiL σ (!σ) k n m =
      z ^ 4 / F.PL k z * (v ^ 2 * F.N₂L k (F.rL (!σ)) v / F.PL k v +
        z⁻¹ ^ 2 * (F.lamL * (F.ρ + F.mL σ) * (v * (F.PL k v - v ^ 4) / F.PL k v) +
          v ^ 2 * (c₂ * v ^ 2 + c₁ * v + c₀) / F.PL k v) -
        F.lamL * F.ρ * (z⁻¹ - (z + v)⁻¹) - F.lamL * F.mL σ * ((z - v)⁻¹ - z⁻¹)) := by
    rw [hχ, hN₀, hr]
    field_simp
    ring
  have hb : ‖z ^ 2 * v ^ 2 * F.chiL σ (!σ) k n m‖ ≤
      CZ * (C₁ + T ^ 2 * (‖F.lamL * (F.ρ + F.mL σ)‖ * C₂ + C₃) +
        ‖F.lamL * F.ρ‖ * (T + (2 * |F.μ|)⁻¹) + ‖F.lamL * F.mL σ‖ * ((2 * |F.h|)⁻¹ + T)) := by
    rw [key]
    refine (norm_opp_shape_le _ _ _ _ _ _ _ _ _ _).trans ?_
    have hZ : ‖z ^ 4 / F.PL k z‖ ≤ CZ := hCZ n
    have hG₁ : ‖v ^ 2 * F.N₂L k (F.rL (!σ)) v / F.PL k v‖ ≤ C₁ := hC₁ m
    have hG₂ : ‖v * (F.PL k v - v ^ 4) / F.PL k v‖ ≤ C₂ := hC₂ m
    have hG₃ : ‖v ^ 2 * (c₂ * v ^ 2 + c₁ * v + c₀) / F.PL k v‖ ≤ C₃ := hC₃ m
    have ht : ‖z⁻¹‖ ≤ T := (F.BL σ).norm_inv_z_le k hc n
    have hp : ‖(z + v)⁻¹‖ ≤ (2 * |F.μ|)⁻¹ := F.norm_inv_zL_add_opp_le k hμ σ n m
    have hq : ‖(z - v)⁻¹‖ ≤ (2 * |F.h|)⁻¹ := F.norm_inv_zL_sub_opp_le k hh σ n m
    have hC₁0 := (norm_nonneg _).trans hG₁
    have hC₂0 := (norm_nonneg _).trans hG₂
    have hC₃0 := (norm_nonneg _).trans hG₃
    have hT0 : 0 ≤ T := inv_nonneg.mpr (abs_nonneg _)
    gcongr
  rw [norm_mul, norm_mul, norm_pow, norm_pow] at hb
  rw [le_div_iff₀ (by positivity)]
  linarith

/-! ### The bubble -/

/-- **The bubble** `b_σ = -β((z + ar_σ)² - w²/4)/P(z)` decays as `-β/z²`: for `μ_σ ≠ 0`,
`tₙ = -z²b_σ/β = z²((z + ar_σ)² - w²/4)/P(z)` has `|z(tₙ - 1)| ≤ C`. -/
theorem norm_zL_mul_tL_le {σ : Bool} (hc : F.cL σ ≠ 0) :
    ∃ C, ∀ n, ‖(F.BL σ).z k n * (-((F.BL σ).z k n ^ 2 / F.β) * F.bL σ k n - 1)‖ ≤ C := by
  refine (F.BL σ).bdd_of_inv_z k
    (fun t ↦ (1 + (2 * F.q₀L k - 4 * F.a ^ 2) * t ^ 2 + F.q₀L k ^ 2 * t ^ 4)⁻¹ *
      (2 * F.a * F.rL σ + (F.a ^ 2 * F.rL σ ^ 2 - F.wL k ^ 2 / 4 -
        (2 * F.q₀L k - 4 * F.a ^ 2)) * t - F.q₀L k ^ 2 * t ^ 3))
    (ContinuousAt.mul (ContinuousAt.inv₀ (by fun_prop) (by simp)) (by fun_prop)) fun n ↦ ?_
  have hz := (F.BL σ).z_ne_zero k hc n
  have hP := F.PL_BL_ne k σ n
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  have hD : 1 + (2 * F.q₀L k - 4 * F.a ^ 2) * ((F.BL σ).z k n)⁻¹ ^ 2 +
      F.q₀L k ^ 2 * ((F.BL σ).z k n)⁻¹ ^ 4 = F.PL k ((F.BL σ).z k n) / (F.BL σ).z k n ^ 4 := by
    rw [F.PL_eq]
    field_simp
  rw [hD, F.bL_eq]
  field_simp
  simp only [F.PL_eq]
  ring

/-- `b_σ(νₙ) ≠ 0`. -/
theorem bL_ne_zero (σ : Bool) (n : ℤ) : F.bL σ k n ≠ 0 := by
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  rw [F.bL_eq]
  exact div_ne_zero (mul_ne_zero (neg_ne_zero.mpr hβ) (F.bL_num_ne k σ n)) (F.PL_BL_ne k σ n)

end FParams

end HubbardAtom
