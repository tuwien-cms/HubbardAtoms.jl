import HubbardAtomLean.FieldLOperator
import HubbardAtomLean.FieldLAxisInput
import HubbardAtomLean.DMContinuity

/-!
# Bounds on the entries of `χ_L` through the axes

Item 6(a) for the longitudinal sector for every `μ`, `h`, on the lattices `z'` of
`OpLatticeAll.lean` of the two spins. `FieldLBounds.lean` needs `μ ≠ 0`, `h ≠ 0` and `μ_σ ≠ 0`.
Here, as in `DMContinuity.lean`, the bounds hold with explicit constants that stay bounded through
the axes:

* off the mirror pairs and off `n = m`, respectively, `|1/(z_σₙ + z_{-σ}ₘ)| ≤ β/(2π)` and
  `|1/(z_σₙ - z_{-σ}ₘ)| ≤ β/(2π)`: the imaginary parts are nonzero multiples of `2π/β`;
* on a mirror pair the pair kernel is `κ_P = ρ/(2μ)`, continued by `βp₀` (`kerPL_mirror`), and at
  `n = m` the spin-flip kernel is `κ_S = m/(2h)`, continued by `βp↓` (`kerSLs_self`); both are
  `e^{∓βU/4}β sinhc(βx)/Z` with `sinhc = dslope sinh 0` (continuity in the parameters is not
  proved here);
* with `v = z_{-σ}ₘ`, `v²P(z)χ_{σ,-σ} = G₃ + λr'G₂ + z²G₁ + λ(v²K_P - ρv) - λ(v²K_S + σmv)`
  (`v_sq_mul_PL_mul_chiL_opp`), whose factors `Gᵢ` are functions of `1/v`; it also gives the row
  of the mirror fixed point, where `z = μ_σ`. The column, where `v = μ_{-σ}`, and the entry at the
  fixed point come from the input `PL_mul_chiL_opp_all` directly.

* `FParams.opLatLAll`: the lattices `z'` of the two spins, enumerated by `encL`;
* `FParams.norm_chiL_opp_le_all_of`, `norm_chiL_self_le_all_of`, `norm_chiL_diag_le_all_of`: the
  bounds on `z'`, with constants;
* `FParams.chiDecay_chiLe_all`, `bubbleDecay_bLe_all`: `ChiDecay` and `BubbleDecay` on `opLatLAll`
  at every `μ`, `h`.
-/

open Complex Filter Topology

namespace HubbardAtom

/-- The mirror fixed point is `(-k - 1)/2`. -/
theorem eq_of_mirror_eq_self {k j : ℤ} (hj : mirror k j = j) : j = (-k - 1) / 2 := by
  unfold mirror at hj
  omega

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### The two lattices through the axes -/

/-- `z_σₙ + z_{-σ}ₘ = zₙ + zₘ` on the lattice of `μ`. -/
theorem zL_add_opp_eq_lat (σ : Bool) (n m : ℤ) :
    (F.BL σ).z k n + (F.BL (!σ)).z k m = (F.lat F.μ).z k n + (F.lat F.μ).z k m := by
  rw [F.zL_add_opp]
  simp only [Params.z, Params.x, Params.w, lat_μ, lat_ν, lat_ω]
  push_cast
  ring

/-- On a mirror pair, `z_σₙ + z_{-σ}ₙ̄ = 2μ`. -/
theorem zL_add_opp_mirror (σ : Bool) (n : ℤ) :
    (F.BL σ).z k n + (F.BL (!σ)).z k (mirror k n) = 2 * F.μ := by
  rw [F.zL_add_opp_eq_lat, (F.lat F.μ).z_add_z_mirror k n]
  rfl

/-- `z_σₙ + z_{-σ}ₘ ≠ 0` off the mirror pairs. -/
theorem zL_add_opp_ne_zero_of_ne (σ : Bool) {n m : ℤ} (hnm : m ≠ mirror k n) :
    (F.BL σ).z k n + (F.BL (!σ)).z k m ≠ 0 := by
  rw [F.zL_add_opp_eq_lat]
  exact (F.lat F.μ).z_add_z_ne_zero_of_ne k hnm

/-- Off the mirror pairs, `|1/(z_σₙ + z_{-σ}ₘ)| ≤ β/(2π)`. -/
theorem norm_inv_zL_add_opp_le_of_ne (σ : Bool) {n m : ℤ} (hnm : m ≠ mirror k n) :
    ‖((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹‖ ≤ F.β / (2 * Real.pi) := by
  rw [F.zL_add_opp_eq_lat]
  exact (F.lat F.μ).norm_inv_z_add_z_le_of_ne k hnm

/-- At `n = m`, `z_σₙ - z_{-σ}ₙ = μ_σ - μ_{-σ}`. -/
theorem zL_sub_opp_self (σ : Bool) (n : ℤ) :
    (F.BL σ).z k n - (F.BL (!σ)).z k n = ((F.cL σ - F.cL (!σ) : ℝ) : ℂ) := by
  simp only [Params.z, Params.x, Params.w, lat_μ, lat_ν, lat_ω]
  push_cast
  ring

/-- Off `n = m`, `|1/(z_σₙ - z_{-σ}ₘ)| ≤ β/(2π)`, as `Im(z_σₙ - z_{-σ}ₘ) = 2(n - m)π/β`. -/
theorem norm_inv_zL_sub_opp_le_of_ne (σ : Bool) {n m : ℤ} (hnm : n ≠ m) :
    ‖((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹‖ ≤ F.β / (2 * Real.pi) := by
  have hβ := F.β_pos
  have hπ := Real.pi_pos
  have h1 : (1 : ℝ) ≤ |(n - m : ℝ)| := by
    have : (1 : ℤ) ≤ |n - m| := by
      have : n - m ≠ 0 := sub_ne_zero.mpr hnm
      rcases abs_cases (n - m) with ⟨h, _⟩ | ⟨h, _⟩ <;> omega
    exact_mod_cast this
  have him : ((F.BL σ).z k n - (F.BL (!σ)).z k m).im = 2 * ((n - m) * Real.pi / F.β) := by
    rw [sub_im, Params.z_im, Params.z_im]
    simp only [lat_β]
    ring
  have hz : 2 * Real.pi / F.β ≤ ‖(F.BL σ).z k n - (F.BL (!σ)).z k m‖ := by
    refine le_trans ?_ (abs_im_le_norm _)
    rw [him, abs_mul, abs_two, abs_div, abs_mul, abs_of_pos hπ, abs_of_pos hβ, mul_div_assoc']
    exact div_le_div_of_nonneg_right (by nlinarith) hβ.le
  rw [norm_inv]
  calc ‖(F.BL σ).z k n - (F.BL (!σ)).z k m‖⁻¹ ≤ (2 * Real.pi / F.β)⁻¹ :=
        inv_anti₀ (by positivity) hz
    _ = F.β / (2 * Real.pi) := inv_div _ _

/-! ### The kernels on their special pairs -/

/-- The pair kernel on a mirror pair: `κ_P = ρ/(2μ)`, continued by `βp₀` at `μ = 0`, as
`e^{-βU/4}β sinhc(βμ)/Z` with `sinhc = dslope sinh 0`. -/
noncomputable def κPL : ℝ :=
  Real.exp (-F.β * (F.U / 4)) * F.β * dslope Real.sinh 0 (F.β * F.μ) / F.Z

/-- The spin-flip kernel at `n = m`: `κ_S = m/(2h)`, continued by `βp↓ = βp↑` at `h = 0`, as
`e^{βU/4}β sinhc(βh)/Z`. -/
noncomputable def κSL : ℝ :=
  Real.exp (F.β * (F.U / 4)) * F.β * dslope Real.sinh 0 (F.β * F.h) / F.Z

/-- At `μ = 0`, `κ_P = βp₀`. -/
theorem κPL_of_mu (hμ : F.μ = 0) : F.κPL = F.β * F.p₀ := by
  rw [κPL, hμ, mul_zero, dslope_same, Real.deriv_sinh, Real.cosh_zero, p₀, boltz₀, hμ, add_zero]
  field_simp

/-- `2μκ_P = ρ`. -/
theorem two_mul_mu_mul_κPL : 2 * F.μ * F.κPL = F.ρ := by
  by_cases hμ : F.μ = 0
  · rw [hμ, mul_zero, zero_mul, ρ, F.p₂_eq_p₀_mul, hμ, mul_zero, Real.exp_zero, mul_one,
      sub_self]
  · have hβ := F.β_pos.ne'
    have hbm : F.β * F.μ ≠ 0 := mul_ne_zero hβ hμ
    rw [κPL, dslope_of_ne _ hbm, slope_def_field, sub_zero, Real.sinh_zero, sub_zero]
    have e2 : Real.exp (-F.β * (F.U / 4 - F.μ)) =
        Real.exp (-F.β * (F.U / 4)) * Real.exp (F.β * F.μ) := by
      rw [← Real.exp_add]
      ring_nf
    have e0 : Real.exp (-F.β * (F.U / 4 + F.μ)) =
        Real.exp (-F.β * (F.U / 4)) * Real.exp (-(F.β * F.μ)) := by
      rw [← Real.exp_add]
      ring_nf
    rw [ρ, p₂, p₀, boltz₂, boltz₀, e2, e0, Real.sinh_eq]
    field_simp

/-- At `h = 0`, `κ_S = βp↓`. -/
theorem κSL_of_h (hh : F.h = 0) : F.κSL = F.β * F.pDn := by
  rw [κSL, hh, mul_zero, dslope_same, Real.deriv_sinh, Real.cosh_zero, pDn, boltzDn, hh,
    add_zero, show -F.β * (-F.U / 4) = F.β * (F.U / 4) by ring]
  field_simp

/-- `2hκ_S = m`. -/
theorem two_mul_h_mul_κSL : 2 * F.h * F.κSL = F.mf := by
  by_cases hh : F.h = 0
  · rw [hh, mul_zero, zero_mul, mf, F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, hh, add_zero, sub_zero,
      sub_self]
  · have hβ := F.β_pos.ne'
    have hbh : F.β * F.h ≠ 0 := mul_ne_zero hβ hh
    rw [κSL, dslope_of_ne _ hbh, slope_def_field, sub_zero, Real.sinh_zero, sub_zero]
    have eu : Real.exp (-F.β * (-F.U / 4 - F.h)) =
        Real.exp (F.β * (F.U / 4)) * Real.exp (F.β * F.h) := by
      rw [← Real.exp_add]
      ring_nf
    have ed : Real.exp (-F.β * (-F.U / 4 + F.h)) =
        Real.exp (F.β * (F.U / 4)) * Real.exp (-(F.β * F.h)) := by
      rw [← Real.exp_add]
      ring_nf
    rw [mf, pUp, pDn, boltzUp, boltzDn, eu, ed, Real.sinh_eq]
    field_simp

/-- `σm = (μ_σ - μ_{-σ})κ_S`. -/
theorem mL_eq_mul_κSL (σ : Bool) : F.mL σ = (F.cL σ - F.cL (!σ)) * F.κSL := by
  have h := F.two_mul_h_mul_κSL
  cases σ
  · rw [mL_false, Bool.not_false]
    simp only [cL, Bool.cond_false, Bool.cond_true]
    linear_combination h
  · rw [mL_true, Bool.not_true]
    simp only [cL, Bool.cond_false, Bool.cond_true]
    linear_combination -h

/-- **The pair kernel on a mirror pair** is `κ_P`, for every `μ`. -/
theorem kerPL_mirror (n : ℤ) : F.kerPL k n (mirror k n) = F.κPL := by
  have hidx : n + mirror k n + k + 1 = 0 := by unfold mirror; ring
  by_cases hμ : F.μ = 0
  · rw [kerPL, ite_eq_left ⟨hμ, hidx⟩, F.κPL_of_mu hμ, ofReal_mul]
  · have e : F.Bup.z k n + F.Bdn.z k (mirror k n) = 2 * F.μ := F.zL_add_opp_mirror k true n
    have hμ' : (F.μ : ℂ) ≠ 0 := ofReal_ne_zero.mpr hμ
    rw [kerPL, ite_eq_right (fun h ↦ hμ h.1), e, ← F.two_mul_mu_mul_κPL]
    push_cast
    field_simp

/-- The pair kernel off the mirror pairs, on the lattices of the row `σ`. -/
theorem kerPL_of_ne_mirror (σ : Bool) {n m : ℤ} (hm : m ≠ mirror k n) :
    F.kerPL k n m = F.ρ / ((F.BL σ).z k n + (F.BL (!σ)).z k m) := by
  rw [kerPL, ite_eq_right (fun h ↦ hm (by have := h.2; unfold mirror; omega)), F.zL_add_opp]

/-- **The spin-flip kernel at `n = m`** is `κ_S`, for every `h`. -/
theorem kerSLs_self (σ : Bool) (n : ℤ) : F.kerSLs k σ n n = F.κSL := by
  by_cases hh : F.h = 0
  · have hud : F.pUp = F.pDn := by
      rw [F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, hh, add_zero, sub_zero]
    rw [kerSLs, ite_eq_left ⟨hh, rfl⟩, F.κSL_of_h hh, ofReal_mul]
    cases σ
    · rw [Bool.not_false, pL_true, hud]
    · rw [Bool.not_true, pL_false]
  · have hd : F.cL σ - F.cL (!σ) ≠ 0 := by
      intro h0
      have := F.abs_cL_sub σ
      rw [h0, abs_zero] at this
      exact hh (by linarith [abs_nonneg F.h, abs_pos.mpr hh])
    rw [kerSLs, ite_eq_right (fun h ↦ hh h.1), F.zL_sub_opp_self, F.mL_eq_mul_κSL]
    push_cast
    field_simp [show ((F.cL σ : ℂ) - F.cL (!σ)) ≠ 0 by exact_mod_cast hd]

/-- The spin-flip kernel off `n = m`. -/
theorem kerSLs_of_ne (σ : Bool) {n m : ℤ} (hnm : n ≠ m) :
    F.kerSLs k σ n m = F.mL σ / ((F.BL σ).z k n - (F.BL (!σ)).z k m) := by
  rw [kerSLs, ite_eq_right (fun h ↦ hnm h.2)]

/-! ### The factors, as functions of `1/v` -/

/-- `d(s) = P(v)/v⁴` at `s = 1/v`. -/
noncomputable def dL (s : ℂ) : ℂ := 1 + (2 * F.q₀L k - 4 * F.a ^ 2) * s ^ 2 + F.q₀L k ^ 2 * s ^ 4

/-- `P(v) = v⁴d(1/v)`. -/
theorem PL_eq_dL {v : ℂ} (hv : v ≠ 0) : F.PL k v = v ^ 4 * F.dL k v⁻¹ := by
  rw [F.PL_eq, dL]
  field_simp

/-- `B(z) = z⁴/P(z)`. -/
noncomputable def BzL (z : ℂ) : ℂ := z ^ 4 / F.PL k z

/-- `G₁(v) = v²N₂(v)/P(v)`, with `r = r_{-σ}`. -/
noncomputable def G₁L (σ : Bool) (v : ℂ) : ℂ := v ^ 2 * F.N₂L k (F.rL (!σ)) v / F.PL k v

/-- `G₂(v) = v(P(v) - v⁴)/P(v)`. -/
noncomputable def G₂L (v : ℂ) : ℂ := v * (F.PL k v - v ^ 4) / F.PL k v

/-- `G₃(v) = v²(N₀(v) + λrv³)/P(v)`, with `r = r_{-σ}`: `N₀` less its cubic term. -/
noncomputable def G₃L (σ : Bool) (v : ℂ) : ℂ :=
  v ^ 2 * (F.N₀L k (F.rL (!σ)) v + F.lamL * F.rL (!σ) * v ^ 3) / F.PL k v

/-- The coefficient `c₂` of `N₀_r(v) = -λrv³ + c₂v² + c₁v + c₀`. -/
noncomputable def cN₂L : ℂ := -2 * F.a * F.q₀L k - kron k 0 * F.ζL * F.a ^ 2

/-- The coefficient `c₁` of `N₀_r`. -/
noncomputable def cN₁L (r : ℝ) : ℂ := -((2 * F.q₀L k - 4 * F.a ^ 2) * F.lamL * r)

/-- The coefficient `c₀` of `N₀_r`. -/
noncomputable def cN₀L : ℂ :=
  (2 * F.q₀L k - 4 * F.a ^ 2) * (-2 * F.a * F.q₀L k - kron k 0 * F.ζL * F.a ^ 2) -
    (-2 * F.a + kron k 0 * F.ζL) * F.q₀L k ^ 2

/-- `N₀_r(v) = -λrv³ + c₂v² + c₁v + c₀`, with explicit coefficients. -/
theorem N₀L_eq (r : ℝ) (v : ℂ) :
    F.N₀L k r v = -(F.lamL * r) * v ^ 3 + F.cN₂L k * v ^ 2 + F.cN₁L k r * v + F.cN₀L k := by
  simp only [N₀L, N₂L, HL, F.PL_eq, lamL, cN₂L, cN₁L, cN₀L]
  ring

/-- `B(z) = 1/d(1/z)`. -/
theorem BzL_eq {z : ℂ} (hz : z ≠ 0) : F.BzL k z = (F.dL k z⁻¹)⁻¹ := by
  rw [BzL, F.PL_eq_dL k hz]
  field_simp

/-- `G₁` as a function of `1/v`. -/
theorem G₁L_eq (σ : Bool) {v : ℂ} (hv : v ≠ 0) :
    F.G₁L k σ v = (F.dL k v⁻¹)⁻¹ * (-2 * F.a * (1 + 2 * F.a * F.rL (!σ) * v⁻¹ +
      F.q₀L k * v⁻¹ ^ 2) + kron k 0 * F.ζL * (1 - F.a ^ 2 * v⁻¹ ^ 2)) := by
  rw [G₁L, F.PL_eq_dL k hv]
  simp only [N₂L, HL]
  field_simp

/-- `G₂` as a function of `1/v`. -/
theorem G₂L_eq {v : ℂ} (hv : v ≠ 0) :
    F.G₂L k v = (F.dL k v⁻¹)⁻¹ * ((2 * F.q₀L k - 4 * F.a ^ 2) * v⁻¹ + F.q₀L k ^ 2 * v⁻¹ ^ 3) := by
  rw [G₂L, F.PL_eq_dL k hv, dL]
  field_simp
  ring

/-- `G₃` as a function of `1/v`. -/
theorem G₃L_eq (σ : Bool) {v : ℂ} (hv : v ≠ 0) :
    F.G₃L k σ v = (F.dL k v⁻¹)⁻¹ *
      (F.cN₂L k + F.cN₁L k (F.rL (!σ)) * v⁻¹ + F.cN₀L k * v⁻¹ ^ 2) := by
  rw [G₃L, F.PL_eq_dL k hv, F.N₀L_eq]
  field_simp
  ring

/-! ### The opposite-spin input, multiplied by `v²` -/

/-- **The opposite-spin input, times `v²`**, for every `μ`, `h`: with `z = z_σₙ`, `v = z_{-σ}ₘ`,
`r' = r_{-σ} = ρ + σm`,
`v²P(z)χ_{σ,-σ} = G₃(v) + λr'G₂(v) + z²G₁(v) + λ(v²K_P - ρv) - λ(v²K_S + σmv)`. -/
theorem v_sq_mul_PL_mul_chiL_opp (σ : Bool) (n m : ℤ) :
    (F.BL (!σ)).z k m ^ 2 * (F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n m) =
      F.G₃L k σ ((F.BL (!σ)).z k m) + F.lamL * F.rL (!σ) * F.G₂L k ((F.BL (!σ)).z k m) +
        (F.BL σ).z k n ^ 2 * F.G₁L k σ ((F.BL (!σ)).z k m) +
        F.lamL * ((F.BL (!σ)).z k m ^ 2 * F.kerPL k n m - F.ρ * (F.BL (!σ)).z k m) -
        F.lamL * ((F.BL (!σ)).z k m ^ 2 * F.kerSLs k σ n m + F.mL σ * (F.BL (!σ)).z k m) := by
  have hPv := F.PL_BL_ne k (!σ) m
  have hr : (F.rL (!σ) : ℂ) = F.ρ + F.mL σ := by rw [F.rL_not σ]; push_cast; ring
  rw [F.PL_mul_chiL_opp_all k σ n m]
  simp only [G₁L, G₂L, G₃L]
  rw [hr]
  field_simp
  ring

/-! ### The kernel terms -/

private theorem nmul {a b : ℂ} {A B : ℝ} (ha : ‖a‖ ≤ A) (hb : ‖b‖ ≤ B) (hA : 0 ≤ A) :
    ‖a * b‖ ≤ A * B := by
  rw [norm_mul]
  exact mul_le_mul ha hb (norm_nonneg _) hA

private theorem nadd {a b : ℂ} {A B : ℝ} (ha : ‖a‖ ≤ A) (hb : ‖b‖ ≤ B) : ‖a + b‖ ≤ A + B :=
  (norm_add_le _ _).trans (add_le_add ha hb)

private theorem nsub {a b : ℂ} {A B : ℝ} (ha : ‖a‖ ≤ A) (hb : ‖b‖ ≤ B) : ‖a - b‖ ≤ A + B :=
  (norm_sub_le _ _).trans (add_le_add ha hb)

private theorem nreal {x A : ℝ} (h : |x| ≤ A) : ‖(x : ℂ)‖ ≤ A := by
  rwa [norm_real, Real.norm_eq_abs]

/-- `z_σₙ - z_{-σ}ₘ ≠ 0` off `n = m`. -/
theorem zL_sub_opp_ne_zero_of_ne (σ : Bool) {n m : ℤ} (hnm : n ≠ m) :
    (F.BL σ).z k n - (F.BL (!σ)).z k m ≠ 0 :=
  fun h0 ↦ hnm ((F.zL_sub_opp_eq_zero_iff k σ n m).mp h0).2

/-- `|σm| = |m|`. -/
theorem norm_mL_le {K : ℝ} (hm : |F.mf| ≤ K) (σ : Bool) : ‖(F.mL σ : ℂ)‖ ≤ K := by
  rw [norm_real, Real.norm_eq_abs]
  cases σ
  · rw [mL_false, abs_neg]
    exact hm
  · rw [mL_true]
    exact hm

/-- The pair-kernel term `z⁻²(v²K_P - ρv)`, off the fixed point: `-ρ(1/z - 1/(z + v))` off the
mirror pairs and `-κ_P v/z` on them, with `v = 2μ - z`. -/
theorem norm_kerPL_term_le {R M K : ℝ} (hR : F.β / Real.pi ≤ R) (hμ : |F.μ| ≤ M)
    (hρ : |F.ρ| ≤ K) (hP : |F.κPL| ≤ K) (σ : Bool) {n : ℤ} (hn : mirror k n ≠ n) (m : ℤ) :
    ‖((F.BL σ).z k n)⁻¹ ^ 2 *
        ((F.BL (!σ)).z k m ^ 2 * F.kerPL k n m - F.ρ * (F.BL (!σ)).z k m)‖ ≤
      K * (2 * R) + K * (1 + 2 * M * R) := by
  have hβ := F.β_pos
  have hπ := Real.pi_pos
  have hz : (F.BL σ).z k n ≠ 0 := (F.BL σ).z_ne_zero_of_ne k hn
  have ht : ‖((F.BL σ).z k n)⁻¹‖ ≤ R := ((F.BL σ).norm_inv_z_le_of_ne k hn).trans hR
  have hR0 : 0 ≤ R := (norm_nonneg _).trans ht
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hμ
  have hK0 : 0 ≤ K := (abs_nonneg _).trans hρ
  have hs2 : F.β / (2 * Real.pi) ≤ R :=
    (div_le_div_of_nonneg_left hβ.le hπ (by linarith)).trans hR
  have hA : 0 ≤ K * (2 * R) := by positivity
  have hB : 0 ≤ K * (1 + 2 * M * R) := by positivity
  by_cases hm : m = mirror k n
  · subst hm
    have hzv : (F.BL σ).z k n + (F.BL (!σ)).z k (mirror k n) = 2 * F.μ :=
      F.zL_add_opp_mirror k σ n
    have e : ((F.BL σ).z k n)⁻¹ ^ 2 * ((F.BL (!σ)).z k (mirror k n) ^ 2 *
        F.kerPL k n (mirror k n) - F.ρ * (F.BL (!σ)).z k (mirror k n)) =
        -(F.κPL * ((F.BL (!σ)).z k (mirror k n) * ((F.BL σ).z k n)⁻¹)) := by
      rw [F.kerPL_mirror, ← F.two_mul_mu_mul_κPL]
      push_cast
      rw [← hzv]
      field_simp
      ring
    have hvz : ‖(F.BL (!σ)).z k (mirror k n) * ((F.BL σ).z k n)⁻¹‖ ≤ 1 + 2 * M * R := by
      have hv : (F.BL (!σ)).z k (mirror k n) = ((2 * F.μ : ℝ) : ℂ) - (F.BL σ).z k n := by
        push_cast
        linear_combination hzv
      rw [hv, sub_mul, mul_inv_cancel₀ hz]
      refine (norm_sub_le _ _).trans ?_
      rw [norm_mul, norm_one, norm_real, Real.norm_eq_abs, abs_mul, abs_two]
      have := mul_le_mul hμ ht (norm_nonneg _) hM0
      linarith
    rw [e, norm_neg, norm_mul, norm_real, Real.norm_eq_abs]
    calc |F.κPL| * ‖(F.BL (!σ)).z k (mirror k n) * ((F.BL σ).z k n)⁻¹‖ ≤
        K * (1 + 2 * M * R) := mul_le_mul hP hvz (norm_nonneg _) hK0
      _ ≤ _ := by linarith
  · have hzv := F.zL_add_opp_ne_zero_of_ne k σ hm
    have hs : ‖((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹‖ ≤ R :=
      (F.norm_inv_zL_add_opp_le_of_ne k σ hm).trans hs2
    have e : ((F.BL σ).z k n)⁻¹ ^ 2 *
        ((F.BL (!σ)).z k m ^ 2 * F.kerPL k n m - F.ρ * (F.BL (!σ)).z k m) =
        -(F.ρ * (((F.BL σ).z k n)⁻¹ - ((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹)) := by
      rw [F.kerPL_of_ne_mirror k σ hm]
      field_simp
      ring
    rw [e, norm_neg]
    calc ‖(F.ρ : ℂ) * (((F.BL σ).z k n)⁻¹ - ((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹)‖ ≤
        K * (R + R) := nmul (nreal hρ) (nsub ht hs) hK0
      _ ≤ _ := by linarith

/-- The spin-flip term `z⁻²(v²K_S + σmv)`, off the fixed point: `σm(1/(z - v) - 1/z)` off
`n = m` and `κ_S v/z` at `n = m`, with `v = z - (μ_σ - μ_{-σ})`. -/
theorem norm_kerSLs_term_le {R M K : ℝ} (hR : F.β / Real.pi ≤ R) (hh : |F.h| ≤ M)
    (hm : |F.mf| ≤ K) (hS : |F.κSL| ≤ K) (σ : Bool) {n : ℤ} (hn : mirror k n ≠ n) (m : ℤ) :
    ‖((F.BL σ).z k n)⁻¹ ^ 2 *
        ((F.BL (!σ)).z k m ^ 2 * F.kerSLs k σ n m + F.mL σ * (F.BL (!σ)).z k m)‖ ≤
      K * (2 * R) + K * (1 + 2 * M * R) := by
  have hβ := F.β_pos
  have hπ := Real.pi_pos
  have hz : (F.BL σ).z k n ≠ 0 := (F.BL σ).z_ne_zero_of_ne k hn
  have ht : ‖((F.BL σ).z k n)⁻¹‖ ≤ R := ((F.BL σ).norm_inv_z_le_of_ne k hn).trans hR
  have hR0 : 0 ≤ R := (norm_nonneg _).trans ht
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hh
  have hK0 : 0 ≤ K := (abs_nonneg _).trans hm
  have hs2 : F.β / (2 * Real.pi) ≤ R :=
    (div_le_div_of_nonneg_left hβ.le hπ (by linarith)).trans hR
  have hA : 0 ≤ K * (2 * R) := by positivity
  have hB : 0 ≤ K * (1 + 2 * M * R) := by positivity
  by_cases hnm : m = n
  · subst hnm
    have hd := F.zL_sub_opp_self k σ m
    have e : ((F.BL σ).z k m)⁻¹ ^ 2 * ((F.BL (!σ)).z k m ^ 2 * F.kerSLs k σ m m +
        F.mL σ * (F.BL (!σ)).z k m) =
        F.κSL * ((F.BL (!σ)).z k m * ((F.BL σ).z k m)⁻¹) := by
      rw [F.kerSLs_self, F.mL_eq_mul_κSL, ofReal_mul, ← hd]
      field_simp
      ring
    have hvz : ‖(F.BL (!σ)).z k m * ((F.BL σ).z k m)⁻¹‖ ≤ 1 + 2 * M * R := by
      have hv : (F.BL (!σ)).z k m =
          (F.BL σ).z k m - ((F.cL σ - F.cL (!σ) : ℝ) : ℂ) := by
        rw [← hd]
        ring
      rw [hv, sub_mul, mul_inv_cancel₀ hz]
      refine (norm_sub_le _ _).trans ?_
      rw [norm_mul, norm_one, norm_real, Real.norm_eq_abs, F.abs_cL_sub]
      have := mul_le_mul hh ht (norm_nonneg _) hM0
      linarith
    rw [e, norm_mul, norm_real, Real.norm_eq_abs]
    calc |F.κSL| * ‖(F.BL (!σ)).z k m * ((F.BL σ).z k m)⁻¹‖ ≤ K * (1 + 2 * M * R) :=
        mul_le_mul hS hvz (norm_nonneg _) hK0
      _ ≤ _ := by linarith
  · have hnm' : n ≠ m := fun h ↦ hnm h.symm
    have hzv := F.zL_sub_opp_ne_zero_of_ne k σ hnm'
    have hs : ‖((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹‖ ≤ R :=
      (F.norm_inv_zL_sub_opp_le_of_ne k σ hnm').trans hs2
    have e : ((F.BL σ).z k n)⁻¹ ^ 2 *
        ((F.BL (!σ)).z k m ^ 2 * F.kerSLs k σ n m + F.mL σ * (F.BL (!σ)).z k m) =
        F.mL σ * (((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹ - ((F.BL σ).z k n)⁻¹) := by
      rw [F.kerSLs_of_ne k σ hnm']
      field_simp
      ring
    rw [e]
    calc ‖(F.mL σ : ℂ) * (((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹ - ((F.BL σ).z k n)⁻¹)‖ ≤
        K * (R + R) := nmul (F.norm_mL_le hm σ) (nsub hs ht) hK0
      _ ≤ _ := by linarith

/-! ### The bounds of item 6(a) on `z'`, with constants -/

/-- At the fixed point, `|z'|² ≤ 4M²` for `|μ_σ| ≤ M`, `π/β ≤ M`. -/
theorem norm_zOpL_sq_fixed_le {M : ℝ} (hMπ : Real.pi / F.β ≤ M) (σ : Bool) (hc : |F.cL σ| ≤ M)
    {j : ℤ} (hj : mirror k j = j) : ‖(F.BL σ).zOp k j‖ ^ 2 ≤ 4 * M ^ 2 := by
  have h : ‖(F.BL σ).zOp k j‖ ≤ |F.cL σ| + Real.pi / F.β := (F.BL σ).norm_zOp_fixed_le k hj
  have h0 := norm_nonneg ((F.BL σ).zOp k j)
  nlinarith

/-- `|1/P(z)| ≤ R⁴` on both lattices, for `β/π ≤ R`. -/
theorem norm_inv_PL_le {R : ℝ} (hR : F.β / Real.pi ≤ R) (σ : Bool) (j : ℤ) :
    ‖(F.PL k ((F.BL σ).z k j))⁻¹‖ ≤ R ^ 4 := by
  have h : ‖(F.PL k ((F.BL σ).z k j))⁻¹‖ ≤ (F.β / Real.pi) ^ 4 := (F.BL σ).norm_inv_Pph_z_le k j
  exact h.trans (pow_le_pow_left₀ (by have := F.β_pos; positivity) hR 4)

/-- The constant of `norm_chiL_opp_le_all_of`: one summand for the generic entries, the row and
the column of the fixed point, and the entry at the fixed point. -/
def cOppL (B R M Λ K C₁ C₂ C₃ CN : ℝ) : ℝ :=
  B * (R ^ 2 * C₃ + Λ * K * (R ^ 2 * C₂) + C₁ + Λ * (K * (2 * R) + K * (1 + 2 * M * R)) +
      Λ * (K * (2 * R) + K * (1 + 2 * M * R))) +
    4 * M ^ 2 * (R ^ 4 * (C₃ + Λ * K * C₂ + M ^ 2 * C₁ + Λ * (K * (M + M ^ 2 * R)) +
      Λ * (K * (M + M ^ 2 * R)))) +
    4 * M ^ 2 * (B * (R ^ 2 * (Λ * (K * R + K * R)) + (R ^ 2 * CN + CN) * R ^ 4)) +
    4 * M ^ 2 * (4 * M ^ 2) * (R ^ 4 * (Λ * (K + K) + (CN + M ^ 2 * CN) * R ^ 4))

/-- **Opposite spins on `z'`, with constants**, for every `μ`, `h`:
`|z'_σₙ|²|z'_{-σ}ₘ|²|χ_{σ,-σ}(n, m)|` is bounded by `cOppL` of the constants below, which bound
`β/π`, `|μ|`, `|h|`, `|μ_σ|`, `π/β`, `|λ|`, `|ρ|`, `|m|`, `|r'|`, `|κ_P|`, `|κ_S|`, the factors off
the fixed point and `N₀`, `N₂` at it. -/
theorem norm_chiL_opp_le_all_of {B R M Λ K C₁ C₂ C₃ CN : ℝ} (σ : Bool)
    (hR : F.β / Real.pi ≤ R) (hMπ : Real.pi / F.β ≤ M) (hμ : |F.μ| ≤ M) (hh : |F.h| ≤ M)
    (hc : ∀ τ, |F.cL τ| ≤ M)
    (hB : ∀ j, mirror k j ≠ j → ‖F.BzL k ((F.BL σ).z k j)‖ ≤ B) (hB0 : 0 ≤ B)
    (hΛ : ‖F.lamL‖ ≤ Λ)
    (hρ : |F.ρ| ≤ K) (hm : |F.mf| ≤ K) (hr : |F.rL (!σ)| ≤ K) (hP : |F.κPL| ≤ K)
    (hS : |F.κSL| ≤ K)
    (hC₁ : ∀ j, mirror k j ≠ j → ‖F.G₁L k σ ((F.BL (!σ)).z k j)‖ ≤ C₁)
    (hC₂ : ∀ j, mirror k j ≠ j → ‖F.G₂L k ((F.BL (!σ)).z k j)‖ ≤ C₂)
    (hC₃ : ∀ j, mirror k j ≠ j → ‖F.G₃L k σ ((F.BL (!σ)).z k j)‖ ≤ C₃)
    (hN₀ : ∀ j, mirror k j = j → ‖F.N₀L k (F.rL (!σ)) ((F.BL (!σ)).z k j)‖ ≤ CN)
    (hN₂ : ∀ j, mirror k j = j → ‖F.N₂L k (F.rL (!σ)) ((F.BL (!σ)).z k j)‖ ≤ CN)
    (hC₁0 : 0 ≤ C₁) (hC₂0 : 0 ≤ C₂) (hC₃0 : 0 ≤ C₃) (hCN0 : 0 ≤ CN) (n m : ℤ) :
    ‖(F.BL σ).zOp k n‖ ^ 2 * ‖(F.BL (!σ)).zOp k m‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖ ≤
      cOppL B R M Λ K C₁ C₂ C₃ CN := by
  have hβ := F.β_pos
  have hπ := Real.pi_pos
  have hR0 : 0 ≤ R := (div_nonneg hβ.le hπ.le).trans hR
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hμ
  have hΛ0 : 0 ≤ Λ := (norm_nonneg _).trans hΛ
  have hK0 : 0 ≤ K := (abs_nonneg _).trans hρ
  have hs2 : F.β / (2 * Real.pi) ≤ R :=
    (div_le_div_of_nonneg_left hβ.le hπ (by linarith)).trans hR
  have hmL := F.norm_mL_le hm σ
  unfold cOppL
  set c₁ := B * (R ^ 2 * C₃ + Λ * K * (R ^ 2 * C₂) + C₁ +
      Λ * (K * (2 * R) + K * (1 + 2 * M * R)) + Λ * (K * (2 * R) + K * (1 + 2 * M * R)))
  set c₂ := 4 * M ^ 2 * (R ^ 4 * (C₃ + Λ * K * C₂ + M ^ 2 * C₁ + Λ * (K * (M + M ^ 2 * R)) +
      Λ * (K * (M + M ^ 2 * R))))
  set c₃ := 4 * M ^ 2 * (B * (R ^ 2 * (Λ * (K * R + K * R)) + (R ^ 2 * CN + CN) * R ^ 4))
  set c₄ := 4 * M ^ 2 * (4 * M ^ 2) * (R ^ 4 * (Λ * (K + K) + (CN + M ^ 2 * CN) * R ^ 4))
  have hc₁ : 0 ≤ c₁ := by positivity
  have hc₂ : 0 ≤ c₂ := by positivity
  have hc₃ : 0 ≤ c₃ := by positivity
  have hc₄ : 0 ≤ c₄ := by positivity
  have hPz := F.PL_BL_ne k σ n
  have hPv := F.PL_BL_ne k (!σ) m
  by_cases hnf : mirror k n = n
  · by_cases hmf : mirror k m = m
    · -- the entry at the fixed point, a mirror pair with `n = m`: `K_P = κ_P`, `K_S = κ_S`
      have hnm : m = n := by unfold mirror at hnf hmf; omega
      subst hnm
      have hP' : F.kerPL k m m = F.κPL := by
        have h := F.kerPL_mirror k m
        rwa [hnf] at h
      have hzM : ‖(F.BL σ).z k m‖ ≤ M := by
        rw [(F.BL σ).z_of_fixed k hnf]
        exact nreal (hc σ)
      have hz2 : ‖(F.BL σ).z k m ^ 2‖ ≤ M ^ 2 := by
        rw [norm_pow]
        exact pow_le_pow_left₀ (norm_nonneg _) hzM 2
      have hχ : F.chiL σ (!σ) k m m = (F.PL k ((F.BL σ).z k m))⁻¹ *
          (F.PL k ((F.BL σ).z k m) * F.chiL σ (!σ) k m m) := by
        field_simp
      rw [F.PL_mul_chiL_opp_all k σ m m, hP', F.kerSLs_self, div_eq_mul_inv] at hχ
      have hb : ‖F.chiL σ (!σ) k m m‖ ≤ R ^ 4 * (Λ * (K + K) + (CN + M ^ 2 * CN) * R ^ 4) := by
        rw [hχ]
        exact nmul (F.norm_inv_PL_le k hR σ m) (nadd (nmul hΛ (nsub (nreal hP) (nreal hS)) hΛ0)
          (nmul (nadd (hN₀ m hmf) (nmul hz2 (hN₂ m hmf) (by positivity)))
            (F.norm_inv_PL_le k hR (!σ) m) (by positivity))) (by positivity)
      calc _ ≤ 4 * M ^ 2 * (4 * M ^ 2) * (R ^ 4 * (Λ * (K + K) + (CN + M ^ 2 * CN) * R ^ 4)) :=
            mul_le_mul (mul_le_mul (F.norm_zOpL_sq_fixed_le k hMπ σ (hc σ) hnf)
              (F.norm_zOpL_sq_fixed_le k hMπ (!σ) (hc (!σ)) hmf) (by positivity)
              (by positivity)) hb (norm_nonneg _) (by positivity)
        _ ≤ c₁ + c₂ + c₃ + c₄ := by linarith
    · -- the row of the fixed point, `z = μ_σ`
      have hmn : m ≠ n := fun h ↦ hmf (h ▸ hnf)
      have hnm : n ≠ m := fun h ↦ hmn h.symm
      have hmir : m ≠ mirror k n := by rw [hnf]; exact hmn
      have hzM : ‖(F.BL σ).z k n‖ ≤ M := by
        rw [(F.BL σ).z_of_fixed k hnf]
        exact nreal (hc σ)
      have hzM' : ‖-(F.BL σ).z k n‖ ≤ M := by rw [norm_neg]; exact hzM
      have hz2 : ‖(F.BL σ).z k n ^ 2‖ ≤ M ^ 2 := by
        rw [norm_pow]
        exact pow_le_pow_left₀ (norm_nonneg _) hzM 2
      have hzv := F.zL_add_opp_ne_zero_of_ne k σ hmir
      have hzmv := F.zL_sub_opp_ne_zero_of_ne k σ hnm
      have hs : ‖((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹‖ ≤ R :=
        (F.norm_inv_zL_add_opp_le_of_ne k σ hmir).trans hs2
      have hs' : ‖((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹‖ ≤ R :=
        (F.norm_inv_zL_sub_opp_le_of_ne k σ hnm).trans hs2
      have eX : (F.BL (!σ)).z k m ^ 2 * F.kerPL k n m - F.ρ * (F.BL (!σ)).z k m =
          F.ρ * (-(F.BL σ).z k n + (F.BL σ).z k n ^ 2 *
            ((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹) := by
        rw [F.kerPL_of_ne_mirror k σ hmir]
        field_simp
        ring
      have eY : (F.BL (!σ)).z k m ^ 2 * F.kerSLs k σ n m + F.mL σ * (F.BL (!σ)).z k m =
          F.mL σ * (-(F.BL σ).z k n + (F.BL σ).z k n ^ 2 *
            ((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹) := by
        rw [F.kerSLs_of_ne k σ hnm]
        field_simp
        ring
      have e : (F.BL (!σ)).z k m ^ 2 * F.chiL σ (!σ) k n m = (F.PL k ((F.BL σ).z k n))⁻¹ *
          ((F.BL (!σ)).z k m ^ 2 * (F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n m)) := by
        field_simp
      rw [F.v_sq_mul_PL_mul_chiL_opp, eX, eY] at e
      have hb : ‖(F.BL (!σ)).z k m‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖ ≤
          R ^ 4 * (C₃ + Λ * K * C₂ + M ^ 2 * C₁ + Λ * (K * (M + M ^ 2 * R)) +
            Λ * (K * (M + M ^ 2 * R))) := by
        rw [← norm_pow, ← norm_mul, e]
        exact nmul (F.norm_inv_PL_le k hR σ n) (nsub (nadd (nadd (nadd (hC₃ m hmf)
          (nmul (nmul hΛ (nreal hr) hΛ0) (hC₂ m hmf) (by positivity)))
          (nmul hz2 (hC₁ m hmf) (by positivity)))
          (nmul hΛ (nmul (nreal hρ) (nadd hzM' (nmul hz2 hs (by positivity))) hK0) hΛ0))
          (nmul hΛ (nmul hmL (nadd hzM' (nmul hz2 hs' (by positivity))) hK0) hΛ0))
          (by positivity)
      have hzOm : ‖(F.BL (!σ)).zOp k m‖ = ‖(F.BL (!σ)).z k m‖ := by
        rw [(F.BL (!σ)).zOp_of_ne k hmf]
      calc ‖(F.BL σ).zOp k n‖ ^ 2 * ‖(F.BL (!σ)).zOp k m‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖ =
          ‖(F.BL σ).zOp k n‖ ^ 2 * (‖(F.BL (!σ)).z k m‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖) := by
            rw [hzOm]
            ring
        _ ≤ c₂ := mul_le_mul (F.norm_zOpL_sq_fixed_le k hMπ σ (hc σ) hnf) hb (by positivity)
            (by positivity)
        _ ≤ c₁ + c₂ + c₃ + c₄ := by linarith
  have hz0 := (F.BL σ).z_ne_zero_of_ne k hnf
  have ht : ‖((F.BL σ).z k n)⁻¹‖ ≤ R := ((F.BL σ).norm_inv_z_le_of_ne k hnf).trans hR
  have ht2 : ‖((F.BL σ).z k n)⁻¹ ^ 2‖ ≤ R ^ 2 := by
    rw [norm_pow]
    exact pow_le_pow_left₀ (norm_nonneg _) ht 2
  have hzOn : ‖(F.BL σ).zOp k n‖ = ‖(F.BL σ).z k n‖ := by rw [(F.BL σ).zOp_of_ne k hnf]
  by_cases hmf : mirror k m = m
  · -- the column of the fixed point, `v = μ_{-σ}`
    have hnm : n ≠ m := fun h ↦ hnf (h ▸ hmf)
    have hmir : m ≠ mirror k n := fun h ↦ hnm (by unfold mirror at hmf h; omega)
    have hzv := F.zL_add_opp_ne_zero_of_ne k σ hmir
    have hzmv := F.zL_sub_opp_ne_zero_of_ne k σ hnm
    have hs : ‖((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹‖ ≤ R :=
      (F.norm_inv_zL_add_opp_le_of_ne k σ hmir).trans hs2
    have hs' : ‖((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹‖ ≤ R :=
      (F.norm_inv_zL_sub_opp_le_of_ne k σ hnm).trans hs2
    have e : (F.BL σ).z k n ^ 2 * F.chiL σ (!σ) k n m = F.BzL k ((F.BL σ).z k n) *
        (((F.BL σ).z k n)⁻¹ ^ 2 * (F.lamL * (F.ρ * ((F.BL σ).z k n + (F.BL (!σ)).z k m)⁻¹ -
          F.mL σ * ((F.BL σ).z k n - (F.BL (!σ)).z k m)⁻¹)) +
        (((F.BL σ).z k n)⁻¹ ^ 2 * F.N₀L k (F.rL (!σ)) ((F.BL (!σ)).z k m) +
          F.N₂L k (F.rL (!σ)) ((F.BL (!σ)).z k m)) * (F.PL k ((F.BL (!σ)).z k m))⁻¹) := by
      have hχ : F.chiL σ (!σ) k n m = (F.lamL * (F.kerPL k n m - F.kerSLs k σ n m) +
          (F.N₀L k (F.rL (!σ)) ((F.BL (!σ)).z k m) + (F.BL σ).z k n ^ 2 *
            F.N₂L k (F.rL (!σ)) ((F.BL (!σ)).z k m)) / F.PL k ((F.BL (!σ)).z k m)) /
          F.PL k ((F.BL σ).z k n) := by
        rw [eq_div_iff hPz, mul_comm, F.PL_mul_chiL_opp_all k σ n m]
      rw [hχ, F.kerPL_of_ne_mirror k σ hmir, F.kerSLs_of_ne k σ hnm, BzL]
      field_simp
    have hb : ‖(F.BL σ).z k n‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖ ≤
        B * (R ^ 2 * (Λ * (K * R + K * R)) + (R ^ 2 * CN + CN) * R ^ 4) := by
      rw [← norm_pow, ← norm_mul, e]
      exact nmul (hB n hnf) (nadd (nmul ht2 (nmul hΛ (nsub (nmul (nreal hρ) hs hK0)
        (nmul hmL hs' hK0)) hΛ0) (by positivity))
        (nmul (nadd (nmul ht2 (hN₀ m hmf) (by positivity)) (hN₂ m hmf))
          (F.norm_inv_PL_le k hR (!σ) m) (by positivity))) hB0
    calc ‖(F.BL σ).zOp k n‖ ^ 2 * ‖(F.BL (!σ)).zOp k m‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖ =
        ‖(F.BL (!σ)).zOp k m‖ ^ 2 * (‖(F.BL σ).z k n‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖) := by
          rw [hzOn]
          ring
      _ ≤ c₃ := mul_le_mul (F.norm_zOpL_sq_fixed_le k hMπ (!σ) (hc (!σ)) hmf) hb
          (by positivity) (by positivity)
      _ ≤ c₁ + c₂ + c₃ + c₄ := by linarith
  -- off the fixed point
  have hv0 := (F.BL (!σ)).z_ne_zero_of_ne k hmf
  have hzOm : ‖(F.BL (!σ)).zOp k m‖ = ‖(F.BL (!σ)).z k m‖ := by
    rw [(F.BL (!σ)).zOp_of_ne k hmf]
  have key : (F.BL σ).z k n ^ 2 * (F.BL (!σ)).z k m ^ 2 * F.chiL σ (!σ) k n m =
      F.BzL k ((F.BL σ).z k n) *
        (((F.BL σ).z k n)⁻¹ ^ 2 * F.G₃L k σ ((F.BL (!σ)).z k m) +
          F.lamL * F.rL (!σ) * (((F.BL σ).z k n)⁻¹ ^ 2 * F.G₂L k ((F.BL (!σ)).z k m)) +
          F.G₁L k σ ((F.BL (!σ)).z k m) +
          F.lamL * (((F.BL σ).z k n)⁻¹ ^ 2 *
            ((F.BL (!σ)).z k m ^ 2 * F.kerPL k n m - F.ρ * (F.BL (!σ)).z k m)) -
          F.lamL * (((F.BL σ).z k n)⁻¹ ^ 2 *
            ((F.BL (!σ)).z k m ^ 2 * F.kerSLs k σ n m + F.mL σ * (F.BL (!σ)).z k m))) := by
    have e : (F.BL σ).z k n ^ 2 * (F.BL (!σ)).z k m ^ 2 * F.chiL σ (!σ) k n m =
        F.BzL k ((F.BL σ).z k n) * (((F.BL σ).z k n)⁻¹ ^ 2 *
          ((F.BL (!σ)).z k m ^ 2 * (F.PL k ((F.BL σ).z k n) * F.chiL σ (!σ) k n m))) := by
      rw [BzL]
      field_simp
    rw [e, F.v_sq_mul_PL_mul_chiL_opp]
    field_simp
  have hlhs : ‖(F.BL σ).zOp k n‖ ^ 2 * ‖(F.BL (!σ)).zOp k m‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖ =
      ‖(F.BL σ).z k n ^ 2 * (F.BL (!σ)).z k m ^ 2 * F.chiL σ (!σ) k n m‖ := by
    rw [hzOn, hzOm, norm_mul, norm_mul, norm_pow, norm_pow]
  rw [hlhs, key]
  exact (nmul (hB n hnf) (nsub (nadd (nadd (nadd (nmul ht2 (hC₃ m hmf) (by positivity))
    (nmul (nmul hΛ (nreal hr) hΛ0) (nmul ht2 (hC₂ m hmf) (by positivity)) (by positivity)))
    (hC₁ m hmf)) (nmul hΛ (F.norm_kerPL_term_le k hR hμ hρ hP σ hnf m) hΛ0))
    (nmul hΛ (F.norm_kerSLs_term_le k hR hh hm hS σ hnf m) hΛ0)) hB0).trans (by linarith)

/-! ### Equal spins -/

/-- `E(z) = z²(z² - a²)/P(z)`. -/
noncomputable def EzL (z : ℂ) : ℂ := z ^ 2 * (z ^ 2 - F.a ^ 2) / F.PL k z

/-- `E` as a function of `1/z`. -/
theorem EzL_eq {z : ℂ} (hz : z ≠ 0) : F.EzL k z = (F.dL k z⁻¹)⁻¹ * (1 - F.a ^ 2 * z⁻¹ ^ 2) := by
  rw [EzL, F.PL_eq_dL k hz]
  field_simp

/-- **Equal spins, off the diagonal, on `z'`**: `χ_σσ(n, m) = 0` for `n ≠ m` unless `k = 0`, where
there is no fixed point and `z²v²χ = ζ_σE(z)E(v)`. -/
theorem norm_chiL_self_le_all_of {Z CE : ℝ} (σ : Bool) (hZ : ‖F.ζLs σ‖ ≤ Z)
    (hE : ∀ j, mirror k j ≠ j → ‖F.EzL k ((F.BL σ).z k j)‖ ≤ CE) (hCE0 : 0 ≤ CE) {n m : ℤ}
    (hnm : n ≠ m) :
    ‖(F.BL σ).zOp k n‖ ^ 2 * ‖(F.BL σ).zOp k m‖ ^ 2 * ‖F.chiL σ σ k n m‖ ≤ Z * CE * CE := by
  have hZ0 : 0 ≤ Z := (norm_nonneg _).trans hZ
  by_cases hk : k = 0
  · subst hk
    have hn : mirror 0 n ≠ n := by unfold mirror; omega
    have hm : mirror 0 m ≠ m := by unfold mirror; omega
    have hz := (F.BL σ).z_ne_zero_of_ne 0 hn
    have hv := (F.BL σ).z_ne_zero_of_ne 0 hm
    have hPz := F.PL_BL_ne 0 σ n
    have hPv := F.PL_BL_ne 0 σ m
    have key : (F.BL σ).z 0 n ^ 2 * (F.BL σ).z 0 m ^ 2 * F.chiL σ σ 0 n m =
        F.ζLs σ * F.EzL 0 ((F.BL σ).z 0 n) * F.EzL 0 ((F.BL σ).z 0 m) := by
      have h := F.PL_mul_chiL_self 0 σ n m
      rw [kron_of_ne hnm, mul_zero, zero_add, kron_self, one_mul] at h
      rw [show F.chiL σ σ 0 n m = F.PL 0 ((F.BL σ).z 0 n) * F.chiL σ σ 0 n m /
        F.PL 0 ((F.BL σ).z 0 n) by field_simp, h, EzL, EzL]
      field_simp
    rw [(F.BL σ).zOp_of_ne 0 hn, (F.BL σ).zOp_of_ne 0 hm, ← norm_pow, ← norm_pow, ← norm_mul,
      ← norm_mul, key]
    exact nmul (nmul hZ (hE n hn) hZ0) (hE m hm) (by positivity)
  · have h0 : F.chiL σ σ k n m = 0 := by
      have h := F.PL_mul_chiL_self k σ n m
      rw [kron_of_ne hnm, kron_of_ne hk, mul_zero, zero_add, zero_mul, zero_mul, zero_div] at h
      exact (mul_eq_zero.mp h).resolve_left (F.PL_BL_ne k σ n)
    rw [h0, norm_zero, mul_zero]
    positivity

/-- `z³(χ_σσ(n, n) + β/z²)` as a function of `t = 1/z`. -/
noncomputable def dgL (σ : Bool) (t : ℂ) : ℂ :=
  (F.dL k t)⁻¹ * (F.β * (-2 * F.a * F.rL σ + (2 * F.q₀L k - 4 * F.a ^ 2 - F.q₀L k) * t +
      F.q₀L k ^ 2 * t ^ 3)) +
    ((F.dL k t)⁻¹) ^ 2 * (kron k 0 * F.ζLs σ * ((1 - F.a ^ 2 * t ^ 2) ^ 2 * t))

/-- `z³(χ_σσ(n, n) + β/z²) = dg(1/z)`, for `z ≠ 0`. -/
theorem z_cube_mul_chiL_diag (σ : Bool) {n : ℤ} (hz : (F.BL σ).z k n ≠ 0) :
    (F.BL σ).z k n ^ 3 * (F.chiL σ σ k n n + F.β / (F.BL σ).z k n ^ 2) =
      F.dgL k σ ((F.BL σ).z k n)⁻¹ := by
  have hP := F.PL_BL_ne k σ n
  have hD : F.dL k ((F.BL σ).z k n)⁻¹ = F.PL k ((F.BL σ).z k n) / (F.BL σ).z k n ^ 4 := by
    rw [F.PL_eq_dL k hz]
    field_simp
  have hχ : F.chiL σ σ k n n = (-F.β * F.HL k (F.rL σ) ((F.BL σ).z k n) +
      kron k 0 * F.ζLs σ * (((F.BL σ).z k n ^ 2 - F.a ^ 2) ^ 2 / F.PL k ((F.BL σ).z k n))) /
        F.PL k ((F.BL σ).z k n) := by
    rw [eq_div_iff hP, mul_comm, F.PL_mul_chiL_self k σ n n, kron_self]
    ring
  rw [dgL, hD, hχ]
  field_simp
  simp only [HL, F.PL_eq]
  ring

/-- **Equal spins, the diagonal, on `z'`, with constants**. -/
theorem norm_chiL_diag_le_all_of {M Cd Cf : ℝ} (σ : Bool) (hMπ : Real.pi / F.β ≤ M)
    (hc : |F.cL σ| ≤ M) (hd : ∀ j, mirror k j ≠ j → ‖F.dgL k σ ((F.BL σ).z k j)⁻¹‖ ≤ Cd)
    (hf : ∀ j, mirror k j = j → ‖F.chiL σ σ k j j + F.β / (F.BL σ).zOp k j ^ 2‖ ≤ Cf)
    (hCd0 : 0 ≤ Cd) (hCf0 : 0 ≤ Cf) (n : ℤ) :
    ‖(F.BL σ).zOp k n‖ ^ 3 * ‖F.chiL σ σ k n n + F.β / (F.BL σ).zOp k n ^ 2‖ ≤
      Cd + 8 * M ^ 3 * Cf := by
  have hM0 : 0 ≤ M := (by have := F.β_pos; positivity : (0 : ℝ) ≤ Real.pi / F.β).trans hMπ
  have h₂ : 0 ≤ 8 * M ^ 3 * Cf := by positivity
  by_cases hnf : mirror k n = n
  · have h : ‖(F.BL σ).zOp k n‖ ≤ |F.cL σ| + Real.pi / F.β := (F.BL σ).norm_zOp_fixed_le k hnf
    have h3 : ‖(F.BL σ).zOp k n‖ ^ 3 ≤ 8 * M ^ 3 := by
      have := pow_le_pow_left₀ (norm_nonneg _) (h.trans (show |F.cL σ| + Real.pi / F.β ≤ 2 * M
        by linarith)) 3
      linarith [show (2 * M) ^ 3 = 8 * M ^ 3 by ring]
    calc _ ≤ 8 * M ^ 3 * Cf := mul_le_mul h3 (hf n hnf) (norm_nonneg _) (by positivity)
      _ ≤ _ := by linarith
  · rw [(F.BL σ).zOp_of_ne k hnf, ← norm_pow, ← norm_mul,
      F.z_cube_mul_chiL_diag k σ ((F.BL σ).z_ne_zero_of_ne k hnf)]
    exact (hd n hnf).trans (by linarith)

/-! ### The bubble -/

/-- `z(-z²b_σ/β - 1)` as a function of `t = 1/z`. -/
noncomputable def TbL (σ : Bool) (t : ℂ) : ℂ :=
  (F.dL k t)⁻¹ * (2 * F.a * F.rL σ + (F.a ^ 2 * F.rL σ ^ 2 - F.wL k ^ 2 / 4 -
    (2 * F.q₀L k - 4 * F.a ^ 2)) * t - F.q₀L k ^ 2 * t ^ 3)

/-- `z(-z²b_σ/β - 1) = T_b(1/z)`, for `z ≠ 0`. -/
theorem zL_mul_tL_eq (σ : Bool) {n : ℤ} (hz : (F.BL σ).z k n ≠ 0) :
    (F.BL σ).z k n * (-((F.BL σ).z k n ^ 2 / F.β) * F.bL σ k n - 1) =
      F.TbL k σ ((F.BL σ).z k n)⁻¹ := by
  have hP := F.PL_BL_ne k σ n
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  have hD : F.dL k ((F.BL σ).z k n)⁻¹ = F.PL k ((F.BL σ).z k n) / (F.BL σ).z k n ^ 4 := by
    rw [F.PL_eq_dL k hz]
    field_simp
  rw [TbL, hD, F.bL_eq]
  field_simp
  simp only [F.PL_eq]
  ring

/-! ### The lattices of the two spins through the axes -/

/-- The point `z'` of `encL.symm j` on the lattice of its spin. -/
noncomputable def zLeOp (j : ℤ) : ℂ := (F.BL (encL.symm j).1).zOp k (encL.symm j).2

/-- **The lattices of the two spins for every `μ`, `h`**, enumerated by `encL`, each with its
mirror fixed point moved as in `OpLatticeAll.lean`. -/
noncomputable def opLatLAll : OpLattice where
  β := F.β
  β_pos := F.β_pos
  z := F.zLeOp k
  z_ne_zero j := (F.BL _).zOp_ne_zero k _
  summable_inv_sq := by
    refine (encL.symm.summable_iff
      (f := fun p : Bool × ℤ ↦ ‖(F.BL p.1).zOp k p.2‖⁻¹ ^ 2)).mpr ?_
    refine (summable_prod_of_nonneg fun _ ↦ by positivity).mpr
      ⟨fun σ ↦ (F.BL σ).summable_inv_norm_zOp_sq k, Summable.of_finite⟩

@[simp] theorem opLatLAll_z (p : Bool × ℤ) :
    (F.opLatLAll k).z (encL p) = (F.BL p.1).zOp k p.2 := by
  simp [opLatLAll, zLeOp]

@[simp] theorem opLatLAll_β : (F.opLatLAll k).β = F.β := rfl

/-! ### The bounds at fixed parameters -/

/-- A function of `1/z` continuous at `0` is bounded on the lattice of `σ` off the fixed point. -/
theorem bdd_inv_zL (σ : Bool) (q : ℂ → ℂ) (hq : ContinuousAt q 0) :
    ∃ C, 0 ≤ C ∧ ∀ j, mirror k j ≠ j → ‖q ((F.BL σ).z k j)⁻¹‖ ≤ C := by
  obtain ⟨C, hC⟩ := (F.BL σ).bdd_of_inv_zOp k (f := fun j ↦ q ((F.BL σ).z k j)⁻¹) q hq
    (((F.BL σ).eventually_zOp_eq k).mono fun j h ↦ by rw [h])
  exact ⟨max C 0, le_max_right _ _, fun j _ ↦ (hC j).trans (le_max_left _ _)⟩

/-- `d(0) = 1`. -/
theorem dL_zero : F.dL k 0 = 1 := by simp [dL]

/-- `1/d` is continuous at `0`. -/
theorem continuousAt_inv_dL : ContinuousAt (fun t ↦ (F.dL k t)⁻¹) 0 :=
  ContinuousAt.inv₀ (by unfold dL; fun_prop) (by rw [dL_zero]; exact one_ne_zero)

/-- `|μ_σ| ≤ |μ| + |h|`. -/
theorem abs_cL_le (τ : Bool) : |F.cL τ| ≤ |F.μ| + |F.h| := by
  cases τ <;> simp only [cL, Bool.cond_true, Bool.cond_false]
  · exact abs_sub _ _
  · exact abs_add_le _ _

/-- **Opposite spins on `z'`**, at fixed parameters. -/
theorem exists_chiL_opp_bound (σ : Bool) : ∃ c, ∀ n m,
    ‖(F.BL σ).zOp k n‖ ^ 2 * ‖(F.BL (!σ)).zOp k m‖ ^ 2 * ‖F.chiL σ (!σ) k n m‖ ≤ c := by
  have hdq := F.continuousAt_inv_dL k
  obtain ⟨B, hB0, hB⟩ := F.bdd_inv_zL k σ _ hdq
  obtain ⟨C₁, hC₁0, hC₁⟩ := F.bdd_inv_zL k (!σ) (fun s ↦ (F.dL k s)⁻¹ *
    (-2 * F.a * (1 + 2 * F.a * F.rL (!σ) * s + F.q₀L k * s ^ 2) +
      kron k 0 * F.ζL * (1 - F.a ^ 2 * s ^ 2))) (hdq.mul (by fun_prop))
  obtain ⟨C₂, hC₂0, hC₂⟩ := F.bdd_inv_zL k (!σ) (fun s ↦ (F.dL k s)⁻¹ *
    ((2 * F.q₀L k - 4 * F.a ^ 2) * s + F.q₀L k ^ 2 * s ^ 3)) (hdq.mul (by fun_prop))
  obtain ⟨C₃, hC₃0, hC₃⟩ := F.bdd_inv_zL k (!σ) (fun s ↦ (F.dL k s)⁻¹ *
    (F.cN₂L k + F.cN₁L k (F.rL (!σ)) * s + F.cN₀L k * s ^ 2)) (hdq.mul (by fun_prop))
  obtain ⟨j₀, hj₀⟩ : ∃ j₀ : ℤ, j₀ = (-k - 1) / 2 := ⟨_, rfl⟩
  have hπβ : 0 ≤ Real.pi / F.β := by have := F.β_pos; positivity
  obtain ⟨M, hM⟩ : ∃ M : ℝ, M = |F.μ| + |F.h| + Real.pi / F.β := ⟨_, rfl⟩
  obtain ⟨K, hKd⟩ : ∃ K : ℝ, K = |F.ρ| + |F.mf| + |F.rL (!σ)| + |F.κPL| + |F.κSL| := ⟨_, rfl⟩
  have hK : |F.ρ| ≤ K ∧ |F.mf| ≤ K ∧ |F.rL (!σ)| ≤ K ∧ |F.κPL| ≤ K ∧ |F.κSL| ≤ K := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
      linarith [abs_nonneg F.ρ, abs_nonneg F.mf, abs_nonneg (F.rL (!σ)), abs_nonneg F.κPL,
        abs_nonneg F.κSL]
  refine ⟨cOppL B (F.β / Real.pi) M ‖F.lamL‖ K C₁ C₂ C₃
    (‖F.N₀L k (F.rL (!σ)) ((F.BL (!σ)).z k j₀)‖ + ‖F.N₂L k (F.rL (!σ)) ((F.BL (!σ)).z k j₀)‖),
    fun n m ↦ F.norm_chiL_opp_le_all_of k σ le_rfl
      (by linarith [abs_nonneg F.μ, abs_nonneg F.h]) (by linarith [abs_nonneg F.h])
      (by linarith [abs_nonneg F.μ]) (fun τ ↦ by linarith [F.abs_cL_le τ])
      (fun j hj ↦ by rw [F.BzL_eq k ((F.BL σ).z_ne_zero_of_ne k hj)]; exact hB j hj) hB0
      le_rfl hK.1 hK.2.1 hK.2.2.1 hK.2.2.2.1 hK.2.2.2.2
      (fun j hj ↦ by rw [F.G₁L_eq k σ ((F.BL (!σ)).z_ne_zero_of_ne k hj)]; exact hC₁ j hj)
      (fun j hj ↦ by rw [F.G₂L_eq k ((F.BL (!σ)).z_ne_zero_of_ne k hj)]; exact hC₂ j hj)
      (fun j hj ↦ by rw [F.G₃L_eq k σ ((F.BL (!σ)).z_ne_zero_of_ne k hj)]; exact hC₃ j hj)
      (fun j hj ↦ by
        rw [eq_of_mirror_eq_self hj, ← hj₀]
        exact le_add_of_nonneg_right (norm_nonneg _))
      (fun j hj ↦ by
        rw [eq_of_mirror_eq_self hj, ← hj₀]
        exact le_add_of_nonneg_left (norm_nonneg _))
      hC₁0 hC₂0 hC₃0 (by positivity) n m⟩

/-- **Equal spins on `z'`**, at fixed parameters. -/
theorem exists_chiL_self_bound (σ : Bool) : ∃ c, ∀ n m, n ≠ m →
    ‖(F.BL σ).zOp k n‖ ^ 2 * ‖(F.BL σ).zOp k m‖ ^ 2 * ‖F.chiL σ σ k n m‖ ≤ c := by
  obtain ⟨CE, hCE0, hCE⟩ := F.bdd_inv_zL k σ (fun t ↦ (F.dL k t)⁻¹ * (1 - F.a ^ 2 * t ^ 2))
    ((F.continuousAt_inv_dL k).mul (by fun_prop))
  exact ⟨‖F.ζLs σ‖ * CE * CE, fun n m hnm ↦ F.norm_chiL_self_le_all_of k σ le_rfl
    (fun j hj ↦ by rw [F.EzL_eq k ((F.BL σ).z_ne_zero_of_ne k hj)]; exact hCE j hj) hCE0 hnm⟩

/-- **The diagonal on `z'`**, at fixed parameters. -/
theorem exists_chiL_diag_bound (σ : Bool) : ∃ c, ∀ n,
    ‖(F.BL σ).zOp k n‖ ^ 3 * ‖F.chiL σ σ k n n + F.β / (F.BL σ).zOp k n ^ 2‖ ≤ c := by
  obtain ⟨Cd, hCd0, hCd⟩ := F.bdd_inv_zL k σ (F.dgL k σ) (by
    unfold dgL dL
    fun_prop (disch := simp))
  obtain ⟨j₀, hj₀⟩ : ∃ j₀ : ℤ, j₀ = (-k - 1) / 2 := ⟨_, rfl⟩
  have hπβ : 0 ≤ Real.pi / F.β := by have := F.β_pos; positivity
  exact ⟨Cd + 8 * (|F.cL σ| + Real.pi / F.β) ^ 3 *
      ‖F.chiL σ σ k j₀ j₀ + F.β / (F.BL σ).zOp k j₀ ^ 2‖, fun n ↦
    F.norm_chiL_diag_le_all_of k σ (by linarith [abs_nonneg (F.cL σ)]) (by linarith)
      hCd (fun j hj ↦ by rw [eq_of_mirror_eq_self hj, ← hj₀]) hCd0 (norm_nonneg _) n⟩

/-- **The bounds of item 6(a)** for `χ_L`, for every `μ`, `h`. -/
theorem chiDecay_chiLe_all : (F.opLatLAll k).ChiDecay (F.chiLe k) := by
  obtain ⟨c₁, hc₁⟩ := F.exists_chiL_self_bound k true
  obtain ⟨c₂, hc₂⟩ := F.exists_chiL_self_bound k false
  obtain ⟨c₃, hc₃⟩ := F.exists_chiL_opp_bound k true
  obtain ⟨c₄, hc₄⟩ := F.exists_chiL_opp_bound k false
  obtain ⟨d₁, hd₁⟩ := F.exists_chiL_diag_bound k true
  obtain ⟨d₂, hd₂⟩ := F.exists_chiL_diag_bound k false
  refine ⟨⟨max (max c₁ c₂) (max c₃ c₄), fun i j hij ↦ ?_⟩, ⟨max d₁ d₂, fun i ↦ ?_⟩⟩
  · obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
    obtain ⟨⟨τ, m⟩, rfl⟩ := encL.surjective j
    rw [chiLe_encL, opLatLAll_z, opLatLAll_z]
    dsimp only
    have hz := (F.BL σ).zOp_ne_zero k n
    have hv := (F.BL τ).zOp_ne_zero k m
    rw [le_div_iff₀ (by positivity), mul_comm]
    by_cases hστ : τ = σ
    · subst hστ
      have hnm : n ≠ m := fun h ↦ hij (by rw [h])
      cases τ
      · exact (hc₂ n m hnm).trans ((le_max_right _ _).trans (le_max_left _ _))
      · exact (hc₁ n m hnm).trans ((le_max_left _ _).trans (le_max_left _ _))
    · obtain rfl : τ = !σ := by cases σ <;> cases τ <;> simp_all
      cases σ
      · exact (hc₄ n m).trans ((le_max_right _ _).trans (le_max_right _ _))
      · exact (hc₃ n m).trans ((le_max_left _ _).trans (le_max_right _ _))
  · obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective i
    rw [chiLe_encL, opLatLAll_z, opLatLAll_β]
    dsimp only
    have hz := (F.BL σ).zOp_ne_zero k n
    rw [le_div_iff₀ (by positivity), mul_comm]
    cases σ
    · exact (hd₂ n).trans (le_max_right _ _)
    · exact (hd₁ n).trans (le_max_left _ _)

/-- **The bubble** decays as `-β/z'²` on both lattices, for every `μ`, `h`. -/
theorem bubbleDecay_bLe_all : (F.opLatLAll k).BubbleDecay (F.bLe k) := by
  have hb : ∀ σ, ∃ C, ∀ n, ‖(F.BL σ).zOp k n *
      (-((F.BL σ).zOp k n ^ 2 / F.β) * F.bL σ k n - 1)‖ ≤ C := fun σ ↦
    (F.BL σ).bdd_of_inv_zOp k (F.TbL k σ) (by unfold TbL dL; fun_prop (disch := simp)) (by
      filter_upwards [eventually_cofinite_ne ((-k - 1) / 2)] with j hj
      have h : mirror k j ≠ j := by unfold mirror; omega
      rw [(F.BL σ).zOp_of_ne k h, F.zL_mul_tL_eq k σ ((F.BL σ).z_ne_zero_of_ne k h)])
  obtain ⟨C₁, hC₁⟩ := hb true
  obtain ⟨C₂, hC₂⟩ := hb false
  refine ⟨fun j ↦ F.bL_ne_zero k _ _, max C₁ C₂, fun j ↦ ?_⟩
  obtain ⟨⟨σ, n⟩, rfl⟩ := encL.surjective j
  rw [OpLattice.tX, opLatLAll_z, opLatLAll_β, bLe_encL]
  dsimp only
  cases σ
  · exact (hC₂ n).trans (le_max_right _ _)
  · exact (hC₁ n).trans (le_max_left _ _)

end FParams

end HubbardAtom
