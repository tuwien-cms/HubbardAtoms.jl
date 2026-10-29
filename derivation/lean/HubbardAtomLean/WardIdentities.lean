import HubbardAtomLean.DividedDifferences
import HubbardAtomLean.DensityMagnetic
import HubbardAtomLean.Singlet
import HubbardAtomLean.FieldTransverse

/-!
# Ward identities: the functions of the divided differences as row sums

The function `f` of `DividedDifferences.lean` is the row sum of the susceptibility,
`-β⁻¹ ∑ₘ χ_nm = f(zₙ)`, by Step 2 of the vertex proofs (`hasSum_const`, identity (A6)):

* `Params.hasSum_chiR`: density and magnetic, for `U ≠ 0`, `μ ≠ 0`;
* `Params.hasSum_chiS`: singlet, for `b ≠ 0`, with `𝒯(t±)` from the weights (`γp_mul_Tpp`,
  `γm_mul_Tpp`);
* `FParams.hasSum_chiTUp`: transverse, for `U ≠ 0`, `μ ≠ 0`, `W ≠ 0`;
* `FParams.hasSum_chiL_updn`, `hasSum_chiL_upup`, `hasSum_chiL_dnup`, `hasSum_chiL_dndn`:
  longitudinal, `∑ₘ χ_στ(z_σₙ, z_τₘ) = -βf_στ(z_σₙ)` with `f_στ` the response of a `σ` pair to
  `n_τ`; the opposite-spin rows off the axes, by the bosonic values
  `S↓(z↑ₙ) = -(β/2)coth βh`, `S↓(-z↑ₙ) = (β/2)coth βμ` (`Bdn_S_Bup_z`, `Bdn_S_neg_Bup_z`) and
  `S₀(±a)` of each spin lattice from the weights (`Bdn_S₀_a`, …); for all `U`.
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- **Ward identity, density and magnetic**: `∑ₘ χ_r,nm = -βf_r(zₙ)`, from Step 2,
`∑ₘ P(zₙ)χ_r,nm = βN₂(zₙ)/(2εa)`. -/
theorem hasSum_chiR (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) (n : ℤ) :
    HasSum (fun m ↦ A.chiR k ε n m) (-A.β * A.fR k ε (A.z k n)) := by
  have hP := A.Pph_z_ne_zero k n
  have hQp := A.Qp_z_ne_zero k n
  have hQm := A.Qm_z_ne_zero k n
  convert (A.hasSum_step2_of k ε hε ha hμ n).div_const (A.Pph k (A.z k n)) using 1
  · funext m
    rw [eq_div_iff hP, mul_comm, A.Pph_mul_chiR k ε hμ n m, Rrow]
  · have hs := A.weights_sum'
    simp only [fR, N₂, Pph, ζ, qm, qp, ρ, Qp, Qm, q₀] at hQp hQm ⊢
    push_cast
    rw [show (A.p₂ : ℂ) = 1 - A.p₀ - 2 * A.p₁ by linear_combination hs]
    by_cases hk : k = 0
    · subst hk
      simp only [kron_self, (A.w_eq_zero_iff 0).mpr rfl, ne_eq, OfNat.ofNat_ne_zero,
        not_false_eq_true, zero_pow, zero_div, sub_zero] at hQp hQm ⊢
      generalize A.z 0 n = z at *
      generalize (A.a : ℂ) = a at *
      have h₁ : z + a ≠ 0 := fun h0 ↦ hQp (by rw [h0]; ring)
      have h₂ : z - a ≠ 0 := fun h0 ↦ hQm (by rw [h0]; ring)
      rw [show z ^ 2 - a ^ 2 = (z - a) * (z + a) by ring]
      rcases hε with rfl | rfl <;> field_simp <;> ring
    · rw [kron_of_ne hk]
      generalize A.z k n = z at *
      generalize (A.a : ℂ) = a at *
      generalize A.w k = W at *
      obtain ⟨c, rfl⟩ : ∃ c, W = 2 * c := ⟨W / 2, by ring⟩
      have eQp : (z + a) ^ 2 - (2 * c) ^ 2 / 4 = (z + a - c) * (z + a + c) := by ring
      have eQm : (z - a) ^ 2 - (2 * c) ^ 2 / 4 = (z - a - c) * (z - a + c) := by ring
      simp only [eQp, eQm] at hQp hQm ⊢
      obtain ⟨h₁, h₂⟩ := mul_ne_zero_iff.mp hQp
      obtain ⟨h₃, h₄⟩ := mul_ne_zero_iff.mp hQm
      simp only [zero_mul, zero_div, sub_zero, add_zero]
      rcases hε with rfl | rfl <;> field_simp <;> ring

/-- `γ₊𝒯(t₊) = -a(1 + ρ)S₀(a)/b`; at `b = 2a` both sides vanish (`γ₊ = 0`, and `μ = a`). -/
theorem γp_mul_Tpp (hb : A.b k ≠ 0) :
    A.γp k * A.Tpp k (A.a - A.b k / 2) = -(A.a * (1 + A.ρ) * A.S 0 A.a / A.b k) := by
  by_cases h2 : A.b k = 2 * A.a
  · have hμ : A.μ = A.a := by
      have := congrArg re h2
      simp [b, w] at this
      linarith
    have hp : (A.p₁ : ℂ) = A.p₂ := by
      rw [A.p₁_eq_p₂_mul, hμ, sub_self, mul_zero, Real.exp_zero, mul_one]
    rw [γp, h2, A.S₀_a, hp]
    ring
  · have hb₁ : 2 * (A.a : ℂ) - A.b k ≠ 0 := fun h0 ↦ h2 (by linear_combination -h0)
    have hb₁' : (A.a : ℂ) * 2 - A.b k ≠ 0 := by rwa [mul_comm]
    rw [A.Tpp_tp k h2, γp]
    field_simp
    ring

/-- `γ₋𝒯(t₋) = a(ρ - 1)S₀(-a)/b`; at `b = -2a` both sides vanish (`γ₋ = 0`, and `μ = -a`). -/
theorem γm_mul_Tpp (hb : A.b k ≠ 0) :
    A.γm k * A.Tpp k (-A.a - A.b k / 2) = A.a * (A.ρ - 1) * A.S 0 (-A.a) / A.b k := by
  by_cases h2 : A.b k = -2 * A.a
  · have hμ : A.μ = -A.a := by
      have := congrArg re h2
      simp [b, w] at this
      linarith
    have hp : (A.p₁ : ℂ) = A.p₀ := by
      rw [A.p₁_eq_p₀_mul, hμ, add_neg_cancel, mul_zero, Real.exp_zero, mul_one]
    rw [γm, h2, A.S₀_neg_a, hp]
    ring
  · have hb₂ : 2 * (A.a : ℂ) + A.b k ≠ 0 := fun h0 ↦ h2 (by linear_combination h0)
    have hb₂' : (A.a : ℂ) * 2 + A.b k ≠ 0 := by rwa [mul_comm]
    rw [A.Tpp_tm k h2, γm]
    field_simp

/-- **Ward identity, singlet**: `∑ₘ χ_s,nm = -βf_s(n)` for `b ≠ 0`, from Proposition "pp":
the diagonal-plus-mirror part gives `L_s(tₙ)`, and the rank-two part `γ±g±(tₙ)𝒯(t±)`. -/
theorem hasSum_chiS (hb : A.b k ≠ 0) (n : ℤ) :
    HasSum (fun m ↦ A.chiS k n m) (-A.β * A.fS k n) := by
  have h₁ := hasSum_single (f := fun m ↦ (A.Ls k (A.t k n) + -A.Bpp k n) / 2 * kron n m) n
    (fun m hm ↦ by simp only [kron_of_ne (Ne.symm hm), mul_zero])
  have h₂ := hasSum_single (f := fun m ↦ (A.Ls k (A.t k n) - -A.Bpp k n) / 2 *
    kron m (mirror k n)) (mirror k n) (fun m hm ↦ by simp only [kron_of_ne hm, mul_zero])
  have h₃ := (A.hasSum_gp k).mul_left (A.γp k / (A.t k n - A.tp k))
  have h₄ := (A.hasSum_gm k).mul_left (A.γm k / (A.t k n - A.tm k))
  convert ((h₁.add h₂).add h₃).add h₄ using 1
  · funext m
    rw [A.chiS_eq k hb, mirrorBlock, chiReg]
    ring
  · simp only [kron_self, mul_one]
    rw [show A.γp k / (A.t k n - A.tp k) * A.Tpp k (A.a - A.b k / 2) =
        A.γp k * A.Tpp k (A.a - A.b k / 2) / (A.t k n - A.tp k) by ring,
      show A.γm k / (A.t k n - A.tm k) * A.Tpp k (-A.a - A.b k / 2) =
        A.γm k * A.Tpp k (-A.a - A.b k / 2) / (A.t k n - A.tm k) by ring,
      A.γp_mul_Tpp k hb, A.γm_mul_Tpp k hb, A.S₀_a, A.S₀_neg_a]
    have hq₁ : (A.p₁ : ℂ) + A.p₂ ≠ 0 := by exact_mod_cast (add_pos A.p₁_pos A.p₂_pos).ne'
    have hq₀ : (A.p₀ : ℂ) + A.p₁ ≠ 0 := by exact_mod_cast (add_pos A.p₀_pos A.p₁_pos).ne'
    have h₅ := A.x_sub_ofReal_ne_zero n A.a
    have h₆ := A.x_add_ofReal_ne_zero n A.a
    have h₇ := A.x_sub_ofReal_ne_zero (mirror k n) A.a
    have h₈ := A.x_add_ofReal_ne_zero (mirror k n) A.a
    have hs := A.weights_sum'
    rw [fS, Ls, A.calP_eq_mul, A.t_sub_tp, A.t_sub_tm, As, G, G, ρ]
    simp only [A.x_mirror, t] at h₇ h₈ ⊢
    push_cast
    rw [show (A.p₂ : ℂ) = 1 - A.p₀ - 2 * A.p₁ by linear_combination hs] at hq₁ ⊢
    generalize A.x n = x at *
    generalize A.b k = b at *
    generalize (A.a : ℂ) = a at *
    rw [show x ^ 2 - a ^ 2 = (x - a) * (x + a) by ring,
      show (b - x) ^ 2 - a ^ 2 = (b - x - a) * (b - x + a) by ring]
    field_simp
    ring

end Params

namespace FParams

variable (F : FParams) (k : ℤ)

set_option maxRecDepth 10000 in
-- `field_simp` on the polynomials of the transverse input
/-- **Ward identity, transverse**: `∑ₘ χ_T↑,nm = -βf_T(zₙ)`, from Step 2,
`∑ₘ P(zₙ)χ_T↑,nm = -βN₂(zₙ)/(2a)`. -/
theorem hasSum_chiTUp (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hW : F.WT k ≠ 0) (n : ℤ) :
    HasSum (fun m ↦ F.chiTUp k n m) (-F.β * F.fT k (F.zT k n)) := by
  have hP := F.PT_zT_ne_zero k n
  have h₁ : HasSum (fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l)
      (-F.β * F.HT k (F.zT k n)) := by
    convert hasSum_single (f := fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero]) using 1
    rw [kron_self, mul_one]
  have h₂ : HasSum (fun l ↦ F.RrowT k n l) (-(F.β * F.N₂T k (F.zT k n) / (2 * F.a))) := by
    convert h₁.add (F.hasSum_constT k ha hW hμ n) using 1
    · funext l
      rw [RrowT]
      ring
    · rw [← F.identity_A6T k ha hW]
      ring
  convert h₂.div_const (F.PT k (F.zT k n)) using 1
  · funext m
    rw [eq_div_iff hP, mul_comm, F.PT_mul_chiTUp k hμ hW n m, RrowT]
  · have hs := F.weights_sum'
    have h₁ := F.lin₁ n
    have h₂ := F.lin₂ n
    have h₃ := F.lin₃ k n
    have h₄ := F.lin₄ k n
    simp only [fT, Gc, N₂T, F.PT_zT, PTf, ζT, q₀T, rUp, rDn, ρ, mf]
    rw [WT_eq] at hW ⊢
    simp only [zT, Params.z]
    push_cast
    rw [show (F.p₂ : ℂ) = 1 - F.p₀ - F.pUp - F.pDn by linear_combination hs]
    generalize (F.lat F.μ).x n = x at *
    generalize (F.lat F.μ).w k = w at *
    generalize (F.a : ℂ) = a at *
    generalize (F.h : ℂ) = h at *
    obtain ⟨c, rfl⟩ : ∃ c, w = 2 * c := ⟨w / 2, by ring⟩
    simp only [mul_div_cancel_left₀ _ (two_ne_zero' ℂ)] at *
    rw [show x + c - (2 * c - 2 * h) / 2 = x + h by ring,
      show x + c + (2 * c - 2 * h) / 2 = x + 2 * c - h by ring,
      show (x + h) ^ 2 - a ^ 2 = (x - a + h) * (x + a + h) by ring,
      show (x + 2 * c - h) ^ 2 - a ^ 2 = (x + 2 * c - a - h) * (x + 2 * c + a - h) by ring]
    have hch : c - h ≠ 0 := fun h0 ↦ hW (by linear_combination 2 * h0)
    generalize hA₁ : x + 2 * c + a - h = A₁ at *
    generalize hA₂ : x + 2 * c - a - h = A₂ at *
    field_simp
    subst hA₁ hA₂
    ring

/-! ### The thermal values on the spin lattices -/

/-- `S₀(a)` on the lattice of spin `↓` (centre `μ - h`): `β(p₂ - p↑)/(2(p₂ + p↑))`. -/
theorem Bdn_S₀_a : F.Bdn.S 0 F.a = F.β * (F.p₂ - F.pUp) / (2 * (F.p₂ + F.pUp)) :=
  F.Bdn.S₀_of_exp F.pUp_pos F.p₂_pos (by rw [F.pUp_eq_p₂_mul]; simp only [lat_β, lat_μ]; ring_nf)

/-- `S₀(-a)` on the lattice of spin `↓`: `β(p↓ - p₀)/(2(p₀ + p↓))`. -/
theorem Bdn_S₀_neg_a : F.Bdn.S 0 (-F.a) = F.β * (F.pDn - F.p₀) / (2 * (F.p₀ + F.pDn)) := by
  have := F.Bdn.S₀_of_exp (x := -F.a) F.p₀_pos F.pDn_pos (by
    rw [F.pDn_eq_p₀_mul, mul_assoc, ← Real.exp_add]
    simp only [lat_β, lat_μ]
    rw [show F.β * (F.a + F.μ - F.h) + F.β * (-F.a - (F.μ - F.h)) = 0 by ring, Real.exp_zero,
      mul_one])
  push_cast at this
  rw [this, add_comm (F.pDn : ℂ)]
  rfl

/-- `S₀(a)` on the lattice of spin `↑` (centre `μ + h`): `β(p₂ - p↓)/(2(p₂ + p↓))`. -/
theorem Bup_S₀_a : F.Bup.S 0 F.a = F.β * (F.p₂ - F.pDn) / (2 * (F.p₂ + F.pDn)) :=
  F.Bup.S₀_of_exp F.pDn_pos F.p₂_pos (by rw [F.pDn_eq_p₂_mul]; simp only [lat_β, lat_μ]; ring_nf)

/-- `S₀(-a)` on the lattice of spin `↑`: `β(p↑ - p₀)/(2(p₀ + p↑))`. -/
theorem Bup_S₀_neg_a : F.Bup.S 0 (-F.a) = F.β * (F.pUp - F.p₀) / (2 * (F.p₀ + F.pUp)) := by
  have := F.Bup.S₀_of_exp (x := -F.a) F.p₀_pos F.pUp_pos (by
    rw [F.pUp_eq_p₀_mul, mul_assoc, ← Real.exp_add]
    simp only [lat_β, lat_μ]
    rw [show F.β * (F.a + F.μ + F.h) + F.β * (-F.a - (F.μ + F.h)) = 0 by ring, Real.exp_zero,
      mul_one])
  push_cast at this
  rw [this, add_comm (F.pUp : ℂ)]
  rfl

/-- The thermal argument at the other spin's lattice: `β(z↑ₙ - (μ - h))/2 = βh + Kπi/2` with
`K = 2n + 1 + k`, whose parity is opposite to that of `k`. -/
private theorem arg_Bup_z (k n : ℤ) :
    (F.β : ℂ) * (F.Bup.z k n - ((F.μ - F.h : ℝ) : ℂ)) / 2 =
      F.β * F.h + ((2 * n + 1 + k : ℤ) : ℂ) * (Real.pi / 2) * I := by
  have hβ : (F.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.β_pos.ne'
  rw [F.Bup.z_eq]
  simp only [lat_μ, lat_β]
  push_cast
  field_simp
  ring

/-- `coth` or `tanh` of `x + Kπi/2`, as chosen by the parity of `k`, is `coth x` when `K` and `k`
have opposite parities. -/
private theorem thermal_shift {k K : ℤ} (hK : Even K ↔ ¬Even k) (x : ℂ) :
    (if Even k then tanh (x + (K : ℂ) * (Real.pi / 2) * I)
      else coth (x + (K : ℂ) * (Real.pi / 2) * I)) = coth x := by
  by_cases hk : Even k
  · rw [ite_eq_left hk, tanh_add_int_mul, ite_eq_right (fun h ↦ hK.mp h hk)]
  · rw [ite_eq_right hk, coth_add_int_mul, ite_eq_left (hK.mpr hk)]

/-- `2n + 1 + k` and `k` have opposite parities. -/
private theorem parity (k n : ℤ) : Even (2 * n + 1 + k) ↔ ¬Even k := by
  rw [Int.even_iff, Int.even_iff]
  omega

/-- The bosonic sum `S↓(z↑ₙ) = PV ∑ₘ 1/(z↓ₘ - z↑ₙ) = -(β/2)coth βh = -β(p↑ + p↓)/(2m)`. -/
theorem Bdn_S_Bup_z (hh : F.h ≠ 0) (k n : ℤ) :
    F.Bdn.S k (F.Bup.z k n) = -(F.β * (F.pUp + F.pDn) / (2 * F.mf)) := by
  have he : F.pUp = F.pDn * Real.exp (2 * (F.β * F.h)) := by
    rw [F.pUp_eq_p₀_mul, F.pDn_eq_p₀_mul, mul_assoc, ← Real.exp_add]
    ring_nf
  rw [Params.S, show F.Bdn.μ = F.μ - F.h from rfl, show F.Bdn.β = F.β from rfl, F.arg_Bup_z,
    thermal_shift (parity k n), coth_eq_exp, mf, he]
  have hE : Real.exp (2 * (F.β * F.h)) ≠ 1 := by
    rw [Ne, Real.exp_eq_one_iff]
    exact mul_ne_zero two_ne_zero (mul_ne_zero F.β_pos.ne' hh)
  have hd : (F.pDn : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.pDn_pos.ne'
  have hE' : cexp (2 * (F.β * F.h)) - 1 ≠ 0 := by
    rw [sub_ne_zero]
    exact_mod_cast hE
  push_cast
  generalize cexp (2 * (F.β * F.h)) = E at *
  have hE'' : (F.pDn : ℂ) * E - F.pDn ≠ 0 := by
    rw [show (F.pDn : ℂ) * E - F.pDn = F.pDn * (E - 1) by ring]
    exact mul_ne_zero hd hE'
  field_simp

/-- The bosonic sum `S↓(-z↑ₙ) = PV ∑ₘ 1/(z↓ₘ + z↑ₙ) = (β/2)coth βμ = β(p₀ + p₂)/(2ρ)`. -/
theorem Bdn_S_neg_Bup_z (hμ : F.μ ≠ 0) (k n : ℤ) :
    F.Bdn.S k (-F.Bup.z k n) = F.β * (F.p₀ + F.p₂) / (2 * F.ρ) := by
  have hρ := F.rho_ne_zeroT hμ
  have h : F.Bdn.S k (-F.Bup.z k n) = (F.lat F.μ).S k (-F.zT k n) := by
    simp only [Params.S, Params.z, Params.x, Params.w, Params.ν, Params.ω, zT, lat_β, lat_μ]
    push_cast
    ring_nf
  have h2 := F.rho_mul_S_neg_zT k hμ n
  rw [cbarT] at h2
  push_cast at h2
  rw [h, eq_div_iff (mul_ne_zero two_ne_zero hρ)]
  linear_combination 2 * h2

/-! ### The longitudinal rows -/

theorem mf_ne_zero (hh : F.h ≠ 0) : (F.mf : ℂ) ≠ 0 := by
  have := F.shiba.rho_ne_zeroT hh
  rwa [shiba_ρ] at this

/-- `N(z, v) = N₀(v) + z²N₂(v)` of the longitudinal input is a cubic in `v`, with `q = q₀`,
`δ = δ_k0`, `ζ = ζ_L`; its `v³` coefficient `-4a²r` cancels the `1/v` tails of the kernels. -/
theorem NL_cubic (r : ℝ) (z v : ℂ) :
    F.N₀L k r v + z ^ 2 * F.N₂L k r v =
      (4 * F.a ^ 4 * kron k 0 * F.ζL + 8 * F.a ^ 3 * F.q₀L k -
          2 * F.a ^ 2 * kron k 0 * F.q₀L k * F.ζL - F.a ^ 2 * kron k 0 * z ^ 2 * F.ζL -
          2 * F.a * F.q₀L k ^ 2 - 2 * F.a * F.q₀L k * z ^ 2 - kron k 0 * F.q₀L k ^ 2 * F.ζL) +
        -4 * F.a ^ 2 * r * (-4 * F.a ^ 2 + 2 * F.q₀L k + z ^ 2) * v +
        (-(F.a ^ 2 * kron k 0 * F.ζL) - 2 * F.a * F.q₀L k - 2 * F.a * z ^ 2 +
          kron k 0 * z ^ 2 * F.ζL) * v ^ 2 +
        -4 * F.a ^ 2 * r * v ^ 3 := by
  simp only [N₀L, N₂L, HL, PL, q₀L]
  ring

theorem PL_Bup_ne (j : ℤ) : F.PL k (F.Bup.z k j) ≠ 0 := by
  rw [F.PL_up]
  have := F.Bup.x_add_ofReal_ne_zero j F.a
  have := F.Bup.x_sub_ofReal_ne_zero j F.a
  have := F.Bup.x_add_ofReal_ne_zero (j + k) F.a
  have := F.Bup.x_sub_ofReal_ne_zero (j + k) F.a
  rw [← F.Bup.x_add_w] at *
  exact mul_ne_zero (mul_ne_zero ‹_› ‹_›) (mul_ne_zero ‹_› ‹_›)

theorem PL_Bdn_ne (j : ℤ) : F.PL k (F.Bdn.z k j) ≠ 0 := by
  rw [F.PL_dn]
  have := F.Bdn.x_add_ofReal_ne_zero j F.a
  have := F.Bdn.x_sub_ofReal_ne_zero j F.a
  have := F.Bdn.x_add_ofReal_ne_zero (j + k) F.a
  have := F.Bdn.x_sub_ofReal_ne_zero (j + k) F.a
  rw [← F.Bdn.x_add_w] at *
  exact mul_ne_zero (mul_ne_zero ‹_› ‹_›) (mul_ne_zero ‹_› ‹_›)

set_option maxRecDepth 10000 in
-- `field_simp` on the polynomials of the longitudinal input
/-- `hasSum_chiL_updn` for `U ≠ 0`. The kernels are summed with
the bosonic values `S↓(±z↑ₙ)`, the numerator `N/P` over the lattice of spin `↓` by
`hasSum_cubic`. -/
private theorem hasSum_chiL_updn_ne (ha : (F.a : ℂ) ≠ 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0)
    (n : ℤ) :
    HasSum (fun m ↦ F.chiL true false k n m) (-F.β * F.fLUpDn k (F.Bup.z k n)) := by
  have hy : ∀ m, F.Bdn.z k m ≠ -F.Bup.z k n := fun m h0 ↦
    F.zUp_add_zDn_ne k hμ n m (by rw [h0]; ring)
  have hx : ∀ m, F.Bdn.z k m ≠ F.Bup.z k n := fun m h0 ↦
    F.zUp_sub_zDn_ne k hh n m (by rw [h0]; ring)
  have hPn := F.PL_Bup_ne k n
  have h₁ := F.Bdn.hasSum_cubic k ha
    (4 * F.a ^ 4 * kron k 0 * F.ζL + 8 * F.a ^ 3 * F.q₀L k -
      2 * F.a ^ 2 * kron k 0 * F.q₀L k * F.ζL - F.a ^ 2 * kron k 0 * F.Bup.z k n ^ 2 * F.ζL -
      2 * F.a * F.q₀L k ^ 2 - 2 * F.a * F.q₀L k * F.Bup.z k n ^ 2 - kron k 0 * F.q₀L k ^ 2 * F.ζL)
    (-4 * F.a ^ 2 * F.rDn * (-4 * F.a ^ 2 + 2 * F.q₀L k + F.Bup.z k n ^ 2))
    (-(F.a ^ 2 * kron k 0 * F.ζL) - 2 * F.a * F.q₀L k - 2 * F.a * F.Bup.z k n ^ 2 +
      kron k 0 * F.Bup.z k n ^ 2 * F.ζL)
    (-4 * F.a ^ 2 * F.rDn) hy
  have h₂ := (F.Bdn.hasSum_sub_inv k hx hy).mul_left (F.lamL * F.mf)
  convert (h₁.add h₂).div_const (F.PL k (F.Bup.z k n)) using 1
  · funext m
    have hPm := F.PL_Bdn_ne k m
    have hp : F.Bdn.z k m + F.Bup.z k n ≠ 0 := fun h0 ↦ hy m (by linear_combination h0)
    have hd : F.Bdn.z k m - F.Bup.z k n ≠ 0 := sub_ne_zero.mpr (hx m)
    have hp' := F.zUp_add_zDn_ne k hμ n m
    have hd' := F.zUp_sub_zDn_ne k hh n m
    rw [eq_div_iff hPn, mul_comm, F.PL_mul_chiL_updn k hμ hh n m, F.NL_cubic,
      show F.Bdn.Pph k (F.Bdn.z k m) = F.PL k (F.Bdn.z k m) from rfl]
    simp only [lamL, rDn, sub_neg_eq_add]
    push_cast
    field_simp
    ring
  · have hρ := F.rho_ne_zeroT hμ
    have hm := F.mf_ne_zero hh
    have hs := F.weights_sum'
    have h₂₁ : (F.p₂ : ℂ) + F.pUp ≠ 0 := by exact_mod_cast (add_pos F.p₂_pos F.pUp_pos).ne'
    have h₀₁ : (F.p₀ : ℂ) + F.pDn ≠ 0 := by exact_mod_cast (add_pos F.p₀_pos F.pDn_pos).ne'
    have hq : F.q₀L k ≠ 0 := F.Bdn.q₀_ne_zero k ha
    rw [show F.Bdn.ℓ₀ k = (F.Bdn.Sp - F.Bdn.Sm) / (4 * F.a * F.q₀L k) +
        (F.Bdn.Tp k + F.Bdn.Tm k) / (4 * F.q₀L k) from rfl,
      show F.Bdn.ℓ₁ k = (F.Bdn.Tm k - F.Bdn.Tp k) / (4 * F.a) from rfl,
      show F.Bdn.ℓ₂ k = (F.Bdn.Sm - F.Bdn.Sp) / (4 * F.a) + (F.Bdn.Tp k + F.Bdn.Tm k) / 4 from rfl,
      show F.Bdn.ℓ₃ k = (F.Bdn.Sp + F.Bdn.Sm) / 2 +
        (F.q₀L k - 2 * F.a ^ 2) * (F.Bdn.Tp k - F.Bdn.Tm k) / (4 * F.a) from rfl,
      show F.Bdn.Tp k = kron k 0 * (F.Bdn.Sp ^ 2 - F.β ^ 2 / 4) from rfl,
      show F.Bdn.Tm k = kron k 0 * (F.Bdn.Sm ^ 2 - F.β ^ 2 / 4) from rfl,
      show F.Bdn.Sp = F.Bdn.S 0 (-F.a) from rfl, show F.Bdn.Sm = F.Bdn.S 0 F.a from rfl,
      F.Bdn_S₀_a, F.Bdn_S₀_neg_a, F.Bdn_S_Bup_z hh, F.Bdn_S_neg_Bup_z hμ]
    clear h₁ h₂ hy hx
    simp only [ρ, mf] at hρ hm
    simp only [fLUpDn, lamL, rDn, ζL, ζUp, ρ, mf]
    push_cast at hρ hm ⊢
    rw [show (F.p₂ : ℂ) = 1 - F.p₀ - F.pUp - F.pDn by linear_combination hs] at h₂₁ hρ ⊢
    by_cases hk : k = 0
    · subst hk
      have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
      simp only [kron_self, q₀L, PL, hw] at hPn hq ⊢
      generalize F.Bup.z 0 n = z at *
      generalize (F.a : ℂ) = a at *
      have h₁ : z - a ≠ 0 := fun h0 ↦ hPn (by rw [show z = a by linear_combination h0]; ring)
      have h₂ : z + a ≠ 0 := fun h0 ↦ hPn (by rw [show z = -a by linear_combination h0]; ring)
      rw [show z ^ 2 - a ^ 2 = (z - a) * (z + a) by ring]
      field_simp
      ring
    · simp only [kron_of_ne hk, zero_mul, zero_div, neg_zero, mul_zero, sub_zero, add_zero]
      generalize F.Bup.z k n = z at *
      generalize F.q₀L k = q at *
      generalize F.PL k z = Pz at *
      field_simp
      ring

set_option maxRecDepth 10000 in
-- `field_simp` on the polynomials of the longitudinal input
/-- `hasSum_chiL_upup` for `U ≠ 0`; at `k = 0` the rank-one term sums to `(S₀(a) - S₀(-a))/(2a)`
on the lattice of spin `↑`. -/
private theorem hasSum_chiL_upup_ne (ha : (F.a : ℂ) ≠ 0) (n : ℤ) :
    HasSum (fun m ↦ F.chiL true true k n m) (-F.β * F.fL k F.rUp (F.Bup.z k n)) := by
  have hPn := F.PL_Bup_ne k n
  have hy : ∀ m, F.Bup.z k m ≠ ((F.μ + F.h + 1 : ℝ) : ℂ) := fun m h0 ↦ by
    have := congrArg re h0
    simp [Params.z, Params.x, Params.w] at this
  have h₁ := hasSum_single (f := fun m ↦ -F.β * F.HL k F.rUp (F.Bup.z k n) * kron n m) n
    (fun m hm ↦ by simp only [kron_of_ne (Ne.symm hm), mul_zero])
  have h₂ := (F.Bup.hasSum_cubic k ha (-F.a ^ 2) 0 1 0 hy).mul_left
    (kron k 0 * F.ζUp * (F.Bup.z k n ^ 2 - F.a ^ 2))
  convert (h₁.add h₂).div_const (F.PL k (F.Bup.z k n)) using 1
  · funext m
    rw [eq_div_iff hPn, mul_comm, F.PL_mul_chiL_upup k n m,
      show F.Bup.Pph k (F.Bup.z k m) = F.PL k (F.Bup.z k m) from rfl]
    ring
  · clear h₁ h₂ hy
    have hs := F.weights_sum'
    have h₂₁ : (F.p₂ : ℂ) + F.pDn ≠ 0 := by exact_mod_cast (add_pos F.p₂_pos F.pDn_pos).ne'
    have h₀₁ : (F.p₀ : ℂ) + F.pUp ≠ 0 := by exact_mod_cast (add_pos F.p₀_pos F.pUp_pos).ne'
    simp only [kron_self, mul_one]
    by_cases hk : k = 0
    · subst hk
      have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
      rw [show -F.a ^ 2 * F.Bup.ℓ₀ 0 + 0 * F.Bup.ℓ₁ 0 + 1 * F.Bup.ℓ₂ 0 +
          0 * (F.Bup.ℓ₃ 0 - F.Bup.S 0 ((F.μ + F.h + 1 : ℝ) : ℂ)) =
          (F.Bup.S 0 F.a - F.Bup.S 0 (-F.a)) / (2 * F.a) by
        simp only [Params.ℓ₀, Params.ℓ₂, Params.Sp, Params.Sm, Params.q₀,
          show F.Bup.w 0 = 0 from hw, lat_a]
        field_simp
        ring]
      rw [F.Bup_S₀_a, F.Bup_S₀_neg_a]
      simp only [fL, HL, kron_self, q₀L, PL, hw, ζL, ζUp, rUp, ρ, mf] at hPn ⊢
      push_cast
      rw [show (F.p₂ : ℂ) = 1 - F.p₀ - F.pUp - F.pDn by linear_combination hs] at h₂₁ ⊢
      generalize F.Bup.z 0 n = z at *
      generalize (F.a : ℂ) = a at *
      have h₁ : z - a ≠ 0 := fun h0 ↦ hPn (by rw [show z = a by linear_combination h0]; ring)
      have h₂ : z + a ≠ 0 := fun h0 ↦ hPn (by rw [show z = -a by linear_combination h0]; ring)
      rw [show z ^ 2 - a ^ 2 = (z - a) * (z + a) by ring]
      field_simp
      ring
    · simp only [kron_of_ne hk, zero_mul, add_zero, fL, HL, PL, q₀L] at hPn ⊢
      generalize F.Bup.z k n = z at *
      generalize (F.a : ℂ) = a at *
      generalize F.wL k = W at *
      obtain ⟨c, rfl⟩ : ∃ c, W = 2 * c := ⟨W / 2, by ring⟩
      have eQp : (z + a) ^ 2 - (2 * c) ^ 2 / 4 = (z + a - c) * (z + a + c) := by ring
      have eQm : (z - a) ^ 2 - (2 * c) ^ 2 / 4 = (z - a - c) * (z - a + c) := by ring
      simp only [eQp, eQm] at hPn ⊢
      obtain ⟨h₁, h₂⟩ := mul_ne_zero_iff.mp hPn
      obtain ⟨h₃, h₄⟩ := mul_ne_zero_iff.mp h₁
      obtain ⟨h₅, h₆⟩ := mul_ne_zero_iff.mp h₂
      simp only [zero_div, sub_zero]
      field_simp
      ring

/-- `hasSum_chiL_upup` at `U = 0`: the rank-one term vanishes (`ζ↑ = 0`). -/
private theorem hasSum_chiL_upup_a0 (ha : (F.a : ℂ) = 0) (n : ℤ) :
    HasSum (fun m ↦ F.chiL true true k n m) (-F.β * F.fL k F.rUp (F.Bup.z k n)) := by
  have hPn := F.PL_Bup_ne k n
  have hζ : F.ζUp = 0 := by rw [ζUp, ha]; ring
  have hζL : F.ζL = 0 := by rw [ζL, ha]; ring
  convert (hasSum_single (f := fun m ↦ -F.β * F.HL k F.rUp (F.Bup.z k n) * kron n m) n
    (fun m hm ↦ by simp only [kron_of_ne (Ne.symm hm), mul_zero])).div_const
    (F.PL k (F.Bup.z k n)) using 1
  · funext m
    rw [eq_div_iff hPn, mul_comm, F.PL_mul_chiL_upup k n m, hζ]
    ring
  · rw [kron_self, mul_one]
    simp only [fL, HL, PL, q₀L, hζL, ha] at hPn ⊢
    simp only [sub_zero, add_zero, mul_zero, zero_mul, div_zero, ne_eq, zero_pow,
      OfNat.ofNat_ne_zero, not_false_eq_true, zero_sub, ← sub_eq_add_neg] at hPn ⊢
    generalize F.Bup.z k n ^ 2 - F.wL k ^ 2 / 4 = D at *
    have h := left_ne_zero_of_mul hPn
    field_simp
    ring

/-- `hasSum_chiL_updn` at `U = 0`: `χ↑↓ = 0` and `f↑↓ = 0`. -/
private theorem hasSum_chiL_updn_a0 (ha : (F.a : ℂ) = 0) (hμ : F.μ ≠ 0) (hh : F.h ≠ 0)
    (n : ℤ) :
    HasSum (fun m ↦ F.chiL true false k n m) (-F.β * F.fLUpDn k (F.Bup.z k n)) := by
  have hz : ∀ m, F.chiL true false k n m = 0 := fun m ↦ by
    have := F.PL_mul_chiL_updn k hμ hh n m
    simp only [lamL, N₀L, N₂L, HL, ζL, ha] at this
    simp only [zero_pow, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, mul_zero, zero_mul,
      sub_zero, zero_div, add_zero, sub_self] at this
    exact (mul_eq_zero.mp this).resolve_left (F.PL_Bup_ne k n)
  have hf : F.fLUpDn k (F.Bup.z k n) = 0 := by
    simp [fLUpDn, ζUp, ha]
  simp only [hz, hf, mul_zero]
  exact hasSum_zero

/-- **Ward identity, longitudinal, opposite spins**: off the axes,
`∑ₘ χ↑↓(z↑ₙ, z↓ₘ) = -βf↑↓(z↑ₙ)`, the response of an `↑` pair to `n↓` (both sides vanish at
`U = 0`). -/
theorem hasSum_chiL_updn (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (n : ℤ) :
    HasSum (fun m ↦ F.chiL true false k n m) (-F.β * F.fLUpDn k (F.Bup.z k n)) := by
  by_cases ha : (F.a : ℂ) = 0
  · exact F.hasSum_chiL_updn_a0 k ha hμ hh n
  · exact F.hasSum_chiL_updn_ne k ha hμ hh n

/-- **Ward identity, longitudinal, equal spins**: `∑ₘ χ↑↑(z↑ₙ, z↑ₘ) = -βf↑↑(z↑ₙ)`, for all `μ`,
`h`. -/
theorem hasSum_chiL_upup (n : ℤ) :
    HasSum (fun m ↦ F.chiL true true k n m) (-F.β * F.fL k F.rUp (F.Bup.z k n)) := by
  by_cases ha : (F.a : ℂ) = 0
  · exact F.hasSum_chiL_upup_a0 k ha n
  · exact F.hasSum_chiL_upup_ne k ha n

/-- **Ward identity, longitudinal, rows `↓`**, by the spin flip: `∑ₘ χ↓↑ = -βf↓↑(z↓ₙ)` off the
axes, with `f↓↑` the `f↑↓` of the flipped atom. -/
theorem hasSum_chiL_dnup (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (n : ℤ) :
    HasSum (fun m ↦ F.chiL false true k n m) (-F.β * F.flip.fLUpDn k (F.Bdn.z k n)) := by
  simp_rw [F.chiL_flip]
  simpa [flip_Bup] using F.flip.hasSum_chiL_updn k hμ (neg_ne_zero.mpr hh) n

/-- `∑ₘ χ↓↓ = -βf↓↓(z↓ₙ)`, by the spin flip. -/
theorem hasSum_chiL_dndn (n : ℤ) :
    HasSum (fun m ↦ F.chiL false false k n m) (-F.β * F.fL k F.rDn (F.Bdn.z k n)) := by
  simp_rw [F.chiL_flip]
  simpa [flip_Bup, flip_fL, flip_rUp] using F.flip.hasSum_chiL_upup k n

end FParams

end HubbardAtom
