import HubbardAtomLean.PHInput
import HubbardAtomLean.LehmannInput
import HubbardAtomLean.FieldTSums
import HubbardAtomLean.FieldLInput

/-!
# Divided differences of the susceptibilities

Off the diagonal, each susceptibility is a divided difference of one function of one frequency:

* `Params.chiR_divided_difference`: density and magnetic (`ε = ±1`), for `n ≠ m`,
  `(zₙ² - zₘ²)χ_r,nm = εU[f_r(zₙ) - f_r(zₘ)]`, with
  `f_r(z) = q₋/Q₊(z) + q₊/Q₋(z) - δ_k0 εζ_r/(2a(z² - a²))`;
  for `k ≠ 0`, `f_r(zₙ) = [G(xₙ) - G(xₙ + w)]/w` (`fR_eq_G`).
* `Params.chiS_divided_difference`: singlet, for `b ≠ 0` and `n ≠ m` (on the mirror both sides
  vanish),
  `(tₙ - tₘ)χ_s,nm = -U[f_s(n) - f_s(m)]`, with `f_s(n) = -[G(xₙ) + G(x_n̄)]/(2b)`.
* `FParams.chiTUp_divided_difference`: transverse, for `W ≠ 0` and `n ≠ m`,
  `(zₙ² - zₘ²)χ_T↑,nm = -U[f_T(zₙ) - f_T(zₘ)]`, with `f_T(zₙ) = [G↑(ν_n) - G↓(ν_n + ω)]/W`
  (`fT_zT`).
* `FParams.chiL_updn_divided_difference`, `chiL_upup_divided_difference`: longitudinal, each row
  on the lattice of its own spin, `z↑ = z_{μ+h}`, `z↓ = z_{μ-h}`:
  `(z↑ₙ² - z↓ₘ²)χ↑↓,nm = U[f↑↑(z↑ₙ) - f↓↓(z↓ₘ)]` off the axes, and
  `(z↑ₙ² - z↑ₘ²)χ↑↑,nm = U[f↑↓(z↑ₙ) - f↑↓(z↑ₘ)]` for `n ≠ m`, where `f_σσ` is `f_r` of the density
  channel with `r_σ` and `ζ_L` (`fL_up_eq_G`, `fL_dn_eq_G`: `[G_σ(ν) - G_σ(ν + ω)]/w` for
  `k ≠ 0`) and
  `f↑↓ = -δ_k0 ζ↑/(2a(z² - a²))`; the rows of spin `↓` by the spin flip
  (`chiL_dnup_divided_difference`, `chiL_dndn_divided_difference`).
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `f_r(z) = q₋/Q₊(z) + q₊/Q₋(z) - δ_k0 εζ_r/(2a(z² - a²))`. -/
noncomputable def fR (z : ℂ) : ℂ :=
  A.qm / A.Qp k z + A.qp / A.Qm k z - kron k 0 * ε * A.ζ ε / (2 * A.a * (z ^ 2 - A.a ^ 2))

/-- For `k ≠ 0`, `f_r(zₙ) = [G(xₙ) - G(xₙ + w)]/w`, the same for `r = d, m`. -/
theorem fR_eq_G (hk : k ≠ 0) (n : ℤ) :
    A.fR k ε (A.z k n) = (A.G (A.x n) - A.G (A.x n + A.w k)) / A.w k := by
  have hw := mt (A.w_eq_zero_iff k).mp hk
  have h₁ := A.Qm_z_ne_zero k n
  have h₂ := A.Qp_z_ne_zero k n
  rw [Qm_z, ← A.x_add_w] at h₁
  rw [Qp_z, ← A.x_add_w] at h₂
  have := left_ne_zero_of_mul h₁
  have := right_ne_zero_of_mul h₁
  have := left_ne_zero_of_mul h₂
  have := right_ne_zero_of_mul h₂
  have hm : A.x n ^ 2 - A.a ^ 2 ≠ 0 := by
    rw [show A.x n ^ 2 - A.a ^ 2 = (A.x n - A.a) * (A.x n + A.a) by ring]
    exact mul_ne_zero ‹_› ‹_›
  have hm' : (A.x n + A.w k) ^ 2 - A.a ^ 2 ≠ 0 := by
    rw [show (A.x n + A.w k) ^ 2 - A.a ^ 2 = (A.x n + A.w k - A.a) * (A.x n + A.w k + A.a) by
      ring]
    exact mul_ne_zero ‹_› ‹_›
  have hs := A.weights_sum'
  rw [fR, kron_of_ne hk, Qm_z, Qp_z, G, G, ← A.x_add_w, qm, qp, ρ]
  push_cast
  rw [show (A.p₂ : ℂ) = 1 - A.p₀ - 2 * A.p₁ by linear_combination hs]
  field_simp
  ring

/-- At `μ = 0`, `f_r` is even (`q₊ = q₋`): on the mirror `zₘ = -zₙ` both sides of
`chiR_divided_difference` vanish. -/
theorem fR_neg (hμ : A.μ = 0) (z : ℂ) : A.fR k ε (-z) = A.fR k ε z := by
  rw [fR, fR, Qp, Qm, Qp, Qm, qp, qm, p₂_eq_p₀_of_mu hμ, neg_sq,
    show (-z + A.a) ^ 2 = (z - A.a) ^ 2 by ring, show (-z - A.a) ^ 2 = (z + A.a) ^ 2 by ring]
  ring

set_option maxRecDepth 10000 in
-- `field_simp` on the polynomials of Eq. (N)
/-- **Divided differences, density and magnetic**: for `n ≠ m`,
`(zₙ² - zₘ²)χ_r,nm = εU[f_r(zₙ) - f_r(zₘ)]`, by Proposition "structure"; on the mirror at `μ = 0`
both sides vanish (`fR_neg`). -/
theorem chiR_divided_difference (hε : ε = 1 ∨ ε = -1) {n m : ℤ} (hnm : n ≠ m) :
    (A.z k n ^ 2 - A.z k m ^ 2) * A.chiR k ε n m =
      ε * A.U * (A.fR k ε (A.z k n) - A.fR k ε (A.z k m)) := by
  by_cases hzv : A.z k n + A.z k m = 0
  · have hμ : A.μ = 0 := by
      have h := congrArg re hzv
      simp only [A.z_eq, add_re, ofReal_re, mul_re, ofReal_im, I_re, I_im, zero_re] at h
      linarith
    rw [show A.z k m = -A.z k n by linear_combination hzv, neg_sq, sub_self, zero_mul,
      A.fR_neg k ε hμ, sub_self, mul_zero]
  have hchi : A.chiR k ε n m =
      (A.lam ε * A.ρ / (A.z k n + A.z k m) +
        A.Nph k ε (A.z k n) (A.z k m) / A.Pph k (A.z k m)) / A.Pph k (A.z k n) := by
    rw [eq_div_iff (A.Pph_z_ne_zero k n), mul_comm, A.Pph_mul_chiR' k ε hzv, kron_of_ne hnm]
    ring
  have hU : (A.U : ℂ) = 2 * A.a := by rw [a]; push_cast; ring
  have hs := A.weights_sum'
  have eQp : ∀ t, A.Qp k t = (t + A.a - A.w k / 2) * (t + A.a + A.w k / 2) := fun t ↦ by
    rw [Qp]; ring
  have eQm : ∀ t, A.Qm k t = (t - A.a - A.w k / 2) * (t - A.a + A.w k / 2) := fun t ↦ by
    rw [Qm]; ring
  have hQpn := A.Qp_z_ne_zero k n
  have hQpm := A.Qp_z_ne_zero k m
  have hQmn := A.Qm_z_ne_zero k n
  have hQmm := A.Qm_z_ne_zero k m
  rw [eQp] at hQpn hQpm
  rw [eQm] at hQmn hQmm
  obtain ⟨h₁, h₂⟩ := mul_ne_zero_iff.mp hQpn
  obtain ⟨h₃, h₄⟩ := mul_ne_zero_iff.mp hQpm
  obtain ⟨h₅, h₆⟩ := mul_ne_zero_iff.mp hQmn
  obtain ⟨h₇, h₈⟩ := mul_ne_zero_iff.mp hQmm
  rw [hchi, hU]
  simp only [fR, Pph, Nph, N₀, N₂, lam, ζ, qm, qp, ρ, q₀, eQp, eQm]
  push_cast
  rw [show (A.p₂ : ℂ) = 1 - A.p₀ - 2 * A.p₁ by linear_combination hs]
  generalize A.z k n = z at *
  generalize A.z k m = v at *
  generalize (A.a : ℂ) = a at *
  by_cases hk : k = 0
  · subst hk
    rw [kron_self, (A.w_eq_zero_iff 0).mpr rfl] at *
    simp only [zero_div, add_zero, sub_zero] at *
    rw [show z ^ 2 - a ^ 2 = (z - a) * (z + a) by ring,
      show v ^ 2 - a ^ 2 = (v - a) * (v + a) by ring]
    rcases hε with rfl | rfl <;> field_simp <;> ring
  · rw [kron_of_ne hk]
    generalize A.w k = W at *
    obtain ⟨c, rfl⟩ : ∃ c, W = 2 * c := ⟨W / 2, by ring⟩
    simp only [mul_div_cancel_left₀ _ (two_ne_zero' ℂ), zero_mul, zero_div, sub_zero] at *
    field_simp
    ring

/-- `f_s(n) = -[G(xₙ) + G(x_n̄)]/(2b)`, even under the mirror `n ↦ n̄`. -/
noncomputable def fS (n : ℤ) : ℂ := -(A.G (A.x n) + A.G (A.x (mirror k n))) / (2 * A.b k)

/-- **Divided differences, singlet**: for `b ≠ 0` and `n ≠ m`,
`(tₙ - tₘ)χ_s,nm = -U[f_s(n) - f_s(m)]`; off the mirror `χ_s = χ_s,reg` (Proposition "pp"), and on
it both sides vanish. -/
theorem chiS_divided_difference (hb : A.b k ≠ 0) {n m : ℤ} (hnm : n ≠ m) :
    (A.t k n - A.t k m) * A.chiS k n m = -A.U * (A.fS k n - A.fS k m) := by
  by_cases hm : m = mirror k n
  · rw [hm, A.t_mirror, sub_self, zero_mul, fS, fS, mirror_mirror, add_comm (A.G (A.x n)),
      sub_self, mul_zero]
  have h₁ := A.x_sub_ofReal_ne_zero n A.a
  have h₂ := A.x_add_ofReal_ne_zero n A.a
  have h₃ := A.x_sub_ofReal_ne_zero (mirror k n) A.a
  have h₄ := A.x_add_ofReal_ne_zero (mirror k n) A.a
  have h₅ := A.x_sub_ofReal_ne_zero m A.a
  have h₆ := A.x_add_ofReal_ne_zero m A.a
  have h₇ := A.x_sub_ofReal_ne_zero (mirror k m) A.a
  have h₈ := A.x_add_ofReal_ne_zero (mirror k m) A.a
  have hU : (A.U : ℂ) = 2 * A.a := by rw [a]; push_cast; ring
  have hs := A.weights_sum'
  rw [A.chiS_eq k hb, mirrorBlock, kron_of_ne hnm, kron_of_ne hm, chiReg, fS, fS, G, G, G, G,
    A.t_sub_tp, A.t_sub_tp, A.t_sub_tm, A.t_sub_tm, t, t, γp, γm, hU, ρ]
  simp only [A.x_mirror] at h₃ h₄ h₇ h₈ ⊢
  push_cast
  rw [show (A.p₂ : ℂ) = 1 - A.p₀ - 2 * A.p₁ by linear_combination hs]
  generalize A.x n = x at *
  generalize A.x m = y at *
  generalize A.b k = b at *
  generalize (A.a : ℂ) = a at *
  rw [show x ^ 2 - a ^ 2 = (x - a) * (x + a) by ring,
    show y ^ 2 - a ^ 2 = (y - a) * (y + a) by ring,
    show (b - x) ^ 2 - a ^ 2 = (b - x - a) * (b - x + a) by ring,
    show (b - y) ^ 2 - a ^ 2 = (b - y - a) * (b - y + a) by ring]
  field_simp
  ring

end Params

namespace FParams

variable (F : FParams) (k : ℤ)

/-- `f_T(z) = [G↑(z - W/2) - G↓(z + W/2)]/W`, with `G_σ = Gc r_σ`. -/
noncomputable def fT (z : ℂ) : ℂ :=
  (F.Gc F.rUp (z - F.WT k / 2) - F.Gc F.rDn (z + F.WT k / 2)) / F.WT k

/-- `f_T(zₙ) = [G↑(ν_n) - G↓(ν_n + ω)]/W`, by the propagators from the Hamiltonian. -/
theorem fT_zT (n : ℤ) :
    F.fT k (F.zT k n) = (F.Gs true (F.ν n) - F.Gs false (F.ν (n + k))) / F.WT k := by
  rw [F.propagator_up, F.propagator_dn, fT]
  simp only [zT, Params.z, Params.x, Params.w, WT, lat_μ, lat_ν, lat_ω, ← Params.ν_add_ω]
  push_cast
  ring_nf

/-- At `μ = 0`, `f_T` is even (`r↓ = -r↑`): on the mirror `zₘ = -zₙ` both sides of
`chiTUp_divided_difference` vanish. -/
theorem fT_neg (hμ : F.μ = 0) (z : ℂ) : F.fT k (-z) = F.fT k z := by
  have hp : F.p₂ = F.p₀ := by rw [F.p₂_eq_p₀_mul, hμ, mul_zero, Real.exp_zero, mul_one]
  rw [fT, fT, Gc, Gc, Gc, Gc, rUp, rDn, ρ, hp, show (-z - F.WT k / 2) ^ 2 = (z + F.WT k / 2) ^ 2 by
    ring, show (-z + F.WT k / 2) ^ 2 = (z - F.WT k / 2) ^ 2 by ring]
  push_cast
  ring

set_option maxRecDepth 10000 in
-- `field_simp` on the polynomials of the transverse input
/-- **Divided differences, transverse**: for `W ≠ 0` and `n ≠ m`,
`(zₙ² - zₘ²)χ_T↑,nm = -U[f_T(zₙ) - f_T(zₘ)]`, by the transverse input (`PT_mul_chiTUp'`); on the
mirror at `μ = 0` both sides vanish (`fT_neg`). -/
theorem chiTUp_divided_difference (hW : F.WT k ≠ 0) {n m : ℤ} (hnm : n ≠ m) :
    (F.zT k n ^ 2 - F.zT k m ^ 2) * F.chiTUp k n m =
      -F.U * (F.fT k (F.zT k n) - F.fT k (F.zT k m)) := by
  by_cases hzv : F.zT k n + F.zT k m = 0
  · have hμ : F.μ = 0 := by
      have h := congrArg re hzv
      simp only [zT, Params.z_eq, add_re, ofReal_re, mul_re, ofReal_im, I_re, I_im, zero_re,
        lat_μ] at h
      linarith
    rw [show F.zT k m = -F.zT k n by linear_combination hzv, neg_sq, sub_self, zero_mul,
      F.fT_neg k hμ, sub_self, mul_zero]
  have hD : ¬(F.μ = 0 ∧ n + m + k + 1 = 0) := fun ⟨hμ, hs⟩ ↦ hzv (by
    rw [zT, zT, show m = mirror k n by unfold mirror; omega,
      (F.lat F.μ).z_mirror_of_mu k (by rw [lat_μ, hμ]) n]
    ring)
  have hchi : F.chiTUp k n m =
      (F.lamT * F.ρ / (F.zT k n + F.zT k m) +
        F.NT k (F.zT k n) (F.zT k m) / F.PT k (F.zT k m)) / F.PT k (F.zT k n) := by
    rw [eq_div_iff (F.PT_zT_ne_zero k n), mul_comm, F.PT_mul_chiTUp' k hW hD, kron_of_ne hnm]
    ring
  have hU : (F.U : ℂ) = 2 * F.a := by rw [a]; push_cast; ring
  have hs := F.weights_sum'
  have h₁ := F.lin₁ n
  have h₂ := F.lin₂ n
  have h₃ := F.lin₃ k n
  have h₄ := F.lin₄ k n
  have h₅ := F.lin₁ m
  have h₆ := F.lin₂ m
  have h₇ := F.lin₃ k m
  have h₈ := F.lin₄ k m
  rw [hchi, hU]
  simp only [fT, Gc, NT, N₀T, N₂T, F.PT_zT, PTf, lamT, ζT, q₀T, rUp, rDn, ρ, mf]
  rw [WT_eq] at hW ⊢
  simp only [zT, Params.z] at hzv ⊢
  push_cast
  rw [show (F.p₂ : ℂ) = 1 - F.p₀ - F.pUp - F.pDn by linear_combination hs]
  generalize (F.lat F.μ).x n = x at *
  generalize (F.lat F.μ).x m = y at *
  generalize (F.lat F.μ).w k = w at *
  generalize (F.a : ℂ) = a at *
  generalize (F.h : ℂ) = h at *
  obtain ⟨c, rfl⟩ : ∃ c, w = 2 * c := ⟨w / 2, by ring⟩
  simp only [mul_div_cancel_left₀ _ (two_ne_zero' ℂ)] at *
  rw [show x + c - (2 * c - 2 * h) / 2 = x + h by ring,
    show x + c + (2 * c - 2 * h) / 2 = x + 2 * c - h by ring,
    show y + c - (2 * c - 2 * h) / 2 = y + h by ring,
    show y + c + (2 * c - 2 * h) / 2 = y + 2 * c - h by ring,
    show (x + h) ^ 2 - a ^ 2 = (x - a + h) * (x + a + h) by ring,
    show (x + 2 * c - h) ^ 2 - a ^ 2 = (x + 2 * c - a - h) * (x + 2 * c + a - h) by ring,
    show (y + h) ^ 2 - a ^ 2 = (y - a + h) * (y + a + h) by ring,
    show (y + 2 * c - h) ^ 2 - a ^ 2 = (y + 2 * c - a - h) * (y + 2 * c + a - h) by ring]
  have hch : c - h ≠ 0 := fun h0 ↦ hW (by linear_combination 2 * h0)
  generalize hA₁ : x + 2 * c + a - h = A₁ at *
  generalize hA₂ : x + 2 * c - a - h = A₂ at *
  generalize hA₃ : y + 2 * c + a - h = A₃ at *
  generalize hA₄ : y + 2 * c - a - h = A₄ at *
  field_simp
  subst hA₁ hA₂ hA₃ hA₄
  ring

/-- `f_σσ(z) = q₋/Q₊(z) + q₊/Q₋(z) - δ_k0 ζ_L/(2a(z² - a²))` with `q± = (1 ± r_σ)/2`. -/
noncomputable def fL (r : ℝ) (z : ℂ) : ℂ :=
  (1 - r) / 2 / ((z + F.a) ^ 2 - F.wL k ^ 2 / 4) + (1 + r) / 2 / ((z - F.a) ^ 2 - F.wL k ^ 2 / 4) -
    kron k 0 * F.ζL / (2 * F.a * (z ^ 2 - F.a ^ 2))

/-- `f↑↓(z) = -δ_k0 ζ↑/(2a(z² - a²))`. -/
noncomputable def fLUpDn (z : ℂ) : ℂ := -(kron k 0 * F.ζUp / (2 * F.a * (z ^ 2 - F.a ^ 2)))

/-- For `k ≠ 0`, `f_σσ(zₙ) = [G(xₙ) - G(xₙ + w)]/w` on the lattice of spin `σ`, with
`G = Gc r_σ`; by `propagator_up`, `propagator_dn` this is `[G_σ(ν_n) - G_σ(ν_n + ω)]/w`. -/
theorem fL_eq_G (hk : k ≠ 0) (c r : ℝ) (n : ℤ) :
    F.fL k r ((F.lat c).z k n) =
      (F.Gc r ((F.lat c).x n) - F.Gc r ((F.lat c).x (n + k))) / F.wL k := by
  set B := F.lat c
  have hw : F.wL k ≠ 0 := mt (B.w_eq_zero_iff k).mp hk
  have h₁ := B.x_sub_ofReal_ne_zero n F.a
  have h₂ := B.x_add_ofReal_ne_zero n F.a
  have h₃ := B.x_sub_ofReal_ne_zero (n + k) F.a
  have h₄ := B.x_add_ofReal_ne_zero (n + k) F.a
  rw [← B.x_add_w] at h₃ h₄ ⊢
  rw [fL, Gc, Gc, kron_of_ne hk, Params.z, show B.w k = F.wL k from rfl]
  rw [show B.w k = F.wL k from rfl] at h₃ h₄
  generalize B.x n = x at *
  generalize F.wL k = w at *
  generalize (F.a : ℂ) = a at *
  obtain ⟨c, rfl⟩ : ∃ c, w = 2 * c := ⟨w / 2, by ring⟩
  rw [show (x + 2 * c / 2 + a) ^ 2 - (2 * c) ^ 2 / 4 = (x + a) * (x + 2 * c + a) by ring,
    show (x + 2 * c / 2 - a) ^ 2 - (2 * c) ^ 2 / 4 = (x - a) * (x + 2 * c - a) by ring,
    show x ^ 2 - a ^ 2 = (x - a) * (x + a) by ring,
    show (x + 2 * c) ^ 2 - a ^ 2 = (x + 2 * c - a) * (x + 2 * c + a) by ring]
  have hc : c ≠ 0 := fun h0 ↦ hw (by rw [h0, mul_zero])
  generalize hA₁ : x + 2 * c + a = A₁ at *
  generalize hA₂ : x + 2 * c - a = A₂ at *
  field_simp
  subst hA₁ hA₂
  ring

/-- `f↑↑(z↑ₙ) = [G↑(ν_n) - G↑(ν_n + ω)]/w` for `k ≠ 0`. -/
theorem fL_up_eq_G (hk : k ≠ 0) (n : ℤ) :
    F.fL k F.rUp (F.Bup.z k n) = (F.Gs true (F.ν n) - F.Gs true (F.ν (n + k))) / F.wL k := by
  rw [F.fL_eq_G k hk, F.propagator_up, F.propagator_up]

/-- `f↓↓(z↓ₙ) = [G↓(ν_n) - G↓(ν_n + ω)]/w` for `k ≠ 0`. -/
theorem fL_dn_eq_G (hk : k ≠ 0) (n : ℤ) :
    F.fL k F.rDn (F.Bdn.z k n) = (F.Gs false (F.ν n) - F.Gs false (F.ν (n + k))) / F.wL k := by
  rw [F.fL_eq_G k hk, F.propagator_dn, F.propagator_dn]

/-- `z↑ₙ ± z↓ₘ`: off the axes the two lattices neither meet nor mirror each other. -/
theorem zUp_sub_zDn_ne (hh : F.h ≠ 0) (n m : ℤ) : F.Bup.z k n - F.Bdn.z k m ≠ 0 := by
  intro h0
  have := congrArg Complex.re h0
  simp [Params.z, Params.x, Params.w] at this
  exact hh (by linarith)

theorem zUp_add_zDn_ne (hμ : F.μ ≠ 0) (n m : ℤ) : F.Bup.z k n + F.Bdn.z k m ≠ 0 := by
  intro h0
  have := congrArg Complex.re h0
  simp [Params.z, Params.x, Params.w] at this
  exact hμ (by linarith)

set_option maxRecDepth 10000 in
-- `field_simp` on the polynomials of the longitudinal input
/-- **Divided differences, longitudinal, opposite spins**: off the axes,
`(z↑ₙ² - z↓ₘ²)χ↑↓,nm = U[f↑↑(z↑ₙ) - f↓↓(z↓ₘ)]`, by `PL_mul_chiL_updn`. -/
theorem chiL_updn_divided_difference (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (n m : ℤ) :
    (F.Bup.z k n ^ 2 - F.Bdn.z k m ^ 2) * F.chiL true false k n m =
      F.U * (F.fL k F.rUp (F.Bup.z k n) - F.fL k F.rDn (F.Bdn.z k m)) := by
  have hp := F.zUp_add_zDn_ne k hμ n m
  have hd := F.zUp_sub_zDn_ne k hh n m
  have hPn : F.PL k (F.Bup.z k n) ≠ 0 := by
    rw [F.PL_up]
    have := F.Bup.x_add_ofReal_ne_zero n F.a
    have := F.Bup.x_sub_ofReal_ne_zero n F.a
    have := F.Bup.x_add_ofReal_ne_zero (n + k) F.a
    have := F.Bup.x_sub_ofReal_ne_zero (n + k) F.a
    rw [← F.Bup.x_add_w] at *
    exact mul_ne_zero (mul_ne_zero ‹_› ‹_›) (mul_ne_zero ‹_› ‹_›)
  have hchi : F.chiL true false k n m =
      (F.lamL * (F.ρ / (F.Bup.z k n + F.Bdn.z k m) - F.mf / (F.Bup.z k n - F.Bdn.z k m)) +
        (F.N₀L k F.rDn (F.Bdn.z k m) + F.Bup.z k n ^ 2 * F.N₂L k F.rDn (F.Bdn.z k m)) /
          F.PL k (F.Bdn.z k m)) / F.PL k (F.Bup.z k n) := by
    rw [eq_div_iff hPn, mul_comm, F.PL_mul_chiL_updn k hμ hh n m]
  have hU : (F.U : ℂ) = 2 * F.a := by rw [a]; push_cast; ring
  have hs := F.weights_sum'
  have h₁ := F.Bup.x_add_ofReal_ne_zero n F.a
  have h₂ := F.Bup.x_sub_ofReal_ne_zero n F.a
  have h₃ := F.Bup.x_add_ofReal_ne_zero (n + k) F.a
  have h₄ := F.Bup.x_sub_ofReal_ne_zero (n + k) F.a
  have h₅ := F.Bdn.x_add_ofReal_ne_zero m F.a
  have h₆ := F.Bdn.x_sub_ofReal_ne_zero m F.a
  have h₇ := F.Bdn.x_add_ofReal_ne_zero (m + k) F.a
  have h₈ := F.Bdn.x_sub_ofReal_ne_zero (m + k) F.a
  rw [← F.Bup.x_add_w] at h₃ h₄
  rw [← F.Bdn.x_add_w] at h₇ h₈
  rw [hchi, hU]
  simp only [fL, N₀L, N₂L, HL, F.PL_up, F.PL_dn, lamL, ζL, q₀L, rUp, rDn, ρ, mf]
  simp only [Params.z] at hp hd ⊢
  rw [show F.Bup.w k = F.wL k from rfl, show F.Bdn.w k = F.wL k from rfl] at *
  push_cast
  rw [show (F.p₂ : ℂ) = 1 - F.p₀ - F.pUp - F.pDn by linear_combination hs]
  generalize F.Bup.x n = x at *
  generalize F.Bdn.x m = y at *
  generalize (F.a : ℂ) = a at *
  by_cases hk : k = 0
  · subst hk
    rw [kron_self, show F.wL 0 = 0 from (F.Bup.w_eq_zero_iff 0).mpr rfl] at *
    simp only [zero_div, add_zero, sub_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
      zero_pow] at *
    rw [show x ^ 2 - a ^ 2 = (x - a) * (x + a) by ring, show y ^ 2 - a ^ 2 = (y - a) * (y + a) by
      ring]
    field_simp
    ring
  · rw [kron_of_ne hk]
    generalize F.wL k = w at *
    obtain ⟨c, rfl⟩ : ∃ c, w = 2 * c := ⟨w / 2, by ring⟩
    simp only [mul_div_cancel_left₀ _ (two_ne_zero' ℂ), zero_mul, zero_div, sub_zero] at *
    rw [show (x + c + a) ^ 2 - (2 * c) ^ 2 / 4 = (x + a) * (x + 2 * c + a) by ring,
      show (x + c - a) ^ 2 - (2 * c) ^ 2 / 4 = (x - a) * (x + 2 * c - a) by ring,
      show (y + c + a) ^ 2 - (2 * c) ^ 2 / 4 = (y + a) * (y + 2 * c + a) by ring,
      show (y + c - a) ^ 2 - (2 * c) ^ 2 / 4 = (y - a) * (y + 2 * c - a) by ring]
    generalize hA₁ : x + 2 * c + a = A₁ at *
    generalize hA₂ : x + 2 * c - a = A₂ at *
    generalize hA₃ : y + 2 * c + a = A₃ at *
    generalize hA₄ : y + 2 * c - a = A₄ at *
    field_simp
    subst hA₁ hA₂ hA₃ hA₄
    ring

/-- **Divided differences, longitudinal, equal spins**: for `n ≠ m`,
`(z↑ₙ² - z↑ₘ²)χ↑↑,nm = U[f↑↓(z↑ₙ) - f↑↓(z↑ₘ)]`, by `PL_mul_chiL_upup`; for `k ≠ 0` both sides
vanish. -/
theorem chiL_upup_divided_difference {n m : ℤ} (hnm : n ≠ m) :
    (F.Bup.z k n ^ 2 - F.Bup.z k m ^ 2) * F.chiL true true k n m =
      F.U * (F.fLUpDn k (F.Bup.z k n) - F.fLUpDn k (F.Bup.z k m)) := by
  have hP : ∀ j, F.PL k (F.Bup.z k j) ≠ 0 := fun j ↦ by
    rw [F.PL_up]
    have := F.Bup.x_add_ofReal_ne_zero j F.a
    have := F.Bup.x_sub_ofReal_ne_zero j F.a
    have := F.Bup.x_add_ofReal_ne_zero (j + k) F.a
    have := F.Bup.x_sub_ofReal_ne_zero (j + k) F.a
    rw [← F.Bup.x_add_w] at *
    exact mul_ne_zero (mul_ne_zero ‹_› ‹_›) (mul_ne_zero ‹_› ‹_›)
  have hchi : F.chiL true true k n m =
      kron k 0 * F.ζUp * ((F.Bup.z k n ^ 2 - F.a ^ 2) * (F.Bup.z k m ^ 2 - F.a ^ 2)) /
        F.PL k (F.Bup.z k m) / F.PL k (F.Bup.z k n) := by
    rw [eq_div_iff (hP n), mul_comm, F.PL_mul_chiL_upup k n m, kron_of_ne hnm]
    ring
  have hU : (F.U : ℂ) = 2 * F.a := by rw [a]; push_cast; ring
  rw [hchi, hU, fLUpDn, fLUpDn]
  by_cases ha : (F.a : ℂ) = 0
  · simp [ζUp, ha]
  by_cases hk : k = 0
  · subst hk
    have hPn := hP n
    have hPm := hP m
    simp only [PL, show F.wL 0 = 0 from (F.Bup.w_eq_zero_iff 0).mpr rfl] at hPn hPm ⊢
    rw [kron_self]
    generalize F.Bup.z 0 n = z at *
    generalize F.Bup.z 0 m = v at *
    generalize (F.a : ℂ) = a at *
    have h₁ : z - a ≠ 0 := fun h0 ↦ hPn (by rw [show z = a by linear_combination h0]; ring)
    have h₂ : z + a ≠ 0 := fun h0 ↦ hPn (by rw [show z = -a by linear_combination h0]; ring)
    have h₃ : v - a ≠ 0 := fun h0 ↦ hPm (by rw [show v = a by linear_combination h0]; ring)
    have h₄ : v + a ≠ 0 := fun h0 ↦ hPm (by rw [show v = -a by linear_combination h0]; ring)
    rw [show z ^ 2 - a ^ 2 = (z - a) * (z + a) by ring, show v ^ 2 - a ^ 2 = (v - a) * (v + a) by
      ring]
    field_simp
    ring
  · simp [kron_of_ne hk]

/-! ### The rows of spin `↓`, by the spin flip -/

theorem flip_Bup : F.flip.Bup = F.Bdn := rfl

theorem flip_Bdn : F.flip.Bdn = F.Bup := by
  simp only [Bup, Bdn, lat, flip_μ, flip_h, flip_U, flip_β, sub_neg_eq_add]

theorem flip_fL (r : ℝ) (z : ℂ) : F.flip.fL k r z = F.fL k r z := by
  simp only [fL, ζL, flip_p₀, flip_p₂, flip_pUp, flip_pDn]
  rw [mul_comm (F.pDn : ℂ)]
  rfl

theorem flip_rUp : F.flip.rUp = F.rDn := by
  rw [rUp, rDn, F.flip_ρ, F.flip_mf, sub_neg_eq_add]

theorem flip_rDn : F.flip.rDn = F.rUp := by
  rw [rUp, rDn, F.flip_ρ, F.flip_mf, ← sub_eq_add_neg]

/-- **Divided differences, longitudinal, opposite spins, rows `↓`**: off the axes,
`(z↓ₙ² - z↑ₘ²)χ↓↑,nm = U[f↓↓(z↓ₙ) - f↑↑(z↑ₘ)]`, by the spin flip. -/
theorem chiL_dnup_divided_difference (hμ : F.μ ≠ 0) (hh : F.h ≠ 0) (n m : ℤ) :
    (F.Bdn.z k n ^ 2 - F.Bup.z k m ^ 2) * F.chiL false true k n m =
      F.U * (F.fL k F.rDn (F.Bdn.z k n) - F.fL k F.rUp (F.Bup.z k m)) := by
  rw [F.chiL_flip]
  simpa [flip_Bup, flip_Bdn, flip_fL, flip_rUp, flip_rDn] using
    F.flip.chiL_updn_divided_difference k hμ (neg_ne_zero.mpr hh) n m

/-- **Divided differences, longitudinal, equal spins, rows `↓`**: for `n ≠ m`,
`(z↓ₙ² - z↓ₘ²)χ↓↓,nm = U[f↓↑(z↓ₙ) - f↓↑(z↓ₘ)]` with `f↓↑ = -δ_k0 ζ↓/(2a(z² - a²))`,
`ζ↓ = 4βa²(p₀ + p↓)(p↑ + p₂)` (`f↑↓` of the flipped atom), by the spin flip. -/
theorem chiL_dndn_divided_difference {n m : ℤ} (hnm : n ≠ m) :
    (F.Bdn.z k n ^ 2 - F.Bdn.z k m ^ 2) * F.chiL false false k n m =
      F.U * (F.flip.fLUpDn k (F.Bdn.z k n) - F.flip.fLUpDn k (F.Bdn.z k m)) := by
  rw [F.chiL_flip]
  simpa [flip_Bup] using F.flip.chiL_upup_divided_difference k hnm

end FParams

end HubbardAtom
