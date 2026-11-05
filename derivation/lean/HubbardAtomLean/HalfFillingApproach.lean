import HubbardAtomLean.SingletLZApproach
import HubbardAtomLean.FieldLAxisGeneric
import HubbardAtomLean.Confluent

/-!
# Corollary "half filling" and its lattice zeros are approached in `μ`

`tendsto_gammaR_of_HFHyp` and `tendsto_gammaR_of_HFLZ` (`LZLimits.lean`) give `gammaR → gammaHF`,
and `gammaR → gammaHFlz` at a lattice zero `z_c² = ŝ`, along any approach to `μ = 0` through points
where the assumptions of Theorem "density and magnetic" hold. Here these points are shown to exist,
as in the note's proofs of Corollary "half filling" and of item 4 of Proposition "special": at fixed
`U`, `β`, `k`, with `ŝ' ≠ 0` for odd `k`, the assumptions `DMGeneric` hold at every `μ ≠ 0` near
`0`, under `HFHyp`, and under `HFLZ` with `c` off the mirror fixed points. With the note's
assumption `e(zₙ) ≠ 0`, this is Corollary "half filling" as stated in the note.

* `Params.exists_root_of_mu`: at `μ = 0` the roots `ŝ ≠ ŝ'` of `e` (in `s = t²`) are simple, so (by
  `exists_continuousAt_sqrt_root_of_simple₀`) a square root of either moves to a root of `e` at
  `A.withMu μ`, continuous at `μ = 0`;
* `Params.Kph_eq_rho_mul`: at a root `u` of `e`, with `X = βH(-u)`,
  `(X - λc̄)(X + λc̄) = -ρ(4aβuX + λ²β²ρ/4)` (`Kph_identity`), so `K = ρΨ(u)` with
  `Ψ = λS_k(u) - (4aβuX + λ²β²ρ/4)/(X + λc̄)`. At `μ = 0`, `Ψ(û) = -2aûD̂ ≠ 0` (`Psi_of_mu`), so
  `K(u₁) ≠ 0` at every `μ ≠ 0` near `0`; `K(u₂) → -2λβp₀ ≠ 0` (`Kph_of_mu'`);
* `Params.Pph_ne_of_shat'`: at `μ = 0`, `P = 4a²(a²(1 + 2ε) + q₀)` at the roots of `ŝ'`, which is
  nonzero except in the static magnetic channel (`k = 0`, `ε = -1`); there `ŝ' = a²`
  (`shat'_static`), and `u₂ = a` is the atomic root at every `μ` (`atomic_root`, `exists_root₂`);
* `Params.eventually_z_sq_ne`: near `0` the roots `u₁²`, `u₂²` stay real (`eventually_conj_sq`: `e`
  has real coefficients, `eph_conj`), while `Im zₙ² = 2μνₙ' ≠ 0` off the mirror fixed points
  (`im_z_sq`), so `e(zₙ) ≠ 0` at every `μ ≠ 0`, also where `u₁(0)` lies on the lattice;
* `Params.eventually_dmGeneric_of`: so `DMGeneric` holds at every `μ ≠ 0` near `0` once `Ψ(u₁) ≠ 0`
  there;
* `Params.not_isUnit_onePlusKR_of_Kph`: for `μ ≠ 0`, `K(u) = 0` at a root `u ≠ 0` of `e` makes the
  mode `φ_u` a null vector of `χ_r` (Step 3, `hasSum_mode_gen`), so `1 + 𝒦` is not invertible; and
  `1 + 𝒦` stays invertible near `μ = 0` (`eventually_isUnit_onePlusKR`);
* `Params.eventually_eph_zero_ne`: `e(0) ≠ 0` at every `μ ≠ 0` near `0`, as
  `e(0) = (q₀ + λ(p₁ - p₀e^{βμ}))(q₀ + λ(p₁ + p₀e^{βμ}))` and `p₁ ± p₀e^{βμ}` takes its value at
  `μ = 0` nowhere else;
* `Params.HFHyp.exists_eventually_dmGeneric`, `Params.HFHyp.exists_tendsto_gammaR`: for `u₁(0) = û`,
  `Ψ(û) ≠ 0` if `û ≠ 0`; at `û = 0` (even `k`), `Ψ(u₁) → 0`, but `K(u₁) ≠ 0` by the two previous
  items, as `u₁ ≠ 0` for `μ ≠ 0`. So **`gammaR → gammaHF` as `μ → 0`, `μ ≠ 0`**, at fixed `U`, `β`,
  `k`; under the note's assumptions, with `e(zₙ) ≠ 0` for all `n`, `exists_tendsto_gammaR_of_eph`;
* `Params.HFLZ.exists_eventually_dmGeneric`, `Params.HFLZ.exists_tendsto_gammaR`: for
  `u₁(0) = z_c`, `S_k⁻¹ = -(2/β)coth` (even `k`) or `-(2/β)tanh` (odd `k`) is continuous with value
  `0` at `z_c` (`Sinv`, `continuousAt_withMu_Sinv`), so `ΨS_k⁻¹ → λ ≠ 0`, and
  **`gammaR → gammaHFlz` as `μ → 0`, `μ ≠ 0`**, at fixed `U`, `β`, `k`.

Not covered, both at the mirror fixed point `z = 0` of odd `k`: `ŝ' = 0` there (`r = m` at
`|U|β = 2π|k|`), a zero of `e(0)` that the note's assumptions of Corollary "half filling" exclude,
but not those of item 4 of Proposition "special"; and a lattice zero `z_c = 0`.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-! ### Continuity in `μ` -/

theorem continuousAt_withMu_p (μ₀ : ℝ) :
    ContinuousAt (fun μ ↦ ((A.withMu μ).p₀ : ℂ)) μ₀ ∧
      ContinuousAt (fun μ ↦ ((A.withMu μ).p₁ : ℂ)) μ₀ ∧
      ContinuousAt (fun μ ↦ ((A.withMu μ).p₂ : ℂ)) μ₀ :=
  continuousAt_fam_p (U := fun _ ↦ A.U) (β := fun _ ↦ A.β) (μ := id) (hβ := fun _ ↦ A.β_pos)
    continuousAt_const continuousAt_const continuousAt_id

theorem continuousAt_withMu_ρ (μ₀ : ℝ) : ContinuousAt (fun μ ↦ ((A.withMu μ).ρ : ℂ)) μ₀ :=
  continuousAt_fam_ρ (U := fun _ ↦ A.U) (β := fun _ ↦ A.β) (μ := id) (hβ := fun _ ↦ A.β_pos)
    continuousAt_const continuousAt_const continuousAt_id

/-- `H` at a moving point is continuous in `μ`. -/
theorem continuousAt_withMu_Hph {μ₀ : ℝ} {u : ℝ → ℂ} (hu : ContinuousAt u μ₀) :
    ContinuousAt (fun μ ↦ (A.withMu μ).Hph k ε (u μ)) μ₀ := by
  obtain ⟨-, h₁, -⟩ := A.continuousAt_withMu_p μ₀
  have hρ := A.continuousAt_withMu_ρ μ₀
  simp only [Hph, withMu_a, show ∀ μ, (A.withMu μ).q₀ k = A.q₀ k from fun _ ↦ rfl]
  fun_prop

/-- `e` at a fixed point is continuous in `μ`. -/
theorem continuousAt_withMu_eph (μ₀ : ℝ) (z : ℂ) :
    ContinuousAt (fun μ ↦ (A.withMu μ).eph k ε z) μ₀ := by
  obtain ⟨h₀, -, h₂⟩ := A.continuousAt_withMu_p μ₀
  have hH := A.continuousAt_withMu_Hph k ε (continuousAt_const (x := μ₀) (y := z))
  have hH' := A.continuousAt_withMu_Hph k ε (continuousAt_const (x := μ₀) (y := -z))
  simp only [eph, show ∀ μ, (A.withMu μ).lam ε = A.lam ε from fun _ ↦ rfl]
  fun_prop

/-! ### The roots of `e` near `μ = 0` -/

/-- The coefficients of `e` as a polynomial in `s = t²`. -/
noncomputable def ecoef (μ : ℝ) : Fin 3 → ℂ :=
  ![(A.withMu μ).eph k ε 0, (A.withMu μ).eph k ε 1 - 1 - (A.withMu μ).eph k ε 0, 1]

theorem sum_ecoef (μ : ℝ) (t : ℂ) :
    ∑ i, A.ecoef k ε μ i * (t ^ 2) ^ (i : ℕ) = (A.withMu μ).eph k ε t := by
  simp only [ecoef, Fin.sum_univ_three, eph, Hph]
  simp
  ring

theorem continuousAt_ecoef (μ₀ : ℝ) : ContinuousAt (A.ecoef k ε) μ₀ := by
  refine continuousAt_pi.mpr fun i ↦ ?_
  have h0 := A.continuousAt_withMu_eph k ε μ₀ 0
  have h1 := A.continuousAt_withMu_eph k ε μ₀ 1
  fin_cases i <;> simp only [ecoef] <;> simp <;> fun_prop

theorem withMu_self : A.withMu A.μ = A := rfl

theorem shat_sub_shat' : A.shat k ε - A.shat' k ε = 8 * ε * A.a ^ 2 * A.p₀ := by
  simp only [shat, shat']
  ring

theorem shat_sub_shat'_ne (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) :
    A.shat k ε - A.shat' k ε ≠ 0 := by
  rw [shat_sub_shat']
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero (by norm_num) hε0) (pow_ne_zero 2 ha))
    (ofReal_ne_zero.mpr A.p₀_pos.ne')

/-- **A simple root of `e` at `μ = 0` moves continuously in `μ`**: for `v` with `e(v) = 0` and
`2v² ≠ ŝ + ŝ'` there is a root `w(μ)` of `e` at `A.withMu μ` near `μ = 0`, continuous at `0`,
with `w(0) = v` (`exists_continuousAt_sqrt_root_of_simple₀`, also at `v = 0`). -/
theorem exists_root_of_mu (hμ : A.μ = 0) {v : ℂ} (he : A.eph k ε v = 0)
    (hs : 2 * v ^ 2 ≠ A.shat k ε + A.shat' k ε) :
    ∃ w : ℝ → ℂ, w A.μ = v ∧ ContinuousAt w A.μ ∧
      ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).eph k ε (w μ) = 0 := by
  have hroot : ∑ i, A.ecoef k ε A.μ i * (v ^ 2) ^ (i : ℕ) = 0 := by
    rw [sum_ecoef, withMu_self, he]
  have hsimple : ∑ i : Fin 3, ((i : ℕ) : ℂ) * A.ecoef k ε A.μ i * (v ^ 2) ^ ((i : ℕ) - 1) ≠ 0 := by
    have h1 := A.eph_of_mu' k ε hμ 1
    have h0 := A.eph_of_mu' k ε hμ 0
    simp only [Fin.sum_univ_three, ecoef, withMu_self]
    simp only [Fin.isValue, Fin.val_zero, Nat.cast_zero, Matrix.cons_val_zero, zero_mul,
      Fin.val_one, Nat.cast_one, Matrix.cons_val_one, one_mul, Nat.sub_self, pow_zero, mul_one,
      zero_add, Fin.val_two, Nat.cast_ofNat, Matrix.cons_val_two, Matrix.tail_cons,
      Matrix.head_cons, Nat.add_one_sub_one, pow_one, h1, h0]
    intro h0'
    exact hs (by linear_combination h0')
  obtain ⟨w, hw₀, hw, hroot'⟩ :=
    exists_continuousAt_sqrt_root_of_simple₀ (A.continuousAt_ecoef k ε A.μ) hroot hsimple
  exact ⟨w, hw₀, hw, hroot'.mono fun μ h ↦ by rwa [sum_ecoef] at h⟩

/-! ### The thermal denominator at a root -/

/-- At a root `u` of `e`, with `X = βH(-u)`: `(X - λc̄)(X + λc̄) = -ρ(4aβuX + λ²β²ρ/4)`. -/
theorem Kph_identity {u : ℂ} (he : A.eph k ε u = 0) :
    (A.β * A.Hph k ε (-u) - A.lam ε * A.cbar) * (A.β * A.Hph k ε (-u) + A.lam ε * A.cbar) =
      -(A.ρ * (4 * A.a * A.β * u * (A.β * A.Hph k ε (-u)) + A.lam ε ^ 2 * A.β ^ 2 * A.ρ / 4)) := by
  simp only [eph, Hph, cbar, ρ] at he ⊢
  push_cast at he ⊢
  linear_combination (A.β : ℂ) ^ 2 * he

/-- `Ψ(u) = λS_k(u) - (4aβuX + λ²β²ρ/4)/(X + λc̄)`, `X = βH(-u)`: `K = ρΨ` at a root of `e`. -/
noncomputable def Psi (u : ℂ) : ℂ :=
  A.lam ε * A.S k u - (4 * A.a * A.β * u * (A.β * A.Hph k ε (-u)) +
    A.lam ε ^ 2 * A.β ^ 2 * A.ρ / 4) / (A.β * A.Hph k ε (-u) + A.lam ε * A.cbar)

/-- **`K = ρΨ` at a root of `e`** (with `X + λc̄ ≠ 0`). -/
theorem Kph_eq_rho_mul {u : ℂ} (he : A.eph k ε u = 0)
    (hX : A.β * A.Hph k ε (-u) + A.lam ε * A.cbar ≠ 0) : A.Kph k ε u = A.ρ * A.Psi k ε u := by
  have hid := A.Kph_identity k ε he
  simp only [Kph, α, Psi]
  field_simp
  linear_combination 4 * hid

/-- At `μ = 0` and `û² = ŝ`, `û ≠ 0`: `Ψ(û) = -2aûD̂`. -/
theorem Psi_of_mu (hμ : A.μ = 0) {u : ℂ} (hu : u ^ 2 = A.shat k ε) (hu0 : u ≠ 0)
    (hl : A.lam ε ≠ 0) :
    A.Psi k ε u = -2 * A.a * u * A.Dhat k ε u := by
  have hH : A.Hph k ε (-u) = A.lam ε * A.p₀ := by
    have := A.Hph_sub_of_mu k ε hμ (-u)
    rw [neg_sq, hu, sub_self, sub_eq_zero] at this
    exact this
  have hc : (A.cbar : ℂ) = A.β * A.p₀ := by
    simp only [cbar, p₂_eq_p₀_of_mu hμ]
    push_cast
    ring
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hp : (A.p₀ : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.p₀_pos.ne'
  simp only [Psi, Dhat, That, hu0, ↓reduceIte, hH, hc, A.rho_eq_zero_of_mu hμ]
  simp only [lam] at hl ⊢
  have hε : ε ≠ 0 := by rintro rfl; simp at hl
  field_simp
  ring

/-- At `μ = 0` and `v² = ŝ'`: `K(v) = -2λβp₀`. -/
theorem Kph_of_mu' (hμ : A.μ = 0) {v : ℂ} (hv : v ^ 2 = A.shat' k ε) :
    A.Kph k ε v = -2 * A.lam ε * A.β * A.p₀ := by
  have hH : A.Hph k ε (-v) = -(A.lam ε * A.p₀) := by
    have := A.Hph_add_of_mu k ε hμ (-v)
    rw [neg_sq, hv, sub_self] at this
    linear_combination this
  simp only [Kph, α, cbar, hH, A.rho_eq_zero_of_mu hμ, p₂_eq_p₀_of_mu hμ]
  push_cast
  ring

/-! ### Continuity of `K` and `Ψ` at a moving root -/

theorem withMu_β (μ : ℝ) : (A.withMu μ).β = A.β := rfl
theorem withMu_lam (μ : ℝ) : (A.withMu μ).lam ε = A.lam ε := rfl

section Moving

variable {μ₀ : ℝ} {u : ℝ → ℂ} (hu : ContinuousAt u μ₀)
include hu

theorem continuousAt_withMu_X :
    ContinuousAt (fun μ ↦ ((A.withMu μ).β : ℂ) * (A.withMu μ).Hph k ε (-u μ) +
      (A.withMu μ).lam ε * (A.withMu μ).cbar) μ₀ := by
  obtain ⟨h₀, -, h₂⟩ := A.continuousAt_withMu_p μ₀
  have hH := A.continuousAt_withMu_Hph k ε hu.neg
  simp only [withMu_β, withMu_lam, cbar]
  push_cast
  fun_prop

variable (hoff : ∀ n, (A.withMu μ₀).z k n ≠ u μ₀)
include hoff

theorem continuousAt_withMu_S : ContinuousAt (fun μ ↦ (A.withMu μ).S k (u μ)) μ₀ :=
  continuousAt_S (fun μ ↦ A.withMu μ) k continuousAt_id continuousAt_const hu hoff

theorem continuousAt_withMu_Kph : ContinuousAt (fun μ ↦ (A.withMu μ).Kph k ε (u μ)) μ₀ := by
  obtain ⟨h₀, -, h₂⟩ := A.continuousAt_withMu_p μ₀
  have hρ := A.continuousAt_withMu_ρ μ₀
  have hH := A.continuousAt_withMu_Hph k ε hu.neg
  have hS := A.continuousAt_withMu_S k hu hoff
  simp only [Kph, α, withMu_β, withMu_lam, cbar]
  push_cast
  fun_prop

theorem continuousAt_withMu_Psi
    (hX : (A.β : ℂ) * (A.withMu μ₀).Hph k ε (-u μ₀) + A.lam ε * (A.withMu μ₀).cbar ≠ 0) :
    ContinuousAt (fun μ ↦ (A.withMu μ).Psi k ε (u μ)) μ₀ := by
  have hρ := A.continuousAt_withMu_ρ μ₀
  have hH := A.continuousAt_withMu_Hph k ε hu.neg
  have hS := A.continuousAt_withMu_S k hu hoff
  have hD := A.continuousAt_withMu_X k ε hu
  simp only [withMu_β, withMu_lam] at hD
  have hN : ContinuousAt (fun μ ↦ 4 * (A.a : ℂ) * A.β * u μ * (A.β * (A.withMu μ).Hph k ε (-u μ)) +
      A.lam ε ^ 2 * (A.β : ℂ) ^ 2 * (A.withMu μ).ρ / 4) μ₀ := by
    fun_prop
  simp only [Psi, withMu_β, withMu_lam, withMu_a]
  exact (continuousAt_const.mul hS).sub (hN.div hD hX)

end Moving

/-! ### Real roots, and the pole of `S_k` on the lattice -/

/-- `q₀ = a² + ω_k²/4` is real. -/
theorem q₀_eq_ofReal : A.q₀ k = ((A.a ^ 2 + A.ω k ^ 2 / 4 : ℝ) : ℂ) := by
  simp only [q₀, w]
  push_cast
  ring_nf
  rw [I_sq]
  ring

/-- `e` has real coefficients. -/
theorem eph_conj (hε : ε = 1 ∨ ε = -1) (t : ℂ) :
    (starRingEnd ℂ) (A.eph k ε t) = A.eph k ε ((starRingEnd ℂ) t) := by
  rcases hε with rfl | rfl <;>
  · simp only [eph, Hph, lam, A.q₀_eq_ofReal, map_sub, map_add, map_mul, map_pow, map_neg,
      map_one, map_ofNat, conj_ofReal]

theorem conj_shat (hε : ε = 1 ∨ ε = -1) : (starRingEnd ℂ) (A.shat k ε) = A.shat k ε := by
  rcases hε with rfl | rfl <;>
  · simp only [shat, A.q₀_eq_ofReal, map_sub, map_mul, map_pow, map_neg, map_one, map_ofNat,
      conj_ofReal]

theorem conj_shat' (hε : ε = 1 ∨ ε = -1) : (starRingEnd ℂ) (A.shat' k ε) = A.shat' k ε := by
  rcases hε with rfl | rfl <;>
  · simp only [shat', A.q₀_eq_ofReal, map_sub, map_mul, map_pow, map_neg, map_one, map_ofNat,
      map_add, conj_ofReal]

/-- `Im zₙ² = 2μνₙ'` with `νₙ' = (2n + 1 + k)π/β`. -/
theorem im_z_sq (n : ℤ) : (A.z k n ^ 2).im = 2 * A.μ * ((2 * n + 1 + k) * Real.pi / A.β) := by
  rw [A.z_eq]
  simp only [sq, mul_im, add_re, add_im, ofReal_re, ofReal_im, mul_re, I_re, I_im]
  ring

/-- `S_k(x)⁻¹`, as `-(2/β)coth` (even `k`) or `-(2/β)tanh` (odd `k`): it vanishes on the lattice. -/
noncomputable def Sinv (x : ℂ) : ℂ :=
  -(2 / (A.β : ℂ)) * if Even k then coth (A.β * (x - A.μ) / 2) else tanh (A.β * (x - A.μ) / 2)

theorem S_mul_Sinv {x : ℂ} (hs : sinh (A.β * (x - A.μ) / 2) ≠ 0)
    (hc : cosh (A.β * (x - A.μ) / 2) ≠ 0) : A.S k x * A.Sinv k x = 1 := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  simp only [S, Sinv]
  split_ifs <;> simp only [coth, tanh_eq_sinh_div_cosh] <;> field_simp

/-- On the lattice, `cosh(β(zₙ - μ)/2) = 0` for even `k` and `sinh(β(zₙ - μ)/2) = 0` for odd `k`;
the other one is nonzero. -/
theorem cosh_sinh_z (n : ℤ) :
    (Even k → cosh (A.β * (A.z k n - A.μ) / 2) = 0 ∧ sinh (A.β * (A.z k n - A.μ) / 2) ≠ 0) ∧
      (¬Even k → sinh (A.β * (A.z k n - A.μ) / 2) = 0 ∧
        cosh (A.β * (A.z k n - A.μ) / 2) ≠ 0) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hy : (A.β : ℂ) * (A.z k n - A.μ) / 2 = ((2 * n + 1 + k) * Real.pi / 2 : ℂ) * I := by
    rw [A.z_eq]
    push_cast
    field_simp
    ring
  have hid := cosh_sq_sub_sinh_sq ((A.β : ℂ) * (A.z k n - A.μ) / 2)
  refine ⟨fun hk ↦ ?_, fun hk ↦ ?_⟩
  · obtain ⟨j, hj⟩ := hk
    have h0 : cosh ((A.β : ℂ) * (A.z k n - A.μ) / 2) = 0 := by
      rw [hy, cosh_mul_I, cos_eq_zero_iff]
      exact ⟨n + j, by rw [hj]; push_cast; ring⟩
    refine ⟨h0, fun h1 ↦ ?_⟩
    rw [h0, h1] at hid
    norm_num at hid
  · obtain ⟨j, hj⟩ := Int.not_even_iff_odd.mp hk
    have h0 : sinh ((A.β : ℂ) * (A.z k n - A.μ) / 2) = 0 := by
      rw [hy, sinh_mul_I, mul_eq_zero]
      refine Or.inl (sin_eq_zero_iff.mpr ⟨n + j + 1, by rw [hj]; push_cast; ring⟩)
    refine ⟨h0, fun h1 ↦ ?_⟩
    rw [h0, h1] at hid
    norm_num at hid

/-- `S_k⁻¹` at a moving point that tends to a lattice point is continuous there, with value `0`. -/
theorem continuousAt_withMu_Sinv {μ₀ : ℝ} {u : ℝ → ℂ} (hu : ContinuousAt u μ₀) {c : ℤ}
    (hc : u μ₀ = (A.withMu μ₀).z k c) :
    ContinuousAt (fun μ ↦ (A.withMu μ).Sinv k (u μ)) μ₀ ∧ (A.withMu μ₀).Sinv k (u μ₀) = 0 := by
  have hcs := (A.withMu μ₀).cosh_sinh_z k c
  rw [← hc] at hcs
  have hy : ContinuousAt (fun μ ↦ (A.β : ℂ) * (u μ - μ) / 2) μ₀ := by fun_prop
  have hsh := continuous_sinh.continuousAt.comp hy
  have hch := continuous_cosh.continuousAt.comp hy
  simp only [Sinv, withMu_β, withMu_μ] at hcs ⊢
  by_cases hk : Even k
  · obtain ⟨h0, h1⟩ := hcs.1 hk
    simp only [hk, ↓reduceIte, coth, h0, zero_div, mul_zero, and_true]
    exact continuousAt_const.mul (hch.div hsh h1)
  · obtain ⟨h0, h1⟩ := hcs.2 hk
    simp only [hk, ↓reduceIte, tanh_eq_sinh_div_cosh, h0, zero_div, mul_zero, and_true]
    exact continuousAt_const.mul (hsh.div hch h1)

/-! ### `K = 0` is a divergence, and `e(0) ≠ 0` near `μ = 0` -/

/-- **`K = 0` at a root is a divergence**: for `μ ≠ 0`, at a root `u ≠ 0` of `e` with `±u` off the
lattice, `P(u) ≠ 0` and `K(u) = 0`, the mode `φ_u(t) = 2uβH(-u)/(t² - u²)` is a null vector of `χ_r`
(Step 3, `hasSum_mode_gen`), so `1 + 𝒦` is not invertible. -/
theorem not_isUnit_onePlusKR_of_Kph (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {u : ℂ} (he : A.eph k ε u = 0) (hu0 : u ≠ 0) (hoff : ∀ l, A.z k l ≠ u)
    (hoff' : ∀ l, A.z k l ≠ -u) (hPu : A.Pph k u ≠ 0) (hK : A.Kph k ε u = 0) :
    ¬IsUnit (A.onePlusKR k ε) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hl := A.lam_ne_zero ε hε ha
  -- `H(-u) ≠ 0`, as `H(u)H(-u) = λ²p₀p₂ ≠ 0`
  have hH : A.Hph k ε (-u) ≠ 0 := by
    intro h0
    rw [eph, h0, mul_zero, zero_sub, neg_eq_zero] at he
    exact mul_ne_zero (mul_ne_zero (pow_ne_zero 2 hl) (ofReal_ne_zero.mpr A.p₀_pos.ne'))
      (ofReal_ne_zero.mpr A.p₂_pos.ne') he
  have hα : A.α k u ≠ 0 := by
    intro h0
    rw [Kph, h0, mul_zero, add_zero] at hK
    exact mul_ne_zero hβ hH hK
  rw [A.isUnit_onePlusKR_iff k ε]
  intro hnull
  have hz : ∀ l, A.z k l ^ 2 - u ^ 2 ≠ 0 := fun l h0 ↦ by
    rcases sq_eq_sq_iff_eq_or_eq_neg.mp (sub_eq_zero.mp h0) with h | h
    exacts [hoff l h, hoff' l h]
  -- `φ_u(zₗ) = 2uβH(-u)/(zₗ² - u²)`
  have hmode : ∀ l, A.mode k ε u (A.z k l) =
      2 * u * (A.β * A.Hph k ε (-u)) / (A.z k l ^ 2 - u ^ 2) := by
    intro l
    have h1 := sub_ne_zero.mpr (hoff l)
    have h2 : A.z k l + u ≠ 0 := fun h0 ↦ hoff' l (by linear_combination h0)
    have hla : A.lam ε * A.α k u = -(A.β * A.Hph k ε (-u)) := by
      rw [Kph] at hK
      linear_combination hK
    have h3 := hz l
    rw [mode, hla]
    field_simp
    ring
  have h0 := congrFun (hnull (fun l ↦ A.mode k ε u (A.z k l)) ?_ fun n ↦ ?_) 0
  · rw [hmode, Pi.zero_apply] at h0
    exact div_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero hu0) (mul_ne_zero hβ hH)) (hz 0) h0
  · -- the class: `φ_u(zₗ) = 2uβH(-u)y²/(1 - u²y²)`, `y = 1/zₗ`, is bounded
    obtain ⟨C, hC⟩ := A.bdd_of_inv_zOp k (f := fun l ↦ A.mode k ε u (A.z k l))
      (fun y ↦ 2 * u * (A.β * A.Hph k ε (-u)) * y ^ 2 / (1 - u ^ 2 * y ^ 2))
      (by fun_prop (disch := simp)) ((A.eventually_zOp_eq k).mono fun l hl ↦ by
        have h0 : A.z k l ≠ 0 := hl ▸ A.zOp_ne_zero k l
        have h1 : 1 - u ^ 2 * (A.z k l)⁻¹ ^ 2 ≠ 0 := by
          rw [show 1 - u ^ 2 * (A.z k l)⁻¹ ^ 2 = (A.z k l ^ 2 - u ^ 2) / A.z k l ^ 2 by
            field_simp]
          exact div_ne_zero (hz l) (pow_ne_zero 2 h0)
        rw [hmode, hl]
        field_simp)
    exact (A.opLatticeAll k).summable_sq_of_norm_mul_z_le (C := C) fun l ↦ by
      change ‖A.zOp k l * (A.mode k ε u (A.z k l) / A.zOp k l)‖ ≤ C
      rw [mul_div_cancel₀ _ (A.zOp_ne_zero k l)]
      exact hC l
  · have hS := A.hasSum_mode_gen k ε hε ha hμ hoff hoff' hPu hα n
    rw [hK, he, mul_zero, zero_mul, zero_mul, add_zero] at hS
    have hPn := A.Pph_z_ne_zero k n
    convert hS.div_const (A.Pph k (A.z k n)) using 1
    · funext l
      rw [Rrow, ← A.Pph_mul_chiR k ε hμ n l]
      field_simp
    · simp

/-- `1 + 𝒦` stays invertible near `μ₀` in `μ` (`ContHyp.eventually_isUnit`, `contHyp_R`). -/
theorem eventually_isUnit_onePlusKR (hu : IsUnit (A.onePlusKR k ε)) :
    ∀ᶠ μ in 𝓝 A.μ, IsUnit ((A.withMu μ).onePlusKR k ε) := by
  have hH := contHyp_R (U := fun _ ↦ A.U) (β := fun _ ↦ A.β) (μ := id) (hβ := fun _ ↦ A.β_pos)
    (x₀ := A.μ) (k := k) (ε := ε) continuousAt_const continuousAt_const continuousAt_id
  filter_upwards [hH.eventually_isUnit (A.chiDecay_chiR_all k ε) hu] with μ hμ
  exact hμ ((A.withMu μ).chiDecay_chiR_all k ε)

/-- `p₁ + s p₀e^{βμ} = (e^{βU/4} + s e^{-βU/4})/(2(e^{-βU/4}cosh βμ + e^{βU/4}))`. -/
theorem p₁_add_p₀_exp (ν s : ℝ) :
    (A.withMu ν).p₁ + s * ((A.withMu ν).p₀ * Real.exp (A.β * ν)) =
      (Real.exp (A.β * A.U / 4) + s * Real.exp (-(A.β * A.U / 4))) /
        (2 * (Real.exp (-(A.β * A.U / 4)) * Real.cosh (A.β * ν) + Real.exp (A.β * A.U / 4))) := by
  have h₀ : (A.withMu ν).boltz₀ = Real.exp (-(A.β * A.U / 4)) * (Real.exp (A.β * ν))⁻¹ := by
    rw [← Real.exp_neg, ← Real.exp_add]; simp only [boltz₀, withMu]; ring_nf
  have h₁ : (A.withMu ν).boltz₁ = Real.exp (A.β * A.U / 4) := by
    simp only [boltz₁, withMu]; ring_nf
  have h₂ : (A.withMu ν).boltz₂ = Real.exp (-(A.β * A.U / 4)) * Real.exp (A.β * ν) := by
    rw [← Real.exp_add]; simp only [boltz₂, withMu]; ring_nf
  have he := Real.exp_pos (A.β * ν)
  have hZ := (A.withMu ν).Z_pos
  simp only [p₁, p₀, Z, h₀, h₁, h₂] at hZ ⊢
  rw [Real.cosh_eq]
  simp only [Real.exp_neg] at hZ ⊢
  field_simp
  ring

/-- `p₁ ± p₀e^{βμ}` takes its value at `μ = 0` nowhere else (for `-`, as `U ≠ 0`). -/
theorem p₁_add_p₀_exp_ne (ha : A.a ≠ 0) {s : ℝ} (hs : s = 1 ∨ s = -1) {ν : ℝ} (hν : ν ≠ 0) :
    (A.withMu ν).p₁ + s * ((A.withMu ν).p₀ * Real.exp (A.β * ν)) ≠
      (A.withMu 0).p₁ + s * ((A.withMu 0).p₀ * Real.exp (A.β * 0)) := by
  rw [p₁_add_p₀_exp, p₁_add_p₀_exp, mul_zero, Real.cosh_zero, mul_one]
  have hm := Real.exp_pos (-(A.β * A.U / 4))
  have hM := Real.exp_pos (A.β * A.U / 4)
  have hnum : Real.exp (A.β * A.U / 4) + s * Real.exp (-(A.β * A.U / 4)) ≠ 0 := by
    rcases hs with rfl | rfl
    · positivity
    · intro h0
      have h1 : A.β * A.U / 4 = -(A.β * A.U / 4) := Real.exp_injective (by linarith)
      have hU : A.U ≠ 0 := fun h ↦ ha (by simp [a, h])
      have := A.β_pos
      have : A.β * A.U = 0 := by linarith
      exact hU ((mul_eq_zero.mp this).resolve_left A.β_pos.ne')
  have hc : Real.cosh (A.β * ν) ≠ 1 := by
    have := Real.one_lt_cosh.mpr (mul_ne_zero A.β_pos.ne' hν)
    exact this.ne'
  have hd := Real.cosh_pos (A.β * ν)
  intro h0
  rw [div_eq_div_iff (by positivity) (by positivity)] at h0
  have h1 := mul_left_cancel₀ hnum h0
  exact hc (mul_left_cancel₀ hm.ne' (by linarith))

/-- `e(0) = (q₀ + λ(p₁ - p₀e^{βμ}))(q₀ + λ(p₁ + p₀e^{βμ}))`, as `p₀p₂ = (p₀e^{βμ})²`. -/
theorem eph_zero_eq (ν : ℝ) : (A.withMu ν).eph k ε 0 =
    (A.q₀ k + A.lam ε *
      (((A.withMu ν).p₁ + -1 * ((A.withMu ν).p₀ * Real.exp (A.β * ν)) : ℝ) : ℂ)) *
    (A.q₀ k + A.lam ε *
      (((A.withMu ν).p₁ + 1 * ((A.withMu ν).p₀ * Real.exp (A.β * ν)) : ℝ) : ℂ)) := by
  have h₂ : (A.withMu ν).p₂ = (A.withMu ν).p₀ * Real.exp (A.β * ν) ^ 2 := by
    rw [p₂_eq_p₀_mul, ← Real.exp_nat_mul]
    simp only [withMu]
    ring_nf
  simp only [eph, Hph, lam, h₂, neg_zero, show (A.withMu ν).q₀ k = A.q₀ k from rfl, withMu_a]
  push_cast
  ring

/-- **`e(0) ≠ 0` at every `μ ≠ 0` near `0`**: each factor of `eph_zero_eq` is `q₀ + λg(μ)` with `g`
continuous and taking its value at `0` nowhere else (`p₁_add_p₀_exp_ne`). -/
theorem eventually_eph_zero_ne (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0) :
    ∀ᶠ ν in 𝓝[≠] A.μ, (A.withMu ν).eph k ε 0 ≠ 0 := by
  have hl := A.lam_ne_zero ε hε ha
  have ha' : A.a ≠ 0 := by exact_mod_cast ha
  have key : ∀ s : ℝ, (s = 1 ∨ s = -1) → ∀ᶠ ν in 𝓝[≠] A.μ, A.q₀ k + A.lam ε *
      (((A.withMu ν).p₁ + s * ((A.withMu ν).p₀ * Real.exp (A.β * ν)) : ℝ) : ℂ) ≠ 0 := by
    intro s hs
    have hF : ContinuousAt (fun ν : ℝ ↦ A.q₀ k + A.lam ε *
        (((A.withMu ν).p₁ + s * ((A.withMu ν).p₀ * Real.exp (A.β * ν)) : ℝ) : ℂ)) A.μ := by
      simp only [p₁_add_p₀_exp]
      have : ∀ ν : ℝ, 0 < 2 * (Real.exp (-(A.β * A.U / 4)) * Real.cosh (A.β * ν) +
          Real.exp (A.β * A.U / 4)) := fun ν ↦ by
        have := Real.cosh_pos (A.β * ν)
        positivity
      fun_prop (disch := exact (this _).ne')
    by_cases h0 : A.q₀ k + A.lam ε *
        (((A.withMu A.μ).p₁ + s * ((A.withMu A.μ).p₀ * Real.exp (A.β * A.μ)) : ℝ) : ℂ) = 0
    · filter_upwards [self_mem_nhdsWithin] with ν hν hF0
      have hν0 : ν ≠ 0 := by rw [← hμ]; exact hν
      apply A.p₁_add_p₀_exp_ne ha' hs hν0
      rw [← hμ]
      exact_mod_cast mul_left_cancel₀ hl (add_left_cancel (hF0.trans h0.symm))
    · exact nhdsWithin_le_nhds (hF.eventually_ne h0)
  filter_upwards [key (-1) (Or.inr rfl), key 1 (Or.inl rfl)] with ν h₁ h₂
  rw [eph_zero_eq]
  exact mul_ne_zero h₁ h₂

/-! ### The assumptions of Theorem "density and magnetic" near `μ = 0` -/

variable {A k ε}

/-- At `μ = 0` with `ŝ' ≠ 0` for odd `k`, `zₙ² ≠ ŝ'` for all `n` if it holds off the mirror fixed
points, where `zₙ = 0` (odd `k`). -/
theorem z_sq_ne'_of (hμ : A.μ = 0) (hs' : Odd k → A.shat' k ε ≠ 0)
    (hodd : ∀ n, n ≠ mirror k n → A.z k n ^ 2 - A.shat' k ε ≠ 0) (n : ℤ) :
    A.z k n ^ 2 - A.shat' k ε ≠ 0 := by
  by_cases hn : n = mirror k n
  · rw [A.z_of_fixed k hn.symm, hμ]
    simpa using hs' ⟨-n - 1, by unfold mirror at hn; omega⟩
  · exact hodd n hn

/-- At `μ = 0`, `p₀ + p₁ = 1/2`. -/
theorem p₀_add_p₁_of_mu (hμ : A.μ = 0) : (A.p₀ : ℂ) + A.p₁ = 1 / 2 := by
  have := A.weights_sum'
  rw [p₂_eq_p₀_of_mu hμ] at this
  linear_combination this / 2

/-- At `μ = 0` and `v² = ŝ'`: `P(v) = 4a²(a²(1 + 2ε) + q₀)`, as `ŝ' = -2εa² - q₀`. -/
theorem Pph_of_shat' (hε : ε = 1 ∨ ε = -1) (hμ : A.μ = 0) {v : ℂ} (hv : v ^ 2 = A.shat' k ε) :
    A.Pph k v = 4 * A.a ^ 2 * (A.a ^ 2 * (1 + 2 * ε) + A.q₀ k) := by
  have hs := p₀_add_p₁_of_mu hμ
  have hε2 : ε ^ 2 = 1 := by rcases hε with rfl | rfl <;> norm_num
  have hs' : A.shat' k ε = -2 * ε * A.a ^ 2 - A.q₀ k := by
    rw [shat']
    linear_combination (-4 * ε * (A.a : ℂ) ^ 2) * hs
  rw [A.Pph_eq_poly, show v ^ 4 = (v ^ 2) ^ 2 by ring, hv, hs', κ]
  linear_combination 4 * (A.a : ℂ) ^ 4 * hε2

/-- At `μ = 0`, `P ≠ 0` at the roots of `ŝ'`, except in the static magnetic channel (`k = 0`,
`ε = -1`), where `ŝ' = a²` is the atomic root. -/
theorem Pph_ne_of_shat' (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0)
    (hk : ¬(k = 0 ∧ ε = -1)) {v : ℂ} (hv : v ^ 2 = A.shat' k ε) : A.Pph k v ≠ 0 := by
  have hq := A.q₀_eq_ofReal k
  have ha' : A.a ≠ 0 := by exact_mod_cast ha
  rw [Pph_of_shat' hε hμ hv]
  refine mul_ne_zero (mul_ne_zero (by norm_num) (pow_ne_zero 2 ha)) ?_
  rcases hε with rfl | rfl
  · rw [hq, show (A.a : ℂ) ^ 2 * (1 + 2 * 1) = ((3 * A.a ^ 2 : ℝ) : ℂ) by push_cast; ring,
      ← ofReal_add, ofReal_ne_zero]
    positivity
  · have hk0 : k ≠ 0 := fun hk0 ↦ hk ⟨hk0, rfl⟩
    have hω : A.ω k ≠ 0 := by
      simp only [ω]
      have := Real.pi_pos
      have := A.β_pos
      have : (k : ℝ) ≠ 0 := by exact_mod_cast hk0
      positivity
    rw [hq, show (A.a : ℂ) ^ 2 * (1 + 2 * -1) = ((-A.a ^ 2 : ℝ) : ℂ) by push_cast; ring,
      ← ofReal_add, ofReal_ne_zero]
    have := pow_pos (abs_pos.mpr hω) 2
    rw [sq_abs] at this
    linarith

/-- In the static magnetic channel at `μ = 0`, `ŝ' = a²`. -/
theorem shat'_static (hμ : A.μ = 0) : A.shat' 0 (-1) = A.a ^ 2 := by
  have hs := p₀_add_p₁_of_mu hμ
  rw [shat', q₀, w_zero]
  linear_combination (4 * (A.a : ℂ) ^ 2) * hs

/-- Near `μ = 0`, a root `u₂(μ)` of `e` with `u₂(0)² = ŝ'` and `P(u₂(0)) ≠ 0` has `P(u₂) ≠ 0` and
`K(u₂) ≠ 0`, as `K(u₂) → -2λβp₀` (`Kph_of_mu'`). -/
theorem eventually_modeOK (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0)
    (hz' : ∀ n, A.z k n ^ 2 - A.shat' k ε ≠ 0) {u₂ : ℝ → ℂ} (hu₂ : u₂ A.μ ^ 2 = A.shat' k ε)
    (hc₂ : ContinuousAt u₂ A.μ) (hP : A.Pph k (u₂ A.μ) ≠ 0) :
    ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).ModeOK k ε (u₂ μ) := by
  have hl := A.lam_ne_zero ε hε ha
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hp : (A.p₀ : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.p₀_pos.ne'
  have hoff : ∀ n, (A.withMu A.μ).z k n ≠ u₂ A.μ := fun n hn ↦
    hz' n (by rw [← hu₂, ← hn, withMu_self, sub_self])
  have hcP : ContinuousAt (fun μ ↦ A.Pph k (u₂ μ)) A.μ := by
    simp only [Pph, Qp, Qm]
    fun_prop
  have hK := (A.continuousAt_withMu_Kph k ε hc₂ hoff).eventually_ne (by
    change A.Kph k ε (u₂ A.μ) ≠ 0
    rw [A.Kph_of_mu' k ε hμ hu₂]
    exact mul_ne_zero (mul_ne_zero (mul_ne_zero (by norm_num) hl) hβ) hp)
  filter_upwards [hcP.eventually_ne hP, hK] with μ hPμ hKμ
  exact Or.inl ⟨hPμ, hKμ⟩

/-- **The second root near `μ = 0`**: for `ŝ'` off the lattice, a root `u₂(μ)` of `e`, continuous at
`0` with `u₂(0)² = ŝ'`, and `ModeOK` near `0`: in the static magnetic channel (`k = 0`, `ε = -1`)
the atomic root `u₂ = a`, at every `μ`; otherwise a continuation of `√ŝ'` (`exists_root_of_mu`),
with `P(u₂) ≠ 0` (`Pph_ne_of_shat'`) and `K(u₂) → -2λβp₀`. -/
theorem exists_root₂ (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0)
    (hz' : ∀ n, A.z k n ^ 2 - A.shat' k ε ≠ 0) :
    ∃ u₂ : ℝ → ℂ, u₂ A.μ ^ 2 = A.shat' k ε ∧ ContinuousAt u₂ A.μ ∧
      (∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).eph k ε (u₂ μ) = 0) ∧
      ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).ModeOK k ε (u₂ μ) := by
  by_cases hk : k = 0 ∧ ε = -1
  · obtain ⟨rfl, rfl⟩ := hk
    exact ⟨fun _ ↦ A.a, (shat'_static hμ).symm, continuousAt_const,
      Eventually.of_forall fun μ ↦ (atomic_root (A := A.withMu μ)).1,
      Eventually.of_forall fun μ ↦ (atomic_root (A := A.withMu μ)).2⟩
  have hss := A.shat_sub_shat'_ne k ε hε ha
  set v : ℂ := A.shat' k ε ^ (2⁻¹ : ℂ)
  have hv : v ^ 2 = A.shat' k ε := by simp [v]
  obtain ⟨u₂, hu₂, hc₂, he₂⟩ := A.exists_root_of_mu k ε hμ
    (by rw [A.eph_of_mu' k ε hμ, hv, sub_self, mul_zero])
    (fun h0 ↦ hss (by rw [hv] at h0; linear_combination -h0))
  have hu₂' : u₂ A.μ ^ 2 = A.shat' k ε := hu₂ ▸ hv
  exact ⟨u₂, hu₂', hc₂, he₂,
    eventually_modeOK hε ha hμ hz' hu₂' hc₂ (Pph_ne_of_shat' hε ha hμ hk hu₂')⟩

/-- At `μ = 0` and `u² = ŝ`: `βH(-u) + λc̄ = 2λβp₀ ≠ 0`. -/
theorem X_ne_of_mu (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0) {u : ℂ}
    (hu : u ^ 2 = A.shat k ε) : (A.β : ℂ) * A.Hph k ε (-u) + A.lam ε * A.cbar ≠ 0 := by
  have hH := A.Hph_sub_of_mu k ε hμ (-u)
  rw [neg_sq, hu, sub_self, sub_eq_zero] at hH
  rw [show (A.β : ℂ) * A.Hph k ε (-u) + A.lam ε * A.cbar = 2 * A.lam ε * A.β * A.p₀ by
    rw [hH]; simp only [cbar, p₂_eq_p₀_of_mu hμ]; push_cast; ring]
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero (A.lam_ne_zero ε hε ha))
    (ofReal_ne_zero.mpr A.β_pos.ne')) (ofReal_ne_zero.mpr A.p₀_pos.ne')

/-- **The roots stay real**: near `μ₀`, if `e = (t² - u₁²)(t² - u₂²)` and `conj(u₁(μ₀)²) ≠ u₂(μ₀)²`,
then `u₁²` is real, as `conj(u₁²)` is a root of the real polynomial `e` in `s = t²`. -/
theorem eventually_conj_sq (hε : ε = 1 ∨ ε = -1) {μ₀ : ℝ} {u₁ u₂ : ℝ → ℂ}
    (hc₁ : ContinuousAt u₁ μ₀) (hc₂ : ContinuousAt u₂ μ₀)
    (he₁ : ∀ᶠ μ in 𝓝 μ₀, (A.withMu μ).eph k ε (u₁ μ) = 0)
    (he₂ : ∀ᶠ μ in 𝓝 μ₀, (A.withMu μ).eph k ε (u₂ μ) = 0)
    (hsq : ∀ᶠ μ in 𝓝 μ₀, u₁ μ ^ 2 ≠ u₂ μ ^ 2) (h₀ : (starRingEnd ℂ) (u₁ μ₀ ^ 2) ≠ u₂ μ₀ ^ 2) :
    ∀ᶠ μ in 𝓝 μ₀, (starRingEnd ℂ) (u₁ μ ^ 2) = u₁ μ ^ 2 := by
  have hc : ContinuousAt (fun μ ↦ (starRingEnd ℂ) (u₁ μ ^ 2) - u₂ μ ^ 2) μ₀ :=
    (continuous_conj.continuousAt.comp (hc₁.pow 2)).sub (hc₂.pow 2)
  filter_upwards [hc.eventually_ne (sub_ne_zero.mpr h₀), he₁, he₂, hsq] with μ hne he₁ he₂ hsq
  have h0 := (A.withMu μ).eph_eq_mul k ε he₁ he₂ hsq ((starRingEnd ℂ) (u₁ μ))
  rw [← (A.withMu μ).eph_conj k ε hε, he₁, map_zero, ← map_pow] at h0
  rcases mul_eq_zero.mp h0.symm with h | h
  · exact sub_eq_zero.mp h
  · exact absurd h hne

/-- **Off the lattice for `μ ≠ 0`**: for roots `u₁(μ)`, `u₂(μ)` of `e` with `u₁(0)² = ŝ` and
`u₂(0)² = ŝ'`, and `ŝ, ŝ' ≠ 0` if there is a mirror fixed point (odd `k`), the squares `u₁²`, `u₂²`
stay distinct and real near `0` (`eventually_conj_sq`), while `Im zₙ² = 2μνₙ' ≠ 0` off the mirror
fixed points (`im_z_sq`) and `zₙ² = μ² → 0` at them; so `zₙ² ≠ u₁², u₂²` for all `n` at every
`μ ≠ 0` near `0`. -/
theorem eventually_z_sq_ne (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0)
    (hs0 : Odd k → A.shat k ε ≠ 0) (hs' : Odd k → A.shat' k ε ≠ 0) {u₁ u₂ : ℝ → ℂ}
    (hu₁ : u₁ A.μ ^ 2 = A.shat k ε) (hc₁ : ContinuousAt u₁ A.μ)
    (he₁ : ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).eph k ε (u₁ μ) = 0) (hu₂ : u₂ A.μ ^ 2 = A.shat' k ε)
    (hc₂ : ContinuousAt u₂ A.μ) (he₂ : ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).eph k ε (u₂ μ) = 0) :
    (∀ᶠ μ in 𝓝 A.μ, u₁ μ ^ 2 ≠ u₂ μ ^ 2) ∧ ∀ᶠ μ in 𝓝[≠] A.μ, ∀ n,
      (A.withMu μ).z k n ^ 2 - u₁ μ ^ 2 ≠ 0 ∧ (A.withMu μ).z k n ^ 2 - u₂ μ ^ 2 ≠ 0 := by
  have hss := A.shat_sub_shat'_ne k ε hε ha
  have hsq : ∀ᶠ μ in 𝓝 A.μ, u₁ μ ^ 2 ≠ u₂ μ ^ 2 := by
    have hc : ContinuousAt (fun μ ↦ u₁ μ ^ 2 - u₂ μ ^ 2) A.μ := by fun_prop
    filter_upwards [hc.eventually_ne (by rw [hu₁, hu₂]; exact hss)] with μ hμ'
    exact fun h0 ↦ hμ' (by rw [h0, sub_self])
  have hr₁ := eventually_conj_sq (A := A) (k := k) hε hc₁ hc₂ he₁ he₂ hsq (by
    rw [hu₁, hu₂, A.conj_shat k ε hε]
    exact sub_ne_zero.mp hss)
  have hr₂ := eventually_conj_sq (A := A) (k := k) hε hc₂ hc₁ he₂ he₁
    (hsq.mono fun _ h ↦ h.symm) (by
      rw [hu₁, hu₂, A.conj_shat' k ε hε]
      exact fun h0 ↦ hss (by rw [h0, sub_self]))
  -- at the mirror fixed point `zₙ = μ`, `μ² - uⱼ² → -ŝ, -ŝ'`
  have hfix : ∀ {w : ℝ → ℂ}, ContinuousAt w A.μ → w A.μ ^ 2 ≠ 0 →
      ∀ᶠ μ : ℝ in 𝓝 A.μ, (μ : ℂ) ^ 2 - w μ ^ 2 ≠ 0 := fun {w} hw hw0 ↦ by
    have hcw : ContinuousAt (fun μ : ℝ ↦ (μ : ℂ) ^ 2 - w μ ^ 2) A.μ := by fun_prop
    refine hcw.eventually_ne ?_
    rw [show ((A.μ : ℝ) : ℂ) = 0 by simp [hμ], zero_pow two_ne_zero, zero_sub, neg_ne_zero]
    exact hw0
  have hfix' : ∀ {w : ℝ → ℂ}, ContinuousAt w A.μ → (Odd k → w A.μ ^ 2 ≠ 0) →
      ∀ᶠ μ : ℝ in 𝓝 A.μ, Odd k → (μ : ℂ) ^ 2 - w μ ^ 2 ≠ 0 := fun {w} hw hw0 ↦ by
    by_cases hk : Odd k
    · exact (hfix hw (hw0 hk)).mono fun _ h _ ↦ h
    · exact Eventually.of_forall fun _ hk' ↦ absurd hk' hk
  refine ⟨hsq, ?_⟩
  filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds hr₁, nhdsWithin_le_nhds hr₂,
    nhdsWithin_le_nhds (hfix' hc₁ (by rw [hu₁]; exact hs0)),
    nhdsWithin_le_nhds (hfix' hc₂ (by rw [hu₂]; exact hs'))] with μ hμ' hr₁ hr₂ hf₁ hf₂ n
  by_cases hn : mirror k n = n
  · have hk : Odd k := ⟨-n - 1, by unfold mirror at hn; omega⟩
    rw [(A.withMu μ).z_of_fixed k hn, withMu_μ]
    exact ⟨hf₁ hk, hf₂ hk⟩
  · have him : ((A.withMu μ).z k n ^ 2).im ≠ 0 := by
      rw [im_z_sq, withMu_μ, withMu_β]
      have h1 : (2 * n + 1 + k : ℝ) ≠ 0 := by
        have : 2 * n + 1 + k ≠ 0 := fun h0 ↦ hn (by unfold mirror; omega)
        exact_mod_cast this
      have h2 : μ ≠ 0 := by rw [← hμ]; exact hμ'
      have := Real.pi_pos
      have := A.β_pos
      positivity
    refine ⟨fun h0 ↦ him ?_, fun h0 ↦ him ?_⟩
    · rw [sub_eq_zero.mp h0]
      exact conj_eq_iff_im.mp hr₁
    · rw [sub_eq_zero.mp h0]
      exact conj_eq_iff_im.mp hr₂

/-- **The assumptions of Theorem "density and magnetic" near `μ = 0`**, for roots `u₁(μ)`, `u₂(μ)`
of `e` with `u₁(0)² = ŝ`, `u₂(0)² = ŝ'` (`ŝ, ŝ' ≠ 0` for odd `k`), `ModeOK` at `u₂` near `0`, and
`Ψ(u₁) ≠ 0` at every `μ ≠ 0` near `0`: then `K(u₁) = ρΨ(u₁) ≠ 0` (`Kph_eq_rho_mul`), and the rest
is `eventually_z_sq_ne` and continuity. -/
theorem eventually_dmGeneric_of (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0)
    (hs0 : Odd k → A.shat k ε ≠ 0) (hs' : Odd k → A.shat' k ε ≠ 0) {u₁ u₂ : ℝ → ℂ}
    (hu₁ : u₁ A.μ ^ 2 = A.shat k ε) (hc₁ : ContinuousAt u₁ A.μ)
    (he₁ : ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).eph k ε (u₁ μ) = 0) (hu₂ : u₂ A.μ ^ 2 = A.shat' k ε)
    (hc₂ : ContinuousAt u₂ A.μ) (he₂ : ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).eph k ε (u₂ μ) = 0)
    (hm₂ : ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).ModeOK k ε (u₂ μ))
    (hΨ : ∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).Psi k ε (u₁ μ) ≠ 0) :
    ∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).DMGeneric k ε (u₁ μ) (u₂ μ) := by
  obtain ⟨hsq, hz⟩ := eventually_z_sq_ne hε ha hμ hs0 hs' hu₁ hc₁ he₁ hu₂ hc₂ he₂
  have hcP : ContinuousAt (fun μ ↦ A.Pph k (u₁ μ)) A.μ := by
    simp only [Pph, Qp, Qm]
    fun_prop
  have hP₁ := hcP.eventually_ne (A.Pph_ne_of_shat k ε hε ha hμ hu₁)
  have hX := (A.continuousAt_withMu_X k ε hc₁).eventually_ne (X_ne_of_mu hε ha hμ hu₁)
  filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds he₁, nhdsWithin_le_nhds he₂,
    nhdsWithin_le_nhds hsq, hz, nhdsWithin_le_nhds hP₁, nhdsWithin_le_nhds hm₂,
    nhdsWithin_le_nhds hX, hΨ] with μ hμ' he₁ he₂ hsq hz hP₁ hm₂ hX hΨ
  have hμ0 : (A.withMu μ).μ ≠ 0 := by rw [withMu_μ, ← hμ]; exact hμ'
  refine ⟨hε, ha, hμ0, fun n ↦ ?_, he₁, he₂, hsq, Or.inl ⟨hP₁, ?_⟩, hm₂⟩
  · rw [(A.withMu μ).eph_eq_mul k ε he₁ he₂ hsq]
    exact mul_ne_zero (hz n).1 (hz n).2
  · rw [(A.withMu μ).Kph_eq_rho_mul k ε he₁ hX]
    exact mul_ne_zero ((A.withMu μ).rho_ne_zero hμ0) hΨ

/-- Under `HFHyp`, `ŝ ≠ 0` for odd `k`: there `z = 0` is a lattice point at `μ = 0`. -/
theorem HFHyp.shat_ne_of_odd {u : ℂ} (h : A.HFHyp k ε u) (hk : Odd k) : A.shat k ε ≠ 0 := by
  obtain ⟨m, hm⟩ := hk
  have hz := h.z_sq_ne (-m - 1)
  rwa [A.z_of_fixed k (by unfold mirror; omega), h.μ_eq, ofReal_zero, zero_pow two_ne_zero,
    zero_sub, neg_ne_zero] at hz

/-- **The assumptions of Theorem "density and magnetic" hold near half filling**: under `HFHyp`
with `ŝ' ≠ 0` for odd `k`, there are roots `u₁(μ)`, `u₂(μ)` of `e`, continuous at `μ = 0` with
`u₁(0) = û`, `u₂(0)² = ŝ'`, such that `DMGeneric` holds at every `μ ≠ 0` near `0`. For `û ≠ 0`,
`Ψ(u₁)` is continuous with `Ψ(û) = -2aûD̂ ≠ 0` (`Psi_of_mu`). For `û = 0` (even `k`),
`Ψ(u₁) → 0`; there `K(u₁) ≠ 0` as `1 + 𝒦` stays invertible (`eventually_isUnit_onePlusKR`),
while `K(u₁) = 0` would make `φ_{u₁}` a null vector (`not_isUnit_onePlusKR_of_Kph`; `u₁ ≠ 0` by
`eventually_eph_zero_ne`). In the static magnetic channel (`k = 0`, `ε = -1`), `u₂ = a` is the
atomic root, at every `μ`. -/
theorem HFHyp.exists_eventually_dmGeneric {u : ℂ} (h : A.HFHyp k ε u)
    (hs' : Odd k → A.shat' k ε ≠ 0) :
    ∃ u₁ u₂ : ℝ → ℂ, u₁ A.μ = u ∧ u₂ A.μ ^ 2 = A.shat' k ε ∧ ContinuousAt u₁ A.μ ∧
      ContinuousAt u₂ A.μ ∧ ∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).DMGeneric k ε (u₁ μ) (u₂ μ) := by
  have hμ := h.μ_eq
  have hl := A.lam_ne_zero ε h.ε_eq h.a_ne
  have hss := A.shat_sub_shat'_ne k ε h.ε_eq h.a_ne
  obtain ⟨u₂, hu₂, hc₂, he₂, hm₂⟩ :=
    exists_root₂ h.ε_eq h.a_ne hμ (z_sq_ne'_of hμ hs' h.odd_ne)
  obtain ⟨u₁, hu₁, hc₁, he₁⟩ := A.exists_root_of_mu k ε hμ
    (by rw [A.eph_of_mu' k ε hμ, h.u_sq, sub_self, zero_mul])
    (fun h0 ↦ hss (by rw [h.u_sq] at h0; linear_combination h0))
  have hu₁' : u₁ A.μ ^ 2 = A.shat k ε := by rw [hu₁, h.u_sq]
  by_cases hu : u = 0
  · -- `û = 0`: `K(u₁) ≠ 0` from the invertibility of `1 + 𝒦`
    subst hu
    refine ⟨u₁, u₂, hu₁, hu₂, hc₁, hc₂, eventually_dmGeneric_of h.ε_eq h.a_ne hμ h.shat_ne_of_odd
      hs' hu₁' hc₁ he₁ hu₂ hc₂ he₂ hm₂ ?_⟩
    have hcP : ContinuousAt (fun μ ↦ A.Pph k (u₁ μ)) A.μ := by
      simp only [Pph, Qp, Qm]
      fun_prop
    filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds he₁,
      (eventually_z_sq_ne h.ε_eq h.a_ne hμ h.shat_ne_of_odd hs' hu₁' hc₁ he₁ hu₂ hc₂ he₂).2,
      nhdsWithin_le_nhds ((A.continuousAt_withMu_X k ε hc₁).eventually_ne
        (X_ne_of_mu h.ε_eq h.a_ne hμ hu₁')),
      nhdsWithin_le_nhds (hcP.eventually_ne (A.Pph_ne_of_shat k ε h.ε_eq h.a_ne hμ hu₁')),
      nhdsWithin_le_nhds (A.eventually_isUnit_onePlusKR k ε h.isUnit_onePlusKR),
      A.eventually_eph_zero_ne k ε h.ε_eq h.a_ne hμ] with μ hμ' he₁ hz hX hP hU he0 hΨ
    have hμ0 : (A.withMu μ).μ ≠ 0 := by rw [withMu_μ, ← hμ]; exact hμ'
    refine (A.withMu μ).not_isUnit_onePlusKR_of_Kph k ε h.ε_eq h.a_ne hμ0 he₁
      (fun h0 ↦ he0 (h0 ▸ he₁)) (fun l h0 ↦ (hz l).1 (by rw [h0, sub_self]))
      (fun l h0 ↦ (hz l).1 (by rw [h0, neg_sq, sub_self])) hP ?_ hU
    rw [(A.withMu μ).Kph_eq_rho_mul k ε he₁ hX, hΨ, mul_zero]
  have hoff : ∀ n, (A.withMu A.μ).z k n ≠ u₁ A.μ := fun n hn ↦
    h.z_sq_ne n (by rw [withMu_self] at hn; rw [hn, hu₁', sub_self])
  have hΨ := (A.continuousAt_withMu_Psi k ε hc₁ hoff
      (X_ne_of_mu h.ε_eq h.a_ne hμ hu₁')).eventually_ne (by
    change A.Psi k ε (u₁ A.μ) ≠ 0
    rw [hu₁, A.Psi_of_mu k ε hμ h.u_sq hu hl]
    exact mul_ne_zero (mul_ne_zero (mul_ne_zero (by norm_num) (by exact_mod_cast h.a_ne)) hu)
      h.D_ne)
  exact ⟨u₁, u₂, hu₁, hu₂, hc₁, hc₂, eventually_dmGeneric_of h.ε_eq h.a_ne hμ h.shat_ne_of_odd hs'
    hu₁' hc₁ he₁ hu₂ hc₂ he₂ hm₂ (nhdsWithin_le_nhds hΨ)⟩

/-- **Corollary "half filling" as a limit in `μ`**: under `HFHyp` with `ŝ' ≠ 0` for odd `k`,
`gammaR → gammaHF` as `μ → 0`, `μ ≠ 0`, at fixed `U`, `β`, `k`, with the roots `u₁`, `u₂` of
`HFHyp.exists_eventually_dmGeneric` (`𝓝[≠] 0` is not empty). -/
theorem HFHyp.exists_tendsto_gammaR {u : ℂ} (h : A.HFHyp k ε u)
    (hs' : Odd k → A.shat' k ε ≠ 0) :
    ∃ u₁ u₂ : ℝ → ℂ, u₁ A.μ = u ∧ u₂ A.μ ^ 2 = A.shat' k ε ∧ ContinuousAt u₁ A.μ ∧
      ContinuousAt u₂ A.μ ∧ (∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).DMGeneric k ε (u₁ μ) (u₂ μ)) ∧
      ∀ n m, Tendsto (fun μ ↦ (A.withMu μ).gammaR k ε (u₁ μ) (u₂ μ) n m) (𝓝[≠] A.μ)
        (𝓝 (A.gammaHF k ε u n m)) := by
  obtain ⟨u₁, u₂, hu₁, hu₂, hc₁, hc₂, hdm⟩ := h.exists_eventually_dmGeneric hs'
  exact ⟨u₁, u₂, hu₁, hu₂, hc₁, hc₂, hdm, fun n m ↦
    tendsto_gammaR_of_HFHyp (U := fun _ ↦ A.U) (β := fun _ ↦ A.β) (μ := id)
      (hβ := fun _ ↦ A.β_pos) k continuousAt_const continuousAt_const continuousAt_id
      nhdsWithin_le_nhds ε h hdm n m⟩

/-- **Corollary "half filling", as stated in the note, is the limit `μ → 0` of Theorem "density and
magnetic"**: under its assumptions `e(zₙ) ≠ 0` for all `n`, `û² = ŝ`, `D̂ ≠ 0` (`HFHyp.of_eph`); for
odd `k`, `e(0) = ŝŝ' ≠ 0` at the mirror fixed point gives `ŝ' ≠ 0`. -/
theorem exists_tendsto_gammaR_of_eph (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ = 0)
    (he : ∀ n, A.eph k ε (A.z k n) ≠ 0) {u : ℂ} (hu : u ^ 2 = A.shat k ε)
    (hD : A.Dhat k ε u ≠ 0) :
    ∃ u₁ u₂ : ℝ → ℂ, u₁ A.μ = u ∧ u₂ A.μ ^ 2 = A.shat' k ε ∧ ContinuousAt u₁ A.μ ∧
      ContinuousAt u₂ A.μ ∧ (∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).DMGeneric k ε (u₁ μ) (u₂ μ)) ∧
      ∀ n m, Tendsto (fun μ ↦ (A.withMu μ).gammaR k ε (u₁ μ) (u₂ μ) n m) (𝓝[≠] A.μ)
        (𝓝 (A.gammaHF k ε u n m)) := by
  refine (HFHyp.of_eph (A := A) (k := k) (ε := ε) hε ha hμ he hu hD).exists_tendsto_gammaR
    fun ⟨m, hm⟩ h0 ↦ he (-m - 1) ?_
  rw [A.eph_of_mu' k ε hμ, A.z_of_fixed k (by unfold mirror; omega), hμ, h0]
  simp

/-! ### At a lattice zero `z_c² = ŝ` -/

/-- **The assumptions of Theorem "density and magnetic" hold near a lattice zero of half filling**:
at `z_c² = ŝ` with `c` off the mirror fixed points and `ŝ' ≠ 0` for odd `k`, there are roots
`u₁(μ)`, `u₂(μ)` of `e`, continuous at `μ = 0` with `u₁(0) = z_c`, `u₂(0)² = ŝ'`, such that
`DMGeneric` holds at every `μ ≠ 0` near `0`. Here `S_k` has a pole at `u₁(0) = z_c`, and
`Ψ(u₁)S_k⁻¹(u₁) → λ ≠ 0`, as `S_k⁻¹(u₁) → 0` (`continuousAt_withMu_Sinv`). -/
theorem HFLZ.exists_eventually_dmGeneric {c : ℤ} (h : A.HFLZ k ε c) (hc : c ≠ mirror k c)
    (hs' : Odd k → A.shat' k ε ≠ 0) :
    ∃ u₁ u₂ : ℝ → ℂ, u₁ A.μ = A.z k c ∧ u₂ A.μ ^ 2 = A.shat' k ε ∧ ContinuousAt u₁ A.μ ∧
      ContinuousAt u₂ A.μ ∧ ∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).DMGeneric k ε (u₁ μ) (u₂ μ) := by
  have hμ := h.μ_eq
  have hl := A.lam_ne_zero ε h.ε_eq h.a_ne
  have hss := A.shat_sub_shat'_ne k ε h.ε_eq h.a_ne
  have hzc : A.z k c ^ 2 = A.shat k ε := sub_eq_zero.mp h.z_c
  have hzc0 : A.z k c ≠ 0 := A.z_ne_zero_of_ne k (Ne.symm hc)
  have hs0 : A.shat k ε ≠ 0 := hzc ▸ pow_ne_zero 2 hzc0
  obtain ⟨u₁, hu₁, hc₁, he₁⟩ := A.exists_root_of_mu k ε hμ
    (by rw [A.eph_of_mu' k ε hμ, hzc, sub_self, zero_mul])
    (fun h0 ↦ hss (by rw [hzc] at h0; linear_combination h0))
  obtain ⟨u₂, hu₂, hc₂, he₂, hm₂⟩ :=
    exists_root₂ h.ε_eq h.a_ne hμ (z_sq_ne'_of hμ hs' h.odd_ne)
  have hu₁' : u₁ A.μ ^ 2 = A.shat k ε := by rw [hu₁, hzc]
  refine ⟨u₁, u₂, hu₁, hu₂, hc₁, hc₂, eventually_dmGeneric_of h.ε_eq h.a_ne hμ (fun _ ↦ hs0) hs'
    hu₁' hc₁ he₁ hu₂ hc₂ he₂ hm₂ ?_⟩
  -- `ΨS⁻¹ = λ - (N/D)S⁻¹ → λ`
  have hD := A.continuousAt_withMu_X k ε hc₁
  simp only [withMu_β, withMu_lam] at hD
  have hD₀ : (A.β : ℂ) * (A.withMu A.μ).Hph k ε (-u₁ A.μ) + A.lam ε * (A.withMu A.μ).cbar ≠ 0 :=
    X_ne_of_mu h.ε_eq h.a_ne hμ hu₁'
  obtain ⟨hSi, hSi₀⟩ := A.continuousAt_withMu_Sinv k hc₁ (c := c) (by rw [hu₁, withMu_self])
  have hρ := A.continuousAt_withMu_ρ A.μ
  have hH := A.continuousAt_withMu_Hph k ε hc₁.neg
  have hN : ContinuousAt (fun μ ↦ 4 * (A.a : ℂ) * A.β * u₁ μ *
      (A.β * (A.withMu μ).Hph k ε (-u₁ μ)) +
      A.lam ε ^ 2 * (A.β : ℂ) ^ 2 * (A.withMu μ).ρ / 4) A.μ := by
    fun_prop
  have hGc : ContinuousAt (fun μ ↦ A.lam ε - (4 * (A.a : ℂ) * A.β * u₁ μ *
      (A.β * (A.withMu μ).Hph k ε (-u₁ μ)) + A.lam ε ^ 2 * (A.β : ℂ) ^ 2 * (A.withMu μ).ρ / 4) /
      ((A.β : ℂ) * (A.withMu μ).Hph k ε (-u₁ μ) + A.lam ε * (A.withMu μ).cbar) *
      (A.withMu μ).Sinv k (u₁ μ)) A.μ :=
    continuousAt_const.sub ((hN.div hD hD₀).mul hSi)
  have hG := hGc.eventually_ne (by
    rw [hSi₀, mul_zero, sub_zero]
    exact hl)
  -- `S_k(u₁)S_k⁻¹(u₁) = 1` off the lattice, where the factor of `S_k` that does not vanish at `z_c`
  -- stays nonzero
  have hy : ContinuousAt (fun μ ↦ (A.β : ℂ) * (u₁ μ - μ) / 2) A.μ := by fun_prop
  have hcs := (A.withMu A.μ).cosh_sinh_z k c
  rw [← (show u₁ A.μ = (A.withMu A.μ).z k c by rw [hu₁, withMu_self])] at hcs
  simp only [withMu_β, withMu_μ] at hcs
  have hoff : ∀ᶠ μ in 𝓝[≠] A.μ, ∀ n, (A.withMu μ).z k n ≠ u₁ μ :=
    (eventually_z_sq_ne h.ε_eq h.a_ne hμ (fun _ ↦ hs0) hs' hu₁' hc₁ he₁ hu₂ hc₂ he₂).2.mono
      fun μ hz n h0 ↦ (hz n).1 (by rw [h0, sub_self])
  have hSS : ∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).S k (u₁ μ) * (A.withMu μ).Sinv k (u₁ μ) = 1 := by
    by_cases hk : Even k
    · obtain ⟨j, hj⟩ := hk
      filter_upwards [hoff, nhdsWithin_le_nhds
        ((continuous_sinh.continuousAt.comp hy).eventually_ne (hcs.1 ⟨j, hj⟩).2)] with μ hoff hs
      exact (A.withMu μ).S_mul_Sinv k hs
        ((A.withMu μ).cosh_ne_zero_of_off (j := j) (by omega) hoff)
    · obtain ⟨j, hj⟩ := Int.not_even_iff_odd.mp hk
      filter_upwards [hoff, nhdsWithin_le_nhds
        ((continuous_cosh.continuousAt.comp hy).eventually_ne (hcs.2 hk).2)] with μ hoff hc
      exact (A.withMu μ).S_mul_Sinv k
        ((A.withMu μ).sinh_ne_zero_of_off (j := j + 1) (by omega) hoff) hc
  filter_upwards [nhdsWithin_le_nhds hG, hSS] with μ hG hSS h0
  apply hG
  have h1 := congrArg (· * (A.withMu μ).Sinv k (u₁ μ)) h0
  simp only [zero_mul] at h1
  rw [← h1, Psi, sub_mul, mul_assoc ((A.withMu μ).lam ε), hSS, mul_one]
  rfl

/-- **At a lattice zero of half filling, Eq. (Gamma-hf-lz) is the limit `μ → 0` of Theorem "density
and magnetic"**: at `z_c² = ŝ` with `c` off the mirror fixed points and `ŝ' ≠ 0` for odd `k`,
`gammaR → gammaHFlz` as `μ → 0`, `μ ≠ 0`, at fixed `U`, `β`, `k`, with the roots `u₁`, `u₂` of
`HFLZ.exists_eventually_dmGeneric`. -/
theorem HFLZ.exists_tendsto_gammaR {c : ℤ} (h : A.HFLZ k ε c) (hc : c ≠ mirror k c)
    (hs' : Odd k → A.shat' k ε ≠ 0) :
    ∃ u₁ u₂ : ℝ → ℂ, u₁ A.μ = A.z k c ∧ u₂ A.μ ^ 2 = A.shat' k ε ∧ ContinuousAt u₁ A.μ ∧
      ContinuousAt u₂ A.μ ∧ (∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).DMGeneric k ε (u₁ μ) (u₂ μ)) ∧
      ∀ n m, Tendsto (fun μ ↦ (A.withMu μ).gammaR k ε (u₁ μ) (u₂ μ) n m) (𝓝[≠] A.μ)
        (𝓝 (A.gammaHFlz k ε c n m)) := by
  obtain ⟨u₁, u₂, hu₁, hu₂, hc₁, hc₂, hdm⟩ := h.exists_eventually_dmGeneric hc hs'
  exact ⟨u₁, u₂, hu₁, hu₂, hc₁, hc₂, hdm, fun n m ↦
    tendsto_gammaR_of_HFLZ (U := fun _ ↦ A.U) (β := fun _ ↦ A.β) (μ := id)
      (hβ := fun _ ↦ A.β_pos) k continuousAt_const continuousAt_const continuousAt_id
      nhdsWithin_le_nhds ε h hdm n m⟩

end Params

end HubbardAtom
