import HubbardAtomLean.FieldLAxisIsolated
import HubbardAtomLean.HalfFillingApproach
import HubbardAtomLean.FieldTZero

/-!
# The `K` conditions of `LAxisGeneric` near the corner

Draft.
-/

open Complex Filter Topology

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### On `h = 0`: the zero-field channels -/

/-- On `h = 0`, `E₈ = e_d e_m`, with `e_ε` of the zero-field channels of `(U, β, μ)`. -/
theorem E8L_h0 (hh : F.h = 0) (v : ℂ) :
    F.E8L k v = (F.lat F.μ).eph k 1 v * (F.lat F.μ).eph k (-1) v := by
  have h₀ := F.lat_p₀ hh
  have h₂ := F.lat_p₂ hh
  have hu := F.lat_p₁_up hh
  have hd := F.lat_p₁_dn hh
  have hρ := F.lat_ρ hh
  have hm := F.mf_of_h hh
  simp only [E8L, HL, rUp, rDn, hm, sub_zero, add_zero, piS, piP, lamL, q₀L, wL,
    Params.eph, Params.Hph, Params.lam, Params.q₀, Params.w, lat_a, lat_ω, hρ, h₀, h₂, ← hu,
    ← hd]
  ring

/-- On `h = 0`, `K_{t,-t} = cK_ε + εβH(-u)e_ε` with `c = -ε[H(u)H(-u) - λ²d_π]`, where `K_ε`,
`e_ε` are those of the zero-field channel `ε` of `(U, β, μ)`. -/
theorem KL_opp_h0 (hh : F.h = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) (t : Bool) (u : ℂ) :
    F.KL k t (!t) u =
      -ε * (F.HL k F.ρ u * F.HL k F.ρ (-u) - F.lamL ^ 2 * (F.piP - F.piS)) *
          (F.lat F.μ).Kph k ε u +
        ε * F.β * F.HL k F.ρ (-u) * (F.lat F.μ).eph k ε u := by
  have h₀ := F.lat_p₀ hh
  have h₂ := F.lat_p₂ hh
  have hu := F.lat_p₁_up hh
  have hd := F.lat_p₁_dn hh
  have hρ := F.lat_ρ hh
  have hm := F.mf_of_h hh
  have hB : ∀ σ, F.BL σ = F.lat F.μ := fun σ ↦ by cases σ <;> simp [Bup, Bdn, hh]
  have hmL : ∀ σ, F.mL σ = 0 := fun σ ↦ by cases σ <;> simp [mL_true, mL_false, hm]
  have hrL : ∀ σ, F.rL σ = F.ρ := fun σ ↦ by simp [rL, hmL]
  have hS : ∀ σ, (F.BL σ).S k u = (F.lat F.μ).S k u := fun σ ↦ by rw [hB]
  simp only [KL, fMp, fMm, Bool.not_eq_self, ite_false, αSL, αPL, hS, hmL, hrL, Bool.not_not,
    Params.Kph, Params.eph, Params.Hph, Params.lam, Params.q₀, Params.w, Params.α,
    Params.cbar, HL, piS, piP, lamL, q₀L, wL, cS, cP, lat_a, lat_ω, lat_β, hρ, h₀, h₂, ← hu, ← hd]
  rcases hε with rfl | rfl <;> push_cast <;> ring

/-- On `h = 0`, `e_ε = [H(u)H(-u) - λ²d_π] + ελp↑[H(u) + H(-u)]`. -/
theorem eph_h0 (hh : F.h = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) (u : ℂ) :
    (F.lat F.μ).eph k ε u =
      F.HL k F.ρ u * F.HL k F.ρ (-u) - F.lamL ^ 2 * (F.piP - F.piS) +
        ε * F.lamL * F.pUp * (F.HL k F.ρ u + F.HL k F.ρ (-u)) := by
  have h₀ := F.lat_p₀ hh
  have h₂ := F.lat_p₂ hh
  have hu := F.lat_p₁_up hh
  have hd := F.lat_p₁_dn hh
  have hρ := F.lat_ρ hh
  simp only [Params.eph, Params.Hph, Params.lam, Params.q₀, Params.w, HL, piS, piP, lamL, q₀L, wL,
    lat_a, lat_ω, hρ, h₀, h₂, ← hu, ← hd]
  rcases hε with rfl | rfl <;> ring

/-- On `h = 0`, at a root `u` of `e_ε` with `u² + q₀ ≠ 0`, `K_{t,-t}(u) ≠ 0` iff `K_ε(u) ≠ 0`; here
the direction `K_ε(u) ≠ 0 → K_{t,-t}(u) ≠ 0`. -/
theorem KL_opp_ne_h0 (hh : F.h = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) (ha : (F.a : ℂ) ≠ 0) {u : ℂ}
    (he : (F.lat F.μ).eph k ε u = 0) (hq : u ^ 2 + F.q₀L k ≠ 0)
    (hK : (F.lat F.μ).Kph k ε u ≠ 0) (t : Bool) : F.KL k t (!t) u ≠ 0 := by
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  have hl : F.lamL ≠ 0 := by rw [lamL]; exact mul_ne_zero (by norm_num) (pow_ne_zero 2 ha)
  have hp : (F.pUp : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.pUp_pos.ne'
  rw [F.KL_opp_h0 k hh hε, he, mul_zero, add_zero]
  refine mul_ne_zero (mul_ne_zero (neg_ne_zero.mpr hε0) fun hc ↦ ?_) hK
  have hs : F.HL k F.ρ u + F.HL k F.ρ (-u) = 2 * (u ^ 2 + F.q₀L k) := by simp only [HL]; ring
  rw [F.eph_h0 k hh hε, hc, zero_add, hs] at he
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero hε0 hl) hp) (mul_ne_zero two_ne_zero hq) he

end FParams

namespace Params

variable {A : Params} {k : ℤ} {ε : ℂ}

/-- **`K ≠ 0` at every root of `e` near half filling**: under `HFHyp` with `ŝ' ≠ 0` for odd `k`,
the roots `u₁`, `u₂` of `HFHyp.exists_eventually_dmGeneric` satisfy `DMGeneric` at every `μ ≠ 0`
near `0`, and there `K(v) ≠ 0` at all four roots `v = ±u₁, ±u₂` of `e`: `K(±u₁) = ρΨ(±u₁)` with
`Ψ(±u₁) ≠ 0` (`HFHyp.eventually_Psi_ne`, at `±û` by `HFHyp.neg`), and `K(±u₂) → -2λβp₀`
(`eventually_Kph_ne_of_shat'`). -/
theorem HFHyp.exists_eventually_Kph_ne {u : ℂ} (h : A.HFHyp k ε u)
    (hs' : Odd k → A.shat' k ε ≠ 0) :
    ∃ u₁ u₂ : ℝ → ℂ, ∀ᶠ μ in 𝓝[≠] A.μ, (A.withMu μ).DMGeneric k ε (u₁ μ) (u₂ μ) ∧
      ∀ v, (A.withMu μ).eph k ε v = 0 → (A.withMu μ).Kph k ε v ≠ 0 := by
  have hμ := h.μ_eq
  have hss := A.shat_sub_shat'_ne k ε h.ε_eq h.a_ne
  have hz' := z_sq_ne'_of hμ hs' h.odd_ne
  obtain ⟨u₂, hu₂, hc₂, he₂, hm₂⟩ := exists_root₂ h.ε_eq h.a_ne hμ hz'
  obtain ⟨u₁, hu₁, hc₁, he₁⟩ := A.exists_root_of_mu k ε hμ
    (by rw [A.eph_of_mu' k ε hμ, h.u_sq, sub_self, zero_mul])
    (fun h0 ↦ hss (by rw [h.u_sq] at h0; linear_combination h0))
  have hu₁' : u₁ A.μ ^ 2 = A.shat k ε := by rw [hu₁, h.u_sq]
  have he₁n : ∀ᶠ μ in 𝓝 A.μ, (A.withMu μ).eph k ε (-u₁ μ) = 0 := by
    filter_upwards [he₁] with μ he
    rwa [eph_neg]
  have hc₁n : ContinuousAt (fun μ ↦ -u₁ μ) A.μ := hc₁.neg
  have hΨ := h.eventually_Psi_ne hs' hu₁ hc₁ he₁
  have hΨn := h.neg.eventually_Psi_ne hs' (u₁ := fun μ ↦ -u₁ μ) (by simp [hu₁]) hc₁n he₁n
  have hX := (A.continuousAt_withMu_X k ε hc₁).eventually_ne (X_ne_of_mu h.ε_eq h.a_ne hμ hu₁')
  have hXn := (A.continuousAt_withMu_X k ε hc₁n).eventually_ne
    (X_ne_of_mu h.ε_eq h.a_ne hμ (show (-u₁ A.μ) ^ 2 = _ by rw [neg_sq]; exact hu₁'))
  refine ⟨u₁, u₂, ?_⟩
  filter_upwards [eventually_dmGeneric_of h.ε_eq h.a_ne hμ h.shat_ne_of_odd hs' hu₁' hc₁ he₁ hu₂
      hc₂ he₂ hm₂ hΨ, self_mem_nhdsWithin, hΨ, hΨn, nhdsWithin_le_nhds hX,
    nhdsWithin_le_nhds hXn,
    nhdsWithin_le_nhds (eventually_Kph_ne_of_shat' h.ε_eq h.a_ne hμ hz' hu₂ hc₂),
    nhdsWithin_le_nhds (eventually_Kph_ne_of_shat' h.ε_eq h.a_ne hμ hz' (v := fun μ ↦ -u₂ μ)
      (by simp [hu₂]) hc₂.neg)] with μ hdm hμ' hΨ hΨn hX hXn hK₂ hK₂n
  refine ⟨hdm, fun v hv ↦ ?_⟩
  have hρ := (A.withMu μ).rho_ne_zero (by rw [withMu_μ, ← hμ]; exact hμ')
  rw [(A.withMu μ).eph_eq_mul k ε hdm.e_u₁ hdm.e_u₂ hdm.s_ne, mul_eq_zero, sub_eq_zero,
    sub_eq_zero, sq_eq_sq_iff_eq_or_eq_neg, sq_eq_sq_iff_eq_or_eq_neg] at hv
  rcases hv with (rfl | rfl) | (rfl | rfl)
  · rw [(A.withMu μ).Kph_eq_rho_mul k ε hdm.e_u₁ hX]
    exact mul_ne_zero hρ hΨ
  · rw [(A.withMu μ).Kph_eq_rho_mul k ε (by rw [eph_neg]; exact hdm.e_u₁) hXn]
    exact mul_ne_zero hρ hΨn
  · exact hK₂
  · exact hK₂n

end Params

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### On `μ = 0`: the zero-field channels of the Shiba image -/

/-- On `μ = 0`, the weights of the zero-field channels of the Shiba image `(-U, β, h)`:
`p₀ = p↓`, `p₂ = p↑`, `p₁ = p₂ = p₀`, `ρ = m`. -/
theorem shiba_lat_weights (hμ : F.μ = 0) :
    (F.shiba.lat F.shiba.μ).p₀ = F.pDn ∧ (F.shiba.lat F.shiba.μ).p₂ = F.pUp ∧
      (F.shiba.lat F.shiba.μ).p₁ = F.p₀ ∧ F.p₂ = F.p₀ ∧ (F.shiba.lat F.shiba.μ).ρ = F.mf := by
  have hh : F.shiba.h = 0 := hμ
  have hu := F.shiba.lat_p₁_up hh
  have hd := F.shiba.lat_p₁_dn hh
  rw [shiba_pUp] at hu
  rw [shiba_pDn] at hd
  exact ⟨(F.shiba.lat_p₀ hh).trans F.shiba_p₀, (F.shiba.lat_p₂ hh).trans F.shiba_p₂, hd,
    hu.symm.trans hd, (F.shiba.lat_ρ hh).trans F.shiba_ρ⟩

/-- On `μ = 0`, `E₈ = e'_d e'_m`, with `e'_ε` of the zero-field channels of the Shiba image
`(-U, β, h)`. -/
theorem E8L_mu0 (hμ : F.μ = 0) (v : ℂ) :
    F.E8L k v = (F.shiba.lat F.shiba.μ).eph k 1 v * (F.shiba.lat F.shiba.μ).eph k (-1) v := by
  obtain ⟨h₀, h₂, h₁, hp, hρ⟩ := F.shiba_lat_weights hμ
  have ha : ((F.shiba.lat F.shiba.μ).a : ℂ) = -F.a := by rw [lat_a, shiba_a, ofReal_neg]
  simp only [E8L, HL, rUp, rDn, ρ, hp, sub_self, zero_sub, zero_add, piS, piP, lamL, q₀L, wL,
    Params.eph, Params.Hph, Params.lam, Params.q₀, Params.w, ha, lat_ω, shiba_ω, hρ, h₀, h₂, h₁]
  push_cast
  ring

/-- On `μ = 0`, `e'_ε = [H↑(u)H↑(-u) + λ²d_π] + ελp₀[H↑(u) + H↑(-u)]`. -/
theorem eph_mu0 (hμ : F.μ = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) (u : ℂ) :
    (F.shiba.lat F.shiba.μ).eph k ε u =
      F.HL k F.rUp u * F.HL k F.rUp (-u) + F.lamL ^ 2 * (F.piP - F.piS) +
        ε * F.lamL * F.p₀ * (F.HL k F.rUp u + F.HL k F.rUp (-u)) := by
  obtain ⟨h₀, h₂, h₁, hp, hρ⟩ := F.shiba_lat_weights hμ
  have ha : ((F.shiba.lat F.shiba.μ).a : ℂ) = -F.a := by rw [lat_a, shiba_a, ofReal_neg]
  simp only [HL, rUp, ρ, hp, sub_self, zero_sub, piS, piP, lamL, q₀L, wL, Params.eph, Params.Hph,
    Params.lam, Params.q₀, Params.w, ha, lat_ω, shiba_ω, hρ, h₀, h₂, h₁]
  rcases hε with rfl | rfl <;> push_cast <;> ring

/-- On `μ = 0`, `K_{↑↓} = cK'_ε + εβH↑(-u)e'_ε` with `c = -ε[H↑(u)H↑(-u) + λ²d_π]`, where
`K'_ε`, `e'_ε` are those of the zero-field channel `ε` of the Shiba image `(-U, β, h)`. -/
theorem KL_opp_mu0 (hμ : F.μ = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) (u : ℂ) :
    F.KL k true false u =
      -ε * (F.HL k F.rUp u * F.HL k F.rUp (-u) + F.lamL ^ 2 * (F.piP - F.piS)) *
          (F.shiba.lat F.shiba.μ).Kph k ε u +
        ε * F.β * F.HL k F.rUp (-u) * (F.shiba.lat F.shiba.μ).eph k ε u := by
  obtain ⟨h₀, h₂, h₁, hp, hρ⟩ := F.shiba_lat_weights hμ
  have ha : ((F.shiba.lat F.shiba.μ).a : ℂ) = -F.a := by rw [lat_a, shiba_a, ofReal_neg]
  have hS : (F.BL true).S k u = (F.shiba.lat F.shiba.μ).S k u := by
    simp only [Params.S, BL_true, Bup, lat_β, lat_μ, shiba_β, shiba_μ, hμ, zero_add]
  simp only [KL, fMp, fMm, Bool.not_false, Bool.not_true, αSL, αPL, mL_false, rL_true, rL_false,
    hS, Params.Kph, Params.eph, Params.Hph, Params.lam, Params.q₀, Params.w, Params.α,
    Params.cbar, HL, rUp, rDn, ρ, hp, sub_self, zero_sub, zero_add, piS, piP, lamL, q₀L, wL, cS, cP, ha, lat_ω, lat_β, shiba_ω, shiba_β,
    hρ, h₀, h₂, h₁]
  rcases hε with rfl | rfl <;> push_cast <;> ring

/-- On `μ = 0`, `K_{↓↑}(u) = K_{↑↓}(-u)`, as `H↓(-u) = H↑(u)` and `S↓(u) = -S↑(-u)`. -/
theorem KL_swap_mu0 (hμ : F.μ = 0) (u : ℂ) : F.KL k false true u = F.KL k true false (-u) := by
  obtain ⟨-, -, -, hp, -⟩ := F.shiba_lat_weights hμ
  have hS : (F.BL false).S k u = -(F.BL true).S k (-u) := by
    simp only [Params.S, BL_true, BL_false, Bup, Bdn, lat_β, lat_μ, hμ, zero_add, zero_sub]
    rw [show (F.β : ℂ) * (u - ((-F.h : ℝ) : ℂ)) / 2 = -(F.β * (-u - (F.h : ℂ)) / 2) by
      push_cast; ring]
    split_ifs <;> simp [tanh_neg, coth_neg]
  simp only [KL, fMp, fMm, Bool.true_eq_false, Bool.false_eq_true, ite_false, Bool.not_false,
    Bool.not_true, αSL, αPL, mL_true, mL_false, rL_true, rL_false, hS, HL, rUp, rDn, ρ, hp,
    sub_self, neg_neg]
  push_cast
  ring

/-- On `μ = 0`, at a root `u` of `e'_ε` with `u² + q₀ ≠ 0`, `K'_ε(u) ≠ 0` gives `K_{↑↓}(u) ≠ 0`. -/
theorem KL_opp_ne_mu0 (hμ : F.μ = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) (ha : (F.a : ℂ) ≠ 0)
    {u : ℂ} (he : (F.shiba.lat F.shiba.μ).eph k ε u = 0) (hq : u ^ 2 + F.q₀L k ≠ 0)
    (hK : (F.shiba.lat F.shiba.μ).Kph k ε u ≠ 0) : F.KL k true false u ≠ 0 := by
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  have hl : F.lamL ≠ 0 := by rw [lamL]; exact mul_ne_zero (by norm_num) (pow_ne_zero 2 ha)
  have hp : (F.p₀ : ℂ) ≠ 0 := ofReal_ne_zero.mpr F.p₀_pos.ne'
  rw [F.KL_opp_mu0 k hμ hε, he, mul_zero, add_zero]
  refine mul_ne_zero (mul_ne_zero (neg_ne_zero.mpr hε0) fun hc ↦ ?_) hK
  have hs : F.HL k F.rUp u + F.HL k F.rUp (-u) = 2 * (u ^ 2 + F.q₀L k) := by simp only [HL]; ring
  rw [F.eph_mu0 k hμ hε, hc, zero_add, hs] at he
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero hε0 hl) hp) (mul_ne_zero two_ne_zero hq) he

/-! ### On `μ = h`: `K = ρΨ/N` at the roots -/

/-- `κ = e^{βU/2}`: on `μ = h`, `p↓ = κp₀` and `p↑ = κp₂`. -/
noncomputable def kapL : ℝ := Real.exp (F.β * F.U / 2)

theorem pDn_of_plus (hμh : F.μ = F.h) : F.pDn = F.kapL * F.p₀ := by
  rw [pDn, p₀, boltzDn, boltz₀, kapL, hμh, mul_div_assoc', ← Real.exp_add]
  congr 2
  ring

theorem pUp_of_plus (hμh : F.μ = F.h) : F.pUp = F.kapL * F.p₂ := by
  rw [pUp, p₂, boltzUp, boltz₂, kapL, hμh, mul_div_assoc', ← Real.exp_add]
  congr 2
  ring

/-- A part of `Φ` of `K_{↑↑}` on `μ = h` (`KL_UU_plus`). -/
noncomputable def DXUU (h X Y P r κ l β : ℂ) : ℂ :=
  2 * Y ^ 2 * P ^ 2 * κ ^ 2 * l ^ 2 * β + 2 * Y ^ 2 * P ^ 2 * κ * l ^ 2 * β -
    2 * h ^ 2 * P ^ 2 * κ ^ 2 * l ^ 2 * β - 2 * h ^ 2 * P ^ 2 * l ^ 2 * β -
    2 * h * Y * P ^ 2 * κ ^ 2 * l ^ 2 * β + 4 * h * Y * P ^ 2 * l ^ 2 * β -
    2 * h * Y * P ^ 2 * κ * l ^ 2 * β - 2 * h ^ 2 * Y ^ 2 * β -
    1 / 4 * Y ^ 2 * r ^ 2 * κ ^ 2 * l ^ 2 * β + 1 / 4 * Y ^ 2 * r ^ 2 * l ^ 2 * β +
    2 * h * Y ^ 3 * β + 1 / 4 * h ^ 2 * r ^ 2 * κ ^ 2 * l ^ 2 * β +
    1 / 4 * h ^ 2 * r ^ 2 * l ^ 2 * β + 1 / 2 * h ^ 2 * r ^ 2 * κ * l ^ 2 * β + 2 * h ^ 4 * β +
    2 * h * X * Y ^ 2 * β - X * Y ^ 3 * β - 2 * h ^ 3 * X * β + h ^ 2 * X * Y * β -
    2 * h ^ 3 * Y * β + 1 / 2 * h * Y * r ^ 2 * κ ^ 2 * l ^ 2 * β - h * Y * r ^ 2 * l ^ 2 * β -
    1 / 2 * h * Y * r ^ 2 * κ * l ^ 2 * β

/-- A part of `Φ` of `K_{↑↑}` on `μ = h` (`KL_UU_plus`). -/
noncomputable def DYUU (h Y P r κ l β : ℂ) : ℂ :=
  - 2 * h ^ 2 * P ^ 2 * l ^ 2 * β + 2 * h ^ 2 * P ^ 2 * κ * l ^ 2 * β -
    2 * h * Y * P ^ 2 * κ ^ 2 * l ^ 2 * β - 2 * h * Y * P ^ 2 * κ * l ^ 2 * β +
    1 / 2 * P ^ 2 * r ^ 2 * κ ^ 3 * l ^ 4 * β + 1 / 4 * P ^ 2 * r ^ 2 * κ ^ 4 * l ^ 4 * β -
    1 / 4 * P ^ 2 * r ^ 2 * l ^ 4 * β - 1 / 2 * P ^ 2 * r ^ 2 * κ * l ^ 4 * β -
    2 * P ^ 4 * κ ^ 3 * l ^ 4 * β - P ^ 4 * κ ^ 4 * l ^ 4 * β + P ^ 4 * l ^ 4 * β +
    2 * P ^ 4 * κ * l ^ 4 * β - h ^ 2 * Y ^ 2 * β + 1 / 2 * h ^ 2 * r ^ 2 * l ^ 2 * β +
    1 / 2 * h ^ 2 * r ^ 2 * κ * l ^ 2 * β + h ^ 4 * β + 1 / 4 * h * Y * r ^ 2 * κ ^ 2 * l ^ 2 * β -
    1 / 4 * h * Y * r ^ 2 * l ^ 2 * β

/-- A part of `Φ` of `K_{↑↑}` on `μ = h` (`KL_UU_plus`). -/
noncomputable def DrUU (h P r κ l β : ℂ) : ℂ :=
  5 / 2 * h * P ^ 2 * κ ^ 2 * l ^ 4 * β + h * P ^ 2 * κ ^ 3 * l ^ 4 * β -
    1 / 4 * h * P ^ 2 * κ ^ 4 * l ^ 4 * β - 1 / 4 * h * P ^ 2 * l ^ 4 * β +
    h * P ^ 2 * κ * l ^ 4 * β + 1 / 4 * h ^ 3 * κ ^ 2 * l ^ 2 * β + 1 / 4 * h ^ 3 * l ^ 2 * β -
    1 / 2 * h ^ 3 * κ * l ^ 2 * β - 1 / 8 * h * r ^ 2 * κ ^ 2 * l ^ 4 * β +
    1 / 16 * h * r ^ 2 * κ ^ 4 * l ^ 4 * β + 1 / 16 * h * r ^ 2 * l ^ 4 * β

/-- A part of `Φ` of `K_{↓↑}` on `μ = h` (`KL_DU_plus`). -/
noncomputable def DXDU (h X Y P r κ l β : ℂ) : ℂ :=
  2 * Y * P ^ 3 * κ ^ 2 * l ^ 3 * β - 2 * Y * P ^ 3 * l ^ 3 * β + 2 * h * P ^ 3 * κ ^ 3 * l ^ 3 * β -
    2 * h * P ^ 3 * κ * l ^ 3 * β - 2 * h * Y ^ 2 * P * κ * l * β - 2 * h ^ 3 * P * κ * l * β -
    2 * X * Y ^ 2 * P * l * β + 2 * h ^ 2 * X * P * κ * l * β - 2 * h * X * Y * P * κ * l * β +
    2 * h * X * Y * P * l * β + 4 * h ^ 2 * Y * P * κ * l * β -
    3 / 4 * Y * P * r ^ 2 * κ ^ 2 * l ^ 3 * β - 1 / 4 * Y * P * r ^ 2 * κ ^ 3 * l ^ 3 * β +
    3 / 4 * Y * P * r ^ 2 * l ^ 3 * β + 1 / 4 * Y * P * r ^ 2 * κ * l ^ 3 * β +
    1 / 4 * h * P * r ^ 2 * κ ^ 2 * l ^ 3 * β - 1 / 4 * h * P * r ^ 2 * κ ^ 3 * l ^ 3 * β -
    1 / 4 * h * P * r ^ 2 * l ^ 3 * β + 1 / 4 * h * P * r ^ 2 * κ * l ^ 3 * β

/-- A part of `Φ` of `K_{↓↑}` on `μ = h` (`KL_DU_plus`). -/
noncomputable def DYDU (h Y P r κ l β : ℂ) : ℂ :=
  2 * h * P ^ 3 * κ ^ 2 * l ^ 3 * β + 2 * h * P ^ 3 * κ ^ 3 * l ^ 3 * β + 2 * h * P ^ 3 * l ^ 3 * β +
    2 * h * P ^ 3 * κ * l ^ 3 * β - 2 * h ^ 3 * P * κ * l * β - 2 * h ^ 3 * P * l * β +
    2 * h ^ 2 * Y * P * κ * l * β + 2 * h ^ 2 * Y * P * l * β +
    1 / 4 * h * P * r ^ 2 * κ ^ 2 * l ^ 3 * β - 1 / 4 * h * P * r ^ 2 * κ ^ 3 * l ^ 3 * β -
    1 / 4 * h * P * r ^ 2 * l ^ 3 * β + 1 / 4 * h * P * r ^ 2 * κ * l ^ 3 * β

/-- A part of `Φ` of `K_{↓↑}` on `μ = h` (`KL_DU_plus`). -/
noncomputable def DrDU (h P r κ l β : ℂ) : ℂ :=
  - 1 / 2 * P ^ 3 * κ ^ 2 * l ^ 5 * β - 1 / 2 * P ^ 3 * κ ^ 3 * l ^ 5 * β +
    1 / 4 * P ^ 3 * κ ^ 4 * l ^ 5 * β + 1 / 4 * P ^ 3 * κ ^ 5 * l ^ 5 * β +
    1 / 4 * P ^ 3 * l ^ 5 * β + 1 / 4 * P ^ 3 * κ * l ^ 5 * β -
    3 / 4 * h ^ 2 * P * κ ^ 2 * l ^ 3 * β - 1 / 4 * h ^ 2 * P * κ ^ 3 * l ^ 3 * β -
    1 / 4 * h ^ 2 * P * l ^ 3 * β - 3 / 4 * h ^ 2 * P * κ * l ^ 3 * β +
    1 / 8 * P * r ^ 2 * κ ^ 2 * l ^ 5 * β + 1 / 8 * P * r ^ 2 * κ ^ 3 * l ^ 5 * β -
    1 / 16 * P * r ^ 2 * κ ^ 4 * l ^ 5 * β - 1 / 16 * P * r ^ 2 * κ ^ 5 * l ^ 5 * β -
    1 / 16 * P * r ^ 2 * l ^ 5 * β - 1 / 16 * P * r ^ 2 * κ * l ^ 5 * β

/-- `N(u) = H↑(-u)H↓(-u) - λ²σ²`, `σ = (p₀ + p₂ + p↑ + p↓)/2`: the factor of `K` on `μ = h`. -/
noncomputable def NLp (u : ℂ) : ℂ :=
  F.HL k F.rUp (-u) * F.HL k F.rDn (-u) - F.lamL ^ 2 * ((F.p₀ + F.p₂ + F.pUp + F.pDn) / 2) ^ 2

/-- The arguments of the parts of `Φ` at the root `u`: `h = u² + q₀`, `X = 2au(1 - κ)ρ`,
`Y = 2au(1 + κ)ρ`, `P = (p₀ + p₂)/2`, `ρ`, `κ`, `λ`, `β`. -/
noncomputable def PhiUU (u : ℂ) : ℂ :=
  let h := u ^ 2 + F.q₀L k
  let x := 2 * F.a * u * (1 - F.kapL)
  let y := 2 * F.a * u * (1 + F.kapL)
  let P := ((F.p₀ : ℂ) + F.p₂) / 2
  x * DXUU h (x * F.ρ) (y * F.ρ) P F.ρ F.kapL F.lamL F.β +
    y * DYUU h (y * F.ρ) P F.ρ F.kapL F.lamL F.β + F.ρ * DrUU h P F.ρ F.kapL F.lamL F.β

/-- **`K_{↑↑}` on `μ = h`**: at a root `u` of `E₈`, `N K_{↑↑} = ρ[Φ + N S↑ 4λ²κP(u² + q₀)]`. -/
theorem KL_UU_plus (hμh : F.μ = F.h) {u : ℂ} (hr : F.E8L k u = 0) :
    F.NLp k u * F.KL k true true u = F.ρ * (F.PhiUU k u + F.NLp k u * (F.BL true).S k u *
      (4 * F.lamL ^ 2 * F.kapL * ((F.p₀ + F.p₂) / 2) * (u ^ 2 + F.q₀L k))) := by
  have hd := F.pDn_of_plus hμh
  have hu := F.pUp_of_plus hμh
  simp only [E8L, HL, rUp, rDn, ρ, mf, piS, piP, hd, hu] at hr
  simp only [NLp, PhiUU, DXUU, DYUU, DrUU, KL, fMp, fMm, QL, αL, DL, cLw, pL_true, pL_false,
    rL_true, rL_false, Bool.not_true, ↓reduceIte, HL, rUp, rDn, ρ, mf, piS, piP, hd, hu]
  push_cast at hr ⊢
  linear_combination (-(F.β * (u ^ 2 + F.q₀L k))) * hr

/-- `Φ` of `K_{↓↑}` on `μ = h`, with the arguments of `PhiUU`. -/
noncomputable def PhiDU (u : ℂ) : ℂ :=
  let h := u ^ 2 + F.q₀L k
  let x := 2 * F.a * u * (1 - F.kapL)
  let y := 2 * F.a * u * (1 + F.kapL)
  let P := ((F.p₀ : ℂ) + F.p₂) / 2
  x * DXDU h (x * F.ρ) (y * F.ρ) P F.ρ F.kapL F.lamL F.β +
    y * DYDU h (y * F.ρ) P F.ρ F.kapL F.lamL F.β + F.ρ * DrDU h P F.ρ F.kapL F.lamL F.β

/-- `L` of `K_{↓↑}` on `μ = h`: `K_{↓↑} = T + ρLS↓`. -/
noncomputable def LDU (u : ℂ) : ℂ :=
  F.lamL * (F.HL k F.rDn (-u) * F.HL k F.rUp (-u) + F.lamL ^ 2 * (F.piP - F.piS)) * F.kapL -
    F.lamL * (F.HL k F.rDn (-u) * F.HL k F.rUp u - F.lamL ^ 2 * (F.piP - F.piS))

set_option maxRecDepth 4000 in
/-- **`K_{↓↑}` on `μ = h`**: at a root `u` of `E₈`, `N K_{↓↑} = ρ[Φ + N S↓ L]`. -/
theorem KL_DU_plus (hμh : F.μ = F.h) {u : ℂ} (hr : F.E8L k u = 0) :
    F.NLp k u * F.KL k false true u =
      F.ρ * (F.PhiDU k u + F.NLp k u * (F.BL false).S k u * F.LDU k u) := by
  have hd := F.pDn_of_plus hμh
  have hu := F.pUp_of_plus hμh
  simp only [E8L, HL, rUp, rDn, ρ, mf, piS, piP, hd, hu] at hr
  simp only [KL, fMp, fMm, Bool.true_eq_false, ↓reduceIte, Bool.not_false, Bool.not_true, αSL,
    αPL, mL_true, rL_true, rL_false]
  simp only [NLp, PhiDU, LDU, DXDU, DYDU, DrDU, cS, cP, HL, rUp, rDn, ρ, mf, piS, piP, hd, hu]
  push_cast at hr ⊢
  linear_combination F.lamL * F.β * ((F.p₀ + F.p₂ + F.kapL * F.p₂ + F.kapL * F.p₀) / 2) * hr

/-! ### At the corner: the values of `N` and `Φ + N S L` -/

/-- At the corner and `u² + q₀ = ελp₀(1 - κ)` (the root `ŝ` of the channel `ε`): `N = -4λ²p₀²κ`. -/
theorem NLp_corner (hμ : F.μ = 0) (hh : F.h = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) {u : ℂ}
    (hu : u ^ 2 + F.q₀L k = ε * F.lamL * F.p₀ * (1 - F.kapL)) :
    F.NLp k u = -4 * F.lamL ^ 2 * F.p₀ ^ 2 * F.kapL := by
  obtain ⟨-, -, -, hp, -⟩ := F.shiba_lat_weights hμ
  have hd := F.pDn_of_plus (hμ.trans hh.symm)
  have hu' := F.pUp_of_plus (hμ.trans hh.symm)
  have hq : F.q₀L k = ε * F.lamL * F.p₀ * (1 - F.kapL) - u ^ 2 := by rw [← hu]; ring
  simp only [NLp, HL, rUp, rDn, ρ, mf, hd, hu', hp, hq]
  push_cast
  rcases hε with rfl | rfl <;> ring

/-- At the corner and `u² + q₀ = ελp₀(1 - κ)`: `Φ + N S 4λ²κP(u² + q₀) = 16p₀⁴λ⁴κ²(1 - κ)(2aβu - ελS)`
for `K_{↑↑}`, at every value `S` of `S↑(u)`. -/
theorem PhiUU_corner (hμ : F.μ = 0) (hh : F.h = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) {u : ℂ}
    (hu : u ^ 2 + F.q₀L k = ε * F.lamL * F.p₀ * (1 - F.kapL)) (S : ℂ) :
    F.PhiUU k u + F.NLp k u * S *
        (4 * F.lamL ^ 2 * F.kapL * ((F.p₀ + F.p₂) / 2) * (u ^ 2 + F.q₀L k)) =
      16 * F.p₀ ^ 4 * F.lamL ^ 4 * F.kapL ^ 2 * (1 - F.kapL) *
        (2 * F.a * F.β * u - ε * F.lamL * S) := by
  obtain ⟨-, -, -, hp, -⟩ := F.shiba_lat_weights hμ
  have hd := F.pDn_of_plus (hμ.trans hh.symm)
  have hu' := F.pUp_of_plus (hμ.trans hh.symm)
  have hq : F.q₀L k = ε * F.lamL * F.p₀ * (1 - F.kapL) - u ^ 2 := by rw [← hu]; ring
  simp only [PhiUU, DXUU, DYUU, DrUU, NLp, HL, rUp, rDn, ρ, mf, hd, hu', hp, hq]
  push_cast
  rcases hε with rfl | rfl <;> ring

/-- At the corner and `u² + q₀ = ελp₀(1 - κ)`: `Φ + N S L = 16p₀⁴λ⁴κ²(1 - κ)ε(2aβu - ελS)` for
`K_{↓↑}`, at every value `S` of `S↓(u)`. -/
theorem PhiDU_corner (hμ : F.μ = 0) (hh : F.h = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) {u : ℂ}
    (hu : u ^ 2 + F.q₀L k = ε * F.lamL * F.p₀ * (1 - F.kapL)) (S : ℂ) :
    F.PhiDU k u + F.NLp k u * S * F.LDU k u =
      16 * F.p₀ ^ 4 * F.lamL ^ 4 * F.kapL ^ 2 * (1 - F.kapL) * ε *
        (2 * F.a * F.β * u - ε * F.lamL * S) := by
  obtain ⟨-, -, -, hp, -⟩ := F.shiba_lat_weights hμ
  have hd := F.pDn_of_plus (hμ.trans hh.symm)
  have hu' := F.pUp_of_plus (hμ.trans hh.symm)
  have hq : F.q₀L k = ε * F.lamL * F.p₀ * (1 - F.kapL) - u ^ 2 := by rw [← hu]; ring
  simp only [PhiDU, DXDU, DYDU, DrDU, NLp, LDU, HL, rUp, rDn, ρ, mf, piS, piP, hd, hu', hp, hq]
  push_cast
  rcases hε with rfl | rfl <;> ring

/-- At the corner and `u² + q₀ = -ελp₀(1 + κ)` (the root `ŝ'` of the channel `ε`):
`K_{↑↑} = 4εβλ³κ(1 + κ)p₀³`. -/
theorem KL_UU_corner' (hμ : F.μ = 0) (hh : F.h = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) {u : ℂ}
    (hu : u ^ 2 + F.q₀L k = -(ε * F.lamL * F.p₀ * (1 + F.kapL))) :
    F.KL k true true u = 4 * ε * F.β * F.lamL ^ 3 * F.kapL * (1 + F.kapL) * F.p₀ ^ 3 := by
  obtain ⟨-, -, -, hp, -⟩ := F.shiba_lat_weights hμ
  have hd := F.pDn_of_plus (hμ.trans hh.symm)
  have hu' := F.pUp_of_plus (hμ.trans hh.symm)
  have hq : F.q₀L k = -(ε * F.lamL * F.p₀ * (1 + F.kapL)) - u ^ 2 := by rw [← hu]; ring
  simp only [KL, fMp, fMm, QL, αL, DL, cLw, pL_true, pL_false, rL_true, rL_false, Bool.not_true,
    ↓reduceIte, HL, rUp, rDn, ρ, mf, piS, piP, hd, hu', hp, hq]
  push_cast
  rcases hε with rfl | rfl <;> ring

/-- At the corner and `u² + q₀ = -ελp₀(1 + κ)`: `K_{↓↑} = 4βλ³κ(1 + κ)p₀³`. -/
theorem KL_DU_corner' (hμ : F.μ = 0) (hh : F.h = 0) {ε : ℂ} (hε : ε = 1 ∨ ε = -1) {u : ℂ}
    (hu : u ^ 2 + F.q₀L k = -(ε * F.lamL * F.p₀ * (1 + F.kapL))) :
    F.KL k false true u = 4 * F.β * F.lamL ^ 3 * F.kapL * (1 + F.kapL) * F.p₀ ^ 3 := by
  obtain ⟨-, -, -, hp, -⟩ := F.shiba_lat_weights hμ
  have hd := F.pDn_of_plus (hμ.trans hh.symm)
  have hu' := F.pUp_of_plus (hμ.trans hh.symm)
  have hq : F.q₀L k = -(ε * F.lamL * F.p₀ * (1 + F.kapL)) - u ^ 2 := by rw [← hu]; ring
  simp only [KL, fMp, fMm, Bool.true_eq_false, ↓reduceIte, Bool.not_false, Bool.not_true, αSL,
    αPL, mL_true, rL_true, rL_false]
  simp only [cS, cP, HL, rUp, rDn, ρ, mf, piS, piP, hd, hu', hp, hq]
  push_cast
  rcases hε with rfl | rfl <;> ring

/-- At the corner, the roots of the zero-field channels in terms of the field data:
`ŝ + q₀ = ελp₀(1 - κ)`, `ŝ' + q₀ = -ελp₀(1 + κ)`. -/
theorem shat_corner (hμ : F.μ = 0) (hh : F.h = 0) (ε : ℂ) :
    (F.lat 0).shat k ε + F.q₀L k = ε * F.lamL * F.p₀ * (1 - F.kapL) ∧
      (F.lat 0).shat' k ε + F.q₀L k = -(ε * F.lamL * F.p₀ * (1 + F.kapL)) := by
  have h₀ := F.lat_p₀ hh
  have h₁ := F.lat_p₁_up hh
  have hu' := F.pUp_of_plus (hμ.trans hh.symm)
  obtain ⟨-, -, -, hp, -⟩ := F.shiba_lat_weights hμ
  rw [hμ] at h₀ h₁
  simp only [Params.shat, Params.shat', Params.q₀, Params.w, q₀L, wL, lamL, lat_a, lat_ω, h₀, h₁,
    hu', hp]
  push_cast
  constructor <;> ring

/-! ### `LAxisGeneric` near the corner on `h = 0` -/

/-- **`LAxisGeneric` near the corner on `h = 0`**: if both zero-field channels of `(U, β, 0)`
satisfy the assumptions of Corollary "half filling" (`HFHyp`, with `ŝ' ≠ 0` for odd `k`), then
there are roots `v(s)` such that `LAxisGeneric` holds at every `μ = s ≠ 0` near `0` on `h = 0`.
The roots are those of the channels near `μ = 0` (`HFHyp.exists_eventually_dmGeneric`), as
`E₈ = e_d e_m` (`E8L_h0`), and `K_{t,-t} = cK_ε` with `c ≠ 0` there (`KL_opp_ne_h0`). -/
theorem eventually_lAxisGeneric_h0 {u u' : ℂ} (hd : (F.lat 0).HFHyp k 1 u)
    (hm : (F.lat 0).HFHyp k (-1) u') (hsd : Odd k → (F.lat 0).shat' k 1 ≠ 0)
    (hsm : Odd k → (F.lat 0).shat' k (-1) ≠ 0) :
    ∃ v : ℝ → Fin 4 → ℂ, ∀ᶠ s in 𝓝[≠] 0, (F.axis .h0 s).LAxisGeneric k (v s) := by
  have ha : (F.a : ℂ) ≠ 0 := hd.a_ne
  have hU : F.U ≠ 0 := fun h ↦ ha (by rw [a, h, zero_div, ofReal_zero])
  obtain ⟨u₁, u₂, hdd⟩ := hd.exists_eventually_Kph_ne hsd
  obtain ⟨w₁, w₂, hdm⟩ := hm.exists_eventually_Kph_ne hsm
  refine ⟨fun s ↦ ![u₁ s, u₂ s, w₁ s, w₂ s], ?_⟩
  filter_upwards [F.eventually_sep_off_axis k .h0 hU 0, hdd, hdm] with s hso ⟨hds, hKd⟩ ⟨hms, hKm⟩
  set G := F.axis .h0 s with hGdef
  set V : Fin 4 → ℂ := ![u₁ s, u₂ s, w₁ s, w₂ s]
  have hG : G.lat G.μ = (F.lat 0).withMu s := by
    simp only [hGdef, FParams.lat, Params.withMu, axis_μ, Axis.dμ, mul_one]
    rfl
  have hh : G.h = 0 := by simp [hGdef, Axis.dh]
  rw [← hG] at hds hms hKd hKm
  have hprod : ∀ v, G.E8L k v = ∏ j, (v ^ 2 - V j ^ 2) := fun v ↦ by
    rw [G.E8L_h0 k hh, (G.lat G.μ).eph_eq_mul k 1 hds.e_u₁ hds.e_u₂ hds.s_ne,
      (G.lat G.μ).eph_eq_mul k (-1) hms.e_u₁ hms.e_u₂ hms.s_ne, Fin.prod_univ_four]
    simp only [V, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.cons_val_three, Matrix.head_cons, Matrix.tail_cons]
    ring
  obtain ⟨sep, off⟩ := hso V hprod
  have root : ∀ j, G.E8L k (V j) = 0 := fun j ↦ by
    rw [hprod, Finset.prod_eq_zero_iff]
    exact ⟨j, Finset.mem_univ _, sub_self _⟩
  refine ⟨ha, root, sep, fun j _ σ l ↦ off j σ l, fun t j hj ↦ Or.inr ?_⟩
  have key : ∃ ε : ℂ, (ε = 1 ∨ ε = -1) ∧ (G.lat G.μ).eph k ε (V j) = 0 ∧
      (G.lat G.μ).Kph k ε (V j) ≠ 0 := by
    fin_cases j
    · exact ⟨1, hd.ε_eq, hds.e_u₁, hKd _ hds.e_u₁⟩
    · exact ⟨1, hd.ε_eq, hds.e_u₂, hKd _ hds.e_u₂⟩
    · exact ⟨-1, hm.ε_eq, hms.e_u₁, hKm _ hms.e_u₁⟩
    · exact ⟨-1, hm.ε_eq, hms.e_u₂, hKm _ hms.e_u₂⟩
  obtain ⟨ε, hε, he, hK⟩ := key
  have hq := G.sq_add_q₀L_ne_of_sep k root sep j
  exact ⟨G.KL_opp_ne_h0 k hh hε ha he hq hK t, hq⟩

/-! ### `LAxisGeneric` near the corner on `μ = 0` -/

/-- **`LAxisGeneric` near the corner on `μ = 0`**: if both zero-field channels of the Shiba image
`(-U, β, 0)` of the corner satisfy the assumptions of Corollary "half filling" (`HFHyp`, with
`ŝ' ≠ 0` for odd `k`), then there are roots `v(s)` such that `LAxisGeneric` holds at every
`h = s ≠ 0` near `0` on `μ = 0`. The roots are those of the Shiba channels near `h = 0`, as
`E₈ = e'_d e'_m` (`E8L_mu0`); `K_{↑↓}(u) = cK'_ε(u)` and `K_{↓↑}(u) = K_{↑↓}(-u)`
(`KL_opp_ne_mu0`, `KL_swap_mu0`), with `K'_ε ≠ 0` at every root (`HFHyp.exists_eventually_Kph_ne`). -/
theorem eventually_lAxisGeneric_mu0 {u u' : ℂ} (hd : (F.shiba.lat 0).HFHyp k 1 u)
    (hm : (F.shiba.lat 0).HFHyp k (-1) u') (hsd : Odd k → (F.shiba.lat 0).shat' k 1 ≠ 0)
    (hsm : Odd k → (F.shiba.lat 0).shat' k (-1) ≠ 0) :
    ∃ v : ℝ → Fin 4 → ℂ, ∀ᶠ s in 𝓝[≠] 0, (F.axis .mu0 s).LAxisGeneric k (v s) := by
  have ha : (F.a : ℂ) ≠ 0 := fun h ↦ hd.a_ne (by rw [lat_a, shiba_a, ofReal_neg, h, neg_zero])
  have hU : F.U ≠ 0 := fun h ↦ ha (by rw [a, h, zero_div, ofReal_zero])
  obtain ⟨u₁, u₂, hdd⟩ := hd.exists_eventually_Kph_ne hsd
  obtain ⟨w₁, w₂, hdm⟩ := hm.exists_eventually_Kph_ne hsm
  refine ⟨fun s ↦ ![u₁ s, u₂ s, w₁ s, w₂ s], ?_⟩
  filter_upwards [F.eventually_sep_off_axis k .mu0 hU 0, hdd, hdm] with s hso ⟨hds, hKd⟩ ⟨hms, hKm⟩
  set G := F.axis .mu0 s with hGdef
  set V : Fin 4 → ℂ := ![u₁ s, u₂ s, w₁ s, w₂ s]
  have hG : G.shiba.lat G.shiba.μ = (F.shiba.lat 0).withMu s := by
    simp only [hGdef, FParams.lat, FParams.shiba, Params.withMu, axis, Axis.dh, mul_one]
  have hμ : G.μ = 0 := by simp [hGdef, Axis.dμ]
  rw [← hG] at hds hms hKd hKm
  have hprod : ∀ v, G.E8L k v = ∏ j, (v ^ 2 - V j ^ 2) := fun v ↦ by
    rw [G.E8L_mu0 k hμ, (G.shiba.lat G.shiba.μ).eph_eq_mul k 1 hds.e_u₁ hds.e_u₂ hds.s_ne,
      (G.shiba.lat G.shiba.μ).eph_eq_mul k (-1) hms.e_u₁ hms.e_u₂ hms.s_ne, Fin.prod_univ_four]
    simp only [V, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.cons_val_three, Matrix.head_cons, Matrix.tail_cons]
    ring
  obtain ⟨sep, off⟩ := hso V hprod
  have root : ∀ j, G.E8L k (V j) = 0 := fun j ↦ by
    rw [hprod, Finset.prod_eq_zero_iff]
    exact ⟨j, Finset.mem_univ _, sub_self _⟩
  refine ⟨ha, root, sep, fun j _ σ l ↦ off j σ l, fun t j _ ↦ Or.inr ?_⟩
  have key : ∃ ε : ℂ, (ε = 1 ∨ ε = -1) ∧ (G.shiba.lat G.shiba.μ).eph k ε (V j) = 0 ∧
      ∀ v, (G.shiba.lat G.shiba.μ).eph k ε v = 0 → (G.shiba.lat G.shiba.μ).Kph k ε v ≠ 0 := by
    fin_cases j
    · exact ⟨1, hd.ε_eq, hds.e_u₁, hKd⟩
    · exact ⟨1, hd.ε_eq, hds.e_u₂, hKd⟩
    · exact ⟨-1, hm.ε_eq, hms.e_u₁, hKm⟩
    · exact ⟨-1, hm.ε_eq, hms.e_u₂, hKm⟩
  obtain ⟨ε, hε, he, hK⟩ := key
  have hq := G.sq_add_q₀L_ne_of_sep k root sep j
  refine ⟨?_, hq⟩
  cases t
  · have he' : (G.shiba.lat G.shiba.μ).eph k ε (-V j) = 0 := by rw [Params.eph_neg]; exact he
    rw [Bool.not_false, G.KL_swap_mu0 k hμ]
    exact G.KL_opp_ne_mu0 k hμ hε ha he' (by rwa [neg_sq]) (hK _ he')
  · exact G.KL_opp_ne_mu0 k hμ hε ha he hq (hK _ he)

end FParams

end HubbardAtom
