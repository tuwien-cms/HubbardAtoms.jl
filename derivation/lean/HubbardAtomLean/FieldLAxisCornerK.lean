import HubbardAtomLean.FieldLAxisIsolated
import HubbardAtomLean.HalfFillingApproach
import HubbardAtomLean.FieldTZero
import HubbardAtomLean.FieldLAxisCorner

/-!
# The `K` conditions of `LAxisGeneric` near the corner

Near every axis point, the point itself excepted, `LAxisGeneric` reduces to its `K` conditions
(`eventually_lAxisGeneric_iff_axis`, `FieldLAxisIsolated.lean`). Here they are shown near the corner
`μ = h = 0` on each axis, at fixed `U`, `β`, `k`, under the assumptions of Corollary "half filling"
for both zero-field channels at the corner (`HFHyp`, with `ŝ' ≠ 0` for odd `k`). Not in the note.

* `h = 0`: `E₈ = e_d e_m` (`E8L_h0`) and `K_{t,-t} = cK_ε + εβH(-u)e_ε` with
  `c = -ε[H(u)H(-u) - λ²d_π]` (`KL_opp_h0`, a `ring` identity), where `c = 0` at a root of `e_ε`
  forces `u² = -q₀` (`KL_opp_ne_h0`), which simple roots exclude (`sq_add_q₀L_ne_of_sep`).
  The roots of the channels near `μ = 0` have `K_ε ≠ 0` at all four `±u₁`, `±u₂`
  (`Params.HFHyp.exists_eventually_Kph_ne`, with `Params.HFHyp.eventually_Psi_ne` at `±û`, as `D̂`
  is even at `μ = 0`). So `LAxisGeneric` holds near the corner on `h = 0`
  (`eventually_lAxisGeneric_h0`).
* `μ = 0`: the same with the channels of the Shiba image `(-U, β, h)` (`E8L_mu0`, `KL_opp_mu0`);
  there `K_{↓↑}(u) = K_{↑↓}(-u)` (`KL_swap_mu0`), so `K'_ε` is needed at `±u`
  (`eventually_lAxisGeneric_mu0`).
* `μ = h`: there `p↓ = κp₀`, `p↑ = κp₂` with `κ = e^{βU/2}` (`pDn_of_plus`, `pUp_of_plus`), and
  at a root of `E₈`, `N K = ρ(Φ + N S L)` for the column `σ = ↑`, `K_{↑↑}` and `K_{↓↑}`
  (`KL_UU_plus`, `KL_DU_plus`; `Φ` is a polynomial in the data, `DXUU`, …, `DrDU`, found by
  computer algebra and checked by `ring`). At the corner, at `û² = ŝ`,
  `Φ + N S L = 16p₀⁴λ⁴κ²(1 - κ)·2aûD̂(û)` up to the sign `ε` (`PhiUU_corner`, `PhiDU_corner`,
  `Params.two_a_beta_sub`), and at `√ŝ'`, `K = ±4βλ³κ(1 + κ)p₀³` (`KL_UU_corner'`,
  `KL_DU_corner'`). So with `û ≠ 0` for both channels, `LAxisGeneric` holds near the corner on
  `μ = h` (`eventually_lAxisGeneric_plus`), and on `μ = -h` by the spin flip (`KL_flip`,
  `LAxisGeneric.flip`, `axis_minus`, `eventually_lAxisGeneric_minus`).
* One set of assumptions for all four axes (`eventually_lAxisGeneric_corner`): `HFHyp` for both
  channels of `(U, β, 0)` gives it for those of `(-U, β, 0)`, as `ŝ(-U, ε) = ŝ(U, -ε)`,
  `ŝ'(-U, ε) = ŝ'(U, ε)`, `D̂(-U, ε) = D̂(U, -ε)` at `μ = 0` (`shiba_lat_zero_hat`,
  `HFHyp.shiba_lat_zero`).
* Approaches to the corner through the domain of Eq. (field-L): off the axes along the line
  `lineL` through each axis point where `LAxisGeneric` holds, Eq. (field-L) holds
  (`LAxisGeneric.tendsto_gammaL_lineL_gammaLAll`), so under the assumptions on `h = 0` there are
  points `y_i → 0` off the axes in the domain, along which
  `gammaL → (gammaHF_d + s_στ gammaHF_m)/2` (`exists_tendsto_gammaL_corner`, with
  `tendsto_gammaL_corner`).

Not shown: `μ = ±h` at `û = 0`. Away from the corner, the `K` conditions fail only at isolated
points of each axis (`eventually_exists_lAxisGeneric_of_corner`, `FieldLAxisKIsolated.lean`).
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

/-- At `u ≠ 0`: `2aβu - ελS_k(u) = 2auD̂(u)`, with `λ = 4a²`. -/
theorem two_a_beta_sub (A : Params) (k : ℤ) {ε u : ℂ} (hu : u ≠ 0) :
    2 * A.a * A.β * u - ε * (4 * A.a ^ 2) * A.S k u = 2 * A.a * u * A.Dhat k ε u := by
  simp only [Dhat, That, hu, ↓reduceIte]
  field_simp
  ring

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
    Params.cbar, HL, rUp, rDn, ρ, hp, sub_self, zero_sub, zero_add, piS, piP, lamL, q₀L, wL, cS,
    cP, ha, lat_ω, lat_β, shiba_ω, shiba_β, hρ, h₀, h₂, h₁]
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
  -2 * h ^ 2 * P ^ 2 * l ^ 2 * β + 2 * h ^ 2 * P ^ 2 * κ * l ^ 2 * β -
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
  2 * Y * P ^ 3 * κ ^ 2 * l ^ 3 * β - 2 * Y * P ^ 3 * l ^ 3 * β +
    2 * h * P ^ 3 * κ ^ 3 * l ^ 3 * β - 2 * h * P ^ 3 * κ * l ^ 3 * β -
    2 * h * Y ^ 2 * P * κ * l * β - 2 * h ^ 3 * P * κ * l * β -
    2 * X * Y ^ 2 * P * l * β + 2 * h ^ 2 * X * P * κ * l * β - 2 * h * X * Y * P * κ * l * β +
    2 * h * X * Y * P * l * β + 4 * h ^ 2 * Y * P * κ * l * β -
    3 / 4 * Y * P * r ^ 2 * κ ^ 2 * l ^ 3 * β - 1 / 4 * Y * P * r ^ 2 * κ ^ 3 * l ^ 3 * β +
    3 / 4 * Y * P * r ^ 2 * l ^ 3 * β + 1 / 4 * Y * P * r ^ 2 * κ * l ^ 3 * β +
    1 / 4 * h * P * r ^ 2 * κ ^ 2 * l ^ 3 * β - 1 / 4 * h * P * r ^ 2 * κ ^ 3 * l ^ 3 * β -
    1 / 4 * h * P * r ^ 2 * l ^ 3 * β + 1 / 4 * h * P * r ^ 2 * κ * l ^ 3 * β

/-- A part of `Φ` of `K_{↓↑}` on `μ = h` (`KL_DU_plus`). -/
noncomputable def DYDU (h Y P r κ l β : ℂ) : ℂ :=
  2 * h * P ^ 3 * κ ^ 2 * l ^ 3 * β + 2 * h * P ^ 3 * κ ^ 3 * l ^ 3 * β +
    2 * h * P ^ 3 * l ^ 3 * β + 2 * h * P ^ 3 * κ * l ^ 3 * β - 2 * h ^ 3 * P * κ * l * β -
    2 * h ^ 3 * P * l * β +
    2 * h ^ 2 * Y * P * κ * l * β + 2 * h ^ 2 * Y * P * l * β +
    1 / 4 * h * P * r ^ 2 * κ ^ 2 * l ^ 3 * β - 1 / 4 * h * P * r ^ 2 * κ ^ 3 * l ^ 3 * β -
    1 / 4 * h * P * r ^ 2 * l ^ 3 * β + 1 / 4 * h * P * r ^ 2 * κ * l ^ 3 * β

/-- A part of `Φ` of `K_{↓↑}` on `μ = h` (`KL_DU_plus`). -/
noncomputable def DrDU (h P r κ l β : ℂ) : ℂ :=
  -1 / 2 * P ^ 3 * κ ^ 2 * l ^ 5 * β - 1 / 2 * P ^ 3 * κ ^ 3 * l ^ 5 * β +
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

/-- At the corner and `u² + q₀ = ελp₀(1 - κ)`:
`Φ + N S 4λ²κP(u² + q₀) = 16p₀⁴λ⁴κ²(1 - κ)(2aβu - ελS)` for `K_{↑↑}`, at every value `S` of
`S↑(u)`. -/
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

/-! ### Continuity of the parts along a family -/

section Family

variable {P : Type*} [TopologicalSpace P] {U β μ h : P → ℝ} {hβ : ∀ x, 0 < β x} {k : ℤ}
  {x₀ : P} (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)
  (hh : ContinuousAt h x₀)

include hU hb hm hh

/-- `N`, `Φ` of `K_{↑↑}` and `K_{↓↑}`, and `L` of `K_{↓↑}`, at a moving point `v(x)`, are
continuous: they are polynomials in the data. -/
theorem continuousAt_famF_plus {v : P → ℂ} (hv : ContinuousAt v x₀) :
    ContinuousAt (fun x ↦ (famF U β μ h hβ x).NLp k (v x)) x₀ ∧
      ContinuousAt (fun x ↦ (famF U β μ h hβ x).PhiUU k (v x)) x₀ ∧
      ContinuousAt (fun x ↦ (famF U β μ h hβ x).PhiDU k (v x)) x₀ ∧
      ContinuousAt (fun x ↦ (famF U β μ h hβ x).LDU k (v x)) x₀ := by
  obtain ⟨ha, hβc, -, -, -⟩ := continuousAt_famF_data hU hb hm hh (hβ := hβ) (k := k)
  have hq := continuousAt_famF_q₀L hU hb hm hh (hβ := hβ) (k := k)
  have p₀ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₀ (by simp)
  have pu := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pUp (by simp)
  have pd := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.pDn (by simp)
  have p₂ := continuousAt_famF_p hU hb hm hh (hβ := hβ) FParams.p₂ (by simp)
  have hκ : ContinuousAt (fun x ↦ ((famF U β μ h hβ x).kapL : ℂ)) x₀ := by
    simp only [kapL, famF]
    fun_prop
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp only [NLp, PhiUU, PhiDU, LDU, DXUU, DYUU, DrUU, DXDU, DYDU, DrDU, HL, lamL, piP, piS, rUp,
      rDn, ρ, mf] <;>
    push_cast <;> fun_prop

end Family

/-! ### `LAxisGeneric` near the corner on `μ = h` -/

/-- **`LAxisGeneric` near the corner on `μ = h`**: if both zero-field channels of `(U, β, 0)`
satisfy the assumptions of Corollary "half filling" (`HFHyp`, with `ŝ' ≠ 0` for odd `k`) with
`û ≠ 0`, then there are roots `v(s)` such that `LAxisGeneric` holds at every `μ = h = s ≠ 0` near
`0`. The roots continue the roots `û`, `√ŝ'` of the channels at the corner
(`exists_famF_E8L_root`). At `û`, `K_{↑↑}` and `K_{↓↑}` are `ρΨ/N` (`KL_UU_plus`, `KL_DU_plus`)
with `Ψ → 16p₀⁴λ⁴κ²(1 - κ)·2aûD̂ ≠ 0` (`PhiUU_corner`, `PhiDU_corner`); at `√ŝ'` they tend to
`±4βλ³κ(1 + κ)p₀³ ≠ 0` (`KL_UU_corner'`, `KL_DU_corner'`). -/
theorem eventually_lAxisGeneric_plus {u u' : ℂ} (hd : (F.lat 0).HFHyp k 1 u)
    (hm : (F.lat 0).HFHyp k (-1) u') (hu0 : u ≠ 0) (hu0' : u' ≠ 0)
    (hsd : Odd k → (F.lat 0).shat' k 1 ≠ 0) (hsm : Odd k → (F.lat 0).shat' k (-1) ≠ 0) :
    ∃ v : ℝ → Fin 4 → ℂ, ∀ᶠ s in 𝓝[≠] 0, (F.axis .plus s).LAxisGeneric k (v s) := by
  -- the axis as a family, and its corner `F₀`
  have hUc : ContinuousAt (fun _ : ℝ ↦ F.U) 0 := continuousAt_const
  have hbc : ContinuousAt (fun _ : ℝ ↦ F.β) 0 := continuousAt_const
  have hmc : ContinuousAt (fun s : ℝ ↦ s * Axis.dμ .plus) 0 := by fun_prop
  have hhc : ContinuousAt (fun s : ℝ ↦ s * Axis.dh .plus) 0 := by fun_prop
  set F₀ := F.axis .plus 0 with hF₀
  have hμ₀ : F₀.μ = 0 := by simp [hF₀]
  have hh₀ : F₀.h = 0 := by simp [hF₀]
  have hl₀ : F₀.lat 0 = F.lat 0 := rfl
  have hB₀ : ∀ σ, F₀.BL σ = F.lat 0 := fun σ ↦ by
    cases σ <;> simp [hF₀, Bup, Bdn, axis_μ, axis_h, Axis.dμ, Axis.dh] <;> rfl
  have ha : (F.a : ℂ) ≠ 0 := hd.a_ne
  have hU : F.U ≠ 0 := fun h ↦ ha (by rw [a, h, zero_div, ofReal_zero])
  have hl : F₀.lamL ≠ 0 := by
    rw [axis_lamL, lamL]; exact mul_ne_zero (by norm_num) (pow_ne_zero 2 ha)
  have hp : (F₀.p₀ : ℂ) ≠ 0 := ofReal_ne_zero.mpr F₀.p₀_pos.ne'
  have hκ0 : (F₀.kapL : ℂ) ≠ 0 := ofReal_ne_zero.mpr (Real.exp_pos _).ne'
  have hκ1 : (1 - F₀.kapL : ℂ) ≠ 0 := by
    rw [sub_ne_zero, ne_comm, ← ofReal_one, ne_eq, ofReal_inj, kapL, Real.exp_eq_one_iff]
    exact div_ne_zero (mul_ne_zero F.β_pos.ne' hU) two_ne_zero
  have hκ1' : (1 + F₀.kapL : ℂ) ≠ 0 := by
    rw [← ofReal_one, ← ofReal_add]
    exact ofReal_ne_zero.mpr (by positivity [Real.exp_pos (F.β * F.U / 2)])
  -- the roots at the corner: `û`, `√ŝ'_d`, `û'`, `√ŝ'_m`
  obtain ⟨hsd₁, hsd₂⟩ := F₀.shat_corner k hμ₀ hh₀ 1
  obtain ⟨hsm₁, hsm₂⟩ := F₀.shat_corner k hμ₀ hh₀ (-1)
  rw [hl₀] at hsd₁ hsd₂ hsm₁ hsm₂
  set vd : ℂ := (F.lat 0).shat' k 1 ^ (2⁻¹ : ℂ)
  set vm : ℂ := (F.lat 0).shat' k (-1) ^ (2⁻¹ : ℂ)
  have hvd : vd ^ 2 = (F.lat 0).shat' k 1 := by simp [vd]
  have hvm : vm ^ 2 = (F.lat 0).shat' k (-1) := by simp [vm]
  set U₀ : Fin 4 → ℂ := ![u, vd, u', vm]
  set c : Fin 4 → ℂ := ![1 - F₀.kapL, -(1 + F₀.kapL), -(1 - F₀.kapL), 1 + F₀.kapL]
  have hsq : ∀ j, U₀ j ^ 2 + F₀.q₀L k = F₀.lamL * F₀.p₀ * c j := fun j ↦ by
    fin_cases j <;> simp only [U₀, c, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, Matrix.tail_cons]
    · rw [hd.u_sq]; linear_combination hsd₁
    · rw [hvd]; linear_combination hsd₂
    · rw [hm.u_sq]; linear_combination hsm₁
    · rw [hvm]; linear_combination hsm₂
  have hc : ∀ i j, i ≠ j → c i - c j ≠ 0 := fun i j hij ↦ by
    fin_cases i <;> fin_cases j <;> simp only [c, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, Matrix.tail_cons] at hij ⊢ <;>
      first
      | exact absurd rfl hij
      | (intro h; apply (two_ne_zero : (2 : ℂ) ≠ 0); linear_combination h)
      | (intro h; apply (two_ne_zero : (2 : ℂ) ≠ 0); linear_combination -h)
      | (intro h; apply hκ0; linear_combination -h / 2)
      | (intro h; apply hκ0; linear_combination h / 2)
      | (intro h; apply hκ1; linear_combination h / 2)
      | (intro h; apply hκ1; linear_combination -h / 2)
      | (intro h; apply hκ1'; linear_combination h / 2)
      | (intro h; apply hκ1'; linear_combination -h / 2)
  have sep₀ : ∀ i j, i ≠ j → U₀ i ^ 2 - U₀ j ^ 2 ≠ 0 := fun i j hij ↦ by
    rw [show U₀ i ^ 2 - U₀ j ^ 2 = F₀.lamL * F₀.p₀ * (c i - c j) by
      linear_combination hsq i - hsq j]
    exact mul_ne_zero (mul_ne_zero hl hp) (hc i j hij)
  have hE : ∀ v, F₀.E8L k v = (v ^ 2 - (F.lat 0).shat k 1) * (v ^ 2 - (F.lat 0).shat' k 1) *
      ((v ^ 2 - (F.lat 0).shat k (-1)) * (v ^ 2 - (F.lat 0).shat' k (-1))) := fun v ↦ by
    rw [F₀.E8L_h0 k hh₀, hμ₀, hl₀, (F.lat 0).eph_of_mu' k 1 rfl, (F.lat 0).eph_of_mu' k (-1) rfl]
  have root₀ : ∀ j, F₀.E8L k (U₀ j) = 0 := fun j ↦ by
    rw [hE]
    fin_cases j <;> simp only [U₀, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three, Matrix.head_cons,
      Matrix.tail_cons]
    · rw [hd.u_sq, sub_self, zero_mul, zero_mul]
    · rw [hvd, sub_self, mul_zero, zero_mul]
    · rw [hm.u_sq, sub_self, zero_mul, mul_zero]
    · rw [hvm, sub_self, mul_zero, mul_zero]
  -- the roots along the axis
  have hpath : ∀ j, ∃ V : ℝ → ℂ, V 0 = U₀ j ∧ ContinuousAt V 0 ∧
      ∀ᶠ s in 𝓝 0, (F.axis .plus s).E8L k (V s) = 0 := fun j ↦ by
    obtain ⟨V, h0, hc, hr⟩ := exists_famF_E8L_root (hβ := fun _ ↦ F.β_pos) (k := k) hUc hbc hmc
      hhc (u₀ := U₀ j) ha (root₀ j) (by
        change ∑ i : Fin 5, (i : ℕ) * F₀.E8c k i * (U₀ j ^ 2) ^ ((i : ℕ) - 1) ≠ 0
        rw [F₀.sum_E8c_deriv k root₀ sep₀ j]
        exact dprod_ne sep₀ j)
    exact ⟨V, h0, hc, hr.mono fun s h ↦ h.1⟩
  choose V hV0 hVc hVr using hpath
  have hroot : ∀ᶠ s in 𝓝 0, ∀ j, (F.axis .plus s).E8L k (V j s) = 0 :=
    Filter.eventually_all.2 hVr
  have hsep : ∀ᶠ s in 𝓝 0, ∀ i j, i ≠ j → V i s ^ 2 - V j s ^ 2 ≠ 0 :=
    Filter.eventually_all.2 fun i ↦ Filter.eventually_all.2 fun j ↦ by
      by_cases hij : i = j
      · exact Eventually.of_forall fun _ h ↦ absurd hij h
      · refine ((((hVc i).pow 2).sub ((hVc j).pow 2)).eventually_ne ?_).mono fun _ h _ ↦ h
        change V i 0 ^ 2 - V j 0 ^ 2 ≠ 0
        rw [hV0, hV0]
        exact sep₀ i j hij
  -- the corner roots are off both lattices
  have hz₀ : ∀ n, (F.lat 0).z k n ^ 2 - (F.lat 0).shat k 1 ≠ 0 := hd.z_sq_ne
  have hz₁ := Params.z_sq_ne'_of (A := F.lat 0) (k := k) (ε := 1) rfl hsd hd.odd_ne
  have hz₂ : ∀ n, (F.lat 0).z k n ^ 2 - (F.lat 0).shat k (-1) ≠ 0 := hm.z_sq_ne
  have hz₃ := Params.z_sq_ne'_of (A := F.lat 0) (k := k) (ε := -1) rfl hsm hm.odd_ne
  have hoff : ∀ j t n, (F₀.BL t).z k n ≠ U₀ j := fun j t n hn ↦ by
    rw [hB₀] at hn
    fin_cases j <;> simp only [U₀, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three, Matrix.head_cons,
      Matrix.tail_cons] at hn
    · exact hz₀ n (by rw [hn, hd.u_sq, sub_self])
    · exact hz₁ n (by rw [hn, hvd, sub_self])
    · exact hz₂ n (by rw [hn, hm.u_sq, sub_self])
    · exact hz₃ n (by rw [hn, hvm, sub_self])
  -- continuity of the data along the axis
  have hqc : ContinuousAt (fun s ↦ (F.axis .plus s).q₀L k) 0 := continuousAt_const
  have hlc : ContinuousAt (fun s ↦ (F.axis .plus s).lamL) 0 := continuousAt_const
  have hκc : ContinuousAt (fun s ↦ ((F.axis .plus s).kapL : ℂ)) 0 := continuousAt_const
  have hp₀c : ContinuousAt (fun s ↦ ((F.axis .plus s).p₀ : ℂ)) 0 :=
    continuousAt_famF_p hUc hbc hmc hhc (hβ := fun _ ↦ F.β_pos) FParams.p₀ (by simp)
  have hp₂c : ContinuousAt (fun s ↦ ((F.axis .plus s).p₂ : ℂ)) 0 :=
    continuousAt_famF_p hUc hbc hmc hhc (hβ := fun _ ↦ F.β_pos) FParams.p₂ (by simp)
  -- at the roots `û`, `û'`: `N ≠ 0` and `Ψ ≠ 0` near the corner
  have hvan : ∀ j ε, (ε = 1 ∨ ε = -1) → U₀ j ^ 2 + F₀.q₀L k = ε * F₀.lamL * F₀.p₀ * (1 - F₀.kapL) →
      U₀ j ≠ 0 → (F.lat 0).Dhat k ε (U₀ j) ≠ 0 →
      ∀ᶠ s in 𝓝 0, (F.axis .plus s).PhiUU k (V j s) + (F.axis .plus s).NLp k (V j s) *
          ((F.axis .plus s).BL true).S k (V j s) * (4 * (F.axis .plus s).lamL ^ 2 *
            (F.axis .plus s).kapL * (((F.axis .plus s).p₀ + (F.axis .plus s).p₂) / 2) *
            (V j s ^ 2 + (F.axis .plus s).q₀L k)) ≠ 0 ∧
        (F.axis .plus s).PhiDU k (V j s) + (F.axis .plus s).NLp k (V j s) *
          ((F.axis .plus s).BL false).S k (V j s) * (F.axis .plus s).LDU k (V j s) ≠ 0 := by
    intro j ε hε hu hu0 hD
    obtain ⟨hN, hΦ, hΦ', hL⟩ := continuousAt_famF_plus hUc hbc hmc hhc (hβ := fun _ ↦ F.β_pos)
      (k := k) (hVc j)
    have hS := continuousAt_famF_S hbc hmc hhc (hβ := fun _ ↦ F.β_pos) (U := fun _ ↦ F.U)
      (k := k) true (hVc j) (fun n ↦ by rw [hV0]; exact hoff j true n)
    have hS' := continuousAt_famF_S hbc hmc hhc (hβ := fun _ ↦ F.β_pos) (U := fun _ ↦ F.U)
      (k := k) false (hVc j) (fun n ↦ by rw [hV0]; exact hoff j false n)
    have hS₀ : ∀ σ, (F₀.BL σ).S k (U₀ j) = (F.lat 0).S k (U₀ j) := fun σ ↦ by rw [hB₀]
    have hval : 2 * F₀.a * F₀.β * U₀ j - ε * F₀.lamL * (F.lat 0).S k (U₀ j) ≠ 0 := by
      have := (F.lat 0).two_a_beta_sub k (ε := ε) hu0
      rw [lat_a, lat_β] at this
      rw [axis_a, axis_β, axis_lamL, lamL, this]
      exact mul_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero ha) hu0) hD
    have hc16 : (16 : ℂ) * F₀.p₀ ^ 4 * F₀.lamL ^ 4 * F₀.kapL ^ 2 * (1 - F₀.kapL) ≠ 0 :=
      mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (by norm_num) (pow_ne_zero 4 hp))
        (pow_ne_zero 4 hl)) (pow_ne_zero 2 hκ0)) hκ1
    have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
    refine (ContinuousAt.eventually_ne ?_ ?_).and (ContinuousAt.eventually_ne ?_ ?_)
    · exact hΦ.add ((hN.mul hS).mul (by fun_prop))
    · change F₀.PhiUU k (V j 0) + F₀.NLp k (V j 0) * (F₀.BL true).S k (V j 0) *
        (4 * F₀.lamL ^ 2 * F₀.kapL * ((F₀.p₀ + F₀.p₂) / 2) * (V j 0 ^ 2 + F₀.q₀L k)) ≠ 0
      rw [hV0, F₀.PhiUU_corner k hμ₀ hh₀ hε hu, hS₀]
      exact mul_ne_zero hc16 hval
    · exact hΦ'.add ((hN.mul hS').mul hL)
    · change F₀.PhiDU k (V j 0) + F₀.NLp k (V j 0) * (F₀.BL false).S k (V j 0) *
        F₀.LDU k (V j 0) ≠ 0
      rw [hV0, F₀.PhiDU_corner k hμ₀ hh₀ hε hu, hS₀]
      exact mul_ne_zero (mul_ne_zero hc16 hε0) hval
  -- at the roots `√ŝ'`: `K_{↑↑}`, `K_{↓↑}` tend to `±4βλ³κ(1 + κ)p₀³ ≠ 0`
  have hβ : (F₀.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr F₀.β_pos.ne'
  have hc4 : (4 : ℂ) * F₀.β * F₀.lamL ^ 3 * F₀.kapL * (1 + F₀.kapL) * F₀.p₀ ^ 3 ≠ 0 :=
    mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (by norm_num) hβ)
      (pow_ne_zero 3 hl)) hκ0) hκ1') (pow_ne_zero 3 hp)
  have hnon : ∀ j ε, (ε = 1 ∨ ε = -1) →
      U₀ j ^ 2 + F₀.q₀L k = -(ε * F₀.lamL * F₀.p₀ * (1 + F₀.kapL)) →
      ∀ᶠ s in 𝓝 0, (F.axis .plus s).KL k true true (V j s) ≠ 0 ∧
        (F.axis .plus s).KL k false true (V j s) ≠ 0 := by
    intro j ε hε hu
    have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
    have h₁ : ContinuousAt (fun s ↦ (F.axis .plus s).KL k true true (V j s)) 0 :=
      continuousAt_famF_KL hUc hbc hmc hhc (hβ := fun _ ↦ F.β_pos) (k := k) true true (hVc j)
        (fun n ↦ by rw [hV0]; exact hoff j true n)
    have h₂ : ContinuousAt (fun s ↦ (F.axis .plus s).KL k false true (V j s)) 0 :=
      continuousAt_famF_KL hUc hbc hmc hhc (hβ := fun _ ↦ F.β_pos) (k := k) false true (hVc j)
        (fun n ↦ by rw [hV0]; exact hoff j false n)
    refine (h₁.eventually_ne ?_).and (h₂.eventually_ne ?_)
    · change F₀.KL k true true (V j 0) ≠ 0
      rw [hV0, F₀.KL_UU_corner' k hμ₀ hh₀ hε hu]
      refine ne_of_eq_of_ne (by ring) (mul_ne_zero hε0 hc4)
    · change F₀.KL k false true (V j 0) ≠ 0
      rw [hV0, F₀.KL_DU_corner' k hμ₀ hh₀ hε hu]
      exact hc4
  have e₀ := hvan 0 1 (Or.inl rfl) (by rw [hsq]; simp [c]) hu0 hd.D_ne
  have e₂ := hvan 2 (-1) (Or.inr rfl) (by rw [hsq]; simp [c]; ring) hu0' hm.D_ne
  have e₁ := hnon 1 1 (Or.inl rfl) (by rw [hsq]; simp [c]; ring)
  have e₃ := hnon 3 (-1) (Or.inr rfl) (by rw [hsq]; simp [c])
  refine ⟨fun s j ↦ V j s, ?_⟩
  filter_upwards [F.eventually_sep_off_axis k .plus hU 0, nhdsWithin_le_nhds hroot,
    nhdsWithin_le_nhds hsep, nhdsWithin_le_nhds e₀, nhdsWithin_le_nhds e₁,
    nhdsWithin_le_nhds e₂, nhdsWithin_le_nhds e₃, self_mem_nhdsWithin]
    with s hso hr hs h₀ h₁ h₂ h₃ hs0
  set G := F.axis .plus s with hGdef
  have hμh : G.μ = G.h := by simp [hGdef, Axis.dμ, Axis.dh]
  have hρ : (G.ρ : ℂ) ≠ 0 := G.rho_ne_zeroT (by simpa [hGdef, Axis.dμ] using hs0)
  have hprod : ∀ v, G.E8L k v = ∏ j, (v ^ 2 - V j s ^ 2) := fun v ↦ by
    rw [G.E8L_eq_prod k hr hs v, Fin.prod_univ_four]
  obtain ⟨-, off⟩ := hso (fun j ↦ V j s) hprod
  have hvan' : ∀ j, G.PhiUU k (V j s) + G.NLp k (V j s) * (G.BL true).S k (V j s) *
        (4 * G.lamL ^ 2 * G.kapL * ((G.p₀ + G.p₂) / 2) * (V j s ^ 2 + G.q₀L k)) ≠ 0 →
      G.PhiDU k (V j s) + G.NLp k (V j s) * (G.BL false).S k (V j s) * G.LDU k (V j s) ≠ 0 →
      G.KL k true true (V j s) ≠ 0 ∧ G.KL k false true (V j s) ≠ 0 := fun j hΨ hΨ' ↦
    ⟨fun hK ↦ by
      have := G.KL_UU_plus k hμh (hr j)
      rw [hK, mul_zero] at this
      exact mul_ne_zero hρ hΨ this.symm,
    fun hK ↦ by
      have := G.KL_DU_plus k hμh (hr j)
      rw [hK, mul_zero] at this
      exact mul_ne_zero hρ hΨ' this.symm⟩
  have hK : ∀ j, G.KL k true true (V j s) ≠ 0 ∧ G.KL k false true (V j s) ≠ 0
    | 0 => hvan' 0 h₀.1 h₀.2
    | 1 => h₁
    | 2 => hvan' 2 h₂.1 h₂.2
    | 3 => h₃
  refine ⟨ha, hr, hs, fun j _ σ l ↦ off j σ l, fun t j _ ↦ ?_⟩
  cases t
  · exact Or.inr ⟨(hK j).2, G.sq_add_q₀L_ne_of_sep k hr hs j⟩
  · exact Or.inl (hK j).1

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

/-! ### The channels of the Shiba image at the corner -/

/-- At `μ = 0`, the weights of `(-U, β, 0)` are those of `(U, β, 0)` with `p₀ ↔ p₁`. -/
theorem shiba_lat_zero_p :
    (F.shiba.lat 0).p₀ = (F.lat 0).p₁ ∧ (F.shiba.lat 0).p₁ = (F.lat 0).p₀ := by
  have hZ : (F.shiba.lat 0).Z = (F.lat 0).Z := by
    simp only [Params.Z, Params.boltz₀, Params.boltz₁, Params.boltz₂, lat_U, lat_β, lat_μ,
      shiba_U, shiba_β]
    ring_nf
  simp only [Params.p₀, Params.p₁, hZ, Params.boltz₀, Params.boltz₁, lat_U, lat_β, lat_μ, shiba_U,
    shiba_β]
  constructor <;> ring_nf

/-- At `μ = 0`: `ŝ(-U, ε) = ŝ(U, -ε)`, `ŝ'(-U, ε) = ŝ'(U, ε)`, `D̂(-U, ε) = D̂(U, -ε)`. -/
theorem shiba_lat_zero_hat (ε u : ℂ) :
    (F.shiba.lat 0).shat k ε = (F.lat 0).shat k (-ε) ∧
      (F.shiba.lat 0).shat' k ε = (F.lat 0).shat' k ε ∧
      (F.shiba.lat 0).Dhat k ε u = (F.lat 0).Dhat k (-ε) u := by
  obtain ⟨h₀, h₁⟩ := F.shiba_lat_zero_p
  have ha : ((F.shiba.lat 0).a : ℂ) = -(F.lat 0).a := by rw [lat_a, lat_a, shiba_a, ofReal_neg]
  have hq : (F.shiba.lat 0).q₀ k = (F.lat 0).q₀ k := by
    simp only [Params.q₀, Params.w, lat_a, lat_ω, shiba_a, shiba_ω]
    push_cast
    ring
  have hT : (F.shiba.lat 0).That k u = (F.lat 0).That k u := by
    simp only [Params.That, Params.S, lat_β, lat_μ, shiba_β]
  refine ⟨?_, ?_, ?_⟩
  · simp only [Params.shat, hq, ha, h₀, h₁]
    ring
  · simp only [Params.shat', hq, ha, h₀, h₁]
    ring
  · simp only [Params.Dhat, hT, ha, lat_β, shiba_β]
    ring

/-- **`HFHyp` for both channels of `(U, β, 0)` gives it for those of `(-U, β, 0)`**: the channel
`ε` of `(-U, β, 0)` satisfies `HFHyp` at the root `û` of the channel `-ε` of `(U, β, 0)`; its `ŝ'`
is that of the channel `ε` (`shiba_lat_zero_hat`). -/
theorem HFHyp.shiba_lat_zero {ε u u' : ℂ} (h : (F.lat 0).HFHyp k (-ε) u)
    (h' : (F.lat 0).HFHyp k ε u') : (F.shiba.lat 0).HFHyp k ε u where
  ε_eq := h'.ε_eq
  a_ne := by rw [lat_a, shiba_a, ofReal_neg, neg_ne_zero]; exact h.a_ne
  μ_eq := rfl
  z_sq_ne n := by rw [(F.shiba_lat_zero_hat k ε u).1]; exact h.z_sq_ne n
  odd_ne n hn := by rw [(F.shiba_lat_zero_hat k ε u).2.1]; exact h'.odd_ne n hn
  u_sq := by rw [(F.shiba_lat_zero_hat k ε u).1]; exact h.u_sq
  D_ne := by rw [(F.shiba_lat_zero_hat k ε u).2.2]; exact h.D_ne

/-! ### `LAxisGeneric` near the corner on `μ = 0` -/

/-- **`LAxisGeneric` near the corner on `μ = 0`**: if both zero-field channels of the Shiba image
`(-U, β, 0)` of the corner satisfy the assumptions of Corollary "half filling" (`HFHyp`, with
`ŝ' ≠ 0` for odd `k`), then there are roots `v(s)` such that `LAxisGeneric` holds at every
`h = s ≠ 0` near `0` on `μ = 0`. The roots are those of the Shiba channels near `h = 0`, as
`E₈ = e'_d e'_m` (`E8L_mu0`); `K_{↑↓}(u) = cK'_ε(u)` and `K_{↓↑}(u) = K_{↑↓}(-u)`
(`KL_opp_ne_mu0`, `KL_swap_mu0`), with `K'_ε ≠ 0` at every root
(`HFHyp.exists_eventually_Kph_ne`). -/
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

/-! ### On `μ = -h`, by the spin flip -/

/-- The spin flip exchanges the spins of `K`: `K^flip_{t,σ} = K_{-t,-σ}`. -/
theorem KL_flip (t σ : Bool) (u : ℂ) : F.flip.KL k t σ u = F.KL k (!t) (!σ) u := by
  have hpi : F.flip.piS = F.piS := by simp only [piS, flip_pUp, flip_pDn]; ring
  have hpP : F.flip.piP = F.piP := by simp only [piP, flip_p₀, flip_p₂]
  have hcS : F.flip.cS = F.cS := by simp only [cS, flip_pUp, flip_pDn, flip_β]; ring
  have hcP : F.flip.cP = F.cP := by simp only [cP, flip_p₀, flip_p₂, flip_β]
  have hD : ∀ σ, F.flip.DL σ = F.DL (!σ) := fun σ ↦ by simp only [DL, flip_pL, flip_p₀, flip_p₂]
  have hc : ∀ σ, F.flip.cLw σ = F.cLw (!σ) := fun σ ↦ by simp only [cLw, flip_pL, flip_p₀, flip_p₂]
  cases t <;> cases σ <;>
    simp only [KL, fMp, fMm, QL, αL, αSL, αPL, HL, flip_rL, flip_mL, flip_BL, flip_ρ, flip_lamL,
      flip_a, flip_q₀L, flip_β, hpi, hpP, hcS, hcP, hD, hc, Bool.not_true, Bool.not_false,
      Bool.true_eq_false, Bool.false_eq_true, ↓reduceIte]

/-- `LAxisGeneric` is invariant under the spin flip. -/
theorem LAxisGeneric.flip {u : Fin 4 → ℂ} (h : F.LAxisGeneric k u) : F.flip.LAxisGeneric k u := by
  have hE : ∀ x, F.flip.E8L k x = F.E8L k x := fun x ↦ by
    rw [F.flip.E8L_eq k true, F.E8L_eq k false]
    simp only [flip_rL, flip_lamL, flip_q₀L, HL, flip_a, Bool.not_true, Bool.not_false, piS, piP,
      flip_pUp, flip_pDn, flip_p₀, flip_p₂]
    ring
  refine ⟨h.a_ne, fun j ↦ (hE _).trans (h.root j), h.sep, fun j hj σ l ↦ ?_, fun t j hj ↦ ?_⟩
  · rw [flip_BL]
    exact h.off j hj (!σ) l
  · have := h.K_ne (!t) j hj
    rw [Bool.not_not] at this
    rwa [KL_flip, KL_flip, Bool.not_not, flip_q₀L]

/-- `μ = -h` is the spin flip of `μ = h`. -/
theorem axis_minus (s : ℝ) : F.axis .minus s = (F.axis .plus s).flip := by
  simp only [axis, flip, Axis.dμ, Axis.dh, mul_one, mul_neg]

/-- **`LAxisGeneric` near the corner on `μ = -h`**, under the assumptions of
`eventually_lAxisGeneric_plus`, by the spin flip (`LAxisGeneric.flip`, `axis_minus`). -/
theorem eventually_lAxisGeneric_minus {u u' : ℂ} (hd : (F.lat 0).HFHyp k 1 u)
    (hm : (F.lat 0).HFHyp k (-1) u') (hu0 : u ≠ 0) (hu0' : u' ≠ 0)
    (hsd : Odd k → (F.lat 0).shat' k 1 ≠ 0) (hsm : Odd k → (F.lat 0).shat' k (-1) ≠ 0) :
    ∃ v : ℝ → Fin 4 → ℂ, ∀ᶠ s in 𝓝[≠] 0, (F.axis .minus s).LAxisGeneric k (v s) := by
  obtain ⟨v, hv⟩ := F.eventually_lAxisGeneric_plus k hd hm hu0 hu0' hsd hsm
  exact ⟨v, hv.mono fun s h ↦ by rw [axis_minus]; exact h.flip⟩

/-! ### All four axes under one set of assumptions -/

/-- **`LAxisGeneric` near the corner on every axis**: if both zero-field channels of `(U, β, 0)`
satisfy the assumptions of Corollary "half filling" (`HFHyp`, with `ŝ' ≠ 0` for odd `k`), with
`û ≠ 0` for both on `μ = ±h`, then on each axis `d` there are roots `v(s)` such that
`LAxisGeneric` holds at every axis point `s ≠ 0` near the corner. On `μ = 0` the channels of
`(-U, β, 0)` satisfy `HFHyp` too (`HFHyp.shiba_lat_zero`). -/
theorem eventually_lAxisGeneric_corner (d : Axis) {u u' : ℂ} (hd : (F.lat 0).HFHyp k 1 u)
    (hm : (F.lat 0).HFHyp k (-1) u') (hsd : Odd k → (F.lat 0).shat' k 1 ≠ 0)
    (hsm : Odd k → (F.lat 0).shat' k (-1) ≠ 0) (hu : d = .plus ∨ d = .minus → u ≠ 0 ∧ u' ≠ 0) :
    ∃ v : ℝ → Fin 4 → ℂ, ∀ᶠ s in 𝓝[≠] 0, (F.axis d s).LAxisGeneric k (v s) := by
  cases d
  · exact F.eventually_lAxisGeneric_h0 k hd hm hsd hsm
  · have hd' := HFHyp.shiba_lat_zero F k (ε := 1) hm hd
    have hm' := HFHyp.shiba_lat_zero F k (ε := -1) (by rwa [neg_neg]) hm
    exact F.eventually_lAxisGeneric_mu0 k hd' hm'
      (by rw [(F.shiba_lat_zero_hat k 1 u').2.1]; exact hsd)
      (by rw [(F.shiba_lat_zero_hat k (-1) u).2.1]; exact hsm)
  · obtain ⟨hu, hu'⟩ := hu (Or.inl rfl)
    exact F.eventually_lAxisGeneric_plus k hd hm hu hu' hsd hsm
  · obtain ⟨hu, hu'⟩ := hu (Or.inr rfl)
    exact F.eventually_lAxisGeneric_minus k hd hm hu hu' hsd hsm

/-! ### Approaches to the corner through the domain of Eq. (field-L) -/

/-- **Approaches to the corner through the domain of Eq. (field-L) exist**: under the assumptions
of `eventually_lAxisGeneric_h0` (`HFHyp` for both channels of `(U, β, 0)`, with `ŝ' ≠ 0` for odd
`k`), there are points `y_i → 0` in the `(μ, h)` plane and roots `v_i` such that Eq. (field-L)
holds at `(U, β, y_i)` for both column spins, and along them
`gammaL → (gammaHF_d + s_στ gammaHF_m)/2` (`tendsto_gammaL_corner`). -/
theorem exists_tendsto_gammaL_corner {u u' : ℂ} (hd : (F.lat 0).HFHyp k 1 u)
    (hm : (F.lat 0).HFHyp k (-1) u') (hsd : Odd k → (F.lat 0).shat' k 1 ≠ 0)
    (hsm : Odd k → (F.lat 0).shat' k (-1) ≠ 0) :
    ∃ (y : ℕ → ℝ × ℝ) (v : ℕ → Fin 4 → ℂ), Tendsto y atTop (𝓝 0) ∧
      (∀ᶠ i in atTop, ∀ t, (famF (fun _ ↦ F.U) (fun _ ↦ F.β) Prod.fst Prod.snd
        (fun _ ↦ F.β_pos) (y i)).LGeneric k t (v i)) ∧
      ∀ σ n τ m, Tendsto (fun i ↦ (famF (fun _ ↦ F.U) (fun _ ↦ F.β) Prod.fst Prod.snd
        (fun _ ↦ F.β_pos) (y i)).gammaL k (v i) σ n τ m) atTop
        (𝓝 (((F.lat 0).gammaHF k 1 u n m + sL σ τ * (F.lat 0).gammaHF k (-1) u' n m) / 2)) := by
  set Fy := famF (fun _ ↦ F.U) (fun _ ↦ F.β) Prod.fst Prod.snd (fun _ ↦ F.β_pos)
  obtain ⟨w, hw⟩ := F.eventually_lAxisGeneric_h0 k hd hm hsd hsm
  -- near every axis point `(s, 0)` there is a point of the domain within `3|s|` of the corner
  have hQ : ∀ᶠ s in 𝓝[≠] (0 : ℝ), ∃ (y : ℝ × ℝ) (v : Fin 4 → ℂ), ‖y‖ ≤ 3 * |s| ∧
      ∀ t, (Fy y).LGeneric k t v := by
    filter_upwards [hw, self_mem_nhdsWithin] with s hs hs0
    obtain ⟨u₁, -, -, hu₁, -⟩ := hs.tendsto_gammaL_lineL_gammaLAll
    have h1 : ∀ᶠ r in 𝓝 (0 : ℝ), |r| < |s| :=
      (continuous_abs.tendsto 0).eventually (gt_mem_nhds (by rw [abs_zero]; exact abs_pos.2 hs0))
    have hr : ∀ᶠ r in 𝓝[≠] (0 : ℝ), |r| < |s| ∧ ∀ t, ((F.axis .h0 s).lineL r).LGeneric k t (u₁ r) :=
      Filter.Eventually.and (nhdsWithin_le_nhds h1 : ∀ᶠ r in 𝓝[≠] (0 : ℝ), |r| < |s|) hu₁
    obtain ⟨r, hr₁, hr₂⟩ := hr.exists
    refine ⟨(s * Axis.dμ .h0 + r, s * Axis.dh .h0 + 2 * r), u₁ r, ?_, hr₂⟩
    simp only [Axis.dμ, Axis.dh, mul_one, mul_zero, zero_add, Prod.norm_def, Real.norm_eq_abs]
    refine max_le ?_ ?_
    · linarith [abs_add_le s r, abs_nonneg s]
    · rw [abs_mul, abs_two]; linarith [abs_nonneg s]
  -- along `s_i = 1/(i + 1)`
  have hs : Tendsto (fun i : ℕ ↦ 1 / ((i : ℝ) + 1)) atTop (𝓝[≠] 0) :=
    tendsto_nhdsWithin_iff.2 ⟨tendsto_one_div_add_atTop_nhds_zero_nat,
      Eventually.of_forall fun i ↦ (by positivity : (0 : ℝ) < 1 / ((i : ℝ) + 1)).ne'⟩
  obtain ⟨y, hy⟩ := (hs.eventually hQ).choice
  obtain ⟨v, hv⟩ := hy.choice
  have hy : Tendsto y atTop (𝓝 0) := by
    refine squeeze_zero_norm' (hv.mono fun i h ↦ h.1) ?_
    have := (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).abs.const_mul (3 : ℝ)
    rwa [abs_zero, mul_zero] at this
  have hg : ∀ᶠ i in atTop, ∀ t, (Fy (y i)).LGeneric k t (v i) := hv.mono fun i h ↦ h.2
  exact ⟨y, v, hy, hg, fun σ n τ m ↦ tendsto_gammaL_corner continuousAt_const continuousAt_const
    continuous_fst.continuousAt continuous_snd.continuousAt rfl rfl hd hm hy hg σ n τ m⟩

end FParams

end HubbardAtom
