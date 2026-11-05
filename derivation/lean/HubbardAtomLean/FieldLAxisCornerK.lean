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

end FParams

end HubbardAtom
