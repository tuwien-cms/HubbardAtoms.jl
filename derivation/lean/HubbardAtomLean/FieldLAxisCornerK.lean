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

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- At `k = 0`, `P(v) = (v² - a²)²`. -/
theorem Pph_zero (v : ℂ) : A.Pph 0 v = (v ^ 2 - A.a ^ 2) ^ 2 := by
  simp only [Pph, Qp, Qm, w_zero]
  ring

/-- `ModeOK` with `P ≠ 0` is `K ≠ 0`. -/
theorem ModeOK.Kph_ne {u : ℂ} (h : A.ModeOK k ε u) (hP : A.Pph k u ≠ 0) : A.Kph k ε u ≠ 0 :=
  h.elim And.right fun h ↦ absurd h.1 hP

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
  have h0 : (F.lat 0).μ = 0 := rfl
  obtain ⟨u₁, u₂, hu₁, hu₂, hc₁, hc₂, hdd⟩ := hd.exists_eventually_dmGeneric hsd
  obtain ⟨w₁, w₂, hw₁, hw₂, hc₁', hc₂', hdm⟩ := hm.exists_eventually_dmGeneric hsm
  rw [h0] at hu₁ hu₂ hw₁ hw₂ hc₁ hc₂ hc₁' hc₂' hdd hdm
  -- `P ≠ 0` at the four roots near the corner, for `k ≠ 0`
  have hP : ∀ᶠ s in 𝓝 (0 : ℝ), k ≠ 0 → (F.lat 0).Pph k (u₁ s) ≠ 0 ∧
      (F.lat 0).Pph k (u₂ s) ≠ 0 ∧ (F.lat 0).Pph k (w₁ s) ≠ 0 ∧ (F.lat 0).Pph k (w₂ s) ≠ 0 := by
    by_cases hk : k = 0
    · exact Eventually.of_forall fun _ h ↦ absurd hk h
    have hc : ∀ {f : ℝ → ℂ}, ContinuousAt f 0 →
        ContinuousAt (fun s ↦ (F.lat 0).Pph k (f s)) 0 := fun hf ↦ by
      simp only [Params.Pph, Params.Qp, Params.Qm]
      fun_prop
    have hk' : ∀ ε : ℂ, ¬(k = 0 ∧ ε = -1) := fun _ h ↦ hk h.1
    filter_upwards [(hc hc₁).eventually_ne (by
        rw [hu₁]; exact Params.Pph_ne_of_shat (A := F.lat 0) k 1 hd.ε_eq ha h0 hd.u_sq),
      (hc hc₂).eventually_ne (Params.Pph_ne_of_shat' (A := F.lat 0) (k := k) hd.ε_eq ha h0
        (hk' 1) hu₂),
      (hc hc₁').eventually_ne (by
        rw [hw₁]; exact Params.Pph_ne_of_shat (A := F.lat 0) k (-1) hm.ε_eq ha h0 hm.u_sq),
      (hc hc₂').eventually_ne (Params.Pph_ne_of_shat' (A := F.lat 0) (k := k) hm.ε_eq ha h0
        (hk' (-1)) hw₂)] with s h₁ h₂ h₃ h₄ _
    exact ⟨h₁, h₂, h₃, h₄⟩
  refine ⟨fun s ↦ ![u₁ s, u₂ s, w₁ s, w₂ s], ?_⟩
  filter_upwards [F.eventually_sep_off_axis k .h0 hU 0, hdd, hdm, nhdsWithin_le_nhds hP]
    with s hso hds hms hPs
  set G := F.axis .h0 s with hGdef
  set V : Fin 4 → ℂ := ![u₁ s, u₂ s, w₁ s, w₂ s]
  have hG : G.lat G.μ = (F.lat 0).withMu s := by
    simp only [hGdef, FParams.lat, Params.withMu, axis_μ, Axis.dμ, mul_one]
    rfl
  have hh : G.h = 0 := by simp [hGdef, Axis.dh]
  rw [← hG] at hds hms
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
      (G.lat G.μ).ModeOK k ε (V j) ∧ (k ≠ 0 → (G.lat G.μ).Pph k (V j) ≠ 0) := by
    fin_cases j
    · exact ⟨1, hd.ε_eq, hds.e_u₁, hds.mode₁, fun hk ↦ (hPs hk).1⟩
    · exact ⟨1, hd.ε_eq, hds.e_u₂, hds.mode₂, fun hk ↦ (hPs hk).2.1⟩
    · exact ⟨-1, hm.ε_eq, hms.e_u₁, hms.mode₁, fun hk ↦ (hPs hk).2.2.1⟩
    · exact ⟨-1, hm.ε_eq, hms.e_u₂, hms.mode₂, fun hk ↦ (hPs hk).2.2.2⟩
  obtain ⟨ε, hε, he, hmo, hPk⟩ := key
  have hP : (G.lat G.μ).Pph k (V j) ≠ 0 := by
    by_cases hk : k = 0
    · subst hk
      rw [Params.Pph_zero]
      exact pow_ne_zero 2 (sub_ne_zero.mpr fun h ↦ hj ⟨rfl, h⟩)
    · exact hPk hk
  have hq := G.sq_add_q₀L_ne_of_sep k root sep j
  exact ⟨G.KL_opp_ne_h0 k hh hε ha he hq (hmo.Kph_ne _ _ _ hP) t, hq⟩

end FParams

end HubbardAtom
