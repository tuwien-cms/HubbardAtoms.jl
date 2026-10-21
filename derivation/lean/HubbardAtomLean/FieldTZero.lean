import HubbardAtomLean.FieldTHalfFilling
import HubbardAtomLean.HalfFilling

/-!
# The transverse sector at a vanishing transfer

At `h = 0` and `k = 0` the complex transfer `W = iω_k - 2h` vanishes, and the transverse sector is
the static magnetic channel: from the Hamiltonian, `χ_T↑ = χ_m` of the zero-field atom
`(U, β, μ)` (`chiTUp_eq_chiR`), and so are the bubbles. Theorem "density and magnetic"
(`DensityMagnetic.lean`, `DMLifts.lean`, with the atomic root dropped) and Corollary
"half filling" (`HalfFilling.lean`) then give the transverse vertex there, as Eq. (field-T) of the
note states ("at h = 0, k = 0 omit the atomic root"); the Shiba map gives the opposite-spin pair at
`μ = 0`, `k = 0`.

* `FParams.chiTUp_eq_chiR`, `chiTUp_eq_chiR_hf`, `bTUp_eq_Bph`: the identification, for `μ ≠ 0`
  and for `μ = 0`.
* `FParams.gammaTUp_static`, `gammaAUp_static`: the vertices under `DMGeneric` of the zero-field
  atom; `gammaTUp_static_hf`, `gammaAUp_static_hf`: under `HFHyp`, at half filling.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams)

/-! ### The weights at `h = 0` -/

theorem lat_Z (hh : F.h = 0) : (F.lat F.μ).Z = F.Z := by
  simp only [Params.Z, Z, Params.boltz₀, Params.boltz₁, Params.boltz₂, boltz₀, boltzUp, boltzDn,
    boltz₂, lat_U, lat_β, lat_μ, hh]
  ring_nf

theorem lat_p₀ (hh : F.h = 0) : (F.lat F.μ).p₀ = F.p₀ := by
  rw [Params.p₀, p₀, F.lat_Z hh]
  rfl

theorem lat_p₂ (hh : F.h = 0) : (F.lat F.μ).p₂ = F.p₂ := by
  rw [Params.p₂, p₂, F.lat_Z hh]
  rfl

theorem lat_p₁_up (hh : F.h = 0) : (F.lat F.μ).p₁ = F.pUp := by
  rw [Params.p₁, pUp, F.lat_Z hh, Params.boltz₁, boltzUp, hh]
  simp only [lat_U, lat_β]
  ring_nf

theorem lat_p₁_dn (hh : F.h = 0) : (F.lat F.μ).p₁ = F.pDn := by
  rw [Params.p₁, pDn, F.lat_Z hh, Params.boltz₁, boltzDn, hh]
  simp only [lat_U, lat_β]
  ring_nf

theorem lat_ρ (hh : F.h = 0) : (F.lat F.μ).ρ = F.ρ := by
  rw [Params.ρ, ρ, F.lat_p₀ hh, F.lat_p₂ hh]

theorem mf_of_h (hh : F.h = 0) : F.mf = 0 := by
  rw [mf, ← F.lat_p₁_up hh, ← F.lat_p₁_dn hh, sub_self]

/-! ### The identification -/

/-- From the Hamiltonian, at `h = 0`, `k = 0`, off the pair resonance: `χ_T↑ = χ_m` of the
zero-field atom. -/
theorem chiTUp_eq_chiR' (hh : F.h = 0) {n m : ℤ} (hD : ¬(F.μ = 0 ∧ n + m + 0 + 1 = 0)) :
    F.chiTUp 0 n m = (F.lat F.μ).chiR 0 (-1) n m := by
  set B := F.lat F.μ with hB
  have hP := B.Pph_z_ne_zero 0 n
  have hzz : B.z 0 n + B.z 0 m ≠ 0 := by
    intro h0
    apply hD
    apply (B.pair_eq_zero_iff 0 n m).mp
    rw [← h0, Params.z, Params.z]
    ring
  apply mul_left_cancel₀ hP
  rw [B.Pph_mul_chiR' 0 (-1) hzz, F.chiTUp_eq_pathsTW 0 n m]
  have hs : (F.p₀ : ℂ) + 2 * F.pUp + F.p₂ = 1 := by
    have := F.weights_sum'
    rw [← F.lat_p₁_dn hh, ← F.lat_p₁_up hh] at this
    rw [← F.lat_p₁_up hh]
    linear_combination this
  have hpp : (F.pDn : ℂ) = F.pUp := by rw [← F.lat_p₁_dn hh, ← F.lat_p₁_up hh]
  have hX : ∀ j, B.z 0 j = B.x j := fun j ↦ by rw [Params.z, B.w_zero, zero_div, add_zero]
  have hE : F.h = 0 ∧ (0 : ℤ) = 0 := ⟨hh, rfl⟩
  rw [pathsTW_congr _ _ _ _ _ _ _ _ _ _ Iff.rfl (iff_false_intro hD) (iff_true_intro hE), hpp,
    hh, B.w_zero, hX, hX]
  push_cast
  have hBa : B.a = F.a := rfl
  have hw0 : B.w 0 = 0 := B.w_zero
  have hPX : B.Pph 0 (B.x n) = PTf F.a 0 (B.x n) 0 := by
    rw [Params.Pph, Params.Qp, Params.Qm, hw0, hBa, PTf]
    ring
  rw [hPX, pathsTW_W0 hs (B.x_add_ofReal_ne_zero n F.a) (B.x_sub_ofReal_ne_zero n F.a)
    (B.x_add_ofReal_ne_zero m F.a) (B.x_sub_ofReal_ne_zero m F.a)
    (by rwa [hX, hX] at hzz) (n = m)
    ((B.x_sub_x_eq_zero_iff n m).symm.trans sub_eq_zero)]
  simp only [rhsMag, Params.Hph, Params.lam, Params.Nph, Params.N₀, Params.N₂, Params.ζ,
    Params.q₀, Params.Pph, Params.Qp, Params.Qm, hw0, kron_self, hBa,
    show B.ρ = F.ρ from F.lat_ρ hh, show B.p₀ = F.p₀ from F.lat_p₀ hh,
    show B.p₂ = F.p₂ from F.lat_p₂ hh, show B.p₁ = F.pUp from F.lat_p₁_up hh, ρ,
    show B.β = F.β from rfl]
  push_cast
  have n₁ := B.x_add_ofReal_ne_zero n F.a
  have n₂ := B.x_sub_ofReal_ne_zero n F.a
  have m₁ := B.x_add_ofReal_ne_zero m F.a
  have m₂ := B.x_sub_ofReal_ne_zero m F.a
  have hnm' : B.x n + B.x m ≠ 0 := by rwa [hX, hX] at hzz
  have g₁ : B.x m ^ 2 - (F.a : ℂ) ^ 2 ≠ 0 := by rw [sq_sub_sq]; exact mul_ne_zero m₁ m₂
  by_cases hnm : n = m
  · subst hnm
    have hx0 : B.x n ≠ 0 := fun h0 ↦ hnm' (by rw [h0, add_zero])
    simp only [ite_true, kron_self]
    generalize (F.p₂ : ℂ) = p₂ at *
    obtain rfl : p₂ = 1 - F.p₀ - 2 * F.pUp := by linear_combination hs
    field_simp
    ring
  · simp only [hnm, ite_false, kron_of_ne hnm]
    generalize (F.p₂ : ℂ) = p₂ at *
    obtain rfl : p₂ = 1 - F.p₀ - 2 * F.pUp := by linear_combination hs
    field_simp
    ring

/-- **From the Hamiltonian, at `h = 0`, `k = 0`**: `χ_T↑ = χ_m` of the zero-field atom, for
`μ ≠ 0`. -/
theorem chiTUp_eq_chiR (hh : F.h = 0) (hμ : F.μ ≠ 0) (n m : ℤ) :
    F.chiTUp 0 n m = (F.lat F.μ).chiR 0 (-1) n m :=
  F.chiTUp_eq_chiR' hh fun h0 ↦ hμ h0.1

/-- **From the Hamiltonian, at `h = 0`, `k = 0`, `μ = 0`**: `χ_T↑ = χ_m` of the zero-field atom,
also on the mirror, where the pair rate vanishes. -/
theorem chiTUp_eq_chiR_hf (hh : F.h = 0) (hμ : F.μ = 0) (n m : ℤ) :
    F.chiTUp 0 n m = (F.lat F.μ).chiR 0 (-1) n m := by
  by_cases hm : m = mirror 0 n
  · subst hm
    set B := F.lat F.μ with hB
    have hP := B.Pph_z_ne_zero 0 n
    have hfix : n ≠ mirror 0 n := by unfold mirror; omega
    apply mul_left_cancel₀ hP
    rw [B.Pph_mul_chiR_mirror 0 (-1) hμ n, F.chiTUp_eq_pathsTW 0 n (mirror 0 n)]
    have hD : F.μ = 0 ∧ n + mirror 0 n + 0 + 1 = 0 := ⟨hμ, by unfold mirror; ring⟩
    have hE : F.h = 0 ∧ (0 : ℤ) = 0 := ⟨hh, rfl⟩
    have hw0 : B.w 0 = 0 := B.w_zero
    have hBa : B.a = F.a := rfl
    have hxm : B.x (mirror 0 n) = -B.x n := by
      rw [B.x_mirror, Params.b, hw0, show B.μ = F.μ from rfl, hμ]
      simp
    have hX : ∀ j, B.z 0 j = B.x j := fun j ↦ by rw [Params.z, hw0, zero_div, add_zero]
    have hpp : (F.pDn : ℂ) = F.pUp := by rw [← F.lat_p₁_dn hh, ← F.lat_p₁_up hh]
    have hp₂ : (F.p₂ : ℂ) = F.p₀ := by rw [F.p₂_eq_p₀_of_mu hμ]
    have hs : 2 * (F.p₀ : ℂ) + 2 * F.pUp = 1 := by
      have := F.weights_sum'
      rw [hpp, hp₂] at this
      linear_combination this
    rw [pathsTW_congr _ _ _ _ _ _ _ _ _ _ (iff_false_intro hfix) (iff_true_intro hD)
      (iff_true_intro hE), hpp, hp₂, hh, hw0, hxm, kron_of_ne hfix, hX, hX, hxm]
    push_cast
    have hPX : B.Pph 0 (B.x n) = PTf F.a 0 (B.x n) 0 := by
      rw [Params.Pph, Params.Qp, Params.Qm, hw0, hBa, PTf]
      ring
    have n₁ := B.x_add_ofReal_ne_zero n F.a
    have n₂ := B.x_sub_ofReal_ne_zero n F.a
    have hx0 : B.x n ≠ 0 := by simpa using B.x_sub_ofReal_ne_zero n 0
    rw [hPX, pathsTW_W0_mirror hs n₁ n₂ hx0]
    simp only [rhsMagM, Params.Hph, Params.lam, Params.Nph, Params.N₀, Params.N₂, Params.ζ,
      Params.q₀, Params.Pph, Params.Qp, Params.Qm, hw0, kron_self, hBa,
      show B.ρ = F.ρ from F.lat_ρ hh, show B.p₀ = F.p₀ from F.lat_p₀ hh,
      show B.p₂ = F.p₂ from F.lat_p₂ hh, show B.p₁ = F.pUp from F.lat_p₁_up hh, ρ,
      show B.β = F.β from rfl]
    push_cast
    rw [hp₂]
    have g₁ : B.x n ^ 2 - (F.a : ℂ) ^ 2 ≠ 0 := by rw [sq_sub_sq]; exact mul_ne_zero n₁ n₂
    have n₃ : -B.x n + F.a ≠ 0 := fun h0 ↦ n₂ (by linear_combination -h0)
    have n₄ : -B.x n - F.a ≠ 0 := fun h0 ↦ n₁ (by linear_combination -h0)
    field_simp
    ring
  · exact F.chiTUp_eq_chiR' hh fun h0 ↦ hm (by unfold mirror; omega)

/-- The bubbles at `h = 0`, `k = 0`: `b_T↑ = B_m` of the zero-field atom. -/
theorem bTUp_eq_Bph (hh : F.h = 0) (n : ℤ) : F.bTUp 0 n = (F.lat F.μ).Bph 0 n := by
  have hν : F.ν n + F.ω 0 = F.ν n := by simp [ω, Params.ω]
  have hμh : F.μ + F.h = F.μ := by rw [hh, add_zero]
  have hμh' : F.μ - F.h = F.μ := by rw [hh, sub_zero]
  rw [bTUp, hν, F.propagator_up, F.propagator_dn, Params.Bph, (F.lat F.μ).w_zero, add_zero]
  simp only [Gc, Params.G, rUp, rDn, F.mf_of_h hh, sub_zero, add_zero, hμh, hμh',
    show (F.lat F.μ).ρ = F.ρ from F.lat_ρ hh, show (F.lat F.μ).a = F.a from rfl,
    show (F.lat F.μ).β = F.β from rfl]

variable {F} {u₁ u₂ : ℂ}

/-- **The transverse vertex at `h = 0`, `k = 0`**: the static magnetic channel of the zero-field
atom, with its inverse (Theorem "density and magnetic" with the atomic root dropped). -/
theorem gammaTUp_static (hh : F.h = 0) (h : (F.lat F.μ).DMGeneric 0 (-1) u₁ u₂) :
    IsRightInverse (F.chiTUp 0) ((F.lat F.μ).invR 0 (-1) u₁ u₂) ∧
      IsLeftInverse (F.chiTUp 0) ((F.lat F.μ).invR 0 (-1) u₁ u₂) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * ((F.lat F.μ).invR 0 (-1) u₁ u₂ n m - kron n m / F.bTUp 0 n) =
        (F.lat F.μ).gammaR 0 (-1) u₁ u₂ n m := by
  have hχ : F.chiTUp 0 = (F.lat F.μ).chiR 0 (-1) :=
    funext fun n ↦ funext fun m ↦ F.chiTUp_eq_chiR hh h.μ_ne n m
  refine ⟨hχ ▸ h.chiR_isRightInverse, hχ ▸ h.chiR_isLeftInverse, fun n m ↦ ?_⟩
  rw [F.bTUp_eq_Bph hh n]
  exact h.gammaR_eq n m

/-- **The opposite-spin pair at `μ = 0`, `k = 0`**, through the Shiba map: the static magnetic
channel of the zero-field atom `(-U, β, h)`. -/
theorem gammaAUp_static (hμ : F.μ = 0)
    (h : (F.shiba.lat F.shiba.μ).DMGeneric 0 (-1) u₁ u₂) :
    IsRightInverse (F.chiAUp 0) ((F.shiba.lat F.shiba.μ).invR 0 (-1) u₁ u₂) ∧
      IsLeftInverse (F.chiAUp 0) ((F.shiba.lat F.shiba.μ).invR 0 (-1) u₁ u₂) ∧
      ∀ n m, (F.β : ℂ) ^ 2 *
          ((F.shiba.lat F.shiba.μ).invR 0 (-1) u₁ u₂ n m - kron n m / F.bAUp 0 n) =
        (F.shiba.lat F.shiba.μ).gammaR 0 (-1) u₁ u₂ n m := by
  have hχ : F.chiAUp 0 = F.shiba.chiTUp 0 :=
    funext fun n ↦ funext fun m ↦ F.chiAUp_eq 0 n m
  have hb : ∀ n, F.bAUp 0 n = F.shiba.bTUp 0 n := F.bAUp_eq 0
  have := F.shiba.gammaTUp_static (u₁ := u₁) (u₂ := u₂) (by rw [shiba_h, hμ]) h
  rw [hχ]
  simp only [hb]
  exact this

variable {u : ℂ}

/-- **The transverse vertex at `h = 0`, `k = 0`, `μ = 0`**: the static magnetic channel of the
half-filled zero-field atom (Corollary "half filling"). -/
theorem gammaTUp_static_hf (hh : F.h = 0) (h : (F.lat F.μ).HFHyp 0 (-1) u) :
    IsRightInverse (F.chiTUp 0) ((F.lat F.μ).invHF 0 (-1) u) ∧
      IsLeftInverse (F.chiTUp 0) ((F.lat F.μ).invHF 0 (-1) u) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * ((F.lat F.μ).invHF 0 (-1) u n m - kron n m / F.bTUp 0 n) =
        (F.lat F.μ).gammaHF 0 (-1) u n m := by
  have hχ : F.chiTUp 0 = (F.lat F.μ).chiR 0 (-1) :=
    funext fun n ↦ funext fun m ↦ F.chiTUp_eq_chiR_hf hh h.μ_eq n m
  refine ⟨hχ ▸ h.chiR_isRightInverse, hχ ▸ h.chiR_isLeftInverse, fun n m ↦ ?_⟩
  rw [F.bTUp_eq_Bph hh n]
  exact h.gammaHF_eq n m

/-- **The opposite-spin pair at `μ = 0`, `k = 0`, `h = 0`**, through the Shiba map. -/
theorem gammaAUp_static_hf (hμ : F.μ = 0) (h : (F.shiba.lat F.shiba.μ).HFHyp 0 (-1) u) :
    IsRightInverse (F.chiAUp 0) ((F.shiba.lat F.shiba.μ).invHF 0 (-1) u) ∧
      IsLeftInverse (F.chiAUp 0) ((F.shiba.lat F.shiba.μ).invHF 0 (-1) u) ∧
      ∀ n m, (F.β : ℂ) ^ 2 *
          ((F.shiba.lat F.shiba.μ).invHF 0 (-1) u n m - kron n m / F.bAUp 0 n) =
        (F.shiba.lat F.shiba.μ).gammaHF 0 (-1) u n m := by
  have hχ : F.chiAUp 0 = F.shiba.chiTUp 0 :=
    funext fun n ↦ funext fun m ↦ F.chiAUp_eq 0 n m
  have hb : ∀ n, F.bAUp 0 n = F.shiba.bTUp 0 n := F.bAUp_eq 0
  have := F.shiba.gammaTUp_static_hf (u := u) (by rw [shiba_h, hμ]) h
  rw [hχ]
  simp only [hb]
  exact this

end FParams

end HubbardAtom
