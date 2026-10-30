import HubbardAtomLean.FieldLZero
import HubbardAtomLean.FieldLehmannKMS

/-!
# The longitudinal vertex at `μ = 0`

The particle–hole (Shiba) map `c↓ ↦ c↓†` sends the atom `(U, β, μ, h)` to `(-U, β, h, μ)`
(`FParams.shiba`). On the longitudinal sector it reflects the frequencies of the spin `↓`,
`R↓ν = -ν - ω` (`n ↦ n̄ = -n - k - 1`), with the signs `ε↑ = 1`, `ε↓ = -1`:

* `FParams.chiL_shiba`: `χ_L,στ(n, m) = ε_σε_τ χ_L,στ^shiba(R_σn, R_τm)`, for all `μ`, `h`, from
  the Hamiltonian (`chiLehmann_conj`, and `chiLehmann_dag_left`, `chiLehmann_dag_right`);
* `FParams.bL_shiba`: `b_σ(n) = b_σ^shiba(R_σn)`.

At `μ = 0` the image has `h = 0`, and `FieldLZero.lean` applies to it with the zero-field source
`D = (-U, β, h)`. For `h ≠ 0`, where `1 + 𝒦_d^D` and `1 + 𝒦_m^D` are invertible:

* `FParams.invLmu0`: `Y_στ(n, m) = ε_σε_τ Y^D_στ(R_σn, R_τm)` is a two-sided inverse of `χ_L`
  (`hasSum_chiL_mul_invLmu0`, `hasSum_invLmu0_mul_chiL`; one spin of the sum at a time,
  `hasSum_chiL_mul_invLh0_fiber`) and the only right inverse with columns in the class
  `∑_l |v_τl/z_τl|² < ∞` (`eq_invLmu0_of_hasSum`);
* `FParams.gammaL_mu0`: **Eq. (field-mu0)**,
  `β²(Y - b⁻¹)_στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)`, entrywise, through
  `Ring.inverse` like `gammaL_h0`.
-/

open Complex Matrix

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### Reflections and signs -/

/-- The reflection `R↑n = n`, `R↓n = n̄ = -n - k - 1` (`ν_n̄ = -νₙ - ω_k`). -/
def RL (k : ℤ) (σ : Bool) (n : ℤ) : ℤ := if σ then n else -n - k - 1

/-- `ε↑ = 1`, `ε↓ = -1`. -/
def εL (σ : Bool) : ℂ := if σ then 1 else -1

theorem RL_RL (σ : Bool) (n : ℤ) : RL k σ (RL k σ n) = n := by
  cases σ
  · simp only [RL, Bool.false_eq_true, ↓reduceIte]
    ring
  · simp [RL]

theorem εL_mul_self (σ : Bool) : εL σ * εL σ = 1 := by
  cases σ <;> simp [εL]

/-- `ν_n̄ = -νₙ - ω_k`. -/
theorem ν_bar (n : ℤ) : F.ν (-n - k - 1) = -F.ν n - F.ω k := by
  simp only [ν, ω, Params.ν, Params.ω]
  push_cast
  field_simp
  ring

/-- The fermionic phase: `e^{iνⱼβ} = -1`. -/
theorem cexp_ν_mul_I (j : ℤ) : cexp ((F.ν j : ℂ) * I * F.β) = -1 := by
  have h := (F.lat 0).cexp_x_sub_mu j
  rwa [show (F.lat 0).x j - ((F.lat 0).μ : ℂ) = (F.ν j : ℂ) * I by
    simp [Params.x], lat_β] at h

theorem cexp_neg_ν_mul_I (j : ℤ) : cexp (-(F.ν j : ℂ) * I * F.β) = -1 := by
  rw [show -(F.ν j : ℂ) * I * F.β = -((F.ν j : ℂ) * I * F.β) by ring, Complex.exp_neg,
    F.cexp_ν_mul_I]
  norm_num

/-! ### The Shiba map on the longitudinal sector -/

theorem shibaV_cs (σ : Bool) : shibaV * cs σ * shibaV = if σ then cUp else cDnᴴ := by
  cases σ
  · exact shibaV_cDn
  · exact shibaV_cUp

theorem shibaV_cs_dag (σ : Bool) : (shibaV * cs σ * shibaV)ᴴ = shibaV * (cs σ)ᴴ * shibaV := by
  cases σ
  · rw [cs, ite_eq_right Bool.false_ne_true, shibaV_cDn, shibaV_cDnDag,
      conjTranspose_conjTranspose]
  · rw [cs, ite_eq_left rfl, shibaV_cUp, shibaV_cUpDag]

/-- **The Shiba map on `χ_L`**, from the Hamiltonian, for all `μ`, `h`:
`χ_L,στ(n, m) = ε_σε_τ χ_L,στ^shiba(R_σn, R_τm)`. -/
theorem chiL_shiba (σ t : Bool) (n m : ℤ) :
    F.chiL σ t k n m = εL σ * εL t * F.shiba.chiL σ t k (RL k σ n) (RL k t m) := by
  have h := chiLehmann_conj (A := F) (B := F.shiba) (V := shibaV) (W := shibaV)
    shibaV_mul_self rfl F.shiba_H (shibaV_cs_dag σ) (shibaV_cs_dag t)
    (F.ν n) (F.ν m) (F.ω k)
  rw [shibaV_cs σ, shibaV_cs t] at h
  have hν : ∀ j, F.shiba.ν j = F.ν j := fun _ ↦ rfl
  have hω : F.shiba.ω k = F.ω k := rfl
  rw [chiL, ← h]
  have hf : ∀ j, cexp (-(F.ν j : ℂ) * I * F.shiba.β) = -1 := fun j ↦ F.cexp_neg_ν_mul_I j
  have hf' : ∀ j, cexp ((F.ν j : ℂ) * I * F.shiba.β) = -1 := fun j ↦ F.cexp_ν_mul_I j
  have hadd : ∀ j, ((F.ν j : ℂ) + F.ω k) = F.ν (j + k) := fun j ↦ by
    have := F.ν_add_ω j k
    exact_mod_cast this
  cases σ <;> cases t <;> simp only [↓reduceIte, Bool.false_eq_true, εL, RL, chiL, cs, hν, hω]
  · -- `↓↓`: both pairs reflected
    rw [F.shiba.chiLehmann_dag_left, F.shiba.chiLehmann_dag_right, F.ν_bar, F.ν_bar]
    · ring
    · rw [← F.ν_bar]
      exact hf _
    · rw [show ((-F.ν n - F.ω k : ℝ) : ℂ) + F.ω k = -(F.ν n : ℂ) by push_cast; ring]
      simpa using hf n
    · rw [hadd]
      exact hf _
  · -- `↓↑`
    rw [F.shiba.chiLehmann_dag_left, F.ν_bar]
    ring
  · -- `↑↓`
    rw [F.shiba.chiLehmann_dag_right, F.ν_bar]
    · ring
    · exact hf n
    · rw [hadd]
      exact hf' _
    · rw [hadd]
      exact hf _
  · ring

/-- **The Shiba map on the bubble**: `b_σ(n) = b_σ^shiba(R_σn)`: on the lattice of `↓` the image
has `z ↦ -z`, and `a ↦ -a`, `r_↑ ↦ -r_↑`, `r_↓ ↦ r_↓`, with `P` even. -/
theorem bL_shiba (σ : Bool) (n : ℤ) : F.bL σ k n = F.shiba.bL σ k (RL k σ n) := by
  have ha : (F.shiba.a : ℂ) = -F.a := by
    simp only [a, shiba_U]
    push_cast
    ring
  have hw : F.shiba.wL k = F.wL k := rfl
  have hβ : (F.shiba.β : ℂ) = F.β := rfl
  rw [F.bL_eq, F.shiba.bL_eq, hβ]
  cases σ
  · have hz : (F.shiba.BL false).z k (RL k false n) = -(F.BL false).z k n := by
      simp only [RL, Bool.false_eq_true, ↓reduceIte, Params.z_eq, cL, lat_μ, lat_β, shiba_μ,
        shiba_h, shiba_β, Bool.cond_false]
      push_cast
      field_simp
      ring
    have hr : (F.shiba.rL false : ℂ) = F.rL false := by
      simp only [rL, mL, pL, ρ, shiba_p₀, shiba_p₂, shiba_pUp, shiba_pDn, Bool.not_false,
        Bool.cond_false, Bool.cond_true]
      push_cast
      ring
    rw [hz, hr]
    simp only [PL, ha, hw]
    ring
  · have hz : (F.shiba.BL true).z k (RL k true n) = (F.BL true).z k n := by
      simp only [RL, ↓reduceIte, Params.z_eq, cL, lat_μ, lat_β, shiba_μ, shiba_h, shiba_β,
        Bool.cond_true]
      push_cast
      ring
    have hr : (F.shiba.rL true : ℂ) = -F.rL true := by
      simp only [rL, mL, pL, ρ, shiba_p₀, shiba_p₂, shiba_pUp, shiba_pDn, Bool.not_true,
        Bool.cond_false, Bool.cond_true]
      push_cast
      ring
    rw [hz, hr]
    simp only [PL, ha, hw]
    ring

/-! ### The inverse at `h = 0`, one spin of the sum at a time -/

variable {F k}

/-- One spin `τ` of the product `χ_L Y` at `h = 0`: with `χ_r⁻¹` of the zero-field atom, the four
products `χ_ε χ_ε'⁻¹` are summable (`hasSum_chiR_mul`), and the cross terms cancel only in the sum
over `τ`. -/
theorem hasSum_chiL_mul_invLh0_fiber (hh : F.h = 0) (hμ : F.μ ≠ 0)
    (hd : IsUnit ((F.lat F.μ).onePlusK k 1 hμ)) (hm : IsUnit ((F.lat F.μ).onePlusK k (-1) hμ))
    (σ τ t : Bool) (a b : ℤ) :
    HasSum (fun l ↦ F.chiL σ τ k a l * F.invLh0 k hμ τ l t b)
      ((kron a b + sL τ t * ∑' l, (F.lat F.μ).chiR k 1 a l * (F.lat F.μ).chiInv k (-1) hμ l b +
          sL σ τ * ∑' l, (F.lat F.μ).chiR k (-1) a l * (F.lat F.μ).chiInv k 1 hμ l b +
          sL σ τ * sL τ t * kron a b) / 4) := by
  have hDD := Params.chiInv_isRightInverse hμ hd a b
  have hMM := Params.chiInv_isRightInverse hμ hm a b
  have hDM := ((F.lat F.μ).hasSum_chiR_mul k 1 hμ
    (Params.summable_chiInv (A := F.lat F.μ) (k := k) (ε := -1) hμ b) a).summable.hasSum
  have hMD := ((F.lat F.μ).hasSum_chiR_mul k (-1) hμ
    (Params.summable_chiInv (A := F.lat F.μ) (k := k) (ε := 1) hμ b) a).summable.hasSum
  have h := (((hDD.add (hDM.mul_left (sL τ t))).add (hMD.mul_left (sL σ τ))).add
    (hMM.mul_left (sL σ τ * sL τ t))).div_const 4
  convert h using 1
  funext l
  rw [F.chiL_of_h k hh, invLh0]
  ring

variable (F k)

/-- **The inverse at `μ = 0`**: `Y_στ(n, m) = ε_σε_τ Y^shiba_στ(R_σn, R_τm)`, with `Y^shiba` the
inverse at `h = 0` of the Shiba image `(-U, β, h, 0)` (`invLh0`). -/
noncomputable def invLmu0 (hh : F.h ≠ 0) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) : ℂ :=
  εL σ * εL t * F.shiba.invLh0 k hh σ (RL k σ n) t (RL k t m)

/-- The reflection `R_σ` as an equivalence of `ℤ`. -/
def RLe (σ : Bool) : ℤ ≃ ℤ := ⟨RL k σ, RL k σ, RL_RL k σ, RL_RL k σ⟩

variable {F k}

/-- **`Y` is a right inverse of `χ_L` at `μ = 0`**, where `1 + 𝒦_d` and `1 + 𝒦_m` of the
zero-field source `(-U, β, h)` are invertible: `∑_{l, τ} χ_L,στ(n, l)Y_τt(l, m) = δ_σtδ_nm`. -/
theorem hasSum_chiL_mul_invLmu0 (hμ0 : F.μ = 0) (hh : F.h ≠ 0)
    (hd : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k 1 hh))
    (hm : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k (-1) hh)) (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * F.invLmu0 k hh τ l t m)
      (if σ = t then kron n m else 0) := by
  have hh' : F.shiba.h = 0 := hμ0
  set a := RL k σ n
  set b := RL k t m
  -- each spin `τ`, reflected back by `R_τ`
  have hf : ∀ τ, HasSum (fun l ↦ F.chiL σ τ k n l * F.invLmu0 k hh τ l t m)
      (εL σ * εL t * ((kron a b + sL τ t * ∑' l, (F.shiba.lat F.shiba.μ).chiR k 1 a l *
          (F.shiba.lat F.shiba.μ).chiInv k (-1) hh l b +
        sL σ τ * ∑' l, (F.shiba.lat F.shiba.μ).chiR k (-1) a l *
          (F.shiba.lat F.shiba.μ).chiInv k 1 hh l b + sL σ τ * sL τ t * kron a b) / 4)) := by
    intro τ
    have h := (RLe k τ).hasSum_iff.mpr
      (hasSum_chiL_mul_invLh0_fiber hh' hh hd hm σ τ t a b)
    convert h.mul_left (εL σ * εL t) using 1
    funext l
    simp only [Function.comp_apply, invLmu0, F.chiL_shiba k σ τ n l, RLe, Equiv.coe_fn_mk]
    linear_combination (εL σ * εL t * F.shiba.chiL σ τ k a (RL k τ l) *
      F.shiba.invLh0 k hh τ (RL k τ l) t b) * εL_mul_self τ
  have h := (hf true).add (hf false)
  convert h using 1
  · funext l
    simp
  · have hk : kron (RL k false n) (RL k false m) = kron n m := by
      by_cases hnm : n = m
      · subst hnm
        rw [kron_self, kron_self]
      · rw [kron_of_ne hnm, kron_of_ne fun h ↦ hnm (by
          have := congrArg (RL k false) h
          rwa [RL_RL, RL_RL] at this)]
    cases σ <;> cases t <;> simp only [a, b, hk] <;>
      simp only [RL, sL, εL, Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte] <;> ring

/-- One spin `τ` of the product `Y χ_L` at `h = 0`, as `hasSum_chiL_mul_invLh0_fiber`; the cross
terms are summable as `χ_r` and `χ_r⁻¹` are symmetric. -/
theorem hasSum_invLh0_mul_chiL_fiber (hh : F.h = 0) (hμ : F.μ ≠ 0)
    (hd : IsUnit ((F.lat F.μ).onePlusK k 1 hμ)) (hm : IsUnit ((F.lat F.μ).onePlusK k (-1) hμ))
    (σ τ t : Bool) (a b : ℤ) :
    HasSum (fun l ↦ F.invLh0 k hμ σ a τ l * F.chiL τ t k l b)
      ((kron a b + sL τ t * ∑' l, (F.lat F.μ).chiR k (-1) b l * (F.lat F.μ).chiInv k 1 hμ l a +
          sL σ τ * ∑' l, (F.lat F.μ).chiR k 1 b l * (F.lat F.μ).chiInv k (-1) hμ l a +
          sL σ τ * sL τ t * kron a b) / 4) := by
  have hDD := Params.chiInv_isLeftInverse hμ hd a b
  have hMM := Params.chiInv_isLeftInverse hμ hm a b
  have hMD := ((F.lat F.μ).hasSum_chiR_mul k (-1) hμ
    (Params.summable_chiInv (A := F.lat F.μ) (k := k) (ε := 1) hμ a) b).summable.hasSum
  have hDM := ((F.lat F.μ).hasSum_chiR_mul k 1 hμ
    (Params.summable_chiInv (A := F.lat F.μ) (k := k) (ε := -1) hμ a) b).summable.hasSum
  have h := (((hDD.add (hMD.mul_left (sL τ t))).add (hDM.mul_left (sL σ τ))).add
    (hMM.mul_left (sL σ τ * sL τ t))).div_const 4
  convert h using 1
  funext l
  rw [F.chiL_of_h k hh, invLh0, (F.lat F.μ).chiR_symm k 1 hμ l b,
    (F.lat F.μ).chiR_symm k (-1) hμ l b, Params.chiInv_symm hμ hd a l,
    Params.chiInv_symm hμ hm a l]
  ring

/-- **`Y` is a left inverse of `χ_L` at `μ = 0`**, where `1 + 𝒦_d` and `1 + 𝒦_m` of the
zero-field source `(-U, β, h)` are invertible: `∑_{l, τ} Y_στ(n, l)χ_L,τt(l, m) = δ_σtδ_nm`. -/
theorem hasSum_invLmu0_mul_chiL (hμ0 : F.μ = 0) (hh : F.h ≠ 0)
    (hd : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k 1 hh))
    (hm : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k (-1) hh)) (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.invLmu0 k hh σ n τ l * F.chiL τ t k l m)
      (if σ = t then kron n m else 0) := by
  have hh' : F.shiba.h = 0 := hμ0
  set a := RL k σ n
  set b := RL k t m
  have hf : ∀ τ, HasSum (fun l ↦ F.invLmu0 k hh σ n τ l * F.chiL τ t k l m)
      (εL σ * εL t * ((kron a b + sL τ t * ∑' l, (F.shiba.lat F.shiba.μ).chiR k (-1) b l *
          (F.shiba.lat F.shiba.μ).chiInv k 1 hh l a +
        sL σ τ * ∑' l, (F.shiba.lat F.shiba.μ).chiR k 1 b l *
          (F.shiba.lat F.shiba.μ).chiInv k (-1) hh l a + sL σ τ * sL τ t * kron a b) / 4)) := by
    intro τ
    have h := (RLe k τ).hasSum_iff.mpr
      (hasSum_invLh0_mul_chiL_fiber hh' hh hd hm σ τ t a b)
    convert h.mul_left (εL σ * εL t) using 1
    funext l
    simp only [Function.comp_apply, invLmu0, F.chiL_shiba k τ t l m, RLe, Equiv.coe_fn_mk]
    linear_combination (εL σ * εL t * F.shiba.invLh0 k hh σ a τ (RL k τ l) *
      F.shiba.chiL τ t k (RL k τ l) b) * εL_mul_self τ
  have h := (hf true).add (hf false)
  convert h using 1
  · funext l
    simp
  · have hk : kron (RL k false n) (RL k false m) = kron n m := by
      by_cases hnm : n = m
      · subst hnm
        rw [kron_self, kron_self]
      · rw [kron_of_ne hnm, kron_of_ne fun h ↦ hnm (by
          have := congrArg (RL k false) h
          rwa [RL_RL, RL_RL] at this)]
    cases σ <;> cases t <;> simp only [a, b, hk] <;>
      simp only [RL, sL, εL, Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte] <;> ring

/-- At `μ = 0` the lattice of `τ` is that of the source at `R_τ`, up to sign. -/
theorem norm_z_shiba (hμ0 : F.μ = 0) (τ : Bool) (l : ℤ) :
    ‖(F.shiba.lat F.shiba.μ).z k (RL k τ l)‖ = ‖(F.BL τ).z k l‖ := by
  cases τ
  · rw [show (F.shiba.lat F.shiba.μ).z k (RL k false l) = -(F.BL false).z k l by
      simp only [RL, Bool.false_eq_true, ↓reduceIte, Params.z_eq, cL, lat_μ, lat_β, shiba_μ,
        shiba_β, Bool.cond_false, hμ0]
      push_cast
      field_simp
      ring, norm_neg]
  · congr 1
    simp only [RL, ↓reduceIte, Params.z_eq, cL, lat_μ, lat_β, shiba_μ, shiba_β, Bool.cond_true,
      hμ0, zero_add]

theorem kron_RL (τ : Bool) (a m : ℤ) : kron (RL k τ a) m = kron a (RL k τ m) := by
  by_cases h : RL k τ a = m
  · rw [← h, RL_RL, kron_self, kron_self]
  · rw [kron_of_ne h, kron_of_ne fun h' ↦ h (by rw [h', RL_RL])]

/-- **`Y` is the only right inverse in the class** at `μ = 0`, where `1 + 𝒦_d` and `1 + 𝒦_m` of
the source are invertible: a column `(t, m)` with `∑_l |Y_τl/z_τl|² < ∞` on the lattice of each
spin and `∑_{l, τ} χ_L,στ(n, l)Y_τl = δ_σtδ_nm` is the column of `Y`. -/
theorem eq_invLmu0_of_hasSum (hμ0 : F.μ = 0) (hh : F.h ≠ 0)
    (hd : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k 1 hh))
    (hm : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k (-1) hh)) {t : Bool} {m : ℤ}
    {Y : Bool → ℤ → ℂ} (hY : ∀ τ, Summable fun l ↦ ‖Y τ l / (F.BL τ).z k l‖ ^ 2)
    (hχ : ∀ σ n, HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * Y τ l)
      (if σ = t then kron n m else 0)) (σ : Bool) (n : ℤ) :
    Y σ n = F.invLmu0 k hh σ n t m := by
  have hh' : F.shiba.h = 0 := hμ0
  have hε : ∀ τ, ‖εL τ‖ = 1 := fun τ ↦ by cases τ <;> simp [εL]
  -- the column, reflected to the source
  set W : Bool → ℤ → ℂ := fun τ l ↦ εL τ * εL t * Y τ (RL k τ l)
  have hW : ∀ τ, Summable fun l ↦ ‖W τ l / (F.shiba.lat F.shiba.μ).z k l‖ ^ 2 := fun τ ↦ by
    refine ((RLe k τ).summable_iff.mpr (hY τ)).congr fun l ↦ ?_
    simp only [Function.comp_apply, RLe, Equiv.coe_fn_mk, W, norm_div, norm_mul, hε, one_mul]
    rw [← norm_z_shiba hμ0 τ (RL k τ l), RL_RL]
  have hs : ∀ σ' τ a, Summable fun l ↦ F.shiba.chiL σ' τ k a l * W τ l := fun σ' τ a ↦ by
    have h1 := (F.shiba.lat F.shiba.μ).hasSum_chiR_mul k 1 hh (hW τ) a
    have h2 := (F.shiba.lat F.shiba.μ).hasSum_chiR_mul k (-1) hh (hW τ) a
    refine ((h1.add (h2.mul_left (sL σ' τ))).div_const 2).summable.congr fun l ↦ ?_
    rw [F.shiba.chiL_of_h k hh']
    ring
  have hG : ∀ σ' a, HasSum (fun l ↦ ∑ τ, F.shiba.chiL σ' τ k a l * W τ l)
      (if σ' = t then kron a (RL k t m) else 0) := by
    intro σ' a
    have hsum := hasSum_sum (s := Finset.univ) fun τ _ ↦ (hs σ' τ a).hasSum
    have hF : HasSum (fun l ↦ ∑ τ, F.chiL σ' τ k (RL k σ' a) l * Y τ l)
        (εL σ' * εL t * ∑ τ, ∑' l, F.shiba.chiL σ' τ k a l * W τ l) := by
      have h := (hasSum_sum (s := Finset.univ) fun τ _ ↦
        (RLe k τ).hasSum_iff.mpr (hs σ' τ a).hasSum).mul_left (εL σ' * εL t)
      convert h using 1
      funext l
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun τ _ ↦ ?_
      simp only [Function.comp_apply, RLe, Equiv.coe_fn_mk, W, F.chiL_shiba k σ' τ, RL_RL]
      linear_combination (-(εL σ' * εL τ * F.shiba.chiL σ' τ k a (RL k τ l) * Y τ l)) *
        εL_mul_self t
    have hv := (hχ σ' (RL k σ' a)).unique hF
    convert hsum using 1
    set V := ∑ τ, ∑' l, F.shiba.chiL σ' τ k a l * W τ l
    by_cases hσ : σ' = t
    · subst hσ
      rw [ite_eq_left rfl, kron_RL] at hv
      rw [ite_eq_left rfl]
      linear_combination hv + V * εL_mul_self σ'
    · rw [ite_eq_right hσ] at hv
      rw [ite_eq_right hσ]
      linear_combination (εL σ' * εL t) * hv + (V * εL t * εL t) * εL_mul_self σ' +
        V * εL_mul_self t
  have e := eq_invLh0_of_hasSum (F := F.shiba) (k := k) hh' hh hd hm hW hG σ (RL k σ n)
  simp only [W, RL_RL] at e
  rw [invLmu0, ← e]
  linear_combination (-(Y σ n) * εL t * εL t) * εL_mul_self σ + (-(Y σ n)) * εL_mul_self t

/-- **The longitudinal vertex at `μ = 0`, Eq. (field-mu0)**: with the inverse `Y` of `χ_L`,
`β²(Y - b⁻¹)_στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)` with `Γ_r^D` of the zero-field
source `D = (-U, β, h)` (`GammaOperator.lean`). Both sides through `Ring.inverse`; `Y` is the
inverse of `χ_L` where `1 + 𝒦_d^D` and `1 + 𝒦_m^D` are invertible (`hasSum_chiL_mul_invLmu0`). -/
theorem gammaL_mu0 (hμ0 : F.μ = 0) (hh : F.h ≠ 0) (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    (F.β : ℂ) ^ 2 * (F.invLmu0 k hh σ n t m - if σ = t then kron n m / F.bL σ k n else 0) =
      εL σ * εL t / 2 * ((F.shiba.lat F.shiba.μ).gammaOp k 1 hh (RL k σ n) (RL k t m) +
        εL σ * εL t * (F.shiba.lat F.shiba.μ).gammaOp k (-1) hh (RL k σ n) (RL k t m)) := by
  have hh' : F.shiba.h = 0 := hμ0
  have h0 := gammaL_h0 (F := F.shiba) (k := k) hh' hh σ (RL k σ n) t (RL k t m)
  have hβ : (F.shiba.β : ℂ) = F.β := rfl
  rw [hβ] at h0
  rw [invLmu0, F.bL_shiba k σ n]
  by_cases hσ : σ = t
  · subst hσ
    have hk : kron (RL k σ n) (RL k σ m) = kron n m := by
      by_cases hnm : n = m
      · subst hnm
        rw [kron_self, kron_self]
      · rw [kron_of_ne hnm, kron_of_ne fun h ↦ hnm (by
          have := congrArg (RL k σ) h
          rwa [RL_RL, RL_RL] at this)]
    simp only [ite_true, ← hk] at h0 ⊢
    have hs : sL σ σ = εL σ * εL σ := by
      rw [εL_mul_self]
      simp [sL]
    rw [hs] at h0
    linear_combination (εL σ * εL σ) * h0 + (F.β : ℂ) ^ 2 * kron (RL k σ n) (RL k σ m) /
      F.shiba.bL σ k (RL k σ n) * (εL_mul_self σ)
  · simp only [hσ, ite_false, sub_zero] at h0 ⊢
    have hs : sL σ t = εL σ * εL t := by
      cases σ <;> cases t <;> simp_all [sL, εL]
    rw [hs] at h0
    linear_combination (εL σ * εL t) * h0

end FParams

end HubbardAtom
