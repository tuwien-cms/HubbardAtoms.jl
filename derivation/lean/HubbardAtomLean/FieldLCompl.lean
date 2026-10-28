import HubbardAtomLean.FieldLPF
import HubbardAtomLean.FieldLModeZo

/-!
# The completion of the longitudinal inverse

For four roots `u_j` of `E₈` with distinct `s_j = u_j²`, each coefficient of `zᶜ` of the completed
defect of the column `(t, v)` is the sum over the roots of the images of the modes,
`D_c(v) = ∑ⱼ 𝔠ⱼ M_c(u_j)/(v² - s_j)` (`compl_s{c}_W`, `compl_o{c}_W` for `k ≠ 0`, `_Z` for `k = 0`):
by the partial fractions of `N_c(v²)/(βE₈(v))` (`completion`) and, at each root, the certificate
`β²N_c(s_j)K_t = βa_tM_c` of `FieldLCert*.lean`. At `k = 0` the atomic root `s_j = a²` drops out:
`N_c(a²) = 0`, and its coefficient `𝔠ⱼ` is set to zero.
-/

open Complex

-- the cubic witnesses are `ring` on the generated polynomials: one at a time
set_option Elab.async false

namespace HubbardAtom

/-- The finite-difference coefficients of a cubic from its values at `0, ±1, ±2`. -/
noncomputable def fd3 (N : ℂ → ℂ) (j : Fin 4) : ℂ :=
  match j with
  | 0 => N 0
  | 1 => (N 1 - N (-1)) / 2 - (N 2 - 2 * N 1 + 2 * N (-1) - N (-2)) / 12
  | 2 => (N 1 + N (-1)) / 2 - N 0
  | 3 => (N 2 - 2 * N 1 + 2 * N (-1) - N (-2)) / 12

/-- `∏_{i ≠ j}(s_j - s_i)`, the derivative of `∏ᵢ(x - s_i)` at `s_j`. -/
noncomputable def dprod (s : Fin 4 → ℂ) : Fin 4 → ℂ
  | 0 => (s 0 - s 1) * (s 0 - s 2) * (s 0 - s 3)
  | 1 => (s 1 - s 0) * (s 1 - s 2) * (s 1 - s 3)
  | 2 => (s 2 - s 0) * (s 2 - s 1) * (s 2 - s 3)
  | 3 => (s 3 - s 0) * (s 3 - s 1) * (s 3 - s 2)

/-- **The completion, abstractly**: if `D = N(x)/(β∏ⱼ(x - s_j))` for a cubic `N` and
`N(s_j)/(β∏_{i ≠ j}(s_j - s_i)) = c_j M_j` at each node, then `D = ∑ⱼ c_j M_j/(x - s_j)`. -/
theorem completion {N : ℂ → ℂ} {c₀ c₁ c₂ c₃ : ℂ}
    (hN : ∀ y, N y = c₀ + c₁ * y + c₂ * y ^ 2 + c₃ * y ^ 3)
    {s : Fin 4 → ℂ} (hs : ∀ i j, i ≠ j → s i - s j ≠ 0) {β x D : ℂ} (hx : ∀ j, x - s j ≠ 0)
    (hD : D = N x / (β * ((x - s 0) * (x - s 1) * (x - s 2) * (x - s 3))))
    {c M : Fin 4 → ℂ} (hj : ∀ j, N (s j) / (β * dprod s j) = c j * M j) :
    D = ∑ j, c j * M j / (x - s j) := by
  have e := pf4 hN (hs 0 1 (by decide)) (hs 0 2 (by decide)) (hs 0 3 (by decide))
    (hs 1 2 (by decide)) (hs 1 3 (by decide)) (hs 2 3 (by decide)) (hx 0) (hx 1) (hx 2) (hx 3)
  rw [Fin.sum_univ_four, ← hj 0, ← hj 1, ← hj 2, ← hj 3, hD, mul_comm β, ← div_div, e]
  simp only [dprod, ← div_div]
  ring

theorem dprod_ne {s : Fin 4 → ℂ} (hs : ∀ i j, i ≠ j → s i - s j ≠ 0) : ∀ j, dprod s j ≠ 0
  | 0 => mul_ne_zero (mul_ne_zero (hs 0 1 (by decide)) (hs 0 2 (by decide))) (hs 0 3 (by decide))
  | 1 => mul_ne_zero (mul_ne_zero (hs 1 0 (by decide)) (hs 1 2 (by decide))) (hs 1 3 (by decide))
  | 2 => mul_ne_zero (mul_ne_zero (hs 2 0 (by decide)) (hs 2 1 (by decide))) (hs 2 3 (by decide))
  | 3 => mul_ne_zero (mul_ne_zero (hs 3 0 (by decide)) (hs 3 1 (by decide))) (hs 3 2 (by decide))

namespace FParams

variable (F : FParams) (k : ℤ)

/-- `E₈(v) = ∏ⱼ(v² - u_j²)` for four roots with distinct squares. -/
theorem E8L_eq_prod {u : Fin 4 → ℂ} (root : ∀ j, F.E8L k (u j) = 0)
    (sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0) (v : ℂ) :
    F.E8L k v = (v ^ 2 - u 0 ^ 2) * (v ^ 2 - u 1 ^ 2) * (v ^ 2 - u 2 ^ 2) * (v ^ 2 - u 3 ^ 2) := by
  rw [F.E8L_eq_quartic]
  exact quartic_eq_prod
    (Q := fun y ↦ y ^ 4 + F.E8b k 3 * y ^ 3 + F.E8b k 2 * y ^ 2 + F.E8b k 1 * y + F.E8b k 0)
    (fun _ ↦ rfl) (sep 0 1 (by decide)) (sep 0 2 (by decide)) (sep 0 3 (by decide))
    (sep 1 2 (by decide)) (sep 1 3 (by decide)) (sep 2 3 (by decide))
    (by simpa only [F.E8L_eq_quartic] using root 0) (by simpa only [F.E8L_eq_quartic] using root 1)
    (by simpa only [F.E8L_eq_quartic] using root 2) (by simpa only [F.E8L_eq_quartic] using root 3)
    (v ^ 2)

/-- `u` is the atomic root `±a` of `E₈` at `k = 0`. -/
def Atomic (u : ℂ) : Prop := k = 0 ∧ u ^ 2 = F.a ^ 2

/-- `a_t(u) = 2λ²D_t(u² + q₀)`, the coefficient of `S_t(u)` in `K_t(u)`. -/
noncomputable def aAL (t : Bool) (u : ℂ) : ℂ := 2 * F.lamL ^ 2 * F.DL t * (u ^ 2 + F.q₀L k)

open Classical in
/-- The coefficient `𝔠ⱼ = a_t(u_j)/(β²K_t(u_j)E₈'(s_j))` of the mode of the root `u_j` in the
inverse, zero at the atomic root. -/
noncomputable def cMode (t : Bool) (u : Fin 4 → ℂ) (j : Fin 4) : ℂ :=
  if F.Atomic k (u j) then 0
  else F.aAL k t (u j) / (F.β ^ 2 * F.KL k t t (u j) * dprod (fun i ↦ u i ^ 2) j)

/-- The four roots of `E₈`, with distinct squares, and the hypotheses on the non-atomic ones: off
both lattices (through the product relations) and `K_t(u_j) ≠ 0`. -/
structure Roots (t : Bool) (u : Fin 4 → ℂ) : Prop where
  root : ∀ j, F.E8L k (u j) = 0
  sep : ∀ i j, i ≠ j → u i ^ 2 - u j ^ 2 ≠ 0
  rel : ∀ j, ¬F.Atomic k (u j) → F.ModeRel k t (u j)
  K_ne : ∀ j, ¬F.Atomic k (u j) → F.KL k t t (u j) ≠ 0

/-- **The completion of one coefficient**: from `D = N(v²)/(βE₈(v))` with a cubic `N`, the
certificate `β²N(s_j)K_t = βa_tM(u_j)` at the non-atomic roots and `N(a²) = 0` at `k = 0`. -/
theorem compl_of {t : Bool} {u : Fin 4 → ℂ} (hu : F.Roots k t u) {D N Mc : ℂ → ℂ}
    (hN : ∀ y, N y = fd3 N 0 + fd3 N 1 * y + fd3 N 2 * y ^ 2 + fd3 N 3 * y ^ 3)
    (hD : ∀ v, F.E8L k v ≠ 0 → D v = N (v ^ 2) / (F.β * F.E8L k v))
    (hcert : ∀ j, ¬F.Atomic k (u j) →
      F.β ^ 2 * N (u j ^ 2) * F.KL k t t (u j) = F.β * F.aAL k t (u j) * Mc (u j))
    (hatom : k = 0 → N (F.a ^ 2) = 0) {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    D v = ∑ j, F.cMode k t u j * Mc (u j) / (v ^ 2 - u j ^ 2) := by
  have hβ := β_ne_zeroT (F := F)
  have hE := F.E8L_eq_prod k hu.root hu.sep v
  have hE0 : F.E8L k v ≠ 0 := by
    rw [hE]; exact mul_ne_zero (mul_ne_zero (mul_ne_zero (hv 0) (hv 1)) (hv 2)) (hv 3)
  refine completion hN (s := fun j ↦ u j ^ 2) (β := F.β) (c := F.cMode k t u)
    (M := fun j ↦ Mc (u j)) hu.sep hv (by rw [hD v hE0, hE]) fun j ↦ ?_
  have hd := dprod_ne hu.sep j
  by_cases ha : F.Atomic k (u j)
  · simp only [cMode, ha, ↓reduceIte]
    rw [zero_mul, ha.2, hatom ha.1, zero_div]
  · have hK := hu.K_ne j ha
    simp only [cMode, ha, ↓reduceIte]
    rw [div_mul_eq_mul_div,
      div_eq_div_iff (mul_ne_zero hβ hd) (mul_ne_zero (mul_ne_zero (pow_ne_zero 2 hβ) hK) hd)]
    linear_combination dprod (fun i ↦ u i ^ 2) j * hcert j ha

/-! ### The identification of `K_t` and `a_t` -/

/-- `K_t(u)` is the generated polynomial `KtW(u, S_t(u))`. -/
theorem KL_W (t : Bool) (u : ℂ) :
    F.KL k t t u = LCert.KtW u ((F.BL t).S k u) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [KL, fMp, fMm, ↓reduceIte, QL, αL, HL, lamL, piP, F.piS_eq t, rL_not, q₀L, DL, cLw]
  push_cast
  simp only [rL_cast, h₀, h₂, ht, hn, hR, hM, LCert.KtW]
  ring

theorem aAL_W (t : Bool) (u : ℂ) :
    F.aAL k t u = LCert.aAW u F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  obtain ⟨P, R, M, h₀, h₂, ht, hn, hR, hM, hP⟩ := F.col_params t
  simp only [aAL, lamL, q₀L, DL]
  push_cast
  simp only [h₀, h₂, ht, hn, hR, hM, LCert.aAW]
  ring

theorem KL_Z (hk : k = 0) (t : Bool) (u : ℂ) :
    F.KL k t t u = LCert.KtZ u ((F.BL t).S k u) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  rw [F.KL_W, hw]
  unfold LCert.KtW LCert.KtZ
  ring

theorem aAL_Z (hk : k = 0) (t : Bool) (u : ℂ) :
    F.aAL k t u = LCert.aAZ u F.a (F.p₀ + F.p₂) F.ρ (F.mL t) := by
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  rw [F.aAL_W, hw]
  unfold LCert.aAW LCert.aAZ
  ring

/-- At `k = 0` a non-atomic root of `E₈ = (u² - a²)E₆` is a root of `E₆`. -/
theorem E6_root (hk : k = 0) (t : Bool) {u : ℂ} (hE : F.E8L k u = 0) (ha : ¬F.Atomic k u) :
    LCert.E6Z u F.a (F.p₀ + F.p₂) F.ρ (F.mL t) = 0 := by
  have h := (F.E8L_W k t u).symm.trans hE
  have hu : u ^ 2 - F.a ^ 2 ≠ 0 := fun h0 ↦ ha ⟨hk, by linear_combination h0⟩
  subst hk
  have hw : F.wL 0 = 0 := (F.Bup.w_eq_zero_iff 0).mpr rfl
  rw [hw, show ∀ P R M : ℂ, LCert.E8W u F.a 0 P R M = (u ^ 2 - F.a ^ 2) * LCert.E6Z u F.a P R M
    from fun _ _ _ ↦ by unfold LCert.E8W LCert.E6Z; ring] at h
  exact (mul_eq_zero.mp h).resolve_left hu

/-! ### The completion, coefficient by coefficient -/

/-- The coefficient of `z^0` in the row of the column spin, `k ≠ 0`. -/
theorem compl_s0_W (hk : k ≠ 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Ds0 k t v = ∑ j, F.cMode k t u j * F.Ms0 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Ds0 k t) (Mc := F.Ms0 k t)
    (N := fun y ↦ LCert.NcW00 y F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcW00; ring)
    (fun _ hE ↦ F.Ds0_W k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_W00 (u j) ((F.BL t).S k (u j)) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ
          (F.mL t) ((F.E8L_W k t (u j)).symm.trans (hu.root j))
      rwa [← F.KL_W, ← F.aAL_W, ← F.Ms0_W k hk t h (hu.rel j hj)] at this)
    (fun h0 ↦ absurd h0 hk) hv

/-- The coefficient of `z^0` in the row of the column spin, `k = 0`. -/
theorem compl_s0_Z (hk : k = 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Ds0 k t v = ∑ j, F.cMode k t u j * F.Ms0 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Ds0 k t) (Mc := F.Ms0 k t)
    (N := fun y ↦ LCert.NcZ00 y F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcZ00; ring)
    (fun _ hE ↦ F.Ds0_Z k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_Z00 (u j) ((F.BL t).S k (u j)) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
          (F.E6_root k hk t (hu.root j) hj)
      rwa [← F.KL_Z k hk, ← F.aAL_Z k hk, ← F.Ms0_Z k hk t h (hu.rel j hj)] at this)
    (fun _ ↦ by unfold LCert.NcZ00; ring) hv

/-- The coefficient of `z^1` in the row of the column spin, `k ≠ 0`. -/
theorem compl_s1_W (hk : k ≠ 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Ds1 k t v = ∑ j, F.cMode k t u j * F.Ms1 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Ds1 k t) (Mc := F.Ms1 k t)
    (N := fun y ↦ LCert.NcW01 y F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcW01; ring)
    (fun _ hE ↦ F.Ds1_W k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_W01 (u j) ((F.BL t).S k (u j)) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ
          (F.mL t) ((F.E8L_W k t (u j)).symm.trans (hu.root j))
      rwa [← F.KL_W, ← F.aAL_W, ← F.Ms1_W k hk t h (hu.rel j hj)] at this)
    (fun h0 ↦ absurd h0 hk) hv

/-- The coefficient of `z^1` in the row of the column spin, `k = 0`. -/
theorem compl_s1_Z (hk : k = 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Ds1 k t v = ∑ j, F.cMode k t u j * F.Ms1 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Ds1 k t) (Mc := F.Ms1 k t)
    (N := fun y ↦ LCert.NcZ01 y F.a (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcZ01; ring)
    (fun _ hE ↦ F.Ds1_Z k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_Z01 (u j) ((F.BL t).S k (u j)) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
          (F.E6_root k hk t (hu.root j) hj)
      rwa [← F.KL_Z k hk, ← F.aAL_Z k hk, ← F.Ms1_Z k hk t h (hu.rel j hj)] at this)
    (fun _ ↦ by unfold LCert.NcZ01; ring) hv

/-- The coefficient of `z^2` in the row of the column spin, `k ≠ 0`. -/
theorem compl_s2_W (hk : k ≠ 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Ds2 k t v = ∑ j, F.cMode k t u j * F.Ms2 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Ds2 k t) (Mc := F.Ms2 k t)
    (N := fun y ↦ LCert.NcW02 y F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcW02; ring)
    (fun _ hE ↦ F.Ds2_W k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_W02 (u j) ((F.BL t).S k (u j)) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ
          (F.mL t) ((F.E8L_W k t (u j)).symm.trans (hu.root j))
      rwa [← F.KL_W, ← F.aAL_W, ← F.Ms2_W k hk t h (hu.rel j hj)] at this)
    (fun h0 ↦ absurd h0 hk) hv

/-- The coefficient of `z^2` in the row of the column spin, `k = 0`. -/
theorem compl_s2_Z (hk : k = 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Ds2 k t v = ∑ j, F.cMode k t u j * F.Ms2 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Ds2 k t) (Mc := F.Ms2 k t)
    (N := fun y ↦ LCert.NcZ02 y F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcZ02; ring)
    (fun _ hE ↦ F.Ds2_Z k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_Z02 (u j) ((F.BL t).S k (u j)) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
          (F.E6_root k hk t (hu.root j) hj)
      rwa [← F.KL_Z k hk, ← F.aAL_Z k hk, ← F.Ms2_Z k hk t h (hu.rel j hj)] at this)
    (fun _ ↦ by unfold LCert.NcZ02; ring) hv

/-- The coefficient of `z^0` in the row of `-t`, `k ≠ 0`. -/
theorem compl_o0_W (hk : k ≠ 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Do0 k t v = ∑ j, F.cMode k t u j * F.Mo0 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Do0 k t) (Mc := F.Mo0 k t)
    (N := fun y ↦ LCert.NcW10 y F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcW10; ring)
    (fun _ hE ↦ F.Do0_W k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_W10 (u j) ((F.BL t).S k (u j)) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ
          (F.mL t) ((F.E8L_W k t (u j)).symm.trans (hu.root j))
      rwa [← F.KL_W, ← F.aAL_W, ← F.Mo0_W k hk t h (hu.rel j hj)] at this)
    (fun h0 ↦ absurd h0 hk) hv

/-- The coefficient of `z^0` in the row of `-t`, `k = 0`. -/
theorem compl_o0_Z (hk : k = 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Do0 k t v = ∑ j, F.cMode k t u j * F.Mo0 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Do0 k t) (Mc := F.Mo0 k t)
    (N := fun y ↦ LCert.NcZ10 y F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcZ10; ring)
    (fun _ hE ↦ F.Do0_Z k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_Z10 (u j) ((F.BL t).S k (u j)) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
          (F.E6_root k hk t (hu.root j) hj)
      rwa [← F.KL_Z k hk, ← F.aAL_Z k hk, ← F.Mo0_Z k hk t h (hu.rel j hj)] at this)
    (fun _ ↦ by unfold LCert.NcZ10; ring) hv

/-- The coefficient of `z^1` in the row of `-t`, `k ≠ 0`. -/
theorem compl_o1_W (hk : k ≠ 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Do1 k t v = ∑ j, F.cMode k t u j * F.Mo1 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Do1 k t) (Mc := F.Mo1 k t)
    (N := fun y ↦ LCert.NcW11 y F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcW11; ring)
    (fun _ hE ↦ F.Do1_W k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_W11 (u j) ((F.BL t).S k (u j)) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ
          (F.mL t) ((F.E8L_W k t (u j)).symm.trans (hu.root j))
      rwa [← F.KL_W, ← F.aAL_W, ← F.Mo1_W k hk t h (hu.rel j hj)] at this)
    (fun h0 ↦ absurd h0 hk) hv

/-- The coefficient of `z^1` in the row of `-t`, `k = 0`. -/
theorem compl_o1_Z (hk : k = 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Do1 k t v = ∑ j, F.cMode k t u j * F.Mo1 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Do1 k t) (Mc := F.Mo1 k t)
    (N := fun y ↦ LCert.NcZ11 y F.a (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcZ11; ring)
    (fun _ hE ↦ F.Do1_Z k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_Z11 (u j) ((F.BL t).S k (u j)) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
          (F.E6_root k hk t (hu.root j) hj)
      rwa [← F.KL_Z k hk, ← F.aAL_Z k hk, ← F.Mo1_Z k hk t h (hu.rel j hj)] at this)
    (fun _ ↦ by unfold LCert.NcZ11; ring) hv

/-- The coefficient of `z^2` in the row of `-t`, `k ≠ 0`. -/
theorem compl_o2_W (hk : k ≠ 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Do2 k t v = ∑ j, F.cMode k t u j * F.Mo2 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Do2 k t) (Mc := F.Mo2 k t)
    (N := fun y ↦ LCert.NcW12 y F.a (F.wL k) (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcW12; ring)
    (fun _ hE ↦ F.Do2_W k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_W12 (u j) ((F.BL t).S k (u j)) F.a (F.wL k) F.β (F.p₀ + F.p₂) F.ρ
          (F.mL t) ((F.E8L_W k t (u j)).symm.trans (hu.root j))
      rwa [← F.KL_W, ← F.aAL_W, ← F.Mo2_W k hk t h (hu.rel j hj)] at this)
    (fun h0 ↦ absurd h0 hk) hv

/-- The coefficient of `z^2` in the row of `-t`, `k = 0`. -/
theorem compl_o2_Z (hk : k = 0) {t : Bool} (h : F.ColNe k t) {u : Fin 4 → ℂ} (hu : F.Roots k t u)
    {v : ℂ} (hv : ∀ j, v ^ 2 - u j ^ 2 ≠ 0) :
    F.Do2 k t v = ∑ j, F.cMode k t u j * F.Mo2 k t (u j) / (v ^ 2 - u j ^ 2) :=
  F.compl_of k hu (D := F.Do2 k t) (Mc := F.Mo2 k t)
    (N := fun y ↦ LCert.NcZ12 y F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t))
    (fun y ↦ by simp only [fd3]; unfold LCert.NcZ12; ring)
    (fun _ hE ↦ F.Do2_Z k hk t h hE)
    (fun j hj ↦ by
      have := LCert.cert_Z12 (u j) ((F.BL t).S k (u j)) F.a F.β (F.p₀ + F.p₂) F.ρ (F.mL t)
          (F.E6_root k hk t (hu.root j) hj)
      rwa [← F.KL_Z k hk, ← F.aAL_Z k hk, ← F.Mo2_Z k hk t h (hu.rel j hj)] at this)
    (fun _ ↦ by unfold LCert.NcZ12; ring) hv

end FParams

end HubbardAtom
