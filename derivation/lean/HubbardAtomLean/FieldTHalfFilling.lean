import HubbardAtomLean.FieldPair

/-!
# The transverse vertex of the half-filled atom in a field

The transverse vertex at half filling, Eq. (field-T-axis) of the note (App. F.6.2): at `μ = 0`,
`ρ = 0`, `p₂ = p₀`, and the Cauchy matrix of the transverse input becomes the mirror `βp₀𝓜`
(`PT_mul_chiTUp_hf`). The proof is that of Corollary "half filling" (`HalfFilling.lean`) with the
transverse data: the inverse of the diagonal-plus-mirror part on the mirror orbits, a constant and
one separable term in `ĝ(z) = 1/(z² - ŝ)`, `ŝ = λp₀ - q₁`.

* `FParams.THFHyp`: the assumptions: `U ≠ 0`, `μ = 0`, `W ≠ 0`, `q₀ ≠ 0`, `zₙ² ≠ ŝ` for all `n`,
  `zₙ² ≠ ŝ'` off the mirror fixed points, `û² = ŝ`, `P(û) ≠ 0` and `D̂ = β + 2aS_k(û)/û ≠ 0`.
* `FParams.THFHyp.chiTUp_isRightInverse`, `chiTUp_isLeftInverse`, `gammaTHF_eq`.
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-- `ŝ = λp₀ - q₁`: at `μ = 0`, `H(z) - λp₀ = z² - ŝ`. -/
noncomputable def shatT : ℂ := F.lamT * F.p₀ - F.q₁T k

/-- `ŝ' = -λp₀ - q₁`. -/
noncomputable def shatT' : ℂ := -F.lamT * F.p₀ - F.q₁T k

/-- `ĝ(z) = 1/(z² - ŝ)`. -/
noncomputable def ghatT (z : ℂ) : ℂ := 1 / (z ^ 2 - F.shatT k)

/-- `T̂ = S_k(û)/û`, with the value `-β²/4` at `û = 0`. -/
noncomputable def ThatT (u : ℂ) : ℂ :=
  if u = 0 then -(F.β : ℂ) ^ 2 / 4 else (F.lat F.μ).S k u / u

/-- `D̂ = β + 2aT̂`. -/
noncomputable def DhatT (u : ℂ) : ℂ := F.β - 2 * (-F.a) * F.ThatT k u

/-- The inverse of the mirror-even eigenvalue `-β(z² - ŝ)/P(z)`. -/
noncomputable def ievnT (z : ℂ) : ℂ := -(F.PT k z / (F.β * (z ^ 2 - F.shatT k)))

/-- The inverse of the mirror-odd eigenvalue `-β(z² - ŝ')/P(z)`. -/
noncomputable def ioddT (z : ℂ) : ℂ := -(F.PT k z / (F.β * (z ^ 2 - F.shatT' k)))

/-- The inverse of `χ_T↑` at `μ = 0`. -/
noncomputable def invTHF (u : ℂ) (n m : ℤ) : ℂ :=
  mirrorBlock k (fun j ↦ F.ievnT k (F.zT k j)) (fun j ↦ F.ioddT k (F.zT k j)) n m +
      2 * (-F.a) / F.β ^ 2 -
    2 * (-F.a) * F.PT k u / (F.β * F.DhatT k u) * F.ghatT k (F.zT k n) * F.ghatT k (F.zT k m)

/-- The closed form of Eq. (field-T-axis) in orbit form. -/
noncomputable def gammaTHF (u : ℂ) (n m : ℤ) : ℂ :=
  F.β * F.PT k (F.zT k n) / F.G₂T k (F.zT k n) * kron n m +
      F.β ^ 2 * mirrorBlock k (fun j ↦ F.ievnT k (F.zT k j)) (fun j ↦ F.ioddT k (F.zT k j)) n m +
      2 * (-F.a) -
    2 * (-F.a) * F.β * F.PT k u / F.DhatT k u * F.ghatT k (F.zT k n) * F.ghatT k (F.zT k m)

/-- Eq. (field-T-axis) as printed:
`βP(zₙ)[1/G₂(zₙ) - H(zₙ)/e(zₙ)]δₙₘ - U - λβp₀F(zₙ)δ_{m n̄} + 2aβP(û)ĝ(zₙ)ĝ(zₘ)/D̂`. -/
noncomputable def gammaTHFnote (u : ℂ) (n m : ℤ) : ℂ :=
  F.β * F.PT k (F.zT k n) *
        (1 / F.G₂T k (F.zT k n) - F.HT k (F.zT k n) / F.eT k (F.zT k n)) * kron n m +
      2 * (-F.a) - F.lamT * (F.β * F.p₀) * F.FT k (F.zT k n) * kron m (mirror k n) -
    2 * (-F.a) * F.β * F.PT k u / F.DhatT k u * F.ghatT k (F.zT k n) * F.ghatT k (F.zT k m)

/-- The assumptions of the transverse corollary at `μ = 0`. -/
structure THFHyp (u : ℂ) : Prop where
  a_ne : (F.a : ℂ) ≠ 0
  μ_eq : F.μ = 0
  W_ne : F.WT k ≠ 0
  q_ne : F.q₀T k ≠ 0
  z_sq_ne : ∀ n, F.zT k n ^ 2 - F.shatT k ≠ 0
  odd_ne : ∀ n, n ≠ mirror k n → F.zT k n ^ 2 - F.shatT' k ≠ 0
  u_sq : u ^ 2 = F.shatT k
  P_ne : F.PT k u ≠ 0
  D_ne : F.DhatT k u ≠ 0

/-! ### The parity at `μ = 0` -/

theorem p₂_eq_p₀_of_mu (hμ : F.μ = 0) : F.p₂ = F.p₀ := by
  rw [F.p₂_eq_p₀_mul, hμ, mul_zero, Real.exp_zero, mul_one]

theorem rho_eq_zeroT (hμ : F.μ = 0) : (F.ρ : ℂ) = 0 := by
  rw [ρ, F.p₂_eq_p₀_of_mu hμ, sub_self, ofReal_zero]

theorem HT_neg_of_mu (hμ : F.μ = 0) (z : ℂ) : F.HT k (-z) = F.HT k z := by
  simp only [HT, F.rho_eq_zeroT hμ]
  ring

theorem NT_neg_of_mu (hμ : F.μ = 0) (z t : ℂ) : F.NT k z (-t) = F.NT k z t := by
  rw [F.NT_eq_coeffs, F.NT_eq_coeffs]
  simp only [cN₁T, cN₃T, F.rho_eq_zeroT hμ]
  ring

/-- `N(z, t) = N₀(z) + t²N₂(z)` at `μ = 0`. -/
theorem NT_of_mu (hμ : F.μ = 0) (z t : ℂ) :
    F.NT k z t = F.cN₀T k z + F.N₂T k z * t ^ 2 := by
  rw [F.NT_eq_coeffs]
  simp only [cN₁T, cN₂T, cN₃T, N₂T, F.rho_eq_zeroT hμ]
  ring

theorem ghatT_neg (z : ℂ) : F.ghatT k (-z) = F.ghatT k z := by
  rw [ghatT, ghatT, neg_sq]

theorem HT_sub_of_mu (hμ : F.μ = 0) (z : ℂ) : F.HT k z - F.lamT * F.p₀ = z ^ 2 - F.shatT k := by
  simp only [HT, shatT, F.rho_eq_zeroT hμ]
  ring

theorem HT_add_of_mu (hμ : F.μ = 0) (z : ℂ) :
    F.HT k z + F.lamT * F.p₀ = z ^ 2 - F.shatT' k := by
  simp only [HT, shatT', F.rho_eq_zeroT hμ]
  ring

theorem eT_of_mu (hμ : F.μ = 0) (z : ℂ) :
    F.eT k z = (z ^ 2 - F.shatT k) * (z ^ 2 - F.shatT' k) := by
  rw [eT, F.HT_neg_of_mu k hμ, F.p₂_eq_p₀_of_mu hμ, ← F.HT_sub_of_mu k hμ,
    ← F.HT_add_of_mu k hμ]
  ring

theorem HT_of_mu (hμ : F.μ = 0) (z : ℂ) : F.HT k z = z ^ 2 - (F.shatT k + F.shatT' k) / 2 := by
  simp only [HT, shatT, shatT', F.rho_eq_zeroT hμ]
  ring

theorem lamT_p₀ : F.lamT * (F.β * F.p₀) = F.β * ((F.shatT k - F.shatT' k) / 2) := by
  simp only [shatT, shatT']
  ring

theorem ievnT_neg (z : ℂ) : F.ievnT k (-z) = F.ievnT k z := by
  rw [ievnT, ievnT, F.PT_neg, neg_sq]

theorem ioddT_neg (z : ℂ) : F.ioddT k (-z) = F.ioddT k z := by
  rw [ioddT, ioddT, F.PT_neg, neg_sq]

theorem zT_mirror_of_mu (hμ : F.μ = 0) (n : ℤ) : F.zT k (mirror k n) = -F.zT k n :=
  (F.lat F.μ).z_mirror_of_mu k hμ n

/-- At `μ = 0` the pair rate vanishes exactly on the mirror. -/
theorem pair_iff_mirror (hμ : F.μ = 0) (n m : ℤ) :
    (F.μ = 0 ∧ n + m + k + 1 = 0) ↔ m = mirror k n := by
  unfold mirror
  constructor
  · rintro ⟨-, h⟩
    omega
  · intro h
    exact ⟨hμ, by omega⟩

/-- The right side of the transverse input in the variables of `FieldTAlg.lean`. -/
theorem NT_zT_eq (n m : ℤ) : F.NT k (F.zT k n) (F.zT k m) =
    Nf F.a F.h ((F.lat F.μ).x n) ((F.lat F.μ).x m) ((F.lat F.μ).w k) 1 F.p₀ F.pUp F.pDn F.p₂ := by
  rw [NT, N₀T, F.PT_zT]
  simp only [N₂T, ζT, q₀T, zT, Params.z, WT_eq, Nf, N₂f, ρ, mf]
  push_cast
  ring

theorem HT_zT_eq (n : ℤ) : F.HT k (F.zT k n) =
    HTf F.a F.h ((F.lat F.μ).x n) ((F.lat F.μ).w k) 1 F.p₀ F.pUp F.pDn F.p₂ := by
  simp only [HT, q₁T, q₀T, zT, Params.z, WT_eq, HTf, ρ, mf]
  push_cast
  ring

/-- **The transverse input at `μ = 0`**, where the Cauchy matrix is the mirror `βp₀𝓜`:
`P(zₙ)χ_T↑,nm = -βH(zₙ)δₙₘ + λβp₀δ_{m n̄} + N(zₙ, zₘ)/P(zₘ)`. -/
theorem PT_mul_chiTUp_hf (hμ : F.μ = 0) (hW0 : F.WT k ≠ 0) (n m : ℤ) :
    F.PT k (F.zT k n) * F.chiTUp k n m =
      -F.β * F.HT k (F.zT k n) * kron n m + F.lamT * (F.β * F.p₀) * kron m (mirror k n) +
        F.NT k (F.zT k n) (F.zT k m) / F.PT k (F.zT k m) := by
  by_cases hm : m = mirror k n
  · subst hm
    set B := F.lat F.μ
    have hs : 2 * (F.p₀ : ℂ) + F.pUp + F.pDn = 1 := by
      have := F.weights_sum'
      rw [F.p₂_eq_p₀_of_mu hμ] at this
      linear_combination this
    have hY : B.x n + B.x (mirror k n) + B.w k = 0 :=
      (B.pair_eq_zero_iff k n (mirror k n)).mpr ⟨hμ, by unfold mirror; ring⟩
    rw [F.chiTUp_eq_pathsTD k hW0, F.p₂_eq_p₀_of_mu hμ, kron_self, F.PT_zT, F.NT_zT_eq,
      F.p₂_eq_p₀_of_mu hμ, F.PT_zT]
    by_cases hfix : n = mirror k n
    · rw [pathsTD_congr _ _ _ _ _ _ _ _ _ _ (iff_true_intro hfix)
        (iff_true_intro ((F.pair_iff_mirror k hμ n _).mpr rfl)), ← hfix, kron_self,
        F.HT_zT_eq, F.p₂_eq_p₀_of_mu hμ]
      have hX : B.x n + B.x n + B.w k = 0 := by
        have := hY
        rwa [← hfix] at this
      rw [pathsTD_fixed hs (F.lin₁ n) (F.lin₂ n) (F.lin₃ k n) (F.lin₄ k n) hW0 hX, rhsTM, lamT]
      ring
    · have hXY : B.x n - B.x (mirror k n) ≠ 0 := fun h0 ↦
        hfix ((B.x_sub_x_eq_zero_iff n (mirror k n)).mp h0)
      rw [pathsTD_congr _ _ _ _ _ _ _ _ _ _ (iff_false_intro hfix)
        (iff_true_intro ((F.pair_iff_mirror k hμ n _).mpr rfl)), kron_of_ne hfix,
        pathsTD_mirror hs (F.lin₁ n) (F.lin₂ n) (F.lin₃ k n) (F.lin₄ k n) hW0 hY hXY, rhsTM,
        lamT]
      ring
  · have hD : ¬(F.μ = 0 ∧ n + m + k + 1 = 0) := fun h ↦ hm ((F.pair_iff_mirror k hμ n m).mp h)
    rw [F.PT_mul_chiTUp' k hW0 hD, kron_of_ne hm, F.rho_eq_zeroT hμ]
    ring

/-! ### The lattice sums -/

theorem hasSum_ℓ₀T (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0) :
    HasSum (fun l ↦ 1 / F.PT k (F.zT k l)) (F.ℓ₀T k) := by
  convert F.hasSum_cubicT k ha hW hq 1 0 0 0 ((F.lat F.μ).z_ne_neg_add k 0) using 1
  · funext l
    ring
  · ring

theorem hasSum_ℓ₂T (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0) :
    HasSum (fun l ↦ F.zT k l ^ 2 / F.PT k (F.zT k l)) (F.ℓ₂T k) := by
  convert F.hasSum_cubicT k ha hW hq 0 0 1 0 ((F.lat F.μ).z_ne_neg_add k 0) using 1
  · funext l
    ring
  · ring

/-- `∑ₗ N(z, zₗ)/P(zₗ) = N₀(z)ℓ₀ + N₂(z)ℓ₂` at `μ = 0`. -/
theorem hasSum_NT_div (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0)
    (hμ : F.μ = 0) (z : ℂ) :
    HasSum (fun l ↦ F.NT k z (F.zT k l) / F.PT k (F.zT k l))
      (F.cN₀T k z * F.ℓ₀T k + F.N₂T k z * F.ℓ₂T k) := by
  convert ((F.hasSum_ℓ₀T k ha hW hq).mul_left (F.cN₀T k z)).add
    ((F.hasSum_ℓ₂T k ha hW hq).mul_left (F.N₂T k z)) using 1
  funext l
  rw [F.NT_of_mu k hμ]
  ring

variable {F k} {u : ℂ}

/-- `∑ₗ ĝ(zₗ) = T̂`: `S_k(û)/û` by Lemma "basic sum", and `∑ₗ zₗ⁻² = -β²/4` at `û = 0`. -/
theorem THFHyp.hasSum_That (h : F.THFHyp k u) :
    HasSum (fun l ↦ F.ghatT k (F.zT k l)) (F.ThatT k u) := by
  have hμ := h.μ_eq
  by_cases hu0 : u = 0
  · -- then `k` is even: for odd `k`, `z = 0` is a lattice point
    have hs : F.shatT k = 0 := by rw [← h.u_sq, hu0]; ring
    have hk : Even k := by
      rcases Int.even_or_odd k with hk | hk
      · exact hk
      · obtain ⟨n, hn⟩ := (F.lat F.μ).exists_z_eq_mu_of_odd hk
        exact absurd (by simp only [zT]; rw [hn, lat_μ, hμ, hs]; simp) (h.z_sq_ne n)
    have hz : ∀ l, F.zT k l ≠ 0 := fun l h0 ↦ h.z_sq_ne l (by rw [h0, hs]; ring)
    have hS : (F.lat F.μ).S k 0 = 0 := by
      have := (F.lat F.μ).S_mu_of_even hk
      have h0 : (((F.lat F.μ).μ : ℝ) : ℂ) = 0 := by rw [lat_μ, hμ, ofReal_zero]
      rwa [h0] at this
    have h₀ := (F.lat F.μ).hasSum_sq (k := k) (x := 0) hz
    rw [hS] at h₀
    rw [ThatT, ite_eq_left hu0]
    convert h₀ using 1
    · funext l
      rw [ghatT, hs, sub_zero, sub_zero]
      rfl
    · simp only [lat_β]
      ring
  · have hzu : ∀ l, F.zT k l ≠ u := fun l h0 ↦ h.z_sq_ne l (by rw [h0, h.u_sq]; ring)
    have hzu' : ∀ l, F.zT k l ≠ -u := fun l h0 ↦ h.z_sq_ne l (by rw [h0, neg_sq, h.u_sq]; ring)
    have hS : (F.lat F.μ).S k (-u) = -(F.lat F.μ).S k u := by
      have := (F.lat F.μ).S_mu_sub k u
      have h0 : (((F.lat F.μ).μ : ℝ) : ℂ) = 0 := by rw [lat_μ, hμ, ofReal_zero]
      rwa [h0, zero_sub, zero_add] at this
    rw [ThatT, ite_eq_right hu0]
    convert ((F.lat F.μ).hasSum_sub_inv k hzu hzu').div_const (2 * u) using 1
    · funext l
      have h₁ := sub_ne_zero.mpr (hzu l)
      have h₂ : F.zT k l + u ≠ 0 := fun h0 ↦ hzu' l (by linear_combination h0)
      have h₃ : F.zT k l ^ 2 - u ^ 2 ≠ 0 := by
        rw [show F.zT k l ^ 2 - u ^ 2 = (F.zT k l - u) * (F.zT k l + u) by ring]
        exact mul_ne_zero h₁ h₂
      simp only [zT] at h₁ h₂ h₃ ⊢
      rw [ghatT, ← h.u_sq, sub_neg_eq_add, div_sub_div _ _ h₁ h₂, div_div,
        eq_div_iff (mul_ne_zero (mul_ne_zero h₁ h₂) (mul_ne_zero two_ne_zero hu0))]
      field_simp
      ring
    · rw [hS]
      field_simp
      ring

set_option maxHeartbeats 4000000 in
-- a rational identity of high degree in the weights and the transfer
variable (F k) in
/-- The rational identity behind `THFHyp.hasSum_NT_ghat`, with `P(û) = ŝ² + κŝ + q₀²` and
`N(z, û) = N₀(z) + N₂(z)ŝ`: `-2a[(N₂(z)P(û) - N(z, û)(κ + ŝ))ℓ₀ - N(z, û)ℓ₂]
= -2aβP(û) - βN(z, û)`. -/
theorem identity_hfT (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0) (hμ : F.μ = 0)
    (z : ℂ) :
    2 * (-F.a) * ((F.N₂T k z * (F.shatT k ^ 2 + F.κT k * F.shatT k + F.q₀T k ^ 2) -
        (F.cN₀T k z + F.N₂T k z * F.shatT k) * (F.κT k + F.shatT k)) * F.ℓ₀T k -
      (F.cN₀T k z + F.N₂T k z * F.shatT k) * F.ℓ₂T k) =
    2 * (-F.a) * F.β * (F.shatT k ^ 2 + F.κT k * F.shatT k + F.q₀T k ^ 2) -
      F.β * (F.cN₀T k z + F.N₂T k z * F.shatT k) := by
  have hβ := F.β_ne_zeroT
  have h₁ : (F.pUp : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos F.pUp_pos F.p₀_pos).ne'
  have h₂ : (F.pDn : ℂ) + F.p₀ ≠ 0 := by exact_mod_cast (add_pos F.pDn_pos F.p₀_pos).ne'
  have h₃ : (F.p₀ : ℂ) + F.pDn ≠ 0 := by exact_mod_cast (add_pos F.p₀_pos F.pDn_pos).ne'
  have h₄ : (F.p₀ : ℂ) + F.pUp ≠ 0 := by exact_mod_cast (add_pos F.p₀_pos F.pUp_pos).ne'
  have hs := F.weights_sum'
  have hp₂ := F.p₂_eq_p₀_of_mu hμ
  simp only [cN₀T, N₂T, ℓ₀T, ℓ₂T, TpT, TmT, SpT, SmT, F.sT₁_eq, F.sT₂_eq, F.sT₃_eq, F.sT₄_eq,
    ζT, κT, shatT, q₁T, lamT, ρ, mf, hp₂]
  rw [hp₂] at hs
  rw [q₀T] at hq ⊢
  push_cast
  generalize F.WT k = W at *
  generalize (F.a : ℂ) = a at *
  generalize (F.β : ℂ) = β at *
  generalize (F.pDn : ℂ) = pd at *
  obtain rfl : pd = 1 - 2 * F.p₀ - F.pUp := by linear_combination hs
  generalize hQ : a ^ 2 - W ^ 2 / 4 = Q at hq ⊢
  field_simp
  subst hQ
  ring

/-- `∑ₗ N(z, zₗ)ĝ(zₗ)/P(zₗ) = β - N(z, û)D̂/(2εaP(û))`, by partial fractions in `zₗ²`:
`N(z, t)ĝ(t)/P(t) = cĝ(t) + (N₂(z) - c(κ + ŝ) - ct²)/P(t)` with `c = N(z, û)/P(û)`. -/
theorem THFHyp.hasSum_NT_ghat (h : F.THFHyp k u) (z : ℂ) :
    HasSum (fun l ↦ F.NT k z (F.zT k l) * F.ghatT k (F.zT k l) / F.PT k (F.zT k l))
      (F.β - F.NT k z u * F.DhatT k u / (2 * (-F.a) * F.PT k u)) := by
  have hμ := h.μ_eq
  have ha := h.a_ne
  have hPu := h.P_ne
  have hPu' : F.PT k u = F.shatT k ^ 2 + F.κT k * F.shatT k + F.q₀T k ^ 2 := by
    rw [F.PT_eq_poly, show u ^ 4 = (u ^ 2) ^ 2 by ring, h.u_sq]
  have hNu : F.NT k z u = F.cN₀T k z + F.N₂T k z * F.shatT k := by
    rw [F.NT_of_mu k hμ, h.u_sq]
  convert ((h.hasSum_That.mul_left (F.NT k z u / F.PT k u)).add
    ((F.hasSum_ℓ₀T k h.a_ne h.W_ne h.q_ne).mul_left
      (F.N₂T k z - F.NT k z u / F.PT k u * (F.κT k + F.shatT k)))).add
    ((F.hasSum_ℓ₂T k h.a_ne h.W_ne h.q_ne).mul_left (-(F.NT k z u / F.PT k u))) using 1
  · funext l
    have h₁ := h.z_sq_ne l
    have h₂ := F.PT_zT_ne_zero k l
    rw [F.PT_eq_poly] at h₂ ⊢
    rw [hPu'] at hPu ⊢
    rw [F.NT_of_mu k hμ, ghatT, hNu]
    -- clear the denominators as atoms, then put the polynomials back
    generalize hD : F.shatT k ^ 2 + F.κT k * F.shatT k + F.q₀T k ^ 2 = D at hPu ⊢
    generalize hPl : F.zT k l ^ 4 + F.κT k * F.zT k l ^ 2 + F.q₀T k ^ 2 = Pl at h₂ ⊢
    generalize hG : F.zT k l ^ 2 - F.shatT k = G at h₁ ⊢
    field_simp
    rw [← hD, ← hPl, ← hG]
    ring
  · have key := F.identity_hfT k ha h.W_ne h.q_ne hμ z
    rw [hPu'] at hPu ⊢
    rw [hNu, DhatT]
    generalize F.shatT k ^ 2 + F.κT k * F.shatT k + F.q₀T k ^ 2 = D at hPu key ⊢
    field_simp
    linear_combination key

/-! ### The right inverse -/

variable (F k) in
/-- The row `P(zₙ)χ_T↑,nl` at `μ = 0` (`PT_mul_chiTUp_hf`). -/
noncomputable def RrowTHF (n l : ℤ) : ℂ :=
  -F.β * F.HT k (F.zT k n) * kron n l + F.lamT * (F.β * F.p₀) * kron l (mirror k n) +
    F.NT k (F.zT k n) (F.zT k l) / F.PT k (F.zT k l)

variable (F k) in
/-- Identity (A6) at `μ = 0`: `-βH(z) + λβp₀ + N₀(z)ℓ₀ + N₂(z)ℓ₂ = -βN₂(z)/(2a)`. -/
theorem identity_A6_hfT (ha : (F.a : ℂ) ≠ 0) (hW : F.WT k ≠ 0) (hq : F.q₀T k ≠ 0)
    (hμ : F.μ = 0) (z : ℂ) :
    -F.β * F.HT k z + F.lamT * (F.β * F.p₀) + (F.cN₀T k z * F.ℓ₀T k + F.N₂T k z * F.ℓ₂T k) =
      F.β * F.N₂T k z / (2 * (-F.a)) := by
  have h6 := F.identity_A6T k ha hW hq z
  have hρ := F.rho_eq_zeroT hμ
  have hc : (F.cbarT : ℂ) = F.β * F.p₀ := by
    rw [cbarT, F.p₂_eq_p₀_of_mu hμ]
    push_cast
    ring
  have h₂ : F.cN₂T k z = F.N₂T k z := by
    simp only [cN₂T, N₂T, hρ]
    ring
  rw [hc, h₂] at h6
  simp only [cN₁T, cN₃T, hρ] at h6
  rw [show F.β * F.N₂T k z / (2 * (-(F.a : ℂ))) = -(F.β * F.N₂T k z / (2 * F.a)) by ring]
  linear_combination h6

variable (F k) in
theorem mul_kron_mirror_eqT (hμ : F.μ = 0) (f : ℂ → ℂ) (hf : ∀ z, f (-z) = f z) (n m : ℤ) :
    f (F.zT k m) * kron n (mirror k m) = f (F.zT k n) * kron n (mirror k m) := by
  by_cases h : n = mirror k m
  · rw [h, F.zT_mirror_of_mu k hμ, hf]
  · rw [kron_of_ne h, mul_zero, mul_zero]

/-- The sum of the row, Step 2 at `μ = 0`: `∑ₗ P(zₙ)χ_r,nl = βN₂(zₙ)/(2εa)`. -/
theorem THFHyp.hasSum_RrowTHF (h : F.THFHyp k u) (n : ℤ) :
    HasSum (fun l ↦ F.RrowTHF k n l) (F.β * F.N₂T k (F.zT k n) / (2 * (-F.a))) := by
  have h₁ : HasSum (fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l)
      (-F.β * F.HT k (F.zT k n)) := by
    convert hasSum_single (f := fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero]) using 1
    rw [kron_self, mul_one]
  have h₂ : HasSum (fun l ↦ F.lamT * (F.β * F.p₀) * kron l (mirror k n))
      (F.lamT * (F.β * F.p₀)) := by
    convert hasSum_single (f := fun l ↦ F.lamT * (F.β * F.p₀) * kron l (mirror k n))
      (mirror k n) (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero]) using 1
    rw [kron_self, mul_one]
  convert (h₁.add h₂).add (F.hasSum_NT_div k h.a_ne h.W_ne h.q_ne h.μ_eq (F.zT k n)) using 1
  · funext l
    rw [RrowTHF]
  · exact (F.identity_A6_hfT k h.a_ne h.W_ne h.q_ne h.μ_eq (F.zT k n)).symm

/-- The image of `ĝ`: `∑ₗ P(zₙ)χ_r,nl ĝ(zₗ) = -N(zₙ, û)D̂/(2εaP(û))`. -/
theorem THFHyp.hasSum_RrowTHF_ghat (h : F.THFHyp k u) (n : ℤ) :
    HasSum (fun l ↦ F.RrowTHF k n l * F.ghatT k (F.zT k l))
      (-(F.NT k (F.zT k n) u * F.DhatT k u / (2 * (-F.a) * F.PT k u))) := by
  have hμ := h.μ_eq
  have h₁ : HasSum (fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l * F.ghatT k (F.zT k l))
      (-F.β * F.HT k (F.zT k n) * F.ghatT k (F.zT k n)) := by
    convert hasSum_single (f := fun l ↦ -F.β * F.HT k (F.zT k n) * kron n l *
      F.ghatT k (F.zT k l)) n
      (fun l hl ↦ by simp only [kron_of_ne (Ne.symm hl), mul_zero, zero_mul]) using 1
    rw [kron_self, mul_one]
  have h₂ : HasSum (fun l ↦ F.lamT * (F.β * F.p₀) * kron l (mirror k n) * F.ghatT k (F.zT k l))
      (F.lamT * (F.β * F.p₀) * F.ghatT k (F.zT k n)) := by
    convert hasSum_single (f := fun l ↦ F.lamT * (F.β * F.p₀) * kron l (mirror k n) *
      F.ghatT k (F.zT k l)) (mirror k n)
      (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero, zero_mul]) using 1
    rw [kron_self, mul_one, F.zT_mirror_of_mu k hμ, F.ghatT_neg]
  have hdiag : -F.β * F.HT k (F.zT k n) * F.ghatT k (F.zT k n) +
      F.lamT * (F.β * F.p₀) * F.ghatT k (F.zT k n) = -F.β := by
    have hg := h.z_sq_ne n
    have hHs := F.HT_sub_of_mu k hμ (F.zT k n)
    rw [ghatT]
    field_simp
    linear_combination -(F.β : ℂ) * hHs
  convert (h₁.add h₂).add (h.hasSum_NT_ghat (F.zT k n)) using 1
  · funext l
    rw [RrowTHF]
    ring
  · linear_combination -hdiag

/-- The diagonal and mirror part of the column `m`: with `ε±(v)` the inverse even and odd
eigenvalues, `P(zₙ)χ_r,nm (ε₊ + ε₋)/2 + P(zₙ)χ_r,nm̄ (ε₊ - ε₋)/2 = P(zₙ)δₙₘ - N(zₙ, v)ĝ(v)/β`,
`v = zₘ`; at a mirror fixed point only `ε₊` enters. -/
theorem THFHyp.delta_part (h : F.THFHyp k u) (n m : ℤ) :
    F.RrowTHF k n m * ((F.ievnT k (F.zT k m) + F.ioddT k (F.zT k m)) / 2) +
        F.RrowTHF k n (mirror k m) * ((F.ievnT k (F.zT k m) - F.ioddT k (F.zT k m)) / 2) =
      F.PT k (F.zT k n) * kron n m -
        F.NT k (F.zT k n) (F.zT k m) * F.ghatT k (F.zT k m) / F.β := by
  have hμ := h.μ_eq
  have hβ := F.β_ne_zeroT
  have hPm := F.PT_zT_ne_zero k m
  have hgm := h.z_sq_ne m
  have hzm := F.zT_mirror_of_mu k hμ m
  simp only [RrowTHF, ievnT, ioddT, ghatT, hzm, F.NT_neg_of_mu k hμ, F.PT_neg,
    kron_mirror_mirror, kron_comm m n, ← kron_mirror_comm k n m, F.lamT_p₀ k]
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self, F.HT_of_mu k hμ]
    by_cases hfix : n = mirror k n
    · -- a mirror fixed point: the odd eigenvalue drops out
      rw [← hfix, kron_self]
      field_simp
      ring
    · have hodd := h.odd_ne n hfix
      rw [kron_of_ne hfix]
      field_simp
      ring
  · rw [kron_of_ne hnm]
    by_cases hmir : n = mirror k m
    · -- `zₙ = -zₘ`
      have hm' : m ≠ mirror k m := fun h0 ↦ hnm (by rw [hmir, ← h0])
      have hodd := h.odd_ne m hm'
      rw [hmir, kron_self, hzm, F.HT_neg_of_mu k hμ, F.HT_of_mu k hμ]
      field_simp
      ring
    · rw [kron_of_ne hmir]
      field_simp
      ring

/-- **Corollary "half filling"**, the inverse: `χ_r invHF = 1`. By Proposition "structure" at
`μ = 0`, `P(zₙ)(χ_r invHF)ₙₘ` is `delta_part` plus `2εa/β²` times the row sum plus the image of `ĝ`,
which add up to `P(zₙ)δₙₘ`, as `N(zₙ, zₘ) - N(zₙ, û) = N₂(zₙ)(zₘ² - ŝ)`. -/
theorem THFHyp.chiTUp_isRightInverse (h : F.THFHyp k u) :
    IsRightInverse (F.chiTUp k) (F.invTHF k u) := by
  intro n m
  have hμ := h.μ_eq
  have ha := h.a_ne
  have hβ := F.β_ne_zeroT
  have hPn := F.PT_zT_ne_zero k n
  have hPu := h.P_ne
  have hD := h.D_ne
  have hgm := h.z_sq_ne m
  set ep := (F.ievnT k (F.zT k m) + F.ioddT k (F.zT k m)) / 2 with hep
  set em := (F.ievnT k (F.zT k m) - F.ioddT k (F.zT k m)) / 2 with hem
  have s₁ : HasSum (fun l ↦ F.RrowTHF k n l * (ep * kron l m)) (F.RrowTHF k n m * ep) := by
    convert hasSum_single (f := fun l ↦ F.RrowTHF k n l * (ep * kron l m)) m
      (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero]) using 1
    rw [kron_self, mul_one]
  have s₂ : HasSum (fun l ↦ F.RrowTHF k n l * (em * kron l (mirror k m)))
      (F.RrowTHF k n (mirror k m) * em) := by
    convert hasSum_single (f := fun l ↦ F.RrowTHF k n l * (em * kron l (mirror k m)))
      (mirror k m) (fun l hl ↦ by simp only [kron_of_ne hl, mul_zero]) using 1
    rw [kron_self, mul_one]
  have s₃ := (h.hasSum_RrowTHF n).mul_left (2 * (-(F.a : ℂ)) / (F.β : ℂ) ^ 2)
  have s₄ := (h.hasSum_RrowTHF_ghat n).mul_left
    (-(2 * (-F.a) * F.PT k u / (F.β * F.DhatT k u)) * F.ghatT k (F.zT k m))
  convert (((s₁.add s₂).add s₃).add s₄).div_const (F.PT k (F.zT k n)) using 1
  · funext l
    have hc : F.chiTUp k n l = F.RrowTHF k n l / F.PT k (F.zT k n) := by
      rw [eq_div_iff hPn, mul_comm, RrowTHF]
      exact F.PT_mul_chiTUp_hf k hμ h.W_ne n l
    have hmb : mirrorBlock k (fun j ↦ F.ievnT k (F.zT k j)) (fun j ↦ F.ioddT k (F.zT k j)) l m =
        ep * kron l m + em * kron l (mirror k m) := by
      rw [mirrorBlock, hep, hem]
      have h₁ := mul_kron_eq (fun j ↦ (F.ievnT k (F.zT k j) + F.ioddT k (F.zT k j)) / 2) l m
      have h₂ : (F.ievnT k (F.zT k l) - F.ioddT k (F.zT k l)) / 2 * kron m (mirror k l) =
          (F.ievnT k (F.zT k m) - F.ioddT k (F.zT k m)) / 2 * kron l (mirror k m) := by
        rw [kron_mirror_comm k m l]
        have := F.mul_kron_mirror_eqT k hμ (fun z ↦ (F.ievnT k z - F.ioddT k z) / 2)
          (fun z ↦ by rw [F.ievnT_neg, F.ioddT_neg]) l m
        simpa using this.symm
      rw [← h₁, h₂]
    rw [hc, invTHF, hmb]
    field_simp
    ring
  · rw [hep, hem, h.delta_part n m, F.NT_of_mu k hμ, F.NT_of_mu k hμ, h.u_sq, ghatT]
    field_simp
    ring

/-- **Corollary "half filling"**, the closed form: `β²(χ_r⁻¹ - B⁻¹) = Γ_r` of Eq. (Gamma-hf). -/
theorem THFHyp.gammaTHF_eq (h : F.THFHyp k u) (n m : ℤ) :
    (F.β : ℂ) ^ 2 * (F.invTHF k u n m - kron n m / F.bTUp k n) = F.gammaTHF k u n m := by
  have hμ := h.μ_eq
  have hβ := F.β_ne_zeroT
  have hPn := F.PT_zT_ne_zero k n
  have hGn := F.G₂T_zT_ne_zero k n
  have hD := h.D_ne
  rw [invTHF, gammaTHF, F.bTUp_eq k n]
  field_simp
  ring

variable (F k) in
/-- Where `e(zₙ) ≠ 0`, the orbit form `gammaHF` is Eq. (Gamma-hf) as printed. -/
theorem gammaTHF_eq_note (hμ : F.μ = 0) {u : ℂ} {n : ℤ} (he : F.eT k (F.zT k n) ≠ 0)
    (m : ℤ) : F.gammaTHF k u n m = F.gammaTHFnote k u n m := by
  have hβ := F.β_ne_zeroT
  have hGn := F.G₂T_zT_ne_zero k n
  rw [F.eT_of_mu k hμ] at he
  have h₁ := left_ne_zero_of_mul he
  have h₂ := right_ne_zero_of_mul he
  rw [gammaTHF, gammaTHFnote, mirrorBlock, FT, F.eT_of_mu k hμ, F.HT_of_mu k hμ,
    F.lamT_p₀ k]
  simp only [ievnT, ioddT]
  field_simp
  ring

variable (F k) in
/-- `χ_r` is symmetric at `μ = 0` (via Proposition "structure"). -/
theorem chiTUp_symm_hf (hμ : F.μ = 0) (hW : F.WT k ≠ 0) (n m : ℤ) :
    F.chiTUp k n m = F.chiTUp k m n := by
  by_cases hnm : n = m
  · rw [hnm]
  have hPn := F.PT_zT_ne_zero k n
  have hPm := F.PT_zT_ne_zero k m
  have h₁ := F.PT_mul_chiTUp_hf k hμ hW n m
  have h₂ := F.PT_mul_chiTUp_hf k hμ hW m n
  rw [kron_of_ne hnm, mul_zero, zero_add] at h₁
  rw [kron_of_ne (Ne.symm hnm), mul_zero, zero_add, ← kron_mirror_comm] at h₂
  have hN : F.NT k (F.zT k m) (F.zT k n) = F.NT k (F.zT k n) (F.zT k m) := by
    rw [F.NT_of_mu k hμ, F.NT_of_mu k hμ]
    simp only [cN₀T, N₂T, F.rho_eq_zeroT hμ]
    ring
  have hPd := F.mul_kron_mirror_eqT k hμ (F.PT k) (F.PT_neg k) m n
  apply mul_left_cancel₀ (mul_ne_zero hPn hPm)
  rw [hN] at h₂
  field_simp at h₁ h₂
  linear_combination h₁ - h₂ - F.lamT * (F.β * F.p₀) * hPd

theorem THFHyp.invTHF_symm (h : F.THFHyp k u) (n m : ℤ) :
    F.invTHF k u n m = F.invTHF k u m n := by
  rw [invTHF, invTHF, mirrorBlock_symm (fun j ↦ by rw [F.zT_mirror_of_mu k h.μ_eq, F.ievnT_neg])
    (fun j ↦ by rw [F.zT_mirror_of_mu k h.μ_eq, F.ioddT_neg])]
  ring

/-- **Corollary "half filling"**, two-sidedness: `invHF χ_r = 1`, as `χ_r` and `invHF` are
symmetric. -/
theorem THFHyp.chiTUp_isLeftInverse (h : F.THFHyp k u) :
    IsLeftInverse (F.chiTUp k) (F.invTHF k u) := by
  intro n m
  have := h.chiTUp_isRightInverse m n
  rw [kron_comm] at this
  convert this using 2 with l
  rw [F.chiTUp_symm_hf k h.μ_eq h.W_ne l m, h.invTHF_symm n l, mul_comm]

/-! ### The other oriented sectors on the axes -/

/-- **The opposite-spin pair `A↑` at `h = 0`**: the image of the transverse sector at `μ = 0`. -/
theorem gammaAUp_eq_hf (h : F.shiba.THFHyp k u) :
    IsRightInverse (F.chiAUp k) (F.shiba.invTHF k u) ∧
      IsLeftInverse (F.chiAUp k) (F.shiba.invTHF k u) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * (F.shiba.invTHF k u n m - kron n m / F.bAUp k n) =
        F.shiba.gammaTHF k u n m := by
  have hχ : F.chiAUp k = F.shiba.chiTUp k := funext fun n ↦ funext fun m ↦ F.chiAUp_eq k n m
  refine ⟨hχ ▸ h.chiTUp_isRightInverse, hχ ▸ h.chiTUp_isLeftInverse, fun n m ↦ ?_⟩
  rw [F.bAUp_eq k n]
  exact h.gammaTHF_eq n m

/-- **The transverse sector `T↓` at `μ = 0`**. -/
theorem gammaTDn_eq_hf (h : F.flip.THFHyp k u) :
    IsRightInverse (F.chiTDn k) (F.flip.invTHF k u) ∧
      IsLeftInverse (F.chiTDn k) (F.flip.invTHF k u) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * (F.flip.invTHF k u n m - kron n m / F.bTDn k n) =
        F.flip.gammaTHF k u n m := by
  have hχ : F.chiTDn k = F.flip.chiTUp k := funext fun n ↦ funext fun m ↦ F.chiTDn_eq k n m
  refine ⟨hχ ▸ h.chiTUp_isRightInverse, hχ ▸ h.chiTUp_isLeftInverse, fun n m ↦ ?_⟩
  rw [F.bTDn_eq k n]
  exact h.gammaTHF_eq n m

/-- **The opposite-spin pair `A↓` at `h = 0`**. -/
theorem gammaADn_eq_hf (h : F.flip.shiba.THFHyp k u) :
    IsRightInverse (F.chiADn k) (F.flip.shiba.invTHF k u) ∧
      IsLeftInverse (F.chiADn k) (F.flip.shiba.invTHF k u) ∧
      ∀ n m, (F.β : ℂ) ^ 2 * (F.flip.shiba.invTHF k u n m - kron n m / F.bADn k n) =
        F.flip.shiba.gammaTHF k u n m := by
  have hχ : F.chiADn k = F.flip.shiba.chiTUp k :=
    funext fun n ↦ funext fun m ↦ F.chiADn_eq k n m
  refine ⟨hχ ▸ h.chiTUp_isRightInverse, hχ ▸ h.chiTUp_isLeftInverse, fun n m ↦ ?_⟩
  rw [F.bADn_eq k n]
  exact h.gammaTHF_eq n m

end FParams

end HubbardAtom
