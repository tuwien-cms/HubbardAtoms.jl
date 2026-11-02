import HubbardAtomLean.WardIdentities

/-!
# The spectral identity of the density and magnetic susceptibilities

For `g_s(n) = 1/(zₙ² - s)`, `s = u²` off the lattice, and `f_r` of `DividedDifferences.lean`:

* `Params.hasSum_g`: `∑ₙ g_s(n) = 𝒯(u)`, with `𝒯(u) = (S_k(u) - S_k(-u))/(2u)` and
  `𝒯(0) = S_k(0)² - β²/4`;
* `Params.chiR_mul_gs`: the divided differences in pointwise form, diagonal included;
  `summable_chiR_mul_gs`, `summable_fR_mul_gs`: convergence of `∑ₘ χ_r,nm g_s(m)` and
  `∑ₘ f_r(zₘ)g_s(m)`;
* `Params.hasSum_chiR_mul_g`: for `U ≠ 0`, `μ ≠ 0`,
  `∑ₘ χ_r,nm g_s(m) = ℓ(s)g_s(n) - β[1 - (εU/β)𝒯(s)]f_r(zₙ)g_s(n)` with
  `ℓ(s) = -εU ∑ₙ f_r(zₙ)g_s(n)`, by the divided differences and the Ward identity
  (`hasSum_chiR`);
* `Params.hasSum_chiR_mul_g_of`: where `1 - (εU/β)𝒯(s) = 0`, `g_s` is an eigenvector of `χ_r`
  with eigenvalue `ℓ(s)`. `SpectralLink.lean` writes both in terms of `Φ_k`, `D_k` and `K_e`.

The singlet, for `g_s(n) = 1/(tₙ - s)` (mirror-even) and `b ≠ 0`:

* `Params.chiS_mul_gt`, `summable_chiS_mul_gt`, `summable_fS_mul_gt`: as above;
* `Params.hasSum_chiS_mul_gt`: `∑ₘ χ_s,nm g_s(m) = ℓ(s)g_s(n) - β[1 + (U/β)𝒯_k(s)]f_s(n)g_s(n)`
  with `ℓ(s) = U ∑ₘ f_s(m)g_s(m)`, the identity above with `ε = -1`;
* `Params.hasSum_chiS_mul_gt_of`: where `1 + (U/β)𝒯_k(s) = 0`, `χ_s g_s = ℓ(s)g_s`.
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- `g_s(n) = 1/(zₙ² - u²)`. -/
noncomputable def gs (u : ℂ) (n : ℤ) : ℂ := 1 / (A.z k n ^ 2 - u ^ 2)

/-- `𝒯(u) = ∑ₙ g_s(n)`: `(S_k(u) - S_k(-u))/(2u)`, and `S_k(0)² - β²/4` at `u = 0`. -/
noncomputable def Tg (u : ℂ) : ℂ :=
  if u = 0 then A.S k 0 ^ 2 - A.β ^ 2 / 4 else (A.S k u - A.S k (-u)) / (2 * u)

theorem z_sq_sub_ne_zero {u : ℂ} (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) (n : ℤ) :
    A.z k n ^ 2 - u ^ 2 ≠ 0 := by
  rw [show A.z k n ^ 2 - u ^ 2 = (A.z k n - u) * (A.z k n - -u) by ring]
  exact mul_ne_zero (sub_ne_zero.mpr (hu n)) (sub_ne_zero.mpr (hu' n))

/-- `∑ₙ g_s(n) = 𝒯(u)` for `±u` off the lattice. -/
theorem hasSum_g {u : ℂ} (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) :
    HasSum (A.gs k u) (A.Tg k u) := by
  by_cases h0 : u = 0
  · subst h0
    rw [Tg, ite_eq_left rfl]
    convert A.hasSum_sq hu using 1
    funext n
    rw [gs]
    ring
  · rw [Tg, ite_eq_right h0]
    convert (A.hasSum_sub_inv k hu hu').div_const (2 * u) using 1
    funext n
    have h₁ := sub_ne_zero.mpr (hu n)
    have h₂ : A.z k n + u ≠ 0 := fun h ↦ hu' n (by linear_combination h)
    rw [gs, sub_neg_eq_add, show A.z k n ^ 2 - u ^ 2 = (A.z k n - u) * (A.z k n + u) by ring]
    field_simp
    ring

/-- The pointwise form of the divided differences, diagonal included:
`χ_r,nm g_s(m) = g_s(n)[εU(f_r(zₙ) - f_r(zₘ))g_s(m) + χ_r,nm]`. -/
theorem chiR_mul_gs (hε : ε = 1 ∨ ε = -1) {u : ℂ} (hu : ∀ m, A.z k m ≠ u)
    (hu' : ∀ m, A.z k m ≠ -u) (n m : ℤ) :
    A.chiR k ε n m * A.gs k u m =
      A.gs k u n * (ε * A.U * (A.fR k ε (A.z k n) - A.fR k ε (A.z k m)) * A.gs k u m +
        A.chiR k ε n m) := by
  have hn := A.z_sq_sub_ne_zero k hu hu' n
  have hm := A.z_sq_sub_ne_zero k hu hu' m
  by_cases hnm : n = m
  · subst hnm
    rw [sub_self, mul_zero, zero_mul, zero_add, gs]
    field_simp
  · have hd := A.chiR_divided_difference k ε hε hnm
    rw [← hd, gs, gs]
    field_simp
    ring

theorem U_ne_zero (ha : (A.a : ℂ) ≠ 0) : (A.U : ℂ) ≠ 0 := by
  rw [show (A.U : ℂ) = 2 * A.a by rw [a]; push_cast; ring]
  exact mul_ne_zero two_ne_zero ha

/-- `∑ₘ χ_r,nm g_s(m)` converges: `χ_r,n·` absolutely (Ward identity), and `g_s` is bounded. -/
theorem summable_chiR_mul_gs (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) {u : ℂ}
    (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) (n : ℤ) :
    Summable fun m ↦ A.chiR k ε n m * A.gs k u m := by
  have hgn : Summable fun m ↦ ‖A.gs k u m‖ :=
    summable_norm_iff.mpr (A.hasSum_g k hu hu').summable
  refine Summable.of_norm_bounded ((summable_norm_iff.mpr
    (A.hasSum_chiR k ε hε ha hμ n).summable).mul_right (∑' j, ‖A.gs k u j‖)) fun m ↦ ?_
  rw [norm_mul]
  exact mul_le_mul_of_nonneg_left (hgn.le_tsum m fun j _ ↦ norm_nonneg _) (norm_nonneg _)

/-- `∑ₘ f_r(zₘ)g_s(m)` converges, by `chiR_mul_gs`. -/
theorem summable_fR_mul_gs (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) {u : ℂ}
    (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) :
    Summable fun m ↦ A.fR k ε (A.z k m) * A.gs k u m := by
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  have hU0 := A.U_ne_zero ha
  have hG : A.gs k u 0 ≠ 0 := by rw [gs]; exact one_div_ne_zero (A.z_sq_sub_ne_zero k hu hu' 0)
  have h := (((A.hasSum_g k hu hu').summable.mul_left (ε * A.U * A.fR k ε (A.z k 0))).add
    (A.hasSum_chiR k ε hε ha hμ 0).summable).sub
      ((A.summable_chiR_mul_gs k ε hε ha hμ hu hu' 0).div_const (A.gs k u 0))
  refine (h.div_const (ε * A.U)).congr fun m ↦ ?_
  have := A.chiR_mul_gs k ε hε hu hu' 0 m
  field_simp
  linear_combination -this

/-- **The spectral identity**: for `U ≠ 0`, `μ ≠ 0` and `±u` off the lattice,
`∑ₘ χ_r,nm g_s(m) = ℓ(s)g_s(n) - β[1 - (εU/β)𝒯(s)]f_r(zₙ)g_s(n)`, with
`ℓ(s) = -εU ∑ₘ f_r(zₘ)g_s(m)` (`summable_fR_mul_gs`). -/
theorem hasSum_chiR_mul_g (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) {u : ℂ}
    (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) (n : ℤ) :
    HasSum (fun m ↦ A.chiR k ε n m * A.gs k u m)
      ((-(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) * A.gs k u n -
        A.β * (1 - ε * A.U / A.β * A.Tg k u) * A.fR k ε (A.z k n) * A.gs k u n) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have h := (((A.hasSum_g k hu hu').mul_left (ε * A.U * A.fR k ε (A.z k n))).sub
    ((A.summable_fR_mul_gs k ε hε ha hμ hu hu').hasSum.mul_left (ε * A.U))).add
      (A.hasSum_chiR k ε hε ha hμ n)
  convert h.mul_left (A.gs k u n) using 1
  · funext m
    rw [A.chiR_mul_gs k ε hε hu hu' n m]
    ring
  · field_simp
    ring

/-- **Eigenvectors**: where `1 - (εU/β)𝒯(s) = 0`, `χ_r g_s = ℓ(s)g_s`. -/
theorem hasSum_chiR_mul_g_of (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) {u : ℂ}
    (hu : ∀ m, A.z k m ≠ u) (hu' : ∀ m, A.z k m ≠ -u) (hT : 1 - ε * A.U / A.β * A.Tg k u = 0)
    (n : ℤ) :
    HasSum (fun m ↦ A.chiR k ε n m * A.gs k u m)
      ((-(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) * A.gs k u n) := by
  simpa [hT] using A.hasSum_chiR_mul_g k ε hε ha hμ hu hu' n

/-! ### The singlet

`g_s(n) = 1/(tₙ - s)` and `f_s` are mirror-even, and `chiS_divided_difference` holds on the mirror
too (both sides vanish), so the identity is that of the particle–hole channels with `ε = -1`, `zₙ²`
replaced by `tₙ` and `𝒯(s) = 𝒯_k(s)` of Lemma "particle–particle sum". -/

/-- `g_s(n) = 1/(tₙ - s)`, mirror-even. -/
noncomputable def gt (s : ℂ) (n : ℤ) : ℂ := 1 / (A.t k n - s)

/-- The pointwise form of the singlet divided differences, diagonal and mirror included:
`χ_s,nm g_s(m) = g_s(n)[-U(f_s(n) - f_s(m))g_s(m) + χ_s,nm]`. -/
theorem chiS_mul_gt (hb : A.b k ≠ 0) {s : ℂ} (hs : ∀ m, A.t k m ≠ s) (n m : ℤ) :
    A.chiS k n m * A.gt k s m =
      A.gt k s n * (-A.U * (A.fS k n - A.fS k m) * A.gt k s m + A.chiS k n m) := by
  have hn := sub_ne_zero.mpr (hs n)
  have hm := sub_ne_zero.mpr (hs m)
  by_cases hnm : n = m
  · subst hnm
    rw [sub_self, mul_zero, zero_mul, zero_add, mul_comm]
  · rw [← A.chiS_divided_difference k hb hnm, gt, gt]
    field_simp
    ring

/-- `∑ₘ χ_s,nm g_s(m)` converges: `χ_s,n·` absolutely (Ward identity), and `g_s` is bounded. -/
theorem summable_chiS_mul_gt (hb : A.b k ≠ 0) {s ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 - s)
    (hs : ∀ m, A.t k m ≠ s) (n : ℤ) :
    Summable fun m ↦ A.chiS k n m * A.gt k s m := by
  have hgn : Summable fun m ↦ ‖A.gt k s m‖ :=
    summable_norm_iff.mpr (A.hasSum_Tpp hξ hs).summable
  refine Summable.of_norm_bounded ((summable_norm_iff.mpr
    (A.hasSum_chiS k hb n).summable).mul_right (∑' j, ‖A.gt k s j‖)) fun m ↦ ?_
  rw [norm_mul]
  exact mul_le_mul_of_nonneg_left (hgn.le_tsum m fun j _ ↦ norm_nonneg _) (norm_nonneg _)

/-- `∑ₘ Uf_s(m)g_s(m)` converges, by `chiS_mul_gt`. -/
theorem summable_fS_mul_gt (hb : A.b k ≠ 0) {s ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 - s)
    (hs : ∀ m, A.t k m ≠ s) :
    Summable fun m ↦ A.U * (A.fS k m * A.gt k s m) := by
  have h := ((A.summable_chiS_mul_gt k hb hξ hs 0).div_const (A.gt k s 0)).add
    (((A.hasSum_Tpp hξ hs).summable.mul_left (A.U * A.fS k 0)).sub
      (A.hasSum_chiS k hb 0).summable)
  refine h.congr fun m ↦ ?_
  have := A.chiS_mul_gt k hb hs 0 m
  have h₀ := sub_ne_zero.mpr (hs 0)
  have hm := sub_ne_zero.mpr (hs m)
  simp only [gt] at this ⊢
  field_simp at this ⊢
  linear_combination this

/-- **The spectral identity, singlet**: for `b ≠ 0` and `s` off the lattice `{tₙ}`, with
`ξ² = b²/4 - s`, `∑ₘ χ_s,nm g_s(m) = ℓ(s)g_s(n) - β[1 + (U/β)𝒯_k(s)]f_s(n)g_s(n)`, with
`ℓ(s) = U ∑ₘ f_s(m)g_s(m)` (`summable_fS_mul_gt`). -/
theorem hasSum_chiS_mul_gt (hb : A.b k ≠ 0) {s ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 - s)
    (hs : ∀ m, A.t k m ≠ s) (n : ℤ) :
    HasSum (fun m ↦ A.chiS k n m * A.gt k s m)
      ((A.U * ∑' m, A.fS k m * A.gt k s m) * A.gt k s n -
        A.β * (1 + A.U / A.β * A.Tpp k ξ) * A.fS k n * A.gt k s n) := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have h := (((A.hasSum_Tpp hξ hs).mul_left (-(A.U * A.fS k n))).add
    (A.summable_fS_mul_gt k hb hξ hs).hasSum).add (A.hasSum_chiS k hb n)
  convert h.mul_left (A.gt k s n) using 1
  · funext m
    rw [A.chiS_mul_gt k hb hs n m, gt, gt]
    ring
  · rw [tsum_mul_left]
    field_simp
    ring

/-- **Eigenvectors, singlet**: where `1 + (U/β)𝒯_k(s) = 0`, `χ_s g_s = ℓ(s)g_s`. -/
theorem hasSum_chiS_mul_gt_of (hb : A.b k ≠ 0) {s ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 - s)
    (hs : ∀ m, A.t k m ≠ s) (hT : 1 + A.U / A.β * A.Tpp k ξ = 0) (n : ℤ) :
    HasSum (fun m ↦ A.chiS k n m * A.gt k s m)
      ((A.U * ∑' m, A.fS k m * A.gt k s m) * A.gt k s n) := by
  simpa [hT] using A.hasSum_chiS_mul_gt k hb hξ hs n

end Params

end HubbardAtom
