import HubbardAtomLean.GammaBoxes

/-!
# Continuity of the vertex in the parameters

For the density and magnetic channels, `Γ_r` of `GammaOperator.lean` is continuous in
`(U, β, μ)` wherever `μ ≠ 0` and `1 + 𝒦` is invertible.

The entries of `𝒦` are continuous in the parameters, and near a point `|𝒦_nm|²` has a summable
bound `C|zₙ|⁻²(|zₘ|⁻² + δₙₘ)` uniform in the parameters. So `𝒦` is continuous in the
Hilbert–Schmidt norm (dominated convergence), hence in the operator norm, and inversion is
continuous at units.

* `Params.norm_z_pow_div_Pph_le`: `|zₙ⁴/P(zₙ)| ≤ (1 + |μ|β/π + |k|)⁴`, from `|xⱼ ± a| ≥ |νⱼ|`;
* `Params.Kr_eq_off`, `Params.Kr_eq_diag`: `𝒦` through `tₙ = 1/zₙ`, `zₙ⁴/P(zₙ)` and polynomials in
  `tₙ` (`n2t`, `n0t`, `r3t`, `dgt`);
* `Params.continuousAt_Kr`, `eventually_norm_Kr_sq_le`, `tendsto_tsum_norm_Kr_sub_sq`: along any
  path `fam U β μ` of parameters continuous at `x₀` with `μ(x₀) ≠ 0`, the entries are continuous
  and dominated, so `∑ₙₘ |𝒦_nm(x) - 𝒦_nm(x₀)|² → 0`;
* `Params.continuousAt_gammaOp`: `Γ_r,nm` (`gammaOpμ`) is continuous at `x₀` if `1 + 𝒦` is
  invertible there, in particular on the domain of Theorem "density and magnetic"
  (`continuousAt_gammaOp_of_DMGeneric`).
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-! ### Bounds for one set of parameters -/

/-- `|zₙ| ≤ L|xⱼ + r|` for `j = n, n + k` and real `r`, with `L = 1 + |μ|β/π + |k|`: both are
bounded by `|νⱼ|`, and `|νⱼ| ≥ π/β`. -/
theorem norm_z_le_mul {n j : ℤ} (hj : j = n ∨ j = n + k) (r : ℝ) :
    ‖A.z k n‖ ≤ (1 + |A.μ| * A.β / Real.pi + |(k : ℝ)|) * ‖A.x j + r‖ := by
  have hβ := A.β_pos
  have hπ := Real.pi_pos
  have hx : |A.ν j| ≤ ‖A.x j + r‖ := by
    have h := abs_im_le_norm (A.x j + r)
    simpa [x] using h
  have hz : ‖A.z k n‖ ≤ |A.μ| + |((2 * n + 1 + k) * Real.pi / A.β : ℝ)| := by
    refine (norm_le_abs_re_add_abs_im _).trans_eq ?_
    simp [A.z_eq]
  -- `|2j + 1| ≥ 1` and `|2n + 1 + k| ≤ |2j + 1| + |k|`
  have h1 : (1 : ℝ) ≤ |(2 * j + 1 : ℝ)| := by
    have : (1 : ℤ) ≤ |2 * j + 1| := by
      rcases abs_cases (2 * j + 1) with ⟨h, _⟩ | ⟨h, _⟩ <;> omega
    exact_mod_cast this
  have h2 : |(2 * n + 1 + k : ℝ)| ≤ |(2 * j + 1 : ℝ)| + |(k : ℝ)| := by
    rcases hj with rfl | rfl
    · exact abs_add_le _ _
    · push_cast
      calc |(2 * n + 1 + k : ℝ)| = |(2 * (n + k) + 1 : ℝ) + -k| := by ring_nf
        _ ≤ _ := (abs_add_le _ _).trans_eq (by rw [abs_neg])
  have hν : |A.ν j| = |(2 * j + 1 : ℝ)| * (Real.pi / A.β) := by
    rw [ν, mul_div_assoc, abs_mul, abs_of_pos (div_pos hπ hβ)]
  rw [abs_div, abs_mul, abs_of_pos hπ, abs_of_pos hβ] at hz
  refine hz.trans ((?_ : _ ≤ (1 + |A.μ| * A.β / Real.pi + |(k : ℝ)|) * |A.ν j|).trans
    (mul_le_mul_of_nonneg_left hx (by positivity)))
  rw [hν]
  have hc : 0 < Real.pi / A.β := div_pos hπ hβ
  have hk := abs_nonneg (k : ℝ)
  have hμ := abs_nonneg A.μ
  calc |A.μ| + |(2 * n + 1 + k : ℝ)| * Real.pi / A.β
      = |A.μ| + |(2 * n + 1 + k : ℝ)| * (Real.pi / A.β) := by ring
    _ ≤ |A.μ| * |(2 * j + 1 : ℝ)| + (|(2 * j + 1 : ℝ)| + |(k : ℝ)| * |(2 * j + 1 : ℝ)|) *
        (Real.pi / A.β) := by
      gcongr
      · nlinarith
      · nlinarith
    _ = _ := by field_simp; ring

/-- **`|zₙ⁴/P(zₙ)| ≤ (1 + |μ|β/π + |k|)⁴`**, for all parameters. -/
theorem norm_z_pow_div_Pph_le (n : ℤ) :
    ‖A.z k n ^ 4 / A.Pph k (A.z k n)‖ ≤ (1 + |A.μ| * A.β / Real.pi + |(k : ℝ)|) ^ 4 := by
  set L := 1 + |A.μ| * A.β / Real.pi + |(k : ℝ)|
  have hP := A.Pph_z_ne_zero k n
  rw [norm_div, norm_pow, div_le_iff₀ (norm_pos_iff.mpr hP), Pph, A.Qp_z, A.Qm_z]
  have h₁ := A.norm_z_le_mul k (n := n) (Or.inl rfl) A.a
  have h₂ := A.norm_z_le_mul k (n := n) (Or.inr rfl) A.a
  have h₃ := A.norm_z_le_mul k (n := n) (Or.inl rfl) (-A.a)
  have h₄ := A.norm_z_le_mul k (n := n) (Or.inr rfl) (-A.a)
  push_cast at h₃ h₄
  rw [← sub_eq_add_neg] at h₃ h₄
  simp only [norm_mul]
  have hL : 0 ≤ L := add_nonneg (add_nonneg zero_le_one (div_nonneg
    (mul_nonneg (abs_nonneg _) A.β_pos.le) Real.pi_pos.le)) (abs_nonneg _)
  have h0 : ∀ y : ℂ, 0 ≤ L * ‖y‖ := fun y ↦ mul_nonneg hL (norm_nonneg y)
  calc ‖A.z k n‖ ^ 4 = ‖A.z k n‖ * ‖A.z k n‖ * (‖A.z k n‖ * ‖A.z k n‖) := by ring
    _ ≤ L * ‖A.x n + A.a‖ * (L * ‖A.x (n + k) + A.a‖) *
        (L * ‖A.x n - A.a‖ * (L * ‖A.x (n + k) - A.a‖)) :=
      mul_le_mul (mul_le_mul h₁ h₂ (norm_nonneg _) (h0 _)) (mul_le_mul h₃ h₄ (norm_nonneg _) (h0 _))
        (by positivity) (mul_nonneg (h0 _) (h0 _))
    _ = _ := by ring

/-- `c|zₙ(B)|² ≤ |zₙ(A)|²` if `cμ_B² ≤ μ_A²` and `c/β_B² ≤ 1/β_A²`: `|zₙ|² = μ² + (sπ/β)²`. -/
theorem norm_z_sq_ge (B : Params) {c : ℝ} (hμ : c * B.μ ^ 2 ≤ A.μ ^ 2)
    (hβ : c / B.β ^ 2 ≤ 1 / A.β ^ 2) (n : ℤ) : c * ‖B.z k n‖ ^ 2 ≤ ‖A.z k n‖ ^ 2 := by
  have hn : ∀ C : Params, ‖C.z k n‖ ^ 2 =
      C.μ ^ 2 + ((2 * n + 1 + k) * Real.pi) ^ 2 * (1 / C.β ^ 2) := fun C ↦ by
    rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply, C.z_eq]
    simp only [add_re, ofReal_re, mul_re, ofReal_im, I_re, I_im, add_im, mul_im]
    field_simp
    ring
  rw [hn, hn, mul_add]
  have hs := sq_nonneg ((2 * n + 1 + k) * Real.pi)
  have h := mul_le_mul_of_nonneg_left hβ hs
  have he : c * (((2 * n + 1 + k) * Real.pi) ^ 2 * (1 / B.β ^ 2)) =
      ((2 * n + 1 + k) * Real.pi) ^ 2 * (c / B.β ^ 2) := by ring
  linarith

/-! ### `𝒦` through `1/zₙ` -/

/-- `N₂(v)/v²` as a polynomial in `t = 1/v`. -/
noncomputable def n2t (t : ℂ) : ℂ :=
  -2 * ε * A.a * (1 + 2 * A.a * A.ρ * t + A.q₀ k * t ^ 2) +
    kron k 0 * A.ζ ε * (1 - A.a ^ 2 * t ^ 2)

/-- The constant coefficient `c₂` of `N₀(v) = -λρv³ + c₂v² + c₁v + c₀`. -/
noncomputable def coefN₂ : ℂ := -2 * ε * A.a * A.q₀ k - kron k 0 * A.ζ ε * A.a ^ 2

/-- The coefficient `c₀` of `N₀(v) = -λρv³ + c₂v² + c₁v + c₀`. -/
noncomputable def coefN₀ : ℂ :=
  (2 * A.q₀ k - 4 * A.a ^ 2) * A.coefN₂ k ε - (kron k 0 * A.ζ ε - 2 * ε * A.a) * A.q₀ k ^ 2

/-- `N₀(v)/v³` as a polynomial in `t = 1/v`. -/
noncomputable def n0t (t : ℂ) : ℂ :=
  -(A.lam ε * A.ρ) + A.coefN₂ k ε * t - (2 * A.q₀ k - 4 * A.a ^ 2) * (A.lam ε * A.ρ) * t ^ 2 +
    A.coefN₀ k ε * t ^ 3

/-- `[λρP(v) + vN₀(v)]/v³` as a polynomial in `t = 1/v`. -/
noncomputable def r3t (t : ℂ) : ℂ :=
  A.coefN₂ k ε + A.coefN₀ k ε * t ^ 2 + A.lam ε * A.ρ * A.q₀ k ^ 2 * t ^ 3

/-- `β[P(z) - z²H(z)]/z³ + λρ/(2z²)` as a polynomial in `t = 1/z`. -/
noncomputable def dgt (t : ℂ) : ℂ :=
  A.β * (-2 * A.a * A.ρ + (A.q₀ k - 4 * A.a ^ 2 - 4 * ε * A.a ^ 2 * A.p₁) * t +
    A.q₀ k ^ 2 * t ^ 3) + A.lam ε * A.ρ / 2 * t ^ 2

/-- **Off the diagonal**, `𝒦_nm = -(1/β)(zₙ⁴/P(zₙ))(zₘ⁴/P(zₘ))Φ tₙtₘ` with
`Φ = n2t(tₘ) + [n0t(tₘ) + r3t(tₘ)tₙ](tₙ - 1/(zₙ + zₘ))`, `tⱼ = 1/zⱼ` (Proposition "structure"). -/
theorem Kr_eq_off (hμ : A.μ ≠ 0) {n m : ℤ} (hnm : n ≠ m) :
    A.Kr k ε n m = -(1 / A.β) * (A.z k n ^ 4 / A.Pph k (A.z k n)) *
      (A.z k m ^ 4 / A.Pph k (A.z k m)) *
      (A.n2t k ε (A.z k m)⁻¹ + (A.n0t k ε (A.z k m)⁻¹ + A.r3t k ε (A.z k m)⁻¹ * (A.z k n)⁻¹) *
        ((A.z k n)⁻¹ - (A.z k n + A.z k m)⁻¹)) * (A.z k n)⁻¹ * (A.z k m)⁻¹ := by
  have hz := A.z_ne_zero k hμ n
  have hv := A.z_ne_zero k hμ m
  have hzv := A.z_add_z_ne_zero k hμ n m
  have hPz := A.Pph_z_ne_zero k n
  have hPv := A.Pph_z_ne_zero k m
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hχ : A.chiR k ε n m = (A.lam ε * A.ρ / (A.z k n + A.z k m) +
      A.Nph k ε (A.z k n) (A.z k m) / A.Pph k (A.z k m)) / A.Pph k (A.z k n) := by
    rw [eq_div_iff hPz, mul_comm, A.Pph_mul_chiR k ε hμ, kron_of_ne hnm]
    ring
  rw [Kr, kron_of_ne hnm, sub_zero, hχ]
  simp only [Nph, N₀, N₂, n2t, n0t, r3t, coefN₂, coefN₀, lam]
  have eP := A.Pph_eq k (A.z k n)
  have eQ := A.Pph_eq k (A.z k m)
  generalize A.Pph k (A.z k n) = Pz at hPz eP ⊢
  generalize A.Pph k (A.z k m) = Pv at hPv eQ ⊢
  field_simp
  subst eP eQ
  ring

/-- **On the diagonal**, `𝒦_nn = -(1/β)(zₙ⁴/P(zₙ))[dgt(tₙ) + (zₙ⁴/P(zₙ))(tₙ²n0t(tₙ) +
tₙn2t(tₙ))]tₙ`. -/
theorem Kr_eq_diag (hμ : A.μ ≠ 0) (n : ℤ) :
    A.Kr k ε n n = -(1 / A.β) * (A.z k n ^ 4 / A.Pph k (A.z k n)) *
      (A.dgt k ε (A.z k n)⁻¹ + A.z k n ^ 4 / A.Pph k (A.z k n) *
        ((A.z k n)⁻¹ ^ 2 * A.n0t k ε (A.z k n)⁻¹ + (A.z k n)⁻¹ * A.n2t k ε (A.z k n)⁻¹)) *
      (A.z k n)⁻¹ := by
  have hz := A.z_ne_zero k hμ n
  have hPz := A.Pph_z_ne_zero k n
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have hχ : A.chiR k ε n n = (-A.β * A.Hph k ε (A.z k n) + A.lam ε * A.ρ / (2 * A.z k n) +
      A.Nph k ε (A.z k n) (A.z k n) / A.Pph k (A.z k n)) / A.Pph k (A.z k n) := by
    rw [eq_div_iff hPz, mul_comm, A.Pph_mul_chiR k ε hμ, kron_self, two_mul]
    ring
  rw [Kr, kron_self, hχ]
  simp only [Nph, N₀, N₂, n2t, n0t, dgt, coefN₂, coefN₀, Hph, lam]
  have eP := A.Pph_eq k (A.z k n)
  generalize A.Pph k (A.z k n) = Pz at hPz eP ⊢
  field_simp
  subst eP
  ring

/-- `abcd ≤ ABCD` for `0 ≤ a ≤ A`, etc. -/
private theorem mul4_le {a b c d A B C D : ℝ} (h₁ : a ≤ A) (h₂ : b ≤ B) (h₃ : c ≤ C) (h₄ : d ≤ D)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) : a * b * c * d ≤ A * B * C * D :=
  mul_le_mul (mul_le_mul (mul_le_mul h₁ h₂ hb (ha.trans h₁)) h₃ hc
    (mul_nonneg (ha.trans h₁) (hb.trans h₂))) h₄ hd
    (mul_nonneg (mul_nonneg (ha.trans h₁) (hb.trans h₂)) (hc.trans h₃))

/-- `Γ_r,nm` where `μ ≠ 0`, and `0` at `μ = 0`. -/
noncomputable def gammaOpμ (n m : ℤ) : ℂ := if h : A.μ ≠ 0 then A.gammaOp k ε h n m else 0

/-! ### Along a path of parameters -/

section Family

variable {X : Type*} [TopologicalSpace X] (U β μ : X → ℝ) (hβ : ∀ x, 0 < β x)

/-- The parameters `(U(x), β(x), μ(x))`. -/
def fam (x : X) : Params := ⟨U x, β x, μ x, hβ x⟩

variable {U β μ hβ} {x₀ : X}

/-- A function of `(x, t)` continuous at `{x₀} × {|t| ≤ R}` is bounded near `x₀`, uniformly for
`|t| ≤ R` (compactness of the closed ball). -/
theorem exists_eventually_norm_le {q : X → ℂ → ℂ} (R : ℝ)
    (hq : ∀ t, ContinuousAt (fun p : X × ℂ ↦ q p.1 p.2) (x₀, t)) :
    ∃ C, ∀ᶠ x in 𝓝 x₀, ∀ t, ‖t‖ ≤ R → ‖q x t‖ ≤ C := by
  have hK := isCompact_closedBall (0 : ℂ) R
  have hc : ContinuousOn (q x₀) (Metric.closedBall 0 R) := fun t _ ↦
    ((hq t).comp (continuous_const.prodMk continuous_id).continuousAt).continuousWithinAt
  obtain ⟨C, hC⟩ := hK.exists_bound_of_continuousOn hc
  refine ⟨C + 1, ?_⟩
  have h := hK.eventually_forall_of_forall_eventually (P := fun x t ↦ ‖q x t‖ < C + 1)
    fun t ht ↦ (hq t).norm.eventually (gt_mem_nhds ((hC t ht).trans_lt (lt_add_one C)))
  exact h.mono fun x hx t ht ↦ (hx t (mem_closedBall_zero_iff.mpr ht)).le

section Cont

variable (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)
include hU hb hm

/-- The polynomial `N₂(v)/v²` in `t` is jointly continuous in the parameters and `t`. -/
theorem continuousAt_n2t (t : ℂ) :
    ContinuousAt (fun p : X × ℂ ↦ (fam U β μ hβ p.1).n2t k ε p.2) (x₀, t) := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [fam, n2t, a, ρ, q₀, w, ω, ζ, p₀, p₁, p₂, Z, boltz₀, boltz₁, boltz₂]
  fun_prop (disch := first | positivity | assumption)

/-- The polynomial `N₀(v)/v³` in `t` is jointly continuous in the parameters and `t`. -/
theorem continuousAt_n0t (t : ℂ) :
    ContinuousAt (fun p : X × ℂ ↦ (fam U β μ hβ p.1).n0t k ε p.2) (x₀, t) := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [fam, n0t, coefN₂, coefN₀, lam, a, ρ, q₀, w, ω, ζ, p₀, p₁, p₂, Z, boltz₀, boltz₁,
    boltz₂]
  fun_prop (disch := first | positivity | assumption)

/-- The polynomial `[λρP(v) + vN₀(v)]/v³` in `t` is jointly continuous in the parameters and `t`. -/
theorem continuousAt_r3t (t : ℂ) :
    ContinuousAt (fun p : X × ℂ ↦ (fam U β μ hβ p.1).r3t k ε p.2) (x₀, t) := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [fam, r3t, coefN₂, coefN₀, lam, a, ρ, q₀, w, ω, ζ, p₀, p₁, p₂, Z, boltz₀, boltz₁,
    boltz₂]
  fun_prop (disch := first | positivity | assumption)

/-- The polynomial `dgt` in `t` is jointly continuous in the parameters and `t`. -/
theorem continuousAt_dgt (t : ℂ) :
    ContinuousAt (fun p : X × ℂ ↦ (fam U β μ hβ p.1).dgt k ε p.2) (x₀, t) := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [fam, dgt, lam, a, ρ, q₀, w, ω, p₀, p₁, p₂, Z, boltz₀, boltz₁, boltz₂]
  fun_prop (disch := first | positivity | assumption)

omit hU in
/-- `zₙ` is continuous in the parameters. -/
theorem continuousAt_z (n : ℤ) : ContinuousAt (fun x ↦ (fam U β μ hβ x).z k n) x₀ := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [z_eq, fam]
  fun_prop (disch := assumption)

/-- `P(zₙ)` is continuous in the parameters. -/
theorem continuousAt_Pph_z (n : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).Pph k ((fam U β μ hβ x).z k n)) x₀ := by
  have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
  simp only [Pph, Qp, Qm, z_eq, fam, a, w, ω]
  fun_prop (disch := assumption)

omit hU in
/-- A polynomial factor at `tₙ = 1/zₙ` is continuous in the parameters. -/
private theorem continuousAt_comp_inv_z {q : X → ℂ → ℂ} (hμ₀ : μ x₀ ≠ 0)
    (hq : ∀ t, ContinuousAt (fun p : X × ℂ ↦ q p.1 p.2) (x₀, t)) (n : ℤ) :
    ContinuousAt (fun x ↦ q x ((fam U β μ hβ x).z k n)⁻¹) x₀ :=
  (hq ((fam U β μ hβ x₀).z k n)⁻¹).comp (f := fun x ↦ (x, ((fam U β μ hβ x).z k n)⁻¹))
    (continuousAt_id.prodMk ((continuousAt_z (U := U) k hb hm n).inv₀
      ((fam U β μ hβ x₀).z_ne_zero k hμ₀ n)))

/-- **The entries of `𝒦` are continuous** in the parameters at `μ ≠ 0`. -/
theorem continuousAt_Kr (hμ₀ : μ x₀ ≠ 0) (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).Kr k ε n m) x₀ := by
  have hβc : ContinuousAt (fun x ↦ ((fam U β μ hβ x).β : ℂ)) x₀ := by simp only [fam]; fun_prop
  have hβ0 : ((fam U β μ hβ x₀).β : ℂ) ≠ 0 := ofReal_ne_zero.mpr (hβ x₀).ne'
  have hz := continuousAt_z (U := U) k hb hm (hβ := hβ)
  have hP := continuousAt_Pph_z k hU hb hm (hβ := hβ)
  have hz0 := (fam U β μ hβ x₀).z_ne_zero k hμ₀
  have hP0 := (fam U β μ hβ x₀).Pph_z_ne_zero k
  have h2 := continuousAt_comp_inv_z (U := U) (hβ := hβ) k hb hm hμ₀
    (q := fun x t ↦ (fam U β μ hβ x).n2t k ε t) (continuousAt_n2t (hβ := hβ) k ε hU hb hm)
  have h0 := continuousAt_comp_inv_z (U := U) (hβ := hβ) k hb hm hμ₀
    (q := fun x t ↦ (fam U β μ hβ x).n0t k ε t) (continuousAt_n0t (hβ := hβ) k ε hU hb hm)
  have h3 := continuousAt_comp_inv_z (U := U) (hβ := hβ) k hb hm hμ₀
    (q := fun x t ↦ (fam U β μ hβ x).r3t k ε t) (continuousAt_r3t (hβ := hβ) k ε hU hb hm)
  have hd := continuousAt_comp_inv_z (U := U) (hβ := hβ) k hb hm hμ₀
    (q := fun x t ↦ (fam U β μ hβ x).dgt k ε t) (continuousAt_dgt (hβ := hβ) k ε hU hb hm)
  have hev := hm.eventually_ne hμ₀
  by_cases hnm : n = m
  · subst hnm
    have hn2 := h2 n
    have hn0 := h0 n
    have hnd := hd n
    have hzn := hz n
    have hPn := hP n
    have hz0n := hz0 n
    have hP0n := hP0 n
    refine ContinuousAt.congr ?_
      (hev.mono fun x hx ↦ ((fam U β μ hβ x).Kr_eq_diag k ε hx n).symm)
    fun_prop (disch := assumption)
  · have hn2 := h2 m
    have hn0 := h0 m
    have hn3 := h3 m
    have hzn := hz n
    have hzm := hz m
    have hPn := hP n
    have hPm := hP m
    have hz0n := hz0 n
    have hz0m := hz0 m
    have hP0n := hP0 n
    have hP0m := hP0 m
    have hzz := (fam U β μ hβ x₀).z_add_z_ne_zero k hμ₀ n m
    refine ContinuousAt.congr ?_
      (hev.mono fun x hx ↦ ((fam U β μ hβ x).Kr_eq_off k ε hx hnm).symm)
    fun_prop (disch := assumption)

/-- **A uniform Hilbert–Schmidt bound**: near `x₀`, `|𝒦_nm|² ≤ C|zₙ(x₀)|⁻²(|zₘ(x₀)|⁻² + δₙₘ)`,
by `Kr_eq_off`, `Kr_eq_diag` with `|zⱼ⁴/P(zⱼ)| ≤ (1 + |μ|β/π + |k|)⁴`, `|1/zⱼ| ≤ 2/|μ(x₀)|`
and `|zₙ(x)|⁻² ≤ 4|zₙ(x₀)|⁻²`. -/
theorem eventually_norm_Kr_sq_le (hμ₀ : μ x₀ ≠ 0) : ∃ C, ∀ᶠ x in 𝓝 x₀, ∀ n m,
    ‖(fam U β μ hβ x).Kr k ε n m‖ ^ 2 ≤ C * (‖(fam U β μ hβ x₀).z k n‖⁻¹ ^ 2 *
      (‖(fam U β μ hβ x₀).z k m‖⁻¹ ^ 2 + if n = m then 1 else 0)) := by
  have hβ₀ := hβ x₀
  have hμa : 0 < |μ x₀| := abs_pos.mpr hμ₀
  set R := 2 / |μ x₀| with hR
  have hR0 : 0 ≤ R := by positivity
  obtain ⟨C₂, h₂⟩ := exists_eventually_norm_le (q := fun x t ↦ (fam U β μ hβ x).n2t k ε t) R
    (continuousAt_n2t k ε hU hb hm)
  obtain ⟨C₃, h₃⟩ := exists_eventually_norm_le (q := fun x t ↦ (fam U β μ hβ x).n0t k ε t) R
    (continuousAt_n0t k ε hU hb hm)
  obtain ⟨C₄, h₄⟩ := exists_eventually_norm_le (q := fun x t ↦ (fam U β μ hβ x).r3t k ε t) R
    (continuousAt_r3t k ε hU hb hm)
  obtain ⟨Cd, hd⟩ := exists_eventually_norm_le (q := fun x t ↦ (fam U β μ hβ x).dgt k ε t) R
    (continuousAt_dgt k ε hU hb hm)
  have c0 : ∀ {C : ℝ} {q : X → ℂ → ℂ}, (∀ᶠ x in 𝓝 x₀, ∀ t, ‖t‖ ≤ R → ‖q x t‖ ≤ C) → 0 ≤ C :=
    fun h ↦ (norm_nonneg _).trans (h.self_of_nhds 0 (by simpa using hR0))
  have hC₂ := c0 h₂
  have hC₃ := c0 h₃
  have hC₄ := c0 h₄
  have hCd := c0 hd
  set L := 1 + |μ x₀| * β x₀ / Real.pi + |(k : ℝ)| + 1 with hLdef
  have hL : ∀ᶠ x in 𝓝 x₀, 1 + |μ x| * β x / Real.pi + |(k : ℝ)| < L := by
    have : ContinuousAt (fun x ↦ 1 + |μ x| * β x / Real.pi + |(k : ℝ)|) x₀ := by fun_prop
    exact this.eventually (gt_mem_nhds (lt_add_one _))
  have hμe : ∀ᶠ x in 𝓝 x₀, |μ x₀| / 2 < |μ x| :=
    (continuous_abs.continuousAt.comp hm).eventually (lt_mem_nhds (half_lt_self hμa))
  have hβ1 : ∀ᶠ x in 𝓝 x₀, β x₀ / 2 < β x := hb.eventually (lt_mem_nhds (half_lt_self hβ₀))
  have hβ2 : ∀ᶠ x in 𝓝 x₀, β x < 2 * β x₀ := hb.eventually (gt_mem_nhds (by linarith))
  obtain ⟨B, hBd⟩ : ∃ c, c = L ^ 4 := ⟨_, rfl⟩
  have hB0 : 0 ≤ B := hBd ▸ by positivity
  obtain ⟨Φb, hΦd⟩ : ∃ c, c = C₂ + (C₃ + C₄ * R) * (R + R) := ⟨_, rfl⟩
  have hΦ0 : 0 ≤ Φb := hΦd ▸ by positivity
  obtain ⟨Ψb, hΨd⟩ : ∃ c, c = Cd + B * (R ^ 2 * C₃ + R * C₂) := ⟨_, rfl⟩
  have hΨ0 : 0 ≤ Ψb := hΨd ▸ by positivity
  obtain ⟨Coff, hCoff⟩ : ∃ c, c = 2 / β x₀ * B * B * Φb := ⟨_, rfl⟩
  obtain ⟨Cdg, hCdg⟩ : ∃ c, c = 2 / β x₀ * B * Ψb := ⟨_, rfl⟩
  clear_value R L
  refine ⟨16 * Coff ^ 2 + 4 * Cdg ^ 2, ?_⟩
  filter_upwards [h₂, h₃, h₄, hd, hL, hμe, hβ1, hβ2] with x e₂ e₃ e₄ ed eL eμ eβ1 eβ2
  intro n m
  set A := fam U β μ hβ x
  set A₀ := fam U β μ hβ x₀
  have hμx : A.μ ≠ 0 := fun h ↦ by
    have : |μ x| = 0 := by rw [show μ x = A.μ from rfl, h, abs_zero]
    linarith
  have ht : ∀ j, ‖(A.z k j)⁻¹‖ ≤ R := fun j ↦ (A.norm_inv_z_le k hμx j).trans (by
    rw [hR, ← inv_div]
    exact inv_anti₀ (half_pos hμa) eμ.le)
  have hs : ∀ j l, ‖(A.z k j + A.z k l)⁻¹‖ ≤ R := fun j l ↦
    (A.norm_inv_z_add_z_le k hμx j l).trans ((inv_anti₀ (abs_pos.mpr hμx)
      (by linarith [abs_nonneg A.μ])).trans (by
        rw [hR, ← inv_div]
        exact inv_anti₀ (half_pos hμa) eμ.le))
  have hB : ∀ j, ‖A.z k j ^ 4 / A.Pph k (A.z k j)‖ ≤ B := fun j ↦
    (A.norm_z_pow_div_Pph_le k j).trans (hBd ▸ pow_le_pow_left₀ (by
      have := mul_nonneg (abs_nonneg (μ x)) (hβ x).le
      have := div_nonneg this Real.pi_pos.le
      have := abs_nonneg (k : ℝ)
      change 0 ≤ 1 + |μ x| * β x / Real.pi + |(k : ℝ)|
      linarith) eL.le 4)
  have hbβ : ‖-(1 / (A.β : ℂ))‖ ≤ 2 / β x₀ := by
    change ‖-(1 / ((β x : ℝ) : ℂ))‖ ≤ _
    rw [norm_neg, norm_div, norm_one, norm_real, Real.norm_of_nonneg (hβ x).le]
    calc 1 / β x ≤ 1 / (β x₀ / 2) := one_div_le_one_div_of_le (half_pos hβ₀) eβ1.le
      _ = 2 / β x₀ := by rw [one_div_div]
  have hg : ∀ j, ‖(A.z k j)⁻¹‖ ^ 2 ≤ 4 * ‖A₀.z k j‖⁻¹ ^ 2 := fun j ↦ by
    have hz₀ := norm_pos_iff.mpr (A₀.z_ne_zero k hμ₀ j)
    have hμ' : 1 / 4 * A₀.μ ^ 2 ≤ A.μ ^ 2 := by
      change 1 / 4 * μ x₀ ^ 2 ≤ μ x ^ 2
      rw [← sq_abs (μ x₀), ← sq_abs (μ x)]
      nlinarith [abs_nonneg (μ x₀)]
    have hβ' : 1 / 4 / A₀.β ^ 2 ≤ 1 / A.β ^ 2 := by
      change 1 / 4 / β x₀ ^ 2 ≤ 1 / β x ^ 2
      rw [div_div]
      exact one_div_le_one_div_of_le (pow_pos (hβ x) 2) (by nlinarith [hβ x])
    have h := A.norm_z_sq_ge k A₀ hμ' hβ' j
    rw [norm_inv, inv_pow, inv_pow]
    calc (‖A.z k j‖ ^ 2)⁻¹ ≤ (1 / 4 * ‖A₀.z k j‖ ^ 2)⁻¹ := inv_anti₀ (by positivity) h
      _ = 4 * (‖A₀.z k j‖ ^ 2)⁻¹ := by rw [mul_inv, one_div, inv_inv]
  have hg0 : ∀ j, 0 ≤ ‖A₀.z k j‖⁻¹ ^ 2 := fun _ ↦ by positivity
  by_cases hnm : n = m
  · subst hnm
    have hK : ‖A.Kr k ε n n‖ ≤ Cdg * ‖(A.z k n)⁻¹‖ := by
      rw [A.Kr_eq_diag k ε hμx]
      simp only [norm_mul]
      have hΨ : ‖A.dgt k ε (A.z k n)⁻¹ + A.z k n ^ 4 / A.Pph k (A.z k n) *
          ((A.z k n)⁻¹ ^ 2 * A.n0t k ε (A.z k n)⁻¹ + (A.z k n)⁻¹ * A.n2t k ε (A.z k n)⁻¹)‖ ≤
          Ψb := by
        rw [hΨd]
        refine (norm_add_le _ _).trans (add_le_add (ed _ (ht n)) ?_)
        rw [norm_mul]
        refine mul_le_mul (hB n) ((norm_add_le _ _).trans (add_le_add ?_ ?_)) (norm_nonneg _) hB0
        · rw [norm_mul, norm_pow]
          exact mul_le_mul (pow_le_pow_left₀ (norm_nonneg _) (ht n) 2) (e₃ _ (ht n))
            (norm_nonneg _) (by positivity)
        · rw [norm_mul]
          exact mul_le_mul (ht n) (e₂ _ (ht n)) (norm_nonneg _) hR0
      rw [hCdg]
      exact mul4_le hbβ (hB n) hΨ le_rfl (norm_nonneg _) (norm_nonneg _) (norm_nonneg _)
        (norm_nonneg _)
    calc ‖A.Kr k ε n n‖ ^ 2 ≤ (Cdg * ‖(A.z k n)⁻¹‖) ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hK 2
      _ = Cdg ^ 2 * ‖(A.z k n)⁻¹‖ ^ 2 := by ring
      _ ≤ Cdg ^ 2 * (4 * ‖A₀.z k n‖⁻¹ ^ 2) := mul_le_mul_of_nonneg_left (hg n) (sq_nonneg _)
      _ ≤ _ := by
        rw [ite_eq_left rfl]
        nlinarith [mul_nonneg (sq_nonneg Coff) (hg0 n), mul_nonneg (sq_nonneg Cdg)
          (mul_nonneg (hg0 n) (hg0 n)), mul_nonneg (sq_nonneg Coff) (mul_nonneg (hg0 n) (hg0 n))]
  · have hK : ‖A.Kr k ε n m‖ ≤ Coff * ‖(A.z k n)⁻¹‖ * ‖(A.z k m)⁻¹‖ := by
      rw [A.Kr_eq_off k ε hμx hnm]
      simp only [norm_mul]
      have hΦ : ‖A.n2t k ε (A.z k m)⁻¹ + (A.n0t k ε (A.z k m)⁻¹ + A.r3t k ε (A.z k m)⁻¹ *
          (A.z k n)⁻¹) * ((A.z k n)⁻¹ - (A.z k n + A.z k m)⁻¹)‖ ≤ Φb := by
        rw [hΦd]
        refine (norm_add_le _ _).trans (add_le_add (e₂ _ (ht m)) ?_)
        rw [norm_mul]
        refine mul_le_mul ((norm_add_le _ _).trans (add_le_add (e₃ _ (ht m)) ?_))
          ((norm_sub_le _ _).trans (add_le_add (ht n) (hs n m))) (norm_nonneg _)
          (by positivity)
        rw [norm_mul]
        exact mul_le_mul (e₄ _ (ht m)) (ht n) (norm_nonneg _) hC₄
      rw [hCoff]
      exact mul_le_mul (mul_le_mul_of_nonneg_right (mul4_le hbβ (hB n) (hB m) hΦ (norm_nonneg _)
        (norm_nonneg _) (norm_nonneg _) (norm_nonneg _)) (norm_nonneg _)) le_rfl (norm_nonneg _)
        (by positivity)
    calc ‖A.Kr k ε n m‖ ^ 2 ≤ (Coff * ‖(A.z k n)⁻¹‖ * ‖(A.z k m)⁻¹‖) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hK 2
      _ = Coff ^ 2 * (‖(A.z k n)⁻¹‖ ^ 2 * ‖(A.z k m)⁻¹‖ ^ 2) := by ring
      _ ≤ Coff ^ 2 * ((4 * ‖A₀.z k n‖⁻¹ ^ 2) * (4 * ‖A₀.z k m‖⁻¹ ^ 2)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul (hg n) (hg m) (sq_nonneg _) (by positivity))
            (sq_nonneg _)
      _ ≤ _ := by
        rw [ite_eq_right hnm, add_zero]
        nlinarith [mul_nonneg (sq_nonneg Cdg) (mul_nonneg (hg0 n) (hg0 m))]

/-- **`𝒦` is continuous in the Hilbert–Schmidt norm**: `∑ₙₘ |𝒦_nm(x) - 𝒦_nm(x₀)|² → 0`, by
dominated convergence with `eventually_norm_Kr_sq_le`. -/
theorem tendsto_tsum_norm_Kr_sub_sq (hμ₀ : μ x₀ ≠ 0) :
    Tendsto (fun x ↦ ∑' p : ℤ × ℤ,
      ‖(fam U β μ hβ x).Kr k ε p.1 p.2 - (fam U β μ hβ x₀).Kr k ε p.1 p.2‖ ^ 2) (𝓝 x₀) (𝓝 0) := by
  obtain ⟨C, hC⟩ := eventually_norm_Kr_sq_le k ε hU hb hm hμ₀
  have hC₀ := hC.self_of_nhds
  set A₀ := fam U β μ hβ x₀
  have hsum := (A₀.summable_majorant k hμ₀).mul_left (4 * C)
  have h := tendsto_tsum_of_dominated_convergence (f := fun x (p : ℤ × ℤ) ↦
      ‖(fam U β μ hβ x).Kr k ε p.1 p.2 - A₀.Kr k ε p.1 p.2‖ ^ 2)
    (g := fun p ↦ ‖A₀.Kr k ε p.1 p.2 - A₀.Kr k ε p.1 p.2‖ ^ 2) hsum
    (fun p ↦ (((continuousAt_Kr k ε hU hb hm hμ₀ p.1 p.2).sub_const _).norm.pow 2))
    (hC.mono fun x hx p ↦ by
      rw [Real.norm_of_nonneg (sq_nonneg _)]
      have h1 := hx p.1 p.2
      have h2 := hC₀ p.1 p.2
      have h3 := norm_sub_le ((fam U β μ hβ x).Kr k ε p.1 p.2) (A₀.Kr k ε p.1 p.2)
      have h4 : 0 ≤ ‖A₀.z k p.1‖⁻¹ ^ 2 * (‖A₀.z k p.2‖⁻¹ ^ 2 + if p.1 = p.2 then 1 else 0) := by
        split_ifs <;> positivity
      nlinarith [pow_le_pow_left₀ (norm_nonneg _) h3 2,
        sq_nonneg (‖(fam U β μ hβ x).Kr k ε p.1 p.2‖ - ‖A₀.Kr k ε p.1 p.2‖)])
  simpa using h

/-- **`Γ_r` is continuous in the parameters** wherever `μ ≠ 0` and `1 + 𝒦` is invertible: `𝒦` is
continuous in norm (`tendsto_tsum_norm_Kr_sub_sq`, `norm_hsOp_le`), and inversion is continuous at
units. -/
theorem continuousAt_gammaOp (hμ₀ : μ x₀ ≠ 0)
    (hu : IsUnit ((fam U β μ hβ x₀).onePlusK k ε hμ₀)) (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).gammaOpμ k ε n m) x₀ := by
  have hev : ∀ᶠ x in 𝓝 x₀, μ x ≠ 0 := hm.eventually_ne hμ₀
  let T : X → (ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ)) := fun x ↦
    if h : (fam U β μ hβ x).μ ≠ 0 then (fam U β μ hβ x).onePlusK k ε h else 1
  have hTx : ∀ x (h : (fam U β μ hβ x).μ ≠ 0), T x = (fam U β μ hβ x).onePlusK k ε h :=
    fun x h ↦ dite_eq_left h
  have hT : Tendsto T (𝓝 x₀) (𝓝 (T x₀)) := by
    rw [tendsto_iff_norm_sub_tendsto_zero]
    have ht := (tendsto_tsum_norm_Kr_sub_sq k ε hU hb hm hμ₀ (hβ := hβ)).sqrt
    rw [Real.sqrt_zero] at ht
    refine squeeze_zero' (Eventually.of_forall fun _ ↦ norm_nonneg _) ?_ ht
    filter_upwards [hev] with x hx
    have hs := (fam U β μ hβ x).summable_Kr_sq k ε hx
    have hs₀ := (fam U β μ hβ x₀).summable_Kr_sq k ε hμ₀
    have hd : Summable fun p : ℤ × ℤ ↦ ‖(fam U β μ hβ x).Kr k ε p.1 p.2 -
        (fam U β μ hβ x₀).Kr k ε p.1 p.2‖ ^ 2 :=
      ((hs.mul_left 2).add (hs₀.mul_left 2)).of_nonneg_of_le (fun _ ↦ by positivity) fun p ↦ by
        nlinarith [pow_le_pow_left₀ (norm_nonneg _) (norm_sub_le ((fam U β μ hβ x).Kr k ε p.1 p.2)
          ((fam U β μ hβ x₀).Kr k ε p.1 p.2)) 2,
          sq_nonneg (‖(fam U β μ hβ x).Kr k ε p.1 p.2‖ - ‖(fam U β μ hβ x₀).Kr k ε p.1 p.2‖)]
    rw [hTx x hx, hTx x₀ hμ₀]
    unfold onePlusK
    rw [add_sub_add_left_eq_sub,
      hsOp_sub ((fam U β μ hβ x).Kr k ε) hs ((fam U β μ hβ x₀).Kr k ε) hs₀ hd]
    exact norm_hsOp_le (fun n m ↦ (fam U β μ hβ x).Kr k ε n m - (fam U β μ hβ x₀).Kr k ε n m) hd
  have hinv : Tendsto (fun x ↦ Ring.inverse (T x)) (𝓝 x₀) (𝓝 (Ring.inverse (T x₀))) := by
    have hu' : IsUnit (T x₀) := (hTx x₀ hμ₀).symm ▸ hu
    have h := NormedRing.inverse_continuousAt hu'.unit
    rw [hu'.unit_spec] at h
    exact h.tendsto.comp hT
  have he := ((lp.evalCLM ℂ (fun _ : ℤ ↦ ℂ) 2 n).continuous.tendsto _).comp
    (((ContinuousLinearMap.apply ℂ _ (lp.single 2 m (1 : ℂ))).continuous.tendsto _).comp hinv)
  -- the scalar factors
  have hz := continuousAt_z (U := U) k hb hm (hβ := hβ)
  have hP := continuousAt_Pph_z k hU hb hm (hβ := hβ)
  have hβc : ContinuousAt (fun x ↦ ((fam U β μ hβ x).β : ℂ)) x₀ := by fun_prop
  have hb0 : ((fam U β μ hβ x₀).β : ℂ) ≠ 0 := ofReal_ne_zero.mpr (hβ x₀).ne'
  have hD : ∀ j, ContinuousAt (fun x ↦ (fam U β μ hβ x).Dhalf k j) x₀ := fun j ↦ by
    have hzj := hz j
    have hz0 := (fam U β μ hβ x₀).z_ne_zero k hμ₀ j
    simp only [Dhalf]
    fun_prop (disch := assumption)
  have ha : ContinuousAt (fun x ↦ ((fam U β μ hβ x).a : ℂ)) x₀ := by
    simp only [fam, a]
    fun_prop
  have hρ : ContinuousAt (fun x ↦ ((fam U β μ hβ x).ρ : ℂ)) x₀ := by
    simp only [fam, ρ, p₀, p₂, Z, boltz₀, boltz₁, boltz₂]
    fun_prop (disch := positivity)
  have hw : ContinuousAt (fun x ↦ (fam U β μ hβ x).w k) x₀ := by
    have hb' : β x₀ ≠ 0 := (hβ x₀).ne'
    simp only [fam, w, ω]
    fun_prop (disch := assumption)
  have htB : ContinuousAt (fun x ↦ (fam U β μ hβ x).tB k m) x₀ := by
    have hzm := hz m
    have hPm := hP m
    have hP0 := (fam U β μ hβ x₀).Pph_z_ne_zero k m
    simp only [tB, G₂]
    fun_prop (disch := assumption)
  have hlim := ((hβc.tendsto.pow 2).div ((hD n).tendsto.mul (hD m).tendsto) (mul_ne_zero
    ((fam U β μ hβ x₀).Dhalf_ne_zero k hμ₀ n) ((fam U β μ hβ x₀).Dhalf_ne_zero k hμ₀ m))).mul
    (he.sub ((tendsto_const_nhds : Tendsto (fun _ : X ↦ kron n m) (𝓝 x₀) _).div htB.tendsto
      ((fam U β μ hβ x₀).tB_ne_zero k hμ₀ m)))
  refine ContinuousAt.congr (f := fun x ↦ ((fam U β μ hβ x).β : ℂ) ^ 2 /
      ((fam U β μ hβ x).Dhalf k n * (fam U β μ hβ x).Dhalf k m) *
      (Ring.inverse (T x) (lp.single 2 m 1) n - kron n m / (fam U β μ hβ x).tB k m)) hlim ?_
  filter_upwards [hev] with x hx
  have hx' : (fam U β μ hβ x).μ ≠ 0 := hx
  have h0 : Ring.inverse ((fam U β μ hβ x).onePlusK0 k hx') (lp.single 2 m 1) n =
      kron n m / (fam U β μ hβ x).tB k m := by
    rw [inverse_apply_of_eq (show IsUnit ((fam U β μ hβ x).onePlusK0 k hx') from
      (fam U β μ hβ x).isUnit_one_add_K0 k hx') (onePlusK0_single hx' m), lp.coeFn_smul,
      Pi.smul_apply, smul_eq_mul, single_apply_eq_kron, div_eq_inv_mul]
  rw [gammaOpμ, dite_eq_left hx', gammaOp, _root_.sub_apply, lp.coeFn_sub, Pi.sub_apply, h0,
    hTx x hx']

/-- On the domain of Theorem "density and magnetic" (generic case), `Γ_r` is continuous in the
parameters. -/
theorem continuousAt_gammaOp_of_DMGeneric {u₁ u₂ : ℂ}
    (h : (fam U β μ hβ x₀).DMGeneric k ε u₁ u₂) (n m : ℤ) :
    ContinuousAt (fun x ↦ (fam U β μ hβ x).gammaOpμ k ε n m) x₀ :=
  continuousAt_gammaOp k ε hU hb hm h.μ_ne h.isUnit_onePlusK n m

end Cont

end Family

end Params

end HubbardAtom
