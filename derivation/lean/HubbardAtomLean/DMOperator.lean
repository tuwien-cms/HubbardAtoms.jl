import HubbardAtomLean.HalfFillingBounds
import HubbardAtomLean.HalfFillingLZ
import HubbardAtomLean.SingletOperator

/-!
# The density and magnetic vertex as an operator, for every `μ`

Items 6(b)–(c) and the boxes of 6(d) for the density and magnetic channels, by `ChiOperator.lean`
applied to `X = χ_r` and `b = B` with the bounds of `HalfFillingBounds.lean`. For every `μ`, on the
lattice `z'ₙ` of `OpLatticeAll.lean` with `𝒟 = diag(-β/z'ₙ²)`, `χ_r = 𝒟^(1/2)(1 + 𝒦)𝒟^(1/2)` with
`𝒦` Hilbert–Schmidt, and `Γ_r = β²(χ_r⁻¹ - B⁻¹)` (Eq. (Gamma-def)):

* `Params.gammaOpR`: `Γ_r = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)`, entrywise;
* `Params.isUnit_onePlusKR_iff`: `1 + 𝒦` is invertible iff `χ_r` has no nonzero null vector `v`
  with `∑ₙ |vₙ/z'ₙ|² < ∞`; `Params.exists_bound_of_nullR`: such null vectors are bounded;
* `Params.gammaOpR_eq_chiInvR`: `Γ_r = β²(χ_r⁻¹ - B⁻¹)` with `χ_r⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`,
  which, where `1 + 𝒦` is invertible, is a two-sided inverse of `χ_r` and the only right inverse
  with columns in the class;
* `Params.gammaOpR_eq_of_isRightInverse`: for every right inverse `Y` of `χ_r` with bounded
  columns, `1 + 𝒦` is invertible and `Γ_r = β²(Y - B⁻¹)`;
* `Params.HFHyp.gammaOpR_eq`, `Params.HFLZ.gammaOpR_eq`: at `μ = 0`, under the assumptions of
  Corollary "half filling", `Γ_r` is the closed form `gammaHF` of Eq. (Gamma-hf), and at a lattice
  zero `z_c² = ŝ` it is `gammaHFlz` of Eq. (Gamma-hf-lz); `Params.DMGeneric.gammaOpR_eq`: for
  `μ ≠ 0`, on the domain of Theorem "density and magnetic", it is `gammaR` of Eq. (Gamma-ph);
* `Params.tendsto_gammaBoxR`: wherever `1 + 𝒦` is invertible, the box matrices `χ_r|_M` are
  invertible for large `M` (`eventually_isUnit_chiBoxR_det`), and the box vertices
  `β²[(χ_r|_M)⁻¹ - B⁻¹]` on `|2n + 1 + k| ≤ 2M + 1` converge to `Γ_r`.
-/

open Complex Filter Topology
open scoped lp

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-! ### The operator and the criterion -/

/-- `1 + 𝒦` for `χ_r` on `ℓ²(ℤ)`. -/
noncomputable def onePlusKR : ℓ²(ℤ, ℂ) →L[ℂ] ℓ²(ℤ, ℂ) :=
  (A.opLatticeAll k).onePlusKX (A.chiR k ε) (A.chiDecay_chiR_all k ε)

/-- **The density and magnetic vertex as an operator**:
`Γ_r,nm = β²/(dₙdₘ) [(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]ₙₘ`, with `Ring.inverse` (zero where `1 + 𝒦` is not
invertible). -/
noncomputable def gammaOpR (n m : ℤ) : ℂ :=
  (A.opLatticeAll k).gammaOpX (A.chiR k ε) (A.Bph k) (A.chiDecay_chiR_all k ε)
    (A.bubbleDecay_Bph k) n m

/-- `χ_r⁻¹ = 𝒟^(-1/2)(1 + 𝒦)⁻¹𝒟^(-1/2)`, entrywise, with `Ring.inverse` as in `gammaOpR`. -/
noncomputable def chiInvR (n m : ℤ) : ℂ :=
  (A.opLatticeAll k).chiInvX (A.chiR k ε) (A.chiDecay_chiR_all k ε) n m

/-- **`Γ_r` exists iff `χ_r` has no null vector**: `1 + 𝒦` is invertible iff `χ_r` has no nonzero
null vector `v` with `∑ₙ |vₙ/z'ₙ|² < ∞`. -/
theorem isUnit_onePlusKR_iff :
    IsUnit (A.onePlusKR k ε) ↔ ∀ v : ℤ → ℂ, Summable (fun n ↦ ‖v n / A.zOp k n‖ ^ 2) →
      (∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) 0) → v = 0 :=
  (A.opLatticeAll k).isUnit_onePlusKX_iff _ _

/-- **Null vectors are bounded**: a null vector `v` of `χ_r` with `∑ₙ |vₙ/z'ₙ|² < ∞` has
`|vₙ| ≤ C`. -/
theorem exists_bound_of_nullR {v : ℤ → ℂ} (hv : Summable fun n ↦ ‖v n / A.zOp k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) 0) : ∃ C, ∀ n, ‖v n‖ ≤ C :=
  (A.opLatticeAll k).exists_bound_of_nullX _ (A.chiDecay_chiR_all k ε) hv hχ

/-! ### `Γ_r = β²(χ_r⁻¹ - B⁻¹)` -/

/-- **`Γ_r = β²(χ_r⁻¹ - B⁻¹)`**, entrywise, for every `μ`. Where `1 + 𝒦` is not invertible,
`χ_r⁻¹ = 0`, so `Γ_r = -β²δₙₘ/B(νₙ)` there. -/
theorem gammaOpR_eq_chiInvR (n m : ℤ) :
    A.gammaOpR k ε n m = A.β ^ 2 * (A.chiInvR k ε n m - kron n m / A.Bph k n) :=
  (A.opLatticeAll k).gammaOpX_eq_chiInvX _ _ _ _ n m

/-- `χ_r` is symmetric, for every `μ`. -/
theorem chiR_symm_all (n m : ℤ) : A.chiR k ε n m = A.chiR k ε m n := by
  by_cases hμ : A.μ = 0
  exacts [A.chiR_symm_hf k ε hμ n m, A.chiR_symm k ε hμ n m]

/-- **`χ_r⁻¹` is a right inverse** where `1 + 𝒦` is invertible. -/
theorem chiInvR_isRightInverse (hK : IsUnit (A.onePlusKR k ε)) :
    IsRightInverse (A.chiR k ε) (A.chiInvR k ε) :=
  (A.opLatticeAll k).chiInvX_isRightInverse _ _ hK

/-- **`χ_r⁻¹` is a left inverse** where `1 + 𝒦` is invertible, as `χ_r` is symmetric. -/
theorem chiInvR_isLeftInverse (hK : IsUnit (A.onePlusKR k ε)) :
    IsLeftInverse (A.chiR k ε) (A.chiInvR k ε) :=
  (A.opLatticeAll k).chiInvX_isLeftInverse _ _ (A.chiR_symm_all k ε) hK

/-- **`χ_r⁻¹` is the only right inverse in the class**: where `1 + 𝒦` is invertible, a `v` with
`∑ₙ |vₙ/z'ₙ|² < ∞` and `∑ₗ χ_r,nl vₗ = δₙₘ` is column `m` of `χ_r⁻¹`. -/
theorem eq_chiInvR_of_hasSum (hK : IsUnit (A.onePlusKR k ε)) {m : ℤ} {v : ℤ → ℂ}
    (hv : Summable fun n ↦ ‖v n / A.zOp k n‖ ^ 2)
    (hχ : ∀ n, HasSum (fun l ↦ A.chiR k ε n l * v l) (kron n m)) (n : ℤ) :
    v n = A.chiInvR k ε n m :=
  (A.opLatticeAll k).eq_chiInvX_of_hasSum _ _ hK hv hχ n

/-! ### The closed forms -/

/-- **`1 + 𝒦` is invertible** if `χ_r` has a right inverse `Y` with bounded columns, as `χ_r` is
symmetric. -/
theorem isUnit_onePlusKR_of_isRightInverse {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiR k ε) Y) (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) :
    IsUnit (A.onePlusKR k ε) :=
  (A.opLatticeAll k).isUnit_onePlusKX_of_isRightInverse _ _ (A.chiR_symm_all k ε) hY hbdd

/-- **`χ_r⁻¹` is every right inverse with bounded columns**. -/
theorem eq_chiInvR_of_isRightInverse {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiR k ε) Y) (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) (n m : ℤ) :
    Y n m = A.chiInvR k ε n m :=
  (A.opLatticeAll k).eq_chiInvX_of_isRightInverse _ _ (A.chiR_symm_all k ε) hY hbdd n m

/-- **The vertex of a right inverse with bounded columns**: `Γ_r = β²(Y - B⁻¹)`. -/
theorem gammaOpR_eq_of_isRightInverse {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiR k ε) Y) (hbdd : ∀ m, ∃ C, ∀ n, ‖Y n m‖ ≤ C) (n m : ℤ) :
    A.gammaOpR k ε n m = A.β ^ 2 * (Y n m - kron n m / A.Bph k n) :=
  (A.opLatticeAll k).gammaOpX_eq_of_isRightInverse _ _ _ _ (A.chiR_symm_all k ε) hY hbdd n m

/-- `ĝ(zₙ) = 1/(zₙ² - ŝ) → 0` along the cofinite filter. -/
theorem tendsto_ghat : Tendsto (fun n ↦ A.ghat k ε (A.z k n)) cofinite (𝓝 0) := by
  have hq : ContinuousAt (fun s ↦ s ^ 2 * (1 - A.shat k ε * s ^ 2)⁻¹) 0 :=
    ContinuousAt.mul (by fun_prop) (ContinuousAt.inv₀ (by fun_prop) (by simp))
  have h := hq.tendsto.comp (A.tendsto_inv_z k)
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_mul] at h
  refine h.congr' ((A.eventually_zOp_eq_ne k).mono fun n ⟨_, _, hz⟩ ↦ ?_)
  simp only [Function.comp_apply, ghat]
  rw [one_div, show 1 - A.shat k ε * (A.z k n)⁻¹ ^ 2 =
    (A.z k n ^ 2 - A.shat k ε) * (A.z k n)⁻¹ ^ 2 by field_simp, mul_inv, inv_pow, inv_inv]
  field_simp

variable {k ε} in
/-- The columns of `invHF` are bounded: off `{m, m̄}` they tend to `2εa/β²`. -/
theorem bdd_invHF (u : ℂ) (m : ℤ) : ∃ C, ∀ n, ‖A.invHF k ε u n m‖ ≤ C := by
  simp only [invHF, add_sub_assoc]
  exact bdd_mirrorBlock_add k (Z := fun n m ↦ 2 * ε * A.a / A.β ^ 2 -
      2 * ε * A.a * A.Pph k u / (A.β * A.Dhat k ε u) * A.ghat k ε (A.z k n) *
        A.ghat k ε (A.z k m))
    (tendsto_const_nhds.sub ((tendsto_const_nhds.mul (A.tendsto_ghat k ε)).mul
      tendsto_const_nhds))

variable {k ε} in
/-- The columns of `invHFlz` are bounded: off `{m, m̄}` they are `Zhf`, which tends to
`2εa/β²`. -/
theorem bdd_invHFlz (c m : ℤ) : ∃ C, ∀ n, ‖A.invHFlz k ε c n m‖ ≤ C := by
  have ho : Tendsto (fun n ↦ orb k c n) cofinite (𝓝 0) :=
    tendsto_const_nhds.congr' (by
      filter_upwards [eventually_cofinite_ne c, eventually_cofinite_ne (mirror k c)] with n h₁ h₂
      simp [orb, h₁, h₂])
  exact bdd_mirrorBlock_add k (Z := A.Zhf k ε c) (by
    unfold Zhf
    exact (tendsto_const_nhds.add (tendsto_const_nhds.mul ((ho.mul tendsto_const_nhds).add
      ((A.tendsto_ghat k ε).mul tendsto_const_nhds)))).add
      ((tendsto_const_nhds.mul ho).mul tendsto_const_nhds))

variable {A k ε}

/-- **`1 + 𝒦` is invertible** under the assumptions of Corollary "half filling". -/
theorem HFHyp.isUnit_onePlusKR {u : ℂ} (h : A.HFHyp k ε u) : IsUnit (A.onePlusKR k ε) :=
  A.isUnit_onePlusKR_of_isRightInverse k ε h.chiR_isRightInverse (A.bdd_invHF u)

/-- **`χ_r⁻¹` is `invHF`** under the assumptions of Corollary "half filling". -/
theorem HFHyp.invHF_eq_chiInvR {u : ℂ} (h : A.HFHyp k ε u) (n m : ℤ) :
    A.invHF k ε u n m = A.chiInvR k ε n m :=
  A.eq_chiInvR_of_isRightInverse k ε h.chiR_isRightInverse (A.bdd_invHF u) n m

/-- **The operator vertex is the closed form** at half filling: under the assumptions of Corollary
"half filling", `Γ_r = β²𝒟^(-1/2)[(1 + 𝒦)⁻¹ - (1 + 𝒦₀)⁻¹]𝒟^(-1/2)` is `gammaHF`,
Eq. (Gamma-hf). -/
theorem HFHyp.gammaOpR_eq {u : ℂ} (h : A.HFHyp k ε u) (n m : ℤ) :
    A.gammaOpR k ε n m = A.gammaHF k ε u n m := by
  rw [A.gammaOpR_eq_of_isRightInverse k ε h.chiR_isRightInverse (A.bdd_invHF u), h.gammaHF_eq]

/-- **`1 + 𝒦` is invertible** at a lattice zero `z_c² = ŝ` of Corollary "half filling". -/
theorem HFLZ.isUnit_onePlusKR {c : ℤ} (h : A.HFLZ k ε c) : IsUnit (A.onePlusKR k ε) :=
  A.isUnit_onePlusKR_of_isRightInverse k ε h.chiR_isRightInverse (A.bdd_invHFlz c)

/-- **The operator vertex at a lattice zero** `z_c² = ŝ` is `gammaHFlz`, Eq. (Gamma-hf-lz). -/
theorem HFLZ.gammaOpR_eq {c : ℤ} (h : A.HFLZ k ε c) (n m : ℤ) :
    A.gammaOpR k ε n m = A.gammaHFlz k ε c n m := by
  rw [A.gammaOpR_eq_of_isRightInverse k ε h.chiR_isRightInverse (A.bdd_invHFlz c),
    h.gammaHF_eq]

/-- **The operator vertex is the closed form** for `μ ≠ 0`: on the domain of Theorem "density and
magnetic" (generic case), `Γ_r` is `gammaR`, Eq. (Gamma-ph). -/
theorem DMGeneric.gammaOpR_eq {u₁ u₂ : ℂ} (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    A.gammaOpR k ε n m = A.gammaR k ε u₁ u₂ n m := by
  rw [A.gammaOpR_eq_of_isRightInverse k ε h.chiR_isRightInverse (fun m ↦ by
    obtain ⟨C, hC⟩ := (A.tendsto_invR k ε u₁ u₂ m).norm.bddAbove_range_of_cofinite
    exact ⟨C, fun n ↦ hC ⟨n, rfl⟩⟩), h.gammaR_eq]

/-! ### Finite boxes -/

variable (A k ε)

/-- **Finite boxes**: wherever `1 + 𝒦` is invertible, the box matrices `χ_r|_M` on
`|2n + 1 + k| ≤ 2M + 1` are invertible for all sufficiently large `M`. -/
theorem eventually_isUnit_chiBoxR_det (hu : IsUnit (A.onePlusKR k ε)) :
    ∀ᶠ M in atTop, IsUnit (A.chiBox k ε (box k M)).det :=
  (A.opLatticeAll k).eventually_isUnit_chiBoxX_det _ (A.chiDecay_chiR_all k ε) (tendsto_box k) hu

/-- **The box vertices converge to `Γ_r`** wherever `1 + 𝒦` is invertible. -/
theorem tendsto_gammaBoxR (hu : IsUnit (A.onePlusKR k ε)) (n m : ℤ) :
    Tendsto (fun M ↦ A.gammaBox k ε (box k M) n m) atTop (𝓝 (A.gammaOpR k ε n m)) :=
  (A.opLatticeAll k).tendsto_gammaBoxX _ _ _ _ (tendsto_box k) hu n m

variable {A k ε}

/-- **Boxes at half filling**: under the assumptions of Corollary "half filling", the box vertices
converge to the closed form `gammaHF`. -/
theorem HFHyp.tendsto_gammaBox {u : ℂ} (h : A.HFHyp k ε u) (n m : ℤ) :
    Tendsto (fun M ↦ A.gammaBox k ε (box k M) n m) atTop (𝓝 (A.gammaHF k ε u n m)) := by
  rw [← h.gammaOpR_eq]
  exact A.tendsto_gammaBoxR k ε h.isUnit_onePlusKR n m

end Params

end HubbardAtom
