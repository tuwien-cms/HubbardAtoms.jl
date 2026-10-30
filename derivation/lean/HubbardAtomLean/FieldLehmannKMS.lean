import HubbardAtomLean.FieldLehmann

/-!
# The four-point function under a shift of the time origin

Moving the operator at `τ = 0` (the trace is cyclic, and the fermionic phases turn the shift of
the time origin into a sign):

* `simplexW_rotate`: the closed form of a path term with the weights rotated and the rates
  `(r₁, r₂, r₃) ↦ (-r₁₂₃, r₁, r₂)`;
* `FParams.pathTerm_rotate`, `FParams.orderedTerm_rotate`: for fermionic phases,
  `∫_{β>s₁>s₂>s₃>0} e^{ψ·s}⟨P₁P₂P₃P₄⟩ = -∫ e^{ψ'·s}⟨P₄P₁P₂P₃⟩` with
  `ψ' = (-(ψ₁ + ψ₂ + ψ₃), ψ₁, ψ₂)`, and `lineTerm_rotate` for two operators;
* `FParams.chiLehmann_dag_left`, `FParams.chiLehmann_dag_right`: **the particle–hole reflection of
  one pair**: `χ[c†, c'](νₙ, νₘ, ω) = -χ[c, c'](ν_n̄, νₘ, ω)` and
  `χ[c, c'†](νₙ, νₘ, ω) = -χ[c, c'](νₙ, ν_m̄, ω)`, with `n̄ = -n - k - 1`.
-/

open Complex Matrix

namespace HubbardAtom

/-- **Rotating a path in closed form**: the simplex integral with the weights `(P, Q₁, Q₂, Q₃)`
rotated to `(Q₃, P, Q₁, Q₂)` and the rates `(r₁, r₂, r₃)` to `(-r₁₂₃, r₁, r₂)` changes sign; the
two degenerate cases trade places. Where `r₂ + r₃ = 0` the weights `Q₁`, `Q₃` agree. -/
theorem simplexW_rotate {β P Q₁ Q₂ Q₃ r₁ r₂ r₃ : ℂ} (h₁ : r₁ ≠ 0) (h₂ : r₂ ≠ 0) (h₃ : r₃ ≠ 0)
    (h₁₂₃ : r₁ + r₂ + r₃ ≠ 0) (hQ : r₂ + r₃ = 0 → Q₃ = Q₁) :
    simplexW β P Q₁ Q₂ Q₃ r₁ r₂ r₃ (r₁ + r₂) (r₂ + r₃) (r₁ + r₂ + r₃) (r₁ + r₂ = 0)
        (r₂ + r₃ = 0) =
      -simplexW β Q₃ P Q₁ Q₂ (-(r₁ + r₂ + r₃)) r₁ r₂ (-(r₂ + r₃)) (r₁ + r₂) (-r₃)
        (r₂ + r₃ = 0) (r₁ + r₂ = 0) := by
  have h₁₂₃' : -(r₁ + r₂ + r₃) ≠ 0 := neg_ne_zero.mpr h₁₂₃
  unfold simplexW
  by_cases h12 : r₁ + r₂ = 0 <;> by_cases h23 : r₂ + r₃ = 0
  · obtain rfl : r₂ = -r₁ := by linear_combination h12
    obtain rfl : r₃ = r₁ := by linear_combination h23
    rw [hQ h23]
    simp only [h12, h23, ↓reduceIte, zero_add]
    field_simp
    ring
  · obtain rfl : r₂ = -r₁ := by linear_combination h12
    have h23' : -(-r₁ + r₃) ≠ 0 := neg_ne_zero.mpr h23
    simp only [h12, h23, ↓reduceIte, zero_add]
    field_simp
    ring
  · obtain rfl : r₃ = -r₂ := by linear_combination h23
    rw [hQ h23]
    simp only [h12, h23, ↓reduceIte, add_neg_cancel_right]
    field_simp
    ring
  · have h23' : -(r₂ + r₃) ≠ 0 := neg_ne_zero.mpr h23
    simp only [h12, h23, ↓reduceIte]
    field_simp
    ring

namespace FParams

variable (A : FParams)

/-! ### Paths and ordered terms -/

/-- **Rotating a path**: for fermionic phases `ψ₁, ψ₂, ψ₃, ψ₄` with `ψ₁ + ψ₂ + ψ₃ + ψ₄ = 0`, the
path `i₀ ← i₁ ← i₂ ← i₃ ← i₀` with the operator at `τ = 0` moved to the front changes sign. -/
theorem pathTerm_rotate (i₀ i₁ i₂ i₃ : Fin 4) {ψ₁ ψ₂ ψ₃ ψ₄ : ℂ} (hψ₁ : cexp (ψ₁ * A.β) = -1)
    (hψ₂ : cexp (ψ₂ * A.β) = -1) (hψ₃ : cexp (ψ₃ * A.β) = -1) (h : ψ₁ + ψ₂ + ψ₃ + ψ₄ = 0) :
    A.pathTerm ψ₁ ψ₂ ψ₃ i₀ i₁ i₂ i₃ = -A.pathTerm ψ₄ ψ₁ ψ₂ i₃ i₀ i₁ i₂ := by
  have hψ₁₂ : cexp ((ψ₁ + ψ₂) * A.β) = 1 := by
    rw [add_mul, Complex.exp_add, hψ₁, hψ₂]
    norm_num
  have hψ₂₃ : cexp ((ψ₂ + ψ₃) * A.β) = 1 := by
    rw [add_mul, Complex.exp_add, hψ₂, hψ₃]
    norm_num
  have hψ₁₂₃ : cexp ((ψ₁ + ψ₂ + ψ₃) * A.β) = -1 := by
    rw [add_mul, Complex.exp_add, hψ₁₂, hψ₃, one_mul]
  have hψ₄ : cexp (ψ₄ * A.β) = -1 := by
    rw [show ψ₄ = -(ψ₁ + ψ₂ + ψ₃) by linear_combination h, neg_mul, Complex.exp_neg, hψ₁₂₃]
    norm_num
  have n₁ := A.rate_ne_zero (i := i₀) (j := i₁) hψ₁
  have n₂ := A.rate_ne_zero (i := i₁) (j := i₂) hψ₂
  have n₃ := A.rate_ne_zero (i := i₂) (j := i₃) hψ₃
  have n₁₂₃ : A.rate i₀ i₁ ψ₁ + A.rate i₁ i₂ ψ₂ + A.rate i₂ i₃ ψ₃ ≠ 0 := by
    rw [A.rate_add, A.rate_add]
    exact A.rate_ne_zero hψ₁₂₃
  have hQ : A.rate i₁ i₂ ψ₂ + A.rate i₂ i₃ ψ₃ = 0 → (A.weight i₃ : ℂ) = A.weight i₁ := fun h0 ↦ by
    have hw := A.weight_mul_rate (i := i₁) (j := i₃) hψ₂₃
    rw [← A.rate_add i₁ i₂ i₃, h0, zero_mul, Complex.exp_zero, mul_one, one_mul] at hw
    exact hw.symm
  have e₄ : A.rate i₃ i₀ ψ₄ =
      -(A.rate i₀ i₁ ψ₁ + A.rate i₁ i₂ ψ₂ + A.rate i₂ i₃ ψ₃) := by
    simp only [rate]
    linear_combination h
  rw [A.pathTerm_eq i₀ i₁ i₂ i₃ hψ₁ hψ₂ hψ₃ rfl rfl rfl rfl rfl rfl Iff.rfl Iff.rfl,
    A.pathTerm_eq i₃ i₀ i₁ i₂ hψ₄ hψ₁ hψ₂ e₄ rfl rfl
      (r₁₂ := -(A.rate i₁ i₂ ψ₂ + A.rate i₂ i₃ ψ₃)) (r₂₃ := A.rate i₀ i₁ ψ₁ + A.rate i₁ i₂ ψ₂)
      (r₁₂₃ := -A.rate i₂ i₃ ψ₃) (by ring) rfl (by ring) neg_eq_zero Iff.rfl]
  exact simplexW_rotate n₁ n₂ n₃ n₁₂₃ hQ

/-- **Rotating an ordered term**: for fermionic phases with `ψ₁ + ψ₂ + ψ₃ + ψ₄ = 0`,
`∫_{β>s₁>s₂>s₃>0} e^{ψ·s}⟨P₁(s₁)P₂(s₂)P₃(s₃)P₄⟩` changes sign when `P₄` moves to the front. -/
theorem orderedTerm_rotate {ψ₁ ψ₂ ψ₃ ψ₄ : ℂ} (hψ₁ : cexp (ψ₁ * A.β) = -1)
    (hψ₂ : cexp (ψ₂ * A.β) = -1) (hψ₃ : cexp (ψ₃ * A.β) = -1) (h : ψ₁ + ψ₂ + ψ₃ + ψ₄ = 0)
    (P₁ P₂ P₃ P₄ : Op) :
    A.orderedTerm ψ₁ ψ₂ ψ₃ P₁ P₂ P₃ P₄ = -A.orderedTerm ψ₄ ψ₁ ψ₂ P₄ P₁ P₂ P₃ := by
  rw [A.orderedTerm_eq, A.orderedTerm_eq]
  have key : ∀ i₀ i₁ i₂ i₃ : Fin 4,
      P₁ i₀ i₁ * P₂ i₁ i₂ * P₃ i₂ i₃ * P₄ i₃ i₀ * A.pathTerm ψ₁ ψ₂ ψ₃ i₀ i₁ i₂ i₃ =
        -(P₄ i₃ i₀ * P₁ i₀ i₁ * P₂ i₁ i₂ * P₃ i₂ i₃ * A.pathTerm ψ₄ ψ₁ ψ₂ i₃ i₀ i₁ i₂) :=
    fun i₀ i₁ i₂ i₃ ↦ by
      rw [A.pathTerm_rotate i₀ i₁ i₂ i₃ hψ₁ hψ₂ hψ₃ h]
      ring
  calc ∑ i₀, ∑ i₁, ∑ i₂, ∑ i₃,
        P₁ i₀ i₁ * P₂ i₁ i₂ * P₃ i₂ i₃ * P₄ i₃ i₀ * A.pathTerm ψ₁ ψ₂ ψ₃ i₀ i₁ i₂ i₃
      = -∑ i₀, ∑ i₁, ∑ i₂, ∑ i₃,
        P₄ i₃ i₀ * P₁ i₀ i₁ * P₂ i₁ i₂ * P₃ i₂ i₃ * A.pathTerm ψ₄ ψ₁ ψ₂ i₃ i₀ i₁ i₂ := by
        simp only [← Finset.sum_neg_distrib]
        exact Finset.sum_congr rfl fun i₀ _ ↦ Finset.sum_congr rfl fun i₁ _ ↦
          Finset.sum_congr rfl fun i₂ _ ↦ Finset.sum_congr rfl fun i₃ _ ↦ key i₀ i₁ i₂ i₃
    _ = -∑ i₀, ∑ i₁, ∑ i₃, ∑ i₂,
        P₄ i₃ i₀ * P₁ i₀ i₁ * P₂ i₁ i₂ * P₃ i₂ i₃ * A.pathTerm ψ₄ ψ₁ ψ₂ i₃ i₀ i₁ i₂ := by
        congr 1
        exact Finset.sum_congr rfl fun _ _ ↦ Finset.sum_congr rfl fun _ _ ↦ Finset.sum_comm
    _ = -∑ i₀, ∑ i₃, ∑ i₁, ∑ i₂,
        P₄ i₃ i₀ * P₁ i₀ i₁ * P₂ i₁ i₂ * P₃ i₂ i₃ * A.pathTerm ψ₄ ψ₁ ψ₂ i₃ i₀ i₁ i₂ := by
        congr 1
        exact Finset.sum_congr rfl fun _ _ ↦ Finset.sum_comm
    _ = _ := by rw [Finset.sum_comm]

/-! ### Two-point terms -/

/-- **Rotating a single-time term**: for a fermionic phase,
`∫₀^β e^{ψτ}⟨P₁(τ)P₂⟩ = -∫₀^β e^{-ψτ}⟨P₂(τ)P₁⟩`. -/
theorem lineTerm_rotate {ψ : ℂ} (hψ : cexp (ψ * A.β) = -1) (P₁ P₂ : Op) :
    ∫ τ in (0 : ℝ)..A.β, cexp (ψ * τ) * A.thermal (A.heis τ P₁ * P₂) =
      -∫ τ in (0 : ℝ)..A.β, cexp (-ψ * τ) * A.thermal (A.heis τ P₂ * P₁) := by
  have hψ' : cexp (-ψ * A.β) = -1 := by
    rw [neg_mul, Complex.exp_neg, hψ]
    norm_num
  rw [A.lineTerm_eq, A.lineTerm_eq, Finset.sum_comm, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun i₁ _ ↦ ?_
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun i₀ _ ↦ ?_
  have hr : A.rate i₁ i₀ (-ψ) = -A.rate i₀ i₁ ψ := by
    simp only [rate]
    ring
  rw [A.pathTerm₁_eq i₀ i₁ hψ rfl, A.pathTerm₁_eq i₁ i₀ hψ' hr]
  have n := A.rate_ne_zero (i := i₀) (j := i₁) hψ
  field_simp
  ring

/-- **The two orders of a pair cancel off `ω = 0`**: for fermionic phases with `φ₁ + φ₂ ≠ 0`
(a nonzero bosonic transfer), `∫_{β>s₁>s₂>0} e^{φ₁s₁+φ₂s₂}⟨P(s₁)Q(s₂)⟩` equals the term with
the order and the phases exchanged. -/
theorem triangleTerm_sub_eq_zero {φ₁ φ₂ : ℂ} (h₁ : cexp (φ₁ * A.β) = -1)
    (h₂ : cexp (φ₂ * A.β) = -1) (h : φ₁ + φ₂ ≠ 0) (P Q : Op) :
    A.triangleTerm φ₁ φ₂ P Q - A.triangleTerm φ₂ φ₁ Q P = 0 := by
  rw [A.triangleTerm_eq, A.triangleTerm_eq, Finset.sum_comm (f := fun i₀ i₁ ↦
    Q i₀ i₁ * P i₁ i₀ * A.pathTerm₂ φ₂ φ₁ i₀ i₁), ← Finset.sum_sub_distrib]
  refine Finset.sum_eq_zero fun i₀ _ ↦ ?_
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_eq_zero fun i₁ _ ↦ ?_
  have e₁₂ : A.rate i₀ i₁ φ₁ + A.rate i₁ i₀ φ₂ = φ₁ + φ₂ := by
    simp only [rate]
    ring
  have e₂₁ : A.rate i₁ i₀ φ₂ + A.rate i₀ i₁ φ₁ = φ₁ + φ₂ := by
    simp only [rate]
    ring
  rw [A.pathTerm₂_eq i₀ i₁ h₁ h₂ rfl rfl e₁₂ (C₁₂ := False) (by simp [h]),
    A.pathTerm₂_eq i₁ i₀ h₂ h₁ rfl rfl e₂₁ (C₁₂ := False) (by simp [h])]
  have n₁ := A.rate_ne_zero (i := i₀) (j := i₁) h₁
  have n₂ := A.rate_ne_zero (i := i₁) (j := i₀) h₂
  simp only [triangleW, ↓reduceIte]
  field_simp
  ring

/-! ### The particle–hole reflection of one pair -/

/-- **Reflecting the first pair**: `χ[c†, c'](ν, ν', ω) = -χ[c, c'](-ν - ω, ν', ω)`, by exchanging
the roles of `τ₁` and `τ₂`. -/
theorem chiLehmann_dag_left (c c' : Op) (ν ν' ω : ℝ) :
    A.chiLehmann cᴴ c' ν ν' ω = -A.chiLehmann c c' (-ν - ω) ν' ω := by
  rw [A.chiLehmann_eq cᴴ c' ν ν' ω (φ₁ := -ν * I) (φ₂ := (ν + ω) * I)
      (φ₃ := -(ν' + ω) * I) (fun τ₁ τ₂ τ₃ ↦ by push_cast; ring),
    A.chiLehmann_eq c c' (-ν - ω) ν' ω (φ₁ := (ν + ω) * I) (φ₂ := -ν * I)
      (φ₃ := -(ν' + ω) * I) (fun τ₁ τ₂ τ₃ ↦ by push_cast; ring),
    Matrix.conjTranspose_conjTranspose]
  ring

/-- **Reflecting the second pair**: `χ[c, c'†](ν, ν', ω) = -χ[c, c'](ν, -ν' - ω, ω)` for fermionic
`ν`, `ν + ω`, `ν' + ω`: the operator at `τ = 0` is moved by rotating every ordered term
(`orderedTerm_rotate`), and the disconnected part by `lineTerm_rotate` (at `ω = 0`) or
`triangleTerm_sub_eq_zero` (off `ω = 0`). -/
theorem chiLehmann_dag_right (c c' : Op) {ν ν' ω : ℝ}
    (h₁ : cexp (-ν * I * A.β) = -1) (h₂ : cexp ((ν + ω) * I * A.β) = -1)
    (h₃ : cexp (-(ν' + ω) * I * A.β) = -1) :
    A.chiLehmann c c'ᴴ ν ν' ω = -A.chiLehmann c c' ν (-ν' - ω) ω := by
  set φ₁ : ℂ := -ν * I
  set φ₂ : ℂ := (ν + ω) * I
  set φ₃ : ℂ := -(ν' + ω) * I
  set φ₄ : ℂ := ν' * I
  have hs : φ₁ + φ₂ + φ₃ + φ₄ = 0 := by
    simp only [φ₁, φ₂, φ₃, φ₄]
    ring
  have h₄ : cexp (φ₄ * A.β) = -1 := by
    have h₁₂₃ : cexp ((φ₁ + φ₂ + φ₃) * A.β) = -1 := by
      rw [add_mul, add_mul, Complex.exp_add, Complex.exp_add, h₁, h₂, h₃]
      norm_num
    rw [show φ₄ = -(φ₁ + φ₂ + φ₃) by linear_combination hs, neg_mul, Complex.exp_neg, h₁₂₃]
    norm_num
  rw [A.chiLehmann_eq c c'ᴴ ν ν' ω (φ₁ := φ₁) (φ₂ := φ₂) (φ₃ := φ₃)
      (fun τ₁ τ₂ τ₃ ↦ by simp only [φ₁, φ₂, φ₃]; push_cast; ring),
    A.chiLehmann_eq c c' ν (-ν' - ω) ω (φ₁ := φ₁) (φ₂ := φ₂) (φ₃ := φ₄)
      (fun τ₁ τ₂ τ₃ ↦ by simp only [φ₁, φ₂, φ₄]; push_cast; ring),
    Matrix.conjTranspose_conjTranspose]
  -- the six ordered terms, rotated until `c'` is at `τ = 0`
  rw [A.orderedTerm_rotate (ψ₄ := φ₄) h₁ h₂ h₃ hs cᴴ c c' c'ᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₄) h₂ h₁ h₃ (by linear_combination hs) c cᴴ c' c'ᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₄) h₁ h₃ h₂ (by linear_combination hs) cᴴ c' c c'ᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₂) h₄ h₁ h₃ (by linear_combination hs) c'ᴴ cᴴ c' c,
    A.orderedTerm_rotate (ψ₄ := φ₄) h₂ h₃ h₁ (by linear_combination hs) c c' cᴴ c'ᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₁) h₄ h₂ h₃ (by linear_combination hs) c'ᴴ c c' cᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₄) h₃ h₁ h₂ (by linear_combination hs) c' cᴴ c c'ᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₂) h₄ h₃ h₁ (by linear_combination hs) c'ᴴ c' cᴴ c,
    A.orderedTerm_rotate (ψ₄ := φ₁) h₂ h₄ h₃ (by linear_combination hs) c c'ᴴ c' cᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₄) h₃ h₂ h₁ (by linear_combination hs) c' c cᴴ c'ᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₁) h₄ h₃ h₂ (by linear_combination hs) c'ᴴ c' c cᴴ,
    A.orderedTerm_rotate (ψ₄ := φ₂) h₁ h₄ h₃ (by linear_combination hs) cᴴ c'ᴴ c' c]
  -- the disconnected part
  by_cases hω : ω = 0
  · have e : -φ₃ = φ₄ := by
      simp only [φ₃, φ₄, hω]
      push_cast
      ring
    rw [A.lineTerm_rotate h₃ c' c'ᴴ, e]
    ring
  · have h₁₂ : φ₁ + φ₂ ≠ 0 := by
      simp only [φ₁, φ₂]
      intro h0
      apply hω
      have : ((ω : ℂ)) * I = 0 := by linear_combination h0
      exact_mod_cast (mul_eq_zero.mp this).resolve_right I_ne_zero
    rw [A.triangleTerm_sub_eq_zero h₁ h₂ h₁₂ cᴴ c]
    ring

end FParams

end HubbardAtom
