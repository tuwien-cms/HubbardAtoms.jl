import HubbardAtomLean.OrderedIntegrals
import HubbardAtomLean.SingletInput

/-!
# The input from the Hamiltonian: definitions and reduction to Fock-space paths

The objects of Section "Setup" of the note, defined from the Hamiltonian of the atom, as in
Ref. [Thunström et al. 2018].

* `cUp`, `cDn`: the annihilators on the Fock space `|0⟩, |↑⟩, |↓⟩, |↑↓⟩` (`Op`); they satisfy the
  canonical anticommutation relations (`anticomm_up`, ...).
* `Params.H`: the Hamiltonian `H = U (n↑ - 1/2)(n↓ - 1/2) - μ (n↑ + n↓ - 1)`; `Params.heis`: the
  imaginary-time evolution `O(τ) = e^{τH} O e^{-τH}` (Mathlib's matrix exponential);
  `Params.thermal`: `⟨X⟩ = Tr(e^{-βH} X)/Tr e^{-βH}`; `Params.timeOrdered`: the fermionic
  time-ordered product, as a sum over the permutations of the times with their signs.
* `Params.propagator`: `G(ν) = -∫₀^β dτ e^{iντ} ⟨T c↑(τ) c↑†⟩`; `Params.chiLehmann`: the
  generalized susceptibility `χ_σσ'(ν, ν', ω)`, the triple integral of the note.

The reduction:
* `H_eq` (`H` is diagonal, with the energies of the note), `heis_apply`, `thermal_eq` (the weights
  `p₀, p₁, p₁, p₂`), `timeOrdered_eq` (at distinct times only the time-ordered term survives).
* `chiLehmann_eq`: `χ` is a sum over the six orderings of `τ₁, τ₂, τ₃`, with their signs, of
  integrals over the ordered simplex (`orderedTerm`), minus the disconnected part, which factorizes
  into a two-time (`triangleTerm`) and a one-time integral.
* `orderedTerm_eq`, `triangleTerm_eq`, `lineTerm_eq`: each as a sum over the paths of Fock states;
  `pathTerm_eq`, `pathTerm₂_eq`, `pathTerm₁_eq`: one path in closed form, through the weights.
-/

open Complex Matrix

namespace HubbardAtom

/-- Operators on the Fock space of the atom, in the basis `|0⟩, |↑⟩, |↓⟩, |↑↓⟩` (indices `0` to
`3`), where `|↑↓⟩ = c↑† c↓† |0⟩`. -/
abbrev Op := Matrix (Fin 4) (Fin 4) ℂ

/-- The creator `c↑†`: `|0⟩ ↦ |↑⟩`, `|↓⟩ ↦ |↑↓⟩`. -/
def cUpDag : Op := !![0, 0, 0, 0; 1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 1, 0]

/-- The creator `c↓†`: `|0⟩ ↦ |↓⟩`, `|↑⟩ ↦ -|↑↓⟩`. -/
def cDnDag : Op := !![0, 0, 0, 0; 0, 0, 0, 0; 1, 0, 0, 0; 0, -1, 0, 0]

/-- The annihilator `c↑`. -/
def cUp : Op := cUpDagᴴ

/-- The annihilator `c↓`. -/
def cDn : Op := cDnDagᴴ

/-- The number operator `n↑ = c↑† c↑`. -/
def nUp : Op := cUpᴴ * cUp

/-- The number operator `n↓ = c↓† c↓`. -/
def nDn : Op := cDnᴴ * cDn

/-! ### The anticommutation relations -/

section CAR

local macro "car" : tactic =>
  `(tactic| (ext i j
             simp only [Matrix.add_apply, Matrix.mul_apply, Fin.sum_univ_four]
             fin_cases i <;> fin_cases j <;> simp [cUp, cDn, cUpDag, cDnDag]))

theorem anticomm_up : cUp * cUpᴴ + cUpᴴ * cUp = 1 := by car

theorem anticomm_dn : cDn * cDnᴴ + cDnᴴ * cDn = 1 := by car

theorem anticomm_up_dnDag : cUp * cDnᴴ + cDnᴴ * cUp = 0 := by car

theorem anticomm_up_dn : cUp * cDn + cDn * cUp = 0 := by car

theorem up_sq : cUp * cUp = 0 := by car

theorem dn_sq : cDn * cDn = 0 := by car

end CAR

namespace Params

variable (A : Params)

/-! ### The Hamiltonian and the thermal average -/

/-- The Hamiltonian `H = U (n↑ - 1/2)(n↓ - 1/2) - μ (n↑ + n↓ - 1)` of Eq. (H). -/
noncomputable def H : Op :=
  (A.U : ℂ) • ((nUp - (1 / 2 : ℂ) • 1) * (nDn - (1 / 2 : ℂ) • 1)) - (A.μ : ℂ) • (nUp + nDn - 1)

/-- The imaginary-time evolution `O(τ) = e^{τH} O e^{-τH}`. -/
noncomputable def heis (τ : ℝ) (O : Op) : Op :=
  NormedSpace.exp ((τ : ℂ) • A.H) * O * NormedSpace.exp ((-τ : ℂ) • A.H)

/-- The thermal average `⟨X⟩ = Tr(e^{-βH} X)/Tr e^{-βH}`. -/
noncomputable def thermal (X : Op) : ℂ :=
  (NormedSpace.exp ((-A.β : ℂ) • A.H) * X).trace / (NormedSpace.exp ((-A.β : ℂ) • A.H)).trace

/-- The fermionic time-ordered product `T O₀(τ₀) ⋯ Oₙ₋₁(τₙ₋₁)`: the operators in the order of
decreasing time, times the sign of the reordering. (It vanishes when two times coincide, a null
set in the integrals below.) -/
noncomputable def timeOrdered {n : ℕ} (τ : Fin n → ℝ) (O : Fin n → Op) : Op :=
  ∑ σ : Equiv.Perm (Fin n), if StrictAnti (τ ∘ σ) then
    ((Equiv.Perm.sign σ : ℤ) : ℂ) • (List.ofFn fun i ↦ A.heis (τ (σ i)) (O (σ i))).prod else 0

/-- The propagator `G(ν) = -∫₀^β dτ e^{iντ} ⟨T c↑(τ) c↑†⟩` of Section "Setup". -/
noncomputable def propagator (ν : ℝ) : ℂ :=
  -∫ τ in (0 : ℝ)..A.β, cexp (ν * τ * I) * A.thermal (A.timeOrdered ![τ, 0] ![cUp, cUpᴴ])

/-- The generalized susceptibility of Ref. [Thunström et al. 2018], for the annihilators `c` (spin
`σ`) and `c'` (spin `σ'`):
`χ_σσ'(ν, ν', ω) = ∫∫∫₀^β dτ₁dτ₂dτ₃ e^{-iντ₁ + i(ν+ω)τ₂ - i(ν'+ω)τ₃}`
`  × [⟨T c†(τ₁) c(τ₂) c'†(τ₃) c'(0)⟩ - ⟨T c†(τ₁) c(τ₂)⟩⟨T c'†(τ₃) c'(0)⟩]`. -/
noncomputable def chiLehmann (c c' : Op) (ν ν' ω : ℝ) : ℂ :=
  ∫ τ₁ in (0 : ℝ)..A.β, ∫ τ₂ in (0 : ℝ)..A.β, ∫ τ₃ in (0 : ℝ)..A.β,
    cexp (((-ν * τ₁ + (ν + ω) * τ₂ - (ν' + ω) * τ₃ : ℝ) : ℂ) * I) *
      (A.thermal (A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![cᴴ, c, c'ᴴ, c']) -
        A.thermal (A.timeOrdered ![τ₁, τ₂] ![cᴴ, c]) *
          A.thermal (A.timeOrdered ![τ₃, 0] ![c'ᴴ, c']))

/-! ### Evaluation -/

/-- The energies `U/4 + μ, -U/4, -U/4, U/4 - μ` of `|0⟩, |↑⟩, |↓⟩, |↑↓⟩` (`a = U/2`). -/
noncomputable def energy : Fin 4 → ℝ := ![A.a / 2 + A.μ, -(A.a / 2), -(A.a / 2), A.a / 2 - A.μ]

/-- The weights `p₀, p₁, p₁, p₂` of `|0⟩, |↑⟩, |↓⟩, |↑↓⟩`. -/
noncomputable def weight : Fin 4 → ℝ := ![A.p₀, A.p₁, A.p₁, A.p₂]

/-- `H` is diagonal, with the energies of Section "Setup". -/
theorem H_eq : A.H = diagonal fun i ↦ (A.energy i : ℂ) := by
  have hU : nUp = diagonal ![0, 1, 0, 1] := by
    ext i j
    simp only [nUp, Matrix.mul_apply, Fin.sum_univ_four]
    fin_cases i <;> fin_cases j <;> simp [cUp, cUpDag]
  have hD : nDn = diagonal ![0, 0, 1, 1] := by
    ext i j
    simp only [nDn, Matrix.mul_apply, Fin.sum_univ_four]
    fin_cases i <;> fin_cases j <;> simp [cDn, cDnDag]
  rw [H, hU, hD]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [energy, a, Matrix.mul_apply, Fin.sum_univ_four] <;> ring

theorem exp_smul_H (c : ℂ) :
    NormedSpace.exp (c • A.H) = diagonal fun i ↦ cexp (c * A.energy i) := by
  rw [H_eq, ← diagonal_smul, exp_diagonal]
  congr 1
  funext i
  rw [Pi.coe_exp, ← Complex.exp_eq_exp_ℂ]
  rfl

/-- `O(τ)_{ij} = e^{τ(E_i - E_j)} O_{ij}`. -/
theorem heis_apply (τ : ℝ) (O : Op) (i j : Fin 4) :
    A.heis τ O i j = cexp (τ * (A.energy i - A.energy j)) * O i j := by
  rw [heis, exp_smul_H, exp_smul_H, Matrix.mul_diagonal, Matrix.diagonal_mul,
    mul_right_comm, ← Complex.exp_add]
  congr 2
  ring

@[simp] theorem heis_zero (O : Op) : A.heis 0 O = O := by
  ext i j
  simp [heis_apply]

theorem continuous_heis (O : Op) : Continuous fun τ : ℝ ↦ A.heis τ O := by
  refine continuous_matrix fun i j ↦ ?_
  simp_rw [heis_apply]
  fun_prop

theorem weight_eq (i : Fin 4) : A.weight i = Real.exp (-A.β * A.energy i) / A.Z := by
  fin_cases i <;> simp [weight, energy, p₀, p₁, p₂, boltz₀, boltz₁, boltz₂, a] <;> ring_nf

/-- `⟨X⟩ = Σᵢ pᵢ X_ii`. -/
theorem thermal_eq (X : Op) : A.thermal X = ∑ i, (A.weight i : ℂ) * X i i := by
  have hZ : (A.Z : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.Z_pos.ne'
  have hexp : ∀ i, cexp (-(A.β : ℂ) * A.energy i) = (Real.exp (-A.β * A.energy i) : ℝ) := by
    intro i
    push_cast
    rfl
  have htr : (NormedSpace.exp ((-A.β : ℂ) • A.H)).trace = A.Z := by
    rw [exp_smul_H, trace_diagonal]
    simp only [hexp, ← ofReal_sum]
    congr 1
    simp [Fin.sum_univ_four, Z, boltz₀, boltz₁, boltz₂, energy, a]
    ring_nf
  rw [thermal, htr, exp_smul_H, trace, Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [diag_apply, diagonal_mul, weight_eq]
  push_cast
  field_simp

theorem continuous_thermal : Continuous A.thermal := by
  simp_rw [funext A.thermal_eq]
  exact continuous_finsetSum _ fun i _ ↦ continuous_const.mul (continuous_id.matrix_elem i i)

/-- `pᵢ e^{β(Eᵢ - Eⱼ)} = pⱼ`. -/
theorem weight_mul_exp (i j : Fin 4) :
    (A.weight i : ℂ) * cexp ((A.energy i - A.energy j) * A.β) = A.weight j := by
  rw [weight_eq, weight_eq]
  push_cast
  rw [div_mul_eq_mul_div, ← Complex.exp_add]
  congr 2
  ring

/-! ### The time-ordered product at distinct times -/

theorem perm_eq_of_strictAnti {n : ℕ} {τ : Fin n → ℝ} {σ π : Equiv.Perm (Fin n)}
    (hσ : StrictAnti (τ ∘ σ)) (hπ : StrictAnti (τ ∘ π)) : π = σ := by
  have hρ : StrictMono (π.trans σ.symm) := by
    intro i j hij
    have h1 := hπ hij
    have h2 : ∀ l, (τ ∘ σ) (σ.symm (π l)) = (τ ∘ π) l := by simp
    rw [← h2, ← h2] at h1
    exact hσ.lt_iff_gt.mp h1
  ext i
  have h := congrFun (StrictMono.eq_id hρ) i
  simp only [Equiv.trans_apply, id_eq] at h
  have h' := congrArg σ h
  rw [Equiv.apply_symm_apply] at h'
  rw [h']

/-- At distinct times, ordered by `σ`, only the term of `σ` survives. -/
theorem timeOrdered_eq {n : ℕ} {τ : Fin n → ℝ} (O : Fin n → Op) (σ : Equiv.Perm (Fin n))
    (h : StrictAnti (τ ∘ σ)) :
    A.timeOrdered τ O =
      ((Equiv.Perm.sign σ : ℤ) : ℂ) • (List.ofFn fun i ↦ A.heis (τ (σ i)) (O (σ i))).prod := by
  rw [timeOrdered, Finset.sum_eq_single σ]
  · rw [ite_eq_left h]
  · intro π _ hπ
    rw [ite_eq_right]
    intro h'
    exact hπ (perm_eq_of_strictAnti h h')
  · simp

theorem thermal_smul (c : ℂ) (X : Op) : A.thermal (c • X) = c * A.thermal X := by
  simp only [thermal_eq, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ ↦ by ring

theorem thermal_neg (X : Op) : A.thermal (-X) = -A.thermal X := by
  rw [← neg_one_smul ℂ X, thermal_smul, neg_one_mul]

@[fun_prop]
theorem continuous_heis_comp {X : Type*} [TopologicalSpace X] {f : X → ℝ} (hf : Continuous f)
    (O : Op) : Continuous fun x ↦ A.heis (f x) O :=
  (A.continuous_heis O).comp hf

@[fun_prop]
theorem continuous_thermal_comp {X : Type*} [TopologicalSpace X] {f : X → Op}
    (hf : Continuous f) : Continuous fun x ↦ A.thermal (f x) :=
  A.continuous_thermal.comp hf

/-! ### The orderings of the times -/

section Orderings

variable {τ₁ τ₂ τ₃ : ℝ} (P₀ P₁ P₂ P₃ : Op)

local macro "anti" : tactic =>
  `(tactic| (rw [Fin.strictAnti_iff_succ_lt]
             intro i
             fin_cases i <;> simp [Equiv.swap_apply_of_ne_of_ne] <;> linarith))

theorem timeOrdered_123 (h₁ : 0 < τ₃) (h₂ : τ₃ < τ₂) (h₃ : τ₂ < τ₁) :
    A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![P₀, P₁, P₂, P₃] =
      A.heis τ₁ P₀ * (A.heis τ₂ P₁ * (A.heis τ₃ P₂ * P₃)) := by
  rw [A.timeOrdered_eq _ 1 (by anti)]
  simp [List.ofFn_succ]

theorem timeOrdered_132 (h₁ : 0 < τ₂) (h₂ : τ₂ < τ₃) (h₃ : τ₃ < τ₁) :
    A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![P₀, P₁, P₂, P₃] =
      -(A.heis τ₁ P₀ * (A.heis τ₃ P₂ * (A.heis τ₂ P₁ * P₃))) := by
  rw [A.timeOrdered_eq _ (Equiv.swap 1 2) (by anti)]
  simp [List.ofFn_succ, Equiv.swap_apply_of_ne_of_ne]

theorem timeOrdered_312 (h₁ : 0 < τ₂) (h₂ : τ₂ < τ₁) (h₃ : τ₁ < τ₃) :
    A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![P₀, P₁, P₂, P₃] =
      A.heis τ₃ P₂ * (A.heis τ₁ P₀ * (A.heis τ₂ P₁ * P₃)) := by
  rw [A.timeOrdered_eq _ (Equiv.swap 0 1 * Equiv.swap 0 2) (by anti)]
  simp [List.ofFn_succ, Equiv.swap_apply_of_ne_of_ne]

theorem timeOrdered_213 (h₁ : 0 < τ₃) (h₂ : τ₃ < τ₁) (h₃ : τ₁ < τ₂) :
    A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![P₀, P₁, P₂, P₃] =
      -(A.heis τ₂ P₁ * (A.heis τ₁ P₀ * (A.heis τ₃ P₂ * P₃))) := by
  rw [A.timeOrdered_eq _ (Equiv.swap 0 1) (by anti)]
  simp [List.ofFn_succ, Equiv.swap_apply_of_ne_of_ne]

theorem timeOrdered_231 (h₁ : 0 < τ₁) (h₂ : τ₁ < τ₃) (h₃ : τ₃ < τ₂) :
    A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![P₀, P₁, P₂, P₃] =
      A.heis τ₂ P₁ * (A.heis τ₃ P₂ * (A.heis τ₁ P₀ * P₃)) := by
  rw [A.timeOrdered_eq _ (Equiv.swap 0 2 * Equiv.swap 0 1) (by anti)]
  simp [List.ofFn_succ, Equiv.swap_apply_of_ne_of_ne]

theorem timeOrdered_321 (h₁ : 0 < τ₁) (h₂ : τ₁ < τ₂) (h₃ : τ₂ < τ₃) :
    A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![P₀, P₁, P₂, P₃] =
      -(A.heis τ₃ P₂ * (A.heis τ₂ P₁ * (A.heis τ₁ P₀ * P₃))) := by
  rw [A.timeOrdered_eq _ (Equiv.swap 0 2) (by anti)]
  simp [List.ofFn_succ, Equiv.swap_apply_of_ne_of_ne]

theorem timeOrdered_12 (h : τ₂ < τ₁) :
    A.timeOrdered ![τ₁, τ₂] ![P₀, P₁] = A.heis τ₁ P₀ * A.heis τ₂ P₁ := by
  rw [A.timeOrdered_eq _ 1 (by anti)]
  simp [List.ofFn_succ]

theorem timeOrdered_21 (h : τ₁ < τ₂) :
    A.timeOrdered ![τ₁, τ₂] ![P₀, P₁] = -(A.heis τ₂ P₁ * A.heis τ₁ P₀) := by
  rw [A.timeOrdered_eq _ (Equiv.swap 0 1) (by anti)]
  simp [List.ofFn_succ]

end Orderings

/-! ### The six orderings of the susceptibility -/

/-- `e^{ψ₁s₁ + ψ₂s₂ + ψ₃s₃} ⟨P₁(s₁) P₂(s₂) P₃(s₃) P₄⟩`. -/
noncomputable def orderedIntegrand (ψ₁ ψ₂ ψ₃ : ℂ) (P₁ P₂ P₃ P₄ : Op) (s₁ s₂ s₃ : ℝ) : ℂ :=
  cexp (ψ₁ * s₁ + ψ₂ * s₂ + ψ₃ * s₃) *
    A.thermal (A.heis s₁ P₁ * (A.heis s₂ P₂ * (A.heis s₃ P₃ * P₄)))

/-- The contribution of one ordering of the times: the integral of `orderedIntegrand` over the
ordered simplex `β > s₁ > s₂ > s₃ > 0`. -/
noncomputable def orderedTerm (ψ₁ ψ₂ ψ₃ : ℂ) (P₁ P₂ P₃ P₄ : Op) : ℂ :=
  simplexIntegral A.β (A.orderedIntegrand ψ₁ ψ₂ ψ₃ P₁ P₂ P₃ P₄)

/-- The integral over the ordered triangle `β > s₁ > s₂ > 0` of
`e^{ψ₁s₁ + ψ₂s₂} ⟨P₁(s₁) P₂(s₂)⟩`. -/
noncomputable def triangleTerm (ψ₁ ψ₂ : ℂ) (P₁ P₂ : Op) : ℂ :=
  triangleIntegral A.β fun s₁ s₂ ↦
    cexp (ψ₁ * s₁ + ψ₂ * s₂) * A.thermal (A.heis s₁ P₁ * A.heis s₂ P₂)

theorem continuous_orderedIntegrand (ψ₁ ψ₂ ψ₃ : ℂ) (P₁ P₂ P₃ P₄ : Op) :
    Continuous₃ (A.orderedIntegrand ψ₁ ψ₂ ψ₃ P₁ P₂ P₃ P₄) := by
  unfold Continuous₃ orderedIntegrand
  fun_prop

/-- The disconnected integrand `e^{φ·τ} ⟨T c†(τ₁) c(τ₂)⟩⟨T c'†(τ₃) c'⟩` for `τ₁ > τ₂`. -/
private noncomputable def disc₁ (c c' : Op) (φ₁ φ₂ φ₃ : ℂ) (τ₁ τ₂ τ₃ : ℝ) : ℂ :=
  cexp (φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃) *
    (A.thermal (A.heis τ₁ cᴴ * A.heis τ₂ c) * A.thermal (A.heis τ₃ c'ᴴ * c'))

/-- The disconnected integrand for `τ₁ < τ₂`. -/
private noncomputable def disc₂ (c c' : Op) (φ₁ φ₂ φ₃ : ℂ) (τ₁ τ₂ τ₃ : ℝ) : ℂ :=
  cexp (φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃) *
    (-A.thermal (A.heis τ₂ c * A.heis τ₁ cᴴ) * A.thermal (A.heis τ₃ c'ᴴ * c'))

/-- `S(-k - d) = -S(k) - S(d)` and `S(k - d) = S(k) - S(d)` on the ordered simplex. -/
private theorem simplexIntegral_sign_sub {ε : ℂ} {k d : ℝ → ℝ → ℝ → ℂ} (hk : Continuous₃ k)
    (hd : Continuous₃ d) :
    simplexIntegral A.β (fun s₁ s₂ s₃ ↦ ε * k s₁ s₂ s₃ - d s₁ s₂ s₃) =
      ε * simplexIntegral A.β k - simplexIntegral A.β d := by
  rw [simplexIntegral_sub _ (by unfold Continuous₃ at hk ⊢; fun_prop) hd,
    simplexIntegral_const_mul]

/-- The susceptibility as six ordered terms (the signs of the orderings) minus the disconnected
part, which factorizes. -/
theorem chiLehmann_eq (c c' : Op) (ν ν' ω : ℝ) {φ₁ φ₂ φ₃ : ℂ}
    (hφ : ∀ τ₁ τ₂ τ₃ : ℝ, (((-ν * τ₁ + (ν + ω) * τ₂ - (ν' + ω) * τ₃ : ℝ) : ℂ) * I) =
      φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃) :
    A.chiLehmann c c' ν ν' ω =
      A.orderedTerm φ₁ φ₂ φ₃ cᴴ c c'ᴴ c' - A.orderedTerm φ₁ φ₃ φ₂ cᴴ c'ᴴ c c' +
        A.orderedTerm φ₃ φ₁ φ₂ c'ᴴ cᴴ c c' - A.orderedTerm φ₂ φ₁ φ₃ c cᴴ c'ᴴ c' +
        A.orderedTerm φ₂ φ₃ φ₁ c c'ᴴ cᴴ c' - A.orderedTerm φ₃ φ₂ φ₁ c'ᴴ c cᴴ c' -
        (A.triangleTerm φ₁ φ₂ cᴴ c - A.triangleTerm φ₂ φ₁ c cᴴ) *
          ∫ τ in (0 : ℝ)..A.β, cexp (φ₃ * τ) * A.thermal (A.heis τ c'ᴴ * c') := by
  have hβ := A.β_pos.le
  have hT₁ : ∀ τ₃ : ℝ, 0 < τ₃ → A.timeOrdered ![τ₃, 0] ![c'ᴴ, c'] = A.heis τ₃ c'ᴴ * c' := by
    intro τ₃ h
    rw [A.timeOrdered_12 _ _ h, heis_zero]
  have cd₁ : Continuous₃ (A.disc₁ c c' φ₁ φ₂ φ₃) := by
    unfold Continuous₃ disc₁
    fun_prop
  have cd₂ : Continuous₃ (A.disc₂ c c' φ₁ φ₂ φ₃) := by
    unfold Continuous₃ disc₂
    fun_prop
  have cp : ∀ {d : ℝ → ℝ → ℝ → ℂ}, Continuous₃ d → Continuous₃ (fun s₁ s₂ s₃ ↦ d s₁ s₃ s₂) ∧
      Continuous₃ (fun s₁ s₂ s₃ ↦ d s₂ s₃ s₁) ∧ Continuous₃ (fun s₁ s₂ s₃ ↦ d s₂ s₁ s₃) ∧
      Continuous₃ (fun s₁ s₂ s₃ ↦ d s₃ s₁ s₂) ∧ Continuous₃ (fun s₁ s₂ s₃ ↦ d s₃ s₂ s₁) :=
    fun hd ↦ ⟨hd.comp (by fun_prop) (by fun_prop) (by fun_prop),
      hd.comp (by fun_prop) (by fun_prop) (by fun_prop),
      hd.comp (by fun_prop) (by fun_prop) (by fun_prop),
      hd.comp (by fun_prop) (by fun_prop) (by fun_prop),
      hd.comp (by fun_prop) (by fun_prop) (by fun_prop)⟩
  have ck := A.continuous_orderedIntegrand
  -- the integrand, and its disconnected part
  obtain ⟨F, hF⟩ : ∃ F : ℝ → ℝ → ℝ → ℂ, F = fun τ₁ τ₂ τ₃ : ℝ ↦
      cexp (φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃) *
        (A.thermal (A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![cᴴ, c, c'ᴴ, c']) -
          A.thermal (A.timeOrdered ![τ₁, τ₂] ![cᴴ, c]) *
            A.thermal (A.timeOrdered ![τ₃, 0] ![c'ᴴ, c'])) := ⟨_, rfl⟩
  obtain ⟨D, hD⟩ : ∃ D : ℝ → ℝ → ℝ → ℂ, D = fun τ₁ τ₂ τ₃ : ℝ ↦
      cexp (φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃) *
        (A.thermal (A.timeOrdered ![τ₁, τ₂] ![cᴴ, c]) *
          A.thermal (A.timeOrdered ![τ₃, 0] ![c'ᴴ, c'])) := ⟨_, rfl⟩
  have hχ : A.chiLehmann c c' ν ν' ω =
      ∫ τ₁ in (0 : ℝ)..A.β, ∫ τ₂ in (0 : ℝ)..A.β, ∫ τ₃ in (0 : ℝ)..A.β, F τ₁ τ₂ τ₃ := by
    simp only [hF, chiLehmann, hφ]
  -- the disconnected part on the orderings with `τ₁ > τ₂` and `τ₁ < τ₂`
  have eD₁ : ∀ τ₁ τ₂ τ₃ : ℝ, τ₂ < τ₁ → 0 < τ₃ →
      D τ₁ τ₂ τ₃ = A.disc₁ c c' φ₁ φ₂ φ₃ τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃ h h₃
    simp only [hD, disc₁]
    rw [A.timeOrdered_12 _ _ h, hT₁ τ₃ h₃]
  have eD₂ : ∀ τ₁ τ₂ τ₃ : ℝ, τ₁ < τ₂ → 0 < τ₃ →
      D τ₁ τ₂ τ₃ = A.disc₂ c c' φ₁ φ₂ φ₃ τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃ h h₃
    simp only [hD, disc₂]
    rw [A.timeOrdered_21 _ _ h, hT₁ τ₃ h₃, thermal_neg]
  have hFD : ∀ τ₁ τ₂ τ₃ : ℝ, F τ₁ τ₂ τ₃ = cexp (φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃) *
      A.thermal (A.timeOrdered ![τ₁, τ₂, τ₃, 0] ![cᴴ, c, c'ᴴ, c']) - D τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃
    simp only [hF, hD]
    ring
  -- the six orderings of the connected part
  have e₁₂₃ : ∀ τ₁ τ₂ τ₃ : ℝ, 0 < τ₃ → τ₃ < τ₂ → τ₂ < τ₁ → τ₁ < A.β → F τ₁ τ₂ τ₃ =
      1 * A.orderedIntegrand φ₁ φ₂ φ₃ cᴴ c c'ᴴ c' τ₁ τ₂ τ₃ - A.disc₁ c c' φ₁ φ₂ φ₃ τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃ h₁ h₂ h₃ _
    rw [hFD, A.timeOrdered_123 _ _ _ _ h₁ h₂ h₃, eD₁ τ₁ τ₂ τ₃ h₃ h₁, orderedIntegrand]
    ring
  have e₁₃₂ : ∀ τ₁ τ₂ τ₃ : ℝ, 0 < τ₂ → τ₂ < τ₃ → τ₃ < τ₁ → τ₁ < A.β → F τ₁ τ₂ τ₃ =
      -1 * A.orderedIntegrand φ₁ φ₃ φ₂ cᴴ c'ᴴ c c' τ₁ τ₃ τ₂ - A.disc₁ c c' φ₁ φ₂ φ₃ τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃ h₁ h₂ h₃ _
    rw [hFD, A.timeOrdered_132 _ _ _ _ h₁ h₂ h₃, eD₁ τ₁ τ₂ τ₃ (h₂.trans h₃) (h₁.trans h₂),
      orderedIntegrand, thermal_neg,
      show φ₁ * (τ₁ : ℂ) + φ₃ * τ₃ + φ₂ * τ₂ = φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃ by ring]
    ring
  have e₃₁₂ : ∀ τ₁ τ₂ τ₃ : ℝ, 0 < τ₂ → τ₂ < τ₁ → τ₁ < τ₃ → τ₃ < A.β → F τ₁ τ₂ τ₃ =
      1 * A.orderedIntegrand φ₃ φ₁ φ₂ c'ᴴ cᴴ c c' τ₃ τ₁ τ₂ - A.disc₁ c c' φ₁ φ₂ φ₃ τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃ h₁ h₂ h₃ _
    rw [hFD, A.timeOrdered_312 _ _ _ _ h₁ h₂ h₃, eD₁ τ₁ τ₂ τ₃ h₂ ((h₁.trans h₂).trans h₃),
      orderedIntegrand,
      show φ₃ * (τ₃ : ℂ) + φ₁ * τ₁ + φ₂ * τ₂ = φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃ by ring]
    ring
  have e₂₁₃ : ∀ τ₁ τ₂ τ₃ : ℝ, 0 < τ₃ → τ₃ < τ₁ → τ₁ < τ₂ → τ₂ < A.β → F τ₁ τ₂ τ₃ =
      -1 * A.orderedIntegrand φ₂ φ₁ φ₃ c cᴴ c'ᴴ c' τ₂ τ₁ τ₃ - A.disc₂ c c' φ₁ φ₂ φ₃ τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃ h₁ h₂ h₃ _
    rw [hFD, A.timeOrdered_213 _ _ _ _ h₁ h₂ h₃, eD₂ τ₁ τ₂ τ₃ h₃ h₁, orderedIntegrand,
      thermal_neg, show φ₂ * (τ₂ : ℂ) + φ₁ * τ₁ + φ₃ * τ₃ = φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃ by ring]
    ring
  have e₂₃₁ : ∀ τ₁ τ₂ τ₃ : ℝ, 0 < τ₁ → τ₁ < τ₃ → τ₃ < τ₂ → τ₂ < A.β → F τ₁ τ₂ τ₃ =
      1 * A.orderedIntegrand φ₂ φ₃ φ₁ c c'ᴴ cᴴ c' τ₂ τ₃ τ₁ - A.disc₂ c c' φ₁ φ₂ φ₃ τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃ h₁ h₂ h₃ _
    rw [hFD, A.timeOrdered_231 _ _ _ _ h₁ h₂ h₃, eD₂ τ₁ τ₂ τ₃ (h₂.trans h₃) (h₁.trans h₂),
      orderedIntegrand,
      show φ₂ * (τ₂ : ℂ) + φ₃ * τ₃ + φ₁ * τ₁ = φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃ by ring]
    ring
  have e₃₂₁ : ∀ τ₁ τ₂ τ₃ : ℝ, 0 < τ₁ → τ₁ < τ₂ → τ₂ < τ₃ → τ₃ < A.β → F τ₁ τ₂ τ₃ =
      -1 * A.orderedIntegrand φ₃ φ₂ φ₁ c'ᴴ c cᴴ c' τ₃ τ₂ τ₁ - A.disc₂ c c' φ₁ φ₂ φ₃ τ₁ τ₂ τ₃ := by
    intro τ₁ τ₂ τ₃ h₁ h₂ h₃ _
    rw [hFD, A.timeOrdered_321 _ _ _ _ h₁ h₂ h₃, eD₂ τ₁ τ₂ τ₃ h₂ ((h₁.trans h₂).trans h₃),
      orderedIntegrand, thermal_neg,
      show φ₃ * (τ₃ : ℂ) + φ₂ * τ₂ + φ₁ * τ₁ = φ₁ * τ₁ + φ₂ * τ₂ + φ₃ * τ₃ by ring]
    ring
  have cg : ∀ {ε : ℂ} {k d : ℝ → ℝ → ℝ → ℂ}, Continuous₃ k → Continuous₃ d →
      Continuous₃ (fun τ₁ τ₂ τ₃ ↦ ε * k τ₁ τ₂ τ₃ - d τ₁ τ₂ τ₃) := by
    intro ε k d hk hd
    unfold Continuous₃ at hk hd ⊢
    fun_prop
  rw [hχ, integral_cube hβ (cg (ck _ _ _ _ _ _ _) cd₁) (cg (cp (ck _ _ _ _ _ _ _)).1 cd₁)
    (cg (cp (ck _ _ _ _ _ _ _)).2.2.2.1 cd₁) (cg (cp (ck _ _ _ _ _ _ _)).2.2.1 cd₂)
    (cg (cp (ck _ _ _ _ _ _ _)).2.1 cd₂) (cg (cp (ck _ _ _ _ _ _ _)).2.2.2.2 cd₂)
    e₁₂₃ e₁₃₂ e₃₁₂ e₂₁₃ e₂₃₁ e₃₂₁]
  rw [A.simplexIntegral_sign_sub (ck _ _ _ _ _ _ _) cd₁,
    A.simplexIntegral_sign_sub (ck _ _ _ _ _ _ _) (cp cd₁).1,
    A.simplexIntegral_sign_sub (ck _ _ _ _ _ _ _) (cp cd₁).2.1,
    A.simplexIntegral_sign_sub (ck _ _ _ _ _ _ _) (cp cd₂).2.2.1,
    A.simplexIntegral_sign_sub (ck _ _ _ _ _ _ _) (cp cd₂).2.2.2.1,
    A.simplexIntegral_sign_sub (ck _ _ _ _ _ _ _) (cp cd₂).2.2.2.2]
  -- the disconnected part, by the same splitting
  have hDsum := integral_cube hβ (F := D) cd₁ cd₁ cd₁ cd₂ cd₂ cd₂
    (fun τ₁ τ₂ τ₃ h₁ h₂ h₃ _ ↦ eD₁ τ₁ τ₂ τ₃ h₃ h₁)
    (fun τ₁ τ₂ τ₃ h₁ h₂ h₃ _ ↦ eD₁ τ₁ τ₂ τ₃ (h₂.trans h₃) (h₁.trans h₂))
    (fun τ₁ τ₂ τ₃ h₁ h₂ h₃ _ ↦ eD₁ τ₁ τ₂ τ₃ h₂ ((h₁.trans h₂).trans h₃))
    (fun τ₁ τ₂ τ₃ h₁ h₂ h₃ _ ↦ eD₂ τ₁ τ₂ τ₃ h₃ h₁)
    (fun τ₁ τ₂ τ₃ h₁ h₂ h₃ _ ↦ eD₂ τ₁ τ₂ τ₃ (h₂.trans h₃) (h₁.trans h₂))
    (fun τ₁ τ₂ τ₃ h₁ h₂ h₃ _ ↦ eD₂ τ₁ τ₂ τ₃ h₂ ((h₁.trans h₂).trans h₃))
  -- which factorizes
  have hfac : ∫ τ₁ in (0 : ℝ)..A.β, ∫ τ₂ in (0 : ℝ)..A.β, ∫ τ₃ in (0 : ℝ)..A.β, D τ₁ τ₂ τ₃ =
      (A.triangleTerm φ₁ φ₂ cᴴ c - A.triangleTerm φ₂ φ₁ c cᴴ) *
        ∫ τ in (0 : ℝ)..A.β, cexp (φ₃ * τ) * A.thermal (A.heis τ c'ᴴ * c') := by
    have hD' : ∀ τ₁ τ₂ τ₃ : ℝ, D τ₁ τ₂ τ₃ =
        cexp (φ₁ * τ₁ + φ₂ * τ₂) * A.thermal (A.timeOrdered ![τ₁, τ₂] ![cᴴ, c]) *
          (cexp (φ₃ * τ₃) * A.thermal (A.timeOrdered ![τ₃, 0] ![c'ᴴ, c'])) := by
      intro τ₁ τ₂ τ₃
      simp only [hD]
      rw [Complex.exp_add]
      ring
    have h₃ : ∫ τ in (0 : ℝ)..A.β, cexp (φ₃ * τ) * A.thermal (A.timeOrdered ![τ, 0] ![c'ᴴ, c']) =
        ∫ τ in (0 : ℝ)..A.β, cexp (φ₃ * τ) * A.thermal (A.heis τ c'ᴴ * c') :=
      integral_congr_Ioo hβ fun τ h ↦ by rw [hT₁ τ h.1]
    simp_rw [hD', intervalIntegral.integral_const_mul, intervalIntegral.integral_mul_const]
    rw [h₃]
    congr 1
    rw [integral_square hβ
      (g₁₂ := fun τ₁ τ₂ ↦ cexp (φ₁ * τ₁ + φ₂ * τ₂) * A.thermal (A.heis τ₁ cᴴ * A.heis τ₂ c))
      (g₂₁ := fun τ₁ τ₂ ↦ -1 * (cexp (φ₂ * τ₂ + φ₁ * τ₁) * A.thermal (A.heis τ₂ c * A.heis τ₁ cᴴ)))
      (by fun_prop) (by fun_prop)
      (fun τ₁ τ₂ _ h _ ↦ by rw [A.timeOrdered_12 _ _ h])
      (fun τ₁ τ₂ _ h _ ↦ by rw [A.timeOrdered_21 _ _ h, thermal_neg, add_comm (φ₂ * _)]; ring),
      triangleIntegral_const_mul, triangleTerm, triangleTerm]
    ring
  unfold orderedTerm
  linear_combination hDsum - hfac

/-! ### Paths through the Fock states -/

/-- The rate `Eᵢ - Eⱼ + ψ` of the step `i ← j` (the left index is the later state). -/
noncomputable def rate (i j : Fin 4) (ψ : ℂ) : ℂ := (A.energy i : ℂ) - A.energy j + ψ

/-- The weighted simplex integral along the path `i₀ ← i₁ ← i₂ ← i₃ ← i₀` of Fock states. -/
noncomputable def pathTerm (ψ₁ ψ₂ ψ₃ : ℂ) (i₀ i₁ i₂ i₃ : Fin 4) : ℂ :=
  A.weight i₀ * simplexIntegral A.β fun s₁ s₂ s₃ ↦
    cexp (A.rate i₀ i₁ ψ₁ * s₁ + A.rate i₁ i₂ ψ₂ * s₂ + A.rate i₂ i₃ ψ₃ * s₃)

/-- The weighted triangle integral along the path `i₀ ← i₁ ← i₀`. -/
noncomputable def pathTerm₂ (ψ₁ ψ₂ : ℂ) (i₀ i₁ : Fin 4) : ℂ :=
  A.weight i₀ * triangleIntegral A.β fun s₁ s₂ ↦
    cexp (A.rate i₀ i₁ ψ₁ * s₁ + A.rate i₁ i₀ ψ₂ * s₂)

theorem orderedIntegrand_eq (ψ₁ ψ₂ ψ₃ : ℂ) (P₁ P₂ P₃ P₄ : Op) :
    A.orderedIntegrand ψ₁ ψ₂ ψ₃ P₁ P₂ P₃ P₄ = fun s₁ s₂ s₃ : ℝ ↦ ∑ i₀, ∑ i₁, ∑ i₂, ∑ i₃,
      P₁ i₀ i₁ * P₂ i₁ i₂ * P₃ i₂ i₃ * P₄ i₃ i₀ * (A.weight i₀ *
        cexp (A.rate i₀ i₁ ψ₁ * s₁ + A.rate i₁ i₂ ψ₂ * s₂ + A.rate i₂ i₃ ψ₃ * s₃)) := by
  funext s₁ s₂ s₃
  simp only [orderedIntegrand, thermal_eq, Matrix.mul_apply, heis_apply, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i₀ _ ↦ Finset.sum_congr rfl fun i₁ _ ↦
    Finset.sum_congr rfl fun i₂ _ ↦ Finset.sum_congr rfl fun i₃ _ ↦ ?_
  have h : cexp (A.rate i₀ i₁ ψ₁ * s₁ + A.rate i₁ i₂ ψ₂ * s₂ + A.rate i₂ i₃ ψ₃ * s₃) =
      cexp (ψ₁ * s₁ + ψ₂ * s₂ + ψ₃ * s₃) * cexp (s₁ * ((A.energy i₀ : ℂ) - A.energy i₁)) *
        cexp (s₂ * ((A.energy i₁ : ℂ) - A.energy i₂)) *
        cexp (s₃ * ((A.energy i₂ : ℂ) - A.energy i₃)) := by
    rw [← Complex.exp_add, ← Complex.exp_add, ← Complex.exp_add]
    congr 1
    simp only [rate]
    ring
  rw [h]
  ring

/-- An ordered term as a sum over the paths of Fock states. -/
theorem orderedTerm_eq (ψ₁ ψ₂ ψ₃ : ℂ) (P₁ P₂ P₃ P₄ : Op) :
    A.orderedTerm ψ₁ ψ₂ ψ₃ P₁ P₂ P₃ P₄ = ∑ i₀, ∑ i₁, ∑ i₂, ∑ i₃,
      P₁ i₀ i₁ * P₂ i₁ i₂ * P₃ i₂ i₃ * P₄ i₃ i₀ * A.pathTerm ψ₁ ψ₂ ψ₃ i₀ i₁ i₂ i₃ := by
  rw [orderedTerm, orderedIntegrand_eq, simplexIntegral_sum₄ _ fun _ _ _ _ ↦ by
    unfold Continuous₃
    fun_prop]
  simp only [simplexIntegral_const_mul, pathTerm]

/-- A triangle term as a sum over the paths of Fock states. -/
theorem triangleTerm_eq (ψ₁ ψ₂ : ℂ) (P₁ P₂ : Op) :
    A.triangleTerm ψ₁ ψ₂ P₁ P₂ = ∑ i₀, ∑ i₁, P₁ i₀ i₁ * P₂ i₁ i₀ * A.pathTerm₂ ψ₁ ψ₂ i₀ i₁ := by
  have h : (fun s₁ s₂ : ℝ ↦ cexp (ψ₁ * s₁ + ψ₂ * s₂) * A.thermal (A.heis s₁ P₁ * A.heis s₂ P₂)) =
      fun s₁ s₂ : ℝ ↦ ∑ i₀, ∑ i₁, P₁ i₀ i₁ * P₂ i₁ i₀ * (A.weight i₀ *
        cexp (A.rate i₀ i₁ ψ₁ * s₁ + A.rate i₁ i₀ ψ₂ * s₂)) := by
    funext s₁ s₂
    simp only [thermal_eq, Matrix.mul_apply, heis_apply, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i₀ _ ↦ Finset.sum_congr rfl fun i₁ _ ↦ ?_
    have h : cexp (A.rate i₀ i₁ ψ₁ * s₁ + A.rate i₁ i₀ ψ₂ * s₂) =
        cexp (ψ₁ * s₁ + ψ₂ * s₂) * cexp (s₁ * ((A.energy i₀ : ℂ) - A.energy i₁)) *
          cexp (s₂ * ((A.energy i₁ : ℂ) - A.energy i₀)) := by
      rw [← Complex.exp_add, ← Complex.exp_add]
      congr 1
      simp only [rate]
      ring
    rw [h]
    ring
  rw [triangleTerm, h, triangleIntegral_sum₂ _ fun _ _ ↦ by fun_prop]
  simp only [triangleIntegral_const_mul, pathTerm₂]

/-- A single-time integral as a sum over the paths of Fock states. -/
theorem lineTerm_eq (ψ : ℂ) (P₁ P₂ : Op) :
    ∫ τ in (0 : ℝ)..A.β, cexp (ψ * τ) * A.thermal (A.heis τ P₁ * P₂) =
      ∑ i₀, ∑ i₁, P₁ i₀ i₁ * P₂ i₁ i₀ * (A.weight i₀ * expInt (A.rate i₀ i₁ ψ) A.β) := by
  have h : (fun τ : ℝ ↦ cexp (ψ * τ) * A.thermal (A.heis τ P₁ * P₂)) =
      fun τ : ℝ ↦ ∑ i₀, ∑ i₁, P₁ i₀ i₁ * P₂ i₁ i₀ * (A.weight i₀ * cexp (A.rate i₀ i₁ ψ * τ)) := by
    funext τ
    simp only [thermal_eq, Matrix.mul_apply, heis_apply, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i₀ _ ↦ Finset.sum_congr rfl fun i₁ _ ↦ ?_
    have h : cexp (A.rate i₀ i₁ ψ * τ) =
        cexp (ψ * τ) * cexp (τ * ((A.energy i₀ : ℂ) - A.energy i₁)) := by
      rw [← Complex.exp_add]
      congr 1
      simp only [rate]
      ring
    rw [h]
    ring
  rw [h, intervalIntegral.integral_finsetSum fun i₀ _ ↦
    (by fun_prop : Continuous _).intervalIntegrable _ _]
  refine Finset.sum_congr rfl fun i₀ _ ↦ ?_
  rw [intervalIntegral.integral_finsetSum fun i₁ _ ↦
    (by fun_prop : Continuous _).intervalIntegrable _ _]
  refine Finset.sum_congr rfl fun i₁ _ ↦ ?_
  rw [intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul, integral_cexp_mul]

/-! ### One path in closed form -/

theorem weight_pos (i : Fin 4) : 0 < A.weight i := by
  fin_cases i <;> simp [weight, p₀_pos, p₁_pos, p₂_pos]

/-- `pᵢ e^{rβ} = ε pⱼ` for the rate `r` of the step `i ← j`, if `e^{ψβ} = ε`. -/
theorem weight_mul_rate {i j : Fin 4} {ψ ε : ℂ} (hψ : cexp (ψ * A.β) = ε) :
    (A.weight i : ℂ) * cexp (A.rate i j ψ * A.β) = ε * A.weight j := by
  rw [rate, add_mul, Complex.exp_add, ← mul_assoc, A.weight_mul_exp, hψ]
  ring

/-- A step with a fermionic frequency has a nonzero rate. -/
theorem rate_ne_zero {i j : Fin 4} {ψ : ℂ} (hψ : cexp (ψ * A.β) = -1) : A.rate i j ψ ≠ 0 := by
  intro h
  have h' := A.weight_mul_rate (i := i) (j := j) hψ
  rw [h, zero_mul, Complex.exp_zero, mul_one] at h'
  have h'' : ((A.weight i + A.weight j : ℝ) : ℂ) = 0 := by
    push_cast
    linear_combination h'
  have := ofReal_eq_zero.mp h''
  linarith [A.weight_pos i, A.weight_pos j]

theorem rate_add (i j l : Fin 4) (ψ ψ' : ℂ) :
    A.rate i j ψ + A.rate j l ψ' = A.rate i l (ψ + ψ') := by
  simp only [rate]
  ring

/-- One path of an ordered term, for fermionic `ψ₁, ψ₂, ψ₃`, through the weights. The rates
`r₁, r₂, r₃`, their partial sums `r₁₂, r₂₃, r₁₂₃`, and the conditions `C₁₂`, `C₂₃` for
`r₁₂ = 0`, `r₂₃ = 0` may be given in any form. -/
theorem pathTerm_eq (i₀ i₁ i₂ i₃ : Fin 4) {ψ₁ ψ₂ ψ₃ : ℂ} (hψ₁ : cexp (ψ₁ * A.β) = -1)
    (hψ₂ : cexp (ψ₂ * A.β) = -1) (hψ₃ : cexp (ψ₃ * A.β) = -1) {r₁ r₂ r₃ r₁₂ r₂₃ r₁₂₃ : ℂ}
    (e₁ : A.rate i₀ i₁ ψ₁ = r₁) (e₂ : A.rate i₁ i₂ ψ₂ = r₂) (e₃ : A.rate i₂ i₃ ψ₃ = r₃)
    (e₁₂ : r₁ + r₂ = r₁₂) (e₂₃ : r₂ + r₃ = r₂₃) (e₁₂₃ : r₁ + r₂ + r₃ = r₁₂₃)
    {C₁₂ C₂₃ : Prop} [Decidable C₁₂] [Decidable C₂₃] (c₁₂ : r₁₂ = 0 ↔ C₁₂)
    (c₂₃ : r₂₃ = 0 ↔ C₂₃) :
    A.pathTerm ψ₁ ψ₂ ψ₃ i₀ i₁ i₂ i₃ = simplexW A.β (A.weight i₀) (A.weight i₁) (A.weight i₂)
      (A.weight i₃) r₁ r₂ r₃ r₁₂ r₂₃ r₁₂₃ C₁₂ C₂₃ := by
  subst e₁ e₂ e₃ e₁₂ e₂₃ e₁₂₃
  have hψ₁₂ : cexp ((ψ₁ + ψ₂) * A.β) = 1 := by
    rw [add_mul, Complex.exp_add, hψ₁, hψ₂]
    norm_num
  have hψ₁₂₃ : cexp ((ψ₁ + ψ₂ + ψ₃) * A.β) = -1 := by
    rw [add_mul, Complex.exp_add, hψ₁₂, hψ₃, one_mul]
  rw [pathTerm, simplexIntegral_cexp (A.rate_ne_zero hψ₁) (A.rate_ne_zero hψ₂)
    (A.rate_ne_zero hψ₃)]
  refine mul_simplex3 rfl rfl rfl (by rw [A.weight_mul_rate hψ₁, neg_one_mul]) ?_ ?_
    (A.rate_ne_zero hψ₁) ?_ c₁₂ c₂₃
  · rw [rate_add, A.weight_mul_rate hψ₁₂, one_mul]
  · rw [rate_add, rate_add, A.weight_mul_rate hψ₁₂₃, neg_one_mul]
  · rw [rate_add, rate_add]
    exact A.rate_ne_zero hψ₁₂₃

/-- One path of a triangle term, through the weights. -/
theorem pathTerm₂_eq (i₀ i₁ : Fin 4) {ψ₁ ψ₂ : ℂ} (hψ₁ : cexp (ψ₁ * A.β) = -1)
    (hψ₂ : cexp (ψ₂ * A.β) = -1) {r₁ r₂ r₁₂ : ℂ} (e₁ : A.rate i₀ i₁ ψ₁ = r₁)
    (e₂ : A.rate i₁ i₀ ψ₂ = r₂) (e₁₂ : r₁ + r₂ = r₁₂) {C₁₂ : Prop} [Decidable C₁₂]
    (c₁₂ : r₁₂ = 0 ↔ C₁₂) :
    A.pathTerm₂ ψ₁ ψ₂ i₀ i₁ =
      triangleW A.β (A.weight i₀) (A.weight i₁) (A.weight i₀) r₁ r₂ r₁₂ C₁₂ := by
  subst e₁ e₂ e₁₂
  have hψ₁₂ : cexp ((ψ₁ + ψ₂) * A.β) = 1 := by
    rw [add_mul, Complex.exp_add, hψ₁, hψ₂]
    norm_num
  rw [pathTerm₂, triangleIntegral_cexp (A.rate_ne_zero hψ₂)]
  exact mul_simplex2 rfl (by rw [A.weight_mul_rate hψ₁, neg_one_mul])
    (by rw [rate_add, A.weight_mul_rate hψ₁₂, one_mul]) (A.rate_ne_zero hψ₁) c₁₂

/-- One path of a single-time integral, through the weights. -/
theorem pathTerm₁_eq (i₀ i₁ : Fin 4) {ψ : ℂ} (hψ : cexp (ψ * A.β) = -1) {r : ℂ}
    (e : A.rate i₀ i₁ ψ = r) :
    (A.weight i₀ : ℂ) * expInt (A.rate i₀ i₁ ψ) A.β = (-A.weight i₁ - A.weight i₀) / r := by
  rw [mul_expInt (by rw [A.weight_mul_rate hψ, neg_one_mul]) (A.rate_ne_zero hψ), e]

end Params

end HubbardAtom
