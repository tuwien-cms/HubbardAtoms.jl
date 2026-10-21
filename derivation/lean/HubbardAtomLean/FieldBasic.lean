import HubbardAtomLean.Basic

/-!
# The atom with doping and Zeeman field: parameters and weights

Section "Doping and Zeeman field" (App. F.6) of the note: the atom
`H = U (n↑ - 1/2)(n↓ - 1/2) - μ (n↑ + n↓ - 1) - h (n↑ - n↓)` at inverse temperature `β > 0`.

* `FParams`: the parameters `U`, `β`, `μ`, `h`.
* `FParams.p₀`, `pUp`, `pDn`, `p₂`: the Boltzmann weights of `|0⟩`, `|↑⟩`, `|↓⟩`, `|↑↓⟩`, with
  energies `U/4 + μ`, `-U/4 - h`, `-U/4 + h`, `U/4 - μ`; `ρ = p₂ - p₀`, `mf = p↑ - p↓`.
* `FParams.lat c`: the zero-field atom `(U, β, c)`, whose Matsubara lattice `z_n = c + iπ(2n+1+k)/β`
  and thermal sum `S_k` (centered at `c`) are those of the field sectors.
* `FParams.shiba`, `FParams.flip`: the parameters `(-U, β, h, μ)` of the particle–hole (Shiba) image
  and `(U, β, μ, -h)` of the spin flip, with their weights.
-/

open Complex

namespace HubbardAtom

/-- Parameters of the Hubbard atom in a field: interaction `U`, inverse temperature `β > 0`,
chemical potential `μ` (measured from half filling) and Zeeman field `h` (splitting `2h`). -/
structure FParams where
  /-- the Hubbard interaction -/
  U : ℝ
  /-- the inverse temperature -/
  β : ℝ
  /-- the chemical potential, measured from half filling -/
  μ : ℝ
  /-- the Zeeman field -/
  h : ℝ
  β_pos : 0 < β

namespace FParams

variable (F : FParams)

/-- `a = U/2`. -/
noncomputable def a : ℝ := F.U / 2

/-- The Boltzmann factor of `|0⟩`, whose energy is `U/4 + μ`. -/
noncomputable def boltz₀ : ℝ := Real.exp (-F.β * (F.U / 4 + F.μ))

/-- The Boltzmann factor of `|↑⟩`, whose energy is `-U/4 - h`. -/
noncomputable def boltzUp : ℝ := Real.exp (-F.β * (-F.U / 4 - F.h))

/-- The Boltzmann factor of `|↓⟩`, whose energy is `-U/4 + h`. -/
noncomputable def boltzDn : ℝ := Real.exp (-F.β * (-F.U / 4 + F.h))

/-- The Boltzmann factor of `|↑↓⟩`, whose energy is `U/4 - μ`. -/
noncomputable def boltz₂ : ℝ := Real.exp (-F.β * (F.U / 4 - F.μ))

/-- The partition function. -/
noncomputable def Z : ℝ := F.boltz₀ + F.boltzUp + F.boltzDn + F.boltz₂

/-- The weight of `|0⟩`. -/
noncomputable def p₀ : ℝ := F.boltz₀ / F.Z

/-- The weight of `|↑⟩`. -/
noncomputable def pUp : ℝ := F.boltzUp / F.Z

/-- The weight of `|↓⟩`. -/
noncomputable def pDn : ℝ := F.boltzDn / F.Z

/-- The weight of `|↑↓⟩`. -/
noncomputable def p₂ : ℝ := F.boltz₂ / F.Z

/-- `ρ = p₂ - p₀ = ⟨n↑ + n↓⟩ - 1`. -/
noncomputable def ρ : ℝ := F.p₂ - F.p₀

/-- `m = p↑ - p↓ = ⟨n↑ - n↓⟩`. -/
noncomputable def mf : ℝ := F.pUp - F.pDn

theorem Z_pos : 0 < F.Z := by
  unfold Z boltz₀ boltzUp boltzDn boltz₂
  positivity

theorem p₀_pos : 0 < F.p₀ := div_pos (Real.exp_pos _) F.Z_pos

theorem pUp_pos : 0 < F.pUp := div_pos (Real.exp_pos _) F.Z_pos

theorem pDn_pos : 0 < F.pDn := div_pos (Real.exp_pos _) F.Z_pos

theorem p₂_pos : 0 < F.p₂ := div_pos (Real.exp_pos _) F.Z_pos

/-- The weights are normalized. -/
theorem weights_sum : F.p₀ + F.pUp + F.pDn + F.p₂ = 1 := by
  have hZ : F.Z ≠ 0 := F.Z_pos.ne'
  calc F.p₀ + F.pUp + F.pDn + F.p₂ = F.Z / F.Z := by
        unfold p₀ pUp pDn p₂ Z
        ring
    _ = 1 := div_self hZ

theorem weights_sum' : (F.p₀ : ℂ) + F.pUp + F.pDn + F.p₂ = 1 := by
  exact_mod_cast F.weights_sum

/-- The ratio of two Boltzmann factors. -/
private theorem div_mul_exp {e₁ e₂ c : ℝ} (h : -F.β * e₂ = -F.β * e₁ + c) :
    Real.exp (-F.β * e₂) / F.Z = Real.exp (-F.β * e₁) / F.Z * Real.exp c := by
  rw [div_mul_eq_mul_div, ← Real.exp_add, h]

/-- `p↑/p₀ = e^{β(a + μ + h)}`. -/
theorem pUp_eq_p₀_mul : F.pUp = F.p₀ * Real.exp (F.β * (F.a + F.μ + F.h)) :=
  F.div_mul_exp (by unfold a; ring)

/-- `p↓/p₀ = e^{β(a + μ - h)}`. -/
theorem pDn_eq_p₀_mul : F.pDn = F.p₀ * Real.exp (F.β * (F.a + F.μ - F.h)) :=
  F.div_mul_exp (by unfold a; ring)

/-- `p↑/p₂ = e^{β(a - μ + h)}`. -/
theorem pUp_eq_p₂_mul : F.pUp = F.p₂ * Real.exp (F.β * (F.a - F.μ + F.h)) :=
  F.div_mul_exp (by unfold a; ring)

/-- `p↓/p₂ = e^{β(a - μ - h)}`. -/
theorem pDn_eq_p₂_mul : F.pDn = F.p₂ * Real.exp (F.β * (F.a - F.μ - F.h)) :=
  F.div_mul_exp (by unfold a; ring)

/-- `p₂/p₀ = e^{2βμ}`. -/
theorem p₂_eq_p₀_mul : F.p₂ = F.p₀ * Real.exp (2 * F.β * F.μ) :=
  F.div_mul_exp (by ring)

/-- `p↑/p↓ = e^{2βh}`. -/
theorem pUp_eq_pDn_mul : F.pUp = F.pDn * Real.exp (2 * F.β * F.h) :=
  F.div_mul_exp (by ring)

/-! ### The lattice -/

/-- The zero-field atom `(U, β, c)`: its lattice `z_n = c + iπ(2n + 1 + k)/β` and its thermal sum
`S_k`, centered at `c`, are those of the field sectors (its weights are not used). -/
def lat (c : ℝ) : Params := ⟨F.U, F.β, c, F.β_pos⟩

@[simp] theorem lat_U (c : ℝ) : (F.lat c).U = F.U := rfl
@[simp] theorem lat_β (c : ℝ) : (F.lat c).β = F.β := rfl
@[simp] theorem lat_μ (c : ℝ) : (F.lat c).μ = c := rfl
@[simp] theorem lat_a (c : ℝ) : (F.lat c).a = F.a := rfl

/-- The fermionic Matsubara frequency `ν_n = (2n + 1)π/β`. -/
noncomputable def ν (n : ℤ) : ℝ := (F.lat 0).ν n

/-- The bosonic Matsubara frequency `ω_k = 2kπ/β`. -/
noncomputable def ω (k : ℤ) : ℝ := (F.lat 0).ω k

@[simp] theorem lat_ν (c : ℝ) (n : ℤ) : (F.lat c).ν n = F.ν n := rfl
@[simp] theorem lat_ω (c : ℝ) (k : ℤ) : (F.lat c).ω k = F.ω k := rfl

/-! ### The particle–hole (Shiba) image and the spin flip -/

/-- The Shiba image `(-U, β, h, μ)`: `c↓ ↦ c↓†` maps the atom `(U, μ, h)` to `(-U, h, μ)`. -/
def shiba : FParams := ⟨-F.U, F.β, F.h, F.μ, F.β_pos⟩

/-- The spin flip `(U, β, μ, -h)`. -/
def flip : FParams := ⟨F.U, F.β, F.μ, -F.h, F.β_pos⟩

@[simp] theorem shiba_U : F.shiba.U = -F.U := rfl
@[simp] theorem shiba_β : F.shiba.β = F.β := rfl
@[simp] theorem shiba_μ : F.shiba.μ = F.h := rfl
@[simp] theorem shiba_h : F.shiba.h = F.μ := rfl
@[simp] theorem flip_U : F.flip.U = F.U := rfl
@[simp] theorem flip_β : F.flip.β = F.β := rfl
@[simp] theorem flip_μ : F.flip.μ = F.μ := rfl
@[simp] theorem flip_h : F.flip.h = -F.h := rfl

theorem shiba_Z : F.shiba.Z = F.Z := by
  simp only [Z, boltz₀, boltzUp, boltzDn, boltz₂, shiba_U, shiba_β, shiba_μ, shiba_h]
  rw [show -F.β * (-F.U / 4 + F.h) = -F.β * (-F.U / 4 + F.h) by ring,
    show -F.β * (-(-F.U) / 4 - F.μ) = -F.β * (F.U / 4 - F.μ) by ring,
    show -F.β * (-(-F.U) / 4 + F.μ) = -F.β * (F.U / 4 + F.μ) by ring,
    show -F.β * (-F.U / 4 - F.h) = -F.β * (-F.U / 4 - F.h) by ring]
  ring

/-- The Shiba image permutes the states: `(p₀, p↑, p↓, p₂) ↦ (p↓, p₂, p₀, p↑)`. -/
theorem shiba_p₀ : F.shiba.p₀ = F.pDn := by
  rw [p₀, pDn, shiba_Z, boltz₀, boltzDn]
  congr 2

theorem shiba_pUp : F.shiba.pUp = F.p₂ := by
  rw [pUp, p₂, shiba_Z, boltzUp, boltz₂]
  congr 2
  simp only [shiba_U, shiba_β, shiba_h]
  ring

theorem shiba_pDn : F.shiba.pDn = F.p₀ := by
  rw [pDn, p₀, shiba_Z, boltzDn, boltz₀]
  congr 2
  simp only [shiba_U, shiba_β, shiba_h]
  ring

theorem shiba_p₂ : F.shiba.p₂ = F.pUp := by
  rw [p₂, pUp, shiba_Z, boltz₂, boltzUp]
  congr 2

theorem flip_Z : F.flip.Z = F.Z := by
  simp only [Z, boltz₀, boltzUp, boltzDn, boltz₂, flip_U, flip_β, flip_μ, flip_h]
  ring_nf

/-- The spin flip exchanges `p↑` and `p↓`. -/
theorem flip_p₀ : F.flip.p₀ = F.p₀ := by
  rw [p₀, p₀, flip_Z, boltz₀, boltz₀]
  rfl

theorem flip_pUp : F.flip.pUp = F.pDn := by
  rw [pUp, pDn, flip_Z, boltzUp, boltzDn]
  congr 2
  simp only [flip_U, flip_β, flip_h]
  ring

theorem flip_pDn : F.flip.pDn = F.pUp := by
  rw [pDn, pUp, flip_Z, boltzDn, boltzUp]
  congr 2

theorem flip_p₂ : F.flip.p₂ = F.p₂ := by
  rw [p₂, p₂, flip_Z, boltz₂, boltz₂]
  rfl

end FParams

end HubbardAtom
