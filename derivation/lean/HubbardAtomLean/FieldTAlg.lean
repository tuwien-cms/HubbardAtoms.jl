import HubbardAtomLean.OrderedIntegrals

/-!
# The algebra of the transverse input

The six Fock-space paths of `χ_T↑` (`pathsT`) against the closed form of Proposition "transverse
input" (`rhsT`), over free variables: `X = xₙ`, `Y = xₘ`, `w = iω_k`, the field `h` and the weights.
Both sides are affine in the weights, so the identity splits into one identity per weight, which
keeps the rational expressions small (`pathsT_lin`, `rhsT_lin`).
-/

namespace HubbardAtom

/-- The six paths of `χ_T↑(νₙ, νₘ, ω_k)` through the weights; `C` says whether `n = m`. -/
noncomputable def pathsT (β a h X Y w p₀ pu pd p₂ : ℂ) (C : Prop) [Decidable C] : ℂ :=
  simplexW β pu p₀ pd p₀ (-X - a - h) (X + a - h + w) (-Y - a + h - w) (-2 * h + w) (X - Y)
      (-Y - a - h) False C -
    simplexW β p₂ pd p₀ pd (-X + a - h) (-Y - a + h - w) (X + a - h + w) (-X - Y - w) (X - Y)
      (-Y + a - h) False C +
    -simplexW β p₂ pu p₀ pd (-Y + a + h - w) (-X - a - h) (X + a - h + w) (-X - Y - w)
      (-2 * h + w) (-Y + a - h) False False +
    simplexW β pu p₂ pd p₀ (X - a - h + w) (-X + a - h) (-Y - a + h - w) (-2 * h + w)
      (-X - Y - w) (-Y - a - h) False False +
    simplexW β pu p₂ pu p₀ (X - a - h + w) (-Y + a + h - w) (-X - a - h) (X - Y) (-X - Y - w)
      (-Y - a - h) C False -
    simplexW β p₂ pu p₂ pd (-Y + a + h - w) (X - a - h + w) (-X + a - h) (X - Y) (-2 * h + w)
      (-Y + a - h) C False

theorem pathsT_lin (β a h X Y w p₀ pu pd p₂ : ℂ) (C : Prop) [Decidable C] :
    pathsT β a h X Y w p₀ pu pd p₂ C =
      p₀ * pathsT β a h X Y w 1 0 0 0 C + pu * pathsT β a h X Y w 0 1 0 0 C +
        pd * pathsT β a h X Y w 0 0 1 0 C + p₂ * pathsT β a h X Y w 0 0 0 1 C := by
  unfold pathsT simplexW
  split_ifs <;> ring

/-- `P(z) = Q₊(z)Q₋(z)` at `z = X + w/2`, through its linear factors. -/
noncomputable def PTf (a h X w : ℂ) : ℂ :=
  (X + a + h) * (X + w + a - h) * ((X - a + h) * (X + w - a - h))

/-- `H(z) = z² + 2aρz + q₁` with the constant terms multiplied by `s` (`= 1`), at `z = X + w/2`. -/
noncomputable def HTf (a h X w s p₀ pu pd p₂ : ℂ) : ℂ :=
  s * ((X + w / 2) ^ 2 + (a ^ 2 - (w - 2 * h) ^ 2 / 4)) + 2 * a * (p₂ - p₀) * (X + w / 2) -
    2 * a ^ 2 * (pu + pd) - a * (w - 2 * h) * (pu - pd)

/-- `N₂(v)`, homogenized, at `v = Y + w/2`. -/
noncomputable def N₂f (a h Y w s p₀ pu pd p₂ : ℂ) : ℂ :=
  2 * a * (s * ((Y + w / 2) ^ 2 + (a ^ 2 - (w - 2 * h) ^ 2 / 4)) +
      2 * a * (p₂ - p₀) * (Y + w / 2)) +
    -4 * a ^ 2 * (pu - pd) / (w - 2 * h) * ((Y + w / 2) ^ 2 - (a ^ 2 - (w - 2 * h) ^ 2 / 4))

/-- `N(z, v) = (v² + 2q₀ - 4a² + z²)N₂(v) - (2a + ζ)P(v)`, homogenized. -/
noncomputable def Nf (a h X Y w s p₀ pu pd p₂ : ℂ) : ℂ :=
  ((Y + w / 2) ^ 2 + 2 * (a ^ 2 - (w - 2 * h) ^ 2 / 4) - 4 * a ^ 2 + (X + w / 2) ^ 2) *
      N₂f a h Y w s p₀ pu pd p₂ -
    (2 * a * s + -4 * a ^ 2 * (pu - pd) / (w - 2 * h)) * PTf a h Y w

/-- The right side of Proposition "transverse input" off the diagonal, homogenized:
`λρ/(z + v) + N(z, v)/P(v)`. -/
noncomputable def rhsT (a h X Y w s p₀ pu pd p₂ : ℂ) : ℂ :=
  -4 * a ^ 2 * (p₂ - p₀) / (X + Y + w) + Nf a h X Y w s p₀ pu pd p₂ / PTf a h Y w

theorem rhsT_lin (a h X Y w s p₀ pu pd p₂ : ℂ) :
    rhsT a h X Y w s p₀ pu pd p₂ =
      s * rhsT a h X Y w 1 0 0 0 0 + p₀ * rhsT a h X Y w 0 1 0 0 0 +
        pu * rhsT a h X Y w 0 0 1 0 0 + pd * rhsT a h X Y w 0 0 0 1 0 +
        p₂ * rhsT a h X Y w 0 0 0 0 1 := by
  unfold rhsT Nf N₂f
  ring

theorem HTf_lin (a h X w s p₀ pu pd p₂ : ℂ) :
    HTf a h X w s p₀ pu pd p₂ =
      s * HTf a h X w 1 0 0 0 0 + p₀ * HTf a h X w 0 1 0 0 0 + pu * HTf a h X w 0 0 1 0 0 +
        pd * HTf a h X w 0 0 0 1 0 + p₂ * HTf a h X w 0 0 0 0 1 := by
  unfold HTf
  ring

/-! ### The identity, one weight at a time -/

section Basis

variable {β a h X Y w : ℂ}
  (d₁ : X + a + h ≠ 0) (d₂ : X - a + h ≠ 0) (d₃ : X + w + a - h ≠ 0) (d₄ : X + w - a - h ≠ 0)
  (hW : w - 2 * h ≠ 0)

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree in six variables
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- Off the diagonal, the weight `p₀`. -/
theorem pathsT_off₀ (e₁ : Y + a + h ≠ 0) (e₂ : Y - a + h ≠ 0) (e₃ : Y + w + a - h ≠ 0)
    (e₄ : Y + w - a - h ≠ 0) (hp : X + Y + w ≠ 0) (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsT β a h X Y w 1 0 0 0 False =
      rhsT a h X Y w 1 0 0 0 0 + rhsT a h X Y w 0 1 0 0 0 := by
  simp only [pathsT, simplexW, rhsT, Nf, N₂f, PTf, ite_false]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, div_neg]
  simp only [show -Y - a + h - w = -(Y + w + a - h) by ring,
    show -Y + a + h - w = -(Y + w - a - h) by ring, show -Y - a - h = -(Y + a + h) by ring,
    show -Y + a - h = -(Y - a + h) by ring, show -X - Y - w = -(X + Y + w) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree in six variables
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- On the diagonal, the weight `p₀`. -/
theorem pathsT_diag₀ (hp : X + X + w ≠ 0) :
    PTf a h X w * pathsT β a h X X w 1 0 0 0 True =
      (-β * HTf a h X w 1 0 0 0 0 + rhsT a h X X w 1 0 0 0 0) +
        (-β * HTf a h X w 0 1 0 0 0 + rhsT a h X X w 0 1 0 0 0) := by
  simp only [pathsT, simplexW, rhsT, Nf, N₂f, PTf, HTf, ite_true, ite_false, sub_self]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -X - X - w = -(X + X + w) by ring,
    show -2 * h + w = w - 2 * h by ring, div_neg]
  simp only [show -X - a + h - w = -(X + w + a - h) by ring,
    show -X + a + h - w = -(X + w - a - h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  -- `field_simp` would normalize `X + X + w` to a form it cannot discharge
  generalize hS : X + X + w = S at hp ⊢
  field_simp
  obtain rfl : w = S - X - X := by linear_combination hS
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree in six variables
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- Off the diagonal, the weight `p↑`. -/
theorem pathsT_offUp (e₁ : Y + a + h ≠ 0) (e₂ : Y - a + h ≠ 0) (e₃ : Y + w + a - h ≠ 0)
    (e₄ : Y + w - a - h ≠ 0) (hp : X + Y + w ≠ 0) (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsT β a h X Y w 0 1 0 0 False =
      rhsT a h X Y w 1 0 0 0 0 + rhsT a h X Y w 0 0 1 0 0 := by
  simp only [pathsT, simplexW, rhsT, Nf, N₂f, PTf, ite_false]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, div_neg]
  simp only [show -Y - a + h - w = -(Y + w + a - h) by ring,
    show -Y + a + h - w = -(Y + w - a - h) by ring, show -Y - a - h = -(Y + a + h) by ring,
    show -Y + a - h = -(Y - a + h) by ring, show -X - Y - w = -(X + Y + w) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree in six variables
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- On the diagonal, the weight `p↑`. -/
theorem pathsT_diagUp (hp : X + X + w ≠ 0) :
    PTf a h X w * pathsT β a h X X w 0 1 0 0 True =
      (-β * HTf a h X w 1 0 0 0 0 + rhsT a h X X w 1 0 0 0 0) +
        (-β * HTf a h X w 0 0 1 0 0 + rhsT a h X X w 0 0 1 0 0) := by
  simp only [pathsT, simplexW, rhsT, Nf, N₂f, PTf, HTf, ite_true, ite_false, sub_self]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -X - X - w = -(X + X + w) by ring,
    show -2 * h + w = w - 2 * h by ring, div_neg]
  simp only [show -X - a + h - w = -(X + w + a - h) by ring,
    show -X + a + h - w = -(X + w - a - h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  -- `field_simp` would normalize `X + X + w` to a form it cannot discharge
  generalize hS : X + X + w = S at hp ⊢
  field_simp
  obtain rfl : w = S - X - X := by linear_combination hS
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree in six variables
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- Off the diagonal, the weight `p↓`. -/
theorem pathsT_offDn (e₁ : Y + a + h ≠ 0) (e₂ : Y - a + h ≠ 0) (e₃ : Y + w + a - h ≠ 0)
    (e₄ : Y + w - a - h ≠ 0) (hp : X + Y + w ≠ 0) (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsT β a h X Y w 0 0 1 0 False =
      rhsT a h X Y w 1 0 0 0 0 + rhsT a h X Y w 0 0 0 1 0 := by
  simp only [pathsT, simplexW, rhsT, Nf, N₂f, PTf, ite_false]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, div_neg]
  simp only [show -Y - a + h - w = -(Y + w + a - h) by ring,
    show -Y + a + h - w = -(Y + w - a - h) by ring, show -Y - a - h = -(Y + a + h) by ring,
    show -Y + a - h = -(Y - a + h) by ring, show -X - Y - w = -(X + Y + w) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree in six variables
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- On the diagonal, the weight `p↓`. -/
theorem pathsT_diagDn (hp : X + X + w ≠ 0) :
    PTf a h X w * pathsT β a h X X w 0 0 1 0 True =
      (-β * HTf a h X w 1 0 0 0 0 + rhsT a h X X w 1 0 0 0 0) +
        (-β * HTf a h X w 0 0 0 1 0 + rhsT a h X X w 0 0 0 1 0) := by
  simp only [pathsT, simplexW, rhsT, Nf, N₂f, PTf, HTf, ite_true, ite_false, sub_self]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -X - X - w = -(X + X + w) by ring,
    show -2 * h + w = w - 2 * h by ring, div_neg]
  simp only [show -X - a + h - w = -(X + w + a - h) by ring,
    show -X + a + h - w = -(X + w - a - h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  -- `field_simp` would normalize `X + X + w` to a form it cannot discharge
  generalize hS : X + X + w = S at hp ⊢
  field_simp
  obtain rfl : w = S - X - X := by linear_combination hS
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree in six variables
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- Off the diagonal, the weight `p₂`. -/
theorem pathsT_off₂ (e₁ : Y + a + h ≠ 0) (e₂ : Y - a + h ≠ 0) (e₃ : Y + w + a - h ≠ 0)
    (e₄ : Y + w - a - h ≠ 0) (hp : X + Y + w ≠ 0) (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsT β a h X Y w 0 0 0 1 False =
      rhsT a h X Y w 1 0 0 0 0 + rhsT a h X Y w 0 0 0 0 1 := by
  simp only [pathsT, simplexW, rhsT, Nf, N₂f, PTf, ite_false]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, div_neg]
  simp only [show -Y - a + h - w = -(Y + w + a - h) by ring,
    show -Y + a + h - w = -(Y + w - a - h) by ring, show -Y - a - h = -(Y + a + h) by ring,
    show -Y + a - h = -(Y - a + h) by ring, show -X - Y - w = -(X + Y + w) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree in six variables
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- On the diagonal, the weight `p₂`. -/
theorem pathsT_diag₂ (hp : X + X + w ≠ 0) :
    PTf a h X w * pathsT β a h X X w 0 0 0 1 True =
      (-β * HTf a h X w 1 0 0 0 0 + rhsT a h X X w 1 0 0 0 0) +
        (-β * HTf a h X w 0 0 0 0 1 + rhsT a h X X w 0 0 0 0 1) := by
  simp only [pathsT, simplexW, rhsT, Nf, N₂f, PTf, HTf, ite_true, ite_false, sub_self]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -X - X - w = -(X + X + w) by ring,
    show -2 * h + w = w - 2 * h by ring, div_neg]
  simp only [show -X - a + h - w = -(X + w + a - h) by ring,
    show -X + a + h - w = -(X + w - a - h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  -- `field_simp` would normalize `X + X + w` to a form it cannot discharge
  generalize hS : X + X + w = S at hp ⊢
  field_simp
  obtain rfl : w = S - X - X := by linear_combination hS
  ring

end Basis

theorem pathsT_of_true {β a h X Y w p₀ pu pd p₂ : ℂ} {C : Prop} [Decidable C] (hC : C) :
    pathsT β a h X Y w p₀ pu pd p₂ C = pathsT β a h X Y w p₀ pu pd p₂ True := by
  simp only [pathsT, simplexW, hC, ite_true]

theorem pathsT_of_false {β a h X Y w p₀ pu pd p₂ : ℂ} {C : Prop} [Decidable C] (hC : ¬C) :
    pathsT β a h X Y w p₀ pu pd p₂ C = pathsT β a h X Y w p₀ pu pd p₂ False := by
  simp only [pathsT, simplexW, hC, ite_false]

/-- **The transverse input, off the diagonal**: the six paths of `χ_T↑` times `P(zₙ)` are
`λρ/(zₙ + zₘ) + N(zₙ, zₘ)/P(zₘ)` (`n ≠ m`). -/
theorem pathsT_offdiag {β a h X Y w p₀ pu pd p₂ : ℂ} (hs : p₀ + pu + pd + p₂ = 1)
    (d₁ : X + a + h ≠ 0) (d₂ : X - a + h ≠ 0) (d₃ : X + w + a - h ≠ 0) (d₄ : X + w - a - h ≠ 0)
    (e₁ : Y + a + h ≠ 0) (e₂ : Y - a + h ≠ 0) (e₃ : Y + w + a - h ≠ 0) (e₄ : Y + w - a - h ≠ 0)
    (hW : w - 2 * h ≠ 0) (hp : X + Y + w ≠ 0) (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsT β a h X Y w p₀ pu pd p₂ False = rhsT a h X Y w 1 p₀ pu pd p₂ := by
  rw [pathsT_lin, rhsT_lin]
  linear_combination p₀ * pathsT_off₀ (β := β) d₁ d₂ d₃ d₄ hW e₁ e₂ e₃ e₄ hp hXY +
    pu * pathsT_offUp (β := β) d₁ d₂ d₃ d₄ hW e₁ e₂ e₃ e₄ hp hXY +
    pd * pathsT_offDn (β := β) d₁ d₂ d₃ d₄ hW e₁ e₂ e₃ e₄ hp hXY +
    p₂ * pathsT_off₂ (β := β) d₁ d₂ d₃ d₄ hW e₁ e₂ e₃ e₄ hp hXY +
    rhsT a h X Y w 1 0 0 0 0 * hs

/-- **The transverse input, on the diagonal**: `-βH(zₙ) + λρ/(2zₙ) + N(zₙ, zₙ)/P(zₙ)`. -/
theorem pathsT_diagonal {β a h X w p₀ pu pd p₂ : ℂ} (hs : p₀ + pu + pd + p₂ = 1)
    (d₁ : X + a + h ≠ 0) (d₂ : X - a + h ≠ 0) (d₃ : X + w + a - h ≠ 0) (d₄ : X + w - a - h ≠ 0)
    (hW : w - 2 * h ≠ 0) (hp : X + X + w ≠ 0) :
    PTf a h X w * pathsT β a h X X w p₀ pu pd p₂ True =
      -β * HTf a h X w 1 p₀ pu pd p₂ + rhsT a h X X w 1 p₀ pu pd p₂ := by
  rw [pathsT_lin, rhsT_lin, HTf_lin]
  linear_combination p₀ * pathsT_diag₀ (β := β) d₁ d₂ d₃ d₄ hW hp +
    pu * pathsT_diagUp (β := β) d₁ d₂ d₃ d₄ hW hp +
    pd * pathsT_diagDn (β := β) d₁ d₂ d₃ d₄ hW hp +
    p₂ * pathsT_diag₂ (β := β) d₁ d₂ d₃ d₄ hW hp +
    (-β * HTf a h X w 1 0 0 0 0 + rhsT a h X X w 1 0 0 0 0) * hs

/-! ### The pair resonance at `μ = 0` -/

/-- The six paths of `χ_T↑` with the condition `D` for a vanishing pair rate `xₙ + xₘ + w = 0`,
which happens at `μ = 0` on the mirror `m = n̄`. -/
noncomputable def pathsTD (β a h X Y w p₀ pu pd p₂ : ℂ) (C D : Prop) [Decidable C] [Decidable D] :
    ℂ :=
  simplexW β pu p₀ pd p₀ (-X - a - h) (X + a - h + w) (-Y - a + h - w) (-2 * h + w) (X - Y)
      (-Y - a - h) False C -
    simplexW β p₂ pd p₀ pd (-X + a - h) (-Y - a + h - w) (X + a - h + w) (-X - Y - w) (X - Y)
      (-Y + a - h) D C +
    -simplexW β p₂ pu p₀ pd (-Y + a + h - w) (-X - a - h) (X + a - h + w) (-X - Y - w)
      (-2 * h + w) (-Y + a - h) D False +
    simplexW β pu p₂ pd p₀ (X - a - h + w) (-X + a - h) (-Y - a + h - w) (-2 * h + w)
      (-X - Y - w) (-Y - a - h) False D +
    simplexW β pu p₂ pu p₀ (X - a - h + w) (-Y + a + h - w) (-X - a - h) (X - Y) (-X - Y - w)
      (-Y - a - h) C D -
    simplexW β p₂ pu p₂ pd (-Y + a + h - w) (X - a - h + w) (-X + a - h) (X - Y) (-2 * h + w)
      (-Y + a - h) C False

theorem pathsT_eq_pathsTD (β a h X Y w p₀ pu pd p₂ : ℂ) (C : Prop) [Decidable C] :
    pathsT β a h X Y w p₀ pu pd p₂ C = pathsTD β a h X Y w p₀ pu pd p₂ C False := rfl

theorem pathsTD_congr (β a h X Y w p₀ pu pd p₂ : ℂ) {C C' D D' : Prop} [Decidable C]
    [Decidable C'] [Decidable D] [Decidable D'] (hC : C ↔ C') (hD : D ↔ D') :
    pathsTD β a h X Y w p₀ pu pd p₂ C D = pathsTD β a h X Y w p₀ pu pd p₂ C' D' := by
  simp only [pathsTD, simplexW, hC, hD]

theorem pathsTD_lin (β a h X Y w p₀ pu pd : ℂ) (C D : Prop) [Decidable C] [Decidable D] :
    pathsTD β a h X Y w p₀ pu pd p₀ C D =
      p₀ * pathsTD β a h X Y w 1 0 0 1 C D + pu * pathsTD β a h X Y w 0 1 0 0 C D +
        pd * pathsTD β a h X Y w 0 0 1 0 C D := by
  unfold pathsTD simplexW
  split_ifs <;> ring

/-- The right side on the mirror at `μ = 0` (`ρ = 0`), homogenized:
`λβp₀ + N(z, v)/P(v)`, `v = -z`. -/
noncomputable def rhsTM (β a h X Y w s p₀ pu pd : ℂ) : ℂ :=
  -4 * a ^ 2 * (β * p₀) + Nf a h X Y w s p₀ pu pd p₀ / PTf a h Y w

theorem rhsTM_lin (β a h X Y w s p₀ pu pd : ℂ) :
    rhsTM β a h X Y w s p₀ pu pd =
      s * rhsTM β a h X Y w 1 0 0 0 + p₀ * rhsTM β a h X Y w 0 1 0 0 +
        pu * rhsTM β a h X Y w 0 0 1 0 + pd * rhsTM β a h X Y w 0 0 0 1 := by
  unfold rhsTM Nf N₂f
  ring

section Mirror

variable {β a h X Y w : ℂ}
  (d₁ : X + a + h ≠ 0) (d₂ : X - a + h ≠ 0) (d₃ : X + w + a - h ≠ 0) (d₄ : X + w - a - h ≠ 0)
  (hW : w - 2 * h ≠ 0) (hY : X + Y + w = 0)

include d₁ d₂ d₃ d₄ hY in
/-- The nonvanishing factors in `Y = -X - w`. -/
private theorem mirror_factors :
    Y + a + h ≠ 0 ∧ Y - a + h ≠ 0 ∧ Y + w + a - h ≠ 0 ∧ Y + w - a - h ≠ 0 := by
  obtain rfl : Y = -X - w := by linear_combination hY
  exact ⟨fun h0 ↦ d₄ (by linear_combination -h0), fun h0 ↦ d₃ (by linear_combination -h0),
    fun h0 ↦ d₂ (by linear_combination -h0), fun h0 ↦ d₁ (by linear_combination -h0)⟩

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW hY in
/-- On the mirror (`n ≠ n̄`), the weight `p₀`. -/
theorem pathsTD_mirror₀ (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsTD β a h X Y w 1 0 0 1 False True =
      2 * rhsTM β a h X Y w 1 0 0 0 + rhsTM β a h X Y w 0 1 0 0 := by
  obtain ⟨e₁, e₂, e₃, e₄⟩ := mirror_factors d₁ d₂ d₃ d₄ hY
  simp only [pathsTD, simplexW, rhsTM, Nf, N₂f, PTf, ite_false, ite_true]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, show -Y - a + h - w = -(Y + w + a - h) by ring,
    show -Y + a + h - w = -(Y + w - a - h) by ring, show -Y - a - h = -(Y + a + h) by ring,
    show -Y + a - h = -(Y - a + h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  obtain rfl : Y = -X - w := by linear_combination hY
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW hY in
/-- On the mirror (`n ≠ n̄`), the weight `pUp`. -/
theorem pathsTD_mirrorUp (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsTD β a h X Y w 0 1 0 0 False True =
      rhsTM β a h X Y w 1 0 0 0 + rhsTM β a h X Y w 0 0 1 0 := by
  obtain ⟨e₁, e₂, e₃, e₄⟩ := mirror_factors d₁ d₂ d₃ d₄ hY
  simp only [pathsTD, simplexW, rhsTM, Nf, N₂f, PTf, ite_false, ite_true]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, show -Y - a + h - w = -(Y + w + a - h) by ring,
    show -Y + a + h - w = -(Y + w - a - h) by ring, show -Y - a - h = -(Y + a + h) by ring,
    show -Y + a - h = -(Y - a + h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  obtain rfl : Y = -X - w := by linear_combination hY
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW hY in
/-- On the mirror (`n ≠ n̄`), the weight `pDn`. -/
theorem pathsTD_mirrorDn (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsTD β a h X Y w 0 0 1 0 False True =
      rhsTM β a h X Y w 1 0 0 0 + rhsTM β a h X Y w 0 0 0 1 := by
  obtain ⟨e₁, e₂, e₃, e₄⟩ := mirror_factors d₁ d₂ d₃ d₄ hY
  simp only [pathsTD, simplexW, rhsTM, Nf, N₂f, PTf, ite_false, ite_true]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, show -Y - a + h - w = -(Y + w + a - h) by ring,
    show -Y + a + h - w = -(Y + w - a - h) by ring, show -Y - a - h = -(Y + a + h) by ring,
    show -Y + a - h = -(Y - a + h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  obtain rfl : Y = -X - w := by linear_combination hY
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- At a mirror fixed point (`X = Y = -w/2`), the weight `p₀`. -/
theorem pathsTD_fixed₀ (hX : X + X + w = 0) :
    PTf a h X w * pathsTD β a h X X w 1 0 0 1 True True =
      -β * (2 * HTf a h X w 1 0 0 0 0 + HTf a h X w 0 1 0 0 0 + HTf a h X w 0 0 0 0 1) +
        (2 * rhsTM β a h X X w 1 0 0 0 + rhsTM β a h X X w 0 1 0 0) := by
  simp only [pathsTD, simplexW, rhsTM, Nf, N₂f, PTf, HTf, ite_true, ite_false, sub_self]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, show -X - a + h - w = -(X + w + a - h) by ring,
    show -X + a + h - w = -(X + w - a - h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  obtain rfl : w = -X - X := by linear_combination hX
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- At a mirror fixed point (`X = Y = -w/2`), the weight `pUp`. -/
theorem pathsTD_fixedUp (hX : X + X + w = 0) :
    PTf a h X w * pathsTD β a h X X w 0 1 0 0 True True =
      -β * (HTf a h X w 1 0 0 0 0 + HTf a h X w 0 0 1 0 0) +
        (rhsTM β a h X X w 1 0 0 0 + rhsTM β a h X X w 0 0 1 0) := by
  simp only [pathsTD, simplexW, rhsTM, Nf, N₂f, PTf, HTf, ite_true, ite_false, sub_self]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, show -X - a + h - w = -(X + w + a - h) by ring,
    show -X + a + h - w = -(X + w - a - h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  obtain rfl : w = -X - X := by linear_combination hX
  ring

set_option maxHeartbeats 4000000 in
-- the six paths give rational expressions of high degree
set_option maxRecDepth 20000 in
include d₁ d₂ d₃ d₄ hW in
/-- At a mirror fixed point (`X = Y = -w/2`), the weight `pDn`. -/
theorem pathsTD_fixedDn (hX : X + X + w = 0) :
    PTf a h X w * pathsTD β a h X X w 0 0 1 0 True True =
      -β * (HTf a h X w 1 0 0 0 0 + HTf a h X w 0 0 0 1 0) +
        (rhsTM β a h X X w 1 0 0 0 + rhsTM β a h X X w 0 0 0 1) := by
  simp only [pathsTD, simplexW, rhsTM, Nf, N₂f, PTf, HTf, ite_true, ite_false, sub_self]
  simp only [show -X - a - h = -(X + a + h) by ring, show -X + a - h = -(X - a + h) by ring,
    show X + a - h + w = X + w + a - h by ring, show X - a - h + w = X + w - a - h by ring,
    show -2 * h + w = w - 2 * h by ring, show -X - a + h - w = -(X + w + a - h) by ring,
    show -X + a + h - w = -(X + w - a - h) by ring, div_neg]
  have hW₂ : w - h * 2 ≠ 0 := fun h0 ↦ hW (by linear_combination h0)
  field_simp
  obtain rfl : w = -X - X := by linear_combination hX
  ring

end Mirror

/-- **The transverse input at a mirror fixed point at `μ = 0`**:
`P(0)χ_T↑,nn = -βH(0) + λβp₀ + N(0, 0)/P(0)`. -/
theorem pathsTD_fixed {β a h X w p₀ pu pd : ℂ} (hs : 2 * p₀ + pu + pd = 1)
    (d₁ : X + a + h ≠ 0) (d₂ : X - a + h ≠ 0) (d₃ : X + w + a - h ≠ 0) (d₄ : X + w - a - h ≠ 0)
    (hW : w - 2 * h ≠ 0) (hX : X + X + w = 0) :
    PTf a h X w * pathsTD β a h X X w p₀ pu pd p₀ True True =
      -β * HTf a h X w 1 p₀ pu pd p₀ + rhsTM β a h X X w 1 p₀ pu pd := by
  rw [pathsTD_lin, HTf_lin, rhsTM_lin]
  linear_combination p₀ * pathsTD_fixed₀ (β := β) d₁ d₂ d₃ d₄ hW hX +
    pu * pathsTD_fixedUp (β := β) d₁ d₂ d₃ d₄ hW hX +
    pd * pathsTD_fixedDn (β := β) d₁ d₂ d₃ d₄ hW hX +
    (-β * HTf a h X w 1 0 0 0 0 + rhsTM β a h X X w 1 0 0 0) * hs

/-- **The transverse input on the mirror at `μ = 0`**: `P(zₙ)χ_T↑,nn̄ = λβp₀ + N(zₙ, zₙ̄)/P(zₙ̄)`
(`n ≠ n̄`, `ρ = 0`). -/
theorem pathsTD_mirror {β a h X Y w p₀ pu pd : ℂ} (hs : 2 * p₀ + pu + pd = 1)
    (d₁ : X + a + h ≠ 0) (d₂ : X - a + h ≠ 0) (d₃ : X + w + a - h ≠ 0) (d₄ : X + w - a - h ≠ 0)
    (hW : w - 2 * h ≠ 0) (hY : X + Y + w = 0) (hXY : X - Y ≠ 0) :
    PTf a h X w * pathsTD β a h X Y w p₀ pu pd p₀ False True = rhsTM β a h X Y w 1 p₀ pu pd := by
  rw [pathsTD_lin, rhsTM_lin]
  linear_combination p₀ * pathsTD_mirror₀ (β := β) d₁ d₂ d₃ d₄ hW hY hXY +
    pu * pathsTD_mirrorUp (β := β) d₁ d₂ d₃ d₄ hW hY hXY +
    pd * pathsTD_mirrorDn (β := β) d₁ d₂ d₃ d₄ hW hY hXY +
    rhsTM β a h X Y w 1 0 0 0 * hs

/-! ### The resonance of the transverse transfer, `W = 0` -/

/-- The six paths of `χ_T↑` with the condition `E` for a vanishing transfer `W = w - 2h` (at
`h = 0`, `k = 0`). -/
noncomputable def pathsTW (β a h X Y w p₀ pu pd p₂ : ℂ) (C D E : Prop) [Decidable C] [Decidable D]
    [Decidable E] : ℂ :=
  simplexW β pu p₀ pd p₀ (-X - a - h) (X + a - h + w) (-Y - a + h - w) (-2 * h + w) (X - Y)
      (-Y - a - h) E C -
    simplexW β p₂ pd p₀ pd (-X + a - h) (-Y - a + h - w) (X + a - h + w) (-X - Y - w) (X - Y)
      (-Y + a - h) D C +
    -simplexW β p₂ pu p₀ pd (-Y + a + h - w) (-X - a - h) (X + a - h + w) (-X - Y - w)
      (-2 * h + w) (-Y + a - h) D E +
    simplexW β pu p₂ pd p₀ (X - a - h + w) (-X + a - h) (-Y - a + h - w) (-2 * h + w)
      (-X - Y - w) (-Y - a - h) E D +
    simplexW β pu p₂ pu p₀ (X - a - h + w) (-Y + a + h - w) (-X - a - h) (X - Y) (-X - Y - w)
      (-Y - a - h) C D -
    simplexW β p₂ pu p₂ pd (-Y + a + h - w) (X - a - h + w) (-X + a - h) (X - Y) (-2 * h + w)
      (-Y + a - h) C E

theorem pathsTD_eq_pathsTW (β a h X Y w p₀ pu pd p₂ : ℂ) (C D : Prop) [Decidable C]
    [Decidable D] :
    pathsTD β a h X Y w p₀ pu pd p₂ C D = pathsTW β a h X Y w p₀ pu pd p₂ C D False := rfl

theorem pathsTW_congr (β a h X Y w p₀ pu pd p₂ : ℂ) {C C' D D' E E' : Prop} [Decidable C]
    [Decidable C'] [Decidable D] [Decidable D'] [Decidable E] [Decidable E'] (hC : C ↔ C')
    (hD : D ↔ D') (hE : E ↔ E') :
    pathsTW β a h X Y w p₀ pu pd p₂ C D E = pathsTW β a h X Y w p₀ pu pd p₂ C' D' E' := by
  simp only [pathsTW, simplexW, hC, hD, hE]

/-- The zero-field magnetic structure at `k = 0` (`ε = -1`, `w = 0`, `q₀ = a²`), in free
variables: `-βH(X)δ + λρ/(X + Y) + N(X, Y)/P(Y)` with the static term `ζ`. -/
noncomputable def rhsMag (β a X Y p₀ p₁ p₂ : ℂ) (C : Prop) [Decidable C] : ℂ :=
  (if C then -β * (X ^ 2 + 2 * a * (p₂ - p₀) * X + (a ^ 2 - 4 * a ^ 2 * p₁)) else 0) +
    -4 * a ^ 2 * (p₂ - p₀) / (X + Y) +
    ((Y ^ 2 + 2 * a ^ 2 - 4 * a ^ 2 + X ^ 2) *
        (2 * a * (Y ^ 2 + 2 * a * (p₂ - p₀) * Y + a ^ 2) +
          β * a ^ 2 * (1 - (p₂ - p₀) ^ 2 - 4 * (p₀ * p₂ - p₁ ^ 2)) * (Y ^ 2 - a ^ 2)) +
      (-2 * a - β * a ^ 2 * (1 - (p₂ - p₀) ^ 2 - 4 * (p₀ * p₂ - p₁ ^ 2))) *
        ((Y + a) * (Y + a) * ((Y - a) * (Y - a)))) /
      ((Y + a) * (Y + a) * ((Y - a) * (Y - a)))

set_option maxHeartbeats 4000000 in
-- six paths at a vanishing transfer: rational expressions of high degree
set_option maxRecDepth 20000 in
/-- **The transverse paths at `h = 0`, `k = 0`** are the zero-field magnetic structure, off the
pair resonance. -/
theorem pathsTW_W0 {β a X Y p₀ p₁ p₂ : ℂ} (hs : p₀ + 2 * p₁ + p₂ = 1) (d₁ : X + a ≠ 0)
    (d₂ : X - a ≠ 0) (e₁ : Y + a ≠ 0) (e₂ : Y - a ≠ 0) (hp : X + Y ≠ 0) (C : Prop) [Decidable C]
    (hC : C ↔ X = Y) :
    PTf a 0 X 0 * pathsTW β a 0 X Y 0 p₀ p₁ p₁ p₂ C False True = rhsMag β a X Y p₀ p₁ p₂ C := by
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  by_cases hXY : X = Y
  · subst hXY
    have hC' : C := hC.mpr rfl
    simp only [pathsTW, simplexW, rhsMag, PTf, hC', ite_true, ite_false, sub_self, add_zero,
      sub_zero, mul_zero]
    simp only [show -X - a = -(X + a) by ring, show -X + a = -(X - a) by ring, div_neg]
    have hX0 : X ≠ 0 := fun h0 ↦ hp (by rw [h0, add_zero])
    field_simp
    ring
  · have hC' : ¬C := fun h0 ↦ hXY (hC.mp h0)
    have hXY' : X - Y ≠ 0 := sub_ne_zero.mpr hXY
    simp only [pathsTW, simplexW, rhsMag, PTf, hC', ite_true, ite_false, add_zero, sub_zero,
      mul_zero]
    simp only [show -X - a = -(X + a) by ring, show -X + a = -(X - a) by ring,
      show -Y - a = -(Y + a) by ring, show -Y + a = -(Y - a) by ring,
      show -X - Y = -(X + Y) by ring, div_neg]
    field_simp
    ring

/-- The zero-field magnetic structure at `k = 0` and `μ = 0` on the mirror `Y = -X`:
`λβp₀ + N(X, -X)/P(-X)` (`ρ = 0`, `p₂ = p₀`). -/
noncomputable def rhsMagM (β a X p₀ p₁ : ℂ) : ℂ :=
  -4 * a ^ 2 * (β * p₀) +
    ((X ^ 2 + 2 * a ^ 2 - 4 * a ^ 2 + X ^ 2) *
        (2 * a * (X ^ 2 + a ^ 2) + β * a ^ 2 * (1 - 4 * (p₀ * p₀ - p₁ ^ 2)) * (X ^ 2 - a ^ 2)) +
      (-2 * a - β * a ^ 2 * (1 - 4 * (p₀ * p₀ - p₁ ^ 2))) *
        ((X + a) * (X + a) * ((X - a) * (X - a)))) /
      ((X + a) * (X + a) * ((X - a) * (X - a)))

set_option maxHeartbeats 4000000 in
-- six paths at a vanishing transfer: rational expressions of high degree
set_option maxRecDepth 20000 in
/-- **The transverse paths at `h = 0`, `k = 0`, `μ = 0` on the mirror**: the zero-field magnetic
structure, where the pair rate vanishes. -/
theorem pathsTW_W0_mirror {β a X p₀ p₁ : ℂ} (hs : 2 * p₀ + 2 * p₁ = 1) (d₁ : X + a ≠ 0)
    (d₂ : X - a ≠ 0) (hX : X ≠ 0) :
    PTf a 0 X 0 * pathsTW β a 0 X (-X) 0 p₀ p₁ p₁ p₀ False True True = rhsMagM β a X p₀ p₁ := by
  obtain rfl : p₁ = (1 - 2 * p₀) / 2 := by linear_combination hs / 2
  simp only [pathsTW, simplexW, rhsMagM, PTf, ite_true, ite_false, add_zero, sub_zero, mul_zero,
    neg_neg]
  simp only [show -X - a = -(X + a) by ring, show -X + a = -(X - a) by ring,
    show X - -X = X + X by ring, div_neg]
  field_simp
  ring

end HubbardAtom
