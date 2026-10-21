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

end HubbardAtom
