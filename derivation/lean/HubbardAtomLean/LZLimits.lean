import HubbardAtomLean.DMContinuity
import HubbardAtomLean.SingletContinuity

/-!
# The values at the lattice zeros as limits of the closed forms

Items 1 and 4 of Proposition "special" give the vertex at a lattice zero by Eqs. (Gamma-s-lz) and
(Gamma-hf-lz). `SingletLZ.lean` and `HalfFillingLZ.lean` show that these are inverses; here they are
shown to be the limits of Eqs. (Gamma-s) and (Gamma-hf) along any approach through points where
these hold, possibly an empty one. Along a family `(U, β, μ)(x)` continuous at `x₀`, with a lattice
zero at `x₀`:

* `tendsto_gammaS_of_SingletLZ`: `gammaS → gammaSlz` along any approach to `x₀` through points
  where the assumptions of Theorem "singlet" hold;
* `tendsto_gammaHF_of_HFLZ`: `gammaHF → gammaHFlz` along any approach through points where those
  of Corollary "half filling" hold.

Likewise, at half filling the closed form of Eq. (Gamma-ph) tends to that of Eq. (Gamma-hf)
(`tendsto_gammaR_of_HFHyp`), and to Eq. (Gamma-hf-lz) at a lattice zero
(`tendsto_gammaR_of_HFLZ`), along any approach through points where the assumptions of Theorem
"density and magnetic" hold.

Each closed form equals the operator vertex where it holds (`SingletHyp.gammaOpS_eq`, …), and the
operator vertex is continuous at `x₀` (`continuousAt_gammaOpS`, `continuousAt_gammaOpR`). That
such approaches exist, i.e. that the assumptions hold at points arbitrarily close to `x₀`, is not
shown. The theorems hold for every filter `l ≤ 𝓝 x₀`, `l = ⊥` included, where they say nothing,
so they do not prove that the values are the limits.
-/

open Filter Topology

namespace HubbardAtom

namespace Params

variable {X : Type*} [TopologicalSpace X] {U β μ : X → ℝ} {hβ : ∀ x, 0 < β x} {x₀ : X}
variable (k : ℤ) (hU : ContinuousAt U x₀) (hb : ContinuousAt β x₀) (hm : ContinuousAt μ x₀)
variable {l : Filter X} (hl : l ≤ 𝓝 x₀)

include hU hb hm hl

/-- **Item 1 of Proposition "special" as a limit**: at a lattice zero `t_c + Aₛ = 0` at `x₀`,
`gammaS → gammaSlz` along `l → x₀` through points where the assumptions of Theorem "singlet"
hold (with `ξ(x)` there). The approach may be empty: that such points exist near `x₀` is not
shown. -/
theorem tendsto_gammaS_of_SingletLZ {c : ℤ} (h₀ : (fam U β μ hβ x₀).SingletLZ k c) {ξ : X → ℂ}
    (h : ∀ᶠ x in l, (fam U β μ hβ x).SingletHyp k (ξ x)) (n m : ℤ) :
    Tendsto (fun x ↦ (fam U β μ hβ x).gammaS k (ξ x) n m) l
      (𝓝 ((fam U β μ hβ x₀).gammaSlz k c n m)) := by
  rw [← h₀.gammaOpS_eq]
  exact ((continuousAt_gammaOpS k hU hb hm h₀.isUnit_onePlusKS n m).tendsto.mono_left hl).congr'
    (h.mono fun x hx ↦ hx.gammaOpS_eq n m)

variable (ε : ℂ)

/-- **Item 4 of Proposition "special" as a limit**: at a lattice zero `z_c² = ŝ` at `x₀`,
`gammaHF → gammaHFlz` along `l → x₀` through points where the assumptions of Corollary
"half filling" hold (with `û(x)` there). The approach may be empty: that such points exist near
`x₀` is not shown. -/
theorem tendsto_gammaHF_of_HFLZ {c : ℤ} (h₀ : (fam U β μ hβ x₀).HFLZ k ε c) {u : X → ℂ}
    (h : ∀ᶠ x in l, (fam U β μ hβ x).HFHyp k ε (u x)) (n m : ℤ) :
    Tendsto (fun x ↦ (fam U β μ hβ x).gammaHF k ε (u x) n m) l
      (𝓝 ((fam U β μ hβ x₀).gammaHFlz k ε c n m)) := by
  rw [← h₀.gammaOpR_eq]
  exact ((continuousAt_gammaOpR k ε hU hb hm h₀.isUnit_onePlusKR n m).tendsto.mono_left
    hl).congr' (h.mono fun x hx ↦ hx.gammaOpR_eq n m)

/-- **Corollary "half filling" as a limit**: under its assumptions at `x₀`, `gammaR → gammaHF`
along `l → x₀` through points where the assumptions of Theorem "density and magnetic" hold
(with `u₁(x)`, `u₂(x)` there). The approach may be empty: that such points exist near `x₀` is
not shown. -/
theorem tendsto_gammaR_of_HFHyp {u₀ : ℂ} (h₀ : (fam U β μ hβ x₀).HFHyp k ε u₀) {u₁ u₂ : X → ℂ}
    (h : ∀ᶠ x in l, (fam U β μ hβ x).DMGeneric k ε (u₁ x) (u₂ x)) (n m : ℤ) :
    Tendsto (fun x ↦ (fam U β μ hβ x).gammaR k ε (u₁ x) (u₂ x) n m) l
      (𝓝 ((fam U β μ hβ x₀).gammaHF k ε u₀ n m)) := by
  rw [← h₀.gammaOpR_eq]
  exact ((continuousAt_gammaOpR k ε hU hb hm h₀.isUnit_onePlusKR n m).tendsto.mono_left
    hl).congr' (h.mono fun x hx ↦ hx.gammaOpR_eq n m)

/-- At a lattice zero `z_c² = ŝ` of Corollary "half filling" at `x₀`, `gammaR → gammaHFlz` along
`l → x₀` through points where the assumptions of Theorem "density and magnetic" hold. The
approach may be empty: that such points exist near `x₀` is not shown. -/
theorem tendsto_gammaR_of_HFLZ {c : ℤ} (h₀ : (fam U β μ hβ x₀).HFLZ k ε c) {u₁ u₂ : X → ℂ}
    (h : ∀ᶠ x in l, (fam U β μ hβ x).DMGeneric k ε (u₁ x) (u₂ x)) (n m : ℤ) :
    Tendsto (fun x ↦ (fam U β μ hβ x).gammaR k ε (u₁ x) (u₂ x) n m) l
      (𝓝 ((fam U β μ hβ x₀).gammaHFlz k ε c n m)) := by
  rw [← h₀.gammaOpR_eq]
  exact ((continuousAt_gammaOpR k ε hU hb hm h₀.isUnit_onePlusKR n m).tendsto.mono_left
    hl).congr' (h.mono fun x hx ↦ hx.gammaOpR_eq n m)

end Params

end HubbardAtom
