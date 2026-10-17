import HubbardAtomLean.Basic

/-!
# The identities behind the Float64 setup of HubbardAtoms.jl

`DopedHubbardAtom` in HubbardAtoms.jl (`src/doped.jl`, PR #12) evaluates the closed forms of
Theorem "density and magnetic" and Theorem "singlet" in floating point. To avoid cancellation it
replaces the naive formulas by equivalent ones. This file proves that the replacements listed below
are exact identities. (It does not prove anything about rounding.) Not covered: `_Y_atomic` (the
other root of the static magnetic channel), the two branches of `_tanhdiff`, and `_Q` on the
diagonal `z = v` (`Q_eq` assumes `z ≠ v`).

The scalars of one transfer and channel are bundled in `PHData`: `a = U/2`, the transfer
`w = iω_k`, the weights `p₀, p₁, p₂` with `p₀ + 2p₁ + p₂ = 1`, and `ε = ±1` (density / magnetic).
The package computes in `ℂ` with real `ω`, so `w² = -ω²`.

* `disc_eq`, `D_eq_naive`: the discriminant of `e` from the weights (`D` in `_ph_setup_raw`).
* `e_eq_mul`, `roots_mul`, `sigma_mul`, `e₀_eq_mul`: the roots of `e` and the products used to
  compute the smaller root from the larger one.
* `offsets_mul` and `tp_sq_add_m` … `e_w_sq_mul`: the offsets `d = s - T` of the roots from
  `T = t₊², t₋², a², w²/4`, which are the roots of `d² + 2(T + m)d + e(T) = 0` with `T + m` and
  `e(T)` written without cancellation (`_offsets`); `κ_of_density`, `κ_of_magnetic`,
  `h_of_magnetic` for the package's case split.
* `P_eq_offsets`, `sub_eq_sq_div`, `efactor`, `H_neg`: `P(u)` as a product of offsets,
  `δ = u - w/2`, the factors `z² - s = (x - δ)(x + w + δ)` of the entries (`_efactors`), and
  `H(-z)`.
* `X_mul_Xp`, `A_mul_B`, `K_div_rho`: the identity `AB = -ρM` at a root and the package's
  alternative form of `K/ρ` (`κa`).
* `K_static_density_pos`, `K_static_density_neg`: the alternative form of `K` at `k = 0` for the
  density channel (both nearest zeros `±a` of `P`).
* `Q_eq`: the divided-difference form of `Q(z, v)` (`_Q`).
-/

namespace HubbardAtom

namespace Stable

/-- The scalars of the particle–hole setup at one transfer and channel, in a field `K`. -/
structure PHData (K : Type*) [Field K] where
  /-- `a = U/2` -/
  a : K
  /-- the bosonic transfer `w = iω_k` -/
  w : K
  /-- the weight of `|0⟩` -/
  p₀ : K
  /-- the weight of each singly occupied state -/
  p₁ : K
  /-- the weight of `|↑↓⟩` -/
  p₂ : K
  /-- `+1` for the density, `-1` for the magnetic channel -/
  ε : K
  weights_sum : p₀ + 2 * p₁ + p₂ = 1
  ε_eq : ε = 1 ∨ ε = -1

namespace PHData

variable {K : Type*} [Field K] [CharZero K] (d : PHData K)

/-- `ρ = p₂ - p₀`. -/
def ρ : K := d.p₂ - d.p₀

/-- `q₊q₋ = (p₁ + p₂)(p₀ + p₁)`. -/
def Q : K := (d.p₁ + d.p₂) * (d.p₀ + d.p₁)

/-- `q₁ = q₀ + 4εa²p₁` with `q₀ = a² - w²/4`. -/
def q₁ : K := d.a ^ 2 - d.w ^ 2 / 4 + 4 * d.ε * d.a ^ 2 * d.p₁

/-- `m = q₁ - 2a²ρ²`. -/
def m : K := d.q₁ - 2 * d.a ^ 2 * d.ρ ^ 2

/-- `e₀ = q₁² - 16a⁴p₀p₂`. -/
def e₀ : K := d.q₁ ^ 2 - 16 * d.a ^ 4 * d.p₀ * d.p₂

/-- The quartic of Eq. (Delta) as a polynomial in `s = z²`: `e(s) = s² + 2ms + e₀`. -/
def e (s : K) : K := s ^ 2 + 2 * d.m * s + d.e₀

/-- `H(z) = z² + 2aρz + q₁`. -/
def H (z : K) : K := z ^ 2 + 2 * d.a * d.ρ * z + d.q₁

/-- `P(z) = Q₊(z) Q₋(z)`, `Q±(z) = (z ± a)² - w²/4`. -/
def P (z : K) : K := ((z + d.a) ^ 2 - d.w ^ 2 / 4) * ((z - d.a) ^ 2 - d.w ^ 2 / 4)

/-- The discriminant of `e` over `4a²`, as the package computes it from the weights. -/
def D : K :=
  4 * d.a ^ 2 * (4 * d.p₀ * d.p₂ * d.Q + d.ρ ^ 2 * d.p₁ * (d.p₁ - 1 - d.ε)) + d.ρ ^ 2 * d.w ^ 2 / 4

/-- `κ = q₊q₋ + εw₁₁`, `w₁₁ = p₁² - p₀p₂`. -/
def κ : K := d.Q + d.ε * (d.p₁ ^ 2 - d.p₀ * d.p₂)

/-- `h = 4a²(2q₊q₋ + εp₁)` (the package's `h`, not the thermal `h` of the note). -/
def h : K := 4 * d.a ^ 2 * (2 * d.Q + d.ε * d.p₁)

/-- `c = 2(1 + ε)a`. -/
def c : K := 2 * (1 + d.ε) * d.a

/-- The zero `t₊ = a + w/2` of `Q₋`. -/
def tp : K := d.a + d.w / 2

/-- The zero `t₋ = -a + w/2` of `Q₊`. -/
def tm : K := -d.a + d.w / 2

/-! ### The discriminant and the roots -/

/-- The discriminant of `e`: `m² - e₀ = 4a²D`. -/
theorem disc_eq : d.m ^ 2 - d.e₀ = 4 * d.a ^ 2 * d.D := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only [m, e₀, q₁, D, ρ, Q]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  ring

/-- `D` equals the naive form `a²ρ⁴ - ρ²q₁ + 4a²p₀p₂`, which cancels at small `p₁` or `p₀p₂`. -/
theorem D_eq_naive : d.D = d.a ^ 2 * d.ρ ^ 4 - d.ρ ^ 2 * d.q₁ + 4 * d.a ^ 2 * d.p₀ * d.p₂ := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only [D, ρ, Q, q₁]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  ring

/-- `e(s) = (s + m)² - 4a²D`. -/
theorem e_eq_sq_sub (s : K) : d.e s = (s + d.m) ^ 2 - 4 * d.a ^ 2 * d.D := by
  rw [← d.disc_eq, e]
  ring

/-- The roots `s₁,₂ = -m ± ζ` of `e`, for either square root `ζ` of `4a²D`. -/
theorem e_eq_mul {ζ : K} (hζ : ζ ^ 2 = 4 * d.a ^ 2 * d.D) (s : K) :
    d.e s = (s - (-d.m + ζ)) * (s - (-d.m - ζ)) := by
  rw [d.e_eq_sq_sub, ← hζ]
  ring

/-- Vieta: `s₁s₂ = e₀`, which gives the smaller root from the larger one. -/
theorem roots_mul {ζ : K} (hζ : ζ ^ 2 = 4 * d.a ^ 2 * d.D) :
    (-d.m + ζ) * (-d.m - ζ) = d.e₀ := by
  have h := d.disc_eq
  linear_combination h - hζ

/-- `σ = s + q₁ = 2a²ρ² ± ζ` at the roots, with `σ₁σ₂ = 4a²(ρ²q₁ - 4a²p₀p₂)`. -/
theorem sigma_mul {ζ : K} (hζ : ζ ^ 2 = 4 * d.a ^ 2 * d.D) :
    (2 * d.a ^ 2 * d.ρ ^ 2 + ζ) * (2 * d.a ^ 2 * d.ρ ^ 2 - ζ) =
      4 * d.a ^ 2 * (d.ρ ^ 2 * d.q₁ - 4 * d.a ^ 2 * d.p₀ * d.p₂) := by
  have h := d.D_eq_naive
  linear_combination (-1 : K) * hζ - 4 * d.a ^ 2 * h

omit [CharZero K] in
theorem sigma_eq (ζ : K) : -d.m + ζ + d.q₁ = 2 * d.a ^ 2 * d.ρ ^ 2 + ζ := by
  rw [m]
  ring

omit [CharZero K] in
/-- `e₀ = (q₁ - 4a²r)(q₁ + 4a²r)` for `r² = p₀p₂` (in the package, `r = √(p₀p₂)`). -/
theorem e₀_eq_mul {r : K} (hr : r ^ 2 = d.p₀ * d.p₂) :
    d.e₀ = (d.q₁ - 4 * d.a ^ 2 * r) * (d.q₁ + 4 * d.a ^ 2 * r) := by
  rw [e₀]
  linear_combination (16 * d.a ^ 4) * hr

/-! ### Offsets of the roots -/

/-- The offsets `s - T` of the roots `s = -m ± ζ` from a point `T` are `-(T + m) ± ζ`, the roots of
`d² + 2(T + m)d + e(T) = 0`; their product is `e(T)`. -/
theorem offsets_mul {ζ : K} (hζ : ζ ^ 2 = 4 * d.a ^ 2 * d.D) (T : K) :
    (-(T + d.m) + ζ) * (-(T + d.m) - ζ) = d.e T := by
  rw [d.e_eq_sq_sub, ← hζ]
  ring

omit [CharZero K] in
theorem offset_eq (ζ T : K) : -d.m + ζ - T = -(T + d.m) + ζ := by ring

theorem tp_sq_add_m : d.tp ^ 2 + d.m = d.h + d.a * d.w := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only [tp, m, q₁, h, ρ, Q]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  ring

theorem tm_sq_add_m : d.tm ^ 2 + d.m = d.h - d.a * d.w := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only [tm, m, q₁, h, ρ, Q]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  ring

theorem a_sq_add_m : d.a ^ 2 + d.m = d.h - d.w ^ 2 / 4 := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only [m, q₁, h, ρ, Q]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  ring

theorem w_sq_add_m : d.w ^ 2 / 4 + d.m = d.a ^ 2 * (1 + 4 * d.ε * d.p₁ - 2 * d.ρ ^ 2) := by
  simp only [m, q₁]
  ring

theorem e_tp_sq : d.e (d.tp ^ 2) = 8 * d.a ^ 2 * (d.w + d.c) * (d.a * d.κ + d.Q * d.w / 2) := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only [e, tp, m, e₀, q₁, c, κ, ρ, Q]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  rcases hε with rfl | rfl <;> ring

theorem e_tm_sq : d.e (d.tm ^ 2) = 8 * d.a ^ 2 * (d.w - d.c) * (d.Q * d.w / 2 - d.a * d.κ) := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only [e, tm, m, e₀, q₁, c, κ, ρ, Q]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  rcases hε with rfl | rfl <;> ring

theorem e_a_sq : d.e (d.a ^ 2) =
    (-d.w ^ 2 / 4 + 4 * (1 + d.ε) * d.a ^ 2 * d.p₁) *
      (-d.w ^ 2 / 4 + 4 * d.a ^ 2 * (1 - (1 - d.ε) * d.p₁)) := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only [e, m, e₀, q₁, ρ]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  rcases hε with rfl | rfl <;> ring

theorem e_w_sq : d.e (d.w ^ 2 / 4) =
    d.a ^ 4 * (1 + 4 * d.ε * d.p₁) ^ 2 - 16 * d.a ^ 4 * d.p₀ * d.p₂ -
      d.a ^ 2 * d.ρ ^ 2 * d.w ^ 2 := by
  simp only [e, m, e₀, q₁]
  ring

/-- The package's form of `e(w²/4)`, with `r = √(p₀p₂)`. -/
theorem e_w_sq_mul {r : K} (hr : r ^ 2 = d.p₀ * d.p₂) :
    d.e (d.w ^ 2 / 4) = d.a ^ 4 * (1 + 4 * d.ε * d.p₁ - 4 * r) * (1 + 4 * d.ε * d.p₁ + 4 * r) -
      d.a ^ 2 * d.ρ ^ 2 * d.w ^ 2 := by
  rw [d.e_w_sq]
  linear_combination (16 * d.a ^ 4) * hr

omit [CharZero K] in
theorem κ_of_density (hd : d.ε = 1) : d.κ = d.p₁ := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only at hd
  subst hd
  simp only [κ, Q]
  linear_combination p₁ * hs

omit [CharZero K] in
theorem κ_of_magnetic (hd : d.ε = -1) : d.κ = d.p₀ * d.p₁ + 2 * d.p₀ * d.p₂ + d.p₁ * d.p₂ := by
  rw [κ, Q, hd]
  ring

omit [CharZero K] in
theorem h_of_magnetic (hd : d.ε = -1) : d.h = 4 * d.a ^ 2 * d.κ := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only at hd
  subst hd
  simp only [h, κ, Q]
  linear_combination (4 * a ^ 2 * p₁) * hs

/-! ### `P`, `δ`, and the entries -/

/-- `P(u) = (u² - t₊²)(u² - t₋²)`: the package takes `P` at a root as a product of two offsets. -/
theorem P_eq_offsets (u : K) : d.P u = (u ^ 2 - d.tp ^ 2) * (u ^ 2 - d.tm ^ 2) := by
  simp only [P, tp, tm]
  ring

omit [CharZero K] in
/-- `u - t = (u² - t²)/(u + t)`: the package's `η = u - t±` and, for `t = w/2`, `δ = u - w/2`. -/
theorem sub_eq_sq_div {u t : K} (h : u + t ≠ 0) : u - t = (u ^ 2 - t ^ 2) / (u + t) := by
  field_simp
  ring

/-- `z² - u² = (x - δ)(x + w + δ)` with `x = z - w/2`, `δ = u - w/2` (`_efactors`). -/
theorem efactor (z u : K) :
    z ^ 2 - u ^ 2 = (z - d.w / 2 - (u - d.w / 2)) * (z - d.w / 2 + d.w + (u - d.w / 2)) := by
  ring

/-- `H(-z) = x(x + w) + a²(1 + 4εp₁) - 2aρz` with `x = z - w/2` (the diagonal term of `_Γraw`). -/
theorem H_neg (z : K) :
    d.H (-z) = (z - d.w / 2) * (z - d.w / 2 + d.w) + d.a ^ 2 * (1 + 4 * d.ε * d.p₁) -
      2 * d.a * d.ρ * z := by
  simp only [H, q₁]
  ring

omit [CharZero K] in
/-- `e(z²) = H(z)H(-z) - 16a⁴p₀p₂`, Eq. (Delta). -/
theorem e_sq_eq (z : K) : d.e (z ^ 2) = d.H z * d.H (-z) - 16 * d.a ^ 4 * d.p₀ * d.p₂ := by
  simp only [e, H, m, e₀]
  ring

/-! ### `K` and `N` at a root -/

/-- `λ = 4εa²`. -/
def lam : K := 4 * d.ε * d.a ^ 2

/-- `c̄ = β(p₀ + p₂)/2`. -/
def cbar (β : K) : K := β * (d.p₀ + d.p₂) / 2

/-- `X = βH(-u)`. -/
def X (β u : K) : K := β * d.H (-u)

/-- `A = X - λc̄`. -/
def At (β u : K) : K := d.X β u - d.lam * d.cbar β

/-- `B = X + λc̄`. -/
def Bt (β u : K) : K := d.X β u + d.lam * d.cbar β

/-- `M = 4aβuX + λ²β²ρ/4`. -/
def M (β u : K) : K := 4 * d.a * β * u * d.X β u + d.lam ^ 2 * β ^ 2 * d.ρ / 4

omit [CharZero K] in
theorem lam_sq : d.lam ^ 2 = 16 * d.a ^ 4 := by
  rcases d.ε_eq with h | h <;> simp only [lam, h] <;> ring

omit [CharZero K] in
/-- At a root `u` of `e`: `βH(-u) βH(u) = 16a⁴β²p₀p₂`, so the package computes the smaller of
`X = βH(-u)`, `X₊ = βH(u)` from the larger. -/
theorem X_mul_Xp {β u : K} (hu : d.e (u ^ 2) = 0) :
    d.X β u * (β * d.H u) = 16 * d.a ^ 4 * β ^ 2 * d.p₀ * d.p₂ := by
  have h := d.e_sq_eq u
  rw [hu] at h
  rw [X]
  linear_combination (-β ^ 2) * h

/-- At a root: `AB = -ρM` (proof of Corollary "half filling" of the note), so the smaller of `A`,
`B` is computed as `-ρM` over the larger. -/
theorem A_mul_B {β u : K} (hu : d.e (u ^ 2) = 0) : d.At β u * d.Bt β u = -d.ρ * d.M β u := by
  have hX := d.X_mul_Xp (β := β) hu
  have hlam := d.lam_sq
  have hXp : β * d.H u = d.X β u + 4 * d.a * β * d.ρ * u := by
    simp only [X, H]
    ring
  rw [hXp] at hX
  simp only [At, Bt, M, cbar, ρ] at hX ⊢
  linear_combination hX + (β ^ 2 * (d.p₂ - d.p₀) ^ 2 / 4 - β ^ 2 * (d.p₀ + d.p₂) ^ 2 / 4) * hlam

/-- The algebraic core of `K_div_rho`, cross-multiplied: if `B² - 2XB = ρM`,
`M = 4aβuX + λ²β²ρ/4`, `λ² = 16a⁴` and `u★² = a²`, then
`-M(B + 2aβρu) = B(-2aβ(u - u★)(B - 2aβρu★) - 2aβu★(B + 2aβρu))`. -/
theorem K_div_rho_core {a β u ρ X B M lam us : K} (hBB : B ^ 2 - 2 * X * B = ρ * M)
    (hM : M = 4 * a * β * u * X + lam ^ 2 * β ^ 2 * ρ / 4) (hlam : lam ^ 2 = 16 * a ^ 4)
    (hs : us ^ 2 = a ^ 2) :
    -M * (B + 2 * a * β * ρ * u) =
      B * (-2 * a * β * (u - us) * (B - 2 * a * β * ρ * us) -
        2 * a * β * us * (B + 2 * a * β * ρ * u)) := by
  linear_combination (2 * a * β * u) * hBB - B * hM + (4 * a ^ 2 * β ^ 2 * ρ * B) * hs -
    (β ^ 2 * ρ * B / 4) * hlam

/-- The package's alternative form of `K/ρ` (`κa`): with `K = A + λρS`, `S = -(β/2)τ`
(`τ = tanh y`), for either `u★` with `u★² = a²`,
`K/ρ = -2aβ (s - a²)/(u + u★) · (B - 2aβρu★)/(B + 2aβρu) - 2aβ(u★ + εaτ)`, `s = u²`.
Choosing `u★ = ∓εa` nearer to `u`, nothing cancels at low temperature, where `τ → ±1`. -/
theorem K_div_rho {β u τ ustar : K} (hu : d.e (u ^ 2) = 0) (hs : ustar ^ 2 = d.a ^ 2)
    (hρ : d.ρ ≠ 0) (hB : d.Bt β u ≠ 0) (hBu : d.Bt β u + 2 * d.a * β * d.ρ * u ≠ 0)
    (hus : u + ustar ≠ 0) :
    (d.At β u + d.lam * d.ρ * (-(β / 2) * τ)) / d.ρ =
      -2 * d.a * β * (u ^ 2 - d.a ^ 2) / (u + ustar) *
          ((d.Bt β u - 2 * d.a * β * d.ρ * ustar) / (d.Bt β u + 2 * d.a * β * d.ρ * u)) -
        2 * d.a * β * (ustar + d.ε * d.a * τ) := by
  have hAB := d.A_mul_B (β := β) hu
  -- `A = 2X - B`, so `AB = -ρM` reads `B² - 2XB = ρM`
  have hA2 : d.At β u = 2 * d.X β u - d.Bt β u := by
    simp only [At, Bt]
    ring
  have hBB : d.Bt β u ^ 2 - 2 * d.X β u * d.Bt β u = d.ρ * d.M β u := by
    rw [hA2] at hAB
    linear_combination (-1 : K) * hAB
  have core := K_div_rho_core hBB (by rw [M]) d.lam_sq hs
  have hA : d.At β u = -(d.ρ * d.M β u) / d.Bt β u := by
    rw [eq_div_iff hB]
    linear_combination hAB
  have e₁ : (d.At β u + d.lam * d.ρ * (-(β / 2) * τ)) / d.ρ =
      -d.M β u / d.Bt β u - d.lam * β * τ / 2 := by
    rw [hA]
    field_simp
    ring
  have e₂ : -2 * d.a * β * (u ^ 2 - d.a ^ 2) / (u + ustar) *
        ((d.Bt β u - 2 * d.a * β * d.ρ * ustar) / (d.Bt β u + 2 * d.a * β * d.ρ * u)) =
      -2 * d.a * β * (u - ustar) * (d.Bt β u - 2 * d.a * β * d.ρ * ustar) /
        (d.Bt β u + 2 * d.a * β * d.ρ * u) := by
    rw [div_mul_div_comm, div_eq_div_iff (mul_ne_zero hus hBu) hBu]
    linear_combination (-2 * d.a * β * (d.Bt β u - 2 * d.a * β * d.ρ * ustar) *
      (d.Bt β u + 2 * d.a * β * d.ρ * u)) * hs
  have e₃ : -d.M β u / d.Bt β u =
      -2 * d.a * β * (u - ustar) * (d.Bt β u - 2 * d.a * β * d.ρ * ustar) /
          (d.Bt β u + 2 * d.a * β * d.ρ * u) - 2 * d.a * β * ustar := by
    rw [div_sub' hBu, div_eq_div_iff hB hBu]
    linear_combination core
  rw [e₁, e₂, e₃, lam]
  ring

/-! ### `K` for the density channel at `k = 0` -/

/-- The polynomial identity behind `K_static_density_pos`: at `w = 0`, `ε = 1`, and
`u = a(1 + Y)` (the root near the double zero `a` of `P`),
`a²([2q₊(βH(-u) - λc̄) + λρβ(p₂ - p₁)](Y² + 2Y + 4) - βa²ρY²(Y² + 4q₋Y + 4)) = β e(u²)`. -/
theorem K_static_density_pos_poly (hw : d.w = 0) (hd : d.ε = 1) (β Y : K) :
    d.a ^ 2 * ((2 * (d.p₁ + d.p₂) * (β * d.H (-(d.a * (1 + Y))) - d.lam * d.cbar β) +
          d.lam * d.ρ * (β * (d.p₂ - d.p₁))) * (Y ^ 2 + 2 * Y + 4) -
        β * d.a ^ 2 * d.ρ * Y ^ 2 * (Y ^ 2 + 4 * (d.p₀ + d.p₁) * Y + 4)) =
      β * d.e ((d.a * (1 + Y)) ^ 2) := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only at hw hd
  subst hw hd
  simp only [H, e, m, e₀, q₁, lam, cbar, ρ]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  ring

/-- The same for the root near `-a`, `u = -a(1 + Y)` (so `H(-u) = H(a(1 + Y))`). -/
theorem K_static_density_neg_poly (hw : d.w = 0) (hd : d.ε = 1) (β Y : K) :
    d.a ^ 2 * ((2 * (d.p₀ + d.p₁) * (β * d.H (d.a * (1 + Y)) - d.lam * d.cbar β) +
          d.lam * d.ρ * (β * (d.p₁ - d.p₀))) * (Y ^ 2 + 2 * Y + 4) +
        β * d.a ^ 2 * d.ρ * Y ^ 2 * (Y ^ 2 + 4 * (d.p₁ + d.p₂) * Y + 4)) =
      β * d.e ((-d.a * (1 + Y)) ^ 2) := by
  obtain ⟨a, w, p₀, p₁, p₂, ε, hs, hε⟩ := d
  simp only at hw hd
  subst hw hd
  simp only [H, e, m, e₀, q₁, lam, cbar, ρ]
  obtain rfl : p₂ = 1 - p₀ - 2 * p₁ := by linear_combination hs
  ring

/-- The package's form of `K` for the density channel at `k = 0` near the double zero `t = a` of
`P`, where `K` and `P(u)` are both `O(p₁²)` for `U < 0` at low temperature: with `u = t(1 + Y)` a
root of `e`, and `S(a) = β(p₂ - p₁)/(2q₊)` (`S₋`, the value of `S₀` at `a`),
`βH(-u) + λρS(a) - λc̄ = βa²ρY²(Y² + 4q₋Y + 4)/(2q₊(Y² + 2Y + 4))`, i.e.
`K - λρ(S(u) - S(a))` without cancellation. -/
theorem K_static_density_pos (hw : d.w = 0) (hd : d.ε = 1) {β Y : K}
    (hu : d.e ((d.a * (1 + Y)) ^ 2) = 0) (ha : d.a ≠ 0) (hq : d.p₁ + d.p₂ ≠ 0)
    (hY : Y ^ 2 + 2 * Y + 4 ≠ 0) :
    β * d.H (-(d.a * (1 + Y))) + d.lam * d.ρ * (β * (d.p₂ - d.p₁) / (2 * (d.p₁ + d.p₂))) -
        d.lam * d.cbar β =
      β * d.a ^ 2 * d.ρ * Y ^ 2 * (Y ^ 2 + 4 * (d.p₀ + d.p₁) * Y + 4) /
        (2 * (d.p₁ + d.p₂) * (Y ^ 2 + 2 * Y + 4)) := by
  have h := d.K_static_density_pos_poly hw hd β Y
  rw [hu, mul_zero] at h
  have h' := (mul_eq_zero.mp h).resolve_left (pow_ne_zero 2 ha)
  rw [eq_div_iff (mul_ne_zero (mul_ne_zero two_ne_zero hq) hY)]
  field_simp
  linear_combination h'

/-- The same near `t = -a`, with `u = -a(1 + Y)` and `S(-a) = β(p₁ - p₀)/(2q₋)` (`S₊`). -/
theorem K_static_density_neg (hw : d.w = 0) (hd : d.ε = 1) {β Y : K}
    (hu : d.e ((-d.a * (1 + Y)) ^ 2) = 0) (ha : d.a ≠ 0) (hq : d.p₀ + d.p₁ ≠ 0)
    (hY : Y ^ 2 + 2 * Y + 4 ≠ 0) :
    β * d.H (d.a * (1 + Y)) + d.lam * d.ρ * (β * (d.p₁ - d.p₀) / (2 * (d.p₀ + d.p₁))) -
        d.lam * d.cbar β =
      -(β * d.a ^ 2 * d.ρ * Y ^ 2 * (Y ^ 2 + 4 * (d.p₁ + d.p₂) * Y + 4)) /
        (2 * (d.p₀ + d.p₁) * (Y ^ 2 + 2 * Y + 4)) := by
  have h := d.K_static_density_neg_poly hw hd β Y
  rw [hu, mul_zero] at h
  have h' := (mul_eq_zero.mp h).resolve_left (pow_ne_zero 2 ha)
  rw [eq_div_iff (mul_ne_zero (mul_ne_zero two_ne_zero hq) hY)]
  field_simp
  linear_combination h'

/-! ### `Q(z, v)` -/

omit [CharZero K] in
/-- The divided-difference form of `Q(z, v) = 1 + (φ(z) - φ(v))/(z - v)`, `φ(z) = z g(z²)`,
`g = R/e` with `R(s) = r₁s + r₀` and `e(s) = (s - s₁)(s - s₂)`:
`Q = 1 + g(z²) + v(z + v) (r₁e(v²) - R(v²) e[z², v²])/(e(z²) e(v²))`,
`e[s, t] = (s - s₁) + (t - s₂)`. Nothing cancels in it for nearby large `z`, `v` (the package uses
it where `|z - v| ≤ |z + v|/2`). -/
theorem Q_eq {z v s₁ s₂ r₀ r₁ : K} (hzv : z - v ≠ 0) (hz₁ : z ^ 2 - s₁ ≠ 0) (hz₂ : z ^ 2 - s₂ ≠ 0)
    (hv₁ : v ^ 2 - s₁ ≠ 0) (hv₂ : v ^ 2 - s₂ ≠ 0) :
    1 + (z * ((r₁ * z ^ 2 + r₀) / ((z ^ 2 - s₁) * (z ^ 2 - s₂))) -
          v * ((r₁ * v ^ 2 + r₀) / ((v ^ 2 - s₁) * (v ^ 2 - s₂)))) / (z - v) =
      1 + (r₁ * z ^ 2 + r₀) / (z ^ 2 - s₁) / (z ^ 2 - s₂) +
        v / (z ^ 2 - s₁) * ((z + v) / (z ^ 2 - s₂)) *
          (r₁ - (r₁ * v ^ 2 + r₀) / (v ^ 2 - s₁) / (v ^ 2 - s₂) *
            ((z ^ 2 - s₁) + (v ^ 2 - s₂))) := by
  field_simp
  ring

end PHData

end Stable

end HubbardAtom
