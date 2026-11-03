import HubbardAtomLean.FieldLAxisOperator
import HubbardAtomLean.FieldLMu0

/-!
# `1 + 𝒦` and `Γ_L` on the axes with a known inverse

On `h = 0` (`μ ≠ 0`) and `μ = 0` (`h ≠ 0`), where the zero-field `1 + 𝒦_d`, `1 + 𝒦_m` are
invertible, `χ_L` has the explicit inverse `Y` of `FieldLZero.lean`, `FieldLMu0.lean`, built from
the zero-field inverses `χ_d⁻¹`, `χ_m⁻¹`. Here `1 + 𝒦` of `FieldLAxisOperator.lean` (lattices `z'`)
is reduced to them; the point `μ = h = 0` is not covered:

* `FParams.isUnit_onePlusKLAll_iff_onePlusK_h0`: at `h = 0`, `μ ≠ 0`, `1 + 𝒦` is invertible iff
  `1 + 𝒦_d` and `1 + 𝒦_m` of the zero-field atom `(U, β, μ)` are: a null vector `v` of `χ_L` in
  the class gives the null vectors `v↑ ± v↓` of `χ_d`, `χ_m`, and conversely `(w, ±w)`;
* `FParams.gammaOpLAll_h0`: there, `χ_L⁻¹ = Y` (`chiInvLAll_eq_invLh0`) and **Eq. (field-h0)**
  holds for the operator vertex, `Γ_L,στ = (Γ_d + s_στΓ_m)/2`;
* `FParams.isUnit_onePlusKLAll_shiba`: for every `μ`, `h`, `1 + 𝒦` is invertible iff that of the
  Shiba image `(-U, β, h, μ)` is, as `v_τl ↦ ε_τv_τ(R_τl)` maps the null vectors in the class onto
  each other (`hasSum_row_shiba`);
* `FParams.isUnit_onePlusKLAll_iff_onePlusK_mu0`: at `μ = 0`, `h ≠ 0`, `1 + 𝒦` is invertible iff
  `1 + 𝒦_d` and `1 + 𝒦_m` of the zero-field source `D = (-U, β, h)` are;
* `FParams.gammaOpLAll_mu0`: there, `χ_L⁻¹ = Y` (`chiInvLAll_eq_invLmu0`) and **Eq. (field-mu0)**
  holds for the operator vertex, `Γ_L,στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)`;
* `FParams.isUnit_onePlusKLAll_of_DMGeneric_h0`, `isUnit_onePlusKLAll_of_DMGeneric_mu0`: in
  particular `1 + 𝒦` is invertible where both zero-field channels satisfy the assumptions of
  Theorem "density and magnetic" (`DMGeneric`).
-/

open Complex

namespace HubbardAtom

namespace FParams

variable (F : FParams) (k : ℤ)

/-! ### Sums over `Bool × ℤ` -/

/-- A sum over `Bool × ℤ` is the sum over `ℤ` of the sums over the spin. -/
theorem hasSum_spin_of_prod {f : Bool × ℤ → ℂ} {a : ℂ} (h : HasSum f a) :
    HasSum (fun l ↦ ∑ τ, f (τ, l)) a :=
  ((Equiv.prodComm ℤ Bool).hasSum_iff.mpr h).prod_fiberwise fun _ ↦ hasSum_fintype _

/-- The sums of the two spins give the sum over `Bool × ℤ`. -/
theorem hasSum_prod_of_spin {f : Bool × ℤ → ℂ} {g : Bool → ℂ}
    (h : ∀ τ, HasSum (fun l ↦ f (τ, l)) (g τ)) : HasSum f (∑ τ, g τ) := by
  have h' : HasSum (f ∘ (Equiv.boolProdEquivSum ℤ).symm) (g false + g true) :=
    HasSum.sum (h false) (h true)
  rw [Fintype.sum_bool, add_comm]
  exact (Equiv.boolProdEquivSum ℤ).symm.hasSum_iff.mp h'

/-- The class on `Bool × ℤ` is the class of each spin. -/
theorem summable_prod_iff_spin {v : Bool × ℤ → ℂ} {z : Bool → ℤ → ℂ} :
    Summable (fun p : Bool × ℤ ↦ ‖v p / z p.1 p.2‖ ^ 2) ↔
      ∀ τ, Summable fun l ↦ ‖v (τ, l) / z τ l‖ ^ 2 := by
  rw [summable_prod_of_nonneg fun _ ↦ by positivity]
  exact ⟨fun h ↦ h.1, fun h ↦ ⟨h, Summable.of_finite⟩⟩

/-- The class `∑ₗ |vₗ/zₗ|² < ∞` is closed under `a + cb`. -/
theorem summable_add_mul_div_sq {z a b : ℤ → ℂ} (ha : Summable fun l ↦ ‖a l / z l‖ ^ 2)
    (hb : Summable fun l ↦ ‖b l / z l‖ ^ 2) (c : ℂ) :
    Summable fun l ↦ ‖(a l + c * b l) / z l‖ ^ 2 :=
  ((ha.mul_left 2).add (hb.mul_left (2 * ‖c‖ ^ 2))).of_nonneg_of_le (fun _ ↦ by positivity)
    fun l ↦ by
      rw [add_div, mul_div_assoc]
      set x := ‖a l / z l‖
      set y := ‖c‖ * ‖b l / z l‖
      have h1 := norm_add_le (a l / z l) (c * (b l / z l))
      rw [norm_mul] at h1
      have h2 := pow_le_pow_left₀ (norm_nonneg _) h1 2
      have hy : y ^ 2 = ‖c‖ ^ 2 * ‖b l / z l‖ ^ 2 := mul_pow _ _ _
      nlinarith [sq_nonneg (x - y)]

/-! ### The columns of `χ_L⁻¹` -/

/-- The columns of `χ_L⁻¹` are in the class `∑_l |v_τl/z_τl|² < ∞` on the lattice `z` of each
spin `τ`. -/
theorem summable_chiInvLAll (t : Bool) (m : ℤ) (τ : Bool) :
    Summable fun l ↦ ‖F.chiInvLAll k τ l t m / (F.BL τ).z k l‖ ^ 2 := by
  have h := (F.opLatLAll k).summable_chiInvX (F.chiLe k) (F.chiDecay_chiLe_all k) (encL (t, m))
  have h' : Summable fun p : Bool × ℤ ↦
      ‖F.chiInvLAll k p.1 p.2 t m / (F.BL p.1).zOp k p.2‖ ^ 2 := by
    refine (F.summable_opLatLAll_iff k (v := fun p ↦ F.chiInvLAll k p.1 p.2 t m)).mp
      (h.congr fun j ↦ ?_)
    simp only [chiInvLAll, Prod.mk.eta, Equiv.apply_symm_apply]
  exact (summable_prod_iff_spin (z := fun τ ↦ (F.BL τ).z k)).mp
    ((F.summable_zOpL_iff k).mp h') τ

variable {F k} in
/-- `χ_L⁻¹` is a right inverse where `1 + 𝒦` is invertible, summed spin by spin:
`∑_l ∑_τ χ_L,στ(n, l)χ_L⁻¹_τt(l, m) = δ_σtδ_nm`. -/
theorem hasSum_chiL_mul_chiInvLAll (hK : IsUnit (F.onePlusKLAll k)) (σ : Bool) (n : ℤ)
    (t : Bool) (m : ℤ) :
    HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * F.chiInvLAll k τ l t m)
      (if σ = t then kron n m else 0) := by
  -- `have` first: elaborating against the expected `fun l ↦ ∑ τ, …` needs higher-order unification
  have h := hasSum_spin_of_prod (chiInvLAll_isRightInverse hK σ n t m)
  exact h

/-! ### `h = 0` -/

variable {F k}

/-- At `h = 0` the lattices of both spins are that of the zero-field atom. -/
theorem BL_of_h (hh : F.h = 0) (σ : Bool) : F.BL σ = F.lat F.μ := by
  cases σ <;> simp [BL, cL, hh]

/-- **`1 + 𝒦` at `h = 0`**: for `μ ≠ 0`, `1 + 𝒦` of `χ_L` is invertible iff `1 + 𝒦_d` and
`1 + 𝒦_m` of the zero-field atom `(U, β, μ)` are. A null vector `v` of `χ_L` with
`∑_{σn} |v_σn/z'_σn|² < ∞` gives the null vectors `v↑ ± v↓` of `χ_d`, `χ_m` in their class, and
a null vector `w` of `χ_d` (of `χ_m`) gives the null vector `(w, w)` (`(w, -w)`) of `χ_L`. -/
theorem isUnit_onePlusKLAll_iff_onePlusK_h0 (hh : F.h = 0) (hμ : F.μ ≠ 0) :
    IsUnit (F.onePlusKLAll k) ↔
      IsUnit ((F.lat F.μ).onePlusK k 1 hμ) ∧ IsUnit ((F.lat F.μ).onePlusK k (-1) hμ) := by
  have hcls : ∀ v : Bool × ℤ → ℂ,
      Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2) ↔
        ∀ τ, Summable fun l ↦ ‖v (τ, l) / (F.lat F.μ).z k l‖ ^ 2 := fun v ↦ by
    rw [F.summable_zOpL_iff k]
    simp only [F.BL_of_h hh]
    exact summable_prod_iff_spin (z := fun _ ↦ (F.lat F.μ).z k)
  -- a row of `χ_L` against `(c↑w, c↓w)`
  have hrow : ∀ w : ℤ → ℂ, Summable (fun l ↦ ‖w l / (F.lat F.μ).z k l‖ ^ 2) →
      ∀ (c : Bool → ℂ) σ n, HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * (c p.1 * w p.2))
        (∑ τ, c τ * ((∑' l, (F.lat F.μ).chiR k 1 n l * w l) +
          sL σ τ * ∑' l, (F.lat F.μ).chiR k (-1) n l * w l) / 2) := by
    intro w hw c σ n
    refine hasSum_prod_of_spin fun τ ↦ ?_
    have hD := ((F.lat F.μ).hasSum_chiR_mul k 1 hμ hw n).summable.hasSum
    have hM := ((F.lat F.μ).hasSum_chiR_mul k (-1) hμ hw n).summable.hasSum
    convert ((hD.add (hM.mul_left (sL σ τ))).mul_left (c τ)).div_const 2 using 1
    funext l
    rw [F.chiL_of_h k hh]
    ring
  rw [F.isUnit_onePlusKLAll_iff k, (F.lat F.μ).isUnit_onePlusK_iff k 1 hμ,
    (F.lat F.μ).isUnit_onePlusK_iff k (-1) hμ]
  constructor
  · intro hL
    refine ⟨fun w hw hχ ↦ ?_, fun w hw hχ ↦ ?_⟩
    · have h0 := hL (fun p ↦ 1 * w p.2) ((hcls _).mpr fun τ ↦ by simpa using hw) fun σ n ↦ by
        convert hrow w hw (fun _ ↦ 1) σ n using 1
        rw [(hχ n).tsum_eq]
        cases σ <;> simp [sL] <;> ring
      funext l
      simpa using congrFun h0 (true, l)
    · have hs : ∀ τ, ‖sL true τ‖ = 1 := fun τ ↦ by cases τ <;> simp [sL]
      have h0 := hL (fun p ↦ sL true p.1 * w p.2)
        ((hcls _).mpr fun τ ↦ by simpa [mul_div_assoc, hs] using hw) fun σ n ↦ by
          convert hrow w hw (sL true) σ n using 1
          rw [(hχ n).tsum_eq]
          simp [sL]
          ring
      funext l
      simpa [sL] using congrFun h0 (true, l)
  · rintro ⟨hd, hm⟩ v hv hχ
    have hv' := (hcls v).mp hv
    have hs : ∀ σ n, HasSum (fun l ↦ ∑ τ, F.chiL σ τ k n l * v (τ, l)) 0 :=
      fun σ n ↦ by
        have h := hasSum_spin_of_prod (hχ σ n)
        exact h
    have hD := hd (fun l ↦ v (true, l) + 1 * v (false, l))
      (summable_add_mul_div_sq (hv' true) (hv' false) 1) fun n ↦ by
        convert (hs true n).add (hs false n) using 1
        · funext l
          simp only [Fintype.sum_bool, F.chiL_of_h k hh, sL]
          norm_num
          ring
        · simp
    have hM := hm (fun l ↦ v (true, l) + -1 * v (false, l))
      (summable_add_mul_div_sq (hv' true) (hv' false) (-1)) fun n ↦ by
        convert (hs true n).sub (hs false n) using 1
        · funext l
          simp only [Fintype.sum_bool, F.chiL_of_h k hh, sL]
          norm_num
          ring
        · simp
    funext ⟨τ, l⟩
    have h1 := congrFun hD l
    have h2 := congrFun hM l
    simp only [Pi.zero_apply] at h1 h2 ⊢
    cases τ
    · linear_combination (h1 - h2) / 2
    · linear_combination (h1 + h2) / 2

/-- **`χ_L⁻¹` at `h = 0` is `Y`**: for `μ ≠ 0`, where `1 + 𝒦_d` and `1 + 𝒦_m` of the zero-field
atom are invertible, `χ_L⁻¹ = Y = (χ_d⁻¹ + s_στχ_m⁻¹)/2` (`invLh0`), entrywise. -/
theorem chiInvLAll_eq_invLh0 (hh : F.h = 0) (hμ : F.μ ≠ 0)
    (hd : IsUnit ((F.lat F.μ).onePlusK k 1 hμ)) (hm : IsUnit ((F.lat F.μ).onePlusK k (-1) hμ))
    (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.chiInvLAll k σ n t m = F.invLh0 k hμ σ n t m := by
  have hK := (isUnit_onePlusKLAll_iff_onePlusK_h0 hh hμ).mpr ⟨hd, hm⟩
  exact eq_invLh0_of_hasSum hh hμ hd hm (Y := fun τ l ↦ F.chiInvLAll k τ l t m)
    (fun τ ↦ by simpa only [F.BL_of_h hh] using F.summable_chiInvLAll k t m τ)
    (fun σ n ↦ hasSum_chiL_mul_chiInvLAll hK σ n t m) σ n

/-- **The operator vertex at `h = 0`, Eq. (field-h0)**: for `μ ≠ 0`, where `1 + 𝒦_d` and
`1 + 𝒦_m` of the zero-field atom are invertible (equivalently `1 + 𝒦`,
`isUnit_onePlusKLAll_iff_onePlusK_h0`), `Γ_L,στ = (Γ_d + s_στΓ_m)/2`, entrywise, with `Γ_r` the
operator vertices of the zero-field atom (`GammaOperator.lean`). -/
theorem gammaOpLAll_h0 (hh : F.h = 0) (hμ : F.μ ≠ 0)
    (hd : IsUnit ((F.lat F.μ).onePlusK k 1 hμ)) (hm : IsUnit ((F.lat F.μ).onePlusK k (-1) hμ))
    (σ : Bool) (n : ℤ) (t : Bool) (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      ((F.lat F.μ).gammaOp k 1 hμ n m + sL σ t * (F.lat F.μ).gammaOp k (-1) hμ n m) / 2 := by
  rw [gammaOpLAll_eq_chiInvLAll, chiInvLAll_eq_invLh0 hh hμ hd hm]
  exact gammaL_h0 hh hμ σ n t m

/-! ### The Shiba map -/

variable (F k)

/-- The lattice of `τ` of the Shiba image at `R_τl` is `±z_τl`, for every `μ`, `h`. -/
theorem norm_BL_z_shiba (τ : Bool) (l : ℤ) :
    ‖(F.shiba.BL τ).z k (RL k τ l)‖ = ‖(F.BL τ).z k l‖ := by
  cases τ
  · rw [show (F.shiba.BL false).z k (RL k false l) = -(F.BL false).z k l by
      simp only [RL, Bool.false_eq_true, ↓reduceIte, Params.z_eq, cL, lat_μ, lat_β, shiba_μ,
        shiba_h, shiba_β, Bool.cond_false]
      push_cast
      field_simp
      ring, norm_neg]
  · congr 1
    simp only [RL, ↓reduceIte, Params.z_eq, cL, lat_μ, lat_β, shiba_μ, shiba_h, shiba_β,
      Bool.cond_true]
    push_cast
    ring

/-- **The Shiba map on the rows of `χ_L`**, for every `μ`, `h`: the row `(σ, n)` of `χ_L` against
`v` sums to `a` iff the row `(σ, R_σn)` of `χ_L^shiba` against `W_τl = ε_τv_τ(R_τl)` sums to
`ε_σa`. -/
theorem hasSum_row_shiba (σ : Bool) (n : ℤ) {v : Bool × ℤ → ℂ} {a : ℂ} :
    HasSum (fun p : Bool × ℤ ↦ F.chiL σ p.1 k n p.2 * v p) a ↔
      HasSum (fun p : Bool × ℤ ↦ F.shiba.chiL σ p.1 k (RL k σ n) p.2 *
        (εL p.1 * v (p.1, RL k p.1 p.2))) (εL σ * a) := by
  refine Iff.trans ?_ (Equiv.prodCongrRight (RLe k)).hasSum_iff
  have hε : εL σ ≠ 0 := by cases σ <;> simp [εL]
  rw [← hasSum_mul_left_iff hε]
  refine iff_of_eq (congrArg₂ HasSum (funext fun p ↦ ?_) rfl)
  obtain ⟨τ, l⟩ := p
  simp only [Function.comp_apply, Equiv.prodCongrRight_apply, RLe, Equiv.coe_fn_mk, RL_RL,
    F.chiL_shiba k σ τ n l]
  linear_combination (εL τ * F.shiba.chiL σ τ k (RL k σ n) (RL k τ l) * v (τ, l)) *
    εL_mul_self σ

/-- **`1 + 𝒦` under the Shiba map**, for every `μ`, `h`: `1 + 𝒦` of `χ_L` is invertible iff
`1 + 𝒦` of the Shiba image `(-U, β, h, μ)` is, as `v_τl ↦ ε_τv_τ(R_τl)` maps the null vectors of
`χ_L` in the class onto those of `χ_L^shiba` (`hasSum_row_shiba`). -/
theorem isUnit_onePlusKLAll_shiba :
    IsUnit (F.onePlusKLAll k) ↔ IsUnit (F.shiba.onePlusKLAll k) := by
  have hε : ∀ τ, ‖εL τ‖ = 1 := fun τ ↦ by cases τ <;> simp [εL]
  have hε0 : ∀ τ, εL τ ≠ 0 := fun τ ↦ by cases τ <;> simp [εL]
  have hcls : ∀ v : Bool × ℤ → ℂ,
      Summable (fun p : Bool × ℤ ↦ ‖v p / (F.BL p.1).zOp k p.2‖ ^ 2) ↔
        Summable (fun p : Bool × ℤ ↦
          ‖εL p.1 * v (p.1, RL k p.1 p.2) / (F.shiba.BL p.1).zOp k p.2‖ ^ 2) := fun v ↦ by
    rw [F.summable_zOpL_iff k, F.shiba.summable_zOpL_iff k]
    refine Iff.trans ?_ (Equiv.prodCongrRight (RLe k)).summable_iff
    refine iff_of_eq (congrArg Summable (funext fun p ↦ ?_))
    obtain ⟨τ, l⟩ := p
    simp only [Function.comp_apply, Equiv.prodCongrRight_apply, RLe, Equiv.coe_fn_mk, RL_RL,
      norm_div, norm_mul, hε, one_mul, F.norm_BL_z_shiba k τ l]
  rw [F.isUnit_onePlusKLAll_iff k, F.shiba.isUnit_onePlusKLAll_iff k]
  constructor
  · intro h W hW hχ
    have hWv : ∀ p : Bool × ℤ, εL p.1 * (εL p.1 * W (p.1, RL k p.1 (RL k p.1 p.2))) = W p :=
      fun p ↦ by rw [RL_RL, ← mul_assoc, εL_mul_self, one_mul]
    have h0 := h (fun p ↦ εL p.1 * W (p.1, RL k p.1 p.2))
      ((hcls _).mpr (by simpa only [hWv] using hW)) fun σ n ↦ (F.hasSum_row_shiba k σ n).mpr
        (by simpa only [hWv, mul_zero] using hχ σ (RL k σ n))
    funext p
    have hp := congrFun h0 (p.1, RL k p.1 p.2)
    simp only [RL_RL, Pi.zero_apply, mul_eq_zero, hε0, false_or] at hp
    exact hp
  · intro h v hv hχ
    have h0 := h _ ((hcls v).mp hv) fun σ n ↦ by
      simpa [RL_RL] using (F.hasSum_row_shiba k σ (RL k σ n)).mp (hχ σ (RL k σ n))
    funext p
    have hp := congrFun h0 (p.1, RL k p.1 p.2)
    simp only [RL_RL, Pi.zero_apply, mul_eq_zero, hε0, false_or] at hp
    exact hp

/-! ### `μ = 0` -/

variable {F k}

/-- **`1 + 𝒦` at `μ = 0`**: for `h ≠ 0`, `1 + 𝒦` of `χ_L` is invertible iff `1 + 𝒦_d` and
`1 + 𝒦_m` of the zero-field source `D = (-U, β, h)` are (`isUnit_onePlusKLAll_shiba`, and
`isUnit_onePlusKLAll_iff_onePlusK_h0` for the Shiba image). -/
theorem isUnit_onePlusKLAll_iff_onePlusK_mu0 (hμ0 : F.μ = 0) (hh : F.h ≠ 0) :
    IsUnit (F.onePlusKLAll k) ↔ IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k 1 hh) ∧
      IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k (-1) hh) :=
  (F.isUnit_onePlusKLAll_shiba k).trans (isUnit_onePlusKLAll_iff_onePlusK_h0 (F := F.shiba) hμ0 hh)

/-- **`χ_L⁻¹` at `μ = 0` is `Y`**: for `h ≠ 0`, where `1 + 𝒦_d` and `1 + 𝒦_m` of the zero-field
source are invertible, `χ_L⁻¹ = Y` (`invLmu0`), entrywise. -/
theorem chiInvLAll_eq_invLmu0 (hμ0 : F.μ = 0) (hh : F.h ≠ 0)
    (hd : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k 1 hh))
    (hm : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k (-1) hh)) (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) : F.chiInvLAll k σ n t m = F.invLmu0 k hh σ n t m := by
  have hK := (isUnit_onePlusKLAll_iff_onePlusK_mu0 hμ0 hh).mpr ⟨hd, hm⟩
  exact eq_invLmu0_of_hasSum hμ0 hh hd hm (Y := fun τ l ↦ F.chiInvLAll k τ l t m)
    (F.summable_chiInvLAll k t m) (fun σ n ↦ hasSum_chiL_mul_chiInvLAll hK σ n t m) σ n

/-- **The operator vertex at `μ = 0`, Eq. (field-mu0)**: for `h ≠ 0`, where `1 + 𝒦_d^D` and
`1 + 𝒦_m^D` of the zero-field source `D = (-U, β, h)` are invertible (equivalently `1 + 𝒦`,
`isUnit_onePlusKLAll_iff_onePlusK_mu0`),
`Γ_L,στ(n, m) = (ε_σε_τ/2)[Γ_d^D + ε_σε_τΓ_m^D](R_σn, R_τm)`, entrywise. -/
theorem gammaOpLAll_mu0 (hμ0 : F.μ = 0) (hh : F.h ≠ 0)
    (hd : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k 1 hh))
    (hm : IsUnit ((F.shiba.lat F.shiba.μ).onePlusK k (-1) hh)) (σ : Bool) (n : ℤ) (t : Bool)
    (m : ℤ) :
    F.gammaOpLAll k σ n t m =
      εL σ * εL t / 2 * ((F.shiba.lat F.shiba.μ).gammaOp k 1 hh (RL k σ n) (RL k t m) +
        εL σ * εL t * (F.shiba.lat F.shiba.μ).gammaOp k (-1) hh (RL k σ n) (RL k t m)) := by
  rw [gammaOpLAll_eq_chiInvLAll, chiInvLAll_eq_invLmu0 hμ0 hh hd hm]
  exact gammaL_mu0 hμ0 hh σ n t m

/-- **`1 + 𝒦` at `h = 0` on the generic domain**: for `μ ≠ 0`, if both zero-field channels of
`(U, β, μ)` satisfy the assumptions of Theorem "density and magnetic", `1 + 𝒦` is invertible. -/
theorem isUnit_onePlusKLAll_of_DMGeneric_h0 (hh : F.h = 0) {u₁ u₂ u₃ u₄ : ℂ}
    (hd : (F.lat F.μ).DMGeneric k 1 u₁ u₂) (hm : (F.lat F.μ).DMGeneric k (-1) u₃ u₄) :
    IsUnit (F.onePlusKLAll k) :=
  (isUnit_onePlusKLAll_iff_onePlusK_h0 hh hd.μ_ne).mpr ⟨hd.isUnit_onePlusK, hm.isUnit_onePlusK⟩

/-- **`1 + 𝒦` at `μ = 0` on the generic domain**: for `h ≠ 0`, if both zero-field channels of the
Shiba source `(-U, β, h)` satisfy the assumptions of Theorem "density and magnetic", `1 + 𝒦` is
invertible. -/
theorem isUnit_onePlusKLAll_of_DMGeneric_mu0 (hμ0 : F.μ = 0) {u₁ u₂ u₃ u₄ : ℂ}
    (hd : (F.shiba.lat F.shiba.μ).DMGeneric k 1 u₁ u₂)
    (hm : (F.shiba.lat F.shiba.μ).DMGeneric k (-1) u₃ u₄) : IsUnit (F.onePlusKLAll k) :=
  (isUnit_onePlusKLAll_iff_onePlusK_mu0 hμ0 hd.μ_ne).mpr ⟨hd.isUnit_onePlusK, hm.isUnit_onePlusK⟩

end FParams

end HubbardAtom
