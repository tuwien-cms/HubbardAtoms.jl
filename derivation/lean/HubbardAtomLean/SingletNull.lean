import HubbardAtomLean.SingletOperator

/-!
# The singlet null vectors

Where `Γ_s` of `SingletOperator.lean` diverges, on the lattice `z'ₙ` of `OpLatticeAll.lean`, for
every `μ` and `k`. Appendix "Vertex divergences", section "Null vectors", of the note: at `Kₛ = 0`
with `𝒫(-Aₛ) ≠ 0`, `gₗ = 1/(tₗ + Aₛ)` is a null vector of `χ_s`.

* `null_identity`, `null_identity_zero`: the cancellation in `(χ_s g)ₙ` at `Kₛ = 0`, as rational
  identities, for `b ≠ 0` (from identities (A4)) and `b = 0`.
* `Params.hasSum_chiS_kappa`: **`χ_s g = 0` at `Kₛ = 0`**, for `U ≠ 0`, `tₙ + Aₛ ≠ 0`, `𝒫(-Aₛ) ≠ 0`.
* `Params.isUnit_onePlusKS_iff_Ks`: under these assumptions, **`Γ_s` exists iff `Kₛ ≠ 0`**.
* `Params.isUnit_onePlusKS_of_edge`: on the plateau edge `b = ±2a` with `U > 0`, where
  `𝒫(-Aₛ) = 0`, `Γ_s` exists (`Kₛ = 1 + βa/2`).

On the lattice curves `t_c + Aₛ = 0`, `Γ_s` exists by `SingletLZ.isUnit_onePlusKS`. That `g`
spans the kernel at `Kₛ = 0` is not shown, as in the note.
-/

open Complex

namespace HubbardAtom

/-- At `Kₛ = 0` and `b ≠ 0`, `κ = 1/(t + Aₛ)` is a null vector of `χ_s` row by row: with
`V± = ∑ g±κ`, `d±V± = T± - T₀`, and identities (A4) at `t±`, `γ±V± = ∓β/(2(t₊ - t₋))`, so
`L_s κ + γ₊g₊V₊ + γ₋g₋V₋ = 0`. -/
theorem null_identity {a β T₀ t As tp tm Tp Tm Vp Vm γp γm : ℂ}
    (hVp : (As + tp) * Vp = Tp - T₀) (hVm : (As + tm) * Vm = Tm - T₀)
    (hK : 1 + 2 * a * T₀ / β = 0)
    (hIp : a * (As + tp) / (tp - tm) + γp * (1 + 2 * a * Tp / β) = 0)
    (hIm : -a * (As + tm) / (tp - tm) + γm * (1 + 2 * a * Tm / β) = 0)
    (ha : a ≠ 0) (hβ : β ≠ 0) (hpm : tp - tm ≠ 0) (hdp : As + tp ≠ 0) (hdm : As + tm ≠ 0)
    (ht₁ : t - tp ≠ 0) (ht₂ : t - tm ≠ 0) (ht₃ : t + As ≠ 0) :
    β / 2 * (t + As) / ((t - tp) * (t - tm)) * (1 / (t + As)) +
      (γp / (t - tp) * Vp + γm / (t - tm) * Vm) = 0 := by
  have hp : γp * Vp + β / (2 * (tp - tm)) = 0 := by
    linear_combination (norm := (field_simp; ring)) β / (2 * a * (As + tp)) * hIp -
      β / (2 * a * (As + tp)) * γp * hK + γp / (As + tp) * hVp
  have hm : γm * Vm - β / (2 * (tp - tm)) = 0 := by
    linear_combination (norm := (field_simp; ring)) β / (2 * a * (As + tm)) * hIm -
      β / (2 * a * (As + tm)) * γm * hK + γm / (As + tm) * hVm
  linear_combination (norm := (field_simp; ring)) 1 / (t - tp) * hp + 1 / (t - tm) * hm

/-- At `Kₛ = 0` and `b = 0` (`t₊ = t₋`), the same with the regular part `chiReg₀`:
`L_s κ + c₁gV₁ + c₂(g²V₁ + gV₂) = 0`, `V₁ = ∑ gκ`, `V₂ = ∑ g²κ`. -/
theorem null_identity_zero {a β p₀ p₁ T₀ T₁ T₂ V₁ V₂ S t As tp tm : ℂ}
    (hs : p₀ + 2 * p₁ + p₀ = 1) (hS : S = β * (p₀ - p₁) / (2 * (p₁ + p₀)))
    (hT₁ : T₁ = -S / a) (hT₂ : T₂ = (S ^ 2 - β ^ 2 / 4) / (2 * a ^ 2) - S / (2 * a ^ 3))
    (htp : tp = -a ^ 2) (htm : tm = -a ^ 2) (hAs : As = a ^ 2 * (1 - 8 * p₁))
    (hV₁ : (As + tp) * V₁ = T₁ - T₀) (hV₂ : (As + tp) ^ 2 * V₂ = (As + tp) * T₂ - T₁ + T₀)
    (hK : 1 + 2 * a * T₀ / β = 0) (ha : a ≠ 0) (hβ : β ≠ 0) (hq : p₁ + p₀ ≠ 0) (hp₁ : p₁ ≠ 0)
    (ht : t - tp ≠ 0) (hA : t + As ≠ 0) :
    β / 2 * (t + As) / ((t - tp) * (t - tm)) * (1 / (t + As)) +
      ((2 * a ^ 2 * β * p₀ - a) * (1 / (t - tp)) * V₁ +
        2 * a ^ 3 * ((1 / (t - tp)) ^ 2 * V₁ + 1 / (t - tp) * V₂)) = 0 := by
  have hd : As + tp ≠ 0 := by
    rw [hAs, htp]
    intro h0
    exact hp₁ (by simpa [ha] using (show -8 * a ^ 2 * p₁ = 0 by linear_combination h0))
  have hT₀ : T₀ = -β / (2 * a) := by
    field_simp at hK
    field_simp
    linear_combination hK
  obtain rfl : V₁ = (T₁ - T₀) / (As + tp) := by field_simp; linear_combination hV₁
  obtain rfl : V₂ = ((As + tp) * T₂ - T₁ + T₀) / (As + tp) ^ 2 := by
    field_simp
    linear_combination hV₂
  subst htp htm hAs
  obtain rfl : p₀ = (1 - 2 * p₁) / 2 := by linear_combination hs / 2
  rw [hT₀, hT₁, hT₂, hS]
  field_simp
  ring_nf
  field_simp
  ring

namespace Params

variable {A : Params} {k : ℤ} {ξ : ℂ}

/-- **At `Kₛ = 0`, `κ = 1/(t + Aₛ)` is a null vector of `χ_s`**: `∑ₘ χ_s,nm κₘ = 0` for every `n`,
for `U ≠ 0`, `tₙ + Aₛ ≠ 0` and `𝒫(-Aₛ) ≠ 0`. -/
theorem hasSum_chiS_kappa (ha : (A.a : ℂ) ≠ 0) (hAs : ∀ n, A.t k n + A.As k ≠ 0)
    (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k) (hd : A.calP k (-A.As k) ≠ 0) (hK : A.Ks k ξ = 0)
    (n : ℤ) : HasSum (fun m ↦ A.chiS k n m * (1 / (A.t k m + A.As k))) 0 := by
  have hβ := A.β_ne_zero
  rw [A.calP_eq_mul, show -A.As k - A.tp k = -(A.As k + A.tp k) by ring,
    show -A.As k - A.tm k = -(A.As k + A.tm k) by ring, neg_mul_neg] at hd
  have hdp := left_ne_zero_of_mul hd
  have hdm := right_ne_zero_of_mul hd
  have hK' : 1 + 2 * A.a * A.Tpp k ξ / A.β = 0 := hK
  have h₁ : HasSum (fun l ↦ mirrorBlock k (fun j ↦ A.Ls k (A.t k j)) (fun j ↦ -A.Bpp k j) n l *
      (1 / (A.t k l + A.As k))) (A.Ls k (A.t k n) * (1 / (A.t k n + A.As k))) :=
    hasSum_mirrorBlock_mul_even (fun l ↦ by rw [A.t_mirror]) n
  have hκ : HasSum (fun l ↦ 1 / (A.t k l + A.As k)) (A.Tpp k ξ) := by
    have := A.hasSum_Tpp (k := k) (t := -A.As k) (ξ := ξ) (by rw [hξ]; ring)
      (fun l h0 ↦ hAs l (by rw [h0]; ring))
    simpa [sub_neg_eq_add] using this
  by_cases hb : A.b k = 0
  · have hT₁ := A.hasSum_gp k
    have ha2 : (A.a : ℂ) - A.b k / 2 ≠ 0 := by rw [hb, zero_div, sub_zero]; exact ha
    have hT₂ := A.hasSum_one_div_t_sub_sq (k := k) (t := A.tp k) (ξ := A.a - A.b k / 2)
      (by rw [tp]; ring) ha2 (fun n h0 ↦ A.t_sub_tp_ne k n (by rw [h0, sub_self]))
    obtain ⟨V₁, hV₁, hV₁'⟩ := A.exists_hasSum_mul_κ k hT₁ hκ (A.t_sub_tp_ne k) hAs
    obtain ⟨V₂, hV₂, hV₂'⟩ := A.exists_hasSum_sq_mul_κ k hT₁ hT₂ hκ (A.t_sub_tp_ne k) hAs
    have h₂ : HasSum (fun l ↦ A.chiReg₀ k n l * (1 / (A.t k l + A.As k)))
        ((2 * A.a ^ 2 * A.β * A.p₀ - A.a) * A.g₀ k n * V₁ +
          2 * A.a ^ 3 * (A.g₀ k n ^ 2 * V₁ + A.g₀ k n * V₂)) := by
      convert (hV₁.mul_left ((2 * A.a ^ 2 * A.β * A.p₀ - A.a) * A.g₀ k n)).add
        ((hV₁.mul_left (2 * A.a ^ 3 * A.g₀ k n ^ 2)).add
          (hV₂.mul_left (2 * A.a ^ 3 * A.g₀ k n))) using 1
      · funext l
        simp only [chiReg₀, g₀]
        ring
      · ring
    convert h₁.add h₂ using 1
    · funext l
      rw [A.chiS_eq_zero k hb]
      ring
    obtain ⟨hμ, rfl⟩ := (A.b_eq_zero_iff k).mp hb
    have hp₂ := p₂_eq_p₀_of_mu hμ
    have hs : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₀ = 1 := by
      have := A.weights_sum
      rw [hp₂] at this
      exact_mod_cast this
    have hq : (A.p₁ : ℂ) + A.p₀ ≠ 0 := by exact_mod_cast (add_pos A.p₁_pos A.p₀_pos).ne'
    have hp₁ : (A.p₁ : ℂ) ≠ 0 := by exact_mod_cast A.p₁_pos.ne'
    have hρ : (A.ρ : ℂ) = 0 := by rw [ρ, hp₂, sub_self, ofReal_zero]
    have hSa : A.S 0 (A.μ + (A.a - A.b 0 / 2)) = A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀)) := by
      rw [hb, zero_div, sub_zero, hμ, ofReal_zero, zero_add, A.S₀_a, hp₂]
    have hTp : A.Tpp 0 (A.a - A.b 0 / 2) = -(A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀))) / A.a := by
      rw [Tpp, ite_eq_right ha2, hSa, hb, zero_div, sub_zero]
    have htp : A.tp 0 = -A.a ^ 2 := by rw [tp, hb, mul_zero, add_zero]
    have htm : A.tm 0 = -A.a ^ 2 := by rw [tm, hb, mul_zero, sub_zero]
    have hAs0 : A.As 0 = A.a ^ 2 * (1 - 8 * A.p₁) := by rw [As, hρ, hb]; ring
    have hN := null_identity_zero (a := A.a) (β := A.β) (p₀ := A.p₀) (p₁ := A.p₁)
      (T₀ := A.Tpp 0 ξ) (T₁ := A.Tpp 0 (A.a - A.b 0 / 2))
      (T₂ := (A.S 0 (A.μ + (A.a - A.b 0 / 2)) ^ 2 - (A.β : ℂ) ^ 2 / 4) /
        (2 * (A.a - A.b 0 / 2) ^ 2) - A.S 0 (A.μ + (A.a - A.b 0 / 2)) /
        (2 * (A.a - A.b 0 / 2) ^ 3))
      (V₁ := V₁) (V₂ := V₂) (S := A.β * (A.p₀ - A.p₁) / (2 * (A.p₁ + A.p₀)))
      (t := A.t 0 n) (As := A.As 0) (tp := A.tp 0) (tm := A.tm 0) hs rfl hTp
      (by rw [hSa, hb, zero_div, sub_zero]) htp htm hAs0 hV₁' hV₂' hK' ha hβ hq hp₁
      (A.t_sub_tp_ne 0 n) (hAs n)
    rw [Ls, A.calP_eq_mul, g₀]
    linear_combination -hN
  · have hgp := A.hasSum_gp k
    have hgm := A.hasSum_gm k
    obtain ⟨Vp, hgpκ, hVp⟩ := A.exists_hasSum_mul_κ k hgp hκ (A.t_sub_tp_ne k) hAs
    obtain ⟨Vm, hgmκ, hVm⟩ := A.exists_hasSum_mul_κ k hgm hκ (A.t_sub_tm_ne k) hAs
    have h₂ : HasSum (fun l ↦ A.chiReg k n l * (1 / (A.t k l + A.As k)))
        (A.γp k / (A.t k n - A.tp k) * Vp + A.γm k / (A.t k n - A.tm k) * Vm) := by
      convert (hgpκ.mul_left (A.γp k / (A.t k n - A.tp k))).add
        (hgmκ.mul_left (A.γm k / (A.t k n - A.tm k))) using 1
      funext l
      rw [chiReg]
      ring
    convert h₁.add h₂ using 1
    · funext l
      rw [A.chiS_eq k hb]
      ring
    have hpm : A.tp k - A.tm k ≠ 0 := by
      rw [tp, tm, show -(A.a : ℂ) ^ 2 + A.a * A.b k - (-A.a ^ 2 - A.a * A.b k) =
        2 * A.a * A.b k by ring]
      exact mul_ne_zero (mul_ne_zero two_ne_zero ha) hb
    have hN := null_identity hVp hVm hK' (A.singlet_Ip k ha hb) (A.singlet_Im k ha hb) ha hβ hpm
      hdp hdm (A.t_sub_tp_ne k n) (A.t_sub_tm_ne k n) (hAs n)
    rw [Ls, A.calP_eq_mul]
    linear_combination -hN

/-- **`Γ_s` diverges iff `Kₛ = 0`**: for `U ≠ 0`, `tₙ + Aₛ ≠ 0` and `𝒫(-Aₛ) ≠ 0`, `1 + 𝒦` is
invertible iff `Kₛ ≠ 0`. At `Kₛ = 0`, `κ = 1/(t + Aₛ)` is a bounded null vector of `χ_s`. -/
theorem isUnit_onePlusKS_iff_Ks (ha : (A.a : ℂ) ≠ 0) (hAs : ∀ n, A.t k n + A.As k ≠ 0)
    (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k) (hd : A.calP k (-A.As k) ≠ 0) :
    IsUnit (A.onePlusKS k) ↔ A.Ks k ξ ≠ 0 := by
  refine ⟨fun hu hK ↦ ?_, fun hK ↦ SingletHyp.isUnit_onePlusKS ⟨ha, hAs, hξ, hK⟩⟩
  set κ : ℤ → ℂ := fun n ↦ 1 / (A.t k n + A.As k)
  have hκ : Summable fun n ↦ ‖κ n‖ := by
    refine summable_norm_iff.mpr (A.hasSum_Tpp (k := k) (t := -A.As k) (ξ := ξ) (by rw [hξ]; ring)
      (fun l h0 ↦ hAs l (by rw [h0]; ring))).summable |>.congr fun n ↦ ?_
    simp only [κ, sub_neg_eq_add]
  have hv : Summable fun n ↦ ‖κ n / A.zOp k n‖ ^ 2 := by
    refine ((A.summable_inv_norm_zOp_sq k).mul_left ((∑' l, ‖κ l‖) ^ 2)).of_nonneg_of_le
      (fun n ↦ by positivity) fun n ↦ ?_
    rw [norm_div, div_eq_mul_inv, mul_pow]
    gcongr
    exact hκ.le_tsum n fun l _ ↦ norm_nonneg _
  have h0 := (A.isUnit_onePlusKS_iff k).mp hu κ hv (hasSum_chiS_kappa ha hAs hξ hd hK)
  exact one_div_ne_zero (hAs 0) (congrFun h0 0)

/-- **On the plateau edge `b = ±2a`** (`k = 0`, `μ = ±a`) with `U > 0`, `Γ_s` is finite: there
`ξ = 0`, `Kₛ = 1 + βa/2 > 0` and `tₙ + Aₛ = tₙ - t± ≠ 0`, so the assumptions of Theorem "singlet"
hold. -/
theorem isUnit_onePlusKS_of_edge (ha : 0 < A.a) (hb : A.b k = 2 * A.a ∨ A.b k = -2 * A.a) :
    IsUnit (A.onePlusKS k) := by
  have hβ := A.β_ne_zero
  have hK : A.Ks k 0 ≠ 0 := by
    rw [Ks, Tpp, ite_eq_left rfl,
      show (1 + 2 * A.a * (A.β ^ 2 / 4) / A.β : ℂ) = ((1 + A.a * A.β / 2 : ℝ) : ℂ) by
        push_cast
        field_simp
        ring]
    exact ofReal_ne_zero.mpr (by have := A.β_pos; positivity)
  have ha' : (A.a : ℂ) ≠ 0 := by exact_mod_cast ha.ne'
  rcases hb with hb | hb
  · have hd := A.dp_eq_zero_of_b k hb
    refine SingletHyp.isUnit_onePlusKS (ξ := 0) ⟨ha', fun n h0 ↦ A.t_sub_tp_ne k n ?_, ?_, hK⟩
    · linear_combination h0 - hd
    · have hp : A.tp k = A.a ^ 2 := by rw [tp, hb]; ring
      rw [hb]
      linear_combination -hd + hp
  · have hd := A.dm_eq_zero_of_b k hb
    refine SingletHyp.isUnit_onePlusKS (ξ := 0) ⟨ha', fun n h0 ↦ A.t_sub_tm_ne k n ?_, ?_, hK⟩
    · linear_combination h0 - hd
    · have hm : A.tm k = A.a ^ 2 := by rw [tm, hb]; ring
      rw [hb]
      linear_combination -hd + hm

end Params

end HubbardAtom
