import HubbardAtomLean

/-!
# Axiom audit

Checks every constant defined in a module of the `HubbardAtomLean` library, private and auxiliary
ones included, with the kernel's own record of the axioms it depends on: a `sorry` anywhere
upstream, or any axiom beyond the three standard ones (for instance from `native_decide`), fails
the check. Run with `lake env lean scripts/AxiomCheck.lean`.
-/

open Lean

/-- The axioms the library is known and intended to depend on. -/
def expectedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

/-- Whether `n` is defined in one of the `HubbardAtomLean` modules. -/
def isOurs (env : Environment) (n : Name) : Bool :=
  match env.getModuleIdxFor? n with
  | none => false
  | some idx =>
    match env.header.moduleNames[idx.toNat]? with
    | some m => (`HubbardAtomLean).isPrefixOf m
    | none => false

run_cmd Elab.Command.liftCoreM do
  let env ← getEnv
  let mut checked : Nat := 0
  let mut audited : NameSet := {}
  for (n, _) in env.constants.toList do
    unless isOurs env n do continue
    checked := checked + 1
    audited := audited.insert n
    let axs ← collectAxioms n
    if axs.contains ``sorryAx then throwError "unproved: {n}"
    let bad := axs.filter (!expectedAxioms.contains ·)
    unless bad.isEmpty do throwError "unexpected axioms on {n}: {bad.toList}"
  -- Vacuity anchors: the filter must have reached the library.
  for anchor in [``HubbardAtom.Params.gammaT_of_isRightInverse,
      ``HubbardAtom.Stable.PHData.K_div_rho, ``HubbardAtom.Params.αeven_eq_package,
      ``HubbardAtom.Params.hasSum_one_div_t_sub, ``HubbardAtom.Params.gammaS_eq,
      ``HubbardAtom.Params.identity_A7, ``HubbardAtom.Params.DMGeneric.gammaR_eq,
      ``HubbardAtom.Params.DMGeneric.chiR_isLeftInverse, ``HubbardAtom.Params.propagator_eq,
      ``HubbardAtom.Params.chiLehmann_upUp, ``HubbardAtom.Params.chiLehmann_upDn,
      ``HubbardAtom.Params.gammaR_zero_of_a_eq_zero, ``HubbardAtom.Params.DMGeneric.gammaR_neg₁,
      ``HubbardAtom.Params.SingletHyp.gammaS_eq, ``HubbardAtom.Params.HFHyp.gammaHF_eq,
      ``HubbardAtom.Params.DMConf.gammaR_eq, ``HubbardAtom.Params.SingletLZ.gammaS_eq,
      ``HubbardAtom.Params.HFLZ.gammaHF_eq, ``HubbardAtom.FParams.TGeneric.gammaTUp_eq,
      ``HubbardAtom.FParams.THFHyp.gammaTHF_eq, ``HubbardAtom.FParams.gammaADn_eq,
      ``HubbardAtom.FParams.gammaEUp_eq, ``HubbardAtom.FParams.PL_mul_chiL_updn,
      ``HubbardAtom.FParams.PL_mul_chiL_opp_all,
      ``HubbardAtom.FParams.gammaTUp_static_hf,
      ``HubbardAtom.FParams.chiL_updn_divided_difference,
      ``HubbardAtom.FParams.hasDerivAt_Gs_up_h, ``HubbardAtom.FParams.hasSum_chiL_updn,
      ``HubbardAtom.FParams.chiL_isRightInverse,
      ``HubbardAtom.FParams.chiL_isLeftInverse_transpose,
      ``HubbardAtom.FParams.LGeneric.gammaL_eq, ``HubbardAtom.LCert.cert_Z12,
      ``HubbardAtom.FParams.chiL_isLeftInverse, ``HubbardAtom.FParams.invL_symm,
      ``HubbardAtom.Params.hasSum_chiR_mul_g,
      ``HubbardAtom.Params.norm_chiR_le,
      ``HubbardAtom.isCompactOperator_hsOp,
      ``HubbardAtom.Params.isUnit_one_add_K0,
      ``HubbardAtom.Params.isUnit_onePlusK_iff,
      ``HubbardAtom.Params.exists_bound_of_null,
      ``HubbardAtom.Params.DMGeneric.gammaOp_eq,
      ``HubbardAtom.Params.DMGeneric.tendsto_gammaBox,
      ``HubbardAtom.Params.gammaOp_eq_chiInv,
      ``HubbardAtom.Params.eq_chiInv_of_hasSum,
      ``HubbardAtom.Params.chiInv_isLeftInverse,
      ``HubbardAtom.Params.DMGeneric.invR_eq_chiInv,
      ``HubbardAtom.Params.continuousAt_gammaOp,
      ``HubbardAtom.Params.lam_rho_Zph_eq,
      ``HubbardAtom.Params.Phik_zero,
      ``HubbardAtom.Params.hasSum_chiR_mul_g_of_Phi,
      ``HubbardAtom.Params.eigen_transform,
      ``HubbardAtom.Params.im_mul_im_Ke_pos,
      ``HubbardAtom.FParams.chiL_of_h,
      ``HubbardAtom.FParams.hasSum_invLh0_mul_chiL,
      ``HubbardAtom.FParams.eq_invLh0_of_hasSum,
      ``HubbardAtom.FParams.gammaL_h0,
      ``HubbardAtom.simplexW_rotate,
      ``HubbardAtom.FParams.chiLehmann_dag_left,
      ``HubbardAtom.FParams.chiLehmann_dag_right,
      ``HubbardAtom.FParams.chiL_shiba,
      ``HubbardAtom.FParams.bL_shiba,
      ``HubbardAtom.FParams.orderedTerm_rotate,
      ``HubbardAtom.FParams.hasSum_chiL_mul_invLmu0,
      ``HubbardAtom.FParams.hasSum_invLmu0_mul_chiL,
      ``HubbardAtom.FParams.eq_invLmu0_of_hasSum,
      ``HubbardAtom.FParams.gammaL_mu0,
      ``HubbardAtom.OpLattice.gammaOpX_eq_of_isRightInverse,
      ``HubbardAtom.OpLattice.tendsto_gammaBoxX,
      ``HubbardAtom.FParams.chiDecay_chiTUp,
      ``HubbardAtom.FParams.isUnit_onePlusKT_iff,
      ``HubbardAtom.FParams.chiInvT_isLeftInverse,
      ``HubbardAtom.FParams.TGeneric.gammaOpT_eq,
      ``HubbardAtom.FParams.TGeneric.tendsto_gammaBoxT,
      ``HubbardAtom.FParams.chiDecay_chiLe,
      ``HubbardAtom.FParams.isUnit_onePlusKL_iff,
      ``HubbardAtom.FParams.chiInvL_isLeftInverse,
      ``HubbardAtom.FParams.LGeneric.gammaOpL_eq,
      ``HubbardAtom.FParams.LGeneric.tendsto_gammaBoxL,
      ``HubbardAtom.OpLattice.continuousAt_gammaOpFam,
      ``HubbardAtom.FParams.continuousAt_gammaOpTFam,
      ``HubbardAtom.FParams.continuousAt_gammaOpTFam_of_TGeneric,
      ``HubbardAtom.Params.differentiable_Ehat,
      ``HubbardAtom.Params.norm_Ehat_le_circ,
      ``HubbardAtom.Params.eq_sum_gs_of_eigen,
      ``HubbardAtom.Params.abs_mu_lt_of_Ks_eq_zero,
      ``HubbardAtom.Params.Ks_static_eq_zero_iff,
      ``HubbardAtom.Params.Ks_ge_of_curve,
      ``HubbardAtom.Params.singletGeneric_of_k_ne,
      ``HubbardAtom.Params.finite_roots_ell,
      ``HubbardAtom.Params.not_isUnit_onePlusK_iff,
      ``HubbardAtom.Params.Ke_ne_zero_mag,
      ``HubbardAtom.Params.isUnit_onePlusK_mag,
      ``HubbardAtom.Params.zero_mode_dens,
      ``HubbardAtom.Params.densF_pos_of_zero_mode,
      ``HubbardAtom.Params.Phik_ne_zero_of_densF_neg,
      ``HubbardAtom.Params.gammaR_divided_difference,
      ``HubbardAtom.Params.gammaS_divided_difference,
      ``HubbardAtom.FParams.gammaTUp_divided_difference,
      ``HubbardAtom.FParams.gammaL_divided_difference,
      ``HubbardAtom.Params.summable_inv_norm_zOp_sq,
      ``HubbardAtom.Params.bdd_of_inv_zOp,
      ``HubbardAtom.Params.chiDecay_chiS,
      ``HubbardAtom.Params.isUnit_onePlusKS_iff,
      ``HubbardAtom.Params.chiInvS_isLeftInverse,
      ``HubbardAtom.Params.SingletHyp.gammaOpS_eq,
      ``HubbardAtom.Params.SingletLZ.gammaOpS_eq,
      ``HubbardAtom.Params.tendsto_gammaBoxS,
      ``HubbardAtom.OpLattice.ChiDecay.of_eventuallyEq,
      ``HubbardAtom.Params.chiDecay_chiR_all,
      ``HubbardAtom.Params.isUnit_onePlusKR_iff,
      ``HubbardAtom.Params.chiInvR_isLeftInverse,
      ``HubbardAtom.Params.HFHyp.gammaOpR_eq,
      ``HubbardAtom.Params.HFLZ.gammaOpR_eq,
      ``HubbardAtom.Params.DMGeneric.gammaOpR_eq,
      ``HubbardAtom.Params.tendsto_gammaBoxR,
      ``HubbardAtom.Params.hasSum_chiS_kappa,
      ``HubbardAtom.Params.isUnit_onePlusKS_iff_Ks,
      ``HubbardAtom.Params.isUnit_onePlusKS_of_edge,
      ``HubbardAtom.Params.isUnit_onePlusK_iff_exists_isRightInverse,
      ``HubbardAtom.Params.DMConf.gammaOp_eq,
      ``HubbardAtom.Params.contHyp_R,
      ``HubbardAtom.Params.continuousAt_gammaOpR,
      ``HubbardAtom.Params.continuousAt_gammaOpR_of_HFHyp,
      ``HubbardAtom.Params.chiRegS_eq_uniform,
      ``HubbardAtom.Params.contHyp_S,
      ``HubbardAtom.Params.continuousAt_gammaOpS,
      ``HubbardAtom.Params.continuousAt_gammaOpS_of_SingletHyp,
      ``HubbardAtom.Params.tendsto_gammaS_of_SingletLZ,
      ``HubbardAtom.Params.tendsto_gammaHF_of_HFLZ,
      ``HubbardAtom.Params.tendsto_gammaR_of_HFHyp,
      ``HubbardAtom.Params.tendsto_gammaR_of_HFLZ,
      ``HubbardAtom.FParams.norm_chiL_opp_le_all_of,
      ``HubbardAtom.FParams.chiDecay_chiLe_all,
      ``HubbardAtom.FParams.bubbleDecay_bLe_all,
      ``HubbardAtom.FParams.chiL_symm_all,
      ``HubbardAtom.FParams.isUnit_onePlusKLAll_iff,
      ``HubbardAtom.FParams.LGeneric.gammaOpLAll_eq,
      ``HubbardAtom.FParams.gammaOpLAll_eq_gammaOpL,
      ``HubbardAtom.FParams.continuousAt_famF_chiL,
      ``HubbardAtom.FParams.contHyp_LAll,
      ``HubbardAtom.FParams.continuousAt_gammaOpLAll,
      ``HubbardAtom.FParams.tendsto_gammaL_of_LGeneric,
      ``HubbardAtom.FParams.continuousAt_gammaOpLAll_of_LGeneric,
      ``HubbardAtom.Params.eventually_singletHyp,
      ``HubbardAtom.Params.tendsto_gammaS_of_SingletLZ_nhdsNE] do
    unless audited.contains anchor do
      throwError "{anchor} was not audited; the check is vacuous"
  if checked < 100 then
    throwError "only {checked} declarations matched; the module filter is broken"
  logInfo m!"axiom audit passed: {checked} declarations, axioms within {expectedAxioms}"
