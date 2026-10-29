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
      ``HubbardAtom.FParams.continuousAt_gammaOpTFam_of_TGeneric] do
    unless audited.contains anchor do
      throwError "{anchor} was not audited; the check is vacuous"
  if checked < 100 then
    throwError "only {checked} declarations matched; the module filter is broken"
  logInfo m!"axiom audit passed: {checked} declarations, axioms within {expectedAxioms}"
