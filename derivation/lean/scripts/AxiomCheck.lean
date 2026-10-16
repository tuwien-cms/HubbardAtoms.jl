import HubbardAtomLean

/-!
# Axiom audit

Checks every declaration of the `HubbardAtomLean` library with the kernel's own record of the
axioms it depends on: a `sorry` anywhere upstream, or any axiom beyond the three standard ones
(for instance from `native_decide`), fails the check. Run with
`lake env lean scripts/AxiomCheck.lean`.
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
  for (n, _) in env.constants.toList do
    unless isOurs env n && !n.isInternal && !n.hasMacroScopes do continue
    checked := checked + 1
    let axs ← collectAxioms n
    if axs.contains ``sorryAx then throwError "unproved: {n}"
    let bad := axs.filter (!expectedAxioms.contains ·)
    unless bad.isEmpty do throwError "unexpected axioms on {n}: {bad.toList}"
  -- Vacuity anchors: the filter must have reached the library.
  for anchor in [``HubbardAtom.Params.gammaT_of_isRightInverse,
      ``HubbardAtom.Stable.PHData.K_div_rho, ``HubbardAtom.Params.αeven_eq_package,
      ``HubbardAtom.Params.hasSum_one_div_t_sub, ``HubbardAtom.Params.gammaS_eq] do
    unless env.contains anchor do
      throwError "{anchor} is missing; the check is vacuous"
  if checked < 100 then
    throwError "only {checked} declarations matched; the module filter is broken"
  logInfo m!"axiom audit passed: {checked} declarations, axioms within {expectedAxioms}"
