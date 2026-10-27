#!/usr/bin/env bash
# Verification gate for the formalization. Run from derivation/lean.
set -euo pipefail

# The modules, in import order.
modules="Basic Triplet StableForms Thermal Matsubara SingletInput Singlet PHInput PHSums
  PHIdentities DensityMagnetic OrderedIntegrals Lehmann LehmannInput UZero DMLifts SingletZero
  HalfFilling Confluent SingletLZ HalfFillingLZ FieldBasic FieldLehmann FieldTAlg FieldInput
  FieldTSums FieldTIdentities FieldTransverse FieldPair FieldEqualSpin FieldTHalfFilling
  FieldLInput FieldTZero DividedDifferences StaticResponse WardIdentities SpectralIdentity"

# 1. Rebuild the library from a cold cache, so that every module is compiled and linted, and fail
#    on any warning (a `sorry` is only a warning). With SERIAL=1 the modules are compiled one at a
#    time, in import order, which lowers the peak memory at some cost in time.
rm -rf .lake/build
: > .lake/build.log
if [ "${SERIAL:-0}" = 1 ]; then
  for m in $modules; do
    lake build "+HubbardAtomLean.$m" 2>&1 | tee -a .lake/build.log
  done
fi
lake build HubbardAtomLean 2>&1 | tee -a .lake/build.log
for m in $modules; do
  grep -q "Built HubbardAtomLean.$m" .lake/build.log \
    || { echo "gate is vacuous: HubbardAtomLean.$m was not compiled" >&2; exit 1; }
done
if grep -n "warning" .lake/build.log; then
  echo "the build produced warnings" >&2
  exit 1
fi

# 2. Axiom audit: no `sorry`, no unexpected axioms.
lake env lean scripts/AxiomCheck.lean

# 3. Every option of the linter profile fires on its fixture.
lake build LinterFixtures 2>&1 | tee .lake/fixtures.log || true
for opt in linter.auxLemma linter.style.maxHeartbeats linter.style.missingEnd \
  linter.style.multiGoal linter.style.setOption linter.style.show; do
  grep -q "$opt" .lake/fixtures.log \
    || { echo "linter option $opt did not fire on its fixture" >&2; exit 1; }
done

echo "all checks passed"
