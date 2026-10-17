import HubbardAtomLean

/-!
# Linter fixtures

Each declaration below violates exactly one option of the linter profile in `lakefile.toml`.
`scripts/check.sh` builds this library and asserts that every option fires: the `weak.` prefix
silently ignores a misspelled option, so a profile without a fixture could be vacuous. This
library is not part of the default target.
-/

-- `linter.style.multiGoal`: the first `trivial` acts while two goals are open.
example : True ∧ True := by
  constructor
  trivial
  trivial

-- `linter.style.show`: this `show` changes the goal.
example : (1 : ℕ) + 1 = 2 := by
  show 2 = 2
  rfl

-- `linter.style.setOption`: a pretty-printer option left in a proof. (The linter unwraps a
-- command-level `set_option ... in`, so the fixture has to be tactic-level.)
example : True := by
  set_option pp.all true in
  trivial

-- `linter.style.maxHeartbeats`: a heartbeat override without an explanation.
set_option maxHeartbeats 1000000 in
example : True := trivial

/-- A function defined by pattern matching, which generates `fixtureMatch.match_1`. -/
def fixtureMatch : ℕ → ℕ
  | 0 => 0
  | _ + 1 => 1

-- `linter.auxLemma`: a reference to an auto-generated auxiliary declaration.
example := @fixtureMatch.match_1

-- `linter.style.missingEnd`: this section is never closed.
section Unclosed
