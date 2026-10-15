(* ::Title:: *)
(*The irreducible vertex of the doped Hubbard atom*)

(* ::Text:: *)
(*A computer-algebra companion to the note doped-gamma-derivation (derivation/ in HubbardAtoms.jl). It starts from the Hamiltonian, derives the generalized susceptibility from the Lehmann representation (replacing the note's one external input, Ref. [Essl et al. 2024, Eq. (D3)]), and then replays the inversion of the note step by step. Objects are computed from earlier results where possible: H, lambda, N, B, L_s, L_t, gamma_+-, the coefficients D_v, E_v of Step 1, the polynomial W of Eq. (source), and the singlet and triplet vertices are extracted, not typed in. The note's formulas are typed in only to compare with. Each comparison goes through check[label, value]; the last section lists all of them.*)

(* ::Text:: *)
(*Scope. Symbolic: the input, all rational identities and lattice sums of the proofs, the pp theorems, and Steps 1-4 of the proof for the density and magnetic channels (the final assembly of Step 4 is checked through its two identities). Numerical: every theorem, including half filling and its limit mu -> 0, against brute-force inversion of the Section 1 susceptibility on finite frequency boxes, and a genuine divergence. Not reproduced, only argued in the note: the analytic statements (convergence, continuity in mu and beta, the finite-box limit, the exceptional parameter sets). The inversion strategy itself (multiplying by P, the Cauchy-column ansatz, the pole pairs) is taken from the note; the notebook executes it.*)

(* ::Input:: *)
$checks = {};
check[label_String, value_] := (AppendTo[$checks, label -> TrueQ[And @@ Flatten[{value}]]]; Last[$checks]);
SetOptions[$Output, PageWidth -> 120];
