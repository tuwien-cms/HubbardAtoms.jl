(* ::Section:: *)
(*3 Matsubara sums*)

(* ::Text:: *)
(*At transfer k the centered lattice is z_n = mu + i pi (2n + 1 + k)/beta: the offsets z_n - mu are fermionic frequencies for even k and bosonic ones for odd k. par = Mod[k, 2]. Mathematica sums the symmetric Mittag-Leffler series in closed form:*)

(* ::Input:: *)
{FullSimplify[Sum[1/((2 n + 1) I Pi/beta - y) + 1/(-(2 n + 1) I Pi/beta - y), {n, 0, Infinity}]],
 FullSimplify[-1/y + Sum[1/(2 n I Pi/beta - y) + 1/(-2 n I Pi/beta - y), {n, 1, Infinity}]]}

(* ::Text:: *)
(*So S_k(x) := PV Sum_n 1/(z_n - x) is (Lemma "basic sum"):*)

(* ::Input:: *)
Sk[0][x_] := -beta/2 Tanh[beta (x - mu)/2]; Sk[1][x_] := -beta/2 Coth[beta (x - mu)/2];
check["S_k equals the two sums", FullSimplify[{Sum[1/((2 n + 1) I Pi/beta - y) + 1/(-(2 n + 1) I Pi/beta - y), {n, 0, Infinity}] - Sk[0][y + mu],
    -1/y + Sum[1/(2 n I Pi/beta - y) + 1/(-2 n I Pi/beta - y), {n, 1, Infinity}] - Sk[1][y + mu]}] === {0, 0}]

(* ::Text:: *)
(*Both satisfy the Riccati equation S' = S^2 - beta^2/4, and S_k(mu + y) is odd in y. So every derivative of S at a point follows from its value there: sTaylor[s0, d, order] is the Taylor polynomial of S(x0 + d) when S(x0) = s0.*)

(* ::Input:: *)
check["Riccati equation and oddness", Simplify[Flatten[Table[{D[Sk[par][x], x] - (Sk[par][x]^2 - beta^2/4), Sk[par][mu + y] + Sk[par][mu - y]}, {par, {0, 1}}]]] === {0, 0, 0, 0}]
sTaylor[s0_, d_, order_] := Module[{poly = s0, tt}, Do[poly = s0 + Integrate[Expand[(poly^2 - beta^2/4) /. d -> tt], {tt, 0, d}]; poly = Normal[Series[poly, {d, 0, order}]], {order}]; poly];
sTaylor[s0, d, 3]

(* ::Text:: *)
(*Residue rule (Lemma "residue rule"): for a proper rational phi without poles on the lattice, Sum_n phi(z_n) = Sum over poles y of Res_(t = y) phi(t) S_k(t). resS gives the residue at one pole y of order m, given the value s0 = S_k(y).*)

(* ::Input:: *)
resS[phi_, t_, y_, m_, s0_] := Module[{d}, Coefficient[Expand[Normal[Series[(phi /. t -> y + d) sTaylor[s0, d, m], {d, 0, -1}]]], d, -1]];
(* test against direct summation, k even, a double and a simple pole *)
With[{bt = 2, m0 = 1/5, x0 = 3/10 + 7/10 I, phi = 1/((tt - 3/10 - 7/10 I)^2 (tt + 1))},
 Module[{z = m0 + I Pi (2 # + 1)/bt &},
  check["residue rule vs direct sum", Abs[N[(resS[phi, tt, x0, 2, Sk[0][x0]] + resS[phi, tt, -1, 1, Sk[0][-1]]) /. {beta -> bt, mu -> m0}] - N[Sum[phi /. tt -> z[n], {n, -40000, 39999}]]] < 10^-8]]]

(* ::Subsection:: *)
(*3.1 Special values of S_k*)

(* ::Text:: *)
(*Thermal weights p_i = e^(-beta E_i)/Z; only here are they needed explicitly. kk is an integer, k = 2 kk + par, and w = i omega_k = 2 pi i k/beta.*)

(* ::Input:: *)
Zsum = Total[Exp[-beta energy] {1, 1, 1, 1}] /. Thread[weight -> weight];
thermal = Thread[{p0, p1, p2} -> (Exp[-beta {energy[[1]], energy[[2]], energy[[4]]}]/Total[Exp[-beta energy]])];
wk[par_] := 2 Pi I (2 kk + par)/beta;
Splus = beta (p1 - p0)/(2 (p1 + p0)); Sminus = beta (p2 - p1)/(2 (p1 + p2));

(* ::Text:: *)
(*Eq. (shift): S_k(x +- w/2) = -beta/2 tanh(beta (x - mu)/2) for both parities. Hence S_k takes the value S_+ at both zeros -a +- w/2 of Q_+, and S_- at both zeros a +- w/2 of Q_-:*)

(* ::Input:: *)
check["Eq. (shift)", Table[FullSimplify[Sk[par][x + s wk[par]/2] + beta/2 Tanh[beta (x - mu)/2], Element[kk, Integers]], {par, {0, 1}}, {s, {1, -1}}] === {{0, 0}, {0, 0}}]
check["S_+ and S_- at the zeros of P", Table[FullSimplify[{Sk[par][-a + s wk[par]/2] - Splus, Sk[par][a + s wk[par]/2] - Sminus} /. thermal, Element[kk, Integers] && beta > 0], {par, {0, 1}}, {s, {1, -1}}] === {{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}}]

(* ::Text:: *)
(*Lemma "values on the reflected lattice": on the points -z_n, S_k is constant, S_k(-z_n) = (beta/2) coth(beta mu), and by the Riccati equation S_k'(-z_n) = beta^2/(4 sinh^2(beta mu)). With the weights: rho S_k(-z_n) = cbar := beta (p0 + p2)/2 and rho^2 S_k'(-z_n) = h^2 := beta^2 p0 p2.*)

(* ::Input:: *)
check["S_k on the reflected lattice", Table[FullSimplify[Sk[par][-(mu + I Pi (2 n + 1 + 2 kk + par)/beta)] - beta/2 Coth[beta mu], Element[{n, kk}, Integers]], {par, {0, 1}}] === {0, 0}]
cbar = beta (p0 + p2)/2; hsq = beta^2 p0 p2;
check["rho S = cbar, rho^2 S' = h^2", FullSimplify[{rho beta/2 Coth[beta mu] - cbar, rho^2 ((beta/2 Coth[beta mu])^2 - beta^2/4) - hsq} /. thermal] === {0, 0}]
