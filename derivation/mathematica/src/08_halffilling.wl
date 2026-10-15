(* ::Section:: *)
(*8 Half filling, the example of the note, and a genuine divergence*)

(* ::Text:: *)
(*At mu = 0 the Cauchy entries rho/(z_n + z_m) become 0/0 on the mirror z_m = -z_n. The Lehmann expression of Section 1 is a divided difference of the entire function e^(beta s), continuous in its nodes, and the weights are continuous in mu, so chi at mu = 0 is the limit mu -> 0. On the mirror z + v = 2 mu, and lambda rho/(z + v) tends to lambda beta p0: C becomes beta p0 times the mirror.*)

(* ::Input:: *)
check["rho/(2 mu) -> beta p0", Simplify[Normal[Series[(rho /. thermal)/(2 mu), {mu, 0, 0}]] - (beta p0 /. thermal /. mu -> 0)] === 0]

(* ::Text:: *)
(*Corollary "half filling" as stated in the note: h = beta p0, s^ = 4 eps a^2 (p0 - p1) - q0, g^(z) = 1/(z^2 - s^), D^ = beta - 2 eps a S_k(u^)/u^.*)

(* ::Input:: *)
closedHF[eps_, k_, rules_] := Module[{st = Boole[k == 0], nr = numRules[k, Append[rules, mu -> 0]], H, Pz, e, l, sh, uh, Dh, bt, aN, p0N},
  {H, Pz, e} = Function[f, Function[x, Evaluate[toNum[f[x], nr]]]] /@ {Hf[eps, st], Pf[st], ef[eps, st]};
  {l, bt, aN, p0N, sh} = toNum[{lamf[eps], beta, a, p0, 4 eps a^2 (p0 - p1) - q0}, nr];
  uh = Sqrt[sh]; Dh = bt - 2 eps aN toNum[Sk[Mod[k, 2]][uh], nr]/uh;
  Function[{zn, zm, same, mir}, If[same, bt Pz[zn] (1/toNum[G2x /. z -> zn, nr] - H[zn]/e[zn]), 0] + 2 eps aN
    - If[mir, l bt p0N Pz[zn]/e[zn], 0] - 2 eps aN bt Pz[uh]/Dh/((zn^2 - sh) (zm^2 - sh))]];

(* ::Text:: *)
(*Box at mu = 0: the form of Section 2.2 with C = beta p0 times the mirror, P(z) chi = -beta H(z) delta + lambda beta p0 delta_(m nbar) + N(z, v)/P(v), with H, lambda, N as extracted there.*)

(* ::Input:: *)
chiBoxHF[eps_, k_, M_, rules_] := Module[{idx = boxIdx[k, M], st = Boole[k == 0], nr = numRules[k, Append[rules, mu -> 0]], bt, zs, X, Y, Pz, mat, n0},
  bt = beta /. nr; zs = N[I Pi (2 idx + 1 + k)/bt]; n0 = Length[idx];
  Pz = toNum[Pf[st][z], nr] /. z -> zs;
  {X, Y} = {Outer[Times, zs, ConstantArray[1, n0]], Outer[Times, ConstantArray[1, n0], zs]};
  mat = (toNum[NN[eps, st], nr] /. {z -> X, v -> Y})/Outer[Times, Pz, Pz] + DiagonalMatrix[-bt (toNum[Hf[eps, st][z], nr] /. z -> zs)/Pz];
  mat += SparseArray[Thread[Transpose[{Range[n0], Flatten[Position[idx, -# - k - 1] & /@ idx]}] -> toNum[lamf[eps] beta p0, nr]/Pz], {n0, n0}];
  <|"idx" -> idx, "chi" -> Normal[mat], "B" -> -bt (toNum[G2x, nr] /. z -> zs)/Pz|>];
compareHF[eps_, k_, rules_, M_ : 32] := Module[{g, win = Range[-3, 3], pos, ex, cf = closedHF[eps, k, rules], bt = N[beta /. rules]},
  g[MM_] := With[{bx = chiBoxHF[eps, k, MM, rules]}, <|"idx" -> bx["idx"], "Gamma" -> bt^2 (Inverse[bx["chi"]] - DiagonalMatrix[1/bx["B"]])|>];
  pos[gg_, n_] := First[FirstPosition[gg["idx"], n]];
  ex = With[{g1 = g[M], g2 = g[2 M]}, Table[2 g2["Gamma"][[pos[g2, n], pos[g2, m]]] - g1["Gamma"][[pos[g1, n], pos[g1, m]]], {n, win}, {m, win}]];
  Max[Abs[ex - Table[cf[N[I Pi (2 n + 1 + k)/bt], N[I Pi (2 m + 1 + k)/bt], n == m, m == -n - k - 1], {n, win}, {m, win}]]]/Max[Abs[ex]]];
devsHF = Table[compareHF[eps, k, rules], {rules, {{a -> 1, beta -> 3}, {a -> -3/2, beta -> 2}}}, {eps, {1, -1}}, {k, {0, 1, 2}}]
check["Corollary half filling vs finite boxes", Max[devsHF] < 10^-3]

(* ::Text:: *)
(*The corollary is the limit mu -> 0 of the theorem: the theorem at mu = 10^-20 (60 digits) against the corollary at mu = 0:*)

(* ::Input:: *)
limitDev = Table[Module[{mu1 = 10^-20, bt = 3, th, hf},
    th = closedPH[eps, k, {a -> 1, beta -> 3, mu -> 10^-20}, 60]; hf = closedHF[eps, k, {a -> 1, beta -> 3}];
    Max[Table[Abs[th[N[mu1 + I Pi (2 n + 1 + k)/bt, 60], N[mu1 + I Pi (2 m + 1 + k)/bt, 60], n == m] - hf[N[I Pi (2 n + 1 + k)/bt], N[I Pi (2 m + 1 + k)/bt], n == m, m == -n - k - 1]], {n, -2, 2}, {m, -2, 2}]]],
   {eps, {1, -1}}, {k, {0, 1, 2}}]
check["limit mu -> 0 of the theorem is the corollary", Max[limitDev] < 10^-12]

(* ::Text:: *)
(*The example of the note (r = d, U = 2, beta = 3, mu = 0.3, k = 1): box inverses for M = 128, 256, the extrapolation, and the closed form, at (n, m) = (0, 0), (0, nbar) = (0, -2), (2, -1):*)

(* ::Input:: *)
Module[{rules = {a -> 1, beta -> 3, mu -> 3/10}, g1, g2, pos, ce, ents = {{0, 0}, {0, -2}, {2, -1}}},
 g1 = gammaBox["d", 1, 128, rules]; g2 = gammaBox["d", 1, 256, rules]; ce = closedEntry["d", 1, rules];
 pos[g_, n_] := First[FirstPosition[g["idx"], n]];
 Grid[Prepend[Table[With[{e1 = g1["Gamma"][[pos[g1, nm[[1]]], pos[g1, nm[[2]]]]], e2 = g2["Gamma"][[pos[g2, nm[[1]]], pos[g2, nm[[2]]]]]},
     {nm, NumberForm[Chop[e1], 6], NumberForm[Chop[e2], 6], NumberForm[Chop[2 e2 - e1], 7], NumberForm[Chop[ce @@ nm], 7]}], {nm, ents}],
   {"(n, m)", "M = 128", "M = 256", "extrapolated", "Theorem"}], Frame -> All]]

(* ::Text:: *)
(*A genuine divergence: at mu = 0, d channel, k = 0, n = 0, the condition beta H(z_0) = -lambda h reads 3 (beta a)^2 = pi^2 (as p0 + p1 = 1/2), i.e. beta U = 2 pi/sqrt(3). There the mirror-odd vector e_0 - e_(-1) is a null vector of chi_d, on every finite box:*)

(* ::Input:: *)
check["beta H(z_0) + lambda h at mu = 0, k = 0", Simplify[(beta^2 (Hf[1, 1][I Pi/beta] + lamf[1] p0) /. norm /. thermal /. mu -> 0) - (3 (beta a)^2 - Pi^2)] === 0]
Module[{bx = chiBoxHF[1, 0, 32, {a -> Pi/(2 Sqrt[3]), beta -> 2}], vec},
 vec = Boole[# == 0] - Boole[# == -1] & /@ bx["idx"];
 check["null vector of chi_d at beta U = 2 pi/sqrt(3)", Norm[bx["chi"] . vec]/Norm[bx["chi"]] < 10^-14]]
