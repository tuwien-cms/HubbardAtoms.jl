(* ::Section:: *)
(*7 The theorems against brute-force inversion on finite boxes*)

(* ::Text:: *)
(*The closed forms as stated in the note, typed in once and evaluated numerically. numRules[k, rules] gives numerical values for the weights (from Section 3.1) and the transfer w = 2 pi i k/beta.*)

(* ::Input:: *)
numRules[k_, rules_, prec_ : MachinePrecision] := Module[{nr = N[Join[rules, thermal /. rules], prec]}, Join[nr, {w -> 2 Pi I k/beta, wpp -> 2 Pi I k/beta} /. nr]];
toNum[ex_, nr_] := ex /. hsq -> beta^2 p0 p2 /. norm /. nr;

(* ::Text:: *)
(*Density and magnetic (Theorem "density and magnetic"): roots s_1,2 of e(z) = (z^2 - s1)(z^2 - s2), alpha_k(u) = rho S_k(u) - cbar, K_j, Z_j, gamma_j; for m and k = 0 the atomic pair s = a^2 is omitted.*)

(* ::Input:: *)
closedPH[eps_, k_, rules_, prec_ : MachinePrecision] := Module[{st = Boole[k == 0], nr = numRules[k, rules, prec], H, Pz, e, l, cl, s12, keep, gam, F, dF, rhoN, bt, aN},
  {H, Pz, e} = Function[f, Function[x, Evaluate[toNum[f[x], nr]]]] /@ {Hf[eps, st], Pf[st], ef[eps, st]};
  {l, rhoN, bt, aN} = toNum[{lamf[eps], rho, beta, a}, nr];
  cl = CoefficientList[Expand[e[x]], x];
  s12 = s /. NSolve[cl[[1]] + cl[[3]] s + s^2 == 0, s];
  keep = If[eps == -1 && k == 0, Select[{1, 2}, Abs[s12[[#]] - aN^2] > 10^-8 &], {1, 2}];
  gam = Table[Module[{uj = Sqrt[s12[[j]]], al, K},
     al = toNum[rho Sk[Mod[k, 2]][uj] - cbar, nr]; K = bt H[-uj] + l al;
     l rhoN uj Pz[uj] (bt H[-uj] - l al)/K/(s12[[j]] - s12[[3 - j]])], {j, keep}];
  F = Function[x, Evaluate[Pz[x]/e[x]]]; dF = Function[x, Evaluate[D[Pz[x]/e[x], x]]];
  Function[{zn, zm, same}, If[same, bt Pz[zn] (1/toNum[G2x /. z -> zn, nr] - H[-zn]/e[zn]) - l rhoN (F[zn] + zn dF[zn])/(2 zn), - l rhoN (zn F[zn] - zm F[zm])/(zn^2 - zm^2)]
    + 2 eps aN + Sum[gam[[i]]/((zn^2 - s12[[keep[[i]]]]) (zm^2 - s12[[keep[[i]]]])), {i, Length[keep]}]]];

(* ::Text:: *)
(*Singlet and triplet (Theorems "triplet", "singlet"), with T_k(t) = -S_k(mu + xi)/xi, xi^2 = b^2/4 - t:*)

(* ::Input:: *)
closedPP[r_, k_, rules_] := Module[{nr = numRules[k, rules], AsN, Ks, diagF, sep},
  AsN = toNum[As, nr];
  If[r === "t",
   diagF = With[{ex = toNum[GtNote, nr]}, ex /. t -> # &];
   Function[{tn, tmn, dg, mr}, diagF[tn] (Boole[dg] - Boole[mr])],
   Ks = toNum[1 + 2 a (-Sk[Mod[k, 2]][mu + Sqrt[bb^2/4 + As]]/Sqrt[bb^2/4 + As])/beta, nr];
   diagF = With[{ex = toNum[beta a^2 (rho^2 - 1 + 8 p1) PPt[t]/(GGt[t] (t + As)), nr]}, ex /. t -> # &];
   sep = Function[{tn, tmn}, Evaluate[toNum[4 a - 4 a PPt[-As]/(kS (tn + As) (tmn + As)), nr] /. kS -> Ks]];
   Function[{tn, tmn, dg, mr}, diagF[tn] (Boole[dg] + Boole[mr]) + sep[tn, tmn]]]];

(* ::Text:: *)
(*Brute force: the susceptibility as derived in Sections 1-2 (generic part plus the delta parts, not the compact forms of the theorems' proofs) on the box |2n + 1 + k| <= 2M + 1, inverted numerically; Gamma = beta^2 (chi^-1 - eta B^-1).*)

(* ::Input:: *)
boxIdx[k_, M_] := Select[Range[-M - k - 2, M + 2], Abs[2 # + 1 + k] <= 2 M + 1 &];
gammaBox[r_, k_, M_, rules_] := Module[{idx = boxIdx[k, M], nr = numRules[k, rules], bt, vs, X, Y, mat, Bv, f},
  bt = beta /. nr; vs = N[I Pi (2 idx + 1)/bt];
  f[ex_, vars_] := Function[Evaluate[vars], Evaluate[toNum[ex, nr]]];
  If[MemberQ[{"d", "m"}, r],
   Module[{c = chiR[If[r == "d", 1, -1]], zs = (mu /. nr) + vs + (w /. nr)/2},
    {X, Y} = {Outer[Times, zs, ConstantArray[1, Length[zs]]], Outer[Times, ConstantArray[1, Length[zs]], zs]};
    mat = f[c["gen"], {z, v}][X, Y] + DiagonalMatrix[f[c["dd"], {z}][zs]];
    If[k == 0, mat += f[c["dk"] /. w -> 0, {z, v}][X, Y]];
    Bv = f[-beta gf[vn] gf[vn + w], {vn}][vs]],
   Module[{c = chiPP[r], mirIdx},
    {X, Y} = {Outer[Times, vs, ConstantArray[1, Length[vs]]], Outer[Times, ConstantArray[1, Length[vs]], vs]};
    mat = If[c["gen"] === 0, 0, f[c["gen"], {vn, vm}][X, Y]] + DiagonalMatrix[f[c["dd"], {vn}][vs]];
    mirIdx = Flatten[Position[idx, -# - k - 1] & /@ idx];
    mat += SparseArray[Thread[Transpose[{Range[Length[idx]], mirIdx}] -> f[c["mir"], {vn}][vs]], {Length[idx], Length[idx]}];
    Bv = f[Bpp, {vn}][vs]]];
  <|"idx" -> idx, "Gamma" -> bt^2 (Inverse[Normal[mat]] - If[r == "s", -1, 1] DiagonalMatrix[1/Bv])|>];

(* ::Text:: *)
(*Relative deviation, over the entries -3 <= n, m <= 3, of the extrapolated box inverse 2 Gamma(2M) - Gamma(M) (box error O(1/M^2)) from the closed form:*)

(* ::Input:: *)
closedEntry[r_, k_, rules_] := Module[{bt = beta /. rules, mu0 = mu /. rules, cf},
  If[MemberQ[{"d", "m"}, r],
   cf = closedPH[If[r == "d", 1, -1], k, rules];
   Function[{n, m}, cf[N[mu0 + I Pi (2 n + 1 + k)/bt], N[mu0 + I Pi (2 m + 1 + k)/bt], n == m]],
   cf = closedPP[r, k, rules];
   Function[{n, m}, Module[{b0 = N[2 mu0 - 2 Pi I k/bt], x1 = N[mu0 + I Pi (2 n + 1)/bt], x2 = N[mu0 + I Pi (2 m + 1)/bt]},
     cf[x1 (b0 - x1), x2 (b0 - x2), n == m, m == -n - k - 1]]]]];
compare[r_, k_, rules_, M_ : 32] := Module[{g1 = gammaBox[r, k, M, rules], g2 = gammaBox[r, k, 2 M, rules], win = Range[-3, 3], pos, ex, ce = closedEntry[r, k, rules], cf},
  pos[g_, n_] := First[FirstPosition[g["idx"], n]];
  ex = Table[2 g2["Gamma"][[pos[g2, n], pos[g2, m]]] - g1["Gamma"][[pos[g1, n], pos[g1, m]]], {n, win}, {m, win}];
  cf = Table[ce[n, m], {n, win}, {m, win}];
  Max[Abs[ex - cf]]/Max[Abs[cf]]];

(* ::Text:: *)
(*All four channels at transfers k = 0, 1, 2, for mu inside and outside the gap and both signs of U:*)

(* ::Input:: *)
paramSets = {{a -> 1, beta -> 3, mu -> 3/10}, {a -> -3/2, beta -> 2, mu -> -4/5}, {a -> 1/2, beta -> 5, mu -> 7/5}};
devs = Flatten[Table[{r, k, rules, compare[r, k, rules]}, {rules, paramSets}, {r, {"d", "m", "s", "t"}}, {k, {0, 1, 2}}], 2];
Grid[Prepend[{#[[1]], #[[2]], Row[{"U = ", 2 a, ", beta = ", beta, ", mu = ", mu} /. #[[3]]], ScientificForm[#[[4]], 2]} & /@ devs, {"channel", "k", "parameters", "rel. deviation"}], Frame -> All]
check["all channels agree with finite boxes", Max[devs[[All, 4]]] < 10^-3]
