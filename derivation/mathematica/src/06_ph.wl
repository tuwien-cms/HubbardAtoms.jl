(* ::Section:: *)
(*6 Density and magnetic channels: the proof of Theorem "density and magnetic"*)

(* ::Text:: *)
(*A case is {eps, st}: eps = +1 (d) or -1 (m), st = delta_(k0) (st = 1 means w = 0). All functions below are the ones extracted in Section 2.2. P(z) chi_r = -beta H(z) delta + lambda C + N(z, v)/P(v). The zeros of P and the values of S_k there (Section 3.1): four simple zeros for k != 0, two double zeros for k = 0.*)

(* ::Input:: *)
cases = Tuples[{{1, -1}, {0, 1}}];
wr[st_] := If[st == 1, {w -> 0}, {}];
Hf[eps_, st_][x_] := Hx[eps] /. z -> x /. wr[st];
Nf[eps_, st_][x_, y_] := NN[eps, st] /. {z -> x, v -> y};
N2f[eps_, st_][x_] := N2x[eps, st] /. v -> x;
Pf[st_][x_] := P[x] /. wr[st];
lamf[eps_] := lam[eps];
pZeros[0] = {{-a + w/2, 1, Splus}, {-a - w/2, 1, Splus}, {a + w/2, 1, Sminus}, {a - w/2, 1, Sminus}};
pZeros[1] = {{-a, 2, Splus}, {a, 2, Sminus}};
check["zeros of P", And @@ Table[Simplify[Pf[st][t] - Product[(t - y[[1]])^y[[2]], {y, pZeros[st]}]] === 0, {st, {0, 1}}]]
ef[eps_, st_][x_] := Hf[eps, st][x] Hf[eps, st][-x] - lamf[eps]^2 hsq/beta^2;

(* ::Subsection:: *)
(*6.1 Lemma "rank-two sums"*)

(* ::Text:: *)
(*I(z; x) = Sum_l N(z, z_l)/(P(z_l)(z_l - x)). By the residue rule its value is the residue at t = x, S(x) N(z, x)/P(x) = [cbar + alpha(x)] N(z, x)/(rho P(x)), plus the residues at the zeros of P. Ibar collects everything except the alpha term:*)

(* ::Input:: *)
Ibar[eps_, st_][zz_, x_] := cbar Nf[eps, st][zz, x]/(rho Pf[st][x]) + Sum[resS[Nf[eps, st][zz, t]/(Pf[st][t] (t - x)), t, y[[1]], y[[2]], y[[3]]], {y, pZeros[st]}];

(* ::Subsection:: *)
(*6.2 Step 2: the constant (identity A6)*)

(* ::Text:: *)
(*For y = 1: -beta H(z) + lambda cbar + PV Sum_l N(z, z_l)/P(z_l), the sum again by the residue rule. It equals beta N2(z)/(2 eps a), so the constant 2 eps a/beta^2 in the solution produces -N2(z)/beta.*)

(* ::Input:: *)
constImage[eps_, st_] := -beta Hf[eps, st][z] + lamf[eps] cbar + Sum[resS[Nf[eps, st][z, t]/Pf[st][t], t, y[[1]], y[[2]], y[[3]]], {y, pZeros[st]}];
check["Step 2 (A6)", Table[Simplify[(constImage @@ c - beta N2f[Sequence @@ c][z]/(2 c[[1]] a)) /. norm] === 0, {c, cases}]]

(* ::Subsection:: *)
(*6.3 Step 1: the Cauchy column (identity A7)*)

(* ::Text:: *)
(*Try y = D delta_(z v) + E/(z + v). Its image under P(z) chi_r, term by term: -beta H(z) y(z); lambda C y = lambda D rho/(z + v) + lambda E (h^2/rho) delta_(z v) (by C^2 = h^2, since 1/(t + v) = C_(t v)/rho); and the rank-two sum D N(z, v)/P(v) + E I(z; -v) with alpha(-v) = 0 on the lattice (Section 3.1), so I(z; -v) = Ibar(z; -v). Split -beta H(z) E/(z + v) into its pole at z = -v and a polynomial. The coefficients of delta_(z v) and of 1/(z + v) must be P(v) and 0:*)

(* ::Input:: *)
step1[eps_, st_] := step1[eps, st] = Module[{H = Hf[eps, st], l = lamf[eps], sol, Dv, Ev, rest},
   sol = First[Solve[{-beta H[v] dv + l hsq ev/rho == Pf[st][v], l rho dv - beta H[-v] ev == 0}, {dv, ev}]];
   {Dv, Ev} = Simplify[{dv, ev} /. sol];
   rest = -beta Ev PolynomialQuotient[H[z], z + v, z] + Dv Nf[eps, st][z, v]/Pf[st][v] + Ev Ibar[eps, st][z, -v];
   <|"D" -> Dv, "E" -> Ev, "rest" -> rest|>];
check["the pole part of -beta H(z) E/(z+v) is -beta H(-v) E/(z+v)", Simplify[PolynomialRemainder[Hf[1, 0][z], z + v, z] - Hf[1, 0][-v]] === 0]
check["D_v = -H(-v) P(v)/(beta e(v)), E_v = -lambda rho F(v)/beta^2, F = P/e", Table[Simplify[({step1[Sequence @@ c]["D"], step1[Sequence @@ c]["E"]} - {-Hf[Sequence @@ c][-v] Pf[c[[2]]][v]/(beta ef[Sequence @@ c][v]), -lamf[c[[1]]] rho Pf[c[[2]]][v]/(ef[Sequence @@ c][v] beta^2)}) /. hsq -> beta^2 p0 p2] === {0, 0}, {c, cases}]]

(* ::Text:: *)
(*The remaining terms, times beta e(v), are the left side of Eq. (source). Define W by Eq. (source), beta e(v) rest = -e(v) N2(z) + lambda rho z [P(v) - e(v)] + lambda W(z; v^2), and check that W is a polynomial of the form W0(s) + z^2 W2(s), both of degree one in s = v^2:*)

(* ::Input:: *)
Wx[eps_, st_] := Wx[eps, st] = Module[{e = ef[eps, st]},
   Expand[Simplify[Together[(beta e[v] step1[eps, st]["rest"] + e[v] N2f[eps, st][z] - lamf[eps] rho z (Pf[st][v] - e[v]))/lamf[eps]] /. norm]]];
AbsoluteTiming[Table[Wx @@ c;, {c, cases}];]
check["W is a polynomial, even in v, W0 + z^2 W2, degree one in v^2", Table[With[{W = Wx @@ c}, {PolynomialQ[W, {z, v}], Expand[W - (W /. v -> -v)] === 0, Exponent[W, z] == 2, Coefficient[W, z, 1] === 0, Exponent[W, v] == 2}], {c, cases}]]
Wf[eps_, st_][zz_, s_] := Wx[eps, st] /. {z -> zz, v -> Sqrt[s]};

(* ::Text:: *)
(*Compare with the note, (A7): with varpi = 4 (p0 p2 - p1^2),*)

(* ::Input:: *)
varpi = 4 (p0 p2 - p1^2);
psiA7[eps_][s_] := -4 eps p1 (p0 + p2 - rho^2) (s + (1 + 2 eps) a^2) + (1 + eps) rho^2 varpi (s - a^2);
W0A7[eps_, st_][s_] := eps a q0 (varpi - 2 eps (1 - rho^2)) (s + q0 + 2 eps a^2) + st beta a^4 psiA7[eps][s];
W2A7[eps_, st_][s_] := eps a varpi (s + q0 + 2 eps a^2) - st beta a^2 psiA7[eps][s];
check["W as in (A7)", Table[Simplify[(Wf[Sequence @@ c][z, s] - (W0A7[Sequence @@ c][s] + z^2 W2A7[Sequence @@ c][s])) /. wr[c[[2]]] /. norm] === 0, {c, cases}]]

(* ::Subsection:: *)
(*6.4 Step 3: pole pairs*)

(* ::Text:: *)
(*phi(t) = fp/(t - u) + fm/(t + u). Its image under P(z) chi_r: -beta H(z) phi(z), lambda C phi = lambda [fp alpha(u)/(z + u) + fm alpha(-u)/(z - u)], and fp I(z; u) + fm I(z; -u) with I = Ibar + alpha N/(rho P). Write al = alpha(u); then alpha(-u) = h^2/al (Section 5). The residues at z = u and z = -u vanish together iff beta^2 H(u) H(-u) - lambda^2 h^2 = beta^2 e(u) = 0. Choose the null vector fp = beta H(-u), fm = lambda al of the second residue. The claim, Eq. (mode-image): at a root u of e (for m, k = 0 at the non-atomic roots; the atomic root u^2 = a^2 is a double zero of P and is not needed, see below), the image is -beta K [z + W(z; u^2)/(rho P(u))] with K = beta H(-u) + lambda al. Check: the numerator of (image - claim) is divisible by e(u) as a polynomial in u.*)

(* ::Input:: *)
step3[eps_, st_] := Module[{H = Hf[eps, st], l = lamf[eps], fp, fm, I, img, claim, num, den},
  fp = beta H[-u]; fm = l al;
  I[x_, alx_] := Ibar[eps, st][z, x] + alx Nf[eps, st][z, x]/(rho Pf[st][x]);
  img = -beta H[z] (fp/(z - u) + fm/(z + u)) + l (fp al/(z + u) + fm (hsq/al)/(z - u)) + fp I[u, al] + fm I[-u, hsq/al];
  claim = -beta (fp + fm) (z + Wf[eps, st][z, u^2]/(rho Pf[st][u]));
  num = Numerator[Together[(img - claim) /. hsq -> beta^2 p0 p2 /. norm]];
  den = Numerator[Together[ef[eps, st][u] /. hsq -> beta^2 p0 p2 /. norm]];
  (* m, k = 0: only the non-atomic roots, e(u)/(u^2 - a^2) = 0 *)
  If[{eps, st} == {-1, 1}, den = PolynomialQuotient[den, u^2 - a^2, u]];
  PolynomialRemainder[num, den, u]];
AbsoluteTiming[r3 = Table[Simplify[step3 @@ c], {c, cases}];]
check["Step 3, Eq. (mode-image)", r3 === {0, 0, 0, 0}]

(* ::Subsection:: *)
(*6.5 Step 4: solving and reading off Gamma*)

(* ::Text:: *)
(*By Steps 1-3, P(z) chi_r maps D_v delta + E_v/(z + v) to P(v) delta_(z v) + (1/beta)[-N2(z) + lambda rho z (F(v) - 1) + lambda W(z; v^2)/e(v)], the constant 2 eps a/beta^2 to N2(z)/beta (Step 2), and phi_j/K_j to -beta [z + W(z; s_j)/(rho P(u_j))]. P - e is even of degree two and W is of degree one in s, so with e(v) = (v^2 - s1)(v^2 - s2) partial fractions give F(v) - 1 = Sum_j F_j g_j(v) and W(z; v^2)/e(v) = Sum_j W(z; s_j)/(s_j - s_l) g_j(v), with F_j = P(u_j)/(s_j - s_l):*)

(* ::Input:: *)
check["P - e is A v^2 + B", Table[With[{d = Expand[Pf[c[[2]]][v] - ef[Sequence @@ c][v]]}, Exponent[d, v] <= 2 && Expand[d - (d /. v -> -v)] === 0], {c, cases}]]
check["m, k = 0: the atomic pair has F_1 = P(a)/(a^2 - s2) = 0 and W(z; a^2) = 0", Simplify[{ef[-1, 1][a], Pf[1][a], Wf[-1, 1][z, a^2]} /. hsq -> beta^2 p0 p2 /. norm] === {0, 0, 0}]
check["partial fractions in s", Simplify[(A s + B)/((s - s1) (s - s2)) - ((A s1 + B)/(s1 - s2)/(s - s1) + (A s2 + B)/(s2 - s1)/(s - s2))] === 0]

(* ::Text:: *)
(*So the column y_v(z) = D_v delta + E_v/(z + v) + (1/beta^2)[2 eps a + lambda rho Sum_j F_j g_j(v) phi_j(z)/K_j] solves P chi y_v = P(v) delta_(z v): the terms of Step 1, Step 2 and Step 3 cancel pairwise. With phi_j(z) = (K_j z + u_j [beta H(-u_j) - lambda alpha_j])/(z^2 - s_j) and the identity F(z) - F(v) = -(z^2 - v^2) Sum_j F_j g_j(z) g_j(v):*)

(* ::Input:: *)
gj[x_, sj_] := 1/(x^2 - sj);
check["phi_j as one fraction", Simplify[beta Hb/(z - uj) + lamb alj/(z + uj) - ((beta Hb + lamb alj) z + uj (beta Hb - lamb alj))/(z^2 - uj^2)] === 0]
check["F(z) - F(v) = -(z^2 - v^2) Sum F_j g_j(z) g_j(v) from F - 1 = Sum F_j g_j", Simplify[(F1 gj[z, s1] + F2 gj[z, s2]) - (F1 gj[v, s1] + F2 gj[v, s2]) + (z^2 - v^2) (F1 gj[z, s1] gj[v, s1] + F2 gj[z, s2] gj[v, s2])] === 0]

(* ::Text:: *)
(*Hence, with DD(z, v) = (z F(z) - v F(v))/(z^2 - v^2) = F(v)/(z + v) - z Sum_j F_j g_j(z) g_j(v) and gamma_j = lambda rho Z_j/(s_j - s_l), Z_j = u_j P(u_j) [beta H(-u_j) - lambda alpha_j]/K_j,*)
(*y_v(z) = D_v delta_(z v) + (1/beta^2)[2 eps a - lambda rho DD(z, v) + Sum_j gamma_j g_j(z) g_j(v)].*)
(*Gamma_r = beta^2 [y - delta/B] with B = -beta G2/P (Section 2.2). The diagonal part:*)

(* ::Input:: *)
check["diagonal part of Gamma_r", Table[Simplify[beta^2 (-Hf[Sequence @@ c][-z] Pf[c[[2]]][z]/(beta ef[Sequence @@ c][z]) - (-Pf[c[[2]]][z]/(beta G2x /. wr[c[[2]]]))) - beta Pf[c[[2]]][z] (1/(G2x /. wr[c[[2]]]) - Hf[Sequence @@ c][-z]/ef[Sequence @@ c][z])] === 0, {c, cases}]]
