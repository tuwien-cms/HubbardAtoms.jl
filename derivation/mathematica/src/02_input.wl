(* ::Section:: *)
(*2 The input in the variables of the note*)

(* ::Subsection:: *)
(*2.1 Comparison with Ref. [Essl et al. 2024, Eq. (D3)] as written in the note*)

(* ::Text:: *)
(*Eqs. (chi-uu), (chi-ud) of the note, with x_n = mu + i nu_n, J(x) = 1/(x^2 - a^2), and the generic part N_nm in the compact form of Lemma "compact input": centered variables z = x_n + w/2, v = x_m + w/2, Q_+-(z) = (z +- a)^2 - w^2/4, q0 = a^2 - w^2/4, Cauchy entry C = rho/(z + v) (mu != 0).*)

(* ::Input:: *)
J[x_] := 1/(x^2 - a^2);
Qp[x_] := (x + a)^2 - w^2/4; Qm[x_] := (x - a)^2 - w^2/4; P[x_] := Qp[x] Qm[x];
q0 = a^2 - w^2/4;
toZ = {vn -> z - mu - w/2, vm -> v - mu - w/2};
NnoteCompact = (4 a^2 rho/(z + v) - 2 a)/(Qm[z] Qm[v]) + 4 a^2 (1 - rho) (z + v + 2 a) (z v + q0)/(P[z] P[v]);
xn = mu + vn; xm = mu + vm;
check["Eq. (chi-uu): generic, delta_k0, delta_nn' parts", Simplify[{uu["gen"],
     uu["dk"] - beta gf[vn] gf[vm] - beta a^2 (1 - rho^2) J[xn] J[xm],
     uu["dd"] - (-beta gf[xn - mu] gf[xn - mu + w] - beta a^2 (1 - rho^2) J[xn + w] J[xn])} /. norm] === {0, 0, 0}]
check["Eq. (chi-ud) and Lemma compact input", Simplify[{ud["dk"] - beta gf[vn] gf[vm] - 4 beta a^2 (p0 p2 - p1^2) J[xn] J[xm],
     ud["dd"] + 4 beta a^2 p1 J[xn] J[xn + w],
     (ud["gen"] /. toZ) - NnoteCompact} /. norm] === {0, 0, 0}]

(* ::Subsection:: *)
(*2.2 Density and magnetic channels: Proposition "structure"*)

(* ::Text:: *)
(*chi_r = chi_upup + eps chi_updn with eps = +1 (d), -1 (m), as functions of the centered points z = z_n, v = z_m. All parts share the denominator P(z) = Q_+(z) Q_-(z), even in z, with no zeros on the lattice:*)

(* ::Input:: *)
chiR[eps_] := chiR[eps] = Map[Simplify[# /. toZ /. norm] &, <|"gen" -> uu["gen"] + eps ud["gen"],
     "dk" -> uu["dk"] + eps ud["dk"] - (1 + eps) beta gf[vn] gf[vm], "dd" -> (uu["dd"] + eps ud["dd"] /. vm -> vn)|>];
check["(x_n - a)(x_n + w - a) = Q_-(z_n), (x_n + a)(x_n + w + a) = Q_+(z_n), P even", Simplify[{(xn - a) (xn + w - a) - Qm[z], (xn + a) (xn + w + a) - Qp[z]} /. vn -> z - mu - w/2] === {0, 0} && Expand[P[z] - P[-z]] === 0]

(* ::Text:: *)
(*Extract, for each channel: the diagonal function H from P(z) chi_dd = -beta H(z); the Cauchy coefficient lambda from the residue of P(z) chi_gen at z = -v (it multiplies rho/(z + v)); and the remaining numerator N(z, v) = P(z) P(v) [chi_gen - lambda rho/((z + v) P(z))], plus the static (k = 0, w = 0) part P(z) P(v) chi_dk.*)

(* ::Input:: *)
Hx[eps_] := Hx[eps] = Factor[-P[z] chiR[eps]["dd"]/beta];
lam[eps_] := lam[eps] = Simplify[(Together[(z + v) P[z] chiR[eps]["gen"]] /. z -> -v)/rho];
Ngen[eps_] := Ngen[eps] = Expand[Simplify[P[z] P[v] (chiR[eps]["gen"] - lam[eps] rho/((z + v) P[z]))]];
Nstat[eps_] := Nstat[eps] = Expand[Simplify[P[z] P[v] chiR[eps]["dk"] /. w -> 0]];
Table[{Hx[eps], lam[eps]}, {eps, {1, -1}}]
check["H, N polynomial; lambda constant", Table[{PolynomialQ[Hx[eps], z], FreeQ[lam[eps], z | v], PolynomialQ[Ngen[eps], {z, v}], PolynomialQ[Nstat[eps], {z, v}]}, {eps, {1, -1}}]]

(* ::Text:: *)
(*H(z) = z^2 + 2 a rho z + q1 with q1 = q0 + 4 eps a^2 p1, and lambda = 4 eps a^2 (note, Section "Density and magnetic"):*)

(* ::Input:: *)
check["H and lambda as in the note", Table[Simplify[{Hx[eps] - (z^2 + 2 a rho z + q0 + 4 eps a^2 p1), lam[eps] - 4 eps a^2} /. norm] === {0, 0}, {eps, {1, -1}}]]

(* ::Text:: *)
(*The numerator N(z, v) (with st = delta_(k0), and w = 0 if st = 1) is quadratic in z without a linear term, N = N0(v) + z^2 N2(v): after multiplication by P, the finite-rank part has only the two output directions 1 and z^2.*)

(* ::Input:: *)
NN[eps_, 0] := Ngen[eps]; NN[eps_, 1] := Expand[(Ngen[eps] + Nstat[eps]) /. w -> 0];
N0x[eps_, st_] := Coefficient[NN[eps, st], z, 0]; N2x[eps_, st_] := Coefficient[NN[eps, st], z, 2];
check["N = N0(v) + z^2 N2(v)", Flatten[Table[{Exponent[NN[eps, st], z] == 2, Coefficient[NN[eps, st], z, 1] === 0}, {eps, {1, -1}}, {st, {0, 1}}]]]
zeta[eps_] := beta a^2 (1 - rho^2 + 4 eps (p0 p2 - p1^2));
N2note[eps_, st_, x_] := -2 eps a (x^2 + 2 a rho x + q0) + st zeta[eps] (x^2 - a^2);
N0note[eps_, st_, x_] := (x^2 + 2 q0 - 4 a^2) N2note[eps, st, x] + (2 eps a - st zeta[eps]) P[x];
check["N0, N2 as in Eq. (N) of the note", Flatten[Table[Simplify[({N0x[eps, st], N2x[eps, st]} - {N0note[eps, st, v], N2note[eps, st, v]}) /. If[st == 1, w -> 0, {}] /. norm] === {0, 0}, {eps, {1, -1}}, {st, {0, 1}}]]]

(* ::Text:: *)
(*The bare bubble of the ph channels, B_n = -beta G(nu_n) G(nu_n + omega) = -beta G2(z_n)/P(z_n) with G2(z) = (z + a rho)^2 - w^2/4:*)

(* ::Input:: *)
G2x = Factor[Simplify[-P[z] (-beta gf[vn] gf[vn + w] /. toZ /. norm)/beta]]
check["G2", Simplify[G2x - ((z + a rho)^2 - w^2/4)] === 0]

(* ::Subsection:: *)
(*2.3 Singlet and triplet channels: Proposition (chi-pp)*)

(* ::Text:: *)
(*chi_pp,ss'(nu, nu', omega) = chi_ss'(nu, nu', -omega - nu - nu'): the ph transfer becomes w -> -wpp - vn - vm, the ph delta_(k0) becomes the pp mirror delta_(m, -n-k-1), and delta_(nn') stays. With the pp bubble B = -beta/2 G(nu) G(-nu - omega), chi_s = (-chi_pp,upup + 2 chi_pp,updn - 2 B)/4 and chi_t = (chi_pp,upup + 2 B)/4. Each has a generic part, a diagonal part ("dd") and a mirror part ("mir", evaluated on the mirror vm = -vn - wpp).*)

(* ::Input:: *)
toPP[f_] := f /. w -> -wpp - vn - vm;
mirror = {vm -> -vn - wpp};
Bpp = -beta/2 gf[vn] gf[-vn - wpp];
chiPP[r_] := chiPP[r] = Module[{c = If[r === "t", {1, 0}, {-1, 2}], s = If[r === "t", 1, -1]},
   Map[Simplify[# /. norm] &, <|"gen" -> toPP[c . {uu["gen"], ud["gen"]}]/4,
     "mir" -> (toPP[c . {uu["dk"] - beta gf[vn] gf[vm], ud["dk"] - beta gf[vn] gf[vm]}] /. mirror)/4,
     "dd" -> (toPP[c . {uu["dd"], ud["dd"]}] /. vm -> vn)/4 + s Bpp/2|>]];
check["chi_t has no generic part", chiPP["t"]["gen"] === 0]

(* ::Text:: *)
(*Variables: b = 2 mu - wpp, x = mu + vn, t = x (b - x). The mirror x -> b - x leaves t invariant. Any function of vn that is invariant under the mirror is a function of t; toT finds it by inserting the root x = (b - sqrt(b^2 - 4t))/2.*)

(* ::Input:: *)
bb = 2 mu - wpp;
toT[f_, from_, to_] := Factor[FullSimplify[f /. from -> (bb - Sqrt[bb^2 - 4 to])/2 - mu]];

(* ::Text:: *)
(*On each orbit {n, nbar} a matrix d delta_nm + m delta_(m nbar) with mirror-invariant d, m acts as d + m on mirror-even and d - m on mirror-odd vectors. So chi_t = B M_+ + L_t M_- and chi_s = L_s M_+ - B M_- + chi_s,reg, with M_+- = (1 +- mirror)/2, where the eigenvalues are:*)

(* ::Input:: *)
{BppT, LtT} = toT[#, vn, t] & /@ {chiPP["t"]["dd"] + chiPP["t"]["mir"], chiPP["t"]["dd"] - chiPP["t"]["mir"]}
{mBppT, LsT} = toT[#, vn, t] & /@ {chiPP["s"]["dd"] - chiPP["s"]["mir"], chiPP["s"]["dd"] + chiPP["s"]["mir"]}
check["the two channels have the same bubble B, and B is the pp bubble", Simplify[{BppT + mBppT, BppT - toT[Bpp /. norm, vn, t]}] === {0, 0}]

(* ::Text:: *)
(*Compare with the note: P(t) = (t + a^2)^2 - a^2 b^2 = (t - t+)(t - t-), t+- = -a^2 +- a b, G(t) = t + a rho b + a^2 rho^2, As = a rho b + a^2 (1 - 8 p1), B = -beta/2 G(t)/P(t), L_t = -beta/2 (t + a rho b + a^2)/P(t), L_s = beta/2 (t + As)/P(t).*)

(* ::Input:: *)
tp = -a^2 + a bb; tm = -a^2 - a bb;
PPt[t_] := (t + a^2)^2 - a^2 bb^2; GGt[t_] := t + a rho bb + a^2 rho^2; As = a rho bb + a^2 (1 - 8 p1);
check["B, L_t, L_s as in the note; P(t) = (t - t+)(t - t-)", Simplify[{BppT + beta/2 GGt[t]/PPt[t], LtT + beta/2 (t + a rho bb + a^2)/PPt[t], LsT - beta/2 (t + As)/PPt[t], PPt[t] - (t - tp) (t - tm)} /. norm] === {0, 0, 0, 0}]

(* ::Text:: *)
(*The generic part of chi_s, as a function of t_n, t_m: it is chi_s,reg of the note, Eq. (chi-s-num), with c = rho/b:*)

(* ::Input:: *)
chiSregT = toT[toT[chiPP["s"]["gen"], vn, tn], vm, tmm]
check["Eq. (chi-s-num)", Simplify[chiSregT - (a (2 a rho/bb - 1) (tn - tm) (tmm - tm) + a^2 (1 - rho) (bb + 2 a) (tn + tmm + 2 a^2))/(PPt[tn] PPt[tmm]) /. norm] === 0]
