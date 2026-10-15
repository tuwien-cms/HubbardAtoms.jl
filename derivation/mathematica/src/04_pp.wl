(* ::Section:: *)
(*4 Singlet and triplet channels*)

(* ::Subsection:: *)
(*4.1 The particle-particle sum (Lemma "particle-particle sum")*)

(* ::Text:: *)
(*With b = 2 mu - wpp and z_n = x_n + wpp/2: t_n = x_n (b - x_n) = b^2/4 - (z_n - mu)^2. For xi^2 = b^2/4 - t, partial fractions and the oddness of S give Sum_n 1/(t_n - t) = [S(mu - xi) - S(mu + xi)]/(2 xi) = -S(mu + xi)/xi =: T_k(t).*)

(* ::Input:: *)
check["t_n = b^2/4 - (z_n - mu)^2", Simplify[(mu + vn) (bb - mu - vn) - (bb^2/4 - (vn + wpp/2)^2)] === 0]
check["partial fractions", Simplify[1/(xi^2 - y^2) - (1/(y + xi) - 1/(y - xi))/(2 xi)] === 0]

(* ::Text:: *)
(*At t+- = -a^2 +- a b: xi = a - b/2 gives mu + xi = a + wpp/2, a zero of Q_-, and xi = -a - b/2 gives mu + xi = -a + wpp/2, a zero of Q_+. So, by Section 3.1, T(t+) = -S_-/(a - b/2) and T(t-) = -S_+/(-a - b/2). T(-As) stays a symbol Ts.*)

(* ::Input:: *)
check["xi at t+-", Simplify[{(a - bb/2)^2 - (bb^2/4 - tp), mu + a - bb/2 - (a + wpp/2), (-a - bb/2)^2 - (bb^2/4 - tm), mu - a - bb/2 - (-a + wpp/2)}] === {0, 0, 0, 0}]
Tval = {{tp, -Sminus/(a - bb/2)}, {tm, -Splus/(-a - bb/2)}, {-As, Ts}};
tAt[y_] := Module[{hit = Select[Tval, Simplify[(y - #[[1]]) /. norm] === 0 &]}, If[Length[hit] == 1, hit[[1, 2]], $Failed]];
tsum[phi_, t_] := Module[{f = Together[phi], poles},
  If[Limit[f, t -> Infinity] =!= 0, Return[$Failed]];
  poles = Union[Simplify[t /. Solve[Denominator[f] == 0, t]]];
  Sum[Residue[f, {t, y}] tAt[y], {y, poles}]];
(* numerical test of T_k: k = 3, U = 2, beta = 3, mu = 0.3 *)
Module[{k = 3, nums = {a -> 1, beta -> 3, mu -> 3/10}, t0 = 1/2 + I/3, tl, xi0},
 tl[n_] := Module[{x = mu + I Pi (2 n + 1)/beta}, x (2 mu - 2 Pi I k/beta - x)] /. nums;
 xi0 = Sqrt[(2 mu - 2 Pi I k/beta)^2/4 - t0] /. nums;
 (* the tail of the sum beyond |n| = N is O(1/N) *)
 check["T_k vs direct sum", Abs[N[Sum[1/(tl[n] - t0), {n, -20000, 19999}]] - N[-Sk[1][mu + xi0]/xi0 /. nums]] < 10^-4]]

(* ::Subsection:: *)
(*4.2 Triplet*)

(* ::Text:: *)
(*On an orbit {n, nbar} with n != nbar, chi_t is the 2x2 block with diagonal (B + L_t)/2 and off-diagonal (B - L_t)/2 (Section 2.3), and bold B is B times the identity. Gamma_t = beta^2 (chi_t^-1 - B^-1) on the block:*)

(* ::Input:: *)
GtBlock = Simplify[beta^2 (Inverse[{{BppT + LtT, BppT - LtT}, {BppT - LtT, BppT + LtT}}/2] - IdentityMatrix[2]/BppT)];
GtNote = beta a^2 (1 - rho^2) PPt[t]/(GGt[t] (t + a rho bb + a^2));
check["Theorem triplet", Simplify[GtBlock - GtNote {{1, -1}, {-1, 1}} /. norm] === {{0, 0}, {0, 0}}]

(* ::Text:: *)
(*At a fixed point n = nbar the orbit has one element, chi_t,nn = (B + L_t)/2 + (B - L_t)/2 = B, and Gamma_t,nn = 0.*)

(* ::Subsection:: *)
(*4.3 Singlet*)

(* ::Text:: *)
(*chi_s = X + g gamma g^T with X = L_s M_+ - B M_-. The rank-two part: partial fractions of chi_s,reg (Section 2.3) in t_n and t_m give gamma_+- (the coefficients of g_+-(t_n) g_+-(t_m), g_+-(t) = 1/(t - t+-)), with no cross terms:*)

(* ::Input:: *)
gp[t_] := 1/(t - tp); gm[t_] := 1/(t - tm);
{gamPlus, gamMinus} = Simplify[{Together[(tn - tp) (tmm - tp) chiSregT] /. {tn -> tp, tmm -> tp}, Together[(tn - tm) (tmm - tm) chiSregT] /. {tn -> tm, tmm -> tm}} /. norm]
check["chi_s,reg = gamma_+ g_+ g_+ + gamma_- g_- g_- (Eq. chi-s-reg)", Simplify[chiSregT - gamPlus gp[tn] gp[tmm] - gamMinus gm[tn] gm[tmm] /. norm] === 0]

(* ::Text:: *)
(*Woodbury: X^-1 = L_s^-1 M_+ - B^-1 M_- and M_+ g = g, so V = X^-1 g has rows g(t_n)/L_s(t_n), and Psi = g^T V = Sum_n g(t_n) g(t_n)^T / L_s(t_n). Its entries are proper rational functions of t_n with simple poles at t+, t-, -As, summed by T_k:*)

(* ::Input:: *)
gvec[t_] := {gp[t], gm[t]};
Psi = Table[tsum[gvec[t][[i]] gvec[t][[j]]/LsT, t], {i, 2}, {j, 2}];
Vn = gvec[tn]/(LsT /. t -> tn); Vm = gvec[tmm]/(LsT /. t -> tmm);
GsDiag = Simplify[beta^2 (1/LsT + 1/BppT)/2];

(* ::Text:: *)
(*The diagonal part of Gamma_s = beta^2 (chi_s^-1 + B^-1) on an orbit is beta^2 (1/L_s + 1/B) M_+, i.e. GsDiag (delta_nm + delta_(m nbar)). The theorem (K_s = 1 + 2a Ts/beta):*)

(* ::Input:: *)
Ks = 1 + 2 a Ts/beta;
check["Theorem singlet, diagonal/mirror part", Simplify[GsDiag - beta a^2 (rho^2 - 1 + 8 p1) PPt[t]/(GGt[t] (t + As)) /. norm] === 0]

(* ::Text:: *)
(*The separable part -beta^2 V (gamma^-1 + Psi)^-1 V^T. Change of basis (note, proof of Theorem "singlet"): with kappa_n = 1/(t_n + As) and d+- = As + t+-, the rows of V are (2/beta) Theta^T (1, kappa_n), and Xi = Theta^-T (gamma^-1 + Psi) Theta^-1 is diagonal (identity A4):*)

(* ::Input:: *)
dp = As + tp; dm = As + tm;
Theta = {{1, 1}, {-dm, -dp}};
kap[t_] := 1/(t + As);
check["V_n = (2/beta) Theta^T (1, kappa_n)", Simplify[(Vn - 2/beta Transpose[Theta] . {1, kap[tn]}) /. norm] === {0, 0}]
Xi = Transpose[Inverse[Theta]] . (Inverse[DiagonalMatrix[{gamPlus, gamMinus}]] + Psi) . Inverse[Theta];
check["Xi = diag(-1/a, K_s/(a d+ d-)) (A4)", Simplify[(Xi - DiagonalMatrix[{-1/a, Ks/(a dp dm)}]) /. Splus -> (Splus /. norm) /. norm] === {{0, 0}, {0, 0}}]
check["d+ d- = P(-As)", Simplify[dp dm - PPt[-As]] === 0]
check["Theorem singlet, separable part", Simplify[-beta^2 (2/beta)^2 {1, kap[tn]} . Inverse[DiagonalMatrix[{-1/a, Ks/(a dp dm)}]] . {1, kap[tmm]} - (4 a - 4 a PPt[-As]/(Ks (tn + As) (tmm + As)))] === 0]
