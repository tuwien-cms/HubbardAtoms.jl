(* ::Section:: *)
(*5 The Cauchy matrix (mu != 0)*)

(* ::Text:: *)
(*C_nm = rho/(z_n + z_m). Lemma "square", C^2 = h^2: for n != m, partial fractions reduce Sum_l C_nl C_lm to rho^2 [S(-z_n) - S(-z_m)]/(z_m - z_n), which vanishes because S is constant on the reflected lattice (Section 3.1). For n = m the sum is rho^2 S'(-z_n) = h^2 (Section 3.1).*)

(* ::Input:: *)
check["partial fractions for C^2", Simplify[1/((zn + zl) (zl + zm)) - (1/(zl + zn) - 1/(zl + zm))/(zm - zn)] === 0]

(* ::Text:: *)
(*The thermal function alpha_k(x) := rho S_k(x) - cbar. Lemma "thermal function": its closed form and alpha_k(x) alpha_k(-x) = h^2.*)

(* ::Input:: *)
alpha[par_][x_] := rho Sk[par][x] - cbar;
check["alpha closed form", Table[FullSimplify[(alpha[par][x] + Sqrt[hsq] If[par == 0, Cosh, Sinh][beta (x + mu)/2]/If[par == 0, Cosh, Sinh][beta (x - mu)/2]) /. thermal, beta > 0 && mu > 0 && a > 0], {par, {0, 1}}] === {0, 0}]
check["alpha(x) alpha(-x) = h^2", Table[FullSimplify[(alpha[par][x] alpha[par][-x] - hsq) /. thermal], {par, {0, 1}}] === {0, 0}]

(* ::Text:: *)
(*Lemma "Cauchy calculus": Sum_n rho/((z + z_n)(z_n - x)) = rho [S(x) - S(-z)]/(z + x) by partial fractions, and rho S(-z) = cbar on the lattice, so C maps 1/(t - x) to alpha_k(x)/(z + x), and the constant 1 to cbar (PV). Numerical test of the first rule (k = 2):*)

(* ::Input:: *)
check["partial fractions for C", Simplify[1/((z + zl) (zl - x)) - (1/(zl - x) - 1/(zl + z))/(z + x)] === 0]
Module[{k = 2, nums, zl, x0 = 0.4 + 0.3 I, zz},
 nums = Join[{a -> 1, beta -> 3, mu -> 3/10}, thermal /. {a -> 1, beta -> 3, mu -> 3/10}];
 zl[n_] := (mu + I Pi (2 n + 1 + k)/beta) /. nums; zz = zl[4];
 check["C acting on 1/(t - x) vs direct sum", Abs[N[Sum[(rho/((zz + zl[n]) (zl[n] - x0))) /. nums, {n, -40000, 39999}]] - N[alpha[0][x0]/(zz + x0) /. nums]] < 10^-4]]
