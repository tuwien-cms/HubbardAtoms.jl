(* ::Section:: *)
(*1 The atom and its correlation functions*)

(* ::Text:: *)
(*Fock states |0>, |up>, |dn>, |updn> = c_up^+ c_dn^+ |0>. Hamiltonian H = U (n_up - 1/2)(n_dn - 1/2) - mu (n_up + n_dn - 1) with a = U/2. Its eigenvalues E_i and Boltzmann weights p0, p1, p1, p2 (p1 for each singly occupied state); the weights stay symbols until they are needed.*)

(* ::Input:: *)
cupD = {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}};
cdnD = {{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, -1, 0, 0}};
cup = Transpose[cupD]; cdn = Transpose[cdnD];
nup = cupD . cup; ndn = cdnD . cdn; id4 = IdentityMatrix[4];
check["canonical anticommutators", {cup . cdn + cdn . cup == 0 id4, cup . cupD + cupD . cup == id4, cdn . cdnD + cdnD . cdn == id4, cup . cdnD + cdnD . cup == 0 id4}]
ham = 2 a (nup - id4/2) . (ndn - id4/2) - mu (nup + ndn - id4);
check["H is diagonal", DiagonalMatrix[Diagonal[ham]] == ham]
energy = Diagonal[ham]
weight = {p0, p1, p1, p2};

(* ::Text:: *)
(*Imaginary-time correlators. With O(tau) = e^(tau H) O e^(-tau H), the integrand of a time-ordered correlator on the ordering beta > t_1 > ... > t_n > 0 is a sum of exponentials e^(x_1 t_1 + ... + x_n t_n). Its integral over that simplex is the divided difference of e^(beta s) at the nodes 0, x_1, x_1 + x_2, ...:*)

(* ::Input:: *)
divdiff[nodes_List, vals_List] := Sum[vals[[i]]/Product[If[j == i, 1, nodes[[i]] - nodes[[j]]], {j, Length[nodes]}], {i, Length[nodes]}];
check["simplex integral = divided difference", Simplify[Integrate[Exp[x1 t1 + x2 t2 + x3 t3], {t1, 0, beta}, {t2, 0, t1}, {t3, 0, t2}] - divdiff[{0, x1, x1 + x2, x1 + x2 + x3}, Exp[beta {0, x1, x1 + x2, x1 + x2 + x3}]]] == 0]

(* ::Text:: *)
(*correlator[ops, phis, point] is Int dtau_1 ... dtau_n e^(phi_1 . f tau_1 + ... + phi_n . f tau_n) Tr[e^(-beta H) T O_1(tau_1) ... O_n(tau_n) O_(n+1)(0)]/Z for fermionic operators O_i, with f = {vn, vm, w} = {i nu, i nu', i omega} and integer coefficient vectors phi_j, at Matsubara frequencies obeying the relations point (e.g. {w -> 0}). The sum runs over time orderings (sign of the permutation) and closed paths of states s_1 -> ... -> s_(n+1) -> s_1. Node j is x_j = E_(s_1) - E_(s_j) + (frequencies so far), and p_(s_1) e^(beta x_j) = p_(s_j) e^(beta c_j . f) = p_(s_j) (-1)^(c_j1 + c_j2) at Matsubara frequencies. Where nodes coincide, the divided difference is confluent: equal nodes are grouped, and m equal nodes give f^(m-1)(x)/(m-1)! = beta^(m-1) e^(beta x)/(m-1)!. No limits are needed.*)

(* ::Input:: *)
freqs = {vn, vm, w};
ddExp[nv_List] := Which[Length[nv] == 1, nv[[1, 2]],
   Expand[nv[[1, 1]] - nv[[-1, 1]]] === 0, beta^(Length[nv] - 1) nv[[1, 2]]/(Length[nv] - 1)!,
   True, (ddExp[Rest[nv]] - ddExp[Most[nv]])/(nv[[-1, 1]] - nv[[1, 1]])];
check["divided difference of Exp, distinct nodes", Simplify[ddExp[{{x1, Exp[beta x1]}, {x2, Exp[beta x2]}, {x3, Exp[beta x3]}}] - divdiff[{x1, x2, x3}, Exp[beta {x1, x2, x3}]]] === 0]
check["confluent divided difference = limit", Simplify[ddExp[{{x1, Exp[beta x1]}, {x1, Exp[beta x1]}, {x3, Exp[beta x3]}}] - Limit[divdiff[{x1, x2, x3}, Exp[beta {x1, x2, x3}]], x2 -> x1]] === 0]
correlator[ops_, phis_, point_ : {}] := Module[{n = Length[phis], total = 0},
  Do[
   Module[{string = Append[ops[[perm]], Last[ops]], ph = phis[[perm]], sgn = Signature[perm]},
    Do[
     Module[{amp = Product[string[[j, path[[j]], path[[j + 1]]]], {j, n + 1}], cum, nv},
      If[amp =!= 0,
       cum = FoldList[Plus, {0, 0, 0}, ph];
       nv = Table[{energy[[path[[1]]]] - energy[[path[[j]]]] + cum[[j]] . (freqs /. point),
          weight[[path[[j]]]] (-1)^(cum[[j, 1]] + cum[[j, 2]])}, {j, n + 1}];
       total += sgn amp ddExp[Flatten[GatherBy[nv, Expand[#[[1]]] &], 1]]]],
     {path, Append[#, First[#]] & /@ Tuples[Range[4], n + 1]}]],
   {perm, Permutations[Range[n]]}];
  Together[total]];

(* ::Text:: *)
(*The propagator G(nu) = -Int dtau e^(i nu tau) <T c_up(tau) c_up^+(0)>, as a function of vn = i nu:*)

(* ::Input:: *)
gf[x_] = Simplify[-correlator[{cup, cupD}, {{1, 0, 0}}] /. vn -> x]
check["G = (p0 + p1)/(i nu + mu + a) + (p1 + p2)/(i nu + mu - a)", Simplify[gf[vn] - ((p0 + p1)/(vn + mu + a) + (p1 + p2)/(vn + mu - a))] === 0]

(* ::Text:: *)
(*The generalized susceptibility (Thunstroem et al. 2018, Eq. 3):*)
(*chi_ss'(nu, nu', omega) = Int dtau_1 dtau_2 dtau_3 e^(-i nu tau_1 + i (nu + omega) tau_2 - i (nu' + omega) tau_3) [<T c_s^+(tau_1) c_s(tau_2) c_s'^+(tau_3) c_s'(0)> - <T c_s^+(tau_1) c_s(tau_2)> <T c_s'^+(tau_3) c_s'(0)>].*)
(*In the disconnected part the first factor depends on tau_1 - tau_2 only, so the tau_2 integral gives beta delta_(omega 0) g(nu) g(nu'), with g(nu) = Int dtau e^(-i nu tau) <T c^+(tau) c(0)>; g equals G:*)

(* ::Input:: *)
phisChi = {{-1, 0, 0}, {1, 0, 1}, {0, -1, -1}};
fourPoint[same_, point_] := correlator[If[same, {cupD, cup, cupD, cup}, {cupD, cup, cdnD, cdn}], phisChi, point];
check["g = G", Simplify[correlator[{cupD, cup}, {{-1, 0, 0}}] - gf[vn]] === 0]

(* ::Text:: *)
(*The four-point function: its generic value, and the corrections at the special points w = 0 ("dk"), vm = vn ("dd") and the mirror vm = -vn - w ("dm"), and at their pairwise intersections (inclusion-exclusion). For mu != 0 no other nodes can coincide. Each correction multiplies a Kronecker delta: delta_(k 0), delta_(n n'), delta_(n', -n-k-1).*)

(* ::Input:: *)
parts[same_] := Module[{f = fourPoint[same, #] &, gen, Dk, Dd, Dm, kd = {w -> 0}, dd = {vm -> vn}, dm = {vm -> -vn - w}},
  gen = f[{}];
  Dk = Together[f[kd] - (gen /. kd)];
  Dd = Together[f[dd] - (gen /. dd)];
  Dm = Together[f[dm] - (gen /. dm)];
  <|"gen" -> gen, "dk" -> Dk, "dd" -> Dd, "dm" -> Dm,
   "dkdd" -> Together[f[{w -> 0, vm -> vn}] - (gen /. {w -> 0, vm -> vn}) - (Dk /. dd) - (Dd /. kd)],
   "dkdm" -> Together[f[{w -> 0, vm -> -vn}] - (gen /. {w -> 0, vm -> -vn}) - (Dk /. vm -> -vn) - (Dm /. kd)],
   "dddm" -> Together[f[{vm -> vn, w -> -2 vn}] - (gen /. {vm -> vn, w -> -2 vn}) - (Dd /. w -> -2 vn) - (Dm /. {vm -> vn, w -> -2 vn})]|>];
uu = parts[True];
ud = parts[False];
check["no mirror correction and no corrections at intersections (mu generic)", Union[Flatten[Values /@ {KeyTake[uu, {"dm", "dkdd", "dkdm", "dddm"}], KeyTake[ud, {"dm", "dkdd", "dkdm", "dddm"}]}]] === {0}]

(* ::Text:: *)
(*So chi_ss' = gen + delta_(k0) (dk - beta G(nu) G(nu')) + delta_(nn') dd, with no further terms. The weights are normalized from now on, and rho = p2 - p0 = <n> - 1.*)

(* ::Input:: *)
norm = {p1 -> (1 - p0 - p2)/2};
rho = p2 - p0;
