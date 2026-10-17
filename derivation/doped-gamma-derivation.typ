// The channel-irreducible vertex of the Hubbard atom away from half filling: a derivation.
// Compile: typst compile derivation/doped-gamma-derivation.typ
// Figures: doped-gamma/figures/ (figure_data.jl, then make_figures.py).

#set document(title: [Irreducible vertex of the Hubbard atom away from half filling])
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)", supplement: [Eq.])
#show heading.where(level: 1): set block(above: 1.6em, below: 0.9em)
#show figure.caption: set text(size: 9.5pt)

// ---------- theorem-like environments (one shared counter, referenceable) ----------
#let thm-kind = "thm"
#show figure.where(kind: thm-kind): set block(breakable: true)
#show figure.where(kind: thm-kind): it => block(
  width: 100%,
  inset: (left: 9pt, y: 4pt),
  stroke: (left: 1.2pt + luma(140)),
  align(left)[
    *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body)]. #it.body
  ],
)
// `lean`: the status of the statement in the Lean formalization (App. E), printed below it
#let env(supplement) = (body, title: none, lean: none) => figure(
  if lean == none { body } else {
    body + block(above: 0.7em, text(size: 9pt, fill: luma(70))[_Lean:_ #lean])
  },
  kind: thm-kind,
  supplement: supplement,
  numbering: "1",
  outlined: false,
  caption: title,
)
#let lemma = env[Lemma]
#let proposition = env[Proposition]
#let theorem = env[Theorem]
#let corollary = env[Corollary]
#let proof(body, of: none) = block(width: 100%, inset: (left: 9pt))[_Proof#if of != none [ of #of]._ #body #h(1fr) $square$]
#let example(body) = block(breakable: false, box(width: 100%, inset: 7pt, fill: luma(245), radius: 2pt)[*Example.* #body])

// ---------- notation ----------
#let PV = math.op("PV")
#let Res = math.op("Res", limits: true)
#let diag = math.op("diag")
#let up = sym.arrow.t
#let dn = sym.arrow.b
#let cC = math.cal("C")
#let Rreg = $R_"reg"$
#let As = $A_"s"$
#let eqrange(a, b) = [Eqs. #ref(a, supplement: none)–#ref(b, supplement: none)]

#align(center)[
  #text(15pt, weight: "bold")[The channel-irreducible vertex of the Hubbard atom \ away from half filling: derivation]
  #v(0.4em)
  #text(11pt)[Samuel Badr]
  #v(0.2em)
  #text(10pt, style: "italic")[TU Wien, October 2026]
  #v(0.2em)
  #text(8.5pt)[Licensed under #link("https://creativecommons.org/licenses/by/4.0/")[CC BY 4.0].]
]
#v(0.6em)

= Overview <sec-overview>

The irreducible vertex $Gamma_r$ of the Hubbard atom is known in closed form at half filling @thunstroem2018. Here we derive it at arbitrary filling (zero magnetic field) in all four channels $r in {"d", "m", "s", "t"}$. By definition $Gamma_r = beta^2 (chi_r^(-1) - eta_r chi_(0, r)^(-1))$, where $chi_r$ is the generalized susceptibility, an infinite matrix in two fermionic Matsubara frequencies. So the task is to invert $chi_r$ in closed form.

At half filling $chi_r$ is a diagonal matrix plus a mirror term plus a matrix of rank at most two, and the inversion is elementary. Away from half filling the mirror term becomes a dense _Cauchy matrix_ $C_(n m) = rho slash (z_n + z_m)$ (@fig-structure). The particle–hole susceptibility is then no longer a finite-rank perturbation of an easily invertible matrix. Two properties of $C$ still allow a closed-form inverse:
+ $C^2 = h^2 bb(1)$ for a number $h$: like the mirror it replaces, $C$ is a scaled involution.
+ $C$ maps a simple pole to a simple pole: $sum_m C_(n m) slash (z_m - x) = alpha(x) slash (z_n + x)$, with a thermal factor obeying $alpha(x) alpha(-x) = h^2$.

After multiplication by a polynomial, $chi_r$ is a diagonal matrix plus $lambda C$ plus a matrix of rank two. Each column of $chi_r^(-1)$ is then built from a diagonal entry, one Cauchy column, a constant, and two pole pairs with one thermal denominator each. All coefficients follow from $2 times 2$ linear systems whose determinants are given by a single quartic.

#figure(
  image("doped-gamma/figures/chi_structure.svg", width: 58%),
  caption: [Structure of $chi_"d"$ at $U = 2$, $beta = 3$, $omega = 0$, as moduli of matrix elements in the fermionic indices $n, m$. (a) At half filling the pair term is a mirror, nonzero only for $m = -n - 1$. (b) Away from half filling it becomes the dense Cauchy matrix $C$. (c) The remaining finite-rank part. (d) The full $chi_"d"$; the diagonal dominates.],
) <fig-structure>

*Results.*
- *Triplet:* $Gamma_"t"$ is a diagonal-minus-mirror term (@thm-t).
- *Singlet:* $Gamma_"s"$ is a diagonal-plus-mirror term, the constant $2U$, and one separable term with a single thermal denominator (@thm-s).
- *Density and magnetic:* $Gamma_"d", Gamma_"m"$ have the same anatomy: a diagonal term, the constant $plus.minus U$, a dense term without thermal factors, and two separable terms with one thermal denominator each (@thm-ph). At half filling the dense term becomes a mirror term and one separable term remains (@cor-hf).

The inverses are two-sided and agree with the limit of inverses on finite frequency boxes, i.e. with what numerical inversion converges to (@prop-box).

*Roadmap.*
- @sec-setup defines all quantities; @sec-input states the input, the susceptibility from the Lehmann representation.
- @sec-results states all results. The rest of the note proves them.
- @sec-sums evaluates the Matsubara sums; @sec-pp proves the particle–particle results, which need no Cauchy matrix.
- @sec-cauchy develops the algebra of $C$, which @sec-ph uses for the particle–hole channels. The limit $mu -> 0$ and the finite-box statement are proved in @app-details.

*What is assumed.* Everything is proved from the input #eqrange(<eq-chi-uu>, <eq-pair>) (Ref. @essl2024) and the rational identities of @app-identities (checked by computer algebra). The Mathematica notebook `mathematica/doped_gamma.nb` derives the input from the Hamiltonian and replays the proofs step by step (@app-checks). Independent numerical checks are listed in @app-checks. Most of the proofs are also formalized in Lean 4, from the same input (@app-lean); a note after each numbered statement says how much of it. Numerical stability in floating point is not discussed here.

*Implementation.* The results are implemented as `DopedHubbardAtom` in the Julia package #link("https://github.com/tuwien-cms/HubbardAtoms.jl")[HubbardAtoms.jl], which also evaluates them stably at low temperatures and near the removable singularities, and is tested against exact diagonalization and finite-box inversion.

= Setup <sec-setup>

== Model and thermal weights

$ cal(H) = U (n_up - 1/2)(n_dn - 1/2) - mu (n_up + n_dn - 1), quad a := U/2, $ <eq-H>
with real $U$, $mu$ (the chemical potential measured from half filling, $mu = 0$) and inverse temperature $beta > 0$. The four states $|0 chevron.r, |up chevron.r, |dn chevron.r, |up dn chevron.r$ have energies $U/4 + mu$, $-U/4$, $-U/4$, $U/4 - mu$ and Boltzmann weights $p_0, p_1, p_1, p_2$ ($p_1$ is the weight of _each_ singly occupied state). Hence
$ p_0 + 2 p_1 + p_2 = 1, quad quad p_2 / p_0 = e^(2 beta mu), quad quad p_1 / p_0 = e^(beta (a + mu)), quad quad p_1 / p_2 = e^(beta (a - mu)). $ <eq-weights>
Define
$ rho := p_2 - p_0 = chevron.l n_up + n_dn chevron.r - 1, quad quad h := beta sqrt(p_0 p_2), quad quad overline(c) := beta (p_0 + p_2) slash 2. $
From @eq-weights, $p_1 + p_2 = (1 + rho) slash 2$, $p_0 + p_1 = (1 - rho) slash 2$, and
$ rho^2 + 4 p_0 p_2 = (p_0 + p_2)^2, quad quad overline(c)^2 - h^2 = beta^2 rho^2 slash 4. $ <eq-weight-ids>
For $mu != 0$, inserting $p_2 = p_0 e^(2 beta mu)$:
$ rho = 2 p_0 e^(beta mu) sinh(beta mu), quad p_0 + p_2 = 2 p_0 e^(beta mu) cosh(beta mu) \
==> quad rho coth(beta mu) = p_0 + p_2, quad rho^2 = 4 p_0 p_2 sinh^2(beta mu). $ <eq-weight-hyp>

== Frequencies and propagator

Fermionic and bosonic Matsubara frequencies are $nu_n = (2n + 1) pi slash beta$ and $Omega_ell = 2 ell pi slash beta$ ($n, ell in ZZ$). The bosonic transfer $omega_k := Omega_k$ is fixed throughout. Write
$ x_n := mu + i nu_n, quad quad w := i omega_k, quad quad J(x) := 1 / (x^2 - a^2). $
The propagator $G(nu) := -integral_0^beta dif tau thin e^(i nu tau) chevron.l T c_up (tau) c_up^dagger chevron.r$ follows from the Lehmann representation: $c_up^dagger$ maps $|0 chevron.r -> |up chevron.r$ (energy change $-a - mu$, weight $p_0 + p_1$) and $|dn chevron.r -> |up dn chevron.r$ (energy change $a - mu$, weight $p_1 + p_2$). So
$ G(nu_n) = (p_0 + p_1) / (x_n + a) + (p_1 + p_2) / (x_n - a) = G(x_n) := (x_n + a rho) / (x_n^2 - a^2), $ <eq-G>
where we use the same letter for the rational function $G(x)$.

== Susceptibilities and the irreducible vertex

Following Ref. @thunstroem2018, the generalized susceptibility is
$ chi_(sigma sigma')(nu, nu', omega) := & integral.triple_0^beta dif tau_1 dif tau_2 dif tau_3 thin e^(-i nu tau_1 + i (nu + omega) tau_2 - i (nu' + omega) tau_3) \
& times [chevron.l T c_sigma^dagger (tau_1) c_sigma (tau_2) c_(sigma')^dagger (tau_3) c_(sigma')(0) chevron.r - chevron.l T c_sigma^dagger (tau_1) c_sigma (tau_2) chevron.r chevron.l T c_(sigma')^dagger (tau_3) c_(sigma')(0) chevron.r]. $
Its particle–particle form is $chi_("pp", sigma sigma')(nu, nu', omega) := chi_(sigma sigma')(nu, nu', -omega - nu - nu')$. The channels and bare susceptibilities are
$ chi_"d" = chi_(up up) + chi_(up dn), quad chi_"m" = chi_(up up) - chi_(up dn), quad chi_"s" = (-chi_("pp", up up) + 2 chi_("pp", up dn) - 2 chi_(0, "pp")) / 4, quad chi_"t" = (chi_("pp", up up) + 2 chi_(0, "pp")) / 4, $
$ chi_(0, r)(nu, nu', omega) = delta_(nu nu') B(nu), quad quad B(nu) = cases(-beta G(nu) G(nu + omega) & quad (r = "d", "m")",", -beta/2 thin G(nu) G(-nu - omega) & quad (r = "s", "t")",") $ <eq-B>
with $chi_(0, "pp") := chi_(0, "s")$. At fixed $omega_k$ all of these are matrices indexed by $(n, m)$ with $nu = nu_n$, $nu' = nu_m$; we write $B_n := B(nu_n)$ and $bold(B) := diag(B_n)$. The channel-irreducible vertex is
$ Gamma_r = beta^2 (chi_r^(-1) - eta_r bold(B)^(-1)), quad quad eta_"d" = eta_"m" = eta_"t" = 1, quad eta_"s" = -1. $ <eq-Gamma-def>

== Lattice and mirror <sec-lattice>

Define the centered lattice and the mirror index
$ z_n := x_n + w/2 = mu + i pi (2n + 1 + k) / beta, quad quad overline(n) := -n - k - 1, \
z_(overline(n)) = 2 mu - z_n, quad quad x_(overline(n)) = 2 mu - w - x_n. $ <eq-z>
The lattice lies on the line $"Re" z = mu$, and the mirror reflects it about the real axis (@fig-plane). The mirror matrix is $cal(M)_(n m) := delta_(m overline(n))$, with $cal(M)^2 = bb(1)$. Products of infinite matrices are taken entrywise. A sum $sum_n$ is the limit of partial sums over the mirror-symmetric boxes ${n : |2n + 1 + k| <= 2M + 1}$, $M -> oo$; we write $PV$ ("principal value") where it converges only in this sense. All transposes are ordinary, not Hermitian. Symbols introduced inside a proof are local to it. A function $phi$ on $CC$ is _proper rational_ if it is rational and $phi(t) -> 0$ for $t -> oo$.

#figure(
  placement: auto,
  image("doped-gamma/figures/complex_plane.svg", width: 70%),
  caption: [The complex plane for $U = 2.4$, $beta = 2$, $mu = 0.4$, $omega = omega_1$. The lattice $z_n$ (filled) lies on $"Re" z = mu$; the reflected lattice $-z_n$ (open), where the Cauchy kernel $1 slash (z + z_m)$ is singular, lies on $"Re" z = -mu$. Crosses: zeros of $Q_(plus.minus)$ (@sec-input); diamonds: zeros of $e$ (@sec-res-ph). None lies on the lattice, so all lattice sums below reduce to finitely many residues.],
) <fig-plane>

= Input: the generalized susceptibility <sec-input>

The Lehmann representation of $chi_(sigma sigma')$ for the atom is evaluated in Ref. @essl2024 [Eq. (D3)]. At zero field and in our notation it reads
$ chi_(up up, n m) &= -beta G(x_n) G(x_n + w) delta_(n m) + beta a^2 (1 - rho^2) J(x_n + w) J(x_m) (delta_(k 0) - delta_(n m)), $ <eq-chi-uu>
$ chi_(up dn, n m) &= -4 beta a^2 p_1 J(x_n) J(x_n + w) delta_(n m) + 4 beta a^2 (p_0 p_2 - p_1^2) J(x_n) J(x_m) delta_(k 0) + cal(N)_(n m), $ <eq-chi-ud>
$ cal(N)_(n m) := c_(n + m + k + 1) Pi(x_n, x_m) + Rreg (x_n, x_m), quad quad c_ell := rho / (2 mu + i Omega_ell), \
Pi(x, y) := (1 / (x + w - a) + 1 / (y - a)) (1 / (y + w - a) + 1 / (x - a)). $ <eq-pair>
The pair coefficient $c_ell$ stems from the transition $|0 chevron.r <-> |up dn chevron.r$. At $mu = 0$ it is understood as its limit, $c_ell = beta p_0 delta_(ell 0)$ (by @eq-weight-hyp, $rho slash 2 mu -> beta p_0$). The regular part $Rreg$ is a sum of twelve simple fractions, reproduced in @app-R. We take #eqrange(<eq-chi-uu>, <eq-pair>) as given; they follow exactly from the Lehmann representation (@app-checks).

*Compact form.* Define the *Cauchy matrix* $C_(n m) := c_(n + m + k + 1)$. Since $2 mu + i Omega_(n + m + k + 1) = z_n + z_m$ by @eq-z,
$ C_(n m) = rho / (z_n + z_m) quad (mu != 0), quad quad C_(n m) = beta p_0 delta_(m overline(n)) quad (mu = 0), $ <eq-C>
and $z_n + z_m = 2 mu + i (dots.c) != 0$ for $mu != 0$. Further let
$ Q_(plus.minus)(z) := (z plus.minus a)^2 - w^2/4, quad quad q_0 := a^2 - w^2/4 = (U^2 + omega_k^2) slash 4. $
Since $x_n = z_n - w slash 2$,
$ (x_n - a)(x_n + w - a) = Q_(-)(z_n), quad quad (x_n + a)(x_n + w + a) = Q_(+)(z_n), quad quad Q_(plus.minus)(-z) = Q_(minus.plus)(z). $ <eq-Qfact>
Neither $Q_(plus.minus)(z_n)$ vanishes: its factors have imaginary parts $nu_n$ and $nu_n + omega_k$, both nonzero.

#lemma(title: [compact input], lean: [for $mu != 0$, `compact_input` (`PHInput.lean`).])[
  With $z = z_n$ and $v = z_m$,
  $ cal(N)_(n m) = frac(4 a^2 C_(n m) - 2 a, Q_(-)(z) Q_(-)(v)) + frac(4 a^2 (1 - rho)(z + v + 2 a)(z v + q_0), Q_(+)(z) Q_(-)(z) Q_(+)(v) Q_(-)(v)). $ <eq-input>
] <lem-input>
#proof[
  Put $y := z + v = x_n + x_m + w$. The two factors of $Pi$ combine to
  $ Pi(x_n, x_m) = (y - 2 a) / ((x_n + w - a)(x_m - a)) dot (y - 2 a) / ((x_m + w - a)(x_n - a)) = frac((y - 2 a)^2, Q_(-)(z) Q_(-)(v)) $
  by @eq-Qfact. For $mu != 0$, $(y - 2 a)^2 slash y = y - 4 a + 4 a^2 slash y$, so $C_(n m)(y - 2 a)^2 = rho (y - 4 a) + 4 a^2 C_(n m)$. For $mu = 0$ this holds too, because $rho = 0$ and $C_(n m) != 0$ only where $y = 0$. Thus @eq-input is equivalent to
  $ Rreg (z - w/2, v - w/2) = frac(-2 a - rho (z + v - 4 a), Q_(-)(z) Q_(-)(v)) + frac(4 a^2 (1 - rho)(z + v + 2 a)(z v + q_0), Q_(+)(z) Q_(-)(z) Q_(+)(v) Q_(-)(v)), $
  which is the rational identity (A1) of @app-identities.
]

= Results <sec-results>

All lattice sums that occur below are expressed through
$ S_k (x) := cases(-beta/2 thin tanh(beta (x - mu) slash 2) & quad k "even", -beta/2 thin coth(beta (x - mu) slash 2) & quad k "odd"), $ <eq-S>
which equals $PV sum_n (z_n - x)^(-1)$ for $x$ off the lattice (@lem-S).

== Singlet and triplet <sec-res-pp>

Let $b := 2 mu - w$ and $t_n := x_n (b - x_n)$; by @eq-z, $x_(overline(n)) = b - x_n$, so $t_(overline(n)) = t_n$. Define
$ cal(P)(t) := (t + a^2)^2 - a^2 b^2, quad quad cal(G)(t) := t + a rho b + a^2 rho^2, quad quad As := a rho b + a^2 (1 - 8 p_1), $
$ cal(T)_k (t) := -frac(S_k (mu + xi), xi), quad quad xi := sqrt(b^2 slash 4 - t). $ <eq-T>
$cal(T)_k$ is even in $xi$, so either branch may be used (for $xi = 0$ and even $k$, take the limit $beta^2 slash 4$). It equals $sum_n (t_n - t)^(-1)$ (@lem-T).

#theorem(title: [triplet], lean: [formalized as stated; moreover every right or left inverse of $chi_"t"$ gives @eq-Gamma-t, so $Gamma_"t"$ is unique (`Triplet.lean`: `exists_inverse_chiT`, `gammaT_of_isRightInverse`, `gammaT_of_isLeftInverse`, `gammaT_eq_zero_of_fixed`).])[
  Assume $t_n + a rho b + a^2 != 0$ for all $n != overline(n)$. Then for $n != overline(n)$
  $ Gamma_("t", n m) = frac(beta a^2 (1 - rho^2) cal(P)(t_n), cal(G)(t_n)(t_n + a rho b + a^2)) (delta_(n m) - delta_(m overline(n))). $ <eq-Gamma-t>
  At a fixed point $n = overline(n)$ (odd $k$, $n = -(k + 1) slash 2$), $Gamma_("t", n m) = Gamma_("t", m n) = 0$ for all $m$.
] <thm-t>

#theorem(title: [singlet], lean: [formalized in the generic case of the proof (@sec-pp: $U != 0$, $b != 0, plus.minus 2 a$, $d_(plus.minus) != 0$, $t_n + As != 0$ for all $n$), with $xi != 0$: an inverse of $chi_"s"$ from both sides, and @eq-Gamma-s (`Singlet.lean`: `chiS_isRightInverse`, `chiS_isLeftInverse`, `gammaS_eq`). The other cases, which the proof reaches by continuity, are not.])[
  Assume
  $ K_"s" := 1 + frac(2 a, beta) cal(T)_k (-As) != 0. $
  Then
  $ Gamma_("s", n m) = frac(beta a^2 (rho^2 - 1 + 8 p_1) cal(P)(t_n), cal(G)(t_n)(t_n + As)) (delta_(n m) + delta_(m overline(n))) + 4 a - frac(4 a thin cal(P)(-As), K_"s" (t_n + As)(t_m + As)). $ <eq-Gamma-s>
  If $t_n + As = 0$ for some $n$, the entries in the rows and columns $n$ and $overline(n)$ are the limits of @eq-Gamma-s as $mu$ approaches its value; the divergent parts cancel.
] <thm-s>

At $mu = 0$ both formulas reduce to Eq. 19 of Ref. @thunstroem2018 (@app-checks).

== Density and magnetic <sec-res-ph>

Let $epsilon := +1$ for $r = "d"$ and $epsilon := -1$ for $r = "m"$. Define
$ H(z) := z^2 + 2 a rho z + q_1, quad quad q_1 := q_0 + 4 epsilon a^2 p_1, quad quad lambda := 4 epsilon a^2, \
P(z) := Q_(+)(z) Q_(-)(z), quad quad G_2 (z) := (z + a rho)^2 - w^2/4, $
$ e(z) := H(z) H(-z) - lambda^2 h^2 slash beta^2 = (z^2 + q_1)^2 - 4 a^2 rho^2 z^2 - 16 a^4 p_0 p_2 = (z^2 - s_1)(z^2 - s_2), $ <eq-Delta>
where $s_(1, 2)$ are the roots of $s^2 + 2 (q_1 - 2 a^2 rho^2) s + q_1^2 - 16 a^4 p_0 p_2$, and $u_j$ is either square root of $s_j$. For $r = "m"$ and $k = 0$, $q_1 = a^2 (1 - 4 p_1)$, so $H(a) = 4 a^2 p_2$ and $H(-a) = 4 a^2 p_0$ by @eq-weights. Then $e(a) = 0$: one root is $s_1 = a^2$, at the double zeros $plus.minus a$ of $P$. We call this root pair _atomic_. With the thermal function
$ alpha_k (x) := rho thin S_k (x) - overline(c), $ <eq-alpha-def>
define
$ F(z) := frac(P(z), e(z)), quad quad g_j (z) := frac(1, z^2 - s_j), \
DD(z, v) := frac(z F(z) - v F(v), z^2 - v^2), quad quad DD(z, z) := frac(F(z) + z F'(z), 2 z), $
$ K_j := beta H(-u_j) + lambda alpha_k (u_j), quad quad Z_j := frac(u_j P(u_j) [beta H(-u_j) - lambda alpha_k (u_j)], K_j), quad quad gamma_j := frac(lambda rho Z_j, s_j - s_ell), $
with $ell != j$.

#theorem(title: [density and magnetic], lean: [formalized for $U != 0$ and $s_1 != s_2$, with $P(u_j) != 0$ except for the atomic pair: an inverse of $chi_r$ from both sides, and @eq-Gamma-ph, including $DD(z, z)$ and the case $r = "m"$, $k = 0$ (`DensityMagnetic.lean`: `DMGeneric.chiR_isRightInverse`, `DMGeneric.chiR_isLeftInverse`, `DMGeneric.gammaR_eq`, `dmGeneric_static_magnetic`). Not formalized: $s_1 = s_2$, the lattice zeros of $e$, and the independence of the sign of $u_j$.])[
  Let $mu != 0$, $e(z_n) != 0$ for all $n$, and $K_j != 0$. Then
  $ Gamma_(r, n m) = beta P(z_n) [frac(1, G_2 (z_n)) - frac(H(-z_n), e(z_n))] delta_(n m) + 2 epsilon a - lambda rho DD(z_n, z_m) + sum_(j = 1)^2 gamma_j thin g_j (z_n) g_j (z_m). $ <eq-Gamma-ph>
  If $s_1 = s_2$, the sum over $j$ is replaced by its limit $lambda rho thin partial_s [Z(s) slash ((z_n^2 - s)(z_m^2 - s))]$ at $s = s_1$, where $Z(s) := [Z(u) + Z(-u)] slash 2$ with $u^2 = s$, and $Z(u)$ is $Z_j$ with $u_j$ replaced by $u$. For $r = "m"$ and $k = 0$ the atomic pair is omitted, both from the sum and from the assumption $K_j != 0$. The formula does not depend on the sign chosen for $u_j$, and the inverse $chi_r^(-1)$ it represents is two-sided.
] <thm-ph>

So $Gamma_"d"$ and $Gamma_"m"$ have the same anatomy as $Gamma_"s"$: a diagonal term, the constant $plus.minus U$, a dense term without thermal factors, and two separable terms with one thermal denominator $K_j$ each. Only $alpha_k (u_j)$, i.e. $tanh$ or $coth$ at the two roots, has to be evaluated. The assumption $e(z_n) != 0$ is not essential either. As $e$ has real coefficients and $z_(overline(n)) = z_n^*$, lattice zeros of $e$ come in mirror orbits ${n, overline(n)}$. If $e$ vanishes on one orbit, @eq-Gamma-ph still holds, with the entries in the rows and columns of the orbit taken as limits as $mu$ approaches its value (@app-details).

At $mu = 0$, $rho = 0$ and @thm-ph does not apply literally. Here the Cauchy matrix is the mirror, $C = h cal(M)$ with $h = overline(c) = beta p_0$, $H(z) = z^2 + q_1$ is even and $e(z) = (z^2 + q_1)^2 - 16 a^4 p_0^2 = (z^2 - hat(s))(z^2 + 4 epsilon a^2 (p_0 + p_1) + q_0)$ with
$ hat(s) := 4 epsilon a^2 (p_0 - p_1) - q_0, quad quad hat(g)(z) := frac(1, z^2 - hat(s)), quad quad hat(D) := beta - frac(2 epsilon a S_k (hat(u)), hat(u)), $
where $hat(u)$ is either square root of $hat(s)$; $hat(D)$ is even in $hat(u)$. Explicitly, $hat(D) = beta [1 + epsilon a tanh(beta hat(u) slash 2) slash hat(u)]$ for even $k$, with the value $beta (1 + epsilon a beta slash 2)$ at $hat(u) = 0$; for odd $k$, $coth$ replaces $tanh$, and $hat(u) = 0$ means $e(0) = 0$ at the mirror fixed point $z_n = 0$ (see below).

#corollary(title: [half filling], lean: [not formalized.])[
  Let $mu = 0$, $e(z_n) != 0$ for all $n$, and $hat(D) != 0$. Then
  $ Gamma_(r, n m) = beta P(z_n) [frac(1, G_2 (z_n)) - frac(H(z_n), e(z_n))] delta_(n m) + 2 epsilon a - lambda h F(z_n) delta_(m overline(n)) - frac(2 epsilon a beta P(hat(u)), hat(D)) hat(g)(z_n) hat(g)(z_m), $ <eq-Gamma-hf>
  and the inverse $chi_r^(-1)$ it represents is two-sided.
] <cor-hf>

This is Eq. 19 of Ref. @thunstroem2018 for $r = "d", "m"$ (@app-checks). Compared with @eq-Gamma-ph, the dense term has become a mirror term, and only one separable term with one thermal denominator remains. The corollary is the limit $mu -> 0$ of @thm-ph (proof in @app-details). The assumption $e(z_n) != 0$ matters only for $beta H(z_n) = -lambda h$ with $n != overline(n)$, a genuine divergence (@sec-res-box). At all other zeros of $e(z_n)$, including both signs at a mirror fixed point (odd $k$, $z_n = 0$), @eq-Gamma-hf holds after taking limits (@app-details).


== Finite boxes and divergences <sec-res-box>

#proposition(title: [finite boxes], lean: [not formalized.])[
  Under the assumptions of @thm-t, @thm-s, @thm-ph or @cor-hf, the restrictions of $chi_r$ to the boxes $|2n + 1 + k| <= 2M + 1$ are invertible for all sufficiently large $M$, and their inverses converge entrywise to $chi_r^(-1) = Gamma_r slash beta^2 + eta_r bold(B)^(-1)$.
] <prop-box>

*Divergences.* An entry of $Gamma_r$ can diverge only where a denominator in its closed form vanishes. Removable are the zeros of $t_n + As$ (@thm-s), of $s_1 - s_2$, and the lattice zeros of $e$ on one mirror orbit for $mu != 0$ (@thm-ph); at $mu = 0$, the zeros of $e(z_n)$ with $beta H(z_n) = lambda h$, and both kinds at a mirror fixed point (@app-details). Away from intersections of these exceptional sets, the remaining zeros are genuine poles under the following nonvanishing conditions:
- $K_j = 0$ with $u_j P(u_j) != 0$ and $s_1 != s_2$ (d, m). Then $gamma_j K_j = 2 lambda rho beta u_j P(u_j) H(-u_j) slash (s_j - s_ell) != 0$, as $alpha_k (u_j) != 0$. At $u_j = 0$ the zero of $K_j$ can cancel, as at $hat(u) = 0$ in @cor-hf.
- $hat(D) = 0$ with $P(hat(u)) != 0$ (d, m at $mu = 0$).
- $beta H(z_n) = -lambda h$ with $n != overline(n)$ (d, m at $mu = 0$). There $N(z, dot) slash P$ is even, so by @eq-chi-ph $P(z) chi_r (bold(e)_n - bold(e)_(overline(n))) = -(beta H(z_n) + lambda h)(bold(e)_n - bold(e)_(overline(n))) = 0$: the mirror-odd vector is a null vector. For $r = "d"$, $k = 0$, $n = 0$ this is $beta U = 2 pi slash sqrt(3)$.
- $K_"s" = 0$ with $cal(P)(-As) != 0$ (s).
- $t_n + a rho b + a^2 = 0$ with $n != overline(n)$ (t); the numerator is $a^2 (1 - rho^2) cal(P)(t_n) != 0$.

#example[
  _Closed form against finite boxes_ ($r = "d"$, $U = 2$, $beta = 3$, $mu = 0.3$, $k = 1$). The inverses of $chi_"d"$ on the boxes of @prop-box ($2M + 1$ frequencies, as $k$ is odd) converge like $1 slash M$; the extrapolation $2 Gamma^((256)) - Gamma^((128))$ removes the leading error.
  #align(center, text(size: 9.5pt, table(
    columns: 5, align: right, stroke: none,
    table.hline(),
    [$(n, m)$], [$M = 128$], [$M = 256$], [extrapolated], [@eq-Gamma-ph],
    table.hline(stroke: 0.4pt),
    [$(0, 0)$], [$25.9568 + 33.8867 i$], [$25.9592 + 33.8867 i$], [$25.96155 + 33.88673 i$], [$25.96157 + 33.88673 i$],
    [$(0, overline(0))$], [$-16.2893$], [$-16.2870$], [$-16.28460$], [$-16.28459$],
    [$(2, -1)$], [$2.2186 + 0.0612 i$], [$2.2210 + 0.0612 i$], [$2.22334 + 0.06123 i$], [$2.22336 + 0.06123 i$],
    table.hline(),
  )))
]


#figure(
  placement: auto,
  image("doped-gamma/figures/gamma_cuts.svg", width: 100%),
  caption: [$Gamma_r (nu, pi slash beta, 0)$ from @eq-Gamma-ph (@eq-Gamma-hf at $mu = 0$), @eq-Gamma-s, @eq-Gamma-t at $U = 2$, $beta = 4$, for $mu = 0$ (half filling, real), $mu = 0.4$ and $mu = 1$. Away from half filling $Gamma_"d"$, $Gamma_"m"$ acquire imaginary parts; all $Gamma_r$ approach their bare values $U$, $-U$, $2U$, $0$ at large $|nu|$.],
) <fig-cuts>


= Matsubara sums <sec-sums>

#lemma(title: [basic sum], lean: [for $j = 1$, along the boxes $|2 n + 1 + k| <= 2 M + 1$, and for $j = 2$, from Mathlib's expansion of the cotangent; with @eq-shift and $S_(plus.minus)$ (`Matsubara.lean`: `hasSum_S_even`, `hasSum_S_odd`, `hasSum_sq`; `Thermal.lean`: `S_shift`, `S₀_neg_a`, `S₀_a`; `PHSums.lean`: `S_shift'`).])[
  For $x$ off the lattice, $PV sum_n (z_n - x)^(-1) = S_k (x)$ (@eq-S), and for $j >= 2$, $sum_n (z_n - x)^(-j) = S_k^((j - 1))(x) slash (j - 1)!$, absolutely convergent. Moreover, for all $k$,
  $ S_k (x plus.minus w slash 2) = -beta/2 thin tanh(beta (x - mu) slash 2). $ <eq-shift>
  In particular, $S_k$ takes the value $S_(plus.minus) := plus.minus (beta slash 2) tanh(beta (a plus.minus mu) slash 2)$ at both zeros $minus.plus a plus.minus w slash 2$ of $Q_(plus.minus)$; by @eq-weights, $S_+ = beta (p_1 - p_0) slash (2 (p_1 + p_0))$ and $S_- = beta (p_2 - p_1) slash (2 (p_1 + p_2))$.
] <lem-S>
#proof[
  For even $k$, $2n + 1 + k = 2 n' + 1$ with $n' = n + k slash 2$, so $z_n - x = i nu_(n') - (x - mu)$. For odd $k$, $2n + 1 + k = 2 n''$, so $z_n - x = i Omega_(n'') - (x - mu)$. In both cases the box is symmetric in $n'$ (resp. $n''$). The symmetric Mittag-Leffler expansions $tanh y = sum_j (y - i pi (j + 1/2))^(-1)$ and $coth y = sum_j (y - i pi j)^(-1)$ at $y = beta (x - mu) slash 2$ read $tanh(beta (x - mu) slash 2) = (2 slash beta) sum_j (x - mu - i nu_j)^(-1)$ and $coth(beta (x - mu) slash 2) = (2 slash beta) sum_j (x - mu - i Omega_j)^(-1)$, which is @eq-S. For $j >= 2$ the $(j - 1)$-fold $x$-derivative of the series converges absolutely and uniformly on compact sets away from the lattice, so it may be differentiated term by term. For @eq-shift: $w slash 2 = i pi k slash beta$ shifts the argument of $tanh$ or $coth$ by $plus.minus i pi k slash 2$. For even $k$ this is a multiple of the period $i pi$; for odd $k$, $coth(y plus.minus i pi slash 2) = tanh y$.
]

#lemma(title: [residue rule], lean: [not formalized as such. The sums it serves, with the denominators $P(t)$ and $P(t)(t - x)$, are evaluated by splitting $P = Q_+ Q_-$, for all $k$ at once (`PHSums.lean`: `hasSum_Q_sub`, `hasSum_inv_Q`, `hasSum_cubic`).])[
  Let $phi$ be proper rational without poles on the lattice. Then
  $ sum_n phi(z_n) = sum_(x in "poles"(phi)) Res_(t = x) [phi(t) S_k (t)], $ <eq-res>
  absolutely convergent if $phi = O(t^(-2))$, and as a PV sum if $phi tilde t^(-1)$. ($Res_(t = x)$ of a function is the coefficient of $(t - x)^(-1)$ in its Laurent expansion at $x$.)
] <lem-res>
#proof[
  Partial fractions write $phi$ as a finite sum of constants times $(t - x)^(-j)$, $j >= 1$, over its poles $x$. By @lem-S each $(t - x)^(-j)$ sums to $S_k^((j - 1))(x) slash (j - 1)!$, which is the residue of $(t - x)^(-j) S_k (t)$ at $x$, since $S_k$ is analytic there.
]

#lemma(title: [particle–particle sum], lean: [for $xi != 0$, with @eq-T-atomic for $b != plus.minus 2 a$ (`Matsubara.lean`: `hasSum_one_div_t_sub`; `Singlet.lean`: `hasSum_gp`, `hasSum_gm`).])[
  For $t != t_n$ ($forall n$), $sum_n (t_n - t)^(-1) = cal(T)_k (t)$ (@eq-T), absolutely convergent. At the two points $t_(plus.minus) := -a^2 plus.minus a b$, for $b != plus.minus 2 a$,
  $ cal(T)_k (t_+) = -frac(2 S_-, 2 a - b), quad quad cal(T)_k (t_-) = frac(2 S_+, 2 a + b). $ <eq-T-atomic>
  If $b = plus.minus 2 a$ (which needs $k = 0$), then $t_(plus.minus) = b^2 slash 4$, i.e. $xi = 0$, and $cal(T)_k (t_(plus.minus)) = beta^2 slash 4$.
] <lem-T>
#proof[
  Since $x_n = z_n - w slash 2$ and $b slash 2 = mu - w slash 2$, $t_n = b^2 slash 4 - (z_n - mu)^2$, so for $xi != 0$, $(t_n - t)^(-1) = [(z_n - mu + xi)^(-1) - (z_n - mu - xi)^(-1)] slash (2 xi)$. By @lem-S and the oddness of $tanh$ and $coth$, the sum is $(S_k (mu - xi) - S_k (mu + xi)) slash (2 xi) = -S_k (mu + xi) slash xi$; it converges absolutely, as $t_n tilde n^2$. For $xi = 0$, $(t_n - t)^(-1) = -(z_n - mu)^(-2)$ sums to $-S'_k (mu) = beta^2 slash 4$ by @lem-S. This occurs only for even $k$: for odd $k$, $t = b^2 slash 4$ is $t_n$ at the mirror fixed point. At $t_(plus.minus)$, $xi^2 = (a minus.plus b slash 2)^2$. At $t_+$ take $xi = a - b slash 2$: then $mu + xi = a + w slash 2$, and $S_k (mu + xi) = S_-$ by @eq-shift. At $t_-$ take $xi = -a - b slash 2$: then $mu + xi = -a + w slash 2$, and $S_k (mu + xi) = S_+$.
]

= Proofs for the singlet and triplet channels <sec-pp>

== Variables

With $t_(plus.minus)$ from @lem-T, $cal(P)(t) = (t - t_+)(t - t_-)$. Define
$ d_(plus.minus) := As + t_(plus.minus), quad quad c := c_(-k) = cases(rho slash b & quad (mu, k) != (0, 0), beta p_0 & quad mu = k = 0). $
Then $d_+ d_- = cal(P)(-As)$. For $t = x (b - x)$, direct expansion gives
$ (x - a)(b - x - a) = t - t_+, quad quad (x + a)(b - x + a) = t - t_-, \
(x + a rho)(b - x + a rho) = cal(G)(t), $ <eq-pp-ids>
and the product of the first two is $(x^2 - a^2)((b - x)^2 - a^2) = cal(P)(t)$. Hence, by @eq-B,
$ B_n = -beta/2 thin frac(cal(G)(t_n), cal(P)(t_n)) = B_(overline(n)). $ <eq-B-pp>

The pp entry $(n, m)$ at transfer $k$ is the ph entry at bosonic index $k' := -k - n - m - 1$, with frequency $w' := i omega_(k') = b - x_n - x_m$. Then $delta_(k' 0) = delta_(m overline(n))$, and the pair coefficient becomes $c_(n + m + k' + 1) = c$. In the variables of @lem-input at transfer $w'$, $z + v = x_n + x_m + w' = b$. By @eq-Qfact and @eq-pp-ids,
$ Q_(-)(z) Q_(-)(v) = (x_n - a)(b - x_n - a)(x_m - a)(b - x_m - a) = (t_n - t_+)(t_m - t_+), $ <eq-QQ-pp>
and likewise $Q_(+)(z) Q_(+)(v) = (t_n - t_-)(t_m - t_-)$.

== Structure

Let $cal(M)_(plus.minus) := (bb(1) plus.minus cal(M)) slash 2$ be the projectors onto mirror-even and mirror-odd vectors. A diagonal matrix whose entries depend on $n$ only through $t_n$ commutes with the mirror $cal(M)$. Define
$ L_"t" (t) := -beta/2 thin frac(t + a rho b + a^2, cal(P)(t)), quad L_"s" (t) := beta/2 thin frac(t + As, cal(P)(t)), quad g_(plus.minus)(t) := frac(1, t - t_(plus.minus)), quad gamma_(plus.minus) := frac(a (rho plus.minus 1)(2 a minus.plus b), 2 b), $
and $bold(L)_("s", "t") := diag(L_("s", "t")(t_n))$.

#proposition(lean: [@eq-chi-pp, for $chi_"s"$ with $b != 0$ and $chi_("s", "reg")$ in the form @eq-chi-s-reg (`Triplet.lean`: `chiT_eq_mirrorBlock`; `SingletInput.lean`: `chiS_eq`).])[
  $ chi_"t" = bold(B) cal(M)_+ + bold(L)_"t" cal(M)_-, quad quad chi_"s" = bold(L)_"s" cal(M)_+ - bold(B) cal(M)_- + chi_("s", "reg"), $ <eq-chi-pp>
  where
  $ (chi_("s", "reg"))_(n m) = frac(a (2 a c - 1)(t_n - t_-)(t_m - t_-) + a^2 (1 - rho)(b + 2 a)(t_n + t_m + 2 a^2), cal(P)(t_n) cal(P)(t_m)), $ <eq-chi-s-num>
  which for $b != 0$ equals
  $ (chi_("s", "reg"))_(n m) = gamma_+ g_+ (t_n) g_+ (t_m) + gamma_- g_- (t_n) g_- (t_m). $ <eq-chi-s-reg>
] <prop-pp>
#proof[
  Insert $w -> w'$ into @eq-chi-uu and @eq-chi-ud. In the $delta_(n m)$ terms, $x_n + w' = b - x_m = x_(overline(n))$; in the $delta_(k' 0) = delta_(m overline(n))$ terms, $x_m = x_(overline(n))$ and $x_n + w' = x_n$. With @eq-pp-ids (abbreviating $cal(P) := cal(P)(t_n)$),
  $ chi_("pp", up up) &= 2 B_n delta_(n m) - (beta a^2 (1 - rho^2)) / cal(P) (delta_(n m) - delta_(m overline(n))), \
  chi_("pp", up dn) &= -(4 beta a^2 p_1) / cal(P) delta_(n m) + (4 beta a^2 (p_0 p_2 - p_1^2)) / cal(P) delta_(m overline(n)) + cal(N)|_(w = w'). $

  _Triplet._ $(chi_("pp", up up) + 2 bold(B)) slash 4$ has $delta_(n m)$ coefficient $B_n - beta a^2 (1 - rho^2) slash (4 cal(P))$ and $delta_(m overline(n))$ coefficient $beta a^2 (1 - rho^2) slash (4 cal(P))$. Using @eq-B-pp, these equal $(B + L_"t") slash 2$ and $(B - L_"t") slash 2$. As $B$ and $L_"t"$ commute with $cal(M)$, $1/2 (B + L_"t") + 1/2 (B - L_"t") cal(M) = B cal(M)_+ + L_"t" cal(M)_-$.

  _Singlet, delta terms._ The $delta_(n m)$ coefficient of $(-chi_("pp", up up) + 2 chi_("pp", up dn) - 2 bold(B)) slash 4$ is $-B_n + beta a^2 (1 - rho^2 - 8 p_1) slash (4 cal(P)) = (L_"s" - B) slash 2$. The $delta_(m overline(n))$ coefficient is $beta a^2 [-(1 - rho^2) + 8 (p_0 p_2 - p_1^2)] slash (4 cal(P))$. By @eq-weight-ids and $p_0 + p_2 = 1 - 2 p_1$, the bracket is $1 - 8 p_1 - rho^2$, and the coefficient is $(L_"s" + B) slash 2$. Together: $L_"s" cal(M)_+ - B cal(M)_-$.

  _Singlet, remaining terms:_ $(2 slash 4) cal(N)|_(w = w')$. Insert into @eq-input: $4 a^2 C -> 4 a^2 c$, $z + v = b$, $z v + q_0 = a^2 + (t_n + t_m) slash 2$ at transfer $w'$ (identity (A3)), and @eq-QQ-pp. This gives @eq-chi-s-num; for $b != 0$, its partial fractions in $t_n$ and $t_m$ are @eq-chi-s-reg (identity (A3)).
]

== Proofs

#proof(of: [@thm-t])[
  $chi_"t"$ and $bold(B)$ act on each orbit ${n, overline(n)}$ separately. For $n != overline(n)$ the orbit carries a $2 times 2$ block with eigenvalue $B_n$ on the even vector and $L_"t" (t_n)$ on the odd vector; $L_"t" (t_n) != 0$ is the assumption $t_n + a rho b + a^2 != 0$, which must hold on every orbit for $chi_"t"$ to be invertible. So $chi_"t"^(-1) = bold(B)^(-1) cal(M)_+ + bold(L)_"t"^(-1) cal(M)_-$, and $Gamma_"t" = beta^2 (bold(L)_"t"^(-1) - bold(B)^(-1)) cal(M)_-$ by @eq-Gamma-def. Further, $(beta^2 slash 2)(1 slash L_"t" - 1 slash B) = -beta cal(P) [cal(G) - (t + a rho b + a^2)] slash ((t + a rho b + a^2) cal(G))$ with $cal(G) - (t + a rho b + a^2) = -a^2 (1 - rho^2)$. At a fixed point the orbit is ${n}$ with $cal(M)_- = 0$; then $chi_("t", n n) = B_n$, $Gamma_("t", n n) = 0$, and $L_"t" (t_n)$ does not enter.
]

#proof(of: [@thm-s])[
  _Idea._ $chi_"s"$ is an easily invertible matrix plus the rank-two term @eq-chi-s-reg, so the Woodbury formula applies. Its $2 times 2$ matrix becomes diagonal in a suitable basis.

  For $U = 0$ both sides vanish ($L_"s" = -B$, $chi_("s", "reg") = 0$). Let $U != 0$, and first assume $b != 0$, $b != plus.minus 2 a$, $d_(plus.minus) != 0$ and $t_n + As != 0$ for all $n$. Then $gamma_(plus.minus) != 0$, since $|rho| < 1$.

  _Woodbury._ Let $bold(g)$ be the $oo times 2$ matrix with rows $(g_+ (t_n), g_- (t_n))$ and $gamma := diag(gamma_+, gamma_-)$, so $chi_("s", "reg") = bold(g) gamma bold(g)^top$ and $cal(M)_+ bold(g) = bold(g)$. The matrix $bold(X) := bold(L)_"s" cal(M)_+ - bold(B) cal(M)_-$ has inverse $bold(X)^(-1) = bold(L)_"s"^(-1) cal(M)_+ - bold(B)^(-1) cal(M)_-$ (all products are finite sums). Set $bold(V)_"s" := bold(X)^(-1) bold(g) = bold(L)_"s"^(-1) bold(g)$ and $Psi := bold(g)^top bold(V)_"s" = sum_n (g_+, g_-)^top (g_+, g_-)(t_n) slash L_"s" (t_n)$, absolutely convergent (terms $O(t_n^(-1)) = O(n^(-2))$). If $gamma^(-1) + Psi$ is invertible, then
  $ Y := bold(X)^(-1) - bold(V)_"s" (gamma^(-1) + Psi)^(-1) bold(V)_"s"^top $
  satisfies $chi_"s" Y = bb(1) + bold(g) [gamma - (bb(1) + gamma Psi)(gamma^(-1) + Psi)^(-1)] bold(V)_"s"^top = bb(1)$. $Y$ is symmetric, so also $Y chi_"s" = bb(1)$. By @eq-Gamma-def ($eta_"s" = -1$),
  $ Gamma_"s" = beta^2 (bold(L)_"s"^(-1) + bold(B)^(-1)) cal(M)_+ - beta^2 bold(V)_"s" (gamma^(-1) + Psi)^(-1) bold(V)_"s"^top. $
  The first term gives the first term of @eq-Gamma-s: $(beta^2 slash 2)(1 slash L_"s" + 1 slash B) = beta cal(P) [1 slash (t + As) - 1 slash cal(G)]$ and $cal(G) - t - As = a^2 (rho^2 - 1 + 8 p_1)$.

  _Change of basis._ Let $kappa_n := 1 slash (t_n + As)$. Since $cal(P) = (t - t_+)(t - t_-)$ and $t - t_(minus.plus) = (t + As) - d_(minus.plus)$,
  $ frac(g_(plus.minus)(t_n), L_"s" (t_n)) = 2/beta (1 - d_(minus.plus) kappa_n), quad "i.e." quad (bold(V)_"s")_(n, :)^top = 2/beta thin Theta^top (1, kappa_n)^top, quad Theta := mat(1, 1; -d_-, -d_+). $
  Hence $(bold(V)_"s" (gamma^(-1) + Psi)^(-1) bold(V)_"s"^top)_(n m) = (2 slash beta)^2 (1, kappa_n) Xi^(-1) (1, kappa_m)^top$ with $Xi := Theta^(-top) (gamma^(-1) + Psi) Theta^(-1)$. Inserting $Psi = sum_n (2 slash beta)^2 L_"s" (t_n) Theta^top (1, kappa_n)^top (1, kappa_n) Theta$ and $(2 slash beta)^2 L_"s" = (2 slash beta)(t + As) slash cal(P)$,
  $ Xi = Theta^(-top) gamma^(-1) Theta^(-1) + 2/beta sum_n frac(1, cal(P)(t_n)) mat(t_n + As, 1; 1, 1 slash (t_n + As)). $
  By partial fractions over $t_+, t_-, -As$, the three sums are combinations of $cal(T)_k (t_+)$, $cal(T)_k (t_-)$ and $cal(T)_k (-As)$. Inserting @eq-T-atomic, the off-diagonal entries of the two terms cancel, and (identity (A4))
  $ Xi = diag(-frac(1, a), frac(K_"s", a d_+ d_-)). $
  So $gamma^(-1) + Psi$ is invertible iff $K_"s" != 0$, and
  $ -beta^2 (2/beta)^2 (1, kappa_n) Xi^(-1) (1, kappa_m)^top = -4 [-a + frac(a d_+ d_- kappa_n kappa_m, K_"s")] = 4 a - frac(4 a thin cal(P)(-As), K_"s" (t_n + As)(t_m + As)). $

  _Excluded cases._ Fix $k$. The values of $mu$ where $b = 0$, $b = plus.minus 2 a$, $d_(plus.minus) = 0$ or $t_n + As = 0$ for some $n$ are isolated, since these are nonconstant analytic functions of $mu$. @eq-Gamma-s contains no $b$, $2 a plus.minus b$ or $d_(plus.minus)$ in a denominator, and its divergent parts at $q := t_n + As -> 0$ cancel. Indeed, the orbit ${n, overline(n)}$ has $r in {1, 2}$ elements, so $cal(T)_k (-As) = r slash q + O(1)$ and $K_"s" = 2 a r slash (beta q) + O(1)$. Since $a^2 (rho^2 - 1 + 8 p_1) = cal(G)(t_n) - q$ and $cal(G)(t_n) != 0$ (@eq-pp-ids), the first term of @eq-Gamma-s is $beta cal(P)(-As)(delta_(n m) + delta_(m overline(n))) slash q + O(1)$, and the last one is $-2 beta cal(P)(-As) slash (r q) + O(1)$ for $m in {n, overline(n)}$ and $O(1)$ otherwise. For $m in {n, overline(n)}$, $delta_(n m) + delta_(m overline(n)) = 2 slash r$, so the $1 slash q$ terms cancel. Near all these values, $(chi_"s")_(n l) = O(|t_l|^(-1))$ and $Y_(l m)$ is bounded for $l in.not {n, overline(n), m, overline(m)}$, uniformly, where $Y = Gamma_"s" slash beta^2 - bold(B)^(-1)$ with @eq-Gamma-s. As $|t_l| tilde l^2$, @lem-cont extends $chi_"s" Y = bb(1)$ to these values.
]

= The Cauchy matrix <sec-cauchy>

Throughout this section $mu != 0$.

#lemma(title: [values on the reflected lattice], lean: [the last identity, `rho_mul_S_neg_z` (`PHSums.lean`); the value of $S'_k$ enters `hasSum_square`.])[
  For every $n$,
  $ S_k (-z_n) = beta/2 coth(beta mu), quad quad S'_k (-z_n) = beta^2 / (4 sinh^2(beta mu)), quad quad rho thin S_k (-z_n) = overline(c). $ <eq-mirror>
] <lem-mirror>
#proof[
  Write $-z_n = x - w slash 2$ with $x := -mu - i pi (2n + 1) slash beta$. By @eq-shift, $S_k (-z_n) = -(beta slash 2) tanh(-beta mu - i pi (2n + 1) slash 2) = (beta slash 2) coth(beta mu)$, as $tanh$ has period $i pi$ and $tanh(y - i pi slash 2) = coth y$. Differentiating @eq-shift in $x$ gives $S'_k (-z_n) = -(beta^2 slash 4) cosh^(-2)(-beta mu - i pi (2n + 1) slash 2) = beta^2 slash (4 sinh^2(beta mu))$. The last identity is @eq-weight-hyp.
]

So $S_k$ is _constant_ on the reflected lattice (@fig-plane). This is the source of all simplifications below.

#lemma(title: [square], lean: [`hasSum_square` (`PHSums.lean`).])[
  $sum_l C_(n l) C_(l m) = h^2 delta_(n m)$, absolutely convergent.
] <lem-C2>
#proof[
  $C_(n l) C_(l m) = O(|z_l|^(-2))$. For $n != m$, partial fractions $[(z_n + z_l)(z_l + z_m)]^(-1) = (z_m - z_n)^(-1) [(z_l + z_n)^(-1) - (z_l + z_m)^(-1)]$ and @lem-S give $rho^2 [S_k (-z_n) - S_k (-z_m)] slash (z_m - z_n) = 0$ by @lem-mirror. For $n = m$, $rho^2 sum_l (z_l + z_n)^(-2) = rho^2 S'_k (-z_n) = h^2$ by @lem-mirror and @eq-weight-hyp.
]

The action of $C$ on rational functions is governed by the thermal function $alpha_k$ of @eq-alpha-def.

#lemma(title: [thermal function], lean: [`αeven_eq`, `αodd_eq` (`Thermal.lean`), `α_mul_α_neg` (`PHSums.lean`).])[
  With $y_(plus.minus) := beta (x plus.minus mu) slash 2$,
  $ alpha_k (x) = cases(-h cosh(y_+) slash cosh(y_-) & quad k "even", -h sinh(y_+) slash sinh(y_-) & quad k "odd"), \
  alpha_k (x) alpha_k (-x) = h^2. $ <eq-alpha>
] <lem-alpha>
#proof[
  For even $k$, insert $S_k (x) = -(beta slash 2) tanh y_-$, $p_2 = p_0 e^(2 beta mu)$ and $h = beta p_0 e^(beta mu)$:
  $ alpha_k (x) = -beta/2 frac(p_2 e^(y_-) + p_0 e^(-y_-), cosh y_-) = -h frac(cosh(y_- + beta mu), cosh y_-), quad y_- + beta mu = y_+. $
  Under $x -> -x$, $y_(plus.minus) -> -y_(minus.plus)$, so the two quotients multiply to $1$. Odd $k$: replace $tanh, cosh$ by $coth, sinh$.
]

Let $cC$ be the action of $C$ on functions sampled on the lattice, $(cC phi)(z_m) := sum_n C_(m n) phi(z_n)$.

#lemma(title: [Cauchy calculus], lean: [for simple poles, `hasSum_cauchy`; the constant rule only together with the sum of @lem-U, in `hasSum_const` (`PHSums.lean`). Higher poles are not needed.])[
  For $x$ with $x$ and $-x$ off the lattice, and every lattice point $z$,
  $ cC[(t - x)^(-1)](z) = frac(alpha_k (x), z + x), quad quad cC[1](z) = overline(c) quad (PV). $ <eq-resolvent>
  Higher poles follow by applying $(1 slash j!) partial_x^j$; so $cC$ maps every proper rational function whose poles $x$ satisfy this condition to an explicit proper rational function.
] <lem-calculus>
#proof[
  Partial fractions and @lem-S give
  $ sum_n frac(rho, (z + z_n)(z_n - x)) = frac(rho [S_k (x) - S_k (-z)], z + x) = frac(alpha_k (x), z + x) $
  by @lem-mirror. The constant rule is $PV sum_n C_(m n) = rho S_k (-z_m) = overline(c)$. The sums of $x$-derivatives converge uniformly near any admissible $x$, so @eq-resolvent may be differentiated.
]

= Proofs for the density and magnetic channels <sec-ph>

== Structure

Let $delta := delta_(k 0)$ and
$ zeta_r := beta a^2 [1 - rho^2 + 4 epsilon (p_0 p_2 - p_1^2)], quad quad N_2 (v) := -2 epsilon a (v^2 + 2 a rho v + q_0) + delta zeta_r (v^2 - a^2), $
$ N_0 (v) := (v^2 + 2 q_0 - 4 a^2) N_2 (v) + (2 epsilon a - delta zeta_r) P(v), quad quad N(z, v) := N_0 (v) + z^2 N_2 (v). $ <eq-N>
$P$ is even by @eq-Qfact and has no zeros on the lattice. In $N_0$ the terms of degree four cancel, so $N_0 (v) slash P(v) = O(v^(-1))$ and $N_2 (v) slash P(v) = O(v^(-2))$. We identify a function $phi$ on the lattice with the vector $(phi(z_n))_n$ and write $(chi_r phi)(z_n) := sum_m chi_(r, n m) phi(z_m)$.

#proposition(title: [structure], lean: [`Pph_mul_chiR` (`PHInput.lean`).])[
  $ P(z_n) chi_(r, n m) = -beta H(z_n) delta_(n m) + lambda C_(n m) + frac(N(z_n, z_m), P(z_m)). $ <eq-chi-ph>
] <prop-ph>
#proof[
  Write $z := z_n$, $v := z_m$, and collect the terms of $chi_r = chi_(up up) + epsilon chi_(up dn)$ (@eq-chi-uu, @eq-chi-ud).

  _(a) $delta_(n m)$ terms:_ $-beta [G(x_n) G(x_n + w) + a^2 (1 - rho^2 + 4 epsilon p_1) J(x_n) J(x_n + w)]$. By @eq-G and @eq-Qfact, $G(x_n) G(x_n + w) = [(z + a rho)^2 - w^2 slash 4] slash P(z)$ and $J(x_n) J(x_n + w) = 1 slash P(z)$. The numerators add up to $(z + a rho)^2 - w^2 slash 4 + a^2 (1 - rho^2) + 4 epsilon a^2 p_1 = H(z)$.

  _(b) $delta_(k 0)$ terms_ (only for $k = 0$, where $w = 0$, $x_n = z$ and $P(z) = (z^2 - a^2)^2$): $zeta_r J(z) J(v)$, and $P(z) J(z) J(v) = (z^2 - a^2)(v^2 - a^2) slash P(v)$.

  _(c) Remaining terms:_ $epsilon cal(N)_(n m)$ with @eq-input. Since $C_(n m)(z + v) = rho$ and $Q_(+)(z) - Q_(-)(v) = (z + v)(z - v + 2 a)$, we have $Q_(+)(z) C_(n m) = Q_(-)(v) C_(n m) + rho (z - v + 2 a)$. With $1 slash Q_(-)(v) = Q_(+)(v) slash P(v)$ this gives
  $ P(z) cal(N)_(n m) = 4 a^2 C_(n m) + frac(1, P(v)) [& 4 a^2 rho (z - v + 2 a) Q_(+)(v) - 2 a Q_(+)(z) Q_(+)(v) \
  & + 4 a^2 (1 - rho)(z + v + 2 a)(z v + q_0)]. $
  The numerator is quadratic in $z$. Its $z$ coefficient is $4 a^2 Q_(+)(v) [rho - 1 + (1 - rho)] = 0$, because $v^2 + 2 a v + q_0 = Q_(+)(v)$. Its $z^2$ coefficient is $-2 a Q_(+)(v) + 4 a^2 (1 - rho) v = -2 a (v^2 + 2 a rho v + q_0)$, and its $z^0$ coefficient is $-2 a (v^2 + 2 q_0 - 4 a^2)(v^2 + 2 a rho v + q_0) + 2 a P(v)$ (identity (A2)). Times $epsilon$, this is the part of @eq-N without $zeta_r$. The term (b) is the rest, since $(v^2 - 2 a^2)(v^2 - a^2) - P(v) = -a^2 (v^2 - a^2)$ at $k = 0$.
]

So, after multiplication by $P$, the finite-rank part of $chi_r$ has only two output directions, $1$ and $z^2$.

== Rank-two sums

#lemma(title: [rank-two sums], lean: [`hasSum_rank_two` (`PHSums.lean`), with the residues at the zeros of $P$ written out (`Rhat`).])[
  For $x$ off the lattice with $P(x) != 0$, and every $z$,
  $ cal(I)(z; x) := sum_l frac(N(z, z_l), P(z_l)(z_l - x)) = overline(cal(I))(z; x) + alpha_k (x) frac(N(z, x), rho P(x)), $ <eq-U>
  absolutely convergent, where
  $ overline(cal(I))(z; x) := frac(overline(c) N(z, x), rho P(x)) + sum_(y : P(y) = 0) Res_(t = y) frac(S_k (t) N(z, t), P(t)(t - x)). $
  $overline(cal(I))$ depends on the temperature only through $beta$, $overline(c)$ and $S_(plus.minus)$ (@lem-S), and $P(x) overline(cal(I))(z; x)$ is a polynomial in $z$ and $x$.
] <lem-U>
#proof[
  At the zeros of $P$, $S_k = S_(plus.minus)$ (@lem-S); for $k = 0$ they are double, and there $S'_0 = S_0^2 - beta^2 slash 4$ by @eq-shift. The summand of $cal(I)$ is $O(z_l^(-2))$, since $N_0$ has degree at most three. @lem-res gives the residue $S_k (x) N(z, x) slash P(x)$ at $t = x$, plus the residues at the zeros of $P$; insert $rho S_k (x) = overline(c) + alpha_k (x)$ (@eq-alpha-def). In $x$, $overline(cal(I))$ has poles only at the zeros of $P$, of at most their multiplicity.
]

== The proof of @thm-ph

_Strategy._ For $U = 0$, $chi_r = bold(B)$ and $Gamma_"d" = Gamma_"m" = 0$; let $U != 0$. Fix a lattice point $v = z_m$. We construct a bounded function $y_v$ on the lattice with $chi_r y_v = bold(e)_m$, the $m$-th unit vector. By @eq-chi-ph this means, for every lattice point $z$,
$ -beta H(z) y_v (z) + lambda (cC y_v)(z) + sum_l frac(N(z, z_l), P(z_l)) y_v (z_l) = P(v) delta_(z v). $ <eq-column>
For $l != n$, $chi_(r, n l) = O(|z_l|^(-2))$ by @eq-chi-uu, @eq-chi-ud and @eq-input, so $sum_l chi_(r, n l) y_v (z_l)$ converges absolutely. Its parts in @eq-column converge at least as PV sums (@lem-calculus, @lem-res), so they may be evaluated separately. We use @lem-U at $x = -v$, and @lem-calculus and @lem-U at $x = plus.minus u_j$. These points are off the lattice ($mu != 0$, $e(z_n) != 0$), and $P(-v) = P(v) != 0$. We first assume that $P(u_j) != 0$ for the non-atomic pairs; Step 5 removes this assumption.

_Step 1 (the Cauchy column)._ Through $C$, a unit vector at $v$ produces the column $lambda rho slash (z + v)$. Conversely, $C$ maps $1 slash (t + v)$ back to a unit vector: $cC[(t + v)^(-1)](z) = rho^(-1) sum_l C(z, z_l) C(z_l, v) = (h^2 slash rho) delta_(z v)$ by @lem-C2. So try $y = D_v delta_(z v) + E_v slash (z + v)$. Insert it into the left side of @eq-column. Use $H(z) = H(-v) + (z + v)(z - v + 2 a rho)$, and @eq-U at $x = -v$, where $alpha_k (-v) = 0$ (@lem-mirror). The left side becomes
$ [-beta H(v) D_v + frac(lambda h^2, rho) E_v] delta_(z v) + frac(lambda rho D_v - beta H(-v) E_v, z + v) \
- beta E_v (z - v + 2 a rho) + D_v frac(N(z, v), P(v)) + E_v overline(cal(I))(z; -v). $
We choose $D_v$, $E_v$ such that the first term is $P(v) delta_(z v)$ and the second vanishes. The determinant of these two linear equations is $beta^2 H(v) H(-v) - lambda^2 h^2 = beta^2 e(v) != 0$, and the solution is
$ D_v = -frac(H(-v) P(v), beta e(v)), quad quad E_v = -frac(lambda rho F(v), beta^2). $
With these values the remaining terms are $1 slash (beta e(v))$ times the left side of the identity (A7),
$ lambda rho (z - v + 2 a rho) P(v) - H(-v) N(z, v) - frac(lambda rho, beta) P(v) overline(cal(I))(z; -v) \
= -e(v) N_2 (z) + lambda rho z [P(v) - e(v)] + lambda W(z; v^2). $ <eq-source>
Here $W(z; s) := W_0 (s) + z^2 W_2 (s)$ with polynomials $W_0$, $W_2$ of degree one in $s$, given in @app-identities. Both sides of @eq-source are polynomials in $z$ and $v$ (@lem-U). So the remaining terms are the polynomial
$ frac(1, beta) [-N_2 (z) + lambda rho z (F(v) - 1) + lambda frac(W(z; v^2), e(v))]. $ <eq-defect>

_Step 2 (the constant)._ For $y = 1$ the left side of @eq-column is $-beta H(z) + lambda overline(c) + PV sum_l N(z, z_l) slash P(z_l) = beta N_2 (z) slash (2 epsilon a)$, by $cC[1] = overline(c)$ (@lem-calculus) and identity (A6). So adding the constant $2 epsilon a slash beta^2$ to $y$ removes the first term of @eq-defect.

_Step 3 (pole pairs)._ The rest of @eq-defect must be removed by functions that are mapped to polynomials. A simple pole $1 slash (t - x)$ is mapped by $-beta H + lambda cC$ to poles at $x$ and at $-x$ (@lem-calculus). So consider $phi = f_+ slash (t - x) + f_- slash (t + x)$. For it, the left side of @eq-column is a polynomial plus
$ frac(-beta H(x) f_+ + lambda alpha_k (-x) f_-, z - x) + frac(lambda alpha_k (x) f_+ - beta H(-x) f_-, z + x). $
For $x != 0$, both residues vanish for some $(f_+, f_-) != 0$ iff the determinant $beta^2 H(x) H(-x) - lambda^2 alpha_k (x) alpha_k (-x) = beta^2 e(x)$ vanishes (@eq-alpha), i.e. iff $x = plus.minus u_j$. For $x = u_j$ we take
$ phi_j (t) := frac(beta H(-u_j), t - u_j) + frac(lambda alpha_j, t + u_j) = frac(K_j t + u_j [beta H(-u_j) - lambda alpha_j], t^2 - s_j), quad quad alpha_j := alpha_k (u_j). $ <eq-mode>
(For $u_j = 0$ the two poles coincide, $phi_j = K_j slash t$, and what follows holds unchanged.) With $H(z) - H(plus.minus u_j) = (z minus.plus u_j)(z plus.minus u_j + 2 a rho)$, the polynomial part is
$ -beta [K_j (z + 2 a rho) + u_j (beta H(-u_j) - lambda alpha_j)] + beta H(-u_j) cal(I)(z; u_j) + lambda alpha_j cal(I)(z; -u_j). $
To evaluate it, set $v = minus.plus u_j$ in @eq-source, where $e(v) = 0$ and $P(v) = P(u_j)$:
$ frac(lambda rho, beta) P(u_j) overline(cal(I))(z; plus.minus u_j) = lambda rho (plus.minus u_j + 2 a rho) P(u_j) - H(plus.minus u_j) N(z, minus.plus u_j) - lambda W(z; s_j). $
Insert this into @eq-U for $cal(I)(z; plus.minus u_j)$, with $alpha_k (-u_j) = h^2 slash alpha_j$ (@eq-alpha):
- The terms with $N(z, -u_j)$ carry the factor $lambda^2 h^2 - beta^2 H(u_j) H(-u_j) = -beta^2 e(u_j) = 0$.
- The terms with $N(z, u_j)$ cancel.
- The terms with $lambda rho P(u_j)$ cancel the constant part of the first bracket.
What remains is
$ P(z) (chi_r phi_j)(z) = -beta K_j [z + frac(W(z; s_j), rho P(u_j))]. $ <eq-mode-image>

_Step 4 (solving)._ $P$ and $e$ are monic and even of degree four, and $W(z; s)$ has degree one in $s$. Partial fractions in $v^2$ therefore give
$ F(v) - 1 = sum_j F_j g_j (v), quad quad frac(W(z; v^2), e(v)) = sum_j frac(W(z; s_j), s_j - s_ell) g_j (v), quad quad F_j := frac(P(u_j), s_j - s_ell). $
So the last two terms of @eq-defect are $beta^(-1) sum_j lambda rho F_j g_j (v) [z + W(z; s_j) slash (rho P(u_j))]$. By Steps 1–3 and @eq-mode-image,
$ y_v (z) := D_v delta_(z v) + frac(E_v, z + v) + frac(1, beta^2) [2 epsilon a + lambda rho sum_j frac(F_j g_j (v), K_j) phi_j (z)] $
solves @eq-column. To read off $Gamma_r$, note that $lambda rho F_j g_j (v) phi_j (z) slash K_j = lambda rho F_j z g_j (z) g_j (v) + gamma_j g_j (z) g_j (v)$ by @eq-mode. Further, $F(z) - F(v) = -(z^2 - v^2) sum_j F_j g_j (z) g_j (v)$ gives
$ DD(z, v) = frac(F(v), z + v) - z sum_j F_j g_j (z) g_j (v). $ <eq-defect-dd>
Hence
$ y_v (z) = -frac(H(-z) P(z), beta e(z)) delta_(z v) + frac(1, beta^2) [2 epsilon a - lambda rho DD(z, v) + sum_j gamma_j g_j (z) g_j (v)]. $
By @eq-G and @eq-Qfact, $B_n = -beta G_2 (z_n) slash P(z_n)$; so by @eq-Gamma-def, $beta^2 [y_(z_m)(z_n) - delta_(n m) slash B_n]$ is the right side of @eq-Gamma-ph.

_Step 5 (two-sidedness, branches, special cases)._
- Let $Y_(n m) := y_(z_m)(z_n)$, so that $chi_r Y = bb(1)$. $Y$ is symmetric because $DD$ is, and so is $chi_r$ (@eq-chi-uu, @eq-chi-ud, @eq-input). Hence $(Y chi_r)_(n m) = sum_l chi_(r, m l) Y_(l n) = (chi_r Y)_(m n) = delta_(n m)$, the same absolutely convergent series.
- Let $K(u) := beta H(-u) + lambda alpha_k (u)$. Since $alpha_k (u) alpha_k (-u) = h^2$ (@eq-alpha), for every $u$
  $ frac(beta H(-u) - lambda alpha_k (u), K(u)) + frac(beta H(u) - lambda alpha_k (-u), K(-u)) = frac(2 beta^2 e(u), K(u) K(-u)). $ <eq-branch>
  At a root the right side vanishes, so $Z_j$ and $gamma_j$ do not depend on the sign of $u_j$.
- _Confluent roots._ By @eq-branch, the odd part $[Z(u) - Z(-u)] slash 2 = beta^2 u P(u) e(u) slash (K(u) K(-u))$ vanishes at the roots, so $Z(s_j) = Z_j$ for the even part $Z(s)$ of @thm-ph. Hence the sum over $j$ in @eq-Gamma-ph is $lambda rho$ times the divided difference of $s |-> Z(s) slash ((z_n^2 - s)(z_m^2 - s))$ at $s_1$, $s_2$. Near a double root $u_star^2$, $Z(u)$ is analytic in $u$ at $plus.minus u_star$: $K(u_star) != 0$, and at a root $K(u) = 0$ iff $K(-u) = 0$ (@eq-alpha, @eq-Delta). So $Z(s)$, an even analytic function of $u$, is analytic in $s$, also for $u_star = 0$. As $mu$ approaches a value where $s_1 = s_2$ (a zero of the discriminant of $e$, a nonconstant analytic function of $mu$), the divided difference converges to the derivative stated in @thm-ph, and @lem-cont extends $chi_r Y = bb(1)$.
- For $r = "m"$ and $k = 0$ the atomic pair has $F_1 = P(a) slash (a^2 - s_2) = 0$ and $W(z; a^2) = 0$ (@app-identities). So its terms in Step 4 vanish, and Step 3 is needed only for the non-atomic pair.
- _Excluded cases._ Steps 1–4 assumed $P(u_j) != 0$ for the non-atomic pairs; they hold for $u_j = 0$ as well. For $r = "m"$, $k = 0$, $P(u_2) = 0$ would mean $s_2 = a^2 = s_1$, which is excluded. Otherwise, fix $k$: the values of $mu$ where the assumption fails are zeros of the resultant of $e$ and $P$, a nonconstant analytic function of $mu$, so such values are isolated. @eq-Gamma-ph has no $P(u_j)$ in a denominator and depends on $s_j$ only. Near such a value, $chi_(r, n l) = O(|z_l|^(-2))$ for $l != n$ and $Y_(l m)$ is bounded, uniformly; so @lem-cont extends $chi_r Y = bb(1)$.
- _Lattice roots._ The case $e(z_n) = 0$ is treated in @app-details. $square$

*Acknowledgment.* The derivations, proofs and verification scripts were developed with the help of AI assistants (Anthropic's Claude and OpenAI's GPT models). Every identity is checked by computer algebra (@app-identities) and every result numerically (@app-checks); @app-lean lists what is formalized in Lean.

#bibliography("doped-gamma-refs.yml", style: "american-physics-society")

#set heading(numbering: "A.1", supplement: [Appendix])
#counter(heading).update(0)

= Rational identities <app-identities>

Each identity is checked exactly with SymPy: (A1)–(A5) by `doped-gamma/verify_identities.py`, (A6) and (A7) by `doped-gamma/verify_dm.py` (both channels, $k != 0$ and $k = 0$, all parameters free). In Lean (@app-lean), (A1) and (A3) are proved as `compact_input` and `regular_pp`, (A4) as `singlet_A4`, (A6) as `identity_A6` and (A7) as `identity_A7`; (A2) and the first part of (A5) are proved within `Pph_mul_chiR`, and the consequences $e(a) = 0$ and $W_0 (a^2) = W_2 (a^2) = 0$ for $r = "m"$, $k = 0$ are `atomic_root`.

/ (A1): the twelve terms of @app-R equal the two-term form in the proof of @lem-input; and $Pi(x_n, x_m) = (z + v - 2 a)^2 slash (Q_(-)(z) Q_(-)(v))$.
/ (A2): the $z^0$ coefficient in the proof of @prop-ph: $4 a^2 rho (2 a - v) Q_(+)(v) - 2 a q_0 Q_(+)(v) + 4 a^2 (1 - rho) q_0 (v + 2 a) = -2 a (v^2 + 2 q_0 - 4 a^2)(v^2 + 2 a rho v + q_0) + 2 a P(v)$.
/ (A3): @eq-pp-ids; $z v + q_0 = a^2 + (t_n + t_m) slash 2$ at transfer $w' = b - x_n - x_m$; @eq-chi-s-num and @eq-chi-s-reg from the twelve terms of @app-R at $w'$ with $c = rho slash b$.
/ (A4): $Xi = diag(-1 slash a, K_"s" slash (a d_+ d_-))$ in the proof of @thm-s, and $d_+ d_- = cal(P)(-As)$.
/ (A5): the $delta_(n m)$ and $delta_(k 0)$ identities in the proof of @prop-ph, and $H(a) = 4 a^2 p_2$, $H(-a) = 4 a^2 p_0$ for $r = "m"$, $k = 0$.
/ (A6): Step 2 of the proof of @thm-ph: $-beta H(z) + lambda overline(c) + PV sum_l N(z, z_l) slash P(z_l) = beta N_2 (z) slash (2 epsilon a)$, the sum evaluated by @lem-res and @lem-U.
/ (A7): @eq-source, where $overline(cal(I))$ is evaluated by @lem-U and, with $pi.alt := 4 (p_0 p_2 - p_1^2)$,
  $ W_0 (s) &:= epsilon a q_0 [pi.alt - 2 epsilon (1 - rho^2)] (s + q_0 + 2 epsilon a^2) + delta beta a^4 psi(s), \
  W_2 (s) &:= epsilon a pi.alt (s + q_0 + 2 epsilon a^2) - delta beta a^2 psi(s), \
  psi(s) &:= -4 epsilon p_1 (p_0 + p_2 - rho^2) [s + (1 + 2 epsilon) a^2] + (1 + epsilon) rho^2 pi.alt (s - a^2). $
  For $r = "m"$ and $k = 0$ ($epsilon = -1$, $q_0 = a^2$) this shows $W_0 (a^2) = W_2 (a^2) = 0$.

= Analytic details <app-details>

#lemma(title: [continuity], lean: [not formalized.])[
  Let $chi(p)$ and $Y(p)$ be matrices indexed by the lattice, depending on a real parameter $p$ ($mu$ or $beta$) near $p_0$, with entries continuous at $p_0$. Suppose that for all $n$, $m$ there are $c_(n m)$ and a finite set of indices $l$ outside of which $|chi(p)_(n l) Y(p)_(l m)| <= c_(n m) (1 + l^2)^(-1)$ for all these $p$. If $chi(p) Y(p) = bb(1)$ for $p != p_0$, then also $chi(p_0) Y(p_0) = bb(1)$.
] <lem-cont>
#proof[
  For fixed $n$, $m$, the finitely many exceptional terms of $sum_l chi_(n l) Y_(l m)$ are continuous at $p_0$, and the sum of the others is continuous by dominated convergence.
]

*Zeros of $e(z_n)$ at $mu != 0$.* Since $e$ has real coefficients and $z_(overline(n)) = z_n^*$, $e(z_(overline(n))) = e(z_n)^*$: lattice zeros of $e$ come in mirror orbits $O = {n, overline(n)}$, with one element if $n = overline(n)$. Let $e$ vanish on one orbit $O$ at $mu = mu_0 != 0$, the other assumptions of @thm-ph holding (with $s_1 != s_2$ at $mu_0$). As $e(z_n)$ is a nonconstant analytic function of $mu$, @thm-ph holds for $mu != mu_0$ near $mu_0$. Each root $u_j$ that tends to a point $z_c$, $c in O$, has $(z_c - u_j) alpha_j -> rho != 0$ and $K_j -> oo$, since $(z_c - u) S_k (u) -> 1$ as $u -> z_c$ (@lem-S). In the column $y_v$ of Step 4 with $v = z_m$, $m in.not O$, the coefficients $D_v$, $E_v$ and $F_j g_j (v)$ stay bounded. By @eq-mode, for $c in O$,
$ frac(phi_j (z_c), K_j) = frac(beta H(-u_j) slash (z_c - u_j) + lambda alpha_j slash (z_c + u_j), beta H(-u_j) + lambda alpha_j) -> frac(beta H(-z_c), lambda rho) + frac(1, 2 z_c) quad (u_j -> z_c), $
it tends to $1 slash (z_c + z_(c'))$ if $u_j -> z_(c')$ with $c' != c$, and it stays bounded otherwise. So $Y_(c m) = y_(z_m)(z_c) = Y_(m c)$ converges for $c in O$, $m in.not O$, boundedly in $m$. For the block $(Y_(c c'))_(c, c' in O)$, row $l in.not O$ of $chi_r Y = bb(1)$ gives $sum_(c in O) chi_(r, l c) Y_(c c') = -sum_(l' in.not O) chi_(r, l l') Y_(l' c')$, whose right side converges by dominated convergence. The columns $P(z_l) chi_(r, l c) = lambda rho slash (z_l + z_c) + N(z_l, z_c) slash P(z_c)$, $c in O$ (@eq-chi-ph), are linearly independent rational functions of $z_l$, with distinct poles at $-z_c$. So $|O|$ rows $l in.not O$ can be chosen for which the matrix $(chi_(r, l c)(mu_0))$ is invertible, and the block converges. All entries of $Y$ converge, boundedly off the diagonal, and @lem-cont gives $chi_r Y = bb(1)$ at $mu_0$.

*Zeros of $e(z_n)$ at $mu = 0$.* Let $beta H(z_n) = lambda h$ with $n != overline(n)$ at $beta = beta_0$, so that $z_n^2 = hat(s)$ and $e(z) = (z^2 - hat(s))(z^2 - hat(s) + 2 lambda h slash beta)$. As $beta -> beta_0$, $hat(u) -> z_n$. The diagonal and mirror terms of @eq-Gamma-hf contribute $-lambda h P(z_n) slash e(z_n) + O(1) = -beta P(z_n) slash (2 (z_n^2 - hat(s))) + O(1)$ to each of the entries $(n, n)$ and $(n, overline(n))$. Since $S_k (hat(u)) = 1 slash (z_n - hat(u)) + O(1)$, $hat(D) = 2 epsilon a slash (hat(u)(hat(u) - z_n)) + O(1)$, so the separable term contributes $beta P(z_n) slash (2 (z_n^2 - hat(s))) + O(1)$ to the same entries and $O(1)$ to the others. The singular parts cancel.

At a mirror fixed point $n = overline(n)$ (odd $k$, $z_n = 0$), let $e(0) = 0$ at $beta = beta_0$. The diagonal and mirror terms of @eq-Gamma-hf act on the same entry $(n, n)$. As $beta^2 e(0) = (beta H(0) - lambda h)(beta H(0) + lambda h)$, they combine to $beta P(0) slash G_2 (0) - beta^2 P(0) slash (beta H(0) - lambda h)$. If $beta H(0) = -lambda h$ at $beta_0$, this stays finite, and under the remaining assumptions of @cor-hf nothing else is singular. If $beta H(0) = lambda h$, then $s := hat(s) -> 0$, and the combination is $beta P(0) slash s + O(1)$, as $beta H(0) - lambda h = -beta s$. For odd $k$, $coth(beta hat(u) slash 2) slash hat(u) = 2 slash (beta s) + O(1)$, so $hat(D) = 2 epsilon a slash s + O(1)$, and the separable term is $-2 epsilon a beta P(hat(u)) hat(g)(0)^2 slash hat(D) = -beta P(0) slash s + O(1)$ at $(n, n)$ and $O(1)$ elsewhere. The singular parts cancel again.

In all these cases @eq-Gamma-hf extends continuously, and @lem-cont (with $p = beta$) gives $chi_r Y = bb(1)$ at $beta_0$.


#proof(of: [@cor-hf])[
  Take $mu -> 0$ at fixed $k$ in @eq-Gamma-ph. Then $rho -> 0$, $overline(c) -> h$ and $C_(n m) -> h delta_(m overline(n))$ (@eq-C). One root $s_j$ of $e$ tends to $hat(s)$, the other, $s_ell$, to $-4 epsilon a^2 (p_0 + p_1) - q_0$. At a root $u_j$ put $X_(plus.minus) := beta H(plus.minus u_j)$. Then $X_+ X_- = lambda^2 h^2$ (@eq-Delta) and $X_+ - X_- = 4 a beta rho u_j$. With $overline(c)^2 = h^2 + beta^2 rho^2 slash 4$ (@eq-weight-ids) this gives
  $ (X_- - lambda overline(c))(X_- + lambda overline(c)) = -rho (4 a beta u_j X_- + lambda^2 beta^2 rho slash 4). $
  By @eq-alpha-def, $K_j = X_- - lambda overline(c) + lambda rho S_k (u_j)$ and $X_- - lambda alpha_k (u_j) = X_- + lambda overline(c) - lambda rho S_k (u_j)$.
  - _The pair $j$._ Here $X_(plus.minus) -> lambda h$, so the identity gives $(X_- - lambda overline(c)) slash rho -> -2 a beta hat(u)$. Hence $K_j slash rho -> lambda S_k (hat(u)) - 2 a beta hat(u) = -2 a hat(u) hat(D)$, $X_- - lambda alpha_k (u_j) -> 2 lambda h$ and $s_j - s_ell -> 2 lambda h slash beta$. So $gamma_j -> -2 epsilon a beta P(hat(u)) slash hat(D)$ if $hat(u) != 0$.
  - _The pair $ell$._ Here $X_(plus.minus) -> -lambda h$, so $K_ell -> -2 lambda h$, $X_- - lambda alpha_k (u_ell) -> 0$ and $gamma_ell -> 0$. (For $r = "m"$, $k = 0$ this is the atomic pair, absent anyway.)
  - _The dense term._ By @eq-defect-dd, $lambda rho DD(z_n, z_m) = lambda C_(n m) F(z_m) - lambda rho z_n sum_j F_j g_j (z_n) g_j (z_m) -> lambda h F(z_n) delta_(m overline(n))$, since $z_(overline(n)) = -z_n$ at $mu = 0$ and $F$ is even.
  So @eq-Gamma-ph tends entrywise to @eq-Gamma-hf. For small $mu != 0$ the assumptions of @thm-ph hold: $e(z_n) != 0$ by continuity (automatically for large $|n|$), $s_1 != s_2$, $K_ell != 0$, and $K_j != 0$ because $K_j slash rho -> -2 a hat(u) hat(D) != 0$ (for $hat(u) != 0$). The entries of $C$ are bounded uniformly near $mu = 0$ (by $beta rho slash (2 pi)$ for $m != overline(n)$, and $rho slash (2 mu) -> beta p_0$). So $chi_(r, n l) = O(|z_l|^(-2))$ for $l != n$, and the entries of $Y = Gamma_r slash beta^2 + bold(B)^(-1)$, with $Gamma_r$ from @eq-Gamma-ph, are bounded off the diagonal, uniformly near $mu = 0$. So @lem-cont gives $chi_r Y = bb(1)$ at $mu = 0$; $Y$ stays symmetric. Finally, for even $k$, $hat(u) = 0$ occurs at isolated $beta$, where @eq-Gamma-hf is continuous in $beta$ (as $hat(D) != 0$); @lem-cont with $p = beta$ covers it. For odd $k$, $hat(u) = 0$ means $e(0) = 0$ at the mirror fixed point, treated above.
]

#proof(of: [@prop-box])[
  Let $theta.alt_n := (1 + |z_n - mu|^2)^(1 slash 2)$, $bold(theta.alt) := diag(theta.alt_n)$, and $hat(chi) := bold(theta.alt) chi_r bold(theta.alt)$ on $ell^2$.
  - _$hat(chi) = kappa_r bb(1) + hat(chi)_c$ with $hat(chi)_c$ compact_, where $kappa_"d" = kappa_"m" = beta$ and $kappa_"s" = -kappa_"t" = beta slash 2$.
    - $r = "d", "m"$: by @eq-chi-ph, $chi_(r, n n) = -beta H(z_n) slash P(z_n) + O(z_n^(-4))$, so $theta.alt_n^2 chi_(r, n n) -> beta$. Off the diagonal, $chi_(r, n m) = O(|z_n|^(-2) |z_m|^(-2))$ by @eq-chi-uu, @eq-chi-ud and @eq-input. So these entries of $hat(chi)$ are square-summable, i.e. Hilbert–Schmidt.
    - $r = "s", "t"$: $t_n slash theta.alt_n^2 -> 1$, so $theta.alt_n^2 B_n -> -beta slash 2$, $theta.alt_n^2 L_"s" (t_n) -> beta slash 2$ and $theta.alt_n^2 L_"t" (t_n) -> -beta slash 2$ (@eq-B-pp). By @eq-chi-pp, $hat(chi) - kappa_r bb(1)$ is a matrix supported on the diagonal and the mirror with entries tending to zero, which is compact, plus (for $r = "s"$) $bold(theta.alt) chi_("s", "reg") bold(theta.alt)$, which is Hilbert–Schmidt by @eq-chi-s-num.
  - _$hat(chi)$ is invertible._ For each $m$, $hat(chi)$ maps $bold(theta.alt)^(-1) chi_r^(-1) bold(e)_m$ to $theta.alt_m bold(e)_m$: the $n$-th entry of the image is $theta.alt_n sum_l chi_(r, n l) (chi_r^(-1))_(l m) = theta.alt_n delta_(n m)$, an absolutely convergent series. The vector $bold(theta.alt)^(-1) chi_r^(-1) bold(e)_m$ lies in $ell^2$, because by the explicit formulas $(chi_r^(-1))_(n m)$ is bounded in $n$ for $n in.not {m, overline(m)}$. So the range of $hat(chi)$ contains all basis vectors; being closed (a compact perturbation of $kappa_r bb(1)$ has closed range), it is all of $ell^2$. A compact perturbation of $kappa_r bb(1)$ that is onto is also injective (Fredholm alternative), hence invertible.
  _Convergence._ Let $bb(1)_M$ be the projection onto the box. Then $hat(chi)_M := bb(1)_M hat(chi) bb(1)_M + kappa_r (bb(1) - bb(1)_M)$ converges to $hat(chi)$ in norm, since $(bb(1) - bb(1)_M) hat(chi)_c -> 0$ and $hat(chi)_c (bb(1) - bb(1)_M) -> 0$ for compact $hat(chi)_c$. So $hat(chi)_M$ is invertible for all sufficiently large $M$, and $hat(chi)_M^(-1) -> hat(chi)^(-1)$ in norm. On the box, $hat(chi)_M^(-1) = bold(theta.alt)^(-1) (bb(1)_M chi_r bb(1)_M)^(-1) bold(theta.alt)^(-1)$.
]

#pagebreak(weak: true)
= The regular part <app-R>

The regular part in @eq-pair is [Ref. @essl2024, Eq. (D3), at zero field]

#text(size: 9.5pt)[$ Rreg (x, y) = (p_0 + p_1) [ & - 1 / ((y + w + a)(y + a)(x - a)) + 1 / ((x + w - a)(x - a)(y + a)) \
  & + 1 / ((x + w + a)(y - a)(x - a)) - 1 / ((x + w + a)(x - a)(y + w + a)) \
  & + 1 / ((y + a)(x + w + a)(y + w + a)) + 1 / ((y + a)(x + w + a)(x + a)) \
  & + 1 / ((y - a)(x + a)(y + w - a)) - 1 / ((y - a)(x + w + a)(x + a)) \
  & + 1 / ((x + w - a)(x + a)(y + w - a)) - 1 / ((y + a)(x + w - a)(x + a))] \
  - (p_1 + p_2) [ & 1 / ((y - a)(x + w - a)(y + w - a)) + 1 / ((y - a)(x + w - a)(x - a))]. $]


= Numerical checks <app-checks>

- *Mathematica:* `mathematica/doped_gamma.nb` derives #eqrange(<eq-chi-uu>, <eq-pair>) exactly from the Lehmann representation of @eq-H, checks @lem-input, @prop-pp, @prop-ph, the lemmas of @sec-sums and @sec-cauchy, @thm-t, @thm-s, Steps 1–4 of the proof of @thm-ph and (A2)–(A4), (A6), (A7) symbolically, and compares @thm-ph, @thm-s, @thm-t and @cor-hf with finite-box inversion (62 checks).
- *Vertices* (`doped-gamma/check_vertices.jl`): @eq-Gamma-t and @eq-Gamma-s agree with an independent residue-based BigFloat evaluator (`doped-gamma/doped_gamma.jl`) to $3 times 10^(-15)$, @eq-Gamma-ph at 256 bits to $10^(-65)$ (300 entries, $k = 0, 1, 2, -3, 5$, $|mu| <= 2 |U|$); at $mu = 0$, @eq-Gamma-t, @eq-Gamma-s and @eq-Gamma-hf agree with `HubbardAtoms.jl` (Eq. 19 of Ref. @thunstroem2018) to $3 times 10^(-14)$.
- *Implementation:* the test suite of `HubbardAtoms.jl` compares `DopedHubbardAtom` with exact diagonalization, with finite-box inversion and with `BigFloat` evaluation, including the removable singularities listed under _Divergences_.


= Formalization in Lean <app-lean>

The Lean 4 project `lean/`, next to this note in the repository of HubbardAtoms.jl, builds on Mathlib and formalizes @thm-t, the generic cases of @thm-s and @thm-ph, and the statements their proofs use. Each numbered statement above is marked with its status. The names are declarations in the namespace `HubbardAtom.Params`, the auxiliary rational identities in `HubbardAtom`; the versions of Lean and Mathlib are pinned in the project. Its script `scripts/check.sh` builds it from scratch, fails on any warning (an unfinished proof, `sorry`, is one), and checks that every declaration depends only on Lean's three standard axioms.

*Taken as given.* The theorems are statements about definitions, which the reader has to compare with this note:
- the model: the weights @eq-weights from the energies of @eq-H, the frequencies and the lattice of @sec-lattice, and $G$ (proved equal to its Lehmann form, @eq-G);
- the input #eqrange(<eq-chi-uu>, <eq-pair>) with @app-R, as stated here; Lean does not derive it from the Hamiltonian;
- the particle–particle form, the channel combinations, @eq-B and @eq-Gamma-def. A right inverse $Y$ of $chi$ means $sum_l chi_(n l) Y_(l m) = delta_(n m)$ for all $n$, $m$, with unconditionally, i.e. absolutely, convergent sums; likewise for left inverses;
- the closed forms @eq-Gamma-t, @eq-Gamma-s and @eq-Gamma-ph, with $DD(z, z)$ through the derivative of $F$, and the assumptions, as typed in Lean.
The inverses constructed in the proofs need no trust: Lean proves that they are inverses from both sides.

*Proofs.* Besides the marked statements, Lean proves Steps 1–4 of the proof of @thm-ph (`DMGeneric.hasSum_step1`, `DMGeneric.hasSum_step2`, `DMGeneric.hasSum_step3`, `step4_identity`, and `DMGeneric.DD_eq` for @eq-defect-dd), and of Step 5 the two-sidedness (`chiR_symm`, `DMGeneric.invR_symm`) and the atomic pair (`atomic_root`); further the identities of @app-identities. Two arguments differ from this note. Lattice sums with the denominator $P$ are evaluated by splitting $P = Q_+ Q_-$ instead of by @lem-res, which needs no separate treatment of the double zeros at $k = 0$. And the inverse of $chi_"s"$ is verified directly; the cancellation of the Woodbury step is `woodbury_identity`.

*Not formalized.* @cor-hf, @prop-box, @lem-cont and the rest of @app-details; the cases of @thm-s and @thm-ph that their proofs reach by continuity; the independence of @eq-Gamma-ph of the sign of $u_j$; that the inverses of $chi_"s"$, $chi_"d"$ and $chi_"m"$ are unique (for $chi_"t"$ it is proved); and floating-point evaluation.
