// The channel-irreducible vertex of the Hubbard atom away from half filling: a derivation.
// Compile: typst compile derivation/doped-gamma-derivation.typ
// Figures: doped-gamma/figures/ (figure_data.jl, then make_figures.py).

#set document(title: [Irreducible vertex of the Hubbard atom away from half filling])
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.72em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)", supplement: [Eq.])
#show heading.where(level: 1): set block(above: 1.6em, below: 0.9em)
#show figure.caption: set text(size: 9.5pt)
#show figure.where(kind: table): set block(breakable: true)

// ---------- theorem-like environments (one shared counter, referenceable) ----------
#let thm-kind = "thm"
// `lean`: "yes" or "part" marks the statement as formalized (in part) in Lean, linking to the
// table of App. E
#let lean-tag(s) = box(inset: (x: 2.5pt), outset: (y: 1.5pt), radius: 2pt, fill: luma(232),
  link(<tab-lean>, text(size: 7.5pt, fill: luma(50), if s == "part" [Lean, in part] else [Lean])))
// Statements stay on one page unless created with `breakable: true`. The flag travels in the
// figure's `gap` (unused, as the caption is printed in the header).
#let breakable-gap = 0.66em
#show figure.where(kind: thm-kind): set block(breakable: true)
#show figure.where(kind: thm-kind): it => block(
  breakable: it.gap == breakable-gap,
  width: 100%,
  inset: (left: 9pt, y: 4pt),
  stroke: (left: 1.2pt + luma(140)),
  align(left)[
    *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [#it.caption.body]. #it.body
  ],
)
#let env(supplement) = (body, title: none, lean: none, breakable: false) => figure(
  body,
  gap: if breakable { breakable-gap } else { 0.65em },
  kind: thm-kind,
  supplement: supplement,
  numbering: "1",
  outlined: false,
  caption: if title == none and lean == none { none } else [#if title != none [ (#title)]#if lean != none [ #lean-tag(lean)]],
)
#let lemma = env[Lemma]
#let proposition = env[Proposition]
#let theorem = env[Theorem]
#let corollary = env[Corollary]
#let proof(body, of: none) = block(width: 100%, inset: (left: 9pt))[_Proof#if of != none [ of #of]._ #body #h(1fr) $square$]
#let example(body) = block(breakable: false, box(width: 100%, inset: 7pt, fill: luma(245), radius: 2pt)[*Example.* #body])
// a short explanation next to a definition in a display
#let gloss(body) = text(size: 9pt, fill: luma(80), body)

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

The irreducible vertex $Gamma_r$ of the Hubbard atom is known in closed form at half filling @thunstroem2018. Here we derive it at arbitrary filling (zero magnetic field) in all four channels $r in {"d", "m", "s", "t"}$. Appendices F and G also state the vertices and analyze their divergences with simultaneous doping and Zeeman field, in the coupled longitudinal, transverse-spin, opposite-spin pair and equal-spin pair sectors. By definition
$ Gamma_r = beta^2 (chi_r^(-1) - eta_r chi_(0, r)^(-1)), $
where $chi_r$ is the generalized susceptibility, an infinite matrix in two fermionic Matsubara frequencies, and $chi_(0, r)$ is its bare part. So the task is to invert $chi_r$ in closed form.

At half filling $chi_r$ is a diagonal matrix plus a mirror term plus a matrix of rank at most two, and the inversion is elementary. Away from half filling the mirror term becomes a dense _Cauchy matrix_ $C_(n m) = rho slash (z_n + z_m)$ (@fig-structure). The particle–hole susceptibility is then no longer a finite-rank perturbation of an easily invertible matrix. Two properties of $C$ still allow a closed-form inverse:
+ Like the mirror it replaces, $C$ is a scaled involution: $C^2 = h^2 bb(1)$ for a number $h$.
+ $C$ maps a simple pole to a simple pole, with a thermal factor $alpha$:
  $ sum_m frac(C_(n m), z_m - x) = frac(alpha(x), z_n + x), quad quad alpha(x) alpha(-x) = h^2. $

After multiplication by a polynomial, $chi_r$ is a diagonal matrix plus $lambda C$ plus a matrix of rank two. Each column of $chi_r^(-1)$ is then built from a diagonal entry, one Cauchy column, a constant, and two pole pairs with one thermal denominator each. All coefficients follow from $2 times 2$ linear systems whose determinants are given by a single quartic.

*Results.*
- *Triplet:* $Gamma_"t"$ is a diagonal-minus-mirror term (@thm-t).
- *Singlet:* $Gamma_"s"$ is a diagonal-plus-mirror term, the constant $2U$, and one separable term with a single thermal denominator (@thm-s).
- *Density and magnetic:* $Gamma_"d", Gamma_"m"$ have the same anatomy: a diagonal term, the constant $plus.minus U$, a dense term without thermal factors, and two separable terms with one thermal denominator each (@thm-ph). At half filling the dense term becomes a mirror term and one separable term remains (@cor-hf).
- *Removable singularities:* where a denominator of these formulas vanishes without $Gamma_r$ diverging, the limits are given explicitly (@prop-special).

The inverses are two-sided and agree with the limit of inverses on finite frequency boxes, i.e. with what numerical inversion converges to (@prop-box).

*Roadmap.*
- @sec-setup defines all quantities; @sec-input states the input, the susceptibility from the Lehmann representation.
- @sec-results states all results. The rest of the note proves them.
- @sec-sums evaluates the Matsubara sums; @sec-pp proves the particle–particle results, which need no Cauchy matrix.
- @sec-cauchy develops the algebra of $C$, which @sec-ph uses for the particle–hole channels. The limit $mu -> 0$, the removable singularities and the finite-box statement are proved in @app-details.
- @app-explicit writes the vertices out in terms of $U$, $beta$, $mu$ and the frequencies, for implementation; @app-div derives and plots the divergences of $Gamma_r$; @app-symbols lists the symbols with the place where each is defined.

*What is assumed.* The zero-field results, except the numerical parts of @app-div, are proved from the input #eqrange(<eq-chi-uu>, <eq-pair>) (Ref. @essl2024) and the rational identities of @app-identities (checked by computer algebra). The Mathematica notebook `mathematica/doped_gamma.nb` derives the input from the Hamiltonian and replays the proofs step by step (@app-checks). Independent numerical checks are listed in @app-checks. Most of the proofs are also formalized in Lean 4, which also derives the input from the Hamiltonian; statements marked #lean-tag("yes") or #lean-tag("part") are formalized, and the table in @app-lean says how much of each. The field formulas are stated without new derivations; their domains and the numerical status of longitudinal axis and exceptional-set extensions are specified in @sec-field-limits. Lean formalizes the transverse, opposite-spin and equal-spin pair vertices and derives the field inputs from the Hamiltonian; the longitudinal vertex is not formalized (@app-lean). Numerical stability in floating point is not discussed here.

*Implementation.* The zero-field results are implemented as `DopedHubbardAtom` in the Julia package #link("https://github.com/tuwien-cms/HubbardAtoms.jl")[HubbardAtoms.jl], which also evaluates them stably at low temperatures and near the removable singularities, and is tested against exact diagonalization and finite-box inversion. `FieldHubbardAtom` implements the four field sectors of @sec-field-explicit.

= Setup <sec-setup>

== Model and thermal weights

The atom is
$ cal(H) = U (n_up - 1/2)(n_dn - 1/2) - mu (n_up + n_dn - 1), quad a := U/2, $ <eq-H>
with real interaction $U$, chemical potential $mu$ (measured from half filling, $mu = 0$) and inverse temperature $beta > 0$. Its four states have

#block(breakable: false, align(center, table(
  columns: 5, align: center, stroke: none, inset: (x: 9pt, y: 4pt),
  table.hline(),
  [state], [$|0 chevron.r$], [$|up chevron.r$], [$|dn chevron.r$], [$|up dn chevron.r$],
  table.hline(stroke: 0.4pt),
  [energy], [$U/4 + mu$], [$-U/4$], [$-U/4$], [$U/4 - mu$],
  [Boltzmann weight], [$p_0$], [$p_1$], [$p_1$], [$p_2$],
  table.hline(),
)))

Here $p_1$ is the weight of _each_ singly occupied state, so
$ p_0 + 2 p_1 + p_2 = 1, quad quad p_2 / p_0 = e^(2 beta mu), quad quad p_1 / p_0 = e^(beta (a + mu)), quad quad p_1 / p_2 = e^(beta (a - mu)). $ <eq-weights>
Three combinations of the weights recur:
$ rho &:= p_2 - p_0 = chevron.l n_up + n_dn chevron.r - 1, && quad #gloss[the filling; $rho = 0$ at half filling] \
h &:= beta sqrt(p_0 p_2), && quad #gloss[$C^2 = h^2 bb(1)$ (@lem-C2)] \
overline(c) &:= beta (p_0 + p_2) slash 2. && quad #gloss[$PV sum_m C_(n m) = overline(c)$ (@lem-calculus)] $ <eq-rho>
By @eq-weights,
$ p_1 + p_2 = (1 + rho) / 2, quad quad p_0 + p_1 = (1 - rho) / 2, quad quad rho^2 + 4 p_0 p_2 = (p_0 + p_2)^2, quad quad overline(c)^2 - h^2 = beta^2 rho^2 slash 4, $ <eq-weight-ids>
and for $mu != 0$, inserting $p_2 = p_0 e^(2 beta mu)$,
$ rho = 2 p_0 e^(beta mu) sinh(beta mu), quad p_0 + p_2 = 2 p_0 e^(beta mu) cosh(beta mu) \
==> quad rho coth(beta mu) = p_0 + p_2, quad rho^2 = 4 p_0 p_2 sinh^2(beta mu). $ <eq-weight-hyp>

#figure(
  placement: auto,
  image("doped-gamma/figures/chi_structure.svg", width: 58%),
  caption: [Structure of $chi_"d"$ at $U = 2$, $beta = 3$, $omega = 0$, as moduli of matrix elements in the fermionic indices $n, m$. (a) At half filling the pair term is a mirror, nonzero only for $m = -n - 1$. (b) Away from half filling it becomes the dense Cauchy matrix $C$. (c) The remaining finite-rank part. (d) The full $chi_"d"$; the diagonal dominates.],
) <fig-structure>

== Frequencies and propagator

The fermionic and bosonic Matsubara frequencies are
$ nu_n = (2n + 1) pi / beta, quad quad Omega_ell = (2 ell pi) / beta quad quad (n, ell in ZZ). $ <eq-freq>
The bosonic transfer $omega_k := Omega_k$ is fixed throughout. Write
$ x_n := mu + i nu_n, quad quad w := i omega_k, quad quad J(x) := 1 / (x^2 - a^2). $ <eq-x>
The propagator
$ G(nu) := -integral_0^beta dif tau thin e^(i nu tau) chevron.l T c_up (tau) c_up^dagger chevron.r $
follows from the Lehmann representation: $c_up^dagger$ maps $|0 chevron.r -> |up chevron.r$ (energy change $-a - mu$, weight $p_0 + p_1$) and $|dn chevron.r -> |up dn chevron.r$ (energy change $a - mu$, weight $p_1 + p_2$). So
$ G(nu_n) = (p_0 + p_1) / (x_n + a) + (p_1 + p_2) / (x_n - a) = G(x_n) := (x_n + a rho) / (x_n^2 - a^2), $ <eq-G>
where we use the same letter for the rational function $G(x)$.

== Susceptibilities and the irreducible vertex

Following Ref. @thunstroem2018, the generalized susceptibility is
$ chi_(sigma sigma')(nu, nu', omega) := & integral.triple_0^beta dif tau_1 dif tau_2 dif tau_3 thin e^(-i nu tau_1 + i (nu + omega) tau_2 - i (nu' + omega) tau_3) \
& times [chevron.l T c_sigma^dagger (tau_1) c_sigma (tau_2) c_(sigma')^dagger (tau_3) c_(sigma')(0) chevron.r - chevron.l T c_sigma^dagger (tau_1) c_sigma (tau_2) chevron.r chevron.l T c_(sigma')^dagger (tau_3) c_(sigma')(0) chevron.r], $ <eq-chi-def>
and its particle–particle form is
$ chi_("pp", sigma sigma')(nu, nu', omega) := chi_(sigma sigma')(nu, nu', -omega - nu - nu'). $
The channels and bare susceptibilities are
$ chi_"d" = chi_(up up) + chi_(up dn), quad chi_"m" = chi_(up up) - chi_(up dn), quad chi_"s" = (-chi_("pp", up up) + 2 chi_("pp", up dn) - 2 chi_(0, "pp")) / 4, quad chi_"t" = (chi_("pp", up up) + 2 chi_(0, "pp")) / 4, $
$ chi_(0, r)(nu, nu', omega) = delta_(nu nu') B(nu), quad quad B(nu) = cases(-beta G(nu) G(nu + omega) & quad (r = "d", "m")",", -beta/2 thin G(nu) G(-nu - omega) & quad (r = "s", "t")",") $ <eq-B>
with $chi_(0, "pp") := chi_(0, "s")$. At fixed $omega_k$ all of these are matrices indexed by $(n, m)$ with $nu = nu_n$, $nu' = nu_m$; we write $B_n := B(nu_n)$ and $bold(B) := diag(B_n)$. The channel-irreducible vertex is
$ Gamma_r = beta^2 (chi_r^(-1) - eta_r bold(B)^(-1)), quad quad eta_"d" = eta_"m" = eta_"t" = 1, quad eta_"s" = -1. $ <eq-Gamma-def>

== Lattice and mirror <sec-lattice>

Define the centered lattice and the mirror index
$ z_n := x_n + w/2 = mu + i pi (2n + 1 + k) / beta, quad quad overline(n) := -n - k - 1, \
z_(overline(n)) = 2 mu - z_n, quad quad x_(overline(n)) = 2 mu - w - x_n. $ <eq-z>
The lattice lies on the line $"Re" z = mu$, and the mirror reflects it about the real axis (@fig-plane).

*Conventions.*
- The mirror matrix is $cal(M)_(n m) := delta_(m overline(n))$, with $cal(M)^2 = bb(1)$. A _mirror fixed point_ is an $n$ with $n = overline(n)$; it exists only for odd $k$.
- Products of infinite matrices are formed entry by entry, $(X Y)_(n m) = sum_l X_(n l) Y_(l m)$.
- A sum $sum_n$ is the limit of partial sums over the mirror-symmetric boxes $|2n + 1 + k| <= 2M + 1$, $M -> oo$. We write $PV$ ("principal value") where it converges only in this sense.
- Transposes are ordinary, not Hermitian.
- A function $phi$ on $CC$ is _proper rational_ if it is rational and $phi(t) -> 0$ for $t -> oo$.
- Symbols introduced inside a proof are local to it; all others are listed in @app-symbols.

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
The pair coefficient $c_ell$ stems from the transition $|0 chevron.r <-> |up dn chevron.r$. At $mu = 0$ it is understood as its limit, $c_ell = beta p_0 delta_(ell 0)$ (by @eq-weight-hyp, $rho slash 2 mu -> beta p_0$). The regular part $Rreg$ is a sum of twelve simple fractions, reproduced in @app-R. We take #eqrange(<eq-chi-uu>, <eq-pair>) as given; they follow exactly from the Lehmann representation (@app-checks), and Lean derives them, with @eq-G, from @eq-H (@app-lean).

*Compact form.* Define the *Cauchy matrix* $C_(n m) := c_(n + m + k + 1)$. Since $2 mu + i Omega_(n + m + k + 1) = z_n + z_m$ by @eq-z,
$ C_(n m) = rho / (z_n + z_m) quad (mu != 0), quad quad C_(n m) = beta p_0 delta_(m overline(n)) quad (mu = 0), $ <eq-C>
and $z_n + z_m = 2 mu + i (dots.c) != 0$ for $mu != 0$. Further let
$ Q_(plus.minus)(z) := (z plus.minus a)^2 - w^2/4, quad quad q_0 := a^2 - w^2/4 = (U^2 + omega_k^2) slash 4. $ <eq-Q>
Since $x_n = z_n - w slash 2$,
$ (x_n - a)(x_n + w - a) = Q_(-)(z_n), quad quad (x_n + a)(x_n + w + a) = Q_(+)(z_n), quad quad Q_(plus.minus)(-z) = Q_(minus.plus)(z). $ <eq-Qfact>
Neither $Q_(plus.minus)(z_n)$ vanishes: its factors have imaginary parts $nu_n$ and $nu_n + omega_k$, both nonzero.

#lemma(title: [compact input], lean: "part")[
  With $z = z_n$ and $v = z_m$,
  $ cal(N)_(n m) = frac(4 a^2 C_(n m) - 2 a, Q_(-)(z) Q_(-)(v)) + frac(4 a^2 (1 - rho)(z + v + 2 a)(z v + q_0), Q_(+)(z) Q_(-)(z) Q_(+)(v) Q_(-)(v)). $ <eq-input>
] <lem-input>
#proof[
  Put $y := z + v = x_n + x_m + w$. The two factors of $Pi$ combine to
  $ Pi(x_n, x_m) = (y - 2 a) / ((x_n + w - a)(x_m - a)) dot (y - 2 a) / ((x_m + w - a)(x_n - a)) = frac((y - 2 a)^2, Q_(-)(z) Q_(-)(v)) $
  by @eq-Qfact. For $mu != 0$, $C_(n m) = rho slash y$ and $(y - 2 a)^2 slash y = y - 4 a + 4 a^2 slash y$, so
  $ C_(n m) (y - 2 a)^2 = rho (y - 4 a) + 4 a^2 C_(n m). $
  For $mu = 0$ this holds too, because $rho = 0$ and $C_(n m) != 0$ only where $y = 0$. Thus @eq-input is equivalent to
  $ Rreg (z - w/2, v - w/2) = frac(-2 a - rho (z + v - 4 a), Q_(-)(z) Q_(-)(v)) + frac(4 a^2 (1 - rho)(z + v + 2 a)(z v + q_0), Q_(+)(z) Q_(-)(z) Q_(+)(v) Q_(-)(v)), $
  which is the rational identity (A1) of @app-identities.
]

= Results <sec-results>

All lattice sums that occur below are expressed through
$ S_k (x) := cases(-beta/2 thin tanh(beta (x - mu) slash 2) & quad k "even", -beta/2 thin coth(beta (x - mu) slash 2) & quad k "odd"), $ <eq-S>
which equals $PV sum_n (z_n - x)^(-1)$ for $x$ off the lattice (@lem-S). @app-explicit restates the regular vertex formulas with their coefficients written out.

== Singlet and triplet <sec-res-pp>

The particle–particle channels depend on the fermionic index $n$ only through
$ b := 2 mu - w, quad quad t_n := x_n (b - x_n), $ <eq-bt>
and $t_(overline(n)) = t_n$, because $x_(overline(n)) = b - x_n$ (@eq-z). Define
$ cal(P)(t) &:= (t + a^2)^2 - a^2 b^2, \
cal(G)(t) &:= t + a rho b + a^2 rho^2, \
As &:= a rho b + a^2 (1 - 8 p_1), \
cal(T)_k (t) &:= -frac(S_k (mu + xi), xi), quad xi := sqrt(b^2 slash 4 - t). $ <eq-T>
In terms of these, the bubble is $B_n = -beta cal(G)(t_n) slash (2 cal(P)(t_n))$ (@eq-B-pp), and $cal(T)_k$ is the particle–particle sum (@lem-T),
$ cal(T)_k (t) = sum_n frac(1, t_n - t). $
It is even in $xi$, so either branch of the square root may be used; at $xi = 0$ (even $k$) it takes the limit value $beta^2 slash 4$.

#theorem(title: [triplet], lean: "yes")[
  Assume $t_n + a rho b + a^2 != 0$ for all $n != overline(n)$. Then for $n != overline(n)$
  $ Gamma_("t", n m) = frac(beta a^2 (1 - rho^2) cal(P)(t_n), cal(G)(t_n)(t_n + a rho b + a^2)) (delta_(n m) - delta_(m overline(n))). $ <eq-Gamma-t>
  At a mirror fixed point $n = overline(n)$, $Gamma_("t", n m) = Gamma_("t", m n) = 0$ for all $m$.
] <thm-t>

#theorem(title: [singlet], lean: "yes")[
  Assume $t_n + As != 0$ for all $n$ and
  $ K_"s" := 1 + frac(2 a, beta) cal(T)_k (-As) != 0. $
  Then
  $ Gamma_("s", n m) = frac(beta a^2 (rho^2 - 1 + 8 p_1) cal(P)(t_n), cal(G)(t_n)(t_n + As)) (delta_(n m) + delta_(m overline(n))) + 4 a - frac(4 a thin cal(P)(-As), K_"s" (t_n + As)(t_m + As)). $ <eq-Gamma-s>
] <thm-s>

At $mu = 0$ both formulas reduce to Eq. 19 of Ref. @thunstroem2018 (@app-checks). If $t_c + As = 0$ for some $c$, @eq-Gamma-s has a removable singularity (@prop-special).

== Density and magnetic <sec-res-ph>

Let $epsilon_"d" := +1$ and $epsilon_"m" := -1$. The particle–hole channels are built from the polynomials
$ H(z) &:= z^2 + 2 a rho z + q_1, quad q_1 := q_0 + 4 epsilon_r a^2 p_1, quad lambda := 4 epsilon_r a^2, \
P(z) &:= Q_(+)(z) Q_(-)(z), quad G_2 (z) := (z + a rho)^2 - w^2/4, $ <eq-HP>
and
$ e(z) := H(z) H(-z) - lambda^2 h^2 slash beta^2 = (z^2 + q_1)^2 - 4 a^2 rho^2 z^2 - 16 a^4 p_0 p_2 = (z^2 - s_1)(z^2 - s_2). $ <eq-Delta>
Here $P(z_n)$ times row $n$ of $chi_r$ is $-beta H(z_n) delta_(n m) + lambda C_(n m)$ plus a finite-rank term (@prop-ph); the bubble is $B_n = -beta G_2 (z_n) slash P(z_n)$; and $beta^2 e$ is the determinant of the $2 times 2$ systems in the proof. The roots $s_1, s_2$ of $e$ in the variable $s = z^2$ solve
$ s^2 + 2 (q_1 - 2 a^2 rho^2) s + q_1^2 - 16 a^4 p_0 p_2 = 0, $
and $u_j$ denotes either square root of $s_j$.

*The atomic pair.* For $r = "m"$ and $k = 0$, $q_1 = a^2 (1 - 4 p_1)$, so $H(a) = 4 a^2 p_2$ and $H(-a) = 4 a^2 p_0$ by @eq-weights. Then $e(a) = 0$: one root is $s_1 = a^2$, at the double zeros $plus.minus a$ of $P$. We call this root pair _atomic_.

The temperature enters only through the thermal function
$ alpha_k (x) := rho thin S_k (x) - overline(c), $ <eq-alpha-def>
and the vertex is assembled from
$ F(z) &:= frac(P(z), e(z)), quad g_j (z) := frac(1, z^2 - s_j), \
DD(z, v) &:= frac(z F(z) - v F(v), z^2 - v^2), quad DD(z, z) := frac(F(z) + z F'(z), 2 z), \
K_j &:= beta H(-u_j) + lambda alpha_k (u_j), quad Z_j := frac(u_j P(u_j) [beta H(-u_j) - lambda alpha_k (u_j)], K_j), \
gamma_j &:= frac(lambda rho Z_j, s_j - s_ell) quad (ell != j). $ <eq-coeffs>
$DD$ gives the dense term; $K_j$ is the thermal denominator and $gamma_j$ the weight of the separable term $j$.

#theorem(title: [density and magnetic], lean: "yes")[
  Let $mu != 0$, $e(z_n) != 0$ for all $n$, $s_1 != s_2$, and $K_j != 0$. Then
  $ Gamma_(r, n m) = beta P(z_n) [frac(1, G_2 (z_n)) - frac(H(-z_n), e(z_n))] delta_(n m) + 2 epsilon_r a - lambda rho DD(z_n, z_m) + sum_(j = 1)^2 gamma_j thin g_j (z_n) g_j (z_m). $ <eq-Gamma-ph>
  For $r = "m"$ and $k = 0$ the atomic pair is omitted, both from the sum and from the assumption $K_j != 0$. The formula does not depend on the sign chosen for $u_j$, and the inverse $chi_r^(-1)$ it represents is two-sided.
] <thm-ph>

So $Gamma_"d"$ and $Gamma_"m"$ have the same anatomy as $Gamma_"s"$: a diagonal term, the constant $plus.minus U$, a dense term without thermal factors, and two separable terms with one thermal denominator $K_j$ each. Only $alpha_k (u_j)$, i.e. $tanh$ or $coth$ at the two roots, has to be evaluated. At $s_1 = s_2$ and at lattice zeros of $e$ the formula has removable singularities (@prop-special).

#figure(
  placement: auto,
  image("doped-gamma/figures/gamma_cuts.svg", width: 100%),
  caption: [$Gamma_r (nu, pi slash beta, 0)$ from @eq-Gamma-ph (@eq-Gamma-hf at $mu = 0$), @eq-Gamma-s, @eq-Gamma-t at $U = 2$, $beta = 4$, for $mu = 0$ (half filling, real), $mu = 0.4$ and $mu = 1$. Away from half filling $Gamma_"d"$, $Gamma_"m"$ acquire imaginary parts; all $Gamma_r$ approach their bare values $U$, $-U$, $2U$, $0$ at large $|nu|$.],
) <fig-cuts>

*Half filling.* At $mu = 0$, $rho = 0$ and @thm-ph does not apply literally. Here the Cauchy matrix is the mirror, $C = h cal(M)$ with $h = overline(c) = beta p_0$ (@eq-C), and $H(z) = z^2 + q_1$ is even. Then $e$ factorizes as
$ e(z) = (z^2 - hat(s))(z^2 - hat(s)'), quad quad hat(s) := 4 epsilon_r a^2 (p_0 - p_1) - q_0, quad quad hat(s)' := -4 epsilon_r a^2 (p_0 + p_1) - q_0, $ <eq-hf-defs>
and we define
$ hat(g)(z) := frac(1, z^2 - hat(s)), quad quad hat(D) := beta - frac(2 epsilon_r a S_k (hat(u)), hat(u)), $ <eq-hf-D>
where $hat(u)$ is either square root of $hat(s)$; $hat(D)$ is even in $hat(u)$. Explicitly, for even $k$,
$ hat(D) = beta [1 + frac(epsilon_r a, hat(u)) tanh frac(beta hat(u), 2)], $
with the value $beta (1 + epsilon_r a beta slash 2)$ at $hat(u) = 0$. For odd $k$, $coth$ replaces $tanh$, and $hat(u) = 0$ means $e(0) = 0$ at the mirror fixed point $z_n = 0$ (@prop-special).

#corollary(title: [half filling], lean: "yes")[
  Let $mu = 0$, $e(z_n) != 0$ for all $n$, and $hat(D) != 0$. Then
  $ Gamma_(r, n m) = beta P(z_n) [frac(1, G_2 (z_n)) - frac(H(z_n), e(z_n))] delta_(n m) + 2 epsilon_r a - lambda h F(z_n) delta_(m overline(n)) - frac(2 epsilon_r a beta P(hat(u)), hat(D)) hat(g)(z_n) hat(g)(z_m), $ <eq-Gamma-hf>
  and the inverse $chi_r^(-1)$ it represents is two-sided.
] <cor-hf>

This is Eq. 19 of Ref. @thunstroem2018 for $r = "d", "m"$ (@app-checks). Compared with @eq-Gamma-ph, the dense term has become a mirror term, and only one separable term with one thermal denominator remains. The corollary is the limit $mu -> 0$ of @thm-ph (proof in @app-details). Its assumption $e(z_n) != 0$ matters only where $beta H(z_n) = -lambda h$ with $n != overline(n)$, a genuine divergence (@sec-res-box); at all other zeros of $e(z_n)$ the formula has a removable singularity (@prop-special).

== Removable singularities <sec-res-special>

At isolated parameters a denominator of @eq-Gamma-s, @eq-Gamma-ph or @eq-Gamma-hf vanishes, but the entries of $Gamma_r$ stay finite. Let $O := {c, overline(c)}$ denote a mirror orbit; it has $|O| = 2$ elements, or $|O| = 1$ at a mirror fixed point.

#proposition(title: [removable singularities], lean: "part", breakable: true)[
  In the following cases the entries of $Gamma_r$ are the limits of the closed forms as the parameters approach the given values, and the inverse $chi_r^(-1)$ they represent is two-sided.
  + _Singlet, $t_c + As = 0$ for some $c$._ This happens on one orbit only, and $K_"s"$ diverges. With
    $ T' := sum_(l in.not O) frac(1, t_l + As), quad quad K' := 1 + frac(2 a T', beta), $
    the separable term of @eq-Gamma-s drops out for $n, m in.not O$, and
    $ Gamma_("s", n m) = 4 a - frac(2 beta cal(P)(-As), |O| (t_m + As)) quad (n in O, m in.not O), \
    Gamma_("s", n m) = 4 a + frac(2 beta, |O|) [cal(P)'(-As) - frac(cal(P)(-As), cal(G)(-As))] + frac(beta^2 cal(P)(-As) K', a |O|^2) quad (n, m in O), $ <eq-Gamma-s-lz>
    and symmetrically for $n in.not O$, $m in O$.
  + _Density and magnetic, $s_1 = s_2$_ (the other assumptions of @thm-ph holding). The sum over $j$ in @eq-Gamma-ph is replaced by
    $ lambda rho thin partial_s [frac(Z(s), (z_n^2 - s)(z_m^2 - s))]_(s = s_1), quad quad Z(s) := frac(Z(u) + Z(-u), 2) quad (u^2 = s), $
    where $Z(u)$ is $Z_j$ with $u_j$ replaced by $u$.
  + _Density and magnetic, $e$ vanishing on one orbit $O$_ (at $mu != 0$, the other assumptions of @thm-ph holding). @eq-Gamma-ph holds, with the entries in the rows and columns of $O$ taken as limits as $mu$ approaches its value.
  + _Half filling, $beta H(z_c) = lambda h$_, i.e. $z_c^2 = hat(s)$, with $beta H(z_n) != -lambda h$ for all $n != overline(n)$. Then $hat(u) = z_c$, and $hat(D)$ diverges. With
    $ hat(T)' := sum_(l in.not O) hat(g)(z_l), quad quad hat(D)' := beta - 2 epsilon_r a hat(T)', $
    the separable term of @eq-Gamma-hf drops out for $n, m in.not O$, and
    $ Gamma_(r, n m) = 2 epsilon_r a + frac(beta P(z_c) hat(g)(z_m), |O|) quad (n in O, m in.not O), \
    Gamma_(r, n m) = beta P(z_c) [frac(delta_(n m), G_2 (z_c)) - frac(delta_(n m) - delta_(m overline(n)), 2 (hat(s) - hat(s)'))] + 2 epsilon_r a + frac(beta P(z_c) hat(D)', 2 epsilon_r a |O|^2) - frac(2 beta (hat(s) + q_0 - 2 a^2), |O|) quad (n, m in O), $ <eq-Gamma-hf-lz>
    and symmetrically for $n in.not O$, $m in O$. At a mirror fixed point with $beta H(0) = -lambda h$, @eq-Gamma-hf holds as a limit: its diagonal and mirror terms, which act on the same entry, stay finite together.
] <prop-special>

Item 1 is proved with @thm-s, item 2 in Step 5 of the proof of @thm-ph, and items 3 and 4 in @app-details.

== Finite boxes and divergences <sec-res-box>

#proposition(title: [finite boxes])[
  Under the assumptions of @thm-t, @thm-s, @thm-ph or @cor-hf, or in the cases of @prop-special, the restrictions of $chi_r$ to the boxes $|2n + 1 + k| <= 2M + 1$ are invertible for all sufficiently large $M$, and their inverses converge entrywise to $chi_r^(-1) = Gamma_r slash beta^2 + eta_r bold(B)^(-1)$.
] <prop-box>

*Divergences.* An entry of $Gamma_r$ can diverge only where a denominator in its closed form vanishes; @app-div finds the null vectors and maps the divergences at $omega = 0$. Apart from the removable cases of @prop-special, and away from intersections of these exceptional sets, the remaining zeros are genuine poles under the following nonvanishing conditions:
- $K_j = 0$ with $u_j P(u_j) != 0$ and $s_1 != s_2$ (d, m). Then, as $alpha_k (u_j) != 0$,
  $ gamma_j K_j = frac(2 lambda rho beta u_j P(u_j) H(-u_j), s_j - s_ell) != 0. $
  At $u_j = 0$ the zero of $K_j$ can cancel, as at $hat(u) = 0$ in @cor-hf.
- $hat(D) = 0$ with $P(hat(u)) != 0$ (d, m at $mu = 0$).
- $beta H(z_n) = -lambda h$ with $n != overline(n)$ (d, m at $mu = 0$). There $N(z, dot) slash P$ is even, so by @eq-chi-ph
  $ P(z) chi_r (bold(e)_n - bold(e)_(overline(n))) = -(beta H(z_n) + lambda h)(bold(e)_n - bold(e)_(overline(n))) = 0. $
  So the mirror-odd vector is a null vector. For $r = "d"$, $k = 0$, $n = 0$ this is $beta U = 2 pi slash sqrt(3)$.
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




= Matsubara sums <sec-sums>

#lemma(title: [basic sum], lean: "part")[
  For $x$ off the lattice, with $S_k$ from @eq-S,
  $ PV sum_n frac(1, z_n - x) = S_k (x), quad quad sum_n frac(1, (z_n - x)^j) = frac(S_k^((j - 1))(x), (j - 1)!) quad (j >= 2), $
  where the second sum converges absolutely. Moreover, for all $k$,
  $ S_k (x plus.minus w slash 2) = -beta/2 thin tanh(beta (x - mu) slash 2). $ <eq-shift>
  In particular, $S_k$ takes the value $S_(plus.minus) := plus.minus (beta slash 2) tanh(beta (a plus.minus mu) slash 2)$ at both zeros $minus.plus a plus.minus w slash 2$ of $Q_(plus.minus)$; by @eq-weights,
  $ S_+ = frac(beta (p_1 - p_0), 2 (p_1 + p_0)), quad quad S_- = frac(beta (p_2 - p_1), 2 (p_1 + p_2)). $
] <lem-S>
#proof[
  Since $2n + 1 + k$ is odd for even $k$ and even for odd $k$,
  $ z_n - x = cases(i nu_(n') - (x - mu) & quad k "even," quad n' := n + k slash 2, i Omega_(n'') - (x - mu) & quad k "odd," quad n'' := (2n + 1 + k) slash 2) $
  and the box is symmetric in $n'$ (resp. $n''$). The symmetric Mittag-Leffler expansions
  $ tanh y = sum_j frac(1, y - i pi (j + 1/2)), quad quad coth y = sum_j frac(1, y - i pi j) $
  at $y = beta (x - mu) slash 2$ read
  $ tanh frac(beta (x - mu), 2) = frac(2, beta) sum_j frac(1, x - mu - i nu_j), quad quad coth frac(beta (x - mu), 2) = frac(2, beta) sum_j frac(1, x - mu - i Omega_j), $
  which is @eq-S. For $j >= 2$ the $(j - 1)$-fold $x$-derivative of the series converges absolutely and uniformly on compact sets away from the lattice, so it may be differentiated term by term. For @eq-shift: $w slash 2 = i pi k slash beta$ shifts the argument of $tanh$ or $coth$ by $plus.minus i pi k slash 2$. For even $k$ this is a multiple of the period $i pi$; for odd $k$ use $coth(y plus.minus i pi slash 2) = tanh y$.
]

#lemma(title: [residue rule])[
  Let $phi$ be proper rational without poles on the lattice. Then
  $ sum_n phi(z_n) = sum_(x in "poles"(phi)) Res_(t = x) [phi(t) S_k (t)], $ <eq-res>
  absolutely convergent if $phi = O(t^(-2))$, and as a PV sum if $phi tilde t^(-1)$. ($Res_(t = x)$ of a function is the coefficient of $(t - x)^(-1)$ in its Laurent expansion at $x$.)
] <lem-res>
#proof[
  Partial fractions write $phi$ as a finite sum of constants times $(t - x)^(-j)$, $j >= 1$, over its poles $x$. By @lem-S each $(t - x)^(-j)$ sums to $S_k^((j - 1))(x) slash (j - 1)!$, which is the residue of $(t - x)^(-j) S_k (t)$ at $x$, since $S_k$ is analytic there.
]

#lemma(title: [particle–particle sum], lean: "yes")[
  For $t != t_n$ for all $n$, the sum converges absolutely to $cal(T)_k$ of @eq-T,
  $ sum_n frac(1, t_n - t) = cal(T)_k (t). $
  At the two points $t_(plus.minus) := -a^2 plus.minus a b$, for $b != plus.minus 2 a$,
  $ cal(T)_k (t_+) = -frac(2 S_-, 2 a - b), quad quad cal(T)_k (t_-) = frac(2 S_+, 2 a + b). $ <eq-T-atomic>
  If $b = plus.minus 2 a$ (which needs $k = 0$), then $t_(plus.minus) = b^2 slash 4$, i.e. $xi = 0$, and $cal(T)_k (t_(plus.minus)) = beta^2 slash 4$.
] <lem-T>
#proof[
  Since $x_n = z_n - w slash 2$ and $b slash 2 = mu - w slash 2$, $t_n = b^2 slash 4 - (z_n - mu)^2$. So for $xi != 0$, by @lem-S and the oddness of $tanh$ and $coth$,
  $ sum_n frac(1, t_n - t) = frac(1, 2 xi) sum_n [frac(1, z_n - mu + xi) - frac(1, z_n - mu - xi)] = frac(S_k (mu - xi) - S_k (mu + xi), 2 xi) = -frac(S_k (mu + xi), xi); $
  the sum converges absolutely, as $t_n tilde n^2$. For $xi = 0$, $(t_n - t)^(-1) = -(z_n - mu)^(-2)$ sums to $-S'_k (mu) = beta^2 slash 4$ by @lem-S. This occurs only for even $k$: for odd $k$, $t = b^2 slash 4$ is $t_n$ at the mirror fixed point.

  At $t_(plus.minus)$, $xi^2 = (a minus.plus b slash 2)^2$. At $t_+$ take $xi = a - b slash 2$: then $mu + xi = a + w slash 2$, and $S_k (mu + xi) = S_-$ by @eq-shift. At $t_-$ take $xi = -a - b slash 2$: then $mu + xi = -a + w slash 2$, and $S_k (mu + xi) = S_+$.
]

= Proofs for the singlet and triplet channels <sec-pp>

== Variables

With $t_(plus.minus)$ from @lem-T, $cal(P)(t) = (t - t_+)(t - t_-)$. Define
$ d_(plus.minus) := As + t_(plus.minus), quad quad c := c_(-k) = cases(rho slash b & quad (mu, k) != (0, 0), beta p_0 & quad mu = k = 0). $ <eq-dc>
Then $d_+ d_- = cal(P)(-As)$. For $t = x (b - x)$, direct expansion gives
$ (x - a)(b - x - a) = t - t_+, quad quad (x + a)(b - x + a) = t - t_-, \
(x + a rho)(b - x + a rho) = cal(G)(t), $ <eq-pp-ids>
and the product of the first two is $(x^2 - a^2)((b - x)^2 - a^2) = cal(P)(t)$. Hence, by @eq-B,
$ B_n = -beta/2 thin frac(cal(G)(t_n), cal(P)(t_n)) = B_(overline(n)). $ <eq-B-pp>

*From particle–hole to particle–particle.* The pp entry $(n, m)$ at transfer $k$ is the ph entry at the bosonic index and frequency
$ k' := -k - n - m - 1, quad quad w' := i omega_(k') = b - x_n - x_m. $
Then $delta_(k' 0) = delta_(m overline(n))$, and the pair coefficient becomes $c_(n + m + k' + 1) = c$. In the variables of @lem-input at transfer $w'$, $z + v = x_n + x_m + w' = b$. By @eq-Qfact and @eq-pp-ids,
$ Q_(-)(z) Q_(-)(v) = (x_n - a)(b - x_n - a)(x_m - a)(b - x_m - a) = (t_n - t_+)(t_m - t_+), $ <eq-QQ-pp>
and likewise $Q_(+)(z) Q_(+)(v) = (t_n - t_-)(t_m - t_-)$.

== Structure

A diagonal matrix whose entries depend on $n$ only through $t_n$ commutes with the mirror $cal(M)$. Define
$ cal(M)_(plus.minus) &:= (bb(1) plus.minus cal(M)) slash 2, && quad #gloss[projectors onto mirror-even and mirror-odd vectors] \
L_"t" (t) &:= -beta/2 thin frac(t + a rho b + a^2, cal(P)(t)), quad L_"s" (t) := beta/2 thin frac(t + As, cal(P)(t)), \
g_(plus.minus)(t) &:= frac(1, t - t_(plus.minus)), quad gamma_(plus.minus) := frac(a (rho plus.minus 1)(2 a minus.plus b), 2 b), $ <eq-L>
and $bold(L)_("s", "t") := diag(L_("s", "t")(t_n))$.

#proposition(title: [pp], lean: "yes")[
  $ chi_"t" = bold(B) cal(M)_+ + bold(L)_"t" cal(M)_-, quad quad chi_"s" = bold(L)_"s" cal(M)_+ - bold(B) cal(M)_- + chi_("s", "reg"), $ <eq-chi-pp>
  where
  $ (chi_("s", "reg"))_(n m) = frac(a (2 a c - 1)(t_n - t_-)(t_m - t_-) + a^2 (1 - rho)(b + 2 a)(t_n + t_m + 2 a^2), cal(P)(t_n) cal(P)(t_m)), $ <eq-chi-s-num>
  which for $b != 0$ equals
  $ (chi_("s", "reg"))_(n m) = gamma_+ g_+ (t_n) g_+ (t_m) + gamma_- g_- (t_n) g_- (t_m). $ <eq-chi-s-reg>
] <prop-pp>
#proof[
  Insert $w -> w'$ into @eq-chi-uu and @eq-chi-ud, using
  $ x_n + w' = b - x_m = x_(overline(n)) quad (delta_(n m) "terms"), quad quad x_m = x_(overline(n)), quad x_n + w' = x_n quad (delta_(k' 0) = delta_(m overline(n)) "terms"). $
  With @eq-pp-ids (abbreviating $cal(P) := cal(P)(t_n)$),
  $ chi_("pp", up up) &= 2 B_n delta_(n m) - (beta a^2 (1 - rho^2)) / cal(P) (delta_(n m) - delta_(m overline(n))), \
  chi_("pp", up dn) &= -(4 beta a^2 p_1) / cal(P) delta_(n m) + (4 beta a^2 (p_0 p_2 - p_1^2)) / cal(P) delta_(m overline(n)) + cal(N)|_(w = w'). $

  _Triplet._ By @eq-B-pp,
  $ frac(chi_("pp", up up) + 2 bold(B), 4) = [B_n - frac(beta a^2 (1 - rho^2), 4 cal(P))] delta_(n m) + frac(beta a^2 (1 - rho^2), 4 cal(P)) delta_(m overline(n)) = frac(B_n + L_"t", 2) delta_(n m) + frac(B_n - L_"t", 2) delta_(m overline(n)), $
  with $L_"t" = L_"t" (t_n)$. As $bold(B)$ and $bold(L)_"t"$ commute with $cal(M)$, this is $bold(B) cal(M)_+ + bold(L)_"t" cal(M)_-$.

  _Singlet, delta terms._ The $delta_(n m)$ and $delta_(m overline(n))$ terms of $(-chi_("pp", up up) + 2 chi_("pp", up dn) - 2 bold(B)) slash 4$ are
  $ [-B_n + frac(beta a^2 (1 - rho^2 - 8 p_1), 4 cal(P))] delta_(n m) + frac(beta a^2 [8 (p_0 p_2 - p_1^2) - (1 - rho^2)], 4 cal(P)) delta_(m overline(n)). $
  The first coefficient is $(L_"s" - B_n) slash 2$. By @eq-weight-ids and $p_0 + p_2 = 1 - 2 p_1$, the bracket of the second is $1 - 8 p_1 - rho^2$, so it is $(L_"s" + B_n) slash 2$. Together: $bold(L)_"s" cal(M)_+ - bold(B) cal(M)_-$.

  _Singlet, remaining terms._ They are $cal(N)|_(w = w') slash 2$. In @eq-input at transfer $w'$,
  $ 4 a^2 C_(n m) -> 4 a^2 c, quad quad z + v = b, quad quad z v + q_0 = a^2 + (t_n + t_m) slash 2 quad "(identity (A3))", $
  and the denominators are @eq-QQ-pp. This gives @eq-chi-s-num; for $b != 0$, its partial fractions in $t_n$ and $t_m$ are @eq-chi-s-reg (identity (A3)).
]

== Proofs

#proof(of: [@thm-t])[
  $chi_"t"$ and $bold(B)$ act on each orbit ${n, overline(n)}$ separately. For $n != overline(n)$ the orbit carries a $2 times 2$ block with eigenvalue $B_n$ on the even vector and $L_"t" (t_n)$ on the odd vector; $L_"t" (t_n) != 0$ is the assumption $t_n + a rho b + a^2 != 0$, which must hold on every orbit for $chi_"t"$ to be invertible. So
  $ chi_"t"^(-1) = bold(B)^(-1) cal(M)_+ + bold(L)_"t"^(-1) cal(M)_-, quad quad Gamma_"t" = beta^2 (bold(L)_"t"^(-1) - bold(B)^(-1)) cal(M)_- $
  by @eq-Gamma-def. With $cal(G) - (t + a rho b + a^2) = -a^2 (1 - rho^2)$,
  $ frac(beta^2, 2) (frac(1, L_"t") - frac(1, B)) = -frac(beta cal(P) [cal(G) - (t + a rho b + a^2)], (t + a rho b + a^2) cal(G)) = frac(beta a^2 (1 - rho^2) cal(P), (t + a rho b + a^2) cal(G)), $
  which is @eq-Gamma-t. At a fixed point the orbit is ${n}$ with $cal(M)_- = 0$; then $chi_("t", n n) = B_n$, $Gamma_("t", n n) = 0$, and $L_"t" (t_n)$ does not enter.
]

#proof(of: [@thm-s])[
  _Idea._ $chi_"s"$ is an easily invertible matrix plus the rank-two term @eq-chi-s-reg, so the Woodbury formula applies. Its $2 times 2$ matrix becomes diagonal in a suitable basis.

  For $U = 0$ both sides vanish ($L_"s" = -B$, $chi_("s", "reg") = 0$). Let $U != 0$, and first assume $b != 0$, $b != plus.minus 2 a$ and $d_(plus.minus) != 0$. Then $gamma_(plus.minus) != 0$, since $|rho| < 1$.

  _Woodbury._ Let $bold(g)$ be the $oo times 2$ matrix with rows $(g_+ (t_n), g_- (t_n))$ and $gamma := diag(gamma_+, gamma_-)$, so that
  $ chi_("s", "reg") = bold(g) gamma bold(g)^top, quad quad cal(M)_+ bold(g) = bold(g). $
  The matrix
  $ bold(X) := bold(L)_"s" cal(M)_+ - bold(B) cal(M)_- quad "has the inverse" quad bold(X)^(-1) = bold(L)_"s"^(-1) cal(M)_+ - bold(B)^(-1) cal(M)_- $
  (all products are finite sums). Set
  $ bold(V)_"s" := bold(X)^(-1) bold(g) = bold(L)_"s"^(-1) bold(g), quad quad Psi := bold(g)^top bold(V)_"s" = sum_n frac((g_+, g_-)^top (g_+, g_-), L_"s") (t_n), $
  absolutely convergent (terms $O(t_n^(-1)) = O(n^(-2))$). If $gamma^(-1) + Psi$ is invertible, then
  $ Y := bold(X)^(-1) - bold(V)_"s" (gamma^(-1) + Psi)^(-1) bold(V)_"s"^top $
  satisfies
  $ chi_"s" Y = bb(1) + bold(g) [gamma - (bb(1) + gamma Psi)(gamma^(-1) + Psi)^(-1)] bold(V)_"s"^top = bb(1). $
  $Y$ is symmetric, so also $Y chi_"s" = bb(1)$. By @eq-Gamma-def ($eta_"s" = -1$),
  $ Gamma_"s" = beta^2 (bold(L)_"s"^(-1) + bold(B)^(-1)) cal(M)_+ - beta^2 bold(V)_"s" (gamma^(-1) + Psi)^(-1) bold(V)_"s"^top. $
  The first term gives the first term of @eq-Gamma-s, since $cal(G) - t - As = a^2 (rho^2 - 1 + 8 p_1)$ and
  $ frac(beta^2, 2) (frac(1, L_"s") + frac(1, B)) = beta cal(P) [frac(1, t + As) - frac(1, cal(G))]. $

  _Change of basis._ Let $kappa_n := 1 slash (t_n + As)$. Since $cal(P) = (t - t_+)(t - t_-)$ and $t - t_(minus.plus) = (t + As) - d_(minus.plus)$,
  $ frac(g_(plus.minus)(t_n), L_"s" (t_n)) = 2/beta (1 - d_(minus.plus) kappa_n), quad "i.e." quad (bold(V)_"s")_(n, :)^top = 2/beta thin Theta^top (1, kappa_n)^top, quad Theta := mat(1, 1; -d_-, -d_+). $
  Hence
  $ (bold(V)_"s" (gamma^(-1) + Psi)^(-1) bold(V)_"s"^top)_(n m) = (2/beta)^2 (1, kappa_n) Xi^(-1) (1, kappa_m)^top, quad quad Xi := Theta^(-top) (gamma^(-1) + Psi) Theta^(-1). $
  Insert
  $ Psi = sum_n (2/beta)^2 L_"s" (t_n) Theta^top (1, kappa_n)^top (1, kappa_n) Theta, quad quad (2/beta)^2 L_"s" = 2/beta frac(t + As, cal(P)). $
  Then
  $ Xi = Theta^(-top) gamma^(-1) Theta^(-1) + 2/beta sum_n frac(1, cal(P)(t_n)) mat(t_n + As, 1; 1, 1 slash (t_n + As)). $
  By partial fractions over $t_+, t_-, -As$, the three sums are combinations of $cal(T)_k (t_+)$, $cal(T)_k (t_-)$ and $cal(T)_k (-As)$. Inserting @eq-T-atomic, the off-diagonal entries of the two terms cancel, and (identity (A4))
  $ Xi = diag(-frac(1, a), frac(K_"s", a d_+ d_-)). $
  So $gamma^(-1) + Psi$ is invertible if and only if $K_"s" != 0$, and
  $ -beta^2 (2/beta)^2 (1, kappa_n) Xi^(-1) (1, kappa_m)^top = -4 [-a + frac(a d_+ d_- kappa_n kappa_m, K_"s")] = 4 a - frac(4 a thin cal(P)(-As), K_"s" (t_n + As)(t_m + As)). $

  _Excluded values._ Fix $k$. The values of $mu$ where $b = 0$, $b = plus.minus 2 a$ or $d_(plus.minus) = 0$, and those where $t_n + As = 0$ for some $n$ (@prop-special), are isolated, since these are zeros of nonconstant analytic functions of $mu$. @eq-Gamma-s contains no $b$, $2 a plus.minus b$ or $d_(plus.minus)$ in a denominator, so it is continuous at the first kind.

  _Lattice zeros ($t_n + As -> 0$; item 1 of @prop-special)._ Put $q := t_n + As$, and let $O := {n, overline(n)}$, on which $t_n$ is constant. Then
  $ cal(T)_k (-As) = frac(|O|, q) + T' + O(q), quad quad K_"s" = frac(2 a |O|, beta q) + O(1). $
  Since $a^2 (rho^2 - 1 + 8 p_1) = cal(G)(t_n) - q$ and $cal(G)(t_n) != 0$ (@eq-pp-ids), the first and the last term of @eq-Gamma-s are
  $ frac(beta cal(P)(-As), q) (delta_(n m) + delta_(m overline(n))) + O(1), quad quad -frac(2 beta cal(P)(-As), |O| q) + O(1) quad (m in O), $
  and the last one is $O(1)$ for $m in.not O$. For $m in O$, $delta_(n m) + delta_(m overline(n)) = 2 slash |O|$, so the $1 slash q$ terms cancel. The next order, with $cal(P)(t_n) = cal(P)(-As) + q cal(P)'(-As) + O(q^2)$, gives @eq-Gamma-s-lz.

  _Two-sidedness at the excluded values._ Near all these values, $(chi_"s")_(n l) = O(|t_l|^(-1))$ and $Y_(l m)$ is bounded for $l in.not {n, overline(n), m, overline(m)}$, uniformly, where $Y = Gamma_"s" slash beta^2 - bold(B)^(-1)$ with @eq-Gamma-s. As $|t_l| tilde l^2$, @lem-cont extends $chi_"s" Y = bb(1)$ to these values.
]

= The Cauchy matrix <sec-cauchy>

Throughout this section $mu != 0$.

#lemma(title: [values on the reflected lattice], lean: "part")[
  For every $n$,
  $ S_k (-z_n) = beta/2 coth(beta mu), quad quad S'_k (-z_n) = beta^2 / (4 sinh^2(beta mu)), quad quad rho thin S_k (-z_n) = overline(c). $ <eq-mirror>
] <lem-mirror>
#proof[
  Write $-z_n = x - w slash 2$ with $x := -mu - i pi (2n + 1) slash beta$. By @eq-shift,
  $ S_k (-z_n) = -beta/2 tanh(-beta mu - i pi (2n + 1) slash 2) = beta/2 coth(beta mu), $
  as $tanh$ has period $i pi$ and $tanh(y - i pi slash 2) = coth y$. Differentiating @eq-shift in $x$ gives
  $ S'_k (-z_n) = -beta^2/4 cosh^(-2)(-beta mu - i pi (2n + 1) slash 2) = beta^2 / (4 sinh^2(beta mu)). $
  The last identity is @eq-weight-hyp.
]

So $S_k$ is _constant_ on the reflected lattice (@fig-plane). This is the source of all simplifications below.

#lemma(title: [square], lean: "yes")[
  $sum_l C_(n l) C_(l m) = h^2 delta_(n m)$, absolutely convergent.
] <lem-C2>
#proof[
  $C_(n l) C_(l m) = O(|z_l|^(-2))$. For $n != m$, partial fractions in $z_l$ and @lem-S give
  $ sum_l C_(n l) C_(l m) = frac(rho^2, z_m - z_n) sum_l [frac(1, z_l + z_n) - frac(1, z_l + z_m)] = rho^2 frac(S_k (-z_n) - S_k (-z_m), z_m - z_n) = 0 $
  by @lem-mirror. For $n = m$, by @lem-mirror and @eq-weight-hyp,
  $ sum_l C_(n l) C_(l n) = rho^2 sum_l frac(1, (z_l + z_n)^2) = rho^2 S'_k (-z_n) = h^2. $
]

The action of $C$ on rational functions is governed by the thermal function $alpha_k$ of @eq-alpha-def.

#lemma(title: [thermal function], lean: "yes")[
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

#lemma(title: [Cauchy calculus], lean: "part")[
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
$ zeta_r := beta a^2 [1 - rho^2 + 4 epsilon_r (p_0 p_2 - p_1^2)], quad quad N_2 (v) := -2 epsilon_r a (v^2 + 2 a rho v + q_0) + delta zeta_r (v^2 - a^2), $
$ N_0 (v) := (v^2 + 2 q_0 - 4 a^2) N_2 (v) + (2 epsilon_r a - delta zeta_r) P(v), quad quad N(z, v) := N_0 (v) + z^2 N_2 (v). $ <eq-N>
$P$ is even by @eq-Qfact and has no zeros on the lattice. In $N_0$ the terms of degree four cancel, so $N_0 (v) slash P(v) = O(v^(-1))$ and $N_2 (v) slash P(v) = O(v^(-2))$. We identify a function $phi$ on the lattice with the vector $(phi(z_n))_n$ and write $(chi_r phi)(z_n) := sum_m chi_(r, n m) phi(z_m)$.

#proposition(title: [structure], lean: "yes")[
  $ P(z_n) chi_(r, n m) = -beta H(z_n) delta_(n m) + lambda C_(n m) + frac(N(z_n, z_m), P(z_m)). $ <eq-chi-ph>
] <prop-ph>
#proof[
  Write $z := z_n$, $v := z_m$, and collect the terms of $chi_r = chi_(up up) + epsilon_r chi_(up dn)$ (@eq-chi-uu, @eq-chi-ud).

  _(a) The $delta_(n m)$ terms_ are
  $ -beta [G(x_n) G(x_n + w) + a^2 (1 - rho^2 + 4 epsilon_r p_1) J(x_n) J(x_n + w)]. $
  By @eq-G and @eq-Qfact,
  $ G(x_n) G(x_n + w) = frac((z + a rho)^2 - w^2 slash 4, P(z)), quad quad J(x_n) J(x_n + w) = frac(1, P(z)), $
  and the numerators add up to $(z + a rho)^2 - w^2 slash 4 + a^2 (1 - rho^2) + 4 epsilon_r a^2 p_1 = H(z)$.

  _(b) The $delta_(k 0)$ terms_ occur only for $k = 0$, where $w = 0$, $x_n = z$ and $P(z) = (z^2 - a^2)^2$. They are $zeta_r J(z) J(v)$, and $P(z) J(z) J(v) = (z^2 - a^2)(v^2 - a^2) slash P(v)$.

  _(c) The remaining terms_ are $epsilon_r cal(N)_(n m)$ with @eq-input. Since $C_(n m)(z + v) = rho$ and $Q_(+)(z) - Q_(-)(v) = (z + v)(z - v + 2 a)$,
  $ Q_(+)(z) C_(n m) = Q_(-)(v) C_(n m) + rho (z - v + 2 a). $
  With $1 slash Q_(-)(v) = Q_(+)(v) slash P(v)$ this gives
  $ P(z) cal(N)_(n m) = 4 a^2 C_(n m) + frac(1, P(v)) [& 4 a^2 rho (z - v + 2 a) Q_(+)(v) - 2 a Q_(+)(z) Q_(+)(v) \
  & + 4 a^2 (1 - rho)(z + v + 2 a)(z v + q_0)]. $
  The bracket is quadratic in $z$. Using $v^2 + 2 a v + q_0 = Q_(+)(v)$, its coefficients are
  $ z^1: quad & 4 a^2 Q_(+)(v) [rho - 1 + (1 - rho)] = 0, \
  z^2: quad & -2 a Q_(+)(v) + 4 a^2 (1 - rho) v = -2 a (v^2 + 2 a rho v + q_0), \
  z^0: quad & -2 a (v^2 + 2 q_0 - 4 a^2)(v^2 + 2 a rho v + q_0) + 2 a P(v) quad "(identity (A2))". $
  Times $epsilon_r$, this is the part of @eq-N without $zeta_r$. The term (b) is the rest, since $(v^2 - 2 a^2)(v^2 - a^2) - P(v) = -a^2 (v^2 - a^2)$ at $k = 0$.
]

So, after multiplication by $P$, the finite-rank part of $chi_r$ has only two output directions, $1$ and $z^2$.

== Rank-two sums

#lemma(title: [rank-two sums], lean: "yes")[
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
For $l != n$, $chi_(r, n l) = O(|z_l|^(-2))$ by @eq-chi-uu, @eq-chi-ud and @eq-input, so $sum_l chi_(r, n l) y_v (z_l)$ converges absolutely. Its parts in @eq-column converge at least as PV sums (@lem-calculus, @lem-res), so they may be evaluated separately.

We use @lem-U at $x = -v$, and @lem-calculus and @lem-U at $x = plus.minus u_j$. These points are off the lattice ($mu != 0$, $e(z_n) != 0$), and $P(-v) = P(v) != 0$. We first assume that $P(u_j) != 0$ for the non-atomic pairs; Step 5 removes this assumption.

_Step 1 (the Cauchy column)._ Through $C$, a unit vector at $v$ produces the column $lambda rho slash (z + v)$. Conversely, $C$ maps $1 slash (t + v)$ back to a unit vector: by @lem-C2,
$ cC[(t + v)^(-1)](z) = rho^(-1) sum_l C(z, z_l) C(z_l, v) = (h^2 slash rho) delta_(z v). $
So try $y = D_v delta_(z v) + E_v slash (z + v)$. Insert it into the left side of @eq-column, using $H(z) = H(-v) + (z + v)(z - v + 2 a rho)$, and @eq-U at $x = -v$, where $alpha_k (-v) = 0$ (@lem-mirror). The left side becomes
$ [-beta H(v) D_v + frac(lambda h^2, rho) E_v] delta_(z v) + frac(lambda rho D_v - beta H(-v) E_v, z + v) \
- beta E_v (z - v + 2 a rho) + D_v frac(N(z, v), P(v)) + E_v overline(cal(I))(z; -v). $
We choose $D_v$, $E_v$ such that the first term is $P(v) delta_(z v)$ and the second vanishes:
$ -beta H(v) D_v + frac(lambda h^2, rho) E_v = P(v), quad quad lambda rho D_v - beta H(-v) E_v = 0. $
The determinant of this system is $beta^2 H(v) H(-v) - lambda^2 h^2 = beta^2 e(v) != 0$, and the solution is
$ D_v = -frac(H(-v) P(v), beta e(v)), quad quad E_v = -frac(lambda rho F(v), beta^2). $
With these values the remaining terms are $1 slash (beta e(v))$ times the left side of the identity (A7),
$ lambda rho (z - v + 2 a rho) P(v) - H(-v) N(z, v) - frac(lambda rho, beta) P(v) overline(cal(I))(z; -v) \
= -e(v) N_2 (z) + lambda rho z [P(v) - e(v)] + lambda W(z; v^2). $ <eq-source>
Here $W(z; s) := W_0 (s) + z^2 W_2 (s)$ with polynomials $W_0$, $W_2$ of degree one in $s$, given in @app-identities. Both sides of @eq-source are polynomials in $z$ and $v$ (@lem-U). So the remaining terms are the polynomial
$ frac(1, beta) [-N_2 (z) + lambda rho z (F(v) - 1) + lambda frac(W(z; v^2), e(v))]. $ <eq-defect>

_Step 2 (the constant)._ For $y = 1$ the left side of @eq-column is
$ -beta H(z) + lambda overline(c) + PV sum_l frac(N(z, z_l), P(z_l)) = frac(beta N_2 (z), 2 epsilon_r a), $
by $cC[1] = overline(c)$ (@lem-calculus) and identity (A6). So adding the constant $2 epsilon_r a slash beta^2$ to $y$ removes the first term of @eq-defect.

_Step 3 (pole pairs)._ The rest of @eq-defect must be removed by functions that are mapped to polynomials. A simple pole $1 slash (t - x)$ is mapped by $-beta H + lambda cC$ to poles at $x$ and at $-x$ (@lem-calculus). So consider $phi = f_+ slash (t - x) + f_- slash (t + x)$. For it, the left side of @eq-column is a polynomial plus
$ frac(-beta H(x) f_+ + lambda alpha_k (-x) f_-, z - x) + frac(lambda alpha_k (x) f_+ - beta H(-x) f_-, z + x). $
For $x != 0$, both residues vanish for some $(f_+, f_-) != 0$ if and only if the determinant
$ beta^2 H(x) H(-x) - lambda^2 alpha_k (x) alpha_k (-x) = beta^2 e(x) $
vanishes (@eq-alpha), i.e. if and only if $x = plus.minus u_j$. For $x = u_j$ we take
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
So the last two terms of @eq-defect are
$ frac(1, beta) sum_j lambda rho F_j g_j (v) [z + frac(W(z; s_j), rho P(u_j))]. $
By Steps 1–3 and @eq-mode-image,
$ y_v (z) := D_v delta_(z v) + frac(E_v, z + v) + frac(1, beta^2) [2 epsilon_r a + lambda rho sum_j frac(F_j g_j (v), K_j) phi_j (z)] $
solves @eq-column. To read off $Gamma_r$, note that by @eq-mode
$ frac(lambda rho F_j g_j (v) phi_j (z), K_j) = lambda rho F_j z g_j (z) g_j (v) + gamma_j g_j (z) g_j (v), $
and that $F(z) - F(v) = -(z^2 - v^2) sum_j F_j g_j (z) g_j (v)$ gives
$ DD(z, v) = frac(F(v), z + v) - z sum_j F_j g_j (z) g_j (v). $ <eq-defect-dd>
Hence
$ y_v (z) = -frac(H(-z) P(z), beta e(z)) delta_(z v) + frac(1, beta^2) [2 epsilon_r a - lambda rho DD(z, v) + sum_j gamma_j g_j (z) g_j (v)]. $
By @eq-G and @eq-Qfact, $B_n = -beta G_2 (z_n) slash P(z_n)$; so by @eq-Gamma-def, $beta^2 [y_(z_m)(z_n) - delta_(n m) slash B_n]$ is the right side of @eq-Gamma-ph.

_Step 5 (two-sidedness, branches, special cases)._
- _Two-sidedness._ Let $Y_(n m) := y_(z_m)(z_n)$, so that $chi_r Y = bb(1)$. $Y$ is symmetric because $DD$ is, and so is $chi_r$ (@eq-chi-uu, @eq-chi-ud, @eq-input). Hence, with the same absolutely convergent series,
  $ (Y chi_r)_(n m) = sum_l chi_(r, m l) Y_(l n) = (chi_r Y)_(m n) = delta_(n m). $
- _Branches._ Let $K(u) := beta H(-u) + lambda alpha_k (u)$. Since $alpha_k (u) alpha_k (-u) = h^2$ (@eq-alpha), for every $u$
  $ frac(beta H(-u) - lambda alpha_k (u), K(u)) + frac(beta H(u) - lambda alpha_k (-u), K(-u)) = frac(2 beta^2 e(u), K(u) K(-u)). $ <eq-branch>
  At a root the right side vanishes, so $Z_j$ and $gamma_j$ do not depend on the sign of $u_j$.
- _Confluent roots (item 2 of @prop-special)._ By @eq-branch, the odd part
  $ frac(Z(u) - Z(-u), 2) = frac(beta^2 u P(u) e(u), K(u) K(-u)) $
  vanishes at the roots, so $Z(s_j) = Z_j$ for the even part $Z(s)$. Hence the sum over $j$ in @eq-Gamma-ph is $lambda rho$ times the divided difference of $s |-> Z(s) slash ((z_n^2 - s)(z_m^2 - s))$ at $s_1$, $s_2$. Near a double root $u_star^2$, $Z(u)$ is analytic in $u$ at $plus.minus u_star$: $K(u_star) != 0$, and at a root $K(u) = 0$ if and only if $K(-u) = 0$ (@eq-alpha, @eq-Delta). So $Z(s)$, an even analytic function of $u$, is analytic in $s$, also for $u_star = 0$. As $mu$ approaches a value where $s_1 = s_2$ (a zero of the discriminant of $e$, a nonconstant analytic function of $mu$), the divided difference converges to the derivative, and @lem-cont extends $chi_r Y = bb(1)$.
- _The atomic pair._ For $r = "m"$ and $k = 0$ the atomic pair has $F_1 = P(a) slash (a^2 - s_2) = 0$ and $W(z; a^2) = 0$ (@app-identities). So its terms in Step 4 vanish, and Step 3 is needed only for the non-atomic pair.
- _Excluded cases._ Steps 1–4 assumed $P(u_j) != 0$ for the non-atomic pairs; they hold for $u_j = 0$ as well. For $r = "m"$, $k = 0$, $P(u_2) = 0$ would mean $s_2 = a^2 = s_1$, which is excluded. Otherwise, fix $k$: the values of $mu$ where the assumption fails are zeros of the resultant of $e$ and $P$, a nonconstant analytic function of $mu$, so such values are isolated. @eq-Gamma-ph has no $P(u_j)$ in a denominator and depends on $s_j$ only. Near such a value, $chi_(r, n l) = O(|z_l|^(-2))$ for $l != n$ and $Y_(l m)$ is bounded, uniformly; so @lem-cont extends $chi_r Y = bb(1)$.
- _Lattice roots._ The case $e(z_n) = 0$ (item 3 of @prop-special) is treated in @app-details. $square$

*Acknowledgment.* The derivations, proofs and verification scripts were developed with the help of AI assistants (Anthropic's Claude and OpenAI's GPT models). Every identity is checked by computer algebra (@app-identities) and every result numerically (@app-checks); @app-lean lists what is formalized in Lean.

#bibliography("doped-gamma-refs.yml", style: "american-physics-society")

#set heading(numbering: "A.1", supplement: [Appendix])
#counter(heading).update(0)

= Rational identities <app-identities>

Each identity is checked exactly with SymPy: (A1)–(A5) by `doped-gamma/verify_identities.py`, (A6) and (A7) by `doped-gamma/verify_dm.py` (both channels, $k != 0$ and $k = 0$, all parameters free). Lean proves them too (@tab-lean).

/ (A1): the twelve terms of @app-R equal the two-term form in the proof of @lem-input; and $Pi(x_n, x_m) = (z + v - 2 a)^2 slash (Q_(-)(z) Q_(-)(v))$.
/ (A2): the $z^0$ coefficient in the proof of @prop-ph:
  $ & 4 a^2 rho (2 a - v) Q_(+)(v) - 2 a q_0 Q_(+)(v) + 4 a^2 (1 - rho) q_0 (v + 2 a) \
  & = -2 a (v^2 + 2 q_0 - 4 a^2)(v^2 + 2 a rho v + q_0) + 2 a P(v). $
/ (A3): @eq-pp-ids; $z v + q_0 = a^2 + (t_n + t_m) slash 2$ at transfer $w' = b - x_n - x_m$; @eq-chi-s-num and @eq-chi-s-reg from the twelve terms of @app-R at $w'$ with $c = rho slash b$.
/ (A4): $Xi = diag(-1 slash a, K_"s" slash (a d_+ d_-))$ in the proof of @thm-s, and $d_+ d_- = cal(P)(-As)$.
/ (A5): the $delta_(n m)$ and $delta_(k 0)$ identities in the proof of @prop-ph, and $H(a) = 4 a^2 p_2$, $H(-a) = 4 a^2 p_0$ for $r = "m"$, $k = 0$.
/ (A6): Step 2 of the proof of @thm-ph, the sum evaluated by @lem-res and @lem-U:
  $ -beta H(z) + lambda overline(c) + PV sum_l frac(N(z, z_l), P(z_l)) = frac(beta N_2 (z), 2 epsilon_r a). $
/ (A7): @eq-source, where $overline(cal(I))$ is evaluated by @lem-U and, with $pi.alt := 4 (p_0 p_2 - p_1^2)$,
  $ W_0 (s) &:= epsilon_r a q_0 [pi.alt - 2 epsilon_r (1 - rho^2)] (s + q_0 + 2 epsilon_r a^2) + delta beta a^4 psi(s), \
  W_2 (s) &:= epsilon_r a pi.alt (s + q_0 + 2 epsilon_r a^2) - delta beta a^2 psi(s), \
  psi(s) &:= -4 epsilon_r p_1 (p_0 + p_2 - rho^2) [s + (1 + 2 epsilon_r) a^2] + (1 + epsilon_r) rho^2 pi.alt (s - a^2). $
  For $r = "m"$ and $k = 0$ ($epsilon_"m" = -1$, $q_0 = a^2$) this shows $W_0 (a^2) = W_2 (a^2) = 0$.

= Analytic details <app-details>

#lemma(title: [continuity])[
  Let $chi(p)$ and $Y(p)$ be matrices indexed by the lattice, depending on a real parameter $p$ ($mu$ or $beta$) near $p_0$, with entries continuous at $p_0$. Suppose that for all $n$, $m$ there are $c_(n m)$ and a finite set of indices $l$ outside of which, for all these $p$,
  $ |chi(p)_(n l) Y(p)_(l m)| <= frac(c_(n m), 1 + l^2). $ If $chi(p) Y(p) = bb(1)$ for $p != p_0$, then also $chi(p_0) Y(p_0) = bb(1)$.
] <lem-cont>
#proof[
  For fixed $n$, $m$, the finitely many exceptional terms of $sum_l chi_(n l) Y_(l m)$ are continuous at $p_0$, and the sum of the others is continuous by dominated convergence.
]

*Lattice zeros at $mu != 0$ (item 3 of @prop-special).* Since $e$ has real coefficients and $z_(overline(n)) = z_n^*$, $e(z_(overline(n))) = e(z_n)^*$: lattice zeros of $e$ come in mirror orbits $O = {n, overline(n)}$. Let $e$ vanish on one orbit $O$ at $mu = mu_0 != 0$, the other assumptions of @thm-ph holding at $mu_0$. As $e(z_n)$ is a nonconstant analytic function of $mu$, @thm-ph holds for $mu != mu_0$ near $mu_0$. We show that the entries of its inverse $Y$ (Step 4) converge as $mu -> mu_0$, boundedly off the diagonal; then @lem-cont gives $chi_r Y = bb(1)$ at $mu_0$.
- _Roots near the lattice._ Each root $u_j$ that tends to a point $z_c$, $c in O$, has $(z_c - u_j) alpha_j -> rho != 0$ and $K_j -> oo$, since $(z_c - u) S_k (u) -> 1$ as $u -> z_c$ (@lem-S).
- _Columns off the orbit._ In the column $y_v$ of Step 4 with $v = z_m$, $m in.not O$, the coefficients $D_v$, $E_v$ and $F_j g_j (v)$ stay bounded. By @eq-mode, for $c in O$,
  $ frac(phi_j (z_c), K_j) = frac(beta H(-u_j) slash (z_c - u_j) + lambda alpha_j slash (z_c + u_j), beta H(-u_j) + lambda alpha_j) -> frac(beta H(-z_c), lambda rho) + frac(1, 2 z_c) quad (u_j -> z_c); $
  it tends to $1 slash (z_c + z_(c'))$ if $u_j -> z_(c')$ with $c' != c$, and it stays bounded otherwise. So $Y_(c m) = y_(z_m)(z_c) = Y_(m c)$ converges for $c in O$, $m in.not O$, boundedly in $m$.
- _The block on the orbit._ Row $l in.not O$ of $chi_r Y = bb(1)$ gives
  $ sum_(c in O) chi_(r, l c) Y_(c c') = -sum_(l' in.not O) chi_(r, l l') Y_(l' c'), $
  whose right side converges by dominated convergence. By @eq-chi-ph, the columns
  $ P(z_l) chi_(r, l c) = frac(lambda rho, z_l + z_c) + frac(N(z_l, z_c), P(z_c)), quad c in O, $
  are linearly independent rational functions of $z_l$, with distinct poles at $-z_c$. So $|O|$ rows $l in.not O$ can be chosen for which the matrix $(chi_(r, l c)(mu_0))$ is invertible, and the block converges.

*Lattice zeros at $mu = 0$ (item 4 of @prop-special).* Let $beta H(z_n) = lambda h$ with $n != overline(n)$ at $beta = beta_0$, so that $z_n^2 = hat(s)$ and
$ e(z) = (z^2 - hat(s))(z^2 - hat(s) + 2 lambda h slash beta). $
As $beta -> beta_0$, $hat(u) -> z_n$. The diagonal and mirror terms of @eq-Gamma-hf contribute
$ -frac(lambda h P(z_n), e(z_n)) + O(1) = -frac(beta P(z_n), 2 (z_n^2 - hat(s))) + O(1) $
to each of the entries $(n, n)$ and $(n, overline(n))$. Since $S_k (hat(u)) = 1 slash (z_n - hat(u)) + O(1)$,
$ hat(D) = frac(2 epsilon_r a, hat(u)(hat(u) - z_n)) + O(1), $
so the separable term contributes $beta P(z_n) slash (2 (z_n^2 - hat(s))) + O(1)$ to the same entries and $O(1)$ to the others. The singular parts cancel.

At a mirror fixed point $n = overline(n)$ (odd $k$, $z_n = 0$), let $e(0) = 0$ at $beta = beta_0$. The diagonal and mirror terms of @eq-Gamma-hf act on the same entry $(n, n)$. As $beta^2 e(0) = (beta H(0) - lambda h)(beta H(0) + lambda h)$, they combine to
$ frac(beta P(0), G_2 (0)) - frac(beta^2 P(0), beta H(0) - lambda h). $
- If $beta H(0) = -lambda h$ at $beta_0$, this stays finite, and under the remaining assumptions of @cor-hf nothing else is singular.
- If $beta H(0) = lambda h$, then $hat(s) -> 0$, and the combination is $beta P(0) slash hat(s) + O(1)$, as $beta H(0) - lambda h = -beta hat(s)$. For odd $k$, $coth(beta hat(u) slash 2) slash hat(u) = 2 slash (beta hat(s)) + O(1)$, so $hat(D) = 2 epsilon_r a slash hat(s) + O(1)$, and the separable term is
  $ -frac(2 epsilon_r a beta P(hat(u)), hat(D)) hat(g)(0)^2 = -frac(beta P(0), hat(s)) + O(1) $
  at $(n, n)$ and $O(1)$ elsewhere. The singular parts cancel again.
In all these cases @eq-Gamma-hf extends continuously, and @lem-cont (with $p = beta$) gives $chi_r Y = bb(1)$ at $beta_0$.

#proof(of: [@cor-hf])[
  Take $mu -> 0$ at fixed $k$ in @eq-Gamma-ph. Then $rho -> 0$, $overline(c) -> h$ and $C_(n m) -> h delta_(m overline(n))$ (@eq-C). One root $s_j$ of $e$ tends to $hat(s)$, the other, $s_ell$, to $hat(s)'$. At a root $u_j$ put $X_(plus.minus) := beta H(plus.minus u_j)$. Then, by @eq-Delta,
  $ X_+ X_- = lambda^2 h^2, quad quad X_+ - X_- = 4 a beta rho u_j, $
  and with $overline(c)^2 = h^2 + beta^2 rho^2 slash 4$ (@eq-weight-ids) this gives
  $ (X_- - lambda overline(c))(X_- + lambda overline(c)) = -rho (4 a beta u_j X_- + lambda^2 beta^2 rho slash 4). $
  By @eq-alpha-def,
  $ K_j = X_- - lambda overline(c) + lambda rho S_k (u_j), quad quad X_- - lambda alpha_k (u_j) = X_- + lambda overline(c) - lambda rho S_k (u_j). $
  - _The pair $j$._ Here $X_(plus.minus) -> lambda h$, so the identity gives $(X_- - lambda overline(c)) slash rho -> -2 a beta hat(u)$. Hence
    $ frac(K_j, rho) -> lambda S_k (hat(u)) - 2 a beta hat(u) = -2 a hat(u) hat(D), quad quad X_- - lambda alpha_k (u_j) -> 2 lambda h, quad quad s_j - s_ell -> frac(2 lambda h, beta), $
    so $gamma_j -> -2 epsilon_r a beta P(hat(u)) slash hat(D)$ if $hat(u) != 0$.
  - _The pair $ell$._ Here $X_(plus.minus) -> -lambda h$, so $K_ell -> -2 lambda h$, $X_- - lambda alpha_k (u_ell) -> 0$ and $gamma_ell -> 0$. (For $r = "m"$, $k = 0$ this is the atomic pair, absent anyway.)
  - _The dense term._ By @eq-defect-dd, since $z_(overline(n)) = -z_n$ at $mu = 0$ and $F$ is even,
    $ lambda rho DD(z_n, z_m) = lambda C_(n m) F(z_m) - lambda rho z_n sum_j F_j g_j (z_n) g_j (z_m) -> lambda h F(z_n) delta_(m overline(n)). $
  So @eq-Gamma-ph tends entrywise to @eq-Gamma-hf.

  It remains to show that the limit is an inverse. For small $mu != 0$ the assumptions of @thm-ph hold: $e(z_n) != 0$ by continuity (automatically for large $|n|$), $s_1 != s_2$, $K_ell != 0$, and $K_j != 0$ because $K_j slash rho -> -2 a hat(u) hat(D) != 0$ (for $hat(u) != 0$). The entries of $C$ are bounded uniformly near $mu = 0$ (by $beta rho slash (2 pi)$ for $m != overline(n)$, and $rho slash (2 mu) -> beta p_0$). So $chi_(r, n l) = O(|z_l|^(-2))$ for $l != n$, and the entries of $Y = Gamma_r slash beta^2 + bold(B)^(-1)$, with $Gamma_r$ from @eq-Gamma-ph, are bounded off the diagonal, uniformly near $mu = 0$. So @lem-cont gives $chi_r Y = bb(1)$ at $mu = 0$; $Y$ stays symmetric. Finally, for even $k$, $hat(u) = 0$ occurs at isolated $beta$, where @eq-Gamma-hf is continuous in $beta$ (as $hat(D) != 0$); @lem-cont with $p = beta$ covers it. For odd $k$, $hat(u) = 0$ means $e(0) = 0$ at the mirror fixed point, treated above.
]

#proof(of: [@prop-box])[
  Rescale $chi_r$ to an operator on $ell^2$,
  $ hat(chi) := bold(theta.alt) chi_r bold(theta.alt), quad quad bold(theta.alt) := diag(theta.alt_n), quad quad theta.alt_n := (1 + |z_n - mu|^2)^(1 slash 2). $
  - _$hat(chi) = kappa_r bb(1) + hat(chi)_c$ with $hat(chi)_c$ compact_, where $kappa_"d" = kappa_"m" = beta$ and $kappa_"s" = -kappa_"t" = beta slash 2$.
    - $r = "d", "m"$: by @eq-chi-ph, $chi_(r, n n) = -beta H(z_n) slash P(z_n) + O(z_n^(-4))$, so $theta.alt_n^2 chi_(r, n n) -> beta$. Off the diagonal, $chi_(r, n m) = O(|z_n|^(-2) |z_m|^(-2))$ by @eq-chi-uu, @eq-chi-ud and @eq-input. So these entries of $hat(chi)$ are square-summable, i.e. Hilbert–Schmidt.
    - $r = "s", "t"$: $t_n slash theta.alt_n^2 -> 1$, so by @eq-B-pp
      $ theta.alt_n^2 B_n -> -beta/2, quad quad theta.alt_n^2 L_"s" (t_n) -> beta/2, quad quad theta.alt_n^2 L_"t" (t_n) -> -beta/2. $
      By @eq-chi-pp, $hat(chi) - kappa_r bb(1)$ is a matrix supported on the diagonal and the mirror with entries tending to zero, which is compact, plus (for $r = "s"$) $bold(theta.alt) chi_("s", "reg") bold(theta.alt)$, which is Hilbert–Schmidt by @eq-chi-s-num.
  - _$hat(chi)$ is invertible._ For each $m$, $hat(chi)$ maps $bold(theta.alt)^(-1) chi_r^(-1) bold(e)_m$ to $theta.alt_m bold(e)_m$: the $n$-th entry of the image is $theta.alt_n sum_l chi_(r, n l) (chi_r^(-1))_(l m) = theta.alt_n delta_(n m)$, an absolutely convergent series. The vector $bold(theta.alt)^(-1) chi_r^(-1) bold(e)_m$ lies in $ell^2$, because by the explicit formulas $(chi_r^(-1))_(n m)$ is bounded in $n$ for $n in.not {m, overline(m)}$. So the range of $hat(chi)$ contains all basis vectors; being closed (a compact perturbation of $kappa_r bb(1)$ has closed range), it is all of $ell^2$. A compact perturbation of $kappa_r bb(1)$ that is onto is also injective (Fredholm alternative), hence invertible.
  - _Convergence._ Let $bb(1)_M$ be the projection onto the box. Then
    $ hat(chi)_M := bb(1)_M hat(chi) bb(1)_M + kappa_r (bb(1) - bb(1)_M) $
    converges to $hat(chi)$ in norm, since $(bb(1) - bb(1)_M) hat(chi)_c -> 0$ and $hat(chi)_c (bb(1) - bb(1)_M) -> 0$ for compact $hat(chi)_c$. So $hat(chi)_M$ is invertible for all sufficiently large $M$, and $hat(chi)_M^(-1) -> hat(chi)^(-1)$ in norm. On the box, $hat(chi)_M^(-1) = bold(theta.alt)^(-1) (bb(1)_M chi_r bb(1)_M)^(-1) bold(theta.alt)^(-1)$.
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
- *Expanded formulas* (`doped-gamma/check_explicit.jl`): @app-explicit, in the notation of Ref. @thunstroem2018, is transcribed without the main text's polynomial and thermal coefficients and compared with `DopedHubbardAtom` at 256 bits, including diagonal entries, mirror fixed points, both signs of each particle–hole square root and the static magnetic atomic root.
- *Field sectors* (`doped-gamma/check_field.jl`): the displayed field formulas of @sec-field-explicit against `FieldHubbardAtom` at 256 bits; divergence conditions, growing susceptibility boxes, null-vector overlaps and branch cancellations (@sec-field-div-checks).
- *Divergences* (`doped-gamma/check_divergences.jl`):
  - On 30 cuts at fixed $beta mu in [0.05, 3.9]$, $|beta U| <= 25$, each of the 70 lines of @eq-div-ph and @eq-div-s matches a change by one of the number of negative real eigenvalues of $chi_"d"$, $chi_"m"$, $chi_"s"$ on the box $|2 n + 1| <= 49$ at an unchanged number of conjugate pairs. The scan steps $beta U$ by $0.05$ and matches within $0.15$. There are no other such changes, and none by more than one. Changes of the number of conjugate pairs (pairs turning real, pseudo-divergences) are not compared.
  - At 8 points on the lines (including $mu = 0$), the box eigenvector of the eigenvalue closest to zero has overlap $|cos angle(bold(v), bold(g))|$ with the $bold(g)$ of @app-div that differs from 1 by less than $2 times 10^(-12)$.
  - At the three exceptional points quoted in @app-div, on the boxes $|2 n + 1| <= 121$ and $241$, exactly two eigenvalues have eigenvectors aligned with $bold(g)_j$ (overlap above $0.99$). They agree in size within $10 %$, are below a tenth of all other eigenvalues, and shrink by more than half from the smaller box to the larger ($10^(-8)$ on the larger). An ordinary divergence and a point off the lines fail this test.
  - On the singlet curve $K_"s" = 0 = d_minus.plus$ at $beta mu = plus.minus 0.3, plus.minus 1, plus.minus 3$, the box eigenvalue whose eigenvector is most aligned with $1 slash (t_n + As)$ stays above $0.02$, and $Gamma_"s"$ (`DopedHubbardAtom`) converges to the same finite value from both sides of the curve.
  - The conditions of @sec-div-thun, transcribed with $B^2 = -s_j$, vanish on six of the lines to $10^(-13)$, and its sum for the exceptional points, summed directly over $nu$, vanishes at two of them to $10^(-10)$ of the sum of the moduli.
  - A point where two arcs of $Phi(s_j) = 0$ cross (equality in the sign condition of @eq-div-ph) is a divergence: the smallest box eigenvalue shrinks from $10^(-10)$ to $3 times 10^(-14)$.
- *Implementation:* the test suite of `HubbardAtoms.jl` compares `DopedHubbardAtom` with exact diagonalization, with finite-box inversion and with `BigFloat` evaluation, including the removable singularities of @prop-special.


= Formalization in Lean <app-lean>

The Lean 4 project `lean/`, next to this note in the repository of HubbardAtoms.jl, builds on Mathlib. It derives the input of @sec-input from the Hamiltonian, and formalizes @thm-t, @thm-s, @thm-ph, @cor-hf and most of @prop-special, together with the statements their proofs use. @tab-lean lists each numbered statement with what is formalized and where. The names are declarations in the namespace `HubbardAtom.Params`, the auxiliary rational identities in `HubbardAtom`; the versions of Lean and Mathlib are pinned in the project. Its script `scripts/check.sh` builds it from scratch, fails on any warning (an unfinished proof, `sorry`, is one), and checks that every declaration depends only on Lean's three standard axioms.

*Taken as given.* The theorems are statements about definitions, which the reader has to compare with this note:
- the model of @sec-setup: @eq-H on the four states, with $c_up$, $c_dn$ as $4 times 4$ matrices (Lean checks their anticommutation relations), $O(tau) = e^(tau cal(H)) O e^(-tau cal(H))$, the thermal average $"Tr"(e^(-beta cal(H)) X) slash "Tr" e^(-beta cal(H))$, the fermionic time ordering, and the integrals defining $G$ and $chi_(sigma sigma')$; the frequencies and the lattice of @sec-lattice. From these, Lean derives the weights, @eq-G and the input #eqrange(<eq-chi-uu>, <eq-pair>) with @app-R, for all $U$, $mu$ and $n$, $m$, $k$;
- the particle–particle form, the channel combinations, @eq-B and @eq-Gamma-def. A right inverse $Y$ of $chi$ means $sum_l chi_(n l) Y_(l m) = delta_(n m)$ for all $n$, $m$, with unconditionally, i.e. absolutely, convergent sums; likewise for left inverses;
- the closed forms @eq-Gamma-t, @eq-Gamma-s, @eq-Gamma-s-lz, @eq-Gamma-ph with its limit in item 2 of @prop-special (where $partial_s Z$ is written out and shown to be the $s$-derivative of $Z(s)$), @eq-Gamma-hf (in a form combining the diagonal and mirror terms, equal to @eq-Gamma-hf where $e(z_n) != 0$) and @eq-Gamma-hf-lz, with $DD(z, z)$ through the derivative of $F$, and the assumptions, as typed in Lean.
The inverses constructed in the proofs need no trust: Lean proves that they are inverses from both sides.

#let lean-row(stmt, what, names) = (stmt, what, text(size: 7.5pt, names))
#figure(
  kind: table,
  caption: [What the Lean project proves, statement by statement (files in `lean/HubbardAtomLean/`).],
  text(size: 8.5pt, {
  set par(justify: false)
  table(
    columns: (19%, 39%, 42%),
    align: left,
    stroke: none,
    fill: (_, y) => if calc.even(y) and y > 0 { luma(242) },
    inset: (x: 4pt, y: 3.5pt),
    table.hline(),
    table.header([*Statement*], [*In Lean*], [*Declarations*]),
    table.hline(stroke: 0.4pt),
    ..lean-row([Input, #eqrange(<eq-chi-uu>, <eq-pair>), @eq-G, @app-R], [derived from @eq-H for all $U$, $mu$, $n$, $m$, $k$], [`propagator_eq`, `chiLehmann_upUp`, `chiLehmann_upDn` (`LehmannInput.lean`)]),
    ..lean-row([@lem-input (compact input)], [for $mu != 0$], [`compact_input` (`PHInput.lean`)]),
    ..lean-row([@thm-t (triplet)], [as stated; moreover every right or left inverse of $chi_"t"$ gives @eq-Gamma-t, so $Gamma_"t"$ is unique], [`exists_inverse_chiT`, `gammaT_of_isRightInverse`, `gammaT_of_isLeftInverse`, `gammaT_eq_zero_of_fixed` (`Triplet.lean`)]),
    ..lean-row([@thm-s (singlet)], [as stated, for all $b$ (including $b = 0, plus.minus 2 a$, $d_(plus.minus) = 0$, $xi = 0$): an inverse of $chi_"s"$ from both sides, and @eq-Gamma-s; for $U = 0$, $Gamma_"s" = 0$ for every inverse], [`SingletHyp.chiS_isRightInverse`, `SingletHyp.chiS_isLeftInverse`, `SingletHyp.gammaS_eq` (`SingletZero.lean`); `gammaS_zero_of_a_eq_zero` (`UZero.lean`)]),
    ..lean-row([@thm-ph (density and magnetic)], [as stated: an inverse of $chi_r$ from both sides and @eq-Gamma-ph, including $DD(z, z)$, the case $r = "m"$, $k = 0$, and the independence of the sign of $u_j$; for $U = 0$, $Gamma_r = 0$ for every inverse], [`DMHyp.toDMGeneric`, `DMGeneric.chiR_isRightInverse`, `DMGeneric.chiR_isLeftInverse`, `DMGeneric.gammaR_eq`, `DMGeneric.gammaR_neg₁`, `dmGeneric_static_magnetic` (`DensityMagnetic.lean`, `DMLifts.lean`); `gammaR_zero_of_a_eq_zero` (`UZero.lean`)]),
    ..lean-row([@cor-hf (half filling)], [for $U != 0$, directly at $mu = 0$ rather than as a limit, including $hat(u) = 0$, assuming instead of $e(z_n) != 0$ only $z_n^2 != hat(s)$ for all $n$ and $z_n^2 != hat(s)'$ off the mirror fixed points; so it also holds at a fixed point with $beta H(0) = -lambda h$], [`HFHyp.of_eph`, `HFHyp.chiR_isRightInverse`, `HFHyp.chiR_isLeftInverse`, `HFHyp.gammaHF_eq`, `gammaHF_eq_note` (`HalfFilling.lean`)]),
    ..lean-row([@prop-special (removable singularities)], [items 1 and 4: the values @eq-Gamma-s-lz (for all $b$) and @eq-Gamma-hf-lz (including $hat(s) = 0$ at a fixed point) give inverses from both sides; that they are the limits is not formalized. Item 2: as stated, including $s_1 = s_2 = 0$. Item 3: not formalized], [item 1: `SingletLZ.chiS_isRightInverse`, `SingletLZ.chiS_isLeftInverse`, `SingletLZ.gammaS_eq` (`SingletLZ.lean`); item 2: `DMConfHyp.toDMConf`, `DMConf.chiR_isRightInverse`, `DMConf.chiR_isLeftInverse`, `DMConf.gammaR_eq`, `DMConf.tendsto_Zeven` (`Confluent.lean`); item 4: `HFLZ.chiR_isRightInverse`, `HFLZ.chiR_isLeftInverse`, `HFLZ.gammaHF_eq` (`HalfFillingLZ.lean`)]),
    ..lean-row([@prop-box (finite boxes)], [not formalized], []),
    ..lean-row([@lem-S (basic sum)], [for $j = 1$ along the boxes $|2 n + 1 + k| <= 2 M + 1$, and for $j = 2$, from Mathlib's expansion of the cotangent; @eq-shift and $S_(plus.minus)$], [`hasSum_S_even`, `hasSum_S_odd`, `hasSum_sq` (`Matsubara.lean`); `S_shift`, `S₀_neg_a`, `S₀_a` (`Thermal.lean`); `S_shift'` (`PHSums.lean`)]),
    ..lean-row([@lem-res (residue rule)], [not as such: the sums it serves, with the denominators $P(t)$ and $P(t)(t - x)$, are evaluated by splitting $P = Q_+ Q_-$, for all $k$ at once], [`hasSum_Q_sub`, `hasSum_inv_Q`, `hasSum_cubic` (`PHSums.lean`)]),
    ..lean-row([@lem-T (particle–particle sum)], [as stated, including $xi = 0$], [`hasSum_Tpp`, `hasSum_one_div_t_sub` (`Matsubara.lean`); `hasSum_gp`, `hasSum_gm`, `Tpp_tp`, `Tpp_tm` (`Singlet.lean`)]),
    ..lean-row([@prop-pp (pp)], [as stated, with $chi_("s", "reg")$ in the form @eq-chi-s-reg for $b != 0$, and as the partial fractions of @eq-chi-s-num for $b = 0$], [`chiT_eq_mirrorBlock` (`Triplet.lean`); `chiS_eq` (`SingletInput.lean`); `chiS_eq_zero` (`SingletZero.lean`)]),
    ..lean-row([@lem-mirror (values on the reflected lattice)], [the last identity; the value of $S'_k$ enters @lem-C2], [`rho_mul_S_neg_z` (`PHSums.lean`)]),
    ..lean-row([@lem-C2 (square)], [as stated], [`hasSum_square` (`PHSums.lean`)]),
    ..lean-row([@lem-alpha (thermal function)], [as stated], [`αeven_eq`, `αodd_eq` (`Thermal.lean`); `α_mul_α_neg` (`PHSums.lean`)]),
    ..lean-row([@lem-calculus (Cauchy calculus)], [for simple poles; the constant rule only together with the sum of @lem-U. Higher poles are not needed], [`hasSum_cauchy`, `hasSum_const` (`PHSums.lean`)]),
    ..lean-row([@prop-ph (structure)], [as stated, for $mu != 0$ and at $mu = 0$, where $C = h cal(M)$], [`Pph_mul_chiR` (`PHInput.lean`); `Pph_mul_chiR_hf` (`HalfFilling.lean`)]),
    ..lean-row([@lem-U (rank-two sums)], [as stated, with the residues at the zeros of $P$ written out], [`hasSum_rank_two`, `Rhat` (`PHSums.lean`)]),
    ..lean-row([Proof of @thm-ph], [Steps 1–4 with @eq-defect-dd; of Step 5 the two-sidedness, @eq-branch and the atomic pair], [`DMGeneric.hasSum_step1`, `DMGeneric.hasSum_step2`, `DMGeneric.hasSum_step3`, `step4_identity`, `DMGeneric.DD_eq`, `chiR_symm`, `DMGeneric.invR_symm`, `atomic_root` (`DensityMagnetic.lean`); `Zph_sub_Zph_neg` (`Confluent.lean`)]),
    ..lean-row([@lem-cont (continuity)], [not formalized; Lean does without it (see below)], []),
    ..lean-row([(A1)–(A7) (@app-identities)], [all; (A2) and the first part of (A5) within `Pph_mul_chiR`, and the consequences $e(a) = 0$, $W_0 (a^2) = W_2 (a^2) = 0$ for $r = "m"$, $k = 0$ in `atomic_root`], [(A1) `compact_input`, (A3) `regular_pp`, (A4) `singlet_A4`, (A6) `identity_A6`, (A7) `identity_A7`]),
    table.hline(),
  )}),
) <tab-lean>

*Proofs.* For the input, Lean splits the integral over $[0, beta]^3$ into the six orderings of the times, sums each ordering over the paths through the four states, and evaluates the integrals over the ordered simplex in closed form, also where a partial sum of the rates vanishes: at $k = 0$, at $n = m$, and at $mu = 0$, $n + m + k + 1 = 0$ (`chiLehmann_eq`, `orderedTerm_eq`, `pathTerm_eq`). Two arguments differ from this note. Lattice sums with the denominator $P$ are evaluated by splitting $P = Q_+ Q_-$ instead of by @lem-res, which needs no separate treatment of the double zeros at $k = 0$. And the inverse of $chi_"s"$ is verified directly; the cancellation of the Woodbury step is `woodbury_identity`. Where this note takes limits, Lean argues directly. @cor-hf is proved at $mu = 0$. At the lattice zeros of items 1 and 4 of @prop-special the limiting inverses are verified with lattice sums from which the orbit is removed (`SingletLZ.hasSum_g_kappa`, `HFLZ.hasSum_Nph_ghat`). For $s_1 = s_2$, Step 3 holds for every $u$ up to a remainder proportional to $e(u)$ (`step3_general`), which vanishes to second order at a double root; the image of the $u$-derivative of the mode then follows by differentiating the lattice sums termwise (`hasSum_deriv_of_bound`, from Mathlib's `Complex.hasSum_deriv_of_summable_norm`), and at $s = 0$ from difference quotients by Tannery's theorem (`DMConf.hasSum_M₁_zero`).

*With doping and field* (@sec-field-explicit; declarations in the namespace `HubbardAtom.FParams`, files `lean/HubbardAtomLean/Field*.lean`). Lean defines the atom with field, its four-point functions and the sectors $"T" sigma$, $"A" sigma$, $"E" sigma$ and $"L"$ from the Hamiltonian as above, and proves the general Shiba map ($c_dn -> c_dn^dagger$ as a $4 times 4$ matrix that sends $cal(H)(U, mu, h)$ to $cal(H)(-U, h, mu)$) and the spin flip, with the relations @eq-field-A of the susceptibilities and bubbles. It derives the transverse input (the analog of @prop-ph with the complex transfer $W$, `PT_mul_chiTUp`), the equal-spin pair $chi_("E" sigma)$ (`chiEUp_eq`) and the longitudinal input with its pair and spin-flip kernels (`PL_mul_chiL_updn`, `PL_mul_chiL_upup`) from the Hamiltonian. It proves @eq-field-T with a two-sided inverse (`TGeneric.gammaTUp_eq`) for $mu != 0$, $W != 0$, including roots at a zero of $P$ (the inverse carries $P(u_j)$ as a factor) and $W = plus.minus 2 a$ ($k = 0$, $|h| = |U| slash 2$), where $P(t) = t^2 (t^2 - 4 a^2)$ has a double zero and, as $q_1^2 = lambda^2 p_0 p_2$ there, $e(0) = 0$: one root is $u = 0$, with $P(0) = K(0) = 0$ for $mu != 0$, which drops out like the atomic pair because $W(z; 0) = 0$ (`q₀T_zero_root`, `TGeneric.of_q₀T_zero`); at $h = 0$, $k = 0$, where $W = 0$, it derives $chi_("T" up) = chi_"m"$ of the zero-field atom from the Hamiltonian, so that @thm-ph and @cor-hf give the vertex there (`gammaTUp_static`, `gammaTUp_static_hf`); @eq-field-T-axis at $mu = 0$ (`THFHyp.gammaTHF_eq`, `gammaTHF_eq_note`), assuming in addition $W != 0$; @eq-field-A for both orientations under the same assumptions, also at $h = 0$ (`gammaAUp_eq`, `gammaADn_eq`, `gammaAUp_eq_hf`, and `gammaAUp_static` at $mu = 0$, $k = 0$); and @eq-field-E with the inverses on $cal(M)_-$, assuming $cal(L)_sigma (t_(sigma nu)) != 0$ at all $nu != -nu - omega$ (`gammaEUp_eq`, `gammaEDn_eq`). Not formalized with field: the longitudinal vertex @eq-field-L, the exceptional sets of @sec-field-limits, the axis reductions @eq-field-h0, @eq-field-mu0 as identities between closed forms, and @sec-field-div.

*Not formalized.* @app-div; @prop-box, @lem-cont and the rest of @app-details: item 3 of @prop-special (the lattice zeros of $e$ for $mu != 0$), and that the values in items 1 and 4 are the limits of @eq-Gamma-s and @eq-Gamma-hf; that the inverses of $chi_"s"$, $chi_"d"$ and $chi_"m"$ are unique (for $chi_"t"$ it is proved, and for $U = 0$ in all channels); and floating-point evaluation.

#pagebreak(weak: true)
= The vertices written out <app-explicit>

This appendix gives the regular formulas of @thm-t, @thm-s, @thm-ph and @cor-hf without the nested coefficient definitions used in the proofs, in the notation of Ref. @thunstroem2018. The frequencies are written directly: $Gamma_r^(nu nu' omega) := Gamma_r (nu, nu', omega)$ in the channel convention of @eq-Gamma-def, which is that of Ref. @thunstroem2018. At $mu = 0$ the formulas are Eq. (19) of that reference. Besides the input panel, each channel defines a few constants, the counterparts of $A_r$ and $B_r$ in Ref. @thunstroem2018; every polynomial and thermal coefficient is written out inside the vertex. @sec-div-thun relates this notation to that of the main text. These are exact identities, not numerically stabilized evaluation rules.

== Input panel

Take $U, mu in RR$, $beta > 0$, and the Matsubara frequencies $nu = (2 n + 1) pi slash beta$, $nu' = (2 m + 1) pi slash beta$, $omega = 2 pi k slash beta$. For $U = 0$, all four vertices vanish; let $U != 0$. The normalized weights and the filling parameter are
$ p_0 := frac(e^(-beta mu), 2 cosh(beta mu) + 2 e^(beta U slash 2)), quad p_1 := frac(e^(beta U slash 2), 2 cosh(beta mu) + 2 e^(beta U slash 2)), quad p_2 := frac(e^(beta mu), 2 cosh(beta mu) + 2 e^(beta U slash 2)), $
$ rho := frac(sinh(beta mu), cosh(beta mu) + e^(beta U slash 2)) = p_2 - p_0. $
The mirror partner of $nu$ is $-nu - omega$, and $delta_(nu nu')$, $delta_(nu (-nu' - omega))$ are Kronecker deltas. Away from half filling the frequencies enter shifted by $mu$: in the particle–hole channels through $tilde(nu) := nu - i mu$ and $tilde(nu)' := nu' - i mu$, in the particle–particle channels through
$ t_nu := (nu - i mu)(nu + omega + i mu), $
which is $nu (nu + omega)$ at $mu = 0$. The particle–hole channels $r = "d", "m"$ carry the sign $epsilon_"d" = +1$ (density) and $epsilon_"m" = -1$ (magnetic). As in Ref. @thunstroem2018, the temperature enters through $tan(beta (W + omega) slash 4)$ for a square root $W$ (with $omega$ shifted by $2 i mu$ in the particle–hole channels); since $beta omega slash 4 = pi k slash 2$, this is $tan(beta W slash 4)$ for even $k$ and $-cot(beta W slash 4)$ for odd $k$.

== Triplet

With
$ C := frac(U, 2) rho (2 mu - i omega) + frac(U^2, 4) rho^2, quad quad A_"t"^2 := -frac(U^2, 4) - frac(U, 2) rho (2 mu - i omega), $
for $nu != -nu - omega$,
$ Gamma_"t"^(nu nu' omega) = -frac(beta (A_"t"^2 + C) [(nu - i mu)^2 + U^2 slash 4] [(nu + omega + i mu)^2 + U^2 slash 4], (t_nu + C)(t_nu - A_"t"^2)) [delta_(nu nu') - delta_(nu (-nu' - omega))], $ <eq-explicit-t>
where $-(A_"t"^2 + C) = (U^2 slash 4)(1 - rho^2)$. The factor $t_nu - A_"t"^2$ must be nonzero for every $nu != -nu - omega$. At a mirror fixed point, $nu = -omega slash 2$ (odd $k$), the entire row and column vanish: $Gamma_"t"^(nu nu' omega) = Gamma_"t"^(nu' nu omega) = 0$ for every $nu'$. At $mu = 0$, $C = 0$ and $A_"t"^2 = -U^2 slash 4$, and this is the triplet case of Eq. (19) of Ref. @thunstroem2018.

#block(breakable: false)[
== Singlet

With $C$ as for the triplet,
$ B_"s"^2 := frac(U^2, 4)(8 p_1 - 1) - frac(U, 2) rho (2 mu - i omega), quad quad W_"s" := sqrt(4 B_"s"^2 + (omega + 2 i mu)^2), $
the vertex is
$ Gamma_"s"^(nu nu' omega) = & frac(beta (B_"s"^2 + C) [(nu - i mu)^2 + U^2 slash 4] [(nu + omega + i mu)^2 + U^2 slash 4], (t_nu + C)(t_nu - B_"s"^2)) [delta_(nu nu') + delta_(nu (-nu' - omega))] + 2 U \
& - frac(2 U [(B_"s"^2 + U^2 slash 4)^2 - (U^2 slash 4)(2 mu - i omega)^2], [1 + U tan(beta (W_"s" + omega) slash 4) slash W_"s"] (t_nu - B_"s"^2)(t_(nu') - B_"s"^2)). $ <eq-explicit-s>
The factors $t_nu - B_"s"^2$ must be nonzero for every $nu$, and so must the thermal denominator. Either sign of $W_"s"$ gives the same value; at $W_"s" = 0$ (even $k$), replace $tan(beta (W_"s" + omega) slash 4) slash W_"s"$ by $beta slash 4$. At $mu = 0$, $B_"s"$ is the $B_"s"$ of Ref. @thunstroem2018, and this is the singlet case of its Eq. (19), where the term with $A_"s" = 0$ drops out. The lattice-zero continuation is item 1 of @prop-special.
]

#block(breakable: false)[
== Density and magnetic away from half filling

Take $mu != 0$. With
$ q := frac(U^2, 4)(1 + 4 epsilon_r p_1), quad quad B_(r, 1)^2, B_(r, 2)^2 := q - frac(U^2 rho^2, 2) minus.plus U sqrt(frac(U^2, 4) rho^4 - rho^2 (q + frac(omega^2, 4)) + U^2 p_0 p_2), $
$ W_j := sqrt(4 B_(r, j)^2 + omega^2), quad quad tilde(t)_nu := tilde(nu) (tilde(nu) + omega), $
and for $B_(r, 1)^2 != B_(r, 2)^2$ and $tilde(t)_nu != B_(r, j)^2$ for every $nu$ and $j$,
$ Gamma_r^(nu nu' omega) = & -beta (tilde(nu)^2 + frac(U^2, 4)) ((tilde(nu) + omega)^2 + frac(U^2, 4)) \
& quad times [frac(1, tilde(t)_nu - i U rho (tilde(nu) + omega slash 2) - U^2 rho^2 slash 4) + frac(q - tilde(t)_nu - i U rho (tilde(nu) + omega slash 2), (tilde(t)_nu - B_(r, 1)^2)(tilde(t)_nu - B_(r, 2)^2))] delta_(nu nu') + epsilon_r U \
& - frac(i epsilon_r U^2 rho, tilde(t)_(nu') - tilde(t)_nu) [frac((tilde(nu) + omega slash 2) (tilde(nu)^2 + U^2 slash 4) ((tilde(nu) + omega)^2 + U^2 slash 4), (tilde(t)_nu - B_(r, 1)^2)(tilde(t)_nu - B_(r, 2)^2)) - frac((tilde(nu)' + omega slash 2) (tilde(nu)'^2 + U^2 slash 4) ((tilde(nu)' + omega)^2 + U^2 slash 4), (tilde(t)_(nu') - B_(r, 1)^2)(tilde(t)_(nu') - B_(r, 2)^2))] \
& + frac(i epsilon_r U^2 rho, 2) sum_(j = 1)^2 lr([frac(W_j [(B_(r, j)^2 + U^2 slash 4)^2 + U^2 omega^2 slash 4], (B_(r, 3 - j)^2 - B_(r, j)^2)(tilde(t)_nu - B_(r, j)^2)(tilde(t)_(nu') - B_(r, j)^2)) \
& quad times frac(q - B_(r, j)^2 - (i slash 2) U rho W_j + (epsilon_r U^2 slash 2) [i rho tan(beta (W_j + omega + 2 i mu) slash 4) + p_0 + p_2], q - B_(r, j)^2 - (i slash 2) U rho W_j - (epsilon_r U^2 slash 2) [i rho tan(beta (W_j + omega + 2 i mu) slash 4) + p_0 + p_2])]). $ <eq-explicit-ph>
The last denominator must be nonzero for each retained root. For $r = "m"$ and $k = 0$, omit the root $B_(r, j)^2 = -U^2 slash 4$ from the sum and from this requirement. Either sign of either $W_j$ gives the same vertex. As $mu -> 0$, $B_(r, 1)^2$ and $B_(r, 2)^2$ tend to $A_r^2$ and $B_r^2$ below, in an order that depends on the signs of $U$ and $epsilon_r$.
]

#block(breakable: false)[
*Diagonal entries.* For $nu = nu'$, replace the second line (the dense term) by
$ i epsilon_r U^2 rho [frac(N, 2 y E) + y frac(2 (tilde(t)_nu + U^2 slash 4) E - N (2 tilde(t)_nu - B_(r, 1)^2 - B_(r, 2)^2), E^2)], $ <eq-explicit-diag>
where $y := tilde(nu) + omega slash 2$, $E := (tilde(t)_nu - B_(r, 1)^2)(tilde(t)_nu - B_(r, 2)^2)$ and $N := (tilde(t)_nu + U^2 slash 4)^2 + U^2 omega^2 slash 4 = (tilde(nu)^2 + U^2 slash 4)((tilde(nu) + omega)^2 + U^2 slash 4)$. The confluent-root and lattice-zero continuations are items 2 and 3 of @prop-special.
]

== Density and magnetic at half filling

Set $mu = 0$, so that $rho = 0$ and $tilde(nu) = nu$. With
$ A_r^2 := frac(U^2, 4) (1 + 2 epsilon_r), quad quad B_r^2 := frac(U^2, 4) frac(3 e^(epsilon_r beta U slash 2) - 1, e^(epsilon_r beta U slash 2) + 1), quad quad W := sqrt(4 B_r^2 + omega^2), $
and for $nu (nu + omega) != A_r^2, B_r^2$ for every $nu$,
$ Gamma_r^(nu nu' omega) = & frac(beta A_r^2 (nu^2 + U^2 slash 4)((nu + omega)^2 + U^2 slash 4), 2 [nu (nu + omega) - A_r^2] nu (nu + omega)) [delta_(nu nu') - delta_(nu (-nu' - omega))] \
& + frac(beta B_r^2 (nu^2 + U^2 slash 4)((nu + omega)^2 + U^2 slash 4), 2 [nu (nu + omega) - B_r^2] nu (nu + omega)) [delta_(nu nu') + delta_(nu (-nu' - omega))] \
& + epsilon_r U - frac(epsilon_r U [(B_r^2 + U^2 slash 4)^2 + U^2 omega^2 slash 4], [1 + epsilon_r U tan(beta (W + omega) slash 4) slash W] [nu (nu + omega) - B_r^2] [nu' (nu' + omega) - B_r^2]). $ <eq-explicit-hf>
This is Eq. (19) of Ref. @thunstroem2018 for $r = "d", "m"$, whose prefactors there are $cal(A)_0^r = cal(B)_0^r = 1$, $|cal(B)_2^r| = 1$ and $(cal(B)_1^r)^2 = -epsilon_r$. $A_r$ and $B_r$ are those of its Table I, where $B_"m"$ carries a minus sign that does not matter; its thermal denominator $U tan(dots.c) slash W plus.minus 1$ is ours up to the sign $epsilon_r$. The thermal denominator must be nonzero. Either sign of $W$ gives the same value; at $W = 0$ (even $k$), replace $tan(beta (W + omega) slash 4) slash W$ by $beta slash 4$. The lattice-zero and mirror-fixed-point continuations are item 4 of @prop-special.

== Doping and Zeeman field <sec-field-explicit>

Here $h$ denotes the physical field, *not* the thermal constant $h$ of @eq-rho; the latter is written $kappa := beta sqrt(p_0 p_2)$ throughout the field parts of Appendices F–H. The Hamiltonian is
$ cal(H) = U(n_up - 1/2)(n_dn - 1/2) - mu(n_up + n_dn - 1) - h(n_up - n_dn). $
These formulas follow by the same inversion; the transverse sector is the magnetic-channel problem with $i omega$ replaced by the complex transfer $i omega - 2h$ and with the thermal data below. No additional derivation is given here; Lean carries out the inversion for @eq-field-T, @eq-field-T-axis, @eq-field-A and @eq-field-E (@app-lean).

=== Field input and sector conventions <sec-field-input>

Take real $U, mu, h$, $beta > 0$, $a = U slash 2$, $w = i omega$, and the frequencies of the input panel. All vertices vanish at $U = 0$. Replace the zero-field weights by
$ Z_f := 2 cosh(beta mu) + 2 e^(beta U slash 2) cosh(beta h), $
$ (p_0,p_up,p_dn,p_2) := frac((e^(-beta mu),e^(beta (U slash 2+h)),e^(beta (U slash 2-h)),e^(beta mu)), Z_f), $
$ rho := p_2-p_0, quad m_f := p_up-p_dn, quad mu_sigma := mu+sigma h, quad r_sigma := rho-sigma m_f, quad sigma,t in {+1,-1} equiv {up,dn}. $ <eq-field-weights>
Thus $r_sigma = 2 chevron.l n_(-sigma) chevron.r-1$, and
$ G_sigma (nu) = frac(i nu+mu_sigma+a r_sigma,(i nu+mu_sigma)^2-a^2). $

The longitudinal sector is the *coupled* connected matrix $chi_(sigma t)$ of @eq-chi-def in spin and frequency. The transverse $chi_("T" sigma)$ is the full transform with the same Fourier factor of $chevron.l T c_sigma^dagger (tau_1)c_(-sigma) (tau_2)c_(-sigma)^dagger (tau_3)c_sigma (0) chevron.r$. The oriented opposite-spin pair is $chi_("A" sigma)^(nu nu' omega) = -chi_("T" sigma)^(nu nu' (-omega-nu-nu'))$; it is not a separate singlet/triplet scalar problem at $h != 0$. The equal-spin pair $chi_("E" sigma)$ is $-1 slash 2$ times the *full*, not connected, same-spin particle–hole four-point function at transfer $-omega-nu-nu'$. The factor $1 slash 2$ removes identical-leg double counting.

All field sectors use $Gamma_X = beta^2 (chi_X^(-1)-b_X^(-1))$ ($eta_X=+1$). Their bubbles are
$ b_("L",sigma t)^(nu nu' omega) &= -beta G_sigma (nu)G_sigma (nu+omega)delta_(sigma t)delta_(nu nu'), \
 b_("T" sigma)^(nu nu' omega) &= -beta G_sigma (nu)G_(-sigma) (nu+omega)delta_(nu nu'), \
 b_("A" sigma)^(nu nu' omega) &= beta G_sigma (nu)G_(-sigma) (-nu-omega)delta_(nu nu'), \
 b_("E" sigma)^(nu nu' omega) &= frac(beta,2)G_sigma (nu)G_sigma (-nu-omega) [delta_(nu nu')-delta_(nu (-nu'-omega))]. $ <eq-field-bubbles>
Only the longitudinal inverse couples both spins. For $"E" sigma$ both inverses are restricted to the mirror-odd subspace $cal(M)_-$, then extended by zero; there is no full-space inverse.

To keep the formulas legible, write the physical frequencies through
$ z_c (nu) := c+i(nu+omega slash 2)=i(nu-i c+omega slash 2), quad
 S_c (x) := -frac(beta,2) cases(tanh(beta(x-c)slash 2) & k " even", coth(beta(x-c)slash 2) & k " odd"). $ <eq-field-centered>
These are explicit abbreviations, not further unknowns. For comparison with the constants of Ref. @thunstroem2018, a centered root $s_j$ below corresponds to $B_j^2=-s_j-omega^2 slash 4$, so $z_c (nu)^2-s_j=B_j^2-(nu-i c)(nu-i c+omega)$.

=== Transverse spin and opposite-spin pairs <sec-field-T>

For $"T" up$ put $z=z_mu (nu)$, $v=z_mu (nu')$, and define
$ W:=i omega-2h, quad q_1:=a^2-W^2 slash 4-2a^2 (p_up+p_dn)-a W m_f, $
$ H_T (x):=x^2+2a rho x+q_1, quad
 P_T (x):=[(x+a)^2-W^2 slash 4][(x-a)^2-W^2 slash 4], $
$ e_T (x):=(x^2+q_1)^2-4a^2 rho^2 x^2-16a^4 p_0 p_2=(x^2-s_1)(x^2-s_2), $
$ s_(1,2):=2a^2 rho^2-q_1 plus.minus sqrt(4a^4 rho^4-4a^2 rho^2 q_1+16a^4 p_0 p_2), quad u_j^2=s_j. $ <eq-field-T-data>
For $mu != 0$, distinct roots off the lattice, and nonzero thermal denominators in the following sum,
$ Gamma_("T" up)^(nu nu' omega) = & beta P_T (z) lr([frac(1,(z+a rho)^2-(W slash 2+a m_f)^2)-frac(z^2-2a rho z+q_1,e_T (z))])delta_(nu nu')-U \
 & +4a^2 rho frac(z P_T (z)slash e_T (z)-v P_T (v)slash e_T (v),z^2-v^2) \
 & -4a^2 rho sum_(j=1)^2 frac(u_j P_T (u_j), (s_j-s_(3-j))(z^2-s_j)(v^2-s_j)) \
 & quad times frac(beta(s_j-2a rho u_j+q_1)+4a^2 [rho S_mu (u_j)-beta(p_0+p_2)slash 2], beta(s_j-2a rho u_j+q_1)-4a^2 [rho S_mu (u_j)-beta(p_0+p_2)slash 2]). $ <eq-field-T>
On $nu=nu'$ the quotient in the second line is $[F_T (z)+z F_T'(z)]slash(2z)$, where $F_T=P_T/e_T$ and the prime is differentiation in $z$. Either sign of each $u_j$ gives the same result. At $h=0$, this is @eq-explicit-ph for $r="m"$; at $h=0$, $k=0$ omit the atomic root $s_j=a^2$, whose coefficient vanishes. Likewise, at $W=plus.minus 2a$ ($k=0$, $|h|=|U| slash 2$), where $P_T (x)=x^2 (x^2-4a^2)$, one root is $s_j=0$, with $P_T (0)=0$ and, for $mu != 0$, a vanishing thermal denominator; omit it as well.

At $mu=0$ use the mirror limit instead. With $rho=0$, $p_0=p_2$, $hat(s)=-4a^2 p_0-q_1$, $hat(u)^2=hat(s)$, and $hat(D)=beta+2a S_0 (hat(u))slash hat(u)$,
$ Gamma_("T" up)^(nu nu' omega) = & beta P_T (z) lr([frac(1,z^2-(W slash 2+a m_f)^2)-frac(z^2+q_1,e_T (z))])delta_(nu nu')-U \
 & +4a^2 beta p_0 frac(P_T (z),e_T (z))delta_(nu (-nu'-omega)) \
 & +frac(2a beta P_T (hat(u)),hat(D)(z^2-hat(s))(v^2-hat(s))). $ <eq-field-T-axis>
Here $e_T (z_n)!=0$ and $hat(D)!=0$ are required for the regular formula. At $hat(u)=0$ and even $k$, $S_0 (hat(u))slash hat(u)=-beta^2 slash 4$ by continuity.

The other orientation and the opposite-spin pair are
$ Gamma_("T" dn) (U,mu,h)=Gamma_("T" up) (U,mu,-h), quad
 Gamma_("A" up) (U,mu,h)=Gamma_("T" up) (-U,h,mu). $ <eq-field-A>
All frequencies and $beta$ are unchanged. Explicitly the second substitution sends $a -> -a$, $mu <-> h$, $(p_0,p_up,p_dn,p_2)->(p_dn,p_2,p_0,p_up)$, $rho <-> m_f$, $W -> i omega-2mu$, and the bare vertex to $+U$. Thus @eq-field-T, or @eq-field-T-axis on $h=0$, writes out the pair vertex too. The down orientation is $Gamma_("A" dn) (U,mu,h)=Gamma_("A" up) (U,mu,-h)=cal(M)Gamma_("A" up) (U,mu,h)cal(M)$. Apply spin interchange *before* the particle–hole map: $Gamma_("A" dn)=Gamma_("T" up) (-U,-h,mu)$, not $Gamma_("T" dn) (-U,h,mu)$ at unchanged frequencies.

=== Equal-spin pairs <sec-field-E>

Set $b_sigma=2mu_sigma-i omega$, $x=mu_sigma+i nu$, $t_(sigma nu)=x(b_sigma-x)$, and define
$ cal(P)_sigma (t):=(t+a^2)^2-a^2 b_sigma^2, quad
 cal(G)_sigma (t):=t+a r_sigma b_sigma+a^2 r_sigma^2, quad
 cal(L)_sigma (t):=t+a r_sigma b_sigma+a^2. $ <eq-field-E-data>
For $nu != -nu-omega$ and $cal(L)_sigma (t_(sigma nu))!=0$,
$ Gamma_("E" sigma)^(nu nu' omega)=-frac(beta a^2 (1-r_sigma^2) [(t_(sigma nu)+a^2)^2-a^2 b_sigma^2],2 [t_(sigma nu)+a r_sigma b_sigma+a^2 r_sigma^2][t_(sigma nu)+a r_sigma b_sigma+a^2]) [delta_(nu nu')-delta_(nu (-nu'-omega))]. $ <eq-field-E>
At a mirror fixed point the entire row and column vanish. This is $-1 slash 2$ times @eq-explicit-t with $mu -> mu_sigma$, $rho -> r_sigma$.

=== Coupled longitudinal vertex <sec-field-L>

Here $z=z_(mu_sigma) (nu)$ and $v=z_(mu_t) (nu')$ belong to their own spin lattices. Define
$ lambda_L:=U^2, quad q_0:=a^2+omega^2 slash 4, quad
 pi_P:=p_0 p_2, quad pi_S:=p_up p_dn, quad d_pi:=pi_P-pi_S, $
$ D_t:=p_2 p_t-p_0 p_(-t), quad c_t:=p_2 p_t+p_0 p_(-t), $
$ H_t (x):=x^2+2a r_t x+q_0, quad P_L (x):=(x^2+q_0)^2-4a^2 x^2, $
$ Q_t (x):=H_t (-x)H_(-t) (x)H_(-t) (-x)-lambda_L^2 [pi_P H_(-t) (-x)+pi_S H_(-t) (x)], $
$ E_8 (x):= & H_up (x)H_up (-x)H_dn (x)H_dn (-x) \
 & -lambda_L^2 lr({pi_S [H_up (x)H_dn (x)+H_up (-x)H_dn (-x)] \
 & quad +pi_P [H_up (x)H_dn (-x)+H_up (-x)H_dn (x)]}) +lambda_L^4 d_pi^2 \
 = & product_(j=1)^4 (x^2-s_j). $ <eq-field-L-data>
The $s_j$ now mean the four longitudinal roots, not the two transverse ones. Put $u_j^2=s_j$ and $E'_(8,j):=product_(i!=j) (s_j-s_i)$ (derivative in $s=x^2$, not in $x$). All mode coefficients are explicit:
$ alpha_t (x):=D_t S_(mu_t) (x)-beta c_t slash 2, $
$ alpha_t^P (x):=rho S_(mu_(-t)) (x)-beta(p_0+p_2)slash 2, quad
 alpha_t^S (x):=-t m_f S_(mu_(-t)) (x)-beta(p_up+p_dn)slash 2, $
$ (f_(j up +),f_(j up -),f_(j dn +),f_(j dn -)):= & (-beta Q_up (u_j), 2lambda_L^2 (s_j+q_0)alpha_up (u_j), \
 & -lambda_L [H_up (-u_j)H_dn (-u_j)+lambda_L^2 d_pi]alpha_dn^S (u_j), \
 & -lambda_L [H_up (-u_j)H_dn (u_j)-lambda_L^2 d_pi]alpha_dn^P (u_j)), $
$ K_(j t):=f_(j t +)+f_(j t -), quad
 frak(c)_(j t):=frac(2lambda_L^2 D_t P_L (u_j)(s_j+q_0), beta^2 E'_(8,j) K_(j t)). $ <eq-field-L-mode>
If this quotient is $0 slash 0$ but $K_(j,-t)!=0$, use the equal, regular quotient
$ frak(c)_(j t)=frac(-lambda_L P_L (u_j) [rho Xi_j+t m_f Pi_j],beta^2 E'_(8,j) K_(j,-t)), $
$ Xi_j:=(s_j+q_0)^2-4a^2 r_up r_dn s_j-lambda_L^2 d_pi, quad
 Pi_j:=(s_j+q_0)^2+4a^2 r_up r_dn s_j+lambda_L^2 d_pi. $ <eq-field-L-quotient>
For $U != 0$, $mu h(mu^2-h^2)!=0$, simple nonzero roots off both spin lattices, $P_L (u_j)!=0$ at non-atomic roots, and $(K_(j up),K_(j dn))!=(0,0)$, the vertex is
$ Gamma_("L",sigma t)^(nu nu' omega) = & beta P_L (z) lr([frac(1,(z+a r_t)^2+omega^2 slash 4)-frac(Q_t (z),E_8 (z))])delta_(sigma t)delta_(nu nu')+U delta_(sigma (-t)) \
 & +delta_(sigma t) frac(2lambda_L^2 D_t (v^2+q_0)P_L (v),E_8 (v)(z+v)) \
 & -delta_(sigma (-t)) frac(lambda_L P_L (v),E_8 (v)) lr([frac(t m_f [H_t (-v)H_(-t) (-v)+lambda_L^2 d_pi],z-v) \
 & quad +frac(rho [H_t (-v)H_(-t) (v)-lambda_L^2 d_pi],z+v)]) \
 & -beta^2 sum_j frak(c)_(j t) frac(1,v^2-s_j) lr([frac(f_(j sigma +),z-u_j)+frac(f_(j sigma -),z+u_j)]). $ <eq-field-L>
At $k=0$ omit the atomic root $s_j=a^2$ from the sum and the nonvanishing requirements: its coefficient is zero. The inverse is two-sided, $Gamma_("L",sigma t)^(nu nu' omega)=Gamma_("L",t sigma)^(nu' nu omega)$, and neither square-root signs nor a common rescaling of each mode affect the answer.

=== Reductions, continuous extensions and exceptional sets <sec-field-limits>

At $h=0$, all functions on the right below are evaluated at $(U,beta,mu)$:
$ Gamma_("L",sigma sigma)=(Gamma_"d"+Gamma_"m")slash 2, quad
 Gamma_("L",sigma (-sigma))=(Gamma_"d"-Gamma_"m")slash 2, $
$ Gamma_("T" sigma)=Gamma_"m", quad Gamma_("A" sigma)=(Gamma_"s"-Gamma_"t")slash 2, quad Gamma_("E" sigma)=-Gamma_"t" slash 2. $ <eq-field-h0>
The particle–hole (Shiba) map $c_dn -> c_dn^dagger$, $(U,mu,h)->(-U,h,mu)$ @essl2024 gives the $mu=0$ values. Write $D$ for the zero-field source $(-U,beta,h)$, $epsilon_up=1$, $epsilon_dn=-1$, $R_up nu=nu$, $R_dn nu=-nu-omega$. Then
$ Gamma_("L",sigma t)^(nu nu' omega)=frac(epsilon_sigma epsilon_t,2)lr([Gamma_"d"^D (R_sigma nu,R_t nu',omega)+epsilon_sigma epsilon_t Gamma_"m"^D (R_sigma nu,R_t nu',omega)]), $
$ Gamma_("T" up)=(Gamma_"s"^D-Gamma_"t"^D)slash 2, quad Gamma_("A" up)=Gamma_"m"^D, quad Gamma_("E" up)=-Gamma_"t"^D slash 2. $ <eq-field-mu0>
For down orientation use the source $(-U,beta,-h)$. In the charge/$S_z$ basis the longitudinal map is the orthogonal similarity $cal(S)diag(Gamma_"d"^D,Gamma_"m"^D)cal(S)$ with $cal(S)=mat(cal(M)_-,cal(M)_+;cal(M)_+,cal(M)_-)$; separate inversion of its diagonal blocks is incorrect.

*Longitudinal axes.* In @eq-field-L replace the three kernels by their limits where their denominators vanish:
$ frac(D_t,z+v) -> beta p_0 p_(-t)delta_(nu (-nu'-omega)) quad (mu_t=0), $
$ frac(rho,z+v) -> beta p_0 delta_(nu (-nu'-omega)) quad (mu=0), quad
 frac(m_f,z-v) -> sigma beta p_dn delta_(nu nu') quad (h=0, sigma=-t). $ <eq-field-kernel-limits>
Together with @eq-field-L-quotient these give the continuous extension on $mu=plus.minus h!=0$. Agreement there with the vertex is *numerically verified, not derived*. On $h=0$ and $mu=0$ the independently known values are @eq-field-h0 and @eq-field-mu0. At $mu=h=0$ use those values, rather than the simultaneous $0 slash 0$ mode coefficients.

*Other exceptional sets.* For the transverse formula $P_T (u_j)=0$ alone is harmless; at $W=plus.minus 2a$, where $P_T$ has a double zero, the root $s_j=0$ is omitted (@sec-field-T). Confluent roots and lattice roots require limiting formulas, not direct substitution. For $"T"$ their removable continuations are those of @prop-special with the present data (at $mu=0$, the mirror-even lattice cancellation and mirror-odd genuine pole must be distinguished). For $"L"$, confluence can be treated by confluent divided differences of the root sum away from thermal poles; this continuation is numerically checked, not established here with uniform bounds. Axis lattice roots can be genuine divergences: at $beta=2$, $mu=h=0.3$, $k=0$ an example occurs near $U=2.3120511518839$ (numerical). Off-axis lattice cancellations are expected from coincident lattice and thermal values, but are not established here. Non-atomic zeros of $P_L (u_j)$, $u_j=0$, and vanishing adjugate modes are excluded from the stated longitudinal domain; a vanishing mode requires another adjugate column. No general claim about those isolated extensions is made. Thermal poles and zeros of $cal(L)_sigma$ are the genuine divergences discussed in @sec-field-div; averaging across them is not a regularization.

#pagebreak(weak: true)
= Vertex divergences <app-div>

The vertex $Gamma_r = beta^2 (chi_r^(-1) - eta_r bold(B)^(-1))$ diverges where $chi_r$ has a zero mode; the entries of $bold(B)^(-1)$ are finite, since $G$ has no zeros on the lattice. These divergences were found analytically at half filling @thunstroem2018, located as zeros of $det chi_r$ on finite boxes @rohshap2023, and mapped away from half filling through the eigenvalues of $chi_r$ on finite boxes @essl2024 @essl2023. @sec-res-box lists the genuine poles of the closed forms. Here we give a null vector of $chi_r$ at each of them, derive the conditions at $omega = 0$, and plot them (@fig-div). The conditions are derived under the nondegeneracy assumptions of @sec-res-box. The exceptional points, the map, and the statements about it are computed from these conditions and checked against finite boxes (@app-checks); they are numerical.

== Null vectors

At each genuine pole, $chi_r bold(g) = 0$ for the following vector $bold(g)$ (all sums converge absolutely):
- *Density and magnetic*, $mu != 0$, $K_j = 0$: $g_l = 1 slash (z_l^2 - s_j)$. At $K_j = 0$, $lambda alpha_j = -beta H(-u_j)$, so the mode @eq-mode is $phi_j = 2 beta u_j H(-u_j) thin bold(g)$, and @eq-mode-image gives $chi_r phi_j = 0$. At a genuine pole, $u_j H(-u_j) != 0$ (@sec-res-box).
- *Singlet*, $K_"s" = 0$ with $cal(P)(-As) != 0$: $g_l = 1 slash (t_l + As)$. In the proof of @thm-s, the matrix $Xi = diag(-1 slash a, K_"s" slash (a d_+ d_-))$ becomes singular, and its null direction corresponds to $bold(g)$.
- *Triplet*, $t_n + a rho b + a^2 = 0$ with $n != overline(n)$: $bold(g) = bold(e)_n - bold(e)_(overline(n))$. By @prop-pp the mirror-odd vector on the orbit has the eigenvalue $L_"t" (t_n) = -beta (t_n + a rho b + a^2) slash (2 cal(P)(t_n)) = 0$.
- *Half filling*, $beta H(z_n) = -lambda h$ with $n != overline(n)$: $bold(g) = bold(e)_n - bold(e)_(overline(n))$ (@sec-res-box). For $hat(D) = 0$, $g_l = hat(g)(z_l)$, the limit $mu -> 0$ of the density and magnetic case (checked numerically, @app-checks).
We do not show that these vectors span the kernel. Following Ref. @thunstroem2018, a divergence is _localized_ if its null vector has finite support, and _global_ otherwise. Only the triplet divergences and the mirror-odd ones at half filling are localized. At half filling the localized null vectors are mirror-odd and the global ones mirror-even: these are the antisymmetric and symmetric divergences of Ref. @essl2024. Away from half filling $bold(g)$ is neither. A global null vector is concentrated where its denominator, $t_l + As$ or $z_l^2 - s_j$, is small. This is how the localized divergences continue away from half filling (below).

== Static transfer

From now on $omega = 0$ ($k = 0$); at $omega != 0$, localized divergences occur in more channels even at half filling @thunstroem2018. The zero-field discussion below is static; @sec-field-div returns to arbitrary transfer. At $omega = 0$, $b = 2 mu$ and $t_n = |x_n|^2$ are real, so $chi_"s"$ and $chi_"t"$ are real symmetric matrices, while $chi_"d"$ and $chi_"m"$ are centrohermitian, $chi_(r, overline(n) overline(m)) = chi_(r, n m)^*$ @essl2024. Their eigenvalues are therefore real or come in complex-conjugate pairs. A conjugate pair can change the sign of its real part without vanishing (a _pseudo-divergence_ @essl2024); $Gamma_r$ stays finite there. Up to a factor $beta^3$, $chi_r$ depends on $beta$, $U$ and $mu$ only through $beta U$ and $beta mu$ @essl2024.

*Triplet.* Since $t_n = mu^2 + nu_n^2$ and $|rho| < 1$,
$ t_n + a rho b + a^2 = (mu + a rho)^2 + nu_n^2 + a^2 (1 - rho^2) > 0, $
so there are no triplet divergences.

*Singlet.* With $xi^2 = mu^2 + 2 a rho mu + a^2 (1 - 8 p_1)$, which is real, @eq-T gives $cal(T)_0 (-As) = beta tanh(beta xi slash 2) slash (2 xi)$, so
$ K_"s" cosh frac(beta xi, 2) = cosh frac(beta xi, 2) + a frac(sinh(beta xi slash 2), xi), $ <eq-div-s>
an even, hence real, function of $xi$. $Gamma_"s"$ diverges where it vanishes, except where $cal(P)(-As) = d_+ d_-$ (@eq-dc) vanishes too. For $U != 0$ and $b != plus.minus 2 a$, $d_(plus.minus) = 0$ in fact forces $K_"s" = 0$: if $d_+ = 0$, then $-As = t_+$, and @eq-T-atomic with the value of $S_-$ in @lem-S gives
$ K_"s" = 1 - frac(4 a S_-, beta (2 a - b)) = frac(4 a p_1 - b (p_1 + p_2), (2 a - b)(p_1 + p_2)) = -frac(d_+, 2 a (2 a - b)(p_1 + p_2)) = 0, $
and likewise for $d_-$. For $mu > 0$ this happens on one curve, at $U < 0$, where $d_- = -2 a [4 a p_1 + 2 mu (p_0 + p_1)] = 0$; for $mu < 0$ on its mirror image, where $d_+ = 0$. There the separable term of @eq-Gamma-s is $0 slash 0$. Numerically it has a finite limit, and on finite boxes no eigenvalue of $chi_"s"$ with an eigenvector aligned with $bold(g)$ becomes small (@app-checks): this curve is not a divergence.

*Density and magnetic, $mu != 0$.* The roots $s_(1, 2)$ of @eq-Delta are real or complex conjugate. Since $K(overline(u)) = overline(K(u))$ for the function $K(u) := beta H(-u) + lambda alpha_0 (u)$, conjugate roots give $K_1 = 0$ only together with $K_2 = 0$. That is two real conditions on the two parameters, which generically hold at isolated points; we do not pursue them. Let $s_j$ be real and $u := u_j$. By @eq-alpha,
$ alpha_0 (plus.minus u) = -h frac(cosh(beta (u plus.minus mu) slash 2), cosh(beta (u minus.plus mu) slash 2)), $
so the parts of $alpha_0$ even and odd in $u$ are
$ alpha_"e" = -h frac(1 + cosh(beta mu) cosh(beta u), cosh(beta u) + cosh(beta mu)), quad quad alpha_"o" = -h frac(sinh(beta mu) sinh(beta u), cosh(beta u) + cosh(beta mu)). $ <eq-div-alpha>
With $h sinh(beta mu) = beta rho slash 2$ (@eq-weight-hyp), the even and odd parts of $K_j = K(u)$ are
$ K_"e" = beta (s_j + q_1) + lambda alpha_"e", quad quad K_"o" = -frac(2 a beta rho thin u thin Phi(s_j), cosh(beta u) + cosh(beta mu)), $
where
$ Phi(s) := cosh(beta mu) + cosh(beta u) + epsilon_r a frac(sinh(beta u), u) quad quad (u^2 = s). $ <eq-div-Phi>
$Phi$ is even in $u$, hence a real function of real $s$; for $s = -y^2$ it reads $cosh(beta mu) + cos(beta y) + epsilon_r a sin(beta y) slash y$. At the root, @eq-Delta and @eq-alpha give $beta^2 H(u) H(-u) = lambda^2 h^2 = lambda^2 alpha_0 (u) alpha_0 (-u)$, and hence
$ beta H(-u) K(-u) = lambda alpha_0 (-u) K(u). $
So $K_j = 0$ if and only if $K(u) = K(-u) = 0$, that is, if and only if $K_"e" = K_"o" = 0$; for $u != 0$ the latter means $Phi(s_j) = 0$. Conversely, if $Phi(s_j) = 0$, then $K_"o" = 0$ and $K(u) = K(-u) = K_"e"$, and the last identity becomes
$ K_"e" A_"e" = 0, quad quad A_"e" := beta (s_j + q_1) - lambda alpha_"e". $
Along each analytic arc of the curve $Phi(s_j) = 0$ in the $(beta mu, beta U)$ plane, one factor therefore vanishes identically. The arcs with $K_"e" = 0$ are divergences; on the others $K_j != 0$ and the vertex is finite. Both factors vanish only where two such arcs cross, and there $K_j = 0$. With @eq-div-alpha, $Gamma_"d"$ and $Gamma_"m"$ thus diverge where, for a real root $s_j != 0$,
$ Phi(s_j) = 0 quad "and" quad beta (s_j + q_1)(cosh(beta u_j) + cosh(beta mu)) = lambda h (1 + cosh(beta mu) cosh(beta u_j)). $ <eq-div-ph>
On $Phi(s_j) = 0$ the two sides of the second equation agree up to sign, as $K_"e" A_"e" = beta^2 (s_j + q_1)^2 - lambda^2 alpha_"e"^2$. So the second condition only selects the sign: $(s_j + q_1) thin epsilon_r (1 + cosh(beta mu) cosh(beta u_j)) slash (cosh(beta u_j) + cosh(beta mu)) >= 0$, with equality exactly at the crossings. By @lem-S,
$ sum_n frac(1, z_n^2 - s) = frac(S_0 (u) - S_0 (-u), 2 u) = -frac(beta sinh(beta u), 2 u (cosh(beta u) + cosh(beta mu))), $ <eq-div-T>
an absolutely convergent sum. So $Phi(s) = (cosh(beta u) + cosh(beta mu)) [1 - (2 epsilon_r a slash beta) sum_n (z_n^2 - s)^(-1)]$, and the first condition in @eq-div-ph reads
$ beta = 2 epsilon_r a sum_n frac(1, z_n^2 - s_j), $
the counterpart of $hat(D) = 0$ (@eq-hf-D) with $s_j$ in place of $hat(s)$.

*Half filling.* At $mu = 0$,
$ Phi(s) = 2 cosh^2 frac(beta u, 2) [1 + epsilon_r a frac(tanh(beta u slash 2), u)], $
and the bracket is $hat(D) slash beta$ with $hat(u)$ replaced by $u$. The zeros of $cosh(beta u slash 2)$ are the lattice points $s = z_n^2$. For $s_j = hat(s)'$ they are the localized divergences $beta H(z_n) = -lambda h$; for $s_j = hat(s)$ they are the removable cases of item 4 of @prop-special. The zeros of the bracket at $s_j = hat(s)$ are the global divergences $hat(D) = 0$. These are Eqs. (20) and (22) of Ref. @thunstroem2018 at $omega = 0$. Localized divergences occur only in the density channel, at $nu_n^2 = 3 a^2$, since $hat(s)' = -a^2 (1 + 2 epsilon_r)$ at $k = 0$.

For $mu != 0$, $cosh(beta u) + cosh(beta mu) > 0$ for real $s$, so no localized divergences remain. The localized lines continue as zeros of $Phi$. Along them $g_j$ is concentrated on the orbit ${n, overline(n)}$ with $z_n^2 approx s_j$: as $mu -> 0$, $z_n^2 - s_j approx 2 i mu nu_n$ and $z_(overline(n))^2 - s_j approx -2 i mu nu_n$, so $bold(g)$ tends to $bold(e)_n - bold(e)_(overline(n))$ up to normalization.

*Exceptional points.* For a finite complex symmetric matrix whose kernel is spanned by $bold(g)$, the zero eigenvalue is defective, an exceptional point, exactly when $bold(g)^T bold(g) = 0$. We use the corresponding condition for the null vectors $bold(g)_j$ as the criterion for the exceptional points of $chi_r$ on the density and magnetic lines:
$ bold(g)_j^T bold(g)_j = sum_n frac(1, (z_n^2 - s_j)^2) = frac(dif, dif s) sum_n frac(1, z_n^2 - s) |_(s = s_j) = 0. $ <eq-div-ep>
We do not prove this for the infinite matrix. On finite boxes, two eigenvalues of $chi_r$ with eigenvectors aligned with $bold(g)_j$ form a near-zero pair of nearly equal size there, which shrinks as the box grows (@app-checks). Generically a pair of real eigenvalues meets at zero and continues as a conjugate pair; the pseudo-divergence lines of Ref. @essl2024 start at these points. The singlet has no exceptional points, as its $bold(g)$ is real and $bold(g)^T bold(g) > 0$.

== The divergence map

#figure(
  placement: auto,
  image("doped-gamma/figures/divergences.svg", width: 82%),
  caption: [Divergences of $Gamma_"d"$ (red), $Gamma_"m"$ (green) and $Gamma_"s"$ (yellow) at $omega = 0$, from @eq-div-ph and @eq-div-s. The map is symmetric under $mu -> -mu$. Open circles: the localized divergences $beta U = plus.minus 2 pi (2 n + 1) slash sqrt(3)$ at half filling. Black dots: exceptional points, @eq-div-ep; on the second and third loop at $U > 0$ they lie near the loop ends, $|beta mu| approx 0.549$, and are not resolved at this scale. Pseudo-divergences @essl2024, where $Gamma_r$ is finite, are not shown.],
) <fig-div>

@fig-div shows the lines in the window $|beta mu| <= 4$, $|beta U| <= 25$. The statements below describe this window; the numbers are from `doped-gamma/check_divergences.jl` (@app-checks).
- *Density, $U > 0$.* Each localized line and the global line above it are the two halves of one closed loop. On the first loop, the exceptional points are at $|beta mu| = 0.564$, $beta U = 4.33$ (a doping of $rho = 0.060$). The loop ends at $|beta mu| = 0.569$, $beta U = 4.45$, where $s_1 = s_2$. For $beta U -> oo$ at fixed $beta mu$, both roots tend to $-3 a^2$, and $Phi(-3 a^2) = cosh(beta mu) + cos(sqrt(3) beta a) + sin(sqrt(3) beta a) slash sqrt(3)$ vanishes for some $a$ only if $cosh(beta mu) <= 2 slash sqrt(3)$. Likewise, with $q_1 -> 3 a^2$ and $rho^2 = 4 p_0 p_2 sinh^2(beta mu)$, the roots are real only if $sinh^2(beta mu) <= 1 slash 3$. Both give $|beta mu| <= "arsinh"(1 slash sqrt(3)) = 0.549$: to leading order, the loops shrink to this range.
- *Singlet, $U > 0$.* The lines start at the global density lines, with which they coincide at half filling @thunstroem2018, and rise with $|beta mu|$.
- *Magnetic, $U > 0$.* There are no divergences in the window, as at half filling @thunstroem2018.
- *$U < 0$.* The density lines start at the localized points, the magnetic ones at the global points of half filling. The first exceptional points are at $|beta mu| = 1.121$ (density) and $0.971$ (magnetic); beyond them the lines leave the window at $beta U = -25$. A line passes from one root to the other where $s_1 = s_2$, for example at $|beta mu| = 1.453$, $beta U = -6.97$ on the first density line.
This is the structure that Ref. @essl2024 found numerically (its Fig. 2a, where $delta mu$ is our $mu$). It reports a doping of about $6 %$ at the first exceptional points for $U > 0$, and Ref. @essl2023 gives them at $beta delta mu approx 0.6$, $1.12$ and $0.975$, in agreement with ours.

== In the notation of Ref. @thunstroem2018 <sec-div-thun>

@app-explicit uses the notation of Ref. @thunstroem2018. It relates to that of the main text as follows (the symbols on the right are local to @app-explicit and to this section):
#figure(
  kind: table,
  caption: [Main-text notation and that of Ref. @thunstroem2018 and @app-explicit, with $tilde(nu) = nu - i mu$.],
  text(size: 9pt, {
  set par(justify: false)
  table(
    columns: (auto, auto),
    align: left,
    stroke: none,
    inset: (x: 6pt, y: 3.5pt),
    fill: (_, y) => if calc.even(y) and y > 0 { luma(242) },
    table.hline(),
    table.header([*Main text*], [*Ref. @thunstroem2018, @app-explicit*]),
    table.hline(stroke: 0.4pt),
    [$Gamma_(r, n m)$, $delta_(m overline(n))$], [$Gamma_r^(nu nu' omega)$ with $nu = nu_n$, $nu' = nu_m$, $omega = omega_k$; $delta_(nu (-nu' - omega))$],
    [$x_n$, $overline(x)_n$, $t_n$], [$i (nu - i mu)$, $-i (nu + omega + i mu)$, $t_nu$],
    [$z_n$, $z_n^2 + omega^2 slash 4$], [$i (tilde(nu) + omega slash 2)$, $-tilde(nu) (tilde(nu) + omega)$],
    [$s_j$, $u_j$], [$-B_(r, j)^2 - omega^2 slash 4$, $(i slash 2) W_j$],
    [$hat(s)'$, $hat(s)$ ($mu = 0$)], [$-A_r^2 - omega^2 slash 4$, $-B_r^2 - omega^2 slash 4$],
    [$-As$, $xi$], [$B_"s"^2$, $(i slash 2) W_"s"$],
    [$cal(G)(t_n)$, $t_n + a rho b + a^2$], [$t_nu + C$, $t_nu - A_"t"^2$],
    [$T(i W slash 2 - mu)$], [$i tan(beta (W + omega + 2 i mu) slash 4)$],
    table.hline(),
  )}),
) <tab-thun>
So $z_n^2 - s_j = B_(r, j)^2 - tilde(nu) (tilde(nu) + omega)$: the denominators of the separable terms are those of Ref. @thunstroem2018 with $nu -> tilde(nu)$ and with $B_r$ replaced by the two constants $B_(r, j)$. At half filling these are $A_r$ and $B_r$; away from it they are the two roots of one quadratic, and the localized and global divergences of half filling become the zeros of one function.

At $omega = 0$, with $s_j = -B_(r, j)^2$, $u_j = i B_(r, j)$ and $q = (U^2 slash 4)(1 + 4 epsilon_r p_1)$, the results of this appendix read:
- *Null vectors:* $g(nu) = 1 slash ((nu - i mu)^2 - B_(r, j)^2)$ (density and magnetic), $1 slash (t_nu - B_"s"^2)$ (singlet), and $bold(e)_nu - bold(e)_(-nu - omega)$ (triplet; localized divergences at half filling).
- *Density and magnetic*, @eq-div-ph:
  $ & cosh(beta mu) + cos(beta B_(r, j)) + epsilon_r U frac(sin(beta B_(r, j)), 2 B_(r, j)) = 0 quad "and" \
  & (q - B_(r, j)^2)(cos(beta B_(r, j)) + cosh(beta mu)) = epsilon_r U^2 sqrt(p_0 p_2) (1 + cosh(beta mu) cos(beta B_(r, j))). $
  The first condition is $beta = epsilon_r U sum_nu (B_(r, j)^2 - (nu - i mu)^2)^(-1)$. Both sides are even in $B_(r, j)$, hence real for real $B_(r, j)^2$.
- *Exceptional points*, @eq-div-ep: $sum_nu ((nu - i mu)^2 - B_(r, j)^2)^(-2) = 0$.
- *Half filling:* the first condition becomes $2 cos^2(beta B slash 2) [1 + epsilon_r U tan(beta B slash 2) slash (2 B)] = 0$. Its first factor vanishes for $B^2 = nu^2$, which for $B = A_r$ is the localized condition $nu (nu + omega) = A_r^2$ of Eq. (20) of Ref. @thunstroem2018. Its second factor, for $B = B_r$, is the global condition of Eq. (22) there, up to the sign $epsilon_r$.
- *Singlet*, @eq-div-s, with $B_"s"^2 = (U^2 slash 4)(8 p_1 - 1) - U rho mu$:
  $ cos frac(beta sqrt(B_"s"^2 - mu^2), 2) + U frac(sin(beta sqrt(B_"s"^2 - mu^2) slash 2), 2 sqrt(B_"s"^2 - mu^2)) = 0, $
  except on the curve where $d_minus.plus = 0$.

== Divergences with doping and field <sec-field-div>

This section restores arbitrary transfer $omega=2pi k slash beta$ and uses the field notation of @sec-field-explicit, in particular $h$ for the physical field and $kappa=beta sqrt(p_0p_2)$. Conditions below concern actual square-summable null vectors, not the accumulation of susceptibility eigenvalues at zero as the frequency tends to infinity. They hold on the regular domains stated there, with thermal poles approached from nonsingular parameters. Numerical statements and the unresolved longitudinal extensions are explicitly distinguished.

=== Transverse spectrum, selection and null vectors <sec-field-T-div>

For $u^2=s$ define the even functions
$ D_k (s):=cosh(beta u)+(-1)^k cosh(beta mu), quad
 R_k (s):=frac(cosh(beta mu)cosh(beta u)+(-1)^k,D_k (s)), $
$ Phi_k^T (s):=D_k (s)-a frac(sinh(beta u),u), quad
 K_e^T (s):=beta(s+q_1)+4a^2 kappa R_k (s), $
$ ell_T (s):=-frac(K_e^T (s),[(s+a^2-W^2 slash 4)^2-4a^2 s]). $ <eq-field-spectrum>
Use $sinh(beta u)slash u=beta$ at $u=0$. Suppose $mu U W!=0$ and $P_T (u)!=0$ at every zero of $Phi_k^T$ (with $D_k!=0$). Then
$ "spec" chi_("T" up)={0} union {ell_T (s):Phi_k^T (s)=0}, quad
 g_s (nu)=frac(1,z_mu (nu)^2-s). $ <eq-field-spectrum-set>
The eigenvectors and their coalescence loci depend on $mu,U,beta$ and the parity of $k$, but *not on the field*. Only the eigenvalues move with $h$ through $q_1$ and $P_T$. The reason is the same Cauchy-column calculation as in the magnetic problem: $chi_("T" up)bold(g)_s=ell_T (s)bold(g)_s+Phi_k^T (s)bold(w)_s$ for a regular remainder; the complex transfer changes the polynomials but not the thermal lattice sum. Compactness gives the accumulation point zero; the same pole calculation exhausts the nonzero spectrum under the stated exclusions.

Consequently the transverse divergence condition is
$ Phi_k^T (s)=0, quad K_e^T (s)=0, quad P_T (u)!=0, $ <eq-field-T-div>
and $s$ is then a root of $e_T$. For a simple zero the null vector is $bold(g)_s$ above. Equivalently at a regular root $s_j$,
$ K_j^T:=beta H_T (-u_j)-4a^2 [rho S_mu (u_j)-beta(p_0+p_2)slash 2]=0. $
With $alpha(u):=rho S_mu (u)-beta(p_0+p_2)slash 2$, $alpha(u)alpha(-u)=kappa^2$ and $e_T (u_j)=0$ give the same even/odd selection as in @eq-div-ph. On $Phi_k^T (s_j)=0$,
$ K_e^T (s_j) A_e^T (s_j)=0, quad A_e^T (s):=beta(s+q_1)-4a^2 kappa R_k (s). $
Only the $K_e^T=0$ branch is selected; $A_e^T=0$ alone cancels. For real $s_j$ at $k=0$ the sign test is $(s_j+q_1)R_k (s_j)<=0$. The pole is genuine provided $u_j P_T (u_j)H_T (-u_j)!=0$.

*Atomic cancellations.* At $Phi_k^T (s)=P_T (u)=0$, $s=(a plus.minus W slash 2)^2$, the equations $K_e^T=0$ and $e_T (u)=0$ follow automatically but imply no zero mode: the eigenvalue formula is $0 slash 0$. At $k=0$ these curves continue the cancelled atomic pair at $h=0$. They must not be counted as divergences without testing the limiting eigenvalue.

*Exceptional points.* Multiple zeros of $Phi_k^T$ coalesce eigenvectors. For a simple null direction, the bilinear self-overlap criterion is
$ sum_n frac(1,(z_mu (nu_n)^2-s)^2)=T_mu'(s)=0, quad
 T_c (s):=-frac(beta sinh(beta u),2u [cosh(beta u)+(-1)^k cosh(beta c)]). $ <eq-field-ep-T>
The prime is differentiation in $s$. This follows by differentiating the absolutely convergent sum $T_c=sum_n (z_c (nu_n)^2-s)^(-1)$; as in @eq-div-ep, its interpretation as a defective zero is checked on finite boxes, not proved here for the infinite operator. At fixed $beta h=0.5$ the first static transverse exceptional divergence is near $(beta mu,beta U)=(0.97773,-8.96512)$ (numerical), where the divergence curve crosses a field-independent coalescence locus.

At $k=0$ or $h=0$ the transverse spectrum is conjugation-closed. At general $k$, $"spec" chi_("T" up) (-omega)=overline("spec" chi_("T" up) (omega))$ and $"spec" chi_("T" dn)=overline("spec" chi_("T" up))$, so both orientations have the same divergences. For a real zero $s$,
$ "Im" K_e^T (s)=beta omega(h-a m_f). $
Thus at $omega!=0$ a real-root divergence requires $h=a m_f$ (and $U>0$ if $h!=0$). For complex roots the condition is complex and generically has codimension two. A numerical example is $k=2$, $beta h=1$, $(beta U,beta mu) approx (-16.69512285,0.75904959)$, $beta^2s approx -101.558+1.632i$; it is an isolated divergence in that fixed-field slice, not a whole line.

=== Opposite-spin pairing <sec-field-A-div>

Apply $(U,mu,h)->(-U,h,mu)$ to the transverse spectrum, null vector, branch selection, exclusions and exceptional-point condition. Explicitly,
$ Phi_k^A (s)=cosh(beta u)+(-1)^k cosh(beta h)+a frac(sinh(beta u),u), $
$ W_A=i omega-2mu, quad q_1^A=a^2-W_A^2 slash 4-2a^2 (p_0+p_2)+a W_A rho, $
$ Phi_k^A (s)=0, quad beta(s+q_1^A)+4a^2 beta sqrt(p_up p_dn)frac(cosh(beta h)cosh(beta u)+(-1)^k,cosh(beta u)+(-1)^k cosh(beta h))=0. $ <eq-field-A-div>
The up-oriented null vector is $1 slash (z_h (nu)^2-s)$; for the down orientation reflect $nu -> -nu-omega$. Also $P_A (u)!=0$ is required as above. At a real zero the imaginary part of the second condition is $beta omega(mu+a rho)$.

=== Equal-spin pairing <sec-field-E-div>

On the mirror-odd orbit $n!=overline(n)$,
$ chi_("E" sigma) (bold(e)_n-bold(e)_(overline(n)))=frac(beta cal(L)_sigma (t_(sigma nu_n)),cal(P)_sigma (t_(sigma nu_n)))(bold(e)_n-bold(e)_(overline(n))). $
It is diagonal in disjoint orbit vectors and has no exceptional points. Its complete divergence set is
$ omega!=0, quad mu_sigma+a r_sigma=0, quad a^2 (1-r_sigma^2)=-nu_n (nu_n+omega), quad n!=overline(n). $ <eq-field-E-div>
The printed orbit vector is the localized null vector. To see the condition, expand
$ cal(L)_sigma (t_(sigma nu))=(mu_sigma+a r_sigma)^2+a^2 (1-r_sigma^2)+nu(nu+omega)-i omega(mu_sigma+a r_sigma). $
At $omega=0$ its real part is positive. At $|k|=1$ the only negative product belongs to the forbidden fixed point, so $|k|>=2$ is necessary. Conversely for an allowed negative product $-c$, choose $a=plus.minus sqrt(mu_sigma^2+c)$; along $mu=mu_sigma-sigma h$, $r_sigma$ continuously ranges from $+1$ to $-1$ as $sigma h$ increases, and takes the required value $-mu_sigma slash a$. Thus the set is nonempty for either sign of $U$; it is invariant under the particle–hole map. This construction fixes $U$ through $a$, rather than asserting existence at arbitrary prescribed $U$. For example, at $k=2$ the orbit ${-1,-2}$ diverges near $(beta U,beta mu,beta h)=(10.97247706,0.56858193,0.13141807)$ (numerical).

=== Longitudinal selection, null vectors and exceptional points <sec-field-L-div>

Use the simple, nonzero, non-atomic roots and off-axis domain of @eq-field-L, and suppose $Q_up (plus.minus u_j)!=0$. Define the branch ratio
$ frak(r)(u):=-frac(2lambda_L^2 (u^2+q_0)alpha_up (u),beta Q_up (u)), quad
 Phi_L (s):=frac(frak(r)(u)-frak(r)(-u),2u), quad u^2=s. $ <eq-field-L-ratio>
At $u=u_j$ this is $frak(r)_j:=f_(j up -)slash f_(j up +)$; write $frak(r)_j (plus.minus u_j):=frak(r)(plus.minus u_j)$ using @eq-field-L-mode. The polynomial identity
$ Q_up (u)Q_up (-u)-4lambda_L^4 pi_P pi_S (u^2+q_0)^2=E_8 (u)H_dn (u)H_dn (-u) $
and $alpha_up (u)alpha_up (-u)=beta^2 pi_P pi_S$ give $frak(r)_j (u)frak(r)_j (-u)=1$. Also
$ K_(j up)=-beta Q_up (u_j) [1+frak(r)_j (u_j)]. $
For real $s_j<0$, $|frak(r)_j|=1$; for real $s_j>0$, $frak(r)_j$ is real. Thus $Phi_L (s_j)=0$ selects *both* $frak(r)_j=plus.minus 1$. The genuine branch is
$ frak(r)_j=-1 quad <==> quad K_(j up)=K_(j dn)=0. $ <eq-field-L-div>
On $frak(r)_j=+1$ the vertex stays finite: $f_(j up +)-f_(j up -)=0$ but $K_(j up)=-2beta Q_up (u_j)!=0$. For negative $s_j$ select $"Re" frak(r)_j=-1$, not merely $Phi_L=0$. For complex roots impose the full complex equation, generically two real conditions.

At a genuine pole a null vector in spin and frequency is
$ g_(j sigma) (nu)=frac(f_(j sigma +),z_(mu_sigma) (nu)^2-s_j), \
 (f_(j up +),f_(j dn +))=(-beta Q_up (u_j),-lambda_L [H_up (-u_j)H_dn (-u_j)+lambda_L^2 d_pi]alpha_dn^S (u_j)). $ <eq-field-L-null>
The reason is that $K_(j sigma)=0$ makes the mode in @eq-field-L proportional to this vector; its polynomial image under $P_L chi_L$ vanishes when the thermal denominators do, whereas the nonzero residue numerator makes the inverse pole genuine. #block(breakable: false)[
At a simple null direction the exceptional-point test is
$ bold(g)_j^T bold(g)_j=sum_(sigma in {up,dn}) f_(j sigma +)^2 T_(mu_sigma)' (s_j)=0, $ <eq-field-ep-L>
]
using $T_c$ of @eq-field-ep-T. Symmetry of the coupled susceptibility gives the same necessary self-orthogonality condition as in @eq-div-ep; finite-box diagnosis of defectiveness remains numerical. The longitudinal spectrum is centrohermitian at every transfer under the joint mirror on both spins. In the slice $beta h=0.5$ the first static loop has an exceptional divergence near $(beta mu,beta U)=(0.64162,4.78028)$ (numerical). Very near the omitted atomic root, $1+frak(r)_j$ can be tiny without crossing zero; double precision is not a reliable branch test there.

=== Axis reductions and numerical checks <sec-field-div-checks>

At $h=0$, $E_8=e_"d"e_"m"$; the longitudinal spectrum is the union of the zero-field density and magnetic spectra, with spin null vectors $(g,g)$ and $(g,-g)$ respectively. The ratio test reduces to their $K_j=0$ selection. The transverse sector is magnetic. The opposite-spin pair has the singlet spectrum on $cal(M)_+$ and the triplet spectrum on $cal(M)_-$, with the physical factors $2$ and $-2$ respectively ($chi_A cal(M)_+=2chi_"s"cal(M)_+$, $chi_A cal(M)_-=-2chi_"t"cal(M)_-$); its vertex reduction is @eq-field-h0. Equal-spin pairing is the triplet problem on the allowed subspace with the factor $-2$ in $chi$.

At $mu=0$ the orthogonal longitudinal similarity of @eq-field-mu0 preserves eigenvalues, nullity and bilinear self-overlap: in the charge/$S_z$ basis a source density null vector maps to $(cal(M)_-g,cal(M)_+g)$ and a magnetic one to $(cal(M)_+g,cal(M)_-g)$. The opposite-spin pair is the magnetic source $(-U,beta,h)$; the transverse sector has the source singlet/triplet split just described. In its mirror form the global pole is $hat(D)=0$, and the localized poles obey $beta H_T (z_n)=4a^2 beta p_0$ for $n!=overline(n)$; the opposite sign is removable. At $mu=h=0$ recover the localized and global conditions already listed in this appendix.

On either $h=0$ or $mu=0$, equal-spin pairing has no divergences, and at $mu=0$, $h!=0$ there are no nonstatic transverse divergences. These axis exclusions follow by substituting the zero-field thermal weights in the pair conditions (the zero-field singlet/triplet nonstatic exclusion); they do not extend to simultaneous doping and field. On $mu=plus.minus h$ the longitudinal ratio theorem does not apply: use the continuous extension, distinguishing genuine axis lattice poles from removable limits as in @sec-field-limits. No off-axis lattice-root exclusion is claimed.

`doped-gamma/check_field.jl` transcribes the displayed field formulas independently of the package's vertex code at 256 bits, tests both orientations and the axis reductions, and tests predicted divergences against the package susceptibility on growing boxes, including overlaps with the printed null vectors. Equal-spin boxes are restricted to $cal(M)_-$ before taking eigenvalues; their structural even-subspace zeros are not divergences. At 256 bits the worst normalized vertex difference over 4322 comparisons, including the double zero of $P_T$ at $W = plus.minus 2 a$, is $6.4 times 10^(-63)$; the branch-ratio identity differs from one by at most $2.3 times 10^(-60)$. At the tested nonlocalized divergences, the largest smallest-box eigenvalue falls from $3.7 times 10^(-8)$ at 24 frequencies to $3.8 times 10^(-11)$ at 96, with overlap deficit at most $3.2 times 10^(-8)$; the restricted equal-spin eigenvalues are below $1.5 times 10^(-18)$. The transverse and longitudinal exceptional pairs shrink from about $1.1 times 10^(-6)$ and $8.4 times 10^(-7)$ at 48 frequencies to $3.4 times 10^(-8)$ and $2.6 times 10^(-8)$ at 192 (per spin in the longitudinal case). These measured box convergence, overlaps and exceptional-point checks are numerical evidence, not error bounds or proofs of the unresolved extensions.

= List of symbols <app-symbols>

Symbols introduced inside a proof, in @app-explicit or in @app-div are local to it and not listed, except those of the proof of @thm-ph that @app-details uses and the field symbols shared by Appendices F and G.

#let sym-table(..rows) = text(size: 9pt, {
  set par(justify: false)
  table(
  columns: (21%, 1fr, 22%),
  align: (left, left, left),
  stroke: none,
  inset: (x: 4pt, y: 3.5pt),
  fill: (_, y) => if calc.even(y) and y > 0 { luma(242) },
  table.hline(),
  table.header([*Symbol*], [*Meaning*], [*Defined in*]),
  table.hline(stroke: 0.4pt),
  ..rows.pos().flatten(),
  table.hline(),
)})

*Setup and input.*
#sym-table(
  ([$U$, $mu$, $beta$, $a$], [interaction, chemical potential (zero at half filling), inverse temperature; $a = U slash 2$], [@eq-H]),
  ([$p_0$, $p_1$, $p_2$], [Boltzmann weights of $|0 chevron.r$, of each singly occupied state, of $|up dn chevron.r$], [@eq-weights]),
  ([$rho$, $h$, $overline(c)$], [$p_2 - p_0$ (filling), $beta sqrt(p_0 p_2)$, $beta (p_0 + p_2) slash 2$], [@eq-rho]),
  ([$nu_n$, $Omega_ell$, $omega_k$], [fermionic and bosonic Matsubara frequencies; the fixed transfer $omega_k = Omega_k$], [@eq-freq]),
  ([$x_n$, $w$, $J$], [$mu + i nu_n$, $i omega_k$, $J(x) = 1 slash (x^2 - a^2)$], [@eq-x]),
  ([$G$], [propagator, as a function of $nu$ and as the rational function $G(x)$], [@eq-G]),
  ([$chi_(sigma sigma')$, $chi_("pp", sigma sigma')$], [generalized susceptibility and its particle–particle form], [@eq-chi-def]),
  ([$chi_r$, $chi_(0, r)$, $B$, $B_n$, $bold(B)$], [channels $r = "d", "m", "s", "t"$, bare susceptibility, bubble], [@eq-B]),
  ([$Gamma_r$, $eta_r$], [channel-irreducible vertex, sign of the bubble term], [@eq-Gamma-def]),
  ([$z_n$, $overline(n)$], [centered lattice, mirror index], [@eq-z]),
  ([$cal(M)$, $PV$], [mirror matrix; principal value along symmetric boxes], [@sec-lattice]),
  ([$cal(N)$, $c_ell$, $Pi$, $Rreg$], [the part of $chi_(up dn)$ with the pair coefficient $c_ell$, and its regular part], [@eq-pair]),
  ([$C$, $cC$], [Cauchy matrix; its action on functions on the lattice], [@eq-C, @lem-calculus]),
  ([$Q_(plus.minus)$, $q_0$], [$Q_(plus.minus)(z) = (z plus.minus a)^2 - w^2 slash 4$, $q_0 = a^2 - w^2 slash 4$], [@eq-Q]),
  ([$S_k$, $S_(plus.minus)$], [thermal sum $PV sum_n (z_n - x)^(-1)$; its values at the zeros of $Q_(plus.minus)$], [@eq-S, @lem-S]),
)

*Singlet and triplet.*
#sym-table(
  ([$b$, $t_n$], [$2 mu - w$; $x_n (b - x_n) = t_(overline(n))$], [@eq-bt]),
  ([$cal(P)$, $cal(G)$, $As$], [polynomials of the particle–particle channels], [@eq-T]),
  ([$cal(T)_k$, $xi$], [particle–particle sum $sum_n (t_n - t)^(-1)$], [@eq-T]),
  ([$K_"s"$], [thermal denominator of the singlet], [@thm-s]),
  ([$t_(plus.minus)$], [zeros of $cal(P)$], [@lem-T]),
  ([$d_(plus.minus)$, $c$], [$As + t_(plus.minus)$; the pair coefficient $c_(-k)$], [@eq-dc]),
  ([$cal(M)_(plus.minus)$, $L_"s"$, $L_"t"$, $g_(plus.minus)$, $gamma_(plus.minus)$], [mirror projectors; eigenvalues and rank-two part of $chi_"s"$, $chi_"t"$], [@eq-L]),
  ([$chi_("s", "reg")$], [rank-two part of $chi_"s"$], [@eq-chi-s-num]),
)

*Density and magnetic.*
#sym-table(
  ([$epsilon_r$], [channel sign, $epsilon_"d" = +1$, $epsilon_"m" = -1$], [@sec-res-ph]),
  ([$H$, $q_1$, $lambda$, $P$, $G_2$, $e$], [polynomials of the particle–hole channels], [@eq-HP, @eq-Delta]),
  ([$s_j$, $u_j$], [roots of $e$ in $z^2$; either square root], [@eq-Delta]),
  ([$alpha_k$], [thermal function $rho S_k - overline(c)$], [@eq-alpha-def]),
  ([$F$, $g_j$, $DD$], [ingredients of the dense and separable terms], [@eq-coeffs]),
  ([$K_j$, $Z_j$, $gamma_j$], [thermal denominator, numerator and weight of the separable term $j$], [@eq-coeffs]),
  ([$delta$, $zeta_r$, $N_2$, $N_0$, $N$], [finite-rank part of $P chi_r$], [@eq-N]),
  ([$cal(I)$, $overline(cal(I))$], [rank-two sums], [@eq-U]),
  ([$y_v$, $D_v$, $E_v$, $F_j$], [column of $chi_r^(-1)$ and its coefficients], [proof of @thm-ph]),
  ([$phi_j$, $alpha_j$], [pole pair (mode) at the root $u_j$], [@eq-mode]),
  ([$W$, $W_0$, $W_2$], [remainder polynomials of Step 1], [@eq-source, (A7)]),
  ([$K(u)$, $Z(u)$, $Z(s)$], [$K_j$, $Z_j$ as functions of the root; even part of $Z$], [@eq-branch, @prop-special]),
)

*Half filling and removable singularities.*
#sym-table(
  ([$hat(s)$, $hat(s)'$], [roots of $e$ in $z^2$ at $mu = 0$], [@eq-hf-defs]),
  ([$hat(g)$, $hat(D)$, $hat(u)$], [separable factor, thermal denominator, $hat(u)^2 = hat(s)$], [@eq-hf-D]),
  ([$O$, $c$], [mirror orbit ${c, overline(c)}$ with $|O|$ elements], [@sec-res-special]),
  ([$T'$, $K'$], [$cal(T)_k (-As)$ and $K_"s"$ without the orbit], [@eq-Gamma-s-lz]),
  ([$hat(T)'$, $hat(D)'$], [$sum_l hat(g)(z_l)$ and $hat(D)$ without the orbit], [@eq-Gamma-hf-lz]),
)

*Doping and field (local to the field parts of Appendices F–G).*
#sym-table(
  ([$h$, $Z_f$, $p_up$, $p_dn$], [physical field (splitting $2h$), normalization, spin-resolved weights], [@sec-field-input]),
  ([$kappa$, $m_f$, $mu_sigma$, $r_sigma$], [thermal constant $beta sqrt(p_0p_2)$ (called $h$ outside the field parts), magnetization, spin chemical potential, $2 chevron.l n_(-sigma) chevron.r-1$], [@sec-field-explicit, @eq-field-weights]),
  ([$"L"$, $"T" sigma$, $"A" sigma$, $"E" sigma$, $b_X$], [coupled longitudinal, transverse, oriented opposite-spin and equal-spin pairs; physical bubbles], [@sec-field-input, @eq-field-bubbles]),
  ([$z_c$, $S_c$, $W$], [centered physical frequency, thermal sum, complex transfer $i omega-2h$], [@eq-field-centered, @eq-field-T-data]),
  ([$H_T$, $P_T$, $e_T$, $q_1$], [transverse polynomials; $s_j$, $u_j$ are their root pairs], [@eq-field-T-data]),
  ([$cal(P)_sigma$, $cal(G)_sigma$, $cal(L)_sigma$, $b_sigma$, $t_(sigma nu)$], [equal-spin pair polynomials and frequency variables], [@sec-field-E]),
  ([$lambda_L$, $pi_P$, $pi_S$, $D_t$, $c_t$, $d_pi$], [longitudinal interaction and weight combinations], [@eq-field-L-data]),
  ([$H_t$, $P_L$, $Q_t$, $E_8$, $E'_(8,j)$], [longitudinal polynomials; derivative in $s$ at a root], [@eq-field-L-data, @sec-field-L]),
  ([$alpha_t$, $alpha_t^P$, $alpha_t^S$, $f_(j sigma plus.minus)$], [thermal factors and pole-pair mode coefficients], [@eq-field-L-mode]),
  ([$K_(j t)$, $frak(c)_(j t)$, $Xi_j$, $Pi_j$], [longitudinal thermal denominator and separable coefficient], [@eq-field-L-mode, @eq-field-L-quotient]),
  ([$epsilon_sigma$, $R_sigma$, $cal(S)$], [spin sign, down-spin frequency reflection, orthogonal Shiba matrix], [@eq-field-mu0]),
  ([$D_k$, $R_k$, $Phi_k^T$, $K_e^T$, $ell_T$, $Phi_k^A$], [transverse spectral and divergence functions; their pair images], [@eq-field-spectrum, @eq-field-A-div]),
  ([$frak(r)_j$, $Phi_L$, $T_c$], [longitudinal branch ratio, scalar selection function, lattice sum for self-overlap], [@eq-field-L-ratio, @eq-field-ep-T]),
)
