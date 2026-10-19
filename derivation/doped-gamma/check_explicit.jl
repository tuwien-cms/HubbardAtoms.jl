# Appendix "The vertices written out" (in the notation of Thunström et al.), transcribed without the
# main text's coefficients and compared with DopedHubbardAtom at 256 bits.
# Run: julia --project=../.. -t 1 check_explicit.jl
using HubbardAtoms, Printf

function explicit_inputs(U, β, μ, k)
    ω = 2k * (π / β)
    den = 2cosh(β * μ) + 2exp(β * U / 2)
    p0, p1, p2 = exp(-β * μ) / den, exp(β * U / 2) / den, exp(β * μ) / den
    ρ = sinh(β * μ) / (cosh(β * μ) + exp(β * U / 2))
    (; U, ω, β, μ, k, p0, p1, p2, ρ, ν=n -> (2n + 1) * (π / β))
end
# tan[β(W + ω)/4]/W, with the printed value β/4 at W = 0 (even k)
tanW(W, ω, β) = iszero(W) ? β / 4 : tan(β * (W + ω) / 4) / W
# δ_νν′ and δ_ν(−ν′−ω), by index (ν′ = −ν − ω means m = −n − k − 1)
δs(n, m, k) = (n == m), (m == -n - k - 1)

function explicit_t(c, n, m)
    (; U, ρ, β, μ, ω, k, ν) = c
    n == -n - k - 1 && return zero(complex(ν(n)))     # mirror fixed point ν = −ω/2
    C = U / 2 * ρ * (2μ - im * ω) + U^2 * ρ^2 / 4
    At² = -U^2 / 4 - U / 2 * ρ * (2μ - im * ω)
    t = (ν(n) - im * μ) * (ν(n) + ω + im * μ)
    P = ((ν(n) - im * μ)^2 + U^2 / 4) * ((ν(n) + ω + im * μ)^2 + U^2 / 4)
    d, dm = δs(n, m, k)
    -β * (At² + C) * P / ((t + C) * (t - At²)) * (d - dm)
end

function explicit_s(c, n, m)
    (; U, ρ, β, μ, ω, k, p1, ν) = c
    C = U / 2 * ρ * (2μ - im * ω) + U^2 * ρ^2 / 4
    Bs² = U^2 / 4 * (8p1 - 1) - U / 2 * ρ * (2μ - im * ω)
    W = sqrt(4Bs² + (ω + 2im * μ)^2)
    Ks = 1 + U * tanW(W, ω, β)
    t(n) = (ν(n) - im * μ) * (ν(n) + ω + im * μ)
    P = ((ν(n) - im * μ)^2 + U^2 / 4) * ((ν(n) + ω + im * μ)^2 + U^2 / 4)
    d, dm = δs(n, m, k)
    β * (Bs² + C) * P / ((t(n) + C) * (t(n) - Bs²)) * (d + dm) + 2U -
    2U * ((Bs² + U^2 / 4)^2 - U^2 / 4 * (2μ - im * ω)^2) /
    (Ks * (t(n) - Bs²) * (t(m) - Bs²))
end

function explicit_ph(c, ε, n, m; sign1=1, sign2=1)
    (; U, ρ, β, μ, ω, k, p0, p1, p2, ν) = c
    q = U^2 / 4 * (1 + 4ε * p1)
    Δ = U^2 / 4 * ρ^4 - ρ^2 * (q + ω^2 / 4) + U^2 * p0 * p2
    B² = (q - U^2 * ρ^2 / 2 - U * sqrt(complex(Δ)),
          q - U^2 * ρ^2 / 2 + U * sqrt(complex(Δ)))
    W = (sign1 * sqrt(4B²[1] + ω^2), sign2 * sqrt(4B²[2] + ω^2))
    ν̃n, ν̃m = ν(n) - im * μ, ν(m) - im * μ
    tn, tm = ν̃n * (ν̃n + ω), ν̃m * (ν̃m + ω)
    yn, ym = ν̃n + ω / 2, ν̃m + ω / 2
    N(t) = (t + U^2 / 4)^2 + U^2 * ω^2 / 4
    E(t) = (t - B²[1]) * (t - B²[2])
    # the printed bracket of the dense term, and its printed replacement on the diagonal
    dense = if n == m
        -(N(tn) / (2yn * E(tn)) +
          yn * (2(tn + U^2 / 4) * E(tn) - N(tn) * (2tn - B²[1] - B²[2])) / E(tn)^2)
    else
        (yn * N(tn) / E(tn) - ym * N(tm) / E(tm)) / (tm - tn)
    end
    sep = zero(tn)
    for j in 1:2
        # The static magnetic atomic root B² = −U²/4 is omitted analytically, not divided by zero.
        if ε == -1 && k == 0 && j == argmin(abs.(B² .+ U^2 / 4))
            @assert abs(B²[j] + U^2 / 4) < big"1e-60"
            continue
        end
        th = ε * U^2 / 2 * (im * ρ * tan(β * (W[j] + ω + 2im * μ) / 4) + p0 + p2)
        sep += im / 2 * ε * U^2 * ρ * W[j] * ((B²[j] + U^2 / 4)^2 + U^2 * ω^2 / 4) /
               ((B²[3-j] - B²[j]) * (tn - B²[j]) * (tm - B²[j])) *
               (q - B²[j] - im / 2 * U * ρ * W[j] + th) /
               (q - B²[j] - im / 2 * U * ρ * W[j] - th)
    end
    -β * (ν̃n^2 + U^2 / 4) * ((ν̃n + ω)^2 + U^2 / 4) *
    (1 / (tn - im * U * ρ * yn - U^2 * ρ^2 / 4) + (q - tn - im * U * ρ * yn) / E(tn)) *
    (n == m) + ε * U - im * ε * U^2 * ρ * dense + sep
end

# Half filling: Eq. (19) of Thunström et al. for r = d, m
function explicit_hf(c, ε, n, m)
    (; U, β, ω, k, ν) = c
    A² = U^2 / 4 * (1 + 2ε)
    B² = U^2 / 4 * (3exp(ε * β * U / 2) - 1) / (exp(ε * β * U / 2) + 1)
    t(n) = ν(n) * (ν(n) + ω)
    P = (ν(n)^2 + U^2 / 4) * ((ν(n) + ω)^2 + U^2 / 4)
    f = 1 + ε * U * tanW(sqrt(complex(4B² + ω^2)), ω, β)
    d, dm = δs(n, m, k)
    β * P * A² / (2t(n) * (t(n) - A²)) * (d - dm) +
    β * P * B² / (2t(n) * (t(n) - B²)) * (d + dm) +
    ε * U - ε * U * ((B² + U^2 / 4)^2 + U^2 * ω^2 / 4) / (f * (t(n) - B²) * (t(m) - B²))
end

function check_explicit()
    worst = Dict{String,BigFloat}()
    counts = Dict{String,Int}()
    for U in (big"1.3", big"-2.1", big"3.7"), β in (big"0.8", big"2.5", big"5.0"),
        μ in (big"0", big"0.37", big"-0.9", big"1.6"), k in -3:3
        c = explicit_inputs(U, β, μ, k)
        at = DopedHubbardAtom(U, β, μ)
        @assert isapprox(c.p0 + 2c.p1 + c.p2, 1; atol=big"1e-70")
        @assert isapprox(c.ρ, c.p2 - c.p0; atol=big"1e-70")
        for (n, m) in ((0, 0), (0, -k - 1), (2, 2), (-3, -2), (3, -1), (-2, -2))
            w = (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(2k))
            for r in
                (TripletChannel(), SingletChannel(), DensityChannel(), MagneticChannel())
                name, value = if r isa TripletChannel
                    "t", explicit_t(c, n, m)
                elseif r isa SingletChannel
                    "s", explicit_s(c, n, m)
                else
                    ε = r isa DensityChannel ? 1 : -1
                    if iszero(μ)
                        "hf-$(ε)", explicit_hf(c, ε, n, m)
                    else
                        val = explicit_ph(c, ε, n, m)
                        # Both independent signs of the square roots W_j must give the same result.
                        for (sg1, sg2) in ((-1, 1), (1, -1), (-1, -1))
                            other = explicit_ph(c, ε, n, m; sign1=sg1, sign2=sg2)
                            @assert isapprox(val, other; rtol=big"1e-40", atol=big"1e-40")
                        end
                        "ph-$(ε)", val
                    end
                end
                ref = gamma(r, at, w)
                err = abs(value - ref) / max(1, abs(ref))
                @assert err < big"1e-40" (U, β, μ, k, n, m, name, err)
                worst[name] = max(get(worst, name, zero(β)), err)
                counts[name] = get(counts, name, 0) + 1
            end
        end
    end
    # At βU = −2 log 3, μ = k = 0, the singlet and the density W (= √(4B²)) vanish: the printed limit β/4
    let U=-2log(big"3"), β=big"2", μ=big"0", k=0
        @assert isapprox(tanW(big"1e-30", 0, β), tanW(0, 0, β); rtol=big"1e-50")
        c = explicit_inputs(U, β, μ, k)
        at = DopedHubbardAtom(U, β, μ)
        for (n, m) in ((0, 0), (0, -1), (2, -3), (3, -1))
            w = (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(0))
            @assert isapprox(explicit_s(c, n, m), gamma(SingletChannel(), at, w);
                             rtol=big"1e-40", atol=big"1e-40")
            @assert isapprox(explicit_hf(c, 1, n, m), gamma(DensityChannel(), at, w);
                             rtol=big"1e-40", atol=big"1e-40")
        end
    end
    # Package-only sanity checks of the printed U = 0 shortcut, not formula comparisons.
    for μ in (big"0", big"0.4"), k in -1:1, r in CHANNELS
        at = DopedHubbardAtom(big"0", big"2", μ)
        @assert iszero(gamma(r, at, (FermionicFreq(1), FermionicFreq(-1), BosonicFreq(2k))))
    end
    for name in sort!(collect(keys(worst)))
        @printf("%-6s %5d entries, worst normalized error %.2e\n", name, counts[name],
                worst[name])
    end
    println("PASS: zero-W formulas (8 entries)")
    println("PASS: package-only U = 0 sanity checks (24 entries)")
end

setprecision(check_explicit, BigFloat, 256)
