@testitem "Frequency shifts full vertex" begin
    using SparseIR

    (d, m, s, t) = CHANNELS

    # Tolerance relative to the size of the combined terms, so the check stays
    # meaningful whether the vertex is tiny (small U) or huge (large β).
    crossing(F, a, Fd, b, Fm) = isapprox(F, a * Fd + b * Fm; atol=1e-14 * (abs(Fd) + abs(Fm)))

    for β in (1e-3, 1e0, 1e3), U in (-1e0, -1e-3, 1e-3, 1e0)
        at = HubbardAtom(U, β)

        for ν in FermionicFreq.(-3:2:3), ν´ in FermionicFreq.(-3:2:3), ω in BosonicFreq.(-4:2:4)
            freqs = (ν, ν´, ω)

            freqs_pp = (ν, ν´, -ν - ν´ - ω)
            Fd = full_vertex(d, at, freqs_pp)
            Fm = full_vertex(m, at, freqs_pp)
            @test crossing(full_vertex(s, at, freqs), 0.5, Fd, -1.5, Fm)
            @test crossing(full_vertex(t, at, freqs), 0.5, Fd, 0.5, Fm)

            freqs_phbar = (ν, ν + ω, ν´ - ν)
            Fd = full_vertex(d, at, freqs_phbar)
            Fm = full_vertex(m, at, freqs_phbar)
            @test crossing(full_vertex(d, at, freqs), -0.5, Fd, -1.5, Fm)
            @test crossing(full_vertex(m, at, freqs), -0.5, Fd, 0.5, Fm)

            freqs_phbar = (ν, -ν´ - ω, ν´ - ν)
            Fd = full_vertex(d, at, freqs_phbar)
            Fm = full_vertex(m, at, freqs_phbar)
            @test crossing(full_vertex(s, at, freqs), 0.5, Fd, -1.5, Fm)
            @test crossing(full_vertex(t, at, freqs), -0.5, Fd, -0.5, Fm)
        end
    end
end

@testitem "F at extremely high frequencies (#14)" begin
    # https://github.com/tuwien-cms/OvercompleteIR.jl/issues/14
    using SparseIR

    beta = 1.0
    U = 1.0

    for ch in CHANNELS
        model = HubbardAtom(U, beta)

        f(n) = full_vertex(ch, model, (FermionicFreq(2n + 1), FermionicFreq(2n + 1), BosonicFreq(2n)))
        @test f(2^10) ≈ f(2^39)
        @test f(2^10) ≈ f(2^60)
    end
end

@testitem "BSE consistency" begin
    using SparseIR

    for channel in CHANNELS
        nf_sum = 10^3
        nf, nb = 8, 7
        U, β = 12.3, 0.456
        atom = HubbardAtom(U, β)

        νmax = FermionicFreq(nf - 1)
        ωmax = BosonicFreq(nb - 1)
        νmax_sum = FermionicFreq(nf_sum - 1)

        ν = -νmax:νmax
        ν´ = -νmax:νmax
        ω = -ωmax:ωmax
        ν₁ = -νmax_sum:νmax_sum

        νν´ω = Iterators.product(ν, ν´, ω)
        νν₁ω = Iterators.product(ν, ν₁, ω)
        ν₁ν´ω = Iterators.product(ν₁, ν´, ω)
        ν₁ω = Iterators.product(ν₁, ω)

        Γ = gamma.(channel, atom, νν₁ω)
        Χ₀ = chi0.(channel, atom, ν₁ω)
        F = full_vertex.(channel, atom, ν₁ν´ω)

        Φ = Array{Float64}(undef, size(νν´ω))
        for I in CartesianIndices(Φ)
            (ν, ν´, ω) = Tuple(I)
            Φ[I] = sum(Γ[ν, ν₁, ω] * Χ₀[ν₁, ω] * F[ν₁, ν´, ω] for ν₁ in eachindex(ν₁))
        end
        κ = channel isa SingletChannel ? 1 : -1
        Φ .*= κ / β

        Φ_ana = @. full_vertex(channel, atom, νν´ω) - gamma(channel, atom, νν´ω)

        @test Φ ≈ Φ_ana rtol = 1e-3
    end
end
