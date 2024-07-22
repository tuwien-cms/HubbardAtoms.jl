@testitem "Frequency shifts full vertex" begin
    using SparseIR

    (d, m, s, t) = CHANNELS

    for β in (1e-3, 1e0, 1e-3), U in (-1e0, -1e-3, 1e-3, 1e0)
        at = HubbardAtom(U, β)

        for ν in FermionicFreq.(-3:2:3), ν´ in FermionicFreq.(-3:2:3), ω in BosonicFreq.(-4:2:4)
            freqs = (ν, ν´, ω)

            freqs_pp = (ν, ν´, -ν - ν´ - ω)
            Fd = full_vertex(d, at, freqs_pp)
            Fm = full_vertex(m, at, freqs_pp)
            @test isapprox(full_vertex(s, at, freqs), +0.5Fd - 1.5Fm; atol=1e-5, rtol=1e-15)
            @test isapprox(full_vertex(t, at, freqs), +0.5Fd + 0.5Fm; atol=1e-3, rtol=1e-15)

            freqs_phbar = (ν, ν + ω, ν´ - ν)
            Fd = full_vertex(d, at, freqs_phbar)
            Fm = full_vertex(m, at, freqs_phbar)
            @test isapprox(full_vertex(d, at, freqs), -0.5Fd - 1.5Fm; atol=1e-8, rtol=1e-14)
            @test isapprox(full_vertex(m, at, freqs), -0.5Fd + 0.5Fm; atol=1e-9, rtol=1e-13)

            freqs_phbar = (ν, -ν´ - ω, ν´ - ν)
            Fd = full_vertex(d, at, freqs_phbar)
            Fm = full_vertex(m, at, freqs_phbar)
            @test isapprox(full_vertex(s, at, freqs), +0.5Fd - 1.5Fm; atol=1e-8, rtol=1e-14)
            @test isapprox(full_vertex(t, at, freqs), -0.5Fd - 0.5Fm; atol=1e-3, rtol=1e-15)
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

        Φ = Array{ComplexF64}(undef, size(νν´ω))
        @inbounds for I in CartesianIndices(Φ)
            (ν, ν´, ω) = Tuple(I)
            Φ[I] = sum(Γ[ν, ν₁, ω] * Χ₀[ν₁, ω] * F[ν₁, ν´, ω] for ν₁ in eachindex(ν₁))
        end
        κ = channel isa SingletChannel ? 1 : -1
        Φ .*= κ / β

        Φ_ana = @. full_vertex(channel, atom, νν´ω) - gamma(channel, atom, νν´ω)

        @test Φ ≈ Φ_ana rtol = 1e-3
    end
end


# TODO: once OvercompleteIR is public, these can be enabled

# @testitem "Bethe-Salpeter equation" begin
#     using OvercompleteIR

#     include("_bethe_salpeter_defs.jl")

#     for channel in CHANNELS, conv in (PHConvention(), PHConventionThunstroem())
#         ϵ = 1e-2
#         n_conv_frequencies = 120
#         n_iterations = 6
#         β = 0.7
#         U = 0.7
#         @test bethe_salpeter(n_conv_frequencies, n_iterations; ϵ, channel, β, U, conv) < ϵ
#     end
# end


#     @testset "Fit vertex $vertex r=$channel" for vertex in (full_vertex,), channel in (d, m, s, t)

#     β = 4.0
#     U = 2.0
#     ωmax = 4 * U
#     ϵ = 1e-3

#     atom = Atom(U, β)
#     conv = PHConventionThunstroem()
#     basis_f, basis_b = SparseIR.finite_temp_bases(β, ωmax, ϵ)
#     basis4 = OvercompleteBasis(DEFAULT_FOURPOINT_SET,
#         AugmentedBasis(basis_f, MatsubaraConst),
#         AugmentedBasis(basis_b, MatsubaraConst))

#     eval4_in = OvercompleteIR.MatsubaraEval(basis4)
#     w_ph_in = to_conv_freq.(conv, eval4_in.wsample)
#         fw_in = vertex.(channel, atom, w_ph_in)
#         fl = fit(eval4_in, fw_in; ϵ)
#         fw_in_rec = evaluate(eval4_in, fl)
#         score_in = norm(fw_in - fw_in_rec) / norm(fw_in)
#         @test score_in < ϵ

#         w_ph_out, share = OvercompleteIR.test_box(conv, w_ph_in)
#         @test 0.5 < share < 1
#         fw_out = vertex.(channel, atom, w_ph_out)
#         eval4_out = OvercompleteIR.MatsubaraEval(basis4, to_full_freq.(conv, w_ph_out))
#         fw_out_rec = evaluate(eval4_out, fl)
#         score_out = norm(fw_out - fw_out_rec) / norm(fw_out)
#         @test ϵ / 3 < score_out < 3 * ϵ
#     end
