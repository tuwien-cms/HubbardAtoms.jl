function bethe_salpeter(n_conv_frequencies, n_iterations;
                        β = 1.0, U = 1.0, ωmax = 10 * U, ϵ = 1e-3, channel, conv)
    (atom, F, Γ, basis4, eval4_in, νν´ω) = setup(; β, U, ωmax, ϵ, channel, conv)

    ν, ν´, ω = to_arrays(νν´ω)
    ν₁ = -FermionicFreq(n_conv_frequencies-1):FermionicFreq(n_conv_frequencies-1)

    # Define frequencies and MatsubaraEval object for Matsubara sum
    ν´ω = unique(zip(ν´, ω))
    freqs_conv = [(ν₁, ν´, ω) for ((ν´, ω), ν₁) in Iterators.product(ν´ω, ν₁)]
    freqs_conv_full = to_full_freq.(conv, vec(freqs_conv))
    eval4_conv = OvercompleteIR.MatsubaraEval(basis4, freqs_conv_full)

    # Starting point for iteration
    F₀ = Γ.(νν´ω)

    A = getA(channel, atom, ν, ν₁, ω, conv)
    Fₙ = complex.(F₀)
    for _ in 1:n_iterations
        Fₗ = fit(eval4_in, Fₙ; ϵ)
        F_r = reshape(evaluate(eval4_conv, Fₗ), size(freqs_conv))
        F_conv = Dict((ν´, ω) => F_r[i, :] for (i, (ν´, ω)) in enumerate(ν´ω))

        # Φₙ = κ/β ∑ᵥ₁ Γ(νν₁ω) Χ₀(ν₁ω) Fₙ₋₁(ν₁ν´ω)
        Φₙ = [inner(A[ν, ω], F_conv[ν´, ω]) for (ν, ν´, ω) in νν´ω]
        @. Fₙ = Γ(νν´ω) + Φₙ
    end
    F_analytic = F.(νν´ω)
    norm(Fₙ - F_analytic) / norm(F_analytic)
end

function getA(channel, atom, ν, ν₁, ω, conv)
	β = atom.beta
	
	Γ(w::FermiFermiBose) = Atom.Γ(channel, atom, w, conv)
	Χ₀(ν, ω) = Atom.χ₀(channel, atom, (ν, ω))
    κ = ifelse(channel isa SingletChannel, +1, -1)
	function A(w::FermiFermiBose)
        (ν, ν´, ω) = w
        κ / β * Γ(w) * Χ₀(ν´, ω)
    end
	
	νω = unique(zip(ν, ω))
	Dict((ν, ω) => map(vp -> A((ν, vp, ω)), ν₁) for (ν, ω) in νω)
end

function setup(; β, U, ωmax, ϵ, channel, conv)
	atom = Atom.HubbardAtom(U, β)

	F(w) = Atom.F(channel, atom, w, conv)
	Γ(w) = Atom.Γ(channel, atom, w, conv)

    basis_f, basis_b = SparseIR.finite_temp_bases(β, ωmax, ϵ)
    basis4 = OvercompleteBasis(DEFAULT_FOURPOINT_SET,
                               AugmentedBasis(basis_f, MatsubaraConst),
                               AugmentedBasis(basis_b, MatsubaraConst))
	
	eval4_in = OvercompleteIR.MatsubaraEval(basis4)
	νν´ω = to_conv_freq.(conv, eval4_in.wsample)

	return (atom, F, Γ, basis4, eval4_in, νν´ω)
end


"Convert array of tuples to tuple of arrays." 
function to_arrays(aot::AbstractArray{T}) where {T}
    Ts = fieldtypes(T)
    N = length(first(aot))
    toa = ntuple(j -> Array{Ts[j]}(undef, size(aot)), Val(N))
    @inbounds for i in eachindex(aot)
        @simd for j in 1:N
            toa[j][i] = aot[i][j]
        end
    end
    return toa
end

inner(x::Vector{<:Complex}, y::Vector{<:Complex}) =
    LinearAlgebra.BLAS.dotu(length(x), x, 1, y, 1)
inner(x::Vector{<:Real}, y::Vector{<:Real}) = dot(x, y)
