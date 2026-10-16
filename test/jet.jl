# Static analysis with JET.jl, run by the JET job of CI on the latest Julia. JET is not a test
# dependency, as each JET version supports only some Julia versions. To run it locally:
#     julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(; path="."); Pkg.add("JET"); include("test/jet.jl")'
using HubbardAtoms, JET, Test

const W3 = Tuple{FermionicFreq,FermionicFreq,BosonicFreq}
const W2 = Tuple{FermionicFreq,BosonicFreq}
const ATOMS = (HubbardAtom{Float64}, DopedHubbardAtom{Float64})
const CHANNEL_TYPES = (DensityChannel, MagneticChannel, SingletChannel, TripletChannel)
const FOUR_POINT = (gamma, full_vertex, chi, chi0, irreducible_vertex,
                    channel_reducible_vertex)

# every function of the package, with concrete argument types
function entry_points(A, R)
    calls = Any[(f, (R, A, W3)) for f in FOUR_POINT]
    push!(calls, (chi, (R, A, BosonicFreq)), (chi0, (R, A, W2)), (g3, (R, A, W2)),
          (gf, (A, FermionicFreq)))
    R === TripletChannel || push!(calls, (hedin, (R, A, W2)))  # for t it throws, by design
    return calls
end

@testset "JET" begin
    JET.test_package(HubbardAtoms; target_modules=(HubbardAtoms,))
    for A in ATOMS, R in CHANNEL_TYPES, (f, types) in entry_points(A, R)
        test_call(f, types; target_modules=(HubbardAtoms,))
        test_opt(f, types; target_modules=(HubbardAtoms,))
    end
end
