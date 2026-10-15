# Independent numerical checks of doped-gamma-derivation.typ (Appendix D, "Finite boxes"): the susceptibility of the
# reference evaluator (doped_gamma.jl) against exact diagonalization (ed_doped.jl), and its Gamma against the inverse of
# chi on finite boxes of 2N frequencies, N = 64, 256, 1024, with Richardson extrapolation in 1/N; all four channels.
# Run: julia --project=../.. indep_check.jl   (several minutes)
include(joinpath(@__DIR__, "doped_gamma.jl")); include(joinpath(@__DIR__, "ed_doped.jl"))
using LinearAlgebra, Printf; const DA = DopedAtom
BLAS.set_num_threads(4)
ganal(at, r, k, n, m) = r in (:d, :m) ? DA.gamma(DA.GammaPH(at, r, k), n, m) : DA.gamma(DA.GammaPP(at, k), r, n, m)
function chi_ed(U, β, r, n, m, k)
    uu, ud = ED.χph(U, β, (:up, :up), n, m, k), ED.χph(U, β, (:up, :dn), n, m, k)
    r == :d && return uu + ud; r == :m && return uu - ud
    kp = -k - n - m - 1; upp, udp = ED.χph(U, β, (:up, :up), n, m, kp), ED.χph(U, β, (:up, :dn), n, m, kp)
    bp = n == m ? -β / 2 * ED.G(U, β, n) * ED.G(U, β, -n - k - 1) : 0
    r == :s ? (-upp + 2udp - 2bp) / 4 : (upp + 2bp) / 4
end
function gnum(at, r, k, N, ps)
    idx = -N-k:N-1
    X = [DA.chi(at, r, n, m, k) for n in idx, m in idx]
    Xi = inv(X); η = r == :s ? -1 : 1
    [at.beta^2 * (Xi[findfirst(==(n), idx), findfirst(==(m), idx)] - (n == m ? η / DA.bubble(at, r, n, k) : 0)) for (n, m) in ps]
end
for (U, β, dμ, k) in ((2.0, 3.0, 0.3, 0), (2.0, 3.0, 0.3, 1), (-1.5, 2.0, -0.4, 2), (4.0, 1.0, 1.1, -1), (3.0, 5.0, 0.7, 0), (-3.0, 4.0, 1.3, 1), (1.0, 0.5, -2.0, 3))
    ED.δμ[] = dμ; at = DA.Atom(U, β, dμ)
    # chi vs ED at this parameter set
    # relative to the largest entry: entries that vanish exactly differ from ED by roundoff only
    pairs = [(r, n, m) for r in (:d, :m, :s, :t), n in -2:1, m in -2:1]
    ref = [chi_ed(U, β, r, n, m, k) for (r, n, m) in pairs]
    echi = maximum(abs.([DA.chi(at, r, n, m, k) for (r, n, m) in pairs] .- ref)) / maximum(abs, ref)
    ps = [(0, 0), (0, 1), (-1, 2), (0, -k - 1), (2, -3)]
    for r in (:d, :m, :s, :t)
        a = [ganal(at, r, k, n, m) for (n, m) in ps]
        Ns = (64, 256, 1024); nums = [gnum(at, r, k, N, ps) for N in Ns]
        errs = [norm(x - a) / norm(a) for x in nums]
        rich = (4nums[3] - nums[2]) / 3   # Richardson for 1/N error
        @printf("(U,β,δμ,k)=%-22s %s  χ-vs-ED %.1e | Γ rel diff N=64,256,1024: %.1e %.1e %.1e | Richardson %.1e\n", string((U, β, dμ, k)), r, echi, errs..., norm(rich - a) / norm(a))
    end
end
