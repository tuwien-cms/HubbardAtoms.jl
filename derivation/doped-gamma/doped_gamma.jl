# Analytic doped Hubbard atom, h=0, Thunstroem et al. (2018) conventions.
# This file uses only Julia standard libraries. It evaluates the vertices from residue sums and a finite
# linear solve, independently of the closed forms of ../doped-gamma-derivation.typ, and serves as a reference.
module DopedAtom
using LinearAlgebra
export Atom, green, chiph, chi, bubble, GammaPP, GammaPH, gamma, fullvertex

struct Atom{T<:AbstractFloat}
    U::T
    beta::T
    delta::T
    p0::T
    p1::T # probability of EACH singly occupied state
    p2::T
end
function Atom(U::Real, beta::Real, delta::Real)
    U, beta, delta = float.(promote(U, beta, delta))
    all(isfinite, (U,beta,delta)) && beta > 0 || throw(ArgumentError("finite U,delta and beta>0 required"))
    E = [U/4+delta, -U/4, U/4-delta]
    w = exp.(-beta .* (E .- minimum(E)))
    Z = w[1]+2w[2]+w[3]
    Atom(U,beta,delta,w[1]/Z,w[2]/Z,w[3]/Z)
end
iz(at,n) = im*(2n+1)*oftype(at.U,pi)/at.beta
iw(at,k) = im*2k*oftype(at.U,pi)/at.beta
xval(at,n) = at.delta+iz(at,n)
rho(at) = at.p2-at.p0
greenx(at,x) = (x+at.U/2*rho(at))/(x*x-(at.U/2)^2)
green(at,n) = greenx(at,xval(at,n))
# (p2-p0)/(2delta+iOmega), including the joint zero-frequency/half-filled limit.
function pairratio(at,k)
    if iszero(k)
        iszero(at.delta) && return at.beta*at.p0
        # avoids subtracting almost equal p0,p2
        return (at.p0+at.p2)*tanh(at.beta*at.delta)/(2at.delta)
    end
    rho(at)/(2at.delta+iw(at,k))
end

# Twelve regular rational terms of Essl et al., Eq. D3 at h=0.
function regular_ph(at,x,y,w)
    a=at.U/2; q=at.p1+at.p2
    (q-1)/((y+w+a)*(y+a)*(x-a)) + (1-q)/((x+w-a)*(x-a)*(y+a)) +
    (1-q)/((x+w+a)*(y-a)*(x-a)) + (q-1)/((x+w+a)*(x-a)*(y+w+a)) +
    (1-q)/((y+a)*(x+w+a)*(y+w+a)) + (1-q)/((y+a)*(x+w+a)*(x+a)) +
    (1-q)/((y-a)*(x+a)*(y+w-a)) + (q-1)/((y-a)*(x+w+a)*(x+a)) +
    (1-q)/((x+w-a)*(x+a)*(y+w-a)) + (q-1)/((y+a)*(x+w-a)*(x+a)) -
    q/((y-a)*(x+w-a)*(y+w-a)) - q/((y-a)*(x+w-a)*(x-a))
end

"Return (chi_upup, chi_updown) in particle-hole notation; n,m,k are integer indices."
function chiph(at,n,m,k)
    a=at.U/2; beta=at.beta; q=at.p1+at.p2
    x=xval(at,n); y=xval(at,m); w=iw(at,k)
    J(x)=inv(x*x-a*a)
    same = -beta*green(at,n)*green(at,n+k)*(n==m) +
        beta*at.U^2*q*(1-q)*J(x+w)*J(y)*((k==0)-(n==m))
    cross = -beta*at.p1*at.U^2*J(x)*J(x+w)*(n==m) +
        beta*at.U^2*(at.p0*at.p2-at.p1^2)*J(x)*J(y)*(k==0)
    cross += pairratio(at,n+m+k+1)*(inv(x+w-a)+inv(y-a))*(inv(y+w-a)+inv(x-a))
    cross += regular_ph(at,x,y,w)
    same,cross
end

function ppdata(at,n,k)
    a=at.U/2; b=2at.delta-iw(at,k); x=xval(at,n); t=x*(b-x)
    P=(t+a*a)^2-a*a*b*b
    Qnum=t+a*rho(at)*b+a*a*rho(at)^2
    B=-at.beta*Qnum/(2P)
    A=a*rho(at)*b+a*a*(1-8at.p1)
    Ls=at.beta*(t+A)/(2P)
    Lt=-at.beta*(t+a*rho(at)*b+a*a)/(2P)
    (;a,b,x,t,P,B,A,Ls,Lt)
end
function singlet_C(at,k)
    a=at.U/2; b=2at.delta-iw(at,k); c=pairratio(at,-k)
    [a^3*(2a^3*c+3a*a-b*b) a*a*(2a*a*c+a-b*b*c);
     a*a*(2a*a*c+a-b*b*c) a*(2a*c-1)]
end
function bubble(at,r,n,k)
    r in (:d,:m) && return -at.beta*green(at,n)*green(at,n+k)
    r in (:s,:t) || throw(ArgumentError("channel must be d,m,s,t"))
    -at.beta/2*green(at,n)*green(at,-n-k-1)
end
"Exact susceptibility; singlet/triplet use the diagonal/reflection plus rank-two formula."
function chi(at,r,n,m,k)
    if r in (:d,:m)
        uu,ud=chiph(at,n,m,k)
        return r==:d ? uu+ud : uu-ud
    end
    d=ppdata(at,n,k); refl=(n==-m-k-1); diag=(n==m)
    if r==:t
        return (d.B+d.Lt)/2*diag + (d.B-d.Lt)/2*refl
    end
    r==:s || throw(ArgumentError("channel must be d,m,s,t"))
    e=ppdata(at,m,k)
    f=[one(d.t),d.t]/d.P; g=[one(e.t),e.t]/e.P
    (d.Ls-d.B)/2*diag+(d.Ls+d.B)/2*refl + transpose(f)*singlet_C(at,k)*g
end

# Precision headroom for coincident roots and exponentially small thermal weights.
# This is a conservative heuristic, not an error bound; compare `bits` at difficult points.
function analytic_precision(at)
    span=max(at.U/4+at.delta,-at.U/4,at.U/4-at.delta)-
         min(at.U/4+at.delta,-at.U/4,at.U/4-at.delta)
    max(256,128+ceil(Int,4at.beta*span/log(2)))
end

"Infinite-box singlet/triplet vertex; only a two-by-two inversion."
struct GammaPP{A,M}
    at::A
    k::Int
    T::M
    bits::Int
end
function GammaPP(at0::Atom,k::Integer;bits=analytic_precision(at0))
    setprecision(BigFloat,bits) do
        at=Atom(BigFloat(at0.U),BigFloat(at0.beta),BigFloat(at0.delta))
        iszero(at.U) && return GammaPP(at,Int(k),zeros(Complex{BigFloat},2,2),Int(bits))
        a=at.U/2; b=2at.delta-iw(at,k)
        A=k==0 && abs(at.delta)==abs(a) ? -a*a : a*rho(at)*b+a*a*(1-8at.p1)
        tr=(-a*a+a*b,-a*a-a*b,-A)
        roots=Complex{BigFloat}[]
        for r in tr
            e=sqrt(complex(b*b/4-r)); append!(roots,(e,-e))
        end
        # t=b²/4-z², z=iν+iω/2; the three t factors give a minus sign.
        sumat=Atom(at.U,at.beta,zero(at.delta))
        moments=[-2/at.beta*residuesum(sumat,k,
            l==0 ? [one(b)] : l==1 ? [b*b/4,0,-1] : pmul([b*b/4,0,-1],[b*b/4,0,-1]),roots)
            for l in 0:2]
        S=[moments[1] moments[2]; moments[2] moments[3]]
        C=singlet_C(at,k)
        T=(I+C*S)\C
        GammaPP(at,Int(k),T,Int(bits))
    end
end
function gamma(g::GammaPP,r,n,m;raw=false)
    setprecision(BigFloat,g.bits) do
        ans=gamma_pp_raw(g,r,n,m)
        raw ? ans : ComplexF64(ans)
    end
end
function gamma_pp_raw(g::GammaPP,r,n,m)
    at=g.at; k=g.k
    r in (:s,:t) || throw(ArgumentError("analytic gamma implemented only for s,t"))
    iszero(at.U) && return zero(ComplexF64)
    d=ppdata(at,n,k); diag=(n==m); refl=(n==-m-k-1)
    if r==:t
        return at.beta^2/2*(inv(d.Lt)-inv(d.B))*(diag-refl)
    end
    e=ppdata(at,m,k)
    u=2/at.beta*[one(d.t),d.t]/(d.t+d.A)
    v=2/at.beta*[one(e.t),e.t]/(e.t+e.A)
    at.beta^2/2*(inv(d.Ls)+inv(d.B))*(diag+refl)-at.beta^2*transpose(u)*g.T*v
end
function fullvertex(at,r,n,m,k)
    sign=r==:s ? -1 : 1
    B=bubble(at,r,n,k); Bp=bubble(at,r,m,k)
    at.beta^2*(sign*(n==m)*B-chi(at,r,n,m,k))/(B*Bp)
end


# Ascending polynomial coefficients.
function pmul(a,b)
    c=zeros(promote_type(eltype(a),eltype(b)),length(a)+length(b)-1)
    for i in eachindex(a), j in eachindex(b); c[i+j-1]+=a[i]*b[j]; end
    c
end
peval(p,z)=evalpoly(z,Tuple(p))
function jetpoly(p,r,L)
    [sum(p[j+1]*binomial(j,i)*r^(j-i) for j in i:length(p)-1; init=zero(r)) for i in 0:L-1]
end
# Taylor multiplication, truncated to L terms.
jmul(a,b,L)=[sum(a[j+1]*b[i-j+1] for j in 0:i if j<length(a) && i-j<length(b); init=zero(a[1]*b[1])) for i in 0:L-1]
function jinv(a,L)
    out=zeros(typeof(inv(a[1])),L); out[1]=inv(a[1])
    for i in 1:L-1
        out[i+1]=-sum(a[j+1]*out[i-j+1] for j in 1:i if j<length(a); init=zero(out[1]))/a[1]
    end
    out
end
function sjet(at,k,r,L)
    u=at.beta*(r-at.delta)/2
    h=iseven(k) ? tanh(u) : coth(u)
    v=zeros(typeof(complex(h)),L); v[1]=-at.beta/2*h
    for i in 0:L-2
        v[i+2]=(sum(v[j+1]*v[i-j+1] for j in 0:i)-(i==0 ? at.beta^2/4 : 0))/(i+1)
    end
    v
end
# Residues at denominator roots; repeated roots are handled by Taylor jets.
function residuesum(at,k,num,roots;leading=one(at.U),extra=nothing,parts=false)
    out=zero(complex(at.U)); used=falses(length(roots))
    terms=Tuple{typeof(complex(at.U)),Int,typeof(complex(at.U))}[]
    for (i,r) in enumerate(roots)
        used[i] && continue
        same=findall(==(r),roots); used[same].=true; L=length(same)
        f=jmul(jetpoly(num,r,L),sjet(at,k,r,L),L)/leading
        for (j,s) in enumerate(roots)
            j in same && continue
            f=jmul(f,jinv([r-s,one(r)],L),L)
        end
        for ell in 1:L
            push!(terms,(r,ell,f[L-ell+1]))
        end
        if extra!==nothing
            f=jmul(f,jinv([r+extra,one(r)],L),L)
        end
        out+=f[L]
    end
    parts ? terms : out
end

struct GammaPH{A,T}
    at::A
    r::Symbol
    k::Int
    lambda::T
    qp::Vector{T}
    qm::Vector{T}
    Hminus::Vector{T}
    E::Vector{T}
    eroots::Vector{T}
    proots::Vector{T}
    mroots::Vector{T}
    Wnum::Matrix{T} # 4 polynomials of degree <=4, rows
    invsmall::Matrix{T}
    cbar::T
    Vparts::Vector{Vector{Tuple{T,Int,T}}}
    bits::Int
end

function GammaPH(at0::Atom,r::Symbol,k::Integer;bits=analytic_precision(at0))
    r in (:d,:m) || throw(ArgumentError("PH channel must be d or m"))
    setprecision(BigFloat,bits) do
        at=Atom(BigFloat(at0.U),BigFloat(at0.beta),BigFloat(at0.delta))
        a=at.U/2; beta=at.beta; w=iw(at,k); rho=DopedAtom.rho(at); ep=r==:d ? 1 : -1
        T=Complex{BigFloat}; lambda=T(ep*4a*a)
        q0=a*a-w*w/4; qp=T[q0,2a,1]; qm=T[q0,-2a,1]
        c0=a*a*(1+4ep*at.p1)-w*w/4
        H=T[c0,2a*rho,1]; Hminus=T[c0,-2a*rho,1]
        p=-beta*pmul(H,qm)
        # R(z,v)=[p(-v)q+(z)-p(z)q- (v)]/(z+v).
        N=zeros(T,5,5)
        for i in 0:4,j in 0:4
            N[i+1,j+1]=(i<3 ? qp[i+1] : 0)*p[j+1]*(-1)^j-p[i+1]*(j<3 ? qm[j+1] : 0)
        end
        R=zeros(T,4,4)
        for i in 4:-1:1, j in 0:3
            R[i,j+1]=N[i+1,j+1]
            N[i+1,j+1]=0; N[i,j+2]-=R[i,j+1]
        end
        # The finite part of chi after dividing by u(z)u(v).
        M=zeros(T,4,4)
        M[1:3,1:3].=ep*T[
            6a^5-8a^5*rho+2a^3*rho*w*w-a^3*w*w-a*w^4/8  -4a^4*rho+a*a*rho*w*w  -2a^3+a*w*w/2;
            -4a^4*rho+a*a*rho*w*w  -8a^3*rho  -4a*a*rho;
            -2a^3+a*w*w/2  -4a*a*rho  -2a]
        if k==0
            q=(1+rho)/2
            stat=q*(1-q)+ep*(at.p0*at.p2-at.p1^2)
            l=T[-a*a,0,1,0]
            M.+=4beta*a*a*stat*l*transpose(l)
        end
        e2=2c0-4a*a*rho*rho; e0=c0*c0-16a^4*at.p0*at.p2
        if r==:m && k==0
            # Exact shared roots with Q±. Do not split a multiple pole by roundoff.
            v2=a*a*(4rho*rho+8at.p1-3)
            e2=-a*a-v2; e0=a*a*v2
            eroots=T[a,-a,sqrt(complex(v2)),-sqrt(complex(v2))]
        else
            disc=sqrt(complex(e2*e2-4*e0)); v1=(-e2+disc)/2; v2=(-e2-disc)/2
            eroots=T[sqrt(complex(v1)),-sqrt(complex(v1)),sqrt(complex(v2)),-sqrt(complex(v2))]
        end
        E=beta^2*T[e0,0,e2,0,1]
        proots=T[-a+w/2,-a-w/2]; mroots=T[a+w/2,a-w/2]
        S0=residuesum(at,k,T[1],proots); S1=residuesum(at,k,T[0,1],proots)
        cbar=T(beta/2*(at.p0+at.p2))
        hp=zeros(T,4,3)
        hp[1,:]=[cbar-rho*S1-2a*rho*S0,rho*S0,0]
        hp[2,:]=[rho*S0*q0,-cbar+rho*S1,0]
        hp[3,:]=[rho*S1*q0,-2a*rho*S1-rho*S0*q0,cbar]
        Wnum=zeros(T,4,5)
        Wnum[:,1:4].+=lambda*rho*R
        for i in 1:4,j in 1:3
            Wnum[i,j:j+2].-=beta*M[i,j]*Hminus
            Wnum[i,1:3].-=lambda*M[i,j]*hp[j,:]
        end
        small=Matrix{T}(I,4,4); roots=[mroots;eroots;proots]
        for i in 1:4,j in 0:3
            num=[zeros(T,j);Wnum[i,:]]
            small[i,j+1]+=residuesum(at,k,num,roots;leading=beta^2)
        end
        Vparts=[residuesum(at,k,[zeros(T,j);one(T)],[proots;eroots];leading=beta^2,parts=true) for j in 0:3]
        GammaPH(at,r,Int(k),lambda,qp,qm,Hminus,E,eroots,proots,mroots,Wnum,inv(small),cbar,Vparts,Int(bits))
    end
end
function fvec(g,z)
    [one(z),z,z*z,z^3]/peval(g.qp,z)
end
function Wvec(g,z)
    [peval(g.Wnum[i,:],z) for i in 1:4]/(peval(g.qm,z)*peval(g.E,z))
end
function Vvec(g,z)
    at=g.at; out=zeros(typeof(z),4)
    for j in 0:3
        out[j+1]=(-at.beta*peval(g.Hminus,z)*z^j-g.lambda*g.cbar*(-z)^j)/(peval(g.qm,z)*peval(g.E,z))
        if !iszero(rho(at))
            out[j+1]-=g.lambda*rho(at)*sum(c*(-1)^(ell-1)/(z+r)^ell for (r,ell,c) in g.Vparts[j+1])
        end
    end
    out
end
function gamma(g::GammaPH,n,m;raw=false)
    setprecision(BigFloat,g.bits) do
        ans=gamma_raw(g,n,m)
        raw ? ans : ComplexF64(ans)
    end
end
function gamma_raw(g::GammaPH,n,m)
    at=g.at; k=g.k
    iszero(at.U) && return zero(complex(at.U))
    z=xval(at,n)+iw(at,k)/2; v=xval(at,m)+iw(at,k)/2
    un=inv(peval(g.qm,z)); um=inv(peval(g.qm,v))
    Dminus=-at.beta*peval(g.Hminus,z)*peval(g.qp,z)/peval(g.qm,z)
    cinv=(n==m ? Dminus/peval(g.E,z) : zero(z))-g.lambda*pairratio(at,n+m+k+1)/peval(g.E,v)
    cinv-=transpose(Vvec(g,z))*g.invsmall*Wvec(g,v)
    ans=at.beta^2*(cinv/(un*um)-(n==m ? inv(bubble(at,g.r,n,k)) : zero(z)))
    ans
end
end
