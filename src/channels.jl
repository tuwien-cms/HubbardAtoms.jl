"""
Supertype of the four channels, `PHChannel` (`DensityChannel`, `MagneticChannel`) and
`PPChannel` (`SingletChannel`, `TripletChannel`). The channels are singleton types;
`CHANNELS` holds one instance of each.
"""
abstract type SpinChannel end

Base.broadcastable(ch::SpinChannel) = Ref(ch)

"Particle-hole channels: `DensityChannel`, `MagneticChannel`."
abstract type PHChannel <: SpinChannel end
"Particle-particle channels: `SingletChannel`, `TripletChannel`."
abstract type PPChannel <: SpinChannel end

"Density (charge) channel, `χ_d = χ_↑↑ + χ_↑↓` (Equation 5a)."
struct DensityChannel <: PHChannel end
"Magnetic (spin) channel, `χ_m = χ_↑↑ - χ_↑↓` (Equation 5b)."
struct MagneticChannel <: PHChannel end
"Singlet particle-particle channel (Equation 5c); frequencies in pp notation (Equation 4)."
struct SingletChannel <: PPChannel end
"Triplet particle-particle channel (Equation 5d); frequencies in pp notation (Equation 4)."
struct TripletChannel <: PPChannel end

# Convenient shorthands
const d = DensityChannel
const m = MagneticChannel
const s = SingletChannel
const t = TripletChannel

"`(DensityChannel(), MagneticChannel(), SingletChannel(), TripletChannel())`"
const CHANNELS = (d(), m(), s(), t())
