include("/Users/tyler/Documents/code/EpidemicModeling/src/InfectionSim.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/ODEFit.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/POMDPsInterface.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/updater.jl")

pomdp = initParams(inf_loss=100.0, test_loss=1.0, testrate_loss=25.0)

function POMDPs.initialstate(pomdp::Params, rng::AbstractRNG=Random.GLOBAL_RNG)
     return ImplicitDistribution(initState, Normal(10_000,5_000), pomdp)
end

T = 5

s0 = rand(initialstate(pomdp))
b0 = initialbelief(pomdp, 50_000)

solver = POMCPOWSolver(criterion=MaxUCB(1.0), max_depth=50, tree_queries=5_000)
planner = POMCPOW.solve(solver, pomdp)
belief_updater = BootstrapFilter(pomdp, n_particles(b0))

POMCPOWhist = Simulate(T, copy(s0), b0, pomdp, planner, belief_updater)
plot(pomdp, POMCPOWhist)
