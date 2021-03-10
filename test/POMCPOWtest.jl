using POMCPOW, ParticleFilters, POMDPModelTools, POMDPSimulators
using ProgressBars
using Plots
include("/Users/tyler/Documents/code/EpidemicModeling/src/InfectionSim.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/POMDPsInterface.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/updater.jl")

pomdp = initParams(inf_loss=100.0, test_loss=1.0, testrate_loss=50.0)

function POMDPs.initialstate(pomdp::Params, rng::AbstractRNG=Random.GLOBAL_RNG)
     return ImplicitDistribution(initState, Normal(5_000,2500), pomdp)
end

solver = POMCPOWSolver(criterion=MaxUCB(1.0), max_depth=50, tree_queries=5_000)
planner = POMCPOW.solve(solver, pomdp)
# upd = updater(planner)
belief_updater = BootstrapFilter(pomdp, 10*planner.solver.tree_queries)

np = 10*planner.solver.tree_queries
particles = [rand(initialstate(pomdp)) for _ in 1:np]
b0 = ParticleCollection(particles)

s0 = rand(initialstate(pomdp))

b = b0
s = s0
beliefHist = [b.particles]
stateHist = [s]
obsHist = []
rewardHist = []
actionHist = []
for i in tqdm(1:50)
        a = POMDPs.action(planner, b)
        (s,o,r) = @gen(:sp,:o,:r)(pomdp, s, a, Random.GLOBAL_RNG)
        b = update(upd, b, a, o)
        push!(beliefHist, copy(b.particles))
        push!(obsHist, o)
        push!(stateHist, copy(s))
        push!(rewardHist, r)
        push!(actionHist, a.testing_prop)
end

l = @layout [a;b]
p1 = plot(0:50,[sum(s.I)/pomdp.N for s in stateHist], label="True", legend=:bottomright)
plot!(p1,0:50, [sum(mean(states, pomdp).I)/pomdp.N for states in beliefHist], label="Estimated")
ylabel!("Infected Prop")
title!("POMCPOW Test")
p2 = plot(actionHist, label="", ylabel="Testing Prop")
plt = plot(p1, p2, layout=l)
xlabel!("Day")
display(plt)
