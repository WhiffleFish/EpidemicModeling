include("/Users/tyler/Documents/code/EpidemicModeling/src/InfectionSim.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/ODEFit.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/MPC.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/POMDPsInterface.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/updater.jl")

pomdp = initParams(inf_loss=100.0, test_loss=1.0, testrate_loss=40.0, test_period=7)

function POMDPs.initialstate(pomdp::Params, rng::AbstractRNG=Random.GLOBAL_RNG)
     return ImplicitDistribution(initState, Normal(9_000,4_000), pomdp)
end

s0 = rand(initialstate(pomdp))
b0 = initialbelief(pomdp, 50_000)
T = 100

## Simulate MPC

SEIRres, SEIRp = FitRandControlledEnsemble(:SEIR, 30, 500, pomdp, show_trace=true)

SEIRmpc = initSEIR_MPC(
    SEIRp,
    pomdp,
    callback=false,
    ControlHorizon=1,
    PredHorizon=75
)

MPChist = Simulate(T, copy(s0), b0, pomdp, SEIRmpc)
plot(pomdp, MPChist)

## Simulate POMCPOW

solver = POMCPOWSolver(criterion=MaxUCB(1.0), max_depth=50, tree_queries=5_000)
planner = POMCPOW.solve(solver, pomdp)

POMCPOWhist = Simulate(T, copy(s0), b0, pomdp, planner)
plot(pomdp, POMCPOWhist)


## Plot Results

l = @layout [a;b]
p1 = plot(0:T-1, MPChist.inf ./MPChist.N, label="MPC", ylabel = "Infected Prop")
plot!(p1, 0:T-1, POMCPOWhist.inf ./POMCPOWhist.N, label="POMCPOW")
ylabel!("Infected Prop")
title!("MPC/POMCPOW Comparison")
p2 = plot(0:T-1,[a.testing_prop for a in MPChist.actions], label="", ylabel="Testing Prop")
plot!(0:T-1,[a.testing_prop for a in POMCPOWhist.actions], label="")
plt = plot(p1, p2, layout=l)
xlabel!("Day")
display(plt)

plot(MPChist.rewards, label="MPC", legend=:bottomright)
plot!(POMCPOWhist.rewards, label="POMCPOW")
xlabel!("Day")
title!("Daily Reward")

plot(cumsum(MPChist.rewards), label="MPC", legend=:topright)
plot!(cumsum(POMCPOWhist.rewards), label="POMCPOW")
xlabel!("Day")
title!("Cumulative Reward")

println("MPC Accumulated Rewards: ",sum(MPChist.rewards))
println("POMCPOW Accumulated Rewards: ",sum(POMCPOWhist.rewards))
