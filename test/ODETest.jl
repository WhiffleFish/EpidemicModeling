println("Importing...")
include("/Users/tyler/Documents/code/EpidemicModeling/src/InfectionSim.jl")
include("/Users/tyler/Documents/code/EpidemicModeling/src/ODEFit.jl")


T = 20 # Sim Time
N = 10 # Number of Ensemble MC Sims to fit
param = initParams(
    symptomatic_isolation_prob = 0.95,
    asymptomatic_prob = 0.65
)

# ---------------------------------------------------------------------------- #
#                                Single SIR Fit                                #
# ---------------------------------------------------------------------------- #

println("\nSINGLE SIR FIT...\n")

state = initState(fill(10_000,14),param, N=1_000_000)
action = Action(0.0)
simHist_SingleSIR = Simulate(T, state, param, action)

res, p = FitModel(:SIR, simHist_SingleSIR)

sol = SolveODE(:SIR, initSIR(simHist_SingleSIR), simHist_SingleSIR.T, p);

SingleSIRPlot = plot(Array(simHist_SingleSIR)'./simHist_SingleSIR.N, labels= ["True Sus" "True Inf" "TrueRec"])
plot!(sol, ls=:dash, label=["Pred Sus" "Pred Inf" "Pred Rec"])
title!("SIR Single Fit")

display(SingleSIRPlot)
println(res)


# ---------------------------------------------------------------------------- #
#                                Single SEIR Fit                               #
# ---------------------------------------------------------------------------- #

println("\nSINGLE SEIR FIT...\n")

state = initState(param, N=1_000_000)
action = Action(0.0)
simHist = Simulate(T, state, param, action)

res, p = FitModel(:SEIR, simHist)

sol = SolveODE(:SEIR, initSEIR(simHist), simHist.T, p);

SingleSEIRPlot = plot(Array(simHist)'./simHist.N, labels= ["True Sus" "True Inf" "TrueRec"])
plot_arr = sol[[1,3,4],:]
plot_arr[2,:] += sol[2,:]
plot!(plot_arr', ls=:dash, label=["Pred Sus" "Pred Inf" "Pred Rec"])
title!("SEIR Single Fit")

display(SingleSEIRPlot)

println(res)

# ---------------------------------------------------------------------------- #
#                               Ensemble SIR Fit                               #
# ---------------------------------------------------------------------------- #

println("\nENSEMBLE SIR FIT...\n")

# Generate Data With random starting parameters
simHist = Simulate(T, initState(param), param, action)

res, p = FitRandEnsemble(:SIR, T, N, param, action)

EnsembleSIRPlot = plot(simHist.inf./simHist.N, label= "True Infected")
sol = Array(SolveODE(:SIR,initSIR(simHist),T,p))
plot!(sol[2,:], label="Predicted Infected")
title!("SIR Ensemble Fit")

display(EnsembleSIRPlot)

println(res)

# ---------------------------------------------------------------------------- #
#                               Ensemble SEIR Fit                              #
# ---------------------------------------------------------------------------- #

println("\nENSEMBLE SEIR FIT...\n")

res, p = FitRandEnsemble(:SEIR, T, N, param, action)

res, p = FitRandEnsemble(:SEIR, T, N, param, action)

EnsembleSEIRPlot = plot(simHist.inf./simHist.N, label="True Infected")
sol = Array(SolveODE(:SEIR,initSEIR(simHist),T,p))
plot!(sol[2,:] + sol[3,:], label="Predicted Infected")
title!("SEIR Ensemble Fit")

display(EnsembleSEIRPlot)

println(res)