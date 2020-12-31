using DifferentialEquations, DiffEqParamEstim, Optim, LossFunctions
include("InfectionSim.jl")

SIR_ODE = @ode_def SIR begin
    dS = -β*I*S
    dI = β*I*S - α*I
    dR = α*I
end α β


SEIR_ODE = @ode_def SEIR begin
    dS = -β*I*S
    dE = β*I*S - γ*E
    dI = γ*E - α*I
    dR = α*I
end α β γ


"""
# Arguments
- `kind::Symbol`
- `u0::Array{Float64, 1}`
- `T::Int`
- `p::Union{Array{Float64,1}, Tuple}`
"""
function SolveODE(kind::Symbol, u0::Array{Float64, 1}, T::Int, p::Union{Array{Float64,1}, Tuple})
    if kind == :SIR
        @assert length(u0) == 3 && length(p) == 2
        prob = ODEProblem(SIR_ODE, u0, (1. , Float64(T)), p)
    elseif kind == :SEIR
        @assert length(u0) == 4 && length(p) == 3
        prob = ODEProblem(SEIR_ODE, u0, (1. , Float64(T)), p)
    else
        throw(DomainError("Unrecognized model kind. Must be :SIR or :SEIR"))
    end
    return solve(prob, saveat=1:T)
end


"""
Get initial SIR state (in proportions of pop) of given Simulation History
"""
function initSIR(simHist::SimHist)
    Array(simHist)[:, 1]./simHist.N
end

"""
Get initial SEIR state (in proportions of pop) of given Simulation History
"""
function initSEIR(simHist::SimHist)
    [simHist.sus[1], 0, simHist.inf[1], simHist.rec[1]]./simHist.N
end

"""
2-D Array L2 SIR Loss
# Arguments
- `x` - Output Data
- `ref_data` - Reference Data
"""
function SIR_loss(x, ref_data)
    sum((Array(x) .- ref_data).^2)
end

"""
2-D Array L2 SEIR Loss
# Arguments
- `x` - Output Data
- `ref_data` - Reference Data
"""
function SEIR_loss(x, ref_data)
    L = zeros(3,size(ref_data)[2])
    L[1,:] = x[1,:]
    L[2,:] = sum(x[2:3,:], dims=1)[1,:]
    L[3,:] = x[4,:]
    sum((L .- ref_data).^2)
end


"""
# Arguments
- `kind::Symbol`
- `simHist::SimHist` 
- `lb::Array{Float64,1}` 
- `ub::Array{Float64,1}`
"""
function FitModel(kind::Symbol, simHist::SimHist, lb::Array{Float64,1}, ub::Array{Float64,1})
    ref = Array(simHist)./simHist.N
    T = simHist.T
    data_times = 1:T

    if kind == :SIR
        u0 = ref[:,1]
        p = ones(Float64,2)
        prob = ODEProblem(SIR_ODE, u0, (1. ,Float64(T)), p, saveat=data_times)
        objective = build_loss_objective(prob, Tsit5(), x -> SIR_loss(x, ref));
    elseif kind == :SEIR
        u0 = initSEIR(simHist)
        p = ones(Float64,3)
        prob = ODEProblem(SEIR_ODE, u0, (1. ,Float64(T)), p, saveat=data_times)
        objective = build_loss_objective(prob, Tsit5(), x->SEIR_loss(x,ref));
    else
        throw(DomainError("Unrecognized Model kind. Must be :SIR or :SEIR"))
    end

    result = optimize(objective, lb, ub, p, Fminbox(BFGS()))
    
    return result, Optim.minimizer(result)
end



#=--------------------------------------------------------------------------------------
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ENSEMBLE FITTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--------------------------------------------------------------------------------------=#


"""
Fit parameters to multiple random simulation outputs
# Arguments
- `kind::Symbol` - Type of differential model to fit (`:SIR` or `:SEIR`)
- `T::Int` - Simulation Time (days)
- `trajectories::Int64` - Number of random simulations to fit
- `params::Params` - Simulation Parameters
- `action::Action` - Simulation Action
"""
function FitRandEnsemble(kind::Symbol, T::Int, trajectories::Int64, params::Params, action::Action)

    sims = SimulateEnsemble(T, trajectories, params, action, N=1_000_000)
    data_times = 1:T
    ref_data = Array(sims)./1_000_000

    if kind == :SIR
        first_guess = [0.1,0.1]
        initial_conditions = [initSIR(sim) for sim in sims]

        LossCalcParams = Dict(:IC=>initial_conditions, :T=>T, :ref=>ref_data)

        result = optimize(x->SIR_param_loss(x,LossCalcParams), first_guess, NelderMead(),Optim.Options(show_trace=false, show_every=10))
    
    elseif kind == :SEIR
        first_guess = [0.1, 0.1, 0.1]
        initial_conditions = [initSEIR(sim) for sim in sims]

        LossCalcParams = Dict(:IC=>initial_conditions, :T=>T, :ref=>ref_data)

        result = optimize(x->SEIR_param_loss(x,LossCalcParams), first_guess, NelderMead(),Optim.Options(show_trace=true, show_every=10))

    else
        throw(DomainError("Unrecognized model kind. Must be :SIR or :SEIR"))
    end

    return result, Optim.minimizer(result)
end


function SIR_param_loss(p::Vector{Float64}, LossCalcParams::Dict)
    loss = 0.
    for i in 1:length(LossCalcParams[:IC])
        data = Array(SolveODE(:SIR, LossCalcParams[:IC][i],LossCalcParams[:T], p))
        loss += sum((data .- LossCalcParams[:ref][:,:,i]).^2)
    end
    return loss
end

function SEIR_param_loss(p::Vector{Float64}, LossCalcParams::Dict)
    loss = 0.
    for i in 1:length(LossCalcParams[:IC])
        data = Array(SolveODE(:SEIR,LossCalcParams[:IC][i],LossCalcParams[:T], p))
        loss += sum(@. (data[[1,4],:] - LossCalcParams[:ref][[1,3],:,i])^2 )
        loss += sum(@. (data[2,:] + data[3,:] - LossCalcParams[:ref][2,:,i])^2 ) 
    end
    return loss
end
