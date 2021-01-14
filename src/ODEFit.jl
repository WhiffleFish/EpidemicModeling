using DifferentialEquations, DiffEqParamEstim, Optim, LossFunctions
include("InfectionSim.jl")

SIR_ODE = @ode_def SIR begin
    dS = -β*I*S
    dI = β*I*S - α*I
    dR = α*I
end α β

# Create Modified ODE where α ← α + δT   where T is testing rate and δ is proportional effectiveness of testing
SIR_CTRL = @ode_def SIR_C begin
    dS = -β*I*S
    dI = β*I*S - (α+δ*T)*I
    dR = (α+δ*T)*I
    dT = 0 # Testing Proportion (does not change over ODE sol)
end α β δ


SEIR_ODE = @ode_def SEIR begin
    dS = -β*I*S
    dE = β*I*S - γ*E
    dI = γ*E - α*I
    dR = α*I
end α β γ

SEIR_CTRL = @ode_def SEIR_C begin
    dS = -β*I*S
    dE = β*I*S - (γ+ϵ*T)*E
    dI = (γ+ϵ*T)*E - (α+δ*T)*I
    dR = (α+δ*T)*I
    dT = 0 # Testing Proportion (does not change over ODE sol)
end α β γ δ ϵ


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
    elseif kind == :SIR_CTRL # SIR w/ control input
        @assert length(u0) == 4 && length(p) == 3
        prob = ODEProblem(SIR_CTRL, u0, (1. , Float64(T)), p)
    elseif kind == :SEIR
        @assert length(u0) == 4 && length(p) == 3
        prob = ODEProblem(SEIR_ODE, u0, (1. , Float64(T)), p)
    elseif kind == :SEIR_CTRL
        @assert length(u0) == 5 && length(p) == 5
        prob = ODEProblem(SEIR_CTRL, u0, (1. , Float64(T)), p)
    else
        throw(DomainError("Unrecognized model kind. Must be :SIR, :SIR_CTRL, :SEIR, or :SEIR_CTRL"))
    end
    return DifferentialEquations.solve(prob, saveat=1:T)
end


"""
Get initial SIR state (in proportions of pop) of given Simulation History
# Arguments
- simHist::simHist
"""
function initSIR(simHist::SimHist)::Vector{Float64}
    Array(simHist)[:, 1]./simHist.N
end

"""
Get initial SIR state (in proportions of pop) of given State
# Arguments
- state::State
"""
function initSIR(state::State)::Vector{Float64}
    Array(state)./simHist.N
end

"""
Get initial SEIR state (in proportions of pop) of given Simulation History
# Arguments
- simHist::simHist
"""
function initSEIR(simHist::SimHist)::Vector{Float64}
    [simHist.sus[1], 0, simHist.inf[1], simHist.rec[1]]./simHist.N
end

"""
Get initial SEIR state (in proportions of pop) of given State
# Arguments
- state::State
"""
function initSEIR(state::State)::Vector{Float64}
    [state.S, 0.0, state.I, state.R]./state.N
end

"""
2-D Array L2 SIR Loss
# Arguments
- `x` - Output Data
- `ref_data` - Reference Data
"""
function SIR_loss(x, ref_data)::Float64
    sum(abs2, Array(x) .- ref_data)
end

"""
2-D Array L2 SEIR Loss
# Arguments
- `x` - Output Data
- `ref_data` - Reference Data
"""
function SEIR_loss(x, ref_data)::Float64
    L = zeros(3,size(ref_data)[2])
    L[1,:] = x[1,:]
    L[2,:] = sum(x[2:3,:], dims=1)[1,:]
    L[3,:] = x[4,:]
    sum(abs2, L .- ref_data)
end


"""
# Arguments
- `kind::Symbol`
- `simHist::SimHist`
"""
function FitModel(kind::Symbol, simHist::SimHist)
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

    result = optimize(objective, p, NelderMead())

    return result, Optim.minimizer(result)
end



# ---------------------------------------------------------------------------- #
#                               ENSEMBLE FITTING                               #
# ---------------------------------------------------------------------------- #


"""
Fit SIR or SEIR model parameters to an ensemble of stochastic simulations
# Arguments
- `kind::Symbol` - Type of differential model to fit (`:SIR` or `:SEIR`)
- `T::Int` - Simulation Time (days)
- `trajectories::Int64` - Number of random simulations to fit
- `params::Params` - Simulation Parameters
- `action::Action` - Simulation Action
- `show_trace::Bool = false` (opt) - print live optimization status
"""
function FitRandEnsemble(kind::Symbol, T::Int, trajectories::Int64, params::Params, action::Action; show_trace::Bool=false)

    sims = SimulateEnsemble(T, trajectories, params, action, N=1_000_000)
    data_times = 1:T
    ref_data = Array(sims)./1_000_000

    if kind == :SIR
        first_guess = [0.1,0.1]
        initial_conditions = [initSIR(sim) for sim in sims]

        LossCalcParams = Dict(:IC=>initial_conditions, :T=>T, :ref=>ref_data)

        result = optimize(
            x->SIR_param_loss(x,LossCalcParams),
            first_guess,
            NelderMead(),
            Optim.Options(show_trace=show_trace, show_every=50)
        )

    elseif kind == :SEIR
        first_guess = [0.1, 0.1, 0.1]
        initial_conditions = [initSEIR(sim) for sim in sims]

        LossCalcParams = Dict(:IC=>initial_conditions, :T=>T, :ref=>ref_data)

        result = optimize(
            x->SEIR_param_loss(x,LossCalcParams),
            first_guess,
            NelderMead(),
            Optim.Options(show_trace=show_trace, show_every=50)
        )

    else
        throw(DomainError("Unrecognized model kind. Must be :SIR or :SEIR"))
    end

    return result, Optim.minimizer(result)
end


"""
Fit SIR or SEIR model parameters to an ensemble of stochastic simulations
# Arguments
- `kind::Symbol` - Type of differential model to fit (`:SIR` or `:SEIR`)
- `T::Int` - Simulation Time (days)
- `trajectories::Int64` - Number of random simulations to fit
- `params::Params` - Simulation Parameters
- `show_trace::Bool = false` (opt) - print live optimization status
"""
function FitRandControlledEnsemble(kind::Symbol, T::Int, trajectories::Int64, params::Params; show_trace::Bool=false)

    actions = rand(trajectories) .|> Action

    sims = SimulateEnsemble(T, trajectories, params, actions, N=1_000_000)
    data_times = 1:T
    ref_data = Array(sims)./1_000_000

    if kind == :SIR
        first_guess = [0.1, 0.1, 0.1]
        initial_conditions = [vcat(initSIR(sim),actions[i].testing_prop) for (i,sim) in enumerate(sims)]

        LossCalcParams = Dict(:IC=>initial_conditions, :T=>T, :ref=>ref_data)

        result = optimize(
            x->SIR_CTRL_param_loss(x, LossCalcParams),
            first_guess,
            NelderMead(),
            Optim.Options(show_trace=show_trace, show_every=50)
        )

    elseif kind == :SEIR
        first_guess = [0.1, 0.1, 0.1, 1., 1.]
        initial_conditions = [vcat(initSEIR(sim),actions[i].testing_prop) for (i,sim) in enumerate(sims)]

        LossCalcParams = Dict(:IC=>initial_conditions, :T=>T, :ref=>ref_data)

        result = optimize(
            x->SEIR_CTRL_param_loss(x, LossCalcParams),
            first_guess,
            NelderMead(),
            Optim.Options(show_trace=show_trace, show_every=50)
        )

    else
        throw(DomainError("Unrecognized model kind. Must be :SIR or :SEIR"))
    end

    return result, Optim.minimizer(result)
end


"""
# Arguments
- `p::Vector{Float64}` - Vector of parameters ``\\alpha, \\beta`` for SIR ODE
- `LossCalcParams::Dict`
"""
function SIR_param_loss(p::Vector{Float64}, LossCalcParams::Dict)::Float64
    loss = 0.
    for i in 1:length(LossCalcParams[:IC])
        data = Array(SolveODE(:SIR, LossCalcParams[:IC][i],LossCalcParams[:T], p))
        loss += sum(abs2, data .- LossCalcParams[:ref][:,:,i])
    end

    # Scale loss by number of MC sims s.t. loss calculated from different number of MC sims is comparable
    return loss/length(LossCalcParams[:IC])
end

"""
# Arguments
- `p::Vector{Float64}` - Vector of parameters ``\\alpha, \\beta`` for SIR ODE
- `LossCalcParams::Dict`
"""
function SIR_CTRL_param_loss(p::Vector{Float64}, LossCalcParams::Dict)::Float64
    loss = 0.
    for i in 1:length(LossCalcParams[:IC])
        data = Array(SolveODE(:SIR_CTRL, LossCalcParams[:IC][i], LossCalcParams[:T], p))
        loss += sum(abs2, data[1:3,:] .- LossCalcParams[:ref][:,:,i])
    end

    # Scale loss by number of MC sims s.t. loss calculated from different number of MC sims is comparable
    return loss/length(LossCalcParams[:IC])
end

"""
# Arguments
- `p::Vector{Float64}` - Vector of parameters ``\\alpha, \\beta, \\gamma `` for SEIR ODE
- `LossCalcParams::Dict`
"""
function SEIR_param_loss(p::Vector{Float64}, LossCalcParams::Dict)::Float64
    loss = 0.
    for i in 1:length(LossCalcParams[:IC])
        data = Array(SolveODE(:SEIR,LossCalcParams[:IC][i],LossCalcParams[:T], p))
        loss += sum(abs2, data[[1,4],:] .- LossCalcParams[:ref][[1,3],:,i] )
        loss += sum(abs2, data[2,:] + data[3,:] .- LossCalcParams[:ref][2,:,i] )
    end

    # Scale loss by number of MC sims s.t. loss calculated from different number of MC sims is comparable
    return loss/length(LossCalcParams[:IC])
end

"""
# Arguments
- `p::Vector{Float64}` - Vector of parameters ``\\alpha, \\beta, \\gamma `` for SEIR ODE
- `LossCalcParams::Dict`
"""
function SEIR_CTRL_param_loss(p::Vector{Float64}, LossCalcParams::Dict)::Float64
    loss = 0.
    for i in 1:length(LossCalcParams[:IC])
        data = Array(SolveODE(:SEIR_CTRL, LossCalcParams[:IC][i], LossCalcParams[:T], p))
        loss += sum(abs2, data[[1,4],:] .- LossCalcParams[:ref][[1,3],:,i] )
        loss += sum(abs2, data[2,:] + data[3,:] .- LossCalcParams[:ref][2,:,i] )
    end

    # Scale loss by number of MC sims s.t. loss calculated from different number of MC sims is comparable
    return loss/length(LossCalcParams[:IC])
end


"""
Convert controlled/uncontrolled SIR/SEIR models to pure SIR arrays
# Arguments
- `sol` - Output of DifferentialEquations.jl solver
"""
function toSIR(sol)::Array{Float64,2}
    syms = sol.prob.f.syms

    if syms == [:S, :I, :R]
        return Array(sol)

    elseif syms == [:S, :I, :R, :T]
        return sol[1:3,:]

    elseif syms == [:S, :E, :I, :R]
        arr = zeros(3, size(sol,2))
        arr[[1,3],:] = sol[[1,4],:]
        arr[2,:] = sol[2,:] .+ sol[3,:]
        return arr

    elseif syms == [:S, :E, :I, :R, :T]
        arr = zeros(3, size(sol,2))
        arr[[1,3],:] = sol[[1,4],:]
        arr[2,:] = sol[2,:] .+ sol[3,:]
        return arr
    else
        error("Unknown Compartmental Solution Type")
    end
end
