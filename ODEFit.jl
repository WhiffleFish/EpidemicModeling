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
    return solve(prob)
end

"""
# Arguments
- `x` - Output Data
- `ref_data` - Reference Data
"""
function SIR_loss(x, ref_data)
    sum((Array(x) .- ref_data).^2)
end

function SIR_loss_grad(x, ref_data)
    2*sum((Array(x) .- ref_data))
end


# function SEIR_loss(x, ref_data, L::Array{Float64,2}) # with cache
#     L[1,:] = x[1,:]
#     L[2,:] = sum(x[2:3,:], dims=1)[1,:]
#     L[3,:] = x[4,:]
#     sum((L .- ref_data).^2)
# end


function SEIR_loss(x, ref_data) # without cache
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
        u0 = ref[:,1] # replace with initSIR
        p = ones(Float64,2)
        prob = ODEProblem(SIR_ODE, u0, (1. ,Float64(T)), p, saveat=data_times)
        objective = build_loss_objective(prob, Tsit5(), x -> SIR_loss(x, ref));
    elseif kind == :SEIR
        u0 = [1. - ref[2,1]-ref[3,1], 0, ref[2,1], ref[3,1]] # replace with initSEIR
        p = ones(Float64,3)
        prob = ODEProblem(SEIR_ODE, u0, (1. ,Float64(T)), p, saveat=data_times)
        objective = build_loss_objective(prob, Tsit5(), x->SEIR_loss(x,ref));
    else
        throw(DomainError("Unrecognized Model kind. Must be :SIR or :SEIR"))
    end

    result = optimize(objective, lb, ub, p, Fminbox(BFGS()))
    
    return result, Optim.minimizer(result)
end

#=
N = 10;
initial_conditions = [[1.0,1.0], [1.0,1.5], [1.5,1.0], [1.5,1.5], [0.5,1.0], [1.0,0.5], [0.5,0.5], [2.0,1.0], [1.0,2.0], [2.0,2.0]]
function prob_func(prob,i,repeat)
  ODEProblem(prob.f,initial_conditions[i],prob.tspan,prob.p)
end
monte_prob = MonteCarloProblem(prob, prob_func=prob_func)

data_times = 0:0.1:10
sim = solve(monte_prob, Tsit5(), trajectories=N, saveat=data_times)
data = Array(sim)

losses = [L2Loss(data_times,data[:,:,i]) for i in 1:N]

loss(sim) = sum(losses[i](sim[i]) for i in 1:N)

obj = build_loss_objective(monte_prob, Tsit5(), loss, trajectories=N, saveat=data_times)

lower = zeros(2)
upper = fill(2.0,2)

result = optimize(obj, lower, upper, [1.3,0.9], Fminbox(BFGS()))
=#
"""
Fit parameters to multiple random simulation outputs
# Arguments
- `kind::Symbol` - Type of differential model to fit (`:SIR` or `:SEIR`)
- `T::Int` - Simulation Time (days)
- `trajectories::Int64` - Number of random simulations to fit
- `params::Params` - Simulation Parameters
- `action::Action` - Simulation Action
- `lb::Array{Float64,1}` - Parameter search lower bound
- `ub::Array{Float64,1}` - Parameter search upper bound
"""
function FitRandEnsemble(kind::Symbol, T::Int, trajectories::Int64, params::Params, action::Action, lb::Array{Float64,1}, ub::Array{Float64,1})

    sims = SimulateEnsemble(T, trajectories, params, action, N=1_000_000)
    data_times = 1:T
    ref_data = Array(sims)./1_000_000

    if kind == :SIR
        @assert length(lb) == length(ub) == 2
        first_guess = [5.,5.]
        initial_conditions = [initSIR(sim) for sim in sims]
        prob = ODEProblem(SIR_ODE, initial_conditions[1], (1.,Float64(T)), first_guess)

        prob_func(prob, i, repeat) = ODEProblem(SIR_ODE, initial_conditions[i], (1.,Float64(T)), first_guess)
        loss_func(x) = SIR_loss(x, ref_data)
    
    elseif kind == :SEIR
        @assert length(lb) == length(ub) == 3
        first_guess = [1., 1., 1.]
        initial_conditions = [initSEIR(sim) for sim in sims]

        prob = ODEProblem(SEIR_ODE, initial_conditions[1], (1.,Float64(T)), first_guess)
        
        prob_func(prob, i, repeat) = ODEProblem(SEIR_ODE, initial_conditions[i], (1.,Float64(T)), first_guess)
        
        loss_func(x) = SEIR_loss(x, ref_data)

    else
        throw(DomainError("Unrecognized model kind. Must be :SIR or :SEIR"))
    end
    
    # SHOULD BE MOVED AFTER IF ELSE
    monte_prob = MonteCarloProblem(prob, prob_func=prob_func)
    obj = build_loss_objective(
        monte_prob, Tsit5(), loss_func, verbose_opt=true,
        trajectories=trajectories, saveat=data_times
    )

    result = optimize(obj, lb, ub, first_guess, Fminbox(BFGS()))
    
    return result, Optim.minimizer(result)
end

function EnsembleFitSIR(T::Int, trajectories::Int64, params::Params, action::Action, lb::Array{Float64,1}, ub::Array{Float64,1})
    
end


function initSEIR(simHist::SimHist)
    [simHist.sus[1], 0, simHist.inf[1], simHist.rec[1]]./simHist.N
end

function initSIR(simHist::SimHist)
    Array(simHist)[:, 1]./simHist.N
end