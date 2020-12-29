using DifferentialEquations, DiffEqParamEstim, Optim, LossFunctions

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


function SIR_loss(x, ref_data)
    sum((Array(x) .- ref_data).^2)
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

function initSEIR(simHist::SimHist)
    [simHist.sus[1], 0, simHist.inf[1], simHist.rec[1]]./simHist.N
end

function initSIR(simHist::SimHist)
    Array(simHist)[:, 1]./simHist.N
end