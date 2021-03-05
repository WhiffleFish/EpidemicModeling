using JuMP
using ParameterJuMP
using Ipopt
using ProgressBars
import Plots.plot
import Plots.plot!
# GLPK does not work with '==' constraints

# List of solvers - https://jump.dev/JuMP.jl/v0.21.1/installation/#Installation-Guide-1

#= Desired Structure
Instantiate MPC - how??
- Have predefined model
- Set IC and optimize control in same function! - MPC struct is IMMUTABLE

input State to MPC

Output
- full::Bool
    - true: output 2d array [S, I, R, control over PredHorizon]
    - false: output control vector of length ControlHorizon
=#

abstract type MPC end

"""
# Arguments
`model::JuMP.Model`
`PredHorizon::Int64` - Prediction Horizon
`ControlHorizon::Int64` - Control Horizon
`InfWeight::Float64` - Infection Weight: How much to penalize infectious population proportion over prediction horizon
`TestWeight::Float64` - Testing Weight: How much to penalize testing frequency over prediction horizon
`TestRateWeight::Float64` - Testing rate weight: How much to penalize changes in testing frequency over prediction horizon
`IC::Vector{ParameterRef}` - ParameterJuMP parameter initial conditions for MPC: `[S0, I0, R0]`
"""
struct SIR_MPC <: MPC
    model::JuMP.Model
    PredHorizon::Int64
    ControlHorizon::Int64
    InfWeight::Float64
    TestWeight::Float64
    TestRateWeight::Float64
    IC::Vector{ParameterRef}
end

"""
# Arguments
`model::JuMP.Model`
`PredHorizon::Int64` - Prediction Horizon
`ControlHorizon::Int64` - Control Horizon
`InfWeight::Float64` - Infection Weight: How much to penalize infectious population proportion over prediction horizon
`ExpWeight::Float64` - Exposed Weight: How much to penalize exposed population proportion over prediction horizon
`TestWeight::Float64` - Testing Weight: How much to penalize testing frequency over prediction horizon
`TestRateWeight::Float64` - Testing rate weight: How much to penalize changes in testing frequency over prediction horizon
`IC::Vector{ParameterRef}` - ParameterJuMP parameter initial conditions for MPC: `[S0, E0, I0, R0]`
"""
struct SEIR_MPC <: MPC
    model::JuMP.Model
    PredHorizon::Int64
    ControlHorizon::Int64
    InfWeight::Float64
    ExpWeight::Float64
    TestWeight::Float64
    TestRateWeight::Float64
    IC::Vector{ParameterRef}
end

"""
- `SIR_params::Vector{Float64}`
- `callback::Bool=true` (opt)
- `PredHorizon::Int64 = 20` (opt)
- `ControlHorizon::Int64 = 3` (opt)
- `InfWeight::Float64 = 30.` (opt)
- `TestWeight::Float64 = 0.5` (opt)
- `TestRateWeight::Float64 = 10.` (opt)
- `optimizer=Ipopt.Optimizer` (opt)
"""
function initSIR_MPC(SIR_params::Vector{Float64}; callback::Bool=true, PredHorizon::Int64 = 20, ControlHorizon::Int64 = 3,
    InfWeight::Float64 = 30., TestWeight::Float64 = 0.5, TestRateWeight::Float64 = 10., optimizer=Ipopt.Optimizer)::SIR_MPC

    α, β, δ = SIR_params
    if callback
        model = JuMP.Model(optimizer)
    else
        model = JuMP.Model(optimizer_with_attributes(optimizer, "print_level"=>0))
    end

    JuMP.@variables model begin
        S[1:PredHorizon]
        I[1:PredHorizon]
        R[1:PredHorizon]
        0 <= T[1:PredHorizon] <= 1
    end

    # Dynamic Constraints
    JuMP.@constraint(
        model,
        [i=2:PredHorizon], S[i] == S[i-1] - β*I[i-1]*S[i-1]
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon], I[i] == I[i-1] + β*I[i-1]*S[i-1] - (α+δ*T[i-1])*I[i-1]
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon], R[i] == R[i-1] + (α+δ*T[i-1])*I[i-1]
    )

    JuMP.@objective( # InfWeight ← 0 : Hard constraint
        model, Min,
        InfWeight*sum(I).^2 + TestWeight*sum(T).^2 + TestRateWeight*sum((T .- circshift(T,1))[2:end].^2)
    )

    S0 = @variable(model,S0 == 1.0, Param())
    I0 = @variable(model,I0 == 0.0, Param())
    R0 = @variable(model,R0 == 0.0, Param())

    # Initial Conditions
    JuMP.@constraint(model, S[1] == S0)
    JuMP.@constraint(model, I[1] == I0)
    JuMP.@constraint(model, R[1] == R0)

    return SIR_MPC(model, PredHorizon, ControlHorizon, InfWeight, TestRateWeight, TestRateWeight, [S0,I0,R0])
end


"""
- `SEIR_params::Vector{Float64}`
- `callback::Bool=true` (opt)
- `PredHorizon::Int64 = 20` (opt)
- `ControlHorizon::Int64 = 3` (opt)
- `InfWeight::Float64 = 30.` (opt)
- `TestWeight::Float64 = 0.5` (opt)
- `TestRateWeight::Float64 = 10.` (opt)
- `optimizer=Ipopt.Optimizer` (opt)
"""
function initSEIR_MPC(SEIR_params::Vector{Float64}; callback::Bool=true, PredHorizon::Int64 = 20, ControlHorizon::Int64 = 3,
    InfWeight::Float64=20., ExpWeight::Float64=20., TestWeight::Float64=0.5, TestRateWeight::Float64=10., optimizer=Ipopt.Optimizer)::SEIR_MPC

    α, β, γ, δ, ϵ = SEIR_params

    if callback
        model = JuMP.Model(optimizer)
    else
        model = JuMP.Model(optimizer_with_attributes(optimizer, "print_level"=>0))
    end

    dS(S,E,I,R,T) = -β*I*S
    dE(S,E,I,R,T) = β*I*S - (γ+ϵ*T)*E
    dI(S,E,I,R,T) = (γ+ϵ*T)*E - (α+δ*T)*I
    dR(S,E,I,R,T) = (α+δ*T)*I

    JuMP.@variables model begin
        S[1:PredHorizon]
        E[1:PredHorizon]
        I[1:PredHorizon]
        R[1:PredHorizon]
        0 <= T[1:PredHorizon] <= 1
    end

    JuMP.@constraint(
        model,
        [i=2:PredHorizon], S[i] == S[i-1] + dS(S[i-1],E[i-1],I[i-1],R[i-1],T[i-1])
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon], E[i] == E[i-1] + dE(S[i-1],E[i-1],I[i-1],R[i-1],T[i-1])
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon], I[i] == I[i-1] + dI(S[i-1],E[i-1],I[i-1],R[i-1],T[i-1])
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon], R[i] == R[i-1] + dR(S[i-1],E[i-1],I[i-1],R[i-1],T[i-1])
    )

    JuMP.@objective(
        model, Min,
        InfWeight*sum(I).^2 + ExpWeight*sum(E).^2 +TestWeight*sum(T).^2 + TestRateWeight*sum((T .- circshift(T,1))[2:end].^2)
    )

    S0 = @variable(model,S0 == 1.0, Param())
    E0 = @variable(model,E0 == 0.0, Param())
    I0 = @variable(model,I0 == 0.0, Param())
    R0 = @variable(model,R0 == 0.0, Param())

    # Initial Conditions
    JuMP.@constraint(model, S[1] == S0)
    JuMP.@constraint(model, E[1] == E0)
    JuMP.@constraint(model, I[1] == I0)
    JuMP.@constraint(model, R[1] == R0)

    return SEIR_MPC(model, PredHorizon, ControlHorizon, InfWeight, ExpWeight, TestRateWeight, TestRateWeight, [S0,E0,I0,R0])
end

function SetIC!(mpc::MPC, state::State)
    if mpc isa SIR_MPC
        S0,I0,R0 = mpc.IC
        set_value(S0, state.S/state.N)
        set_value(I0, sum(state.I)/state.N)
        set_value(R0, state.R/state.N)
    else
        S0,E0,I0,R0 = mpc.IC
        set_value(S0, state.S/state.N)
        set_value(E0, 0.)
        set_value(I0, sum(state.I)/state.N)
        set_value(R0, state.R/state.N)
    end
end

function OptimalAction(mpc::SIR_MPC, state::State)

    SetIC!(mpc, state)

    model = mpc.model

    optimize!(model)

    return JuMP.value.(model[:T])[1:mpc.ControlHorizon]
end

function OptimalAction(mpc::SEIR_MPC, state::State)

    SetIC!(mpc, state)

    model = mpc.model

    optimize!(model)

    return JuMP.value.(model[:T])[1:mpc.ControlHorizon]
end

function MPCSimulate(T::Int, state::State, params::Params, mpc::MPC)
    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    incidentHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = zeros(Float64,T)

    actions = nothing
    for day in ProgressBar(1:T)

        if ((day-1) % mpc.ControlHorizon) == 0
            actions = OptimalAction(mpc, state) |> reverse
        end

        action = Action(pop!(actions))

        susHist[day] = state.S
        infHist[day] = sum(state.I)
        recHist[day] = state.R
        actionHist[day] = action.testing_prop

        state, new_infections, pos_tests = SimStep(state, params, action, state_only=false)
        incidentHist[day] = new_infections
        testHist[day] = sum(pos_tests)
    end

    return SimHist(susHist, infHist, recHist, state.N, T, incidentHist, testHist), actionHist
end


# ---------------------------------------------------------------------------- #
#                                   Plotting                                   #
# ---------------------------------------------------------------------------- #


function plot(mpc::MPC, var::Symbol)
    yLabelDict = Dict(
        :S=>"Susceptible Proportion", :E=>"Exposed Proportion",
        :I=>"Infected Proportion", :R=>"Recovered Proportion", :T=>"Testing Strength"
    ) # "testing strength" is ambiguous -> change
    if var ∉ keys(yLabelDict)
        error("Available var names are :S,:E,:I,:R,:T")
    end

    plot(JuMP.value.(mpc.model[var]), label="")
    xlabel!("Time (days)")
    ylabel!(yLabelDict[var])
end

function plot!(mpc::MPC, var::Symbol)
    yLabelDict = Dict(
        :S=>"Susceptible Proportion", :E=>"Exposed Proportion",
        :I=>"Infected Proportion", :R=>"Recovered Proportion", :T=>"Testing Strength"
    ) # "testing strength" is ambiguous -> change
    if var ∉ keys(yLabelDict)
        error("Available var names are :S,:E,:I,:R,:T")
    end

    plot!(JuMP.value.(mpc.model[var]), label=yLabelDict[var])
end
