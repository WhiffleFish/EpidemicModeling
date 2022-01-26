# GLPK does not work with '==' constraints

# List of solvers - https://jump.dev/JuMP.jl/v0.21.1/installation/#Installation-Guide-1

#= Desired Structure
Instantiate MPC
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
function MPC(
    SIR_params::NTuple{3,Float64};
    callback::Bool=true,
    PredHorizon::Int64 = 20,
    ControlHorizon::Int64 = 3,
    InfWeight::Float64 = 30.,
    TestWeight::Float64 = 0.5,
    TestRateWeight::Float64 = 10.,
    optimizer=Ipopt.Optimizer,
    test_period::Int=1
    )::SIR_MPC
    # NOTE: if PredHorizon=1, we should really be tracking 2 states (s0,s1), as just tracking s0 would not make sense

    α, β, δ = SIR_params
    if callback
        model = JuMP.Model(optimizer)
    else
        model = JuMP.Model(optimizer_with_attributes(optimizer, "print_level"=>0))
    end

    n_actions = ceil(Int, PredHorizon/test_period)
    action_counts = repeat([test_period],n_actions)
    action_counts[end] = n_actions*test_period - PredHorizon

    JuMP.@variables model begin
        S[1:PredHorizon]
        I[1:PredHorizon]
        R[1:PredHorizon]
        0 <= T[1:n_actions] <= 1
    end

    a_ind(k) = ceil(Int, k/test_period)

    # Dynamic Constraints
    JuMP.@constraint(
        model,
        [i=2:PredHorizon], S[i] == S[i-1] - β*I[i-1]*S[i-1]
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon], I[i] == I[i-1] + β*I[i-1]*S[i-1] - (α+δ*T[a_ind(i-1)])*I[i-1]
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon], R[i] == R[i-1] + (α+δ*T[a_ind(i-1)])*I[i-1]
    )

    a0 = @variable(model, a0)
    @variable(model, a0_const)
    @constraint(model, a0con, a0 == a0_const)
    fix(a0_const, 0.0)

    JuMP.@objective( # InfWeight ← 0 : Hard constraint
        model, Min,
        InfWeight*sum(I.^2) +
        TestWeight*sum((T .* action_counts).^2) +
        TestRateWeight*(sum((T .- circshift(T,1))[2:end].^2) + (T[1]-a0)^2)
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

function MPC(
    SIR_params::NTuple{3,Float64},
    params::Params;
    callback::Bool=true,
    PredHorizon::Int64 = 20,
    ControlHorizon::Int64 = 3,
    optimizer=Ipopt.Optimizer)::SIR_MPC

    MPC(
        SIR_params,
        callback = callback,
        PredHorizon = PredHorizon,
        ControlHorizon = ControlHorizon,
        InfWeight = params.inf_loss,
        TestWeight = params.test_loss,
        TestRateWeight = params.testrate_loss,
        optimizer=optimizer,
        test_period=params.test_period
    )
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
function MPC(
    SEIR_params::NTuple{5,Float64};
    callback::Bool = true,
    PredHorizon::Int64 = 20,
    ControlHorizon::Int64 = 3,
    InfWeight::Float64=20.,
    ExpWeight::Float64=20.,
    TestWeight::Float64=0.5,
    TestRateWeight::Float64=10.,
    optimizer=Ipopt.Optimizer,
    test_period::Int=1)::SEIR_MPC

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

    n_actions = ceil(Int, PredHorizon/test_period)
    action_counts = repeat([test_period],n_actions)
    action_counts[end] = n_actions*test_period - PredHorizon

    JuMP.@variables model begin
        S[1:PredHorizon]
        E[1:PredHorizon]
        I[1:PredHorizon]
        R[1:PredHorizon]
        0 <= T[1:n_actions] <= 1
    end

    a_ind(k) = ceil(Int, k/test_period)

    JuMP.@constraint(
        model,
        [i=2:PredHorizon],
        S[i] == S[i-1] + dS(S[i-1],E[i-1],I[i-1],R[i-1],T[a_ind(i-1)])
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon],
        E[i] == E[i-1] + dE(S[i-1],E[i-1],I[i-1],R[i-1],T[a_ind(i-1)])
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon],
        I[i] == I[i-1] + dI(S[i-1],E[i-1],I[i-1],R[i-1],T[a_ind(i-1)])
    )

    JuMP.@constraint(
        model,
        [i=2:PredHorizon],
        R[i] == R[i-1] + dR(S[i-1],E[i-1],I[i-1],R[i-1],T[a_ind(i-1)])
    )

    a0 = @variable(model, a0)
    @variable(model, a0_const)
    @constraint(model, a0con, a0 == a0_const)
    fix(a0_const, 0.0)

    JuMP.@objective(
        model, Min,
            InfWeight*sum(I.^2) +
            ExpWeight*sum(E.^2) +
            TestWeight*sum((T .* action_counts).^2) +
            TestRateWeight*(sum((T .- circshift(T,1))[2:end].^2) + (T[1]-a0)^2)
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

function MPC(
    SEIR_params::NTuple{5,Float64},
    params::Params;
    callback::Bool=true,
    PredHorizon::Int64 = 20,
    ControlHorizon::Int64 = 3,
    optimizer=Ipopt.Optimizer)::SEIR_MPC

    return MPC(
        SEIR_params,
        callback = callback,
        PredHorizon = PredHorizon,
        ControlHorizon = ControlHorizon,
        InfWeight = params.inf_loss,
        ExpWeight = params.inf_loss,
        TestWeight = params.test_loss,
        TestRateWeight = params.testrate_loss,
        optimizer = optimizer,
        test_period = params.test_period
    )
end

function SetIC!(mpc::MPC, state::State)
    if mpc isa SIR_MPC
        S0,I0,R0 = mpc.IC
        set_value(S0, state.S/state.N)
        set_value(I0, sum(state.I)/state.N)
        set_value(R0, state.R/state.N)
        # fix(a0, state.prev_action.testing_prop)
        fix(mpc.model[:a0_const], state.prev_actions.testing_prop)
    else
        S0,E0,I0,R0 = mpc.IC
        set_value(S0, state.S/state.N)
        set_value(E0, 0.)
        set_value(I0, sum(state.I)/state.N)
        set_value(R0, state.R/state.N)
        fix(mpc.model[:a0_const], state.prev_action.testing_prop)
    end

end

function OptimalAction(mpc::MPC, state::State)

    SetIC!(mpc, state)

    model = mpc.model

    optimize!(model)

    return JuMP.value.(model[:T])[1:mpc.ControlHorizon]
end

OptimalAction(params::Params, mpc::MPC, s::State) = OptimalAction(mpc, s)
OptimalAction(params::Params, mpc::MPC, pc::ParticleCollection) = OptimalAction(mpc,mean(pc, params))

function Simulate(T::Int, state::State, params::Params, mpc::MPC)
    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = zeros(Action,T)
    rewardHist  = zeros(Float64,T)

    if mpc.ControlHorizon != 1
        warn("Non-unity control horizon feature removed due to conflict with test_period > 1")
    end

    @showprogress for day in 1:T

        if (day-1)%params.test_period == 0
            action = Action(first(OptimalAction(params, mpc, state)))
        else
            action = actionHist[day-1]
        end

        susHist[day] = state.S
        infHist[day] = sum(state.I)
        recHist[day] = state.R
        actionHist[day] = action

        sp, new_infections, pos_tests = SimStep(state, params, action)
        r = reward(params, state, action, sp)

        testHist[day] = sum(pos_tests)
        rewardHist[day] = r

        state = sp
    end

    sim_hist = SimHist(susHist, infHist, recHist, params.N, T, testHist, actionHist, rewardHist, Vector{ParticleCollection}[])
    return sim_hist, actionHist
end

function Simulate(T::Int, state::State, b0::ParticleCollection, params::Params, mpc::MPC)
    upd = BootstrapFilter(unity_test_period(params), n_particles(b0))
    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = zeros(Action,T)
    rewardHist = zeros(Float64,T)
    beliefHist = Vector{ParticleCollection{State}}(undef, T)

    if mpc.ControlHorizon != 1
        warn("Non-unity control horizon feature removed due to conflict with test_period > 1")
    end

    b = b0
    @showprogress for day in 1:T

        if (day-1)%params.test_period == 0
            action = Action(first(OptimalAction(params, mpc, b)))
        else
            action = actionHist[day-1]
        end

        susHist[day] = state.S
        infHist[day] = sum(state.I)
        recHist[day] = state.R
        actionHist[day] = action
        beliefHist[day] = b

        # state, new_infections, pos_tests = SimStep(state, params, action)
        state,o,r = params.interface.gen(params, state, action)
        b = update(upd, b, action, o)
        testHist[day] = sum(o)
        rewardHist[day] = r
    end

    sim_hist = SimHist(susHist, infHist, recHist, params.N, T, testHist, actionHist, rewardHist, beliefHist)
    return sim_hist, actionHist
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
