"""
Simulation History

# Arguments
- `sus::Vector{Int}` - Susceptible Population History
- `inf::Vector{Int}` - Infected Population History
- `rec::Vector{Int}` - Recovered Population History
- `N::Int` - Total Population
- `T::Int` - Simulation Time
- `incident::Array{Int, 1}` = nothing - Incident Infections need not always be recorded
"""
Base.@kwdef struct SimHist
    sus::Vector{Int} # Susceptible Population History
    inf::Vector{Int} # Infected Population History
    rec::Vector{Int} # Recovered Population History
    N::Int # Total Population
    T::Int # Simulation Time
    pos_test::Vector{Int} = Int[]
    actions::Vector{Action} = Action[]
    rewards::Vector{Float64} = Float64[]
    beliefs::Vector{ParticleCollection{State}} = ParticleCollection{State}[]
end


"""
Convert `simHist` struct to 2-dimentional array - Collapse infected array to sum
"""
function Base.Array(simHist::SimHist)::Array{Int64,2}
    hcat(simHist.sus, simHist.inf, simHist.rec) |> transpose |> Array
end

"""
# Arguments
- `state::State` - Current Sim State
- `params::CovidPOMDP` - Simulation parameters
"""
function incident_infections(params::CovidPOMDP, S::Int, I::Vector{Int}, R::Int)
    infSum = 0
    for (i, inf) in enumerate(I)
        infSum += rand(RVsum(params.Infdistributions[i], inf))
    end
    susceptible_prop = S/(params.N - R)

    return floor(Int, susceptible_prop*infSum)
end


"""
# Arguments
- `I::Array{Int,1}` - Current infectious population vector (divided by infection age)
- `params::CovidPOMDP` - Simulation parameters
"""
function SymptomaticIsolation(I::Vector{Int}, params::CovidPOMDP)::Vector{Int64}
    isolating = zero(I)
    for (i, inf) in enumerate(I)
        symptomatic_prob = cdf(params.symptom_dist,i) - cdf(params.symptom_dist,i-1)
        isolation_prob = symptomatic_prob*(1-params.asymptomatic_prob)*params.symptomatic_isolation_prob
        isolating[i] = rand(Binomial(inf,isolation_prob))
    end
    return isolating
end


"""
# Arguments
- `state::State` - Current Sim State
- `params::CovidPOMDP` - Simulation parameters
- `action::Action` - Current Sim Action

# Returns
- `pos_tests::Vector{Int}` - Vector of positive tests stratified by infection age
"""
function PositiveTests(I::Vector{Int}, tests::Matrix{Int}, params::CovidPOMDP, a::Action)
    pos_tests = zeros(Int, length(params.pos_test_probs))

    for (i, inf) in enumerate(I)
        num_already_tested = sum(@view tests[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        pos_tests[i] = rand(Binomial(num_tested,params.pos_test_probs[i]))
    end
    return pos_tests
end


"""
Given current state and simulation parameters, return state for which Infectious population is decreased by surveillance testing and symptomatic isolation.
Isolation due to receiving a positive test and isolation due to developing symptoms are not disjoint, therefore both must be calculated together.
Some waiting for a test to return may develop symptoms and isolate, therefore we cannot simply count these two events as two deductions from the
infectious population.

# Arguments
- `state::State` - Current Sim State
- `params::CovidPOMDP` - Simulation parameters
- `action::Action` - Current Sim Action
- `ret_tests::Bool=false` (opt) - return both new state and positive tests
"""
# Only record number that have taken the test, the number that return postive is
# Binomial dist, such that s' is stochastic on s.
function UpdateIsolations(params::CovidPOMDP, I, R, tests, a::Action)

    sympt = SymptomaticIsolation(I, params) # Number of people isolating due to symptoms
    pos_tests = PositiveTests(I, tests, params, a)

    sympt_prop = sympt ./ I # Symptomatic Isolation Proportion
    replace!(sympt_prop, NaN=>0.0)

    R += sum(sympt)
    I -= sympt

    tests[end,:] .= pos_tests

    @. tests = floor(Int, (1 - sympt_prop)' * tests)

    R += sum(@view tests[1,:])
    @views I .-= tests[1,:]

    @assert all(≥(0), I)

    # Progress testing state forward
    # People k days from receiving test back are now k-1 days from receiving test
    # Tested individuals with infection age t move to infection age t + 1
    tests = circshift(tests,(-1,1))

    # Tests and infection ages do not roll back to beginning; clear last row and first column
    tests[:,1] .= 0
    tests[end,:] .= 0

    return I, R, tests, pos_tests
end

"""
# Arguments
- `state::State` - Current Sim State
- `params::CovidPOMDP` - Simulation parameters
- `action::Action` - Current Sim Action
"""
function SimStep(state::State, params::CovidPOMDP, a::Action)
    (;S, I, R, Tests, prev_action) = state

    # Update symptomatic and testing-based isolations
    I, R, Tests, pos_tests = UpdateIsolations(params, I, R, Tests, a)

    # Incident Infections
    R += I[end]
    I = circshift(I, 1)
    new_infections = incident_infections(params, S, I, R)
    I[1] = new_infections
    S -= new_infections
    sp = State(S, I, R, Tests, a)
    return sp, new_infections, pos_tests
end

function reward(m::CovidPOMDP, s::State, a::Action, sp::State)
    inf_loss = m.inf_loss*sum(sp.I)/m.N
    test_loss = m.test_loss*a.testing_prop
    testrate_loss = m.testrate_loss*abs(a.testing_prop-s.prev_action.testing_prop)
    return -(inf_loss + test_loss + testrate_loss)
end

function continuous_gen(m::CovidPOMDP, s::State, a::Action, rng::AbstractRNG=Random.GLOBAL_RNG)
    sp, new_inf, o = SimStep(s, m, a)
    r = reward(m, s, a, sp)
    o = sum(o)

    return (sp=sp, o=o, r=r)
end

"""
# Arguments
- `T::Int` - Simulation duration (days)
- `state::State` - Current Sim State
- `params::CovidPOMDP` - Simulation parameters
- `action::Action` - Current Sim Action
"""
function Simulate(T::Int, state::State, params::CovidPOMDP, action::Action)::SimHist
    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = fill(action, T)
    rewardHist = zeros(Float64, T)

    for day in 1:T
        susHist[day] = state.S
        infHist[day] = sum(state.I)
        recHist[day] = state.R

        sp, new_infections, pos_tests = SimStep(state, params, action)
        r = reward(params, state, action,sp)

        testHist[day] = sum(pos_tests)
        rewardHist[day] = r
        state = sp
    end
    return SimHist(susHist, infHist, recHist, params.N, T, testHist, actionHist, rewardHist, ParticleCollection{State}[])
end


"""
Run multiple simulations with random initial states but predefined actions
# Arguments
- `T::Int64` - Simulation Time (days)
- `trajectories::Int64` - Total number of simulations
- `params::CovidPOMDP` - Simulation Parameters
- `action::Action` - Testing Action
- `N::Int=1_000_000` - (opt) Population Size
# Return
- `Vector{SimHist}`
"""
function SimulateEnsemble(T::Int64, trajectories::Int64, params::CovidPOMDP, action::Action)::Vector{SimHist}
    [Simulate(T, State(params), params, action) for _ in 1:trajectories]
end

"""
Run multiple simulations with varying predefined actions and random initial states
# Arguments
- `T::Int64` - Simulation Time (days)
- `trajectories::Int64` - Total number of simulations
- `params::CovidPOMDP` - Simulation Parameters
- `actions::Vector{Action}` -
- `N::Int=1_000_000` - (opt) Population Size
# Return
- `Vector{SimHist}`
"""
function SimulateEnsemble(T::Int64, trajectories::Int64, params::CovidPOMDP, actions::Vector{Action})::Vector{SimHist}
    [Simulate(T, State(params), params, actions[i]) for i in 1:trajectories]
end

"""
Provided some state initial condition, simulate resulting epidemic and return vector of all intermediary states
- `T::Int`
- `state::State`
- `params::CovidPOMDP`
- `action::Action` (opt)
"""
function GenSimStates(T::Int, state::State, params::CovidPOMDP; action::Action=Action(0.0))::Vector{State}
    [first(SimStep(s, params, action)) for day in 1:T]
end

"""
Provided some state initial conditions, simulate resulting epidemic and return vector of all intermediary states
- `T::Int`
- `states::Vector{State}`
- `params::CovidPOMDP`
- `action::Action` (opt)
"""
function GenSimStates(T::Int, states::Vector{State}, params::CovidPOMDP; action::Action=Action(0.0))::Vector{State}
    svec = Vector{State}(undef, 0)
    for state in states
        for day in 1:T
            push!(svec, first(SimStep(state, params, action)))
        end
    end
    return svec
end

function FullArr(state::State, param::CovidPOMDP)::Vector{Float64}
    vcat(state.S,state.I,state.R)./param.N
end

function FullArrToSIR(arr::Array{Float64,2})::Matrix{Float64}
    hcat(
        view(arr,1,:),
        reshape(sum(view(arr,2:15,:),dims=1), size(arr,2)),
        view(arr,16,:)
    )'
end

"""
Provided some state initial conditions, simulate resulting epidemic and return vector of all intermediary states
- `T::Int`
- `state::State`
- `params::CovidPOMDP`
- `action::Action` (opt)
"""
function SimulateFull(T::Int, state::State, params::CovidPOMDP; action::Action=Action(0.0))::Array{Float64,2}
    StateArr = Array{Float64,2}(undef,16,T)
    StateArr[:,1] = FullArr(s)
    for day in 2:T
        StateArr[:,day] = FullArr(first(SimStep(s, params, action)))
    end
    return StateArr
end


"""
Convert Simulation SimulateEnsemble output to 3D Array
# Arguments
- `histvec::Vector{SimHist}` - Vector of SimHist structs
"""
function Base.Array(histvec::Vector{SimHist})::Array{Int64,3}
    arr = zeros(Int64, 3, histvec[1].T, length(histvec))
    for i in eachindex(histvec)
        arr[:,:,i] .= Array(histvec[i])
    end
    return arr
end
