"""
Action input to influence epidemic simulation dynamics

# Arguments
- `testing_prop::Real` - Proportion of population to be tested on one day
    - Simplification of typical "x-days between tests per person"  action strategy due to non agent-based model
"""
struct Action
    testing_prop::Float64
end

Base.zero(Action) = Action(0.0)


"""
# Arguments
- `S::Int` - Current Susceptible Population
- `I::Vector{Int}` - Current Infected Population
- `R::Int` - Current Recovered Population
- `Tests::Matrix{Int}` - Array for which people belonging to array element ``T_{i,j}`` are ``i-1`` days away
    from receiving positive test and have infection age ``j``
"""
struct State
    S::Int # Current Susceptible Population
    I::Vector{Int} # Current Infected Population
    R::Int # Current Recovered Population
    Tests::Matrix{Int} # Rows: Days from receiving test result; Columns: Infection Age
    prev_action::Action
end


"""
# Arguments
- `symptom_dist::Distribution` - Distribution over infection age giving probability of developing symptoms
- `Infdistributions::Array{UnivariateDistribution,1}` - Fitted Distributions for secondary infections per index case as a function of infection age
- `symptomatic_isolation_prob::Real= 1` - Probability of isolating after developing symptoms
- `asymptomatic_prob::Real = 0` - Probability that an infected individual displays no symptoms
- `pos_test_probs::Array{Float64,1} = zeros(length(Infdistributions))` - Probability of testing positive by exceeding test LOD as a function of infection age ``\\tau``(Default to no testing)
- `test_delay::Int = 0` - Delay between test being administered and received by subject (days)
- `N::Int=1_000_000` - Total Population
- `discount::Float64=0.95` - POMDP discount factor
"""
struct CovidPOMDP{D<:Distribution} <: POMDP{State, Action, Int}
    symptom_dist::D
    Infdistributions::Vector{Gamma{Float64}}
    symptomatic_isolation_prob::Float64
    asymptomatic_prob::Float64
    pos_test_probs::Vector{Float64}
    test_delay::Int
    N::Int
    discount::Float64
    inf_loss::Float64
    test_loss::Float64
    testrate_loss::Float64
    test_period::Int
end


"""
...
# Arguments (All optional kwargs)
- `test_delay::Int=0`: Time between test is administered and result is returned.
- `discount::Float64=0.95`: POMDP discount factor
- `actions::Vector{Action} = Action.(0:0.1:1.0)`: Action space discretization
- `N::Int=1_000_000`: Total Population Size
- `inf_loss::Float64 = 50.0`: Penalty coefficient for infected population proportion
- `test_loss::Float64 = 1.0`: Penalty coefficient for testing proportion (cost of testing)
- `testrate_loss::Float64 = 0.1`: Penalty coefficient for change in testing proportion
- `symptomatic_isolation_prob::Float64=0.95`: Probability that individual isolates upon becoming symptomatic
- `asymptomatic_prob::Float64=0.40`: Probability that individual becomes symptomatic by infection age
- `LOD::Real=6`: Surveillance Test Limit of Detection (Log Scale).
- `infections_path::String="data/Sample50.csv"`: Path to csv containing MC simulations for daily individual infections.
- `sample_size::Int=50`: Sample size for `infections_path` csv where row entry is average infections for given sample size.
- `viral_loads_path::String="data/raw_viral_load.csv"`: Path to csv containing viral load trajectories (cp/ml) tabulated on a daily basis for each individual.
- `horizon::Int=14`: Number of days in infection age before individual is considered naturally recovered and completely uninfectious.
...
"""
function CovidPOMDP(;
    test_delay::Int = 0,
    discount::Float64 = 0.95,
    N::Int = 1_000_000,
    inf_loss::Float64 = 50.0,
    test_loss::Float64 = 1.0,
    testrate_loss::Float64 = 0.1,
    symptomatic_isolation_prob::Float64 = 0.95,
    asymptomatic_prob::Float64 = 0.40,
    LOD::Real = 6,
    infections_path::String = joinpath(@__DIR__, "data", "Sample50.csv"),
    sample_size::Int = 50,
    viral_loads_path::String = joinpath(@__DIR__, "data", "raw_viral_load.csv"),
    horizon::Int = 14,
    test_period::Int = 1
    )

    df = File(infections_path) |> DataFrame;
    viral_loads = File(viral_loads_path) |> DataFrame;

    infDistributions = FitInfectionDistributions(df, horizon, sample_size)
    pos_test_probs = [prop_above_LOD(viral_loads,day,LOD) for day in 1:horizon]
    symptom_dist = LogNormal(1.644,0.363)

    return CovidPOMDP(
        symptom_dist,
        infDistributions,
        symptomatic_isolation_prob,
        asymptomatic_prob,
        pos_test_probs,
        test_delay,
        N,
        discount,
        inf_loss,
        test_loss,
        testrate_loss,
        test_period
    )
end


"""
Take given CovidPOMDP obj and return same CovidPOMDP obj only with test_period changed to 1
"""
function unity_test_period(pomdp::CovidPOMDP)::CovidPOMDP
    return CovidPOMDP(
        pomdp.symptom_dist,
        pomdp.interface,
        pomdp.Infdistributions,
        pomdp.symptomatic_isolation_prob,
        pomdp.asymptomatic_prob,
        pomdp.pos_test_probs,
        pomdp.test_delay,
        pomdp.N,
        pomdp.discount,
        pomdp.inf_loss,
        pomdp.test_loss,
        pomdp.testrate_loss,
        1
    )
end

function population(s::State)
    return s.S + sum(s.I) + s.R
end

"""
Convert `State` struct to Vector - Collapse infected array to sum
"""
function Base.Array(state::State)::Vector{Int64}
    [state.S, sum(state.I), state.R]
end

"""
# Arguments
- `I`: Initial infections
    - `I::Distribution` - Sample all elements in Infections array from given distribution
    - `I::Array{Int,1}` - Take given array as initial infections array
    - `I::Int` - Take first element of infections array (infection age 0) as given integer
- `params::CovidPOMDP` - Simulation parameters
"""
function State(I, params::CovidPOMDP, rng::AbstractRNG=Random.GLOBAL_RNG)::State
    N = params.N
    horizon = length(params.Infdistributions)
    if isa(I, Distribution)
        I0 = round.(Int,rand(rng, truncated(I,0,Inf),horizon))
        @assert sum(I0) <= N # Ensure Sampled infected is not greater than total population
    elseif isa(I, Vector{Int})
        @assert all(≥(0), I)
        @assert sum(I) <= N
        I0 = I
    elseif isa(I, Int)
        I0 = zeros(Int,horizon)
        I0[1] = I
    else
        throw(DomainError(I, "`I` must be distribution, array of integers, or single integer"))
    end

    S0 = N - sum(I0)
    R0 = 0
    tests = zeros(Int, params.test_delay+1, horizon)

    return State(S0, I0, R0, tests, Action(0.0))
end

function simplex_sample(N::Int, m::Float64, rng::AbstractRNG=Random.GLOBAL_RNG)
    v = rand(rng, N-1)*m
    push!(v, 0, m)
    sort!(v)
    return (v - circshift(v,1))[2:end]
end


"""
Random Initial State using Bayesian Bootstrap / Simplex sampling
# Arguments
- `params::CovidPOMDP` - Simulation parameters
"""
function State(params::CovidPOMDP, rng=Random.GLOBAL_RNG)::State
    N = params.N
    S, inf, R = round.(Int,simplex_sample(3, Float64(N), rng))

    horizon = length(params.Infdistributions)

    I = round.(Int,simplex_sample(horizon, Float64(inf), rng))

    leftover = N - (S + sum(I) + R)
    max_idx = argmax([S, sum(I), R])
    if max_idx == 1
        S += leftover
    elseif max_idx == 2
        I[end] += leftover
    else
        R += leftover
    end

    tests = zeros(Int, params.test_delay+1, horizon)

    return State(S, I, R, tests, Action(0.0))
end
