"""
Fit Distributions to MC sim data for secondary infections per index case as a function of infection age

# Arguments
- `df::DataFrame` - DataFrame for csv containing MC simulations for daily individual infections.
- `horizon::Int=14` - Number of days in infection age before individual is considered naturally recovered and completely uninfectious.
- `sample_size::Int=50` - Sample size for `infections_path` csv where row entry is average infections for given sample size.
"""
function FitInfectionDistributions(df::DataFrame, horizon::Int=14, sample_size::Int=50)
    distributions = Gamma{Float64}[]
    for day in 1:horizon
        try # Initially try to fit Gamma
            shape, scale = Distributions.params(fit(Gamma, df[!,day]))
            push!(distributions, Gamma(shape/sample_size, scale*sample_size))
        catch e

            if isa(e, ArgumentError)
                try # If this doesn't work, try Exponential
                    weights = min.(log.(1 ./ df[!,day]),10)
                    β = Distributions.params(fit(Exponential, df[!,day], weights))[1]
                    push!(distributions, Gamma(1/sample_size, β*sample_size))
                catch e # If exponential doesn't work either, use dirac 0 dist
                    if isa(e, ArgumentError)
                        push!(distributions, Gamma(1e-100,1e-100))
                    else
                        throw(e)
                    end
                end
            end

        end
    end
    return distributions
end


"""
Proportion of infectious population above some limit of detection from MC simulations

# Arguments
- `df::DataFrame` - DataFrame for csv containing MC simulations for daily individual infections.
- `day::Int` - Infection age (day)
- `LOD::Real` - Limit of detection (Log scale: ``10^x \\rightarrow x``)
"""
function prop_above_LOD(df::DataFrame, day::Int, LOD::Real)::Float64
    sum(df[!,day] .> LOD)/size(df,1)
end


"""
Return distribution resulting from sum of i.i.d RV's characterized by Normal distribution

# Arguments
- `dist::Normal` - Distribution characterizing random variable
- `N::Int` - Number of i.i.d RV's summed
"""
function RVsum(dist::Normal, N::Int)::Normal
    μ, σ = Distributions.params(dist)
    return Normal(μ*N,σ*N)
end


"""
Return distribution resulting from sum of i.i.d RV's characterized by Gamma distribution

# Arguments
- `dist::Gamma` - Distribution characterizing random variable
- `N::Int` - Number of i.i.d RV's summed
"""
function RVsum(dist::Gamma, N::Int)::UnivariateDistribution
    k, θ = Distributions.params(dist)
    if k*N > 0
        return Gamma(k*N, θ)
    else
        return Normal(0,0)
    end
end

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
- `N::Int` - Total Population
- `Tests::Matrix{Int}` - Array for which people belonging to array element ``T_{i,j}`` are ``i-1`` days away
    from receiving positive test and have infection age ``j``
"""
struct State
    S::Int # Current Susceptible Population
    I::Vector{Int} # Current Infected Population
    R::Int # Current Recovered Population
    N::Int # Total Population - move to params
    Tests::Matrix{Int} # Rows: Days from receiving test result; Columns: Infection Age
    prev_action::Action
end

abstract type SolverInterface end

struct ContinuousSolverInterface{O<:Function,G<:Function} <: SolverInterface
    actions::Vector{Action}
    observation::O
    gen::G
end

struct DiscreteSolverInterface{O<:Function,G<:Function} <: SolverInterface
    actions::Vector{Action}
    observation::O
    gen::G
    c::Float64
    n_obs::Int
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
- `interface::SolverInterface` - Used for discrete/continuous observations/gen
"""
struct CovidPOMDP{D<:Distribution, IF<:SolverInterface} <: POMDP{State, Action, Int64}
    symptom_dist::D
    interface::IF
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

## Continuous

function reward(m::CovidPOMDP, s::State, a::Action, sp::State)
    inf_loss = m.inf_loss*sum(sp.I)/m.N
    test_loss = m.test_loss*a.testing_prop
    testrate_loss = m.testrate_loss*abs(a.testing_prop-s.prev_action.testing_prop)
    return -(inf_loss + test_loss + testrate_loss)
end

function ContinuousGen(m::CovidPOMDP, s::State, a::Action, rng::AbstractRNG=Random.GLOBAL_RNG)
    sp, new_inf, o = SimStep(s, m, a)
    r = reward(m, s, a, sp)
    o = sum(o)

    return (sp=sp, o=o, r=r)
end

function ContinuousObservation(m::CovidPOMDP, state::State, a::Action)
    tot_mean = 0.0
    tot_variance = 0.0

    for (i,inf) in enumerate(state.I)
        num_already_tested = sum(@view state.Tests[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        dist = Binomial(num_tested,m.pos_test_probs[i])
        tot_mean += mean(dist)
        tot_variance += std(dist)^2
    end
    return Normal(tot_mean, sqrt(tot_variance))
end


## Discrete

expansion_map(x::Float64, c::Float64=2.0) = log((exp(c)-1)*x + 1)
contraction_map(x::Float64, c::Float64=2.0) = (exp(x)-1)/(exp(c)-1)

"""
Returns tuple `(bin_centers, bin_edges)`
OUTPUT IS EXPANSION MAPPED
"""
function get_bins(c::Float64=2.0, n_obs::Int=10)
    bin_edges = LinRange(0,c,n_obs+1)
    bin_centers = 0.5*(circshift(bin_edges,1) + bin_edges)[2:end]
    return bin_centers, bin_edges
end

function DiscreteGen(m::CovidPOMDP, s::State, a::Action, rng::AbstractRNG)
    sp, new_inf, o = SimStep(s, m, a)
    r = reward(m, s, a, sp)
    o = expansion_map(sum(o)/m.N)

    bin_centers, _ = get_bins(m.interface.c, m.interface.n_obs)

    obs = argmin(abs.(o .- bin_centers))
    return (sp=sp, o=obs, r=r)
end

function DiscreteObservation(m::CovidPOMDP, state::State, a::Action)
    c = m.interface.c
    n_obs = m.interface.n_obs
    _, bin_edges = get_bins(c, n_obs)

    bin_edges = round.(Int,contraction_map.(bin_edges).*m.N)

    tot_mean = 0.0
    tot_variance = 0.0
    for (i,inf) in enumerate(state.I)
        num_already_tested = sum(state.Tests[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        dist = Binomial(num_tested,m.pos_test_probs[i])
        tot_mean += mean(dist)
        tot_variance += std(dist)^2
    end
    continuous_dist = Normal(tot_mean, sqrt(tot_variance))
    if mean(continuous_dist) ≈ 0.0
        return SparseCat(collect(1:n_obs), vcat(1.0, zeros(Float64, n_obs-1)))
    end
    probs = [
        cdf(continuous_dist,bin_edges[idx+1])-cdf(continuous_dist,bin_edges[idx])
        for idx in eachindex(bin_edges[1:end-1])
    ]
    probs ./= sum(probs)
    @assert all(≥(0.), probs)
    # return SparseCat(observations(m), probs)
    return SparseCat(collect(1:n_obs), probs)
end


"""
Simulation History

# Arguments
- `sus::Array{Int, 1}` - Susceptible Population History
- `inf::Array{Int, 1}` - Infected Population History
- `rec::Array{Int, 1}` - Recovered Population History
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
Convert `State` struct to Vector - Collapse infected array to sum
"""
function Base.Array(state::State)::Vector{Int64}
    [state.S, sum(state.I), state.R]
end

"""
# Arguments
- `state::State` - Current Sim State
- `params::CovidPOMDP` - Simulation parameters
"""
function IncidentInfections(params::CovidPOMDP, S::Int, I::Vector{Int}, R::Int)
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
# requires: I, R, tests
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
    (;S, I, R, N, Tests, prev_action) = state

    # Update symptomatic and testing-based isolations
    I, R, Tests, pos_tests = UpdateIsolations(params, I, R, Tests, a)

    # Incident Infections
    R += I[end]
    I = circshift(I, 1)
    new_infections = IncidentInfections(params, S, I, R)
    I[1] = new_infections
    S -= new_infections
    sp = State(S, I, R, N, Tests, a)
    return sp, new_infections, pos_tests
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

"""
# Arguments
- `hist::SimHist` - Simulation Data History
- `prop::Bool=true` - Graph as subpopulations as percentage (proportion) of total population
- `kind::Symbol=:line` - `:line` to graph all trajectories on top of each other; ``:stack` for stacked line plot
- `order::String="SIR"` - Pertains to stacking order for stacked line plot and legend order (arg must be some permutation of chars S,I,R)
"""
function plotHist(hist::SimHist; prop::Bool=true, kind::Symbol=:line, order::String="SIR")
    @assert length(order) == 3
    data_dict = Dict('S'=>hist.sus, 'I' => hist.inf, 'R' => hist.rec)
    label_dict = Dict('S'=>"Sus", 'I'=>"Inf", 'R'=>"Rec")
    color_dict = Dict('S'=>:blue, 'I'=>:red, 'R'=>:green)

    data = [data_dict[letter] for letter in order]
    data = prop ? hcat(data...)*100/hist.N : hcat(data...)
    ylabel = prop ? "Population Percentage" : "Population"

    labels = reshape([label_dict[letter] for letter in order],1,3)
    colors = hcat([color_dict[letter] for letter in order]...)

    if kind == :line
        plot(data, labels=labels, ylabel=ylabel, xlabel="Time (Days)", color=colors)
    else
        areaplot(data, labels=labels, ylabel=ylabel, xlabel="Time (Days)", color=colors)
    end
end

"""
Alias for `plotHist`
# Arguments
- `hist::SimHist` - Simulation Data History
- `prop::Bool=true` - Graph as subpopulations as percentage (proportion) of total population
- `kind::Symbol=:line` - `:line` to graph all trajectories on top of each other; ``:stack` for stacked line plot
- `order::String="SIR"` - Pertains to stacking order for stacked line plot and legend order (arg must be some permutation of chars S,I,R)
"""
function plot(hist::SimHist; prop::Bool=true, kind::Symbol=:line, order::String="SIR")
    plotHist(hist, prop=prop, kind=kind, order=order)
end

function plot(pomdp::CovidPOMDP, hist::SimHist)
    l = @layout [a;b]
    as = [a.testing_prop for a in hist.actions]

    p1_label = length(hist.beliefs) > 0 ? "True" : ""
    p1 = plot(0:hist.T-1, hist.inf./hist.N, label=p1_label)
    if length(hist.beliefs) > 0
        plot!(p1,0:hist.T-1, [sum(mean(states, pomdp).I)/pomdp.N for states in hist.beliefs], label="Estimated")
    end
    ylabel!("Infected Prop")
    p2 = plot(0:hist.T-1, as, label="", ylabel="Testing Prop")
    plt = plot(p1, p2, layout=l)
    xlabel!("Day")
    display(plt)
end

"""
...
# Arguments (All optional kwargs)
- `test_delay::Int=0`: Time between test is administered and result is returned.
- `discount::Float64=0.95`: POMDP discount factor
- `actions::Vector{Action} = Action.(0:0.1:1.0)`: Action space discretization
- `N::Int=1_000_000`: Total Population Size
- `c::Float64=2.0`: Discrete observation expansion factor, ref `expansion_map`
- `n_obs::Int=0`: Number of observations (`n_obs==0` for continuous observation space; `n_obs>0` for discretized observation space)
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
    actions::Vector{Action} = Action.(0:0.1:1.0),
    N::Int = 1_000_000,
    c::Float64 = 2.0,
    n_obs::Int = 0,
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

    if n_obs ≤ 0
        interface = ContinuousSolverInterface(actions, ContinuousObservation, ContinuousGen)
    else
        interface = DiscreteSolverInterface(actions, DiscreteObservation, DiscreteGen, c, n_obs)
    end

    return CovidPOMDP(
        symptom_dist,
        interface,
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

    return State(S0, I0, R0, N, tests, Action(0.0))
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

    return State(S, I, R, N, tests, Action(0.0))
end
