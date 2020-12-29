using Random, Distributions, Plots, Parameters
import CSV.File
import DataFrames.DataFrame


"""
Fit Distributions to MC sim data for secondary infections per index case as a function of infection age

# Arguments
- `df::DataFrame` - DataFrame for csv containing MC simulations for daily individual infections.
- `horizon::Int=14` - Number of days in infection age before individual is considered naturally recovered and completely uninfectious.
- `sample_size::Int=50` - Sample size for `infections_path` csv where row entry is average infections for given sample size.
"""
function FitInfectionDistributions(df::DataFrame, horizon::Int=14, sample_size::Int=50)
    distributions = []
    for day in 1:horizon
        try # Initially try to fit Gamma
            shape, scale = params(fit(Gamma, df[!,day]))
            push!(distributions, Gamma(shape/sample_size, scale*sample_size))
        catch e

            if isa(e, ArgumentError)
                try # If this doesn't work, try Exponential
                    weights = min.(log.(1 ./ df[!,day]),10)
                    β = params(fit(Exponential, df[!,day], weights))[1]
                    push!(distributions, Gamma(1/sample_size, β*sample_size))
                catch e # If exponential doesn't work either, use dirac 0 dist
                    if isa(e,ArgumentError)
                        push!(distributions, Normal(0,0))
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
function prop_above_LOD(df::DataFrame, day::Int, LOD::Real)
    sum(df[!,day] .> LOD)/size(df)[1]
end


"""
Return distribution resulting from sum of i.i.d RV's characterized by Normal distribution

# Arguments
- `dist::Normal` - Distribution characterizing random variable
- `N::Int` - Number of i.i.d RV's summed
"""
function RVsum(dist::Normal, N::Int)
    μ, σ = params(dist)
    return Normal(μ*N,σ*N)
end


"""
Return distribution resulting from sum of i.i.d RV's characterized by Gamma distribution

# Arguments
- `dist::Gamma` - Distribution characterizing random variable
- `N::Int` - Number of i.i.d RV's summed
"""
function RVsum(dist::Gamma, N::Int)
    k, θ = params(dist)
    if k*N > 0
        return Gamma(k*N, θ)
    else
        return Normal(0,0)
    end
end


"""
# Arguments
- `symptom_dist::Distribution` - Distribution over infection age giving probability of developing symptoms
- `Infdistributions::Array{UnivariateDistribution,1}` - Fitted Distributions for secondary infections per index case as a function of infection age
- `symptomatic_isolation_prob::Real= 1` - Probability of isolating after developing symptoms
- `asymptomatic_prob::Real = 0` - Probability that an infected individual displays no symptoms
- `pos_test_probs::Array{Float64,1} = zeros(length(Infdistributions))` - Probability of testing positive by exceeding test LOD as a function of infection age ``\\tau``(Default to no testing)
- `test_delay::Int = 0` - Delay between test being administered and received by subject (days)
"""
@with_kw mutable struct Params
    symptom_dist::Distribution
    Infdistributions::Array{UnivariateDistribution,1}
    symptomatic_isolation_prob::Float64 = 1.0
    asymptomatic_prob::Float64 = 0.0
    pos_test_probs::Array{Float64,1} = zeros(length(Infdistributions)) # Default to no testing
    test_delay::Int = 0
end


"""
# Arguments
- `S::Int` - Current Susceptible Population
- `I::Array{Int,1}` - Current Infected Population 
- `R::Int` - Current Recovered Population
- `N::Int` - Total Population
- `Tests::Array{Int,2}` - Array for which people belonging to array element ``T_{i,j}`` are ``i-1`` days away 
    from receiving positive test and have infection age ``j``
"""
mutable struct State
    S::Int # Current Susceptible Population
    I::Array{Int,1} # Current Infected Population 
    R::Int # Current Recovered Population
    N::Int # Total Population
    Tests::Array{Int,2} # Rows: Days from receiving test result; Columns: Infection Age
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
@with_kw mutable struct SimHist
    sus::Array{Int, 1} # Susceptible Population History
    inf::Array{Int, 1} # Infected Population History
    rec::Array{Int, 1} # Recovered Population History
    N::Int # Total Population
    T::Int # Simulation Time
    incident::Array{Int, 1} = nothing # Incident Infections need not always be recorded
end
 

"""
Action input to influence epidemic simulation dynamics

# Arguments
- `testing_prop::Real` - Proportion of population to be tested on one day
    - Simplification of typical "x-days between tests per person"  action strategy due to non agent-based model
"""
mutable struct Action
    testing_prop::Float64
end


function Array(simHist::SimHist)
    hcat(simHist.sus,simHist.inf, simHist.rec) |> transpose |> Array
end

function Array(state::State)
    [state.sus, state.inf, state.rec]
end

"""
# Arguments
- `state::State` - Current Sim State
- `params::Params` - Simulation parameters
"""
function IncidentInfections(state::State, params::Params)
    infSum = 0
    for (i, inf) in enumerate(state.I)
        infSum += rand(RVsum(params.Infdistributions[i],inf))
    end
    susceptible_prop = state.S/(state.N - state.R)

    return floor(Int, susceptible_prop*infSum)
end


"""
# Arguments
- `I::Array{Int,1}` - Current infectious population vector (divided by infection age)
- `params::Params` - Simulation parameters
"""
function SymptomaticIsolation(I::Array{Int,1}, params::Params)
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
- `params::Params` - Simulation parameters
- `action::Action` - Current Sim Action
"""
function PositiveTests(state::State, params::Params, action::Action)
    @assert 0 <= action.testing_prop <= 1
    # TODO: Don't test the same people twice!
    pos_tests = zeros(Int,length(params.pos_test_probs))

    for (i,inf) in enumerate(state.I)
        num_already_tested = sum(state.Tests[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*action.testing_prop)
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
- `params::Params` - Simulation parameters
- `action::Action` - Current Sim Action
"""
function UpdateIsolations(state::State, params::Params, action::Action)

    sympt = SymptomaticIsolation(state.I, params) # Number of people isolating due to symptoms
    pos_tests = PositiveTests(state, params, action)

    sympt_prop = sympt ./ state.I # Symptomatic Isolation Proportion
    replace!(sympt_prop, NaN=>0)

    state.R += sum(sympt)
    state.I -= sympt

    state.Tests[end,:] = pos_tests

    state.Tests = (1 .- sympt_prop)'.*state.Tests |> x -> floor.(Int,x)
    
    state.R += sum(state.Tests[1,:])
    state.I -= state.Tests[1,:]
    
    @assert all(state.I .>= 0)

    # Progress testing state forward
    # People k days from receiving test back are now k-1 days from receiving test
    # Tested individuals with infection age t move to infection age t + 1
    state.Tests = circshift(state.Tests,(-1,1))
    
    # Tests and infection ages do not roll back to beginning; clear last row and first column
    state.Tests[:,1] *= 0
    state.Tests[end,:] *= 0

    return state
end



"""
# Arguments
- `state::State` - Current Sim State
- `params::Params` - Simulation parameters
- `action::Action` - Current Sim Action
"""
function SimStep(state::State, params::Params, action::Action)
    
    # Update symptomatic and testing-based isolations
    state = UpdateIsolations(state, params, action)

    # Incident Infections
    state.R += state.I[end]
    state.I = circshift(state.I,1)
    new_infections = IncidentInfections(state, params)
    state.I[1] = new_infections
    state.S -= new_infections

    return state, new_infections
end


"""
# Arguments
- `T::Int` - Simulation duration (days)
- `state::State` - Current Sim State
- `params::Params` - Simulation parameters
- `action::Action` - Current Sim Action
"""
function Simulate(T::Int, state::State, params::Params, action::Action)
    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    incidentHist = zeros(Int,T)

    for day in 1:T
        susHist[day] = state.S 
        infHist[day] = sum(state.I)
        recHist[day] = state.R


        state, new_infections = SimStep(state, params, action)
        incidentHist[day] = new_infections

    end
    return SimHist(susHist, infHist, recHist, state.N, T, incidentHist)
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
...
# Arguments
- `symptomatic_isolation_prob::Real=1`: Probability that individual isolates upon becoming symptomatic `` \\in [0,1] ``.
- `asymptomatic_prob::Real=0`: Probability that individual becomes symptomatic by infection age ``\\tau``, `` \\in [0,1] ``.
- `LOD::Real=6`: Surveillance Test Limit of Detection (Log Scale).
- `test_delay::Int=0`: Time between test is administered and result is returned.
- `infections_path::String="../data/Sample50.csv"`: Path to csv containing MC simulations for daily individual infections.
- `sample_size::Int=50`: Sample size for `infections_path` csv where row entry is average infections for given sample size.
- `viral_loads_path::String="../data/raw_viral_load.csv"`: Path to csv containing viral load trajectories (cp/ml) tabulated on a daily basis for each individual.
- `horizon::Int=14`: Number of days in infection age before individual is considered naturally recovered and completely uninfectious.
...
"""
function initParams(;symptomatic_isolation_prob::Real=1, asymptomatic_prob::Real=0, LOD::Real=6, 
    test_delay::Int=0, infections_path::String="../data/Sample50.csv", sample_size::Int=50,
    viral_loads_path::String="../data/raw_viral_load.csv", horizon::Int=14)
    
    df = File(infections_path) |> DataFrame;
    viral_loads = File(viral_loads_path) |> DataFrame;

    infDistributions = FitInfectionDistributions(df, horizon, sample_size)
    pos_test_probs = [prop_above_LOD(viral_loads,day,LOD) for day in 1:horizon]
    symptom_dist = LogNormal(1.644,0.363);

    return Params(
        symptom_dist, infDistributions, symptomatic_isolation_prob, 
        asymptomatic_prob, pos_test_probs, test_delay
        )
end


"""
# Arguments
- `I`: Initial infections
    - `I::Distribution` - Sample all elements in Infections array from given distribution
    - `I::Array{Int,1}` - Take given array as initial infections array
    - `I::Int` - Take first element of infections array (infection age 0) as given integer
- `params::Params` - Simulation parameters
- `N::Int` (opt) - Total Population Size
"""
function initState(I, params::Params; N=1_000_000)
    horizon = length(params.Infdistributions)
    if isa(I, Distribution)
        I0 = round.(Int,rand(truncated(I,0,Inf),horizon))
        @assert sum(I0) <= N # Ensure Sampled infected is not greater than total population
    elseif isa(I, Array{Int, 1})
        @assert all(I .>= 0 )
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

    return State(S0, I0, R0, N, tests)

end
