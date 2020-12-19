using Random, Distributions, Plots, Parameters
import CSV.File
import DataFrames.DataFrame


function FitInfectionDistributions(df, horizon=14, sample_size=50)
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


function prop_above_LOD(df, day, LOD)
    sum(df[!,day] .> LOD)/size(df)[1]
end


function RVsum(dist::Normal, N)
    μ, σ = params(dist)
    return Normal(μ*N,σ*N)
end


function RVsum(dist::Gamma, N)
    k, θ = params(dist)
    if k*N > 0
        return Gamma(k*N, θ)
    else
        return Normal(0,0)
    end
end


@with_kw mutable struct Params
    symptom_dist
    Infdistributions
    symptomatic_isolation_prob = 1
    asymptomatic_prob = 0
    pos_test_probs = zeros(length(Infdistributions)) # Default to no testing
    test_delay::Int = 0
end


 mutable struct State
    S::Int # Current Susceptible Population
    I::Array{Int,1} # Current Infected Population 
    R::Int # Current Recovered Population
    N::Int # Total Population
    Tests::Array{Int,2}
 end


 mutable struct SimHist
    sus # Susceptible Population History
    inf # Infected Population History
    rec # Recovered Population History
    N # Total Population
    T # Simulation Time
    incident
 end
 

mutable struct Action
    testing_prop
end


function IncidentInfections(state::State, params::Params)
    infSum = 0
    for (i, inf) in enumerate(state.I)
        infSum += rand(RVsum(params.Infdistributions[i],inf))
    end
    susceptible_prop = state.S/(state.N - state.R)

    return floor(Int, susceptible_prop*infSum)
end


function SymptomaticIsolation(I, params::Params)
    isolating = zero(I)
    for (i, inf) in enumerate(I)
        symptomatic_prob = cdf(params.symptom_dist,i) - cdf(params.symptom_dist,i-1)
        isolation_prob = symptomatic_prob*(1-params.asymptomatic_prob)*params.symptomatic_isolation_prob
        isolating[i] = rand(Binomial(inf,isolation_prob))
    end
    return isolating
end


function PositiveTests(state::State, params::Params, action::Action)
    @assert 0 <= action.testing_prop <= 1
    pos_tests = zeros(Int32,length(params.pos_test_probs))

    for (i,inf) in enumerate(state.I)
        num_tested = floor(Int,inf*action.testing_prop)
        pos_tests[i] = rand(Binomial(num_tested,params.pos_test_probs[i]))
    end
    return pos_tests
end


function UpdateIsolations(state::State, params::Params, action::Action)
    

    sympt = SymptomaticIsolation(state.I, params) # Number of people isolating due to symptoms
    pos_tests = PositiveTests(state, params, action)
    sympt_prop = sympt ./ state.I # Symptomatic Isolation Proportion
    replace!(sympt_prop, NaN=>0)

    state.R += sum(sympt)
    state.I -= sympt
    
    state.Tests[end,:] = pos_tests

    state.Tests = vcat([transpose(state.Tests[i,:].*(1 .- sympt_prop)) for i in 1:size(state.Tests)[1]]...) |> x -> round.(Int32,x)
    state.R += sum(state.Tests[1,:])
    state.I -= state.Tests[1,:]

    state.Tests = circshift(state.Tests,(0,-1))
    
    return state
end


function Simulate(T::Int, state::State, params::Params, action::Action)
    susHist = zeros(Int32,T)
    infHist = zeros(Int32,T)
    recHist = zeros(Int32,T)
    incidentHist = zeros(Int32,T)

    for day in 1:T
        susHist[day] = state.S 
        infHist[day] = sum(state.I)
        recHist[day] = state.R

        # Update symptomatic and testing-based isolations (Move to recovered)
        state = UpdateIsolations(state, params, action)
        
        # Incident Infections
        state.R += state.I[end]
        state.I = circshift(state.I,1)
        new_infections = IncidentInfections(state, params)
        incidentHist[day] = new_infections
        state.I[1] = new_infections
        state.S -= new_infections
    end
    return SimHist(susHist, infHist, recHist, state.N, T, incidentHist)
end


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
end;


function initParams(;symptomatic_isolation_prob=1, asymptomatic_prob=0, LOD=6, test_delay=0, infections_path="../data/Sample50.csv", sample_size=50 ,viral_loads_path="../data/raw_viral_load.csv", horizon=14)
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


function initState(I, params::Params; N=1_000_000)
    horizon = length(params.Infdistributions)
    if isa(I, Distribution)
        I0 = round.(Int32,rand(truncated(I,0,Inf),horizon))
    elseif isa(I, Array)
        @assert all(I .>= 0 )
        I0 = I
    elseif isa(I, Int)
        I0 = zeros(Int32,horizon)
        I0[1] = I
    else
        throw(DomainError(I, "I must be distribution, array of integers, or single integer"))
    end
    
    S0 = N - sum(I0)
    R0 = 0
    tests = zeros(Int32, params.test_delay+1, horizon)

    return State(S0, I0, R0, N, tests)

end


# --------------------------------------------------------------------

Bdist = LogNormal(1.644,0.363);

df = File("../data/Sample50.csv") |> DataFrame;
viral_loads = File("../data/raw_viral_load.csv") |> DataFrame;
LOD = 6 # 10^6

k = 50 # Sample size for dataframe 

#=
Assume that individual infection rates are distributed as Gamma RV's, s.t. sample size 50 data 
is mean of Gamma RV's
---
Entry 1 - Infections from initial infection to 1 day after infection
=#
distributions = []
for day in 1:14
    if day <= 2
        push!(distributions, Normal(0,0))
    elseif day == 3
        weights = min.(log.(1 ./ df[!,day]),10)
        β = params(fit(Exponential, df[!,day], weights))[1]
        push!(distributions, Gamma(1/k, β*k))
    else
        shape, scale = params(fit(Gamma, df[!,day]))
        push!(distributions, Gamma(shape/k, scale*k))
    end
end

pos_test_probs = [prop_above_LOD(viral_loads,day,LOD) for day in 1:14];
println()