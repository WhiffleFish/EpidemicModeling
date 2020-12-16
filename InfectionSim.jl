using Random, Distributions, Plots
import CSV.File
import DataFrames.DataFrame

Bdist = LogNormal(1.644,0.363);

df = File("Sample50.csv") |> DataFrame;
viral_loads = File("raw_viral_load.csv") |> DataFrame;

k = 50 # Sample size for dataframe 

#=
Assume that individual infection rates are distributed as Gamma RV's, s.t. sample size 50 data 
is mean of Gamma RV's
=#
distributions = []
for day in 1:14
    if day <= 3
        push!(distributions, Normal(0,0))
    else
        shape, scale = params(fit(Gamma, df[!,day]))

        push!(distributions, Gamma(shape/k, scale*k))
    end
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

struct Params
    symptomatic_isolation_prob
    asymptomatic_prob
    symptom_dist
    Infdistributions
    N # Population
end

 mutable struct State
    S
    I
    R
 end

 mutable struct SimHist
    sus
    inf
    rec
    N
    T
 end
 
function IncidentInfections(state::State, params::Params)
    infSum = 0
    for (i, inf) in enumerate(state.I)
        infSum += rand(RVsum(params.Infdistributions[i],inf))
    end
    susceptible_prop = state.S/(params.N- state.R)

    return floor(Int, susceptible_prop*infSum)
end

function SymptomaticIsolation(I, params::Params)
    isolating = zeros(Int,length(I))
    for (i, inf) in enumerate(I)
        symptomatic_prob = cdf(params.symptom_dist,i)-cdf(params.symptom_dist,i-1)
        isolation_prob = symptomatic_prob*(1-params.asymptomatic_prob)*params.symptomatic_isolation_prob
        isolating[i] = rand(Binomial(inf,isolation_prob))
    end
    return isolating
end


function Simulate(T, state::State, params::Params)
    susHist = zeros(Int64,T)
    infHist = zeros(Int64,T)
    recHist = zeros(Int64,T)

    for day in 1:T
        susHist[day] = state.S 
        infHist[day] = sum(state.I)
        recHist[day] = state.R
        
        sympt = SymptomaticIsolation(state.I,params)
        state.R += sum(sympt)
        state.I -= sympt
        
        state.R += state.I[end]
        state.I = circshift(state.I,1)
        new_infections = IncidentInfections(state, params)
        state.I[1] = new_infections
        state.S -= new_infections
    end
    return SimHist(susHist, infHist, recHist, params.N, T)
end

function plotHist(hist::SimHist; prop::Bool=true, kind::Symbol=:line, order::String="SIR")
    @assert length(order) == 3
    data_dict = Dict('S'=>hist.sus, 'I' => hist.inf, 'R' => hist.rec)
    label_dict = Dict('S'=>"Sus", 'I'=>"Inf", 'R'=>"Rec")

    data = [data_dict[letter] for letter in order]
    data = prop ? hcat(data...)*100/hist.N : hcat(data...)
    ylabel = prop ? "Population Percentage" : "Population"

    labels = reshape([label_dict[letter] for letter in order],1,3)

    if kind == :line
        plot(data, labels=labels, ylabel=ylabel, xlabel="Time (Days)")
    else
        areaplot(data, labels=labels, ylabel=ylabel, xlabel="Time (Days)")
    end
end;