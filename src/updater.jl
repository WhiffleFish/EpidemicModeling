function initialbelief(pomdp::CovidPOMDP, np::Int)
    ParticleCollection([rand(initialstate(pomdp)) for _ in 1:np])
end

function mean(states::Vector{State}, N::Int)::State
    n_states = length(states)
    sumS = 0
    sumI = zeros(Int,length(first(states).I))
    sumTests = zeros(Int,size(first(states).Tests))
    for s in states
        sumS += s.S
        sumI .+= s.I
        sumTests .+= s.Tests
    end
    avgS = round(Int,sumS/n_states)
    avgI = round.(Int, sumI./n_states)
    avgR = N - (avgS + sum(avgI))
    @assert avgR ≥ 0
    avgTests = round.(Int, sumTests./n_states)
    return State(avgS, avgI, avgR, avgTests, first(states).prev_action)
end

mean(states::Vector{State}) = mean(states, population(first(states)))

mean(pc::ParticleCollection{State}, pomdp::CovidPOMDP) = mean(pc.particles, pomdp.N)
