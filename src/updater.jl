import Statistics.mean

function Statistics.mean(states::Vector{State}, pomdp::Params)::State
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
    avgR = pomdp.N - (avgS + sum(avgI))
    @assert avgR >= 0
    avgTests = round.(Int, sumTests./n_states)
    return State(avgS, avgI, avgR, pomdp.N, avgTests, first(states).prev_action)
end

Statistics.mean(pc::ParticleCollection{State}, pomdp::Params) = mean(pc.particles, pomdp)
