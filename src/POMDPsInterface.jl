using ProgressBars

function POMDPs.gen(pomdp::Params, s::State, a::Action, rng::AbstractRNG=Random.GLOBAL_RNG)
    pomdp.interface.gen(pomdp, s, a, rng)
end

function POMDPs.observation(pomdp::Params, s::State, a::Action, sp::State)
    pomdp.interface.observation(pomdp, s, a)
end

POMDPs.actions(pomdp::Params) = pomdp.interface.actions

POMDPs.discount(pomdp::Params) = pomdp.discount

function Simulate(T::Int, state::State, b::ParticleCollection{State}, pomdp::Params, planner::Policy, upd::Updater)
    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = zeros(Action,T)
    rewardHist = zeros(Float64,T)
    beliefHist = ParticleCollection{State}[]

    for day in ProgressBar(1:T)

        action = POMDPs.action(planner, b)

        susHist[day] = state.S
        infHist[day] = sum(state.I)
        recHist[day] = state.R
        actionHist[day] = action
        push!(beliefHist, b)

        state, o, r = POMDPs.gen(pomdp, state, action)
        b = update(upd, b, action, o)

        rewardHist[day] = r
        testHist[day] = sum(o)
    end
    return SimHist(susHist, infHist, recHist, state.N, T, testHist, actionHist, rewardHist, beliefHist)
end
