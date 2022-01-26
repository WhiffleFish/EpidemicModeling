function POMDPs.gen(pomdp::Params, s::State, a::Action, rng::AbstractRNG=Random.GLOBAL_RNG)
    rsum = 0.0
    o = nothing
    for i in 1:pomdp.test_period
        s,o,r = pomdp.interface.gen(pomdp, s, a, rng)
        rsum += r
    end
    return (sp=s,o=o,r=rsum)
end

function POMDPs.observation(pomdp::Params, s::State, a::Action, sp::State)
    pomdp.interface.observation(pomdp, s, a)
end

POMDPs.actions(pomdp::Params) = pomdp.interface.actions

POMDPs.discount(pomdp::Params) = pomdp.discount

function Simulate(T::Int, state::State, b::ParticleCollection{State}, pomdp::Params, planner::Policy)
    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = zeros(Action,T)
    rewardHist = zeros(Float64,T)
    beliefHist = Vector{ParticleCollection{State}}(undef, T)

    single_step_pomdp = unity_test_period(pomdp)
    upd = BootstrapFilter(single_step_pomdp, n_particles(b))

    @showprogress for day in 1:T

        if (day-1)%pomdp.test_period == 0
            action = POMDPs.action(planner, b)
        else
            action = actionHist[day-1]
        end

        susHist[day] = state.S
        infHist[day] = sum(state.I)
        recHist[day] = state.R
        actionHist[day] = action
        beliefHist[day] = b

        state, o, r = POMDPs.gen(single_step_pomdp, state, action)
        b = update(upd, b, action, o)

        rewardHist[day] = r
        testHist[day] = sum(o)
    end
    return SimHist(susHist, infHist, recHist, pomdp.N, T, testHist, actionHist, rewardHist, beliefHist)
end
