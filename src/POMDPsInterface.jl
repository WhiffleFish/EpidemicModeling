function POMDPs.gen(pomdp::Params, s::State, a::Action, rng::AbstractRNG=Random.GLOBAL_RNG)
    pomdp.interface.gen(pomdp, s, a, rng)
end

function POMDPs.observation(pomdp::Params, s::State, a::Action, sp::State)
    pomdp.interface.observation(pomdp, s, a)
end

POMDPs.actions(pomdp::Params) = pomdp.interface.actions

POMDPs.discount(pomdp::Params) = pomdp.discount
