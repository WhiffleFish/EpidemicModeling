include("/Users/tyler/Documents/code/EpidemicModeling/src/InfectionSim.jl")
using DifferentialEquations, Flux, DiffEqFlux
using BSON
using BSON: @save @load

struct TrainingData
    T::Int
    batch_size::Int
    num_batches::Int
    batches::Array{Float64,4}
    actions::Array{Float64,2}
end

function TrainingData(T::Int, params::Params, batch_size::Int, num_batches::Int; full::Bool=false)::TrainingData
    ICs = GetICs(params)
    tspan = (1. , Float64(T))
    times = 1.:Float64(T)

    if full
        batches, actionArr = GetFullBatches(ICs, T, batch_size, num_batches)
    else
        batches, actionArr = GetBatches(ICs, T, batch_size, num_batches)
    end

    return TrainingData(T, batch_size, num_batches, batches, actionArr)
end


function GetICs(params::Params)::Vector{State}
    dists = repeat([Normal(1_000,500), Normal(10_000,5_000), Normal(50_000,10_000)],4)

    ICs = GenSimStates(50, [initState(dists[i],params) for i in eachindex(dists)], params)

    dist = zeros(Float64,3,length(ICs))
    for (i,state) in enumerate(ICs)
        dist[1,i] = state.S/1_000_000
        dist[2,i] = sum(state.I)/1_000_000
        dist[3,i] = state.R/1_000_000
    end

    indices = sort(rand(findall(<=(0.05), dist[2,:]),100)) |> unique

    deleteat!(ICs,indices)
    return ICs
end


"""
Returns batches of full simulated SIR state vectors(16x1) and their respective actions
# Arguments
- `ICs::Vector{State}`
- `T::Int`
- `batch_size::Int`
- `num_batches::Int`
"""
function GetFullBatches(ICs::Vector{State}, T::Int, batch_size::Int, num_batches::Int; action_dist=Beta(1,2))::Tuple{Array{Float64,4},Array{Float64,2}}
    tspan = (1.,Float64(T))
    times = 1.:Float64(T)

    ICbatches = [rand(ICs,batch_size) for _ in 1:num_batches]
    # ActionArr = rand(num_batches, batch_size)
    ActionArr = rand(Beta(1,2),num_batches, batch_size) # For favoring lower actions

    # batches = (batch, sim, [S,I,R], day)
    batches = Array{Float64,4}(undef, num_batches, batch_size, 16, T)# Well shit this is going to be 4 dimensional
    for i in 1:num_batches
        for j in 1:batch_size
            batches[i,j,:,:] = SimulateFull(T, copy(ICbatches[i][j]), param, action=Action(ActionArr[i,j]))
        end
    end
    return (batches, ActionArr)
end


"""
Returns batches of simulated SIR state vectors(3x1) and their respective actions
# Arguments
- `ICs::Vector{State}`
- `T::Int`
- `batch_size::Int`
- `num_batches::Int`
"""
function GetBatches(ICs::Vector{State}, T::Int, batch_size::Int, num_batches::Int; action_dist=Beta(1,2))::Tuple{Array{Float64,4},Array{Float64,2}}
    tspan = (1., Float64(T))
    times = 1.:Float64(T)

    ICbatches = [rand(ICs,batch_size) for _ in 1:num_batches]
    ActionArr = rand(action_dist, num_batches, batch_size) # For favoring lower actions

    # batches = (batch, sim, [S,I,R], day)
    batches = Array{Float64,4}(undef, num_batches, batch_size, 3, T)
    for i in 1:num_batches
        for j in 1:batch_size
            batches[i,j,:,:] = Array(Simulate(T, copy(ICbatches[i][j]), param, Action(ActionArr[i,j])))
        end
    end
    return (batches, ActionArr)
end


"""
# Output
- `Flux.train!` data argument
- `[(batch1, actions1), (batch2, actions2),...,(batchN, actionsN)]::Vector{Tuple{Array{Float64,3},Vector{Float64}}}`

# Arguments
- `batches::Array{Float64,4}`
- `actions::Array{Float64,2}`
"""
function GetFluxData(batches::Array{Float64,4}, actions::Array{Float64,2})::Vector{Tuple{Array{Float64,3},Vector{Float64}}}
    [(batches[i,:,:,:],ActionArr[i,:]) for i in 1:num_batches]
end

function GetFluxData(data::TrainingData)::Vector{Tuple{Array{Float64,3},Vector{Float64}}}
    [(data.batches[i,:,:,:],data.actions[i,:]) for i in 1:data.num_batches]
end

function SquaredError(pred::Array{T, N}, label::Array{T, N}) where {T,N}
    sum(abs2, pred .- label)
end


"""
Loss over single simulation
"""
function NODELoss(NODE::DiffEqFlux.NeuralDELayer, sim::Array{Float64,2}, action::Float64)::Float64
    SquaredError(predict(NODE, sim[:,1], action), sim)
end

"""
Loss over batch
"""
function NODELoss(NODE::DiffEqFlux.NeuralDELayer, batch::Array{Float64,3}, batch_actions::Vector{Float64})::Float64
    loss = 0.
    batch_size = size(batch,1)
    for i in 1:batch_size
        loss += NODELoss(batch[i,:,:],batch_actions[i]) # Pass to 2D NODELoss
    end
    return loss/batch_size
end

"""
Loss over batch
"""
function NODELoss(NODE::DiffEqFlux.NeuralDELayer, batch::Array{Float64,3}, batch_actions::Vector{Float64}, batch_size::Int)::Float64
    loss = 0.
    for i in 1:batch_size
        loss += NODELoss(batch[i,:,:],batch_actions[i]) # Pass to 2D NODELoss
    end
    return loss/batch_size
end

"""
Loss over full Epoch
"""
function NODELoss(NODE::DiffEqFlux.NeuralDELayer, batches::Array{Float64,4}, actions::Array{Float64,2})::Float64
    loss = 0.
    num_batches, batch_size = (size(batches,1), size(batches,2))
    for i in 1:num_batches
        loss += NODELoss(batches[i,:,:,:],batch_actions[i,:], batch_size) # pass to 3D NODE Loss
    end
    return loss/num_batches
end

function NODELoss(NODE::DiffEqFlux.NeuralDELayer, data::TrainingData)
    loss = 0.
    for i in 1:data.num_batches
        loss += NODELoss(data.batches[i,:,:,:],data.actions[i,:], data.batch_size) # pass to 3D NODE Loss
    end
    return loss/num_batches
end


function predict(NODE::DiffEqFlux.NeuralDELayer, u0::Vector{Float64}, action::Action)::Array{Float64,2}
    NODE(vcat(u0,action.testing_prop)) |> Array
end

function predict(NODE::DiffEqFlux.NeuralDELayer, u0::Vector{Float64}, action::Float64)::Array{Float64,2}
    NODE(vcat(u0,action)) |> Array
end

function predict(NODE::DiffEqFlux.NeuralDELayer, state::State, action::Action)::Array{Float64,2}
    NODE(vcat(Array(state)./state.N,action.testing_prop)) |> Array
end

function predict(NODE::DiffEqFlux.NeuralDELayer, state::State, action::Float64)::Array{Float64,2}
    NODE(vcat(Array(state)./state.N,action)) |> Array
end

function predict(NODE::DiffEqFlux.NeuralDELayer, u0::Array{Float64})::Array{Float64,2}
    NODE(u0) |> Array
end

function callback(NODE::DiffEqFlux.NeuralDELayer, data::TrainingData)
    times = 1:data.T
    i, j = rand(1:data.num_batches), rand(1:data.batch_size)

    pred = predict(NODE, data.batches[i,j,:,1], data.actions[i,j])
    display(SquaredError(pred, data.batches[i,j,:,:]))

    plt = plot(times, data.batches[i,j,:,:]', labels = ["True S" "True I" "True R"], lc = [:blue :red :green])
    plot!(times, pred', labels = ["Pred S" "Pred I" "Pred R"], lc = [:blue :red :green], ls=:dash)
    display(plt)
end

function train!(NODE::DiffEqFlux.NeuralDELayer, epochs::Int, α::Float64, training_data::TrainingData; cbthrottle::Int=1)
    ps = Flux.params(NODE)
    data = GetFluxData(training_data)
    opt = ADAM(α)

    EpochLosses = Vector{Float64}()
    push!(EpochLosses, NODELoss(NODE, training_data))

    for epoch in 1:epochs
        Flux.train!(NODELoss, ps, data, opt, cb = Flux.throttle(() -> callback(NODE, training_data), cbthrottle))
        EpochLoss = NODELoss(NODE, training_data)
        push!(EpochLosses, EpochLoss)
        println("Epoch Loss: ", EpochLoss)
        if EpochLoss ≈ EpochLosses[end-1]
            println("No further loss Improvement")
            break
        end
    end
    display(plot(EpochLosses))
end

# Do I even need to save both though? -> Flux.params(n_ode)
function savemodel(model, model_path::String, weights, weights_path::String)
    @save model_path model
    @save weights_path weights
end
