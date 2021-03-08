include("/Users/tyler/Documents/code/EpidemicModeling/src/NODE.jl")

param = initParams(symptomatic_isolation_prob=0.95, asymptomatic_prob=0.50)

T = 10
tspan = (1. , Float64(T))
times = 1.:Float64(T)
batch_size = 10
num_batches = 100

data = TrainingData(T, param, batch_size, num_batches)

DNNLayer = FastChain(
  (x,p) -> vcat(x[1]*x[2],x[2]*x[3], x), # Add SI, IT terms
  FastDense(6,50, tanh),
  FastDense(50,30, tanh),
  FastDense(30,4)
)

NODE = NeuralODE(DNNLayer, tspan, Tsit5(), saveat=times)
ps = Flux.params(NODE)

epochs = 1
LR = 0.005
train!(NODE, epochs, LR, data)
