using Test
println("Importing InfectionSim.jl ...")
@time include("../src/InfectionSim.jl")

println("Creating Params...")
TestParams = [
    initParams(symptomatic_isolation_prob=1.0, asymptomatic_prob=0.0,LOD=5),
    initParams(symptomatic_isolation_prob=0.0, asymptomatic_prob=1.0,LOD=5),
    initParams(symptomatic_isolation_prob=0.95, asymptomatic_prob=0.40,LOD=5),
    initParams(symptomatic_isolation_prob=0.95, asymptomatic_prob=0.40,LOD=5, test_delay=1),
    initParams(symptomatic_isolation_prob=0.95, asymptomatic_prob=0.40,LOD=5, test_delay=5),
    initParams(symptomatic_isolation_prob=0.95, asymptomatic_prob=0.65,LOD=3),
    initParams(symptomatic_isolation_prob=0.95, asymptomatic_prob=0.65,LOD=3)
]

TestActions  = [Action(x) for x in 0:0.1:1]

function test_sim(state::State, action::Action, param::Params; T=100)
    simHist = Simulate(T, state, param, action)
    plt = plotHist(simHist, kind=:stack, order="IRS",prop=true)
    display(plt)
    @test all(0 .<= Array(simHist) .<= simHist.N)
    @test all(0 .<= simHist.incident .<= simHist.N)
end

println("TESTING")
for param in TestParams
    for _ in 1:5
        for action in TestActions

            test_sim(initState(param), action, param)

        end
    end
end
