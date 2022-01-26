param = CovidPOMDP(
    symptomatic_isolation_prob = 0.95,
    asymptomatic_prob = 0.50,
    LOD = 5
)

SIRres, SIRp = FitRandControlledEnsemble(:SIR, 30, 500, param, show_trace=true)
SEIRres, SEIRp = FitRandControlledEnsemble(:SEIR, 30, 500, param, show_trace=true)

SIRmpc = MPC(SIRp)
SEIRmpc = MPC(
    SEIRp,
    callback = false,
    ControlHorizon = 1,
    TestWeight = 2.0,
    TestRateWeight = 50.0,
    PredHorizon=30
)

T = 150
state = State(Normal(10,0), param)
simHist, actionHist = Simulate(T, state, param, SEIRmpc)
actionHist = getfield.(actionHist, :testing_prop)

l = @layout [a;b]
p1 = plot(simHist.inf./simHist.N, label="")
ylabel!("Infected Prop")
title!("MPC Test")
p2 = plot(actionHist, label="", ylabel="Testing Prop")
plt = plot(p1, p2, layout=l)
xlabel!("Day")
display(plt)
