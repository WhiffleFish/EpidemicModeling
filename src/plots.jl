"""
# Arguments
- `hist::SimHist` - Simulation Data History
- `prop::Bool=true` - Graph as subpopulations as percentage (proportion) of total population
- `kind::Symbol=:line` - `:line` to graph all trajectories on top of each other; ``:stack` for stacked line plot
- `order::String="SIR"` - Pertains to stacking order for stacked line plot and legend order (arg must be some permutation of chars S,I,R)
"""
function plotHist(hist::SimHist; prop::Bool=true, kind::Symbol=:line, order::String="SIR")
    @assert length(order) == 3
    data_dict = Dict('S'=>hist.sus, 'I' => hist.inf, 'R' => hist.rec)
    label_dict = Dict('S'=>"Sus", 'I'=>"Inf", 'R'=>"Rec")
    color_dict = Dict('S'=>:blue, 'I'=>:red, 'R'=>:green)

    data = [data_dict[letter] for letter in order]
    data = prop ? hcat(data...)*100/hist.N : hcat(data...)
    ylabel = prop ? "Population Percentage" : "Population"

    labels = reshape([label_dict[letter] for letter in order],1,3)
    colors = hcat([color_dict[letter] for letter in order]...)

    if kind == :line
        plot(data, labels=labels, ylabel=ylabel, xlabel="Time (Days)", color=colors)
    else
        areaplot(data, labels=labels, ylabel=ylabel, xlabel="Time (Days)", color=colors)
    end
end

"""
Alias for `plotHist`
# Arguments
- `hist::SimHist` - Simulation Data History
- `prop::Bool=true` - Graph as subpopulations as percentage (proportion) of total population
- `kind::Symbol=:line` - `:line` to graph all trajectories on top of each other; ``:stack` for stacked line plot
- `order::String="SIR"` - Pertains to stacking order for stacked line plot and legend order (arg must be some permutation of chars S,I,R)
"""
function Plots.plot(hist::SimHist; prop::Bool=true, kind::Symbol=:line, order::String="SIR")
    plotHist(hist, prop=prop, kind=kind, order=order)
end

function Plots.plot(pomdp::CovidPOMDP, hist::SimHist)
    l = @layout [a;b]
    as = [a.testing_prop for a in hist.actions]

    p1_label = length(hist.beliefs) > 0 ? "True" : ""
    p1 = plot(0:hist.T-1, hist.inf./hist.N, label=p1_label)
    if length(hist.beliefs) > 0
        plot!(p1,0:hist.T-1, [sum(mean(states, pomdp).I)/pomdp.N for states in hist.beliefs], label="Estimated")
    end
    ylabel!("Infected Prop")
    p2 = plot(0:hist.T-1, as, label="", ylabel="Testing Prop")
    plt = plot(p1, p2, layout=l)
    xlabel!("Day")
    display(plt)
end
