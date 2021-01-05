"""
FitInfectionDistributions
"""

include("/Users/tyler/Documents/code/EpidemicModeling/src/InfectionSim.jl")

df = File(infections_path) |> DataFrame;
viral_loads = File(viral_loads_path) |> DataFrame;

FitInfectionDistributions()
