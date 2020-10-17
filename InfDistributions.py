import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from scipy import integrate
import pandas as pd
from tqdm import tqdm

def interp(x1,y1,x2,y2, x):
    m = (y2-y1)/(x2-x1)
    return y1 + m*(x-x1)


def rand_beta(t):
    α = 0.05 # Infectiousness scaling factor
    t0 = np.random.rand() + 2.5
    Vpeak = np.random.rand()*4 + 7
    tpeak = t0 + 0.2 + np.random.gamma(1.8)
    tf = tpeak + np.random.rand()*5 + 5
    
    if t0 < t <= tpeak:
        conc = interp(t0,3,tpeak,Vpeak,t)
        if conc >= 6:
            return α*conc
    elif tpeak < t <= tf:
        conc = interp(tpeak,Vpeak,tf,6,t)
        if conc >= 6:
            return α*conc
    return 0


for i in tqdm(range(int(9e5))):
    pass


N_samples = 10_000
start_day, end_day = 2, 20
infections = np.zeros((N_samples,end_day-start_day+1))
for i in tqdm(range(start_day,end_day+1)):
    infections[:,i-start_day] = np.array([integrate.quad(rand_beta,i,i+1)[0] for _ in tqdm(range(N_samples))])
    
pd.DataFrame(infections).to_csv("Incident_Infections_10000.csv")
