import numpy as np
from scipy import integrate
import pandas as pd
from tqdm import tqdm

def interp(x1,y1,x2,y2, x):
    m = (y2-y1)/(x2-x1)
    return y1 + m*(x-x1)

# Fix Infectiousness Scaling Factor: beta should return alpha*max(conc-6,0)
def rand_beta(t,alpha):
    t0 = np.random.rand() + 2.5
    Vpeak = np.random.rand()*4 + 7
    tpeak = t0 + 0.2 + np.random.gamma(1.8)
    tf = tpeak + np.random.rand()*5 + 5
    
    if t0 < t <= tpeak:
        conc = interp(t0,3,tpeak,Vpeak,t)
        return alpha*max(conc-6,0)
    elif tpeak < t <= tf:
        conc = interp(tpeak,Vpeak,tf,6,t)
        return alpha*max(conc-6,0)
    
    return 0

ALPHA = 1
N_samples = 10_000
start_day, end_day = 0, 20
infections = np.zeros((N_samples,end_day-start_day+1))
for i in tqdm(range(start_day,end_day+1)):
    infections[:,i-start_day] = np.array([integrate.quad(rand_beta,i,i+1)[0] for _ in tqdm(range(N_samples))])
    
pd.DataFrame(infections).to_csv("TotalInf_N10000_NEW_UNSCALED.csv", index=False)
