import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

def gen_params():
    t0 = np.random.uniform(2.5,3.5)
    tpeak = t0 + 0.2 + np.random.gamma(1.8)
    tf = tpeak + np.random.uniform(5,10)
    Vpeak = np.random.uniform(7,11)
    return t0,tpeak,tf,Vpeak


def interp(x1,y1,x2,y2, x):
    m = (y2-y1)/(x2-x1)
    return y1 + m*(x-x1)

def integrate_full_inf(alpha=1):
    """
    Integrate over full stochastic infectiousness curve
    """
    t0 = np.random.uniform(2.5,3.5)
    Vpeak = np.random.uniform(7,11)
    tpeak = t0 + 0.2 + np.random.gamma(1.8)
    tf = tpeak + np.random.uniform(5,10)
    
    t0n = t0 + 3*(tpeak-t0)/(Vpeak-3)
    Vpeakn = alpha*max(Vpeak-6,0)
    I = 0.5*(tpeak-t0n)*(Vpeakn) + 0.5*(tf-tpeak)*(Vpeakn)
    return I

def get_alpha(R0, N=10_000):
    Infs = np.array([integrate_full_inf(alpha=1) for i in range(N)])
    R0_unscaled = Infs.mean()
    return R0/R0_unscaled

def integrate_inf(t1,t2,alpha, param_list=None):
    """
    Integrate over infectiousness curve from time t1 to t2 scaled by alpha
    """
    if not param_list:
        t0 = np.random.uniform(2.5,3.5)
        tpeak = t0 + 0.2 + np.random.gamma(1.8)
        tf = tpeak + np.random.uniform(5,10)
        Vpeak = np.random.uniform(7,11)
    else:
        t0,tpeak,tf,Vpeak = param_list

    t0n = t0 + 3*(tpeak-t0)/(Vpeak-3)
    Vpeakn = alpha*max(Vpeak-6,0)

    if t0n < t1 < tpeak:
        V1 = interp(t0n,0,tpeak,Vpeakn,t1)
    elif tpeak < t1 < tf:
        V1 = interp(tpeak,Vpeakn,tf,0,t1)
    else:
        V1 = 0

    if t0n < t2 < tpeak:
        V2 = interp(t0n,0,tpeak,Vpeakn,t2)
    elif tpeak < t2 < tf:
        V2 = interp(tpeak,Vpeakn,tf,0,t2)
    else: 
        V2 = 0

    base_times = np.array([t0n,tpeak,tf])
    base_vals = np.array([0,Vpeakn,0])
    concat_base_times = base_times[(base_times>t1) & (base_times<t2)]
    concat_base_vals = base_vals[(base_times>t1) & (base_times<t2)]

    ts = np.concatenate([[t1],concat_base_times,[t2]])
    Is = np.concatenate([[V1],concat_base_vals,[V2]])
    
    assert len(ts) == len(Is)
    
    return sum(((np.roll(Is,-1) + Is)*(np.roll(ts,-1) - ts))[:-1]/2)

def plot_beta(params, ax=None, figsize=(12,7),**kwargs):
    t0,tpeak,tf,Vpeak = params
    xs = [t0,tpeak,tf]
    ys = [3,Vpeak,6]
    if ax:
        ax.plot(xs,ys,**kwargs)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xs,ys,**kwargs)
        return fig, ax

def get_beta(t, params, alpha):

    t0,tpeak,tf,Vpeak = params
    Vpeak = min(alpha*max(Vpeak-6,0)+6,Vpeak)
    
    if t <= t0:
        return 0
    elif t0 < t < tpeak:
        return interp(t0,3,tpeak,Vpeak,t)
    else: # tpeak < t 
        return  max(interp(tpeak,Vpeak,tf,3,t),0)

def get_trajectory_data(N=1000, alpha=1, start_day=0, end_day=20):
    traj_arr = []
    beta_arr = []
    for _ in tqdm(range(N)):
        params = gen_params()
        traj_arr.append([get_beta(i,params,alpha) for i in range(start_day,end_day+1)])

    return traj_arr

def gen_inf_data(N=1_000, alpha=1, start_day=0, end_day=20, beta_pts=False):
    '''
    Entry 1: Number of people infected from start_day to start_day + 1
    '''
    inf_arr = []
    beta_arr = []
    for _ in trange(N):
        params = gen_params()
        if beta_pts:
            beta_arr.append(params)
        inf_arr.append([integrate_inf(i,i+1,alpha,params) for i in range(start_day,end_day+1)])
    
    if beta_pts:
        return np.array(beta_arr), np.array(inf_arr)
    
    return np.array(inf_arr)