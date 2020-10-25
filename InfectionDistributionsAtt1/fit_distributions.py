import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm
import warnings
import pickle

def best_fit_distribution(data):
    # Get histogram of original data
    y, x = np.histogram(data, bins='auto', density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    # Removed: st.levy_stable (too slow)
    DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    best_distribution = None
    best_params = None
    best_sse = np.inf

    for distribution in tqdm(DISTRIBUTIONS):
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)
                
                # Assess Fitness
                pdf = distribution.pdf(x,*params)
                sse = np.sum(np.power(y - pdf, 2.0))
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

df = pd.read_csv("Incident_Infections_10000.csv", index_col=0)
df.columns = list(map(lambda x: int(x)+2,df.columns))



day_range = range(3,16)
day_list = []
for day in tqdm(day_range):
    dist_name, params = best_fit_distribution(df[day])
    day_dict = {"day":day,
                "data":df[day].to_numpy(),
                "dist":getattr(st,dist_name),
                "params":params}
    day_list.append(day_dict)

pickle.dump(day_list, open("DistDayData.p","wb"))
