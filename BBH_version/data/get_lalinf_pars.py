import h5py
import pandas as pd
from sympy import Eq, Symbol, solve
import dill
import pickle
import numpy as np
from sys import exit
gw150914_posteriors = pd.read_hdf('/home/hunter.gabbard/parameter_estimation/john_bayesian_tutorial/injection_run_MassNotFixed/lalinferencenest/posterior_samples/posterior_H1_1126259462-0.hdf5','lalinference/lalinference_nest/posterior_samples')


# load lalinference chirp mass
post_mc = gw150914_posteriors['mc']
# load lalinference q
post_q = gw150914_posteriors['q']

do_m1m2 = True
do_mc_M = False
do_mc_q = True

if do_m1m2:
    post_m1 = []
    post_m2 = []
    for idx,par in enumerate(post_mc):
        print('convert posterior estimate number: {0}/{1}'.format(idx,len(post_mc)))
        m1 = Symbol('m1')
        eqn_m1 = Eq((m1 + (m1/post_q[idx])) * (m1*(m1/post_q[idx])/(m1+(m1/post_q[idx]))**2)**(3.0/5.0), post_mc[idx])
        post_m1.append(float(solve(eqn_m1)[0]))

        m2 = Symbol('m2')
        eqn_m2 = Eq((post_q[idx]*m2 + m2) * ((post_q[idx]*m2)*m2/((post_q[idx]*m2)+m2)**2)**(3.0/5.0), post_mc[idx])
        post_m2.append(float(solve(eqn_m2)[0]))

    lalinf_pars = np.array([post_m1,post_m2])
    with open('gw150914_m1_m2_lainf_post.sav', 'wb') as f:
        pickle.dump(lalinf_pars, f)

if do_mc_M:
    post_M = []

    for idx,par in enumerate(post_mc):
        print('convert posterior estimate number: {0}'.format(idx))
        m1 = Symbol('m1')
        eqn_m1 = Eq((m1 + (m1/post_q[idx])) * (m1*(m1/post_q[idx])/(m1+(m1/post_q[idx]))**2)**(3.0/5.0), post_mc[idx])
        post_m1 = float(solve(eqn_m1)[0])

        m2 = Symbol('m2')
        eqn_m2 = Eq((post_q[idx]*m2 + m2) * ((post_q[idx]*m2)*m2/((post_q[idx]*m2)+m2)**2)**(3.0/5.0), post_mc[idx])
        post_m2 = float(solve(eqn_m2)[0])

        post_M.append(post_m1 + post_m2)

    lalinf_pars = np.array([post_mc,post_M]) 
    with open('gw150914_mc_M_lainf_post.sav', 'wb') as f:
        pickle.dump(lalinf_pars, f)

if do_mc_q:
    lalinf_pars = np.array([post_mc,post_q])
    with open('gw150914_mc_q_lalinf_post.sav', 'wb') as f:
        pickle.dump(lalinf_pars, f)
