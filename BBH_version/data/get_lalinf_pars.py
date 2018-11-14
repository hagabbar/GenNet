import h5py
import pandas as pd
from sympy import Eq, Symbol, solve
import dill
import pickle
import numpy as np
from sys import exit

event_name = 'gw150914'
event_time = '1126259462'
posteriors = pd.read_hdf('/home/hunter.gabbard/parameter_estimation/john_bayesian_tutorial/injection_run_mass-time-varry_%s_srate-2048/lalinferencenest/posterior_samples/posterior_H1_%s-0.hdf5' % (event_name,event_time),'lalinference/lalinference_nest/posterior_samples')


# load lalinference chirp mass
post_mc = posteriors['mc']
# load lalinference q
post_q = posteriors['q']

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
    with open('%s_m1_m2_lainf_post_srate-2048.sav' % event_name, 'wb') as f:
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
    with open('%s_mc_M_lainf_post_srate-2048.sav' % event_name, 'wb') as f:
        pickle.dump(lalinf_pars, f)

if do_mc_q:
    lalinf_pars = np.array([post_mc,post_q])
    with open('%s_mc_q_lalinf_post_srate-2048.sav' % event_name, 'wb') as f:
        pickle.dump(lalinf_pars, f)
