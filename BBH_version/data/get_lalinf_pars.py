# Copyright (C) 2018  Hunter Gabbard, Chris Messenger
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

'''
This is a script which will take as input posteriors produced by lalinference and convert those to 
chirp mass, inverse mass ratio, and compoenent values. The user gets to decide which of these values
he or she would like to generate. This script assumes that you are rusing Python 2.7
'''

import h5py
import pandas as pd
from sympy import Eq, Symbol, solve
import dill
import pickle
import numpy as np
from sys import exit

event_name = 'gw150914' # name of event to do PE on
event_time = '1126259462' # time stamp of event to do PE on
posteriors = pd.read_hdf('/home/hunter.gabbard/parameter_estimation/john_bayesian_tutorial/injection_run_mass-time-varry_%s_srate-2048/lalinferencenest/posterior_samples/posterior_H1_%s-0.hdf5' % (event_name,event_time),'lalinference/lalinference_nest/posterior_samples') # load all posteriors produced from lalinference


post_mc = posteriors['mc'] # load lalinference chirp mass
post_q = posteriors['q'] # load lalinference q

# choose parameters you would like to convert posterior estimates to
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
