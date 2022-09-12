# Copyright 2022 Hyun-Yong Lee

import numpy as np
import model
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
import os
import os.path
import sys
import matplotlib.pyplot as plt
import pickle

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

import logging.config
conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
}
logging.config.dictConfig(conf)

L = int(sys.argv[1])
t = float(sys.argv[2])
tp = float(sys.argv[3])
U = float(sys.argv[4])
Ut = float(sys.argv[5])
mu = float(sys.argv[6])
Ncut = int(sys.argv[7])
CHI = int(sys.argv[8])
RM = sys.argv[9]
QN = sys.argv[10]
PATH = sys.argv[11]
BC_MPS = sys.argv[12]
BC = sys.argv[13]
IS = sys.argv[14]
TOL = float(sys.argv[15])
h = float(sys.argv[16])

model_params = {
    "L": L,
    "t": t,
    "tp": tp,
    "h": h,
    "U": U,
    "Ut": Ut,
    "mu": mu,
    "Ncut": Ncut,
    "bc_MPS": BC_MPS,
    "bc": BC,
    "QN": QN
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.DIPOLAR_BOSE_HUBBARD(model_params)

# initial state
if IS == '0-1half':
    product_state = ['1','1','0','0'] * int(M.lat.N_sites/4)
    
elif IS == '1-1half':
    product_state = ['2','2','1','1'] * int(M.lat.N_sites/4)
    
elif IS == '1-1half+1':
    product_state = ['2','2','1','1'] * int(M.lat.N_sites/4)
    product_state[2] = '2'
    
elif IS == '1-1half-1':
    product_state = ['2','2','1','1'] * int(M.lat.N_sites/4)
    product_state[1] = '1'
    
elif IS == '2-1half':
    product_state = ['3','3','2','2'] * int(M.lat.N_sites/4)
    
elif IS == '3-1half':
    product_state = ['4','4','3','3'] * int(M.lat.N_sites/4)
    
elif any( IS == frac for frac in ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9',
    '1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9', 
    '2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8','2.9',
    '3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','4.5','5.5'] ):
    
    n = float(IS)
    dn = np.remainder(n,1)

    product_state = [str(int(np.floor(n)))] * M.lat.N_sites
    Ls = round(dn*M.lat.N_sites)

    a = 0
    I = 0
    for i in range(M.lat.N_sites):

        s = 2*i
        if s < M.lat.N_sites:
            a = 0    
        else:
            a = M.lat.N_sites-1
    
        product_state[s-a] = str(int(product_state[s-a])+1)
        I = I+1
        if I==Ls:
            break

else:
    product_state = [IS] * M.lat.N_sites
    
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

if RM == 'random':
    TEBD_params = {'N_steps': 4, 'trunc_params':{'chi_max': 4}, 'verbose': 0}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    psi.canonical_form() 


chi_list = {0: 4, 4: 8, 8: 16, 12: 32, 16: 64, 20: CHI}
dmrg_params = {
    # 'mixer': True,  # setting this to True helps to escape local minima
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-3,
        'decay': 2.0,
        'disable_after': 30
    },
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-9
    },
    'chi_list': chi_list,
    'max_E_err': 1.0e-8,
    'max_S_err': TOL,
    'max_sweeps': 500,
    'combine' : True
}

ensure_dir(PATH + "mps/")

# ground state
eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
print(psi.expectation_value("N"))

dmrg_params['orthogonal_to'] = [psi]
psi1 = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
eng1 = dmrg.TwoSiteDMRGEngine(psi1, M, dmrg_params)
E1, psi1 = eng1.run()  # equivalent to dmrg.run() up to the return parameters.
print(psi1.expectation_value("N"))

# measuring exciton condensation
hs = []
if BC == 'periodic':
    for i in range(0,int(L/2-1)): 
        I = 2*i
        hs.append( np.abs( psi.expectation_value_term([('Bd',I+2),('B',I)]) ) )
    hs.append( np.abs( psi.expectation_value_term([('Bd',L-1),('B',L-2 )]) ) )
    for i in range(0,int(L/2-1)):
        I = L-1 - 2*i
        hs.append( np.abs( psi.expectation_value_term([('Bd',I-2),('B',I)]) ) )
    hs.append( np.abs( psi.expectation_value_term([('Bd',0),('B',1)]) ) )

else:
    for i in range(0,L-1): 
        hs.append( np.abs( psi.expectation_value_term([('Bd',i+1),('B',i)]) ) )
#

file = open( PATH + "energy.txt","a")
file.write(repr(L) + " " + repr(t) + " " + repr(E) + " " + repr(E1) + " " + repr( np.mean(hs) ) + " " + "\n")

with open( PATH + 'mps/gs_L%d_t_%.2f.pkl' % (L,t), 'wb') as f:
    pickle.dump(psi, f)

with open( PATH + 'mps/exs_L%d_t_%.2f.pkl' % (L,t), 'wb') as f:
    pickle.dump(psi1, f)



print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
