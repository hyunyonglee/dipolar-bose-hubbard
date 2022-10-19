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
h = float(sys.argv[7])
l = float(sys.argv[8])
q = float(sys.argv[9])
Ncut = int(sys.argv[10])
CHI = int(sys.argv[11])
RM = sys.argv[12]
IS = sys.argv[13]

model_params = {
    "L": L,
    "t": t,
    "tp": tp,
    "h": 0.0,
    "U": 1.0,
    "Ut": 0.0,
    "mu": 0.0,
    "l": l,
    "q": q,
    "Ncut": Ncut,
    "bc_MPS": 'finite',
    "bc": 'periodic',
    "QN": 'N'
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.DIPOLAR_BOSE_HUBBARD(model_params)

# initial state
if IS == '1-1half':
    product_state = ['2','2','1','1'] * int(M.lat.N_sites/4)
    
elif IS == '2-1half':
    product_state = ['3','3','2','2'] * int(M.lat.N_sites/4)
    
else:
    product_state = [IS] * M.lat.N_sites
    
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
if RM == 'random':
    TEBD_params = {'N_steps': 4, 'trunc_params':{'chi_max': 4}, 'verbose': 0}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    psi.canonical_form() 


dmrg_params = {
    # 'mixer': True,  # setting this to True helps to escape local minima
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-3,
        'decay': 1.5,
        'disable_after': 50
    },
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-9
    },
    # 'chi_list': chi_list,
    'max_E_err': 1.0e-8,
    'max_S_err': 1.0e-6,
    'max_sweeps': 200,
    'combine' : True
}

ensure_dir("observables/")
ensure_dir("entanglement/")
ensure_dir("logs/")
ensure_dir("mps/")

# ground state
eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
# eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
psi.canonical_form() 

N = psi.expectation_value("N")
B = np.abs( psi.expectation_value("B") )
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()

file1 = open( "observables/energy.txt","a")
file1.write(repr(t) + " " + repr(tp) + " " + repr(q) + " " + repr(l) + " " + repr(E) + "\n")

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
